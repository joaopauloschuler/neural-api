program StarCoder2Complete;
(*
Copyright (C) 2026 Joao Paulo Schwarz Schuler

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License along
with this program; if not, write to the Free Software Foundation, Inc.,
51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

Coded by Claude (AI).
*)

// StarCoder2Complete -- a minimal CPU code-completion demo for the Starcoder2
// importer (neural/neuralpretrained.pas, model_type "starcoder2", the
// bigcode/starcoder2-3b/7b/15b code-LLM family). It loads a Starcoder2
// checkpoint and its stock BPE tokenizer.json, encodes a code PROMPT and
// greedily extends it one token at a time, printing the decoded completion.
//
// Starcoder2 is a CODE-specialised decoder: RoPE + GQA +
// (optional) sliding-window attention paired with biased nn.LayerNorm norms
// (NOT RMSNorm), bias=True on every linear (q/k/v AND o_proj), and a plain
// two-matrix gelu_pytorch_tanh FFN (c_fc -> GELU -> c_proj). All of that is
// handled by BuildStarCoder2FromSafeTensors / BuildFromPretrained; this demo
// is just the generation harness on top.
//
// KV-CACHE DECODE. Generation runs through a TNNetStreamingDecoder
// (neural/neuraldecode.pas) instead of re-encoding the whole prefix every
// step. The model is built at INPUT WIDTH 1 (BuildFromPretrained pSeqLen=1):
// every weight shape in a streamable decoder is sequence-length independent,
// RoPE rotates per-position from PositionOffset (no SeqLen-sized table), and
// Starcoder2's per-token TNNetTokenLayerNorm normalizes each token on its own,
// so a width-1 streamed step is BIT-IDENTICAL to the position's row in a full
// forward. The session owns the KV cache; its budget (SeqLen) bounds the
// longest sequence. The prompt is prefilled token-at-a-time and each new
// token costs ONE width-1 forward over the cached past -- O(cache) per token
// instead of the O(prefix) re-encode the previous full-Compute loop paid.
//
// Usage:
//   StarCoder2Complete <model-dir-or-safetensors> <tokenizer.json> \
//                      [SeqLen] [MaxNewTokens] [prompt...]
// <model-dir-or-safetensors> is a directory holding config.json +
// model.safetensors (or pytorch_model.bin), or the weights file directly
// (config.json read from its directory). SeqLen defaults to 256 (keep it
// small on a real 3B checkpoint - the full 16384 context is slow and
// memory-hungry on CPU; pass pInferenceOnly is on). MaxNewTokens defaults to
// 48. The remaining arguments are joined with spaces as the prompt; the
// default prompt is a short Python function header.
//
// Example (after downloading bigcode/starcoder2-3b to ./starcoder2-3b):
//   StarCoder2Complete ./starcoder2-3b ./starcoder2-3b/tokenizer.json 256 64 \
//     'def fibonacci(n):'

{$mode objfpc}{$H+}

uses
  {$IFDEF UNIX}cthreads, cmem,{$ENDIF}
  Classes, SysUtils,
  neuralvolume, neuralnetwork, neuralsafetensors, neuralpretrained,
  neuralhftokenizer, neuraldecode;

var
  NN: TNNet;
  Tokenizer: TNeuralHFTokenizer;
  Session: TNNetStreamingDecoder;
  InV, Output: TNNetVolume;
  PromptIds, Tokens: TNeuralIntegerArray;
  ModelPath, TokenizerPath, Prompt: string;
  SeqLen, MaxNewTokens, VocabSize: integer;
  Len, StepCnt, Cnt, BestId, ArgCnt: integer;
  BestVal: TNeuralFloat;
  Completion: string;
begin
  if ParamCount < 2 then
  begin
    WriteLn(StdErr, 'Usage: StarCoder2Complete <model-dir-or-safetensors> ',
      '<tokenizer.json> [SeqLen] [MaxNewTokens] [prompt...]');
    Halt(1);
  end;
  ModelPath := ParamStr(1);
  TokenizerPath := ParamStr(2);
  SeqLen := 256;
  MaxNewTokens := 48;
  if ParamCount >= 3 then SeqLen := StrToIntDef(ParamStr(3), SeqLen);
  if ParamCount >= 4 then MaxNewTokens := StrToIntDef(ParamStr(4), MaxNewTokens);
  Prompt := '';
  for ArgCnt := 5 to ParamCount do
  begin
    if Prompt <> '' then Prompt := Prompt + ' ';
    Prompt := Prompt + ParamStr(ArgCnt);
  end;
  if Prompt = '' then Prompt := 'def fibonacci(n):' + LineEnding + '    ';

  WriteLn(StdErr, '==> Loading tokenizer: ', TokenizerPath);
  Tokenizer := TNeuralHFTokenizer.Create();
  NN := nil;
  Session := nil;
  InV := nil;
  try
    Tokenizer.LoadFromFile(TokenizerPath);

    WriteLn(StdErr, '==> Loading Starcoder2 checkpoint: ', ModelPath);
    // pInferenceOnly=true frees training volumes during construction (~1/3 the
    // RAM); BuildFromPretrained dispatches model_type "starcoder2". The net is
    // built at INPUT WIDTH 1 (pSeqLen=1): streamed decode feeds one token per
    // forward and the KV cache (sized below to SeqLen) holds the context, so no
    // wide input layer is ever needed.
    NN := BuildFromPretrained(ModelPath, {pSeqLen=}1, {pInferenceOnly=}true);
    VocabSize := NN.GetLastLayer().Output.Size; // (1,1,vocab) flattened

    PromptIds := Tokenizer.Encode(Prompt);
    Len := Length(PromptIds);
    if Len = 0 then
    begin
      WriteLn(StdErr, 'Prompt encoded to zero tokens.');
      Halt(1);
    end;
    if Len >= SeqLen then
    begin
      WriteLn(StdErr, 'Prompt (', Len, ' tokens) does not fit the ',
        SeqLen, '-token context.');
      Halt(1);
    end;

    WriteLn(StdErr, '==> Prompt (', Len, ' tokens):');
    WriteLn(Prompt);
    WriteLn(StdErr, '==> Completion:');

    SetLength(Tokens, Len);
    for Cnt := 0 to Len - 1 do Tokens[Cnt] := PromptIds[Cnt];

    // KV-cache session over the width-1 net; the budget (SeqLen) must cover the
    // longest sequence (prompt + completion) ever produced.
    Session := TNNetStreamingDecoder.Create(NN, SeqLen);
    InV := TNNetVolume.Create(1, 1, 1);
    Session.Reset();
    // Prefill tokens 0..Len-2 (each at its absolute position); the LAST prompt
    // token is fed as the first decode step's input -- its output row predicts
    // the first new token.
    for Cnt := 0 to Len - 2 do
    begin
      InV.FData[0] := Tokens[Cnt];
      Session.StepForward(InV, Cnt);
    end;
    for StepCnt := 1 to MaxNewTokens do
    begin
      if Len >= SeqLen then break;
      // One width-1 forward of the last committed token over the cached past.
      InV.FData[0] := Tokens[Len - 1];
      Session.StepForward(InV, Len - 1);
      Output := Session.Output(); // (1,1,vocab) -- the single logits row
      // Greedy: argmax over the vocabulary.
      BestId := 0;
      BestVal := Output.FData[0];
      for Cnt := 1 to VocabSize - 1 do
        if Output.FData[Cnt] > BestVal then
        begin
          BestVal := Output.FData[Cnt];
          BestId := Cnt;
        end;
      SetLength(Tokens, Len + 1);
      Tokens[Len] := BestId;
      Inc(Len);
    end;

    // Decode only the GENERATED tail (tokens past the prompt).
    SetLength(Tokens, Len);
    Completion := Tokenizer.Decode(Copy(Tokens, Length(PromptIds),
      Len - Length(PromptIds)));
    WriteLn(Completion);
  finally
    InV.Free;
    Session.Free; // frees BEFORE NN: Destroy ends incremental decode on NN's layers
    NN.Free;
    Tokenizer.Free;
  end;
end.
