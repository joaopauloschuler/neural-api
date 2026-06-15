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
  neuralhftokenizer;

var
  NN: TNNet;
  Tokenizer: TNeuralHFTokenizer;
  Input, Output: TNNetVolume;
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
  Input := nil;
  Output := nil;
  try
    Tokenizer.LoadFromFile(TokenizerPath);

    WriteLn(StdErr, '==> Loading Starcoder2 checkpoint: ', ModelPath);
    // pInferenceOnly=true frees training volumes during construction (~1/3 the
    // RAM); BuildFromPretrained dispatches model_type "starcoder2".
    NN := BuildFromPretrained(ModelPath, SeqLen, {pInferenceOnly=}true);
    VocabSize := NN.GetLastLayer().Output.Size; // (SeqLen,1,vocab) flattened

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

    Input := TNNetVolume.Create(SeqLen, 1, 1);
    Output := TNNetVolume.Create();
    for StepCnt := 1 to MaxNewTokens do
    begin
      if Len >= SeqLen then break;
      Input.Fill(0);
      for Cnt := 0 to Len - 1 do Input.FData[Cnt] := Tokens[Cnt];
      NN.Compute(Input);
      NN.GetOutput(Output);
      // Greedy: argmax of the logits row at the last real position (depends
      // only on tokens 0..Len-1 thanks to the causal mask).
      BestId := 0;
      BestVal := Output.FData[(Len - 1) * VocabSize];
      for Cnt := 1 to VocabSize - 1 do
        if Output.FData[(Len - 1) * VocabSize + Cnt] > BestVal then
        begin
          BestVal := Output.FData[(Len - 1) * VocabSize + Cnt];
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
    Output.Free;
    Input.Free;
    NN.Free;
    Tokenizer.Free;
  end;
end.
