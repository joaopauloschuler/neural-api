program GPT2Import;
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

// GPT2Import -- imports a pretrained HuggingFace GPT-2 checkpoint
// (model.safetensors) into a CAI TNNet, prints the inferred configuration
// and the built architecture, and greedily generates a continuation from a
// raw token-id prompt or, when a HuggingFace tokenizer.json sits next to
// the checkpoint, from a text prompt (neuralhftokenizer.pas).
//
// Usage:
//   GPT2Import <model.safetensors> [SeqLen] [NumHeads] [t0 t1 t2 ...]
//   GPT2Import <model.safetensors> [SeqLen] [NumHeads] -t "prompt text"
//
//   SeqLen   - context window to build (default 64; 0 = the full n_ctx,
//              which is SLOW for the real 1024-context GPT-2 on CPU).
//   NumHeads - attention heads (default 0 = infer n_embd/64; the tiny test
//              fixture needs an explicit 2).
//   t0 t1... - prompt token ids (default: 464 = "The" for the real GPT-2
//              vocabulary).
//   -t text  - text prompt; encoded with the tokenizer.json found in the
//              checkpoint's directory, and the continuation is decoded
//              back to text.
//
// Try it with the tiny committed fixture (from the repo root):
//   GPT2Import tests/fixtures/tiny_gpt2.safetensors 16 2 0 1 2
// or with the real 124M-parameter GPT-2 (see README.md for the download
// link; ~500 MB, tokenizer.json next to it enables text prompts):
//   GPT2Import /tmp/model.safetensors 64 0 -t "The"

{$mode objfpc}{$H+}

uses
  {$IFDEF UNIX}cthreads, cmem,{$ENDIF}
  Classes, SysUtils,
  neuralvolume, neuralnetwork,
  neuralsafetensors, neuralpretrained, neuralhftokenizer;

const
  csDefaultSeqLen = 64;
  csNewTokens = 16; // tokens to generate

var
  NN: TNNet;
  Config: TGPT2Config;
  Input, Output: TNNetVolume;
  Prompt: array of integer;
  FileName: string;
  SeqLen, NumHeads, ParamCnt: integer;
  TokenCnt, PromptLen, StepCnt, PosIdx: integer;
  BestToken, TokCnt: integer;
  BestLogit: TNeuralFloat;
  Tokenizer: TNeuralHFTokenizer;
  TokenizerPath, PromptText: string;
begin
  Tokenizer := nil;
  PromptText := '';
  if ParamCount < 1 then
  begin
    WriteLn('Usage: GPT2Import <model.safetensors> [SeqLen] [NumHeads] [t0 t1 ...]');
    WriteLn('See examples/GPT2Import/README.md.');
    Halt(1);
  end;
  FileName := ParamStr(1);
  SeqLen := csDefaultSeqLen;
  if ParamCount >= 2 then SeqLen := StrToIntDef(ParamStr(2), csDefaultSeqLen);
  NumHeads := 0;
  if ParamCount >= 3 then NumHeads := StrToIntDef(ParamStr(3), 0);
  SetLength(Prompt, 0);
  if (ParamCount >= 5) and (ParamStr(4) = '-t') then
    PromptText := ParamStr(5)
  else
    for ParamCnt := 4 to ParamCount do
    begin
      SetLength(Prompt, Length(Prompt) + 1);
      Prompt[High(Prompt)] := StrToIntDef(ParamStr(ParamCnt), 0);
    end;

  // Optional text-prompt mode: encode with the HF tokenizer.json that
  // ships next to the checkpoint (raw-id mode keeps working without it).
  if PromptText <> '' then
  begin
    TokenizerPath := ExtractFilePath(ExpandFileName(FileName)) +
      'tokenizer.json';
    if not FileExists(TokenizerPath) then
    begin
      WriteLn('Text prompts need a HuggingFace tokenizer.json next to the');
      WriteLn('checkpoint; not found: ', TokenizerPath);
      Halt(1);
    end;
    Tokenizer := TNeuralHFTokenizer.Create();
    Tokenizer.LoadFromFile(TokenizerPath);
    Prompt := Tokenizer.Encode(PromptText);
    WriteLn('Loaded ', TokenizerPath, ' (vocab: ',
      Tokenizer.GetVocabSize(), ').');
    if Length(Prompt) = 0 then
    begin
      WriteLn('The prompt text encoded to zero tokens.');
      Halt(1);
    end;
  end;
  if Length(Prompt) = 0 then
  begin
    SetLength(Prompt, 1);
    Prompt[0] := 464; // "The" in the real GPT-2 BPE vocabulary
  end;

  WriteLn('Loading ', FileName, ' ...');
  // This program only generates (never trains), so free the training
  // volumes during construction: ~1/3 the memory, full GPT-2 fits in RAM.
  NN := BuildGPT2FromSafeTensorsEx(FileName, Config, SeqLen, NumHeads,
    {pInferenceOnly=}true);
  try
    WriteLn(GPT2ConfigToString(Config));
    if SeqLen <= 0 then SeqLen := Config.NCtx;
    WriteLn('Context window built: ', SeqLen, ' tokens.');
    WriteLn;
    WriteLn('--- Architecture ---');
    NN.DebugStructure();
    WriteLn;

    // Greedy generation from raw token ids. The net is a fixed-width
    // causal LM: feed the current sequence left-aligned (positions past
    // the end padded with token 0 - the causal mask keeps them from
    // influencing earlier positions) and read the logits row of the last
    // real position.
    for TokenCnt := 0 to High(Prompt) do
      if (Prompt[TokenCnt] < 0) or (Prompt[TokenCnt] >= Config.VocabSize) then
      begin
        WriteLn('Prompt token ', Prompt[TokenCnt], ' is outside the vocabulary (0..',
          Config.VocabSize - 1, ').');
        Halt(1);
      end;
    PromptLen := Length(Prompt);
    if PromptLen >= SeqLen then
    begin
      WriteLn('Prompt is longer than the context window.');
      Halt(1);
    end;
    Write('Prompt token ids:');
    for TokenCnt := 0 to PromptLen - 1 do Write(' ', Prompt[TokenCnt]);
    WriteLn;
    Input := TNNetVolume.Create(SeqLen, 1, 1);
    Output := TNNetVolume.Create;
    try
      Write('Generated continuation:');
      for StepCnt := 1 to csNewTokens do
      begin
        if PromptLen >= SeqLen then break;
        Input.Fill(0);
        for TokenCnt := 0 to PromptLen - 1 do
          Input.FData[TokenCnt] := Prompt[TokenCnt];
        NN.Compute(Input);
        NN.GetOutput(Output);
        PosIdx := PromptLen - 1; // logits row predicting the NEXT token
        BestToken := 0;
        BestLogit := Output.FData[PosIdx * Config.VocabSize];
        for TokCnt := 1 to Config.VocabSize - 1 do
          if Output.FData[PosIdx * Config.VocabSize + TokCnt] > BestLogit then
          begin
            BestLogit := Output.FData[PosIdx * Config.VocabSize + TokCnt];
            BestToken := TokCnt;
          end;
        Write(' ', BestToken);
        SetLength(Prompt, PromptLen + 1);
        Prompt[PromptLen] := BestToken;
        Inc(PromptLen);
      end;
      WriteLn;
      if Tokenizer <> nil then
      begin
        WriteLn('Decoded text (prompt + continuation):');
        WriteLn(Tokenizer.Decode(Prompt, {SkipSpecialTokens=}true));
      end
      else
      begin
        WriteLn('(Token ids only - rerun with -t "your prompt" and a');
        WriteLn(' tokenizer.json next to the checkpoint for text in/out.)');
      end;
    finally
      Output.Free;
      Input.Free;
    end;
  finally
    Tokenizer.Free;
    NN.Free;
  end;
end.
