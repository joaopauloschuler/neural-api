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
// raw token-id prompt (the repo's tokenizer cannot read HF vocab.json /
// merges.txt yet, so prompts and outputs are token ids).
//
// Usage:
//   GPT2Import <model.safetensors> [SeqLen] [NumHeads] [t0 t1 t2 ...]
//
//   SeqLen   - context window to build (default 64; 0 = the full n_ctx,
//              which is SLOW for the real 1024-context GPT-2 on CPU).
//   NumHeads - attention heads (default 0 = infer n_embd/64; the tiny test
//              fixture needs an explicit 2).
//   t0 t1... - prompt token ids (default: 464 = "The" for the real GPT-2
//              vocabulary).
//
// Try it with the tiny committed fixture (from the repo root):
//   GPT2Import tests/fixtures/tiny_gpt2.safetensors 16 2 0 1 2
// or with the real 124M-parameter GPT-2 (see README.md for the download
// link; ~500 MB):
//   GPT2Import /tmp/model.safetensors 64 0 464

{$mode objfpc}{$H+}

uses
  {$IFDEF UNIX}cthreads, cmem,{$ENDIF}
  Classes, SysUtils,
  neuralvolume, neuralnetwork,
  neuralsafetensors, neuralpretrained;

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
begin
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
  for ParamCnt := 4 to ParamCount do
  begin
    SetLength(Prompt, Length(Prompt) + 1);
    Prompt[High(Prompt)] := StrToIntDef(ParamStr(ParamCnt), 0);
  end;
  if Length(Prompt) = 0 then
  begin
    SetLength(Prompt, 1);
    Prompt[0] := 464; // "The" in the real GPT-2 BPE vocabulary
  end;

  WriteLn('Loading ', FileName, ' ...');
  NN := BuildGPT2FromSafeTensorsEx(FileName, Config, SeqLen, NumHeads);
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
      WriteLn('(Token ids only - decode them with any GPT-2 BPE tokenizer;');
      WriteLn(' teaching TNeuralTokenizer to read HF vocab.json/merges.txt');
      WriteLn(' is a noted follow-up.)');
    finally
      Output.Free;
      Input.Free;
    end;
  finally
    NN.Free;
  end;
end.
