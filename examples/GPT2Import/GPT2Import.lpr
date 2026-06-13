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
//   -temp X  - sampling temperature (softmax of logits/X); enables
//              stochastic sampling instead of the default greedy argmax.
//   -topk K  - keep only the K most probable tokens, renormalize and draw
//              proportionally (HF semantics: combines with -temp; -topk
//              alone samples at temperature 1.0).
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

// Stable softmax of the logits row at PosIdx divided by Temperature, then a
// weighted draw over the TopK most probable tokens (TopK <= 0 or >= vocab
// keeps the whole distribution). This is the standard HF top-k sampling.
// The library TNNetSamplerWeightedTopK now implements the same weighted-draw
// semantics on a post-softmax row; this routine is kept because it fuses the
// temperatured softmax and the top-K draw over the packed multi-position
// logits buffer in a single pass (the legacy TNNetSamplerTopK draws UNIFORMLY
// among the top K and would ignore the temperature).
function SampleNextToken(Logits: TNNetVolume; PosIdx, VocabSize: integer;
  Temperature: TNeuralFloat; TopK: integer): integer;
var
  Base, TokCnt, InsCnt, Kept: integer;
  MaxLogit, Sum, Roll, Cum: TNeuralFloat;
  Probs: array of TNeuralFloat;
  KeptProb: array of TNeuralFloat;
  KeptTok: array of integer;
begin
  if Temperature < 1e-6 then Temperature := 1e-6; // -> 0 degenerates to greedy
  Base := PosIdx * VocabSize;
  MaxLogit := Logits.FData[Base];
  for TokCnt := 1 to VocabSize - 1 do
    if Logits.FData[Base + TokCnt] > MaxLogit then
      MaxLogit := Logits.FData[Base + TokCnt];
  SetLength(Probs, VocabSize);
  Sum := 0;
  for TokCnt := 0 to VocabSize - 1 do
  begin
    Probs[TokCnt] := Exp((Logits.FData[Base + TokCnt] - MaxLogit) / Temperature);
    Sum := Sum + Probs[TokCnt];
  end;
  if (TopK <= 0) or (TopK >= VocabSize) then
  begin
    // Full ancestral sampling: cumulative roll over the whole vocabulary.
    Roll := Random * Sum;
    Cum := 0;
    Result := VocabSize - 1;
    for TokCnt := 0 to VocabSize - 1 do
    begin
      Cum := Cum + Probs[TokCnt];
      if Roll < Cum then exit(TokCnt);
    end;
    exit;
  end;
  // Single pass keeping the TopK largest probabilities in a descending
  // insertion array (most tokens fail the [TopK-1] cut immediately).
  SetLength(KeptProb, TopK);
  SetLength(KeptTok, TopK);
  Kept := 0;
  for TokCnt := 0 to VocabSize - 1 do
  begin
    if (Kept = TopK) and (Probs[TokCnt] <= KeptProb[TopK - 1]) then continue;
    if Kept < TopK then Inc(Kept);
    InsCnt := Kept - 1;
    while (InsCnt > 0) and (KeptProb[InsCnt - 1] < Probs[TokCnt]) do
    begin
      KeptProb[InsCnt] := KeptProb[InsCnt - 1];
      KeptTok[InsCnt] := KeptTok[InsCnt - 1];
      Dec(InsCnt);
    end;
    KeptProb[InsCnt] := Probs[TokCnt];
    KeptTok[InsCnt] := TokCnt;
  end;
  Sum := 0;
  for TokCnt := 0 to Kept - 1 do Sum := Sum + KeptProb[TokCnt];
  if Sum <= 0 then exit(KeptTok[0]); // degenerate distribution
  Roll := Random * Sum;
  Cum := 0;
  Result := KeptTok[Kept - 1]; // numeric-safety fallback
  for TokCnt := 0 to Kept - 1 do
  begin
    Cum := Cum + KeptProb[TokCnt];
    if Roll < Cum then exit(KeptTok[TokCnt]);
  end;
end;

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
  Temperature: TNeuralFloat;
  TopK: integer;
  UseSampling: boolean;
  FloatFmt: TFormatSettings;
begin
  Tokenizer := nil;
  PromptText := '';
  Temperature := 1.0;
  TopK := 0;
  UseSampling := false;
  FloatFmt := DefaultFormatSettings;
  FloatFmt.DecimalSeparator := '.'; // -temp 0.8 regardless of locale
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
  ParamCnt := 4;
  while ParamCnt <= ParamCount do
  begin
    if (ParamStr(ParamCnt) = '-t') and (ParamCnt < ParamCount) then
    begin
      Inc(ParamCnt);
      PromptText := ParamStr(ParamCnt);
    end
    else if (ParamStr(ParamCnt) = '-temp') and (ParamCnt < ParamCount) then
    begin
      Inc(ParamCnt);
      Temperature := StrToFloatDef(ParamStr(ParamCnt), 1.0, FloatFmt);
      UseSampling := true;
    end
    else if (ParamStr(ParamCnt) = '-topk') and (ParamCnt < ParamCount) then
    begin
      Inc(ParamCnt);
      TopK := StrToIntDef(ParamStr(ParamCnt), 0);
      UseSampling := true;
    end
    else
    begin
      SetLength(Prompt, Length(Prompt) + 1);
      Prompt[High(Prompt)] := StrToIntDef(ParamStr(ParamCnt), 0);
    end;
    Inc(ParamCnt);
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
    if UseSampling then
    begin
      Randomize;
      Write('Sampling: temperature ', Temperature:0:2);
      if TopK > 0 then Write(', top-k ', TopK);
      WriteLn('.');
    end;
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
        if UseSampling then
          BestToken := SampleNextToken(Output, PosIdx, Config.VocabSize,
            Temperature, TopK)
        else
        begin
          BestToken := 0;
          BestLogit := Output.FData[PosIdx * Config.VocabSize];
          for TokCnt := 1 to Config.VocabSize - 1 do
            if Output.FData[PosIdx * Config.VocabSize + TokCnt] > BestLogit then
            begin
              BestLogit := Output.FData[PosIdx * Config.VocabSize + TokCnt];
              BestToken := TokCnt;
            end;
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
