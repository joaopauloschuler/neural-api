program HellaSwagEval;
(*
HellaSwagEval: an end-to-end multiple-choice scoring demo for the
EvaluateMultipleChoice API in neuralnlpmetrics.pas (the lm-evaluation-harness
HellaSwag / ARC / PIQA scoring pattern: for each item, score every candidate
completion of a shared context and let the argmax win, reporting acc
(sum-logprob argmax) and acc_norm (mean / length-normalized logprob argmax)).

To stay within a tiny CPU / memory budget (no multi-GB checkpoint download)
this demo trains a small CHAR-LEVEL autoregressive next-token model on a
handful of toy "sentences", then scores a handful of hand-written multiple-
choice items through the SAME EvaluateMultipleChoice path that an imported
SmolLM2 / pythia checkpoint would use. The scoring code is checkpoint-
agnostic: swap BuildAndTrainToyModel for a BuildLlamaFromSafeTensors (or any
importer) and feed TNeuralHFTokenizer-produced token ids into the same
TNNetMultipleChoiceItem records -- the harness below is unchanged.

Each item is built from text via a fixed char vocabulary (a stand-in for a
real subword tokenizer); the gold completion is the one the toy language
makes most likely. The program prints the per-item winning candidate plus the
aggregate acc / acc_norm.

Pure CPU, well under a minute, well under the 3 GB ulimit.

Copyright (C) 2026 Joao Paulo Schwarz Schuler

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
any later version.

Coded by Claude (AI).
*)

{$mode objfpc}{$H+}

uses {$IFDEF UNIX} cthreads, {$ENDIF}
  Classes, SysUtils, Math,
  neuralnetwork,
  neuralvolume,
  neuralnlpmetrics;

const
  // Char vocabulary: ids 0..26 = ' ' + 'a'..'z'. id 0 (space) doubles as the
  // pad / boundary symbol. The toy "language" only uses these characters.
  cVocab        = 27;
  cContextLen   = 24;       // model receptive field (chars)
  cEmbedDim     = 16;
  cHidden       = 32;
  cTrainPasses  = 400;
  cLearningRate = 0.05;

  // The toy corpus: short phrases whose final word UNIQUELY determines the
  // continuation (the discriminating word sits right before the completion, so
  // it stays inside the receptive field). A briefly-trained char model then
  // produces a clear next-token ranking that EvaluateMultipleChoice can score.
  cCorpus: array[0..7] of string = (
    'lion roar lion roar lion roar',
    'bird fly bird fly bird fly',
    'fish swim fish swim fish swim',
    'cup coffee cup coffee cup coffee',
    'lion roar lion roar lion roar',
    'bird fly bird fly bird fly',
    'fish swim fish swim fish swim',
    'cup coffee cup coffee cup coffee'
  );

// --- char <-> token id ----------------------------------------------------

function CharToId(C: char): integer;
begin
  if C = ' ' then Result := 0
  else if (C >= 'a') and (C <= 'z') then Result := Ord(C) - Ord('a') + 1
  else Result := 0; // map anything else to the space/boundary token
end;

function Encode(const S: string): TNeuralIntegerArray;
var
  I: integer;
begin
  SetLength(Result, Length(S));
  for I := 1 to Length(S) do
    Result[I - 1] := CharToId(S[I]);
end;

// --- model ----------------------------------------------------------------

procedure BuildModel(out NN: TNNet);
begin
  NN := TNNet.Create();
  NN.AddLayer(TNNetInput.Create(cContextLen, 1, 1));
  NN.AddLayer(TNNetEmbedding.Create(cVocab, cEmbedDim));
  NN.AddLayer(TNNetPointwiseConvReLU.Create(cHidden));
  // Collapse the sequence to a single next-token distribution over the vocab
  // (single next-token head -> ScoreSequence uses the right-aligned-prefix
  // path, exactly like an autoregressive LM scored teacher-forced).
  NN.AddLayer(TNNetFullConnectReLU.Create(cHidden));
  NN.AddLayer(TNNetFullConnectLinear.Create(cVocab));
  NN.AddLayer(TNNetSoftMax.Create());
  NN.SetLearningRate(cLearningRate, 0.9);
end;

procedure TrainModel(NN: TNNet);
// Online next-char SGD over every corpus phrase: for each position predict the
// next char from the prefix Toks[0..T-1]. The input encoding MUST match the
// one neuralnlpmetrics.ScoreSequence uses for a single-next-token head -- it
// zero-fills the window and lays the prefix in REVERSED order (most-recent
// char at index 0) via CopyReversedNoChecksIntArr. We replicate that exactly
// so train and eval see identically-encoded inputs.
var
  Input, Target: TNNetVolume;
  Pass, Line, T: integer;
  Toks, Prefix: TNeuralIntegerArray;
begin
  Input := TNNetVolume.Create(cContextLen, 1, 1);
  Target := TNNetVolume.Create(NN.GetLastLayer().Output);
  try
    for Pass := 1 to cTrainPasses do
      for Line := 0 to High(cCorpus) do
      begin
        Toks := Encode(cCorpus[Line]);
        for T := 1 to High(Toks) do
        begin
          // Prefix Toks[0..T-1] (clipped to the window), reversed into Input.
          Prefix := Copy(Toks, Max(0, T - cContextLen),
            Min(T, cContextLen));
          Input.Fill(0);
          Input.CopyReversedNoChecksIntArr(Prefix);
          Target.Fill(0);
          Target.FData[Toks[T]] := 1.0;
          NN.Compute(Input);
          NN.Backpropagate(Target);
        end;
      end;
  finally
    Target.Free;
    Input.Free;
  end;
end;

// --- multiple-choice items -------------------------------------------------

function MakeItem(const Context: string; const Candidates: array of string;
  Gold: integer): TNNetMultipleChoiceItem;
var
  I: integer;
begin
  Result.ContextTokens := Encode(Context);
  SetLength(Result.Candidates, Length(Candidates));
  for I := 0 to High(Candidates) do
    Result.Candidates[I] := Encode(Candidates[I]);
  Result.GoldIndex := Gold;
end;

var
  NN: TNNet;
  Items: array[0..3] of TNNetMultipleChoiceItem;
  Stats: TNNetMultipleChoiceStats;
  I, C: integer;
  Score: TNNetCompletionScore;
  BestSum, BestNorm: integer;
  BestSumLP, BestNormLP: TNeuralFloat;
  CandNames: array[0..3] of array of string;
  Contexts: array[0..3] of string;
begin
  WriteLn('HellaSwagEval: end-to-end multiple-choice scoring through ',
    'EvaluateMultipleChoice');
  WriteLn('(toy char-level model; the scoring path is identical for an ',
    'imported checkpoint).');
  WriteLn;

  RandSeed := 1234;
  BuildModel(NN);
  try
    WriteLn('Architecture:');
    NN.PrintSummary();
    WriteLn;
    WriteLn('Training char-level model on ', Length(cCorpus),
      ' toy phrases (', cTrainPasses, ' passes) ...');
    TrainModel(NN);
    WriteLn('Done.');
    WriteLn;

    // Four HellaSwag-style items: a context + several continuations, with the
    // gold continuation being the one the toy corpus makes likely.
    Contexts[0] := 'lion roar lion ';
    Items[0] := MakeItem(Contexts[0], ['roar', 'swim', 'fly'], 0);
    CandNames[0] := ['roar', 'swim', 'fly'];

    Contexts[1] := 'bird fly bird ';
    Items[1] := MakeItem(Contexts[1], ['roar', 'swim', 'fly'], 2);
    CandNames[1] := ['roar', 'swim', 'fly'];

    Contexts[2] := 'fish swim fish ';
    Items[2] := MakeItem(Contexts[2], ['fly', 'swim', 'roar'], 1);
    CandNames[2] := ['fly', 'swim', 'roar'];

    Contexts[3] := 'cup coffee cup ';
    Items[3] := MakeItem(Contexts[3], ['coffee', 'roar', 'swim'], 0);
    CandNames[3] := ['coffee', 'roar', 'swim'];

    // Per-item detail: re-score with the public ScoreCompletion so we can
    // print which candidate each metric picks (mirrors EvaluateMultipleChoice).
    WriteLn('Per-item scoring:');
    for I := 0 to High(Items) do
    begin
      BestSum := 0; BestNorm := 0;
      BestSumLP := -1e30; BestNormLP := -1e30;
      for C := 0 to High(Items[I].Candidates) do
      begin
        Score := ScoreCompletion(NN, Items[I].ContextTokens,
          Items[I].Candidates[C]);
        if Score.SumLogProb > BestSumLP then
        begin BestSumLP := Score.SumLogProb; BestSum := C; end;
        if Score.MeanLogProb > BestNormLP then
        begin BestNormLP := Score.MeanLogProb; BestNorm := C; end;
      end;
      WriteLn(Format('  item %d  "%s..."  gold="%s"  acc-pick="%s"  acc_norm-pick="%s"',
        [I, Contexts[I], CandNames[I][Items[I].GoldIndex],
         CandNames[I][BestSum], CandNames[I][BestNorm]]));
    end;
    WriteLn;

    Stats := EvaluateMultipleChoice(NN, Items);
    WriteLn('Aggregate over ', Stats.ItemCount, ' items:');
    WriteLn(Format('  acc      = %.4f  (%d / %d correct by sum-logprob)',
      [Stats.Accuracy, Stats.CorrectCount, Stats.ItemCount]));
    WriteLn(Format('  acc_norm = %.4f  (%d / %d correct by mean-logprob)',
      [Stats.AccuracyNorm, Stats.CorrectNormCount, Stats.ItemCount]));
  finally
    NN.Free;
  end;
end.
