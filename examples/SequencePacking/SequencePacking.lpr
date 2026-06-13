program SequencePacking;
(*
SequencePacking: padded vs packed feeding for autoregressive LM pretraining.
Copyright (C) 2026 Joao Paulo Schwarz Schuler

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

Coded by Claude (AI).
*)

// Demonstrates TNNetSequencePacker (neuraldatasets.pas): instead of padding
// every short document to the context length, documents are concatenated
// (each followed by one separator token = 1) and cut into full ContextLen
// windows - the GPT-2/GPT-3 pretraining recipe. On a pad-heavy corpus this is
// a large throughput win: almost every target position carries loss, instead
// of mostly pad targets that must be masked out.
//
// The demo trains the SAME tiny causal transformer three times for the SAME
// number of optimizer steps - padded (pmOneDocPerWindow), no-split packed
// (pmNoSplitGreedy) and GPT-style stream-packed (pmSplitAcrossWindows) - and
// reports:
//   1. window utilization (% of target slots that are real, non-pad tokens);
//   2. held-out perplexity (neuralnlpmetrics.Perplexity, teacher-forced).
// Packed feeding sees ~2-3x more real tokens per step here and reaches a
// lower held-out perplexity at the same step count (and same wall-clock: a
// step costs the same regardless of how much of the window is padding). RoPE
// (relative positions) is used so what is learned at packed offsets transfers
// to any position. The step budget is deliberately small: this tiny corpus is
// memorized within a few hundred steps, after which all three runs only
// overfit and the throughput contrast washes out.
//
// Note: attention may cross document boundaries inside a packed window
// (TNNetScaledDotProductAttention has no per-sample dynamic mask). This is
// standard GPT-2/GPT-3 behaviour and works fine in practice.

{$mode objfpc}{$H+}

uses {$IFDEF UNIX} cthreads, {$ENDIF}
  Classes, SysUtils, Math,
  neuralnetwork, neuralvolume, neuraldatasets, neuralnlpmetrics;

const
  csContextLen = 16;
  csDModel = 32;
  csHeads = 2;
  csDFF = 32;
  csSteps = 400;  // identical optimizer step budget for every run
  csLR = 0.01;
  csInertia = 0.9;

var
  Dict: TStringListInt;

// Sorted word-level vocabulary; '<eos>' and '<pad>' sort first (ids 0 and 1),
// real words get ids >= 2 - matching the packer's pad=0 / separator=1
// convention (token ids < 2 are special across the NLP pipeline).
function BuildDict(): TStringListInt;
const
  Words: array[0..14] of string = ('<eos>', '<pad>',
    'ball', 'bird', 'cat', 'chases', 'dog', 'finds', 'fish', 'fox',
    'likes', 'sees', 'star', 'the', 'tree');
var
  W: integer;
begin
  Result := TStringListInt.Create();
  // Byte-exact sort keeps '<eos>'/'<pad>' first even when LazUtils swaps in
  // its UTF-8 collation (which would otherwise ignore the '<' characters).
  Result.CaseSensitive := true;
  Result.UseLocale := false;
  Result.Sorted := true;
  for W := Low(Words) to High(Words) do Result.Add(Words[W]);
  Result.SaveCurrentPosition(); // token id = sorted index
end;

// 64 five-word sentences "the <subject> <verb> the <object>"; the last 8 are
// held out for perplexity evaluation, the first 56 are the training corpus.
procedure BuildCorpus(Train, HeldOut: TStringList);
const
  Subjects: array[0..3] of string = ('cat', 'dog', 'fox', 'bird');
  Verbs: array[0..3] of string = ('sees', 'likes', 'chases', 'finds');
  Objects: array[0..3] of string = ('ball', 'tree', 'fish', 'star');
var
  S, V, O, Idx: integer;
  Line: string;
begin
  Idx := 0;
  for S := 0 to 3 do
    for V := 0 to 3 do
      for O := 0 to 3 do
      begin
        Line := 'the ' + Subjects[S] + ' ' + Verbs[V] + ' the ' + Objects[O];
        if Idx < 56 then Train.Add(Line) else HeldOut.Add(Line);
        Inc(Idx);
      end;
end;

// Tiny decoder-only LM with a per-position next-token head: token ids ->
// embedding -> causal transformer block with RoPE -> per-position softmax.
function BuildModel(VocabSize: integer): TNNet;
begin
  Result := TNNet.Create();
  Result.AddLayer([
    TNNetInput.Create(csContextLen, 1, 1),
    TNNetEmbedding.Create(VocabSize, csDModel)
  ]);
  // TNNetDyT norm on purpose: it is per-element (no cross-token statistics).
  // TNNetLayerNorm normalizes over the WHOLE sample including the sequence
  // axis, so trailing pad rows would shift the normalization of early rows -
  // a train/eval mismatch for the packed model (trained on full windows,
  // evaluated on short padded prefixes).
  Result.AddTransformerEncoderBlock({Heads=}csHeads, {d_ff=}csDFF,
    {PreNorm=}true, {CausalMask=}true, {UseRoPE=}true, {NormClass=}TNNetDyT);
  Result.AddLayer([
    TNNetPointwiseConvLinear.Create(VocabSize),
    TNNetPointwiseSoftMax.Create(1)
  ]);
  Result.SetLearningRate(csLR, csInertia);
  Result.SetL2Decay(0.0);
end;

type
  TDocArray = array of TNeuralIntegerArray;

// Tokenizes the training corpus once.
procedure TokenizeCorpus(Corpus: TStringList; out Docs: TDocArray);
var
  I: integer;
  Toks: TNeuralIntegerArray;
begin
  SetLength(Docs, Corpus.Count);
  for I := 0 to Corpus.Count - 1 do
  begin
    Dict.Tokenize(Corpus[I], Toks);
    Docs[I] := Copy(Toks, 0, Length(Toks));
  end;
  SetLength(Toks, 0);
end;

// Re-packs the documents in a fresh random order. Shuffling documents before
// packing each epoch is the standard GPT pretraining recipe: with a FIXED
// order the packed stream is deterministic, so the model can memorize
// cross-document order instead of learning the language - which destroys
// held-out generalization (we verified: ~500 held-out PPL without shuffling).
procedure ShuffleAndPack(Packer: TNNetSequencePacker; const Docs: TDocArray);
var
  Order: array of integer;
  I, J, Tmp: integer;
begin
  SetLength(Order, Length(Docs));
  for I := 0 to High(Order) do Order[I] := I;
  for I := High(Order) downto 1 do
  begin
    J := Random(I + 1);
    Tmp := Order[I]; Order[I] := Order[J]; Order[J] := Tmp;
  end;
  Packer.Clear();
  for I := 0 to High(Order) do Packer.AddDocument(Docs[Order[I]]);
  Packer.Pack();
  SetLength(Order, 0);
end;

// Trains NN for csSteps optimizer steps; one epoch = one pass over the packed
// windows, re-shuffled and re-packed every epoch (both modes, for fairness).
// Per step: forward, mask pad-target positions (desired := actual there, so
// e = Output - Desired is exactly zero and no loss flows), backward.
procedure Train(NN: TNNet; Packer: TNNetSequencePacker; const Docs: TDocArray;
  VocabSize: integer; out Seconds: double);
var
  StepsDone, W: integer;
  InputV, TargetV: TNNetVolume;
  StartTime: TDateTime;
begin
  InputV := TNNetVolume.Create(csContextLen, 1, 1);
  TargetV := TNNetVolume.Create(csContextLen, 1, VocabSize);
  try
    StartTime := Now();
    StepsDone := 0;
    while StepsDone < csSteps do
    begin
      ShuffleAndPack(Packer, Docs);
      W := 0;
      while (W < Packer.WindowCount()) and (StepsDone < csSteps) do
      begin
        Packer.GetTrainingPair(W, InputV, TargetV);
        NN.Compute(InputV);
        Packer.ApplyLossMask(W, TargetV, NN.GetLastLayer().Output);
        NN.Backpropagate(TargetV); // per-sample SGD update
        Inc(W);
        Inc(StepsDone);
      end;
    end;
    Seconds := (Now() - StartTime) * 86400.0;
  finally
    TargetV.Free;
    InputV.Free;
  end;
end;

procedure RunOne(const Title: string; Mode: TNNetPackingMode;
  Train_, HeldOut: TStringList);
var
  Packer: TNNetSequencePacker;
  NN: TNNet;
  Stats: TNNetPerplexityStats;
  Docs: TDocArray;
  Seconds: double;
begin
  RandSeed := 424242; // identical initialization for every run
  Packer := TNNetSequencePacker.Create(csContextLen, Mode);
  NN := BuildModel(Dict.Count);
  try
    TokenizeCorpus(Train_, Docs);
    ShuffleAndPack(Packer, Docs);
    WriteLn(Title, ':');
    WriteLn(Format('  windows/epoch: %d  utilization (real targets): %.1f%%',
      [Packer.WindowCount(), 100.0 * Packer.Utilization()]));
    Train(NN, Packer, Docs, Dict.Count, Seconds);
    Stats := Perplexity(NN, Dict, HeldOut);
    WriteLn(Format('  %d steps in %.1fs  ->  held-out perplexity: %.3f' +
      '  (mean NLL %.4f nats, %d tokens scored)',
      [csSteps, Seconds, Stats.Perplexity, Stats.MeanNLL,
       Stats.PredictedTokens]));
    WriteLn();
  finally
    NN.Free;
    Packer.Free;
  end;
end;

var
  TrainCorpus, HeldOut: TStringList;
begin
  WriteLn('Sequence packing for LM pretraining: padded vs packed feeding');
  WriteLn('(same model, same data, same ', csSteps, ' optimizer steps)');
  WriteLn();
  Dict := BuildDict();
  TrainCorpus := TStringList.Create();
  HeldOut := TStringList.Create();
  try
    BuildCorpus(TrainCorpus, HeldOut);
    WriteLn('Corpus: ', TrainCorpus.Count, ' training sentences, ',
      HeldOut.Count, ' held out; vocab ', Dict.Count, ' (2 special).');
    WriteLn();
    RunOne('PADDED  (one document per window, pmOneDocPerWindow)',
      pmOneDocPerWindow, TrainCorpus, HeldOut);
    RunOne('PACKED  (no-split greedy bin fill, pmNoSplitGreedy)',
      pmNoSplitGreedy, TrainCorpus, HeldOut);
    RunOne('PACKED  (GPT-style stream packing, pmSplitAcrossWindows)',
      pmSplitAcrossWindows, TrainCorpus, HeldOut);
  finally
    HeldOut.Free;
    TrainCorpus.Free;
    Dict.Free;
  end;
end.
