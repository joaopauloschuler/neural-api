program CharTokenizer;
(*
CharTokenizer: minimal in-memory character tokenizer + trainable
embedding lookup. Builds a unique-char vocabulary from a small
hard-coded corpus, trains a tiny "current char -> next char" model
through a TNNetEmbedding lookup, then prints the 5 nearest characters
in embedding space (by cosine similarity) for a handful of probe
chars.

Pipeline:
  TNNetInput(1, 1, 1)               { single token id }
    -> TNNetEmbedding(Vocab, D)     { learned per-char lookup }
    -> TNNetFullConnectLinear(Vocab){ readout }
    -> TNNetSoftMax                 { next-char distribution }

Training data is generated on the fly by drawing a random position in
the corpus and using (corpus[i], corpus[i+1]) as (input, target).
This gives the embedding row for each char a stable gradient signal,
so characters that appear in similar contexts end up nearby in
embedding space.

After training, we walk every row of the embedding matrix, L2-
normalise it, and for a few probe chars print the top-5 most similar
chars by cosine similarity (excluding the probe itself).

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

{$mode objfpc}{$H+}

uses {$IFDEF UNIX} cthreads, {$ENDIF}
  Classes, SysUtils, Math,
  neuralnetwork,
  neuralvolume;

const
  cCorpus =
    'the quick brown fox jumps over the lazy dog. ' +
    'the quick brown fox jumps over the lazy dog. ' +
    'pack my box with five dozen liquor jugs. ' +
    'how vexingly quick daft zebras jump! ' +
    'sphinx of black quartz, judge my vow. ' +
    'the five boxing wizards jump quickly. ';
  cEmbedDim = 16;
  cSteps    = 600;
  cBatch    = 64;
  cLR       = 0.05;
  cInertia  = 0.9;
  cTopK     = 5;

var
  GCharToId: array[0..255] of integer; // -1 = absent
  GIdToChar: array of char;
  GVocab: integer;
  GTokens: array of integer;            // tokenised corpus

procedure BuildVocab;
var
  I, NextId: integer;
  Ch: char;
begin
  for I := 0 to 255 do GCharToId[I] := -1;
  SetLength(GIdToChar, 0);
  NextId := 0;
  for I := 1 to Length(cCorpus) do
  begin
    Ch := cCorpus[I];
    if GCharToId[Ord(Ch)] < 0 then
    begin
      GCharToId[Ord(Ch)] := NextId;
      SetLength(GIdToChar, NextId + 1);
      GIdToChar[NextId] := Ch;
      Inc(NextId);
    end;
  end;
  GVocab := NextId;
  SetLength(GTokens, Length(cCorpus));
  for I := 1 to Length(cCorpus) do
    GTokens[I - 1] := GCharToId[Ord(cCorpus[I])];
end;

function CharLabel(Ch: char): string;
begin
  if Ch = ' ' then Result := '''_'''
  else Result := '''' + Ch + '''';
end;

procedure BuildModel(out NN: TNNet);
begin
  NN := TNNet.Create();
  // 1-token input: a single char index per example.
  NN.AddLayer(TNNetInput.Create(1, 1, 1));
  // Learned per-char embedding. EncodeZero=1 so token 0 trains too.
  NN.AddLayer(TNNetEmbedding.Create(GVocab, cEmbedDim, 1));
  // Linear readout into vocab logits, then softmax.
  NN.AddLayer(TNNetFullConnectLinear.Create(GVocab));
  NN.AddLayer(TNNetSoftMax.Create());
  NN.SetLearningRate(cLR, cInertia);
end;

function CrossEntropy(Output, Target: TNNetVolume): TNeuralFloat;
var
  I: integer;
  P: TNeuralFloat;
begin
  Result := 0;
  for I := 0 to Output.Size - 1 do
    if Target.FData[I] > 0 then
    begin
      P := Output.FData[I];
      if P < 1e-12 then P := 1e-12;
      Result := Result - Target.FData[I] * Ln(P);
    end;
end;

procedure Train(NN: TNNet);
var
  Step, B, Idx, NumPairs: integer;
  InputV, TargetV: TNNetVolume;
  SumLoss: TNeuralFloat;
  StartTime, Elapsed: double;
begin
  NumPairs := Length(GTokens) - 1;
  InputV  := TNNetVolume.Create(1, 1, 1);
  TargetV := TNNetVolume.Create(1, 1, GVocab);
  try
    StartTime := Now();
    for Step := 1 to cSteps do
    begin
      SumLoss := 0;
      for B := 1 to cBatch do
      begin
        Idx := Random(NumPairs);
        InputV.FData[0] := GTokens[Idx];
        TargetV.Fill(0);
        TargetV.FData[GTokens[Idx + 1]] := 1.0;
        NN.Compute(InputV);
        SumLoss := SumLoss + CrossEntropy(NN.GetLastLayer.Output, TargetV);
        NN.Backpropagate(TargetV);
      end;
      if (Step = 1) or (Step mod 50 = 0) or (Step = cSteps) then
      begin
        Elapsed := (Now() - StartTime) * 86400.0;
        WriteLn(Format('  step %4d / %4d   mean-CE=%.5f   elapsed=%.1fs',
          [Step, cSteps, SumLoss / cBatch, Elapsed]));
      end;
    end;
  finally
    InputV.Free;
    TargetV.Free;
  end;
end;

procedure PrintNearest(EmbLayer: TNNetLayer);
const
  cProbes: array[0..3] of char = ('q', 'e', 't', ' ');
var
  Weights: TNNetVolume;
  Norm: array of TNeuralFloat;
  Unit_: array of array of TNeuralFloat;
  I, J, P, ProbeId, K, BestIdx: integer;
  Sum, Sim, BestSim: TNeuralFloat;
  Taken: array of boolean;
  Ptr: TNeuralFloatArrPtr;
  ProbeCh: char;
begin
  Weights := EmbLayer.Neurons[0].Weights;
  // Materialise rows + L2 norms.
  SetLength(Unit_, GVocab);
  SetLength(Norm, GVocab);
  for I := 0 to GVocab - 1 do
  begin
    SetLength(Unit_[I], cEmbedDim);
    Ptr := Weights.GetRawPtr(I, 0, 0);
    Sum := 0;
    for J := 0 to cEmbedDim - 1 do
    begin
      Unit_[I][J] := Ptr^[J];
      Sum := Sum + Ptr^[J] * Ptr^[J];
    end;
    Norm[I] := Sqrt(Sum);
    if Norm[I] < 1e-9 then Norm[I] := 1e-9;
    for J := 0 to cEmbedDim - 1 do
      Unit_[I][J] := Unit_[I][J] / Norm[I];
  end;

  WriteLn;
  WriteLn('Top-', cTopK, ' nearest characters by cosine similarity:');
  SetLength(Taken, GVocab);
  for P := 0 to High(cProbes) do
  begin
    ProbeCh := cProbes[P];
    ProbeId := GCharToId[Ord(ProbeCh)];
    if ProbeId < 0 then
    begin
      WriteLn('  ', CharLabel(ProbeCh), ' : (not in vocab)');
      Continue;
    end;
    for I := 0 to GVocab - 1 do Taken[I] := False;
    Taken[ProbeId] := True;
    Write('  ', CharLabel(ProbeCh), ' -> ');
    for K := 1 to cTopK do
    begin
      BestIdx := -1;
      BestSim := -1e30;
      for I := 0 to GVocab - 1 do
      begin
        if Taken[I] then Continue;
        Sim := 0;
        for J := 0 to cEmbedDim - 1 do
          Sim := Sim + Unit_[ProbeId][J] * Unit_[I][J];
        if Sim > BestSim then
        begin
          BestSim := Sim;
          BestIdx := I;
        end;
      end;
      if BestIdx < 0 then Break;
      Taken[BestIdx] := True;
      Write(CharLabel(GIdToChar[BestIdx]), '(', BestSim:5:3, ')');
      if K < cTopK then Write(', ');
    end;
    WriteLn;
  end;
end;

var
  NN: TNNet;
begin
  SetExceptionMask([exInvalidOp, exDenormalized, exZeroDivide,
                    exOverflow, exUnderflow, exPrecision]);
  RandSeed := 2026;
  BuildVocab;
  WriteLn('CharTokenizer demo');
  WriteLn('  corpus length : ', Length(cCorpus), ' chars');
  WriteLn('  vocabulary    : ', GVocab, ' unique chars');
  WriteLn('  embedding dim : ', cEmbedDim);

  BuildModel(NN);
  try
    WriteLn;
    WriteLn('Architecture:');
    NN.DebugStructure;
    WriteLn;
    WriteLn('Training next-char prediction for ', cSteps,
      ' steps of batch ', cBatch, '...');
    Train(NN);
    // The embedding layer is layer index 1.
    PrintNearest(NN.Layers[1]);
    WriteLn;
    WriteLn('Note: cosine neighbours come from the (vocab x embed) ',
      'matrix at NN.Layers[1].Neurons[0].Weights.');
  finally
    NN.Free;
  end;
end.
