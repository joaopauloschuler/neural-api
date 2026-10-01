program TCNClassifier;
(*
TCNClassifier: a Temporal Convolutional Network (Bai, Kolter & Koltun 2018,
arXiv:1803.01271) for MULTIVARIATE time-series classification, built with
TNNet.AddTCNBlock on a SYNTHETIC series generated in-code (no data files).

Series: cNumFeatures sensor channels of Gaussian noise. Two kinds of events:
  - a CUE: a 3-step pulse on ONE channel, every cMinCueGap..cMaxCueGap steps;
  - DISTRACTORS: frequent 1-step pulses on random channels.
The label of a rolling window (cWindowLen,1,cNumFeatures) is the channel of the
most recent cue. Cues are sparse, so the most recent one is usually far back in
the window and only a long receptive field can see it.

Model (cDilations = 1, 2, 4, 8, 16; KernelSize 2; receptive field 63 steps):
  Input (cWindowLen,1,cNumFeatures)
    -> 5x NN.AddTCNBlock(cChannels, cKernelSize, Dilation, DropoutRate, UseNormalization)
    -> TNNetCrop(cWindowLen-1, 0, 1, 1)   (last time step: it sees the whole window causally)
    -> TNNetFullConnectLinear(cNumFeatures) -> TNNetSoftMax
A baseline with every dilation set to 1 (same layers, same weight count) is
trained on the same data so the effect of dilation is visible.

Options: --dropout <rate> (default 0, i.e. no dropout layers)
         --norm             (adds TNNetMovingStdNormalization; default off)

Pure CPU, a few minutes.

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
  neuralfit;

const
  cNumFeatures   = 4;     // sensor channels = classes
  cWindowLen     = 64;
  cSeriesLen     = 20000;
  cTrainEnd      = 16000; // chronological split: train | validation | test
  cValidEnd      = 18000;
  cWindowStride  = 4;     // step between consecutive rolling windows
  cMinCueGap     = 30;
  cMaxCueGap     = 60;   // < cWindowLen, so every window holds at least one cue
  cCueLen        = 3;
  cDistractorP   = 0.04;  // per-step probability of a distractor pulse
  cPulse         = 3.0;
  cNoiseStd      = 0.3;
  cChannels      = 16;
  cKernelSize    = 2;
  cDilations: array[0..4] of integer = (1, 2, 4, 8, 16);
  cEpochs        = 20;
  cBatchSize     = 32;

var
  Series: array of array[0..cNumFeatures - 1] of TNeuralFloat;
  LastCue: array of integer; // channel of the most recent cue at each step (-1: none yet)
  DropoutRate: TNeuralFloat = 0;
  UseNormalization: boolean = false;

procedure ParseOptions();
var
  ArgCnt: integer;
begin
  ArgCnt := 1;
  while ArgCnt <= ParamCount do
  begin
    if (ParamStr(ArgCnt) = '--dropout') and (ArgCnt < ParamCount) then
    begin
      Inc(ArgCnt);
      DropoutRate := StrToFloat(ParamStr(ArgCnt));
    end
    else if ParamStr(ArgCnt) = '--norm' then UseNormalization := true
    else WriteLn('Ignoring unknown option ', ParamStr(ArgCnt));
    Inc(ArgCnt);
  end;
end;

procedure BuildSeries();
var
  t, f, NextCue, Cue, CueStep: integer;
begin
  SetLength(Series, cSeriesLen);
  SetLength(LastCue, cSeriesLen);
  NextCue := RandomRange(0, cMinCueGap);
  Cue := -1;
  CueStep := cCueLen;
  for t := 0 to cSeriesLen - 1 do
  begin
    for f := 0 to cNumFeatures - 1 do Series[t][f] := RandG(0, cNoiseStd);
    if Random < cDistractorP then
      Series[t][Random(cNumFeatures)] += cPulse;
    if t = NextCue then
    begin
      Cue := Random(cNumFeatures);
      CueStep := 0;
      NextCue := t + RandomRange(cMinCueGap, cMaxCueGap + 1);
    end;
    if CueStep < cCueLen then
    begin
      Series[t][Cue] += cPulse;
      Inc(CueStep);
    end;
    LastCue[t] := Cue;
  end;
end;

// Rolling windows whose LAST step lies in [FirstEnd, LastEnd).
function CreatePairs(FirstEnd, LastEnd: integer): TNNetVolumePairList;
var
  WindowEnd, k, f: integer;
  Window, Target: TNNetVolume;
begin
  Result := TNNetVolumePairList.Create();
  WindowEnd := Max(FirstEnd, cWindowLen - 1);
  while WindowEnd < LastEnd do
  begin
    Window := TNNetVolume.Create(cWindowLen, 1, cNumFeatures);
    for k := 0 to cWindowLen - 1 do
      for f := 0 to cNumFeatures - 1 do
        Window[k, 0, f] := Series[WindowEnd - cWindowLen + 1 + k][f];
    Target := TNNetVolume.Create(1, 1, cNumFeatures, 0);
    Target.FData[LastCue[WindowEnd]] := 1.0;
    Result.Add(TNNetVolumePair.Create(Window, Target));
    Inc(WindowEnd, cWindowStride);
  end;
end;

function BuildTCN(Dilated: boolean): TNNet;
var
  BlockCnt, Dilation: integer;
begin
  Result := TNNet.Create();
  Result.AddLayer(TNNetInput.Create(cWindowLen, 1, cNumFeatures));
  for BlockCnt := 0 to High(cDilations) do
  begin
    if Dilated then Dilation := cDilations[BlockCnt] else Dilation := 1;
    Result.AddTCNBlock(cChannels, cKernelSize, Dilation, DropoutRate, UseNormalization);
  end;
  Result.AddLayer(TNNetCrop.Create(cWindowLen - 1, 0, 1, 1));
  Result.AddLayer(TNNetFullConnectLinear.Create(cNumFeatures));
  Result.AddLayer(TNNetSoftMax.Create());
end;

// Two causal convs per block: each adds (KernelSize-1)*Dilation steps.
function ReceptiveField(Dilated: boolean): integer;
var
  BlockCnt: integer;
begin
  Result := 1;
  for BlockCnt := 0 to High(cDilations) do
    if Dilated then Inc(Result, 2 * (cKernelSize - 1) * cDilations[BlockCnt])
    else Inc(Result, 2 * (cKernelSize - 1));
end;

function EvaluateAccuracy(NN: TNNet; Pairs: TNNetVolumePairList): TNeuralFloat;
var
  PairCnt, Hits: integer;
begin
  Hits := 0;
  for PairCnt := 0 to Pairs.Count - 1 do
  begin
    NN.Compute(Pairs[PairCnt].I);
    if NN.GetLastLayer.Output.GetClass() = Pairs[PairCnt].O.GetClass() then Inc(Hits);
  end;
  if Pairs.Count = 0 then Result := 0 else Result := Hits / Pairs.Count;
end;

// Fraction of test windows whose most recent cue started inside the last
// RField steps, i.e. windows a model with that receptive field can still see.
function CueVisibleFraction(Pairs: TNNetVolumePairList; RField: integer): TNeuralFloat;
var
  PairCnt, k, Visible: integer;
  Cue: integer;
  Found: boolean;
begin
  Visible := 0;
  for PairCnt := 0 to Pairs.Count - 1 do
  begin
    Cue := Pairs[PairCnt].O.GetClass();
    Found := false;
    // A cue is cCueLen consecutive pulses on its channel; look for its tail.
    for k := cWindowLen - 1 downto Max(cWindowLen - RField, cCueLen - 1) do
      if (Pairs[PairCnt].I[k, 0, Cue] > cPulse / 2) and
         (Pairs[PairCnt].I[k - 1, 0, Cue] > cPulse / 2) and
         (Pairs[PairCnt].I[k - 2, 0, Cue] > cPulse / 2) then
      begin
        Found := true;
        break;
      end;
    if Found then Inc(Visible);
  end;
  if Pairs.Count = 0 then Result := 0 else Result := Visible / Pairs.Count;
end;

function TrainAndTest(const Name: string; Dilated: boolean;
  TrainPairs, ValPairs, TestPairs: TNNetVolumePairList): TNeuralFloat;
var
  NN: TNNet;
  NFit: TNeuralFit;
  StartTime: TDateTime;
begin
  RandSeed := 20261001; // same initial-weight stream for both models
  NN := BuildTCN(Dilated);
  WriteLn(Name, ': receptive field = ', ReceptiveField(Dilated), ' steps, ',
    NN.CountWeights(), ' weights, ', NN.CountLayers(), ' layers');
  NFit := TNeuralFit.Create();
  NFit.InitialLearningRate := 0.003;
  NFit.LearningRateDecay := 0.01;
  NFit.StaircaseEpochs := 5;
  NFit.Inertia := 0.9;
  NFit.L2Decay := 0;
  NFit.Verbose := false;
  NFit.InferHitFn := @ClassCompare;
  StartTime := Now();
  NFit.Fit(NN, TrainPairs, ValPairs, TestPairs, cBatchSize, cEpochs);
  // Fit reloads the best (validation) net into NN.
  Result := EvaluateAccuracy(NN, TestPairs);
  WriteLn(Name, ': test accuracy ', (Result * 100):6:2, '%  (trained in ',
    ((Now() - StartTime) * 86400):0:1, ' s)');
  WriteLn;
  NFit.Free;
  NN.Free;
end;

var
  TrainPairs, ValPairs, TestPairs: TNNetVolumePairList;
  DilatedAcc, BaselineAcc: TNeuralFloat;
begin
  ParseOptions();
  RandSeed := 20260930;
  BuildSeries();
  TrainPairs := CreatePairs(0, cTrainEnd);
  ValPairs   := CreatePairs(cTrainEnd + cWindowLen, cValidEnd);
  TestPairs  := CreatePairs(cValidEnd + cWindowLen, cSeriesLen);

  WriteLn('=== TCNClassifier: multivariate TCN with NN.AddTCNBlock ===');
  WriteLn('features=', cNumFeatures, '  window=', cWindowLen, '  series=', cSeriesLen,
    '  windows train/val/test=', TrainPairs.Count, '/', ValPairs.Count, '/', TestPairs.Count);
  WriteLn('label = channel of the most recent cue (', cMinCueGap, '..', cMaxCueGap,
    ' steps apart); chance = ', (100 / cNumFeatures):0:1, '%');
  WriteLn('DropoutRate=', DropoutRate:0:2, '  UseNormalization=', UseNormalization);
  WriteLn('test windows whose last cue is inside the receptive field: dilated ',
    (CueVisibleFraction(TestPairs, ReceptiveField(true)) * 100):0:1, '%, baseline ',
    (CueVisibleFraction(TestPairs, ReceptiveField(false)) * 100):0:1, '%');
  WriteLn;

  DilatedAcc  := TrainAndTest('TCN (dilations 1,2,4,8,16)', true, TrainPairs, ValPairs, TestPairs);
  BaselineAcc := TrainAndTest('Baseline (all dilations 1)', false, TrainPairs, ValPairs, TestPairs);

  WriteLn('Summary: dilated TCN ', (DilatedAcc * 100):0:2, '%, non-dilated baseline ',
    (BaselineAcc * 100):0:2, '%, chance ', (100 / cNumFeatures):0:1, '%');

  TestPairs.Free;
  ValPairs.Free;
  TrainPairs.Free;
end.
