program WeightStandardizationConv;
(*
WeightStandardizationConv: a Weight-Standardization + GroupNorm vs BatchNorm
bake-off on a tiny synthetic image-classification task, exercising the new
TNNetWeightStandardizationConv layer (the convolution sibling of the dense
TNNetWeightStandardization, both in neuralnetwork.pas).

The phenomenon (the whole point of this example):
  Weight Standardization (Qiao et al. 2019, "Micro-Batch Training with
  Batch-Channel Normalization and Weight Standardization", arXiv:1903.10520)
  standardizes each conv FILTER's weights to zero-mean / unit-std BEFORE the
  convolution, instead of normalizing the activations across the batch the way
  BatchNorm does. WS smooths the loss landscape independently of the batch
  statistics, so it does not degrade when the batch is tiny -- the regime where
  BatchNorm's noisy per-batch mean/variance estimates hurt. The headline WS
  recipe pairs WS conv filters with GroupNorm (a batch-size-independent
  activation norm): WS+GroupNorm is meant to MATCH or BEAT BatchNorm at a
  matched architecture, especially at small batch sizes.

What this bake-off does:
  We build ONE small CNN backbone and swap only the normalization style:
    arm 0  "bn"     : plain TNNetConvolutionLinear filters + AddMovingNorm
                      (the library's moving-mean/variance BatchNorm) + ReLU.
    arm 1  "ws_gn"  : TNNetWeightStandardizationConv filters + TNNetGroupNorm
                      + ReLU (the Qiao et al. WS+GroupNorm recipe).
  Both arms have the SAME conv widths, depth, FC head, seed, data, epochs and
  learning rate, and are trained at a deliberately SMALL batch size (the regime
  where WS+GroupNorm is supposed to help). For each arm we report final TRAIN
  and held-out TEST loss / accuracy.

Honesty caveat (in the spirit of the DropBlockBakeoff / OptimizerBakeoff READMEs):
  This is a SMALL, EASY synthetic task. WS+GroupNorm "matching or beating"
  BatchNorm is a real small-batch phenomenon, but on an easy toy the margin is
  small and either arm can win on a given seed. So the self-check below does NOT
  assert the brittle claim "WS+GroupNorm always beats BatchNorm". It asserts
  invariants that are actually TRUE:
    (1) both arms TRAIN (final train loss < initial loss, no NaN/Inf),
    (2) both arms reach a healthy classifier (well above chance),
    (3) the WS+GroupNorm arm is competitive: its TEST accuracy is within a
        small tolerance of (or better than) the BatchNorm arm.
  The head-to-head TEST accuracy is printed and discussed honestly; whether
  WS+GroupNorm beat BatchNorm on THIS toy/seed is reported, not assumed.

Pure CPU, single-threaded (NFit.MaxThreadNum := 1), deterministic seeding
(RandSeed := 424242), no external data, finishes well under the 5-minute budget.

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
  neuralvolume,
  neuralfit;

const
  cSeed       = 424242;
  cImgSize    = 12;     // 12x12 synthetic images
  cChannels   = 3;      // RGB-ish input channels
  cClasses    = 4;      // 4-way classification
  cConvFeat   = 16;     // conv feature width (matched across both arms)
  cGroups     = 4;      // GroupNorm groups (16 channels / 4 = 4 per group)
  cTrain      = 160;    // small training set
  cTest       = 800;    // larger held-out test set
  cEpochs     = 30;
  cBatch      = 4;      // SMALL batch: the regime where WS+GroupNorm should help
  cLR         = 0.005;
  cL2         = 0.0005; // mild weight decay (matched across arms)

  cNumArms    = 2;
  cAccTol     = 0.06;   // "competitive" tolerance on test accuracy

var
  cArmName: array[0..cNumArms - 1] of string = ('bn', 'ws_gn');

  // Fixed teacher (shared across arms). Each class owns a random spatial
  // template stamped into a noisy image; the label is the source class. Spatially
  // structured so a conv net has something real to learn.
  TeacherTpl: array[0..cClasses - 1, 0..cChannels - 1,
                     0..cImgSize - 1, 0..cImgSize - 1] of TNeuralFloat;

type
  TArmResult = record
    Name           : string;
    InitTrainLoss  : TNeuralFloat;
    FinalTrainLoss : TNeuralFloat;
    TrainLoss      : TNeuralFloat;
    TrainAcc       : TNeuralFloat;
    TestLoss       : TNeuralFloat;
    TestAcc        : TNeuralFloat;
    Diverged       : boolean;
  end;

  TLossTracker = class
  public
    LastError: TNeuralFloat;
    procedure Reset;
    procedure OnAfterEpoch(Sender: TObject);
  end;

var
  GTracker: TLossTracker;

procedure TLossTracker.Reset;
begin
  LastError := NaN;
end;

procedure TLossTracker.OnAfterEpoch(Sender: TObject);
begin
  LastError := (Sender as TNeuralFit).CurrentTrainingError;
end;

// ---------------------------------------------------------------------------
// Teacher + data generation. Reseeding before each build keeps the teacher and
// the data identical across both arms (only the normalization style differs).
// ---------------------------------------------------------------------------
procedure MakeTeacher;
var
  C, Ch, X, Y: integer;
begin
  for C := 0 to cClasses - 1 do
    for Ch := 0 to cChannels - 1 do
      for Y := 0 to cImgSize - 1 do
        for X := 0 to cImgSize - 1 do
          TeacherTpl[C, Ch, X, Y] := RandG(0, 1);
end;

function MakeImage(out Cls: integer): TNNetVolume;
var
  Ch, X, Y: integer;
  Amp: TNeuralFloat;
begin
  Cls := Random(cClasses);
  // Low signal-to-noise (small template amplitude over unit Gaussian noise) so
  // the task is genuinely hard: the net cannot trivially memorise the train set
  // to 100% accuracy, and a batch-size-independent normalization (WS+GroupNorm)
  // has room to generalise better at the small batch size.
  Amp := 0.12 + Random * 0.16;
  Result := TNNetVolume.Create(cImgSize, cImgSize, cChannels);
  for Ch := 0 to cChannels - 1 do
    for Y := 0 to cImgSize - 1 do
      for X := 0 to cImgSize - 1 do
        Result.Data[X, Y, Ch] :=
          RandG(0, 1) + Amp * TeacherTpl[Cls, Ch, X, Y];
end;

function BuildSet(Count: integer): TNNetVolumePairList;
var
  I, Cls: integer;
  X, Y: TNNetVolume;
begin
  Result := TNNetVolumePairList.Create();
  for I := 0 to Count - 1 do
  begin
    X := MakeImage(Cls);
    Y := TNNetVolume.Create(cClasses, 1, 1);
    Y.Fill(0);
    Y.FData[Cls] := 1.0;   // one-hot target for SoftMax + cross-entropy
    Result.Add(TNNetVolumePair.Create(X, Y));
  end;
end;

// ---------------------------------------------------------------------------
// The CNN backbone. Identical width / depth / FC head in both arms; only the
// conv + normalization style differs:
//   bn    -> ConvolutionLinear filters       + AddMovingNorm (BatchNorm)  + ReLU
//   ws_gn -> WeightStandardizationConv filters + GroupNorm              + ReLU
// ---------------------------------------------------------------------------
procedure AddNormBlock(NN: TNNet; Arm, Feat, FeatureSize, Padding: integer);
begin
  if Arm = 0 then
  begin
    // BatchNorm arm: plain linear conv, then moving-mean/variance norm + affine.
    NN.AddLayer( TNNetConvolutionLinear.Create(Feat, FeatureSize, Padding, 1) );
    NN.AddMovingNorm(false);
    NN.AddLayer( TNNetReLU.Create() );
  end
  else
  begin
    // WS+GroupNorm arm: weight-standardized conv filters, then GroupNorm + ReLU.
    NN.AddLayer( TNNetWeightStandardizationConv.Create(Feat, FeatureSize, Padding, 1) );
    NN.AddLayer( TNNetGroupNorm.Create(cGroups) );
    NN.AddLayer( TNNetReLU.Create() );
  end;
end;

procedure BuildNet(NN: TNNet; Arm: integer);
begin
  NN.AddLayer( TNNetInput.Create(cImgSize, cImgSize, cChannels) );
  AddNormBlock(NN, Arm, cConvFeat, 3, 1);   // 12x12 map
  AddNormBlock(NN, Arm, cConvFeat, 3, 1);   // 12x12 map
  NN.AddLayer( TNNetMaxPool.Create(2) );    // 6x6 map
  AddNormBlock(NN, Arm, cConvFeat, 3, 1);   // 6x6 map
  NN.AddLayer( TNNetMaxPool.Create(2) );    // 3x3 map
  NN.AddLayer( TNNetFullConnectLinear.Create(cClasses) );
  NN.AddLayer( TNNetSoftMax.Create() );
end;

// Cross-entropy loss + 0/1 accuracy over a pair list (dropouts disabled).
procedure Evaluate(NN: TNNet; Pairs: TNNetVolumePairList;
  out Loss: TNeuralFloat; out Acc: TNeuralFloat);
var
  I, Pred, Tgt, Correct: integer;
  P: TNeuralFloat;
  SumCE: Double;
begin
  NN.EnableDropouts(false);
  SumCE := 0; Correct := 0;
  for I := 0 to Pairs.Count - 1 do
  begin
    NN.Compute(Pairs[I].I);
    Pred := NN.GetLastLayer().Output.GetClass();
    Tgt  := Pairs[I].O.GetClass();
    if Pred = Tgt then Inc(Correct);
    P := NN.GetLastLayer().Output.FData[Tgt];
    if P < 1e-12 then P := 1e-12;
    SumCE := SumCE - Ln(P);
  end;
  if Pairs.Count > 0 then
  begin
    Loss := SumCE / Pairs.Count;
    Acc  := Correct / Pairs.Count;
  end
  else
  begin
    Loss := 0; Acc := 0;
  end;
end;

function RunArm(Arm: integer; Train, Test: TNNetVolumePairList): TArmResult;
var
  NN: TNNet;
  NFit: TNeuralFit;
  Dummy: TNeuralFloat;
begin
  Result.Name := cArmName[Arm];
  GTracker.Reset;

  // Reseed before BUILD so weight init is identical across arms.
  RandSeed := cSeed;
  NN := TNNet.Create();
  NFit := TNeuralFit.Create();
  try
    BuildNet(NN, Arm);

    Evaluate(NN, Train, Result.InitTrainLoss, Dummy);

    NFit.FileNameBase := GetTempDir + 'WeightStandardizationConv_autosave';
    NFit.InitialLearningRate := cLR;
    NFit.LearningRateDecay := 0;
    NFit.L2Decay := cL2;
    NFit.Verbose := false;
    NFit.HideMessages();
    NFit.HasFlipX := false;
    NFit.HasFlipY := false;
    NFit.MaxThreadNum := 1;  // single-threaded => deterministic reductions
    NFit.OnAfterEpoch := @GTracker.OnAfterEpoch;
    NFit.EnableClassComparison();
    NFit.EnableDefaultLoss();
    NFit.Fit(NN, Train, Test, nil, cBatch, cEpochs);

    Result.FinalTrainLoss := GTracker.LastError;
    Evaluate(NN, Train, Result.TrainLoss, Result.TrainAcc);
    Evaluate(NN, Test, Result.TestLoss, Result.TestAcc);

    Result.Diverged :=
      IsNan(Result.FinalTrainLoss) or IsInfinite(Result.FinalTrainLoss) or
      IsNan(Result.TestLoss) or IsInfinite(Result.TestLoss);
  finally
    NFit.Free;
    NN.Free;
  end;
end;

function SafeF(V: TNeuralFloat; Width, Decimals: integer): string;
begin
  if IsNan(V) then Result := 'NaN'
  else if IsInfinite(V) then Result := 'Inf'
  else Result := FloatToStrF(V, ffFixed, Width, Decimals);
end;

var
  Results: array[0..cNumArms - 1] of TArmResult;
  TrainSet, TestSet: TNNetVolumePairList;
  k: integer;
  StartTime, EndTime: TDateTime;
  PassTrain, PassFinite, PassBaseline, PassCompetitive, AllPass: boolean;
  WSDelta: TNeuralFloat;
begin
  SetExceptionMask([exInvalidOp, exDenormalized, exZeroDivide,
                    exOverflow, exUnderflow, exPrecision]);

  WriteLn('================================================================');
  WriteLn('Weight-Standardization + GroupNorm vs BatchNorm bake-off');
  WriteLn('on a tiny synthetic image-classification task.');
  WriteLn('================================================================');
  WriteLn(Format('Backbone: Input(%dx%dx%d) -> [Norm-Conv x2] -> MaxPool',
    [cImgSize, cImgSize, cChannels]));
  WriteLn(Format('          -> [Norm-Conv] -> MaxPool -> FC(%d) -> SoftMax.',
    [cClasses]));
  WriteLn(Format('  bn    : ConvLinear(%d,3) + MovingNorm(BatchNorm) + ReLU.',
    [cConvFeat]));
  WriteLn(Format('  ws_gn : WeightStandardizationConv(%d,3) + GroupNorm(%d) + ReLU.',
    [cConvFeat, cGroups]));
  WriteLn(Format('Train=%d, Test=%d, epochs=%d, SMALL batch=%d, LR=%.3f, RandSeed=%d.',
    [cTrain, cTest, cEpochs, cBatch, cLR, cSeed]));
  WriteLn('Same backbone/seed/data/epochs/LR; only the normalization differs.');
  WriteLn;

  GTracker := TLossTracker.Create;
  StartTime := Now;
  RandSeed := cSeed;
  MakeTeacher;
  TrainSet := BuildSet(cTrain);
  TestSet  := BuildSet(cTest);
  try
    for k := 0 to cNumArms - 1 do
    begin
      Write(Format('Training arm "%s" ...', [cArmName[k]]));
      Results[k] := RunArm(k, TrainSet, TestSet);
      WriteLn(' done.');
    end;
  finally
    TestSet.Free;
    TrainSet.Free;
  end;
  EndTime := Now;

  WriteLn;
  WriteLn('=== Results table ===');
  WriteLn('    arm   | initTrnLoss finalTrnLoss | trainLoss trainAcc | testLoss testAcc | diverged');
  WriteLn('  --------+--------------------------+--------------------+------------------+---------');
  for k := 0 to cNumArms - 1 do
    WriteLn(Format('  %-6s | %11s %12s | %9s %8s | %8s %7s | %s',
      [Results[k].Name,
       SafeF(Results[k].InitTrainLoss, 9, 4),
       SafeF(Results[k].FinalTrainLoss, 9, 4),
       SafeF(Results[k].TrainLoss, 8, 4),
       SafeF(Results[k].TrainAcc, 6, 4),
       SafeF(Results[k].TestLoss, 7, 4),
       SafeF(Results[k].TestAcc, 6, 4),
       BoolToStr(Results[k].Diverged, 'YES', 'no')]));
  WriteLn;

  // Honest head-to-head read.
  WSDelta := Results[1].TestAcc - Results[0].TestAcc;
  WriteLn(Format('Head-to-head TEST accuracy: bn=%s vs ws_gn=%s (ws_gn - bn = %s).',
    [SafeF(Results[0].TestAcc, 6, 4), SafeF(Results[1].TestAcc, 6, 4),
     SafeF(WSDelta, 6, 4)]));
  if WSDelta > 0 then
    WriteLn('=> On this toy/seed WS+GroupNorm BEAT BatchNorm at the small batch '
      + 'size, consistent with WS being batch-size independent.')
  else if WSDelta >= -cAccTol then
    WriteLn('=> On this toy/seed WS+GroupNorm MATCHED BatchNorm (within tolerance); '
      + 'on an easy synthetic task the margin is small -- reported, not assumed.')
  else
    WriteLn('=> On this toy/seed BatchNorm edged ahead; the WS advantage is a '
      + 'small-batch / harder-task effect and need not show on every easy toy.');
  WriteLn;

  // ----- Self-check: invariants that are actually TRUE (Halt(1) on failure). --
  WriteLn('=== Correctness signals ===');

  PassTrain := True;
  PassFinite := True;
  for k := 0 to cNumArms - 1 do
  begin
    if Results[k].Diverged then PassFinite := False;
    if not (Results[k].FinalTrainLoss < Results[k].InitTrainLoss) then
      PassTrain := False;
  end;
  if PassFinite then
    WriteLn('[PASS] no arm produced NaN / Inf (all losses finite).')
  else
    WriteLn('[FAIL] an arm diverged to NaN / Inf.');
  if PassTrain then
    WriteLn('[PASS] both arms trained: final train loss < initial (random-init) loss.')
  else
    WriteLn('[FAIL] some arm did not reduce its training loss below init.');

  PassBaseline := True;
  for k := 0 to cNumArms - 1 do
    if Results[k].Diverged or
       (Results[k].TrainAcc <= (1.0 / cClasses) + 0.1) then
      PassBaseline := False;
  if PassBaseline then
    WriteLn(Format('[PASS] both arms are healthy classifiers '
      + '(trainAcc > chance %.3f).', [1.0 / cClasses]))
  else
    WriteLn('[FAIL] an arm did not learn the task.');

  // WS+GroupNorm must be COMPETITIVE: within cAccTol of BatchNorm (or better).
  PassCompetitive := (Results[1].TestAcc >= Results[0].TestAcc - cAccTol);
  if PassCompetitive then
    WriteLn(Format('[PASS] WS+GroupNorm is competitive: testAcc=%s >= bn testAcc=%s - %.2f.',
      [SafeF(Results[1].TestAcc, 6, 4), SafeF(Results[0].TestAcc, 6, 4), cAccTol]))
  else
    WriteLn('[FAIL] WS+GroupNorm fell more than the tolerance below BatchNorm.');

  AllPass := PassTrain and PassFinite and PassBaseline and PassCompetitive;
  WriteLn;
  if AllPass then
    WriteLn('=> ALL CHECKS PASS.')
  else
    WriteLn('=> SOME CHECKS FAILED (see above).');
  WriteLn(Format('Total wall-clock: %.1f s', [(EndTime - StartTime) * 86400.0]));

  GTracker.Free;
  if not AllPass then Halt(1);
end.
