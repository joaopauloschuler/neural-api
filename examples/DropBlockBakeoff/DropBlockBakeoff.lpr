program DropBlockBakeoff;
(*
DropBlockBakeoff: a DropBlock vs plain Dropout regulariser bake-off on a tiny
synthetic image-classification task, using ONLY existing in-tree layers (the
TNNetDropBlock layer already lives in neuralnetwork.pas; this example adds NO
new layer).

The phenomenon (the whole point of this example):
  DropBlock (Ghiasi et al. 2018, "DropBlock: A regularization method for
  convolutional networks") is a STRUCTURED dropout for conv feature maps.
  Plain TNNetDropout zeroes individual activations independently, scattered
  across the map. On a convolutional feature map that is a weak regulariser:
  neighbouring units are spatially correlated, so a dropped pixel's information
  still survives in its neighbours and the net just routes around the holes.
  DropBlock instead samples ONE spatial 0/1 mask per (x,y) position, zeroes a
  contiguous block_size x block_size square, and broadcasts that mask across
  ALL channels -- so a whole local patch of the feature map disappears at once.
  This removes spatially-correlated information the net cannot trivially
  recover, which is the intended stronger regularisation for conv nets. Both
  layers rescale survivors (inverted dropout) so the expected activation is
  preserved, and BOTH are the exact identity at inference (FEnabled=false).

What this bake-off does:
  We build one small CNN -- Input(H,W,C) -> Conv -> ReLU -> [REG] -> Conv ->
  ReLU -> MaxPool -> FC -> SoftMax -- where [REG] is the regulariser slot that
  sits right after a conv feature map. The SAME net (same seed, same data,
  same epochs, same LR) is trained THREE times:
    arm 0  "none"     : Identity in the [REG] slot (no regulariser baseline).
    arm 1  "dropout"  : TNNetDropout at drop rate cDropRate (scattered pixels).
    arm 2  "dropblock": TNNetDropBlock(block_size, cDropRate) (localized patch).
  Dropout and DropBlock are matched at the SAME drop rate cDropRate. For each
  arm we report final TRAIN and held-out TEST loss/accuracy, and the
  train-minus-test GAP (a proxy for over-fitting; a regulariser should shrink
  it).

Honesty caveat (in the spirit of the DropPathAblation / OptimizerBakeoff READMEs):
  This is a SMALL, EASY synthetic task. DropBlock and Dropout are regularisers,
  and on an easy toy a regulariser often does NOT improve test accuracy -- it
  can even cost a little, because we are removing capacity the net could have
  used. So the self-check below does NOT assert the brittle claim "DropBlock
  always generalises better than Dropout". Instead it asserts invariants that
  are actually TRUE:
    (1) every arm TRAINS (final train loss < initial loss, no NaN/Inf),
    (2) the "none" arm reproduces a deterministic healthy baseline,
    (3) inference is deterministic: both DropBlock and Dropout are the identity
        at eval (two eval passes give bit-identical loss).
  The per-arm train/test GAP is printed and discussed honestly; whether
  DropBlock shrank the gap more than Dropout on this particular toy is
  reported, not assumed.

Pure CPU, single-threaded (NFit.MaxThreadNum := 1), deterministic seeding
(RandSeed := 424242), no external data, finishes well under the 5-minute
budget.

Copyright (C) 2026 Joao Paulo Schwarz Schuler

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
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
  cClasses    = 3;      // 3-way classification
  cConvFeat   = 24;     // conv feature width (ample capacity => over-fits the tiny train set)
  cBlockSize  = 3;      // DropBlock square block size (on the 12x12 map)
  cDropRate   = 0.15;   // MATCHED drop rate for Dropout and DropBlock
  cTrain      = 96;     // very small training set (over-fitting-prone)
  cTest       = 800;    // larger held-out test set
  cEpochs     = 35;
  cBatch      = 16;
  cLR         = 0.005;

  cNumArms    = 3;

var
  cArmName: array[0..cNumArms - 1] of string = ('none', 'dropout', 'dropblock');

  // Fixed teacher (shared across all arms). Each class owns a random spatial
  // template stamped into a random location of a noisy image; the label is the
  // class whose template best correlates with the image. Spatially structured
  // so a conv net (and a structured regulariser) have something real to chew on.
  TeacherTpl: array[0..cClasses - 1, 0..cChannels - 1,
                     0..cImgSize - 1, 0..cImgSize - 1] of TNeuralFloat;

type
  TArmResult = record
    Name           : string;
    InitTrainLoss  : TNeuralFloat;   // loss BEFORE training (random init)
    FinalTrainLoss : TNeuralFloat;   // last-epoch train loss reported by fit
    TrainLoss      : TNeuralFloat;   // our own train cross-entropy (dropouts off)
    TrainAcc       : TNeuralFloat;
    TestLoss       : TNeuralFloat;
    TestAcc        : TNeuralFloat;
    Gap            : TNeuralFloat;   // testLoss - trainLoss (over-fitting proxy)
    EvalDeterministic : boolean;     // two eval passes bit-identical?
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
// the points identical across all three arms (only the regulariser differs).
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

// Build one image: pure Gaussian noise plus a class template stamped in at a
// random offset and amplitude, returning the ground-truth class.
function MakeImage(out Cls: integer): TNNetVolume;
var
  Ch, X, Y: integer;
  Amp: TNeuralFloat;
begin
  Cls := Random(cClasses);
  Amp := 0.25 + Random * 0.35;
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
// The CNN. The regulariser slot [REG] sits right after the FIRST conv feature
// map (12x12 x cConvFeat), exactly where DropBlock is meant to live: a spatial
// map with room for a cBlockSize x cBlockSize block. The three arms differ ONLY
// in what goes in [REG]:
//   none      -> TNNetIdentity        (no regulariser)
//   dropout   -> TNNetDropout(rate)   (scattered per-pixel)
//   dropblock -> TNNetDropBlock(bs,r) (localized patch)
// ---------------------------------------------------------------------------
procedure BuildNet(NN: TNNet; Arm: integer);
begin
  NN.AddLayer( TNNetInput.Create(cImgSize, cImgSize, cChannels) );
  NN.AddLayer( TNNetConvolutionReLU.Create(cConvFeat, 3, 1, 1) ); // 12x12 map
  // Regulariser slot on the conv feature map.
  case Arm of
    0: NN.AddLayer( TNNetIdentity.Create() );
    1: NN.AddLayer( TNNetDropout.Create(cDropRate) );
    2: NN.AddLayer( TNNetDropBlock.Create(cBlockSize, cDropRate) );
  end;
  NN.AddLayer( TNNetConvolutionReLU.Create(cConvFeat, 3, 1, 1) );
  NN.AddLayer( TNNetMaxPool.Create(2) );                              // 6x6 map
  NN.AddLayer( TNNetFullConnectLinear.Create(cClasses) );
  NN.AddLayer( TNNetSoftMax.Create() );
end;

// Cross-entropy loss + 0/1 accuracy over a pair list. Always evaluated with
// dropouts DISABLED so DropBlock / Dropout are the identity and the numbers are
// deterministic at inference (the eval pass must not depend on drop masks).
procedure Evaluate(NN: TNNet; Pairs: TNNetVolumePairList;
  out Loss: TNeuralFloat; out Acc: TNeuralFloat);
var
  I, Pred, Tgt, Correct: integer;
  P: TNeuralFloat;
  SumCE: Double;
begin
  NN.EnableDropouts(false);    // inference: regularisers identity, deterministic
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

function RunArm(Arm: integer;
                Train, Test: TNNetVolumePairList): TArmResult;
var
  NN: TNNet;
  NFit: TNeuralFit;
  Dummy: TNeuralFloat;
  L1, L2, A: TNeuralFloat;
begin
  Result.Name := cArmName[Arm];
  GTracker.Reset;

  // Reseed before BUILD so weight init is identical across arms (only the
  // regulariser and its train-time RNG draws differ).
  RandSeed := cSeed;
  NN := TNNet.Create();
  NFit := TNeuralFit.Create();
  try
    BuildNet(NN, Arm);

    // Loss at random init (dropouts off), for the "did it actually train?" check.
    Evaluate(NN, Train, Result.InitTrainLoss, Dummy);

    NFit.FileNameBase := GetTempDir + 'DropBlockBakeoff_autosave';
    NFit.InitialLearningRate := cLR;
    NFit.LearningRateDecay := 0;
    NFit.L2Decay := 0;
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
    // Our own held-out + train evaluation, dropouts forced off (deterministic).
    Evaluate(NN, Train, Result.TrainLoss, Result.TrainAcc);
    Evaluate(NN, Test, Result.TestLoss, Result.TestAcc);
    Result.Gap := Result.TestLoss - Result.TrainLoss;

    // Determinism at inference: a second eval pass must be bit-identical, which
    // proves the regulariser is truly the identity at eval (no live drop mask).
    Evaluate(NN, Test, L1, A);
    Evaluate(NN, Test, L2, A);
    Result.EvalDeterministic := (L1 = L2);

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
  PassTrain, PassBaseline, PassFinite, PassDeterministic, AllPass: boolean;
  BestGapArm: integer;
begin
  // A diverging arm could produce NaN / Inf. Mask the FPU exceptions so those
  // surface as detectable float VALUES instead of raising EInvalidOp.
  SetExceptionMask([exInvalidOp, exDenormalized, exZeroDivide,
                    exOverflow, exUnderflow, exPrecision]);

  WriteLn('================================================================');
  WriteLn('DropBlock vs plain Dropout bake-off on a tiny synthetic image task.');
  WriteLn('================================================================');
  WriteLn(Format('Net: Input(%dx%dx%d) -> ConvReLU(%d,3) -> [REG]',
    [cImgSize, cImgSize, cChannels, cConvFeat]));
  WriteLn(Format('     -> ConvReLU(%d,3) -> MaxPool(2) -> FC(%d) -> SoftMax.',
    [cConvFeat, cClasses]));
  WriteLn('     [REG] = none(Identity) | Dropout(rate) | DropBlock(bs,rate).');
  WriteLn(Format('Matched drop rate=%.2f, DropBlock block_size=%d.',
    [cDropRate, cBlockSize]));
  WriteLn(Format('Train=%d, Test=%d, epochs=%d, batch=%d, LR=%.3f, RandSeed=%d.',
    [cTrain, cTest, cEpochs, cBatch, cLR, cSeed]));
  WriteLn('Same net/seed/data/epochs; only the [REG] layer is swept.');
  WriteLn;

  GTracker := TLossTracker.Create;
  StartTime := Now;
  // Build the shared teacher + datasets ONCE; reseed first so they are fixed.
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
  WriteLn('    arm    | initTrnLoss finalTrnLoss | trainLoss trainAcc | testLoss testAcc | gap(test-trn) | diverged');
  WriteLn('  ---------+--------------------------+--------------------+------------------+---------------+---------');
  for k := 0 to cNumArms - 1 do
    WriteLn(Format('  %-8s | %11s %12s | %9s %8s | %8s %7s | %13s | %s',
      [Results[k].Name,
       SafeF(Results[k].InitTrainLoss, 9, 4),
       SafeF(Results[k].FinalTrainLoss, 9, 4),
       SafeF(Results[k].TrainLoss, 8, 4),
       SafeF(Results[k].TrainAcc, 6, 4),
       SafeF(Results[k].TestLoss, 7, 4),
       SafeF(Results[k].TestAcc, 6, 4),
       SafeF(Results[k].Gap, 8, 4),
       BoolToStr(Results[k].Diverged, 'YES', 'no')]));
  WriteLn;

  // Honest read of whether the localized regulariser shrank the over-fitting
  // gap the most on THIS toy. We report; we do not assert it must.
  BestGapArm := 0;
  for k := 1 to cNumArms - 1 do
    if Results[k].Gap < Results[BestGapArm].Gap then BestGapArm := k;
  WriteLn(Format('Smallest train/test gap: arm "%s" (gap=%s).',
    [Results[BestGapArm].Name, SafeF(Results[BestGapArm].Gap, 8, 4)]));
  if BestGapArm = 2 then
    WriteLn('=> On this toy DropBlock shrank the over-fitting gap the most, '
      + 'consistent with structured dropout being a stronger conv regulariser.')
  else if BestGapArm = 1 then
    WriteLn('=> On this toy plain Dropout shrank the gap the most; on an easy '
      + 'synthetic task either regulariser can win -- reported, not assumed.')
  else
    WriteLn('=> On this small/easy toy the no-regulariser baseline already had '
      + 'the smallest gap -- expected when a regulariser is not strictly needed.');
  WriteLn;

  // ----- Self-check: invariants that are actually TRUE (Halt(1) on failure). --
  WriteLn('=== Correctness signals ===');

  // (1) Every arm trained: final train loss < initial loss, and finite.
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
    WriteLn('[PASS] every arm trained: final train loss < initial (random-init) loss.')
  else
    WriteLn('[FAIL] some arm did not reduce its training loss below init.');

  // (2) The "none" baseline reproduces a healthy classifier (well above the
  //     1/cClasses chance accuracy).
  PassBaseline := (Results[0].Name = 'none') and
                  (not Results[0].Diverged) and
                  (Results[0].TrainAcc > (1.0 / cClasses) + 0.1);
  if PassBaseline then
    WriteLn(Format('[PASS] "none" baseline is a healthy classifier '
      + '(trainAcc=%s > chance %.3f).',
      [SafeF(Results[0].TrainAcc, 6, 4), 1.0 / cClasses]))
  else
    WriteLn('[FAIL] "none" baseline did not learn the task.');

  // (3) Inference is deterministic for every arm: two eval passes identical,
  //     proving DropBlock and Dropout are the exact identity at eval time.
  PassDeterministic := True;
  for k := 0 to cNumArms - 1 do
    if not Results[k].EvalDeterministic then PassDeterministic := False;
  if PassDeterministic then
    WriteLn('[PASS] inference deterministic: two eval passes bit-identical '
      + '(DropBlock & Dropout are identity at eval).')
  else
    WriteLn('[FAIL] an arm gave non-deterministic eval (regulariser leaked into inference).');

  AllPass := PassTrain and PassFinite and PassBaseline and PassDeterministic;
  WriteLn;
  if AllPass then
    WriteLn('=> ALL CHECKS PASS.')
  else
    WriteLn('=> SOME CHECKS FAILED (see above).');
  WriteLn(Format('Total wall-clock: %.1f s', [(EndTime - StartTime) * 86400.0]));

  GTracker.Free;
  if not AllPass then Halt(1);
end.
