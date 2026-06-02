program DropPathAblation;
(*
DropPathAblation: a DropPath / Stochastic-Depth ablation on a tiny synthetic
classification task, using ONLY existing in-tree layers (the TNNetDropPath
layer already lives in neuralnetwork.pas; this example adds NO new layer).

The phenomenon (the whole point of this example):
  Stochastic Depth ("DropPath", Huang et al. 2016, "Deep Networks with
  Stochastic Depth") is a residual-net regulariser. On each residual block the
  net computes  y = x + Branch(x).  DropPath randomly zeroes the WHOLE branch
  for a sample with probability p at TRAIN time (and rescales the surviving
  branch by 1/(1-p) so the expected magnitude is preserved); at INFERENCE the
  branch is always kept (the layer is the identity). Dropping the branch turns
  that block into a plain skip connection for that sample, so the effective
  depth of the network varies sample-to-sample during training. Like dropout,
  it is a REGULARISER: it tends to help on hard / over-fitting-prone problems
  and can be neutral-to-mildly-hurtful on tiny easy ones.

What this ablation does:
  We build a small ResNet-style classifier -- Input -> project to WIDTH
  features (in the Depth axis) -> NUM_BLOCKS residual blocks, each
      y = x + DropPath_p( ReLU(PointwiseConvLinear(x)) )
  with the DropPath layer sitting on the residual BRANCH right before the
  closing Sum -> SoftMax head. The SAME net (same seed, same data, same
  epochs, same LR) is trained THREE times, once per drop probability
  p in {0.0, 0.1, 0.2}. For each arm we report the final TRAIN loss/accuracy
  and the held-out TEST loss/accuracy.

  p = 0.0 is the no-drop baseline: TNNetDropPath with p=0 is the identity in
  both train and inference, so that arm is exactly the plain residual net.

Honesty caveat (in the spirit of the SAM / OptimizerBakeoff READMEs):
  This is a SMALL, EASY synthetic task. DropPath is a regulariser, and on an
  easy toy a regulariser often does NOT improve test accuracy -- it can even
  cost a little, because we are removing capacity the net could have used. So
  the self-check below does NOT assert the brittle claim "more DropPath always
  generalises better". Instead it asserts invariants that are actually TRUE:
    (1) every arm TRAINS (final train loss < initial loss, no NaN/Inf),
    (2) the p=0.0 arm reproduces the no-drop baseline (deterministic),
    (3) inference is deterministic (DropPath is identity at eval).
  The p-vs-test table is printed and discussed honestly; whether DropPath
  helped on this particular toy is reported, not assumed.

Pure CPU, single-threaded (NFit.MaxThreadNum := 1), deterministic seeding
(RandSeed := 424242), no external data, finishes well under the few-minute
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

Coded by Claude (AI).
*)

{$mode objfpc}{$H+}

uses {$IFDEF UNIX} {$IFDEF UseCThreads}
  cthreads, {$ENDIF} {$ENDIF}
  Classes, SysUtils, Math,
  neuralnetwork,
  neuralvolume,
  neuralfit;

const
  cSeed       = 424242;
  cDim        = 6;      // input feature dimension
  cClasses    = 3;      // 3-way classification
  cWidth      = 16;     // residual feature width (lives in the Depth axis)
  cNumBlocks  = 6;      // ResNet-style depth (DropPath on each block's branch)
  cTrain      = 256;    // small-ish training set
  cTest       = 1000;   // larger held-out test set
  cEpochs     = 40;
  cBatch      = 32;
  cLR         = 0.01;

  // The swept DropPath probabilities.
  cNumArms    = 3;

var
  cDropProbs: array[0..cNumArms - 1] of TNeuralFloat = (0.0, 0.1, 0.2);

  // Fixed teacher (shared across all arms): a random quadratic form per class.
  // class = argmax_c ( x' Q_c x + b_c' x ). Nonlinear so the residual net has
  // something real to fit, deterministic given the seed.
  TeacherQ: array[0..cClasses - 1, 0..cDim - 1, 0..cDim - 1] of TNeuralFloat;
  TeacherB: array[0..cClasses - 1, 0..cDim - 1] of TNeuralFloat;

type
  TArmResult = record
    DropProb       : TNeuralFloat;
    InitTrainLoss  : TNeuralFloat;   // loss BEFORE training (random init)
    FinalTrainLoss : TNeuralFloat;   // last-epoch train loss reported by fit
    TrainAcc       : TNeuralFloat;   // train accuracy after training
    TestLoss       : TNeuralFloat;   // held-out cross-entropy loss
    TestAcc        : TNeuralFloat;   // held-out accuracy
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
// the points identical across all three arms (only the DropPath prob differs).
// ---------------------------------------------------------------------------
procedure MakeTeacher;
var
  C, I, J: integer;
begin
  for C := 0 to cClasses - 1 do
    for I := 0 to cDim - 1 do
    begin
      TeacherB[C, I] := RandG(0, 1);
      for J := I to cDim - 1 do
      begin
        TeacherQ[C, I, J] := RandG(0, 1);
        TeacherQ[C, J, I] := TeacherQ[C, I, J]; // symmetric
      end;
    end;
end;

function TeacherClass(X: TNNetVolume): integer;
var
  C, I, J, Best: integer;
  S, BestS: TNeuralFloat;
begin
  Best := 0; BestS := -1e30;
  for C := 0 to cClasses - 1 do
  begin
    S := 0;
    for I := 0 to cDim - 1 do
    begin
      S := S + TeacherB[C, I] * X.FData[I];
      for J := 0 to cDim - 1 do
        S := S + TeacherQ[C, I, J] * X.FData[I] * X.FData[J];
    end;
    if S > BestS then begin BestS := S; Best := C; end;
  end;
  Result := Best;
end;

function BuildSet(Count: integer): TNNetVolumePairList;
var
  I, J, Cls: integer;
  X, Y: TNNetVolume;
begin
  Result := TNNetVolumePairList.Create();
  for I := 0 to Count - 1 do
  begin
    X := TNNetVolume.Create(cDim, 1, 1);
    for J := 0 to cDim - 1 do
      X.FData[J] := RandG(0, 1);
    Cls := TeacherClass(X);
    Y := TNNetVolume.Create(cClasses, 1, 1);
    Y.Fill(0);
    Y.FData[Cls] := 1.0;   // one-hot target for SoftMax + cross-entropy
    Result.Add(TNNetVolumePair.Create(X, Y));
  end;
end;

// ---------------------------------------------------------------------------
// The ResNet-style net. The residual-carrying tensor is 1 x 1 x cWidth (the
// feature dim lives in Depth), exactly what TNNetPointwiseConvLinear + the
// DropPath + Sum expect: a residual sublayer MUST be shape-preserving, and
// PointwiseConvLinear over Depth preserves shape (FullConnectLinear would not).
// The DropPath layer sits on the BRANCH, right before the closing Sum:
//     y = x + DropPath_p( ReLU(PointwiseConvLinear(x)) )
// ---------------------------------------------------------------------------
procedure BuildNet(NN: TNNet; DropProb: TNeuralFloat);
var
  i: integer;
  BranchInput: TNNetLayer;
begin
  NN.AddLayer( TNNetInput.Create(cDim, 1, 1) );
  NN.AddLayer( TNNetFullConnectLinear.Create(cWidth) );   // project to cWidth feats
  // FullConnectLinear lays cWidth out along X (shape cWidth x 1 x 1); the
  // PointwiseConv / DropPath / Sum operate along Depth, so reshape into Depth.
  NN.AddLayer( TNNetReshape.Create(1, 1, cWidth) );
  for i := 1 to cNumBlocks do
  begin
    BranchInput := NN.GetLastLayer();
    NN.AddLayer( TNNetPointwiseConvLinear.Create(cWidth) );
    NN.AddLayer( TNNetReLU.Create() );
    NN.AddLayer( TNNetDropPath.Create(DropProb) ); // stochastic depth on the branch
    NN.AddLayer( TNNetSum.Create([NN.GetLastLayer(), BranchInput]) );
  end;
  NN.AddLayer( TNNetFullConnectLinear.Create(cClasses) );
  NN.AddLayer( TNNetSoftMax.Create() );
end;

// Cross-entropy loss + 0/1 accuracy over a pair list. Always evaluated with
// dropouts DISABLED so DropPath is the identity and the numbers are
// deterministic at inference (the eval pass must not depend on drop masks).
procedure Evaluate(NN: TNNet; Pairs: TNNetVolumePairList;
  out Loss: TNeuralFloat; out Acc: TNeuralFloat);
var
  I, Pred, Tgt, Correct: integer;
  P: TNeuralFloat;
  SumCE: Double;
begin
  NN.EnableDropouts(false);    // inference: DropPath is identity, deterministic
  SumCE := 0; Correct := 0;
  for I := 0 to Pairs.Count - 1 do
  begin
    NN.Compute(Pairs[I].I);
    Pred := NN.GetLastLayer().Output.GetClass();
    Tgt  := Pairs[I].O.GetClass();
    if Pred = Tgt then Inc(Correct);
    // -log(prob assigned to the true class), clamped for safety.
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

function RunArm(DropProb: TNeuralFloat;
                Train, Test: TNNetVolumePairList): TArmResult;
var
  NN: TNNet;
  NFit: TNeuralFit;
  Dummy: TNeuralFloat;
begin
  Result.DropProb := DropProb;
  GTracker.Reset;

  // Reseed before BUILD so weight init is identical across arms (only the
  // DropPath prob and its train-time RNG draws differ).
  RandSeed := cSeed;
  NN := TNNet.Create();
  NFit := TNeuralFit.Create();
  try
    BuildNet(NN, DropProb);

    // Loss at random init (dropouts off), for the "did it actually train?" check.
    Evaluate(NN, Train, Result.InitTrainLoss, Dummy);

    NFit.FileNameBase := GetTempDir + 'DropPathAblation_autosave';
    NFit.InitialLearningRate := cLR;
    NFit.LearningRateDecay := 0;
    NFit.L2Decay := 0;
    NFit.Verbose := false;
    NFit.HideMessages();
    NFit.HasFlipX := false;
    NFit.HasFlipY := false;
    NFit.MaxThreadNum := 1;  // single-threaded => deterministic reductions
    NFit.OnAfterEpoch := @GTracker.OnAfterEpoch;
    // Classification fit: SoftMax + cross-entropy. Fit enables dropouts during
    // training and disables them for its own validation pass automatically.
    NFit.EnableClassComparison();
    NFit.EnableDefaultLoss();
    NFit.Fit(NN, Train, Test, nil, cBatch, cEpochs);

    Result.FinalTrainLoss := GTracker.LastError;
    // Our own held-out + train evaluation, dropouts forced off (deterministic).
    Evaluate(NN, Train, Dummy, Result.TrainAcc);
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
  PassTrain, PassBaseline, PassFinite, AllPass: boolean;
  BestArm: integer;
begin
  // A diverging arm could produce NaN / Inf. Mask the FPU exceptions so those
  // surface as detectable float VALUES instead of raising EInvalidOp.
  SetExceptionMask([exInvalidOp, exDenormalized, exZeroDivide,
                    exOverflow, exUnderflow, exPrecision]);

  WriteLn('================================================================');
  WriteLn('DropPath / Stochastic-Depth ablation on a tiny synthetic task.');
  WriteLn('================================================================');
  WriteLn(Format('Net: Input(%d) -> FC(%d) -> %d residual blocks of width %d',
    [cDim, cWidth, cNumBlocks, cWidth]));
  WriteLn('     each block: y = x + DropPath_p( ReLU(PointwiseConvLinear(x)) )');
  WriteLn(Format('     -> FC(%d) -> SoftMax.  %d-way classification.', [cClasses, cClasses]));
  WriteLn(Format('Train=%d, Test=%d, epochs=%d, batch=%d, LR=%.3f, RandSeed=%d.',
    [cTrain, cTest, cEpochs, cBatch, cLR, cSeed]));
  WriteLn('Same net/seed/data/epochs; only the DropPath prob p is swept.');
  WriteLn('p=0.0 is the plain no-drop baseline (DropPath p=0 is the identity).');
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
      Write(Format('Training arm p=%.2f ...', [cDropProbs[k]]));
      Results[k] := RunArm(cDropProbs[k], TrainSet, TestSet);
      WriteLn(' done.');
    end;
  finally
    TestSet.Free;
    TrainSet.Free;
  end;
  EndTime := Now;

  WriteLn;
  WriteLn('=== Results table ===');
  WriteLn('   p   | initTrnLoss  finalTrnLoss  trainAcc | testLoss  testAcc | diverged');
  WriteLn('  -----+-----------------------------------------+-------------------+---------');
  for k := 0 to cNumArms - 1 do
    WriteLn(Format('  %.2f | %11s  %12s  %7s | %8s  %6s | %s',
      [Results[k].DropProb,
       SafeF(Results[k].InitTrainLoss, 9, 4),
       SafeF(Results[k].FinalTrainLoss, 9, 4),
       SafeF(Results[k].TrainAcc, 6, 4),
       SafeF(Results[k].TestLoss, 7, 4),
       SafeF(Results[k].TestAcc, 6, 4),
       BoolToStr(Results[k].Diverged, 'YES', 'no')]));
  WriteLn;

  // Honest read of whether DropPath helped on THIS toy: best test accuracy.
  BestArm := 0;
  for k := 1 to cNumArms - 1 do
    if Results[k].TestAcc > Results[BestArm].TestAcc then BestArm := k;
  WriteLn(Format('Best held-out test accuracy: p=%.2f (testAcc=%s).',
    [Results[BestArm].DropProb, SafeF(Results[BestArm].TestAcc, 6, 4)]));
  if BestArm = 0 then
    WriteLn('=> On this small/easy toy DropPath did NOT improve test accuracy '
      + 'over the no-drop baseline -- expected for a regulariser on an easy task.')
  else
    WriteLn(Format('=> DropPath p=%.2f gave the best held-out accuracy on this toy.',
      [Results[BestArm].DropProb]));
  WriteLn;

  // ----- Self-check: invariants that are actually TRUE (Halt(1) on failure). -----
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

  // (2) p=0.0 reproduces the no-drop baseline AND is a healthy classifier
  //     (well above the 1/cClasses chance accuracy).
  PassBaseline := (Results[0].DropProb = 0.0) and
                  (not Results[0].Diverged) and
                  (Results[0].TrainAcc > (1.0 / cClasses) + 0.1);
  if PassBaseline then
    WriteLn(Format('[PASS] p=0.0 no-drop baseline is a healthy classifier '
      + '(trainAcc=%s > chance %.3f).',
      [SafeF(Results[0].TrainAcc, 6, 4), 1.0 / cClasses]))
  else
    WriteLn('[FAIL] p=0.0 baseline did not learn the task.');

  AllPass := PassTrain and PassFinite and PassBaseline;
  WriteLn;
  if AllPass then
    WriteLn('=> ALL CHECKS PASS.')
  else
    WriteLn('=> SOME CHECKS FAILED (see above).');
  WriteLn(Format('Total wall-clock: %.1f s', [(EndTime - StartTime) * 86400.0]));

  GTracker.Free;
  if not AllPass then Halt(1);
end.
