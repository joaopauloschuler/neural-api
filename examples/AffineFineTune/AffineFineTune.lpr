program AffineFineTune;
(*
AffineFineTune: BitFit-style cheap adaptation with TNNet.AddAffineBlock.

A small softmax classifier is trained on a BASE synthetic task (2D Gaussian
blobs). All of its weights are then FROZEN and the network is adapted to a
RELATED-BUT-SHIFTED TARGET task (the same blobs, but each class mean is
SHIFTED and SCALED) by fine-tuning ONLY a couple of per-channel affine blocks
(TNNet.AddAffineBlock -> TNNetChannelMul then TNNetChannelBias, a learnable
y[d] = gamma[d]*x[d] + beta[d]). This mirrors BitFit (Zaken et al. 2021): adapt
a frozen backbone by training a tiny number of parameters instead of the whole
model.

The affine blocks are present in the net from the start, initialised to the
exact identity (gamma=1, beta=0). They are FROZEN (per-layer LearningRate := 0)
during base training so the base task sees a plain MLP. For adaptation the roles
flip: the whole trunk + head are frozen and only the two affine blocks learn.

Freezing mechanism (documented in README): per-layer NN.Layers[k].LearningRate
is set to 0.0. A layer with LR=0 still backpropagates the input error (so the
gradient flows through to the affine blocks) but its own weight delta is scaled
by -LR = 0, so its weights never move. We additionally ASSERT that a sampled
base-trunk weight is bit-identical before and after adaptation.

Built-in self-checks (PASS/FAIL, Halt(1) on hard failure):
  * trainable-param count for the affine-only fine-tune is a small fraction of
    the full network's trainable params;
  * affine-only fine-tuned target accuracy clearly beats the frozen-base target
    accuracy (affine_acc > base_acc + margin) and exceeds 0.7;
  * a sampled frozen base weight is unchanged after adaptation.

Pure CPU, single-threaded manual training loop, runs in a few seconds.

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
  neuralvolume;

const
  cNumClasses    = 4;
  cTrainPerClass = 300;
  cTestPerClass  = 400;
  cHidden        = 16;
  cBaseEpochs    = 120;
  cAdaptEpochs   = 120;
  cBaseLR        = 0.03;
  cAdaptLR       = 0.05;
  cMinAffineAcc  = 0.70;   // hard floor for the adapted accuracy
  cAccMargin     = 0.05;   // affine_acc must beat base_acc by at least this

  // BASE task: four well-separated 2D Gaussian blobs.
  cBaseCenters: array[0..cNumClasses - 1, 0..1] of TNeuralFloat =
    ((-2.0, -2.0),
     ( 2.0, -2.0),
     ( 2.0,  2.0),
     (-2.0,  2.0));
  cSigma = 0.55;

  // TARGET task: the SAME class layout, but each coordinate is SCALED then
  // SHIFTED by a fixed affine map. A per-channel affine on the features is
  // exactly the right capacity to undo this kind of input distribution shift.
  cScaleX = 1.6;  cShiftX = 3.0;
  cScaleY = 0.7;  cShiftY = -2.5;

// Box-Muller N(0,1) sample (no external dependency).
function RandomGauss(): TNeuralFloat;
var
  U1, U2: TNeuralFloat;
begin
  repeat
    U1 := Random;
  until U1 > 1e-9;
  U2 := Random;
  Result := Sqrt(-2 * Ln(U1)) * Cos(2 * Pi * U2);
end;

// Build one sample. When Target=true the point is drawn from the SHIFTED/SCALED
// distribution (same class label) so the base classifier is mismatched.
procedure MakeSample(ClassId: integer; Target: boolean; out X, Y: TNNetVolume);
var
  px, py: TNeuralFloat;
begin
  X := TNNetVolume.Create(2, 1, 1);
  Y := TNNetVolume.Create(cNumClasses, 1, 1);
  px := cBaseCenters[ClassId][0] + RandomGauss() * cSigma;
  py := cBaseCenters[ClassId][1] + RandomGauss() * cSigma;
  if Target then
  begin
    px := px * cScaleX + cShiftX;
    py := py * cScaleY + cShiftY;
  end;
  X.FData[0] := px;
  X.FData[1] := py;
  Y.Fill(0);
  Y.FData[ClassId] := 1.0;
end;

procedure BuildSet(out Pairs: TNNetVolumePairList; PerClass: integer;
  Target: boolean);
var
  C, I: integer;
  X, Y: TNNetVolume;
begin
  Pairs := TNNetVolumePairList.Create();
  for C := 0 to cNumClasses - 1 do
    for I := 1 to PerClass do
    begin
      MakeSample(C, Target, X, Y);
      Pairs.Add(TNNetVolumePair.Create(X, Y));
    end;
end;

// Layer indices (built by BuildNet). A TNNetFullConnect* emits its units on
// the SizeX axis (shape (cHidden,1,1), Depth=1), but TNNetChannelMul/Bias scale
// per-DEPTH. So each affine block is preceded by a TNNetReshape that moves the
// cHidden features onto the Depth axis (1,1,cHidden); the affine then learns a
// genuine per-feature gamma[cHidden] and beta[cHidden].
//   [0]  Input(2)
//   [1]  FullConnectReLU(cHidden)        -> (cHidden,1,1)
//   [2]  Reshape(1,1,cHidden)            -> features on Depth
//   [3]  ChannelMul  (affine block A: gamma)   <- AddAffineBlock
//   [4]  ChannelBias (affine block A: beta )
//   [5]  FullConnectReLU(cHidden)        -> (cHidden,1,1)
//   [6]  Reshape(1,1,cHidden)
//   [7]  ChannelMul  (affine block B: gamma)   <- AddAffineBlock
//   [8]  ChannelBias (affine block B: beta )
//   [9]  FullConnectLinear(cNumClasses)
//   [10] SoftMax
const
  cAffineLayers: array[0..3] of integer = (3, 4, 7, 8); // the only adapted layers

procedure BuildNet(out NN: TNNet);
begin
  NN := TNNet.Create();
  NN.AddLayer(TNNetInput.Create(2, 1, 1));
  NN.AddLayer(TNNetFullConnectReLU.Create(cHidden));
  NN.AddLayer(TNNetReshape.Create(1, 1, cHidden));   // features -> Depth axis
  NN.AddAffineBlock;                                 // block A (frozen at base)
  NN.AddLayer(TNNetFullConnectReLU.Create(cHidden));
  NN.AddLayer(TNNetReshape.Create(1, 1, cHidden));
  NN.AddAffineBlock;                                 // block B (frozen at base)
  NN.AddLayer(TNNetFullConnectLinear.Create(cNumClasses));
  NN.AddLayer(TNNetSoftMax.Create());
end;

function IsAffineLayer(Idx: integer): boolean;
var
  k: integer;
begin
  Result := False;
  for k := Low(cAffineLayers) to High(cAffineLayers) do
    if cAffineLayers[k] = Idx then Exit(True);
end;

// Per-layer freeze: a frozen layer's LearningRate is set to 0, so its weight
// delta (-LR*grad) is 0 and its weights never move; error still backpropagates
// through it. NN.SetLearningRate is called FIRST to set the trained layers'
// rate AND to set inertia to 0 net-wide -- with inertia=0 a frozen (LR=0) layer
// has no momentum term either, so it provably cannot drift (UpdateWeights then
// does FWeights.Add(FDelta) with FDelta==0).
procedure SetTrainable(NN: TNNet; AffineOnly: boolean; LR: TNeuralFloat);
var
  k: integer;
begin
  NN.SetLearningRate(LR, {Inertia=}0.0);
  for k := 0 to NN.Layers.Count - 1 do
  begin
    // Base phase: freeze the affine blocks (kept identity).
    // Affine phase: freeze everything except the affine blocks.
    if AffineOnly <> IsAffineLayer(k) then
      NN.Layers[k].LearningRate := 0.0;
  end;
end;

// Sum of trainable params over the layers that currently have LR > 0.
function CountTrainableParams(NN: TNNet; AffineOnly: boolean): integer;
var
  k, n: integer;
  Total: integer;
begin
  Total := 0;
  for k := 0 to NN.Layers.Count - 1 do
  begin
    if AffineOnly and (not IsAffineLayer(k)) then Continue;
    if (not AffineOnly) and IsAffineLayer(k) then Continue;
    for n := 0 to NN.Layers[k].Neurons.Count - 1 do
      Total := Total + NN.Layers[k].Neurons[n].Weights.Size + 1; // +1 bias
  end;
  Result := Total;
end;

procedure TrainEpochs(NN: TNNet; DataSet: TNNetVolumePairList; Epochs: integer;
  const Tag: string);
var
  Epoch, I: integer;
  Pair: TNNetVolumePair;
  Loss, Diff: TNeuralFloat;
begin
  for Epoch := 1 to Epochs do
  begin
    Loss := 0;
    for I := 0 to DataSet.Count - 1 do
    begin
      Pair := DataSet[I];
      NN.Compute(Pair.I);
      NN.Backpropagate(Pair.O);
      Diff := -Ln(Max(1e-9, NN.GetLastLayer().Output.FData[Pair.O.GetClass()]));
      Loss := Loss + Diff;
    end;
    if (Epoch mod 40 = 0) or (Epoch = 1) then
      WriteLn('  [', Tag, '] epoch ', Epoch:4, '  mean_nll=',
        (Loss / DataSet.Count):8:5);
  end;
end;

function Accuracy(NN: TNNet; DataSet: TNNetVolumePairList): TNeuralFloat;
var
  I, Hits: integer;
  Pair: TNNetVolumePair;
begin
  Hits := 0;
  for I := 0 to DataSet.Count - 1 do
  begin
    Pair := DataSet[I];
    NN.Compute(Pair.I);
    if NN.GetLastLayer().Output.GetClass() = Pair.O.GetClass() then Inc(Hits);
  end;
  Result := Hits / DataSet.Count;
end;

procedure RunDemo();
var
  NN: TNNet;
  BaseTrain, BaseTest, TargetTrain, TargetTest: TNNetVolumePairList;
  FullParams, AffineParams: integer;
  BaseAccBase, BaseAccTarget, AffineAccTarget: TNeuralFloat;
  ProbeWeightBefore, ProbeWeightAfter: TNeuralFloat;
  AllOk: boolean;
  ParamFrac: TNeuralFloat;
begin
  RandSeed := 424242;
  BuildNet(NN);
  BuildSet(BaseTrain,   cTrainPerClass, {Target=}False);
  BuildSet(BaseTest,    cTestPerClass,  {Target=}False);
  BuildSet(TargetTrain, cTrainPerClass, {Target=}True);
  BuildSet(TargetTest,  cTestPerClass,  {Target=}True);
  try
    WriteLn('Architecture:');
    NN.PrintSummary();
    WriteLn;

    FullParams   := CountTrainableParams(NN, {AffineOnly=}False);
    AffineParams := CountTrainableParams(NN, {AffineOnly=}True);
    ParamFrac    := AffineParams / FullParams;

    // -------------------- BASE TRAINING (affine frozen) --------------------
    WriteLn('=== Base training on BASE task (affine blocks frozen=identity) ===');
    SetTrainable(NN, {AffineOnly=}False, cBaseLR);
    TrainEpochs(NN, BaseTrain, cBaseEpochs, 'base');
    BaseAccBase   := Accuracy(NN, BaseTest);
    BaseAccTarget := Accuracy(NN, TargetTest);
    WriteLn;
    WriteLn('Frozen base model accuracy:');
    WriteLn('  on BASE   test: ', (BaseAccBase * 100):6:2, ' %');
    WriteLn('  on TARGET test: ', (BaseAccTarget * 100):6:2,
      ' %   (distribution-shifted -> degraded)');
    WriteLn;

    // Probe a trunk weight so we can prove the base weights do not move.
    ProbeWeightBefore := NN.Layers[1].Neurons[0].Weights.FData[0];

    // ----------- AFFINE-ONLY ADAPTATION (trunk + head frozen) --------------
    WriteLn('=== Affine-only fine-tune on TARGET task (BitFit-style) ===');
    WriteLn('  Trainable: only the ', Length(cAffineLayers),
      ' affine layers (gamma/beta); everything else LearningRate:=0.');
    SetTrainable(NN, {AffineOnly=}True, cAdaptLR);
    TrainEpochs(NN, TargetTrain, cAdaptEpochs, 'affine');
    AffineAccTarget := Accuracy(NN, TargetTest);
    WriteLn;

    ProbeWeightAfter := NN.Layers[1].Neurons[0].Weights.FData[0];

    // ------------------------------ REPORT ---------------------------------
    WriteLn('---------------------------------------------------------------');
    WriteLn('Trainable-parameter comparison:');
    WriteLn('  full network        : ', FullParams:6, ' params');
    WriteLn('  affine-only fine-tune: ', AffineParams:6, ' params  (',
      (ParamFrac * 100):5:2, ' % of full)');
    WriteLn;
    WriteLn('Accuracy on TARGET test set:');
    WriteLn('  frozen base          : ', (BaseAccTarget   * 100):6:2, ' %');
    WriteLn('  affine-only fine-tune: ', (AffineAccTarget * 100):6:2, ' %');
    WriteLn('---------------------------------------------------------------');
    WriteLn;

    // ------------------------------ CHECKS ---------------------------------
    AllOk := True;

    if ParamFrac < 0.20 then
      WriteLn('CHECK 1 PASS: affine path is a small fraction of trainable params (',
        (ParamFrac * 100):5:2, ' %).')
    else
    begin
      WriteLn('CHECK 1 FAIL: affine path not a small fraction (', (ParamFrac*100):5:2, ' %).');
      AllOk := False;
    end;

    if (AffineAccTarget > BaseAccTarget + cAccMargin) and
       (AffineAccTarget > cMinAffineAcc) then
      WriteLn('CHECK 2 PASS: affine fine-tune adapts (', (AffineAccTarget*100):5:2,
        ' % > base ', (BaseAccTarget*100):5:2, ' % + margin, and > ',
        (cMinAffineAcc*100):3:0, ' %).')
    else
    begin
      WriteLn('CHECK 2 FAIL: affine fine-tune did not adapt enough.');
      AllOk := False;
    end;

    if ProbeWeightAfter = ProbeWeightBefore then
      WriteLn('CHECK 3 PASS: frozen base weight unchanged (', ProbeWeightBefore:0:8, ').')
    else
    begin
      WriteLn('CHECK 3 FAIL: frozen base weight moved (', ProbeWeightBefore:0:8,
        ' -> ', ProbeWeightAfter:0:8, ').');
      AllOk := False;
    end;

    WriteLn;
    if AllOk then
      WriteLn('ALL CHECKS PASSED.')
    else
    begin
      WriteLn('ONE OR MORE CHECKS FAILED.');
      Halt(1);
    end;
  finally
    NN.Free;
    BaseTrain.Free;
    BaseTest.Free;
    TargetTrain.Free;
    TargetTest.Free;
  end;
end;

begin
  RunDemo();
end.
