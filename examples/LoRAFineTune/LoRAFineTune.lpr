program LoRAFineTune;
(*
LoRAFineTune: parameter-efficient adaptation with TNNet.AddLoRAAdapter.

A small softmax classifier is trained on a BASE synthetic task (2D Gaussian
blobs). All of its weights are then FROZEN and the network is adapted to a
RELATED-BUT-SHIFTED TARGET task (the same blobs, but each class mean is SHIFTED
and SCALED) by training ONLY a low-rank LoRA bypass (Hu et al. 2021,
https://arxiv.org/abs/2106.09685) added to ONE frozen projection of the trunk.

A LoRA adapter wraps a frozen d_in -> d_out projection with a rank-r residual:
  adapted = base(x) + (alpha/r) * B*A*x
where A: d_in -> r and B: r -> d_out are the only trainable parameters and B is
ZERO-initialised, so at step 0 the adapter is the exact identity (the adapted
forward equals the frozen base bit-for-bit). The builder uses pointwise (over
Depth) projections so the per-token (SizeX,1,Depth) stream is shape-preserving.

This demo runs the textbook LoRA experiment: it sweeps the rank r in {1,2,4,8}
and charts, for each, the number of trainable adapter params (a small fraction
of the full model) against the recovered TARGET accuracy, comparing to two
references:
  * frozen-base baseline (no adaptation -- the lower bound), and
  * full fine-tune (every weight trainable -- the upper bound).
The expected shape is "recover most of the full-fine-tune accuracy at a few %
of the parameters".

Freezing mechanism (same idiom as examples/AffineFineTune): a frozen layer has
per-layer LearningRate := 0 and net inertia 0, so its weight delta (-LR*grad) is
0 and its weights never move; error still backpropagates through it. We
additionally ASSERT a sampled frozen base weight is bit-identical after
adaptation.

Built-in self-checks (PASS/FAIL, Halt(1) on hard failure):
  * with B zero-init the adapted forward == frozen base BEFORE training (<1e-6);
  * the rank-1 adapter trains far fewer params than the full network;
  * the best LoRA rank clearly beats the frozen-base TARGET accuracy;
  * a sampled frozen base weight is unchanged after LoRA adaptation.

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
  cHidden        = 12;
  cBaseEpochs    = 120;
  cAdaptEpochs   = 200;
  cBaseLR        = 0.03;
  cAdaptLR       = 0.02;
  cAlpha         = 2.0;
  cMinBestAcc    = 0.70;   // hard floor for the best adapted accuracy
  cAccMargin     = 0.05;   // best LoRA acc must beat base acc by at least this

  cRanks: array[0..3] of integer = (1, 2, 4, 8);

  // BASE task: four well-separated 2D Gaussian blobs.
  cBaseCenters: array[0..cNumClasses - 1, 0..1] of TNeuralFloat =
    ((-2.0, -2.0),
     ( 2.0, -2.0),
     ( 2.0,  2.0),
     (-2.0,  2.0));
  cSigma = 0.55;

  // TARGET task: the SAME class layout, but each coordinate is SCALED then
  // SHIFTED by a fixed affine map -- a distribution shift the trunk must adapt
  // to. The LoRA bypass on the trunk projection supplies that extra capacity.
  cScaleX = 1.25;  cShiftX = 1.4;
  cScaleY = 0.85;  cShiftY = -1.2;

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
// distribution (same class label) so the base classifier is mismatched. The
// input is a (1,1,2) per-token stream so the trunk's pointwise projections act
// per-token (the shape LoRA's pointwise bypass preserves).
procedure MakeSample(ClassId: integer; Target: boolean; out X, Y: TNNetVolume);
var
  px, py: TNeuralFloat;
begin
  X := TNNetVolume.Create(1, 1, 2);
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

// Base trunk + head. The trunk's SECOND projection (cFrozenProjIdx) is the one
// a LoRA adapter wraps:
//   [0] Input(1,1,2)
//   [1] PointwiseConvReLU(cHidden)     -> (1,1,cHidden)
//   [2] PointwiseConvLinear(cHidden)   <- FROZEN base projection (LoRA target)
//   [3] ReLU
//   [4] FullConnectLinear(cNumClasses)
//   [5] SoftMax
const
  cFrozenProjIdx = 2; // index of the projection LoRA wraps

procedure BuildBaseNet(out NN: TNNet; out ProjLayer: TNNetLayer);
begin
  NN := TNNet.Create();
  NN.AddLayer(TNNetInput.Create(1, 1, 2));
  NN.AddLayer(TNNetPointwiseConvReLU.Create(cHidden));
  ProjLayer := NN.AddLayer(TNNetPointwiseConvLinear.Create(cHidden));
  NN.AddLayer(TNNetReLU.Create());
  NN.AddLayer(TNNetFullConnectLinear.Create(cNumClasses));
  NN.AddLayer(TNNetSoftMax.Create());
end;

// Build the SAME trunk, wrap its projection with a rank-r LoRA adapter, then
// chain the head ON TOP of the adapter output so the classifier consumes the
// ADAPTED projection. Returns the frozen projection, the adapter (Sum) output,
// and the indices of the two trainable adapter layers (A=down, B=up).
//   [0] Input  [1] PointwiseConvReLU  [2] PointwiseConvLinear (frozen proj)
//   [3] A(down) [4] B(up) [5] MulByConstant [6] Sum   <- adapter
//   [7] ReLU   [8] FullConnectLinear   [9] SoftMax    <- head on adapter
procedure BuildAdaptedNet(out NN: TNNet; Rank: integer;
  out FrozenProj, Adapted: TNNetLayer; out ADownIdx, BUpIdx: integer);
begin
  NN := TNNet.Create();
  NN.AddLayer(TNNetInput.Create(1, 1, 2));
  NN.AddLayer(TNNetPointwiseConvReLU.Create(cHidden));
  FrozenProj := NN.AddLayer(TNNetPointwiseConvLinear.Create(cHidden));
  Adapted := NN.AddLoRAAdapter(FrozenProj, Rank, cAlpha);
  ADownIdx := FrozenProj.LayerIdx + 1; // A (down)
  BUpIdx   := FrozenProj.LayerIdx + 2; // B (up)
  // Head chains off the adapter output (GetLastLayer = the Sum).
  NN.AddLayer(TNNetReLU.Create());
  NN.AddLayer(TNNetFullConnectLinear.Create(cNumClasses));
  NN.AddLayer(TNNetSoftMax.Create());
end;

// Copy the pretrained base weights into the adapted net. The adapted net has
// extra adapter layers, so CopyWeights (whole-net, index-matched) does not
// apply; copy the weight-bearing layers explicitly. The base FullConnect head
// lives at base[4]; in the adapted net it is the FullConnect AFTER the adapter.
procedure CopyBaseInto(Adapted, Base: TNNet);
var
  k: integer;
begin
  // Trunk: base[1]->adapted[1], base[2]->adapted[2] (same indices).
  for k := 1 to cFrozenProjIdx do
    Adapted.Layers[k].CopyWeights(Base.Layers[k]);
  // Head FullConnect: base[4] -> the adapted net's FullConnect (its last
  // weight-bearing layer, just before SoftMax).
  Adapted.Layers[Adapted.Layers.Count - 2].CopyWeights(Base.Layers[4]);
end;

// Sum of trainable params over layers with the given index set.
function CountParamsOf(NN: TNNet; const Idxs: array of integer): integer;
var
  k, n, Total: integer;
begin
  Total := 0;
  for k := Low(Idxs) to High(Idxs) do
    for n := 0 to NN.Layers[Idxs[k]].Neurons.Count - 1 do
      Total := Total + NN.Layers[Idxs[k]].Neurons[n].Weights.Size + 1; // +1 bias
  Result := Total;
end;

// Total trainable params over the whole net (all layers).
function CountAllParams(NN: TNNet): integer;
var
  k, n, Total: integer;
begin
  Total := 0;
  for k := 0 to NN.Layers.Count - 1 do
    for n := 0 to NN.Layers[k].Neurons.Count - 1 do
      Total := Total + NN.Layers[k].Neurons[n].Weights.Size + 1;
  Result := Total;
end;

procedure TrainEpochs(NN: TNNet; DataSet: TNNetVolumePairList; Epochs: integer);
var
  Epoch, I: integer;
  Pair: TNNetVolumePair;
begin
  for Epoch := 1 to Epochs do
    for I := 0 to DataSet.Count - 1 do
    begin
      Pair := DataSet[I];
      NN.Compute(Pair.I);
      NN.Backpropagate(Pair.O);
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

// Train a BASE classifier and return it together with the frozen-projection
// layer reference. The base is trained ONCE; each rank sweep reloads a copy of
// these base weights so all ranks start from the identical pretrained model.
procedure TrainBase(out NN: TNNet; BaseTrain: TNNetVolumePairList);
var
  ProjLayer: TNNetLayer;
begin
  BuildBaseNet(NN, ProjLayer);
  NN.SetLearningRate(cBaseLR, 0.9);
  TrainEpochs(NN, BaseTrain, cBaseEpochs);
end;

procedure RunDemo();
var
  BaseNN, NN: TNNet;
  BaseTrain, BaseTest, TargetTrain, TargetTest: TNNetVolumePairList;
  ProjLayer, FrozenProj, Adapted: TNNetLayer;
  BaseAccTarget, FullAccTarget: TNeuralFloat;
  LoRAAcc: array[Low(cRanks)..High(cRanks)] of TNeuralFloat;
  LoRAParams: array[Low(cRanks)..High(cRanks)] of integer;
  FullParams: integer;
  ProbeBefore, ProbeAfter, IdentityMaxDiff, BestAcc: TNeuralFloat;
  AdapterIdxs: array of integer;
  ADownIdx, BUpIdx: integer;
  rk, k, i: integer;
  AllOk: boolean;
begin
  RandSeed := 424242;
  BuildSet(BaseTrain,   cTrainPerClass, {Target=}False);
  BuildSet(BaseTest,    cTestPerClass,  {Target=}False);
  BuildSet(TargetTrain, cTrainPerClass, {Target=}True);
  BuildSet(TargetTest,  cTestPerClass,  {Target=}True);

  // ---------------------- pretrain the BASE classifier ----------------------
  WriteLn('=== Pretraining base classifier on BASE task ===');
  TrainBase(BaseNN, BaseTrain);
  WriteLn('  base accuracy on BASE   test: ',
    (Accuracy(BaseNN, BaseTest) * 100):6:2, ' %');
  BaseAccTarget := Accuracy(BaseNN, TargetTest);
  WriteLn('  base accuracy on TARGET test: ', (BaseAccTarget * 100):6:2,
    ' %   (distribution-shifted -> degraded)');
  WriteLn;

  // ---------------- reference: FULL fine-tune (upper bound) ------------------
  // Clone the pretrained base and fine-tune EVERYTHING on the target task.
  WriteLn('=== Reference: full fine-tune (every weight trainable) ===');
  BuildBaseNet(NN, ProjLayer);
  NN.CopyWeights(BaseNN);
  FullParams := CountAllParams(NN);
  NN.SetLearningRate(cAdaptLR, 0.9);
  TrainEpochs(NN, TargetTrain, cAdaptEpochs);
  FullAccTarget := Accuracy(NN, TargetTest);
  WriteLn('  full fine-tune TARGET accuracy: ', (FullAccTarget * 100):6:2, ' %  (',
    FullParams, ' trainable params)');
  NN.Free;
  WriteLn;

  // ----------------------- LoRA rank sweep -----------------------------------
  WriteLn('=== LoRA rank sweep (only A/B adapter params trainable) ===');
  IdentityMaxDiff := 0;
  for rk := Low(cRanks) to High(cRanks) do
  begin
    // Fresh wrapped net (trunk + rank-r adapter + head) loaded from the
    // pretrained base, so every rank starts from the identical base model.
    BuildAdaptedNet(NN, cRanks[rk], FrozenProj, Adapted, ADownIdx, BUpIdx);
    CopyBaseInto(NN, BaseNN);

    SetLength(AdapterIdxs, 2);
    AdapterIdxs[0] := ADownIdx; // A (down)
    AdapterIdxs[1] := BUpIdx;   // B (up)
    LoRAParams[rk] := CountParamsOf(NN, AdapterIdxs);

    // Identity property: with B zero-init the adapted projection equals the base
    // projection output, so the whole net's forward is unchanged at step 0. We
    // probe a sample and compare the adapted-sum layer to the frozen projection.
    NN.Compute(BaseTest[0].I);
    for i := 0 to Adapted.Output.Size - 1 do
      IdentityMaxDiff := Max(IdentityMaxDiff,
        Abs(Adapted.Output.FData[i] - FrozenProj.Output.FData[i]));

    // Freeze everything EXCEPT the two adapter layers (LR := 0, inertia 0).
    NN.SetLearningRate(cAdaptLR, 0.0);
    for k := 0 to NN.Layers.Count - 1 do
      if (k <> AdapterIdxs[0]) and (k <> AdapterIdxs[1]) then
        NN.Layers[k].LearningRate := 0.0;

    // Probe a frozen base weight before/after to prove it does not move.
    ProbeBefore := NN.Layers[cFrozenProjIdx].Neurons[0].Weights.FData[0];
    TrainEpochs(NN, TargetTrain, cAdaptEpochs);
    ProbeAfter := NN.Layers[cFrozenProjIdx].Neurons[0].Weights.FData[0];
    LoRAAcc[rk] := Accuracy(NN, TargetTest);

    if ProbeAfter <> ProbeBefore then
    begin
      WriteLn('  CHECK FAIL: frozen base weight moved at rank ', cRanks[rk]);
      Halt(1);
    end;
    NN.Free;
  end;
  WriteLn;

  // ------------------------------- REPORT ------------------------------------
  WriteLn('---------------------------------------------------------------');
  WriteLn('Rank sweep: trainable adapter params vs recovered TARGET accuracy');
  WriteLn('  full network (upper bound): ', FullParams:5, ' params  -> ',
    (FullAccTarget * 100):6:2, ' %');
  WriteLn('  frozen base (lower bound) :     0 params  -> ',
    (BaseAccTarget * 100):6:2, ' %');
  WriteLn('  ---------------------------------------------------');
  WriteLn('   rank | adapter params | % of full | TARGET acc');
  WriteLn('  ------+----------------+-----------+-----------');
  BestAcc := 0;
  for rk := Low(cRanks) to High(cRanks) do
  begin
    WriteLn('   ', cRanks[rk]:4, ' | ', LoRAParams[rk]:14, ' | ',
      ((LoRAParams[rk] / FullParams) * 100):8:2, ' % | ',
      (LoRAAcc[rk] * 100):7:2, ' %');
    if LoRAAcc[rk] > BestAcc then BestAcc := LoRAAcc[rk];
  end;
  WriteLn('---------------------------------------------------------------');
  WriteLn;

  // ------------------------------- CHECKS ------------------------------------
  AllOk := True;

  if IdentityMaxDiff < 1e-6 then
    WriteLn('CHECK 1 PASS: B zero-init -> adapted == frozen base before training (max diff ',
      IdentityMaxDiff:0:9, ').')
  else
  begin
    WriteLn('CHECK 1 FAIL: adapter not identity at init (max diff ',
      IdentityMaxDiff:0:9, ').');
    AllOk := False;
  end;

  if LoRAParams[Low(cRanks)] < FullParams div 2 then
    WriteLn('CHECK 2 PASS: rank-', cRanks[Low(cRanks)], ' adapter uses far fewer params (',
      LoRAParams[Low(cRanks)], ' < ', FullParams, ').')
  else
  begin
    WriteLn('CHECK 2 FAIL: adapter not parameter-efficient.');
    AllOk := False;
  end;

  if (BestAcc > BaseAccTarget + cAccMargin) and (BestAcc > cMinBestAcc) then
    WriteLn('CHECK 3 PASS: best LoRA recovers accuracy (', (BestAcc * 100):5:2,
      ' % > base ', (BaseAccTarget * 100):5:2, ' % + margin, and > ',
      (cMinBestAcc * 100):3:0, ' %).')
  else
  begin
    WriteLn('CHECK 3 FAIL: LoRA did not recover enough accuracy.');
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

  BaseNN.Free;
  BaseTrain.Free;
  BaseTest.Free;
  TargetTrain.Free;
  TargetTest.Free;
end;

begin
  RunDemo();
end.
