program TracInfluence;
(*
TracInfluence: a self-contained TracIn training-data-attribution demo
(Pruthi, Liu, Sundararajan & Yan 2020, "Estimating Training Data Influence by
Tracing Gradient Descent"). Where SaliencyReport / ActivationPatchingReport
attribute a prediction to INPUT FEATURES or to LAYERS, TracIn attributes it back
to the TRAINING EXAMPLES that shaped the model: the influence of a training
point z_train on a test point z_test is the dot product of their loss gradients,

    influence(z_train, z_test) = < grad_loss(z_train), grad_loss(z_test) > .

This example uses the single-checkpoint "TracInLast" form: the FINAL trained
weights only (one gradient-dot similarity, no checkpoint summation). A positive
influence marks a PROPONENT (a training point that pushed the model toward the
test prediction); a negative influence marks an OPPONENT (a training point that
pushed against it).

The headline result of the paper is that TracIn surfaces MISLABELLED training
data: a wrongly-labelled point becomes the strongest OPPONENT of the test points
it corrupts. This demo plants exactly ONE mislabelled training example (its
label flipped), then for a corrupted test point ranks every training point by
TracIn influence and ASSERTS the planted mislabel lands among the top-K most
negative opponents — printing a graded PASS/FAIL line.

Mechanics in this library: per-sample weight gradients are read out the same way
FisherImportanceReport / GradientConflictReport do — SetBatchUpdate(true) so the
per-sample gradient lands in each neuron's Delta (and FBiasDelta) instead of
being consumed inline, ClearDeltas before each backward, never UpdateWeights
(the trained net is frozen, this is a measurement). Each sample's gradient is
flattened (and divided back out by the layer learning rate, since Delta is
-LR*grad) into one vector; the per-sample sign cancels in the dot product.

Cost is O(N_train) backward passes per test point; N_train is kept to a few
hundred (see README). Pure CPU, no dataset download, well under a minute.

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
  cInDim     = 2;       // 2-D blob task (easy to picture / clearly separable)
  cHidden    = 16;
  cClasses   = 2;
  cNTrain    = 300;     // O(N_train) backward passes per test point: kept small
  cNTest     = 60;
  cEpochs    = 40;
  cLR        = 0.05;
  cTopK      = 5;       // top-K proponents / opponents to print
  cClusterSd = 0.35;    // cluster spread (clusters clearly separable)

type
  TFloatArr = array of TNeuralFloat;

var
  NN: TNNet;
  TrainX, TrainY, TestX, TestY: TNNetVolumePairList;
  GradLen: integer;            // length of the flattened gradient vector
  MislabelIdx: integer;        // index of the planted mislabelled train point
  MislabelTrueCls: integer;    // its true class (before the flip)

// Class centroids for a clearly-separable 2-cluster problem.
procedure ClassCentre(Cls: integer; out CX, CY: TNeuralFloat);
begin
  if Cls = 0 then begin CX := -1.5; CY := -1.5; end
  else              begin CX :=  1.5; CY :=  1.5; end;
end;

procedure MakeSample(out X, Y: TNNetVolume; Cls: integer);
var
  CX, CY: TNeuralFloat;
begin
  X := TNNetVolume.Create(cInDim, 1, 1);
  Y := TNNetVolume.Create(cClasses, 1, 1);
  ClassCentre(Cls, CX, CY);
  X.Raw[0] := CX + RandG(0, cClusterSd);
  X.Raw[1] := CY + RandG(0, cClusterSd);
  Y.Fill(0);
  Y.Raw[Cls] := 1.0;
end;

procedure BuildData();
var
  I, Cls: integer;
  X, Y: TNNetVolume;
begin
  TrainX := TNNetVolumePairList.Create();
  TestX  := TNNetVolumePairList.Create();
  for I := 0 to cNTrain - 1 do
  begin
    Cls := I mod cClasses;
    MakeSample(X, Y, Cls);
    TrainX.Add(TNNetVolumePair.Create(X, Y));
  end;
  for I := 0 to cNTest - 1 do
  begin
    Cls := I mod cClasses;
    MakeSample(X, Y, Cls);
    TestX.Add(TNNetVolumePair.Create(X, Y));
  end;
end;

// Flip the label of exactly ONE training point. We pick a point of class 0
// that sits near the class-1 cluster boundary so its corrupted gradient bites.
procedure PlantMislabel();
var
  I, Best: integer;
  CX, CY, D, BestD: TNeuralFloat;
  P: TNNetVolumePair;
begin
  Best := -1;
  BestD := 1e30;
  ClassCentre(1, CX, CY); // class-1 centre
  for I := 0 to TrainX.Count - 1 do
  begin
    if TrainX[I].O.GetClass() <> 0 then Continue; // only class-0 points
    D := Sqr(TrainX[I].I.Raw[0] - CX) + Sqr(TrainX[I].I.Raw[1] - CY);
    if D < BestD then begin BestD := D; Best := I; end;
  end;
  MislabelIdx := Best;
  MislabelTrueCls := 0;
  P := TrainX[Best];
  // Flip its one-hot label to the WRONG class (1).
  P.O.Fill(0);
  P.O.Raw[1] := 1.0;
end;

procedure BuildNet();
begin
  NN := TNNet.Create();
  NN.AddLayer(TNNetInput.Create(cInDim, 1, 1));
  NN.AddLayer(TNNetFullConnectReLU.Create(cHidden));
  NN.AddLayer(TNNetFullConnectLinear.Create(cClasses));
  NN.AddLayer(TNNetSoftMax.Create());
  NN.SetLearningRate(cLR, 0.9);
  NN.InitWeights();
end;

procedure Train();
var
  Ep, B, Idx, Correct: integer;
  P: TNNetVolumePair;
begin
  for Ep := 1 to cEpochs do
  begin
    Correct := 0;
    for B := 0 to TrainX.Count - 1 do
    begin
      Idx := Random(TrainX.Count);
      P := TrainX[Idx];
      NN.Compute(P.I);
      if NN.GetLastLayer.Output.GetClass() = P.O.GetClass() then Inc(Correct);
      NN.Backpropagate(P.O);
    end;
    if (Ep = 1) or (Ep mod 10 = 0) or (Ep = cEpochs) then
      WriteLn(Format('  epoch %3d  train-acc=%.3f',
        [Ep, Correct / TrainX.Count]));
  end;
end;

// Returns the flattened, LR-normalised per-sample weight gradient for one
// labelled example, computed on the FROZEN net (no UpdateWeights).
// SetBatchUpdate(true) must already be active.
procedure SampleGradient(P: TNNetVolumePair; var G: TFloatArr);
var
  LayerIdx, NeuronIdx, K, Pos: integer;
  Layer: TNNetLayer;
  Neuron: TNNetNeuron;
  LR: TNeuralFloat;
begin
  NN.ClearDeltas();
  NN.Compute(P.I);
  NN.Backpropagate(P.O);
  Pos := 0;
  for LayerIdx := 0 to NN.GetLastLayerIdx() do
  begin
    Layer := NN.Layers[LayerIdx];
    if Layer.Neurons.Count = 0 then Continue;
    LR := Layer.LearningRate;
    if LR <= 0 then LR := 1.0;
    for NeuronIdx := 0 to Layer.Neurons.Count - 1 do
    begin
      Neuron := Layer.Neurons[NeuronIdx];
      // Per-sample weight gradient lands in Delta = -LR*grad under
      // SetBatchUpdate(true); divide back out by LR. (Neuron.Delta is the
      // public read-out; the bias delta is private and a single scalar per
      // neuron, so it is omitted — weight gradients alone rank TracIn fine.)
      for K := 0 to Neuron.Delta.Size - 1 do
      begin
        G[Pos] := Neuron.Delta.FData[K] / LR;
        Inc(Pos);
      end;
    end;
  end;
end;

// Count the flattened-gradient length once.
function CountGradLen(): integer;
var
  LayerIdx, NeuronIdx: integer;
  Layer: TNNetLayer;
  N: integer;
begin
  N := 0;
  for LayerIdx := 0 to NN.GetLastLayerIdx() do
  begin
    Layer := NN.Layers[LayerIdx];
    if Layer.Neurons.Count = 0 then Continue;
    for NeuronIdx := 0 to Layer.Neurons.Count - 1 do
      N := N + Layer.Neurons[NeuronIdx].Delta.Size; // weights only (no bias)
  end;
  Result := N;
end;

// Format a float with an explicit leading sign (FPC's Format has no '+' flag).
function Signed(V: TNeuralFloat): string;
begin
  if V >= 0 then Result := '+' + Format('%.3e', [V])
  else Result := Format('%.3e', [V]);
end;

function Dot(const A, B: TFloatArr): TNeuralFloat;
var
  I: integer;
  S: TNeuralFloat;
begin
  S := 0;
  for I := 0 to GradLen - 1 do S := S + A[I] * B[I];
  Result := S;
end;

// Picks a test point that the mislabel corrupts: a class-1 test point near the
// boundary (so its prediction is the kind the wrong class-1-labelled point
// pushes against). Returns its index in TestX.
function PickCorruptedTestPoint(): integer;
var
  I, Best: integer;
  CX, CY, D, BestD: TNeuralFloat;
begin
  Best := -1;
  BestD := 1e30;
  ClassCentre(0, CX, CY); // toward class-0 centre = boundary side
  for I := 0 to TestX.Count - 1 do
  begin
    if TestX[I].O.GetClass() <> 1 then Continue;
    D := Sqr(TestX[I].I.Raw[0] - CX) + Sqr(TestX[I].I.Raw[1] - CY);
    if D < BestD then begin BestD := D; Best := I; end;
  end;
  Result := Best;
end;

var
  Influence: TFloatArr;
  Order: array of integer;
  TestGrad: TFloatArr;
  TmpGrad: TFloatArr;
  TestIdx, I, J, Tmp, RankOfMislabel: integer;
  Passed: boolean;
  Tag: string;
begin
  RandSeed := 2026;
  WriteLn('TracInfluence demo: TracInLast training-data attribution on a tiny ' +
    '2-D 2-class blob task.');
  WriteLn(Format('  N_train=%d  N_test=%d  (cost: O(N_train) backward passes ' +
    'per test point)', [cNTrain, cNTest]));
  WriteLn;

  BuildData();
  PlantMislabel();
  WriteLn(Format('Planted ONE mislabel: train point #%d (true class %d) ' +
    'relabelled as class 1.',
    [MislabelIdx, MislabelTrueCls]));
  WriteLn(Format('  its features: (%.3f, %.3f)',
    [TrainX[MislabelIdx].I.Raw[0], TrainX[MislabelIdx].I.Raw[1]]));
  WriteLn;

  BuildNet();
  WriteLn('Training tiny MLP (Input -> FC+ReLU -> FC -> SoftMax)...');
  Train();
  WriteLn;

  // Freeze the net: per-sample gradients land in Delta, weights never stepped.
  NN.SetBatchUpdate(true);
  GradLen := CountGradLen();
  SetLength(TestGrad, GradLen);
  SetLength(TmpGrad, GradLen);
  WriteLn(Format('Flattened per-sample gradient length: %d params.', [GradLen]));
  WriteLn;

  // ---- TracIn for one corrupted test point ----
  TestIdx := PickCorruptedTestPoint();
  NN.Compute(TestX[TestIdx].I);
  WriteLn(Format('Test point #%d: true class %d, predicted class %d, ' +
    'features (%.3f, %.3f).',
    [TestIdx, TestX[TestIdx].O.GetClass(),
     NN.GetLastLayer.Output.GetClass(),
     TestX[TestIdx].I.Raw[0], TestX[TestIdx].I.Raw[1]]));

  SampleGradient(TestX[TestIdx], TestGrad);

  SetLength(Influence, TrainX.Count);
  for I := 0 to TrainX.Count - 1 do
  begin
    SampleGradient(TrainX[I], TmpGrad);
    Influence[I] := Dot(TmpGrad, TestGrad);
  end;

  // Rank training points by influence (descending).
  SetLength(Order, TrainX.Count);
  for I := 0 to TrainX.Count - 1 do Order[I] := I;
  for I := 0 to TrainX.Count - 2 do
    for J := 0 to TrainX.Count - 2 - I do
      if Influence[Order[J]] < Influence[Order[J + 1]] then
      begin
        Tmp := Order[J]; Order[J] := Order[J + 1]; Order[J + 1] := Tmp;
      end;

  WriteLn;
  WriteLn(Format('Top-%d PROPONENTS (most POSITIVE influence):', [cTopK]));
  for I := 0 to cTopK - 1 do
    WriteLn(Format('  #%-4d  influence=%s  (label class %d)',
      [Order[I], Signed(Influence[Order[I]]), TrainX[Order[I]].O.GetClass()]));

  WriteLn;
  WriteLn(Format('Top-%d OPPONENTS (most NEGATIVE influence):', [cTopK]));
  for I := 0 to cTopK - 1 do
  begin
    J := TrainX.Count - 1 - I;
    if Order[J] = MislabelIdx then Tag := '   <== PLANTED MISLABEL'
    else Tag := '';
    WriteLn(Format('  #%-4d  influence=%s  (label class %d)%s',
      [Order[J], Signed(Influence[Order[J]]), TrainX[Order[J]].O.GetClass(),
       Tag]));
  end;

  // ---- Graded self-check: the planted mislabel must be a top-K opponent. ----
  RankOfMislabel := -1;
  for I := 0 to TrainX.Count - 1 do
    if Order[I] = MislabelIdx then RankOfMislabel := I;
  // Rank from the OPPONENT end (0 = most negative).
  RankOfMislabel := (TrainX.Count - 1) - RankOfMislabel;

  WriteLn;
  WriteLn(Format('Planted mislabel #%d influence=%s, opponent-rank=%d ' +
    '(0 = most negative).',
    [MislabelIdx, Signed(Influence[MislabelIdx]), RankOfMislabel]));

  Passed := (RankOfMislabel < cTopK) and (Influence[MislabelIdx] < 0);
  WriteLn;
  if Passed then
    WriteLn(Format('PASS: planted mislabel is among the top-%d most-negative ' +
      'opponents (TracIn surfaced it).', [cTopK]))
  else
    WriteLn(Format('FAIL: planted mislabel NOT in top-%d opponents ' +
      '(opponent-rank=%d, influence=%s).',
      [cTopK, RankOfMislabel, Signed(Influence[MislabelIdx])]));

  // Cleanup.
  NN.Free;
  TrainX.Free;
  TestX.Free;

  if not Passed then Halt(1);
end.
