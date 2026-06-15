program MagnitudePruneFineTune;
(*
MagnitudePruneFineTune: demonstrates PERSISTENT magnitude-pruning masks and the
accuracy RECOVERY you get by fine-tuning after pruning (the classic
prune-then-retrain loop, torch.nn.utils.prune style).

Unlike MagnitudePruning (which prunes, measures, and RESTORES the weights),
this example KEEPS the pruned weights at zero:
  1. train a small 3-class classifier on a self-contained synthetic problem;
  2. measure held-out top-1 accuracy (DENSE baseline);
  3. NN.PruneWeightsByMagnitude(s) zeros the smallest s% of |weights| ACROSS
     the network and installs a persistent mask;
  4. measure accuracy RIGHT AFTER pruning (typically a drop -- capacity removed);
  5. fine-tune for a few epochs. The mask is re-enforced after every weight
     update, so the pruned weights stay EXACTLY zero (they never grow back) and
     the surviving weights adapt to compensate;
  6. measure accuracy AFTER fine-tuning -- it climbs back toward (often up to)
     the dense baseline, at the reported sparsity.

We sweep a couple of target sparsities so the before/after recovery is visible
side by side. Synthetic, pure CPU, well under a minute.

Copyright (C) 2026 Joao Paulo Schwarz Schuler

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
any later version.

Coded by Claude (AI).
*)

{$mode objfpc}{$H+}

uses {$IFDEF UNIX} cthreads, {$ENDIF}
  Classes, SysUtils,
  neuralnetwork,
  neuralvolume;

const
  cInDim    = 24;   // input feature dimension
  cClasses  = 4;    // number of classes
  cProbeCnt = 300;  // held-out probe samples
  cTrainCnt = 500;  // synthetic training samples per epoch
  cEpochs   = 40;   // initial training epochs
  cFineTune = 30;   // fine-tuning epochs after pruning

var
  Proto: array of TNeuralFloat;

// Synthetic 3-class problem: each class has a random prototype; the label is
// argmax over the class scores (proto_c . x), consistent with the features.
procedure MakeSample(out X: TNNetVolume; out Cls: integer);
var
  I, C: integer;
  Score, Best: TNeuralFloat;
begin
  Cls := Random(cClasses);
  X := TNNetVolume.Create(cInDim, 1, 1);
  for I := 0 to cInDim - 1 do
    X.Raw[I] := Proto[Cls * cInDim + I] + (Random - 0.5) * 2.6;
  // Non-linear label: argmax over class scores of a SQUARED (XOR-like) feature
  // interaction, so the task genuinely needs hidden capacity and gets harder
  // under heavy pruning.
  Best := -1e30;
  for C := 0 to cClasses - 1 do
  begin
    Score := 0;
    for I := 0 to cInDim - 1 do
      Score := Score + Proto[C * cInDim + I] *
        (X.Raw[I] * X.Raw[(I + 1) mod cInDim]);
    if Score > Best then begin Best := Score; Cls := C; end;
  end;
end;

procedure BuildProbes(Probes: TNNetVolumeList; Labels: TNNetVolumeList);
var
  K, Cls: integer;
  X, L: TNNetVolume;
begin
  for K := 0 to cProbeCnt - 1 do
  begin
    MakeSample(X, Cls);
    Probes.Add(X);
    L := TNNetVolume.Create(cClasses, 1, 1);
    L.Raw[Cls] := 1.0;
    Labels.Add(L);
  end;
end;

function ProbeAccuracy(NN: TNNet; Probes, Labels: TNNetVolumeList): TNeuralFloat;
var
  K, Correct: integer;
begin
  Correct := 0;
  for K := 0 to Probes.Count - 1 do
  begin
    NN.Compute(Probes[K]);
    if NN.GetLastLayer.Output.GetClass() = Labels[K].GetClass() then
      Inc(Correct);
  end;
  Result := Correct / Probes.Count;
end;

procedure TrainEpochs(NN: TNNet; Epochs: integer);
var
  Ep, Step, Cls: integer;
  X, Yt: TNNetVolume;
begin
  for Ep := 1 to Epochs do
    for Step := 1 to cTrainCnt do
    begin
      MakeSample(X, Cls);
      Yt := TNNetVolume.Create(cClasses, 1, 1);
      Yt.Raw[Cls] := 1.0;
      try
        NN.Compute(X);
        NN.Backpropagate(Yt);
      finally
        X.Free;
        Yt.Free;
      end;
    end;
end;

// FPC's Format has no '+' flag; render a signed delta by hand.
function SignedPts(V: TNeuralFloat): string;
begin
  if V >= 0 then Result := '+' + Format('%.2f', [V])
  else Result := Format('%.2f', [V]);
end;

procedure RunCase(Sparsity: TNeuralFloat; Probes, Labels: TNNetVolumeList);
var
  NN: TNNet;
  DenseAcc, PrunedAcc, RecoveredAcc, RealSparsity: TNeuralFloat;
  Pruned: integer;
begin
  WriteLn(StringOfChar('-', 78));
  WriteLn(Format('Target sparsity: %.0f%%', [Sparsity * 100.0]));
  WriteLn(StringOfChar('-', 78));

  NN := TNNet.Create();
  try
    NN.AddLayer(TNNetInput.Create(cInDim, 1, 1));
    NN.AddLayer(TNNetFullConnectReLU.Create(32));
    NN.AddLayer(TNNetFullConnectReLU.Create(32));
    NN.AddLayer(TNNetFullConnectLinear.Create(cClasses));
    NN.AddLayer(TNNetSoftMax.Create());
    NN.SetLearningRate(0.02, 0.9);
    NN.InitWeights();

    // 1-2. dense training + baseline accuracy.
    TrainEpochs(NN, cEpochs);
    DenseAcc := ProbeAccuracy(NN, Probes, Labels);
    WriteLn(Format('  [1] dense baseline accuracy        : %6.2f%%',
      [DenseAcc * 100.0]));

    // 3. prune with a PERSISTENT mask.
    Pruned := NN.PruneWeightsByMagnitude(Sparsity, {PerLayer=}False);
    RealSparsity := NN.GetPruneSparsity();
    WriteLn(Format('  [2] pruned %d weights -> realised sparsity %.2f%% ' +
      '(persistent mask installed)', [Pruned, RealSparsity * 100.0]));

    // 4. accuracy right after pruning (no retraining yet).
    PrunedAcc := ProbeAccuracy(NN, Probes, Labels);
    WriteLn(Format('  [3] accuracy right after pruning   : %6.2f%%  ' +
      '(change %s pts vs dense)',
      [PrunedAcc * 100.0, SignedPts((PrunedAcc - DenseAcc) * 100.0)]));

    // 5. fine-tune WITH the mask enforced (pruned weights stay at zero).
    TrainEpochs(NN, cFineTune);

    // 6. accuracy after fine-tuning + confirm the mask survived.
    RecoveredAcc := ProbeAccuracy(NN, Probes, Labels);
    WriteLn(Format('  [4] accuracy after fine-tuning     : %6.2f%%  ' +
      '(recovered %s pts)',
      [RecoveredAcc * 100.0, SignedPts((RecoveredAcc - PrunedAcc) * 100.0)]));
    WriteLn(Format('      mask still holds %d weights at zero (sparsity %.2f%%).',
      [NN.CountPrunedWeights(), NN.GetPruneSparsity() * 100.0]));
    WriteLn(Format('  SUMMARY @ %.0f%% sparsity:  dense %.2f%%  ->  pruned %.2f%%  ' +
      '->  fine-tuned %.2f%%',
      [Sparsity * 100.0, DenseAcc * 100.0, PrunedAcc * 100.0,
       RecoveredAcc * 100.0]));
  finally
    NN.Free;
  end;
  WriteLn;
end;

var
  Probes, Labels: TNNetVolumeList;
  I: integer;
begin
  RandSeed := 2026;

  WriteLn('MagnitudePruneFineTune: PERSISTENT magnitude pruning + ' +
    'fine-tune-after-prune recovery.');
  WriteLn('Train a small classifier, prune the smallest-|w| weights to a ' +
    'target sparsity with a');
  WriteLn('mask that STAYS enforced, then fine-tune and watch accuracy climb ' +
    'back -- the pruned');
  WriteLn('weights never grow back because the mask is re-applied after every ' +
    'weight update.');
  WriteLn;

  SetLength(Proto, cClasses * cInDim);
  for I := 0 to Length(Proto) - 1 do Proto[I] := (Random - 0.5) * 2.0;

  // One held-out probe set shared by every sparsity case (same problem).
  Probes := TNNetVolumeList.Create(True);
  Labels := TNNetVolumeList.Create(True);
  try
    BuildProbes(Probes, Labels);

    RunCase(0.50, Probes, Labels);
    RunCase(0.80, Probes, Labels);
    RunCase(0.90, Probes, Labels);

    WriteLn('Read the rows: [3] usually dips below the [1] dense baseline ' +
      '(capacity removed),');
    WriteLn('then [4] recovers toward it after fine-tuning. The deeper the ' +
      'sparsity, the larger');
    WriteLn('the initial dip and the more recovery fine-tuning has to do -- ' +
      'but the mask keeps the');
    WriteLn('network exactly that sparse throughout.');
  finally
    Labels.Free;
    Probes.Free;
  end;
end.
