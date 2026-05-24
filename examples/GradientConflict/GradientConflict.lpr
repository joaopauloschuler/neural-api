program GradientConflict;
(*
GradientConflict: builds a small softmax classifier, then prints
TNNet.GradientConflictReport on two labelled synthetic probe batches so the
contrast is visible: a CLEAN, linearly-separable 3-cluster set whose per-sample
gradients pull the weights in compatible directions (a high within-class cosine
and an empty strong-conflict tail), versus a deliberately label-noised /
overlapping set whose gradients actively fight each other (a fat negative-cosine
tail near cos=-1 and a high strong-conflict fraction emerge).

The report runs, per labelled sample, one forward + one backward on a FROZEN net
(ClearDeltas before each, never UpdateWeights), snapshots that sample's full
flattened per-parameter weight-gradient vector g_i, and reports the pairwise
gradient cosine similarity cos(g_i,g_j) = <g_i,g_j>/(||g_i|| ||g_j||) across the
batch: a 10-bin histogram, the conflict fraction (share of pairs with cos < 0),
the mean/median cosine, the most-conflicting sample pair, and a per-class-pair
mean-cosine matrix. The trained weights are never stepped.

Pure CPU, no dataset download, well under a minute.

Copyright (C) 2026 Joao Paulo Schwarz Schuler

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
any later version.
*)

{$mode objfpc}{$H+}

uses {$IFDEF UNIX} {$IFDEF UseCThreads}
  cthreads, {$ENDIF} {$ENDIF}
  Classes, SysUtils,
  neuralnetwork,
  neuralvolume;

const
  cInDim    = 2;
  cHidden   = 12;
  cClasses  = 3;
  cEpochs   = 40;
  cPerClass = 24;            // probe samples per class
  // Three well-separated cluster centers (the clean problem).
  cCenters: array[0..2, 0..1] of TNeuralFloat =
    ((-2.0, -2.0), (2.0, 2.0), (2.0, -2.0));

  // Builds a tiny MLP classifier: Input -> FC+ReLU -> FC -> SoftMax.
  procedure BuildNet(out NN: TNNet; LR: TNeuralFloat);
  begin
    NN := TNNet.Create();
    NN.AddLayer(TNNetInput.Create(cInDim, 1, 1));
    NN.AddLayer(TNNetFullConnectReLU.Create(cHidden));
    NN.AddLayer(TNNetFullConnectLinear.Create(cClasses));
    NN.AddLayer(TNNetSoftMax.Create());
    NN.SetLearningRate(LR, 0.9);
    NN.InitWeights();
  end;

  // Clean linearly-separable sample: tight cluster around the class center.
  procedure MakeCleanSample(out X, Y: TNNetVolume; Cls: integer);
  begin
    X := TNNetVolume.Create(cInDim, 1, 1);
    Y := TNNetVolume.Create(cClasses, 1, 1);
    Y.Fill(0);
    Y.Raw[Cls] := 1.0;
    X.Raw[0] := cCenters[Cls][0] + (Random - 0.5) * 0.6;
    X.Raw[1] := cCenters[Cls][1] + (Random - 0.5) * 0.6;
  end;

  // Noisy / overlapping sample: heavy jitter that overlaps the other clusters,
  // and a 40% chance the label is corrupted to a random WRONG class - so two
  // samples sitting in the same region carry opposing targets, which is exactly
  // what makes their gradients oppose.
  procedure MakeNoisySample(out X, Y: TNNetVolume; Cls: integer);
  var
    TrueCls, LabelCls: integer;
  begin
    TrueCls := Cls;
    X := TNNetVolume.Create(cInDim, 1, 1);
    Y := TNNetVolume.Create(cClasses, 1, 1);
    X.Raw[0] := cCenters[TrueCls][0] + (Random - 0.5) * 5.0;
    X.Raw[1] := cCenters[TrueCls][1] + (Random - 0.5) * 5.0;
    LabelCls := TrueCls;
    if Random < 0.4 then
      LabelCls := (TrueCls + 1 + Random(cClasses - 1)) mod cClasses;
    Y.Fill(0);
    Y.Raw[LabelCls] := 1.0;
  end;

  procedure BuildBatch(out Probes: TNNetVolumePairList; Clean: boolean);
  var
    C, K: integer;
    X, Y: TNNetVolume;
  begin
    Probes := TNNetVolumePairList.Create();
    for C := 0 to cClasses - 1 do
      for K := 0 to cPerClass - 1 do
      begin
        if Clean then MakeCleanSample(X, Y, C)
        else MakeNoisySample(X, Y, C);
        Probes.Add(TNNetVolumePair.Create(X, Y));
      end;
  end;

  // Trains the net on the clean problem (a sensible non-degenerate set of
  // weights to measure conflict at).
  procedure TrainOnce(NN: TNNet; Epochs: integer);
  var
    Ep, B, Cls: integer;
    X, Y: TNNetVolume;
  begin
    for Ep := 1 to Epochs do
      for B := 1 to 90 do
      begin
        Cls := Random(cClasses);
        MakeCleanSample(X, Y, Cls);
        try
          NN.Compute(X);
          NN.Backpropagate(Y);
        finally
          X.Free;
          Y.Free;
        end;
      end;
  end;

var
  NN: TNNet;
  CleanProbes, NoisyProbes: TNNetVolumePairList;
begin
  RandSeed := 2026;
  WriteLn('GradientConflict demo: tiny 3-class softmax MLP on a synthetic ' +
    '2-D problem.');
  WriteLn('Pairwise gradient cosine cos(g_i,g_j) = <g_i,g_j>/(||g_i|| ' +
    '||g_j||) across a frozen batch.');

  // Train one net on the clean separable problem so the weights are sensible.
  BuildNet(NN, 0.05);
  try
    TrainOnce(NN, cEpochs);

    BuildBatch(CleanProbes, True);
    BuildBatch(NoisyProbes, False);
    try
      WriteLn;
      WriteLn(StringOfChar('=', 100));
      WriteLn('RUN 1: CLEAN linearly-separable 3-cluster batch. ' +
        'Expect a high within-class cosine and an EMPTY strong-conflict tail.');
      WriteLn(StringOfChar('=', 100));
      Write(TNNet.GradientConflictReport(NN, CleanProbes));

      WriteLn;
      WriteLn(StringOfChar('=', 100));
      WriteLn('RUN 2: NOISY / overlapping + label-corrupted batch. ' +
        'Expect a fat NEGATIVE-cosine tail and a high strong-conflict fraction.');
      WriteLn(StringOfChar('=', 100));
      Write(TNNet.GradientConflictReport(NN, NoisyProbes));

      WriteLn;
      WriteLn(StringOfChar('=', 100));
      WriteLn('RUN 3: same noisy batch, restricted to the classifier head ' +
        '(layer 2) only. The conflict is often concentrated there.');
      WriteLn(StringOfChar('=', 100));
      Write(TNNet.GradientConflictReport(NN, NoisyProbes, True, 2));

      WriteLn;
      WriteLn(
        'Read it as: cross-class pairs of a softmax head sit just below zero ' +
        'by construction, so watch the STRONG-conflict fraction (cos < -0.5), ' +
        'not the raw cos<0 share. In RUN 1 (clean) the within-class diagonal ' +
        'is ~+1 and the strong-conflict tail is empty; in RUN 2 (noisy) ' +
        'overlapping regions carry contradictory labels, so a fat tail near ' +
        'cos=-1 appears and the strong-conflict fraction climbs sharply. The ' +
        'per-class-pair matrix points at WHICH class pairs fight; restricting ' +
        'to one layer (RUN 3) localizes it.');
    finally
      CleanProbes.Free;
      NoisyProbes.Free;
    end;
  finally
    NN.Free;
  end;
end.
