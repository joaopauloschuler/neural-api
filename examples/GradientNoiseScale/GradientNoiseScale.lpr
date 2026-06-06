program GradientNoiseScale;
(*
GradientNoiseScale: builds a small softmax classifier, then prints
TNNet.GradientNoiseScaleReport on two labelled synthetic probe batches so the
contrast is visible: a CLEAN, linearly-separable 3-cluster set whose per-sample
gradients agree (high per-parameter SNR, a tiny simple noise scale B_simple, so
small batches are already near-optimal), versus a deliberately label-noised /
overlapping set whose gradients scatter (low SNR, a large B_simple, so a bigger
batch genuinely buys faster convergence).

The report runs, per labelled sample, one forward + one backward on a FROZEN net
(ClearDeltas before each, never UpdateWeights), snapshots that sample's full
flattened per-parameter weight-gradient vector g_i, forms the mean gradient
g_bar and the per-parameter gradient variance across samples, and reports the
per-parameter gradient SNR |g_bar_k|/(std_k+eps), the McCandlish simple noise
scale B_simple = tr(Sigma)/||g_bar||^2 (the "critical batch size"), the
effective-batch noise curve noise(B) = B_simple/B, and per-layer signal- vs
noise-dominated flags. The trained weights are never stepped.

To let the prediction be eyeballed against reality, the demo also prints a quick
EMPIRICAL batch-size sweep on the noisy problem: it trains fresh copies of the
net at a few batch sizes and shows how the final loss falls as the batch grows
toward (and past) the predicted B_simple.

Pure CPU, no dataset download, well under a minute.

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
  // what scatters their gradients (low SNR, large noise scale).
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
  // weights to measure noise at).
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

  // One mean-squared-error number over a frozen probe batch (forward only).
  function BatchLoss(NN: TNNet; Probes: TNNetVolumePairList): TNeuralFloat;
  var
    I, K: integer;
    Pair: TNNetVolumePair;
    Sum, D: TNeuralFloat;
    Outp: TNNetVolume;
  begin
    Sum := 0;
    for I := 0 to Probes.Count - 1 do
    begin
      Pair := Probes[I];
      NN.Compute(Pair.I);
      Outp := NN.GetLastLayer().Output;
      for K := 0 to Outp.Size - 1 do
      begin
        D := Outp.FData[K] - Pair.O.FData[K];
        Sum := Sum + D * D;
      end;
    end;
    Result := Sum / Probes.Count;
  end;

  // A quick EMPIRICAL batch-size sweep on the noisy problem: train a fresh net
  // from the same seed at a given batch size for a fixed compute budget, then
  // report the final probe loss. Larger batches should keep helping while the
  // problem is noise-limited (B << B_simple) and stop helping once it isn't.
  function SweepFinalLoss(BatchSize: integer;
    EvalBatch: TNNetVolumePairList): TNeuralFloat;
  const
    cTotalSteps = 6000;      // fixed gradient-evaluation budget
  var
    NN: TNNet;
    Step, B, C: integer;
    X, Y: TNNetVolume;
  begin
    BuildNet(NN, 0.05);
    try
      NN.SetBatchUpdate(true);
      Step := 0;
      while Step < cTotalSteps do
      begin
        NN.ClearDeltas();
        for B := 1 to BatchSize do
        begin
          C := Random(cClasses);
          MakeNoisySample(X, Y, C);
          try
            NN.Compute(X);
            NN.Backpropagate(Y);
          finally
            X.Free;
            Y.Free;
          end;
          Inc(Step);
        end;
        NN.UpdateWeights();
      end;
      Result := BatchLoss(NN, EvalBatch);
    finally
      NN.Free;
    end;
  end;

var
  NN: TNNet;
  CleanProbes, NoisyProbes: TNNetVolumePairList;
  BSizes: array[0..3] of integer = (1, 4, 16, 64);
  I: integer;
  Loss: TNeuralFloat;
begin
  RandSeed := 2026;
  WriteLn('GradientNoiseScale demo: tiny 3-class softmax MLP on a synthetic ' +
    '2-D problem.');
  WriteLn('Predicts the batch-size sweep from the gradient signal-to-noise ' +
    'ratio (McCandlish et al. 2018).');

  // Train one net on the clean separable problem so the weights are sensible.
  BuildNet(NN, 0.05);
  try
    TrainOnce(NN, cEpochs);

    BuildBatch(CleanProbes, True);
    BuildBatch(NoisyProbes, False);
    try
      WriteLn;
      WriteLn(StringOfChar('=', 100));
      WriteLn('RUN 1: CLEAN linearly-separable batch. ' +
        'Expect high SNR and a TINY B_simple (small batches near-optimal).');
      WriteLn(StringOfChar('=', 100));
      Write(TNNet.GradientNoiseScaleReport(NN, CleanProbes));

      WriteLn;
      WriteLn(StringOfChar('=', 100));
      WriteLn('RUN 2: NOISY / overlapping + label-corrupted batch. ' +
        'Expect low SNR and a LARGE B_simple (bigger batches genuinely help).');
      WriteLn(StringOfChar('=', 100));
      Write(TNNet.GradientNoiseScaleReport(NN, NoisyProbes));

      WriteLn;
      WriteLn(StringOfChar('=', 100));
      WriteLn('RUN 3: same noisy batch, restricted to the classifier head ' +
        '(layer 2) only. Head and stem usually have different noise scales.');
      WriteLn(StringOfChar('=', 100));
      Write(TNNet.GradientNoiseScaleReport(NN, NoisyProbes, True, 2));

      WriteLn;
      WriteLn(StringOfChar('=', 100));
      WriteLn('RUN 4: EMPIRICAL batch-size sweep on the noisy problem ' +
        '(fixed compute budget). Eyeball it against the predicted B_simple.');
      WriteLn(StringOfChar('=', 100));
      WriteLn('  batch   final probe loss');
      for I := 0 to High(BSizes) do
      begin
        Loss := SweepFinalLoss(BSizes[I], NoisyProbes);
        WriteLn(Format('  %5d   %12.6f', [BSizes[I], Loss]));
      end;

      WriteLn;
      WriteLn(
        'Read it as: B_simple from RUN 2 is the predicted critical batch ' +
        'size. While the empirical batch (RUN 4) is below it the loss keeps ' +
        'dropping as the batch grows (noise-limited); once the batch passes ' +
        'B_simple the loss flattens (signal-limited - bigger batches waste ' +
        'compute). The clean batch (RUN 1) has a tiny B_simple, so even a ' +
        'batch of 1 is already near-optimal there.');
    finally
      CleanProbes.Free;
      NoisyProbes.Free;
    end;
  finally
    NN.Free;
  end;
end.
