program FisherImportance;
(*
FisherImportance: builds a small softmax classifier, then prints
TNNet.FisherImportanceReport on a labelled synthetic probe batch (i) for a
freshly-initialised network and (ii) for the same architecture after a short
training run, so the contrast is visible: at init the diagonal Fisher
information is diffuse and the per-layer mass is spread fairly evenly; after
training the mass redistributes toward the layers doing the discriminative work
and the per-parameter max Fisher and the heavy right tail of the log10(Fisher)
histogram both grow.

The report estimates, per trainable parameter, the diagonal (empirical) Fisher
information F[theta] = E_x[(d log p(y|x)/d theta)^2] by accumulating per-sample
squared parameter gradients over the probe batch (one forward + one backward
per sample), then aggregates per layer and reports the prune-/reuse-relevant
statistics. The network is frozen throughout — its trained weights are never
stepped.

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
  Classes, SysUtils, Math,
  neuralnetwork,
  neuralvolume;

const
  cInDim    = 6;
  cHidden   = 24;
  cClasses  = 4;
  cEpochs   = 60;
  cProbeCnt = 64;
  // A modest class overlap (small bumps + sizeable noise) keeps the trained
  // minimum away from the degenerate "everything saturates, all gradients
  // vanish" regime, so the trained net retains a heavy-tailed, concentrated
  // Fisher rather than a uniformly tiny one.
  cBump     = 0.9;
  cNoise    = 0.9;

  // Builds a tiny MLP classifier: Input -> FC+ReLU -> FC+ReLU -> FC -> SoftMax.
  procedure BuildNet(out NN: TNNet; LR: TNeuralFloat);
  begin
    NN := TNNet.Create();
    NN.AddLayer(TNNetInput.Create(cInDim, 1, 1));
    NN.AddLayer(TNNetFullConnectReLU.Create(cHidden));
    NN.AddLayer(TNNetFullConnectReLU.Create(cHidden));
    NN.AddLayer(TNNetFullConnectLinear.Create(cClasses));
    NN.AddLayer(TNNetSoftMax.Create());
    NN.SetLearningRate(LR, 0.9);
    NN.InitWeights();
  end;

  // Synthetic multi-class problem: each class has a distinct (overlapping) mean
  // vector so a small net learns it but never reaches a degenerate minimum.
  procedure MakeSample(out X, Y: TNNetVolume; Cls: integer);
  var
    I: integer;
  begin
    X := TNNetVolume.Create(cInDim, 1, 1);
    Y := TNNetVolume.Create(cClasses, 1, 1);
    Y.Fill(0);
    Y.Raw[Cls] := 1.0;
    for I := 0 to cInDim - 1 do
      X.Raw[I] := (Random - 0.5) * 2.0 * cNoise;
    // Class-conditional bump on two coordinates (overlapping classes).
    X.Raw[Cls mod cInDim] := X.Raw[Cls mod cInDim] + cBump;
    X.Raw[(Cls + 1) mod cInDim] := X.Raw[(Cls + 1) mod cInDim] - cBump;
  end;

  procedure BuildProbes(out Probes: TNNetVolumePairList);
  var
    K: integer;
    X, Y: TNNetVolume;
  begin
    Probes := TNNetVolumePairList.Create();
    for K := 0 to cProbeCnt - 1 do
    begin
      MakeSample(X, Y, K mod cClasses);
      Probes.Add(TNNetVolumePair.Create(X, Y));
    end;
  end;

  procedure TrainOnce(NN: TNNet; Epochs: integer);
  var
    Ep, B, Cls, Correct: integer;
    X, Y: TNNetVolume;
  begin
    for Ep := 1 to Epochs do
    begin
      Correct := 0;
      for B := 1 to 96 do
      begin
        Cls := Random(cClasses);
        MakeSample(X, Y, Cls);
        try
          NN.Compute(X);
          if NN.GetLastLayer.Output.GetClass() = Cls then Inc(Correct);
          NN.Backpropagate(Y);
        finally
          X.Free;
          Y.Free;
        end;
      end;
      if (Ep = 1) or (Ep mod 20 = 0) or (Ep = Epochs) then
        WriteLn(Format('  epoch %3d  train-acc=%.3f', [Ep, Correct / 96.0]));
    end;
  end;

var
  NN: TNNet;
  Probes: TNNetVolumePairList;
begin
  RandSeed := 2026;
  BuildProbes(Probes);
  try
    WriteLn('FisherImportance demo: tiny 4-class softmax MLP on a synthetic ' +
      'Gaussian problem.');
    WriteLn('Diagonal empirical Fisher = E_x[(d log p(y|x)/d theta)^2] per ' +
      'parameter.');

    // ---- (i) fresh init: diffuse, low Fisher ----
    BuildNet(NN, 0.01);
    try
      WriteLn;
      WriteLn(StringOfChar('=', 108));
      WriteLn('RUN 1: freshly-initialised network (no training). ' +
        'Expect diffuse, low-concentration Fisher.');
      WriteLn(StringOfChar('=', 108));
      Write(TNNet.FisherImportanceReport(NN, Probes));
    finally
      NN.Free;
    end;

    // ---- (ii) after a short training run: importance redistributes ----
    RandSeed := 2026;
    BuildNet(NN, 0.01);
    try
      WriteLn;
      WriteLn(StringOfChar('=', 108));
      WriteLn('RUN 2: same architecture after a short training run. ' +
        'Expect importance to redistribute and sharpen.');
      WriteLn(StringOfChar('=', 108));
      WriteLn('Training for ', cEpochs, ' epochs...');
      TrainOnce(NN, cEpochs);
      WriteLn;
      Write(TNNet.FisherImportanceReport(NN, Probes));
    finally
      NN.Free;
    end;

    WriteLn;
    WriteLn(
      'Read it as: a high per-layer "Share%" / "FisherMass" marks the ' +
      'layers you can least afford to prune; the "Zero%" column and the ' +
      'low-Fisher histogram tail mark free-to-prune parameters; the ' +
      'effective-parameter-count (participation ratio) is a one-number ' +
      'concentration proxy. At fresh init the Fisher mass is spread fairly ' +
      'evenly across layers; after training it redistributes toward the ' +
      'layers doing the discriminative work and the per-parameter max Fisher ' +
      'and heavy right tail of the log10 histogram both grow.');
  finally
    Probes.Free;
  end;
end.
