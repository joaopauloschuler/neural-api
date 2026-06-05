program LogitLens;
(*
LogitLens: builds a small constant-width softmax classifier, then prints
TNNet.LogitLensReport on an UNLABELLED synthetic probe batch for (i) a
freshly-initialised network and (ii) the same architecture after a short
training run, so the contrast is visible in ONE run.

The logit-lens (nostalgebraist 2020; cf. "Tuned Lens", Belrose et al. 2023)
asks: "if we read out the prediction at THIS layer using the network's OWN
trained output head, what would it already say?". For every intermediate layer
whose flat activation is shape-compatible with the head's expected input it
SPLICES that activation into the head's input slot, recomputes ONLY the head
layers and reads the resulting "lens distribution" p_L. No probe is fitted - the
lens reuses the model's own trained head (ZERO fitted parameters), which is what
distinguishes it from LinearProbeReport (which FITS a fresh ridge probe per
layer).

The classifier BODY is kept at a CONSTANT WIDTH so every hidden layer feeds the
head at the same size and is therefore lens-compatible; the raw input layer has
a different width and shows up explicitly as a SKIPPED layer (the honest
width-compatibility constraint of the classic lens).

Contrast in a single run (the lens at the head input is always the exact
reference, so KL there is 0 and agreement 1.0 by construction):
  fresh init  -> the untrained body carries no usable signal: lens confidence
                 stays near 1/NumClasses and lens entropy stays high/flat;
  trained     -> lens confidence and sharpness (lower entropy) climb steadily
                 with depth as the readout commits to an answer.
The exact per-layer numbers (agreement / KL / crystallization) depend on the
seed and the tiny synthetic task; the robust, seed-independent signals are the
entropy/confidence sharpening with depth and the built-in correctness checks.

The report also prints two BUILT-IN correctness checks as PASS/FAIL:
  - the lens AT the head input (no substitution) reproduces p_final EXACTLY
    (agreement 1.0, KL 0);
  - a single-layer head degenerates to the trivial profile.

Pure CPU, no dataset download, well under a minute.

Copyright (C) 2026 Joao Paulo Schwarz Schuler

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
any later version.

Coded by Claude (AI).
*)

{$mode objfpc}{$H+}

uses {$IFDEF UNIX} {$IFDEF UseCThreads}
  cthreads, {$ENDIF} {$ENDIF}
  Classes, SysUtils, Math,
  neuralnetwork,
  neuralvolume;

const
  cInDim    = 6;
  cWidth    = 10;   // CONSTANT body width => every hidden layer is lens-compatible
  cClasses  = 4;
  cEpochs   = 140;
  cProbeCnt = 160;  // unlabelled probe batch
  cBlob     = 1.4;  // distance of each blob from the origin
  cNoise    = 0.35; // per-blob Gaussian-ish spread

  // Builds a tiny constant-width MLP classifier:
  //   Input(cInDim) -> FC+ReLU(cWidth) x4 -> FC(cClasses) -> SoftMax
  // The stacked equal-width ReLU blocks give several intermediate layers all
  // feeding the head at the SAME size, so they are lens-compatible; the input
  // layer (width cInDim) is shape-incompatible and is reported as SKIPPED.
  procedure BuildNet(out NN: TNNet; LR: TNeuralFloat);
  begin
    NN := TNNet.Create();
    NN.AddLayer(TNNetInput.Create(cInDim, 1, 1));
    NN.AddLayer(TNNetFullConnectReLU.Create(cWidth));
    NN.AddLayer(TNNetFullConnectReLU.Create(cWidth));
    NN.AddLayer(TNNetFullConnectReLU.Create(cWidth));
    NN.AddLayer(TNNetFullConnectReLU.Create(cWidth));
    NN.AddLayer(TNNetFullConnectLinear.Create(cClasses));
    NN.AddLayer(TNNetSoftMax.Create());
    NN.SetLearningRate(LR, 0.9);
    NN.InitWeights();
  end;

  // Synthetic multi-class problem (XOR-quadrant + concentric rings), the same
  // not-linearly-separable structure used by the LinearProbeReport example. The
  // first two coordinates carry the label; the rest are noise distractors.
  procedure MakeSample(out X, Y: TNNetVolume; Cls: integer);
  var
    I, SignA, SignB: integer;
    R, Theta, Radius: TNeuralFloat;
  begin
    X := TNNetVolume.Create(cInDim, 1, 1);
    Y := TNNetVolume.Create(cClasses, 1, 1);
    Y.Fill(0);
    Y.Raw[Cls] := 1.0;
    for I := 0 to cInDim - 1 do
      X.Raw[I] := (Random - 0.5) * 2.0 * cNoise;
    case Cls of
      0, 1:
        begin
          if Random(2) = 0 then SignA := 1 else SignA := -1;
          if Cls = 0 then SignB := SignA
          else SignB := -SignA;
          X.Raw[0] := SignA * cBlob + (Random - 0.5) * 2.0 * cNoise;
          X.Raw[1] := SignB * cBlob + (Random - 0.5) * 2.0 * cNoise;
        end;
      2, 3:
        begin
          if Cls = 2 then Radius := 0.45 * cBlob else Radius := 1.7 * cBlob;
          Theta := Random * 2.0 * Pi;
          R := Radius + (Random - 0.5) * cNoise;
          X.Raw[0] := R * Cos(Theta);
          X.Raw[1] := R * Sin(Theta);
        end;
    end;
  end;

  // Unlabelled probe batch: the lens needs inputs only (it reads the model's
  // own predicted argmax as the reference, not a ground-truth label).
  procedure BuildProbes(out Probes: TNNetVolumeList; Count: integer);
  var
    K: integer;
    X, Y: TNNetVolume;
  begin
    Probes := TNNetVolumeList.Create(True);
    for K := 0 to Count - 1 do
    begin
      MakeSample(X, Y, K mod cClasses);
      Y.Free;             // labels not used by the lens
      Probes.Add(X);
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
  Probes: TNNetVolumeList;
begin
  RandSeed := 2026;
  BuildProbes(Probes, cProbeCnt);
  try
    WriteLn('LogitLens demo: tiny constant-width 4-class softmax MLP on a ' +
      'synthetic XOR + concentric-rings problem.');
    WriteLn('Per layer the model''s OWN trained head is re-applied to that ' +
      'layer''s activation (the logit lens) - ZERO fitted parameters.');

    // ---- (i) fresh init: lens agrees with final only at the very last layer ----
    BuildNet(NN, 0.01);
    try
      WriteLn;
      WriteLn(StringOfChar('=', 100));
      WriteLn('RUN 1: freshly-initialised network (no training). ' +
        'Expect lens confidence near 1/NumClasses and high, flat entropy.');
      WriteLn(StringOfChar('=', 100));
      Write(TNNet.LogitLensReport(NN, Probes));
    finally
      NN.Free;
    end;

    // ---- (ii) after a short training run: agreement climbs, KL falls ----
    RandSeed := 2026;
    BuildNet(NN, 0.01);
    try
      WriteLn;
      WriteLn(StringOfChar('=', 100));
      WriteLn('RUN 2: same architecture after a short training run. ' +
        'Expect lens confidence rising and entropy falling toward the head.');
      WriteLn(StringOfChar('=', 100));
      WriteLn('Training for ', cEpochs, ' epochs...');
      TrainOnce(NN, cEpochs);
      WriteLn;
      Write(TNNet.LogitLensReport(NN, Probes));
    finally
      NN.Free;
    end;

    // ---- (iii) degenerate single-layer-head sanity (explicit HeadStartIdx) ----
    RandSeed := 2026;
    BuildNet(NN, 0.01);
    try
      WriteLn;
      WriteLn(StringOfChar('=', 100));
      WriteLn('RUN 3: forcing HeadStartIdx to the LAST layer => single-layer ' +
        'head; the lens degenerates to "everything resolves at the last layer".');
      WriteLn(StringOfChar('=', 100));
      Write(TNNet.LogitLensReport(NN, Probes, NN.GetLastLayerIdx()));
    finally
      NN.Free;
    end;

    WriteLn;
    WriteLn(
      'Read it as: the per-layer agreement answers "when does the model''s ' +
      'running self-decoded belief already match its final answer?". The lens ' +
      'at the head input is the exact reference (agreement 1.0, KL 0). At fresh ' +
      'init the untrained body carries no usable signal so the lens stays near ' +
      '1/NumClasses confidence with high, flat entropy; after training the lens ' +
      'confidence rises and the entropy falls steadily toward the head as the ' +
      'readout commits to an answer. The input layer (width <> head input) is ' +
      'reported as SKIPPED. The correctness check (lens at the head input ' +
      'reproduces p_final exactly) must print PASS in every run.');
  finally
    Probes.Free;
  end;
end.
