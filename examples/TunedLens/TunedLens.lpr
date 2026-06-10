program TunedLens;
(*
TunedLens: builds the SAME small constant-width softmax classifier used by the
LogitLens example, trains it briefly, then prints TNNet.TunedLensReport AND
TNNet.LogitLensReport on the SAME UNLABELLED synthetic probe batch so the two
lenses are directly comparable in ONE run.

The logit lens (nostalgebraist 2020) splices a raw hidden activation straight
into the model's OWN frozen head. The TUNED lens (Belrose et al. 2023,
"Eliciting Latent Predictions with the Tuned Lens") first passes that activation
through a small per-layer LEARNED AFFINE translator (one TNNetFullConnectLinear
of the head-input width) that is TRAINED to map the layer's residual state into
the final-layer basis BEFORE the frozen head decodes it - correcting the
representation drift that makes the raw logit lens biased / mis-calibrated at
early depths. The trunk and head are FROZEN; only the per-layer translators are
fit, by minimising each layer's KL to the model's OWN final output distribution
on the unlabelled probe batch (distillation-to-self; no ground-truth labels).

The headline Belrose result, visible side by side in the report's KL-to-final
columns: the TUNED curve commits EARLIER and tracks the final answer more
faithfully (lower KL-to-final, more monotone) than the raw logit lens. The
report also prints three BUILT-IN correctness checks as PASS/FAIL:
  1. an UNTRAINED (identity-seeded) translator does NO better than the raw lens
     (its mean KL-to-final ties the logit lens - no free lunch before fitting);
  2. fitting the translators LOWERS the mean KL-to-final;
  3. at the head input the translator collapses to the identity, so there
     tuned == logit == final (max |dp| ~ 0).

The classifier BODY is kept at a CONSTANT WIDTH so every hidden layer feeds the
head at the same size and is therefore lens-compatible (and the translators can
map to the head-input basis); the raw input layer has a different width and is
reported as a SKIPPED layer.

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
  cInDim     = 6;
  cWidth     = 10;   // CONSTANT body width => every hidden layer is lens-compatible
  cClasses   = 4;
  cEpochs    = 140;
  cProbeCnt  = 160;  // unlabelled probe batch (shared by BOTH lenses)
  cBlob      = 1.4;  // distance of each blob from the origin
  cNoise     = 0.35; // per-blob Gaussian-ish spread
  cTrainIter = 800;  // translator fit iterations per lens-compatible layer
  cLensLR    = 0.005;

  // Tiny constant-width MLP classifier (identical to the LogitLens example):
  //   Input(cInDim) -> FC+ReLU(cWidth) x4 -> FC(cClasses) -> SoftMax
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

  // Synthetic multi-class problem (XOR-quadrant + concentric rings) - the same
  // not-linearly-separable structure used by the LogitLens example.
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

  // Unlabelled probe batch: both lenses read the model's own predicted
  // distribution as the reference, not a ground-truth label.
  procedure BuildProbes(out Probes: TNNetVolumeList; Count: integer);
  var
    K: integer;
    X, Y: TNNetVolume;
  begin
    Probes := TNNetVolumeList.Create(True);
    for K := 0 to Count - 1 do
    begin
      MakeSample(X, Y, K mod cClasses);
      Y.Free;             // labels not used by the lenses
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
    WriteLn('TunedLens demo: tiny constant-width 4-class softmax MLP on a ' +
      'synthetic XOR + concentric-rings problem.');
    WriteLn('LOGIT lens = the model''s OWN frozen head re-applied at each depth ' +
      '(zero fitted params). TUNED lens = a per-layer LEARNED affine translator ' +
      'maps each depth into the final basis FIRST, then the frozen head decodes.');

    RandSeed := 2026;
    BuildNet(NN, 0.01);
    try
      WriteLn;
      WriteLn('Training the classifier for ', cEpochs, ' epochs (trunk + head ' +
        'are then FROZEN for both lenses)...');
      TrainOnce(NN, cEpochs);

      WriteLn;
      WriteLn(StringOfChar('=', 100));
      WriteLn('RAW LOGIT LENS (reference; zero fitted parameters):');
      WriteLn(StringOfChar('=', 100));
      Write(TNNet.LogitLensReport(NN, Probes));

      WriteLn;
      WriteLn(StringOfChar('=', 100));
      WriteLn('TUNED LENS (per-layer learned translator, fit on the SAME ' +
        'unlabelled probe batch):');
      WriteLn(StringOfChar('=', 100));
      Write(TNNet.TunedLensReport(NN, Probes, -1, cTrainIter, cLensLR));
    finally
      NN.Free;
    end;

    WriteLn;
    WriteLn(
      'Read it as: both lenses ask "what would the model already predict if we ' +
      'decoded THIS layer?". The logit lens reuses the frozen head as-is; the ' +
      'tuned lens first learns a per-layer affine correction for representation ' +
      'drift. Compare the KL-to-final columns: the TUNED curve sits LOWER at the ' +
      'early/middle layers (it commits earlier and tracks the final answer more ' +
      'faithfully) - the headline Belrose result. The three built-in checks must ' +
      'all print PASS: an untrained translator ties the raw lens, fitting lowers ' +
      'the mean KL-to-final, and at the head input the translator is the identity ' +
      'so tuned == logit == final.');
  finally
    Probes.Free;
  end;
end.
