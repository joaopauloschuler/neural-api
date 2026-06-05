program LinearProbeReport;
(*
LinearProbeReport: builds a small softmax classifier, then prints
TNNet.LinearProbeReport on a labelled synthetic probe batch (i) for a
freshly-initialised network and (ii) for the same architecture after a short
training run, so the contrast is visible. The synthetic task (an XOR-quadrant +
concentric-rings 4-class problem) is deliberately NOT linearly separable from
the raw input, so a linear probe on the input layer sits near the 1/NumClasses
random baseline in both runs. At fresh init the untrained random layers then
scramble that signal and probe accuracy DEGRADES toward the head; after training
the probe accuracy is preserved and CLIMBS with depth, holding high all the way
to the head - the "where does the model become a classifier?" contrast.

For every intermediate layer the report fits a CLOSED-FORM ridge-regularised
linear probe W = (X^T X + Lambda*I)^-1 X^T Y on that layer's flat activation
tensor (a self-contained Double-precision Gauss-Jordan solve - no SGD loop, no
backward pass) and reports per-layer top-1 probe accuracy, a held-out probe
accuracy (the train/val gap flags overfit probes), the one-hot regression MSE,
the per-layer accuracy delta, an ASCII bar chart of probe accuracy across depth,
and collapse / saturation / near-random flags. The backbone is pure forward-only
and never modified.

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
  cHidden   = 8;
  cClasses  = 4;
  cEpochs   = 120;
  cProbeCnt = 120;  // probe (train-the-probe) batch
  cValCnt   = 120;  // held-out batch for the probe's val accuracy
  cBlob     = 1.4;  // distance of each blob from the origin
  cNoise    = 0.35; // per-blob Gaussian-ish spread

  // Builds a tiny MLP classifier:
  //   Input -> FC+ReLU -> FC+ReLU -> FC+ReLU -> FC -> SoftMax
  // The stacked ReLU blocks give several intermediate layers to probe so the
  // depth-wise climb in linear separability is visible.
  procedure BuildNet(out NN: TNNet; LR: TNeuralFloat);
  begin
    NN := TNNet.Create();
    NN.AddLayer(TNNetInput.Create(cInDim, 1, 1));
    NN.AddLayer(TNNetFullConnectReLU.Create(cHidden));
    NN.AddLayer(TNNetFullConnectReLU.Create(cHidden));
    NN.AddLayer(TNNetFullConnectReLU.Create(cHidden));
    NN.AddLayer(TNNetFullConnectLinear.Create(cClasses));
    NN.AddLayer(TNNetSoftMax.Create());
    NN.SetLearningRate(LR, 0.9);
    NN.InitWeights();
  end;

  // Synthetic multi-class problem that is DELIBERATELY NOT linearly separable
  // from the raw input. The first two coordinates carry an XOR-style structure:
  // each class occupies TWO diagonally-opposite blobs in the (x0, x1) plane
  //
  //     class 0: top-right + bottom-left      (x0,x1 same sign)
  //     class 1: top-left  + bottom-right     (x0,x1 opposite sign)
  //     class 2: a centre ring (small radius), independent of quadrant
  //     class 3: an outer ring (large radius)
  //
  // No single hyperplane separates these classes (the diagonal blobs of one
  // class are split by any line through the data), so a LINEAR probe on the raw
  // input - or on a freshly-initialised net's activations - sits near the 1/4
  // random baseline. A trained ReLU stack folds the input into a representation
  // where the classes become linearly separable, so the probe accuracy climbs
  // with depth. The remaining coordinates are pure noise (distractors).
  procedure MakeSample(out X, Y: TNNetVolume; Cls: integer);
  var
    I, SignA, SignB: integer;
    R, Theta, Radius: TNeuralFloat;
  begin
    X := TNNetVolume.Create(cInDim, 1, 1);
    Y := TNNetVolume.Create(cClasses, 1, 1);
    Y.Fill(0);
    Y.Raw[Cls] := 1.0;
    // distractor coordinates: pure noise.
    for I := 0 to cInDim - 1 do
      X.Raw[I] := (Random - 0.5) * 2.0 * cNoise;
    case Cls of
      0, 1:
        begin
          // XOR quadrant blobs in (x0, x1).
          if Random(2) = 0 then SignA := 1 else SignA := -1;
          if Cls = 0 then SignB := SignA            // same sign  -> class 0
          else SignB := -SignA;                     // opposite   -> class 1
          X.Raw[0] := SignA * cBlob + (Random - 0.5) * 2.0 * cNoise;
          X.Raw[1] := SignB * cBlob + (Random - 0.5) * 2.0 * cNoise;
        end;
      2, 3:
        begin
          // concentric rings (radius carries the label, angle is uniform).
          if Cls = 2 then Radius := 0.45 * cBlob else Radius := 1.7 * cBlob;
          Theta := Random * 2.0 * Pi;
          R := Radius + (Random - 0.5) * cNoise;
          X.Raw[0] := R * Cos(Theta);
          X.Raw[1] := R * Sin(Theta);
        end;
    end;
  end;

  procedure BuildBatch(out Batch: TNNetVolumePairList; Count: integer);
  var
    K: integer;
    X, Y: TNNetVolume;
  begin
    Batch := TNNetVolumePairList.Create();
    for K := 0 to Count - 1 do
    begin
      MakeSample(X, Y, K mod cClasses);
      Batch.Add(TNNetVolumePair.Create(X, Y));
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
  Probes, ValProbes: TNNetVolumePairList;
begin
  RandSeed := 2026;
  BuildBatch(Probes, cProbeCnt);
  BuildBatch(ValProbes, cValCnt);
  try
    WriteLn('LinearProbeReport demo: tiny 4-class softmax MLP on a synthetic ' +
      'XOR + concentric-rings problem (not linearly separable from the raw ' +
      'input).');
    WriteLn('Per layer a closed-form ridge linear probe W=(X^T X+lambda I)^-1 ' +
      'X^T Y is fit on that layer''s activations.');
    WriteLn('Random baseline = 1/NumClasses = ', Format('%.4f', [1.0 / cClasses]),
      '.');

    // ---- (i) fresh init: probe acc near random at every depth ----
    BuildNet(NN, 0.01);
    try
      WriteLn;
      WriteLn(StringOfChar('=', 100));
      WriteLn('RUN 1: freshly-initialised network (no training). ' +
        'Untrained layers scramble the signal: probe acc degrades toward the ' +
        'head.');
      WriteLn(StringOfChar('=', 100));
      Write(TNNet.LinearProbeReport(NN, Probes, ValProbes));
    finally
      NN.Free;
    end;

    // ---- (ii) after a short training run: depth-wise climb + saturation ----
    RandSeed := 2026;
    BuildNet(NN, 0.01);
    try
      WriteLn;
      WriteLn(StringOfChar('=', 100));
      WriteLn('RUN 2: same architecture after a short training run. ' +
        'Expect a depth-wise climb with a saturation knee.');
      WriteLn(StringOfChar('=', 100));
      WriteLn('Training for ', cEpochs, ' epochs...');
      TrainOnce(NN, cEpochs);
      WriteLn;
      Write(TNNet.LinearProbeReport(NN, Probes, ValProbes));
    finally
      NN.Free;
    end;

    WriteLn;
    WriteLn(
      'Read it as: the per-layer probe accuracy answers "where does the model ' +
      'become a classifier?". The raw input is not linearly separable so the ' +
      'input-layer probe is near the random baseline in both runs. At fresh ' +
      'init the untrained layers then scramble the signal and probe acc ' +
      'degrades toward the head (C collapse flags, R near-random at the head); ' +
      'after training the accuracy is preserved and climbs with depth and the ' +
      'S flag marks the shallowest layer already within 1 point of the final ' +
      'layer - the natural transfer-learning cut point. The ProbeAcc-vs-ValAcc ' +
      'gap flags probes that overfit the probe batch.');
  finally
    Probes.Free;
    ValProbes.Free;
  end;
end.
