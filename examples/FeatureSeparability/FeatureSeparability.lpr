program FeatureSeparability;
(*
FeatureSeparability: builds a tiny 3-class softmax MLP, trains it briefly on a
synthetic 2-D Gaussian-blob problem, then prints TNNet.FeatureSeparabilityReport
- a label-aware class-GEOMETRY / Neural-Collapse diagnostic (Papyan, Han &
Donoho 2020). The report answers "how tightly does each layer cluster the
samples of a class, and how far apart are the classes?" by computing, per
trainable layer, the Fisher-style scatter decomposition tr(S_w) / tr(S_b), the
Fisher ratio tr(S_b)/tr(S_w), the mean silhouette coefficient and the class-mean
pairwise-cosine matrix with a simplex-ETF (NC2) check - all with NO classifier
fit, purely forward.

Two variants run back to back so the contrast is visible:
  (i)  WELL-SEPARATED blobs (the three class centres are far apart relative to
       their spread) - high Fisher ratio, silhouette near 1, off-diagonal
       class-mean cosines approaching the simplex-ETF target -1/(K-1)=-0.5;
  (ii) OVERLAPPING blobs (the centres are close, the spread is large) - low
       Fisher ratio, low silhouette, class means poorly separated.
In both the Fisher ratio and silhouette climb toward the penultimate layer (the
trained ReLU stack progressively tightens the per-class clusters), but the
well-separated variant climbs higher.

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
  cHidden   = 10;
  cClasses  = 3;
  cEpochs   = 80;
  cProbeCnt = 120;  // probe batch (must be a multiple of cClasses to stay balanced)

  // Builds a tiny MLP classifier:
  //   Input -> FC+ReLU -> FC+ReLU -> FC+ReLU -> FC -> SoftMax
  // The stacked ReLU blocks give several trainable layers whose class geometry
  // the report inspects, so the depth-wise climb in separability is visible.
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

  // Synthetic 3-class 2-D Gaussian-blob problem. Each class is an isotropic
  // blob centred at one vertex of an equilateral triangle of radius Sep, with
  // per-blob Gaussian-ish spread Spread. The remaining input coordinates are
  // pure noise (distractors). Small Sep / large Spread => overlapping classes.
  procedure MakeSample(out X, Y: TNNetVolume; Cls: integer;
    Sep, Spread: TNeuralFloat);
  var
    I: integer;
    Angle, Cx, Cy: TNeuralFloat;
  begin
    X := TNNetVolume.Create(cInDim, 1, 1);
    Y := TNNetVolume.Create(cClasses, 1, 1);
    Y.Fill(0);
    Y.Raw[Cls] := 1.0;
    // distractor coordinates: pure noise.
    for I := 0 to cInDim - 1 do
      X.Raw[I] := (Random - 0.5) * 2.0 * Spread;
    // class centre on an equilateral triangle.
    Angle := (2.0 * Pi * Cls) / cClasses;
    Cx := Sep * Cos(Angle);
    Cy := Sep * Sin(Angle);
    X.Raw[0] := Cx + (Random - 0.5) * 2.0 * Spread;
    X.Raw[1] := Cy + (Random - 0.5) * 2.0 * Spread;
  end;

  // Builds a CLASS-BALANCED probe batch (round-robin over classes) so the
  // scatter-decomposition identity tr(Stot)=tr(Sw)+tr(Sb) is exact.
  procedure BuildBatch(out Batch: TNNetVolumePairList; Count: integer;
    Sep, Spread: TNeuralFloat);
  var
    K: integer;
    X, Y: TNNetVolume;
  begin
    Batch := TNNetVolumePairList.Create();
    for K := 0 to Count - 1 do
    begin
      MakeSample(X, Y, K mod cClasses, Sep, Spread);
      Batch.Add(TNNetVolumePair.Create(X, Y));
    end;
  end;

  procedure TrainOnce(NN: TNNet; Epochs: integer; Sep, Spread: TNeuralFloat);
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
        MakeSample(X, Y, Cls, Sep, Spread);
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

  // One full variant: build, train and report on its own balanced probe batch.
  procedure RunVariant(const Title: string; Sep, Spread: TNeuralFloat);
  var
    NN: TNNet;
    Probes: TNNetVolumePairList;
  begin
    RandSeed := 2026;
    BuildBatch(Probes, cProbeCnt, Sep, Spread);
    BuildNet(NN, 0.01);
    try
      WriteLn;
      WriteLn(StringOfChar('=', 100));
      WriteLn(Title);
      WriteLn(StringOfChar('=', 100));
      WriteLn(Format('blob separation=%.2f, spread=%.2f. Training for %d epochs...',
        [Sep, Spread, cEpochs]));
      TrainOnce(NN, cEpochs, Sep, Spread);
      WriteLn;
      Write(TNNet.FeatureSeparabilityReport(NN, Probes, cClasses));
    finally
      NN.Free;
      Probes.Free;
    end;
  end;

begin
  WriteLn('FeatureSeparabilityReport demo: tiny 3-class softmax MLP on synthetic ' +
    '2-D Gaussian blobs.');
  WriteLn('Per trainable layer the report computes the Fisher scatter ' +
    'decomposition tr(Sb)/tr(Sw), the mean silhouette and the class-mean ' +
    'cosine matrix - a fit-free Neural-Collapse geometry probe.');
  WriteLn('Simplex-ETF target off-diagonal class-mean cosine = -1/(K-1) = ',
    Format('%.4f', [-1.0 / (cClasses - 1)]), '.');

  // (i) well-separated blobs: centres far apart, tight spread.
  RunVariant('RUN 1: WELL-SEPARATED blobs (high Fisher ratio expected).',
    2.2, 0.30);

  // (ii) overlapping blobs: centres close, large spread.
  RunVariant('RUN 2: OVERLAPPING blobs (low Fisher ratio expected).',
    0.5, 0.90);

  WriteLn;
  WriteLn(
    'Read it as: tr(Sw) is within-class cluster tightness (NC1, smaller=' +
    'tighter), tr(Sb) is the spread of the class means, and the Fisher ratio ' +
    'tr(Sb)/tr(Sw) summarises cluster cleanliness (higher=cleaner; S flag when ' +
    '>=1, R flag when classes overlap). Both runs climb toward the penultimate ' +
    'layer, but RUN 1 reaches a much higher Fisher ratio and silhouette and its ' +
    'final-layer off-diagonal class-mean cosines sit closer to the simplex-ETF ' +
    'target than RUN 2. The scatter identity tr(Stot)=tr(Sw)+tr(Sb) holds ' +
    '(balanced batch) as a built-in faithfulness check.');
end.
