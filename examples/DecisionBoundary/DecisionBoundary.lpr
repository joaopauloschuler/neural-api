program DecisionBoundary;
(*
DecisionBoundary: visualizes the LEARNED FUNCTION of a 2-input classifier head
over its whole input plane using TNNet.DecisionBoundaryReport. It sweeps a grid
over the input domain, runs one forward pass per grid cell, and prints (a) an
ASCII class map (argmax glyph per cell), (b) a confidence-shaded overlay, (c) a
single boundary-length scalar (how convoluted the learned boundary is), and (d)
a true-class probe overlay so misclassified points stand out.

Two runs are shown, all on synthetic data (no download), well under a minute:

  1. A clean 3-cluster 2D Gaussian problem. The class map is printed BEFORE
     training (a near-constant single-class plane at init) and AFTER a short
     training run (clean separated regions, short boundary length).

  2. A deliberately-overfit run on a tiny noisy "two-moons"-style 2-class set
     using an oversized MLP and far too many epochs. The class map develops
     wiggly islands and the boundary-length scalar in (c) jumps up — the
     overfitting pathology visible both as art and as a single number.

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
  cGauss3Classes  = 3;
  cTrainPerClass  = 200;
  cEpochs         = 60;
  cLearningRate   = 0.05;

  // Three 2D Gaussian clusters (same generator as ConfusionMatrixReport /
  // MarginReport) so the carved-up regions come out cleanly separated.
  cCenters: array[0..2, 0..1] of TNeuralFloat =
    ((-1.2, -1.2),
     ( 1.2, -1.0),
     ( 0.0,  1.3));
  cSigma: TNeuralFloat = 0.55;

// Why: Box-Muller gives N(0,1) samples without pulling in a dependency.
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

procedure MakeGaussSample(ClassId: integer; out X, Y: TNNetVolume);
begin
  X := TNNetVolume.Create(2, 1, 1);
  Y := TNNetVolume.Create(cGauss3Classes, 1, 1);
  X.FData[0] := cCenters[ClassId][0] + RandomGauss() * cSigma;
  X.FData[1] := cCenters[ClassId][1] + RandomGauss() * cSigma;
  Y.Fill(0);
  Y.FData[ClassId] := 1.0;
end;

procedure BuildGaussSet(out Pairs: TNNetVolumePairList; PerClass: integer);
var
  C, I: integer;
  X, Y: TNNetVolume;
begin
  Pairs := TNNetVolumePairList.Create();
  for C := 0 to cGauss3Classes - 1 do
    for I := 1 to PerClass do
    begin
      MakeGaussSample(C, X, Y);
      Pairs.Add(TNNetVolumePair.Create(X, Y));
    end;
end;

procedure TrainClassifier(NN: TNNet; Pairs: TNNetVolumePairList;
  Epochs: integer; LR: TNeuralFloat);
var
  Epoch, I: integer;
  Pair: TNNetVolumePair;
  Loss, Diff: TNeuralFloat;
begin
  NN.SetLearningRate(LR, 0.9);
  for Epoch := 1 to Epochs do
  begin
    Loss := 0;
    for I := 0 to Pairs.Count - 1 do
    begin
      Pair := Pairs[I];
      NN.Compute(Pair.I);
      NN.Backpropagate(Pair.O);
      Diff := -Ln(Max(1e-9,
        NN.GetLastLayer().Output.FData[Pair.O.GetClass()]));
      Loss := Loss + Diff;
    end;
    if (Epoch mod 20 = 0) or (Epoch = 1) then
      WriteLn('  epoch ', Epoch:4, '  mean_nll=', (Loss / Pairs.Count):8:5);
  end;
end;

// ---- run 1: clean 3-cluster Gaussian, before/after training ----
procedure RunCleanDemo();
var
  NN: TNNet;
  TrainSet: TNNetVolumePairList;
begin
  WriteLn('############################################################');
  WriteLn('# RUN 1: clean 3-cluster Gaussian (before vs after training)');
  WriteLn('############################################################');
  RandSeed := 12345;
  NN := TNNet.Create();
  NN.AddLayer(TNNetInput.Create(2, 1, 1));
  NN.AddLayer(TNNetFullConnectReLU.Create(16));
  NN.AddLayer(TNNetFullConnectReLU.Create(16));
  NN.AddLayer(TNNetFullConnectLinear.Create(cGauss3Classes));
  NN.AddLayer(TNNetSoftMax.Create());
  NN.InitWeights();
  // Shrink the fresh weights toward zero so the head is near-constant at init:
  // the softmax then collapses to a single dominant class over the whole plane,
  // giving the expected single-class "BEFORE" map. We re-init before training.
  NN.MulWeights(0.001);
  BuildGaussSet(TrainSet, cTrainPerClass);
  try
    WriteLn;
    WriteLn('--- BEFORE training (expect a near-constant single-class plane) ---');
    WriteLn(TNNet.DecisionBoundaryReport(NN, TrainSet));

    // Re-initialise to a normal scale for the actual training run.
    NN.InitWeights();
    WriteLn('Training for ', cEpochs, ' epochs on ', TrainSet.Count,
      ' samples...');
    TrainClassifier(NN, TrainSet, cEpochs, cLearningRate);
    WriteLn;

    WriteLn('--- AFTER training (expect clean separated regions, ' +
      'short boundary) ---');
    WriteLn(TNNet.DecisionBoundaryReport(NN, TrainSet));
  finally
    TrainSet.Free;
    NN.Free;
  end;
end;

// ---- run 2: tiny noisy two-moons, deliberate overfit ----
procedure MakeMoonSample(out X, Y: TNNetVolume);
var
  Cls: integer;
  T, R: TNeuralFloat;
begin
  X := TNNetVolume.Create(2, 1, 1);
  Y := TNNetVolume.Create(2, 1, 1);
  Cls := Random(2);
  T := Random * Pi;
  R := 1.0;
  if Cls = 0 then
  begin
    X.FData[0] := R * Cos(T) - 0.5;
    X.FData[1] := R * Sin(T) - 0.25;
  end
  else
  begin
    X.FData[0] := R * Cos(T) + 0.5;
    X.FData[1] := -R * Sin(T) + 0.25;
  end;
  // heavy label/coordinate noise so an oversized net can memorise wiggles.
  X.FData[0] := X.FData[0] + RandomGauss() * 0.35;
  X.FData[1] := X.FData[1] + RandomGauss() * 0.35;
  Y.Fill(0);
  Y.FData[Cls] := 1.0;
end;

procedure BuildMoonSet(out Pairs: TNNetVolumePairList; N: integer);
var
  I: integer;
  X, Y: TNNetVolume;
begin
  Pairs := TNNetVolumePairList.Create();
  for I := 1 to N do
  begin
    MakeMoonSample(X, Y);
    Pairs.Add(TNNetVolumePair.Create(X, Y));
  end;
end;

// Builds a 2-class MLP of the given hidden width / depth.
function BuildMoonNet(Width, Depth: integer): TNNet;
var
  L: integer;
begin
  Result := TNNet.Create();
  Result.AddLayer(TNNetInput.Create(2, 1, 1));
  for L := 1 to Depth do
    Result.AddLayer(TNNetFullConnectReLU.Create(Width));
  Result.AddLayer(TNNetFullConnectLinear.Create(2));
  Result.AddLayer(TNNetSoftMax.Create());
  Result.InitWeights();
end;

procedure RunOverfitDemo();
var
  Small, Big: TNNet;
  TrainSet: TNNetVolumePairList;
begin
  WriteLn;
  WriteLn('############################################################');
  WriteLn('# RUN 2: tiny noisy two-moons, well-fit vs deliberate overfit');
  WriteLn('#        (the overfit net should print wigglier art AND a');
  WriteLn('#         larger boundary-length scalar than the small net)');
  WriteLn('############################################################');
  RandSeed := 2024;
  // A tiny, heavily-noised two-moons training set shared by both nets.
  BuildMoonSet(TrainSet, 40);

  // A small, well-regularised-by-capacity net: short, smooth boundary.
  Small := BuildMoonNet(6, 1);
  // An oversized net trained far too long on the tiny set: memorised wiggles.
  Big := BuildMoonNet(64, 3);
  try
    WriteLn;
    WriteLn('-- small net (1x6 hidden), modest training --');
    TrainClassifier(Small, TrainSet, 80, 0.05);
    WriteLn;
    WriteLn('--- small-net decision boundary (expect a short boundary) ---');
    WriteLn(TNNet.DecisionBoundaryReport(Small, TrainSet));

    WriteLn('-- big net (3x64 hidden), 400 epochs (deliberate overfit) --');
    TrainClassifier(Big, TrainSet, 400, 0.05);
    WriteLn;
    WriteLn('--- overfit-net decision boundary (expect wiggly islands + a ' +
      'larger boundary length) ---');
    WriteLn(TNNet.DecisionBoundaryReport(Big, TrainSet));
  finally
    TrainSet.Free;
    Big.Free;
    Small.Free;
  end;
end;

begin
  RunCleanDemo();
  RunOverfitDemo();
end.
