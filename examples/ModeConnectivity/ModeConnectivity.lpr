program ModeConnectivity;
(*
ModeConnectivity: demonstrates TNNet.ModeConnectivityReport, the linear
mode-connectivity / loss-barrier diagnostic (Garipov et al. 2018; Frankle
et al. 2020).

It trains the SAME tiny MLP twice on a tiny synthetic 3-cluster 2D
classification task and prints the loss barrier between the two solutions
along the line theta(alpha) = (1-alpha)*A + alpha*B for alpha in [0,1]:

  RUN 1 (same basin):  both nets start from the SAME random init and only
                       differ in batch-shuffle order -> expect a LOW barrier
                       ("linearly connected", same loss basin).
  RUN 2 (two basins):  the two nets start from DIFFERENT random inits ->
                       expect a HIGHER barrier ("separated" / "weak barrier").

The contrast between the two barriers is the whole point of the demo. Pure
forward-only along the path; the live net's weights are restored exactly
afterwards. Self-contained synthetic data, runs in well under a minute on CPU.

Copyright (C) 2026 Joao Paulo Schwarz Schuler

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
any later version.
*)

{$mode objfpc}{$H+}

uses {$IFDEF UNIX} {$IFDEF UseCThreads}
  cthreads, {$ENDIF} {$ENDIF}
  Classes, SysUtils, Math,
  neuralnetwork,
  neuralvolume;

const
  cEpochs       = 60;
  cTrainPerCls  = 60;
  cProbePerCls  = 12;
  cLearningRate = 0.05;
  cClasses      = 3;
  Centers: array[0..2, 0..1] of TNeuralFloat =
    ((-2.0, -2.0), (2.0, 2.0), (2.0, -2.0));

  procedure BuildNet(out NN: TNNet);
  begin
    NN := TNNet.Create();
    NN.AddLayer(TNNetInput.Create(2, 1, 1));
    NN.AddLayer(TNNetFullConnectReLU.Create(12));
    NN.AddLayer(TNNetFullConnectReLU.Create(12));
    NN.AddLayer(TNNetFullConnectLinear.Create(cClasses));
    NN.SetLearningRate(cLearningRate, 0.9);
  end;

  // Deterministic sample for class C at index seed (so two nets can be fed
  // the SAME data, differing only in batch order).
  procedure MakeSample(C: integer; out X, Y: TNNetVolume);
  begin
    X := TNNetVolume.Create(2, 1, 1);
    Y := TNNetVolume.Create(cClasses, 1, 1);
    X.FData[0] := Centers[C][0] + (Random - 0.5) * 0.6;
    X.FData[1] := Centers[C][1] + (Random - 0.5) * 0.6;
    Y.Fill(0);
    Y.FData[C] := 1.0;
  end;

  procedure TrainNet(NN: TNNet);
  var
    Epoch, I, C: integer;
    X, Y: TNNetVolume;
  begin
    for Epoch := 1 to cEpochs do
      for I := 1 to cTrainPerCls do
        for C := 0 to cClasses - 1 do
        begin
          MakeSample(C, X, Y);
          try
            NN.Compute(X);
            NN.Backpropagate(Y);
          finally
            X.Free;
            Y.Free;
          end;
        end;
  end;

  procedure BuildProbes(out Probes: TNNetVolumePairList);
  var
    C, I: integer;
    X, Y: TNNetVolume;
  begin
    Probes := TNNetVolumePairList.Create();
    RandSeed := 777;
    for C := 0 to cClasses - 1 do
      for I := 1 to cProbePerCls do
      begin
        MakeSample(C, X, Y);
        Probes.Add(TNNetVolumePair.Create(X, Y));
      end;
  end;

var
  NNA, NNB: TNNet;
  Probes: TNNetVolumePairList;
  SnapB, Report: string;

begin
  WriteLn('=== ModeConnectivity demo: loss barrier between two solutions ===');
  WriteLn;

  BuildProbes(Probes);
  try
    // ---------------------------------------------------------------
    // RUN 1: SAME init, only batch-shuffle order differs -> same basin.
    // ---------------------------------------------------------------
    WriteLn('RUN 1 - SAME init (only batch order differs): expect a tiny ' +
      'absolute barrier (same basin).');
    RandSeed := 2024;
    BuildNet(NNA);
    BuildNet(NNB);
    // Why: copy A's freshly-initialised weights into B so both start IDENTICAL.
    NNB.LoadDataFromString(NNA.SaveDataToString());

    RandSeed := 11;  TrainNet(NNA);
    RandSeed := 22;  TrainNet(NNB);   // same data, different shuffle order

    SnapB := NNB.SaveDataToString();
    Report := TNNet.ModeConnectivityReport(NNA, SnapB, Probes, 10);
    WriteLn(Report);
    NNA.Free;
    NNB.Free;
    WriteLn;

    // ---------------------------------------------------------------
    // RUN 2: DIFFERENT random inits -> expect a higher barrier.
    // ---------------------------------------------------------------
    WriteLn('RUN 2 - DIFFERENT inits: expect a HIGHER barrier.');
    RandSeed := 101;  BuildNet(NNA);
    RandSeed := 999;  BuildNet(NNB);

    RandSeed := 11;  TrainNet(NNA);
    RandSeed := 22;  TrainNet(NNB);

    SnapB := NNB.SaveDataToString();
    Report := TNNet.ModeConnectivityReport(NNA, SnapB, Probes, 10);
    WriteLn(Report);
    NNA.Free;
    NNB.Free;
    WriteLn;

    // ---------------------------------------------------------------
    // CHECK: B := A collapses the curve to a flat zero-barrier line.
    // ---------------------------------------------------------------
    WriteLn('CHECK - B := A (self-connectivity): barrier must be ~0.');
    RandSeed := 303;  BuildNet(NNA);
    RandSeed := 11;   TrainNet(NNA);
    SnapB := NNA.SaveDataToString();   // B == A
    Report := TNNet.ModeConnectivityReport(NNA, SnapB, Probes, 8);
    WriteLn(Report);
    NNA.Free;
  finally
    Probes.Free;
  end;
end.
