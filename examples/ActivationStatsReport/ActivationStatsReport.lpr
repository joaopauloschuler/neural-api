program ActivationStatsReport;
(*
ActivationStatsReport: builds a small MLP, makes a tiny probe batch, then prints
TNNet.ActivationStatsReport (i) on the fresh-init network and (ii) on the same
architecture after a short training run on a trivial synthetic task, so a
reviewer can eyeball how training reshapes the per-layer activation
distribution.

The report is pure forward-only: it runs one NN.Compute per probe sample and
walks every layer's Output volume, printing a per-layer table of
mean / std / min / max / |median| / |skew| / kurtosis plus saturation,
negative and near-zero fractions and a compact 16-bin ASCII histogram, then a
flag list (near-collapsed / saturating layers) and a 10-bin ASCII histogram of
per-layer std across the network.

Pure CPU, well under a minute.

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
  cInDim    = 8;
  cHidden   = 32;
  cEpochs   = 40;
  cBatch    = 32;
  cProbeCnt = 48;

  // Small MLP with a mix of bounded (tanh) and unbounded (linear) activations
  // so the report's saturation / spread statistics have something to chew on.
  procedure BuildMLP(out NN: TNNet);
  begin
    NN := TNNet.Create();
    NN.AddLayer(TNNetInput.Create(cInDim, 1, 1));
    NN.AddLayer(TNNetFullConnectLinear.Create(cHidden));
    NN.AddLayer(TNNetReLU.Create());
    NN.AddLayer(TNNetFullConnectLinear.Create(cHidden));
    NN.AddLayer(TNNetHyperbolicTangent.Create());
    NN.AddLayer(TNNetFullConnectLinear.Create(1));
    NN.SetLearningRate(0.01, 0.9);
  end;

  // Target: y = ||x|| (hypotenuse). Smooth scalar regression.
  procedure MakePair(out X, Y: TNNetVolume);
  var
    I: integer;
    Acc: TNeuralFloat;
  begin
    X := TNNetVolume.Create(cInDim, 1, 1);
    Y := TNNetVolume.Create(1, 1, 1);
    Acc := 0;
    for I := 0 to cInDim - 1 do
    begin
      X.Raw[I] := (Random - 0.5) * 2.0;
      Acc := Acc + X.Raw[I] * X.Raw[I];
    end;
    Y.Raw[0] := Sqrt(Acc);
  end;

  procedure BuildProbes(out Probes: TNNetVolumeList);
  var
    K, I: integer;
    V: TNNetVolume;
  begin
    Probes := TNNetVolumeList.Create(True);
    for K := 0 to cProbeCnt - 1 do
    begin
      V := TNNetVolume.Create(cInDim, 1, 1);
      for I := 0 to cInDim - 1 do
        V.Raw[I] := (Random - 0.5) * 2.0;
      Probes.Add(V);
    end;
  end;

  procedure TrainOnce(NN: TNNet; Epochs: integer);
  var
    Ep, B: integer;
    X, Yt: TNNetVolume;
  begin
    for Ep := 1 to Epochs do
    begin
      for B := 1 to cBatch do
      begin
        MakePair(X, Yt);
        try
          NN.Compute(X);
          NN.Backpropagate(Yt);
        finally
          X.Free;
          Yt.Free;
        end;
      end;
    end;
  end;

var
  NN: TNNet;
  Probes: TNNetVolumeList;
begin
  RandSeed := 2026;

  BuildProbes(Probes);
  try
    WriteLn('ActivationStatsReport demo: small MLP (ReLU + tanh) on y=||x||.');

    // ---- (i) fresh-init network ----
    BuildMLP(NN);
    try
      NN.InitWeights();
      WriteLn;
      WriteLn(StringOfChar('=', 92));
      WriteLn('(i) FRESH-INIT network (before any training).');
      WriteLn(StringOfChar('=', 92));
      Write(TNNet.ActivationStatsReport(NN, Probes));

      // ---- (ii) same network after a short training run ----
      WriteLn;
      WriteLn('Training for ', cEpochs, ' epochs of batch size ', cBatch, '...');
      TrainOnce(NN, cEpochs);
      WriteLn;
      WriteLn(StringOfChar('=', 92));
      WriteLn('(ii) AFTER ', cEpochs, ' training epochs (same architecture).');
      WriteLn(StringOfChar('=', 92));
      Write(TNNet.ActivationStatsReport(NN, Probes));
    finally
      NN.Free;
    end;

    WriteLn;
    WriteLn(
      'Compare (i) vs (ii): training reshapes the per-layer activation ' +
      'distribution - the mean/std columns and per-layer-std histogram shift ' +
      'as the network learns to fit y=||x||.');
  finally
    Probes.Free;
  end;
end.
