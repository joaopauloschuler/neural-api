program NeuronCorrelationReport;
(*
NeuronCorrelationReport: builds a small MLP, prints
TNNet.NeuronCorrelationReport on a probe batch BEFORE training (fresh init,
neurons start independent -> near-zero off-diagonal correlations) and AGAIN
after a short training run on a simple synthetic task (neurons specialise and
partially align -> a heavier |rho| tail and a lower effective neuron count).
Printing both makes the redundancy contrast visible.

Pure CPU, well under a minute.

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
  cInDim    = 32;
  cHidden   = 32;
  cEpochs   = 400;
  cBatch    = 48;
  cProbeCnt = 128;
  cLR       = 0.05;

var
  // Fixed random teacher direction for the target feature (filled in main).
  gW: array[0..cInDim - 1] of TNeuralFloat;

  procedure BuildMLP(out NN: TNNet);
  begin
    NN := TNNet.Create();
    NN.AddLayer(TNNetInput.Create(cInDim, 1, 1));
    NN.AddLayer(TNNetFullConnectReLU.Create(cHidden));
    NN.AddLayer(TNNetFullConnectReLU.Create(cHidden));
    NN.AddLayer(TNNetFullConnectLinear.Create(1));
    NN.SetLearningRate(cLR, 0.9);
  end;

  // Target: a SINGLE rectified-linear teacher feature, y = max(0, gW . x),
  // where gW is one fixed random direction. The whole label is explained by
  // that one ReLU feature, so gradient descent drives a large fraction of the
  // 32 hidden units to replicate the SAME direction (and its rectification).
  // That is exactly the intra-layer redundancy this report measures: trained
  // hidden units become strongly correlated with each other across the probe
  // batch, while at fresh init the same neurons are near-independent.
  procedure MakePair(out X, Y: TNNetVolume);
  var
    I: integer;
    Dot: TNeuralFloat;
  begin
    X := TNNetVolume.Create(cInDim, 1, 1);
    Y := TNNetVolume.Create(1, 1, 1);
    Dot := 0;
    for I := 0 to cInDim - 1 do
    begin
      X.Raw[I] := (Random - 0.5) * 2.0;
      Dot := Dot + gW[I] * X.Raw[I];
    end;
    if Dot < 0 then Dot := 0;
    Y.Raw[0] := Dot;
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
    Ep, B, I: integer;
    X, Yt, Out0: TNNetVolume;
    TotalLoss, Diff: TNeuralFloat;
  begin
    for Ep := 1 to Epochs do
    begin
      TotalLoss := 0;
      for B := 1 to cBatch do
      begin
        MakePair(X, Yt);
        try
          NN.Compute(X);
          Out0 := NN.GetLastLayer.Output;
          for I := 0 to Out0.Size - 1 do
          begin
            Diff := Out0.Raw[I] - Yt.Raw[I];
            TotalLoss := TotalLoss + Diff * Diff;
          end;
          NN.Backpropagate(Yt);
        finally
          X.Free;
          Yt.Free;
        end;
      end;
      if (Ep = 1) or (Ep mod 40 = 0) or (Ep = Epochs) then
        WriteLn(Format('  epoch %3d  mean-MSE=%.6f', [Ep, TotalLoss / cBatch]));
    end;
  end;

var
  NN: TNNet;
  Probes: TNNetVolumeList;
  I: integer;
begin
  RandSeed := 2026;

  // Fixed random teacher direction for the single-feature target.
  for I := 0 to cInDim - 1 do
    gW[I] := (Random - 0.5) * 2.0;

  BuildProbes(Probes);
  BuildMLP(NN);
  try
    WriteLn('NeuronCorrelationReport demo: 2-hidden-layer ReLU MLP ' +
      '(32 -> 32 -> 32 -> 1) on a single-ReLU-feature target ' +
      'y=max(0, w.x).');

    // ---- BEFORE training: fresh init ----
    WriteLn;
    WriteLn(StringOfChar('=', 92));
    WriteLn('FRESH INIT: neurons start independent -> expect an empty ' +
      'high-|rho| tail and no near-duplicate pairs.');
    WriteLn(StringOfChar('=', 92));
    Write(TNNet.NeuronCorrelationReport(NN, Probes));

    // ---- train briefly ----
    WriteLn;
    WriteLn('Training for ', cEpochs, ' epochs of batch size ', cBatch, '...');
    TrainOnce(NN, cEpochs);

    // ---- AFTER training: neurons specialise / align ----
    WriteLn;
    WriteLn(StringOfChar('=', 92));
    WriteLn('AFTER TRAINING: neurons specialise -> expect a heavier |rho| ' +
      'tail and a lower effective neuron count.');
    WriteLn(StringOfChar('=', 92));
    Write(TNNet.NeuronCorrelationReport(NN, Probes));

    WriteLn;
    WriteLn('Compare the two |rho| histograms and the per-layer effective ' +
      'neuron count: at fresh init the high-|rho| tail is empty and no ' +
      'near-duplicate flags fire, whereas the trained net grows a clear ' +
      '|rho|>0.8 tail, raises near-duplicate-pair flags, and its effective ' +
      'neuron count drops as hidden units align onto the single useful ' +
      'direction (intra-layer redundancy).');
  finally
    NN.Free;
    Probes.Free;
  end;
end.
