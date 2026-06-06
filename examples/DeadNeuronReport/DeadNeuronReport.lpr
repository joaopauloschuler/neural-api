program DeadNeuronReport;
(*
DeadNeuronReport: builds a small ReLU MLP, trains it briefly on a synthetic
regression task, then prints TNNet.DeadNeuronReport across a probe set.
Runs twice: once with a sane learning rate (Glorot init + small LR) and
once with a deliberately huge learning rate to induce dying-ReLU units,
so the contrast is visible.

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
  cHidden   = 48;
  cEpochs   = 80;
  cBatch    = 32;
  cProbeCnt = 48;

  procedure BuildMLP(out NN: TNNet; LR: TNeuralFloat);
  begin
    NN := TNNet.Create();
    NN.AddLayer(TNNetInput.Create(cInDim, 1, 1));
    NN.AddLayer(TNNetFullConnectLinear.Create(cHidden));
    NN.AddLayer(TNNetReLU.Create());
    NN.AddLayer(TNNetFullConnectLinear.Create(cHidden));
    NN.AddLayer(TNNetReLU.Create());
    NN.AddLayer(TNNetFullConnectLinear.Create(cHidden));
    NN.AddLayer(TNNetReLU.Create());
    NN.AddLayer(TNNetFullConnectLinear.Create(1));
    NN.SetLearningRate(LR, 0.9);
  end;

  // Target: y = ||x|| (hypotenuse). Smooth scalar regression that gives
  // every hidden unit a useful signal under sane training, so dead units
  // are an obvious failure when they show up.
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
    Ep, B, I: integer;
    X, Yt: TNNetVolume;
    Out0: TNNetVolume;
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
      if (Ep = 1) or (Ep mod 20 = 0) or (Ep = Epochs) then
        WriteLn(Format('  epoch %3d  mean-MSE=%.6f',
          [Ep, TotalLoss / (cBatch * 1)]));
    end;
  end;

var
  NNSane, NNBad: TNNet;
  Probes: TNNetVolumeList;
begin
  RandSeed := 2026;

  BuildProbes(Probes);
  try
    WriteLn('DeadNeuronReport demo: 3-hidden-layer ReLU MLP on y=||x||.');

    // ---- Run 1: sane LR ----
    BuildMLP(NNSane, 0.01);
    try
      WriteLn;
      WriteLn(StringOfChar('=', 92));
      WriteLn('RUN 1: sane learning rate (LR=0.01). Expect ~0% dead units.');
      WriteLn(StringOfChar('=', 92));
      WriteLn('Training for ', cEpochs, ' epochs of batch size ', cBatch, '...');
      TrainOnce(NNSane, cEpochs);
      WriteLn;
      Write(TNNet.DeadNeuronReport(NNSane, Probes));
    finally
      NNSane.Free;
    end;

    // ---- Run 2: aggressively high LR (chosen to kill ReLU units without
    // overflowing). ----
    RandSeed := 2026;
    BuildMLP(NNBad, 0.5);
    try
      WriteLn;
      WriteLn(StringOfChar('=', 92));
      WriteLn('RUN 2: aggressive learning rate (LR=0.5). Expect many dead units.');
      WriteLn(StringOfChar('=', 92));
      WriteLn('Training for ', cEpochs, ' epochs of batch size ', cBatch, '...');
      TrainOnce(NNBad, cEpochs);
      WriteLn;
      Write(TNNet.DeadNeuronReport(NNBad, Probes));
    finally
      NNBad.Free;
    end;

    WriteLn;
    WriteLn(
      'Expect: RUN 1 reports low dead% (healthy ReLU MLP); ' +
      'RUN 2 reports a large fraction of dead units in deeper layers - ' +
      'the classic "dying ReLU" pattern caused by an LR-too-high schedule.');
  finally
    Probes.Free;
  end;
end.
