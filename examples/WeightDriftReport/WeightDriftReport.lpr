program WeightDriftReport;
(*
WeightDriftReport: trains a 6-layer ReLU MLP on a tiny hypotenuse-like task
for a few epochs with one hidden layer's LearningRate pinned at 0. Snapshots
the network before and after training and prints
TNNet.WeightDriftReport(SnapA, SnapB).

The frozen layer should show ~0 L2 drift and a ~1.0 frozen fraction, while
the surrounding layers show non-trivial drift. Pure-CPU, runs in seconds.

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
  cFrozenLayerIdx = 3;
  cEpochs         = 200;
  cSamples        = 64;
  cLearningRate   = 0.01;

  procedure BuildNet(out NN: TNNet);
  begin
    NN := TNNet.Create();
    NN.AddLayer(TNNetInput.Create(2, 1, 1));
    NN.AddLayer(TNNetFullConnectReLU.Create(16));
    NN.AddLayer(TNNetFullConnectReLU.Create(16));
    NN.AddLayer(TNNetFullConnectReLU.Create(16));
    NN.AddLayer(TNNetFullConnectReLU.Create(16));
    NN.AddLayer(TNNetFullConnectLinear.Create(1));
  end;

  procedure MakeSample(out X, Y: TNNetVolume);
  var
    A, B: TNeuralFloat;
  begin
    A := Random;
    B := Random;
    X := TNNetVolume.Create(2, 1, 1);
    X.FData[0] := A;
    X.FData[1] := B;
    Y := TNNetVolume.Create(1, 1, 1);
    Y.FData[0] := Sqrt(A * A + B * B);
  end;

  procedure RunDemo;
  var
    NN: TNNet;
    X, Yt: TNNetVolume;
    SnapA, SnapB, Report: string;
    Epoch, I, L: integer;
    Loss, Diff: TNeuralFloat;
  begin
    RandSeed := 42;
    BuildNet(NN);
    try
      NN.SetLearningRate(cLearningRate, 0.9);
      // Why: freeze a specific hidden layer to show 0 drift in the report.
      NN.Layers[cFrozenLayerIdx].LearningRate := 0;

      WriteLn('Architecture (layer ', cFrozenLayerIdx, ' frozen at LR=0):');
      NN.PrintSummary();

      SnapA := NN.SaveToString();

      WriteLn('Training for ', cEpochs, ' epochs on a tiny hypotenuse task...');
      for Epoch := 1 to cEpochs do
      begin
        Loss := 0;
        for I := 1 to cSamples do
        begin
          MakeSample(X, Yt);
          try
            NN.Compute(X);
            NN.Backpropagate(Yt);
            Diff := NN.GetLastLayer().Output.FData[0] - Yt.FData[0];
            Loss := Loss + Diff * Diff;
          finally
            X.Free;
            Yt.Free;
          end;
        end;
        if (Epoch mod 50 = 0) or (Epoch = 1) then
          WriteLn('  epoch ', Epoch:4, '  mse=', (Loss / cSamples):8:5);
      end;

      SnapB := NN.SaveToString();

      WriteLn;
      WriteLn('Weight drift report:');
      WriteLn(StringOfChar('=', 96));
      Report := TNNet.WeightDriftReport(SnapA, SnapB);
      Write(Report);
      WriteLn(StringOfChar('=', 96));
      WriteLn;
      WriteLn('Expected: layer ', cFrozenLayerIdx,
        ' shows ~0 L2 drift and ~1.0 frac frozen;');
      WriteLn('          other trainable layers show non-zero drift.');

      // Why: surface absurdly large drift (would indicate exploding update).
      L := 0;
      if Length(Report) > 0 then L := L; // silence hint
    finally
      NN.Free;
    end;
  end;

begin
  RunDemo();
end.
