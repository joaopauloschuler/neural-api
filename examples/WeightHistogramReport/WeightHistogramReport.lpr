program WeightHistogramReport;
(*
WeightHistogramReport: builds a small MLP, prints TNNet.WeightHistogramReport
on freshly-initialised weights, trains briefly on a synthetic hypotenuse
regression task, then prints the histogram again. Useful for sanity-checking
that training is actually moving the weight distribution.

Pure CPU, well under 30s.

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
  cInDim  = 8;
  cHidden = 32;
  cEpochs = 60;
  cBatch  = 32;

  procedure BuildMLP(out NN: TNNet);
  begin
    NN := TNNet.Create();
    NN.AddLayer(TNNetInput.Create(cInDim, 1, 1));
    NN.AddLayer(TNNetFullConnectReLU.Create(cHidden));
    NN.AddLayer(TNNetFullConnectReLU.Create(cHidden));
    NN.AddLayer(TNNetFullConnectLinear.Create(1));
    NN.SetLearningRate(0.01, 0.9);
  end;

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

  function MaxAbsWeight(NN: TNNet): TNeuralFloat;
  var
    L, N, K: integer;
    Layer: TNNetLayer;
    Neuron: TNNetNeuron;
    V: TNeuralFloat;
  begin
    Result := 0;
    for L := 0 to NN.GetLastLayerIdx() do
    begin
      Layer := NN.Layers[L];
      if Layer.Neurons.Count = 0 then Continue;
      for N := 0 to Layer.Neurons.Count - 1 do
      begin
        Neuron := Layer.Neurons[N];
        for K := 0 to Neuron.Weights.Size - 1 do
        begin
          V := Abs(Neuron.Weights.FData[K]);
          if V > Result then Result := V;
        end;
      end;
    end;
  end;

var
  NN: TNNet;
  BeforeMaxAbs, AfterMaxAbs: TNeuralFloat;
begin
  RandSeed := 2026;

  BuildMLP(NN);
  try
    WriteLn('WeightHistogramReport demo: 2-hidden-layer ReLU MLP on y=||x||.');
    WriteLn;
    WriteLn(StringOfChar('=', 78));
    WriteLn('BEFORE TRAINING (fresh init):');
    WriteLn(StringOfChar('=', 78));
    BeforeMaxAbs := MaxAbsWeight(NN);
    Write(TNNet.WeightHistogramReport(NN));

    WriteLn;
    WriteLn(Format('Training for %d epochs of batch size %d...',
      [cEpochs, cBatch]));
    TrainOnce(NN, cEpochs);
    WriteLn;
    WriteLn(StringOfChar('=', 78));
    WriteLn('AFTER TRAINING:');
    WriteLn(StringOfChar('=', 78));
    AfterMaxAbs := MaxAbsWeight(NN);
    Write(TNNet.WeightHistogramReport(NN));

    WriteLn;
    WriteLn(Format(
      'Before training: max |w| = %.4f. After training: max |w| = %.4f.',
      [BeforeMaxAbs, AfterMaxAbs]));
  finally
    NN.Free;
  end;
end.
