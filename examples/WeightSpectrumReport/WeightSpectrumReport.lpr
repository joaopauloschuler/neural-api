program WeightSpectrumReport;
(*
WeightSpectrumReport: builds a small MLP, then prints TNNet.WeightSpectrumReport
(i) on the fresh-init network and (ii) on the same architecture after a short
training run on a trivial synthetic task, so a reviewer can eyeball how training
pushes each layer's weight spectrum away from the Gaussian-init baseline.

The report is pure forward-only on the weight tensors (no probe batch needed):
for every trainable layer it treats the weights as a matrix W
[num_neurons (fan-out) x weights_per_neuron (fan-in)] (biases excluded) and
estimates the top singular value sigma_1 via a handful of power-iteration steps,
then reports sigma_1, the Frobenius norm ||W||_F, the stable-rank-flavoured
ratio sigma_1/||W||_F (near 1 => rank-1 collapse), and a Marchenko-Pastur
baseline ratio (near 1 => init-like top mode). It closes with a 10-bin ASCII
histogram of the per-layer fan-in baseline ratio and a flag list
(high-spectral-norm / rank-1-collapse layers).

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
  cInDim    = 8;
  cHidden   = 32;
  cEpochs   = 60;
  cBatch    = 32;

  // Small MLP. A mix of widths so fan-in / fan-out baselines differ per layer.
  procedure BuildMLP(out NN: TNNet);
  begin
    NN := TNNet.Create();
    NN.AddLayer(TNNetInput.Create(cInDim, 1, 1));
    NN.AddLayer(TNNetFullConnectReLU.Create(cHidden));
    NN.AddLayer(TNNetFullConnectReLU.Create(cHidden));
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
begin
  RandSeed := 2026;

  WriteLn('WeightSpectrumReport demo: small MLP on y=||x||.');

  // ---- (i) fresh-init network ----
  BuildMLP(NN);
  try
    NN.InitWeights();
    WriteLn;
    WriteLn(StringOfChar('=', 92));
    WriteLn('(i) FRESH-INIT network (before any training).');
    WriteLn(StringOfChar('=', 92));
    Write(TNNet.WeightSpectrumReport(NN));

    // ---- (ii) same network after a short training run ----
    WriteLn;
    WriteLn('Training for ', cEpochs, ' epochs of batch size ', cBatch, '...');
    TrainOnce(NN, cEpochs);
    WriteLn;
    WriteLn(StringOfChar('=', 92));
    WriteLn('(ii) AFTER ', cEpochs, ' training epochs (same architecture).');
    WriteLn(StringOfChar('=', 92));
    Write(TNNet.WeightSpectrumReport(NN));
  finally
    NN.Free;
  end;

  WriteLn;
  WriteLn(
    'Compare (i) vs (ii): on a fresh Gaussian-style init the MP-ratio sits ' +
    'near 1 and the spectrum is well spread; training grows sigma_1 and the ' +
    'sr-ratio / MP-ratio columns and the fan-in-baseline histogram shift as ' +
    'one direction starts to dominate.');
end.
