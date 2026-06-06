program WeightSpectralTail;
(*
WeightSpectralTail: builds a small MLP, then prints TNNet.WeightSpectralTailReport
(i) on the fresh-init network and (ii) on the same architecture after a short
training run on a trivial synthetic task, so a reviewer can see how the
heavy-tailed self-regularization (HT-SR) power-law exponent alpha — computed from
the WEIGHTS ALONE, with no probe batch, no labels and no test set — drops out of
the under-trained / random-like band as the layers actually learn something.

For every trainable layer the report forms the smaller Gram matrix of the weight
tensor (W^T W when fan-out >= fan-in, else W W^T), computes its full eigenvalue
spectrum {lambda_i} (= the squared singular values of W) with a self-contained
symmetric cyclic Jacobi eigensolver in Double precision, and fits a power law
rho(lambda) ~ lambda^(-alpha) to the upper tail via the Clauset/Hill MLE
alpha = 1 + n / sum_i ln(lambda_i / lambda_min) swept over candidate lambda_min
cut points (picking the cut minimising the KS distance to the fitted power law).
Well-trained layers land in alpha in [2, 4]; alpha > 6 flags an under-trained /
still-random-like layer, alpha < 2 flags an over-correlated / memorising one. The
report closes with the network-level average weighted alpha (a single label-free
model-quality scalar, lower is better-trained), an alpha-across-depth bar chart
and per-layer flags.

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
  cInDim    = 16;
  cHidden   = 48;
  cEpochs   = 80;
  cBatch    = 32;

  // Small MLP. A mix of widths so the per-layer Gram dimensions differ.
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

  WriteLn('WeightSpectralTailReport demo: small MLP on y=||x||.');
  WriteLn('alpha is read off the WEIGHTS ALONE — no probe batch, no labels.');

  // ---- (i) fresh-init network ----
  BuildMLP(NN);
  try
    NN.InitWeights();
    WriteLn;
    WriteLn(StringOfChar('=', 100));
    WriteLn('(i) FRESH-INIT network (before any training): alpha should sit ' +
      'high (random-like / under-trained).');
    WriteLn(StringOfChar('=', 100));
    Write(TNNet.WeightSpectralTailReport(NN));

    // ---- (ii) same network after a short training run ----
    WriteLn;
    WriteLn('Training for ', cEpochs, ' epochs of batch size ', cBatch, '...');
    TrainOnce(NN, cEpochs);
    WriteLn;
    WriteLn(StringOfChar('=', 100));
    WriteLn('(ii) AFTER ', cEpochs, ' training epochs (same architecture): ' +
      'alpha should move toward the well-shaped band.');
    WriteLn(StringOfChar('=', 100));
    Write(TNNet.WeightSpectralTailReport(NN));
  finally
    NN.Free;
  end;

  WriteLn;
  WriteLn(
    'Compare (i) vs (ii): on a fresh Gaussian-style init each layer''s tail ' +
    'exponent alpha is large (steep / random-like) and the average weighted ' +
    'alpha is high; training builds a heavier-tailed spectrum, pulling alpha ' +
    'down toward the well-trained [2,4] band and lowering the average ' +
    'weighted alpha — the HT-SR "training quality with no data" story.');
end.
