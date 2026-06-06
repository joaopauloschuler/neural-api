program HessianCurvature;
(*
HessianCurvature: trains the SAME tiny MLP twice on a trivial synthetic
classification task — once with a small batch + high learning rate (which
settles into a SHARP minimum) and once with a large batch + low learning rate
and mild weight decay (which settles into a FLATTER minimum) — then prints
TNNet.HessianCurvatureReport for each, so the lambda_max (top Hessian
eigenvalue) gap the generalization literature ties to sharpness is visible on a
pure-CPU toy.

The report estimates loss-surface curvature with Hessian-vector products (HVPs)
finite-differenced from the gradient,
  H v ~= (grad L(theta + eps*v) - grad L(theta - eps*v)) / (2*eps),
so it needs no second-order autograd: it reuses the whole-batch forward+backward
gradient machinery on a FROZEN net (weights are snapshotted and restored
bit-for-bit between probes, never stepped). It reports the Hessian trace tr(H)
(Hutchinson estimator over Rademacher probes = mean curvature), the top
eigenvalue lambda_max (power iteration on the HVP = the flat-vs-sharp metric),
the curvature-concentration ratio, a per-layer trace breakdown, a per-probe
v^T H v histogram (Hutchinson spread) and a flat / moderate / sharp verdict.

Pure CPU, well under a minute. Weights are never stepped by the report (a
measurement, not training).

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
  cHidden   = 16;
  cClasses  = 3;
  cProbe    = 48;   // probe-batch size handed to the report

  // Two 2D-ish cluster centres lifted into cInDim dims (first two coords carry
  // the signal, the rest is noise) — a trivial, fully separable problem.
  Centers: array[0..cClasses - 1, 0..1] of TNeuralFloat =
    ((-2.0, -2.0), (2.0, 2.0), (2.0, -2.0));

  // Small MLP shared by both training runs.
  procedure BuildMLP(out NN: TNNet);
  begin
    NN := TNNet.Create();
    NN.AddLayer(TNNetInput.Create(cInDim, 1, 1));
    NN.AddLayer(TNNetFullConnectReLU.Create(cHidden));
    NN.AddLayer(TNNetFullConnectLinear.Create(cClasses));
  end;

  // One labelled sample: class C cluster + per-feature noise, one-hot target.
  procedure MakePair(out X, Y: TNNetVolume);
  var
    C, I: integer;
  begin
    C := Random(cClasses);
    X := TNNetVolume.Create(cInDim, 1, 1);
    Y := TNNetVolume.Create(cClasses, 1, 1);
    X.Raw[0] := Centers[C][0] + (Random - 0.5) * 0.8;
    X.Raw[1] := Centers[C][1] + (Random - 0.5) * 0.8;
    for I := 2 to cInDim - 1 do
      X.Raw[I] := (Random - 0.5) * 0.5;
    Y.Fill(0);
    Y.Raw[C] := 1.0;
  end;

  procedure TrainOnce(NN: TNNet; Epochs, Batch: integer);
  var
    Ep, B: integer;
    X, Yt: TNNetVolume;
  begin
    for Ep := 1 to Epochs do
    begin
      NN.ClearDeltas();
      for B := 1 to Batch do
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
      NN.UpdateWeights();
    end;
  end;

  procedure FillProbeBatch(Samples: TNNetVolumePairList; Count: integer);
  var
    I: integer;
    X, Yt: TNNetVolume;
  begin
    for I := 1 to Count do
    begin
      MakePair(X, Yt);
      Samples.Add(TNNetVolumePair.Create(X, Yt));
    end;
  end;

var
  SharpNN, FlatNN: TNNet;
  Probe: TNNetVolumePairList;
begin
  RandSeed := 2026;

  WriteLn('HessianCurvatureReport demo: the same tiny MLP trained into a ');
  WriteLn('SHARP vs a FLAT minimum on a separable 3-class toy problem.');
  WriteLn('Curvature is read off finite-difference Hessian-vector products.');

  // Shared probe batch for both reports (so only the minimum differs).
  Probe := TNNetVolumePairList.Create();
  try
    FillProbeBatch(Probe, cProbe);

    // ---- SHARP minimum: small batch + high learning rate, no weight decay. --
    BuildMLP(SharpNN);
    try
      SharpNN.InitWeights();
      SharpNN.SetBatchUpdate(true);
      SharpNN.SetLearningRate(0.20, 0.9);   // high LR
      TrainOnce(SharpNN, 600, 4);           // small batch

      // ---- FLAT minimum: large batch + low LR + mild weight decay. ----
      BuildMLP(FlatNN);
      try
        FlatNN.InitWeights();
        FlatNN.SetBatchUpdate(true);
        FlatNN.SetLearningRate(0.02, 0.9);  // low LR
        FlatNN.SetL2Decay(0.01);            // mild weight decay
        TrainOnce(FlatNN, 600, 64);         // large batch

        WriteLn;
        WriteLn(StringOfChar('=', 100));
        WriteLn('(i) SHARP-minimum net (small batch=4, LR=0.20): ' +
          'expect a LARGER lambda_max.');
        WriteLn(StringOfChar('=', 100));
        Write(TNNet.HessianCurvatureReport(SharpNN, Probe));

        WriteLn;
        WriteLn(StringOfChar('=', 100));
        WriteLn('(ii) FLAT-minimum net (large batch=64, LR=0.02, wd=0.01): ' +
          'expect a SMALLER lambda_max.');
        WriteLn(StringOfChar('=', 100));
        Write(TNNet.HessianCurvatureReport(FlatNN, Probe));
      finally
        FlatNN.Free;
      end;
    finally
      SharpNN.Free;
    end;
  finally
    Probe.Free;
  end;

  WriteLn;
  WriteLn(
    'Compare (i) vs (ii): the high-LR / small-batch run lands in a sharper ' +
    'minimum (larger lambda_max and curvature concentration); the low-LR / ' +
    'large-batch / weight-decayed run lands in a flatter one (smaller ' +
    'lambda_max) — the flat-vs-sharp sharpness gap the generalization ' +
    'literature (Keskar 2017, SAM 2021) cares about, on a pure-CPU toy.');
end.
