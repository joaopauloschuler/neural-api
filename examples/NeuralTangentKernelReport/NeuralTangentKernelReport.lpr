program NeuralTangentKernelReport;
(*
NeuralTangentKernelReport: demonstrates TNNet.NeuralTangentKernelReport — a
forward-only diagnostic that measures the EMPIRICAL NEURAL TANGENT KERNEL
(Jacot, Gabriel & Hongler 2018) of a classifier over a small probe batch.

The NTK couples training examples through their parameter gradients:
  K(x,x') = <grad_theta f(x), grad_theta f(x')>
where f is the scalar target-class logit. On a FROZEN net the report runs one
forward + one backward per probe (ClearDeltas before each, NEVER UpdateWeights)
to snapshot each probe's per-parameter weight-gradient vector g_i, forms the
N x N Gram matrix K_ij = <g_i, g_j>, and reports:
  - the kernel as a glyph-shaded ASCII heatmap;
  - its full eigenspectrum via a self-contained cyclic Jacobi eigensolver;
  - the condition number lambda_max/lambda_min;
  - the kernel-target ALIGNMENT (Cristianini et al. 2001) — the headline number;
  - the effective rank / participation ratio (sum lambda)^2 / sum lambda^2;
  - a log10(lambda) histogram.

This demo runs entirely on synthetic data (no dataset download), well under a
minute, with little memory:
  (A) FRESH-INIT network on a synthetic 3-class blob task.
  (B) TRAINED network on the same task.
Training markedly RESHAPES the empirical NTK: the eigenspectrum, the condition
number, the effective rank and the kernel-target alignment all move (the two
reports are visibly different), which is the point of the demo — the NTK is a
LIVE picture of how the current weights couple the probe examples.

A possible FOLLOW-UP (not done here) is a fresh-init-vs-trained NTK-DRIFT
contrast: how far the empirical kernel itself moves during training.

Pure CPU; forward+backward READ only — the weights are never stepped.

Copyright (C) 2026 Joao Paulo Schwarz Schuler

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
any later version.

Coded by Claude (AI).
*)

{$mode objfpc}{$H+}

uses {$IFDEF UNIX} cthreads, {$ENDIF}
  Classes, SysUtils,
  neuralnetwork,
  neuralvolume;

const
  cInDim    = 12;     // synthetic input dimension
  cHidden   = 24;
  cClasses  = 3;
  cEpochs   = 120;
  cBatch    = 24;
  cProbeN   = 9;      // probe-batch size (kept small: NTK is O(N^2) entries)

  // A tiny MLP classifier for the synthetic blob task.
  procedure BuildMLP(out NN: TNNet);
  begin
    NN := TNNet.Create();
    NN.AddLayer(TNNetInput.Create(cInDim, 1, 1));
    NN.AddLayer(TNNetFullConnectReLU.Create(cHidden));
    NN.AddLayer(TNNetFullConnectReLU.Create(cHidden));
    NN.AddLayer(TNNetFullConnectLinear.Create(cClasses));
    NN.AddLayer(TNNetSoftMax.Create());
    NN.SetLearningRate(0.01, 0.9);
  end;

  // Synthetic few-class blob: each class is a Gaussian cloud around a fixed
  // random center in cInDim space (a clean, linearly-separable-ish task).
  procedure FillCenters(var Centers: array of TNeuralFloat);
  var
    I: integer;
  begin
    for I := 0 to Length(Centers) - 1 do
      Centers[I] := (Random - 0.5) * 2.4;
  end;

  procedure MakePair(const Centers: array of TNeuralFloat;
    Cls: integer; out X, Y: TNNetVolume);
  var
    I: integer;
  begin
    X := TNNetVolume.Create(cInDim, 1, 1);
    Y := TNNetVolume.Create(cClasses, 1, 1);
    for I := 0 to cInDim - 1 do
      X.Raw[I] := Centers[Cls * cInDim + I] + (Random - 0.5) * 1.6;
    Y.Raw[Cls] := 1.0;
  end;

  procedure TrainOnce(NN: TNNet; const Centers: array of TNeuralFloat;
    Epochs: integer);
  var
    Ep, B, Cls: integer;
    X, Yt: TNNetVolume;
  begin
    // The report leaves the net in batch-update mode (it reads deltas without
    // stepping); restore per-sample updates so this training loop actually
    // steps the weights.
    NN.SetBatchUpdate(False);
    for Ep := 1 to Epochs do
      for B := 1 to cBatch do
      begin
        Cls := Random(cClasses);
        MakePair(Centers, Cls, X, Yt);
        try
          NN.Compute(X);
          NN.Backpropagate(Yt);
        finally
          X.Free;
          Yt.Free;
        end;
      end;
  end;

  // A small probe batch (unlabelled inputs), a few examples per class so the
  // kernel block structure / alignment is meaningful.
  function MakeProbeBatch(const Centers: array of TNeuralFloat): TNNetVolumeList;
  var
    I, Cls: integer;
    X, Yt: TNNetVolume;
  begin
    Result := TNNetVolumeList.Create(True);   // owns its volumes
    for I := 0 to cProbeN - 1 do
    begin
      Cls := I mod cClasses;
      MakePair(Centers, Cls, X, Yt);
      Yt.Free;          // the report is unlabelled; drop the target
      Result.Add(X);
    end;
  end;

var
  NN: TNNet;
  Probes: TNNetVolumeList;
  Centers: array of TNeuralFloat;
begin
  RandSeed := 2026;

  WriteLn('NeuralTangentKernelReport demo: empirical NTK of a tiny classifier.');

  SetLength(Centers, cClasses * cInDim);
  FillCenters(Centers);
  Probes := MakeProbeBatch(Centers);
  try
    // ===================== (A) FRESH-INIT network =======================
    BuildMLP(NN);
    try
      NN.InitWeights();
      WriteLn;
      WriteLn(StringOfChar('=', 78));
      WriteLn('(A) FRESH-INIT network (kernel not yet aligned with the task).');
      WriteLn(StringOfChar('=', 78));
      Write(TNNet.NeuralTangentKernelReport(NN, Probes));

      // ===================== (B) TRAINED network ========================
      WriteLn;
      WriteLn('Training for ', cEpochs, ' epochs of batch size ', cBatch, '...');
      TrainOnce(NN, Centers, cEpochs);
      WriteLn;
      WriteLn(StringOfChar('=', 78));
      WriteLn('(B) TRAINED network (the NTK is reshaped by training).');
      WriteLn(StringOfChar('=', 78));
      Write(TNNet.NeuralTangentKernelReport(NN, Probes));
    finally
      NN.Free;
    end;
  finally
    Probes.Free;
  end;

  WriteLn;
  WriteLn(
    'Read it as: compare (A) vs (B). Training RESHAPES the empirical NTK — the ' +
    'KERNEL-TARGET ALIGNMENT (the headline number), the condition number and ' +
    'the effective rank all move as the weights change. A high alignment with ' +
    'a small condition number means the kernel''s dominant eigen-directions ' +
    'separate the target class and kernel regression / gradient descent on it ' +
    'is well-conditioned; near convergence the per-logit gradients shrink and ' +
    'the kernel often collapses toward a low-rank, ill-conditioned state.');
end.
