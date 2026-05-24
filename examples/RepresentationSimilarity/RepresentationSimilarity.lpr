program RepresentationSimilarity;
(*
RepresentationSimilarity: demonstrates TNNet.RepresentationSimilarityReport,
a forward-only linear-CKA (Centered Kernel Alignment, Kornblith et al. 2019)
diagnostic that measures how the representation reshapes itself with depth and
which layers do redundant work.

It runs three contrasts:
  (1) A deliberately OVER-DEEP MLP at FRESH INIT: adjacent-layer CKA is already
      high because untrained layers barely transform their input.
  (2) The SAME over-deep MLP AFTER a short training run on a simple task: a
      clearer block structure emerges, yet the middle layers stay near-
      duplicates of one another -> "depth is wasted" lights up.
  (3) CROSS-CKA between two independently-initialised nets of the same shape
      over the same probe batch -> "do these two nets learn the same
      intermediate features?".

Self-contained and synthetic (no dataset download). Pure CPU, well under a
minute.

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
  cInDim    = 16;
  cHidden   = 16;
  cDepth    = 6;     // hidden layers -> deliberately over-deep for the task
  cEpochs   = 300;
  cBatch    = 48;
  cProbeCnt = 64;
  cLR       = 0.05;

var
  // Fixed random teacher direction for the single-feature target.
  gW: array[0..cInDim - 1] of TNeuralFloat;

  // An over-deep MLP: cDepth hidden ReLU layers on a task that needs ~1.
  procedure BuildDeepMLP(out NN: TNNet);
  var
    D: integer;
  begin
    NN := TNNet.Create();
    NN.AddLayer(TNNetInput.Create(cInDim, 1, 1));
    for D := 1 to cDepth do
      NN.AddLayer(TNNetFullConnectReLU.Create(cHidden));
    NN.AddLayer(TNNetFullConnectLinear.Create(1));
    NN.SetLearningRate(cLR, 0.9);
  end;

  // Target: a SINGLE rectified-linear teacher feature, y = max(0, gW . x).
  // The whole label is one ReLU feature, so the extra depth is unnecessary
  // and the middle layers end up doing near-redundant transformations.
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
      if (Ep = 1) or (Ep mod 50 = 0) or (Ep = Epochs) then
        WriteLn(Format('  epoch %3d  mean-MSE=%.6f', [Ep, TotalLoss / cBatch]));
    end;
  end;

var
  NN, NN2: TNNet;
  Probes: TNNetVolumeList;
  I: integer;
begin
  RandSeed := 2026;

  // Fixed random teacher direction for the single-feature target.
  for I := 0 to cInDim - 1 do
    gW[I] := (Random - 0.5) * 2.0;

  BuildProbes(Probes);
  BuildDeepMLP(NN);
  try
    WriteLn('RepresentationSimilarityReport demo: over-deep ReLU MLP ' +
      Format('(%d -> %d x%d -> 1) on a single-ReLU-feature target y=max(0, w.x).',
        [cInDim, cHidden, cDepth]));

    // ---- (1) FRESH INIT: adjacent-layer CKA already high ----
    WriteLn;
    WriteLn(StringOfChar('=', 92));
    WriteLn('(1) FRESH INIT: untrained layers barely transform their input, ' +
      'so adjacent-layer CKA is already high.');
    WriteLn(StringOfChar('=', 92));
    Write(TNNet.RepresentationSimilarityReport(NN, Probes));

    // ---- train briefly ----
    WriteLn;
    WriteLn('Training for ', cEpochs, ' epochs of batch size ', cBatch, '...');
    TrainOnce(NN, cEpochs);

    // ---- (2) AFTER TRAINING: clearer block structure, wasted middle depth ----
    WriteLn;
    WriteLn(StringOfChar('=', 92));
    WriteLn('(2) AFTER TRAINING: a clearer block structure emerges; the ' +
      'over-deep middle layers stay near-duplicates -> depth is wasted.');
    WriteLn(StringOfChar('=', 92));
    Write(TNNet.RepresentationSimilarityReport(NN, Probes));

    // ---- (3) CROSS-CKA between two independent nets of the same shape ----
    BuildDeepMLP(NN2);
    try
      WriteLn;
      WriteLn(StringOfChar('=', 92));
      WriteLn('(3) CROSS-CKA: trained net vs a second FRESH-init net of the ' +
        'same shape -> do they learn the same intermediate features?');
      WriteLn(StringOfChar('=', 92));
      Write(TNNet.RepresentationSimilarityReport(NN, Probes, NN2));
    finally
      NN2.Free;
    end;

    WriteLn;
    WriteLn('Read the heatmap and the adjacent-layer vector: a high adjacent ' +
      'CKA flags a near-pass-through redundant layer, a sharp dip flags where ' +
      'the representation genuinely reorganizes. The "most-redundant pair" and ' +
      'the block / verdict lines summarise which depth is wasted.');
  finally
    NN.Free;
    Probes.Free;
  end;
end.
