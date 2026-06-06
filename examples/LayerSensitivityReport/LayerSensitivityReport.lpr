program LayerSensitivityReport;
(*
LayerSensitivityReport: builds three small networks from different model
families -- a tiny MLP, a small CIFAR-style convolutional stack, and a small
attention stack -- and prints TNNet.LayerSensitivityReport on a probe batch
for each. The report multiplicatively jitters every TRAINABLE layer's weights
by small Gaussian noise (W *= 1 + eta, eta ~ N(0, sigma^2)), measures the
resulting change in the forward output (and, for the MLP, the MSE loss against
targets), restores the weights exactly between trials, and ranks layers by
sensitivity. Running it across model families shows how the "fragility" shape
shifts: which layers carry the model and how concentrated that is.

To stay self-contained and fast, the inputs are tiny synthetic volumes of the
right shape (no dataset download); the point is the sensitivity SHAPE across
layers, not accuracy.

Pure CPU, forward-only, well under a minute.

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
  cProbeCnt = 48;   // synthetic probe samples per model
  cVocab    = 16;   // attention vocabulary
  cSeqLen   = 12;   // attention sequence length
  cDModel   = 16;   // attention embedding width
  cDk       = 16;   // attention key/query/value width

// ---------------------------------------------------------------------------
// (i) tiny MLP on a synthetic regression target (run WITH loss-delta).
// ---------------------------------------------------------------------------
procedure RunMLP();
const
  cInDim  = 8;
  cHidden = 16;
  cEpochs = 120;
  cBatch  = 32;
var
  NN: TNNet;
  Probes, Targets: TNNetVolumeList;
  W: array[0..cInDim - 1] of TNeuralFloat;
  V, T, X, Yt, Out0: TNNetVolume;
  K, I, Ep, B: integer;
  Dot, Diff, TotalLoss: TNeuralFloat;
begin
  WriteLn(StringOfChar('=', 92));
  WriteLn('(i) MLP  (8 -> 16 -> 16 -> 1)  on a synthetic y = max(0, w.x) ' +
    'target -- WITH loss-delta.');
  WriteLn(StringOfChar('=', 92));

  for I := 0 to cInDim - 1 do W[I] := (Random - 0.5) * 2.0;

  NN := TNNet.Create();
  Probes := TNNetVolumeList.Create(True);
  Targets := TNNetVolumeList.Create(True);
  try
    NN.AddLayer(TNNetInput.Create(cInDim, 1, 1));
    NN.AddLayer(TNNetFullConnectReLU.Create(cHidden));
    NN.AddLayer(TNNetFullConnectReLU.Create(cHidden));
    NN.AddLayer(TNNetFullConnectLinear.Create(1));
    NN.SetLearningRate(0.05, 0.9);

    // probe batch + matching targets.
    for K := 0 to cProbeCnt - 1 do
    begin
      V := TNNetVolume.Create(cInDim, 1, 1);
      Dot := 0;
      for I := 0 to cInDim - 1 do
      begin
        V.Raw[I] := (Random - 0.5) * 2.0;
        Dot := Dot + W[I] * V.Raw[I];
      end;
      if Dot < 0 then Dot := 0;
      Probes.Add(V);
      T := TNNetVolume.Create(1, 1, 1);
      T.Raw[0] := Dot;
      Targets.Add(T);
    end;

    // brief training so sensitivities are not all at the fresh-init scale.
    for Ep := 1 to cEpochs do
    begin
      TotalLoss := 0;
      for B := 1 to cBatch do
      begin
        X := TNNetVolume.Create(cInDim, 1, 1);
        Yt := TNNetVolume.Create(1, 1, 1);
        Dot := 0;
        for I := 0 to cInDim - 1 do
        begin
          X.Raw[I] := (Random - 0.5) * 2.0;
          Dot := Dot + W[I] * X.Raw[I];
        end;
        if Dot < 0 then Dot := 0;
        Yt.Raw[0] := Dot;
        try
          NN.Compute(X);
          Out0 := NN.GetLastLayer.Output;
          Diff := Out0.Raw[0] - Yt.Raw[0];
          TotalLoss := TotalLoss + Diff * Diff;
          NN.Backpropagate(Yt);
        finally
          X.Free;
          Yt.Free;
        end;
      end;
    end;
    WriteLn(Format('  (trained %d epochs, final batch mean-MSE=%.6f)',
      [cEpochs, TotalLoss / cBatch]));
    WriteLn;
    Write(TNNet.LayerSensitivityReport(NN, Probes, Targets));
  finally
    Targets.Free;
    Probes.Free;
    NN.Free;
  end;
end;

// ---------------------------------------------------------------------------
// (ii) small CIFAR-style conv stack on synthetic 8x8x3 volumes (no targets).
// ---------------------------------------------------------------------------
procedure RunConv();
var
  NN: TNNet;
  Probes: TNNetVolumeList;
  V: TNNetVolume;
  K, X, Y, D: integer;
begin
  WriteLn;
  WriteLn(StringOfChar('=', 92));
  WriteLn('(ii) CIFAR-style conv stack on synthetic 8x8x3 inputs ' +
    '-- output-delta only (no targets).');
  WriteLn(StringOfChar('=', 92));

  NN := TNNet.Create();
  Probes := TNNetVolumeList.Create(True);
  try
    NN.AddLayer(TNNetInput.Create(8, 8, 3));
    NN.AddLayer(TNNetConvolutionReLU.Create(16, 3, 1, 1, 0));
    NN.AddLayer(TNNetMaxPool.Create(2));
    NN.AddLayer(TNNetConvolutionReLU.Create(32, 3, 1, 1, 0));
    NN.AddLayer(TNNetMaxPool.Create(2));
    NN.AddLayer(TNNetFullConnectReLU.Create(32));
    NN.AddLayer(TNNetFullConnectLinear.Create(10));
    NN.SetLearningRate(0.01, 0.9);
    NN.InitWeights();

    for K := 0 to cProbeCnt - 1 do
    begin
      V := TNNetVolume.Create(8, 8, 3);
      for X := 0 to 7 do
        for Y := 0 to 7 do
          for D := 0 to 2 do
            V[X, Y, D] := (Random - 0.5) * 2.0;
      Probes.Add(V);
    end;

    Write(TNNet.LayerSensitivityReport(NN, Probes));
  finally
    Probes.Free;
    NN.Free;
  end;
end;

// ---------------------------------------------------------------------------
// (iii) small attention stack on synthetic token-ID sequences (no targets).
// ---------------------------------------------------------------------------
procedure RunAttention();
var
  NN: TNNet;
  Probes: TNNetVolumeList;
  V: TNNetVolume;
  K, I: integer;
begin
  WriteLn;
  WriteLn(StringOfChar('=', 92));
  WriteLn('(iii) attention stack (embedding -> SDPA head -> projection) on ' +
    'synthetic token IDs -- output-delta only.');
  WriteLn(StringOfChar('=', 92));

  NN := TNNet.Create();
  Probes := TNNetVolumeList.Create(True);
  try
    // Token IDs along X; embedding -> sinusoidal positions -> Q|K|V pack ->
    // single-head non-causal SDPA -> per-position projection -> softmax.
    NN.AddLayer(TNNetInput.Create(cSeqLen, 1, 1));
    NN.AddLayer(TNNetEmbedding.Create(cVocab, cDModel, 1));
    NN.AddLayer(TNNetSinusoidalPositionalEmbedding.Create());
    NN.AddLayer(TNNetPointwiseConvLinear.Create(3 * cDk));
    NN.AddLayer(TNNetScaledDotProductAttention.Create(cDk, False));
    NN.AddLayer(TNNetPointwiseConvLinear.Create(cVocab));
    NN.AddLayer(TNNetPointwiseSoftMax.Create(1));
    NN.SetLearningRate(0.001, 0.9);
    NN.InitWeights();

    for K := 0 to cProbeCnt - 1 do
    begin
      V := TNNetVolume.Create(cSeqLen, 1, 1);
      for I := 0 to cSeqLen - 1 do
        V.Raw[I] := Random(cVocab);
      Probes.Add(V);
    end;

    Write(TNNet.LayerSensitivityReport(NN, Probes));
  finally
    Probes.Free;
    NN.Free;
  end;
end;

begin
  RandSeed := 2026;

  WriteLn('LayerSensitivityReport demo: per-layer weight-noise sensitivity ' +
    'across three model families.');
  WriteLn('Each report perturbs one trainable layer at a time (W *= 1 + eta, ' +
    'eta ~ N(0, sigma^2)),');
  WriteLn('measures the forward output-delta L2 over a probe batch, restores ' +
    'the weights exactly,');
  WriteLn('and ranks layers by sensitivity (with a fragility verdict).');
  WriteLn;

  RunMLP();
  RunConv();
  RunAttention();

  WriteLn;
  WriteLn('Compare the per-layer tables and fragility verdicts across the ' +
    'three families: a high max/median');
  WriteLn('ratio means a few layers dominate the model''s output sensitivity ' +
    'to small weight perturbations.');
end.
