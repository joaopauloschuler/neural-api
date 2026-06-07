program AnomalyAutoencoder;
(*
AnomalyAutoencoder: reconstruction-error anomaly detection with a small
autoencoder, on a SYNTHETIC dataset (generated in-code; no data files).

Idea: train an undercomplete autoencoder on ONE "normal" data distribution, then
score every test sample by its reconstruction error ||x - decode(encode(x))||².
Samples drawn from the normal distribution reconstruct well (low error); samples
from a held-out "anomaly" distribution the autoencoder never saw reconstruct
poorly (high error). Ranking by reconstruction error therefore separates normal
from anomalous points, which we quantify with AUROC.

Data (8-dimensional vectors):
  NORMAL  : points lying near a low-dimensional curved manifold (a 2-parameter
            "swiss-roll-ish" embedding into 8-D) + small Gaussian jitter. The
            autoencoder's 2-unit bottleneck can capture this manifold.
  ANOMALY : isotropic Gaussian blobs OFF that manifold (shifted + wider), so they
            do not lie on the learned 2-D surface and reconstruct badly.

Model (undercomplete AE):
  Input(8) -> FC-ReLU(16) -> FC-ReLU(2 bottleneck) -> FC-ReLU(16) -> FC-Linear(8)
Trained with plain SGD to minimise MSE against the input (the autoencoder target
IS the input). Only NORMAL samples are used for training.

We report:
  * mean reconstruction error on held-in NORMAL test points vs ANOMALY points,
  * the AUROC separating the two by reconstruction error (anomalies = positives),
    computed in-code via the Mann-Whitney-U rank statistic (helper copied from
    examples/MahalanobisOOD; kept local, not promoted to the library).

Self-checking: prints PASS/FAIL and Halt(1)s if AUROC <= cMinAUROC.

Pure CPU, single-threaded manual training loop, runs in a few seconds.

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
  cDim        = 8;     // ambient data dimension
  cBottleneck = 2;     // autoencoder latent dimension (matches manifold dim)
  cHidden     = 16;
  cTrainN     = 2000;  // NORMAL training samples
  cTestNormal = 600;   // held-in NORMAL test points (negatives)
  cTestAnom   = 600;   // ANOMALY points (positives)
  cEpochs     = 40;
  cLearnRate  = 0.02;
  cMinAUROC   = 0.85;  // self-check threshold

function RandomGauss(): TNeuralFloat;
var U1, U2: TNeuralFloat;
begin
  repeat U1 := Random; until U1 > 1e-9;
  U2 := Random;
  Result := Sqrt(-2 * Ln(U1)) * Cos(2 * Pi * U2);
end;

// NORMAL sample: point on a 2-parameter curved manifold embedded in 8-D plus
// small jitter. Two latent angles (a,b) drive a fixed nonlinear embedding, so
// the cloud lives near a 2-D surface that a 2-unit bottleneck can capture.
procedure MakeNormal(X: TNNetVolume);
var a, b: TNeuralFloat;
begin
  a := Random * 2 * Pi;
  b := Random * 2 * Pi;
  X.FData[0] := Sin(a)                + 0.03 * RandomGauss();
  X.FData[1] := Cos(a)                + 0.03 * RandomGauss();
  X.FData[2] := Sin(b)                + 0.03 * RandomGauss();
  X.FData[3] := Cos(b)                + 0.03 * RandomGauss();
  X.FData[4] := Sin(a + b)            + 0.03 * RandomGauss();
  X.FData[5] := Cos(a - b)            + 0.03 * RandomGauss();
  X.FData[6] := 0.5 * Sin(2 * a)      + 0.03 * RandomGauss();
  X.FData[7] := 0.5 * Cos(2 * b)      + 0.03 * RandomGauss();
end;

// ANOMALY sample: isotropic Gaussian blob OFF the manifold (shifted centre,
// wider spread). It does not lie on the learned 2-D surface, so the trained AE
// reconstructs it poorly.
procedure MakeAnomaly(X: TNNetVolume);
var d: integer;
begin
  for d := 0 to cDim - 1 do
    X.FData[d] := 0.6 + 0.8 * RandomGauss();
end;

// ---- AUROC via the Mann-Whitney U / rank statistic (tie-averaged ranks) ----
// Copied verbatim from examples/MahalanobisOOD (kept local on purpose).
function AUROC(const Pos, Neg: array of TNeuralFloat): TNeuralFloat;
var
  nPos, nNeg, n, I, J, K: integer;
  Vals: array of TNeuralFloat;
  IsPos: array of boolean;
  Rank: array of TNeuralFloat;
  AvgRank, RankSumPos, Tmp: TNeuralFloat;
  TmpB: boolean;
begin
  nPos := Length(Pos);
  nNeg := Length(Neg);
  n := nPos + nNeg;
  if (nPos = 0) or (nNeg = 0) then Exit(0.5);

  SetLength(Vals, n);
  SetLength(IsPos, n);
  for I := 0 to nPos - 1 do begin Vals[I] := Pos[I]; IsPos[I] := True; end;
  for I := 0 to nNeg - 1 do
  begin
    Vals[nPos + I] := Neg[I];
    IsPos[nPos + I] := False;
  end;

  for I := 1 to n - 1 do
  begin
    Tmp := Vals[I]; TmpB := IsPos[I];
    J := I - 1;
    while (J >= 0) and (Vals[J] > Tmp) do
    begin
      Vals[J + 1] := Vals[J]; IsPos[J + 1] := IsPos[J];
      Dec(J);
    end;
    Vals[J + 1] := Tmp; IsPos[J + 1] := TmpB;
  end;

  SetLength(Rank, n);
  I := 0;
  while I < n do
  begin
    J := I;
    while (J + 1 < n) and (Vals[J + 1] = Vals[I]) do Inc(J);
    AvgRank := ((I + 1) + (J + 1)) / 2.0;
    for K := I to J do Rank[K] := AvgRank;
    I := J + 1;
  end;

  RankSumPos := 0;
  for I := 0 to n - 1 do
    if IsPos[I] then RankSumPos := RankSumPos + Rank[I];

  Result := (RankSumPos - nPos * (nPos + 1) / 2.0) / (nPos * nNeg);
end;

function BuildAE(): TNNet;
begin
  Result := TNNet.Create();
  Result.AddLayer(TNNetInput.Create(cDim, 1, 1));
  Result.AddLayer(TNNetFullConnectReLU.Create(cHidden));
  Result.AddLayer(TNNetFullConnectReLU.Create(cBottleneck));  // latent bottleneck
  Result.AddLayer(TNNetFullConnectReLU.Create(cHidden));
  Result.AddLayer(TNNetFullConnectLinear.Create(cDim));       // reconstruction
end;

// Reconstruction error = sum of squared (output - input) for the last forward.
function ReconError(NN: TNNet; X: TNNetVolume): TNeuralFloat;
var d: integer; diff, acc: TNeuralFloat;
begin
  acc := 0;
  for d := 0 to cDim - 1 do
  begin
    diff := NN.GetLastLayer.Output.FData[d] - X.FData[d];
    acc := acc + diff * diff;
  end;
  Result := acc;
end;

var
  NN: TNNet;
  X: TNNetVolume;
  epoch, i: integer;
  loss: TNeuralFloat;
  NormScores, AnomScores: array of TNeuralFloat;
  meanNorm, meanAnom, auc: TNeuralFloat;
begin
  RandSeed := 20260607;
  WriteLn('=== AnomalyAutoencoder: reconstruction-error anomaly detection ===');
  WriteLn('dim=', cDim, '  bottleneck=', cBottleneck,
          '  train_normal=', cTrainN,
          '  test_normal=', cTestNormal, '  test_anomaly=', cTestAnom);
  WriteLn('NORMAL = points near a 2-D curved manifold in 8-D;',
          ' ANOMALY = off-manifold Gaussian blobs');
  WriteLn;

  NN := BuildAE();
  WriteLn('autoencoder params = ', NN.CountWeights());
  NN.SetLearningRate(cLearnRate, 0.9);
  NN.PrintSummary();
  WriteLn;

  X := TNNetVolume.Create(cDim, 1, 1);

  // ----------------------------- TRAIN (NORMAL only) -----------------------
  WriteLn('training the autoencoder on NORMAL samples (', cEpochs, ' epochs)...');
  WriteLn('  epoch   mean_recon_MSE');
  for epoch := 1 to cEpochs do
  begin
    loss := 0;
    for i := 1 to cTrainN do
    begin
      MakeNormal(X);
      NN.Compute(X);
      NN.Backpropagate(X);              // target IS the input (autoencoder)
      loss := loss + ReconError(NN, X);
    end;
    if (epoch = 1) or (epoch mod 5 = 0) then
      WriteLn('  ', epoch:5, '   ', (loss / (cTrainN * cDim)):12:6);
  end;
  WriteLn;

  // ------------------------------- SCORE -----------------------------------
  // Negatives = held-in NORMAL test points; positives = ANOMALY points.
  RandSeed := 111222;
  SetLength(NormScores, cTestNormal);
  for i := 0 to cTestNormal - 1 do
  begin
    MakeNormal(X);
    NN.Compute(X);
    NormScores[i] := ReconError(NN, X);
  end;
  SetLength(AnomScores, cTestAnom);
  for i := 0 to cTestAnom - 1 do
  begin
    MakeAnomaly(X);
    NN.Compute(X);
    AnomScores[i] := ReconError(NN, X);
  end;

  meanNorm := 0; for i := 0 to cTestNormal - 1 do meanNorm := meanNorm + NormScores[i];
  meanNorm := meanNorm / cTestNormal;
  meanAnom := 0; for i := 0 to cTestAnom - 1 do meanAnom := meanAnom + AnomScores[i];
  meanAnom := meanAnom / cTestAnom;

  auc := AUROC(AnomScores, NormScores);   // anomalies are positives

  WriteLn('reconstruction error (anomaly score):');
  WriteLn('  NORMAL  test  n=', cTestNormal:4, '  mean = ', meanNorm:10:5);
  WriteLn('  ANOMALY test  n=', cTestAnom:4,   '  mean = ', meanAnom:10:5);
  WriteLn;
  WriteLn('AUROC (normal vs anomaly by recon error) = ', auc:0:4);
  WriteLn;

  if auc <= cMinAUROC then
  begin
    WriteLn('FAIL: AUROC ', auc:0:4, ' <= ', cMinAUROC:0:2);
    X.Free; NN.Free;
    Halt(1);
  end;
  WriteLn('PASS: AUROC ', auc:0:4, ' > ', cMinAUROC:0:2,
          ' -- reconstruction error separates anomalies from normal points.');

  X.Free;
  NN.Free;
end.
