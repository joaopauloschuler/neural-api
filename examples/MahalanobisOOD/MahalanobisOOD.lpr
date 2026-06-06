program MahalanobisOOD;
(*
MahalanobisOOD: a tiny pure-CPU reproduction of Lee et al. 2018
"A Simple Unified Framework for Detecting Out-of-Distribution Samples and
Adversarial Attacks" (NeurIPS 2018).

Pipeline:
  TRAIN  -> fit a small softmax MLP on K IN-distribution Gaussian blobs
            (the classifier is then FROZEN: feature extraction is forward-only).
  FIT    -> read the PENULTIMATE-layer feature vector f for every training
            sample (NN.Layers[cPenultimateIdx].Output after NN.Compute), then
            fit one class-conditional Gaussian per class: per-class mean mu_c
            and a single SHARED (tied) covariance Sigma pooled across classes.
  SCORE  -> for a new x, score(x) = max_c -(f-mu_c)^T Sigma^-1 (f-mu_c)
            (the maximum NEGATIVE squared Mahalanobis distance over classes).
            IN-distribution points sit close to a class mean -> high score;
            OOD points sit far from every mean -> low score.

OOD test set = far-away Gaussian blobs in a region the classifier never saw.
We report a single AUROC (held-in test samples = positives, OOD = negatives)
computed via the Mann-Whitney U / rank-statistic form with tie-averaged ranks.

Numerical note: the tied covariance Sigma is regularised with a small ridge
(+cRidge on the diagonal, cRidge = 1e-3) before inversion so the Cholesky
factorisation stays well-conditioned even when a feature is near-constant.

Built-in self-check (this is a self-checking example, since the AUROC helper
lives locally in this .lpr): the program prints PASS/FAIL and Halt(1)s if the
observed AUROC on the easy synthetic split is not > cMinAUROC (0.8).

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
  cInputDim       = 8;     // input feature dimension
  cNumClasses     = 5;     // K in-distribution classes
  cFeatDim        = 16;    // penultimate-layer width (Mahalanobis works here)
  cTrainPerClass  = 200;
  cTestPerClass   = 200;   // held-in test points (positives)
  cOODCount       = 1000;  // out-of-distribution points (negatives)
  cEpochs         = 60;
  cLearningRate   = 0.03;
  cRidge          = 1e-3;  // diagonal ridge added to Sigma before inversion
  cMinAUROC       = 0.8;   // self-check threshold

  // index of the penultimate layer (the FullConnectReLU feeding the logits):
  //   0 Input  1 FCReLU  2 FCReLU(<-penultimate)  3 FCLinear  4 SoftMax
  cPenultimateIdx = 2;

  cBlobSpread     = 1.0;   // per-class Gaussian std in input space
  cClassSep       = 6.0;   // spacing between in-distribution class centres
  cOODShift       = 30.0;  // how far the OOD blob sits from the in-dist cloud

type
  TFeat   = array[0..cFeatDim - 1] of TNeuralFloat;
  TMatrix = array of array of TNeuralFloat;

// Why: Box-Muller gives N(0,1) samples without pulling in a dependency.
function RandomGauss(): TNeuralFloat;
var
  U1, U2: TNeuralFloat;
begin
  repeat
    U1 := Random;
  until U1 > 1e-9;
  U2 := Random;
  Result := Sqrt(-2 * Ln(U1)) * Cos(2 * Pi * U2);
end;

procedure BuildNet(out NN: TNNet);
begin
  NN := TNNet.Create();
  NN.AddLayer(TNNetInput.Create(cInputDim, 1, 1));
  NN.AddLayer(TNNetFullConnectReLU.Create(cFeatDim));
  NN.AddLayer(TNNetFullConnectReLU.Create(cFeatDim));   // penultimate (idx 2)
  NN.AddLayer(TNNetFullConnectLinear.Create(cNumClasses));
  NN.AddLayer(TNNetSoftMax.Create());
end;

// In-distribution class centre: class c lives along a coordinate axis so the
// K blobs are linearly separable but the features still need K means.
procedure ClassCenter(ClassId: integer; out C: array of TNeuralFloat);
var
  D: integer;
begin
  for D := 0 to cInputDim - 1 do C[D] := 0;
  C[ClassId mod cInputDim] := cClassSep * (1 + ClassId div cInputDim);
end;

procedure MakeInSample(ClassId: integer; X, Y: TNNetVolume);
var
  D: integer;
  C: array[0..cInputDim - 1] of TNeuralFloat;
begin
  ClassCenter(ClassId, C);
  for D := 0 to cInputDim - 1 do
    X.FData[D] := C[D] + RandomGauss() * cBlobSpread;
  Y.Fill(0);
  Y.FData[ClassId] := 1.0;
end;

// OOD sample: a Gaussian blob shifted far from every in-distribution centre.
procedure MakeOODSample(X: TNNetVolume);
var
  D: integer;
begin
  for D := 0 to cInputDim - 1 do
    X.FData[D] := cOODShift + RandomGauss() * cBlobSpread;
end;

// ---- AUROC via the Mann-Whitney U / rank statistic (tie-averaged ranks) ----
// Positives = Pos[], negatives = Neg[]. Merges, sorts ascending, assigns
// 1-based ranks averaging ties, then
//   AUROC = (sum of positive ranks - nPos*(nPos+1)/2) / (nPos * nNeg).
// Equals P(score(pos) > score(neg)) with ties counted as 1/2.
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

  // insertion sort ascending by value (n ~ a few thousand, called once)
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

  // tie-averaged 1-based ranks
  SetLength(Rank, n);
  I := 0;
  while I < n do
  begin
    J := I;
    while (J + 1 < n) and (Vals[J + 1] = Vals[I]) do Inc(J);
    // ranks i+1 .. j+1 averaged
    AvgRank := ((I + 1) + (J + 1)) / 2.0;
    for K := I to J do Rank[K] := AvgRank;
    I := J + 1;
  end;

  RankSumPos := 0;
  for I := 0 to n - 1 do
    if IsPos[I] then RankSumPos := RankSumPos + Rank[I];

  Result := (RankSumPos - nPos * (nPos + 1) / 2.0) / (nPos * nNeg);
end;

// In-place Cholesky inverse of an SPD matrix A (size cFeatDim).
// A = L L^T, then invert via forward/back substitution on the columns of I.
// Returns the inverse in Inv. Assumes A is SPD (guaranteed by the ridge).
procedure CholeskyInverse(const A: TMatrix; out Inv: TMatrix);
var
  n, i, j, k: integer;
  L: TMatrix;
  Y, Col: array of TNeuralFloat;
  Sum: TNeuralFloat;
begin
  n := Length(A);
  SetLength(L, n, n);
  for i := 0 to n - 1 do
    for j := 0 to n - 1 do L[i][j] := 0;

  // Cholesky factorisation A = L L^T (lower triangular L)
  for j := 0 to n - 1 do
  begin
    Sum := A[j][j];
    for k := 0 to j - 1 do Sum := Sum - L[j][k] * L[j][k];
    L[j][j] := Sqrt(Sum);
    for i := j + 1 to n - 1 do
    begin
      Sum := A[i][j];
      for k := 0 to j - 1 do Sum := Sum - L[i][k] * L[j][k];
      L[i][j] := Sum / L[j][j];
    end;
  end;

  SetLength(Inv, n, n);
  SetLength(Y, n);
  SetLength(Col, n);
  // Solve A * x = e_c for each unit column e_c -> column c of the inverse.
  for k := 0 to n - 1 do
  begin
    // forward solve L y = e_k
    for i := 0 to n - 1 do
    begin
      Sum := 0;
      if i = k then Sum := 1;
      for j := 0 to i - 1 do Sum := Sum - L[i][j] * Y[j];
      Y[i] := Sum / L[i][i];
    end;
    // back solve L^T x = y
    for i := n - 1 downto 0 do
    begin
      Sum := Y[i];
      for j := i + 1 to n - 1 do Sum := Sum - L[j][i] * Col[j];
      Col[i] := Sum / L[i][i];
    end;
    for i := 0 to n - 1 do Inv[i][k] := Col[i];
  end;
end;

procedure RunDemo();
var
  NN: TNNet;
  X, Y: TNNetVolume;
  Epoch, I, C, D, E, S, N, Idx: integer;
  Loss, Diff: TNeuralFloat;
  // class-conditional Gaussian parameters
  Mu: array[0..cNumClasses - 1] of TFeat;
  ClassCount: array[0..cNumClasses - 1] of integer;
  Sigma, SigmaInv: TMatrix;
  Feat: TFeat;
  Diffv: array[0..cNumClasses - 1] of TFeat; // reused buffer
  // scores
  PosScores: array of TNeuralFloat;
  NegScores: array of TNeuralFloat;
  Score, BestScore, Quad, TotalN: TNeuralFloat;
  Auc: TNeuralFloat;
  MeanPos, MeanNeg: TNeuralFloat;
  Tmp: array[0..cFeatDim - 1] of TNeuralFloat;

  // read the penultimate features of the LAST forward pass into F
  procedure ReadFeatures(out F: TFeat);
  var d2: integer;
  begin
    for d2 := 0 to cFeatDim - 1 do
      F[d2] := NN.Layers[cPenultimateIdx].Output.FData[d2];
  end;

begin
  RandSeed := 424242;
  BuildNet(NN);
  X := TNNetVolume.Create(cInputDim, 1, 1);
  Y := TNNetVolume.Create(cNumClasses, 1, 1);
  try
    NN.SetLearningRate(cLearningRate, 0.9);

    WriteLn('Architecture (penultimate feature layer = index ',
      cPenultimateIdx, ', width ', cFeatDim, '):');
    NN.PrintSummary();
    WriteLn;
    WriteLn('In-distribution classes=', cNumClasses,
      '  train/class=', cTrainPerClass, '  test/class=', cTestPerClass,
      '  OOD points=', cOODCount);
    WriteLn;

    // ----------------------------- TRAIN ---------------------------------
    WriteLn('Training the classifier for ', cEpochs, ' epochs...');
    for Epoch := 1 to cEpochs do
    begin
      Loss := 0;
      for I := 1 to cTrainPerClass * cNumClasses do
      begin
        C := Random(cNumClasses);
        MakeInSample(C, X, Y);
        NN.Compute(X);
        NN.Backpropagate(Y);
        Diff := -Ln(Max(1e-9, NN.GetLastLayer().Output.FData[C]));
        Loss := Loss + Diff;
      end;
      if (Epoch mod 20 = 0) or (Epoch = 1) then
        WriteLn('  epoch ', Epoch:4, '  mean_nll=',
          (Loss / (cTrainPerClass * cNumClasses)):8:5);
    end;
    WriteLn;

    // -- FIT class-conditional Gaussians on FROZEN penultimate features ----
    // mu_c = mean feature per class; Sigma = pooled (tied) covariance.
    for C := 0 to cNumClasses - 1 do
    begin
      ClassCount[C] := 0;
      for D := 0 to cFeatDim - 1 do Mu[C][D] := 0;
    end;
    SetLength(Sigma, cFeatDim, cFeatDim);
    for D := 0 to cFeatDim - 1 do
      for E := 0 to cFeatDim - 1 do Sigma[D][E] := 0;

    // Pass 1: per-class means. Re-seed so the SAME training points are
    // regenerated deterministically (forward-only, no backprop = frozen).
    RandSeed := 989898;
    for C := 0 to cNumClasses - 1 do
      for S := 1 to cTrainPerClass do
      begin
        MakeInSample(C, X, Y);
        NN.Compute(X);
        ReadFeatures(Feat);
        for D := 0 to cFeatDim - 1 do Mu[C][D] := Mu[C][D] + Feat[D];
        Inc(ClassCount[C]);
      end;
    for C := 0 to cNumClasses - 1 do
      for D := 0 to cFeatDim - 1 do
        Mu[C][D] := Mu[C][D] / ClassCount[C];

    // Pass 2: pooled covariance Sigma = (1/N) sum_c sum_i (f-mu_c)(f-mu_c)^T.
    RandSeed := 989898;
    TotalN := 0;
    for C := 0 to cNumClasses - 1 do
      for S := 1 to cTrainPerClass do
      begin
        MakeInSample(C, X, Y);
        NN.Compute(X);
        ReadFeatures(Feat);
        for D := 0 to cFeatDim - 1 do Tmp[D] := Feat[D] - Mu[C][D];
        for D := 0 to cFeatDim - 1 do
          for E := 0 to cFeatDim - 1 do
            Sigma[D][E] := Sigma[D][E] + Tmp[D] * Tmp[E];
        TotalN := TotalN + 1;
      end;
    for D := 0 to cFeatDim - 1 do
      for E := 0 to cFeatDim - 1 do
        Sigma[D][E] := Sigma[D][E] / TotalN;

    // ridge: +cRidge on the diagonal so Sigma is well-conditioned SPD.
    for D := 0 to cFeatDim - 1 do Sigma[D][D] := Sigma[D][D] + cRidge;

    CholeskyInverse(Sigma, SigmaInv);

    // ------------------------ SCORE held-in test -------------------------
    // score(x) = max_c -(f-mu_c)^T SigmaInv (f-mu_c)
    SetLength(PosScores, cTestPerClass * cNumClasses);
    RandSeed := 555111;
    N := 0;
    for C := 0 to cNumClasses - 1 do
      for S := 1 to cTestPerClass do
      begin
        MakeInSample(C, X, Y);
        NN.Compute(X);
        ReadFeatures(Feat);
        BestScore := -Infinity;
        for E := 0 to cNumClasses - 1 do
        begin
          for D := 0 to cFeatDim - 1 do Diffv[E][D] := Feat[D] - Mu[E][D];
          Quad := 0;
          for D := 0 to cFeatDim - 1 do
            for Idx := 0 to cFeatDim - 1 do
              Quad := Quad + Diffv[E][D] * SigmaInv[D][Idx] * Diffv[E][Idx];
          Score := -Quad;
          if Score > BestScore then BestScore := Score;
        end;
        PosScores[N] := BestScore;
        Inc(N);
      end;

    // ----------------------------- SCORE OOD -----------------------------
    SetLength(NegScores, cOODCount);
    for I := 0 to cOODCount - 1 do
    begin
      MakeOODSample(X);
      NN.Compute(X);
      ReadFeatures(Feat);
      BestScore := -Infinity;
      for E := 0 to cNumClasses - 1 do
      begin
        for D := 0 to cFeatDim - 1 do Diffv[E][D] := Feat[D] - Mu[E][D];
        Quad := 0;
        for D := 0 to cFeatDim - 1 do
          for Idx := 0 to cFeatDim - 1 do
            Quad := Quad + Diffv[E][D] * SigmaInv[D][Idx] * Diffv[E][Idx];
        Score := -Quad;
        if Score > BestScore then BestScore := Score;
      end;
      NegScores[I] := BestScore;
    end;

    // ------------------------------- REPORT ------------------------------
    MeanPos := 0; for I := 0 to High(PosScores) do MeanPos := MeanPos + PosScores[I];
    MeanPos := MeanPos / Length(PosScores);
    MeanNeg := 0; for I := 0 to High(NegScores) do MeanNeg := MeanNeg + NegScores[I];
    MeanNeg := MeanNeg / Length(NegScores);

    Auc := AUROC(PosScores, NegScores);

    WriteLn('Mahalanobis OOD detection (Lee et al. 2018, tied covariance)');
    WriteLn('Feature layer idx=', cPenultimateIdx, '  dim=', cFeatDim,
      '  ridge=', cRidge:0:4);
    WriteLn(StringOfChar('=', 64));
    WriteLn('held-in test (positives) n=', Length(PosScores),
      '   mean score = ', MeanPos:12:4);
    WriteLn('OOD          (negatives) n=', Length(NegScores),
      '   mean score = ', MeanNeg:12:4);
    WriteLn(StringOfChar('-', 64));
    WriteLn('AUROC (held-in vs OOD, score = -d_Mahalanobis^2) = ', Auc:0:4);
    WriteLn(StringOfChar('=', 64));

    if MeanPos <= MeanNeg then
      WriteLn('WARN: held-in mean score not above OOD mean score');

    if Auc > cMinAUROC then
      WriteLn('PASS: AUROC ', Auc:0:4, ' > ', cMinAUROC:0:2,
        ' (Mahalanobis cleanly separates in-dist from OOD)')
    else
    begin
      WriteLn('FAIL: AUROC ', Auc:0:4, ' <= ', cMinAUROC:0:2);
      Halt(1);
    end;
  finally
    Y.Free;
    X.Free;
    NN.Free;
  end;
end;

begin
  RunDemo();
end.
