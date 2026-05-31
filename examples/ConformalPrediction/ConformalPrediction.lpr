program ConformalPrediction;
(*
ConformalPrediction: split (inductive) conformal prediction on a small
softmax classifier. Trains a tiny MLP on a synthetic multi-class 2D
Gaussian-blob dataset, then wraps the FROZEN model in a guaranteed-coverage
SET predictor using the LAC / threshold nonconformity score
(s = 1 - softmax[true_class]); Vovk; Angelopoulos & Bates 2021.

Unlike CalibrationReport / MarginReport / DeepEnsembleUncertainty /
MCDropoutUncertainty, which emit a SCALAR / point confidence per input,
this emits a variable-size LABEL SET per input with a finite-sample,
distribution-free marginal-coverage GUARANTEE: P(true label in set) >= 1-alpha.

Pipeline:
  TRAIN  -> fit the softmax classifier
  CALIB  -> nonconformity scores s_i = 1 - softmax[true_i]
            qhat = the ceil((n+1)(1-alpha))/n empirical quantile of {s_i}
  TEST   -> predict the set { k : 1 - softmax[k] <= qhat }

Built-in correctness signals (PASS/FAIL, Halt(1) on hard failure):
  * empirical TEST coverage >= 1-alpha - slack across alpha in
    {0.01, 0.05, 0.10, 0.20}  (the conformal guarantee);
  * mean set size shrinks monotonically as alpha grows.

Pure CPU, single-threaded manual training loop, runs in a few seconds.

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
  cNumClasses     = 5;
  cTrainPerClass  = 300;
  cCalibPerClass  = 200;   // calibration split -> n = 1000
  cTestPerClass   = 800;   // test split -> 4000 points (band is meaningful)
  cEpochs         = 80;
  cLearningRate   = 0.04;
  cSlack          = 0.06;  // finite-sample coverage tolerance band

  // Five 2D Gaussian blobs with deliberate overlap so the model is genuinely
  // uncertain on some points (otherwise every set is a singleton and the
  // coverage guarantee is trivial).
  cCenters: array[0..cNumClasses - 1, 0..1] of TNeuralFloat =
    ((-1.8, -1.8),
     ( 1.8, -1.8),
     ( 1.8,  1.8),
     (-1.8,  1.8),
     ( 0.0,  0.0));
  cSigma: array[0..cNumClasses - 1] of TNeuralFloat =
    (1.05, 1.05, 1.05, 1.05, 1.25);

  cAlphas: array[0..3] of TNeuralFloat = (0.01, 0.05, 0.10, 0.20);

procedure BuildNet(out NN: TNNet);
begin
  NN := TNNet.Create();
  NN.AddLayer(TNNetInput.Create(2, 1, 1));
  NN.AddLayer(TNNetFullConnectReLU.Create(24));
  NN.AddLayer(TNNetFullConnectReLU.Create(24));
  NN.AddLayer(TNNetFullConnectLinear.Create(cNumClasses));
  NN.AddLayer(TNNetSoftMax.Create());
end;

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

procedure MakeSample(ClassId: integer; out X, Y: TNNetVolume);
begin
  X := TNNetVolume.Create(2, 1, 1);
  Y := TNNetVolume.Create(cNumClasses, 1, 1);
  X.FData[0] := cCenters[ClassId][0] + RandomGauss() * cSigma[ClassId];
  X.FData[1] := cCenters[ClassId][1] + RandomGauss() * cSigma[ClassId];
  Y.Fill(0);
  Y.FData[ClassId] := 1.0;
end;

procedure BuildSet(out Pairs: TNNetVolumePairList; PerClass: integer);
var
  C, I: integer;
  X, Y: TNNetVolume;
begin
  Pairs := TNNetVolumePairList.Create();
  for C := 0 to cNumClasses - 1 do
    for I := 1 to PerClass do
    begin
      MakeSample(C, X, Y);
      Pairs.Add(TNNetVolumePair.Create(X, Y));
    end;
end;

// The ceil((n+1)(1-alpha))/n empirical quantile of a SORTED ascending array.
// Returns +inf when the rank exceeds n (set becomes all labels -> trivial
// but valid coverage).
function ConformalQuantile(const SortedScores: array of TNeuralFloat;
  n: integer; Alpha: TNeuralFloat): TNeuralFloat;
var
  Rank: integer;
begin
  // 1-based rank; ceil((n+1)(1-alpha)).
  Rank := Ceil((n + 1) * (1.0 - Alpha));
  if Rank > n then
    Result := Infinity      // edge case: qhat = +inf, set = all labels
  else
    Result := SortedScores[Rank - 1];
end;

// Simple ascending insertion-free sort (n ~ 1000, called once).
procedure SortAsc(var A: array of TNeuralFloat; n: integer);
var
  I, J: integer;
  Tmp: TNeuralFloat;
begin
  for I := 1 to n - 1 do
  begin
    Tmp := A[I];
    J := I - 1;
    while (J >= 0) and (A[J] > Tmp) do
    begin
      A[J + 1] := A[J];
      Dec(J);
    end;
    A[J + 1] := Tmp;
  end;
end;

procedure RunDemo();
var
  NN: TNNet;
  TrainSet, CalibSet, TestSet: TNNetVolumePairList;
  Epoch, I, K, A, TrueK: integer;
  Pair: TNNetVolumePair;
  Loss, Diff, QHat, Prob: TNeuralFloat;
  CalScores: array of TNeuralFloat;
  // per-alpha aggregates
  CovHits, SetSizeSum, Singletons, Empties: array[0..High(cAlphas)] of integer;
  Covered: boolean;
  ThisSetSize: integer;
  EmpCov, MeanSet, PrevMeanSet, TargetCov: TNeuralFloat;
  AllOk, MonoOk: boolean;
begin
  RandSeed := 424242;
  BuildNet(NN);
  BuildSet(TrainSet, cTrainPerClass);
  BuildSet(CalibSet, cCalibPerClass);
  BuildSet(TestSet,  cTestPerClass);
  try
    NN.SetLearningRate(cLearningRate, 0.9);

    WriteLn('Architecture:');
    NN.PrintSummary();
    WriteLn;

    WriteLn('Splits: train=', TrainSet.Count, '  calib=', CalibSet.Count,
            '  test=', TestSet.Count, '  classes=', cNumClasses);
    WriteLn;

    WriteLn('Training for ', cEpochs, ' epochs...');
    for Epoch := 1 to cEpochs do
    begin
      Loss := 0;
      for I := 0 to TrainSet.Count - 1 do
      begin
        Pair := TrainSet[I];
        NN.Compute(Pair.I);
        NN.Backpropagate(Pair.O);
        Diff := -Ln(Max(1e-9,
          NN.GetLastLayer().Output.FData[Pair.O.GetClass()]));
        Loss := Loss + Diff;
      end;
      if (Epoch mod 20 = 0) or (Epoch = 1) then
        WriteLn('  epoch ', Epoch:4, '  mean_nll=',
          (Loss / TrainSet.Count):8:5);
    end;
    WriteLn;

    // -------- CALIBRATION: nonconformity scores s = 1 - softmax[true] -------
    SetLength(CalScores, CalibSet.Count);
    for I := 0 to CalibSet.Count - 1 do
    begin
      Pair := CalibSet[I];
      NN.Compute(Pair.I);
      TrueK := Pair.O.GetClass();
      CalScores[I] := 1.0 - NN.GetLastLayer().Output.FData[TrueK];
    end;
    SortAsc(CalScores, CalibSet.Count);

    // -------- TEST: emit set { k : 1 - softmax[k] <= qhat } per alpha -------
    for A := 0 to High(cAlphas) do
    begin
      CovHits[A] := 0; SetSizeSum[A] := 0;
      Singletons[A] := 0; Empties[A] := 0;
    end;

    for I := 0 to TestSet.Count - 1 do
    begin
      Pair := TestSet[I];
      NN.Compute(Pair.I);
      TrueK := Pair.O.GetClass();
      for A := 0 to High(cAlphas) do
      begin
        QHat := ConformalQuantile(CalScores, CalibSet.Count, cAlphas[A]);
        Covered := False;
        ThisSetSize := 0;
        for K := 0 to cNumClasses - 1 do
        begin
          Prob := NN.GetLastLayer().Output.FData[K];
          if (1.0 - Prob) <= QHat then
          begin
            Inc(ThisSetSize);
            if K = TrueK then Covered := True;
          end;
        end;
        SetSizeSum[A] := SetSizeSum[A] + ThisSetSize;
        if Covered then Inc(CovHits[A]);
        if ThisSetSize = 1 then Inc(Singletons[A]);
        if ThisSetSize = 0 then Inc(Empties[A]);
      end;
    end;

    // ------------------------------- report --------------------------------
    WriteLn('Split-conformal prediction sets (LAC / threshold score)');
    WriteLn('Calibration n=', CalibSet.Count, '  Test n=', TestSet.Count);
    WriteLn(StringOfChar('=', 78));
    WriteLn('alpha | target-cov | emp-cov | mean-set | singleton% | empty%');
    WriteLn(StringOfChar('-', 78));

    AllOk := True;
    MonoOk := True;
    PrevMeanSet := Infinity; // sets must shrink as alpha grows
    for A := 0 to High(cAlphas) do
    begin
      TargetCov := 1.0 - cAlphas[A];
      EmpCov  := CovHits[A] / TestSet.Count;
      MeanSet := SetSizeSum[A] / TestSet.Count;
      WriteLn(
        cAlphas[A]:5:2, ' | ',
        TargetCov:10:3, ' | ',
        EmpCov:7:3, ' | ',
        MeanSet:8:3, ' | ',
        (100.0 * Singletons[A] / TestSet.Count):9:1, '% | ',
        (100.0 * Empties[A] / TestSet.Count):5:1, '%');

      // GUARANTEE: empirical coverage >= 1 - alpha - slack.
      if EmpCov < (TargetCov - cSlack) then
      begin
        WriteLn('  FAIL: coverage ', EmpCov:6:3, ' below target ',
          TargetCov:6:3, ' - slack ', cSlack:5:3, ' at alpha=', cAlphas[A]:5:2);
        AllOk := False;
      end;

      // Mean set size must be monotonically non-increasing in alpha.
      if MeanSet > PrevMeanSet + 1e-9 then
      begin
        WriteLn('  FAIL: mean set size not monotone at alpha=', cAlphas[A]:5:2,
          ' (', MeanSet:6:3, ' > ', PrevMeanSet:6:3, ')');
        MonoOk := False;
      end;
      PrevMeanSet := MeanSet;
    end;
    WriteLn(StringOfChar('=', 78));

    if AllOk then
      WriteLn('PASS: marginal coverage >= 1-alpha-', cSlack:4:2,
        ' across all alpha (the conformal guarantee holds)')
    else
      WriteLn('FAIL: coverage guarantee violated');

    if MonoOk then
      WriteLn('PASS: mean set size shrinks monotonically as alpha grows')
    else
      WriteLn('FAIL: mean set size not monotone in alpha');

    if not (AllOk and MonoOk) then
      Halt(1);
  finally
    TestSet.Free;
    CalibSet.Free;
    TrainSet.Free;
    NN.Free;
  end;
end;

begin
  RunDemo();
end.
