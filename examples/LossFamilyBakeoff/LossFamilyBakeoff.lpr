program LossFamilyBakeoff;
(*
LossFamilyBakeoff: head-to-head comparison of robust regression loss
heads on a noisy hypotenuse task y = sqrt(X^2 + Y^2) with deliberately
injected OUTLIERS in the TRAINING targets.

Five arms share the SAME trunk / seed / learning rate / epochs and
differ ONLY in the output loss head:
  A) MSE baseline      (plain TNNetFullConnectLinear, default L2/MSE)
  B) TNNetHuberLoss
  C) TNNetSmoothL1Loss
  D) TNNetCharbonnierLoss
  E) TNNetLogCoshLoss

The TRAINING targets get Gaussian noise plus a fraction of large
additive outliers. The held-out TEST set is CLEAN (no noise, no
outliers). The headline: robust losses should beat plain MSE on the
clean test set because MSE over-weights squared outlier residuals.

Tiny MLP 2 -> 32 -> 32 -> 1, fixed RandSeed, single thread for
determinism. Prints a table of clean-test MSE, clean-test MAE and
final training loss per arm.

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
  neuralvolume,
  neuralfit;

type
  THeadKind = (hkMSE, hkHuber, hkSmoothL1, hkCharbonnier, hkLogCosh);

  TRunResult = record
    Name: string;
    TestMSE: TNeuralFloat;
    TestMAE: TNeuralFloat;
    FinalTrainLoss: TNeuralFloat;
  end;

const
  INPUT_SCALE     = 100.0;  // inputs in [0,1)
  // Targets are scaled so that CLEAN residuals stay small (well inside the
  // quadratic region of Huber/SmoothL1, |r| << delta=1) while injected
  // OUTLIER residuals land far outside it (|r| >> 1). That is exactly the
  // regime where the robust losses diverge from plain MSE.
  TARGET_SCALE    = 40.0;   // hypotenuse max ~141 -> target in [0, ~3.5]
  TRAIN_SIZE      = 1000;
  TEST_SIZE       = 500;
  HIDDEN_UNITS    = 32;
  NUM_EPOCHS      = 200;
  BATCH_SIZE      = 32;
  SEED            = 42;
  NOISE_SIGMA     = 4.0;    // Gaussian noise on train targets (orig units)
  OUTLIER_FRAC    = 0.10;   // 10% of train samples corrupted
  OUTLIER_MAG     = 80.0;   // big additive corruption (orig units, |r|~2 norm)
  LEARNING_RATE   = 0.001;

  function RandG01(): TNeuralFloat;
  // Box-Muller standard normal.
  var
    U1, U2: TNeuralFloat;
  begin
    U1 := Random;
    if U1 < 1e-12 then U1 := 1e-12;
    U2 := Random;
    Result := Sqrt(-2.0 * Ln(U1)) * Cos(2.0 * Pi * U2);
  end;

  // Clean pairs: exact hypotenuse, no noise.
  function CreateCleanPairList(MaxCnt: integer): TNNetVolumePairList;
  var
    Cnt: integer;
    LocalX, LocalY, Hypotenuse: TNeuralFloat;
  begin
    Result := TNNetVolumePairList.Create();
    for Cnt := 1 to MaxCnt do
    begin
      LocalX := Random(100);
      LocalY := Random(100);
      Hypotenuse := sqrt(LocalX*LocalX + LocalY*LocalY);
      Result.Add(
        TNNetVolumePair.Create(
          TNNetVolume.Create([LocalX / INPUT_SCALE, LocalY / INPUT_SCALE]),
          TNNetVolume.Create([Hypotenuse / TARGET_SCALE])
        )
      );
    end;
  end;

  // Training pairs: Gaussian noise on every target, plus large additive
  // outliers on a fraction of samples.
  function CreateNoisyOutlierPairList(MaxCnt: integer): TNNetVolumePairList;
  var
    Cnt: integer;
    LocalX, LocalY, Hypotenuse, Target, Sign: TNeuralFloat;
  begin
    Result := TNNetVolumePairList.Create();
    for Cnt := 1 to MaxCnt do
    begin
      LocalX := Random(100);
      LocalY := Random(100);
      Hypotenuse := sqrt(LocalX*LocalX + LocalY*LocalY);
      Target := Hypotenuse + NOISE_SIGMA * RandG01();
      if Random < OUTLIER_FRAC then
      begin
        if Random < 0.5 then Sign := 1.0 else Sign := -1.0;
        Target := Target + Sign * OUTLIER_MAG;
      end;
      Result.Add(
        TNNetVolumePair.Create(
          TNNetVolume.Create([LocalX / INPUT_SCALE, LocalY / INPUT_SCALE]),
          TNNetVolume.Create([Target / TARGET_SCALE])
        )
      );
    end;
  end;

  function LocalFloatCompare(A, B: TNNetVolume; ThreadId: integer): boolean;
  begin
    Result := ( Abs(A.FData[0]-B.FData[0]) < 0.02 );
  end;

  function HeadName(Kind: THeadKind): string;
  begin
    case Kind of
      hkMSE:         Result := 'MSE-baseline';
      hkHuber:       Result := 'HuberLoss';
      hkSmoothL1:    Result := 'SmoothL1Loss';
      hkCharbonnier: Result := 'CharbonnierLoss';
      hkLogCosh:     Result := 'LogCoshLoss';
    end;
  end;

  // Clean-test error metrics, reported in ORIGINAL target units.
  procedure EvaluateClean(NN: TNNet; Pairs: TNNetVolumePairList;
    out MSE, MAE: TNeuralFloat);
  var
    I: integer;
    Pred, Diff: TNeuralFloat;
    SumSq, SumAbs: Double;
  begin
    SumSq := 0;
    SumAbs := 0;
    for I := 0 to Pairs.Count - 1 do
    begin
      NN.Compute(Pairs[I].I);
      Pred := NN.GetLastLayer().Output.FData[0];
      Diff := (Pred - Pairs[I].O.FData[0]) * TARGET_SCALE;
      SumSq := SumSq + Sqr(Diff);
      SumAbs := SumAbs + Abs(Diff);
    end;
    if Pairs.Count > 0 then
    begin
      MSE := SumSq / Pairs.Count;
      MAE := SumAbs / Pairs.Count;
    end
    else
    begin
      MSE := 0;
      MAE := 0;
    end;
  end;

  function RunOne(Kind: THeadKind;
    Train, Test: TNNetVolumePairList): TRunResult;
  var
    NN: TNNet;
    NFit: TNeuralFit;
  begin
    Result.Name := HeadName(Kind);

    NN := TNNet.Create();
    NFit := TNeuralFit.Create();
    try
      NN.AddLayer([
        TNNetInput.Create(2),
        TNNetFullConnectReLU.Create(HIDDEN_UNITS),
        TNNetFullConnectReLU.Create(HIDDEN_UNITS),
        TNNetFullConnectLinear.Create(1)
      ]);
      case Kind of
        hkHuber:       NN.AddLayer(TNNetHuberLoss.Create());
        hkSmoothL1:    NN.AddLayer(TNNetSmoothL1Loss.Create());
        hkCharbonnier: NN.AddLayer(TNNetCharbonnierLoss.Create());
        hkLogCosh:     NN.AddLayer(TNNetLogCoshLoss.Create());
        hkMSE:         ; // no loss head: default L2/MSE backprop
      end;

      NFit.InitialLearningRate := LEARNING_RATE;
      NFit.LearningRateDecay := 0;
      NFit.L2Decay := 0;
      NFit.Verbose := false;
      NFit.HasFlipX := false;
      NFit.HasFlipY := false;
      NFit.MaxThreadNum := 1;
      NFit.InferHitFn := @LocalFloatCompare;
      NFit.HideMessages();
      // Train and validate on the noisy/outlier set; the clean test set is
      // only used for the final report.
      NFit.Fit(NN, Train, Train, nil, BATCH_SIZE, NUM_EPOCHS);

      Result.FinalTrainLoss := NFit.CurrentTrainingError;
      EvaluateClean(NN, Test, Result.TestMSE, Result.TestMAE);
    finally
      NFit.Free;
      NN.Free;
    end;
  end;

  procedure RunAlgo();
  var
    TrainPairs, TestPairs: TNNetVolumePairList;
    Kind: THeadKind;
    Results: array[THeadKind] of TRunResult;
    StartTime, EndTime: TDateTime;
  begin
    WriteLn('Loss-family bake-off on noisy hypotenuse y = sqrt(X^2 + Y^2).');
    WriteLn('Net: 2 -> ', HIDDEN_UNITS, ' -> ', HIDDEN_UNITS,
            ' -> 1 (ReLU), ', NUM_EPOCHS, ' epochs, ', TRAIN_SIZE,
            ' train pairs, LR=', FloatToStrF(LEARNING_RATE, ffFixed, 6, 4),
            ', seed=', SEED, '.');
    WriteLn('Train targets: Gaussian noise sigma=', NOISE_SIGMA:0:1,
            ' + ', Round(OUTLIER_FRAC*100), '% outliers of magnitude +/-',
            OUTLIER_MAG:0:0, ' (original units).');
    WriteLn('Test set (', TEST_SIZE, ' pairs) is CLEAN: no noise, no outliers.');
    WriteLn('All arms share trunk/seed/LR/epochs; only the loss head differs.');
    WriteLn;

    StartTime := Now;
    for Kind := Low(THeadKind) to High(THeadKind) do
    begin
      // Reseed before data generation and before each fit so every arm
      // sees IDENTICAL data and IDENTICAL weight initialization.
      RandSeed := SEED;
      TrainPairs := CreateNoisyOutlierPairList(TRAIN_SIZE);
      TestPairs  := CreateCleanPairList(TEST_SIZE);
      try
        RandSeed := SEED;
        Write('Training ', HeadName(Kind), ' ...');
        Results[Kind] := RunOne(Kind, TrainPairs, TestPairs);
        WriteLn(' done.');
      finally
        TestPairs.Free;
        TrainPairs.Free;
      end;
    end;
    EndTime := Now;

    WriteLn;
    WriteLn('=== Results (clean-test error, original units) ===');
    WriteLn('loss_head        clean_test_MSE  clean_test_MAE  final_train_loss');
    for Kind := Low(THeadKind) to High(THeadKind) do
    begin
      WriteLn(
        Format('%-15s  %14s  %14s  %16s', [
          Results[Kind].Name,
          FloatToStrF(Results[Kind].TestMSE, ffFixed, 8, 4),
          FloatToStrF(Results[Kind].TestMAE, ffFixed, 8, 4),
          FloatToStrF(Results[Kind].FinalTrainLoss, ffFixed, 8, 6)
        ]));
    end;
    WriteLn;
    WriteLn('CSV:');
    WriteLn('loss_head,clean_test_mse,clean_test_mae,final_train_loss');
    for Kind := Low(THeadKind) to High(THeadKind) do
      WriteLn(Results[Kind].Name, ',',
              FloatToStrF(Results[Kind].TestMSE, ffFixed, 8, 4), ',',
              FloatToStrF(Results[Kind].TestMAE, ffFixed, 8, 4), ',',
              FloatToStrF(Results[Kind].FinalTrainLoss, ffFixed, 8, 6));
    WriteLn;
    WriteLn('Total wall time: ', FormatFloat('0.00', (EndTime - StartTime)*86400), ' s');
  end;

var
  // Stops Lazarus errors
  Application: record Title: string; end;

begin
  Application.Title := 'Loss Family Bakeoff';
  RandSeed := SEED;
  RunAlgo();
end.
