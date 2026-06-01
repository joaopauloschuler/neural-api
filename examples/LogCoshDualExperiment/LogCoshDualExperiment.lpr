program LogCoshDualExperiment;
(*
LogCoshDualExperiment: pairs TNNetLogCoshActivation (hidden) with
TNNetLogCoshLoss (output head) vs. a plain MSE head on the toy
hypotenuse task y = sqrt(X^2 + Y^2).

Two configurations, fixed RandSeed for reproducibility:
  A) LogCosh hidden + plain MSE head      (no explicit loss layer)
  B) LogCosh hidden + TNNetLogCoshLoss    (appended after linear output)

Tiny MLP 2 -> 32 -> 1, 50 epochs, 1000 training pairs, batch size 32.
Prints a CSV-style table of final validation MSE (in original target
units) and epochs-to-converge (MSE < 5 on the original hypotenuse
scale).

Copyright (C) 2026 Joao Paulo Schwarz Schuler

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
any later version.
*)

{$mode objfpc}{$H+}

uses {$IFDEF UNIX} {$IFDEF UseCThreads}
  cthreads, {$ENDIF} {$ENDIF}
  Classes, SysUtils,
  neuralnetwork,
  neuralvolume,
  neuralfit;

type
  THeadKind = (hkPlainMSE, hkLogCoshLoss);

  // Regression statistics, all in the ORIGINAL target scale (the network
  // output is multiplied back by TARGET_SCALE before measuring) except for
  // LogCosh, which is the mean log(cosh(residual)) on the normalized scale
  // the network actually optimizes.
  TEvalStats = record
    MSE: TNeuralFloat;     // mean squared error
    RMSE: TNeuralFloat;    // sqrt(MSE)
    MAE: TNeuralFloat;     // mean absolute error
    MaxAbsErr: TNeuralFloat; // worst single-sample absolute error
    R2: TNeuralFloat;      // coefficient of determination
    LogCosh: TNeuralFloat; // mean log(cosh(residual)) on normalized scale
  end;

  TRunResult = record
    Name: string;
    Val: TEvalStats;
    Test: TEvalStats;
    EpochsToConverge: integer; // -1 if never converged
    Epochs: integer;
    Seconds: TNeuralFloat;     // per-config training wall time
  end;

const
  INPUT_SCALE        = 100.0;
  TARGET_SCALE       = 200.0;
  CONVERGE_THRESHOLD = 5.0;  // MSE on original target scale
  TRAIN_SIZE         = 1000;
  VAL_SIZE           = 200;
  TEST_SIZE          = 200;
  HIDDEN_UNITS       = 32;
  NUM_EPOCHS         = 50;
  BATCH_SIZE         = 32;

type
  TConvergenceTracker = class
  public
    ConvergedAtEpoch: integer;
    NN: TNNet;
    Validation: TNNetVolumePairList;
    procedure OnAfterEpoch(Sender: TObject);
  end;

function EvaluateMSEHelper(NN: TNNet; Pairs: TNNetVolumePairList): TNeuralFloat; forward;
function ComputeStats(NN: TNNet; Pairs: TNNetVolumePairList): TEvalStats; forward;

procedure TConvergenceTracker.OnAfterEpoch(Sender: TObject);
var
  Fit: TNeuralFit;
  Mse: TNeuralFloat;
begin
  Fit := Sender as TNeuralFit;
  if NN = nil then Exit;
  Mse := EvaluateMSEHelper(NN, Validation);
  if (ConvergedAtEpoch < 0) and (Mse < CONVERGE_THRESHOLD) then
    ConvergedAtEpoch := Fit.CurrentEpoch;
end;

var
  GTracker: TConvergenceTracker;

function CreateHypotenusePairList(MaxCnt: integer): TNNetVolumePairList;
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

function LocalFloatCompare(A, B: TNNetVolume; ThreadId: integer): boolean;
begin
  Result := ( Abs(A.FData[0]-B.FData[0])<0.005 );
end;

function HeadName(Kind: THeadKind): string;
begin
  case Kind of
    hkPlainMSE:    Result := 'LogCoshHidden+PlainMSE';
    hkLogCoshLoss: Result := 'LogCoshHidden+LogCoshLoss';
  end;
end;

function EvaluateMSEHelper(NN: TNNet; Pairs: TNNetVolumePairList): TNeuralFloat;
begin
  // Thin wrapper kept for the convergence tracker; MSE is in original scale.
  Result := ComputeStats(NN, Pairs).MSE;
end;

function ComputeStats(NN: TNNet; Pairs: TNNetVolumePairList): TEvalStats;
var
  I: integer;
  PredN, TgtN, Pred, Tgt, Err, ResidN: TNeuralFloat;
  SumSq, SumAbs, SumLogCosh, SumTgt, SumTgtSq, SSRes, SSTot, MeanTgt: Double;
  N: integer;
begin
  FillChar(Result, SizeOf(Result), 0);
  N := Pairs.Count;
  if N = 0 then Exit;

  SumSq := 0; SumAbs := 0; SumLogCosh := 0; SumTgt := 0; SumTgtSq := 0;
  Result.MaxAbsErr := 0;
  for I := 0 to N - 1 do
  begin
    NN.Compute(Pairs[I].I);
    PredN := NN.GetLastLayer().Output.FData[0];  // normalized prediction
    TgtN  := Pairs[I].O.FData[0];                 // normalized target
    ResidN := PredN - TgtN;

    Pred := PredN * TARGET_SCALE;                 // original scale
    Tgt  := TgtN  * TARGET_SCALE;
    Err  := Pred - Tgt;

    SumSq      := SumSq + Sqr(Err);
    SumAbs     := SumAbs + Abs(Err);
    // log(cosh x) computed stably as |x| + log1p(exp(-2|x|)) - log(2).
    SumLogCosh := SumLogCosh + (Abs(ResidN) + Ln(1 + Exp(-2*Abs(ResidN))) - Ln(2));
    SumTgt     := SumTgt + Tgt;
    SumTgtSq   := SumTgtSq + Sqr(Tgt);
    if Abs(Err) > Result.MaxAbsErr then Result.MaxAbsErr := Abs(Err);
  end;

  Result.MSE     := SumSq / N;
  Result.RMSE    := Sqrt(Result.MSE);
  Result.MAE     := SumAbs / N;
  Result.LogCosh := SumLogCosh / N;

  // R^2 = 1 - SSres/SStot, with SStot the variance of the targets.
  MeanTgt := SumTgt / N;
  SSRes   := SumSq;
  SSTot   := SumTgtSq - N * Sqr(MeanTgt);
  if SSTot > 0 then
    Result.R2 := 1 - SSRes / SSTot
  else
    Result.R2 := 0;
end;

function RunOne(Kind: THeadKind;
                Train, Validation, Test: TNNetVolumePairList): TRunResult;
var
  NN: TNNet;
  NFit: TNeuralFit;
  StartTime, EndTime: TDateTime;
begin
  Result.Name := HeadName(Kind);
  Result.Epochs := NUM_EPOCHS;
  GTracker.ConvergedAtEpoch := -1;

  NN := TNNet.Create();
  NFit := TNeuralFit.Create();
  try
    StartTime := Now;
    NN.AddLayer(TNNetInput.Create(2));
    NN.AddLayer(TNNetFullConnectLinear.Create(HIDDEN_UNITS));
    NN.AddLayer(TNNetLogCoshActivation.Create());
    NN.AddLayer(TNNetFullConnectLinear.Create(1));
    if Kind = hkLogCoshLoss then
      NN.AddLayer(TNNetLogCoshLoss.Create());

    NFit.InitialLearningRate := 0.01;
    NFit.LearningRateDecay := 0;
    NFit.L2Decay := 0;
    NFit.Verbose := false;
    NFit.HasFlipX := false;
    NFit.HasFlipY := false;
    NFit.InferHitFn := @LocalFloatCompare;
    GTracker.NN := NN;
    GTracker.Validation := Validation;
    NFit.OnAfterEpoch := @GTracker.OnAfterEpoch;
    NFit.Fit(NN, Train, Validation, Test, BATCH_SIZE, NUM_EPOCHS);
    EndTime := Now;

    Result.Val  := ComputeStats(NN, Validation);
    Result.Test := ComputeStats(NN, Test);
    Result.EpochsToConverge := GTracker.ConvergedAtEpoch;
    Result.Seconds := (EndTime - StartTime) * 86400;
  finally
    GTracker.NN := nil;
    GTracker.Validation := nil;
    NFit.Free;
    NN.Free;
  end;
end;

procedure RunAlgo();
var
  TrainingPairs, ValidationPairs, TestPairs: TNNetVolumePairList;
  Kind: THeadKind;
  Results: array[THeadKind] of TRunResult;
  Convergence: string;
  StartTime, EndTime: TDateTime;
begin
  WriteLn('LogCosh dual experiment: TNNetLogCoshActivation (hidden) paired with');
  WriteLn('plain MSE head vs. TNNetLogCoshLoss head on the hypotenuse toy task.');
  WriteLn('Net: 2 -> ', HIDDEN_UNITS, ' (LogCosh activation) -> 1, ',
          NUM_EPOCHS, ' epochs, ', TRAIN_SIZE, ' train pairs, LR=0.01, RandSeed=42.');
  WriteLn('Convergence threshold (validation MSE, original scale): ',
          CONVERGE_THRESHOLD:0:2);
  WriteLn;

  StartTime := Now;
  for Kind := Low(THeadKind) to High(THeadKind) do
  begin
    // Reseed before generating data and before each fit so both
    // configurations see the same data and the same weight init.
    RandSeed := 42;
    TrainingPairs   := CreateHypotenusePairList(TRAIN_SIZE);
    ValidationPairs := CreateHypotenusePairList(VAL_SIZE);
    TestPairs       := CreateHypotenusePairList(TEST_SIZE);
    try
      RandSeed := 42;
      Write('Training ', HeadName(Kind), ' ...');
      Results[Kind] := RunOne(Kind, TrainingPairs, ValidationPairs, TestPairs);
      WriteLn(' done.');
    finally
      TestPairs.Free;
      ValidationPairs.Free;
      TrainingPairs.Free;
    end;
  end;
  EndTime := Now;

  WriteLn;
  WriteLn('=== Results (CSV) ===');
  WriteLn('config,val_mse,val_rmse,val_mae,val_max_err,val_r2,val_logcosh,',
          'test_mse,test_rmse,test_mae,test_r2,epochs_to_converge,total_epochs,seconds');
  for Kind := Low(THeadKind) to High(THeadKind) do
  begin
    if Results[Kind].EpochsToConverge < 0 then
      Convergence := 'NA'
    else
      Convergence := IntToStr(Results[Kind].EpochsToConverge);
    WriteLn(Results[Kind].Name, ',',
            FloatToStrF(Results[Kind].Val.MSE,       ffFixed, 8, 4), ',',
            FloatToStrF(Results[Kind].Val.RMSE,      ffFixed, 8, 4), ',',
            FloatToStrF(Results[Kind].Val.MAE,       ffFixed, 8, 4), ',',
            FloatToStrF(Results[Kind].Val.MaxAbsErr, ffFixed, 8, 4), ',',
            FloatToStrF(Results[Kind].Val.R2,        ffFixed, 8, 4), ',',
            FloatToStrF(Results[Kind].Val.LogCosh,   ffFixed, 8, 6), ',',
            FloatToStrF(Results[Kind].Test.MSE,      ffFixed, 8, 4), ',',
            FloatToStrF(Results[Kind].Test.RMSE,     ffFixed, 8, 4), ',',
            FloatToStrF(Results[Kind].Test.MAE,      ffFixed, 8, 4), ',',
            FloatToStrF(Results[Kind].Test.R2,       ffFixed, 8, 4), ',',
            Convergence, ',',
            Results[Kind].Epochs, ',',
            FloatToStrF(Results[Kind].Seconds, ffFixed, 8, 2));
  end;

  WriteLn;
  WriteLn('=== Per-config summary ===');
  for Kind := Low(THeadKind) to High(THeadKind) do
  begin
    WriteLn(Results[Kind].Name, ':');
    WriteLn('  validation : MSE=', FloatToStrF(Results[Kind].Val.MSE, ffFixed, 8, 4),
            '  RMSE=', FloatToStrF(Results[Kind].Val.RMSE, ffFixed, 8, 4),
            '  MAE=', FloatToStrF(Results[Kind].Val.MAE, ffFixed, 8, 4),
            '  maxErr=', FloatToStrF(Results[Kind].Val.MaxAbsErr, ffFixed, 8, 4),
            '  R2=', FloatToStrF(Results[Kind].Val.R2, ffFixed, 8, 4),
            '  LogCosh=', FloatToStrF(Results[Kind].Val.LogCosh, ffFixed, 8, 6));
    WriteLn('  test       : MSE=', FloatToStrF(Results[Kind].Test.MSE, ffFixed, 8, 4),
            '  RMSE=', FloatToStrF(Results[Kind].Test.RMSE, ffFixed, 8, 4),
            '  MAE=', FloatToStrF(Results[Kind].Test.MAE, ffFixed, 8, 4),
            '  R2=', FloatToStrF(Results[Kind].Test.R2, ffFixed, 8, 4));
    if Results[Kind].EpochsToConverge < 0 then
      WriteLn('  converged  : NA (MSE < ', CONVERGE_THRESHOLD:0:2,
              ' not reached in ', Results[Kind].Epochs, ' epochs)')
    else
      WriteLn('  converged  : epoch ', Results[Kind].EpochsToConverge);
    WriteLn('  train time : ', FormatFloat('0.00', Results[Kind].Seconds), ' s');
  end;

  WriteLn;
  WriteLn('Total wall time: ', FormatFloat('0.00', (EndTime - StartTime)*86400), ' s');
end;

var
  Application: record Title: string; end;

begin
  Application.Title := 'LogCosh Dual Experiment';
  RandSeed := 42;
  GTracker := TConvergenceTracker.Create;
  try
    RunAlgo();
  finally
    GTracker.Free;
  end;
end.
