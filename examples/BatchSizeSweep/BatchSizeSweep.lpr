program BatchSizeSweep;
(*
BatchSizeSweep: trains the SAME tiny MLP on the SAME synthetic
hypotenuse task (y = sqrt(X^2 + Y^2)) across a sweep of batch sizes
{1, 8, 32, 128} and prints how the batch-size knob trades off
wall-clock-per-epoch against epochs-to-converge.

Tiny MLP (2 -> 16 -> 1), fixed RandSeed (424242) restored before each
run so every batch size sees the same data and the same weight init.
For each batch size it reports:
  - wall-clock time per epoch,
  - the first epoch validation MSE crosses the convergence threshold
    (or NA if it never does within the epoch budget),
  - wall-clock time to converge,
  - final train and validation MSE (in original hypotenuse units).

The beginner takeaway: small batches converge in FEWER epochs but each
epoch is slower per-sample (more weight updates) and noisier; large
batches give faster, smoother epochs but may need MORE epochs at the
same learning rate.

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

const
  // Inputs/targets normalized to ~[0,1] before training.
  INPUT_SCALE        = 100.0;
  TARGET_SCALE       = 200.0;
  // Convergence threshold on MSE in ORIGINAL hypotenuse units (~0..141).
  CONVERGE_THRESHOLD = 8.0;
  TRAIN_SIZE         = 1000;
  VAL_SIZE           = 200;
  TEST_SIZE          = 200;
  HIDDEN_UNITS       = 16;
  NUM_EPOCHS         = 80;
  INITIAL_LR         = 0.01;
  RAND_SEED          = 424242;

type
  TSweepResult = record
    BatchSize: integer;
    SecPerEpoch: Double;
    EpochsToConverge: integer; // -1 if never converged
    SecToConverge: Double;     // -1 if never converged
    FinalTrainMSE: TNeuralFloat;
    FinalValMSE: TNeuralFloat;
    Epochs: integer;
  end;

function EvaluateMSEHelper(NN: TNNet; Pairs: TNNetVolumePairList): TNeuralFloat; forward;

type
  TConvergenceTracker = class
  public
    ConvergedAtEpoch: integer;
    SecToConverge: Double;
    StartTime: TDateTime;
    NN: TNNet;
    Validation: TNNetVolumePairList;
    procedure OnAfterEpoch(Sender: TObject);
  end;

procedure TConvergenceTracker.OnAfterEpoch(Sender: TObject);
var
  Fit: TNeuralFit;
  Mse: TNeuralFloat;
begin
  Fit := Sender as TNeuralFit;
  if NN = nil then Exit;
  Mse := EvaluateMSEHelper(NN, Validation);
  if (ConvergedAtEpoch < 0) and (Mse < CONVERGE_THRESHOLD) then
  begin
    ConvergedAtEpoch := Fit.CurrentEpoch;
    SecToConverge := (Now - StartTime) * 86400;
  end;
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

function EvaluateMSEHelper(NN: TNNet; Pairs: TNNetVolumePairList): TNeuralFloat;
var
  I: integer;
  Pred: TNeuralFloat;
  SumSq: Double;
begin
  SumSq := 0;
  for I := 0 to Pairs.Count - 1 do
  begin
    NN.Compute(Pairs[I].I);
    Pred := NN.GetLastLayer().Output.FData[0];
    SumSq := SumSq + Sqr((Pred - Pairs[I].O.FData[0]) * TARGET_SCALE);
  end;
  if Pairs.Count > 0 then
    Result := SumSq / Pairs.Count
  else
    Result := 0;
end;

function RunOne(BatchSize: integer;
                Train, Validation, Test: TNNetVolumePairList): TSweepResult;
var
  NN: TNNet;
  NFit: TNeuralFit;
  StartTime, EndTime: TDateTime;
  TotalSec: Double;
begin
  Result.BatchSize := BatchSize;
  Result.Epochs := NUM_EPOCHS;
  GTracker.ConvergedAtEpoch := -1;
  GTracker.SecToConverge := -1;

  NN := TNNet.Create();
  NFit := TNeuralFit.Create();
  try
    NN.AddLayer(TNNetInput.Create(2));
    NN.AddLayer(TNNetFullConnectLinear.Create(HIDDEN_UNITS));
    NN.AddLayer(TNNetReLU.Create());
    NN.AddLayer(TNNetFullConnectLinear.Create(1));

    NFit.FileNameBase := GetTempDir + 'BatchSizeSweep_autosave';
    NFit.InitialLearningRate := INITIAL_LR;
    NFit.LearningRateDecay := 0;
    NFit.L2Decay := 0;
    NFit.Verbose := false;
    NFit.HasFlipX := false;
    NFit.HasFlipY := false;
    NFit.InferHitFn := @LocalFloatCompare;

    GTracker.NN := NN;
    GTracker.Validation := Validation;
    StartTime := Now;
    GTracker.StartTime := StartTime;
    NFit.OnAfterEpoch := @GTracker.OnAfterEpoch;
    NFit.Fit(NN, Train, Validation, Test, BatchSize, NUM_EPOCHS);
    EndTime := Now;
    TotalSec := (EndTime - StartTime) * 86400;

    Result.SecPerEpoch := TotalSec / NUM_EPOCHS;
    Result.EpochsToConverge := GTracker.ConvergedAtEpoch;
    Result.SecToConverge := GTracker.SecToConverge;
    Result.FinalTrainMSE := EvaluateMSEHelper(NN, Train);
    Result.FinalValMSE := EvaluateMSEHelper(NN, Validation);
  finally
    GTracker.NN := nil;
    GTracker.Validation := nil;
    NFit.Free;
    NN.Free;
  end;
end;

const
  BATCH_SIZES: array[0..3] of integer = (1, 8, 32, 128);

procedure RunAlgo();
var
  TrainingPairs, ValidationPairs, TestPairs: TNNetVolumePairList;
  Results: array[0..High(BATCH_SIZES)] of TSweepResult;
  I: integer;
  Convergence, ConvTime: string;
  StartTime, EndTime: TDateTime;
  AllConverged: boolean;
  AllLowError: boolean;
begin
  WriteLn('Batch-size sweep on the hypotenuse toy task: y = sqrt(X^2 + Y^2).');
  WriteLn('Net: 2 -> ', HIDDEN_UNITS, ' (ReLU) -> 1, up to ', NUM_EPOCHS,
          ' epochs, ', TRAIN_SIZE, ' train pairs, LR=', INITIAL_LR:0:3,
          ', RandSeed=', RAND_SEED, '.');
  WriteLn('Same net, same data, same seed -- the ONLY variable is the batch size.');
  WriteLn('Convergence threshold (validation MSE, original units): ',
          CONVERGE_THRESHOLD:0:2);
  WriteLn;

  StartTime := Now;
  for I := Low(BATCH_SIZES) to High(BATCH_SIZES) do
  begin
    // Reseed before generating data AND before each fit so every batch
    // size trains on the same data with the same weight initialization.
    RandSeed := RAND_SEED;
    TrainingPairs   := CreateHypotenusePairList(TRAIN_SIZE);
    ValidationPairs := CreateHypotenusePairList(VAL_SIZE);
    TestPairs       := CreateHypotenusePairList(TEST_SIZE);
    try
      RandSeed := RAND_SEED;
      Write('Training batch size ', BATCH_SIZES[I]:4, ' ...');
      Results[I] := RunOne(BATCH_SIZES[I], TrainingPairs, ValidationPairs, TestPairs);
      WriteLn(' done.');
    finally
      TestPairs.Free;
      ValidationPairs.Free;
      TrainingPairs.Free;
    end;
  end;
  EndTime := Now;

  WriteLn;
  WriteLn('=== Batch-size trade-off ===');
  WriteLn('batch  sec/epoch  epochs_to_conv  sec_to_conv  final_train_mse  final_val_mse');
  for I := Low(BATCH_SIZES) to High(BATCH_SIZES) do
  begin
    if Results[I].EpochsToConverge < 0 then
    begin
      Convergence := 'NA';
      ConvTime := 'NA';
    end
    else
    begin
      Convergence := IntToStr(Results[I].EpochsToConverge);
      ConvTime := FormatFloat('0.00', Results[I].SecToConverge);
    end;
    WriteLn(
      Results[I].BatchSize:5, '  ',
      FormatFloat('0.000', Results[I].SecPerEpoch):9, '  ',
      Convergence:14, '  ',
      ConvTime:11, '  ',
      FloatToStrF(Results[I].FinalTrainMSE, ffFixed, 8, 4):15, '  ',
      FloatToStrF(Results[I].FinalValMSE, ffFixed, 8, 4):13);
  end;
  WriteLn;

  WriteLn('Reading the table:');
  WriteLn('  - sec/epoch tends to FALL as the batch grows (fewer weight updates,');
  WriteLn('    bigger vectorized chunks) -- a large batch makes each epoch cheaper.');
  WriteLn('  - epochs_to_conv tends to RISE as the batch grows: a small batch takes');
  WriteLn('    many noisy steps and reaches the threshold in fewer epochs.');
  WriteLn('  - That is the knob: small batch = fewer, noisier, slower-per-sample');
  WriteLn('    epochs; large batch = more, smoother, cheaper epochs (often wanting');
  WriteLn('    a larger learning rate to keep the epoch count down).');
  WriteLn;

  // Self-checking correctness gate.
  AllConverged := true;
  AllLowError := true;
  for I := Low(BATCH_SIZES) to High(BATCH_SIZES) do
  begin
    if Results[I].EpochsToConverge < 0 then AllConverged := false;
    if Results[I].FinalValMSE >= CONVERGE_THRESHOLD then AllLowError := false;
  end;

  WriteLn('Total wall time: ', FormatFloat('0.00', (EndTime - StartTime)*86400), ' s');
  if AllConverged and AllLowError then
    WriteLn('Correctness gate: PASS (every batch size converged below MSE ',
            CONVERGE_THRESHOLD:0:2, ').')
  else
    WriteLn('Correctness gate: FAIL (at least one batch size did not converge).');
end;

begin
  RandSeed := RAND_SEED;
  GTracker := TConvergenceTracker.Create;
  try
    RunAlgo();
  finally
    GTracker.Free;
  end;
end.
