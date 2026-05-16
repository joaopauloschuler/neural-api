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

  TRunResult = record
    Name: string;
    FinalValLoss: TNeuralFloat;
    EpochsToConverge: integer; // -1 if never converged
    Epochs: integer;
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

function RunOne(Kind: THeadKind;
                Train, Validation, Test: TNNetVolumePairList): TRunResult;
var
  NN: TNNet;
  NFit: TNeuralFit;
begin
  Result.Name := HeadName(Kind);
  Result.Epochs := NUM_EPOCHS;
  GTracker.ConvergedAtEpoch := -1;

  NN := TNNet.Create();
  NFit := TNeuralFit.Create();
  try
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

    Result.FinalValLoss := EvaluateMSEHelper(NN, Validation);
    Result.EpochsToConverge := GTracker.ConvergedAtEpoch;
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
  WriteLn('config,final_val_mse,epochs_to_converge,total_epochs');
  for Kind := Low(THeadKind) to High(THeadKind) do
  begin
    if Results[Kind].EpochsToConverge < 0 then
      Convergence := 'NA'
    else
      Convergence := IntToStr(Results[Kind].EpochsToConverge);
    WriteLn(Results[Kind].Name, ',',
            FloatToStrF(Results[Kind].FinalValLoss, ffFixed, 8, 4), ',',
            Convergence, ',',
            Results[Kind].Epochs);
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
