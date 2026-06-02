program PReLUvsLeakyReLU;
(*
PReLUvsLeakyReLU: three-config activation bake-off comparing
  - TNNetReLU       (baseline, no leak)
  - TNNetLeakyReLU  (fixed slope 0.01 on the negative half)
  - TNNetPReLU      (single learnable scalar slope, shared across
                     all elements; He et al. 2015, init alpha=0.25)
on the toy hypotenuse regression task:  y = sqrt(X^2 + Y^2).

Same architecture, same RandSeed for every variant so they see
identical inputs and identical weight init. Reports a CSV-style
table of final validation loss, epochs-to-converge, and trainable
parameter count (so the +1 learnable scalar from PReLU is visible).

Copyright (C) 2026 Joao Paulo Schwarz Schuler

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
any later version.

Coded by Claude (AI).
*)

{$mode objfpc}{$H+}

uses {$IFDEF UNIX} {$IFDEF UseCThreads}
  cthreads, {$ENDIF} {$ENDIF}
  Classes, SysUtils,
  neuralnetwork,
  neuralvolume,
  neuralfit;

type
  TActivationKind = (akReLU, akLeakyReLU, akPReLU);

  TBakeOffResult = record
    Name: string;
    FinalValLoss: TNeuralFloat;
    EpochsToConverge: integer; // -1 if never converged
    Epochs: integer;
    Weights: integer;
    Neurons: integer;
  end;

const
  // Inputs are normalized to [0,1] before training (X/100, Y/100, hypot/200).
  INPUT_SCALE = 100.0;
  TARGET_SCALE = 200.0;
  // Convergence threshold on validation MSE in ORIGINAL target units
  // (hypotenuse units, ~0..141).
  CONVERGE_THRESHOLD = 5.0;
  TRAIN_SIZE         = 1000;
  VAL_SIZE           = 200;
  TEST_SIZE          = 200;
  HIDDEN_UNITS       = 32;
  NUM_EPOCHS         = 150;
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

procedure AddActivation(NN: TNNet; Kind: TActivationKind);
begin
  case Kind of
    akReLU:      NN.AddLayer(TNNetReLU.Create());
    akLeakyReLU: NN.AddLayer(TNNetLeakyReLU.Create());
    akPReLU:     NN.AddLayer(TNNetPReLU.Create()); // init alpha = 0.25
  end;
end;

function ActivationName(Kind: TActivationKind): string;
begin
  case Kind of
    akReLU:      Result := 'TNNetReLU';
    akLeakyReLU: Result := 'TNNetLeakyReLU';
    akPReLU:     Result := 'TNNetPReLU';
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

function RunOne(Kind: TActivationKind;
                Train, Validation, Test: TNNetVolumePairList): TBakeOffResult;
var
  NN: TNNet;
  NFit: TNeuralFit;
begin
  Result.Name := ActivationName(Kind);
  Result.Epochs := NUM_EPOCHS;
  GTracker.ConvergedAtEpoch := -1;

  NN := TNNet.Create();
  NFit := TNeuralFit.Create();
  try
    NN.AddLayer(TNNetInput.Create(2));
    NN.AddLayer(TNNetFullConnectLinear.Create(HIDDEN_UNITS));
    AddActivation(NN, Kind);
    NN.AddLayer(TNNetFullConnectLinear.Create(1));

    Result.Weights := NN.CountWeights();
    Result.Neurons := NN.CountNeurons();

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
  Kind: TActivationKind;
  Results: array[TActivationKind] of TBakeOffResult;
  Convergence: string;
  StartTime, EndTime: TDateTime;
  BaselineWeights: integer;
  Delta: integer;
  DeltaStr: string;
begin
  WriteLn('ReLU / LeakyReLU / PReLU bake-off on the hypotenuse toy task.');
  WriteLn('Net: 2 -> ', HIDDEN_UNITS, ' (activation) -> 1, ',
          NUM_EPOCHS, ' epochs, ', TRAIN_SIZE, ' train pairs, LR=0.01, RandSeed=42.');
  WriteLn('Convergence threshold (validation MSE, target units): ',
          CONVERGE_THRESHOLD:0:2);
  WriteLn;

  StartTime := Now;
  for Kind := Low(TActivationKind) to High(TActivationKind) do
  begin
    // Reseed before generating data and before each fit so every
    // activation sees the same data and the same weight init.
    RandSeed := 42;
    TrainingPairs   := CreateHypotenusePairList(TRAIN_SIZE);
    ValidationPairs := CreateHypotenusePairList(VAL_SIZE);
    TestPairs       := CreateHypotenusePairList(TEST_SIZE);
    try
      RandSeed := 42;
      Write('Training ', ActivationName(Kind), ' ...');
      Results[Kind] := RunOne(Kind, TrainingPairs, ValidationPairs, TestPairs);
      WriteLn(' done.');
    finally
      TestPairs.Free;
      ValidationPairs.Free;
      TrainingPairs.Free;
    end;
  end;
  EndTime := Now;

  BaselineWeights := Results[akReLU].Weights;

  WriteLn;
  WriteLn('=== Results (CSV) ===');
  WriteLn('activation,final_val_loss,epochs_to_converge,total_epochs,trainable_weights,weight_delta_vs_relu');
  for Kind := Low(TActivationKind) to High(TActivationKind) do
  begin
    if Results[Kind].EpochsToConverge < 0 then
      Convergence := 'NA'
    else
      Convergence := IntToStr(Results[Kind].EpochsToConverge);
    Delta := Results[Kind].Weights - BaselineWeights;
    if Delta >= 0 then
      DeltaStr := '+' + IntToStr(Delta)
    else
      DeltaStr := IntToStr(Delta);
    WriteLn(Results[Kind].Name, ',',
            FloatToStrF(Results[Kind].FinalValLoss, ffFixed, 8, 4), ',',
            Convergence, ',',
            Results[Kind].Epochs, ',',
            Results[Kind].Weights, ',',
            DeltaStr);
  end;
  WriteLn;
  WriteLn('Total wall time: ', FormatFloat('0.00', (EndTime - StartTime)*86400), ' s');
end;

var
  Application: record Title: string; end;

begin
  Application.Title := 'PReLU vs LeakyReLU vs ReLU Bake-Off';
  RandSeed := 42;
  GTracker := TConvergenceTracker.Create;
  try
    RunAlgo();
  finally
    GTracker.Free;
  end;
end.
