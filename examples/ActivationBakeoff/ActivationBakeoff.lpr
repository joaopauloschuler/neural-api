program ActivationBakeoff;
(*
ActivationBakeoff: compares the mainstream / modern ReLU-family
activations on the toy hypotenuse task: y = sqrt(X^2 + Y^2).

Tiny MLP (2 -> 32 -> 1), 150 epochs, 1000 training pairs, fixed RNG
seed for reproducibility. Reports a CSV-style table of final
validation loss and epochs-to-converge per activation.

This is the ReLU-family counterpart to HyperbolicActivationBakeOff
(which compares only the hyperbolic-tanh family). Same task, same net
shape, same seed: the only variable is the activation.

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
  TActivationKind = (akReLU, akLeakyReLU, akVeryLeakyReLU, akReLU6, akPReLU,
                     akELU, akSELU, akCELU, akSwish, akSiLU, akGELU,
                     akHardSwish, akMish, akSoftPlus, akAconC);

  TBakeOffResult = record
    Name: string;
    FinalValLoss: TNeuralFloat;
    EpochsToConverge: integer; // -1 if never converged
    Epochs: integer;
  end;

const
  // Inputs are normalized to [0,1] before training (X/100, Y/100, hypot/200)
  // so any exponential-tail activation does not overflow.
  INPUT_SCALE = 100.0;
  TARGET_SCALE = 200.0;
  // Convergence threshold is on MSE in the ORIGINAL target scale (hypotenuse units, ~0..141).
  CONVERGE_THRESHOLD = 5.0;
  TRAIN_SIZE         = 10000;
  VAL_SIZE           = 200;
  TEST_SIZE          = 200;
  HIDDEN_UNITS       = 32;
  NUM_EPOCHS         = 60;
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
    akReLU:          NN.AddLayer(TNNetReLU.Create());
    akLeakyReLU:     NN.AddLayer(TNNetLeakyReLU.Create());
    akVeryLeakyReLU: NN.AddLayer(TNNetVeryLeakyReLU.Create());
    akReLU6:         NN.AddLayer(TNNetReLU6.Create());
    akPReLU:         NN.AddLayer(TNNetPReLU.Create());
    akELU:           NN.AddLayer(TNNetELU.Create());
    akSELU:          NN.AddLayer(TNNetSELU.Create());
    akCELU:          NN.AddLayer(TNNetCELU.Create());
    akSwish:         NN.AddLayer(TNNetSwish.Create());
    akSiLU:          NN.AddLayer(TNNetSiLU.Create());
    akGELU:          NN.AddLayer(TNNetGELU.Create());
    akHardSwish:     NN.AddLayer(TNNetHardSwish.Create());
    akMish:          NN.AddLayer(TNNetMish.Create());
    akSoftPlus:      NN.AddLayer(TNNetSoftPlus.Create());
    akAconC:         NN.AddLayer(TNNetAconC.Create());
  end;
end;

function ActivationName(Kind: TActivationKind): string;
begin
  case Kind of
    akReLU:          Result := 'TNNetReLU';
    akLeakyReLU:     Result := 'TNNetLeakyReLU';
    akVeryLeakyReLU: Result := 'TNNetVeryLeakyReLU';
    akReLU6:         Result := 'TNNetReLU6';
    akPReLU:         Result := 'TNNetPReLU';
    akELU:           Result := 'TNNetELU';
    akSELU:          Result := 'TNNetSELU';
    akCELU:          Result := 'TNNetCELU';
    akSwish:         Result := 'TNNetSwish';
    akSiLU:          Result := 'TNNetSiLU';
    akGELU:          Result := 'TNNetGELU';
    akHardSwish:     Result := 'TNNetHardSwish';
    akMish:          Result := 'TNNetMish';
    akSoftPlus:      Result := 'TNNetSoftPlus';
    akAconC:         Result := 'TNNetAconC';
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

    NFit.FileNameBase := GetTempDir + 'ActivationBakeoff_autosave';
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
begin
  WriteLn('Mainstream (ReLU-family) activation bake-off on the hypotenuse toy task.');
  WriteLn('Net: 2 -> ', HIDDEN_UNITS, ' (activation) -> 1, ',
          NUM_EPOCHS, ' epochs, ', TRAIN_SIZE, ' train pairs, LR=0.01, RandSeed=42.');
  WriteLn('Convergence threshold (validation loss): ', CONVERGE_THRESHOLD:0:2);
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

  WriteLn;
  WriteLn('=== Results (CSV) ===');
  WriteLn('activation,final_val_loss,epochs_to_converge,total_epochs');
  for Kind := Low(TActivationKind) to High(TActivationKind) do
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
  Application.Title := 'Activation Bake-Off';
  RandSeed := 42;
  GTracker := TConvergenceTracker.Create;
  try
    RunAlgo();
  finally
    GTracker.Free;
  end;
end.
