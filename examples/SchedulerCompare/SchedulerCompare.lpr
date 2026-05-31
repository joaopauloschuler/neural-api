program SchedulerCompare;
(*
SchedulerCompare: a learning-rate SCHEDULE bake-off. The SAME tiny
regression net is trained several times on the SAME tiny synthetic task;
the ONLY thing that differs between arms is the learning-rate SCHEDULE
that drives the optimiser:

  - constant     flat baseLR for every epoch (the control arm)
  - Step         TStepLR(baseLR, stepSize, gamma): staircase decay
  - Cosine       TCosineAnnealingLR(etaMax, etaMin, T): smooth anneal
  - WarmupCosine TWarmupCosineLR(etaMax, etaMin, warmup, T): ramp + anneal

The scheduler classes live in neural/neuralscheduler.pas. They are NOT
wired into TNeuralFit (that is a separate open task), so here we drive the
LR by hand: a plain epoch loop calls Sched.NextLR(epoch, epoch) at the top
of every epoch and pushes the result into the net with NN.SetLearningRate.
The schedulers key on the STEP argument (t := Step); we pass the epoch
index as the step, so the horizon T is measured in epochs.

Task: a tiny synthetic regression. Each sample is an 8-dim input vector
drawn from [-1, 1]; the scalar target is a fixed, deterministic, mildly
nonlinear function of those features (a trig term, a feature product and a
linear term). A small 2-hidden-layer MLP is exactly the right tool, and
every arm can learn it.

Every arm reseeds RandSeed to the same value before generating its data
and before building/initialising its net, so all arms see identical inputs
and identical weight init; only the schedule differs.

The headline is the ASCII LR chart: you can SEE constant is flat, Step
drops in stairs, Cosine curves down, and WarmupCosine ramps then curves.
The summary table then reports the final train/val loss each schedule
reaches. The per-arm ranking is SEED-DEPENDENT; this is a comparison
harness, not a claim that any one schedule is universally best.

Pure CPU, no external dataset, single-threaded, finishes in a few seconds.

Copyright (C) 2026 Joao Paulo Schwarz Schuler

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
*)

{$mode objfpc}{$H+}

uses {$IFDEF UNIX} {$IFDEF UseCThreads}
  cthreads, {$ENDIF} {$ENDIF}
  Classes, SysUtils, Math,
  neuralnetwork,
  neuralvolume,
  neuralscheduler;

const
  cInDim     = 8;      // features per sample (input depth)
  cHiddenW   = 16;     // hidden width of the MLP
  cTrain     = 400;    // training samples
  cVal       = 150;    // validation samples
  cEpochs    = 40;     // also the scheduler horizon T (steps = epochs)
  cBaseLR    = 0.05;   // peak / base learning rate, shared by all arms
  cMinLR     = 0.001;  // floor for the cosine arms (etaMin)
  cInertia   = 0.9;
  cSeed      = 42;

type
  TSchedKind = (skConst, skStep, skCosine, skWarmupCosine);

  TArm = record
    Name: string;
    Kind: TSchedKind;
  end;

  TArmResult = record
    Name      : string;
    InitTrain : TNeuralFloat;           // train MSE before any training
    InitVal   : TNeuralFloat;           // val   MSE before any training
    FinalTrain: TNeuralFloat;           // train MSE after training
    FinalVal  : TNeuralFloat;           // val   MSE after training
    Seconds   : TNeuralFloat;           // wall-clock for the arm
    LRCurve   : array[0..cEpochs - 1] of TNeuralFloat;  // applied LR per epoch
  end;

const
  cArms: array[0..3] of TArm =
  (
    (Name: 'constant';     Kind: skConst),
    (Name: 'Step';         Kind: skStep),
    (Name: 'Cosine';       Kind: skCosine),
    (Name: 'WarmupCosine'; Kind: skWarmupCosine)
  );

function RandUniform: TNeuralFloat;
begin
  Result := (Random * 2.0) - 1.0;
end;

// Deterministic, learnable, mildly nonlinear scalar target.
function TargetFn(const f: array of TNeuralFloat): TNeuralFloat;
begin
  Result := Sin(f[0]) + 0.5 * f[1] * f[2] - 0.3 * f[3] + 0.2 * f[4] * f[5];
end;

// One sample: a (cInDim x 1 x 1) input and a (1 x 1 x 1) scalar target.
procedure MakeSample(out X, Y: TNNetVolume);
var
  C: integer;
  f: array[0..cInDim - 1] of TNeuralFloat;
begin
  X := TNNetVolume.Create(cInDim, 1, 1);
  Y := TNNetVolume.Create(1, 1, 1);
  for C := 0 to cInDim - 1 do
  begin
    f[C] := RandUniform;
    X.Data[C, 0, 0] := f[C];
  end;
  Y.Data[0, 0, 0] := TargetFn(f);
end;

procedure BuildSet(out Pairs: TNNetVolumePairList; Count: integer);
var
  I: integer;
  X, Y: TNNetVolume;
begin
  Pairs := TNNetVolumePairList.Create();
  for I := 1 to Count do
  begin
    MakeSample(X, Y);
    Pairs.Add(TNNetVolumePair.Create(X, Y));
  end;
end;

// Shared MLP; identical for every arm (only the LR schedule differs).
procedure BuildNet(out NN: TNNet);
begin
  NN := TNNet.Create();
  NN.AddLayer(TNNetInput.Create(cInDim, 1, 1));
  NN.AddLayer(TNNetFullConnectReLU.Create(cHiddenW));
  NN.AddLayer(TNNetFullConnectReLU.Create(cHiddenW));
  NN.AddLayer(TNNetFullConnectLinear.Create(1));
end;

// Mean squared error over a pair list.
function MeanMSE(NN: TNNet; Pairs: TNNetVolumePairList): TNeuralFloat;
var
  I: integer;
  P: TNNetVolume;
  Sum: Double;
  Diff: TNeuralFloat;
begin
  Sum := 0;
  for I := 0 to Pairs.Count - 1 do
  begin
    NN.Compute(Pairs[I].I);
    P := NN.GetLastLayer().Output;
    Diff := P.Data[0, 0, 0] - Pairs[I].O.Data[0, 0, 0];
    Sum := Sum + Diff * Diff;
  end;
  if Pairs.Count > 0 then
    Result := Sum / Pairs.Count
  else
    Result := 0;
end;

// Build the scheduler for an arm, or nil for the constant control.
function MakeSched(Arm: TArm): TNeuralLRScheduler;
begin
  case Arm.Kind of
    skConst       : Result := nil;
    skStep        : Result := TStepLR.Create(cBaseLR, cEpochs div 4, 0.5);
    skCosine      : Result := TCosineAnnealingLR.Create(cBaseLR, cMinLR, cEpochs);
    skWarmupCosine: Result := TWarmupCosineLR.Create(cBaseLR, cMinLR,
                                cEpochs div 5, cEpochs);
  else
    Result := nil;
  end;
end;

function RunArm(Arm: TArm): TArmResult;
var
  NN: TNNet;
  Sched: TNeuralLRScheduler;
  TrainSet, ValSet: TNNetVolumePairList;
  Epoch, I: integer;
  LR: TNeuralFloat;
  StartTime, EndTime: TDateTime;
begin
  Result.Name := Arm.Name;

  // Reseed BEFORE data gen and BEFORE net build so every arm sees the
  // same inputs and the same weight initialisation.
  RandSeed := cSeed;
  BuildSet(TrainSet, cTrain);
  BuildSet(ValSet, cVal);

  RandSeed := cSeed;
  BuildNet(NN);
  Sched := MakeSched(Arm);
  try
    NN.SetLearningRate(cBaseLR, cInertia);
    Result.InitTrain := MeanMSE(NN, TrainSet);
    Result.InitVal   := MeanMSE(NN, ValSet);

    StartTime := Now;
    for Epoch := 0 to cEpochs - 1 do
    begin
      // Resolve this epoch's learning rate from the schedule (or hold the
      // base LR for the constant control) and push it into the net. The
      // schedulers key on the STEP argument; we feed the epoch index.
      if Sched = nil then
        LR := cBaseLR
      else
        LR := Sched.NextLR(Epoch, Epoch);
      Result.LRCurve[Epoch] := LR;
      NN.SetLearningRate(LR, cInertia);

      for I := 0 to TrainSet.Count - 1 do
      begin
        NN.Compute(TrainSet[I].I);
        NN.Backpropagate(TrainSet[I].O);
      end;
    end;
    EndTime := Now;

    Result.Seconds := (EndTime - StartTime) * 86400.0;
    Result.FinalTrain := MeanMSE(NN, TrainSet);
    Result.FinalVal   := MeanMSE(NN, ValSet);
  finally
    Sched.Free;
    NN.Free;
    ValSet.Free;
    TrainSet.Free;
  end;
end;

function SafeF(V: TNeuralFloat; Dec: integer): string;
begin
  if IsNan(V) or IsInfinite(V) then
    Result := 'NaN'
  else
    Result := FloatToStrF(V, ffFixed, 8, Dec);
end;

// One row of the ASCII LR chart: scale [0, cBaseLR] to a fixed-width bar.
function LRBar(LR, MaxLR: TNeuralFloat; Width: integer): string;
var
  Filled, K: integer;
begin
  if (MaxLR <= 0) or IsNan(LR) or IsInfinite(LR) then
  begin
    Result := StringOfChar(' ', Width);
    Exit;
  end;
  Filled := Round((LR / MaxLR) * Width);
  if Filled < 0 then Filled := 0;
  if Filled > Width then Filled := Width;
  Result := '';
  for K := 1 to Filled do Result := Result + '#';
  for K := Filled + 1 to Width do Result := Result + ' ';
end;

procedure PrintLRChart(const Results: array of TArmResult);
const
  cBarW = 32;
  cRows = 10;          // sample at most this many epoch rows to keep it short
var
  A, R, Epoch: integer;
begin
  WriteLn('=== Learning-rate schedules (bar = LR for that epoch, full bar = ',
          SafeF(cBaseLR, 3), ') ===');
  for A := 0 to High(Results) do
  begin
    WriteLn;
    WriteLn(Results[A].Name, ':');
    for R := 0 to cRows - 1 do
    begin
      // Spread cRows samples across the cEpochs-long curve.
      Epoch := Round(R * (cEpochs - 1) / (cRows - 1));
      WriteLn(Format('  epoch %3d  lr=%s  |%s|',
              [Epoch, SafeF(Results[A].LRCurve[Epoch], 4),
               LRBar(Results[A].LRCurve[Epoch], cBaseLR, cBarW)]));
    end;
  end;
end;

procedure RunBakeoff();
var
  K: integer;
  Results: array[0..High(cArms)] of TArmResult;
  AllFinite, AllLearned: boolean;
  R: TArmResult;
begin
  WriteLn('Learning-rate schedule bake-off: one tiny net, four LR schedules.');
  WriteLn('Net: Input(', cInDim, ') -> FullConnectReLU(', cHiddenW,
          ') -> FullConnectReLU(', cHiddenW, ') -> FullConnectLinear(1)');
  WriteLn('Target: y = sin(x0) + 0.5*x1*x2 - 0.3*x3 + 0.2*x4*x5');
  WriteLn('Train=', cTrain, ' Val=', cVal, '  Epochs=', cEpochs,
          '  baseLR=', SafeF(cBaseLR, 3), '  minLR=', SafeF(cMinLR, 3),
          '  RandSeed=', cSeed);
  WriteLn('Same data, same init, same net; only the LR schedule changes.');
  WriteLn('LR is driven by hand: Sched.NextLR(epoch, epoch) -> SetLearningRate.');
  WriteLn;

  for K := 0 to High(cArms) do
  begin
    Write('Training ', cArms[K].Name, ' ...');
    Results[K] := RunArm(cArms[K]);
    WriteLn(' done.  final_val_mse=', SafeF(Results[K].FinalVal, 4),
            '  ', SafeF(Results[K].Seconds, 2), 's');
  end;

  WriteLn;
  PrintLRChart(Results);

  WriteLn;
  WriteLn('=== Comparison (MSE, wall-clock) ===');
  WriteLn(Format('%-14s %11s %11s %11s %11s %9s',
          ['schedule', 'init_train', 'final_train', 'init_val',
           'final_val', 'seconds']));
  for K := 0 to High(cArms) do
  begin
    R := Results[K];
    WriteLn(Format('%-14s %11s %11s %11s %11s %9s',
            [R.Name, SafeF(R.InitTrain, 4), SafeF(R.FinalTrain, 4),
             SafeF(R.InitVal, 4), SafeF(R.FinalVal, 4), SafeF(R.Seconds, 2)]));
  end;

  // --- Sanity checks ---
  WriteLn;
  WriteLn('=== Sanity checks ===');
  AllFinite := True;
  AllLearned := True;
  for K := 0 to High(cArms) do
  begin
    R := Results[K];
    if IsNan(R.FinalVal) or IsInfinite(R.FinalVal) or
       IsNan(R.FinalTrain) or IsInfinite(R.FinalTrain) then
      AllFinite := False;
    if not (R.FinalVal < R.InitVal) then
      AllLearned := False;
  end;

  if AllFinite then
    WriteLn('[PASS] all ', Length(cArms),
            ' arms produced a finite (no NaN/Inf) final loss.')
  else
    WriteLn('[FAIL] at least one arm produced NaN/Inf final loss.');

  if AllLearned then
    WriteLn('[PASS] all ', Length(cArms),
            ' arms reduced val loss below their pre-training baseline.')
  else
    WriteLn('[FAIL] at least one arm did not improve over its baseline.');

  WriteLn;
  WriteLn('Note: the per-arm ranking above is seed-dependent. This harness shows');
  WriteLn('that all four schedules train this net; it does not crown a winner.');
end;

begin
  RandSeed := cSeed;
  RunBakeoff();
end.
