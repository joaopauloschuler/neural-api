program NeuralODE;
(*
NeuralODE: a tiny, synthetic, no-download demonstration of the Chen et al. 2018
"Neural Ordinary Differential Equations" idea (https://arxiv.org/abs/1806.07366)
built on TNNet.AddNeuralODEBlock.

A residual block x_{n+1} = x_n + f(x_n) is one explicit Euler step of the ODE
dx/dt = f(x,t). A Neural ODE replaces a STACK of distinct residual blocks with
ONE SHARED residual function f integrated forward over `Steps` Euler sub-steps
of size h = 1/Steps:
    y := x
    repeat Steps times:  y := y + h * f(y)
Because the SAME f (the SAME weights) is reused at every step, the parameter
count is INDEPENDENT of Steps -- the headline "depth for free" property.

This demo trains a tiny classifier on a synthetic 2-class concentric-rings task
(nonlinearly separable). The only trainable trunk is ONE AddNeuralODEBlock. We
sweep Steps in {1, 2, 4}, averaging each arm over a few seeds, and print per
Steps:
  - the trainable parameter count (CountNeurons / CountWeights), which stays
    CONSTANT across the sweep, and
  - the validation accuracy, which stays roughly FLAT and high (more integration
    steps refine the same flow without adding parameters).

Honest note on budget: the task is small and the trunk is tiny so the whole
sweep runs in seconds on one CPU thread. The learning rate is scaled by Steps
because each Euler step multiplies f by h=1/Steps (shrinking the effective
gradient to f's shared weights ~1/Steps); this keeps the fixed-epoch compare
fair without changing the parameter count. The point being demonstrated is
constant-parameter continuous depth, not a new state of the art.

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
  neuralvolume,
  neuralfit;

const
  D_MODEL    = 8;    // width of the residual/ODE state (lives in the Depth axis)
  HIDDEN_DIM = 16;   // hidden width of the shared residual function f
  TRAIN_SIZE = 600;
  VAL_SIZE   = 200;
  NUM_EPOCHS = 60;
  BATCH_SIZE = 32;
  SEED       = 42;

type
  TArmResult = record
    Steps: integer;
    Neurons: integer;
    Weights: integer;
    ValAcc: TNeuralFloat;
  end;

// Two concentric noisy rings -> a 2-class problem that a linear head alone
// cannot separate (it needs a nonlinear "fold"), so the ODE trunk has to do
// real work, yet is easy enough that the tiny trunk solves it at every Steps.
function CreateSpiralPairList(MaxCnt: integer): TNNetVolumePairList;
var
  Cnt, Cls: integer;
  R, Theta, Px, Py: TNeuralFloat;
  Target: TNNetVolume;
begin
  Result := TNNetVolumePairList.Create();
  for Cnt := 1 to MaxCnt do
  begin
    Cls := Random(2);
    // Inner ring (class 0) vs outer ring (class 1), each with radial noise.
    R := 0.35 + Cls * 0.45 + (Random(200) - 100) / 1000.0;
    Theta := Random(1000) / 1000.0 * 2.0 * PI;
    Px := R * Cos(Theta);
    Py := R * Sin(Theta);
    Target := TNNetVolume.Create(2);
    Target.Fill(0);
    Target.FData[Cls] := 1;                      // one-hot
    Result.Add(
      TNNetVolumePair.Create(
        TNNetVolume.Create([Px, Py]),
        Target
      )
    );
  end;
end;

function LocalClassCompare(A, B: TNNetVolume; ThreadId: integer): boolean;
begin
  // Hit = predicted argmax class matches the one-hot target.
  Result := ( A.GetClass() = B.GetClass() );
end;

// Build the classifier whose ONLY trunk is one AddNeuralODEBlock(Steps).
//   Input(2) -> project to D_MODEL features -> reshape into Depth
//   -> AddNeuralODEBlock(HIDDEN_DIM, Steps) -> 2-class SoftMax head.
procedure BuildNet(NN: TNNet; Steps: integer);
begin
  NN.AddLayer( TNNetInput.Create(2) );
  NN.AddLayer( TNNetFullConnectLinear.Create(D_MODEL) );
  // FullConnectLinear lays D_MODEL out along X; the ODE block convolves over the
  // Depth axis (pointwise convs), so reshape the feature vector into Depth.
  NN.AddLayer( TNNetReshape.Create(1, 1, D_MODEL) );
  NN.AddNeuralODEBlock( NN.GetLastLayer(), HIDDEN_DIM, Steps );
  NN.AddLayer( TNNetFullConnectLinear.Create(2) );
  NN.AddLayer( TNNetSoftMax.Create() );
end;

function EvaluateAccuracy(NN: TNNet; Pairs: TNNetVolumePairList): TNeuralFloat;
var
  I, Hits: integer;
begin
  Hits := 0;
  for I := 0 to Pairs.Count - 1 do
  begin
    NN.Compute(Pairs[I].I);
    if NN.GetLastLayer().Output.GetClass() = Pairs[I].O.GetClass() then
      Inc(Hits);
  end;
  if Pairs.Count > 0 then
    Result := Hits / Pairs.Count
  else
    Result := 0;
end;

const
  NUM_SEEDS = 5; // average each arm over a few seeds so the headline is stable

function TrainOnce(Steps, Seed: integer;
                   Train, Validation: TNNetVolumePairList;
                   out Neurons, Weights: integer): TNeuralFloat;
var
  NN: TNNet;
  NFit: TNeuralFit;
begin
  RandSeed := Seed;
  NN := TNNet.Create();
  NFit := TNeuralFit.Create();
  try
    BuildNet(NN, Steps);
    Neurons := NN.CountNeurons();
    Weights := NN.CountWeights();

    NFit.MaxThreadNum := 1; // single-threaded for determinism on this tiny demo
    NFit.FileNameBase := GetTempDir + 'NeuralODE_autosave';
    // Each Euler step scales f by h=1/Steps, so the effective gradient reaching
    // f's shared weights shrinks ~1/Steps. Scale the LR by Steps to keep the
    // per-weight update comparable across the sweep (a fair fixed-budget compare;
    // the parameter count is still identical).
    NFit.InitialLearningRate := 0.05 * Steps;
    NFit.LearningRateDecay := 0;
    NFit.L2Decay := 0;
    NFit.Verbose := false;
    NFit.HideMessages();
    NFit.HasFlipX := false;
    NFit.HasFlipY := false;
    NFit.InferHitFn := @LocalClassCompare;
    NFit.Fit(NN, Train, Validation, nil, BATCH_SIZE, NUM_EPOCHS);

    Result := EvaluateAccuracy(NN, Validation);
  finally
    NFit.Free;
    NN.Free;
  end;
end;

function RunOne(Steps: integer;
                Train, Validation: TNNetVolumePairList): TArmResult;
var
  s: integer;
  Acc, Sum: TNeuralFloat;
begin
  Result.Steps := Steps;
  // Average over NUM_SEEDS independent runs: the parameter count is identical
  // for every seed/Steps, only the integration depth changes.
  Sum := 0;
  for s := 0 to NUM_SEEDS - 1 do
  begin
    Acc := TrainOnce(Steps, SEED + s, Train, Validation,
                     Result.Neurons, Result.Weights);
    Sum := Sum + Acc;
  end;
  Result.ValAcc := Sum / NUM_SEEDS;
end;

procedure RunAlgo();
const
  STEP_SWEEP: array[0..2] of integer = (1, 2, 4);
var
  TrainingPairs, ValidationPairs: TNNetVolumePairList;
  Results: array[0..2] of TArmResult;
  i: integer;
  StartTime, EndTime: TDateTime;
begin
  WriteLn('Neural ODE (continuous-depth residual) constant-parameter sweep.');
  WriteLn('Trunk = ONE TNNet.AddNeuralODEBlock(HiddenDim=', HIDDEN_DIM,
          '), d_model=', D_MODEL, ', on synthetic 2-class concentric rings.');
  WriteLn('Sweep Steps in {1,2,4}: parameter count is INDEPENDENT of Steps',
          ' (one shared f).');
  WriteLn(NUM_EPOCHS, ' epochs, ', TRAIN_SIZE, ' train / ', VAL_SIZE,
          ' val pairs, LR=0.05*Steps, RandSeed=', SEED, '.');
  WriteLn;

  // Same data for every arm (generated once, before the sweep).
  RandSeed := SEED;
  TrainingPairs   := CreateSpiralPairList(TRAIN_SIZE);
  ValidationPairs := CreateSpiralPairList(VAL_SIZE);
  try
    StartTime := Now;
    for i := 0 to High(STEP_SWEEP) do
    begin
      Write('Training Steps=', STEP_SWEEP[i], ' ...');
      Results[i] := RunOne(STEP_SWEEP[i], TrainingPairs, ValidationPairs);
      WriteLn(' done.');
    end;
    EndTime := Now;
  finally
    ValidationPairs.Free;
    TrainingPairs.Free;
  end;

  WriteLn;
  WriteLn('=== Steps vs accuracy vs parameter count ===');
  WriteLn('steps  neurons  weights  val_accuracy (mean of ', NUM_SEEDS, ' seeds)');
  for i := 0 to High(STEP_SWEEP) do
    WriteLn(Results[i].Steps:5, Results[i].Neurons:9, Results[i].Weights:9,
            '      ', FormatFloat('0.000', Results[i].ValAcc));
  WriteLn;
  WriteLn('Note: weights/neurons are CONSTANT across the sweep (shared f);');
  WriteLn('accuracy stays roughly flat as Steps grows -- continuous depth at a');
  WriteLn('fixed parameter budget. This is the Neural ODE "depth for free" point.');
  WriteLn;
  WriteLn('Total wall time: ', FormatFloat('0.00', (EndTime - StartTime)*86400), ' s');
end;

begin
  // Tiny dims: single-threaded for determinism and to keep the demo honest/fast.
  SetExceptionMask([exInvalidOp, exDenormalized, exZeroDivide,
                    exOverflow, exUnderflow, exPrecision]);
  RandSeed := SEED;
  RunAlgo();
end.
