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

Coded by Claude (AI).
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

// Two interleaving half-moons: a 2-class problem that is NOT linearly separable
// but, unlike concentric rings, CAN be untangled by a 2-D diffeomorphism (the
// flow of a Neural ODE) -- so it is the right toy for the trajectory picture.
function CreateMoonsPairList(MaxCnt: integer): TNNetVolumePairList;
var
  Cnt, Cls: integer;
  Theta, Px, Py: TNeuralFloat;
  Target: TNNetVolume;
begin
  Result := TNNetVolumePairList.Create();
  for Cnt := 1 to MaxCnt do
  begin
    Cls := Random(2);
    Theta := Random(1000) / 1000.0 * PI;       // half-circle
    if Cls = 0 then
    begin
      // upper moon
      Px := Cos(Theta) - 0.5;
      Py := Sin(Theta) - 0.20;
    end
    else
    begin
      // lower moon, shifted right and down so the two interleave
      Px := -Cos(Theta) + 0.5;
      Py := -Sin(Theta) + 0.20;
    end;
    Px := Px + (Random(200) - 100) / 2000.0;
    Py := Py + (Random(200) - 100) / 2000.0;
    Target := TNNetVolume.Create(2);
    Target.Fill(0);
    Target.FData[Cls] := 1;
    Result.Add(
      TNNetVolumePair.Create(TNNetVolume.Create([Px, Py]), Target)
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

// ---------------------------------------------------------------------------
// 2-D trajectory ASCII-frame visualisation (the textbook Neural-ODE picture).
//
// We train a tiny classifier whose ODE state is genuinely 2-D (d_model = 2,
// integrated with the RK2/midpoint method), so the integrated state can be
// drawn directly as an (x,y) scatter -- no projection needed. Then we replay a
// grid of validation points through the trunk and, at EACH integration step,
// render an ASCII frame of where the two classes sit. As t advances the shared
// flow f untangles the two interleaved half-moons until a straight line
// separates them -- continuous-depth deformation made visible in pure stdout
// ASCII.
// ---------------------------------------------------------------------------
const
  VIZ_STEPS  = 5;    // integration steps to render as frames
  VIZ_HID    = 16;
  VIZ_EPOCHS = 50;
  VIZ_TRAIN  = 500;
  VIZ_POINTS = 220;  // points scattered into each frame
  GRID_W     = 41;
  GRID_H     = 17;

procedure RenderFrame(StepIdx: integer; T: TNeuralFloat;
                      const Xs, Ys: array of TNeuralFloat;
                      const Cls: array of integer; Count: integer);
var
  Grid: array of array of Char;
  gx, gy, p, row, col: integer;
  minX, maxX, minY, maxY, sx, sy, cx, cy, halfX, halfY: TNeuralFloat;
  c: Char;
begin
  SetLength(Grid, GRID_H, GRID_W);
  for row := 0 to GRID_H - 1 do
    for col := 0 to GRID_W - 1 do
      Grid[row][col] := ' ';

  // Auto-scale the plot window to this frame's data (the integrated state can
  // grow/shrink as the flow deforms the plane); centre it and pad a little so
  // the class structure is always visible regardless of absolute scale.
  minX := Xs[0]; maxX := Xs[0]; minY := Ys[0]; maxY := Ys[0];
  for p := 1 to Count - 1 do
  begin
    if Xs[p] < minX then minX := Xs[p];
    if Xs[p] > maxX then maxX := Xs[p];
    if Ys[p] < minY then minY := Ys[p];
    if Ys[p] > maxY then maxY := Ys[p];
  end;
  cx := (minX + maxX) * 0.5;  cy := (minY + maxY) * 0.5;
  halfX := (maxX - minX) * 0.55 + 1e-3;
  halfY := (maxY - minY) * 0.55 + 1e-3;
  minX := cx - halfX;  maxX := cx + halfX;
  minY := cy - halfY;  maxY := cy + halfY;
  sx := (GRID_W - 1) / (maxX - minX);
  sy := (GRID_H - 1) / (maxY - minY);

  for p := 0 to Count - 1 do
  begin
    gx := Round( (Xs[p] - minX) * sx );
    gy := Round( (maxY - Ys[p]) * sy );  // y down in text
    if (gx < 0) or (gx > GRID_W - 1) or (gy < 0) or (gy > GRID_H - 1) then
      Continue;
    if Cls[p] = 0 then c := 'o' else c := '#';
    // '#' wins a collision so the outer class stays visible.
    if (Grid[gy][gx] = ' ') or (c = '#') then
      Grid[gy][gx] := c;
  end;

  WriteLn('--- step ', StepIdx, '/', VIZ_STEPS, '   (t = ',
          FormatFloat('0.00', T), ')   o = class 0   # = class 1 ---');
  for row := 0 to GRID_H - 1 do
  begin
    Write('  |');
    for col := 0 to GRID_W - 1 do
      Write(Grid[row][col]);
    WriteLn('|');
  end;
  WriteLn;
end;

procedure RunTrajectoryViz();
var
  NN: TNNet;
  NFit: TNeuralFit;
  Train, Val: TNNetVolumePairList;
  StepY: array of TNNetLayer;       // each step's integrated state (the Y Sums)
  AllSums: array of integer;
  li, s, p, n, nSums: integer;
  Xs, Ys: array of TNeuralFloat;
  Cls: array of integer;
begin
  WriteLn;
  WriteLn('============================================================');
  WriteLn(' 2-D trajectory visualisation (RK2 / midpoint integrator)');
  WriteLn('============================================================');
  WriteLn('Training a d_model=2 ODE trunk (', VIZ_STEPS,
          ' midpoint steps) on two interleaving half-moons,');
  WriteLn('then drawing the state at every integration step as it untangles.');
  WriteLn;

  RandSeed := SEED;
  Train := CreateMoonsPairList(VIZ_TRAIN);
  Val   := CreateMoonsPairList(VIZ_POINTS);

  NN := TNNet.Create();
  NFit := TNeuralFit.Create();
  try
    // State lives directly in a 2-D Depth so we can scatter it as (x,y).
    NN.AddLayer( TNNetInput.Create(2) );
    NN.AddLayer( TNNetReshape.Create(1, 1, 2) );
    NN.AddNeuralODEBlock( NN.GetLastLayer(), VIZ_HID, VIZ_STEPS, odeMidpoint );
    NN.AddLayer( TNNetFullConnectLinear.Create(2) );
    NN.AddLayer( TNNetSoftMax.Create() );

    NFit.MaxThreadNum := 1;
    NFit.FileNameBase := GetTempDir + 'NeuralODE_viz_autosave';
    // Modest LR, no decay/L2: the flow learns to push the two moons apart
    // without collapsing the 2-D state onto a single line.
    NFit.InitialLearningRate := 0.03;
    NFit.LearningRateDecay := 0;
    NFit.L2Decay := 0;
    NFit.Verbose := false;
    NFit.HideMessages();
    NFit.HasFlipX := false;
    NFit.HasFlipY := false;
    NFit.InferHitFn := @LocalClassCompare;
    NFit.Fit(NN, Train, Val, nil, BATCH_SIZE, VIZ_EPOCHS);
    WriteLn('Trained. Validation accuracy: ',
            FormatFloat('0.000', EvaluateAccuracy(NN, Val)));
    WriteLn;

    // Collect the per-step integrated-state layers. In a midpoint block every
    // step ends with a TNNetSum that feeds the next step (or the head). There
    // are 2 Sums per step (the (h/2)*k1 midpoint Sum and the final h*k2 Sum);
    // the step-FINAL ones are every 2nd Sum, so we take indices 1,3,5,...
    SetLength(AllSums, 0);
    for li := 0 to NN.CountLayers() - 1 do
      if NN.Layers[li] is TNNetSum then
      begin
        SetLength(AllSums, Length(AllSums) + 1);
        AllSums[High(AllSums)] := li;
      end;
    nSums := Length(AllSums);
    SetLength(StepY, VIZ_STEPS);
    for s := 0 to VIZ_STEPS - 1 do
      StepY[s] := NN.Layers[ AllSums[ 2 * s + 1 ] ];  // final Sum of step s+1

    SetLength(Xs, Val.Count);
    SetLength(Ys, Val.Count);
    SetLength(Cls, Val.Count);

    // Frame 0: the raw input (t = 0), before any flow.
    for p := 0 to Val.Count - 1 do
    begin
      Xs[p] := Val[p].I.FData[0];
      Ys[p] := Val[p].I.FData[1];
      Cls[p] := Val[p].O.GetClass();
    end;
    RenderFrame(0, 0.0, Xs, Ys, Cls, Val.Count);

    // One frame per integration step: read the 2-D state off StepY[s].
    for s := 0 to VIZ_STEPS - 1 do
    begin
      for p := 0 to Val.Count - 1 do
      begin
        NN.Compute(Val[p].I);
        Xs[p] := StepY[s].Output.FData[0];
        Ys[p] := StepY[s].Output.FData[1];
        Cls[p] := Val[p].O.GetClass();
      end;
      RenderFrame(s + 1, (s + 1) / VIZ_STEPS, Xs, Ys, Cls, Val.Count);
    end;

    WriteLn('At t=0 the two half-moons (o / #) interleave and overlap; as t');
    WriteLn('advances the single shared flow f deforms the plane until the two');
    WriteLn('classes pull apart into linearly-separable clusters -- the');
    WriteLn('textbook Neural ODE "untangling" picture.');
    n := nSums; // silence "unused" on some FPC versions
    if n < 0 then WriteLn(n);
  finally
    NFit.Free;
    NN.Free;
    Val.Free;
    Train.Free;
  end;
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
  RunTrajectoryViz();
end.
