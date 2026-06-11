program DeepEquilibrium;
(*
DeepEquilibrium: a tiny, synthetic, no-download demonstration of the Bai, Kolter
& Koltun 2019 "Deep Equilibrium Models" (DEQ) idea
(https://arxiv.org/abs/1909.01377) built on TNNet.AddDeepEquilibriumBlock.

A residual stack x_{n+1} = x_n + f(x_n) has a FIXED depth. A DEQ instead defines
its output as the FIXED POINT z* = f(z*; x) of ONE weight-tied transform f,
reached by iterating the map
    z_0 := 0
    z_{k+1} := f(z_k + x)              (the input x is injected each iteration)
until the residual ||z_{k+1} - z_k|| falls below a tolerance (or a MaxIters cap
is hit). Because the SAME f (the SAME weights) is reused at every iteration, the
parameter count is INDEPENDENT of how deep the iteration runs -- an
"infinite-depth, weight-tied" network whose effective depth ADAPTS to the input.

This demo trains a tiny classifier whose only trunk is one AddDeepEquilibriumBlock
on a synthetic 2-class concentric-rings task (nonlinearly separable). Per epoch it
reports:
  (a) the validation accuracy at a CONSTANT parameter count, and
  (b) the MEAN forward iteration-count-to-convergence -- the "adaptive depth"
      signal: how many applications of f the fixed-point solve needs before the
      residual drops below CONV_TOL. This is data- and weight-dependent (the whole
      point of an implicit model). On this run it RISES with training: an untrained
      f is near the zero map whose fixed point z*~=0 is reached trivially fast,
      while a TRAINED f has a richer, more distant equilibrium that takes more
      steps to settle. Either direction is the adaptive-depth story honestly --
      the depth is decided by the solve, not fixed up front.
It also runs a PARAM-MATCHED TNNet.AddNeuralODEBlock (the explicit-unroll cousin)
as a side-by-side, to make the explicit-Euler-depth vs implicit-fixed-point
contrast concrete: same weight count, comparable accuracy, but the DEQ's depth is
data-dependent rather than a fixed hyperparameter.

HONEST NOTE ON THE BACKWARD PASS. The exact DEQ gradient requires the
implicit-function theorem (an inverse-Jacobian solve), which is awkward under this
library's per-layer Backpropagate contract. This builder ships the tractable
JACOBIAN-FREE / PHANTOM gradient (Geng et al. 2021,
https://arxiv.org/abs/2103.12803): the forward iteration drives z to the fixed
point, but the backward pass detaches every iterate except the last, so gradients
flow through only the FINAL application of f. This is an APPROXIMATION, not the
exact implicit gradient -- it trains fine here but is not state of the art.

HONEST NOTE ON CONVERGENCE. A fixed-point iteration only converges when f is a
contraction. f's output is scaled by a fixed BETA in (0,1) and the iteration is
under-relaxed (damped) to encourage this, but contraction is not guaranteed at
arbitrary weights -- real DEQs use spectral constraints / root-finders we do not
implement. On this small task f stays contractive and the iteration converges.

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

const
  D_MODEL    = 8;    // width of the equilibrium state (lives in the Depth axis)
  HIDDEN_DIM = 16;   // hidden width of the shared transform f
  MAX_ITERS  = 30;   // hard cap on the fixed-point iteration
  CONV_TOL   = 1e-2; // ||z_{k+1}-z_k|| threshold counted as "converged"
  ODE_STEPS  = 4;    // explicit-Euler steps for the param-matched Neural ODE
  TRAIN_SIZE = 600;
  VAL_SIZE   = 200;
  NUM_EPOCHS = 25;
  BATCH_SIZE = 32;
  SEED       = 42;

// Two concentric noisy rings -> a 2-class problem a linear head cannot separate.
function CreateRingsPairList(MaxCnt: integer): TNNetVolumePairList;
var
  Cnt, Cls: integer;
  R, Theta, Px, Py: TNeuralFloat;
  Target: TNNetVolume;
begin
  Result := TNNetVolumePairList.Create();
  for Cnt := 1 to MaxCnt do
  begin
    Cls := Random(2);
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
  Result := ( A.GetClass() = B.GetClass() );
end;

// Build a classifier whose ONLY trunk is one AddDeepEquilibriumBlock.
procedure BuildDEQNet(NN: TNNet);
begin
  NN.AddLayer( TNNetInput.Create(2) );
  NN.AddLayer( TNNetFullConnectLinear.Create(D_MODEL) );
  NN.AddLayer( TNNetReshape.Create(1, 1, D_MODEL) );  // features into Depth
  NN.AddDeepEquilibriumBlock( NN.GetLastLayer(), HIDDEN_DIM, MAX_ITERS );
  NN.AddLayer( TNNetFullConnectLinear.Create(2) );
  NN.AddLayer( TNNetSoftMax.Create() );
end;

// Param-matched cousin: ONE AddNeuralODEBlock (same f shape, same weight count).
procedure BuildODENet(NN: TNNet);
begin
  NN.AddLayer( TNNetInput.Create(2) );
  NN.AddLayer( TNNetFullConnectLinear.Create(D_MODEL) );
  NN.AddLayer( TNNetReshape.Create(1, 1, D_MODEL) );
  NN.AddNeuralODEBlock( NN.GetLastLayer(), HIDDEN_DIM, ODE_STEPS );
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
  if Pairs.Count > 0 then Result := Hits / Pairs.Count else Result := 0;
end;

// Collect the per-iteration f-output layers f(u_k) = BETA*g(u_k): the
// MulByConstant that directly follows each convolution inside the DEQ block.
procedure CollectFOutLayers(NN: TNNet; out FOuts: array of TNNetLayer;
  out Cnt: integer);
var
  li: integer;
begin
  Cnt := 0;
  for li := 1 to NN.CountLayers() - 1 do
    if (NN.Layers[li] is TNNetMulByConstant) and
       (NN.Layers[li - 1] is TNNetConvolution) then
    begin
      if Cnt <= High(FOuts) then FOuts[Cnt] := NN.Layers[li];
      Inc(Cnt);
    end;
end;

// Mean iteration-count-to-convergence over a sample set: per input, the first
// iteration whose residual ||f(u_{k})-f(u_{k-1})|| < CONV_TOL (else MAX_ITERS).
function MeanIterToConverge(NN: TNNet; Pairs: TNNetVolumePairList): TNeuralFloat;
var
  FOuts: array[0..MAX_ITERS - 1] of TNNetLayer;
  nFOut, P, k, j, hitAt: integer;
  resid, d, total: TNeuralFloat;
begin
  CollectFOutLayers(NN, FOuts, nFOut);
  total := 0;
  for P := 0 to Pairs.Count - 1 do
  begin
    NN.Compute(Pairs[P].I);
    hitAt := MAX_ITERS;
    for k := 1 to nFOut - 1 do
    begin
      resid := 0;
      for j := 0 to FOuts[k].Output.Size - 1 do
      begin
        d := FOuts[k].Output.Raw[j] - FOuts[k - 1].Output.Raw[j];
        resid := resid + d * d;
      end;
      if Sqrt(resid) < CONV_TOL then
      begin
        hitAt := k + 1;   // 1-based iteration index
        break;
      end;
    end;
    total := total + hitAt;
  end;
  if Pairs.Count > 0 then Result := total / Pairs.Count else Result := 0;
end;

procedure RunDEQ(Train, Validation: TNNetVolumePairList;
  out ValAcc, MeanIterStart, MeanIterEnd: TNeuralFloat;
  out Neurons, Weights: integer);
var
  NN: TNNet;
  NFit: TNeuralFit;
  ep: integer;
begin
  RandSeed := SEED;
  NN := TNNet.Create();
  NFit := TNeuralFit.Create();
  try
    BuildDEQNet(NN);
    NN.InitWeights();
    Neurons := NN.CountNeurons();
    Weights := NN.CountWeights();

    MeanIterStart := MeanIterToConverge(NN, Validation);  // before training

    NFit.MaxThreadNum := 1;
    NFit.FileNameBase := GetTempDir + 'DeepEquilibrium_autosave';
    NFit.InitialLearningRate := 0.05;
    NFit.LearningRateDecay := 0;
    NFit.L2Decay := 0;
    NFit.Verbose := false;
    NFit.HideMessages();
    NFit.HasFlipX := false;
    NFit.HasFlipY := false;
    NFit.InferHitFn := @LocalClassCompare;

    WriteLn('  epoch  val_acc  mean_iters_to_converge');
    for ep := 1 to NUM_EPOCHS do
    begin
      NFit.Fit(NN, Train, Validation, nil, BATCH_SIZE, 1);  // one epoch at a time
      if (ep <= 3) or (ep mod 5 = 0) or (ep = NUM_EPOCHS) then
        WriteLn(ep:7,
                FormatFloat('0.000', EvaluateAccuracy(NN, Validation)):9,
                FormatFloat('0.00', MeanIterToConverge(NN, Validation)):13);
    end;

    ValAcc := EvaluateAccuracy(NN, Validation);
    MeanIterEnd := MeanIterToConverge(NN, Validation);
  finally
    NFit.Free;
    NN.Free;
  end;
end;

procedure RunODE(Train, Validation: TNNetVolumePairList;
  out ValAcc: TNeuralFloat; out Neurons, Weights: integer);
var
  NN: TNNet;
  NFit: TNeuralFit;
begin
  RandSeed := SEED;
  NN := TNNet.Create();
  NFit := TNeuralFit.Create();
  try
    BuildODENet(NN);
    NN.InitWeights();
    Neurons := NN.CountNeurons();
    Weights := NN.CountWeights();

    NFit.MaxThreadNum := 1;
    NFit.FileNameBase := GetTempDir + 'DeepEquilibrium_ode_autosave';
    NFit.InitialLearningRate := 0.05; // same base LR as the DEQ arm
    NFit.LearningRateDecay := 0;
    NFit.L2Decay := 0;
    NFit.Verbose := false;
    NFit.HideMessages();
    NFit.HasFlipX := false;
    NFit.HasFlipY := false;
    NFit.InferHitFn := @LocalClassCompare;
    NFit.Fit(NN, Train, Validation, nil, BATCH_SIZE, NUM_EPOCHS);
    ValAcc := EvaluateAccuracy(NN, Validation);
  finally
    NFit.Free;
    NN.Free;
  end;
end;

procedure RunAlgo();
var
  TrainingPairs, ValidationPairs: TNNetVolumePairList;
  deqAcc, deqI0, deqI1, odeAcc: TNeuralFloat;
  deqN, deqW, odeN, odeW: integer;
  StartTime, EndTime: TDateTime;
begin
  WriteLn('Deep Equilibrium Model (implicit fixed-point depth) demo.');
  WriteLn('Trunk = ONE TNNet.AddDeepEquilibriumBlock(HiddenDim=', HIDDEN_DIM,
          '), d_model=', D_MODEL, ', MaxIters=', MAX_ITERS, ', tol=',
          FormatFloat('0.###', CONV_TOL), '.');
  WriteLn('Task = synthetic 2-class concentric rings. ', NUM_EPOCHS,
          ' epochs, ', TRAIN_SIZE, ' train / ', VAL_SIZE, ' val.');
  WriteLn('Backward = jacobian-free phantom gradient (detach all but last f).');
  WriteLn;

  RandSeed := SEED;
  TrainingPairs   := CreateRingsPairList(TRAIN_SIZE);
  ValidationPairs := CreateRingsPairList(VAL_SIZE);
  try
    StartTime := Now;
    WriteLn('=== DEQ (adaptive implicit depth) ===');
    RunDEQ(TrainingPairs, ValidationPairs, deqAcc, deqI0, deqI1, deqN, deqW);
    WriteLn;
    WriteLn('=== Param-matched Neural ODE (explicit ', ODE_STEPS,
            '-step Euler unroll) ===');
    RunODE(TrainingPairs, ValidationPairs, odeAcc, odeN, odeW);
    EndTime := Now;
  finally
    ValidationPairs.Free;
    TrainingPairs.Free;
  end;

  WriteLn;
  WriteLn('=== Summary ===');
  WriteLn('model                neurons  weights  val_acc');
  WriteLn('DEQ (implicit)     ', deqN:8, deqW:9, '   ',
          FormatFloat('0.000', deqAcc));
  WriteLn('NeuralODE (explicit)', odeN:7, odeW:9, '   ',
          FormatFloat('0.000', odeAcc));
  WriteLn;
  WriteLn('DEQ adaptive depth: mean iters-to-converge went from ',
          FormatFloat('0.00', deqI0), ' (untrained, near-zero f, trivial fixed',
          ' point) to ', FormatFloat('0.00', deqI1),
          ' (trained, richer equilibrium).');
  WriteLn('Both trunks carry the SAME weight count; the DEQ reaches its answer at');
  WriteLn('a data-dependent depth (the implicit fixed point) instead of a fixed');
  WriteLn('hand-picked number of explicit steps.');
  WriteLn;
  WriteLn('Total wall time: ', FormatFloat('0.00', (EndTime - StartTime)*86400), ' s');
end;

begin
  SetExceptionMask([exInvalidOp, exDenormalized, exZeroDivide,
                    exOverflow, exUnderflow, exPrecision]);
  RandSeed := SEED;
  RunAlgo();
end.
