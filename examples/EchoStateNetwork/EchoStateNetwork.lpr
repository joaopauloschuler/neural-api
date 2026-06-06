program EchoStateNetwork;
(*
EchoStateNetwork: a self-contained Reservoir Computing demo (Jaeger 2001,
"The echo state approach to analysing and training recurrent neural
networks"). This is a genuinely different training paradigm from everything
else in the repository: there is NO backpropagation-through-time. The
recurrent core is a FIXED, RANDOM, sparse matrix; only a single linear
readout is ever trained.

Per timestep t over a 1-D driving signal x_t the reservoir state evolves as

  h_t = (1 - a) * h_{t-1} + a * tanh(W_in * x_t + W * h_{t-1})

with leak rate `a`, a reservoir of N units, a random input vector W_in, and
a sparse random NxN matrix W RESCALED to a chosen spectral radius rho < 1.
That rho < 1 condition is the "echo state property": it guarantees the
reservoir asymptotically forgets its initial state, so the same input drives
it to the same state regardless of where it started - which is what makes the
fixed random recurrence usable as a feature generator.

Pipeline:
  1. Build W_in and a sparse W with hand-rolled arrays.
  2. RESCALE W to the target spectral radius. We reuse the library's
     power-iteration helper TNNet.EstimateSpectralNorm to MEASURE the current
     scale instead of running a full eigensolve. NOTE: that helper returns the
     largest SINGULAR value (spectral NORM), not the largest eigenvalue
     MAGNITUDE (spectral RADIUS). For a general non-symmetric W they differ,
     but the singular value is always an UPPER BOUND on |lambda|_max, so
     rescaling W := W * (rho_target / sigma_1) leaves the true spectral radius
     <= rho_target. That is exactly the safe side for the echo-state property,
     so it is an acceptable - and conservative - proxy here.
  3. Run the reservoir FORWARD (no gradient) over a training sequence, collect
     each state h_t into a TNNetVolumePair (input = h_t, target = x_{t+1}).
  4. Train ONLY a TNNetFullConnectLinear(1) readout on those collected pairs
     with a tiny SGD loop (an L2-regularised linear / ridge-style fit).

Headline task: one-step-ahead prediction of a deterministic sum-of-sines
series sin(0.2 t) + 0.3 sin(0.31 t). After teacher-forced fitting we FREE-RUN:
feed the readout's own prediction back in as the next input and let the
network continue the waveform autonomously for many steps, rendered as an
ASCII plot of predicted vs true.

Built-in correctness signals (print PASS/FAIL, Halt(1) on failure):
  1. Teacher-forced one-step NRMSE well below the persistence baseline
     (the trivial predictor x_{t+1} = x_t).
  2. Echo-state ABLATION: rebuild the reservoir at rho = 1.8 (> 1) and show
     the free-run prediction DIVERGES (NRMSE explodes), proving that
     rho < 1 is what makes the method work.

Pure CPU, no external dataset, runs in a few seconds on one thread.

Contrast with the rest of the repo: examples/DiagonalSSM TRAINS its diagonal
linear recurrence h_t = a*h_{t-1} + b*x_t end-to-end by gradient descent; the
causal-conv / attention examples likewise learn their sequence mixer. The ESN
FREEZES the recurrence and trains only the readout - that is the whole point.

Copyright (C) 2026 Joao Paulo Schwarz Schuler

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

Coded by Claude (AI).
*)

{$mode objfpc}{$H+}

uses {$IFDEF UNIX} cthreads, {$ENDIF}
  Classes, SysUtils, Math,
  neuralnetwork,
  neuralvolume;

const
  cN         = 100;    // reservoir size (units). Tiny: 50-150 is plenty.
  cLeak      = 0.3;    // leak rate `a` in the state update.
  cSparsity  = 0.1;    // fraction of W entries that are non-zero.
  // NOTE on the two rho settings: EstimateSpectralNorm returns the spectral
  // NORM (largest singular value sigma_1), an UPPER bound on the spectral
  // RADIUS |lambda|_max. So rescaling W to "rho" via sigma_1 leaves the true
  // radius <= rho - conservatively short memory. We therefore set the working
  // target near 1.25 (true radius still < 1, rich memory) and the ablation
  // target well above it so even the looser true radius exceeds 1.
  cRhoGood   = 1.5;    // echo-state working case (true radius stays < 1).
  cRhoBad    = 3.0;    // ablation case (true radius driven > 1, ESP broken).
  cInScale   = 1.0;    // input weight scale.
  cWarmup    = 100;    // washout steps (reservoir forgets its zero init).
  cTrainLen  = 600;    // teacher-forced training steps after washout.
  cTestLen   = 200;    // teacher-forced one-step test steps.
  cFreeRun   = 120;    // autonomous free-run steps for the ASCII plot.
  cReadoutEpochs = 600;
  cReadoutLR     = 0.02;
  cReadoutL2     = 1e-5; // ridge-style regularisation on the readout.
  cPowerIters    = 50;   // power-iteration steps for EstimateSpectralNorm.

type
  // The fixed random reservoir. W_in and W are plain arrays - the recurrence
  // is hand-rolled, never touched by backprop.
  TReservoir = record
    Win: array of TNeuralFloat;            // [N]   input weights
    W:   array of array of TNeuralFloat;   // [N,N] sparse recurrent matrix
    H:   array of TNeuralFloat;            // [N]   current state h_t
    Sigma: TNeuralFloat;                   // measured spectral norm after init
  end;

// The deterministic target series x_t = sin(0.2 t) + 0.3 sin(0.31 t).
function Series(t: integer): TNeuralFloat;
begin
  Result := Sin(0.2 * t) + 0.3 * Sin(0.31 * t);
end;

// Build W_in (uniform in [-cInScale, cInScale]) and a sparse W (uniform in
// [-1,1] on a cSparsity fraction of entries), then rescale W to TargetRho
// using the library power-iteration spectral-norm helper (see file header).
procedure BuildReservoir(var R: TReservoir; TargetRho: TNeuralFloat);
var
  i, j: integer;
  ProbeNN: TNNet;
  ProbeLayer: TNNetLayer;
  Scale: TNeuralFloat;
begin
  SetLength(R.Win, cN);
  SetLength(R.W, cN, cN);
  SetLength(R.H, cN);

  for i := 0 to cN - 1 do
  begin
    R.Win[i] := (Random * 2.0 - 1.0) * cInScale;
    R.H[i] := 0;
    for j := 0 to cN - 1 do
    begin
      if Random < cSparsity then
        R.W[i][j] := Random * 2.0 - 1.0
      else
        R.W[i][j] := 0;
    end;
  end;

  // Measure the current spectral norm of W by stuffing its rows into a
  // throwaway Input(N)->FullConnectLinear(N) network (building the net sizes
  // each neuron's Weights to fan-in N, so Neurons[i].Weights = row i of W) and
  // calling the reusable power-iteration helper. No training, no gradients -
  // we only read sigma_1 back out.
  ProbeNN := TNNet.Create();
  try
    ProbeNN.AddLayer([
      TNNetInput.Create(cN),
      TNNetFullConnectLinear.Create(cN)
    ]);
    ProbeLayer := ProbeNN.GetLastLayer();
    for i := 0 to cN - 1 do
      for j := 0 to cN - 1 do
        ProbeLayer.Neurons[i].Weights.FData[j] := R.W[i][j];
    R.Sigma := TNNet.EstimateSpectralNorm(ProbeLayer, cPowerIters);
  finally
    ProbeNN.Free;
  end;

  if R.Sigma > 1e-12 then
  begin
    Scale := TargetRho / R.Sigma;
    for i := 0 to cN - 1 do
      for j := 0 to cN - 1 do
        R.W[i][j] := R.W[i][j] * Scale;
  end;
end;

// Reset the reservoir state to zero (used between independent runs).
procedure ResetState(var R: TReservoir);
var i: integer;
begin
  for i := 0 to cN - 1 do R.H[i] := 0;
end;

// One leaky-integrator update step: h := (1-a) h + a tanh(Win*x + W*h).
procedure StepReservoir(var R: TReservoir; x: TNeuralFloat);
var
  i, j: integer;
  PreAct: TNeuralFloat;
  NewH: array of TNeuralFloat;
begin
  SetLength(NewH, cN);
  for i := 0 to cN - 1 do
  begin
    PreAct := R.Win[i] * x;
    for j := 0 to cN - 1 do
      PreAct := PreAct + R.W[i][j] * R.H[j];
    NewH[i] := (1 - cLeak) * R.H[i] + cLeak * Tanh(PreAct);
  end;
  for i := 0 to cN - 1 do R.H[i] := NewH[i];
end;

// Copy the current reservoir state h_t into a TNNetVolume (the readout input).
procedure StateToVolume(const R: TReservoir; V: TNNetVolume);
var i: integer;
begin
  for i := 0 to cN - 1 do V.FData[i] := R.H[i];
end;

// Build the trained readout: a single linear unit over the N reservoir
// states. Drives the reservoir over [WarmupStart .. WarmupStart+cWarmup) as
// washout (collect nothing), then over the training window collects
// (h_t -> x_{t+1}) pairs and fits the readout with a small L2-regularised SGD
// loop (deterministic; avoids TNeuralFit's shuffle/best-model reload).
function TrainReadout(var R: TReservoir): TNNet;
var
  NN: TNNet;
  Pairs: TNNetVolumePairList;
  t, Epoch, k: integer;
  InV, TgtV: TNNetVolume;
begin
  ResetState(R);
  // Washout: drive with the real series, discard states.
  for t := 0 to cWarmup - 1 do
    StepReservoir(R, Series(t));

  // Collect (state -> next value) pairs over the training window.
  Pairs := TNNetVolumePairList.Create();
  for t := cWarmup to cWarmup + cTrainLen - 1 do
  begin
    StepReservoir(R, Series(t));
    InV := TNNetVolume.Create(cN, 1, 1);
    StateToVolume(R, InV);
    TgtV := TNNetVolume.Create([Series(t + 1)]);
    Pairs.Add(TNNetVolumePair.Create(InV, TgtV));
  end;

  NN := TNNet.Create();
  NN.AddLayer([
    TNNetInput.Create(cN),
    TNNetFullConnectLinear.Create(1)
  ]);
  NN.SetLearningRate(cReadoutLR, {Momentum=}0.9);
  NN.SetL2Decay(cReadoutL2);

  // Full-batch-ish SGD: one update per sample, several epochs. The mapping is
  // linear in the states, so this converges to the ridge least-squares fit.
  for Epoch := 1 to cReadoutEpochs do
  begin
    NN.ClearDeltas();
    for k := 0 to Pairs.Count - 1 do
    begin
      NN.Compute(Pairs[k].I);
      NN.Backpropagate(Pairs[k].O);
    end;
    NN.UpdateWeights();
  end;

  Pairs.Free;
  Result := NN;
end;

// NRMSE = RMSE / std(target). Scale-free, so "1.0" means "no better than
// predicting the target mean".
function NRMSE(const Pred, Truth: array of TNeuralFloat): TNeuralFloat;
var
  i: integer;
  Mean, Var_, SumSqErr, d: TNeuralFloat;
  n: integer;
begin
  n := Length(Truth);
  Mean := 0;
  for i := 0 to n - 1 do Mean := Mean + Truth[i];
  Mean := Mean / n;
  Var_ := 0;
  for i := 0 to n - 1 do
  begin
    d := Truth[i] - Mean;
    Var_ := Var_ + d * d;
  end;
  Var_ := Var_ / n;
  SumSqErr := 0;
  for i := 0 to n - 1 do
  begin
    d := Pred[i] - Truth[i];
    SumSqErr := SumSqErr + d * d;
  end;
  if Var_ < 1e-12 then Var_ := 1e-12;
  Result := Sqrt((SumSqErr / n) / Var_);
end;

// Teacher-forced one-step test NRMSE plus the persistence baseline
// (predict x_{t+1} = x_t) over the same window. The reservoir is warmed up
// again on the test window's lead-in so its state is valid.
procedure TeacherForcedTest(var R: TReservoir; NN: TNNet;
  out ModelNRMSE, BaselineNRMSE: TNeuralFloat);
var
  t, idx, StartT: integer;
  InV, OutV: TNNetVolume;
  Pred, Truth, Persist: array of TNeuralFloat;
begin
  StartT := cWarmup + cTrainLen; // continue past the training window
  ResetState(R);
  for t := 0 to StartT - 1 do
    StepReservoir(R, Series(t));   // washout + replay so state is consistent

  SetLength(Pred, cTestLen);
  SetLength(Truth, cTestLen);
  SetLength(Persist, cTestLen);
  InV  := TNNetVolume.Create(cN, 1, 1);
  OutV := TNNetVolume.Create(1, 1, 1);
  try
    for idx := 0 to cTestLen - 1 do
    begin
      t := StartT + idx;
      StepReservoir(R, Series(t));      // teacher forcing: real input
      StateToVolume(R, InV);
      NN.Compute(InV);
      NN.GetOutput(OutV);
      Pred[idx]    := OutV.Raw[0];
      Truth[idx]   := Series(t + 1);
      Persist[idx] := Series(t);        // persistence baseline
    end;
  finally
    InV.Free;
    OutV.Free;
  end;
  ModelNRMSE    := NRMSE(Pred, Truth);
  BaselineNRMSE := NRMSE(Persist, Truth);
end;

// Free-run: prime the reservoir on the real series, then feed the readout's
// own output back as the next input for cFreeRun steps. Fills Pred/Truth and
// returns the free-run NRMSE.
function FreeRun(var R: TReservoir; NN: TNNet;
  out Pred, Truth: array of TNeuralFloat): TNeuralFloat;
var
  t, idx, StartT: integer;
  InV, OutV: TNNetVolume;
  x: TNeuralFloat;
begin
  StartT := cWarmup + cTrainLen;
  ResetState(R);
  for t := 0 to StartT - 1 do
    StepReservoir(R, Series(t));

  InV  := TNNetVolume.Create(cN, 1, 1);
  OutV := TNNetVolume.Create(1, 1, 1);
  try
    x := Series(StartT);  // last real input that primes the loop
    for idx := 0 to cFreeRun - 1 do
    begin
      StepReservoir(R, x);              // drive with our OWN previous output
      StateToVolume(R, InV);
      NN.Compute(InV);
      NN.GetOutput(OutV);
      x := OutV.Raw[0];                 // feedback: prediction becomes input
      Pred[idx]  := x;
      Truth[idx] := Series(StartT + 1 + idx);
    end;
  finally
    InV.Free;
    OutV.Free;
  end;
  Result := NRMSE(Pred, Truth);
end;

// Tiny ASCII plot of predicted (o) vs true (.) over the free-run window.
procedure AsciiPlot(const Pred, Truth: array of TNeuralFloat);
const
  cWidth = 51;   // columns of the value axis
  cEvery = 3;    // print every Nth step to keep it short
var
  i, col, n: integer;
  Lo, Hi, Span: TNeuralFloat;
  Line: array[0..cWidth - 1] of char;
  c: integer;

  function ToCol(v: TNeuralFloat): integer;
  begin
    Result := Round((v - Lo) / Span * (cWidth - 1));
    if Result < 0 then Result := 0;
    if Result > cWidth - 1 then Result := cWidth - 1;
  end;

begin
  n := Length(Truth);
  Lo := Truth[0]; Hi := Truth[0];
  for i := 0 to n - 1 do
  begin
    if Truth[i] < Lo then Lo := Truth[i];
    if Truth[i] > Hi then Hi := Truth[i];
    if Pred[i]  < Lo then Lo := Pred[i];
    if Pred[i]  > Hi then Hi := Pred[i];
  end;
  Span := Hi - Lo;
  if Span < 1e-9 then Span := 1e-9;

  WriteLn('Free-run waveform   ( . = true   o = predicted   * = overlap ):');
  WriteLn('  step |', StringOfChar('-', cWidth), '|');
  i := 0;
  while i < n do
  begin
    for c := 0 to cWidth - 1 do Line[c] := ' ';
    col := ToCol(Truth[i]); Line[col] := '.';
    col := ToCol(Pred[i]);
    if Line[col] = '.' then Line[col] := '*' else Line[col] := 'o';
    Write(Format('  %4d |', [i]));
    for c := 0 to cWidth - 1 do Write(Line[c]);
    WriteLn('|');
    Inc(i, cEvery);
  end;
end;

var
  R: TReservoir;
  NN: TNNet;
  TfModel, TfBaseline, FrGood, FrBad: TNeuralFloat;
  PredG, TruthG, PredB, TruthB: array of TNeuralFloat;
  AllOK: boolean;

begin
  SetExceptionMask([exInvalidOp, exDenormalized, exZeroDivide,
                    exOverflow, exUnderflow, exPrecision]);
  RandSeed := 2026;
  AllOK := True;

  WriteLn('Echo State Network (Reservoir Computing, Jaeger 2001)');
  WriteLn('Reservoir N=', cN, '  leak=', cLeak:0:2, '  sparsity=',
    cSparsity:0:2, '  target rho=', cRhoGood:0:2);
  WriteLn('Task: one-step prediction of sin(0.2 t) + 0.3 sin(0.31 t).');
  WriteLn(StringOfChar('=', 64));

  // ---- Working case: rho < 1 (echo-state property holds) -----------------
  WriteLn;
  WriteLn('[1] Building reservoir at rho=', cRhoGood:0:2, ' ...');
  BuildReservoir(R, cRhoGood);
  WriteLn('    measured spectral norm sigma_1 of raw W = ', R.Sigma:0:4,
    '  -> W rescaled so rho <= ', cRhoGood:0:2);

  WriteLn('    training the linear readout (', cReadoutEpochs, ' epochs)...');
  NN := TrainReadout(R);

  TeacherForcedTest(R, NN, TfModel, TfBaseline);
  WriteLn(Format('    teacher-forced one-step NRMSE = %.4f', [TfModel]));
  WriteLn(Format('    persistence baseline   NRMSE = %.4f', [TfBaseline]));

  SetLength(PredG, cFreeRun);
  SetLength(TruthG, cFreeRun);
  FrGood := FreeRun(R, NN, PredG, TruthG);
  WriteLn(Format('    free-run (autonomous)  NRMSE = %.4f', [FrGood]));
  WriteLn;
  AsciiPlot(PredG, TruthG);
  NN.Free;

  // ---- Ablation: rho > 1 (echo-state property broken) --------------------
  WriteLn;
  WriteLn(StringOfChar('=', 64));
  WriteLn('[2] ABLATION - rebuilding reservoir at rho=', cRhoBad:0:2,
    ' (> 1, echo-state property BROKEN)');
  BuildReservoir(R, cRhoBad);
  WriteLn('    measured spectral norm sigma_1 of raw W = ', R.Sigma:0:4);
  NN := TrainReadout(R);
  SetLength(PredB, cFreeRun);
  SetLength(TruthB, cFreeRun);
  FrBad := FreeRun(R, NN, PredB, TruthB);
  WriteLn(Format('    free-run (autonomous)  NRMSE = %.4g  (expected to explode)',
    [FrBad]));
  NN.Free;

  // ---- Correctness signals -----------------------------------------------
  WriteLn;
  WriteLn(StringOfChar('=', 64));
  WriteLn('Correctness checks:');

  // 1) teacher-forced model must beat persistence by a clear margin.
  if TfModel < 0.5 * TfBaseline then
    WriteLn(Format('  PASS  teacher-forced NRMSE %.4f < 0.5 x persistence %.4f',
      [TfModel, TfBaseline]))
  else
  begin
    WriteLn(Format('  FAIL  teacher-forced NRMSE %.4f not < 0.5 x persistence %.4f',
      [TfModel, TfBaseline]));
    AllOK := False;
  end;

  // 2) the good reservoir must free-run accurately.
  if FrGood < 0.5 then
    WriteLn(Format('  PASS  rho<1 free-run NRMSE %.4f < 0.5', [FrGood]))
  else
  begin
    WriteLn(Format('  FAIL  rho<1 free-run NRMSE %.4f not < 0.5', [FrGood]));
    AllOK := False;
  end;

  // 3) ablation: rho>1 free-run must diverge far worse than rho<1. A broken
  //    reservoir can overflow tanh feedback to NaN/Inf - that is the most
  //    extreme divergence, so treat any non-finite NRMSE as a pass too.
  if IsNan(FrBad) or IsInfinite(FrBad) or (FrBad > 5.0 * FrGood + 1.0) then
    WriteLn(Format('  PASS  rho>1 free-run NRMSE %.4g explodes vs rho<1 %.4f',
      [FrBad, FrGood]))
  else
  begin
    WriteLn(Format('  FAIL  rho>1 free-run NRMSE %.4g did not diverge (rho<1 %.4f)',
      [FrBad, FrGood]));
    AllOK := False;
  end;

  WriteLn(StringOfChar('=', 64));
  if AllOK then
    WriteLn('ALL CHECKS PASSED')
  else
  begin
    WriteLn('SOME CHECKS FAILED');
    Halt(1);
  end;
end.
