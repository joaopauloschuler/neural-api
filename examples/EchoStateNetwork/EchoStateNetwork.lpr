program EchoStateNetwork;
(*
EchoStateNetwork: a self-contained Reservoir Computing demo (Jaeger 2001,
"The echo state approach to analysing and training recurrent neural
networks"). This is a different training paradigm from everything
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

The reservoir core is the reusable TNNetEchoStateReservoir layer (see
neural/neuralnetwork.pas). It OWNS the two frozen matrices W_in and W, owns the
leak rate a, and does the one-shot spectral-radius rescale of W at build time
via the library power-iteration helper TNNet.EstimateSpectralRadius (it measures
the TRUE largest eigenvalue MAGNITUDE rho = |lambda|_max directly with a W*v
iteration - no transpose step - so rescaling W := W * (rho_target / rho) targets
the exact spectral radius that governs the echo-state property). The layer is a
shape-(SeqLen,1,1) -> (SeqLen,1,N) sequence map: feed it a whole driving
sequence and it returns the whole sequence of reservoir states in one Compute.
The matrices are NEVER touched by a gradient.

Pipeline:
  1. Build an Input(1) -> TNNetEchoStateReservoir(N) net (the layer builds and
     spectral-rescales W_in and W itself).
  2. Run the reservoir FORWARD (no gradient) over a driving sequence; each
     output column h_t is a reservoir state. Pair (h_t -> x_{t+1}).
  3. Train ONLY a TNNetFullConnectLinear(1) readout on those collected pairs
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
  // NOTE on the two rho settings: the layer's spectral rescale targets the TRUE
  // spectral RADIUS rho = |lambda|_max directly. The echo-state property holds
  // iff rho < 1, so the working case sets rho just below 1 (rich memory, still
  // contractive) and the ablation drives rho > 1.
  cRhoGood   = 0.9;    // echo-state working case (true radius < 1, ESP holds).
  cRhoBad    = 1.8;    // ablation case (true radius > 1, ESP broken).
  cInScale   = 1.0;    // input weight scale.
  cSeed      = 20260628; // deterministic seed for the frozen reservoir matrices.
  cWarmup    = 100;    // washout steps (reservoir forgets its zero init).
  cTrainLen  = 600;    // teacher-forced training steps after washout.
  cTestLen   = 200;    // teacher-forced one-step test steps.
  cFreeRun   = 120;    // autonomous free-run steps for the ASCII plot.
  cReadoutEpochs = 600;
  cReadoutLR     = 0.02;
  cReadoutL2     = 1e-5; // ridge-style regularisation on the readout.
  // Longest driving sequence we ever feed (washout + train + test, and the
  // free-run prefix + appended predictions). The reservoir net's Input layer is
  // sized to this; shorter runs zero-pad the tail (harmless: the recurrence is
  // causal so states at index < Len never see the padding).
  cMaxLen    = cWarmup + cTrainLen + cTestLen + cFreeRun + 8;

// The deterministic target series x_t = sin(0.2 t) + 0.3 sin(0.31 t).
function Series(t: integer): TNeuralFloat;
begin
  Result := Sin(0.2 * t) + 0.3 * Sin(0.31 * t);
end;

// ----------------------------------------------------------------------------
// The reservoir is the reusable TNNetEchoStateReservoir layer wrapped in a
// one-layer Input(1) -> Reservoir(N) net. Because the layer re-inits its state
// to zero at the start of every Compute and runs a whole sequence at once, we
// drive it by REPLAYING a full driving sequence (washout + window) and reading
// the output column at the desired step. The echo-state property (rho < 1)
// guarantees the state after a few-dozen-step washout is independent of the
// zero initialisation, so this is exactly equivalent to a persistent step loop.
// ----------------------------------------------------------------------------

var
  gReservoir: TNNet;        // Input(1) -> TNNetEchoStateReservoir(cN)
  gResLayer: TNNetEchoStateReservoir;
  gDrive: TNNetVolume;      // reused (Len,1,1) driving-sequence buffer

// Build the reservoir net at the given target spectral radius. The layer builds
// and spectral-rescales its own W_in / W from cSeed.
procedure BuildReservoir(TargetRho: TNeuralFloat);
begin
  if Assigned(gReservoir) then gReservoir.Free;
  gReservoir := TNNet.Create();
  gReservoir.AddLayer([
    // 1-D driving signal laid out along SizeX (sized to the longest sequence;
    // shorter runs zero-pad the causal tail).
    TNNetInput.Create(cMaxLen, 1, 1),
    TNNetEchoStateReservoir.Create(cN, cLeak, TargetRho, cSparsity, cInScale, cSeed)
  ]);
  gResLayer := gReservoir.GetLastLayer() as TNNetEchoStateReservoir;
end;

// Run the reservoir over the driving values drive[0..Len-1] and return the
// reservoir-state OUTPUT volume (shape (Len,1,cN)). The returned volume is owned
// by the reservoir layer - do not free it; copy what you need.
function RunReservoir(const drive: array of TNeuralFloat; Len: integer): TNNetVolume;
var
  i: integer;
begin
  // Always a cMaxLen-long buffer (the Input layer's fixed shape); zero-pad the
  // tail beyond Len. The causal recurrence makes states at index < Len
  // independent of the padding, so reading those indices is exact.
  gDrive.Fill(0);
  for i := 0 to Len - 1 do
    gDrive.FData[gDrive.GetRawPos(i, 0, 0)] := drive[i];
  gReservoir.Compute(gDrive);
  Result := gResLayer.Output;
end;

// Copy reservoir state h at step `idx` (0-based into the last RunReservoir call)
// into volume V (shape (cN,1,1)) - the readout input layout.
procedure StateToVolume(States: TNNetVolume; idx: integer; V: TNNetVolume);
var i: integer;
begin
  for i := 0 to cN - 1 do
    V.FData[i] := States.FData[States.GetRawPos(idx, 0, i)];
end;

// Build the trained readout: a single linear unit over the N reservoir states.
// Drives the reservoir over the washout + training window, collects
// (h_t -> x_{t+1}) pairs and fits the readout with a small L2-regularised SGD
// loop (deterministic; avoids TNeuralFit's shuffle/best-model reload).
function TrainReadout(): TNNet;
var
  NN: TNNet;
  Pairs: TNNetVolumePairList;
  States: TNNetVolume;
  drive: array of TNeuralFloat;
  t, Epoch, k, Len: integer;
  InV, TgtV: TNNetVolume;
begin
  Len := cWarmup + cTrainLen;
  SetLength(drive, Len);
  for t := 0 to Len - 1 do drive[t] := Series(t);
  States := RunReservoir(drive, Len);

  // Collect (state -> next value) pairs over the training window (post-washout).
  Pairs := TNNetVolumePairList.Create();
  for t := cWarmup to cWarmup + cTrainLen - 1 do
  begin
    InV := TNNetVolume.Create(cN, 1, 1);
    StateToVolume(States, t, InV);
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

// ----------------------------------------------------------------------------
// Closed-form RIDGE (Tikhonov) readout - the CLASSIC ESN training.
//
// The readout is linear in the reservoir state, so the optimal weights are the
// one-shot ridge-regression solution. Collect the state matrix S (rows =
// timesteps, cols = N reservoir units PLUS a bias/intercept column) and the
// target matrix Y (a single column x_{t+1}). The ridge readout minimises
// ||S*Wout - Y||^2 + lambda||Wout||^2, whose normal equations are
//
//     (S^T S + lambda I) Wout = S^T Y          ->   A Wout = B
//
// We form A ((N+1)x(N+1)) and B ((N+1)x1) and solve the small dense system via
// neuralvolume's shared NeuralLinearSolve (Gauss-Jordan with partial pivoting).
// ----------------------------------------------------------------------------
function RidgeReadout(Lambda: TNeuralFloat): TNNet;
var
  NN: TNNet;
  States: TNNetVolume;
  drive: array of TNeuralFloat;
  t, i, j, d, rows, Len: integer;
  S: array of TNeuralFloat;   // rows x d   (d = N+1, last col = bias 1.0)
  Y: array of TNeuralFloat;   // rows x 1
  A: array of TNeuralFloat;   // d x d  normal-equations matrix
  Bmat: array of TNeuralFloat; // d x 1  right-hand side (becomes Wout)
  acc: TNeuralFloat;
begin
  d := cN + 1;                 // reservoir units + 1 bias/intercept column
  rows := cTrainLen;
  Len := cWarmup + cTrainLen;

  SetLength(drive, Len);
  for t := 0 to Len - 1 do drive[t] := Series(t);
  States := RunReservoir(drive, Len);

  SetLength(S, rows * d);
  SetLength(Y, rows * 1);
  for i := 0 to rows - 1 do
  begin
    t := cWarmup + i;
    for j := 0 to cN - 1 do
      S[i * d + j] := States.FData[States.GetRawPos(t, 0, j)];
    S[i * d + cN] := 1.0;             // bias/intercept column
    Y[i] := Series(t + 1);
  end;

  // A = S^T S + lambda*I.
  SetLength(A, d * d);
  for i := 0 to d - 1 do
    for j := 0 to d - 1 do
    begin
      acc := 0;
      for t := 0 to rows - 1 do acc := acc + S[t * d + i] * S[t * d + j];
      if i = j then acc := acc + Lambda;
      A[i * d + j] := acc;
    end;

  // B = S^T Y.
  SetLength(Bmat, d * 1);
  for i := 0 to d - 1 do
  begin
    acc := 0;
    for t := 0 to rows - 1 do acc := acc + S[t * d + i] * Y[t];
    Bmat[i] := acc;
  end;

  if not NeuralLinearSolve(A, Bmat, d, 1) then
    WriteLn('    WARNING: ridge normal-equations matrix was singular.');

  NN := TNNet.Create();
  NN.AddLayer([
    TNNetInput.Create(cN),
    TNNetFullConnectLinear.Create(1)
  ]);
  for j := 0 to cN - 1 do
    NN.GetLastLayer().Neurons[0].Weights.FData[j] := Bmat[j];
  NN.GetLastLayer().Neurons[0].BiasWeight := Bmat[cN]; // intercept

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
// (predict x_{t+1} = x_t) over the test window.
procedure TeacherForcedTest(NN: TNNet;
  out ModelNRMSE, BaselineNRMSE: TNeuralFloat);
var
  t, idx, StartT, Len: integer;
  drive: array of TNeuralFloat;
  States: TNNetVolume;
  InV, OutV: TNNetVolume;
  Pred, Truth, Persist: array of TNeuralFloat;
begin
  StartT := cWarmup + cTrainLen; // continue past the training window
  Len := StartT + cTestLen;
  SetLength(drive, Len);
  for t := 0 to Len - 1 do drive[t] := Series(t);
  States := RunReservoir(drive, Len);

  SetLength(Pred, cTestLen);
  SetLength(Truth, cTestLen);
  SetLength(Persist, cTestLen);
  InV  := TNNetVolume.Create(cN, 1, 1);
  OutV := TNNetVolume.Create(1, 1, 1);
  try
    for idx := 0 to cTestLen - 1 do
    begin
      t := StartT + idx;
      StateToVolume(States, t, InV);   // teacher forcing: real input drove it
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
// own output back as the next input for cFreeRun steps. Because the layer runs
// a whole sequence per Compute (state re-derived from the prefix each call,
// exact under the echo-state property), each free-run step EXTENDS the driving
// sequence with our own prediction and re-runs, reading the final state.
function FreeRun(NN: TNNet; out Pred, Truth: array of TNeuralFloat): TNeuralFloat;
var
  t, idx, StartT, Len: integer;
  drive: array of TNeuralFloat;
  States: TNNetVolume;
  InV, OutV: TNNetVolume;
  x: TNeuralFloat;
begin
  StartT := cWarmup + cTrainLen;
  // Driving sequence: real series up to StartT (inclusive), then our own
  // predictions appended one at a time.
  SetLength(drive, StartT + 1 + cFreeRun);
  for t := 0 to StartT do drive[t] := Series(t);
  Len := StartT + 1;  // current driving length (last real input at StartT)

  InV  := TNNetVolume.Create(cN, 1, 1);
  OutV := TNNetVolume.Create(1, 1, 1);
  try
    for idx := 0 to cFreeRun - 1 do
    begin
      States := RunReservoir(drive, Len);     // drive with seq incl. own outputs
      StateToVolume(States, Len - 1, InV);    // state after the latest input
      NN.Compute(InV);
      NN.GetOutput(OutV);
      x := OutV.Raw[0];                        // feedback: prediction = next input
      Pred[idx]  := x;
      Truth[idx] := Series(StartT + 1 + idx);
      drive[Len] := x;                         // append for the next step
      Inc(Len);
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

const
  cLambdaSweep: array[0..3] of TNeuralFloat = (0.0, 1e-6, 1e-4, 1e-2);

var
  NN, RidgeNN: TNNet;
  TfModel, TfBaseline, FrGood, FrBad: TNeuralFloat;
  RidgeTf, RidgeFr, BestRidgeTf, BestRidgeFr: TNeuralFloat;
  RidgeTfDummy, BestLambda: TNeuralFloat;
  PredG, TruthG, PredB, TruthB: array of TNeuralFloat;
  PredR, TruthR: array of TNeuralFloat;
  AllOK: boolean;
  li: integer;

begin
  SetExceptionMask([exInvalidOp, exDenormalized, exZeroDivide,
                    exOverflow, exUnderflow, exPrecision]);
  RandSeed := 2026;
  AllOK := True;
  gReservoir := nil;
  gDrive := TNNetVolume.Create(cMaxLen, 1, 1);

  WriteLn('Echo State Network (Reservoir Computing, Jaeger 2001)');
  WriteLn('Reservoir N=', cN, '  leak=', cLeak:0:2, '  sparsity=',
    cSparsity:0:2, '  target rho=', cRhoGood:0:2);
  WriteLn('Recurrent core: reusable TNNetEchoStateReservoir layer.');
  WriteLn('Task: one-step prediction of sin(0.2 t) + 0.3 sin(0.31 t).');
  WriteLn(StringOfChar('=', 64));

  // ---- Working case: rho < 1 (echo-state property holds) -----------------
  WriteLn;
  WriteLn('[1] Building reservoir at rho=', cRhoGood:0:2, ' ...');
  BuildReservoir(cRhoGood);
  WriteLn('    measured raw W: spectral RADIUS rho = ', gResLayer.MeasuredRho:0:4);
  WriteLn('    -> W rescaled by the layer so its true spectral radius = ',
    cRhoGood:0:2);

  WriteLn('    training the linear readout (', cReadoutEpochs, ' epochs)...');
  NN := TrainReadout();

  TeacherForcedTest(NN, TfModel, TfBaseline);
  WriteLn(Format('    teacher-forced one-step NRMSE = %.4f', [TfModel]));
  WriteLn(Format('    persistence baseline   NRMSE = %.4f', [TfBaseline]));

  SetLength(PredG, cFreeRun);
  SetLength(TruthG, cFreeRun);
  FrGood := FreeRun(NN, PredG, TruthG);
  WriteLn(Format('    free-run (autonomous)  NRMSE = %.4f', [FrGood]));
  WriteLn;
  AsciiPlot(PredG, TruthG);
  NN.Free;

  // ---- Closed-form RIDGE readout (one-shot) + lambda sweep ----------------
  WriteLn;
  WriteLn(StringOfChar('-', 64));
  WriteLn('[1b] Closed-form RIDGE readout  Wout = (S^T S + lambda I)^-1 S^T Y');
  WriteLn('     one-shot solve (no LR, no epochs); lambda regularisation sweep:');
  WriteLn('       lambda     teacher-NRMSE   free-run-NRMSE');
  SetLength(PredR, cFreeRun);
  SetLength(TruthR, cFreeRun);
  BestRidgeTf := 1e30;
  BestRidgeFr := 1e30;
  BestLambda  := cLambdaSweep[0];
  for li := 0 to High(cLambdaSweep) do
  begin
    RidgeNN := RidgeReadout(cLambdaSweep[li]);
    TeacherForcedTest(RidgeNN, RidgeTf, RidgeTfDummy);
    RidgeFr := FreeRun(RidgeNN, PredR, TruthR);
    WriteLn(Format('       %-9.0e  %12.4f   %12.4f',
      [cLambdaSweep[li], RidgeTf, RidgeFr]));
    if RidgeFr < BestRidgeFr then
    begin
      BestRidgeTf := RidgeTf;
      BestRidgeFr := RidgeFr;
      BestLambda  := cLambdaSweep[li];
    end;
    RidgeNN.Free;
  end;
  WriteLn;
  WriteLn('     SGD-vs-ridge headline (same reservoir, same task):');
  WriteLn(Format('       SGD readout   (%d epochs, LR=%.3g): teacher %.4f  free-run %.4f',
    [cReadoutEpochs, cReadoutLR, TfModel, FrGood]));
  WriteLn(Format('       ridge readout (one-shot, lambda=%.0e):  teacher %.4f  free-run %.4f',
    [BestLambda, BestRidgeTf, BestRidgeFr]));

  // ---- Ablation: rho > 1 (echo-state property broken) --------------------
  WriteLn;
  WriteLn(StringOfChar('=', 64));
  WriteLn('[2] ABLATION - rebuilding reservoir at rho=', cRhoBad:0:2,
    ' (> 1, echo-state property BROKEN)');
  BuildReservoir(cRhoBad);
  WriteLn('    measured raw W: spectral RADIUS rho = ', gResLayer.MeasuredRho:0:4);
  NN := TrainReadout();
  SetLength(PredB, cFreeRun);
  SetLength(TruthB, cFreeRun);
  FrBad := FreeRun(NN, PredB, TruthB);
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

  // 3) ablation: rho>1 free-run must diverge far worse than rho<1.
  if IsNan(FrBad) or IsInfinite(FrBad) or (FrBad > 5.0 * FrGood + 1.0) then
    WriteLn(Format('  PASS  rho>1 free-run NRMSE %.4g explodes vs rho<1 %.4f',
      [FrBad, FrGood]))
  else
  begin
    WriteLn(Format('  FAIL  rho>1 free-run NRMSE %.4g did not diverge (rho<1 %.4f)',
      [FrBad, FrGood]));
    AllOK := False;
  end;

  gReservoir.Free;
  gDrive.Free;

  WriteLn(StringOfChar('=', 64));
  if AllOK then
    WriteLn('ALL CHECKS PASSED')
  else
  begin
    WriteLn('SOME CHECKS FAILED');
    Halt(1);
  end;
end.
