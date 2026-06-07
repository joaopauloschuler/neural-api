// HamiltonianPendulum example
//
// Learns the dynamics of an ideal (undamped) pendulum from NOISY phase-space
// samples, then rolls the learned model out autoregressively for a LONG horizon
// and measures how well each model conserves energy.
//
//   True system:  H(q,p) = 0.5*p^2 + (1 - cos q)   (unit mass/length/gravity)
//                 dq/dt = +dH/dp =  p
//                 dp/dt = -dH/dq = -sin q
//                 conserved energy E = 0.5*p^2 + (1 - cos q)
//
// Two arms are trained on the SAME one-step (q,p) -> (q,p)_next regression with
// the SAME additive Gaussian observation noise:
//   * HNN arm: a TNNetHamiltonianCell. It does NOT regress the next state
//     directly; it parameterizes a scalar learned Hamiltonian H_theta(q,p) and
//     takes a SYMPLECTIC step (dq=+dH/dp, dp=-dH/dq). Energy conservation is
//     baked into the geometry of the update.
//   * Baseline arm: an unconstrained per-step MLP field of identical width that
//     regresses the next state directly (a plain learned dynamics map, the kind
//     a NeuralODE residual field would use). Nothing constrains it to conserve
//     anything.
//
// Headline: over a long free-swing rollout the HNN stays near its initial energy
// level set, while the unconstrained map's energy DRIFTS (typically spiralling
// in or blowing up). Pure CPU, finishes in well under a minute on 2 cores.
// Coded by Claude (AI).
program HamiltonianPendulum;

{$mode objfpc}{$H+}

uses
  {$IFDEF UNIX}cthreads,{$ENDIF}
  Classes, SysUtils, Math,
  neuralnetwork, neuralvolume;

const
  cDt      = 0.1;     // integration / sampling step
  cHidden  = 64;      // inner-MLP hidden width for the scalar Hamiltonian
  cEpochs  = 100000;  // supervised one-step SGD steps
  cRollout = 800;     // long-horizon autoregressive rollout length
  cNoise   = 0.02;    // observation noise std on the (next) phase samples

var
  RngState: cardinal;

// Tiny deterministic LCG so the demo is reproducible across runs/platforms.
function NextRand(): TNeuralFloat;
begin
  RngState := (RngState * 1103515245 + 12345) and $7FFFFFFF;
  Result := RngState / 2147483647.0;  // in [0,1)
end;

function NextGauss(AStd: TNeuralFloat): TNeuralFloat;
begin
  Result := Sqrt(-2 * Ln(NextRand() + 1e-12)) * Cos(2 * Pi * NextRand()) * AStd;
end;

// One exact symplectic (leapfrog) step of the TRUE pendulum, used to make the
// ground-truth training targets and the reference rollout.
procedure TruePendulumStep(var q, p: TNeuralFloat; dt: TNeuralFloat);
var phalf: TNeuralFloat;
begin
  phalf := p - 0.5 * dt * Sin(q);
  q := q + dt * phalf;
  p := phalf - 0.5 * dt * Sin(q);
end;

function Energy(q, p: TNeuralFloat): TNeuralFloat;
begin
  Result := 0.5 * p * p + (1 - Cos(q));
end;

// Build a single training pair: a random starting phase (q,p) from a free-swing
// region, with the NOISY next state as target. Volumes are (1,1,2).
procedure FillSample(StartV, NextV: TNNetVolume);
var
  q, p, q1, p1: TNeuralFloat;
begin
  q := (NextRand() * 2 - 1) * 2.0;   // q in [-2, 2] rad
  p := (NextRand() * 2 - 1) * 1.5;   // p in [-1.5, 1.5]
  q1 := q; p1 := p;
  TruePendulumStep(q1, p1, cDt);
  StartV.Raw[0] := q;
  StartV.Raw[1] := p;
  NextV.Raw[0] := q1 + NextGauss(cNoise);
  NextV.Raw[1] := p1 + NextGauss(cNoise);
end;

// Roll a single model out autoregressively from (q0,p0) for ASteps and report
// the absolute energy drift |E_end - E_0| and the max |E_t - E_0| along the way.
procedure RolloutEnergy(NN: TNNet; q0, p0: TNeuralFloat; ASteps: integer;
  out EndDrift, MaxDrift: TNeuralFloat);
var
  st, nx: TNNetVolume;
  q, p, e0, e, drift: TNeuralFloat;
  t: integer;
begin
  st := TNNetVolume.Create(1, 1, 2);
  nx := TNNetVolume.Create(1, 1, 2);
  try
    q := q0; p := p0;
    e0 := Energy(q, p);
    MaxDrift := 0;
    for t := 0 to ASteps - 1 do
    begin
      st.Raw[0] := q; st.Raw[1] := p;
      NN.Compute(st);
      nx.Copy(NN.GetLastLayer.Output);
      q := nx.Raw[0]; p := nx.Raw[1];
      e := Energy(q, p);
      drift := Abs(e - e0);
      if drift > MaxDrift then MaxDrift := drift;
    end;
    EndDrift := Abs(Energy(q, p) - e0);
  finally
    nx.Free;
    st.Free;
  end;
end;

procedure TrueRolloutEnergy(q0, p0: TNeuralFloat; ASteps: integer;
  out EndDrift, MaxDrift: TNeuralFloat);
var q, p, e0, drift: TNeuralFloat; t: integer;
begin
  q := q0; p := p0; e0 := Energy(q, p); MaxDrift := 0;
  for t := 0 to ASteps - 1 do
  begin
    TruePendulumStep(q, p, cDt);
    drift := Abs(Energy(q, p) - e0);
    if drift > MaxDrift then MaxDrift := drift;
  end;
  EndDrift := Abs(Energy(q, p) - e0);
end;

// One-step training MSE estimate over a fresh sample of pairs (diagnostic).
function OneStepMSE(NN: TNNet; ASamples: integer): TNeuralFloat;
var
  StartV, NextV: TNNetVolume;
  i: integer;
  d0, d1: TNeuralFloat;
begin
  StartV := TNNetVolume.Create(1, 1, 2);
  NextV  := TNNetVolume.Create(1, 1, 2);
  Result := 0;
  try
    for i := 0 to ASamples - 1 do
    begin
      FillSample(StartV, NextV);
      NN.Compute(StartV);
      d0 := NN.GetLastLayer.Output.Raw[0] - NextV.Raw[0];
      d1 := NN.GetLastLayer.Output.Raw[1] - NextV.Raw[1];
      Result := Result + 0.5 * (d0 * d0 + d1 * d1);
    end;
    Result := Result / ASamples;
  finally
    NextV.Free;
    StartV.Free;
  end;
end;

procedure TrainArm(NN: TNNet; ALR: TNeuralFloat);
var
  StartV, NextV: TNNetVolume;
  step: integer;
begin
  StartV := TNNetVolume.Create(1, 1, 2);
  NextV  := TNNetVolume.Create(1, 1, 2);
  try
    // Per-sample SGD with momentum (the per-sample path averages cleanly; the
    // batch-update path SUMS deltas without dividing -- see the library note).
    NN.SetLearningRate(ALR, 0.9);
    NN.SetBatchUpdate(false);
    for step := 0 to cEpochs - 1 do
    begin
      FillSample(StartV, NextV);
      NN.Compute(StartV);
      NN.Backpropagate(NextV);
    end;
  finally
    NextV.Free;
    StartV.Free;
  end;
end;

var
  HNN, BNN: TNNet;
  LH: TNNetHamiltonianCell;
  BInput, BField: TNNetLayer;
  hEnd, hMax, bEnd, bMax, tEnd, tMax: TNeuralFloat;
  q0, p0: TNeuralFloat;
begin
  // Mask FP exceptions during training (tanh/Exp can momentarily underflow).
  SetExceptionMask([exInvalidOp, exDenormalized, exZeroDivide,
                    exOverflow, exUnderflow, exPrecision]);
  // The manual Compute/Backpropagate loop here is single-threaded already.
  RngState := 424242;

  // --- Arm 1: TNNetHamiltonianCell (symplectic, energy-conserving by design) --
  // Input is a single (1,1,2) phase vector; the cell integrates it one step
  // forward. 2 phase dims => D=1 coordinate (q,p).
  HNN := TNNet.Create();
  HNN.AddLayer(TNNetInput.Create(1, 1, 2, 1));
  // 2 internal symplectic sub-steps of dt/2 advance the same total dt per call,
  // with a smaller per-step dt for better integration stability over long rollouts.
  LH := TNNetHamiltonianCell.Create({Steps=}2, {dt=}cDt / 2, {Hidden=}cHidden);
  HNN.AddLayer(LH);

  // --- Arm 2: unconstrained NeuralODE-style residual MLP field ----------------
  // z_new = z + f_theta(z), where f_theta is an MLP of identical hidden width and
  // the same smooth (tanh) activation as the HNN's inner H-MLP. This is exactly
  // the kind of free vector field a residual NeuralODE block uses -- it has NO
  // symplectic / energy structure, nothing forces f_theta to be the gradient of
  // a scalar, so the rollout is free to gain or lose energy.
  BNN := TNNet.Create();
  BInput := BNN.AddLayer(TNNetInput.Create(1, 1, 2, 1));
  BNN.AddLayer(TNNetPointwiseConvLinear.Create(cHidden));
  BNN.AddLayer(TNNetHyperbolicTangent.Create());
  BField := BNN.AddLayer(TNNetPointwiseConvLinear.Create(2));
  BNN.AddLayer(TNNetSum.Create([BInput, BField]));   // residual: z + f(z)

  WriteLn('=== HamiltonianPendulum: energy conservation, HNN vs free MLP ===');
  WriteLn('dt=', cDt:0:3, '  steps=', cEpochs,
    '  noise std=', cNoise:0:3, '  rollout=', cRollout, ' steps');
  WriteLn('HNN params (inner H-MLP) = ', HNN.CountWeights(),
    '   baseline MLP params = ', BNN.CountWeights());
  WriteLn;

  WriteLn('training (one-step regression on noisy pendulum samples)...');
  // The HNN's loss flows through the dt-scaled symplectic update, so its raw
  // gradients w.r.t. the inner H-MLP are ~dt smaller than the direct-regression
  // baseline's -- it gets a correspondingly larger learning rate.
  TrainArm(HNN, 0.05);
  TrainArm(BNN, 0.005);
  WriteLn('  one-step MSE  HNN = ', OneStepMSE(HNN, 2000):0:6,
    '   free MLP = ', OneStepMSE(BNN, 2000):0:6,
    '   (noise floor ~ ', (cNoise * cNoise):0:6, ')');

  // Long autoregressive free-swing rollout from a fixed energetic start.
  q0 := 1.6; p0 := 0.0;
  WriteLn;
  WriteLn('long-horizon rollout from (q0,p0)=(', q0:0:2, ',', p0:0:2,
    '),  E0 = ', Energy(q0, p0):0:5);
  WriteLn;

  RolloutEnergy(HNN, q0, p0, cRollout, hEnd, hMax);
  RolloutEnergy(BNN, q0, p0, cRollout, bEnd, bMax);
  TrueRolloutEnergy(q0, p0, cRollout, tEnd, tMax);

  WriteLn('energy drift over ', cRollout, ' autoregressive steps (lower = better):');
  WriteLn('  true leapfrog : end |dE|=', tEnd:0:5, '   max |dE|=', tMax:0:5,
    '   (reference)');
  WriteLn('  HNN (symplectic): end |dE|=', hEnd:0:5, '   max |dE|=', hMax:0:5);
  WriteLn('  free MLP field  : end |dE|=', bEnd:0:5, '   max |dE|=', bMax:0:5);
  WriteLn;

  if (hMax < bMax) then
    WriteLn('OK: the Hamiltonian cell conserves energy far better than the ',
      'unconstrained MLP field (', (bMax / Max(hMax, 1e-9)):0:1,
      'x smaller max drift).')
  else
    WriteLn('WARNING: HNN did not beat the free MLP on energy drift.');

  BNN.Free;
  HNN.Free;
end.
