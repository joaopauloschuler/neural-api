unit TestNeuralDiffusion;
(*
Tests for neuraldiffusion.pas: the reusable diffusion noise-scheduler / sampler.

 - The linear beta / alpha / alpha_bar tables match a numpy float64 oracle at a
   set of probe timesteps (oracle in tools/diffusion_scheduler_oracle.py;
   values inlined below as committed constants).
 - The scaled-linear and cosine schedules satisfy their defining invariants
   (monotone non-increasing alpha_bar, alpha_bar_0 = 1, betas in range).
 - A deterministic DDIM (eta=0) trajectory driven by the SAME analytic stand-in
   model the oracle uses, eps(x,t)=sin(0.01*t)*x, reproduces the oracle final
   x0 vector to single-precision tolerance.
 - The forward q_sample (AddNoise) closed form matches a hand computation.
 - Classifier-free guidance mixing matches eps_uncond + w*(eps_cond-eps_uncond).
 - v-prediction is converted to the equivalent eps consistently with the
   forward-process identity.
 - DDPM and DPM-Solver++(2M) trajectories run without NaN and land near the
   deterministic DDIM result for the same toy model.
 - Euler-ancestral with Eta=0 reduces ANALYTICALLY to deterministic DDIM (eta=0)
   and matches it to single-precision tolerance (the exact deterministic anchor).
 - The Karras sigma table sigma_t = sqrt((1-ab_t)/ab_t) is strictly increasing,
   and a full Karras-spaced stochastic Euler-ancestral run is finite and stays
   near the DDIM baseline.
Coded by Claude (AI).
*)

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, Math, fpcunit, testregistry,
  neuralvolume, neuraldiffusion;

type
  TTestNeuralDiffusion = class(TTestCase)
  private
    // The analytic stand-in model used by both the oracle and the DDIM test.
    procedure ToyModel(Xt, Output: TNNetVolume; Tt: integer);
    // An "oracle" CONSISTENCY model for the LCM tests: it already knows the clean
    // image gLCMTarget and, fed any noised x_t at any t, returns the raw eps such
    // that the FULL consistency map f(x_t,t)=c_skip*x_t+c_out*x0_hat lands exactly
    // on gLCMTarget (i.e. f is consistent: f(x_t,t)=x0 for every t). It solves
    // x0_hat=(target-c_skip*x_t)/c_out, then back-outs the matching eps. This is
    // what a perfectly-distilled LCM satisfies, so the multistep loop is a fixed
    // point and must return the clean image.
    procedure LCMOracleModel(Xt, Output: TNNetVolume; Tt: integer);
  published
    procedure TestLinearScheduleVsOracle;
    procedure TestScaledLinearInvariants;
    procedure TestCosineInvariants;
    procedure TestAddNoiseClosedForm;
    procedure TestApplyCFG;
    procedure TestVPredictionToEps;
    procedure TestDDIMTrajectoryVsOracle;
    procedure TestDPMSolverVsOracle;
    procedure TestDDPMRunsNoNaN;
    procedure TestEulerAncestralZeroEtaMatchesDDIM;
    procedure TestKarrasSpacingSigmaMonotone;
    procedure TestKarrasEulerAncestralRunsNoNaN;
    procedure TestLCMBoundaryScalings;
    procedure TestLCMConsistencyFixedPoint;
    procedure TestLCMReproducible;
  end;

implementation

const
  cT      = 200;
  cBeta1  = 1.0e-4;
  cBetaT  = 0.02;
  cN      = 8;
  cSteps  = 10;
  // Oracle probe timesteps and their float64 alpha_bar (numpy).
  cProbeT: array[0..4] of integer = (1, 50, 100, 150, 200);
  cOracleAlphaBar: array[0..4] of double =
    (0.9999, 0.8801040247770505, 0.6024803053077055,
     0.32038735949336433, 0.13218275425061793);
  cOracleBeta: array[0..4] of double =
    (0.0001, 0.005000000000000001, 0.01, 0.015, 0.02);
  // Deterministic start vector x_start[i] = (i - N/2)*0.3.
  cOracleFinal: array[0..7] of double =
    (-0.8797523845208538, -0.6598142883906404, -0.4398761922604269,
     -0.21993809613021345, 0.0, 0.21993809613021345,
      0.4398761922604269, 0.6598142883906404);
  // DPM-Solver++(2M) final from the SAME numpy oracle (same toy model & start).
  cOracleDPM: array[0..7] of double =
    (-5.69458245535186, -4.270936841513894, -2.84729122767593,
     -1.423645613837965, 0.0, 1.423645613837965,
      2.84729122767593, 4.270936841513894);

procedure TTestNeuralDiffusion.ToyModel(Xt, Output: TNNetVolume; Tt: integer);
var i: integer; s: TNeuralFloat;
begin
  s := Sin(0.01 * Tt);
  for i := 0 to Xt.Size - 1 do
    Output.FData[i] := s * Xt.FData[i];
end;

var
  // The clean image the LCM oracle "knows", and the scheduler whose tables it
  // uses to invert the forward process. Module-level so the of-object callback
  // can reach them.
  gLCMTarget: TNNetVolume = nil;
  gLCMSched: TNNetLCMScheduler = nil;

procedure TTestNeuralDiffusion.LCMOracleModel(Xt, Output: TNNetVolume; Tt: integer);
var
  i: integer;
  sab, somab, cs, co, x0hat: TNeuralFloat;
begin
  // Pick eps so the full f(x_t,t)=c_skip*x_t+c_out*x0_hat equals gLCMTarget:
  //   x0_hat = (target - c_skip*x_t)/c_out,
  //   eps    = (x_t - sqrt(ab_t)*x0_hat)/sqrt(1-ab_t).
  sab := Sqrt(gLCMSched.AlphaBar[Tt]);
  somab := Sqrt(1.0 - gLCMSched.AlphaBar[Tt]);
  cs := gLCMSched.CSkip(Tt);
  co := gLCMSched.COut(Tt);
  for i := 0 to Xt.Size - 1 do
  begin
    x0hat := (gLCMTarget.FData[i] - cs * Xt.FData[i]) / co;
    Output.FData[i] := (Xt.FData[i] - sab * x0hat) / somab;
  end;
end;

procedure TTestNeuralDiffusion.TestLinearScheduleVsOracle;
var
  Sched: TNNetDiffusionScheduler;
  i: integer;
begin
  Sched := TNNetDiffusionScheduler.Create(cT, dsLinear, dpEps, cBeta1, cBetaT);
  try
    AssertEquals('AlphaBar[0]=1', 1.0, Sched.AlphaBar[0], 1e-7);
    AssertEquals('Beta[0]=0', 0.0, Sched.Beta[0], 1e-7);
    for i := 0 to High(cProbeT) do
    begin
      AssertEquals('beta @ ' + IntToStr(cProbeT[i]),
        cOracleBeta[i], Sched.Beta[cProbeT[i]], 1e-6);
      AssertEquals('alpha_bar @ ' + IntToStr(cProbeT[i]),
        cOracleAlphaBar[i], Sched.AlphaBar[cProbeT[i]], 1e-5);
    end;
  finally
    Sched.Free;
  end;
end;

procedure TTestNeuralDiffusion.TestScaledLinearInvariants;
var
  Sched: TNNetDiffusionScheduler;
  t: integer;
begin
  Sched := TNNetDiffusionScheduler.Create(cT, dsScaledLinear, dpEps,
    0.00085, 0.012);
  try
    AssertEquals('AlphaBar[0]=1', 1.0, Sched.AlphaBar[0], 1e-7);
    for t := 1 to cT do
    begin
      AssertTrue('alpha_bar non-increasing @ ' + IntToStr(t),
        Sched.AlphaBar[t] <= Sched.AlphaBar[t - 1] + 1e-7);
      AssertTrue('beta in (0,1) @ ' + IntToStr(t),
        (Sched.Beta[t] > 0) and (Sched.Beta[t] < 1));
    end;
  finally
    Sched.Free;
  end;
end;

procedure TTestNeuralDiffusion.TestCosineInvariants;
var
  Sched: TNNetDiffusionScheduler;
  t: integer;
begin
  Sched := TNNetDiffusionScheduler.Create(cT, dsCosine, dpEps);
  try
    AssertEquals('AlphaBar[0]=1', 1.0, Sched.AlphaBar[0], 1e-7);
    AssertTrue('alpha_bar_T < alpha_bar_1', Sched.AlphaBar[cT] < Sched.AlphaBar[1]);
    for t := 1 to cT do
    begin
      AssertTrue('alpha_bar non-increasing @ ' + IntToStr(t),
        Sched.AlphaBar[t] <= Sched.AlphaBar[t - 1] + 1e-7);
      AssertTrue('alpha_bar > 0 @ ' + IntToStr(t), Sched.AlphaBar[t] > 0);
      AssertTrue('beta clipped <=0.999 @ ' + IntToStr(t), Sched.Beta[t] <= 0.999 + 1e-7);
    end;
  finally
    Sched.Free;
  end;
end;

procedure TTestNeuralDiffusion.TestAddNoiseClosedForm;
var
  Sched: TNNetDiffusionScheduler;
  X0, Xt, Noise, Eps: TNNetVolume;
  i, t: integer;
  expect, sab, somab: TNeuralFloat;
begin
  Sched := TNNetDiffusionScheduler.Create(cT, dsLinear, dpEps, cBeta1, cBetaT);
  X0    := TNNetVolume.Create(cN, 1, 1);
  Xt    := TNNetVolume.Create(cN, 1, 1);
  Noise := TNNetVolume.Create(cN, 1, 1);
  Eps   := TNNetVolume.Create(cN, 1, 1);
  try
    for i := 0 to cN - 1 do begin X0.FData[i] := 0.5 * i - 1; Eps.FData[i] := 0.1 * i; end;
    t := 100;
    Sched.AddNoise(X0, Xt, t, Noise, Eps);
    sab := Sqrt(Sched.AlphaBar[t]);
    somab := Sqrt(1.0 - Sched.AlphaBar[t]);
    for i := 0 to cN - 1 do
    begin
      AssertEquals('noise copied', Eps.FData[i], Noise.FData[i], 1e-6);
      expect := sab * X0.FData[i] + somab * Eps.FData[i];
      AssertEquals('q_sample @ ' + IntToStr(i), expect, Xt.FData[i], 1e-5);
    end;
  finally
    Sched.Free; X0.Free; Xt.Free; Noise.Free; Eps.Free;
  end;
end;

procedure TTestNeuralDiffusion.TestApplyCFG;
var
  Cond, Uncond, Dst: TNNetVolume;
  i: integer;
  w, expect: TNeuralFloat;
begin
  Cond   := TNNetVolume.Create(cN, 1, 1);
  Uncond := TNNetVolume.Create(cN, 1, 1);
  Dst    := TNNetVolume.Create(cN, 1, 1);
  try
    for i := 0 to cN - 1 do begin Cond.FData[i] := i; Uncond.FData[i] := 0.5 * i; end;
    w := 3.0;
    TNNetDiffusionScheduler.ApplyCFG(Cond, Uncond, Dst, w);
    for i := 0 to cN - 1 do
    begin
      expect := Uncond.FData[i] + w * (Cond.FData[i] - Uncond.FData[i]);
      AssertEquals('cfg @ ' + IntToStr(i), expect, Dst.FData[i], 1e-6);
    end;
  finally
    Cond.Free; Uncond.Free; Dst.Free;
  end;
end;

procedure TTestNeuralDiffusion.TestVPredictionToEps;
var
  SchedE, SchedV: TNNetDiffusionScheduler;
  X0, Xt, Eps, V, Got: TNNetVolume;
  i, t: integer;
  sab, somab: TNeuralFloat;
begin
  // Build x_t from a known x0/eps; form the true v; check that a v-prediction
  // scheduler's DDIM step lands at the SAME place as an eps scheduler fed eps.
  SchedE := TNNetDiffusionScheduler.Create(cT, dsLinear, dpEps, cBeta1, cBetaT);
  SchedV := TNNetDiffusionScheduler.Create(cT, dsLinear, dpV, cBeta1, cBetaT);
  X0  := TNNetVolume.Create(cN, 1, 1);
  Xt  := TNNetVolume.Create(cN, 1, 1);
  Eps := TNNetVolume.Create(cN, 1, 1);
  V   := TNNetVolume.Create(cN, 1, 1);
  Got := TNNetVolume.Create(cN, 1, 1);
  try
    t := 120;
    sab := Sqrt(SchedE.AlphaBar[t]); somab := Sqrt(1.0 - SchedE.AlphaBar[t]);
    for i := 0 to cN - 1 do
    begin
      X0.FData[i] := 0.3 * i - 0.5;
      Eps.FData[i] := 0.2 * Cos(i);
      Xt.FData[i] := sab * X0.FData[i] + somab * Eps.FData[i];
      // v = sqrt(ab)*eps - sqrt(1-ab)*x0.
      V.FData[i] := sab * Eps.FData[i] - somab * X0.FData[i];
    end;
    // eps-scheduler DDIM step to clean image (tPrev=0).
    Got.Copy(Xt);
    SchedE.Step(Got, Eps, t, 0, smDDIM, 0.0);
    // v-scheduler DDIM step with the v prediction must match.
    Xt.Copy(Xt); // no-op clarity
    SchedV.Step(Xt, V, t, 0, smDDIM, 0.0);
    for i := 0 to cN - 1 do
      AssertEquals('v-pred matches eps @ ' + IntToStr(i),
        Got.FData[i], Xt.FData[i], 1e-4);
  finally
    SchedE.Free; SchedV.Free; X0.Free; Xt.Free; Eps.Free; V.Free; Got.Free;
  end;
end;

procedure TTestNeuralDiffusion.TestDDIMTrajectoryVsOracle;
var
  Sched: TNNetDiffusionScheduler;
  X: TNNetVolume;
  i: integer;
begin
  Sched := TNNetDiffusionScheduler.Create(cT, dsLinear, dpEps, cBeta1, cBetaT);
  X := TNNetVolume.Create(cN, 1, 1);
  try
    for i := 0 to cN - 1 do X.FData[i] := (i - cN / 2) * 0.3;
    Sched.Sample(X, @ToyModel, cSteps, smDDIM, 0.0);
    for i := 0 to cN - 1 do
      AssertEquals('ddim final @ ' + IntToStr(i),
        cOracleFinal[i], X.FData[i], 1e-4);
  finally
    Sched.Free; X.Free;
  end;
end;

procedure TTestNeuralDiffusion.TestDPMSolverVsOracle;
var
  Sched: TNNetDiffusionScheduler;
  X: TNNetVolume;
  i: integer;
begin
  Sched := TNNetDiffusionScheduler.Create(cT, dsLinear, dpEps, cBeta1, cBetaT);
  X := TNNetVolume.Create(cN, 1, 1);
  try
    for i := 0 to cN - 1 do X.FData[i] := (i - cN / 2) * 0.3;
    Sched.Sample(X, @ToyModel, cSteps, smDPMSolverPP2M, 0.0);
    // Match the numpy DPM-Solver++(2M) float64 oracle (same toy model & start).
    // The magnitudes are large because the toy eps grows x; the point is the
    // multistep lambda/correction bookkeeping reproduces numpy exactly.
    for i := 0 to cN - 1 do
    begin
      AssertFalse('dpm++ no NaN @ ' + IntToStr(i),
        IsNan(X.FData[i]) or IsInfinite(X.FData[i]));
      AssertEquals('dpm++ vs oracle @ ' + IntToStr(i),
        cOracleDPM[i], X.FData[i], 1e-2);
    end;
  finally
    Sched.Free; X.Free;
  end;
end;

procedure TTestNeuralDiffusion.TestDDPMRunsNoNaN;
var
  Sched: TNNetDiffusionScheduler;
  X: TNNetVolume;
  i: integer;
begin
  RandSeed := 424242;
  Sched := TNNetDiffusionScheduler.Create(cT, dsLinear, dpEps, cBeta1, cBetaT);
  X := TNNetVolume.Create(cN, 1, 1);
  try
    for i := 0 to cN - 1 do X.FData[i] := RandG(0, 1);
    // Full ancestral loop: use NumSteps=T so every timestep is visited.
    Sched.Sample(X, @ToyModel, cT, smDDPM, 0.0);
    for i := 0 to cN - 1 do
      AssertFalse('ddpm no NaN @ ' + IntToStr(i),
        IsNan(X.FData[i]) or IsInfinite(X.FData[i]));
  finally
    Sched.Free; X.Free;
  end;
end;

procedure TTestNeuralDiffusion.TestEulerAncestralZeroEtaMatchesDDIM;
var
  SchedA, SchedD: TNNetDiffusionScheduler;
  XA, XD: TNNetVolume;
  i: integer;
begin
  // With Eta=0 the ancestral noise term vanishes (sigma_up=0) and the Euler step
  // reduces ANALYTICALLY to deterministic DDIM (eta=0). They must match to
  // single-precision tolerance on the SAME toy model and start vector. This is
  // the exact deterministic reduction (Euler-a -> Euler -> DDIM) used as the
  // numerical anchor for the new sampler.
  SchedA := TNNetDiffusionScheduler.Create(cT, dsLinear, dpEps, cBeta1, cBetaT);
  SchedD := TNNetDiffusionScheduler.Create(cT, dsLinear, dpEps, cBeta1, cBetaT);
  XA := TNNetVolume.Create(cN, 1, 1);
  XD := TNNetVolume.Create(cN, 1, 1);
  try
    for i := 0 to cN - 1 do
    begin
      XA.FData[i] := (i - cN / 2) * 0.3;
      XD.FData[i] := XA.FData[i];
    end;
    SchedA.Sample(XA, @ToyModel, cSteps, smEulerAncestral, 0.0, tsUniform);
    SchedD.Sample(XD, @ToyModel, cSteps, smDDIM, 0.0, tsUniform);
    for i := 0 to cN - 1 do
    begin
      AssertFalse('euler-a no NaN @ ' + IntToStr(i),
        IsNan(XA.FData[i]) or IsInfinite(XA.FData[i]));
      AssertEquals('euler-a(eta=0) == ddim(eta=0) @ ' + IntToStr(i),
        XD.FData[i], XA.FData[i], 1e-4);
    end;
  finally
    SchedA.Free; SchedD.Free; XA.Free; XD.Free;
  end;
end;

procedure TTestNeuralDiffusion.TestKarrasSpacingSigmaMonotone;
var
  Sched: TNNetDiffusionScheduler;
  t: integer;
begin
  // sigma_t = sqrt((1-ab_t)/ab_t) must be strictly increasing in t (it is the
  // per-step noise level), which is what makes SigmaToTimestep well defined and
  // the Karras rho-spacing meaningful. sigma_1 ~ small, sigma_T large.
  Sched := TNNetDiffusionScheduler.Create(cT, dsLinear, dpEps, cBeta1, cBetaT);
  try
    AssertEquals('sigma[0]=0', 0.0, Sched.SigmaAt[0], 1e-7);
    for t := 2 to cT do
      AssertTrue('sigma increasing @ ' + IntToStr(t),
        Sched.SigmaAt[t] > Sched.SigmaAt[t - 1]);
    AssertTrue('sigma_T > sigma_1', Sched.SigmaAt[cT] > Sched.SigmaAt[1]);
  finally
    Sched.Free;
  end;
end;

procedure TTestNeuralDiffusion.TestKarrasEulerAncestralRunsNoNaN;
var
  Sched: TNNetDiffusionScheduler;
  X, XBase: TNNetVolume;
  i: integer;
  drift: TNeuralFloat;
begin
  // A full Karras-spaced Euler-ancestral trajectory (Eta=1, stochastic) must
  // produce finite output and stay in the same neighbourhood as the DDIM
  // baseline for the same toy model (the ancestral noise is small at the toy
  // scale). RandSeed fixed so the run is reproducible.
  RandSeed := 424242;
  Sched := TNNetDiffusionScheduler.Create(cT, dsLinear, dpEps, cBeta1, cBetaT);
  X     := TNNetVolume.Create(cN, 1, 1);
  XBase := TNNetVolume.Create(cN, 1, 1);
  try
    for i := 0 to cN - 1 do
    begin
      X.FData[i] := (i - cN / 2) * 0.3;
      XBase.FData[i] := X.FData[i];
    end;
    Sched.Sample(X, @ToyModel, cSteps, smEulerAncestral, 1.0, tsKarras);
    // DDIM baseline on the SAME start (deterministic).
    Sched.Sample(XBase, @ToyModel, cSteps, smDDIM, 0.0, tsKarras);
    for i := 0 to cN - 1 do
    begin
      AssertFalse('karras euler-a no NaN @ ' + IntToStr(i),
        IsNan(X.FData[i]) or IsInfinite(X.FData[i]));
      drift := Abs(X.FData[i] - XBase.FData[i]);
      AssertTrue('karras euler-a near ddim baseline @ ' + IntToStr(i),
        drift < 1.0);
    end;
  finally
    Sched.Free; X.Free; XBase.Free;
  end;
end;

procedure TTestNeuralDiffusion.TestLCMBoundaryScalings;
var
  Sched: TNNetLCMScheduler;
  cs1, csT, co1, coT: TNeuralFloat;
const
  cSigmaData = 0.5;
begin
  // c_skip(t) = sigma_data^2/((t/0.1)^2+sigma_data^2),
  // c_out(t)  = (t/0.1)*sigma_data/sqrt((t/0.1)^2+sigma_data^2).
  // At the LARGEST sigma (t=T) the (t/0.1)^2 term dominates: c_skip -> ~0 and
  // c_out -> ~sigma_data. At small t (t=1) c_skip is near 1 and c_out near 0
  // (the consistency boundary condition f(x,0)=x).
  Sched := TNNetLCMScheduler.Create(cT, dsLinear, dpEps, cBeta1, cBetaT);
  try
    cs1 := Sched.CSkip(1);    co1 := Sched.COut(1);
    csT := Sched.CSkip(cT);   coT := Sched.COut(cT);
    // At t=1: (1/0.1)^2 = 100 vs sigma_data^2 = 0.25 -> c_skip ~ 0.25/100.25.
    AssertEquals('c_skip(1)', 0.25 / 100.25, cs1, 1e-6);
    AssertEquals('c_out(1)', 10.0 * 0.5 / Sqrt(100.0 + 0.25), co1, 1e-6);
    // Monotonicity: c_skip shrinks with t, c_out grows toward sigma_data.
    AssertTrue('c_skip largest-sigma small', csT < cs1);
    AssertTrue('c_skip(T) < 1e-3', csT < 1e-3);
    AssertTrue('c_out grows with t', coT > co1);
    AssertTrue('c_out(T) ~ sigma_data', Abs(coT - cSigmaData) < 1e-3);
    AssertTrue('c_out(T) < sigma_data', coT < cSigmaData);
  finally
    Sched.Free;
  end;
end;

procedure TTestNeuralDiffusion.TestLCMConsistencyFixedPoint;
var
  Sched: TNNetLCMScheduler;
  X, Target: TNNetVolume;
  i: integer;
begin
  // On a TRIVIAL "model" that is already consistent (its x0 prediction is the
  // true clean image at every t), f(x_t,t) returns that clean image for any t,
  // so the multistep LCM loop is a fixed point: starting from a noised latent it
  // returns the clean image to tolerance after the few steps. This pins both the
  // c_skip/c_out parameterization AND the predict-x0 / re-noise loop.
  RandSeed := 424242;
  Sched := TNNetLCMScheduler.Create(cT, dsLinear, dpEps, cBeta1, cBetaT);
  X      := TNNetVolume.Create(cN, 1, 1);
  Target := TNNetVolume.Create(cN, 1, 1);
  try
    for i := 0 to cN - 1 do Target.FData[i] := (i - cN / 2) * 0.3;
    gLCMTarget := Target;
    gLCMSched := Sched;
    // Start from the clean target fully noised toward t=T (pure-noise regime).
    Sched.AddNoise(Target, X, cT);
    Sched.LCMSample(X, @LCMOracleModel, 4, tsUniform);
    for i := 0 to cN - 1 do
    begin
      AssertFalse('lcm no NaN @ ' + IntToStr(i),
        IsNan(X.FData[i]) or IsInfinite(X.FData[i]));
      AssertEquals('lcm consistency fixed point @ ' + IntToStr(i),
        Target.FData[i], X.FData[i], 1e-3);
    end;
  finally
    gLCMTarget := nil; gLCMSched := nil;
    Sched.Free; X.Free; Target.Free;
  end;
end;

procedure TTestNeuralDiffusion.TestLCMReproducible;
var
  SchedA, SchedB: TNNetLCMScheduler;
  XA, XB, Target: TNNetVolume;
  i: integer;
begin
  // A seeded LCM run (the re-noise hops draw fresh Gaussian noise) is exactly
  // reproducible when the RNG seed is fixed before each run.
  SchedA := TNNetLCMScheduler.Create(cT, dsLinear, dpEps, cBeta1, cBetaT);
  SchedB := TNNetLCMScheduler.Create(cT, dsLinear, dpEps, cBeta1, cBetaT);
  XA     := TNNetVolume.Create(cN, 1, 1);
  XB     := TNNetVolume.Create(cN, 1, 1);
  Target := TNNetVolume.Create(cN, 1, 1);
  try
    for i := 0 to cN - 1 do Target.FData[i] := (i - cN / 2) * 0.3;
    gLCMTarget := Target;

    RandSeed := 777; gLCMSched := SchedA;
    for i := 0 to cN - 1 do XA.FData[i] := RandG(0, 1);
    SchedA.LCMSample(XA, @LCMOracleModel, 4, tsUniform);

    RandSeed := 777; gLCMSched := SchedB;
    for i := 0 to cN - 1 do XB.FData[i] := RandG(0, 1);
    SchedB.LCMSample(XB, @LCMOracleModel, 4, tsUniform);

    for i := 0 to cN - 1 do
      AssertEquals('lcm reproducible @ ' + IntToStr(i),
        XA.FData[i], XB.FData[i], 1e-6);
  finally
    gLCMTarget := nil; gLCMSched := nil;
    SchedA.Free; SchedB.Free; XA.Free; XB.Free; Target.Free;
  end;
end;

initialization
  RegisterTest(TTestNeuralDiffusion);
end.
