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
 - DDPM runs without NaN; DPM-Solver++(2M) matches the diffusers
   DPMSolverMultistepScheduler form (tools/diffusion_scheduler_oracle.py).
 - PNDM (PLMS, eps and v), deterministic Euler and Euler-ancestral (replayed
   noise) match diffusers float64 finals (tools/diffusers_sampler_toy_oracle.py),
   and BuildSamplerSchedule reproduces their timesteps.
 - Euler-ancestral with Eta=0 reduces ANALYTICALLY to deterministic DDIM (eta=0)
   and matches it to single-precision tolerance (the exact deterministic anchor).
 - The Karras sigma table sigma_t = sqrt((1-ab_t)/ab_t) is strictly increasing,
   and a full Karras-spaced stochastic Euler-ancestral run is finite and stays
   near the DDIM baseline.
 - tsTrailing reproduces diffusers' "trailing" timesteps; the zero-terminal-SNR
   rescale gives ab_T = 0 with ab_1 unchanged, and every sampler stays finite
   there (DDIM and Euler with v-prediction exactly: x0 = -v at ab_T = 0).
 - pFinalAlphaBarIsOne = false (diffusers set_alpha_to_one=False) reproduces a
   4-step Marigold-config DDIM trajectory computed with diffusers' DDIMScheduler.
 - TNNetFlowMatchEulerScheduler matches the Qwen-Image-2.1 diffusers oracle
   (tests/fixtures/qwenimage21_scheduler_oracle.json, from
   tools/make_pico_qwenimage21_fixture.py): mu, sigmas, timesteps, two steps.
 - Its fsDPMSolverPP2M sampler equals Euler when every step is first order and
   matches diffusers DPMSolverMultistepScheduler(use_flow_sigmas) trajectories.
Coded by Claude (AI).
*)

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, Math, fpcunit, testregistry,
  fpjson, jsonparser, neuralvolume, neuraldiffusion;

type
  TTestNeuralDiffusion = class(TTestCase)
  private
    // Timesteps recorded by RecordingModel, in call order.
    FRecT: array[0..63] of integer;
    FRecN: integer;
    // The analytic stand-in model used by both the oracle and the DDIM test.
    procedure ToyModel(Xt, Output: TNNetVolume; Tt: integer);
    // A do-nothing denoiser that records the timesteps the driver visits, so a
    // test can see the schedule the (private) spacing builder produced.
    procedure RecordingModel(Xt, Output: TNNetVolume; Tt: integer);
    // An "oracle" CONSISTENCY model for the LCM tests: it already knows the clean
    // image gLCMTarget and, fed any noised x_t at any t, returns the raw eps such
    // that the FULL consistency map f(x_t,t)=c_skip*x_t+c_out*x0_hat lands exactly
    // on gLCMTarget (i.e. f is consistent: f(x_t,t)=x0 for every t). It solves
    // x0_hat=(target-c_skip*x_t)/c_out, then back-outs the matching eps. This is
    // what a perfectly-distilled LCM satisfies, so the multistep loop is a fixed
    // point and must return the clean image.
    procedure LCMOracleModel(Xt, Output: TNNetVolume; Tt: integer);
    // CFG guidance-rescale assertions; driven by TestApplyCFG (not a test itself).
    procedure TestApplyCFGRescale(Cond, Uncond: TNNetVolume; w: TNeuralFloat);
  published
    procedure TestLinearScheduleVsOracle;
    procedure TestScaledLinearInvariants;
    procedure TestCosineInvariants;
    procedure TestAddNoiseClosedForm;
    // AlphaBarRootsAt (integer: the tables; fractional: 1/(1+sigma^2)) and
    // SplitPrediction recovering x0 and eps from eps and v predictions.
    procedure TestSplitPredictionAt;
    procedure TestApplyCFG;
    procedure TestAPGMatchesCFG;
    procedure TestAPGProjectionAndClamp;
    procedure TestAPGMomentum;
    procedure TestAPGAliasing;
    procedure TestRawPredFromX0RoundTrip;
    procedure TestAPGCreateFromSpec;
    procedure TestPerturbedGuidanceBranches;
    procedure TestPerturbedGuidanceWindows;
    procedure TestPerturbedGuidanceRescaleAndAliasing;
    procedure TestVPredictionToEps;
    procedure TestDDIMTrajectoryVsOracle;
    procedure TestDPMSolverVsOracle;
    procedure TestSamplerSchedulesVsDiffusers;
    procedure TestDiffusersSchedulerSamplerMapping;
    procedure TestPNDMVsDiffusers;
    procedure TestEulerVsDiffusers;
    procedure TestEulerAncestralVsDiffusers;
    procedure TestUniPCVsOracle;
    procedure TestHeunVsOracle;
    procedure TestHeunZeroSigmaRunsFinite;
    procedure TestMinSNRWeight;
    procedure TestDDPMRunsNoNaN;
    procedure TestEulerAncestralZeroEtaMatchesDDIM;
    procedure TestKarrasSpacingSigmaMonotone;
    procedure TestKarrasSnapMatchesNearestSigmaScan;
    procedure TestSNRWeightClosedForm;
    procedure TestKarrasEulerAncestralRunsNoNaN;
    procedure TestLCMBoundaryScalings;
    procedure TestLCMConsistencyFixedPoint;
    procedure TestLCMReproducible;
    procedure TestLCMOriginScheduleVsDiffusers;
    procedure TestLCMFromDiffusersConfig;
    procedure TestFlowMatchCalculateShiftVsOracle;
    procedure TestFlowMatchSigmasVsOracle;
    procedure TestFlowMatchStepVsOracle;
    procedure TestFlowMatchStaticAndLinearShift;
    procedure TestFlowMatchFromDiffusersConfig;
    procedure TestFlowMatchImg2ImgStartStepVsOracle;
    procedure TestFlowMatchScaleNoiseVsOracle;
    procedure TestFlowDPMSolverFirstOrderMatchesEuler;
    procedure TestFlowDPMSolverTrajectoryVsDiffusers;
    procedure TestFlowUniPCTrajectoryVsDiffusers;
    procedure TestTrailingSpacingMatchesDiffusers;
    procedure TestZeroTerminalSNRRescale;
    procedure TestZeroTerminalSNRDDIMVEndpoint;
    procedure TestZeroTerminalSNRTrailingSamplersFinite;
    procedure TestDDIMFinalAlphaBarVsDiffusers;
    procedure TestZeroTerminalSNREulerVEndpoint;
    procedure TestZeroTerminalSNRDDPMVEndpoint;
    procedure TestDDPMFirstStepPosteriorMean;
    procedure TestSchedulerFromDiffusersConfig;
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
  // DPM-Solver++(2M) final (diffusers DPMSolverMultistepScheduler, linspace
  // spacing) from the numpy oracle, same toy model & start; diffusers itself
  // gives -0.96552890 at [0] (its sigmas are float32).
  cOracleDPM: array[0..7] of double =
    (-0.9655285887419517, -0.7241464415564635, -0.48276429437097584,
     -0.24138214718548792, 0.0, 0.24138214718548792,
      0.48276429437097584, 0.7241464415564635);
  // diffusers float64 finals of tools/diffusers_sampler_toy_oracle.py (same
  // toy model & start; the model sees scale_model_input).
  // PNDMScheduler(skip_prk_steps, linspace, set_alpha_to_one=False), 10 steps.
  cOraclePNDM: array[0..7] of double =
    (-0.9828584041251481, -0.7371438030938611, -0.49142920206257407,
     -0.24571460103128703, 0.0, 0.24571460103128703, 0.49142920206257407,
      0.7371438030938611);
  // The same with prediction_type v_prediction.
  cOraclePNDMV: array[0..7] of double =
    (-0.5442082901924374, -0.40815621764432813, -0.2721041450962187,
     -0.13605207254810936, 0.0, 0.13605207254810936, 0.2721041450962187,
      0.40815621764432813);
  // EulerDiscreteScheduler, trailing (init_noise_sigma = sigma_max), 10 steps.
  cOracleEuler: array[0..7] of double =
    (-0.825409763608135, -0.6190572941448793, -0.4127048818040675,
     -0.20635244090203375, 0.0, 0.20635244090203375, 0.4127048818040675,
      0.6190572941448793);
  // EulerAncestralDiscreteScheduler, leading, 4 steps, replaying its noise.
  cEulerANoise: array[0..23] of double =
    (-0.1467950974849952, 0.7861412987235358, 0.9468216090039564,
     -1.1143440212610307, 1.6907901257942333, -0.8948278791726386,
     -0.3556250270062136, 1.2323857451651945, 0.13817266368739245,
     -1.6821984860536687, 0.31767831539796804, 0.13280696453151092,
      0.1373240964733236, 0.2405461286050214, 1.3954507962006726,
      1.3470226203485132, 2.4382081623118323, 0.202758282866977,
      2.4505412457460647, 2.0256017922932585, 1.77915501934496,
     -0.9179307346473358, -0.45781890986397766, -0.724473294750947);
  cOracleEulerA: array[0..7] of double =
    (-0.4240828388146988, -0.5385330522760837, 0.20056166990819768,
     -0.41460828703677105, 0.6050061463886979, -0.1233071142778826,
      0.486501166264323, 1.0959327831214518);
  // UniPC (order-2 bh2, predict_x0) final from the float64 numpy oracle
  // tools/unipc_scheduler_oracle.py (same toy model, start & uniform schedule).
  cOracleUniPC: array[0..7] of double =
    (-1.0913574282060237, -0.8185180711545179, -0.5456787141030118,
     -0.2728393570515059, 0.0, 0.2728393570515059,
      0.5456787141030118, 0.8185180711545179);
  // Heun 2nd-order (smHeun) with tsKarras spacing, from the float64 numpy oracle
  // tools/heun_scheduler_oracle.py (same toy model & start). Two denoiser evals
  // per step; corrector skipped on the final hop (sigma=0).
  cOracleHeun: array[0..7] of double =
    (-0.9874468254279581, -0.7405851190709684, -0.49372341271397907,
     -0.24686170635698954, 0.0, 0.24686170635698954,
      0.49372341271397907, 0.7405851190709684);
  // Min-SNR-gamma loss-weight probes (tools/min_snr_oracle.py). SNR(t)=ab/(1-ab).
  cSNRProbeT: array[0..5] of integer = (1, 25, 50, 100, 150, 200);
  cSNREpsGamma5: array[0..5] of double =
    (0.000500050005000445, 0.16531233763786016, 0.6811466136251436,
     1.0, 1.0, 1.0);
  cSNRVGamma5: array[0..5] of double =
    (0.0004999999999999449, 0.16002162776613316, 0.5994798761147474,
     0.6024803053077055, 0.32038735949336433, 0.13218275425061793);
  cSNRVGammaInf: array[0..5] of double =
    (0.9999, 0.9679956744467734, 0.8801040247770505,
     0.6024803053077055, 0.32038735949336433, 0.13218275425061793);

procedure TTestNeuralDiffusion.ToyModel(Xt, Output: TNNetVolume; Tt: integer);
var i: integer; s: TNeuralFloat;
begin
  s := Sin(0.01 * Tt);
  for i := 0 to Xt.Size - 1 do
    Output.FData[i] := s * Xt.FData[i];
end;

procedure TTestNeuralDiffusion.RecordingModel(Xt, Output: TNNetVolume;
  Tt: integer);
begin
  if FRecN <= High(FRecT) then FRecT[FRecN] := Tt;
  Inc(FRecN);
  Output.Fill(0);
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

procedure TTestNeuralDiffusion.TestSplitPredictionAt;
var
  Sched: TNNetDiffusionScheduler;
  X0, Eps, Xt, RawPred, X0Out, EpsOut: TNNetVolume;
  Prediction: TNNetPredictionType;
  i: integer;
  sab, somab, NearSab, NearSomab: TNeuralFloat;
begin
  X0 := TNNetVolume.Create(cN, 1, 1);
  Eps := TNNetVolume.Create(cN, 1, 1);
  Xt := TNNetVolume.Create(cN, 1, 1);
  RawPred := TNNetVolume.Create(cN, 1, 1);
  X0Out := TNNetVolume.Create;
  EpsOut := TNNetVolume.Create;
  try
    for i := 0 to cN - 1 do begin X0.FData[i] := 0.5 * i - 1; Eps.FData[i] := 0.3 - 0.1 * i; end;
    for Prediction := dpEps to dpV do
    begin
      Sched := TNNetDiffusionScheduler.Create(cT, dsLinear, Prediction, cBeta1, cBetaT);
      try
        Sched.AlphaBarRootsAt(99, sab, somab);
        AssertEquals('integer sqrt(ab)', Sqrt(Sched.AlphaBar[100]), sab, 1e-6);
        AssertEquals('integer sqrt(1-ab)', Sqrt(1 - Sched.AlphaBar[100]), somab, 1e-6);
        Sched.AlphaBarRootsAt(99 + 1e-6, NearSab, NearSomab);
        AssertEquals('fractional sqrt(ab)', sab, NearSab, 1e-5);
        AssertEquals('fractional sqrt(1-ab)', somab, NearSomab, 1e-5);
        AssertEquals('fractional roots on the unit circle', 1,
          Sqr(NearSab) + Sqr(NearSomab), 1e-6);
        for i := 0 to cN - 1 do
        begin
          Xt.FData[i] := sab * X0.FData[i] + somab * Eps.FData[i];
          if Prediction = dpEps then RawPred.FData[i] := Eps.FData[i]
          else RawPred.FData[i] := sab * Eps.FData[i] - somab * X0.FData[i];
        end;
        Sched.SplitPrediction(Xt, RawPred, X0Out, EpsOut, sab, somab);
        for i := 0 to cN - 1 do
        begin
          AssertEquals('x0 @ ' + IntToStr(i), X0.FData[i], X0Out.FData[i], 1e-4);
          AssertEquals('eps @ ' + IntToStr(i), Eps.FData[i], EpsOut.FData[i], 1e-4);
        end;
      finally
        Sched.Free;
      end;
    end;
  finally
    X0.Free; Eps.Free; Xt.Free; RawPred.Free; X0Out.Free; EpsOut.Free;
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
    TestApplyCFGRescale(Cond, Uncond, w);
  finally
    Cond.Free; Uncond.Free; Dst.Free;
  end;
end;

// Fills a 2-element volume; the APG tests below are hand-computed on 2 elements.
procedure SetPair(V: TNNetVolume; A, B: TNeuralFloat);
begin
  V.ReSize(2, 1, 1);
  V.FData[0] := A;
  V.FData[1] := B;
end;

procedure TTestNeuralDiffusion.TestAPGMatchesCFG;
var
  Cond, Uncond, Expected, Dst: TNNetVolume;
  APG: TNNetAdaptiveProjectedGuidance;
  i: integer;
begin
  Cond := TNNetVolume.Create(cN, 1, 1);
  Uncond := TNNetVolume.Create(cN, 1, 1);
  Expected := TNNetVolume.Create(cN, 1, 1);
  Dst := TNNetVolume.Create(cN, 1, 1);
  APG := TNNetAdaptiveProjectedGuidance.Create(1, 0, 0);
  try
    for i := 0 to cN - 1 do
    begin
      Cond.FData[i] := Sin(i * 0.7) + 0.3;
      Uncond.FData[i] := Cos(i * 1.3) * 0.5;
    end;
    TNNetDiffusionScheduler.ApplyCFG(Cond, Uncond, Expected, 4.5);
    APG.ApplyAPG(Cond, Uncond, Dst, 4.5);
    for i := 0 to cN - 1 do
      AssertEquals('eta=1,m=0,r=0 is CFG @ ' + IntToStr(i),
        Expected.FData[i], Dst.FData[i], 0);
    // W = 1 is the diffusers no-op scale: the conditional comes back unchanged.
    APG.Eta := 0;
    APG.ApplyAPG(Cond, Uncond, Dst, 1);
    for i := 0 to cN - 1 do
      AssertEquals('W=1 returns cond @ ' + IntToStr(i),
        Cond.FData[i], Dst.FData[i], 0);
  finally
    APG.Free; Cond.Free; Uncond.Free; Expected.Free; Dst.Free;
  end;
end;

procedure TTestNeuralDiffusion.TestAPGProjectionAndClamp;
var
  Cond, Uncond, Dst: TNNetVolume;
  APG: TNNetAdaptiveProjectedGuidance;
begin
  // Cond = (3,4), Uncond = (1,1): diff = (2,3), u = (0.6,0.8), diff.u = 3.6,
  // par = (2.16,2.88), orth = (-0.16,0.12), ||diff|| = sqrt(13).
  Cond := TNNetVolume.Create; Uncond := TNNetVolume.Create;
  Dst := TNNetVolume.Create;
  APG := TNNetAdaptiveProjectedGuidance.Create(0, 0, 0);
  try
    SetPair(Cond, 3, 4); SetPair(Uncond, 1, 1);
    // eta = 0: Uncond + 2*orth.
    APG.ApplyAPG(Cond, Uncond, Dst, 2);
    AssertEquals('eta=0 [0]', 0.68, Dst.FData[0], 1e-6);
    AssertEquals('eta=0 [1]', 1.24, Dst.FData[1], 1e-6);
    // eta = 0.5: Uncond + 2*(orth + 0.5*par) = (2.84, 4.12).
    APG.Eta := 0.5;
    APG.ApplyAPG(Cond, Uncond, Dst, 2);
    AssertEquals('eta=0.5 [0]', 2.84, Dst.FData[0], 1e-5);
    AssertEquals('eta=0.5 [1]', 4.12, Dst.FData[1], 1e-5);
    // Original formulation adds to Cond: (3,4) + 2*orth with eta = 0.
    APG.Eta := 0;
    APG.UseOriginalFormulation := true;
    APG.ApplyAPG(Cond, Uncond, Dst, 2);
    AssertEquals('original [0]', 2.68, Dst.FData[0], 1e-6);
    AssertEquals('original [1]', 4.24, Dst.FData[1], 1e-6);
    APG.UseOriginalFormulation := false;
    // r = 1 clamps diff to diff/sqrt(13); eta = 1 keeps it whole.
    APG.Eta := 1;
    APG.NormThreshold := 1;
    APG.ApplyAPG(Cond, Uncond, Dst, 2);
    AssertEquals('clamp eta=1 [0]', 1 + 4 / Sqrt(13), Dst.FData[0], 1e-6);
    AssertEquals('clamp eta=1 [1]', 1 + 6 / Sqrt(13), Dst.FData[1], 1e-6);
    // Clamp and projection together: Uncond + 2*orth/sqrt(13).
    APG.Eta := 0;
    APG.ApplyAPG(Cond, Uncond, Dst, 2);
    AssertEquals('clamp eta=0 [0]', 1 - 0.32 / Sqrt(13), Dst.FData[0], 1e-6);
    AssertEquals('clamp eta=0 [1]', 1 + 0.24 / Sqrt(13), Dst.FData[1], 1e-6);
    // A threshold above ||diff|| leaves diff alone: plain CFG (5,7).
    APG.Eta := 1;
    APG.NormThreshold := 10;
    APG.ApplyAPG(Cond, Uncond, Dst, 2);
    AssertEquals('no clamp [0]', 5, Dst.FData[0], 1e-6);
    AssertEquals('no clamp [1]', 7, Dst.FData[1], 1e-6);
  finally
    APG.Free; Cond.Free; Uncond.Free; Dst.Free;
  end;
end;

procedure TTestNeuralDiffusion.TestAPGMomentum;
var
  Cond, Uncond, Dst: TNNetVolume;
  APG: TNNetAdaptiveProjectedGuidance;
begin
  // Step 1: Cond (3,4), Uncond (1,1), avg = (2,3). Step 2: Cond (2,1), Uncond
  // (1,2), diff = (1,-1), avg = (1,-1) - 0.5*(2,3) = (0,-2.5).
  Cond := TNNetVolume.Create; Uncond := TNNetVolume.Create;
  Dst := TNNetVolume.Create;
  APG := TNNetAdaptiveProjectedGuidance.Create(0, -0.5, 0);
  try
    SetPair(Cond, 3, 4); SetPair(Uncond, 1, 1);
    APG.ApplyAPG(Cond, Uncond, Dst, 2);
    AssertEquals('m step1 [0]', 0.68, Dst.FData[0], 1e-6);
    AssertEquals('m step1 [1]', 1.24, Dst.FData[1], 1e-6);
    // Step 2, eta = 0: par = (-2.5/5)*(2,1) = (-1,-0.5), orth = (1,-2),
    // Dst = (1,2) + 2*(1,-2) = (3,-2).
    SetPair(Cond, 2, 1); SetPair(Uncond, 1, 2);
    APG.ApplyAPG(Cond, Uncond, Dst, 2);
    AssertEquals('m step2 [0]', 3, Dst.FData[0], 1e-6);
    AssertEquals('m step2 [1]', -2, Dst.FData[1], 1e-6);
    // After Reset the history is gone: avg = (1,-1), eta = 1 -> (3,0).
    APG.Reset;
    APG.Eta := 1;
    APG.ApplyAPG(Cond, Uncond, Dst, 2);
    AssertEquals('reset [0]', 3, Dst.FData[0], 1e-6);
    AssertEquals('reset [1]', 0, Dst.FData[1], 1e-6);
    // The clamp must not leak into the running average: with r = 1 step 2
    // sees avg = (0,-2.5), clamps it to (0,-1), Dst = (1,2) + 2*(0,-1).
    APG.Reset;
    APG.NormThreshold := 1;
    SetPair(Cond, 3, 4); SetPair(Uncond, 1, 1);
    APG.ApplyAPG(Cond, Uncond, Dst, 2);
    AssertEquals('m+clamp step1 [0]', 1 + 4 / Sqrt(13), Dst.FData[0], 1e-6);
    SetPair(Cond, 2, 1); SetPair(Uncond, 1, 2);
    APG.ApplyAPG(Cond, Uncond, Dst, 2);
    AssertEquals('m+clamp step2 [0]', 1, Dst.FData[0], 1e-6);
    AssertEquals('m+clamp step2 [1]', 0, Dst.FData[1], 1e-6);
  finally
    APG.Free; Cond.Free; Uncond.Free; Dst.Free;
  end;
end;

procedure TTestNeuralDiffusion.TestAPGAliasing;
var
  Cond, Uncond, Expected, Alias: TNNetVolume;
  APG: TNNetAdaptiveProjectedGuidance;
  Formulation, i: integer;
begin
  Cond := TNNetVolume.Create(cN, 1, 1);
  Uncond := TNNetVolume.Create(cN, 1, 1);
  Expected := TNNetVolume.Create(cN, 1, 1);
  Alias := TNNetVolume.Create(cN, 1, 1);
  APG := TNNetAdaptiveProjectedGuidance.Create(0.3, 0, 0.5);
  try
    for i := 0 to cN - 1 do
    begin
      Cond.FData[i] := Sin(i * 0.9) + 0.2;
      Uncond.FData[i] := Cos(i * 0.4) * 0.3;
    end;
    for Formulation := 0 to 1 do
    begin
      APG.UseOriginalFormulation := Formulation = 1;
      APG.ApplyAPG(Cond, Uncond, Expected, 3);
      Alias.Copy(Cond);
      APG.ApplyAPG(Alias, Uncond, Alias, 3);
      for i := 0 to cN - 1 do
        AssertEquals('Dst = Cond @ ' + IntToStr(i),
          Expected.FData[i], Alias.FData[i], 1e-5);
      Alias.Copy(Uncond);
      APG.ApplyAPG(Cond, Alias, Alias, 3);
      for i := 0 to cN - 1 do
        AssertEquals('Dst = Uncond @ ' + IntToStr(i),
          Expected.FData[i], Alias.FData[i], 1e-5);
    end;
  finally
    APG.Free; Cond.Free; Uncond.Free; Expected.Free; Alias.Free;
  end;
end;

// Fills V with the three values A, B, C.
procedure SetVolume3(V: TNNetVolume; A, B, C: TNeuralFloat);
begin
  V.ReSize(3, 1, 1);
  V.FData[0] := A;
  V.FData[1] := B;
  V.FData[2] := C;
end;

procedure AssertVolume3(const Msg: string; V: TNNetVolume;
  A, B, C, Tolerance: TNeuralFloat);
begin
  TAssert.AssertEquals(Msg + ' [0]', A, V.FData[0], Tolerance);
  TAssert.AssertEquals(Msg + ' [1]', B, V.FData[1], Tolerance);
  TAssert.AssertEquals(Msg + ' [2]', C, V.FData[2], Tolerance);
end;

// Cond = (1,2,3), Uncond = (0,1,-1), Skip = (2,0,1), W = 3, PerturbedScale = 2:
// Cond-Uncond = (1,1,4), Cond-Skip = (-1,2,2).
procedure TTestNeuralDiffusion.TestPerturbedGuidanceBranches;
var
  Cond, Uncond, Skip, Dst, Expected: TNNetVolume;
  Guider: TNNetPerturbedGuidance;
  i: integer;
begin
  Cond := TNNetVolume.Create; Uncond := TNNetVolume.Create;
  Skip := TNNetVolume.Create; Dst := TNNetVolume.Create;
  Expected := TNNetVolume.Create;
  Guider := TNNetPerturbedGuidance.Create(3, 2);
  try
    SetVolume3(Cond, 1, 2, 3);
    SetVolume3(Uncond, 0, 1, -1);
    SetVolume3(Skip, 2, 0, 1);
    // NumSteps 0: no window, both guidances on.
    Guider.Apply(0, 0, Cond, Uncond, Skip, Dst);
    AssertVolume3('CFG+PAG: Uncond + 3(C-U) + 2(C-S)', Dst, 1, 8, 15, 1e-6);
    Guider.UseOriginalFormulation := true;
    Guider.Apply(0, 0, Cond, Uncond, Skip, Dst);
    AssertVolume3('CFG+PAG original: Cond + 3(C-U) + 2(C-S)', Dst, 2, 9, 19,
      1e-6);
    // CFG off: GuidanceScale is the no-op scale (0 original, 1 otherwise).
    Guider.GuidanceScale := 0;
    Guider.Apply(0, 0, Cond, nil, Skip, Dst);
    AssertVolume3('PAG only original: Cond + 2(C-S)', Dst, -1, 6, 7, 1e-6);
    Guider.UseOriginalFormulation := false;
    Guider.GuidanceScale := 1;
    Guider.Apply(0, 0, Cond, nil, Skip, Dst);
    AssertVolume3('PAG only: Skip + 2(C-S)', Dst, 0, 4, 5, 1e-6);
    // PAG off at scale 0: CFG alone, bit-equal to ApplyCFG (the unguided-by-PAG
    // pipeline), and Cond itself when CFG is off too.
    Guider.GuidanceScale := 3;
    Guider.PerturbedScale := 0;
    Guider.Apply(0, 0, Cond, Uncond, nil, Dst);
    AssertVolume3('CFG only: Uncond + 3(C-U)', Dst, 3, 4, 11, 1e-6);
    TNNetDiffusionScheduler.ApplyCFG(Cond, Uncond, Expected, 3);
    for i := 0 to 2 do
      AssertEquals('PerturbedScale 0 is ApplyCFG bit for bit @ ' + IntToStr(i),
        Expected.FData[i], Dst.FData[i], 0);
    Guider.UseOriginalFormulation := true;
    Guider.Apply(0, 0, Cond, Uncond, nil, Dst);
    AssertVolume3('CFG only original: Cond + 3(C-U)', Dst, 4, 5, 15, 1e-6);
    Guider.UseOriginalFormulation := false;
    Guider.GuidanceScale := 1;
    Guider.Apply(0, 0, Cond, nil, nil, Dst);
    AssertVolume3('neither: Cond', Dst, 1, 2, 3, 0);
    // A guidance that is on needs its prediction.
    Guider.PerturbedScale := 2;
    try
      Guider.Apply(0, 0, Cond, nil, nil, Dst);
      Fail('PAG on without Skip must raise');
    except
      on E: EArgumentException do ;
    end;
    try
      TNNetPerturbedGuidance.Create(3, 2, 0.5, 0.2).Free;
      Fail('stop below start must raise');
    except
      on E: EArgumentException do ;
    end;
    Guider.SetPerturbedLayers([1, 4]);
    AssertTrue('layer 4 perturbed', Guider.PerturbsLayer(4));
    AssertFalse('layer 2 not perturbed', Guider.PerturbsLayer(2));
  finally
    Guider.Free; Cond.Free; Uncond.Free; Skip.Free; Dst.Free; Expected.Free;
  end;
end;

procedure TTestNeuralDiffusion.TestPerturbedGuidanceWindows;
var
  Guider: TNNetPerturbedGuidance;
begin
  // Defaults 0.01 .. 0.2 over 100 steps: Trunc(1) < Step < Trunc(20).
  Guider := TNNetPerturbedGuidance.Create(3, 2);
  try
    AssertFalse('step 1 is the strict start', Guider.IsPerturbedEnabled(1, 100));
    AssertTrue('step 2 inside', Guider.IsPerturbedEnabled(2, 100));
    AssertTrue('step 19 inside', Guider.IsPerturbedEnabled(19, 100));
    AssertFalse('step 20 is the stop', Guider.IsPerturbedEnabled(20, 100));
    // 10 steps: Trunc(0.1) = 0 < Step < Trunc(2.0) = 2, only step 1.
    AssertFalse('10 steps: step 0', Guider.IsPerturbedEnabled(0, 10));
    AssertTrue('10 steps: step 1', Guider.IsPerturbedEnabled(1, 10));
    AssertFalse('10 steps: step 2', Guider.IsPerturbedEnabled(2, 10));
    // The CFG window includes its start: Trunc(0.5*10) <= Step < Trunc(1*10).
    Guider.GuidanceStart := 0.5;
    AssertFalse('CFG step 4', Guider.IsCFGEnabled(4, 10));
    AssertTrue('CFG step 5 is the inclusive start', Guider.IsCFGEnabled(5, 10));
    AssertTrue('CFG step 9', Guider.IsCFGEnabled(9, 10));
    AssertFalse('CFG step 10', Guider.IsCFGEnabled(10, 10));
    Guider.GuidanceScale := 1;
    AssertFalse('CFG at the no-op scale', Guider.IsCFGEnabled(5, 10));
    Guider.PerturbedScale := 0;
    AssertFalse('PAG at scale 0', Guider.IsPerturbedEnabled(2, 100));
  finally
    Guider.Free;
  end;
end;

procedure TTestNeuralDiffusion.TestPerturbedGuidanceRescaleAndAliasing;
var
  Cond, Uncond, Skip, Dst, Expected: TNNetVolume;
  Guider: TNNetPerturbedGuidance;
  i: integer;
begin
  Cond := TNNetVolume.Create; Uncond := TNNetVolume.Create;
  Skip := TNNetVolume.Create; Dst := TNNetVolume.Create;
  Expected := TNNetVolume.Create;
  Guider := TNNetPerturbedGuidance.Create(3, 2, 0.01, 0.2, 0.5);
  try
    // The mix is (1,8,15): std sqrt(98/3); std(Cond) = sqrt(2/3), so the
    // factor is 0.5*(1/7) + 0.5 = 4/7.
    SetVolume3(Uncond, 0, 1, -1);
    SetVolume3(Skip, 2, 0, 1);
    SetVolume3(Cond, 1, 2, 3);
    Guider.Apply(0, 0, Cond, Uncond, Skip, Dst);
    AssertVolume3('rescale 0.5', Dst, 4 / 7, 32 / 7, 60 / 7, 1e-5);
    // Dst aliasing each input gives the same result.
    Guider.Apply(0, 0, Cond, Uncond, Skip, Cond);
    AssertVolume3('Dst = Cond', Cond, 4 / 7, 32 / 7, 60 / 7, 1e-5);
    SetVolume3(Cond, 1, 2, 3);
    Guider.Apply(0, 0, Cond, Uncond, Skip, Uncond);
    AssertVolume3('Dst = Uncond', Uncond, 4 / 7, 32 / 7, 60 / 7, 1e-5);
    SetVolume3(Uncond, 0, 1, -1);
    Guider.Apply(0, 0, Cond, Uncond, Skip, Skip);
    AssertVolume3('Dst = Skip', Skip, 4 / 7, 32 / 7, 60 / 7, 1e-5);
    // ApplyCFG with rescale takes std(EpsCond) before Dst overwrites it.
    SetVolume3(Cond, 1, 2, 3);
    TNNetDiffusionScheduler.ApplyCFG(Cond, Uncond, Expected, 3, 0.5);
    TNNetDiffusionScheduler.ApplyCFG(Cond, Uncond, Cond, 3, 0.5);
    for i := 0 to 2 do
      AssertEquals('ApplyCFG rescale with Dst = EpsCond @ ' + IntToStr(i),
        Expected.FData[i], Cond.FData[i], 1e-6);
  finally
    Guider.Free; Cond.Free; Uncond.Free; Skip.Free; Dst.Free; Expected.Free;
  end;
end;

procedure TTestNeuralDiffusion.TestAPGCreateFromSpec;
var
  APG: TNNetAdaptiveProjectedGuidance;
  Raised: boolean;
begin
  APG := TNNetAdaptiveProjectedGuidance.CreateFromSpec('0, -0.5,15.25');
  try
    AssertEquals('eta', 0, APG.Eta, 0);
    AssertEquals('momentum', -0.5, APG.Momentum, 0);
    AssertEquals('threshold', 15.25, APG.NormThreshold, 0);
    AssertFalse('diffusers default formulation', APG.UseOriginalFormulation);
  finally
    APG.Free;
  end;
  Raised := false;
  try
    TNNetAdaptiveProjectedGuidance.CreateFromSpec('0,-0.5').Free;
  except
    on EConvertError do Raised := true;
  end;
  AssertTrue('two fields raise', Raised);
end;

procedure TTestNeuralDiffusion.TestRawPredFromX0RoundTrip;
const
  cTt = 400;
var
  Sched: TNNetDiffusionScheduler;
  Xt, Raw, Back: TNNetVolume;
  Prediction: TNNetPredictionType;
  i: integer;
begin
  Xt := TNNetVolume.Create(cN, 1, 1);
  Raw := TNNetVolume.Create(cN, 1, 1);
  Back := TNNetVolume.Create(cN, 1, 1);
  try
    for i := 0 to cN - 1 do
    begin
      Xt.FData[i] := 0.3 * i - 1.1;
      Raw.FData[i] := 0.7 * Sin(i * 1.7);
    end;
    for Prediction := dpEps to dpV do
    begin
      Sched := TNNetDiffusionScheduler.Create(1000, dsLinear, Prediction);
      try
        Sched.PredictX0(Xt, Raw, Back, cTt);
        Sched.RawPredFromX0(Xt, Back, Back, cTt);
        for i := 0 to cN - 1 do
          AssertEquals('raw round trip @ ' + IntToStr(i),
            Raw.FData[i], Back.FData[i], 1e-5);
      finally
        Sched.Free;
      end;
    end;
  finally
    Xt.Free; Raw.Free; Back.Free;
  end;
end;

procedure TTestNeuralDiffusion.TestApplyCFGRescale(
  Cond, Uncond: TNNetVolume; w: TNeuralFloat);
var
  Plain, Rescaled: TNNetVolume;
  i: integer;
  stdCond, stdPlain, stdRescaled, phi: TNeuralFloat;
begin
  Plain    := TNNetVolume.Create(cN, 1, 1);
  Rescaled := TNNetVolume.Create(cN, 1, 1);
  try
    // (a) GuidanceRescale = 0 reproduces the plain CFG mix exactly.
    TNNetDiffusionScheduler.ApplyCFG(Cond, Uncond, Plain, w);
    TNNetDiffusionScheduler.ApplyCFG(Cond, Uncond, Rescaled, w, 0.0);
    for i := 0 to cN - 1 do
      AssertEquals('rescale phi=0 @ ' + IntToStr(i),
        Plain.FData[i], Rescaled.FData[i], 1e-6);

    // (b) phi>0 pulls std(result) toward std(Cond) (w>1 overshoots it).
    phi := 0.7;
    TNNetDiffusionScheduler.ApplyCFG(Cond, Uncond, Rescaled, w, phi);
    stdCond     := Cond.GetStdDeviation();
    stdPlain    := Plain.GetStdDeviation();
    stdRescaled := Rescaled.GetStdDeviation();
    AssertTrue('plain CFG std overshoots cond std for w>1',
      stdPlain > stdCond);
    AssertTrue('rescaled std is closer to cond std than plain CFG std',
      Abs(stdRescaled - stdCond) < Abs(stdPlain - stdCond));
  finally
    Plain.Free; Rescaled.Free;
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

// BuildSamplerSchedule against diffusers' timesteps (+1, then the clean 0):
// PNDM linspace repeats the second one, DPMSolverMultistep linspace drops the
// last of N+1 points. The driver evaluates the model once per entry.
procedure TTestNeuralDiffusion.TestSamplerSchedulesVsDiffusers;
const
  cPNDMTimesteps: array[0..11] of integer =
    (200, 178, 178, 156, 134, 112, 89, 67, 45, 23, 1, 0);
  cDPMTimesteps: array[0..10] of integer =
    (200, 180, 160, 140, 120, 101, 81, 61, 41, 21, 0);
var
  Sched: TNNetDiffusionScheduler;
  Schedule: TNeuralIntegerArray;
  X: TNNetVolume;
  StepPos: integer;
begin
  Sched := TNNetDiffusionScheduler.Create(cT, dsLinear, dpEps, cBeta1, cBetaT);
  X := TNNetVolume.Create(cN, 1, 1);
  try
    Schedule := Sched.BuildSamplerSchedule(cSteps, tsUniform, smPNDM);
    AssertEquals('pndm length', Length(cPNDMTimesteps), Length(Schedule));
    for StepPos := 0 to High(cPNDMTimesteps) do
      AssertEquals('pndm t ' + IntToStr(StepPos), cPNDMTimesteps[StepPos],
        Schedule[StepPos]);
    Schedule := Sched.BuildSamplerSchedule(cSteps, tsUniform, smDPMSolverPP2M);
    AssertEquals('dpm length', Length(cDPMTimesteps), Length(Schedule));
    for StepPos := 0 to High(cDPMTimesteps) do
      AssertEquals('dpm t ' + IntToStr(StepPos), cDPMTimesteps[StepPos],
        Schedule[StepPos]);
    // One PNDM step is a single plain hop; N >= 2 adds the repeated eval.
    AssertEquals('pndm 1 step', 2,
      Length(Sched.BuildSamplerSchedule(1, tsUniform, smPNDM)));
    FRecN := 0;
    Sched.Sample(X, @RecordingModel, cSteps, smPNDM, 0.0, tsUniform);
    AssertEquals('pndm evals', cSteps + 1, FRecN);
  finally
    Sched.Free; X.Free;
  end;
end;

// _class_name -> sampler; unknown classes and unimplemented options run with a
// notice instead of failing.
procedure TTestNeuralDiffusion.TestDiffusersSchedulerSamplerMapping;
var
  Notice: string;

  function Mapped(const Json: string): TNNetSamplerMethod;
  begin
    Result := DiffusersSchedulerSampler(Json, Notice);
  end;

begin
  AssertTrue('ddim', Mapped('{"_class_name": "DDIMScheduler"}') = smDDIM);
  AssertEquals('ddim notice', '', Notice);
  AssertTrue('pndm', Mapped('{"_class_name": "PNDMScheduler", ' +
    '"skip_prk_steps": true}') = smPNDM);
  AssertEquals('pndm notice', '', Notice);
  AssertTrue('pndm prk', Mapped('{"_class_name": "PNDMScheduler"}') = smPNDM);
  AssertTrue('prk noticed', Pos('skip_prk_steps', Notice) > 0);
  AssertTrue('euler', Mapped('{"_class_name": "EulerDiscreteScheduler", ' +
    '"interpolation_type": "linear", "use_karras_sigmas": false}') = smEuler);
  AssertEquals('euler notice', '', Notice);
  AssertTrue('euler a', Mapped('{"_class_name": ' +
    '"EulerAncestralDiscreteScheduler"}') = smEulerAncestral);
  AssertTrue('dpm', Mapped('{"_class_name": "DPMSolverMultistepScheduler", ' +
    '"algorithm_type": "dpmsolver++", "solver_order": 2}') = smDPMSolverPP2M);
  AssertEquals('dpm notice', '', Notice);
  AssertTrue('dpm karras', Mapped('{"_class_name": ' +
    '"DPMSolverMultistepScheduler", "use_karras_sigmas": true, ' +
    '"solver_order": 3}') = smDPMSolverPP2M);
  AssertTrue('karras followed, not noticed',
    Pos('use_karras_sigmas', Notice) = 0);
  AssertTrue('order noticed', Pos('solver_order 3', Notice) > 0);
  AssertTrue('dpm karras flag', DiffusersSchedulerUsesKarrasSigmas(
    '{"_class_name": "DPMSolverMultistepScheduler", "use_karras_sigmas": true}'));
  AssertTrue('euler karras flag', DiffusersSchedulerUsesKarrasSigmas(
    '{"_class_name": "EulerDiscreteScheduler", "use_karras_sigmas": true}'));
  AssertFalse('euler a has no karras', DiffusersSchedulerUsesKarrasSigmas(
    '{"_class_name": "EulerAncestralDiscreteScheduler", ' +
    '"use_karras_sigmas": true}'));
  AssertFalse('karras off', DiffusersSchedulerUsesKarrasSigmas(
    '{"_class_name": "EulerDiscreteScheduler"}'));
  Mapped('{"_class_name": "EulerDiscreteScheduler", "use_karras_sigmas": ' +
    'true, "sigma_max": 10.0}');
  AssertTrue('karras sigma_max noticed', Pos('sigma_max', Notice) > 0);
  AssertTrue('lms', Mapped('{"_class_name": "LMSDiscreteScheduler"}') = smDDIM);
  AssertTrue('lms noticed', Pos('LMSDiscreteScheduler', Notice) > 0);
  AssertTrue('not an object', Mapped('[1]') = smDDIM);
  AssertTrue('array noticed', Notice <> '');
end;

// PLMS with diffusers' quirks: the target is t - T div N (not the next linspace
// entry), the second eval re-runs the first hop with the mean eps, and v is
// converted after the multistep mix.
procedure TTestNeuralDiffusion.TestPNDMVsDiffusers;
var
  Sched: TNNetDiffusionScheduler;
  X: TNNetVolume;
  Prediction: TNNetPredictionType;
  i: integer;
begin
  X := TNNetVolume.Create(cN, 1, 1);
  try
    for Prediction := dpEps to dpV do
    begin
      Sched := TNNetDiffusionScheduler.Create(cT, dsLinear, Prediction, cBeta1,
        cBetaT, 0.008, false, {pFinalAlphaBarIsOne=}false);
      try
        for i := 0 to cN - 1 do X.FData[i] := (i - cN / 2) * 0.3;
        Sched.Sample(X, @ToyModel, cSteps, smPNDM, 0.0, tsUniform);
        for i := 0 to cN - 1 do
          if Prediction = dpEps then
            AssertEquals('pndm eps @ ' + IntToStr(i), cOraclePNDM[i],
              X.FData[i], 1e-4)
          else
            AssertEquals('pndm v @ ' + IntToStr(i), cOraclePNDMV[i],
              X.FData[i], 1e-4);
      finally
        Sched.Free;
      end;
    end;
  finally
    X.Free;
  end;
end;

// Deterministic Euler with trailing spacing, whose init_noise_sigma is
// sigma_max: the VP start is the unit noise times InitialNoiseScale.
procedure TTestNeuralDiffusion.TestEulerVsDiffusers;
var
  Sched: TNNetDiffusionScheduler;
  X: TNNetVolume;
  Scale: TNeuralFloat;
  i: integer;
begin
  Sched := TNNetDiffusionScheduler.Create(cT, dsLinear, dpEps, cBeta1, cBetaT);
  X := TNNetVolume.Create(cN, 1, 1);
  try
    Scale := Sched.InitialNoiseScale(smEuler, tsTrailing,
      Sched.BuildSamplerSchedule(cSteps, tsTrailing, smEuler)[0]);
    AssertEquals('leading scale', 1.0,
      Sched.InitialNoiseScale(smEuler, tsLeading, cT), 0);
    AssertEquals('ddim scale', 1.0,
      Sched.InitialNoiseScale(smDDIM, tsTrailing, cT), 0);
    for i := 0 to cN - 1 do X.FData[i] := (i - cN / 2) * 0.3 * Scale;
    Sched.Sample(X, @ToyModel, cSteps, smEuler, 1.0, tsTrailing);
    for i := 0 to cN - 1 do
      AssertEquals('euler @ ' + IntToStr(i), cOracleEuler[i], X.FData[i], 1e-4);
  finally
    Sched.Free; X.Free;
  end;
end;

// Euler-ancestral (Eta = 1) replaying the noise diffusers drew.
procedure TTestNeuralDiffusion.TestEulerAncestralVsDiffusers;
const
  cAncestralSteps = 4;
var
  Sched: TNNetDiffusionScheduler;
  X, Pred, Noise: TNNetVolume;
  Schedule: TNeuralIntegerArray;
  StepPos, i: integer;
begin
  Sched := TNNetDiffusionScheduler.Create(cT, dsLinear, dpEps, cBeta1, cBetaT);
  X := TNNetVolume.Create(cN, 1, 1);
  Pred := TNNetVolume.Create(cN, 1, 1);
  Noise := TNNetVolume.Create(cN, 1, 1);
  try
    Schedule := Sched.BuildSamplerSchedule(cAncestralSteps, tsLeading,
      smEulerAncestral);
    AssertEquals('first t', 151, Schedule[0]);
    for i := 0 to cN - 1 do X.FData[i] := (i - cN / 2) * 0.3;
    for StepPos := 0 to cAncestralSteps - 1 do
    begin
      ToyModel(X, Pred, Schedule[StepPos]);
      if StepPos < cAncestralSteps - 1 then
        for i := 0 to cN - 1 do Noise.FData[i] := cEulerANoise[StepPos * cN + i];
      Sched.Step(X, Pred, Schedule[StepPos], Schedule[StepPos + 1],
        smEulerAncestral, 1.0, Noise);
    end;
    for i := 0 to cN - 1 do
      AssertEquals('euler-a @ ' + IntToStr(i), cOracleEulerA[i], X.FData[i],
        1e-4);
  finally
    Sched.Free; X.Free; Pred.Free; Noise.Free;
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
    // Match the DPM-Solver++(2M) float64 oracle (same toy model & start).
    for i := 0 to cN - 1 do
    begin
      AssertFalse('dpm++ no NaN @ ' + IntToStr(i),
        IsNan(X.FData[i]) or IsInfinite(X.FData[i]));
      AssertEquals('dpm++ vs oracle @ ' + IntToStr(i),
        cOracleDPM[i], X.FData[i], 1e-4);
    end;
  finally
    Sched.Free; X.Free;
  end;
end;

procedure TTestNeuralDiffusion.TestUniPCVsOracle;
var
  Sched: TNNetDiffusionScheduler;
  X: TNNetVolume;
  i: integer;
begin
  Sched := TNNetDiffusionScheduler.Create(cT, dsLinear, dpEps, cBeta1, cBetaT);
  X := TNNetVolume.Create(cN, 1, 1);
  try
    for i := 0 to cN - 1 do X.FData[i] := (i - cN / 2) * 0.3;
    Sched.Sample(X, @ToyModel, cSteps, smUniPC, 0.0);
    // Match the numpy UniPC (order-2 bh2) float64 oracle to <1e-4. UniPC is a
    // genuinely distinct predictor-corrector trajectory: it differs from both
    // the DDIM and DPM-Solver++(2M) finals on the SAME toy model & start.
    for i := 0 to cN - 1 do
    begin
      AssertFalse('unipc no NaN @ ' + IntToStr(i),
        IsNan(X.FData[i]) or IsInfinite(X.FData[i]));
      AssertEquals('unipc vs oracle @ ' + IntToStr(i),
        cOracleUniPC[i], X.FData[i], 1e-4);
    end;
  finally
    Sched.Free; X.Free;
  end;
end;

procedure TTestNeuralDiffusion.TestHeunVsOracle;
var
  Sched: TNNetDiffusionScheduler;
  X: TNNetVolume;
  i: integer;
begin
  // Heun 2nd-order deterministic sampler (k-diffusion sample_heun) with the
  // Karras rho=7 spacing must reproduce the float64 numpy oracle to <1e-4. This
  // is a genuine intra-step predictor-corrector: the Sample() driver calls the
  // denoiser TWICE per step (Euler predict + 2nd-derivative correction) and
  // skips the corrector on the final hop (sigma=0). It is a DISTINCT trajectory
  // from the single-eval DDIM/DPM++/UniPC finals on the SAME toy model & start.
  Sched := TNNetDiffusionScheduler.Create(cT, dsLinear, dpEps, cBeta1, cBetaT);
  X := TNNetVolume.Create(cN, 1, 1);
  try
    for i := 0 to cN - 1 do X.FData[i] := (i - cN / 2) * 0.3;
    Sched.Sample(X, @ToyModel, cSteps, smHeun, 0.0, tsKarras);
    for i := 0 to cN - 1 do
    begin
      AssertFalse('heun no NaN @ ' + IntToStr(i),
        IsNan(X.FData[i]) or IsInfinite(X.FData[i]));
      AssertEquals('heun vs oracle @ ' + IntToStr(i),
        cOracleHeun[i], X.FData[i], 1e-4);
    end;
  finally
    Sched.Free; X.Free;
  end;
end;

procedure TTestNeuralDiffusion.TestHeunZeroSigmaRunsFinite;
var
  Sched: TNNetDiffusionScheduler;
  X: TNNetVolume;
  i: integer;
begin
  // Beta1 = 0 makes alpha_bar_1 exactly 1, so SigmaOf(1) = 0. With uniform
  // spacing and NumSteps = T the Heun driver visits Tt = 1 (sigma = 0) and
  // Tt = 2 -> TtPrev = 1 (sigma_next = 0). Both reciprocals must stay finite.
  Sched := TNNetDiffusionScheduler.Create(cT, dsLinear, dpEps, 0.0, cBetaT);
  X := TNNetVolume.Create(cN, 1, 1);
  try
    AssertEquals('sigma at Tt=1 is 0', 0.0, Sched.SigmaAt[1], 0.0);
    for i := 0 to cN - 1 do X.FData[i] := (i - cN / 2) * 0.3;
    Sched.Sample(X, @ToyModel, cT, smHeun, 0.0, tsUniform);
    for i := 0 to cN - 1 do
      AssertFalse('heun zero-sigma finite @ ' + IntToStr(i),
        IsNan(X.FData[i]) or IsInfinite(X.FData[i]));
  finally
    Sched.Free; X.Free;
  end;
end;

procedure TTestNeuralDiffusion.TestMinSNRWeight;
var
  SchedE, SchedV: TNNetDiffusionScheduler;
  i, t: integer;
const
  cBigGamma = 1.0e30;   // stands in for +Inf (clamp never binds).
begin
  // Min-SNR-gamma loss weighting (Hang et al. 2023). eps-prediction weight is
  // min(SNR,gamma)/SNR; v-prediction is min(SNR,gamma)/(SNR+1). Pin BOTH
  // prediction types against the float64 numpy oracle for gamma=5.0, and verify
  // the gamma=+inf limits: eps -> 1 (unweighted), v -> SNR/(SNR+1).
  SchedE := TNNetDiffusionScheduler.Create(cT, dsLinear, dpEps, cBeta1, cBetaT);
  SchedV := TNNetDiffusionScheduler.Create(cT, dsLinear, dpV, cBeta1, cBetaT);
  try
    for i := 0 to High(cSNRProbeT) do
    begin
      t := cSNRProbeT[i];
      AssertEquals('eps gamma=5 @ ' + IntToStr(t),
        cSNREpsGamma5[i], SchedE.SNRWeight(t, 5.0), 1e-5);
      AssertEquals('v gamma=5 @ ' + IntToStr(t),
        cSNRVGamma5[i], SchedV.SNRWeight(t, 5.0), 1e-5);
      // gamma = +inf: eps weight collapses to the unweighted objective (1.0).
      AssertEquals('eps gamma=inf -> 1 @ ' + IntToStr(t),
        1.0, SchedE.SNRWeight(t, cBigGamma), 1e-5);
      // gamma = +inf: v weight collapses to SNR/(SNR+1) = ab_t.
      AssertEquals('v gamma=inf -> SNR/(SNR+1) @ ' + IntToStr(t),
        cSNRVGammaInf[i], SchedV.SNRWeight(t, cBigGamma), 1e-5);
    end;
  finally
    SchedE.Free; SchedV.Free;
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

procedure TTestNeuralDiffusion.TestKarrasSnapMatchesNearestSigmaScan;
var
  Sched: TNNetDiffusionScheduler;
  X: TNNetVolume;
  k, t, best: integer;
  d, bestD, invRho, pMax, pMin, frac: double;
  target: TNeuralFloat;
const
  cRho = 7.0;
begin
  // The Karras spacing snaps each warped sigma back to the timestep whose
  // schedule sigma is NEAREST. The scheduler searches its precomputed sigma
  // table; this pins that search against the defining exhaustive scan over the
  // public SigmaAt, including the tie rule (lowest timestep wins).
  Sched := TNNetDiffusionScheduler.Create(cT, dsLinear, dpEps, cBeta1, cBetaT);
  X := TNNetVolume.Create(cN, 1, 1);
  try
    for k := 0 to cN - 1 do X.FData[k] := 0.1 * k;
    FRecN := 0;
    Sched.Sample(X, @RecordingModel, cSteps, smDDIM, 0.0, tsKarras);
    AssertEquals('one denoiser call per step', cSteps, FRecN);
    invRho := 1.0 / cRho;
    pMax := Power(Sched.SigmaAt[cT], invRho);
    pMin := Power(Sched.SigmaAt[1], invRho);
    for k := 0 to cSteps - 1 do
    begin
      frac := k / (cSteps - 1);
      target := Power(pMax + frac * (pMin - pMax), cRho);
      best := 1; bestD := Abs(Sched.SigmaAt[1] - target);
      for t := 2 to cT do
      begin
        d := Abs(Sched.SigmaAt[t] - target);
        if d < bestD then begin bestD := d; best := t; end;
      end;
      AssertEquals('karras snap @ ' + IntToStr(k), best, FRecT[k]);
    end;
  finally
    Sched.Free; X.Free;
  end;
end;

procedure TTestNeuralDiffusion.TestSNRWeightClosedForm;
var
  SchedE, SchedV: TNNetDiffusionScheduler;
  t: integer;
  ab, snr, clamped: double;
begin
  // SNR(t) = exp(2*lambda_t) = ab_t/(1-ab_t). Pin the weights against that
  // closed form over the WHOLE schedule (the oracle test probes 5 timesteps).
  SchedE := TNNetDiffusionScheduler.Create(cT, dsLinear, dpEps, cBeta1, cBetaT);
  SchedV := TNNetDiffusionScheduler.Create(cT, dsLinear, dpV, cBeta1, cBetaT);
  try
    for t := 1 to cT do
    begin
      ab := SchedE.AlphaBar[t];
      snr := ab / (1.0 - ab);
      if snr < 5.0 then clamped := snr else clamped := 5.0;
      AssertEquals('eps weight @ ' + IntToStr(t),
        clamped / snr, SchedE.SNRWeight(t, 5.0), 1e-6);
      AssertEquals('v weight @ ' + IntToStr(t),
        clamped / (snr + 1.0), SchedV.SNRWeight(t, 5.0), 1e-6);
    end;
  finally
    SchedE.Free; SchedV.Free;
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
begin
  // diffusers get_scalings_for_boundary_condition_discrete at t = Tt - 1,
  // timestep_scaling 10: c_skip = 0.25/((10t)^2+0.25), c_out = 10t/sqrt(...).
  Sched := TNNetLCMScheduler.Create(cT, dsLinear, dpEps, cBeta1, cBetaT);
  try
    AssertEquals('c_skip(t=0)', 1.0, Sched.CSkip(1), 0);
    AssertEquals('c_out(t=0)', 0.0, Sched.COut(1), 0);
    AssertEquals('c_skip(t=1)', 0.0024937655860349127, Sched.CSkip(2), 1e-9);
    AssertEquals('c_out(t=1)', 0.9987523388778445, Sched.COut(2), 1e-7);
    AssertEquals('c_skip(t=199)', 6.312971496113009e-08, Sched.CSkip(cT), 1e-12);
    AssertEquals('c_out(t=199)', 0.9999999684351419, Sched.COut(cT), 1e-7);
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
    Sched.Sample(X, @LCMOracleModel, 4, smLCM);
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
    SchedA.Sample(XA, @LCMOracleModel, 4, smLCM);

    RandSeed := 777; gLCMSched := SchedB;
    for i := 0 to cN - 1 do XB.FData[i] := RandG(0, 1);
    SchedB.Sample(XB, @LCMOracleModel, 4, smLCM);

    for i := 0 to cN - 1 do
      AssertEquals('lcm reproducible @ ' + IntToStr(i),
        XA.FData[i], XB.FData[i], 1e-6);
  finally
    gLCMTarget := nil; gLCMSched := nil;
    SchedA.Free; SchedB.Free; XA.Free; XB.Free; Target.Free;
  end;
end;


// diffusers LCMScheduler.set_timesteps (SD 1.5 betas) as 1-based Tt, for
// step counts, strengths and original_inference_steps; the refusals; and the
// inherited samplers for other methods.
procedure TTestNeuralDiffusion.TestLCMOriginScheduleVsDiffusers;
var
  Sched: TNNetLCMScheduler;
  Base: TNNetDiffusionScheduler;
  X: TNNetVolume;

  procedure CheckSchedule(const Name: string; const Actual: TNeuralIntegerArray;
    const Expected: array of integer);
  var
    StepPos: integer;
  begin
    AssertEquals(Name + ' length', Length(Expected) + 1, Length(Actual));
    for StepPos := 0 to High(Expected) do
      AssertEquals(Name + ' @ ' + IntToStr(StepPos), Expected[StepPos] + 1,
        Actual[StepPos]);
    AssertEquals(Name + ' final', 0, Actual[Length(Expected)]);
  end;

  function Refused(NumSteps: integer; Strength: double): boolean;
  begin
    Result := false;
    try
      Sched.BuildOriginSchedule(NumSteps, Strength);
    except
      on Exception do Result := true;
    end;
  end;

begin
  Sched := TNNetLCMScheduler.Create(1000, dsScaledLinear, dpEps, 0.00085, 0.012);
  Base := TNNetDiffusionScheduler.Create(1000, dsScaledLinear, dpEps, 0.00085,
    0.012);
  X := TNNetVolume.Create(cN, 1, 1);
  try
    AssertEquals('default original steps', 50, Sched.OriginalInferenceSteps);
    CheckSchedule('4 steps', Sched.BuildSamplerSchedule(4, tsLeading, smLCM),
      [999, 759, 499, 259]);
    CheckSchedule('8 steps', Sched.BuildOriginSchedule(8),
      [999, 879, 759, 639, 499, 379, 259, 139]);
    CheckSchedule('1 step', Sched.BuildOriginSchedule(1), [999]);
    CheckSchedule('strength 0.5', Sched.BuildImg2ImgSchedule(4, tsLeading,
      smLCM, 0.5), [499, 379, 259, 139]);
    CheckSchedule('strength 0.3', Sched.BuildOriginSchedule(3, 0.3),
      [299, 199, 99]);
    AssertTrue('51 steps refused', Refused(51, 1));
    AssertTrue('strength 0.05 leaves 2 of 4 points', Refused(4, 0.05));
    AssertTrue('strength 0 refused', Refused(1, 0));
    AssertFalse('50 steps run', Refused(50, 1));
    Sched.OriginalInferenceSteps := 25;
    CheckSchedule('original 25', Sched.BuildOriginSchedule(4),
      [999, 759, 519, 279]);
    // Other methods: the inherited schedule and img2img cut.
    CheckSchedule('ddim', Sched.BuildSamplerSchedule(4, tsLeading, smDDIM),
      [750, 500, 250, 0]);
    CheckSchedule('ddim strength 0.5', Sched.BuildImg2ImgSchedule(4, tsLeading,
      smDDIM, 0.5), [250, 0]);
    AssertEquals('base matches', Base.BuildSamplerSchedule(4, tsLeading,
      smDDIM)[0], Sched.BuildSamplerSchedule(4, tsLeading, smDDIM)[0]);
    // The base class does not run smLCM.
    try
      Base.BuildSamplerSchedule(4, tsLeading, smLCM);
      Fail('base BuildSamplerSchedule ran smLCM');
    except
      on E: Exception do
        AssertTrue('schedule refusal names the class',
          Pos('TNNetLCMScheduler', E.Message) > 0);
    end;
    try
      Base.Step(X, X, 10, 5, smLCM);
      Fail('base Step ran smLCM');
    except
      on E: Exception do
        AssertTrue('step refusal names the class',
          Pos('TNNetLCMScheduler', E.Message) > 0);
    end;
  finally
    X.Free;
    Base.Free;
    Sched.Free;
  end;
end;

// original_inference_steps / timestep_scaling come from the config (defaults
// 50 / 10 for a config without them); DiffusersSchedulerSampler maps the class.
procedure TTestNeuralDiffusion.TestLCMFromDiffusersConfig;
const
  csLCMConfig = '{"_class_name": "LCMScheduler", "beta_end": 0.012, ' +
    '"beta_schedule": "scaled_linear", "beta_start": 0.00085, ' +
    '"num_train_timesteps": 1000, "original_inference_steps": 25, ' +
    '"timestep_scaling": 5.0, "clip_sample": false, "set_alpha_to_one": true}';
  csDDIMConfig = '{"_class_name": "DDIMScheduler", "beta_end": 0.012, ' +
    '"beta_schedule": "scaled_linear", "beta_start": 0.00085, ' +
    '"clip_sample": false}';
var
  FileName, Notice: string;
  Lines: TStringList;
  Sched: TNNetLCMScheduler;
begin
  FileName := GetTempDir(false) + 'cai_lcm_scheduler_config.json';
  Lines := TStringList.Create;
  Sched := nil;
  try
    Lines.Text := csLCMConfig;
    Lines.SaveToFile(FileName);
    Sched := TNNetLCMScheduler.CreateFromDiffusersConfig(FileName);
    AssertEquals('original steps', 25, Sched.OriginalInferenceSteps);
    AssertEquals('timestep scaling', 5.0, Sched.TimestepScaling, 0);
    AssertEquals('c_out at t = 1, scaling 5', 5 / Sqrt(25.25), Sched.COut(2),
      1e-7);
    AssertTrue('LCMScheduler -> smLCM',
      DiffusersSchedulerSampler(csLCMConfig, Notice) = smLCM);
    AssertEquals('lcm notice', '', Notice);
    FreeAndNil(Sched);
    Lines.Text := csDDIMConfig;
    Lines.SaveToFile(FileName);
    Sched := TNNetLCMScheduler.CreateFromDiffusersConfig(FileName);
    AssertEquals('default original steps', 50, Sched.OriginalInferenceSteps);
    AssertEquals('default timestep scaling', 10.0, Sched.TimestepScaling, 0);
  finally
    Sched.Free;
    Lines.Free;
    DeleteFile(FileName);
  end;
end;

const
  cFlowOracleFile = 'qwenimage21_scheduler_oracle.json';

// Parses tests/fixtures/<FileName> (run from the repo root or from tests/).
function LoadFlowOracle(const FileName: string): TJSONObject;
var
  Path: string;
  SS: TStringStream;
  Parser: TJSONParser;
begin
  Path := 'fixtures' + DirectorySeparator + FileName;
  if not FileExists(Path) then
    Path := 'tests' + DirectorySeparator + 'fixtures' + DirectorySeparator +
      FileName;
  if not FileExists(Path) then
    raise Exception.Create('Fixture not found: ' + FileName +
      ' (run tools/make_pico_qwenimage21_fixture.py).');
  SS := TStringStream.Create('');
  try
    SS.LoadFromFile(Path);
    Parser := TJSONParser.Create(SS.DataString, []);
    try
      Result := Parser.Parse as TJSONObject;
    finally
      Parser.Free;
    end;
  finally
    SS.Free;
  end;
end;

// The Qwen-Image-2.1 scheduler_config.json settings stored in the oracle.
function CreateFlowSchedulerFromOracle(Config: TJSONObject): TNNetFlowMatchEulerScheduler;
begin
  TAssert.AssertEquals('time_shift_type', 'exponential', Config.Get('time_shift_type', ''));
  Result := TNNetFlowMatchEulerScheduler.Create(
    Config.Get('num_train_timesteps', 0), Config.Get('shift', 0.0),
    Config.Get('use_dynamic_shifting', false), ftsExponential,
    Config.Get('shift_terminal', 0.0));
end;

function OracleMu(Config, CaseObj: TJSONObject): double;
begin
  Result := TNNetFlowMatchEulerScheduler.CalculateShift(
    CaseObj.Get('image_seq_len', 0), Config.Get('base_image_seq_len', 0),
    Config.Get('max_image_seq_len', 0), Config.Get('base_shift', 0.0),
    Config.Get('max_shift', 0.0));
end;

procedure TTestNeuralDiffusion.TestFlowMatchCalculateShiftVsOracle;
var
  Root, Config, CaseObj: TJSONObject;
  Cases: TJSONArray;
  CasePos: integer;
begin
  // mu is linear in the token count and extrapolates past 8192 and below 256.
  Root := LoadFlowOracle(cFlowOracleFile);
  try
    Config := Root.Objects['config'];
    Cases := Root.Arrays['cases'];
    AssertEquals('oracle case count', 6, Cases.Count);
    for CasePos := 0 to Cases.Count - 1 do
    begin
      CaseObj := Cases.Objects[CasePos];
      AssertEquals('mu case ' + IntToStr(CasePos),
        CaseObj.Get('mu', 0.0), OracleMu(Config, CaseObj), 1e-12);
    end;
  finally
    Root.Free;
  end;
end;

procedure TTestNeuralDiffusion.TestFlowMatchSigmasVsOracle;
var
  Root, Config, CaseObj: TJSONObject;
  Cases, Expected, Linspace: TJSONArray;
  Sched, SchedFromList: TNNetFlowMatchEulerScheduler;
  CasePos, SigmaPos, NumSteps: integer;
  InputSigmas: array of double;
begin
  // The schedule is computed in double, so it must match the float64 oracle
  // to rounding (1e-12); diffusers' own float32 table differs by ~1e-7.
  Root := LoadFlowOracle(cFlowOracleFile);
  try
    Config := Root.Objects['config'];
    Cases := Root.Arrays['cases'];
    for CasePos := 0 to Cases.Count - 1 do
    begin
      CaseObj := Cases.Objects[CasePos];
      NumSteps := CaseObj.Get('num_steps', 0);
      Sched := CreateFlowSchedulerFromOracle(Config);
      SchedFromList := CreateFlowSchedulerFromOracle(Config);
      try
        Sched.SetTimesteps(NumSteps, CaseObj.Get('mu', 0.0));
        AssertEquals('num steps', NumSteps, Sched.NumSteps);
        Linspace := CaseObj.Arrays['sigmas_linspace'];
        SetLength(InputSigmas, Linspace.Count);
        for SigmaPos := 0 to Linspace.Count - 1 do
          InputSigmas[SigmaPos] := Linspace.Floats[SigmaPos];
        SchedFromList.SetSigmas(InputSigmas, CaseObj.Get('mu', 0.0));
        Expected := CaseObj.Arrays['sigmas_f64'];
        AssertEquals('sigma count', NumSteps + 1, Expected.Count);
        for SigmaPos := 0 to NumSteps do
        begin
          AssertEquals('sigma case ' + IntToStr(CasePos) + ' @ ' +
            IntToStr(SigmaPos), Expected.Floats[SigmaPos],
            Sched.Sigma[SigmaPos], 1e-12);
          AssertEquals('sigma from list case ' + IntToStr(CasePos) + ' @ ' +
            IntToStr(SigmaPos), Expected.Floats[SigmaPos],
            SchedFromList.Sigma[SigmaPos], 1e-12);
        end;
        AssertEquals('last shifted sigma = shift_terminal',
          Config.Get('shift_terminal', 0.0), Sched.Sigma[NumSteps - 1], 1e-12);
        Expected := CaseObj.Arrays['timesteps_f64'];
        for SigmaPos := 0 to NumSteps - 1 do
          AssertEquals('timestep case ' + IntToStr(CasePos) + ' @ ' +
            IntToStr(SigmaPos), Expected.Floats[SigmaPos],
            Sched.Timestep[SigmaPos], 1e-9);
      finally
        Sched.Free;
        SchedFromList.Free;
      end;
    end;
  finally
    Root.Free;
  end;
end;

procedure TTestNeuralDiffusion.TestFlowMatchStepVsOracle;
var
  Root, Config, CaseObj: TJSONObject;
  Cases, SampleData, Velocities, Outputs, Expected: TJSONArray;
  Sched: TNNetFlowMatchEulerScheduler;
  Sample, Velocity: TNNetVolume;
  CasePos, StepPos, ElementPos: integer;
begin
  // Two Euler steps in single precision against the float64 formula. The
  // values are O(1) and each step adds one rounding, so 2e-6 absolute is
  // ~10 single-precision ulps.
  Root := LoadFlowOracle(cFlowOracleFile);
  Sample := TNNetVolume.Create;
  Velocity := TNNetVolume.Create;
  try
    Config := Root.Objects['config'];
    Cases := Root.Arrays['cases'];
    for CasePos := 0 to Cases.Count - 1 do
    begin
      CaseObj := Cases.Objects[CasePos];
      SampleData := CaseObj.Arrays['step_sample'];
      Velocities := CaseObj.Arrays['step_velocity'];
      Outputs := CaseObj.Arrays['step_out_f64'];
      Sample.ReSize(SampleData.Count, 1, 1);
      Velocity.ReSize(SampleData.Count, 1, 1);
      for ElementPos := 0 to SampleData.Count - 1 do
        Sample.FData[ElementPos] := SampleData.Floats[ElementPos];
      Sched := CreateFlowSchedulerFromOracle(Config);
      try
        Sched.SetTimesteps(CaseObj.Get('num_steps', 0), OracleMu(Config, CaseObj));
        for StepPos := 0 to Velocities.Count - 1 do
        begin
          for ElementPos := 0 to SampleData.Count - 1 do
            Velocity.FData[ElementPos] :=
              Velocities.Arrays[StepPos].Floats[ElementPos];
          Sched.Step(Sample, Velocity, StepPos);
          Expected := Outputs.Arrays[StepPos];
          for ElementPos := 0 to SampleData.Count - 1 do
            AssertEquals('step case ' + IntToStr(CasePos) + ' step ' +
              IntToStr(StepPos) + ' @ ' + IntToStr(ElementPos),
              Expected.Floats[ElementPos], Sample.FData[ElementPos], 2e-6);
        end;
      finally
        Sched.Free;
      end;
    end;
  finally
    Root.Free;
    Sample.Free;
    Velocity.Free;
  end;
end;

procedure TTestNeuralDiffusion.TestFlowMatchStaticAndLinearShift;
var
  Sched: TNNetFlowMatchEulerScheduler;
begin
  // Static shift 3 (SD3): sigma' = 3s / (1 + 2s); no terminal stretch.
  Sched := TNNetFlowMatchEulerScheduler.Create(1000, 3.0);
  try
    Sched.SetSigmas([1.0, 0.5, 0.25]);
    AssertEquals('static shift s=1', 1.0, Sched.Sigma[0], 1e-15);
    AssertEquals('static shift s=0.5', 0.75, Sched.Sigma[1], 1e-15);
    AssertEquals('static shift s=0.25', 0.5, Sched.Sigma[2], 1e-15);
    AssertEquals('final sigma', 0.0, Sched.Sigma[3], 0);
    AssertEquals('timestep', 750.0, Sched.Timestep[1], 1e-12);
  finally
    Sched.Free;
  end;
  // Linear dynamic shift: sigma' = mu / (mu + 1/s - 1); mu = 2, s = 0.5 -> 2/3.
  Sched := TNNetFlowMatchEulerScheduler.Create(1000, 1.0, true, ftsLinear);
  try
    Sched.SetSigmas([0.5], 2.0);
    AssertEquals('linear shift', 2 / 3, Sched.Sigma[0], 1e-15);
    AssertEquals('num steps', 1, Sched.NumSteps);
  finally
    Sched.Free;
  end;
end;

// CreateFromDiffusersConfig on the real Qwen-Image-2.1 scheduler_config.json
// (copied into the pico folder): every field, mu at both ends of the line, the
// 3-step schedule the pipeline oracle recorded, and a refused Karras config.
procedure TTestNeuralDiffusion.TestFlowMatchFromDiffusersConfig;
var
  Sched: TNNetFlowMatchEulerScheduler;
  ConfigFile, KarrasFile: string;
  Lines: TStringList;
  Refused: boolean;
begin
  ConfigFile := 'fixtures' + DirectorySeparator + 'tiny_qwenimage21' +
    DirectorySeparator + 'scheduler' + DirectorySeparator +
    'scheduler_config.json';
  if not FileExists(ConfigFile) then
    ConfigFile := 'tests' + DirectorySeparator + ConfigFile;
  Sched := TNNetFlowMatchEulerScheduler.CreateFromDiffusersConfig(ConfigFile);
  try
    AssertEquals('num_train_timesteps', 1000, Sched.NumTrainTimesteps);
    AssertTrue('use_dynamic_shifting', Sched.UseDynamicShifting);
    AssertTrue('exponential', Sched.TimeShiftType = ftsExponential);
    AssertEquals('shift_terminal', 0.02, Sched.ShiftTerminal, 1e-15);
    AssertEquals('base_image_seq_len', 256, Sched.BaseImageSeqLen);
    AssertEquals('max_image_seq_len', 8192, Sched.MaxImageSeqLen);
    AssertEquals('mu at 256 tokens', 0.5, Sched.ShiftForImageSeqLen(256), 1e-12);
    AssertEquals('mu at 8192 tokens', 0.9, Sched.ShiftForImageSeqLen(8192),
      1e-12);
    // tiny_qwenimage21_pipeline_io.json: 8 image tokens, 3 steps (float32).
    Sched.SetTimesteps(3, Sched.ShiftForImageSeqLen(8));
    AssertEquals('sigma 0', 1.0, Sched.Sigma[0], 1e-7);
    AssertEquals('sigma 1', 0.5823222398757935, Sched.Sigma[1], 1e-7);
    AssertEquals('sigma 2', 0.019999980926513672, Sched.Sigma[2], 1e-7);
  finally
    Sched.Free;
  end;
  KarrasFile := GetTempDir(true) + 'karras_scheduler_config.json';
  Lines := TStringList.Create;
  try
    Lines.Text := '{"use_karras_sigmas": true}';
    Lines.SaveToFile(KarrasFile);
  finally
    Lines.Free;
  end;
  Refused := false;
  try
    try
      TNNetFlowMatchEulerScheduler.CreateFromDiffusersConfig(KarrasFile).Free;
    except
      on Exception do Refused := true;
    end;
  finally
    DeleteFile(KarrasFile);
  end;
  AssertTrue('Karras sigmas are refused', Refused);
end;

const
  cImg2ImgOracleFile = 'qwenimage21_img2img_oracle.json';

procedure TTestNeuralDiffusion.TestFlowMatchImg2ImgStartStepVsOracle;
var
  Root, Config, CaseObj: TJSONObject;
  Cases: TJSONArray;
  Sched: TNNetFlowMatchEulerScheduler;
  CasePos, NumSteps, StartStep: integer;
  Mu: double;
  Refused: boolean;

  function IsRefused(StepCount: integer; Strength: double): boolean;
  begin
    Result := false;
    try
      TNNetFlowMatchEulerScheduler.Img2ImgStartStep(StepCount, Strength);
    except
      on Exception do Result := true;
    end;
  end;

begin
  // t_start = int(N - min(N * strength, N)) from QwenImageImg2ImgPipeline,
  // and Sigma[t_start] of the 2.1 schedule to the float64 formula.
  Root := LoadFlowOracle(cImg2ImgOracleFile);
  Sched := nil;
  try
    Config := Root.Objects['config'];
    Cases := Root.Arrays['cases'];
    AssertTrue('oracle cases', Cases.Count >= 10);
    for CasePos := 0 to Cases.Count - 1 do
    begin
      CaseObj := Cases.Objects[CasePos];
      NumSteps := CaseObj.Get('num_steps', 0);
      StartStep := TNNetFlowMatchEulerScheduler.Img2ImgStartStep(NumSteps,
        CaseObj.Get('strength', 0.0));
      AssertEquals('t_start case ' + IntToStr(CasePos),
        CaseObj.Get('t_start', -1), StartStep);
      AssertEquals('steps run case ' + IntToStr(CasePos),
        CaseObj.Get('steps_run', -1), NumSteps - StartStep);
      Sched := CreateFlowSchedulerFromOracle(Config);
      Mu := Sched.CalculateShift(Root.Get('image_seq_len', 0),
        Config.Get('base_image_seq_len', 0), Config.Get('max_image_seq_len', 0),
        Config.Get('base_shift', 0.0), Config.Get('max_shift', 0.0));
      AssertEquals('mu case ' + IntToStr(CasePos), CaseObj.Get('mu', 0.0), Mu,
        1e-12);
      Sched.SetTimesteps(NumSteps, Mu);
      AssertEquals('sigma[t_start] case ' + IntToStr(CasePos),
        CaseObj.Get('sigma_start_f64', 0.0), Sched.Sigma[StartStep], 1e-12);
      FreeAndNil(Sched);
    end;
    Cases := Root.Arrays['zero_step_cases'];
    AssertEquals('zero-step cases', 2, Cases.Count);
    for CasePos := 0 to Cases.Count - 1 do
    begin
      CaseObj := Cases.Objects[CasePos];
      AssertTrue('no step left is refused, case ' + IntToStr(CasePos),
        IsRefused(CaseObj.Get('num_steps', 0), CaseObj.Get('strength', 0.0)));
    end;
    AssertTrue('strength above 1 is refused', IsRefused(10, 1.5));
    AssertTrue('negative strength is refused', IsRefused(10, -0.1));
    AssertTrue('NaN strength is refused', IsRefused(10, NaN));
    Refused := IsRefused(10, 1e-10);
    AssertFalse('a tiny strength still runs one step', Refused);
    AssertEquals('which is the last', 9,
      TNNetFlowMatchEulerScheduler.Img2ImgStartStep(10, 1e-10));
  finally
    Sched.Free;
    Root.Free;
  end;
end;

procedure TTestNeuralDiffusion.TestFlowMatchScaleNoiseVsOracle;
var
  Root, Config, Oracle: TJSONObject;
  Sched: TNNetFlowMatchEulerScheduler;
  Noise, Sample: TNNetVolume;
  Expected: TJSONArray;
  ElementPos, StartStep: integer;
begin
  // sigma * noise + (1 - sigma) * sample in single precision: 1e-6 absolute.
  Root := LoadFlowOracle(cImg2ImgOracleFile);
  Sched := nil;
  Noise := TNNetVolume.Create;
  Sample := TNNetVolume.Create;
  try
    Config := Root.Objects['config'];
    Oracle := Root.Objects['scale_noise'];
    Sample.ReSize(6, 1, 1);
    Noise.ReSize(6, 1, 1);
    for ElementPos := 0 to 5 do
    begin
      Sample.FData[ElementPos] := Oracle.Arrays['sample'].Floats[ElementPos];
      Noise.FData[ElementPos] := Oracle.Arrays['noise'].Floats[ElementPos];
    end;
    Sched := CreateFlowSchedulerFromOracle(Config);
    Sched.SetTimesteps(Oracle.Get('num_steps', 0), Sched.CalculateShift(
      Root.Get('image_seq_len', 0), Config.Get('base_image_seq_len', 0),
      Config.Get('max_image_seq_len', 0), Config.Get('base_shift', 0.0),
      Config.Get('max_shift', 0.0)));
    StartStep := TNNetFlowMatchEulerScheduler.Img2ImgStartStep(
      Oracle.Get('num_steps', 0), Oracle.Get('strength', 0.0));
    AssertEquals('t_start', Oracle.Get('t_start', -1), StartStep);
    Sched.ScaleNoise(Noise, Sample, StartStep);
    Expected := Oracle.Arrays['noised_f64'];
    for ElementPos := 0 to 5 do
      AssertEquals('noised @ ' + IntToStr(ElementPos),
        Expected.Floats[ElementPos], Noise.FData[ElementPos], 1e-6);
  finally
    Sched.Free;
    Noise.Free;
    Sample.Free;
    Root.Free;
  end;
end;

// The synthetic velocity of the DPM-Solver++ flow tests:
// v[i] = x[i]*(0.5 + sigma) - 0.25*(i - 2).
procedure FlowSyntheticVelocity(Sample, Velocity: TNNetVolume; Sigma: double);
var
  ElementPos, MaxElementPos: integer;
begin
  Velocity.ReSize(Sample);
  MaxElementPos := Sample.Size - 1;
  for ElementPos := 0 to MaxElementPos do
    Velocity.FData[ElementPos] := Sample.FData[ElementPos] * (0.5 + Sigma) -
      0.25 * (ElementPos - 2);
end;

procedure TTestNeuralDiffusion.TestFlowDPMSolverFirstOrderMatchesEuler;
var
  EulerSched, DPMSched: TNNetFlowMatchEulerScheduler;
  EulerSample, DPMSample, Velocity: TNNetVolume;
  StepPos, ElementPos: integer;
begin
  // With no history (ResetMultistep before each step) DPM-Solver++ is
  // algebraically Euler; single-precision rounding differs by a few ulps.
  EulerSched := TNNetFlowMatchEulerScheduler.Create(1000, 1.0, true,
    ftsExponential, 0.02);
  DPMSched := TNNetFlowMatchEulerScheduler.Create(1000, 1.0, true,
    ftsExponential, 0.02);
  EulerSample := TNNetVolume.Create(6, 1, 1);
  DPMSample := TNNetVolume.Create(6, 1, 1);
  Velocity := TNNetVolume.Create;
  try
    DPMSched.Sampler := fsDPMSolverPP2M;
    EulerSched.SetTimesteps(5, 0.9);
    DPMSched.SetTimesteps(5, 0.9);
    for ElementPos := 0 to 5 do
      EulerSample.FData[ElementPos] := (ElementPos - 2.5) * 0.4;
    DPMSample.Copy(EulerSample);
    for StepPos := 0 to 4 do
    begin
      FlowSyntheticVelocity(EulerSample, Velocity, EulerSched.Sigma[StepPos]);
      EulerSched.Step(EulerSample, Velocity, StepPos);
      FlowSyntheticVelocity(DPMSample, Velocity, DPMSched.Sigma[StepPos]);
      DPMSched.ResetMultistep;
      DPMSched.Step(DPMSample, Velocity, StepPos);
      for ElementPos := 0 to 5 do
        AssertEquals('first order = Euler, step ' + IntToStr(StepPos) +
          ' @ ' + IntToStr(ElementPos), EulerSample.FData[ElementPos],
          DPMSample.FData[ElementPos], 2e-6);
    end;
  finally
    EulerSched.Free;
    DPMSched.Free;
    EulerSample.Free;
    DPMSample.Free;
    Velocity.Free;
  end;
end;

procedure TTestNeuralDiffusion.TestFlowDPMSolverTrajectoryVsDiffusers;
const
  // diffusers 0.41.0.dev0 DPMSolverMultistepScheduler(use_flow_sigmas=True,
  // prediction_type="flow_prediction", solver_order=2, lower_order_final=True,
  // final_sigmas_type="zero") stepped on the sigma lists below with
  // FlowSyntheticVelocity from x[i] = (i - 2.5)*0.4 (its float32 sample math).
  // A: SetSigmas([0.95, 0.6, 0.3], mu 0.7); one genuine second-order step.
  cSigmasA: array[0..3] of double = (0.9745296268011178, 0.7512827010955053,
    0.463241971419116, 0.0);
  cStepsA: array[0..2, 0..5] of double = (
    (-0.7824392295667284, -0.4583012093361499, -0.13416315930324893,
      0.18997489072965204, 0.514112940762553, 0.8382509609931316),
    (-0.6796619052657934, -0.38724843550802984, -0.0948349574459361,
      0.19757851316557706, 0.48999200612883204, 0.7824054162819507),
    (-0.6080081589563989, -0.33026321244354107, -0.05251826133186225,
      0.22522668565379014, 0.5029716450175218, 0.7807165585221684));
  // B: SetTimesteps(4, mu 1.1), shift_terminal 0.02; sigma 1 first (alpha 0).
  cSigmasB: array[0..4] of double = (1.0, 0.8041087361931913,
    0.5101699356554767, 0.019999999999999907, 0.0);
  cStepsB: array[0..3, 0..5] of double = (
    (-0.804108738899231, -0.47267070399105604, -0.14123262437939757,
      0.19020544033109973, 0.5216435199427583, 0.8530815548509332),
    (-0.6428406036386997, -0.36496741175716274, -0.08709418341764345,
      0.19077903773626875, 0.46865225117462683, 0.7465254728584861),
    (-0.8502000205115119, -0.48088724952555567, -0.11157443469503514,
      0.2577383705284962, 0.6270511684727749, 0.9963639713685698),
    (-0.8513579402981921, -0.4808860221304899, -0.11041406057420679,
      0.2600578914749998, 0.630529836320658, 1.0010017860663367));
  // C: schedule B entered at step 1 (img2img), so step 1 is first order.
  cStepsC: array[1..3, 0..5] of double = (
    (-0.7636412204236053, -0.44348782202959985, -0.1233343640309496,
      0.19681906416537825, 0.5169725221640284, 0.8371259205580339),
    (-1.0049550234468114, -0.581478010454454, -0.15800093038279034,
      0.26547611545072813, 0.688953197385037, 1.1124302066521041),
    (-1.0045034912029647, -0.5804306391457277, -0.15635772070680934,
      0.26771516385004057, 0.6917880841322326, 1.1158609325029223));
var
  Sched: TNNetFlowMatchEulerScheduler;
  Sample, Velocity: TNNetVolume;
  StepPos: integer;

  procedure StartSample;
  var
    ElementPos: integer;
  begin
    for ElementPos := 0 to 5 do
      Sample.FData[ElementPos] := (ElementPos - 2.5) * 0.4;
  end;

  procedure RunStep(StepIdx: integer; const Expected: array of double;
    const CaseName: string);
  var
    ElementPos: integer;
  begin
    FlowSyntheticVelocity(Sample, Velocity, Sched.Sigma[StepIdx]);
    Sched.Step(Sample, Velocity, StepIdx);
    for ElementPos := 0 to 5 do
      AssertEquals(CaseName + ' step ' + IntToStr(StepIdx) + ' @ ' +
        IntToStr(ElementPos), Expected[ElementPos], Sample.FData[ElementPos],
        2e-5);
  end;

begin
  Sched := nil;
  Sample := TNNetVolume.Create(6, 1, 1);
  Velocity := TNNetVolume.Create;
  try
    Sched := TNNetFlowMatchEulerScheduler.Create(1000, 1.0, true);
    Sched.Sampler := fsDPMSolverPP2M;
    Sched.SetSigmas([0.95, 0.6, 0.3], 0.7);
    for StepPos := 0 to 3 do
      AssertEquals('sigma A ' + IntToStr(StepPos), cSigmasA[StepPos],
        Sched.Sigma[StepPos], 1e-12);
    StartSample;
    for StepPos := 0 to 2 do RunStep(StepPos, cStepsA[StepPos], 'A');
    Sched.Free;
    Sched := TNNetFlowMatchEulerScheduler.Create(1000, 1.0, true,
      ftsExponential, 0.02);
    Sched.Sampler := fsDPMSolverPP2M;
    Sched.SetTimesteps(4, 1.1);
    for StepPos := 0 to 4 do
      AssertEquals('sigma B ' + IntToStr(StepPos), cSigmasB[StepPos],
        Sched.Sigma[StepPos], 1e-12);
    StartSample;
    for StepPos := 0 to 3 do RunStep(StepPos, cStepsB[StepPos], 'B');
    // A new trajectory: history from the end of B must not leak into C.
    Sched.ResetMultistep;
    StartSample;
    for StepPos := 1 to 3 do RunStep(StepPos, cStepsC[StepPos], 'C');
  finally
    Sched.Free;
    Sample.Free;
    Velocity.Free;
  end;
end;

procedure TTestNeuralDiffusion.TestFlowUniPCTrajectoryVsDiffusers;
const
  // diffusers 0.41.0.dev0 UniPCMultistepScheduler(use_flow_sigmas=True,
  // flow_shift=1.0, prediction_type="flow_prediction", solver_order=2,
  // lower_order_final=True, final_sigmas_type="zero") given the sigma lists
  // below through set_timesteps(sigmas=...), stepped with FlowSyntheticVelocity
  // from x[i] = (i - 2.5)*0.4 (its float32 sample math; it moves the first
  // sigma 1 to float32(1 - 1e-6)).
  // A: Create(1000, 3.0) static shift (Wan 480p), SetTimesteps(7), bh2.
  cSigmasA: array[0..7] of double = (1.0, 0.9473684210526314,
    0.8823529411764705, 0.7999999999999999, 0.6923076923076923,
    0.5454545454545453, 0.33333333333333337, 0.0);
  cStepsA: array[0..6, 0..5] of double = (
    (-0.9473694562911987, -0.5657901763916016, -0.18421083688735962,
      0.19736847281455994, 0.5789477825164795, 0.9605270624160767),
    (-0.894055187702179, -0.5308968424797058, -0.1677384376525879,
      0.19541993737220764, 0.5585783123970032, 0.9217367172241211),
    (-0.8415402173995972, -0.49613019824028015, -0.15072016417980194,
      0.19468984007835388, 0.5400998592376709, 0.8855099081993103),
    (-0.7867394089698792, -0.4593363404273987, -0.13193318247795105,
      0.19546988606452942, 0.522873044013977, 0.8502761721611023),
    (-0.7368398904800415, -0.4246120750904083, -0.11238420754671097,
      0.1998436450958252, 0.5120715498924255, 0.8242993950843811),
    (-0.7077057957649231, -0.4009164869785309, -0.09412719309329987,
      0.21266208589076996, 0.5194514393806458, 0.8262407183647156),
    (-0.6777875423431396, -0.3728841245174408, -0.06798075139522552,
      0.23692262172698975, 0.5418260097503662, 0.8467293977737427));
  // B: schedule A with bh1. diffusers returns NaN on the last step (B(h) = -h
  // is -inf at sigma 0); step 6 is the limit x0 = x - sigma*v of step 5.
  cStepsB: array[0..6, 0..5] of double = (
    (-0.9473694562911987, -0.5657901763916016, -0.18421083688735962,
      0.19736847281455994, 0.5789477825164795, 0.9605270624160767),
    (-0.9238690137863159, -0.5496666431427002, -0.17546434700489044,
      0.1987379640340805, 0.5729402303695679, 0.9471425414085388),
    (-0.8687102794647217, -0.5132340788841248, -0.15775780379772186,
      0.19771847128868103, 0.5531946420669556, 0.908670961856842),
    (-0.8121171593666077, -0.4753067195415497, -0.1384962797164917,
      0.19831416010856628, 0.5351245403289795, 0.8719350099563599),
    (-0.7614530920982361, -0.4400811791419983, -0.11870924383401871,
      0.20266267657279968, 0.5240345597267151, 0.8454064726829529),
    (-0.7407553791999817, -0.4215664863586426, -0.10237767547369003,
      0.2168111652135849, 0.5359999537467957, 0.85518878698349),
    (-0.7016566395759583, -0.387798011302948, -0.07393942773342133,
      0.23991917073726654, 0.5537777543067932, 0.8676363229751587));
  // C: schedule A, bh2, disable_corrector on every step (UniP only).
  cStepsC: array[0..6, 0..5] of double = (
    (-0.9473694562911987, -0.5657901763916016, -0.18421083688735962,
      0.19736847281455994, 0.5789477825164795, 0.9605270624160767),
    (-0.8910644054412842, -0.5290139317512512, -0.1669633984565735,
      0.19508709013462067, 0.5571375489234924, 0.9191880226135254),
    (-0.8349348902702332, -0.4919765293598175, -0.1490180641412735,
      0.19394034147262573, 0.536898672580719, 0.879857063293457),
    (-0.7793577313423157, -0.45470312237739563, -0.13004839420318604,
      0.1946062445640564, 0.5192608833312988, 0.8439154624938965),
    (-0.7297177314758301, -0.42015835642814636, -0.1105988472700119,
      0.1989605724811554, 0.5085200071334839, 0.8180793523788452),
    (-0.7021526098251343, -0.39747434854507446, -0.09279599785804749,
      0.21188226342201233, 0.5165605545043945, 0.8212388157844543),
    (-0.6737768650054932, -0.3703981339931488, -0.06701932847499847,
      0.23635941743850708, 0.5397381782531738, 0.8431169390678406));
  // D: Create(1000, 1.0, true, ftsExponential, 0.02), SetTimesteps(6, mu 1.1),
  // bh2, entered at step 1 (img2img): step 1 is first order, no corrector.
  cSigmasD: array[0..6] of double = (1.0, 0.9020764511595901,
    0.7761497985579633, 0.6082039227865721, 0.372995806619427,
    0.020000000000000018, 0.0);
  cStepsD: array[1..5, 0..5] of double = (
    (-0.8864045739173889, -0.5255464315414429, -0.16468825936317444,
      0.196169912815094, 0.5570281147956848, 0.9178861975669861),
    (-0.8106069564819336, -0.47391602396965027, -0.13722506165504456,
      0.19946591556072235, 0.536156952381134, 0.8728477954864502),
    (-0.7580286264419556, -0.43486616015434265, -0.11170361936092377,
      0.21145889163017273, 0.5346214771270752, 0.8577839136123657),
    (-0.9156957864761353, -0.5218769907951355, -0.12805809080600739,
      0.2657608389854431, 0.6595797538757324, 1.053398609161377),
    (-0.9161725640296936, -0.5214494466781616, -0.12672628462314606,
      0.2679969370365143, 0.6627201437950134, 1.0574432611465454));
var
  Sched: TNNetFlowMatchEulerScheduler;
  Sample, Velocity: TNNetVolume;
  StepPos: integer;

  procedure StartSample;
  var
    ElementPos: integer;
  begin
    for ElementPos := 0 to 5 do
      Sample.FData[ElementPos] := (ElementPos - 2.5) * 0.4;
  end;

  procedure RunStep(StepIdx: integer; const Expected: array of double;
    const CaseName: string);
  var
    ElementPos: integer;
  begin
    FlowSyntheticVelocity(Sample, Velocity, Sched.Sigma[StepIdx]);
    Sched.Step(Sample, Velocity, StepIdx);
    for ElementPos := 0 to 5 do
      AssertEquals(CaseName + ' step ' + IntToStr(StepIdx) + ' @ ' +
        IntToStr(ElementPos), Expected[ElementPos], Sample.FData[ElementPos],
        2e-5);
  end;

begin
  Sched := nil;
  Sample := TNNetVolume.Create(6, 1, 1);
  Velocity := TNNetVolume.Create;
  try
    Sched := TNNetFlowMatchEulerScheduler.Create(1000, 3.0);
    Sched.Sampler := fsUniPC;
    AssertTrue('bh2 is the default', Sched.UniPCSolverType = ustBh2);
    AssertTrue('corrector on by default', Sched.UniPCCorrector);
    Sched.SetTimesteps(7);
    for StepPos := 0 to 7 do
      AssertEquals('sigma A ' + IntToStr(StepPos), cSigmasA[StepPos],
        Sched.Sigma[StepPos], 1e-12);
    StartSample;
    for StepPos := 0 to 6 do RunStep(StepPos, cStepsA[StepPos], 'A');
    Sched.UniPCSolverType := ustBh1;
    Sched.ResetMultistep;
    StartSample;
    for StepPos := 0 to 6 do RunStep(StepPos, cStepsB[StepPos], 'B');
    Sched.UniPCSolverType := ustBh2;
    Sched.UniPCCorrector := false;
    Sched.ResetMultistep;
    StartSample;
    for StepPos := 0 to 6 do RunStep(StepPos, cStepsC[StepPos], 'C');
    Sched.Free;
    Sched := TNNetFlowMatchEulerScheduler.Create(1000, 1.0, true,
      ftsExponential, 0.02);
    Sched.Sampler := fsUniPC;
    Sched.SetTimesteps(6, 1.1);
    for StepPos := 0 to 6 do
      AssertEquals('sigma D ' + IntToStr(StepPos), cSigmasD[StepPos],
        Sched.Sigma[StepPos], 1e-12);
    StartSample;
    for StepPos := 1 to 5 do RunStep(StepPos, cStepsD[StepPos], 'D');
  finally
    Sched.Free;
    Sample.Free;
    Velocity.Free;
  end;
end;

procedure TTestNeuralDiffusion.TestTrailingSpacingMatchesDiffusers;
const
  // diffusers DDIMScheduler(1000, timestep_spacing="trailing") gives
  // [999, 749, 499, 249] for N=4 and [999, 666, 332] for N=3; Tt = t + 1 here.
  cTrailing4: array[0..3] of integer = (1000, 750, 500, 250);
  cTrailing3: array[0..2] of integer = (1000, 667, 333);
var
  Sched: TNNetDiffusionScheduler;
  X: TNNetVolume;
  k: integer;
begin
  Sched := TNNetDiffusionScheduler.Create(1000, dsScaledLinear, dpV,
    0.00085, 0.012);
  X := TNNetVolume.Create(cN, 1, 1);
  try
    FRecN := 0;
    Sched.Sample(X, @RecordingModel, 4, smDDIM, 0.0, tsTrailing);
    AssertEquals('N=4 call count', 4, FRecN);
    for k := 0 to 3 do
      AssertEquals('trailing N=4 @ ' + IntToStr(k), cTrailing4[k], FRecT[k]);
    FRecN := 0;
    Sched.Sample(X, @RecordingModel, 3, smDDIM, 0.0, tsTrailing);
    AssertEquals('N=3 call count', 3, FRecN);
    for k := 0 to 2 do
      AssertEquals('trailing N=3 @ ' + IntToStr(k), cTrailing3[k], FRecT[k]);
  finally
    Sched.Free; X.Free;
  end;
end;

procedure TTestNeuralDiffusion.TestZeroTerminalSNRRescale;
const
  cTZ = 1000;
var
  Base, Zero: TNNetDiffusionScheduler;
  t: integer;
  sqrtFirst, sqrtLast, expected: double;
begin
  // Marigold's scheduler_config: scaled_linear 0.00085..0.012, rescaled.
  Base := TNNetDiffusionScheduler.Create(cTZ, dsScaledLinear, dpV,
    0.00085, 0.012);
  Zero := TNNetDiffusionScheduler.Create(cTZ, dsScaledLinear, dpV,
    0.00085, 0.012, 0.008, true);
  try
    AssertTrue('flag', Zero.RescaleZeroTerminalSNR);
    AssertEquals('ab_T is exactly 0', 0.0, Zero.AlphaBar[cTZ], 0.0);
    AssertEquals('beta_T is 1', 1.0, Zero.Beta[cTZ], 1e-7);
    AssertEquals('ab_1 unchanged', Base.AlphaBar[1], Zero.AlphaBar[1], 1e-7);
    AssertEquals('beta_1 unchanged', Base.Beta[1], Zero.Beta[1], 1e-7);
    // Algorithm 1 applied by hand to the unrescaled table.
    sqrtFirst := Sqrt(Base.AlphaBar[1]);
    sqrtLast := Sqrt(Base.AlphaBar[cTZ]);
    for t := 1 to cTZ do
    begin
      expected := Sqr((Sqrt(Base.AlphaBar[t]) - sqrtLast) * sqrtFirst /
        (sqrtFirst - sqrtLast));
      AssertEquals('rescaled ab @ ' + IntToStr(t), expected, Zero.AlphaBar[t], 2e-6);
      AssertTrue('non-increasing @ ' + IntToStr(t),
        Zero.AlphaBar[t] <= Zero.AlphaBar[t - 1]);
    end;
    // Reciprocal-based quantities stay finite at ab_T = 0.
    AssertFalse('sigma_T finite', IsNan(Zero.SigmaAt[cTZ]) or IsInfinite(Zero.SigmaAt[cTZ]));
    AssertTrue('sigma_T above sigma_{T-1}', Zero.SigmaAt[cTZ] > Zero.SigmaAt[cTZ - 1]);
    AssertEquals('v SNR weight at SNR 0', 0.0, Zero.SNRWeight(cTZ), 1e-7);
  finally
    Base.Free; Zero.Free;
  end;
end;

procedure TTestNeuralDiffusion.TestZeroTerminalSNRDDIMVEndpoint;
const
  cTZ = 1000;
  cTPrev = 750;
var
  Sched: TNNetDiffusionScheduler;
  Xt, V, Got: TNNetVolume;
  i: integer;
  sqrtAbPrev, somabPrev: double;
begin
  // At ab_T = 0 the v identities give x0 = -v and eps = x_t, so DDIM (eta 0)
  // must land on sqrt(ab_prev)*(-v) + sqrt(1-ab_prev)*x_t.
  Sched := TNNetDiffusionScheduler.Create(cTZ, dsScaledLinear, dpV,
    0.00085, 0.012, 0.008, true);
  Xt := TNNetVolume.Create(cN, 1, 1);
  V := TNNetVolume.Create(cN, 1, 1);
  Got := TNNetVolume.Create(cN, 1, 1);
  try
    for i := 0 to cN - 1 do
    begin
      Xt.FData[i] := 0.25 * i - 0.9;
      V.FData[i] := 0.4 * Cos(i);
    end;
    Sched.PredictX0(Xt, V, Got, cTZ);
    for i := 0 to cN - 1 do
      AssertEquals('x0 = -v @ ' + IntToStr(i), -V.FData[i], Got.FData[i], 1e-7);
    sqrtAbPrev := Sqrt(Sched.AlphaBar[cTPrev]);
    somabPrev := Sqrt(1.0 - Sched.AlphaBar[cTPrev]);
    Got.Copy(Xt);
    Sched.Step(Got, V, cTZ, cTPrev, smDDIM, 0.0);
    for i := 0 to cN - 1 do
      AssertEquals('DDIM endpoint @ ' + IntToStr(i),
        -sqrtAbPrev * V.FData[i] + somabPrev * Xt.FData[i], Got.FData[i], 1e-6);
  finally
    Sched.Free; Xt.Free; V.Free; Got.Free;
  end;
end;

procedure TTestNeuralDiffusion.TestZeroTerminalSNRDDPMVEndpoint;
const
  cTZ = 1000;
  cSeed = 4242;
var
  Sched: TNNetDiffusionScheduler;
  Xt, V, Got: TNNetVolume;
  Z: array[0..cN - 1] of double;
  i: integer;
  sqrtAbPrev, sigma: double;
begin
  // At ab_T = 0, x0 = -v and alpha_T = 0, so the DDPM posterior mean is
  // sqrt(ab_{T-1})*(-v); the step adds sqrt(beta_T)*z with the seeded z.
  Sched := TNNetDiffusionScheduler.Create(cTZ, dsScaledLinear, dpV,
    0.00085, 0.012, 0.008, true);
  Xt := TNNetVolume.Create(cN, 1, 1);
  V := TNNetVolume.Create(cN, 1, 1);
  Got := TNNetVolume.Create(cN, 1, 1);
  try
    for i := 0 to cN - 1 do
    begin
      Xt.FData[i] := 0.25 * i - 0.9;
      V.FData[i] := 0.4 * Cos(i);
    end;
    RandSeed := cSeed;
    for i := 0 to cN - 1 do Z[i] := RandG(0, 1);
    sqrtAbPrev := Sqrt(Sched.AlphaBar[cTZ - 1]);
    sigma := Sqrt(Sched.Beta[cTZ]);
    Got.Copy(Xt);
    RandSeed := cSeed;
    Sched.Step(Got, V, cTZ, cTZ - 1, smDDPM, 0.0);
    // sqrt(ab_{T-1}) is ~1e-3, so check the x0 the step used, noise removed.
    for i := 0 to cN - 1 do
      AssertEquals('DDPM endpoint x0 = -v @ ' + IntToStr(i), -V.FData[i],
        (Got.FData[i] - sigma * Z[i]) / sqrtAbPrev, 1e-2);
  finally
    Sched.Free; Xt.Free; V.Free; Got.Free;
  end;
end;

procedure TTestNeuralDiffusion.TestDDPMFirstStepPosteriorMean;
var
  Sched: TNNetDiffusionScheduler;
  Xt, Raw, Got: TNNetVolume;
  Prediction: TNNetPredictionType;
  i: integer;
  ab, Expected: double;
begin
  Xt := TNNetVolume.Create(cN, 1, 1);
  Raw := TNNetVolume.Create(cN, 1, 1);
  Got := TNNetVolume.Create(cN, 1, 1);
  try
    for i := 0 to cN - 1 do
    begin
      Xt.FData[i] := 0.3 * i - 1.1;
      Raw.FData[i] := 0.5 * Sin(i + 1);
    end;
    for Prediction := Low(TNNetPredictionType) to High(TNNetPredictionType) do
    begin
      // ab_0 = 1 makes the posterior mean of the last step equal x0 (diffusers
      // DDPMScheduler at t = 0), with no noise added.
      Sched := TNNetDiffusionScheduler.Create(cT, dsLinear, Prediction,
        cBeta1, cBetaT);
      try
        ab := Sched.AlphaBar[1];
        Got.Copy(Xt);
        Sched.Step(Got, Raw, 1, 0, smDDPM, 0.0);
        for i := 0 to cN - 1 do
        begin
          if Prediction = dpV then
            Expected := Sqrt(ab) * Xt.FData[i] - Sqrt(1.0 - ab) * Raw.FData[i]
          else
            Expected := (Xt.FData[i] - Sqrt(1.0 - ab) * Raw.FData[i]) / Sqrt(ab);
          AssertEquals('t=1 x0 @ ' + IntToStr(i), Expected, Got.FData[i], 1e-5);
        end;
      finally
        Sched.Free;
      end;
      // Beta1 = 0 gives ab_1 = 1: x_1 is already clean and must pass through.
      Sched := TNNetDiffusionScheduler.Create(cT, dsLinear, Prediction,
        0.0, cBetaT);
      try
        Got.Copy(Xt);
        Sched.Step(Got, Raw, 1, 0, smDDPM, 0.0);
        for i := 0 to cN - 1 do
          AssertEquals('Beta1=0 @ ' + IntToStr(i), Xt.FData[i], Got.FData[i], 1e-6);
      finally
        Sched.Free;
      end;
    end;
  finally
    Xt.Free; Raw.Free; Got.Free;
  end;
end;

procedure TTestNeuralDiffusion.TestSchedulerFromDiffusersConfig;
const
  // prs-eth/marigold-depth-v1-1 scheduler/scheduler_config.json.
  csMarigoldConfig = '{"_class_name": "DDIMScheduler", "beta_end": 0.012, ' +
    '"beta_schedule": "scaled_linear", "beta_start": 0.00085, ' +
    '"clip_sample": false, "num_train_timesteps": 1000, ' +
    '"prediction_type": "v_prediction", "rescale_betas_zero_snr": true, ' +
    '"set_alpha_to_one": false, "steps_offset": 1, "thresholding": false, ' +
    '"timestep_spacing": "trailing"}';
  csSD15Config = '{"_class_name": "PNDMScheduler", "beta_end": 0.012, ' +
    '"beta_schedule": "scaled_linear", "beta_start": 0.00085, ' +
    '"num_train_timesteps": 1000, "set_alpha_to_one": false, ' +
    '"skip_prk_steps": true, "steps_offset": 1, "trained_betas": null}';
var
  FileName: string;
  Lines: TStringList;
  FromConfig, Reference: TNNetDiffusionScheduler;
  Schedule: TNeuralIntegerArray;
  Tt: integer;

  function Refused(const Json: string): boolean;
  var
    Probe: TNNetDiffusionScheduler;
  begin
    Lines.Text := Json;
    Lines.SaveToFile(FileName);
    Result := false;
    Probe := nil;
    try
      try
        Probe := TNNetDiffusionScheduler.CreateFromDiffusersConfig(FileName);
      except
        Result := true;
      end;
    finally
      Probe.Free;
    end;
  end;

begin
  FileName := GetTempDir(false) + 'cai_ddim_scheduler_config.json';
  Lines := TStringList.Create;
  FromConfig := nil;
  Reference := TNNetDiffusionScheduler.Create(1000, dsScaledLinear, dpV,
    0.00085, 0.012, 0.008, true, false);
  try
    Lines.Text := csMarigoldConfig;
    Lines.SaveToFile(FileName);
    FromConfig := TNNetDiffusionScheduler.CreateFromDiffusersConfig(FileName);
    AssertEquals('T', 1000, FromConfig.NumTimesteps);
    AssertTrue('v prediction', FromConfig.PredictionType = dpV);
    AssertTrue('zero SNR', FromConfig.RescaleZeroTerminalSNR);
    AssertEquals('final alpha bar = ab_1', Reference.AlphaBar[1],
      FromConfig.FinalAlphaBar, 0);
    for Tt := 0 to 1000 do
      AssertEquals('alpha bar @ ' + IntToStr(Tt), Reference.AlphaBar[Tt],
        FromConfig.AlphaBar[Tt], 0);
    AssertTrue('clip_sample defaults to true',
      Refused('{"timestep_spacing": "trailing"}'));
    AssertFalse('linspace spacing loads', Refused('{"clip_sample": false, ' +
      '"timestep_spacing": "linspace"}'));
    AssertTrue('unknown spacing', Refused('{"clip_sample": false, ' +
      '"timestep_spacing": "karras"}'));
    AssertTrue('sample prediction', Refused('{"clip_sample": false, ' +
      '"timestep_spacing": "trailing", "prediction_type": "sample"}'));
    AssertTrue('trained betas', Refused('{"clip_sample": false, ' +
      '"timestep_spacing": "trailing", "trained_betas": [0.1, 0.2]}'));
    AssertFalse('null trained betas load', Refused('{"clip_sample": false, ' +
      '"timestep_spacing": "trailing", "trained_betas": null}'));
    // runwayml/stable-diffusion-v1-5: PNDM (no clip_sample default), leading
    // spacing, steps_offset 1; diffusers 3 steps are t = 667, 334, 1.
    FromConfig.Free;
    FromConfig := nil;
    Lines.Text := csSD15Config;
    Lines.SaveToFile(FileName);
    FromConfig := TNNetDiffusionScheduler.CreateFromDiffusersConfig(FileName);
    AssertTrue('SD 1.5 leading', FromConfig.ConfigSpacing = tsLeading);
    AssertEquals('SD 1.5 steps_offset', 1, FromConfig.StepsOffset);
    AssertTrue('SD 1.5 eps', FromConfig.PredictionType = dpEps);
    Schedule := FromConfig.BuildTimestepSchedule(3, tsLeading);
    AssertEquals('leading length', 4, Length(Schedule));
    AssertEquals('leading 0', 668, Schedule[0]);
    AssertEquals('leading 1', 335, Schedule[1]);
    AssertEquals('leading 2', 2, Schedule[2]);
    AssertEquals('leading end', 0, Schedule[3]);
    AssertTrue('PNDM without clip_sample loads', not Refused(
      '{"_class_name": "PNDMScheduler"}'));
  finally
    FromConfig.Free;
    Reference.Free;
    Lines.Free;
    DeleteFile(FileName);
  end;
end;

procedure TTestNeuralDiffusion.TestZeroTerminalSNRTrailingSamplersFinite;
var
  Sched: TNNetDiffusionScheduler;
  X: TNNetVolume;
  Method: TNNetSamplerMethod;
  Prediction: TNNetPredictionType;
  i: integer;
  Eta: TNeuralFloat;
begin
  // Every sampler, both prediction types, starting at the ab_T = 0 endpoint:
  // no FP trap and a finite result. Eta 1 also covers the stochastic branches.
  for Prediction := Low(TNNetPredictionType) to High(TNNetPredictionType) do
  begin
    // TNNetLCMScheduler runs smLCM and hands every other method to the base.
    Sched := TNNetLCMScheduler.Create(1000, dsScaledLinear, Prediction,
      0.00085, 0.012, 0.008, true);
    X := TNNetVolume.Create(cN, 1, 1);
    try
      for Method := Low(TNNetSamplerMethod) to High(TNNetSamplerMethod) do
        for Eta in [0.0, 1.0] do
        begin
          for i := 0 to cN - 1 do X.FData[i] := (i - cN / 2) * 0.3;
          Sched.Sample(X, @ToyModel, 4, Method, Eta, tsTrailing);
          for i := 0 to cN - 1 do
            AssertFalse('finite method ' + IntToStr(Ord(Method)) + ' pred ' +
              IntToStr(Ord(Prediction)) + ' @ ' + IntToStr(i),
              IsNan(X.FData[i]) or IsInfinite(X.FData[i]));
        end;
    finally
      Sched.Free; X.Free;
    end;
  end;
end;

procedure TTestNeuralDiffusion.TestDDIMFinalAlphaBarVsDiffusers;
const
  // diffusers DDIMScheduler(scaled_linear 0.00085..0.012, v_prediction,
  // trailing, rescale_betas_zero_snr, eta 0), 4 steps, v = sin(0.01*(t+1))*x,
  // x = (i - 4)*0.3; set_alpha_to_one False then True.
  cNotOne: array[0..cN - 1] of double = (-0.530490551, -0.397867913,
    -0.265245276, -0.132622638, 0, 0.132622638, 0.265245276, 0.397867913);
  cOne: array[0..cN - 1] of double = (-0.496718779, -0.372539085,
    -0.24835939, -0.124179695, 0, 0.124179695, 0.24835939, 0.372539085);
var
  SchedNotOne, SchedOne: TNNetDiffusionScheduler;
  X: TNNetVolume;
  i: integer;
begin
  SchedNotOne := TNNetDiffusionScheduler.Create(1000, dsScaledLinear, dpV,
    0.00085, 0.012, 0.008, true, false);
  SchedOne := TNNetDiffusionScheduler.Create(1000, dsScaledLinear, dpV,
    0.00085, 0.012, 0.008, true);
  X := TNNetVolume.Create(cN, 1, 1);
  try
    AssertEquals('final ab = ab_1', SchedNotOne.AlphaBar[1],
      SchedNotOne.FinalAlphaBar, 0.0);
    AssertEquals('default final ab = 1', 1.0, SchedOne.FinalAlphaBar, 0.0);
    for i := 0 to cN - 1 do X.FData[i] := (i - cN / 2) * 0.3;
    SchedNotOne.Sample(X, @ToyModel, 4, smDDIM, 0.0, tsTrailing);
    for i := 0 to cN - 1 do
      AssertEquals('not-one @ ' + IntToStr(i), cNotOne[i], X.FData[i], 1e-5);
    for i := 0 to cN - 1 do X.FData[i] := (i - cN / 2) * 0.3;
    SchedOne.Sample(X, @ToyModel, 4, smDDIM, 0.0, tsTrailing);
    for i := 0 to cN - 1 do
      AssertEquals('one @ ' + IntToStr(i), cOne[i], X.FData[i], 1e-5);
  finally
    SchedNotOne.Free; SchedOne.Free; X.Free;
  end;
end;

procedure TTestNeuralDiffusion.TestZeroTerminalSNREulerVEndpoint;
const
  cTZ = 1000;
  cTPrev = 750;
var
  Sched: TNNetDiffusionScheduler;
  Xt, V, Got: TNNetVolume;
  i: integer;
  s1, s2, sqrtAbPrev, kA, kX0: double;
begin
  // Deterministic Euler from ab_T = 0 with v: x0 = -v exactly, and
  // x = sqrt(ab_prev)*((s2/s1)*x_t/sqrt(ab_T) + (1 - s2/s1)*x0), where the
  // floored 1/sqrt(ab_T) = sqrt(1 + s1^2).
  Sched := TNNetDiffusionScheduler.Create(cTZ, dsScaledLinear, dpV,
    0.00085, 0.012, 0.008, true);
  Xt := TNNetVolume.Create(cN, 1, 1);
  V := TNNetVolume.Create(cN, 1, 1);
  Got := TNNetVolume.Create(cN, 1, 1);
  try
    for i := 0 to cN - 1 do
    begin
      Xt.FData[i] := 0.25 * i - 0.9;
      V.FData[i] := 0.4 * Cos(i);
    end;
    Got.Copy(Xt);
    Sched.Step(Got, V, cTZ, 0, smEulerAncestral, 0.0);
    for i := 0 to cN - 1 do
      AssertEquals('final x0 = -v @ ' + IntToStr(i), -V.FData[i], Got.FData[i], 1e-7);
    s1 := Sched.SigmaAt[cTZ];
    s2 := Sched.SigmaAt[cTPrev];
    sqrtAbPrev := Sqrt(Sched.AlphaBar[cTPrev]);
    kA := sqrtAbPrev * (s2 / s1) * Sqrt(1.0 + s1 * s1);
    kX0 := sqrtAbPrev * (1.0 - s2 / s1);
    Got.Copy(Xt);
    Sched.Step(Got, V, cTZ, cTPrev, smEulerAncestral, 0.0);
    for i := 0 to cN - 1 do
      AssertEquals('Euler endpoint @ ' + IntToStr(i),
        kA * Xt.FData[i] - kX0 * V.FData[i], Got.FData[i], 1e-5);
  finally
    Sched.Free; Xt.Free; V.Free; Got.Free;
  end;
end;

initialization
  RegisterTest(TTestNeuralDiffusion);
end.
