unit neuraldiffusion;
(*
neuraldiffusion: a reusable, CPU-only, MODEL-AGNOSTIC reverse-process sampler /
noise-scheduler for denoising diffusion models. It centralises the beta/alpha
bookkeeping and the standard reverse-process update rules that the diffusion
EXAMPLES (examples/DiffusionMNIST, examples/FlowMatching) previously hand-rolled
inline.

WHAT IT PROVIDES
  * A configurable BETA SCHEDULE (linear / scaled-linear / cosine) over T steps,
    precomputing beta_t, alpha_t = 1-beta_t and the cumulative product
    alpha_bar_t = prod_{s<=Tt} alpha_s (plus the sqrt() tables the updates need).
    Convention matches the existing examples: 1-BASED timesteps 1..T, with index
    0 reserved as the "clean image" anchor (alpha_bar_0 = 1).
  * The forward process q_sample (AddNoise): x_t = sqrt(ab_t)*x0 + sqrt(1-ab_t)*eps.
  * The REVERSE updates, each one step:
      - DDPM ancestral (stochastic),
      - DDIM (deterministic for eta=0; eta>0 adds the calibrated stochastic term),
      - DPM-Solver++(2M) (a 2nd-order multistep update that keeps the previous
        model output for its correction term),
      - Euler-ancestral ("Euler a", Karras 2022): deterministic Euler drift in
        the sigma parameterisation plus per-step ancestral noise (Eta scales the
        injection; Eta=0 reduces exactly to deterministic Euler == DDIM eta=0).
  * Two timestep SPACINGS for the Sample() driver: the original uniform stride
    and Karras et al. (2022) rho=7 sigma spacing (sigma_t = sqrt((1-ab_t)/ab_t),
    geometrically warped between sigma_min/sigma_max, snapped back to timesteps).
  * PREDICTION TYPES eps and v. A v-prediction model output is converted to the
    equivalent eps internally so all the updates share one code path.
  * A classifier-free-guidance (CFG) MIXER: given eps_cond and eps_uncond and a
    guidance weight w, returns eps_uncond + w*(eps_cond - eps_uncond).

MODEL-AGNOSTIC. The denoiser is supplied by the CALLER as a Pascal
procedure-of-object (TNNetDenoiseCallback): given a noisy volume x_t and an
integer timestep it must fill an output volume with the model's prediction
(eps or v). The scheduler never touches a TNNet, so any predictor -- a network's
Compute, a CFG wrapper, a closure over class labels -- plugs straight in. The
high-level Sample() driver runs a full reverse trajectory over a strided
subsequence of timesteps using the callback and the chosen method.

All math is in TNeuralFloat (single) for the volumes, but the schedule tables
are built in double precision and then stored, so the per-step coefficients
match a numpy float64 oracle to single-precision tolerance.

Copyright (C) 2026 Joao Paulo Schwarz Schuler

This library is free software; you can redistribute it and/or modify it under
the terms of the GNU Lesser General Public License as published by the Free
Software Foundation; either version 2 of the License, or (at your option) any
later version.

Coded by Claude (AI).
*)

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, Math, neuralvolume;

type
  // Beta schedule families.
  //   dsLinear       : beta linear in [Beta1 .. BetaT]                (Ho 2020)
  //   dsScaledLinear : sqrt(beta) linear in [sqrt(Beta1)..sqrt(BetaT)] (LDM/SD)
  //   dsCosine       : alpha_bar = cos^2 cosine schedule              (Nichol 2021)
  TNNetBetaSchedule = (dsLinear, dsScaledLinear, dsCosine);

  // What the supplied model predicts.
  //   dpEps : the added noise epsilon (the classic DDPM target).
  //   dpV   : the "velocity" v = sqrt(ab)*eps - sqrt(1-ab)*x0 (Salimans 2022).
  TNNetPredictionType = (dpEps, dpV);

  // Reverse-process update family.
  //   smEulerAncestral : deterministic Euler drift in sigma-space + per-step
  //     ancestral Gaussian noise injection (Karras et al. 2022 "Euler a").
  //   smUniPC : UniPC (UniPCMultistepScheduler, Zhao et al. 2023,
  //     arXiv:2302.04867). A unified PREDICTOR-CORRECTOR ODE solver of order
  //     B(h). Unlike the DPM-Solver++(2M) predictor it REUSES the previous
  //     step's model output a SECOND time as a free corrector before the
  //     predictor, giving noticeably better quality at very low step counts
  //     (5-10). This implements the order-2 bh2 variant (predict_x0=True,
  //     thresholding=False, lower_order_final=True). Shares the same
  //     lambda/sigma/alpha schedule plumbing and prev-output history that
  //     DPM-Solver++(2M) keeps, plus the previous sample for the corrector.
  //   smHeun : Heun 2nd-order deterministic sampler (k-diffusion sample_heun /
  //     Karras et al. 2022 EDM "Algorithm 2"). A genuine intra-step
  //     predictor-corrector in sigma-space costing TWO denoiser evals per step:
  //     an Euler predict from sigma_t to sigma_{t-1}, a SECOND denoiser eval at
  //     the predicted point, then the trapezoidal average of the two drifts.
  //     Unlike the MULTISTEP DPM-Solver++(2M)/UniPC (which reuse the PREVIOUS
  //     step's stored output, 1 eval/step), Heun re-runs the model inside the
  //     step, so the Sample() driver calls Denoise TWICE per step and SKIPS the
  //     corrector on the final step (sigma_{t-1} = 0), matching k-diffusion's
  //     `if sigmas[i+1] == 0` branch. Pairs naturally with tsKarras spacing.
  TNNetSamplerMethod = (smDDPM, smDDIM, smDPMSolverPP2M, smEulerAncestral,
    smUniPC, smHeun);

  // Timestep SPACING for the Sample() driver.
  //   tsUniform : evenly strided timesteps in 1..T (the original behaviour).
  //   tsKarras  : Karras et al. (2022) rho=7 sigma spacing. The sampling sigmas
  //     are placed geometrically-warped between sigma_min and sigma_max and then
  //     mapped back to the nearest schedule timestep. Pairs naturally with the
  //     Euler-ancestral sampler but works with any method.
  TNNetTimestepSpacing = (tsUniform, tsKarras);

  // Caller-supplied denoiser. Given noisy Xt at integer timestep Tt (1..T) it
  // must write the model prediction (eps or v, per the scheduler's
  // PredictionType) into Output. Output is pre-sized to match Xt.
  TNNetDenoiseCallback = procedure(Xt, Output: TNNetVolume; Tt: integer) of object;

  { TNNetDiffusionScheduler }
  // Reusable noise scheduler + reverse-process sampler. Holds the precomputed
  // schedule tables for T steps and a chosen beta-schedule family.
  // Coded by Claude (AI).
  TNNetDiffusionScheduler = class(TObject)
  private
    FT: integer;
    FSchedule: TNNetBetaSchedule;
    FPrediction: TNNetPredictionType;
    // Tables indexed 0..T. Index 0 is the clean-image anchor.
    FBeta, FAlpha, FAlphaBar: array of TNeuralFloat;
    FSqrtAlphaBar, FSqrtOneMinusAlphaBar: array of TNeuralFloat;
    // DPM-Solver++ multistep state (previous step's converted x0 prediction).
    FHasPrev: boolean;
    FPrevX0: TNNetVolume;
    FPrevLambda: TNeuralFloat;
    // UniPC predictor-corrector extra state. FUniPrevT is the integer timestep
    // of the previous step (s0 of the corrector), FUniPrevSample is that step's
    // (corrected) output sample x_{s0}, FUniLowerOrderNums counts how many
    // model outputs have accumulated (caps the per-step order during warm-up),
    // and FUniThisOrder is the order the PREVIOUS predictor used (= the order
    // the current corrector must use, matching diffusers' self.this_order).
    FUniThisOrder: integer;
    FUniLowerOrderNums: integer;
    FUniPrevSample: TNNetVolume;
    // Two-deep UniPC model-output history (x0 predictions + their timesteps),
    // newest at index 1. FUniHistLen counts valid entries (0..2).
    FUniHistX0: array[0..1] of TNNetVolume;
    FUniHistT: array[0..1] of integer;
    FUniHistLen: integer;
    // Persistent, lazily-sized eps/x0 scratch buffers (rule #17). Amortized
    // resize replaces the per-step TNNetVolume.Create/Free in Step/PredictX0/
    // LCMStep. FEpsScratch is shared by Step and PredictX0 (never nested);
    // FX0HatScratch backs LCMStep's x0_hat.
    FEpsScratch: TNNetVolume;
    FX0HatScratch: TNNetVolume;
    procedure BuildTables(Beta1, BetaT, CosineS: TNeuralFloat);
    function GetBeta(Tt: integer): TNeuralFloat;
    function GetAlpha(Tt: integer): TNeuralFloat;
    function GetAlphaBar(Tt: integer): TNeuralFloat;
    // log(sqrt(ab/(1-ab))) half-log-SNR lambda used by DPM-Solver++.
    function Lambda(Tt: integer): TNeuralFloat;
    // Karras sigma_t = sqrt((1-ab_t)/ab_t) for timestep Tt (the per-step noise
    // level in the variance-exploding parameterisation Euler/Karras use).
    function SigmaOf(Tt: integer): TNeuralFloat;
    // Map a (continuous) target sigma back to the nearest schedule timestep in
    // 1..T whose Sigma() is closest. Used by the Karras spacing.
    function SigmaToTimestep(TargetSigma: TNeuralFloat): integer;
    // Build a list of NumSteps+1 timesteps (descending, last entry 0 = clean
    // image) using the requested spacing. Caller-owned dynamic array.
    function BuildTimestepSchedule(NumSteps: integer;
      Spacing: TNNetTimestepSpacing): TNeuralIntegerArray;
    // Heun 2nd-order trajectory (smHeun). Separated from Sample() because it is
    // the only method that calls Denoise TWICE per step (a genuine intra-step
    // second derivative), so it cannot share the single-eval Step() dispatch.
    procedure SampleHeun(X: TNNetVolume; Denoise: TNNetDenoiseCallback;
      NumSteps: integer; Spacing: TNNetTimestepSpacing);
  protected
    // Convert a raw model prediction (eps or v) at timestep Tt into eps.
    procedure ToEps(Xt, RawPred, EpsOut: TNNetVolume; Tt: integer);
    // Convert a raw model prediction at Tt into the predicted clean image x0:
    //   x0 = (x_t - sqrt(1-ab_t)*eps)/sqrt(ab_t),  eps = ToEps(raw).
    // The subclasses (LCM) reuse this to share one parameterization with the
    // iterative samplers (eps/v plumbing). X0Out may alias RawPred.
    procedure PredictX0(Xt, RawPred, X0Out: TNNetVolume; Tt: integer);
  public
    // T = number of diffusion steps. Beta1/BetaT used by linear & scaled-linear;
    // CosineS is the small offset of the cosine schedule (default 0.008).
    constructor Create(pT: integer; pSchedule: TNNetBetaSchedule = dsLinear;
      pPrediction: TNNetPredictionType = dpEps;
      Beta1: TNeuralFloat = 1.0e-4; BetaT: TNeuralFloat = 0.02;
      CosineS: TNeuralFloat = 0.008);
    destructor Destroy; override;

    // Schedule tables (1-based timesteps; index 0 = clean anchor).
    property Beta[Tt: integer]: TNeuralFloat read GetBeta;
    property Alpha[Tt: integer]: TNeuralFloat read GetAlpha;
    property AlphaBar[Tt: integer]: TNeuralFloat read GetAlphaBar;
    property NumTimesteps: integer read FT;
    property PredictionType: TNNetPredictionType read FPrediction;

    // FORWARD process. Xt := sqrt(ab_t)*X0 + sqrt(1-ab_t)*Noise. If NoiseOut is
    // given it receives the sampled noise (the eps training target). When
    // PreSampledNoise is supplied that noise is used (and copied to NoiseOut)
    // instead of drawing fresh Gaussian samples -- handy for deterministic tests.
    procedure AddNoise(X0, Xt: TNNetVolume; Tt: integer;
      NoiseOut: TNNetVolume = nil; PreSampledNoise: TNNetVolume = nil);

    // Classifier-free guidance mix into Dst:
    //   Dst := EpsUncond + W*(EpsCond - EpsUncond).
    // Dst may alias EpsCond or EpsUncond.
    class procedure ApplyCFG(EpsCond, EpsUncond, Dst: TNNetVolume;
      W: TNeuralFloat); overload;
    // Same CFG mix, then the guidance-rescale of Lin et al. 2023 ("Common
    // Diffusion Noise Schedules and Sample Steps Are Flawed"), i.e. diffusers'
    // rescale_noise_cfg. Let cfg := EpsUncond + W*(EpsCond - EpsUncond); rescale
    // it by std(EpsCond)/std(cfg) and blend back: Dst := GuidanceRescale*rescaled
    // + (1-GuidanceRescale)*cfg. GuidanceRescale=0 reproduces the plain CFG mix
    // exactly. std is the population std over the whole volume.
    class procedure ApplyCFG(EpsCond, EpsUncond, Dst: TNNetVolume;
      W, GuidanceRescale: TNeuralFloat); overload;

    // Reset the DPM-Solver++ multistep history. Call before each new trajectory.
    procedure ResetMultistep;

    // ONE reverse step from timestep Tt to TtPrev (TtPrev=0 -> clean image), in
    // place on Xt. RawPred is the model's raw prediction (eps or v) at Tt.
    // Method selects the update rule; Eta is the DDIM stochasticity (0 = the
    // deterministic DDIM used by the examples; ignored by DDPM/DPM++). For DDPM
    // TtPrev is ignored (it uses Tt-1 by construction). For smEulerAncestral Eta
    // scales the injected ancestral noise (Eta=1 = the standard "Euler a";
    // Eta=0 reduces it to a deterministic Euler step).
    procedure Step(Xt, RawPred: TNNetVolume; Tt, TtPrev: integer;
      Method: TNNetSamplerMethod = smDDIM; Eta: TNeuralFloat = 0.0);

    // Per-timestep noise level sigma_t = sqrt((1-ab_t)/ab_t) (Karras VE form).
    property SigmaAt[Tt: integer]: TNeuralFloat read SigmaOf;

    // Min-SNR-gamma per-timestep LOSS weight (Hang et al. 2023, "Efficient
    // Diffusion Training via Min-SNR Weighting Strategy"; diffusers snr_gamma).
    // SNR(t) = exp(2*Lambda(t)) = ab_t/(1-ab_t). The clamp min(SNR, Gamma)
    // down-weights the easy low-noise (high-SNR) timesteps. The denominator
    // depends on the scheduler's PredictionType (the model's target):
    //   eps-prediction : weight = min(SNR, Gamma) / SNR,
    //   v-prediction   : weight = min(SNR, Gamma) / (SNR + 1).
    // Gamma = 5.0 is the paper default; Gamma = +Inf reproduces the unweighted
    // objective (weight = 1 for eps, SNR/(SNR+1) for v -- the standard v-loss
    // reweighting, which is exactly diffusers' base case).
    function SNRWeight(Tt: integer; Gamma: TNeuralFloat = 5.0): TNeuralFloat;

    // High-level driver: run a complete reverse trajectory in place on X
    // (which should start as N(0,I) noise) over NumSteps evenly-strided
    // timesteps using the supplied Denoise callback. Leaves the sampled x0 in X.
    procedure Sample(X: TNNetVolume; Denoise: TNNetDenoiseCallback;
      NumSteps: integer; Method: TNNetSamplerMethod = smDDIM;
      Eta: TNeuralFloat = 0.0; Spacing: TNNetTimestepSpacing = tsUniform);
  end;

  { TNNetLCMScheduler }
  // Latent Consistency Model (LCM) few-step sampler (Luo et al. 2023,
  // arXiv:2310.04378). Unlike the iterative noise-prediction integrators (DDIM /
  // DPM++ / Euler) it does NOT integrate the probability-flow ODE step by step;
  // instead it evaluates a learned CONSISTENCY function f(x_t,t) that maps any
  // noised latent DIRECTLY toward the clean solution x0, parameterized with the
  // standard boundary scalings c_skip(t), c_out(t) (sigma_data=0.5):
  //   f(x,t) = c_skip(t)*x + c_out(t)*x0_hat,
  //   c_skip(t) = sigma_data^2 / ((t/0.1)^2 + sigma_data^2),
  //   c_out(t)  = (t/0.1)*sigma_data / sqrt((t/0.1)^2 + sigma_data^2),
  // where x0_hat is the model's x0 prediction (eps/v converted via the inherited
  // shared plumbing -- the LCM sampler consumes the SAME raw model output the
  // other samplers do). Generation is a SMALL number of steps (1-4): predict x0
  // via f, then re-noise x0 forward to the NEXT (smaller) timestep with the
  // inherited AddNoise using fresh Gaussian noise, repeat. Guidance is BAKED IN
  // (the distilled consistency model already absorbed CFG), so there is no
  // cond/uncond double pass per step. All schedule tables (alpha-bar/sigma) and
  // the timestep-schedule builder are inherited unchanged.
  // Coded by Claude (AI).
  TNNetLCMScheduler = class(TNNetDiffusionScheduler)
  public
    // Boundary scalings at (integer) timestep Tt with sigma_data=0.5. At the
    // largest Tt c_skip -> small and c_out -> ~sigma_data; at Tt->0 c_skip -> 1
    // and c_out -> 0 (the consistency boundary condition f(x,0)=x).
    function CSkip(Tt: integer): TNeuralFloat;
    function COut(Tt: integer): TNeuralFloat;
    // ONE consistency step: evaluate f(Xt,Tt) IN PLACE into Xt, writing the
    // consistency-mapped clean-image estimate x0_consistent. RawPred is the raw
    // model output (eps or v) at Tt. After this Xt holds the f-estimate of x0.
    procedure LCMStep(Xt, RawPred: TNNetVolume; Tt: integer);
    // High-level few-step driver. X starts as N(0,I) noise (text->image) or a
    // partially-noised latent. Runs NumSteps (typically 1-4) consistency steps
    // over the inherited timestep schedule: at each step predict x0 via f, and
    // (unless it is the last step) re-noise that x0 forward to the NEXT, smaller
    // timestep with fresh Gaussian noise. Leaves the sampled x0 in X.
    procedure LCMSample(X: TNNetVolume; Denoise: TNNetDenoiseCallback;
      NumSteps: integer = 4; Spacing: TNNetTimestepSpacing = tsUniform);
  end;

implementation

constructor TNNetDiffusionScheduler.Create(pT: integer;
  pSchedule: TNNetBetaSchedule; pPrediction: TNNetPredictionType;
  Beta1, BetaT, CosineS: TNeuralFloat);
begin
  inherited Create;
  if pT < 1 then raise Exception.Create('Diffusion T must be >= 1.');
  FT := pT;
  FSchedule := pSchedule;
  FPrediction := pPrediction;
  SetLength(FBeta, FT + 1);
  SetLength(FAlpha, FT + 1);
  SetLength(FAlphaBar, FT + 1);
  SetLength(FSqrtAlphaBar, FT + 1);
  SetLength(FSqrtOneMinusAlphaBar, FT + 1);
  BuildTables(Beta1, BetaT, CosineS);
  FPrevX0 := nil;
  FUniPrevSample := nil;
  FUniHistX0[0] := nil;
  FUniHistX0[1] := nil;
  FEpsScratch := nil;
  FX0HatScratch := nil;
  FHasPrev := false;
  FUniThisOrder := 1;
  FUniLowerOrderNums := 0;
  FUniHistLen := 0;
end;

destructor TNNetDiffusionScheduler.Destroy;
begin
  if Assigned(FPrevX0) then FPrevX0.Free;
  if Assigned(FUniPrevSample) then FUniPrevSample.Free;
  if Assigned(FUniHistX0[0]) then FUniHistX0[0].Free;
  if Assigned(FUniHistX0[1]) then FUniHistX0[1].Free;
  if Assigned(FEpsScratch) then FEpsScratch.Free;
  if Assigned(FX0HatScratch) then FX0HatScratch.Free;
  inherited Destroy;
end;

procedure TNNetDiffusionScheduler.BuildTables(Beta1, BetaT, CosineS: TNeuralFloat);
var
  i: integer;
  // Work in double precision so the tables match a numpy float64 oracle.
  bd, prod, sb1, sbT: double;
  abPrev, abCur, f0, fcur: double;
begin
  FBeta[0] := 0; FAlpha[0] := 1; FAlphaBar[0] := 1;
  FSqrtAlphaBar[0] := 1; FSqrtOneMinusAlphaBar[0] := 0;
  case FSchedule of
    dsCosine:
      begin
        // alpha_bar_t = f(Tt)/f(0), f(Tt) = cos^2( ((Tt/T + s)/(1+s)) * pi/2 ).
        // We build alpha_bar directly then recover beta from the ratio.
        f0 := Sqr(Cos((CosineS / (1.0 + CosineS)) * (PI / 2.0)));
        abPrev := 1.0;
        for i := 1 to FT do
        begin
          fcur := Sqr(Cos((((i / FT) + CosineS) / (1.0 + CosineS)) * (PI / 2.0)));
          abCur := fcur / f0;
          // beta_t = 1 - ab_t/ab_{Tt-1}, clipped for stability (Nichol 2021).
          bd := 1.0 - (abCur / abPrev);
          if bd > 0.999 then bd := 0.999;
          if bd < 0.0   then bd := 0.0;
          FBeta[i]  := bd;
          FAlpha[i] := 1.0 - bd;
          // Recompute alpha_bar from the clipped betas so all tables are consistent.
          FAlphaBar[i] := FAlphaBar[i - 1] * FAlpha[i];
          FSqrtAlphaBar[i] := Sqrt(FAlphaBar[i]);
          FSqrtOneMinusAlphaBar[i] := Sqrt(1.0 - FAlphaBar[i]);
          abPrev := abCur;
        end;
      end;
    dsScaledLinear:
      begin
        // sqrt(beta) linear between sqrt(Beta1) and sqrt(BetaT).
        sb1 := Sqrt(Beta1); sbT := Sqrt(BetaT);
        prod := 1.0;
        for i := 1 to FT do
        begin
          if FT = 1 then bd := Sqr(sb1)
          else bd := Sqr(sb1 + (sbT - sb1) * (i - 1) / (FT - 1));
          FBeta[i]  := bd;
          FAlpha[i] := 1.0 - bd;
          prod := prod * (1.0 - bd);
          FAlphaBar[i] := prod;
          FSqrtAlphaBar[i] := Sqrt(prod);
          FSqrtOneMinusAlphaBar[i] := Sqrt(1.0 - prod);
        end;
      end;
    else // dsLinear
      begin
        prod := 1.0;
        for i := 1 to FT do
        begin
          if FT = 1 then bd := Beta1
          else bd := Beta1 + (BetaT - Beta1) * (i - 1) / (FT - 1);
          FBeta[i]  := bd;
          FAlpha[i] := 1.0 - bd;
          prod := prod * (1.0 - bd);
          FAlphaBar[i] := prod;
          FSqrtAlphaBar[i] := Sqrt(prod);
          FSqrtOneMinusAlphaBar[i] := Sqrt(1.0 - prod);
        end;
      end;
  end;
end;

function TNNetDiffusionScheduler.GetBeta(Tt: integer): TNeuralFloat;
begin Result := FBeta[Tt]; end;

function TNNetDiffusionScheduler.GetAlpha(Tt: integer): TNeuralFloat;
begin Result := FAlpha[Tt]; end;

function TNNetDiffusionScheduler.GetAlphaBar(Tt: integer): TNeuralFloat;
begin Result := FAlphaBar[Tt]; end;

function TNNetDiffusionScheduler.Lambda(Tt: integer): TNeuralFloat;
var ab: double;
begin
  ab := FAlphaBar[Tt];
  // lambda = log( sqrt(ab) / sqrt(1-ab) ) = 0.5*(log ab - log(1-ab)).
  Result := 0.5 * (Ln(ab) - Ln(1.0 - ab));
end;

function TNNetDiffusionScheduler.SigmaOf(Tt: integer): TNeuralFloat;
var ab: double;
begin
  // sigma_t = sqrt((1-ab_t)/ab_t). At Tt=0 (clean anchor) this is 0.
  ab := FAlphaBar[Tt];
  if ab >= 1.0 then Result := 0
  else Result := Sqrt((1.0 - ab) / ab);
end;

function TNNetDiffusionScheduler.SNRWeight(Tt: integer;
  Gamma: TNeuralFloat): TNeuralFloat;
var
  snr, clamped: double;
begin
  // SNR(t) = exp(2*Lambda(t)) (Lambda is the half-log-SNR the DPM-Solver uses).
  snr := Exp(2.0 * Lambda(Tt));
  // min(SNR, Gamma); Gamma = +Inf leaves the clamp inactive (clamped = snr).
  if snr < Gamma then clamped := snr else clamped := Gamma;
  case FPrediction of
    dpV:
      // v-prediction: divide by (SNR + 1). Gamma = +Inf -> SNR/(SNR+1).
      Result := clamped / (snr + 1.0);
    else // dpEps
      // eps-prediction: divide by SNR. Gamma = +Inf -> 1.
      Result := clamped / snr;
  end;
end;

function TNNetDiffusionScheduler.SigmaToTimestep(TargetSigma: TNeuralFloat): integer;
var
  t, best: integer;
  d, bestD: double;
begin
  // The schedule's Sigma() is monotone increasing in Tt, so the nearest match
  // is well defined. Linear scan keeps this dependency-free and exact.
  best := 1; bestD := Abs(SigmaOf(1) - TargetSigma);
  for t := 2 to FT do
  begin
    d := Abs(SigmaOf(t) - TargetSigma);
    if d < bestD then begin bestD := d; best := t; end;
  end;
  Result := best;
end;

function TNNetDiffusionScheduler.BuildTimestepSchedule(NumSteps: integer;
  Spacing: TNNetTimestepSpacing): TNeuralIntegerArray;
var
  k, Tt: integer;
  NumStepsM1: integer;
  sigMin, sigMax, invRho, frac, targetSigma: double;
  pMax, pMin: double;
const
  cRho = 7.0; // Karras et al. (2022) recommended rho.
begin
  if NumSteps < 1 then NumSteps := 1;
  SetLength(Result, NumSteps + 1);
  NumStepsM1 := NumSteps - 1;
  case Spacing of
    tsKarras:
      begin
        // Geometrically warped sigmas between sigma_min (Tt=1) and sigma_max
        // (Tt=T):  sigma_i = (sig_max^(1/rho) + i/(n-1)*(sig_min^(1/rho)
        //          - sig_max^(1/rho)))^rho,  i = 0..n-1 (descending),
        // then snapped back to the nearest schedule timestep.
        sigMin := SigmaOf(1);
        sigMax := SigmaOf(FT);
        invRho := 1.0 / cRho;
        pMax := Power(sigMax, invRho);   // #5: invariant across the k loop
        pMin := Power(sigMin, invRho);
        for k := 0 to NumStepsM1 do
        begin
          if NumSteps = 1 then frac := 0.0
          else frac := k / (NumSteps - 1);
          // k=0 -> sigma_max (most noise, highest Tt); k=n-1 -> sigma_min.
          targetSigma := Power(pMax + frac * (pMin - pMax), cRho);
          Tt := SigmaToTimestep(targetSigma);
          // Index 0 of Result is the FIRST (highest-noise) step.
          Result[k] := Tt;
        end;
        Result[NumSteps] := 0; // clean-image anchor.
      end;
    else // tsUniform
      begin
        for k := 0 to NumStepsM1 do
        begin
          // Descending: Result[0] is the highest timestep.
          if NumSteps = 1 then Tt := FT
          else Tt := 1 + Round((NumSteps - 1 - k) * (FT - 1) / (NumSteps - 1));
          Result[k] := Tt;
        end;
        Result[NumSteps] := 0;
      end;
  end;
end;

procedure TNNetDiffusionScheduler.ToEps(Xt, RawPred, EpsOut: TNNetVolume; Tt: integer);
var
  sab, somab: TNeuralFloat;
begin
  if FPrediction = dpEps then
  begin
    if EpsOut <> RawPred then EpsOut.Copy(RawPred);
    Exit;
  end;
  // v-prediction: eps = sqrt(ab)*v + sqrt(1-ab)*x_t.
  // Bulk form (rule #13). EpsOut never aliases RawPred or Xt at the (internal)
  // call sites, so capturing RawPred into EpsOut first is safe.
  sab := FSqrtAlphaBar[Tt];
  somab := FSqrtOneMinusAlphaBar[Tt];
  EpsOut.Copy(RawPred);
  EpsOut.Mul(sab);
  EpsOut.MulAdd(somab, Xt);
end;

procedure TNNetDiffusionScheduler.PredictX0(Xt, RawPred, X0Out: TNNetVolume;
  Tt: integer);
var
  somab, invSqrtAb: TNeuralFloat;
  Eps: TNNetVolume;
begin
  // Reuse the shared eps plumbing, then map eps -> x0. Use a persistent scratch
  // buffer (rule #17) so X0Out may safely alias RawPred.
  if not Assigned(FEpsScratch) then FEpsScratch := TNNetVolume.Create(Xt)
  else FEpsScratch.ReSize(Xt);
  Eps := FEpsScratch;
  ToEps(Xt, RawPred, Eps, Tt);
  somab := FSqrtOneMinusAlphaBar[Tt];
  invSqrtAb := 1.0 / FSqrtAlphaBar[Tt];
  // x0 = (Xt - somab*Eps)*invSqrtAb (rule #13). Eps is a distinct scratch and
  // X0Out never aliases Xt at any call site, so the copy-first form is safe.
  X0Out.Copy(Xt);
  X0Out.MulAdd(-somab, Eps);
  X0Out.Mul(invSqrtAb);
end;

procedure TNNetDiffusionScheduler.AddNoise(X0, Xt: TNNetVolume; Tt: integer;
  NoiseOut, PreSampledNoise: TNNetVolume);
var
  i: integer;
  X0SizeM1: integer;
  sab, somab, eps: TNeuralFloat;
  hasPre, hasNoiseOut: boolean;
begin
  sab := FSqrtAlphaBar[Tt];
  somab := FSqrtOneMinusAlphaBar[Tt];
  X0SizeM1 := X0.Size - 1;
  // Hoist the per-element Assigned() tests (rule #5 / no-nil-tests-in-hot-loops).
  hasPre := Assigned(PreSampledNoise);
  hasNoiseOut := Assigned(NoiseOut);
  for i := 0 to X0SizeM1 do
  begin
    if hasPre then eps := PreSampledNoise.FData[i]
    else eps := RandG(0, 1);
    if hasNoiseOut then NoiseOut.FData[i] := eps;
    Xt.FData[i] := sab * X0.FData[i] + somab * eps;
  end;
end;

class procedure TNNetDiffusionScheduler.ApplyCFG(EpsCond, EpsUncond,
  Dst: TNNetVolume; W: TNeuralFloat);
var i, DstSizeM1: integer;
  eu: TNeuralFloat;
begin
  // Dst := (1-W)*EpsUncond + W*EpsCond (rule #13). The copy-first bulk form is
  // safe when Dst aliases EpsUncond, but WRONG when Dst aliases EpsCond (Copy
  // would clobber EpsCond before it is read); fall back to the scalar loop then.
  if Dst = EpsCond then
  begin
    DstSizeM1 := Dst.Size - 1;
    for i := 0 to DstSizeM1 do
    begin
      eu := EpsUncond.FData[i];   // #4: read once
      Dst.FData[i] := eu + W * (EpsCond.FData[i] - eu);
    end;
  end
  else
  begin
    Dst.Copy(EpsUncond);
    Dst.Mul(1.0 - W);
    Dst.MulAdd(W, EpsCond);
  end;
end;

class procedure TNNetDiffusionScheduler.ApplyCFG(EpsCond, EpsUncond,
  Dst: TNNetVolume; W, GuidanceRescale: TNeuralFloat);
var
  stdCond, stdCfg, factor: TNeuralFloat;
begin
  // Plain CFG mix first (also fills Dst when GuidanceRescale = 0).
  ApplyCFG(EpsCond, EpsUncond, Dst, W);
  if GuidanceRescale = 0 then exit;

  // Rescale the over-saturated CFG result toward the conditional's std, then
  // blend back by GuidanceRescale (diffusers rescale_noise_cfg, Lin et al. 2023).
  stdCond := EpsCond.GetStdDeviation();
  stdCfg  := Dst.GetStdDeviation();
  if stdCfg = 0 then exit;
  factor := stdCond / stdCfg;
  // Whole expression is Dst[i]*K with invariant K (rule #5/#13) -> one bulk scale.
  Dst.Mul(GuidanceRescale * factor + (1 - GuidanceRescale));
end;

procedure TNNetDiffusionScheduler.ResetMultistep;
begin
  FHasPrev := false;
  FUniThisOrder := 1;
  FUniLowerOrderNums := 0;
  FUniHistLen := 0;
end;

procedure TNNetDiffusionScheduler.Step(Xt, RawPred: TNNetVolume;
  Tt, TtPrev: integer; Method: TNNetSamplerMethod; Eta: TNeuralFloat);
var
  i: integer;
  XtSizeM1: integer;
  Eps: TNNetVolume;
  invSqrtAlpha, coef, sigma, z: TNeuralFloat;
  abT, abPrev, sqrtAbPrev, x0, dirCoef: TNeuralFloat;
  // DPM-Solver++ locals.
  lamT, lamPrev, h, hLast, r0: TNeuralFloat;
  curX0, dCoef: TNeuralFloat;
  // UniPC locals (order-2 bh2, predict_x0).
  alphaT, sigmaTuni, sigmaS0, hh, hphi1, Bh, rk, rhoC0, rhoC1: TNeuralFloat;
  detR, sigRatio: TNeuralFloat;
  cOrder, pOrder: integer;
  // Loop-invariant hoists (rule #5): per-step coefficients pulled out of the
  // per-element loops below.
  needNoise: boolean;
  invSqrtAbT, somabT, cA, cB, kX, kD, kM0, kBh, invRk: TNeuralFloat;
  cD1: TNeuralFloat;           // UniPC D1 coef
  kA, cE, kZ: TNeuralFloat;    // per-step affine coeffs for the sampler loops (#5)
  m0Vol, m1Vol: TNNetVolume;   // UniPC history rows bound once (#5)
begin
  // Persistent lazily-sized eps scratch (rule #17). Step and PredictX0 never
  // run nested, so they safely share FEpsScratch.
  if not Assigned(FEpsScratch) then FEpsScratch := TNNetVolume.Create(Xt)
  else FEpsScratch.ReSize(Xt);
  Eps := FEpsScratch;
  // All updates work from eps; v-prediction is converted up front.
  ToEps(Xt, RawPred, Eps, Tt);
  XtSizeM1 := Xt.Size - 1;
  case Method of
      smDDPM:
        begin
          // x_{Tt-1} = 1/sqrt(a_t)*(x_t - beta_t/sqrt(1-ab_t)*eps) + sigma*z.
          invSqrtAlpha := 1.0 / Sqrt(FAlpha[Tt]);
          coef := FBeta[Tt] / FSqrtOneMinusAlphaBar[Tt];
          needNoise := Tt > 1;
          if needNoise then sigma := Sqrt(FBeta[Tt]) else sigma := 0;
          if needNoise then
          begin
            cE := invSqrtAlpha * coef;   // #5: invariant across the element loop
            for i := 0 to XtSizeM1 do
            begin
              z := RandG(0, 1);
              Xt.FData[i] := invSqrtAlpha * Xt.FData[i] -
                cE * Eps.FData[i] + sigma * z;
            end;
          end
          else
          begin
            // No noise: uniform affine Xt := invSqrtAlpha*(Xt - coef*Eps) (#13).
            Xt.Mul(invSqrtAlpha);
            Xt.MulAdd(-invSqrtAlpha * coef, Eps);
          end;
        end;
      smDDIM:
        begin
          // x0_pred = (x_t - sqrt(1-ab_t)*eps)/sqrt(ab_t)
          // x_{TtPrev} = sqrt(ab_prev)*x0 + sqrt(1-ab_prev - sigma^2)*eps + sigma*z
          // sigma = eta*sqrt((1-ab_prev)/(1-ab_t))*sqrt(1-ab_t/ab_prev).
          abT := FAlphaBar[Tt];
          abPrev := FAlphaBar[TtPrev];
          sqrtAbPrev := Sqrt(abPrev);
          if (Eta > 0) and (TtPrev > 0) then
            sigma := Eta * Sqrt((1.0 - abPrev) / (1.0 - abT)) *
                     Sqrt(1.0 - abT / abPrev)
          else
            sigma := 0;
          dirCoef := Sqrt(Max(0.0, 1.0 - abPrev - sigma * sigma));
          somabT := FSqrtOneMinusAlphaBar[Tt];
          invSqrtAbT := 1.0 / Sqrt(abT);
          if sigma > 0 then
          begin
            // #5: fold the affine into per-step coeffs. Expanding
            //   sqrt(ab_prev)*(x_t - somab_t*eps)/sqrt(ab_t) + dirCoef*eps
            // gives kA*x_t + cE*eps, kA = sqrt(ab_prev)/sqrt(ab_t).
            kA := sqrtAbPrev * invSqrtAbT;
            cE := dirCoef - kA * somabT;
            for i := 0 to XtSizeM1 do
            begin
              z := RandG(0, 1);
              Xt.FData[i] := kA * Xt.FData[i] + cE * Eps.FData[i] + sigma * z;
            end;
          end
          else
          begin
            // sigma=0 (z=0): Xt := k*Xt + c*Eps, a uniform affine map (#13).
            //   k = sqrt(ab_prev)/sqrt(ab_t),
            //   c = dirCoef - sqrt(ab_prev)/sqrt(ab_t)*somab_t.
            Xt.Mul(sqrtAbPrev * invSqrtAbT);
            Xt.MulAdd(dirCoef - sqrtAbPrev * invSqrtAbT * somabT, Eps);
          end;
        end;
      smEulerAncestral:
        begin
          // Euler-ancestral ("Euler a", Karras et al. 2022). Work in the VE
          // sigma parameterisation: sigma = sqrt((1-ab)/ab). The denoised
          // estimate is x0 = (x_t - somab_t*eps)/sqrt(ab_t). The ancestral split
          // of the remaining noise sigma_next = Sigma(TtPrev):
          //   sigma_up   = eta * sqrt( (s2^2*(s1^2 - s2^2)) / s1^2 )   (injected)
          //   sigma_down = sqrt( max(0, s2^2 - sigma_up^2) )          (Euler drift)
          // with s1 = Sigma(Tt), s2 = Sigma(TtPrev). The Euler drift along the
          // probability-flow ODE from sigma=s1 to sigma=sigma_down is, in terms
          // of the denoised x0:  x = x0 + sigma_down*((x_t/sqrt(ab_t) - x0)/s1)
          //                        = x0 + (sigma_down/s1)*(x_t/sqrt(ab_t) - x0).
          // Finally re-noise to VP scale at TtPrev: multiply by sqrt(ab_prev)
          // and add the ancestral term sqrt(ab_prev)*sigma_up*z.
          abT := FAlphaBar[Tt];
          somabT := FSqrtOneMinusAlphaBar[Tt];
          invSqrtAbT := 1.0 / Sqrt(abT);
          if TtPrev = 0 then
          begin
            // Final hop: emit the denoised x0 directly.
            // (Xt - somabT*Eps)*invSqrtAbT as a uniform affine map (#13).
            Xt.Mul(invSqrtAbT);
            Xt.MulAdd(-somabT * invSqrtAbT, Eps);
            Exit;
          end;
          abPrev := FAlphaBar[TtPrev];
          sqrtAbPrev := Sqrt(abPrev);
          begin
            // Reuse DPM locals as scratch for the sigma split.
            lamT := SigmaOf(Tt);                       // s1
            lamPrev := SigmaOf(TtPrev);                // s2
            if (Eta > 0) and (lamT > 0) then
              sigma := Eta * Sqrt((lamPrev * lamPrev *
                Max(0.0, lamT * lamT - lamPrev * lamPrev)) / (lamT * lamT))
            else
              sigma := 0;                            // sigma_up
            h := Sqrt(Max(0.0, lamPrev * lamPrev - sigma * sigma)); // sigma_down
            if lamT > 0 then r0 := h / lamT else r0 := 0;           // drift ratio
            if sigma > 0 then
            begin
              // #5: curX0 = invSqrtAbT*x_t - (1-r0)*somab_t*invSqrtAbT*eps, so
              // the whole update folds to kA*x_t + cE*eps + kZ*z per step.
              kA := sqrtAbPrev * invSqrtAbT;
              cE := -sqrtAbPrev * (1 - r0) * somabT * invSqrtAbT;
              kZ := sqrtAbPrev * sigma;
              for i := 0 to XtSizeM1 do
              begin
                z := RandG(0, 1);
                Xt.FData[i] := kA * Xt.FData[i] + cE * Eps.FData[i] + kZ * z;
              end;
            end
            else
            begin
              // sigma=0 (z=0): the whole update collapses to a uniform affine
              // Xt := k*Xt + c*Eps (#13), with
              //   k = sqrt(ab_prev)/sqrt(ab_t),
              //   c = -sqrt(ab_prev)*(1-r0)*somab_t/sqrt(ab_t).
              Xt.Mul(sqrtAbPrev * invSqrtAbT);
              Xt.MulAdd(-(sqrtAbPrev * (1 - r0) * somabT * invSqrtAbT), Eps);
            end;
          end;
        end;
      smDPMSolverPP2M:
        begin
          // DPM-Solver++(2M) in the data-prediction (x0) form (Lu et al. 2022,
          // arXiv:2211.01095). lambda = half-log-SNR INCREASES as Tt decreases.
          // For a step Tt -> TtPrev: h = lambda_{TtPrev} - lambda_t > 0, and the
          // base (first-order) update is
          //   x_{TtPrev} = (sqrt(ab_prev)/sqrt(ab_t))*x_t
          //               - sqrt(1-ab_prev)*(exp(-h)-1)*D
          // where D is the data-prediction x0 (2nd-order corrected when history
          // is available): D = (1+1/(2r))*x0_cur - (1/(2r))*x0_prev,
          //   r = h_last / h, h_last = lambda_t - lambda_{Tt of previous step}.
          abT := FAlphaBar[Tt];
          somabT := FSqrtOneMinusAlphaBar[Tt];
          invSqrtAbT := 1.0 / Sqrt(abT);
          // Convert current eps -> x0 prediction; reuse Eps to hold curX0.
          // (Xt - somabT*Eps)*invSqrtAbT as a uniform affine map (#13).
          Eps.Mul(-somabT * invSqrtAbT);
          Eps.MulAdd(invSqrtAbT, Xt);
          lamT := Lambda(Tt);
          if TtPrev = 0 then
          begin
            // Final hop to the clean image: emit the (corrected) x0 prediction.
            if FHasPrev then
            begin
              hLast := lamT - FPrevLambda;          // < 0
              h := -hLast;                          // approximate remaining step
              if Abs(h) < 1e-12 then r0 := 1.0 else r0 := hLast / h;
              cA := 1.0 + 0.5 / r0;
              cB := 0.5 / r0;
              // cA*Eps - cB*FPrevX0 as a uniform affine map (#13).
              Xt.Copy(Eps);
              Xt.Mul(cA);
              Xt.MulAdd(-cB, FPrevX0);
            end
            else
              Xt.Copy(Eps);
            FHasPrev := false;
            Exit;
          end;
          abPrev := FAlphaBar[TtPrev];
          lamPrev := Lambda(TtPrev);
          sqrtAbPrev := Sqrt(abPrev);
          h := lamPrev - lamT;                      // > 0
          // Step-invariant update coefficients (rule #5): kX scales x_t, kD
          // scales the data-prediction term.
          kX := sqrtAbPrev * invSqrtAbT;
          kD := Sqrt(1.0 - abPrev) * (Exp(-h) - 1.0);
          if not FHasPrev then
          begin
            // First-order (DDIM-equivalent) bootstrap.
            // kX*Xt - kD*Eps as a uniform affine map (#13).
            Xt.Mul(kX);
            Xt.MulAdd(-kD, Eps);
          end
          else
          begin
            hLast := lamT - FPrevLambda;            // < 0
            if Abs(h) < 1e-12 then r0 := 1.0 else r0 := hLast / h;
            cA := 1.0 + 0.5 / r0;
            cB := 0.5 / r0;
            // Xt := kX*Xt - kD*(cA*Eps - cB*FPrevX0) as a uniform affine map (#13).
            Xt.Mul(kX);
            Xt.MulAdd(-kD * cA, Eps);
            Xt.MulAdd(kD * cB, FPrevX0);
          end;
          // Store this step's x0 and lambda for the next multistep correction.
          if not Assigned(FPrevX0) then FPrevX0 := TNNetVolume.Create(Xt);
          FPrevX0.Copy(Eps);   // Eps currently holds curX0
          FPrevLambda := lamT;
          FHasPrev := true;
        end;
      smUniPC:
        begin
          // UniPC order-2 bh2 predictor-corrector (Zhao et al. 2023), data
          // (x0) prediction form, mirroring diffusers UniPCMultistepScheduler
          // (multistep_uni_c_bh_update then multistep_uni_p_bh_update).
          //   alpha_t = sqrt(ab_t), sigma_t = sqrt(1-ab_t),
          //   lambda_t = log(alpha_t/sigma_t) (= Lambda()), increases as Tt falls.
          //   h  = lambda_target - lambda_current,  hh = -h (predict_x0),
          //   h_phi_1 = expm1(hh),  B_h = expm1(hh) (bh2).
          // 1. Convert current eps -> current x0 prediction m_t (reuse Eps).
          abT := FAlphaBar[Tt];
          somabT := FSqrtOneMinusAlphaBar[Tt];
          invSqrtAbT := 1.0 / Sqrt(abT);
          // (Xt - somabT*Eps)*invSqrtAbT as a uniform affine map (#13).
          Eps.Mul(-somabT * invSqrtAbT);
          Eps.MulAdd(invSqrtAbT, Xt);
          lamT := Lambda(Tt);

          // 2. CORRECTOR: uses the PREVIOUS step's stored output (m0 = the
          //    newest history entry FUniHistX0[FUniHistLen-1] at FUniHistT),
          //    the previous (corrected) sample FUniPrevSample = x_{s0}, and
          //    m_t (this step's x0). Order = FUniThisOrder (the order the
          //    previous predictor used). Runs from step 1 onward. The model is
          //    NOT re-run; m_t (Eps) is left unchanged for the history push.
          if FHasPrev then
          begin
            sigmaTuni := FSqrtOneMinusAlphaBar[Tt];        // sigma at this Tt
            alphaT := Sqrt(abT);                           // alpha at this Tt
            lamPrev := Lambda(FUniHistT[FUniHistLen - 1]); // lambda at s0
            sigmaS0 := FSqrtOneMinusAlphaBar[FUniHistT[FUniHistLen - 1]];
            h := lamT - lamPrev;
            hh := -h;
            hphi1 := Exp(hh) - 1.0;
            Bh := hphi1;                                   // bh2
            sigRatio := sigmaTuni / sigmaS0;
            cOrder := FUniThisOrder;
            // Step-invariant products hoisted out of the per-element loops (#5).
            kM0 := alphaT * hphi1;
            kBh := alphaT * Bh;
            if cOrder >= 2 then
            begin
              // rk for the second history point (FUniHistX0[FUniHistLen-2] at
              // FUniHistT[FUniHistLen-2]); D1 = (m1 - m0)/rk.
              rk := (Lambda(FUniHistT[FUniHistLen - 2]) - lamPrev) / h;
              invRk := 1.0 / rk;
              // Solve the 2x2 system R*rhos = b for the corrector weights.
              //   R = [[1, 1],[rk, 1]], b = [hphi1/hh - 1, ...]/Bh-scaled terms.
              //   b1 = (hphi1/hh - 1)/Bh,
              //   b2 = ((hphi1/hh - 1)/hh - 1/2)*2/Bh.
              rhoC0 := (hphi1 / hh - 1.0) / Bh;                       // b1
              rhoC1 := ((hphi1 / hh - 1.0) / hh - 0.5) * 2.0 / Bh;    // b2
              detR := 1.0 - rk;   // det([[1,1],[rk,1]]) = 1 - rk
              // rhos = R^{-1} b ; R^{-1} = 1/detR * [[1,-1],[-rk,1]].
              curX0 := (rhoC0 - rhoC1) / detR;        // rhos_c[0]
              dCoef := (-rk * rhoC0 + rhoC1) / detR;  // rhos_c[1] (multiplies D1_t)
              // Uniform affine over Xt (#13). Expanding the scalar body with
              //   D1 = (m1 - m0)*invRk, m0=hist[-1], m1=hist[-2], mt=Eps:
              //   Xt = sigRatio*prev + m0*(-kM0 + kBh*(curX0*invRk + dCoef))
              //                      + m1*(-kBh*curX0*invRk) + mt*(-kBh*dCoef).
              m0Vol := FUniHistX0[FUniHistLen - 1];
              m1Vol := FUniHistX0[FUniHistLen - 2];
              Xt.Copy(FUniPrevSample);
              Xt.Mul(sigRatio);
              Xt.MulAdd(-kM0 + kBh * (curX0 * invRk + dCoef), m0Vol);
              Xt.MulAdd(-kBh * curX0 * invRk, m1Vol);
              Xt.MulAdd(-kBh * dCoef, Eps);
            end
            else
            begin
              // First-order corrector: rhos_c = [0.5]. Uniform affine (#13):
              //   Xt = sigRatio*prev + m0*(-kM0 + 0.5*kBh) + mt*(-0.5*kBh).
              m0Vol := FUniHistX0[FUniHistLen - 1];
              Xt.Copy(FUniPrevSample);
              Xt.Mul(sigRatio);
              Xt.MulAdd(-kM0 + 0.5 * kBh, m0Vol);
              Xt.MulAdd(-0.5 * kBh, Eps);
            end;
          end;

          // 3. Push m_t (pre-correction x0) + Tt into the 2-deep history.
          if FUniHistLen = 2 then
          begin
            // Drop oldest: shift [1] -> [0], free/reuse [1] for the new entry.
            FUniHistX0[0].Copy(FUniHistX0[1]);
            FUniHistT[0] := FUniHistT[1];
            FUniHistX0[1].Copy(Eps);
            FUniHistT[1] := Tt;
          end
          else
          begin
            if not Assigned(FUniHistX0[FUniHistLen]) then
              FUniHistX0[FUniHistLen] := TNNetVolume.Create(Xt);
            FUniHistX0[FUniHistLen].Copy(Eps);
            FUniHistT[FUniHistLen] := Tt;
            Inc(FUniHistLen);
          end;

          // 4. Determine this step's predictor order (lower_order_final clamps
          //    the FINAL step -- TtPrev=0 -- to first order), then store it as
          //    FUniThisOrder for the next step's corrector. Save the current
          //    sample as last_sample for that corrector BEFORE the predictor
          //    overwrites Xt.
          if TtPrev = 0 then pOrder := 1
          else pOrder := 2;
          if pOrder > FUniLowerOrderNums + 1 then pOrder := FUniLowerOrderNums + 1;
          FUniThisOrder := pOrder;
          if not Assigned(FUniPrevSample) then FUniPrevSample := TNNetVolume.Create(Xt);
          FUniPrevSample.Copy(Xt);

          // 5. PREDICTOR: produce x_{TtPrev}. m0 = m_t (Eps, the newest output).
          if TtPrev = 0 then
          begin
            // Final hop to the clean image. With sigma_target=0, alpha_target=1,
            // h -> +inf, h_phi_1 -> -1, so x = m0 (the x0 prediction).
            Xt.Copy(Eps);
          end
          else
          begin
            abPrev := FAlphaBar[TtPrev];
            alphaT := Sqrt(abPrev);                  // alpha at target
            sigmaTuni := FSqrtOneMinusAlphaBar[TtPrev];
            sigmaS0 := FSqrtOneMinusAlphaBar[Tt];
            lamPrev := Lambda(TtPrev);
            h := lamPrev - lamT;
            hh := -h;
            hphi1 := Exp(hh) - 1.0;
            Bh := hphi1;                             // bh2
            sigRatio := sigmaTuni / sigmaS0;
            // Step-invariant products hoisted out of the per-element loops (#5).
            kM0 := alphaT * hphi1;
            kBh := alphaT * Bh;
            if (pOrder >= 2) and (FUniHistLen >= 2) then
            begin
              // Predictor D1 from the previous history entry (rhos_p = [0.5]).
              //   rk = (lambda_{prev hist} - lambda_t)/h.
              rk := (Lambda(FUniHistT[FUniHistLen - 2]) - lamT) / h;
              invRk := 1.0 / rk;
              // Uniform affine over Xt (#13). With m0=Eps, m1=hist[-2],
              //   D1 = (m1 - m0)*invRk and cD1 = 0.5*kBh*invRk:
              //   Xt = sigRatio*prev + Eps*(-kM0 + cD1) + m1*(-cD1).
              cD1 := 0.5 * kBh * invRk;
              m1Vol := FUniHistX0[FUniHistLen - 2];
              Xt.Copy(FUniPrevSample);
              Xt.Mul(sigRatio);
              Xt.MulAdd(-kM0 + cD1, Eps);
              Xt.MulAdd(-cD1, m1Vol);
            end
            else
            begin
              // First-order predictor (no correction term).
              // sigRatio*FUniPrevSample - kM0*Eps as a uniform affine map (#13).
              Xt.Copy(FUniPrevSample);
              Xt.Mul(sigRatio);
              Xt.MulAdd(-kM0, Eps);
            end;
          end;

          if FUniLowerOrderNums < 2 then Inc(FUniLowerOrderNums);
          FHasPrev := true;
        end;
    end;
end;

procedure TNNetDiffusionScheduler.SampleHeun(X: TNNetVolume;
  Denoise: TNNetDenoiseCallback; NumSteps: integer; Spacing: TNNetTimestepSpacing);
var
  k, Tt, TtPrev: integer;
  NumStepsM1: integer;
  Pred, X0, Y, Ye, X0b, XPred: TNNetVolume;
  Schedule: TNeuralIntegerArray;
  sigma, sigmaNext, sqrtAbT, sqrtAbPrev: TNeuralFloat;
  kEuler, cCorr1, cCorr2: TNeuralFloat;
  invSqrtAbT, invSigma, invSigmaNext, dSig, halfDSig: TNeuralFloat;
begin
  // Deterministic Heun 2nd-order sampler in the VE sigma parameterisation:
  //   y     = x_t / sqrt(ab_t)                    (VE sample),
  //   x0    = PredictX0(x_t, t)                   (denoiser output, 1st eval),
  //   d     = (y - x0)/sigma_t,                   (drift),
  //   y_e   = y + d*(sigma_{t-1} - sigma_t)       (Euler predictor).
  // On the final step (sigma_{t-1} = 0) emit y_e directly (skip corrector, as
  // k-diffusion does when sigmas[i+1] == 0). Otherwise the SECOND denoiser eval
  // at the predicted VP point x_e = sqrt(ab_prev)*y_e gives x0_2, and:
  //   d2    = (y_e - x0_2)/sigma_{t-1},
  //   y_new = y + (sigma_{t-1} - sigma_t)*0.5*(d + d2)  (trapezoidal average).
  // Re-noise back to VP scale: x_{t-1} = sqrt(ab_prev)*y_new.
  if NumSteps < 1 then NumSteps := 1;
  NumStepsM1 := NumSteps - 1;
  Schedule := BuildTimestepSchedule(NumSteps, Spacing);
  Pred  := TNNetVolume.Create(X);
  X0    := TNNetVolume.Create(X);
  Y     := TNNetVolume.Create(X);
  Ye    := TNNetVolume.Create(X);
  X0b   := TNNetVolume.Create(X);
  XPred := TNNetVolume.Create(X);
  try
    for k := 0 to NumStepsM1 do
    begin
      Tt := Schedule[k];
      TtPrev := Schedule[k + 1];
      sigma := SigmaOf(Tt);
      sigmaNext := SigmaOf(TtPrev);          // 0 when TtPrev = 0
      sqrtAbT := FSqrtAlphaBar[Tt];
      // Step-invariant reciprocals/deltas hoisted out of the element loops (#5).
      invSqrtAbT := 1.0 / sqrtAbT;
      invSigma := 1.0 / sigma;
      dSig := sigmaNext - sigma;
      // First denoiser eval at (x_t, t) -> x0.
      Denoise(X, Pred, Tt);
      PredictX0(X, Pred, X0, Tt);
      // VE sample and Euler predictor y_e.
      //   y  = x_t * invSqrtAbT
      //   ye = y + (dSig*invSigma)*(y - x0) = (1+k)*y - k*x0,  k = dSig*invSigma.
      // Uniform affine maps (#13).
      kEuler := dSig * invSigma;
      Y.Copy(X);
      Y.Mul(invSqrtAbT);
      Ye.Copy(Y);
      Ye.Mul(1.0 + kEuler);
      Ye.MulAdd(-kEuler, X0);
      if TtPrev = 0 then
      begin
        // Final step: sqrt(ab_prev) = 1, y_e is already the clean image.
        X.Copy(Ye);
      end
      else
      begin
        sqrtAbPrev := FSqrtAlphaBar[TtPrev];
        invSigmaNext := 1.0 / sigmaNext;
        halfDSig := 0.5 * dSig;
        // Build the predicted VP sample for the 2nd eval, then re-evaluate.
        XPred.Copy(Ye);
        XPred.Mul(sqrtAbPrev);
        Denoise(XPred, Pred, TtPrev);
        PredictX0(XPred, Pred, X0b, TtPrev);
        // x = sqrtAbPrev*(Y + halfDSig*(d + d2)), with
        //   d  = (Y - X0)*invSigma, d2 = (Ye - X0b)*invSigmaNext.
        // Expands to a uniform affine combination of Y, X0, Ye, X0b (#13):
        //   x = (sqrtAbPrev + c1)*Y - c1*X0 + c2*Ye - c2*X0b,
        //   c1 = sqrtAbPrev*halfDSig*invSigma, c2 = sqrtAbPrev*halfDSig*invSigmaNext.
        cCorr1 := sqrtAbPrev * halfDSig * invSigma;
        cCorr2 := sqrtAbPrev * halfDSig * invSigmaNext;
        X.Copy(Y);
        X.Mul(sqrtAbPrev + cCorr1);
        X.MulAdd(-cCorr1, X0);
        X.MulAdd(cCorr2, Ye);
        X.MulAdd(-cCorr2, X0b);
      end;
    end;
  finally
    Pred.Free; X0.Free; Y.Free; Ye.Free; X0b.Free; XPred.Free;
  end;
end;

procedure TNNetDiffusionScheduler.Sample(X: TNNetVolume;
  Denoise: TNNetDenoiseCallback; NumSteps: integer;
  Method: TNNetSamplerMethod; Eta: TNeuralFloat; Spacing: TNNetTimestepSpacing);
var
  k, Tt, TtPrev: integer;
  NumStepsM1: integer;
  Pred: TNNetVolume;
  Schedule: TNeuralIntegerArray;
begin
  if Method = smHeun then
  begin
    // Heun needs two denoiser evals per step; handled by its own driver.
    SampleHeun(X, Denoise, NumSteps, Spacing);
    Exit;
  end;
  if NumSteps < 1 then NumSteps := 1;
  NumStepsM1 := NumSteps - 1;
  ResetMultistep;
  // Schedule[0..NumSteps]: descending timesteps, Schedule[NumSteps]=0 (clean).
  Schedule := BuildTimestepSchedule(NumSteps, Spacing);
  Pred := TNNetVolume.Create(X);
  try
    for k := 0 to NumStepsM1 do
    begin
      Tt := Schedule[k];
      TtPrev := Schedule[k + 1];
      Denoise(X, Pred, Tt);
      Step(X, Pred, Tt, TtPrev, Method, Eta);
    end;
  finally
    Pred.Free;
  end;
end;

{ TNNetLCMScheduler }

const
  cLCMSigmaData = 0.5;   // sigma_data in the LCM boundary scalings.
  cLCMTimeScale = 0.1;   // the 1/0.1 = 10x timestep scaling (Luo et al. 2023).

function TNNetLCMScheduler.CSkip(Tt: integer): TNeuralFloat;
var ts2, sd2: double;
begin
  // c_skip(t) = sigma_data^2 / ((t/0.1)^2 + sigma_data^2).
  ts2 := Sqr(Tt / cLCMTimeScale);
  sd2 := Sqr(cLCMSigmaData);
  Result := sd2 / (ts2 + sd2);
end;

function TNNetLCMScheduler.COut(Tt: integer): TNeuralFloat;
var ts, sd2: double;
begin
  // c_out(t) = (t/0.1)*sigma_data / sqrt((t/0.1)^2 + sigma_data^2).
  ts := Tt / cLCMTimeScale;
  sd2 := Sqr(cLCMSigmaData);
  Result := ts * cLCMSigmaData / Sqrt(Sqr(ts) + sd2);
end;

procedure TNNetLCMScheduler.LCMStep(Xt, RawPred: TNNetVolume; Tt: integer);
var
  cs, co: TNeuralFloat;
  X0Hat: TNNetVolume;
begin
  // f(x,t) = c_skip(t)*x + c_out(t)*x0_hat. x0_hat reuses the shared eps/v->x0
  // plumbing so LCM consumes the SAME raw model output the iterative samplers do.
  cs := CSkip(Tt);
  co := COut(Tt);
  // Persistent lazily-sized x0_hat scratch (rule #17); PredictX0 uses the
  // separate FEpsScratch internally, so the two never collide.
  if not Assigned(FX0HatScratch) then FX0HatScratch := TNNetVolume.Create(Xt)
  else FX0HatScratch.ReSize(Xt);
  X0Hat := FX0HatScratch;
  PredictX0(Xt, RawPred, X0Hat, Tt);
  // Xt := cs*Xt + co*X0Hat (rule #13); X0Hat is a distinct scratch.
  Xt.Mul(cs);
  Xt.MulAdd(co, X0Hat);
end;

procedure TNNetLCMScheduler.LCMSample(X: TNNetVolume;
  Denoise: TNNetDenoiseCallback; NumSteps: integer; Spacing: TNNetTimestepSpacing);
var
  k, Tt, TtNext: integer;
  NumStepsM1: integer;
  Pred, X0: TNNetVolume;
  Schedule: TNeuralIntegerArray;
begin
  if NumSteps < 1 then NumSteps := 1;
  NumStepsM1 := NumSteps - 1;
  // Schedule[0..NumSteps]: descending timesteps, Schedule[NumSteps]=0 (clean).
  Schedule := BuildTimestepSchedule(NumSteps, Spacing);
  Pred := TNNetVolume.Create(X);
  X0   := TNNetVolume.Create(X);
  try
    for k := 0 to NumStepsM1 do
    begin
      Tt := Schedule[k];
      TtNext := Schedule[k + 1];
      // 1. evaluate the consistency function f(x_t,t) -> x0 estimate.
      Denoise(X, Pred, Tt);
      LCMStep(X, Pred, Tt);   // X now holds the consistency x0 estimate.
      if TtNext > 0 then
      begin
        // 2. re-noise the x0 estimate FORWARD to the next, smaller timestep with
        //    fresh Gaussian noise (the LCM multistep "noise back" hop).
        X0.Copy(X);
        AddNoise(X0, X, TtNext);
      end;
      // On the last step (TtNext=0) X already holds the final x0 estimate.
    end;
  finally
    Pred.Free;
    X0.Free;
  end;
end;

end.
