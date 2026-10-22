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
  * TNNetFlowMatchEulerScheduler: a separate flow-matching (rectified-flow)
    sampler (Euler or DPM-Solver++(2M)) with static or resolution-dependent
    sigma shifting.

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
  Classes, SysUtils, Math, fpjson, neuralvolume;

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
  //   smPNDM : diffusers PNDMScheduler with skip_prk_steps (the PLMS sampler
  //     of the SD 1.x configs): a linear multistep on eps of up to 4th order.
  //     It evaluates the model NumSteps+1 times; the second eval repeats the
  //     second timestep and averages the first two eps (a Heun-like bootstrap),
  //     so drive it with BuildSamplerSchedule, which repeats that timestep.
  //     Step derives the target timestep from the schedule stride, as diffusers.
  //   smEuler : diffusers EulerDiscreteScheduler (s_churn = 0): deterministic
  //     Euler in sigma-space. Samples stay in the VP scale x_t = y/sqrt(sigma^2+1)
  //     that diffusers' scale_model_input feeds the model; InitialNoiseScale is
  //     the matching init_noise_sigma. For eps-prediction it equals DDIM.
  //   smLCM : diffusers LCMScheduler (Latent Consistency Models): the
  //     consistency step of TNNetLCMScheduler over its origin-grid timesteps.
  //     Only a TNNetLCMScheduler runs it; the base class raises.
  TNNetSamplerMethod = (smDDPM, smDDIM, smDPMSolverPP2M, smEulerAncestral,
    smUniPC, smHeun, smPNDM, smEuler, smLCM);

  // Timestep SPACING for the Sample() driver.
  //   tsUniform : evenly strided timesteps in 1..T (the original behaviour).
  //   tsKarras  : Karras et al. (2022) rho=7 sigma spacing. The sampling sigmas
  //     are placed geometrically-warped between sigma_min and sigma_max and then
  //     mapped back to the nearest schedule timestep. Pairs naturally with the
  //     Euler-ancestral sampler but works with any method.
  //   tsTrailing : diffusers timestep_spacing="trailing", round(arange(T,0,-T/N))-1
  //     in diffusers' 0-based timesteps, i.e. Tt = round(T - k*T/N) here, so the
  //     first step is always Tt=T (T=1000, N=4: 1000,750,500,250). A plain
  //     timestep list, so it works with every method; it replaces tsKarras
  //     rather than combining with it. Used with the zero-terminal-SNR rescale.
  //   tsLeading : diffusers timestep_spacing="leading" (SD 1.x/2.x): diffusers t
  //     = k*(T div N) + steps_offset, i.e. Tt = k*(T div N) + StepsOffset + 1.
  TNNetTimestepSpacing = (tsUniform, tsKarras, tsTrailing, tsLeading);

  // Caller-supplied denoiser. Given noisy Xt at integer timestep Tt (1..T) it
  // must write the model prediction (eps or v, per the scheduler's
  // PredictionType) into Output. Output is pre-sized to match Xt.
  TNNetDenoiseCallback = procedure(Xt, Output: TNNetVolume; Tt: integer) of object;

  // Continuous-sigma schedule (diffusers Euler / DPM set_timesteps): Sigmas has
  // NumSteps+1 entries ending in 0; ModelTimesteps[k] is the 0-based model input.
  TNNetSigmaSchedule = record
    Sigmas: array of double;
    ModelTimesteps: array of double;
  end;

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
    // Half-log-SNR lambda_t and Karras sigma_t. Both are pure functions of
    // FAlphaBar, so they are tabulated once in BuildTables instead of paying an
    // Ln pair / a Sqrt per sampling step (rule #27).
    FLambda, FSigma: array of TNeuralFloat;
    // 1/sqrt(ab_t); where ab_t = 0 (zero terminal SNR) it uses cMinAlphaBar,
    // as do FSigma and FLambda, so no sampler divides by zero.
    FInvSqrtAlphaBar: array of TNeuralFloat;
    FRescaleZeroTerminalSNR: boolean;
    // diffusers steps_offset (tsLeading) and timestep_spacing of the config.
    FStepsOffset: integer;
    FConfigSpacing: TNNetTimestepSpacing;
    // ab_prev of DDIM's final step: 1, or ab_1 (diffusers alphas_cumprod[0]).
    FFinalAlphaBar: TNeuralFloat;
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
    // PNDM (PLMS) state, as diffusers PNDMScheduler: FPndmCounter counts the
    // steps of this trajectory, FPndmEts holds the last FPndmEtsLen raw model
    // outputs (oldest first), FPndmCurSample the sample of the first step (the
    // second step restarts from it), FPndmMix the combined eps scratch.
    // FPndmStride = T div NumSteps (set by BuildSamplerSchedule; 0 = unknown).
    FPndmCounter: integer;
    FPndmEtsLen: integer;
    FPndmEts: array[0..3] of TNNetVolume;
    FPndmCurSample, FPndmMix: TNNetVolume;
    FPndmStride: integer;
    FPndmFirstT: integer;
    // Persistent, lazily-sized eps/x0 scratch buffers (rule #17). Amortized
    // resize replaces the per-step TNNetVolume.Create/Free in Step/PredictX0/
    // LCMStep. FEpsScratch is shared by Step and PredictX0 (never nested);
    // FX0HatScratch backs LCMStep's x0_hat.
    FEpsScratch: TNNetVolume;
    FX0HatScratch: TNNetVolume;
    procedure BuildTables(Beta1, BetaT, CosineS: TNeuralFloat);
    // Lin et al. 2023 Algorithm 1 (diffusers rescale_zero_terminal_snr): shifts
    // sqrt(ab) so ab_T = 0, keeps ab_1, and rewrites beta/alpha/ab to match.
    procedure RescaleToZeroTerminalSNR;
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
    // Heun 2nd-order trajectory (smHeun). Separated from Sample() because it is
    // the only method that calls Denoise TWICE per step (a genuine intra-step
    // second derivative), so it cannot share the single-eval Step() dispatch.
    procedure SampleHeun(X: TNNetVolume; Denoise: TNNetDenoiseCallback;
      NumSteps: integer; Spacing: TNNetTimestepSpacing);
    // diffusers PNDMScheduler.step_plms (smPNDM of Step).
    procedure StepPNDM(Xt, RawPred: TNNetVolume; Tt, TtPrev: integer);
    // X0Out := x0 of RawPred at a point with sqrt(ab), sqrt(1-ab), 1/sqrt(ab).
    procedure PredictX0At(Xt, RawPred, X0Out: TNNetVolume;
      SqrtAb, SqrtOneMinusAb, InvSqrtAb: TNeuralFloat);
    // Euler(-ancestral) move of Xt (VP scale) from SigmaCur to SigmaNext > 0 given X0.
    procedure EulerAncestralUpdate(Xt, X0, Noise: TNNetVolume; SigmaCur,
      SigmaNext, InvSqrtAbCur, SqrtAbNext, Eta: TNeuralFloat);
    // DPM-Solver++(2M) move of Xt to a point of ab AbNext > 0 given X0; KX is
    // sqrt(1-ab_next)/sqrt(1-ab_cur). Updates the multistep history.
    procedure DPMSolverPP2MUpdate(Xt, X0: TNNetVolume; LambdaCur, LambdaNext,
      KX, AbNext: TNeuralFloat);
    // np.interp of the sigma table at a fractional 0-based timestep.
    function InterpolatedSigma(ModelTimestep: double): double;
  protected
    // Convert a raw model prediction (eps or v) at timestep Tt into eps.
    procedure ToEps(Xt, RawPred, EpsOut: TNNetVolume; Tt: integer);
    // CreateFromDiffusersConfig hands the parsed config here after Create, so a
    // descendant reads its own keys.
    procedure ReadDiffusersConfig(Config: TJSONObject); virtual;
  public
    // Build a list of NumSteps+1 timesteps (descending, last entry 0 = clean
    // image) using the requested spacing. Caller-owned dynamic array.
    function BuildTimestepSchedule(NumSteps: integer;
      Spacing: TNNetTimestepSpacing): TNeuralIntegerArray;
    // The model-eval timesteps of Method as the matching diffusers scheduler
    // sets them, plus the final 0: BuildTimestepSchedule except that
    //   smPNDM repeats the second timestep (NumSteps+1 evals, NumSteps >= 2)
    //     and records the stride T div NumSteps that its Step uses;
    //   smDPMSolverPP2M with tsLeading/tsUniform takes NumSteps of NumSteps+1
    //     points (leading stride T div (NumSteps+1); linspace over 0..T-1).
    // Calls ResetMultistep. Pairs Schedule[k] -> Schedule[k+1] drive Step.
    function BuildSamplerSchedule(NumSteps: integer;
      Spacing: TNNetTimestepSpacing; Method: TNNetSamplerMethod):
      TNeuralIntegerArray; virtual;
    // The img2img/inpaint schedule at Strength, Schedule[0] the start: here
    // BuildSamplerSchedule from the Img2ImgStartStep entry on (get_timesteps).
    function BuildImg2ImgSchedule(NumSteps: integer;
      Spacing: TNNetTimestepSpacing; Method: TNNetSamplerMethod;
      Strength: double): TNeuralIntegerArray; virtual;
    // diffusers SD img2img/inpaint get_timesteps: the first schedule entry that
    // runs at Strength, N - min(int(N * Strength), N); raises when no step runs.
    class function Img2ImgStartStep(NumSteps: integer;
      Strength: double): integer;
    // Factor on unit Gaussian noise for the first sample of Method at FirstTt:
    // diffusers init_noise_sigma over the model-input scale. 1 except for
    // smEuler/smEulerAncestral with tsUniform/tsTrailing, whose init sigma is
    // sigma_max, i.e. sqrt(1-ab) in the VP scale these samplers keep.
    function InitialNoiseScale(Method: TNNetSamplerMethod;
      Spacing: TNNetTimestepSpacing; FirstTt: integer): TNeuralFloat;
    // InitialNoiseScale for a TNNetSigmaSchedule starting at FirstSigma.
    function InitialNoiseScaleAtSigma(Method: TNNetSamplerMethod;
      Spacing: TNNetTimestepSpacing; FirstSigma: double): TNeuralFloat;
    // diffusers _sigma_to_t: the fractional 0-based timestep of Sigma by
    // log-sigma interpolation of the schedule table.
    function SigmaToModelTimestep(Sigma: double): double;
    // diffusers set_timesteps of Euler / Euler a (Spacing tsUniform = linspace,
    // fractional) or, with KarrasSigmas, Euler / DPM++ 2M use_karras_sigmas.
    function BuildSigmaSchedule(NumSteps: integer;
      Spacing: TNNetTimestepSpacing; Method: TNNetSamplerMethod;
      KarrasSigmas: boolean): TNNetSigmaSchedule;
    // Step from Sigma to NextSigma (0: clean) for smEuler, smEulerAncestral and
    // smDPMSolverPP2M; Xt in the VP scale, as Step. Call ResetMultistep first.
    procedure StepAtSigma(Xt, RawPred: TNNetVolume; Sigma, NextSigma: double;
      Method: TNNetSamplerMethod; Eta: TNeuralFloat = 0.0;
      Noise: TNNetVolume = nil);
    // x0 from eps or v (dpV: sqrt(ab_t)*x_t - sqrt(1-ab_t)*v, exact at ab_t = 0).
    // X0Out may alias RawPred, never Xt.
    procedure PredictX0(Xt, RawPred, X0Out: TNNetVolume; Tt: integer);
    // Inverse of PredictX0: the raw prediction (eps or v) whose x0 is X0, Tt >= 1.
    // RawOut may alias X0, never Xt.
    procedure RawPredFromX0(Xt, X0, RawOut: TNNetVolume; Tt: integer);
    // T = number of diffusion steps. Beta1/BetaT used by linear & scaled-linear;
    // CosineS is the small offset of the cosine schedule (default 0.008).
    // pRescaleZeroTerminalSNR forces ab_T = 0 (needs pT >= 2; pair with dpV + DDIM).
    // pFinalAlphaBarIsOne = diffusers set_alpha_to_one; false ends DDIM on ab_1.
    constructor Create(pT: integer; pSchedule: TNNetBetaSchedule = dsLinear;
      pPrediction: TNNetPredictionType = dpEps;
      Beta1: TNeuralFloat = 1.0e-4; BetaT: TNeuralFloat = 0.02;
      CosineS: TNeuralFloat = 0.008; pRescaleZeroTerminalSNR: boolean = false;
      pFinalAlphaBarIsOne: boolean = true);
    // Reads a diffusers DDIM/DDPM/PNDM-style scheduler_config.json (beta and
    // spacing fields); raises on what this class does not implement.
    // ForwardProcessOnly (AddNoise use) accepts the step-only clip_sample.
    constructor CreateFromDiffusersConfig(const FileName: string;
      ForwardProcessOnly: boolean = false);
    destructor Destroy; override;

    // Schedule tables (1-based timesteps; index 0 = clean anchor).
    property Beta[Tt: integer]: TNeuralFloat read GetBeta;
    property Alpha[Tt: integer]: TNeuralFloat read GetAlpha;
    property AlphaBar[Tt: integer]: TNeuralFloat read GetAlphaBar;
    property NumTimesteps: integer read FT;
    property PredictionType: TNNetPredictionType read FPrediction;
    property RescaleZeroTerminalSNR: boolean read FRescaleZeroTerminalSNR;
    property FinalAlphaBar: TNeuralFloat read FFinalAlphaBar;
    // The config's timestep_spacing (tsUniform from Create) and steps_offset.
    property ConfigSpacing: TNNetTimestepSpacing read FConfigSpacing;
    property StepsOffset: integer read FStepsOffset;

    // FORWARD process. Xt := sqrt(ab_t)*X0 + sqrt(1-ab_t)*Noise. If NoiseOut is
    // given it receives the sampled noise (the eps training target). When
    // PreSampledNoise is supplied that noise is used (and copied to NoiseOut)
    // instead of drawing fresh Gaussian samples -- handy for deterministic tests.
    // Xt may alias X0.
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
    // The rescale step alone on a guided Dst, StdCond = std of the conditional
    // prediction: Dst *= GuidanceRescale*StdCond/std(Dst) + 1 - GuidanceRescale.
    class procedure RescaleGuidance(Dst: TNNetVolume;
      StdCond, GuidanceRescale: TNeuralFloat);

    // Reset the DPM-Solver++ multistep history. Call before each new trajectory.
    procedure ResetMultistep;

    // ONE reverse step from timestep Tt to TtPrev (TtPrev=0 -> clean image), in
    // place on Xt. RawPred is the model's raw prediction (eps or v) at Tt.
    // Method selects the update rule; Eta is the DDIM stochasticity (0 = the
    // deterministic DDIM used by the examples; ignored by DDPM/DPM++). For DDPM
    // TtPrev is ignored (it uses Tt-1 by construction). For smEulerAncestral Eta
    // scales the injected ancestral noise (Eta=1 = the standard "Euler a";
    // Eta=0 reduces it to a deterministic Euler step). smEuler ignores Eta.
    // smPNDM ignores TtPrev once BuildSamplerSchedule set its stride.
    // Noise (smEulerAncestral, smLCM) supplies the standard-normal draw instead
    // of RandG, e.g. to replay another implementation's noise.
    procedure Step(Xt, RawPred: TNNetVolume; Tt, TtPrev: integer;
      Method: TNNetSamplerMethod = smDDIM; Eta: TNeuralFloat = 0.0;
      Noise: TNNetVolume = nil); virtual;

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
  // arXiv:2310.04378), diffusers LCMScheduler. Each step evaluates the
  // consistency function f(x_t,t) = c_skip(t)*x_t + c_out(t)*x0_hat, with
  // x0_hat the model's x0 prediction (eps/v through PredictX0) and, at the
  // diffusers timestep t = Tt - 1 scaled by TimestepScaling (s, default 10):
  //   c_skip(t) = sigma_data^2 / ((s*t)^2 + sigma_data^2), sigma_data = 0.5,
  //   c_out(t)  = s*t / sqrt((s*t)^2 + sigma_data^2),
  // then re-noises f to the next timestep with fresh Gaussian noise; the last
  // step returns f. Sample/Step/BuildSamplerSchedule run it as Method smLCM over
  // the origin grid: k*i - 1, k = T div OriginalInferenceSteps (4 of 50 steps:
  // diffusers 999, 759, 499, 259). Other methods run the inherited samplers.
  // Coded by Claude (AI).
  TNNetLCMScheduler = class(TNNetDiffusionScheduler)
  private
    FOriginalInferenceSteps: integer;
    FTimestepScaling: TNeuralFloat;
  protected
    // original_inference_steps and timestep_scaling.
    procedure ReadDiffusersConfig(Config: TJSONObject); override;
  public
    constructor Create(pT: integer; pSchedule: TNNetBetaSchedule = dsLinear;
      pPrediction: TNNetPredictionType = dpEps;
      Beta1: TNeuralFloat = 1.0e-4; BetaT: TNeuralFloat = 0.02;
      CosineS: TNeuralFloat = 0.008; pRescaleZeroTerminalSNR: boolean = false;
      pFinalAlphaBarIsOne: boolean = true);
    // Boundary scalings at the 1-based timestep Tt: c_skip(1) = 1, c_out(1) = 0
    // (f(x,0) = x at diffusers t = 0); c_out -> 1 at large Tt.
    function CSkip(Tt: integer): TNeuralFloat;
    function COut(Tt: integer): TNeuralFloat;
    // ONE consistency evaluation in place: Xt := f(Xt, Tt). RawPred is the raw
    // model output (eps or v) at Tt.
    procedure LCMStep(Xt, RawPred: TNNetVolume; Tt: integer);
    // diffusers LCMScheduler.set_timesteps(NumSteps, strength): NumSteps of the
    // first int(OriginalInferenceSteps * Strength) origin points as Tt, then 0.
    function BuildOriginSchedule(NumSteps: integer;
      Strength: double = 1.0): TNeuralIntegerArray;
    // smLCM: BuildOriginSchedule; Spacing does not apply (diffusers ignores
    // timestep_spacing too). Other methods: inherited.
    function BuildSamplerSchedule(NumSteps: integer;
      Spacing: TNNetTimestepSpacing; Method: TNNetSamplerMethod):
      TNeuralIntegerArray; override;
    // smLCM: the origin grid scaled by Strength, every one of NumSteps run.
    function BuildImg2ImgSchedule(NumSteps: integer;
      Spacing: TNNetTimestepSpacing; Method: TNNetSamplerMethod;
      Strength: double): TNeuralIntegerArray; override;
    // smLCM: LCMStep, then (TtPrev > 0) Xt := sqrt(ab_TtPrev)*Xt +
    // sqrt(1-ab_TtPrev)*(Noise, or RandG when nil). Other methods: inherited.
    procedure Step(Xt, RawPred: TNNetVolume; Tt, TtPrev: integer;
      Method: TNNetSamplerMethod = smDDIM; Eta: TNeuralFloat = 0.0;
      Noise: TNNetVolume = nil); override;
    // diffusers original_inference_steps (50): the origin grid size.
    property OriginalInferenceSteps: integer read FOriginalInferenceSteps
      write FOriginalInferenceSteps;
    // diffusers timestep_scaling (10) of the boundary scalings.
    property TimestepScaling: TNeuralFloat read FTimestepScaling
      write FTimestepScaling;
  end;

  { TNNetAdaptiveProjectedGuidance }
  // Adaptive Projected Guidance (Sadat et al., ICLR 2025), diffusers
  // normalized_guidance + MomentumBuffer, with every reduction over the whole
  // volume. One instance per trajectory: it owns the momentum running average.
  // Coded by Claude (AI).
  TNNetAdaptiveProjectedGuidance = class(TObject)
  private
    FEta, FMomentum, FNormThreshold: TNeuralFloat;
    FUseOriginalFormulation: boolean;
    FRunningDiff: TNNetVolume;
    FHasRunningDiff: boolean;
  public
    // Momentum is the paper's beta (negative, e.g. -0.5; 0 = none); NormThreshold
    // is r (0 = no clamp); Eta weighs the component parallel to Cond.
    constructor Create(pEta, pMomentum, pNormThreshold: TNeuralFloat;
      pUseOriginalFormulation: boolean = false);
    // Parses the command-line form 'ETA,MOMENTUM,THRESHOLD' (dot decimals);
    // raises EConvertError on anything else.
    constructor CreateFromSpec(const Spec: string);
    destructor Destroy; override;
    // Forgets the momentum running average; call before each new trajectory.
    procedure Reset;
    // Dst := Uncond + W*(orth + Eta*par) (Cond instead of Uncond with the original
    // formulation). Dst may alias Cond or Uncond. Allocates only on first use.
    procedure ApplyAPG(Cond, Uncond, Dst: TNNetVolume; W: TNeuralFloat);
    property Eta: TNeuralFloat read FEta write FEta;
    property Momentum: TNeuralFloat read FMomentum write FMomentum;
    property NormThreshold: TNeuralFloat read FNormThreshold write FNormThreshold;
    property UseOriginalFormulation: boolean read FUseOriginalFormulation
      write FUseOriginalFormulation;
  end;

  // What the third (perturbed) pass of TNNetPerturbedGuidance changes in its
  // PerturbedLayers: the attention score matrix becomes the identity (PAG,
  // TNNetFusedSDPA.SkipAttentionScores), or the whole block is skipped (SLG).
  TNNetPerturbedGuidanceMode = (pgmAttentionScores, pgmSkipBlock);

  { TNNetPerturbedGuidance }
  // Perturbed-Attention Guidance (Ahn et al. 2024) and Skip Layer Guidance:
  // diffusers PerturbedAttentionGuidance / SkipLayerGuidance.forward. Skip is
  // the conditional prediction with PerturbedLayers perturbed (see Mode); with
  // both guidances on, Dst := Uncond + W*(Cond - Uncond) + PerturbedScale*(Cond
  // - Skip), Cond in place of the leading Uncond with UseOriginalFormulation.
  // Step counts from 0 to NumSteps - 1, as the pipeline loop index does.
  // Coded by Claude (AI).
  TNNetPerturbedGuidance = class(TObject)
  private
    FGuidanceScale, FPerturbedScale, FGuidanceRescale: TNeuralFloat;
    FAdaptiveScale: TNeuralFloat;
    FPerturbedStart, FPerturbedStop, FGuidanceStart, FGuidanceStop: double;
    FUseOriginalFormulation: boolean;
    FMode: TNNetPerturbedGuidanceMode;
    FPerturbedLayers: TNeuralIntegerArray;
    procedure SetPerturbedWindow(pStart, pStop: double);
    // Dst := CondCoef*Cond + UncondCoef*Uncond + SkipCoef*Skip, the Dst-aliased
    // term first; Uncond may be nil (its term dropped).
    procedure MixPredictions(CondCoef, UncondCoef, SkipCoef: TNeuralFloat;
      Cond, Uncond, Skip, Dst: TNNetVolume);
  public
    // Fractions of the schedule as in diffusers: 0 <= pPerturbedStart < 1 and
    // pPerturbedStart <= pPerturbedStop <= 1, else EArgumentException.
    constructor Create(pGuidanceScale, pPerturbedScale: TNeuralFloat;
      pPerturbedStart: double = 0.01; pPerturbedStop: double = 0.2;
      pGuidanceRescale: TNeuralFloat = 0;
      pUseOriginalFormulation: boolean = false);
    // CFG runs when Trunc(GuidanceStart*N) <= Step < Trunc(GuidanceStop*N) and
    // GuidanceScale is not the no-op scale (1, or 0 with the original formula).
    function IsCFGEnabled(Step, NumSteps: integer): boolean;
    // The perturbed pass runs when Trunc(PerturbedStart*N) < Step <
    // Trunc(PerturbedStop*N) (both strict, as diffusers) and PerturbedScale <> 0.
    function IsPerturbedEnabled(Step, NumSteps: integer): boolean;
    procedure SetPerturbedLayers(const pLayers: array of integer);
    function PerturbsLayer(LayerIdx: integer): boolean;
    // The guided prediction of step Step into Dst (may alias Cond, Uncond or
    // Skip). Uncond / Skip may be nil when their guidance is off at this step.
    procedure Apply(Step, NumSteps: integer; Cond, Uncond, Skip,
      Dst: TNNetVolume);
    // diffusers PAGMixin._get_pag_scale: Max(0, PerturbedScale - AdaptiveScale
    // * (1000 - Timestep)) when both are > 0, else PerturbedScale.
    function PerturbedScaleAt(Timestep: TNeuralFloat): TNeuralFloat;
    // diffusers PAGMixin._apply_perturbed_attention_guidance: Dst := Uncond + W*(Cond
    // - Uncond) (Cond when Uncond is nil) + PerturbedScaleAt(Timestep)*(Cond - Skip).
    procedure ApplyPerturbedAttentionGuidance(Timestep, W: TNeuralFloat;
      Cond, Uncond, Skip, Dst: TNNetVolume);
    property GuidanceScale: TNeuralFloat read FGuidanceScale write FGuidanceScale;
    property PerturbedScale: TNeuralFloat read FPerturbedScale
      write FPerturbedScale;
    // diffusers pag_adaptive_scale (default 0 = off); read by PerturbedScaleAt.
    property AdaptiveScale: TNeuralFloat read FAdaptiveScale
      write FAdaptiveScale;
    property GuidanceRescale: TNeuralFloat read FGuidanceRescale
      write FGuidanceRescale;
    property PerturbedStart: double read FPerturbedStart;
    property PerturbedStop: double read FPerturbedStop;
    // The CFG window, diffusers' guider start / stop (default 0 .. 1).
    property GuidanceStart: double read FGuidanceStart write FGuidanceStart;
    property GuidanceStop: double read FGuidanceStop write FGuidanceStop;
    property UseOriginalFormulation: boolean read FUseOriginalFormulation
      write FUseOriginalFormulation;
    property Mode: TNNetPerturbedGuidanceMode read FMode write FMode;
    property PerturbedLayers: TNeuralIntegerArray read FPerturbedLayers;
  end;

  // Resolution-dependent sigma shift of a flow-matching schedule, mu > 0:
  //   ftsExponential : sigma' = e^mu / (e^mu + (1/sigma - 1))  (FLUX, Qwen-Image)
  //   ftsLinear      : sigma' = mu   / (mu   + (1/sigma - 1))
  TNNetFlowTimeShift = (ftsExponential, ftsLinear);

  // Update rule of TNNetFlowMatchEulerScheduler.Step:
  //   fsEuler         : x_next = x + (sigma_next - sigma)*v.
  //   fsDPMSolverPP2M : diffusers DPMSolverMultistepScheduler(use_flow_sigmas,
  //     flow_prediction, dpmsolver++, solver_order=2, final_sigmas_type="zero").
  //   fsUniPC         : diffusers UniPCMultistepScheduler(use_flow_sigmas,
  //     flow_prediction, predict_x0, solver_order=2, lower_order_final,
  //     final_sigmas_type="zero"), the Wan scheduler: the order-2 UniP
  //     predictor plus the UniC corrector that reuses this step's x0 on the
  //     previous step (no extra model pass).
  TNNetFlowSampler = (fsEuler, fsDPMSolverPP2M, fsUniPC);

  // UniPC B(h) (diffusers solver_type): ustBh1 B(h) = -h, ustBh2 B(h) = e^-h - 1.
  TNNetUniPCSolverType = (ustBh1, ustBh2);

  { TNNetFlowMatchEulerScheduler }
  // Deterministic sampler for rectified-flow models that predict the velocity
  // v = noise - x0 (diffusers FlowMatchEulerDiscreteScheduler): sigma runs from
  // ~1 (noise) to 0 (clean), one Step per sigma interval, Euler by default. The
  // schedule is the sigma list itself, in double (N+1 values, not a hot path).
  // Coded by Claude (AI).
  TNNetFlowMatchEulerScheduler = class(TObject)
  private
    FNumTrainTimesteps: integer;
    FShift: double;
    FUseDynamicShifting: boolean;
    FTimeShiftType: TNNetFlowTimeShift;
    FShiftTerminal: double;
    // calculate_shift's two (image token count, mu) points.
    FBaseImageSeqLen, FMaxImageSeqLen: integer;
    FBaseShift, FMaxShift: double;
    // NumSteps + 1 entries, descending, the last one 0 (the clean sample).
    FSigmas: array of double;
    FSampler: TNNetFlowSampler;
    // Multistep history: the x0 prediction of step FPrevStepIndex (-1: none),
    // sized on first use. UniPC also keeps the x0 of the step before it
    // (FPrevX0Older, valid when FHistoryLen = 2), the corrected sample of step
    // FPrevStepIndex (FPrevSample, diffusers last_sample) and the order its
    // predictor used (FPrevOrder, the order of the next corrector).
    FPrevX0: TNNetVolume;
    FPrevStepIndex: integer;
    FPrevX0Older, FPrevSample: TNNetVolume;
    FHistoryLen, FPrevOrder: integer;
    FUniPCSolverType: TNNetUniPCSolverType;
    FUniPCCorrector: boolean;
    procedure SetSampler(Value: TNNetFlowSampler);
    // DPM-Solver++(2M) step; second order when step StepIndex-1 left history.
    procedure StepDPMSolverPP2M(Sample, Velocity: TNNetVolume;
      StepIndex: integer);
    // UniPC step: corrector + order-2 predictor when step StepIndex-1 left
    // history, first order otherwise and on the last step.
    procedure StepUniPC(Sample, Velocity: TNNetVolume; StepIndex: integer);
    // Sigma[StepIndex] as UniPC reads it: diffusers moves a first sigma of 1
    // to float32(1 - 1e-6), so lambda = ln((1 - sigma)/sigma) stays finite.
    function UniPCSigma(StepIndex: integer): double;
    function GetNumSteps: integer;
    function GetSigma(StepIndex: integer): double;
    function GetTimestep(StepIndex: integer): double;
  public
    // pShift is the static shift, ignored when pUseDynamicShifting is true
    // (then SetSigmas needs mu). pShiftTerminal = 0 disables the terminal stretch.
    constructor Create(pNumTrainTimesteps: integer = 1000;
      pShift: double = 1.0; pUseDynamicShifting: boolean = false;
      pTimeShiftType: TNNetFlowTimeShift = ftsExponential;
      pShiftTerminal: double = 0.0);
    destructor Destroy; override;
    // Reads a diffusers FlowMatchEulerDiscreteScheduler scheduler_config.json;
    // raises on a setting this class does not implement (Karras sigmas, ...).
    constructor CreateFromDiffusersConfig(const FileName: string);
    // diffusers calculate_shift: mu linear in the image token count through
    // (BaseSeqLen, BaseShift) and (MaxSeqLen, MaxShift); extrapolates outside.
    class function CalculateShift(ImageSeqLen: integer;
      BaseSeqLen: integer = 256; MaxSeqLen: integer = 4096;
      BaseShift: double = 0.5; MaxShift: double = 1.15): double;
    // CalculateShift through this scheduler's BaseImageSeqLen/MaxImageSeqLen
    // and BaseShift/MaxShift (the pipelines' mu for ImageSeqLen image tokens).
    function ShiftForImageSeqLen(ImageSeqLen: integer): double;
    // Applies the dynamic time shift of type TimeShiftType to one sigma in (0, 1].
    function TimeShift(Mu, Sigma: double): double;
    // diffusers set_timesteps(sigmas=...): shift (static or dynamic with Mu),
    // stretch to ShiftTerminal, append the final sigma 0. Calls ResetMultistep.
    procedure SetSigmas(const Sigmas: array of double; Mu: double = 0.0);
    // SetSigmas with linspace(1, 1/NumSteps, NumSteps), the input sigmas the
    // FLUX and Qwen-Image pipelines pass.
    procedure SetTimesteps(NumSteps: integer; Mu: double = 0.0);
    // One Sampler step in place from Sigma[StepIndex] to Sigma[StepIndex+1];
    // Velocity is the model output at Timestep[StepIndex].
    procedure Step(Sample, Velocity: TNNetVolume; StepIndex: integer);
    // Forgets the multistep history, so the next Step is first order.
    procedure ResetMultistep;
    // diffusers img2img get_timesteps: the first step that runs at Strength,
    // int(N - min(N * Strength, N)); raises when no step runs.
    class function Img2ImgStartStep(NumSteps: integer;
      Strength: double): integer;
    // diffusers scale_noise (the img2img start): Noise becomes
    // Sigma[StepIndex] * Noise + (1 - Sigma[StepIndex]) * Sample.
    procedure ScaleNoise(Noise, Sample: TNNetVolume; StepIndex: integer);

    property NumSteps: integer read GetNumSteps;
    // Sigma[0..NumSteps]; Sigma[NumSteps] = 0.
    property Sigma[StepIndex: integer]: double read GetSigma;
    // Timestep[0..NumSteps-1] = Sigma * NumTrainTimesteps (the model input).
    property Timestep[StepIndex: integer]: double read GetTimestep;
    property NumTrainTimesteps: integer read FNumTrainTimesteps;
    property Shift: double read FShift;
    property UseDynamicShifting: boolean read FUseDynamicShifting;
    property TimeShiftType: TNNetFlowTimeShift read FTimeShiftType;
    // Setting it calls ResetMultistep.
    property Sampler: TNNetFlowSampler read FSampler write SetSampler;
    // fsUniPC options: diffusers solver_type (default ustBh2, the Wan setting)
    // and the UniC corrector (default true; false is disable_corrector on
    // every step, the plain UniP predictor).
    property UniPCSolverType: TNNetUniPCSolverType read FUniPCSolverType
      write FUniPCSolverType;
    property UniPCCorrector: boolean read FUniPCCorrector write FUniPCCorrector;
    property ShiftTerminal: double read FShiftTerminal;
    property BaseImageSeqLen: integer read FBaseImageSeqLen;
    property MaxImageSeqLen: integer read FMaxImageSeqLen;
    property BaseShift: double read FBaseShift;
    property MaxShift: double read FMaxShift;
  end;

// The sampler for a diffusers scheduler_config.json (its text) by _class_name:
// DDIMScheduler -> smDDIM, PNDMScheduler -> smPNDM, EulerDiscreteScheduler ->
// smEuler, EulerAncestralDiscreteScheduler -> smEulerAncestral (eta 1),
// DPMSolverMultistepScheduler -> smDPMSolverPP2M, LCMScheduler -> smLCM (needs
// a TNNetLCMScheduler). Any other class samples with
// smDDIM: its betas, spacing and prediction type still load, so it stays
// usable. Notice is '' or says what differs from the named scheduler (the
// fallback, or options such as solver_order 3 this unit does not implement).
function DiffusersSchedulerSampler(const ConfigJson: string;
  out Notice: string): TNNetSamplerMethod;

// True when the config is an Euler or DPM-Solver multistep scheduler with
// use_karras_sigmas (follow it with BuildSigmaSchedule).
function DiffusersSchedulerUsesKarrasSigmas(const ConfigJson: string): boolean;

// Blended (no-retrain) inpainting composite, in place on Latents:
// Latents := Known + Mask * (Latents - Known). Mask is 1 in the hole (follow
// the sampler) and 0 where Known is kept; a 0 keeps Known exactly. All three
// have the same size.
procedure InpaintBlend(Latents, Known, Mask: TNNetVolume);

// Tile starts along one axis every Stride from 0. Plain: the last tile is
// clamped to end at Extent. Circular: ceil(Extent/Stride) tiles, wrapping.
procedure ComputeTileStarts(Extent, TileLen, Stride: integer;
  out Starts: TNeuralIntegerArray; Circular: boolean = false);

type
  // Fills WindowOutput from WindowInput for window WindowIdx (same token
  // layout; WindowOutput may change depth and must keep the token count).
  TMultiDiffusionPredictEvent = procedure(WindowIdx: integer;
    WindowInput, WindowOutput: TNNetVolume) of object;

  // MultiDiffusion (Bar-Tal et al. 2023, diffusers
  // StableDiffusionPanoramaPipeline): a denoiser of a fixed window size runs
  // on overlapping windows of a larger row-major token canvas; each canvas
  // cell gets the mean of the windows covering it. Canvases are row-major
  // (CanvasWidth*CanvasHeight,1,Depth) or (CanvasWidth,CanvasHeight,Depth);
  // windows take the same form. Optional wrap in X for 360-degree
  // panoramas. Coded by Claude (AI).
  TMultiDiffusionWindows = class(TObject)
  private
    FCanvasWidth, FCanvasHeight, FWindowWidth, FWindowHeight: integer;
    FCircularX: boolean;
    FStartsX, FStartsY: TNeuralIntegerArray;
    // 1 / (windows covering the cell), one per canvas cell.
    FInvCoverage: TNeuralFloatDynArr;
    FWindowInput, FWindowOutput, FAccumulator: TNNetVolume;
    function GetWindowCount(): integer;
    function GetWindowX(WindowIdx: integer): integer;
    function GetWindowY(WindowIdx: integer): integer;
    function GetCoverage(CellX, CellY: integer): integer;
    procedure CheckCanvas(Canvas: TNNetVolume; const Caller: string);
    procedure ShapeWindow(Canvas, Window: TNNetVolume; Depth: integer);
    // IsGather: Window := Canvas's window cells; else Canvas cells += Window.
    procedure TransferWindow(Canvas, Window: TNNetVolume; WindowIdx: integer;
      IsGather: boolean);
  public
    constructor Create();
    destructor Destroy(); override;
    // Windows of WindowWidth x WindowHeight cells (each side clamped to the
    // canvas) every StrideX / StrideY cells; 1 <= Stride <= window side.
    procedure Plan(CanvasWidth, CanvasHeight, WindowWidth, WindowHeight,
      StrideX, StrideY: integer; CircularX: boolean = false);
    // Window := window WindowIdx of Canvas (resized to the window).
    procedure GatherWindow(Canvas: TNNetVolume; WindowIdx: integer;
      Window: TNNetVolume);
    // Zeroes the canvas accumulator, Depth values per cell.
    procedure ClearAccumulator(Depth: integer);
    // Adds Window into the accumulator cells of window WindowIdx.
    procedure AccumulateWindow(WindowIdx: integer; Window: TNNetVolume);
    // CanvasOutput := accumulator / coverage, shaped like Canvas.
    procedure NormalizeInto(Canvas, CanvasOutput: TNNetVolume);
    // Every window: gather from Canvas, Predict, accumulate; then
    // NormalizeInto(Canvas, CanvasOutput); one window equals Predict(Canvas).
    procedure PredictCanvas(Canvas, CanvasOutput: TNNetVolume;
      Predict: TMultiDiffusionPredictEvent);
    property CanvasWidth: integer read FCanvasWidth;
    property CanvasHeight: integer read FCanvasHeight;
    property WindowWidth: integer read FWindowWidth;
    property WindowHeight: integer read FWindowHeight;
    property CircularX: boolean read FCircularX;
    property WindowCount: integer read GetWindowCount;
    // Top-left cell of window WindowIdx (row-major over the starts grid).
    property WindowX[WindowIdx: integer]: integer read GetWindowX;
    property WindowY[WindowIdx: integer]: integer read GetWindowY;
    // Windows covering cell (CellX, CellY).
    property Coverage[CellX, CellY: integer]: integer read GetCoverage;
  end;

implementation

uses
  jsonparser;

const
  // Stand-in for an alpha_bar of exactly 0 in reciprocals (diffusers uses 2**-24).
  cMinAlphaBar = 5.9604644775390625e-8;

// DPM-Solver++(2M) midpoint weight w of the previous x0 for step size H after a
// step of HLast: D = (1 + w)*x0 - w*x0_prev, w = 0.5/r, r = HLast/H.
function DPMSolverPP2MHistoryWeight(HLast, H: double): double;
var
  StepRatio: double;
begin
  if Abs(H) < 1e-12 then StepRatio := 1.0 else StepRatio := HLast / H;
  Result := 0.5 / StepRatio;
end;

// UniPC B(h) weights (Zhao et al. 2023; diffusers multistep_uni_p_bh_update and
// multistep_uni_c_bh_update with predict_x0) of one hop from s0 to t:
//   x_t = (sigma_t/sigma_s0)*x_s0 + WeightM0*m0 + WeightM1*m1 + WeightMT*m_t,
// m0 the x0 at s0, m1 the x0 one step earlier at s1 (read when Order = 2),
// m_t the x0 at t (corrector only). HH = lambda_s0 - lambda_t (-h, < 0),
// Rk = (lambda_s1 - lambda_s0)/h. The order-2 predictor takes rho = 1/2, the
// order-1 corrector rho = 1/2, the order-2 corrector solves
// [[1, 1], [Rk, 1]] * rho = b.
procedure UniPCWeights(HH, Rk, AlphaT: double; Order: integer;
  IsCorrector: boolean; SolverType: TNNetUniPCSolverType;
  out WeightM0, WeightM1, WeightMT: double);
var
  HPhi1, Bh, DataScale, CorrectionScale, PhiRatio, B1, B2, DetR: double;
  RhoHistory, RhoCurrent, HistoryD1Scale: double;
begin
  HPhi1 := Exp(HH) - 1.0;
  if SolverType = ustBh1 then Bh := HH else Bh := HPhi1;
  DataScale := AlphaT * HPhi1;
  CorrectionScale := AlphaT * Bh;
  WeightM0 := -DataScale;
  WeightM1 := 0;
  WeightMT := 0;
  if IsCorrector then
  begin
    if Order >= 2 then
    begin
      PhiRatio := HPhi1 / HH - 1.0;
      B1 := PhiRatio / Bh;
      B2 := (PhiRatio / HH - 0.5) * 2.0 / Bh;
      DetR := 1.0 - Rk;
      RhoHistory := (B1 - B2) / DetR;
      RhoCurrent := (-Rk * B1 + B2) / DetR;
      // corr = RhoHistory*(m1 - m0)/Rk + RhoCurrent*(m_t - m0).
      HistoryD1Scale := CorrectionScale * RhoHistory / Rk;
      WeightM0 := -DataScale + HistoryD1Scale + CorrectionScale * RhoCurrent;
      WeightM1 := -HistoryD1Scale;
      WeightMT := -CorrectionScale * RhoCurrent;
    end
    else
    begin
      WeightM0 := -DataScale + 0.5 * CorrectionScale;
      WeightMT := -0.5 * CorrectionScale;
    end;
  end
  else if Order >= 2 then
  begin
    HistoryD1Scale := 0.5 * CorrectionScale / Rk;
    WeightM0 := -DataScale + HistoryD1Scale;
    WeightM1 := -HistoryD1Scale;
  end;
end;

constructor TNNetDiffusionScheduler.Create(pT: integer;
  pSchedule: TNNetBetaSchedule; pPrediction: TNNetPredictionType;
  Beta1, BetaT, CosineS: TNeuralFloat; pRescaleZeroTerminalSNR: boolean;
  pFinalAlphaBarIsOne: boolean);
begin
  inherited Create;
  if pT < 1 then raise Exception.Create('Diffusion T must be >= 1.');
  if pRescaleZeroTerminalSNR and (pT < 2) then
    raise Exception.Create('Zero-terminal-SNR rescale needs T >= 2.');
  FT := pT;
  FSchedule := pSchedule;
  FPrediction := pPrediction;
  FRescaleZeroTerminalSNR := pRescaleZeroTerminalSNR;
  FStepsOffset := 0;
  FConfigSpacing := tsUniform;
  SetLength(FBeta, FT + 1);
  SetLength(FAlpha, FT + 1);
  SetLength(FAlphaBar, FT + 1);
  SetLength(FSqrtAlphaBar, FT + 1);
  SetLength(FSqrtOneMinusAlphaBar, FT + 1);
  SetLength(FLambda, FT + 1);
  SetLength(FSigma, FT + 1);
  SetLength(FInvSqrtAlphaBar, FT + 1);
  BuildTables(Beta1, BetaT, CosineS);
  if pFinalAlphaBarIsOne then FFinalAlphaBar := 1.0
  else FFinalAlphaBar := FAlphaBar[1];
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

constructor TNNetDiffusionScheduler.CreateFromDiffusersConfig(
  const FileName: string; ForwardProcessOnly: boolean);
var
  JsonText: TStringList;
  Root, TrainedBetas: TJSONData;
  Obj: TJSONObject;
  ScheduleName, PredictionName, SpacingName, SchedulerClass: string;
  Schedule: TNNetBetaSchedule;
  Prediction: TNNetPredictionType;
  Spacing: TNNetTimestepSpacing;
  ClipSampleDefault: boolean;

  procedure Refuse(const Setting: string);
  begin
    raise Exception.Create('Diffusion scheduler config ' + FileName + ': ' +
      Setting + ' is not implemented.');
  end;

begin
  JsonText := TStringList.Create;
  Root := nil;
  try
    JsonText.LoadFromFile(FileName);
    Root := GetJSON(JsonText.Text);
    if not (Root is TJSONObject) then
      raise Exception.Create('Diffusion scheduler config ' + FileName +
        ' is not a JSON object.');
    Obj := TJSONObject(Root);
    // clip_sample exists (default true) only on the DDIM/DDPM schedulers.
    SchedulerClass := Obj.Get('_class_name', '');
    ClipSampleDefault := (SchedulerClass = '') or
      (SchedulerClass = 'DDIMScheduler') or (SchedulerClass = 'DDPMScheduler');
    if Obj.Get('clip_sample', ClipSampleDefault) and not ForwardProcessOnly then
      Refuse('"clip_sample" = true');
    if Obj.Get('thresholding', false) then Refuse('"thresholding" = true');
    TrainedBetas := Obj.Find('trained_betas');
    if (TrainedBetas <> nil) and not (TrainedBetas.JSONType = jtNull) then
      Refuse('"trained_betas"');
    ScheduleName := Obj.Get('beta_schedule', 'linear');
    if ScheduleName = 'linear' then Schedule := dsLinear
    else if ScheduleName = 'scaled_linear' then Schedule := dsScaledLinear
    else if ScheduleName = 'squaredcos_cap_v2' then Schedule := dsCosine
    else Refuse('beta_schedule "' + ScheduleName + '"');
    PredictionName := Obj.Get('prediction_type', 'epsilon');
    if PredictionName = 'epsilon' then Prediction := dpEps
    else if PredictionName = 'v_prediction' then Prediction := dpV
    else Refuse('prediction_type "' + PredictionName + '"');
    SpacingName := Obj.Get('timestep_spacing', 'leading');
    if SpacingName = 'trailing' then Spacing := tsTrailing
    else if SpacingName = 'leading' then Spacing := tsLeading
    // diffusers linspace(0, T-1, N).round() is tsUniform.
    else if SpacingName = 'linspace' then Spacing := tsUniform
    else Refuse('timestep_spacing "' + SpacingName + '"');
    Create(Obj.Get('num_train_timesteps', 1000), Schedule, Prediction,
      Obj.Get('beta_start', 0.0001), Obj.Get('beta_end', 0.02), 0.008,
      Obj.Get('rescale_betas_zero_snr', false),
      Obj.Get('set_alpha_to_one', true));
    FConfigSpacing := Spacing;
    FStepsOffset := Obj.Get('steps_offset', 0);
    ReadDiffusersConfig(Obj);
  finally
    Root.Free;
    JsonText.Free;
  end;
end;

procedure TNNetDiffusionScheduler.ReadDiffusersConfig(Config: TJSONObject);
begin
end;

destructor TNNetDiffusionScheduler.Destroy;
begin
  if Assigned(FPrevX0) then FPrevX0.Free;
  if Assigned(FUniPrevSample) then FUniPrevSample.Free;
  if Assigned(FUniHistX0[0]) then FUniHistX0[0].Free;
  if Assigned(FUniHistX0[1]) then FUniHistX0[1].Free;
  if Assigned(FEpsScratch) then FEpsScratch.Free;
  if Assigned(FX0HatScratch) then FX0HatScratch.Free;
  FPndmEts[0].Free;
  FPndmEts[1].Free;
  FPndmEts[2].Free;
  FPndmEts[3].Free;
  FPndmCurSample.Free;
  FPndmMix.Free;
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
  if FRescaleZeroTerminalSNR then RescaleToZeroTerminalSNR;
  // Derived per-timestep tables: lambda_t = 0.5*(ln ab - ln(1-ab)),
  // sigma_t = sqrt((1-ab)/ab) and 1/sqrt(ab), paid once here instead of per
  // sampling step. An ab of 0 (zero terminal SNR) is read as cMinAlphaBar, as
  // diffusers' Euler/DPM/UniPC schedulers do, so the reciprocals stay finite.
  for i := 0 to FT do
  begin
    abCur := FAlphaBar[i];
    if abCur <= 0.0 then abCur := cMinAlphaBar;
    if abCur >= 1.0 then
    begin
      FSigma[i] := 0;          // clean anchor
      FLambda[i] := 0;         // unused: Lambda() falls back to the direct form
      FInvSqrtAlphaBar[i] := 1;
    end
    else
    begin
      FSigma[i] := Sqrt((1.0 - abCur) / abCur);
      FLambda[i] := 0.5 * (Ln(abCur) - Ln(1.0 - abCur));
      FInvSqrtAlphaBar[i] := 1.0 / Sqrt(abCur);
    end;
  end;
end;

procedure TNNetDiffusionScheduler.RescaleToZeroTerminalSNR;
var
  i: integer;
  SqrtAlphaBar: array of double;
  prod, sqrtFirst, sqrtLast, shiftScale, abPrev, abCur: double;
begin
  // Double-precision cumprod of the base betas, then sqrt(ab).
  SetLength(SqrtAlphaBar, FT + 1);
  prod := 1.0;
  for i := 1 to FT do
  begin
    prod := prod * (1.0 - FBeta[i]);
    SqrtAlphaBar[i] := Sqrt(prod);
  end;
  sqrtFirst := SqrtAlphaBar[1];
  sqrtLast := SqrtAlphaBar[FT];
  if sqrtFirst <= sqrtLast then
    raise Exception.Create('Zero-terminal-SNR rescale needs a decreasing alpha_bar.');
  shiftScale := sqrtFirst / (sqrtFirst - sqrtLast);
  abPrev := 1.0;
  for i := 1 to FT do
  begin
    // Shift so the last entry is 0, scale so the first is unchanged, square.
    if i = FT then abCur := 0.0
    else abCur := Sqr((SqrtAlphaBar[i] - sqrtLast) * shiftScale);
    FAlpha[i] := abCur / abPrev;
    FBeta[i] := 1.0 - FAlpha[i];
    FAlphaBar[i] := abCur;
    FSqrtAlphaBar[i] := Sqrt(abCur);
    FSqrtOneMinusAlphaBar[i] := Sqrt(1.0 - abCur);
    abPrev := abCur;
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
  // lambda = log( sqrt(ab) / sqrt(1-ab) ) = 0.5*(log ab - log(1-ab)), tabulated
  // in BuildTables for every ab < 1 (ab = 0 reads as cMinAlphaBar).
  if ab < 1.0 then Result := FLambda[Tt]
  else Result := 0.5 * (Ln(ab) - Ln(1.0 - ab));
end;

function TNNetDiffusionScheduler.SigmaOf(Tt: integer): TNeuralFloat;
begin
  // sigma_t = sqrt((1-ab_t)/ab_t), tabulated in BuildTables (0 at the clean
  // anchor Tt=0, cMinAlphaBar-based where ab_t = 0).
  Result := FSigma[Tt];
end;

function TNNetDiffusionScheduler.SNRWeight(Tt: integer;
  Gamma: TNeuralFloat): TNeuralFloat;
var
  ab, snr, clamped: double;
begin
  // SNR(t) = exp(2*Lambda(t)) = ab_t/(1-ab_t) (Lambda is the half-log-SNR the
  // DPM-Solver uses); the closed form skips the log/exp round trip.
  ab := FAlphaBar[Tt];
  snr := ab / (1.0 - ab);
  // min(SNR, Gamma); Gamma = +Inf leaves the clamp inactive (clamped = snr).
  if snr < Gamma then clamped := snr else clamped := Gamma;
  case FPrediction of
    dpV:
      // v-prediction: divide by (SNR + 1). Gamma = +Inf -> SNR/(SNR+1).
      Result := clamped / (snr + 1.0);
    else // dpEps
      // eps-prediction: min(SNR, Gamma)/SNR, written so SNR = 0 gives 1.
      if snr < Gamma then Result := 1.0 else Result := Gamma / snr;
  end;
end;

function TNNetDiffusionScheduler.SigmaToTimestep(TargetSigma: TNeuralFloat): integer;
var
  lo, hi, mid, best: integer;
begin
  // The schedule's sigma is monotone increasing in Tt, so a binary search for
  // the first table entry >= TargetSigma plus one neighbour comparison finds the
  // nearest match exactly, in O(log T) instead of a full scan.
  lo := 1; hi := FT;
  while lo < hi do
  begin
    mid := (lo + hi) shr 1;
    if FSigma[mid] < TargetSigma then lo := mid + 1 else hi := mid;
  end;
  best := lo;
  // On a tie the lower timestep wins, as an ascending scan would pick it.
  if (lo > 1) and (Abs(FSigma[lo - 1] - TargetSigma) <=
                   Abs(FSigma[lo] - TargetSigma)) then best := lo - 1;
  while (best > 1) and (FSigma[best - 1] = FSigma[best]) do Dec(best);
  Result := best;
end;

function TNNetDiffusionScheduler.BuildTimestepSchedule(NumSteps: integer;
  Spacing: TNNetTimestepSpacing): TNeuralIntegerArray;
var
  k, Tt: integer;
  NumStepsM1, FTM1, LeadingStride: integer;
  sigMin, sigMax, invRho, frac, targetSigma: double;
  pMax, pMin, trailingStride: double;
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
        // A single step sits at sigma_max; the general case interpolates
        // (rule #20: the NumSteps test is loop-invariant).
        if NumSteps = 1 then Result[0] := SigmaToTimestep(Power(pMax, cRho))
        else
          for k := 0 to NumStepsM1 do
          begin
            frac := k / NumStepsM1;
            // k=0 -> sigma_max (most noise, highest Tt); k=n-1 -> sigma_min.
            targetSigma := Power(pMax + frac * (pMin - pMax), cRho);
            Tt := SigmaToTimestep(targetSigma);
            // Index 0 of Result is the FIRST (highest-noise) step.
            Result[k] := Tt;
          end;
        Result[NumSteps] := 0; // clean-image anchor.
      end;
    tsTrailing:
      begin
        // Tt = round(T - k*T/N); diffusers' 0-based t is Tt-1. Both use
        // round-half-to-even. Clamped to 1 when N > T would reach 0.
        trailingStride := FT / NumSteps;
        for k := 0 to NumStepsM1 do
        begin
          Tt := Round(FT - k * trailingStride);
          if Tt < 1 then Tt := 1;
          Result[k] := Tt;
        end;
        Result[NumSteps] := 0;
      end;
    tsLeading:
      begin
        // diffusers: (arange(N) * (T div N))[::-1] + steps_offset, 0-based.
        LeadingStride := FT div NumSteps;
        for k := 0 to NumStepsM1 do
        begin
          Tt := (NumStepsM1 - k) * LeadingStride + FStepsOffset + 1;
          if Tt < 1 then Tt := 1
          else if Tt > FT then Tt := FT;
          Result[k] := Tt;
        end;
        Result[NumSteps] := 0;
      end;
    else // tsUniform
      begin
        // Descending: Result[0] is the highest timestep. The NumSteps test and
        // the FT-1 span are loop-invariant (rules #20/#5).
        if NumSteps = 1 then Result[0] := FT
        else
        begin
          FTM1 := FT - 1;
          for k := 0 to NumStepsM1 do
          begin
            Tt := 1 + Round((NumStepsM1 - k) * FTM1 / NumStepsM1);
            Result[k] := Tt;
          end;
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
begin
  PredictX0At(Xt, RawPred, X0Out, FSqrtAlphaBar[Tt],
    FSqrtOneMinusAlphaBar[Tt], FInvSqrtAlphaBar[Tt]);
end;

procedure TNNetDiffusionScheduler.PredictX0At(Xt, RawPred,
  X0Out: TNNetVolume; SqrtAb, SqrtOneMinusAb, InvSqrtAb: TNeuralFloat);
var
  RawCoef, XtCoef: TNeuralFloat;
begin
  // Both forms are X0Out := RawCoef*RawPred + XtCoef*Xt (rule #13), so X0Out
  // may alias RawPred. v needs no 1/sqrt(ab), which keeps ab_t = 0 exact.
  if FPrediction = dpV then
  begin
    RawCoef := -SqrtOneMinusAb;
    XtCoef := SqrtAb;
  end
  else
  begin
    XtCoef := InvSqrtAb;
    RawCoef := -SqrtOneMinusAb * XtCoef;
  end;
  if X0Out <> RawPred then X0Out.Copy(RawPred);
  X0Out.Mul(RawCoef);
  X0Out.MulAdd(XtCoef, Xt);
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
  // Hoist the per-element Assigned() tests (rule #5 / no-nil-tests-in-hot-loops)
  // and unswitch the loop on them (rule #20).
  hasPre := Assigned(PreSampledNoise);
  hasNoiseOut := Assigned(NoiseOut);
  if hasPre then
  begin
    // Supplied noise: the whole update is a uniform affine map (rule #13).
    if hasNoiseOut then NoiseOut.Copy(PreSampledNoise);
    if Xt = PreSampledNoise then
    begin
      // Xt IS the noise: scale it in place before folding X0 in.
      Xt.Mul(somab);
      Xt.MulAdd(sab, X0);
    end
    else
    begin
      if Xt <> X0 then Xt.Copy(X0);
      Xt.Mul(sab);
      Xt.MulAdd(somab, PreSampledNoise);
    end;
  end
  else if hasNoiseOut then
    for i := 0 to X0SizeM1 do
    begin
      eps := RandG(0, 1);
      NoiseOut.FData[i] := eps;
      Xt.FData[i] := sab * X0.FData[i] + somab * eps;
    end
  else
    for i := 0 to X0SizeM1 do
      // Fresh noise, nothing to report: one transcendental per element.
      Xt.FData[i] := sab * X0.FData[i] + somab * RandG(0, 1);
end;

procedure TNNetDiffusionScheduler.RawPredFromX0(Xt, X0, RawOut: TNNetVolume;
  Tt: integer);
var
  X0Coef, XtCoef, InvSomab: TNeuralFloat;
begin
  // eps = (x_t - sqrt(ab)*x0)/sqrt(1-ab); v = (sqrt(ab)*x_t - x0)/sqrt(1-ab).
  InvSomab := 1.0 / FSqrtOneMinusAlphaBar[Tt];
  if FPrediction = dpV then
  begin
    X0Coef := -InvSomab;
    XtCoef := FSqrtAlphaBar[Tt] * InvSomab;
  end
  else
  begin
    X0Coef := -FSqrtAlphaBar[Tt] * InvSomab;
    XtCoef := InvSomab;
  end;
  if RawOut <> X0 then RawOut.Copy(X0);
  RawOut.Mul(X0Coef);
  RawOut.MulAdd(XtCoef, Xt);
end;

class procedure TNNetDiffusionScheduler.ApplyCFG(EpsCond, EpsUncond,
  Dst: TNNetVolume; W: TNeuralFloat);
begin
  // Dst := (1-W)*EpsUncond + W*EpsCond (rule #13). The copy-first bulk form is
  // WRONG when Dst aliases EpsCond (Copy would clobber EpsCond before it is
  // read), so scale EpsCond in place instead and fold EpsUncond in afterwards.
  if Dst = EpsCond then
  begin
    Dst.Mul(W);
    Dst.MulAdd(1.0 - W, EpsUncond);
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
  stdCond: TNeuralFloat;
begin
  if GuidanceRescale = 0 then
  begin
    ApplyCFG(EpsCond, EpsUncond, Dst, W);
    exit;
  end;
  // Before the mix: Dst may alias EpsCond.
  stdCond := EpsCond.GetStdDeviation();
  ApplyCFG(EpsCond, EpsUncond, Dst, W);
  RescaleGuidance(Dst, stdCond, GuidanceRescale);
end;

class procedure TNNetDiffusionScheduler.RescaleGuidance(Dst: TNNetVolume;
  StdCond, GuidanceRescale: TNeuralFloat);
var
  stdGuided: TNeuralFloat;
begin
  // Rescale the over-saturated guided result toward the conditional's std, then
  // blend back by GuidanceRescale (diffusers rescale_noise_cfg, Lin et al. 2023).
  if GuidanceRescale = 0 then exit;
  stdGuided := Dst.GetStdDeviation();
  if stdGuided = 0 then exit;
  // Whole expression is Dst[i]*K with invariant K (rule #5/#13) -> one bulk scale.
  Dst.Mul(GuidanceRescale * (StdCond / stdGuided) + (1 - GuidanceRescale));
end;

procedure TNNetDiffusionScheduler.ResetMultistep;
begin
  FHasPrev := false;
  FUniThisOrder := 1;
  FUniLowerOrderNums := 0;
  FUniHistLen := 0;
  FPndmCounter := 0;
  FPndmEtsLen := 0;
end;

procedure TNNetDiffusionScheduler.Step(Xt, RawPred: TNNetVolume;
  Tt, TtPrev: integer; Method: TNNetSamplerMethod; Eta: TNeuralFloat;
  Noise: TNNetVolume);
var
  i: integer;
  XtSizeM1: integer;
  Eps: TNNetVolume;
  invSqrtAlpha, sigma, z: TNeuralFloat;
  abT, abPrev, sqrtAbPrev, dirCoef: TNeuralFloat;
  // DPM-Solver++ locals.
  lamT, lamPrev, h: TNeuralFloat;
  // UniPC locals (order-2 bh2, predict_x0).
  alphaT, sigmaTuni, sigmaS0, rk: TNeuralFloat;
  WeightM0, WeightM1, WeightMT: double;
  cOrder, pOrder: integer;
  // Loop-invariant hoists (rule #5): per-step coefficients pulled out of the
  // per-element loops below.
  needNoise: boolean;
  somabT, cA, cB, kX: TNeuralFloat;
  kA, cE: TNeuralFloat;    // per-step affine coeffs for the sampler loops (#5)
begin
  if Method = smLCM then
    raise Exception.Create('Step: smLCM needs a TNNetLCMScheduler.');
  // Persistent lazily-sized eps scratch (rule #17). Step and PredictX0 never
  // run nested, so they safely share FEpsScratch.
  if not Assigned(FEpsScratch) then FEpsScratch := TNNetVolume.Create(Xt)
  else FEpsScratch.ReSize(Xt);
  Eps := FEpsScratch;
  // The other updates work from eps; v-prediction is converted up front.
  // DDPM, DDIM, DPM++, UniPC and Euler read RawPred directly or through PredictX0.
  if Method = smPNDM then
  begin
    StepPNDM(Xt, RawPred, Tt, TtPrev);
    Exit;
  end;
  if not (Method in [smDDPM, smDDIM, smDPMSolverPP2M, smUniPC, smEulerAncestral,
    smEuler]) then
    ToEps(Xt, RawPred, Eps, Tt);
  // smEuler is the Eta = 0 (sigma_up = 0) case of the Euler-ancestral update.
  if Method = smEuler then Eta := 0;
  XtSizeM1 := Xt.Size - 1;
  case Method of
      smDDPM:
        begin
          // Posterior mean (Ho 2020 eq. 7) c0*x0 + c1*x_t, x0 from eps or v,
          // folded into kA*x_t + cE*RawPred (#5/#13) + sigma*z.
          somabT := FSqrtOneMinusAlphaBar[Tt];
          needNoise := Tt > 1;
          if needNoise then sigma := Sqrt(FBeta[Tt]) else sigma := 0;
          if somabT = 0 then
          begin
            // ab_t = 1 (Beta1 = 0): beta_t = 0 and x_t is already clean.
            kA := 1;
            cE := 0;
          end
          else if FPrediction = dpV then
          begin
            // c0 = sqrt(ab_prev)*beta_t/(1-ab_t), c1 = sqrt(a_t)*(1-ab_prev)/(1-ab_t)
            // with x0 = sqrt(ab_t)*x_t - somab_t*v: exact at ab_t = 0 (x0 = -v).
            cA := FSqrtAlphaBar[Tt - 1] * FBeta[Tt] / (somabT * somabT);
            cB := Sqrt(FAlpha[Tt]) * Sqr(FSqrtOneMinusAlphaBar[Tt - 1]) /
              (somabT * somabT);
            kA := cA * FSqrtAlphaBar[Tt] + cB;
            cE := -cA * somabT;
          end
          else
          begin
            // eps: 1/sqrt(a_t)*(x_t - beta_t/sqrt(1-ab_t)*eps); alpha_T floored as ab.
            invSqrtAlpha := 1.0 / Sqrt(Max(FAlpha[Tt], cMinAlphaBar));
            kA := invSqrtAlpha;
            cE := -invSqrtAlpha * FBeta[Tt] / somabT;
          end;
          if needNoise then
          begin
            for i := 0 to XtSizeM1 do
            begin
              z := RandG(0, 1);
              Xt.FData[i] := kA * Xt.FData[i] + cE * RawPred.FData[i] + sigma * z;
            end;
          end
          else
          begin
            Xt.Mul(kA);
            Xt.MulAdd(cE, RawPred);
          end;
        end;
      smDDIM:
        begin
          // x0_pred = (x_t - sqrt(1-ab_t)*eps)/sqrt(ab_t)
          // x_{TtPrev} = sqrt(ab_prev)*x0 + sqrt(1-ab_prev - sigma^2)*eps + sigma*z
          // sigma = eta*sqrt((1-ab_prev)/(1-ab_t))*sqrt(1-ab_t/ab_prev).
          abT := FAlphaBar[Tt];
          if TtPrev = 0 then abPrev := FFinalAlphaBar
          else abPrev := FAlphaBar[TtPrev];
          sqrtAbPrev := Sqrt(abPrev);
          if (Eta > 0) and (abPrev < 1.0) then
            sigma := Eta * Sqrt((1.0 - abPrev) / (1.0 - abT)) *
                     Sqrt(1.0 - abT / abPrev)
          else
            sigma := 0;
          dirCoef := Sqrt(Max(0.0, 1.0 - abPrev - sigma * sigma));
          somabT := FSqrtOneMinusAlphaBar[Tt];
          // #5/#13: the update is kA*x_t + cE*RawPred (+ sigma*z). For v,
          //   x0 = sqrt(ab_t)*x_t - somab_t*v, eps = somab_t*x_t + sqrt(ab_t)*v,
          // which has no 1/sqrt(ab_t) and so stays exact at ab_t = 0.
          if FPrediction = dpV then
          begin
            kA := sqrtAbPrev * FSqrtAlphaBar[Tt] + dirCoef * somabT;
            cE := dirCoef * FSqrtAlphaBar[Tt] - sqrtAbPrev * somabT;
          end
          else
          begin
            kA := sqrtAbPrev * FInvSqrtAlphaBar[Tt];
            cE := dirCoef - kA * somabT;
          end;
          if sigma > 0 then
          begin
            for i := 0 to XtSizeM1 do
            begin
              z := RandG(0, 1);
              Xt.FData[i] := kA * Xt.FData[i] + cE * RawPred.FData[i] + sigma * z;
            end;
          end
          else
          begin
            Xt.Mul(kA);
            Xt.MulAdd(cE, RawPred);
          end;
        end;
      smEulerAncestral, smEuler:
        begin
          // Euler-ancestral ("Euler a", Karras et al. 2022) in the VE sigma
          // parameterisation sigma = sqrt((1-ab)/ab). With x0 from PredictX0
          // (exact for v at ab_t = 0) and s1 = Sigma(Tt), s2 = Sigma(TtPrev):
          //   sigma_up   = eta * sqrt( (s2^2*(s1^2 - s2^2)) / s1^2 )   (injected)
          //   sigma_down = sqrt( max(0, s2^2 - sigma_up^2) )          (Euler drift)
          //   x = sqrt(ab_prev)*(x0 + (sigma_down/s1)*(x_t/sqrt(ab_t) - x0)
          //       + sigma_up*z).
          PredictX0(Xt, RawPred, Eps, Tt);
          if TtPrev = 0 then
          begin
            // Final hop: emit the denoised x0 directly.
            Xt.Copy(Eps);
            Exit;
          end;
          EulerAncestralUpdate(Xt, Eps, Noise, SigmaOf(Tt), SigmaOf(TtPrev),
            FInvSqrtAlphaBar[Tt], FSqrtAlphaBar[TtPrev], Eta);
        end;
      smDPMSolverPP2M:
        begin
          // DPM-Solver++(2M) in the data-prediction (x0) form (Lu et al. 2022,
          // arXiv:2211.01095), as diffusers DPMSolverMultistepScheduler
          // (dpmsolver++, midpoint, final_sigmas_type "zero"). With
          // alpha = sqrt(ab), sigma = sqrt(1-ab), lambda = log(alpha/sigma)
          // (it INCREASES as Tt decreases) and h = lambda_{TtPrev} - lambda_t:
          //   x_{TtPrev} = (sigma_prev/sigma_t)*x_t - alpha_prev*(exp(-h)-1)*D,
          // D = x0_cur on the first step, else the midpoint 2nd-order
          //   D = (1+1/(2r))*x0_cur - (1/(2r))*x0_prev, r = h_last/h,
          // h_last = lambda_t - lambda of the previous step.
          PredictX0(Xt, RawPred, Eps, Tt);   // Eps holds x0_cur from here on.
          lamT := Lambda(Tt);
          if TtPrev = 0 then
          begin
            // Final hop to sigma = 0 is first order (diffusers lower_order_final):
            // alpha_prev = 1, sigma_prev = 0, h = +inf, so x = x0_cur.
            Xt.Copy(Eps);
            FHasPrev := false;
            Exit;
          end;
          kX := FSqrtOneMinusAlphaBar[TtPrev] / FSqrtOneMinusAlphaBar[Tt];
          DPMSolverPP2MUpdate(Xt, Eps, lamT, Lambda(TtPrev), kX,
            FAlphaBar[TtPrev]);
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
          // Current x0 prediction m_t into Eps.
          PredictX0(Xt, RawPred, Eps, Tt);
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
            cOrder := FUniThisOrder;
            // rk of the second history point (FUniHistX0[FUniHistLen-2]).
            if cOrder >= 2 then
              rk := (Lambda(FUniHistT[FUniHistLen - 2]) - lamPrev) / h
            else
              rk := 0;
            UniPCWeights(-h, rk, alphaT, cOrder, true, ustBh2,
              WeightM0, WeightM1, WeightMT);
            Xt.Copy(FUniPrevSample);
            Xt.Mul(sigmaTuni / sigmaS0);
            Xt.MulAdd(WeightM0, FUniHistX0[FUniHistLen - 1]);
            if cOrder >= 2 then Xt.MulAdd(WeightM1, FUniHistX0[FUniHistLen - 2]);
            Xt.MulAdd(WeightMT, Eps);
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
            if (pOrder >= 2) and (FUniHistLen >= 2) then
            begin
              // Predictor D1 from the previous history entry (m1 = hist[-2]).
              rk := (Lambda(FUniHistT[FUniHistLen - 2]) - lamT) / h;
              pOrder := 2;
            end
            else
            begin
              rk := 0;
              pOrder := 1;
            end;
            UniPCWeights(-h, rk, alphaT, pOrder, false, ustBh2,
              WeightM0, WeightM1, WeightMT);
            // Xt already holds FUniPrevSample.
            Xt.Mul(sigmaTuni / sigmaS0);
            Xt.MulAdd(WeightM0, Eps);
            if pOrder >= 2 then Xt.MulAdd(WeightM1, FUniHistX0[FUniHistLen - 2]);
          end;

          if FUniLowerOrderNums < 2 then Inc(FUniLowerOrderNums);
          FHasPrev := true;
        end;
    end;
end;

procedure TNNetDiffusionScheduler.StepPNDM(Xt, RawPred: TNNetVolume;
  Tt, TtPrev: integer);
var
  Timestep, PrevT: integer;
  Oldest, EpsVol: TNNetVolume;
  abT, abPrev, SampleCoef, EpsCoef: double;
begin
  // diffusers PNDMScheduler.step_plms. The target is Tt - T div N whatever
  // the spacing; the second step re-runs the FIRST hop from the first sample
  // with the mean of its two eps.
  if FPndmStride > 0 then PrevT := Tt - FPndmStride else PrevT := TtPrev;
  Timestep := Tt;
  if FPndmCounter <> 1 then
  begin
    // ets = ets[-3:] + [model_output]; the dropped volume is recycled.
    if FPndmEtsLen = 4 then
    begin
      Oldest := FPndmEts[0];
      FPndmEts[0] := FPndmEts[1];
      FPndmEts[1] := FPndmEts[2];
      FPndmEts[2] := FPndmEts[3];
      FPndmEts[3] := Oldest;
      FPndmEtsLen := 3;
    end;
    if not Assigned(FPndmEts[FPndmEtsLen]) then
      FPndmEts[FPndmEtsLen] := TNNetVolume.Create(RawPred)
    else FPndmEts[FPndmEtsLen].ReSize(RawPred);
    FPndmEts[FPndmEtsLen].Copy(RawPred);
    Inc(FPndmEtsLen);
  end
  else
  begin
    PrevT := Tt;
    if FPndmStride > 0 then Timestep := Min(Tt + FPndmStride, FT)
    else Timestep := FPndmFirstT;
  end;
  if not Assigned(FPndmMix) then FPndmMix := TNNetVolume.Create(Xt)
  else FPndmMix.ReSize(Xt);
  // Adams-Bashforth weights on the stored raw outputs, newest last (#13).
  if (FPndmEtsLen = 1) and (FPndmCounter = 0) then
  begin
    FPndmMix.Copy(RawPred);
    FPndmFirstT := Tt;
    if not Assigned(FPndmCurSample) then FPndmCurSample := TNNetVolume.Create(Xt)
    else FPndmCurSample.ReSize(Xt);
    FPndmCurSample.Copy(Xt);
  end
  else if (FPndmEtsLen = 1) and (FPndmCounter = 1) then
  begin
    FPndmMix.Copy(RawPred);
    FPndmMix.Add(FPndmEts[0]);
    FPndmMix.Mul(0.5);
    Xt.Copy(FPndmCurSample);
  end
  else if FPndmEtsLen = 2 then
  begin
    FPndmMix.Copy(FPndmEts[1]);
    FPndmMix.Mul(1.5);
    FPndmMix.MulAdd(-0.5, FPndmEts[0]);
  end
  else if FPndmEtsLen = 3 then
  begin
    FPndmMix.Copy(FPndmEts[2]);
    FPndmMix.Mul(23 / 12);
    FPndmMix.MulAdd(-16 / 12, FPndmEts[1]);
    FPndmMix.MulAdd(5 / 12, FPndmEts[0]);
  end
  else
  begin
    FPndmMix.Copy(FPndmEts[3]);
    FPndmMix.Mul(55 / 24);
    FPndmMix.MulAdd(-59 / 24, FPndmEts[2]);
    FPndmMix.MulAdd(37 / 24, FPndmEts[1]);
    FPndmMix.MulAdd(-9 / 24, FPndmEts[0]);
  end;
  // v is converted after the mix, against the sample the hop starts from.
  if FPrediction = dpV then
  begin
    ToEps(Xt, FPndmMix, FEpsScratch, Timestep);
    EpsVol := FEpsScratch;
  end
  else EpsVol := FPndmMix;
  // _get_prev_sample (PNDM paper eq. 9):
  //   x_prev = sqrt(ab_prev/ab_t)*x_t - (ab_prev - ab_t)*eps /
  //     (ab_t*sqrt(1-ab_prev) + sqrt(ab_t*(1-ab_t)*ab_prev)).
  // ab_t = 0 (zero terminal SNR) reads as cMinAlphaBar, as FInvSqrtAlphaBar.
  abT := Max(FAlphaBar[Timestep], cMinAlphaBar);
  if PrevT >= 1 then abPrev := FAlphaBar[PrevT] else abPrev := FFinalAlphaBar;
  SampleCoef := Sqrt(abPrev / abT);
  EpsCoef := -(abPrev - abT) /
    (abT * Sqrt(1.0 - abPrev) + Sqrt(abT * (1.0 - abT) * abPrev));
  Xt.Mul(SampleCoef);
  Xt.MulAdd(EpsCoef, EpsVol);
  Inc(FPndmCounter);
end;

function TNNetDiffusionScheduler.BuildSamplerSchedule(NumSteps: integer;
  Spacing: TNNetTimestepSpacing; Method: TNNetSamplerMethod):
  TNeuralIntegerArray;
var
  k, Tt, MaxStepPos, LeadingStride: integer;
  LinspaceStep: double;
begin
  if Method = smLCM then
    raise Exception.Create('BuildSamplerSchedule: smLCM needs a ' +
      'TNNetLCMScheduler.');
  ResetMultistep;
  FPndmStride := 0;
  if NumSteps < 1 then NumSteps := 1;
  MaxStepPos := NumSteps - 1;
  if (Method = smDPMSolverPP2M) and (Spacing in [tsLeading, tsUniform]) then
  begin
    // diffusers DPMSolverMultistepScheduler drops the last of NumSteps+1
    // points: leading (arange(N+1)*(T div (N+1)))[::-1][:-1] + steps_offset,
    // linspace linspace(0, T-1, N+1).round()[::-1][:-1] (numpy's i*step).
    SetLength(Result, NumSteps + 1);
    LeadingStride := FT div (NumSteps + 1);
    LinspaceStep := (FT - 1) / NumSteps;
    for k := 0 to MaxStepPos do
    begin
      if Spacing = tsLeading then
        Tt := (NumSteps - k) * LeadingStride + FStepsOffset + 1
      else
        Tt := 1 + Round((NumSteps - k) * LinspaceStep);
      if Tt < 1 then Tt := 1
      else if Tt > FT then Tt := FT;
      Result[k] := Tt;
    end;
    Result[NumSteps] := 0;
  end
  else
    Result := BuildTimestepSchedule(NumSteps, Spacing);
  if Method = smPNDM then
  begin
    // plms_timesteps: the second timestep twice (_timesteps[-2:-1] again).
    if Spacing <> tsKarras then FPndmStride := FT div NumSteps;
    if NumSteps >= 2 then
    begin
      SetLength(Result, NumSteps + 2);
      for k := NumSteps + 1 downto 2 do Result[k] := Result[k - 1];
    end;
  end;
end;

function TNNetDiffusionScheduler.BuildImg2ImgSchedule(NumSteps: integer;
  Spacing: TNNetTimestepSpacing; Method: TNNetSamplerMethod;
  Strength: double): TNeuralIntegerArray;
var
  StartStep: integer;
begin
  Result := BuildSamplerSchedule(NumSteps, Spacing, Method);
  StartStep := Img2ImgStartStep(NumSteps, Strength);
  Result := Copy(Result, StartStep, Length(Result) - StartStep);
end;

class function TNNetDiffusionScheduler.Img2ImgStartStep(NumSteps: integer;
  Strength: double): integer;
var
  InitStepCount: integer;
begin
  if NumSteps < 1 then
    raise Exception.Create('Img2ImgStartStep: NumSteps must be >= 1.');
  if not ((Strength >= 0) and (Strength <= 1)) then
    raise Exception.Create('img2img strength ' + FloatToStr(Strength) +
      ' is outside [0, 1].');
  // Double product, so the truncation lands where Python's int() does.
  InitStepCount := Trunc(NumSteps * Strength);
  if InitStepCount > NumSteps then InitStepCount := NumSteps;
  if InitStepCount < 1 then
    raise Exception.Create('img2img strength ' + FloatToStr(Strength) +
      ' with ' + IntToStr(NumSteps) + ' steps leaves no step to run; raise ' +
      'the strength or the step count.');
  Result := NumSteps - InitStepCount;
end;

function TNNetDiffusionScheduler.InitialNoiseScale(Method: TNNetSamplerMethod;
  Spacing: TNNetTimestepSpacing; FirstTt: integer): TNeuralFloat;
begin
  // diffusers Euler init_noise_sigma is sigma_max for linspace/trailing and
  // sqrt(sigma_max^2+1) otherwise; over the model-input scale sqrt(sigma^2+1)
  // that is sqrt(1-ab) or 1.
  if (Method in [smEuler, smEulerAncestral]) and
    (Spacing in [tsUniform, tsTrailing]) then
    Result := FSqrtOneMinusAlphaBar[FirstTt]
  else
    Result := 1;
end;

function TNNetDiffusionScheduler.InitialNoiseScaleAtSigma(
  Method: TNNetSamplerMethod; Spacing: TNNetTimestepSpacing;
  FirstSigma: double): TNeuralFloat;
begin
  // sigma_max over sqrt(sigma_max^2 + 1): the sqrt(1-ab) of InitialNoiseScale.
  if (Method in [smEuler, smEulerAncestral]) and
    (Spacing in [tsUniform, tsTrailing]) then
    Result := FirstSigma / Sqrt(1 + FirstSigma * FirstSigma)
  else
    Result := 1;
end;

procedure TNNetDiffusionScheduler.EulerAncestralUpdate(Xt, X0,
  Noise: TNNetVolume; SigmaCur, SigmaNext, InvSqrtAbCur, SqrtAbNext,
  Eta: TNeuralFloat);
var
  i, XtSizeM1: integer;
  SigmaUp, SigmaDown, DriftRatio, kA, cE, kZ, z: TNeuralFloat;
begin
  // Euler-ancestral ("Euler a", Karras et al. 2022) in the VE sigma
  // parameterisation sigma = sqrt((1-ab)/ab), with s1 = SigmaCur, s2 = SigmaNext:
  //   sigma_up   = eta * sqrt( (s2^2*(s1^2 - s2^2)) / s1^2 )   (injected)
  //   sigma_down = sqrt( max(0, s2^2 - sigma_up^2) )          (Euler drift)
  //   x = sqrt(ab_next)*(x0 + (sigma_down/s1)*(x_t/sqrt(ab_t) - x0)
  //       + sigma_up*z).
  if (Eta > 0) and (SigmaCur > 0) then
    SigmaUp := Eta * Sqrt((SigmaNext * SigmaNext *
      Max(0.0, SigmaCur * SigmaCur - SigmaNext * SigmaNext)) /
      (SigmaCur * SigmaCur))
  else
    SigmaUp := 0;
  SigmaDown := Sqrt(Max(0.0, SigmaNext * SigmaNext - SigmaUp * SigmaUp));
  if SigmaCur > 0 then DriftRatio := SigmaDown / SigmaCur else DriftRatio := 0;
  // #5: the update folds to kA*x_t + cE*x0 + kZ*z per step.
  kA := SqrtAbNext * DriftRatio * InvSqrtAbCur;
  cE := SqrtAbNext * (1 - DriftRatio);
  if (SigmaUp > 0) and Assigned(Noise) then
  begin
    Xt.Mul(kA);
    Xt.MulAdd(cE, X0);
    Xt.MulAdd(SqrtAbNext * SigmaUp, Noise);
  end
  else if SigmaUp > 0 then
  begin
    kZ := SqrtAbNext * SigmaUp;
    XtSizeM1 := Xt.Size - 1;
    for i := 0 to XtSizeM1 do
    begin
      z := RandG(0, 1);
      Xt.FData[i] := kA * Xt.FData[i] + cE * X0.FData[i] + kZ * z;
    end;
  end
  else
  begin
    // sigma_up = 0 (z = 0): a uniform affine map (#13).
    Xt.Mul(kA);
    Xt.MulAdd(cE, X0);
  end;
end;

procedure TNNetDiffusionScheduler.DPMSolverPP2MUpdate(Xt, X0: TNNetVolume;
  LambdaCur, LambdaNext, KX, AbNext: TNeuralFloat);
var
  h, hLast, kD, cA, cB: TNeuralFloat;
begin
  // x_next = KX*x_t - alpha_next*(exp(-h)-1)*D, h = lambda_next - lambda_cur;
  // D = x0 on the first step, else the midpoint (1+1/(2r))*x0 - (1/(2r))*x0_prev.
  h := LambdaNext - LambdaCur;              // > 0
  kD := Sqrt(AbNext) * (Exp(-h) - 1.0);
  if not FHasPrev then
  begin
    // First order (equal to DDIM): KX*Xt - kD*X0 (#13).
    Xt.Mul(KX);
    Xt.MulAdd(-kD, X0);
  end
  else
  begin
    hLast := LambdaCur - FPrevLambda;       // > 0
    cB := DPMSolverPP2MHistoryWeight(hLast, h);
    cA := 1.0 + cB;
    // Xt := KX*Xt - kD*(cA*X0 - cB*FPrevX0) as a uniform affine map (#13).
    Xt.Mul(KX);
    Xt.MulAdd(-kD * cA, X0);
    Xt.MulAdd(kD * cB, FPrevX0);
  end;
  // Store this step's x0 and lambda for the next multistep correction.
  if not Assigned(FPrevX0) then FPrevX0 := TNNetVolume.Create(Xt);
  FPrevX0.Copy(X0);
  FPrevLambda := LambdaCur;
  FHasPrev := true;
end;

function TNNetDiffusionScheduler.InterpolatedSigma(
  ModelTimestep: double): double;
var
  LowPos: integer;
  Frac: double;
begin
  // The 0-based timestep t reads FSigma[t + 1]; np.interp clamps at the ends.
  if ModelTimestep <= 0 then Exit(FSigma[1]);
  if ModelTimestep >= FT - 1 then Exit(FSigma[FT]);
  LowPos := Trunc(ModelTimestep);
  Frac := ModelTimestep - LowPos;
  Result := FSigma[LowPos + 1] + Frac * (FSigma[LowPos + 2] - FSigma[LowPos + 1]);
end;

function TNNetDiffusionScheduler.SigmaToModelTimestep(Sigma: double): double;
var
  LogSigma, LogLow, LogHigh, Weight: double;
  Lo, Hi, Mid, LowPos: integer;
begin
  // diffusers _sigma_to_t: LowPos = the last 0-based index whose log sigma is
  // <= log(Sigma) (0 when none), at most T-2; then linear in log sigma.
  LogSigma := Ln(Max(Sigma, 1e-10));
  Lo := 0;
  Hi := FT - 1;
  while Lo < Hi do
  begin
    Mid := (Lo + Hi + 1) shr 1;
    if Ln(FSigma[Mid + 1]) <= LogSigma then Lo := Mid else Hi := Mid - 1;
  end;
  LowPos := Lo;
  if LowPos > FT - 2 then LowPos := FT - 2;
  LogLow := Ln(FSigma[LowPos + 1]);
  LogHigh := Ln(FSigma[LowPos + 2]);
  if LogLow = LogHigh then Weight := 0
  else Weight := EnsureRange((LogLow - LogSigma) / (LogLow - LogHigh), 0, 1);
  Result := (1 - Weight) * LowPos + Weight * (LowPos + 1);
end;

function TNNetDiffusionScheduler.BuildSigmaSchedule(NumSteps: integer;
  Spacing: TNNetTimestepSpacing; Method: TNNetSamplerMethod;
  KarrasSigmas: boolean): TNNetSigmaSchedule;
const
  cRho = 7.0;
var
  IntSchedule: TNeuralIntegerArray;
  StepPos, MaxStepPos: integer;
  LinspaceStride, SigmaMin, SigmaMax, InvRhoMin, InvRhoMax, Ramp: double;
  IsDPM: boolean;
begin
  if not (Method in [smEuler, smEulerAncestral, smDPMSolverPP2M]) then
    raise Exception.Create('BuildSigmaSchedule: only Euler, Euler a and ' +
      'DPM++ 2M sample at continuous sigmas.');
  IsDPM := Method = smDPMSolverPP2M;
  if IsDPM and not KarrasSigmas then
    raise Exception.Create('BuildSigmaSchedule: DPM++ 2M without Karras ' +
      'sigmas samples at integer timesteps (BuildSamplerSchedule).');
  if NumSteps < 1 then NumSteps := 1;
  MaxStepPos := NumSteps - 1;
  SetLength(Result.Sigmas, NumSteps + 1);
  SetLength(Result.ModelTimesteps, NumSteps);
  if IsDPM then
  begin
    // DPMSolverMultistepScheduler: the Karras range is the whole table.
    SigmaMin := FSigma[1];
    SigmaMax := FSigma[FT];
  end
  else
  begin
    // Euler: the spacing's own timesteps, sigmas interpolated linearly
    // (fractional for linspace, float32 as diffusers' np.linspace).
    if Spacing = tsUniform then
    begin
      if NumSteps > 1 then LinspaceStride := (FT - 1) / MaxStepPos
      else LinspaceStride := 0;
      for StepPos := 0 to MaxStepPos do
        Result.ModelTimesteps[StepPos] :=
          Single((MaxStepPos - StepPos) * LinspaceStride);
    end
    else if Spacing in [tsLeading, tsTrailing] then
    begin
      IntSchedule := BuildTimestepSchedule(NumSteps, Spacing);
      for StepPos := 0 to MaxStepPos do
        Result.ModelTimesteps[StepPos] := IntSchedule[StepPos] - 1;
    end
    else
      raise Exception.Create('BuildSigmaSchedule: Euler takes linspace ' +
        '(tsUniform), leading or trailing spacing.');
    for StepPos := 0 to MaxStepPos do
      Result.Sigmas[StepPos] := InterpolatedSigma(Result.ModelTimesteps[StepPos]);
    SigmaMax := Result.Sigmas[0];
    SigmaMin := Result.Sigmas[MaxStepPos];
  end;
  if KarrasSigmas then
  begin
    // Karras et al. (2022) rho = 7 ramp, each sigma mapped back to its
    // fractional timestep (rounded for DPM unless the schedule is cosine).
    InvRhoMax := Power(SigmaMax, 1 / cRho);
    InvRhoMin := Power(SigmaMin, 1 / cRho);
    for StepPos := 0 to MaxStepPos do
    begin
      if MaxStepPos > 0 then Ramp := StepPos / MaxStepPos else Ramp := 0;
      Result.Sigmas[StepPos] :=
        Power(InvRhoMax + Ramp * (InvRhoMin - InvRhoMax), cRho);
      Result.ModelTimesteps[StepPos] :=
        SigmaToModelTimestep(Result.Sigmas[StepPos]);
      if IsDPM and (FSchedule <> dsCosine) then
        Result.ModelTimesteps[StepPos] := Round(Result.ModelTimesteps[StepPos]);
    end;
  end;
  Result.Sigmas[NumSteps] := 0;
  ResetMultistep;
end;

procedure TNNetDiffusionScheduler.StepAtSigma(Xt, RawPred: TNNetVolume;
  Sigma, NextSigma: double; Method: TNNetSamplerMethod; Eta: TNeuralFloat;
  Noise: TNNetVolume);
var
  InvSqrtAb, SqrtAb, NextInvSqrtAb: double;
begin
  if not (Method in [smEuler, smEulerAncestral, smDPMSolverPP2M]) then
    raise Exception.Create('StepAtSigma: only Euler, Euler a and DPM++ 2M ' +
      'step at continuous sigmas.');
  if not (Sigma > 0) then
    raise Exception.Create('StepAtSigma: sigma ' + FloatToStr(Sigma) +
      ' must be > 0.');
  if not Assigned(FEpsScratch) then FEpsScratch := TNNetVolume.Create(Xt)
  else FEpsScratch.ReSize(Xt);
  // ab = 1/(1 + sigma^2) at a continuous sigma.
  InvSqrtAb := Sqrt(1 + Sigma * Sigma);
  SqrtAb := 1 / InvSqrtAb;
  PredictX0At(Xt, RawPred, FEpsScratch, SqrtAb, Sigma * SqrtAb, InvSqrtAb);
  if NextSigma <= 0 then
  begin
    // Final hop to sigma 0 (final_sigmas_type "zero"): the x0 prediction.
    Xt.Copy(FEpsScratch);
    FHasPrev := false;
    Exit;
  end;
  NextInvSqrtAb := Sqrt(1 + NextSigma * NextSigma);
  if Method = smDPMSolverPP2M then
    // lambda = log(alpha/sigma_vp) = -ln(sigma); sqrt(1-ab) = sigma*sqrt(ab).
    DPMSolverPP2MUpdate(Xt, FEpsScratch, -Ln(Sigma), -Ln(NextSigma),
      (NextSigma / NextInvSqrtAb) / (Sigma * SqrtAb), 1 / Sqr(NextInvSqrtAb))
  else
  begin
    if Method = smEuler then Eta := 0;
    EulerAncestralUpdate(Xt, FEpsScratch, Noise, Sigma, NextSigma, InvSqrtAb,
      1 / NextInvSqrtAb, Eta);
  end;
end;

procedure TNNetDiffusionScheduler.SampleHeun(X: TNNetVolume;
  Denoise: TNNetDenoiseCallback; NumSteps: integer; Spacing: TNNetTimestepSpacing);
var
  k, Tt, TtPrev: integer;
  NumStepsM1: integer;
  Pred, X0, Y, Ye, X0b, XPred: TNNetVolume;
  Schedule: TNeuralIntegerArray;
  sigma, sigmaNext, sqrtAbPrev: TNeuralFloat;
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
      // Step-invariant reciprocals/deltas hoisted out of the element loops (#5).
      invSqrtAbT := FInvSqrtAlphaBar[Tt];
      // sigma_t is 0 wherever ab_t reaches 1 (Beta1 = 0, or a cosine schedule
      // whose first beta clips to 0). There is no drift at a clean sample, so
      // the reciprocal collapses to 0 instead of overflowing.
      if sigma > 0 then invSigma := 1.0 / sigma else invSigma := 0;
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
      if (TtPrev = 0) or (sigmaNext <= 0) then
      begin
        // Final step: sqrt(ab_prev) = 1, y_e is already the clean image. A
        // sigma_{t-1} of 0 at TtPrev > 0 means ab_prev = 1 as well, so the same
        // holds there; k-diffusion likewise skips the corrector at sigma 0.
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
  MaxStepPos: integer;
  Pred: TNNetVolume;
  Schedule: TNeuralIntegerArray;
begin
  if Method = smHeun then
  begin
    // Heun needs two denoiser evals per step; handled by its own driver.
    SampleHeun(X, Denoise, NumSteps, Spacing);
    Exit;
  end;
  // Descending timesteps ending in 0 (clean); PNDM visits one extra.
  Schedule := BuildSamplerSchedule(NumSteps, Spacing, Method);
  MaxStepPos := Length(Schedule) - 2;
  Pred := TNNetVolume.Create(X);
  try
    for k := 0 to MaxStepPos do
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

{ TNNetAdaptiveProjectedGuidance }

constructor TNNetAdaptiveProjectedGuidance.Create(pEta, pMomentum,
  pNormThreshold: TNeuralFloat; pUseOriginalFormulation: boolean);
begin
  inherited Create;
  FEta := pEta;
  FMomentum := pMomentum;
  FNormThreshold := pNormThreshold;
  FUseOriginalFormulation := pUseOriginalFormulation;
  FRunningDiff := TNNetVolume.Create;
  FHasRunningDiff := false;
end;

constructor TNNetAdaptiveProjectedGuidance.CreateFromSpec(const Spec: string);
var
  Fields: array[0..2] of TNeuralFloat;
  DotFormat: TFormatSettings;
  Rest: string;
  FieldIdx, CommaPos: integer;
begin
  DotFormat := DefaultFormatSettings;
  DotFormat.DecimalSeparator := '.';
  Rest := Spec;
  for FieldIdx := 0 to 2 do
  begin
    CommaPos := Pos(',', Rest);
    if (FieldIdx < 2) and (CommaPos = 0) then
      raise EConvertError.Create('APG spec needs ETA,MOMENTUM,THRESHOLD: ' + Spec);
    if FieldIdx = 2 then CommaPos := Length(Rest) + 1;
    Fields[FieldIdx] := StrToFloat(Trim(Copy(Rest, 1, CommaPos - 1)), DotFormat);
    Delete(Rest, 1, CommaPos);
  end;
  Create(Fields[0], Fields[1], Fields[2]);
end;

destructor TNNetAdaptiveProjectedGuidance.Destroy;
begin
  FRunningDiff.Free;
  inherited Destroy;
end;

procedure TNNetAdaptiveProjectedGuidance.Reset;
begin
  FHasRunningDiff := false;
end;

procedure TNNetAdaptiveProjectedGuidance.ApplyAPG(Cond, Uncond,
  Dst: TNNetVolume; W: TNeuralFloat);
var
  DiffNorm, ClampScale, CondSumSqr, ParCoef: TNeuralFloat;
  CondCoef, DiffCoef: TNeuralFloat;
begin
  // diffusers skips guidance (and the momentum update) at the no-op scale.
  if (FUseOriginalFormulation and (W = 0)) or
    ((not FUseOriginalFormulation) and (W = 1)) then
  begin
    if Dst <> Cond then Dst.Copy(Cond);
    exit;
  end;
  // Eta = 1 makes the projection cancel out; without momentum or clamp it is CFG.
  if (FEta = 1) and (FMomentum = 0) and (FNormThreshold <= 0) and
    (not FUseOriginalFormulation) then
  begin
    TNNetDiffusionScheduler.ApplyCFG(Cond, Uncond, Dst, W);
    exit;
  end;

  // Running average avg := (Cond - Uncond) + Momentum*avg (starts at zero).
  if FHasRunningDiff and (FMomentum <> 0) and (FRunningDiff.Size = Cond.Size) then
  begin
    FRunningDiff.Mul(FMomentum);
    FRunningDiff.Add(Cond);
  end
  else
    FRunningDiff.Copy(Cond);
  FRunningDiff.Sub(Uncond);
  FHasRunningDiff := true;

  // diff := ClampScale*avg keeps avg itself unclamped, as in diffusers.
  ClampScale := 1;
  if FNormThreshold > 0 then
  begin
    DiffNorm := FRunningDiff.GetMagnitude();
    if DiffNorm > FNormThreshold then ClampScale := FNormThreshold / DiffNorm;
  end;

  // orth + Eta*par = diff + (Eta-1)*par with par = ((diff.Cond)/(Cond.Cond))*Cond,
  // so Dst = base + CondCoef*Cond + DiffCoef*avg and avg is never rewritten.
  ParCoef := 0;
  if FEta <> 1 then
  begin
    CondSumSqr := Cond.GetSumSqr();
    if CondSumSqr > 0 then
      ParCoef := ClampScale * FRunningDiff.DotProduct(Cond) / CondSumSqr;
  end;
  DiffCoef := W * ClampScale;
  CondCoef := W * (FEta - 1) * ParCoef;
  if FUseOriginalFormulation then
  begin
    if Dst <> Cond then Dst.Copy(Cond);
    Dst.Mul(CondCoef + 1);
  end
  else if Dst = Cond then
  begin
    Dst.Mul(CondCoef);
    Dst.Add(Uncond);
  end
  else
  begin
    if Dst <> Uncond then Dst.Copy(Uncond);
    if CondCoef <> 0 then Dst.MulAdd(CondCoef, Cond);
  end;
  Dst.MulAdd(DiffCoef, FRunningDiff);
end;

{ TNNetPerturbedGuidance }

constructor TNNetPerturbedGuidance.Create(pGuidanceScale,
  pPerturbedScale: TNeuralFloat; pPerturbedStart, pPerturbedStop: double;
  pGuidanceRescale: TNeuralFloat; pUseOriginalFormulation: boolean);
begin
  inherited Create;
  FGuidanceScale := pGuidanceScale;
  FPerturbedScale := pPerturbedScale;
  FGuidanceRescale := pGuidanceRescale;
  FUseOriginalFormulation := pUseOriginalFormulation;
  FGuidanceStart := 0;
  FGuidanceStop := 1;
  FMode := pgmAttentionScores;
  SetPerturbedWindow(pPerturbedStart, pPerturbedStop);
end;

procedure TNNetPerturbedGuidance.SetPerturbedWindow(pStart, pStop: double);
begin
  if not ((pStart >= 0) and (pStart < 1)) then
    raise EArgumentException.Create('TNNetPerturbedGuidance: start ' +
      FloatToStr(pStart) + ' is outside [0, 1).');
  if not ((pStart <= pStop) and (pStop <= 1)) then
    raise EArgumentException.Create('TNNetPerturbedGuidance: stop ' +
      FloatToStr(pStop) + ' is outside [start, 1].');
  FPerturbedStart := pStart;
  FPerturbedStop := pStop;
end;

function TNNetPerturbedGuidance.IsCFGEnabled(Step, NumSteps: integer): boolean;
begin
  if FUseOriginalFormulation
    then Result := FGuidanceScale <> 0
    else Result := FGuidanceScale <> 1;
  if Result and (NumSteps > 0) then
    Result := (Trunc(FGuidanceStart * NumSteps) <= Step) and
      (Step < Trunc(FGuidanceStop * NumSteps));
end;

function TNNetPerturbedGuidance.IsPerturbedEnabled(Step,
  NumSteps: integer): boolean;
begin
  Result := FPerturbedScale <> 0;
  if Result and (NumSteps > 0) then
    Result := (Trunc(FPerturbedStart * NumSteps) < Step) and
      (Step < Trunc(FPerturbedStop * NumSteps));
end;

procedure TNNetPerturbedGuidance.SetPerturbedLayers(
  const pLayers: array of integer);
var
  LayerPos, MaxLayerPos: integer;
begin
  MaxLayerPos := High(pLayers);
  SetLength(FPerturbedLayers, MaxLayerPos + 1);
  for LayerPos := 0 to MaxLayerPos do
    FPerturbedLayers[LayerPos] := pLayers[LayerPos];
end;

function TNNetPerturbedGuidance.PerturbsLayer(LayerIdx: integer): boolean;
var
  LayerPos, MaxLayerPos: integer;
begin
  Result := true;
  MaxLayerPos := High(FPerturbedLayers);
  for LayerPos := 0 to MaxLayerPos do
    if FPerturbedLayers[LayerPos] = LayerIdx then exit;
  Result := false;
end;

procedure TNNetPerturbedGuidance.Apply(Step, NumSteps: integer; Cond, Uncond,
  Skip, Dst: TNNetVolume);
var
  CFGOn, PerturbedOn: boolean;
  CondCoef, UncondCoef, SkipCoef, BaseCoef, StdCond: TNeuralFloat;
begin
  CFGOn := IsCFGEnabled(Step, NumSteps);
  PerturbedOn := IsPerturbedEnabled(Step, NumSteps);
  if CFGOn and (Uncond = nil) then
    raise EArgumentException.Create('TNNetPerturbedGuidance.Apply: step ' +
      IntToStr(Step) + ' runs CFG and needs Uncond.');
  if PerturbedOn and (Skip = nil) then
    raise EArgumentException.Create('TNNetPerturbedGuidance.Apply: step ' +
      IntToStr(Step) + ' runs the perturbed pass and needs Skip.');
  // Original formulation: base Cond, i.e. ApplyCFG's base plus one more W.
  if FUseOriginalFormulation then BaseCoef := 1 else BaseCoef := 0;
  if not PerturbedOn then
  begin
    if CFGOn then
      TNNetDiffusionScheduler.ApplyCFG(Cond, Uncond, Dst,
        FGuidanceScale + BaseCoef, FGuidanceRescale)
    // diffusers rescales Cond by std(Cond)/std(Cond) = 1 here: a no-op.
    else if Dst <> Cond then Dst.Copy(Cond);
    exit;
  end;
  if not CFGOn then
  begin
    // Skip + W*(Cond - Skip): the CFG mix with Skip as the negative.
    TNNetDiffusionScheduler.ApplyCFG(Cond, Skip, Dst,
      FPerturbedScale + BaseCoef, FGuidanceRescale);
    exit;
  end;
  CondCoef := BaseCoef + FGuidanceScale + FPerturbedScale;
  UncondCoef := (1 - BaseCoef) - FGuidanceScale;
  SkipCoef := -FPerturbedScale;
  StdCond := 0;
  if FGuidanceRescale <> 0 then StdCond := Cond.GetStdDeviation();
  MixPredictions(CondCoef, UncondCoef, SkipCoef, Cond, Uncond, Skip, Dst);
  TNNetDiffusionScheduler.RescaleGuidance(Dst, StdCond, FGuidanceRescale);
end;

procedure TNNetPerturbedGuidance.MixPredictions(CondCoef, UncondCoef,
  SkipCoef: TNeuralFloat; Cond, Uncond, Skip, Dst: TNNetVolume);
var
  HasUncond: boolean;
begin
  HasUncond := Assigned(Uncond);
  if HasUncond and (Dst = Uncond) then
  begin
    Dst.Mul(UncondCoef);
    Dst.MulAdd(CondCoef, Cond);
    Dst.MulAdd(SkipCoef, Skip);
  end
  else if Dst = Skip then
  begin
    Dst.Mul(SkipCoef);
    Dst.MulAdd(CondCoef, Cond);
    if HasUncond then Dst.MulAdd(UncondCoef, Uncond);
  end
  else
  begin
    if Dst <> Cond then Dst.Copy(Cond);
    Dst.Mul(CondCoef);
    if HasUncond then Dst.MulAdd(UncondCoef, Uncond);
    Dst.MulAdd(SkipCoef, Skip);
  end;
end;

function TNNetPerturbedGuidance.PerturbedScaleAt(
  Timestep: TNeuralFloat): TNeuralFloat;
begin
  Result := FPerturbedScale;
  if (FAdaptiveScale > 0) and (FPerturbedScale > 0) then
  begin
    Result := FPerturbedScale - FAdaptiveScale * (1000 - Timestep);
    if Result < 0 then Result := 0;
  end;
end;

procedure TNNetPerturbedGuidance.ApplyPerturbedAttentionGuidance(Timestep,
  W: TNeuralFloat; Cond, Uncond, Skip, Dst: TNNetVolume);
var
  StepScale, CondCoef: TNeuralFloat;
begin
  StepScale := PerturbedScaleAt(Timestep);
  if Assigned(Uncond) then CondCoef := W + StepScale
  else CondCoef := 1 + StepScale;
  MixPredictions(CondCoef, 1 - W, -StepScale, Cond, Uncond, Skip, Dst);
end;

{ TNNetLCMScheduler }

const
  cLCMSigmaData = 0.5;   // sigma_data in the LCM boundary scalings.

constructor TNNetLCMScheduler.Create(pT: integer; pSchedule: TNNetBetaSchedule;
  pPrediction: TNNetPredictionType; Beta1: TNeuralFloat; BetaT: TNeuralFloat;
  CosineS: TNeuralFloat; pRescaleZeroTerminalSNR: boolean;
  pFinalAlphaBarIsOne: boolean);
begin
  inherited Create(pT, pSchedule, pPrediction, Beta1, BetaT, CosineS,
    pRescaleZeroTerminalSNR, pFinalAlphaBarIsOne);
  FOriginalInferenceSteps := 50;
  FTimestepScaling := 10;
end;

procedure TNNetLCMScheduler.ReadDiffusersConfig(Config: TJSONObject);
begin
  FOriginalInferenceSteps := Config.Get('original_inference_steps', 50);
  FTimestepScaling := Config.Get('timestep_scaling', 10.0);
end;

function TNNetLCMScheduler.CSkip(Tt: integer): TNeuralFloat;
var
  ScaledT2, SigmaData2: double;
begin
  ScaledT2 := Sqr(double(Tt - 1) * FTimestepScaling);
  SigmaData2 := Sqr(cLCMSigmaData);
  Result := SigmaData2 / (ScaledT2 + SigmaData2);
end;

function TNNetLCMScheduler.COut(Tt: integer): TNeuralFloat;
var
  ScaledT: double;
begin
  ScaledT := double(Tt - 1) * FTimestepScaling;
  Result := ScaledT / Sqrt(Sqr(ScaledT) + Sqr(cLCMSigmaData));
end;

procedure TNNetLCMScheduler.LCMStep(Xt, RawPred: TNNetVolume; Tt: integer);
var
  SkipCoef, OutCoef: TNeuralFloat;
begin
  SkipCoef := CSkip(Tt);
  OutCoef := COut(Tt);
  // Persistent lazily-sized x0_hat scratch (rule #17); PredictX0 uses the
  // separate FEpsScratch internally, so the two never collide.
  if not Assigned(FX0HatScratch) then FX0HatScratch := TNNetVolume.Create(Xt)
  else FX0HatScratch.ReSize(Xt);
  PredictX0(Xt, RawPred, FX0HatScratch, Tt);
  Xt.Mul(SkipCoef);
  Xt.MulAdd(OutCoef, FX0HatScratch);
end;

function TNNetLCMScheduler.BuildOriginSchedule(NumSteps: integer;
  Strength: double): TNeuralIntegerArray;
var
  GridStride, OriginCount, MaxStepPos, StepPos, OriginPos: integer;
  OriginPerStep: double;
begin
  if (FOriginalInferenceSteps < 1) or (FOriginalInferenceSteps > FT) then
    raise Exception.Create('LCM original_inference_steps ' +
      IntToStr(FOriginalInferenceSteps) + ' is outside 1..' + IntToStr(FT) + '.');
  if not ((Strength > 0) and (Strength <= 1)) then
    raise Exception.Create('LCM strength ' + FloatToStr(Strength) +
      ' is outside (0, 1].');
  if (NumSteps < 1) or (NumSteps > FOriginalInferenceSteps) then
    raise Exception.Create('LCM step count ' + IntToStr(NumSteps) +
      ' is outside 1..' + IntToStr(FOriginalInferenceSteps) +
      ' (original_inference_steps).');
  GridStride := FT div FOriginalInferenceSteps;
  // Double product, so the truncation lands where Python's int() does.
  OriginCount := Trunc(FOriginalInferenceSteps * Strength);
  if OriginCount < NumSteps then
    raise Exception.Create('LCM: ' + IntToStr(NumSteps) + ' steps need at ' +
      'least that many origin points; strength ' + FloatToStr(Strength) +
      ' leaves ' + IntToStr(OriginCount) + ' of ' +
      IntToStr(FOriginalInferenceSteps) + '.');
  ResetMultistep;
  Result := nil;
  SetLength(Result, NumSteps + 1);
  // numpy floor(linspace(0, OriginCount, NumSteps, endpoint=False)) picks
  // descending origin entry OriginPos, diffusers t = (OriginCount-OriginPos)*k-1.
  OriginPerStep := OriginCount / NumSteps;
  MaxStepPos := NumSteps - 1;
  for StepPos := 0 to MaxStepPos do
  begin
    OriginPos := Floor(StepPos * OriginPerStep);
    Result[StepPos] := (OriginCount - OriginPos) * GridStride;
  end;
  Result[NumSteps] := 0;
end;

function TNNetLCMScheduler.BuildSamplerSchedule(NumSteps: integer;
  Spacing: TNNetTimestepSpacing; Method: TNNetSamplerMethod):
  TNeuralIntegerArray;
begin
  if Method = smLCM then Result := BuildOriginSchedule(NumSteps)
  else Result := inherited BuildSamplerSchedule(NumSteps, Spacing, Method);
end;

function TNNetLCMScheduler.BuildImg2ImgSchedule(NumSteps: integer;
  Spacing: TNNetTimestepSpacing; Method: TNNetSamplerMethod;
  Strength: double): TNeuralIntegerArray;
begin
  if Method = smLCM then Result := BuildOriginSchedule(NumSteps, Strength)
  else Result := inherited BuildImg2ImgSchedule(NumSteps, Spacing, Method,
    Strength);
end;

procedure TNNetLCMScheduler.Step(Xt, RawPred: TNNetVolume; Tt,
  TtPrev: integer; Method: TNNetSamplerMethod; Eta: TNeuralFloat;
  Noise: TNNetVolume);
begin
  if Method <> smLCM then
  begin
    inherited Step(Xt, RawPred, Tt, TtPrev, Method, Eta, Noise);
    exit;
  end;
  LCMStep(Xt, RawPred, Tt);
  if TtPrev > 0 then AddNoise(Xt, Xt, TtPrev, nil, Noise);
end;

{ TNNetFlowMatchEulerScheduler }

constructor TNNetFlowMatchEulerScheduler.Create(pNumTrainTimesteps: integer;
  pShift: double; pUseDynamicShifting: boolean;
  pTimeShiftType: TNNetFlowTimeShift; pShiftTerminal: double);
begin
  inherited Create;
  if pNumTrainTimesteps < 1 then
    raise Exception.Create('Flow-matching NumTrainTimesteps must be >= 1.');
  FNumTrainTimesteps := pNumTrainTimesteps;
  FShift := pShift;
  FUseDynamicShifting := pUseDynamicShifting;
  FTimeShiftType := pTimeShiftType;
  FShiftTerminal := pShiftTerminal;
  // The diffusers pipelines' calculate_shift defaults.
  FBaseImageSeqLen := 256;
  FMaxImageSeqLen := 4096;
  FBaseShift := 0.5;
  FMaxShift := 1.15;
  SetLength(FSigmas, 0);
  FSampler := fsEuler;
  FPrevX0 := nil;
  FPrevX0Older := nil;
  FPrevSample := nil;
  FPrevStepIndex := -1;
  FHistoryLen := 0;
  FPrevOrder := 1;
  FUniPCSolverType := ustBh2;
  FUniPCCorrector := true;
end;

destructor TNNetFlowMatchEulerScheduler.Destroy;
begin
  FPrevX0.Free;
  FPrevX0Older.Free;
  FPrevSample.Free;
  inherited Destroy;
end;

constructor TNNetFlowMatchEulerScheduler.CreateFromDiffusersConfig(
  const FileName: string);
var
  JsonText: TStringList;
  Root: TJSONData;
  Obj: TJSONObject;
  ShiftTypeName: string;
  ShiftType: TNNetFlowTimeShift;

  procedure RefuseFlag(const Key: string);
  begin
    if Obj.Get(Key, false) then
      raise Exception.Create('Flow-matching scheduler config ' + FileName +
        ': "' + Key + '" = true is not implemented.');
  end;

begin
  JsonText := TStringList.Create;
  Root := nil;
  try
    JsonText.LoadFromFile(FileName);
    Root := GetJSON(JsonText.Text);
    if not (Root is TJSONObject) then
      raise Exception.Create('Flow-matching scheduler config ' + FileName +
        ' is not a JSON object.');
    Obj := TJSONObject(Root);
    RefuseFlag('use_karras_sigmas');
    RefuseFlag('use_exponential_sigmas');
    RefuseFlag('use_beta_sigmas');
    RefuseFlag('invert_sigmas');
    RefuseFlag('stochastic_sampling');
    ShiftTypeName := Obj.Get('time_shift_type', 'exponential');
    if ShiftTypeName = 'exponential' then ShiftType := ftsExponential
    else if ShiftTypeName = 'linear' then ShiftType := ftsLinear
    else
      raise Exception.Create('Flow-matching scheduler config ' + FileName +
        ': unknown time_shift_type "' + ShiftTypeName + '".');
    Create(Obj.Get('num_train_timesteps', 1000), Obj.Get('shift', 1.0),
      Obj.Get('use_dynamic_shifting', false), ShiftType,
      Obj.Get('shift_terminal', 0.0));
    FBaseImageSeqLen := Obj.Get('base_image_seq_len', FBaseImageSeqLen);
    FMaxImageSeqLen := Obj.Get('max_image_seq_len', FMaxImageSeqLen);
    FBaseShift := Obj.Get('base_shift', FBaseShift);
    FMaxShift := Obj.Get('max_shift', FMaxShift);
  finally
    Root.Free;
    JsonText.Free;
  end;
end;

function TNNetFlowMatchEulerScheduler.ShiftForImageSeqLen(
  ImageSeqLen: integer): double;
begin
  Result := CalculateShift(ImageSeqLen, FBaseImageSeqLen, FMaxImageSeqLen,
    FBaseShift, FMaxShift);
end;

class function TNNetFlowMatchEulerScheduler.CalculateShift(ImageSeqLen: integer;
  BaseSeqLen: integer; MaxSeqLen: integer; BaseShift: double;
  MaxShift: double): double;
var
  Slope, Intercept: double;
begin
  Slope := (MaxShift - BaseShift) / (MaxSeqLen - BaseSeqLen);
  Intercept := BaseShift - Slope * BaseSeqLen;
  Result := ImageSeqLen * Slope + Intercept;
end;

function TNNetFlowMatchEulerScheduler.TimeShift(Mu, Sigma: double): double;
var
  Numerator: double;
begin
  if Sigma <= 0 then Exit(0);
  if FTimeShiftType = ftsExponential then Numerator := Exp(Mu)
  else Numerator := Mu;
  Result := Numerator / (Numerator + (1 / Sigma - 1));
end;

procedure TNNetFlowMatchEulerScheduler.SetSigmas(const Sigmas: array of double;
  Mu: double);
var
  SigmaPos, MaxSigmaPos: integer;
  Shifted, StretchScale: double;
begin
  if Length(Sigmas) < 1 then
    raise Exception.Create('Flow-matching schedule needs at least one sigma.');
  MaxSigmaPos := High(Sigmas);
  SetLength(FSigmas, Length(Sigmas) + 1);
  for SigmaPos := 0 to MaxSigmaPos do
  begin
    if FUseDynamicShifting then Shifted := TimeShift(Mu, Sigmas[SigmaPos])
    else Shifted := FShift * Sigmas[SigmaPos] /
      (1 + (FShift - 1) * Sigmas[SigmaPos]);
    FSigmas[SigmaPos] := Shifted;
  end;
  // shift_terminal: rescale 1 - sigma so the last sigma lands on ShiftTerminal.
  if FShiftTerminal <> 0 then
  begin
    StretchScale := (1 - FSigmas[MaxSigmaPos]) / (1 - FShiftTerminal);
    for SigmaPos := 0 to MaxSigmaPos do
      FSigmas[SigmaPos] := 1 - (1 - FSigmas[SigmaPos]) / StretchScale;
  end;
  FSigmas[MaxSigmaPos + 1] := 0;
  ResetMultistep;
end;

procedure TNNetFlowMatchEulerScheduler.SetTimesteps(NumSteps: integer;
  Mu: double);
var
  Sigmas: array of double;
  SigmaPos, MaxSigmaPos: integer;
  SigmaStride: double;
begin
  if NumSteps < 1 then
    raise Exception.Create('Flow-matching NumSteps must be >= 1.');
  SetLength(Sigmas, NumSteps);
  MaxSigmaPos := NumSteps - 1;
  if MaxSigmaPos > 0 then SigmaStride := (1 / NumSteps - 1) / MaxSigmaPos
  else SigmaStride := 0;
  for SigmaPos := 0 to MaxSigmaPos do
    Sigmas[SigmaPos] := 1 + SigmaPos * SigmaStride;
  // numpy linspace pins the endpoint exactly.
  Sigmas[MaxSigmaPos] := 1 / NumSteps;
  SetSigmas(Sigmas, Mu);
end;

function TNNetFlowMatchEulerScheduler.GetNumSteps: integer;
begin
  Result := Length(FSigmas) - 1;
  if Result < 0 then Result := 0;
end;

function TNNetFlowMatchEulerScheduler.GetSigma(StepIndex: integer): double;
begin
  Result := FSigmas[StepIndex];
end;

function TNNetFlowMatchEulerScheduler.GetTimestep(StepIndex: integer): double;
begin
  Result := FSigmas[StepIndex] * FNumTrainTimesteps;
end;

procedure TNNetFlowMatchEulerScheduler.Step(Sample, Velocity: TNNetVolume;
  StepIndex: integer);
begin
  if (StepIndex < 0) or (StepIndex >= GetNumSteps) then
    raise Exception.Create('Flow-matching step index ' + IntToStr(StepIndex) +
      ' outside 0..' + IntToStr(GetNumSteps - 1) + '.');
  case FSampler of
    fsDPMSolverPP2M: StepDPMSolverPP2M(Sample, Velocity, StepIndex);
    fsUniPC: StepUniPC(Sample, Velocity, StepIndex);
  else
    Sample.MulAdd(FSigmas[StepIndex + 1] - FSigmas[StepIndex], Velocity);
  end;
end;

procedure TNNetFlowMatchEulerScheduler.ResetMultistep;
begin
  FPrevStepIndex := -1;
end;

procedure TNNetFlowMatchEulerScheduler.SetSampler(Value: TNNetFlowSampler);
begin
  FSampler := Value;
  ResetMultistep;
end;

procedure TNNetFlowMatchEulerScheduler.StepDPMSolverPP2M(Sample,
  Velocity: TNNetVolume; StepIndex: integer);
var
  SigmaS, SigmaT, SigmaPrev, AlphaS, AlphaT, AlphaPrev: double;
  LambdaS, HistoryWeight: double;
  SampleScale, DataScale, CurrentX0Scale, PrevX0Scale, StepSigma: TNeuralFloat;
  CurrentX0: TNeuralFloat;
  ElementPos, MaxElementPos: integer;
  HasHistory: boolean;
begin
  // alpha = 1 - sigma, lambda = ln(alpha/sigma), x0 = x - sigma*v, and
  // x_next = (sigma_t/sigma_s)*x - alpha_t*(exp(-h) - 1)*D (Lu et al. 2022).
  if Velocity.Size <> Sample.Size then
    raise Exception.Create('Flow-matching DPM-Solver++ step: ' +
      IntToStr(Velocity.Size) + ' velocity values for ' +
      IntToStr(Sample.Size) + ' sample values.');
  SigmaS := FSigmas[StepIndex];
  SigmaT := FSigmas[StepIndex + 1];
  StepSigma := SigmaS;
  if SigmaT <= 0 then
  begin
    // Final step to sigma 0 (lambda = +inf): first order, x_next = x0.
    Sample.MulAdd(-StepSigma, Velocity);
    FPrevStepIndex := -1;
    Exit;
  end;
  AlphaS := 1 - SigmaS;
  AlphaT := 1 - SigmaT;
  // -alpha_t*(exp(-h) - 1) with exp(-h) = sigma_t*alpha_s/(alpha_t*sigma_s),
  // finite at sigma_s = 1 (lambda_s = -inf).
  SampleScale := SigmaT / SigmaS;
  DataScale := AlphaT - SigmaT * AlphaS / SigmaS;
  if not Assigned(FPrevX0) then FPrevX0 := TNNetVolume.Create(Sample)
  else if FPrevX0.Size <> Sample.Size then
  begin
    FPrevX0.ReSize(Sample);
    FPrevStepIndex := -1;
  end;
  HistoryWeight := 0;
  if (StepIndex > 0) and (FPrevStepIndex = StepIndex - 1) and (AlphaS > 0) then
  begin
    SigmaPrev := FSigmas[StepIndex - 1];
    AlphaPrev := 1 - SigmaPrev;
    // AlphaPrev = 0 is lambda_prev = -inf: r = inf, so the weight is 0.
    if AlphaPrev > 0 then
    begin
      LambdaS := Ln(AlphaS / SigmaS);
      HistoryWeight := DPMSolverPP2MHistoryWeight(
        LambdaS - Ln(AlphaPrev / SigmaPrev), Ln(AlphaT / SigmaT) - LambdaS);
    end;
  end;
  HasHistory := HistoryWeight <> 0;
  MaxElementPos := Sample.Size - 1;
  // One fused pass that also stores x0 for the next step: composing it from
  // bulk ops needs a second x0 volume and four passes.
  if HasHistory then
  begin
    CurrentX0Scale := DataScale * (1 + HistoryWeight);
    PrevX0Scale := -DataScale * HistoryWeight;
    for ElementPos := 0 to MaxElementPos do
    begin
      CurrentX0 := Sample.FData[ElementPos] - StepSigma * Velocity.FData[ElementPos];
      Sample.FData[ElementPos] := SampleScale * Sample.FData[ElementPos] +
        CurrentX0Scale * CurrentX0 + PrevX0Scale * FPrevX0.FData[ElementPos];
      FPrevX0.FData[ElementPos] := CurrentX0;
    end;
  end
  else
  begin
    for ElementPos := 0 to MaxElementPos do
    begin
      CurrentX0 := Sample.FData[ElementPos] - StepSigma * Velocity.FData[ElementPos];
      Sample.FData[ElementPos] := SampleScale * Sample.FData[ElementPos] +
        DataScale * CurrentX0;
      FPrevX0.FData[ElementPos] := CurrentX0;
    end;
  end;
  FPrevStepIndex := StepIndex;
end;

function TNNetFlowMatchEulerScheduler.UniPCSigma(StepIndex: integer): double;
const
  cSigmaOneNudge: double = 1e-6;
var
  NudgedSigma: single;
begin
  Result := FSigmas[StepIndex];
  if Abs(Result - 1) < cSigmaOneNudge then
  begin
    NudgedSigma := Result - cSigmaOneNudge;
    Result := NudgedSigma;
  end;
end;

procedure TNNetFlowMatchEulerScheduler.StepUniPC(Sample, Velocity: TNNetVolume;
  StepIndex: integer);
var
  SigmaS, SigmaT, SigmaPrev, AlphaS, AlphaT, LambdaS, LambdaT, LambdaPrev: double;
  StepH, Rk, WeightM0, WeightM1, WeightMT: double;
  StepSigma, PredSampleScale, PredCurrentScale, PredPrevScale: TNeuralFloat;
  CorrSampleScale, CorrPrevScale, CorrOlderScale, CorrCurrentScale: TNeuralFloat;
  CurrentX0, CorrectedSample: TNeuralFloat;
  ElementPos, MaxElementPos, PredOrder: integer;
  HasHistory, UseCorrector: boolean;
begin
  // alpha = 1 - sigma, lambda = ln(alpha/sigma), x0 = x - sigma*v. The UniC
  // corrector rewrites this step's sample from the previous step's corrected
  // sample with this step's x0 (diffusers multistep_uni_c_bh_update), then the
  // UniP predictor steps from the corrected sample (multistep_uni_p_bh_update).
  // Both are affine in the stored x0 values (UniPCWeights).
  if Velocity.Size <> Sample.Size then
    raise Exception.Create('Flow-matching UniPC step: ' +
      IntToStr(Velocity.Size) + ' velocity values for ' +
      IntToStr(Sample.Size) + ' sample values.');
  SigmaS := UniPCSigma(StepIndex);
  SigmaT := FSigmas[StepIndex + 1];
  StepSigma := SigmaS;
  if SigmaT <= 0 then
  begin
    // Last step to sigma 0 (lower_order_final, first order): x_next is the x0
    // of the uncorrected sample, the corrector cannot reach it. diffusers
    // evaluates this limit as NaN for bh1.
    Sample.MulAdd(-StepSigma, Velocity);
    FPrevStepIndex := -1;
    Exit;
  end;
  if not Assigned(FPrevX0) then
  begin
    FPrevX0 := TNNetVolume.Create(Sample);
    FPrevStepIndex := -1;
  end;
  if not Assigned(FPrevX0Older) then FPrevX0Older := TNNetVolume.Create(Sample);
  if not Assigned(FPrevSample) then FPrevSample := TNNetVolume.Create(Sample);
  if (FPrevX0.Size <> Sample.Size) or (FPrevX0Older.Size <> Sample.Size) or
    (FPrevSample.Size <> Sample.Size) then
  begin
    FPrevX0.ReSize(Sample);
    FPrevX0Older.ReSize(Sample);
    FPrevSample.ReSize(Sample);
    FPrevStepIndex := -1;
  end;
  HasHistory := (StepIndex > 0) and (FPrevStepIndex = StepIndex - 1);
  if not HasHistory then FHistoryLen := 0;
  AlphaS := 1 - SigmaS;
  AlphaT := 1 - SigmaT;
  LambdaS := Ln(AlphaS / SigmaS);
  LambdaT := Ln(AlphaT / SigmaT);
  UseCorrector := HasHistory and FUniPCCorrector;
  CorrSampleScale := 0;
  CorrPrevScale := 0;
  CorrOlderScale := 0;
  CorrCurrentScale := 0;
  LambdaPrev := 0;
  if HasHistory then
  begin
    SigmaPrev := UniPCSigma(StepIndex - 1);
    LambdaPrev := Ln((1 - SigmaPrev) / SigmaPrev);
  end;
  if UseCorrector then
  begin
    // UniC from step StepIndex-1 to this sigma at the previous predictor's order.
    StepH := LambdaS - LambdaPrev;
    if (FPrevOrder >= 2) and (FHistoryLen >= 2) then
    begin
      SigmaPrev := UniPCSigma(StepIndex - 2);
      Rk := (Ln((1 - SigmaPrev) / SigmaPrev) - LambdaPrev) / StepH;
      PredOrder := 2;
    end
    else
    begin
      Rk := 0;
      PredOrder := 1;
    end;
    UniPCWeights(-StepH, Rk, AlphaS, PredOrder, true, FUniPCSolverType,
      WeightM0, WeightM1, WeightMT);
    CorrSampleScale := SigmaS / UniPCSigma(StepIndex - 1);
    CorrPrevScale := WeightM0;
    CorrOlderScale := WeightM1;
    CorrCurrentScale := WeightMT;
  end;
  // UniP to the next sigma: order 2 with history, first order on a fresh start
  // (the last step, lower_order_final, left above).
  StepH := LambdaT - LambdaS;
  if HasHistory then
  begin
    Rk := (LambdaPrev - LambdaS) / StepH;
    PredOrder := 2;
  end
  else
  begin
    Rk := 0;
    PredOrder := 1;
  end;
  UniPCWeights(-StepH, Rk, AlphaT, PredOrder, false, FUniPCSolverType,
    WeightM0, WeightM1, WeightMT);
  PredSampleScale := SigmaT / SigmaS;
  PredCurrentScale := WeightM0;
  PredPrevScale := WeightM1;
  MaxElementPos := Sample.Size - 1;
  // One fused pass per case: x0, the corrected sample, the predictor and the
  // history shift (older := prev, prev := x0, last sample := corrected).
  if UseCorrector then
  begin
    for ElementPos := 0 to MaxElementPos do
    begin
      CurrentX0 := Sample.FData[ElementPos] - StepSigma * Velocity.FData[ElementPos];
      CorrectedSample := CorrSampleScale * FPrevSample.FData[ElementPos] +
        CorrPrevScale * FPrevX0.FData[ElementPos] +
        CorrOlderScale * FPrevX0Older.FData[ElementPos] +
        CorrCurrentScale * CurrentX0;
      Sample.FData[ElementPos] := PredSampleScale * CorrectedSample +
        PredCurrentScale * CurrentX0 + PredPrevScale * FPrevX0.FData[ElementPos];
      FPrevSample.FData[ElementPos] := CorrectedSample;
      FPrevX0Older.FData[ElementPos] := FPrevX0.FData[ElementPos];
      FPrevX0.FData[ElementPos] := CurrentX0;
    end;
  end
  else if HasHistory then
  begin
    for ElementPos := 0 to MaxElementPos do
    begin
      CurrentX0 := Sample.FData[ElementPos] - StepSigma * Velocity.FData[ElementPos];
      FPrevSample.FData[ElementPos] := Sample.FData[ElementPos];
      Sample.FData[ElementPos] := PredSampleScale * Sample.FData[ElementPos] +
        PredCurrentScale * CurrentX0 + PredPrevScale * FPrevX0.FData[ElementPos];
      FPrevX0Older.FData[ElementPos] := FPrevX0.FData[ElementPos];
      FPrevX0.FData[ElementPos] := CurrentX0;
    end;
  end
  else
  begin
    // Fresh start: the older x0 slot gets this x0 too, so a weight-0 read of it
    // on the next corrector never touches stale data.
    for ElementPos := 0 to MaxElementPos do
    begin
      CurrentX0 := Sample.FData[ElementPos] - StepSigma * Velocity.FData[ElementPos];
      FPrevSample.FData[ElementPos] := Sample.FData[ElementPos];
      Sample.FData[ElementPos] := PredSampleScale * Sample.FData[ElementPos] +
        PredCurrentScale * CurrentX0;
      FPrevX0Older.FData[ElementPos] := CurrentX0;
      FPrevX0.FData[ElementPos] := CurrentX0;
    end;
  end;
  if FHistoryLen < 2 then Inc(FHistoryLen);
  FPrevOrder := PredOrder;
  FPrevStepIndex := StepIndex;
end;

class function TNNetFlowMatchEulerScheduler.Img2ImgStartStep(NumSteps: integer;
  Strength: double): integer;
var
  InitStepCount, StartStepExact: double;
begin
  if NumSteps < 1 then
    raise Exception.Create('Flow-matching NumSteps must be >= 1.');
  if not ((Strength >= 0) and (Strength <= 1)) then
    raise Exception.Create('img2img strength ' + FloatToStr(Strength) +
      ' is outside [0, 1].');
  // Double throughout (Min/Max would pick their single overloads), so the
  // truncation lands where Python's does.
  InitStepCount := NumSteps * Strength;
  if InitStepCount > NumSteps then InitStepCount := NumSteps;
  StartStepExact := NumSteps - InitStepCount;
  if StartStepExact < 0 then StartStepExact := 0;
  Result := Trunc(StartStepExact);
  if Result >= NumSteps then
    raise Exception.Create('img2img strength ' + FloatToStr(Strength) +
      ' with ' + IntToStr(NumSteps) + ' steps leaves no step to run; raise ' +
      'the strength or the step count.');
end;

function DiffusersSchedulerSampler(const ConfigJson: string;
  out Notice: string): TNNetSamplerMethod;
var
  Root: TJSONData;
  Obj: TJSONObject;
  SchedulerClass: string;

  procedure Note(const Text: string);
  begin
    if Notice <> '' then Notice := Notice + '; ';
    Notice := Notice + Text;
  end;

  procedure NoteIfSet(const Key: string);
  begin
    if Obj.Get(Key, false) then
      Note(SchedulerClass + ' ' + Key + ' is not implemented; sampling without it');
  end;

  procedure NoteIfNot(const Key, Expected: string);
  begin
    if Obj.Get(Key, Expected) <> Expected then
      Note(SchedulerClass + ' ' + Key + ' "' + Obj.Get(Key, Expected) +
        '" is not implemented; sampling with "' + Expected + '"');
  end;

begin
  Notice := '';
  Result := smDDIM;
  Root := GetJSON(ConfigJson);
  try
    if not (Root is TJSONObject) then
    begin
      Note('the scheduler config is not a JSON object; sampling with DDIM');
      Exit;
    end;
    Obj := TJSONObject(Root);
    SchedulerClass := Obj.Get('_class_name', '');
    if SchedulerClass = 'DDIMScheduler' then Result := smDDIM
    else if SchedulerClass = 'PNDMScheduler' then
    begin
      Result := smPNDM;
      if not Obj.Get('skip_prk_steps', false) then
        Note('PNDMScheduler skip_prk_steps = false: the Runge-Kutta warm-up ' +
          'is not implemented; sampling with PLMS from the first step');
    end
    else if SchedulerClass = 'EulerDiscreteScheduler' then
    begin
      Result := smEuler;
      if Obj.Get('use_karras_sigmas', false) and
        ((Obj.Find('sigma_min') is TJSONNumber) or
        (Obj.Find('sigma_max') is TJSONNumber)) then
        Note(SchedulerClass + ' sigma_min / sigma_max is not implemented; ' +
          'the Karras range comes from the timestep schedule');
      NoteIfSet('use_exponential_sigmas');
      NoteIfSet('use_beta_sigmas');
      NoteIfNot('interpolation_type', 'linear');
      NoteIfNot('timestep_type', 'discrete');
      NoteIfNot('final_sigmas_type', 'zero');
    end
    else if SchedulerClass = 'EulerAncestralDiscreteScheduler' then
      Result := smEulerAncestral
    else if SchedulerClass = 'DPMSolverMultistepScheduler' then
    begin
      Result := smDPMSolverPP2M;
      NoteIfNot('algorithm_type', 'dpmsolver++');
      NoteIfNot('solver_type', 'midpoint');
      if Obj.Get('solver_order', 2) <> 2 then
        Note(SchedulerClass + ' solver_order ' +
          IntToStr(Obj.Get('solver_order', 2)) +
          ' is not implemented; sampling with order 2');
      NoteIfSet('use_lu_lambdas');
      NoteIfSet('use_exponential_sigmas');
      NoteIfSet('use_beta_sigmas');
      NoteIfNot('final_sigmas_type', 'zero');
    end
    else if SchedulerClass = 'LCMScheduler' then Result := smLCM
    else
      Note('scheduler "' + SchedulerClass + '" has no sampler here; ' +
        'sampling with DDIM');
  finally
    Root.Free;
  end;
end;

function DiffusersSchedulerUsesKarrasSigmas(const ConfigJson: string): boolean;
var
  Root: TJSONData;
  SchedulerClass: string;
begin
  Result := false;
  Root := GetJSON(ConfigJson);
  try
    if not (Root is TJSONObject) then Exit;
    SchedulerClass := TJSONObject(Root).Get('_class_name', '');
    Result := ((SchedulerClass = 'EulerDiscreteScheduler') or
      (SchedulerClass = 'DPMSolverMultistepScheduler')) and
      TJSONObject(Root).Get('use_karras_sigmas', false);
  finally
    Root.Free;
  end;
end;

procedure InpaintBlend(Latents, Known, Mask: TNNetVolume);
begin
  if (Known.Size <> Latents.Size) or (Mask.Size <> Latents.Size) then
    raise Exception.Create('InpaintBlend: ' + IntToStr(Latents.Size) +
      ' latent values, ' + IntToStr(Known.Size) + ' known and ' +
      IntToStr(Mask.Size) + ' mask values.');
  // m*L + (1-m)*K = K + m*(L-K), in place (rule #13, no scratch).
  Latents.Sub(Known);
  TNNetVolume.Mul(Latents.GetRawPtr(0), Mask.GetRawPtr(0), Latents.Size);
  Latents.Add(Known);
end;

procedure TNNetFlowMatchEulerScheduler.ScaleNoise(Noise, Sample: TNNetVolume;
  StepIndex: integer);
var
  StartSigma: double;
begin
  if (StepIndex < 0) or (StepIndex >= GetNumSteps) then
    raise Exception.Create('Flow-matching step index ' + IntToStr(StepIndex) +
      ' outside 0..' + IntToStr(GetNumSteps - 1) + '.');
  if Noise.Size <> Sample.Size then
    raise Exception.Create('ScaleNoise: noise has ' + IntToStr(Noise.Size) +
      ' values, the sample ' + IntToStr(Sample.Size) + '.');
  StartSigma := FSigmas[StepIndex];
  Noise.Mul(StartSigma);
  Noise.MulAdd(1 - StartSigma, Sample);
end;

procedure ComputeTileStarts(Extent, TileLen, Stride: integer;
  out Starts: TNeuralIntegerArray; Circular: boolean);
var
  LastStart, StartCount, MaxStartPos, StartPos: integer;
begin
  LastStart := Extent - TileLen;
  StartCount := 1;
  if LastStart > 0 then
  begin
    if Circular
      then StartCount := (Extent + Stride - 1) div Stride
      else StartCount := (LastStart + Stride - 1) div Stride + 1;
  end;
  Starts := nil;
  SetLength(Starts, StartCount);
  MaxStartPos := StartCount - 1;
  for StartPos := 0 to MaxStartPos do
    Starts[StartPos] := StartPos * Stride;
  if (LastStart >= 0) and not Circular then
    Starts[StartCount - 1] := LastStart;
end;

{ TMultiDiffusionWindows }

constructor TMultiDiffusionWindows.Create();
begin
  inherited Create();
  FWindowInput := TNNetVolume.Create();
  FWindowOutput := TNNetVolume.Create();
  FAccumulator := TNNetVolume.Create();
end;

destructor TMultiDiffusionWindows.Destroy();
begin
  FWindowInput.Free;
  FWindowOutput.Free;
  FAccumulator.Free;
  inherited Destroy();
end;

procedure TMultiDiffusionWindows.Plan(CanvasWidth, CanvasHeight, WindowWidth,
  WindowHeight, StrideX, StrideY: integer; CircularX: boolean);
var
  CellCoverage: TNeuralIntegerArray;
  WindowIdx, MaxWindowPos, RowPos, MaxRowPos, ColPos, MaxColPos, CellPos,
    MaxCellPos, RowBase: integer;
begin
  if (CanvasWidth < 1) or (CanvasHeight < 1) or (WindowWidth < 1) or
     (WindowHeight < 1) then
    raise Exception.Create('TMultiDiffusionWindows.Plan: canvas ' +
      IntToStr(CanvasWidth) + 'x' + IntToStr(CanvasHeight) + ' or window ' +
      IntToStr(WindowWidth) + 'x' + IntToStr(WindowHeight) + ' is empty.');
  if WindowWidth > CanvasWidth then WindowWidth := CanvasWidth;
  if WindowHeight > CanvasHeight then WindowHeight := CanvasHeight;
  if (StrideX < 1) or (StrideX > WindowWidth) or (StrideY < 1) or
     (StrideY > WindowHeight) then
    raise Exception.Create('TMultiDiffusionWindows.Plan: stride ' +
      IntToStr(StrideX) + 'x' + IntToStr(StrideY) + ' must be in 1..the ' +
      'window side (' + IntToStr(WindowWidth) + 'x' +
      IntToStr(WindowHeight) + ') so every cell is covered.');
  FCanvasWidth := CanvasWidth;
  FCanvasHeight := CanvasHeight;
  FWindowWidth := WindowWidth;
  FWindowHeight := WindowHeight;
  FCircularX := CircularX;
  ComputeTileStarts(CanvasWidth, WindowWidth, StrideX, FStartsX, CircularX);
  ComputeTileStarts(CanvasHeight, WindowHeight, StrideY, FStartsY);
  CellCoverage := nil;
  SetLength(CellCoverage, CanvasWidth * CanvasHeight);
  MaxRowPos := WindowHeight - 1;
  MaxColPos := WindowWidth - 1;
  MaxWindowPos := GetWindowCount() - 1;
  for WindowIdx := 0 to MaxWindowPos do
    for RowPos := 0 to MaxRowPos do
    begin
      RowBase := (GetWindowY(WindowIdx) + RowPos) * CanvasWidth;
      for ColPos := 0 to MaxColPos do
        Inc(CellCoverage[RowBase +
          (GetWindowX(WindowIdx) + ColPos) mod CanvasWidth]);
    end;
  FInvCoverage := nil;
  SetLength(FInvCoverage, CanvasWidth * CanvasHeight);
  MaxCellPos := CanvasWidth * CanvasHeight - 1;
  for CellPos := 0 to MaxCellPos do
    FInvCoverage[CellPos] := 1 / CellCoverage[CellPos];
end;

function TMultiDiffusionWindows.GetWindowCount(): integer;
begin
  Result := Length(FStartsX) * Length(FStartsY);
end;

function TMultiDiffusionWindows.GetWindowX(WindowIdx: integer): integer;
begin
  Result := FStartsX[WindowIdx mod Length(FStartsX)];
end;

function TMultiDiffusionWindows.GetWindowY(WindowIdx: integer): integer;
begin
  Result := FStartsY[WindowIdx div Length(FStartsX)];
end;

function TMultiDiffusionWindows.GetCoverage(CellX, CellY: integer): integer;
begin
  Result := Round(1 / FInvCoverage[CellY * FCanvasWidth + CellX]);
end;

procedure TMultiDiffusionWindows.CheckCanvas(Canvas: TNNetVolume;
  const Caller: string);
var
  CellCount: integer;
begin
  CellCount := FCanvasWidth * FCanvasHeight;
  if (CellCount = 0) or not (((Canvas.SizeX = FCanvasWidth) and
     (Canvas.SizeY = FCanvasHeight)) or ((Canvas.SizeX = CellCount) and
     (Canvas.SizeY = 1))) then
    raise Exception.Create('TMultiDiffusionWindows.' + Caller + ': a ' +
      IntToStr(Canvas.SizeX) + 'x' + IntToStr(Canvas.SizeY) + ' volume is ' +
      'not the planned ' + IntToStr(FCanvasWidth) + 'x' +
      IntToStr(FCanvasHeight) + ' canvas (call Plan first).');
end;

procedure TMultiDiffusionWindows.ShapeWindow(Canvas, Window: TNNetVolume;
  Depth: integer);
begin
  if (Canvas.SizeX = FCanvasWidth) and (Canvas.SizeY = FCanvasHeight)
    then Window.ReSize(FWindowWidth, FWindowHeight, Depth)
    else Window.ReSize(FWindowWidth * FWindowHeight, 1, Depth);
end;

procedure TMultiDiffusionWindows.TransferWindow(Canvas, Window: TNNetVolume;
  WindowIdx: integer; IsGather: boolean);
var
  Depth, StartX, StartY, FirstRunFloats, WrapRunFloats, WindowRowFloats,
    CanvasRowFloats, CanvasRowBase, CanvasWrapBase, WindowRowBase, RowPos,
    MaxRowPos, FirstRunCells: integer;
begin
  Depth := Window.Depth;
  StartX := GetWindowX(WindowIdx);
  StartY := GetWindowY(WindowIdx);
  // A wrapped window row is two runs: StartX..the right edge, then from 0.
  FirstRunCells := FCanvasWidth - StartX;
  if FirstRunCells > FWindowWidth then FirstRunCells := FWindowWidth;
  FirstRunFloats := FirstRunCells * Depth;
  WindowRowFloats := FWindowWidth * Depth;
  WrapRunFloats := WindowRowFloats - FirstRunFloats;
  CanvasRowFloats := FCanvasWidth * Depth;
  CanvasWrapBase := StartY * CanvasRowFloats;
  CanvasRowBase := CanvasWrapBase + StartX * Depth;
  WindowRowBase := 0;
  MaxRowPos := FWindowHeight - 1;
  for RowPos := 0 to MaxRowPos do
  begin
    if IsGather then
    begin
      Move(Canvas.FData[CanvasRowBase], Window.FData[WindowRowBase],
        FirstRunFloats * csNeuralFloatSize);
      if WrapRunFloats > 0 then
        Move(Canvas.FData[CanvasWrapBase],
          Window.FData[WindowRowBase + FirstRunFloats],
          WrapRunFloats * csNeuralFloatSize);
    end
    else
    begin
      TNNetVolume.Add(Canvas.GetRawPtr(CanvasRowBase),
        Window.GetRawPtr(WindowRowBase), FirstRunFloats);
      if WrapRunFloats > 0 then
        TNNetVolume.Add(Canvas.GetRawPtr(CanvasWrapBase),
          Window.GetRawPtr(WindowRowBase + FirstRunFloats), WrapRunFloats);
    end;
    Inc(CanvasRowBase, CanvasRowFloats);
    Inc(CanvasWrapBase, CanvasRowFloats);
    Inc(WindowRowBase, WindowRowFloats);
  end;
end;

procedure TMultiDiffusionWindows.GatherWindow(Canvas: TNNetVolume;
  WindowIdx: integer; Window: TNNetVolume);
begin
  CheckCanvas(Canvas, 'GatherWindow');
  ShapeWindow(Canvas, Window, Canvas.Depth);
  TransferWindow(Canvas, Window, WindowIdx, true);
end;

procedure TMultiDiffusionWindows.ClearAccumulator(Depth: integer);
begin
  FAccumulator.ReSize(FCanvasWidth * FCanvasHeight, 1, Depth);
  FAccumulator.Fill(0);
end;

procedure TMultiDiffusionWindows.AccumulateWindow(WindowIdx: integer;
  Window: TNNetVolume);
begin
  if (Window.Depth <> FAccumulator.Depth) or
     (Window.Size <> FWindowWidth * FWindowHeight * Window.Depth) then
    raise Exception.Create('TMultiDiffusionWindows.AccumulateWindow: ' +
      IntToStr(Window.Size) + ' values do not form a ' +
      IntToStr(FWindowWidth) + 'x' + IntToStr(FWindowHeight) + 'x' +
      IntToStr(FAccumulator.Depth) + ' window (ClearAccumulator first).');
  TransferWindow(FAccumulator, Window, WindowIdx, false);
end;

procedure TMultiDiffusionWindows.NormalizeInto(Canvas,
  CanvasOutput: TNNetVolume);
var
  Depth, CellPos, MaxCellPos: integer;
  CellInvCoverage: TNeuralFloat;
begin
  CheckCanvas(Canvas, 'NormalizeInto');
  Depth := FAccumulator.Depth;
  if (Canvas.SizeX = FCanvasWidth) and (Canvas.SizeY = FCanvasHeight)
    then CanvasOutput.ReSize(FCanvasWidth, FCanvasHeight, Depth)
    else CanvasOutput.ReSize(FCanvasWidth * FCanvasHeight, 1, Depth);
  Move(FAccumulator.FData[0], CanvasOutput.FData[0],
    FAccumulator.Size * csNeuralFloatSize);
  MaxCellPos := FCanvasWidth * FCanvasHeight - 1;
  for CellPos := 0 to MaxCellPos do
  begin
    CellInvCoverage := FInvCoverage[CellPos];
    if CellInvCoverage <> 1 then
      TNNetVolume.Mul(CanvasOutput.GetRawPtr(CellPos * Depth),
        CellInvCoverage, Depth);
  end;
end;

procedure TMultiDiffusionWindows.PredictCanvas(Canvas,
  CanvasOutput: TNNetVolume; Predict: TMultiDiffusionPredictEvent);
var
  WindowIdx, MaxWindowPos: integer;
begin
  MaxWindowPos := GetWindowCount() - 1;
  for WindowIdx := 0 to MaxWindowPos do
  begin
    GatherWindow(Canvas, WindowIdx, FWindowInput);
    Predict(WindowIdx, FWindowInput, FWindowOutput);
    if WindowIdx = 0 then ClearAccumulator(FWindowOutput.Depth);
    AccumulateWindow(WindowIdx, FWindowOutput);
  end;
  NormalizeInto(Canvas, CanvasOutput);
end;

end.
