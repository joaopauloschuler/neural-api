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
  * The three standard REVERSE updates, each one step:
      - DDPM ancestral (stochastic),
      - DDIM (deterministic for eta=0; eta>0 adds the calibrated stochastic term),
      - DPM-Solver++(2M) (a 2nd-order multistep update that keeps the previous
        model output for its correction term).
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
  TNNetSamplerMethod = (smDDPM, smDDIM, smDPMSolverPP2M);

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
    procedure BuildTables(Beta1, BetaT, CosineS: TNeuralFloat);
    function GetBeta(Tt: integer): TNeuralFloat;
    function GetAlpha(Tt: integer): TNeuralFloat;
    function GetAlphaBar(Tt: integer): TNeuralFloat;
    // log(sqrt(ab/(1-ab))) half-log-SNR lambda used by DPM-Solver++.
    function Lambda(Tt: integer): TNeuralFloat;
    // Convert a raw model prediction (eps or v) at timestep Tt into eps.
    procedure ToEps(Xt, RawPred, EpsOut: TNNetVolume; Tt: integer);
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
      W: TNeuralFloat);

    // Reset the DPM-Solver++ multistep history. Call before each new trajectory.
    procedure ResetMultistep;

    // ONE reverse step from timestep Tt to TtPrev (TtPrev=0 -> clean image), in
    // place on Xt. RawPred is the model's raw prediction (eps or v) at Tt.
    // Method selects the update rule; Eta is the DDIM stochasticity (0 = the
    // deterministic DDIM used by the examples; ignored by DDPM/DPM++). For DDPM
    // TtPrev is ignored (it uses Tt-1 by construction).
    procedure Step(Xt, RawPred: TNNetVolume; Tt, TtPrev: integer;
      Method: TNNetSamplerMethod = smDDIM; Eta: TNeuralFloat = 0.0);

    // High-level driver: run a complete reverse trajectory in place on X
    // (which should start as N(0,I) noise) over NumSteps evenly-strided
    // timesteps using the supplied Denoise callback. Leaves the sampled x0 in X.
    procedure Sample(X: TNNetVolume; Denoise: TNNetDenoiseCallback;
      NumSteps: integer; Method: TNNetSamplerMethod = smDDIM;
      Eta: TNeuralFloat = 0.0);
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
  FHasPrev := false;
end;

destructor TNNetDiffusionScheduler.Destroy;
begin
  if Assigned(FPrevX0) then FPrevX0.Free;
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

procedure TNNetDiffusionScheduler.ToEps(Xt, RawPred, EpsOut: TNNetVolume; Tt: integer);
var
  i: integer;
  sab, somab: TNeuralFloat;
begin
  if FPrediction = dpEps then
  begin
    if EpsOut <> RawPred then EpsOut.Copy(RawPred);
    Exit;
  end;
  // v-prediction: eps = sqrt(ab)*v + sqrt(1-ab)*x_t.
  sab := FSqrtAlphaBar[Tt];
  somab := FSqrtOneMinusAlphaBar[Tt];
  for i := 0 to EpsOut.Size - 1 do
    EpsOut.FData[i] := sab * RawPred.FData[i] + somab * Xt.FData[i];
end;

procedure TNNetDiffusionScheduler.AddNoise(X0, Xt: TNNetVolume; Tt: integer;
  NoiseOut, PreSampledNoise: TNNetVolume);
var
  i: integer;
  sab, somab, eps: TNeuralFloat;
begin
  sab := FSqrtAlphaBar[Tt];
  somab := FSqrtOneMinusAlphaBar[Tt];
  for i := 0 to X0.Size - 1 do
  begin
    if Assigned(PreSampledNoise) then eps := PreSampledNoise.FData[i]
    else eps := RandG(0, 1);
    if Assigned(NoiseOut) then NoiseOut.FData[i] := eps;
    Xt.FData[i] := sab * X0.FData[i] + somab * eps;
  end;
end;

class procedure TNNetDiffusionScheduler.ApplyCFG(EpsCond, EpsUncond,
  Dst: TNNetVolume; W: TNeuralFloat);
var i: integer;
begin
  for i := 0 to Dst.Size - 1 do
    Dst.FData[i] := EpsUncond.FData[i] +
      W * (EpsCond.FData[i] - EpsUncond.FData[i]);
end;

procedure TNNetDiffusionScheduler.ResetMultistep;
begin
  FHasPrev := false;
end;

procedure TNNetDiffusionScheduler.Step(Xt, RawPred: TNNetVolume;
  Tt, TtPrev: integer; Method: TNNetSamplerMethod; Eta: TNeuralFloat);
var
  i: integer;
  Eps: TNNetVolume;
  invSqrtAlpha, coef, sigma, z: TNeuralFloat;
  abT, abPrev, sqrtAbPrev, x0, dirCoef: TNeuralFloat;
  // DPM-Solver++ locals.
  lamT, lamPrev, h, hLast, r0: TNeuralFloat;
  curX0, dCoef: TNeuralFloat;
begin
  Eps := TNNetVolume.Create(Xt);
  try
    // All updates work from eps; v-prediction is converted up front.
    ToEps(Xt, RawPred, Eps, Tt);
    case Method of
      smDDPM:
        begin
          // x_{Tt-1} = 1/sqrt(a_t)*(x_t - beta_t/sqrt(1-ab_t)*eps) + sigma*z.
          invSqrtAlpha := 1.0 / Sqrt(FAlpha[Tt]);
          coef := FBeta[Tt] / FSqrtOneMinusAlphaBar[Tt];
          if Tt > 1 then sigma := Sqrt(FBeta[Tt]) else sigma := 0;
          for i := 0 to Xt.Size - 1 do
          begin
            if Tt > 1 then z := RandG(0, 1) else z := 0;
            Xt.FData[i] := invSqrtAlpha *
              (Xt.FData[i] - coef * Eps.FData[i]) + sigma * z;
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
          for i := 0 to Xt.Size - 1 do
          begin
            if sigma > 0 then z := RandG(0, 1) else z := 0;
            // x_{TtPrev} = sqrt(ab_prev)*x0 + dirCoef*eps + sigma*z,
            //   x0 = (x_t - somab_t*eps)/sqrt(ab_t).
            x0 := (Xt.FData[i] - FSqrtOneMinusAlphaBar[Tt] * Eps.FData[i]) / Sqrt(abT);
            Xt.FData[i] := sqrtAbPrev * x0 + dirCoef * Eps.FData[i] + sigma * z;
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
          // Convert current eps -> x0 prediction; reuse Eps to hold curX0.
          for i := 0 to Xt.Size - 1 do
            Eps.FData[i] := (Xt.FData[i] - FSqrtOneMinusAlphaBar[Tt] * Eps.FData[i]) / Sqrt(abT);
          lamT := Lambda(Tt);
          if TtPrev = 0 then
          begin
            // Final hop to the clean image: emit the (corrected) x0 prediction.
            if FHasPrev then
            begin
              hLast := lamT - FPrevLambda;          // < 0
              h := -hLast;                          // approximate remaining step
              if Abs(h) < 1e-12 then r0 := 1.0 else r0 := hLast / h;
              for i := 0 to Xt.Size - 1 do
                Xt.FData[i] := (1.0 + 1.0 / (2.0 * r0)) * Eps.FData[i]
                  - (1.0 / (2.0 * r0)) * FPrevX0.FData[i];
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
          if not FHasPrev then
          begin
            // First-order (DDIM-equivalent) bootstrap.
            for i := 0 to Xt.Size - 1 do
              Xt.FData[i] := (sqrtAbPrev / Sqrt(abT)) * Xt.FData[i]
                - Sqrt(1.0 - abPrev) * (Exp(-h) - 1.0) * Eps.FData[i];
          end
          else
          begin
            hLast := lamT - FPrevLambda;            // < 0
            if Abs(h) < 1e-12 then r0 := 1.0 else r0 := hLast / h;
            for i := 0 to Xt.Size - 1 do
            begin
              curX0 := Eps.FData[i];
              dCoef := (1.0 + 1.0 / (2.0 * r0)) * curX0
                       - (1.0 / (2.0 * r0)) * FPrevX0.FData[i];
              Xt.FData[i] := (sqrtAbPrev / Sqrt(abT)) * Xt.FData[i]
                - Sqrt(1.0 - abPrev) * (Exp(-h) - 1.0) * dCoef;
            end;
          end;
          // Store this step's x0 and lambda for the next multistep correction.
          if not Assigned(FPrevX0) then FPrevX0 := TNNetVolume.Create(Xt);
          FPrevX0.Copy(Eps);   // Eps currently holds curX0
          FPrevLambda := lamT;
          FHasPrev := true;
        end;
    end;
  finally
    Eps.Free;
  end;
end;

procedure TNNetDiffusionScheduler.Sample(X: TNNetVolume;
  Denoise: TNNetDenoiseCallback; NumSteps: integer;
  Method: TNNetSamplerMethod; Eta: TNeuralFloat);
var
  k, Tt, TtPrev: integer;
  Pred: TNNetVolume;
begin
  if NumSteps < 1 then NumSteps := 1;
  ResetMultistep;
  Pred := TNNetVolume.Create(X);
  try
    for k := NumSteps - 1 downto 0 do
    begin
      // Map subsequence index k -> actual timestep in 1..T (evenly strided).
      if NumSteps = 1 then Tt := FT
      else Tt := 1 + Round(k * (FT - 1) / (NumSteps - 1));
      if k > 0 then
      begin
        if NumSteps = 1 then TtPrev := 0
        else TtPrev := 1 + Round((k - 1) * (FT - 1) / (NumSteps - 1));
      end
      else TtPrev := 0;
      Denoise(X, Pred, Tt);
      Step(X, Pred, Tt, TtPrev, Method, Eta);
    end;
  finally
    Pred.Free;
  end;
end;

end.
