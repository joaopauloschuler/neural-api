program ConsistencyDistill;
(*
ConsistencyDistill: FEW-STEP generative sampling by CONSISTENCY DISTILLATION
(Song et al. 2023, "Consistency Models", https://arxiv.org/abs/2303.01469).
The landed diffusion examples (DiffusionMNIST, ConditionalDiffusion,
FlowMatching) all generate with MANY reverse steps (25-200). This example
distils a trained MNIST diffusion TEACHER into a CONSISTENCY MODEL whose
self-consistency property lets it generate a digit in 1, 2 or 4 steps.

WHAT A CONSISTENCY MODEL IS. Diffusion defines a probability-flow ODE whose
trajectory carries a noisy point x_t (at any timestep t) back to its clean
origin x_0. A consistency model learns a function f(x_t, t) that maps ANY point
on a given trajectory DIRECTLY to that trajectory's origin x_0 -- so a single
network evaluation is already a (rough) sample. The defining "self-consistency"
property is that f gives the SAME answer for every t on one trajectory, and the
boundary condition f(x, t->0) = x pins it. Because one call lands near x_0, we
can sample in 1 step; 2-4 steps (re-noise + re-denoise) sharpen it.

BOUNDARY-CONDITION PARAMETERISATION. To enforce f(x,0)=x exactly we do NOT let
the network output x_0 directly. We wrap a raw network F_theta with skip/out
scalings (Karras-style):
    f(x,t) = c_skip(t) * x  +  c_out(t) * F_theta(x, t)
    c_skip(t) = sigma_data^2 / (sigma(t)^2 + sigma_data^2)
    c_out(t)  = sigma(t) * sigma_data / sqrt(sigma(t)^2 + sigma_data^2)
where sigma(t) is the noise level. At t->0 sigma->0 so c_skip->1, c_out->0 and
f(x,0)=x as required. We use the diffusion scheduler's alpha_bar to define the
VP-equivalent noise level sigma(t) = sqrt((1-alpha_bar_t)/alpha_bar_t) on the
"x0 + sigma*eps" scaled view (x_t / sqrt(alpha_bar_t)). c_skip/c_out are plain
example-side arithmetic -- NO new layer is needed.

CONSISTENCY DISTILLATION LOSS (the new code). With a PRETRAINED
teacher eps_phi we:
  1. draw clean x_0, pick adjacent timesteps t_{n+1} > t_n on a sub-grid;
  2. forward-noise x_0 to x_{t_{n+1}};
  3. take ONE deterministic teacher ODE (DDIM) step x_{t_{n+1}} -> x_{t_n}
     using eps_phi (this is the trajectory the student must be consistent on);
  4. minimise   || f_theta(x_{t_{n+1}}, t_{n+1}) - f_target(x_{t_n}, t_n) ||^2
     where f_target is an EMA copy of f_theta (stop-grad target network).
The EMA target net is the repo's TNNetEMAWrapper (neuralnetwork.pas): its shadow
net IS f_target and Update() folds the live student weights with decay.

The c_out(t_{n+1}) scaling is folded into the backprop target so the student
network F_theta is trained on the right residual: from the loss on f we get the
gradient on F_theta's raw output as (f_theta - f_target) * c_out(t_{n+1}), which
is exactly what we hand to Backpropagate.

FEW-STEP SAMPLER. Start from x_T ~ N(0, I) (on the scaled view). One step:
x0 = f(x_T, T). For K>1 steps we re-noise x0 to a lower timestep and call f
again, following the multistep consistency sampler:
    x0 = f(x_t, t);  if more steps: x_{t'} = x0 + sigma(t') * eps;  repeat.

THE NETWORKS reuse the DiffusionMNIST tiny time-conditioned U-Net wholesale
(sinusoidal time embedding -> shared cond MLP -> FiLM into every conv block,
TNNetDeepConcat skips, TNNetUpsample decoder). Two such nets are built: the
eps-prediction TEACHER and the raw-output STUDENT F_theta (+ its EMA target).

RUN MODES.
  default (SMOKE): short teacher pretrain + short distillation; finishes well
    under 5 min on one CPU. Reports 1/2/4-step vs teacher MSE-to-data and writes
    a PNG grid (rows: teacher-multistep, 1-step, 2-step, 4-step).
  --full : many more steps for sharper digits.

METRIC. For each sampler we report mean per-sample MSE of generated digits to
their NEAREST training digit (a cheap fidelity proxy -- lower is closer to the
data manifold) plus a NaN check. The headline is that the 1/2/4-step consistency
samples approach the multi-step teacher quality at a fraction of the steps.

DATA. Standard MNIST idx-ubyte files in the working directory (symlinked from
../DiffusionMNIST). If they are absent the program falls back to a SYNTHETIC
dataset (random axis-aligned bright bars) so the demo still runs in CI; the
pipeline and metrics are identical.

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
  neuraldiffusion,
  neuraldatasets;

const
  cImgSize   = 28;
  cEmbDim    = 64;          // sinusoidal time-embedding width (must be even)

  // U-Net channel widths (small for a fast CPU smoke run).
  cC1        = 16;
  cC2        = 32;
  cC3        = 48;

  // Linear beta schedule (Ho et al. 2020). Modest T for quick CPU sampling.
  cT         = 200;
  cBeta1     = 1.0e-4;
  cBetaT     = 0.02;

  // Consistency parameterisation: data std on the scaled (x0+sigma*eps) view.
  // MNIST in [-1,1] has roughly unit-scale features; 0.5 is a stable choice.
  cSigmaData = 0.5;

  // Number of discretisation points N on the consistency sub-grid (the
  // boundary t_1 .. t_N spanning 1..cT). Adjacent (n, n+1) pairs form the
  // teacher ODE step the student must be consistent on.
  cNumSub    = 18;

  cEMADecay  = 0.95;        // target-net EMA decay (short runs -> faster track)

  cLR        = 0.0005;
  cInertia   = 0.9;

  // Teacher multistep + few-step student sampler step counts demonstrated.
  cTeacherSteps = 16;       // teacher DDIM reverse steps (the "many-step" ref)

  // Default = fast smoke. --full overrides below. Each conv-U-Net fwd+bwd on
  // CPU is ~0.07s, and distillation does THREE forwards + one backward per
  // example, so the step counts/batch are kept small to stay well under the
  // 5-min budget; the headline is the relative 1/2/4-step-vs-teacher trend.
  cSmokeTeach  = 60;
  cSmokeDist   = 60;
  cSmokeBatch  = 8;
  cSmokeGrid   = 5;

  cFullTeach   = 4000;
  cFullDist    = 4000;
  cFullBatch   = 32;
  cFullGrid    = 10;

type
  // TNNetDenoiseCallback adapter driving the TEACHER eps-net through the
  // reusable scheduler's multi-step reverse loop (the "many-step" reference).
  // Coded by Claude (AI).
  TTeacherDenoiser = class(TObject)
  public
    Net: TNNet;
    TimeVol: TNNetVolume;
    procedure Denoise(Xt, Output: TNNetVolume; t: integer);
  end;

var
  Sched: TNNetDiffusionScheduler;

  Teacher, Student: TNNet;
  TargetEMA: TNNetEMAWrapper;          // shadow net = f_target (stop-grad)

  TrainV, ValV, TestV: TNNetVolumeList;
  UseSynthetic: boolean;

  // Reusable working volumes.
  Img0, ImgT, EpsTrue, TimeVol, SchedTime: TNNetVolume;

  TeachSteps, DistSteps, BatchSz, GridN: integer;
  FullMode: boolean;

  TeacherDenoiser: TTeacherDenoiser;

  // ---- tiny time-conditioned U-Net (reused from DiffusionMNIST) -------------
  // EpsHead=true builds the eps-prediction teacher; false builds the raw-output
  // student F_theta (same architecture, both heads are linear convs).
  CurNet: TNNet;
  CurTimeEmb: TNNetLayer;

  function ConvBlock(Channels: integer): TNNetLayer;
  var
    Feat: TNNetLayer;
  begin
    CurNet.AddLayer(TNNetConvolutionReLU.Create(Channels, 3, 1, 1, 0));
    if (Channels mod 4) = 0
      then CurNet.AddLayer(TNNetGroupNorm.Create(4))
      else CurNet.AddLayer(TNNetGroupNorm.Create(Channels));
    Feat := CurNet.GetLastLayer;
    CurNet.AddFiLMConditioned(Feat, CurTimeEmb);
    Result := CurNet.AddLayer(TNNetConvolutionReLU.Create(Channels, 3, 1, 1, 0));
  end;

  function BuildUNet: TNNet;
  var
    ImgIn, TimeIn: TNNetLayer;
    SkipA, SkipB: TNNetLayer;
  begin
    CurNet := TNNet.Create();
    ImgIn  := CurNet.AddLayer(TNNetInput.Create(cImgSize, cImgSize, 1, 1));
    TimeIn := CurNet.AddLayerAfter(TNNetInput.Create(1, 1, 1, 1), 0);

    CurTimeEmb := CurNet.AddLayerAfter(TNNetSinusoidalTimeEmbedding.Create(cEmbDim), TimeIn);
    CurTimeEmb := CurNet.AddLayerAfter(TNNetFullConnectReLU.Create(cEmbDim), CurTimeEmb);
    CurTimeEmb := CurNet.AddLayerAfter(TNNetFullConnectReLU.Create(cEmbDim), CurTimeEmb);

    CurNet.AddLayerAfter(TNNetConvolutionReLU.Create(cC1, 3, 1, 1, 0), ImgIn);
    SkipA := ConvBlock(cC1);
    CurNet.AddLayer(TNNetMaxPool.Create(2));      // -> 14x14
    SkipB := ConvBlock(cC2);
    CurNet.AddLayer(TNNetMaxPool.Create(2));      // -> 7x7
    ConvBlock(cC3);                               // bottleneck
    CurNet.AddLayer(TNNetUpsample.Create());      // 7 -> 14
    CurNet.AddLayer(TNNetDeepConcat.Create([CurNet.GetLastLayer, SkipB]));
    ConvBlock(cC2);
    CurNet.AddLayer(TNNetUpsample.Create());      // 14 -> 28
    CurNet.AddLayer(TNNetDeepConcat.Create([CurNet.GetLastLayer, SkipA]));
    ConvBlock(cC1);
    CurNet.AddLayer(TNNetConvolutionLinear.Create(1, 3, 1, 1, 0)); // head

    CurNet.SetLearningRate(cLR, cInertia);
    CurNet.SetBatchUpdate(true);
    Result := CurNet;
  end;

  // ---- data ----------------------------------------------------------------
  // Pull a clean digit into Img0 rescaled to [-1,1] (mirrors DiffusionMNIST).
  procedure LoadCleanDigit(Src: TNNetVolume);
  var
    x, y: integer;
    v, b: TNeuralFloat;
  begin
    if UseSynthetic then
    begin
      Img0.CopyChannels(Src, [0]);     // synthetic vols already in [-1,1]
      Exit;
    end;
    for y := 0 to cImgSize - 1 do
      for x := 0 to cImgSize - 1 do
      begin
        v := Src[x, y, 0];
        b := (v + 2.0) * 64.0;
        if b < 0 then b := 0;
        if b > 255 then b := 255;
        Img0[x, y, 0] := b / 127.5 - 1.0;
      end;
  end;

  // Build a synthetic dataset: each image is a single bright horizontal or
  // vertical bar on a dark background, in [-1,1]. Keeps the demo runnable
  // without MNIST files; the pipeline and metrics are unchanged.
  procedure BuildSyntheticData(Count: integer);
  var
    i, x, y, pos, thick: integer;
    Vol: TNNetVolume;
    horiz: boolean;
  begin
    TrainV := TNNetVolumeList.Create();
    ValV   := TNNetVolumeList.Create();
    TestV  := TNNetVolumeList.Create();
    for i := 0 to Count - 1 do
    begin
      Vol := TNNetVolume.Create(cImgSize, cImgSize, 1);
      Vol.Fill(-1.0);
      horiz := (i mod 2) = 0;
      thick := 3 + Random(3);
      pos   := 4 + Random(cImgSize - thick - 8);
      for y := 0 to cImgSize - 1 do
        for x := 0 to cImgSize - 1 do
          if horiz then
            begin if (y >= pos) and (y < pos + thick) then Vol[x, y, 0] := 1.0; end
          else
            begin if (x >= pos) and (x < pos + thick) then Vol[x, y, 0] := 1.0; end;
      TrainV.Add(Vol);
    end;
  end;

  // ---- consistency parameterisation ----------------------------------------
  // VP noise level on the scaled view: sigma(t) = sqrt((1-ab_t)/ab_t).
  function SigmaT(t: integer): TNeuralFloat;
  var ab: TNeuralFloat;
  begin
    ab := Sched.AlphaBar[t];
    if ab < 1.0e-8 then ab := 1.0e-8;
    Result := Sqrt((1.0 - ab) / ab);
  end;

  function CSkip(sigma: TNeuralFloat): TNeuralFloat;
  begin
    Result := (cSigmaData * cSigmaData) /
              (sigma * sigma + cSigmaData * cSigmaData);
  end;

  function COut(sigma: TNeuralFloat): TNeuralFloat;
  begin
    Result := sigma * cSigmaData /
              Sqrt(sigma * sigma + cSigmaData * cSigmaData);
  end;

  procedure SetTimeInput(Net: TNNet; t: integer);
  begin
    // The scalar timestep is fed as the SECOND Compute input (TimeVol); set it
    // there so Compute([Img, TimeVol]) injects it into the time-input layer.
    TimeVol.FData[0] := t;
  end;

  // Evaluate the consistency function f(Xs, t) = c_skip*Xs + c_out*F(Xs,t),
  // where Xs is the SCALED view (x0 + sigma*eps). Leaves f in Dst.
  // Net is either the live Student or the EMA target shadow net.
  procedure EvalConsistency(Net: TNNet; Xs: TNNetVolume; t: integer;
    Dst: TNNetVolume);
  var
    i: integer;
    cs, co: TNeuralFloat;
    Raw: TNNetVolume;
  begin
    cs := CSkip(SigmaT(t));
    co := COut(SigmaT(t));
    SetTimeInput(Net, t);
    Net.Compute([Xs, TimeVol]);
    Raw := Net.GetLastLayer.Output;
    for i := 0 to Dst.Size - 1 do
      Dst.FData[i] := cs * Xs.FData[i] + co * Raw.FData[i];
  end;

  // ---- teacher pretrain (eps-prediction DDPM, like DiffusionMNIST) ----------
  function EpsMSE: TNeuralFloat;
  var
    i: integer;
    d, s: TNeuralFloat;
    Pred: TNNetVolume;
  begin
    Pred := Teacher.GetLastLayer.Output;
    s := 0;
    for i := 0 to EpsTrue.Size - 1 do
    begin
      d := Pred.FData[i] - EpsTrue.FData[i];
      s := s + d * d;
    end;
    Result := s / EpsTrue.Size;
  end;

  procedure TrainTeacher;
  var
    Step, B, t, Idx: integer;
    SumLoss: TNeuralFloat;
    StartTime, Elapsed: double;
  begin
    WriteLn(Format('Teacher pretrain (eps-MSE DDPM): %d steps, batch %d, T=%d.',
      [TeachSteps, BatchSz, cT]));
    StartTime := Now();
    for Step := 1 to TeachSteps do
    begin
      SumLoss := 0;
      Teacher.ClearDeltas();
      for B := 1 to BatchSz do
      begin
        Idx := Random(TrainV.Count);
        LoadCleanDigit(TrainV[Idx]);
        t := 1 + Random(cT);
        Sched.AddNoise(Img0, ImgT, t, EpsTrue);
        SetTimeInput(Teacher, t);
        Teacher.Compute([ImgT, TimeVol]);
        SumLoss := SumLoss + EpsMSE;
        Teacher.Backpropagate(EpsTrue);
      end;
      Teacher.ForceMaxAbsoluteDelta(0.03);
      Teacher.UpdateWeights();
      if (Step = 1) or (Step mod 50 = 0) or (Step = TeachSteps) then
      begin
        Elapsed := (Now() - StartTime) * 86400.0;
        WriteLn(Format('  teacher step %4d / %4d   eps-MSE = %.5f   elapsed = %.1fs',
          [Step, TeachSteps, SumLoss / BatchSz, Elapsed]));
      end;
    end;
  end;

  // ---- consistency distillation --------------------------------------------
  // ONE deterministic teacher DDIM step from timestep tHi down to tLo on the
  // UNSCALED diffusion view (x_t = sqrt(ab)*x0 + sqrt(1-ab)*eps), in place on Xt.
  procedure TeacherDDIMStep(Xt: TNNetVolume; tHi, tLo: integer);
  var
    i: integer;
    sabHi, somabHi, sabLo, somabLo, x0p: TNeuralFloat;
    Eps: TNNetVolume;
  begin
    SetTimeInput(Teacher, tHi);
    Teacher.Compute([Xt, TimeVol]);
    Eps := Teacher.GetLastLayer.Output;
    sabHi   := Sqrt(Sched.AlphaBar[tHi]);
    somabHi := Sqrt(1.0 - Sched.AlphaBar[tHi]);
    sabLo   := Sqrt(Sched.AlphaBar[tLo]);
    somabLo := Sqrt(1.0 - Sched.AlphaBar[tLo]);
    for i := 0 to Xt.Size - 1 do
    begin
      x0p := (Xt.FData[i] - somabHi * Eps.FData[i]) / sabHi;     // predicted x0
      Xt.FData[i] := sabLo * x0p + somabLo * Eps.FData[i];       // DDIM to tLo
    end;
  end;

  // Distillation loss for ONE example, accumulates gradient on Student.
  // Returns the consistency MSE on f. Uses scaled view xs = x_t / sqrt(ab_t).
  function DistillExample: TNeuralFloat;
  var
    n, tLo, tHi, i: integer;
    XtHi, XtLo, XsHi, XsLo, FStud, FTarg, GradF, GradRaw: TNNetVolume;
    coHi, sabHi, sabLo, d, s: TNeuralFloat;
  begin
    XtHi  := TNNetVolume.Create(cImgSize, cImgSize, 1);
    XtLo  := TNNetVolume.Create(cImgSize, cImgSize, 1);
    XsHi  := TNNetVolume.Create(cImgSize, cImgSize, 1);
    XsLo  := TNNetVolume.Create(cImgSize, cImgSize, 1);
    FStud := TNNetVolume.Create(cImgSize, cImgSize, 1);
    FTarg := TNNetVolume.Create(cImgSize, cImgSize, 1);
    GradF   := TNNetVolume.Create(cImgSize, cImgSize, 1);
    GradRaw := TNNetVolume.Create(cImgSize, cImgSize, 1);
    try
      // Adjacent timesteps on the sub-grid: pick n in [1..cNumSub-1].
      n   := 1 + Random(cNumSub - 1);
      tHi := Round(n       * cT / cNumSub);
      tLo := Round((n - 1) * cT / cNumSub);
      if tHi < 1 then tHi := 1;
      if tHi > cT then tHi := cT;
      if tLo < 1 then tLo := 1;
      if tLo >= tHi then tLo := tHi - 1;
      if tLo < 1 then tLo := 1;

      // Forward-noise x0 to x_{tHi}; one teacher ODE step to x_{tLo}.
      Sched.AddNoise(Img0, XtHi, tHi, EpsTrue);
      XtLo.Copy(XtHi);
      TeacherDDIMStep(XtLo, tHi, tLo);

      // Scaled views xs = x_t / sqrt(ab_t)  (so xs = x0 + sigma*eps).
      sabHi := Sqrt(Sched.AlphaBar[tHi]);
      sabLo := Sqrt(Sched.AlphaBar[tLo]);
      for i := 0 to XtHi.Size - 1 do
      begin
        XsHi.FData[i] := XtHi.FData[i] / sabHi;
        XsLo.FData[i] := XtLo.FData[i] / sabLo;
      end;

      // Target net f_target(x_{tLo}, tLo) -- stop gradient.
      EvalConsistency(TargetEMA.ShadowNet, XsLo, tLo, FTarg);
      // Student f_theta(x_{tHi}, tHi) -- a full forward pass we will backprop.
      EvalConsistency(Student, XsHi, tHi, FStud);

      // Loss = mean || FStud - FTarg ||^2 over pixels (per-pixel MSE).
      // dL/dF = (FStud - FTarg) * (2/N). The c_out(tHi) scaling carries the
      // gradient from f down to the raw network output F_theta:
      //   dL/dRaw = dL/dF * c_out(tHi).
      coHi := COut(SigmaT(tHi));
      s := 0;
      for i := 0 to FStud.Size - 1 do
      begin
        d := FStud.FData[i] - FTarg.FData[i];
        s := s + d * d;
        GradF.FData[i]   := d;                 // proportional to dL/dF
        GradRaw.FData[i] := d * coHi;          // gradient on the raw head
      end;

      // Backpropagate uses (output - target); Student's raw output is in the
      // last layer. We want gradient GradRaw on it, so pass target =
      // rawOutput - GradRaw.
      with Student.GetLastLayer.Output do
        for i := 0 to Size - 1 do
          GradF.FData[i] := FData[i] - GradRaw.FData[i];
      Student.Backpropagate(GradF);

      Result := s / FStud.Size;
    finally
      XtHi.Free; XtLo.Free; XsHi.Free; XsLo.Free;
      FStud.Free; FTarg.Free; GradF.Free; GradRaw.Free;
    end;
  end;

  procedure Distill;
  var
    Step, B, Idx: integer;
    SumLoss: TNeuralFloat;
    StartTime, Elapsed: double;
  begin
    WriteLn(Format('Consistency distillation: %d steps, batch %d, sub-grid N=%d, EMA=%.2f.',
      [DistSteps, BatchSz, cNumSub, cEMADecay]));
    StartTime := Now();
    for Step := 1 to DistSteps do
    begin
      SumLoss := 0;
      Student.ClearDeltas();
      for B := 1 to BatchSz do
      begin
        Idx := Random(TrainV.Count);
        LoadCleanDigit(TrainV[Idx]);
        SumLoss := SumLoss + DistillExample;
      end;
      Student.ForceMaxAbsoluteDelta(0.03);
      Student.UpdateWeights();
      TargetEMA.Update();    // f_target := EMA(f_theta)
      if (Step = 1) or (Step mod 50 = 0) or (Step = DistSteps) then
      begin
        Elapsed := (Now() - StartTime) * 86400.0;
        WriteLn(Format('  distill step %4d / %4d   consistency-MSE = %.5f   elapsed = %.1fs',
          [Step, DistSteps, SumLoss / BatchSz, Elapsed]));
      end;
    end;
  end;

  // ---- samplers ------------------------------------------------------------
  // Few-step consistency sampler: K in {1,2,4}. Leaves x0 (in [-1,1] image
  // space) in Dst. Uses the live Student (EMA could also be used).
  procedure SampleConsistency(Dst: TNNetVolume; K: integer);
  var
    Xs, F: TNNetVolume;
    i, k_i, t: integer;
    schedule: array[0..3] of integer;
    nSteps: integer;
  begin
    Xs := TNNetVolume.Create(cImgSize, cImgSize, 1);
    F  := TNNetVolume.Create(cImgSize, cImgSize, 1);
    try
      // Time schedule from high to low. For K steps use evenly spaced points.
      nSteps := K;
      if nSteps > 4 then nSteps := 4;
      case nSteps of
        1: begin schedule[0] := cT; end;
        2: begin schedule[0] := cT; schedule[1] := cT div 2; end;
        4: begin schedule[0] := cT; schedule[1] := (3*cT) div 4;
                 schedule[2] := cT div 2; schedule[3] := cT div 4; end;
      else
        begin schedule[0] := cT; schedule[1] := cT div 2; nSteps := 2; end;
      end;

      // Start from pure noise on the scaled view: xs ~ N(0, sigma(T)^2 + data).
      // The scaled view at t=T is x0 + sigma(T)*eps; with x0 unknown we start
      // from sigma(T)*eps which is the standard consistency-model init.
      for i := 0 to Xs.Size - 1 do
        Xs.FData[i] := SigmaT(cT) * RandG(0, 1);

      for k_i := 0 to nSteps - 1 do
      begin
        t := schedule[k_i];
        EvalConsistency(Student, Xs, t, F);     // F ~= x0 estimate
        if k_i < nSteps - 1 then
        begin
          // Re-noise x0 to the next (lower) timestep on the scaled view:
          // xs := x0 + sigma(t_next)*eps.
          for i := 0 to Xs.Size - 1 do
            Xs.FData[i] := F.FData[i] + SigmaT(schedule[k_i + 1]) * RandG(0, 1);
        end
        else
          Dst.Copy(F);                          // final x0 estimate
      end;
    finally
      Xs.Free; F.Free;
    end;
  end;

  procedure TTeacherDenoiser.Denoise(Xt, Output: TNNetVolume; t: integer);
  begin
    SetTimeInput(Net, t);
    Net.Compute([Xt, TimeVol]);
    Output.Copy(Net.GetLastLayer.Output);
  end;

  // Teacher reference: full multi-step DDIM from noise. Leaves x0 in Dst.
  procedure SampleTeacher(Dst: TNNetVolume);
  var i: integer;
  begin
    for i := 0 to Dst.Size - 1 do
      Dst.FData[i] := RandG(0, 1);
    Sched.Sample(Dst, @TeacherDenoiser.Denoise, cTeacherSteps, smDDIM, 0.0);
  end;

  // ---- metric: mean MSE of a generated image to its nearest train digit -----
  function NearestDataMSE(Gen: TNNetVolume; NumProbe: integer): TNeuralFloat;
  var
    p, i, Idx: integer;
    best, d, s: TNeuralFloat;
  begin
    best := 1.0e30;
    for p := 1 to NumProbe do
    begin
      Idx := Random(TrainV.Count);
      LoadCleanDigit(TrainV[Idx]);   // into Img0 ([-1,1])
      s := 0;
      for i := 0 to Gen.Size - 1 do
      begin
        d := Gen.FData[i] - Img0.FData[i];
        s := s + d * d;
      end;
      s := s / Gen.Size;
      if s < best then best := s;
    end;
    Result := best;
  end;

  // ---- output grid ----------------------------------------------------------
  // Rows: 0 = teacher multistep, 1 = 1-step, 2 = 2-step, 3 = 4-step. Each row
  // has GridN independent samples. Also computes per-sampler mean nearest-data
  // MSE (the fidelity proxy) and a NaN count.
  procedure RunAndReport(const FileName: string);
  var
    Grid, One: TNNetVolume;
    rowDef: array[0..3] of integer;   // K per row; -1 = teacher
    r, c, x, y, gx, gy, i: integer;
    v, px, msum: TNeuralFloat;
    nanCount: integer;
    rowName: string;
  begin
    rowDef[0] := -1; rowDef[1] := 1; rowDef[2] := 2; rowDef[3] := 4;
    Grid := TNNetVolume.Create(GridN * cImgSize, 4 * cImgSize, 3);
    One  := TNNetVolume.Create(cImgSize, cImgSize, 1);
    Grid.Fill(0);
    try
      WriteLn;
      WriteLn('Sampling & metric (nearest-train-digit MSE; lower = closer to data):');
      for r := 0 to 3 do
      begin
        msum := 0; nanCount := 0;
        for c := 0 to GridN - 1 do
        begin
          if rowDef[r] < 0
            then SampleTeacher(One)
            else SampleConsistency(One, rowDef[r]);
          for i := 0 to One.Size - 1 do
            if IsNan(One.FData[i]) or IsInfinite(One.FData[i]) then
              begin Inc(nanCount); One.FData[i] := 0; end;
          msum := msum + NearestDataMSE(One, 32);
          for y := 0 to cImgSize - 1 do
            for x := 0 to cImgSize - 1 do
            begin
              v := One[x, y, 0];
              px := (v + 1.0) * 127.5;
              if px < 0 then px := 0;
              if px > 255 then px := 255;
              gx := c * cImgSize + x;
              gy := r * cImgSize + y;
              Grid[gx, gy, 0] := px;
              Grid[gx, gy, 1] := px;
              Grid[gx, gy, 2] := px;
            end;
        end;
        if rowDef[r] < 0
          then rowName := Format('teacher %d-step DDIM', [cTeacherSteps])
          else rowName := Format('consistency %d-step', [rowDef[r]]);
        WriteLn(Format('  %-22s  mean nearest-data MSE = %.4f   (NaN/Inf pixels: %d)',
          [rowName, msum / GridN, nanCount]));
      end;
      if SaveImageFromVolumeIntoFile(Grid, FileName)
        then WriteLn('Wrote sample grid: ', FileName)
        else WriteLn('FAILED to write: ', FileName);
    finally
      Grid.Free; One.Free;
    end;
  end;

var
  i: integer;
begin
  SetExceptionMask([exInvalidOp, exDenormalized, exZeroDivide,
                    exOverflow, exUnderflow, exPrecision]);
  RandSeed := 2026;

  FullMode := false;
  for i := 1 to ParamCount do
    if (ParamStr(i) = '--full') then FullMode := true;

  if FullMode then
  begin
    TeachSteps := cFullTeach; DistSteps := cFullDist;
    BatchSz := cFullBatch; GridN := cFullGrid;
    WriteLn('ConsistencyDistill [FULL mode]');
  end
  else
  begin
    TeachSteps := cSmokeTeach; DistSteps := cSmokeDist;
    BatchSz := cSmokeBatch; GridN := cSmokeGrid;
    WriteLn('ConsistencyDistill [SMOKE mode -- pass --full for sharper output]');
  end;
  WriteLn('Consistency distillation (Song et al. 2023): distil a multi-step DDPM');
  WriteLn('teacher into a 1/2/4-step consistency model. c_skip/c_out boundary');
  WriteLn('parameterisation + EMA target net (TNNetEMAWrapper).');
  WriteLn;

  Sched := TNNetDiffusionScheduler.Create(cT, dsLinear, dpEps, cBeta1, cBetaT);

  UseSynthetic := not (CheckMNISTFile('train') and CheckMNISTFile('t10k'));
  if UseSynthetic then
  begin
    WriteLn('MNIST idx-ubyte files not found -> using SYNTHETIC bar dataset (CI fallback).');
    BuildSyntheticData(2000);
  end
  else
  begin
    CreateMNISTVolumes(TrainV, ValV, TestV, 'train', 't10k');
    WriteLn('Loaded MNIST: ', TrainV.Count, ' train digits.');
  end;

  Teacher := BuildUNet;
  Student := BuildUNet;
  TargetEMA := TNNetEMAWrapper.Create(Student, cEMADecay);

  WriteLn('Teacher / Student architecture (identical tiny time-conditioned U-Net):');
  Teacher.PrintSummary();
  WriteLn;

  Img0    := TNNetVolume.Create(cImgSize, cImgSize, 1);
  ImgT    := TNNetVolume.Create(cImgSize, cImgSize, 1);
  EpsTrue := TNNetVolume.Create(cImgSize, cImgSize, 1);
  TimeVol := TNNetVolume.Create(1, 1, 1);
  SchedTime := TNNetVolume.Create(1, 1, 1);

  TeacherDenoiser := TTeacherDenoiser.Create;
  TeacherDenoiser.Net := Teacher;
  TeacherDenoiser.TimeVol := TimeVol;

  try
    TrainTeacher;
    WriteLn;
    Distill;
    WriteLn;
    RunAndReport('consistency_samples.png');
    WriteLn;
    WriteLn('Done. Rows: teacher / 1-step / 2-step / 4-step consistency samples.');
    if not FullMode then
      WriteLn('(Smoke samples are rough; run with --full for sharper output.)');
  finally
    Img0.Free; ImgT.Free; EpsTrue.Free; TimeVol.Free; SchedTime.Free;
    TeacherDenoiser.Free;
    TargetEMA.Free;
    Teacher.Free;
    Student.Free;
    Sched.Free;
    TrainV.Free;
    if Assigned(ValV) then ValV.Free;
    if Assigned(TestV) then TestV.Free;
  end;
end.
