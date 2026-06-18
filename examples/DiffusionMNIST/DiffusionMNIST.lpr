program DiffusionMNIST;
(*
DiffusionMNIST: a GENERATIVE-BY-DIFFUSION example -- a
Denoising Diffusion Probabilistic Model (DDPM, Ho et al. 2020,
https://arxiv.org/abs/2006.11239) that learns to synthesise 28x28 MNIST
digits from pure Gaussian noise. It now also supports CLASS-CONDITIONAL
generation (pick the digit you want) via classifier-free guidance (CFG) and a
DETERMINISTIC DDIM fast sampler (10-50 steps instead of the full ancestral loop).

WHAT A DDPM IS (in one paragraph). A fixed FORWARD process gradually corrupts
a clean image x_0 into pure noise x_T over T steps by repeatedly adding a
little Gaussian noise on a LINEAR beta schedule (beta_1=1e-4 .. beta_T=0.02).
The closed form jumps directly to any timestep t:

    x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * eps,  eps~N(0,I)

where alpha_t = 1 - beta_t and alpha_bar_t = prod_{s<=t} alpha_s. A neural
network eps_theta(x_t, t, y) is trained to PREDICT the noise eps that was added
(the simple epsilon-prediction MSE objective). Once it can denoise, we run the
REVERSE (ancestral / DDPM) sampling loop from x_T~N(0,I) back down to x_0,
subtracting the predicted noise step by step, to draw fresh digits.

CLASS CONDITIONING + CLASSIFIER-FREE GUIDANCE (CFG). The network additionally
takes the digit label y (0..9) so we can ask for a SPECIFIC digit. We embed y
with a TNNetEmbedding and ADD that vector to the sinusoidal time embedding
BEFORE the shared cond MLP, so a single FiLM cond vector carries both "how much
noise" and "which digit". To get CFG we reserve label index 10 as a special
NULL / unconditional token and, during training, with probability ~0.1 replace
the real label with 10 (LABEL DROPOUT). The one network thus learns both the
conditional branch eps(x_t,t,y) and the unconditional branch eps(x_t,t,null).
At sampling time we run it TWICE per step and extrapolate:

    eps = eps_uncond + s * (eps_cond - eps_uncond)        (guidance scale s)

DDIM FAST SAMPLER. Instead of the T-step stochastic ancestral loop we use the
DETERMINISTIC DDIM update (eta=0) over a short STRIDED subsequence of timesteps:

    x0_pred    = (x_t - sqrt(1-abar_t) * eps) / sqrt(abar_t)
    x_{t_prev} = sqrt(abar_{t_prev}) * x0_pred + sqrt(1-abar_{t_prev}) * eps

which produces good digits in 10-50 steps. DDIM is combined with CFG above.

TIME CONDITIONING. The integer timestep t must be fed to the network so it
knows how much noise to expect. This example USES TNNetSinusoidalTimeEmbedding
(coded specifically for diffusion) to map the scalar t to a sinusoidal vector,
then injects it into every U-Net block as a per-channel scale/shift via
TNNet.AddFiLMConditioned (Feature-wise Linear Modulation / TNNetFiLM). This is
the standard DDPM time-embedding -> FiLM recipe.

THE NETWORK is a small three-input U-Net (image + timestep + label branch):

  image  (28,28,1) -- TNNetInput
  t      (1,1,1)   -- TNNetInput  -> TNNetSinusoidalTimeEmbedding(cEmbDim)  --\
  y      (1,1,1)   -- TNNetInput  -> TNNetEmbedding(11 -> cEmbDim)  ----------> Sum
                                   -> FullConnect MLP (shared cond vector)

  enc1: Conv(C1,28x28) -> GroupNorm -> FiLM(t) -> Conv(C1)   ........ skip A
        MaxPool /2  -> 14x14
  enc2: Conv(C2,14x14) -> GroupNorm -> FiLM(t) -> Conv(C2)   ........ skip B
        MaxPool /2  -> 7x7
  mid : Conv(C3,7x7)  -> GroupNorm -> FiLM(t) -> Conv(C3)
  dec2: Upsample x2 -> 14x14 ; DeepConcat(skip B) ; Conv(C2) -> GroupNorm
        -> FiLM(t) -> Conv(C2)
  dec1: Upsample x2 -> 28x28 ; DeepConcat(skip A) ; Conv(C1) -> GroupNorm
        -> FiLM(t) -> Conv(C1)
  head: ConvLinear(1, 3x3)   -> predicted noise eps_hat (28,28,1)

Skip connections reuse the existing TNNetDeepConcat (depth-axis concat); the
decoder upsamples with the existing parameter-free TNNetUpsample. All layers
already existed; the new content here is the noise schedule, the
epsilon-prediction training loop and the ancestral sampling loop.

RUN MODES.
  default (SMOKE): a small number of training steps then a tiny sample grid;
    finishes in a few minutes on one CPU. Good enough to see the loss fall and
    to confirm a PNG grid is written without NaN.
  full training:  pass  --full  for many more steps / a bigger grid.

OUTPUT. A PNG grid of generated digits is written to the working directory
(diffusion_samples.png). Pixels are in [-1,1] internally and mapped to 0..255
for the grayscale PNG.

DATA. Standard MNIST idx-ubyte files in the working directory (the same files
every MNIST example here uses: train-images.idx3-ubyte etc.). If they are not
present the program prints the download hint and exits cleanly.

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

  // Class conditioning. Labels 0..9 are the real digits; index 10 is the
  // special NULL / unconditional token used for label dropout and CFG.
  cNumClasses = 11;         // 10 digits + 1 null token
  cNullLabel  = 10;
  cLabelDrop  = 0.1;        // prob of replacing the label with null during train

  // Classifier-free guidance scale and DDIM step count used at sampling time.
  cGuidance   = 3.0;        // s in eps_uncond + s*(eps_cond - eps_uncond)
  cDDIMSteps  = 25;         // strided reverse steps (<< cT)
  // Channel widths per U-Net level (kept small so the smoke run is fast).
  cC1        = 16;
  cC2        = 32;
  cC3        = 48;

  // Linear beta schedule (Ho et al. 2020). T modest so CPU sampling is quick.
  cT         = 200;
  cBeta1     = 1.0e-4;
  cBetaT     = 0.02;

  cLR        = 0.0005;
  cInertia   = 0.9;

  // Default = fast smoke. --full overrides via cmd line below.
  // Grid width is fixed to 10 so each row r requests digit (r mod 10): a
  // 10x10 grid then shows every class 0..9 across the rows.
  cSmokeSteps = 500;
  cSmokeBatch = 16;
  cSmokeGrid  = 10;         // 10x10 = each row a requested digit 0..9

  cFullSteps  = 6000;
  cFullBatch  = 32;
  cFullGrid   = 10;         // 10x10 class-conditional grid

type
  // Wraps the network's classifier-free-guided eps prediction as a
  // TNNetDenoiseCallback (procedure-of-object) so the reusable
  // TNNetDiffusionScheduler can drive the reverse process. ReqDigit is the
  // class requested for the current sampling trajectory.
  // Coded by Claude (AI).
  TCFGDenoiser = class(TObject)
  public
    ReqDigit: integer;
    procedure Denoise(Xt, Output: TNNetVolume; t: integer);
  end;

var
  // Reusable scheduler holding the precomputed beta/alpha/alpha_bar tables.
  Sched: TNNetDiffusionScheduler;
  Denoiser: TCFGDenoiser;

  NN: TNNet;
  ImgIn, TimeIn, LabelIn, TimeEmb: TNNetLayer;
  TrainV, ValV, TestV: TNNetVolumeList;

  // Reusable working volumes for the training/sampling loops.
  Img0, ImgT, EpsTrue, TimeVol, LabelVol: TNNetVolume;

  Steps, BatchSz, GridN: integer;
  FullMode: boolean;

  // One conv -> GroupNorm -> FiLM(time) -> conv block at the current spatial
  // resolution. Returns the block output layer (the post-FiLM refinement).
  function ConvBlock(Channels: integer): TNNetLayer;
  var
    Feat: TNNetLayer;
  begin
    NN.AddLayer(TNNetConvolutionReLU.Create(Channels, 3, 1, 1, 0));
    // 4 groups when divisible, else fall back to instance-style (1 ch/group).
    if (Channels mod 4) = 0
      then NN.AddLayer(TNNetGroupNorm.Create(4))
      else NN.AddLayer(TNNetGroupNorm.Create(Channels));
    Feat := NN.GetLastLayer;
    // Inject the timestep: per-channel gamma/beta from the shared time embedding.
    NN.AddFiLMConditioned(Feat, TimeEmb);
    Result := NN.AddLayer(TNNetConvolutionReLU.Create(Channels, 3, 1, 1, 0));
  end;

  procedure BuildNet;
  var
    SkipA, SkipB, LabelEmb: TNNetLayer;
  begin
    NN := TNNet.Create();
    // Three inputs: the (noisy) image, the scalar timestep and the class label.
    ImgIn   := NN.AddLayer(TNNetInput.Create(cImgSize, cImgSize, 1, 1));
    TimeIn  := NN.AddLayerAfter(TNNetInput.Create(1, 1, 1, 1), 0);
    LabelIn := NN.AddLayerAfter(TNNetInput.Create(1, 1, 1, 1), 0);

    // Time branch: scalar t -> sinusoidal vector (shape 1 x 1 x cEmbDim).
    TimeEmb := NN.AddLayerAfter(TNNetSinusoidalTimeEmbedding.Create(cEmbDim), TimeIn);
    // Label branch: integer y -> embedding row (shape 1 x 1 x cEmbDim). Index 10
    // is the null/unconditional token; EncodeZero=1 so label 0 is also embedded.
    LabelEmb := NN.AddLayerAfter(TNNetEmbedding.Create(cNumClasses, cEmbDim, 1), LabelIn);
    // ADD the label embedding onto the time embedding, then a shared cond MLP.
    // A single FiLM cond vector therefore carries both t and y information.
    TimeEmb := NN.AddLayer(TNNetSum.Create([TimeEmb, LabelEmb]));
    TimeEmb := NN.AddLayerAfter(TNNetFullConnectReLU.Create(cEmbDim), TimeEmb);
    TimeEmb := NN.AddLayerAfter(TNNetFullConnectReLU.Create(cEmbDim), TimeEmb);

    // Encoder. Always continue the image path from ImgIn explicitly because the
    // last added layer above belongs to the time branch.
    NN.AddLayerAfter(TNNetConvolutionReLU.Create(cC1, 3, 1, 1, 0), ImgIn);
    SkipA := ConvBlock(cC1);                       // 28x28, skip A
    NN.AddLayer(TNNetMaxPool.Create(2));           // -> 14x14
    SkipB := ConvBlock(cC2);                       // 14x14, skip B
    NN.AddLayer(TNNetMaxPool.Create(2));           // -> 7x7

    // Bottleneck.
    ConvBlock(cC3);                                // 7x7

    // Decoder level 2: upsample to 14x14, concat skip B, refine.
    NN.AddLayer(TNNetUpsample.Create());           // 7 -> 14
    NN.AddLayer(TNNetDeepConcat.Create([NN.GetLastLayer, SkipB]));
    ConvBlock(cC2);

    // Decoder level 1: upsample to 28x28, concat skip A, refine.
    NN.AddLayer(TNNetUpsample.Create());           // 14 -> 28
    NN.AddLayer(TNNetDeepConcat.Create([NN.GetLastLayer, SkipA]));
    ConvBlock(cC1);

    // Noise-prediction head: linear conv to a single channel (eps_hat).
    NN.AddLayer(TNNetConvolutionLinear.Create(1, 3, 1, 1, 0));

    NN.SetLearningRate(cLR, cInertia);
    NN.SetBatchUpdate(true);
  end;

  // Pull a clean MNIST digit into Img0 rescaled to [-1, 1] and return its digit
  // label (0..9). The label lives in the volume's Tag (set by CreateMNISTVolumes).
  // The loader stores pixels as byte/64 - 2, so byte = (v+2)*64 and the
  // [-1,1] target is byte/127.5 - 1.
  function LoadCleanDigit(Src: TNNetVolume): integer;
  var
    x, y: integer;
    v, b: TNeuralFloat;
  begin
    for y := 0 to cImgSize - 1 do
      for x := 0 to cImgSize - 1 do
      begin
        v := Src[x, y, 0];
        b := (v + 2.0) * 64.0;        // approx original 0..255 byte
        if b < 0 then b := 0;
        if b > 255 then b := 255;
        Img0[x, y, 0] := b / 127.5 - 1.0;
      end;
    Result := Src.Tag;
  end;

  // Forward-noise Img0 to timestep t: ImgT = sqrt(ab)*x0 + sqrt(1-ab)*eps,
  // filling EpsTrue with the sampled noise (the training target). The forward
  // process now comes from the reusable scheduler (q_sample).
  procedure NoiseToTimestep(t: integer);
  begin
    Sched.AddNoise(Img0, ImgT, t, {NoiseOut=}EpsTrue);
  end;

  // Mean-squared error between the network's eps_hat and EpsTrue.
  function EpsMSE: TNeuralFloat;
  var
    i: integer;
    d, s: TNeuralFloat;
    Pred: TNNetVolume;
  begin
    Pred := NN.GetLastLayer.Output;
    s := 0;
    for i := 0 to EpsTrue.Size - 1 do
    begin
      d := Pred.FData[i] - EpsTrue.FData[i];
      s := s + d * d;
    end;
    Result := s / EpsTrue.Size;
  end;

  procedure SetTimestepInput(t: integer);
  begin
    TimeVol.FData[0] := t;
    TimeIn.Output.FData[0] := t;
  end;

  // Set the class-label input (0..9, or cNullLabel=10 for unconditional).
  procedure SetLabelInput(y: integer);
  begin
    LabelVol.FData[0] := y;
    LabelIn.Output.FData[0] := y;
  end;

  procedure TrainLoop;
  var
    Step, B, t, Idx, y: integer;
    SumLoss: TNeuralFloat;
    StartTime, Elapsed: double;
  begin
    WriteLn(Format('Training: %d steps, batch %d, schedule T=%d (beta %.0e..%.3g).',
      [Steps, BatchSz, cT, cBeta1, cBetaT]));
    StartTime := Now();
    for Step := 1 to Steps do
    begin
      SumLoss := 0;
      NN.ClearDeltas();
      for B := 1 to BatchSz do
      begin
        Idx := Random(TrainV.Count);
        y := LoadCleanDigit(TrainV[Idx]);    // real digit label 0..9
        // Label dropout: with prob cLabelDrop train the UNCONDITIONAL branch by
        // replacing the real label with the null token. This is what lets the
        // single network produce both eps(.,.,y) and eps(.,.,null) for CFG.
        if Random < cLabelDrop then y := cNullLabel;
        t := 1 + Random(cT);                 // uniform timestep in 1..T
        NoiseToTimestep(t);
        SetTimestepInput(t);
        SetLabelInput(y);
        NN.Compute([ImgT, TimeVol, LabelVol]); // three-input forward
        SumLoss := SumLoss + EpsMSE;
        // Train eps-prediction: target is the sampled noise EpsTrue.
        NN.Backpropagate(EpsTrue);
      end;
      // Tame the early gradient spikes a from-scratch diffusion net produces
      // before the noise-schedule statistics settle. The extra label-embedding
      // path makes the cold start a touch more volatile, so the clamp is a bit
      // tighter than the unconditional example.
      NN.ForceMaxAbsoluteDelta(0.03);
      NN.UpdateWeights();
      if (Step = 1) or (Step mod 25 = 0) or (Step = Steps) then
      begin
        Elapsed := (Now() - StartTime) * 86400.0;
        WriteLn(Format('  step %4d / %4d   eps-MSE = %.5f   elapsed = %.1fs',
          [Step, Steps, SumLoss / BatchSz, Elapsed]));
      end;
    end;
  end;

  // Classifier-free-guided noise prediction for image Xt at timestep t and
  // class y. Runs the net TWICE -- once with the wanted label y, once with the
  // null label -- and stores the guided eps into EpsCFG:
  //   eps = eps_uncond + s * (eps_cond - eps_uncond).
  // When s = 0 (or y is already the null token) this reduces to the plain
  // unconditional prediction; s > 1 sharpens the requested class.
  procedure PredictEpsCFG(Xt, EpsCFG: TNNetVolume; t, y: integer; s: TNeuralFloat);
  var
    i: integer;
    Cond: TNNetVolume;
  begin
    SetTimestepInput(t);
    // Conditional pass eps(x_t, t, y).
    SetLabelInput(y);
    NN.Compute([Xt, TimeVol, LabelVol]);
    Cond := NN.GetLastLayer.Output;
    EpsCFG.Copy(Cond);                          // EpsCFG := eps_cond
    if (s <> 0) and (y <> cNullLabel) then
    begin
      // Unconditional pass eps(x_t, t, null) and extrapolate.
      SetLabelInput(cNullLabel);
      NN.Compute([Xt, TimeVol, LabelVol]);
      Cond := NN.GetLastLayer.Output;           // now eps_uncond
      for i := 0 to EpsCFG.Size - 1 do
        EpsCFG.FData[i] := Cond.FData[i] +
          s * (EpsCFG.FData[i] - Cond.FData[i]);
    end;
  end;

  // TNNetDenoiseCallback adapter: produce the CFG-guided eps for the digit set
  // in Denoiser.ReqDigit. This is the single model hook the reusable scheduler
  // calls at each reverse step.
  procedure TCFGDenoiser.Denoise(Xt, Output: TNNetVolume; t: integer);
  begin
    PredictEpsCFG(Xt, Output, t, ReqDigit, cGuidance);
  end;

  // Ancestral DDPM reverse sampling for ONE image of class y, leaving x_0 in
  // Dst, via the reusable scheduler (full T-step ancestral loop + CFG).
  procedure SampleOne(Dst: TNNetVolume; y: integer; Eps: TNNetVolume);
  var i: integer;
  begin
    for i := 0 to Dst.Size - 1 do
      Dst.FData[i] := RandG(0, 1);              // start from pure noise x_T
    Denoiser.ReqDigit := y;
    Sched.Sample(Dst, @Denoiser.Denoise, cT, smDDPM, 0.0);
  end;

  // DETERMINISTIC DDIM (eta=0) sampling for ONE image of class y, leaving x_0 in
  // Dst, over a strided subsequence of NumSteps timesteps (NumSteps << cT), via
  // the reusable scheduler. Uses CFG at each step.
  procedure SampleOneDDIM(Dst: TNNetVolume; y, NumSteps: integer; Eps: TNNetVolume);
  var i: integer;
  begin
    for i := 0 to Dst.Size - 1 do
      Dst.FData[i] := RandG(0, 1);
    Denoiser.ReqDigit := y;
    Sched.Sample(Dst, @Denoiser.Denoise, NumSteps, smDDIM, 0.0);
  end;

  // Sample a CLASS-CONDITIONAL grid: each ROW r requests the same digit
  // (r mod 10), each COLUMN is an independent sample, so a 10-wide grid shows
  // every digit 0..9. UseDDIM picks the fast deterministic sampler (cDDIMSteps)
  // over the full ancestral loop. Saved as one grayscale PNG grid.
  procedure SampleGridToPng(const FileName: string; UseDDIM: boolean);
  var
    Grid, One, Eps: TNNetVolume;
    r, c, x, y, gx, gy, digit: integer;
    v, px: TNeuralFloat;
    NumNan: integer;
  begin
    Grid := TNNetVolume.Create(GridN * cImgSize, GridN * cImgSize, 3);
    One  := TNNetVolume.Create(cImgSize, cImgSize, 1);
    Eps  := TNNetVolume.Create(cImgSize, cImgSize, 1);
    Grid.Fill(0);
    NumNan := 0;
    try
      if UseDDIM
        then WriteLn(Format('Sampling a %dx%d class-conditional grid via ' +
               '%d-step DDIM + CFG (s=%.1f)...', [GridN, GridN, cDDIMSteps, cGuidance]))
        else WriteLn(Format('Sampling a %dx%d class-conditional grid via ' +
               '%d-step ancestral DDPM + CFG (s=%.1f)...', [GridN, GridN, cT, cGuidance]));
      for r := 0 to GridN - 1 do
      begin
        digit := r mod 10;                       // requested class for this row
        for c := 0 to GridN - 1 do
        begin
          if UseDDIM
            then SampleOneDDIM(One, digit, cDDIMSteps, Eps)
            else SampleOne(One, digit, Eps);
          for y := 0 to cImgSize - 1 do
            for x := 0 to cImgSize - 1 do
            begin
              v := One[x, y, 0];
              if IsNan(v) or IsInfinite(v) then begin Inc(NumNan); v := 0; end;
              // [-1,1] -> 0..255 grayscale.
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
      end;
      if SaveImageFromVolumeIntoFile(Grid, FileName)
        then WriteLn('Wrote sample grid: ', FileName)
        else WriteLn('FAILED to write: ', FileName);
      if NumNan > 0
        then WriteLn('WARNING: ', NumNan, ' NaN/Inf pixels were clamped.')
        else WriteLn('No NaN/Inf in generated samples.');
    finally
      Grid.Free;
      One.Free;
      Eps.Free;
    end;
  end;

var
  i: integer;
  PngName: string;
begin
  SetExceptionMask([exInvalidOp, exDenormalized, exZeroDivide,
                    exOverflow, exUnderflow, exPrecision]);
  RandSeed := 2026;

  FullMode := false;
  for i := 1 to ParamCount do
    if (ParamStr(i) = '--full') then FullMode := true;

  if FullMode then
  begin
    Steps := cFullSteps; BatchSz := cFullBatch; GridN := cFullGrid;
    WriteLn('DiffusionMNIST [FULL training mode]');
  end
  else
  begin
    Steps := cSmokeSteps; BatchSz := cSmokeBatch; GridN := cSmokeGrid;
    WriteLn('DiffusionMNIST [SMOKE mode -- pass --full for real training]');
  end;
  WriteLn('DDPM epsilon-prediction (Ho et al. 2020), tiny time/label-conditioned U-Net.');
  WriteLn('Class-conditional via label embedding + label dropout; sampling uses CFG + DDIM.');
  WriteLn;

  if not (CheckMNISTFile('train')) or not (CheckMNISTFile('t10k')) then
  begin
    WriteLn('MNIST idx-ubyte files not found in the working directory; exiting.');
    Exit;
  end;

  // Reusable linear-beta scheduler (neuraldiffusion.pas) replaces the former
  // hand-rolled inline schedule tables and sampling loops.
  Sched := TNNetDiffusionScheduler.Create(cT, dsLinear, dpEps, cBeta1, cBetaT);
  Denoiser := TCFGDenoiser.Create;

  CreateMNISTVolumes(TrainV, ValV, TestV, 'train', 't10k');
  WriteLn('Loaded MNIST: ', TrainV.Count, ' train digits.');

  BuildNet;
  WriteLn('Network architecture:');
  NN.PrintSummary();
  WriteLn;

  Img0    := TNNetVolume.Create(cImgSize, cImgSize, 1);
  ImgT    := TNNetVolume.Create(cImgSize, cImgSize, 1);
  EpsTrue := TNNetVolume.Create(cImgSize, cImgSize, 1);
  TimeVol  := TNNetVolume.Create(1, 1, 1);
  LabelVol := TNNetVolume.Create(1, 1, 1);
  try
    TrainLoop;
    WriteLn;
    // Default deliverable: a class-conditional grid via fast DDIM + CFG, each
    // row a requested digit 0..9.
    PngName := 'diffusion_samples.png';
    SampleGridToPng(PngName, {UseDDIM=}true);
    WriteLn;
    WriteLn('Done. View ', PngName, ' (row r = requested digit r mod 10).');
    if not FullMode then
      WriteLn('(Smoke samples are noisy; run with --full for sharp digits.)');
  finally
    Img0.Free;
    ImgT.Free;
    EpsTrue.Free;
    TimeVol.Free;
    LabelVol.Free;
    NN.Free;
    Sched.Free;
    Denoiser.Free;
    TrainV.Free;
    ValV.Free;
    TestV.Free;
  end;
end.
