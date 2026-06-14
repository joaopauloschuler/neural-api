program DiffusionMNIST;
(*
DiffusionMNIST: the repository's first GENERATIVE-BY-DIFFUSION example -- a
Denoising Diffusion Probabilistic Model (DDPM, Ho et al. 2020,
https://arxiv.org/abs/2006.11239) that learns to synthesise 28x28 MNIST
digits from pure Gaussian noise.

WHAT A DDPM IS (in one paragraph). A fixed FORWARD process gradually corrupts
a clean image x_0 into pure noise x_T over T steps by repeatedly adding a
little Gaussian noise on a LINEAR beta schedule (beta_1=1e-4 .. beta_T=0.02).
The closed form jumps directly to any timestep t:

    x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * eps,  eps~N(0,I)

where alpha_t = 1 - beta_t and alpha_bar_t = prod_{s<=t} alpha_s. A neural
network eps_theta(x_t, t) is trained to PREDICT the noise eps that was added
(the simple epsilon-prediction MSE objective). Once it can denoise, we run the
REVERSE (ancestral / DDPM) sampling loop from x_T~N(0,I) back down to x_0,
subtracting the predicted noise step by step, to draw fresh digits.

TIME CONDITIONING. The integer timestep t must be fed to the network so it
knows how much noise to expect. This example USES TNNetSinusoidalTimeEmbedding
(coded specifically for diffusion) to map the scalar t to a sinusoidal vector,
then injects it into every U-Net block as a per-channel scale/shift via
TNNet.AddFiLMConditioned (Feature-wise Linear Modulation / TNNetFiLM). This is
the standard DDPM time-embedding -> FiLM recipe.

THE NETWORK is a small two-input U-Net (image branch + timestep branch):

  image  (28,28,1) -- TNNetInput
  t      (1,1,1)   -- TNNetInput  -> TNNetSinusoidalTimeEmbedding(cEmbDim)
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
  neuraldatasets;

const
  cImgSize   = 28;
  cEmbDim    = 64;          // sinusoidal time-embedding width (must be even)
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
  cSmokeSteps = 500;
  cSmokeBatch = 16;
  cSmokeGrid  = 4;          // 4x4 = 16 generated digits

  cFullSteps  = 6000;
  cFullBatch  = 32;
  cFullGrid   = 8;          // 8x8 = 64 generated digits

var
  // Precomputed schedule tables, index 1..cT (index 0 unused / = identity).
  Beta, Alpha, AlphaBar, SqrtAlphaBar, SqrtOneMinusAlphaBar: array[0..cT] of TNeuralFloat;

  NN: TNNet;
  ImgIn, TimeIn, TimeEmb: TNNetLayer;
  TrainV, ValV, TestV: TNNetVolumeList;

  // Reusable working volumes for the training/sampling loops.
  Img0, ImgT, EpsTrue, TimeVol: TNNetVolume;

  Steps, BatchSz, GridN: integer;
  FullMode: boolean;

  // Build the linear-beta schedule and its derived alpha_bar tables.
  procedure BuildSchedule;
  var
    i: integer;
    Prod: TNeuralFloat;
  begin
    Beta[0] := 0; Alpha[0] := 1; AlphaBar[0] := 1;
    SqrtAlphaBar[0] := 1; SqrtOneMinusAlphaBar[0] := 0;
    Prod := 1.0;
    for i := 1 to cT do
    begin
      Beta[i]  := cBeta1 + (cBetaT - cBeta1) * (i - 1) / (cT - 1);
      Alpha[i] := 1.0 - Beta[i];
      Prod     := Prod * Alpha[i];
      AlphaBar[i] := Prod;
      SqrtAlphaBar[i]         := Sqrt(Prod);
      SqrtOneMinusAlphaBar[i] := Sqrt(1.0 - Prod);
    end;
  end;

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
    SkipA, SkipB: TNNetLayer;
  begin
    NN := TNNet.Create();
    // Two inputs: the (noisy) image and the scalar timestep.
    ImgIn  := NN.AddLayer(TNNetInput.Create(cImgSize, cImgSize, 1, 1));
    TimeIn := NN.AddLayerAfter(TNNetInput.Create(1, 1, 1, 1), 0);

    // Shared time-embedding branch: scalar t -> sinusoidal vector -> small MLP.
    TimeEmb := NN.AddLayerAfter(TNNetSinusoidalTimeEmbedding.Create(cEmbDim), TimeIn);
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

  // Pull a clean MNIST digit into Img0 rescaled to [-1, 1].
  // The loader stores pixels as byte/64 - 2, so byte = (v+2)*64 and the
  // [-1,1] target is byte/127.5 - 1.
  procedure LoadCleanDigit(Src: TNNetVolume);
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
  end;

  // Forward-noise Img0 to timestep t: ImgT = sqrt(ab)*x0 + sqrt(1-ab)*eps,
  // filling EpsTrue with the sampled noise (the training target).
  procedure NoiseToTimestep(t: integer);
  var
    i: integer;
    eps: TNeuralFloat;
  begin
    for i := 0 to Img0.Size - 1 do
    begin
      eps := RandG(0, 1);       // mean 0, std 1
      EpsTrue.FData[i] := eps;
      ImgT.FData[i] := SqrtAlphaBar[t] * Img0.FData[i] +
                       SqrtOneMinusAlphaBar[t] * eps;
    end;
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

  procedure TrainLoop;
  var
    Step, B, t, Idx: integer;
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
        LoadCleanDigit(TrainV[Idx]);
        t := 1 + Random(cT);                 // uniform timestep in 1..T
        NoiseToTimestep(t);
        SetTimestepInput(t);
        NN.Compute([ImgT, TimeVol]);         // two-input forward
        SumLoss := SumLoss + EpsMSE;
        // Train eps-prediction: target is the sampled noise EpsTrue.
        NN.Backpropagate(EpsTrue);
      end;
      // Tame the early gradient spikes a from-scratch diffusion net produces
      // before the noise-schedule statistics settle.
      NN.ForceMaxAbsoluteDelta(0.05);
      NN.UpdateWeights();
      if (Step = 1) or (Step mod 25 = 0) or (Step = Steps) then
      begin
        Elapsed := (Now() - StartTime) * 86400.0;
        WriteLn(Format('  step %4d / %4d   eps-MSE = %.5f   elapsed = %.1fs',
          [Step, Steps, SumLoss / BatchSz, Elapsed]));
      end;
    end;
  end;

  // Ancestral DDPM reverse sampling for ONE image, leaving x_0 in Dst.
  // x_{t-1} = 1/sqrt(alpha_t) * (x_t - beta_t/sqrt(1-ab_t) * eps_hat)
  //           + sqrt(beta_t) * z     (z=0 at t=1)
  procedure SampleOne(Dst: TNNetVolume);
  var
    t, i: integer;
    Pred: TNNetVolume;
    coef, invSqrtAlpha, sigma, z: TNeuralFloat;
  begin
    // Start from pure Gaussian noise x_T.
    for i := 0 to Dst.Size - 1 do
      Dst.FData[i] := RandG(0, 1);
    for t := cT downto 1 do
    begin
      SetTimestepInput(t);
      NN.Compute([Dst, TimeVol]);
      Pred := NN.GetLastLayer.Output;
      invSqrtAlpha := 1.0 / Sqrt(Alpha[t]);
      coef := Beta[t] / SqrtOneMinusAlphaBar[t];
      if t > 1 then sigma := Sqrt(Beta[t]) else sigma := 0;
      for i := 0 to Dst.Size - 1 do
      begin
        if t > 1 then z := RandG(0, 1) else z := 0;
        Dst.FData[i] := invSqrtAlpha *
          (Dst.FData[i] - coef * Pred.FData[i]) + sigma * z;
      end;
    end;
  end;

  // Sample GridN*GridN digits and save them as one grayscale PNG grid.
  procedure SampleGridToPng(const FileName: string);
  var
    Grid, One: TNNetVolume;
    r, c, x, y, gx, gy: integer;
    v, px: TNeuralFloat;
    NumNan: integer;
  begin
    Grid := TNNetVolume.Create(GridN * cImgSize, GridN * cImgSize, 3);
    One  := TNNetVolume.Create(cImgSize, cImgSize, 1);
    Grid.Fill(0);
    NumNan := 0;
    try
      WriteLn(Format('Sampling a %dx%d grid via %d-step ancestral DDPM...',
        [GridN, GridN, cT]));
      for r := 0 to GridN - 1 do
        for c := 0 to GridN - 1 do
        begin
          SampleOne(One);
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
      if SaveImageFromVolumeIntoFile(Grid, FileName)
        then WriteLn('Wrote sample grid: ', FileName)
        else WriteLn('FAILED to write: ', FileName);
      if NumNan > 0
        then WriteLn('WARNING: ', NumNan, ' NaN/Inf pixels were clamped.')
        else WriteLn('No NaN/Inf in generated samples.');
    finally
      Grid.Free;
      One.Free;
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
  WriteLn('DDPM epsilon-prediction (Ho et al. 2020), tiny time-conditioned U-Net.');
  WriteLn;

  if not (CheckMNISTFile('train')) or not (CheckMNISTFile('t10k')) then
  begin
    WriteLn('MNIST idx-ubyte files not found in the working directory; exiting.');
    Exit;
  end;

  BuildSchedule;

  CreateMNISTVolumes(TrainV, ValV, TestV, 'train', 't10k');
  WriteLn('Loaded MNIST: ', TrainV.Count, ' train digits.');

  BuildNet;
  WriteLn('Network architecture:');
  NN.PrintSummary();
  WriteLn;

  Img0    := TNNetVolume.Create(cImgSize, cImgSize, 1);
  ImgT    := TNNetVolume.Create(cImgSize, cImgSize, 1);
  EpsTrue := TNNetVolume.Create(cImgSize, cImgSize, 1);
  TimeVol := TNNetVolume.Create(1, 1, 1);
  try
    TrainLoop;
    WriteLn;
    PngName := 'diffusion_samples.png';
    SampleGridToPng(PngName);
    WriteLn;
    WriteLn('Done. View ', PngName, ' to inspect the generated digits.');
    if not FullMode then
      WriteLn('(Smoke samples are noisy; run with --full for sharp digits.)');
  finally
    Img0.Free;
    ImgT.Free;
    EpsTrue.Free;
    TimeVol.Free;
    NN.Free;
    TrainV.Free;
    ValV.Free;
    TestV.Free;
  end;
end.
