program FlowMatching;
(*
FlowMatching: a Flow Matching / Rectified Flow generative model that learns to
synthesise 28x28 MNIST digits by transporting Gaussian noise to data along
STRAIGHT interpolation paths. This is the modern ODE/transport alternative to
the DDPM diffusion example (examples/DiffusionMNIST) -- same tiny conditional
U-Net, completely different (and simpler) objective and sampler.

WHAT FLOW MATCHING / RECTIFIED FLOW IS (in one paragraph). Pick a noise sample
x0 ~ N(0,I) and a real data sample x1 (an MNIST digit). Define the LINEAR
interpolant (a straight line in pixel space) between them:

    x_t = (1 - t) * x0 + t * x1 ,    t ~ U(0,1)

The constant-velocity field that moves along this line is simply

    dx_t/dt = x1 - x0

so we train a neural network v_theta(x_t, t) to REGRESS that velocity with a
plain MSE loss (this is the conditional flow-matching / rectified-flow
objective, Lipman et al. 2023, Liu et al. 2023):

    L = E_{t,x0,x1}  || v_theta(x_t, t) - (x1 - x0) ||^2

There is NO noise schedule, NO alpha_bar tables and NO score / epsilon
parameterisation -- just a velocity regressed along straight lines.

SAMPLING is forward ODE integration. Start from pure noise x_0 = x0 ~ N(0,I)
(here t runs 0 -> 1) and take a handful of explicit Euler steps:

    x_{t+dt} = x_t + dt * v_theta(x_t, t) ,    dt = 1 / NumSteps

After NumSteps steps we land at t=1, i.e. a generated digit. Because the
learned paths are (near) straight, 10-50 Euler steps already give good samples,
far fewer than the ancestral DDPM loop.

CONTRAST WITH DDPM (examples/DiffusionMNIST). DDPM corrupts x1 on a fixed beta
schedule x_t = sqrt(abar_t)*x1 + sqrt(1-abar_t)*eps, trains the net to PREDICT
the noise eps (a score-like target), and SAMPLES with a stochastic ancestral
reverse loop (or DDIM). Flow matching replaces the schedule with a single
straight line, replaces noise-prediction with velocity-regression, and replaces
the reverse diffusion loop with a deterministic forward ODE. Same U-Net, much
simpler maths.

TIME CONDITIONING. The network must know how far along the path it is. We reuse
TNNetSinusoidalTimeEmbedding (the same layer DDPM uses). That embedding maps a
scalar to a sinusoidal vector via angle = t * freq[i] and was designed for the
INTEGER timestep range of diffusion (t up to a few hundred). Flow matching uses
a CONTINUOUS t in [0,1], so to keep the embedding well-conditioned (i.e. to use
the same numeric range it was built for) we feed it t * cTimeScale, with
cTimeScale = 1000. The training target and the ODE step both still use the true
continuous t in [0,1]; only the value handed to the embedding is rescaled.

THE NETWORK is a small two-input U-Net (image + time):

  image  (28,28,1) -- TNNetInput
  t      (1,1,1)   -- TNNetInput -> TNNetSinusoidalTimeEmbedding(cEmbDim)
                                  -> FullConnect MLP (shared cond vector)

  enc1: Conv(C1,28x28) -> GroupNorm -> FiLM(t) -> Conv(C1)   ........ skip A
        MaxPool /2 -> 14x14
  enc2: Conv(C2,14x14) -> GroupNorm -> FiLM(t) -> Conv(C2)   ........ skip B
        MaxPool /2 -> 7x7
  mid : Conv(C3,7x7)  -> GroupNorm -> FiLM(t) -> Conv(C3)
  dec2: Upsample x2 -> 14x14 ; DeepConcat(skip B) ; Conv(C2) -> GroupNorm
        -> FiLM(t) -> Conv(C2)
  dec1: Upsample x2 -> 28x28 ; DeepConcat(skip A) ; Conv(C1) -> GroupNorm
        -> FiLM(t) -> Conv(C1)
  head: ConvLinear(1, 3x3) -> predicted velocity v_hat (28,28,1)

The time embedding is injected into every block as a per-channel scale/shift via
TNNet.AddFiLMConditioned (FiLM). All layers already existed; the only new
content versus the DDPM example is the linear-interpolant velocity target and
the Euler ODE sampler.

RUN MODES.
  default (SMOKE): a small number of training steps then a tiny sample grid;
    finishes in a few minutes on one CPU. Good enough to see the loss fall and
    to confirm a PNG grid is written without NaN.
  full training:  pass  --full  for many more steps / a bigger grid.

OUTPUT. A PNG grid of generated digits (flowmatching_samples.png) in the working
directory. Pixels are in [-1,1] internally and mapped to 0..255 for the PNG.

DATA. Reuses the MNIST idx-ubyte files committed under examples/DiffusionMNIST
(loaded via the relative path ../DiffusionMNIST/...). No data is copied here.

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

  // Continuous t in [0,1] is scaled by this before the sinusoidal embedding so
  // the embedding sees roughly the integer-step numeric range it was built for.
  cTimeScale = 1000.0;

  // Channel widths per U-Net level (kept small so the smoke run is fast).
  cC1        = 16;
  cC2        = 32;
  cC3        = 48;

  // Number of Euler ODE steps used at sampling time (no schedule -- just 1/N).
  cEulerSteps = 25;

  cLR        = 0.0005;
  cInertia   = 0.9;

  // Default = fast smoke. --full overrides via cmd line below.
  cSmokeSteps = 500;
  cSmokeBatch = 16;
  cSmokeGrid  = 8;

  cFullSteps  = 6000;
  cFullBatch  = 32;
  cFullGrid   = 10;

  // MNIST data lives in the sibling DiffusionMNIST example -- do not duplicate.
  cTrainBase = '../DiffusionMNIST/train';
  cTestBase  = '../DiffusionMNIST/t10k';

var
  NN: TNNet;
  ImgIn, TimeIn, TimeEmb: TNNetLayer;
  TrainV, ValV, TestV: TNNetVolumeList;

  // Reusable working volumes for the training / sampling loops.
  X0, X1, Xt, VTrue, TimeVol: TNNetVolume;

  Steps, BatchSz, GridN: integer;
  FullMode: boolean;

  // One conv -> GroupNorm -> FiLM(time) -> conv block at the current spatial
  // resolution. Returns the block output layer (the post-FiLM refinement).
  function ConvBlock(Channels: integer): TNNetLayer;
  var
    Feat: TNNetLayer;
  begin
    NN.AddLayer(TNNetConvolutionReLU.Create(Channels, 3, 1, 1, 0));
    if (Channels mod 4) = 0
      then NN.AddLayer(TNNetGroupNorm.Create(4))
      else NN.AddLayer(TNNetGroupNorm.Create(Channels));
    Feat := NN.GetLastLayer;
    NN.AddFiLMConditioned(Feat, TimeEmb);
    Result := NN.AddLayer(TNNetConvolutionReLU.Create(Channels, 3, 1, 1, 0));
  end;

  procedure BuildNet;
  var
    SkipA, SkipB: TNNetLayer;
  begin
    NN := TNNet.Create();
    // Two inputs: the (interpolated) image and the scalar continuous time t.
    ImgIn  := NN.AddLayer(TNNetInput.Create(cImgSize, cImgSize, 1, 1));
    TimeIn := NN.AddLayerAfter(TNNetInput.Create(1, 1, 1, 1), 0);

    // Time branch: scalar (t*cTimeScale) -> sinusoidal vector -> shared cond MLP.
    TimeEmb := NN.AddLayerAfter(TNNetSinusoidalTimeEmbedding.Create(cEmbDim), TimeIn);
    TimeEmb := NN.AddLayerAfter(TNNetFullConnectReLU.Create(cEmbDim), TimeEmb);
    TimeEmb := NN.AddLayerAfter(TNNetFullConnectReLU.Create(cEmbDim), TimeEmb);

    // Encoder. Continue the image path explicitly from ImgIn (the last added
    // layer above belongs to the time branch).
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

    // Velocity-prediction head: linear conv to a single channel (v_hat).
    NN.AddLayer(TNNetConvolutionLinear.Create(1, 3, 1, 1, 0));

    NN.SetLearningRate(cLR, cInertia);
    NN.SetBatchUpdate(true);
  end;

  // Pull a clean MNIST digit into X1 rescaled to [-1, 1]. The loader stores
  // pixels as byte/64 - 2, so byte = (v+2)*64 and the [-1,1] target is
  // byte/127.5 - 1 (identical convention to the DDPM example).
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
        X1[x, y, 0] := b / 127.5 - 1.0;
      end;
  end;

  // Build the flow-matching training pair for a continuous time t in [0,1]:
  //   X0 ~ N(0,I)  (noise),  X1 = clean digit (already loaded),
  //   Xt = (1-t)*X0 + t*X1,  VTrue = X1 - X0  (the constant velocity target).
  procedure MakeInterpolant(t: TNeuralFloat);
  var
    i: integer;
    x0v: TNeuralFloat;
  begin
    for i := 0 to X1.Size - 1 do
    begin
      x0v := RandG(0, 1);
      X0.FData[i] := x0v;
      Xt.FData[i] := (1.0 - t) * x0v + t * X1.FData[i];
      VTrue.FData[i] := X1.FData[i] - x0v;
    end;
  end;

  // Feed continuous t (scaled) into the time input so the sinusoidal embedding
  // sees its designed numeric range.
  procedure SetTimeInput(t: TNeuralFloat);
  begin
    TimeVol.FData[0] := t * cTimeScale;
    TimeIn.Output.FData[0] := t * cTimeScale;
  end;

  // Mean-squared error between the network's v_hat and the true velocity.
  function VelMSE: TNeuralFloat;
  var
    i: integer;
    d, s: TNeuralFloat;
    Pred: TNNetVolume;
  begin
    Pred := NN.GetLastLayer.Output;
    s := 0;
    for i := 0 to VTrue.Size - 1 do
    begin
      d := Pred.FData[i] - VTrue.FData[i];
      s := s + d * d;
    end;
    Result := s / VTrue.Size;
  end;

  procedure TrainLoop;
  var
    Step, B, Idx: integer;
    t, SumLoss: TNeuralFloat;
    StartTime, Elapsed: double;
  begin
    WriteLn(Format('Training: %d steps, batch %d. Flow-matching MSE on '+
      'v_theta(x_t,t) vs (x1-x0), t~U(0,1).', [Steps, BatchSz]));
    StartTime := Now();
    for Step := 1 to Steps do
    begin
      SumLoss := 0;
      NN.ClearDeltas();
      for B := 1 to BatchSz do
      begin
        Idx := Random(TrainV.Count);
        LoadCleanDigit(TrainV[Idx]);
        t := Random;                     // continuous t ~ U(0,1)
        MakeInterpolant(t);
        SetTimeInput(t);
        NN.Compute([Xt, TimeVol]);       // two-input forward
        SumLoss := SumLoss + VelMSE;
        NN.Backpropagate(VTrue);         // regress the velocity (x1 - x0)
      end;
      // Tame early gradient spikes a from-scratch net produces.
      NN.ForceMaxAbsoluteDelta(0.03);
      NN.UpdateWeights();
      if (Step = 1) or (Step mod 25 = 0) or (Step = Steps) then
      begin
        Elapsed := (Now() - StartTime) * 86400.0;
        WriteLn(Format('  step %4d / %4d   vel-MSE = %.5f   elapsed = %.1fs',
          [Step, Steps, SumLoss / BatchSz, Elapsed]));
      end;
    end;
  end;

  // Deterministic forward-ODE Euler sampler for ONE image, leaving the result
  // in Dst. Start at x0 ~ N(0,I) (t=0) and integrate to t=1:
  //   x_{t+dt} = x_t + dt * v_theta(x_t, t),   dt = 1/NumSteps.
  procedure SampleOne(Dst: TNNetVolume; NumSteps: integer);
  var
    k, i: integer;
    t, dt: TNeuralFloat;
    V: TNNetVolume;
  begin
    dt := 1.0 / NumSteps;
    for i := 0 to Dst.Size - 1 do
      Dst.FData[i] := RandG(0, 1);       // x at t=0 is pure noise
    for k := 0 to NumSteps - 1 do
    begin
      t := k * dt;                       // current time along the path
      SetTimeInput(t);
      NN.Compute([Dst, TimeVol]);
      V := NN.GetLastLayer.Output;
      for i := 0 to Dst.Size - 1 do
        Dst.FData[i] := Dst.FData[i] + dt * V.FData[i];
    end;
  end;

  // Sample a grid of digits via the Euler ODE sampler; save as one PNG.
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
      WriteLn(Format('Sampling a %dx%d grid via %d-step Euler ODE '+
        '(no schedule)...', [GridN, GridN, cEulerSteps]));
      for r := 0 to GridN - 1 do
        for c := 0 to GridN - 1 do
        begin
          SampleOne(One, cEulerSteps);
          for y := 0 to cImgSize - 1 do
            for x := 0 to cImgSize - 1 do
            begin
              v := One[x, y, 0];
              if IsNan(v) or IsInfinite(v) then begin Inc(NumNan); v := 0; end;
              px := (v + 1.0) * 127.5;             // [-1,1] -> 0..255
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
    WriteLn('FlowMatching [FULL training mode]');
  end
  else
  begin
    Steps := cSmokeSteps; BatchSz := cSmokeBatch; GridN := cSmokeGrid;
    WriteLn('FlowMatching [SMOKE mode -- pass --full for real training]');
  end;
  WriteLn('Rectified-flow / flow-matching velocity regression on a tiny '+
          'time-conditioned U-Net.');
  WriteLn('Loss: MSE(v_theta(x_t,t), x1-x0) on x_t=(1-t)x0+t*x1; '+
          'sampling: forward Euler ODE.');
  WriteLn;

  if not (CheckMNISTFile(cTrainBase)) or not (CheckMNISTFile(cTestBase)) then
  begin
    WriteLn('MNIST idx-ubyte files not found at ../DiffusionMNIST/; exiting.');
    Exit;
  end;

  CreateMNISTVolumes(TrainV, ValV, TestV, cTrainBase, cTestBase);
  WriteLn('Loaded MNIST: ', TrainV.Count, ' train digits.');

  BuildNet;
  WriteLn('Network architecture:');
  NN.PrintSummary();
  WriteLn;

  X0     := TNNetVolume.Create(cImgSize, cImgSize, 1);
  X1     := TNNetVolume.Create(cImgSize, cImgSize, 1);
  Xt     := TNNetVolume.Create(cImgSize, cImgSize, 1);
  VTrue  := TNNetVolume.Create(cImgSize, cImgSize, 1);
  TimeVol := TNNetVolume.Create(1, 1, 1);
  try
    TrainLoop;
    WriteLn;
    PngName := 'flowmatching_samples.png';
    SampleGridToPng(PngName);
    WriteLn;
    WriteLn('Done. View ', PngName, '.');
    if not FullMode then
      WriteLn('(Smoke samples are noisy; run with --full for sharp digits.)');
  finally
    X0.Free;
    X1.Free;
    Xt.Free;
    VTrue.Free;
    TimeVol.Free;
    NN.Free;
    TrainV.Free;
    ValV.Free;
    TestV.Free;
  end;
end.
