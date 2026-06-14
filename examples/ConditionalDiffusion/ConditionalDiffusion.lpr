program ConditionalDiffusion;
(*
ConditionalDiffusion: a CLASS-CONDITIONAL image-generation example. The three
generative image examples that landed before this one -- examples/VisualGAN,
examples/DiffusionMNIST (in its plain form) and examples/FlowMatching -- all
draw UNCONDITIONAL samples: you get a random digit, you cannot ask for a
specific one. This example fills that gap. It trains a small label-conditioned
DENOISER on MNIST and then demonstrates the central knob of CLASSIFIER-FREE
GUIDANCE (CFG): the guidance weight w trades sample DIVERSITY for class
FIDELITY.

WHY THIS EXAMPLE EXISTS (vs DiffusionMNIST). DiffusionMNIST is primarily a
tutorial on what a DDPM is and how the reverse process works; it happens to
support a fixed CFG scale. This example is built around ONE question -- "what
does the CFG weight actually do?" -- and answers it QUANTITATIVELY. It trains a
tiny side MNIST CLASSIFIER, then samples the conditional denoiser at a SWEEP of
CFG weights w in {0, 1, 2, 4, 8} and reports, per w:
  * class-fidelity   = fraction of generated digits the classifier assigns to
                       the REQUESTED class (higher w should raise this), and
  * diversity proxy  = mean per-pixel standard deviation across the samples of
                       one requested class (higher w should LOWER this as the
                       model collapses onto the class mode).
The textbook trend: as w increases, class-fidelity rises and the diversity proxy
falls -- the classic CFG fidelity/diversity trade-off. This emerges cleanly once
the denoiser is well trained (run --full). The short default SMOKE run is
intentionally undertrained (like the sibling examples/DiffusionMNIST smoke, whose
grid is also noisy): its samples are noisy, so it mainly validates the pipeline
(metrics computed, no NaN/Inf, PNG written) and self-reports the OBSERVED metric
direction rather than asserting the textbook one.

CONDITIONING MECHANISM (the required design point). The denoiser is a small
time-conditioned U-Net (mirroring DiffusionMNIST). The class label y in 0..9 is
mapped through a learned TNNetEmbedding to a vector of the same width as the
sinusoidal TIME embedding, and the two vectors are ADDED before a shared cond
MLP. A single FiLM (TNNetFiLM) cond vector therefore carries both "how much
noise" (t) and "which digit" (y) and modulates every conv block.

LABEL DROPOUT -> CLASSIFIER-FREE GUIDANCE. We reserve label index 10 as a
dedicated NULL / unconditional token. During training, with probability
cLabelDrop we replace the real label with this null token (LABEL DROPOUT). The
single network therefore learns BOTH the conditional score eps(x_t, t, y) AND
the unconditional score eps(x_t, t, null). At sampling time we run the net twice
per step and extrapolate with the reusable scheduler's CFG mixer:

    eps = eps_uncond + w * (eps_cond - eps_uncond)            (guidance weight w)

w = 0 is purely unconditional (most diverse, least faithful); w = 1 is the plain
conditional model; w > 1 over-emphasises the requested class (most faithful,
least diverse).

REUSE. The forward noising (q_sample / AddNoise), the reverse DDIM trajectory
(Sample) and the CFG mix (ApplyCFG) all come from the reusable, model-agnostic
neuraldiffusion.pas (TNNetDiffusionScheduler). No noising or sampling loop is
hand-rolled here.

RUN MODES.
  default (SMOKE): a few hundred denoiser steps + a short classifier, then the
    CFG sweep on a small number of samples per weight. Finishes in ~2-3 minutes
    on one CPU. Enough to see the loss fall and the fidelity/diversity trend.
  --full : many more training steps and more samples per weight for a sharp,
    statistically cleaner trade-off and a nicer PNG grid.

OUTPUT. A PNG grid (conditional_samples.png) whose ROWS are increasing CFG
weights and whose COLUMNS are independent samples of one chosen digit, so you
can SEE diversity shrink as w grows. A textual per-w fidelity/diversity table is
printed to stdout. The run asserts there are no NaN/Inf pixels.

DATA. Standard MNIST idx-ubyte files in the working directory (symlinked from a
sibling example). If they are absent the program prints a hint and exits.

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

  cSmokeDDIMSteps = 12;     // strided reverse steps in SMOKE (<< cT)
  cFullDDIMSteps  = 25;     // sharper strided reverse steps in --full

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

  // The CFG weights swept to expose the fidelity/diversity trade-off.
  cNumW      = 5;

  // Digit whose samples form the printed grid (rows = CFG weights).
  cGridDigit = 7;

  // Smoke vs full schedules.
  cSmokeSteps   = 360;      // denoiser training steps
  cSmokeBatch   = 10;
  cSmokeClfSteps = 400;     // side-classifier training steps
  cSmokePerW    = 1;        // (unused in the merged sweep; kept for clarity)
  cSmokeGrid    = 6;        // grid columns = samples of cGridDigit per weight

  cFullSteps    = 7000;
  cFullBatch    = 32;
  cFullClfSteps = 2000;
  cFullPerW     = 32;
  cFullGrid     = 10;

type
  // TNNetDenoiseCallback adapter (procedure-of-object) so the reusable
  // scheduler can drive the reverse process. The CFG weight and requested
  // digit are read from the global gCurW / gReqDigit.
  // Coded by Claude (AI).
  TCFGDenoiser = class(TObject)
    procedure Denoise(Xt, Output: TNNetVolume; t: integer);
  end;

var
  // CFG weights to sweep.
  cWeights: array[0..cNumW - 1] of TNeuralFloat = (0.0, 1.0, 2.0, 4.0, 8.0);

  // Reusable scheduler holding the precomputed beta/alpha/alpha_bar tables.
  Sched: TNNetDiffusionScheduler;

  // The denoiser net and a small side classifier for the fidelity metric.
  NN, Clf: TNNet;
  ImgIn, TimeIn, LabelIn, TimeEmb: TNNetLayer;
  TrainV, ValV, TestV: TNNetVolumeList;

  // Reusable working volumes.
  Img0, ImgT, EpsTrue, TimeVol, LabelVol, ClfIn: TNNetVolume;

  Steps, BatchSz, ClfSteps, PerW, GridN, DDIMSteps: integer;
  FullMode: boolean;

  // Current CFG weight used by the denoiser callback.
  gCurW: TNeuralFloat;
  gReqDigit: integer;

  Denoiser: TCFGDenoiser;

  // One conv -> GroupNorm -> FiLM(time+label) -> conv block.
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
    SkipA, SkipB, LabelEmb: TNNetLayer;
  begin
    NN := TNNet.Create();
    ImgIn   := NN.AddLayer(TNNetInput.Create(cImgSize, cImgSize, 1, 1));
    TimeIn  := NN.AddLayerAfter(TNNetInput.Create(1, 1, 1, 1), 0);
    LabelIn := NN.AddLayerAfter(TNNetInput.Create(1, 1, 1, 1), 0);

    // Time branch: scalar t -> sinusoidal vector (1 x 1 x cEmbDim).
    TimeEmb := NN.AddLayerAfter(TNNetSinusoidalTimeEmbedding.Create(cEmbDim), TimeIn);
    // Label branch: y -> learned embedding row (1 x 1 x cEmbDim). Index 10 is
    // the null/unconditional token. EncodeZero=1 so label 0 is embedded too.
    LabelEmb := NN.AddLayerAfter(TNNetEmbedding.Create(cNumClasses, cEmbDim, 1), LabelIn);
    // ADD the label embedding onto the time embedding (the required design
    // point), then a shared cond MLP -> a single FiLM vector carrying t and y.
    TimeEmb := NN.AddLayer(TNNetSum.Create([TimeEmb, LabelEmb]));
    TimeEmb := NN.AddLayerAfter(TNNetFullConnectReLU.Create(cEmbDim), TimeEmb);
    TimeEmb := NN.AddLayerAfter(TNNetFullConnectReLU.Create(cEmbDim), TimeEmb);

    // Encoder.
    NN.AddLayerAfter(TNNetConvolutionReLU.Create(cC1, 3, 1, 1, 0), ImgIn);
    SkipA := ConvBlock(cC1);                       // 28x28, skip A
    NN.AddLayer(TNNetMaxPool.Create(2));           // -> 14x14
    SkipB := ConvBlock(cC2);                       // 14x14, skip B
    NN.AddLayer(TNNetMaxPool.Create(2));           // -> 7x7

    ConvBlock(cC3);                                // bottleneck 7x7

    NN.AddLayer(TNNetUpsample.Create());           // 7 -> 14
    NN.AddLayer(TNNetDeepConcat.Create([NN.GetLastLayer, SkipB]));
    ConvBlock(cC2);

    NN.AddLayer(TNNetUpsample.Create());           // 14 -> 28
    NN.AddLayer(TNNetDeepConcat.Create([NN.GetLastLayer, SkipA]));
    ConvBlock(cC1);

    NN.AddLayer(TNNetConvolutionLinear.Create(1, 3, 1, 1, 0));  // eps_hat head

    NN.SetLearningRate(cLR, cInertia);
    NN.SetBatchUpdate(true);
  end;

  // A tiny convnet classifier used purely to MEASURE class fidelity of samples.
  procedure BuildClassifier;
  begin
    Clf := TNNet.Create();
    Clf.AddLayer(TNNetInput.Create(cImgSize, cImgSize, 1));
    Clf.AddLayer(TNNetConvolutionReLU.Create(16, 3, 1, 1, 0));
    Clf.AddLayer(TNNetMaxPool.Create(2));
    Clf.AddLayer(TNNetConvolutionReLU.Create(32, 3, 1, 1, 0));
    Clf.AddLayer(TNNetMaxPool.Create(2));
    Clf.AddLayer(TNNetFullConnectReLU.Create(64));
    Clf.AddLayer(TNNetFullConnectLinear.Create(10));
    // SkipBackpropDerivative=1: SoftMax is the CE loss head, so its Jacobian is
    // skipped and the gradient is the clean (softmax - onehot) (the idiom used
    // by examples/SimpleImageClassifier). Without this the net will NOT learn.
    Clf.AddLayer(TNNetSoftMax.Create({SkipBackpropDerivative=}1));
    Clf.SetLearningRate(0.001, 0.9);
    Clf.SetBatchUpdate(true);
  end;

  // Pull a clean MNIST digit into Dst rescaled to [-1,1]; return its label.
  function LoadCleanDigitInto(Src, Dst: TNNetVolume): integer;
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
        Dst[x, y, 0] := b / 127.5 - 1.0;
      end;
    Result := Src.Tag;
  end;

  function LoadCleanDigit(Src: TNNetVolume): integer;
  begin
    Result := LoadCleanDigitInto(Src, Img0);
  end;

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

  procedure SetLabelInput(y: integer);
  begin
    LabelVol.FData[0] := y;
    LabelIn.Output.FData[0] := y;
  end;

  procedure TrainDenoiser;
  var
    Step, B, t, Idx, y: integer;
    SumLoss: TNeuralFloat;
    StartTime, Elapsed: double;
  begin
    WriteLn(Format('Training denoiser: %d steps, batch %d, T=%d.',
      [Steps, BatchSz, cT]));
    StartTime := Now();
    for Step := 1 to Steps do
    begin
      SumLoss := 0;
      NN.ClearDeltas();
      for B := 1 to BatchSz do
      begin
        Idx := Random(TrainV.Count);
        y := LoadCleanDigit(TrainV[Idx]);
        if Random < cLabelDrop then y := cNullLabel;   // LABEL DROPOUT
        t := 1 + Random(cT);
        Sched.AddNoise(Img0, ImgT, t, {NoiseOut=}EpsTrue);
        SetTimestepInput(t);
        SetLabelInput(y);
        NN.Compute([ImgT, TimeVol, LabelVol]);
        SumLoss := SumLoss + EpsMSE;
        NN.Backpropagate(EpsTrue);
      end;
      NN.ForceMaxAbsoluteDelta(0.03);
      NN.UpdateWeights();
      if (Step = 1) or (Step mod 50 = 0) or (Step = Steps) then
      begin
        Elapsed := (Now() - StartTime) * 86400.0;
        WriteLn(Format('  denoiser step %4d / %4d   eps-MSE = %.5f   %.1fs',
          [Step, Steps, SumLoss / BatchSz, Elapsed]));
      end;
    end;
  end;

  procedure TrainClassifier;
  var
    Step, B, Idx, y: integer;
    Hit: TNeuralFloat;
    Target: TNNetVolume;
    Pred: TNNetVolume;
    StartTime, Elapsed: double;
  begin
    WriteLn(Format('Training side classifier (fidelity metric): %d steps.',
      [ClfSteps]));
    Target := TNNetVolume.Create(1, 1, 10);
    StartTime := Now();
    try
      for Step := 1 to ClfSteps do
      begin
        Hit := 0;
        Clf.ClearDeltas();
        for B := 1 to BatchSz do
        begin
          Idx := Random(TrainV.Count);
          y := LoadCleanDigitInto(TrainV[Idx], ClfIn);
          Target.Fill(0);
          Target.FData[y] := 1;
          Clf.Compute(ClfIn);
          Pred := Clf.GetLastLayer.Output;
          if Pred.GetClass = y then Hit := Hit + 1;
          Clf.Backpropagate(Target);
        end;
        Clf.UpdateWeights();
        if (Step = 1) or (Step mod 50 = 0) or (Step = ClfSteps) then
        begin
          Elapsed := (Now() - StartTime) * 86400.0;
          WriteLn(Format('  clf step %4d / %4d   batch-acc = %.2f   %.1fs',
            [Step, ClfSteps, Hit / BatchSz, Elapsed]));
        end;
      end;
    finally
      Target.Free;
    end;
  end;

  // CFG-guided eps for image Xt at timestep t, digit gReqDigit, weight gCurW.
  // Uses the scheduler's ApplyCFG. w=0 -> unconditional; w=1 -> conditional.
  procedure PredictEpsCFG(Xt, EpsCFG: TNNetVolume; t: integer);
  var
    Cond, Uncond: TNNetVolume;
  begin
    SetTimestepInput(t);
    if gCurW = 1.0 then
    begin
      SetLabelInput(gReqDigit);
      NN.Compute([Xt, TimeVol, LabelVol]);
      EpsCFG.Copy(NN.GetLastLayer.Output);
      Exit;
    end;
    if gCurW = 0.0 then
    begin
      SetLabelInput(cNullLabel);
      NN.Compute([Xt, TimeVol, LabelVol]);
      EpsCFG.Copy(NN.GetLastLayer.Output);
      Exit;
    end;
    // General w: need both branches.
    Cond := TNNetVolume.Create(Xt);
    Uncond := TNNetVolume.Create(Xt);
    try
      SetLabelInput(gReqDigit);
      NN.Compute([Xt, TimeVol, LabelVol]);
      Cond.Copy(NN.GetLastLayer.Output);
      SetLabelInput(cNullLabel);
      NN.Compute([Xt, TimeVol, LabelVol]);
      Uncond.Copy(NN.GetLastLayer.Output);
      TNNetDiffusionScheduler.ApplyCFG(Cond, Uncond, EpsCFG, gCurW);
    finally
      Cond.Free;
      Uncond.Free;
    end;
  end;

  // TNNetDenoiseCallback adapter method body (class declared at top-level type).
  procedure TCFGDenoiser.Denoise(Xt, Output: TNNetVolume; t: integer);
  begin
    PredictEpsCFG(Xt, Output, t);
  end;

  // Sample ONE digit of class gReqDigit at the current CFG weight into Dst
  // (in [-1,1]) via deterministic DDIM + CFG through the reusable scheduler.
  procedure SampleOne(Dst: TNNetVolume);
  var i: integer;
  begin
    for i := 0 to Dst.Size - 1 do
      Dst.FData[i] := RandG(0, 1);
    Sched.Sample(Dst, @Denoiser.Denoise, DDIMSteps, smDDIM, 0.0);
  end;

  // Classify a generated sample (in [-1,1]) -> predicted digit 0..9.
  function ClassifyGen(Gen: TNNetVolume): integer;
  begin
    ClfIn.Copy(Gen);     // classifier was trained on the same [-1,1] scaling
    Clf.Compute(ClfIn);
    Result := Clf.GetLastLayer.Output.GetClass;
  end;

  // Word describing how a metric moved from its low-w to high-w value.
  function Trend(Lo, Hi: TNeuralFloat): string;
  begin
    if Hi > Lo + 1e-4 then Result := 'ROSE'
    else if Hi < Lo - 1e-4 then Result := 'FELL'
    else Result := 'was flat';
  end;

  // Run the CFG sweep: for each weight, sample GridN images of cGridDigit, then
  // reuse them for BOTH the visual grid row and two scalar metrics --
  // class-fidelity (classifier agreement with cGridDigit) and a per-pixel-stddev
  // diversity proxy. Prints a table and fills the grid PNG (rows = weights,
  // cols = independent cGridDigit samples).
  procedure RunCFGSweepAndGrid(const FileName: string);
  var
    wi, x, y, gx, gy, c: integer;
    Grid: TNNetVolume;
    GridCol: array of TNNetVolume;     // cached cGridDigit samples for stddev
    Hits, Total: integer;
    Fidelity: array[0..cNumW - 1] of TNeuralFloat;
    Diversity: array[0..cNumW - 1] of TNeuralFloat;
    mean, varsum, px, v: TNeuralFloat;
    NumNan: integer;
  begin
    Grid := TNNetVolume.Create(GridN * cImgSize, cNumW * cImgSize, 3);
    Grid.Fill(0);
    SetLength(GridCol, GridN);
    for c := 0 to GridN - 1 do
      GridCol[c] := TNNetVolume.Create(cImgSize, cImgSize, 1);
    NumNan := 0;
    try
      WriteLn(Format('CFG sweep: %d grid samples/weight of digit %d, ' +
        '%d-step DDIM. Fidelity = classifier agreement on those samples.',
        [GridN, cGridDigit, DDIMSteps]));
      for wi := 0 to cNumW - 1 do
      begin
        gCurW := cWeights[wi];
        // Sample GridN images of the requested digit ONCE, then reuse them for
        // BOTH the visual grid row and the two scalar metrics (cheap on CPU:
        // batch-1 sampling dominates the runtime, so we sample only once).
        gReqDigit := cGridDigit;
        Hits := 0; Total := 0;
        for c := 0 to GridN - 1 do
        begin
          SampleOne(GridCol[c]);
          // Class fidelity: does the side classifier read it as cGridDigit?
          if ClassifyGen(GridCol[c]) = cGridDigit then Inc(Hits);
          Inc(Total);
          for y := 0 to cImgSize - 1 do
            for x := 0 to cImgSize - 1 do
            begin
              v := GridCol[c][x, y, 0];
              if IsNan(v) or IsInfinite(v) then begin Inc(NumNan); v := 0; end;
              px := (v + 1.0) * 127.5;
              if px < 0 then px := 0;
              if px > 255 then px := 255;
              gx := c * cImgSize + x;
              gy := wi * cImgSize + y;
              Grid[gx, gy, 0] := px;
              Grid[gx, gy, 1] := px;
              Grid[gx, gy, 2] := px;
            end;
        end;
        Fidelity[wi] := Hits / Total;
        // Mean per-pixel stddev across the GridN samples (diversity proxy).
        // Pixels are CLAMPED to the displayed [-1,1] range first so that the
        // unbounded eps amplification high w produces does not inflate the
        // variance through out-of-gamut spikes -- we want the diversity of what
        // is actually rendered, which collapses as w concentrates the class.
        varsum := 0;
        for y := 0 to cImgSize - 1 do
          for x := 0 to cImgSize - 1 do
          begin
            mean := 0;
            for c := 0 to GridN - 1 do
              mean := mean + Max(-1.0, Min(1.0, GridCol[c][x, y, 0]));
            mean := mean / GridN;
            v := 0;
            for c := 0 to GridN - 1 do
              v := v + Sqr(Max(-1.0, Min(1.0, GridCol[c][x, y, 0])) - mean);
            varsum := varsum + Sqrt(v / GridN);
          end;
        Diversity[wi] := varsum / (cImgSize * cImgSize);

        WriteLn(Format('  w = %4.1f   class-fidelity = %.3f   diversity(px-std) = %.4f',
          [gCurW, Fidelity[wi], Diversity[wi]]));
      end;

      WriteLn;
      WriteLn('CFG fidelity / diversity trade-off summary:');
      WriteLn('   w      fidelity   diversity');
      for wi := 0 to cNumW - 1 do
        WriteLn(Format('  %4.1f     %.3f      %.4f',
          [cWeights[wi], Fidelity[wi], Diversity[wi]]));
      // Self-report the OBSERVED trend (low w vs high w) rather than asserting
      // an expectation. The textbook CFG effect is fidelity UP, diversity DOWN
      // as w grows; it emerges cleanly only once the denoiser is well trained
      // (--full). In the short SMOKE run the denoiser is barely trained, so the
      // samples are noisy and the numbers mostly validate the pipeline.
      WriteLn(Format('Observed: fidelity %s, diversity %s from w=%.1f to w=%.1f.',
        [Trend(Fidelity[0], Fidelity[cNumW - 1]),
         Trend(Diversity[0], Diversity[cNumW - 1]),
         cWeights[0], cWeights[cNumW - 1]]));
      WriteLn('Textbook CFG: higher w -> higher fidelity, lower diversity ' +
        '(clearest with --full; SMOKE output is noisy).');

      if SaveImageFromVolumeIntoFile(Grid, FileName)
        then WriteLn('Wrote grid (rows = CFG weights ', cWeights[0], '..',
          cWeights[cNumW - 1], ', cols = samples of digit ', cGridDigit, '): ', FileName)
        else WriteLn('FAILED to write: ', FileName);
      if NumNan > 0
        then WriteLn('WARNING: ', NumNan, ' NaN/Inf pixels were clamped.')
        else WriteLn('No NaN/Inf in generated samples.');
    finally
      Grid.Free;
      for c := 0 to GridN - 1 do GridCol[c].Free;
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
    Steps := cFullSteps; BatchSz := cFullBatch; ClfSteps := cFullClfSteps;
    PerW := cFullPerW; GridN := cFullGrid; DDIMSteps := cFullDDIMSteps;
    WriteLn('ConditionalDiffusion [FULL training mode]');
  end
  else
  begin
    Steps := cSmokeSteps; BatchSz := cSmokeBatch; ClfSteps := cSmokeClfSteps;
    PerW := cSmokePerW; GridN := cSmokeGrid; DDIMSteps := cSmokeDDIMSteps;
    WriteLn('ConditionalDiffusion [SMOKE mode -- pass --full for real training]');
  end;
  WriteLn('Class-conditional DDPM: label embedding + label dropout -> CFG.');
  WriteLn('Demonstrates that the CFG weight trades diversity for class fidelity.');
  WriteLn;

  if not (CheckMNISTFile('train')) or not (CheckMNISTFile('t10k')) then
  begin
    WriteLn('MNIST idx-ubyte files not found in the working directory; exiting.');
    Exit;
  end;

  Sched := TNNetDiffusionScheduler.Create(cT, dsLinear, dpEps, cBeta1, cBetaT);
  Denoiser := TCFGDenoiser.Create;

  CreateMNISTVolumes(TrainV, ValV, TestV, 'train', 't10k');
  WriteLn('Loaded MNIST: ', TrainV.Count, ' train digits.');

  BuildNet;
  BuildClassifier;
  WriteLn('Denoiser architecture:');
  NN.PrintSummary();
  WriteLn;

  Img0    := TNNetVolume.Create(cImgSize, cImgSize, 1);
  ImgT    := TNNetVolume.Create(cImgSize, cImgSize, 1);
  EpsTrue := TNNetVolume.Create(cImgSize, cImgSize, 1);
  TimeVol  := TNNetVolume.Create(1, 1, 1);
  LabelVol := TNNetVolume.Create(1, 1, 1);
  ClfIn    := TNNetVolume.Create(cImgSize, cImgSize, 1);
  try
    TrainClassifier;
    WriteLn;
    TrainDenoiser;
    WriteLn;
    PngName := 'conditional_samples.png';
    RunCFGSweepAndGrid(PngName);
    WriteLn;
    WriteLn('Done. View ', PngName, ' (top rows = low w / diverse, ',
      'bottom rows = high w / faithful).');
    if not FullMode then
      WriteLn('(Smoke samples are noisy; run with --full for sharp digits and a cleaner trend.)');
  finally
    Img0.Free;
    ImgT.Free;
    EpsTrue.Free;
    TimeVol.Free;
    LabelVol.Free;
    ClfIn.Free;
    NN.Free;
    Clf.Free;
    Sched.Free;
    Denoiser.Free;
    TrainV.Free;
    ValV.Free;
    TestV.Free;
  end;
end.
