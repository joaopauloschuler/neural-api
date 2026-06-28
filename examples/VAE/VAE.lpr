program VAE;
(*
VAE: a CONTINUOUS-latent Variational Autoencoder on MNIST -- the Gaussian /
KL companion to the DISCRETE examples/VQVAE (which uses a vector-quantized
codebook bottleneck). Here the bottleneck is a diagonal Gaussian posterior
q(z|x) = N(mu, sigma^2) and we SAMPLE z with the reparameterization trick,
using the new repository layer TNNetGaussianReparameterize so the sampling is
no longer hand-rolled.

WHAT A VAE IS (Kingma & Welling 2014, "Auto-Encoding Variational Bayes",
https://arxiv.org/abs/1312.6114). An encoder maps an image to TWO vectors of
the same latent dimension: the posterior mean mu and the log-variance log_var
(packed together on the depth axis as mu | log_var). We then draw a latent
sample with the reparameterization trick
    z = mu + sigma * eps,   sigma = exp(0.5*log_var),   eps ~ N(0,1)
which keeps the draw differentiable w.r.t. mu and log_var. A decoder
reconstructs the image from z. Training minimizes
    loss = reconstruction-MSE  +  beta * KL( q(z|x) || N(0,1) )
where the KL term pulls the posterior toward a standard-normal prior so that,
at generation time, sampling z ~ N(0,1) and decoding produces plausible digits.

THE NEW LAYER (TNNetGaussianReparameterize). It splits its input depth in half
(mu | log_var) and emits z = mu + sigma*eps. eps is drawn ONCE per forward and
FROZEN, then REUSED in the matching backward (the standard fixed-noise pattern,
like TNNetGumbelSoftmax / TNNetDropout). Its backward routes the incoming
gradient g = dL/dz to mu (dz/dmu = 1) and to log_var (dz/dlog_var =
0.5*sigma*eps). It contributes ONLY the reconstruction (reparameterization)
gradient; the KL term is a SEPARATE penalty (companion head
TNNetVAEKLDivergence, whose analytic gradient is dKL/dmu = mu,
dKL/dlog_var = 0.5*(exp(log_var)-1)). This example keeps the composition
explicit: the reconstruction gradient flows through the reparameterize layer,
and the KL gradient is added directly onto the (mu|log_var) tensor's error --
the two gradients SUM at that fork, exactly as a two-headed VAE would wire it.

STAGES.
  Stage 1 (train): encoder -> (mu|log_var) -> reparameterize -> decoder, with
    reconstruction MSE + beta*KL. Report MSE, KL and elapsed; write a
    reconstruction PNG (top rows = originals, bottom rows = reconstructions).
  Stage 2 (generate): sample z ~ N(0,1), inject it where the reparameterize
    layer's output lives, run the decoder tail, and write a grid of brand-new
    digits sampled purely from the prior.

RUN MODES.
  default (SMOKE): a short run that finishes in well under a few minutes on one
    CPU -- enough to watch the MSE fall and the KL stay finite and write both
    PNGs without NaN.
  --full : more epochs for noticeably sharper output.

DATA. Standard MNIST idx-ubyte files in the working directory (same files every
MNIST example here uses). If absent the program prints a hint and exits cleanly.

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
  cImgSize  = 28;
  cGrid     = 7;     // latent grid side (28 -> /2 -> /2 = 7)
  cLatent   = 16;    // latent channels per grid cell (z depth)
  cC1       = 16;    // encoder/decoder channels at 28x28
  cC2       = 32;    // encoder/decoder channels at 14x14

  cBeta     = 0.0005;// KL weight (small: a from-scratch CPU run needs the
                     // reconstruction term to dominate early or the posterior
                     // collapses to the prior and every digit looks identical).
  cReconW   = 0.125; // reconstruction-gradient weight (= 1/8, > 1/cBatch)

  // SMOKE defaults.
  cEpochs   = 10;
  cBatch    = 32;
  cStepsEp  = 70;    // mini-batches per epoch
  cLR       = 0.001;

  cReconRows = 8;    // reconstruction PNG: cReconRows columns
  cGenGrid   = 8;    // generated PNG side (cGenGrid x cGenGrid digits)

var
  Epochs, StepsEp: integer;

  Net: TNNet;               // encoder -> muLogVar -> reparameterize -> decoder
  MuLogVar: TNNetLayer;     // (cGrid, cGrid, 2*cLatent) packed mu|log_var
  Reparam: TNNetGaussianReparameterize;
  ZLayer: TNNetLayer;       // reparameterize output z (cGrid,cGrid,cLatent)
  MuLogVarIdx, ZIdx, DecInIdx: integer;

  TrainV, ValV, TestV: TNNetVolumeList;
  Img, Pseudo: TNNetVolume;

// ----- MNIST: load digit Idx into Img rescaled to [-1,1]. ------------------
// Loader stores pixels as byte/64 - 2, so byte = (v+2)*64 and the [-1,1]
// target is byte/127.5 - 1 (same convention as examples/VQVAE).
procedure LoadDigit(Src: TNNetVolume);
var
  x, y: integer;
  v, b: TNeuralFloat;
begin
  for y := 0 to cImgSize - 1 do
    for x := 0 to cImgSize - 1 do
    begin
      v := Src[x, y, 0];
      b := (v + 2.0) * 64.0;
      if b < 0 then b := 0;
      if b > 255 then b := 255;
      Img[x, y, 0] := b / 127.5 - 1.0;
    end;
end;

// ----- Build encoder -> muLogVar -> reparameterize -> decoder. -------------
procedure BuildVAE;
begin
  Net := TNNet.Create();
  Net.AddLayer(TNNetInput.Create(cImgSize, cImgSize, 1));
  // Encoder: conv -> GroupNorm -> 2x downsample, twice (28 -> 14 -> 7), then a
  // pointwise linear projection to 2*cLatent channels = (mu | log_var) at every
  // 7x7 position. GroupNorm is essential for this small conv AE to train.
  Net.AddLayer(TNNetConvolutionReLU.Create(cC1, 3, 1, 1, 0));
  Net.AddLayer(TNNetGroupNorm.Create(4));
  Net.AddLayer(TNNetMaxPool.Create(2));                        // 28 -> 14
  Net.AddLayer(TNNetConvolutionReLU.Create(cC2, 3, 1, 1, 0));
  Net.AddLayer(TNNetGroupNorm.Create(4));
  Net.AddLayer(TNNetMaxPool.Create(2));                        // 14 -> 7
  Net.AddLayer(TNNetConvolutionReLU.Create(cC2, 3, 1, 1, 0));  // 7x7 refine
  // (mu | log_var): first cLatent channels = mu, next cLatent = log_var.
  MuLogVar := Net.AddLayer(TNNetPointwiseConvLinear.Create(2 * cLatent));
  MuLogVarIdx := Net.GetLastLayerIdx();

  // Gaussian reparameterization: z = mu + exp(0.5*log_var)*eps (depth halves).
  Reparam := TNNetGaussianReparameterize.Create();
  ZLayer := Net.AddLayer(Reparam);                             // z (7,7,cLatent)
  ZIdx := Net.GetLastLayerIdx();

  // Decoder: pointwise expand, two x2 upsamples (7 -> 14 -> 28), conv refine
  // (GroupNorm), linear conv head to a single grayscale channel.
  Net.AddLayer(TNNetPointwiseConvReLU.Create(cC2));            // decoder entry
  DecInIdx := Net.GetLastLayerIdx();
  Net.AddLayer(TNNetUpsample.Create());                        // 7 -> 14
  Net.AddLayer(TNNetConvolutionReLU.Create(cC2, 3, 1, 1, 0));
  Net.AddLayer(TNNetGroupNorm.Create(4));
  Net.AddLayer(TNNetUpsample.Create());                        // 14 -> 28
  Net.AddLayer(TNNetConvolutionReLU.Create(cC1, 3, 1, 1, 0));
  Net.AddLayer(TNNetConvolutionLinear.Create(1, 3, 1, 1, 0));  // reconstruction

  Net.SetLearningRate(cLR, 0.9);
  Net.SetL2Decay(0.0);
  Net.SetBatchUpdate(True);
end;

// Reconstruction MSE between Net output and the clean Img (current input).
function ReconMSE: TNeuralFloat;
var
  i: integer;
  d, s: TNeuralFloat;
  Pred: TNNetVolume;
begin
  Pred := Net.GetLastLayer.Output;
  s := 0;
  for i := 0 to Img.Size - 1 do
  begin
    d := Pred.FData[i] - Img.FData[i];
    s := s + d * d;
  end;
  Result := s / Img.Size;
end;

// Per-sample KL( N(mu,sigma^2) || N(0,1) ) summed over the latent grid, and --
// as a side effect -- OVERWRITE MuLogVar's OutputError with the PURE analytic
// KL gradient (beta * dKL). This is the SECOND, KL-only backward pass: the
// first (reconstruction) pass already pushed dL_recon all the way back through
// the encoder, so this pass adds ONLY the KL contribution on top, and the two
// gradients SUM at every encoder weight -- the whole VAE objective. The error
// is overwritten (not added) precisely so the encoder is not double-charged
// for the reconstruction term it already received.
//   dKL/dmu = mu ;  dKL/dlog_var = 0.5*(exp(log_var)-1).
function KLAndSeedGrad(Beta: TNeuralFloat): TNeuralFloat;
var
  gx, gy, d, Half: integer;
  mu, lv, kl: TNeuralFloat;
  V, E: TNNetVolume;
begin
  V := Net.Layers[MuLogVarIdx].Output;
  E := Net.Layers[MuLogVarIdx].OutputError;
  Half := cLatent;
  E.Fill(0);
  kl := 0;
  for gx := 0 to cGrid - 1 do
    for gy := 0 to cGrid - 1 do
      for d := 0 to Half - 1 do
      begin
        mu := V[gx, gy, d];
        lv := V[gx, gy, d + Half];
        kl := kl + (-0.5) * (1 + lv - mu * mu - Exp(lv));
        E[gx, gy, d]        := Beta * mu;
        E[gx, gy, d + Half] := Beta * 0.5 * (Exp(lv) - 1.0);
      end;
  Result := kl;
end;

// ----- Stage 1: train the VAE. ---------------------------------------------
procedure TrainVAE;
var
  Epoch, Step, B, i: integer;
  SumMSE, SumKL, kl: TNeuralFloat;
  Pred: TNNetVolume;
  StartTime, Elapsed: double;
begin
  WriteLn(Format('Stage 1: training encoder->reparameterize->decoder  (%d epochs x %d steps x batch %d).',
    [Epochs, StepsEp, cBatch]));
  StartTime := Now();
  for Epoch := 1 to Epochs do
  begin
    SumMSE := 0; SumKL := 0;
    for Step := 1 to StepsEp do
    begin
      Net.ClearDeltas();
      for B := 1 to cBatch do
      begin
        LoadDigit(TrainV[Random(TrainV.Count)]);
        Net.Compute(Img);                       // samples z via reparameterize
        Pred := Net.GetLastLayer.Output;
        SumMSE := SumMSE + ReconMSE;
        // Reconstruction pseudo-target (weighted MSE), as in examples/VQVAE:
        // the error fed to backprop is cReconW*(out - target).
        for i := 0 to Img.Size - 1 do
          Pseudo.FData[i] := Pred.FData[i] -
            cReconW * (Pred.FData[i] - Img.FData[i]);
        Net.Backpropagate(Pseudo);              // recon grad reaches mu|log_var
        // Second, KL-only backward from the (mu|log_var) fork through the
        // ENCODER (the reparameterize/decoder branch does not depend on KL).
        // Reset the per-pass backprop call counters first so MuLogVar's guard
        // (one departing branch = the reparameterize layer) is satisfied by
        // this fresh call; deltas accumulate into this batch (no ClearDeltas).
        kl := KLAndSeedGrad(cBeta);
        SumKL := SumKL + kl;
        Net.ResetBackpropCallCurrCnt();
        Net.Layers[MuLogVarIdx].Backpropagate();
      end;
      // Tame early gradient spikes on a from-scratch conv VAE.
      Net.ForceMaxAbsoluteDelta(0.02);
      Net.UpdateWeights();
    end;
    Elapsed := (Now() - StartTime) * 86400.0;
    WriteLn(Format('  epoch %2d/%2d  recon-MSE = %.5f  KL = %.4f  elapsed = %.1fs',
      [Epoch, Epochs, SumMSE / (StepsEp * cBatch), SumKL / (StepsEp * cBatch), Elapsed]));
  end;
end;

// Map a [-1,1] pixel to a 0..255 grayscale byte, clamping NaN/Inf to 0.
function PixToByte(v: TNeuralFloat; var NumNan: integer): TNeuralFloat;
begin
  if IsNan(v) or IsInfinite(v) then begin Inc(NumNan); v := 0; end;
  Result := (v + 1.0) * 127.5;
  if Result < 0 then Result := 0;
  if Result > 255 then Result := 255;
end;

// ----- Reconstruction PNG: top row originals, bottom row reconstructions. ---
procedure WriteReconPng(const FileName: string);
var
  Grid: TNNetVolume;
  c, x, y, gx, gy, NumNan, Idx: integer;
  pxO, pxR: TNeuralFloat;
  Pred: TNNetVolume;
begin
  Grid := TNNetVolume.Create(cReconRows * cImgSize, 2 * cImgSize, 3);
  Grid.Fill(0);
  NumNan := 0;
  // Deterministic reconstructions: z = mu (no sampling) for a clean PNG.
  Reparam.Enabled := false;
  try
    for c := 0 to cReconRows - 1 do
    begin
      Idx := Random(TestV.Count);
      LoadDigit(TestV[Idx]);
      Net.Compute(Img);
      Pred := Net.GetLastLayer.Output;
      for y := 0 to cImgSize - 1 do
        for x := 0 to cImgSize - 1 do
        begin
          pxO := PixToByte(Img[x, y, 0], NumNan);
          pxR := PixToByte(Pred[x, y, 0], NumNan);
          gx := c * cImgSize + x;
          gy := y;                          // top: original
          Grid[gx, gy, 0] := pxO; Grid[gx, gy, 1] := pxO; Grid[gx, gy, 2] := pxO;
          gy := cImgSize + y;               // bottom: reconstruction
          Grid[gx, gy, 0] := pxR; Grid[gx, gy, 1] := pxR; Grid[gx, gy, 2] := pxR;
        end;
    end;
    if SaveImageFromVolumeIntoFile(Grid, FileName)
      then WriteLn('Wrote reconstruction grid (top=original, bottom=reconstruction): ', FileName)
      else WriteLn('FAILED to write: ', FileName);
    if NumNan > 0 then WriteLn('WARNING: ', NumNan, ' NaN/Inf pixels clamped.');
  finally
    Reparam.Enabled := true;
    Grid.Free;
  end;
end;

// Decode a latent z (cGrid,cGrid,cLatent) by writing it into the reparameterize
// layer's output and running the decoder tail (DecIn onward).
procedure DecodeLatent(Z: TNNetVolume; Dst: TNNetVolume);
var
  i: integer;
  ZVol: TNNetVolume;
begin
  ZVol := Net.Layers[ZIdx].Output;
  ZVol.Copy(Z);
  for i := DecInIdx to Net.CountLayers - 1 do
    Net.Layers[i].Compute();
  Dst.Copy(Net.GetLastLayer.Output);
end;

// ----- Generated PNG: sample z ~ N(0,1) from the prior and decode. ----------
procedure WriteGeneratedPng(const FileName: string);
var
  Grid, One, Z: TNNetVolume;
  r, c, x, y, gx, gy, d, NumNan: integer;
  px: TNeuralFloat;
begin
  Grid := TNNetVolume.Create(cGenGrid * cImgSize, cGenGrid * cImgSize, 3);
  One  := TNNetVolume.Create(cImgSize, cImgSize, 1);
  Z    := TNNetVolume.Create(cGrid, cGrid, cLatent);
  Grid.Fill(0);
  NumNan := 0;
  try
    WriteLn(Format('Sampling a %dx%d grid of generated digits (prior z~N(0,1) -> decode)...',
      [cGenGrid, cGenGrid]));
    for r := 0 to cGenGrid - 1 do
      for c := 0 to cGenGrid - 1 do
      begin
        // Draw z from the standard-normal prior the KL term trained toward.
        for gx := 0 to cGrid - 1 do
          for gy := 0 to cGrid - 1 do
            for d := 0 to cLatent - 1 do
              Z[gx, gy, d] := Z.RandomGaussianValue();
        DecodeLatent(Z, One);
        for y := 0 to cImgSize - 1 do
          for x := 0 to cImgSize - 1 do
          begin
            px := PixToByte(One[x, y, 0], NumNan);
            gx := c * cImgSize + x;
            gy := r * cImgSize + y;
            Grid[gx, gy, 0] := px; Grid[gx, gy, 1] := px; Grid[gx, gy, 2] := px;
          end;
      end;
    if SaveImageFromVolumeIntoFile(Grid, FileName)
      then WriteLn('Wrote generated-samples grid: ', FileName)
      else WriteLn('FAILED to write: ', FileName);
    if NumNan > 0
      then WriteLn('WARNING: ', NumNan, ' NaN/Inf pixels clamped.')
      else WriteLn('No NaN/Inf in generated samples.');
  finally
    Grid.Free;
    One.Free;
    Z.Free;
  end;
end;

var
  i: integer;
  FullMode: boolean;
begin
  SetExceptionMask([exInvalidOp, exDenormalized, exZeroDivide,
                    exOverflow, exUnderflow, exPrecision]);
  RandSeed := 2026;

  FullMode := false;
  for i := 1 to ParamCount do
    if (ParamStr(i) = '--full') then FullMode := true;

  if FullMode then
  begin
    Epochs := 20; StepsEp := 150;
    WriteLn('VAE [FULL training mode]');
  end
  else
  begin
    Epochs := cEpochs; StepsEp := cStepsEp;
    WriteLn('VAE [SMOKE mode -- pass --full for sharper output]');
  end;
  WriteLn('Continuous-latent Variational Autoencoder (Kingma & Welling 2014).');
  WriteLn(Format('  latent grid = %dx%d   latent channels = %d   beta = %.5f',
    [cGrid, cGrid, cLatent, cBeta]));
  WriteLn;

  if not (CheckMNISTFile('train')) or not (CheckMNISTFile('t10k')) then
  begin
    WriteLn('MNIST idx-ubyte files not found in the working directory; exiting.');
    WriteLn('Copy train-images.idx3-ubyte etc. here (see any MNIST example).');
    Exit;
  end;

  CreateMNISTVolumes(TrainV, ValV, TestV, 'train', 't10k');
  WriteLn('Loaded MNIST: ', TrainV.Count, ' train / ', TestV.Count, ' test digits.');

  BuildVAE;
  WriteLn('VAE architecture:');
  Net.PrintSummary();
  WriteLn;

  Img    := TNNetVolume.Create(cImgSize, cImgSize, 1);
  Pseudo := TNNetVolume.Create(cImgSize, cImgSize, 1);
  try
    TrainVAE;
    WriteLn;
    WriteReconPng('vae_reconstructions.png');
    WriteLn;
    WriteGeneratedPng('vae_generated.png');
    WriteLn;
    WriteLn('Done. View vae_reconstructions.png and vae_generated.png.');
    if not FullMode then
      WriteLn('(Smoke output is rough; run with --full for sharper digits.)');
  finally
    Img.Free;
    Pseudo.Free;
    Net.Free;
    TrainV.Free;
    ValV.Free;
    TestV.Free;
  end;
end.
