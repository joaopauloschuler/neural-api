program VQVAE;
(*
VQVAE: a FULL discrete-image VQ-VAE pipeline on MNIST -- the generative
companion to the two VQ DIAGNOSTIC examples (examples/VQCodebookUsage and
examples/VQCodebookCollapse, which only probe codebook health on synthetic
blobs). Here the same vector-quantization bottleneck is wired into a real
convolutional encoder/decoder that reconstructs 28x28 digits, and then a small
autoregressive transformer PRIOR is fitted over the discrete code grids so we
can SAMPLE brand-new index grids and DECODE them into freshly generated digits.

WHAT A VQ-VAE IS (van den Oord et al. 2017, "Neural Discrete Representation
Learning", https://arxiv.org/abs/1711.00937). An encoder maps an image to a
low-resolution grid of continuous feature vectors z_e (here 7x7, each of
dimension cEmb). A learnable CODEBOOK of cK vectors quantizes every position to
its nearest code z_q (the discrete bottleneck). A decoder reconstructs the
image from z_q. Training minimizes reconstruction MSE; the gradient flows
through the non-differentiable argmin via the STRAIGHT-THROUGH estimator, and
two auxiliary terms (the commitment loss and the codebook loss) keep encoder
outputs and codebook vectors close. All of that quantization math + the
commitment/codebook gradients + the straight-through pass already live in the
repository layer TNNetVectorQuantizer; this example just builds an encoder and
decoder around it. The new public accessor LVQ.ChosenCodeIndex(X,Y) reads back
the discrete argmin token at each grid cell -- that is what turns the image into
a 7x7 grid of integers in 0..cK-1, i.e. a short sequence of 49 DISCRETE tokens.

THE TWO STAGES.
  Stage 1 (representation): train encoder -> TNNetVectorQuantizer -> decoder to
    reconstruct MNIST. Report reconstruction MSE + codebook usage, and write a
    reconstruction grid PNG (top rows = originals, bottom rows = reconstructions).
  Stage 2 (prior + generation): FREEZE the encoder, run it over a batch of
    digits to harvest their 7x7 code grids, then fit a tiny CAUSAL transformer
    language model over those 49-token sequences (vocab = cK). Once trained we
    AUTOREGRESSIVELY sample new 49-token grids, look up each token's codebook
    vector to rebuild a 7x7xcEmb z_q tensor, and DECODE it through the frozen
    decoder to synthesise a digit. Write a generated-samples PNG.

This mirrors the classic two-stage VQ-VAE recipe: a discrete autoencoder first,
an autoregressive prior over its codes second (PixelCNN in the paper; a small
GPT-style decoder here, reusing TNNet.AddTransformerEncoderBlock with
CausalMask=true exactly like examples/TinyGPT).

RUN MODES.
  default (SMOKE): a short representation run + a short prior run; finishes well
    under a few minutes on one CPU. Enough to watch the reconstruction MSE fall,
    confirm the codebook does not collapse, and write both PNGs without NaN.
  --full : more epochs / bigger harvest for noticeably sharper output.

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
  cGrid     = 7;     // latent grid side (28 -> /2 -> /2 = 7); 7*7 = 49 tokens
  cTokens   = cGrid * cGrid;
  cEmb      = 16;    // codebook vector dimension (encoder output depth)
  cK        = 32;    // codebook size (vocabulary of the discrete tokens)
  // Commitment cost. The paper uses 0.25, but with our small per-sample
  // reconstruction gradient a large commitment term over-pulls the encoder onto
  // a single code (collapse), so we use a SMALLER beta and also weight the
  // reconstruction gradient more heavily (cReconW) than 1/batch.
  cBeta     = 0.05;
  cReconW   = 0.125; // reconstruction-gradient weight (= 1/8, > 1/cAEBatch)

  cC1       = 16;    // encoder channels at 28x28 / decoder channels at 28x28
  cC2       = 32;    // encoder channels at 14x14 / decoder channels at 14x14

  // Prior (tiny GPT over the 49-token code grid).
  cDModel   = 48;
  cHeads    = 3;
  cDFF      = 96;
  cBlocks   = 2;

  // SMOKE defaults.
  cAEEpochs    = 10;
  cAEBatch     = 32;
  cAEStepsEp   = 70;   // mini-batches per epoch
  cAELR        = 0.001;

  cHarvest     = 2000; // # digits whose code grids train the prior
  cPriorEpochs = 6;
  cPriorBatch  = 32;
  cPriorStepsEp= 80;
  cPriorLR     = 0.004;

  cReconRows   = 8;    // reconstruction PNG: cReconRows columns
  cGenGrid     = 8;    // generated PNG side (cGenGrid x cGenGrid digits)

var
  // --full scales these up.
  AEEpochs, AEStepsEp, Harvest, PriorEpochs, PriorStepsEp: integer;

  AE: TNNet;            // encoder -> VQ -> decoder
  LVQ: TNNetVectorQuantizer;
  EncOut, DecIn: TNNetLayer; // encoder output (z_e) and decoder input layers
  VQIdx, DecInIdx: integer;

  Prior: TNNet;         // autoregressive transformer over the code grid
  TrainV, ValV, TestV: TNNetVolumeList;

  Img, Pseudo: TNNetVolume; // 28x28x1 working image + MSE pseudo-target

// ----- MNIST: load digit Idx into Img rescaled to [-1,1]; returns nothing. ---
// The loader stores pixels as byte/64 - 2, so byte = (v+2)*64 and the [-1,1]
// target is byte/127.5 - 1 (identical convention to examples/DiffusionMNIST).
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

// ----- Build the discrete autoencoder: encoder -> VQ -> decoder. -----------
procedure BuildAutoEncoder;
begin
  AE := TNNet.Create();
  AE.AddLayer(TNNetInput.Create(cImgSize, cImgSize, 1));
  // Encoder: conv -> GroupNorm -> 2x downsample, twice (28 -> 14 -> 7), then a
  // pointwise linear projection to the cEmb codebook dimension at every 7x7
  // position. The GroupNorm layers are ESSENTIAL: without normalization this
  // small conv autoencoder fails to learn and the decoder just predicts the
  // dataset mean (constant ~0.32 MSE on [-1,1] MNIST).
  AE.AddLayer(TNNetConvolutionReLU.Create(cC1, 3, 1, 1, 0));
  AE.AddLayer(TNNetGroupNorm.Create(4));
  AE.AddLayer(TNNetMaxPool.Create(2));                       // 28 -> 14
  AE.AddLayer(TNNetConvolutionReLU.Create(cC2, 3, 1, 1, 0));
  AE.AddLayer(TNNetGroupNorm.Create(4));
  AE.AddLayer(TNNetMaxPool.Create(2));                       // 14 -> 7
  AE.AddLayer(TNNetConvolutionReLU.Create(cC2, 3, 1, 1, 0)); // 7x7 refine
  EncOut := AE.AddLayer(TNNetPointwiseConvLinear.Create(cEmb)); // z_e (7,7,cEmb)

  // Discrete bottleneck: nearest-codebook quantization with straight-through
  // gradient + commitment/codebook losses (all inside TNNetVectorQuantizer).
  LVQ := TNNetVectorQuantizer.Create(cK, cBeta);
  AE.AddLayer(LVQ);
  VQIdx := AE.GetLastLayerIdx();

  // Decoder: pointwise expand, two x2 upsamples (7 -> 14 -> 28), conv refine
  // (GroupNorm again), linear conv head to a single grayscale channel.
  DecIn := AE.AddLayer(TNNetPointwiseConvReLU.Create(cC2)); // decoder entry
  DecInIdx := AE.GetLastLayerIdx();
  AE.AddLayer(TNNetUpsample.Create());                       // 7 -> 14
  AE.AddLayer(TNNetConvolutionReLU.Create(cC2, 3, 1, 1, 0));
  AE.AddLayer(TNNetGroupNorm.Create(4));
  AE.AddLayer(TNNetUpsample.Create());                       // 14 -> 28
  AE.AddLayer(TNNetConvolutionReLU.Create(cC1, 3, 1, 1, 0));
  AE.AddLayer(TNNetConvolutionLinear.Create(1, 3, 1, 1, 0)); // reconstruction

  AE.SetLearningRate(cAELR, 0.9);
  AE.SetL2Decay(0.0);
  AE.SetBatchUpdate(True);
end;

// Revive codebook entries that won ZERO argmins last epoch by resetting them to
// a live encoder output z_e (captured during the epoch) plus small jitter, so
// they re-enter competition near data the encoder actually produces. This is
// the practical antidote to VQ-VAE codebook collapse; without it a short CPU
// run pins the codebook to a single code and every reconstruction/sample is the
// same blob. ZeBank holds recent z_e vectors harvested from EncOut.
var
  ZeBank: array of array of TNeuralFloat; // recent encoder-output vectors
  ZeBankCount: integer;

procedure CaptureZe;
var
  EncVol: TNNetVolume;
  gx, gy, d, slot: integer;
begin
  EncVol := EncOut.Output; // (cGrid, cGrid, cEmb) z_e
  // Sample a couple of positions per call into a ring buffer.
  gx := Random(cGrid); gy := Random(cGrid);
  slot := ZeBankCount mod Length(ZeBank);
  for d := 0 to cEmb - 1 do
    ZeBank[slot][d] := EncVol[gx, gy, d];
  Inc(ZeBankCount);
end;

procedure ReviveDeadCodes(out NumRevived: integer);
var
  code, d, src: integer;
  W: TNNetVolume;
begin
  NumRevived := 0;
  if ZeBankCount = 0 then Exit;
  for code := 0 to cK - 1 do
    if LVQ.CodebookUsageCount(code) = 0 then
    begin
      src := Random(Min(ZeBankCount, Length(ZeBank)));
      W := AE.Layers[VQIdx].Neurons[code].Weights;
      for d := 0 to cEmb - 1 do
        W.Raw[d] := ZeBank[src][d] + 0.05 * (Random - 0.5);
      Inc(NumRevived);
    end;
  if NumRevived > 0 then AE.Layers[VQIdx].FlushWeightCache();
end;

// Reconstruction MSE between AE output and the clean Img (current input).
function ReconMSE: TNeuralFloat;
var
  i: integer;
  d, s: TNeuralFloat;
  Pred: TNNetVolume;
begin
  Pred := AE.GetLastLayer.Output;
  s := 0;
  for i := 0 to Img.Size - 1 do
  begin
    d := Pred.FData[i] - Img.FData[i];
    s := s + d * d;
  end;
  Result := s / Img.Size;
end;

// ----- Stage 1: train the discrete autoencoder. ----------------------------
procedure TrainAutoEncoder;
var
  Epoch, Step, B, i, Idx, Active, Revived: integer;
  SumLoss: TNeuralFloat;
  Pred: TNNetVolume;
  StartTime, Elapsed: double;
begin
  WriteLn(Format('Stage 1: training encoder->VQ->decoder  (%d epochs x %d steps x batch %d).',
    [AEEpochs, AEStepsEp, cAEBatch]));
  SetLength(ZeBank, 256);
  for i := 0 to Length(ZeBank) - 1 do SetLength(ZeBank[i], cEmb);
  ZeBankCount := 0;
  StartTime := Now();
  for Epoch := 1 to AEEpochs do
  begin
    SumLoss := 0;
    LVQ.ResetCodebookUsage();
    for Step := 1 to AEStepsEp do
    begin
      AE.ClearDeltas();
      for B := 1 to cAEBatch do
      begin
        Idx := Random(TrainV.Count);
        LoadDigit(TrainV[Idx]);
        AE.Compute(Img);
        Pred := AE.GetLastLayer.Output;
        SumLoss := SumLoss + ReconMSE;
        CaptureZe; // stash a live z_e for dead-code revival
        // MSE pseudo-target with reconstruction weight cReconW (> 1/batch): the
        // error fed to backprop is cReconW*(out - target). A bigger cReconW
        // makes the reconstruction gradient dominate the (per-sample) VQ
        // commitment term, which is what keeps the codebook from collapsing
        // onto a single code; see cBeta/cReconW comments.
        for i := 0 to Img.Size - 1 do
          Pseudo.FData[i] := Pred.FData[i] -
            cReconW * (Pred.FData[i] - Img.FData[i]);
        AE.Backpropagate(Pseudo);
      end;
      // Tame early gradient spikes (a from-scratch VQ-VAE produces large
      // deltas before the codebook settles; without this the first epoch can
      // overflow to +Inf MSE).
      AE.ForceMaxAbsoluteDelta(0.02);
      AE.UpdateWeights();
    end;
    Active := LVQ.ActiveCodeCount();
    // Revive codes that won nothing this epoch (anti-collapse).
    ReviveDeadCodes(Revived);
    Elapsed := (Now() - StartTime) * 86400.0;
    WriteLn(Format('  epoch %2d/%2d  recon-MSE = %.5f  active codes = %2d/%2d  revived = %2d  elapsed = %.1fs',
      [Epoch, AEEpochs, SumLoss / (AEStepsEp * cAEBatch), Active, cK, Revived, Elapsed]));
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

// ----- Reconstruction PNG: pairs of (original, reconstruction) columns. -----
procedure WriteReconPng(const FileName: string);
var
  Grid: TNNetVolume;
  c, x, y, gx, gy, NumNan, Idx: integer;
  pxO, pxR: TNeuralFloat;
  Pred: TNNetVolume;
begin
  // 2 rows of cReconRows columns: row 0 originals, row 1 reconstructions.
  Grid := TNNetVolume.Create(cReconRows * cImgSize, 2 * cImgSize, 3);
  Grid.Fill(0);
  NumNan := 0;
  try
    for c := 0 to cReconRows - 1 do
    begin
      Idx := Random(TestV.Count);
      LoadDigit(TestV[Idx]);
      AE.Compute(Img);
      Pred := AE.GetLastLayer.Output;
      for y := 0 to cImgSize - 1 do
        for x := 0 to cImgSize - 1 do
        begin
          pxO := PixToByte(Img[x, y, 0], NumNan);
          pxR := PixToByte(Pred[x, y, 0], NumNan);
          gx := c * cImgSize + x;
          gy := y;                          // top row: original
          Grid[gx, gy, 0] := pxO; Grid[gx, gy, 1] := pxO; Grid[gx, gy, 2] := pxO;
          gy := cImgSize + y;               // bottom row: reconstruction
          Grid[gx, gy, 0] := pxR; Grid[gx, gy, 1] := pxR; Grid[gx, gy, 2] := pxR;
        end;
    end;
    if SaveImageFromVolumeIntoFile(Grid, FileName)
      then WriteLn('Wrote reconstruction grid (top=original, bottom=reconstruction): ', FileName)
      else WriteLn('FAILED to write: ', FileName);
    if NumNan > 0 then WriteLn('WARNING: ', NumNan, ' NaN/Inf pixels clamped.');
  finally
    Grid.Free;
  end;
end;

// ----- Encode one image to its 7x7 discrete code grid (49 ints). -----------
procedure EncodeCodeGrid(out Codes: array of integer);
var
  gx, gy: integer;
begin
  AE.Compute(Img); // forward through encoder+VQ caches the argmin per position
  for gx := 0 to cGrid - 1 do
    for gy := 0 to cGrid - 1 do
      Codes[gx * cGrid + gy] := LVQ.ChosenCodeIndex(gx, gy);
end;

// ===========================================================================
//                       Stage 2: autoregressive prior
// ===========================================================================

var
  // Harvested code grids: HarvestCodes[sample][token] in 0..cK-1.
  HarvestCodes: array of array of integer;

// Build a tiny causal transformer LM over the cTokens-long code sequence.
// Input is one-hot tokens (cTokens, 1, cK); the head predicts the next-token
// distribution over the whole sequence (FullConnect LM head, like TinyGPT).
procedure BuildPrior;
var
  i: integer;
begin
  Prior := TNNet.Create();
  Prior.AddLayer([
    TNNetInput.Create(cTokens, 1, cK),
    TNNetPointwiseConvLinear.Create(cDModel),   // token projection
    TNNetAddPositionalEmbedding.Create(10000)
  ]);
  for i := 1 to cBlocks do
    Prior.AddTransformerEncoderBlock(
      {Heads=}cHeads, {d_ff=}cDFF,
      {PreNorm=}true, {CausalMask=}true,
      {UseRoPE=}false, {NormClass=}nil);
  Prior.AddLayer([
    TNNetPointwiseConvReLU.Create(cDModel),
    TNNetFullConnectReLU.Create(cDModel),
    TNNetFullConnectLinear.Create(cK),
    TNNetSoftMax.Create()
  ]);
  Prior.SetLearningRate(cPriorLR, 0.9);
  Prior.SetL2Decay(0.0);
  Prior.SetBatchUpdate(True);
end;

// Harvest code grids from the FROZEN encoder over Harvest random train digits.
procedure HarvestCodeGrids;
var
  n: integer;
  Codes: array[0..cTokens - 1] of integer;
  t: integer;
begin
  WriteLn(Format('Stage 2: harvesting %d code grids from the frozen encoder...', [Harvest]));
  SetLength(HarvestCodes, Harvest);
  for n := 0 to Harvest - 1 do
  begin
    LoadDigit(TrainV[Random(TrainV.Count)]);
    EncodeCodeGrid(Codes);
    SetLength(HarvestCodes[n], cTokens);
    for t := 0 to cTokens - 1 do
      HarvestCodes[n][t] := Codes[t];
  end;
end;

// Fill a (cTokens,1,cK) one-hot input from a code sequence shifted right by one
// (teacher forcing): position p sees token[p-1]; position 0 sees a zero vector
// (BOS). Returns nothing; the next-token target at the LAST position is set by
// the caller. We train ALL positions: target at position p is token[p].
// To reuse the single-distribution FullConnect head, we train one (prefix ->
// next token) pair per sample per step: pick a random split point.
procedure FillPriorInput(const Seq: array of integer; PrefixLen: integer;
  InVol: TNNetVolume);
var
  p, tok: integer;
begin
  InVol.Fill(0);
  // Place tokens 0..PrefixLen-1 at sequence positions 0..PrefixLen-1 (one-hot).
  for p := 0 to PrefixLen - 1 do
  begin
    tok := Seq[p];
    if (tok >= 0) and (tok < cK) then
      InVol[p, 0, tok] := 1.0;
  end;
end;

// Train the prior: each step draws a sample and a split, feeds the prefix, and
// trains the next-token softmax via a one-hot target on the LM head.
procedure TrainPrior;
var
  Epoch, Step, B, n, split, target: integer;
  SumLoss: TNeuralFloat;
  InVol, Tgt: TNNetVolume;
  Pred: TNNetVolume;
  StartTime, Elapsed: double;
  i: integer;
begin
  InVol := TNNetVolume.Create(cTokens, 1, cK);
  Tgt   := TNNetVolume.Create(1, 1, cK);
  WriteLn(Format('  training prior LM  (%d epochs x %d steps x batch %d).',
    [PriorEpochs, PriorStepsEp, cPriorBatch]));
  StartTime := Now();
  try
    for Epoch := 1 to PriorEpochs do
    begin
      SumLoss := 0;
      for Step := 1 to PriorStepsEp do
      begin
        Prior.ClearDeltas();
        for B := 1 to cPriorBatch do
        begin
          n := Random(Harvest);
          // Predict token at position 'split' from the prefix [0..split-1].
          split := Random(cTokens);                 // 0..cTokens-1
          target := HarvestCodes[n][split];
          FillPriorInput(HarvestCodes[n], split, InVol);
          Prior.Compute(InVol);
          Pred := Prior.GetLastLayer.Output;
          // Cross-entropy proxy via the stock softmax error: target one-hot.
          Tgt.Fill(0);
          Tgt[0, 0, target] := 1.0;
          // Accumulate -log p(target) for reporting.
          if Pred.FData[target] > 1e-9 then
            SumLoss := SumLoss - Ln(Pred.FData[target]);
          Prior.Backpropagate(Tgt);
        end;
        Prior.UpdateWeights();
      end;
      Elapsed := (Now() - StartTime) * 86400.0;
      WriteLn(Format('  epoch %2d/%2d  prior NLL = %.4f  elapsed = %.1fs',
        [Epoch, PriorEpochs, SumLoss / (PriorStepsEp * cPriorBatch), Elapsed]));
    end;
  finally
    InVol.Free;
    Tgt.Free;
  end;
  // Silence unused warning for i in some FPC configs.
  i := 0; if i <> 0 then WriteLn(i);
end;

// Sample a full 49-token grid autoregressively from the prior (temperature
// softmax draw, position by position).
procedure SamplePriorGrid(out Seq: array of integer; Temperature: TNeuralFloat);
var
  InVol: TNNetVolume;
  pos, k, target: integer;
  Pred: TNNetVolume;
  logits: array[0..cK - 1] of TNeuralFloat;
  sum, r, acc, mx: TNeuralFloat;
begin
  InVol := TNNetVolume.Create(cTokens, 1, cK);
  try
    for pos := 0 to cTokens - 1 do
    begin
      FillPriorInput(Seq, pos, InVol); // prefix = already-sampled tokens 0..pos-1
      Prior.Compute(InVol);
      Pred := Prior.GetLastLayer.Output; // softmax probabilities over cK
      // Temperature-reweight the probabilities (p^(1/T)) and renormalize.
      mx := 0;
      for k := 0 to cK - 1 do
        logits[k] := Power(Max(Pred.FData[k], 1e-12), 1.0 / Temperature);
      sum := 0;
      for k := 0 to cK - 1 do sum := sum + logits[k];
      if sum <= 0 then sum := 1;
      r := Random * sum;
      acc := 0;
      target := cK - 1;
      for k := 0 to cK - 1 do
      begin
        acc := acc + logits[k];
        if r <= acc then begin target := k; break; end;
      end;
      Seq[pos] := target;
      mx := mx; // no-op to keep mx referenced
    end;
  finally
    InVol.Free;
  end;
end;

// Decode a 49-token code grid into a 28x28 digit: write each token's codebook
// vector into the VQ output volume, then run the decoder tail from DecIn.
procedure DecodeCodeGrid(const Seq: array of integer; Dst: TNNetVolume);
var
  gx, gy, d, tok, i: integer;
  CodeW: TNNetVolume;
  VQVol: TNNetVolume;
  Pred: TNNetVolume;
begin
  VQVol := AE.Layers[VQIdx].Output;       // (cGrid, cGrid, cEmb) z_q tensor
  for gx := 0 to cGrid - 1 do
    for gy := 0 to cGrid - 1 do
    begin
      tok := Seq[gx * cGrid + gy];
      if (tok < 0) or (tok > cK - 1) then tok := 0;
      CodeW := AE.Layers[VQIdx].Neurons[tok].Weights; // codebook vector
      for d := 0 to cEmb - 1 do
        VQVol[gx, gy, d] := CodeW.Raw[d];
    end;
  // Run the decoder tail (DecIn onward) over the assembled z_q. Compute the
  // decoder sub-net by forwarding from DecIn's layer index to the end.
  for i := DecInIdx to AE.CountLayers - 1 do
    AE.Layers[i].Compute();
  Pred := AE.GetLastLayer.Output;
  Dst.Copy(Pred);
end;

// ----- Generated-samples PNG: cGenGrid x cGenGrid sampled digits. ----------
procedure WriteGeneratedPng(const FileName: string);
var
  Grid, One: TNNetVolume;
  r, c, x, y, gx, gy, NumNan: integer;
  Seq: array[0..cTokens - 1] of integer;
  px: TNeuralFloat;
begin
  Grid := TNNetVolume.Create(cGenGrid * cImgSize, cGenGrid * cImgSize, 3);
  One  := TNNetVolume.Create(cImgSize, cImgSize, 1);
  Grid.Fill(0);
  NumNan := 0;
  try
    WriteLn(Format('Sampling a %dx%d grid of generated digits (prior -> codes -> decode)...',
      [cGenGrid, cGenGrid]));
    for r := 0 to cGenGrid - 1 do
      for c := 0 to cGenGrid - 1 do
      begin
        SamplePriorGrid(Seq, 1.0);
        DecodeCodeGrid(Seq, One);
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
    AEEpochs := 16; AEStepsEp := 120; Harvest := 6000;
    PriorEpochs := 16; PriorStepsEp := 160;
    WriteLn('VQVAE [FULL training mode]');
  end
  else
  begin
    AEEpochs := cAEEpochs; AEStepsEp := cAEStepsEp; Harvest := cHarvest;
    PriorEpochs := cPriorEpochs; PriorStepsEp := cPriorStepsEp;
    WriteLn('VQVAE [SMOKE mode -- pass --full for sharper output]');
  end;
  WriteLn('Discrete-image VQ-VAE (van den Oord et al. 2017) + autoregressive prior.');
  WriteLn(Format('  latent grid = %dx%d = %d tokens   codebook K = %d   emb dim = %d',
    [cGrid, cGrid, cTokens, cK, cEmb]));
  WriteLn;

  if not (CheckMNISTFile('train')) or not (CheckMNISTFile('t10k')) then
  begin
    WriteLn('MNIST idx-ubyte files not found in the working directory; exiting.');
    WriteLn('Copy train-images.idx3-ubyte etc. here (see any MNIST example).');
    Exit;
  end;

  CreateMNISTVolumes(TrainV, ValV, TestV, 'train', 't10k');
  WriteLn('Loaded MNIST: ', TrainV.Count, ' train / ', TestV.Count, ' test digits.');

  BuildAutoEncoder;
  WriteLn('Autoencoder architecture:');
  AE.PrintSummary();
  WriteLn;

  Img    := TNNetVolume.Create(cImgSize, cImgSize, 1);
  Pseudo := TNNetVolume.Create(cImgSize, cImgSize, 1);
  try
    TrainAutoEncoder;
    WriteLn;
    WriteReconPng('vqvae_reconstructions.png');
    WriteLn;

    BuildPrior;
    WriteLn('Prior (causal transformer over code grid) architecture:');
    Prior.PrintSummary();
    WriteLn;
    HarvestCodeGrids;
    TrainPrior;
    WriteLn;
    WriteGeneratedPng('vqvae_generated.png');
    WriteLn;
    WriteLn('Done. View vqvae_reconstructions.png and vqvae_generated.png.');
    if not FullMode then
      WriteLn('(Smoke output is rough; run with --full for sharper digits.)');
  finally
    Img.Free;
    Pseudo.Free;
    AE.Free;
    if Assigned(Prior) then Prior.Free;
    TrainV.Free;
    ValV.Free;
    TestV.Free;
  end;
end.
