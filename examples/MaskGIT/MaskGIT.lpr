program MaskGIT;
(*
MaskGIT: NON-AUTOREGRESSIVE (parallel iterative) image generation over discrete
VQ codebook tokens, after Chang et al. 2022, "MaskGIT: Masked Generative Image
Transformer" (https://arxiv.org/abs/2202.04200). This is the masked-token
counterpart of examples/VQVAE: it reuses the SAME discrete VQ-VAE front end
(a tiny convolutional encoder -> TNNetVectorQuantizer -> decoder over MNIST that
turns every 28x28 digit into a 7x7 grid of 49 DISCRETE code indices), but it
replaces VQVAE's CAUSAL autoregressive prior + position-by-position sampling
with:

  1. a BIDIRECTIONAL transformer (AddTransformerEncoderBlock with
     CausalMask=FALSE -- the exact opposite of TinyGPT/VQVAE's causal prior),
  2. a MASKED-TOKEN training objective: a random fraction of the 49 tokens is
     replaced by a dedicated [MASK] token (id = cK, an extra vocab slot) and the
     model predicts the ORIGINAL ids at the masked positions only, and
  3. PARALLEL ITERATIVE DECODING: generation starts with the WHOLE grid masked
     and, over only ~8-12 steps, predicts every position at once, keeps the
     most-confident fraction per a COSINE mask schedule, re-masks the rest, and
     repeats until the grid is full. That is ~10 forward passes for the whole
     49-token image instead of VQVAE's 49 autoregressive passes.

This is a masked-token parallel image decoder, structurally
distinct from the autoregressive (TinyGPT, VQVAE prior), GAN (VisualGAN,
StyleGAN2) and diffusion (DiffusionMNIST, ConsistencyDistill) generators here.

WHAT IS GENUINELY NEW HERE (everything else is landed and reused):
  - the [MASK]-token corruption used as the training objective (FillMaskedInput),
  - the confidence-based COSINE unmasking SCHEDULER, i.e. the generation loop
    (GenerateGrid): gamma(t/T) = cos(pi/2 * t/T) sets how many tokens stay
    masked after step t, and per-position confidence (the sampled token's
    probability) decides WHICH tokens to commit.
The transformer block builder, the VQ encode/decode, and the PNG image I/O are
all reused from the library / from examples/VQVAE. NO new TNNet layer class.

THE TWO STAGES (Stage 1 is identical in spirit to examples/VQVAE).
  Stage 1 (representation): train encoder -> TNNetVectorQuantizer -> decoder to
    reconstruct MNIST; report reconstruction MSE + codebook usage; revive dead
    codes to avoid collapse; write a reconstruction grid PNG.
  Stage 2 (masked transformer + parallel generation): FREEZE the encoder, harvest
    7x7 code grids, train the bidirectional masked transformer (report MASKED-
    TOKEN PREDICTION ACCURACY rising), then GENERATE a grid of digits by parallel
    cosine-schedule decoding and DECODE through the frozen VQ decoder. Report
    codebook usage of the generated grids and a NaN/Inf check; write a samples PNG.

RUN MODES.
  default (SMOKE): short representation run + short masked-transformer run; well
    under five minutes on one CPU. The point of the smoke run is to watch the
    masked-token ACCURACY rise and to confirm the parallel-decode pipeline fills
    a coherent 49-token grid in ~10 steps without NaN -- NOT to produce sharp
    digits. The smoke output is deliberately rough/undertrained (exactly as the
    DiffusionMNIST smoke example documents); pass --full for sharper digits.
  --full : more epochs / bigger harvest.

DATA. Standard MNIST idx-ubyte files, looked up first in the working directory
then in ../VQVAE and ../DiffusionMNIST (the files every MNIST example here uses;
NOT copied/committed). If absent the program falls back to a SYNTHETIC dataset
(random axis-aligned bright bars, mirroring examples/ConsistencyDistill) so the
demo still runs offline in CI; the pipeline and metrics are unchanged.

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
  cMaskId   = cK;    // dedicated [MASK] token id (extra vocab slot)
  cVocab    = cK + 1;// transformer input/output vocab includes [MASK]
  cBeta     = 0.05;
  cReconW   = 0.125;

  cC1       = 16;
  cC2       = 32;

  // Bidirectional masked transformer over the 49-token code grid.
  cDModel   = 48;
  cHeads    = 3;
  cDFF      = 96;
  cBlocks   = 2;

  // SMOKE defaults. Kept deliberately short so the whole pipeline (Stage 1 +
  // Stage 2 + generation) finishes WELL under five minutes on one CPU; the
  // point of the smoke run is to watch the masked-token accuracy rise and to
  // confirm the parallel-decode loop fills a coherent grid, NOT sharp digits.
  cAEEpochs    = 5;
  cAEBatch     = 32;
  cAEStepsEp   = 45;
  cAELR        = 0.001;

  cHarvest     = 1200;
  cMaskEpochs  = 5;
  cMaskBatch   = 24;
  cMaskStepsEp = 55;
  cMaskLR      = 0.004;

  cReconRows   = 8;
  cGenGrid     = 6;     // generated PNG side (cGenGrid x cGenGrid digits)
  cDecodeSteps = 10;    // parallel-decode iterations (MaskGIT: 8-12)

var
  AEEpochs, AEStepsEp, Harvest, MaskEpochs, MaskStepsEp: integer;

  AE: TNNet;            // encoder -> VQ -> decoder
  LVQ: TNNetVectorQuantizer;
  EncOut: TNNetLayer;
  VQIdx, DecInIdx: integer;

  MaskT: TNNet;         // bidirectional masked transformer over the code grid
  TrainV, ValV, TestV: TNNetVolumeList;

  Img, Pseudo: TNNetVolume;
  UseSynthetic: boolean;

// ----- MNIST: load digit Idx into Img rescaled to [-1,1]. -------------------
procedure LoadDigit(Src: TNNetVolume);
var
  x, y: integer;
  v, b: TNeuralFloat;
begin
  if UseSynthetic then
  begin
    // Synthetic vols are already stored in [-1,1]; copy channel 0 straight in.
    for y := 0 to cImgSize - 1 do
      for x := 0 to cImgSize - 1 do
        Img[x, y, 0] := Src[x, y, 0];
    Exit;
  end;
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

// ----- Synthetic fallback dataset (bars), mirroring ConsistencyDistill. -----
procedure BuildSyntheticData(Count: integer);
var
  i, x, y, pos, thick: integer;
  Vol, VolT: TNNetVolume;
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
    if i mod 5 = 0 then
    begin
      VolT := TNNetVolume.Create(cImgSize, cImgSize, 1);
      VolT.Copy(Vol);
      TestV.Add(VolT);
    end;
  end;
end;

// ----- Build the discrete autoencoder: encoder -> VQ -> decoder. ------------
procedure BuildAutoEncoder;
begin
  AE := TNNet.Create();
  AE.AddLayer(TNNetInput.Create(cImgSize, cImgSize, 1));
  AE.AddLayer(TNNetConvolutionReLU.Create(cC1, 3, 1, 1, 0));
  AE.AddLayer(TNNetGroupNorm.Create(4));
  AE.AddLayer(TNNetMaxPool.Create(2));                       // 28 -> 14
  AE.AddLayer(TNNetConvolutionReLU.Create(cC2, 3, 1, 1, 0));
  AE.AddLayer(TNNetGroupNorm.Create(4));
  AE.AddLayer(TNNetMaxPool.Create(2));                       // 14 -> 7
  AE.AddLayer(TNNetConvolutionReLU.Create(cC2, 3, 1, 1, 0)); // 7x7 refine
  EncOut := AE.AddLayer(TNNetPointwiseConvLinear.Create(cEmb));

  LVQ := TNNetVectorQuantizer.Create(cK, cBeta);
  AE.AddLayer(LVQ);
  VQIdx := AE.GetLastLayerIdx();

  AE.AddLayer(TNNetPointwiseConvReLU.Create(cC2));
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

// ----- dead-code revival (anti-collapse), as in examples/VQVAE. -------------
var
  ZeBank: array of array of TNeuralFloat;
  ZeBankCount: integer;

procedure CaptureZe;
var
  EncVol: TNNetVolume;
  gx, gy, d, slot: integer;
begin
  EncVol := EncOut.Output;
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

// ----- Stage 1: train the discrete autoencoder. -----------------------------
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
        CaptureZe;
        for i := 0 to Img.Size - 1 do
          Pseudo.FData[i] := Pred.FData[i] -
            cReconW * (Pred.FData[i] - Img.FData[i]);
        AE.Backpropagate(Pseudo);
      end;
      AE.ForceMaxAbsoluteDelta(0.02);
      AE.UpdateWeights();
    end;
    Active := LVQ.ActiveCodeCount();
    ReviveDeadCodes(Revived);
    Elapsed := (Now() - StartTime) * 86400.0;
    WriteLn(Format('  epoch %2d/%2d  recon-MSE = %.5f  active codes = %2d/%2d  revived = %2d  elapsed = %.1fs',
      [Epoch, AEEpochs, SumLoss / (AEStepsEp * cAEBatch), Active, cK, Revived, Elapsed]));
  end;
end;

function PixToByte(v: TNeuralFloat; var NumNan: integer): TNeuralFloat;
begin
  if IsNan(v) or IsInfinite(v) then begin Inc(NumNan); v := 0; end;
  Result := (v + 1.0) * 127.5;
  if Result < 0 then Result := 0;
  if Result > 255 then Result := 255;
end;

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
          gy := y;
          Grid[gx, gy, 0] := pxO; Grid[gx, gy, 1] := pxO; Grid[gx, gy, 2] := pxO;
          gy := cImgSize + y;
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

procedure EncodeCodeGrid(out Codes: array of integer);
var
  gx, gy: integer;
begin
  AE.Compute(Img);
  for gx := 0 to cGrid - 1 do
    for gy := 0 to cGrid - 1 do
      Codes[gx * cGrid + gy] := LVQ.ChosenCodeIndex(gx, gy);
end;

// ===========================================================================
//        Stage 2: BIDIRECTIONAL masked transformer + parallel decode
// ===========================================================================

var
  HarvestCodes: array of array of integer;

// Bidirectional masked transformer over the cTokens-long code sequence. Input
// is one-hot over the (cK+1)-symbol vocab (cVocab includes [MASK]); the head
// predicts a per-POSITION distribution over the cVocab symbols using a
// PointwiseConvLinear head + a PER-POSITION PointwiseSoftMax (so position p's
// prediction is independent of the other positions' head -- crucial because we
// read every position's distribution at once during parallel decoding).
// The defining difference from the VQVAE prior is CausalMask=FALSE: every token
// attends to every other token (past AND future), which is what makes masked
// in-filling possible.
procedure BuildMaskedTransformer;
var
  i: integer;
begin
  MaskT := TNNet.Create();
  MaskT.AddLayer([
    TNNetInput.Create(cTokens, 1, cVocab),
    TNNetPointwiseConvLinear.Create(cDModel),
    TNNetAddPositionalEmbedding.Create(10000)
  ]);
  for i := 1 to cBlocks do
    MaskT.AddTransformerEncoderBlock(
      {Heads=}cHeads, {d_ff=}cDFF,
      {PreNorm=}true, {CausalMask=}false,   // BIDIRECTIONAL
      {UseRoPE=}false, {NormClass=}nil);
  MaskT.AddLayer([
    TNNetPointwiseConvReLU.Create(cDModel),
    TNNetPointwiseConvLinear.Create(cVocab),
    TNNetPointwiseSoftMax.Create()          // per-position softmax over vocab
  ]);
  MaskT.SetLearningRate(cMaskLR, 0.9);
  MaskT.SetL2Decay(0.0);
  MaskT.SetBatchUpdate(True);
end;

procedure HarvestCodeGrids;
var
  n, t: integer;
  Codes: array[0..cTokens - 1] of integer;
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

// Fill the (cTokens,1,cVocab) one-hot input from a token sequence, where any
// position flagged Masked carries the [MASK] symbol (cMaskId) instead of its
// real code. This is the NEW masked-corruption: it is used both at TRAIN time
// (random masking) and at GENERATE time (schedule-driven masking).
procedure FillMaskedInput(const Seq: array of integer;
  const Masked: array of boolean; InVol: TNNetVolume);
var
  p, tok: integer;
begin
  InVol.Fill(0);
  for p := 0 to cTokens - 1 do
  begin
    if Masked[p] then tok := cMaskId else tok := Seq[p];
    if (tok >= 0) and (tok < cVocab) then
      InVol[p, 0, tok] := 1.0;
  end;
end;

// Train the masked transformer. Per example we draw a masking RATIO r from a
// cosine-derived schedule (r = cos(pi/2 * u), u~Uniform(0,1) -- the same family
// MaskGIT uses), mask that fraction of positions with [MASK], and train the
// per-position softmax to predict the ORIGINAL code at the masked positions
// only (cross-entropy on masked positions). Reports masked-token prediction
// ACCURACY so the rise is visible.
procedure TrainMaskedTransformer;
var
  Epoch, Step, B, n, p, target, predIdx, k, nMask: integer;
  ratio, r2, mx: TNeuralFloat;
  SumLoss: TNeuralFloat;
  Correct, Counted: int64;
  Masked: array[0..cTokens - 1] of boolean;
  InVol, Tgt: TNNetVolume;
  Pred: TNNetVolume;
  StartTime, Elapsed: double;
begin
  InVol := TNNetVolume.Create(cTokens, 1, cVocab);
  Tgt   := TNNetVolume.Create(cTokens, 1, cVocab);
  WriteLn(Format('  training BIDIRECTIONAL masked transformer  (%d epochs x %d steps x batch %d).',
    [MaskEpochs, MaskStepsEp, cMaskBatch]));
  StartTime := Now();
  try
    for Epoch := 1 to MaskEpochs do
    begin
      SumLoss := 0; Correct := 0; Counted := 0;
      for Step := 1 to MaskStepsEp do
      begin
        MaskT.ClearDeltas();
        for B := 1 to cMaskBatch do
        begin
          n := Random(Harvest);
          // cosine-family masking ratio in (0,1]; ensure >=1 token masked.
          ratio := Cos(0.5 * Pi * Random);
          nMask := 0;
          for p := 0 to cTokens - 1 do
          begin
            Masked[p] := (Random < ratio);
            if Masked[p] then Inc(nMask);
          end;
          if nMask = 0 then begin Masked[Random(cTokens)] := true; end;

          FillMaskedInput(HarvestCodes[n], Masked, InVol);
          MaskT.Compute(InVol);
          Pred := MaskT.GetLastLayer.Output;

          // Build a per-position target: at MASKED positions the one-hot true
          // code; at unmasked positions copy the prediction so the error (and
          // thus the gradient) is ZERO there -> loss on masked positions only.
          Tgt.Copy(Pred);
          for p := 0 to cTokens - 1 do
            if Masked[p] then
            begin
              target := HarvestCodes[n][p];
              for k := 0 to cVocab - 1 do Tgt[p, 0, k] := 0.0;
              Tgt[p, 0, target] := 1.0;
              if Pred[p, 0, target] > 1e-9 then
                SumLoss := SumLoss - Ln(Pred[p, 0, target]);
              // accuracy: argmax over the real codes 0..cK-1.
              predIdx := 0; mx := Pred[p, 0, 0];
              for k := 1 to cK - 1 do
                if Pred[p, 0, k] > mx then begin mx := Pred[p, 0, k]; predIdx := k; end;
              if predIdx = target then Inc(Correct);
              Inc(Counted);
            end;
          MaskT.Backpropagate(Tgt);
        end;
        MaskT.UpdateWeights();
      end;
      Elapsed := (Now() - StartTime) * 86400.0;
      r2 := 0; if Counted > 0 then r2 := Correct / Counted;
      WriteLn(Format('  epoch %2d/%2d  masked-NLL = %.4f  masked-token acc = %5.1f%%  elapsed = %.1fs',
        [Epoch, MaskEpochs, SumLoss / Max(Counted, 1), 100.0 * r2, Elapsed]));
    end;
  finally
    InVol.Free;
    Tgt.Free;
  end;
end;

// ----- PARALLEL ITERATIVE DECODING (the MaskGIT generation loop). -----------
// Start with the whole grid masked. For cDecodeSteps iterations: predict every
// position at once; at currently-masked positions sample a token (temperature)
// and record its probability as CONFIDENCE; then a COSINE schedule
//   gamma(t/T) = cos(pi/2 * t/T)
// fixes how many tokens should REMAIN masked after step t; we keep (commit) the
// highest-confidence newly-predicted tokens and re-mask the rest. The final
// step commits everything. Returns the filled cTokens-long code sequence.
procedure GenerateGrid(out Seq: array of integer; Temperature: TNeuralFloat);
var
  Masked: array[0..cTokens - 1] of boolean;
  Conf: array[0..cTokens - 1] of TNeuralFloat;
  Cand: array[0..cTokens - 1] of integer;
  InVol: TNNetVolume;
  Pred: TNNetVolume;
  step, p, k, target, nMasked, keepMasked, toCommit, i, j: integer;
  logits: array[0..cK - 1] of TNeuralFloat;
  sum, rr, acc, gamma, frac: TNeuralFloat;
  order: array[0..cTokens - 1] of integer;
  tmp: integer; tf: TNeuralFloat;
begin
  InVol := TNNetVolume.Create(cTokens, 1, cVocab);
  try
    for p := 0 to cTokens - 1 do begin Masked[p] := true; Seq[p] := cMaskId; end;
    for step := 1 to cDecodeSteps do
    begin
      FillMaskedInput(Seq, Masked, InVol);
      MaskT.Compute(InVol);
      Pred := MaskT.GetLastLayer.Output;

      // At every currently-masked position, temperature-sample a real code and
      // record its probability as confidence.
      for p := 0 to cTokens - 1 do
      begin
        if not Masked[p] then Continue;
        sum := 0;
        for k := 0 to cK - 1 do
        begin
          logits[k] := Power(Max(Pred[p, 0, k], 1e-12), 1.0 / Temperature);
          sum := sum + logits[k];
        end;
        if sum <= 0 then sum := 1;
        rr := Random * sum; acc := 0; target := cK - 1;
        for k := 0 to cK - 1 do
        begin
          acc := acc + logits[k];
          if rr <= acc then begin target := k; break; end;
        end;
        Cand[p] := target;
        Conf[p] := Pred[p, 0, target];   // confidence = sampled token prob
      end;

      // Cosine schedule: how many tokens should STAY masked AFTER this step.
      frac := step / cDecodeSteps;
      gamma := Cos(0.5 * Pi * frac);          // 1 -> 0 across steps
      keepMasked := Round(gamma * cTokens);
      if step = cDecodeSteps then keepMasked := 0;

      nMasked := 0;
      for p := 0 to cTokens - 1 do if Masked[p] then Inc(nMasked);
      toCommit := nMasked - keepMasked;
      if toCommit < 0 then toCommit := 0;

      // Order the currently-masked positions by DESCENDING confidence and
      // commit the top 'toCommit' of them (insertion sort over <= cTokens).
      j := 0;
      for p := 0 to cTokens - 1 do if Masked[p] then begin order[j] := p; Inc(j); end;
      for i := 1 to j - 1 do
      begin
        tmp := order[i]; tf := Conf[tmp]; k := i - 1;
        while (k >= 0) and (Conf[order[k]] < tf) do
        begin order[k + 1] := order[k]; Dec(k); end;
        order[k + 1] := tmp;
      end;
      for i := 0 to toCommit - 1 do
      begin
        p := order[i];
        Seq[p] := Cand[p];
        Masked[p] := false;
      end;
    end;
    // Safety: commit any stragglers.
    for p := 0 to cTokens - 1 do
      if Masked[p] or (Seq[p] >= cK) then
      begin
        if Cand[p] < cK then Seq[p] := Cand[p] else Seq[p] := 0;
        Masked[p] := false;
      end;
  finally
    InVol.Free;
  end;
end;

// Decode a 49-token code grid into a 28x28 image (frozen VQ decoder tail).
procedure DecodeCodeGrid(const Seq: array of integer; Dst: TNNetVolume);
var
  gx, gy, d, tok, i: integer;
  CodeW: TNNetVolume;
  VQVol: TNNetVolume;
  Pred: TNNetVolume;
begin
  VQVol := AE.Layers[VQIdx].Output;
  for gx := 0 to cGrid - 1 do
    for gy := 0 to cGrid - 1 do
    begin
      tok := Seq[gx * cGrid + gy];
      if (tok < 0) or (tok > cK - 1) then tok := 0;
      CodeW := AE.Layers[VQIdx].Neurons[tok].Weights;
      for d := 0 to cEmb - 1 do
        VQVol[gx, gy, d] := CodeW.Raw[d];
    end;
  for i := DecInIdx to AE.CountLayers - 1 do
    AE.Layers[i].Compute();
  Pred := AE.GetLastLayer.Output;
  Dst.Copy(Pred);
end;

// ----- Generated-samples PNG + generation-side metrics. ---------------------
procedure WriteGeneratedPng(const FileName: string);
var
  Grid, One: TNNetVolume;
  r, c, x, y, gx, gy, NumNan, p: integer;
  Seq: array[0..cTokens - 1] of integer;
  px: TNeuralFloat;
  usedCode: array[0..cK - 1] of boolean;
  distinct: integer;
begin
  Grid := TNNetVolume.Create(cGenGrid * cImgSize, cGenGrid * cImgSize, 3);
  One  := TNNetVolume.Create(cImgSize, cImgSize, 1);
  Grid.Fill(0);
  NumNan := 0;
  for p := 0 to cK - 1 do usedCode[p] := false;
  try
    WriteLn(Format('Parallel decoding a %dx%d grid of digits (%d cosine-schedule steps each)...',
      [cGenGrid, cGenGrid, cDecodeSteps]));
    for r := 0 to cGenGrid - 1 do
      for c := 0 to cGenGrid - 1 do
      begin
        GenerateGrid(Seq, 1.0);
        for p := 0 to cTokens - 1 do
          if (Seq[p] >= 0) and (Seq[p] < cK) then usedCode[Seq[p]] := true;
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
    distinct := 0;
    for p := 0 to cK - 1 do if usedCode[p] then Inc(distinct);
    WriteLn(Format('  generated-grid codebook usage: %d/%d distinct codes used.',
      [distinct, cK]));
    if SaveImageFromVolumeIntoFile(Grid, FileName)
      then WriteLn('Wrote generated-samples grid: ', FileName)
      else WriteLn('FAILED to write: ', FileName);
    if NumNan > 0
      then WriteLn('WARNING: ', NumNan, ' NaN/Inf pixels clamped.')
      else WriteLn('NaN/Inf check: clean (no NaN/Inf in generated samples).');
  finally
    Grid.Free;
    One.Free;
  end;
end;

// ----- MNIST locator: working dir, then sibling examples. -------------------
function LocateMNIST: boolean;
begin
  Result := false;
  if CheckMNISTFile('train') and CheckMNISTFile('t10k') then begin Result := true; Exit; end;
  if (FileExists('../VQVAE/train-images.idx3-ubyte')) then
  begin SetCurrentDir('../VQVAE'); Result := CheckMNISTFile('train') and CheckMNISTFile('t10k'); if Result then Exit; end;
  if (FileExists('../DiffusionMNIST/train-images.idx3-ubyte')) then
  begin SetCurrentDir('../DiffusionMNIST'); Result := CheckMNISTFile('train') and CheckMNISTFile('t10k'); end;
end;

var
  i: integer;
  FullMode: boolean;
  StartDir, OutDir: string;
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
    MaskEpochs := 16; MaskStepsEp := 160;
    WriteLn('MaskGIT [FULL training mode]');
  end
  else
  begin
    AEEpochs := cAEEpochs; AEStepsEp := cAEStepsEp; Harvest := cHarvest;
    MaskEpochs := cMaskEpochs; MaskStepsEp := cMaskStepsEp;
    WriteLn('MaskGIT [SMOKE mode -- pass --full for sharper output]');
  end;
  WriteLn('Non-autoregressive masked-token image generation (Chang et al. 2022).');
  WriteLn(Format('  latent grid = %dx%d = %d tokens   codebook K = %d  [MASK] id = %d  emb dim = %d',
    [cGrid, cGrid, cTokens, cK, cMaskId, cEmb]));
  WriteLn(Format('  parallel decode = %d cosine-schedule steps (vs %d autoregressive).',
    [cDecodeSteps, cTokens]));
  WriteLn;

  StartDir := GetCurrentDir;
  // OutDir: write PNGs back into the example directory regardless of where the
  // MNIST data was found (we may chdir to a sibling example to load it).
  OutDir := StartDir;

  UseSynthetic := not LocateMNIST;
  if UseSynthetic then
  begin
    WriteLn('MNIST idx-ubyte files not found -> using SYNTHETIC bar dataset (CI fallback).');
    BuildSyntheticData(2000);
  end
  else
  begin
    CreateMNISTVolumes(TrainV, ValV, TestV, 'train', 't10k');
    WriteLn('Loaded MNIST: ', TrainV.Count, ' train / ', TestV.Count, ' test digits.');
  end;
  SetCurrentDir(OutDir);

  BuildAutoEncoder;
  WriteLn('Autoencoder architecture:');
  AE.PrintSummary();
  WriteLn;

  Img    := TNNetVolume.Create(cImgSize, cImgSize, 1);
  Pseudo := TNNetVolume.Create(cImgSize, cImgSize, 1);
  try
    TrainAutoEncoder;
    WriteLn;
    WriteReconPng('maskgit_reconstructions.png');
    WriteLn;

    BuildMaskedTransformer;
    WriteLn('Masked transformer (bidirectional encoder over code grid) architecture:');
    MaskT.PrintSummary();
    WriteLn;
    HarvestCodeGrids;
    TrainMaskedTransformer;
    WriteLn;
    WriteGeneratedPng('maskgit_generated.png');
    WriteLn;
    WriteLn('Done. View maskgit_reconstructions.png and maskgit_generated.png.');
    if not FullMode then
      WriteLn('(Smoke output is rough/undertrained; the headline is the parallel-decode');
    if not FullMode then
      WriteLn(' pipeline fills a coherent grid in ~10 steps. Run --full for sharper digits.)');
  finally
    Img.Free;
    Pseudo.Free;
    AE.Free;
    if Assigned(MaskT) then MaskT.Free;
    TrainV.Free;
    ValV.Free;
    TestV.Free;
  end;
end.
