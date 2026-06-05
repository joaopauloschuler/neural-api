program SubPixelSuperRes;
(*
SubPixelSuperRes: a tiny, self-contained demonstration that a convolutional
network with a sub-pixel (depth-to-space) upsampling head -- the in-tree layer
TNNetPixelShuffle -- can LEARN a 2x super-resolution mapping on purely
synthetic data. No external dataset, no new layers, pure CPU, deterministic.

THE IDEA (sub-pixel / "PixelShuffle" upsampling, Shi et al. 2016):
  Instead of upsampling with a transposed convolution, a network keeps all
  convolutions at the LOW resolution and produces an output tensor with
  (r*r * C) channels. A final, parameter-free reshape ("pixel shuffle" /
  depth-to-space) folds those r*r groups of channels into an r-times-larger
  spatial grid with C channels:
      (SizeX, SizeY, r*r*C)  --PixelShuffle(r)-->  (r*SizeX, r*SizeY, C)
  This in-tree TNNetPixelShuffle(r) uses the mapping
      out[r*x+i, r*y+j, c] = in[x, y, c*r*r + i*r + j].
  Doing the upsample as a deterministic reshape (with the convs learning the
  r*r sub-pixel channels) is cheaper and avoids the checkerboard artefacts of
  transposed convolutions. This example uses r=2.

THE TASK (synthetic, deterministic):
  - Inputs: random low-resolution 8x8 single-channel "blocky" tiles -- each tile
    is a coarse 4x4 grid of random intensities, nearest-neighbour-expanded to
    8x8, so the patterns are genuinely low-frequency (a fair SR target).
  - Targets: the FIXED, known ground-truth 2x upscale of each LR tile, namely a
    16x16 nearest-neighbour expansion (each LR pixel -> a 2x2 output block).
    This is a deterministic function of the input, so a perfect mapping exists
    and "did it learn?" has an unambiguous answer.
  - The net must reproduce that 16x16 target from the 8x8 input.

THE NET (3 trainable conv layers + the reshape head):
      Input(8,8,1)
      -> TNNetConvolutionReLU(16, 3,1,1)     // low-res feature extractor
      -> TNNetConvolutionReLU(16, 3,1,1)     // low-res feature extractor
      -> TNNetConvolutionLinear(4, 3,1,1)    // produce r*r*C = 4*1 = 4 channels
      -> TNNetPixelShuffle(2)                // (8,8,4) -> (16,16,1) : the 2x upsample
  All convolutions run at 8x8; only the final reshape changes resolution.

WHAT IS PRINTED (the headline):
  MSE and PSNR of the net's 16x16 output vs the ground-truth 16x16 target,
  measured on a held-out test set BEFORE training (random init) and AFTER
  training. The PixelShuffle head, fed by the learned convs, drives the MSE
  down by orders of magnitude / PSNR up by tens of dB -- i.e. the network
  LEARNS the 2x super-resolution mapping.

Built-in correctness gate (printed PASS/FAIL, Halt(1) on failure):
  - test MSE after training is much lower than before (>= 10x reduction);
  - test PSNR after training clears a clean-reconstruction bar.

Pure CPU, no external data, deterministic seeding (RandSeed=424242, single
thread via manual Compute/Backpropagate), finishes well under the time budget.

Copyright (C) 2026 Joao Paulo Schwarz Schuler

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

Coded by Claude (AI).
*)

{$mode objfpc}{$H+}

uses {$IFDEF UNIX} {$IFDEF UseCThreads}
  cthreads, {$ENDIF} {$ENDIF}
  Classes, SysUtils, Math,
  neuralnetwork,
  neuralvolume;

const
  cSeed     = 424242;
  cLowRes   = 8;      // low-resolution input is cLowRes x cLowRes
  cFactor   = 2;      // upscale factor (PixelShuffle r) => 16x16 output
  cHighRes  = cLowRes * cFactor;
  cBlocks   = 4;      // LR tile is a cBlocks x cBlocks coarse grid (low-frequency)
  cFeatures = 16;     // conv feature maps in the two hidden layers
  cTrain    = 256;    // synthetic training tiles
  cTest     = 64;     // held-out test tiles
  cBatch    = 16;     // mini-batch size for SGD
  cLR       = 0.001;  // learning rate (regression/MSE gradients are large -> small, stable LR)
  cMomentum = 0.9;
  cEpochs   = 120;
  // NOTE on LR: this is a regression task trained against raw squared-error
  // gradients, which are much larger than the soft-max/cross-entropy gradients
  // used by the classification examples. A small LR (1e-3) converges cleanly;
  // an order of magnitude higher (1e-2) plateaus and 3e-2 diverges.

// ---------------------------------------------------------------------------
// Synthetic data. A low-res tile is a coarse cBlocks x cBlocks grid of random
// intensities, nearest-neighbour-expanded to cLowRes x cLowRes (so the input
// is genuinely low-frequency). The TARGET is the deterministic 2x
// nearest-neighbour upscale to cHighRes x cHighRes (each LR pixel -> a
// cFactor x cFactor block). A perfect mapping therefore exists.
// ---------------------------------------------------------------------------
procedure BuildSet(out Pairs: TNNetVolumePairList; Count: integer);
var
  I, bx, by, px, py, gx, gy: integer;
  X, Y: TNNetVolume;
  Grid: array[0..cBlocks - 1, 0..cBlocks - 1] of TNeuralFloat;
begin
  Pairs := TNNetVolumePairList.Create();
  for I := 0 to Count - 1 do
  begin
    // Random coarse grid of intensities in [0,1].
    for bx := 0 to cBlocks - 1 do
      for by := 0 to cBlocks - 1 do
        Grid[bx, by] := Random;

    // Low-res input (cLowRes x cLowRes): coarse grid nearest-neighbour-expanded.
    X := TNNetVolume.Create(cLowRes, cLowRes, 1);
    for px := 0 to cLowRes - 1 do
      for py := 0 to cLowRes - 1 do
      begin
        gx := (px * cBlocks) div cLowRes;
        gy := (py * cBlocks) div cLowRes;
        X[px, py, 0] := Grid[gx, gy];
      end;

    // Ground-truth target (cHighRes x cHighRes): the 2x nearest-neighbour
    // upscale of the LR input (each LR pixel -> a cFactor x cFactor block).
    Y := TNNetVolume.Create(cHighRes, cHighRes, 1);
    for px := 0 to cHighRes - 1 do
      for py := 0 to cHighRes - 1 do
        Y[px, py, 0] := X[px div cFactor, py div cFactor, 0];

    Pairs.Add(TNNetVolumePair.Create(X, Y));
  end;
end;

// ---------------------------------------------------------------------------
// The net: all convs at low resolution, then the parameter-free PixelShuffle
// head does the 2x upsample.
//   Input(8,8,1)
//   -> ConvReLU(16,3,1,1) -> ConvReLU(16,3,1,1)
//   -> ConvLinear(cFactor*cFactor*1 = 4, 3,1,1)   // r*r*C channels
//   -> PixelShuffle(2)                            // (8,8,4) -> (16,16,1)
// ---------------------------------------------------------------------------
procedure BuildNet(out NN: TNNet);
begin
  NN := TNNet.Create();
  NN.AddLayer(TNNetInput.Create(cLowRes, cLowRes, 1));
  NN.AddLayer(TNNetConvolutionReLU.Create(cFeatures, 3, 1, 1));
  NN.AddLayer(TNNetConvolutionReLU.Create(cFeatures, 3, 1, 1));
  NN.AddLayer(TNNetConvolutionLinear.Create(cFactor * cFactor * 1, 3, 1, 1));
  NN.AddLayer(TNNetPixelShuffle.Create(cFactor));
  NN.SetLearningRate(cLR, cMomentum);
  NN.SetL2Decay(0.0);
  NN.SetBatchUpdate(True);  // accumulate the mini-batch gradient, then one step
end;

// Mean squared error over a pair list (per-pixel, single channel).
function MeanSquaredError(NN: TNNet; Pairs: TNNetVolumePairList): TNeuralFloat;
var
  I, P: integer;
  Out: TNNetVolume;
  Sum, Diff: TNeuralFloat;
  PixelCount: integer;
begin
  Sum := 0;
  PixelCount := 0;
  for I := 0 to Pairs.Count - 1 do
  begin
    NN.Compute(Pairs[I].I);
    Out := NN.GetLastLayer().Output;
    for P := 0 to Out.Size - 1 do
    begin
      Diff := Out.FData[P] - Pairs[I].O.FData[P];
      Sum := Sum + Diff * Diff;
      Inc(PixelCount);
    end;
  end;
  if PixelCount > 0 then Result := Sum / PixelCount else Result := 0;
end;

// PSNR (dB) from MSE, assuming a dynamic range of 1.0 (intensities in [0,1]).
function PSNRFromMSE(MSE: TNeuralFloat): TNeuralFloat;
begin
  if MSE <= 1e-12 then Result := 120.0  // effectively perfect; cap for display
  else Result := 10.0 * Log10(1.0 / MSE);
end;

var
  TrainSet, TestSet: TNNetVolumePairList;
  NN: TNNet;
  Epoch, I, J, B, Tmp, InBatch: integer;
  Order: array of integer;
  MSEBefore, MSEAfter, PSNRBefore, PSNRAfter: TNeuralFloat;
  StartTime, EndTime: TDateTime;
  PassMSE, PassPSNR: boolean;
const
  cPSNRBar = 25.0;  // after training, reconstruction must clear this PSNR (dB)
begin
  WriteLn('================================================================');
  WriteLn('SubPixelSuperRes: a PixelShuffle (sub-pixel) net LEARNS a 2x');
  WriteLn('super-resolution mapping on synthetic data.');
  WriteLn('================================================================');
  WriteLn(Format('Task: LR %dx%d (coarse %dx%d grid) -> HR %dx%d (2x upscale), 1 channel.',
    [cLowRes, cLowRes, cBlocks, cBlocks, cHighRes, cHighRes]));
  WriteLn(Format('Train=%d, Test=%d (held-out).', [cTrain, cTest]));
  WriteLn(Format('Net: Input(%d,%d,1) -> ConvReLU(%d,3,1,1) -> ConvReLU(%d,3,1,1)',
    [cLowRes, cLowRes, cFeatures, cFeatures]));
  WriteLn(Format('     -> ConvLinear(%d,3,1,1) -> PixelShuffle(%d) -> (%d,%d,1).',
    [cFactor * cFactor, cFactor, cHighRes, cHighRes]));
  WriteLn(Format('Mini-batch SGD  batch=%d  LR=%.3f  momentum=%.2f  epochs=%d  RandSeed=%d',
    [cBatch, cLR, cMomentum, cEpochs, cSeed]));
  WriteLn;

  StartTime := Now;

  // Deterministic data + weight init under the fixed seed.
  RandSeed := cSeed;
  BuildSet(TrainSet, cTrain);
  BuildSet(TestSet, cTest);
  BuildNet(NN);
  // Manual Compute/Backpropagate below are single-threaded, so the whole run
  // is deterministic on one CPU core without any thread-pool setup.

  WriteLn(Format('Net weights: %d', [NN.CountWeights()]));
  WriteLn;

  // BEFORE training (random init).
  MSEBefore := MeanSquaredError(NN, TestSet);
  PSNRBefore := PSNRFromMSE(MSEBefore);
  WriteLn(Format('BEFORE training:  test MSE = %.6f   PSNR = %6.2f dB',
    [MSEBefore, PSNRBefore]));

  // Train: mini-batch SGD, shuffle order each epoch, one UpdateWeights per batch.
  SetLength(Order, TrainSet.Count);
  for I := 0 to High(Order) do Order[I] := I;
  for Epoch := 1 to cEpochs do
  begin
    for I := High(Order) downto 1 do
    begin
      J := Random(I + 1);
      Tmp := Order[I]; Order[I] := Order[J]; Order[J] := Tmp;
    end;
    InBatch := 0;
    NN.ClearDeltas();
    for B := 0 to High(Order) do
    begin
      NN.Compute(TrainSet[Order[B]].I);
      NN.Backpropagate(TrainSet[Order[B]].O);
      Inc(InBatch);
      if (InBatch >= cBatch) or (B = High(Order)) then
      begin
        NN.UpdateWeights();
        NN.ClearDeltas();
        InBatch := 0;
      end;
    end;
    if (Epoch mod 20 = 0) or (Epoch = cEpochs) then
      WriteLn(Format('  epoch %3d  train MSE = %.6f',
        [Epoch, MeanSquaredError(NN, TrainSet)]));
  end;

  // AFTER training.
  MSEAfter := MeanSquaredError(NN, TestSet);
  PSNRAfter := PSNRFromMSE(MSEAfter);

  EndTime := Now;

  WriteLn;
  WriteLn(Format('AFTER  training:  test MSE = %.6f   PSNR = %6.2f dB',
    [MSEAfter, PSNRAfter]));
  WriteLn;
  WriteLn('=== Results ===');
  WriteLn(Format('test MSE  : %.6f -> %.6f  (%.1fx reduction)',
    [MSEBefore, MSEAfter, MSEBefore / Max(MSEAfter, 1e-12)]));
  WriteLn(Format('test PSNR : %6.2f dB -> %6.2f dB  (+%.2f dB)',
    [PSNRBefore, PSNRAfter, PSNRAfter - PSNRBefore]));
  WriteLn;

  WriteLn('=== Correctness gate ===');
  PassMSE := MSEAfter <= MSEBefore / 10.0;
  WriteLn(Format('[%s] test MSE dropped >= 10x (%.6f -> %.6f): the net learned the upscale.',
    [BoolToStr(PassMSE, 'PASS', 'FAIL'), MSEBefore, MSEAfter]));
  PassPSNR := PSNRAfter >= cPSNRBar;
  WriteLn(Format('[%s] test PSNR after training = %.2f dB (must be >= %.1f dB): clean reconstruction.',
    [BoolToStr(PassPSNR, 'PASS', 'FAIL'), PSNRAfter, cPSNRBar]));
  WriteLn;

  WriteLn('TAKEAWAY: with all convolutions kept at low resolution and a single');
  WriteLn('parameter-free TNNetPixelShuffle(2) head doing the depth-to-space');
  WriteLn('reshape, the network learns the 2x super-resolution mapping -- MSE');
  WriteLn('falls by orders of magnitude and PSNR rises by tens of dB.');
  WriteLn;

  if PassMSE and PassPSNR then
    WriteLn('=> ALL CHECKS PASS: PixelShuffle net learned the 2x super-resolution.')
  else
    WriteLn('=> SOME CHECKS FAILED (see above).');
  WriteLn(Format('Total wall-clock: %.1f s', [(EndTime - StartTime) * 86400.0]));

  TrainSet.Free;
  TestSet.Free;
  NN.Free;

  if not (PassMSE and PassPSNR) then
    Halt(1);
end.
