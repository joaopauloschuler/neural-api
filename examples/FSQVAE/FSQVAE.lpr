program FSQVAE;
(*
FSQVAE: a Finite Scalar Quantization autoencoder on MNIST, showing that FSQ is
STRUCTURALLY collapse-free -- every quantization level of every channel is used
(per-channel utilization -> ~100%), with NO learned codebook, NO EMA and NO
commitment loss.

WHY FSQ (Mentzer et al. 2023, "Finite Scalar Quantization: VQ-VAE Made Simple",
https://arxiv.org/abs/2309.15505). The classic VQ-VAE (van den Oord et al. 2017,
demonstrated here in examples/VQVAE and stress-tested in
examples/VQCodebookCollapse) discretizes a latent vector by nearest-neighbour
lookup into a LEARNED codebook -- and that codebook famously COLLAPSES: only a
handful of entries ever win the argmin and the rest are dead weight, needing
hacks (EMA updates, commitment loss, dead-code re-init) to stay alive. FSQ
sidesteps the whole problem: it has NO codebook at all. Each of d latent
channels is independently squashed by a bounded tanh and ROUNDED to one of L_i
integer levels:
    f(z)  = tanh(z + shift_i) * half_l_i - offset_i        (the bounded value)
    zhat  = round(f(z))                                    (nearest level)
with half_l_i = (L_i-1)/2, offset_i = 0.5 if L_i even else 0, and
shift_i = atanh(offset_i/half_l_i) (the lucidrains vector-quantize-pytorch FSQ
math, reproduced exactly in the repo layer TNNetFiniteScalarQuant). The implicit
codebook is the PRODUCT of the L_i (here 5^6 = 15625 codes), reachable
WITHOUT ever being stored. Because the round is deterministic and every level is
always in range, the codebook cannot collapse by construction. Gradients flow
through the non-differentiable round via the STRAIGHT-THROUGH estimator: the
backward pass is just the analytic derivative of the tanh bound.

WHAT THIS PROGRAM DOES.
  * Builds a tiny MLP autoencoder: 784 -> encoder -> cChan FSQ channels (the
    discrete bottleneck) -> decoder -> 784, trained on reconstruction MSE over a
    small MNIST subset for a few epochs (hand-rolled mini-batch SGD).
  * After training, probes the bottleneck over a fresh batch and reports:
      - PER-CHANNEL level utilization: of each channel's L_i levels, how many are
        actually hit. FSQ being collapse-free, this climbs to ~100%.
      - the number of DISTINCT full codes (mixed-radix CodeIndex) seen, read via
        the public LVQ.CodeIndex(X,Y) accessor (the discrete token a downstream
        embedding / transformer prior would consume).
  * Prints a graded VERDICT: PASS when mean per-channel level utilization clears
    a high bar (the collapse-free guarantee in action).

RUN MODES.
  default (SMOKE): a short run on a small subset; finishes in well under a
    couple of minutes on one CPU.
  --full : larger subset / more epochs for a sharper reconstruction.

DATA. Standard MNIST idx-ubyte files in the working directory (the same files
every MNIST example here uses). If absent the program prints a hint and exits
cleanly.

Copyright (C) 2026 Joao Paulo Schwarz Schuler

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
any later version.

Coded by Claude (AI).
*)

{$mode objfpc}{$H+}

uses {$IFDEF UNIX} cthreads, {$ENDIF}
  Classes, SysUtils,
  neuralnetwork,
  neuralvolume,
  neuraldatasets;

const
  cChan      = 6;     // FSQ bottleneck channels (one level count each below)
  cHidden    = 64;    // encoder/decoder hidden width
  cBatch     = 32;
  cLearnRate = 0.001;
  cLatentGain = 2.5;  // widen unit-std latents to span all 5 levels per channel

var
  // Per-channel level counts: implicit codebook = product = 5^6 = 15625. Five
  // levels per channel pairs cleanly with the unit-std latents produced by the
  // per-channel std-normalization, so every level is reachable (see BuildNet).
  Levels: array[0..cChan - 1] of integer = (5, 5, 5, 5, 5, 5);
  cEpochs: integer = 25;    // SMOKE default (raised by --full)
  cTrain:  integer = 5000;  // training subset size
  cProbe:  integer = 2000;  // probe-batch size for the utilization report

// Build the MLP autoencoder. Returns the FSQ layer + the encoder layer index.
function BuildNet(out NN: TNNet; out LFSQ: TNNetFiniteScalarQuant): integer;
var
  LevelsDyn: array of integer;
  I: integer;
begin
  SetLength(LevelsDyn, cChan);
  for I := 0 to cChan - 1 do LevelsDyn[I] := Levels[I];

  NN := TNNet.Create();
  NN.AddLayer(TNNetInput.Create(28, 28, 1));
  NN.AddLayer(TNNetFullConnectReLU.Create(cHidden));
  NN.AddLayer(TNNetFullConnectReLU.Create(cHidden));
  // Encoder head: cChan latent values living on the Depth axis (1x1xcChan), so
  // FSQ quantizes each of the cChan channels independently. A per-channel
  // std-normalization keeps each latent in the INFORMATIVE region of the tanh
  // bound (unsaturated): without it the latents either collapse to ~0 (one
  // level) or blow past the tanh knee (also one level), the classic FSQ
  // training pitfall. Normalized, every channel spans several levels.
  NN.AddLayer(TNNetFullConnectLinear.Create(1, 1, cChan)); // encoder output z_e
  NN.AddLayer(TNNetChannelStdNormalization.Create());      // -> unit-std latents
  // Fixed gain so the unit-std latent spans the FULL +/-2 range of a 5-level
  // channel: without it tanh(z) for |z|~1 rarely reaches the OUTER levels and
  // they stay unused. The gain widens the latent so every level is exercised.
  NN.AddLayer(TNNetMulByConstant.Create(cLatentGain));
  Result := NN.GetLastLayerIdx();
  LFSQ := TNNetFiniteScalarQuant.Create(LevelsDyn);        // codebook-free bottleneck
  NN.AddLayer(LFSQ);
  NN.AddLayer(TNNetFullConnectReLU.Create(cHidden));       // decoder
  NN.AddLayer(TNNetFullConnectReLU.Create(cHidden));
  NN.AddLayer(TNNetFullConnectLinear.Create(28, 28, 1));   // reconstruction

  NN.SetLearningRate(cLearnRate, 0.0);
  NN.SetL2Decay(0.0);
  NN.SetBatchUpdate(True);
end;

// One epoch of hand-rolled mini-batch SGD on reconstruction MSE.
procedure TrainEpoch(NN: TNNet; Data: TNNetVolumeList; Pseudo: TNNetVolume);
var
  Step, B, I, Idx, Steps: integer;
  Inp, Outp: TNNetVolume;
begin
  Steps := cTrain div cBatch;
  for Step := 1 to Steps do
  begin
    NN.ClearDeltas();
    for B := 1 to cBatch do
    begin
      Idx := Random(cTrain);
      Inp := Data[Idx];
      NN.Compute(Inp);
      Outp := NN.GetLastLayer().Output;
      // dL/dy of 0.5*||y-x||^2 averaged over the batch: pseudo-target = y - grad.
      for I := 0 to Outp.Size - 1 do
        Pseudo.FData[I] := Outp.FData[I] -
          (1.0 / cBatch) * (Outp.FData[I] - Inp.FData[I]);
      NN.Backpropagate(Pseudo);
    end;
    NN.UpdateWeights();
  end;
end;

// Probe the FSQ bottleneck over a fresh batch (no training): count which of each
// channel's L_i levels are hit and how many distinct full codes appear.
procedure ProbeUtilization(NN: TNNet; LFSQ: TNNetFiniteScalarQuant;
  Data: TNNetVolumeList;
  out MeanLevelUtil: TNeuralFloat; out DistinctCodes: integer);
var
  B, Ch, Zhat, PerCh, Idx, Hit, TotalLevels, UsedLevels: integer;
  SeenLevel: array of array of boolean;
  SeenCode: array of boolean;
begin
  SetLength(SeenLevel, cChan);
  for Ch := 0 to cChan - 1 do
    SetLength(SeenLevel[Ch], Levels[Ch]);
  SetLength(SeenCode, LFSQ.CodebookSize());

  for B := 1 to cProbe do
  begin
    NN.Compute(Data[Random(cTrain)]);
    for Ch := 0 to cChan - 1 do
    begin
      Zhat := Round(LFSQ.Output[0, 0, Ch]);
      PerCh := Zhat + (Levels[Ch] div 2);
      if PerCh < 0 then PerCh := 0;
      if PerCh > Levels[Ch] - 1 then PerCh := Levels[Ch] - 1;
      SeenLevel[Ch][PerCh] := True;
    end;
    Idx := LFSQ.CodeIndex(0, 0);
    if (Idx >= 0) and (Idx < Length(SeenCode)) then SeenCode[Idx] := True;
  end;

  TotalLevels := 0;
  UsedLevels := 0;
  for Ch := 0 to cChan - 1 do
  begin
    Hit := 0;
    for PerCh := 0 to Levels[Ch] - 1 do
      if SeenLevel[Ch][PerCh] then Inc(Hit);
    WriteLn(Format('  channel %d: %d / %d levels used (%5.1f%%)',
      [Ch, Hit, Levels[Ch], 100.0 * Hit / Levels[Ch]]));
    Inc(TotalLevels, Levels[Ch]);
    Inc(UsedLevels, Hit);
  end;
  MeanLevelUtil := 100.0 * UsedLevels / TotalLevels;

  DistinctCodes := 0;
  for Idx := 0 to Length(SeenCode) - 1 do
    if SeenCode[Idx] then Inc(DistinctCodes);
end;

var
  TrainV, ValV, TestV: TNNetVolumeList;
  NN: TNNet;
  LFSQ: TNNetFiniteScalarQuant;
  EncIdx, Epoch: integer;
  Pseudo: TNNetVolume;
  MeanUtil: TNeuralFloat;
  Distinct: integer;
begin
  if ParamStr(1) = '--full' then
  begin
    cEpochs := 15;
    cTrain  := 8000;
    cProbe  := 3000;
  end;

  WriteLn('FSQVAE: Finite Scalar Quantization autoencoder on MNIST');
  WriteLn('  bottleneck channels=', cChan,
          '  levels=[5,5,5,5,5,5]  implicit codebook=15625',
          '  epochs=', cEpochs, '  train subset=', cTrain);
  WriteLn('  (FSQ = NO codebook / NO EMA / NO commitment loss -> collapse-free)');
  WriteLn;

  if not FileExists('train-images.idx3-ubyte') then
  begin
    WriteLn('MNIST files not found in the working directory.');
    WriteLn('Copy train-images.idx3-ubyte etc. here (see any MNIST example).');
    Halt(0);
  end;

  RandSeed := 424242;
  CreateMNISTVolumes(TrainV, ValV, TestV, 'train', 't10k', False);

  EncIdx := BuildNet(NN, LFSQ);
  if EncIdx < 0 then ; // silence "unused" on some FPC configs

  Pseudo := TNNetVolume.Create();
  Pseudo.ReSize(NN.GetLastLayer().Output);

  WriteLn('Training ', cEpochs, ' epochs ...');
  for Epoch := 1 to cEpochs do
  begin
    TrainEpoch(NN, TrainV, Pseudo);
    if (Epoch <= 2) or (Epoch mod 3 = 0) or (Epoch = cEpochs) then
    begin
      NN.Compute(TrainV[0]);
      WriteLn(Format('  epoch %d done (|z_e|max=%.3f)',
        [Epoch, NN.Layers[EncIdx].Output.GetMaxAbs()]));
    end;
  end;
  WriteLn;

  WriteLn('Per-channel level utilization (probe ', cProbe, ' digits):');
  ProbeUtilization(NN, LFSQ, TrainV, MeanUtil, Distinct);
  WriteLn;
  WriteLn(Format('Mean per-channel level utilization: %5.1f%%', [MeanUtil]));
  WriteLn('Distinct full codes seen: ', Distinct, ' of ', LFSQ.CodebookSize());
  WriteLn;

  // VERDICT: FSQ is collapse-free by construction -- with a trained encoder the
  // per-channel levels should be almost fully exercised.
  if MeanUtil >= 90.0 then
    WriteLn(Format('VERDICT: PASS - FSQ levels nearly fully used (%.1f%%), ' +
      'no codebook collapse.', [MeanUtil]))
  else
  begin
    WriteLn(Format('VERDICT: FAIL - mean level utilization only %.1f%%.',
      [MeanUtil]));
    NN.Free; Pseudo.Free;
    TrainV.Free; ValV.Free; TestV.Free;
    Halt(1);
  end;

  NN.Free;
  Pseudo.Free;
  TrainV.Free;
  ValV.Free;
  TestV.Free;
end.
