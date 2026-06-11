program VQCodebookCollapse;
(*
VQCodebookCollapse: a STRESS test for VQ-VAE codebook collapse and one
published mitigation that reverses it.

Where the sibling examples/VQCodebookUsage/ deliberately AVOIDS collapse (a
healthy codebook converging to ~one code per cluster), this example does the
OPPOSITE: it deliberately DRIVES codebook collapse, charts the active-code
count FALLING over training, and then shows a published fix -- DEAD-CODE
RE-INITIALIZATION -- lifting the active-code count back up.

  Codebook collapse is the headline VQ-VAE failure mode: only a handful of
  codes ever win the nearest-neighbour argmin and the rest of the codebook is
  dead weight. We provoke it by stacking the deck:
    * FAR more codes than data modes (cK = 64 codes for cClusters = 16 blobs);
    * the codebook is SEEDED from data latents so it starts well-spread and
      MANY codes are active at epoch 1 -- the codebook starts HEALTHY;
    * an over-regularized encoder: after every epoch the encoder weights are
      multiplied by cContract (< 1), so the latent cloud contracts toward a
      point. Peripheral codes lose all their points and die, so the active-code
      count FALLS sharply over training (a clear, monotone collapse).

THE TWO ARMS
  COLLAPSE arm   : train as above, with NO intervention. Each epoch we probe
                   the codebook usage (ResetCodebookUsage -> forward pass over
                   a probe batch -> ActiveCodeCount) and watch active codes
                   fall toward a tiny number.
  MITIGATED arm  : identical training, but every cReinitEvery epochs we run
                   DEAD-CODE RE-INITIALIZATION: any code whose usage counter
                   is zero/near-zero (CodebookUsageCount(idx) <= cDeadThresh)
                   is re-seeded to a fresh encoder output (a live latent z_e
                   drawn from the data) plus small jitter. This is the standard
                   "random restart"/dead-code revival used in VQ-VAE-2,
                   Jukebox, and SoundStream to keep the codebook alive. It
                   touches ONLY the public codebook accessor
                   LVQ.Neurons[code].Weights -- no core layer change.

The probe and the codebook are both public: the usage counters via
ResetCodebookUsage / ActiveCodeCount / CodebookUsageCount, the codebook
vectors via LVQ.Neurons[code].Weights (a TNNetVolume, like any other layer's
trainable weights). The mitigation only WRITES codebook entries through that
public accessor; it never adds a core method.

Headline: the two active-code trajectories are printed side by side and the
program ends with a VERDICT, e.g.
  "PASS - dead-code re-init lifted active codes from N to M".

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
  neuralfit;

const
  cDim         = 8;     // input / reconstruction dimension
  cEmb         = 4;     // encoder output (codebook vector) dimension
  cClusters    = 16;    // ground-truth data modes the codebook COULD represent
  cK           = 64;    // codebook size (deliberately HUGE vs what survives)
  cBeta        = 0.25;  // commitment cost (paper default)
  cEpochs      = 30;
  cBatch       = 64;
  cProbe       = 384;   // probe-batch size for the usage report
  cLearnRate   = 0.01;  // learning rate
  cContract    = 0.70;  // per-epoch encoder shrink (over-regularization): the
                        // latent cloud contracts toward a point, peripheral
                        // codes lose all their points -> active count FALLS.
  cReinitEvery = 3;     // mitigation: revive dead codes every N epochs
  cDeadThresh  = 0;     // a code with usage <= this (in the probe) is "dead"

var
  Centers: array[0..cClusters - 1, 0..cDim - 1] of TNeuralFloat;

// Draw one sample: pick a random cluster, jitter its center tightly.
procedure DrawSample(V: TNNetVolume; out ClusterId: integer);
var
  I: integer;
begin
  ClusterId := Random(cClusters);
  for I := 0 to cDim - 1 do
    V.FData[I] := Centers[ClusterId, I] + 0.05 * (Random - 0.5);
end;

procedure InitCenters();
var
  C, I: integer;
begin
  for C := 0 to cClusters - 1 do
    for I := 0 to cDim - 1 do
      // Blobs scattered in a bounded cube around the origin: many distinct
      // modes the codebook COULD cover, but the collapse drivers (huge K,
      // high commitment cost, bunched codebook init) pin usage to a few.
      Centers[C, I] := 1.5 * (Random - 0.5);
end;

// Seed each codebook entry from a data-derived latent z_e (one fresh sample per
// code). This spreads the codebook over the encoder output so MANY codes win at
// epoch 1 -- the codebook starts healthy and then collapses over training.
// Writes ONLY through the public LVQ.Neurons[code].Weights accessor.
procedure SeedCodebookFromData(NN: TNNet; LVQ: TNNetVectorQuantizer;
  EncIdx: integer);
var
  Code, I, ClusterId: integer;
  Tmp, ZeLatent: TNNetVolume;
begin
  Tmp := TNNetVolume.Create(cDim, 1, 1);
  for Code := 0 to cK - 1 do
  begin
    DrawSample(Tmp, ClusterId);
    NN.Compute(Tmp);
    ZeLatent := NN.Layers[EncIdx].Output;
    for I := 0 to cEmb - 1 do
      LVQ.Neurons[Code].Weights.Raw[I] := ZeLatent.Raw[I];
  end;
  Tmp.Free;
end;

// Build the encoder -> VQ -> decoder autoencoder. Returns the VQ layer index.
function BuildNet(out NN: TNNet; out LVQ: TNNetVectorQuantizer): integer;
var
  EncIdx: integer;
begin
  NN := TNNet.Create();
  NN.AddLayer(TNNetInput.Create(cDim, 1, 1));
  // Encoder: produce a single cEmb-dimensional latent VECTOR living on the
  // Depth axis (shape 1x1xcEmb), so the quantizer treats the whole cEmb-vector
  // as one codebook lookup (D = Depth = cEmb), i.e. true vector quantization
  // rather than cEmb independent scalar quantizations.
  NN.AddLayer(TNNetFullConnectLinear.Create(1, 1, cEmb)); // encoder
  EncIdx := NN.GetLastLayerIdx();
  LVQ := TNNetVectorQuantizer.Create(cK, cBeta);    // quantizer bottleneck
  NN.AddLayer(LVQ);
  Result := NN.GetLastLayerIdx();
  NN.AddLayer(TNNetFullConnectLinear.Create(cDim)); // decoder

  NN.SetLearningRate(cLearnRate, 0.0);
  NN.SetL2Decay(0.0);
  NN.SetBatchUpdate(True);

  // Seed the codebook from data-derived latents so it spans the encoder output
  // at epoch 1 (high initial active count) -- the codebook starts HEALTHY. The
  // collapse is then driven over training by the per-epoch encoder contraction
  // in TrainEpoch (cContract): the latent cloud shrinks toward a point so
  // peripheral codes lose all their points and the active-code count FALLS --
  // the textbook over-regularized-encoder collapse.
  SeedCodebookFromData(NN, LVQ, EncIdx);
end;

// One epoch of hand-rolled mini-batch SGD on reconstruction MSE. EncIdx names
// the encoder layer: after each epoch its weights are multiplied by cContract
// (< 1), an over-regularization that shrinks the latent cloud toward a point.
// This is the deliberate collapse driver: peripheral codes lose all their
// points so the active-code count FALLS over training.
procedure TrainEpoch(NN: TNNet; Inp, Pseudo: TNNetVolume; EncIdx: integer);
var
  Step, B, I, ClusterId: integer;
  Outp: TNNetVolume;
begin
  for Step := 1 to (cProbe div cBatch) + 1 do
  begin
    NN.ClearDeltas();
    for B := 1 to cBatch do
    begin
      DrawSample(Inp, ClusterId);
      NN.Compute(Inp);
      Outp := NN.GetLastLayer().Output;
      for I := 0 to cDim - 1 do
        Pseudo.FData[I] := Outp.FData[I] -
          (1.0 / cBatch) * (Outp.FData[I] - Inp.FData[I]);
      NN.Backpropagate(Pseudo);
    end;
    NN.UpdateWeights();
  end;
  // Over-regularize the encoder: contract the latent cloud toward the origin.
  NN.Layers[EncIdx].MulWeights(cContract);
end;

// Probe the codebook over a fresh batch (no training) and return active count.
function ProbeActive(NN: TNNet; LVQ: TNNetVectorQuantizer;
  Inp: TNNetVolume): integer;
var
  B, ClusterId: integer;
begin
  LVQ.ResetCodebookUsage();
  for B := 1 to cProbe do
  begin
    DrawSample(Inp, ClusterId);
    NN.Compute(Inp);
  end;
  Result := LVQ.ActiveCodeCount();
end;

// DEAD-CODE RE-INITIALIZATION (published "random restart" mitigation):
// any code whose probe usage is <= cDeadThresh is re-seeded to a live encoder
// output (z_e for a fresh data sample) plus small jitter, so it re-enters the
// argmin competition near where the data actually lives. Returns # revived.
// Writes ONLY through the public LVQ.Neurons[code].Weights accessor.
function ReinitDeadCodes(NN: TNNet; LVQ: TNNetVectorQuantizer;
  EncIdx: integer; Inp: TNNetVolume): integer;
var
  Code, I, ClusterId: integer;
  ZeLatent: TNNetVolume;
begin
  Result := 0;
  // First, probe once so CodebookUsageCount reflects current behaviour.
  LVQ.ResetCodebookUsage();
  for I := 1 to cProbe do
  begin
    DrawSample(Inp, ClusterId);
    NN.Compute(Inp);
  end;
  ZeLatent := NN.Layers[EncIdx].Output; // encoder output z_e (cEmb dims)
  for Code := 0 to cK - 1 do
    if LVQ.CodebookUsageCount(Code) <= cDeadThresh then
    begin
      // Fresh live latent as the re-init target.
      DrawSample(Inp, ClusterId);
      NN.Compute(Inp);
      for I := 0 to cEmb - 1 do
        LVQ.Neurons[Code].Weights.Raw[I] :=
          ZeLatent.Raw[I] + 0.1 * (Random - 0.5);
      Inc(Result);
    end;
end;

procedure RunArm(const ArmName: string; DoMitigate: boolean;
  out Traj: array of integer; out FinalActive: integer);
var
  NN: TNNet;
  LVQ: TNNetVectorQuantizer;
  VQIdx, EncIdx: integer;
  Inp, Pseudo: TNNetVolume;
  Epoch, Active, Revived: integer;
begin
  RandSeed := 424242; // identical data + init across both arms
  InitCenters();
  VQIdx := BuildNet(NN, LVQ);
  LVQ := NN.Layers[VQIdx] as TNNetVectorQuantizer;
  EncIdx := VQIdx - 1;

  Inp    := TNNetVolume.Create(cDim, 1, 1);
  Pseudo := TNNetVolume.Create(cDim, 1, 1);

  WriteLn('=== ', ArmName, ' arm ===');
  WriteLn('epoch | active codes (of ', cK, ') | note');
  WriteLn('------+----------------------+----------------------------');

  for Epoch := 1 to cEpochs do
  begin
    TrainEpoch(NN, Inp, Pseudo, EncIdx);

    if DoMitigate and (Epoch mod cReinitEvery = 0) then
    begin
      Revived := ReinitDeadCodes(NN, LVQ, EncIdx, Inp);
      Active := ProbeActive(NN, LVQ, Inp);
      Traj[Epoch - 1] := Active;
      WriteLn(Format('%5d | %20d | dead-code re-init: revived %d', [Epoch, Active, Revived]));
    end
    else
    begin
      Active := ProbeActive(NN, LVQ, Inp);
      Traj[Epoch - 1] := Active;
      if (Epoch <= 3) or (Epoch mod 3 = 0) or (Epoch = cEpochs) then
        WriteLn(Format('%5d | %20d |', [Epoch, Active]));
    end;
  end;

  FinalActive := ProbeActive(NN, LVQ, Inp);
  WriteLn('Final active codes (', ArmName, '): ', FinalActive, ' of ', cK);
  WriteLn;

  NN.Free;
  Inp.Free;
  Pseudo.Free;
end;

var
  CollapseTraj, MitigTraj: array[0..cEpochs - 1] of integer;
  CollapseFinal, MitigFinal, PeakCollapse, I: integer;
begin
  WriteLn('VQCodebookCollapse: VQ-VAE codebook-COLLAPSE stress test');
  WriteLn('  input dim=', cDim, '  emb dim=', cEmb,
          '  clusters=', cClusters, '  codebook K=', cK,
          '  beta=', cBeta:0:2, '  LR=', cLearnRate:0:3,
          '  encoder contraction/epoch=', cContract:0:2);
  WriteLn('  (K >> data modes + per-epoch encoder contraction are the collapse drivers)');
  WriteLn;

  RunArm('COLLAPSE', False, CollapseTraj, CollapseFinal);
  RunArm('MITIGATED', True, MitigTraj, MitigFinal);

  // Peak active count the unmitigated arm reached (its best, before collapse).
  PeakCollapse := 0;
  for I := 0 to cEpochs - 1 do
    if CollapseTraj[I] > PeakCollapse then PeakCollapse := CollapseTraj[I];

  WriteLn('Active-code trajectory (every 3rd epoch):');
  WriteLn('epoch | COLLAPSE | MITIGATED');
  WriteLn('------+----------+----------');
  for I := 0 to cEpochs - 1 do
    if ((I + 1) mod 3 = 0) or (I = 0) then
      WriteLn(Format('%5d | %8d | %8d', [I + 1, CollapseTraj[I], MitigTraj[I]]));
  WriteLn;

  WriteLn('COLLAPSE arm : peaked at ', PeakCollapse,
          ' active codes, ended at ', CollapseFinal, ' of ', cK, '.');
  WriteLn('MITIGATED arm: ended at ', MitigFinal, ' of ', cK,
          ' active codes (dead-code re-init every ', cReinitEvery, ' epochs).');
  WriteLn;

  // VERDICT: the mitigation must keep MORE codes alive than the collapsed arm,
  // and must clear the ground-truth-cluster floor.
  if (MitigFinal > CollapseFinal) and (MitigFinal >= cClusters) then
    WriteLn('VERDICT: PASS - dead-code re-init lifted active codes from ',
            CollapseFinal, ' to ', MitigFinal,
            ' (collapse arm stuck at ', CollapseFinal, ').')
  else
  begin
    WriteLn('VERDICT: FAIL - mitigation did not lift active codes ',
            '(collapse=', CollapseFinal, ', mitigated=', MitigFinal, ').');
    Halt(1);
  end;
end.
