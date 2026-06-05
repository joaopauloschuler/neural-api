program VQCodebookUsage;
(*
VQCodebookUsage: trains a tiny VQ-VAE-style bottleneck (encoder -> vector
quantizer -> decoder) on synthetic CLUSTERED data and reports, over training,
how many of the codebook entries are actually used -- the headline VQ-VAE
failure mode is CODEBOOK COLLAPSE, where only a handful of codes ever win the
nearest-neighbour argmin and the rest of the codebook is dead weight.

This is the demonstration companion to the new TNNetVectorQuantizer
codebook-usage probe:
  LVQ.ResetCodebookUsage();        // zero the per-code selection counters
  ... NN.Compute(probe sample) ... // each forward pass increments the winner
  LVQ.ActiveCodeCount();           // # distinct codes selected >= once
  LVQ.CodebookUsageCount(idx);     // win count for one code (histogram)
The probe is pure runtime bookkeeping: it does NOT change the quantization
math or any gradient, and it is NOT serialized.

The synthetic data is drawn from cClusters well-separated Gaussian blobs in
input space, so a HEALTHY codebook should converge to use roughly one code per
cluster. We deliberately allocate MORE codes than clusters (cK > cClusters) so
unused codes are visible. Each epoch we probe the codebook over a fixed batch
and print active-code count + a usage histogram, ending with a PASS/FAIL verdict
on whether the codebook stayed healthy (used >= cClusters distinct codes) rather
than collapsing.

Copyright (C) 2026 Joao Paulo Schwarz Schuler

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
any later version.

Coded by Claude (AI).
*)

{$mode objfpc}{$H+}

uses {$IFDEF UNIX} {$IFDEF UseCThreads}
  cthreads, {$ENDIF} {$ENDIF}
  Classes, SysUtils,
  neuralnetwork,
  neuralvolume,
  neuralfit;

const
  cDim       = 6;     // input / reconstruction dimension
  cEmb       = 4;     // encoder output (codebook vector) dimension
  cClusters  = 5;     // number of ground-truth data clusters
  cK         = 12;    // codebook size (deliberately > cClusters)
  cBeta      = 0.25;  // VQ commitment cost
  cEpochs    = 40;
  cBatch     = 64;
  cProbe     = 200;   // probe-batch size for the usage report
  cLearnRate = 0.002;

var
  Centers: array[0..cClusters - 1, 0..cDim - 1] of TNeuralFloat;

// Draw one sample: pick a random cluster, jitter its center.
procedure DrawSample(V: TNNetVolume; out ClusterId: integer);
var
  I: integer;
begin
  ClusterId := Random(cClusters);
  for I := 0 to cDim - 1 do
    V.FData[I] := Centers[ClusterId, I] + 0.08 * (Random - 0.5);
end;

procedure InitCenters();
var
  C, I: integer;
begin
  RandSeed := 424242;
  for C := 0 to cClusters - 1 do
    for I := 0 to cDim - 1 do
      // Well-separated blobs at a MODEST scale (keeps the linear AE stable):
      // each cluster lives near a distinct region of input space.
      Centers[C, I] := 0.4 * (Random - 0.5) + 1.0 * C;
end;

var
  NN: TNNet;
  LVQ: TNNetVectorQuantizer;
  VQIdx: integer;
  Inp, Pseudo, Outp: TNNetVolume;
  Epoch, Step, B, I, Active, ClusterId, Used: integer;
  Histo: string;

begin
  WriteLn('VQCodebookUsage: VQ-VAE codebook-collapse probe demo');
  WriteLn('  input dim=', cDim, '  emb dim=', cEmb,
          '  clusters=', cClusters, '  codebook K=', cK);
  WriteLn;

  InitCenters();

  // Encoder -> VQ bottleneck -> decoder; MSE reconstruction.
  NN := TNNet.Create();
  NN.AddLayer(TNNetInput.Create(cDim, 1, 1));
  NN.AddLayer(TNNetFullConnectLinear.Create(cEmb)); // encoder
  LVQ := TNNetVectorQuantizer.Create(cK, cBeta);    // quantizer bottleneck
  NN.AddLayer(LVQ);
  VQIdx := NN.GetLastLayerIdx();
  NN.AddLayer(TNNetFullConnectLinear.Create(cDim)); // decoder

  NN.SetLearningRate(cLearnRate, 0.0);
  NN.SetL2Decay(0.0);
  NN.SetBatchUpdate(True); // accumulate per-sample deltas, apply once per batch

  Inp    := TNNetVolume.Create(cDim, 1, 1);
  Pseudo := TNNetVolume.Create(cDim, 1, 1);

  LVQ := NN.Layers[VQIdx] as TNNetVectorQuantizer;

  WriteLn('epoch | active codes (of ', cK, ') | usage histogram');
  WriteLn('------+----------------------+-------------------------------');

  for Epoch := 1 to cEpochs do
  begin
    // ---- one epoch of hand-rolled mini-batch SGD on reconstruction MSE ----
    for Step := 1 to (cProbe div cBatch) + 1 do
    begin
      NN.ClearDeltas();
      for B := 1 to cBatch do
      begin
        DrawSample(Inp, ClusterId);
        NN.Compute(Inp);
        Outp := NN.GetLastLayer().Output;
        // Mean-MSE pseudo-target: stock error = (1/cBatch)*(out - in).
        for I := 0 to cDim - 1 do
          Pseudo.FData[I] := Outp.FData[I] -
            (1.0 / cBatch) * (Outp.FData[I] - Inp.FData[I]);
        NN.Backpropagate(Pseudo);
      end;
      NN.UpdateWeights();
    end;

    // ---- probe codebook usage over a fresh batch (no training) ----
    LVQ.ResetCodebookUsage();
    for B := 1 to cProbe do
    begin
      DrawSample(Inp, ClusterId);
      NN.Compute(Inp);
    end;
    Active := LVQ.ActiveCodeCount();

    if (Epoch <= 5) or (Epoch mod 5 = 0) or (Epoch = cEpochs) then
    begin
      Histo := '';
      for I := 0 to cK - 1 do
        if LVQ.CodebookUsageCount(I) > 0 then
          Histo := Histo + IntToStr(I) + ':' +
            IntToStr(LVQ.CodebookUsageCount(I)) + ' ';
      WriteLn(Format('%5d | %20d | %s', [Epoch, Active, Histo]));
    end;
  end;

  WriteLn;
  // Final verdict: a healthy codebook on cClusters blobs uses at least
  // cClusters distinct codes; collapse would pin it far below that.
  LVQ.ResetCodebookUsage();
  for B := 1 to cProbe do
  begin
    DrawSample(Inp, ClusterId);
    NN.Compute(Inp);
  end;
  Used := LVQ.ActiveCodeCount();
  WriteLn('Final active codes: ', Used, ' of ', cK,
          '  (ground-truth clusters: ', cClusters, ')');
  if Used >= cClusters then
    WriteLn('VERDICT: PASS - codebook is healthy (>= ', cClusters,
            ' distinct codes used, no collapse).')
  else
  begin
    WriteLn('VERDICT: FAIL - codebook COLLAPSED (only ', Used,
            ' distinct codes used).');
    NN.Free; Inp.Free; Pseudo.Free;
    Halt(1);
  end;

  NN.Free;
  Inp.Free;
  Pseudo.Free;
end.
