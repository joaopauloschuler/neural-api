program GumbelAnnealingAutoencoder;
(*
GumbelAnnealingAutoencoder: a temperature-annealing micro-experiment for the
TNNetGumbelSoftmax bottleneck.

This is a genuinely different experiment from examples/GumbelSoftmaxDemo/, which
only sweeps tau on a FIXED logit vector at inference (no training). Here we TRAIN
a tiny discrete-latent autoencoder end-to-end:

    input (D dims)
      -> encoder MLP -> K logits
      -> TNNetGumbelSoftmax bottleneck (the discrete latent)
      -> decoder MLP -> reconstruction (D dims)

The synthetic dataset is genuinely K-category-structured: K well-separated
cluster prototypes in D dimensions plus small Gaussian noise. A categorical
(one-of-K) bottleneck is therefore the right inductive bias — a good model learns
to route each sample to its cluster's latent code and the decoder reconstructs
the prototype.

Temperature annealing
---------------------
We anneal tau from ~2.0 down to ~0.1 across a handful of training PHASES. Because
TNNetGumbelSoftmax stores tau in a PROTECTED field (FFloatSt[0]) set at
construction and exposes no public setter, we anneal by rebuilding the network
with the new tau at the start of each phase and copying the learned encoder /
decoder weights forward via TNNet.CopyWeights (the Gumbel layer itself has no
trainable weights, so nothing is lost). This is the simplest correct way to vary
tau per phase without touching the core library.

Headline output
---------------
For each phase we report (tau, reconstruction MSE, mean bottleneck output
entropy) on the deterministic inference path. As tau drops the categorical
sharpens towards one-hot, so the bottleneck entropy falls towards 0, while
reconstruction stays good (or improves) because the data really is K-category
structured. A graded PASS/FAIL verdict checks exactly that.

Copyright (C) 2026 Joao Paulo Schwarz Schuler

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.

Coded by Claude (AI).
*)

{$mode objfpc}{$H+}

uses
  SysUtils,
  neuralnetwork,
  neuralvolume;

const
  D          = 8;     // input / output dimensionality
  K          = 6;     // number of latent categories (= number of clusters)
  Hidden     = 16;    // MLP hidden width
  NSamples   = 240;   // synthetic dataset size
  NoiseStd   = 0.10;  // per-feature Gaussian noise on cluster prototypes
  EpochsPhase= 30;    // SGD passes over the dataset per annealing phase
  LR         = 0.01;  // learning rate
  // Annealing schedule: tau from ~2.0 down to ~0.1.
  TauSched: array[0..4] of TNeuralFloat = (2.0, 1.0, 0.5, 0.25, 0.1);

type
  TDataset = record
    X      : array of TNNetVolume;   // noisy samples (also the recon targets)
    Cluster: array of integer;       // ground-truth cluster id (for reference)
  end;

var
  Prototypes: array[0..K - 1] of array[0..D - 1] of TNeuralFloat;

// ---- synthetic data ---------------------------------------------------------

function Gauss(): TNeuralFloat;
// Box-Muller standard normal.
var
  u1, u2: TNeuralFloat;
begin
  u1 := Random(); if u1 < 1e-12 then u1 := 1e-12;
  u2 := Random();
  Result := Sqrt(-2.0 * Ln(u1)) * Cos(2.0 * Pi * u2);
end;

procedure BuildPrototypes();
var
  c, j: integer;
begin
  // Well-separated prototypes: each cluster gets a distinct large-magnitude
  // signature so the clusters do not overlap.
  for c := 0 to K - 1 do
    for j := 0 to D - 1 do
      // values in roughly [-2.5 .. 2.5], structured per (cluster, feature).
      Prototypes[c][j] := 2.5 * Sin(0.9 * c + 1.7 * j + 0.3 * c * j);
end;

procedure BuildDataset(var DS: TDataset);
var
  i, j, c: integer;
  V: TNNetVolume;
begin
  SetLength(DS.X, NSamples);
  SetLength(DS.Cluster, NSamples);
  for i := 0 to NSamples - 1 do
  begin
    c := Random(K);
    V := TNNetVolume.Create(1, 1, D);
    for j := 0 to D - 1 do
      V.Raw[j] := Prototypes[c][j] + NoiseStd * Gauss();
    DS.X[i] := V;
    DS.Cluster[i] := c;
  end;
end;

procedure FreeDataset(var DS: TDataset);
var
  i: integer;
begin
  for i := 0 to High(DS.X) do DS.X[i].Free;
  SetLength(DS.X, 0);
  SetLength(DS.Cluster, 0);
end;

// ---- model ------------------------------------------------------------------

function BuildAutoencoder(Tau: TNeuralFloat; out GumbelIdx: integer): TNNet;
begin
  Result := TNNet.Create();
  Result.AddLayer(TNNetInput.Create(1, 1, D));
  // Encoder MLP -> K logits.
  Result.AddLayer(TNNetFullConnectReLU.Create(Hidden));
  Result.AddLayer(TNNetFullConnectLinear.Create(K));
  // Discrete-latent bottleneck. Soft mode (hard=0) keeps a smooth gradient; the
  // Gumbel noise is added only while EnableDropouts(true) is set (training).
  GumbelIdx := Result.GetLastLayerIdx() + 1;
  Result.AddLayer(TNNetGumbelSoftmax.Create(Tau, 0));
  // Decoder MLP -> reconstruction.
  Result.AddLayer(TNNetFullConnectReLU.Create(Hidden));
  Result.AddLayer(TNNetFullConnectLinear.Create(D));
  Result.SetLearningRate(LR, 0.9);
end;

// ---- metrics ----------------------------------------------------------------

function ReconMSE(NN: TNNet; const DS: TDataset): TNeuralFloat;
var
  i, j: integer;
  Acc, Diff: TNeuralFloat;
  Outp: TNNetVolume;
begin
  // Deterministic inference path (dropouts/Gumbel-noise disabled).
  NN.EnableDropouts(false);
  Acc := 0;
  for i := 0 to High(DS.X) do
  begin
    NN.Compute(DS.X[i]);
    Outp := NN.GetLastLayer.Output;
    for j := 0 to D - 1 do
    begin
      Diff := Outp.Raw[j] - DS.X[i].Raw[j];
      Acc := Acc + Diff * Diff;
    end;
  end;
  Result := Acc / (NSamples * D);
end;

function MeanBottleneckEntropy(NN: TNNet; GumbelIdx: integer;
  const DS: TDataset): TNeuralFloat;
var
  i, j: integer;
  Acc, H, p: TNeuralFloat;
  Bottleneck: TNNetVolume;
begin
  // Mean Shannon entropy (nats) of the bottleneck's categorical output over the
  // dataset, on the deterministic inference path: y = softmax(logits / tau).
  NN.EnableDropouts(false);
  Acc := 0;
  for i := 0 to High(DS.X) do
  begin
    NN.Compute(DS.X[i]);
    Bottleneck := NN.Layers[GumbelIdx].Output;
    H := 0;
    for j := 0 to Bottleneck.Size - 1 do
    begin
      p := Bottleneck.Raw[j];
      if p > 1e-12 then H := H - p * Ln(p);
    end;
    Acc := Acc + H;
  end;
  Result := Acc / NSamples;
end;

// ---- training ---------------------------------------------------------------

procedure TrainPhase(NN: TNNet; const DS: TDataset);
var
  Ep, i: integer;
begin
  // Stochastic Gumbel sampling is enabled while training; the reconstruction
  // target IS the input (autoencoder).
  NN.EnableDropouts(true);
  for Ep := 1 to EpochsPhase do
    for i := 0 to High(DS.X) do
    begin
      NN.Compute(DS.X[i]);
      NN.Backpropagate(DS.X[i]);   // per-sample SGD update (target = input)
    end;
end;

// ---- main -------------------------------------------------------------------

var
  DS: TDataset;
  NN, NextNN: TNNet;
  Phase, GumbelIdx: integer;
  Tau, MSE, Ent, MaxLnK: TNeuralFloat;
  EntFirst, EntLast, MseFirst, MseLast: TNeuralFloat;
  EntFalls, MseOk, Verdict: boolean;
begin
  RandSeed := 424242;
  WriteLn('GumbelAnnealingAutoencoder: discrete-latent AE with tau annealing.');
  WriteLn(Format('D=%d  K=%d  Hidden=%d  N=%d  epochs/phase=%d  LR=%.3f',
    [D, K, Hidden, NSamples, EpochsPhase, LR]));
  MaxLnK := Ln(K);
  WriteLn(Format('Max possible bottleneck entropy = ln(K) = %.4f nats.', [MaxLnK]));
  WriteLn;

  BuildPrototypes();
  BuildDataset(DS);

  WriteLn('Annealing schedule (tau falls; weights carried forward via CopyWeights):');
  WriteLn('  phase    tau    recon-MSE    entropy   entropy/ln(K)');
  WriteLn('  -----  -----  -----------  ---------  -------------');

  NN := BuildAutoencoder(TauSched[0], GumbelIdx);
  EntFirst := 0; EntLast := 0; MseFirst := 0; MseLast := 0;

  for Phase := 0 to High(TauSched) do
  begin
    Tau := TauSched[Phase];
    if Phase > 0 then
    begin
      // Rebuild with the lower tau and carry encoder/decoder weights forward.
      NextNN := BuildAutoencoder(Tau, GumbelIdx);
      NextNN.CopyWeights(NN);
      NN.Free;
      NN := NextNN;
    end;

    TrainPhase(NN, DS);

    MSE := ReconMSE(NN, DS);
    Ent := MeanBottleneckEntropy(NN, GumbelIdx, DS);
    WriteLn(Format('  %5d  %5.2f  %11.6f  %9.5f  %13.4f',
      [Phase, Tau, MSE, Ent, Ent / MaxLnK]));

    if Phase = 0 then begin EntFirst := Ent; MseFirst := MSE; end;
    EntLast := Ent; MseLast := MSE;
  end;
  WriteLn;

  // ---- verdict --------------------------------------------------------------
  // The categorical should sharpen: entropy at the lowest tau must be well below
  // the entropy at the highest tau (and small in absolute terms). Reconstruction
  // should remain good — we require the final MSE to be no worse than ~1.5x the
  // initial MSE (it usually improves as the latent discretizes).
  EntFalls := (EntLast < 0.5 * EntFirst) and (EntLast < 0.25 * MaxLnK);
  MseOk    := (MseLast <= 1.5 * MseFirst + 1e-6);
  Verdict  := EntFalls and MseOk;

  WriteLn(Format('Entropy: %.5f (tau=%.2f)  ->  %.5f (tau=%.2f)   [%.0f%% of start]',
    [EntFirst, TauSched[0], EntLast, TauSched[High(TauSched)],
     100.0 * EntLast / (EntFirst + 1e-12)]));
  WriteLn(Format('Recon-MSE: %.6f  ->  %.6f', [MseFirst, MseLast]));
  if EntFalls then
    WriteLn('  [OK] Bottleneck sharpened toward one-hot as tau dropped.')
  else
    WriteLn('  [!!] Bottleneck entropy did NOT collapse as expected.');
  if MseOk then
    WriteLn('  [OK] Reconstruction stayed good as the latent discretized.')
  else
    WriteLn('  [!!] Reconstruction degraded as tau dropped.');
  WriteLn;
  if Verdict then
    WriteLn('VERDICT: PASS - annealing sharpened the categorical while keeping recon.')
  else
    WriteLn('VERDICT: FAIL - expected trend (entropy down, recon kept) not observed.');

  NN.Free;
  FreeDataset(DS);
  WriteLn('Done.');
end.
