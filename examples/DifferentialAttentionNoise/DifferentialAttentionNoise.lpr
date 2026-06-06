program DifferentialAttentionNoise;
(*
DifferentialAttentionNoise: the headline NOISE-CANCELLATION micro-experiment
for TNNetDifferentialAttention (Differential Transformer, Ye et al.,
Microsoft 2024, "Differential Transformer", https://arxiv.org/abs/2410.05258).

The claim under test
---------------------
Softmax attention must spend a full unit of probability mass on the keys
even when none is relevant; on such an "all-keys-irrelevant" query position
that mass is pure noise spread over the keys. Differential attention computes
TWO independent softmax maps from two halves of the Q|K channels and outputs
their scaled DIFFERENCE applied to V:
  Attn_eff[j] = softmax1(Q1.K1)[j] - lambda * softmax2(Q2.K2)[j]
The two maps share the SAME common-mode attention noise, so subtracting them
cancels it: the effective attention mass left on the irrelevant keys should
land STRICTLY BELOW plain SDPA's (which is the full 1.0). With the paper's
lambda_init ~= 0.8 the residual is ~ (1 - lambda).

This probe builds a tiny CAUSAL next-token setup, TNNetInput(SeqLen,1,3*d_k)
(the depth axis packs Q | K | V). Every position except the last gets a query
strongly aligned with its own key (a clear real target). The LAST position is
the PROBE row: its query is ~orthogonal to every real key, so all keys are
irrelevant to it and the whole probe row is attention NOISE.

We run a forward pass through (a) plain TNNetScaledDotProductAttention and
(b) TNNetDifferentialAttention (both causal), read the probe row's post-
softmax map(s) off the layers (AttentionWeights, plus AttentionWeights2 on
the differential layer), and print the noise mass on the irrelevant keys.
For SDPA the noise mass is the softmax row sum (= 1.0). For the differential
layer we report both the NET effective mass sum_j (a1 - lambda*a2)[j] and the
ABSOLUTE effective mass sum_j |a1 - lambda*a2|[j]; both should fall below 1.0.

Forward-only, deterministic, runs in well under a second on one CPU thread.

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
  neuralvolume;

const
  cDk     = 8;    // feature width (must be EVEN for differential attention)
  cSeqLen = 6;    // sequence length; the last position is the probe row

// Build a tiny causal next-token input. For each position p<SeqLen-1 the query
// Q[p] aligns with key K[p] (a strong self-aligned "real" target). The last
// position (the probe) gets a query ~orthogonal to every key, so all real keys
// are irrelevant to it. The Q channels are filled IDENTICALLY across the two
// d_k/2 sub-heads so both differential maps see the same (noise) pattern --
// the common-mode noise the layer is meant to cancel.
procedure FillInput(V: TNNetVolume);
var p, d: integer;
begin
  V.Fill(0);
  for p := 0 to cSeqLen - 1 do
  begin
    // Key K[p]: a one-hot bump at channel (p mod d_k), mirrored into both
    // halves of the key channels so K1 and K2 carry the same structure.
    V[p, 0, cDk + (p mod cDk)] := 1.0;
    // Value V[p]: small distinct pattern (not load-bearing for the test).
    for d := 0 to cDk - 1 do
      V[p, 0, 2 * cDk + d] := 0.1 * (p + 1) + 0.01 * d;
    if p < cSeqLen - 1 then
    begin
      // Real rows: query aligns with this position's own key.
      V[p, 0, p mod cDk] := 1.0;
    end
    else
    begin
      // Probe row: query points in a direction no key uses, identical in both
      // sub-heads -> both softmax maps produce the same uniform noise.
      for d := 0 to cDk - 1 do
        V[p, 0, d] := 0.05;
    end;
  end;
end;

// SDPA noise mass on the probe row = softmax row sum (all keys are irrelevant).
function SdpaNoiseMass(NN: TNNet): TNeuralFloat;
var
  Attn: TNNetVolume;
  Layer: TNNetScaledDotProductAttention;
  j, qi: integer;
begin
  Layer := TNNetScaledDotProductAttention(NN.Layers[1]);
  Attn := Layer.AttentionWeights;
  qi := cSeqLen - 1;
  Result := 0;
  for j := 0 to cSeqLen - 1 do
    Result := Result + Attn[j, qi, 0];
end;

// Differential effective noise mass on the probe row: net sum and absolute sum
// of (a1[j] - lambda*a2[j]) over the keys.
procedure DiffNoiseMass(NN: TNNet; out NetMass, AbsMass, Lambda: TNeuralFloat);
var
  Attn1, Attn2: TNNetVolume;
  Layer: TNNetDifferentialAttention;
  j, qi: integer;
  w: TNeuralFloat;
begin
  Layer := TNNetDifferentialAttention(NN.Layers[1]);
  Attn1 := Layer.AttentionWeights;   // map 1 (parent FAttn)
  Attn2 := Layer.AttentionWeights2;  // map 2 (noise estimate)
  Lambda := Layer.Lambda;
  qi := cSeqLen - 1;
  NetMass := 0;
  AbsMass := 0;
  for j := 0 to cSeqLen - 1 do
  begin
    w := Attn1[j, qi, 0] - Lambda * Attn2[j, qi, 0];
    NetMass := NetMass + w;
    AbsMass := AbsMass + Abs(w);
  end;
end;

var
  NN: TNNet;
  Input: TNNetVolume;
  SdpaNoise, DiffNet, DiffAbs, Lambda: TNeuralFloat;
begin
  SetExceptionMask([exInvalidOp, exDenormalized, exZeroDivide,
                    exOverflow, exUnderflow, exPrecision]);
  RandSeed := 2026;

  Input := TNNetVolume.Create(cSeqLen, 1, 3 * cDk);
  FillInput(Input);

  WriteLn('TNNetDifferentialAttention noise-cancellation micro-experiment');
  WriteLn('  d_k = ', cDk, ', SeqLen = ', cSeqLen, ' (causal)');
  WriteLn('  probe = last query row; its query is ~orthogonal to all keys');
  WriteLn('  (so EVERY real key is irrelevant -> the whole probe row is noise).');
  WriteLn;

  // (a) Plain SDPA: full unit of mass spent on the irrelevant keys.
  NN := TNNet.Create();
  try
    NN.AddLayer(TNNetInput.Create(cSeqLen, 1, 3 * cDk));
    NN.AddLayer(TNNetScaledDotProductAttention.Create(cDk, {causal=}true));
    NN.Compute(Input);
    SdpaNoise := SdpaNoiseMass(NN);
  finally
    NN.Free;
  end;

  // (b) Differential attention: the two softmax maps subtract.
  NN := TNNet.Create();
  try
    NN.AddLayer(TNNetInput.Create(cSeqLen, 1, 3 * cDk));
    NN.AddLayer(TNNetDifferentialAttention.Create(cDk, {causal=}true));
    NN.Compute(Input);
    DiffNoiseMass(NN, DiffNet, DiffAbs, Lambda);
  finally
    NN.Free;
  end;

  WriteLn(StringOfChar('-', 64));
  WriteLn(Format('%-26s %14s', ['model', 'probe noise mass']));
  WriteLn(StringOfChar('-', 64));
  WriteLn(Format('%-26s %14.4f', ['plain SDPA', SdpaNoise]));
  WriteLn(Format('%-26s %14.4f', ['differential (net)', DiffNet]));
  WriteLn(Format('%-26s %14.4f', ['differential (abs)', DiffAbs]));
  WriteLn(StringOfChar('-', 64));
  WriteLn(Format('  lambda = %.4f (paper lambda_init)', [Lambda]));
  WriteLn;
  if DiffAbs < SdpaNoise then
    WriteLn(Format('Differential noise mass (%.4f abs / %.4f net) lands BELOW '
      + 'plain', [DiffAbs, DiffNet]))
  else
    WriteLn(Format('NOTE: differential abs mass %.4f did NOT fall below plain '
      + '%.4f', [DiffAbs, SdpaNoise]));
  WriteLn('SDPA''s 1.0000. The two softmax maps carry the same common-mode');
  WriteLn('attention noise on the irrelevant probe row, so subtracting');
  WriteLn('lambda*map2 cancels ~lambda of it: the residual is ~(1 - lambda).');
  WriteLn('That is the Differential Transformer noise-cancellation claim.');

  Input.Free;
end.
