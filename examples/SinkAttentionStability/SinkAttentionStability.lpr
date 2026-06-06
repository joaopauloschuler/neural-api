program SinkAttentionStability;
(*
SinkAttentionStability: attention-SINK stability micro-experiment for
TNNetSinkAttention (the StreamingLLM "attention sink" idea, Xiao et al.
2023, "Efficient Streaming Language Models with Attention Sinks",
https://arxiv.org/abs/2309.17453).

The claim under test
---------------------
Softmax attention is forced to distribute a full unit of probability mass
over the keys EVEN WHEN no key is relevant (the softmax denominator can
never be zero). On such an "all-keys-irrelevant" query position the mass
gets dumped onto whatever key scores least-badly -- noise. StreamingLLM's
fix is to append a few always-available learnable SINK slots that are
never masked; softmax can park the otherwise-misplaced mass there, so the
mass landing on the REAL keys drops.

This probe builds a tiny CAUSAL next-token setup: SeqLen positions whose
Q|K|V slabs are packed along the depth axis as TNNetInput(SeqLen,1,3*d_k)
(the same layout SDPA uses). Every position except the last is given a
query that strongly aligns with its own key (a clear "real" target). The
LAST position is the PROBE row: its query is ~orthogonal to every real
key (all keys irrelevant), so plain softmax has nowhere good to put its
mass and spreads it across the real keys.

We run a forward pass through:
  (a) plain TNNetScaledDotProductAttention (causal), and
  (b) TNNetSinkAttention (causal) with NumSinks in {1, 2, 4},
and read the post-softmax attention map of the PROBE row directly from the
layer (AttentionWeights for SDPA; SinkAttentionWeights for the sink layer,
whose row is [sink slots ++ real keys]). We print the fraction of mass on
the SINK slot(s) vs the REAL keys. As NumSinks grows the sink slots should
absorb more mass, so the real-key mass on the irrelevant row should DROP
below plain SDPA's (which by construction is 1.0 -- it has no sink).

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
  cDk     = 8;    // feature width; input depth = 3*d_k (Q|K|V)
  cSeqLen = 6;    // sequence length; the last position is the probe row

// Build a tiny causal next-token input. For each position p<SeqLen-1 the
// query Q[p] equals key K[p] (a strong self-aligned "real" target). The last
// position (the probe) gets a query that is ~orthogonal to every key, so all
// real keys are irrelevant to it. V is filled with a benign pattern.
procedure FillInput(V: TNNetVolume);
var p, d: integer;
begin
  V.Fill(0);
  for p := 0 to cSeqLen - 1 do
  begin
    // Key K[p]: a one-hot-ish bump at channel (p mod d_k), value 1.0.
    V[p, 0, cDk + (p mod cDk)] := 1.0;
    // Value V[p]: small distinct pattern (not load-bearing for the test).
    for d := 0 to cDk - 1 do
      V[p, 0, 2 * cDk + d] := 0.1 * (p + 1) + 0.01 * d;
    if p < cSeqLen - 1 then
    begin
      // Real rows: query aligns with this position's own key -> high score.
      V[p, 0, p mod cDk] := 1.0;
    end
    else
    begin
      // Probe row: query points in a direction no key uses (orthogonal to
      // all the one-hot key bumps), so every real key scores ~0.
      for d := 0 to cDk - 1 do
        V[p, 0, d] := 0.05;  // tiny uniform -> ~0 dot with any one-hot key
    end;
  end;
end;

// Sum of post-softmax mass on the REAL keys for the probe (last) query row.
// SDPA map layout: [key j, query i, 0]; all keys are real.
function SdpaRealKeyMass(NN: TNNet): TNeuralFloat;
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
    Result := Result + Attn[j, qi, 0];  // causal: all j<=qi visible here
end;

// For the sink layer, split the probe row's augmented map into sink mass and
// real-key mass. Layout: X in 0..K-1 = sink slots, K..K+SeqLen-1 = real keys.
procedure SinkMasses(NN: TNNet; NumSinks: integer;
  out SinkMass, RealMass: TNeuralFloat);
var
  Attn: TNNetVolume;
  Layer: TNNetSinkAttention;
  x, qi: integer;
begin
  Layer := TNNetSinkAttention(NN.Layers[1]);
  Attn := Layer.SinkAttentionWeights;
  qi := cSeqLen - 1;
  SinkMass := 0;
  RealMass := 0;
  for x := 0 to NumSinks - 1 do
    SinkMass := SinkMass + Attn[x, qi, 0];
  for x := NumSinks to NumSinks + cSeqLen - 1 do
    RealMass := RealMass + Attn[x, qi, 0];
end;

var
  NN: TNNet;
  Input: TNNetVolume;
  SinkSweep: array[0..2] of integer = (1, 2, 4);
  Idx, K: integer;
  SdpaReal, SinkMass, RealMass: TNeuralFloat;
begin
  SetExceptionMask([exInvalidOp, exDenormalized, exZeroDivide,
                    exOverflow, exUnderflow, exPrecision]);
  RandSeed := 2026;

  Input := TNNetVolume.Create(cSeqLen, 1, 3 * cDk);
  FillInput(Input);

  WriteLn('TNNetSinkAttention attention-sink stability micro-experiment');
  WriteLn('  d_k = ', cDk, ', SeqLen = ', cSeqLen, ' (causal)');
  WriteLn('  probe = last query row; its query is ~orthogonal to all keys');
  WriteLn('  (so EVERY real key is irrelevant to it).');
  WriteLn;

  // (a) Plain SDPA: no sink, so softmax MUST put all mass on the real keys.
  NN := TNNet.Create();
  try
    NN.AddLayer(TNNetInput.Create(cSeqLen, 1, 3 * cDk));
    NN.AddLayer(TNNetScaledDotProductAttention.Create(cDk, {causal=}true));
    NN.Compute(Input);
    SdpaReal := SdpaRealKeyMass(NN);
    WriteLn(Format('plain SDPA  : real-key mass on probe row = %.4f  '
      + '(sink mass = n/a, no sink slot)', [SdpaReal]));
  finally
    NN.Free;
  end;

  WriteLn;
  WriteLn(StringOfChar('-', 64));
  WriteLn(Format('%-10s %14s %14s %18s',
    ['NumSinks', 'sink mass', 'real mass', 'real - plainSDPA']));
  WriteLn(StringOfChar('-', 64));

  // (b) Sink attention across the NumSinks sweep.
  for Idx := 0 to High(SinkSweep) do
  begin
    K := SinkSweep[Idx];
    NN := TNNet.Create();
    try
      NN.AddLayer(TNNetInput.Create(cSeqLen, 1, 3 * cDk));
      NN.AddLayer(TNNetSinkAttention.Create(cDk, {causal=}true, {NumSinks=}K));
      NN.Compute(Input);
      SinkMasses(NN, K, SinkMass, RealMass);
      WriteLn(Format('%-10d %14.4f %14.4f %18.4f',
        [K, SinkMass, RealMass, RealMass - SdpaReal]));
    finally
      NN.Free;
    end;
  end;
  WriteLn(StringOfChar('-', 64));
  WriteLn;
  WriteLn('Reading: plain SDPA dumps the full unit of mass (1.0) onto the');
  WriteLn('irrelevant real keys -- it has no outlet. With sink slots, softmax');
  WriteLn('parks part of that mass on the never-masked sinks, so the real-key');
  WriteLn('mass on the probe row falls below SDPA''s. More sinks -> more');
  WriteLn('absorbed mass -> lower real-key noise. This is exactly the');
  WriteLn('StreamingLLM attention-sink stabilisation claim.');

  Input.Free;
end.
