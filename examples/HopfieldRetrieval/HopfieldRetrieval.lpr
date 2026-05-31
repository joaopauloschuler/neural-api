program HopfieldRetrieval;
(*
HopfieldRetrieval: a MODERN Hopfield network run as a single step of
softmax attention (Ramsauer et al. 2020, "Hopfield Networks is All You
Need", https://arxiv.org/abs/2008.02217).

The central observation of that paper is that the update rule of a
continuous, exponential-capacity Hopfield network is EXACTLY scaled
dot-product attention. Store K patterns as the rows of a matrix
X (K x d). Given a (possibly corrupted/partial) query vector q in R^d,
one Hopfield retrieval step is:

    retrieved = X^T * softmax(beta * X * q)

i.e. attention with the query q, keys = values = the stored patterns X,
and an inverse-temperature beta that plays the role of the 1/sqrt(d)
scaling in a transformer. The softmax produces a distribution over the
K stored patterns; the retrieved vector is the corresponding convex
combination of those patterns.

The qualitative behaviour is governed entirely by beta:

  - LOW beta  -> the softmax is near-uniform, so the retrieved vector is
                 a blurry AVERAGE of all stored patterns (a metastable
                 "global" fixed point). Useless as a memory.
  - HIGH beta -> the softmax saturates onto the single nearest stored
                 pattern, so ONE step SNAPS the corrupted query cleanly
                 to that pattern (a sharp, pattern-separated fixed
                 point). This is the associative-memory / pattern-
                 completion regime.

With well-separated patterns and a large enough beta, retrieval is a
single forward step: no training, no iteration, no backprop.

This demo (route B in the brief: explicit TNNetVolume math, so beta is
literally a number in the score and the whole thing is self-evidently
correct) does the following, all in-code with a fixed RandSeed:

  1. Builds K random BIPOLAR (+/-1) patterns of dimension d.
  2. For each pattern, corrupts it by flipping a fraction of its signs,
     then runs ONE Hopfield step softmax(beta * X q) X.
  3. Reports, per pattern: cosine similarity between the retrieved
     vector and the TRUE stored pattern, and whether the argmax of the
     attention weights points at the correct stored pattern.
  4. Sweeps beta (showing the blurry-average -> clean-snap transition)
     and sweeps corruption level.
  5. Prints NaN/Inf-guarded PASS/FAIL sanity checks: at high beta and
     low corruption every pattern must recover with cosine > 0.95 AND
     every query's argmax-attention must select the correct pattern.

Pure CPU, no external dataset, forward-only, finishes in ~1 second.

Copyright (C) 2026 Joao Paulo Schwarz Schuler

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
*)

{$mode objfpc}{$H+}

uses {$IFDEF UNIX} {$IFDEF UseCThreads}
  cthreads, {$ENDIF} {$ENDIF}
  Classes, SysUtils, Math,
  neuralvolume;

const
  cK         = 6;        // number of stored patterns
  cD         = 32;       // pattern dimension
  cSeed      = 42;
  cPassBeta  = 8.0;      // beta used for the headline sanity check
  cPassFlip  = 0.15;     // corruption (sign-flip fraction) for the check
  cPassCos   = 0.95;     // cosine-sim threshold for "recovered"

type
  // K stored patterns, each a length-d vector. Index [k, 0, c].
  TPatternBank = array[0..cK - 1] of array[0..cD - 1] of TNeuralFloat;

var
  GBank: TPatternBank;   // the stored patterns X (bipolar +/-1)

// ---------------------------------------------------------------------
// Pattern bank
// ---------------------------------------------------------------------

// Build K random bipolar (+/-1) patterns. Bipolar random vectors in R^d
// are near-orthogonal for d >> 1, so they are well separated and make a
// good associative-memory bank.
procedure BuildBank;
var
  K, C: integer;
begin
  for K := 0 to cK - 1 do
    for C := 0 to cD - 1 do
      if Random < 0.5 then GBank[K, C] := -1.0 else GBank[K, C] := 1.0;
end;

// Cosine similarity between two length-d vectors.
function CosineSim(const A, B: array of TNeuralFloat): TNeuralFloat;
var
  C: integer;
  Dot, NA, NB: Double;
begin
  Dot := 0; NA := 0; NB := 0;
  for C := 0 to cD - 1 do
  begin
    Dot := Dot + A[C] * B[C];
    NA  := NA + A[C] * A[C];
    NB  := NB + B[C] * B[C];
  end;
  if (NA <= 0) or (NB <= 0) then
    Result := 0
  else
    Result := Dot / Sqrt(NA * NB);
end;

// ---------------------------------------------------------------------
// Corruption: flip the sign of a fraction of the dimensions.
// ---------------------------------------------------------------------
procedure CorruptPattern(K: integer; FlipFrac: TNeuralFloat;
  out Q: array of TNeuralFloat);
var
  C: integer;
begin
  for C := 0 to cD - 1 do
  begin
    Q[C] := GBank[K, C];
    if Random < FlipFrac then Q[C] := -Q[C];
  end;
end;

// ---------------------------------------------------------------------
// THE Hopfield step: retrieved = X^T softmax(beta * X q).
// Returns the retrieved vector in Retrieved and the argmax stored-pattern
// index in BestK (the pattern the attention weights point at).
// ---------------------------------------------------------------------
procedure HopfieldStep(const Q: array of TNeuralFloat; Beta: TNeuralFloat;
  out Retrieved: array of TNeuralFloat; out BestK: integer;
  out BestW: TNeuralFloat);
var
  K, C: integer;
  Score: array[0..cK - 1] of Double;
  W: array[0..cK - 1] of Double;
  MaxScore, SumExp, Dot: Double;
begin
  // scores[k] = beta * <X[k], q>
  MaxScore := -1e30;
  for K := 0 to cK - 1 do
  begin
    Dot := 0;
    for C := 0 to cD - 1 do
      Dot := Dot + GBank[K, C] * Q[C];
    Score[K] := Beta * Dot;
    if Score[K] > MaxScore then MaxScore := Score[K];
  end;

  // softmax over k (max-subtracted for numerical stability)
  SumExp := 0;
  for K := 0 to cK - 1 do
  begin
    W[K] := Exp(Score[K] - MaxScore);
    SumExp := SumExp + W[K];
  end;
  for K := 0 to cK - 1 do
    W[K] := W[K] / SumExp;

  // retrieved = sum_k w[k] * X[k]
  for C := 0 to cD - 1 do
    Retrieved[C] := 0;
  for K := 0 to cK - 1 do
    for C := 0 to cD - 1 do
      Retrieved[C] := Retrieved[C] + W[K] * GBank[K, C];

  // argmax weight = the pattern the attention selected
  BestK := 0; BestW := W[0];
  for K := 1 to cK - 1 do
    if W[K] > BestW then begin BestW := W[K]; BestK := K; end;
end;

function SafeF(V: TNeuralFloat; Dec: integer): string;
begin
  if IsNan(V) or IsInfinite(V) then
    Result := 'NaN'
  else
    Result := FloatToStrF(V, ffFixed, 8, Dec);
end;

// ---------------------------------------------------------------------
// Per-pattern retrieval table at a fixed beta / corruption level.
// Returns whether ALL patterns recovered (cos > thresh AND argmax right).
// ---------------------------------------------------------------------
function RetrievalTable(Beta, FlipFrac: TNeuralFloat; Verbose: boolean):
  boolean;
var
  K, BestK: integer;
  Q, Retrieved: array[0..cD - 1] of TNeuralFloat;
  TrueBank: array[0..cD - 1] of TNeuralFloat;
  Cos, BestW: TNeuralFloat;
  CosNoisy: TNeuralFloat;
  C: integer;
  AllOk: boolean;
begin
  AllOk := True;
  if Verbose then
  begin
    WriteLn(Format('  beta=%s  flip=%s', [SafeF(Beta, 2), SafeF(FlipFrac, 2)]));
    WriteLn(Format('    %-8s %10s %10s %8s %8s %6s',
      ['pattern', 'cos(in)', 'cos(out)', 'argmax', 'weight', 'ok']));
  end;
  for K := 0 to cK - 1 do
  begin
    CorruptPattern(K, FlipFrac, Q);
    HopfieldStep(Q, Beta, Retrieved, BestK, BestW);
    for C := 0 to cD - 1 do TrueBank[C] := GBank[K, C];
    Cos      := CosineSim(Retrieved, TrueBank);
    CosNoisy := CosineSim(Q, TrueBank);
    if (IsNan(Cos) or IsInfinite(Cos) or (Cos < cPassCos) or (BestK <> K)) then
      AllOk := False;
    if Verbose then
      WriteLn(Format('    p%-7d %10s %10s %8d %8s %6s',
        [K, SafeF(CosNoisy, 3), SafeF(Cos, 3), BestK, SafeF(BestW, 3),
         BoolToStr((Cos >= cPassCos) and (BestK = K), 'yes', 'no')]));
  end;
  Result := AllOk;
end;

// Mean retrieval cosine over all patterns at a given beta/flip (for the
// sweep tables; one fresh corrupted query per pattern).
function MeanRetrievalCos(Beta, FlipFrac: TNeuralFloat): TNeuralFloat;
var
  K, BestK, C: integer;
  Q, Retrieved, TrueBank: array[0..cD - 1] of TNeuralFloat;
  BestW, Cos: TNeuralFloat;
  Sum: Double;
begin
  Sum := 0;
  for K := 0 to cK - 1 do
  begin
    CorruptPattern(K, FlipFrac, Q);
    HopfieldStep(Q, Beta, Retrieved, BestK, BestW);
    for C := 0 to cD - 1 do TrueBank[C] := GBank[K, C];
    Cos := CosineSim(Retrieved, TrueBank);
    Sum := Sum + Cos;
  end;
  Result := Sum / cK;
end;

procedure Run;
var
  LowCos, HighCos: TNeuralFloat;
  K, J: integer;
  Betas: array[0..4] of TNeuralFloat = (0.1, 0.5, 1.0, 4.0, 16.0);
  Flips: array[0..4] of TNeuralFloat = (0.0, 0.1, 0.2, 0.3, 0.4);
  C: integer;
  PassHigh, PassLow: boolean;
  Dot: Double;
  MaxAbsCorr: TNeuralFloat;
begin
  WriteLn('Modern Hopfield network as ONE softmax-attention step.');
  WriteLn('Ramsauer et al. 2020, "Hopfield Networks is All You Need"');
  WriteLn('https://arxiv.org/abs/2008.02217');
  WriteLn;
  WriteLn(Format('Stored bank: K=%d bipolar (+/-1) patterns of dim d=%d  RandSeed=%d',
    [cK, cD, cSeed]));
  WriteLn('Retrieval rule:  retrieved = X^T softmax(beta * X q)');
  WriteLn('  (attention with query q, keys = values = stored patterns X,');
  WriteLn('   inverse temperature beta playing the role of 1/sqrt(d)).');
  WriteLn;

  // Report bank separation: worst-case |cosine| between distinct patterns.
  MaxAbsCorr := 0;
  for K := 0 to cK - 1 do
    for J := K + 1 to cK - 1 do
    begin
      Dot := 0;
      for C := 0 to cD - 1 do Dot := Dot + GBank[K, C] * GBank[J, C];
      // patterns are +/-1 so each has norm sqrt(d); cosine = dot / d
      if Abs(Dot / cD) > MaxAbsCorr then MaxAbsCorr := Abs(Dot / cD);
    end;
  WriteLn(Format('Worst-case |cosine| between distinct stored patterns: %s',
    [SafeF(MaxAbsCorr, 3)]));
  WriteLn('(near 0 => well separated => clean single-step retrieval).');
  WriteLn;

  // --- Headline per-pattern table at the sanity-check operating point ---
  WriteLn('=== Per-pattern retrieval (single Hopfield step) ===');
  RandSeed := cSeed + 1;  // fresh corruption stream, reproducible
  PassHigh := RetrievalTable(cPassBeta, cPassFlip, True);
  WriteLn;

  // --- Low-beta contrast: blurry average, no clean snap ---
  WriteLn('=== Same queries, LOW beta (blurry average, NOT a memory) ===');
  RandSeed := cSeed + 1;  // SAME corruption stream as above for contrast
  RetrievalTable(0.1, cPassFlip, True);
  WriteLn;

  // --- Beta sweep: mean retrieval cosine at fixed corruption ---
  WriteLn(Format('=== Beta sweep (mean retrieval cosine, flip=%s) ===',
    [SafeF(cPassFlip, 2)]));
  WriteLn('  blurry average <----------------------------> clean snap');
  Write(Format('    %-12s', ['beta']));
  for K := 0 to High(Betas) do Write(Format('%9s', [SafeF(Betas[K], 1)]));
  WriteLn;
  Write(Format('    %-12s', ['mean_cos']));
  for K := 0 to High(Betas) do
  begin
    RandSeed := cSeed + 100;
    Write(Format('%9s', [SafeF(MeanRetrievalCos(Betas[K], cPassFlip), 3)]));
  end;
  WriteLn;
  WriteLn;

  // --- Corruption sweep at fixed high beta ---
  WriteLn(Format('=== Corruption sweep (mean retrieval cosine, beta=%s) ===',
    [SafeF(cPassBeta, 1)]));
  Write(Format('    %-12s', ['flip_frac']));
  for K := 0 to High(Flips) do Write(Format('%9s', [SafeF(Flips[K], 2)]));
  WriteLn;
  Write(Format('    %-12s', ['mean_cos']));
  for K := 0 to High(Flips) do
  begin
    RandSeed := cSeed + 200;
    Write(Format('%9s', [SafeF(MeanRetrievalCos(cPassBeta, Flips[K]), 3)]));
  end;
  WriteLn;
  WriteLn;

  // --- Sanity checks ---
  WriteLn('=== Sanity checks ===');

  // (1) high beta + low corruption: every pattern recovers, argmax correct
  if PassHigh then
    WriteLn(Format('[PASS] beta=%s flip=%s: all %d patterns recovered ' +
      '(cos>%s) and argmax-attention picked the correct stored pattern.',
      [SafeF(cPassBeta, 1), SafeF(cPassFlip, 2), cK, SafeF(cPassCos, 2)]))
  else
    WriteLn('[FAIL] high-beta low-corruption retrieval did not recover all patterns.');

  // (2) the blurry-average -> clean-snap transition is real: at the
  // sanity-check corruption, HIGH beta gives essentially perfect
  // recovery (>0.99) while LOW beta is a visibly blurry average
  // (mean cosine to the true pattern strictly worse, and < 0.97).
  RandSeed := cSeed + 300;
  LowCos  := MeanRetrievalCos(0.1, cPassFlip);
  RandSeed := cSeed + 300;
  HighCos := MeanRetrievalCos(cPassBeta, cPassFlip);
  PassLow := (not IsNan(LowCos)) and (not IsNan(HighCos)) and
             (HighCos > 0.99) and (LowCos < 0.97) and (HighCos > LowCos);
  if PassLow then
    WriteLn(Format('[PASS] blurry->snap: low beta=0.1 is a blurry average ' +
      '(mean_cos=%s) while high beta=%s snaps cleanly (mean_cos=%s).',
      [SafeF(LowCos, 3), SafeF(cPassBeta, 1), SafeF(HighCos, 3)]))
  else
    WriteLn(Format('[FAIL] expected high-beta clean snap vs low-beta blur ' +
      '(low=%s high=%s).', [SafeF(LowCos, 3), SafeF(HighCos, 3)]));

  WriteLn;
  WriteLn('Modern-Hopfield retrieval is a single softmax-attention step: with a');
  WriteLn('well-separated bank and large beta, one step completes a corrupted');
  WriteLn('query to the nearest stored pattern.');
end;

begin
  RandSeed := cSeed;
  BuildBank;
  Run;
end.
