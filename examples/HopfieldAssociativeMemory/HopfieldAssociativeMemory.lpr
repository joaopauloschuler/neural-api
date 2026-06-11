program HopfieldAssociativeMemory;
(*
HopfieldAssociativeMemory: a classic ONE-SHOT associative-recall demo of the
continuous modern-Hopfield layer (TNNetModernHopfield / the
TNNet.AddModernHopfieldRetrieval builder, Ramsauer et al. 2020 "Hopfield
Networks is All You Need", arXiv:2008.02217).

Idea: a modern continuous Hopfield network is an ENERGY-BASED associative
memory. We STORE a handful of binary 8x8 "pixel" patterns in the layer's
learnable pattern bank, then present a HALF-MASKED (bottom rows blanked) and
NOISE-CORRUPTED version of one stored pattern as the query. The layer iterates

    xi := X^T * softmax(beta * X * xi)

for K update steps. The whole point of the layer is that K>1 SHARPENS the
retrieval toward the single nearest stored memory, so a few iterations clean the
corrupted query back to the exact stored pattern, while K=1 (ordinary one-pass
attention against the bank) only produces a blurry softmax-average of several
patterns and does NOT complete the pattern.

What it prints, for each test query:
  - the corrupted query (as ascii pixels),
  - the K=1 retrieval, its Hamming distance and continuous L2 distance to the
    true stored pattern,
  - the K=3 retrieval and the same two distances,
and a final summary table. The continuous L2 distance is the regime-robust
headline (iterating always pulls the iterate closer to one stored memory);
the Hamming column shows the sign-thresholded recall. Pixel values live in
{-1,+1}; "on" pixels (>0) print as '#', "off" as '.'.

No training: the patterns are written DIRECTLY into the layer's bank (the layer
is used purely as the retrieval dynamics here). beta is set deliberately SOFT so
a single retrieval pass (K=1) only blends several stored patterns while the
iterated retrieval (K=3) sharpens toward the nearest one. Small CPU toy, well
under a minute.

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
  GRID    = 8;             // 8x8 pixel patterns
  DIM     = GRID * GRID;   // d = 64
  NPAT    = 4;             // number of stored memories
  BETA    = 0.08;          // inverse temperature: deliberately SOFT so a single
                           // retrieval pass (K=1) only blends patterns, while
                           // iterating (K=3) sharpens to one stored memory
  SEED    = 42;
  NOISE_FLIPS = 6;         // random pixel flips on top of the half-mask

// The four stored 8x8 patterns, drawn as ascii art ('#' = +1, '.' = -1).
type
  TPatternArt = array[0..GRID - 1] of string;

const
  PatternArt: array[0..NPAT - 1] of TPatternArt = (
    // 0: a plus sign
    ( '...##...',
      '...##...',
      '...##...',
      '########',
      '########',
      '...##...',
      '...##...',
      '...##...' ),
    // 1: a hollow box
    ( '########',
      '#......#',
      '#......#',
      '#......#',
      '#......#',
      '#......#',
      '#......#',
      '########' ),
    // 2: an X
    ( '#......#',
      '.#....#.',
      '..#..#..',
      '...##...',
      '...##...',
      '..#..#..',
      '.#....#.',
      '#......#' ),
    // 3: diagonal stripes
    ( '#...#...',
      '.#...#..',
      '..#...#.',
      '...#...#',
      '#...#...',
      '.#...#..',
      '..#...#.',
      '...#...#' )
  );

var
  Patterns: array[0..NPAT - 1] of TNNetVolume; // (1,1,DIM) ground-truth memories

// ---------------------------------------------------------------------------
// Build the {-1,+1} memory volumes from the ascii art.
procedure BuildPatterns;
var
  p, r, c, idx: integer;
begin
  for p := 0 to NPAT - 1 do
  begin
    Patterns[p] := TNNetVolume.Create(1, 1, DIM);
    for r := 0 to GRID - 1 do
      for c := 0 to GRID - 1 do
      begin
        idx := r * GRID + c;
        if PatternArt[p][r][c + 1] = '#'
          then Patterns[p].Raw[idx] := 1.0
          else Patterns[p].Raw[idx] := -1.0;
      end;
  end;
end;

procedure FreePatterns;
var p: integer;
begin
  for p := 0 to NPAT - 1 do Patterns[p].Free;
end;

// Print a DIM-vector (taken from V, channel-contiguous) as an 8x8 ascii grid.
// When Ref<>nil, each printed line is annotated with the same row of Ref so two
// grids can be compared side by side by the reader.
procedure PrintGrid(const Caption: string; V: TNNetVolume);
var r, c: integer; line: string;
begin
  WriteLn(Caption);
  for r := 0 to GRID - 1 do
  begin
    line := '    ';
    for c := 0 to GRID - 1 do
      if V.Raw[r * GRID + c] > 0 then line := line + '#' else line := line + '.';
    WriteLn(line);
  end;
end;

// Hamming distance (count of sign disagreements) between a retrieved vector and
// a ground-truth pattern, both interpreted as {-1,+1} via sign.
function HammingToPattern(Retrieved, Truth: TNNetVolume): integer;
var i, cnt: integer;
begin
  cnt := 0;
  for i := 0 to DIM - 1 do
    if (Retrieved.Raw[i] > 0) <> (Truth.Raw[i] > 0) then Inc(cnt);
  Result := cnt;
end;

// Continuous L2 distance between the (real-valued) retrieval and the {-1,+1}
// stored pattern. This is the regime-robust metric: more update steps move the
// iterate closer to a single stored memory regardless of the sign threshold.
function L2ToPattern(Retrieved, Truth: TNNetVolume): TNeuralFloat;
var i: integer; d, acc: TNeuralFloat;
begin
  acc := 0;
  for i := 0 to DIM - 1 do
  begin
    d := Retrieved.Raw[i] - Truth.Raw[i];
    acc := acc + d * d;
  end;
  Result := Sqrt(acc);
end;

// Make a corrupted query from stored pattern p: blank the bottom half (set to 0,
// the "unknown" value) and flip a few random pixels in the visible top half.
procedure MakeCorruptedQuery(p: integer; Q: TNNetVolume);
var r, c, f, idx: integer;
begin
  Q.Copy(Patterns[p]);
  // Mask the bottom half of the image to 0 (information removed).
  for r := GRID div 2 to GRID - 1 do
    for c := 0 to GRID - 1 do
      Q.Raw[r * GRID + c] := 0.0;
  // Flip a handful of visible (top-half) pixels as noise.
  for f := 0 to NOISE_FLIPS - 1 do
  begin
    idx := Random(GRID div 2) * GRID + Random(GRID);
    Q.Raw[idx] := -Q.Raw[idx];
  end;
end;

// Build a 1-position-sequence net whose single layer is a ModernHopfield with
// the given number of update steps, its bank loaded with the stored patterns.
function BuildHopfieldNet(KSteps: integer): TNNet;
var
  L: TNNetModernHopfield;
  p, i: integer;
begin
  Result := TNNet.Create();
  Result.AddLayer(TNNetInput.Create(1, 1, DIM)); // SeqLen=1 query
  L := Result.AddModernHopfieldRetrieval(NPAT, KSteps, BETA) as TNNetModernHopfield;
  // Write the stored memories DIRECTLY into the bank (pattern p is row p, its
  // DIM values depth-contiguous at GetRawPtr(p,0,0)).
  for p := 0 to NPAT - 1 do
    for i := 0 to DIM - 1 do
      L.Neurons[0].Weights[p, 0, i] := Patterns[p].Raw[i];
end;

var
  Net1, Net3: TNNet;
  Query: TNNetVolume;
  Out1, Out3: TNNetVolume;
  p, sum1, sum3: integer;
  h1, h3: integer;
  l2sum1, l2sum3, l1, l3: TNeuralFloat;
begin
  SetExceptionMask([exInvalidOp, exDenormalized, exZeroDivide, exOverflow,
    exUnderflow, exPrecision]);
  RandSeed := SEED;
  WriteLn('Modern Hopfield associative memory - one-shot recall demo');
  WriteLn('  grid=', GRID, 'x', GRID, ' (d=', DIM, ')  stored patterns=', NPAT,
    '  beta=', BETA:0:3);
  WriteLn('  query = bottom-half masked + ', NOISE_FLIPS, ' random pixel flips');
  WriteLn('  contrast: K=1 (one-pass attention) vs K=3 (iterated retrieval)');
  WriteLn;

  BuildPatterns;
  Query := TNNetVolume.Create(1, 1, DIM);
  Out1 := TNNetVolume.Create(1, 1, DIM);
  Out3 := TNNetVolume.Create(1, 1, DIM);
  Net1 := BuildHopfieldNet(1);
  Net3 := BuildHopfieldNet(3);
  sum1 := 0;
  sum3 := 0;
  l2sum1 := 0;
  l2sum3 := 0;
  try
    for p := 0 to NPAT - 1 do
    begin
      MakeCorruptedQuery(p, Query);

      Net1.Compute(Query);
      Out1.Copy(Net1.GetLastLayer.Output);
      Net3.Compute(Query);
      Out3.Copy(Net3.GetLastLayer.Output);

      h1 := HammingToPattern(Out1, Patterns[p]);
      h3 := HammingToPattern(Out3, Patterns[p]);
      l1 := L2ToPattern(Out1, Patterns[p]);
      l3 := L2ToPattern(Out3, Patterns[p]);
      sum1 := sum1 + h1;
      sum3 := sum3 + h3;
      l2sum1 := l2sum1 + l1;
      l2sum3 := l2sum3 + l3;

      WriteLn('================ stored pattern #', p, ' ================');
      PrintGrid('  corrupted query (masked + noised):', Query);
      PrintGrid('  K=1 retrieval (one-pass attention):', Out1);
      WriteLn('    -> Hamming = ', h1, '   L2 to true pattern = ', l1:0:3);
      PrintGrid('  K=3 retrieval (iterated Hopfield):', Out3);
      WriteLn('    -> Hamming = ', h3, '   L2 to true pattern = ', l3:0:3);
      WriteLn;
    end;

    WriteLn('==================== SUMMARY ====================');
    WriteLn('  total over ', NPAT, ' queries        Hamming      L2 distance');
    WriteLn('    K=1 (attention)        ', sum1:7, '       ', l2sum1:9:3);
    WriteLn('    K=3 (iterated)         ', sum3:7, '       ', l2sum3:9:3);
    WriteLn('=================================================');
    // The continuous L2 distance is the regime-robust headline: iterating the
    // retrieval always pulls the iterate closer to a single stored memory.
    if (l2sum3 < l2sum1) and (sum3 <= sum1) then
      WriteLn('Verdict: iterating the retrieval (K=3) CLEANED the queries back ',
        'toward the stored memories better than one-pass attention (K=1) - ',
        'the defining behaviour of a modern Hopfield associative memory.')
    else
      WriteLn('Verdict: K=3 did not beat K=1 on this seed (try a larger beta or ',
        'more update steps).');
  finally
    Net1.Free;
    Net3.Free;
    Query.Free;
    Out1.Free;
    Out3.Free;
    FreePatterns;
  end;
end.
