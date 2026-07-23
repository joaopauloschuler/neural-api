program IntraLayerThreadingBench;
(*
IntraLayerThreadingBench: validates the intra-layer threading decision.

  * Experiment A sweeps single-layer FC / pointwise / conv shapes from tiny to
    large - plus the token-wise transformer layers (SwiGLU / TokenRMSNorm /
    RotaryEmbedding), the channel-chunked TNNetDepthwiseConv1D and the k-head-
    chunked TNNetGatedDeltaNet (both in full-sweep AND incremental-decode
    mode) at decode (seq=1) and prefill-like sequence
    lengths - reports each shape's neuron count, whether it is eligible to chunk
    (the real ChunkEligible() verdict), the OFF-vs-ON speedup, and a bit-parity
    check (threaded output MUST equal serial output - only independent outputs
    are partitioned). It is run twice, once with the low-memory inference
    contract off and once on, since conv/pointwise chunk in both modes.
  * Experiment B (no longer run - kept in the source for reference) took the
    small shapes from A and swept them across NN.StartThreadWorkers()
    policies - hot-core counts 1..4 crossed with cool-down timeouts 0..2
    seconds - since the small end is where the pool policy actually moves
    the OFF-vs-ON speedup.
  * Experiment C (no longer run - kept in the source for reference) timed a
    few realistic multi-layer nets OFF vs ON end to end.

ChunkEligible returns true for every size, so a parallel pass always chunks.
Experiment A measures where that pays: the tiny shapes at the top of each block
are the ones at risk of regressing (speedup < 1.00x) when the work is too small
to repay scheduler/pool overhead, while the large shapes should scale toward the
core count.

A speedup > 1.00x means threading pays off at that size; <= 1.00x means the
serial path would have been faster on that shape.

Experiments A and B each close with a SUMMARY block of geometric means, so a
run is comparable across machines and not only within itself: A reports the
all-shapes / chunk-eligible / not-eligible geomeans plus per-family subtotals
and approximate GOP/s throughput, then a memory-mode comparison across its two
passes; B reports each small shape's best pool policy, its sensitivity to the
policy choice, and a ranking of the policies. Summaries carry a shapeset
version (SHAPESET_A / SHAPESET_B) and the build tag - bump the version whenever
a shape is added, removed or resized, or old numbers stop being comparable.

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
any later version.

Coded by Claude (AI).
*)

{$mode objfpc}{$H+}

uses {$IFDEF UNIX} cthreads, {$ENDIF}
  Classes, SysUtils, DateUtils, Math,
  neuralnetwork,
  neuralvolume,
  neuralthread;

// Warm up, then best-of-5 timing (ms per Compute). Best-of rejects scheduler
// jitter / turbo wobble far better than an average on a noisy box. Parallel
// MUST be threaded through to Compute: the compute path itself drives the chunk
// path (ComputeParallel enables intra-layer threading, ComputeSerial disables
// it), so timing the threaded case with the default (serial) overload measures
// the serial path and nothing threaded.
function BestTimeMs(NN: TNNet; MyIn: TNNetVolume; Parallel: boolean): double;
var
  r, i, iters: integer;
  StartT: TDateTime;
  el, best: double;
begin
  iters := 1;
  repeat
    StartT := Now();
    for i := 1 to iters do NN.Compute(MyIn, 0, Parallel);
    el := MilliSecondsBetween(Now(), StartT);
    if el >= 150 then break;
    if el < 1 then iters := iters * 8
    else iters := Max(iters + 1, Round(iters * (180 / el)));
  until false;
  iters := Max(1, Round(iters * (80 / Max(el, 1))));
  best := 1e30;
  for r := 0 to 4 do
  begin
    StartT := Now();
    for i := 1 to iters do NN.Compute(MyIn, 0, Parallel);
    el := MilliSecondsBetween(Now(), StartT) / iters;
    if el < best then best := el;
  end;
  Result := best;
end;

// ============================ Summary machinery ===============================
// Shapeset versions. A summary line is only comparable across boxes/commits
// when the shape list behind it is identical, so every summary carries its
// version - BUMP THESE whenever a shape is added, removed or resized.
const
  SHAPESET_A = 'A/v1';
  SHAPESET_B = 'B/v1';

// Build identity, so pasted results are self-describing (a run built without
// AVX is not comparable with one built with it).
function BuildTag: string;
begin
  Result := '';
  {$IFDEF AVX512}
  Result := Result + 'AVX512 ';
  {$ELSE}{$IFDEF AVX2}
  Result := Result + 'AVX2 ';
  {$ELSE}{$IFDEF AVX}
  Result := Result + 'AVX ';
  {$ELSE}
  Result := Result + 'no-SIMD ';
  {$ENDIF}{$ENDIF}{$ENDIF}
  {$IFDEF OpenCL}
  Result := Result + 'OpenCL ';
  {$ENDIF}
  {$IFDEF Debug}
  Result := Result + 'Debug ';
  {$ENDIF}
  Result := Result + 'FPC ' + {$I %FPCVERSION%} + ' ' +
    {$I %FPCTARGETOS%} + '-' + {$I %FPCTARGETCPU%};
end;

// Geometric-mean accumulator. Times across the sweep span orders of magnitude
// (a 64x64 FC against a 512-token GDN), so an arithmetic mean would report the
// largest shape and nothing else. The geomean weights every shape equally and
// has the property that geomean(off)/geomean(on) IS geomean(per-shape speedup),
// so the summary speedup never disagrees with the speedup column above it.
type
  TAggr = record
    N: integer;         // shapes accumulated
    NTp: integer;       // shapes with a usable op count (throughput columns)
    LOff, LOn: double;      // sum of ln(time ms)
    LTpOff, LTpOn: double;  // sum of ln(GOP/s)
  end;

procedure AggrReset(var G: TAggr);
begin
  G.N := 0; G.NTp := 0;
  G.LOff := 0; G.LOn := 0; G.LTpOff := 0; G.LTpOn := 0;
end;

// Ops/(ms * 1e6) = GOP/s. Times are clamped away from zero: a shape that timed
// below the clock's resolution would otherwise take Ln(0) down with it.
procedure AggrAdd(var G: TAggr; Off, On_, Ops: double);
begin
  Off := Max(Off, 1e-6); On_ := Max(On_, 1e-6);
  Inc(G.N);
  G.LOff := G.LOff + Ln(Off);
  G.LOn := G.LOn + Ln(On_);
  if Ops > 0 then
  begin
    Inc(G.NTp);
    G.LTpOff := G.LTpOff + Ln(Ops / (Off * 1e6));
    G.LTpOn := G.LTpOn + Ln(Ops / (On_ * 1e6));
  end;
end;

function GeoMean(LogSum: double; N: integer): double;
begin
  if N <= 0 then Result := 0 else Result := Exp(LogSum / N);
end;

function AggrOff(const G: TAggr): double;
begin
  Result := GeoMean(G.LOff, G.N);
end;

function AggrOn(const G: TAggr): double;
begin
  Result := GeoMean(G.LOn, G.N);
end;

procedure WriteAggrHeader;
begin
  WriteLn(Format('  %-22s %6s %10s %10s %8s %10s %9s',
    ['', 'shapes', 'off ms', 'on ms', 'speedup', 'off GOP/s', 'on GOP/s']));
end;

procedure WriteAggrRow(const Tag: string; const G: TAggr);
var off, on_, sp: double;
begin
  if G.N = 0 then exit;
  off := AggrOff(G); on_ := AggrOn(G);
  if on_ > 0 then sp := off / on_ else sp := 0;
  WriteLn(Format('  %-22s %6d %10.3f %10.3f %7.2fx %10.2f %9.2f',
    [Tag, G.N, off, on_, sp,
     GeoMean(G.LTpOff, G.NTp), GeoMean(G.LTpOn, G.NTp)]));
end;

// =============================== Experiment A =================================
type
  TShape = record Name: string; Kind, A, B, C, D: integer; end;
var
  ShA: array of TShape;

// Kind 0 = FullConnect(input A -> B neurons)
//      1 = Convolution(input AxAxB, C features, DxD kernel, stride 1, pad 0)
//      2 = PointwiseConv(input AxAxB -> C features)
//      3 = SwiGLU(input Ax1x2B -> Ax1xB; A = seqLen, B = half depth)
//      4 = TokenRMSNorm(input Ax1xB; A = seqLen, B = depth)
//      5 = RotaryEmbedding(input Ax1xB; A = seqLen, B = depth, B even)
//      6 = DepthwiseConv1D(input Ax1xB; A = seqLen, B = channels, C = kernel)
//      7 = DepthwiseConv1D as 6 but in INCREMENTAL-DECODE mode (stateful
//          history path - the kernel real decode steps run)
//      8 = GatedDeltaNet(A = seqLen, B = Hk k-heads, C = Hv v-heads,
//          D = head dim for BOTH Dk and Dv; input depth derived)
//      9 = GatedDeltaNet as 8 but in INCREMENTAL-DECODE mode (in-place
//          per-head state carry - the real autoregressive decode path)
procedure AddA(const N: string; K, A, B, C, D: integer);
var idx: integer;
begin
  idx := Length(ShA);
  SetLength(ShA, idx + 1);
  ShA[idx].Name := N; ShA[idx].Kind := K;
  ShA[idx].A := A; ShA[idx].B := B; ShA[idx].C := C; ShA[idx].D := D;
end;

// Approximate op count for ONE forward pass of the shape's single compute
// layer (one MAC = one op). This drives the GOP/s summary columns only: raw ms
// is comparable across boxes only when the shape list is identical, whereas a
// per-shape work/time figure stays meaningful when a shape is resized. It is a
// consistent work proxy, NOT a certified FLOP count - the element-wise layers
// (SwiGLU/RMSN/RoPE) have no MACs to count, so their op weights are estimates
// and their absolute GOP/s should be read as a within-family comparison.
function WorkOps(const S: TShape): double;
var outXY: double;
begin
  case S.Kind of
    // FC: in x out MACs.
    0: Result := double(S.A) * S.B;
    // Conv, stride 1, no pad: out positions x features x kernel x in depth.
    1: begin
         outXY := Sqr(double(Max(1, S.A - S.D + 1)));
         Result := outXY * S.C * S.D * S.D * S.B;
       end;
    // Pointwise: positions x in depth x out features.
    2: Result := double(S.A) * S.A * S.B * S.C;
    // SwiGLU: one gated output per element (2B in -> B out).
    3: Result := double(S.A) * S.B;
    // TokenRMSNorm: sum-of-squares pass + scale pass.
    4: Result := double(S.A) * S.B * 2;
    // RotaryEmbedding: ~3 multiplies per element (pair rotation).
    5: Result := double(S.A) * S.B * 3;
    // DepthwiseConv1D: C taps per channel per token.
    6, 7: Result := double(S.A) * S.B * S.C;
    // GatedDeltaNet: per token, per v-head, ~4 sweeps of the Dk x Dv state.
    8, 9: Result := double(S.A) * S.C * S.D * S.D * 4;
  else
    Result := 0;
  end;
end;

// Family buckets for the per-family subtotals. One global geomean hides why a
// box wins: FC/PW/Conv lean on cache and SIMD width, the token-wise layers on
// memory bandwidth, and the decode bucket is the ChatTerminal-relevant path.
const
  FAM_COUNT = 5;
  FAM_NAMES: array[0..FAM_COUNT - 1] of string =
    ('FullConnect', 'Convolution', 'Pointwise', 'Norm/RoPE/SwiGLU',
     'Decode (Dw/GDN)');

function FamilyOf(Kind: integer): integer;
begin
  case Kind of
    0: Result := 0;
    1: Result := 1;
    2: Result := 2;
    3, 4, 5: Result := 3;
  else Result := 4;  // 6..9: DepthwiseConv1D / GatedDeltaNet, sweep and decode
  end;
end;

function BuildOne(const S: TShape): TNNet;
var NN: TNNet;
begin
  NN := TNNet.Create();
  case S.Kind of
    0: begin NN.AddLayer(TNNetInput.Create(S.A));
             NN.AddLayer(TNNetFullConnectReLU.Create(S.B)); end;
    1: begin NN.AddLayer(TNNetInput.Create(S.A, S.A, S.B));
             NN.AddLayer(TNNetConvolutionReLU.Create(S.C, S.D, 0, 1)); end;
    2: begin NN.AddLayer(TNNetInput.Create(S.A, S.A, S.B));
             NN.AddLayer(TNNetPointwiseConvReLU.Create(S.C)); end;
    3: begin NN.AddLayer(TNNetInput.Create(S.A, 1, 2 * S.B));
             NN.AddLayer(TNNetSwiGLU.Create()); end;
    4: begin NN.AddLayer(TNNetInput.Create(S.A, 1, S.B));
             NN.AddLayer(TNNetTokenRMSNorm.Create()); end;
    5: begin NN.AddLayer(TNNetInput.Create(S.A, 1, S.B));
             NN.AddLayer(TNNetRotaryEmbedding.Create()); end;
    6, 7:
       begin NN.AddLayer(TNNetInput.Create(S.A, 1, S.B));
             NN.AddLayer(TNNetDepthwiseConv1D.Create(S.C, {causal=}true)); end;
    8, 9:
       // Input depth layout [q|k|v|z|b|a] = 2*Hk*Dk + 2*Hv*Dv + 2*Hv (Dk=Dv=D).
       begin NN.AddLayer(TNNetInput.Create(S.A, 1,
               2 * S.B * S.D + 2 * S.C * S.D + 2 * S.C));
             NN.AddLayer(TNNetGatedDeltaNet.Create(S.B, S.C, S.D, S.D)); end;
  end;
  NN.GetLastLayer().InitDefault();
  // Kinds 7/9 time the stateful decode kernel (history/state carry), the path
  // a real autoregressive decode runs.
  if (S.Kind = 7) or (S.Kind = 9) then
    TNNetRecurrentDecodeBase(NN.GetLastLayer()).BeginIncrementalDecode();
  Result := NN;
end;

// The Experiment A shape sweep. Each block goes small -> large across FC, PW
// and Conv. Every shape chunks, so the small shapes are the probe: they show
// whether threading regresses (speedup < 1x) on work too small to repay the
// pool, while the large ones should scale toward core count. "Neurons" per
// shape is the last layer's neuron count: FC -> output neurons (B), PW/Conv ->
// features (C).
procedure BuildShapesA;
begin
  SetLength(ShA, 0);
  // FullConnect: input A -> B neurons.  neurons = B
  AddA('FC 64x64',          0, 64,   64,   0, 0);
  AddA('FC 128x128',        0, 128,  128,  0, 0);
  AddA('FC 256x256',        0, 256,  256,  0, 0);
  AddA('FC 512x512',        0, 512,  512,  0, 0);
  AddA('FC 768x768',        0, 768,  768,  0, 0);
  AddA('FC 1024x1024',      0, 1024, 1024, 0, 0);
  AddA('FC 1536x1536',      0, 1536, 1536, 0, 0);
  AddA('FC 2048x2048',      0, 2048, 2048, 0, 0);
  AddA('FC 3072x3072',      0, 3072, 3072, 0, 0);
  AddA('FC 768x3072 up',    0, 768,  3072, 0, 0);
  AddA('FC 3072x768 dn',    0, 3072, 768,  0, 0);
  // PointwiseConv: input AxAxB -> C features.  neurons = C
  AddA('Pw 8x8x64->64',     2, 8,  64,  64,  0);
  AddA('Pw 8x8x128->128',   2, 8,  128, 128, 0);
  AddA('Pw 32x32x256->256', 2, 32, 256, 256, 0);
  AddA('Pw 16x16x512->512', 2, 16, 512, 512, 0);
  // Convolution: input AxAxB, C features, DxD kernel.  neurons = C
  AddA('Conv 8x8x32 k3',    1, 8,  32,  32,  3);
  AddA('Conv 8x8x64 k3',    1, 8,  64,  64,  3);
  AddA('Conv 16x16x64 k3',  1, 16, 64,  128, 3);
  AddA('Conv 32x32x64 k3',  1, 32, 64,  64,  3);
  AddA('Conv 8x8x256 k3',   1, 8,  256, 256, 3);
  // Token-wise transformer layers: input A(seq) x 1 x B(depth). These report
  // their real ChunkEligible() verdict - 'no' with a ~1.00x speedup until each
  // gains a ComputeRange, then the same rows become the payoff measurement.
  // seq=1 rows are the KV-cache decode shape (the case that needs a finer-than-
  // token chunk axis); the larger seq rows are prefill-like.
  // SwiGLU: input Ax1x2B -> Ax1xB (token-wise gate).
  AddA('SwiGLU s1 d3072',   3, 1,   3072, 0, 0);
  AddA('SwiGLU s8 d3072',   3, 8,   3072, 0, 0);
  AddA('SwiGLU s64 d3072',  3, 64,  3072, 0, 0);
  AddA('SwiGLU s512 d1536', 3, 512, 1536, 0, 0);
  // TokenRMSNorm: per-token RMS reduction + gain over depth B.
  AddA('RMSN s1 d1024',     4, 1,   1024, 0, 0);
  AddA('RMSN s64 d1024',    4, 64,  1024, 0, 0);
  AddA('RMSN s512 d1024',   4, 512, 1024, 0, 0);
  // RotaryEmbedding: interleaved pair rotation, B/2 sincos pairs per token.
  AddA('RoPE s1 d128',      5, 1,   128,  0, 0);
  AddA('RoPE s64 d128',     5, 64,  128,  0, 0);
  AddA('RoPE s512 d1024',   5, 512, 1024, 0, 0);
  // DepthwiseConv1D: input A(seq) x 1 x B(channels), C-tap causal kernel.
  // neurons = B (one per channel = the chunk axis, so seq=1 decode still has
  // B-way parallelism). d2048-d8192 with k4 bracket the short conv of the
  // linear-attention/Mamba blocks; s1 is the decode shape, s512 prefill-like.
  AddA('Dw s1 d512 k4',     6, 1,   512,  4, 0);
  AddA('Dw s1 d2048 k4',    6, 1,   2048, 4, 0);
  AddA('Dw s1 d8192 k4',    6, 1,   8192, 4, 0);
  AddA('Dw s64 d2048 k4',   6, 64,  2048, 4, 0);
  AddA('Dw s512 d4096 k4',  6, 512, 4096, 4, 0);
  // Same layer on the stateful incremental-decode path (history taps +
  // per-chunk history advance) - what a real autoregressive decode step runs.
  AddA('DwDec s1 d2048 k4', 7, 1,   2048, 4, 0);
  AddA('DwDec s1 d8192 k4', 7, 1,   8192, 4, 0);
  // GatedDeltaNet: A=seq, B=Hk, C=Hv, D=head dim (Dk=Dv). The chunk axis is
  // the K-HEAD (Hk work items, each carrying Hv/Hk value heads), so seq=1
  // decode has Hk-way parallelism. h16/32 d128 is the Qwen3-Next linear-
  // attention geometry; h4/8 d64 probes the small end. s64 stays the largest
  // full-sweep row: the training-mode FS cache is Seq*Hv*Dk*Dv floats.
  AddA('GDN s1 h4/8 d64',    8, 1,  4,  8,  64);
  AddA('GDN s1 h16/32 d128', 8, 1,  16, 32, 128);
  AddA('GDN s64 h16/32 d128',8, 64, 16, 32, 128);
  // Same layer on the stateful incremental-decode path (in-place FDecS carry).
  AddA('GDNDec s1 h4/8 d64',    9, 1, 4,  8,  64);
  AddA('GDNDec s1 h16/32 d128', 9, 1, 16, 32, 128);
end;

// One pass of the shape sweep for a given memory mode. LowMem toggles the
// inference-time low-memory contract (SetTrainable(False, LowMem)).
// Conv/pointwise chunk under BOTH modes, so running the sweep twice (LowMem
// False then True) shows whether the mode shifts the eligible set, the chunk
// counts or the speedup. FC is unaffected by the memory mode (no released
// weight caches); under low memory conv/pointwise release the concatenated-
// weight caches and take the per-neuron ranged path, so their timing can differ.
// Returns the ALL-shapes aggregate for this pass, so the caller can compare the
// two memory modes against each other.
function ExperimentA(LowMem: boolean): TAggr;
var
  si, i, fam: integer;
  S: TShape;
  NN: TNNet;
  L: TNNetLayer;
  MyIn, OutSerial: TNNetVolume;
  Ts, Tt, sp, maxdiff, d, ops, worstdiff: double;
  elig, memtag, worstname: string;
  workers, chunks, neurons: integer;
  AggElig, AggNo: TAggr;
  AggFam: array[0..FAM_COUNT - 1] of TAggr;
begin
  BuildShapesA;
  AggrReset(Result); AggrReset(AggElig); AggrReset(AggNo);
  for i := 0 to FAM_COUNT - 1 do AggrReset(AggFam[i]);
  worstdiff := -1; worstname := '(none)';
  if LowMem then memtag := 'low-memory ON' else memtag := 'low-memory OFF';
  WriteLn('=== Experiment A: per-layer threading OFF vs ON  [', memtag, '] ===');
  WriteLn('Cores (NeuralDefaultThreadCount) = ', NeuralDefaultThreadCount(),
    '   shapeset ', SHAPESET_A, ' (', Length(ShA), ' shapes)');
  WriteLn('build: ', BuildTag);
  WriteLn('FC/PW/Conv/Dw/GDN: every size is chunk-eligible; ',
    'SwiGLU/RMSN/RoPE show their current verdict');
  WriteLn(Format('%-20s %8s %6s %9s %9s %8s %7s %9s',
    ['shape', 'neurons', 'elig?', 'off ms', 'on ms', 'speedup',
     'chunks', 'maxdiff']));
  WriteLn(StringOfChar('-', 86));
  for si := 0 to High(ShA) do
  begin
    S := ShA[si];
    NN := BuildOne(S);
    MyIn := TNNetVolume.Create(NN.Layers[0].Output);
    MyIn.Randomize();
    neurons := NN.GetLastLayer().CountNeurons();

    // Serial reference: Compute serial disables intra-layer threading on its
    // own. Snapshot the output for the parity check below.
    NN.Compute(MyIn, 0, {parallel=}false);
    OutSerial := TNNetVolume.Create(NN.GetLastLayer().Output);
    Ts := BestTimeMs(NN, MyIn, {parallel=}false);

    // Same net, same weights, threading on: routing Compute through the
    // parallel scheduler enables intra-layer threading on its own
    // (SetTrainable(False) is the inference-only contract the parallel pass
    // expects). LowMem is the mode under test this pass - conv/pointwise chunk
    // in both modes, so the parity check must hold either way.
    NN.SetTrainable(False, {pLowMemory=}LowMem);
    // Keep the worker pool alive/hot across the timed passes (default policy).
    NN.StartThreadWorkers();
    // Stateful decode shapes: the serial timing passes advanced the carried
    // state (conv history / delta-rule state bank), so restart from zero -
    // OutSerial was snapshotted from the very first (fresh-state) pass, and
    // the parity pass below must see the same state. The timed passes after
    // it may advance state freely.
    if (S.Kind = 7) or (S.Kind = 9) then
      TNNetRecurrentDecodeBase(NN.GetLastLayer()).ResetState();
    NN.Compute(MyIn, 0, {parallel=}true);
    maxdiff := 0;
    for i := 0 to OutSerial.Size - 1 do
    begin
      d := Abs(OutSerial.FData[i] - NN.GetLastLayer().Output.FData[i]);
      if d > maxdiff then maxdiff := d;
    end;
    NN.ResetSchedulerStats();  // count only this shape's parallel timing passes
    Tt := BestTimeMs(NN, MyIn, {parallel=}true);

    // Chunks the scheduler emitted for the single compute layer:
    // clamp(workers, 1, ChunkWorkCount) when eligible, else 0 (whole-layer).
    // The elig column is the real ChunkEligible() verdict; eligible layers show
    // min(workers, work), non-eligible layers (e.g. OpenCL / int8 / Winograd
    // paths) show 0.
    L := NN.GetLastLayer();
    workers := NN.SchedulerWorkerCount();
    if L.ChunkEligible() then
    begin
      elig := 'yes';
      chunks := Max(1, Min(workers, L.ChunkWorkCount()));
    end
    else
    begin
      elig := 'no';
      chunks := 0;
    end;

    if Tt > 0 then sp := Ts / Tt else sp := 0;
    WriteLn(Format('%-20s %8d %6s %9.3f %9.3f %7.2fx %7d %9.2g',
      [S.Name, neurons, elig, Ts, Tt, sp, chunks, maxdiff]));
    // Parallel/serial pass counts for this shape's timed run (passes: P/S).
    WriteLn('      ', NN.SchedulerStatsReport());

    // Accumulate. The eligible/not-eligible split matters for the speedup
    // reading: non-eligible shapes are ~1.00x by construction, so they only
    // dilute a global parallel-gain figure - but they are real work, so they
    // stay in the absolute (ALL) score.
    ops := WorkOps(S);
    AggrAdd(Result, Ts, Tt, ops);
    if L.ChunkEligible() then AggrAdd(AggElig, Ts, Tt, ops)
    else AggrAdd(AggNo, Ts, Tt, ops);
    fam := FamilyOf(S.Kind);
    AggrAdd(AggFam[fam], Ts, Tt, ops);
    if maxdiff > worstdiff then
    begin
      worstdiff := maxdiff;
      worstname := S.Name;
    end;

    OutSerial.Free;
    MyIn.Free;
    NN.Free;
  end;

  WriteLn;
  WriteLn('--- Experiment A summary  [', memtag, ']  shapeset ', SHAPESET_A,
    '  ', StringOfChar('-', 20));
  WriteAggrHeader;
  WriteAggrRow('ALL (geomean)', Result);
  WriteAggrRow('chunk-eligible', AggElig);
  WriteAggrRow('not eligible', AggNo);
  WriteLn;
  WriteLn('  by family');
  for i := 0 to FAM_COUNT - 1 do
    WriteAggrRow('  ' + FAM_NAMES[i], AggFam[i]);
  WriteLn;
  WriteLn(Format('  parity: worst maxdiff %.2g  (%s)',
    [worstdiff, worstname]));
  WriteLn;
end;

// =============================== Experiment B =================================
// Small shapes from Experiment A, swept across StartThreadWorkers() policies:
// hot-core counts 1..4 crossed with cool-down timeouts 0..2 seconds. The small
// shapes are the pool-policy probe - the fixed scheduler/pool overhead is a big
// fraction of their tiny compute, so keeping more workers hot (or letting them
// nap sooner) is where the OFF-vs-ON speedup visibly shifts. The larger A shapes
// amortise the overhead and barely move with policy, so only the small end is
// swept here.
//
// pHotThreadNum (arg 2) = how many low-index workers stay hot (never napping
// while idle within a pass); pHotThreadTimeoutSeconds (arg 3) = the cool-down
// window before a non-primary hot worker is allowed to nap.
// Policy grid: hot-core counts HOT_MIN..HOT_MAX crossed with cool-down
// timeouts 0..TMO_MAX, flattened to (hot - HOT_MIN) * (TMO_MAX + 1) + tmo.
const
  HOT_MIN = 1;
  HOT_MAX = 4;
  TMO_MAX = 2;
  POL_COUNT = (HOT_MAX - HOT_MIN + 1) * (TMO_MAX + 1);

type
  TSmallResult = record
    Name: string;
    Off, BestOn, WorstOn: double;
    BestHot, BestTmo: integer;
  end;

procedure ExperimentB;
var
  si, hot, tmo, pol, i, j, t: integer;
  S: TShape;
  NN: TNNet;
  MyIn: TNNetVolume;
  off, on_, sp, sens, sumOff, sumBest, sumSens: double;
  Small: array of TShape;
  Res: array of TSmallResult;
  polLog: array[0..POL_COUNT - 1] of double;   // sum of ln(on ms) over shapes
  polN: array[0..POL_COUNT - 1] of integer;
  polGeo: array[0..POL_COUNT - 1] of double;
  ord_: array[0..POL_COUNT - 1] of integer;

  procedure AddSmall(const N: string; K, A, B, C, D: integer);
  var idx: integer;
  begin
    idx := Length(Small);
    SetLength(Small, idx + 1);
    Small[idx].Name := N; Small[idx].Kind := K;
    Small[idx].A := A; Small[idx].B := B; Small[idx].C := C; Small[idx].D := D;
  end;
begin
  WriteLn('=== Experiment B: small shapes x StartThreadWorkers() policy ===');
  WriteLn('Cores (NeuralDefaultThreadCount) = ', NeuralDefaultThreadCount(),
    '   shapeset ', SHAPESET_B);
  WriteLn('build: ', BuildTag);
  WriteLn('hot = hot-core count (StartThreadWorkers arg 2), ',
    'tmo = cool-down seconds (arg 3)');

  SetLength(Small, 0);
  AddSmall('FC 64x64',       0, 64, 64,  0,  0);
  AddSmall('FC 128x128',     0, 128, 128, 0,  0);
  AddSmall('Pw 8x8x64->64',  2, 8,  64,  64, 0);
  AddSmall('Conv 8x8x32 k3', 1, 8,  32,  32, 3);
  SetLength(Res, Length(Small));
  for i := 0 to POL_COUNT - 1 do
  begin
    polLog[i] := 0;
    polN[i] := 0;
  end;

  for si := 0 to High(Small) do
  begin
    S := Small[si];
    WriteLn;
    WriteLn(Format('  %s', [S.Name]));
    WriteLn(Format('    %4s %4s %10s %10s %8s',
      ['hot', 'tmo', 'off ms', 'on ms', 'speedup']));

    // OFF reference is policy-independent (the serial path never touches the
    // worker pool), so time it once per shape. MyIn is an independent copy, so
    // it outlives the net it was sized from and is reused for every policy.
    NN := BuildOne(S);
    MyIn := TNNetVolume.Create(NN.Layers[0].Output);
    MyIn.Randomize();
    NN.SetTrainable(False, {pLowMemory=}False);
    off := BestTimeMs(NN, MyIn, {parallel=}false);
    NN.Free;

    Res[si].Name := S.Name;
    Res[si].Off := off;
    Res[si].BestOn := 1e30;
    Res[si].WorstOn := 0;
    Res[si].BestHot := 0;
    Res[si].BestTmo := 0;

    for hot := HOT_MIN to HOT_MAX do
      for tmo := 0 to TMO_MAX do
      begin
        NN := BuildOne(S);
        NN.SetTrainable(False, {pLowMemory=}False);
        NN.ResetSchedulerStats();
        // Fresh pool per policy: arg 2 = hot cores, arg 3 = cool-down seconds.
        NN.StartThreadWorkers({StopOnFinish=}False, {pHotThreadNum=}hot,
          {pHotThreadTimeoutSeconds=}tmo);
        on_ := BestTimeMs(NN, MyIn, {parallel=}true);
        if on_ > 0 then sp := off / on_ else sp := 0;
        WriteLn(Format('    %4d %4d %10.3f %10.3f %7.2fx',
          [hot, tmo, off, on_, sp]));
        NN.Free;

        pol := (hot - HOT_MIN) * (TMO_MAX + 1) + tmo;
        polLog[pol] := polLog[pol] + Ln(Max(on_, 1e-6));
        Inc(polN[pol]);
        if on_ < Res[si].BestOn then
        begin
          Res[si].BestOn := on_;
          Res[si].BestHot := hot;
          Res[si].BestTmo := tmo;
        end;
        if on_ > Res[si].WorstOn then Res[si].WorstOn := on_;
      end;
    MyIn.Free;

    // Per-shape verdict: the policy that won, and how much the policy choice
    // moved this shape at all (worst/best on-time).
    sens := Res[si].WorstOn / Max(Res[si].BestOn, 1e-6);
    WriteLn(Format('    best: hot=%d tmo=%d   on %.3f ms  %.2fx' +
      '     sensitivity %.2fx (%.3f..%.3f)',
      [Res[si].BestHot, Res[si].BestTmo, Res[si].BestOn,
       Res[si].Off / Max(Res[si].BestOn, 1e-6), sens,
       Res[si].BestOn, Res[si].WorstOn]));
  end;

  // ------------------------------- summary ------------------------------
  // Averaging all POL_COUNT cells would fold deliberately-bad policies into the
  // box's number, which is meaningless for a hardware comparison. The per-shape
  // BEST policy is the box's real small-shape latency; the spread across
  // policies is reported separately as sensitivity.
  WriteLn;
  WriteLn('--- Experiment B summary  shapeset ', SHAPESET_B, ' (',
    Length(Small), ' small shapes) ', StringOfChar('-', 20));
  WriteLn(Format('  %-18s %9s %12s %8s   %-13s %11s',
    ['shape', 'off ms', 'best on ms', 'speedup', 'best policy',
     'sensitivity']));
  sumOff := 0; sumBest := 0; sumSens := 0;
  for si := 0 to High(Res) do
  begin
    sens := Res[si].WorstOn / Max(Res[si].BestOn, 1e-6);
    if Res[si].BestOn > 0 then sp := Res[si].Off / Res[si].BestOn else sp := 0;
    WriteLn(Format('  %-18s %9.3f %12.3f %7.2fx   hot=%d tmo=%d %10.2fx',
      [Res[si].Name, Res[si].Off, Res[si].BestOn, sp,
       Res[si].BestHot, Res[si].BestTmo, sens]));
    sumOff := sumOff + Ln(Max(Res[si].Off, 1e-6));
    sumBest := sumBest + Ln(Max(Res[si].BestOn, 1e-6));
    sumSens := sumSens + Ln(Max(sens, 1e-6));
  end;
  WriteLn('  ', StringOfChar('-', 78));
  WriteLn(Format('  %-18s %9.3f %12.3f %7.2fx   %-13s %10.2fx',
    ['geomean', GeoMean(sumOff, Length(Res)), GeoMean(sumBest, Length(Res)),
     GeoMean(sumOff, Length(Res)) / Max(GeoMean(sumBest, Length(Res)), 1e-6),
     '', GeoMean(sumSens, Length(Res))]));

  // Which pool policy this class of machine actually wants - the answer the
  // per-shape tables above make you eyeball across POL_COUNT x shapes numbers.
  for i := 0 to POL_COUNT - 1 do
  begin
    polGeo[i] := GeoMean(polLog[i], polN[i]);
    ord_[i] := i;
  end;
  for i := 0 to POL_COUNT - 2 do
    for j := 0 to POL_COUNT - 2 - i do
      if polGeo[ord_[j]] > polGeo[ord_[j + 1]] then
      begin
        t := ord_[j]; ord_[j] := ord_[j + 1]; ord_[j + 1] := t;
      end;
  WriteLn;
  WriteLn('  policy ranking (geomean on ms across all shapes, best first)');
  for i := 0 to POL_COUNT - 1 do
  begin
    pol := ord_[i];
    Write(Format('    hot=%d tmo=%d %7.3f',
      [HOT_MIN + pol div (TMO_MAX + 1), pol mod (TMO_MAX + 1), polGeo[pol]]));
    if (i mod 3 = 2) or (i = POL_COUNT - 1) then WriteLn;
  end;
  WriteLn(Format('  small-shape latency floor (this box): %.3f ms geomean ' +
    'at best policy', [GeoMean(sumBest, Length(Res))]));
  WriteLn;
end;

// =============================== Experiment C =================================
// Transformer-decoder-ish forward: a stack of per-token MLP blocks over a short
// sequence - the shape where intra-layer threading is meant to pay off.
function BuildStack(dModel, dHidden, nBlocks, seqLen: integer): TNNet;
var NN: TNNet; b: integer;
begin
  NN := TNNet.Create();
  NN.AddLayer(TNNetInput.Create(dModel, seqLen, 1));
  for b := 0 to nBlocks - 1 do
  begin
    NN.AddLayer(TNNetPointwiseConvReLU.Create(dHidden));
    NN.AddLayer(TNNetPointwiseConvLinear.Create(dModel));
  end;
  for b := 0 to NN.CountLayers - 1 do NN.Layers[b].InitDefault();
  // Inference contract for the parallel pass. pLowMemory=False keeps the
  // concatenated-weight caches (the faster pointwise path); low memory also
  // chunks but takes the slower per-neuron path, so it is left off here to time
  // the fast case.
  NN.SetTrainable(False, {pLowMemory=}False);
  Result := NN;
end;

// Transformer-decoder-ish forward with the modern token-wise layers: RoPE on
// the input stream, then per block TokenRMSNorm -> up-projection to 2*dHidden
// -> SwiGLU gate (halves the depth back to dHidden) -> down-projection to
// dModel. Unlike BuildStack, tokens run along X and channels along Depth
// (seqLen x 1 x dModel), the layout the token-wise layers reduce over. This is
// the end-to-end probe for the SwiGLU/RMSNorm/RoPE ComputeRange work: until
// they chunk, only the two pointwise projections thread and the token layers
// are the serial gaps in the pass.
function BuildStackSwiGLU(dModel, dHidden, nBlocks, seqLen: integer): TNNet;
var NN: TNNet; b: integer;
begin
  NN := TNNet.Create();
  NN.AddLayer(TNNetInput.Create(seqLen, 1, dModel));
  NN.AddLayer(TNNetRotaryEmbedding.Create());
  for b := 0 to nBlocks - 1 do
  begin
    NN.AddLayer(TNNetTokenRMSNorm.Create());
    NN.AddLayer(TNNetPointwiseConvLinear.Create(2 * dHidden));
    NN.AddLayer(TNNetSwiGLU.Create());
    NN.AddLayer(TNNetPointwiseConvLinear.Create(dModel));
  end;
  for b := 0 to NN.CountLayers - 1 do NN.Layers[b].InitDefault();
  NN.SetTrainable(False, {pLowMemory=}False);
  Result := NN;
end;

// Per-layer chunk report for a net that has just run a parallel pass. The
// scheduler splits a chunk-eligible layer into clamp(WorkerCount, 1, WorkCount)
// slices (see SchedEnqueueReady); a non-eligible layer runs as one whole-layer
// item. WorkCount is the ComputeRange index space (FullConnect: output neurons;
// Convolution: output positions), so this exposes the seq=1 pointwise case
// where positions=1 forces a single chunk and no intra-layer parallelism.
procedure DumpChunks(NN: TNNet);
var
  li, workers, work, chunks: integer;
  L: TNNetLayer;
begin
  workers := NN.SchedulerWorkerCount();
  WriteLn(Format('      per-layer chunks (workers=%d):', [workers]));
  for li := 0 to NN.CountLayers - 1 do
  begin
    L := NN.Layers[li];
    work := L.ChunkWorkCount();
    if L.ChunkEligible() then
      chunks := Max(1, Min(workers, work))  // matches SchedEnqueueReady
    else
      chunks := 0;                          // 0 => whole-layer, not chunked
    WriteLn(Format('        [%2d] %-26s work=%-7d chunks=%d',
      [li, L.ClassName, work, chunks]));
  end;
end;

procedure SweepNet(const Tag: string; dModel, dHidden, nBlocks, seqLen: integer;
  UseSwiGLU: boolean = false);
var
  MyIn: TNNetVolume; NN: TNNet; off, on_: double;
begin
  if UseSwiGLU then NN := BuildStackSwiGLU(dModel, dHidden, nBlocks, seqLen)
  else NN := BuildStack(dModel, dHidden, nBlocks, seqLen);
  MyIn := TNNetVolume.Create(NN.Layers[0].Output);
  MyIn.Randomize();
  off := BestTimeMs(NN, MyIn, {parallel=}false);
  NN.Free;

  if UseSwiGLU then NN := BuildStackSwiGLU(dModel, dHidden, nBlocks, seqLen)
  else NN := BuildStack(dModel, dHidden, nBlocks, seqLen);
  NN.ResetSchedulerStats();  // clean per-net pass/worker counts
  NN.StartThreadWorkers();   // keep the pool alive/hot across the timed passes
  on_ := BestTimeMs(NN, MyIn, {parallel=}true);

  WriteLn(Format('  %-16s dM=%-5d dH=%-5d blk=%d seq=%-3d  off=%8.3f  on=%8.3f  %6.2fx',
    [Tag, dModel, dHidden, nBlocks, seqLen, off, on_, off / on_]));
  // Proof the parallel path actually ran (passes: P parallel / S serial); if
  // this shows 0 parallel the ON timing measured the serial fallback.
  WriteLn('      ', NN.SchedulerStatsReport());
  DumpChunks(NN);
  NN.Free;
  MyIn.Free;
end;

var
  AggMemOff, AggMemOn: TAggr;
begin
  Randomize;
  // Experiment A is run twice: once with the low-memory inference contract off
  // (concatenated-weight caches kept) and once with it on (caches released,
  // per-neuron ranged path). Conv/pointwise chunk in both modes, so the two
  // passes expose whether the memory mode shifts eligibility or timing.
  AggMemOff := ExperimentA({LowMem=}False);
  AggMemOn := ExperimentA({LowMem=}True);
  // What the low-memory contract costs on this box, as one ratio per column.
  WriteLn('--- Experiment A: memory-mode comparison  shapeset ', SHAPESET_A,
    '  ', StringOfChar('-', 20));
  WriteLn(Format('  %-22s %10s %10s %8s', ['', 'off ms', 'on ms', 'speedup']));
  WriteLn(Format('  %-22s %10.3f %10.3f %7.2fx',
    ['low-memory OFF', AggrOff(AggMemOff), AggrOn(AggMemOff),
     AggrOff(AggMemOff) / Max(AggrOn(AggMemOff), 1e-6)]));
  WriteLn(Format('  %-22s %10.3f %10.3f %7.2fx',
    ['low-memory ON', AggrOff(AggMemOn), AggrOn(AggMemOn),
     AggrOff(AggMemOn) / Max(AggrOn(AggMemOn), 1e-6)]));
  WriteLn(Format('  %-22s %9.2fx %9.2fx',
    ['ON / OFF cost',
     AggrOff(AggMemOn) / Max(AggrOff(AggMemOff), 1e-6),
     AggrOn(AggMemOn) / Max(AggrOn(AggMemOff), 1e-6)]));
  // Experiments B (pool-policy sweep) and C (end-to-end nets) are no longer
  // relevant; kept in the source for reference but not run.
  // ExperimentB;
  // WriteLn('=== Experiment C: realistic nets  threading OFF vs ON (end-to-end) ===');
  // SweepNet('small GPT MLP', 512,  2048, 6, 8);
  // SwiGLU-FFN stack (RoPE + per-block RMSNorm -> up-proj -> SwiGLU ->
  // down-proj): the end-to-end probe for the token-layer ComputeRange work.
  // seq=8 is prefill-like; seq=1 is the KV-cache decode shape where the token
  // layers need a finer-than-token chunk axis to thread at all.
  // SweepNet('GPT SwiGLU FFN', 512,  2048, 6, 8, {UseSwiGLU=}true);
  // SweepNet('SwiGLU 1-token', 1024, 4096, 4, 1, {UseSwiGLU=}true);
  // SweepNet('mid GPT MLP',   768,  3072, 6, 16);
  // SweepNet('wide 1-token',  2048, 8192, 4, 1);
  WriteLn;
  WriteLn('speedup > 1.00x => chunking that shape pays off on this box;');
  WriteLn('<= 1.00x on a SINGLE-layer shape overstates the in-net cost: the');
  WriteLn('scheduler pass overhead charged here to one layer is paid once per');
  WriteLn('forward in a real net - validate borderline shapes on a real model.');
  WriteLn;
  WriteLn('Summary rows are GEOMETRIC means, so every shape counts equally and');
  WriteLn('the summary speedup equals the geomean of the per-shape speedups.');
  WriteLn('Timings are BEST-of-5 after warm-up: good for comparing boxes, but a');
  WriteLn('best case, so a noisy machine looks better here than in real use.');
  WriteLn('ms figures compare across boxes only for the SAME shapeset version;');
  WriteLn('GOP/s is an approximate work proxy (element-wise layer op weights are');
  WriteLn('estimates) - read it within a family, not as a certified FLOP rate.');
end.
