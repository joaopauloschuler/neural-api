program IntraLayerThreadingBench;
(*
IntraLayerThreadingBench: validates the intra-layer threading decision.

  * Experiment A sweeps single-layer FC / pointwise / conv shapes from tiny to
    large - plus the token-wise transformer layers (SwiGLU / TokenRMSNorm /
    RotaryEmbedding) at decode (seq=1) and prefill-like sequence lengths -
    reports each shape's neuron count, whether it is eligible to chunk
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
procedure AddA(const N: string; K, A, B, C, D: integer);
var idx: integer;
begin
  idx := Length(ShA);
  SetLength(ShA, idx + 1);
  ShA[idx].Name := N; ShA[idx].Kind := K;
  ShA[idx].A := A; ShA[idx].B := B; ShA[idx].C := C; ShA[idx].D := D;
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
  end;
  NN.GetLastLayer().InitDefault();
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
end;

// One pass of the shape sweep for a given memory mode. LowMem toggles the
// inference-time low-memory contract (SetTrainable(False, LowMem)).
// Conv/pointwise chunk under BOTH modes, so running the sweep twice (LowMem
// False then True) shows whether the mode shifts the eligible set, the chunk
// counts or the speedup. FC is unaffected by the memory mode (no released
// weight caches); under low memory conv/pointwise release the concatenated-
// weight caches and take the per-neuron ranged path, so their timing can differ.
procedure ExperimentA(LowMem: boolean);
var
  si, i: integer;
  S: TShape;
  NN: TNNet;
  L: TNNetLayer;
  MyIn, OutSerial: TNNetVolume;
  Ts, Tt, sp, maxdiff, d: double;
  elig, memtag: string;
  workers, chunks, neurons: integer;
begin
  BuildShapesA;
  if LowMem then memtag := 'low-memory ON' else memtag := 'low-memory OFF';
  WriteLn('=== Experiment A: per-layer threading OFF vs ON  [', memtag, '] ===');
  WriteLn('Cores (NeuralDefaultThreadCount) = ', NeuralDefaultThreadCount(),
    '   FC/PW/Conv: every size is chunk-eligible; ',
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

    OutSerial.Free;
    MyIn.Free;
    NN.Free;
  end;
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
procedure ExperimentB;
var
  si, hot, tmo: integer;
  S: TShape;
  NN: TNNet;
  MyIn: TNNetVolume;
  off, on_, sp: double;
  Small: array of TShape;

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
  WriteLn('Cores (NeuralDefaultThreadCount) = ', NeuralDefaultThreadCount());
  WriteLn('hot = hot-core count (StartThreadWorkers arg 2), ',
    'tmo = cool-down seconds (arg 3)');

  SetLength(Small, 0);
  AddSmall('FC 64x64',       0, 64, 64,  0,  0);
  AddSmall('FC 128x128',     0, 128, 128, 0,  0);
  AddSmall('Pw 8x8x64->64',  2, 8,  64,  64, 0);
  AddSmall('Conv 8x8x32 k3', 1, 8,  32,  32, 3);

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

    for hot := 1 to 4 do
      for tmo := 0 to 2 do
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
      end;
    MyIn.Free;
  end;
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

begin
  Randomize;
  // Experiment A is run twice: once with the low-memory inference contract off
  // (concatenated-weight caches kept) and once with it on (caches released,
  // per-neuron ranged path). Conv/pointwise chunk in both modes, so the two
  // passes expose whether the memory mode shifts eligibility or timing.
  ExperimentA({LowMem=}False);
  ExperimentA({LowMem=}True);
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
  WriteLn('speedup > 1.00x => the 1M threading decision is winning on this box.');
end.
