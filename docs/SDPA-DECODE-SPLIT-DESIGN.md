# Split-row SDPA decode kernels (design, 2026-09-03)

Applies to `TNNetFusedSDPA` under OpenCL: the two cached-decode entry points
`cai_sdpa_decode` and `cai_sdpa_decode_int8` in `neural/neural.cl`, launched by
`TNNetFusedSDPACL.Compute` and `TNNetFusedSDPACL.ComputeInt8` in
`neural/neuralnetwork.pas`. The user authorized the build on 2026-09-03.

## 1. The problem

Measured by the user on Qwen3.8-27B (`--gpu --int4 --prefill-window 256`):
the decode step costs 26.8 ms of forward at a 24-token context and 65 ms at a
7.9k-token context. The 16 `TNNetFusedSDPA` layers are the only layers whose
per-token cost grows with context (the GatedDeltaNet and DepthwiseConv1D
layers are O(1) per token), so the growth is about 2.4 ms per attention layer
per token at 7.9k rows. The int8 K and V rows one layer reads at that context
are a few megabytes, under 0.1 ms of memory traffic. The kernel is
latency-bound, by a factor of about 100, because of its launch shape:

1. `ComputeInt8` launches one work-group of 256 lanes per (query head, band
   row). One decode token has BandRows = 1, so the launch is QHeads
   work-groups (32 on the 27B) on a GPU with 188 compute units. Each busy
   compute unit holds 8 warps, too few to hide memory latency.
2. Phase 3 is a serial chain over every cached row: lane d runs
   `for j in jStart..LiveLen-1` with one dependent `mad` and one global load
   per row. 7.9k dependent iterations at about 300 ns each is the 2.4 ms.
3. Phase 1 scores one row per lane with byte loads `krow[d]`, so a warp's 32
   lanes touch 32 rows 256 bytes apart on every load instruction, one cache
   line per lane per byte, and the 256 mads per row are a dependent chain.
4. Every query head re-reads its KV group's rows: K and V cross the memory
   bus GroupSize times per layer (16 times at 32 query heads over 2 KV heads).

The 24-token case is fast because both chains are short and occupancy does
not matter. The comment above `csFusedSDPABandRows` shows the constant was
chosen for the prefill window, not the long single-token decode.

## 2. Facts (verified at 13aae5e2 on 2026-09-03)

- F1. `TNNetFusedSDPACL` (neuralnetwork.pas:4558) binds four entry points on
  ONE `TNeuralKernel` and therefore one in-order queue: the FP32 and int8
  append, the FP32 decode (the base kernel), the int8 decode. Two `cl_kernel`
  handles are used instead of one launched twice because `clSetKernelArg` on
  a kernel with a launch in flight is undefined.
- F2. Both decode kernels take `FTokenCnt` token rows per launch (prefill
  window or speculative window, `cMaxStepTokens` = 4 on the decode path, the
  prefill window on the twins), with the causal bound `LiveLen = FCacheSlot +
  t + 1` per token row and the sliding-window bound `jStart = LiveLen -
  FWindow` when `FWindow > 0`.
- F3. The score band `FBufScores` is `QHeads * BandRows * CacheMax` floats in
  global memory, written and read inside one launch. At 32 heads and a 32k
  context it is 64 MiB.
- F4. The int8 cache is head-major: KV head g's codes are the block
  `g*FCacheMax*FDk` of `FBufKCodes`, its row scales `g*FCacheMax` of
  `FBufKScales`; V the same. The FP32 cache is head-major in `FBufK/FBufV`.
- F5. `TNNetFusedSDPA.ComputeOpenCL` (neuralnetwork.pas:~35012) is the only
  caller; it passes `FCacheLen` as CacheSlot and leaves the result resident
  (`pKeepResultOnOpenCL = true`). `FinishForward` names the queue.
- F6. The split-K GEMV (`TDotProductSharedKernel.ChooseSplitK`,
  neuralopencl.pas:~1340) is the pattern for a two-launch reduction: a
  target work-item count from `DeviceMaxComputeUnits()` times a constant, a
  minimum slab, a maximum split count, a grow-only `FPartialBuffer`, and
  a reduce kernel on the same queue.
- F7. Nine parity tests in `tests/TestNeuralNumerical.pas` (FusedSDPA*OpenCL
  Parity, lines 71643..71960) compare OpenCL decode against the CPU cached
  decode at tolerance 1e-3 through `RunFusedSDPADecodeParity`, covering both
  cache formats, a window mask, a multi-token step, and the host handoff.
  Their contexts are short (StepCnt 9), so today they never exercise more
  than a few rows.
- F8. This box has no GPU: PoCL only, `ulimit -v 3145728`. All timing is
  the user's.

## 3. Design

### 3.1 Two launches per attention layer per step

Replace each decode kernel with a split-row pair, and share the merge:

- `cai_sdpa_decode_split` and `cai_sdpa_decode_split_int8` (pass 1): ONE
  WORK-GROUP PER (KV HEAD g, TOKEN ROW t, CHUNK c). The group holds the
  GroupSize query rows of head group g in local memory and processes the
  chunk of cache rows `[c*ChunkRows, min((c+1)*ChunkRows, LiveLen))`,
  clipped below by jStart. It writes, per (query head h, t, c), the chunk's
  partial softmax state: the chunk max m, the chunk exp-sum l, and the
  unnormalized value sum acc[0..Dk-1]. Partial stride Dk + 2 floats.
- `cai_sdpa_decode_merge` (pass 2, one kernel for both formats): one
  work-group per (query head h, token row t); lanes split d. M = max over
  chunks of m_c; l = sum l_c * exp(m_c - M); y[d] = (sum acc_c[d] *
  exp(m_c - M)) / l. A chunk with no live rows writes m = -1e30 and l = 0,
  which the merge weights to zero; chunk 0 always has at least one row
  (LiveLen >= 1), so M is finite.

Pass 1 launch: global (LocalSize, KVHeads * TokenCnt * Splits), local
(LocalSize, 1), dimension 1 = (g * TokenCnt + t) * Splits + c. Pass 2:
global (LocalSize, QHeads * TokenCnt). Both on `FKernel`'s queue, enqueued
in that order, so the in-order queue is the synchronization, as today.

### 3.2 Inside pass 1

- Phase 1 (scores): lanes split the chunk's rows. A lane reads its row K[j]
  ONCE, as 16-byte (int8) or float4 (FP32) vector loads, and scores it
  against all GroupSize query rows from local memory, so each row costs Dk/16
  load instructions instead of Dk and each K byte serves GroupSize heads.
  Scale, soft-cap, then the score goes to a chunk-local score tile in
  local memory: ChunkRows * GroupSize floats. Then one tree max per head.
- Phase 2: exp in place on the tile, one tree sum per head.
- Phase 3 (values): lanes split d. Lane d loops the chunk's rows only
  (ChunkRows, not LiveLen), reads V[j][d] once and accumulates GroupSize
  accumulators, one per query head of the group. V byte loads are coalesced
  across lanes because consecutive lanes hold consecutive d.
- Write m, l, acc per head to the partial buffer.

The global score band `FBufScores` and `csFusedSDPABandRows` go away: the
tile is local memory. The old `cai_sdpa_decode` and `cai_sdpa_decode_int8`
kernels are deleted, not kept as a Splits = 1 fast path: one path, and the
merge at Splits = 1 is one small launch per attention layer.

### 3.3 Split sizing (host, `TNNetFusedSDPACL`)

Mirror F6. Per launch, with MaxLive = CacheSlot + TokenCnt (the longest
live range of any token row in the step):

- TargetGroups = DeviceMaxComputeUnits() * csFusedSDPAGroupsPerUnit.
- Splits = ceil(TargetGroups / (KVHeads * TokenCnt)), then
  Splits <= ceil(MaxLive / csFusedSDPAMinChunkRows), Splits <=
  csFusedSDPAMaxSplits, Splits >= 1.
- ChunkRows = ceil(MaxLive / Splits), and ChunkRows must also satisfy the
  local-memory budget of 3.4. If the budget forces a smaller ChunkRows, raise
  Splits to cover MaxLive (Splits may then exceed csFusedSDPAMaxSplits; the
  partial buffer is sized from the final Splits).

Starting constants, to be tuned by the user on the GPU: GroupsPerUnit 4,
MinChunkRows 64, MaxSplits 64. An environment override for the three,
following the `LoadInt4SplitKOverrides` pattern, so the user can sweep
without a rebuild. The constants live next to the split-K ones.

The partial buffer `FBufPartials` (grow-only, `csize_t` capacity, released in
the destructor) replaces `FBufScores`: QHeads * TokenCnt * Splits * (Dk + 2)
floats.

### 3.4 Local memory budget

Per work-group: the query tile GroupSize * Dk floats, the score tile
ChunkRows * GroupSize floats, and LocalSize floats of reduction scratch. At
GroupSize 16, Dk 256, ChunkRows 128, LocalSize 256: 16 KB + 8 KB + 1 KB. The
host reads the device's local memory size once (`CL_DEVICE_LOCAL_MEM_SIZE`,
add an accessor on `TEasyOpenCL` beside `DeviceMaxComputeUnits` if none
exists; check first, rule 3) and shrinks ChunkRows until the three tiles fit.
If the query tile alone does not fit (GroupSize * Dk floats), the layer must
not take the OpenCL path: make `TNNetFusedSDPA.WillOpenCL` (or the helper)
report that, and fall back to the host path, exactly as an oversized step
does today.

### 3.5 What does not change

- The append kernels, the cache layout, `UploadCache*`/`DownloadCache*`, the
  int8 quantization rule, `FinishForward`, the residency flags, the
  checkpoint code, the CPU path `ComputeCachedToken`.
- The kernel interface seen by `TNNetFusedSDPA.ComputeOpenCL`: `Compute` and
  `ComputeInt8` keep their signatures; the split is internal to the helper.
- The arguments FInvSqrtDk, FScoreSoftCap, FInvScoreSoftCap, FWindow,
  FCacheSlot and the causal rule of F2.

## 4. Tests (same agent, same commit or the next)

The nine existing parity tests (F7) must stay green on both builds. Add, in
`tests/TestNeuralNumerical.pas` beside them, using `RunFusedSDPADecodeParity`
where it fits and a `ForcedSplits`/`ForcedChunkRows` override on the helper
(test-only, so PoCL's small compute-unit count does not decide the split):

1. Long-context split parity, FP32 and int8: enough rows that Splits >= 3
   with a chunk boundary inside the live range and the last chunk partially
   filled.
2. Multi-token step where the later chunks are EMPTY for the early token rows
   (LiveLen below the chunk start), both formats.
3. Sliding window with jStart inside a chunk, and with jStart past whole
   chunks (those chunks empty).
4. GroupSize 1 (QHeads = KVHeads) and MQA (KVHeads = 1, QHeads >= 4).
5. Soft-cap on, split on.
6. The local-memory shrink path: a forced tiny budget makes ChunkRows shrink
   and Splits grow past the maximum, still bit-close (1e-3) to the CPU path.

Tolerance stays 1e-3; the summation order changes ulps only. Report the test
counts before and after on both builds (`-dAVX2` and plain).

## 5. Rules for the implementer

- Distrust this document: verify every cited line before building on it.
- Rule 17 of the optimization guide: no allocation per forward. The partial
  buffer is grow-only and sized in the helper, as `FBufScores` was.
- Names: `Max<Entity>Pos` loop bounds; `Splits`, `ChunkRows`, `ChunkStart`,
  `ChunkLive`; no `n`, `tmp`, `cnt`.
- Comment budget: two lines per declaration; the kernel header comments
  state the launch geometry, the partial layout and the empty-chunk rule,
  and nothing about what the old kernels did.
- Both suites, `lazbuild -B`, one OpenCL process at a time,
  `( ulimit -v 3145728; ... )`. Nothing OpenCL is timed on this box.
- Stage named paths only; never `git add -A`. Commit trailer as the session
  prescribes. One commit for kernels + host, one for tests, or one for both
  if green together.
- `neural/neuralopencl.pas` is CRLF; preserve the line endings.
