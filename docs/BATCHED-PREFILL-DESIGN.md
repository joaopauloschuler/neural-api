# Batched prefill for the chat engine - design

Status: design only, nothing implemented. Written 2026-09-02 from a read-only
audit of branch a16. This is the brief for the implementing agent.

## 1. The problem

`TChatEngine` prefills a prompt one token at a time
(`neural/neuralchatengine.pas:1595`, `for Cnt := Reused to LenM2 do
Session.StepForward(InV, Cnt)`). Every prompt token is a full width-1 forward,
so the whole weight set is streamed from device memory once per token.

Measured by the user on Qwen3.8-27B, `--int4 --gpu --max-fast-memory`,
RTX PRO 6000, ChatServer driven by a bpsa CodeAgent prompt:

| prompt tokens | prefill | per prompt token | decode at that context |
|---|---|---|---|
| 7880 | 360 s | ~46 ms | 13.8 tok/s (67 ms/step) |
| 54 | ~1.4 s | ~27 ms | 32.7 tok/s (24 ms/step) |

Prefill is the whole time-to-first-token and it exceeds bpsa's 120 s client
timeout. Goal: prefill a K-token window per forward so the weights are read
once per window, turning minutes into seconds. A second, separate problem is
visible in the same log (decode step cost grows 24 -> 67 ms with context; the
fused SDPA decode kernel is the suspect). It is NOT part of this design.

## 2. Facts that shape the design (all verified on a16)

F1. **A net's input width is fixed at build.** `TNNet.Compute` refuses a
    size mismatch (`neuralnetwork.pas:130151-130170`) through `FErrorProc`,
    which only PRINTS - a wrong-width step silently computes nothing. Every
    shape-dependent buffer is sized in `SetPrevLayer`, never in `Compute`
    (rule 17, `docs/OPTIMIZATION-GUIDE.md:9-37`). The chat engine builds at
    width 1 (`neuralchatengine.pas:1179`).

F2. **The streaming session is already width-agnostic.**
    `TNNetStreamingDecoder.StepForward` (`neural/neuraldecode.pas:6034`) sets
    every RoPE `PositionOffset := AbsPos` and calls `Compute`; its contract
    (`neuraldecode.pas:1234-1241`) promises a width-K window rotates like the
    full forward. Only requirement: `MaxCacheLen >= committed + one window`.

F3. **Every stateful layer of the Qwen3.5/3.8 hybrid handles a K-row window
    with carried state on BOTH the CPU and the device path**, except fused
    attention on the device:
    - `TNNetGatedDeltaNet`: one kernel serves full scan and decode, `t`-loop
      inside (`neural/neural.cl:2546`, host `neuralnetwork.pas:70567`); CPU
      `ComputeCPURange` aliases the decode state in place (`:70185`).
    - `TNNetDepthwiseConv1D`: decode kernel takes any `FSeqLen`
      (`neural.cl:2430`); CPU handles K < KernelSize-1 (`:61818-61849`).
      Doc promises bit-exactness for any window split (`:9464-9468`).
    - `TNNetTokenRMSNorm`/`HeadRMSNorm`, `TNNetRotaryEmbedding`,
      `TNNetSwiGLU`, `TNNetEmbedding` gather: per-token, no SizeX gate.
    - `TNNetFusedSDPA` CPU: `ComputeIncrementalFused` appends K rows then
      scores row p against `base+p+1` rows (`neuralnetwork.pas:34534-34565`),
      i.e. exact intra-window causality. Chunk path handles it too.
    - **`TNNetFusedSDPA.WillOpenCL` requires `FPrevLayer.FOutput.SizeX = 1`**
      (`neuralnetwork.pas:34710`); `cai_sdpa_append_kv`/`cai_sdpa_decode`
      (+ `_int8`) are single-slot by construction. A wider window drops
      attention to the host AND pulls the KV cache home
      (`ForceCacheOnRAM`, `:34745`), then re-uploads it - twice per layer
      per window. `TNNetCellMulByCell` follows it to the host.

F4. **The quantized matmul kernels already take K columns.** `FNumBs` is the
    B axis of `cai_dot_product_int8`, `_splitk`, `cai_dot_product_int4_splitk`
    and the reduce (`neural.cl:205/449/592/657`); one launch regardless of K
    (`neuralopencl.pas:1487-1517`). `Int8SplitCount` (`:1123-1145`) already
    degrades split-K to a plain 2-D launch when `FNumAs*FNumBs` fills the
    device. BUT they are untiled: every work-item re-reads its whole weight
    row, so a window saves launches, not weight bytes, unless L2 catches the
    reuse. The FP32 tiled GEMMs in `neural.cl` (`myGEMM6:1044` etc.) are
    dead code, referenced from no Pascal source.

F5. **The matmul helper freezes `FNumBs` at `EnableOpenCL`** and its
    `UnprepareForCompute` releases AND re-uploads the codes
    (`neuralopencl.pas:954/1023, 646-706`). Re-arming per window is not an
    option. `ComputeResidentCodes` hard-checks `VBs.Size = FSize*FNumBs`
    (`:1443-1449`).

F6. **Three device buffers are sized once at `EnableOpenCL` and never
    re-checked in `Compute`**: `TNNetSum.FSumBuffer` (`neuralnetwork.pas:79241`),
    `TNNetSplitChannels.FSplitBuffer` (`:98536`), `TNNetDeepConcat.FConcatBuffer`
    (`:98134`). `TNNetCellMulByCell` guards (`:53029`). The purpose-built
    helpers (norm, RoPE, GLU, conv1d, GDN, SDPA) are grow-only.

F7. **The multi-token SDPA device kernels exist but were never merged.**
    Commit `f2192858` on branch a11 gives all four SDPA entry points an
    `FTokenCnt` row axis (one work-group per (head, row)), score band
    `FQHeads*FTokenCnt*FCacheMax` floats, causal bound
    `LiveLen = FCacheSlot + t + 1`, and widened `WillOpenCL` to
    `SizeX <= 4`. Its message says the bound is 4 "because the score band
    grows linearly with it ... and a prefill belongs on the host path". The
    negative control (off-by-one LiveLen) blows parity to 0.25/1.68.

F8. **Never zero-pad a window on a hybrid** (a11 `d1e45f05`): a pad row
    advances recurrent state and `TruncateTo` undoes attention K/V only
    (`neuraldecode.pas:6052-6058`). The attention-only precedent
    `examples/SpeculativeDecoding` pads + truncates and gets away with it.

F9. **Twin nets are the established pattern for two widths.**
    `TMusicGenModel` keeps a prefill decoder and a width-1 step twin
    (`neuralpretrained.pas:33200-33245`); the Qwen builders take the width
    (`BuildQwen35FromSafeTensorsEx(..., pSeqLen)`, `:1334`). Weight sharing
    between nets does not exist: `CopyWeights` duplicates.
    `Snapshot()/RestoreSnapshot()` is sanctioned across twins of the same
    architecture and `MaxCacheLen` (`neuraldecode.pas:1364-1366`) and is
    what turn-boundary reuse already uses per turn.

F10. **Prefill logits are discarded.** The engine never reads the output of
    a prefill step; the LM head (248k x hidden at 27B, ~10% of the weight
    bytes) runs anyway on every prompt token. `TNNet.Compute` has a
    `FromLayerIdx` but no end index.

F11. **No test drives `StepForward` with K > 1** on a16. a11 has a reusable
    harness `RunWindowedDecodeStream(Session, Tokens, StepTokens, Logits)`
    (a11 `tests/TestNeuralPretrained.pas:9029`) and three tests worth porting
    (`TestQwen35WindowedDecodeParity`, `...OpenCLParity`,
    intra-window causality at tolerance 0).

F12. **Two residency bugs from a11 are still live here**: host writes need
    `MarkOutputWrittenOnRAM` (a11 `cd27f4cb`), and `HiddenState` returns a
    stale host copy under OpenCL (a11 `704a9366`). Prefill reads neither, so
    they are out of scope, but any reviewer will hit them if the plan drifts
    toward reading hidden states mid-window.

## 3. Decision

**Prefill on a width-K twin net; decode on the existing width-1 net; hand
the session state across with the existing snapshot.** The tail of the
prompt that does not fill a window is fed token by token through the width-1
net. No padding, ever.

Why this and not an in-place width switch: an in-place `ResizeSequenceWidth`
would need a per-class override for ~20 layer classes, a split of the matmul
helper so activations can be reshaped without re-uploading codes (F5), and
size guards on the three fixed buffers (F6). It is the better end state
(no weight duplication, no per-request handoff) but it is a large audit
with silent-failure modes, and it delivers nothing measurable until every
class is done. The twin delivers a measurable prefill number after Phase 1
and Phase 2, using kernels that already accept K columns (F4). The cost of
the twin is real and must be stated in the flag's help text: a second copy
of the weights in host RAM and on the device, and a second load/upload at
start-up. Phase 4 removes that cost.

Window width `K`: a new `--prefill-window N` option (default 0 = off, i.e.
today's path). Start the GPU sweep at 64; the constraints on K are
`FCacheLen + K <= FCacheMax` (F2), the SDPA score band
`FQHeads*K*FCacheMax*4` bytes (F7; 27B, 32k ctx, K=64 is ~270 MB), and
activation memory (K x every layer output).

## 4. Phases, each ending green with a commit

### Phase 0 - parity harness (CPU, this box)
- Port a11's `RunWindowedDecodeStream` into `tests/TestNeuralPretrained.pas`
  (or `TestNeuralDecode.pas` on `BuildTinyQwen35HybridLM`, `:3593`).
- Test A: full forward vs width-1 stream vs width-K windows (K=4 and a K
  that does not divide the prompt, so the tail path runs), tolerance 2e-4,
  on the tiny qwen3_5 fixture. Hybrid, so it covers GDN + conv1d + SDPA.
- Test B: intra-window causality at tolerance 0: two windows differing only
  in row 1 give bit-identical row-0 logits.
- Test C: a window starting at a NON-ZERO `AbsPos` compared against the
  width-1 stream. A11 `09311fbc` notes RoPE is relative, so a uniform
  position error is invisible to a contiguous full-vs-windowed comparison;
  this test is the one that catches it.
- Test D: snapshot handoff. Prefill on a width-K net, `Snapshot`, restore
  into a width-1 net, decode; compare with the all-width-1 run. Tolerance 0
  (the snapshot contract is bit-exact), int8 KV variant too.

### Phase 1 - engine window loop (CPU path works end to end)
- `TChatEngine.LoadModel`: when `--prefill-window K > 0`, build the twin
  with `BuildFromPretrained(..., pSeqLen := K)` and a second
  `TNNetStreamingDecoder` over it with the same `MaxCacheLen` and int8-KV
  arming. Check the importer for a build-without-load route before paying a
  second disk read; if none exists, accept the double load for now and
  record it.
- Prefill loop (`neuralchatengine.pas:1555-1599`): tokens to feed are
  `Reused .. Len-2` exactly as today. Feed `(count div K)` windows through
  the K-session, hand over with `Snapshot/RestoreSnapshot`, feed the
  `count mod K` tail through the width-1 session. Preserve: `Tokens[Len-1]`
  stays unfed; `CachedTokens`/`TurnSnapPos` truthfulness; `ClearTime` after
  the LAST prefill step; `TStart`/`TFirst` untouched; `ContextFull` check.
- Turn-boundary reuse: the restored snapshot goes into whichever session
  steps first. Define one `ActiveSession` and a `SwitchTo` that snapshots
  from the active one and restores into the other; the engine's turn
  snapshot is taken from the width-1 session as today.
- Debug-mode assert that the window volume width equals the active net's
  input width (F1 makes a mismatch silent).
- Stats: add `prefill X tok/s` = (prompt - reused) / (TTFT - one mean decode
  step) to the `[stats]` line, omitted when nothing was prefilled. The user
  already asked for this.
- Measure here on CPU with the tiny fixture for parity only; CPU speed on a
  real model is the user's box.

### Phase 2 - device path stays resident for a window
- Cherry-pick `f2192858` (neural.cl + FusedSDPA host side + its tests), then
  lift the `SizeX <= 4` bound. Bound the score band instead by scoring the
  window in row blocks inside `ComputeOpenCL` (append all K rows once, then
  launch decode-score per block of R rows with the per-row LiveLen), so the
  band is `FQHeads*R*FCacheMax`. R is a constant, not a flag.
- Add `FOutput.Size` guards to the three fixed buffers (F6) so a mismatch
  falls back loudly or re-creates at `EnableOpenCL`; with a twin they are
  right by construction, the guard is insurance.
- Port a11's `TestQwen35WindowedDecodeOpenCLParity` including the
  `ForwardGPUCnt` assertions: without them an unarmed device path compares
  CPU against CPU and proves nothing. Runs on PoCL here; timings do not.
- The user measures TTFT on the GPU box, sweeping K in {32, 64, 128, 256}.

### Phase 3 - skip the LM head during prefill (F10)
- Add an optional end layer to `TNNet.Compute` (serial and parallel
  scheduler) and a `StepForwardTo(HiddenLayer)` on the session; the K-net's
  prefill stops at the LM-head input slot. Saves the vocab projection and a
  `K x 248k` logits volume per window. Independent of Phase 2; can go first
  if Phase 2 stalls.

### Phase 4 - tiled quantized GEMM (only if Phase 2 measurement says the
### projections still dominate)
- `__local`-tiled int8 and int4 kernels for `FNumBs > 1` that read each
  weight row once per tile of columns. Template: the dead `myGEMM6`.
  Keep the split-K/GEMV pair for `FNumBs = 1`. Parity test against the
  existing kernel at 2e-6. This is the "weights read once per window"
  promise; F4 says the current kernels may or may not get it from L2.

### Phase 5 - remove the weight duplication
- Either share weight storage between the twins (FPC dynamic arrays share
  by assignment; device codes via `clRetainMemObject` and a
  `PrepareInt4DotCLFrom(Other)` that owns only its B/result buffers), or the
  in-place width switch described in section 3. Decide after Phase 2 with
  real numbers; the twin's memory cost is what makes this phase mandatory
  for the 27B, where it may not even fit twice.

## 5. Rules for the implementer

- No allocation inside any `Compute*` or decode helper (rule 17). Window
  volumes and snapshots are allocated once at `LoadModel`.
- `FErrorProc` prints and returns. Never rely on it to abort. Assert in
  Debug.
- `ForceOutputOnRAM` moves data and leaves `FOutputOnOpenCL` true (F12).
- One test, one build, then report. Wrap every run in
  `ulimit -v 3145728`. Force recompiles with `lazbuild -B`; stale PPUs
  produce fake OpenCL crashes.
- Do not touch the sampler, the decode loop or the SDPA decode-kernel
  parallelism (the context-growth problem in section 1 is a separate task).
- Comments state the current rationale, not history.
- State what was not verified. Nothing OpenCL is timed on this box.

## 6. Decisions taken by the user (2026-09-02)

1. A second copy of the weights (host RAM and device) is accepted as the
   price of Phases 1-5. Phase 5 removes it.
2. A second checkpoint read at start-up is accepted during Phases 1-4. The
   implementer still checks the importer for a build-structure-only route
   and uses it if one exists.
3. Validation runs on the 4B model first, then the 27B once Phase 5 has
   removed the duplication.

## 7. Cache checkpoints: `--cache-checkpoints N` (design, 2026-09-03)

### 7.1 The problem

A hybrid (Qwen3.5/3.8, Mamba) resumes a prompt only from a whole-state
snapshot, and the engine holds two: the end of the previous reply and the
end of the previous prompt (`neuralchatengine.pas:1997-2008`). Any prompt
that diverges before those two positions pays the full prefill. Measured on
the 27B (bpsa step 2, 8151 tokens, 53 s): `reused 0`, because bpsa appends a
suffix to the LAST user message on every request, so the previous task
message re-renders without it and the token ids diverge inside the previous
prompt. Every agent framework moves something near the end of the prompt;
the engine must reuse everything before the divergence, not nothing.

Reuse works on token ids, not characters. `CommonPrefixLen`
(`neuralchatengine.pas:887`) already yields the last shared token; BPE may
re-split the token just before the differing character, which costs at most
one or two tokens.

### 7.2 Facts (verified at 6fbfd037, the a18 tip on 2026-09-03)

F13. **Pure-attention nets already reuse up to the divergence.** The
     `CacheReuse` route computes the common prefix, calls
     `Session.TruncateTo(Reused)` (`neuralchatengine.pas:2037`) and prefills
     the tail. Nothing to add for Llama/Qwen2.5.

F14. **The hybrid's state splits into a truncatable half and a
     non-truncatable half.** Attention K/V is position-indexed:
     `TNNetScaledDotProductAttention.TruncateCache`
     (`neuralnetwork.pas:33172`) rewinds `FCacheLen` and the next append
     overwrites. The recurrent state (`TNNetGatedDeltaNet.FDecS`,
     `TNNetDepthwiseConv1D.FDecHist`, every `TNNetRecurrentDecodeBase`
     descendant, `neuralnetwork.pas:9532`) is one fixed-size volume with no
     per-position history. Only that half needs a checkpoint.

F15. **The recurrent half is small; the K/V half is the snapshot's bulk.**
     GDN state is `FNumVHeads * FHeadDimK * FHeadDimV` floats
     (`neuralnetwork.pas:70776`); conv history is `(KernelSize-1) * Depth`.
     On the 27B (48 GDN layers, 64 value heads of 128 x 128 by the published
     config, NOT verified against the user's `config.json`): about 4 MB per
     GDN layer, ~192 MB per checkpoint. The int8 K/V of 16 attention layers
     at 32k context is ~1.1 GB, which is what each of today's two snapshots
     copies (`TNNetStreamingDecoder.SnapshotInto`, `neuraldecode.pas:6129`).

F16. **Today's capture and restore cross the bus.**
     `TNNetGatedDeltaNet.CaptureState` (`:70787`) calls
     `ForceDecodeStateOnRAM` (a queue drain plus a blocking read) and
     `RestoreState` (`:70795`) copies on the host and clears `FDecSOnOpenCL`,
     so the next forward re-uploads. `TNNetDepthwiseConv1D` (`:62199`,
     `:62207`) does the same with `FDecHist`. A checkpoint taken after every
     prefill window through these routes would repeat the cost that made
     the MTP verify window slow. The device-side copy that fixed it there
     (commit `eca33396`, `MarkStateCheckpointOnOpenCL`) exists ONLY on branch
     `a11`, which never worked well and is not merged: read it for the shape
     of a `clEnqueueCopyBuffer` between two resident buffers, trust nothing
     from it, write and test fresh.

F17. **`TruncateCache` downloads the resident cache first.** Its first line
     is `ForceCacheOnRAM()`, which `TNNetFusedSDPA` overrides
     (`neuralnetwork.pas:34958`) with a blocking `DownloadCache[Int8]` that
     also clears `FCacheOnOpenCL`, so the next forward re-uploads the whole
     cache. The rewind itself is one host field; the kernels take
     `FCacheSlot` per launch and own no length. The download is therefore
     unnecessary for the rewind and must not be paid on the resume path.

F18. **State lives in whichever twin is active.** The width-N twin, the
     tail twin and the width-1 session are three nets with their own layers;
     `SwitchTo` (`neuralchatengine.pas:1175`) moves state between them by a
     full snapshot copy. A checkpoint captured on the twin's layers during a
     window prefill must be restorable into the width-1 session's layers, so
     it cannot live inside one layer's OpenCL helper. The three nets share one
     OpenCL context (`EnableOpenCLInContextOf`, Phase 5a), so a `cl_mem`
     owned by an object outside the nets is reachable from all three.

F19. **The resume already runs on the width-1 session.** The `CacheReuse`
     route truncates `Session`, sets it active and calls
     `SwitchTo(FirstSession)` (`neuralchatengine.pas:2035-2040`). The
     checkpoint route slots into the same place: truncate `Session`, restore
     the recurrent half into `Session`, then `SwitchTo`.

### 7.3 Design

A **cache checkpoint** is: a position (tokens fed), plus one copy of the
recurrent state and step count of every `TNNetRecurrentDecodeBase` layer of
the session. It holds NO attention K/V. The engine keeps up to N of them in
a **checkpoint store** allocated once at `LoadModel` (rule 17), N given by
`--cache-checkpoints N`.

**Capture points.** After every window the width-N twin feeds, after every
tail-twin window, at the end of the prompt (where `PromptSnap` is taken
today, `:2068`) and at the end of the reply (where `TurnSnap` is taken,
`:2243`). Under `--gpu` a capture is one `clEnqueueCopyBuffer` per layer from
the layer's resident state buffer into the store's slot buffer, on the
layer's own queue; on CPU it is `CaptureState` into the slot's host volume.
Nothing is allocated per capture.

**Resume.** `Reused := CommonPrefixLen(CachedTokens, PromptIds)`, capped at
`Len - 1` as today. Pick the checkpoint with the largest position `<=
Reused`. If none, full reset. Else `Session.TruncateTo(Pos)`, restore the
recurrent half into `Session` from the slot (device-to-device under `--gpu`),
`Reused := Pos`, and continue down the existing ladder. The two whole-state
snapshots (`TurnSnap`, `PromptSnap`, `TransferSnap` stays) and the 2 x 1.1 GB
they hold go away; the end-of-reply and end-of-prompt captures are ordinary
checkpoints at N >= 2.

**Retention (which N to keep).** Let `E` be the current fed position and
`d = E - Pos` a checkpoint's distance from it. Divergence is far more likely
near the end of a prompt than near its start, so slots are spent
geometrically: with window `W` and context `C`, band `k` covers distances
`[W * r^k, W * r^(k+1))` for `k = 0 .. N-1`, with `r = (C / W)^(1 / N)`. The
store keeps the deepest checkpoint (largest `Pos`) in each band; on every
capture and at the end of every request it recomputes the bands against the
new `E` and drops the rest. The end-of-reply and end-of-prompt checkpoints
are always kept (they are the two innermost). Bound: a divergence at
distance `d` resumes from the nearest kept checkpoint beyond it, so the
extra prefill is at most `(r - 1) * d` on top of the unavoidable `d`. At the
defaults (`W = 256`, `C = 32768`):

| N | r | extra prefill, worst case | memory at 27B |
| --- | --- | --- | --- |
| 8 (CPU default) | 2.00 | 1.0 x d | 1.5 GB host |
| 16 (`--gpu` default) | 1.38 | 0.38 x d | 3.0 GB OpenCL memory |

`--cache-checkpoints 0` turns the checkpoint route off (a hybrid then
re-prefills every turn, as `--no-cache-reuse` does). Values 1 and above
2048 are rejected at `LoadModel` with an error and a hard stop, like the
window flags. `--no-prompt-snapshot` is removed: its job (skip the
end-of-prompt capture) has no meaning once that capture is one cheap slot.

**Diagnostic.** The `[stats] input:` line becomes
`prompt %d tokens (reused %d, prefix %d of %d cached)`, so a `reused 0` in a
server log states where the ids diverged without a client-side dump.

### 7.4 Agents, serial, each ending green with a commit

**Agent 1 - session layer** (`neuraldecode.pas`, `neuralnetwork.pas`,
`neuralopencl.pas` if a helper needs a buffer accessor). Deliverables:

1. `TNNetRecurrentDecodeBase`: `CaptureStateToOpenCL(Slot: cl_mem)` and
   `RestoreStateFromOpenCL(Slot: cl_mem)` (device-to-device copies on the
   layer's queue, leaving residency flags TRUE), plus `StateBytes()` so a
   store can size its slots. Descendants without a resident state fall back
   to `CaptureState` / `RestoreState`.
2. `TNNetFusedSDPA.TruncateCache` override (or a base change) that rewinds
   `FCacheLen` WITHOUT `ForceCacheOnRAM` (F17), with the same range checks.
3. `TNNetDecoderStateCheckpoint` (per-layer host volumes + step counts, or
   per-layer `cl_mem` slots when the session runs OpenCL) and on
   `TNNetStreamingDecoder`: `CaptureStateInto(Chk)`, `RestoreStateFrom(Chk)`,
   both usable across the twin sessions (F18).
4. Tests (`tests/TestNeuralPretrained.pas`, the tiny Qwen3.5 fixture): run to
   position p, checkpoint, run on to q, `TruncateTo(p)` + restore, run the
   same tail again, compare hidden states bit for bit with an uninterrupted
   run. CPU, OpenCL, FP32 KV, int8 KV, and capture on the width-N twin with
   restore into the width-1 session. A test that asserts the resident cache
   stayed resident across `TruncateTo` (F17), and one that asserts the
   recurrent state stayed resident across a capture (F16).

**Agent 2 - engine and programs** (`neuralchatengine.pas`, `ChatTerminal.lpr`,
`ChatServer.lpr`, `examples/ChatTerminal/README.md`). Deliverables:

1. `--cache-checkpoints N` with the validation, defaults and notices above;
   `--no-prompt-snapshot` removed everywhere it is mentioned.
2. The checkpoint store with the retention rule, the capture points, and the
   resume path in `GenerateFromIds`; `TurnSnap` / `PromptSnap` deleted.
3. The `prefix %d of %d cached` stats diagnostic; `LastReusedTokens` keeps
   its meaning (tokens actually resumed from).
4. Tests: extend the `TestQwen35ChatPromptSnapshot*` family
   (`tests/TestNeuralPretrained.pas:350-353`) into `TestQwen35ChatCheckpoint*`
   with divergence at the start, inside the previous prompt, at the
   prompt/reply boundary and inside the reply; each compares the reply and
   the reused count with a fresh engine; ladder, int8 KV and OpenCL variants;
   a retention test that feeds a long prompt and checks the kept positions
   against the band rule; a flag-validation test.

**The user measures after Agent 2** with the bpsa run of 2026-09-03. Success:
request 3 reports `reused` within one window of the divergence (about 7600
or above), and the process holds about 2.2 GB less host RAM.

### 7.5 Traps for both agents

- Rule 17: slots are allocated at `LoadModel` / session build. A capture or
  restore allocates nothing. `SnapshotInto` shows the reuse pattern.
- A capture must copy the RESIDENT buffer. Copying the host mirror of a
  layer whose flag says the state is in OpenCL memory saves stale data, and
  the suite cannot see it unless a test asserts residency (the sentinel
  trap in the residency notes).
- `FErrorProc` prints and returns. Range errors in `TruncateCache` and the
  checkpoint store raise in Debug.
- `LoadModel` validates the flag and fails hard; a silently ignored value
  cost a 27B load once already.
- Distrust the summary and this section alike: verify every cited line
  before building on it. Both suites, `lazbuild -B`, one OpenCL process at a
  time, `ulimit -v 3145728`. Nothing OpenCL is timed on this box.
- Comments state the current rationale; the two-snapshot history belongs in
  the commit message.
