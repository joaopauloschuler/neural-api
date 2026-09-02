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
