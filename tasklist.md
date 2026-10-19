# Task List — Feature & Enhancement Ideas

## Authorship convention (AI-coded classes)

Every class that was newly added to `neural/neuralnetwork.pas` by Claude
(i.e. not present in the upstream `../neural-master` baseline) carries an
attribution comment as the **last comment line directly above the class
declaration**, written exactly as:

```
  // Coded by Claude (AI).
```

Rules:
- One attribution per **class** (not per method), placed immediately above
  the `TNNet... = class(...)` line, after any `///`/`//` doc comment.
- Use the literal text `// Coded by Claude (AI).` (plain `//`, not `///`,
  trailing period) so it can be audited with
  `grep -c "Coded by Claude" neural/neuralnetwork.pas`.
- Applies only to genuinely **new** classes. Do NOT retrofit it onto
  pre-existing upstream classes that were merely edited.
- Human-authored hand-coding of new classes is no longer the norm here;
  new classes are Claude-authored and should be marked as such.

### Example programs (`examples/**/*.lpr`)

Every example program newly added by Claude (i.e. not present in the
`../neural-master` baseline) carries the attribution `Coded by Claude (AI).`
inserted with a blank-line separator immediately **before the closing `*)`**
of the file's header `(* ... *)` license comment block.

Rules:
- One attribution per file, inside the header comment block (so it never
  affects compilation).
- Applies only to genuinely **new** example `.lpr` files; skip stray
  `backup/` copies. Identify "new" by diffing `find examples -name '*.lpr'`
  against the `../neural-master` baseline.
- Audit with `grep -rl "Coded by Claude" examples --include='*.lpr' | wc -l`.

## DO NOT REINTRODUCE — removed by intent

The following layer types were intentionally removed because they
duplicated the forward pass of existing layers:

- `TNNetBias` — duplicate of `TNNetChannelBias` (forward pass).
- `TNNetLayerScale` (and its alias `TNNetLearnableScale`) — duplicate of
  `TNNetChannelMul` (forward pass).
- `TNNetNeg` — duplicate of `TNNetNegate` (which is itself just
  `TNNetMulByConstant.Create(-1)`). Use `TNNetNegate` for `y = -x`.
- `TNNetGlobalAvgPool` — empty-body subclass of `TNNetAvgChannel`. Use
  `TNNetAvgChannel` for global average pooling.
- `TNNetGlobalMaxPool` — overlapped `TNNetMaxChannel`. Use
  `TNNetMaxChannel` for global max pooling. `TNNetMaxChannel` now does a
  true (x, y)-over-all-positions reduction per depth channel, so it is
  correct on RECTANGULAR (`SizeX <> SizeY`) inputs too — no reason to
  reintroduce this class.
- `TNNetGlobalMinPool` — overlapped `TNNetMinChannel`. Use
  `TNNetMinChannel` for global min pooling. `TNNetMinChannel` is now
  rectangular-correct as well (see `TNNetMaxChannel` above).
- `TNNetThresholdedReLU` — strict subset of `TNNetThreshold`.
  `ThresholdedReLU(x; θ)` is `TNNetThreshold.Create(theta=θ, value=0)`.
  Use `TNNetThreshold` directly.

Do NOT add them back under any name. The minor differences they had
(true-sum vs spatial-mean weight-gradient scaling; constructor-
configurable initial scale) are not worth the code duplication. If a
true-sum gradient or a configurable initial multiplier is genuinely
needed, extend `TNNetChannelBias` / `TNNetChannelMul` with an option
flag instead of forking a new layer class. Any task entry below that
references these removed layers is obsolete and should be ignored
rather than acted on.


## Bugs

- [x] BUG: `SetTrainable(False)` then `SetTrainable(True)` did not give back a
      trainable layer. `TNNetLayer.SetTrainable(False)` shrinks `FOutputError` /
      `FOutputErrorDeriv` to 1x1x1; the True branch only set flags, so nothing
      regrew them. Effects: `TNNetFullConnect.BackpropagateCPU` wrote past the
      end of both buffers, convolutions kept the `FCalculatePrevLayerError` flag
      and `FPrevLayerErrorPadded` size they cached at wiring time, a layer
      flipped from low-memory inference kept its 1x1x1 concatenated weight
      caches (wrong output), and a `TNNetInput(pError=1)` lost its input
      gradient. Adam moments are freed by the False flip and are not restored:
      run `InitAdam` again after the True flip.
  - [x] Part 1 (Done 2026-10-18, commit 4c39cee9): `TNNetLayer.SetTrainable(True)` on an
        inference-only layer regrows the error buffers to `FOutput` (a class that
        uses `SetOutputErrorSizeNoDeriv` keeps its deriv buffer at 1x1x1) and runs
        `AfterWeightUpdate` when low-memory mode was on. Quantized and
        weight-borrowing layers (`HasInferenceOnlyWeights`) stay inference-only.
        `TNNetInput` regrows its buffers when `FStruct[3]=1`. `TNNet.SetTrainable`
        then runs the new `TNNet.RefreshPrevLayerErrorCaches` (conv family:
        flag + padded error buffer; `EnableInputGradient` uses it too). Size
        checks added in `ComputeOutputErrorForIdx`, `ComputeErrorDeriv`,
        `TNNetFullConnect.Backpropagate` and `TNNetFullConnectDiff.Backpropagate`.
        New tests `TestSetTrainableRoundTripMatchesTwin` (8 nets x 4 flip modes vs
        a never-flipped twin, 1e-6) and `TestSetTrainableKeepsInferenceOnlyWeights`.
        Suites green: cpu 3283 (+1 run alone), avx2 3284, opencl 3284; 0 errors,
        0 failures. Not done in part 1: re-arming OpenCL after a flip.
  - [x] Part 2: the 35 layers whose `SetTrainable` override only frees
        (`if not pTrainable then FreeBackpropScratch()`), plus AconC, MetaAconC,
        DyT, PReLUChannel and SReLU, lose backprop scratch that the True flip
        does not size again. Add a per-class
        `SizeBackpropScratch` helper called from `SetPrevLayer` and from the
        `SetTrainable` True branch.
        The 35 overrides in file order, split in two batches:
    - [x] Batch A (Done 2026-10-18, commit 6caa428a): ArcFace, ForgetGateBias,
          InducedSetAttention, ProductKeyMemory, AttentionPooling, Retention,
          CrossAttention, T5RelPosBiasAttention, DifferentialAttention,
          LinformerAttention, GatedResidual, CapsuleRouting, HardConcrete, GRN,
          APL, SplineActivation, Invertible1x1Conv, MLSTMCell. Each got a private
          `SizeBackpropScratch` (sizes what `FreeBackpropScratch` frees), called
          from `SetPrevLayer` and from the True branch when `FIsTrainable` and
          `FPrevLayer` is assigned. T5RelPosBiasAttention and DifferentialAttention
          need nothing: their `Backpropagate` resizes the scratch. New test
          `TestSetTrainableRoundTripScratchLayersA` (18 nets x 4 modes, 1e-6,
          shared `CheckSetTrainableRoundTrip` helper); fails on the old code under
          `-Cr` with a range check error in `TNNetArcFace.Backpropagate`. Suites
          green: cpu, avx2, opencl 3285 each, 0 errors, 0 failures.
          Found on the way: `TNNetLinformerAttention.Backpropagate` only
          accumulates deltas and never applies them when `FBatchUpdate` is false
          (the helper now calls `TNNet.UpdateWeights`); not changed.
    - [x] Batch B (Done 2026-10-18, commit f351b7ac): DeltaNet, GatedDeltaNet,
          GatedLinearAttention, NTMMemory, TestTimeTraining, TitansMemory,
          SpectralConv1D, DWT1D, SpectralConv2D, CondConv, MonarchLinear,
          KroneckerLinear, SoftDecisionTree, HouseholderLinear, HyperbolicLinear,
          HyperbolicDistance, GraphAttention (17 overrides). Each got the batch-A
          `SizeBackpropScratch`, except GatedDeltaNet, whose override already
          re-runs `SizeSequenceCaches`. AconC, MetaAconC, DyT, PReLUChannel and
          SReLU got `FreeBackpropScratch` + `SizeBackpropScratch` + a
          `SetTrainable` override (free on False, size on True). No other class
          needed it: the remaining `if FIsTrainable` blocks in `SetPrevLayer`
          size neuron Delta/BackInertia (regrown by `TNNetNeuron.SetTrainable`)
          or already have a regrowing override. The NoForward mask
          (TNNetDotProducts, TNNetPointwiseSoftMax) is rebuilt on the True flip
          and built in `SetPrevLayer` only when trainable (an inference-only
          DotProducts(NoForward) used to write the mask past its 1x1x1 buffer).
          `TNNet.EnableInputGradient` now calls `TNNetInput.EnableErrorCollection`.
          examples/GPT2Import/README.md no longer says an inference-only net can
          never train. New test `TestSetTrainableRoundTripScratchLayersB` (26
          nets x 4 modes, 1e-6; DeltaNet lr 0.2, HyperbolicDistance lr 0.01 with
          c=0.01); on the old code under `-Cr -Ct` every class but GatedDeltaNet
          fails. Suites green: cpu, avx2, opencl 3286 each, 0 errors, 0 failures.
- [x] BUG: `TNNetLinformerAttention.Backpropagate`
      accumulates its weight deltas but never applies them when `FBatchUpdate`
      is false, so the layer does not train outside batch mode unless the
      caller runs `TNNet.UpdateWeights`. Found during SetTrainable part 2,
      batch A.
      Done 2026-10-18: `Backpropagate` now ends with the sibling-layer tail
      (`if not FBatchUpdate` -> `UpdateWeights(FInertia)` on E and F, then
      `AfterWeightUpdate`); batch mode still accumulates for
      `TNNet.UpdateWeights` (base `TNNetLayer.UpdateWeights` / Adam / ClearDeltas
      already cover both neurons; Linformer has no Q/K/V/O neurons, only E/F).
      New test `TestLinformerAttentionTrainsWithoutBatchUpdate` (inertia 0 and
      0.9: E/F move and their deltas are cleared after one per-sample step, and
      match a batch-mode step + `UpdateWeights` within 1e-6); fails on the old
      code. The weight/input gradient checks already existed. The
      `CheckSetTrainableRoundTrip` helper no longer calls `TNNet.UpdateWeights`.
      Suites green: cpu, avx2, opencl 3287 each, 0 errors, 0 failures.
      Scan of every `Backpropagate*` that writes neuron deltas without a
      non-batch apply found no other layer with the same defect (the conv /
      depthwise / local-connect `BackpropagateAtOutputPos` and the circulant
      FFT helper are called from routines that apply). The next item is a
      related defect of a different shape.
- [x] BUG: `TNNetWeightStandardizationConv.Backpropagate`
      (neural/neuralnetwork.pas ~100865) and `TNNetSpectralNormConv.Backpropagate`
      (~101080) remap the per-call delta increment AFTER `inherited Backpropagate`.
      Outside batch mode the inherited conv backprop
      (`BackpropagateFastTiledCPU`) has already applied and cleared the deltas,
      so the raw weights get the w_hat / w_eff gradient unmapped and the remap
      sees a zero increment. Batch mode is correct.
      Done 2026-10-18 (430b02a4): worse than reported - the inherited step went
      to the in-place standardised / scaled copy, which `RestoreRawFilters`
      overwrote, so the raw weights never moved (only the bias did). Both
      overrides now run the inherited backward with `FBatchUpdate` forced true,
      remap, then call `UpdateWeights()` when not in batch mode and the output
      error is non-zero (the inherited `MaxError > 0` guard). `TNNetConvolution`
      is unchanged; the conv OpenCL backward is disabled, so both layers use the
      CPU path. New tests `TestWeightStandardizationConvTrainsWithoutBatchUpdate`
      and `TestSpectralNormConvTrainsWithoutBatchUpdate` (inertia 0 and 0.9;
      step and next forward match batch + `UpdateWeights` within 1e-6 / 1e-5)
      fail on the old code; `TestSpectralNormConvWeightGradientMatchesScaledConv`
      adds the missing raw-weight check (constant-sigma, so against a plain conv
      holding W/sigma, not finite differences). Scan of every `Backpropagate`
      override in neural/ for delta work or weight restore after `inherited`
      found no other layer with this shape. Suites green: cpu, avx2, opencl 3290
      each, 0 errors, 0 failures.
- [ ] Possible defect (not fixed, awaits user decision): dense `TNNetSpectralNorm`
      (neural/neuralnetwork.pas ~13905) keeps the inherited FC weight delta
      (gradient w.r.t. w_eff) without the 1/sigma factor its conv sibling now
      applies; its declaration comment calls this intended, but dL/dW =
      (1/sigma) dL/dw_eff under the same constant-sigma rule.
- [x] `TestSetTrainableKeepsOutputs` fails ONLY on `-dAVX2` builds (~1e-7 logit
      drift) — pre-existing, unrelated to audio/Mimi (confirmed by `git stash`
      during the RunMimiConv AVX work, commit 30c2342). The scalar build passes.
      Likely an AVX-vs-scalar reassociation crossing the test's exact-equality
      tolerance; either loosen that one assertion to a small epsilon or pin the
      reassociation. Whole suite is otherwise 0/0 on both builds.
      FIXED (commit b04218db): the two forwards did not use the same kernel.
      SetTrainable's pLowMemory defaults to True, which drops the concatenated-
      weight caches and computes per-neuron - a different accumulation order from
      the interleaved/pointwise kernels (5.6e-7 relative on the conv diamond,
      1.3e-6 on tiny GPT-2). The test now compares the low-memory build at 1e-4
      and pins pLowMemory=False for the exact-equality half. Both builds green.
- [X] BUG (FIXED): `MRoPEOpenCLIncrementalCacheParity` (TestNeuralNumerical.pas, proc at
      ~line 63088) fails NONDETERMINISTICALLY — `max|diff|` swings between
      `0.000000000` (pass) and garbage like `1.86e34` (fail, asserted `< 1e-6`).
      ROOT CAUSE confirmed = the suspected `FOutput` SizeX mismatch below. SetPrevLayer
      (TNNetIdentity) sizes `FOutput` to the FULL `(MaxLen,1,Depth)` and RoPE's
      Compute/ComputeOpenCL never resized it per step, so on an incremental forward
      (`SeqLen < MaxLen`) `FMRoPECL.Rotate` allocated a `MaxLen*Depth` device output
      buffer (`CreateOutputBuffer(FOutput)`), the kernel wrote only the `SeqLen*Depth`
      prefix, and `ReadBuffer` copied the WHOLE buffer back — so the tail
      `[SeqLen*Depth..MaxLen*Depth)` was uninitialized device memory (nondeterministic
      garbage). Both cached and reference paths copy the full `FOutput`; identical
      prefix, differing garbage tail → exploding `max|diff|`. FIX: resize `FOutput` to
      `(SeqLen,1,Depth)` before the `Rotate` round-trip in BOTH
      `TNNetMRotaryEmbedding.ComputeOpenCL` (~39664) and `TNNetRotaryEmbedding.
      ComputeOpenCL` (~39129) — no-op when SeqLen is the full length. Verified: test
      now `max|diff|=0.0` deterministically over 5 isolated runs; all RoPE/MRoPE
      forward+parity+grad tests pass; full numerical suite 1307/1307; clean
      non-OpenCL `-B` build. Original diagnosis kept below for the record.
- [x] (original report) `max|diff|` swings between
      `0.000000000` (pass) and garbage like `1.86e34` (fail, asserted `< 1e-6`).
      It passed 1306/1307 inside the full suite on one run but fails ~5/5 when run
      in ISOLATION (`./RunTests --suite=MRoPEOpenCLIncrementalCacheParity`); the
      outcome depends on surrounding allocations, i.e. an UNINITIALIZED-MEMORY /
      heap over-read, not a logic error. Surfaced during the per-layer
      `WillOpenCL`/`FShouldOpenCL` dispatch refactor (drop of the global
      `NeuralConvOpenCLMinWork` gate): adding the `FForceOpenCL: boolean` field to
      every `TNNetLayer` shifted heap layout and EXPOSED a pre-existing latent bug
      — it is NOT caused by the dispatch change. Evidence it is pre-existing/
      memory, not dispatch logic: (1) the M-RoPE dispatch predicate is
      functionally identical before/after (old guard `FHasOpenCL and FShouldOpenCL
      and Assigned(FMRoPECL) and (FOutput.Size >= NeuralConvOpenCLMinWork)` forced
      via the test's old `SetNeuralConvOpenCLMinWork(0)`; new
      `Assigned(FMRoPECL) and FHasOpenCL and (FShouldOpenCL or FForceOpenCL)`
      forced via the test's new `NN.ForceOpenCL(True)` — both dispatch the GPU
      path on every step); (2) on the PRISTINE tree the test passes 8/8 in
      isolation (rebuilt `-dOpenCL`, PoCL CPU device); (3) the NON-incremental
      `M-RoPE OpenCL parity` test passes (~6e-8), so only the incremental-cache
      REUSE path is affected; (4) the value is a garbage float, and context-
      dependence = uninitialized read. Ruled out so far: `TVolume.ReSize`
      (neuralvolume.pas ~6080) uses `SetLength(FData,...)` which PRESERVES the
      cached prefix on grow, so `FAngleTable`'s reused rows `[0,FAngleCacheRows)`
      survive; `BuildThetaCache`/`FTheta` and the SetPositions length check run
      BEFORE the `WillOpenCL` dispatch so `FTheta[k]` is valid; `HalfD` is held
      constant by the `FCachedHalfD = HalfD` reuse guard. Suspected remaining
      cause: a heap over-read in `TNNetMRotaryEmbedding.ComputeOpenCL`
      (neuralnetwork.pas ~39622) and/or the `FMRoPECL.Rotate(FPrevLayer.FOutput,
      FAngleTable, FOutput, SeqLen, Depth, HalfD, FOutScale)` device-buffer
      handling — note `FOutput` is NOT resized per incremental step (stays at the
      SetPrevLayer full `(MaxLen,1,Depth)`) while `FPrevLayer.FOutput` IS resized
      to the length-`step` prefix by the test, so the kernel runs with mismatched
      input/output `SizeX` (input SizeX=step, output SizeX=MaxLen) though both
      stride by `Depth`; verify the OpenCL input/output buffer sizes and the cache
      snapshot arrays (`FCachedPosT/H/W`, sized via `SetLength(...,SeqLen)`) for an
      out-of-range index when `SeqLen < FOutput.SizeX`. Repro: build RunTests with
      `-dOpenCL` (PoCL device present), `./RunTests --suite=MRoPEOpenCLIncrementalCacheParity`
      a few times. Next step: run under valgrind/heaptrc to pinpoint the exact
      uninitialized read, then fix the M-RoPE GPU incremental cache (this is GPU
      kernel/cache code, separate from the dispatch refactor).
- [ ] FFT-path FPU denormal/invalid-op traps in TNNetSpectralConv2D needed an
      example-side SetExceptionMask workaround — consider masking/guarding the
      denormals inside the layer's FFT so callers don't have to.
- [ ] SentencePiece `.model` `precompiled_charsmap` parsing — the opaque
      per-model normalization trie (NOT standard NFKC) is still not
      parsed/applied. A tokenizer.json that declares a standard NFKC/NFKD
      normalizer already works in full (`UnicodeNormalize` + `AddNormalizer`,
      landed); only the embedded charsmap is unhandled.
- [X] `TNNetKANConv` OpenCL forward allocated a `Taps²`-blown-up result buffer
      and OOMed/segfaulted for any realistic conv shape (the `{$IFDEF OpenCL}`
      GEMM path materialized the FULL all-pairs `cai_dot_product` cross product
      `NumAs x NumBs = OutDepth*NumPos*Taps^2`, then kept only the tap-diagonal).
      Concrete repro (Input 32x32x64 -> `TNNetKANConv.Create(64,3,1,1)`, degree 3):
      result = 21,743,271,936 floats (~81 GB) -> `clCreateBuffer :-61` then an
      uncatchable `EAccessViolation` inside the driver. FIXED (commit on `a3`):
      the wanted output `out[p,oo] = sum_tap sum_k A[oo,tap,k]*B[p,tap,k]` is one
      dot product over the COMBINED `Taps*C` axis, so KANConv now lays A out as
      `[OutDepth x (Taps*C)]` (B was already in that layout), calls
      `PrepareForCompute(..., Taps*C)` with `NumAs=OutDepth, NumBs=NumPos`, and
      reads `Res[p*OutDepth + oo]` directly. Buffer drops `OutDepth*NumPos*Taps^2`
      -> `OutDepth*NumPos` (~331,000x at that shape: 81 GB -> ~256 KB) with NO
      wasted off-diagonal compute - on POCL it went from OOM to a 6.43x speedup.
      SHARED PATTERN (RESOLVED) — the same all-pairs-then-extract-diagonal offload
      was used by three more layers; status after this batch:
      * `TNNetDepthwiseConv1D` (~51931, was `Channels^2*SeqLen`, channel-diagonal)
        and `TNNetDepthwiseConv` (~48751, was `NumAs*NumBs`, depth-diagonal) are
        GENUINELY block-diagonal (each group feeds a different output, so the
        group axis canNOT be folded into the contraction the way KAN's tap axis
        can). They now TILE the B axis so the transient result never exceeds
        `csOpenCLMaxGemmResElems` (2^24 floats = 64 MB) - numerics identical to the
        un-tiled GEMM (DepthwiseConv1D is bit-exact), no OOM at any shape. The
        per-group cross-product waste remains (a perf, not correctness, issue);
        a true batched/grouped dot-product kernel would remove it - left as a
        follow-up.
      * `TNNetDeconvolution.ComputeOpenCL` (~86547 `FGemmRes.ReSize(NumBs*NumAs)`)
        was MISCLASSIFIED here: its scatter step consumes the FULL `NumBs x NumAs`
        result (every input cell x tap x out-channel), so it is a legitimate dense
        GEMM, not a diagonal-extraction waste. No change; buffer is proportional to
        real deconv FLOPs.
      Verified on the POCL CPU device via a device-vs-CPU parity harness: all
      four layers fire on the device with max abs diff at f32 rounding (~1e-6),
      and the two previously-OOMing shapes (KAN 81 GB, DepthwiseConv1D 2 GB) now
      run. `OpenCLForwardBenchmark` re-includes `TNNetKANConv`.

## Hardware Support
  - [ ] CUDA/HIP Support
    - [ ] Software Spec 
  - [ ] ARM SIMD: NEON → SVE → SME kernels behind the TNNetVolume dispatch (requires ARM processor) 
    - [ ] Software Spec

## Infrastructure / dev experience

- [ ] OpenCL test suite under 2 minutes (user goal and authorization
      2026-10-05; today ~600 s on PoCL). Diagnosis (single-test runs): idle
      scheduler workers spin hot against PoCL (one VAE test: OpenCL + parallel
      pass 95 s vs 3.3 s serial, 415k clock syscalls); neural.cl is built again
      for every EnableOpenCL; a few heavy test shapes. Serial tasks, each with a
      fresh read-only Opus reviewer, the three suites and a commit:
  - [x] S1. Scheduler: with OpenCL armed, only worker 0 stays hot (OpenCL layers
        all go to it on a shared kernel); the hot loop reads the clock every N
        passes, not every pass (PrepareInferenceThreads in neuralpretrained.pas,
        StartThreadWorkers and the hot loop in neuralnetwork.pas). Coverage
        identical. Also a product fix: frees CPU cores during OpenCL passes.
  - [x] S2. Build neural.cl once per process and share the context and program,
        keyed by platform, device, source and build options. Coverage identical.
  - [x] S3. Shrink the slowest test shapes (depth-4096 norm chains, Reptile
        meta-iterations, the tiled VAE grid, the Qwen-Image pipeline steps and
        tokens) without weakening what they check; also helps the default and
        AVX2 suites.
  - [x] S4. Fewer distinct work-group sizes per kernel (e.g. cai_volume_sum 6,
        cai_dot_product 15+): each size is a separate cold PoCL compile, ~200 s
        on the first run after a neural.cl edit. User-approved 2026-10-05.
  - [x] S5. Move the end-to-end Qwen-Image OpenCL tests (PipelineOpenCLKeepLoaded,
        EditPipelineOpenCL, the VAE shared-output pair, TransformerOpenCLAttention;
        or merge KeepLoaded with PipelineOpenCL) to an opt-in slow runner. Reduces
        the default coverage; user-approved 2026-10-05.

- [ ] ChatServer: multiple chat sessions in cache, so parallel chats do not
      re-prefill (design agreed with the user 2026-10-02). Today TChatEngine is
      single-session (one KV cache, one position; neuralchatengine.pas header):
      each request is diffed (CommonPrefixLen) against the ONE resident token
      sequence, so two clients taking turns diverge right after the shared system
      prompt and every request pays a near-full prefill; for hybrid/recurrent nets
      DropCheckpointsAbove also frees the other chat's checkpoints. Compute stays
      serialized (one request at a time). Process: one Opus coder per task, then
      an independent Opus review, commit after review + green suites
      (lazbuild -B; default AND -dAVX2 suites; the PoCL suite when OpenCL code is
      touched; 3 GB ulimit). Each task ships its own tests.
  - [x] T1. Token-prefix hash primitive. A running 64-bit hash over token ids
        (h := Mix(h, Token[i]), {$PUSH}{$Q-}{$R-} around the wrapping math) plus
        a last-64-token guard compared exactly on a hash match. Search for an
        existing 64-bit mix first (coding guide #3). Tests: equal prefixes give
        equal hashes, a one-token change at any position changes it, the guard
        rejects a forced hash collision.
  - [x] T2. One eviction policy + hash matching for the cache checkpoints.
        Per checkpoint slot: PrefixHash, the guard tokens and LastUsedTurn
        (engine-side records; one GenerateFromIds call = one turn). Replace
        CheckpointBand, RetainCheckpointsBefore and DropCheckpointsAbove with
        ONE routine, DeleteTheLongestUnusedCheckpoint, run only when a capture
        finds no free slot (ties: the shallowest goes first). A request never
        frees a checkpoint because the prompt diverged. Resume: ONE pass over the
        prompt computes the running hash, finds the deepest matching checkpoint
        and the live-cache common prefix together (no second CommonPrefixLen
        pass). Only the checkpoint a request actually resumes from (and a new
        capture) is marked used this turn; a matching checkpoint that is not
        resumed keeps its old turn, so unused ones age out. CORRECTNESS: on a net with
        attention layers a checkpoint is resumable only when its Position <= the
        live-cache common prefix (its K/V rows must still be in the live cache);
        a pure recurrent net has no such limit. Tests: a growing single
        conversation still resumes at the previous reply end; a diverging prompt
        leaves the old checkpoints in place; LRU order of eviction; resumed
        output equals a full re-prefill.
  - [x] T3. Capture points. Capture at the end of the system prompt, the end of
        the prompt and the end of the reply only; remove the per-window captures
        in FeedWindows (one long prefill must not flush the store). The
        system-prompt token length comes from the chat format (e.g. render the
        system message alone and confirm it is a token prefix of the full
        prompt); under a windowed prefill the capture lands on the last window
        end at or below that boundary. Raw mode (no system role): prompt end and
        reply end only. Revisit the --cache-checkpoints default (16/8) once the
        store no longer holds per-window captures.
  - [x] T4a. Session snapshots copy only the live KV rows (per KV head, int8
        codes+scales verbatim), keep an OpenCL-resident KV cache resident on
        capture and mark it stale on restore, validate every layer before
        writing (RestoreSnapshot raises), and take a capacity hint so the twin
        SwitchTo allocates nothing. Open: GatedDeltaNet / DepthwiseConv1D
        recurrent state still leaves OpenCL memory on capture (one extra
        upload per capture).
  - [x] T3 fix (with T4b): capture also at the end of the LAST USER message
        (conversation rendered without the generation prompt, accepted only
        as a token prefix). cfQwen3_5 re-renders earlier assistant turns
        without <think>, so prompts diverge one token before the previous
        end-of-prompt; without this point qwen3_5 resumed only at the system
        checkpoint. Max 4 captures per turn.
  - [x] T4 (T4b). Conversation slots (--kv-slots N, default 0): a slot holds
        the reply-end session snapshot (host RAM, truncatable K/V) plus its
        resume points (system, last-user, reply end) and, on recurrent nets,
        the recurrent state at the last-user point (OpenCL memory under
        --gpu). Saved when a request leaves the live conversation; matched in
        the same one pass; same single eviction rule. Open: the twin-in-own-
        OpenCL-context restore branch and a forced save failure are untested;
        not yet run on a real model. Original plan text:
        Conversation slots (--kv-slots N, small, e.g. 2-4): a full saved
        session state (attention K/V rows + recurrent state, via
        TNNetDecoderSessionSnapshot / SnapshotInto) per slot, with PrefixHash,
        guard tokens and LastUsedTurn. When a request diverges from the live
        conversation and a slot matches deeper than the live cache, save the
        live conversation into a slot (DeleteTheLongestUnusedCheckpoint-style
        LRU when full) and restore the matching slot; prefill only the tail.
        Each slot costs up to one full KV cache (the live cache size at that
        position), so slots live in host RAM and follow --kv-int8. Check first:
        whether SnapshotInto copies only the live rows or the whole
        MaxCacheLen buffers, and how it handles a KV cache resident in OpenCL
        memory (FusedSDPA resident KV); fix to copy live rows only. Covers
        hybrid AND pure-attention nets (Llama/Qwen2.5 have no recurrent
        checkpoints, only KV).
  - [x] T5. Flags, stats, docs: --kv-slots in ParseArgs + help text; --stats
        reports which checkpoint/slot matched, reused tokens, and slot memory;
        update examples/ChatTerminal/ChatServer.md "Concurrency and cache
        reuse", the ChatTerminal README and the neuralchatengine.pas header
        ("The engine is single-session").
  - [x] T6. (covered in T4b: TestChatKVSlotSwitch [+PrefillWindow, +OpenCL],
        TestChatThinkingResume, TestChatKVSlotSingleSlot,
        TestChatKVSlotSystemPointNoOverwrite, TestChatKVSlotRegenerateKeepsPoints)
        End-to-end tests on pico nets (pure attention, hybrid, pure
        recurrent): requests A, B, A, B - the second A and B resume deep, and
        every reply equals the full re-prefill reply.
  - [ ] T7. User acceptance on the GPU box: TTFT of two alternating real
        conversations on a real hybrid (qwen3_5) before and after.
  - [ ] Follow-up (not in this task): shared paged KV blocks, so conversations
        share the blocks of their common prefix instead of one full KV copy per
        slot (vLLM-style); a large KV-cache rewrite.
- [ ] Gradient checkpointing for training deeper nets in less memory
- [ ] GGUF import beyond Llama — open follow-ups (core `BuildFromGGUF`/`BuildFromGGUFEx`
      arch dispatch with llama/qwen2/gemma2 LANDED & verified):
  - [ ] cross-tool llama.cpp gemma2 GGUF parity (DEFERRED: needs the llama.cpp
        shared lib / CLI — network + native build — unavailable here; the Llama
        drift test does not shell out either).
  - [ ] starcoder2 GGUF (out of scope for v1): needs LayerNorm-not-RMSNorm + full
        per-projection biases + non-gated GELU on the GGUF import path.
  - [ ] Extend dispatch to more Llama-backbone GGUF archs (phi3, qwen3, gemma3,
        gpt2/neox, MoE families) once each has an offline parity oracle.
- [ ] ONNX import
- [ ] Gemma 4 import
- [x] Qwen 3.5 import
      DONE (commit bfe209ed): BuildQwen35FromSafeTensors[Ex] /
      BuildQwen35MoeFromSafeTensors[Ex] import model_type qwen3_5 / qwen3_5_moe
      (the Qwen3.5/3.6 hybrid GatedDeltaNet + MoE trunk). neuralchat.pas renders
      the family's chat template as cfQwen3_5, and Qwen3.8 as cfQwen3_8 with the
      reasoning-effort spans; Qwen3.6-27B and Qwen3.8-27B are user-tested in
      ChatTerminal.
- [ ] MiniCPM importer follow-ups (`BuildMiniCPMFromSafeTensors[Ex]` LANDED,
      model_type `minicpm`; OpenBMB μP-style scale_emb / per-sublayer
      scale_depth / dim_model_base logits folds on the core Llama path +
      SentencePiece `.model` tokenizer; pico parity TestMiniCPM{Config,Logit}Parity
      < 1e-4 vs float64 HF):
  - [ ] MiniCPM real-checkpoint slicer follow-up (slice_llama.py reuse) to
        parity-check a sliced openbmb/MiniCPM-1.2B/2B against the random pico
        fixture (network/RAM-gated; the landed parity is the hand-built pico
        oracle only).
  - [ ] MiniCPM4 / MiniCPM-MoE importers (separate model families; MiniCPM4
        adds its own long-context/sparse-attention pieces, MiniCPM-MoE adds the
        Mixtral-style MoE FFN — neither is the dense-1.2B/2B path landed here).
- [ ] InternLM2 / InternLM2.5 importer follow-up (`BuildInternLM2FromSafeTensors[Ex]`,
      model_type "internlm2"; internlm/internlm2_5-1_8b/7b/20b) — LANDED.
      A plain Llama-backbone family (RMSNorm + RoPE + SwiGLU, GQA) whose ONLY
      genuinely new piece is the checkpoint LAYOUT, resolved by an
      InternLM2->HF translating reader (`TNNetInternLM2Reader`, a
      `TNNetSafeTensorsReader` subclass) that rides the existing core Llama
      builder with NO new layer/config flags. It (a) unpacks the fused
      `attention.wqkv` (reshaped `[num_kv_heads, q_per_kv+2, head_dim, hidden]`,
      per-group q slices then single K then single V) into standard separate
      W_q/W_k/W_v with the group-interleaved row reorder (all Q heads
      group-major, then all K, then all V — HF rotate_half order, so the core
      builder's own rotate_half→interleaved q/k de-permute for RoPE applies
      unchanged), and (b) renames the InternLM2 tensors (tok_embeddings /
      wo / feed_forward.w1=gate|w2=down|w3=up / attention_norm / ffn_norm /
      output) to the HF Llama names the core builder reads. Tokenizer is a
      SentencePiece `.model` (already supported); `bias=true` rejected.
      Verified by a HAND-BUILT pico fixture (tools/internlm2_tiny_fixture.py —
      InternLM2 is NOT in transformers, needs trust_remote_code offline, so the
      reference is a self-contained numpy float64 forward) asserting next-token
      logits < 1e-4: tests `TestInternLM2ConfigParity` / `TestInternLM2LogitParity`.
  - [ ] real-checkpoint slicer not exercised: the translating reader's wqkv
        unpack rematerializes each q/k/v slice in RAM per tensor (fine at pico
        and 1.8B/7B widths); a streaming/int8 path for the 20B checkpoint and a
        live trust_remote_code parity run against a downloaded internlm2_5-* are
        left open.
- [ ] IBM Granite 3.x importer follow-ups (`BuildGraniteFromSafeTensors[Ex]`
      LANDED, model_type `granite`/`granitemoe`; four multipliers folded at
      load on the Llama path — embedding/residual/attention multipliers + a
      DIVIDING logits_scaling into the LM head, wired through the existing
      QScale/W_q slot not hardcoded; granitemoe reuses the Mixtral MoE WIRING
      with a dedicated LoadGraniteMoEExperts loader for its fused 3-D slabs;
      pico parity TestGranite{Config,Logit,MoeLogit}Parity < 1e-4 vs float64
      HF transformers):
  - [ ] real-checkpoint slicer (make_pico_*_fixture.py reuse) to parity-check
        a sliced ibm-granite/granite-3.1-* against the random pico fixture.
- [ ] OLMoE real-checkpoint slicer follow-up (BuildOlmoeFromSafeTensors[Ex],
      model_type `olmoe`, LANDED: OLMo backbone with QK-norm full-width +
      original-OLMo pre-norm placement × Qwen3-MoE/Mixtral top-k MoE wiring,
      uniformly all-MoE; pico parity TestOlmoeLogitParity < 1e-4 vs float64 HF
      OlmoeForCausalLM): the pico fixture is random-init only — add a real
      allenai/OLMoE-1B-0924 slicer (slice_llama.py reuse) to parity-check a
      sliced real checkpoint against the random pico fixture.
- [ ] BitNet b1.58 importer follow-ups (`BuildBitNetFromSafeTensors[Ex]` LANDED,
      model_type "bitnet" / the released `microsoft/bitnet-b1.58-2B-4T`; the
      ternary-weight LLM mapped onto the Llama backbone with the SubLN
      norm-before-quantized-linear RMSNorms (attn_sub_norm before o_proj,
      ffn_sub_norm before down_proj via Config.BitNetSubLN, reusing
      TNNetTokenRMSNorm), the relu2 gated FFN via the new TNNetReGLUSquared
      activation (up*ReLU(gate)^2) on the separate-projection loader, and
      rope_theta from the transformers-5.x rope_parameters dict; the FP "shadow"
      checkpoint loads bit-for-bit since the absmean ternarize is a no-op on
      already-ternary weights; pico parity TestBitNet{Config,Logit}Parity < 1e-4
      vs float64 HF BitNetForCausalLM):
  - [ ] native I2_S packed-ternary de-quant-at-load (2-bit unpack: 4 ternary
        values per byte + a separate weight_scale tensor). The shadow-weight
        (FP effective-weights) path is covered; the GGUF/bitnet.cpp-style I2_S
        packed format is NOT yet read (the HF transformers checkpoint ships
        the shadow form, so the parity fixture exercises only that path).
  - [ ] real-checkpoint slicer (slice_llama.py reuse) to parity-check a sliced
        microsoft/bitnet-b1.58-2B-4T against the random pico fixture — SKIPPED
        (network/RAM-gated: requires downloading the 2B checkpoint).
- [ ] M2M100/NLLB translate demo + real-vocab check (follow-up to the landed
      BuildM2M100FromSafeTensors, commit cb21550): an examples/NLLBTranslate
      seq2seq demo that loads a real (small) NLLB/M2M100 checkpoint and
      round-trips a translation on CPU via DecodeSeq2SeqGreedy/BeamSearch, plus
      verifying the importer against a downloaded sentencepiece.bpe.model vocab
      (the parity fixture today uses raw token ids only).
- [ ] GPT-OSS importer follow-ups (BuildGptOssFromSafeTensors[Ex] LANDED,
      model_type "gpt_oss", with TNNetGptOssSinkAttention per-head scalar sink +
      alternating sliding/full window + YaRN truncate flag + top-k MoE +
      TNNetGptOssGatedSwiGLU clamped experts + MXFP4 dequant-at-load; pico parity
      TestGptOssLogitParity / TestGptOssMXFP4LogitParity <1e-4): (a) MXFP4 -> int8
      weight-only STORAGE path — pQuantizeInt8 currently re-quantizes the
      dequantized FP32 rather than a direct 4->8 transcode; (b) no real-weight
      20B/120B smoke run yet (RAM-gated; pico parity only).
- [ ] Jamba importer follow-ups (BuildJambaFromSafeTensors[Ex] LANDED, model_type
      "jamba", first hybrid Mamba+attention+MoE import; TNNetSelectiveSSM
      CreateJambaInner inner-norm mode; pico parity TestJambaLogitParity <1e-4):
      (a) mamba_proj_bias=true (in/out_proj biases) is REJECTED — wire the bias
      path if a checkpoint needs it; (b) training through the Jamba inner-norm SSM
      is forward-only — add backward through the dt/B/C RMSNorms for fine-tuning;
      (c) no Jamba-specific tokenizer demo (raw ids work); (d) no real-checkpoint
      slicer fixture (parity uses a random pico model).
- [ ] Nemotron-H-MoE importer follow-ups ('E' MoE block LANDED on the landed
      BuildNemotronHFromSafeTensors: DeepSeek-V3-style sparse non-gated relu2
      experts + always-on shared expert via the landed
      TNNetBiasBalancedTopKGate/TNNetTopKGate + TNNetSigmoid + TNNetMulByConstant
      (NO new layer); TestNemotronHMoELogitParity, schedule "M*E-", < 1e-4 vs a
      float64 HF oracle):
  - [ ] Nemotron-H-MoE grouped-routing follow-up: the importer REJECTS
        n_group>1 and moe_latent_size with a clear message. Wire the
        group-limited / latent-MoE routing path if a real Nemotron-H-MoE
        checkpoint needs it (DeepSeek-V2-style grouped topk gate is landed —
        reuse it).
- [ ] Zamba2 importer (model_type "zamba2") — the sibling Mamba-2 hybrid that
      additionally shares a small set of transformer blocks across depth with a
      per-invocation LoRA adapter; needs a shared-block + LoRA wiring helper
      (larger than Nemotron-H, distinct from the per-layer-independent schedule).
- [ ] SigLIP follow-up: NaFlex / variable-resolution siglip2 (the landed
      BuildSigLIPFromSafeTensors scopes to the FIXED-resolution siglip/siglip2
      base configs: square image_size, learned position table over a fixed
      num_patches grid). NaFlex siglip2 supports per-image patch counts
      (aspect-ratio-preserving resize to a variable token grid) with attention
      pooling over the variable grid and a padding/attention mask — needs a
      variable-SeqLen vision input, an interpolated/resampled position table,
      and key-padding-masked MAP pooling. Not yet wired.
- [x] examples/TCNClassifier — a Temporal Convolutional Network (Bai, Kolter &
      Koltun 2018, arXiv:1803.01271) for MULTIVARIATE time-series
      classification, built only from existing layers (no new leaf class). A
      user asked whether CAI Neural supports a standard TCN; it does, but no
      example shows the layer structure (`examples/TimeSeriesForecast` is a
      2-layer univariate causal-conv stack with no residuals). Scope:
      - Input `(WindowLen, 1, NumFeatures)`: time on X, features on Depth;
        fixed rolling windows cut from a synthetic multivariate series
        generated in-code (no downloads, fixed RandSeed).
      - Residual block: `TNNetCausalConv1D(C, K, 0, Dilation)` -> ReLU ->
        `TNNetSpatialDropout1D` -> same again -> `TNNetSum` with the skip path
        (a `TNNetPointwiseConvLinear(C)` 1x1 projection when the channel
        count changes) -> ReLU. Dropout optional (README must say so).
      - Dilations 1, 2, 4, 8, 16; print the receptive field
        (1 + 2*(K-1)*sum(dilations)) and choose WindowLen >= it.
      - Head: `TNNetCrop(WindowLen-1, 0, 1, 1)` (last time step) ->
        `TNNetFullConnectLinear(NumClasses)` -> `TNNetSoftMax`; trained with
        `TNeuralFit`. Report test accuracy vs chance.
      - Synthetic task should need the long receptive field (e.g. the class
        depends on an event far back in the window), so a non-dilated stack
        scores worse; print both.
      - Pure CPU, under 3 GB (`ulimit -v 3145728`), a few minutes at most;
        .lpr (with `cthreads` first in uses) + .lpi + README + an
        examples/README.md entry.
      Related follow-ups (separate, each needs user OK): a
      `TNNet.AddTCNBlock` builder; a dilated INPUT-gradient check for
      `TNNetCausalConv1D` (only the dilated weight gradient is checked
      today); a faster `TNNetCausalConv1D` forward/backward for narrow
      InputDepth (one AVX call per tap is overhead-bound at e.g. 5 features).
      DONE: builder `TNNet.AddTCNBlock` (68cf0d77, tests tightened 777a6d08)
      and examples/TCNClassifier using it. Measured (window 64, 4 channels,
      KernelSize 2, receptive field 63): dilated TCN 97.11% test accuracy vs
      non-dilated baseline 39.46% (receptive field 11, same 4864 weights);
      chance 25%. Peak RSS 36 MB. Wall clock varies widely, cause not yet
      diagnosed: 8:08 and 33:34 measured. The builder and the dilated input-gradient
      check are done; the narrow-InputDepth speedup is still open.

### Computer vision & generative models

- [ ] FastSAM real-time segment-anything importer (`BuildFastSAMFromSafeTensors[Ex]`,
      ultralytics FastSAM-s / FastSAM-x). Architecturally distinct from the landed
      ViT-encoder SAM path: FastSAM reframes "segment everything" as a YOLOv8
      instance-segmentation backbone (CSP/C2f conv stages + a prototype-mask head)
      followed by a CLIP-text or point/box prompt-selection step over the emitted
      masks — no heavy image encoder, so it runs interactively on CPU. v1 scope:
      the YOLOv8-seg backbone + proto-mask head producing the full mask set on a
      fixed input, reusing the landed conv/SPPF building blocks; the CLIP-prompt
      mask ranking as a follow-up. Pico-fixture parity < 1e-4 on the proto-mask
      logits vs a float64 reference.

- [ ] LLaVA-NeXT (llava-v1.6, model_type `llava_next`) dynamic-resolution VLM
      importer follow-up to the landed `BuildLlavaFromSafeTensors`. The v1.6
      difference is the "anyres" image processing: the input is split into a grid
      of native-aspect tiles plus a global thumbnail, each tile is CLIP-encoded
      separately, and the per-tile patch grids are spatially unpadded and
      concatenated into one visual-token stream (a large accuracy gain on dense
      OCR/chart images over the v1.5 single-square crop). v1 scope: the anyres
      tile-split + per-tile encode + token-merge connector reusing the landed
      CLIP vision tower and Llama/Vicuna decoder; `examples/LlavaNextDescribe`
      caption. Pico-fixture parity < 1e-4 on the merged visual-token features.

- [ ] OpenCLIP / timm vision-tower real-checkpoint round-trip follow-up
      (`BuildOpenClipVisionTower` LANDED, commit 794a10ce, neuralpretrained.pas —
      `TNNetOpenClipReader` maps flat `visual.*` open_clip keys onto the HF
      `vision_model.*` graph `BuildClipVisionTower` reads; pico parity
      TestOpenClipVisionTowerParity < 1e-4 on the already-parity-verified CLIP
      vision graph): verify against the actual `open_clip` reference (laion
      ViT-L/14, ViT-g/14) — open_clip lib + LAION weights unavailable offline.

- [ ] Janus-Pro unified multimodal importer (`BuildJanusProFromSafeTensors[Ex]`,
      deepseek-ai/Janus-Pro-1B and Janus-Pro-7B). Architecturally distinct from
      every landed VLM (LLaVA / BLIP-2 / Qwen2-VL / PaliGemma / Florence-2) and
      from the diffusion image-generation stacks: Janus-Pro is a SINGLE
      autoregressive Llama-backbone transformer with TWO decoupled vision paths
      sharing one token stream — a SigLIP "understanding" encoder (reuse
      `BuildClipVisionTower` / SigLIP path) projected in for image→text, and a
      VQ "generation" path (`gen_vision_model` VQ tokenizer + `gen_embed` +
      `gen_head`) that emits DISCRETE image tokens decoded back to pixels.
      v1 scope: the understanding path (image-conditioned text, an
      `examples/JanusDescribe` caption) reusing the landed Llama decoder + VQ
      codebook layers. Image-GENERATION head (sample image tokens, VQ-decode to
      an RGB grid) as the noted follow-up. Pico-fixture parity < 1e-4 vs a
      float64 transformers oracle on the understanding logits.

- [ ] VMamba / Vision Mamba (Vim) state-space vision-backbone importer
      (`BuildVMambaFromSafeTensors[Ex]`, state-spaces / MzeroMiko VMamba and
      hustvl/Vim-* checkpoints). Ports the 2-D selective-scan vision backbone on
      top of the already-landed `TNNetSelectiveSSM` / `AddMambaBlock`: patch-embed
      stem → stacked SS2D blocks that run the selective scan over FOUR directional
      flattenings of the (H,W) grid (the cross-scan) and merge them, with the
      hierarchical down-sampling stages of a classification backbone. The reusable
      new piece is the 2-D cross-scan wrapper (four directional traversals of a
      depth-contiguous feature map feeding the existing 1-D SSM, then summed);
      everything downstream reuses the landed Mamba layer + classifier head.
      Pico-fixture top-1 parity vs a float64 oracle, plus reuse of the tracked
      ImageNet-val eval harness on a sliced real checkpoint.

- [ ] HunyuanVideo / Mochi-1 text-to-VIDEO DiT importer
      (`BuildHunyuanVideoFromSafeTensors[Ex]`, tencent/HunyuanVideo and the
      genmo/mochi-1 sibling). Distinct from the landed CogVideoX and the planned
      Wan / SVD / AnimateDiff video stacks: a dual-stream → single-stream MMDiT (separate
      image- and text-stream blocks that later fuse into joint blocks, like
      FLUX's MMDiT but over a spatio-temporal latent) with 3-D RoPE over the
      (T,H,W) token grid and a 3-D causal-VAE latent space. v1 scope: the
      transformer denoiser fed pre-computed text embeddings + a random latent,
      reusing the landed diffusion schedulers (`TNNetDiffusionScheduler` flow /
      DPM paths) for the sampling loop; the 3-D causal VAE decoder as a noted
      follow-up (a few latent frames → pixel frames). Pico-fixture parity < 1e-4
      vs a float64 transformers/diffusers oracle on one denoise step.

- [ ] BiRefNet / RMBG-2.0 background-removal (dichotomous high-resolution image
      segmentation) importer (`BuildBiRefNetFromSafeTensors[Ex]`,
      ZhengPeng7/BiRefNet and briaai/RMBG-2.0, which ships the same BiRefNet
      architecture). Distinct from the landed semantic/instance/panoptic
      segmentation paths (SegFormer / Mask2Former / Mask-R-CNN): BiRefNet predicts
      a SINGLE high-resolution foreground alpha matte via a bilateral-reference
      decoder — a Swin-L backbone (reuse `BuildSwinFromSafeTensors`) feeding a
      localization module + a reconstruction decoder whose blocks each take a
      gradient/inward-reference and an outward-reference feature, upsampled to a
      full-resolution sigmoid mask. The reusable new pieces are the
      bilateral-reference decoder block and the multi-scale supervision head;
      reuse the landed bilinear upsample (`TNNetBilinearUpsample`) and Swin tower.
      Pico-fixture parity < 1e-4 vs a float64 transformers oracle, plus an
      `examples/BackgroundRemoval` cutting a subject out of a tiny CPU image
      (sigmoid mask -> RGBA composite).

- [ ] SDXL residual: real stabilityai/stable-diffusion-xl-base-1.0 checkpoint
      parity still deferred (only the pico oracle is verified). The 2048 cross
      width is already just CrossAttentionDim, and the highest-res "no attn"
      block layout is handled by down_block_types=DownBlock2D (DownHasAttn=false
      skips the Transformer2DModel) — transformer_layers_per_block list (e.g.
      [1,2,10]) already supported; no depth=0 special case needed. The dual
      text-encoder concatenation lives in the (still-deferred) SDXL pipeline
      driver, not the UNet importer.
- [ ] Hunyuan-DiT bilingual text-to-image importer
      (`BuildHunyuanDiTFromSafeTensors[Ex]`, Tencent `Hunyuan-DiT`). A distinct
      DiT from the landed PixArt/MMDiT/Sana stacks: a cross-attention DiT
      conditioned on BOTH a bilingual CLIP and an mT5 text encoder (two separate
      cross-attn streams concatenated), 2-D rotary position embedding on the
      image tokens, and a "skip-connection" (U-Net-style long skips between the
      first and second half of the transformer blocks, fused by a Linear). The
      reusable new pieces are the dual-encoder cross-attention wiring and the
      DiT-level long-skip Linear merge; reuse the landed VAE decoder + scheduler
      and scope a small parity fixture.
- [ ] LTX-Video text-to-video DiT importer (`BuildLTXVideoFromSafeTensors[Ex]`,
      Lightricks `LTX-Video`). Distinct from the landed CogVideoX and the planned
      Wan / SVD video importers: a single-stream MMDiT-style video DiT operating on tokens
      from a high-compression causal video VAE (patchified space-time latents with
      3-D rotary position embedding), conditioned on a T5 text encoder (landed),
      with rectified-flow sampling. High-value because it reuses the landed
      T5 tower + rectified-flow path and only the causal video-VAE patch
      tokenizer + 3-D RoPE wiring are new; scope a pico latent-space parity
      fixture (the full VAE decode can be a follow-up like the other video
      importers).
- [ ] Pixtral vision-language importer (`BuildPixtralFromSafeTensors[Ex]`,
      mistralai `Pixtral-12B`, model_type "pixtral"). Distinct from the landed
      LLaVA / Qwen2-VL / PaliGemma VLMs: a native-resolution ViT vision encoder
      with 2-D RoPE and a block-diagonal attention mask over a variable number of
      image patches, a GELU MLP connector, and a Mistral decoder (importer already
      landed) where each image is spliced in as a flat run of patch tokens framed
      by `[IMG]` / `[IMG_BREAK]` / `[IMG_END]` markers. The reusable new piece is
      the 2-D-RoPE variable-resolution vision tower + the patch-token splicing
      position builder; verify the tower against a transformers reference and the
      decoder against the landed Mistral parity fixture.
- [ ] Sana text-to-image importer (`BuildSanaFromSafeTensors[Ex]`, NVIDIA /
      Efficient-Large-Model `Sana_*`). A genuinely different image-generation
      stack from the landed PixArt/MMDiT path: a LINEAR-attention DiT denoiser
      (Sana's "linear DiT" replaces softmax SDPA with the landed
      `TNNetLinearAttention` plus a Mix-FFN), a Gemma decoder-only text encoder
      (reuse the landed Gemma importer), and a deep-compression autoencoder
      (DC-AE, f32 spatial compression) in place of the SD VAE. High-value because
      it exercises the linear-attention path end-to-end in a real generative model
      and reuses three already-landed subsystems; scope a small `Sana-0.6B` /
      `Sana-Sprint` parity fixture like the other importers.
- [ ] InternVL vision-language importer (`BuildInternVLFromSafeTensors[Ex]`,
      model_type "internvl_chat", e.g. OpenGVLab/InternVL2-1B). Distinct from the
      landed LLaVA / PaliGemma / Qwen2-VL VLMs: an InternViT vision tower + a
      PIXEL-UNSHUFFLE downsampler (`TNNetSpaceToDepth`) feeding an MLP projector
      into a Qwen2 / InternLM2 decoder (both importers already landed). The reusable
      new piece is the pixel-unshuffle token reducer + dynamic-tiling image
      preprocessor; verify on the 1B checkpoint against a transformers reference.
- [ ] DINOv3 ViT backbone importer follow-ups (`BuildDINOv3FromSafeTensors[Ex/WithConfig]`
      LANDED, HF model_type "dinov3_vit", facebook/dinov3-*; pre-LN ViT + LayerScale +
      2-D AXIAL RoPE on patch tokens (TNNetVisionRoPE2D + AddMultiHeadVisionRoPE2DAttention),
      register tokens prepended after CLS, BERT-unfused q/k/v/o_proj with UNbiased key;
      parity < 1e-4 vs a float64 HF DINOv3ViTModel oracle, scalar + AVX2 green):
  - [ ] use_gated_mlp (SwiGLU giant) variant rejected (only plain
        up_proj/down_proj+gelu wired).
  - [ ] non-square / non-native image_size (dynamic RoPE grid) not yet plumbed —
        TNNetVisionRoPE2D is fixed to the build-time grid. Still unblocks
        DINOv3-backed dense-prediction once the "register-token DINOv2 backbones
        rejected by BuildDPT" follow-up lands.

- [ ] Kandinsky 2.2 unCLIP text-to-image importer
      (`BuildKandinskyPriorFromSafeTensors[Ex]` + `BuildKandinskyDecoderFromSafeTensors[Ex]`,
      kandinsky-community `kandinsky-2-2-prior` + `kandinsky-2-2-decoder`). Distinct
      from every landed diffusion stack (SD UNet / PixArt / DiT / MMDiT block), which
      denoise in a VAE LATENT conditioned on a TEXT SEQUENCE: Kandinsky is a two-stage
      unCLIP pipeline. Stage 1 is a DIFFUSION PRIOR — a small transformer
      (`PriorTransformer`) that denoises a single CLIP IMAGE embedding vector
      conditioned on the CLIP TEXT embedding + pooled text (the genuinely-new piece:
      the model's "tokens" are [text-emb, pooled-text, time-emb, noised-image-emb,
      learnable-final] and the target is the clean image embedding, NOT a spatial
      latent). Stage 2 is a UNet decoder conditioned on that predicted image embedding
      (via an `image_embeds` add-embedding, no cross-attn text stream) denoising in a
      MoVQ (VQ-VAE) latent. Reuse: the landed CLIP text/image towers (image-emb
      conditioning), `BuildSDUNetFromSafeTensors` for the decoder UNet trunk (add the
      image-embeds projection path behind a config flag), `BuildVqModelFromSafeTensors`
      / VQ decode for MoVQ, and an existing DDIM/UniPC scheduler for both diffusion
      loops. The reusable new code is the `PriorTransformer` (a sequence-of-embeddings
      denoiser) + the image-embedding add-conditioning on the decoder UNet. Scope a
      pico parity fixture for each stage against a diffusers float64 oracle and an
      `examples/KandinskyGenerate` end-to-end ("a red fox in snow" -> tiny CPU image).

- [ ] Llama 3.2 Vision (mllama) cross-attention VLM importer
      (BuildMllamaFromSafeTensors[Ex], model_type "mllama", e.g.
      meta-llama/Llama-3.2-11B-Vision-Instruct). DISTINCT from the landed
      Qwen2-VL / Florence-2 / LLaVA paths, which fuse vision by PREPENDING visual
      tokens into the text sequence: mllama interleaves dedicated CROSS-ATTENTION
      decoder layers (cross_attn_layers) that let the text stream attend to a frozen
      ViT vision tower's patch features, gated by a learnable tanh gate
      (cross_attn_attn_gate / cross_attn_mlp_gate, zero-init so the base LM is
      unchanged at start). Genuinely-new code: the vision tower (a ViT with a
      pre/post tile-position embedding + global+local transformer stack over image
      TILES) feeding the cross-attention layers, the tanh-gated cross-attn residual
      injection (reuse TNNetCrossAttention + a learnable scalar gate), and the
      per-layer self-attn-vs-cross-attn schedule; the text self-attention decoder is
      the landed Llama backbone. v1 may take the vision tower's tile features as a
      precomputed TNNetInput (the Qwen2-VL "merged tokens as input v1" stance), then
      drop the shortcut. Pico parity < 1e-4 vs a float64 HF MllamaForConditionalGeneration
      oracle + an examples/LlamaVisionDescribe captioning a tiny CPU image.

- [ ] Grounding DINO open-vocabulary detection importer
      (`BuildGroundingDINOFromSafeTensors[Ex]`, model_type "grounding-dino",
      e.g. IDEA-Research/grounding-dino-tiny). Distinct from the landed OWL-ViT
      path: GDINO is a text-conditioned DETR with a Swin backbone, a BERT text
      encoder, a cross-modality feature enhancer (deformable image self-attn +
      bi-directional text<->image cross-attn) and a cross-modality decoder whose
      queries attend to BOTH towers. Reuse BuildSwinFromSafeTensors (image),
      BuildBertFromSafeTensors (text) and TNNetCrossAttention for the fusion;
      output = boxes + per-query token-similarity logits scored against the
      prompt's token spans. Pico-fixture parity + an examples/GroundingDINODetect
      ("detect: a cat. a remote." -> boxes) demo.

- [ ] SAM 2 promptable image + VIDEO segmentation importer
      (`BuildSAM2FromSafeTensors[Ex]`, e.g. facebook/sam2-hiera-tiny). The landed
      SAM v1 path covers the single-image ViT encoder + mask decoder; SAM 2 adds
      the pieces that make it a distinct model: a Hiera hierarchical image
      encoder, a memory-attention block that conditions the current frame on a
      memory bank of past-frame embeddings + object pointers, and a memory
      encoder that writes mask predictions back into the bank. v1 scope = single
      image (skip the bank); follow-up = the streaming video loop. Reuse the SAM
      prompt encoder + two-source TNNetCrossAttention. examples/SAM2Segment
      (point/box prompt on one frame, then carry the mask across a short clip).

- [ ] Depth Anything V2 importer follow-ups (BuildDepthAnythingV2FromSafeTensors[Ex]
      LANDED, commit 51036f9, depth_anything model_type, e.g.
      depth-anything/Depth-Anything-V2-Small-hf — DINOv2 ViT backbone -> DPT
      reassemble+fusion neck + 3-conv depth head, TDPTConfig.OutIndices block
      selection, pico parity TestDepthAnythingV2Parity max|diff|<1e-4 vs
      transformers float64 DepthAnythingForDepthEstimation; examples/DepthAnythingV2):
  - [ ] real-checkpoint parity not run (no weights downloaded).
  - [ ] metric-depth Depth-Anything-V2 (indoor/outdoor, depth_estimation_type
        "metric") untested on a real ckpt.
  - [ ] register-token / SwiGLU DINOv2 backbones still rejected by BuildDPT.

- [ ] ResNet importer follow-ups (BuildResNetFromSafeTensors[Ex] LANDED, commit
      317a19c: torchvision resnet18/50 state_dict, conv-BN fold at load, config
      reader/ToString, resnet18 parity 1.0e-6 vs a numpy float64 oracle —
      torchvision not installed; ConvNeXt LayerScale+GRN+depthwise-7x7 is the
      modern-CNN stretch goal on the same path):
  - [ ] real downloaded torchvision resnet18/50 top-1 over ImageNet val
        (torchvision not installed here); ConvNeXt modern-CNN stretch goal. (The
        torchvision .pth load path + PyTorch stem maxpool reconciliation have
        landed.)
- [ ] Stable Diffusion VAE decoder importer follow-ups (BuildVaeDecoderFromSafeTensors[Ex]
      LANDED, commit a7870a1: diffusers AutoencoderKL decoder — post_quant_conv ->
      mid block single self-attention over HxW -> up blocks of ResNet groups +
      nearest upsample -> conv_out, latent scaled by 1/0.18215; reuse-only (nearest
      upsample = TNNetDeMaxPool(2) FSpacing=0; mid attention flattens (H,W,C)->
      (H*W,1,C) per-token SDPA; added GroupNormEpsilon to TNNetGroupNorm to pin
      diffusers eps 1e-6); TestVaeDecoderParity <1e-4 vs a numpy float64 oracle):
  - [ ] real-checkpoint (stabilityai/sd-vae-ft-mse) parity once diffusers is
        installable; SDXL VAE uses different group counts / multi-head attention.
  - [ ] SD UNet importer LANDED (BuildSDUNetFromSafeTensors[Ex] + SDUNetDenoise,
        UNet2DConditionModel v1; TestSDUNetParity 2.75e-6 < 1e-4). Open follow-ups:
    - [ ] real-checkpoint parity (runwayml/stable-diffusion-v1-5 / SDXL unet)
          once diffusers is installable. (Per-block attention-head arrays —
          SDXL-style attention_head_dim list via Config.HeadsPerBlock — have
          landed, parity < 1e-4.)
    - [ ] the end-to-end LatentTextToImage capstone (CLIP text -> this UNet ->
          scheduler loop -> VAE decoder), incl. the SDXL dual-text-encoder pooled
          embedding + real-checkpoint parity.
    - [ ] Real Stable Diffusion text-to-image from a diffusers folder (audit
          2026-10-02). No example loads a real SD checkpoint today: every SD
          example defaults to the tests/fixtures pico nets, and all of them fill
          the UNet text input with random/synthetic states instead of CLIP over a
          prompt (DiffusionInpainting.lpr:260, FillText in ControlNetCanny and
          T2IAdapterSketch). ControlNetCanny/T2IAdapterSketch run ONE denoise step
          and write no image. Missing pieces:
      - [ ] `TStableDiffusionPipeline` in neuralpretrained.pas with the same
            public shape as `TQwenImage21Pipeline` (Create(ModelFolder) over
            model_index.json + text_encoder/ tokenizer/ unet/ vae/ scheduler/,
            TokenizePrompt, EncodePrompt, Denoise, DecodeLatents,
            GenerateFromEmbeds, OnPhase/OnStep, load-use-free one component at a
            time). Scheduler from scheduler/scheduler_config.json onto
            TNNetDiffusionScheduler (dsScaledLinear; dpEps for SD 1.x, dpV for
            SD 2.x 768). Latent (H/8,W/8,4) scaled by 1/0.18215 (SDXL 0.13025).
      - [ ] CFG inside the SD loop: a cond + an empty-prompt uncond UNet pass per
            step (twice the per-step cost).
      - [ ] CLIP text encoder for SD: BuildClipFromSafeTensors loads a full CLIP
            (text + vision); check / add a text-only path for the
            CLIPTextModel-only text_encoder/ folder (unverified). SD 1.x feeds the
            last hidden state (77 tokens, padded); SD 2.x the penultimate layer;
            SDXL two encoders (CLIP-L + OpenCLIP-bigG) concatenated + the pooled
            vector into SDUNetDenoiseSDXL.
      - [ ] CLIP tokenizer from vocab.json + merges.txt: the SD 1.5 tokenizer/
            folder is believed to ship no tokenizer.json, which is what
            neuralhftokenizer.pas reads (unverified; else convert once).
      - [ ] F16 weights: the safetensors reader decodes F16, but the SD UNet / VAE
            loaders are unverified on the F16 (and *.fp16.safetensors variant)
            files most SD repos ship.
      - [ ] Resolution: the UNet TNNetInput is sized from Config.LatentGrid at
            build, so each image size needs a rebuild (or a resizable input).
      - [ ] Memory/speed: FP32 SD 1.5 UNet ~3.4 GB (over the 3 GB cap on the dev
            box, so the first real run is on the user's box); int8/int4 UNet
            weights (the int8 conv path exists, untried on this UNet), UNet
            OpenCL arming, per-prompt caching of the text cross-attention K/V.
      - [ ] CLI: either teach examples/QwenImage to pick the pipeline from
            model_index.json `_class_name` (QwenImage21Pipeline vs
            StableDiffusionPipeline) so the argument parsing / reporter / OpenCL
            selection are shared, or a separate examples/StableDiffusion over the
            same pipeline class. Then wire DiffusionInpainting / ControlNetCanny /
            T2IAdapterSketch to the real CLIP text states and a full sampling loop.
      - [ ] Per-component parity against diffusers on real weights (the ~/x venv
            oracle recipe from the Qwen-Image project), SD 1.5 first, then SD 2.x,
            then SDXL.
- [ ] CogVLM / CogVLM2 vision-language importer (`BuildCogVLMFromSafeTensors[Ex]`,
      model_type "cogvlm"/"cogvlm2"). Architecturally distinct from the
      shared-trunk VLMs already imported (LLaVA / Florence2 / Blip2 / Pixtral):
      CogVLM routes IMAGE tokens and TEXT tokens through SEPARATE per-block
      weights — a dedicated "visual expert" with its own QKV and FFN matrices
      running in parallel with the language QKV/FFN, selected per token by
      modality. The importer needs a token-type-routed block builder (text
      experts vs vision experts sharing the attention score space), the EVA-CLIP
      ViT image encoder + MLP adapter, and a pico safetensors parity fixture
      (`make_pico_cogvlm_fixture.py`, reusing the slice-real-checkpoint pattern)
      checked < 1e-4 vs a transformers float64 oracle. Add a CV captioning /
      VQA smoke under examples/ once a small checkpoint is obtainable offline.
- [ ] Mask2Former universal-segmentation importer LANDED
      (BuildMask2FormerFromSafeTensors[Ex], model_type "mask2former";
      masked-attention decoder + per-query mask/class heads, RunMask2FormerSemantic +
      DecodeMask2FormerSemantic, TestMask2FormerParity ~1.2e-7 < 1e-4,
      examples/UniversalSegmentation). Deferred follow-ups:
      (a) wire the full Swin backbone + FPN pixel decoder (mask_features + 3 multi-scale
          memory levels with level_embed + sine pos) into ONE forward — v1 takes the
          pixel-decoder outputs as PRECOMPUTED TNNetInput feature maps (mirroring Mask
          R-CNN v1's FPN-input stance); the masked-attention decoder + heads are complete;
      (b) instance + panoptic post-processing (v1 is SEMANTIC argmax only);
      (c) multi-head cross-attention beyond the pico 1-head fixture is wired generically
          (Heads loop) but only parity-verified at Heads=1;
      (d) a real-checkpoint smoke once a swin-tiny-semantic checkpoint is obtainable
          offline (the importer is config-driven and ready).
- [ ] NAFNet image-restoration importer follow-ups (BuildNAFNetFromSafeTensors[Ex]
      + TNAFNetConfig LANDED — TNNetSimpleGate gate layer + Simplified Channel
      Attention + U-Net of NAFBlocks with LayerNorm2d / depthwise 3x3 / PixelShuffle;
      pico parity TestNAFNetParity < 1e-4 vs a numpy float64 oracle;
      examples/ImageRestoration end-to-end CPU denoise):
  - [ ] Real official NAFNet denoise/deblur checkpoint parity (offline env — the
        weights are large / not redistributable; importer uses the canonical
        NAFNet state_dict key scheme so a real checkpoint should drop in).
  - [ ] PNG input/output (the example uses deterministic synthetic images + PPM;
        no PNG decoder in-tree).
  - [ ] NAFBlock dropout + the wider official width/block configs are wired by
        config but only the small pico shape is parity-pinned.
- [ ] SwinIR transformer image-restoration importer follow-ups
      (BuildSwinIRFromSafeTensors[Ex] + TSwinIRConfig LANDED — RSTB of Swin
      window/shifted-window attention + 3x3 conv residual + pixel-shuffle upsample
      tail, reusing the landed Swin attention blocks, NO new leaf layers; pico parity
      TestSwinIRParity / TestSwinIRConfigFromJSONFile < 1e-4 vs a float64 numpy
      oracle; examples/SwinIRRestore 2x SR smoke):
  - [ ] Pinned the classical-SR (pixel-shuffle upscale) variant; the
        same-resolution denoise tail (upscale=1, single conv reconstruction) is
        wired by config but only the SR shape is parity-pinned.
  - [ ] Real-checkpoint parity deferred (official SwinIR .safetensors are large /
        not obtainable offline); the importer accepts a real checkpoint path.
  - [ ] Lightweight-SR (no conv_after_body / single shared upsample) and the
        nearest+conv "real-world" SR upsampler variants not yet wired.
- [ ] RIFE video frame-interpolation importer follow-ups (BuildRIFEFromSafeTensors[Ex]
      + TRIFEConfig + TNNetBackwardWarp LANDED — IFNet intermediate-frame synthesis
      with accumulated flow residuals + sigmoid fusion-mask blend; pico parity
      TestRIFEParity < 1e-4 vs a float64 numpy oracle;
      examples/VideoFrameInterpolation one middle frame t=0.5, inference-only, on CPU):
  - [ ] Real Practical-RIFE checkpoint parity (offline-download deferred): the real
          IFNet is MULTI-SCALE (stride-2 IFBlocks downsample then bilinearly upsample
          the predicted flow) and RE-FEEDS the warped frames + previous flow/mask into
          each successive block; v1 refines from the SAME [frame0|frame1] input at full
          resolution. Wire the scale schedule + warped-frame re-feeding and verify
          against an exported hzwer/Practical-RIFE checkpoint.
  - [ ] Arbitrary-t interpolation (t != 0.5) and the recursive 2x->4x multi-frame
        schedule (RIFE's timestep encoding + recursion).
  - [ ] Privileged-distillation teacher (the training-time IFNet that sees the GT
        middle frame); inference-only path is landed.
- [ ] GFPGAN blind face-restoration importer (BuildGFPGANFromSafeTensors[Ex],
      TencentARC/GFPGAN) — leverages the LANDED StyleGAN2 generator
      (BuildStyleGAN2GeneratorFromSafeTensors) as a fixed facial prior: a U-Net
      degradation-removal encoder maps a low-quality face to the StyleGAN2 latent/
      noise inputs, and per-resolution encoder features are spatially modulated into
      the generator via CS-SFT (channel-split spatial feature transform) layers. The
      only genuinely new code is the small U-Net encoder + the CS-SFT scale/shift
      injection (an affine FiLM-like modulation — reuse the landed TNNetFiLM); the
      synthesis backbone is reuse. Distinct from NAFNet/SwinIR (generic restoration
      with no generative face prior). Pico parity vs a float64 oracle on the SFT
      injection for a fixed latent + examples/FaceRestoration on a tiny CPU image.
- [ ] LPIPS follow-ups — the metric LANDED (ComputeLPIPSDistance /
      LPIPSStageDistance / LPIPSUnitNormalize in neuralpretrained.pas, reusing the
      VGG importer's 5 relu taps; unit-normalize -> squared-diff -> per-stage lin
      head -> spatial-mean -> sum; parity test TestLPIPSDistanceParity vs a numpy
      float64 oracle, LPIPS(x,x)=0). The lin head is a LOADABLE per-stage weight
      vector; nil = the unweighted (1/C channel-mean) lpips lin_layers=False
      baseline (what the oracle pins). OPEN: (a) ship/import the OFFICIAL
      richzhang/PerceptualSimilarity lin{0..4}.model 1x1-conv weights + parity vs
      the reference calibrated LPIPS (needs the `lpips` weights, unobtainable
      offline here). (b) DONE: ComputePerceptualLossAndGradient (neuralpretrained.pas)
      exposes the Johnson-2016 deep-feature perceptual loss WITH the d(Loss)/d(pixels)
      gradient (per-stage feature L2, plain — calibrated unit-normalize is a follow-up),
      driving a multi-tap backward through the frozen VGG to the input layer; tests
      TestPerceptualLossGradientCheck (FD max rel err 7.1e-3) + TestPerceptualLossSelfZero.
      Added public TNNetLayer.GetDepartingBranchesCnt for the idempotent intermediate-tap
      IncDepartingBranchesCnt guard. OPEN: wire it into an SR example as an opt-in objective.
- [ ] ControlNet spatial-conditioning importer LANDED
      (BuildControlNetFromSafeTensors[Ex], lllyasviel/sd-controlnet-canny;
      hint stem + per-resolution zero-conv residual taps + ControlNetResiduals driver,
      TestControlNetParity 2.89e-6 < 1e-4). First conditioning-by-feature-injection model.
      DEFERRED follow-ups: (a) end-to-end base-UNet-plus-ControlNet generation
      LANDED — BuildSDUNet(..., pWithControl=true) splices a zero-default
      TNNetInput+TNNetSum onto every down-path skip + the mid output (encoder
      forward unperturbed, plain SDUNetDenoise stays bit-identical), and the new
      SDUNetDenoiseWithControl driver ADDS the ControlNetResiduals into those
      skips exactly as diffusers (down += controlnet_down, mid += controlnet_mid);
      combined forward parity-checked < 1e-4 in TestControlNetCombinedParity
      (tools/controlnet_combined_fixture.py reuses the two config-compatible pico
      fixtures) + examples/ControlNetCanny smoke. (b) real-checkpoint slicer (slice sd-controlnet-canny like
      slice_llama.py) for parity vs real weights on top of the synthetic fixture;
      (c) the diffusers per-residual conditioning_scale multiplier (constant 1.0
      here); (d) T2I-Adapter (lighter feature-injection sibling) LANDED —
      BuildT2IAdapterFromSafeTensors[Ex] imports the diffusers T2IAdapter
      full_adapter (PixelUnshuffle hint stem via TNNetSpaceToDepth with a conv_in
      input-channel permutation, an AvgPool ResNet-ish ladder of block1 3x3 ->
      ReLU -> block2 1x1 -> +identity ResnetBlocks, 1x1 in_conv channel changes);
      the T2IAdapterFeatures driver produces the per-stage feature pyramid and
      BuildSDUNet(..., pWithAdapter=true) + SDUNetDenoiseWithAdapter ADD those
      features into the SD UNet down-block hidden state (diffusers
      down_intrablock_additional_residuals); reuses existing layers (no new
      class). Feature parity < 1e-4 in TestT2IAdapterParity vs a numpy float64
      oracle (tools/t2i_adapter_tiny_fixture.py, diffusers not installed).
      Follow-ups: real-checkpoint slicer (TencentARC/t2iadapter_*_sd15v2),
      light_adapter variant, multi-adapter blending.
- [ ] Cohere real-checkpoint slicer follow-up (BuildCohereFromSafeTensors[Ex]
      for cohere + cohere2 LANDED on a dedicated parallel-residual builder,
      parity 3.96e-7/2.15e-7 vs HF float64 against SYNTHETIC config-faithful
      pico fixtures): add a real-weight slicer for Aya-Expanse-8B /
      Command-R7B (like slice_llama.py) so parity is checked against real
      Cohere weights on top of the synthetic fixture. Also: order_of_interleaved_layers
      (legacy cohere2 spelling) maps to sliding_window_pattern but was not seen
      in a published config — wire it if one surfaces.
- [ ] ONNX (or simpler JSON) export path — minimal viable: dump a
      forward-only graph for the currently-supported subset of layers,
      enough to run inference in onnxruntime. Doc which layers are
      out-of-scope for v1.
- [ ] Quantized inference follow-up (int8 weight-only v1 is landed): int8
      activation quantization + true
      int8 matmul kernels (AVX2 maddubs / dot-product paths) so quantized
      layers stop paying the per-forward dequantize cost; today the int8
      win is RAM only — compute still runs the FP32 kernels.
- [ ] Quantized inference follow-up: GPTQ/AWQ-style calibrated quantization
      (error-compensating rounding / activation-aware scale search) and
      4-bit (int4 pairs packed per byte, group-wise scales); also quantized
      EMBEDDING table storage (the one remaining FP32 heavyweight — vocab x
      d_model stays FP32 in v1, e.g. ~262MB for TinyLlama) and FP16/BF16
      weight storage as a zero-drift middle rung. GGUF dovetails (reader
      landed in neural/neuralgguf.pas): load Q8_0 blocks straight into the
      int8 weight-only storage instead of dequantize-then-requantize (the
      k-quant Q4_K/Q6_K/Q5_K/Q2_K dequant-at-load READ path has landed in
      neural/neuralgguf.pas).
- [X] Quantization-aware training (QAT) — port torch.ao.quantization's
      fake-quantize-during-training so accuracy lost to post-training int8 can be
      recovered (the landed int8 path is post-training, inference-only). LANDED:
      `TNNetFakeQuantize` (neural/neuralnetwork.pas) per-tensor SYMMETRIC int8
      fake-quant of the layer INPUT — FORWARD `x_q = round(clip(x/scale,-127,127))
      *scale = dequant(quant(x))` with `scale = max|x|/127` (exactly the landed
      `QuantizeWeightsInt8` weight-path convention, so QAT activation scales line
      up with the int8 weight-only export format); observer-driven scale via a
      MovingAverageMinMaxObserver-style running max-abs EMA (first obs seeds
      directly, frozen at inference / via `Freeze`); BACKWARD straight-through,
      reusing the `TNNetSoftCapping`/`TNNetStraightThroughEstimator` deriv-mask
      clamp machinery (grad passes through inside `[-qmax*scale,qmax*scale]`, zero
      outside). Registered in both CreateLayer dispatch tables; round-trips via
      Save/Load (FStruct[0]=qmax, FStruct[1]=frozen, FFloatSt[0]=momentum,
      FFloatSt[1]=running max-abs). Tests (all pass, full suite 2521/0/0):
      TestFakeQuantizeForwardDequantQuant (forward = dequant(quant(x))),
      TestFakeQuantizeSTEGradientCheck (STE grad = upstream inside band, 0
      outside), TestFakeQuantizeObserverAndRoundTrip (EMA + freeze + round-trip),
      plus registry round-trip builder #376. README layer table updated.
      FOLLOW-UPS (v1 scope = activations only): weight fake-quant, learned scale,
      per-channel/asymmetric variants, full int8 export wiring, and an
      `examples/QATFineTune` PTQ-drop-then-QAT-recover smoke run were deferred.
- [ ] FP16 (half-precision) OpenCL compute path for the dot-product/matmul
      offload. ENV NOTE (2026-06-27): the only OpenCL device available here is
      PoCL on the host CPU, which does NOT advertise `cl_khr_fp16` (clinfo shows
      "Half-precision Floating-point support (n/a)"). The detection + opt-in
      buffer/kernel path can be written and the FP32-fallback verified, but the
      actual half-precision kernel CANNOT be exercised/parity-tested here — defer
      until a `cl_khr_fp16` device is available, or scope a host-side
      half<->float conversion test only and mark the GPU kernel unverified.
      Every GEMM kernel in `neural/neural.cl` (cai_dot_product,
      simpleGEMMT, myGEMM5/6) is FP32 today, and the landed FP16/BF16 weight
      storage is RAM-only — compute still runs the FP32 kernels (tasklist
      note above is explicit). For GPU LLM decode the binding constraint is
      device memory bandwidth, so a `cl_khr_fp16`-guarded half-precision
      kernel variant (FP16 device-side weight/input buffers, FP32 accumulate)
      would roughly halve buffer traffic and lift throughput on GPUs that
      advertise the extension. Scope: detect `cl_khr_fp16` at device init,
      add an opt-in FP16 buffer/kernel path in neuralopencl.pas wired through
      the existing TDotProductSharedKernel offload (reuse the
      NewVAs/weight-residency machinery), keep the FP32 kernel as the
      fallback when the extension is absent, and add a numerical test that
      checks the FP16 result matches the FP32 path within a half-precision
      tolerance on a fixed fixture. Distinct from the int8 work (this is a
      GPU compute path, not CPU storage) and from FP16 weight storage (this
      is the missing compute half).
- [x] Keep activations resident on the OpenCL device across consecutive offloaded
      matmul/SDPA layers (eliminate the per-layer host round-trip during decode).
      Today each offloaded layer copies its input host->device, runs the kernel, and
      reads the result device->host (`FInputPrepared` in, `FOutput` out); the next
      offloaded layer immediately re-uploads that same buffer. For an LLM decode step
      a single block is qkv-proj -> SDPA -> o-proj -> FFN-up -> FFN-down, i.e. ~5
      back-to-back GEMM/attention layers whose intermediate tensors never need to
      touch the host. The weight buffers are already device-resident (the
      TDotProductSharedKernel weight-residency machinery the FP16 note references);
      this is the missing ACTIVATION half. Scope: add an opt-in "device output
      handle" so a layer can leave its result in a `cl_mem` buffer and the next
      offloaded layer can consume that buffer directly, with an automatic
      read-back only when the consumer is a CPU layer (norm, softmax, activation)
      or when the caller reads `.Output`. The non-offloaded layers in between
      (RMSNorm, RoPE, residual add) still run on host, so a first cut can fuse just
      the adjacent GEMM pairs (qkv->split, up->down) and measure; a full
      block-resident path is the stretch goal. Real value: device<->host traffic,
      not FLOPs, dominates the GPU decode wall-clock for ChatTerminal-scale models,
      so removing the redundant copies should lift tokens/sec materially. Guard
      everything behind the existing `FShouldOpenCL`, keep the host round-trip as the
      fallback, and pin parity with the SDPAOpenCLParity-style exact-vs-CPU test.
      DONE: TNNetLayer carries FOutputOnOpenCL / FOutputOnRAM plus
      OpenCLOutputBuffer(), so a producer offers its result in OpenCL memory and
      the consumer binds that buffer instead of downloading it. Wired through
      TNNetInput, the convolution/fullconnect GEMM family (whatever the size
      verdict), TNNetRMSNorm, TNNetTokenRMSNorm, TNNetRotaryEmbedding, the
      activation layers, TNNetPointwiseSoftMax, TNNetCellMulByCell,
      TNNetChannelMulByLayer, TNNetEmbedding (int8 device gather),
      TNNetDepthwiseConv1D, TNNetGatedDeltaNet and TNNetFusedSDPA (FP32 and int8
      resident KV cache). ForceOutputOnRAM stays the fallback for a host reader.
- [ ] Tokenizer follow-ups for neuralhftokenizer.pas:
      (b) DONE — raw SentencePiece .model protobuf path landed
      (LoadSentencePieceModel; hand-decoded ModelProto wire format, no
      vendored proto lib; auto-dispatched from LoadFromFile by the '.model'
      extension or a non-'{' first byte). Populates the SAME Unigram
      structures as the tokenizer.json path; pico fixture + spm-oracle parity
      test (tools/make_pico_spm_fixture.py, tests/fixtures/tiny_spm.model,
      TestSentencePieceModelParity). UNIGRAM model_type only; a BPE/WORD/CHAR
      ModelProto raises EHFTokenizerError ("use tokenizer.json").
      BPE-in-.model is now DONE — model_type=BPE (NLLB/mBART-BPE/DeBERTa-v3
      sentencepiece.bpe.model exports): LoadSentencePieceModel reconstructs the
      merges from the scored pieces (ReconstructBPEMerges, byte-for-byte the
      HF transformers generate_merges algorithm — per piece, every codepoint
      split whose both halves are in the vocab becomes a candidate ordered by
      (id_l,id_r), then a stable sort by (piece_id,len_l,len_r) gives the merge
      rank), sets FUnigram=false and routes encode/decode through the SAME
      metaspace + byte-level-BPE (BPEWord/EmitTokenOrFallback) machinery the
      tokenizer.json/GGUF-gpt2 BPE paths use; byte_fallback (type-6 pieces)
      covers non-ASCII via <0xNN>. Pico fixture tiny_spm_bpe.model
      (tools/make_pico_spm_fixture.py BPE variant) + TestSentencePieceBPEModelParity.
      Parity is FULL (exact), NOT an approximation: verified id-identical to the
      sentencepiece encoder over ASCII *and* non-ASCII (café/CJK/emoji/control-
      byte) inputs, and that encoder is itself id-identical to a tokenizers BPE
      model built from these same reconstructed merges. WORD/CHAR model_types
      still raise EHFTokenizerError. This UNBLOCKED the mBART/NLLB BART-family
      import (BuildMBartFromSafeTensors / BuildM2M100FromSafeTensors, LANDED)
      and the DeBERTa-v3 import (BuildDebertaV2FromSafeTensors, LANDED). No
      residual non-ASCII parity gap. BYTE
      (type=6) byte-fallback is now DONE: LoadSentencePieceModel sets
      FByteFallback when any type-6 piece is present, so (1) encode routes
      unknown chars through the <0xNN> BYTE pieces — the Unigram Viterbi
      unk node now decomposes the char into its UTF-8 bytes via byte
      pieces (EmitByteFallback) instead of one fused <unk>; and (2) decode
      emits the RAW byte for each <0xNN> piece (DecodeToken), so a run of
      byte pieces reassembles into the original multi-byte UTF-8 string
      (the metaspace U+2581 handling is unaffected). Round-trip parity test
      TestSentencePieceByteFallbackRoundTrip + tiny_spm_bytefallback.model
      fixture (tools/make_pico_spm_fixture.py byte_fallback variant; cases
      read via TJSONParser(s,[]) to keep their UTF-8 bytes intact). This
      UNBLOCKED the mBART/NLLB BART-family import and the DeBERTa-v3 import
      (both ship raw sentencepiece .model and are Unigram; all LANDED).
      (c) exact full-Unicode \p{L}/\p{N} tables (current
      classifier covers Latin/Greek/Cyrillic/Armenian/Hebrew/Arabic/
      Devanagari/Kana/CJK/Hangul; exotic scripts fall into the
      punctuation class of the GPT-2 regex); (d) DONE — build a tokenizer from
      GGUF tokenizer.ggml.* metadata (TNeuralHFTokenizer.LoadFromGGUF: opens
      via TNNetGGUFReader, reads tokenizer.ggml.model + tokens/scores/
      token_type [+ merges] and the bos/eos/unknown_token_id scalars, and
      populates the SAME internal structures the tokenizer.json/.model loaders
      use), so a single .gguf is self-contained for generation. SUPPORTED
      tokenizer.ggml.model values: "llama" (SentencePiece-with-scores →
      Unigram/Viterbi path: metaspace U+2581, <0xNN> byte fallback, CONTROL/
      UNKNOWN/USER_DEFINED pieces exposed as added tokens) and "gpt2" (byte-
      level BPE → GPT-2 byte table + cl100k Split pre-tokenizer + merges).
      DEFERRED: "bert" (WordPiece) and "no_vocab" raise EHFTokenizerError;
      gpt2 path assumes the cl100k Split pre-tok (GGUF carries no pre-tok
      config). Oracle = write-then-read round-trip: a tokenizer.json fixture's
      vocab/scores(+merges) is emitted to a temp .gguf via TNNetGGUFWriter and
      read back, then Encode/Decode are asserted id-identical to the
      tokenizer.json source (TestLoadFromGGUF{Llama,Gpt2}RoundTrip,
      TestLoadFromGGUFRejectsUnsupportedModel in TestNeuralHFTokenizer.pas).
- [ ] neuralhftokenizer.pas pre_tokenizer leftovers from the Split/Metaspace
      batch: ~~(a) a STANDALONE ByteLevel pre_tokenizer with use_regex=false
      silently applies the GPT-2 regex anyway (the flag is only honored
      implicitly inside the Sequence[Split, ByteLevel] path, which bypasses
      ByteLevelPieces; parse use_regex and skip the regex split when false)~~
      DONE: FByteLevelUseRegex parsed; standalone ByteLevel(use_regex=false)
      now feeds the whole segment as ONE chunk to the byte alphabet (no GPT-2
      regex). Test TestByteLevelNoRegexParityWithHF + bytelevel_noregex
      fixture group.
      (b) only the Qwen2 and Llama-3/cl100k Split pattern literals are
      recognized — the o200k (GPT-4o-family) and DeepSeek pattern strings
      raise EHFTokenizerError. ~~the DeepSeek family is now importable
      (BuildDeepSeekV2FromSafeTensors), so its pattern is the live gap~~ DONE
      for DeepSeek-V2/V2-Lite: the multi-Split+Digits Sequence is detected as
      a WHOLE (MatchesDeepSeekSequence) and dispatched to SplitDeepSeekPieces
      (a distinct splitter: \s?-prefixed letter/punct runs, separate CJK run,
      individual digits, \s+$ trailing). Test TestSplitDeepSeekParityWithHF +
      split_deepseek fixture. REMAINING sub-tasks: (i) DONE — o200k_base
      (GPT-4o family: gpt-4o/gpt-4o-mini/o1/o3) Split pattern recognized
      verbatim (csO200kSplitPattern) and dispatched to a dedicated
      case-aware SplitO200kPieces splitter (two letter alternations
      [\p{Lu}...]*[\p{Ll}...]+ and [\p{Lu}...]+[\p{Ll}...]*, \p{N}{1,3}
      digits, the [\r\n/]* punct trailing run that adds '/'). The Lu/Ll
      split uses the ASCII case tables; non-ASCII letters fall into the
      Ll/Lo class, so byte parity is guaranteed over ASCII letters (incl
      case transitions like HelloWorld->Hello+World, iPhone->i+Phone),
      digits, punct (incl '/'), and whitespace — the SAME approximation
      stance as the cl100k/DeepSeek splitters. Test
      TestSplitO200kParityWithHF + split_o200k fixture group
      (tools/hf_pretok_fixture.py build_o200k_split). RESIDUAL GAP: non-ASCII
      Latin AND CJK parity for SplitO200kPieces (would need the exact o200k
      Lu/Ll/Lo class tables ported to Pascal); (ii) DeepSeek-V3
      pre-tokenizer (different shape: \p{N}{1,3} + packed punct+letter Split,
      NO Digits stage) — not matched by MatchesDeepSeekSequence, falls through
      and raises; add a V3 variant when V3 is imported; (iii) non-ASCII Latin
      parity for SplitDeepSeekPieces — the HF onig engine isolates some
      precomposed Latin-1 letters that the \p{L} table groups, so exact byte
      parity is only guaranteed over ASCII/digit/punct/whitespace/CJK (the
      same approximation stance as the cl100k splitter; would need the exact
      DeepSeek letter-class range table ported to Pascal). Test: per-pattern
      parity fixtures like tools/hf_pretok_fixture.py.
- [ ] GGUF writer follow-up: byte-level-BPE end-to-end model export
      (SaveTokenizerToGGUF gpt2/llama tokenizer block + verify_gguf_writer.py
      llama-cpp-python logit-parity hook LANDED): SaveLlamaToGGUFEx itself still
      only takes a plain `Tokens` array (SP "llama" model) — route the gpt2
      tokenizer block through SaveTokenizerToGGUF from the MODEL exporter (not
      just the tokenizer unit test). DONE — SaveLlamaToGGUFEx gained an optional
      `Tokenizer: TNeuralHFTokenizer` arg (+ SaveLlamaToGGUFExWithTokenizer
      convenience overload) that, when non-nil, routes the FULL block through
      TNeuralHFTokenizer.SaveTokenizerToGGUF (gpt2 vocab+merges+ids / llama
      Unigram scores), taking precedence over the plain Tokens SP path (which is
      unchanged); covered by TestGGUFWriterGpt2TokenizerRoundTrip (one-call
      weights+gpt2-tokenizer export, read back self-contained: model logits
      < 1e-5 + Encode/Decode id-identical via LoadFromGGUF on the SAME file).
      Also: the llama-cpp-python parity arm is
      wired but UNVERIFIED end-to-end (lib not installed here; pico demo GGUF may
      be too small for llama.cpp) — confirm argmax/ranking agree on a real
      checkpoint once the lib is available.
- [ ] Optimizer zoo expansion (SGD/Adam/AdamW + Lion + Adafactor exist now):
      Lion + Adafactor DONE (commit 89da1f9). TNeuralOptimizerLion (Chen et al.
      2023, sign-based update with ONE momentum buffer vs Adam's two; decoupled
      AdamW-style weight decay via TNNet.ApplyDecoupledWeightDecay) and
      TNeuralOptimizerAdafactor (Shazeer & Stern 2018, factored R+C second-moment
      vectors vs Adam's R*C — 12<27 buffers proven for a 3x3x3 kernel; full
      per-element fallback for 1-D/non-factorable params) added to neuralfit.pas,
      with the per-neuron math in neuralnetwork.pas mirroring the
      CalcAdamDelta/UpdateWeightsAdam chain. Both require SetBatchUpdate(True)
      (TNeuralFit already sets it). Tests TestLionOptimizer / TestAdafactorOptimizer
      / TestAdafactorUsesFewerBuffersThanAdam (convergence + buffer count); full
      suite 0/0. Adafactor knobs deliberately omitted (documented in the class
      header): optional first-moment beta1 EMA + update RMS clipping; LR comes from
      the host fit schedule, not Adafactor's internal relative-step rule.
      REMAINING:
  - [ ] optional Adafactor follow-up: the omitted first-moment beta1 EMA + update
        RMS clipping + internal relative-step LR rule if a real fine-tune needs them.
- [ ] True single-kernel batched forward (real batch axis + attention padding
      mask for left-pad) — the lockstep DecodeBatchGreedy orchestration landed,
      but NN.Compute has no SIMD batch axis on the char path, so each step still
      pays one forward per running row. The vectorized batch is the actual
      prerequisite for an efficient speculative-decoding verify step.
- [ ] FlashAttention tiled online-softmax SDPA follow-ups (opt-in
      EnableTiledForward/ComputeTiled v1 LANDED — STANDARD-softmax forward only,
      forward-only, parity-gated TestSDPATiledOnlineSoftmaxParity < 1e-5): the
      exotic attention subclasses (differential/sink/ALiBi/cosine-sim/T5/soft-cap/
      segment/prefix-LM/block-causal/KV-cache) and the tiled (recompute-scores)
      BACKWARD are still open.
- [ ] longrope short-factor / dynamic switching follow-up (static long-context
      import landed): the import statically picks the long_factor table + long
      attention scaling. HF switches to short_factor when seq_len <=
      original_max_position_embeddings. Add a decode-time mode that selects the
      short table for short sequences (or document that the static long import
      is intentional for the 128k use case).
- [ ] CFG follow-up: a full-width-net -> width-1 unconditional-twin auto-clone
      so MakeUnconditionalTwin (commit 48e2fd2) works from a single imported
      model with no hand-built width-1 net. A SaveToString->LoadFromString
      round-trip preserves the ORIGINAL input width, so this needs either a
      rebuild-architecture-at-width-1 + CopyWeights walk (per-importer, or a
      generic layer-shape-independent rebuilder) or a width-rewrite on the
      serialized string. Assert the twin's logits match the source on a pinned
      input (the existing TestMakeUnconditionalTwinMatchesSourceLogits is the
      template).
- [ ] Preference-optimization follow-ups on the landed DPO/GRPO trainers
      (TNeuralGRPOTrainer in neural/neuraldpo.pas LANDED: group-relative
      advantages + PG + DeepSeek-k3 per-token KL reusing the DPO softmax-backward
      plumbing, tests in tests/TestNeuralGRPO.pas). ORPO / SimPO / KTO
      loss-formula deltas LANDED on TNeuralDPOTrainer as a TNeuralPreferenceLossMode
      (plmSimPO/plmORPO are reference-free via CreateReferenceFree; plmKTO uses a
      reference, paired-batch simplification documented in the unit header), tests
      in tests/TestNeuralPreference.pas. Bradley-Terry pairwise reward-model
      trainer DONE: TNeuralRewardModelTrainer in neural/neuraldpo.pas learns a
      scalar reward head r(x)=RewardNet(prompt+response) from (chosen,rejected)
      pairs with loss=-ln sigmoid(r_w-r_l) and the sigmoid-margin gradient
      (dL/dr_w=-(1-sigmoid(delta)), dL/dr_l=+(...)) via the DPO batch-update +
      direct-output-error plumbing; plugs into GRPO/Best-of-N as a learned
      reward; tests in tests/TestNeuralRewardModel.pas. REMAINING: a full
      unpaired-batch KTO variant (current KTO uses the paired pair-mean KL point).
- [ ] Sequence-length warmup curriculum in neuralfit.pas: train at short
      context first and grow SeqLen on a schedule (the rebuild-same-
      architecture-at-a-new-width idiom this list already notes near the
      variable-context trick) — large early-epoch throughput win for LM
      pretraining. Needs CopyWeights across width rebuilds + a schedule
      hook; the Trainer-callbacks task above is the natural home. Test:
      weights survive a width hop bit-for-bit, loss continuous across the
      hop.
- [x] Streaming/lazy tensor materialization with load-time quantization:
      the import path materializes full FP32 tensor buffers before copying
      into layers, so PEAK import memory, not steady-state, can be the gate
      on commodity RAM (the TinyLlama ~4.4GB case). Read one tensor at a
      time from the (seekable) safetensors stream straight into the
      destination layer — or straight into the landed int8 storage —
      keeping only one tensor-sized scratch buffer. Assert peak RSS during import
      stays within tensor-size + model-size on the parity fixture.
      DONE: TNNetSafeTensorsReader.CanStreamTensorRows plus the row-streaming
      loaders in neuralpretrained.pas read one tensor row at a time straight into
      the layer's int8 storage, so no FP32 copy of the checkpoint is ever
      allocated. Qwen2.5-7B peak import RSS dropped from 2431 MB to 1018 MB.
- [ ] NumPy .npz follow-up (neural/neuralnumpy.pas reader/writer landed; WRITER
      is STORED-only): DEFLATE-compressed .npz WRITER (savez_compressed) +
      zip64 / >4GB archive support (reader currently rejects zip64).
- [ ] MMLU few-shot eval follow-ups (EvaluateMMLU + MMLUReport single-token
      A/B/C/D answer-letter scoring, per-subject + macro + micro, 0/5-shot, in
      neuralnlpmetrics.pas; TestNeuralNLPMetrics MMLU tests; examples/MMLUEval
      tiny embedded smoke subset — all LANDED):
  - [ ] examples/MMLUEval --full <path> hook: dump cais/mmlu (hendrycks_test)
        dev+test splits to a small text file via the venv-x datasets package and
        feed the questions through the same FormatQuestion/BuildPrompt builder
        + a real subword tokenizer (the smoke build hard-codes its questions).
  - [ ] EvaluateMMLU large-model path: reuse the left-padded DecodeBatchGreedy /
        KV-cache batch scoring instead of the per-question per-letter
        ScoreCompletion loop (acceptable for the smoke subset, slow at the full
        14k-question x 57-subject scale).
- [ ] TinyStories reference-vs-from-scratch perplexity bake-off (follow-up
      to the landed, parity-verified roneneldan/TinyStories-1M import on
      the GPT-Neo route; the published pytorch_model.bin-only checkpoints
      now load DIRECTLY via the landed TNNetTorchBinReader — no Python
      conversion step needed, see examples/GPT2Import/README.md): the
      repo's NLP examples already train
      on the TinyStories dataset — compare neuralnlpmetrics perplexity of
      a from-scratch CAI training run against the imported
      roneneldan/TinyStories-1M..33M reference at matched vocab/context;
      at 1M-33M params the comparison runs against FULL checkpoints
      instead of sliced fixtures, the only importer family where that is
      true.
- [ ] RWKV/recurrent decode follow-ups (O(1) incremental decode for TNNetWKV +
      TNNetTokenShift + net-wide TNNet.BeginIncrementalDecode driver over
      TokenShift/WKV/SelectiveSSM/DiagonalSSM LANDED, commits 62165c1/cc1bfcb):
  - [ ] TNNetCrossWKV incremental path (two-source + asymmetric modes + receptance
        gate — non-trivial, deferred).
  - [x] Wire the net-wide recurrent driver INTO TNNetStreamingDecoder so GenerateTokens*
        drives TokenShift/WKV/SSM uniformly alongside the SDPA KV-cache (decoder
        currently only collects SDPA + TNNetDiagonalSSM, not TokenShift/WKV/SelectiveSSM).
        DONE: TNNetStreamingDecoder.IsRecurrentStateLayer collects every
        TNNetRecurrentDecodeBase descendant plus TNNetDiagonalSSM and
        TNNetTokenShift, and drives Begin/End/Reset/Capture/RestoreState on them
        alongside the SDPA KV-cache.
- [x] Mamba decode follow-up (O(1) incremental TNNetSelectiveSSM state-carry decode
      LANDED, commit 27ba256):
      DONE (see the sub-item).
  - [x] Full Mamba-BLOCK token-by-token decode: causal DepthwiseConv1D must carry
        its (kernel-1)-token ring buffer + in/out projections driven one token at a
        time, then wire into TNNetStreamingDecoder (mirrors the RWKV TokenShift
        block-integration follow-up).
        DONE: TNNetDepthwiseConv1D is a TNNetRecurrentDecodeBase descendant that
        carries its (kernel-1)-token history ring buffer across steps, and
        TNNetStreamingDecoder drives it, so a whole Mamba block decodes one token
        at a time. Verified end-to-end against HF on a real mamba-130m checkpoint.
- [ ] Refactor examples/WhisperTranscribe to USE the landed cached forced-prefix
      seq2seq decode helper (`DecodeSeq2SeqForcedPrefixCached`, neural/neuraldecode.pas).
      NOT a clean drop-in today: the word-timestamp cross-attention alignment step
      needs a full-width forward over the wide decoder, so the width-1 cached
      decode would have to either skip alignment or run a second full pass.
      (The example still re-encodes the full prefix every step; needs ~4 GB
      virtual memory, ulimit -v 4000000.)
- [ ] Streaming corpus loader with shuffle buffer: the landed packing
      pipeline materializes the whole token stream in RAM (neuraldatasets
      builds one concatenated Stream array). Read large text/token files
      in chunks through a fixed-size shuffle buffer (the datasets-library
      streaming pattern) feeding the existing packer windows — the only
      way to pretrain on corpora bigger than RAM. Assert: same model
      quality on a small corpus vs the in-memory path at matched
      examples-seen, and bounded RSS on a corpus larger than the buffer.
- [ ] Wav2Vec2 -large / robust LayerNorm variant + pretraining (follow-up to the
      landed Wav2Vec2/HuBERT CTC importer, which supports ONLY the wav2vec2-base
      "group" feat_extract_norm + post-norm encoder; ReadWav2Vec2ConfigFromJSONFile
      rejects feat_extract_norm="layer" and do_stable_layer_norm=true loudly). The
      -large/robust path needs: LayerNorm on EVERY conv feature-extractor layer
      (not just GroupNorm on conv 0), a PRE-norm transformer encoder
      (do_stable_layer_norm), and the final encoder LayerNorm placement that
      pre-norm implies. Then wav2vec2 SELF-SUPERVISED pretraining (the quantizer /
      contrastive masked-prediction heads currently dropped as ignorable tensors).
- [ ] HiFi-GAN neural vocoder importer follow-ups (`BuildHiFiGANFromSafeTensors[Ex]`
      LANDED + `ReadHiFiGANConfigFromJSONFile`/`HiFiGANConfigToString` +
      `THiFiGANConfig` + the `TNNetHiFiGAN` channel-major holder
      (Synthesize / SynthesizeVolume); mel-spectrogram -> waveform, the reusable
      synthesis backend shared by Tacotron2 / FastSpeech2 / SpeechT5 / Bark /
      VITS-style models). The generator is implemented as a self-contained
      convolutional holder (like TEnCodecModel) doing the conv_pre ->
      ConvTranspose1d-upsample + MRF (AVERAGE of num_kernels dilated ResBlocks) ->
      conv_post -> tanh math directly on channel-major arrays; conv weights fold
      weight_norm g/v at import (or load plain folded `.weight` as SpeechT5HifiGan
      ships). Discriminators skipped (training-only). Parity-gated < 1e-4 against an
      HF SpeechT5HifiGan float64 oracle (tools/make_pico_hifigan_fixture.py ->
      tests/fixtures/tiny_hifigan*, TestHiFiGANSynthesisParity). Open follow-ups:
  - [ ] real-checkpoint smoke: resynthesize a clip with a downloaded `hifigan` /
        SpeechT5HifiGan generator (weight_norm fold path) and write it via
        SaveVolumeToWav16 (offline + RAM-gated here, so deferred).
- [ ] AVX (and optional OpenCL) acceleration for the channel-major conv1d inner
      loops in the self-contained audio holders — `TEnCodecModel`, `TNNetHiFiGAN`,
      `TNNetVits`, `TNNetMimi`, and the MusicGen EnCodec decode path. These holders
      deliberately do the conv1d / ConvTranspose1d math DIRECTLY on channel-major
      arrays (not via TNNet layers), and those inner loops are currently scalar
      Pascal. After the EnCodec LSTM was AVX'd (~12.9x, real-T4 profiled), the 1D
      convolutions became the dominant cost of audio decode, so this is the next
      profiled bottleneck rather than a speculative optimization.
      FORWARD conv1d DONE (commit 8e15052): EnCodec (RunEnCodecConv serial +
      threaded TEnCodecConvWorker — covers TEnCodecModel + MusicGen EnCodec decode)
      and HiFiGAN/Vits (RunHiFiGANConv — covers TNNetHiFiGAN and the nested Vits
      decoder) now gather an im2col receptive-field patch (size InCh*K) per output
      position and run each output channel as one contiguous TNNetVolume.DotProduct;
      all 17 audio parity tests stay < 1e-4 on both scalar-fallback and real -dAVX2
      builds.
      CONVTRANSPOSE1D (upsample) DONE (commit 2873a1f for EnCodec + HiFiGAN/Vits;
      Mimi grouped transpose in 30c2342 and DAC ungrouped transpose in 8277d25 were
      vectorized inline when those holders landed): the overlap-add scatter is now a
      transposed-im2col — InT packs InSig as [t*InCh+i] (each output column
      InCh-contiguous) and WT repacks W as [(o*K+k2)*InCh+i] (each (o,k2) tap
      InCh-contiguous), so Full[o][t*Stride+k2] += DotProduct(WT_{o,k2}, InT_t, InCh)
      dispatches to AVX (Mimi/DAC use the Double-precision MimiDotProductD per group).
      EnCodec's threaded TEnCodecConvWorker.RunTranspose splits over output channels.
      The (t,k2) overlap-add order matches the original scatter; only the inner
      in-channel sum reassociates (parity < 1e-4). Re-verified all 17 audio parity
      tests (EnCodec round-trips, TestMimiParity, TestDACRoundTripParity,
      TestHiFiGANSynthesisParity, TestVitsSynthesisParity, TestMusicGen*DecoderParity)
      pass on BOTH scalar-fallback and real -dAVX2 builds. REMAINING:
  - [ ] OpenCL offload of the Mimi (`RunMimiConv`) / DAC (`RunDACConv`) holders —
        blocked on a Double-precision shared dot-product kernel (their oracle gate
        is `< 1e-4` against a float64 reference; a Float kernel would not hold it).
        (Forward conv1d + ConvTranspose1d OpenCL offload for EnCodec/HiFiGAN/Vits
        has landed via the `EnableConvOpenCL`/`SetConvOpenCLMinWork` gated path.)
      Design note (the landed forward path follows this): channel-major layout
      means the contraction (sum over in-channels for a fixed kernel tap) is
      depth-axis-contiguous — exactly the case `TNNetVolume.DotProduct` / `MulAdd`
      already vectorize — so the win is reusing the existing AVX volume primitives
      (or a small im2col-into-DotProduct reshape) for the per-tap accumulation
      instead of triple-nested scalar loops. Gate any remaining rewrite on the
      existing parity tests (TestHiFiGANSynthesisParity / TestVitsSynthesisParity /
      EnCodec round-trip) staying `< 1e-4`, and re-profile decode wall-clock
      before/after.

- [x] Single-sample `TNNetFullConnect` forward threading follow-ups (opt-in
      `EnableFullConnectThreading` multi-core `ComputeCPU` v1 LANDED for
      FullConnect/FullConnectLinear/FullConnectReLU — off by default, bit-identical
      to serial, work-thresholded via `SetFullConnectThreadingMinWork`):
      (a) the per-token projection HOLDERS (ChatTerminal / MusicGenText /
      `RunLlama`-style decoders) still need to call
      `EnableFullConnectThreading(true)` to opt in — wire it into the imported-model
      run paths; (b) re-profile one-token decode wall-clock before/after on a real
      imported decoder to tune the default threshold; (c) the int8-quantized
      `ComputeQuantizedInt8CPU` and `TNNetFullConnectSigmoid`/other activation
      siblings are still serial.
      DONE: EnableFullConnectThreading is gone - the inference scheduler chunks
      any layer that overrides ComputeRange, and TNNetFullConnect,
      TNNetFullConnectLinear and TNNetFullConnectReLU all do, int8 included
      (ComputeQuantizedInt8Range is the body of both the serial and the chunked
      path). ChatTerminal computes parallel by default (--serial opts out);
      measured ~2x fast-memory / ~1.3x low-memory on real Qwen decode.
- [ ] Parler-TTS importer end-to-end follow-up (`BuildParlerTTSFromSafeTensors[Ex]`
      + `TParlerConfig` + `TParlerTTSModel` holder + `examples/ParlerTTS` LANDED,
      model_type `parler_tts`; (By)T5 description encoder cross-attention + shared
      `BuildMusicGenDecoderNet` codec-LM decoder over delay-patterned DAC codes +
      SDPA KV-cache width-1-twin AR decode; pico parity TestParlerTTSParity < 1e-4):
      pair the decoder with the real (By)T5 encoder + DAC decoder to a WAVEFORM
      end-to-end (all three importable in-tree); the real parler-tts-mini-v1
      checkpoint key mapping + tokenizers; classifier-free guidance; sampling controls.
- [ ] VITS / MMS-TTS end-to-end text-to-speech importer (`BuildVitsFromSafeTensors[Ex]`
      LANDED + `ReadVitsConfigFromJSONFile`/`VitsConfigToString` + the `TVitsConfig`
      record + the `TNNetVits` channel-major holder (Analyze / ExpandPrior /
      FlowReverse / Synthesize) + examples/TextToSpeech) — the FIRST text-to-speech
      model in the library (model_type `vits`: facebook/mms-tts-* and
      kakao-enterprise/vits-ljs; Kim et al. 2021). Like the HiFi-GAN vocoder it is a
      self-contained channel-major holder doing the INFERENCE math directly (conv1d /
      relative-position attention / WaveNet residual stacks). All four stages landed
      and parity-gated `< 1e-4` vs the HF `VitsModel` float64 oracle
      (`TestVitsSynthesisParity`, `tools/make_pico_vits_fixture.py` ->
      `tests/fixtures/tiny_vits*`): (1) the conditional-prior normalizing FLOW run in
      reverse — VITS's residual coupling is RealNVP/Glow ADDITIVE coupling
      (`log_stddev≡0`, so `second_half -= mean(first_half)` with the mean from a
      `conv_pre`->WaveNet->`conv_post` stack, weight_norm g/v folded) — note this
      REUSES the WaveNet/coupling math directly rather than the generic
      `TNNetAffineCoupling`/`TNNetInvertible1x1Conv` Glow layers (VITS's coupling is
      its own WaveNet-conditioned variant, not a layer-graph fit); (2) the HiFi-GAN
      DECODER reused from `TNNetHiFiGAN` via a shared `BuildVitsDecoderInto` (loaded
      under the `decoder.` key prefix, bias-free `conv_post`); (3) the
      relative-position text encoder + the DETERMINISTIC duration predictor + the
      monotonic length-regulator expansion; (4) the full text->waveform graph (the
      prior noise `z` is an EXPLICIT input so the parity test is deterministic). The
      posterior encoder (training-only) is dropped from the committed fixture and
      never required. Example writes a smoke WAV via `SaveVolumeToWav16`. Open
      follow-ups:
  - [ ] MULTI-SPEAKER models (`num_speakers>1` / `speaker_embedding_size!=0`, the
        global-conditioning `cond` convs into the WaveNet/duration/decoder) —
        rejected loudly.
  - [ ] real-checkpoint smoke: synthesize a sentence with a downloaded
        `facebook/mms-tts-eng` / `kakao-enterprise/vits-ljs` and write it via
        `SaveVolumeToWav16` (offline + RAM-gated here, so deferred).
- [ ] Kokoro-82M TTS, real checkpoint (`hexgrad/Kokoro-82M`, Apache-2.0, StyleTTS2 +
      iSTFTNet). Plan agreed 2026-10-12. Target: load the HF snapshot folder
      (`config.json`, `kokoro-v1_0.pth`, `voices/*.pt`) and write a 24 kHz WAV.
      Reference: the `kokoro` 0.9.4 package in the `x` venv (`kokoro/model.py`,
      `modules.py`, `istftnet.py`). One Opus agent per task, run serially, test +
      commit after each.
      Decisions:
      - REPLACE the pico `TNNetKokoro` with the real architecture. Its AdaIN used
        plain `gamma`; the checkpoint needs `(1+gamma)`. Removed in commit 403d17f2
        (`TNNetKokoro`, its config/AdaIN records and builders,
        `tools/make_pico_kokoro_fixture.py`, the `tiny_kokoro*` pico fixtures,
        `TestKokoroSynthesisParity`, examples/KokoroTTS); the names are free for
        the real model.
      - Noise: zero noise by default (repeatable output, simple parity), random noise
        as an option.
      - Code structure: `TNNet` stages run by `TNNet.Compute`, so OpenCL comes with
        the layers; as many `TNNet`s as needed, host steps between them (duration
        rounding + frame repeat, sine source + forward STFT).
      - G2P comes later (task 3.2). Until then the input is a phoneme string.
      Checkpoint facts (from the cached snapshot and the package source):
      - 81.7M parameters, 327 MB FP32; fits under the 3 GB ulimit, so real-checkpoint
        runs are possible on the dev box.
      - `kokoro-v1_0.pth`: torch zip, entries stored (no DEFLATE), root = dict of 5
        state dicts `bert`, `bert_encoder`, `predictor`, `text_encoder`, `decoder`,
        every key prefixed `module.`. Weight norm stored as legacy `weight_g`/`weight_v`.
      - `voices/<name>.pt`: torch zip whose root is a BARE tensor [510,1,256]. Row =
        phoneme count - 1. Columns 0..127 = decoder style, 128..255 = prosody style.
        54 voices, 9 languages; only `af_heart` is downloaded here.
      - PL-BERT = ALBERT: word embedding 178x128, `embedding_hidden_mapping_in`
        128->768, ONE shared layer (`albert_layer_groups.0.albert_layers.0`) run 12
        times, 12 heads, FFN 2048, max 512 positions. Then `bert_encoder` 768->512.
      - AdaIN1d: InstanceNorm WITHOUT affine weights (none in the checkpoint), then
        `(1+gamma)*x + beta` with gamma/beta from Linear(style 128 -> 2C).
      - frame rate: decoder upsamples x2, generator x10 x6, iSTFT hop 5 = 600 samples
        per predicted frame = 40 frames/s at 24 kHz.
  Phase 0 — oracle and structure:
  - [x] 0.1 `tools/` script: run the installed `KModel` on a fixed phoneme string
        with `af_heart`; dump every stage (BERT out, durations, F0/N, decoder blocks,
        harmonic source, waveform) as the real-checkpoint reference. Patch `SineGen`
        noise to zero (or dump it) so Pascal can match.
        Done: `tools/make_kokoro82m_fixture.py real` -> ~/.cache/neural-api/kokoro82m_ref/
        `kokoro82m_ref.{safetensors,json,wav}` (not committed, 92 MB).
  - [x] 0.2 Same script: tiny real-architecture fixture (the package's torch
        classes, small dims, random weights) -> `tests/fixtures/`, replacing the
        numpy pico fixture.
        Done: `tools/make_kokoro82m_fixture.py tiny` -> `tests/fixtures/tiny_kokoro82m/`
        (`config.json`, `kokoro-v1_0.pth`, `voices/af_heart.pt`) + `tiny_kokoro82m_ref.{safetensors,json}`.
  - [x] 0.3 Per-utterance length: token count and frame count change every
        sentence, and a `TNNet`'s length is fixed when it is built. Measure the cost
        of rebuilding the stage nets from in-memory weights vs. padded buckets
        (padding changes InstanceNorm statistics and the reverse LSTM, so buckets need
        masking). Report before Phase 2.
        Done: use a BORROWING REBUILD per utterance, the `TQwenImage21VaeDecoder`
        pattern: one owner net per stage built once at a small length, then per
        utterance `Net.BuildWeightOwner := Owner`, build at T/F, `BuildWeightOwner :=
        nil` (layers link the owner's neurons, size no weights; OpenCL: arm the owner,
        `EnableOpenCLInContextOf`, layers borrow its device weights). Measured on a
        stage-C-like net (133 layers, 78.7M weights: 4 TDNN k3 blocks at 1024 ch,
        Deconvolution stand-ins x2/x10/x6, 2 MRFs of TDNN k3/7/11 d1/3/5 at 256/128 ch),
        CPU, F=100 frames, two runs (VM noise ~2-3x): owner build at F=8 119-690 ms,
        355 MB; borrowing build 244-863 ms, 0 weight elements, +410 MB (activations
        only, ~4.1 MB/frame; F=300 +1225 MB); rebuild + CopyWeights 350-1174 ms +
        38-101 ms copy, +720 MB (a second 315 MB weight copy); forward 6.2-8.9 s.
        Borrow output == copy output bit-exact. In the same run the borrowing build
        (863 ms) is ~13% of one stage-C forward. Rejected: re-propagating sizes in place (`SetPrevLayer` is private,
        many overrides call `InitDefault` and wipe weights, `TNNetSum`/concat size in
        their constructor, OpenCL buffers are sized at `EnableOpenCL`); padded buckets
        (needs a valid length on ~58 AdaIN InstanceNorms + ~65 conv inputs re-zeroed
        in stage C, the reverse LSTM start, an ALBERT key mask, and pays the padding
        in compute) to save ~13%. Traps: `TNNetInstanceNorm.Create(false)` is
        a PER-ELEMENT affine sized by length (use the default `true`, gamma 1 beta 0);
        build the owner long enough that `TNNetConvolutionAbstract.SetPrevLayer` does
        not clamp a kernel (Deconvolution K20 p5 needs input >= 10). OpenCL arming
        per rebuild not measured (no GPU here).
  - [ ] 0.4 Layers for the `TNNet` route:
        - [x] dilated / strided 1D conv (torch Conv1d geometry). Done: new
          `TNNetConv1DLinear` (a `TNNetConvolutionLinear` with a Kx1 kernel,
          X-only padding, dilation along X) instead of extending `TNNetTDNNConv1D`:
          `TNNetConvolutionAbstract` gained `FPaddingY`/`FDilationX`, the CPU im2col
          (FP32 and int8), `BackpropagateFastTiledCPU`, `cai_im2col` and the implicit
          GEMM honour them, so the tiled/threaded CPU and both OpenCL forwards serve
          every Kokoro geometry (commit 2ff0b5d4). Tests: `TestConv1DLinear{TorchParity,OpenCLParity,
          SaveLoadRoundTrip,BorrowingBuild,GradientCheck}` against
          `tests/fixtures/conv1d_oracle.json` (`tools/conv1d_oracle.py`, torch).
          CPU, 256 ch k7 d3 L=2000: old TDNN 304-335 ms, new 260-266 ms serial,
          161-189 ms threaded (4 vCPU VM).
        - [ ] `ShouldUseInterleavedDotProduct` picks `InterleavedDotProduct` for
          every conv with a multiple of 32 neurons; on 1D convs it is slower than
          `DotProductsTiled`: same conv at 250 ch (tiled) 199 ms serial / 89 ms
          threaded vs 256 ch (interleaved) 260 / 161-189 ms. Kokoro's 128/256/512
          channel convs all take the interleaved path. Measure on 2D convs too
          before changing the heuristic.
        - [ ] OpenCL weight borrowing (`BuildWeightOwner` + `EnableOpenCLInContextOf`)
          is not tested for `TNNetConv1DLinear` (the CPU borrowing build is).
        - [ ] The unused backward variants (`BackpropagateCPU`, `BackpropagateFastCPU`,
          `BackpropagateFastCPUDev`, `BackpropagateOpenCL`) ignore `FPaddingY` and
          `FDilationX`; only `BackpropagateFastTiledCPU` is called.
        - [x] transposed 1D conv (torch ConvTranspose1d). Done: new
          `TNNetDeconv1DLinear(NumFeatures, K, Stride, Padding, OutputPadding,
          Groups=1, SuppressBias=0)`, a `TNNetDeconvolutionLinear` with a Kx1
          kernel, no Y padding / output padding, no kernel clamp, groups in
          FStruct[6] (commit cf181f32). `TNNetDeconvolution` gained
          `FOutputPaddingY`, `FGroups` (grouped CPU scatter forward + backward;
          grouped layers stay off OpenCL) and a CPU GEMM forward for dense
          inference-only layers (`FConcatedWeights` x input positions with
          `DotProductsTiled`, chunked, then a host scatter shared with the OpenCL
          path). The old "11x wasted work" was weight memory, not CPU time: the
          scatter skipped out-of-range rows. CPU, ups1 512->256 k20 s10 p5, L=200:
          old 2D stand-in (20x11) 397-459 ms, new 1D inference 90-109 ms, new 1D
          trainable (scatter) 394-470 ms. Tests: `TestDeconv1DLinear{TorchParity,
          OpenCLParity,SaveLoadRoundTrip,BorrowingBuild,GradientCheck}` against
          `tests/fixtures/conv_transpose1d_oracle.json`
          (`tools/conv_transpose1d_oracle.py`, torch). Loader mapping: torch weight
          [In, Out/G, K]; neuron Oc, group g = Oc div (Out/G), gets
          `Weights[k, 0, c] := w[g*(In/G) + c, Oc mod (Out/G), k]` (no flip; dense:
          `w[c, Oc, k]`), bias as is, then `FlushWeightCache`.
        - [ ] The CPU GEMM needs `SetTrainable(False, {pLowMemory=}False)`: with the
          default low memory (and when trainable) the deconv runs the per-neuron
          scatter, ~4x slower at ups1. Decide for the Kokoro stage nets.
        - [ ] `TNNetDeconvolution` is single-threaded (`ChunkEligible` false); a
          neuron-axis `ComputeRange` over the CPU GEMM needs per-thread chunk
          scratch.
        - [ ] Depthwise pool runs one length-1 `DotProduct` per (position, tap,
          channel); vectorize over channels if it shows in a stage-C profile.
        - [ ] OpenCL: the scatter GEMM uploads the input, downloads all L*K*N
          products and scatters on the host; no resident source/output, and no
          borrowing (`EnableOpenCLInContextOf`) test for the deconv.
        - [ ] Base `SetPrevLayer` still builds `FConcatedWInter` (unused by the
          deconv) for 32-multiple channel counts: one extra weight copy (dense
          deconvs only now).
        - [x] Grouped deconv under AVX: the base `SetPrevLayer` armed the
          interleaved copy with the dense vector size (K*InDepth) over the
          K*InDepth/G rows, so `InterleaveWithXFrom` divided by zero at >= 32
          channels and >= 40 output positions (real stage B at 102 frames; the
          8-frame owner never reached it). Fixed: `FNoInterleavedWeights` (set
          for groups > 1) keeps `ShouldUseInterleavedDotProduct` false. Test
          `TestDeconv1DLinearDepthwiseWide` (raised EDivByZero under -dAVX2
          before the fix).
        - [x] per-channel Snake. Done: `TNNetSnake.Create(Alpha, PerChannelAlpha=
          true)` learns one alpha per channel in `Neurons[0].Weights` (Depth long,
          FStruct[0] = 1; scalar files unchanged); inference forward is row-wise
          `TNNetVolume` Mul/Sin/Mul/Mul/Add, trainable forward a scalar loop with
          the derivative; backward has the alpha gradient (commit 9a6ec901). Loader: torch
          `alpha1[j]` [1,C,1] -> `Weights.Raw[c]`, then `FlushWeightCache`.
        - [x] AdaIN1d / AdaLayerNorm / channel LayerNorm, no new class. AdaIN1d =
          `TNNetInstanceNorm` (default per-channel affine; reduces over time per
          channel, torch InstanceNorm1d), AdaLayerNorm and TextEncoder `LayerNorm` =
          `TNNetTokenLayerNorm(1e-5)` (commit 9a6ec901). Per utterance the HOST computes
          [gamma; beta] = fc(s) and writes `Neurons[0].Weights[c] := (1+gamma)*w`,
          `Neurons[1].Weights[c] := (1+gamma)*b + beta` (w/b = the norm's own affine,
          1/0 when absent), then calls `FlushWeightCache` on the layer of the net
          that runs (a borrowing net's layers share the owner's neurons; the
          TokenLayerNorm re-uploads its OpenCL gamma/beta only on that flag,
          the InstanceNorm and the per-channel Snake upload theirs every forward).
          `TNNetGroupNorm`/`TNNetInstanceNorm` OpenCL: `cai_group_norm` is now one
          work-group per group (local tree reduction, was one work-item per
          group), binds a source in OpenCL memory and keeps its output there when
          not trainable. Activations: `TNNetLeakyReLU.Create(0.2 / 0.1 / 0.01)`,
          `TNNetHyperbolicTangent`, `TNNetSigmoid` (all OpenCL already); new
          `cai_activation` opcodes MulConstant (`TNNetMulByConstant`, for the
          rsqrt(2) and /3 scalings after `TNNetSum`), Snake and per-channel Snake
          (a per-layer channel-parameter buffer on `TNNetIdentity`). Tests:
          `TestKokoroStyleNormTorchParity`, `TestKokoroActivationTorchParity`
          (tests/fixtures/kokoro_norm_activation_oracle.json from
          `tools/kokoro_norm_activation_oracle.py`, the kokoro classes),
          `TestKokoroLayersOpenCLParity` (chain stays in OpenCL memory, then a
          host style rewrite), `TestSnakePerChannelSaveLoadRoundTrip`,
          `TestKokoroLayersBorrowingBuild`, `TestSnakePerChannelGradientCheck`;
          `TestActivationOpenCLParity` gained the 3 opcodes.
        - [x] Generator head `exp(x[:, :11])` / `sin(x[:, 11:])`: `TNNetExp` /
          `TNNetSin` exist on CPU only (no opcode); 22 x frames, so either
          `TNNetSplitChannels` + those layers or a host step before the iSTFT.
          Done: host step (the net ends at conv_post; 22 x (120F+1) floats,
          one download, and the iSTFT is on the host anyway).
        - [x] Host helper for the style fold (fc(s) -> per-channel gamma/beta)
          lives only in the test (`WriteStyleIntoChannelAffine`); move it into the
          Kokoro loader (2.5) rather than writing a second copy.
          Done: `WriteStyleIntoChannelAffine(NormLayer, FcWeight, FcBias, Style,
          NormWeight, NormBias)` in neuralpretrained (TNNetVolume arguments); the
          test helper converts the oracle JSON and calls it (commit 3f1f670e).
        - [ ] `TNNetSnake` scalar-alpha inference still runs the scalar
          `pcr_sinf` loop; the per-channel path is vectorized.
        - [ ] The new `cai_group_norm` and the OpenCL residency of the norm /
          activation chain are unmeasured on a GPU (PoCL parity only).
  - [ ] 0.5 Stage-net holder for Kokoro: owner nets for stages A/B/C built once
        from the checkpoint, `PrepareNet(T)` / `PrepareNet(F)` borrowing rebuild that
        exits early when the length is unchanged, OpenCL arming as in
        `TQwenImage21VaeDecoder.PrepareNet` (neuralpretrained.pas:69716). Reuse that
        code path; do not clone it.
        Stage A done (commit 3f1f670e): `TNNetKokoro.Create(Reader, Config)` builds
        the stage-A owner once at 8 tokens and loads it (a stage-A tensor left
        unread raises); `PrepareStageA(T)` = `BuildWeightOwner := owner`, build,
        `nil`, `PrepareInferenceThreads`, `ShareHostOutputsByLiveness`, OpenCL
        arming; `RunStageA(Ids, ProsodyStyle, Speed, d, t_en, PredDur)`. The VAE
        decoder's context-net creation and owner arming became
        `CreateOpenCLContextNet` / `ArmWeightOwnerInOpenCLContext`, used by both.
        Inference: `SetTrainable(False)` with low memory: the shared ALBERT
        repeats keep no per-layer weight cache. Measured (real, 32 tokens, CPU,
        AVX2 release, VM noise): fast memory gave no gain and +1.2 s per rebuild;
        switching the sized net to fast memory after the build without
        `FlushWeightCache` raised EInvalidOp (empty caches). Real stage A: owner
        load 1.2-2.2 s, PrepareStageA 0.09-0.22 s, forward 0.32-0.44 s parallel /
        0.56-0.88 s serial (other runs on a busy VM: 1.3-3.9 s).
        - [x] Stage C: add it beside stages A and B (`FStage*Owner`,
          `FStage*Net`, one shared `FOpenCLContextNet`, `ArmStageNet`).
          Decoder part done: `TNNetKokoro.LoadStageCDecoder` builds the owner
          once at 8 frames (`KokoroStageCDecoderOwnerFrameCount`) and loads
          every `decoder.*` tensor except `decoder.generator.*` (one left unread
          raises); `PrepareStageCDecoder(F)` / `ReleaseStageCDecoder` /
          `RunStageCDecoder(t_en, PredDur, F0Curve, NCurve, DecoderStyle, x)`,
          same borrowing rebuild, `ArmStageNet` and inference settings as
          stages A and B. Real decoder, 102 frames (`fpc -O2` test build, no
          AVX2, threaded, CPU): owner 31,155,936 weights (124.6 MB); 3.1-4.5 s
          build + forward (2 runs); activations 171.5 MB with host-output
          sharing off, 163.5 MB on (20 of 58 layers shared); the test process
          (all three owners + the decoder net) peaks at 802 MB RSS.
        - [x] Stage C generator: recommended as a FOLLOWING net
          (`StageCGenerator`), not appended to the decoder net. Inputs: x
          (2F,1,512) at index 0, the host harmonic spectrum (120F+1,1,22) at
          index 1. The host writes `Layers[1].Output`, then
          `ComputeFromLayerOutput(StageCDecoderNet.GetLastLayer())` hands x
          over (an OpenCL buffer copy when x is resident; `RunStageCDecoder`
          would then skip its host copy). Why: generator parity can start from
          the dump's dec_decode3 alone; one net would share no host storage
          across the two (exact-size liveness slots, no common shapes); the
          hand-over costs one 0.4 MB copy.
          Done as recommended: `LoadStageCGenerator` builds the owner once at
          2 frames (`KokoroStageCGeneratorOwnerFrameCount`; no generator kernel
          clamps at any length); `PrepareStageCGenerator(F)` /
          `ReleaseStageCGenerator` / `RunStageCGenerator(x, spectrum, style,
          Audio)`, same borrowing rebuild and `ArmStageNet`. `Synthesize`
          hands x over with `ComputeFromLayerOutput(StageCDecoderNet last
          layer)`; `RunStageCDecoder` still downloads and copies x (0.4 MB)
          because it returns it. Inference: low memory, except the two ups
          deconvs in fast memory (`SetTrainable(False, False)` +
          `FlushWeightCache` after the borrowing build) so they run the CPU
          GEMM: standalone real shapes, CPU, best of 4: ups0 512->256 k20 s10
          L=204 407 -> 285 ms (borrowed weights 398 -> 284 ms), ups1 256->128
          k12 s6 L=2040 471 -> 378 ms; +12 MB weight cache. Not measured
          inside the real generator (no real-size runs on the dev VM).
          `TNNetKokoro.Create` now checks EVERY checkpoint tensor is consumed
          (one shared Consumed list, `RaiseOnUnreadKokoroTensors` over
          bert/bert_encoder/predictor/text_encoder/decoder; real file passes).
          - [ ] Real-size generator memory / time not measured here: the
            generator parity and end-to-end tests on the real checkpoint (and
            the real stage-C decoder ones) are opt-in,
            `NEURAL_KOKORO82M_REAL_HEAVY=1`; an earlier real generator run hit
            the VM's OOM killer (2.7 GB RSS + swap) with host-output sharing off.
          - [ ] OpenCL: `TNNetPadXY` (the reflection pad) has no kernel, so it
            downloads ups1's output and its consumer re-uploads (6 MB at 102
            frames); the ups deconvs scatter on the host (see 0.4). Not
            measured on a GPU (PoCL parity only).
        Stage B done: `TNNetKokoro.LoadStageB` builds the stage-B owner once at 8
        frames (`KokoroStageBOwnerFrameCount`) and loads it (a stage-B tensor left
        unread raises, via `RaiseOnUnreadKokoroTensors`, now shared with stage A);
        `PrepareStageB(F)` = the stage-A borrowing rebuild with early exit; the
        post-build steps (threads, host-output sharing, OpenCL owner arming once,
        `EnableOpenCLInContextOf`) are `ArmStageNet`, used by both stages.
        `RunStageB(d, PredDur, ProsodyStyle, F0Curve, NCurve)`: host frame
        expansion (`RepeatTokenRows`), 12 AdaIN1d style folds
        (`WriteStyleIntoChannelAffine`), `TNNet.Compute`, then the (2F,1,2)
        output split into F0_pred / N_pred (2F,1,1). Inference settings as stage A
        (`SetTrainable(False)`, low memory): the depthwise pool is grouped, and
        grouped `TNNetDeconvolution` runs the CPU scatter in either memory mode
        (the GEMM path is dense-only), so fast memory buys nothing here. The
        borrowing net sizes no weights (`WeightElementsSized` = 0). Real stage B,
        102 frames, CPU (`fpc -O2` test build, no AVX2, threaded, one run):
        236 ms build + forward.
        - [ ] `TNNetEmbedding` sizes its table in its constructor (before the net
          is known), so a borrowing build allocates and frees each table: 3
          tables, 114,176 floats on the real checkpoint. Other layers size none.
        - [ ] The owner build sizes each ALBERT repeat's rows before
          `LinkWeightsFrom` frees them (transient, one block at a time).
        - [ ] CPU profile of the real stage A: `TNNetLSTMCell` ~51% of layer time
          (the CPU input projection is per-step dot products, no GEMM; the OpenCL
          path batches it only at Depth >= 32), `TNNetPointwiseConvLinear` ~44%.
        - [ ] Real-checkpoint stage A on a GPU: not run (PoCL parity on the tiny
          fixture only). The AdaLayerNorms after the bi-LSTMs run on the CPU
          unless forced, because the LSTM output is on RAM.
  - [x] 0.6 `TNNetLSTMCell.SetPrevLayer` sizes its 12 rows directly, so a borrowing
        build allocates and `InitDefault`s them, then `LinkNeuronsFrom` frees them.
        Correct but wasted; honour `DeferWeightSizingToBuildLink` there (~5 lines).
        Done: `TNNetLSTMCell` and `TNNetGRUCell` (same issue) skip sizing under
        `DeferWeightSizingToBuildLink` (the `TNNetGatedDeltaNet` pattern) and now
        count `WeightElementsSized`. Test `TestBidirectionalRecurrentBorrowingBuild`
        (LSTM and GRU stacks, owner at 6, borrower at 11). Commit 74287878.
  - [ ] 0.7 Stage-C activation memory is ~4.1 MB/frame, so > 3 GB above ~600 frames
        (15 s of audio). `ShareHostOutputsByLiveness` shared only 14 of 133 layers
        (32 MB) on the test net: find out why before Phase 4, or cap F per chunk.
        Same on the real stage-C decoder: 20 of 58 layers shared, 171.5 -> 163.5 MB.
  Phase 1 — loading:
  - [x] 1.1 Read `kokoro-v1_0.pth` with `TNNetTorchBinReader` (first check: its
        pickle parser accepts this file's `OrderedDict`s); map names (strip `module.`).
        Done: the unpickler accepts BUILD only as a dict's attribute dict (the
        state dicts' `_metadata`, dropped); the five nested state dicts flatten
        to `<component>.module.<name>` and `OpenKokoroCheckpoint` renames them to
        `<component>.<name>` (`RenameTensorPrefix`, no reader copy). Real file:
        548 tensors, 81,763,410 parameters, all F32; every tensor read once
        (1.45 s, 19.5 MB RSS); counts and sum/sum-abs match torch. The extra zip
        entries of the tiny fixture (`.format_version`, `.storage_alignment`)
        needed no change.
  - [x] 1.2 `TNNetTorchBinReader`: accept a bare-tensor root (rejected today at
        neuraltorchbin.pas:1018); voice-pack row selection.
        Done: a bare-tensor root is exposed as the one tensor
        `TorchBinBareTensorName` ('tensor'). `LoadKokoroVoicePack` (checks
        [rows, 1, 2*style_dim]) + `SelectKokoroVoiceStyle(Pack, PhonemeCount,
        DecoderStyle, ProsodyStyle)`: row PhonemeCount-1 (kokoro's
        `pack[len(ps)-1]`, code points of the phoneme string incl. symbols the
        vocab skips); raises outside 1..rows.
  - [x] 1.3 Config reader: `plbert`, `istftnet`, `vocab` (keys are multibyte IPA
        characters: look up per UTF-8 code point).
        Done: `TKokoroConfig` / `ReadKokoroConfigFromJSONFile` (escapes decoded,
        raw-UTF-8 parse; each vocab key must be one code point; embedding_size
        defaults to 128). The istftnet arrays go into a `THiFiGANConfig` through
        the new `ReadHiFiGANGeneratorFields`, now shared by the HiFi-GAN and VITS
        config readers. `KokoroPhonemesToIds` (binary search per code point,
        skipped symbols returned, raises past max_position_embeddings).
        `ReadKokoroWidthsFromWeights` sets DecoderWidth / DecoderBlockInputWidth /
        DecoderOutputWidth / AsrResWidth / DecoderBlockCount (real 1024 / 1090 /
        512 / 64 / 4) and checks word-embedding and conv_post shapes against the
        config. The fpjson 255-byte key limit does not apply (keys are 1-3 bytes).
  - [x] 1.4 Load from the snapshot folder: one call takes the folder plus a voice
        name (`af_heart`) and finds the three files. The example takes the folder as
        a command-line argument (the HF cache path contains a commit hash).
        Done: `OpenKokoroSnapshot(Dir, VoiceName, out Config, out Voice)` returns
        the reader; errors name the missing folder / config.json / voice, and
        list the *.pth files when there is not exactly one.
        Tests: `TestKokoroCheckpointTensors`, `TestKokoroVoicePackRowSelection`,
        `TestKokoroConfigFromJSONFile`, `TestKokoroPhonemesToIds`,
        `TestKokoroSnapshotLoad`, `TestKokoroRealSnapshotSmoke` (Ignored when no
        snapshot is in the HF cache; `NEURAL_KOKORO82M_DIR` overrides).
        Commit d6db9f49.
  Phase 2 — components, in graph order:
  - [x] 2.1 ALBERT: extend `BuildBertFromSafeTensors` with the embedding projection
        and one shared layer (do not clone it); then `bert_encoder` 768->512.
        Done (a77647bb, 3f1f670e): the BERT builder's encoder and loader became
        `AddBertEncoder` / `LoadBertEncoderWeights`; family `bfAlbert` adds
        `EmbeddingSize` tables, `embedding_hidden_mapping_in`, and repeats 1.. that
        `LinkWeightsFrom` block 0 (one weight copy; the borrowing build links the
        owner's root via `LinkBuildWeightOwner`). AlbertConfig defaults: gelu_new
        (`TNNetGELU`), eps 1e-12, 2 token types. Tiny max |diff| <= 1.1e-6; real
        bert_layer0..11 <= 2.0e-5 abs (max |ref| ~11.4), d_en 3.3e-5.
        - [ ] `bfAlbert` through `BuildBertFromSafeTensorsWithConfig` (HF
          `albert-*` checkpoints) is untested; no config reader for model_type
          "albert"; pooler / classifier / int8 refused.
  - [x] 2.2 TextEncoder: embedding, 3x (weight-norm Conv1d k5, channel LayerNorm,
        LeakyReLU 0.2), bi-LSTM 512 -> 2x256.
        Done (3f1f670e): `TNNetEmbedding`, `TNNetConv1DLinear` k5 p2 loaded by
        `LoadTorchConv1DLinear` (weight_norm fold `LoadTorchConv1DWeight`, now
        shared with `LoadHiFiGANConv`), `TNNetTokenLayerNorm(1e-5)`,
        `TNNetLeakyReLU(0.2)`, bi-LSTM. The bi-LSTMs need input != hidden: new
        `TNNetLSTMCell.Create(Hidden)` + `AddBidirectionalLSTM(...,
        CellsReadInputWidth)` (b5381f69); `LoadTorchLSTMInto(..., StackInput)`
        loads one stack of several. Real: text_cnn <= 8.6e-6, t_en 1.9e-6.
        - [ ] `TNNetGRUCell` has no input-width option.
  - [x] 2.3 `AdaLayerNorm` (LayerNorm, `(1+gamma)`, beta from Linear(style)) and the
        `DurationEncoder` loop (3x bi-LSTM + AdaLayerNorm, style concatenated each
        round).
        Done (3f1f670e): the style is input channels 2.. of stage A (host-filled,
        the same row on every token); one `TNNetSplitChannels` feeds every
        `[norm; style]` `TNNetDeepConcat`. No weights (nothing to borrow), one
        upload with the ids, OpenCL-resident after. `RunStageA` writes each
        AdaLayerNorm's gamma/beta with `WriteStyleIntoChannelAffine` before the
        forward. Real: dur_enc_adanorm0..2 <= 8.4e-5 abs (rel. 1.7e-5), d 1.9e-5.
  - [x] 2.4 Duration head: bi-LSTM, Linear 512->50, sigmoid, sum, /speed, round,
        clamp >= 1; frame expansion by repeat (no alignment matmul).
        Done (3f1f670e) up to the durations: bi-LSTM, `TNNetPointwiseConvLinear`,
        `TNNetSigmoid` in the net; sum / speed / `Round` (half to even, as
        torch.round) / clamp on the host. pred_dur exact on tiny and real.
        Tests: `TestKokoroStageATinyParity`, `TestKokoroStageABorrowingRebuild`,
        `TestKokoroStageAOpenCLParity`, `TestKokoroRealStageAParity`.
        - [x] Frame expansion by repeat (d and t_en, per pred_dur): the host step
          before stage B. Done: `RepeatTokenRows(Source, Durations, Dest)` (one
          `Move` per frame); stage C reuses it for t_en -> asr.
  - [x] 2.5 AdaIN1d as above.
        Done: `TNNetInstanceNorm` + the host fold `WriteStyleIntoChannelAffine`;
        the fc of each AdaIN1d is loaded by `LoadKokoroStyleFc` (shape-checked,
        shared with stage A's AdaLayerNorms). The checkpoint has no InstanceNorm
        affine (torch builds `affine=True` and leaves it at 1/0).
        Bug fixed on the way: `NeuralPerChannelMeanInvStd` (the CPU
        instance-norm path of `TNNetGroupNorm` and `TNNetAdaIN`) computed the
        variance as E[x^2] - mean^2, contrary to its declared contract; on a
        near-constant channel (tiny F0_block0 norm1: variance 2e-5, eps 1e-5)
        that cancelled to a 2.4e-4 norm1 error. Now centered (sum then
        sum((x-mean)^2)); norm1 error 1.9e-6. Costs one extra pass (a `Move` +
        2 `MulAdd` per position); not timed. The OpenCL kernel was already
        centered.
  - [x] 2.6 `AdainResBlk1d`: two weight-norm convs + shortcut (nearest x2 upsample,
        1x1 conv without bias), times rsqrt(2). The x2 block uses a DEPTHWISE
        ConvTranspose1d (k3, s2, groups=C, output_padding 1): add groups to
        `THiFiGANConv` (or the `TNNet` equivalent).
        Done: `AddKokoroAdainResBlock(NN, InputLayer, DimOut, Upsample)` returns
        `TKokoroAdainResBlockLayers` (Norm1/Norm2/Pool/Conv1/Conv2/Shortcut/
        Output) for stage C's encode/decode blocks too: InstanceNorm, LeakyReLU
        0.2, pool = `TNNetDeconv1DLinear(C, 3, 2, 1, 1, groups=C)`, conv1/conv2 =
        `TNNetConv1DLinear` k3 p1; shortcut = `TNNetDeMaxPool(2, 0, pPoolSizeY=1)`
        (new Y factor: an X-only nearest x2 on (L,1,C); `FStruct[3]`, 0 = old
        behaviour; OpenCL gather when Y factor = X factor or SizeY = 1) then
        `TNNetPointwiseConvLinear(DimOut, SuppressBias=1)`; `TNNetSum` +
        `TNNetMulByConstant(1/sqrt 2)`. `LoadKokoroAdainResBlockWeights` loads
        the convs; new `LoadTorchConvTranspose1DLinear` (grouped torch layout,
        weight_norm folded) and `LoadTorchConv1DLinear` accept a conv without
        bias (shared `LoadTorchConvBias`). Test `TestDeMaxPoolAlongXOnly`.
        - [ ] Grouped `TNNetDeconvolution` stays on the CPU under OpenCL: each
          pool downloads its source and re-uploads its output (2 pools in stage
          B, 9 of 63 layers off OpenCL in a forced run; stage C has more).
        - [ ] A Y factor other than the X factor on SizeY > 1 runs
          `TNNetDeMaxPool` on the CPU (`cai_upsample_gather` has one factor).
  - [x] 2.7 `F0Ntrain` (shared bi-LSTM, 2x3 `AdainResBlk1d`, 1x1 projections) and
        the Decoder (`F0_conv`/`N_conv` k3 s2, `encode`, `asr_res`, 4 `decode` blocks
        with the concat).
        F0Ntrain done: `BuildKokoroStageBNet(NN, Config, F)`: input en (F,1,
        hidden+style = 640 real), shared bi-LSTM (cells read the input width),
        then the F0 and N branches fork from it in one net (3 blocks each:
        d_hid->d_hid, d_hid->d_hid/2 x2, d_hid/2->d_hid/2), `F0_proj`/`N_proj` as
        `TNNetPointwiseConvLinear(1)`, `TNNetDeepConcat` -> (2F,1,2). Tests:
        `TestKokoroStageBTinyParity` (abs: en 0, f0n_shared 6.0e-8, F0_block0..2
        4.5e-6/1.3e-5/1.8e-5, N_block0..2 5.2e-6/3.4e-6/8.8e-6; F0_pred / N_pred
        checked relative because F0 is in Hz: 5.2e-4 of 140.7 = 3.7e-6, 6.3e-6 of
        2.1), `TestKokoroStageBBorrowingRebuild` (F = 10 and 6, vs a fresh build),
        `TestKokoroStageBOpenCLParity` (PoCL, incl. a style rewrite),
        `TestKokoroRealStageBParity` (real dump, 102 frames, max rel. diff
        1.6e-6: f0n_shared 1.2e-6 abs, F0_block1 5.3e-5 of 35.9, F0_pred 1.5e-4 of
        312.6 Hz, N_pred 5.7e-6 of 9.6). Stage-B parity feeds the dump's d and
        pred_dur, so it does not include stage A's error (d: 7.4e-6 rel.).
        - [x] Decoder half (stage C).
          Done: `BuildKokoroStageCDecoderNet(NN, Config, F)`: three
          `TNNetInput`s, the first layers of the net, fed by
          `TNNet.Compute([asr, F0, N])` (the RAFT / F5-TTS multi-input idiom;
          inputs of different lengths need no packing): asr (F,1,hidden),
          F0 and N (2F,1,1). F0_conv / N_conv = `TNNetConv1DLinear(1, 3, 1, 2)`
          on the curves, `TNNetDeepConcat([asr, F0, N])`, encode =
          `AddKokoroAdainResBlock(.., 1024)`, asr_res =
          `TNNetPointwiseConvLinear(64)` on asr, then per decode block
          `TNNetDeepConcat([x, asr_res, F0, N])` (torch order) and
          `AddKokoroAdainResBlock`; only the last block upsamples (to 2F, 512
          out), so the package's `res` flag never stops the concat. The host
          expands t_en to asr with `RepeatTokenRows`; the decoder style is the
          FIRST half of ref_s. The norm1/norm2 fc loader is now
          `LoadKokoroAdainResBlockStyleFcs`, shared with stage B. Tests:
          `TestKokoroStageCDecoderTinyParity` (abs: asr 0, dec_N_conv 2.4e-7,
          dec_encode 3.8e-6 of 49.7, dec_asr_res 1.5e-8, dec_decode0..3
          1.5e-5/1.1e-5/7.6e-6/1.5e-5 of up to 106; dec_F0_conv relative
          9.2e-8), `TestKokoroStageCDecoderBorrowingRebuild` (F = 10 and 6 vs a
          fresh build), `TestKokoroStageCDecoderOpenCLParity` (PoCL, incl. a
          style rewrite: 57 of 58 layers on OpenCL, the grouped pool stays on
          the CPU; max |diff| x 7.6e-6 of 49.6), `TestKokoroRealStageCDecoderParity` (real dump, 102
          frames, max rel. diff 2.6e-6: dec_F0_conv 1.5e-7, dec_N_conv 6.6e-8,
          dec_encode 1.3e-6, dec_asr_res 8.1e-7, dec_decode0..3 6.9e-7/7.5e-7/
          8.4e-7/2.6e-6; host-output sharing on == off).
        - [x] Real stage A + B chained (our d into stage B) not compared to the dump.
          Done: `TestKokoroRealStagesChainedToDecoder` runs A -> B -> decoder on
          the real sentence from input_ids and ref_s only: pred_dur exact,
          F0_pred 4.8e-4 Hz (rel. 1.5e-6), dec_decode3 1.5e-4 abs of 35.3 (rel. 4.2e-6);
          3.3 s for the three stages, 999 MB peak RSS. Tolerance 1e-3 relative.
        - [ ] CPU profile of the real decoder (3.1-4.5 s at 102 frames is slow
          for ~31M weights x 102-204 frames); not run. Real decoder on a GPU:
          not run.
        - [ ] Stage B on a GPU and a CPU profile: not run.
  - [x] 2.8 `SineGen` / `SourceModuleHnNSF`: F0 upsampled x300, 9 harmonics, the
        down / cumsum / up linear-interpolation phase, voiced threshold 10, Linear
        9->1 + tanh.
        Done: `TNNetHarmonicSource` in neuralaudio.pas (a general hn-NSF piece,
        not Kokoro-specific): `Create(SampleRate, UpsampleScale, HarmonicCount,
        SineAmp, NoiseStd, VoicedThreshold)`, `SetMergeWeights`, `Compute(F0,
        HarmonicSource)`, `ComputeUpsampledF0AndVoicing`. Zero noise by default;
        `RandomNoise` + `NoiseSeed` (FPC RNG, global RandSeed restored, the seed
        advances per call) add rand_ini and the Gaussian noise. Torch semantics
        verified bit-exact in Python against the real dump (scratch scripts):
        nearest index = floor(float(i) * float(1/300)); interpolate with
        scale_factor uses the coordinate scale float(1/scale_factor)
        (recompute_scale_factor unset) and output length floor(L * scale_factor)
        in double; the source coordinate is FMA-contracted,
        fma(scale, i + 0.5, -0.5) in float, clamped at 0; the blend is
        fma(x0, w0, x1 * w1); cumsum accumulates in double; phase =
        ((float(cumsum) * 2) * float(pi)) * 300. The down-sample reads samples
        300j+149/150 with weight 0.5, so rand_ini (added to sample 0 only) never
        reaches the phase: the random initial phase is a no-op in Kokoro.
        Tests: `TestHarmonicSourceTorchParity` (kokoro SourceModuleHnNSF, U=6,
        negative / unvoiced / high F0; f0 up and uv exact, har_source < 1e-6),
        `TestHarmonicSourceRandomNoise`; fixture
        tests/fixtures/stft_harmonic_source_oracle.json
        (`tools/stft_harmonic_source_oracle.py`).
  - [x] 2.9 Public forward complex STFT (n_fft 20, hop 5, Hann, centre reflect pad)
        in neuralaudio.pas; only power-spectrum and private FFT routines exist.
        Done: `TNNetForwardSTFT(NFFT, Hop)`: torch.stft defaults and sign,
        `Transform(Wave, Re, Im)`, `TransformToMagnitudePhase(Wave, Dest)`
        (Dest = [|X|; atan2] along Depth, torch.cat order). Direct DFT with
        `TNNetVolume.DotProduct` against window-premultiplied cos / -sin tables
        built in Create (the radix-2 FFT does not take 20; `RealFramePowerSpectrum`
        returns power only); interior frames read the wave in place, edge frames
        a reflect-padded scratch; DC / Nyquist Im forced to +0 (torch gives
        phase 0 or +pi there). Tests: `TestForwardSTFTTorchParity` (n_fft
        20/16/15), `TestForwardSTFTCosinePeakBins`.
        Kokoro wiring: `TNNetKokoro.ComputeHarmonicSpectrum(F0Curve, Spectrum)`
        -> (120F+1, 1, 22), the generator's second input; `LoadHarmonicSource`
        reads `decoder.generator.m_source.l_linear` (upsample scale =
        prod(upsample_rates) * hop = 300). Tests:
        `TestKokoroHarmonicSpectrumTinyParity`, `TestKokoroRealHarmonicSpectrumParity`.
        Real sentence (204 F0 frames -> 61200 samples -> 12241 STFT frames):
        gen_f0_upsampled and gen_uv exact; gen_har_source 3.0e-8 abs (rel.
        2.1e-7, phases up to ~3.8e4 rad); gen_har_spec 2.4e-7 (rel. 1.9e-7);
        complex spectrum 2.4e-7; phase (only where |X| > 1e-4 max |X|, wrapped)
        4.0e-4 rad. Tiny: har_source 1.5e-8, spec 2.4e-7, phase 3.9e-4 rad.
        Host step 41 ms on the real sentence (`fpc -O2` test build, no AVX2).
        - [x] ARBITRARY PHASE BINS (kept, user decision 2026-10-14): zero noise makes har_source the constant
          tanh(bias) on unvoiced spans, so bins 2..10 of those frames are
          rounding noise. Real sentence: 44,238 of 134,651 bins at |X| <=
          1e-7 max (52,990 at <= 1e-4). Torch (MKL FFT) is deterministic there
          (same pattern on every constant frame, e.g. bin 2 phase 2.03) but the
          values are MKL rounding artefacts, so no Pascal FFT can reproduce
          them. They feed the generator's noise_convs: replacing torch's
          phases with a float64 STFT's (a stand-in for any other FFT) moved
          the torch audio by 0.087 max abs (peak 0.32), RMS 0.0037 vs signal
          RMS 0.047. So: generator parity (2.11) must feed the dump's
          gen_har_spec / gen_har_phase; end-to-end audio parity will not get
          below ~0.1 abs; decide whether to zero or keep those phases.
        - [ ] `TNNetForwardSTFT` is O(NFFT * bins) per frame; a long NFFT would
          want `RealFramePowerSpectrum`'s two-factor path with complex output.
        - [ ] The harmonic source runs scalar `pcr_sinf` (arguments up to ~4e4
          rad need its exact range reduction; the vector `Sin` was not checked
          at that range) on one thread; 41 ms total is small next to the nets.
  - [x] 2.10 `AdaINResBlock1`: HiFi-GAN resblock1 + AdaIN + per-channel Snake
        (`TNNetDAC.RunSnake`); extend `THiFiGANResBlock` rather than clone it.
        Done as `TNNet` layers instead (the host-run `THiFiGANResBlock` does not
        fit the TNNet-stage decision): `AddKokoroAdaINResBlock1(NN, Input, K,
        Dilations)` -> `TKokoroAdaINResBlock1Layers`; per dilation d:
        `TNNetInstanceNorm`, per-channel `TNNetSnake`, `TNNetConv1DLinear(C, K,
        d*(K-1)/2, 1, d)`, again with d = 1, `TNNetSum` with the residual.
        `LoadKokoroAdaINResBlock1Weights` loads convs1/convs2 (weight_norm
        folded), alpha1/alpha2 (`LoadKokoroSnakeAlpha`) and the adain fc
        (`LoadKokoroStyleFc`). Used for noise_res and the MRF resblocks.
  - [x] 2.11 Generator: noise convs (k12 s6 p3; k1), `noise_res`, ConvTranspose1d
        ups, reflection pad (1,0) on the last stage, MRF average of 3 resblocks,
        `conv_post` 128->22, exp (magnitude) / sin (phase), iSTFT via
        `ISTFTOverlapAdd` with the centre trim and the phase-sign flip.
        Done: `BuildKokoroStageCGeneratorNet(NN, Config, F)`: inputs x
        (2F,1,512) and the harmonic spectrum (120F+1,1,22); per stage
        `TNNetLeakyReLU(0.1)`, noise conv (`TNNetConv1DLinear(c, 2*s, (s+1)
        div 2, s)`, s = prod(rates[i+1:]); last stage
        `TNNetPointwiseConvLinear`), noise_res (k7 / last k11, d 1,3,5), ups
        `TNNetDeconv1DLinear(c, k, u, (k-u) div 2)`, on the last stage
        `TNNetPadXY(1, 0, pmReflect, 0)`, `TNNetSum`, 3 resblocks, `TNNetSum` +
        `TNNetMulByConstant(1/3)`; then `TNNetLeakyReLU(0.01)` and conv_post.
        Lengths (real): x 2F -> ups0 20F (= noise conv0 (120F+1+6-12) div 6 +
        1) -> ups1 120F -> pad 120F+1 (= the 1x1 noise conv); the builder
        raises on a mismatch. `TNNetPadXY` gained `TrailingPaddingX`
        (FStruct[3] = it + 1, 0 = symmetric as before; test
        `TestPadXYTrailingPaddingX`). Host: exp / sin per element, then new
        `ISTFTCenteredMagnitudePhase` (neuralaudio: torch.istft center=True:
        `ISTFTOverlapAddReIm` with Im negated, NFFT div 2 trimmed at both ends,
        600F samples). Top level: `Synthesize(ids, decoder style, prosody
        style, speed, Audio)`, `SynthesizePhonemes(phonemes, voice, ...)`,
        `SynthesizePhonemesToWav` (`SaveVolumeToWav16`, 24 kHz). Tests (tiny,
        relative to max |ref|): gen_stage0 leaky 0, noise_conv 3.7e-7,
        noise_res 8.4e-7, ups 1.8e-7, sum 5.7e-7, resblock0..2 8.7e-7/7.2e-7/
        6.2e-7, mrf 5.9e-7; gen_stage1 noise_conv 1.4e-7, noise_res 6.2e-6,
        ups / pad 5.4e-7, sum 5.6e-6, resblock0..2 4.4e-6/4.3e-6/3.8e-6, mrf
        4.3e-6; post_leaky 3.8e-6; gen_spec 9.4e-6; gen_phase 7.0e-5 abs;
        audio 0.81 of 47963 (1.7e-5) (`TestKokoroStageCGeneratorTinyParity`);
        `TestKokoroISTFTTinyParity` (dump gen_spec/gen_phase -> audio, 1.2e-7);
        `TestKokoroStageCGeneratorBorrowingRebuild` (F = 10 and 6 vs a fresh
        build); `TestKokoroStageCGeneratorOpenCLParity` (PoCL: 120 of 185
        layers on OpenCL as armed, 162 forced; conv_post <= 2.0e-5 abs of ~7,
        audio <= 3.8e-5 of ~8);
        `TestKokoroTinySynthesizePhonemes` (equals the stages one by one;
        dec_decode3 7.4e-6, F0_pred 9.0e-6 rel.).
        - [ ] Tiny end to end vs the dump audio: 43763 of 47963 max abs. The
          generator on the dump's x with OUR harmonic spectrum gives the same
          43763, so it is the arbitrary phase bins of 2.9 (random tiny weights
          and exp() amplify them). Real-checkpoint end-to-end diff not
          measured here (opt-in test `TestKokoroRealEndToEnd`, writes
          pascal.wav beside the dump).
        - [ ] `ISTFTCenteredMagnitudePhase` and `ISTFTOverlapAddReIm` allocate
          their scratch per call (once per utterance).
  Phase 3 — text input:
  - [x] 3.1 Phoneme string -> ids (vocab), wrapped `0 ... 0`, at most 510 tokens;
        split long input into chunks.
        Done: ids by `KokoroPhonemesToIds` (1.3; raises past
        max_position_embeddings, as KModel's assert); chunking of text by 5.6.
        A raw phoneme string (`--phonemes`) is not chunked, as in KPipeline
        (`generate_from_tokens` raises on a phoneme string over 510).
  - [ ] 3.2 G2P (text -> phonemes). misaki English = gold/silver dictionaries
        (~6 MB JSON), spaCy part-of-speech tagging, num2words, espeak-ng fallback.
        Options: (a) phonemes as input (user runs misaki); (b) Pascal dictionary
        lookup, English only (`a*`/`b*` voices); (c) espeak-ng shared-library
        binding. With (a) every voice works; (b) covers English only.
        User 2026-10-12: no Python dependency, so (a) is out. Found: misaki
        `us_gold.json` 90,201 + `us_silver.json` 93,361 entries (Apache-2.0), only 790
        keyed by part of speech (`read`: DEFAULT `ɹˈid`, VBD `ɹˈɛd`);
        `libespeak-ng.so.1` (1.51, GPL-3.0) is installed on the dev box.
        User 2026-10-12: no C library either (pure Pascal), so (c) is out. Scope:
        (b), English only (`a*`/`b*` voices), DEFERRED until synthesis works.
        SUPERSEDED 2026-10-13 by Phase 5 below (no download: the user passes the
        dictionaries in a folder; unknown words go through sub-word splitting).
  Phase 4 — integration:
  - [x] 4.1 examples/KokoroTTS: `--model-dir`, `--voice`, `--speed`, `--phonemes`,
        writes 24 kHz WAV. The pico example of that name was removed in 403d17f2:
        recreate it (new .lpr needs cthreads first; re-add its `.gitignore` entry
        for the output WAV, the binary entry is still there) and its
        examples/README.md entry.
        Done: `KokoroTTS --model-dir DIR --phonemes "IPA" [--voice af_heart]
        [--speed 1] [--out kokoro.wav] [--serial] [--max-threads N] [--gpu]
        [--gpu-platform N] [--gpu-device N] [--no-gpu-shared-kernel]` (OpenCL and
        thread flags named as in QwenImage; OpenCL off by default because no
        Kokoro stage has run on a GPU). `OpenKokoroSnapshot` + `TNNetKokoro` +
        `SynthesizePhonemes` + `SaveVolumeToWav16`; prints skipped symbols, token
        count, frame count, audio seconds, load / synthesis / WAV-write / total
        time and peak RSS (VmHWM). No per-stage times: `Synthesize` does not
        report them. README: phonemes from misaki's English G2P (reference
        `həlˈO wˈɜɹld. ðˌɪs ɪz ɐ tˈɛst.`), the three files to download, zero
        noise, English voices only until G2P, and only the memory figures
        measured here (no real end-to-end peak yet). `lazbuild -B` Default,
        Debug, Release: all build. Tiny-fixture smoke (`hˈO.`): 6 tokens, 10
        frames, 0.25 s audio, WAV written, peak RSS 21 MB (CPU); the Debug build
        (range checks) and `--gpu` on PoCL also wrote it.
  - [ ] 4.2 Per-stage parity on the tiny fixture (< 1e-4) + one real-checkpoint
        test against the 0.1 dump, under the 3 GB ulimit.
        Tiny per-stage parity is in the default suite (see 2.x). The real
        checkpoint part stays open: the real stage-C decoder, chained, generator
        and end-to-end tests (`TestKokoroRealStageCDecoderParity`,
        `TestKokoroRealStagesChainedToDecoder`,
        `TestKokoroRealStageCGeneratorParity`, `TestKokoroRealEndToEnd`) are
        opt-in (`NEURAL_KOKORO82M_REAL_HEAVY=1`, plus the dump from
        `tools/make_kokoro82m_fixture.py real`) and have not run end to end on
        this VM; the user's real-machine check is those tests and
        `KokoroTTS --model-dir <snapshot> --phonemes "həlˈO wˈɜɹld. ðˌɪs ɪz ɐ
        tˈɛst." --out hello.wav`.
        - [x] DECIDED (2.9, 2026-10-14, after the user's listening test): keep
          the arbitrary harmonic-spectrum phase bins on unvoiced spans (our
          STFT's rounding values).
  - [x] 4.3 README (planned vs supported split; "supported" only after the user
        tests it), this entry.
        Done: root README "More audio" highlight names `KokoroTTS` as planned
        (coded), not yet tested on the real checkpoint; examples/README.md entry
        and examples/KokoroTTS/README.md say the same. Moved to supported on
        2026-10-14 after the user's real run.
  Phase 5 — text input and REPL (plan agreed with the user 2026-10-13; order:
  5.1-5.7 G2P, then 5.8 positional model folder, then 5.9 REPL). CODED:
  5.1-5.9 done; the real checkpoint with text input is untested (user run
  pending).
  Rules: pure Pascal at run time (no Python, no C library, no OpenSSL download);
  no Python at test time (a `tools/` script writes fixtures once on the dev box).
  - [x] 5.1 New unit `neural/neuralg2p.pas`: English G2P, not Kokoro-specific
        (a later TTS model can reuse it); `TNNetKokoro` calls it. Port of misaki
        `en.py` (0.9.4, 712 lines) WITHOUT spaCy: word splitting, capital letters
        and acronyms, possessives (`'s`), contractions, hyphenated words, the
        -s / -ed / -ing suffix rules for words found only as their stem, stress
        adjustment, punctuation passed through (Kokoro's vocab has
        `, . ! ? ; : — …` and quotes).
        Done (commits c1af8ee1, 51b1c903). API: `TNeuralEnglishG2P.Create(DictDir,
        g2pAmerican|g2pBritish)` (neuralg2p.pas:248), `Phonemize(Text): string`
        and `Phonemize(Text, out Tokens)`; each `TNeuralG2PToken` (:73) has
        Text, Whitespace, Tag (rule tagger, Penn tags), Phonemes, IsKnown and
        Resolutions, a set of punctuation / dictionary / context rule / POS
        rule / stem rule / spelled / unknown / number pending
        (`G2PResolutionName` prints them, for `/show`). Also exported:
        `ApplyG2PStress`, `G2PSubTokenize` (misaki's sub-token regex, matched by
        hand), `EnglishG2PAccentFromVoice` (`a*` US, `b*` GB, others raise; it
        lives in neuralg2p, not in the Kokoro code), `EnglishG2PDictionaryFileName`.
        `TNNetKokoro.SynthesizeText` calls it (5.6); 5.8/5.9 wire it into the example.
        Ported as in en.py: retokenize grouping, the right-to-left loop with
        future_vowel / future_to, the longest-merge loop over sub-tokens,
        resolve_tokens (junk pieces, prespace, demotion of the lighter
        primaries), merge_tokens, get_special_case (a/am/an/I/by/to/in/the/vs/
        used, % & + @, ADD dot/slash), get_NNP spelling, is_known, lookup
        (grown keys, gold then silver, the "None" key at a clause end, NNP
        spelling when the entry has no primary stress), get_word (case folding,
        `s'` and `'` endings, stems), _s/_ed/_ing (US ᵻ, GB ɪ, US flap, GB
        refuses -ing after ə/ː), the ɾ -> T / ʔ -> t mapping. spaCy's tokenizer
        is approximated: whitespace chunks; prefixes `( { " “ ‘ /`, the
        backtick and `$ £ € ¥` (read ''); suffixes `. , ! ? ; : … ) } " ” %` (and `’` only after an
        opening `‘`), one token per character; a final `.` stays on a dictionary
        abbreviation (Dr., Mr.) or a dotted one (U.S., Ph.D.); infixes `--`, `—`,
        `–` between letters/digits; clitics `'s n't 're 've 'll 'd 'm` and
        plural `s'` split off (straight and curly). Unicode helpers come from
        neuralhftokenizer (IsLetterCP, LowercaseCP, the new UppercaseCP, NFKC
        UnicodeNormalize); \p{Lu}/\p{Ll} are exact for ASCII, Latin-1 and
        Latin Extended-A only.
        Hook points: numbers (5.3, now done: `ReadWord`'s `IsNumberWord`
        branch calls `ReadNumber`);
        unknown words (5.5): tokens/pieces with g2prUnknown, empty phonemes as
        misaki with fallback=None; chunking (5.6): the per-token output.
        Gaps: misaki's markdown overrides (`[word](/IPA/)`, `[w](+1)`,
        `[n](#flags#)`) are not parsed (brackets read as text); numeric_if_needed
        (Unicode digits) not ported; merges inside one word are capped at 32
        sub-tokens (`csMaxMergedPieces`; misaki's loop is cubic, the longest key
        in its four files has 13; a pathological 8000-sub-token word takes
        3.4 s); a run of 2+ spaces keeps only the first character, like spaCy.
        Speed: ~0.2 ms per 20-word sentence on this VM.
        Tests (`TestNeuralG2P`, 15): dictionary parse/lookup, missing-file
        error, voice -> accent, ApplyG2PStress, sub-token regex, US and GB stem
        rules, stress demotion, acronyms / dotted abbreviations / shouting
        sentences, next-word context, POS rules, per-token resolutions, the
        fixture agreement (US, GB) and the real-dictionary load (ignored when
        misaki's data/ is missing; `NEURAL_MISAKI_DATA_DIR`).
        Fixture agreement (both accents identical in shape; numbers, currency,
        percent groups and every token with a digit or money sign excluded =
        5.3; tokens misaki left unknown excluded = 5.5; POS-keyed tokens
        counted apart):
        | group | tokens US | tokens GB | sentences |
        | plain | 203/203 | 203/203 | 25/25 |
        | punctuation | 212/212 | 212/212 | 26/26 |
        | capitals_acronyms | 182/183 | 182/183 | 24/26 |
        | possessives | 109/109 | 109/109 | 20/20 |
        | contractions | 115/115 | 115/115 | 20/20 |
        | hyphenated | 99/99 | 98/98 | 20/20 |
        | suffix_stems | 178/178 | 177/177 | 30/30 |
        | heteronyms | 216/216 | 216/216 | 36/38 |
        | unknown_and_names | 85/85 | 85/85 | 24/24 |
        | long_mixed | 542/543 | 542/543 | 0/6 (numbers) |
        | total | 1941/1943 (99.9%) | 1939/1941 (99.9%) | 225/235 |
        Asserted: 100% for every group except capitals_acronyms and long_mixed
        (>= 99%). Remaining mismatch classes (same in US and GB): (1) `NEAR` in
        the all-caps "THE END IS NEAR": spaCy tagged it NNP and misaki spelled
        it; the port reads a sentence whose words are all upper case as words.
        (2) `the` before a number in long_mixed ("the 1st of June", sentence
        296): misaki's context comes from the number reading (5.3); the port
        skips the unread number and sees "of". The POS-keyed mismatches are listed under 5.4.
        Suites after the commit: cpu, avx2, opencl all `TESTS=3259 ERRORS=0
        FAILURES=0 IGNORED=4` (was 3244).
  - [x] 5.2 Dictionaries: misaki's `us_gold.json` + `us_silver.json` (US) and
        `gb_gold.json` + `gb_silver.json` (GB), Apache-2.0, from misaki's GitHub
        repository (`misaki/data/`). The USER downloads them, like the model: by
        default the program looks in `<model-dir>/misaki/`; `--dict-dir <folder>`
        overrides. Never downloaded by the program, never committed to the repo.
        Clear error naming the missing file. Dictionary picked from the voice:
        `a*` = US, `b*` = GB; other voices raise until their language has a G2P.
        Load cost matters (~183k entries per accent): parse once at startup,
        hash lookup (reuse an existing string hash map; search first).
        Done (commit 51b1c903), library side: `TNeuralG2PDictionary`
        (neuralg2p.pas:99) `LoadFromFile` / `LoadFromString`, `IndexOf`,
        `IndexOfGrown` (misaki grow_dictionary: a lowercase key also answers
        its Capitalized form and vice versa, computed at lookup time instead
        of doubling the table), `TagReading` (POS-keyed values incl. "None"
        and null), `DefaultReading`, `KeyAt`. Search result: no public string
        hash map in the repo (neuralhftokenizer's merge table and
        neuralsafetensors' name table are private FNV tables), so the unit has
        its own FNV-1a open-addressing index. Parsing bypasses TJSONObject
        entirely (no 255-byte key limit; misaki's longest key is 45 bytes):
        a dedicated scanner keeps keys and readings as byte spans inside the
        file text (JSON escapes decoded in place; a repeated key keeps the last
        value, as Python). Missing files raise `ENeuralG2PError` naming every
        missing path and where to get them. Measured on the real files
        (`x` venv, misaki 0.9.4 data/, release -O2, this VM, 2 runs each):
        US 90,201 gold + 93,361 silver entries in 82-184 ms, +16.5 MB RSS
        (peak +19 MB); GB 87,352 + 109,766 in 92-131 ms, +17.7 MB RSS (peak
        +21 MB); the suite's load test saw 67-74 ms. Entry counts equal
        notes.json's `full_entries`. NOT done here: the `<model-dir>/misaki/`
        default and `--dict-dir` belong to the example (5.8/5.9); the
        constructor takes the folder. The example part is done in 5.8.
  - [x] 5.3 Text normalization, matching misaki and no more: cardinals,
        ordinals (`3rd`), years (`1984`), decimals, currency (`$`, `£`, `€`),
        `%`. Small English number-to-words port (misaki uses `num2words`).
        Done (commits 71db06b4 fixture, db271fb1 code). API in neuralg2p.pas:
        `EnglishCardinalWords` / `EnglishOrdinalWords` / `EnglishYearWords` /
        `EnglishDecimalWords` (:281-288), num2words 0.5.14's exact English
        strings ("one hundred and one", "twenty-first", "ten sixty-six",
        "nineteen oh-five", "two thousand and one", "2.0" -> "two"), computed
        on digit strings up to num2words' 10^306 limit; they stay in
        neuralg2p because the English G2P is their only consumer.
        `TNeuralEnglishG2P.ReadNumber` (:2779) ports get_number: ordinals,
        bare 4-digit years (not after a currency sign), digits glued to
        letters (A380 three eighty, A305 three O five, 555-1234), version
        numbers, minus only at a word head, commas removed, leading zeros
        dropped, unstressed "point", `$ £ €` with at most 2 decimals (dollars
        and cents, pence never plural, zero part dropped), -s/-ed/-ing
        (1990s, 80s), num2words' "and" dropped. Currency follows retokenize
        (`BuildPiecesAndGroups`): the last piece of a CD token after the sign,
        unless the next token is CD, so "$12 million" puts "dollars" after
        million (append_currency); ¥ reads nothing. The tagger now tags
        number tokens and cardinal words CD; g2prNumberPending became
        g2prNumber. Two tagger fixes found by the new group: A380-like
        pieces keep the token's tag (only hyphenated words are tagged piece
        by piece), "A" before % & + @ is NNP.
        Fixture: new group numbers_more (23 sentences, ids 297-319, earlier
        output byte-identical, subsets regrown under the same rule).
        Agreement, US = GB (non-POS tokens; sentences): numbers 213/213,
        32/32; currency 138/138, 20/20; percent 54/54, 9/10; numbers_more
        188/188, 23/23; long_mixed 580/580, 5/6; capitals_acronyms 182/183;
        total US 2572/2573, GB 2570/2571 tokens, 314/320 sentences. Asserted
        100% for every group but capitals_acronyms (99%). Every remaining
        mismatch is an earlier class: NEAR (5.1) and the POS-keyed tokens
        (5.4: "content" in percent #228, "minute" in long_mixed #292).
        Deviations, only where misaki/num2words would crash or round: 10^306
        and above reads digit by digit; a number word missing from the
        dictionary is skipped; decimals read their written digits (num2words
        reads the float64, which differs past 15 significant digits).
        Not done: number flags from markdown (`[n](#&a#)`, no markdown
        parser), Unicode digits (numeric_if_needed). Tests:
        TestEnglishNumberWords, TestNumberReading, fixture agreement now
        counts every digit/money token. Suites: cpu, avx2, opencl
        `TESTS=3261 ERRORS=0 FAILURES=0 IGNORED=4`.
  - [x] 5.4 Words spelled the same but pronounced differently (790 entries keyed
        by part of speech, e.g. `read`, `record`): use the dictionary's DEFAULT
        reading (user asked me to pick, 2026-10-13). The 5.7 fixture reports how
        often this differs from misaki's spaCy choice; context rules (after "to"
        -> verb, after "the"/"a" -> noun) only if that rate is noticeable.
        Done (commit 51b1c903) with context rules, since DEFAULT disagreed on
        30%: `TNeuralEnglishG2P.TagRawTokens` / `TagWord`, tables at
        neuralg2p.pas:309 (`cPreviousWordCues`), `cPreviousSuffixCues`,
        `cObjectPronouns`, `cDeterminers`, `cConjunctions`,
        `cThatComplementNext`, `cClosedClassTags`. Order: `@` in a word -> ADD;
        ALL-CAPS (2+ letters) -> NNP unless the whole sentence is upper case;
        closed class (a/an/the DT, `A` alone before punctuation NNP, I PRP,
        to TO, in/by/vs IN, be/have/do forms VB*; a contraction tags as its
        first word, that'll -> that); `used` VBD before "to"; `that` IN when
        a determiner or subject pronoun follows, else DT; Capitalized inside a
        sentence -> NNP; previous word: to/modals/do/not/please -> VB,
        I/you/we/they -> VBP, he/she/it -> VBD (VBZ for an -s word),
        had/has/have/was/were/been -> VBN, is/are/am/be/very/so/too/stay -> JJ,
        determiners/possessives/prepositions -> NN, `'ll`/`n't`/`'d` -> VB,
        `'ve` -> VBN; next word: me/him/them/us -> VB; a determiner after a
        clause start or and/or/but -> VB; before punctuation -> NN; -ed VBD,
        -ing VBG; otherwise no tag (DEFAULT). Pieces of a hyphenated word take
        their own closed-class tag (spaCy tags them one by one), so the `a` in
        one-of-a-kind reads ɐ.
        Result on the fixture's 158 POS-keyed tokens (all groups): 153/158
        (96.8%) in US and GB, vs 111/158 (70.3%) reading DEFAULT; asserted
        >= 95%. Heteronym group 74/76. Remaining 5 (US = GB): "LOL, that was
        funny" (spaCy WDT, port DT); "Fat content changed" (spaCy NN after an
        adjective; the port has no adjective list -> DEFAULT); "the change is minute" (port JJ -> mInˈut, the right
        English; spaCy NN -> DEFAULT mˈɪnət); "the minute hand" (spaCy JJ ->
        mInˈut; port NN -> DEFAULT mˈɪnət, again the right English); "Have you
        read it?" (spaCy VB -> ɹˈid; port VBP -> ɹˈɛd, the right English).
        Three of the five are spaCy errors the port does not copy.
  - [x] 5.5 Unknown words, in this order: (1) misaki's own rules (suffixes,
        possessives, hyphen parts); (2) SPLIT the word into dictionary words by
        dynamic programming, fewest pieces, each piece at least 2 letters
        ("sunflowerfield" -> sun + flower + field), join their phonemes; only
        the first piece keeps primary stress (ˈ), later primary stresses become
        secondary (ˌ); (3) SPELL letter by letter (the dictionaries' own letter
        entries: `A` -> `ˈA`, `B` -> `bˈi`) ONLY for unknown ALL-CAPS strings and
        for leftover single characters the split cannot cover. Report which
        words took route 2 or 3. Names may split oddly; that is accepted.
        Done (commit 95cb6312). Rule as implemented, after misaki's own rules
        fail: `TNeuralEnglishG2P.ReadUnknownWord` (neuralg2p.pas:300, body
        :3273). (a) Letters folded to ASCII: NFKC, apostrophes dropped,
        Latin-1 / Latin Extended-A accents stripped with neuralhftokenizer's
        `StripAccentBaseCP` (now exported, :480); any other character (ł, ø,
        æ, digits, symbols) leaves the word unknown. Folding is a table
        lookup, so it is safe mechanically, but it can pick the wrong word:
        résumé -> resume reads the verb (ɹəzˈum). (b) An ALL-CAPS string of 2+
        letters is spelled as misaki spells acronyms (all secondary, last
        primary); misaki already spells ASCII ALL-CAPS words itself, so in
        practice this fires after folding (ÉCOLE -> E+C+O+L+E). (c) Any other word,
        lowercased, at most `csG2PMaxSplitLetters` = 64 letters (:72; longer
        stays unknown), is covered by dynamic programming
        (`SplitIntoWords`, :3154): pieces are dictionary words of 2+ letters
        that contain a vowel letter a e i o u y (this keeps out abbreviation
        keys such as mr, dr, st, ft, which read as whole words), or single
        letters spelled with their gold letter entry. Cost: fewest spelled
        letters, then fewest pieces, then the longest first piece. So a full
        cover always exists for ASCII letters; "leftover single characters"
        = the spelled letters of the best cover, and a word with no
        dictionary piece is spelled letter by letter (Qwrtz, Hrrm, shh,
        Bpsa) — the literal reading of the rule; kept (user decision
        2026-10-14: spell, not silent). A piece reads its DEFAULT (POS-free) reading,
        gold before silver, the exact key before its Capitalized form
        (misaki grow_dictionary), null entries skipped. Readings are joined
        without spaces; only the first piece keeps ˈ, later ˈ become ˌ. A
        Capitalized or mixed-case word then gets misaki's capital stress
        (ApplyG2PStress 0.5, which only adds a mark when the result has
        none). Fewest pieces means sunflowerfield -> sunflower+field (2
        pieces), not sun+flower+field. DP cost: at most 64 start positions
        x the longest dictionary key (45 bytes) x 4 hash probes, no string
        built per probe (`TNeuralG2PDictionary.IndexOfBytes` :152,
        `MaxKeyLength` :165). (d) Multi-piece words (`ResolveUnknownRuns`,
        :3636): after misaki's merge loop, the run of adjacent letter pieces
        around an unread piece (taking whole any misaki merge made only of
        letter pieces) is read as one word, so camelCase WiFi -> wifi,
        PyTorch -> pytorch, possessive Schuler's -> schuler + misaki's 's,
        blorp-zing -> blorp + misaki's zing. A clitic piece after an
        apostrophe piece (I'd've, GB that'll) stays unread as in misaki.
        The word before a now-read word sees its first sound (misaki's
        right-to-left context). Report: new kinds `g2prUnknownSplit` /
        `g2prUnknownSpelled` (:90; `G2PResolutionName`: "unknown word
        split" / "unknown word spelled"); a cover with both words and
        spelled letters has both; `TNeuralG2PToken.Pieces` (:106) holds the
        pieces, '+'-joined, spelled letters upper case ("pa+U+lo").
        Tests (3 new, 20 in TestNeuralG2P): TestUnknownWords (pieces,
        one primary, kinds, folding, Wrocław unknown, camelCase, possessive,
        clitic, context "the applegarden" -> ði, the 64/65-letter cap),
        TestUnknownWordReport (prints the table below), and
        TestUnknownWordRealDictionaries (full misaki files, ignored when
        missing): with the full files only the 4 folded words differ —
        Café -> cafe kæfˈA (GB kˈafA), naïve -> naive nɑˈiv (GB nIˈiːv),
        résumé -> resume ɹəzˈum (GB ɹɪzjˈuːm), Raphaël -> raphael ɹˈæfiəl
        (GB ɹˈafAəl); the subset lacks cafe/naive/resume/raphael because
        they are not substrings of the accented fixture words. Every other
        row is identical. Fixture agreement: tokens misaki read are
        unchanged except 8 it read only in part, now read whole and not
        counted (blorp-zing, Mia's, Nguyen's, PyTorch, Schuler's,
        usr/local/bin, WiFi, www.example.org/page; asserted as a list) and
        one context change not counted ("got" before "hashtagged" in #293:
        misaki's clause-end "None" reading ɡˈɑt -> ɡɑt). Totals US
        2563/2564, GB 2561/2562 (same mismatches as before: NEAR + the 5.4
        POS cases); sentence agreement drops to 270/320 (US) and 269/320
        (GB) because sentences with unknown words now have phonemes where
        misaki has none. The fixture's `*` and `#` stay as before (not
        letters). Suites: cpu, avx2, opencl `TESTS=3264 ERRORS=0
        FAILURES=0 IGNORED=4`.
        Every fixture word misaki left unknown (or read in part), dictionary
        subset; pieces -> phonemes (US | GB; ɾ already mapped to T):
        | word | US | GB |
        | --- | --- | --- |
        | usr/local/bin | us+R -> ˌʌsˌɑɹ lˈOkəl bˈɪn | us+R -> ˌʌsˌɑː lˈQkᵊl bˈɪn |
        | www.example.org/page | W+W+W -> dˈʌbᵊljudˌʌbᵊljudˌʌbᵊlju ɪɡzˈæmpəl ˈɔɹɡ pˈAʤ | W+W+W, or+G -> dˈʌbᵊljuːdˌʌbᵊljuːdˌʌbᵊljuː ɪɡzˈɑːmpᵊl ɔːʤˌiː pˈAʤ |
        | PyTorch | P+Y+torch -> pˈiwˌItˌɔɹʧ | P+Y+torch -> pˈiːwˌItˌɔːʧ |
        | WiFi | W+if+I -> dˈʌbᵊljuɪfˌI | W+if+I -> dˈʌbᵊljuːɪfˌI |
        | Schuler's | S+C+H+U+ler -> ˈɛssˌiˌAʧjˌulˌɜɹs | S+C+H+U+ler -> ˈɛssˌiːˌAʧjˌuːlˌɜːs |
        | Nguyen's | N+guy+en -> ˈɛnɡˌIˌɛns | N+guy+en -> ˈɛnɡˌIˌɛns |
        | Mia's | mi+A -> mˈiˌAs | M+I+A -> ˈɛmˌIˌAs |
        | blorp-zing | blo+R+P -> blˈOˌɑɹpˌizˌɪŋ | blo+R+P -> blˈQˌɑːpˌiːzˌɪŋ |
        | blogged | B+logged -> bˈilˌɔɡd | blog+G+E+D -> blˈɒɡʤˌiːˌiːdˌiː |
        | livestreamed | livest+reamed -> lˈIvstɹˌimd | livest+reamed -> lˈIvstɹˌiːmd |
        | livestreaming | livest+reaming -> lˈIvstɹˌimɪŋ | livest+reaming -> lˈIvstɹˌiːmɪŋ |
        | photoshopped | photos+hopped -> fˈOTOzhˌɑpt | photo+shopped -> fˈQtQʃˌɒpt |
        | hashtagged | hash+tagged -> hˈæʃtˌæɡd | hashtag+G+E+D -> hˈaʃtaɡʤˌiːˌiːdˌiː |
        | facetimed | face+timed -> fˈAstˌImd | face+timed -> fˈAstˌImd |
        | smurfs | S+M+urf+S -> ˈɛsˌɛmˌɜɹfˌɛs | S+M+urf+S -> ˈɛsˌɛmˌɜːfˌɛs |
        | photocopied | photo+co+pied -> fˈOTOkˌOpˌId | photo+co+pied -> fˈQtQkˌQpˌId |
        | vlogged | V+logged -> vˈilˌɔɡd | vlog+G+E+D -> vlˈɒɡʤˌiːˌiːdˌiː |
        | geotagged | geotag+G+E+D -> ʤˈiOtˌæɡʤˌiˌidˌi | geotag+G+E+D -> ʤˈiːQtaɡʤˌiːˌiːdˌiː |
        | deduped | D+E+duped -> dˈiˌidˌupt | D+E+duped -> dˈiːˌiːdjˌuːpt |
        | sunflowerfield | sunflower+field -> sˈʌnflˌWəɹfˌild | sunflower+field -> sˈʌnflˌWəfˌiːld |
        | keyboardstand | keyboard+stand -> kˈibˌɔɹdstˌænd | keyboard+stand -> kˈiːbɔːdstˌand |
        | rainbowbridge | rainbow+bridge -> ɹˈAnbˌObɹˌɪʤ | rainbow+bridge -> ɹˈAnbQbɹˌɪʤ |
        | moonlightgarden | moonlight+garden -> mˈunlˌItɡˌɑɹdən | moonlight+garden -> mˈuːnlItɡˌɑːdᵊn |
        | Joao | J+O+A+O -> ʤˈAˌOˌAˌO | jo+A+O -> ʤˈQˌAˌQ |
        | Paulo | pa+U+lo -> pˈɑjˌulˌO | pa+U+lo -> pˈɑːjˌuːlˌQ |
        | Schuler | S+C+H+U+ler -> ˈɛssˌiˌAʧjˌulˌɜɹ | S+C+H+U+ler -> ˈɛssˌiːˌAʧjˌuːlˌɜː |
        | Nguyen | N+guy+en -> ˈɛnɡˌIˌɛn | N+guy+en -> ˈɛnɡˌIˌɛn |
        | Xochitl | X+O+chit+L -> ˈɛksˌOʧˌɪtˌɛl | X+och+it+L -> ˈɛksˌɒkɪtˌɛl |
        | Siobhan | si+ob+han -> sˈiˌɑbhˌæn | si+ob+han -> sˈiːˌɒbhˌan |
        | Okonkwo | ok+on+K+W+O -> ˌOkˈAˌɔnkˌAdˌʌbᵊljuˌO | ok+on+K+wo -> ˌQkˈAˌɒnkˌAwˌQ |
        | blorptastic | blo+R+P+ta+S+tic -> blˈOˌɑɹpˌitˌɑˌɛstˌɪk | blo+R+P+ta+S+tic -> blˈQˌɑːpˌiːtˌɑːˌɛstˌɪk |
        | zingwhizzle | zing+whizzle -> zˈɪŋwˌɪzᵊl | zing+whizzle -> zˈɪŋwˌɪzᵊl |
        | Llanfairpwllgwyngyll | L+lan+fair+P+W+L+L+G+W+Y+N+G+Y+L+L -> ˈɛllˌænfˌɛɹpˌidˌʌbᵊljuˌɛlˌɛlʤˌidˌʌbᵊljuwˌIˌɛnʤˌiwˌIˌɛlˌɛl | L+lan+fair+P+W+L+L+G+W+Y+N+G+Y+L+L -> ˈɛllˌanfˌɛːpˌiːdˌʌbᵊljuːˌɛlˌɛlʤˌiːdˌʌbᵊljuːwˌIˌɛnʤˌiːwˌIˌɛlˌɛl |
        | Zbigniew | Z+big+N+I+ew -> zˈibˌɪɡˌɛnˌIjˌɪw | Z+big+N+I+ew -> zˈiːbˌɪɡˌɛnˌIjˌɪw |
        | Brzezinski | B+R+ze+zin+ski -> bˈiˌɑɹzˌizˌɪnskˌi | B+R+ze+zin+ski -> bˈiːˌɑːzˌiːzˌɪnskˌiː |
        | Dmitri | D+mi+T+R+I -> dˈimˌitˌiˌɑɹˌI | D+M+it+R+I -> dˈiːˌɛmɪtˌɑːˌI |
        | Shostakovich | sho+sta+K+O+V+I+C+H -> ʃˈOstˌɑkˌAˌOvˌiˌIsˌiˌAʧ | sho+sta+K+O+V+I+C+H -> ʃˈQstˌɑːkˌAˌQvˌiːˌIsˌiːˌAʧ |
        | Qwrtz | Q+W+R+T+Z -> kjˈudˌʌbᵊljuˌɑɹtˌizˌi | Q+W+R+T+Z -> kjˈuːdˌʌbᵊljuːˌɑːtˌiːzˌiː |
        | Wrocław | stays unknown (ł has no ASCII base) | stays unknown |
        | Café | C+A+F+E -> sˈiˌAˌɛfˌi | C+A+F+E -> sˈiːˌAˌɛfˌiː |
        | naïve | na+I+V+E -> nˈʌˌIvˌiˌi | na+I+V+E -> nˈʌˌIvˌiːˌiː |
        | résumé | re+sum+E -> ɹˌAsˌʌmˌi | res+um+E -> ɹˈiːzˌʌmˌiː |
        | Okafor | oka+for -> ˈOkəfɔɹ | oka+for -> ˈQkəfɔː |
        | Tanaka | ta+na+ka -> tˈɑnˌʌkˌɑ | tana+ka -> tˈɑːnəkˌɑː |
        | Hrrm | H+R+R+M -> ˈAʧˌɑɹˌɑɹˌɛm | H+R+R+M -> ˈAʧˌɑːˌɑːˌɛm |
        | shh | S+H+H -> ˈɛsˌAʧˌAʧ | S+H+H -> ˈɛsˌAʧˌAʧ |
        | firetruckladder | fire+truck+ladder -> fˈIəɹtɹˌʌklˌædəɹ | fire+truck+ladder -> fˈIətɹˌʌklˌadə |
        | Greenhousegas | greenhouse+gas -> ɡɹˈinhˌWsɡˌæs | greenhouse+gas -> ɡɹˈiːnhWsɡˌas |
        | snowmanhat | snowman+hat -> snˈOmˌænhˌæt | snowman+hat -> snˈQmanhˌat |
        | Bpsa | B+P+S+A -> bˈipˌiˌɛsˌA | B+P+S+A -> bˈiːpˌiːˌɛsˌA |
        | joaopauloschuler | J+O+A+opa+U+lo+S+C+H+U+ler -> ʤˈAˌOˌAˌOpəjˌulˌOˌɛssˌiˌAʧjˌulˌɜɹ | jo+A+opa+U+lo+S+C+H+U+ler -> ʤˈQˌAˌQpəjˌuːlˌQˌɛssˌiːˌAʧjˌuːlˌɜː |
        | Kokoro | K+ok+or+O -> kˈAˌOkˌAɔɹˌO | K+O+koro -> kˈAˌQkˌɔːɹQ |
        | Raphaël | rap+ha+el -> ɹˈæphˌɑˌɛl | rap+ha+el -> ɹˈaphˌɑːˌɛl |
        | Zoë | Z+O+E -> zˈiˌOˌi | Z+O+E -> zˈiːˌQˌiː |
        | tokenizer | tokenize+R -> tˈOkənˌIzˌɑɹ | tokenize+R -> tˈQkənIzˌɑː |
        | hummed | (misaki reads it) | hum+med -> hˈʌmmˌɛd |
        | refactoring | (misaki reads it) | refactor+ing -> ɹiːfˈaktəˌɪŋ |
        Seen in the table, for the user to judge: words with a doubled
        consonant before -ed come out as spelled letters (blogged B+logged
        US, blog+G+E+D GB; hashtagged, vlogged, geotagged), because pieces
        are dictionary words only and -ed is not one; a doubled-consonant
        -ed stem rule (not in misaki) would fix that group. Names mostly
        spell (Joao J+O+A+O, Schuler S+C+H+U+ler). Not done: no stem rules
        on pieces, no letter-to-sound rules, no folding beyond Latin-1 /
        Latin Extended-A.
        Done (user-approved follow-up): the doubled-consonant -ed stem rule.
        `StemEd` gets a third branch after misaki's two: a word of 6+ letters
        ending in `([bcdgklmnprstvxz])\1ed` or `cked` whose word[:-3] is known
        reads as that stem + `SuffixEd` (stem_ing's branch, shared through
        `EndsWithDoubledConsonant`). Fixture words leaving the unknown routes
        (US | GB): blogged blˈɑɡd | blˈɒɡd, vlogged vlˈɑɡd | vlˈɒɡd,
        hashtagged hˈæʃtˌæɡd | hˈaʃtaɡd, geotagged ʤˈiOtˌæɡd | ʤˈiːQtaɡd,
        photoshopped fˈOTəʃˌɑpt | fˈQtQʃɒpt (grown key `Photoshop`), GB hummed
        hˈʌmd.
        No counted fixture token changes; the agreement test treats the word
        before such a token as context, as it does for the unknown routes.
  - [x] 5.6 Long text (also closes 3.1's chunking): split at sentence
        punctuation into chunks of at most 510 tokens, synthesize each, join the
        audio into one WAV (as Kokoro's own pipeline does).
        Done. Chunking lives in neuralg2p (guide rule 4: it reads only
        `TNeuralG2PToken` fields, and the limit is a parameter, so a later TTS
        model reuses it): `SplitG2PTokensIntoChunks` (neuralg2p.pas:337) ports
        KPipeline.en_tokenize + waterfall_last exactly (code-point counts,
        the pending trailing space counted, cut after the last `! . ? …`, then
        `: ;`, then `, —` whose cut leaves the rest within the limit, a `)` or
        `”` right after the cut goes with the first chunk, no punctuation =
        cut before the overflowing token, an empty chunk when the first token
        alone is over the limit); `TNeuralEnglishG2P.PhonemizeInChunks`
        (:314) ports KPipeline.__call__ for English: `re.split('\n+',
        text.strip())` into paragraphs (a blank part keeps its index), G2P per
        paragraph, empty chunks dropped, a chunk over the limit cut to the
        limit in code points (`IsTruncated`; KPipeline's `ps[:510]`); the
        chunk (`TNeuralG2PChunk`, :115) keeps its tokens.
        `DescribeG2PUnknownWords` (:342) prints "word: routes (pieces) ->
        phonemes" for split / spelled / unknown tokens. KPipeline inserts no
        silence: it yields one audio per chunk and `python -m kokoro` writes
        them back to back, so `SynthesizeText` joins them with no gap.
        TNNetKokoro (neuralpretrained.pas): `KokoroMaxChunkPhonemes` = 510
        (:5383); `MaxChunkPhonemes(Voice)` (:5799) = min(510,
        max_position_embeddings - 2, voice rows), 510 on the real checkpoint,
        30 on the tiny fixture; `EnglishG2PForVoice(Voice)` (:5802) owns one
        `TNeuralEnglishG2P`, accent from `EnglishG2PAccentFromVoice(Voice.Name)`,
        re-created only when the accent changes (the new one is loaded before
        the old is freed, so a failed load keeps the old); property
        `EnglishDictionaryDir` (a new folder drops the G2P), `EnglishG2P`;
        `SynthesizeText(Text, Voice, Speed, Audio, out Report)` (:5805): each
        chunk through `SynthesizePhonemes`, so each takes its own voice row
        (pack[len(ps)-1]) and builds stage nets for that chunk only; raises
        when no chunk has phonemes; `SynthesizeTextToWav` (:5808).
        `TKokoroTextReport` (:5449) = per chunk the G2P chunk, the symbols
        outside the vocab and FirstSample/SampleCount;
        `DescribeKokoroTextReport` (:5483) prints it for `/show` (5.9);
        `KokoroEnglishDictionaryDir(SnapshotDir)` (:5479) = `<dir>/misaki` for
        the example and a future server. `SynthesizePhonemes` unchanged as
        the expert path (now counts code points with `CodePointCount`).
        examples/KokoroTTS command line unchanged (5.8/5.9); lazbuild -B
        Default, Debug, Release build.
        Chunk fixture (long_mixed, 6 sentences, US and GB): misaki's own
        tokens through `SplitG2PTokensIntoChunks` give the fixture's chunks
        exactly (text and phonemes, 12/12 sentences). The port's own tokens
        give the same cuts in 5 of 6 per accent; #294 differs (US and GB): the
        port reads Qwrtz and moonlightgarden (5.5), which misaki left empty,
        so the first chunk would pass 510 and the cut moves one sentence
        earlier ("... since dawn." / rest). Chunk phonemes differ in #291,
        #293, #295 (unknown words now read) and #292 ("minute", 5.4).
        Tests: TestChunkSplitRules, TestChunksMatchKPipelineUS/GB
        (TestNeuralG2P), TestKokoroTinySynthesizeText (one chunk =
        SynthesizePhonemes of the G2P phonemes, bit-equal; "The quick brown
        fox jumps over the lazy dog.\n\nThe sunflowerfield." = 3 chunks of
        27/20/19 phonemes = the per-chunk audio joined, bit-equal; the report
        names sunflowerfield split sunflower+field),
        TestKokoroEnglishG2PAccentSwitch (a* -> a* same object, a* -> b*
        new, jf_* raises and keeps the G2P, no folder raises).
  - [x] 5.7 Tests: a `tools/` script runs misaki ONCE on a few hundred varied
        English sentences (numbers, currency, acronyms, contractions,
        possessives, hyphens, heteronyms, unknown words) for US and GB and writes
        a fixture of sentence -> expected phonemes. Pascal tests read only the
        fixture and report the phoneme agreement rate; unit tests for number
        words, sub-word splitting and spelling. Test dictionaries: a small
        committed subset (or the fixture's own word list), never the full files.
        Fixture part DONE 2026-10-13: `tools/make_g2p_english_fixture.py`
        writes `tests/fixtures/g2p_english/` (sentences.json 297 sentences in
        13 groups, US + GB, per-token output; 4 dictionary subsets in misaki's
        format, 101.5 KB total, verified to reproduce misaki exactly;
        notes.json with counts, en.py behaviours and quirks). misaki left
        62 (US) / 64 (GB) of 2605 tokens empty. Forcing DEFAULT changes 47 of
        158 POS-keyed tokens (30 of 76 in the heteronym group), so 5.4's
        context rules look worth doing. Pascal tests come with 5.1-5.6.
        Pascal tests DONE with 5.1-5.6: fixture agreement per group, US and
        GB (`TestNeuralG2P`, rates printed and asserted), number words
        (`TestEnglishNumberWords`, `TestNumberReading`), sub-word splitting
        and spelling (`TestUnknownWords`, `TestUnknownWordReport`), chunks
        (`TestChunkSplitRules`, `TestChunksMatchKPipelineUS/GB`).
  - [x] 5.8 examples/KokoroTTS: the model folder becomes the first POSITIONAL
        argument, as in ChatTerminal (`KokoroTTS <model-dir> [options]`;
        `--model-dir` removed). Text input by default; `--phonemes "<IPA>"` stays
        as an expert override.
        Done (with 5.9, same commits). `KokoroTTS <model-dir> [options]`; the
        first argument without a leading `-` is the folder, a second one is
        an error, `--model-dir` is rejected with a hint. `--dict-dir`
        overrides the default `<model-dir>/misaki`
        (`KokoroEnglishDictionaryDir`). Text needs the dictionaries, so the
        program loads the G2P at startup (`TKokoroSession.CheckTextInput`); a
        missing file stops it with exit code 1, the G2P's list of missing
        paths and one line naming misaki/data/{us,gb}_{gold,silver}.json on
        github.com/hexgrad/misaki, `<model-dir>/misaki` and `--dict-dir`.
        `--phonemes` skips the G2P (no dictionaries needed; non-English
        voices work there). Kept: `--voice`, `--speed`, `--serial`,
        `--max-threads`, `--gpu`, `--gpu-platform`, `--gpu-device`,
        `--no-gpu-shared-kernel` (CPU stays the default, unlike ChatTerminal).
        New `KokoroVoiceFileName(SnapshotDir, VoiceName)` in neuralpretrained
        (`OpenKokoroSnapshot` uses it too).
  - [x] 5.9 REPL following ChatTerminal: REPL by default (model loaded once,
        each typed line synthesized to a WAV); `-p "text"` one-shot that never
        reads stdin; `--stats` timing to stderr; `--selftest` (argument parsing
        and REPL commands, no model). Output names from `--out <pattern>`
        (default `kokoro_%03d.wav`). Commands: `/exit`, `/help`,
        `/voice <name>`, `/speed <x>`, `/out <pattern>`, `/phonemes <IPA>`,
        `/show` (phonemes of the last line). Errors print and the REPL goes on.
        External programs (not linked libraries, so pure Pascal holds), unset
        by default: `--play-cmd "aplay {wav}"` and `--encode-cmd "lame --quiet
        {wav} {base}.mp3"` run after each WAV (`{wav}` = WAV path, `{base}` =
        path without `.wav`); `--keep-wav false` deletes the WAV after a
        successful encode. Library side (load, voice switch, text -> phonemes ->
        audio) lives in neural/, the `.lpr` keeps only the REPL loop and argument
        parsing, so a later TTS HTTP server can reuse it (as ChatServer reuses
        `neuralchatengine.pas`).
        Done. Library: new unit `neural/neuralkokorosession.pas`,
        `TKokoroSession` + `TKokoroSessionOptions` (folder, dictionary
        folder, voice, speed, threads, OpenCL). Why a unit and not methods
        of `TNNetKokoro` (guide rule 4): the session owns things the model
        holder does not know about (the snapshot folder layout for voice
        switching, the `TEasyOpenCL` device object, the current voice and
        speed, the result of the last line), as `TChatEngine` owns them for
        the chat frontends; it uses only `TNNetKokoro`'s public API.
        `Create` = `OpenKokoroSnapshot` + `TNNetKokoro.Create`, threads,
        `EnableOpenCL`, `EnglishDictionaryDir`; `CheckTextInput`;
        `SelectVoice(Name, CheckText=true)` loads `voices/<name>.pt`, then
        `EnglishG2PForVoice` (accent switch re-creates the G2P), and keeps the
        old voice on any error (missing file, non-English voice, missing
        dictionaries, a name with a path); `SpeakTextToWav(Text, File)` =
        `SynthesizeText` + `SaveVolumeToWav16`, `SpeakPhonemesToWav` =
        `SynthesizePhonemes` as a one-chunk report; `TKokoroLineResult`
        (file, `TKokoroTextReport`, phoneme count, samples, synthesis and
        write ms) and `LastLine` for `/show`. Argument parsing stays in the
        .lpr as agreed (a server can move `ParseKokoroArgs` later).
        Example: REPL by default; `-p "text"` / `--phonemes` one-shot, stdin
        never read, exit 0 or 1 (1 also when a play/encode command fails;
        `-p` with `--phonemes` is an error). `--out` pattern: `%d`, `%Nd`,
        `%0Nd` (N <= 9) = line counter, 1 for the first WAV written, +1 per
        WAV written (failed lines do not count; `/out` keeps the counter);
        `%%` = `%`; other `%` sequences rejected at parse time; a pattern
        without a counter is used as is, so -p "Hi" --out hello.wav writes
        hello.wav and REPL lines overwrite it. Commands `/exit`, `/help`,
        `/voice [name]`, `/speed [x]`, `/out [pattern]` (no argument prints
        the current value), `/phonemes <IPA>`, `/show`
        (`DescribeKokoroTextReport` of the last line); unknown commands and
        failed lines print `Error: ...` and the REPL goes on. External
        programs: `SplitCommandTemplate` splits the template once at startup
        (spaces/tabs separate, '...' and "..." group with the quotes
        removed, no escapes, "" = empty argument, unbalanced quote = parse
        error), `ExpandCommandArguments` replaces `{wav}` / `{base}` inside
        each argument in one pass (a substituted path stays one argument and
        is not re-scanned; `{base}` = path without a final .wav, any case),
        `RunExternalCommand` runs it with FPC `TProcess` (`poWaitOnExit`, no
        shell, program searched in PATH, inherits the terminal). Order: play,
        encode, then delete the WAV when `--keep-wav false` and the encode
        exited 0 (`--keep-wav false` without `--encode-cmd` is a parse
        error). A failure prints the exit code (FPC on Unix: `ExitStatus`
        after `poWaitOnExit` is the exit code, minus the wait status for a
        signal; `ExitCode` read 0 for `false`). `--stats` to stderr: load
        time + peak RSS once; per line chunks, phonemes, audio s, synthesis
        s, real-time factor (synthesis / audio), WAV write s, peak RSS.
        `--selftest`: 73 checks (positional folder, every option, -p /
        --phonemes, --out patterns, command splitting and placeholders,
        --keep-wav, every REPL command), no model.
        Test `TestKokoroSessionLinesAndVoiceSwitch` (tiny fixture copied to a
        temp folder with the G2P subsets in `misaki/`, so the default folder
        is used): line 1 = `SynthesizeText` (samples, chunks, phonemes, WAV
        size), switch to a b* copy of the voice (British G2P), line 2 at
        speed 1.5, refused switches (missing, `../af_heart`, a jf* copy) keep
        the voice and the G2P, speed 0 raises, a phoneme line, a dictionary
        folder without the files names us_gold.json.
        Smoke runs (tiny fixture, `--dict-dir tests/fixtures/g2p_english`,
        REPL fed from a here-doc): text, /show, /voice to a missing and a
        non-English voice (refused), /speed, /out, /phonemes, /bogus, /help,
        /exit; a b* copy switched accent; -p with `--out hello.wav`; encode
        with `cp {wav} '{base} copy.bin'` + `--keep-wav false` (WAV
        deleted); `false` as play and as encode (exit code 1 printed, WAV
        kept, process exit 1); a missing program; `sh -c 'kill -9 $$'`
        (signal reported); --phonemes with a jf* voice; missing dictionaries
        (exit 1). Debug build (range/overflow checks) ran the REPL smoke too.
        lazbuild -B Default, Debug, Release build.
        Suites: cpu, avx2, opencl `TESTS=3270 ERRORS=0 FAILURES=0 IGNORED=4`
        (was 3269).
        Not done: no real-checkpoint run (dev-box rule); the README's raw
        GitHub download URLs (branch `main`) were not fetched; output files
        are overwritten without a question; the child program inherits
        stdin, so a program that reads stdin would consume REPL input.
  Phase 6 — memory: stages B and C as plain Pascal code (plan agreed with the
  user 2026-10-17; route B). DONE 2026-10-17 (6.1-6.6): the 26 x "Hello"
  line runs at 724 MB peak RSS (Python kokoro 1345 MB); `--gpu` = stage A
  only. Problem: a 26 x "Hello" REPL line (one chunk,
  ~157 tokens, ~344 frames) ran out of memory on the user's machine. Measured
  on the dev VM (real checkpoint, short lines, 2 GB cgroup cap): the stage-C
  generator holds ~32 MB per frame (~45 MB at the peak) against ~0.27 MB per
  frame for the Python `kokoro` package (FP32, `torch.no_grad()`, eager
  freeing; it does not split text further than we do). Cause: every layer of
  the stage nets keeps its own buffers (per-conv im2col `FInputPrepared`,
  `FBiasOutput`, `FOutputRaw` + `FOutput`, padded `FInputCopy`), and the stage
  nets are never released in `Synthesize`. Stage B is ~0.29 MB per frame,
  the decoder ~0.44 MB per frame.
  User decisions: rewrite stages B and C (prosody F0/N predictor, decoder,
  generator) as plain Pascal code on `TNNetVolume` buffers with a few reused
  buffers, like PyTorch eager mode; stage A stays a `TNNet`; do not change
  `TNNet` much; do NOT remove working TNNet layers (`TNNetConv1DLinear`,
  `TNNetDeconv1DLinear`, per-channel `TNNetSnake` stay, with their tests);
  CPU only for now (`--gpu` then covers stage A only).
  Goal: memory per frame close to Python's ~0.27 MB (report the ratio), peak
  for the 26 x "Hello" line near Python's ~1.2 GB, audio unchanged (parity
  with the existing torch dumps at today's tolerances).
  Finding (2026-10-17, read-only check + scratch program): the conv math can
  be reused without touching neuralnetwork.pas/neuralvolume.pas: a tile im2col
  gather (new) + `TNNetVolume.DotProductsTiled` + bias `Add` matched
  `TNNetConv1DLinear` with max abs diff 0 on every Kokoro conv shape; scratch
  for the widest generator conv (128 ch, k11) is 192 KiB per thread at 32
  positions per tile, independent of length (today ~920 KB per frame per such
  layer). Reusable as is: `NeuralPerChannelMeanInvStd` + `MulChannels` /
  `AddToChannels` (instance norm with the style folded into scale and shift),
  the elementwise class procedures (LeakyRelu, Add, Mul, MulAdd, Sigmoid,
  Tanh, Sin), the neuralaudio host steps, the thread pool (`fNTL`).
  Prior art: `RunEnCodecConv` (neuralpretrained.pas) already does a host
  im2col conv; the new routines go in one reusable place so EnCodec can move
  to them later (no second copy).
  Every task: one Opus agent at a time; parity tests; the 2 GB cgroup cap
  (`tests/run_suite_capped.sh`, `systemd-run ... MemoryMax=2G`); no real-size
  forwards on the dev VM (short real lines only); before/after memory numbers;
  commit per task.
  - [x] 6.1 Plain-volume ops, reusable (not Kokoro-specific), CPU, threaded
        over time tiles with per-thread scratch allocated once (never inside
        the per-tile work): 1D conv with stride / dilation / zero padding
        (tile gather + `DotProductsTiled` + bias), pointwise conv, dense
        transposed 1D conv (GEMM + a scatter written in this code, not
        extracted from `TNNetDeconvolution`), depthwise transposed 1D conv
        (k3 s2 p1 op1, groups = C), nearest x2 upsample, reflect pad, instance
        norm + style affine, Snake per channel, LeakyReLU, residual sum,
        bidirectional LSTM keeping only h and c. Unit tests against the
        existing layers on small random cases (exact where the arithmetic
        order is the same).
        Done 2026-10-17: `neural/neuralplainops.pas` (no change to
        neuralnetwork.pas / neuralvolume.pas). `TNNetPlainConvWeights` (rows
        per op: conv [Cout][K][Cin], transposed [K][Cout][Cin], depthwise
        taps [K][C]; loaders from the layers and from torch tensors),
        `TNNetPlainLSTMWeights` (from `TNNetLSTMCell` or torch ih/hh),
        `TNNetPlainOps` (Conv1D, PointwiseConv, ConvTranspose1D,
        DepthwiseConvTranspose1D, UpsampleNearest, ReflectPad, InstanceNorm,
        Snake, LeakyReLU, BiLSTM; residual sum / x constant stay
        `TNNetVolume.Add` / `Mul`), `ComputeStyleChannelAffine`. Scratch per
        pool thread, grown at op entry and kept: patch + tile result, 1.02 MiB
        per thread at 32 positions after the widest Kokoro ops (decoder conv
        1090 -> 1024 k3 patch, ups0 512 -> 256 k20 products); BiLSTM keeps
        an input projection of 8 KiB per frame (H 256). Tests
        `TestNeuralPlainOps` (12): conv / pointwise / transposed / Snake /
        upsample / pad / LeakyReLU / style affine exact on cpu (conv and
        pointwise <= 2.4e-7 under AVX2: the layers' low-memory DotProduct
        kernel), depthwise 0, instance norm 1.2e-7, BiLSTM 6.7e-8; serial ==
        threaded (3-thread pool). Suites cpu / avx2 / opencl
        `TESTS=3282 ERRORS=0 FAILURES=0 IGNORED=4` (was 3270).
  - [x] 6.2 Stage-C decoder as plain code: its weights loaded once into
        plain volumes owned by `TNNetKokoro` (concatenated [Cout][K][Cin]
        conv rows; no owner net), the style projections reused
        (`LoadKokoroStyleFc`; split a scale/shift routine out of
        `WriteStyleIntoChannelAffine` if useful). Parity on the tiny fixture
        and the real dump (opt-in heavy test). The old net path stays until
        6.5. Measure memory per frame.
        Done 2026-10-17: `TKokoroPlainDecoder` (weights read once at
        `TNNetKokoro.Create` via `LoadTorchPlainConv`, weight_norm folded by
        `LoadTorchConv1DWeight`; norm fcs borrowed from
        `FStageCDecoderNormFc*`), `TKokoroPlainAdainResBlock` (reusable for
        stage B; conv1x1 runs before the nearest x2), `TKokoroPlainWorkspace`
        (one `TNNetPlainOps` + Work1 / Work2 / Shortcut / norm scale+shift),
        new op `TNNetPlainOps.ConcatChannels`. Entry
        `TNNetKokoro.RunStageCDecoderPlain` (same arguments as
        `RunStageCDecoder`; frees block and work buffers after). Parity: tiny
        dec_decode3 1.5e-5 abs (same as the net), plain vs net 5.7e-6 (max
        |x| 106); real dump (102 frames) dec_decode3 9.4e-5 abs, rel.
        2.7e-6, RSS 894 MB. Memory (real checkpoint, synthetic F = 56 / 112
        / 224, scratch probe): plain buffers ~24.5 KB per frame (decoder
        10.5 + workspace 10 + x 4) + 0.55 MB ops scratch, measured VmHWM
        delta 0.3 MB (reuses heap already resident); net RSS delta
        124 / 154 / 204 MB (0.48 MB per frame + ~97 MB fixed), VmHWM delta
        129 / 188 / 299 MB (1.0 MB per frame). Both paths loaded: +118.8 MB
        of plain decoder weights until 6.5. Tests
        `TestKokoroStageCDecoderPlainTinyParity`,
        `TestKokoroRealStageCDecoderPlainParity` (opt-in),
        `TestConcatChannelsMatchesLayer`. Suites cpu / avx2 / opencl
        `TESTS=3285 ERRORS=0 FAILURES=0 IGNORED=5` (was 3282 / 4).
  - [x] 6.3 Stage-C generator as plain code (noise convs, ups, AdaINResBlock1
        stacks, conv_post, then the existing exp / sin and iSTFT). Parity
        tiny + real dump; memory per frame on short real lines.
        Done 2026-10-17 (cae177eb): `TKokoroPlainGenerator`,
        `TKokoroPlainAdaINResBlock1` (weights read once at Create; adain fcs
        borrowed from `FStageCGeneratorNormFc*`; Snake alphas via
        `LoadKokoroSnakeAlphaVolume`), entry
        `TNNetKokoro.RunStageCGeneratorPlain` (same arguments as
        `RunStageCGenerator`; `ConvPostToAudio` shared with the net path).
        New op `TNNetPlainOps.NormSnakeConv1D` (prepared instance norm +
        Snake per 1024-position chunk inside the conv; Accumulate adds conv2
        into x), `PrepareInstanceNorm` / `ApplyInstanceNorm`, `MaxThreads`
        (the Kokoro plain decoder and generator now honour `MaxThreads`).
        Buffers per frame (real, stage 1 = 120F+1 x 128 = 15360 floats):
        x / x_source, resblock sum, middle resblock, conv1 output = 4 x 15360
        = 61440 floats = 240 KB per frame at the peak (stage 0: 4 x 5120);
        conv_post 2640 floats; ~2 MB constant ops scratch. Python ~0.27 MB
        per frame: 0.89x; 4 full volumes is the floor without recomputation
        (instance norm needs whole-sequence stats of x and of conv1's output).
        Parity: tiny every checkpoint as the net test (stage1 noise_res 6.1e-6
        rel., audio 1.4e-5 rel.), plain vs net audio 0.23 of 47963; real dump
        (102 frames) conv_post 3.0e-6 rel., gen_spec 1.5e-5, gen_phase
        1.4e-4 abs (test tolerance 5e-4: sin keeps conv_post's 1.8e-4 abs
        error), audio 8.4e-5 rel. Memory (real checkpoint, synthetic inputs,
        AVX2 probe, one process each, 2 GB cap): plain VmHWM delta 13.5 /
        27.0 / 54.0 / 82.7 MB at F = 56 / 112 / 224 / 344 (240 KB per frame,
        the 26 x "Hello" line fits: 16.2 s on 1 core); net VmHWM delta 441 /
        614 / 801 MB at F = 8 / 12 / 16 (45 MB per frame; RSS 34 MB per
        frame). Both paths loaded: +65.9 MB of plain generator weights until
        6.5. Tests `TestKokoroStageCGeneratorPlainTinyParity`,
        `TestKokoroRealStageCGeneratorPlainParity` (opt-in),
        `TestNormSnakeConv1DMatchesSteps`, `TestMaxThreadsCapsTheJob`. Suites
        cpu / avx2 / opencl `TESTS=3289 ERRORS=0 FAILURES=0 IGNORED=6` (was
        3285 / 5).
  - [x] 6.4 Stage B as plain code (shared bi-LSTM, F0 / N AdainResBlk1d
        branches with the depthwise-transposed upsample, projections). Parity
        tiny + real.
        Done 2026-10-17: `TKokoroPlainF0NPredictor` (shared bi-LSTM as two
        `TNNetPlainLSTMWeights` via `LoadTorchLSTM`, F0 / N branches of
        `TKokoroPlainAdainResBlock`, F0_proj / N_proj; weights read once at
        `TNNetKokoro.Create`, norm fcs borrowed from `FStageBNormFc*`; block
        widths from the shared `KokoroF0NBlockHalvesWidth`), entry
        `TNNetKokoro.RunStageBPlain` (same arguments as `RunStageB`; honours
        Parallel / MaxThreads; frees its buffers after). Buffers per frame
        (real, hidden 512): en 640 floats (freed after the bi-LSTM), bi-LSTM
        output 512, two block outputs 512 each, workspace Work1 / Work2 /
        Shortcut up to 512 / 1024 / 512, ops' bi-LSTM projection scratch
        2048 (kept): peak ~5632 floats = 22 KB per frame; measured end-of-run
        buffers 20 KB per frame + 0.27 MB constant. Memory (real checkpoint,
        synthetic d, AVX2 probe, 2 GB cap): plain VmHWM delta 0.19 MB at
        F = 56 / 112 / 224 / 344 (reuses heap already resident); net VmHWM
        delta 40 / 64 / 142 / 212 MB (0.60 MB per frame), RSS delta 0.34 MB
        per frame. Both paths loaded: +27.5 MB of plain stage-B weights until
        6.5. Parity (same tolerances as the net tests): tiny F0_pred 4.1e-6
        rel., N_pred 3.8e-6 (net 3.7e-6 / 3.0e-6), plain vs net F0 2.6e-4
        of 141, N 2.5e-6; real dump (102 frames) F0_pred 5.9e-7 rel., N_pred
        6.0e-7, every block <= 1.7e-6 rel. (net 4.9e-7 / 6.0e-7). Preview
        for 6.5, 26 x "Hello" (155 code points, 157 tokens, 415 frames,
        10.38 s of audio): stage A net (released after) -> plain B -> plain
        decoder -> harmonic spectrum -> plain generator under the 2 GB cap,
        peak RSS (VmHWM) 1265 MB, set by stage A (load 1026 MB; B / C add
        nothing to the peak), 40.6 s total on 1 core (load 14.8 s, A 5.3 s,
        B 0.4 s, decoder 1.1 s, generator 19.0 s). Tests
        `TestKokoroStageBPlainTinyParity`, `TestKokoroRealStageBPlainParity`
        (same gate as `TestKokoroRealStageBParity`: snapshot + dump present).
        Suites cpu / avx2 / opencl `TESTS=3291 ERRORS=0 FAILURES=0
        IGNORED=6` (was 3289); lazbuild -B KokoroTTS builds.
  - [x] 6.5 Switch `Synthesize` / `SynthesizeText` to the plain-code stages;
        release stage A's net after use and drop its spare weight copies
        (`FlushWeightCache` after `SetTrainable(false)`); rewrite the stage
        B / C parity tests on the new code. Full cpu / avx2 / opencl suites.
        REMOVE (neuralpretrained.pas unless noted):
        - stage B net: `BuildKokoroStageBNet`, `TNNetKokoro.LoadStageB`,
          `PrepareStageB`, `ReleaseStageB`, `RunStageB`,
          `AddKokoroAdainResBlock`, `LoadKokoroAdainResBlockWeights`, types
          `TKokoroStageBLayers`, `TKokoroAdainResBlockLayers`, fields and
          properties `StageBOwner`, `StageBNet`, `StageBLayers`,
          `StageBInput`, `StageBOwnerOnOpenCL`;
        - decoder net: `BuildKokoroStageCDecoderNet`,
          `TNNetKokoro.LoadStageCDecoder`, `PrepareStageCDecoder`,
          `ReleaseStageCDecoder`, `RunStageCDecoder`, type
          `TKokoroStageCDecoderLayers`, fields and properties
          `StageCDecoderOwner`, `StageCDecoderNet`, `StageCDecoderLayers`,
          `StageCDecoderAsr`, `StageCDecoderOwnerOnOpenCL`,
          `FStageCDecoderNormFcWeights` / `FStageCDecoderNormFcBiases`;
        - generator net: `BuildKokoroStageCGeneratorNet`,
          `AddKokoroAdaINResBlock1`, `LoadKokoroAdaINResBlock1Weights`,
          `LoadKokoroSnakeAlpha`, `TNNetKokoro.LoadStageCGenerator`,
          `PrepareStageCGenerator`, `ReleaseStageCGenerator`,
          `RunStageCGenerator`, `RunStageCGeneratorFrom`, types
          `TKokoroAdaINResBlock1Layers`, `TKokoroGeneratorStageLayers`,
          `TKokoroStageCGeneratorLayers`, fields and properties
          `StageCGeneratorOwner`, `StageCGeneratorNet`,
          `StageCGeneratorLayers`, `StageCGeneratorOwnerOnOpenCL`,
          `FStageCGeneratorNormFcWeights` / `FStageCGeneratorNormFcBiases`;
        - the stage B / C parts of `ArmStageNet` and `EnableOpenCL`, and
          `ShareOpenCLOutputs` if stage A does not need it;
        - tests (tests/TestNeuralPretrained.pas):
          `TestKokoroStageB/StageCDecoder/StageCGeneratorBorrowingRebuild`,
          `TestKokoroStageB/StageCDecoder/StageCGeneratorOpenCLParity`.
        REWRITE on the new code (same fixture dumps):
        `TestKokoroStageBTinyParity`, `TestKokoroRealStageBParity`,
        `TestKokoroStageCDecoderTinyParity`,
        `TestKokoroRealStageCDecoderParity`,
        `TestKokoroStageCGeneratorTinyParity`,
        `TestKokoroRealStageCGeneratorParity`,
        `TestKokoroRealStagesChainedToDecoder`, `TestKokoroRealEndToEnd`.
        KEEP: every TNNet layer (`TNNetConv1DLinear`, `TNNetDeconv1DLinear`,
        per-channel `TNNetSnake`, `TNNetDeMaxPool`, `TNNetPadXY`,
        `TNNetInstanceNorm`, `TNNetLSTMCell`) with their own tests and
        OpenCL kernels; stage A as a net (`BuildKokoroStageANet`,
        `RunStageA`); `WriteStyleIntoChannelAffine` (stage A uses it);
        `LoadKokoroStyleFc`; loaders and voice packs; neuralaudio host
        steps; tools/make_kokoro82m_fixture.py.
        Done 2026-10-17 (7e5929da switch, 6a9c17fd removal + tests,
        d48c6ff4 stage-A flush): `Synthesize` runs stage A (net), frees
        StageANet (`ReleaseStageA`), then plain B, decoder, harmonic
        spectrum, plain generator. The REMOVE list is gone (also
        `ArmStageNet`, folded into `PrepareStageA`); `ShareOpenCLOutputs`
        kept (stage A reads it). The plain blocks read and own their style
        fc volumes; `TKokoroPlainStage` holds `OnCheckpoint` (the decoder now
        reports dec_encode / dec_decode<i>). The plain entry points and
        properties took the old names (`RunStageB` / `RunStageCDecoder` /
        `RunStageCGenerator`, `StageB` / `StageCDecoder` /
        `StageCGenerator`). `--gpu` / `EnableOpenCL` = stage A only.
        Stage A: `FlushWeightCache` per owner layer after load: load RSS
        456 -> 442 MB; inference-only owner build skipped (load VmHWM = RSS,
        no transient peak; the bi-LSTMs come from neuralnetwork.pas).
        Real KokoroTTS, 26 x "Hello" (155 phonemes, 10.38 s audio), 2 GB
        cap, 1 core: load 1.72 s / 462 MB, synthesis 30.19 s, peak RSS
        724 MB (6.4 preview: 1265 MB). REPL, 2 lines: both written, peak
        724 MB after each. Real heavy tests (one run each, cpu):
        decoder dec_decode3 2.7e-6 rel., generator audio 8.4e-5 rel. /
        phase 1.4e-4 (tolerance 5e-4 kept), chained dec_decode3 4.0e-6,
        end to end audio max |diff| 0.102 (phases), peak 549 MB.
        Suites cpu / avx2 / opencl `TESTS=3279 ERRORS=0 FAILURES=0
        IGNORED=4` (was 3291 / 6: 12 net tests removed).
  - [x] 6.6 Measure and document: memory per frame and peak against Python
        (same short real lines under the cap, extrapolated to the 26 x
        "Hello" line); update examples/KokoroTTS/README.md (Memory section,
        `--gpu` = stage A only), examples/README.md, and the code examples:
        `SimpleKokoroTTS.lpr` (speak a long multi-sentence text to show it
        fits), `KokoroTTS.lpr` (`--gpu` help text, `--stats`), the README's
        "From your own code" snippet; lazbuild -B every mode.
        Done 2026-10-17: measured, real checkpoint, 1 core, 2 GB cap, one
        process per run; ours = KokoroTTS `-p ... --stats`, Python = kokoro
        `KModel` FP32, `torch.set_num_threads(1)`, VmHWM reset after load
        (`/proc/self/clear_refs` 5), same phonemes, same audio lengths.
        Load RSS: ours 462 MB, Python 1092 MB (1281 MB peak while loading).
        Peak RSS, ours / Python: 1 x "Hello" (56 frames) 512 / 1110 MB;
        8 x (128) 586 / 1147; 26 x (415) 724 / 1345; the SimpleKokoroTTS
        story (260 phonemes, 636 frames) 862 / 1354 (0.46x to 0.64x).
        Per frame: ours 0.48 to 1.03 MB between lines, 0.60 from 1 x to the
        story; Python 0.04 to 0.69 (uneven). A 510-phoneme chunk (~1250 frames) not measured,
        estimate ~1.2 GB for ours. REPL: a second 26 x line raised the peak
        724 -> 738 MB. Synthesis on 1 core: ours 3.8 / 8.3 / 28.8 / 44.1 s,
        Python forward 2.8 / 5.8 / 16.8 / 28.8 s. Docs: Memory section
        rewritten (table above), `--gpu` = stage A (options table, session
        options, How it works: stage A net, B / C plain code in
        neuralplainops.pas), examples/README.md entry; root README unchanged
        (still right). Code: SimpleKokoroTTS speaks `fox.wav` + a 5-sentence
        paragraph `story.wav`; KokoroTTS help for `--gpu` and `--stats`.
        Real SimpleKokoroTTS run: fox 1.95 s, story 15.90 s (one chunk),
        peak RSS 872 MB. Tiny fixture smoke: both WAVs written (the fixture
        dictionaries split or spell 6 story words). lazbuild -B Default /
        Debug / Release of both .lpi; `--selftest` OK. Script:
        scratchpad pyk/pymem.py (written for this task).
  Phase 7 — stage A memory, then generator speed (plan agreed with the
  user 2026-10-18, after a read-only diagnosis). Measured on the dev VM
  (26 x "Hello", 157 tokens, 415 frames, 1 core, 2 GB cap): the peak
  (708 MB in the probe, 724 MB in KokoroTTS) is reached during the stage-A
  compute; stage A holds 260 MB = ~98 MB fixed + ~0.96 MB per token +
  5.2e-4 * T^2 MB (one T x T score matrix per attention head). Largest
  holders: the 51 TNNetPointwiseConvLinear layers (`FOutputRaw` +
  `FBiasOutput`, 120 MB), the 10 TNNetLSTMCell (`FGradW` / `FGradB` training
  accumulators allocated when not trainable, 46 MB), the 3 TNNetConv1DLinear
  (own weight caches, 39 MB), attention 19 MB. The stage-A owner also keeps
  ~35 MB of LSTM `FGradW` for the whole session; ~89 MB of heap freed during
  load is not reused. The generator peaks 146 MB below stage A. Time
  (ours / Python, s): stage A 2.2-3.1 / 0.56, decoder ~1.0 / 0.39,
  generator 14-16 / 9.3. The generator (~254 GMAC of 1D convs) runs at
  14-18 GMAC/s against PyTorch's ~27: `TNNetVolume.DotProductsTiled` computes
  each output as its own dot product (not register-blocked).
  User rules: adding TNNet layers is fine; small changes inside TNNet layers
  are fine when required; do not remove working layers; CPU only; pure
  Pascal. One Opus agent at a time; test and commit after each task.
  - [x] 7.1 Build ALBERT's shared layer once and run it 12 times, instead of
        12 unrolled copies in the stage-A net (`neuralpretrained.pas`:
        stage-A builder and `RunStageA`). Expected: ~150 MB less at 157 tokens
        (per-token buffers ~112, attention T^2 ~13, out-projection weight
        caches ~26), ~500 MB less on a 510-phoneme chunk, build -0.3 s.
        Parity: the existing stage-A tests must still pass at their current
        tolerances. Measure the 26 x "Hello" line peak (one run).
        Done 2026-10-18 (1cf2fbbf): stage A = three nets with their own
        weight owners: front (PL-BERT embeddings), block (one ALBERT layer,
        `RunStageA` runs it 12 times feeding each output back), tail
        (bert_encoder, duration predictor, text encoder; its input carries
        the PL-BERT output, written by the host). `AddBertBlock` extracted
        from `AddBertEncoder`; `OnStageACheckpoint` reports bert_layer<i>
        for the parity tests (tolerances unchanged). Real KokoroTTS,
        26 x "Hello", 1 core, 2 GB cap: peak RSS 723 -> 557 MB (-166 MB),
        load RSS 459-462 -> 441 MB, WAV bit-identical to HEAD (max |diff|
        0). Synthesis time not comparable on the loaded VM (load average
        ~2.5): HEAD 24.7 / 36.9 s, new 32.8 / 35.0 s. Suites cpu / avx2 /
        opencl `TESTS=3279 ERRORS=0 FAILURES=0 IGNORED=4`.
  - [x] 7.2 Free what stage A holds and never reads:
        (a) TNNetLSTMCell allocates `FGradW` / `FGradB` and the backprop-only
        per-step sequences only when the layer is trainable (small change in
        `neuralnetwork.pas`; training must behave exactly as before; add a
        test that a non-trainable LSTM holds no gradient buffers);
        (b) drop the weight caches on the stage-A borrower layers the
        low-memory path never reads (out-projections, bert_encoder, the
        text-encoder convs): first try `FlushWeightCache` after
        `PrepareStageA`; if they come back, find why (`AfterWeightUpdate`)
        and make the small layer fix. Expected: ~36 MB off the load RSS and
        ~70-100 MB off the line peak. Measure load + the 26 x line (one run).
        Done 2026-10-18: TNNetLSTMCell sizes its gradient buffers and backward
        carries only when trainable, keeps two alternating per-step cache rows
        when inference-only (`SizeSequenceCaches`, re-run by a `SetTrainable`
        override; `NonWeightBytes` reports them); new test
        `TestLSTMCellInferenceOnlyBuffers`. Freeing after the build was not
        enough (owner 441 -> 435 MB, line 557 -> 548 MB): the buffers and
        weight caches were allocated while each net was still trainable,
        before `SetTrainable(false)`, and that transient set the high-water
        mark. So the stage-A builder now creates its layers inference-only:
        `AddBidirectionalLSTM(..., pTrainable)` (new default-true parameter),
        and `SetTrainable(false)` chained on bert_encoder, duration_proj and
        the 3 text-encoder convs (low memory from the start, no concatenated
        caches); `PrepareStageANet` flushes the remaining borrower caches
        (the ALBERT attention out-projection) before OpenCL is armed. Real
        KokoroTTS, 26 x "Hello", 1 core, 2 GB cap: load RSS 441 -> 348 MB,
        peak RSS 557 -> 466 MB, WAV bit-identical to HEAD. Suites cpu / avx2
        / opencl `TESTS=3280 ERRORS=0 FAILURES=0 IGNORED=4`; lazbuild -B
        Default / Debug / Release of both .lpi, `--selftest` OK.
  - [x] 7.3 Measure and decide: the 26 x "Hello" line and the SimpleKokoroTTS
        story, and if it fits under the 2 GB cap, one full 510-phoneme chunk;
        compare against the Phase 6 Python numbers. Update the README Memory
        section. Then report to the user for two decisions: (i) a lean
        inference-only pointwise layer (new TNNet layer, used only by Kokoro
        stage A, without the per-position bias copy and the separate raw
        output), (ii) investigating the ~89 MB of load heap not reused.
        Done 2026-10-18: measured at 7fe10b98 (lazbuild -B), real checkpoint,
        1 core, 2 GB cgroup cap, one process per run, KokoroTTS `--stats`.
        Load RSS 348 MB (Python 1092). Peak RSS, ours / Python: 1 x "Hello"
        (5 phonemes, 56 frames) 364 / 1110 MB; 8 x (47, 128) 392 / 1147;
        26 x (155, 415) 466 / 1345; the SimpleKokoroTTS story (260, 636)
        531 / 1354; a full chunk, 9 sentences of the fox story (506
        phonemes, one chunk, 27.70 s, 1108 frames) 648 / 1609 (0.33x to
        0.40x; Python numbers from 6.6 except the full chunk, run now with
        pymem.py on our phonemes). Estimate before the full-chunk run:
        ~0.7 GB ours, <= 1.7 GB Python. Growth, ours: peak = 355 MB + 0.27 MB
        per frame (least-squares over the 5 lines, residuals <= 6 MB); per
        token the slope falls 0.67 -> 0.69 -> 0.62 -> 0.48 MB between lines
        (residuals up to 14 MB). Python: 0.54 MB per frame story -> full
        chunk. What sets the peak: the per-frame stages (B/C; most likely
        the generator), not stage A: the peak tracks frames, not tokens, and
        its per-token slope falls at 508 tokens, which stage A (per-token
        buffers + a T^2 score matrix) cannot do (inferred from `--stats`, no
        probe). REPL, two 26 x lines: 465 -> 477 MB. SimpleKokoroTTS (fox +
        story): peak 538 MB (/usr/bin/time). Synthesis (noisy, VM shared):
        3.67 / 9.33 / 31.28 / 43.04 / 78.00 s; Python forward on the full
        chunk 57.63 s. Docs: examples/KokoroTTS/README.md Memory section and
        examples/README.md Kokoro entry.
        Pending user decisions, my view: (i) lean inference-only pointwise
        layer for stage A: no expected change to the line peak, since stage
        A no longer sets it; it would only lower stage A's own transient,
        which is already below the per-frame stages. Worth it only after the
        generator's per-frame memory is cut. (ii) the ~89 MB of load heap
        not reused (measured before 7.2): load RSS is now 348 MB against
        312 MB (327.2e6 bytes) of raw FP32 weights in kokoro-v1_0.pth plus
        17-21 MB of G2P dictionaries, so at most ~20 MB is left to find
        (~6% of load, ~3% of the full-chunk peak). Low payoff for both.
  - [x] 7.4 Generator speed: a register-blocked FP32 matrix-multiply routine
        (new routine in `neuralvolume.pas`, called from `neuralplainops`;
        asm must be PIC-safe; scalar fallback; not bit-identical, which is
        fine). Do not change `DotProductsTiled` itself. Expected: generator
        14 -> ~7-9 s, decoder -0.5 s on 1 core. Parity tests vs the current
        conv; all three suites; one timed run of the 26 x line.
        Done 2026-10-18 (de864351): `TNNetVolume.PackDotProductRows` (panels
        of 16 rows, [element][row]) + `DotProductsPacked`, AVX64 kernel
        `AVXDotProductsPacked6x16` (6 positions x 16 channels in 12 ymm
        accumulators, FMA on AVX2, mul + add on AVX; partial blocks via a
        stack block; Pascal fallback sums in element order, bit-identical
        to the scalar DotProduct). `TNNetPlainConvWeights` packs its conv /
        transposed rows at load; Conv1D (bias in registers, writes the
        output directly), PointwiseConv and ConvTranspose1D (input read in
        place) call it; default tile 36. Block shape: 6x16 and 4x24 measured
        equal; 6x16 fits 128 / 256 channels. Micro-benchmark (synthetic,
        1 core, sustained; the CPU runs ~1.5x faster for the first seconds
        after idle): convs at generator shapes 9-19 -> 24-46 GMAC/s (ups
        9 -> 29-35). Real KokoroTTS, 26 x "Hello": synthesis 22.18 s (load
        average 0.70) -> 16.28 s (1.90), peak RSS 466 / 466 MB; WAV vs HEAD
        max |diff| 0.038, correlation 0.99959 (the pipeline amplifies ulp
        changes; end-to-end vs Python is 0.056). Per stage (HelloLineProbe,
        one run each, throttled): generator 24.0 -> 11.8 s, decoder 2.0 ->
        0.7 s. Full 506-phoneme chunk 78.0 -> 43.9 s, peak 649 MB. Real
        generator parity (HEAVY=1) passes: audio rel. 8.3e-5 vs PyTorch.
        Suites cpu / avx2 / opencl `TESTS=3282 ERRORS=0 FAILURES=0
        IGNORED=4`; plain-AVX build passes TTestNeuralPlainOps; lazbuild -B
        all modes of both .lpi, `--selftest` OK. TestConvTranspose1DMatches
        Layer: exact -> csDotKernelTolerance (exact on scalar builds).
  Later (user 2026-10-13): MP3 in pure Pascal — decoder (minimp3 port, CC0)
  and/or encoder (shine port, LGPL); not scheduled.
  Not in scope: `hexgrad/Kokoro-82M-v1.1-zh` (different repo and vocab); int8.
- [ ] Mimi streaming neural-codec importer (`BuildMimiFromSafeTensors[Ex]`,
      model_type "mimi", e.g. `kyutai/mimi`) — LANDED (conv encoder/decoder +
      RoPE transformer bottleneck + semantic/acoustic split-VQ; codes match the
      HF `MimiModel` float64 oracle exactly, round-trip waveform max|diff| ~1e-12,
      `TestMimiParity`; examples/MimiCodec round-trip smoke). Open follow-ups:
  - [ ] Mimi REAL-checkpoint parity vs the downloaded `kyutai/mimi` (encode codes +
        decode waveform) — the pico fixture pins the wiring; a real run pins widths,
        the GELU/erf path at scale, and the F32-storage budget on a trained model.
  - [ ] Mimi STREAMING chunk-at-a-time encode/decode (HF `MimiConv1dPaddingCache`
        per-conv padding cache + KV-cache transformer decode via the landed SDPA
        Begin/EndIncrementalDecode) for O(1) per-frame Moshi-style inference.
- [ ] F5-TTS flow-matching TTS follow-ups (`BuildF5TTSFromSafeTensors[Ex]` + `TF5Config`
      + `ReadF5ConfigFromJSONFile` + `examples/F5TTS` LANDED, model `SWivid/F5-TTS`;
      NON-autoregressive, NON-GAN voice cloner regressing a mel via a conditional-flow-
      matching ODE through an adaLN-zero RoPE-SDPA DiT trunk conditioned in-context on a
      masked reference mel + a ConvNeXt-V2 char embed, reusing the landed DiT/FiLM/RoPE-
      SDPA blocks, no new leaf; pico parity TestF5TTSParity ~1.4e-5 < 1e-4 on the DiT
      velocity field): real `SWivid/F5-TTS` checkpoint key-mapping parity (offline/RAM-
      gated); the mel->waveform Vocos/HiFi-GAN vocoder pairing (v1 emits mel); non-default
      `rope_theta` (the SDPA RoPE base is fixed at 10000, rejected loudly); the E2-TTS
      flat-UNet variant (same flow-matching objective, simpler trunk); classifier-free-
      guidance strength sweep on the cond/uncond DiT pass.
- [ ] CLAP audio-text contrastive importer (`BuildClapFromSafeTensors[Ex]` +
      `BuildClapFromSafeTensorsWithConfig`, `TClapConfig`/`ReadClapConfigFromJSONFile`)
      + examples/ZeroShotAudioTag — LANDED. Audio-domain analogue of the CLIP
      dual-encoder (two nets, shared L2-normalized space, `exp(logit_scale_a)*cosine`),
      but genuinely distinct: the audio tower is an HTS-AT Swin transformer over a
      log-mel spectrogram (PURE REUSE of the landed Swin window machinery —
      `TNNetWindowAttention`/`TNNetGatherTokens`/`SwinBuildWindowLayout`/
      `SwinSetWindowBias`/patch-merge — under the `clap_audio_model` key spelling, plus
      a Conv2d patch-embed + token mean-pool + 2-layer `ClapProjectionLayer`), and the
      text tower is RoBERTa (built inline like `BuildBertFromSafeTensors` + BERT pooler
      + 2-layer projection). The HF `BatchNorm2d`(mel) + `reshape_mel2img` freq<->time
      transpose are folded into `ClapBatchNormMelImage` (caller supplies the
      `(time,mel,1)` image). NO new leaf layers. Reuses `ClipExtractEmbedding`/
      `ClipSimilarity` + new `ClapSimilarityMatrix`. Pico parity `< 1e-4` on BOTH
      embeddings vs the float64 HF `ClapModel` oracle (`TestClapParity`, generator
      `tools/clap_tiny_fixture.py`, committed `tests/fixtures/tiny_clap.*`).
      Follow-ups:
  - [ ] "fused" CLAP (`enable_fusion = true`, clap-htsat-FUSED): the local/global
        mel-fusion patch-embed (`mel_conv2d` + the attention-feature-fusion block) is
        rejected. Distinct windowing — own importer branch + fixture.
  - [ ] Real laion checkpoint parity + the log-mel frontend wiring: `ClapFeatureExtractor`
        produces the 1024x64 mel (depends on freq_ratio=4); wire `neuralaudio.pas`
        log-mel -> `ClapBatchNormMelImage` end-to-end and verify against a downloaded
        `clap-htsat-unfused`. (The RoBERTa BPE tokenizer is already importable.)
- [ ] AudioLDM 2 / Stable Audio latent text-to-audio (music & sound) capstone —
      text-prompt -> audio via LATENT diffusion, the audio analogue of the landed
      LatentTextToImage (PixArt/DiT + VAE) pipeline and the natural home for the
      music priority. Reuses the landed `TNNetDiffusionScheduler` samplers
      (DDIM / DPM-Solver++(2M) / Euler) and a VAE/EnCodec latent codec; the new piece
      is the importer for the audio U-Net/DiT denoiser conditioned on a text encoder
      (CLAP or T5/FLAN-T5, both already importable) plus the mel-VAE decode ->
      HiFi-GAN vocode -> WAV tail. Scope v1 as: denoiser + scheduler loop producing a
      latent, decoded to a short (~5 s) clip written via the WAV writer; defer
      classifier-free-guidance tuning and long-form generation to a follow-up.
      The WAV writer + HiFi-GAN vocoder it builds on have landed; the open
      piece is the audio U-Net/DiT denoiser importer.
- [ ] MusicGen MELODY decoder follow-ups (BuildMusicGenMelodyFromSafeTensors[Ex],
      facebook/musicgen-melody, model_type "musicgen_melody" LANDED — decoder-only
      chroma+text prepended conditioning, ComputeMusicgenMelodyChroma chromagram,
      pico parity TestMusicGenMelodyParity, examples/MusicGenMelody smoke):
  - [ ] a --download real-checkpoint mode for facebook/musicgen-melody (the example
        is pico-only) + a real-melody-conditioned smoke clip; stereo; classifier-free
        guidance / KV-cache for the melody decoder (text-MusicGen has them).
- [ ] SeamlessM4T-v2 follow-ups deferred from the landed S2TT v1:
      (1) the text-to-speech (T2ST) unit vocoder path (TextToUnit decoder +
      HiFi-GAN-style unit vocoder). (2) the UnitY2 two-pass decoding. (3) a real
      downloaded facebook/seamless-m4t-v2-large checkpoint + real SentencePiece
      tokenizer + a runnable examples/SeamlessTranslate S2TT end-to-end demo.
- [ ] examples/SpeechCommands keyword-spotting trainer — the audio analogue of
      SimpleImageClassifier and the first FROM-SCRATCH (no-import) audio training
      example. Feed the existing log-mel frontend (neuralaudio.pas) into a small
      conv stack to classify the Google Speech Commands v2 spoken-word set
      ("yes"/"no"/"up"/"down"/...). Pure CPU, ships a tiny downloader + a
      reproducible accuracy number; demonstrates that the audio frontend is usable
      for ordinary supervised training, not only for pretrained-model import.
      LANDED (examples/SpeechCommands, .lpr + .lpi + README + scripts/download_speech_commands.sh):
      default no-network SMOKE generates a deterministic synthetic 6-keyword set
      (fixed RandSeed; low/mid/high tones + up/down chirps + band-limited noise)
      and runs it through the REAL ComputeWhisperLogMel ((100,1,40) log-mel) into a
      Conv(24,5)/MaxPool4 -> Conv(32,3)/MaxPool4 -> Conv(48,3)/MaxPool2 ->
      FullConnectReLU(64) -> Dropout -> FullConnectLinear -> SoftMax stack trained
      with TNeuralFit. Smoke validation climbs 83.5% -> 100% over 2 epochs and the
      held-out test set scores 93.89% (chance 16.7%), well under the 280 s /
      ulimit -v 3000000 budget. An optional `--full <dir>` loads real Speech
      Commands WAVs via LoadWav16ToVolume (same frontend), documented but NOT
      exercised by the smoke. Open follow-ups:
  - [ ] run the `--full` path on real downloaded Speech Commands v2 and record a
        real-data accuracy number (the synthetic smoke is trivially separable —
        validation saturates at epoch 2 — so it proves the path, not a hard
        benchmark).
- [ ] Whisper word-timestamp follow-ups (v1 landed, scoped to one 30 s
      greedy window). DONE in commit d0b27b9: (a)(e)(d) + partial (c) —
      (a) median-filter smoothing (WhisperMedianFilterRow, odd-kernel reflect-
      padded median along the audio-frame axis; optional MedianKernel param on
      WhisperCollectCrossAttention + WhisperWordTimestamps, default 0 = disabled =
      bit-identical to v1, 7 = openai-whisper default); (e) per-word confidence
      (Confidence: TNeuralFloat field on TWhisperWordTimestamp = mean DTW-path
      attention over the word's frame segment); (d) examples/WhisperTranscribe
      `--word-timestamps` flag printing `start - end [confidence] word`, documented
      in both READMEs; (c partial) baked-in alignment heads added for medium
      (24L/16H) and large (32L/20H) shapes. Tests in TestWhisperWordTimestamps
      (kernel-1 bit-identical to no-smoothing, kernel-7 monotonic non-decreasing
      boundaries + valid prob rows, confidence in [0,1.0001], non-empty
      medium/large head lists); suite 2273/0/0; example compiles (not run e2e — no
      checkpoint). REMAINING:
  - [ ] (b) multi-window stitching for clips > 30 s (carry the running time offset
        and merge the per-window word lists).
  - [ ] (c rest) read `generation_config.alignment_heads` from arbitrary model
        configs when present (v1 still hardcodes the curated tiny/base/small/medium/
        large head lists and falls back to all-heads otherwise; no config-plumbing
        + fixture yet).
- [ ] Speaker diarization importer (`BuildPyannoteSegmentationFromSafeTensors[Ex]` +
      `TPyannoteConfig`/`ReadPyannoteConfigFromJSONFile`, model_type `pyannote`) — LANDED.
      New leaf layer `TNNetSincConv1D` (SincNet band-pass, kernels materialized from two
      scalars (low_freq, band) per filter, Hamming-windowed; full forward+BPTT, input &
      weight numerical-gradient tests `TestSincConv1D*`, README layer row). Importer
      pipeline: SincConv front-end → abs/MaxPool/TokenLayerNorm → Conv+ReLU/MaxPool/
      TokenLayerNorm → bidirectional `TNNetMinLSTM` trunk (forward + time-reversed concat)
      → linear → per-frame 7-class **powerset** head + `PyannotePowersetDecode` to a
      per-speaker binary activity matrix. Parity-gated `< 1e-4` (observed ~2e-7) vs a
      hand-written numpy float64 forward oracle on a pico fixture
      (`tools/make_pico_pyannote_fixture.py` → `tests/fixtures/tiny_pyannote*`,
      `TestPyannoteParity`) — the `pyannote.audio` python package is NOT installed here, so
      the oracle reimplements the exact forward math on a re-randomized fixture.
      `examples/SpeakerDiarization` synthesizes a two-tone clip, prints a speaker-activity
      timeline + RTTM lines, saves a WAV via `SaveVolumeToWav16`. Open follow-ups:
  - [ ] Real `pyannote/segmentation-3.0` checkpoint key-mapping (the on-disk tensor
        names + the actual conv-stack depths/strides) instead of the pico fixture's
        re-keyed names, verified against a real `pyannote.audio` float64 oracle once the
        package is available.
  - [ ] Sliding-window inference + overlap stitching for clips longer than the model's
        receptive field, and turning the per-frame activity matrix into final diarized
        speaker turns (clustering across windows).
- [ ] ECAPA-TDNN speaker-verification real-checkpoint follow-up
      (`BuildEcapaTdnnFromSafeTensors[Ex]` + `TEcapaTdnnConfig` + `EcapaCosineScore`,
      new leaves `TNNetTDNNConv1D`/`TNNetAttentiveStatsPooling`, builder
      `TNNet.AddSERes2Block`, examples/SpeakerVerification, pico parity TestEcapaParity
      <1e-4 LANDED): real speechbrain/spkrec-ecapa-voxceleb checkpoint key-mapping
      (BatchNorm1d eval-mode stats, SpeechBrain TDNN naming — the pico SE block uses
      the simplified `/N²` AvgChannel scaling) + a VoxCeleb EER smoke (network/RAM-gated).
- [ ] SAM mask decoder as a real TNNet layer graph (TNNetCrossAttention two-source
      wiring) instead of the plain-array RunSAMMaskDecoder forward, so the decoder is
      trainable / fine-tunable end-to-end (v1 is inference-only). Needs a builder that
      threads the per-step query/key positional-embedding additions through the
      asymmetric token<->image cross-attentions.
- [ ] SAM full end-to-end importer + processor parity: a single BuildSAMModel that wires
      encoder + decoder + the HF SamProcessor coordinate transform (longest-side resize
      to image_size, point rescaling) so a real sam-vit-base checkpoint segments a real
      image from pixel-space clicks (v1 assumes input_image_size == raw pixel coords).
- [ ] Medusa / EAGLE tree-attention speculative decoding — a follow-up that is
      genuinely distinct from the landed SEQUENTIAL self-speculative paths
      (MTP-draft SelfSpeculativeDecoding + LayerSkip/CALM EarlyExitSelfSpeculative,
      which verify ONE linear draft sequence per step). Tree drafting proposes a
      TREE of candidate continuations (multiple top-k branches per draft head) and
      verifies them all in ONE forward pass using a block-diagonal TREE-ATTENTION
      mask so every root-to-node path is scored simultaneously, committing the
      longest accepted path. New pieces: the draft-tree builder (Medusa multi-head
      or EAGLE feature-autoregressive draft), the tree-attention mask construction,
      and the path-acceptance/commit walk. Assert greedy output is bit-identical to
      plain greedy on the target while issuing fewer target forwards; an
      examples/TreeSpeculativeDecoding demo reporting the accepted-tokens/forward
      speedup vs the linear self-speculative baseline.

- [ ] ImageNet top-1 / top-5 parity eval harness LANDED (EvaluateImageNet +
      ImageNetReport + TopKIndices + TNNetImageNetSample/Stats in
      neuralimagemetrics.pas, examples/ImageNetEval; synthetic smoke + --full <dir>
      real-val hook, TestNeuralImageMetrics green). Open follow-up:
  - [ ] Run against REAL ImageNet-val: wire one landed classifier importer
        (BuildResNetFromSafeTensors etc.) over a real checkpoint + the 50k-image
        ImageNet-val set and compare top-1/top-5 to the published torchvision numbers
        (the smoke is synthetic-only by design). Mind the multi-GB RAM budget.

- [ ] End-to-end latent text-to-image generation example (examples/LatentTextToImage)
      that finally CHAINS the imported generative pieces into one pipeline on CPU.
      NOW UNBLOCKED via the landed PixArt path (no SD-UNet needed): T5 text encoder
      (BuildT5FromSafeTensors) -> PixArt latent denoiser (BuildPixArtFromSafeTensors,
      LANDED, parity < 1e-4) -> VAE decoder (BuildVaeDecoderFromSafeTensors, LANDED)
      -> RGB, driven by the existing TNNetDiffusionScheduler DDIM/DPM-Solver++ loop
      with classifier-free guidance (PixArt uses a null/empty-caption uncond branch).
      Steps 1 & 2 LANDED: examples/LatentTextToImage runs a CFG DDIM / DPM-Solver++(2M)
      loop over the pico PixArt (regression TestLatentTextToImageSmoke) and decodes the
      (6,6,4) latent through a matched pico VAE decoder to a (12,12,3) RGB P6 PPM.
      Remaining steps:
  - [ ] Step 3 — add the real T5 tower (BuildT5FromSafeTensors) + CFG (cond vs
        empty-caption uncond) and a hard-coded prompt; ulimit-bounded demo that
        generates one small image. The CV-generative-stack-composes capstone.
        (Steps 1+2 supply only DETERMINISTIC SYNTHETIC T5 states + pico fixtures;
        Step 3 swaps in a real T5 encoder + real PixArt/VAE checkpoint.)
  - [ ] Follow-up — the pico PixArt + matched pico VAE are RANDOM (smoke only);
        once a real checkpoint is wired (Step 3) re-verify the chain produces a
        sensible image, and consider a Karras-spaced / Euler-ancestral variant.
      Edit examples/README.md. Mind the 5-min/ulimit budget — default to a smoke run.

- [ ] Mask R-CNN instance-segmentation importer follow-ups (`BuildMaskRCNNFromSafeTensors[Ex]`
      + `BuildMaskRCNN` + `RunMaskRCNN` + `TMaskRCNNConfig` + `examples/InstanceSegmentation`
      LANDED, the FIRST instance-segmentation vertical — per-OBJECT binary masks; FPN
      top-down pyramid + `TNNetRoIAlign` of an externally supplied proposal + box head
      (fc6/fc7 -> cls_score + bbox_pred) + mask head (4x conv -> ConvTranspose2d -> 1x1
      per-class mask logits); pico parity TestMaskRCNNParity 1.8e-6 < 1e-4 vs a numpy
      float64 oracle). Deferred follow-ups (v1 scope was bounded to the tested core):
  - [ ] RPN / anchor generation + proposal NMS (v1 takes EXTERNALLY supplied proposal
        boxes; the RPN head + anchor grid + objectness/box-delta decode + top-k/NMS
        proposal selection are the missing front end for an end-to-end detector).
  - [ ] wire the real ResNet-50 + FPN BACKBONE into one forward (v1 feeds the FPN-input
        feature maps directly as TNNetInput levels; a full run should tap C2..C5 from
        BuildResNetFromSafeTensors and feed all FPN levels, then route each proposal to
        its FPN level by box area like torchvision's MultiScaleRoIAlign).
  - [ ] real torchvision maskrcnn_resnet50_fpn checkpoint parity (the landed parity is
        the hand-built pico numpy oracle only); + box-delta decode/clip + mask paste-to-
        image + per-class NMS for a meaningful instance overlay on a real photo.

- [ ] TrOCR optical-character-recognition importer follow-up (BuildTrOCRFromSafeTensors
      LANDED, commit 2000f69; DeiT encoder + Bart decoder on the T5EncoderStates two-net
      convention, TestTrOCRParity decoder logits < 1e-4 + examples/OCRTranscribe):
  - [ ] tensor-name mapping targets the transformers 5.11 layout
        (encoder.layers.N.attention.q_proj, decoder.model.decoder.*); add the published
        checkpoints' older DeiT naming if it differs (the fixture is the parity contract).
- [ ] EfficientNet (timm / torchvision efficientnet_b0..b7) importer follow-up
      (BuildEfficientNetFromSafeTensors LANDED, commit d35970c; AddEfficientNetMBConv
      reusing MobileNetV3 MBConv + conv-BN fold, b0..b7 share one JSON-driven builder,
      TestEfficientNetImageClassificationParity max|diff| < 1e-4 vs a numpy float64 oracle):
  - [ ] real torchvision/timm efficientnet_b0 checkpoint parity + top-1 via the tracked
        ImageNet eval harness (the landed parity is the hand-built pico oracle only).

- [ ] StyleGAN2 generator importer follow-ups (BuildStyleGAN2Generator + NEW leaf layer
      TNNetModulatedConv2D LANDED, commit 8d72b95; mapping MLP z->w + synthesis tower of
      modulated conv + ReZero noise + LeakyReLU + summed toRGB skips, inference-only,
      TestStyleGAN2GeneratorParity < 1e-4 vs a numpy float64 oracle + examples/StyleGAN2Generate):
  - [ ] per-pixel 1-channel BROADCAST random noise (v1 stores full-depth fixed noise maps
        in the safetensors for deterministic oracle-exact synthesis — wire the standard
        per-pixel broadcast noise for real stochastic synthesis).
  - [ ] real stylegan2 checkpoint (NVIDIA .pkl / a rosinality safetensors) parity once
        obtainable; the training path (discriminator + path-length reg) and StyleGAN3.
- [ ] MMDiT (Stable Diffusion 3 / FLUX.1) full text-to-image importer + sampler
      (dual-stream joint-attention BLOCK v1 LANDED — AddMMDiTJointBlock +
      BuildMMDiTBlockFromSafeTensors + ReadMMDiTConfigFromJSONFile): full
      BuildMMDiTFromSafeTensors stack (patch embed + combined CLIP+T5 prompt tower +
      pooled-projection conditioning + N joint blocks + final adaLN/unpatchify); the
      CONTEXT-FREE final block (context_pre_only=True, text stream dropped); the
      end-to-end rectified-flow Euler sampler (reuse examples/FlowMatching) into the
      LatentTextToImage capstone with NO SD UNet; real stable-diffusion-3-medium
      (or Flux-schnell) checkpoint parity; SD3.5 RMSNorm QK-norm support.
- [ ] AnimateDiff text-to-VIDEO motion-module importer (BuildAnimateDiffFromSafeTensors,
      e.g. guoyww/animatediff-motion-adapter-v1-5-2) — the FIRST video-GENERATIVE
      importer (a sequence of frames from a text prompt), a brand-new generative
      modality vs every landed image generator (DiT/PixArt diffusion, VisualGAN/
      StyleGAN2, MaskGIT). AnimateDiff inserts a TEMPORAL motion module after each
      spatial block of a frozen SD UNet: the per-frame token grids are transposed so
      attention runs along the TIME axis (each spatial location attends across frames)
      with a sinusoidal temporal position embedding, learning motion while the spatial
      weights stay fixed. The genuinely new code is that temporal-axis attention block
      (reuse the landed SDPA over a (NumFrames,1,C) reshape per spatial cell, exactly
      the VideoMAE space<->time transpose trick) + the zero-initialised residual
      injection into the frozen UNet (same wiring as the tracked ControlNet zero-conv).
      Builds on the landed SD UNet importer (BuildSDUNetFromSafeTensors); track as its
      natural successor alongside ControlNet. Pico parity vs a diffusers
      float64 oracle on one motion-module output for a fixed multi-frame latent; an
      examples/TextToVideo that writes a short animated GIF/PPM sequence on CPU. Note:
      a cheaper no-UNet route to video is bolting the same temporal block onto the
      landed PixArt DiT.
- [ ] CogVideoX text-to-VIDEO DiT real-checkpoint follow-up
      (BuildCogVideoXFromSafeTensors[Ex] + TCogVideoXConfig + DecodeCogVideoXVae +
      examples/TextToVideo LANDED, reusing TNNetMRotaryEmbedding 3D RoPE +
      TNNetCausalConv1D temporal VAE, no new leaf; pico parity TestCogVideoXParity
      < 1e-4): real-checkpoint parity (THUDM/CogVideoX-2b) vs a sliced real checkpoint
      (the make_pico_*_fixture slicer pattern) once weights are available, incl. the
      real diffusers 3D RoPE frequency layout / temporal VAE up/down blocks and a real
      T5 encoder over a tokenized prompt feeding BuildT5FromSafeTensors instead of the
      synthetic text states. Further gaps (audit 2026-10-02), which together mean
      the repo has NO usable video generator today:
  - [ ] Full AutoencoderKLCogVideoX decoder: DecodeCogVideoXVae is only the decode
        TAIL (one causal temporal conv + SiLU + pointwise conv per spatial cell),
        not the real up/down-block VAE, so even real denoiser weights cannot
        produce real frames.
  - [ ] A `TCogVideoXPipeline` over a diffusers model_index.json folder (the
        TQwenImage21Pipeline shape: tokenizer + T5 -> transformer -> VAE, one
        component in memory at a time, OnPhase/OnStep) and a CLI that loads it;
        examples/TextToVideo takes only a single .safetensors today.
  - [ ] CFG in the TextToVideo loop with a real empty-prompt uncond branch.
  - [ ] Output: frames are written as separate P6 PPMs only; add an animated
        GIF / PNG-sequence or a video-container writer.
- [ ] Wan 2.1 text-to-VIDEO DiT importer (`BuildWanFromSafeTensors[Ex]` +
      `TWanConfig`, model_type "wan", e.g. Wan-AI/Wan2.1-T2V-1.3B) — the current
      most-downloaded open text-to-video model, a flow-matching MMDiT distinct from
      the landed CogVideoX (Wan uses a 3D causal-VAE latent + a different DiT block:
      self-attn + cross-attn to a T5/umT5 text encoder + a per-block time-modulated
      AdaLN, no QK-norm). Reuse the landed MMDiT/DiT cross-attention + AdaLN
      modulation blocks and the 3D-RoPE / TNNetCausalConv1D temporal-VAE primitives
      from the CogVideoX path; add only the Wan-specific patch-embed (1 x 2 x 2
      patchify), the flow-matching sampler, and the Wan VAE decoder key mapping.
      Pico parity TestWanParity < 1e-4 vs a float64 HF WanTransformer3DModel; defer
      real Wan-AI/Wan2.1 weights (network/RAM-gated) to a slicer follow-up, plus an
      examples/TextToVideo --wan mode. Note: distinct enough from CogVideoX that the
      MMDiT-video path is reused, not duplicated.
- [ ] Stable Video Diffusion IMAGE-to-video importer (`BuildSVDFromSafeTensors[Ex]`
      + `TSVDConfig`, e.g. stabilityai/stable-video-diffusion-img2vid) — the
      image-CONDITIONED video generator that complements the landed CogVideoX
      TEXT-to-video DiT (different modality: an input frame, not a prompt, drives a
      short clip). A spatio-temporal U-Net denoiser: reuse the landed SD/diffusion
      2D conv-resnet + cross-attention blocks for the SPATIAL layers and add the
      TEMPORAL mixing (per-pixel attention/conv across the frame axis + the
      learnable per-block temporal mix factor) interleaved between them; CLIP-image
      embedding conditioning (reuse `BuildClipVisionTower`) plus the
      fps/motion-bucket/noise-aug micro-conditioning embeddings; EDM-preconditioned
      denoiser + the temporal VAE decode (reuse the landed VAE decoder path). Pico
      parity `< 1e-4` on one denoiser step + one VAE decode vs a first-principles
      float64 oracle on a committed random pico config (the CogVideoX fixture
      pattern), `tools/make_pico_svd_fixture.py` / `TestSVDParity`,
      `examples/ImageToVideo` driving the pico denoiser through EDM sampling and
      writing a per-frame PPM sequence. Real-checkpoint parity is the network/RAM-
      gated follow-up.
- [ ] VAR (Visual AutoRegressive, next-scale prediction) image-generation follow-ups
      (class-conditional v1 LANDED — BuildVARFromSafeTensors[Ex] + ReadVARConfigFromJSONFile
      + the TNNetScaledDotProductAttention.BlockCausalSegments scale-mask flag):
  - [ ] VAR real-checkpoint parity (FoundationVision/var) — v1 parity is vs a
        first-principles float64 oracle on a random pico config; verify against a sliced
        real checkpoint once weights are available (the make_pico_*_fixture slicer
        pattern).
- [ ] Structured-vision accuracy eval harness for the imported DETECTION and DENSE-
      prediction backbones (EvaluateDetectionMAP / EvaluateSegmentationMIoU + reports in
      neuralimagemetrics.pas, plus examples/VisionEval) — the verification backstop that
      the tracked ImageNet top-1 harness is for CLASSIFIERS, but for the importers whose
      output is NOT a class vector: boxes (DETR / YOLO / OWL-ViT) and dense maps
      (SegFormer / DPT-Depth). Each detection importer's parity test only compares raw
      head logits on one image, which catches a transposed weight but NOT a wrong box
      decode (cxcywh<->xyxy, sigmoid placement), NMS, or label permutation. Add: (a) a
      COCO-style mean-Average-Precision scorer (per-class precision/recall over IoU
      thresholds 0.50:0.95, the standard 101-point interpolation) over a small folder of
      labelled boxes; (b) a semantic-segmentation mean-IoU / pixel-accuracy scorer over a
      small folder of label-map PNGs. Pure CPU post-process on the landed importers'
      outputs; reports a per-class table + a few visual overlays. Distinct from the
      logit-parity tests (those pin the math; this pins the end-to-end pipeline incl.
      preprocessing/decode). The missing import-VERIFICATION mirror of MMLUEval for vision.
- [ ] Florence-2 unified vision importer follow-ups (BuildFlorence2FromSafeTensors
      [WithConfig] + TFlorence2Config + RunFlorence2Logits/RunFlorence2Projector +
      Florence2Quantize/DequantizeCoord location tokens + examples/Florence2 LANDED —
      visual-prefix BART encoder + BART seq2seq decoder, pure builder reuse, no new leaf;
      pico parity TestFlorence2Parity < 1e-4 + TestFlorence2LocationTokens vs the real HF
      Florence2ForConditionalGeneration float64 oracle):
  - [ ] The DaViT vision tower itself is the DEFERRED gap: v1 takes the tower's
        last_hidden_state feature map as a PRECOMPUTED input (like the tracked Qwen2-VL
        "merged visual tokens as input v1"). Build the real DaViT: a 4-stage conv-embed
        (kernel/stride/pad per stage, pre/post LayerNorm), per-block depthwise 3x3 convs
        (TNNetDepthwiseConv), WINDOW attention (partition H*W into windows; reuse
        TNNetWindowAttention), and the genuinely-new GROUPED CHANNEL attention (attention
        ACROSS channels within groups, scale = num_tokens^-0.5 — NO existing Pascal layer,
        needs a new leaf or a transposed-SDPA builder). Then drop the precomputed-feature
        shortcut and add the patch_embed/window-partition reshape plumbing.
  - [ ] Real-checkpoint parity (microsoft/Florence-2-base) + the real DaViT image
        preprocessing + the Florence-2 tokenizer (post-processing the <loc_>/box token
        stream into pixel boxes per task) + autoregressive DecodeFlorence2Greedy with a
        proper caption/OD parse; v1 runs a single greedy step + the location-token math.
  - [ ] <OCR>, <REFERRING_EXPRESSION_SEGMENTATION> (polygon <loc_> streams) and the other
        task tokens beyond <CAPTION>/<OD>; the polygon (de)quantization reuses the same
        per-coordinate <loc_> helpers but needs the polygon-point grouping parse.
- [ ] Qwen2-VL / Qwen2.5-VL vision-language importer follow-ups (M-RoPE v1 LANDED —
      BuildQwen2VLFromSafeTensors[Ex] + TNNetMRotaryEmbedding + Qwen2VLRunLogits):
  - [ ] The native-dynamic-resolution VISION TOWER (Conv3d patch embed + window
        attention over a variable patch grid + the 2x2 spatial patch-merger MLP) — v1
        takes the MERGED visual tokens as input (precomputed). Reuse the landed
        BuildClipVisionTower/SigLIP ViT path; build the variable-grid window-attention
        mask + the patch merger. Then drop the precomputed-embeds shortcut.
  - [ ] Qwen2.5-VL specifics (the temporal M-RoPE section for VIDEO, the per-frame
        time_interval, the windowed full-attention layer pattern, MRoPE rope_deltas for
        incremental decode) — the M-RoPE index builder already has the temporal slot.
  - [ ] Multi-image / multi-modality prompts (the v1 position builder assumes ONE
        contiguous still-image block); generalize the get_rope_index grouping.
  - [ ] Real-checkpoint parity (Qwen/Qwen2-VL-2B-Instruct) + an examples/Qwen2VLDescribe
        that captions a tiny CPU image (ulimit-bounded).
- [ ] CLIPSeg text-prompted zero-shot segmentation importer follow-ups
      (BuildCLIPSegFromSafeTensors[WithConfig] + TCLIPSegConfig + RunCLIPSeg LANDED —
      frozen CLIP dual encoder + a FiLM-conditioned transposed-conv decoder emitting a
      single-channel mask for a free-text prompt; pico parity TestCLIPSegParity < 1e-4
      vs the REAL HF CLIPSegForImageSegmentation float64 oracle; examples/CLIPSegPrompt
      writes a binary-mask PPM; v1 = single text prompt -> one mask, inference-only,
      use_complex_transposed_convolution = false):
  - [ ] Real-checkpoint parity (CIDAS/clipseg-rd64-refined) — verify against the REAL
        HF checkpoint logits (the offline env could not download it; the pico parity
        already pins the math vs the real HF classes). Needs the ~600 MB checkpoint +
        CLIP tokenizer ids; also exercises the default extract_layers [3,6,9] and
        reduce_dim 64 (the pico uses [0,1]/6).
  - [ ] Image-prompt (visual) conditioning — CLIPSeg can also condition on a PROMPT
        IMAGE (conditional_pixel_values -> clip.get_image_features pooled embedding)
        instead of text; v1 does text only. Add a RunCLIPSegImagePrompt path reusing
        the vision tower's pooled class-token embedding as the conditional vector.
- [ ] BEiT / data2vec-vision ViT importer follow-ups (BuildBeitFromSafeTensors[Ex]
      /WithConfig + TBeitConfig + ReadBeitConfigFromJSONFile/BeitConfigToString
      LANDED, e.g. microsoft/beit-base-patch16-224, facebook/data2vec-vision-base —
      full global attention with a per-LAYER learned cls-aware relative_position_bias
      reusing TNNetWindowAttention (no new leaf layer) + LayerScale (TNNetChannelMul)
      on both branches, learnable cls token, query/value biased + KEY bias-free, no
      absolute positions; BeitBuildRelPosIndex matches HF generate_relative_position_index
      exactly; pico parity TestBeitParity/TestBeitConfigFromJSONFile < 1e-4 vs HF
      float64 oracle via tools/beit_tiny_fixture.py + tests/fixtures/tiny_beit.*):
  - [ ] use_shared_relative_position_bias=true (one model-level table shared by
        all layers) is rejected; only the per-layer table is wired.
  - [ ] use_absolute_position_embeddings=true and use_relative_position_bias=
        false variants rejected (the published checkpoints don't use them).
  - [ ] BEiTv2 (vector-quantized) not validated.

- [ ] OPT decoder importer (BuildOPTFromSafeTensors[Ex], model_type "opt", e.g.
      facebook/opt-125m..2.7b) — LANDED (learned-absolute +2-offset positions,
      LayerNorm + ReLU FFN, pre-/post-LN per do_layer_norm_before, optional
      final_layer_norm, BuildFromPretrained dispatch; TestOPTNextTokenLogitsParity
      < 1e-4 vs float64 HF OPTForCausalLM). Open follow-ups:
  - [ ] Wire OPT as the blip2-opt decode tail (the original motivation): feed
        the Q-Former query/text states into BuildOPTFromSafeTensors and verify
        a BLIP-2-OPT generation path.

- [ ] DepthPro (Apple ml-depth-pro) sharp metric monocular-depth importer
      (BuildDepthProFromSafeTensors[Ex], apple/DepthPro). DISTINCT from the landed
      DPT and Depth-Anything importers, which produce RELATIVE/affine-invariant
      depth at a single patch scale: DepthPro is a MULTI-SCALE patch-ViT that runs a
      shared ViT (reuse the landed DINOv2/ViT tower) over the image at several
      resolutions, splits each into overlapping 384px patches, then fuses the
      per-scale patch features through a DPT-style convolutional decoder to emit a
      high-resolution METRIC depth map (plus a focal-length head for true scale).
      The genuinely new code is the image->patch tiling + per-scale feature
      stitching/merge and the metric (not normalized) depth head; the ViT encoder
      and the DPT fusion-block decoder are already landed and reusable. Real value:
      first sharp, boundary-accurate, metric-depth model in the tree. Pico-fixture
      smoke + a real-checkpoint parity follow-up.

- [ ] RT-DETR real-time detection-transformer importer
      (BuildRtDetrFromSafeTensors[Ex], e.g. PekingU/rtdetr_r50vd). A DISTINCT
      detection model family from the landed DETR (ResNet+sinusoidal+vanilla
      transformer decoder, the slow NMS-free baseline): RT-DETR is the real-time
      redesign — a CNN backbone (reuse the landed ResNet) feeding an EFFICIENT
      HYBRID ENCODER (intra-scale single-layer self-attention "AIFI" on the top
      feature map only + a CNN cross-scale fusion / PANet-style neck) and an
      IoU-AWARE / uncertainty-minimal QUERY SELECTION that seeds the decoder from
      the top-K encoder proposals instead of fixed learned object queries. The new
      code is the hybrid-encoder neck (AIFI block + CCFM up/down conv fusion) and
      the top-K query-selection head; the transformer decoder, box/class heads, and
      the DETR-style post-process can be reused from the landed DETR. Real value:
      the first real-time / production-grade detector importer (DETR is reference,
      not deployable speed). Pico-fixture smoke + a real-checkpoint top-1/box
      parity follow-up via the structured-vision detection eval harness.

- [ ] Fuyu decoder-only multimodal importer (BuildFuyuFromSafeTensors[Ex],
      model_type "fuyu", e.g. adept/fuyu-8b). Architecturally DISTINCT from every
      landed vision-language importer (CLIP/SigLIP-ViT tower + projector feeding a
      decoder — LLaVA, Qwen2-VL, PaliGemma, BLIP-2, Florence-2): Fuyu has NO
      separate vision encoder at all. Raw image patches are unfolded
      (patch_size x patch_size x channels), linearly projected by a single
      `vision_embed_tokens` dense layer straight into the decoder's token stream,
      interleaved with text tokens via `|SPEAKER|`/`|NEWLINE|` image-newline
      markers. The decoder is a Persimmon stack (standard RoPE attention with a
      QK-LayerNorm, square-ReLU MLP, separate qkv proj). New code is: the
      patchify+linear-projection input adapter (reuse TNNetReshape/SpaceToDepth for
      the unfold + a FullConnect for the projection) and the Persimmon block wiring
      (QK-LayerNorm before RoPE, ReLU^2 FFN — TNNetReGLUSquared already exists). Real
      value: covers the "patches-as-tokens" multimodal family (Fuyu, and the recipe
      generalizes to UI/document screenshots) that the encoder-tower importers
      cannot express. Pico-fixture parity < 1e-4 vs a float64 HF
      FuyuForCausalLM.forward, then a real adept/fuyu-8b caption smoke.

- [ ] Donut / Nougat OCR-free document-understanding importer
      (BuildDonutFromSafeTensors[Ex], model_type "vision-encoder-decoder" with a
      Donut-Swin encoder, e.g. naver-clova-ix/donut-base / facebook/nougat-base).
      A DISTINCT task family from the landed TrOCR (ViT encoder + text decoder for
      single-line OCR) and the caption VLMs: Donut/Nougat read a FULL document image
      and autoregressively emit STRUCTURED output (Donut: a `<s_...>` key-value
      sequence for receipts/forms; Nougat: Markdown+LaTeX for scientific PDFs) with
      NO external OCR step. New code is mostly wiring: the encoder is a Swin
      transformer (reuse the landed Swin/SwinIR window-attention blocks) and the
      decoder is an MBart cross-attention stack (reuse the landed mBART importer);
      the importer maps the `encoder.`/`decoder.` HF key namespaces and the
      shifted-token-id generation prompt. The genuinely new pieces are the
      Donut-Swin patch-embed/relative-position-index loading and the
      task-prompt-conditioned `RunSeq2Seq`-style greedy/beam decode over the
      document. Real value: first document-AI (forms, receipts, scientific-PDF
      transcription) importer; reuses Swin + mBART that already ship. Pico-fixture
      cross-entropy parity < 1e-4 vs a float64 HF VisionEncoderDecoderModel.

- [~] Stable Diffusion img2img + inpainting pipelines. DONE (commit 2191b80): the
      VAE ENCODER gap is closed — BuildVaeEncoderFromSafeTensors[Ex] +
      ReadVaeEncoderConfigFromJSONFile land the diffusers AutoencoderKL encoder
      (conv_in -> down blocks with stride-2 downsample convs reproducing the
      asymmetric (0,1,0,1) pad -> mid self-attn -> conv_norm_out/conv_out ->
      quant_conv), reusing every decoder helper (TVaeDecoderConfig,
      AddVaeResnetBlock, AddVaeAttention, LoadVaeConv). The net's final layer crops
      to mean and applies the 0.18215 scaling_factor = the deterministic img2img
      init-latent setup. Pico fixture (tools/vae_encoder_tiny_fixture.py ->
      tests/fixtures/tiny_vae_encoder*) + TestVaeEncoderParity (numpy float64
      oracle, max|diff| < 1e-4) + TestVaeRoundTrip (encode->decode end to end).
      DONE (img2img sampler loop): examples/ImageToImage drives the full
      encode->partial-noise->denoise->decode SDEdit trajectory (strength knob).
      DONE (blended inpainting): SDUNetDenoiseInpaint (neuralpretrained.pas) runs a
      complete reverse loop on the standard 4-channel SD UNet and at EVERY step
      blends the known region back in (latents = mask*denoised + (1-mask)*AddNoise(
      z0,t)); examples/DiffusionInpainting smoke-runs it on the pico VAE+SD-UNet
      fixtures (right-half hole), writing masked_input.ppm/inpainted.ppm, and
      TestDiffusionInpaintSmoke asserts the kept latent region equals z0 exactly
      while the masked hole changes. This is the no-retrain blended mode.
      STILL OPEN: (1) the non-deterministic VAE reparameterize head
      (mean+exp(0.5*logvar)*noise); (2) the 9-channel inpaint-specialized UNet
      (conv_in widened to 4 latent + 1 mask + 4 masked-latent) importer support.
      ORIGINAL NOTE — The landed SD path
      (BuildVaeDecoderFromSafeTensors + BuildSDUNetFromSafeTensors + SDUNetDenoise +
      the neuraldiffusion sampler zoo) does text-to-image only — it starts denoising
      from pure Gaussian latents. The two highest-value missing modes reuse ALL of
      that and add only the latent setup/blend: (a) img2img — VAE-ENCODE an init
      image to latents (the VAE decoder importer's encoder twin is the gap; add
      BuildVaeEncoderFromSafeTensors for the down-blocks + the
      `mean,logvar`->reparameterize sampling head), then start the existing sampler
      at timestep `int(strength * num_steps)` with noise added to those latents
      (`scheduler.add_noise` — already have the alphas_cumprod schedule); (b)
      inpainting — keep img2img's noised init but at every denoise step BLEND the
      known region back in (`latents = mask*denoised + (1-mask)*noised_known`), and
      support the 9-channel inpaint UNet (4 latent + 1 mask + 4 masked-latent in
      `conv_in`) by widening the importer's first-conv loader. Real value: img2img
      and inpainting are the two most-used SD modes after txt2img and unlock the
      ImageToImage/Inpainting example stubs. Pico parity: VAE-encode round-trip <
      1e-4 vs a float64 HF AutoencoderKL.encode, and a fixed-seed img2img latent
      trajectory matching diffusers `StableDiffusionImg2ImgPipeline` step-for-step.
- [ ] Kosmos-2 grounded vision-language importer (BuildKosmos2FromSafeTensors[Ex],
      model_type "kosmos-2", e.g. microsoft/kosmos-2-patch14-224). DISTINCT from the
      landed caption VLMs (LLaVA/Blip2/Florence-2 fuse image features as prepended
      tokens for free-form text): Kosmos-2 emits GROUNDED text where phrases are
      tied to image regions via special `<grounding>`/`<phrase>`/`<object>` tokens
      whose `<patch_index_NNN>` ids decode to bounding boxes over the patch grid.
      It pairs a CLIP ViT image encoder (landed BuildClipVisionTower) with a Magneto
      transformer decoder (a Llama-ish stack with a sub-LayerNorm / scaled residual
      variant) and an image-to-text projection. New code: the Magneto decoder block
      (the extra normalization placement vs the landed Llama backbone) and the
      grounding token-id <-> patch-index <-> bbox decode helper + an
      examples/Kosmos2Ground demo that prints phrase->box pairs. Real value: first
      grounded-captioning / phrase-localization importer, a different capability
      from the landed open-vocabulary DETECTION and free-form caption paths.
      Pico-fixture parity < 1e-4 vs a float64 HF Kosmos2ForConditionalGeneration.
- [ ] SmolVLM2 image-AND-video VLM importer (`BuildSmolVLM2FromSafeTensors[Ex]`,
      model_type "smolvlm"/Idefics3-family, e.g. HuggingFaceTB/SmolVLM2-2.2B-Instruct
      and the 256M/500M tiers). Pairs a SigLIP vision tower (landed
      `BuildClipVisionTower`/SigLIP path) with a **pixel-shuffle space-to-depth
      connector** (reuse the landed `TNNetSpaceToDepth`/`TNNetPixelShuffle` to fold a
      `scale_factor²` spatial block into the channel axis, shrinking the per-tile
      token count) feeding a SmolLM2 (Llama-path) text decoder. New code is the
      image-splitting / tiling preprocessor and the **multi-frame VIDEO token
      packing** (sample N frames, encode each through the shared vision tower, and
      interleave the per-frame token grids with frame/`<row>_<col>` separators) —
      the rest is wiring landed layers. Pico fixture + `TestSmolVLM2LogitParity`
      < 1e-4 vs a float64 HF `SmolVLMForConditionalGeneration`, plus an
      examples/SmolVLM2Describe demo (one image, then a short clip). Real value:
      the landed caption VLMs (LLaVA/Blip2/Florence-2/PaliGemma) are image-only;
      this is the efficient pixel-shuffle-connector + native short-VIDEO path.
- [ ] RegNet importer follow-ups (`BuildRegNetFromSafeTensors[Ex]` LANDED,
      transformers model_type `regnet`, e.g. facebook/regnet-y-040 / regnet-x-040;
      conv+BN fold at load, grouped 3x3 via per-group SplitChannels/DeepConcat,
      regnet_y SE gate, NO new leaf layers; pico parity
      `TestRegNet{Config,XImageClassification,YImageClassification}Parity` < 1e-4
      vs a REAL float64 HF RegNetForImageClassification):
  - [ ] real-checkpoint top-1 sweep (facebook/regnet-y-040 / regnet-x-040 needs
        the HF image-processor preprocessing path).
  - [ ] timm naming variant.
  - [ ] downsample-in-first-stage=true real configs (code handles it, untested
        on a real ckpt).
- [ ] Qwen-Image-2.1 text-to-image and image editing (Qwen/Qwen-Image-2.1, diffusers
      `QwenImage21Pipeline`, Qwen Research License). Target: Phase A (text-to-image)
      first, 1024x1024 on the OpenCL GPU box, branch a19; B and C are recorded, not
      scheduled. One Opus agent per task, run serially, test + commit after each.
      Reference: diffusers `transformer_qwenimage21.py`, `autoencoder_kl_qwenimage21.py`,
      `pipelines/qwenimage21/pipeline_qwenimage21.py` (merged 2026-09-18).
      Checkpoint facts (from the configs and safetensors headers):
      - text encoder `text_encoder/`: Qwen3-VL-8B (`qwen3_vl`). Text part = Qwen3 dense
        decoder, 36 layers, hidden 4096, 32 q / 8 kv heads, head_dim 128, per-head QK
        RMSNorm, rope_theta 5e6, interleaved M-RoPE [24,20,20]. For a text-only prompt
        the three M-RoPE positions are equal, so it reduces to plain 1D RoPE. The
        transformer reads the LAST decoder layer's output BEFORE the final RMSNorm
        (diffusers hooks the norm away), minus the system-prompt tokens (drop_idx =
        tokenized length of the system message). Prompt template:
        `<|im_start|>system\nComprehend and analyze the provided prompt.<|im_end|>\n
        <|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n`.
      - transformer `transformer/`: 7B, BF16, 2 shards, NO biases anywhere. 32
        SINGLE-stream blocks, dim 4096, 32 heads x 128, SwiGLU 12288 (`gate_layer`,
        `proj`, `out`). Block: non-affine LayerNorm -> x(1+scale) -> attention (QK
        RMSNorm, 3-axis RoPE) -> residual with tanh(gate); same again for the MLP.
        ONE shared `modulation` Linear 4096->16384 (scale1, gate1, scale2, gate2) feeds
        every block. `txt_in` = zero-centred RMSNorm (stored weight-1) -> Linear ->
        GELU-tanh -> Linear. `img_in` Linear 64->4096. `norm_out` = non-affine LayerNorm
        x (1+scale), no shift. `proj_out` 4096->64. Timestep embedding: t in [0,1] x
        1000, cos half FIRST then sin, dim 256 -> Linear -> SiLU -> Linear.
      - attention mask: block-causal, `(q >= kv) or same_image_block`. With
        `causal_condition`, text and condition-image tokens take modulation from t=0,
        so their K/V are step-independent: the pipeline computes them once (extract)
        and later steps run only the target-image tokens against the cached prefix
        (cached). RoPE: axes (frame, h, w) = (16, 56, 56) dims, theta 10000,
        consecutive-pair (complex) layout; text tokens get position p on all 3 axes;
        an image block gets frame = position after the preceding text and an (h, w)
        grid centred on zero (negative indices allowed).
      - VAE `vae/` (`AutoencoderKLQwenImage21`, F32): 2D for images (all conv weights
        [out,in,3,3]) despite the Wan-style names. RGBA in/out, z_dim 64, 16x spatial.
        Encoder base 96, decoder base 144 (up to 1152 channels). Channel RMSNorm
        (F.normalize x sqrt(C) x gamma), SiLU, residual AvgDown/DupUp shortcuts,
        nearest-exact upsample + conv, single-head mid-block attention. Latents are
        de-normalised with 64 `latents_mean` / `latents_std` values before decode.
      - scheduler: `FlowMatchEulerDiscreteScheduler`, sigmas = linspace(1, 1/N, N),
        exponential dynamic shift, mu linear in image token count (256 -> 0.5,
        8192 -> 0.9), `shift_terminal` 0.02; x += (sigma_next - sigma) * v. Guidance
        `true_cfg_scale` defaults to 1.0 (off): one transformer pass per step. Default
        40 steps.
      - cost: 1024x1024 = 4096 image tokens, ~66 TFLOP per step, ~2.6 PFLOP for 40
        steps (2048x2048 ~6x more). Compute-bound GEMM, not decode GEMV. int8:
        transformer ~7 GB, text encoder ~8 GB, loaded one after the other.
  Phase A — text-to-image, correctness first:
  - [x] A0. Parity oracle: install diffusers (git main) + transformers >= 5.17 into
        the `x` venv; a script that builds tiny random-weight versions of each
        component and writes float64 inputs/outputs to `tests/fixtures/`.
        Done: `tools/make_pico_qwenimage21_fixture.py` -> `tests/fixtures/tiny_qwenimage21/` (pico diffusers folder), `qwenimage21_{scheduler_oracle,prompt_tokens}.json`, `tiny_qwenimage21_{text_encoder,rope,transformer,vae,pipeline}_io.json`.
  - [x] A1. Flow-matching Euler scheduler in `neuraldiffusion.pas` (decide: extend
        `TNNetDiffusionScheduler` or a new class): dynamic exponential shift,
        `shift_terminal`, velocity step. JSON oracle test. Independent.
        Done: new class `TNNetFlowMatchEulerScheduler` (CalculateShift, TimeShift, SetSigmas, SetTimesteps, Step); tests `TestFlowMatch*` in `tests/TestNeuralDiffusion.pas`.
  - [x] A2. Qwen3-VL text encoder as prompt encoder: route `qwen3_vl` to the Qwen3
        builder (`model.language_model.` prefix, skip `model.visual.*` and `lm_head`,
        int8 load); test that text-only interleaved M-RoPE equals 1D RoPE; API returning
        pre-final-norm last-layer hidden states; template + tokenization + drop_idx.
        Done: `BuildQwen3VLTextEncoderFromSafeTensors` (Llama builder, `pStopBeforeFinalNorm`), `Qwen3VLEncodeHiddenStates`, `QwenImage21TextToImagePrompt`, `QwenImage21EncodeTextToImagePrompt`; `qwen3_vl` also in `BuildFromPretrained`; tests `TestQwen3VL*`, `TestQwenImage21PromptTemplateIds`.
  - [x] A3. Qwen-Image RoPE: extend `TNNetMRotaryEmbedding` for the consecutive-pair
        layout, 16/56/56 sections and negative positions; position builder (text
        0..L-1, image frame = L, centred h/w grid).
        Done: subclass `TNNetAxialRotaryEmbedding` (per-axis tables; M-RoPE gained `pRotaryHeadDim` head tiling), `BuildQwenImage21RopePositions`; tests `TestQwenImage21Rope*`, `TestAxialRoPE*`, `AxialRoPEOpenCLParity`, `TestMRoPETiledMatchesPerHead`.
  - [x] A4. Transformer block, step pass: image-token queries attend (no causal mask)
        to [cached prefix K/V ; own K/V]. Reuse `TNNetHeadRMSNorm`, `TNNetLayerNorm`,
        `TNNetFiLM` / `TNNetChannelMulByLayer`, the fused SDPA KV cache and batched
        prefill where they fit; fold the zero-centred +1 into the RMSNorm weight at
        load. Single-block parity vs HF.
        Done: `AddQwenImage21Block` (qibStep / qibPrefix), `AddQwenImage21Modulation`, `AddQwenImage21TextProjection`, `LoadQwenImage21BlockWeights`, `ReadQwenImage21TransformerConfig`; `TNNetFusedSDPA` gained `CachedForwardNonCausal` + `AppendCacheRowsFrom`; tests `TestQwenImage21StepBlockParity`, `TestQwenImage21PrefixBlockKVParity`, `TestCachedForwardNonCausal`.
  - [x] A5. `BuildQwenImage21Transformer`: a PREFIX network (text tokens, causal, t=0
        modulation, emits per-layer K/V) and a STEP network (target tokens); timestep
        embedding, shared modulation computed once per step, `norm_out`, `proj_out`;
        sharded diffusers-folder loader. Parity vs HF extract and cached modes.
        Depends on A3 + A4. Memory: ONE reusable step block whose layers are pointed
        at block i's weights before block i runs (each block keeps its own KV cache);
        32 blocks in one TNNet would need ~64 GB of activations at 1024x1024 (~2 GB
        per block at N=4096). One code path for every image size.
        Done: class `TQwenImage21Transformer` (weight store `BlockStore[i]`, `SelectBlockWeights` re-links via `TNNet.LinkWeightsFrom`, `EncodePrefix`, `PredictVelocity`, `qiwFP32/qiwInt8/qiwInt4`), `QwenImage21SetRopePositions`; `TNNetLayer.LinkWeightsFrom` may re-link; tests `TestQwenImage21Transformer*`.
  - [x] A6. VAE decoder: first check whether image decode ever runs `time_conv`;
        channel RMSNorm, DupUp shortcuts, mid attention, latent de-normalisation,
        tiled decode (2048x2048 full-resolution activations are ~2.4 GB each). Pico
        parity. Independent.
        Done: class `TQwenImage21VaeDecoder` (1x1-latent weight owner + borrowing sized net; `Decode`, `DecodeTiled` = diffusers tiled_decode), `ReadQwenImage21VaeConfig`, `BuildQwenImage21VaeDecoderNet`, `LoadQwenImage21VaeDecoderWeights` (skips time_conv, folds latents_mean/std into post_quant_conv), `QwenImage21DupUpChannels` + `AddQwenImage21DupUp`; conv kernels now clamp to the PADDED input; tests `TestQwenImage21Vae*`. Real-size activations (weights 1.01 GB): 0.58 GB at 64x64, ~9.3 GB at 256x256 (one default tile), ~149 GB untiled at 1024x1024, ~73% of it per-conv im2col/pad/raw buffers.
  - [x] A7. Pipeline + `examples/QwenImage` CLI: read the `model_index.json` folder,
        tokenize, encode, free the text encoder, denoise, decode, save PNG WITH ALPHA
        (check that `SaveImageFromVolumeIntoFile` writes RGBA). `--width --height
        --steps --seed --int8/--int4 --opencl`. End-to-end pico parity with fixed
        initial latents (torch RNG is not reproducible). Test sizes: 64x64 with the
        pico checkpoint on the dev box (extend the generator with a 64x64 case);
        real weights on the GPU box, a 256x256 smoke test first, then 1024x1024 and
        beyond, compared per step against diffusers with the same initial latents.
        Done: class `TQwenImage21Pipeline` (neuralpretrained.pas: `TokenizePrompt`, `EncodeTokenIds`, `MakeInitialLatents`, `Denoise`, `DecodeLatents`, `GenerateFromEmbeds`, `Generate`, `RoundDownImageSide`, events `OnPhase`/`OnStep`), `TNNetFlowMatchEulerScheduler.CreateFromDiffusersConfig` + `ShiftForImageSeqLen`, RGBA PNG in `SaveImageFromVolumeIntoFile` / `LoadImageIntoVolume(.., pWithAlpha)`; `examples/QwenImage` (`--model --prompt --output --width --height --steps --seed --int8|--int4 --int8-input --vae-tile S[,T] --token-ids --drop-count`, no `--opencl`); tests `TestQwenImage21Pipeline*`, `TestQwen3VLEncodeRefusesOutOfVocabIds`, `TestFlowMatchFromDiffusersConfig`, `TestSaveImageRGBAPngRoundTrip`; fixture `tiny_qwenimage21_pipeline_64_io.json`.
        Follow-ups: `cthreads` in the example's uses clause (a26c3f72; without it the
        first worker thread aborted with runtime error 232); parallel forward on every
        core (74fa882d: `Parallel` / `MaxThreads` on the pipeline, transformer and VAE,
        `PrepareInferenceThreads`, CLI `--serial` / `--max-threads N`; bit-identical to
        serial). First real run (user, 2026-09-24, real Qwen/Qwen-Image-2.1, 12 threads):
        256x256, `--int4`, 10 steps -> the image shows a fox. Step time 16.4 s int4,
        34.3 s int8 (was ~162 s int8 single-threaded); load transformer 148 s int4 /
        64 s int8; VAE decode 31 s.
  - [x] A8. Docs: README entry for Qwen-Image-2.1 text-to-image (no tested-configuration
        list, user decision 2026-10-04; measured numbers labelled as the user's L4 run).
        Build line and binary path: `cd examples/QwenImage && lazbuild -B QwenImage.lpi`
        writes `bin/x86_64-linux/bin/QwenImage` at the REPO ROOT (the A7 report gave a
        path inside `examples/QwenImage`, which is wrong).
        DONE: `examples/QwenImage/README.md` (model, build, one-shot + REPL, options, the L4 run, limitations; no tested-configuration table, user decision) + entries in `README.md` and `examples/README.md`.
  - [x] A9. Keep-loaded mode + REPL: `TQwenImage21Pipeline` gains `LoadComponents` /
        `UnloadComponents` so `Generate` reuses loaded weights (the one-shot CLI keeps
        today's load-and-free order). The text encoder is built once for a maximum
        prompt length (padding is exact, A2); prefix/step/VAE nets are rebuilt per
        image by borrowing weights; activations are freed between images, weights
        kept. `examples/QwenImage` follows ChatTerminal: `-p "prompt"` (or
        `--token-ids`) is a one-shot run that keeps the load-and-free order;
        `--prompt` is removed (no alias). Without either, the REPL starts: one
        prompt per line from stdin (so a piped file is a batch), output files
        numbered from the `--output` base (`out.png` -> `out_0001.png`, ...),
        `/size WxH`, `/steps N`, `/seed N` (else the seed increments),
        `/tile SIZE[,STRIDE]`, `/quit`; weights stay resident between prompts;
        print resident memory at startup (user decisions, 2026-10-03). Test: two
        prompts through one loaded pipeline equal two one-shot runs (pico). Estimated
        resident at 1024x1024: ~19 GB int8, ~16 GB int4 (test boxes have 50-150 GB).
        Removes the per-image loads measured on the first real run: text encoder
        49-67 s, transformer 64 s int8 / 148 s int4.
        DONE: `LoadComponents` / `UnloadComponents` / `ComponentsLoaded` and the
        `Loaded*` properties; the text encoder keeps a 1-token weight owner and
        builds a borrowing twin per prompt (bit-identical to one-shot, no length
        cap) instead of a max-length net; the tokenizer stays loaded;
        `TQwenImage21Transformer.ReleasePasses` frees the per-image nets and the
        prefix K/V; format/int8/OpenCL settings refuse changes while loaded. REPL
        numbers outputs past existing files, `.png` when the base has none,
        prints the prompt per image, sides 32..8192. Tests
        `TestQwenImage21PipelineKeepLoaded` (FP32 + int8 encoder, recovery after
        a failed image), `...UnloadComponents`, `...OpenCLKeepLoaded`. Computed resident memory at
        1024x1024 int4 during the VAE decode: ~16 GB OpenCL (transformer codes
        ~4.4 + VAE weights 1 + 256-px tile buffers ~10.7), ~24-26 GB host; int8
        ~18.7 GB OpenCL (tight on a 24 GB L4; `/tile 128` lowers it). Unmeasured.
        Follow-ups:
        - [ ] OpenCL errors only print (`FErrorProc`); nothing lets the pipeline
              detect a failed allocation, so a REPL image can be wrong yet
              reported as written. Needs a counting error hook on the armed nets.
        - [ ] `--steps 1` raises EInvalidOp in
              `TNNetFlowMatchEulerScheduler.SetSigmas` (shift_terminal stretch
              divides 0/0; diffusers yields NaN). One-shot too.
  Phase B — speed (after a first real measurement on the GPU box):
  - [ ] B1. OpenCL for the transformer step pass. Int4 is required; int8 comes along
        with it (the tiled OpenCL kernels share one layout). The text encoder and
        the VAE stay on the CPU in this task. Target GPUs have 24-100 GB of memory.
        Runs BEFORE B0 and A9 (user decision, 2026-09-25).
        Already in the library: `cai_dot_product_int4_tiled` / `_int8_tiled`
        (many-token projections, used by --prefill-window); resident-codes sharing
        (`EnableOpenCLInContextOf`, `BorrowOwnerOpenCLCodes`); OpenCL paths for
        TokenLayerNorm, ChannelMulByLayer, HeadRMSNorm (via TokenRMSNorm), axial
        RoPE (via M-RoPE), DeepConcat, Sum, SwiGLU.
        Estimate (not measured): 1024x1024 step ~57 TFLOP of projections + ~18
        TFLOP of attention, limited by arithmetic, so int4 saves GPU memory
        (~4.4 GB of weights vs ~7 GB int8), not time.
        First GPU run (user, 2026-09-30, NVIDIA L4, tip dd2c2971, before B1d):
        256x256 --int4, 40 steps: 2.2 s/step (CPU int4 16.4 s/step), denoise
        91.7 s; load transformer 240.3 s (CPU int4 load 148 s); load text
        encoder 39.8 s; VAE decode 33.3 s (CPU); peak RSS 8.8 GB. 2.2 s/step is
        ~1.6 TFLOPS of projections, far below the L4, so host layers,
        transfers and synchronisation dominate at 256 tokens.
        Second GPU run (user, 2026-09-30, L4): 1024x1024 --int4, 10 steps:
        36.3 s/step (16.5x the 256x256 step for 16x the tokens: linear, so
        attention does not dominate; ~75 TFLOP/step, ~2 TFLOPS achieved),
        denoise 370.4 s, VAE decode 528.2 s (CPU, now the largest cost), load
        transformer 245.2 s, peak RSS 10.9 GB. B1f runs before B1d2 to tell
        the int4 projection kernel, the host layers and the transfers apart.
    - [x] B1f. Per-layer time profile of one GPU step (user's GPU box), to
          measure where the step time goes before further kernel work.
          DONE: `examples/QwenImage --profile` (TNNet.LayerProfiling drains the
          net queue and the layer's output queue after every layer that
          enqueued OpenCL work; per-role and per-class tables with OpenCL
          forwards and host<->OpenCL transfer counts/MB). User run pending:
          `--int4 --width 1024 --height 1024 --steps 2 --profile`.
          L4 profile (user, 1024x1024 int4, b03fae2f): 34.6 s/step; transfers
          22.5 GB up + 12.3 GB down per step at only ~2 GB/s effective, so about
          half the step is host layers + transfers (Q/K RoPE 5.2 s, Norm2 2.6,
          QKV concat 1.9, Norm1 1.5, Modulated1/2 2.2, BlockInput 1.0, input
          uploads in Q/K/V/GateUp ~4). Attention ~7.8 s/step (~1.1 TFLOPS, the
          weakest OpenCL kernel). Projections with a resident input are fine
          (OutProj 13.7 ms/block, ~10 TFLOPS). Image-in net 1.16 s/step on the
          CPU for ~2 GFLOP (odd; look later). Order chosen: B1d2 -> B1d3 ->
          B1d4, then B2 (VAE 519 s), then the B1c deferred attention items.
    - [x] B1g. Transformer load time with OpenCL (240 s vs 148 s on the CPU):
          find the extra ~92 s (suspect: the host-side int4 repack loop in
          `PrepareInt4DotCL`, one nested call per weight pair; unverified).
          Investigated 2026-10-02 (real-shape block, this box): the extra time
          is the host repack in `PrepareInt4DotCL` (2.4-2.9 s/block, ~76-92 s
          for 32 blocks; byte-at-a-time writes with a NumAs stride, each on a
          new cache line). The upload is ~2 s total. A 64-row tiled loop gives
          byte-identical output ~3.5-4x faster single-threaded. The CPU int4
          load is slow for a second reason: `QuantizeWeightsInt4` (int8 ->
          dequantize -> int4, serial, scalar `QuantizeRow` with per-element
          IsNan/IsInfinite) costs 2.4-3.9 s/block (~80 s). Neuron object
          construction costs ~0.6-0.8 s/block (also the text encoder).
      - [x] B1g1. Tiled + threaded repack in PrepareInt4DotCL and the int8
            transpose in PrepareInt8DotCL (byte-identical layout).
            DONE: 64-row tiles (TransposeByteTile; TNNetVolumeQuant8
            .CopyRowsTransposedTo, TNNetVolumeQuant4
            .CopyRowsAsPairedTransposedTo) fanned by TQuantRowsTransposeFan.
            Arming one int4 block on PoCL 2.8-4.6 s -> 0.7-1.1 s (32 blocks
            ~90-145 s -> ~23-36 s, extrapolated). Found: the shared fNTL pool
            is 1 thread on Linux (FPC 3.2.2 TThread.ProcessorCount = 1), so
            every fNTL fan runs inline; fix awaits the user's decision.
      - [x] B1g2. Quantize to int4 directly in the loader (FP32/BF16 rows ->
            Q4_0 in the threaded row fan), branch-free QuantizeRow.
            DONE: TNNet.BeginInt4QuantImports + LoadLlamaLinearWeights'
            DirectInt4FromRows route (Qwen-Image transformer only); QuantizeRow
            branch-free with an AVX2 path (~15x, output identical for finite
            input; tiny-max blocks no longer raise EOverflow). One real-shape
            int4 block loads in 0.8-2.3 s instead of 3.9-7.5 s (this box).
            Int4 results change slightly (one rounding instead of two).
            Follow-up:
            - [x] ChatTerminal / LLM --int4 quantize streamed rows straight
                  to int4 too (NeuralImportInt4FromRows, set by TChatEngine;
                  LoadLlamaLinearWeights and the tied LM head open the import;
                  BuildFromPretrained refuses partial imports). The borrowing
                  twin reads no checkpoint; the fallback twin takes the same
                  route. Layers that cannot stream rows keep the int8 -> int4
                  sweep; the MoE banks have no int4 forward and stay int8.
      - [x] B1g3. Skip creating the four training volumes per neuron in
            inference-only builds.
            DONE differently: the cost was TVolume's per-volume TFormatSettings
            copy (~45 managed strings per constructor/destructor), not the
            training volumes. The field is gone (TNNetVolume 432 -> 56 bytes);
            formatting reads GetDefaultNumericFormat at conversion time.
            Neuron Create+Free 12.4 -> ~1.1 us; real-shape block build median
            ~830 -> ~170 ms (~21 s per 32-block transformer load; text encoder
            ~15 s, extrapolated). Skipping the training volumes would now save
            only ~0.7 s per load (not done).
    - [x] B1a. Read-only audit (2026-09-25, pico block 0 on PoCL, transfer trace;
          probe in the session scratchpad, not in the repo). Only 9 of the
          block's 31 layers run on OpenCL (6 projections, QNorm/KNorm, SwiGLU);
          the activation chain leaves OpenCL memory ~6 times per block, ~1 GB of
          transfers per block at 1024x1024 (estimate). Parity where OpenCL runs:
          int8 block 2.7e-7, int4 tiled GEMM 1.1e-6, HeadRMSNorm + forced axial
          RoPE 7.2e-7. PARITY-TEST TRAP: the CPU int4 path quantizes the input to
          int8, OpenCL keeps it FP32 (3.6e-2 apart, both correct), so B1 parity
          tests compare against an FP32-input oracle. Verified arming order that
          shares one copy of the codes: BlockStore[0].EnableOpenCL, then
          EnableOpenCLInContextOf(BlockStore[0]) for blocks 1-31 and the step net.
    - [x] B1b. All 32 blocks' int4/int8 codes resident in OpenCL memory, one
          DONE 88aad8ca (handle swap + RefreshResidentBias); GPU-by-default CLI
          with --cpu: 807d61c5.
          context, uploaded once. `SelectBlockWeights` swaps only handles: in
          `TDotProductSharedKernel` FCodesBuffer / FScalesBuffer /
          FBlockScalesBuffer (+ caps, FCodesBorrowed, FInt8Ready/FInt4Ready;
          retain new, release old), AND re-set the tiled OpenCL kernel's args
          4/9/10 (`PrepareTiled` binds them once; split-K and single-pass compare
          handles and re-bind by themselves). `PrepareForComputeBorrowingCodes`
          cannot be used per block (it re-creates the result and input buffers,
          403 MB for GateUp at 4096 tokens). Layer side:
          `TNNetLayerConcatedWeights.LinkWeightsFrom` refuses when FHasOpenCL, and
          the host tables (FQuantTable, FQuantTableInt4, FWeightOwner, FNeurons)
          must follow the swap for the CPU fallback. The prefix net stays on the
          CPU. Parity test: pico, PoCL vs FP32-input oracle.
    - [x] B1c. Tiled non-causal attention OpenCL kernel for
          `CachedForwardNonCausal` (query tile + key tile in local memory, online
          softmax, keys = text prefix + image rows). The decode kernel
          (`cai_sdpa_decode_split`) re-reads all K/V per query row: ~540 GB per
          block at 1024x1024. Keep the 32 blocks' prefix K/V resident once per
          prompt (today `AppendCacheRowsFrom` goes through the host and
          `UploadCache` re-uploads them every block of every step); the image
          rows' K/V can be read straight from the packed QKV input. `WillOpenCL`
          accepts FCachedForwardNonCausal when the new kernel fits (its own
          local-memory test, not `QueryTileFits`). Parity test: pico, PoCL vs CPU.
          DONE: `cai_sdpa_noncausal_tiled` (query tile x key tile in local
          memory, online softmax; PoCL 64x32 tiles, NVIDIA 48 KB computed 26x32);
          prefix K/V uploaded once per EncodePrefix and appended per block inside
          OpenCL memory (`NewCacheRowsOnOpenCL` / `AppendCacheRowsFromOpenCL`).
          Parity vs CPU 1.8e-7..3.0e-7 (layer), 6.0e-7 (pico velocity). Resident
          prefixes cost NumLayers x L x hidden x 8 B of OpenCL memory (~256 MB at
          L=256, ~1 GB at L=1024). Deferred until a GPU measurement:
          - [x] O accumulator in private registers (raises R on 48 KB devices).
          - [x] Register-blocked score and P.V loops (today 2 local reads per mad).
          - [x] Tree reduction for the running-max fold (one lane per row today).
          DONE (kernel rewrite): O in registers (3x2 float4 per lane), 3x2
          score and P.V blocking with float4 local loads, parallel row max and
          per-lane partial sums, one work-group per 128 head-dim columns,
          reqd_work_group_size(16,16,1); 48 KB tiles 26x32 -> 42x32 (computed),
          PoCL ~2.1x on one shape (not a GPU timing). L4 gain unmeasured.
    - [x] B1d. Layers that send the activations back to RAM (B1a), worst first:
      - [x] B1d1. `TNNetChannelMulByLayer` x4: the modulation operand is a host
            row (TNNetAddConstant has no OpenCL path; SplitChannels/Tanh follow
            their host source). Accept a host-only FLayerMul of Depth floats,
            upload it to a small per-layer buffer once per step.
            DONE: Gated1/Gated2 bind OutProj/Down and upload the host row (16 KB,
            blocking write) every forward; both TNNetSum layers follow onto
            OpenCL. Modulated1/2 stay on the host until B1d2. Follow-ups:
            - [ ] Non-blocking row write + a queue Finish at the end of
                  TNNet.Compute when the last layer ends on the host (worth it
                  after B1d2, when these writes are the only mid-block waits).
            - [ ] Run the modulation chain (SplitChannels, AddConstant, tanh) on
                  OpenCL so the four rows stay resident for all 32 blocks.
            - [ ] Shared "bind or upload the operand" helper for
                  TNNetChannelMulByLayer and TNNetCellMulByCell.
            - [ ] SE / CBAM / DiT importers now take this OpenCL path with no
                  OpenCL parity test.
      - [x] B1d2. `TNNetTokenLayerNorm` x2: FShouldOpenCL pinned false
            (deliberate: OpenCLForwardBenchmark ~0.54x standalone), Compute
            downloads before WillOpenCL, no residency clause, no bind/keep, no
            OpenCLOutputBuffer. Mirror TNNetTokenRMSNorm; one work-item per token
            is weak at depth 4096.
            DONE: TNNetTokenLayerNorm binds a resident source and keeps its
            output (FShouldOpenCL stays pinned for host sources);
            cai_volume_norm runs one segment per work-group (cai_token_norm
            removed; TokenRMSNorm/HeadRMSNorm share it); TNNetLayer
            .PrevOutputOnOpenCLSameSize replaces 5 copies. Norm1/2,
            Modulated1/2 on OpenCL; Q/K/V/GateUp no longer upload, Norm2 no
            longer downloads (~10.7 GB/step less at 1024x1024, computed).
            Follow-ups:
            - [ ] Shared parent class for the four norm layers' identical
                  FTokenNormCL / EnableOpenCL / WillOpenCL / output overrides.
            - [ ] Forced-OpenCL TRAINING through TokenLayerNorm/TokenRMSNorm
                  never fills FNormalized/FInvStd (FInvRMS): backprop reads
                  stale values (pre-existing). TNNetRMSNorm.WillOpenCL has no
                  trainable exclusion.
      - [x] B1d3. `TNNetAxialRotaryEmbedding` x2 (via TNNetMRotaryEmbedding):
            Compute downloads before WillOpenCL, no residency clause,
            `TNNetMRoPECL.Rotate` has no bind/keep and uploads the whole angle
            table every forward (32x redundant across heads). Add bind/keep as
            TNNetRoPECL.Rotate has; keep the table resident, re-upload only when
            positions change.
            DONE: TNNetMRotaryEmbedding binds a resident source and keeps its
            output (no size test: growing decode); the angle table is one
            head's worth (cai_mrope FHalfTile) and stays resident, re-uploaded
            only when positions/offset change. QRope/KRope, the QKV concat and
            the Attn input stay in OpenCL memory (~23.6 GB/step less at
            1024x1024, computed). Follow-up:
            - [ ] Qwen2-VL/Qwen3-VL decode: M-RoPE is per head there and the
                  positions change every token, so the blocking table write
                  waits ~900 times per token (no worse than before; a
                  non-blocking write would be safe in current graphs).
      - [x] B1d4. Block boundary: TNNet.Compute downloads the last layer and
            TNNetInput re-uploads it every block. Copy block i's output buffer
            into the next pass's input inside OpenCL memory; download only after
            block 31.
            DONE: TNNet.KeepLastOutputOnOpenCL (a full forward drains the
            queue instead of downloading the last layer) +
            TNNet.ComputeFromLayerOutput (TNNetInput.CopyNextInputFrom: its
            next Compute copies the source buffer with clEnqueueCopyBuffer on
            the input's queue).
            PredictVelocity: 1 upload + 1 download of the activation per step
            instead of 32 + 32 (~4.2 GB/step less at 1024x1024, computed); 31
            copies inside OpenCL memory (64 MiB each).
      TNNetSum and TNNetDeepConcat follow once these are fixed.
    - [ ] B1e. Pipeline wiring + `examples/QwenImage --gpu` (int4 and int8). User
          runs on the GPU box: 256x256 first, then 1024x1024 and larger; record
          s/step and peak GPU memory.
      Risks from B1a (not measured): the tiled int4/int8 OpenCL kernel streams
      the whole weight matrix once per 16-token tile (~35 GB of int4 codes per
      block at 4096 tokens unless L2 absorbs it; a wider token tile is the likely
      fix); if a device rejects the tiled kernel, int4 falls back to split-K
      (correct, very slow); kernel int32 indices overflow past ~87k tokens; the
      GateUp result buffer (403 MB at 4096 tokens) can hit
      CL_DEVICE_MAX_MEM_ALLOC_SIZE at larger images.
  - [ ] B0. CPU many-token int8/int4 projections. Evidence: at 256x256 a step is
        34.3 s int8 and 16.4 s int4 (~2.1x, the weight-byte ratio), i.e. ~105 GFLOPS
        and ~53 GB/s: the projections stream all ~7 GB of block weights once per
        TOKEN instead of reusing each weight row across the 256 tokens. Step 1
        (read-only): which kernel a many-token `TNNetPointwiseConvLinear` int8/int4
        forward runs in inference-only low-memory mode, and whether the
        `--prefill-window` batched path (27B TTFT 341 -> 53.9 s) can be reused; report
        + proposed fix. Step 2: the fix, after approval. At 1024x1024 (16x the tokens)
        today's rate is ~9 min per step.
  - [ ] B2. VAE decode speed: 3x3 convolutions at 1152 channels.
        Estimate (not measured): a CPU VAE decode at 1024x1024 is ~8 min (16x the
        31 s measured at 256x256), so it is the next bottleneck once B1 lands.
    - [x] B2a. Read-only audit (2026-10-01, pico VAE armed on PoCL: all 44
          convolutions ran on OpenCL, clamped image 2.5e-6 vs CPU). Limits:
          the FP32 GEMM is the naive `cai_dot_product` (one work-item per
          output); every padded 3x3 conv downloads its source, pads on the host
          and uploads (`ShouldBindPrevOutputAsIm2ColSrc` needs FPadding = 0);
          Reshape, DeMaxPool, GatherChannels, PixelShuffle, the mid SDPA and its
          one-source DeepConcat have no resident path, so the residual stream
          drops to RAM after attention and at every upsampler. Memory with
          today's per-layer buffers: 3.4 GB at 128 px tiles, 10.7 GB at 256,
          22.8 GB at 384; untiled 1024 needs ~155 GB and overflows int32 im2col
          indexing (2.72e9 elements). CPU decode measured ~56 GFLOPS (29 TFLOP
          with 128/96 tile overlap in 519 s).
    - [x] B2b. Arm OpenCL on the VAE in the pipeline
          (`TQwenImage21VaeDecoder.EnableOpenCL`, sized nets in one context).
          DONE: the CLI arms it by default (also with FP32 transformer
          weights); closing "Compute :" line names where the VAE ran;
          --profile prints a VAE table per tile shape. Follow-ups:
          - [ ] FP32 weight sharing across the tile-shape nets (each uploads
                the 1 GB of weights again).
          - [ ] Drop the armed nets' host FConcatedWeights/FConcatedWInter
                copies (~2 GB) after the upload.
          - [ ] One OpenCL context per pipeline (today the transformer and the
                VAE each build the program per image).
          - [ ] Parallel scheduler + OpenCL on PoCL is 20-90x slower than
                --serial (CPU oversubscription with PoCL's threads; worker
                count 4 -> 2 gives 32 s -> 3.3 s). Measure --serial vs default
                on the L4; cap the workers for CPU OpenCL devices.
    - [x] B2c. FP32 tiled GEMM OpenCL kernel (twin of
          `cai_dot_product_int8_tiled`), used by every FP32 conv/FC with >= 16
          columns.
          DONE: `cai_dot_product_tiled` shares `cai_dot_product_tiled_body`
          with the int8 kernel; FP32 Compute takes it when FNumBs >= 16 and
          FNumAs >= 64 (small-row GEMMs such as gram / attention P.V stay on
          `cai_dot_product`). GPU speed unmeasured.
    - [x] B2d. Padding-aware OpenCL im2col: bind the unpadded resident source.
          DONE: `cai_im2col` / `cai_im2col_h` take InSizeY and Padding and
          write zeros for out-of-source taps; a padded conv binds a resident
          source (no download + host pad + upload). Pico VAE: 26 of 40 3x3
          convs move 0 bytes, 14 still read a host source until B2e (~22 of
          ~39 GB per 1024x1024 decode removed, computed). OpenCL binaries must
          be rebuilt with the new neural.cl (kernel arguments changed).
    - [x] B2e. VAE residency gaps: Reshape alias, nearest-2x upsample kernel,
          DupUp gather/pixel shuffle bind+keep, SDPA output.
          DONE (inference only): TNNetReshape offers its source's OpenCL
          buffer when the Size matches (alias, no copy; MoveOutputToRAM copies
          from the source's host copy); `cai_upsample_gather` (replaces
          `cai_pixel_shuffle`, integer indexing, no index map) serves
          TNNetDeMaxPool, TNNetUpsample and TNNetPixelShuffle forwards (bind a
          resident source, keep the output); TNNetGatherChannels is now a
          TNNetSplitChannels descendant (cai_split_channels). Pico VAE: all 40
          3x3 convs bind; per decode 20 up / 17 down -> 2 up / 1 down besides
          the image download. 1024x1024 at 128/96 tiles: ~17.6 GB -> ~0.16 GB
          of activation transfers (computed). Left on the host: the mid-block
          SDPA (downloads Q|K|V, the out-projection uploads; softmax on host).
          TNNetSum needed no change. Follow-up:
          - [ ] Mid-block attention in OpenCL memory (device softmax, or the
                VAE through TNNetFusedSDPA's non-causal tiled kernel).
    - [ ] B2f. Big tiles / untiled decode. Plan (2026-10-03, computed from the
          code and the real config, not measured; dims 1152/1152/1152/576/288/
          144, 1.06 TFLOP per 256-px tile; 1024x1024 = 29.2 TFLOP at 256/192,
          26.5 at 512/384, 17.0 untiled). OpenCL bytes per sized net at a
          256 tile / 512 / untiled 1024: today 10.4 / 38.5 / 151 GB (im2col
          6.1 GB at 256); after B2f1 4.2 / 13.6 / 51.4; after B2f2 3.5 / 11.0 /
          41.0; after B2f3 ~1.3 / ~1.9 / ~4.7. Host today 9.9 + 2.0 GB at 256.
          Corrections to earlier notes: the 1 GB FP32 upload is inside the net
          (first forward) and small; blending is milliseconds; the ~22 s outside
          the nets is most likely host-side arming per tile shape
          (`TNNetLayerConcatedWeights.EnableOpenCL` calls `AfterWeightUpdate`
          twice: 1 GB concat + scalar transpose + output-sized bias fill per
          call) plus ~10 GB of zero-filled host volumes per build. Untiled 1024
          also overflows on the host (`TVolume.ReSize` integer size;
          `FInputPrepared` 2.72e9 elements). In the REPL the transformer stays
          resident, so the VAE has ~10-14 GB on a 24 GB L4.
          - [x] B2f0. Phase timers in `TQwenImage21VaeDecoder` under --profile
                (build, PrepareInferenceThreads, arming split into
                AfterWeightUpdate vs buffers, forwards, blend, release) and
                `TNNet.OpenCLBufferBytes()`. No behaviour change. L4 run with
                --profile and --serial.
                DONE: per-shape phase table + a decode-wall split line; arming
                weight prep via `TimedAfterWeightUpdate`; OpenCLBufferBytes from
                `clGetMemObjectInfo` (partial: FDotCL + overriding classes; the
                VAE's classes are covered). Pico shows no real-size signal;
                waiting for the user's L4 run.
          - [x] B2f1. Implicit-GEMM conv on OpenCL (FP32, inference-only):
                gather stage in `cai_dot_product_tiled_body`, naive twin for
                Cout < 64 (conv_out); lazy `FInputBufferBs`; inference-only
                spatial convs stop sizing host `FInputPrepared`. Int8 keeps
                im2col. Fixes the int32 overflow.
                DONE: `cai_conv_implicit_tiled` (BIsConv + `cai_tiled_stage_b_conv`,
                same Bs layout, equal to explicit on PoCL) and the untiled
                `cai_conv_implicit`; `TNNetConvolution.ShouldOpenCLImplicitConv`;
                `NEURAL_OPENCL_IMPLICIT_CONV=0` / `SetOpenCLImplicitConv` for an
                A/B. Pico VAE OpenCL bytes per net 12.6 -> 4.7 MB (-63%), image
                byte-identical. Computed at a real 256 tile: -6.1 GB OpenCL and
                -6.1 GB host per net. GPU speed unmeasured. Deferred: int8 convs
                keep the host column matrix; `SetTrainable(false)` after the
                build keeps it too; the CPU path still overflows untiled 1024.
          - [x] B2f2. Per-row bias in OpenCL (`Bias[row]`); lazy host
                `FBiasOutput` / `FOutputRaw`.
                DONE: every dot-product kernel reads `Bias[row]`; a conv's
                OpenCL bias buffer is Cout*4 bytes (`FNeuronBias`); a wrong-size
                bias raises. `TNNetConvolution.PrepareHostOutput` sizes host
                `FOutputRaw` and the per-position `FBiasOutput` on the first host
                forward, so a fused OpenCL forward holds neither. Pico VAE 64x64:
                OpenCL 4.5 -> 3.6 MB, host 5.8 -> 4.0 MB, image byte-identical.
                Computed at a real 256 tile: -0.65 GB OpenCL, -1.3 GB host per
                net. Deferred: `TNNetGroupedConvolutionLinear` keeps its
                per-position buffers; host buffers built once stay allocated.
          - [x] B2f3. OpenCL output buffer reuse by liveness (opt-in per TNNet;
                reuse only when every consumer of the producer is an ancestor;
                single in-order queue; last layer pinned).
                DONE: `TNNet.ShareOpenCLOutputsByLiveness`, planned when armed;
                on for the VAE nets (`NEURAL_OPENCL_SHARE_OUTPUTS=0` turns it
                off). Adopters: conv, identity/SiLU, Sum, DeepConcat, Split/
                Gather, TokenRMSNorm, DeMaxPool, PixelShuffle. Forwards from
                FromLayerIdx > 0 raise on a sharing net. Pico VAE 4x4 latent
                3.78 -> 1.07 MB OpenCL, images byte-identical. Computed: 256
                tile ~3.5 -> ~1.4 GB per net, 512 ~11 -> ~2.7, untiled 1024
                ~41 -> ~7.7 GB (the ancestor rule keeps the DupUp shortcut in
                its own slot; a serial-only index-order rule or extra scheduler
                edges would give ~4.9 GB untiled; not done, user decision).
                Not adopted yet: RMSNorm, LayerNorm, TokenLayerNorm, RoPE,
                fused SDPA, GLU.
          - [x] B2f4. Host `FOutput` sharing by liveness (exact-size aliases,
                opt-in).
                DONE: `TNNet.ShareHostOutputsByLiveness` (same planner as B2f3,
                exact-size slots, `TNNetVolume.ShareDataWith`/`UnshareData`);
                on for the VAE nets (`NEURAL_SHARE_HOST_OUTPUTS=0` turns it
                off); inference only (any trainable layer unshares). On armed
                nets a shared layer drains the queue only while a non-blocking
                upload is pending. Pico VAE host MB 1.3 -> 0.7 (OpenCL) and
                3.1 -> 2.5 (CPU), images byte-identical. Computed host FOutput
                per net: 256 tile ~2.4 -> ~0.56 GB, untiled 1024 ~38 -> ~9 GB.
                Follow-ups: the net build still sizes every FOutput before the
                first forward unshares them (untiled 1024 build peak ~38 GB
                host); exact size and the DupUp branch limit the saving; the
                drain cost on the L4 is unmeasured.
          - [ ] B2f5. Nearest-2x + 3x3 conv fold as four 2x2 phase convs on
                OpenCL (-17% of decode FLOPs); CPU path unchanged.
          - [ ] B2f6. Untiled path: memory estimator vs the device limits,
                `--vae-tile 0`; L4 runs 256/192 -> 512/384 -> untiled in the REPL;
                then the default decision with the user.
          Side note: `FInputBufferBs` (READ_ONLY) is written by cai_im2col and
          `FResultBuffer` (WRITE_ONLY) is read by consumer kernels; works on
          NVIDIA/PoCL, undefined per spec; pool buffers must be READ_WRITE.
  - [ ] B3. Optional guidance (negative prompt) with its own prefix cache.
  Phase D — example usability (`examples/QwenImage`):
  - [x] D1. Defaults int8 weights and 18 steps; `--fp32` asks for FP32.
  - [x] D2. Step line rewritten in place (`#13` + `ESC[K`) when stdout is a
        terminal; one line per step when it is a pipe or a file.
  - [x] D3. `/repeat N PROMPT` in the REPL and `--repeat N` with `-p`: N images,
        consecutive seeds, prompt encoded once.
  - [x] D4. `GPU %` column (forward count share) and a count/time share summary
        line in `TNNet.LayerGroupTimingReport`.
  - [x] D5. QwenImage per-image stage table (CPU/OpenCL, wall, % of image),
        `--stats`, host<->OpenCL MB per step and per VAE tile, peak OpenCL
        bytes, `/profile` and `/stats` REPL toggles.
  Phase C — editing and reference images (plan from source reading of
  diffusers 0.41.0.dev0 `pipeline_qwenimage21.py` / `transformer_qwenimage21.py`
  / `autoencoder_kl_qwenimage21.py` and transformers 5.17.0 `modeling_qwen3_vl.py`;
  order C2 -> C2b -> C1a -> C1b -> C3a -> C3b, C4 decided after C3a). diffusers has
  one pipeline for text-to-image and editing (`image=`): no strength, no mask, no
  second-image input; masks and circles are extra input images. Condition images
  are resized with our own resize, not PIL Lanczos (accepted divergence).
  - [x] C2. VAE encoder: conv_in 4->base, 5 residual down blocks, mid
        (res, attn, res), RMS norm, SiLU, conv_out -> 2z, quant_conv 1x1, posterior
        mean, `(z - mean)/std` folded into quant_conv. Trap: the AvgDown3D shortcut
        (blocks 1-3 pad a zero frame in front: channel 2c+ft, even channels 0).
        `time_conv` unused for one frame. The pico fixture already has the weights.
  - [x] C2b. SDEdit img2img: `--image FILE --strength S` (default 0.6) VAE-encodes
        the image, `latents = sigma[t_start]*noise + (1 - sigma[t_start])*x0` with
        `t_start = int(N - min(N*strength, N))` on the shifted sigmas (Qwen-Image v1
        img2img formula; 2.1 has no img2img pipeline), denoises from t_start. Error
        when no step remains. Re-styles; does not follow edit instructions.
  - [x] C1a. Qwen3-VL vision tower: patch 16 (the 2 temporal kernel slices summed),
        learned 48x48 position table bilinear (align_corners), 2-D RoPE (rotate-half
        -> interleaved permutation), 27 pre-LN blocks, bidirectional attention via the
        `TNNetFusedSDPA` cached path (no 1 GB score map), 2x2 merger + DeepStack
        mergers (layers 8/16/24). Preprocessing: composite over white, mean/std and
        max_pixels from `preprocessor_config.json`. Pico fixture regenerated with
        vision patch 16, depth 3, deepstack [0, 2].
  - [x] C1b. Text encoder with images: vision embeddings spliced into the
        `<|image_pad|>` rows, DeepStack features added after decoder layers 0-2,
        interleaved M-RoPE sections, N-image `get_rope_index` positions, edit template
        `<image1><|vision_start|><|image_pad|><|vision_end|> <image2>...`, image-pad
        mask output.
  - [x] C3a. Transformer prefix with condition images: each image slot expanded x4
        and filled with `img_in(VAE latents)`, t = 0 modulation, block-causal mask
        (causal text, bidirectional per image) via a per-row key limit in
        `TNNetFusedSDPA.ComputeCachedRows`; `PrepareStepPass` must place the target
        RoPE frame after the images (today it assumes a text-only prefix).
  - [x] C3b. Edit pipeline and example: target size from the last condition
        image's aspect, per-image resize, RGBA to the VAE and white-composited RGB to
        the vision tower; `--image FILE` repeatable, `/image FILE` and
        `/images clear` in the REPL.
  - [ ] C4. Prefix pass on OpenCL, int8 prefix K/V and a windowed text-encoder
        prefill. Estimates at 1024^2: prefix ~4.1k tokens / ~4.3 GB FP32 K/V for one
        reference image, ~41k tokens / ~43 GB for ten; text-encoder scores ~13.6 GB
        unwindowed at ten. Decide scope after C3a with an L4 measurement.
  Phase F — OpenCL residency of the step pass:
  - [x] F1. Modulation chain in OpenCL memory (L4 int8 1024^2 run 2026-10-04:
        1280 blocking 16 KB operand uploads per image, all modulation layers
        on the host).
        (a) `TNNetInput.ComputeOpenCL`: move the upload that `TNNetInput.Compute`
        does inline into its own ComputeOpenCL; it only puts FOutput into the
        OpenCL buffer and sets FOutputOnOpenCL. FOutputOnRAM stays true (the
        host copy is current; producers offer, consumers decide).
        (b) Consumers bind any source with FOutputOnOpenCL, whether or not it
        also has a RAM copy: first `TNNetSplitChannels.WillOpenCL` (ends with
        `Result := not FPrevLayer.FOutputOnRAM`); read-only grep of the other
        WillOpenCL predicates that fall back to the host when the source is in
        RAM before changing them.
        (c) `TNNetAddConstant` opcode in `cai_activation` (ParamA = constant),
        Compute through ComputeActivationOnOpenCL (also fixes its "-" count).
        Then tanh follows and `TNNetChannelMulByLayer` binds its operand.
        Doing (b) without (c) is worse (AddConstant would download).
        Tests: PoCL sentinel test that the split/tanh/AddConstant/ChannelMul
        chain binds (no activation upload), parity, counts.
  - [x] F2. Image-in net and output net on OpenCL (L4 run: image-in 880 ms +
        output 216 ms + "outside the nets" 126 ms per step; 64 MB up + 64 MB
        down per step). Arm FImageInNet and FOutputNet in the step net's
        context (EnableOpenCLInContextOf(FBlockStore[0], ...)), run block 0 as
        FStepNet.ComputeFromLayerOutput(FImageInNet.GetLastLayer()) and the head
        as FOutputNet.ComputeFromLayerOutput(FStepNet.GetLastLayer()), download
        only the velocity. Check first that a net whose weights come from a
        host-only build owner (FImageInOwner/FOutputOwner) can arm its own
        buffers.
  - [x] F3. Per-layer residency statistic in LayerGroupTimingReport, counted in
        TNNetLayer.RunProfiled only (zero cost without profiling): per layer,
        forwards whose source was bound in OpenCL memory / pulled to RAM /
        activation-uploaded / host forward with a resident-only source (stale
        suspect) / output left resident; plus a net-level "transfers outside
        layer rows" line from OpenCLProcessTransferTotals. Source lists built
        in SetLayerProfiling(true) (no alloc in Compute). PoCL tests incl. a
        deliberate stale-read layer.
  - [x] F4. VAE mid-block attention as one non-causal TNNetFusedSDPA reading the
        QKV slab (as the transformer block does: BeginIncrementalDecode(N)
        before AddLayer, TruncateCache(0) per tile pass), replacing the split /
        TNNetScaledDotProductAttention (host softmax, 4 uploads + 3 blocking
        downloads per pass) / one-input TNNetDeepConcat. Check
        FlashTilesFit at the VAE width on 48 KB local memory.
  - [x] F5. Norm1/Norm2 (non-affine TNNetTokenLayerNorm) re-upload gamma=1 /
        beta=0 on every block weight swap (20 MB per image): weightless variant
        or no re-upload when unchanged.
  - [x] F6. VAE weights shared across the tile-shape nets: each of the 4 nets
        prepares and uploads the same 945 MB of conv weights (weight prep
        11.3 s of the 26.2 s decode on the L4).
  Phase E — speed gap against other implementations:
  - [ ] E1. The Unsloth build of Qwen-Image-2.1 is reported to be at least 20x
        faster than ours (our L4 int8 1024^2, 10 steps, --profile, 2026-10-04:
        322 s total, 19.7 s/step, projections 15.0 s/step = ~3.8 TFLOPS, VAE
        decode 26.2 s of which 11.3 s weight preparation). Find why, stage by
        stage. First pin down the comparison: Unsloth artifact and runner (GGUF
        in ComfyUI / diffusers / other), quantization, steps (distilled or
        lightning LoRA?), CFG, resolution, GPU, and whether its time includes the
        loads. Then compare per stage: projections (tensor-core FP16/BF16 or
        int8 MMA via cuBLAS vs our OpenCL dot-product kernels), attention (flash
        attention vs cai_sdpa_noncausal_tiled), text encoder and VAE on the GPU vs
        ours on the CPU / per-tile-net arming, weight load. Output: a ranked list
        of the gaps with measured or estimated seconds each, and which are
        reachable in OpenCL (no tensor cores in portable OpenCL; NVIDIA-only
        inline PTX was an earlier idea).
        Done as a read-only analysis 2026-10-04: a 1024^2 step is ~66 TFLOP, so
        ~1 s/step needs tensor cores; our projections run at 3.8 TFLOPS (int8
        tiled GEMM bound by one-byte weight loads, 2x16 micro-tile), attention at
        3.0 TFLOPS. L4 clinfo (driver 580.82.07): no cl_khr_integer_dot_product,
        no cl_khr_fp16, no sub-groups, OpenCL C 1.2, 48 KB local memory, max
        single allocation 5.5 GB. Portable ceiling ~4-5 s/step.
  - [ ] E2. Generic masked flash attention (FP32) in TNNetFusedSDPA (user-
        authorized 2026-10-05). One tiled OpenCL kernel replacing
        cai_sdpa_noncausal_tiled, with a mask mode (none / causal / per-row key
        end / window), used for many query rows; cai_sdpa_decode_split stays for
        one or few rows. Users: Qwen-Image step pass, Qwen3-VL vision tower, LLM
        prefill (--prefill-window, ChatTerminal TTFT), the edit prefix (C4), the
        VAE attention (F4). Target: attention 2.9 -> ~0.6-1.0 s/step at 1024^2
        (unmeasured). Serial stages, each with a fresh read-only reviewer, the
        three suites and a commit:
    - [x] E2.0 Read-only design: mask encoding, tile shapes and register /
          local-memory budget (48 KB, 64K registers, OpenCL C 1.2, no
          sub-groups), dispatch rule tiled vs decode-split, test plan, benchmark
          spec. Brought to the user before coding.
    - [x] E2.1 Kernel (coder A): non-causal mode first, register tiling, vector
          loads, fewer barriers, 2-3 tile variants behind an environment switch;
          a small benchmark example timing the kernel at given shapes without a
          model. User times the variants on the L4.
    - [x] E2.2 (coder A continued): lock in the winning tiles; causal mode and
          the TNNetFusedSDPA dispatch for LLM prefill windows; parity on LLM
          prefill and Qwen-Image. User measures ChatTerminal TTFT and a Qwen-Image
          --profile run.
    - [x] E2.3 (coder B): per-row key ends (C3a edit prefix, unblocks C4) and
          int8 K/V tiles (ChatTerminal's default KV cache).
    - [x] E2.4 (coder B continued): head-dimension (Dk) splitting for wide
          single heads, then TNNetScaledDotProductAttention and TNNetDeepConcat
          on OpenCL (user request 2026-10-05): the generic
          TNNetScaledDotProductAttention (AddMultiHeadSelfAttention and ~90 other
          call sites, incl. the Qwen-Image VAE mid-block) runs its whole
          forward on the flash kernel, binds its source and leaves its output in
          OpenCL memory (today: Q.K^T and P.V on OpenCL, softmax on the host, 4
          uploads + 3 blocking downloads per pass, output never resident); the
          following TNNetDeepConcat then binds a resident source and runs on
          OpenCL (today 0% on the VAE nets). Its other modes (masks, causal,
          training/backprop) keep their current paths unless the flash kernel
          covers them. Resolves F4 without changing the VAE graph. One L4 VAE
          profile.
  - [ ] E3. Faster int8/int4 tiled GEMM for TNNetPointwiseConvLinear
        (ComputeResidentCodes -> RunTiledGemm; user-authorized 2026-10-06).
        Projections are 14.1 of 16.1 s/step on the L4 (GateUp 3.6 TFLOPS, Down
        4.0, Q/K/V/O 4.7-5.1); the kernel is latency-bound at ~12.5% of FMA peak
        (one-byte weight loads inside the FMA loop, int8 K loop with a run-time
        bound and no unroll). Serial stages, each with a fresh read-only
        reviewer, the three suites and a commit; the user times each on the L4:
    - [x] E3.1 GEMM fix 1 (fdde6fc6): a kernel-only benchmark example at the Qwen-Image
          projection shapes, then a constant-32 unrolled K loop, 4 rows per
          lane read as 4-byte codes, 4x16 micro-tile per lane (int8 and int4).
          Estimate: 16 -> 8-10 s/step.
    - [x] E3.2 GEMM fix 2 (451a6b5f): 128x128 register-blocked tiled GEMM (8x8 per lane,
          local-memory staging of codes and inputs). Estimate: 5-7 s/step.
          L4 measured 2026-10-06: int8 block 9.9-11.4 TFLOPS, Qwen-Image step
          16.1 -> 7.4 s.
    - [x] E3.3 (076c1d0a) Remove the _small grid (user-authorized 2026-10-06): on the L4
          it lost every measured window shape (1024/2560 rows x 2560 x 16/64
          tokens: int8 13-17% and int4 52-56% slower than the large grid).
    - [x] E3.4 (39e37080; L4 GemmBench: window shapes 1.1-12.8x, auto S best or tied everywhere) Split-K with plain FP32 partials for tiled code GEMMs with too
          few tiles to fill the compute units (user-authorized 2026-10-06):
          every measured window shape takes a flat ~0.2 ms (serial K loop over
          1-20 work-groups) against a ~10 us weight-read floor. GemmBench A/B,
          then the user measures ChatTerminal TTFT with --prefill-window.
    - [x] E3.5 (efd64173; L4 run pending) GemmBench FP16-activation option (user-authorized 2026-10-06):
          time the int8 code kernels with the B operand in FP16 (the existing
          _h kernels) next to FP32 at the Qwen-Image shapes, to test whether
          the block kernel (10-11 TFLOPS of ~30) is bound by FP32 activation
          traffic (64 MB at 4096 tokens > 48 MB L2). Measurement only.
- [ ] Jev-style typed decision models ("System One": a state plus typed questions in,
      calibrated option probabilities out, no text generated). Plan agreed 2026-10-18,
      branch a19. Scope: stage 0 (shared decision layer + `/v1/systemone`), stage 1
      (AnyJev L0 readout on any loaded decoder), stage 2 (Kev and Open-Jev: Qwen3.5 +
      LoRA + a small head). Laya (ModernBERT encoder + decision head) is LATER, not in
      this plan. One Opus agent per task, run serially, test + commit after each.
      Decisions:
      - Wire format = TypeSafe's `POST /v1/systemone` (OpenAPI 0.2.0 at
        api.typesafe.ai/openapi.json). Request `{model, state, questions:{id:{type,
        instructions, criteria}}}`; `state`/`instructions`/descriptions may be a
        string, an object or an array. `choice.criteria` = name -> description map
        (1..255, key order = option order, null description allowed);
        `score.criteria` = ordered array of level descriptions (1..10, levels 0..n-1);
        `noul.criteria` = optional `{true, false}`. Response `{model, answers, usage:
        {input_tokens, output_tokens}}`; choice `{type, choice, probabilities (by
        name), confidence}`; score `{type, score (= sum i*p_i), legend, probabilities
        (keys "0".."n-1"), confidence}`; noul `{type, noul}` only. Validation errors
        = HTTP 422 `{"detail":[{loc, msg, type}]}`; other errors `{"detail": str}`.
        `box` questions and image `assets` are rejected (422).
      - Confidence = Jev's published formulas on EVERY scorer, computed from the
        probabilities only: choice `(p_max-1/n)/(1-1/n)`; score
        `max(0, 1 - sum p_i|i-mode| / D)`, `D = (1/n) sum |i-(n-1)/2|`, mode = first
        argmax; both 1.0 when n = 1; all-zero p -> uniform. Noul has no confidence.
        Checks: score (0,0.57,0.43) -> 0.355; (0,0.95,0.05) -> 0.925.
      - Rendering of object/array values (state, instructions, descriptions): Kev's
        `api.render` (str/number/bool as text, list -> "- item" lines, dict ->
        "key: value" lines, nested values indented). One renderer for all scorers.
      - Special-token strings in caller text: rewrite `<|x|>` -> `<¦x¦>` before
        tokenizing (Kev's rule), on every scorer. A general "specials as plain text"
        tokenizer flag is a separate follow-up.
      - ChatServer always registers `/v1/systemone`. Without `--decision-adapter` it
        answers with AnyJev L0 on the loaded chat model (nothing extra is loaded).
        With `--decision-adapter <dir>` the server merges the adapter, reads the head
        and becomes decision-only: the chat endpoints return an error naming the
        reason. ChatTerminal does not change.
      - LoRA is merged in HF TENSOR space before the build (base + scale*B@A in FP32,
        rounded once), so the importer's row permutations (rotate_half q/k, the
        Qwen3.5 q/gate split, fused gate+up, fused z/b/a) apply unchanged and
        int8/int4 quantization follows as usual. No live (unmerged) LoRA path.
      - `head.pt` is read by `neuraltorchbin.pas` (extended for top-level scalar /
        string metadata). No Python conversion step for users.
      - Multi-question and multi-candidate scoring: prefill the shared prefix once,
        then per branch `TruncateTo(PrefixLen)` + `RestoreStateFrom(checkpoint)`,
        then feed the branch. Exact on attention-only and GatedDeltaNet hybrids; no
        new attention masks, no per-token RoPE positions.
      - AnyJev label prior: the content-free prior (states "N/A", "" and "[MASK]",
        alpha 1.0) is the default. The running-mean batch prior (stateful across
        requests) is a follow-up. AnyJev L1/L2 are out of scope (removed upstream in
        0.3.0, and both need labelled data).
      - GPT-AGI/OpenJev is not ported: AnyJev L0 covers it, and its first-token-only
        label scoring is a correctness bug.
      - Tests: pico fixtures with an HF/PyTorch float64 oracle made by
        `tools/*_fixture.py` scripts in the `~/x` venv under the 2 GB cap, plus
        model-free unit tests. Real-checkpoint checks run on the user's box.
      Reference facts (research 2026-10-18):
      - AnyJev L0 (`anyjev` 0.3.0 `readout.py`, `calibrate/permute.py`,
        `calibrate/contextual.py`): system prompt "You are a decision function. You
        will be given a state and one question. Reply with the answer label only: no
        words, no punctuation, no explanation."; user message lines "State:", state
        (or "(empty)"), "", "Question: <q>", then per kind: choice "Options:" + "A. opt"
        lines + "Answer with the letter only."; score "Pick the level that applies (the
        levels are ordered):" + "1. lvl" lines + "Answer with the number only."; noul
        "Answer Yes or No." / "Answer No or Yes.". Chat template with
        add_generation_prompt and thinking OFF; no template -> "{system}\n\n{user}\n
        Answer:". Encoded without special tokens. Labels: choice A..Z (max 26), score
        "1".."K" (letters if a digit is not one token), noul Yes/No. Each label = the
        first of `lab`, `" "+lab` that is ONE token, else error. Readout = full-vocab
        log_softmax at the last position, then softmax over the label logits.
        Rotations: choice cyclic shifts `perm[j] = (j+s) mod K`, noul both orders,
        score none. Per rotation p = p_raw / prior^alpha, clip 1e-8, renormalise; then
        map positions back to options and combine by log-mean (mean of log(clip
        1e-12), subtract max, exp, normalise). Content-free prior = mean over the 3
        empty-state probes per rotation, clip 1e-8, renormalise. Model-free oracle:
        raw (0.62,0.38), prior (0.70,0.30), alpha 1 -> 0.4115.
      - Kev (`jaredpalmer/kev`, HF `jaredpalmer/kev-0.8b|4b|9b|27b`): bases
        `Qwen/Qwen3.5-{0.8B,4B,9B}-Base` (frozen, LoRA r16 alpha32 on q/k/v/o,
        gate/up/down, in_proj_qkv/z/a/b, out_proj) and `Qwen/Qwen3.8-27B` (27B = full
        fine-tune, `Qwen3_5TextModel` keys without `model.` prefix, no lm_head).
        Adapter keys `base_model.model.layers.N.<module>.lora_{A,B}.weight`, F32.
        Tokens reused: state `<|fim_prefix|>`, question `<|fim_middle|>`, option open
        `<|box_start|>`, option close `<|box_end|>`, decide `<|fim_suffix|>`. Row =
        `[<|fim_prefix|>] + state` then per question `[<|fim_middle|>] + instructions
        + (<|box_start|> opt <|box_end|>)* + [<|fim_suffix|>]`; no BOS, no chat
        template. Options: noul `["no[: false desc]", "yes[: true desc]"]` -> answer
        p[1]; choice "name" or "name: desc"; score rendered level text. Head reads the
        FINAL hidden state (after the final RMSNorm): `z = (K h_</opt>) . (Q h_decide)
        / sqrt(256) / T`, Q and K = Linear(d -> 256) with bias. `head.pt` = torch.save
        dict: `head` (OrderedDict q.weight/q.bias/k.weight/k.bias, fp32), `temperature`
        (0.8B 2.3510958), `head_dim` 256, `base`, `base_revision`, `lora`, plus nested
        training metadata. State limit 65536 tokens (422 above it).
      - Open-Jev (`Zefan-Cai/Open-Jev`, MIT code; HF `ZefanCai/Open-Jev-2B|9B|27B-v1.1`,
        Apache-2.0): bases `Qwen/Qwen3.5-2B`, `Qwen/Qwen3.5-9B`, `Qwen/Qwen3.8-27B`
        (post-trained). Package `package/checkpoint/{adapter/, head.pt, model.json,
        temperature.json}`. LoRA r8 alpha16 on q/k/v/o, in_proj_qkv, out_proj, same key
        naming as Kev. Head = plain state_dict `weight [1,d]`, `bias [1]` (storage
        location may be 'cuda:0'). One chat-template prompt per candidate (user turn
        only, thinking off): "Context:\n{state}\n\nQuestion: {q}\n" + "Proposed
        answer: {option}\nIs this proposed answer correct? Answer Yes or No."; noul =
        ONE prompt "...Is the answer to this question yes? Answer Yes or No." with
        "\nYes means: ..."/"\nNo means: ..." appended to the question. Score s = head(
        final hidden state of the last token); choice/score p = softmax(s/T), noul
        p = sigmoid(s/T). Non-string values rendered with json.dumps(sort_keys).
  Stage 0 — shared decision layer:
  - [x] D0.1 `neural/neuraldecision.pas`: question/answer records, the Jev request
        parser + validator (422 detail list), the response builder, the confidence
        formulas, the value renderer, the special-token rewrite, and a restricted
        softmax over a set of logit ids (search `TNNetVolume`/`neuralvolume.pas` for an
        existing row softmax first). Model-free unit tests, including the 0.355/0.925
        confidence checks and JSON round trips.
        Done: 20 tests in `tests/TestNeuralDecision.pas`. Servers must call the
        STRING entry point of `ParseDecisionRequest`: only it can see option names
        over fpjson's 255-byte key limit. Open: `HFDecodeUnicodeEscapes`
        (`neuralhftokenizer.pas`) turns an escaped quote or backslash into the raw
        character, which breaks the JSON; the decision parser decodes escapes itself.
  - [x] D0.2 `TChatEngine` public decision primitives: tokenize + prefill a shared
        prefix and capture its state; score a branch from that state, returning the
        last-position logits and/or the final hidden state at requested branch
        positions; leave the engine consistent (`CachedTokens` = session contents, or
        cleared) on success and on exception. Tests on the Qwen3.5 hybrid pico and an
        attention-only pico: branch-after-restore equals a full fresh forward.
        Done: `PrefillDecisionPrefix` + `ScoreDecisionBranch`; `CachedTokens` stays
        exact (a later chat reuses what the session holds). Prefill windowing is
        shared with chat through `PlanPrefillLadder`/`FeedPrefillLadder`. Tests
        `TestChatDecisionBranch{Parity,PrefillWindow,OpenCL}` (OpenCL run on PoCL
        only, no real GPU).
  - [x] D0.3 ChatServer: `POST /v1/systemone` route (dispatch to the active scorer),
        decision-only mode switch for later stage-2 use, `--selftest` cases.
        Done: `TDecisionScorer` base (scorers take their engine/head in the
        constructor); no scorer -> 501, validation -> 422, wrong model -> 422 (the
        model name or `jev-latest`), scorer exception -> 500. 18 new selftest cases
        (run by hand: no test script runs ChatServer `--selftest`).
  Stage 1 — AnyJev L0 readout (no trained weights):
  - [x] D1.1 AnyJev L0 scorer (in `neuralanyjev.pas`, not `neuraldecision.pas`): prompt builder per kind, label
        token mapping, rotations, content-free prior, log-mean marginalisation, plus a
        `raw` level (one prompt, no prior). Jev mapping: choice options rendered
        "name: desc", score levels = descriptions, noul criteria appended as
        "Yes means: ..."/"No means: ..." lines. Port AnyJev's model-free tests
        (`tests/test_calibrate.py`).
        Done: `TAnyJevL0Scorer`, 15 tests in `tests/TestNeuralAnyJev.pas`. Prompts are
        grouped by state text (request state + the 3 content-free probes); each group
        prefills its longest common token prefix once, so the model reads exactly the
        full-prompt encodings (no sentinel split). Open: the Qwen3 closed-think append
        for cfChatML/cfQwen templates is unchecked against a real template (D1.2); a
        multi-token Yes/No label answers 500, not 422 (D1.3); batch prior,
        adaptive_shifts, canonical_order not ported.
  - [x] D1.2 Parity: `tools/anyjev_tiny_fixture.py` runs the real `anyjev` 0.3.0
        Decider (L0, content-free prior) on a pico Qwen-family decoder with a chat
        template; Pascal test compares probabilities.
        Done: fixtures `tests/fixtures/anyjev_qwen25/` (Qwen2, verbatim Qwen2.5
        template -> cfQwen) and `anyjev_qwen3/` (Qwen3, verbatim Qwen3-0.6B template
        -> cfChatML + closed think, which matches HF `enable_thinking=False`). Prompt
        ids and label ids are exact; raw rows, priors and answers agree within 3.1e-6
        (HFBackend runs float64 but casts logits to float32 before log_softmax).
        Note for D1.3: on prompts whose prefix has >= 256 tokens AnyJev scores the
        rotations through `score_shared` (prefix and suffix encoded apart, used only
        when that equals the joint encoding), so it matches the Pascal full encoding.
  - [x] D1.3 Wire L0 as ChatServer's default scorer; user runs the real-model check
        on Qwen2.5-0.5B-Instruct against Python AnyJev.
        Done (coded): ChatServer creates a `TAnyJevL0Scorer` after the model loads,
        named like the chat model in `/v1/models` (the model directory's name; also
        `jev-latest`). Flags `--decision-level l0|raw`, `--decision-prior
        content_free|none`, `--decision-max-rotations N` (ParseServerArgs takes them
        out before ParseArgs, so ChatTerminal does not see them).
        `EAnyJevLabelTokenError` now derives from `EDecisionUnsupported`, so a label
        that is not one token answers 422 naming the question. Docs: ChatServer.md
        *Typed decisions*, the folder README and the top README (planned (coded)).
        Smoke test on `tests/fixtures/anyjev_qwen25` with `--fp32`:
        `tools/anyjev_real_check.py` max abs difference 3.9e-5 (the server rounds to
        4 decimals); decision, chat, chat, decision gave identical decision answers
        and the same greedy chat reply.
    - [ ] User check on a real checkpoint (this VM cannot hold one):
          ```
          git clone https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct q25
          lazbuild -B neural-api/examples/ChatTerminal/ChatServer.lpi
          neural-api/bin/x86_64-linux/bin/ChatServer q25/ --fp32 --ctx 4096 --port 8080 &
          pip install anyjev==0.3.0
          pip install --no-deps git+https://github.com/jaredpalmer/kev@1f04cfa28e6a708485b28a9585010b9e436b776e  # kev pins torch<2.9
          python3 neural-api/tools/anyjev_real_check.py q25/ --server http://127.0.0.1:8080
          kill %1
          ```
          PASS = max abs difference <= 5e-3 (exit 0). `--fp32` is required: int8
          weights move the probabilities far more than FP32 rounding.
  Stage 2 — Kev and Open-Jev:
  - [x] D2.1 `neuraltorchbin.pas`: expose top-level scalar and string values of a
        torch.save dict (temperature, head_dim, base, base_revision, lora), keep
        nested-dict tensors under dotted names (`head.q.weight`), accept any storage
        location string. Unit test on a generated `head.pt`-shaped fixture.
        Done: every non-tensor leaf (None/bool/int/float/str, list lengths) is
        metadata named by its dotted path from the pickle root (list items by index,
        `temperature_fit.cross_validation.temperatures.1`); `GetMetadata{Str,Int,
        Float,Bool}(name, default)` return the default when absent and raise on a
        kind mismatch. A depth cap (64) refuses self-containing containers.
        Fixtures `tests/fixtures/torch_head/` by `tools/torch_head_fixture.py`
        (Kev through kev's own `write_meta`; Open-Jev with 'cpu' and 'cuda:0'
        locations). The real kev-0.8b and Open-Jev-2B `head.pt` files parse.
  - [ ] D2.2 LoRA merge in HF tensor space: read `adapter_config.json` (r,
        lora_alpha, target_modules), map adapter keys onto base checkpoint keys, merge
        in FP32, and build through the normal importer (export the in-memory tensor
        reader path). Test: pico Qwen3.5 + random adapter; merged build logits = HF
        reference with the merged weights.
  - [ ] D2.3 Kev scorer: row construction, per-question rows from the shared state,
        pointer head, temperature from `head.pt`, Jev answer mapping, 65536-token state
        limit. Fixture: pico Qwen3.5 + random LoRA + random head through Kev's own
        `kev/model.py` math.
  - [ ] D2.4 Open-Jev scorer: chat-template prompt per candidate, shared prefix up to
        the first token where the candidates diverge (check encode(prefix + suffix) =
        encode(prefix) + encode(suffix), else score the full prompt), Linear(d,1) head,
        `temperature.json`. Fixture as in D2.3.
  - [ ] D2.5 ChatServer `--decision-adapter <dir>`: detect Kev vs Open-Jev layout,
        load the base given by the usual model directory (warn when it differs from
        the adapter's recorded base), merge, read the head, switch to decision-only.
  - [ ] D2.6 Kev-27B full weights: import `Qwen3_5TextModel` keys without the
        `model.` prefix and without `lm_head` (build ends at the final norm). Pico
        fixture in that layout.
  - [ ] D2.7 User-box real checks: Kev-0.8B (Kev's own test record) and
        Open-Jev-2B against their Python servers; commands written by the agents.
  Follow-ups (not in this plan): Laya; AnyJev batch prior, L1, L2; Kev
  `KEV_DATE_FACTS`; `/v1/systemone/permute` and `/separate`; per-question
  `TruncateTo` downloads the resident KV cache on OpenCL (measure first); tokenizer
  "specials as plain text" flag for chat requests; live (unmerged) LoRA.
- [ ] Flow-matching sampler clean-ups surfaced by `TNNetFlowMatchEulerScheduler`
      (fd99162f):
  - [ ] `examples/F5TTS/F5TTS.lpr` (~108-122): replace the per-element Euler loop
        `Xt.FData[i] + dt*Vel.FData[i]` with `TNNetVolume.MulAdd`. Upstream F5-TTS
        also uses sway sampling, which the scheduler does not cover.
  - [ ] `examples/FlowMatching/FlowMatching.lpr` (~296-312): optionally drive the
        loop with `TNNetFlowMatchEulerScheduler`. The example integrates t from 0
        (noise) to 1 (data) with dt = 1/N and no shift, so it needs time = 1 - sigma
        and a reversed velocity sign; the gain is small (the loop is already one
        `MulAdd` per step).

## OpenCL forward coverage — layers still on the host

Forward pass only; a device backward is a separate project (nothing in the tree
has one today). 52 of the 487 `TNNet*` classes in `neuralnetwork.pas` declare a
`ComputeOpenCL`, and `cai_activation` covers 24 elementwise opcodes on top of
that. `csActivationOpenCLMinSize` is 64 MB, so a small layer reaches the device
only when its source is ALREADY resident — which is why the ranking below is by
"does this layer evict a resident chain to RAM", not by FLOPs.

Every item: guard behind the existing `WillOpenCL()`/`FShouldOpenCL` verdict,
bind the resident source instead of downloading it (`ShouldBindPrevOutputOnOpenCL`)
and leave the result in OpenCL memory, keep the host path as the fallback, and
pin an exact-vs-CPU parity test that skips cleanly with no device. PoCL on the
dev box proves parity; throughput numbers have to come from the GPU box.

### Residency breakers on the decode path (small kernels, largest payoff)

- [ ] `TNNetMoEExpertBankGateUp` single-launch device forward. The clearest
      single gap: `TNNetMoEExpertBankDown` got one (commit d105047c) but its
      sibling did not, so every MoE block downloads the expert activations
      between the two banks. Mirror the down-bank kernel (one launch over the
      selected experts, resident int8 or FP32 weight slab) so gate/up -> down
      stays device-side end to end. Hits Qwen3.5-MoE, Nemotron-H-MoE, Mixtral,
      OLMoE and GPT-OSS.
- [ ] Reshape/gather glue: `TNNetReshape`, `TNNetCrop`, `TNNetPadXY`,
      `TNNetTransposeXD`, `TNNetDepthToSpace`, `TNNetGatherTokens`,
      `TNNetConcat`. None has a device path, and several sit between attention
      and FFN in the importers, so each one evicts the chain. For most of these
      a kernel is the wrong answer: the bytes do not change, only the shape, so
      the layer should re-label the resident buffer (a view) rather than copy
      it. Scope `TNNetReshape` FIRST as the test case for that view mechanism —
      whether `OpenCLOutputBuffer()` can hand back the source's buffer with a
      different declared shape — then apply the outcome to the rest.
      `TNNetCrop`/`TNNetPadXY`/`TNNetTransposeXD` DO move bytes and need a real
      copy/gather kernel; keep the two halves separate.
- [ ] `TNNetReGLU` (and BitNet's `TNNetReGLUSquared`) device forward.
      `TNNetSwiGLU` and `TNNetGEGLU` inherit `TNNetGLUGateBase.ComputeOpenCL`;
      `TNNetReGLU` descends straight from `TNNetLayer` and misses it. Check
      whether re-parenting onto `TNNetGLUGateBase` (with the ReLU gate as the
      activation opcode) is the whole fix before writing a kernel.
- [ ] Per-channel affine glue: `TNNetChannelBias`, `TNNetChannelMul`,
      `TNNetFiLM`, `TNNetMulByConstant`, `TNNetAddConstant`, `TNNetAvgChannel`.
      All are one scale/shift per channel over a resident volume.
      `TNNetChannelMulByLayer` and `TNNetCellMulByCell` already show the
      two-source resident pattern, and `TNNetFiLM` is that same shape, so it
      follows them directly; the constant-parameter ones fit the
      `cai_activation` ParamA/ParamB convention.
- [ ] `TNNetTopKGate` / `TNNetBiasBalancedTopKGate` device forward. Tiny work
      per token, but the router sits between two resident MoE banks, so running
      it on the host costs two transfers per token. Only worth landing together
      with the gate/up bank item above.

### Real device-side work still running on the host

- [ ] `TNNetSelectiveSSM` device forward. The Mamba-1/2, Jamba, Nemotron-H and
      falcon_mamba trunks are majority-SSM and run entirely on the CPU.
      `TNNetGatedDeltaNet` already proved the recipe (commit 63047d0f: one
      work-group per head, the whole scan in OpenCL memory, state carried on the
      device across decode steps); `TNNetSelectiveSSM` is the same shape. Cover
      both the prefill scan and the `TNNetRecurrentDecodeBase` one-token path.
- [ ] `TNNetWKV`, `TNNetTokenShift` and `TNNetDiagonalSSM` device forward — the
      RWKV trunk, same per-head-recurrence recipe as `TNNetSelectiveSSM`. Lower
      priority only because fewer tested checkpoints use it.
- [ ] Attention subclasses that override `Compute` and so lose
      `TNNetScaledDotProductAttention`'s device path:
      `TNNetGptOssSinkAttention` (GPT-OSS is a live importer, and the sink is
      `TNNetFusedSDPA` plus one per-head scalar, so it should reuse the fused
      decode kernels), `TNNetDifferentialAttention` and `TNNetWindowAttention`
      (Swin). Take the sink first.
- [ ] `TNNetDeltaNet`, `TNNetGatedLinearAttention`, `TNNetRetention` and the
      CAUSAL half of `TNNetLinearAttention`. COUPLED, not independent: each is a
      strict per-token scan today, so there is no GEMM-shaped kernel to offload
      until the chunked/parallel forward tasks in "Layer follow-ups that fix real
      limitations" land. Sequence them after those, or fold the device path into
      each chunked rewrite.
- [ ] `TNNetTokenAndPositionalEmbedding` device gather. `TNNetEmbedding` got the
      int8 device gather (`cai_embedding_gather_int8`) and leaves its rows
      resident; its sibling still gathers on the host. Reuse the same entry
      point, adding the positional table as a second resident source.


### Vision and audio layers with no device forward

Companion to the two groups above, same contract (forward only, bind the
resident source, leave the result resident, host fallback, exact-vs-CPU parity
test). The ranking differs though: a vision or audio forward runs over large
tensors, so here the FLOPs matter alongside the transfers, and a kernel can pay
for itself even when it sits between two host layers.

Derived by walking the class graph: a layer counts as covered if it declares
`ComputeOpenCL`, sets an `FActivationOpcode`, dispatches to OpenCL inside its
own `Compute`, or inherits one of those without overriding `Compute`. 310 of
the 487 classes fail that test; the ones below are the subset the vision and
audio builders actually instantiate. Note the third case: a class that
overrides `Compute` with no device branch silently loses its parent's kernel,
which is how `TNNetMaxPoolPortable` lost `TNNetMaxPool`'s.

Training-only layers (`TNNetDropout`, `TNNetDropPath`) are out of scope.

- [ ] `TNNetConvolution3D` (VideoMAE, 3-D VAEs) and `TNNetCausalConv1D` (the
      CogVideoX VAE decoder) device forward. Both are the existing conv
      contraction with one extra axis, so the question to answer first is
      whether `cai_dot_product`'s im2col can be fed a 3-D receptive field
      instead of a new kernel.
- [ ] `TNNetModulatedConv2D` device forward — the core operator of
      `BuildStyleGAN2Generator`, and the layer the whole generator stack waits
      on. The per-sample weight modulation/demodulation folds into the column
      gather, so this is a conv kernel with a scaled weight slab, not a new
      contraction.
- [ ] The decoder upsample inverses: `TNNetDeMaxPool`, `TNNetDeAvgPool`,
      `TNNetUpsample`, plus `TNNetMaxPoolPortable`. `TNNetMaxPool` and
      `TNNetAvgPool` have kernels but their inverses do not, so UNet, SD UNet,
      the VAE decoder, RRDBNet, StyleGAN2, Mask R-CNN and SuperResolution all
      drop to the host on every upsample step. `TNNetMaxPoolPortable` is a
      different fault - it overrides `Compute` and loses the pooling kernel it
      inherits - and it sits in the ResNet, Inception-V3 and DETR stems.
- [ ] Vision attention subclasses that override `Compute` and so lose
      `TNNetScaledDotProductAttention`'s device path:
      `TNNetSAMVisionAttention` (SAM), `TNNetSwinV2WindowAttention`,
      `TNNetWindowAttention` (Swin / MaxViT / BEiT / SwinIR / the CLAP audio
      tower) and `TNNetT5RelPosBiasAttention`. `TNNetDotProducts`, the raw
      contraction behind the CAI self-attention builders, has the same problem.
      `TNNetSinkAttention` and `TNNetDifferentialAttention` are named in the
      decode-path group above; landing them there covers the vision builders
      that also use them.
- [ ] The long tail of one-model vision layers: `TNNetRoIAlign` (Mask R-CNN),
      `TNNetConvGRUCell` (RAFT), `TNNetSpectralConv2D` (FNO - coordinate with
      the open FFT denormal-trap item, since the device path would move that
      math off the host FPU entirely) and `TNNetLocalResponseNormDepth`
      (AlexNet, minor). Rank these by which importer gets used, not by size.
- [ ] Extend the reshape/gather glue item above to its vision members. Pure
      re-label, so they want the view mechanism `TNNetReshape` is scoping:
      `TNNetDepthToSpace`, `TNNetSpaceToDepth`, `TNNetInterleaveChannels` and
      the MobileViT trio (`TNNetMobileViTFold`, `TNNetMobileViTUnfold`,
      `TNNetMobileViTPatchSegments`, which are three reshapes wearing a
      model-specific name). Real byte movement, so a kernel: `TNNetTransposeYD`
      beside the already-listed `TNNetTransposeXD`, `TNNetCrop` and
      `TNNetPadXY`.
- [ ] Extend the per-channel affine item above to the conv-stack members:
      `TNNetCellBias`, `TNNetReZero`, `TNNetPReLUChannel`, `TNNetGRN` and
      `TNNetMovingStdNormalization` (ConvNeXt, RIFE, StyleGAN2, SAM, the moving
      -norm blocks). Same one-scale-and-shift-per-channel shape as the
      `TNNetChannelBias`/`TNNetChannelMul`/`TNNetFiLM` group.
- [ ] Activations with no `cai_activation` opcode, all reachable from the
      vision and audio towers: `TNNetErf`, `TNNetGEGLU`, `TNNetGEGLUErf`,
      `TNNetGLU` (Conformer) and `TNNetSwishLearnable`. `TNNetReGLU` is listed
      above. Each is an opcode plus a `neural.cl` switch case, except the
      gated ones, which need the `TNNetGLUGateBase` split-and-multiply shape.
      These are the vision and audio members of the full opcode gap; the
      activation subsection below carries the whole list and the two cautions
      that come with it.
- [ ] Diffusion-step conditioning recomputed on every denoise step:
      `TNNetSinusoidalTimeEmbedding`, `TNNetLearnedPositionalEmbedding` and
      `TNNetSoftPrompt`. Individually small; they matter because a sampler runs
      them 20-50 times per image, each time bouncing the conditioning vector
      through the host.
- [ ] `TNNetSincConv1D` and `TNNetTDNNConv1D` device forward — the pyannote
      segmentation frontend and the ECAPA-TDNN trunk. Both are convolutions
      whose kernel is generated (sinc band-pass) or dilated, so the existing
      conv offload applies once the kernel slab is materialized; the closest
      thing to a free win on the audio side.
- [ ] `TNNetAttentiveStatsPooling` (ECAPA-TDNN), `TNNetGLU` (Conformer, also
      listed with the activations) and `TNNetFlipX`. `TNNetFlipX` is the one
      host hop inside an otherwise device-capable bidirectional stack -
      `TNNetLSTMCell` and `TNNetGRUCell` both have kernels - and reversing a
      sequence is a copy kernel or, better, an index flip in the consumer.

The channel-major audio holders (`TEnCodecModel`, `TNNetHiFiGAN`, `TNNetVits`,
`TNNetMimi`, `TNNetDAC`, `TNNetDemucs`, `TNNetKokoro`) are NOT TNNet layers and
are deliberately absent from this list: they run conv1d directly on
channel-major arrays through `RunEnCodecConv` / `RunHiFiGANConv` /
`RunMimiConv` / `RunDACConv`. EnCodec and HiFiGAN already have the
`EnableConvOpenCL` GEMM route and Demucs and Kokoro inherit it by calling them;
Mimi and DAC are the remaining gap and are already tracked in the audio section
above (blocked on a Double-precision shared dot-product kernel).


### Activation coverage — the fused GEMM tail and the standalone kernel

Two separate mechanisms, easy to confuse. (1) The FUSED tail inside the matmul
kernels: `cai_dot_product` applies `ActFN` in-register after the bias add, and
it implements exactly three opcodes — 1=ReLU, 2=Sigmoid, 3=Tanh — with the same
three copied into `cai_dot_product_int8` and into `cai_fused_act` (the split-K
reduce helper). `TNNetLayer.IsActivationFunctionInOpenCL` mirrors that set on
the host and returns false for everything else. (2) The STANDALONE
`cai_activation` kernel, which has 24 opcodes and 25 layer classes wired to it
through `ComputeActivationOnOpenCL()`.

Both are forward-only by construction: neither produces `FOutputRaw` or the
derivative mask, so `IsActivationFunctionInOpenCL` requires `not FIsTrainable`
and the standalone path requires it too. `csActivationOpenCLMinSize` (64 MB)
keeps the standalone kernel off the device unless the source is already
resident or `ForceOpenCL` is set — deliberate, since a 1-flop-per-word kernel
cannot pay for its own transfer.

- [x] Widen the fused GEMM tail beyond ReLU/Sigmoid/Tanh, starting with
      Swish/SiLU and GELU. The math already exists as `cai_activation` opcodes
      4 and 5, parity-tested, so this is mostly moving code: extend the tail's
      switch, widen `IsActivationFunctionInOpenCL` to match, and dedup the
      THREE copies of the tail (`cai_dot_product` inline,
      `cai_dot_product_int8` inline, `cai_fused_act`) through the helper that
      already exists. Why it matters beyond one saved kernel launch: when the
      verdict is false, `TNNetConvolution.ComputeOpenCL` does not merely skip
      the fusion — it sets `FOutputOnOpenCL := false`, downloads `FOutputRaw`
      and runs the bias add and the activation sweep on the host, so the whole
      residency chain breaks at the GEMM. Today only
      `TNNetConvolutionSwish`, `TNNetConvolutionHardSwish` (the CAI CIFAR
      examples) and `TNNetFullConnectDiff` carry a non-fusable fused
      activation, since the importers build Linear convolutions with separate
      activation layers — so measure before assuming a win, and treat the
      residency break as the reason rather than the launch count.
      DONE: `cai_fused_act` moved to the top of `neural.cl` and gained opcodes
      4, 5 and 7, and it is now the ONLY copy of that math — `cai_dot_product`,
      `cai_dot_product_int8`, the split-K reduce and `cai_activation`'s own
      cases for those opcodes all call it (four copies collapsed into one).
      `IsActivationFunctionInOpenCL` accepts `@Swish` and `@HardSwish`, so
      those convolutions and FullConnects now fuse and stay resident.
      Device-vs-CPU parity cases added to all three harnesses — `TestConvDeviceIm2ColOpenCLParity` (FP32,
      9.5e-7), `TestInt8QuantizedOpenCLParity` (int8 GEMM, 3.9e-7) and
      `TestInt8SplitKOpenCLParity` (split-K, 2.0e-6) — and the conv harness now
      asserts `OutputBindableOnOpenCL` so a silent fallback to the host can no
      longer pass as parity. Suite 2837/0/0 on both the scalar and the
      `-dOpenCL` build. Two notes: the tail's ReLU is now `(v > 0) ? v : 0`
      like `cai_activation`, which differs from the old `if (v < 0) v = 0` only
      in returning +0.0f for a -0.0f input; and GELU rides in the kernel but no
      host layer can request it yet — `TNNetGELU` is a standalone layer and
      there is no `@GELU` activation-function pointer, so the complete set a
      layer can carry is Identity/ReLU/Sigmoid/Tanh/Swish/HardSwish/DiffAct.
  - [x] `@HardSwish` (opcode 7) landed the same way in a follow-up, so
        `TNNetConvolutionHardSwish` fuses and stays resident too. Opcodes 6
        (GELUErf) and 8 (HardSigmoid) followed, so `cai_fused_act` now holds
        EVERY parameterless opcode (1-8) and `cai_activation` keeps only the
        parameterized cases (9, 10, 15, 18-21, 24). The remaining limit is on
        the host, not in the kernel: GELU, GELUErf and HardSigmoid exist only as
        standalone layers with no activation-function pointer, so no layer can
        request them fused, and `@DiffAct` is the one pointer with no opcode.
        Fused parity cases for Swish and HardSwish (bias and bias-suppressed)
        live in `TestConvActivationFusionOpenCLParity` and
        `TestFullConnectActivationFusionOpenCLParity`;
        `TestActivationOpenCLParity` already sweeps all 30 standalone variants,
        which covers the delegation for 6 and 8.
- [ ] The 34 elementwise activations with no `cai_activation` opcode, all
      `TNNetReLUBase` descendants: `TNNetErf`, `TNNetExp`, `TNNetLog`,
      `TNNetSqrt`, `TNNetSin`, `TNNetCos`, `TNNetArcSinh`, `TNNetSinhAct`,
      `TNNetReciprocal`, `TNNetPower`, `TNNetMish`, `TNNetSerf`, `TNNetSmish`,
      `TNNetPhish`, `TNNetSnake`, `TNNetSinc`, `TNNetLisht`,
      `TNNetLogCoshActivation`, `TNNetLogSigmoid`, `TNNetTanhExp`,
      `TNNetTanhShrink`, `TNNetLeCunTanh`, `TNNetPenalizedTanh`,
      `TNNetSoftPlus`, `TNNetSoftPlusBeta`, `TNNetSoftExponential`,
      `TNNetGaussianActivation`, `TNNetISRU`, `TNNetISRLU`, `TNNetCELU`,
      `TNNetESwish`, `TNNetSwish6`, `TNNetReLUSqrt`, `TNNetRReLU`. Each is one
      `case` in the `cai_activation` switch plus one constant and the
      `ComputeActivationOnOpenCL()` call at the top of `Compute`. Take
      `TNNetErf` first — it is on the ModernBERT, CLIP, Whisper and BART paths.
      Two cautions: `TNNetRReLU` draws a random slope, so its device form must
      reproduce the host draw or stay inference-frozen; and `TNNetSoftSign`
      and `TNNetESwish` raise hardware FP exceptions at far-extreme inputs on
      the host (see the FP-exception item in the test section), so pick the
      clamping policy there BEFORE pinning device parity. The gated
      activations (`TNNetGEGLU`, `TNNetGLU`, `TNNetReGLU`, `TNNetGEGLUErf`)
      are a different shape and stay with the `TNNetGLUGateBase` items above.
- [ ] The 11 per-channel LEARNABLE activations, which need a kernel shape the
      current one does not have — a parameter VECTOR indexed by channel rather
      than the scalar `FParamA`/`FParamB`/`FParamC`: `TNNetPReLU`,
      `TNNetPReLUChannel`, `TNNetAPL`, `TNNetAconC`, `TNNetMetaAconC`,
      `TNNetSReLU`, `TNNetSplineActivation`, `TNNetSwishLearnable`,
      `TNNetMishLearnable`, `TNNetSoftPlusBetaLearnable`,
      `TNNetPolynomialActivation`. All descend from
      `TNNetChannelTransformBase`, so one kernel taking a `__global const
      float*` parameter table plus an opcode covers the family; the per-channel
      broadcast is the same indexing `cai_group_norm`'s affine tail already
      does. Forward only — the learnable parameters' gradients stay on the host.


## Layer follow-ups that fix real limitations

- [~] Bidirectional + multi-layer stacking for `TNNetLSTMCell` / `TNNetGRUCell`
      (and a real `nn.LSTM`/`nn.GRU` num_layers>1 / bidirectional=True checkpoint
      importer). The single-cell `TNNetLSTMCell`/`TNNetGRUCell` (torch fused
      `weight_ih`/`weight_hh` gate layout, full forward + exact BPTT) have landed.
      DONE: `TNNet.AddBidirectionalLSTM(Hidden,NumLayers,Bidirectional)` and
      `AddBidirectionalGRU(...)` builders (shared `AddBidirectionalRecurrentStack`
      engine) — pure composition of cells + `TNNetFlipX` reversal + `TNNetDeepConcat`
      ([forward;backward] torch concat order); multi-layer feeds layer k's output
      into k+1, each direction gets a `TNNetPointwiseConvLinear` projection to
      Hidden when incoming Depth≠Hidden (covers layer 0 and the 2*Hidden→Hidden
      feed above a bidirectional layer, matching nn.LSTM input_size). Tests:
      builder wiring/shape/SaveToString round-trip (LSTM+GRU), unidirectional
      1-layer == bare cell equivalence, bidirectional forward-half == forward cell
      alone (pins concat order), and a 2-layer bidirectional stack input-gradient
      check (all in TestNeuralNumerical, passing; full suite green 2376/0/0). This
      was applied to the pyannote `segmentation-3.0` trunk, which now uses true
      nn.LSTM cells.
  - [ ] Faithful STACKED-BIDIRECTIONAL torch import follow-up: the
        `AddBidirectionalRecurrentStack` engine inserts a learned
        `TNNetPointwiseConvLinear` projection on every incoming Depth≠Hidden feed
        (layer-0 input_size≠Hidden, and the 2*Hidden→Hidden feed above each
        bidirectional layer). torch nn.LSTM/nn.GRU has NO such projection — layer
        k's `weight_ih_l{k}` is a plain `[gates*Hidden, 2*Hidden]` slab consuming
        the concatenated [forward;backward] hidden directly. To import a real
        multi-layer bidirectional checkpoint, add an opt-in builder mode that
        feeds the 2*Hidden concat straight into the next layer's cell (cell
        input_size = 2*Hidden, no projection) so the torch `weight_ih_l{k}` rows
        map 1:1. Then extend `LoadTorchLSTMInto`/`LoadTorchGRUInto` to accept that
        projection-free stack and drop the loud rejection; add a 2-layer
        bidirectional pico-oracle parity fixture (the generator already supports
        the shape — `tools/torch_rnn_tiny_fixture.py`). Unblocks the pyannote
        `segmentation-3.0` real bidirectional-LSTM trunk drop-in.

- [ ] OpenCL forward offload for `TNNetCausalLinearAttention` (the non-causal global
      `TNNetLinearAttention` is now DONE — `ComputeOpenCL` two-GEMM offload behind
      `FShouldOpenCL` + `LinearAttentionOpenCLParity` exact-vs-CPU test, PoCL-verified
      max|diff| ≈ 1.5e-8). The causal sibling was deliberately LEFT CPU-only: its
      forward is a left-to-right prefix-sum scan (`S_t = S_{t-1} + ϕ(K_t)⊗V_t`,
      per-query `Out_t = ϕ(Q_t)·S_t / ϕ(Q_t)·Z_t`), NOT a pair of dense GEMMs, so it
      does not map onto the `FDotCL` matmul kernel without a chunked-scan rewrite.
      Tackle it with the chunked-forward family below (intra-chunk dense GEMM +
      inter-chunk running state), then add a causal parity test alongside.

- [X] AVX-vectorize the remaining trig/inverse-hyperbolic activation FORWARD
      loops left scalar: `TNNetSin` / `TNNetCos` / `TNNetArcSinh`. DONE — added
      three dedicated 8-wide AVX2 kernels to `neuralvolume.pas`: `AVXLn` (Cephes
      `logf`, x=m*2^e decompose + degree-8 poly in (m-1)), `AVXSinCos` (Cephes
      `sinf`/`cosf` with 3-part Cody-Waite pi/4 range reduction, octant sign/poly
      select via `vpandn`/`vblendvps`; cos uses the `(~m)&4` sign and `(m&2)==0`
      poly convention), plus alias-safe class wrappers `TNNetVolume.VectorLn` /
      `VectorSin` / `VectorCos`. `VectorArcSinh` = `ln(x+sqrt(x^2+1))` is built on
      `VectorLn` (scratch-buffer prep so dst may alias src). 32-bit / non-AVX2
      builds fall back to scalar RTL loops (register pressure on the bit-tricks).
      `TNNetSin.Compute` / `TNNetCos.Compute` / `TNNetArcSinh.Compute` converted to
      the new vector primitives (forward vectorized; derivative filled by a scalar
      finishing pass reading x back, so backward math is byte-identical). Parity
      tests `TestVectorLnScalarParity` / `TestVectorSinScalarParity` /
      `TestVectorCosScalarParity` / `TestVectorArcSinhScalarParity` (< 1e-4,
      body+tail+aliased+large-magnitude) green on both builds; measured AVX2 max abs
      err ~2.4e-7 (ln), ~6e-8 (sin/cos), ~4.8e-7 (arcsinh). Earlier `TNNetSinhAct`
      was DONE via two `VectorExp` passes (`VectorSinh`). Earlier tanh/erf batch:
      `TNNetGELU`, `TNNetGELUErf`, `TNNetMish`, `TNNetSerf`, `TNNetSmish`,
      `TNNetPhish`, `TNNetLogCoshActivation`, `TNNetLisht` vectorized via
      `VectorTanh`/`VectorErf` (built on `VectorExp`); parity tests
      `TestVectorTanhScalarParity` / `TestVectorErfScalarParity` (< 1e-4) green.

(The sub-quadratic / chunked-forward family below is one coherent systems effort:
every recurrence currently trains as a strict per-token left-to-right scan.)

- [ ] TNNetDeltaNet chunked/parallel forward (the paper's WY-matrix
      reformulation of Yang et al. 2024, arXiv:2406.06484) so training is
      sub-quadratic instead of the current strict per-token left-to-right scan in
      TNNetDeltaNet; gate it behind an exact-vs-chunked equivalence assert.
- [ ] TNNetWKV chunked/parallel forward (RWKV-5/6 style) so training is not a
      strict per-token left-to-right scan; gate behind an exact-vs-chunked
      equivalence assert (mirrors the open TNNetDeltaNet chunked-forward task).
- [ ] TNNetGatedLinearAttention chunked/parallel hardware-efficient forward (the
      paper's main systems contribution; v1 ships the exact per-token scan only) —
      gate behind an exact-vs-chunked equivalence assert (mirrors open
      DeltaNet/WKV chunked tasks).
- [ ] TNNetMinGRU / TNNetMinLSTM follow-up (landed 2026-06-07, commit 69f8d53):
      Parallel-prefix-scan forward (the paper's main systems win) so training is
      not a strict per-token left-to-right loop; gate behind an exact-vs-parallel
      equivalence assert (mirrors the open DeltaNet/WKV chunked-forward tasks).
- [ ] TNNetLRU parallel/associative-scan forward (LTI recurrence → parallelizable)
      gated behind an exact-vs-scan equivalence assert.
- [ ] TNNetRetention chunkwise-recurrent hybrid form (a throughput optimisation
      skipped in v1 — the parallel and naive-recurrent forms both landed).
- [ ] Mini-batch / chunked Test-Time Training — follow-up to the landed
      TNNetTestTimeTraining (both TTT-Linear and TTT-MLP arms, with exact
      second-order BPTT through the inner update, shipped with tests + the
      examples/TestTimeTraining parity-binding recall demo). Apply ONE inner
      gradient-descent step per CHUNK of b tokens (mini-batch the inner SGD)
      instead of per token, for the same sub-quadratic-training motive as the
      open DeltaNet/WKV chunked-forward tasks. Optional second follow-up: a
      learnable PER-CHANNEL inner LR eta (currently a single learnable per-layer
      scalar via softplus(eta_raw)).
- [ ] `TNNetTitansMemory` follow-up — a **gated-DeltaNet-style chunked parallel
      scan** forward for `TNNetTitansMemory`, replacing the sequential O(SeqLen)
      inner-gradient scan with a chunked associative/parallel recurrence (the
      hardware-efficient training path Titans/Gated-DeltaNet use); must keep the
      exact second-order BPTT semantics and pass the existing gradient checks.
- [ ] TNNetImplicitLongConv / AddHyenaOperator FFT-based O(L log L)
      forward/backward path as an opt-in fast mode for long sequences — the
      current forward is the direct O(L^2) causal time-domain sum, fine for small
      sizes but quadratic in SeqLen. Gate it so the exact time-domain path stays
      the default and assert FFT-vs-direct equivalence to <1e-5.
- [ ] TNNetHyperLinear CHUNKED weight generation so the main layer can be larger
      than the generator's output width (generate W in tiles) — the landed layer
      generates the whole Din*Dout matrix in one shot, which caps main-layer
      size; document the memory/param trade-off.
- [ ] TNNet.AddDeepEquilibriumBlock follow-up: (a) the EXACT
      implicit-function-theorem gradient (inverse-Jacobian solve via a second
      fixed-point iteration) vs the phantom approximation; (b) spectral /
      contraction constraints so convergence is guaranteed at arbitrary init
      (v1 uses damped Picard + output bounding, not guaranteed).
- [ ] TNNetReversibleBlock follow-up: the MEMORY-SAVING recompute path (the
      actual point of RevNet — discard activations in forward, RECONSTRUCT them in
      backward via the analytic inverse instead of storing them). The landed
      builder demonstrates the inverse FORMULA and trains via ordinary stored-
      activation backprop; the O(1)-activation-memory training mode is still open
      and needs a custom backward that recomputes x1,x2 from y1,y2. Pairs with the
      open "Gradient checkpointing" infrastructure task.
- [ ] TNNetReversibleBlock follow-up: stack N reversible blocks into a deep net
      and show constant activation memory vs a plain residual stack of equal depth
      (the headline RevNet scaling claim) — depends on the recompute path above.
- [ ] TNNetSpectralNormConv follow-up: the landed wrapper normalizes by sigma_1
      of the FLATTENED kernel matrix (out_channels x in*kx*ky), a single scalar
      that BOUNDS but does not equal the true conv-OPERATOR spectral norm. Add a
      true-operator variant (Sedghi et al. 2019 FFT-based conv spectral norm, or
      a per-output-channel sigma) plus a small conv-SN bake-off / example vs the
      flattened-matrix version.
- [ ] `TNNet.AddTitansMemory` builder — a **MAC residual builder** wrapping
      `TNNetTitansMemory` (token-shift + per-token k/v/q projections + the neural
      memory leaf + residual/out-projection), the drop-in Memory-as-Context block,
      mirroring `AddGatedLinearAttention` / `AddRWKVTimeMix`.
- [ ] SDPA-level KV-cache aliasing for GQA decode (follow-up to the GQA
      verification batch, commit c1f8c8a): `AddMultiHeadGroupedQueryAttention`'s
      PARAMETER saving is real, but at decode time `TNNetStreamingDecoder` keeps
      one KV cache per SDPA layer, so the QueryHeads/KVHeads heads in a group
      each cache an identical K/V copy. Let grouped SDPA layers share a single
      KVHeads-sized cache (cache aliasing keyed by the shared K/V projection
      layers) so the GQA memory win materializes at inference; assert streamed
      output stays bit-identical to the unaliased path.
- [ ] Segment-mask MHA-builder wiring follow-up (the SDPA-layer + packer half
      landed above): thread an optional segment-id source through the
      multi-head attention BUILDERS (AddMultiHeadSelfAttention and friends) so
      packed-window training masks cross-document attention end-to-end, not just
      at the bare TNNetScaledDotProductAttention layer. Each per-head SDPA in the
      builder takes the same shared `pSegmentSource`; assert a packed two-doc MHA
      stack matches the concatenation of independent per-document MHA runs (the
      builder-level analogue of TestSegmentMaskMatchesUnpackedBaseline). KV-cache
      incremental decode stays intentionally unmasked (single-stream = one doc).

- [ ] Residual scalar backward loops left by the AVX/OpenCL vectorization batch
      (the forward + cleanly-mappable backward paths are done; these are the
      strided-on-one-operand remainders that a single DotProduct/MulAdd cannot cover
      without an extra gather):
  - [ ] TNNetTestTimeTraining / TNNetTitansMemory backward "undo" loops (interleaved
        scalar etaGrad/dEta/dTheta accumulation) — the per-token forward rank-1 writes
        are vectorized; this is the lower-value remainder.

## Tests / numerical-gradient audit

- [ ] Property-based gradient harness v0: randomize input shape (keeping
      layer type fixed) for the 6 most recently landed layers. Catches
      shape-edge bugs hand-written tests miss.
- [ ] Continue upsampling/deconvolution audit: TNNetDeconvolution input
      AND weight gradients (Upsample / DeMaxPool / DeAvgPool already
      covered). Likely benefits from the Double-precision SSE accumulator
      helper above.
- [ ] Recurrent-style layer audit: TNNetEmbedding's weight-gradient path
      (sparse-update pattern — easy place for a silent broadcast/reduction
      bug), TNNetTokenAndPositionalEmbedding, etc.
- [ ] Multi-branch layer-registry round-trip audit: extend the landed
      single-source registry round-trip (tests/TestNeuralRegistry.pas) with a
      multi-branch builder to cover the skipped two-source layers
      (Concat/Sum/CrossAttention/FiLM/HyperLinear/etc.).
- [ ] Shape-inference smoke test — instantiate every concrete layer at a
      small canonical input shape, assert declared output shape matches
      actual.
- [ ] TestExtensions check: every layer that declares `FStruct[k]`
      constructor parameters should be tested for LoadFromString round-trip
      with NON-default values.
- [ ] Find-or-falsify pass: scan neuralnetwork.pas for any Backpropagate
      override whose body is just `inherited;` plus a tiny tweak — flag
      candidates for gradient-check coverage. The exact-softmax-Jacobian
      story teaches us how silent the diagonal-only bug class can be.
- [ ] Audit any remaining TNNet* layers that compute a softmax-like
      normalization (search for "Exp(" near a normalization loop) to
      confirm none still ship the diagonal-only approximation.
- [ ] Numerical-gradient stress test for TNNetSoftMax / TNNetPointwiseSoftMax
      across SeqLen / Depth / SizeX combinations.
- [ ] Backward audit for TNNetPointwiseNorm — its backward is the scalar-
      only `Mul(1/n)` approximation; TNNetL2Normalize now implements the
      exact Jacobian. Either replace or add a deprecation comment.
- [ ] Random-architecture forward/backward fuzz — generate ~50 random
      stacks, seed-controlled, assert no NaN/Inf in forward, backward,
      or parameter gradients.
- [ ] Cross-layer composition gradient test: build a 3-layer stack
      (LayerNorm → SwiGLU → Dense) and run a single end-to-end central-
      difference check on the input.
- [ ] Activation derivative-cache invariants sweep: for each cache-using
      activation, run `Compute(A); Compute(B);` and assert FOutputErrorDeriv
      matches a fresh recomputation against B. Catches the Sigmoid-class
      stale-cache bug pattern.
- [ ] Activation golden-values regression test — for every registered
      activation, evaluate forward/backward on a pinned input at fixed
      seed and assert against pinned outputs within 1e-5.
- [ ] Per-activation derivative-sign sanity test — for each strictly
      monotone activation, assert FOutputErrorDeriv has the expected sign
      on a grid of inputs.
- [ ] Saturation-safety tests for TNNetTanhExp / TNNetSmish at ±extreme
      inputs, mirroring the HardTanh/SoftCapping pattern.
- [ ] TNNetDigital forward-equality test — pin threshold and output for
      three inputs straddling it (non-differentiable, so forward-only).
- [ ] TNNetMaxPoolWithPosition correctness check — the auxiliary "position
      channels" should round-trip through TNNetDeMaxPool to exactly
      reconstruct the upsample pattern.
- [ ] TNNetAddPositionalEmbedding scale-factor backward check on
      rectangular (X≠Y) shapes (square inputs can hide off-by-one bugs).
- [ ] Gradient-flow regression test — train a 12-layer ReLU MLP one epoch
      with and without a single TNNetLayerNorm/RMSNorm at the midpoint;
      assert per-layer gradient norms with the norm layer are uniformly
      bounded above the no-norm case.
- [ ] TNNetDotProducts numerical-gradient test — standalone class still
      ships, weight-gradient path looks like the kind of place a silent
      bug could live.
- [ ] TNNetLocalConnect / TNNetDeLocalConnect input + weight gradient tests.
- [ ] Kink-region test parametric helper: with Clamp / HardShrink /
      SoftShrink / Threshold / ShiftedReLU / HardTanh all in tree, the
      "no-central-difference, hand-picked kink convention" pattern
      repeats. Capture as `AssertKinkDerivative(layer, x_kink, expected_dydx)`.
- [ ] TNNetHardShrink / TNNetSoftShrink kink-region tests at hand-picked
      inputs (no central differences).
- [ ] FP-exception robustness for TNNetSoftSign / TNNetESwish at truly extreme
      inputs (surfaced 2026-05-31 while writing the saturation tests above):
      both raise a HARDWARE FP exception rather than returning a finite value
      at far-extreme magnitudes — TNNetSoftSign's closed-form derivative
      `1/(1+|x|)^2` overflows float32 around |x|~1e30 (EInvalidOp), and
      TNNetESwish's `Exp(-beta*x)` overflows the RTL `Exp` around beta*x~-570
      (EOverflow). The landed tests stay inside the safe-but-saturating band
      (SoftSign ±1e6, ESwish beta*x up to ±625) and document the limit. Decide
      a policy: either clamp the offending intermediate (saturate the derivative
      to 0 / the sigmoid to its asymptote) so the layers stay finite at any
      input like HardTanh/SoftCapping do, or document the input-range contract.
      Then extend the tests to the far-extreme range under the chosen policy.
- [ ] LiSHT / BentIdentity gradient-magnitude sanity at large |x| — both
      grow unboundedly, finite-difference eps must scale with input
      magnitude.
- [ ] Shape-edge test for TNNetTokenShift: assert SetPrevLayer raises the
      documented error when SizeY > 1.
- [ ] Two-layer TokenShift composition test (catches subtle double-pass
      bugs in the t-1 / t+1 input-gradient scatter).
- [ ] TNNetStraightThroughEstimator `step ≤ 0` guard test.
- [ ] Audit TNNetSigmoid and TNNetHardSigmoid for negative-x / positive-x
      symmetric-stability (same question as SoftPlus).
- [ ] Add a "FP32 SSE accumulator warning" comment near LayerInputGradientCheck
      pointing future audits at the DeMaxPool case and the Double-precision
      workaround.
- [ ] TNNetPointwiseSoftMax: now that the exact Jacobian lives in
      Backpropagate, opt cross-entropy training paths into the cheap
      (y - target) shortcut explicitly, and add a regression test that
      checks the shortcut and the full-Jacobian path agree to 1e-5.
- [ ] Cross-entropy regression-style check: confirm classification
      examples (SimpleImage CIFAR) converge to the same loss curve they
      did before the TNNetSoftMax.Backpropagate exact-Jacobian change.
- [ ] Re-validate examples that use TNNetDeMaxPool / TNNetDeAvgPool after
      the gradient fix: DenseNet helper, VisualGAN, SuperResolution. The
      fix increases backward magnitude by `PoolSize` (=2 in practice), so
      existing learning rates may be off by 2x.
- [ ] Loss-layer gradient-check helper — parameterized helper that takes
      (LossLayer, BatchSize, Shape) and runs a single central-difference
      check.
- [ ] Backward-pass sign-correlation test — for every layer that overrides
      Backpropagate, perturb input by ±ε, assert gradient direction agrees
      with loss-difference direction >90% of the time across a small grid.
- [ ] Coverage matrix at the top of TestNeuralNumerical.pas: per-class
      `[grad] [serialize]` block, written by a small script.

## Vision / generative & accelerator batch 2026-06-27b

(Verified absent in source before listing. An NF4 dequant path did not exist
then — only int8 + MXFP4-dequant did.)

- [ ] NF4 (bitsandbytes 4-bit) dequantizing import path. PARTIALLY LANDED
      (commit 90b6c931): the `neural/neuralnf4.pas` dequant unit ships — the exact
      16-level NF4 codebook (`NF4Code`) + `DequantizeNF4(Codes, Absmax, N, Dest,
      BlockSize=64)` with HIGH-nibble-first packing and per-block FP32 absmax
      (`dequant[i] = NF4_CODE[nibble] * absmax[i div BlockSize]`); single-quant
      only. Cross-checked to <1e-5 against a pure-numpy reference
      (`tools/make_pico_nf4_fixture.py` → `tests/fixtures/pico_nf4.json`;
      `bitsandbytes` is NOT importable under `venv x` — CPU torch, no CUDA bnb
      build) via `TestNF4DequantFixtureParity` + `TestNF4DequantHandBlock`.
      OPEN FOLLOW-UPS:
  - [ ] Double-quantized absmax: handle `*.nested_absmax` / `*.nested_quant_map`
        / `*.quant_state.*` (the secondary int8 quant + nested absmax + offset).
        DEFERRED. `LoadNF4QuantizedTensorFlat` now RAISES a clear
        `EPretrainedImportError` on these nested/double-quant markers (and on a
        non-FP32 absmax sibling) rather than feeding quantized absmax into
        DequantizeNF4 — no silent garbage path. `TestNF4ImporterRejectsDoubleQuant`
        covers the rejection (fixture `pico_nf4_doublequant.safetensors`). To
        finish: expand the int8-quantized absmax (nested absmax + offset +
        quant_map) to FP32 upstream, then drop the guard. A real `bnb-4bit`
        Llama/Qwen slicer fixture is RAM/network-gated; today's parity is the
        random pico numpy oracle only.

## Accelerator & dedup batch 2026-06-27f

(All LANDED and removed: OpenCL softmax-head offload — new `cai_softmax` kernel +
`TNNetSoftMaxCL` host helper for `TNNetPointwiseSoftMax`/`TNNetSoftMax`, commit
e8042aa0; AVX `TNNetReLU` forward via `AVXCopyRelu`, commit 10433bfd; AVX
SoftPlus/Gaussian/SoftExponential forwards via `AVXExp` — note `AVXExp`'s scalar
remainder tail has NO internal clamp so extreme inputs must be pre-clamped to
[-88,88] before the call, commit 3e5e649d. Open follow-ups surfaced by this batch:)

- [x] Keep the softmax-head activation resident on the OpenCL device across the
      SDPA score-matrix producer -> `cai_softmax` -> consumer chain. The new
      `cai_softmax` offload (e8042aa0) still uploads/downloads the volume per call;
      when the producer is already device-resident this is a wasted round-trip.
      Tie into the existing "keep activations resident across consecutive offloaded
      layers" follow-up in the vision/generative section so attention blocks chain
      device-side. Forward-only, parity-tested, skip-clean when no device.
      DONE: TNNetPointwiseSoftMax binds a resident source and leaves its result in
      OpenCL memory, so the SDPA producer -> cai_softmax -> consumer chain stays
      device-side; TNNetSoftMax shares the same helper.

## Lucky-day batch 2026-06-28f (follow-ups surfaced by 28e landings)

(All three 28e items LANDED: Heun sampler [891e7537], Min-SNR-gamma loss
weighting [f8e1823c], tiled VAE decode [2e28a44b]. Genuinely-novel follow-ups
surfaced while landing them:)

- [ ] Fully-convolutional VAE-decoder Compute path so `TiledVaeDecode`
      (neuralpretrained.pas) can reuse ONE decoder net across tiles instead of
      rebuilding a tile-sized decoder per tile-grid. Today `TNNet.Compute` errors
      on a latent whose size differs from the fixed `TNNetInput`, so the tiled
      path rebuilds the decoder at the tile latent size. A size-agnostic forward
      (re-derive layer shapes from the actual input volume on each Compute, or a
      Resize-input hook) would let a single resident net decode arbitrary tile
      sizes — also unblocks variable-resolution inference generally. Scope first:
      audit which layers in the VAE decoder graph hard-bind to TNNetInput size.
- [ ] Tiled VAE decode is an APPROXIMATION because the decoder MID block has
      GLOBAL self-attention (same caveat as diffusers): per-tile attention can't
      see across seams. Follow-up: a windowed/overlapped-attention tiled mode, or
      document+measure the approximation error vs whole-image decode on a real
      (trained) checkpoint so users know the quality trade-off. Pure host /
      analysis; no new layer.

## Lucky-day batch 2026-06-28g (follow-ups surfaced by 28g landings)

(All five 28g items LANDED and removed: SDPA attention-variant OpenCL safety
guard for all four variants + correct ALiBi device offload [9957892f]; AVX
`TNNetSoftPool` [6e373ae4]; AVX `TNNetCumSum` depth-axis [ff593de0];
`TNNetFiniteScalarQuant`/FSQ [dfaa9c41]; `TNNetLookupFreeQuant`/LFQ [8cfaea3b].
The open OpenCL follow-up LANDED and removed: device offload for the remaining
three SDPA variants (cosine-sim / disentangled / conformer) via per-variant
EnableOpenCL + ComputeOpenCL, gathered position dots stay on the CPU gap, new
CosineSimilarity/Disentangled/ConformerRelPosAttentionOpenCLParity tests RAN on
PoCL at max|diff| ~1e-7, suite 2493/2493 on both builds [914cbdc9].)

## Lucky-day batch 2026-06-28h (verified-novel AVX / dedup) — ALL LANDED

(All three 28h items LANDED and removed: AVX-vectorized SDPA `ComputeTiled`
online-softmax accumulation via `TNNetVolume.Mul`/`MulAdd` depth-contiguous ops,
tiled-vs-naive parity green on scalar + -dAVX2 [02d11b09]; fused-QKV unpackers
unified into `LoadFusedQKVWeights` + `TQKVPackLayout` enum, three importer
wrappers preserved, pico fixtures bit-identical [6083a4d0]; GPT-2/Llama/Cohere/
Bark affine norm-weight loaders unified into `LoadAffineNormWeights`, four thin
wrappers preserved, all parity fixtures bit-identical [dd5373fc].)

## Lucky-day batch 2026-06-28i (verified-novel vision dedup / AVX / torch port) — ALL LANDED

(All three 28i items LANDED and removed: reusable `TNNetGridSample` layer — torch
`F.grid_sample` bilinear/nearest x zeros/border x align_corners, with a shared
`NeuralBilinearGatherColumn` depth-column gather helper that `TNNetAffineGridSample`
now delegates to [d9f91ab4]; AVX-vectorized `TNNetDeformableConv` bilinear gather
routed through that same shared helper (+ DCNv2 sigmoid modulation applied after the
gather), forward bit-identical / backward parity-tested [ab92389f]; reusable
`TNNet.AddPatchEmbedding` ViT patchify builder (conv-stride-patch -> reshape ->
optional CLS + positional embedding) + `examples/TinyViT` regression [91024a9e].)

## Lucky-day batch 2026-06-28j (verified-novel accelerator / generative / torch port) — ALL THREE LANDED

(All three 28j items LANDED and removed: AVX per-channel + OpenCL per-depth
forward for `TNNetL2Normalize` (`cai_l2norm_perdepth` kernel + `TNNetL2NormCL`,
commit b01eab21); OpenCL per-pixel RMS forward for `TNNetPixelNorm` reusing that
kernel with InvScale=1/Depth + Eps (commit d50725e5); `TNNetResize2D` unified
arbitrary-size resize — nearest/bilinear/bicubic + align_corners, size-based
source maps for down/non-integer ratios, exact-transpose backward, FStruct[0..3]
round-trip (commit e37d07bb). Open follow-ups surfaced by the Resize2D landing:)

- [ ] `TNNetResize2D` `antialias=True` box-prefilter for downscale was SCOPED OUT
      of v1 (it changes the kernel support to scale-dependent, breaking the fixed
      2-tap / 4-tap separable gather; torch's antialias widens the kernel and
      renormalises). Add it as an opt-in 5th flag (FStruct[4]) when needed.
- [X] DID NOT dedup the integer-factor up-samplers onto `TNNetResize2D` — the
      risk note said not to force it. `TNNetBilinearUpsample`/`Upsample` keep
      their OpenCL gather/scatter paths. The OpenCL forward IS NOW DONE (the dedup
      was deliberately left out): `TNNetResize2D.EnableOpenCL`/`ComputeOpenCL`
      reuse `TNNetBilinearGatherCL` (4 corners, bilinear mode) and
      `TNNetBicubicGatherCL` (16 corners, bicubic mode) exactly as
      `TNNetBilinearResize`/`TNNetBicubicUpsample` do; the SIZE-based source maps
      (`BilinearResizeMap`/`BicubicResizeMap`) drive the gather, nearest stays a
      pure host Move, backward stays on the CPU, host round-trip is the
      `FShouldOpenCL` fallback. `TestResize2DOpenCLParity` covers bilinear
      upscale/downscale/rectangular + bicubic upscale/rectangular vs CPU < 1e-4
      (ran on the PoCL CPU device); skips cleanly with no device.

## Lucky-day batch 2026-06-28k (verified-novel AVX / dedup / OpenCL — generative vision)

(Checked against `neural/*.pas` before listing on 2026-06-28. `TNNetAdaIN`
exists with a full CPU `Compute`/`Backpropagate` but has NO `ComputeOpenCL`/
`EnableOpenCL` — `grep -c 'TNNetAdaIN.ComputeOpenCL' neuralnetwork.pas` = 0 — and
its forward uses strided per-channel scalar access, unlike the AVX'd GroupNorm/
L2Normalize/PixelNorm reductions that already landed.)

- [X] AVX + dedup the `TNNetAdaIN.Compute` per-channel spatial statistics, and
      add a forward-only OpenCL offload. (DONE commit 1d86f6b3: depth-contiguous
      Add/MulAdd reduction, shared `NeuralPerChannelMeanInvStd` helper reused by
      GroupNorm ChannelsPerGroup=1 fast path, `cai_group_norm` Groups=Depth OpenCL
      offload; `TestAdaINForwardAVXParity` 3.58e-7 + `AdaINOpenCLParity`.) AdaIN (Adaptive Instance Normalization,
      Huang & Belongie 2017) is the core layer of `examples/AdaINStyleTransfer`
      and the StyleGAN-style generators. Today its forward computes the content
      mean/variance and the style mean/variance with FOUR nested
      `for CntX/CntY` loops that index `ContentOut[CntX, CntY, C]` /
      `StyleOut[CntX, CntY, C]` ONE element at a time — the stride-`Depth`
      column-gather pattern the depth-axis-contiguous rule warns against
      (`neuralnetwork.pas` ~line 44034). Every other per-channel/per-position
      reduction in the library was already converted to a depth-contiguous AVX
      accumulation: GroupNorm reduces each `(x,y)` column with `DotProduct`
      against a ones vector (~line 64560), and the per-CHANNEL L2Normalize path
      accumulates `ss[d] += x[d]*x[d]` over whole depth columns via the 3-pointer
      `TVolume.MulAdd` form (commit b01eab21). Do the same here: walk the spatial
      positions ONCE, accumulating per-channel `sum[d] += col[d]` (via `Add`) and
      `sumsq[d] += col[d]*col[d]` (via `MulAdd`) into two `Depth`-length scratch
      vectors, then finish mean/inv-std element-wise — no transpose, no
      stride-`Depth` scalar gather. The content half of this is exactly the
      InstanceNorm reduction (`TNNetInstanceNorm` = `TNNetGroupNorm` with
      `Groups=Depth`, where GroupNorm's column `DotProduct` degenerates to a
      single-element call per `(x,y)`), so factor the per-channel mean/inv-std
      accumulation into a shared `NeuralPerChannelMeanInvStd` helper that AdaIN
      and InstanceNorm both call (dedup, not a fork). Then add
      `TNNetAdaIN.EnableOpenCL`/`ComputeOpenCL`: the content instance-normalize
      reuses the existing per-position reduction kernel family (`cai_group_norm`
      with `Groups=Depth`, or the `cai_l2norm_perdepth` mean/rsqrt shape) and the
      `style_std*(x-mean)*inv_std + style_mean` affine is a per-channel
      scale+shift on the device — keeping a device-resident generator stack from
      bouncing the activation to the host between convs. Forward-only (backward,
      and the `FNormContent`/`FContentMean`/`FContentInvStd` caches it needs, stay
      on CPU); parity-test scalar-vs-AVX (wide `Depth`, `SizeX<>SizeY`) and
      device-vs-CPU, skip-clean when no device. Hot path of `AdaINStyleTransfer`
      and `StyleGAN2Generate`.
