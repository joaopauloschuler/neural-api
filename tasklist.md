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
  `TNNetMaxChannel` for global max pooling. Note: `TNNetMaxChannel`
  currently assumes square feature maps (`SizeX == SizeY`); the deleted
  `TNNetGlobalMaxPool` had a direct (X, Y) loop that also worked on
  rectangular inputs. If you ever need global max on a non-square
  tensor, fix `TNNetMaxChannel` rather than reintroducing this class.
- `TNNetGlobalMinPool` — overlapped `TNNetMinChannel`. Use
  `TNNetMinChannel` for global min pooling. Same square-only caveat as
  `TNNetMaxChannel` above.
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

## New layer types
- [X] TNNetDeformableConv — deformable convolution (Dai et al. 2017): a conv
      whose KxK sampling grid is shifted by learnable, per-output-location
      (and per-tap) 2-D offsets, so the receptive field adapts to content
      instead of being a rigid axis-aligned window. This is a genuinely new
      paradigm in the fork — distinct from regular/dilated conv (fixed grid),
      TNNetInvolution (location-specific *kernels*, fixed grid), and
      TNNetGroupConvP4 (rotation copies of a fixed grid): here the *sampling
      positions themselves* are predicted and bilinearly interpolated.
      Plan: (a) a small "offset head" sub-conv producing 2*K*K offset maps
      from the same input; (b) forward = gather each tap via bilinear
      interpolation at (base + offset), then the usual weighted sum;
      (c) backward must propagate gradients into BOTH the conv weights AND the
      offsets through the bilinear-interpolation weights (the offset gradient
      is the interesting part — dOut/d(offset) flows through the 4 corner
      pixels' interpolation coefficients). Zero-init the offset head so the
      layer starts identical to a plain conv. Deliverables: leaf layer with
      `// Coded by Claude (AI).`, LoadFromString/serialization wiring,
      numerical-gradient tests for weights AND offsets (bound inputs so the
      bilinear kinks at integer positions don't trip finite differences),
      shape-inference smoke entry, and an examples/DeformableConv/ demo on a
      geometrically-warped tiny dataset (rotated/scaled MNIST digits) showing
      it beats a param-matched rigid conv.
  - [ ] Optional v2: modulated deformable conv (DCNv2) adding a learnable
        per-tap sigmoid amplitude (modulation mask).
- [ ] TNNetSpectralConv2D follow-ups (the 2-D Fourier Neural Operator leaf layer
      + examples/SpectralConv2D/ resolution-invariance demo + numerical-gradient/
      shape/save-load tests all LANDED 2026-06-06 on a2; separable 2-D radix-2 FFT
      reusing FourierMixFFT, ModesX*ModesY low-pass truncation, exact complex
      adjoint backward): the PDE-surrogate example LANDED 2026-06-07 on a2
      (commit 47b93ea, examples/DarcyFlowFNO/) — coefficient→solution map for the
      LINEAR Poisson member of the Darcy family (-Δu = (a-1), a=exp(band-limited
      field), Jacobi-solved target, h² folded into the source so the same continuous
      operator holds at any resolution); lift→AddFourierNeuralOperator2D(width=8,
      6×6 modes)→project trains relL2 ≈0.64→0.025 at 16×16 and the SAME weights
      (CopyWeights) hold 0.0284 on an unseen 32×32 grid (resolution invariance).
      Open follow-ups:
      - [ ] The FULLY NONLINEAR -div(a grad u)=f Darcy operator does NOT converge
            under plain SGD within the CPU budget (single linear spectral conv can't
            represent it; deeper/ReLU FNO stacks freeze at ~zero or destabilize).
            Revisit with a better optimizer / normalization / curriculum, or a
            deeper FNO stack tuned to actually fit the non-diagonal operator.
      - [ ] FFT-path FPU denormal/invalid-op traps in TNNetSpectralConv2D needed an
            example-side SetExceptionMask workaround — consider masking/guarding the
            denormals inside the layer's FFT so callers don't have to.
- [ ] TNNetImplicitLongConv / AddHyenaOperator follow-ups (the leaf layer,
      order-2 builder, numerical-gradient + save/load tests, and the
      examples/HyenaOperator/ recall bake-off all LANDED 2026-06-05):
      (a) FFT-based O(L log L) forward/backward path as an opt-in fast mode for
          long sequences — the current forward is the direct O(L^2) causal
          time-domain sum, fine for small sizes but quadratic in SeqLen. Gate it
          so the exact time-domain path stays the default and assert FFT-vs-direct
          equivalence to <1e-5.
      (b) wire AddHyenaOperator into the downstream ../gpt-3-for-pascal decoder as
          an attention-free block option and contrast its loss / wall-clock vs the
          attention decoder on the existing tiny corpus.
- [X] TNNetCapsule follow-up (TNNetCapsuleSquash + TNNetCapsuleRouting — the
      squash nonlinearity, the fixed-iteration routing-by-agreement loop,
      LoadFromString wiring, and numerical-gradient + serialization tests all
      landed): the reconstruction-decoder pose-perturbation STRETCH goal — feed
      the winning digit-capsule's output vector to a small reconstruction
      decoder, perturb one dimension, and show it varies an interpretable pose
      factor (stroke thickness / skew). DONE 2026-06-07 on a2:
      examples/CapsuleReconstruction trains a CapsNet with the paper's MARGIN loss
      + masked-reconstruction MSE on a SYNTHETIC 12x12 bars set (controllable
      thickness/position pose factors — chosen over a real MNIST parse to stay in
      the <5-min CPU budget; ~17 s). Sweeping the most-responsive pose dimension
      varies the decoded bar SMOOTHLY and monotonically (stroke intensity/
      thickness; total ink 32.6->24.9). Accuracy: CapsNet 100% vs param-matched
      plain MLP 100% (easy 2-class task; the capsule win is the interpretable
      pose vector, reported honestly).

- [ ] TNNet.AddDeepEquilibriumBlock follow-up (builder + examples/DeepEquilibrium/
      landed 2026-06-05; weight-tied f iterated to its fixed point, jacobian-free
      PHANTOM backward, TNNetDeepEquilibriumSharedConv per-forward weight cache,
      and shape/convergence/gradient/save-load tests): (a) the EXACT
      implicit-function-theorem gradient (inverse-Jacobian solve via a second
      fixed-point iteration) vs the phantom approximation; (b) spectral /
      contraction constraints so convergence is guaranteed at arbitrary init
      (v1 uses damped Picard + output bounding, not guaranteed).

- [ ] TNNetRetention follow-up (layer + TNNet.AddRetention builder +
      examples/RetentionDualForm/ all landed; learnable-gamma variant
      TNNetRetention.Create(d_k, gamma, LearnGamma:=true) — gamma=sigmoid(raw)
      trained scalar — LANDED 2026-06-06 with gradient + serialization tests and
      a learnable-gamma example arm). Remaining: the chunkwise-recurrent hybrid
      form (a throughput optimisation skipped in v1 — the parallel and
      naive-recurrent forms both landed).

- [ ] TNNetHyperLinear follow-ups (weightless context-generated-weights leaf layer
      + examples/HyperNetwork/ + two-path gradient tests + the
      TNNet.AddHyperLinear(Din, Dout, ContextLayer) generator+HyperLinear builder
      all LANDED 2026-06-05; first layer in the fork that owns zero trainable
      weights — reads its W from a second input tensor). Deferred follow-up:
      CHUNKED weight generation so the main layer can be larger than the
      generator's output width (generate W in tiles) — the landed layer
      generates the whole Din*Dout matrix in one shot, which caps main-layer
      size; document the memory/param trade-off.

## Interesting applications / examples
- [ ] MahalanobisOOD follow-up: the AUROC / Mann-Whitney-U rank helper currently
      lives LOCAL to the example. If a second consumer appears (calibration ECE
      report, anomaly autoencoder, etc.), promote it to a public function in
      neuralvolume.pas / neuralcalibration.pas with its own unit test (pin AUROC
      against a hand-computed tiny example including ties).
- [ ] Forward-Forward follow-up: scale to a tiny-MNIST few-class subset (the
      paper's actual task) and report whether the per-layer local objective still
      beats chance within the <5-min budget. Builds on the landed
      examples/ForwardForward/.
- [ ] Forward-Forward follow-up: deeper FF stack (4+ layers) — does
      accumulated-goodness accuracy keep improving with depth, or does the
      length-normalised signal saturate?
- [ ] Reinforcement learning: minimal DQN solving CartPole or a grid world
- [ ] Style transfer or diffusion-lite denoiser (building on SuperResolution / VisualGAN)
- [ ] Growing Neural Cellular Automata demo (`examples/NeuralCellularAutomata/`) —
      reproduce Mordvintsev et al. 2020 "Growing Neural CA" on a TINY pure-CPU
      target (e.g. a 16x16 RGBA emoji-like glyph, channels = 4 visible RGBA +
      ~8 hidden state = 12-deep grid). One CA "rule" step is a shared-weight
      conv stack applied in place: per-cell perceive (fixed 3x3 Sobel-x/Sobel-y/
      identity depthwise filters, or a small learned 3x3 conv) -> 1x1
      TNNetPointwiseConvReLU -> 1x1 TNNetPointwiseConvLinear update added
      residually to the grid, with a stochastic per-cell update mask and an
      alpha>0.1 "alive" mask. Train by UNROLLING T in {48..64} steps sharing one
      rule via TNNetConvolutionSharedWeights (the SharedWeights layer is the key
      enabler — without it each step would learn separate weights), L2 loss to the
      target RGBA at the final step, pool-based sample replacement for
      persistence. Headline payoff: a net that GROWS the target from a single
      seed pixel and (stretch) REGENERATES after the grid is damaged — visually
      striking and conceptually unlike anything in the suite (it is recurrent-in-
      space self-organisation, not a feed-forward classifier or a diagnostic
      report). Render frames as ASCII/ppm so it stays dependency-free. Feasibility
      risk to settle in the first version: confirm the unrolled shared-weight
      gradient flows correctly under the SetBatchUpdate(True) idiom across T steps
      (the manual-gradient gotcha noted in [[manual-gradient-and-snapshot-gotchas]]);
      if full backprop-through-time over 48+ steps blows the CPU budget, fall back
      to a shorter T or truncated BPTT and document it (the same "what did NOT fit
      the budget" honesty the Grokking entry uses). Distinct from VisualGAN
      (adversarial image synthesis), SuperResolution (feed-forward upscaler) and
      DiagonalSSM (1-D sequence state space, not a 2-D self-organising grid).
- [ ] Neural ODE follow-ups (builder `TNNet.AddNeuralODEBlock` + `examples/NeuralODE/`
      landed 2026-05-31, Euler-only, trains via stored-activation backprop through the
      unrolled steps). Deferred:
        - [ ] Adjoint-sensitivity O(1)-in-Steps backward (integrate the adjoint ODE
              backwards using the `SetBatchUpdate(True)` weight-accumulation idiom).
              Shares the O(1)-memory goal with the open `TNNetReversibleBlock`
              recompute path and the "Gradient checkpointing" infra task.
- [ ] TNNetHamiltonianCell follow-ups (leaf layer + input/weight numerical-gradient
      tests + LoadFromString round-trip + examples/HamiltonianPendulum/ all LANDED
      2026-06-07 on a2, commit 4fd7309):
      - [ ] builder TNNet.AddHamiltonianBlock (skipped at landing; leaf+tests+example
            were the priority).
      - [ ] separable H(q,p)=T(p)+V(q) (kinetic+potential split) variant that makes the
            leapfrog step exact and cheaper.
- [ ] Reptile follow-up (TNNetReptileMetaTrainer + examples/MetaLearningReptile/
      landed 2026-06-01, sine-regression task distribution, manual inner-loop
      Compute/Backpropagate/UpdateWeights path): (a) a CLASSIFICATION task
      distribution (e.g. rotated/shifted 2-D blob few-shot tasks) so the
      meta-init claim is shown on more than 1-D regression; (b) the tiny ReLU
      net proved numerically delicate under the summed full-batch gradient —
      ForceMaxAbsoluteDelta was a no-op on the manual path, so it needed input
      normalisation + a small stable LR. Worth a short note in
      docs/ (or the example README) on why the manual UpdateWeights path
      bypasses the clip and what the stable-training recipe is, so the next
      manual-inner-loop example (HyperNetwork, MAML, Growing-CA) doesn't
      rediscover the divergence the hard way.

## Infrastructure / dev experience
- [ ] **Gradient-verification coverage audit.** For a *scientific-discovery*
      library the cardinal sin is a silently-wrong result, so every layer with
      a backward pass must have a numerical-gradient test proving Compute's
      analytic gradient matches a finite-difference estimate. Task: enumerate
      all `TNNet*` layers (esp. the 165 Claude-added ones — see the Authorship
      convention section), cross-reference against the cases in
      `tests/TestNeuralNumerical.pas`, and produce a coverage list of which
      layers are gradient-verified vs. compile-only. Then backfill tests for
      the gaps, prioritising layers with hand-written backward passes (loss
      heads, attention variants, sparsemax, the SSM) over thin activation
      subclasses. Treat "has a numerical-gradient test" as the definition of
      done for any future layer. Watch the shared-RNG ordering sensitivity in
      that test unit (reseed `RandSeed := 424242`, don't loosen tolerances).
- [ ] Mixed-precision (FP16) volumes for the OpenCL path
- [ ] Gradient checkpointing for training deeper nets in less memory
- [ ] Model zoo loader that pulls pre-trained weights from the companion repo
- [ ] ONNX import
- [ ] ONNX (or simpler JSON) export path — minimal viable: dump a
      forward-only graph for the currently-supported subset of layers,
      enough to run inference in onnxruntime. Doc which layers are
      out-of-scope for v1.
- [ ] CI-friendly headless test runner with coverage reporting

## Documentation / learning
- [ ] Interactive "build your first transformer in Pascal" tutorial
- [ ] Auto-generated layer API reference from doc comments

## Added ideas

### Smaller follow-up ideas
- [ ] Multi-threaded determinism investigation: SeededReproducibility passes
      with `MaxThreadNum := 1` but no test pins what breaks at `>1`. Add a
      sibling demo (or extend it) that runs with `MaxThreadNum := 4` twice
      and prints which weights diverge first — useful starting point for any
      future "make TNeuralFit deterministic under parallelism" work.

### Ideas from JP
- [ ] Better integrate TBytePredictionViaNNet and TEasyBytePredictionViaNNet with
      TNNet. Find a way to backpropagate. May open a new class of problem solving.
      This is very interesting.
- [ ] Use TBytePredictionViaNNet and TEasyBytePredictionViaNNet as inspiration for
      new float32 based layers. This is also curious and interesting.
- [ ] More image generative examples and or experiments.

### Attention variants / siblings

- [ ] TNNet.AddMultiHeadLatentAttention follow-up (builder + examples/LatentAttention/
      landed 2026-06-05, NoPE; down-proj x->c_KV + per-head K/V up-projections +
      per-head SDPA + DeepConcat + out-proj, shape + input-gradient + save/load
      tests, MLA-vs-MHA copy bake-off): (a) the paper's DECOUPLED-RoPE slice — a
      separate rope-only Q/K slice concatenated to the content slice before the
      dot product (RoPE cannot be applied to the compressed latent because the
      up-projection would smear positions); (b) the headline KV-cache win, which
      needs the open [[KV-cache incremental-decode]] path.
- [ ] GQA follow-up: wire AddMultiHeadGroupedQueryAttention into the downstream
      ../gpt-3-for-pascal decoder and compose with the open [[KV-cache
      incremental-decode]] task — the KV footprint shrinks by QueryHeads/KVHeads,
      exactly the bottleneck that task fights.

- [ ] TNNetDifferentialAttention follow-up: fold differential heads into the
      MHA breakdown ([[TNNetMultiHeadSelfAttention]] /
      TNNetTransformerDecoderBlock) behind a flag, so a decoder block can opt
      into differential attention per head — a natural drop-in for the
      downstream ../gpt-3-for-pascal long-context retrieval.
- [ ] TNNetSinkAttention follow-up: fold sink slots into the MHA breakdown
      ([[TNNetMultiHeadSelfAttention]] / TNNetTransformerDecoderBlock) so a
      decoder block can opt into sinks per head behind a flag.
- [ ] TNNetTalkingHeadsProjection — pre/post-softmax linear mix across
      heads (Shazeer et al.). A tiny learnable HxH multiply applied to
      attention logits along the head axis. NOTE (2026-05-31): this repo has
      NO single head-axis tensor — multi-head attention is built from H
      SEPARATE TNNetScaledDotProductAttention layers (each FAttn is
      [key,query,1]) concatenated along Depth. So a clean standalone HxH-mix
      layer has nothing to operate on. To do this properly, either (a) add an
      explicit multi-head SDPA tensor representation first, or (b) scope it as
      a BUILDER that inserts the HxH mix between the per-head logit slabs
      inside AddMultiHeadSDPAConcat / AddSplitQKVHeads — not a drop-in layer.
      Re-scope before attempting.

### Bake-off / experiment follow-ups
- [ ] Numerical-precision study: re-run the activation bake-off using FP32
      vs a simulated-FP16 path (round-trip volumes through fewer mantissa
      bits) and report the convergence-quality gap. Useful baseline for
      any future mixed-precision work.
- [ ] DropPath schedule study: linearly increasing drop probability with
      depth (Stochastic-Depth schedule) vs constant `p`.
- [ ] Lottery-ticket IMP follow-up (make the headline land): the landed
      examples/LotteryTicketIMP showed NO IMP advantage because the toy is too
      easy and the budget too generous (one-shot never collapses). Build the
      regime where IMP provably wins: either (a) STARVE the per-arm epoch budget
      (e.g. ~100 epochs/arm, matching the sibling that collapses) so one-shot
      buckles at 95% while IMP's rewind+retrain recovers, or (b) a harder task
      (overlapping/noisier classes or a deeper net) where the extreme-sparsity
      ticket is genuinely hard to find one-shot. Reuse the existing IMP harness;
      just swap the dataset/budget and re-grade. Pure CPU, <5 min.
- [ ] Init-scheme × depth heatmap: for depths {2, 4, 8, 16} and inits
      {Glorot, He, LeCun, plain N(0, 0.01)}, plot first-step gradient norm
      at the deepest layer.
- [ ] "Which init wins per activation" matrix: cross-product of init schemes
      × activation functions on a fixed tiny MLP, report epochs-to-converge.
- [ ] First-batch gradient-norm heatmap across (depth, width, init):
      enumerate a small grid, print one number per cell.
### Composite blocks / builders I'd enjoy shipping

#### Attention / sequence
- [ ] KV-cache incremental-decode path for TNNetScaledDotProductAttention —
      the single biggest efficiency gap for autoregressive generation with
      the downstream ../gpt-3-for-pascal model. Today, sampling the next
      token re-encodes the entire prefix every step, so generating N tokens
      costs O(N^2) attention work. Add an inference-only mode that, given a
      one-token query at position t, appends its K and V to a persistent
      per-layer cache and attends over the cached keys/values [0..t] instead
      of recomputing them — turning per-step cost from O(t) to O(1) in the
      prefix length. Scope notes: (a) cache lives on the SDPA layer, gated by
      an explicit BeginIncrementalDecode / EndIncrementalDecode (or a
      FCacheEnabled flag) so the training forward/backward path is untouched
      and bit-for-bit unchanged — the cache only activates for single-token
      forward passes at inference; (b) RoPE/ALiBi position offsets must be
      driven by the running cache length, not the (now length-1) input
      SizeX, so positional encoding stays correct mid-stream; (c) provide a
      ResetCache for starting a fresh sequence and document the max-context
      preallocation. Headline correctness test, and a tiny
      examples/IncrementalDecode/ demo: feed a SeqLen sequence two ways —
      one full re-encode, and one token-at-a-time through the cached path —
      and assert the final-position logits match to < 1e-5 (the built-in
      faithfulness check), then print measured per-token wall-clock vs prefix
      length to show the O(t)->O(1) flattening. Builds on the existing
      AttentionWeights accessor and the MHA breakdown above
      ([[TNNetMultiHeadSelfAttention]] / TNNetTransformerDecoderBlock); a
      genuinely new capability, not a re-skin of an existing layer.
- [ ] SpeculativeDecoding follow-up: the toy `mod`-sum target distribution is
      fairly FLAT, so absolute accept rates are high even for a weak draft and
      the speedup headline is carried by the monotone accept-rate RISE, not the
      absolute %. Add a PEAKED-target variant (sharper next-token distribution,
      e.g. a near-deterministic rule + low-temperature target) so the
      weak-draft accept rate drops well below 1 and the calls-saved gap between
      a good and bad draft widens — a more discriminating speedup chart.
- [ ] SpeculativeDecoding follow-up: KV-cache composition — once the open
      KV-cache incremental-decode path lands, remove the per-verification-pass
      prefix recompute (the v1 demo recomputes the whole prefix each pass) so the
      two efficiency wins (fewer big-model calls x O(1)-per-step) compose.
- [ ] KV-cache / incremental-decode O(1)-per-step path for
      TNNetDiagonalSSM (a linear recurrence is O(1)-per-step by nature;
      the SDPA incremental-decode notes above apply doubly here).
- [ ] TNNetTokenHistoryPenalty follow-up: wire it into the downstream
      ../gpt-3-for-pascal generation loop (call `Apply` before the sampler
      and `RegisterToken` after each emit, `ResetHistory` per sequence) and
      show a qualitative before/after on a repetition-prone prompt — the
      class landed this lucky-day batch (neuralvolume.pas, 7 tests in
      tests/TestNeuralSamplers.pas) but no in-tree generator calls it yet.
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
- [ ] ShakeShake follow-up (a): PER-SAMPLE alpha/beta grain — the landed
      TNNetShakeShakeMerge / TNNetShakeDropMerge sample one alpha/beta/b_l per
      forward/backward PASS (per-batch). Sample an independent coefficient per
      batch-item instead (the paper's "Shake-Shake-Image" best variant). Needs a
      per-sample scalar broadcast in Compute/Backpropagate and the
      SetBatchUpdate(True) idiom from [[manual-gradient-and-snapshot-gotchas]];
      keep eval deterministic. Add a test that two samples in one batch get
      different effective scales.
- [ ] ShakeShake follow-up (b): examples/ShakeShakeReg/ demo — contrast an
      AddShakeShakeBlock stack vs a plain two-branch (deterministic 0.5/0.5)
      residual on a small noisy/over-parameterised task, charting the train/val
      gap narrowing (the headline regularisation win, à la the Mixup/SAM
      follow-ups). Pure CPU, <5 min, no binaries committed.

#### Channel attention / conditioning
- [ ] TNNetCBAM follow-up: the landed AddCBAM uses TWO SEPARATE channel MLPs
      (avg-branch + max-branch summed before the sigmoid) instead of the paper's
      single SHARED MLP applied to both pooled descriptors. If a weight-sharing
      mechanism for two parallel FC branches becomes easy (TNNetConvolutionSharedWeights
      is conv-only), revisit to share the reduce->ReLU->expand weights.
- [ ] TNNetCBAM follow-up: the landed spatial branch uses a LEARNED pointwise
      C->2 descriptor instead of the paper's FIXED avg-over-depth + max-over-depth
      channel reduction (no such (X,Y,1)-producing channel-axis reduction layer
      exists). If a fixed avg/max-over-Depth reduction primitive lands, offer it as
      the paper-faithful spatial-descriptor variant behind a flag.
- [ ] TNNetFiLM follow-up: a CLASS-CONDITIONAL generator/decoder demo — the
      headline FiLM use case (cGAN / conditional generation, Perez et al.). Build a
      tiny decoder (a couple of conv/upsample blocks) whose feature maps are
      FiLM-modulated by gamma|beta produced from a learned class EMBEDDING
      (TNNetEmbedding -> FC -> 2*Depth), and show one shared trunk emits visibly
      different per-class outputs on a small synthetic multi-class target. Fork the
      examples/FiLMConditioning/ wiring (already proves the conditioning FC trains
      end-to-end through TNNetFiLM) and scale it from a 1x1 feature vector to a
      small spatial map. Keep it CPU-only and well under 5 minutes.
- [ ] TNNetFiLM builder follow-up (AddFiLMConditioned landed 2026-05-31): a
      RESIDUAL/spatial-map variant. The landed AddFiLMConditioned uses a
      FullConnectLinear conditioning FC sized for a (1,1,C) cond vector. The
      class-conditional generator/decoder follow-up above needs FiLM over a
      SPATIAL feature map (D over SizeX>1,SizeY>1) — confirm the landed builder
      already broadcasts gamma|beta correctly per-channel over a spatial map
      (it should, FiLM modulates per Depth channel), and if a per-token/
      per-spatial-position conditioning is ever wanted, add a PointwiseConv
      cond path variant gated by a flag.
- [ ] TNNetMaxBlurPool follow-up: rectangular-input support — the landed layer
      inherits TNNetMaxPool's square-only (SizeX = SizeY) assumption. If a
      non-square blur-pool use case shows up, generalize the dense-max + blur
      loops to independent (X, Y) extents (the same caveat noted for the removed
      TNNetGlobalMaxPool at the top of this file) rather than forking a class.

#### Activations (gradient-checkable, mostly TNNetReLUBase descendants)
- [ ] TNNetMetaAconC follow-up: the FULL cross-channel-bottleneck β generator
      (the paper's true Meta-ACON: squeeze → FC channel-reduce → ReLU → FC
      channel-expand → sigmoid, so β[c] depends on ALL channels' spatial
      means, not just channel c's). The landed TNNetMetaAconC uses a per-channel
      affine-over-squeeze simplification; this variant needs a small two-FC
      sub-block inside the layer (or a builder that wires an SE-style squeeze
      into the β path) and is NOT a per-channel-transform shape, so scope it as
      its own layer/builder rather than a ChannelTransformBase descendant.
#### Probability projections / sparsity
- [ ] TNNetGumbelSoftmax / annealing follow-up: re-run the annealing AE on a
      LESS-separated (overlapping) cluster set so the high-tau bottleneck entropy
      actually starts near uniform (~ln K) and the sharpening sweep spans the full
      entropy range — the current well-separated toy lives in the confident-routing
      regime where even tau=2.0 is already near one-hot (honest caveat in its README).
      Pure CPU, ~2 s.
- [ ] AddMixtureOfDepths follow-up (builder + examples/MixtureOfDepths/ landed
      2026-06-01): (a) add a load-balancing / capacity-utilisation auxiliary loss so
      the router spreads its budget instead of fixating on a few positions; (b) a
      Gumbel/learned-threshold router variant (the v1 uses a sigmoid + the existing
      TNNetTopK top-Capacity mask); (c) the v1 example's learned allocation came out
      mostly POSITIONAL on the test seed rather than a sharp hard-vs-easy content
      split — design a next-token task where the triage provably tracks "hard"
      tokens to make the interpretability headline land.
#### Normalization primitives
- [ ] TNNetUnitNormConstraint hard-projection variant: a true *post-step hard
      projection* (renormalize the previous layer's weights after each update,
      non-differentiable) — still open if a hard constraint is ever wanted. The
      differentiable reparametrization (TNNetWeightNormLinear, landed) already
      covers the headline use case.

### Loss layers
- [ ] ArcFaceEmbedding follow-up: contrast the landed examples/ArcFaceEmbedding/
      against an actual plain-softmax head arm side by side — the demo currently
      shows the separation trend WITHIN ArcFace across margins m in {0,0.3,0.5},
      not ArcFace-vs-softmax head to head. Pairs with [[FeatureSeparability]] and
      the landed examples/CenterLossSoftmaxJoint/ (softmax-only vs softmax+center
      head-to-head).
- [ ] TNNetKLDivergence distillation follow-up
      (examples/KnowledgeDistillation/): temperature sweep T in {1,2,4,8} on this
      example — chart how soft-target sharpness changes the distilled student's
      accuracy/agreement.
- [ ] TNNetCenterLoss follow-up: cross-batch EMA-updated centers — needs a
      batch-aware loss hook (the per-sample FOutputError path is blind to other
      minibatch samples, the same limitation logged for a true cross-batch
      InfoNCE). Track alongside that batch-aware-loss-hook item.
- [ ] TNNetVectorQuantizer follow-up: EMA codebook update variant (van den Oord
      et al.'s recommended alternative to the codebook-loss gradient) — track a
      cross-batch EMA of assigned encoder vectors per code. Needs the same
      batch-aware loss hook logged for cross-batch InfoNCE / CenterLoss-EMA; track
      alongside those.
### Training infrastructure (the "missing plumbing")
- [ ] SWA/EMA integration follow-up: the landed TNNetSWAWrapper / TNNetEMAWrapper
      (neuralnetwork.pas) are standalone wrappers the CALLER must drive — nothing in
      TNeuralFit calls them yet. Wire an optional hook into the training loop (call
      EMA Update each step / SWA Accumulate every N steps after epoch W, gated behind
      an opt-in property so the default path is byte-for-byte unchanged), plus an
      examples/WeightAveraging/ demo that contrasts the live net vs the SWA-averaged
      and EMA-shadow nets' val accuracy on a tiny noisy task (the headline SWA/EMA
      win is a flatter, better-generalising averaged solution). Mirror the open
      TNeuralLRScheduler-wiring follow-up's "opt-in, regression-test the default is
      unchanged" discipline.
- [ ] TNNetGrokfastWrapper follow-up (caller-driven slow-gradient amplifier in
      the SWA/EMA/Lookahead wrapper family landed 2026-06-05; per-weight gradient
      EMA mu := beta*mu + (1-beta)*g rewriting live gradients g := g + lambda*mu,
      three tests in TestNeuralTraining.pas): wire it into the still-open
      examples/Grokking/ demo as the Grokfast-on vs -off grok-epoch contrast (the
      published fix for that demo's <5-min pure-CPU budget). Also ship the paper's
      cheaper Grokfast-MA (windowed moving average) variant alongside the EMA form.
- [ ] Layerwise learning-rate multipliers — per-layer `LRMult` field that
      the optimizer respects. Unlocks discriminative fine-tuning.
- [ ] Mixup follow-up: the landed examples/Mixup/ toy is LINEARLY SEPARABLE so both
      the plain and mixup-augmented arms hit ~100% val accuracy — the helper is
      pinned by unit tests but the demo does not yet SHOW mixup winning. Add a
      harder/over-parameterised arm (label noise, overlapping clusters, or a small
      net trained to memorise) where the soft mixup targets measurably narrow the
      train/val gap, so the regulariser's benefit is visible (mirror the SAM /
      RandomLabelMemorization follow-ups that make a flat saturating signal
      discriminating). Also exercise CreateMixedVolumePairList with general
      alpha != 1 (the demo path uses the Beta(1,1)=Uniform default).
- [ ] SAM follow-up: the noisy-label 2D-blob clusters are easily separable so
      clean val-accuracy saturates (~99%) across all rho — the flatness signal
      carries the story but the val-acc-vs-rho curve is flat. A harder task
      (overlapping clusters / higher label-noise / a tiny MLP on a small image
      stub) where SAM's flat minimum actually buys measurable val-accuracy over
      plain SGD would complete the demonstration. Builds directly on the landed
      examples/SharpnessAwareMinimization/.
- [ ] Muon optimizer follow-up (`examples/MuonOptimizer/` landed): the demo
      uses the published 5-step quintic, whose stable fixed points are
      `sigma ~ 0.868` / `~1.264` (NOT 1), so the orthogonalized update is only
      *semi*-orthogonal (singular values squeezed into ~[0.7,1.3], the headline
      check asserts that band rather than `||O^T O - I||_F ~ 0`). Possible
      follow-ups: (a) add a "tight orthogonality" variant that normalizes by the
      spectral norm and runs more iterations / coefficient schedule that
      actually drives sigma -> 1, and assert a strict `||O^T O - I||_F < 1e-3`;
      (b) make the bake-off favorable to Muon on a problem where it should win
      (wider hidden layers / anisotropic gradients) rather than the tiny 3-input
      toy where SGD/Adam are already saturated; (c) factor the
      `MatMul`/`MatTranspose` packed-matrix helpers (note the
      `DotProducts` output layout `out[b*NumAs+a]`) into a reusable utility if a
      second orthogonalization example ever needs them.
### Introspection / debugging tools
- [ ] ActivationStatsReport follow-up: the per-layer `|median|` is currently
      approximated from the last probe sample only (streaming moments keep
      memory bounded). Add an exact per-layer median across the whole probe
      batch via a two-pass or bounded reservoir approach if it proves worth
      the memory; the single-sample approximation is documented in the doc
      comment for now.
- [ ] WeightSpectrumReport follow-up: a probe-batch `JacobianSpectrumReport`
      variant that estimates the top singular value of each layer's
      input-output Jacobian (the entry already flags this as the natural
      next knob). Reuses the landed `TNNet.EstimateSpectralNorm` power-
      iteration helper.
- [ ] GradientConflictReport follow-up: the raw `cos<0` conflict fraction is
      dominated by a softmax head's mildly-anti-correlated cross-class pairs, so
      the report falls back on a `cos<-0.5` strong-conflict tail. Add a sibling
      run (or example variant) with a raw-logit `TNNetFullConnectLinear` head and
      document whether the plain `cos<0` fraction becomes discriminating there —
      the same raw-logit-vs-softmax-head question MarginReport's follow-up raises.
- [ ] GradientConflictReport follow-up: the per-class-pair mean-cosine matrix is
      the natural precursor to gradient-surgery / PCGrad — add an experiment that
      reweights or projects out the most-conflicting class pair's gradient and
      charts the batch-loss delta.
- [ ] NeuralTangentKernelReport follow-up: extend the existing
      `examples/NeuralTangentKernelReport/` to contrast a WIDE vs NARROW hidden
      layer and show the wide net's NTK drifts less (lazy vs rich regime made
      visible) — the report machinery (incl. the landed `SnapshotB` drift arg) is
      in place, only the example arm remains.

### Bugs surfaced by the introspection-report batch
- [ ] `TNNetFlipX.Backpropagate` (and likely `TNNetFlipY`) range-check
      overflow when the NEXT layer is a padded convolution: the flip layer's
      `OutputError` is sized exactly to its output, but a padded conv writes a
      larger (padded) error region into it, overflowing. Surfaced while wiring
      an `Input -> FlipX -> Conv -> ...` flip-invariant net for
      EquivarianceReport (worked around by using a global-avg construction
      instead). Add a numerical-gradient / forward+backward regression test
      for `FlipX -> padded Conv` and fix the unpad sizing.

### Tests / numerical-gradient audit
- [ ] Shared `LayerInputAndWeightGradientCheck(layer, inputShape)` helper
      in tests/TestNeuralNumerical.pas. Three-line tests instead of
      copy-pasted blocks. Should handle both input and weight central-
      difference checks with a `Tolerance` parameter (default 1e-2) so
      the DeMaxPool-style Double-precision SSE accumulator can be opted
      into per-test.
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
- [ ] Layer-registry round-trip audit — for every concrete TNNet* in the
      LoadFromString/CreateLayer dispatch table, instantiate with defaults,
      save, load, save again, assert bit-for-bit string equality. Highest-
      leverage single test for the "added a layer but forgot to register
      it" bug.
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
- [ ] Promote DeMaxPoolFamilyGradientCheck's Double-precision SSE
      accumulator into the shared LayerInputGradientCheck (and weight-grad
      variant). Sum the SSE in Double; eps and tolerance stay TNeuralFloat.
      NEW DATA POINT: TNNetAdaptiveMaxPool's gradient check hit the same
      float32 subtractive-cancellation issue (a single cell carrying the
      whole window error, num=1.2588 vs ana=1.2709) and had to be loosened
      to tol 0.02 with an in-code comment — verified NOT a layer bug
      (double-precision central difference matches analytic exactly). A
      strong candidate to convert once the Double accumulator helper lands.
- [ ] Add a "FP32 SSE accumulator warning" comment near LayerInputGradientCheck
      pointing future audits at the DeMaxPool case and the Double-precision
      workaround.
- [ ] TNNetPointwiseSoftMax: now that the exact Jacobian lives in
      Backpropagate, opt cross-entropy training paths into the cheap
      (y - target) shortcut explicitly, and add a regression test that
      checks the shortcut and the full-Jacobian path agree to 1e-5.
- [ ] TNNetSoftmaxTemperature refactor attempt: extract a shared softmax-
      Jacobian helper so SoftMax / PointwiseSoftMax / SoftmaxTemperature
      reduce to one Backpropagate body parameterised by axis +
      inv-temperature. Pure refactor, gradient tests pin behavior.
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
- [ ] Scheduler unit tests — given seed and schedule parameters, NextLR
      must produce a deterministic, finite, monotonically-correct sequence.
- [ ] Backward-pass sign-correlation test — for every layer that overrides
      Backpropagate, perturb input by ±ε, assert gradient direction agrees
      with loss-difference direction >90% of the time across a small grid.
- [ ] Coverage matrix at the top of TestNeuralNumerical.pas: per-class
      `[grad] [serialize]` block, written by a small script.
- [ ] LoadFromString round-trip for the entire activation menagerie — one
      parameterised test walking every TNNetReLUBase descendant.

### Tooling / dev experience
- [ ] `scripts/new_layer.sh <Name>` scaffolder — drops a Compute/Backpropagate
      skeleton into neuralnetwork.pas plus a numerical-gradient test stub.
- [ ] `scripts/new_activation.sh <Name>` scaffolder — narrower version for
      the most common landing type.
- [ ] `scripts/check_layer_dispatch.sh` — grep every `TNNet... = class`
      line, cross-reference against the two CreateLayer dispatch tables
      and the LoadFromString cascade, print any missing class.
- [ ] `scripts/audit_landed.sh` — companion to `audit_tasklist.sh`. Every
      `[x]` line claiming a TNNet* landed must point at a real class in
      the dispatch AND at least one Test* method.
- [ ] `scripts/audit_tasklist.sh --strict` mode — only match lines starting
      with the canonical `- [ ] TNNet… — …` re-pin shape. Two-line awk
      change; cuts the false-positive rate.
- [ ] `scripts/list_untested_layers.sh` filter pass: drop Base/Class/Abstract
      names, emit file:line for surviving entries.
- [ ] `scripts/audit_one_layer.sh <TNNet...>` — bundles grep_layer.sh,
      list_untested_layers.sh, and the numerical-gradient test runner
      filtered to tests that mention the layer.
- [ ] `scripts/plot_activation.sh <TNNet...>` — sample at 41 points in
      [-5, +5], print forward values and numerical derivative as a tiny
      ASCII chart. Pairs with a one-shot Pascal helper.
- [ ] `tests/SmokeTest.lpr` — five fastest gradient checks, exits in under
      a second. Real signal for a future CI shim.
- [ ] `tests/TestActivationMenagerie.pas` smoke test — walks every
      TNNetReLUBase descendant and confirms forward returns finite values
      on (-5, +5) spanning input.
- [ ] `--quick` flag on the test runner that skips heavier `SeqLen > 4`
      cases / `Slow` / `Large` markers.
- [ ] `--quick` flag on RunAll.sh that passes through.
- [ ] CI-friendly summary mode for the test runner: print a single
      "N passed / M failed" line plus failing-test names to stderr.
- [ ] `bin/layer_bench <ClassName> <SizeX> <SizeY> <Depth>` — tiny CLI
      that builds a 1-layer net and reports ns/op for forward + backward.
      Subsumes the long-pinned Volume micro-benchmark and extends it to
      layers.
### Examples I'd enjoy writing
- [ ] EchoStateNetwork follow-up: an optional ridge closed-form readout solve
      (normal equations) as a deterministic alternative to the SGD readout loop,
      showing the classic ESN one-shot linear fit (the SGD loop is LR-sensitive).
- [ ] InductionHeads follow-up: ablate the causal mask and the second layer
      independently (1-layer causal, 2-layer NON-causal) and show in-context
      copy accuracy COLLAPSES to chance in each — a built-in proof that both the
      causal mask AND the two-layer composition are necessary, not incidental.
- [ ] InductionHeads follow-up: sweep the number of repeats / prefix length and
      chart the in-context learning score vs sequence position (the textbook ICL
      curve — loss should drop sharply at the first repeated position).
- [ ] DeadReLU follow-up (open): chart the LR=1.0+ chaotic regime the LR-sweep
      demo deliberately excluded — above the monotone band ReLU's dead fraction
      stops climbing cleanly (bounces ~14% at LR=1.0). A finer high-LR grid with
      a few seeds (mean +/- std) would show whether the bounce is a seed artifact
      or a real saturation/recovery transition.
- [ ] `examples/AnomalyAutoencoder/` — train an autoencoder on MNIST
      digit "0", evaluate reconstruction error on all 10 digits, print
      AUROC.
- [ ] `examples/SpokenDigitKWS/` — 1D-conv keyword-spotting on FSDD:
      MFCCs → 1D conv stack → classification.
- [ ] `examples/TimeSeriesForecast/` — one-screen forecasting demo on a
      synthetic seasonal+trend series with a 1D-conv or tiny attention
      model.
- [ ] `examples/GradientFlowVisualizer/` — train a deep MLP with and
      without LayerNorm/RMSNorm and print per-layer gradient-norm tables
      across steps.
- [ ] `examples/NormalizationBakeoff/` — same idea comparing no-norm /
      BatchNorm / LayerNorm / RMSNorm / GroupNorm / InstanceNorm /
      ChannelStdNorm.
- [ ] OptimizerBakeoff follow-up: a per-optimizer LR shoot-out. The landed
      demo HOLDS LR fixed (0.05 SGD-family / 0.01 Adam-family) to isolate the
      update rule, which leaves plain SGD stalling around 1e-1 while the
      others converge — fair for "same LR, different rule" but not a tuned
      comparison. Add a variant that sweeps a small LR grid per optimizer and
      reports each at its OWN best LR, so the "with tuning, plain SGD also
      converges" caveat in the README becomes a chart. Library note for the
      builder: RMSProp is reached via the Adam path with Beta1=0
      (InitAdam(0.0, beta2, eps) + CalcAdamDelta/UpdateWeightsAdam); plain
      SGD/momentum is SetLearningRate(lr, inertia) + UpdateWeights — both
      documented in examples/OptimizerBakeoff/README.md.
- [ ] `examples/EmbeddingVisualization/` — contrastive head on a 4-class
      toy 2D dataset, dump learned embeddings to CSV with README plotting
      instructions.
- [ ] `examples/MixUpAblation/` — train SimpleImageClassifier with and
      without MixUp on CIFAR-10 and report the delta.
- [ ] `examples/AttentionViz/` — load a tiny trained SDPA model and dump
      the per-head attention matrix as a PGM image.
- [ ] `examples/BiasOnlyTuning/` — freeze a pretrained classifier and
      fine-tune only inserted TNNetChannelBias layers on a new task
      (BitFit-style cheap adaptation). NOTE (2026-05-31): the landed
      examples/AffineFineTune/ already provides the freeze idiom (per-layer
      `LearningRate := 0` + net inertia 0, asserting base weights are
      bit-identical after fine-tune) and the trainable-param-count comparison —
      fork it and drop the TNNetChannelMul half so only bias adapts, then chart
      bias-only vs full-affine (mul+bias) target accuracy at matched frozen
      trunk. This is now a ~30-line diff on AffineFineTune, not a from-scratch
      example.
- [ ] AddAffineBlock follow-up (landed 2026-05-31): the landed AffineFineTune
      inserts a `TNNetReshape(1,1,C)` before each affine block because
      TNNetFullConnect emits units on the SizeX axis while TNNetChannelMul/Bias
      scale per-Depth. Consider an `AddAffineBlock` overload (or a sibling
      builder) that auto-reshapes a (C,1,1) feature vector to (1,1,C) so the
      per-channel affine "just works" on dense-layer output without the manual
      reshape — verify it stays a no-op on already-(*,*,C) conv output.
- [ ] `examples/ReZeroDeepMLP/` — train a 16-layer residual MLP with and
      without TNNetReZero on each residual branch on the hypotenuse toy.
- [ ] `examples/SpaceToDepthStem/` — show the SpaceToDepth → Conv stem
      replacing a stride-2 conv on a tiny CIFAR stub.
- [ ] `examples/MaxoutMnist/` — minimum-viable Maxout demo on a tiny-MNIST
      subset (or synthetic 2D classification).
- [ ] `examples/SWADemo/` — CIFAR-10 baseline vs same network with SWA
      enabled from epoch 75% on.
- [ ] `examples/LossLandscapeCompare/` — MSE vs LogCosh vs Huber vs MAE
      on Hypotenuse with a handful of injected outliers.
- [ ] `examples/QuantizationAwareMnist/` — STE-MNIST demo: baseline vs
      STE on penultimate activation, compare test accuracy and final-weight
      histograms.
- [ ] `examples/CharbonnierSR/` — minimal variant of SuperResolution that
      swaps the MSE head for TNNetCharbonnierLoss and prints PSNR delta.
- [ ] `examples/SparseAttentionDemo/` — toy "predict next char of a
      periodic sequence" using Sparsemax in place
      of softmax over a tiny K|V bank. Print attention-weight histogram
      per step.
- [ ] `examples/TripletEmbedding/` MNIST follow-up: a true MNIST version of the
      landed synthetic TripletEmbedding demo, with a PGM scatter-plot output.
- [ ] `examples/VQAutoencoder/` — extend VisualAutoencoder with a
      TNNetVectorQuantizer bottleneck.

- [ ] `examples/ReverseXYAugmentation/`, `examples/AutoencoderMNIST/`,
      `examples/AutoencoderReconstructionGrid/` — additional small demos.
- [ ] `examples/ActivationPlayground/` — prints one CSV row per activation:
      name, forward ns/op, backward ns/op, output range on [-8, 8],
      derivative range, "is monotone?" check.
- [ ] `examples/ActivationGallery/` — constructs a single-layer net per
      activation, sweeps inputs `x ∈ [-5, 5]`, prints `(x, y, dy/dx)`.
      Smoke test that every activation's forward + cached derivative
      agree with their textbook formula.

### Experiments I'm curious about (additional)
- [ ] TNNetInformationPlaneReport follow-up: examples/InformationPlane/ LANDED
      2026-06-07 on a2 (commit dd5e983) — tiny tanh FC classifier whose binning-MI
      information-plane trajectory reproduces the compression bend (tanh arm) while
      a ReLU arm stays flat (the Saxe et al. 2018 contrast); MI is computed inline.
      The optional `TNNet.InformationPlaneReport` introspection method
      ([[introspection-report-pattern]]) was NOT added — promote the binning-MI
      collector to a report method if a second consumer appears.
- [ ] `examples/CurriculumLearning/` — does **sample ordering by difficulty**
      help? Reproduce the curriculum-learning effect (Bengio et al. 2009,
      "Curriculum Learning") on a tiny self-contained task and, crucially, ship
      the contrast arm that shows when it does NOT, in the house "what did NOT
      reproduce" style. Pipeline, no new layer/gradient machinery: (1) build a
      small synthetic classification set with a controllable difficulty signal
      per sample — e.g. 2-D two-spirals or concentric-rings where difficulty =
      radius / closeness to the decision boundary, OR a noisy-label task where a
      known fraction of labels are corrupted (corrupted = hardest). (2) Score
      every training sample's difficulty ONCE up front with a cheap proxy
      already in the library: the per-sample loss / classification margin from a
      brief warm-up model (reuse the margin/difficulty scoring that
      [[MarginReport]] and the "hardest examples" pool already compute — do NOT
      invent a new scorer). (3) Train four arms from the SAME seed/init/budget,
      differing only in the order/schedule in which samples enter the minibatch
      stream: **easy->hard** (curriculum, difficulty threshold relaxed over
      epochs — "baby steps" pacing), **hard->easy** (anti-curriculum),
      **random** (the honest baseline), and **easy-only** (drop the hardest
      tail entirely — a difficulty-based data-pruning control). Headline plot is
      an ASCII val-accuracy-vs-epoch curve per arm plus final test accuracy, so
      the reader sees both the convergence-SPEED story (curriculum's main
      documented win is faster early progress / better optimization on hard,
      structured tasks) and the final-accuracy story (often a wash on easy
      tasks). HONEST framing, because the literature is genuinely mixed: cite
      Hacohen & Weinshall 2019 ("On the Power of Curriculum Learning") for when
      ordering helps, and state plainly that on a clean, easily-separable task
      random order frequently ties or beats curriculum — the robust,
      reproducible signal here is (a) curriculum's faster EARLY epochs and (b)
      anti-curriculum's reliably WORSE early epochs, not a final-accuracy
      miracle. On the noisy-label variant, the expected and most useful result
      is the opposite-of-naive one: feeding the hardest (mislabeled) samples
      first hurts, and easy-only data-pruning can match full-data training —
      tying this experiment to the label-noise / [[FisherImportance]] /
      RandomLabelMemorization thread already in the repo. This is a DIFFERENT
      axis from everything shipped: it varies the DATA ORDER at fixed
      architecture/optimizer, whereas the scheduler/optimizer bake-offs vary the
      training rule and the pruning experiments vary the weights. Pure CPU, tiny
      MLP, deterministic seed, <5-min budget.
- [ ] `examples/MaximalUpdateParametrization/` (optionally backed by a
      `TNNet.CoordinateCheckReport` introspection method,
      [[introspection-report-pattern]]) — demonstrate **muP / hyperparameter
      transfer across width** (Yang & Hu 2021, "Tensor Programs V: Tuning Large
      Neural Networks via Zero-Shot Hyperparameter Transfer"). Train the SAME
      tiny MLP at a ladder of hidden widths (e.g. 32 / 128 / 512 / 2048) and
      sweep the learning rate at each width, under two parametrizations:
      (a) standard parametrization (SP) — the usual `1/sqrt(fan_in)` init the
      library already uses, plain global LR; (b) muP — input/output-layer and
      hidden-layer init variances and per-layer LR multipliers rescaled by
      width per the muP table (hidden LR scaled `1/width`, output logits scaled
      `1/width`, input untouched). The headline is the LR-vs-loss curve: under
      SP the loss-minimizing LR DRIFTS leftward as width grows (the optimum you
      tuned on the narrow net is wrong for the wide one), while under muP the
      curves' minima stay ALIGNED at a single width-invariant LR — so you can
      tune on width-32 and transfer the winner straight to width-2048 with no
      re-tuning. Ship the *coordinate check* as the mechanistic diagnostic that
      explains WHY: log each layer's mean absolute activation (`GetAvgAbs`) at
      init and after a handful of SGD steps across the width ladder (mean abs =
      `GetSumAbs/Size` on each layer's `Output`) — under muP
      every layer's coordinate magnitude stays O(1) / flat vs width, under SP
      the pre-logit activations grow (or the updates do), which is exactly the
      mechanism that shifts the stable LR. This is a DIFFERENT axis from
      everything shipped: it's about *parametrization & width-scaling of
      training dynamics*, not representation geometry
      ([[RepresentationSimilarity]]), posterior degeneracy / curvature
      (LocalLearningCoefficient, HessianCurvature, FisherImportance), or the
      single-width LR-stability threshold of [[EdgeOfStability]] (EoS asks "how
      large can LR be before divergence at ONE width"; muP asks "does the GOOD
      LR stay put as width changes" — orthogonal questions). HONEST headline in
      the house "what did NOT fit the budget" style: the clean transfer needs
      the full muP recipe (init scaling AND per-layer LR multipliers AND the
      `1/width` output scaling) — ship a built-in ablation arm that applies only
      the init half and show the minima still drift, so the example proves all
      three pieces are load-bearing rather than overclaiming from a partial
      implementation. Note the pitfall to document: the alignment is most
      visible when the narrowest rung is already wide enough to be in the
      asymptotic regime (width >= 32) and the LR grid is logarithmic and shared
      across widths; state that the robust, reproducible signal is the
      *alignment of the minima* and the flat coordinate-check, not the absolute
      loss values. Pure CPU, tiny MLPs, no new gradient machinery — only
      per-layer LR/init scaling (the per-layer `TNNetLayer.LearningRate`
      property and weight-init knobs already exist) plus forward-pass
      `GetSumAbs` activation collection.
- [ ] LogSoftMax+NLL vs SoftMax+CE convergence parity test: same seed,
      same tiny classifier, plot val-loss curves.
- [ ] Shrink-activation sparsity sweep: ReLU / SoftShrink / HardShrink as
      bottleneck activations, sweep lambda over `{0.1, 0.25, 0.5, 1.0}`,
      report (sparsity %, recon loss).
- [ ] Activation "kink at zero" finite-difference noise audit on every
      activation on a `[-0.05, 0.05]` window stepping by 1e-3.
- [ ] Periodic-activation toy benchmark — fit `y = sin(3x) + 0.3 sin(11x)`
      with TNNetSnake vs ReLU/GELU/Tanh MLPs of equal width/depth.
- [ ] Sinc-vs-Sin head-to-head on the SIREN-flavored fit.
- [ ] Dropout-vs-DropPath head-to-head — same small CIFAR model with
      element-wise TNNetDropout and whole-sample TNNetDropPath at matched
      effective drop rates.
- [ ] LeCunTanh-vs-Tanh ablation reproduction with a small seed/LR sweep
      on top of examples/HyperbolicActivationBakeOff/. Average over 5 seeds
      and report mean+std.
- [ ] LossFamilyBakeoff follow-up: a multi-seed (e.g. 5 seeds, mean ± std)
      variant so the ranking AMONG the robust heads is statistically
      meaningful — the landed single-seed run cleanly separates MSE from the
      robust group but the ordering within {Huber, SmoothL1, Charbonnier,
      LogCosh} is seed-dependent (Huber == SmoothL1 at the default delta=1).
      Sweep the outlier fraction / magnitude too, charting clean-test MSE vs
      contamination level per head. Keep dims tiny so 5 seeds still fit the
      <5-min budget. Mirrors the open GatedFFNBakeoff multi-seed follow-up.
- [ ] GatedFFNBakeoff follow-up: a multi-seed (e.g. 5 seeds, mean ± std)
      variant of the landed examples/GatedFFNBakeoff/ so the gate ranking is
      statistically meaningful rather than a single-seed snapshot. Keep the
      per-arm dims tiny so 5 seeds x 5 gates still fits the <5-min budget.
- [ ] LogCoshDualExperiment longer-horizon follow-up: 200-300 epochs and
      5 seeds (mean ± std reporting).
- [ ] Plain-Tanh vs TanhGLU FFN ablation in a minimal-transformer-without-
      attention skeleton.
- [ ] DyT-vs-LayerNorm bake-off — a 30-line swap in the existing normalization
      bake-off harness (or a small standalone synthetic-regression A/B).
- [ ] GRN-as-drop-in: take SimpleImage CIFAR, swap each
      TNNetMovingStdNormalization for TNNetGRN and chart accuracy.
- [ ] TNNetChannelBias-vs-TNNetChannelMul ablation: train a small
      classifier four ways — (a) no affine, (b) bias only, (c) mul only,
      (d) both — print final accuracy and learnable params per variant.
- [ ] Maxout vs ReLU width-trade study at matched parameter count.
- [ ] Sinusoidal vs learned positional embedding head-to-head on the
      binary-addition task.
- [ ] PReLU vs LeakyReLU vs RReLU on a tiny CIFAR stub at matched param
      count. (Remember to flip TNNetRReLU's `Enabled` flag off for the eval
      pass so the fixed average slope is used.)
- [ ] TopK sparsity sweep: train the same tiny autoencoder bottleneck
      with K ∈ {1, 2, 4, 8, 16, full}, chart reconstruction loss vs sparsity.
- [ ] STE bit-width sweep: same network, vary `step ∈ {1.0, 0.5, 0.25,
      0.125, 0.0625}`, plot accuracy vs bit-width.
- [ ] Sequence-length scaling micro-benchmark — TNNetScaledDotProductAttention
      at seq_len ∈ {16, 32, 64, 128, 256} with d_k fixed. Confirms O(n²)
      scaling.
- [ ] Channel-attention bake-off: fixed tiny CIFAR backbone, four variants
      — (a) no attention, (b) SE, (c) CBAM, (d) hand-rolled "1x1 + sigmoid".
- [ ] FiLM-vs-concat conditioning bake-off on a class-conditional MNIST
      decoder.
- [ ] "Memorize a sentence" demo: train a 1-layer SDPA+RoPE model to
      perfectly memorize a 32-token sequence, print training loss curve
      and reconstructed sample.
- [X] "Learn to reverse" toy: SeqLen=8 input → output the reversed sequence.
- [ ] "Smallest net that can learn parity-N" study — sweep N ∈ {2, 4, 6, 8}.
- [ ] Grokking demo (`examples/Grokking/`) — reproduce delayed generalization
      (Power et al. 2022) on a pure-CPU toy. Train a tiny MLP on modular
      addition `(a + b) mod P` for a small prime (e.g. P=23 → 529 input
      pairs), one-hot inputs concatenated, softmax over P classes, with a
      fixed ~40/60 train/val split and weight decay on. The headline is the
      curve shape: train accuracy hits 100% early while val accuracy stays
      at chance for many epochs, then *suddenly* jumps to ~100% long after
      the training loss flatlines. Print a two-column ASCII chart of
      train-acc vs val-acc over a log-spaced epoch axis and flag the
      "grok epoch" (first epoch where val-acc crosses a threshold, well
      after train-acc saturates). Distinct from the parity-N capacity
      study (that asks "can it fit at all?", not "when does it
      generalize?") and from the "memorize a sequence" demo (which never
      generalizes by design). A small ablation toggling weight decay
      on/off makes the point that the regularizer is what eventually drives
      the late generalization. Pairs naturally with [[WeightSpectrumReport]]
      / [[WeightHistogramReport]] to watch the weights reorganize at the
      grok transition.
      ATTEMPTED 2026-05-24 (lucky-day batch) — NOT shipped, findings to
      save the next attempt: (1) the clean weight-decay-driven jump to ~100%
      val accuracy is NOT reproducible within the ~5-minute CPU budget here —
      the library's full-batch Adam is slow/unstable at the LRs grokking
      needs, so the thousands of epochs required do not fit the budget.
      (2) CONCATENATED one-hot input (depth 2P) never generalizes (val ~0): the
      two independent input weight-blocks act as a pure lookup table and weight
      decay cannot align them. (3) SUMMED one-hot input (depth P) DOES show
      delayed generalization (train ~100% by ~epoch 200, val lags near chance
      then climbs to a ~0.5 plateau) but weight decay made NO difference at
      this scale. Even the summed variant with two runs + P=23 ran >4m50s and
      blew the budget. SUGGESTED BREAKDOWN for a future attempt that fits the
      budget: (a) ship the reproducible *representation contrast* alone
      (concat-never-generalizes vs summed-delayed-generalizes) as a fast
      single-run demo, dropping the weight-decay claim; OR (b) first add a
      faster optimizer / mini-batch path so enough epochs fit, THEN retry the
      true grok. Do not re-attempt the full weight-decay grok as a single
      monolithic example until (b) lands.
- [ ] Lottery-ticket / magnitude-pruning follow-up to double descent: the
      over-parameterised models on the RIGHT arm of examples/DoubleDescent are
      the compressible regime — prune the H=128 interpolating net by weight
      magnitude and show it keeps the low test error down to a small fraction
      of its weights. Pairs with [[WeightSpectrumReport]] /
      [[WeightHistogramReport]] (watch the weight-norm spike at the
      interpolation threshold).
- [ ] "Surgery" experiment: train a small classifier, then zero out the
      top-K most-active hidden units and chart accuracy degradation vs K.
- [ ] SWA effect-size sweep: vary SWA start-epoch fraction ∈ {0.5, 0.6,
      0.7, 0.8, 0.9} and chart final test accuracy.
- [ ] Cosine-LR vs constant-LR on SimpleImageClassifier, three seeds each.
- [ ] BatchSizeSweep follow-up: the linear LR-scaling rule (Goyal et al. 2017,
      "Accurate, Large Minibatch SGD"; Krizhevsky 2014). The landed
      examples/BatchSizeSweep/ holds the learning rate FIXED at 0.01 while varying
      the batch size, so the large batch (128) needs noticeably more epochs — and
      the README's closing line even gestures at the fix ("a large batch often
      wants a larger learning rate") but never demonstrates it. Fork that example
      into examples/LRBatchScaling/ that, for each batch size B in {8,16,32,64,128},
      sweeps a small LR grid (e.g. base_lr * B/B0 spanning a few multipliers around
      the linear prediction) on the SAME fixed seed/data/net/epoch-budget and
      records epochs-to-converge at each (B, LR) cell. Print a B x LR grid of
      epochs-to-converge (or final val MSE) and, per batch size, flag the
      best-LR column; the headline payoff is that the best-LR locus tracks
      ~linearly with B (the diagonal of the grid), so doubling the batch and
      doubling the LR keeps epochs-to-converge roughly constant — the cleanest
      single demonstration that batch size and learning rate are coupled knobs,
      not independent ones. Built-in correctness gate: the per-batch best LR must
      be MONOTONE NON-DECREASING in B (a flat or decreasing best-LR curve means
      the grid is mis-centred or the task saturates too fast — shrink the net /
      tighten the convergence threshold). Stretch: overlay the sqrt-scaling
      prediction (best_lr ~ sqrt(B)) as a second reference curve and report which
      rule the toy actually follows at this scale (small full-batch-ish toys often
      sit between the two). Keep dims tiny and MaxThreadNum := 1 so the whole
      B x LR grid still fits the <5-min CPU budget. Distinct from the landed
      BatchSizeSweep (fixed LR, cost/quality trade only), from the open
      OptimizerBakeoff per-optimizer LR shoot-out (varies LR per OPTIMIZER at a
      fixed batch, not LR-vs-batch coupling), and from SchedulerCompare (LR
      SCHEDULE shape over training, not the batch-coupled base LR).
- [ ] Activation saturation visualizer: train a tiny net with Sigmoid /
      Tanh / HardSigmoid and print the fraction of saturated units per
      layer per epoch.
- [ ] "Where does the gradient go?" visualizer for in-tree saturation
      activations: for each of HardTanh / SoftCapping / Clamp / ReLU6 /
      SoftSign / Tanh, feed a 1D ramp and print y(x), dy/dx as PGM strips.
- [ ] Softmax-vs-SoftMaxOne-vs-Sparsemax bake-off as a pure-forward-pass
      comparison: same Q,K,V tensors, print entropy and max-weight of each.
- [ ] SDPA + TNNetSoftMaxOne micro-experiment: replace the softmax inside
      SDPA with SoftMaxOne and check whether attention-mass on a "all-keys-
      irrelevant" probe sequence drops toward zero. Needs either a flag
      on SDPA or a small standalone wiring.
- [ ] Softmax stability micro-test: feed a deliberately huge logit vector
      (`x = [1000, 0, 0, ...]`) through TNNetSoftMax / TNNetLogSoftMax;
      assert finite output, finite gradient, sum-to-one within 1e-6.
- [ ] PixelNorm Jacobian-blow-up empirical test at `||x|| ∈ {1, 1e-2,
      1e-4, 1e-6, 1e-8, 1e-10}`; record central-difference vs analytic-
      gradient relative error.
- [ ] L2Normalize vs PixelNorm head comparison on a tiny generator-shaped
      net.
- [ ] "Does exact-softmax-Jacobian matter?" controlled experiment: same
      classification example with new exact Jacobian vs old diagonal
      approximation restored on a branch.
- [ ] Polynomial-activation bake-off micro-experiment: swap each TNNetReLU
      in a small MLP for TNNetPolynomialActivation, compare final loss on
      the hypotenuse toy.
- [ ] Threshold-as-sparsifier sweep: theta ∈ {0, 0.1, 0.5, 1.0}, report
      (active-units %, recon loss) on a 64-unit autoencoder.
- [ ] Activation derivative-at-zero study: plot Compute + ChainDeriv
      around x=0 for ReLU, GELU, Swish, Mish, CELU, ELU on a [-2,2] grid.
- [ ] Width vs depth at fixed parameter budget on a tiny MNIST-shaped
      task: 4 widths × 4 depths, plot val-loss heatmap.
- [ ] Init-scheme sensitivity heatmap — ASCII heatmap of per-layer
      activation magnitudes across init schemes for a deep ReLU MLP.
- [ ] Flip-augmentation efficacy sweep: TNNetFlipX/Y at p ∈ {0, 0.25,
      0.5, 0.75, 1.0} on a synthetic orientation task.
- [ ] CELU vs ELU alpha-sensitivity micro-experiment: sweep `alpha in
      {0.1, 0.5, 1.0, 2.0}` for both.
- [ ] ESwish vs Swish bake-off: β ∈ {1.0, 1.25, 1.5, 2.0}.
- [ ] TNNetSnake α-sweep: fit `f(x) = sin(8x)` with a 3-layer Snake MLP
      across α ∈ {0.5, 1, 2, 4}.
- [ ] Trig identity composition tests: `Sin(x)² + Cos(x)² = 1`, Snake at
      α=1 derivative trig identity.
- [ ] Test that `TNNetNegate.Compose(TNNetNegate)` round-trips to identity.

### Norm / extra audits
- [ ] PixelNorm + StyleGAN-flavored generator micro-example (the layer's
      headline use case; pairs with VisualGAN).
- [ ] PixelNorm vs InstanceNorm vs no-norm bake-off on a tiny generator-
      shaped net.
- [ ] TNNetInstanceNorm CIFAR-style integration example (SimpleImage path
      with InstanceNorm replacing ChannelStdNorm).
- [ ] CIFAR/segmentation example using TNNetReverseChannels — channel-flip
      data augmentation as headline use case.
- [ ] CIFAR/ImageNet-style ShuffleNet block example (1x1 conv →
      ChannelShuffle → depthwise conv) integrated into one of the SimpleImage
      paths.
- [ ] ChannelShuffle group-count sweep: train the same tiny conv net with
      `groups ∈ {1, 2, 4, 8}` and chart accuracy.
- [ ] "Does ChannelShuffle help small models?" experiment on SimpleImage
      CIFAR at matched parameter count.
- [ ] "Why bother with a learnable per-channel scale?" experiment: train
      two deep MLPs (one with a small-init TNNetChannelMul on each
      residual branch, one without) on a toy regression task; chart
      per-layer gradient norms.
- [ ] DropPath-strict-drop teaching artifact: confirm a deep residual net
      with p=1 DropPath after every block collapses to the identity-path
      loss curve.
- [ ] LogSigmoid + BCE-with-logits training-loss smoke example.
- [ ] LogSigmoid kink-region test at the `x=0` branch crossover.
- [ ] Threshold "kink at theta" test at hand-picked x = theta documenting
      the chosen convention.
- [ ] TNNetMaxOut serialization-after-wire test: build a small net with
      MaxOut in the middle (so SetPrevLayer fires post-load), save/load
      whole net via SaveToString/LoadFromString, assert Compute matches
      end-to-end.
- [ ] TNNetMaxOut CIFAR-style example wired into one of the SimpleImage
      paths.
- [ ] TNNetCELU CIFAR-style smoke example.

### Documentation I'd enjoy writing
- [ ] "Activations cheat sheet" in `docs/activations.md`: one row per
      activation with formula, derivative, saturating?, smooth-at-zero?,
      typical use case.
- [ ] `docs/activation_taxonomy.md` — organise the ~50 activations now in
      the repo by mathematical family.
- [ ] "Saturation activations cheat sheet" in `docs/saturation.md` covering
      Clamp / HardTanh / SoftCapping / ReLU6 / SoftSign / Tanh / HardSigmoid.
- [ ] "Periodic activations" README subsection covering TNNetSin, TNNetCos,
      TNNetSnake, TNNetGaussianActivation.
- [ ] "Sparsity & routing" README subsection covering TNNetTopK,
      TNNetHardShrink, TNNetSoftShrink, TNNetThreshold (TopK has landed).
- [ ] "Elementwise transcendental layers" README subsection covering
      TNNetSqrt / TNNetExp / TNNetLog / TNNetReciprocal / TNNetAbs /
      TNNetSquare. Common eps-guard convention plus a tiny Euclidean-norm-
      reciprocal composition snippet.
- [ ] "Involution layers" README subsection covering TNNetReverseChannels /
      TNNetReverseXY / TNNetFlipX / TNNetFlipY. One paragraph plus a
      `Net.AddLayer([TNNetFlipX.Create(), TNNetFlipX.Create()])` identity
      snippet.
- [ ] "Bounded activations" README subsection covering Sigmoid, Tanh,
      HardSigmoid, HardTanh, SoftSign.
- [ ] "Building a transformer block" README walkthrough — pull
      TNNetMaskedFill / SDPA / RotaryEmbedding / GEGLU / SwiGLU /
      LayerNorm / RMSNorm references into one walkthrough with a single
      assembled code snippet. The MHSA half is the landed
      `TNNet.AddMultiHeadSelfAttention`.
- [ ] "Loss functions" README subsection grouping MSE, MAE, CE, Huber,
      SmoothL1, LogCosh, Charbonnier, Dice, KL, Focal, LabelSmoothing, and
      CosineEmbedding into a single short table.
- [ ] "Robust regression losses" README entry under
      TNNetHuberLoss / TNNetSmoothL1Loss / TNNetLogCoshLoss /
      TNNetCharbonnierLoss.
- [ ] "Learning-rate schedulers" README subsection — one paragraph per
      schedule with a snippet showing how to wire it into TNeuralImageFit.
      The scheduler classes (TNeuralLRScheduler family) and the TNeuralFit
      integration (the optional Scheduler property driving NextLR each epoch)
      have both landed; this is the remaining README write-up.
- [ ] "Layer index by family" README appendix — alphabetical-within-family
      table (Convolution / Pooling / Activation / Normalization / Attention
      / Loss / Shape / Regularization).
- [ ] "Position encodings in this repo" comparison page covering sinusoidal
      AddPositionalEmbedding, RoPE, and ALiBi: when each is the right pick,
      input expectations, code snippets.
- [ ] "Softmax variants in this repo" note: TNNetSoftMax, TNNetPointwiseSoftMax,
      TNNetSoftmaxTemperature, TNNetSoftMin, TNNetSoftMaxOne, TNNetLogSoftMax
      — when to pick each, which axis, exact vs approximate Jacobian.
- [ ] `docs/numerical_gradient.md` — short tutorial: why central differences,
      how the existing TestNumericalGradient helper works, how to add a
      test in five lines. Includes a "non-differentiable forward" pattern
      note (STE is the first in-tree layer where central-difference is
      provably wrong).
- [ ] "How to add a new layer" cookbook anchored to a real recent landing
      step by step from constructor declaration through CreateLayer dispatch
      to the numerical-gradient test.
- [ ] "How to add a new activation in ~30 lines" walkthrough — reuses
      TNNetReLUBase with a worked example.
- [ ] "Elementwise activation layer authoring" mini-guide capturing the
      recurring 4-step pattern (Compute, Backpropagate via FOutputErrorDeriv,
      dispatch entry, four-test shape).
- [ ] `docs/channel_attention.md` — compare the landed CBAM (TNNet.AddCBAM)
      on the same axes with SE (TNNet.AddSEBlock) and FiLM (TNNetFiLM).
- [ ] `docs/loss_layers.md` — once the loss family is complete, table input
      shape required, scalar vs per-sample, drop-in vs auxiliary.
- [ ] Short note in `tests/README` on "how to add a numerical-gradient
      test in three lines" cookbook for the shared helper.
- [ ] CHANGELOG.md / "What's new" section — one bullet per landed layer
      with date + commit short SHA.
- [ ] `docs/lucky_day_log.md` — rolling changelog of what each lucky-day
      batch shipped, letting tasklist.md itself be pruned of historical
      batches.
- [ ] Auto-generator for `docs/layer_taxonomy.md` walking the class
      hierarchy of TNNetLayer and emitting a tree.
- [ ] Inline-comment cleanup pass on TNNetScaledDotProductAttention: shape
      annotations on every loop, named strides, link to the planned
      annotated walkthrough.
- [ ] Annotated SDPA walkthrough: softmax-Jacobian derivation, shape
      annotations, worked tiny example (d_k=2, SeqLen=2).
- [ ] Annotated TNNetSoftmaxTemperature.Backpropagate walkthrough: full
      softmax-Jacobian derivation with shapes on every line.
- [ ] Doc-comment pass on the activation layers added in recent batches
      (GEGLU, SwiGLU, DropPath, RotaryEmbedding) so the auto-generated
      layer reference has clean source to pull from.
- [ ] README "What landed this month" entry: SDPA, RoPE, MaskedFill,
      SoftCapping, DropPath, GEGLU/SwiGLU/GLU, SquaredReLU,
      SpatialDropouts. One line each.
- [ ] One-pager "transformer building blocks landed in this repo": table
      of LayerNorm / MaskedFill / SDPA / RoPE / SoftCapping / DropPath /
      GEGLU / SwiGLU / GLU / SquaredReLU / AddPositionalEmbedding
      / ChannelShuffle / ALiBi / TanhGLU with "what it is" + "use it when".
- [ ] Short "where the test suite lives" map: tests/TestNeuralNumerical vs
      older `tests/*` programs, how RunAll.sh orchestrates them.
- [ ] "Gradient-check failure triage protocol" note: when a new numerical-
      gradient test fails, the bug is almost always in the layer under test,
      not the test harness. Document the DeMaxPool case as the canonical
      worked example.

### Stretch / ambitious
- [ ] `examples/TinyDiffusion/` — a 20-step denoising-diffusion model on
      8x8 grayscale MNIST patches using a tiny FiLM-conditioned U-Net with
      TNNetSinusoidalTimeEmbedding (FiLM and the timestep embedding are both
      in tree).
- [ ] Mixed-precision experiment first step: add `TNeuralFloat16 = packed
      record ...` in neuralvolume.pas with conversion helpers, plus a
      one-layer forward-only test validating FP16 matches FP32 to within
      1e-2.

### Introspection (added)
- [ ] LRPReport follow-up (landed 2026-06-05): the shipped epsilon-rule only
      back-distributes through the DENSE (TNNetFullConnect family) + activation
      stack — conv / pointwise-conv layers are honestly SKIPPED. Add a CONV
      relevance rule (epsilon-rule over the conv receptive field, distributing
      R_j across the input patch that produced output j) so the report works
      end-to-end on a real SimpleImage-style CNN, not just an MLP. Reuse the
      conv Compute's patch-iteration; assert conservation residual stays O(eps)
      on a tiny conv probe in the smoke test.
- [ ] LRPReport follow-up: add the gamma-rule and alpha-beta (LRP-αβ) variants
      alongside the epsilon-rule (selectable via a parameter), which separate
      positive/negative contributions and are the standard "explanation-quality"
      upgrade over plain epsilon. Contrast their input-relevance maps vs epsilon
      on the existing examples/LRP probe.
- [ ] TNNetCosineSimilarityAttention bake-off (now easier — learnable scale
      landed 2026-06-05): plain SDPA vs cosine-attn (fixed scale) vs cosine-attn
      (learnable scale) vs SDPA+TNNetSoftCapping on the PositionEncodingBakeoff
      tiny next-token harness — does the bounded `[-scale,+scale]` logit remove
      the NaN/overflow events SoftCapping targets, at matched final loss, and
      does the learnable scale beat the fixed one?
- [ ] FeatureSeparabilityReport follow-up: the scatter-
      decomposition identity `tr(Stot)=tr(Sw)+tr(Sb)` is only exact for
      class-balanced batches (the report uses class-balanced `mean_c`
      definitions and prints the worst residual). Add a count-weighted scatter
      mode so the identity holds exactly for imbalanced probe batches too,
      gated by a flag (balanced stays default). Re-pin the smoke-test
      assertion under the weighted mode.
- [ ] FeatureSeparabilityReport follow-up: a training-trajectory variant that
      calls the report every N epochs on a fixed probe set and charts
      `tr(Sw)` collapse + Fisher-ratio climb over training — the cleanest
      single-number window into the terminal phase / neural collapse. Pairs
      with the open grokking experiment and the landed lottery-ticket experiment
      ([[WeightSpectrumReport]]).
- [ ] NeuralCollapseReport follow-up: the landed NC1 uses the diagonal trace-ratio
      surrogate `tr(Sw)/tr(Sb)`. A true `tr(Sw·Sb^+)/C` NC1 (full matrix
      pseudo-inverse) would need an eigensolver/SVD helper — track if a second
      consumer wants it.
- [ ] ModeConnectivityReport follow-up: make the connected/weak-barrier/
      separated verdict robust when the endpoint losses are near zero. As
      landed, the verdict uses a barrier-relative-to-endpoint-loss ratio, so
      the same-basin run (barrier ~3e-5 on an endpoint loss ~3e-4) misreads
      as "weak barrier" purely because the denominator is tiny. Add an
      absolute-floor term (e.g. treat barriers below an absolute epsilon as
      "connected" regardless of ratio) and re-pin the example's RUN 1 verdict.
- [ ] LogitLensReport follow-up: the spec's optional `Project` flag (reuse
      [[LinearProbeReport]]'s deterministic random-projection to FORCE a
      width-incompatible layer through the head, flagged "heuristic") was not
      implemented — the landed report honestly SKIPs incompatible layers. Add the
      projected-lens path behind a flag so deeper/narrower stems get a (heuristic)
      depth profile too, keeping SKIP as the default honest behaviour.
- [ ] MagnitudePruningReport follow-up: the report sweeps a FIXED
      sparsity menu `{0,10,...,90,95,99}%`. Add a "find-the-knee" refinement that
      bisects between the last surviving and first failing sparsity to report the
      knee to ~1% resolution instead of the 10%-grid step. ~20 lines on top of the
      landed sweep; the curve already brackets the knee.
- [ ] MagnitudePruningReport follow-up: this is the no-retrain precursor to the
      landed "Lottery-ticket" experiment (examples/LotteryTicket) — wire the two together so the
      knee found here seeds the prune level, then RETRAIN from the original init
      and chart whether the pruned-then-retrained net recovers the baseline
      accuracy (the lottery-ticket claim). The report already snapshots/restores
      the unpruned weights, so the retrain path is the only new piece.
- [ ] ActivationPatchingReport follow-up: the report and example
      shipped, but a KEY finding emerged — on a strictly FEEDFORWARD stack,
      whole-layer patching + downstream recompute lands on the clean-class
      manifold at EVERY layer, so `r_L ≈ 1` is flat and nothing localises. The
      landed example works around this with a BRANCHED net (a raw-input skip
      fused by `Concat`) so the trace is graded by construction (main-branch
      patches recover ~0.04, recovery jumps to 1.0 at the fusion layer). The
      genuine follow-up is finer granularity that DOES localise on a plain
      feedforward net: per-NEURON / per-CHANNEL activation patching (restore a
      single channel's clean activation, not the whole layer) and/or the
      "denoising" direction (patch a CLEAN activation into a CORRUPT run vs the
      reverse). Add a `Granularity` flag (layer | channel) on top of the landed
      whole-layer path.
- [ ] IntrinsicDimensionReport follow-up: the PCA participation
      ratio under-counts a known `k`-dim RANDOM subspace (lands ~2.4 for k=3,
      since PR equals `k` only when the `k` eigenvalues are EQUAL); TwoNN gives
      the cleaner ~k. The smoke test bands `PCA_ID` loosely as a result. Worth a
      doc note (or an equal-variance subspace generator in the example so the
      PCA estimate also reads ~k), and consider reporting the spectral
      `effective rank exp(entropy(lambda))` alongside PR as a less variance-skewed
      linear-ID estimate.
- [ ] IntrinsicDimensionReport follow-up: a training-trajectory variant that
      calls the report every N epochs on a fixed probe set and charts the
      final-layer ID dropping over training — a single-number window into
      representation compression. Pairs with the open grokking experiment and the
      landed lottery-ticket experiment ([[WeightSpectrumReport]]) and mirrors the open
      FeatureSeparabilityReport training-trajectory follow-up.
- [ ] WeightSpectralTailReport follow-up: the spec's 3-way example (fresh /
      well-trained / over-fit nets ranked by held-out accuracy, validating
      label-free model selection) was simplified to a fresh-vs-trained contrast
      at landing; the accuracy-ranking demo is still open.
- [ ] RepresentationSimilarityReport follow-up: add an RBF-kernel CKA mode
      alongside the landed linear-CKA one (Gaussian Gram `K_ij =
      exp(-||x_i - x_j||^2 / (2*sigma^2))` with sigma a median-distance
      heuristic, then the same centered HSIC ratio). RBF-CKA catches
      non-linear representational similarity the linear version misses;
      gate behind a kernel selector so linear stays the default. Self-CKA
      diagonal must still read 1.0.
- [ ] TopLogitMarginReport follow-up: the shipped `examples/MarginReport/`
      net ends in a `TNNetSoftMax`, so its "logits" are post-softmax
      probabilities and the margin lands in `[0, 1]`. Add a second run (or a
      sibling example) with a raw-logit `TNNetFullConnectLinear` head so the
      unbounded-margin case is also pinned, and document the interpretation
      difference in the README.
- [ ] Shared report-smoke-test helper for the `TNNet.*Report` family. The
      per-report smoke tests (TopLogitMargin / NeuronCorrelation /
      LayerSensitivity / DeadNeuron / WeightSpectrum / ...) all repeat the
      same three assertions: report string non-empty, contains an expected
      header substring, and a nil-NN call returns gracefully. Capture as
      `AssertReportSmoke(reportFn, expectedHeader)` in
      tests/TestNeuralLayersExtra.pas so new report tasks are a one-liner.
### Test-time augmentation evaluator
- [ ] TTAReport follow-up: the shipped report runs on a single synthetic probe
      set; add the spec's second run on a model trained WITH `TNNetRandomFlipX`
      augmentation, to show TTA gains shrink when the invariance is already
      learned (TTA and train-time augmentation are substitutes, not complements).
      Extends the landed examples/TestTimeAugmentation/.

### Adversarial robustness
- [ ] AdversarialRobustnessReport follow-up: add the optional
      `eps,accuracy` CSV side-output (skipped in the initial landing as
      "optional") so the degradation curve can be plotted outside the
      terminal. ~10 lines mirroring the CSV side-output already in
      DecisionBoundaryReport.
- [ ] AdversarialRobustnessReport follow-up: add a multi-step PGD
      (projected gradient descent) attack mode alongside single-step FGSM —
      iterate `x <- clip(x + alpha*sign(grad), x0 +/- eps)` for K steps. PGD
      is the standard stronger baseline; the report's accuracy-vs-eps curve
      should drop faster under PGD than FGSM at matched eps, which is itself
      a built-in sanity check. Gate behind a `Steps` parameter (Steps=1 ==
      today's FGSM).

### Loss-landscape degeneracy (Singular Learning Theory)
- [ ] `TNNet.LocalLearningCoefficientReport` + `examples/LocalLearningCoefficient/`
      — estimate the *local learning coefficient* (LLC, an empirical
      RLCT) of a trained network from Singular Learning Theory (Watanabe;
      Lau, Murfet, Wei et al. 2023, "Quantifying Degeneracy in Singular
      Models via the LLC"). This measures the volume-scaling /
      effective-dimensionality of the minimum the optimizer settled into —
      a fundamentally DIFFERENT quantity from the already-landed
      `HessianCurvatureReport` / [[EdgeOfStability]] top-eigenvalue (which
      is purely 2nd-order and blind to flat, degenerate directions that the
      LLC is precisely designed to count). Estimator is the SGLD-based
      WBIC/free-energy form: from the trained weights `w*`, run a short
      tempered SGLD chain that samples the local posterior with a Gaussian
      anchor `gamma/2 * ||w - w*||^2` pinning it to the basin, then
      `LLC_hat = n*beta * ( mean_chain[ L(w) ] - L(w*) )` where `L` is the
      average training NLL, `n` the sample count, and `beta = 1/log(n)` the
      WBIC inverse-temperature. Pure CPU, tiny MLP. The honest headline (in
      the "what did NOT fit the budget" style the [[Grokking]] entry uses):
      report the LLC and contrast it with the raw parameter count to show
      `LLC_hat << dim(w)` — the network uses far fewer *effective* degrees
      of freedom than it has weights. A clean built-in sanity check: a
      deliberately over-parameterised net (duplicate-then-halve a hidden
      layer so two units are forced redundant) should have a LOWER LLC than
      a minimal net fitting the same function, because the duplicated
      directions are flat/degenerate. The SGLD chain reuses the existing
      backward pass (no new gradient machinery); the only new infrastructure
      is the anchored-Langevin weight update + chain averaging, which can
      live entirely inside the report method. NOTE the known pitfall to call
      out in the README: LLC estimates are sensitive to the SGLD step size
      `epsilon` and localisation `gamma` — ship a fixed, documented
      (epsilon, gamma, chain-length) and a one-line caveat that the absolute
      value is calibration-dependent but the *ordering* (minimal < redundant
      < random-init) is the robust, reproducible signal. Pairs with
      [[FisherImportance]] (Fisher = local 2nd-order curvature; LLC = the
      degeneracy-aware generalization of "effective parameter count").

