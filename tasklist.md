# Task List — Feature & Enhancement Ideas

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

## Interesting applications / examples
- [X] Mahalanobis out-of-distribution detector (`examples/MahalanobisOOD/`) —
      reproduce Lee et al. 2018 "A Simple Unified Framework for Detecting
      Out-of-Distribution Samples" on a TINY pure-CPU target. Train a small
      classifier on an IN-distribution split (e.g. synthetic Gaussian blobs for
      classes 0..K-1, or MNIST digits 0..4), then FREEZE it and read the
      penultimate-layer feature vector for each training sample. Fit one
      class-conditional Gaussian per class: per-class mean mu_c plus a single
      shared (tied) covariance Sigma pooled across classes; the OOD score for a
      new x is the max over c of the negative squared Mahalanobis distance
      -(f-mu_c)^T Sigma^-1 (f-mu_c). Show that held-in samples score higher than
      OOD samples (held-out classes 5..9, or blobs from a far-away region) and
      report a single AUROC separating the two score distributions. Note:
      neuralcalibration.pas already has temperature scaling/ECE but there is no
      AUROC helper and no OOD example anywhere — the AUROC computation (sort the
      merged scores, sum ranks / rank-statistic form of the Mann-Whitney U) is
      the small reusable piece this example introduces. Keep it forward-only
      after training (no backprop through the detector), invert the KxK feature
      covariance with the existing volume math (small K so a plain Cholesky or
      Gauss-Jordan is fine), and add a smoke test asserting AUROC > 0.8 on the
      easy synthetic split.
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
## Infrastructure / dev experience
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
- [ ] CumSumPositionEncoding follow-up: actually train a tiny position-
      dependent model with and without the CumSum feature concatenated and
      chart loss delta. Forward-only demo landed; the bake-off is still open.
- [ ] Quick-start example: tiny char-level sequence model (XOR-of-bits or
      counting task) that trains in well under a minute on CPU.
#### Documentation
- [ ] Write a one-page "layer authoring checklist" — constructor + LoadFromString
      round-trip, CreateLayer dispatch entry, Compute/Backpropagate, and the
      mandatory numerical-gradient test. Captures the recurring steps every
      new-layer task in this file actually follows.
- [ ] "Reading a numerical-gradient failure" mini-guide — when the harness
      reports a mismatch, what does the magnitude tell you (analytic-bug
      vs. tolerance-too-tight vs. discontinuity-near-the-eps-step)?
- [ ] "Picking a tolerance" mini-guide for numerical-gradient tests — when
      1e-2 is fine, when to tighten to 1e-3, when to shrink eps instead of
      loosening the tolerance.

### Added ideas
- [ ] TNNetMaskedFill currently hard-codes the upper-triangle (strictly causal)
      pattern. Consider a follow-up that allows masking the lower triangle or a
      configurable offset, if a non-causal masking use case shows up.

### Ideas from JP
- [ ] Better integrate TBytePredictionViaNNet and TEasyBytePredictionViaNNet with
      TNNet. Find a way to backpropagate. May open a new class of problem solving.
      This is very interesting.
- [ ] Use TBytePredictionViaNNet and TEasyBytePredictionViaNNet as inspiration for
      new float32 based layers. This is also curious and interesting.
- [ ] Integrate with float32 functions from https://github.com/joaopauloschuler/pas-core-math
      for faster and more precise math.
- [ ] More image generative examples and or experiments.

### TNNetMultiHeadSelfAttention — breakdown
SDPA + RoPE + MaskedFill + ALiBi are all in tree. Suggested commit-sized
breakdown:
- [ ] TNNetTransformerDecoderBlock helper — adds the causal MaskedFill in
      front of self-attention and an optional cross-attention sub-block.
      Built on top of the encoder helper above to avoid duplication.
      Compose `AddTransformerEncoderBlock` (causal self-attention via
      CausalMask=True) and `AddMultiHeadCrossAttention(d_model, Heads,
      QuerySource, KeyValueSource)` for the encoder-decoder cross-attention
      sub-block. The decoder block is then: causal self-attn residual ->
      cross-attn residual (Q from the decoder stream, K|V from the encoder
      output) -> pointwise SwiGLU FFN residual. Mind that cross-attention takes
      EXPLICIT source layers, so the decoder builder must thread the
      encoder-output layer ref through.

### Attention variants / siblings

- [ ] GQA follow-up: exact KVHeads=QueryHeads vs AddMultiHeadSelfAttention
      equivalence to <1e-5 by copying identical weights. Deferred because
      AddMultiHeadSelfAttention consumes a pre-projected 3*d_model slab (one
      external projection) whereas AddMultiHeadGroupedQueryAttention does its own
      three Q/K/V projections from a d_model input, so a weight-for-weight wiring
      is fiddly; the landed test asserts equal output shape + the exact K/V
      projection param saving instead.
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
- [ ] TNNetSlidingWindowMaskedFill follow-up (landed 2026-05-31): a tiny
      next-token bake-off — full causal (TNNetMaskedFill / TriangularCausalMask)
      vs sliding-window at W in {2, 4, full} on a task whose answer lives within
      the window, charting loss + per-query key count. Shows the long-context
      cost/quality trade the layer enables. Fork PositionEncodingBakeoff's
      tiny next-token harness.
- [ ] TNNetSlidingWindowMaskedFill follow-up: an attention-shape stress test
      mirroring the open "Attention numerical-gradient stress test" — run the
      forward-mask + grad check across SeqLen in {1,2,3,5,8} and W in
      {1,2,SeqLen,SeqLen+2}, pinning the W>=SeqLen==full-causal equivalence and
      the W=1 diagonal-only edge at every shape.
- [ ] TNNetCosineSimilarityAttention follow-up: bake-off vs plain SDPA and vs
      SDPA+TNNetSoftCapping on a tiny next-token task — does the bounded
      `[-scale,+scale]` logit actually remove the NaN/overflow events SoftCapping
      targets, at matched final loss?
- [ ] TNNetCosineSimilarityAttention follow-up: make `scale` a learnable scalar
      (sibling to ReZero's single-weight pattern) instead of a fixed FFloatSt[0]
      constant, and check whether training drives it toward the cargo-culted
      `1/τ` temperatures used in cosine-attention papers.
- [ ] SDPA all-masked-row policy decision and test: currently a row where
      every key is masked produces NaN (softmax of all -inf). Concrete
      proposal: detect the all-masked row in Compute, output a zero row,
      skip the softmax for that row (what JAX/Flax MHA does). Document
      the choice in code and add the pinning test.
- [ ] "Attention numerical-gradient stress test" running the SDPA grad check
      across SeqLen ∈ {1, 2, 3, 5, 8} and asserting the max error vs
      tolerance at each. Pins shape-edge behavior the existing single-shape
      test can't see.

### Bake-off / experiment follow-ups
- [ ] Position-encoding bake-off follow-up: the landed bake-off uses a
      predict-the-PREVIOUS-token task on which ALiBi lands just above the
      no-position baseline — a single head's `2^-8` slope is a weak recency
      bias that under the causal mask favours the query's own position and
      injects NO positional content into the values, so it cannot do
      fixed-offset (-1) retrieval. Add an ALiBi-FAVOURABLE second task
      (a long-context recency / "attend-to-the-nearest-recent-match" task)
      and/or a multi-head variant with per-head slopes `2^(-8h/H)`, so the
      arm where ALiBi's locality prior actually wins is also demonstrated.
      Pairs with the open "ALiBi slope-base sweep" entry.
- [ ] Numerical-precision study: re-run the activation bake-off using FP32
      vs a simulated-FP16 path (round-trip volumes through fewer mantissa
      bits) and report the convergence-quality gap. Useful baseline for
      any future mixed-precision work.
- [ ] SoftCapping logit-stability micro-experiment: train a tiny classifier
      with and without a `TNNetSoftCapping(c)` before the final softmax,
      and print the rate of NaN/overflow events under an aggressive LR.
- [ ] DropPath schedule study: linearly increasing drop probability with
      depth (Stochastic-Depth schedule) vs constant `p`.
- [ ] RoPE base-frequency sweep: same tiny next-token model, sweep
      `base ∈ {1e2, 1e3, 1e4, 1e5}`, chart loss and qualitative sample
      quality.
- [ ] ALiBi slope-base sweep: vary slope from `2^(-8h/H)` to `2^(-kh/H)` for
      `k ∈ {4, 6, 8, 12}` on a tiny next-token task and chart loss.
      Empirical check of the cargo-culted "8" constant.
- [ ] Causal-mask + SoftCapping interaction study: with logits clipped via
      `TNNetSoftCapping(c)`, sweep `c ∈ {5, 10, 20, 30, ∞}` on a tiny
      next-token task and chart loss + max-logit-norm.
- [ ] "Lottery-ticket"-flavored experiment: train a small dense net,
      magnitude-prune the bottom X% of weights, retrain from the original
      init, and compare. Pure CPU, finishes in seconds.
- [ ] Init-scheme × depth heatmap: for depths {2, 4, 8, 16} and inits
      {Glorot, He, LeCun, plain N(0, 0.01)}, plot first-step gradient norm
      at the deepest layer.
- [ ] "Which init wins per activation" matrix: cross-product of init schemes
      × activation functions on a fixed tiny MLP, report epochs-to-converge.
- [ ] First-batch gradient-norm heatmap across (depth, width, init):
      enumerate a small grid, print one number per cell.
- [ ] Train-time vs inference-time delta sweep for the noise layers
      (TNNetDropout, TNNetDropPath, TNNetSpatialDropout1D/2D): same tiny
      classifier, sweep `p ∈ {0.0, 0.1, 0.2, 0.4}`, chart train vs val loss.
- [ ] Numerical-gradient eps sweep: pick one well-tested layer, run the
      gradient check with `eps ∈ {1e-2, 1e-3, 1e-4, 1e-5, 1e-6}` and print
      max-error vs eps.
- [ ] Random-label memorization STRETCH follow-up: the landed
      examples/RandomLabelMemorization/ does the binary true-vs-fully-shuffled
      contrast; add the label-corruption-fraction sweep `p ∈ {0.0, 0.25, 0.5,
      1.0}` and chart epochs-to-fit-train (rises with p) against the test gap
      (widens with p) — the smooth interpolation between "real structure" and
      "pure memorization". Fork the landed demo's net/data/training loop and add
      a per-p corruption knob + an epochs-to-train>=0.99 counter. Keep dims tiny
      so 4 corruption levels still fit the <5-min budget.
### Composite blocks / builders I'd enjoy shipping
- [ ] TNNetAffineBlock — `Mul → Bias` builder for a learnable per-channel
      affine transform separable from FullConnect (built on TNNetChannelMul +
      TNNetChannelBias, both in tree).

#### Attention / sequence
- [ ] TNNetCausalConv1D — 1D conv with left-only padding so output at
      position t depends only on positions ≤ t. Backward is the standard
      conv backward minus the masked-future part. Pairs with TNNetTokenShift
      and unblocks attention-free baseline experiments. (Already possible with existing layers?)
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
- [ ] TNNetDiagonalSSM follow-up: add it as the fourth contender in the
      open "causal-conv vs token-shift vs SDPA on the same toy next-token
      task" experiment — its selling point is matching attention quality
      at linear cost.
- [ ] KV-cache / incremental-decode O(1)-per-step path for
      TNNetDiagonalSSM (a linear recurrence is O(1)-per-step by nature;
      the SDPA incremental-decode notes above apply doubly here).
- [ ] TNNetTokenHistoryPenalty follow-up: wire it into the downstream
      ../gpt-3-for-pascal generation loop (call `Apply` before the sampler
      and `RegisterToken` after each emit, `ResetHistory` per sequence) and
      show a qualitative before/after on a repetition-prone prompt — the
      class landed this lucky-day batch (neuralvolume.pas, 7 tests in
      tests/TestNeuralSamplers.pas) but no in-tree generator calls it yet.
- [ ] TNNetReversibleBlock — RevNet-style additive coupling
      (`y1 = x1 + F(x2)`, `y2 = x2 + G(y1)`). Forward + inverse round-trip
      to within fp tolerance is the headline test.
- [ ] TNNetWeightStandardization follow-up: a CONVOLUTION variant
      (standardize a conv layer's filters per output channel). The dense
      form landed; the conv form is the headline WS use case (Qiao et al.
      pair it with GroupNorm in a conv stack). Mirror the dense Jacobian
      per output-channel filter. Pairs with a tiny WS+GroupNorm vs
      BatchNorm CIFAR-stub bake-off.
- [ ] TNNetSpectralNorm wrapper — wraps an existing FullConnect or
      Convolution layer and divides its weight matrix by its largest
      singular value (one power-iteration step per forward pass).
      NOTE: the reusable power-iteration helper `TNNet.EstimateSpectralNorm`
      now exists (landed with WeightSpectrumReport) — build the wrapper on
      top of it rather than re-deriving the iteration.
- [ ] TNNetStochasticPool follow-up: a bake-off vs TNNetMaxPool / TNNetAvgPool /
      TNNetSoftPool on a tiny image-classifier stub — does the stochastic
      regularisation lower the train/val gap at matched architecture? Fork an
      existing pooling example (examples/PoolingBakeoff/) and add the new arm.
- [ ] TNNetShakeShake / TNNetShakeDrop — Shake-Shake regularization and
      its single-branch ShakeDrop generalization.

#### Channel attention / conditioning
- [ ] TNNetCBAM — SE block plus a spatial-attention sibling.
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
- [ ] TNNetBitLinear follow-up: activation-quantization variant — BitNet b1.58
      also quantizes the *activations* to int8 (absmax per-token). Consider a flag
      or sibling that rounds the layer INPUT through an absmax STE before the
      ternary matmul, so the "fully-quantized linear" path is reachable. Scope as
      its own flag on TNNetBitLinear (forward adds an input absmax-round; backward
      STE-passes the input gradient unchanged).
- [ ] TNNetAPL follow-up: APL-vs-PReLU-vs-ReLU bake-off on the hypotenuse toy
      (or a tiny CIFAR stub) at matched param count, sweeping the hinge count
      S ∈ {1, 2, 4} — does the extra piecewise capacity buy lower final loss?
      This is a ~30-line activation swap.
#### Probability projections / sparsity
- [ ] TNNetGumbelSoftmax follow-up: temperature-annealing
      micro-experiment — train a tiny discrete-latent autoencoder whose
      bottleneck is a `TNNetGumbelSoftmax`, anneal `tau` from ~2.0 down to
      ~0.1 over training, and chart reconstruction loss vs `tau` plus the
      bottleneck's output entropy (the categorical sharpens as tau drops).
      The layer + its soft/hard modes are in tree; this is the headline
      use case. Pairs with the open TNNetMixtureOfExperts routing gate.
- [ ] TNNetMixtureOfExperts — top-k softmax gate over N expert sub-networks
      plus a load-balancing auxiliary loss. (The just-landed
      TNNetGumbelSoftmax is the natural differentiable hard-routing gate.)
#### Normalization primitives
- [ ] TNNetMinMaxNorm follow-up: a per-channel variant (min/max reduced over
      spatial only, independently per depth channel) gated by a flag, mirroring
      the per-(x,y)-over-depth vs full-volume split discussed for L2Normalize.
      Builds on the landed full-volume TNNetMinMaxNorm.
- [ ] TNNetUnitNormConstraint hard-projection variant: a true *post-step hard
      projection* (renormalize the previous layer's weights after each update,
      non-differentiable) — still open if a hard constraint is ever wanted. The
      differentiable reparametrization (TNNetWeightNormLinear, landed) already
      covers the headline use case.

#### Reduction / shape
- [ ] TNNetGather follow-up: a MULTI-index variant that selects an ordered
      SUBSET of depth channels (output depth = number of selected indices),
      so it doubles as a learnable-free channel reorder/prune. The landed
      single-channel form is the degenerate one-index case; backward scatters
      each output channel's error back to its source channel.
- [ ] TNNetUpsampleNearest backward consistency: assert summing the
      per-block output errors equals the input error.
### Loss layers
- [ ] TNNetCosineEmbeddingLoss follow-up: a tiny
      siamese-pair embedding micro-example — train two shared-weight MLP
      branches whose outputs are concatenated into the `a|b|y` layout, on a
      synthetic "same vs different class" pair task, and print the learned
      same-pair vs different-pair cosine histograms. Headline use case for
      the landed head; pairs with [[TripletEmbedding]].
- [ ] TNNetKLDivergence distillation follow-up
      (examples/KnowledgeDistillation/): temperature sweep T in {1,2,4,8} on this
      example — chart how soft-target sharpness changes the distilled student's
      accuracy/agreement.
- [ ] Tversky α/β asymmetry sweep on the segmentation micro-example: with a
      deliberately class-imbalanced mask, sweep `(α,β) ∈ {(0.5,0.5),(0.3,0.7),
      (0.7,0.3)}` and show how β>α trades precision for recall (fewer false
      negatives). Pure α/β knob study on the landed TNNetTverskyLoss.
      Fork examples/DiceSegmentation/ and swap the head for TNNetTverskyLoss
      with the three (α,β) pairs.
- [ ] LabelSmoothing calibration check: train SimpleImageClassifier with
      `TNNetLabelSmoothingLoss(eps)` at `eps ∈ {0, 0.05, 0.1, 0.2}` and feed
      each into the `neuralcalibration` ECE/Brier report — the textbook claim
      is smoothing improves calibration at a small accuracy cost.
- [ ] TNNetCenterLoss follow-up: a true SOFTMAX-JOINT variant (or an example)
      that wires the landed penalty head alongside a classification head and
      shows the headline Wen et al. result — center loss tightens intra-class
      feature clusters (visualise a 2-D embedding before/after). Pairs with the
      [[FeatureSeparability]] example.
- [ ] TNNetCenterLoss follow-up: cross-batch EMA-updated centers — needs a
      batch-aware loss hook (the per-sample FOutputError path is blind to other
      minibatch samples, the same limitation logged for a true cross-batch
      InfoNCE). Track alongside that batch-aware-loss-hook item.
- [ ] TNNetArcFace — additive angular-margin softmax for face/embedding
      recognition heads.
- [ ] TNNetVectorQuantizer (VQ-VAE bottleneck) — codebook of K vectors with
      straight-through assignment plus commitment/codebook losses.

### Training infrastructure (the "missing plumbing")
- [ ] TNeuralLRScheduler interface (`function NextLR(Epoch, Step): TNeuralFloat;`)
      with concrete implementations: TStepLR, TCosineAnnealingLR
      (η_min + (η_max-η_min)·0.5·(1+cos(π·t/T))), TWarmupCosineLR (linear
      warmup then cosine), and PolyLR (`η · (1 - t/T)^p`).
- [ ] StochasticWeightAveraging helper — TNNet wrapper maintaining a running
      average of live weights every N steps after epoch W.
- [ ] TNNetEMAWrapper / SetEmaShadow — exponential moving average of network
      weights for inference, sibling to SWA.
- [ ] Lookahead optimizer wrapper — every k inner SGD steps, set slow weights
      `φ ← φ + α·(θ - φ)` and rewind fast weights to φ.
- [ ] GradientClipping options on TNeuralFit — both `clip_norm` (global)
      and `clip_value` (element-wise).
- [ ] Layerwise learning-rate multipliers — per-layer `LRMult` field that
      the optimizer respects. Unlocks discriminative fine-tuning.
- [ ] NaN/Inf guard hook for TNeuralFit — optional "abort training and
      print the offending layer" check after each forward+backward pass.
      Plus a regression test that deliberately seeds a NaN and confirms
      the assertion fires at the right layer.
- [ ] Mixup data augmentation helper.
- [ ] SAM follow-up: the noisy-label 2D-blob clusters are easily separable so
      clean val-accuracy saturates (~99%) across all rho — the flatness signal
      carries the story but the val-acc-vs-rho curve is flat. A harder task
      (overlapping clusters / higher label-noise / a tiny MLP on a small image
      stub) where SAM's flat minimum actually buys measurable val-accuracy over
      plain SGD would complete the demonstration. Builds directly on the landed
      examples/SharpnessAwareMinimization/.
- [ ] Muon optimizer experiment (`examples/MuonOptimizer/`) — Newton-Schulz
      orthogonalized-momentum update (Jordan et al. 2024) for the 2D weight
      matrices of `TNNetFullConnectLinear` layers, framed as a hand-rolled
      gradient-surgery demo in the SAM / Lookahead style (NOT a core optimizer
      rewrite). Per step under `NN.SetBatchUpdate(True)` (so the accumulated
      `Neurons[].Delta` gradient tensor is actually populated — see
      [[manual-gradient-and-snapshot-gotchas]]): maintain a momentum buffer
      `M <- mu*M + G` per dense layer, then replace `M` by its nearest
      orthogonal matrix via ~5 fixed Newton-Schulz iterations on the normalized
      `X = M/||M||_F` (the quintic `X <- a*X + b*(XX^T X) + c*(XX^T)^2 X` with the
      paper's (a,b,c) ~ (3.4445,-4.7750,2.0315), all expressible with the
      existing `TNNetVolume` matrix ops), and apply `W <- W - lr*O` with the
      `sqrt(max(rows,cols))` scale so the update RMS matches Adam's. Bake it off
      against plain SGD-momentum and Adam on a small MLP (the hypotenuse toy or
      a tiny image stub), charting loss-vs-step and wall-clock/step. Headline
      correctness signal: after the Newton-Schulz pass the singular values of
      `O` are all ~1 (assert `||O^T O - I||_F` is small on a probe matrix) — can
      spot-check the spectrum with the existing `TNNet.EstimateSpectralNorm`
      power-iteration helper. Genuinely new (no orthogonalized-update path
      exists in tree); distinct from the differentiable `TNNetWeightNormLinear` /
      `TNNetWeightStandardization` reparametrizations, which normalize the
      FORWARD weights, not the update.
### Introspection / debugging tools
- [ ] WriteLayerTimings(NN, Sample) — runs one forward pass and prints
      per-layer wall-clock to stdout.
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
- [ ] EffectiveReceptiveFieldReport follow-up: add the optional `(radius, mass-
      fraction)` CSV side-output so the cumulative-mass curve can be plotted
      outside the terminal (~10 lines, mirrors the CSV side-output in
      DecisionBoundaryReport / the AdversarialRobustnessReport CSV follow-up).
- [ ] EffectiveReceptiveFieldReport follow-up: sweep dilation / kernel size on
      the stem and chart effective-RF growth vs theoretical-RF growth — the
      headline Luo et al. 2016 "effective RF grows sub-linearly" curve.
- [ ] NeuralTangentKernelReport follow-up: the fresh-init-vs-trained NTK-DRIFT
      contrast deliberately left out of the first landing (commit 857f679). Add an
      optional second-net / snapshot argument (mirror `ModeConnectivityReport`'s
      `SnapshotB` or `RepresentationSimilarityReport`'s `OtherNet`) so the report
      quantifies how far the empirical NTK moved between two checkpoints — e.g. the
      relative Frobenius drift `||K_trained - K_init||_F / ||K_init||_F` and the
      change in kernel-target alignment. Headline payoff: ≈0 drift = the
      infinite-width "lazy / kernel" regime, large drift = "rich" feature learning
      (the lazy-vs-rich question made visible). Then extend the existing
      `examples/NeuralTangentKernelReport/` to contrast a WIDE vs NARROW hidden
      layer and show the wide net's NTK drifts less. Reuse the snapshot machinery
      already proven in ModeConnectivity/PermutationAlign.
- [ ] `TNNet.TunedLensReport` — the *learned* sibling of the already-landed
      zero-parameter `LogitLensReport` (Belrose et al. 2023, "Eliciting Latent
      Predictions with the Tuned Lens"). The logit lens splices a raw hidden
      activation straight into the model's OWN frozen head; the tuned lens first
      runs each layer's activation through a small per-layer learned AFFINE
      "translator" (`TNNetFullConnectLinear(headInputDim)`, one per lens-
      compatible layer) that is TRAINED to map that layer's residual state into
      the final-layer basis BEFORE the frozen head decodes it — correcting the
      representation drift / basis-mismatch that makes the raw logit lens biased
      and over/under-confident at early depths. Scope: (a) freeze the trunk +
      head, attach one translator per lens-compatible layer, train only the
      translators by minimising each layer's KL to the model's final output
      distribution on an UNLABELLED probe batch (the distillation-to-self target,
      reusing the frozen-body + downstream-recompute splice idiom from
      `LogitLensReport` / `ActivationPatchingReport`); (b) emit the per-layer
      tuned-lens distribution, its entropy, and its KL-to-final, side by side
      with the raw logit-lens columns so the headline Belrose result is visible —
      the tuned curve commits EARLIER and tracks the final answer more faithfully
      (lower KL-to-final, monotone-ish) than the raw lens. Built-in correctness
      signals: at the LAST layer the translator collapses to identity and tuned
      == logit == final (max |Δp| ≈ 0); an UNTRAINED translator must do no better
      than the raw logit lens (KL-to-final not lower) — only after fitting does it
      win. Ship `examples/TunedLens/` forking the existing LogitLens net/task
      (constant-width `6 -> FC10+ReLU x4 -> FC4 -> SoftMax`) so the two lenses are
      directly comparable on the SAME probe batch, plus a `TestTunedLensSmoke`
      following the introspection-report test recipe. Distinct from LogitLens
      (zero params, no fitting), LinearProbeReport (probes for an EXTERNAL label,
      not the model's own next-layer basis), and ActivationSteering (edits
      activations, doesn't decode them). See [[introspection-report-pattern]].

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
- [ ] TNNetClamp kink-region test at `x = MinValue` and `x = MaxValue`.
- [ ] TNNetHardShrink / TNNetSoftShrink kink-region tests at hand-picked
      inputs (no central differences).
- [ ] TNNetSoftSign saturation test on ±1e6: assert `|y| < 1` and
      Backpropagate doesn't NaN.
- [ ] TNNetESwish saturation test at ±extreme inputs.
- [ ] LiSHT / BentIdentity gradient-magnitude sanity at large |x| — both
      grow unboundedly, finite-difference eps must scale with input
      magnitude.
- [ ] TNNetAbs near-zero gradient handling test — explicitly skip x = 0
      sampling and pin the convention (currently `sign(0) = 0`).
- [ ] TNNetSquare gradient-magnitude sanity test at large |x|.
- [ ] Shape-edge test for TNNetTokenShift: assert SetPrevLayer raises the
      documented error when SizeY > 1.
- [ ] Two-layer TokenShift composition test (catches subtle double-pass
      bugs in the t-1 / t+1 input-gradient scatter).
- [ ] TNNetStraightThroughEstimator `step ≤ 0` guard test.
- [ ] TNNetSoftMin saturation test on extreme inputs.
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
- [ ] `examples/TinyGPT/` — char-level transformer end-to-end demo on
      a short text snippet (Tiny Shakespeare or repeated arithmetic).
      Highest-value example missing from the repo; natural capstone for
      the transformer-building-blocks line of work.
- [X] `examples/InductionHeads/` — pure-CPU reproduction of the headline result of
      Olsson et al. 2022 "In-context Learning and Induction Heads"
      (<https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html>):
      a TINY 2-layer CAUSAL attention-only transformer spontaneously forms an
      *induction head* that does in-context copying — on a sequence that contains
      a repeat `... [A][B] ... [A] -> ?`, it predicts `[B]` by finding the earlier
      occurrence of the current token `[A]`, looking at the token that FOLLOWED it,
      and copying that token forward. This is a genuinely different phenomenon from
      the landed `examples/AttentionCopyTask/` (a SINGLE NON-CAUSAL head learning a
      position-based identity copy of a non-repeated sequence): induction requires
      (a) a causal mask, (b) repeated random sequences so the only way to win is
      content-based prefix-matching, not position, and (c) the two-head composition
      the paper identifies — a layer-1 "previous-token head" that writes token t-1's
      identity into position t, feeding a layer-2 "prefix-matching head" that
      attends from the current token back to where that same token appeared before.
      The toy: vocab ~16, draw a random prefix and CONCATENATE it with itself (one
      or more times) so every second-half position has a deterministic
      copy-the-previous-occurrence answer; train next-token CE under a causal mask.
      Build it from in-tree pieces only — `TNNetEmbedding` +
      `TNNetSinusoidalPositionalEmbedding` -> two stacked causal blocks via the
      `AddTransformerEncoderBlock(..., CausalMask=True)` builder (or hand-wired
      `TNNetMaskedFill` -> packed Q|K|V `TNNetPointwiseConvLinear` ->
      `TNNetScaledDotProductAttention`) -> `TNNetPointwiseConvLinear(Vocab)` ->
      `TNNetPointwiseSoftMax(1)`. Headline built-in correctness signals
      (printed PASS/FAIL, `Halt(1)` on failure): (1) in-context accuracy on the
      REPEATED half climbs to near-100% while accuracy on the first (unseen) half
      stays at chance — the model is copying, not memorising; (2) the
      "in-context learning score" (loss at a late repeated position minus loss at
      the matching early position) is strongly negative — the textbook ICL metric;
      (3) read the layer-2 head's attention matrix back via the existing
      `AttentionWeights` accessor and assert a "prefix-matching score": each query
      at a repeat puts most of its attention mass on the position ONE AFTER the
      earlier occurrence of its own token (the induction stripe), and the layer-1
      head's mass concentrates on the immediately-previous position
      (previous-token head). Render that attention matrix as a glyph-shaded ASCII
      heatmap so the induction stripe is visible directly. Distinct from
      AttentionCopyTask, the PositionEncodingBakeoff recency study, and the
      single-net diagnostic reports — this trains a real (tiny) two-layer causal
      transformer and surfaces an EMERGENT algorithmic circuit. Mind the
      manual-gradient / best-model-reload gotchas in
      [[manual-gradient-and-snapshot-gotchas]] and the per-token projection rule in
      [[mha-builder-and-seq-projection]]; keep dims tiny so the whole thing trains
      well under the 5-minute CPU budget.
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
- [ ] `examples/TinyTransformerFFN/` — SwiGLU + RMSNorm + residual FFN
      block on a toy denoising or autoregressive-bit task. No MHSA
      needed; demonstrates the FFN half-block.
- [ ] `examples/BiasOnlyTuning/` — freeze a pretrained classifier and
      fine-tune only inserted TNNetChannelBias layers on a new task
      (BitFit-style cheap adaptation).
- [ ] `examples/AffineFineTune/` — once TNNetAffineBlock lands, same
      pattern but freezing everything except the inserted Affine blocks
      (built on TNNetChannelBias + TNNetChannelMul).
- [ ] `examples/TokenShiftBaseline/` — train a tiny next-token char model
      with `TNNetEmbedding → TNNetTokenShift → MLP` and compare against
      the MHA-based version (`TNNet.AddMultiHeadSelfAttention`).
- [ ] `examples/ReZeroDeepMLP/` — train a 16-layer residual MLP with and
      without TNNetReZero on each residual branch on the hypotenuse toy.
- [ ] `examples/SpaceToDepthStem/` — show the SpaceToDepth → Conv stem
      replacing a stride-2 conv on a tiny CIFAR stub.
- [ ] `examples/MaxoutMnist/` — minimum-viable Maxout demo on a tiny-MNIST
      subset (or synthetic 2D classification).
- [ ] `examples/SchedulerCompare/` — same network trained four times with
      constant LR, StepLR, CosineLR, WarmupCosineLR; one chart.
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
- [ ] Causal-conv vs token-shift vs SDPA on the same toy next-token task.
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
- [ ] VQ codebook collapse stress test: K in {16, 64, 256} and a few
      commitment-loss weights, report per-run active codebook entries.
- [ ] "Memorize a sentence" demo: train a 1-layer SDPA+RoPE model to
      perfectly memorize a 32-token sequence, print training loss curve
      and reconstructed sample.
- [ ] "Learn to reverse" toy: SeqLen=8 input → output the reversed sequence.
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
      schedule with a snippet showing how to wire it into TNeuralImageFit
      (once the scheduler interface lands).
- [ ] "Introspection" README subsection — group CountLayers/Neurons/Weights
      with the new PrintSummary / FLOPs / WeightHistogram / DeadNeuronReport
      utilities.
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
- [ ] `docs/channel_attention.md` — once CBAM lands, compare it on the same
      axes with the already-landed SE (TNNet.AddSEBlock) and FiLM (TNNetFiLM).
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
      SoftCapping, DropPath, GEGLU/SwiGLU/GLU, SquaredReLU, LayerScale,
      SpatialDropouts. One line each.
- [ ] One-pager "transformer building blocks landed in this repo": table
      of LayerNorm / MaskedFill / SDPA / RoPE / SoftCapping / DropPath /
      GEGLU / SwiGLU / GLU / SquaredReLU / LayerScale / AddPositionalEmbedding
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
- [ ] `examples/HopfieldRetrieval/` — modern Hopfield network as attention
      (Ramsauer et al.): store K patterns, retrieve via a single softmax-
      attention step against a query.
- [ ] Mixed-precision experiment first step: add `TNeuralFloat16 = packed
      record ...` in neuralvolume.pas with conversion helpers, plus a
      one-layer forward-only test validating FP16 matches FP32 to within
      1e-2.

### Introspection (added)
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
      with the open grokking / lottery-ticket experiments
      ([[WeightSpectrumReport]]).
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
      open "Lottery-ticket"-flavored experiment — wire the two together so the
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
      representation compression. Pairs with the open grokking / lottery-ticket
      experiments ([[WeightSpectrumReport]]) and mirrors the open
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

### Parameter importance (continual learning / pruning)
- [ ] EWC two-task experiment building on the landed
      TNNet.FisherImportanceReport: train on task A, snapshot the diagonal
      Fisher, then train on task B with an L2 penalty pulling high-Fisher
      params back toward their task-A values; chart task-A retention with and
      without the penalty.
