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
<!-- (Sparse / mixture-of-experts routing layer removed: duplicate of the
     concrete TNNetMixtureOfExperts entry under "Probability projections /
     sparsity".) -->

## Interesting applications / examples
- [ ] Reinforcement learning: minimal DQN solving CartPole or a grid world
- [ ] Style transfer or diffusion-lite denoiser (building on SuperResolution / VisualGAN)

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
- [ ] Expand layer test coverage — numerical-gradient checks for layers that lack them

## Documentation / learning
- [ ] Interactive "build your first transformer in Pascal" tutorial
- [ ] Auto-generated layer API reference from doc comments
- [ ] Improve source-code comments for TNNetChannelShuffle and
      TNNetInterleaveChannels in neural/neuralnetwork.pas. Both are
      parameter-free channel-permutation layers and are easily confused.
      The doc comments should make the distinction explicit: spell out
      the permutation formula each one computes, the meaning of the
      constructor parameter (Groups vs StepSize), any constraints
      (ChannelShuffle requires Depth mod Groups = 0; InterleaveChannels
      has no divisibility constraint), and a one-line note on the
      typical use case (ShuffleNet-style group mixing vs generic
      stride-based interleaving after grouped/parallel convs).

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
<!-- (Volume unit micro-benchmark removed: duplicate of the `bin/layer_bench`
     CLI under "Tooling / dev experience", whose own entry notes it "subsumes
     the long-pinned Volume micro-benchmark and extends it to layers".) -->
#### Experiments I'm curious about
- [ ] Weight-initialization sensitivity demo: show how a deep-ish net's
      first-epoch gradient magnitudes change across the available init schemes.
#### Documentation
<!-- ("How numerical gradient testing works in this repo" contributor note
     removed: duplicate of the docs/numerical_gradient.md tutorial entry under
     "Documentation I'd enjoy writing".) -->
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

#### Experiments I'm curious about
- [ ] Batch-size sweep demo: same net and data, vary the batch size, and print
      how wall-clock-per-epoch and epochs-to-converge trade off. A concrete,
      visible illustration of a tuning knob beginners always ask about.

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
- [ ] (MHA-a) Add a small `TNNetSplitChannels`-based helper or example that
      carves a (3*d_model) Q|K|V depth slab into H per-head (3*d_k) slices,
      with a sanity test on H=2, d_model=8.
- [ ] (MHA-b) Add a wiring helper or example that runs one
      `TNNetScaledDotProductAttention(d_k)` per head slice and concats
      the H outputs via `TNNetDeepConcat` back to depth d_model. Test on
      a tiny (H=2, d_k=4, SeqLen=3) shape, numerical-gradient through.
- [ ] (MHA-c) Wrap (a)+(b)+a `TNNetFullConnectLinear(d_model)` out-projection
      into a `TNNetMultiHeadSelfAttention` helper class or builder function.
      Add a numerical-gradient test on the same tiny shape.
- [ ] TNNetTransformerEncoderBlock helper — LayerNorm → MHA → residual →
      LayerNorm → SwiGLU FFN → residual. Single call, configurable
      d_model / heads / d_ff. Companion numerical-gradient test on a tiny
      shape (d_model=8, heads=2, seq=3). Pre-norm and post-norm variants
      behind a flag.
- [ ] TNNetTransformerDecoderBlock helper — adds the causal MaskedFill in
      front of self-attention and an optional cross-attention sub-block.
      Built on top of the encoder helper above to avoid duplication.
- [ ] TNNetCrossAttention — same SDPA core but with separate Q vs K|V
      input branches (so encoder-decoder is reachable). Forward +
      gradient-check on a tiny shape.

### Attention variants / siblings
- [X] TNNetSinkAttention — prepend K learnable "attention sink" key/value
      slots that every query can attend to regardless of the causal mask.
      Helps long-context stability, ~30 lines on top of SDPA.
- [ ] TNNetSinkAttention follow-up (now landed): attention-sink stability
      micro-experiment. On a tiny causal next-token task with an "all-keys-
      irrelevant" probe row, compare plain SDPA vs TNNetSinkAttention and
      print the attention mass that lands on the sink slot(s) vs real keys —
      the StreamingLLM claim is the sink absorbs the otherwise-misplaced mass.
      Sweep K ∈ {1, 2, 4}. ~30-line wiring swap; all pieces in tree.
- [ ] TNNetSinkAttention follow-up: fold sink slots into the MHA breakdown
      ([[TNNetMultiHeadSelfAttention]] / TNNetTransformerDecoderBlock) so a
      decoder block can opt into sinks per head behind a flag.
- [ ] TNNetTalkingHeadsProjection — pre/post-softmax linear mix across
      heads (Shazeer et al.). A tiny learnable HxH multiply applied to
      attention logits along the head axis.
- [ ] TNNetCosineSimilarityAttention follow-up: bake-off vs plain SDPA and vs
      SDPA+TNNetSoftCapping on a tiny next-token task — does the bounded
      `[-scale,+scale]` logit actually remove the NaN/overflow events SoftCapping
      targets, at matched final loss? All three pieces are now in tree.
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
<!-- (Inline shape-annotation comments on TNNetScaledDotProductAttention
     Compute/Backpropagate removed: duplicate of the "Inline-comment cleanup
     pass on TNNetScaledDotProductAttention" entry under "Documentation I'd
     enjoy writing".) -->

### Bake-off / experiment follow-ups
- [ ] Position-encoding bake-off: same tiny seq model trained with
      (a) no position info, (b) sinusoidal AddPositionalEmbedding,
      (c) RoPE, (d) ALiBi, printing final loss and a sample generation
      per scheme. All four are in tree.
- [ ] Causal-mask sanity experiment: train a tiny attention model on
      next-token prediction WITH and WITHOUT TNNetMaskedFill, and show
      the unmasked one cheats (near-zero loss but useless at generation).
- [ ] Numerical-precision study: re-run the activation bake-off using FP32
      vs a simulated-FP16 path (round-trip volumes through fewer mantissa
      bits) and report the convergence-quality gap. Useful baseline for
      any future mixed-precision work.
- [ ] SoftCapping logit-stability micro-experiment: train a tiny classifier
      with and without a `TNNetSoftCapping(c)` before the final softmax,
      and print the rate of NaN/overflow events under an aggressive LR.
- [ ] DropPath ablation: train a small ResNet-style net on a tiny synthetic
      task with `TNNetDropPath(p)` after each residual block, sweeping
      `p ∈ {0.0, 0.1, 0.2}` and printing final loss.
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

### Composite blocks / builders I'd enjoy shipping
- [ ] TNNetPreNormResidual helper — `y = x + Sublayer(LayerNorm(x))`
      single-line builder. Take the sublayer as a TNNet builder closure.
- [ ] AddRMSNormResidual(NN, Sublayer) — companion builder using RMSNorm
      in place of LayerNorm (LLaMA-style blocks).
- [ ] AddPostNormResidual(NN, Sublayer) — post-norm pattern (`Sublayer →
      residual add → LayerNorm`), companion to PreNorm/RMSNormResidual.
- [ ] TNNetAffineBlock — once TNNetMul lands, `Mul → Bias` builder for a
      learnable per-channel affine transform separable from FullConnect.

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
#### Norm / regularization
- [ ] TNNetGatedResidual — per-channel zero-initialised learnable gate
      `y = x + alpha[c] * Sublayer(x)` (ReZero-with-channel-dim variant).
- [ ] TNNetRMSNormGated — RMSNorm followed by a learnable per-channel
      sigmoid gate.
- [ ] TNNetSwitchableNorm — learnable softmax-weighted combination of
      LayerNorm and RMSNorm outputs.
- [ ] TNNetReversibleBlock — RevNet-style additive coupling
      (`y1 = x1 + F(x2)`, `y2 = x2 + G(y1)`). Forward + inverse round-trip
      to within fp tolerance is the headline test.
- [ ] TNNetWeightStandardization — normalize convolution weights per
      output channel (zero-mean, unit-variance) before forward.
- [ ] TNNetSpectralNorm wrapper — wraps an existing FullConnect or
      Convolution layer and divides its weight matrix by its largest
      singular value (one power-iteration step per forward pass).
      NOTE: the reusable power-iteration helper `TNNet.EstimateSpectralNorm`
      now exists (landed with WeightSpectrumReport) — build the wrapper on
      top of it rather than re-deriving the iteration.
- [ ] TNNetStochasticPool — sample one cell per pooling window weighted by
      its activation (softmax of activations over the window) at training,
      take the expectation at inference.
- [ ] TNNetShakeShake / TNNetShakeDrop — Shake-Shake regularization and
      its single-branch ShakeDrop generalization.

#### Channel attention / conditioning
- [ ] TNNetCBAM — SE block plus a spatial-attention sibling.
- [ ] TNNetFiLM (Feature-wise Linear Modulation) — `y = gamma * x + beta`
      with gamma/beta from a separate conditioning input branch.
- [ ] TNNetMaxBlurPool — anti-aliased max-pool: max-pool followed by a
      fixed (non-trainable) binomial blur filter.

#### Activations (gradient-checkable, mostly TNNetReLUBase descendants)
- [ ] TNNetMishExact / TNNetMish-stable — stable formulation for large |x|
      using softplus's stable form (parallel to the SoftPlus negative-x
      derivative guard).
- [ ] TNNetMetaAconC follow-up to the landed TNNetAconC — make the β switch
      data-dependent (β computed per-channel from a tiny squeeze over the
      spatial mean, as in the ACON paper's Meta-ACON). Builds directly on
      the now-landed AconC forward/backward.
- [ ] AconC-vs-Swish-vs-ReLU bake-off: swap each activation into a fixed
      3-layer MLP on the hypotenuse toy, chart final loss + epochs-to-
      converge. AconC has landed, so this is a ~30-line activation swap.
- [ ] TNNetSplineActivation — KAN-flavored per-channel learnable piecewise-
      linear activation with K+1 control points at fixed knots.
- [ ] TNNetBitLinear (BitNet ternary-weight FullConnect) — `sign(W) *
      mean(|W|)` forward with straight-through estimator backward.
- [ ] TNNetMaxOut2 — two-piece special case of TNNetMaxOut with a tighter
      API (no group-count parameter).
- [ ] TNNetAPL follow-up: APL-vs-PReLU-vs-ReLU bake-off on the hypotenuse toy
      (or a tiny CIFAR stub) at matched param count, sweeping the hinge count
      S ∈ {1, 2, 4} — does the extra piecewise capacity buy lower final loss?
      The activation has landed, so this is a ~30-line activation swap.
#### Probability projections / sparsity
- [ ] TNNetGumbelSoftmax — differentiable categorical sampling:
      `softmax((logits + g) / tau)` where `g ~ Gumbel(0,1)`. Two modes
      (soft / hard straight-through).
- [ ] TNNetMixtureOfExperts — top-k softmax gate over N expert sub-networks
      plus a load-balancing auxiliary loss.

#### Normalization primitives
- [ ] TNNetMinMaxNorm follow-up: a per-channel variant (min/max reduced over
      spatial only, independently per depth channel) gated by a flag, mirroring
      the per-(x,y)-over-depth vs full-volume split discussed for L2Normalize.
      Builds on the landed full-volume TNNetMinMaxNorm.
- [ ] TNNetUnitNormConstraint — projection layer that L2-normalizes the
      *weights* of the previous trainable layer after each step.

#### Reduction / shape
- [ ] TNNetAdaptiveAvgPool example/usage: swap a fixed global-avg head
      (`TNNetAvgChannel`) for `TNNetAdaptiveAvgPool.Create(1)` in one
      SimpleImage path, or a tiny demo showing the same conv stack accepting
      two different input resolutions and producing a fixed-size head.
- [ ] TNNetGather — single-channel index-into-a-channel layer.
- [ ] TNNetUpsampleNearest backward consistency: assert summing the
      per-block output errors equals the input error.
- [ ] TNNetSoftPool follow-up: add an optional `beta` temperature to the
      softmax weights (`w_i = exp(beta*x_i)/sum_j exp(beta*x_j)`) so
      `beta -> inf` recovers max-pool and `beta -> 0` recovers avg-pool — a
      single knob spanning the family. Store `beta` in `FFloatSt[0]`,
      extend BOTH dispatch tables to `Create(St[0],St[1],St[2],Ft[0])`, and
      add a `beta`-sweep gradient/limit test. Base layer landed (forward
      `y=sum w_i x_i`, backward `dy/dx_i = w_i*(1+x_i-y)`, beta fixed at 1).
- [ ] Pooling bake-off example `examples/PoolingBakeoff/`: same tiny conv
      classifier, swap the pooling head across `TNNetAvgPool` / `TNNetMaxPool`
      / `TNNetLpPool` (sweep `p ∈ {1, 2, 4, 8}`) / `TNNetSoftPool`, chart final
      loss/accuracy on a small CIFAR stub. Visualises the average<->max
      interpolation empirically. All four pooling layers are in tree.
- [ ] TNNetAdaptiveMaxPool example/usage: a tiny demo showing the same conv
      stack accepting two different input resolutions and producing a
      fixed-size head via `TNNetAdaptiveMaxPool.Create(1)` (global-max head),
      sibling to the open TNNetAdaptiveAvgPool example task above. Layer +
      gradient/forward tests already landed.
### Loss layers
- [X] TNNetCosineEmbeddingLoss — y·(1-cos) + (1-y)·max(0, cos-margin)²
      loss layer. Self-contained `a|b|y` depth-split head (reads FOutput
      directly, no external target — same pattern as TNNetTripletLoss).
- [ ] TNNetCosineEmbeddingLoss follow-up (now landed): a tiny
      siamese-pair embedding micro-example — train two shared-weight MLP
      branches whose outputs are concatenated into the `a|b|y` layout, on a
      synthetic "same vs different class" pair task, and print the learned
      same-pair vs different-pair cosine histograms. Headline use case for
      the landed head; pairs with [[TripletEmbedding]].
- [ ] TNNetKLDivergence follow-up: a knowledge-distillation micro-example —
      train a small "student" against soft targets from a fixed "teacher"
      distribution using the landed TNNetKLDivergence head, and contrast its
      loss curve with a hard-label cross-entropy baseline.
- [ ] Dice/Tversky segmentation micro-example: a tiny synthetic binary-mask
      task (e.g. predict a filled disc/box mask from a noisy input grid) with
      a sigmoid head + `TNNetDiceLoss`, contrasting the IoU it reaches vs an
      MSE-head baseline. Headline use case for the landed Dice/Tversky losses.
- [ ] Tversky α/β asymmetry sweep on the segmentation micro-example: with a
      deliberately class-imbalanced mask, sweep `(α,β) ∈ {(0.5,0.5),(0.3,0.7),
      (0.7,0.3)}` and show how β>α trades precision for recall (fewer false
      negatives). Pure α/β knob study on the landed TNNetTverskyLoss.
- [ ] LabelSmoothing calibration check: train SimpleImageClassifier with
      `TNNetLabelSmoothingLoss(eps)` at `eps ∈ {0, 0.05, 0.1, 0.2}` and feed
      each into the `neuralcalibration` ECE/Brier report — the textbook claim
      is smoothing improves calibration at a small accuracy cost. Both pieces
      (the loss and the calibration report) have landed.
- [ ] TNNetContrastiveLoss / InfoNCE — input split into query/key, computes
      InfoNCE against other samples in the minibatch.
- [ ] TNNetCenterLoss — joint softmax + `λ·||x - c_y||²` with EMA-updated
      class centers stored as the layer's weight tensor.
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
### Introspection / debugging tools
- [ ] TNNet.ToGraphvizDot — emit a `.dot` file describing the layer DAG.
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
- [ ] `TNNet.NeuralTangentKernelReport(NN, Samples)` — the empirical
      Neural-Tangent-Kernel diagnostic (Jacot et al. 2018), the gradient-space
      object that actually governs gradient-descent training dynamics. On a
      FROZEN net (`ClearDeltas` per sample, never `UpdateWeights`) it snapshots
      each sample's full flattened per-parameter weight-gradient vector `g_i` of
      the scalar target-class logit — REUSING the exact per-sample gradient
      machinery `FisherImportanceReport` / `GradientConflictReport` /
      `GradientNoiseScaleReport` already share (no input-gradient enablement) —
      and forms the empirical NTK Gram matrix `K_ij = <g_i, g_j>` over the probe
      batch. It then reports: the kernel as a glyph-shaded ASCII heatmap; its
      FULL eigenspectrum via the SAME self-contained Double-precision cyclic
      Jacobi eigensolver `WeightSpectralTailReport` already ships (so no new
      numerical code); the condition number `lambda_max/lambda_min` (a predictor
      of convergence speed — ill-conditioned NTK ⇒ slow training); the
      **kernel-target alignment** `<K, yy^T>_F / (||K||_F ||yy^T||_F)`
      (Cristianini et al. 2001 — high alignment predicts good generalization,
      the headline NTK-theory number); the effective rank / participation ratio
      `(sum lambda)^2 / sum lambda^2`; a `log10(lambda)` histogram; and an
      optional fresh-init-vs-trained contrast quantifying NTK DRIFT (≈0 drift =
      the infinite-width "lazy/kernel" regime; large drift = "rich"
      feature-learning — the lazy-vs-rich question made visible). Built-in
      correctness checks: symmetry `K_ij == K_ji`, PSD (all Jacobi eigenvalues
      `>= 0` since `K = G G^T`), and the diagonal `K_ii == ||g_i||^2 > 0`.
      DISTINCT from `GradientConflictReport` (which reports normalised pairwise
      gradient COSINES + a conflict fraction — sign geometry, not the
      un-normalised kernel, its eigenspectrum, or target alignment), from
      `RepresentationSimilarityReport` (linear-CKA on forward ACTIVATIONS, not
      gradients) and from `HessianCurvatureReport` (loss-surface curvature, not
      the gradient Gram). Ships with an `examples/NeuralTangentKernel/` demo
      (small classifier; contrast a wide vs narrow hidden layer to show the wide
      net's NTK drifting less) and the standard report test trio. Forward+
      backward only on a frozen net; weights are never stepped.

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
- [ ] Determinism test: same seed → bit-identical forward+backward across
      two runs of a 3-layer net.
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
- [ ] PrintSummary smoke test — capture summary output for canonical
      networks, assert row count and total-parameter line.
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
<!-- (`neural-bench` CLI removed: duplicate of `bin/layer_bench` above, which
     already times forward+backward per layer at a chosen shape with CSV.) -->
### Examples I'd enjoy writing
- [ ] `examples/TinyGPT/` — char-level transformer end-to-end demo on
      a short text snippet (Tiny Shakespeare or repeated arithmetic).
      Highest-value example missing from the repo; natural capstone for
      the transformer-building-blocks line of work.
- [ ] `examples/DeadReLUDiagnostic/` — train a small ReLU net on MNIST
      and print the per-epoch fraction of units that never fire; repeat
      with LeakyReLU/GELU/Swish.
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
      BatchNorm / LayerNorm / RMSNorm / GroupNorm / InstanceNorm.
- [ ] `examples/OptimizerBakeoff/` — SGD / SGD+momentum / Adam / RMSProp
      on a fixed toy dataset with a loss-vs-epoch table.
<!-- (`examples/EmbeddingHeadDemo/` removed: duplicate of the landed
     `examples/TripletEmbedding/` example, which already learns a toy
     multi-class embedding with TNNetL2Normalize and prints the per-class
     cosine-similarity matrix. The only stated difference was a hand-rolled
     triplet loss vs the in-tree TNNetTripletLoss head — not worth a second
     near-identical demo.) -->
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
- [ ] `examples/SubPixelSuperRes/` — once TNNetPixelShuffle lands, a
      3-layer net that learns to 2x-upsample 8x8 random checkerboards.
- [ ] `examples/BiasOnlyTuning/` — freeze a pretrained classifier and
      fine-tune only inserted TNNetChannelBias layers on a new task
      (BitFit-style cheap adaptation).
- [ ] `examples/AffineFineTune/` — once TNNetAffineBlock lands, same
      pattern but freezing everything except the inserted Affine blocks
      (built on TNNetChannelBias + TNNetChannelMul).
- [ ] `examples/TokenShiftBaseline/` — train a tiny next-token char model
      with `TNNetEmbedding → TNNetTokenShift → MLP` and compare against
      the eventual MHA-based version.
- [ ] `examples/ReZeroDeepMLP/` — train a 16-layer residual MLP with and
      without TNNetReZero on each residual branch on the hypotenuse toy.
- [ ] `examples/EuclideanNormHead/` — demo composing `Reciprocal(Sqrt(
      Square(x)))` as a Euclidean-norm-reciprocal head.
- [ ] `examples/SIREN/` — 1D periodic-function fit with TNNetSin.
- [ ] `examples/SpaceToDepthStem/` — show the SpaceToDepth → Conv stem
      replacing a stride-2 conv on a tiny CIFAR stub.
- [ ] `examples/PreNormVsPostNorm/` — toy sequence task with the same
      sublayer wired through PreNorm vs PostNorm builders.
- [ ] `examples/MaxoutMnist/` — minimum-viable Maxout demo on a tiny-MNIST
      subset (or synthetic 2D classification).
- [ ] `examples/ModelSummaryDemo/` — three networks printed via
      PrintSummary; doubles as a smoke test for the summary output format.
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
- [ ] `examples/SparseAttentionDemo/` — once TNNetSparsemax lands, toy
      "predict next char of a periodic sequence" using Sparsemax in place
      of softmax over a tiny K|V bank. Print attention-weight histogram
      per step.
- [ ] `examples/FiLMConditional/` — toy "draw a digit of class C" generator
      with FiLM conditioning on a 10-way one-hot class input.
- [ ] `examples/TripletEmbedding/` MNIST follow-up: a true MNIST version of the
      landed synthetic TripletEmbedding demo, with a PGM scatter-plot output.
- [ ] `examples/VQAutoencoder/` — extend VisualAutoencoder with a
      TNNetVectorQuantizer bottleneck.
- [ ] `examples/AntiAliasedMaxPool/` — train the same tiny CIFAR-10 net
      once with TNNetMaxPool and once with TNNetMaxBlurPool; report
      shift-equivariance delta.
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
- [ ] InstanceNorm vs GroupNorm vs LayerNorm vs ChannelStdNorm single-seed
      bake-off on a 3-layer CIFAR-ish conv stack.
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
- [ ] Charbonnier-vs-Huber-vs-MSE-vs-LogCosh head-to-head on the noisy-
      hypotenuse harness.
- [ ] Loss-family bake-off (output heads): hypotenuse with MSE / Huber /
      SmoothL1 / Charbonnier / LogCosh, printing final MSE and epochs-to-
      converge.
- [ ] TanhGLU vs SwiGLU vs GEGLU vs GLU vs ReGLU bake-off: same tiny seq
      model, swap the gating layer, chart final loss and wall-clock.
- [ ] LogCoshDualExperiment longer-horizon follow-up: 200-300 epochs and
      5 seeds (mean ± std reporting).
- [ ] Plain-Tanh vs TanhGLU FFN ablation in a minimal-transformer-without-
      attention skeleton.
- [ ] DyT-vs-LayerNorm bake-off. TNNetDyT has now landed, so this is
      unblocked — a 30-line swap in the existing normalization bake-off
      harness (or a small standalone synthetic-regression A/B).
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
      count. (UNBLOCKED: TNNetRReLU has now landed — remember to flip its
      `Enabled` flag off for the eval pass so the fixed average slope is used.)
- [ ] TopK sparsity sweep: train the same tiny autoencoder bottleneck
      with K ∈ {1, 2, 4, 8, 16, full}, chart reconstruction loss vs sparsity.
- [ ] STE bit-width sweep: same network, vary `step ∈ {1.0, 0.5, 0.25,
      0.125, 0.0625}`, plot accuracy vs bit-width.
- [ ] Straight-through quantization demo: small classifier with one hidden
      layer's outputs passed through a TNNetStraightThroughEstimator;
      compare accuracy against unquantized baseline.
<!-- (Lottery-ticket sanity check removed: duplicate of the "Lottery-ticket"-
     flavored experiment under "Bake-off / experiment follow-ups".) -->
- [ ] Sequence-length scaling micro-benchmark — TNNetScaledDotProductAttention
      at seq_len ∈ {16, 32, 64, 128, 256} with d_k fixed. Confirms O(n²)
      scaling.
- [ ] Channel-attention bake-off: fixed tiny CIFAR backbone, four variants
      — (a) no attention, (b) SE, (c) CBAM, (d) hand-rolled "1x1 + sigmoid".
- [ ] FiLM-vs-concat conditioning bake-off on a class-conditional MNIST
      decoder.
- [ ] VQ codebook collapse stress test: K in {16, 64, 256} and a few
      commitment-loss weights, report per-run active codebook entries.
- [ ] "Tiny induction-heads" demo: train a 2-layer attention-only model on
      a repeat-the-pattern toy task and show the second layer's attention
      diagonal jumps to the previous-occurrence position.
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
- [ ] "Surgery" experiment: train a small classifier, then zero out the
      top-K most-active hidden units and chart accuracy degradation vs K.
<!-- (Plain "label-smoothing sweep — tabulate test accuracy" removed:
     subsumed by the "LabelSmoothing calibration check" entry under
     "### Loss layers", which runs the SAME SimpleImageClassifier sweep over
     the SAME eps ∈ {0, 0.05, 0.1, 0.2} and additionally feeds each into the
     neuralcalibration ECE/Brier report — a strict superset.) -->
- [ ] SWA effect-size sweep: vary SWA start-epoch fraction ∈ {0.5, 0.6,
      0.7, 0.8, 0.9} and chart final test accuracy.
- [ ] Cosine-LR vs constant-LR on SimpleImageClassifier, three seeds each.
- [ ] "Activation cost" microbenchmark — measure forward+backward ns/op
      on a fixed 64x64x32 volume for each activation.
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
- [ ] TNNetHardShrink / TNNetSoftShrink sparsity micro-experiment: train a
      tiny autoencoder with each as the bottleneck activation, print
      fraction of zero activations vs reconstruction loss.

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
      assembled code snippet. Blocked only on MHSA.
- [ ] "Loss functions" README subsection grouping MSE, MAE, CE, Huber,
      SmoothL1, LogCosh, Charbonnier, and (once landed) Dice/KL/Focal/
      LabelSmoothing/CosineEmbedding into a single short table.
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
- [ ] `docs/channel_attention.md` — once SE / CBAM / FiLM land, compare
      them on the same axes.
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
      TNNetSinusoidalTimeEmbedding. Depends on FiLM + timestep embedding.
- [ ] `examples/HopfieldRetrieval/` — modern Hopfield network as attention
      (Ramsauer et al.): store K patterns, retrieve via a single softmax-
      attention step against a query.
- [ ] Mixed-precision experiment first step: add `TNeuralFloat16 = packed
      record ...` in neuralvolume.pas with conversion helpers, plus a
      one-layer forward-only test validating FP16 matches FP32 to within
      1e-2.

### Introspection (added)
- [ ] FeatureSeparabilityReport follow-up (now landed): the scatter-
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
- [X] TNNet.FeatureSeparabilityReport(NN, Samples, NumClasses) — a label-aware
      **class-geometry / Neural-Collapse** diagnostic (Papyan, Han & Donoho 2020,
      "Prevalence of neural collapse during the terminal phase of training")
      answering a question none of the landed reports answer: *"how tightly does
      each layer cluster the samples of a class, and how far apart are the
      classes?"* — the geometry of the feature space, not its decodability. For
      every trainable layer it runs one forward pass over the labelled probe
      batch, flattens each sample's activation to a row, and computes the
      Fisher-style scatter decomposition: per-class mean `mu_c`, the global mean
      `mu`, the **within-class scatter** `tr(S_w) = mean_c mean_{i in c}
      ||x_i - mu_c||^2` (cluster tightness — the Neural-Collapse "NC1" variability
      collapse), the **between-class scatter** `tr(S_b) = mean_c ||mu_c - mu||^2`
      (how spread the class means are), and their ratio `tr(S_b)/tr(S_w)` (the
      Fisher discriminant ratio — higher == cleaner clusters). It also reports the
      mean **silhouette coefficient** over the batch (cohesion-vs-separation in
      `[-1,1]`, label-aware, no classifier fit), the class-mean pairwise-cosine
      matrix (numeric + glyph heatmap) with the **simplex-ETF angle check** — under
      neural collapse off-diagonal cosines converge to `-1/(NumClasses-1)`, so the
      report prints the mean off-diagonal cosine next to that target as a built-in
      NC2 indicator — a per-layer separability-across-depth ASCII bar chart, and
      `C`ollapse / well-`S`eparated / near-`R`andom flags. The story to show in the
      example: at fresh init the ratio is ~0 and silhouette ~0 (classes fully
      overlap) and *degrades* slightly with depth; after training the ratio and
      silhouette climb monotonically toward the penultimate layer and the off-
      diagonal class-mean cosines approach the simplex-ETF target — the geometric
      signature of neural collapse made visible. Built-in correctness checks:
      `tr(S_total) == tr(S_w) + tr(S_b)` to <1e-4 per layer (the scatter
      decomposition identity — the faithfulness check), a single-class batch makes
      `tr(S_b)` collapse to ~0, and identical per-class samples make `tr(S_w)`
      collapse to ~0 with silhouette -> 1. Pure forward-only; weights are never
      touched and no backward pass is run. **Distinct from** [[LinearProbeReport]]
      (fits a ridge classifier and reports *decodability* / accuracy — a model can
      be linearly decodable yet have loose, barely-separated clusters; this measures
      the cluster geometry directly with no fit), from `RepresentationSimilarityReport`
      (CKA *between two layers*, label-free — says nothing about per-class structure),
      from `NeuronCorrelationReport` (intra-layer neuron redundancy, no labels), and
      from `TopLogitMarginReport` (margin on the *output* logits only, not the
      intermediate feature geometry). Follows the introspection-report-pattern:
      declaration + impl in neuralnetwork.pas, a `examples/FeatureSeparability/`
      demo on a synthetic 3-class 2D-Gaussian-blob set (well-separated vs
      deliberately-overlapped variants so the ratio/silhouette contrast is visible
      in one run), and a smoke test in tests/ (non-empty report, expected header,
      nil-NN graceful return, plus the scatter-decomposition identity assertion).
      Pairs naturally with the open lottery-ticket / grokking experiments
      ([[WeightSpectrumReport]]) — watching `tr(S_w)` collapse is the cleanest
      single-number window into the terminal phase of training.
- [ ] TNNet.PredictionDepthReport(NN, Support, SupportLabels, Queries
      [, K, QueryLabels]) — a **per-example difficulty** diagnostic built on
      the *prediction depth* of Baldock, Maennel & Neyshabur 2021 ("Deep
      Learning Through the Lens of Example Difficulty"), answering a question
      none of the landed reports answer: *"at how deep a layer does the network
      actually make up its mind about THIS example?"* — a per-sample resolution
      depth, not a per-layer aggregate. The recipe is forward-only and
      non-parametric: run one forward pass over a labelled **support** batch and
      over the **query** batch, snapshot each layer's flattened activation, and
      for every query, at every trainable layer, take a **k-NN vote** (default
      K=5, cosine distance) over the support activations at that same layer. The
      *prediction depth* of a query is the index of the **shallowest layer after
      which the k-NN vote agrees with the network's final argmax and never
      disagrees again** (deeper layers all confirm it) — easy examples are
      decided early (shallow depth), hard / ambiguous / mislabelled examples stay
      contested until the last layers (deep depth). Reports: a 10-bin ASCII
      histogram of prediction depth across the query batch, mean / median depth,
      the per-layer "newly-resolved" count (how many queries first lock in at
      each layer — a depth-vs-layer profile), the K deepest (= hardest) query
      indices as a ready-made hard-example / relabel-candidate queue, and — when
      `QueryLabels` are supplied — a correctness cross-tab (mean prediction depth
      of correctly vs incorrectly classified queries; the literature's headline
      result is that depth correlates with error and with margin). Over-wide
      layers are random-projected to a `MaxFeatDim` cap (default 256) to bound
      the distance cost, reusing the projection trick already in
      [[LinearProbeReport]]. Built-in correctness checks: feeding the **support
      set as its own queries** must give every sample a finite depth and the
      final-layer k-NN vote must match the network argmax for ≥ a high fraction
      (the support is its own nearest neighbour at distance 0); a query whose
      final argmax is already the majority vote at layer 0 reads depth 0; and a
      one-class support set drives every depth to 0 (trivially agreeing). The
      story for the example: at fresh init prediction depths pile up at the
      LAST layer (nothing is resolved early — the histogram is right-skewed),
      while after training the mass shifts shallow for the well-separated
      clusters and a hard / label-noised subset keeps a deep tail — example
      difficulty made visible. **Distinct from** [[LinearProbeReport]] (fits a
      *parametric* ridge classifier per layer and reports per-layer *accuracy* /
      decodability — a global "where does the net become linearly separable?"
      number, not a per-example depth, and it needs a matrix solve where this
      needs only distances), from `FeatureSeparabilityReport` (per-layer *cluster
      geometry*, label-aware but aggregate, no per-sample score), from
      `TopLogitMarginReport` (output-logit margin only — a confidence number at
      the *last* layer, with no notion of *which* layer resolved the example),
      and from `MCDropoutUncertaintyReport` (stochastic epistemic uncertainty,
      not a deterministic depth). Follows the introspection-report-pattern:
      declaration + impl in neuralnetwork.pas, a `examples/PredictionDepth/` demo
      on a synthetic multi-class 2D-blob set (a clean split plus a deliberately
      label-noised subset so the easy-shallow / hard-deep contrast and the
      correct-vs-incorrect depth gap are visible in one run), and a smoke test in
      tests/ (non-empty report, expected header, nil-NN graceful return, plus the
      support-as-its-own-queries finite-depth / final-layer-agreement assertion).
      Pairs naturally with the active-learning queue use of
      [[MCDropoutUncertaintyReport]] and the hard-example pools of
      `TopLogitMarginReport` / `ConfusionMatrixReport`.
- [ ] TNNet.ActivationPatchingReport(NN, CleanInput, CorruptInput
      [, TargetIdx]) — a **causal** mechanistic-interpretability diagnostic
      (activation patching / causal tracing; Vig et al. 2020, Meng et al.
      ROME 2022, Wang et al. IOI 2022) answering a question NONE of the
      landed reports answer: *"which layer's activations CARRY the
      information that decides this prediction?"* — measured by intervention,
      not correlation. The recipe: take a `(CleanInput, CorruptInput)` pair
      the trained net maps to different argmax classes (two samples of
      different true class, or one sample and a noised/blanked copy). Run a
      clean forward and cache every layer's `Output`; run a corrupt forward;
      then for each intermediate layer L in turn, restore that single layer's
      cached CLEAN activation into the corrupt run (overwrite
      `FLayers[L].Output` via `CopyNoChecks`) and recompute layers L+1..last
      (the existing per-layer `Compute()` loop / `Compute(pInput,
      FromLayerIdx)` machinery already forwards from a layer), reading off the
      **recovery** `r_L = (logit_c(patch_L) - logit_c(corrupt)) /
      (logit_c(clean) - logit_c(corrupt))` where `c` is the clean argmax class
      (`TargetIdx`, default = clean argmax). `r_L≈1` ⇒ patching layer L alone
      restores the clean decision (the information lives there / is read by
      the rest of the net); `r_L≈0` ⇒ that layer carries nothing causal for
      this contrast. Reports: a per-layer `r_L` ASCII bar chart across depth
      (the causal-trace curve — the headline plot), the argmax-flip layer
      (shallowest L whose patch flips the corrupt argmax back to `c`), the
      peak-recovery layer and its `r_L`, an early-/late-/distributed
      localisation verdict, and a per-layer un-normalised delta-logit column
      (`logit_c(patch_L) - logit_c(corrupt)`) so a near-zero normalisation
      denominator is visible. Built-in correctness checks: patching the INPUT
      layer (L=0) gives `r_0 == 1` exactly (it reconstructs the full clean
      run — the faithfulness check), patching the LAST layer gives
      `r_last == 1` exactly (its output IS the logits), `CorruptInput :=
      CleanInput` collapses the denominator so the report warns rather than
      dividing by zero, and the net's live state is restored by a final clean
      recompute. **Distinct from** [[SaliencyReport]] /
      `AdversarialRobustnessReport` (INPUT-space gradient attribution — which
      pixels matter, a correlational gradient, not a layer-localising causal
      intervention), from `LayerSensitivityReport` (jitters WEIGHTS with
      random Gaussian noise and measures output drift — perturbation
      magnitude, never swaps real activations between two inputs), from
      `FisherImportanceReport` (per-PARAMETER importance averaged over a
      batch, not a per-layer causal recovery on one contrastive pair), and
      from `LinearProbeReport` / `FeatureSeparabilityReport` (decodability /
      geometry — what's PRESENT in a layer, not what the rest of the net
      causally USES). Forward-only (no backward pass, no weight steps); the
      only mutation is the transient activation overwrite, reverted by the
      final clean recompute. Follows the introspection-report-pattern:
      declaration + impl in neuralnetwork.pas, an `examples/ActivationPatching/`
      demo on a synthetic task where the causal layer is KNOWN by construction
      (a 2-stage net whose stage 1 must compute an intermediate feature the
      clean/corrupt pair differ on, so the trace peaks at the stage boundary —
      a ground-truth localisation check), and a smoke test in tests/ (non-empty
      report, expected header, nil-NN graceful return, plus the `r_0==1` /
      `r_last==1` exact-recovery assertions).
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
- [ ] (TTAReport follow-up) the shipped report runs on a single synthetic
      probe set; a natural next step is the spec's second run on a model
      trained WITH `TNNetRandomFlipX` augmentation, to show TTA gains shrink
      when the invariance is already learned. given a trained classifier, a validation set,
      and a configurable list of input-side transforms (default menu reuses
      the existing in-tree augmentations: identity, `TNNetFlipX`,
      `TNNetFlipY`, `TNNetReverseChannels`, and a 1-pixel `Roll(X=+1)`),
      run a forward pass per transform, average the resulting logits (and,
      optionally, the post-softmax probabilities — both modes selectable via
      a flag so the linear-vs-geometric-mean question is empirically
      checkable), and report:
      (a) baseline top-1 accuracy on the untransformed inputs,
      (b) per-transform top-1 accuracy (each transform applied alone — a
          built-in correctness check for the augmentation; a healthy model
          shouldn't lose much accuracy under any single near-invariant
          transform),
      (c) full-ensemble TTA top-1 accuracy (all transforms averaged
          together) and the delta vs baseline,
      (d) per-class accuracy delta so classes that *lose* under TTA (a
          sign of a non-equivariant decision boundary for that class) are
          visible,
      (e) per-sample agreement rate `mean(argmax(avg_logits) ==
          argmax(baseline_logits))` — high agreement + small accuracy lift
          means TTA mostly confirms existing decisions; low agreement +
          large lift means TTA is genuinely flipping borderline samples,
      (f) a one-line verdict: "TTA helps" / "TTA neutral" / "TTA hurts"
          based on a configurable threshold.
      Pure forward-only — no training-time changes, no backward pass.
      Distinct from [[EquivarianceReport]] (measures how *output* reacts
      to transforms in isolation, without using a label or computing
      accuracy — answers "is the model invariant?" not "does averaging
      under transforms improve val accuracy?"), from [[SaliencyReport]]
      (per-sample input attribution, not aggregate accuracy lift), and
      from the calibration / margin-report tools (those summarise
      confidence quality on the untransformed set, no ensembling). The
      output is the natural input for any future "should we ship TTA at
      inference?" decision — you need to know which transforms actually
      help before paying their inference cost. Companion
      `examples/TestTimeAugmentation/` runs it on the SimpleImageClassifier
      CIFAR baseline so the per-transform table and ensemble delta can be
      eyeballed; the example also runs against a model trained *with*
      `TNNetRandomFlipX` augmentation so reviewers can see the expected
      pattern (TTA gains shrink when the model has already learned the
      invariance during training — TTA and train-time augmentation are
      substitutes, not complements).

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
