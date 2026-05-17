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
- [ ] Sparse / mixture-of-experts routing layer

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
- [ ] SwiGLU FFN follow-up: package the `Dense(2*D) → SwiGLU → Dense(D_out)`
      pattern as `TNNet.AddSwiGLUFeedForward(D_in, D_hidden, D_out)` builder
      so the example becomes one line. Companion of the
      [[TNNetSwiGLUFeedForward]] block helper task below.
- [ ] Quick-start example: tiny char-level sequence model (XOR-of-bits or
      counting task) that trains in well under a minute on CPU.
- [ ] Volume unit micro-benchmark printing ns/op for Add, Mul, DotProduct so
      regressions are visible without OpenCL/AVX hardware differences.
#### Experiments I'm curious about
- [ ] Weight-initialization sensitivity demo: show how a deep-ish net's
      first-epoch gradient magnitudes change across the available init schemes.
#### Documentation
- [ ] Write a short "how numerical gradient testing works in this repo" note so
      contributors can add layer tests confidently — it's the project's main
      correctness safety net but isn't explained anywhere. Should cover the
      eps choice, central differences, tolerance picking, and where to add
      new tests.
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
- [ ] TNNetSinkAttention — prepend K learnable "attention sink" key/value
      slots that every query can attend to regardless of the causal mask.
      Helps long-context stability, ~30 lines on top of SDPA.
- [ ] TNNetTalkingHeadsProjection — pre/post-softmax linear mix across
      heads (Shazeer et al.). A tiny learnable HxH multiply applied to
      attention logits along the head axis.
- [ ] TNNetCosineSimilarityAttention — replace `Q·Kᵀ / √d` with
      `(Q/||Q||)·(K/||K||)ᵀ * scale`. Bounded-by-1 logits remove the need
      for SoftCapping in the attention head.
- [ ] SDPA all-masked-row policy decision and test: currently a row where
      every key is masked produces NaN (softmax of all -inf). Concrete
      proposal: detect the all-masked row in Compute, output a zero row,
      skip the softmax for that row (what JAX/Flax MHA does). Document
      the choice in code and add the pinning test.
- [ ] "Attention numerical-gradient stress test" running the SDPA grad check
      across SeqLen ∈ {1, 2, 3, 5, 8} and asserting the max error vs
      tolerance at each. Pins shape-edge behavior the existing single-shape
      test can't see.
- [ ] Inline shape annotations on TNNetScaledDotProductAttention's Compute
      and Backpropagate — most algorithmically dense layer in the repo;
      deserves a comments pass.

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
- [ ] TNNetSwiGLUFeedForward block helper — `LayerNorm → Dense → SwiGLU →
      Dense out` builder. All ingredients in tree.
- [ ] TNNetGLUFeedForward block — same shape but using the plain TNNetGLU.
      Gives a working FFN to test the pre-norm-residual builder against
      today, no waiting on new gating layers.
- [ ] AddGEGLUFeedForward(NN, d_model, d_ff) — GEGLU twin of the SwiGLU
      FFN builder.
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
- [ ] TNNetMaskedMean / TNNetMaskedMax — pooling over a variable-length
      sequence with a {0,1} mask supplied as an extra depth channel.
      Replaces the "pad with zeros and hope average is small" workaround.

#### Norm / regularization
- [ ] TNNetGatedResidual — per-channel zero-initialised learnable gate
      `y = x + alpha[c] * Sublayer(x)` (ReZero-with-channel-dim variant).
- [ ] TNNetDyT (Dynamic Tanh, Liu et al. 2025) — `gamma[c] * tanh(alpha * x)
      + beta[c]`. Per-layer learnable alpha plus per-channel gamma/beta.
- [ ] TNNetRMSNormGated — RMSNorm followed by a learnable per-channel
      sigmoid gate.
- [ ] TNNetGRN (Global Response Normalization, ConvNeXt-V2) — channel-wise
      contrast normalization with learnable scale/bias.
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
- [ ] TNNetStochasticPool — sample one cell per pooling window weighted by
      its activation (softmax of activations over the window) at training,
      take the expectation at inference.
- [ ] TNNetShakeShake / TNNetShakeDrop — Shake-Shake regularization and
      its single-branch ShakeDrop generalization.

#### Channel attention / conditioning
- [ ] TNNetCBAM — SE block plus a spatial-attention sibling.
- [ ] TNNetFiLM (Feature-wise Linear Modulation) — `y = gamma * x + beta`
      with gamma/beta from a separate conditioning input branch.
- [ ] TNNetCoordConv — concatenate two normalized X/Y coordinate channels
      before a convolution. Parameter-free.
- [ ] TNNetMaxBlurPool — anti-aliased max-pool: max-pool followed by a
      fixed (non-trainable) binomial blur filter.

#### Activations (gradient-checkable, mostly TNNetReLUBase descendants)
- [ ] ~~TNNetIdentityScale~~ — fixed scalar multiplier is already covered
      by the in-tree `TNNetMulByConstant(c)` (constructor stores the
      multiplier in FFloatSt[0] and the forward pass is a pure scalar
      multiply; the "Learning" part of its parent class doesn't engage
      unless wired up). Do NOT add a separate class — use
      TNNetMulByConstant for the warm-up-scaling trick.
- [ ] TNNetSwishLearnable — TNNetSwish with a single learnable β.
- [ ] TNNetMishLearnable — TNNetMish with a single learnable α.
- [ ] TNNetMishExact / TNNetMish-stable — stable formulation for large |x|
      using softplus's stable form (parallel to the SoftPlus negative-x
      derivative guard).
- [ ] TNNetRReLU — Randomized Leaky ReLU; slope sampled uniformly per
      neuron per forward pass during training, fixed at average at inference.
- [ ] TNNetISRU / TNNetISRLU — Inverse-Square-Root (Linear) Unit.
      `y = x / sqrt(1 + α·x²)`.
- [ ] TNNetSoftPlusBeta — generalized SoftPlus with learnable-or-fixed β.
- [ ] TNNetSoftExponential — `(exp(α·x) - 1)/α + α` for α>0, identity for
      α=0, `-log(1 - α·(x + α))/α` for α<0.
- [ ] TNNetAconC — "Activate Or Not": `(p1-p2)·x·sigmoid(β(p1-p2)x) + p2·x`
      with channel-wise learnable `(p1, p2, β)`. Generalizes Swish.
- [ ] TNNetPhish — `x · tanh(softplus(x))` (Mish sibling using softplus
      inside tanh).
- [ ] TNNetErf — closed-form GELU partner. Caveat: check FPC math.erf
      portability (or reuse the SerfErf A&S polynomial helper).
- [ ] TNNetSReLU — S-shaped ReLU with four learnable knee parameters per
      channel.
- [ ] TNNetSplineActivation — KAN-flavored per-channel learnable piecewise-
      linear activation with K+1 control points at fixed knots.
- [ ] TNNetBitLinear (BitNet ternary-weight FullConnect) — `sign(W) *
      mean(|W|)` forward with straight-through estimator backward.
- [ ] TNNetMaxOut2 — two-piece special case of TNNetMaxOut with a tighter
      API (no group-count parameter).
- [ ] TNNetSinusoidalTimeEmbedding — scalar-timestep encoder for diffusion
      models (distinct from sequence-axis TNNetSinusoidalPositionalEmbedding).
- [ ] TNNetAPL (Adaptive Piecewise Linear) — sum of hinge functions with
      per-channel learnable knees and slopes.
- [ ] TNNetCenteredSoftmax — softmax preceded by per-sample mean subtraction.

#### Probability projections / sparsity
- [ ] TNNetSparsemax — Martins & Astudillo's exact-sparse alternative to
      softmax. Yields true zeros; natural drop-in for sparse attention.
- [ ] TNNetGumbelSoftmax — differentiable categorical sampling:
      `softmax((logits + g) / tau)` where `g ~ Gumbel(0,1)`. Two modes
      (soft / hard straight-through).
- [ ] TNNetMixtureOfExperts — top-k softmax gate over N expert sub-networks
      plus a load-balancing auxiliary loss.

#### Normalization primitives
- [ ] TNNetL2Normalize variants beyond the landed per-(x,y)-over-depth
      version: full-volume L2 normalization, per-channel L2 over spatial,
      configurable axis via FStruct[0].
- [ ] TNNetUnitNorm — alias for L2Normalize on the full volume (Keras name).
- [ ] TNNetMinMaxNorm — `(x - min(x)) / (max(x) - min(x) + eps)` per sample,
      with subgradient routing for the argmin/argmax cells.
- [ ] TNNetUnitNormConstraint — projection layer that L2-normalizes the
      *weights* of the previous trainable layer after each step.

#### Reduction / shape
- [ ] TNNetPixelShuffle (sub-pixel convolution) — output spatial size =
      input * r, output channels = input / (r*r). Deterministic index
      permutation; backward is its inverse.
- [ ] TNNetAdaptiveAvgPool — target output (X,Y) regardless of input size.
- [ ] TNNetCumSum follow-up: configurable axis (X / Y / Depth) via
      FStruct[0]. Depth-only version already landed.
- [ ] TNNetRoll follow-up: configurable axis selector. Depth-only
      version with Shift in FStruct[0] already landed.
- [ ] TNNetGather — single-channel index-into-a-channel layer.
- [ ] TNNetSqueeze / TNNetExpandDims — numpy-style single-axis shape
      helpers, less error-prone than open-coding TNNetReshape.
- [ ] TNNetLpPool — generalized pooling `(mean(|x|^p))^(1/p)` with
      configurable p.
- [ ] TNNetUpsampleNearest backward consistency: assert summing the
      per-block output errors equals the input error.

### Loss layers
- [ ] TNNetLabelSmoothingLoss helper — pure target-side transform
      `(1 - eps) * one_hot + eps / NumClasses`.
- [ ] TNNetNLLLoss — companion to TNNetLogSoftMax. NLL over (X,Y,Depth)
      with class index targets.
- [ ] TNNetCosineEmbeddingLoss — y·(1-cos) + (1-y)·max(0, cos-margin)²
      loss layer.
- [ ] TNNetKLDivergence — `sum(p · log(p/q))` with stability clamps on q.
- [ ] TNNetDiceLoss — `1 - 2·sum(p·q + ε) / (sum(p²) + sum(q²) + ε)`,
      IoU-flavored segmentation loss.
- [ ] TNNetTverskyLoss — generalized Dice with separate FP/FN weights α, β.
- [ ] TNNetWingLoss — facial-landmark regression loss with log-shaped wing
      near zero and a linear tail.
- [ ] TNNetTripletLoss — `max(0, ||a-p||² - ||a-n||² + margin)`. Input
      depth split into 3 equal anchor/positive/negative chunks.
- [ ] TNNetContrastiveLoss / InfoNCE — input split into query/key, computes
      InfoNCE against other samples in the minibatch.
- [ ] TNNetCenterLoss — joint softmax + `λ·||x - c_y||²` with EMA-updated
      class centers stored as the layer's weight tensor.
- [ ] TNNetArcFace — additive angular-margin softmax for face/embedding
      recognition heads.
- [ ] TNNetEntropyRegularizer — passthrough layer adding
      `-λ * sum(p * log(p))` to the gradient.
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
- [ ] `neural-bench` tiny CLI: time forward + backward for a chosen layer
      at a chosen shape, print ns/op. CSV output so future regressions
      are visible.
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
- [ ] `examples/ActivationBakeoff/` — pure-CPU bake-off of many
      activations on a fixed small MLP and dataset; one table of final
      loss and epochs-to-converge.
- [ ] `examples/NormalizationBakeoff/` — same idea comparing no-norm /
      BatchNorm / LayerNorm / RMSNorm / GroupNorm / InstanceNorm.
- [ ] `examples/OptimizerBakeoff/` — SGD / SGD+momentum / Adam / RMSProp
      on a fixed toy dataset with a loss-vs-epoch table.
- [ ] `examples/CharTokenizer/` — minimal in-memory char tokenizer +
      trainable embedding lookup, with a nearest-neighbor printout
      ("nearest 5 chars to 'q'").
- [ ] `examples/EmbeddingHeadDemo/` — train a small net to learn an
      embedding space on a toy 3-class dataset using TNNetL2Normalize +
      a hand-rolled triplet loss, print the per-class cosine-similarity
      matrix.
- [ ] `examples/EmbeddingVisualization/` — contrastive head on a 4-class
      toy 2D dataset, dump learned embeddings to CSV with README plotting
      instructions.
- [ ] `examples/MixUpAblation/` — train SimpleImageClassifier with and
      without MixUp on CIFAR-10 and report the delta.
- [ ] `examples/AttentionViz/` — load a tiny trained SDPA model and dump
      the per-head attention matrix as a PGM image.
- [ ] `examples/PositionalEncodingDemo/` — visualize (ASCII heatmap) the
      sin/cos table built by TNNetSinusoidalPositionalEmbedding vs the
      learnable TNNetAddPositionalEmbedding table after a few epochs.
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
- [ ] `examples/EnergyHeads/` — tiny regression demo using TNNetAbs
      (L1-energy target) and TNNetSquare (L2-energy target) as final
      feature heads, side-by-side.
- [ ] `examples/EuclideanNormHead/` — demo composing `Reciprocal(Sqrt(
      Square(x)))` as a Euclidean-norm-reciprocal head.
- [ ] `examples/SIREN/` — 1D periodic-function fit with TNNetSin.
- [ ] `examples/PReLUvsLeakyReLU/` — three-config bake-off.
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
- [ ] `examples/TripletEmbedding/` — learn a 2D embedding of MNIST digits
      using TNNetTripletLoss; output a PGM scatter plot.
- [ ] `examples/VQAutoencoder/` — extend VisualAutoencoder with a
      TNNetVectorQuantizer bottleneck.
- [ ] `examples/CoordConvSpiral/` — minimal CoordConv vs plain conv
      comparison on "predict (x, y) from a one-hot pixel image".
- [ ] `examples/AntiAliasedMaxPool/` — train the same tiny CIFAR-10 net
      once with TNNetMaxPool and once with TNNetMaxBlurPool; report
      shift-equivariance delta.
- [ ] `examples/AbsSquareEnergy/`, `examples/ReverseXYAugmentation/`,
      `examples/AutoencoderMNIST/`, `examples/AutoencoderReconstructionGrid/`
      — additional small demos.
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
- [ ] Numerical-gradient epsilon study on an existing test with
      `epsilon ∈ {1e-2, 1e-3, 1e-4, 1e-5}` tabulating observed max abs
      error.
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
- [ ] DyT-vs-LayerNorm bake-off (depends on TNNetDyT). Once DyT lands,
      a 30-line swap in the existing normalization bake-off harness.
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
      count.
- [ ] TopK sparsity sweep: train the same tiny autoencoder bottleneck
      with K ∈ {1, 2, 4, 8, 16, full}, chart reconstruction loss vs sparsity.
- [ ] STE bit-width sweep: same network, vary `step ∈ {1.0, 0.5, 0.25,
      0.125, 0.0625}`, plot accuracy vs bit-width.
- [ ] Straight-through quantization demo: small classifier with one hidden
      layer's outputs passed through a TNNetStraightThroughEstimator;
      compare accuracy against unquantized baseline.
- [ ] Lottery-ticket sanity check — train, record top-k% magnitude mask,
      reset to original init, retrain with mask applied.
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
- [ ] "Learn to copy" toy: SeqLen=8 input → output the same sequence.
- [ ] "Learn to reverse" toy: same shape, output reversed.
- [ ] "Smallest net that can learn parity-N" study — sweep N ∈ {2, 4, 6, 8}.
- [ ] "Surgery" experiment: train a small classifier, then zero out the
      top-K most-active hidden units and chart accuracy degradation vs K.
- [ ] Label-smoothing sweep — train SimpleImageClassifier with `ε ∈ {0,
      0.05, 0.1, 0.2}`, tabulate test accuracy.
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
- [ ] Exp/Log compose-as-identity test on a small input range.
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
- [ ] Numerical-gradient test confirming TNNetRMSNorm matches the analytical
      gradient under non-trivial input distributions (mean ≠ 0, var ≠ 1).
- [ ] TNNetHardShrink / TNNetSoftShrink sparsity micro-experiment: train a
      tiny autoencoder with each as the bottleneck activation, print
      fraction of zero activations vs reconstruction loss.

### Documentation I'd enjoy writing
- [ ] "Activations cheat sheet" in `docs/activations.md`: one row per
      activation with formula, derivative, saturating?, smooth-at-zero?,
      typical use case.
- [ ] `docs/activation_taxonomy.md` — organise the ~50 activations now in
      the repo by mathematical family.
- [ ] "Normalization cheat sheet" in `docs/normalization.md`: LayerNorm vs
      RMSNorm vs GroupNorm vs InstanceNorm vs ChannelStdNorm vs PixelNorm —
      axes each reduces over, learnable params, typical use.
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
- [ ] "Embedding heads" README subsection — group TNNetL2Normalize,
      TNNetCosineSimilarity, TNNetCosineEmbeddingLoss, TNNetTripletLoss with
      a one-paragraph "how to build a contrastive head" recipe.
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

### Model calibration / reliability
- [ ] Model-calibration / reliability-diagram tool — a small unit
      (`neuralcalibration.pas`) that takes a trained classifier, a
      validation set, and a bin count, and reports Expected Calibration
      Error (ECE), Maximum Calibration Error (MCE), and Brier score, plus
      dumps a reliability diagram as a PGM (per-bin accuracy vs. confidence,
      with a y=x reference line). Pair with a tiny example on top of an
      existing classifier that prints metrics before and after a one-parameter
      temperature-scaling fit on the logits.

### Introspection (added)
- [ ] Top-logit margin report — small helper in a new
      `neuralintrospection.pas` (or extending the calibration unit above)
      that, given a trained classifier and a validation set, computes the
      per-sample `(top1_logit - top2_logit)` margin and prints:
      (a) overall margin histogram (10 bins) as an ASCII bar chart,
      (b) per-class mean+median margin (catches a class the model is
      systematically uncertain about),
      (c) the N lowest-margin sample indices per class (a ready-made
      "hard examples" pool for active learning, label-noise auditing, or
      curriculum work). Reuses the existing forward pass — no training-
      time changes. Pure-CPU, finishes in one validation pass.
      Companion `examples/MarginReport/` runs it against the
      SimpleImageClassifier CIFAR baseline and prints the three sections
      to stdout, so the output format is pinned and easy to eyeball.
      Distinct from the calibration tool above (which summarises
      confidence quality across the whole set) — this one localises
      *which samples* the model is least sure about.
### Input attribution
- [ ] TNNet.SaliencyReport — given a trained classifier and a probe sample,
      compute three flavours of input attribution and print them side-by-side
      as compact ASCII heatmaps over the input plane (one row per channel,
      one cell per pixel, 10 intensity buckets):
      (a) vanilla input-gradient saliency `|d logit_c / d x|` for the
          predicted class c (one forward + one backward pass),
      (b) SmoothGrad over N noisy copies of the input
          (`x + eta`, `eta ~ N(0, sigma^2)`, default N=16, sigma=0.15 *
          (max(x)-min(x))) — averages (a) to denoise the saliency,
      (c) Integrated Gradients along a straight line from a zero baseline
          to x using K steps (default K=20) — the "did this pixel
          contribute to the decision?" measure that satisfies the
          completeness axiom (sum of attributions ≈ logit_c(x) - logit_c(0)).
      Also report, per channel: total attribution mass, top-K most-attributing
      pixel coordinates, and the completeness gap for the IG variant as a
      one-number sanity check. Pure-CPU, reuses the existing
      forward/backward path — no training-time changes, no new layer types
      (the input-gradient already flows through TNNetInput's backward).
      Distinct from [[DeadNeuronReport]] / [[ActivationStatsReport]]
      (activation statistics, not input attribution),
      [[GradientNormReport]] (per-layer backward magnitudes summarised, not
      per-pixel input gradients), [[AttentionEntropyReport]] (attention
      weights inside an SDPA stack, not classifier-input attribution), and
      [[ConfusionMatrixReport]] (aggregate label confusions, not per-sample
      explanations). Companion `examples/SaliencyReport/` runs it on a
      handful of SimpleImageClassifier CIFAR samples — one correctly
      classified and one misclassified — so the three heatmaps can be
      eyeballed against the actual image content, and the completeness gap
      acts as a built-in regression check on the IG implementation.

### Activation distribution
- [ ] TNNet.ActivationStatsReport — given a probe batch, walk every layer's
      forward output and print a per-layer table of
      `mean / std / min / max / |median| / |skew| / kurtosis` plus
      `pct_saturated_low` and `pct_saturated_high` (configurable thresholds,
      default ±0.99·OutputRange for bounded activations, |x|>6 for unbounded),
      `pct_negative`, `pct_near_zero` (|x| < 1e-6), and a compact 16-bin ASCII
      histogram over `[-MaxAbs, +MaxAbs]`. End with a flag list:
      "near-collapsed layers" (std < 1e-4), "saturating layers" (>50%
      saturated either side), and a 10-bin ASCII histogram of per-layer std
      across the network so vanishing/exploding activation patterns jump out
      at a glance. Pure forward-only — no training-time changes. Distinct
      from [[DeadNeuronReport]] (zero-fraction on ReLU-family layers only),
      [[WeightHistogramReport]] (weights, not activations),
      [[GradientNormReport]] (backward magnitudes), and
      [[AttentionEntropyReport]] (attention-weights only). Companion
      `examples/ActivationStatsReport/` runs it on (i) a fresh-init network
      and (ii) the same architecture after a short training run, so reviewers
      can eyeball how training reshapes the activation distribution.

### Memory footprint
- [ ] TNNet.MemoryFootprintReport — given a network (and implicitly the
      configured input shape from `TNNetInput`), walk every layer and report:
      (a) per-layer activation tensor size in elements and MiB
          (`SizeX * SizeY * Depth * sizeof(TNeuralFloat)`),
      (b) per-layer parameter tensor size in elements and MiB (weights +
          biases, zero for parameter-free layers),
      (c) per-layer error/gradient tensor size in MiB (mirrors activation;
          flagged as "transient — recoverable via checkpointing"),
      (d) running totals plus the "peak forward residency" (sum of
          activations that must be kept alive for backward) and the
          "parameters + optimizer-state" baseline (1x for SGD, 2x for
          momentum, 3x for Adam — configurable via a `OptimizerKind` flag),
      (e) a 10-bin ASCII histogram of per-layer activation MiB so the
          memory-hot layers jump out at a glance,
      (f) a flag list: "activation-heavy" layers (>10% of the activation
          total — natural gradient-checkpointing candidates) and
          "parameter-heavy" layers (>10% of the parameter total — natural
          LoRA / quantization candidates),
      (g) a one-line "would-fit-in" verdict against a configurable budget
          (default: 2 GiB) for `forward-only`, `train-SGD`, `train-Adam`.
      Pure structure inspection — no probe batch, no forward pass needed
      (sizes come from the existing `Output.Size` / weight metadata).
      Distinct from [[FLOPsReport]] (compute, not memory),
      [[WeightHistogramReport]] (weight value distribution, not byte
      footprint), [[WeightSpectrumReport]] (weight geometry), and the
      already-listed `Gradient checkpointing` infrastructure task (this
      one *measures*; that one *acts*). The output is the natural input
      for any future checkpointing or mixed-precision work — you need
      to know which layers cost what before you can decide where to
      trade compute for memory. Companion `examples/MemoryFootprintReport/`
      runs it on (i) a tiny MLP, (ii) a small CIFAR conv stack, and
      (iii) a small attention stack so reviewers can eyeball how the
      shape of the bottleneck shifts across model families (parameter-
      heavy MLP vs activation-heavy conv vs both-heavy attention).

### Weight-matrix spectrum
- [ ] TNNet.WeightSpectrumReport — for every trainable layer in a network,
      estimate the top singular value `sigma_1(W)` of its weight matrix via
      a handful of power-iteration steps (default 10), and report per layer:
      (a) `sigma_1`,
      (b) `||W||_F` (Frobenius norm, cheap exact),
      (c) the ratio `sigma_1 / ||W||_F` — a stable-rank-flavoured signal
          where values near 1 hint at rank-1 collapse (one direction
          dominates) and values near `1/sqrt(min(in, out))` hint at a
          well-spread spectrum,
      (d) `sigma_1` divided by a Marchenko-Pastur baseline
          `(sqrt(in) + sqrt(out)) * std(W)` — a one-number "is this layer's
          top mode larger than what a Gaussian init of matching std would
          produce?" check,
      (e) a 10-bin ASCII histogram of per-layer `sigma_1 / fan_in_baseline`
          across the network,
      (f) a flag list: "spectral-norm > threshold" layers (Lipschitz risk)
          and "stable-rank ≈ 1" layers (representation collapse risk).
      Pure forward-only on the weight tensors — no training-time changes,
      no probe batch needed (a probe-batch variant could land later as
      `JacobianSpectrumReport` if it proves worth the extra knob).
      Distinct from [[WeightDriftReport]] (deltas across training, not a
      snapshot), [[GradientNormReport]] (backward magnitudes, not weight
      geometry), and [[LossLandscapeProbe]] (forward loss along a random
      direction, not weight spectrum). The shared spectral-norm helper
      this would introduce is reusable by the already-listed
      [[TNNetSpectralNorm]] wrapper task. Companion
      `examples/WeightSpectrumReport/` runs it on (i) a freshly-initialised
      net (baseline) and (ii) the same architecture after a short training
      run, so reviewers can eyeball how training pushes the spectrum away
      from the init baseline.
