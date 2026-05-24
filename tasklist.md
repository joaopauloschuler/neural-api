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
<!-- (Two TNNetFourierFeatures spectral-bias follow-ups removed: completed,
     landed in examples/FourierFeaturesSpectralBias/.) -->

## Interesting applications / examples
- [ ] Reinforcement learning: minimal DQN solving CartPole or a grid world
- [ ] Style transfer or diffusion-lite denoiser (building on SuperResolution / VisualGAN)
- [ ] Early-exit / adaptive-inference demo (`examples/EarlyExitNetwork/`) — the
      BranchyNet (Teerapittayanon et al. 2016) "anytime inference" pattern, which
      nothing in the tree covers yet. Build ONE trunk of stacked FC+ReLU (or conv)
      blocks with an AUXILIARY softmax classifier head branching off after each
      intermediate block as well as the final block, on a synthetic difficulty-
      graded N-class task (mix easy well-separated blobs with hard near-the-margin
      points). Train ALL heads JOINTLY by summing their cross-entropy losses (deep
      supervision via existing `TNNetConcat`/split-free parallel heads + a manual
      multi-head loss loop, no new layer type needed). Then at INFERENCE run a
      confidence-gated dynamic-compute policy: walk heads shallow→deep and EXIT at
      the first head whose softmax max-prob exceeds a threshold tau, recording the
      exit depth per sample. Sweep tau and chart the accuracy-vs-average-exit-depth
      (i.e. accuracy vs FLOPs saved, reusing `TNNet.CountFLOPsPerLayer`) trade-off
      curve as an ASCII plot, showing easy samples leave early while hard ones run
      the full depth. Two built-in invariants: tau=1.0 forces every sample to the
      final head (== the plain full-depth net's accuracy, bit-for-bit) and average
      exit depth is monotone non-decreasing in tau. This is DISTINCT from the
      existing PredictionDepth example (a post-hoc k-NN probe on a FIXED single-head
      net measuring where it "makes up its mind") — here the early heads are TRAINED
      and actually gate compute. README should spell out that contrast explicitly.
<!-- (Activation-steering / concept-vector demo removed: completed, landed
     2026-05-24 as examples/ActivationSteering/. Trains a small softmax
     classifier on a synthetic sign(x0) two-cluster task, computes the
     diff-of-class-means steering vector v at hidden layer k=2, and sweeps
     alpha in {-3..+3} injecting Output_k.MulAdd(alpha, v) + downstream recompute
     (reuses the ActivationPatchingReport CopyNoChecks-then-recompute pattern).
     All three checks PASS: alpha=0 reproduces the unsteered forward bit-for-bit,
     P(target) is monotonic in alpha, and v shifts the output ~1.4x more per unit
     norm than an equal-norm random direction (measured as mean |dP| across the
     sweep — the max-swing variant nearly ties because both saturate at alpha=3).
     Possible follow-ups: a MULTI-LAYER steering sweep (which layer k gives the
     cleanest monotone control?), and steering toward a class the input is NOT,
     charting the flip threshold.) -->
<!-- (ActivationSteering depth-sweep follow-up removed: completed, landed
     2026-05-24 as examples/ActivationSteeringDepthSweep/. Grows the net to four
     steerable FC+ReLU hidden layers and, for each k, computes the diff-of-class-
     means v_k, sweeps alpha in {-3..+3} (downstream recompute), and prints the
     P(target)-vs-alpha curve + a monotonicity up-fraction + alpha-to-flip per
     layer, then a summary naming the cleanest-monotone layer. Finding: depth
     MATTERS — k=1/3/4 are perfectly monotone but k=2's diff-of-means direction
     is genuinely NON-monotone (up-fraction 0.5, pushes P(target) the wrong way
     for alpha<0); on the coarse grid every layer flips by alpha=1 so the
     monotonicity measure (not alpha-to-flip) is what discriminates. The alpha=0
     bit-for-bit no-op check passes at EVERY k. Open follow-up: a finer alpha
     grid would break the alpha-to-flip tie.) -->

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
<!-- (TNNetLinearAttention — kernelized / softmax-free non-causal attention
     (Katharopoulos et al. 2020) removed: completed. Landed as a standalone
     TNNetLayer descendant in neuralnetwork.pas reusing SDPA's Q|K|V
     input-split: phi(x)=elu(x)+1 feature map, accumulate S = sum_s
     phi(K_s)(x)V_s and Z = sum_s phi(K_s) once, then Out_t = (phi(Q_t).S)/
     (phi(Q_t).Z) — O(SeqLen*d_k*d_v), no NxN score matrix. Registered in
     both CreateLayer dispatch + LoadFromString. Tests in
     TestNeuralNumerical.pas: input numerical-gradient check (d_k=4,SeqLen=3),
     SeqLen=1 degeneracy (Out_1==V_1), serialization round-trip. Wall-clock
     scaling probe in examples/LinearAttention/ confirms ~linear (ratio ~2x
     per SeqLen doubling, not ~4x). The CAUSAL variant remains OPEN — see
     the follow-up below.) -->
<!-- (TNNetLinearAttention CAUSAL follow-up removed: completed, landed
     2026-05-24 as TNNetCausalLinearAttention in neuralnetwork.pas. The causal
     (autoregressive) variant of the non-causal TNNetLinearAttention: carries
     the running prefix-sum S_t = sum_{s<=t} phi(K_s)(x)V_s and Z_t = sum_{s<=t}
     phi(K_s) (the "attention is an RNN" identity) so each query t reads its own
     causal S_t/Z_t — still O(SeqLen*d_k*d_v) with no NxN matrix. Registered in
     both CreateLayer dispatch + LoadFromString. Tests in TestNeuralNumerical.pas:
     input numerical-gradient check (TestCausalLinearAttentionGradientCheck),
     a causality / no-future-leak + SeqLen=1 degeneracy (Out_1==V_1) check
     (TestCausalLinearAttentionCausality), and a serialization round-trip
     (TestCausalLinearAttentionSerializationRoundTrip). Still-open downstream
     work lives in its own entries: the "softmax vs linear attention quality-
     vs-cost" bake-off and folding linear attention into the MHA breakdown
     ([[TNNetMultiHeadSelfAttention]]) as a per-head opt-in.) -->

- [ ] TNNetDifferentialAttention follow-up: the paper's headline
      NOISE-CANCELLATION micro-experiment (deferred at landing — only the four
      correctness tests shipped). On a tiny causal next-token task with an
      "all-keys-irrelevant" probe row, compare plain SDPA vs
      TNNetDifferentialAttention and print the attention-noise mass on the
      irrelevant keys — the differential output should land strictly below
      plain SDPA's. ~30-line wiring swap, mirrors the open SinkAttention
      stability micro-experiment; all pieces now in tree.
- [ ] TNNetDifferentialAttention follow-up: fold differential heads into the
      MHA breakdown ([[TNNetMultiHeadSelfAttention]] /
      TNNetTransformerDecoderBlock) behind a flag, so a decoder block can opt
      into differential attention per head — a natural drop-in for the
      downstream ../gpt-3-for-pascal long-context retrieval.
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
<!-- (Position-encoding bake-off removed: completed, landed in
     examples/PositionEncodingBakeoff/.) -->
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
<!-- (PreNorm / RMSNorm / PostNorm residual builders removed: completed,
     landed as TNNet.AddPreNormResidual / AddRMSNormResidual /
     AddPostNormResidual.) -->
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
- [ ] TNNetTokenHistoryPenalty follow-up: wire it into the downstream
      ../gpt-3-for-pascal generation loop (call `Apply` before the sampler
      and `RegisterToken` after each emit, `ResetHistory` per sequence) and
      show a qualitative before/after on a repetition-prone prompt — the
      class landed this lucky-day batch (neuralvolume.pas, 7 tests in
      tests/TestNeuralSamplers.pas) but no in-tree generator calls it yet.
<!-- (TNNetGatedResidual / TNNet.AddGatedResidual builder removed: completed,
     landed 2026-05-24 as a sibling to AddPreNormResidual/AddRMSNormResidual/
     AddPostNormResidual (per-channel gate inits to 0 so the branch starts as
     identity); ships forward-wiring + input numerical-gradient tests. The
     ReZero-vs-GatedResidual depth-ablation follow-up landed below.) -->
<!-- (ReZero-vs-GatedResidual depth ablation removed: completed, landed
     2026-05-24 as examples/ReZeroVsGatedResidual/. Finding: the per-channel
     gate opens UNEVENLY (many channels stay exactly 0, a handful grow both
     signs) while the ReZero scalar opens uniformly. Gotcha for future
     gate-dump examples: TNeuralFit.Fit reloads the best model at the end
     (rebuilding every layer instance) so gate-layer LAYER REFERENCES go
     stale — capture layer INDICES at build time and read
     NN.Layers[idx].Neurons[0].Weights after Fit.) -->
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
<!-- (TNNetMishExact / TNNetMish-stable removed: the in-tree TNNetMish ALREADY
     implements the stable formulation — its Compute() branches on x>20
     (softplus≈x) and x<-20 (softplus≈exp(x)≈0) to avoid the exp overflow.
     A separate "stable Mish" class would be a forward-pass duplicate of
     TNNetMish, which the "DO NOT REINTRODUCE" policy at the top of this file
     forbids. Verified 2026-05-24 lucky-day batch.) -->
<!-- (TNNetSplineActivation KAN-vs-MLP toy-fit removed: completed, landed
     2026-05-24 as examples/SplineActivationKAN/ — matched-param (21 vs 20
     weights via TNNet.CountWeights), spline arm reaches MSE 0.061 vs ReLU 0.187
     (67.5% lower); dumps learned control points + sampled activation showing one
     channel bent away from the identity (a sparse KAN fit). Open follow-ups
     remain below: the knot-count/Range sweep and the TNNetAPL bake-off.) -->
- [ ] TNNetSplineActivation follow-up: knot-count / Range sweep — same toy fit
      with K ∈ {2, 4, 8, 16} and a couple of Range values, charting the
      capacity↔overfitting trade and where extra knots stop helping.
- [ ] TNNetMetaAconC follow-up: the FULL cross-channel-bottleneck β generator
      (the paper's true Meta-ACON: squeeze → FC channel-reduce → ReLU → FC
      channel-expand → sigmoid, so β[c] depends on ALL channels' spatial
      means, not just channel c's). The landed TNNetMetaAconC uses a per-channel
      affine-over-squeeze simplification; this variant needs a small two-FC
      sub-block inside the layer (or a builder that wires an SE-style squeeze
      into the β path) and is NOT a per-channel-transform shape, so scope it as
      its own layer/builder rather than a ChannelTransformBase descendant.
- [ ] TNNetBitLinear follow-up: a ternary-vs-full-precision bake-off — train the
      same tiny classifier with `TNNetFullConnectLinear` heads vs `TNNetBitLinear`
      heads at matched architecture, report final accuracy/loss AND the effective
      model size (ternary weights are ~1.58 bits each vs 32). The headline BitNet
      claim is "near-FP accuracy at a fraction of the weight memory"; this is a
      ~30-line head swap. Pairs with the STE bit-width sweep and the
      TNNetStraightThroughEstimator quantization demos already in the list.
- [ ] TNNetBitLinear follow-up: activation-quantization variant — BitNet b1.58
      also quantizes the *activations* to int8 (absmax per-token). Consider a flag
      or sibling that rounds the layer INPUT through an absmax STE before the
      ternary matmul, so the "fully-quantized linear" path is reachable. Scope as
      its own flag on TNNetBitLinear (forward adds an input absmax-round; backward
      STE-passes the input gradient unchanged).
<!-- (TNNetMaxOut2 removed: a "two-piece special case of TNNetMaxOut" is just
     TNNetMaxOut.Create(2) with a thinner constructor — i.e. a forward-pass
     duplicate of the existing TNNetMaxOut, which the "DO NOT REINTRODUCE"
     policy at the top of this file forbids. If a no-group-count convenience
     constructor is genuinely wanted, add an overload to TNNetMaxOut instead of
     a new class. Flagged 2026-05-24 lucky-day batch.) -->
- [ ] TNNetAPL follow-up: APL-vs-PReLU-vs-ReLU bake-off on the hypotenuse toy
      (or a tiny CIFAR stub) at matched param count, sweeping the hinge count
      S ∈ {1, 2, 4} — does the extra piecewise capacity buy lower final loss?
      The activation has landed, so this is a ~30-line activation swap.
#### Probability projections / sparsity
- [ ] TNNetGumbelSoftmax follow-up (now landed): temperature-annealing
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
<!-- (TNNetUnitNormConstraint removed: completed, landed as TNNetWeightNormLinear
     — a differentiable unit-L2 weight reparametrization (the simple g=1 form of
     Weight Normalization). Each output neuron's weight vector is L2-normalized to
     unit norm inside the forward pass (`ŵ = w/sqrt(Σwᵢ² + eps)`) and the exact
     unit-norm Jacobian is backpropagated to the raw weights (gradient-checked).
     Modelled exactly on TNNetWeightStandardization. The hard-projection variant
     below is the only open piece.) -->
- [ ] TNNetUnitNormConstraint hard-projection variant: a true *post-step hard
      projection* (renormalize the previous layer's weights after each update,
      non-differentiable) — still open if a hard constraint is ever wanted. The
      differentiable reparametrization (TNNetWeightNormLinear, landed) already
      covers the headline use case.

#### Reduction / shape
<!-- (TNNetAdaptiveAvgPool example/usage removed: completed, landed 2026-05-24 as
     examples/AdaptivePoolResolution/ — one conv stack fed 16x16 and 24x24, fixed
     1x1 and 2x2 adaptive heads, with built-in Create(1)==global and
     Create(N)==identity degeneracy assertions and a train-at-16/infer-at-24
     sanity step. Exercises TNNetAdaptiveMaxPool in the same demo.) -->
- [ ] TNNetGather — single-channel index-into-a-channel layer.
- [ ] TNNetUpsampleNearest backward consistency: assert summing the
      per-block output errors equals the input error.
<!-- (Pooling bake-off example removed: completed, landed 2026-05-24 as
     examples/PoolingBakeoff/ — same tiny conv classifier swapping the pooling
     head across TNNetAvgPool / TNNetMaxPool / TNNetLpPool (p ∈ {1,2,4,8}) /
     TNNetSoftPool (beta ∈ {0.5,1,2,8}) on a synthetic blob-quadrant task where
     the class-mean is invariant and only energy CONCENTRATION is
     discriminative: AvgPool/LpPool(p=1) sit at chance, MaxPool solves it, and
     both LpPool's p and SoftPool's beta interpolate avg→max.) -->
<!-- (TNNetAdaptiveMaxPool example/usage removed: completed — landed in the same
     examples/AdaptivePoolResolution/ demo as the AdaptiveAvgPool entry above;
     both adaptive layers are exercised in one demo.) -->
### Loss layers
- [ ] TNNetCosineEmbeddingLoss follow-up (now landed): a tiny
      siamese-pair embedding micro-example — train two shared-weight MLP
      branches whose outputs are concatenated into the `a|b|y` layout, on a
      synthetic "same vs different class" pair task, and print the learned
      same-pair vs different-pair cosine histograms. Headline use case for
      the landed head; pairs with [[TripletEmbedding]].
- [ ] TNNetKLDivergence distillation follow-up (now landed
      examples/KnowledgeDistillation/): temperature sweep T in {1,2,4,8} on this
      example — chart how soft-target sharpness changes the distilled student's
      accuracy/agreement.
- [ ] Tversky α/β asymmetry sweep on the segmentation micro-example: with a
      deliberately class-imbalanced mask, sweep `(α,β) ∈ {(0.5,0.5),(0.3,0.7),
      (0.7,0.3)}` and show how β>α trades precision for recall (fewer false
      negatives). Pure α/β knob study on the landed TNNetTverskyLoss.
      NOW UNBLOCKED: fork examples/DiceSegmentation/ (landed above) and swap the
      head for TNNetTverskyLoss with the three (α,β) pairs.
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
<!-- (Sharpness-Aware Minimization (SAM) experiment removed: completed, landed
     2026-05-24 as examples/SharpnessAwareMinimization/ — hand-rolled two-pass
     SAM (ascent perturb via global grad-norm, snapshot/restore via
     SaveDataToString) vs plain SGD on a noisy-label 2D-blob toy. Both
     invariants hold: rho=0 == plain SGD bit-for-bit (max weight diff 0.0), and
     sharpness (LossLandscapeProbe) falls as rho rises (0.41→0.21). KEY LIBRARY
     NOTE for any future manual gradient-surgery example (PCGrad, Lookahead,
     grad clipping): the library defaults to per-sample updates
     (FBatchUpdate=false) where Backpropagate applies updates immediately and
     leaves Neurons[].Delta at ZERO — call NN.SetBatchUpdate(True) to access the
     accumulated gradient tensor or the access is a silent no-op.) -->
- [ ] SAM follow-up: the noisy-label 2D-blob clusters are easily separable so
      clean val-accuracy saturates (~99%) across all rho — the flatness signal
      carries the story but the val-acc-vs-rho curve is flat. A harder task
      (overlapping clusters / higher label-noise / a tiny MLP on a small image
      stub) where SAM's flat minimum actually buys measurable val-accuracy over
      plain SGD would complete the demonstration. Builds directly on the landed
      examples/SharpnessAwareMinimization/.
### Introspection / debugging tools
<!-- (TNNet.ToGraphvizDot removed: completed, landed 2026-05-24 — instance
     method returning a `digraph` string for the layer DAG (node per layer
     with index/class/output-shape, edges from the real DAG incl. multi-input
     TNNetConcatBase layers). Ships examples/GraphvizExport/ +
     TestToGraphvizDotSmoke. The sibling WriteLayerTimings(NN, Sample)
     follow-up remains open below.) -->
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
      BatchNorm / LayerNorm / RMSNorm / GroupNorm / InstanceNorm /
      ChannelStdNorm.
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
<!-- (`examples/PreNormVsPostNorm/` removed: completed, landed 2026-05-24 —
     a 12-block residual MLP wired three ways via AddPreNormResidual/
     AddRMSNormResidual/AddPostNormResidual on the synthetic hypotenuse task,
     surfacing the pre-norm vs post-norm stability gap. Open follow-ups (push
     NUM_BLOCKS/LR to full-NaN divergence; an AddGatedResidual fourth arm)
     remain worth adding.) -->
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
<!-- (InstanceNorm vs GroupNorm vs LayerNorm vs ChannelStdNorm single-seed
     bake-off removed: subsumed by the `examples/NormalizationBakeoff/` entry
     under "Examples I'd enjoy writing", which compares a strict superset of
     norms on a small conv stack. ChannelStdNorm — the only norm unique to this
     entry — was folded into that example's menu so nothing is lost.) -->
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
<!-- (Charbonnier-vs-Huber-vs-MSE-vs-LogCosh head-to-head removed: subsumed by
     the "Loss-family bake-off (output heads)" entry just below — its loss set
     {Charbonnier, Huber, MSE, LogCosh} is a strict subset of that entry's
     {MSE, Huber, SmoothL1, Charbonnier, LogCosh}, on the same noisy/outlier
     hypotenuse harness.) -->
- [ ] Loss-family bake-off (output heads): hypotenuse with MSE / Huber /
      SmoothL1 / Charbonnier / LogCosh, printing final MSE and epochs-to-
      converge.
<!-- (TanhGLU vs SwiGLU vs GEGLU vs GLU vs ReGLU bake-off removed: completed,
     landed 2026-05-24 as examples/GatedFFNBakeoff/. Same FFN block
     (PointwiseConvLinear(2*d_ff) -> GATE -> PointwiseConvLinear(1)) built five
     times, identical except the parameter-free gate (so all arms have identical
     param counts), trained at matched seed/LR/epochs on a synthetic per-position
     sequence-regression target. Prints init/final MSE + wall-clock + epochs-to-
     converge per arm and two PASS checks (no NaN/Inf; every arm beats its
     pre-training baseline). Ranking is seed-dependent by design — the value is
     the matched harness. Open follow-up: a multi-seed (mean ± std) version so a
     statistically meaningful ranking is reportable.) -->
- [ ] GatedFFNBakeoff follow-up: a multi-seed (e.g. 5 seeds, mean ± std)
      variant of the landed examples/GatedFFNBakeoff/ so the gate ranking is
      statistically meaningful rather than a single-seed snapshot. Keep the
      per-arm dims tiny so 5 seeds x 5 gates still fits the <5-min budget.
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
<!-- (Double-descent demo removed: completed, landed 2026-05-24 as
     examples/DoubleDescent/. Existing layers only, no new layer. Fixed
     nonlinear teacher class=sign(x'Qx+b'x) over D=4; SMALL train=60 with 15%
     of labels FLIPPED, LARGE clean test=2000. SAME single-hidden-layer MLP
     Input(4)->TNNetFullConnectReLU(H)->TNNetFullConnectLinear(1) trained as an
     MSE regression onto the +-1 target (MSE interpolates the noisy set cleanly
     where a saturating softmax/cross-entropy head left ~2 hard noisy points
     permanently misclassified and never reached train-err 0), full-batch GD
     (SetBatchUpdate(True) + ClearDeltas/accumulate/UpdateWeights per epoch),
     LR=0.03 mom=0.9 up to 6000 epochs w/ early-stop at train MSE~0. Width
     sweep H in {1,2,3,4,5,6,8,12,20,32,64,128}, param count via
     TNNet.CountWeights printed. Two-arm ablation (noise ON vs OFF, same
     teacher/points/seeds). RESULT (seed 20260524, all 3 checks PASS, ~57s):
     noisy test err 0.39(underfit,H=1) -> 0.21 VALLEY(H=6) -> 0.38 PEAK(H=8,
     right at the interpolation threshold where train MSE collapses 0.27->0.027
     by H=12 and train 0/1 err first hits 0) -> 0.34(over-param,H=128) — the
     textbook non-monotone U-then-peak-then-down. Clean arm ~monotone
     (post-min rise 0.017 vs noisy 0.172), confirming the peak is NOISE-driven.
     Three built-in signals: interpolation-threshold-exists, peak-at/around-
     threshold (peak is the max test err AT/AFTER the bias-variance VALLEY, not
     the global argmax — the underfit left edge is also high), and the
     noise-on/off ablation. CAVEATS: (1) softmax/cross-entropy classification
     did NOT cleanly interpolate at this scale (stuck at ~2/60 train err across
     all widths) — switched to MSE-regression-on-+-1; (2) at this single seed
     the clean arm shows one transient 0.262 wobble at H=12, a seed
     fluctuation, not a systematic peak — the tested signal (aggregate post-min
     rise) is 10x smaller than the noisy arm's; (3) the peak HEIGHT and exact
     threshold are seed-dependent, only the noise-gated SHAPE is the robust
     claim. Suite stays green at 793. -->
- [ ] Lottery-ticket / magnitude-pruning follow-up to double descent: the
      over-parameterised models on the RIGHT arm of examples/DoubleDescent are
      the compressible regime — prune the H=128 interpolating net by weight
      magnitude and show it keeps the low test error down to a small fraction
      of its weights. Pairs with [[WeightSpectrumReport]] /
      [[WeightHistogramReport]] (watch the weight-norm spike at the
      interpolation threshold).
<!-- (Toy-models-of-superposition demo removed: completed, landed 2026-05-24 as
     examples/Superposition/. Importance-weighted-MSE autoencoder
     Input(N=20) -> TNNetFullConnectLinear(M=5){encoder} -> TNNetFullConnectReLU(N){decoder}
     (existing layers only, no new layer), hand-rolled batch training with the
     weighted-MSE gradient delivered through the stock Backpropagate via a
     pseudo-target (pseudo_i = out_i - (I_i/batch)*(out_i - in_i), so the summed
     batch error becomes the MEAN weighted gradient — the /batch is what keeps
     the dense regime from diverging). Reads the effective UNTIED pre-ReLU map
     G = D*W (N×N): column-norm = represented norm, off-diag = interference;
     reports per-feature norm ASCII bars, the glyph-shaded G heatmap,
     superposition ratio = kept/M, and mean |off-diag|. Sweep
     S ∈ {0.0,0.7,0.9,0.99} shows the textbook transition: ratio 1.00 (dense:
     clean M×M diagonal, monosemantic) -> 1.80 -> 3.80 (sparse: 19/20 features
     packed into 5 dims). All three built-in correctness signals PASS (dense
     kept≈M and near-diagonal; kept features higher mean-importance than dropped;
     represented-feature count grows with S). ~2 min, pure CPU, deterministic,
     suite stays green at 793. CAVEAT: the marginal kept-feature identity at
     intermediate sparsities is mildly noisy (untied ReLU geometry, near-tied
     importances), so signal (2) uses a robust mean-importance criterion rather
     than a strict feature-order prefix test, and "total represented norm" is
     reported only as a diagnostic (raw G magnitudes are not comparable across S)
     with represented-feature COUNT as the growth axis. -->
- [ ] Sparse-autoencoder feature extraction (`examples/SparseAutoencoder/`) —
      the mechanistic-interpretability SOLUTION companion to the landed
      examples/Superposition/ (which shows the PROBLEM: a model packing N
      ground-truth features into M<N dimensions). Reproduce the headline of
      Anthropic's *Towards Monosemanticity* (Bricken et al. 2023) on a pure-CPU
      toy, existing layers only — NO new layer. Recipe: generate synthetic data
      where K sparse "ground-truth" features (one-hot-ish, active with low
      probability, random magnitudes) are linearly mixed down into a DENSE
      d-dim activation vector `a = G·f` with d < K (the superposition regime,
      reusing the Superposition example's data generator). Then train an
      OVERCOMPLETE sparse autoencoder ON those dense activations —
      `Input(d) -> TNNetFullConnectReLU(H) {H >> d, the dictionary}
      -> TNNetFullConnectLinear(d) {tied-or-untied decoder}` — with an L1
      sparsity penalty on the hidden code (delivered through the stock
      Backpropagate via the pseudo-target trick the Superposition example
      already uses: `pseudo_i = out_i - lr_term`, plus an L1 sub-gradient
      `lambda*sign(hidden)` injected into the hidden layer's OutputError, the
      same manual-gradient-surgery idiom as the SAM / Superposition examples —
      remember SetBatchUpdate(True)). The payoff: after training, MATCH each
      learned dictionary atom (a decoder column / hidden unit) to its
      best ground-truth feature by max cosine similarity and report (a) a
      mono-semanticity score — mean max-cosine of recovered atoms to true
      features (should climb well above the raw activation-to-feature cosine
      baseline), (b) the recovered-feature COUNT vs the L1 weight `lambda`
      swept over `{0, 1e-3, 1e-2, 1e-1}` (too little -> dense polysemantic
      atoms, too much -> dead atoms; a clear interior sweet spot), (c) the mean
      L0 (active atoms per sample) vs reconstruction MSE trade curve, and (d)
      an ASCII atom-to-feature confusion/cosine heatmap so the
      "one atom == one interpretable feature" recovery is visible against the
      tangled raw activations. Built-in correctness signals: at lambda=0 the SAE
      is a plain (over-parameterised) autoencoder and recon MSE ~0 but atoms are
      polysemantic (low max-cosine); the swept curves must be NON-monotone with
      an interior peak in recovered-feature count; and a "dead atom" fraction
      reported per lambda. DISTINCT from examples/Superposition/ (that reads the
      model's OWN encoder geometry G=D·W to MEASURE packing; this trains a
      SEPARATE overcomplete dictionary to UNPACK it), from the open "TopK
      sparsity sweep" entry (an UNDERCOMPLETE bottleneck recon-loss-vs-K study,
      no ground-truth-feature recovery, no dictionary) and from the
      "Shrink-activation sparsity sweep" (bottleneck activation choice, not
      dictionary learning). Pure CPU, deterministic, a-few-minutes budget;
      pairs with [[WeightHistogramReport]] (dead-atom column norms) and the
      open TNNetTopK sparsity work (a hard-TopK SAE variant is the natural
      follow-up once the L1 version lands).
- [ ] Edge-of-Stability demo (`examples/EdgeOfStability/`) — reproduce the
      "progressive sharpening" + "edge of stability" phenomenon (Cohen et al.
      2021, *Gradient Descent on Neural Networks Typically Occurs at the Edge
      of Stability*) on a pure-CPU toy. The headline, which NO in-tree
      experiment shows: under plain full-batch gradient descent at a FIXED
      learning rate `eta`, the top Hessian eigenvalue `lambda_max` (the
      sharpness) RISES throughout early training ("progressive sharpening")
      until it reaches `2/eta` — the classical GD stability limit — and then
      HOVERS just above that threshold for the rest of training while the loss
      keeps falling NON-monotonically (small ripples), instead of diverging as
      the textbook quadratic-bowl analysis predicts. Recipe, reusing the
      already-landed sharpness machinery — NO new layer or report needed: train
      a tiny MLP (e.g. `FullConnectReLU -> FullConnectLinear`) on a small
      synthetic regression/classification batch with FULL-batch plain SGD (not
      Adam, not mini-batch — the EoS story is specific to deterministic GD) at a
      fixed `eta`, and every K steps call `TNNet.HessianCurvatureReport` over a
      fixed probe batch to read off `lambda_max` (its power-iteration-on-HVP
      `lambda_max` is exactly the number we need — the report already estimates
      it without forming the Hessian). Chart a two-row ASCII time series of
      `lambda_max` and the constant `2/eta` line over training steps, plus the
      loss curve, and flag the "EoS entry step" (first step where `lambda_max`
      first crosses `2/eta` and stays within a band of it thereafter). Sweep
      `eta in {small, medium, large}` to show the punchline: the plateau height
      TRACKS `2/eta` (smaller `eta` -> the net is allowed to get sharper before
      stalling), the cleanest single demonstration that the optimizer's own
      stability limit — not the data — caps the curvature it settles at.
      Built-in correctness signals: `lambda_max` must rise before it plateaus
      (no progressive sharpening at init means the probe/eta is off); the
      plateau must sit at/just above `2/eta` (well below it = not yet at the
      edge, far above = diverging); and the loss must still trend down across
      the plateau despite the ripples. DISTINCT from the landed
      `examples/HessianCurvature/` (a STATIC flat-vs-sharp contrast of two
      ALREADY-trained minima — no time axis, no `2/eta` threshold, no eta
      sweep), from the SAM example (which MINIMISES sharpness via a perturbed
      gradient — here sharpness is left to evolve under plain GD and merely
      observed), from the open grokking demo (delayed GENERALISATION over time,
      a val-accuracy axis, not a curvature-vs-stability-limit axis) and from
      double-descent (a CAPACITY axis). Pure CPU, no external data, fits a
      few-minute budget at tiny width/sample-count. Pairs naturally with
      [[LossLandscapeProbe]] (its scalar "sharpness" should rise in lockstep
      with `lambda_max`) and the SAM example (contrast: SAM bends the plateau
      down). KEY LIBRARY NOTE for whoever builds it: use full-batch GD with
      `MaxThreadNum := 1` for a deterministic sharpening curve, and remember the
      report is a pure measurement — it never steps the weights, so interleaving
      it inside the training loop is safe.
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
<!-- ("Activation cost" microbenchmark removed: duplicate of the per-activation
     forward/backward ns/op CSV in `examples/ActivationPlayground/` and of
     `bin/layer_bench <Activation> 64 64 32` under "Tooling / dev experience",
     both of which already time forward+backward per activation/layer at a
     chosen shape.) -->
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
<!-- (TNNetHardShrink / TNNetSoftShrink sparsity micro-experiment removed:
     subsumed by the "Shrink-activation sparsity sweep" entry under
     "### Experiments I'm curious about (additional)", which trains the SAME
     tiny autoencoder with each shrink activation as the bottleneck and reports
     sparsity vs reconstruction loss — and additionally adds a ReLU baseline and
     a lambda sweep, so nothing is lost.) -->

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
<!-- (TNNet.PermutationAlignReport removed: completed, landed 2026-05-24 via the
     [[introspection-report-pattern]] — the Git Re-Basin DUAL of
     ModeConnectivityReport. Signature
     `PermutationAlignReport(NN; const SnapshotB; Samples: TNNetVolumePairList;
     ScoreMode: integer = 0; K: integer = 10)`. Catalogues trainable layers,
     greedily solves each hidden layer's best B->A unit permutation (ScoreMode=0
     weight-row cosine / ScoreMode=1 activation correlation over Samples), applies
     it via ReorderLayerOutputNeurons + PermuteNextLayerInputColumns
     (next-layer input-column compensation so the function is unchanged), reuses
     ModeConnectivity's MulMulAdd (1-alpha)*A+alpha*B sweep for pre/post barriers,
     restores endpoint-A bit-for-bit in try/finally, and emits the three PASS/FAIL
     checks + collapsed/partially-reduced/unchanged verdict. Ships
     examples/PermutationAlign/ + TestPermutationAlignReportSmoke. All checks PASS:
     weight-matching barrier 0.0957->0.0332 (65% reduction), activation-matching
     0.0957->0.0250 (74%), align-to-self churn 0 / flat-zero barrier, permutation-
     invariance max output drift <= 1.94e-7. HONEST NOTE for any follow-up: at toy
     scale the greedy (non-Hungarian) solve lands in the 26-35% post/pre band, so
     the honest verdict is PARTIALLY REDUCED not COLLAPSED — a true Hungarian
     assignment is the open follow-up if full collapse is wanted.) -->
<!-- (TNNet.PredictionDepthReport removed: completed, landed 2026-05-24 via the
     [[introspection-report-pattern]] — the per-example difficulty diagnostic
     (prediction depth, Baldock/Maennel/Neyshabur 2021). Two overloads:
     `PredictionDepthReport(NN, Support, SupportLabels, Queries [, K=5,
     MaxFeatDim=256])` and the labelled variant adding QueryLabels. Forward-only,
     non-parametric: snapshots each trainable layer's L2-normed flat activation
     over support+query batches, per layer takes a K=5 cosine k-NN vote over the
     support, depth = shallowest layer after which the vote agrees with the net's
     final argmax and never disagrees again. Reuses LinearProbeReport's sign-random
     projection for layers wider than MaxFeatDim. Reports depth histogram +
     mean/median, per-layer newly-resolved profile, K hardest query indices, and a
     correct-vs-incorrect cross-tab (labelled overload). Ships
     examples/PredictionDepth/ + TestPredictionDepthReportSmoke. Checks PASS:
     support-as-own-query agreement 1.0000 / all depth 0, one-class collapse to 0,
     positive correct-vs-incorrect depth gap after training (fresh-init histogram
     right-skewed -> trained mass shifts shallow with a deep hard-query tail).
     DESIGN NOTE for any follow-up: "hard" example queries must be points the
     NETWORK finds ambiguous (between-blob band), NOT merely relabelled clean
     points — depth is measured vs the net's own argmax, so a wrong stored label
     alone doesn't raise depth.) -->
<!-- (TNNet.LogitLensReport removed: completed, landed 2026-05-24 via the
     [[introspection-report-pattern]]. Signature
     `LogitLensReport(NN; pInput: TNNetVolumeList; HeadStartIdx: integer = -1)`.
     Default head = last trainable layer + its activation/softmax tail; for each
     earlier layer whose flat activation matches the head input size it splices
     that activation into the head-input slot (Copy then CopyNoChecks, ActivationPatching
     idiom) and recomputes only the head, reading off p_L. Reports per-layer
     argmax-agreement bar chart, crystallization depth (mean + 10-bin histogram),
     mean confidence / lens entropy, and the KL(p_L || p_final) curve; width-
     incompatible layers listed as SKIPPED. Ships examples/LogitLens/ (fresh-vs-
     trained contrast on a constant-width classifier) + TestLogitLensReportSmoke.
     Correctness checks PASS: lens at the head input reproduces p_final exactly
     (agreement 1.0, KL 0) and the single-layer head degenerates to "resolves at
     the last layer". HONEST NOTE for any follow-up: on a tiny synthetic net the
     trained per-layer KL is NOT strictly monotone across the body (a sharper
     trained head penalises a wrong intermediate readout more), so the example's
     narrative claims only the seed-robust signals (entropy/confidence sharpening
     with depth, exact KL=0 / agreement=1.0 at the head input). The `Project`-to-
     force-a-fit flag from the spec was NOT implemented — incompatible layers are
     honestly SKIPPED instead; a projected-lens variant is the open follow-up.) -->
- [ ] LogitLensReport follow-up: the spec's optional `Project` flag (reuse
      [[LinearProbeReport]]'s deterministic random-projection to FORCE a
      width-incompatible layer through the head, flagged "heuristic") was not
      implemented — the landed report honestly SKIPs incompatible layers. Add the
      projected-lens path behind a flag so deeper/narrower stems get a (heuristic)
      depth profile too, keeping SKIP as the default honest behaviour.
- [ ] MagnitudePruningReport follow-up (now landed): the report sweeps a FIXED
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
- [ ] ActivationPatchingReport follow-up (now landed): the report and example
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
- [ ] IntrinsicDimensionReport follow-up (now landed): the PCA participation
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
