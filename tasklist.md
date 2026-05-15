# Task List — Feature & Enhancement Ideas

## New layer types
- [ ] TNNetMultiHeadSelfAttention + full transformer encoder/decoder blocks
- [x] TNNetLayerNorm — proper layer normalization
- [x] TNNetRotaryEmbedding (RoPE)
- [x] TNNetGEGLU / TNNetSwiGLU gated activations
- [x] TNNetGroupNorm
- [x] TNNetDropPath (stochastic depth)
- [ ] Sparse / mixture-of-experts routing layer

## Interesting applications / examples
- [ ] Tiny GPT — char-level transformer trained end-to-end in Pascal on CPU
- [ ] Tokenizer + trainable word embeddings example with nearest-neighbor visualization
- [ ] Audio: 1D-conv keyword-spotting example (spoken digit recognition)
- [ ] Time series forecasting example (energy load / weather)
- [ ] Reinforcement learning: minimal DQN solving CartPole or a grid world
- [ ] On-device anomaly detection autoencoder
- [ ] Style transfer or diffusion-lite denoiser (building on SuperResolution / VisualGAN)

## Infrastructure / dev experience
- [ ] Benchmark suite reporting throughput per layer type across AVX/AVX2/AVX512/OpenCL
- [ ] Mixed-precision (FP16) volumes for the OpenCL path
- [ ] Gradient checkpointing for training deeper nets in less memory
- [ ] Model zoo loader that pulls pre-trained weights from the companion repo
- [ ] ONNX import
- [ ] ONNX (or simpler JSON) export path
- [ ] CI-friendly headless test runner with coverage reporting
- [ ] Expand layer test coverage — numerical-gradient checks for layers that lack them

## Documentation / learning
- [ ] Interactive "build your first transformer in Pascal" tutorial
- [ ] Auto-generated layer API reference from doc comments

## Added ideas

### Normalization layers (broken down for easy implementation)
- [x] TNNetLayerNorm — per-sample layer normalization over all elements, with
      learnable scale/bias. Add to LoadFromString/CreateLayer dispatch. Add
      forward + numerical-gradient tests in TestNeuralNumerical.pas.
- [x] TNNetGroupNorm — normalize within channel groups (configurable group
      count). Reuse the channel-iteration patterns from
      TNNetChannelStdNormalization. Add tests in TestNeuralNumerical.pas.

### Test coverage
- [x] Numerical-gradient checks for activation layers currently lacking them
      (TNNetSwish, TNNetHardSwish, TNNetGELU already done, TNNetMish already
      done — audit and cover the gaps: e.g. TNNetSELU, TNNetLeakyReLU,
      TNNetHardSigmoid). Add to TestNeuralNumerical.pas.

### Smaller follow-up ideas
- [x] TNNetRMSNorm — root-mean-square layer norm (no mean subtraction); cheaper
      transformer-friendly variant. TNNetLayerNorm has now landed and is a ready
      template (neural/neuralnetwork.pas + TestNeuralNumerical.pas).
- [x] Document TNNetLayerNorm / TNNetGroupNorm in the README.md layer reference
      with a short usage snippet.
- [x] Audit other layers that override Backpropagate but lack a numerical
      gradient test — the activation-layer audit uncovered two real bugs
      (TNNetLeakyReLU.Compute and TNNetSigmoid.Backpropagate), so other layer
      families may hide similar issues. Audited the transform/reshape/pooling/
      element-wise families; added numerical-gradient checks for TNNetPadXY,
      TNNetCrop, TNNetInterleaveChannels, TNNetAvgPool, TNNetCellBias and
      TNNetCellMul (input + weight gradients). No new bugs found in those
      layers. Note: TNNetPointwiseSoftMax.Backpropagate uses the diagonal-only
      x*(1-x) approximation rather than the full softmax Jacobian — left as a
      known approximation, not added as a failing test.
- [ ] Quick-start example: tiny char-level sequence model (XOR-of-bits or
      counting task) that trains in well under a minute on CPU.
- [ ] Volume unit micro-benchmark printing ns/op for Add, Mul, DotProduct so
      regressions are visible without OpenCL/AVX hardware differences.
- [ ] Investigate TNNetPointwiseSoftMax.Backpropagate: it uses the diagonal-only
      x*(1-x) approximation instead of the full softmax Jacobian. Decide whether
      to implement the exact Jacobian (and add a numerical-gradient test) or
      document the approximation explicitly in the code/README.
- [ ] Continue the Backpropagate audit: the transform/reshape/pooling/element-
      wise families are now covered (TNNetPadXY, TNNetCrop,
      TNNetInterleaveChannels, TNNetAvgPool, TNNetCellBias, TNNetCellMul).
      The concat/split/branch family is now also covered: TNNetConcat,
      TNNetDeepConcat, TNNetSplitChannels, TNNetSum each have a numerical-
      gradient test in TestNeuralNumerical.pas. Remaining uncovered families
      to check next: upsampling/deconvolution layers and recurrent-style
      layers. Add numerical-gradient checks in TestNeuralNumerical.pas, one
      family at a time.
- [ ] TNNetRMSNorm has now landed (neural/neuralnetwork.pas +
      TestNeuralNumerical.pas) and is, alongside TNNetLayerNorm, a ready
      template for any further normalization-layer variants.

#### Layers I'd enjoy building
- [x] TNNetScaledDotProductAttention — the single-head core (Q·Kᵀ / √d → softmax
      → ·V) as a standalone, fully gradient-checked layer. Landed: parameter-free
      layer with input depth = 3*d_k (Q|K|V split along the depth axis), optional
      causal mask flag, full backward pass through the softmax Jacobian, and three
      numerical-gradient tests (forward sanity, non-causal grad check, causal grad
      check) in TestNeuralNumerical.pas.
- [x] TNNetDropPath (stochastic depth) — already listed above; I'd like to take
      it. Identity at inference, whole-sample drop at training. Pairs well with
      a numerical-gradient test that fixes the RNG seed. Landed: parameter-free
      layer descended from TNNetAddNoiseBase (so TNNet.EnableDropouts toggles
      training/inference), inverted-dropout scaling 1/(1-p) on kept samples
      with backward pass reusing the stored scalar, registered in both
      CreateLayer dispatches, and three tests in TestNeuralNumerical.pas
      (inference identity, training scaling + grad mirror, seeded
      central-difference gradient check).
- [x] TNNetGEGLU / TNNetSwiGLU gated activations — split input in half, gate one
      half with GELU/Swish of the other. Cheap, transformer-relevant, easy to
      gradient-check. Landed: both layers split along the depth axis (output
      depth = input depth / 2), no learnable weights, with forward +
      numerical-gradient tests in TestNeuralNumerical.pas.
- [x] TNNetLearnableScale / TNNetLayerScale — per-channel learnable multiplier
      (the "γ" trick from CaiT/ConvNeXt) that stabilizes deep residual nets.
      Landed as TNNetLayerScale (with TNNetLearnableScale type alias);
      constructor-configurable initial scale, gradient-checked for both input
      and learnable-weight gradients.
- [x] TNNetSoftPlus and TNNetGaussianActivation — round out the activation
      family; both have clean closed-form derivatives for the test. Landed
      with forward + numerical-gradient tests; TNNetSoftPlus uses a
      numerically stable formulation for large x.

#### Correctness / audit work I find rewarding
- [ ] Implement the exact softmax Jacobian for TNNetPointwiseSoftMax.Backpropagate
      (replacing the diagonal-only x*(1-x) approximation) and add a numerical-
      gradient test proving it. This is the unfinished thread from the second
      pass above — I'd like to actually close it.
- [ ] Continue the Backpropagate audit into the upsampling/deconvolution family
      (TNNetUpsample, TNNetDeconvolution, TNNetDeMaxPool) — one numerical-
      gradient test per layer, looking for bugs the way the activation audit did.
- [ ] Add a shared numerical-gradient helper in TestNeuralNumerical.pas so each
      new layer test is ~3 lines instead of a copy-pasted block. Reduces the
      friction of every future test task.

#### Experiments I'm curious about
- [ ] Activation-function bake-off: train the same small MLP on a fixed toy
      dataset with ReLU / GELU / Swish / Mish / SELU and print a comparison
      table of final loss and epochs-to-converge. Pure-CPU, runs in seconds.
- [ ] Normalization bake-off: same idea, comparing no-norm / LayerNorm /
      RMSNorm / GroupNorm on a small net, showing convergence-speed differences.
      Uses the layers that just landed.
- [ ] Weight-initialization sensitivity demo: show how a deep-ish net's
      first-epoch gradient magnitudes change across the available init schemes.
- [ ] Tiny "learns to add two binary numbers" sequence example — a fun, fast,
      self-contained demo of the library on a task with an obvious right answer.

#### Documentation
- [ ] Write a short "how numerical gradient testing works in this repo" note so
      contributors can add layer tests confidently — it's the project's main
      correctness safety net but isn't explained anywhere.

### Added ideas
- [x] Document the newly-landed layers in README.md: TNNetLayerScale /
      TNNetLearnableScale, TNNetSoftPlus, TNNetGaussianActivation, TNNetGEGLU
      and TNNetSwiGLU. Each needs a one-line description plus a short usage
      snippet, matching the existing layer-reference style. Done — also
      documented TNNetGLU, TNNetSquaredReLU and TNNetMaskedFill in the same
      pass (new "Gated Linear Units" and "Attention Masking" subsections).
- [ ] TNNetSwiGLU/TNNetGEGLU are the gating half of a transformer FFN — a
      natural next step is a TNNetSwiGLUFeedForward example or block that
      pairs them with the dense projections, ready for the transformer-encoder
      task at the top of the list.

#### Layers I'd enjoy building
- [x] TNNetRotaryEmbedding (RoPE) — apply rotary position encoding to a
      Q/K projection. Listed at the top of the file; I'd like to take it as a
      standalone, gradient-checked layer (the rotation is a fixed, parameter-free
      transform so the backward pass is just the transpose rotation).
      Landed: parameter-free layer derived from TNNetIdentity; Create() defaults
      base=10000. Forward + numerical-gradient + inverse-rotation tests in
      TestNeuralNumerical.pas.
- [x] TNNetMaskedFill / causal-mask layer — add -inf to the upper triangle of an
      attention-score map before softmax. Landed as a parameter-free layer
      derived from TNNetIdentity; Create() defaults to -1e9, Create(value)
      configurable. Forward + numerical-gradient tests in TestNeuralNumerical.pas.
- [x] TNNetGLU (plain gated linear unit) — the ungated-activation sibling of the
      GEGLU/SwiGLU pair that just landed: split in half, gate one half with a
      plain sigmoid of the other. Landed with forward + numerical-gradient tests.
- [x] TNNetSquaredReLU — relu(x)^2, the activation from the Primer paper. Landed
      as an activation layer (descends from TNNetReLUBase) with forward +
      numerical-gradient tests.

#### Correctness / audit work I find rewarding
- [x] Continue the Backpropagate audit into the concat/split/branch family
      (TNNetConcat, TNNetDeepConcat, TNNetSplitChannels, TNNetSum) — one
      numerical-gradient test per layer, hunting for bugs the activation audit
      style turned up before. (Done: TestConcatGradientCheck,
      TestDeepConcatGradientCheck, TestSplitChannelsGradientCheck,
      TestSumGradientCheck in TestNeuralNumerical.pas. No bugs found; all four
      layers' input-gradient paths match finite differences within 1e-2.)
- [x] Add a numerical-gradient test for TNNetAddPositionalEmbedding (and any
      other layer carrying learnable weights that lacks one) — the learnable-
      weight gradient path is the easiest place for a silent bug to hide.
      (Done: forward + input-gradient + constant-encoding regression tests.
      Note: this layer has no learnable weights; FPositionalEmbedding is a
      fixed sinusoidal encoding per Vaswani et al., so the "learnable-weight"
      branch does not exist here.)

#### Experiments I'm curious about
- [ ] Optimizer bake-off: train the same small MLP with SGD / SGD+momentum /
      Adam / RMSProp on a fixed toy dataset and print a loss-vs-epoch table.
      Pure-CPU, runs in seconds, and complements the activation/normalization
      bake-offs already on the list.
- [ ] Batch-size sweep demo: same net and data, vary the batch size, and print
      how wall-clock-per-epoch and epochs-to-converge trade off. A concrete,
      visible illustration of a tuning knob beginners always ask about.
- [ ] Dead-ReLU diagnostic: train a small ReLU net and print the fraction of
      units that never fire across an epoch, then repeat with LeakyReLU/GELU to
      show the difference. A fun, instructive correctness-adjacent experiment.

#### Documentation
- [ ] Write a one-page "layer authoring checklist" — constructor + LoadFromString
      round-trip, CreateLayer dispatch entry, Compute/Backpropagate, and the
      mandatory numerical-gradient test. Captures the recurring steps every
      new-layer task in this file actually follows.

### Added ideas
- [x] Add `tests/RunTests` (and any other fpc build artifacts under tests/) to
      .gitignore — already present in .gitignore (line "tests/RunTests").
- [x] Now that TNNetMaskedFill and the gated-activation family have landed, the
      remaining transformer building blocks are TNNetScaledDotProductAttention
      and TNNetRotaryEmbedding (RoPE). SDPA has now landed too; with masking and
      SDPA done, the highest-leverage next layer is TNNetRotaryEmbedding (RoPE)
      or going directly to TNNetMultiHeadSelfAttention composed on top of SDPA.
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

### Ideas added on 2026-05-15 (lucky seed 623999)

#### Layers I'd enjoy building next
- [x] TNNetScaledDotProductAttention — the single-head attention core
      (Q·Kᵀ / √d → softmax → ·V) as a standalone, gradient-checked layer.
      Landed: parameter-free layer takes Q|K|V concatenated along the depth axis
      and supports an optional causal mask. Pairs cleanly with TNNetMaskedFill.
- [x] TNNetRotaryEmbedding (RoPE) — parameter-free rotation applied to Q/K.
      Backward pass is just the inverse rotation, so the numerical-gradient
      test is straightforward. Companion piece to ScaledDotProductAttention.
      Landed: with forward, numerical-gradient and inverse-rotation tests.
- [ ] TNNetMultiHeadSelfAttention — SDPA has now landed, so this is unblocked
      (RoPE is optional / additive). Compose split-heads → SDPA per head →
      concat → out projection. Suggested breakdown:
      (a) head-split helper: a TNNetSplitChannels-based wiring example that
          carves a (3*d_model) Q|K|V input into H per-head (3*d_k) slices.
      (b) per-head SDPA: just instantiate TNNetScaledDotProductAttention(d_k)
          on each slice — already tested.
      (c) head-concat: TNNetDeepConcat across the H heads back to depth d_model.
      (d) output projection: a TNNetFullConnectLinear(d_model).
      (e) wrap it all in a TNNetMultiHeadSelfAttention helper class or a
          builder function on TNNet. Add a numerical-gradient test on a tiny
          H=2, d_k=4, SeqLen=3 example.
- [x] TNNetDropPath (stochastic depth) — identity at inference, whole-sample
      drop at training. Use a fixed RNG seed in the test so the masked
      forward/backward is deterministic. Landed: see TNNetDropPath entry above.
- [ ] TNNetMishExact / TNNetGELUExact audit — confirm whether the current
      implementations use the approximation or the exact tanh/erf form, and
      document the choice in the code.

#### Experiments I'm curious about
- [ ] Tiny GPT proof-of-life: once SDPA + RoPE + MultiHeadSelfAttention land,
      train a 2-layer char-level transformer on a short text snippet (e.g.
      Tiny Shakespeare excerpt or repeated arithmetic patterns) and print
      generated samples. End-to-end demo of the new transformer stack.
- [ ] Causal-mask sanity experiment: train a tiny attention model on next-token
      prediction WITH and WITHOUT TNNetMaskedFill, and show the unmasked one
      cheats (near-zero loss but useless at generation). A concrete teaching
      example of why causal masking exists.
- [ ] Gradient-magnitude visualizer: print per-layer gradient norms across
      training steps for a deep MLP, with and without LayerNorm/RMSNorm, to
      visualize the vanishing/exploding-gradient story. Builds on the
      normalization bake-off already in the list.
- [ ] Numerical-precision study: re-run the activation bake-off using FP32 vs
      a simulated-FP16 path (round-trip volumes through fewer mantissa bits)
      and report the convergence-quality gap. Useful baseline for any future
      mixed-precision work.

#### Correctness / audit work I'd enjoy
- [x] Continue the Backpropagate audit into the concat/split/branch family
      (TNNetConcat, TNNetDeepConcat, TNNetSplitChannels, TNNetSum) — one
      numerical-gradient test per layer. Listed above; I want to take it.
      (Done: see TestConcatGradientCheck / TestDeepConcatGradientCheck /
      TestSplitChannelsGradientCheck / TestSumGradientCheck. No bugs found.)
- [x] Numerical-gradient test for TNNetAddPositionalEmbedding's learnable
      weight path — already listed; the learnable-weight branch is the
      easiest place for a silent bug to hide. (Done: see above. The layer
      has no learnable weights, so only the input-gradient path applies.)
- [ ] Add a shared `AssertLayerGradient(layer, inputShape)` helper in
      TestNeuralNumerical.pas so each new layer test becomes ~3 lines instead
      of a copy-pasted block. Big quality-of-life win for the audit work.
- [ ] Property-based-style test: pick N random small layer configurations,
      build a 1-layer net, run a numerical-gradient check on each. Catches
      shape-edge-case bugs the hand-written tests miss.

#### Documentation
- [ ] Write the "how numerical gradient testing works in this repo" note
      promised above — it's the project's main correctness safety net but
      isn't explained anywhere. Should cover the eps choice, central
      differences, tolerance picking, and where to add new tests.
- [ ] Write the "layer authoring checklist" note also promised above —
      constructor, LoadFromString round-trip, CreateLayer dispatch,
      Compute/Backpropagate, numerical-gradient test. Captures the recurring
      pattern every new-layer task in this file actually follows.
- [ ] Short "transformer-from-scratch in Pascal" walkthrough that wires up
      LayerNorm + MaskedFill + SDPA + RoPE + MHA + GEGLU into a working
      encoder block, once those layers land. Companion to the Tiny GPT
      example at the top of the file.

#### Infrastructure niceties
- [ ] Add `tests/RunTests` and other fpc build artifacts to .gitignore
      (already listed above; small, satisfying cleanup).
- [ ] Volume-unit micro-benchmark printing ns/op for Add, Mul, DotProduct
      (already listed; would enjoy actually shipping it).
- [ ] Add a `make test` / `make smoketest` shortcut that builds + runs the
      numerical-gradient test suite with one command — lowers the friction
      for every future contributor.

### Ideas added on 2026-05-15 (lucky seed 9052)

#### Layers I'd enjoy building next
- [x] TNNetRotaryEmbedding (RoPE) — the natural companion to the SDPA layer
      that just landed. Parameter-free rotation of Q/K pairs of channels by
      position-dependent angles. Backward = inverse rotation, so the
      numerical-gradient test is short. Should accept an optional base
      frequency (default 10000) and operate in-place across the depth axis.
      Landed: Create(pBase=10000.0), validates even Depth at SetPrevLayer.
- [ ] TNNetALiBi positional bias — alternative to RoPE: add a static
      slope * (j - i) bias to attention scores. Pairs with TNNetMaskedFill
      and gives a second option for position handling in the eventual MHA.
- [x] TNNetSoftCapping — `c * tanh(x / c)` logit-capping layer used by
      Gemma-style models. Parameter-free, single closed-form derivative,
      cheap stabilizer for attention scores and final logits.
      Landed: descends from TNNetIdentity, stores the cap in FFloatSt[0]
      (default 30.0), caches `1 - tanh(x/c)^2` in FOutputErrorDeriv so
      Backpropagate is a single elementwise multiply. Save/Load
      round-trip wired through both dispatch tables, with forward,
      saturation, round-trip, and central-difference gradient tests
      in TestNeuralNumerical.
- [ ] TNNetMixtureOfExperts (top-k gating) — even a tiny CPU-friendly
      version is fun: a softmax gate over E experts, top-1 routing, route
      each sample through one expert FullConnect. Numerical gradient
      becomes interesting because the routing is non-differentiable;
      document and test the straight-through approximation.
- [ ] TNNetReversibleBlock — i.e. a RevNet-style additive coupling
      (`y1 = x1 + F(x2)`, `y2 = x2 + G(y1)`). Lets the test prove that
      forward + inverse round-trips to within fp tolerance, on top of the
      usual gradient check.

#### Composite helpers / blocks
- [ ] TNNetSwiGLUFeedForward block helper — wrap LayerNorm → dense projection
      → SwiGLU → dense out-projection into one builder. Already half-listed
      above; I'd like to actually ship it once MHA lands so the transformer
      example becomes a few lines.
- [ ] TNNetTransformerEncoderBlock helper — LayerNorm → MHA → residual →
      LayerNorm → SwiGLU FFN → residual. Single call, configurable
      d_model / heads / d_ff. Companion numerical-gradient test on a tiny
      shape (d_model=8, heads=2, seq=3).
- [ ] TNNetTransformerDecoderBlock helper — adds the causal MaskedFill in
      front of self-attention and an optional cross-attention sub-block.
      Built on top of the encoder helper above to avoid duplication.

#### Experiments I'm curious about
- [ ] Attention-pattern visualizer: after training the tiny GPT proof-of-life,
      dump the softmax attention matrix to a PGM image so the diagonal /
      induction-head patterns are visible. Small, satisfying, and a great
      teaching artifact.
- [ ] Position-encoding bake-off: same tiny seq model trained with (a) no
      position info, (b) sinusoidal AddPositionalEmbedding, (c) RoPE,
      (d) ALiBi, printing final loss and a sample generation per scheme.
      Becomes possible once RoPE + ALiBi land.
- [ ] Causal-mask leak test: deliberately remove TNNetMaskedFill from a
      next-token model and show that validation loss drops to ~0 while
      sampled completions are garbage. A reproducible cautionary tale.
- [ ] "Lottery-ticket"-flavored experiment: train a small dense net,
      magnitude-prune the bottom X% of weights, retrain from the original
      init, and compare. Pure CPU, finishes in seconds, fun to watch.
- [ ] Init-scheme × depth heatmap: for depths {2, 4, 8, 16} and inits
      {Glorot, He, LeCun, plain N(0, 0.01)}, plot first-step gradient norm
      at the deepest layer. Concrete visualization of vanishing gradients.

#### Correctness / audit work
- [ ] Property-based numerical-gradient harness: write a small generator that
      yields random (shape, layer-type, hyperparams) tuples and runs the
      gradient check on each. Already half-listed above — I want to commit
      to actually shipping it as a new test program under tests/.
- [ ] Determinism audit: with a fixed RNG seed and the OpenCL path disabled,
      assert that a tiny model produces bit-identical loss across 3 runs.
      Add as a smoketest so future PRs can't silently introduce nondeterminism.
- [ ] SDPA edge-case tests: SeqLen=1 (degenerate softmax), all-masked rows
      (the row where every key is masked — currently softmax produces NaN;
      decide policy and add a test pinning the chosen behavior).
- [ ] LoadFromString / SaveToString round-trip property test: for every
      registered layer type, build a default instance, serialize, deserialize,
      and assert the rebuilt layer reports the same Compute output on a
      fixed random input. Catches dispatch-table gaps the moment they appear.

#### Tooling / dev experience
- [ ] Tiny CLI `neural-bench` that times forward + backward for a chosen
      layer at a chosen shape and prints ns/op. Subsumes the volume-unit
      micro-benchmark idea above and extends it to whole layers.
- [ ] Coverage script: parse the test programs and report which `TNNet*`
      classes lack at least one Compute test and at least one numerical-
      gradient test. Generates the next batch of audit tasks automatically.
- [ ] `scripts/new_layer.sh <Name>` scaffolder that drops a Compute /
      Backpropagate skeleton into neuralnetwork.pas and a matching
      numerical-gradient test stub into TestNeuralNumerical.pas. Captures
      the "layer authoring checklist" doc as executable form.

#### Documentation
- [ ] "Reading a numerical-gradient failure" mini-guide — when the harness
      reports a mismatch, what does the magnitude tell you (analytic-bug
      vs. tolerance-too-tight vs. discontinuity-near-the-eps-step)? A
      page-long companion to the testing note already on the list.
- [ ] Annotated SDPA walkthrough: the new TNNetScaledDotProductAttention is
      the most algorithmically dense layer in the repo — a short doc
      stepping through Compute and Backpropagate (with shapes and the
      softmax-Jacobian derivation) would lower the barrier for the next
      contributor touching attention code.

### Ideas added on 2026-05-15 (post SoftCapping + DropPath + RoPE landings)

With TNNetSoftCapping, TNNetDropPath and TNNetRotaryEmbedding now landed,
the highest-leverage next layer is clearly TNNetMultiHeadSelfAttention
(SDPA + RoPE + MaskedFill are all in place). Suggested doable breakdown
for the next contributor — each step is its own commit-sized task:

- [ ] (MHA-a) Add a small `TNNetSplitChannels`-based helper or example that
      carves a (3*d_model) Q|K|V depth slab into H per-head (3*d_k) slices,
      with a sanity test on H=2, d_model=8.
- [ ] (MHA-b) Add a wiring helper or example that runs one
      `TNNetScaledDotProductAttention(d_k)` per head slice and concats
      the H outputs via `TNNetDeepConcat` back to depth d_model. Test on
      a tiny (H=2, d_k=4, SeqLen=3) shape, numerical-gradient through.
- [ ] (MHA-c) Wrap (a)+(b)+a `TNNetFullConnectLinear(d_model)` out-projection
      into a `TNNetMultiHeadSelfAttention` helper class or builder function.
      Add a numerical-gradient test on the same tiny shape; mark line 4 of
      this file `[x]` once it lands.

Bake-off / experiment follow-ups for the layers that just landed:

- [ ] SoftCapping logit-stability micro-experiment: train a tiny classifier
      with and without a `TNNetSoftCapping(c)` before the final softmax,
      and print the rate of NaN/overflow events under an aggressive
      learning rate. Tiny, CPU-fast, illustrative.
- [ ] DropPath ablation: train a small ResNet-style net on a tiny synthetic
      task with `TNNetDropPath(p)` after each residual block, sweeping
      `p ∈ {0.0, 0.1, 0.2}` and printing final loss. Shows the regularizer
      doing something even at toy scale.
- [ ] RoPE vs sinusoidal `TNNetAddPositionalEmbedding` mini-comparison on
      a tiny next-token task — pair with the existing position-encoding
      bake-off entry above (now unblocked since RoPE landed).

Tiny correctness follow-ups:

- [x] SoftCapping saturation test on extreme inputs (±1e6) — assert
      output stays within ±c and `Backpropagate` doesn't produce NaNs.
      Done: TestSoftCappingExtremeInputSaturation in TestNeuralNumerical.pas.
- [x] DropPath determinism test: with a fixed `RandSeed`, two Compute
      calls produce identical masks/outputs across runs. Pin the
      RNG-reset behavior so future refactors can't silently break it.
- [x] RoPE odd-depth guard test: assert constructing+wiring a
      `TNNetRotaryEmbedding` after a layer with odd Depth raises the
      expected error (validates the even-Depth precondition).
      Done: TestRotaryEmbeddingOddDepthGuard via the FErrorProc capture
      pattern in TestNeuralNumerical.pas.

### Ideas added on 2026-05-15 (lucky seed 164520)

The MHA breakdown (MHA-a/b/c) above is the headline next step. The ideas
below are things I would personally enjoy taking on either as warm-up
tasks before MHA lands, or as parallel tracks while it does.

#### Layers I'd enjoy building
- [ ] TNNetALiBi — finally pick up the position-bias entry above. It is
      the smallest possible "alternative to RoPE": precompute a single
      `slope[h]` per head and add `slope * (key_pos - query_pos)` into
      the attention score map before softmax. Parameter-free (slopes are
      deterministic from head count, per the paper), so the gradient
      check is the same shape as TNNetMaskedFill's.
- [ ] TNNetSinkAttention — small, fun variant: prepend K "attention sink"
      key/value slots that every query can attend to. Helps long-context
      stability and is a 30-line addition on top of SDPA once the per-row
      mask/key-padding plumbing exists.
- [ ] TNNetTalkingHeadsProjection — the cross-head mixing matrix from
      Shazeer et al. A tiny learnable HxH multiply applied to attention
      logits across heads. Drops into the eventual MHA helper, parameter
      count is trivial, gradient check is a straight matmul backward.
- [ ] TNNetTokenShift — the Lite-RWKV / time-mixing trick: blend each
      token with the previous token via a learnable per-channel mix.
      Cheap, useful for the eventual tiny-GPT story, and a clean
      sequence-aware layer that does not need attention to be done.
- [x] TNNetSpatialDropout1D / TNNetSpatialDropout2D — drop entire
      channels/feature-maps instead of individual elements; common in
      conv and seq nets. Landed: both descend from TNNetAddNoiseBase,
      keep_prob in FFloatSt[0], use TNNetVolume.MulChannels for the
      per-channel mask, registered in both CreateLayer dispatches.
      Forward identity (inference), per-channel mask shape, seeded
      gradient check, and SaveToString round-trip tests for each
      (TestSpatialDropout{1D,2D}{InferenceIdentity,TrainingMaskShape,
      GradientCheck,SerializationRoundTrip}) in TestNeuralNumerical.pas.
      1D and 2D share the same Compute (mask along Depth, broadcast over
      SizeX*SizeY by MulChannels) — the split is a Keras/PyTorch-style
      naming convention.
- [ ] TNNetStochasticPool — the stochastic-pooling alternative to MaxPool
      (sample one cell per window weighted by its activation). Cute,
      parameter-free, RNG-seeded test similar to DropPath's.

#### Composite blocks / examples
- [ ] TNNetPreNormResidual helper — `y = x + Sublayer(LayerNorm(x))`
      single-line builder. Once MHA + SwiGLU FFN land, this is the
      shortest path to a "real" transformer block.
- [ ] TNNetGLUFeedForward block — same shape as the SwiGLU FFN entry
      above but using the plain TNNetGLU that already landed; gives a
      working FFN to test the pre-norm-residual builder against today,
      no waiting on new gating layers.
- [ ] Tiny "induction-heads" demo: train a 2-layer attention-only model
      on a repeat-the-pattern toy task and show the second layer's
      attention diagonal jumps to the previous-occurrence position.
      Lovely small experiment that exercises SDPA + MaskedFill + RoPE
      end-to-end and produces a publishable PGM artifact.

#### Experiments I'm curious about
- [ ] Causal-mask + SoftCapping interaction study: with logits clipped
      via `TNNetSoftCapping(c)`, sweep `c ∈ {5, 10, 20, 30, ∞}` on a
      tiny next-token task and chart loss + max-logit-norm. Pairs the
      two stabilizers that just landed.
- [ ] DropPath schedule study: linearly increasing drop probability with
      depth (the Stochastic-Depth schedule from the paper) vs constant
      `p`. Train a small residual stack on a toy task and chart final
      loss for both schedules. Concrete demonstration of why the
      schedule matters.
- [ ] RoPE base-frequency sweep: same tiny next-token model, sweep
      `base ∈ {1e2, 1e3, 1e4, 1e5}`, chart loss and qualitative sample
      quality. Tiny knowledge-building experiment about a number that
      gets cargo-culted as "10000".
- [ ] "Surgery" experiment: train a small classifier, then zero out the
      top-K most-active hidden units and chart accuracy degradation
      vs K. Cheap, fun, teaches "redundancy in representations".
- [ ] Numerical-gradient eps sweep: pick one well-tested layer, run the
      gradient check with `eps ∈ {1e-2, 1e-3, 1e-4, 1e-5, 1e-6}` and
      print max-error vs eps. Produces the canonical curve we should
      reference when picking a tolerance for new tests.

#### Correctness / audit work I'd enjoy
- [ ] SDPA all-masked-row policy decision and test: currently a row
      where every key is masked produces NaN (softmax of all -inf).
      Decide between "row outputs zeros" or "row passes through V mean"
      or "raise an error", document the choice in the code, and add the
      test pinning the behavior. Already flagged above; I want to close it.
- [x] DropPath p=0 / p=1 boundary tests: p=0 must be exact identity
      (including in training mode), p=1 must zero the sample but still
      produce gradient zeros without NaNs. Done: TestDropPathPZeroBoundary
      and TestDropPathPOneBoundary. The p=1 test surfaced that
      TNNetDropPath.Create silently clamps `pDropProb >= 1` to 0.99
      (inverted-dropout div-by-zero guard) — see follow-up below.
- [x] SoftCapping `c → ∞` continuity test: as `c` grows the layer should
      approach identity. Done: TestSoftCappingLargeCapContinuity (c=1e6).
- [x] RoPE forward-then-inverse round-trip test at SeqLen > 1 — extend
      the existing inverse-rotation test to a non-trivial sequence
      length to catch position-indexing bugs. Done:
      TestRotaryEmbeddingInverseSeqLen5.
- [x] LoadFromString round-trip for the recent landings: explicitly
      cover TNNetSoftCapping, TNNetDropPath, TNNetRotaryEmbedding,
      TNNetMaskedFill, TNNetScaledDotProductAttention. Done: five
      Test*SerializationRoundTrip methods sharing a SerializationRoundTrip
      helper in TestNeuralNumerical.pas.
- [ ] Property-based gradient harness (kickoff): even a v0 that only
      randomizes input shape (keeping layer type fixed) for the 6 most
      recently landed layers is enough to start catching shape-edge
      bugs and lays the groundwork for the full version listed above.

#### Tooling / dev experience
- [x] `tests/RunAll.sh` that builds + runs every test program in tests/
      with a single command and a non-zero exit on any failure. Landed:
      uses `set -euo pipefail`, cd's via BASH_SOURCE so it works from any
      cwd, honors `LAZUTILS_PATH` env override, errors clearly if the
      lazarus path is missing, execs `./RunTests -a -p` so the runner's
      exit code propagates.
- [ ] Add a `--quick` flag to the numerical-gradient test runner that
      skips the heavier `SeqLen > 4` cases — useful for local
      iterate-fast loops while audits are in flight.
- [x] Tiny `scripts/list_untested_layers.sh` that greps neuralnetwork.pas
      for `TNNet*` class declarations and reports which ones never appear
      in any test file. Landed: v0 reports 53 untested classes (mix of
      genuinely-untested layers like TNNetMaxPoolWithPosition and
      base/abstract classes like TNNetReLUBase / TNNetPoolBase /
      TNNetConcatBase that don't need direct tests). Natural follow-ups
      below.

#### Documentation
- [ ] Annotated walkthrough of the SDPA Compute + Backpropagate pair —
      the softmax-Jacobian derivation, shape annotations on every line,
      and a worked tiny example (d_k=2, SeqLen=2). Most algorithmically
      dense layer in the repo; deserves the longest doc.
- [ ] "Picking a tolerance" doc for numerical-gradient tests — when is
      1e-2 fine, when do you need 1e-3, when should you tighten the eps
      instead of loosening the tolerance. Companion to the eps-sweep
      experiment above.
- [ ] One-pager "transformer building blocks landed in this repo" —
      a table of LayerNorm / MaskedFill / SDPA / RoPE / SoftCapping /
      DropPath / GEGLU / SwiGLU / GLU / SquaredReLU / LayerScale /
      AddPositionalEmbedding with a one-line "what it is" + "use it
      when". Becomes the index entry into the eventual MHA + encoder
      block helpers.

### Ideas added on 2026-05-15 (post SpatialDropout + safety-net + devx batch)

#### Bugs / quirks surfaced by the new safety nets
- [x] TNNetDropPath.Create silently clamps `pDropProb >= 1` to 0.99 to
      avoid a div-by-zero in the inverted-dropout `1/(1-p)` scaling.
      Consequence: `p=1` does NOT mean "always drop" — about 1% of samples
      survive and are amplified ~100x. Fix: special-case `pDropProb >= 1`
      in the constructor to set `Scale := 0` (and skip the division)
      instead of clamping the probability. Then tighten
      `TestDropPathPOneBoundary` to assert strict all-zero output.
      Done: TNNetDropPath.Compute now special-cases `P >= 1` (output := 0,
      Scale := 0); constructor preserves p verbatim; TestDropPathPOneBoundary
      asserts strict zero outputs and gradients.

#### Layers I'd enjoy building next (warm-up before MHA)
- [ ] TNNetSpatialDropout1D/2D follow-ups now that they have landed:
      - A schedule-aware variant (e.g. linearly-increasing channel-drop
        probability with depth) for the eventual ResNet/ConvNeXt examples.
      - Wire one of them into an existing CIFAR-10 example as an opt-in
        regularizer, with a one-line README mention.
- [ ] TNNetALiBi positional bias — still the smallest "alternative to
      RoPE" task on the list and now the most isolated next layer (no
      MHA dependency). Parameter-free, gradient check shape mirrors
      TNNetMaskedFill, and it slots into the eventual MHA helper.

#### Tooling follow-ups now that the v0 scripts shipped
- [ ] Filter `scripts/list_untested_layers.sh` to skip obvious base/
      abstract classes (names ending in `Base`, `Class`, or `Abstract`,
      plus a small explicit allowlist) so the report shows only the
      ~25-ish concrete-but-untested layers worth auditing. The current
      53-line output is still actionable but mixes signal with noise.
- [ ] Extend `scripts/list_untested_layers.sh` to also report each
      reported class's source file:line (from the `class(...)` declaration)
      so a contributor can jump straight to it.
- [ ] Add a `--quick` flag to RunAll.sh that passes through to a
      yet-to-be-added test-runner option to skip the slow numerical-
      gradient cases (pairs with the `--quick` flag idea already on
      the list).
- [ ] CI shim: a tiny GitHub Actions workflow that runs
      `tests/RunAll.sh` on push. The script already has the right
      exit semantics — the workflow is ~15 lines.

### Ideas added on 2026-05-15 (lucky seed 51855)

Picking up where the post-SpatialDropout batch left off — MHA is still
the headline next step, but there are several small, self-contained
tasks I'd personally enjoy landing while it incubates. Everything below
is sized to fit in a single focused commit.

#### Quick wins I'd enjoy taking first
- [x] Fix TNNetDropPath `p=1` clamping (already flagged above as a bug):
      special-case `pDropProb >= 1` in the constructor so Scale := 0
      instead of silently clamping probability to 0.99. Then tighten
      `TestDropPathPOneBoundary` to assert strict all-zero output and
      drop the existing tolerance-based check. Smallest non-trivial
      correctness fix on the list with a ready test to extend. Landed:
      see the "Bugs / quirks" entry above.
- [x] DropPath determinism test (the open follow-up flagged above):
      with a fixed `RandSeed`, two consecutive Compute calls in
      training mode produce identical masks/outputs. Add as
      `TestDropPathDeterminismFixedSeed` next to the existing tests.
      Done: see TestDropPathDeterminismFixedSeed in TestNeuralNumerical.pas.

#### Layers I'd enjoy building (no MHA dependency)
- [ ] TNNetALiBi — same entry as the previous two batches. The fact that
      it has been listed three times now is the universe telling me to
      take it. Per-head deterministic slopes (`slope[h] = 2^(-8h/H)`),
      adds `slope * (key_pos - query_pos)` into the attention score map
      before softmax. Parameter-free, gradient check mirrors MaskedFill.
- [ ] TNNetSwitchableNorm — a tiny learnable convex combination of
      LayerNorm and RMSNorm outputs (two learnable scalars summed via
      softmax). Cute, parameter-cheap experiment in "let the network
      pick its own normalizer". All ingredients already exist.
- [x] TNNetChannelShuffle — the ShuffleNet operation: reshape `(C)` to
      `(groups, C/groups)`, transpose, flatten back. Parameter-free,
      gradient is the inverse permutation, easy numerical check.
      Slots cleanly into the existing channel-manipulation family
      (TNNetInterleaveChannels, TNNetSplitChannels, etc.).
- [ ] TNNetSoftmaxTemperature — `softmax(x / T)` with configurable T,
      saved/loaded via FFloatSt[0]. Useful for the eventual tiny-GPT
      sampling demo and a clean small layer. Backprop is the standard
      softmax Jacobian scaled by 1/T.
- [ ] TNNetGatedResidual — `y = x + gate * Sublayer(x)` with a per-
      channel learnable gate initialized at zero (the "ReZero" trick).
      A different lever than LayerScale; pairs well with the
      PreNormResidual helper already listed above.

#### Composite blocks I'd enjoy shipping
- [ ] TNNetPreNormResidual helper (already listed above) — concrete
      commitment to actually ship the one-liner `y = x + Sublayer(LN(x))`
      builder so the eventual transformer-block helpers can use it.
- [ ] TNNetGLUFeedForward block (already listed) — same: I'd like to
      actually ship the plain-GLU FFN today rather than wait on SwiGLU
      to be the gating choice.

#### Experiments I'm curious about
- [ ] Train-time vs inference-time delta sweep for the noise layers
      (TNNetDropout, TNNetDropPath, the new TNNetSpatialDropout1D/2D):
      same tiny classifier, sweep `p ∈ {0.0, 0.1, 0.2, 0.4}`, chart
      train loss vs val loss. Concrete demonstration of which
      regularizers actually help at toy scale and which are noise.
- [ ] "Which init wins per activation" matrix: cross-product of init
      schemes × activation functions on a fixed tiny MLP, report
      epochs-to-converge. Sits between the activation bake-off and
      init-sensitivity demo already listed, and surfaces the
      Glorot-with-tanh / He-with-ReLU folklore on real numbers.
- [ ] Token-shift ablation (depends on TNNetTokenShift above): does a
      single token-shift layer in front of an MLP solve the next-
      token-prediction toy task without any attention? A cheap, fun
      data point about how much of "transformers" is really just
      mixing-along-time.

#### Correctness / audit follow-ups
- [x] LoadFromString round-trip for the SpatialDropout pair — the
      existing round-trip tests cover SoftCapping/DropPath/RoPE/
      MaskedFill/SDPA but not the just-landed 1D/2D spatial dropouts.
      Add `TestSpatialDropout{1D,2D}SerializationRoundTrip` mirroring
      the existing pattern.
- [x] LayerNorm / RMSNorm / GroupNorm round-trip via LoadFromString —
      these three landed earlier and I'd bet have at least one
      learnable-scale dispatch quirk hiding. Add to the existing
      round-trip suite.
- [ ] Backpropagate audit, upsample/deconv family (already listed):
      I'll take TNNetUpsample first because it's the simplest of the
      three and the rest will follow the same recipe.
- [ ] SDPA all-masked-row policy decision (already listed). Concrete
      proposal: detect the all-masked row in Compute, output a zero
      row, and skip the softmax for that row entirely (this is what
      JAX/Flax MHA does). Document the choice in code, add the
      pinning test, close the open thread.

#### Tooling / dev experience
- [ ] Implement the `--quick` filter on the test runner that pairs
      with the `--quick` flag idea already listed for RunAll.sh.
      Concrete proposal: skip any test whose name ends in
      `_Slow` or contains `Large` / `SeqLen>4` markers, controlled
      by a `--quick` switch on the existing test runner.
- [ ] Filter + line-numbers patch for `scripts/list_untested_layers.sh`
      (the two follow-up entries already listed) — bundle them into
      one v1 of the script that drops base/abstract classes and
      reports `file:line` for each surviving entry. Two-line awk
      change in practice.
- [ ] `scripts/grep_layer.sh <TNNet...>` helper that prints the
      class declaration, its Compute, its Backpropagate, and any
      test methods referencing it. Captures the "first 30 seconds
      after picking a layer to audit" workflow I keep doing by hand.
- [ ] Tiny `tests/SmokeTest.lpr` that builds + runs the five fastest
      gradient checks and exits in under a second. Lets the eventual
      CI shim above start with a real signal even before RunAll.sh
      is wired in.

#### Documentation
- [ ] Three-paragraph "what landed this month" entry pinned at the
      top of the README: SDPA, RoPE, MaskedFill, SoftCapping,
      DropPath, GEGLU/SwiGLU/GLU, SquaredReLU, LayerScale, the
      SpatialDropouts. One line each plus the layer-reference link.
      Becomes the public-facing landing page for the transformer
      push that's been happening across these task batches.
- [ ] Short "where the test suite lives" map: tests/TestNeuralNumerical
      vs the older `tests/*` programs, how RunAll.sh orchestrates
      them, and what a contributor should add when they ship a new
      layer. Sits next to the numerical-gradient testing note
      already on the list — companion piece, not a replacement.
- [ ] Inline-comment cleanup pass on TNNetScaledDotProductAttention:
      the layer now has six tests pinning its behavior and is the
      most algorithmically dense in the repo. Add shape annotations
      on every loop, name the strides, and link to the planned
      annotated walkthrough doc. Pure readability win, no behavior
      change.

### Ideas added on 2026-05-15 (post DropPath-fix + ChannelShuffle + Norm-roundtrip batch)

This batch landed: TNNetDropPath p>=1 strict-drop fix + determinism test,
TNNetChannelShuffle (ShuffleNet permutation) with full test set, and
LayerNorm/RMSNorm/GroupNorm LoadFromString round-trip coverage. No bugs
surfaced in the norm-serialization audit — gamma/beta and the GroupNorm
FStruct[0]=Groups all round-trip cleanly through the generic dispatch.

Natural follow-ups:

- [ ] TNNetChannelShuffle CIFAR/ImageNet-style integration example —
      drop it into one of the existing conv examples (e.g. SimpleImage)
      as a ShuffleNet-flavored block (1x1 conv -> ChannelShuffle ->
      depthwise conv). Visible end-to-end use of the new layer.
- [ ] TNNetChannelShuffle inverse property test: ChannelShuffle(G)
      composed with ChannelShuffle(C/G) is the identity. Cute one-line
      property check on top of the existing forward test.
- [ ] Now that DropPath p=1 is strict-drop, add a tiny experiment that
      confirms a deep residual net with p=1 DropPath after every block
      collapses to the identity path (loss curve matches a no-residual
      baseline). Cheap teaching artifact for the fix.
- [ ] Now that the norm round-trip suite covers Layer/RMS/Group, extend
      it to TNNetChannelStdNormalization and TNNetLocalResponseNorm2D
      (the older normalization layers that predate the audit) — same
      shared helper, two-line additions.
