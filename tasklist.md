# Task List — Feature & Enhancement Ideas

## New layer types
- [ ] TNNetMultiHeadSelfAttention + full transformer encoder/decoder blocks
- [x] TNNetLayerNorm — proper layer normalization
- [ ] TNNetRotaryEmbedding (RoPE)
- [x] TNNetGEGLU / TNNetSwiGLU gated activations
- [x] TNNetGroupNorm
- [ ] TNNetDropPath (stochastic depth)
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
      Remaining uncovered families to check next: upsampling/deconvolution
      layers, concat/split/branch layers, and recurrent-style layers. Add
      numerical-gradient checks in TestNeuralNumerical.pas, one family at a time.
- [ ] TNNetRMSNorm has now landed (neural/neuralnetwork.pas +
      TestNeuralNumerical.pas) and is, alongside TNNetLayerNorm, a ready
      template for any further normalization-layer variants.

#### Layers I'd enjoy building
- [ ] TNNetScaledDotProductAttention — the single-head core (Q·Kᵀ / √d → softmax
      → ·V) as a standalone, fully gradient-checked layer. Building block for the
      multi-head attention already on the list, but small enough to land and test
      on its own first.
- [ ] TNNetDropPath (stochastic depth) — already listed above; I'd like to take
      it. Identity at inference, whole-sample drop at training. Pairs well with
      a numerical-gradient test that fixes the RNG seed.
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
- [ ] TNNetRotaryEmbedding (RoPE) — apply rotary position encoding to a
      Q/K projection. Listed at the top of the file; I'd like to take it as a
      standalone, gradient-checked layer (the rotation is a fixed, parameter-free
      transform so the backward pass is just the transpose rotation).
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
- [ ] Continue the Backpropagate audit into the concat/split/branch family
      (TNNetConcat, TNNetDeepConcat, TNNetSplitChannels, TNNetSum) — one
      numerical-gradient test per layer, hunting for bugs the activation audit
      style turned up before.
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
- [ ] Add `tests/RunTests` (and any other fpc build artifacts under tests/) to
      .gitignore — the build leaves an untracked binary behind after each test
      run, which clutters `git status` for every future agent.
- [ ] Now that TNNetMaskedFill and the gated-activation family have landed, the
      remaining transformer building blocks are TNNetScaledDotProductAttention
      and TNNetRotaryEmbedding (RoPE) — both already listed above. With masking
      done, ScaledDotProductAttention is the highest-leverage next layer for
      the Tiny GPT example.
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
- [ ] TNNetScaledDotProductAttention — the single-head attention core
      (Q·Kᵀ / √d → softmax → ·V) as a standalone, gradient-checked layer.
      Highest-leverage missing transformer block now that TNNetMaskedFill,
      TNNetLayerNorm, TNNetRMSNorm and the gated activations have all landed.
      Should pair cleanly with TNNetMaskedFill for causal attention.
- [ ] TNNetRotaryEmbedding (RoPE) — parameter-free rotation applied to Q/K.
      Backward pass is just the inverse rotation, so the numerical-gradient
      test is straightforward. Companion piece to ScaledDotProductAttention.
- [ ] TNNetMultiHeadSelfAttention — once SDPA + RoPE land, compose them into
      a real multi-head block (split-heads → SDPA per head → concat → out
      projection). Add a numerical-gradient test on a tiny 2-head example.
- [ ] TNNetDropPath (stochastic depth) — identity at inference, whole-sample
      drop at training. Use a fixed RNG seed in the test so the masked
      forward/backward is deterministic.
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
- [ ] Continue the Backpropagate audit into the concat/split/branch family
      (TNNetConcat, TNNetDeepConcat, TNNetSplitChannels, TNNetSum) — one
      numerical-gradient test per layer. Listed above; I want to take it.
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
