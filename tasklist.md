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

- [ ] SoftCapping saturation test on extreme inputs (±1e6) — assert
      output stays within ±c and `Backpropagate` doesn't produce NaNs.
- [ ] DropPath determinism test: with a fixed `RandSeed`, two Compute
      calls produce identical masks/outputs across runs. Pin the
      RNG-reset behavior so future refactors can't silently break it.
- [ ] RoPE odd-depth guard test: assert constructing+wiring a
      `TNNetRotaryEmbedding` after a layer with odd Depth raises the
      expected error (validates the even-Depth precondition).
