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
- [x] Investigate TNNetPointwiseSoftMax.Backpropagate: it uses the diagonal-only
      x*(1-x) approximation instead of the full softmax Jacobian. Decide whether
      to implement the exact Jacobian (and add a numerical-gradient test) or
      document the approximation explicitly in the code/README.
      Done: replaced the diagonal y*(1-y) with the exact softmax Jacobian
      (per-(X,Y) over depth for TNNetPointwiseSoftMax; new global override
      for TNNetSoftMax). Added TestPointwiseSoftMaxExactJacobianGradientCheck
      and TestSoftMaxExactJacobianGradientCheck — both pass at 1e-2 against
      the central-difference check that the old approximation would have
      failed. Cross-entropy training paths should opt into
      SkipBackpropDerivative=1 (already the pattern in several examples).
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
- [x] Implement the exact softmax Jacobian for TNNetPointwiseSoftMax.Backpropagate
      (replacing the diagonal-only x*(1-x) approximation) and add a numerical-
      gradient test proving it. This is the unfinished thread from the second
      pass above — I'd like to actually close it.
      Done: TNNetPointwiseSoftMax.Backpropagate uses the O(N) per-group form
      y_i * (dL/dy_i - sum_j y_j * dL/dy_j) and TNNetSoftMax adds a global
      override using the same formula over the entire volume. Two new
      gradient-check tests in TestNeuralNumerical.pas pass at 1e-2.
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
- [x] TNNetALiBi positional bias — alternative to RoPE: add a static
      slope * (j - i) bias to attention scores. Pairs with TNNetMaskedFill
      and gives a second option for position handling in the eventual MHA.
      Landed: descends from TNNetIdentity, no constructor params, per-head
      slopes `slope[h] = 2^(-8*(h+1)/Depth)` precomputed at SetPrevLayer
      into a cached TNNetVolume freed in Destroy. Layout convention:
      SizeX = key position, SizeY = query position, Depth = head index;
      forward adds `slope[h] * (X - Y)` to every position; backward is
      the inherited TNNetIdentity gradient passthrough. Dispatched from
      both CreateLayer sites with no extra fields. Forward, central-
      difference gradient, and SerializationRoundTrip tests added to
      TestNeuralNumerical.
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
- [x] TNNetALiBi — finally pick up the position-bias entry above. It is
      the smallest possible "alternative to RoPE": precompute a single
      `slope[h]` per head and add `slope * (key_pos - query_pos)` into
      the attention score map before softmax. Parameter-free (slopes are
      deterministic from head count, per the paper), so the gradient
      check is the same shape as TNNetMaskedFill's.
      Landed: see the main TNNetALiBi entry above for details (layout
      `SizeX=key`, `SizeY=query`, `Depth=head`).
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
- [x] TNNetALiBi positional bias — still the smallest "alternative to
      RoPE" task on the list and now the most isolated next layer (no
      MHA dependency). Parameter-free, gradient check shape mirrors
      TNNetMaskedFill, and it slots into the eventual MHA helper.
      Landed: see the main TNNetALiBi entry near the top of this file.

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
- [x] TNNetALiBi — same entry as the previous two batches. The fact that
      it has been listed three times now is the universe telling me to
      take it. Per-head deterministic slopes (`slope[h] = 2^(-8h/H)`),
      adds `slope * (key_pos - query_pos)` into the attention score map
      before softmax. Parameter-free, gradient check mirrors MaskedFill.
      Landed: see the main TNNetALiBi entry near the top of this file.
- [ ] TNNetSwitchableNorm — a tiny learnable convex combination of
      LayerNorm and RMSNorm outputs (two learnable scalars summed via
      softmax). Cute, parameter-cheap experiment in "let the network
      pick its own normalizer". All ingredients already exist.
- [x] TNNetChannelShuffle — the ShuffleNet operation: reshape `(C)` to
      `(groups, C/groups)`, transpose, flatten back. Parameter-free,
      gradient is the inverse permutation, easy numerical check.
      Slots cleanly into the existing channel-manipulation family
      (TNNetInterleaveChannels, TNNetSplitChannels, etc.).
- [x] TNNetSoftmaxTemperature — `softmax(x / T)` with configurable T,
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
- [x] TNNetChannelShuffle inverse property test: ChannelShuffle(G)
      composed with ChannelShuffle(C/G) is the identity. Cute one-line
      property check on top of the existing forward test.
- [ ] Now that DropPath p=1 is strict-drop, add a tiny experiment that
      confirms a deep residual net with p=1 DropPath after every block
      collapses to the identity path (loss curve matches a no-residual
      baseline). Cheap teaching artifact for the fix.
- [x] Now that the norm round-trip suite covers Layer/RMS/Group, extend
      it to TNNetChannelStdNormalization and TNNetLocalResponseNorm2D
      (the older normalization layers that predate the audit) — same
      shared helper, two-line additions.

### Ideas added on 2026-05-15 (lucky seed 726151)

Coming in fresh after the DropPath-fix + ChannelShuffle + Norm-roundtrip
batch. MHA is still the headline next step, but several small,
self-contained items below would be a genuine pleasure to land. Each is
sized to fit in a single commit.

#### Quick wins I'd take first
- [x] TNNetChannelShuffle inverse property test (already listed above):
      assert that ChannelShuffle(G) composed with ChannelShuffle(C/G)
      is the identity over a random tensor. A one-screen test that
      pins the most surprising algebraic property of the new layer.
- [x] Extend the norm round-trip suite to TNNetChannelStdNormalization
      and TNNetLocalResponseNorm2D (also already listed): I want to
      actually ship the two-line additions and close the open thread.
- [ ] Filter + file:line patch for `scripts/list_untested_layers.sh`
      (the two follow-ups listed in the previous batch) — bundle them
      into one v1 of the script. Two-line awk change; would make
      every future audit task start with a sharper actionable list.

#### Layers I'd enjoy building (no MHA dependency)
- [x] TNNetALiBi — fourth time on the list. Treating its repeated
      appearance as a personal commitment device. Per-head deterministic
      slopes (`slope[h] = 2^(-8h/H)`), parameter-free, gradient check
      mirrors TNNetMaskedFill. The smallest "alternative to RoPE" task
      we have, and it slots straight into the eventual MHA helper.
      Landed: see the main TNNetALiBi entry near the top of this file.
- [x] TNNetSoftmaxTemperature — `softmax(x / T)` with configurable T,
      saved/loaded via FFloatSt[0]. Useful for the eventual tiny-GPT
      sampling demo. Already listed once; would enjoy taking it next.
- [ ] TNNetGatedResidual (ReZero gating) — already listed. The
      zero-init learnable gate is a tiny variation on LayerScale and
      provides a second lever for stabilizing deep residual stacks.
- [ ] TNNetBitLinear — the BitNet "ternary weight" experiment:
      forward uses sign(W)*scale, backward keeps full-precision
      shadow weights via the straight-through estimator. Tiny, fun,
      and a clean self-contained way to explore quantization-aware
      training without dragging in any new infra.
- [ ] TNNetDyT (Dynamic Tanh) — the Liu-et-al 2025 drop-in replacement
      for LayerNorm: `gamma * tanh(alpha * x) + beta`, with `alpha`
      a per-layer learnable scalar. Cheap, transformer-relevant, and
      a one-evening implementation given the existing LayerNorm/LayerScale
      templates. Numerical gradient check is straightforward.
- [x] TNNetMaxOut — the classic Goodfellow MaxOut activation: take K
      linear projections and reduce by max along the K axis. Easy to
      gradient-check (argmax is piecewise-constant), and a nostalgic
      addition that rounds out the activation menagerie.
      Landed: depth-grouped MaxOut layer (input depth = K * out depth),
      forward caches argmax per output cell, backward routes the
      gradient to the winning slab. Forward, numerical-gradient, and
      serialization round-trip tests in TestNeuralNumerical.
- [ ] TNNetPolynomialActivation — per-channel learnable degree-2
      polynomial `a*x^2 + b*x + c`. Three learnable params per channel,
      smooth analytic gradient. Fun "what if the activation were
      learned" experiment that pairs with the activation bake-off.

#### Composite blocks / examples
- [ ] TNNetPreNormResidual helper — already listed three times. Bundle
      it with TNNetGLUFeedForward (also listed) into a tiny
      "minimal-transformer-without-attention" example so the
      pre-norm-residual + GLU-FFN combo can be exercised end-to-end
      while MHA is still pending.
- [ ] Tiny "memorize a sentence" demo: train a 1-layer SDPA+RoPE
      model to perfectly memorize a 32-token sequence, print the
      training loss curve and the reconstructed sample. End-to-end
      exercise of the transformer layers that landed, no full MHA
      required (single head is fine for memorization).
- [ ] ShuffleNet-flavored CIFAR block example: 1x1 conv -> ReLU ->
      ChannelShuffle -> depthwise 3x3 conv -> 1x1 conv, swapped into
      one of the existing SimpleImage examples. Concrete end-to-end
      use of the new permutation layer (already listed above; I want
      to actually take it).

#### Experiments I'm curious about
- [ ] "Softmax temperature × generation quality" sweep (depends on
      TNNetSoftmaxTemperature above): generate samples from a tiny
      trained char model with `T ∈ {0.5, 0.8, 1.0, 1.2, 1.5}` and
      print the resulting strings. Concrete visualization of what
      temperature actually does.
- [ ] DyT vs LayerNorm bake-off (depends on TNNetDyT above): same
      tiny encoder block, swap the LayerNorm for DyT, chart final
      loss and wall-clock. Reproduces the headline claim of the
      DyT paper at toy scale.
- [ ] ChannelShuffle group-count sweep: with the inverse property
      test in hand, train the same tiny conv net with
      `groups ∈ {1, 2, 4, 8}` and chart accuracy. Pure-CPU
      illustration of the ShuffleNet group/accuracy trade-off.
- [ ] "Why bother with LayerScale?" experiment: train two deep MLPs
      (one with TNNetLayerScale, one without) on a toy regression
      task and chart per-layer gradient norms over training. Shows
      the γ trick doing its job at toy scale.
- [ ] First-batch gradient-norm heatmap across (depth, width, init):
      enumerate a small grid, print one number per cell. The
      cheapest possible "make the vanishing-gradient problem
      concrete" diagram, and a natural artifact for the eventual
      numerical-gradient testing doc.

#### Correctness / audit work
- [ ] SDPA all-masked-row policy: still open after three batches. I'd
      like to take the concrete proposal already on the list (detect
      the all-masked row in Compute, output a zero row, skip the
      softmax for it), document the choice in code, and add the
      pinning test. Close the open thread.
- [ ] Upsample/deconv Backpropagate audit (already listed) — TNNetUpsample
      first because it's the simplest, then TNNetDeconvolution, then
      TNNetDeMaxPool. One numerical-gradient test per layer, the
      same way the activation audit caught real bugs.
- [x] LoadFromString round-trip for TNNetChannelShuffle — the new
      layer carries a Groups parameter in FStruct[0]; the
      norm-roundtrip pass found zero bugs but we should still pin
      the dispatch for the just-landed layer before it has a chance
      to drift.
- [ ] Property-based gradient harness v0 (already listed): even a
      version that only randomizes input shape for the six most
      recently landed layers is enough to start catching shape-edge
      bugs. I want to commit to actually shipping the v0.

#### Tooling / dev experience
- [ ] `scripts/grep_layer.sh <TNNet...>` helper (already listed): print
      the class declaration, its Compute, its Backpropagate, and any
      test methods referencing it. Captures the "first 30 seconds
      after picking a layer to audit" workflow.
- [ ] Tiny `tests/SmokeTest.lpr` that builds + runs the five fastest
      gradient checks and exits in under a second (already listed).
      Lets the eventual CI shim start with a real signal even before
      RunAll.sh is wired in.
- [ ] `--quick` flag on the test runner (already listed twice): skip
      any test whose name contains `Slow` / `Large` / `LongSeq`
      markers, controlled by a single switch. Pairs with the existing
      RunAll.sh `--quick` idea.
- [ ] One-shot `scripts/audit_one_layer.sh <TNNet...>` that runs
      grep_layer.sh, list_untested_layers.sh, and the existing
      numerical-gradient test runner filtered to tests that mention
      the layer. Bundles the audit workflow into a single command.

#### Documentation
- [ ] "How numerical gradient testing works in this repo" note (already
      listed multiple times). I'd like to actually write it, since
      every audit task in this file relies on it. Cover the eps
      choice, central-differences math, tolerance picking, and the
      AssertGradientCheck helper pattern.
- [ ] One-pager "transformer building blocks landed in this repo"
      (already listed): table of LayerNorm / MaskedFill / SDPA / RoPE /
      SoftCapping / DropPath / GEGLU / SwiGLU / GLU / SquaredReLU /
      LayerScale / AddPositionalEmbedding / ChannelShuffle, one-line
      "what it is" + "use it when" per layer. Becomes the index entry
      into the eventual MHA + encoder-block helpers.
- [ ] Inline-comment cleanup pass on TNNetScaledDotProductAttention
      (already listed) — shape annotations on every loop, named
      strides, link to the planned annotated walkthrough. Pure
      readability win, no behavior change.
- [ ] "Picking a tolerance" mini-guide for numerical-gradient tests
      (already listed). Companion piece to the testing note above.
      Cover when 1e-2 is fine, when to tighten to 1e-3, and when the
      right move is to shrink eps instead of loosening the tolerance.

### Ideas added on 2026-05-15 (post ALiBi + SoftmaxTemperature + ChannelShuffle-audit batch)

This batch landed: TNNetALiBi (parameter-free per-head positional bias,
SizeX=key / SizeY=query / Depth=head layout, slopes = 2^(-8(h+1)/H)),
TNNetSoftmaxTemperature (softmax(x/T) with full softmax-Jacobian backprop
overriding the diagonal y*(1-y) approximation inherited from
TNNetSoftMax), TNNetChannelShuffle inverse-property + strengthened
round-trip tests, and norm round-trip extension to
TNNetChannelStdNormalization and TNNetLocalResponseNorm2D.

Notable finding: TNNetSoftmaxTemperature had to override Backpropagate
with the full softmax Jacobian because TNNetSoftMax/TNNetPointwiseSoftMax
use the diagonal y*(1-y) approximation (valid only when paired with
cross-entropy). This reinforces the still-open TODO at line 120
("implement exact softmax Jacobian for TNNetPointwiseSoftMax").
Done (post-batch): the TODO is now closed — TNNetPointwiseSoftMax and
TNNetSoftMax both use the full softmax Jacobian, gradient-checked.

Natural follow-ups:

- [ ] Position-encoding bake-off now unblocked: with RoPE, ALiBi and
      sinusoidal AddPositionalEmbedding all landed, the bake-off entry
      higher up in the file is finally fully unblocked. Same tiny seq
      model trained with (a) no position info, (b) sinusoidal, (c) RoPE,
      (d) ALiBi, printing final loss and a sample generation per scheme.
- [ ] ALiBi-with-MaskedFill composition test: stack TNNetMaskedFill on
      top of TNNetALiBi on a tiny SeqLen=4, Depth=2 input and assert the
      causal upper-triangle stays at -1e9 while the lower triangle picks
      up the ALiBi slope contribution. Pins the composition for the
      eventual MHA path.
- [ ] SoftmaxTemperature × generation experiment (already listed under
      lucky seed 51855): now actually buildable since the layer landed.
      Train a tiny char model, generate at T ∈ {0.5, 0.8, 1.0, 1.2, 1.5}.
- [x] Re-open exact-softmax-Jacobian work on TNNetPointwiseSoftMax: the
      SoftmaxTemperature implementation is effectively a working template
      for what the fix should look like. Closing this would let
      TNNetSoftmaxTemperature drop its override and inherit cleanly.
      Done: the Jacobian work landed. TNNetSoftmaxTemperature still keeps
      its own Backpropagate override because the 1/T chain-rule factor
      must scale only the layer's contribution to FPrevLayer.OutputError
      (which is additive), not the accumulated total — calling inherited
      and post-scaling would corrupt other branches' gradients. The
      override now reads as a near-copy of TNNetSoftMax.Backpropagate
      with an InvT multiplier; that small duplication is acceptable.

### Ideas added on 2026-05-15 (lucky seed 55717)

Coming in after the ALiBi + SoftmaxTemperature + ChannelShuffle-audit
batch. MHA remains the headline; the items below are things I would
personally enjoy taking on either as warm-ups or as parallel tracks.
Each is sized for a single focused commit.

#### Quick wins I'd take first
- [x] Exact softmax Jacobian for TNNetPointwiseSoftMax (the open TODO at
      line ~120 and the explicit follow-up at the end of the previous
      batch). TNNetSoftmaxTemperature.Backpropagate is now a ready
      template: copy the full-Jacobian inner loop, drop the 1/T scaling,
      and add a numerical-gradient test that the diagonal-only
      approximation currently fails. Let TNNetSoftmaxTemperature drop
      its override and inherit cleanly afterwards.
      Done: TNNetPointwiseSoftMax.Backpropagate now uses the exact
      Jacobian per-(X,Y) over depth, and a new TNNetSoftMax.Backpropagate
      override applies the same formula globally (matching its global
      softmax forward). TestPointwiseSoftMaxExactJacobianGradientCheck
      and TestSoftMaxExactJacobianGradientCheck pin the result at the
      standard 1e-2 tolerance. TNNetSoftmaxTemperature keeps its
      override (the 1/T factor must scale only the layer's contribution
      to FPrevLayer.OutputError, which is additive across branches).
- [ ] ALiBi-with-MaskedFill composition test (listed at the end of the
      previous batch): I want to actually take it. Stack
      TNNetMaskedFill on TNNetALiBi, assert upper triangle stays at
      -1e9 while the lower triangle picks up the slope contribution.
      ~30 lines in TestNeuralNumerical, pins the composition before MHA.
- [x] LoadFromString round-trip for TNNetSoftmaxTemperature — the new
      layer carries T in FFloatSt[0]; mirror the SoftCapping pattern
      and pin the dispatch before it has time to drift.
      Done: TestSoftmaxTemperatureSerializationRoundTrip in
      TestNeuralNumerical.pas.
- [x] LoadFromString round-trip for TNNetALiBi — parameter-free but the
      cached per-head slope volume is rebuilt at SetPrevLayer; a
      round-trip test pins that the rebuild fires on the deserialized
      layer. Done: TestALiBiSerializationRoundTrip in
      TestNeuralNumerical.pas.

#### Layers I'd enjoy building (no MHA dependency)
- [ ] TNNetReZero / TNNetGatedResidual — already on the list. Per-channel
      zero-initialized learnable gate `y = x + alpha * Sublayer(x)`.
      Tiny variation on LayerScale, complements PreNormResidual, and
      gives a second concrete stabilizer to bake-off against.
- [ ] TNNetDyT (Dynamic Tanh) — already listed under lucky seed 726151.
      `gamma * tanh(alpha * x) + beta`, per-layer learnable alpha plus
      per-channel gamma/beta. One-evening implementation given the
      LayerNorm + LayerScale templates already in tree. Numerical
      gradient test is straightforward.
- [ ] TNNetRMSNormGated — RMSNorm followed by a learnable per-channel
      sigmoid gate. Cheap "RMSNorm-with-attention-to-which-channels-matter"
      that pairs naturally with the eventual transformer FFN.
- [ ] TNNetGRN (Global Response Normalization, from ConvNeXt-V2): channel-
      wise contrast normalization with learnable scale/bias. Pure-CPU
      friendly, small, and an interesting alternative normalizer for
      the existing CIFAR conv examples.
- [ ] TNNetCosineSimilarityAttention — replace `Q·Kᵀ / √d` with
      `(Q/||Q||)·(K/||K||)ᵀ * scale`. A nice small variant of SDPA that
      lets us compare numerical stability against the standard dot-product
      formulation without reaching for SoftCapping.
- [x] TNNetTanhShrink and TNNetHardTanh — round out the activation
      menagerie alongside the already-landed SoftPlus / GaussianActivation
      / SquaredReLU. Closed-form derivatives, easy gradient checks.
      Landed: both classes descend from TNNetReLUBase, register in
      both CreateLayer dispatch sites, and ship with forward +
      numerical-gradient tests in TestNeuralNumerical (HardTanh check
      keeps inputs away from the +/-1 kinks).

#### Composite blocks / examples
- [ ] TNNetPreNormResidual helper — listed five times now. Treating
      that as universe pressure to actually ship the one-liner
      `y = x + Sublayer(LayerNorm(x))` builder.
- [ ] "Attention-free toy transformer" example: PreNormResidual +
      TokenShift (when it lands) + GLUFeedForward, no SDPA. A useful
      baseline to compare against the eventual MHA-based variant on
      the same toy next-token task.
- [ ] Tiny "echo the previous token" SDPA demo: train a single SDPA +
      RoPE layer with SeqLen=4 to output input[t-1] at position t.
      Smallest possible end-to-end test that attention is actually
      learning to look at the right key position. Produces a single
      attention-matrix PGM as artifact.

#### Experiments I'm curious about
- [ ] SoftmaxTemperature × diagonal-vs-exact-Jacobian study: train a
      tiny classifier with TNNetSoftmaxTemperature in the head, swap
      between the (current) exact Jacobian and the diagonal y*(1-y)
      approximation, and chart the convergence-quality gap. Concrete
      motivation for the open exact-Jacobian TODO.
- [ ] ALiBi slope-base sweep: vary the slope formula from the canonical
      `2^(-8h/H)` to `2^(-kh/H)` for `k ∈ {4, 6, 8, 12}` on a tiny
      next-token task and chart loss. Empirical check of the cargo-culted
      "8" constant in the paper.
- [ ] Position-encoding bake-off (now fully unblocked, listed in the
      previous batch): I'd genuinely enjoy taking this one. Four
      training runs, one figure, big teaching value.
- [ ] "Does ChannelShuffle help small models?" experiment: take the
      existing SimpleImage CIFAR example, drop in a 1x1 -> shuffle ->
      depthwise -> 1x1 block, and chart accuracy vs the baseline at
      matched parameter count. Honest empirical answer at toy scale.
- [ ] Activation-saturation visualizer: train a tiny net with Sigmoid /
      Tanh / HardSigmoid and print the fraction of saturated units per
      layer per epoch. A concrete picture of why ReLU/GELU win in
      practice, using only layers already in tree.

#### Correctness / audit work
- [x] TNNetUpsample numerical-gradient test (already listed twice as
      the kickoff for the upsample/deconv audit). Smallest of the
      three, would unblock the rest. Done: TestUpsampleGradientCheck
      in TestNeuralNumerical.pas on a 2x2x4 input shape (becomes 4x4x1
      after depth_to_space). No bugs found —
      TNNetUpsample.ComputePreviousLayerError correctly inverts the
      forward permutation.
- [ ] Audit which TNNet* classes still override Backpropagate but lack
      a numerical-gradient test, after the activation / concat-split /
      transform-pooling audits already done. Produce a fresh TODO list
      so the next contributor has actionable targets.
- [ ] Add an "attention numerical-gradient stress test" that runs the
      SDPA grad check across SeqLen ∈ {1, 2, 3, 5, 8} and asserts the
      max error vs tolerance at each. Pins shape-edge behavior the
      existing single-shape test can't see.

#### Tooling / dev experience
- [ ] `scripts/grep_layer.sh <TNNet...>` — listed three times now. Print
      the class declaration, Compute, Backpropagate, and any tests that
      mention the class. Captures the first 30 seconds of every audit
      I keep doing by hand.
- [ ] Filter + file:line patch for `scripts/list_untested_layers.sh`:
      drop names ending in Base/Class/Abstract, and emit a
      `file:line` pointer for every surviving entry. Two-line awk
      change in practice; would sharpen every future audit task.
- [ ] `scripts/new_layer.sh <Name>` scaffolder (already listed): drop a
      Compute/Backpropagate skeleton into neuralnetwork.pas plus a
      matching numerical-gradient test stub. Captures the layer-authoring
      checklist as executable form.

#### Documentation
- [ ] "How to add a new layer" cookbook: a single page walking through
      a real recent landing (e.g. TNNetSoftmaxTemperature), step by
      step, from constructor declaration through CreateLayer dispatch
      to the numerical-gradient test. Companion to the testing note.
- [ ] Annotated TNNetSoftmaxTemperature.Backpropagate walkthrough: the
      full softmax-Jacobian derivation with shapes on every line.
      Doubles as the template for the eventual exact-Jacobian fix on
      TNNetPointwiseSoftMax.
- [ ] Short "position encodings in this repo" comparison page covering
      sinusoidal AddPositionalEmbedding, RoPE, and ALiBi: when each is
      the right pick, what the layer expects on input, and a tiny
      code snippet for each. Becomes the natural companion to the
      bake-off experiment above.

### Ideas added on 2026-05-15 (post Upsample-grad + ALiBi-MaskedFill + exact-softmax-Jacobian batch)

This batch landed:
- TestALiBiMaskedFillComposition pinning the MaskedFill + ALiBi stack
  for the eventual MHA path.
- TestUpsampleGradientCheck — first entry in the upsample/deconv audit.
  No bugs found.
- Exact softmax Jacobian in both TNNetPointwiseSoftMax.Backpropagate
  AND TNNetSoftMax.Backpropagate (the latter newly overrides the
  inherited approximation). TNNetSoftmaxTemperature was kept as-is
  because its 1/T factor must scale only this layer's additive
  contribution to FPrevLayer.OutputError — a `inherited + post-scale`
  pattern would corrupt accumulated gradients from sibling branches.

Natural follow-ups:

- [ ] Continue the upsample/deconv audit: next up is TNNetDeMaxPool
      (the parent of TNNetUpsample — its ComputePreviousLayerError is
      what TNNetUpsample reuses indirectly). Same recipe as the
      Upsample test: tiny shape, LayerInputGradientCheck. Then
      TNNetDeconvolution to close out the family.
- [ ] TNNetSoftmaxTemperature cleanup attempt: prove (or disprove) that
      a shared softmax-Jacobian helper extracted from TNNetSoftMax /
      TNNetPointwiseSoftMax / TNNetSoftmaxTemperature would let all
      three reduce to one Backpropagate body parameterised by axis +
      inv-temperature. Pure refactor, gradient tests pin the behavior.
- [ ] Cross-entropy vs exact-Jacobian regression-style check: confirm
      that the existing classification examples (e.g. SimpleImage CIFAR)
      converge to the same loss curve they did before the
      TNNetSoftMax.Backpropagate change. The cross-entropy loss layer
      should be the path that benefits most from the exact Jacobian
      cancelling cleanly to `(y - target)` — verify no regression on
      a fast example.
- [ ] Now that the exact-Jacobian template exists in three places,
      audit any remaining TNNet* layers that compute a softmax-like
      normalization (search for "Exp(" near a normalization loop in
      neuralnetwork.pas) to confirm none still ship the diagonal-only
      approximation under another name.
- [ ] Numerical-gradient stress test for TNNetSoftMax / TNNetPointwiseSoftMax
      across SeqLen / Depth / SizeX combinations (pairs with the
      already-listed SDPA stress test idea) — pins the new exact-Jacobian
      code path across shape edge cases.

### Lucky-day batch — 2026-05-15 (seed 35900)

A fresh wave of ideas I'd personally enjoy working on, organised by how
much fun-vs-effort each looks like. Skewed toward "complete the
transformer story" since so many of the building blocks are now in tree
(SDPA, RoPE, MaskedFill, ALiBi, RMSNorm, LayerNorm, GEGLU/SwiGLU,
LayerScale, DropPath, exact softmax Jacobian).

#### Layers I'd enjoy building next
- [ ] TNNetMultiHeadSelfAttention — assemble the existing SDPA + RoPE +
      MaskedFill + dense projections into a real MHA block. Forward +
      numerical-gradient test on a small (SeqLen=4, Heads=2, d_k=4)
      shape. Top-level transformer task on the list; the per-piece
      gradient tests already pin every internal layer.
- [ ] TNNetTransformerEncoderBlock helper that wires
      MHA → residual+LayerNorm → SwiGLU-FFN → residual+LayerNorm.
      Pre-norm and post-norm variants behind a flag. Single
      numerical-gradient test against the composed block.
- [ ] TNNetCrossAttention — same SDPA core but with separate Q vs K|V
      input branches (so encoder-decoder is reachable). Forward +
      gradient-check on a tiny shape.
- [ ] TNNetSinkAttention / attention sinks — keep K positions 0..s-1
      always unmasked, mask the rest causally. Tiny variant on the
      existing MaskedFill that unlocks streaming-LLM experiments
      cheaply. Gradient-checked.
- [ ] TNNetTalkingHeads — pre/post softmax linear mix across heads (the
      "Talking-Heads Attention" trick). Worth a tiny standalone test
      before it ever lands inside MHA.
- [x] TNNetMaxOut — k-way max of linear projections. Classical, easy
      gradient (route to argmax), nice teaching example for the
      activation menagerie.
      Landed: see earlier TNNetMaxOut entry (depth-grouped variant,
      forward + numerical-gradient + round-trip tests).
- [ ] TNNetMishExact — current Mish uses the standard formulation, but a
      stable formulation for large |x| using softplus's stable form
      would parallel what the SoftPlus landing did. Pure correctness /
      numerical-stability win, drop-in.

#### Correctness / audit work I'd take next
- [ ] TNNetDeMaxPool numerical-gradient test (next entry in the
      upsample/deconv audit per the line above). Smallest of the
      remaining two, would unblock TNNetDeconvolution.
- [ ] TNNetDeconvolution numerical-gradient test — closes the
      upsample/deconv family audit. Input AND weight gradients
      (the weight path is the more interesting one).
- [ ] Recurrent-style layer audit: TNNetEmbedding's weight-gradient
      path and any of the "previous-layer-output as state" patterns —
      list the offenders first, then test one at a time.
- [ ] Shared `LayerInputAndWeightGradientCheck` helper in
      TestNeuralNumerical.pas (companion to the
      LayerInputGradientCheck helper idea already in the list).
      Three-line tests instead of copy-pasted blocks; would have
      saved time on the LayerScale + CellBias + CellMul landings.
- [ ] Find-or-falsify pass: scan neuralnetwork.pas for any
      `Backpropagate` override whose body is just
      `inherited;` plus a tiny tweak — flag the candidates for
      gradient-check coverage. The exact-softmax-Jacobian story
      teaches us how silent the diagonal-only bug class can be.

#### Experiments I'm curious about
- [ ] Position-encoding bake-off (now fully unblocked since
      RoPE/ALiBi/MaskedFill/AddPositionalEmbedding all exist):
      same toy next-token task, four runs, one chart. Already in
      the list — I'd happily take it.
- [ ] DropPath ablation: drop the existing tiny CNN/MLP example into
      a deeper-than-needed stack with DropPath probabilities
      {0.0, 0.1, 0.2}, chart final accuracy and training-time
      variance. Empirical case for the layer that just landed.
- [ ] "Does exact-softmax-Jacobian matter?" controlled experiment:
      run the same classification example twice — once with the
      new exact Jacobian (current master), once with the old
      diagonal approximation restored on a branch. Chart the
      convergence-quality gap. Concrete evidence for the change.
- [ ] LayerNorm vs RMSNorm convergence-speed table on a tiny
      transformer-shaped MLP. RMSNorm is the cheaper sibling; the
      experiment justifies (or disproves) the cost saving at this
      scale. Pairs cleanly with the normalization bake-off above.
- [ ] Activation-function bake-off (already in the list): I'd take
      it next-to-RMSNorm-experiment since both use the same harness
      and produce one chart each.
- [ ] Attention-pattern visualiser: train the "echo previous token"
      SDPA demo from the previous batch, dump the full SeqLen×SeqLen
      attention matrix as PGM at each epoch, watch the diagonal
      shift one column. Tiny artifact, big teaching value.

#### Examples I'd enjoy writing
- [ ] Tiny GPT char-level transformer (already at the top of the
      list — every building block now exists). Train on a few
      kilobytes of text; print a sampled continuation. Single
      ~150-line example showing the library can stand up a real
      transformer end-to-end on CPU.
- [ ] "Learn to copy" toy: SeqLen=8 input → output the same
      sequence, trained with the encoder block above. Smallest
      end-to-end transformer task with an obvious right answer.
- [ ] "Learn to reverse" toy: same shape, output reversed. One
      bit harder than copy — needs full self-attention, not just
      identity. Pairs with the copy example for a teaching arc.
- [ ] "Learn binary addition" example (already in the list as a
      sequence task). I'd take it specifically because the right
      answer is exact, so any divergence is a real bug rather
      than a stochastic miss.
- [ ] Tokenizer + embedding-NN visualisation (already in the list):
      I'd enjoy the embedding-cluster bit specifically — print
      nearest neighbours of a handful of tokens, no plotting
      required.

#### Tooling / dev experience
- [ ] `scripts/grep_layer.sh <TNNet...>` — listed multiple times,
      I'd actually write it now. Print the declaration, Compute,
      Backpropagate, and any tests that mention the class. Saves
      the first 30 seconds of every audit.
- [ ] `scripts/list_untested_layers.sh` filter pass already in the
      list: drop Base/Class/Abstract names, emit file:line for
      surviving entries. Two-line awk change.
- [ ] `scripts/new_layer.sh <Name>` scaffolder (listed three+ times):
      Compute/Backpropagate skeleton into neuralnetwork.pas plus a
      numerical-gradient test stub. Encodes the checklist as
      executable form.
- [ ] Tiny benchmark microharness for Backpropagate cost across
      the softmax family — three layers, three shapes, one
      printed table. Validates that the exact-Jacobian change
      didn't tank performance on the common shapes.

#### Documentation
- [ ] "How numerical gradient testing works in this repo" note
      (already in the list as the missing safety-net doc). Single
      page; I'd anchor it to the recent TNNetMaskedFill landing
      as the worked example.
- [ ] "Building a transformer in this repo, layer by layer" — once
      the MHA + encoder-block tasks above land, write the
      walkthrough that strings every existing layer together into
      one teaching artifact. Complement to the "Tiny GPT" example.
- [ ] Short "softmax variants in this repo" note: TNNetSoftMax,
      TNNetPointwiseSoftMax, TNNetSoftmaxTemperature — when to
      pick each, what axis each operates on, which now use the
      exact Jacobian. Naturally companions the existing
      "position encodings in this repo" doc idea above.

### Lucky-day batch — 2026-05-15 (post MaxOut + TanhShrink/HardTanh + DeMaxPool-bug)

This batch landed:
- TNNetTanhShrink (`y = x - tanh(x)`) and TNNetHardTanh (`y = clamp(x,-1,1)`),
  both as activation layers descending from TNNetReLUBase, with forward +
  numerical-gradient tests in TestNeuralNumerical.pas.
- TNNetMaxOut (Goodfellow K-way max): depth-grouped variant — input depth D
  must be divisible by K, output depth = D/K, output is the elementwise max
  across the K depth slabs. Argmax cached per output cell for backward.
  Forward, numerical-gradient, and serialization round-trip tests.

Notable finding (NOT fixed, flagged for triage):

- [x] **BUG: TNNetDeMaxPool.ComputePreviousLayerError gradient is off by a
      factor of FPoolSize.** FIXED: removed the `FOutputError.Divi(floatPoolSize)`
      pre-scaling. Added `TestDeMaxPoolGradientCheck`, `TestDeAvgPoolGradientCheck`,
      and `TestDeMaxPoolForwardReplication` in `tests/TestNeuralNumerical.pas`. Forward replicates each input cell into a
      `PoolSize x PoolSize` output block (`neuralnetwork.pas:11471-11487`),
      so the correct input gradient is the SUM of the block's output errors.
      Current code (`neuralnetwork.pas:11552`) does
      `if (FSpacing=0) then FOutputError.Divi(floatPoolSize);` before the
      accumulating loop — dividing by `PoolSize`, not `PoolSize*PoolSize`,
      and arguably it should not divide at all for a pure-replication
      forward. Audit attempt added `TestDeMaxPoolGradientCheck` and it
      failed at PoolSize=2 with numerical=4.90 vs analytical=2.45
      (exactly 2x off). The failing test was reverted (not committed)
      per the audit-bug protocol. TNNetUpsample inherits from
      TNNetDeMaxPool but overrides both Compute and
      ComputePreviousLayerError, so its existing
      `TestUpsampleGradientCheck` is unaffected. TNNetDeAvgPool DOES
      inherit the buggy backward.
      Follow-up plan:
      (a) Decide the correct backward (likely: remove the `Divi` call
          entirely so the sum-of-block matches the pure-replication
          forward).
      (b) Check whether any in-tree example/training relies on the
          current scaling. `TNNetDeMaxPool` is used at
          neuralnetwork.pas:10561 in at least one example. TNNetDeAvgPool
          semantics (if it should average rather than replicate) may
          warrant a different forward instead.
      (c) Apply the fix, re-add `TestDeMaxPoolGradientCheck`, and
          additionally add a forward shape/replication test plus
          `TestDeAvgPoolGradientCheck`.

Natural follow-ups (non-bug):

- [x] HardTanh saturation test on extreme inputs (±1e6) — assert
      output stays within ±1 and `Backpropagate` doesn't produce NaNs
      (mirroring the TNNetSoftCapping saturation test pattern).
- [x] TanhShrink × Tanh composition sanity: `TanhShrink(x) + tanh(x)`
      should reconstruct `x` to within fp tolerance, on a tiny random
      input. One-liner check on top of the existing primitives.
- [ ] TNNetMaxOut serialization-after-wire test: build a small net
      with MaxOut in the middle (so SetPrevLayer fires post-load),
      save/load the whole net via `TNNet.SaveToString`/`LoadFromString`,
      and assert Compute matches end-to-end. The existing
      `TestMaxOutSerializationRoundTrip` only round-trips the single
      layer, not the net.
- [x] MaxOut "depth not divisible by K" guard test: assert that wiring
      a `TNNetMaxOut(K)` after a layer whose Depth is not a multiple of
      K raises the expected error (validates the divisibility
      precondition, mirroring the RoPE odd-depth guard test pattern).
- [ ] TNNetMaxOut CIFAR-style example wired into one of the existing
      SimpleImage paths — a real end-to-end use of the new activation,
      similar to the still-open ChannelShuffle integration task.
- [ ] Activation menagerie bake-off (already on the list) is now sweeter
      with TanhShrink/HardTanh/MaxOut in the menu; whoever picks up the
      bake-off should include the three new activations.

### Lucky-day batch — 2026-05-15 (seed 38154, second pass)

A grab-bag of things I'd personally enjoy building. Each is scoped to
land in one sitting and pairs with a numerical-gradient test or an
end-to-end smoke check.

#### Bug-fix follow-through I'd take first
- [x] **Fix TNNetDeMaxPool gradient (see bug entry above).** Concrete
      plan: delete the `FOutputError.Divi(floatPoolSize)` line at
      `neuralnetwork.pas:11552` (the sum-of-block backward matches the
      pure-replication forward without any scaling). Re-add
      `TestDeMaxPoolGradientCheck` from the reverted audit, plus a
      shape/replication forward test, plus `TestDeAvgPoolGradientCheck`
      (TNNetDeAvgPool inherits the same buggy backward today). Audit
      every in-tree call site of TNNetDeMaxPool / TNNetDeAvgPool before
      shipping — the only known one is around `neuralnetwork.pas:10561`
      in an example. If TNNetDeAvgPool semantics should AVERAGE rather
      than REPLICATE, change the forward instead of the backward.

#### Small activations I'd enjoy adding
- [ ] TNNetSoftPlus — `y = ln(1 + exp(x))`, the smooth ReLU. Stable
      implementation: for `x > 20` return `x`, for `x < -20` return
      `exp(x)`. Numerical-gradient test on a range that exercises both
      stable branches. Companion to TanhShrink/HardTanh — descend from
      TNNetReLUBase, plug into both CreateLayer dispatches.
- [x] TNNetELU — `y = x if x>0 else alpha*(exp(x)-1)`, configurable
      alpha (default 1.0). Backward reuses the cached output via
      `dy/dx = 1 if x>0 else y + alpha`. Numerical-gradient test plus
      a constructor round-trip via SaveToString/LoadFromString.
- [X] TNNetCELU — continuously differentiable ELU variant, `y = max(0,x)
      + min(0, alpha*(exp(x/alpha)-1))`. Same harness as TNNetELU; the
      difference is one extra division. Cheap, complete the family.
- [x] TNNetReLU6 — `y = clamp(x, 0, 6)`, the MobileNet activation. Tiny;
      forward is one min/max, backward is a 0/1 mask. Numerical-gradient
      test plus a saturation-at-extreme-inputs check (mirror the
      TNNetSoftCapping pattern).
- [ ] TNNetSquaredReLU — `y = max(0,x)^2`, the Primer-paper activation.
      Backward `dy/dx = 2*max(0,x)`. Worth landing because the squaring
      changes gradient magnitude in a way numerical-gradient tests
      should catch if a future refactor breaks it.
- [x] TNNetSiLU — alias / synonym for Swish with beta=1, registered so
      LoadFromString accepts the canonical name. One-liner; documents
      the equivalence in code rather than only in comments.

#### Pooling / shape layers I'd enjoy adding
- [x] TNNetGlobalMaxPool — companion to the existing GlobalAvgPool. One
      output cell per channel = max over the (X,Y) plane. Argmax cached
      for backward (gradient passes through the single argmax cell).
      Numerical-gradient test on a small random input.
- [ ] TNNetPixelShuffle / sub-pixel convolution — output spatial size
      = input * r, output channels = input / (r*r). The standard
      super-resolution upsample. Forward is a deterministic index
      permutation; backward is its inverse. Shape test + numerical-
      gradient test. Useful for the SuperResolution example.
- [ ] TNNetSpaceToDepth + TNNetDepthToSpace — inverse pair, both
      deterministic permutations. Numerical-gradient tests are trivial
      because the forward is linear, but landing them gives us the
      Vision Transformer "patchify" step as a one-line layer.
- [ ] TNNetLpPool — generalized pooling `(mean(|x|^p))^(1/p)` with
      configurable p (p=1 is L1, p=2 is L2, p→∞ approximates max).
      Backward via the chain rule on `|x|^p`. Numerical-gradient test
      at p=2 (the most useful setting in practice).

#### Test coverage I'd enjoy filling in
- [ ] Continue the Backpropagate audit on the still-uncovered families
      called out above: upsampling/deconvolution (TNNetDeconvolution,
      TNNetDeMaxPool, TNNetDeAvgPool, TNNetUpsample — partial coverage)
      and recurrent-style layers. Add one family per sitting. Companion
      to the DeMaxPool bug fix above.
- [ ] TNNetUpsampleNearest backward consistency: assert that summing
      the per-block output errors equals the input error (mirrors the
      pure-replication invariant the DeMaxPool fix should restore).
- [ ] Numerical-gradient cross-check helper: a single function in
      TestNeuralNumerical.pas that takes a layer factory + an input
      shape and runs both the input-gradient and (where applicable)
      weight-gradient central-difference checks. Removes ~10 lines of
      boilerplate from every new activation/normalization test.
- [ ] Deterministic-seed smoke test: build a small net (Conv → ReLU →
      Pool → Dense), train for 50 steps on a fixed-seed random dataset,
      assert the final loss matches a recorded value to 1e-4. Catches
      cross-platform numerical drift that the unit tests miss.

#### Training utilities I'd enjoy adding
- [ ] Cosine-annealing LR schedule helper (callable each epoch, takes
      `epoch`, `total_epochs`, `lr_max`, `lr_min`, returns the current
      LR). Tiny pure-math function; pairs naturally with an example
      that demonstrates the schedule on a CIFAR run.
- [ ] Linear warmup + cosine decay schedule — the standard transformer
      LR curve. Same shape as above; one extra parameter (warmup
      steps). Document in the README alongside the existing LR helpers.
- [ ] Label-smoothing helper: takes a one-hot target volume and a
      smoothing factor `eps`, returns the smoothed target volume that
      can be fed into the existing softmax-cross-entropy fit path.
      Standard regularization missing from the toolbox.
- [ ] Mixup data augmentation helper: takes two (input, target) pairs
      and a lambda from a Beta distribution, returns the convex
      combination. One short function; demonstrates with a CIFAR run.

#### Examples I'd enjoy writing
- [ ] "Smallest possible MaxOut net": one-layer MaxOut(K=2) on a 2D
      toy classification problem with two interleaved spirals.
      Visualises the piecewise-linear decision boundary as PGM. Tiny
      teaching artifact for the new layer.
- [ ] "Activation menagerie demo" mini-example: same fixed seed, same
      tiny CIFAR-stub net, swap the activation each run, print
      (final loss, wall-clock) into one table. Cheap qualitative
      bake-off — narrower than the full bake-off task, can land
      separately.
- [ ] "Sub-pixel super-res toy": once TNNetPixelShuffle lands, wire
      it into a 3-layer net that learns to 2x-upsample 8x8 random
      checkerboards to 16x16. Smallest possible end-to-end use of
      the new layer.

#### Tooling / dev experience
- [ ] `scripts/grep_layer.sh <TNNet...>` — print the declaration,
      Compute, Backpropagate, and every test that mentions the class.
      Listed multiple times across the lucky-day batches; I'd take
      it on the first DeMaxPool audit so it pays for itself
      immediately.
- [ ] `scripts/list_activations.sh` — emit the list of TNNetReLUBase
      descendants and which ones have a corresponding `Test*Gradient`
      check in TestNeuralNumerical.pas. Two-line grep + comm.
- [ ] `scripts/new_activation.sh <Name>` scaffolder — drop in a
      forward/backward skeleton (override Compute + Backpropagate
      with `Self.FOutput.CopyFrom(pPrevLayer.Output);`), a
      LoadFromString/CreateLayer registration line, and a
      numerical-gradient test stub. Encodes the
      "add a new activation" checklist.

#### Documentation
- [ ] "Activation layers in this repo" reference page: one table,
      one row per activation, columns = formula, derivative, range,
      typical use, file:line of the Compute method. Naturally
      grows as the small-activations batch above lands.
- [ ] "How to add a new layer" walkthrough — anchor it to one of
      the recent landings (TNNetTanhShrink is the smallest worked
      example). Companion to the scaffolder script above.
- [ ] "Gradient-check failure triage protocol" note: when a new
      numerical-gradient test fails, the bug is almost always in
      the layer under test, not the test harness. Document the
      DeMaxPool case as the canonical worked example of "audit
      first, revert the failing test, file the bug, fix
      separately." This is the audit-bug protocol the recent
      commits keep referencing — worth writing down.

### Lucky-day batch — 2026-05-15 (post DeMaxPool-fix + ReLU6/GlobalMaxPool + small-correctness batch)

This batch landed:
- TNNetDeMaxPool gradient bug fixed: removed the spurious
  `FOutputError.Divi(floatPoolSize)` in ComputePreviousLayerError so the
  sum-of-block backward matches the pure-replication forward. TNNetDeAvgPool
  (which just inherits) is fixed transitively. New tests:
  TestDeMaxPoolGradientCheck, TestDeAvgPoolGradientCheck, and
  TestDeMaxPoolForwardReplication. Needed a Double-precision local SSE
  accumulator (DeMaxPoolFamilyGradientCheck) because the generic helper
  suffered catastrophic single-precision cancellation when many output
  cells receive the same large value.
- TNNetReLU6 coverage: discovered the layer already existed (as a
  TNNetReLUL subclass with leakiness=0). Added the missing
  TestReLU6Forward and TestReLU6ExtremeInputSaturation tests.
- TNNetGlobalMaxPool (new layer): per-channel max over (SizeX,SizeY),
  argmax cached for backward. Forward and gradient tests landed.
- TestHardTanhExtremeInputSaturation, TestTanhShrinkTanhComposition
  (`TanhShrink(x) + tanh(x) == x`), and TestMaxOutDepthNotDivisibleByKGuard.

Notable lesson — the **single-precision SSE accumulator pitfall**: the
generic LayerInputGradientCheck sums per-cell squared errors in
TNeuralFloat. For the DeMaxPool family the replication step makes many
output cells share the same large value, and the squared loss exceeds the
FP32 mantissa's accumulation precision long before the central-difference
step matters. The DeMaxPool tests work around it with a Double-precision
local helper. **Generalising this fix into the shared gradient-check
helper would prevent every future replication/upsample layer audit from
hitting the same wall.**

#### Bug-class follow-ups I'd take first
- [ ] Promote DeMaxPoolFamilyGradientCheck's Double-precision SSE
      accumulator into the shared gradient-check helper in
      TestNeuralNumerical.pas (LayerInputGradientCheck and the
      weight-gradient variant). Sum the SSE in Double; the eps and
      tolerance stay TNeuralFloat. Once landed, drop the DeMaxPool-
      specific helper and confirm the DeMaxPool/DeAvgPool tests still
      pass. Future audits of upsampling/deconvolution/replication
      layers will not silently fail from FP32 cancellation.
- [ ] Close the upsample/deconv audit: TNNetDeconvolution numerical-
      gradient test (input AND weight gradients). Last entry in the
      family — DeMaxPool, DeAvgPool, Upsample all now have coverage.
      Likely benefits from the Double-precision helper above.

#### Layers I'd enjoy building next
- [ ] TNNetGlobalMaxPool follow-ups, now that the layer landed:
      - Serialization round-trip test (mirror the SoftCapping pattern).
      - Argmax-tie behaviour test: when two cells share the max value,
        the deterministic tie-break (likely "first wins") should be
        documented in code and pinned with a tiny test.
      - CIFAR-style example replacing a TNNetGlobalAvgPool head with
        TNNetGlobalMaxPool on one of the SimpleImage runs — small
        empirical "does it matter?" data point.
- [x] TNNetELU — `y = x if x>0 else alpha*(exp(x)-1)`, configurable
      alpha. Backward via cached output. Sits next to the TNNetReLU6
      coverage just added; same harness shape.
- [X] TNNetCELU — continuously differentiable ELU variant. One line
      different from TNNetELU; rounds out the family.
- [x] TNNetSiLU alias for Swish(beta=1). One-line LoadFromString
      registration so the canonical name parses. Pure naming cleanup.
- [ ] TNNetPixelShuffle (sub-pixel convolution). Forward = deterministic
      index permutation; backward = inverse. Useful for the
      SuperResolution example. Gradient check is trivial since the
      forward is linear.
- [ ] TNNetSpaceToDepth + TNNetDepthToSpace pair (inverse permutations).
      Tiny, deterministic, and unlocks the ViT "patchify" step as a
      one-line layer.

#### Correctness / audit work
- [ ] Re-validate the in-tree examples that use TNNetDeMaxPool /
      TNNetDeAvgPool after the gradient fix: the DenseNet helper at
      neuralnetwork.pas:~9383, examples/VisualGAN, examples/SuperResolution.
      The fix increases backward magnitude by `PoolSize` (=2 in practice),
      so the existing learning rates may be off by 2x. Run each example
      for a handful of epochs and confirm it still converges; flag any
      that need an LR retune.
- [x] LoadFromString round-trip for TNNetReLU6 and TNNetGlobalMaxPool —
      mirror the existing round-trip pattern (TNNetReLU6 is actually a
      TNNetReLUL with `Threshold=6, Leakiness=0`, so the round-trip
      mostly verifies the registration dispatch returns the right class).
- [x] TNNetGlobalAvgPool numerical-gradient test (companion to the
      new TNNetGlobalMaxPool test). Both share the shape transformation;
      the AvgPool variant is simpler and worth pinning while the
      GlobalMaxPool harness is fresh.

#### Experiments I'm curious about
- [ ] GlobalMaxPool vs GlobalAvgPool head bake-off on one of the
      SimpleImage CIFAR examples — same net, swap the head, chart
      validation accuracy. Cheap, visible data point.
- [ ] DeMaxPool fix regression check: train one of the affected
      examples (VisualGAN or SuperResolution) for a handful of epochs
      pre- and post-fix and chart the loss curve. Confirms the
      correctness fix doesn't tank training in practice.

#### Tooling / dev experience
- [ ] Add a "FP32 SSE accumulator warning" comment near
      LayerInputGradientCheck in TestNeuralNumerical.pas that points
      future audits at the DeMaxPool case and the Double-precision
      workaround. Useful even before the helper itself is upgraded.

## Lucky-day batch (seed 927654) — ideas I'd enjoy taking on

#### Activation layers I'd like to add (small, gradient-checkable)
- [x] TNNetELU — `y = x if x>0 else alpha*(exp(x)-1)`, configurable
      alpha (default 1.0). Backward via cached output: `dy/dx = 1`
      when `x>0`, else `y + alpha`. Mirrors the TNNetReLU6 harness
      shape; add LoadFromString registration and a numerical-gradient
      test in TestNeuralNumerical.pas.
- [X] TNNetCELU — continuously differentiable ELU; `y = max(0,x) +
      min(0, alpha*(exp(x/alpha)-1))`. One-line variant of TNNetELU.
- [x] TNNetSiLU — pure-naming alias for Swish(beta=1). Just a
      LoadFromString registration so the canonical PyTorch/JAX name
      parses without surprising the user. Document the equivalence.
- [ ] TNNetSoftPlus — `y = ln(1+exp(x))` with the standard
      large-x linearization (`x` when `x>20`) to avoid overflow.
      Backward = sigmoid(x). Numerical-gradient test, plus an
      identity-vs-Swish unit-test confirming SoftPlus(0)=ln(2).
- [x] TNNetSoftSign — `y = x / (1 + |x|)`. Cheap, smooth, no exp.
      Backward = `1 / (1+|x|)^2`. Add to the activation gradient-
      check sweep alongside the SELU/HardSigmoid additions.

#### Permutation/reshape layers (deterministic, easy to land)
- [ ] TNNetPixelShuffle (sub-pixel convolution). Forward is a pure
      index permutation: `(C*r*r, H, W)` → `(C, H*r, W*r)`. Backward
      is the inverse permutation. Useful for the SuperResolution
      example as a faster Upsample alternative. Gradient check is
      trivial since the forward is linear.
- [ ] TNNetSpaceToDepth + TNNetDepthToSpace pair (inverse
      permutations). `(C, H, W)` ↔ `(C*r*r, H/r, W/r)`. Tiny,
      deterministic, and unlocks the ViT "patchify" step as a
      one-line layer. Add a round-trip test that composes both and
      confirms the identity at the volume level.
- [ ] TNNetChannelShuffle — the ShuffleNet primitive. Splits
      channels into G groups and interleaves them. Pure permutation;
      forward and backward are mirror-image gathers. One numerical-
      gradient test plus a small "compose twice with same G returns
      identity-ish" sanity check.

#### Normalization follow-ups
- [ ] TNNetWeightStandardization — wrapper that normalizes a
      Conv/Dense layer's weights to zero-mean / unit-variance per
      output channel before the forward pass. Pairs especially well
      with TNNetGroupNorm. Gradient check via the existing
      LayerWeightGradientCheck helper.
- [x] TNNetInstanceNorm — per-sample, per-channel normalization
      (the GroupNorm limit at G=C). The TNNetGroupNorm code is the
      ready template; this is essentially a one-line constructor
      override plus a dedicated test in TestNeuralNumerical.pas.
      Landed: subclass of TNNetGroupNorm; SetPrevLayer sets
      Groups := Depth so each channel is its own group. Forward /
      gradient-check / serialization round-trip tests.

#### Attention / transformer building blocks
- [ ] TNNetMultiHeadSelfAttention — wrap the existing
      TNNetScaledDotProductAttention with a head-split / head-concat
      reshape pair. Parameter-free layer (the Q/K/V projections stay
      external Dense layers, mirroring the SDPA convention). Add a
      forward sanity test plus a small grad-check on a 2-head, d_k=4
      shape so the test stays fast.
- [ ] TNNetCausalMask helper — pure forward layer that adds a
      causal additive mask of `-1e9` to the upper triangle. Lets
      users build masked attention without baking the flag into
      SDPA. Trivial to test (forward only).
- [ ] Tiny char-level transformer example using LayerNorm + RoPE +
      SDPA + GEGLU + LayerScale, all of which already landed. Train
      on a tiny corpus (Shakespeare snippet or the bin/ examples'
      built-in text) for a small number of steps to demonstrate the
      stack composes end-to-end. Print sample completions.

#### Tests & audit follow-ups
- [x] TNNetGlobalAvgPool numerical-gradient test (companion to the
      new TNNetGlobalMaxPool test). Same shape transformation;
      forward is even simpler than the max variant. Worth pinning
      while the GlobalMaxPool harness is still fresh.
- [X] LoadFromString round-trip tests for the recently-landed
      activation/pooling additions: TNNetReLU6, TNNetGlobalMaxPool,
      TNNetSwiGLU, TNNetGEGLU, TNNetLayerScale, TNNetRMSNorm,
      TNNetDropPath. Mirror the SoftCapping pattern. One small
      TestSuite entry that builds, serializes, deserializes, and
      checks the layer class survives the round-trip.
- [ ] Promote DeMaxPoolFamilyGradientCheck's Double-precision SSE
      accumulator into the shared LayerInputGradientCheck /
      LayerWeightGradientCheck helpers (see also the bug-class
      follow-up above). Listed again here because it would unblock
      every audit in this batch.
- [ ] Argmax-tie behaviour test for TNNetGlobalMaxPool: when two
      cells share the max, the deterministic tie-break (likely
      "first wins") should be documented in code and pinned with a
      tiny test.

#### Examples / experiments I'm curious about
- [ ] GlobalMaxPool vs GlobalAvgPool head bake-off on one of the
      SimpleImage CIFAR examples — same net, swap the head, chart
      validation accuracy across a few seeds. Cheap, visible data
      point on whether the new pooling head matters in practice.
- [ ] Activation bake-off mini-experiment: same small CIFAR net,
      swap ReLU → ReLU6 → GELU → Swish → Mish → SwiGLU-as-block
      one at a time. Report final accuracy and wall-clock per epoch
      in a markdown table at examples/<dir>/README.md.
- [ ] LayerScale ablation on a deep residual stack — turn the
      learnable γ on/off, confirm the deeper net trains stably
      with it and diverges (or trains slower) without it. Single
      numeric data point in the README is enough.
- [ ] Tiny "hello attention" toy task: copy-task or reverse-string
      task that a 1-layer SDPA + small MLP solves perfectly.
      Demonstrates the attention layer pipeline end-to-end without
      needing a real corpus.

#### Tooling / dev experience
- [ ] Volume unit micro-benchmark printing ns/op for Add, Mul,
      DotProduct, and the new normalization layers (LayerNorm,
      RMSNorm, GroupNorm). One small bin/ entry that runs without
      OpenCL/AVX hardware differences and writes a CSV so future
      regressions are visible.
- [ ] Layer-by-layer forward-pass timing helper: a debug method on
      TNNet that prints per-layer wall-clock for one inference,
      so users can spot the slow layer without an external profiler.
      Pure additive, no behaviour change.
- [ ] README "supported layers" table auto-generated from the
      LoadFromString dispatch — single source of truth so newly
      registered layers always show up in docs. Could start as a
      Pascal helper that emits markdown to stdout.

### Lucky-day batch — 2026-05-15 (post ELU/SiLU/SoftSign + GlobalAvgPool-grad landings)

This batch landed:
- TNNetELU (configurable alpha, cached-output backward) + TNNetSiLU alias
  for Swish(beta=1). Forward / gradient / serialization round-trip tests
  in TestNeuralNumerical.pas.
- TNNetSoftSign (`y = x/(1+|x|)`), parameter-free, cached
  `1/(1+|x|)^2` derivative for a single-multiply backward. Forward /
  gradient / serialization round-trip tests.
- Numerical-gradient test for TNNetAvgChannel (the existing
  global-avg-pool implementation) and serialization round-trip tests for
  TNNetReLU6 and TNNetGlobalMaxPool.

Notable finding: there is no `TNNetGlobalAvgPool` class — the
global-avg-pool role is filled by `TNNetAvgChannel`. The earlier
"GlobalAvgPool numerical-gradient test" entries actually target
TNNetAvgChannel; this is now covered.

#### Small follow-ups
- [X] Add a `TNNetGlobalAvgPool = TNNetAvgChannel` type alias (mirroring
      `TNNetGlobalMaxPool` naming) so the LoadFromString dispatch and
      future docs match the canonical Keras/PyTorch name. One-line type
      decl + dispatch registration + a round-trip test.
- [X] TNNetCELU — continuously differentiable ELU (`y = max(0,x) +
      min(0, alpha*(exp(x/alpha)-1))`). With TNNetELU now landed, this
      is a one-method variant; reuse the ELU test harness shape.
- [x] TNNetSoftPlus identity-vs-Swish unit test: confirm
      `SoftPlus(0) = ln(2)` and that the large-x linearization branch
      kicks in correctly. The layer already exists; this is a tiny
      coverage gap. Landed: TestSoftPlusIdentityAtZero,
      TestSoftPlusLargeXLinearization and TestSoftPlusExtremeInputSaturation
      in TestNeuralNumerical.pas (suite 376 -> 379).
- [X] Continue the LoadFromString round-trip sweep called out around
      line 1748: TNNetSwiGLU, TNNetGEGLU, TNNetLayerScale, TNNetRMSNorm,
      TNNetDropPath. TNNetReLU6 and TNNetGlobalMaxPool are now done; the
      remaining five share the same harness shape and can land as one
      commit each (or one bundled commit).
- [ ] TNNetSoftSign saturation test on ±1e6: assert `|y| < 1` and
      Backpropagate doesn't NaN. Mirrors the HardTanh / SoftCapping
      saturation tests; closes the "extreme inputs" coverage gap for
      the new layer.

### Lucky-day batch — 2026-05-15 (seed 401988)

Ideas I'd enjoy tackling next. Mix of bite-sized layer additions,
audit follow-ups, and small experiments that exercise the layers that
have landed recently.

#### Tiny new layers (each ~one-method + one numerical-gradient test)
- [X] TNNetCELU — continuously differentiable ELU
      (`y = max(0,x) + min(0, alpha*(exp(x/alpha)-1))`). With TNNetELU
      now landed, this is a one-method variant; reuse the ELU test
      harness shape. (Duplicated up from the previous batch because
      it's the highest-ROI next layer in the activation family.)
- [ ] TNNetTanhShrink — `y = x - tanh(x)`. Parameter-free, derivative
      is `tanh(x)^2`. Pairs naturally with TNNetSoftSign as another
      "centered around zero" activation in the family.
- [x] TNNetThreshold — `y = x if x > theta else value`. Generalizes
      ReLU (`theta=0, value=0`). Useful as a building block for
      sparsity experiments; gradient is the indicator function.
      Landed: see TNNetThreshold entry in the post-lucky-day batch
      at the bottom of this file (commit 1f8d555).
- [x] TNNetLogSigmoid — `y = log(sigmoid(x)) = -softplus(-x)`. Useful
      in losses and the numerically stable log-likelihood path.
      Derivative is `sigmoid(-x)`. Landed in commit 59c9b93.
- [x] TNNetHardShrink — `y = x if |x| > lambda else 0`. The L1-prox
      activation. Trivial forward; derivative is the indicator on
      `|x| > lambda`. Landed: lambda in FFloatSt[0] (default 0.5),
      cached indicator derivative, three tests (forward / gradient /
      round-trip) in TestNeuralNumerical.pas.
- [x] TNNetSoftShrink — `y = x - lambda*sign(x) if |x| > lambda else 0`.
      The L1-prox cousin of HardShrink; smooth-ish flavor of the same
      sparsity-inducing activation. Landed alongside TNNetHardShrink,
      same harness shape.

#### Permutation / reshape primitives I'd enjoy writing
- [ ] TNNetSpaceToDepth + TNNetDepthToSpace pair (already on the list
      higher up — calling it out again because the round-trip test is
      a fun ~30-line job and unblocks the ViT-style patchify
      one-liner).
- [ ] TNNetChannelShuffle — ShuffleNet primitive. Pure permutation,
      mirror-image gathers for forward/backward. One grad-check + one
      "compose twice with same G returns identity" sanity check.
- [x] TNNetReverseChannels — flips the channel axis. Silly, tiny, but
      a great smoke test for the LoadFromString round-trip harness.
      Landed: parameter-free TNNetIdentity descendant, involution
      backward = forward, four tests in TestNeuralNumerical.pas
      (forward, gradient check, involution property, serialization
      round-trip).

#### Normalization follow-ups
- [x] TNNetInstanceNorm — per-sample, per-channel normalization (the
      GroupNorm limit at G=C). One-line constructor override on
      TNNetGroupNorm plus a dedicated grad-check. Landed (see earlier
      TNNetInstanceNorm entry).
- [ ] TNNetWeightStandardization wrapper — normalize a Conv/Dense
      layer's weights to zero-mean / unit-variance per output channel
      before the forward pass. Pairs especially well with GroupNorm.
- [ ] Numerical-gradient test that confirms TNNetRMSNorm matches the
      analytical gradient under non-trivial input distributions
      (mean != 0, variance != 1). The current test passes; a second
      one with shifted/scaled input pins the gradient path harder.

#### Audit follow-ups (extending the Backpropagate sweep)
- [ ] Upsampling / deconvolution family numerical-gradient checks —
      called out around line 88 as the next uncovered family. Pick
      one layer (e.g. TNNetUpsample) and add an input-gradient test
      mirroring the AvgPool / GlobalMaxPool harnesses.
- [ ] Recurrent-style layer numerical-gradient checks — the second
      family flagged uncovered. Identify the simplest recurrent layer
      in neuralnetwork.pas, add a single input-gradient test, and
      pin whatever shape it expects.
- [ ] Argmax-tie behaviour test for TNNetGlobalMaxPool: when two
      cells share the max, the deterministic tie-break should be
      documented in code and pinned with a tiny test (called out at
      line 1759; cheap and easy).

#### LoadFromString round-trip sweep (continuation of line 1827)
- [X] TNNetSwiGLU LoadFromString round-trip test.
- [X] TNNetGEGLU LoadFromString round-trip test.
- [X] TNNetLayerScale LoadFromString round-trip test.
- [X] TNNetRMSNorm LoadFromString round-trip test.
- [X] TNNetDropPath LoadFromString round-trip test (forward in inference
      mode must be the identity; pin that too).

#### Small experiments / examples I'd enjoy
- [ ] Activation bake-off on a tiny CIFAR-10 net: ReLU vs ReLU6 vs
      GELU vs Swish vs Mish vs ELU vs SiLU vs SoftSign. Single seed,
      few epochs, markdown table with final val-accuracy + wall-clock
      per epoch. Cheap, high-signal answer to "does the new family
      actually matter."
- [ ] Normalization bake-off on the same tiny CIFAR-10 net:
      ChannelStdNorm vs LayerNorm vs GroupNorm(G=4) vs RMSNorm.
      Same format; pin which one wins at this scale.
- [ ] Saturating-activation bake-off: SoftSign vs Tanh vs HardTanh
      on a deep stack (8+ layers) — does SoftSign's slower saturation
      actually help gradient flow vs Tanh in practice? One small chart
      in an examples/<dir>/README.md is plenty.
- [ ] Tiny "hello attention" toy task (already on the list at line
      1777) — copy-task or reverse-string solved by 1-layer SDPA + MLP.
      Demonstrates the full attention pipeline end-to-end without a
      real corpus. Would lean on the recently-landed SDPA layer.

#### Tooling / quality of life
- [X] One-line type alias: `TNNetGlobalAvgPool = class(TNNetAvgChannel)`
      (called out at line 1816) so the canonical Keras/PyTorch name
      resolves. Dispatch entry + round-trip test.
- [ ] Helper proc `WriteLayerTimings(NN: TNNet; Sample: TNNetVolume)`
      that runs one forward pass and prints per-layer wall-clock to
      stdout. Pure additive; no behaviour change. Lets users spot the
      slow layer without an external profiler.
- [ ] Tiny `bin/` micro-benchmark that prints ns/op for Add, Mul,
      DotProduct and for the LayerNorm / RMSNorm / GroupNorm forward
      pass. Writes a CSV so future regressions are visible at a glance.
- [ ] Volume invariant assertion helper: `AssertNoNaN(V: TNNetVolume)`
      that loops once and raises if any element is NaN/Inf. Sprinkle
      into the activation saturation tests so a regression to NaN is
      caught immediately instead of as a downstream gradient mismatch.

#### Documentation
- [ ] README activation-family table — one row per activation with
      formula, derivative, "saturating?" flag, and a 5-word use-case
      hint. The family has grown enough (ELU, SiLU, SoftSign, ReLU6,
      Swish, Mish, GELU, HardSwish, SELU, LeakyReLU, HardSigmoid,
      HardTanh, SoftCapping, CELU) that a table beats prose.

### Lucky-day batch — 2026-05-15 (post CELU + GlobalAvgPool + round-trip-sweep)

This batch landed:
- TNNetCELU (continuously differentiable ELU):
  `y = max(0,x) + min(0, alpha*(exp(x/alpha)-1))` with configurable
  alpha in FFloatSt[0]. Follows the TNNetELU template — overrides
  Compute only (cached derivative path inherited from TNNetReLUBase).
  Forward, numerical-gradient, and serialization round-trip tests in
  TestNeuralNumerical.pas; registered in both CreateLayer dispatch sites.
- TNNetGlobalAvgPool: parameter-less subclass of TNNetAvgChannel so the
  canonical Keras/PyTorch name resolves via LoadFromString. Mirrors
  TNNetGlobalMaxPool's naming. Both dispatch sites updated;
  serialization round-trip test added.
- LoadFromString round-trip sweep follow-through: new round-trip tests
  for TNNetSwiGLU, TNNetGEGLU, and TNNetLayerScale; TNNetDropPath's
  existing round-trip strengthened to pin inference-mode identity
  before AND after round-trip. RMSNorm and DropPath round-trips
  were already covered (NormSerializationRoundTripWithPerturbedWeights
  / TestDropPathSerializationRoundTrip). No dispatch bugs surfaced.

#### Natural follow-ups
- [ ] TNNetCELU CIFAR-style smoke example wired into one of the
      SimpleImage paths — pairs with the still-open activation
      bake-off entry. CELU should sit in the same comparison table as
      ELU/ReLU6/SiLU.
- [ ] CELU vs ELU alpha-sensitivity micro-experiment: tiny classifier,
      sweep `alpha in {0.1, 0.5, 1.0, 2.0}` for both, chart final
      loss. Visualises the "continuous-derivative-at-zero" claim
      concretely without leaving the existing test harness.
- [ ] Add TNNetCELU and TNNetGlobalAvgPool to the README layer
      reference once the activation-family table (above) lands.
- [ ] LoadFromString round-trip for the remaining activation-family
      additions that lack one (e.g. confirm TNNetTanhShrink and
      TNNetHardTanh have explicit round-trip coverage; both descend
      from TNNetReLUBase so the dispatch is generic, but a one-line
      test would close the audit-coverage gap).

### Lucky-day batch — 2026-05-15 (seed 708478)

Today's lucky-number-driven batch. These are things I would genuinely
enjoy building, in roughly increasing order of size. Each one is sized
to land in a single focused session.

#### Tiny, high-signal layers
- [ ] TNNetMaxOut — output is max over k linear branches per unit.
      Classic Goodfellow 2013 piecewise-linear activation that subsumes
      ReLU/leaky-ReLU. Parameterise k via FStruct[0]. Forward, numerical
      gradient, and round-trip tests. Cheap given the existing
      multi-branch infrastructure.
- [ ] TNNetSquaredReLU — `y = max(0,x)^2`. Used in Primer / So et al.
      2021 as a drop-in replacement that often improves transformer
      perplexity. Trivial Compute + ChainDeriv override, single-line
      derivative `2*max(0,x)`. Mirrors the TNNetReLU template exactly.
- [x] TNNetShiftedReLU — `y = max(-1, x)`. Tiny but useful: keeps a
      small negative range without ELU's exp cost. Good template for
      future "shifted activation" experiments. Landed: parameter-free
      subclass of TNNetReLUBase; cached `{0,1}` indicator in
      FOutputErrorDeriv. Forward / gradient (inputs biased away from
      the x=-1 kink) / serialization round-trip tests in
      TestNeuralNumerical.pas.
- [ ] TNNetSnake — `y = x + (1/alpha) * sin(alpha*x)^2`. Periodic
      activation from "Neural Networks Fail to Learn Periodic Functions"
      (Ziyin 2020). Niche but slots cleanly into the activation
      bake-off; configurable alpha via FFloatSt[0].
- [ ] TNNetGaussianActivation — `y = exp(-x^2)`. RBF-style activation;
      useful for the saturating-activation bake-off entry above.
- [x] TNNetLogSoftMax — exact log-softmax with numerically stable
      `x - max - log(sum(exp(x-max)))`. Pairs with NLL-style loss paths
      and avoids the log(softmax) trick at training time. Backward is
      `dy - softmax(x) * sum(dy)` over the depth axis. Numerical-grad
      test required. Landed: per-(X,Y) over depth, parameter-free,
      backward simplifies to `prev.err[d] += dy[d] - exp(out[d])*sum_d dy[d]`.
      Forward / gradient-check / serialization round-trip tests in
      TestNeuralNumerical.pas.

#### Composite blocks I'd enjoy building
- [ ] TNNetPreNormTransformerBlock helper — convenience builder
      `AddPreNormTransformerBlock(NN; d_model; n_heads; d_ff)` that
      stacks LayerNorm → SDPA → residual → LayerNorm → SwiGLU MLP →
      residual. Pure additive sugar over the existing primitives; saves
      ~20 lines per use site. No new layer types.
- [ ] TNNetSEBlock — Squeeze-and-Excitation channel-attention block as
      a builder helper. Already feasible with GlobalAvgPool + FullConnect
      + ReLU + FullConnect + Sigmoid + CellMul. Worth packaging because
      every modern CNN paper uses it.
- [ ] TNNetCBAMChannelAttention — Channel + spatial attention from
      Woo et al. 2018. Builder over existing pool/conv primitives.

#### Numerical hygiene
- [ ] AssertFinite(V: TNNetVolume; const Where: string) — global helper
      that scans for NaN/Inf and raises with a labelled message. Sprinkle
      into the activation-saturation tests and the new SDPA softmax
      path so a regression points at a layer name, not a "gradient
      mismatch" line 200 lines later.
- [ ] Numerical-gradient harness extension: opt-in central-difference
      check at fp64 internally even when the network is fp32. Cuts
      false-positive failures when a layer is correct but cancellation
      noise drives the diff over the threshold.
- [ ] Softmax stability micro-test: feed a deliberately huge logit
      vector (`x = [1000, 0, 0, ...]`) through TNNetSoftMax and
      TNNetLogSoftMax (once it lands); assert finite output, finite
      gradient, and sum-to-one within 1e-6. Pins the
      subtract-the-max-before-exp invariant.

#### Tiny experiments I'd enjoy running
- [ ] Activation derivative-at-zero study: plot Compute + ChainDeriv
      output around x=0 for ReLU, GELU, Swish, Mish, CELU, ELU on a
      [-2,2] grid; dump CSV. One-screen visual of which activations
      are genuinely smooth at zero vs only "approximately" smooth.
      Pairs naturally with [[lucky-day-celu-vs-elu]] alpha sweep.
- [ ] Width vs depth at fixed parameter budget on a tiny MNIST-shaped
      task: 4 widths × 4 depths matched to the same param count, plot
      val-loss heatmap. Cheap, surprisingly informative on which regime
      a given activation/normalization combo prefers.
- [ ] Gradient-flow sanity sweep: 16-layer MLP with ReLU, GELU, Swish,
      Mish, CELU; log per-layer gradient L2 norms after one backward
      pass on random input. Visualises vanishing/exploding gradient
      behaviour without needing a real training loop.
- [ ] Normalization-position study (Pre-LN vs Post-LN) on a 4-layer
      transformer stack solving the copy-task — does Pre-LN really
      train more stably at this scale? Single chart, single seed.

#### Round-trip & dispatch audit follow-through
- [ ] Sweep every layer registered in CreateLayer for an explicit
      LoadFromString round-trip test. The CELU/SwiGLU/GEGLU/LayerScale/
      DropPath/RMSNorm batch closed obvious gaps; finish the audit by
      generating a one-line "for each registered class, build with
      defaults → save → load → assert structural equality" test driven
      by the dispatch table itself, so newly-added layers are
      automatically covered.
- [ ] TNNetTanhShrink + TNNetHardTanh explicit round-trip tests (the
      open item directly above this batch). Tiny, two-line tests.

#### Documentation
- [ ] One-page "activations cheat sheet" in docs/activations.md:
      formula, derivative, saturating-or-not, smooth-at-zero-or-not,
      typical use case. Lands the same content the README table entry
      proposes, but in a dedicated page that can grow without bloating
      the README.
- [ ] Short README section: "How to add a new activation in ~30 lines"
      walking through the TNNetReLUBase template with TNNetCELU as the
      worked example. Lowers the activation-contribution barrier
      meaningfully.

#### Stretch / ambitious
- [ ] TNNetMultiHeadSelfAttention as a true layer (not a builder),
      wrapping num_heads parallel SDPA cores with Q/K/V projection and
      output projection. The single-head SDPA core is already gradient-
      checked; this is "stack and project". Would finally close the
      transformer-encoder-block top-line item at the head of this file.
- [ ] Mixture-of-Experts routing layer (also at the top of this file):
      top-k softmax gate over N expert sub-networks, with load-balancing
      auxiliary loss. Stretch but well-scoped if the SDPA / transformer
      pieces are in place.


### Lucky-day batch — 2026-05-15 (post LogSoftMax + InstanceNorm + Shrink batch)

This batch landed:
- TNNetLogSoftMax (exact, stable per-(X,Y) log-softmax over depth).
  Forward, gradient-check, and serialization round-trip tests.
- TNNetInstanceNorm (TNNetGroupNorm with Groups := Depth at SetPrevLayer).
  Forward / gradient-check / serialization round-trip tests.
- TNNetHardShrink + TNNetSoftShrink L1-prox activations, lambda in
  FFloatSt[0] (default 0.5). Six new tests (forward / gradient /
  round-trip per layer).

Natural follow-ups (each commit-sized):

- [ ] TNNetLogSoftMax + cross-entropy training-loss smoke example:
      replace `TNNetSoftMax` + cross-entropy with `TNNetLogSoftMax` +
      NLL on a tiny classifier and confirm matching convergence. The
      log-domain path is the whole reason this layer landed.
- [ ] TNNetLogSoftMax stability test on a huge logit vector
      (e.g. `x = [1000, 0, 0, ...]`): assert finite output, exp(out)
      sums to 1, gradient is finite. Pairs with the softmax stability
      micro-test already on the list.
- [ ] TNNetLogSoftMax "global vs pointwise" question: confirm whether
      the per-(X,Y) over-depth axis is the right default vs a global
      variant matching the existing TNNetSoftMax. If both are useful,
      add a global sibling using TNNetSoftMax's axis convention.
- [ ] TNNetInstanceNorm vs TNNetGroupNorm bake-off on a tiny conv
      example — single chart, single seed, demonstrates the
      per-channel limit case.
- [ ] TNNetInstanceNorm CIFAR-style integration example (a simple
      SimpleImage path with InstanceNorm replacing the existing
      ChannelStdNorm) — a real end-to-end use of the new layer.
- [ ] TNNetHardShrink / TNNetSoftShrink kink-region gradient test:
      currently the gradient checks bias inputs away from `±lambda`
      to avoid finite-difference noise. A dedicated test that
      asserts the derivative is exactly the indicator on `|x| > lambda`
      at hand-picked inputs (no central differences) would close the
      coverage gap at the kink.
- [ ] TNNetHardShrink / TNNetSoftShrink sparsity micro-experiment:
      train a tiny autoencoder with each as the bottleneck activation
      and print the fraction of zero activations vs reconstruction
      loss. Concrete demonstration of the L1-prox sparsity claim.
- [x] TNNetLogSigmoid (`y = log(sigmoid(x)) = -softplus(-x)`) — the
      natural companion to TNNetLogSoftMax for the BCE-style loss
      path. Standard stable form; derivative is `sigmoid(-x)`. Landed
      in commit 59c9b93 — see TNNetLogSigmoid entry below.
- [x] TNNetThreshold (`y = x if x > theta else value`) — still open
      from the seed-401988 batch; generalizes ReLU. Two FFloatSt
      params (theta, value), indicator derivative. Landed in commit
      1f8d555: TWO-float variant via FFloatSt[0]=theta, FFloatSt[1]=value
      (pattern verified against existing TNNetMovingScale dispatch).
      Four tests: forward, ReLU equivalence at defaults, gradient
      check (inputs biased away from theta), serialization round-trip.
- [ ] README activation/normalization reference: TNNetLogSoftMax,
      TNNetInstanceNorm, TNNetHardShrink, TNNetSoftShrink each need a
      one-line description + tiny usage snippet next to their
      siblings (TNNetSoftMax / TNNetGroupNorm / TNNetTanhShrink).


### Lucky-day batch — 2026-05-15 (seed 286083)

A fresh batch of small, well-scoped ideas I'd genuinely enjoy taking on.
Most are one-commit-sized and pair naturally with layers/tests already in
the tree. Anything stretchy is called out explicitly.

#### New activations (tiny, ReLUBase-style)
- [x] TNNetLogSigmoid (`y = log(sigmoid(x)) = -softplus(-x)`) — already
      hinted at in the previous batch; standard stable form, derivative
      is `sigmoid(-x)`. Companion to TNNetLogSoftMax for BCE-with-logits.
      Forward / numerical-gradient / serialization round-trip tests.
      Landed: parameter-free subclass of TNNetReLUBase with branched
      stable formulation (x>=0: `y=-ln(1+exp(-x))`, x<0: `y=x-ln(1+exp(x))`).
      Derivative `sigmoid(-x)` cached in FOutputErrorDeriv. Four tests
      in TestNeuralNumerical.pas including ±1e6 saturation test.
- [ ] TNNetSquaredReLU (`y = max(0, x)^2`) — the Primer/Pythia
      activation. Cheap, derivative is `2 * max(0, x)`. A single Compute
      / ChainDeriv pair plus three tests.
- [ ] TNNetReLU6 (`y = min(max(0, x), 6)`) — MobileNet-style bounded
      ReLU. Indicator-style derivative on `(0, 6)`. Useful for
      quantization-friendly experiments later on.
- [ ] TNNetSoftSign (`y = x / (1 + |x|)`) — bounded smooth saturating
      activation, derivative `1 / (1 + |x|)^2`. Sibling to Tanh,
      computationally cheaper.
- [ ] TNNetMaxout (k=2 fixed first cut) — per-position max over k linear
      pieces of the input depth. Larger commit; clean fit for the
      element-wise family if depth-grouping is added carefully.

#### New normalization variants
- [ ] TNNetWeightStandardization — normalize convolution weights per
      output channel (zero-mean, unit-variance over input-channel ×
      spatial). Pairs with GroupNorm in the literature; cheap forward,
      gradient flows through the standardization (auto-diff via
      Backpropagate override). Test as a wrapper around TNNetConvolution.
- [x] TNNetPixelNorm — landed 2026-05-15 (commit 2b62787). StyleGAN-style
      per-pixel L2 normalization across the depth axis with eps=1e-8;
      parameter-free, exact backward via the unit-norm Jacobian (depth-
      reduced, dividing by Depth to match the `mean` forward). Forward,
      central-difference gradient, and SerializationRoundTrip tests in
      TestNeuralNumerical.pas.
- [ ] TNNetSpectralNormHook — power-iteration weight spectral norm
      tracked as a non-trainable buffer on TNNetFullConnect /
      TNNetConvolution. Stretch: makes GAN-style examples honest.

#### Attention follow-through
- [ ] TNNetMultiHeadSelfAttention as an actual layer (not a builder).
      Wraps `num_heads` parallel TNNetScaledDotProductAttention cores +
      Q/K/V projections + output projection. Single-head SDPA core is
      already gradient-checked, so this is "stack and project + test".
      Would finally close the top-of-file transformer-encoder-block
      open item. (Echoes the existing stretch entry; keeping a copy in
      this batch so the new lucky-day pass picks it up too.)
- [ ] TNNetCausalMask — extract the causal-mask flag from
      TNNetScaledDotProductAttention into a reusable depth-axis mask
      layer so non-attention experiments can reuse it. Trivial forward,
      identity backward except for masked positions.
- [ ] Attention-pattern visualization helper: dump the post-softmax
      attention matrix (B, H, T, T) to PGM/CSV for a trained tiny model.
      Pure offline utility; no test infra needed.

#### Loss layers / objectives
- [ ] TNNetNLLLoss companion to TNNetLogSoftMax: NLL over (X,Y,Depth)
      with class index targets. Closes the "log-softmax + NLL" pairing
      and is the natural way to demonstrate the new log-domain path.
- [ ] TNNetFocalLoss (`(1 - p_t)^gamma * CE`) — class-imbalance loss,
      gamma in `FFloatSt[0]` (default 2.0). One forward, one backward,
      one gradient-check.
- [ ] TNNetLabelSmoothingCE — cross-entropy with epsilon smoothing
      (default 0.1). One float param, identical Jacobian to standard CE
      after the smoothing transform on targets.
- [ ] TNNetHuberLoss / TNNetSmoothL1Loss — delta in `FFloatSt[0]`.
      Useful for the time-series forecasting example already on the
      ideas list.

#### Tiny experiments I'd enjoy running
- [ ] LogSoftMax+NLL vs SoftMax+CE convergence parity test: same seed,
      same tiny classifier, plot val-loss curves. Pins the claim that
      the log-domain path is numerically equivalent (and demonstrates
      the new TNNetLogSoftMax in anger).
- [ ] InstanceNorm vs GroupNorm vs LayerNorm vs ChannelStdNorm
      single-seed bake-off on a 3-layer CIFAR-ish conv stack. One chart,
      one CSV, reuses the existing SimpleImage example skeleton.
- [ ] Shrink-activation sparsity sweep: train a tiny autoencoder with
      ReLU / SoftShrink / HardShrink as the bottleneck; sweep lambda
      over `{0.1, 0.25, 0.5, 1.0}`; report (sparsity %, recon loss).
      Concrete demonstration of the L1-prox claim. (Pairs with the
      existing shrink-sparsity entry; this version pins the sweep grid.)
- [ ] Activation "kink at zero" finite-difference noise audit:
      generate central-difference gradient errors for every activation
      on a `[-0.05, 0.05]` window stepping by 1e-3; identify which
      activations need the inputs biased away from zero in tests, and
      which are genuinely smooth. Direct follow-up to the shrink test
      kink-region gap.
- [ ] Numerical-gradient epsilon study: replicate one existing test
      with `epsilon ∈ {1e-2, 1e-3, 1e-4, 1e-5}` and tabulate the
      observed max abs error. Quantifies the central-difference noise
      floor; informs future test thresholds.

#### Test coverage / hygiene
- [ ] Auto-generated "every registered layer round-trips through
      LoadFromString" test (already on the list; re-pinning it because
      it would catch the next dispatch regression for free). Drive it
      off the CreateLayer dispatch table directly.
- [ ] Numerical-gradient checks for the remaining uncovered families
      flagged earlier: upsampling/deconvolution and recurrent-style
      layers. One family per commit.
- [ ] TNNetLogSoftMax stability micro-test (huge logits, e.g.
      `x = [1000, 0, 0, ...]`): assert finite output, `exp(out).sum() ≈
      1`, finite gradient. Pairs with the existing softmax stability
      entry.
- [ ] Determinism test: same seed → bit-identical forward+backward
      across two runs of the same tiny net. Tiny but catches future
      nondeterminism regressions (e.g. parallel reductions reordered).

#### Documentation
- [ ] One-page "activations cheat sheet" in `docs/activations.md`:
      formula, derivative, saturating-or-not, smooth-at-zero, typical
      use case. (Echoing the earlier batch — re-pinning because the
      activation roster has grown meaningfully since.)
- [ ] One-page "normalization cheat sheet" in `docs/normalization.md`:
      LayerNorm vs RMSNorm vs GroupNorm vs InstanceNorm vs
      ChannelStdNorm — what axes each one reduces over, learnable
      params, typical use case. The roster is now big enough to need
      a map.
- [ ] "How to add a new activation in ~30 lines" walkthrough — reuses
      TNNetReLUBase, with TNNetCELU or TNNetSquaredReLU as the worked
      example. Lowers the contribution barrier.

#### Stretch / ambitious
- [ ] Tiny GPT char-level example end-to-end in Pascal on CPU. Now
      well-scoped given LayerNorm + RoPE + SDPA + Shrinks + LogSoftMax
      all landed; the remaining gap is mostly MultiHeadSelfAttention
      and a tokenizer wrapper. Same top-of-file item, but lucky-day
      version: just `tinyshakespeare.txt`, a 2-layer model, and
      generation that produces non-trivial text.
- [ ] Mixture-of-Experts routing layer (also at the top of this file):
      top-k softmax gate over N expert sub-networks with load-balancing
      auxiliary loss. Stretch but well-scoped once MHSA lands.
- [ ] ONNX (or simpler JSON) export path — minimal viable: dump a
      forward-only graph for the currently-supported subset of layers,
      enough to run inference in onnxruntime. Doc which layers are
      out-of-scope for v1.


### Lucky-day batch — 2026-05-15 (post LogSigmoid + ShiftedReLU + Threshold batch)

This batch landed three small activations dispatched serially to opus
sub-agents on a self-described "lucky day":

- TNNetLogSigmoid (commit 59c9b93) — `y = log(sigmoid(x)) = -softplus(-x)`,
  branched stable formulation. Derivative `sigmoid(-x)` cached in
  FOutputErrorDeriv. Four tests including a ±1e6 saturation check.
- TNNetShiftedReLU (commit aa50051) — `y = max(-1, x)`, parameter-free
  subclass of TNNetReLUBase with `{0,1}` indicator derivative.
- TNNetThreshold (commit 1f8d555) — `y = x if x > theta else value`,
  TWO-float dispatch (theta in FFloatSt[0], value in FFloatSt[1]).
  First in-repo activation with two configurable floats; the
  TNNetMovingScale dispatch pattern was the precedent. Generalizes
  ReLU at the defaults theta=value=0.

Test suite grew from 360 → 367 tests, all passing. No bugs surfaced.

#### Natural follow-ups
- [x] README activation reference: TNNetLogSigmoid, TNNetShiftedReLU,
      and TNNetThreshold each need a one-line description + tiny
      snippet next to their siblings in the activation table.
      Done 2026-05-15 (commit bc39126): three rows added to README.md
      §"Layers with Activation Functions and no Trainable Parameter"
      right after TNNetSquaredReLU. Matched the newer "Created with ..."
      style. Minor pre-existing inconsistency noted: older rows in the
      same table omit that suffix.
- [ ] LogSigmoid + BCE-with-logits training-loss smoke example: pair
      TNNetLogSigmoid with a tiny binary classifier and confirm
      matching convergence vs a sigmoid + BCE baseline.
- [ ] Threshold-as-sparsifier micro-experiment: use TNNetThreshold
      with theta>0 as a hidden-layer activation, sweep theta over
      `{0.0, 0.1, 0.5, 1.0}` on a tiny autoencoder, report
      (sparsity %, recon loss). Demonstrates the indicator-gradient
      flavor of sparsification without the L1-prox shrink layers.
- [ ] Activation menagerie bake-off (already on the list) is now
      sweeter with LogSigmoid / ShiftedReLU / Threshold in the menu.
- [ ] LogSigmoid kink-region test: at x near the `x=0` branch
      crossover, confirm the two branches join smoothly (continuity
      of both y and dy/dx) to within fp tolerance. Closes a small
      gap the saturation test does not cover.
- [ ] Threshold "kink at theta" test: at hand-picked x exactly equal
      to theta, document the chosen convention (currently: x > theta,
      so x=theta routes to the `else` branch). Pin with a tiny
      no-central-differences test, mirroring the open HardShrink/
      SoftShrink kink-region entry above.


### Lucky-day batch — 2026-05-15 (seed 135518)

A fresh draw on a lucky day. Random number printed first to set the
mood, then ideas that I would genuinely enjoy building. Bias: small,
mergeable, gradient-checkable, with a real punchline.

#### Activations I'd enjoy adding
- [x] TNNetSquaredReLU — already landed in earlier batch (see line 186).
      Lucky-day re-pin from seed 135518 was stale.
- [x] TNNetCELU — already landed; class at neuralnetwork.pas:740,
      constructor takes optional `alpha`. Lucky-day re-pin was stale.
- [x] TNNetTanhShrink — already landed; class at neuralnetwork.pas:646,
      derivative cached in FOutputErrorDeriv. Lucky-day re-pin was stale.
      Serialization round-trip test added 2026-05-15 (commit 942bb52)
      to close a small gap; same commit added TNNetHardTanh round-trip.
- [x] TNNetSoftSign — already landed; class at neuralnetwork.pas:757,
      forward + grad-check + round-trip tests all present. Lucky-day
      re-pin was stale.
- [ ] TNNetAPL (Adaptive Piecewise Linear) — sum of hinge functions
      `y = max(0, x) + sum_s a_s * max(0, -x + b_s)`. Per-channel
      learnable knees and slopes; stretch but very fun. Would also be
      the first activation in the repo with vector-valued learnable
      parameters beyond a scalar.

#### Layers I'd enjoy building
- [ ] TNNetMultiHeadSelfAttention — finally close the headline gap.
      Built on TNNetScaledDotProductAttention (already landed) plus
      per-head Q/K/V projections and a concat+output projection.
      Causal-mask flag, full numerical-gradient test on a tiny
      d_model=8, n_heads=2 config. Unblocks Tiny GPT.
- [ ] TNNetALiBiBias — additive linear position bias to attention
      scores, alternative to RoPE. Parameter-free per slope schedule,
      so the test surface is small. Pairs with the SDPA layer.
- [ ] TNNetMaskedMean / TNNetMaskedMax — pooling over a variable-
      length sequence given a {0,1} mask channel. Useful for
      sequence-classification examples and avoids the existing
      "pad with zeros and hope average is small" workaround.
- [ ] TNNetStochasticDepth wrapper that randomly skips a residual
      branch at train time (à la DropPath but at the block level
      rather than per-token). DropPath has landed; this is the
      coarser sibling and is one parameter.

#### Tests / numerical-gradient audit
- [ ] Continue the Backpropagate audit on the upsampling /
      deconvolution family next (TNNetUpsample, TNNetDeMaxPool,
      TNNetDeAvgPool, any TNNet*Deconvolution*). One numerical-
      gradient test per layer in TestNeuralNumerical.pas, mirroring
      the pattern already used for TNNetPadXY / TNNetCrop.
- [ ] Recurrent-style layer audit: identify any RNN/GRU/LSTM-shaped
      layers in the repo and add numerical-gradient tests if missing,
      treating per-timestep state as a normal tensor input.
- [ ] Threshold + ShiftedReLU + LogSigmoid kink-region tests
      consolidated into a single parametric helper in
      TestNeuralNumerical.pas. Reduces copy-paste and makes adding a
      new activation's kink test a 3-line addition.
- [ ] Determinism CI test (echoing earlier entry): same seed → bit-
      identical forward + backward across two runs of a 3-layer net.
      Catches future nondeterminism regressions from parallel
      reductions or hash-map iteration order changes.

#### Experiments I'd enjoy running
- [ ] Activation menagerie bake-off on CIFAR-10-tiny: train the same
      small CNN with ReLU, GELU, Swish, Mish, SquaredReLU, CELU,
      SoftSign — report train loss curve + final test accuracy in a
      single Markdown table under `docs/experiments/`. The roster
      is now big enough to justify an in-repo comparison.
- [ ] Norm-layer bake-off: same tiny CNN with no-norm vs BatchNorm
      vs LayerNorm vs RMSNorm vs GroupNorm. Tabulated convergence
      speed + final accuracy. Would settle a recurring intuition
      question and ship a reusable benchmark harness.
- [ ] Threshold-as-sparsifier sweep (already pinned in the previous
      batch). I'd enjoy actually running it: theta ∈ {0, 0.1, 0.5,
      1.0}, report (active-units %, recon loss) on a 64-unit
      autoencoder.
- [ ] "Smallest net that can learn parity-N" study — sweep N ∈
      {2, 4, 6, 8} and report the smallest hidden-width that fits
      cleanly with a fixed budget. Tiny, reproducible, and a nice
      teaching artifact.
- [ ] Initialization scheme sweep: He vs Xavier vs orthogonal vs
      identity-on-residual on a 6-block residual tower. Plot final
      train loss vs init.

#### Examples I'd enjoy writing
- [ ] `examples/TinySequence/` — char-level next-token model on
      `tinyshakespeare.txt` using TNNetScaledDotProductAttention plus
      a hand-rolled token embedding lookup. Stays single-head until
      MHSA lands; runs in well under a minute on CPU.
- [ ] `examples/AutoencoderMNIST/` — tiniest possible MNIST
      autoencoder demonstrating the encode/decode split using
      TNNetUpsample. Doubles as a regression target for the
      upsampling-family numerical-gradient tests above.
- [ ] `examples/AttentionViz/` — visualize attention weights for the
      single-head SDPA layer on a toy copy task. Dump per-head
      weight matrices as PGM/PPM so no plotting dependency.
- [ ] `examples/SineRegression/` — 1D function-fitting toy
      (`y = sin(2πx) + noise`). Two-layer MLP, prints train loss per
      epoch. Smallest possible "does the library still train?" demo
      for the README quick-start.

#### Infrastructure / tooling
- [ ] Volume unit micro-benchmark (already pinned): I'd enjoy
      writing it — print ns/op for Add, Mul, DotProduct, AddArea,
      InterleaveSplit. Single binary under `bin/`, no OpenCL
      required, table-formatted output.
- [ ] `scripts/run_all_tests.sh` wrapper that builds and runs every
      test binary in `tests/` and prints a pass/fail summary. Today
      the canonical command is buried in CI; surface it.
- [ ] Layer-name registry self-check: enumerate every TNNetLayer
      subclass via the dispatch table and assert each one round-trips
      through SaveToString → LoadFromString unchanged. Catches
      "added a layer, forgot the dispatch entry" regressions —
      a real category of past bugs.
- [ ] `docs/CONTRIBUTING.md` "anatomy of a layer" section: walk
      through TNNetShiftedReLU as the canonical 50-line activation
      example, showing Compute, Backpropagate, dispatch entry, and
      the numerical-gradient test stub. Lowers the contribution
      barrier dramatically.

#### Documentation
- [ ] Activations cheat sheet in `docs/activations.md` — re-pinned
      again; with LogSigmoid / ShiftedReLU / Threshold now in the
      menu, the roster has crossed the "needs a map" threshold.
      One row per activation: formula, derivative, bounded?,
      smooth-at-zero?, typical use.
- [ ] Normalization cheat sheet in `docs/normalization.md` — same
      story for norm layers. LayerNorm vs RMSNorm vs GroupNorm vs
      InstanceNorm vs ChannelStdNorm in a single table.
- [ ] `docs/numerical_gradient.md` — short tutorial: why central
      differences, how the existing TestNumericalGradient helper
      works, how to add a test for a new layer in five lines.
      Useful onboarding doc and points at the contribution path.

#### Stretch / ambitious
- [ ] Tiny GPT end-to-end (re-pinned, again). The remaining gap is
      MHSA + a tokenizer wrapper. Once MHSA above lands, this is the
      logical next item. Lucky-day flavor: ship `tinyshakespeare.txt`
      under `examples/TinyGPT/data/`, 2-layer model, generation that
      produces recognizably Shakespeare-flavored output.
- [ ] Mixture-of-Experts routing layer (re-pinned). Top-k softmax
      gate over N expert sub-networks plus a load-balancing
      auxiliary loss. Best done after MHSA so it has a real host
      architecture to slot into.
- [ ] Minimal JSON model export — forward-only graph dump that an
      external Python script could load into onnxruntime. Doc which
      layers are out-of-scope for v1. Solves the "I trained it,
      now what?" gap for non-Pascal downstream users.


### Lucky-day batch — 2026-05-15 (post PixelNorm + README-activations pass)

Three serial opus agents dispatched on a self-described lucky day.
Landed:

- `942bb52` — TanhShrink/HardTanh serialization round-trip tests
  (agent #1 found its assigned target TNNetTanhShrink was already
  fully implemented; pivoted to closing the round-trip-test gap
  flagged in this file at line 2090).
- `2b62787` — TNNetPixelNorm StyleGAN-style per-pixel L2 norm across
  the depth axis. Parameter-free; depth-reduced unit-norm Jacobian.
  Three new tests (forward unit-RMS, central-difference gradient,
  serialization round-trip).
- `bc39126` — README activation reference: TNNetLogSigmoid,
  TNNetShiftedReLU, TNNetThreshold each get a row in the
  "Layers with Activation Functions and no Trainable Parameter"
  table.

Test suite: 367 → 372, all passing. No bugs surfaced.

#### Surprise / meta-observation

The lucky-day batch at seed 135518 (above) re-pinned several
activations that were *already implemented* on this branch:
TNNetSquaredReLU, TNNetCELU, TNNetTanhShrink, TNNetSoftSign all
landed in earlier passes but were re-listed as if open. Two of the
three agents in this pass had to grep their target out of the code
before pivoting. This costs orchestration budget on every lucky-day
draw.

#### Follow-ups added by this pass

- [x] Tasklist staleness sweep: write a small `scripts/audit_tasklist.sh`
      (or similar) that greps each unchecked `TNNet*` mentioned in
      tasklist.md against `neural/neuralnetwork.pas` and flags any
      name already present in the dispatch tables. Would have caught
      the four stale entries above before they hit an agent.
      Landed: scripts/audit_tasklist.sh — see entry in seed 993208 batch.
- [ ] PixelNorm follow-ups now that it has landed:
  - [ ] PixelNorm + StyleGAN-flavored generator micro-example (the
        layer's headline use case; pairs with the existing VisualGAN).
  - [ ] PixelNorm vs InstanceNorm vs no-norm bake-off on a tiny
        generator-shaped net — concrete demonstration of why per-pixel
        norm is the chosen StyleGAN trick.
  - [ ] Numerical-gradient eps sensitivity check specifically for
        PixelNorm at small inputs (where `||x||` approaches eps and
        the Jacobian blows up). Pins the chosen eps=1e-8.
- [ ] LogSigmoid/ShiftedReLU/Threshold README rows landed using the
      newer "Created with ..." style; older table rows omit that
      suffix. Small cleanup task: backfill the older entries (or
      strip the suffix from the newer ones) for consistency.


### Lucky-day batch — 2026-05-15 (seed 993208)

Fresh draw on a lucky day. Before adding anything, a sanity check:
the "Tasklist staleness sweep" item from the previous pass is still
unchecked, so several entries below were chosen by first grep-ing
their target out of `neural/neuralnetwork.pas` to avoid the
re-pinning trap noted in the previous batch's meta-observation.

#### Layers I'd enjoy building
- [ ] TNNetSoftPlus — `log(1 + exp(x))` activation, with the
      standard `x > threshold ? x : log1p(exp(x))` numerical guard
      to avoid overflow on positive inputs. Parameter-free, smooth
      ReLU sibling. Single-file change in `neural/neuralnetwork.pas`
      + dispatch entry + a Compute / kink-region / numerical-grad
      triplet in `TestNeuralNumerical.pas`. Verified absent from
      the dispatch table.
- [ ] TNNetMultiHeadSelfAttention — the headline blocker for
      TinyGPT. Now that `TNNetScaledDotProductAttention` has landed
      as the single-head core, the MHSA wrapper is a depth-reshape
      around H copies of it plus a final projection. I'd enjoy
      writing it as a thin composition layer that internally builds
      and owns H SDPA sub-layers rather than a brand-new
      monolithic Backpropagate — keeps the gradient path
      auditable. Verified absent.
- [ ] TNNetBias — bias-only "add a learnable per-channel offset"
      layer. Surprisingly missing as a standalone primitive; today
      bias only ships fused into Dense/Conv. Useful for residual
      towers that want a bias on the skip path. Verified absent.
- [ ] TNNetMaxOut — Goodfellow-style maxout with K linear pieces;
      pairs well with the activation menagerie bake-off below.

#### Experiments I'd enjoy running
- [ ] PixelNorm Jacobian-blow-up empirical test: feed the layer
      inputs at `||x|| ∈ {1, 1e-2, 1e-4, 1e-6, 1e-8, 1e-10}` and
      record the central-difference vs analytic-gradient relative
      error. Output a Markdown table under `docs/experiments/`.
      Concretely answers whether `eps=1e-8` is well-chosen or
      should be tightened/loosened. Pairs with the PixelNorm
      eps-sensitivity follow-up pinned above.
- [ ] "Does dispatch round-trip everything?" census: enumerate all
      `TNNet*` subclasses, instantiate each via the layer-name
      dispatch, SaveToString → LoadFromString, assert equality of
      the serialized form. A single test binary; would have caught
      historical "forgot to add the dispatch entry" bugs in a
      single CI run. Same intent as the existing "Layer-name
      registry self-check" entry but spelled out as the concrete
      test plan.
- [ ] Optimizer bake-off on the same tiny CNN used in the
      activation/norm bake-offs: SGD vs SGD+momentum vs Adam vs
      AdamW. Plot train loss + final test acc. Reuses the
      bake-off harness if/when it lands.

#### Tests / numerical-gradient audit
- [ ] Mirror the "Backpropagate audit" pattern for the activation
      family one more time: explicitly enumerate every activation
      layer (LeakyReLU, SELU, HardSigmoid, HardTanh, TanhShrink,
      LogSigmoid, ShiftedReLU, Threshold, SoftSign, SquaredReLU,
      CELU, Mish, Swish, HardSwish, GELU, PixelNorm) and assert each
      has at least one numerical-gradient test by greping
      `TestNeuralNumerical.pas`. Output a coverage matrix as a
      comment block at the top of that file. Cheap, mechanical,
      catches drift.
- [ ] Add a small `TestRMSNormVsLayerNorm` equivalence test:
      under inputs with zero empirical mean, `TNNetRMSNorm` and
      `TNNetLayerNorm` should produce identical outputs (modulo
      learnable scale/bias). Doubles as a sanity test for both
      layers.
- [ ] Determinism CI test re-pin: this keeps appearing across
      lucky-day batches; concretely it's a 20-line test binary
      under `tests/` that builds a 3-layer net, seeds, runs one
      forward+backward, dumps the gradient vector, repeats, and
      `assert(equal)`. Worth doing on its own pass.

#### Examples I'd enjoy writing
- [ ] `examples/SoftPlusVsReLU/` — micro-experiment showing that
      SoftPlus and ReLU train to similar test acc on a tiny MNIST
      subset, with SoftPlus's smoother loss curve visible in the
      per-epoch log. Lands alongside the SoftPlus layer above.
- [ ] `examples/AttentionCopyTask/` — single-head SDPA learns to
      copy a 16-token input to its output. Smallest possible
      end-to-end attention training demo; uses the existing SDPA
      layer with no MHSA dependency. Would close the "we have
      SDPA but no public example" gap.

#### Infrastructure / tooling
- [x] Implement `scripts/audit_tasklist.sh` from the previous
      pass's follow-ups — picking it up explicitly here so the
      next lucky-day pass can grep this tasklist itself for stale
      entries before dispatching. ~30 lines of bash; reads
      `tasklist.md`, extracts unchecked TNNet* names, greps the
      dispatch tables in `neural/neuralnetwork.pas`, flags hits.
      Landed: scripts/audit_tasklist.sh (50 lines). Two-pass grep
      using a sorted known-names set built from every TNNet* token
      in neuralnetwork.pas (covers both class decls and dispatch
      literals). First run surfaces 73 stale occurrences — some
      duplicates (e.g. TNNetScaledDotProductAttention x3,
      TNNetMaskedFill x3), confirming the re-pin pattern; many
      hits are also context-references rather than re-pins, so
      a contributor still needs to judge.
- [ ] `bin/` Volume micro-benchmark (third re-pin). I would
      genuinely enjoy writing this on a future lucky day —
      pinning it again so it stays visible.

#### Documentation
- [ ] `docs/lucky_day_log.md` — short rolling changelog of what
      each lucky-day batch shipped (one row per batch: date,
      seed, landed-items, surprise). Today the record lives only
      in the bottom of `tasklist.md`; a separate file would let
      `tasklist.md` itself be pruned of historical batches once
      they're fully landed.


### Lucky-day batch — 2026-05-15 (post ReverseChannels + SoftPlus-tests + audit_tasklist)

Three serial opus agents dispatched on a lucky day:

- `307335c` — TNNetReverseChannels (channel-axis flip). Parameter-free
  TNNetIdentity descendant; involution so backward == forward. Four
  tests (forward, gradient check, involution, serialization round-trip).
- `a5568ce` — TNNetSoftPlus coverage tests (3 new): identity at zero
  (`SoftPlus(0)=ln(2)`), large-x linearization branch (x>30 returns x),
  and ±extreme-input saturation.
- `15a6de1` — `scripts/audit_tasklist.sh` staleness sweep. First run
  surfaced 73 TNNet* occurrences in unchecked entries, many duplicated.

Test suite: 372 -> 379, all passing. No new bugs in landed layers.

#### Surfaced (not fixed)

- [ ] TNNetSoftPlus derivative-path overflow on very negative x:
      `TNNetSoftPlus.Compute`'s derivative is `1/(1+Exp(-x))`, which raises
      `EOverflow` when `-x` is large (e.g. x=-1e3 → Exp(1e3)). The forward
      output path has the `x>30 ⇒ x` stable branch, but the derivative
      path is not symmetrically guarded. Fix is one extra branch
      (`x < -30 ⇒ deriv := Exp(x)` ≈ 0) and a regression test feeding
      x=-1e3 through Backpropagate asserting no exception and a finite
      input gradient.

#### Follow-ups

- [ ] Triage the audit_tasklist.sh output: 73 hits include both genuine
      re-pins (TNNetScaledDotProductAttention x3, TNNetMaskedFill x3) and
      mere context-references (e.g. "ALiBi-with-MaskedFill composition
      test" — the test is open, not the layer). A `--strict` mode that
      only flags lines starting with the canonical "TNNet… — …" pattern
      (the way a re-pin reads) would cut the false-positive rate.
- [ ] CIFAR/segmentation example using TNNetReverseChannels — its
      headline use case (channel-flip data augmentation or a residual
      branch with reversed channel order) — would give the new layer
      an end-to-end home rather than only living in the test suite.
- [ ] README activation reference: TNNetSoftPlus already has rows in
      the activation table; once the negative-x derivative guard above
      lands, add a "stable on both tails" note next to its entry.


### Lucky-day batch — 2026-05-15 (seed 566186)

Ideas I'd personally enjoy taking on. Mix of bite-sized fixes,
new layers small enough to land cleanly, test/audit work, and a
couple of examples that would exercise recently-landed primitives.

#### Bug-shaped fixes (small, satisfying)

- [ ] Fix TNNetSoftPlus derivative overflow on very-negative x. Add
      the symmetric `x < -30 ⇒ deriv := Exp(x)` branch in
      `TNNetSoftPlus.Compute`/Backpropagate path and a regression
      test in TestNeuralNumerical.pas feeding x=-1e3 through
      Backpropagate (no EOverflow, finite input gradient). This is
      the item already surfaced in the previous lucky-day batch —
      worth closing the loop.
- [ ] Audit TNNetSigmoid and TNNetHardSigmoid for the same negative-x
      / positive-x symmetric-stability question as SoftPlus. If
      Exp(-x) can blow up for x ≪ 0 anywhere along the
      forward/backward path, add a guarded branch + test.
- [ ] TNNetPointwiseSoftMax: now that the exact Jacobian lives in
      Backpropagate, opt the cross-entropy training paths into the
      cheap (y - target) shortcut explicitly, and add a regression
      test that checks the shortcut and the full-Jacobian path agree
      to 1e-5 on a small random batch.

#### Small new layers (each a 1-commit landing)

- [ ] TNNetReverseXY — spatial 180° flip layer (analogue of the
      just-landed TNNetReverseChannels but along X and Y). Involution,
      parameter-free, same four-test shape (forward / gradient check /
      involution / serialization round-trip).
- [x] TNNetFlipX and TNNetFlipY — horizontal and vertical flip
      layers. Used as training-time augmentation modules inside a
      net rather than only as preprocessing. Each is an involution
      and re-uses the ReverseChannels test scaffolding.
- [ ] TNNetAbs — elementwise `|x|` activation. Tiny, but its
      derivative discontinuity at 0 makes it a nice unit-test
      target for the numerical-gradient harness (skip x=0 sampling).
- [ ] TNNetSquare — elementwise `x^2`. Pairs naturally with TNNetAbs
      for "energy"-style heads; derivative is `2*x` so the gradient
      check is cleanly satisfied.
- [ ] TNNetSign — elementwise sign, with straight-through-estimator
      backward (pass gradient through unchanged). Useful for
      binarized-net experiments; add an STE-vs-numerical-gradient
      test that documents the intentional mismatch.
- [x] TNNetClamp — elementwise clamp to [min,max] with subgradient
      passthrough on the active region. Two scalar parameters
      stored in NeuronWeights[0].FData[0..1] (or via Struct[]).
      Serialization round-trip test included.
- [ ] TNNetChannelShuffle — ShuffleNet-style channel permutation
      with a `groups` parameter. Parameter-free; permutation chosen
      by group count. Tests: forward shape, inverse-permutation
      identity when called twice with matching group counts,
      gradient check.

#### Test coverage / audit

- [ ] Numerical-gradient checks for any pooling layer that still
      lacks one. The previous audit covered TNNetAvgPool — sweep
      TNNetMaxPool, TNNetMinPool (if it exists), TNNetAvgChannel,
      TNNetMaxChannel, and add the missing ones.
- [ ] Numerical-gradient check for TNNetEmbedding's weight gradient
      (input gradient is undefined for integer indices; weight side
      should pass). Add to TestNeuralNumerical.pas.
- [ ] Extend `scripts/audit_tasklist.sh` with the `--strict` mode
      flagged in the previous batch: only complain when a line
      matches the canonical "TNNet… — …" re-pin pattern, so the
      false-positive rate from context-references drops.
- [ ] CI-friendly summary mode for the test runner: print a single
      "N passed / M failed" line plus the failing-test names to
      stderr, so a grep-based CI check has a stable contract.

#### Examples (small, runnable, < 1 min CPU)

- [ ] Tiny char-level sequence example (the "quick-start" already
      open in the list). Train on a hand-coded 200-char string,
      predict next char, print loss every epoch — finishes well
      under a minute. Uses TNNetEmbedding + a single
      TNNetFullConnectReLU + TNNetSoftMax. Lands as
      `examples/char_quickstart/`.
- [ ] Channel-flip augmentation example using TNNetReverseChannels:
      train a small CIFAR-ish toy on a 32×32×3 synthetic 2-class
      dataset where the class is encoded in channel ordering, and
      show that a net *with* TNNetReverseChannels as a random
      augmentation generalizes while one without it overfits.
- [ ] Volume micro-benchmark example printing ns/op for
      `TVolume.Add`, `Mul`, `DotProduct`, and `MulAdd` across a
      handful of sizes (already in the open list — promote it to a
      tiny example under `examples/volume_microbench/` so it lives
      alongside other runnable demos).

#### Documentation / polish

- [ ] README: short "involution layers" subsection grouping
      TNNetReverseChannels (landed), TNNetReverseXY (planned),
      TNNetFlipX/Y (planned). One paragraph on what "involution"
      means here and why you'd compose two of them as identity.
- [ ] Doc-comment pass on the activation layers added in the last
      few batches (GEGLU, SwiGLU, DropPath, RotaryEmbedding) so the
      auto-generated layer reference idea (already on the list)
      has clean source to pull from when someone picks it up.

#### Experiments (research-shaped, longer)

- [ ] Compare LayerNorm vs RMSNorm vs GroupNorm on the existing
      CIFAR baseline — same net, swap only the norm layer, log
      validation accuracy and wall-clock. One short markdown
      report under `docs/experiments/`. Should reuse the
      benchmarking scaffolding planned in the open infra item.
- [ ] Activation A/B: re-run the CIFAR baseline with ReLU,
      LeakyReLU, GELU, Swish, Mish, SwiGLU as the only changed
      layer, log curves. Useful for the README activation table
      to cite real numbers.


### Lucky-day batch — 2026-05-15 (seed 995227)

Drew 995227 as my lucky number. With TNNetAbs, TNNetSquare,
TNNetReverseXY and the TNNetSoftPlus negative-x derivative guard
all freshly landed in the last few commits (1821cdb, 134e7ea,
e9f7f60), the natural next bites are the small-layer siblings of
those (FlipX/FlipY/Clamp/Sign), the still-open MHSA breakdown, and
some test/example tasks that exercise what already shipped. Before
adding anything I grep-confirmed each name is absent from the
dispatch tables in `neural/neuralnetwork.pas`.

#### Layers I'd enjoy building (each a 1-commit landing)

- [x] TNNetFlipX — horizontal flip layer (mirror along width).
      Involution, parameter-free; can sit inside the net as a
      training-time augmentation rather than only as preprocessing.
      Reuses the TNNetReverseXY / TNNetReverseChannels four-test
      scaffolding (forward / numerical-gradient / involution /
      SerializationRoundTrip). Verified absent from dispatch.
- [x] TNNetFlipY — vertical flip sibling of TNNetFlipX. Same
      four-test shape; landing them as one PR pair keeps the
      "involution layers" README subsection ready to write in a
      single pass. Verified absent from dispatch.
- [x] TNNetClamp(MinValue, MaxValue) — elementwise clamp with
      sub-gradient passthrough on the active region (zero outside
      `[MinValue, MaxValue]`). Two scalar params stored in
      FFloatSt[0..1]. Tests: forward saturation at both ends,
      numerical-gradient inside the active region (skip sampling
      near the kinks), SerializationRoundTrip. Verified absent.
- [ ] TNNetSign — elementwise `sign(x)` with straight-through-
      estimator backward (pass gradient through unchanged on the
      active region, zero exactly at x=0). Useful for binarized-
      net experiments. Tests: forward sign correctness, STE
      passthrough check, an intentional-mismatch comment in the
      test where the analytic STE differs from the central-
      difference derivative. Verified absent.
- [ ] TNNetBias — bias-only "add a learnable per-channel offset"
      layer (re-pinning from seed 993208 because nothing has
      landed it yet, and I'd genuinely enjoy writing it). The
      gradient on the bias param is just the channel-summed
      output gradient; the input-gradient path is identity. Tests:
      forward additive shape, input-grad passthrough, weight-grad
      central-difference check, SerializationRoundTrip. Verified
      absent from dispatch.
- [ ] TNNetReciprocal — elementwise `1/x` with a small-epsilon
      guard (`x = sign(x) * max(|x|, eps)` before the divide).
      Derivative is `-1/x^2`. Niche but pairs naturally with
      TNNetSquare for "compute Euclidean-norm reciprocal" toy
      heads. Tests: forward, eps-guard saturation at x=0,
      numerical-grad away from zero, SerializationRoundTrip.
- [ ] TNNetExp — elementwise `exp(x)` with an overflow guard
      (clamp x to ≤ 30 before exp), to round out the
      Square/Abs/SoftPlus elementwise family. Derivative is the
      output itself, so the backward path is a one-liner.
- [x] TNNetLog — elementwise `log(max(x, eps))` companion to
      TNNetExp; derivative `1/max(x, eps)`. Tests pin the eps
      behavior. Together with TNNetExp gives a clean Log/Exp
      pair for any future probabilistic-output work.

#### Composite blocks / examples I'd enjoy writing

- [ ] `examples/InvolutionDemo/` — tiny illustration that
      composing two involutions (`TNNetReverseChannels` twice,
      or `TNNetReverseXY` twice, or `TNNetFlipX` twice) acts as
      identity within fp tolerance. Five-screen example that
      doubles as documentation for the new "involution layers"
      README subsection already pinned in the previous batch.
- [ ] `examples/AbsSquareEnergy/` — tiny demo using TNNetAbs and
      TNNetSquare as the headline "energy" feature heads on a
      synthetic regression target `y = ||x||_1` and `y = ||x||_2^2`.
      Both layers landed but neither has an example showing the
      training shape they were designed for.
- [ ] `examples/ReverseXYAugmentation/` — toy 2-class dataset
      where class label depends on spatial orientation; show
      that training with TNNetReverseXY as a random augmentation
      forces a rotation-invariant classifier, while a baseline
      net memorizes orientation. Concrete use-case for the
      just-landed layer.

#### Tests / correctness audit I'd enjoy

- [ ] Numerical-gradient test for TNNetSoftPlus at the freshly-
      guarded x < -30 region. The e9f7f60 fix added the
      `x < -30 ⇒ deriv := Exp(x)` branch; pin it with a test that
      feeds `x = -1e3` and asserts (a) no EOverflow, (b) finite
      input gradient, (c) gradient magnitude ≈ exp(x) at machine
      precision. Closes the "Surfaced (not fixed)" loop for real.
- [ ] Coverage matrix at the top of TestNeuralNumerical.pas: a
      comment block enumerating every TNNet* class and a
      `[grad] [serialize]` column per class, written by a small
      script. Mechanical, but it makes the next audit batch
      pick itself.
- [ ] TNNetAbs near-zero gradient handling: write the test that
      explicitly skips x = 0 sampling (the derivative kink is
      undefined there) and pins the convention used for the
      derivative-at-zero (currently `sign(0) = 0`, i.e. zero
      gradient). Documents the choice in the test, not just code.
- [ ] TNNetSquare gradient-magnitude sanity test at large |x|:
      derivative `2x` grows linearly, so finite-difference eps
      must shrink relative to |x|. Use this layer to pin the
      relative-error tolerance convention for layers whose
      Jacobian scales with the input.
- [ ] Re-pin the property-based gradient harness — even a v0 that
      only randomizes input shape (keeping layer type fixed) for
      the 6 most recently landed layers would catch a lot of
      shape-edge bugs. Already in the list; I want to take it.
- [ ] All-checked-in `[x]` audit: parse the file for every `[x]`
      line that names a `TNNet*` class and assert each name is
      present in either `neural/neuralnetwork.pas`'s dispatch or
      a `Test*` method in `tests/TestNeuralNumerical.pas`. Sister
      tool to `scripts/audit_tasklist.sh` for catching the
      reverse failure mode (claimed-landed but actually missing).

#### Experiments I'm curious about

- [ ] Abs vs Square as the L1-vs-L2 head: train the same small
      regressor with `Sum(TNNetAbs(x))` vs `Sum(TNNetSquare(x))`
      as the final feature on a synthetic noisy-regression task.
      Print loss curves and the implied robust-vs-MSE behavior.
      Concrete teaching artifact for "why L1 is robust to outliers".
- [ ] Flip-augmentation efficacy sweep: with TNNetFlipX (when it
      lands), train a small classifier on a 2-class synthetic
      orientation task with augmentation probability sweeping
      `p ∈ {0.0, 0.25, 0.5, 0.75, 1.0}`. Chart final accuracy.
      Companion to the existing ReverseChannels augmentation
      example pinned above.
- [ ] Activation menagerie unification: with TNNetAbs and
      TNNetSquare now landed, the activation A/B item can be
      extended to include them as the "non-monotone weirdo"
      column. Single chart, one extra config each. Re-pinning
      the bake-off item explicitly with this column added.

#### Infrastructure / tooling

- [ ] `scripts/audit_tasklist.sh --strict` mode flagged in the
      previous batch's follow-ups. Only match lines that begin
      with the canonical `- [ ] TNNet… — …` re-pin pattern, so
      the 73-hit false-positive rate from context-references
      drops to something a human can review in one sitting.
- [ ] Companion script `scripts/audit_landed.sh` that does the
      reverse direction of `audit_tasklist.sh`: every line with
      `[x]` claiming a `TNNet*` landed must point at a real
      class in the dispatch and a `Test*` in the test file.
      Catches the failure mode where the tick is wishful.
- [ ] Tiny CLI `bin/layer_bench <ClassName> <SizeX> <SizeY> <Depth>`
      that builds a 1-layer net and reports ns/op for forward
      and backward. Subsumes the perpetually-re-pinned Volume
      micro-benchmark and extends it to the layer level.

#### Documentation

- [ ] "Elementwise activation layer authoring" mini-guide: with
      Abs/Square/SoftPlus/SquaredReLU all recently landed, capture
      the recurring 4-step pattern (Compute override, Backpropagate
      override using FOutputErrorDeriv, dispatch entry, four-test
      shape) into a short doc. Cheaper than the full layer-authoring
      checklist already pinned and is the most common landing type.
- [ ] Activation reference table: add rows for TNNetAbs, TNNetSquare,
      and TNNetSoftPlus's "stable on both tails" note (once the
      negative-x derivative test above lands). Pure README pass.


### Lucky-day batch — 2026-05-15 (post FlipX/FlipY + Clamp batch)

Two serial opus agents dispatched on a lucky day (seed 921753).
Landed:

- `1c86823` — TNNetFlipX (horizontal / SizeX mirror) and TNNetFlipY
  (vertical / SizeY mirror) as sibling involution layers. Parameter-free
  TNNetIdentity descendants modeled after TNNetReverseXY. Backward
  reuses the same index map as forward. Both registered in both
  CreateLayer dispatch sites. Eight new tests (four per layer:
  Forward / GradientCheck / Involution / SerializationRoundTrip).
- `c375c1c` — TNNetClamp(MinValue, MaxValue) elementwise saturation,
  TNNetReLUBase descendant. Two-float serialization via
  FFloatSt[0]=MinValue, FFloatSt[1]=MaxValue (precedent: TNNetThreshold).
  Cached in-range indicator in FOutputErrorDeriv so Backpropagate is
  a single multiply. Four new tests
  (Forward / GradientCheck / ExtremeInputSaturation / SerializationRoundTrip).

Test suite: 390 → 402, all passing. No bugs surfaced.

#### Natural follow-ups

- [ ] README "involution layers" subsection (already pinned earlier):
      with TNNetReverseChannels + TNNetReverseXY + TNNetFlipX +
      TNNetFlipY all landed, the four-layer family is finally complete
      and the subsection can be written in one pass.
- [ ] Flip-augmentation example using TNNetFlipX/TNNetFlipY as
      training-time augmentation modules inside the net (rather than
      preprocessing). Pairs with the existing ReverseXYAugmentation /
      ReverseChannels-augmentation example ideas.
- [ ] TNNetClamp README activation-table row — add next to TNNetSoftCapping
      (the closest existing entry; both saturate, Clamp is the
      hard-edged sibling). One-line description + tiny snippet.
- [ ] TNNetClamp vs TNNetReLU6 vs TNNetHardTanh comparison: all three
      are bounded-output activations with subgradient passthrough.
      Worth a short note in the eventual activation cheat sheet
      clarifying when to pick each (Clamp = arbitrary bounds, ReLU6
      = MobileNet quantization, HardTanh = standard ±1).
- [ ] TNNetClamp kink-region test: at hand-picked x exactly equal to
      MinValue and MaxValue, document the convention chosen by the
      indicator (currently: derivative=0 at the exact boundary, since
      the in-range indicator is a strict-inequality check). Pin with
      a tiny no-central-differences test mirroring the open HardShrink
      kink test entry.


### Lucky-day batch — 2026-05-15 (seed 176582)

Lucky number 176582 drawn. With the involution family complete
(ReverseChannels/ReverseXY/FlipX/FlipY), Clamp landed, and the
elementwise Abs/Square family freshly in tree, today's wishlist is
biased toward (a) closing small obvious gaps around those landings,
(b) a fresh round of tiny gradient-checkable layers I'd enjoy
authoring, and (c) one or two experiments that turn the recent
additions into visible artifacts. Names below were grep-verified
absent from `neural/neuralnetwork.pas` dispatch before listing.

#### Tiny new elementwise layers (each a one-commit landing)

- [ ] TNNetSign — `y = sign(x)` with straight-through-estimator
      backward (gradient passes through unchanged, zero at x=0).
      Forward, STE-vs-numerical mismatch test (documenting the
      intentional gap), serialization round-trip. Building block
      for binarized-net experiments. Verified absent from dispatch.
- [x] TNNetNeg — `y = -x`. Trivial, but plays well as a connective
      in residual-subtraction blocks (`x - F(x)` patterns). Backward
      is identity-negated. Four-test shape (forward / gradient /
      involution-via-double-negate / round-trip). Landed: TNNetReLUBase
      descendant caching -1 in FOutputErrorDeriv, registered in both
      CreateLayer dispatch tables, four numerical tests added.
- [ ] TNNetReciprocal — `y = 1 / sign(x) * max(|x|, eps)`, eps in
      FFloatSt[0] (default 1e-6). Derivative `-1/x^2`. Tests pin
      the eps-guard saturation at x=0 and the numerical gradient
      away from zero. Pairs naturally with TNNetSquare for "compute
      Euclidean-norm reciprocal" toy heads.
- [x] TNNetExp — `y = exp(min(x, 30))` with the symmetric overflow
      guard, derivative is the output itself (one-line Backpropagate
      via FOutputErrorDeriv := FOutput).
- [x] TNNetLog — `y = log(max(x, eps))` companion to TNNetExp, eps
      in FFloatSt[0] (default 1e-8). Derivative `1/max(x, eps)`.
      Tests pin the eps-region behavior plus the standard
      gradient check away from the floor.
- [x] TNNetSqrt — `y = sqrt(max(x, eps))`. Derivative `1/(2*y)`
      reuses the cached output, so backward is a single multiply.
      Sister to TNNetSquare; together they let "energy" heads be
      composed both ways.
- [ ] TNNetBias — bias-only "add a learnable per-channel offset"
      (re-pinned three batches in a row now; I'd genuinely enjoy
      writing it). Bias-gradient is the channel-summed output
      gradient, input-grad is identity. Four-test shape (forward
      additive shape, input-grad passthrough, weight-grad
      central-difference check, SerializationRoundTrip). Verified
      absent.
- [ ] TNNetMul — multiplicative companion to TNNetBias: learnable
      per-channel scale. Input gradient is `dy * scale`, weight
      gradient is the channel sum of `dy * x`. Pairs with
      TNNetBias to give a hand-rolled affine transform separable
      from FullConnect.
- [ ] TNNetIdentityScale — fixed (non-learnable) per-tensor scalar
      multiplier stored in FFloatSt[0]. Useful for the
      "warm-up scaling" trick where you damp a residual branch by
      a constant. Tiny but explicit about non-learnability.

#### Normalization variants I'd enjoy adding

- [ ] TNNetL2Normalize — divide by `sqrt(sum(x^2) + eps)` per
      sample (or per channel, configurable axis via FStruct[0]).
      Parameter-free; backward is the unit-norm Jacobian shared
      with PixelNorm. PixelNorm reduces over depth; L2Normalize
      can reduce over the full volume or a chosen axis, so they
      complement rather than duplicate. Tests: forward unit-norm,
      central-difference gradient, axis-config round-trip.
- [ ] TNNetUnitNorm — alias for L2Normalize on the full volume
      (the Keras name). One-line registration so the canonical
      name parses.
- [ ] TNNetMinMaxNorm — `(x - min(x)) / (max(x) - min(x) + eps)`
      per sample. Non-differentiable at the min/max argmax points;
      backward routes the gradient through the in-range slope
      with zero at the active endpoints. A small, fun test bed
      for the "discontinuous Jacobian" failure mode.
- [ ] TNNetZScore — `(x - mean) / std` per sample (no learnable
      scale/bias) — the unparameterised core of LayerNorm. Useful
      as a normalization primitive that doesn't add weights.

#### Reduction / shape layers I'd enjoy adding

- [ ] TNNetCumSum — cumulative sum along a configurable axis
      (FStruct[0]). Linear, so backward is a reverse cumulative
      sum. Useful for any sequence-position-dependent feature
      you want without RoPE-style trigonometry.
- [ ] TNNetGather — index-into-a-channel layer for "select head h
      from a (3*d_model) Q|K|V slab" patterns. Today this is done
      via TNNetSplitChannels; a single-channel Gather would be
      lighter for the MHA breakdown. Configurable channel index.
- [ ] TNNetTopK — keep only the top-K activations per spatial cell
      along the depth axis, zeroing the rest. Useful for sparse
      attention experiments; the K-th largest is the cutoff, with
      argmax-style routing of the gradient. K in FStruct[0].
- [ ] TNNetRoll — circular shift along a chosen axis (FStruct[0]
      axis, FStruct[1] offset). Parameter-free, deterministic
      permutation; backward is the inverse roll. Pairs with the
      involution-layer family already in tree.

#### Composite blocks I'd enjoy shipping

- [ ] TNNetSwiGLUFeedForward block helper (re-pinned several
      times). With TNNetSwiGLU + TNNetLayerNorm + dense layers all
      in tree, this is a 10-line `AddSwiGLUFeedForward(NN, d_model,
      d_ff)` builder. No new layer types; pure ergonomics win.
- [ ] TNNetPreNormResidual helper (also re-pinned for the sixth
      time). `y = x + Sublayer(LayerNorm(x))` as a single
      AddPreNormResidual entrypoint. Treat the repeat-pinning as
      universe pressure to actually ship it on the next lucky day.
- [ ] TNNetAffineBlock — `Bias(Mul(x))` (or Mul then Bias)
      builder, once TNNetBias and TNNetMul land. Lets one compose
      a learnable affine transform out of two primitives instead
      of a FullConnect when no cross-channel mixing is needed.

#### Tests / numerical-gradient audit I'd enjoy

- [ ] Negative-x derivative regression test for TNNetSoftPlus
      (the "Surfaced (not fixed)" item from the post-ReverseChannels
      batch — the e9f7f60 commit landed the fix but a pinning test
      with `x = -1e3` would close the loop explicitly).
- [ ] TNNetClamp kink-region test (pinned in the previous batch).
      Hand-picked `x = MinValue` and `x = MaxValue`, no central
      differences — assert the derivative-at-boundary convention
      (currently 0 from the strict-inequality indicator). Two
      assertions plus a comment block documenting the choice.
- [ ] Kink-region test parametric helper: with the Clamp /
      HardShrink / SoftShrink / Threshold / ShiftedReLU / HardTanh
      all now in tree, the "no-central-difference, hand-picked
      kink convention" pattern repeats. Capture it as a single
      `AssertKinkDerivative(layer, x_kink, expected_dydx)` helper.
- [ ] Numerical-gradient check for TNNetMinPool (already exists;
      it's the sibling of TNNetAvgPool/MaxPool that the
      transform/reshape/pooling audit didn't reach because MinPool
      pre-dates the audit). Pure coverage gap.
- [ ] Numerical-gradient check for TNNetMinChannel (the
      global-min-pool sibling of TNNetAvgChannel/MaxChannel).
      Mirrors the existing TestAvgChannel test pattern.
- [ ] Coverage matrix at the top of TestNeuralNumerical.pas —
      re-pinning the per-class `[grad] [serialize]` block from
      seed 995227. Mechanical to generate; would let the next
      batch pick its targets by reading the comment instead of
      grep-ing the dispatch.
- [ ] Run `scripts/audit_tasklist.sh` (landed in 15a6de1) plus the
      planned `--strict` mode in one pass and triage the resulting
      hit list. A focused "tasklist hygiene" sitting that prunes
      stale entries instead of adding new ones — the inverse of
      a normal lucky-day batch.

#### Experiments I'd enjoy running

- [ ] Involution-composition smoke check (across the whole family
      now that it's complete): for each of TNNetReverseChannels,
      TNNetReverseXY, TNNetFlipX, TNNetFlipY, build a 2-layer net
      composing the layer with itself and assert the output equals
      the input to within fp tolerance, on a random volume. Four
      lines per layer; one short test file under `tests/`.
- [ ] FlipX/FlipY augmentation efficacy on a synthetic orientation
      task (re-pinning from seed 995227 now that the layers
      actually exist). Train a small classifier with `p ∈ {0, 0.25,
      0.5, 0.75, 1.0}` flip probability; chart accuracy.
- [ ] TNNetClamp vs TNNetReLU6 vs TNNetHardTanh convergence
      comparison on a tiny CIFAR stub (re-pin of the comparison
      already in the list — the three are all bounded subgradient
      activations and the test is now buildable).
- [ ] "Where does the gradient go?" visualizer for the in-tree
      saturation activations: for each of HardTanh / SoftCapping /
      Clamp / ReLU6 / SoftSign / Tanh, feed a 1D ramp `x ∈
      [-3, 3]` and print y(x), dy/dx as a tiny PGM strip. Cheap
      teaching artifact; reuses the existing forward path only.

#### Examples I'd enjoy writing

- [ ] `examples/InvolutionDemo/` — re-pinned from seed 995227.
      With the full ReverseChannels + ReverseXY + FlipX + FlipY
      family now in tree, the demo is "compose any of these
      twice, assert identity to fp tolerance" in 50 lines.
      Doubles as the example backing the "involution layers"
      README subsection.
- [ ] `examples/EnergyHeads/` — tiny regression demo using
      TNNetAbs (L1-energy target) and TNNetSquare (L2-energy
      target) as the final feature heads. Two side-by-side
      models, one chart, demonstrates the "L1 robust to outliers"
      claim concretely. Pairs with the Abs/Square layers that
      just landed.
- [ ] `examples/BiasOnlyTuning/` — when TNNetBias lands, ship a
      tiny example that freezes a pretrained classifier and
      fine-tunes only the new TNNetBias layers on a new task.
      Concrete demonstration of "bias-tuning" / BitFit-style
      cheap adaptation.

#### Tooling / dev experience I'd enjoy shipping

- [ ] `scripts/audit_landed.sh` companion to `audit_tasklist.sh`
      (already pinned in seed 995227): every `[x]` claiming a
      `TNNet*` landed must point at a real class in the dispatch
      and at least one `Test*` method. Catches wishful ticks.
- [ ] `scripts/audit_tasklist.sh --strict` mode (also pinned).
      Only match lines beginning with the canonical
      `- [ ] TNNet… — …` re-pin shape, trim the 73-hit
      false-positive rate. Two-line awk change in practice.
- [ ] `bin/layer_bench` CLI (pinned in seed 995227) — build a
      1-layer net and report ns/op for forward + backward,
      configurable shape and class. Subsumes the perpetually
      re-pinned Volume micro-benchmark and extends it to layers.
- [ ] `tests/SmokeTest.lpr` (pinned in earlier batches) — the
      five fastest gradient checks, runs in under a second.
      A real signal for any future CI shim before RunAll.sh is
      wired in.

#### Documentation I'd enjoy writing

- [ ] README "involution layers" subsection (re-pinned multiple
      times; the family is now complete and the subsection can
      be written in one pass). Four rows, one paragraph on
      involutions, one tiny snippet showing
      `Net.AddLayer([TNNetFlipX.Create(), TNNetFlipX.Create()])`
      acts as identity.
- [ ] "Elementwise activation layer authoring" mini-guide
      (pinned in seed 995227). With Abs/Square/SoftPlus/Clamp
      all recently landed, the recurring four-step pattern
      (Compute, Backpropagate using FOutputErrorDeriv, dispatch
      entry, four-test shape) is well-established and worth
      capturing as a short doc with TNNetClamp as the worked
      example.
- [ ] "Saturation activations cheat sheet" — `docs/saturation.md`
      one-pager covering Clamp / HardTanh / SoftCapping / ReLU6 /
      SoftSign / Tanh / HardSigmoid: formula, range, kink count,
      typical use. Narrower than the full activations cheat sheet
      already pinned, and the roster is mature enough to write it
      today.

#### Stretch / ambitious (re-pinning for visibility)

- [ ] TNNetMultiHeadSelfAttention as a real layer (pinned now
      across many batches). Wrap H copies of TNNetSDPA + Q/K/V
      projections + output projection. The blocker for Tiny GPT
      and the highest-leverage open layer in the file.
- [ ] Tiny GPT char-level example end-to-end on CPU. Once MHSA
      lands, the building blocks are all present (LayerNorm,
      RoPE, SDPA, MaskedFill, SwiGLU, LogSoftMax, AddPositional
      Embedding). Ship `tinyshakespeare.txt` under
      `examples/TinyGPT/data/`, 2-layer model, generation that
      produces recognizably-Shakespeare-flavored output.
- [ ] Mixture-of-Experts routing layer (pinned multiple batches).
      Top-k softmax gate over N experts with load-balancing
      auxiliary loss. Best done after MHSA.


### Lucky-day batch — 2026-05-15 (seed 64793, post Sqrt+Exp+Log batch)

Three serial opus agents dispatched on a self-described lucky day.
Landed (each its own commit):

- `37b24f0` — TNNetSqrt: `y = sqrt(max(x, 1e-6))`, parameter-free
  TNNetReLUBase descendant. Derivative `0.5/y` cached in
  FOutputErrorDeriv. Three tests (Forward / GradientCheck /
  SerializationRoundTrip).
- `e830784` — TNNetExp: `y = exp(min(x, 30))`, parameter-free
  TNNetReLUBase descendant. Derivative equals output (single
  cache write into FOutputErrorDeriv). Three tests.
- `3f55c84` — TNNetLog: `y = ln(max(x, 1e-8))`, parameter-free
  (hard-coded eps to keep the Sqrt/Exp four-test shape; the
  earlier tasklist entry called for FFloatSt[0] but the
  parameter-free shape was preferred for consistency).
  Derivative `1/max(x, 1e-8)`. Three tests.

Test suite: 402 -> 411, zero failures. No bugs surfaced.

The elementwise transcendental family (Sqrt / Exp / Log) now
complements the just-landed Abs / Square / Clamp elementwise
landings. Together they cover the common building blocks for
"compute Euclidean-norm reciprocal", "log-domain probability
heads", and other tiny-but-useful composed primitives.

#### Natural follow-ups

- [ ] TNNetReciprocal — `y = 1/sign(x) * max(|x|, eps)`. The
      remaining open elementwise from the seed 176582 batch.
      Pairs with TNNetSquare to give a hand-rolled
      `Reciprocal(Sqrt(Square))` Euclidean-norm-reciprocal head.
- [ ] Exp/Log compose-as-identity test: `Log(Exp(x))` should
      reconstruct x to within fp tolerance on a small input
      range. One-line property check on top of the new layers.
- [x] Activation reference table rows in README for TNNetSqrt,
      TNNetExp, TNNetLog matching the "Created with ..." style
      already used for the recently-landed activations.
      Landed alongside Reciprocal/Neg/Sin/Cos rows in the
      "docs: README activation-table rows for transcendentals
      + Neg/Sin/Cos" commit.
- [ ] Sqrt/Exp/Log saturation tests at extreme inputs (mirroring
      the HardTanh / SoftCapping / Clamp pattern): for Exp at
      x=1e3 assert no overflow (clamp triggers), for Log at
      x=-1e3 assert no exception (eps clamp), for Sqrt at
      x=-1e3 assert no exception (eps clamp). Closes the
      coverage gap on the new layers.


### Lucky-day batch — 2026-05-15 (seed 747583)

Fresh draw on a lucky day. With Sqrt/Exp/Log just landed completing
the transcendental elementwise trio, this batch focuses on the
remaining gaps the previous batch's "natural follow-ups" pointed at,
plus a few small layers I'd personally enjoy authoring. Every TNNet*
name below was grep-verified absent from `neural/neuralnetwork.pas`
before listing (the one exception, TNNetGaussianActivation, is
explicitly already-landed and is referenced only for context).

#### Closing the Sqrt/Exp/Log follow-up loop
- [x] TNNetReciprocal — `y = 1 / (sign(x) * max(|x|, eps))`, eps in
      FFloatSt[0] (default 1e-6). Derivative `-1 / (sign(x) *
      max(|x|, eps))^2 = -y^2 * sign(x) / sign(x) = -y * y * sign(x)`
      (cache the output, multiply-and-negate in backward). Four-test
      shape: forward at typical inputs, eps-guard saturation at x=0,
      central-difference gradient away from zero, SerializationRoundTrip.
      Pairs with TNNetSquare/TNNetSqrt to give a hand-rolled
      "Euclidean-norm reciprocal" head as `Reciprocal(Sqrt(Square(x)))`.
      Landed parameter-free (matching the Sqrt/Exp/Log family decision)
      in 08914d9; derivative simplifies to -y*y in the unclamped region
      because s*|x| = x, so the sign factor cancels.
- [x] Exp/Log compose-as-identity test: `Log(Exp(x))` should
      reconstruct `x` to within fp tolerance on a small bounded input
      (say `|x| <= 5` to stay well clear of the eps and overflow
      clamps). Three-line test on top of the existing layers; mirrors
      the TanhShrink+Tanh composition test pattern.
      Landed in becaf13 as TestExpLogComposeAsIdentity (1e-4 tol on
      values ~[-4.8, 4.2]).
- [x] Sqrt/Exp/Log saturation tests at ±extreme inputs (mirrors
      HardTanh/SoftCapping/Clamp): Exp at x=1e3 must not overflow
      (clamp at 30 triggers, output ≈ exp(30)); Log at x=-1e3 must
      not raise (eps clamp triggers, output ≈ ln(1e-8)); Sqrt at
      x=-1e3 must not raise (eps clamp). Three tiny tests, one per
      layer, closes the coverage gap flagged at the bottom of the
      previous batch.
      Landed in becaf13 as Test{Sqrt,Exp,Log}ExtremeInputSaturation;
      exp(-1e3) underflows cleanly to 0 in fp32 (no clamp needed on
      the negative side).
- [ ] README activation-table rows for TNNetSqrt, TNNetExp, TNNetLog
      in the "Created with ..." style. One row each; bring the
      reference table back in sync with the dispatch.

#### Tiny elementwise layers I'd enjoy adding
- [x] TNNetNeg — `y = -x`. Trivial Compute / Backpropagate
      (`prev.err -= self.err`), but plays nicely as a connective in
      residual-subtraction blocks (`x - F(x)` patterns) and as a
      smoke test for the involution-via-double-negate property.
      Four-test shape (forward / gradient / involution-twice /
      SerializationRoundTrip). Landed as TNNetReLUBase descendant.
- [ ] TNNetBias — bias-only "add a learnable per-channel offset"
      (re-pinning explicitly: this is the fourth batch in a row
      where it shows up unimplemented, so promoting it to a
      take-it-next item). Bias-gradient is the channel-summed
      output gradient, input-grad is identity. Four-test shape
      (forward additive correctness, input-grad passthrough,
      weight-grad central-difference, SerializationRoundTrip).
      Useful for BitFit-style fine-tuning experiments.
- [ ] TNNetMul — multiplicative companion: learnable per-channel
      scale (no bias). Input gradient `dy * scale`; weight gradient
      `sum_pix(dy * x)` per channel. Together with TNNetBias gives
      a hand-rolled separable affine transform (TNNetAffineBlock
      builder, also pinned). Verified absent.
- [ ] TNNetIdentityScale — fixed (non-learnable) per-tensor scalar
      multiplier in FFloatSt[0]. Useful for the "warm-up scaling"
      trick where a residual branch is damped by a constant. Tiny
      but explicit about non-learnability — contrasts with
      TNNetLayerScale which is learnable.

#### Normalization layers I'd enjoy adding
- [ ] TNNetL2Normalize — divide by `sqrt(sum(x^2) + eps)` per
      sample, configurable reduction axis via FStruct[0] (0 = full
      volume, 1 = per-channel over spatial, 2 = per-position over
      depth). Parameter-free; backward is the unit-norm Jacobian
      shared with TNNetPixelNorm (which reduces over depth only).
      Tests: forward unit-norm assertion, central-difference
      gradient, axis-config round-trip. Verified absent.
- [ ] TNNetZScore — `(x - mean) / std` per sample with no
      learnable scale/bias — the unparameterised core of LayerNorm.
      Useful as a normalization primitive that doesn't add weights
      (e.g. when normalization sits inside a frozen feature
      extractor). Tests: forward zero-mean unit-std, gradient
      check, SerializationRoundTrip. Verified absent.
- [ ] TNNetMinMaxNorm — `(x - min(x)) / (max(x) - min(x) + eps)`
      per sample. Non-differentiable at the argmin/argmax cells;
      backward routes the gradient through the in-range slope
      with zero at the active endpoints. A small, fun test bed
      for the "discontinuous Jacobian" failure mode. Verified
      absent.

#### Reduction / shape layers I'd enjoy adding
- [ ] TNNetCumSum — cumulative sum along a configurable axis
      (FStruct[0] ∈ {0, 1, 2}). Linear, so backward is the reverse
      cumulative sum along the same axis. Useful for any
      sequence-position-dependent feature you want without
      RoPE-style trigonometry, and a clean test bed for the
      reverse-cumsum identity. Verified absent.
- [ ] TNNetTopK — keep only the top-K activations per spatial
      cell along the depth axis, zeroing the rest. K in
      FStruct[0]. Argmax-style routing of the gradient (only
      the top-K positions receive incoming error). Useful for
      sparse-attention experiments and as a tiny stepping stone
      toward MoE routing. Tests: forward sparsity count,
      seeded numerical-gradient at typical inputs, round-trip.
      Verified absent.
- [ ] TNNetRoll — circular shift along a chosen axis (FStruct[0]
      axis, FStruct[1] offset). Parameter-free, deterministic
      permutation; backward is the inverse roll. Pairs naturally
      with the involution-layer family (a single roll is not an
      involution, but a roll(+k) followed by roll(-k) is). Tests:
      forward shift correctness, inverse-roll round-trip,
      gradient check, SerializationRoundTrip. Verified absent.

#### Activation completists
- [ ] TNNetSnake — `y = x + (1/alpha) * sin(alpha * x)^2`. The
      Ziyin 2020 periodic activation. Configurable alpha in
      FFloatSt[0] (default 1.0). Closed-form derivative
      `1 + sin(2 * alpha * x)`. Niche but slots cleanly into the
      activation menagerie bake-off and exercises a non-monotone
      activation path. Verified absent.
- [ ] TNNetSwish-with-learnable-beta — current TNNetSwish has a
      fixed beta=1 (which is TNNetSiLU's whole point). A version
      with a single learnable scalar beta exercises the
      learnable-scalar-on-an-activation path and is fun for an
      ablation study. Tests follow the LayerScale template.
- [ ] TNNetMish-stable — current TNNetMish uses the standard
      `x * tanh(softplus(x))` form. A "stable on both tails"
      variant using the softplus stable-branch trick from the
      recent TNNetSoftPlus negative-x guard would close the
      symmetric-stability story. Pairs with a regression test at
      x = -1e3 and x = +1e3.

#### Composite block helpers I'd enjoy shipping
- [ ] TNNetAffineBlock — once TNNetBias and TNNetMul land,
      ship `AddAffineBlock(NN)` builder that wires
      `Mul → Bias` (or `Bias → Mul`, pick a convention) as a
      learnable per-channel affine without cross-channel mixing.
      Concrete user-facing payoff for the two new primitives.
- [ ] TNNetSwiGLUFeedForward — re-pinning for the seventh+ time.
      With SwiGLU + LayerNorm + dense layers all in tree, this
      is a 10-line `AddSwiGLUFeedForward(NN, d_model, d_ff)`
      builder. Treating the repeated re-pinning as universe
      pressure to actually ship it on the next lucky day.
- [ ] TNNetPreNormResidual — re-pinning. `y = x + Sublayer(LayerNorm(x))`
      one-liner builder. Sister to the SwiGLUFeedForward task.

#### Tests / numerical-gradient audit
- [ ] Run `scripts/audit_tasklist.sh` (landed in 15a6de1) on
      the current tasklist and triage the resulting hit list.
      A focused "tasklist hygiene" sitting — the inverse of a
      normal lucky-day batch. The previous batch added new
      entries; this one would prune stale ones.
- [ ] Coverage matrix at the top of TestNeuralNumerical.pas:
      enumerate every TNNet* class with a `[grad] [serialize]`
      column. Mechanical to generate; would let the next batch
      pick its targets by reading the comment instead of
      grep-ing the dispatch. Re-pinning across multiple batches.
- [ ] Numerical-gradient check for TNNetMinPool (the sibling of
      MaxPool/AvgPool that the transform/reshape/pooling audit
      didn't reach). Pure coverage gap. Re-pinning from seed 176582.
- [ ] Numerical-gradient check for TNNetMinChannel (the
      global-min-pool sibling of TNNetAvgChannel/MaxChannel).
      Mirrors the existing TestAvgChannel pattern.
- [ ] Property-based gradient harness v0 — even a version that
      only randomizes input shape (keeping layer type fixed) for
      the 6 most recently landed layers would catch a lot of
      shape-edge bugs. Re-pinning across many batches; I want
      to take it.

#### Experiments I'm curious about
- [ ] Euclidean-norm reciprocal head composition test: once
      TNNetReciprocal lands, build `Reciprocal(Sqrt(Square(x)))`
      as a 3-layer "compute 1/||x||" head, train a tiny regressor
      to fit a known target, and confirm training works end-to-end
      through three elementwise transcendental layers. Headline
      use case for the new Sqrt/Exp/Log/Reciprocal family.
- [ ] CumSum + AddPositionalEmbedding comparison: with TNNetCumSum
      landed, the integer-position-encoding family becomes
      composable. Train a tiny next-token model with sinusoidal
      AddPositionalEmbedding vs a CumSum-based learned position
      encoding (CumSum applied to a constant ones tensor gives
      a linear position feature). One-chart, two-config experiment.
- [ ] TopK sparsity sweep: with TNNetTopK landed, train the same
      tiny autoencoder bottleneck with K ∈ {1, 2, 4, 8, 16, full},
      chart reconstruction loss vs sparsity. Concrete companion
      to the HardShrink/SoftShrink sparsity experiments already
      pinned.
- [ ] L2Normalize vs PixelNorm head comparison on a tiny
      generator-shaped net. PixelNorm reduces over depth only;
      L2Normalize over the full volume. Bake-off shows whether
      the StyleGAN choice of depth-only is doing real work.

#### Examples I'd enjoy writing
- [ ] `examples/EuclideanNormHead/` — once TNNetReciprocal lands,
      tiny demo showing `Reciprocal(Sqrt(Square(x)))` composed as
      a Euclidean-norm-reciprocal head. The headline use case for
      the elementwise family's 2026-05-15 batch.
- [ ] `examples/BiasOnlyTuning/` — once TNNetBias lands, freeze a
      pretrained classifier and fine-tune only the TNNetBias
      layers on a new task. BitFit demonstration in <100 lines.
      Already pinned in seed 995227 batch; re-pinning because
      TNNetBias still hasn't landed.
- [ ] `examples/CumSumPositionEncoding/` — tiny char-level demo
      using CumSum on a constant input as a learned linear
      position feature. Compares to AddPositionalEmbedding on the
      same toy task. Companion to the CumSum experiment above.

#### Tooling / dev experience
- [ ] `scripts/audit_landed.sh` — companion to audit_tasklist.sh
      (re-pinned several batches now). Every `[x]` line claiming
      a TNNet* landed must point at a real class in the dispatch
      AND at least one Test* method in the test files. Catches
      the wishful-tick failure mode the audit_tasklist.sh script
      can't see.
- [ ] `scripts/audit_tasklist.sh --strict` — re-pinned. Match
      only lines beginning with the canonical
      `- [ ] TNNet… — …` re-pin shape. Two-line awk change in
      practice, trims the 73-hit false-positive rate.
- [ ] `bin/layer_bench <ClassName> <SizeX> <SizeY> <Depth>` —
      tiny CLI that builds a 1-layer net and reports ns/op for
      forward + backward. Re-pinned across multiple batches;
      subsumes the perpetually-re-pinned Volume micro-benchmark.

#### Documentation
- [ ] README "elementwise transcendental layers" subsection
      covering TNNetSqrt / TNNetExp / TNNetLog (just landed),
      TNNetAbs / TNNetSquare (recently landed) and the planned
      TNNetReciprocal. One paragraph on the eps-guard convention
      shared across the family, plus a tiny snippet showing the
      Euclidean-norm-reciprocal composition. Becomes the natural
      home for the "compute a norm without FullConnect" pattern.
- [ ] "Saturation activations cheat sheet" in `docs/saturation.md`
      (re-pinned from seed 176582). One-pager covering Clamp /
      HardTanh / SoftCapping / ReLU6 / SoftSign / Tanh /
      HardSigmoid: formula, range, kink count, typical use.
- [ ] "Elementwise activation layer authoring" mini-guide
      (re-pinned from seed 995227). With Abs/Square/Sqrt/Exp/Log/
      SoftPlus/Clamp all recently landed, the recurring four-step
      pattern is well-established and worth capturing as a short
      doc, with TNNetSqrt or TNNetExp as the worked example
      (the smallest non-trivial elementwise landings on record).

#### Stretch / ambitious (re-pinned for visibility)
- [ ] TNNetMultiHeadSelfAttention — still the headline blocker
      for TinyGPT. The SDPA + RoPE + MaskedFill + ALiBi pieces
      are all in tree; MHSA is the final missing layer.
- [ ] Tiny GPT char-level example on CPU — pinned across every
      lucky-day batch. Once MHSA lands, all the pieces exist.
- [ ] Mixture-of-Experts routing — pinned multiple batches.
      Best done after MHSA so it has a real host architecture.


### Lucky-day batch — 2026-05-15 (seed 602689, post Reciprocal + Sqrt/Exp/Log-saturation batch)

Two serial opus agents dispatched on a lucky day. Landed:

- `08914d9` — TNNetReciprocal: `y = 1/(sign(x) * max(|x|, eps))`, eps=1e-6.
  Parameter-free TNNetReLUBase descendant (chose to match the Sqrt/Exp/Log
  family convention rather than the FFloatSt[0]-configurable spec the
  tasklist suggested). Four new tests: Forward, EpsGuard (x=0 -> 1e6),
  GradientCheck (inputs biased away from zero), SerializationRoundTrip.
- `becaf13` — Sqrt/Exp/Log saturation + Exp/Log compose-as-identity tests.
  Four new tests: TestSqrtExtremeNegativeInputSaturation,
  TestExpExtremeInputSaturation, TestLogExtremeNegativeInputSaturation,
  TestExpLogComposeAsIdentity.

Test suite: 411 -> 419, all passing. No bugs surfaced.

#### Surprises / clarifications

- **TNNetReciprocal derivative spec correction.** The tasklist (and most
  textbooks) wrote the derivative as `-y^2 * sign(x)`. In the unclamped
  region `sign(x) * |x| = x` exactly, so `y = 1/x` and `dy/dx = -1/x^2 = -y^2`
  — the sign factor cancels. Using `-y*y*sign(x)` would flip the gradient
  on negative inputs and fail the central-difference check. The landed
  implementation uses `-y*y` and the doc comment notes the cancellation.
  Future Reciprocal-style layers should be checked for the same trap.
- **exp(-1e3) underflows cleanly to 0.0 in FP32**, no NaN/Inf — so
  TNNetExp's one-sided positive-only clamp at 30 is sufficient. A
  symmetric negative-side clamp is NOT needed and should not be added
  (would break the cached-output backward path on tiny inputs).
- **ln(1e-8) ≈ -18.420680... is exactly representable in FP32** to
  within ~1e-7, so the saturation test's 1e-3 tolerance is comfortably
  over-spec.

#### Natural follow-ups

- [x] TNNetReciprocal README activation-table row (matching the
      "Created with ..." style used for the recently-landed activations).
      Landed alongside Sqrt/Exp/Log/Neg/Sin/Cos rows in the
      "docs: README activation-table rows for transcendentals
      + Neg/Sin/Cos" commit.
- [ ] Euclidean-norm reciprocal head example
      (`examples/EuclideanNormHead/`) — now buildable: compose
      `Reciprocal(Sqrt(Square(x)))` on a tiny regression target. Pinned
      multiple batches; the final missing primitive (Reciprocal) is
      now in tree.
- [ ] TNNetReciprocal gradient-magnitude sanity check at small |x|
      (where `|dy/dx| = 1/x^2` blows up): mirror the TNNetSquare
      "Jacobian scales with input" relative-tolerance test pinned in
      seed 995227. Pins the central-difference noise floor for this
      family.

### Lucky-day batch — 2026-05-15 (seed 141347)

Fresh draw on a lucky day. The elementwise transcendental family
(Sqrt / Exp / Log / Reciprocal) just completed in earlier batches; this
draw deliberately steps away from that family to look at the periodic /
parametric / sparsity-routing corners of the layer space, plus a couple
of small composite-block builders that are unblocked by all the recent
landings. Every TNNet* name below was grep-verified absent from
`neural/neuralnetwork.pas` before listing.

#### Periodic activations I'd enjoy authoring
- [x] TNNetSin — `y = sin(x)`, derivative `cos(x)`. The SIREN paper's
      core ingredient; pairs naturally with a tiny SIREN-flavored
      regression example fitting a 1D function with three Sin-activated
      dense layers. Parameter-free, four-test shape (forward at typical
      inputs, gradient check, periodicity-as-property
      `Compute(x) == Compute(x + 2π)` to within fp tol,
      SerializationRoundTrip). Verified absent.
      Landed: TNNetReLUBase descendant caching cos(x) into FOutputErrorDeriv;
      four tests (forward, gradient check, periodicity, serialization).
- [x] TNNetCos — `y = cos(x)`, derivative `-sin(x)`. Sibling of TNNetSin;
      composed as a phase-shifted Sin in tests (`cos(x) ≈ sin(x + π/2)`).
      Verified absent.
      Landed: TNNetReLUBase descendant caching -sin(x) into FOutputErrorDeriv;
      four tests (forward, gradient check, phase-shift-from-Sin, serialization).
- [ ] TNNetSnake — `y = x + (1/α) * sin(α x)^2`, derivative
      `1 + sin(2 α x)`. α in FFloatSt[0] default 1.0. The Ziyin 2020
      periodic activation that learns periodic functions without an
      explicit SIREN frequency. Already pinned earlier but worth
      re-pinning alongside Sin/Cos — the three form a coherent
      "periodic activation menagerie" mini-batch.
- [ ] TNNetErf — `y = erf(x)`, derivative `(2/√π) exp(-x^2)`. The
      closed-form GELU partner (GELU uses an erf-based form internally;
      exposing the bare erf makes the relation legible and is a fun
      isolated layer). Verified absent. Caveat: FPC `math.erf` exists
      on recent versions; check before claiming portability.

#### Parametric activations I'd enjoy authoring
- [ ] TNNetPReLU — parametric ReLU with a single learnable negative-slope
      scalar (broadcast across the volume). Slope lives as the layer's
      one neuron weight, gradient is `sum_neg(dy * x)` summed across all
      negative-side input cells, input-grad is the LeakyReLU forward
      derivative with the learned slope. Four-test shape including a
      weight-grad central-difference check (the easiest place for a
      bug). Verified absent.
- [ ] TNNetPReLUChannel — per-channel PReLU. Same as above but one
      learnable slope per output depth channel (matches the original
      He 2015 paper). Builds on the TNNetChannelBias channel-iteration
      pattern. Verified absent.
- [ ] TNNetSwishLearnable — TNNetSwish with a single learnable β
      (`y = x * sigmoid(β x)`). Exercises the
      "learnable-scalar-on-an-activation" path that the TNNetLayerScale
      template already covers; mostly a copy-paste with one extra term
      in Backpropagate. Pinned in an earlier batch as
      "Swish-with-learnable-beta"; re-pinning under the canonical name.

#### Sparsity / routing layers I'd enjoy authoring
- [ ] TNNetTopK — keep only the top-K activations per spatial cell along
      the depth axis, zeroing the rest. K in FStruct[0]. Argmax-style
      routing of the gradient (only top-K positions receive incoming
      error). Useful for sparse-attention stubs and a stepping stone
      toward MoE routing. Re-pinning from a previous batch.
- [ ] TNNetStraightThroughEstimator — forward `y = round(x)` (or a
      configurable quantization step in FFloatSt[0]), backward passes
      the gradient through unchanged ("identity STE"). The standard
      trick for training through non-differentiable quantizers; one
      tiny layer that opens the door to integer-quantization
      experiments. Verified absent.
- [ ] TNNetGumbelSoftmax — softmax over depth with added Gumbel noise
      at training time and temperature in FFloatSt[0]. Inference path
      degenerates to plain softmax. Useful for any "differentiable
      categorical sample" experiment (the Concrete distribution).
      Verified absent. Test plan: training-vs-inference branch
      coverage, temperature → 0 limit collapses to argmax, gradient
      flows through the noise.

#### Reduction / shape layers I'd enjoy authoring
- [ ] TNNetCumSum — cumulative sum along a configurable axis
      (FStruct[0] ∈ {0,1,2}). Linear, so backward is the reverse
      cumulative sum along the same axis. Re-pinned for visibility;
      the "CumSum as a learned linear position feature" experiment
      already pinned in an earlier batch needs this layer.
- [ ] TNNetRoll — circular shift along a chosen axis (FStruct[0] axis,
      FStruct[1] offset). Parameter-free, deterministic permutation;
      backward is the inverse roll. Re-pinned for visibility.
- [ ] TNNetL2Normalize — divide by `sqrt(sum(x^2) + eps)`, reduction
      axis via FStruct[0] (0 = full volume, 1 = per-channel over
      spatial, 2 = per-position over depth). Re-pinned for visibility.

#### Composite block builders unblocked by recent landings
- [ ] AddPreNormResidual(NN, Sublayer) — one-liner builder wiring
      `LayerNorm → Sublayer → residual add`. The transformer
      pre-norm pattern. All pieces (TNNetLayerNorm, TNNetSum) are in
      tree. Re-pinned multiple batches; with LayerNorm and Sum both
      gradient-checked, the block itself can ship as a small builder
      test.
- [ ] AddRMSNormResidual(NN, Sublayer) — companion builder using
      RMSNorm in place of LayerNorm (matches LLaMA-style blocks).
      Two-line variant of the PreNorm builder.
- [ ] AddSwiGLUFeedForward(NN, d_model, d_ff) — wires
      `Dense(d_model → 2*d_ff) → SwiGLU → Dense(d_ff → d_model)`.
      With SwiGLU + Dense in tree, this is ~10 lines of builder code
      and one composite-block test. Re-pinned across batches; the
      headline "ready next step" item from the existing tasklist.

#### Tests I'd enjoy adding
- [ ] TNNetReciprocal small-|x| gradient-magnitude sanity check (was
      flagged as a follow-up in the previous lucky-day batch and is
      still open). Mirror TNNetSquare's "Jacobian scales with input"
      relative-tolerance test.
- [ ] Test that `TNNetNegate.Compose(TNNetNegate)` round-trips to
      identity within fp tolerance on a random volume. Cheap
      involution check; doubles as the smoke test for the
      involution-via-double-negate property already pinned in the
      InvolutionDemo example.
- [ ] Numerical-gradient checks for the deconvolution / upsampling
      family (TNNetUpsample, TNNetDeconvolution, TNNetDeMaxPool,
      TNNetDeAvgPool, TNNetDeLocalConnect) — the family is still
      uncovered after the activation / pooling / concat audits.
      Already pinned in earlier batches; re-pinning here because
      this is the next coherent audit unit.

#### Experiments I'm curious about
- [ ] SIREN-flavored 1D function fit: train a 3-layer TNNetSin MLP to
      fit `f(x) = sin(8x) + sin(3x)` on `x ∈ [-π, π]`, compare to a
      ReLU MLP of equal width. One chart, two configs; demonstrates
      the periodic-activation payoff. Unblocked by TNNetSin landing.
- [ ] PReLU vs LeakyReLU vs ReLU on a tiny CIFAR stub: same model
      depth, same epochs, three activations; show whether the learned
      negative slope of PReLU actually wins. Unblocked by TNNetPReLU.
- [ ] TopK bottleneck sparsity sweep (re-pinned): train the same tiny
      autoencoder with K ∈ {1, 2, 4, 8, 16, full} and chart
      reconstruction loss vs sparsity. Unblocked by TNNetTopK.
- [ ] Straight-through quantization demo: train a small classifier
      where one hidden layer's outputs are passed through a
      TNNetStraightThroughEstimator quantizer; compare accuracy
      against the unquantized baseline. Unblocked by
      TNNetStraightThroughEstimator.

#### Examples I'd enjoy writing
- [ ] `examples/SIREN/` — 1D periodic-function fit with TNNetSin
      (unblocked by the Sin landing above). 50-line self-contained
      demo with a tiny matplotlib-free ASCII chart of the fit.
- [ ] `examples/PReLUvsLeakyReLU/` — three-config bake-off described
      in the experiment above. Doubles as the headline use case for
      the PReLU landing.

#### Documentation I'd enjoy writing
- [ ] "Periodic activations" README subsection covering TNNetSin,
      TNNetCos, TNNetSnake (and TNNetGaussianActivation as the
      non-periodic-but-related smooth-bump partner). One paragraph
      on when periodicity helps, one tiny SIREN snippet.
- [ ] "Sparsity & routing" README subsection covering TNNetTopK,
      TNNetHardShrink, TNNetSoftShrink and TNNetThreshold once
      TopK lands — the four together form the "make activations
      sparse" toolkit and deserve a single home in the layer
      reference.
- [ ] `docs/numerical-gradient.md` — short note on how the project's
      central-difference gradient checks work and how to add one
      for a new layer. Already pinned as "how numerical gradient
      testing works in this repo"; re-pinned because the
      `TestNeuralNumerical.pas` patterns are now mature enough that
      the note essentially writes itself.

#### Stretch / ambitious (re-pinned for visibility)
- [ ] TNNetMultiHeadSelfAttention — still the headline blocker for
      Tiny GPT. With SDPA + RoPE + MaskedFill + ALiBi + LayerNorm +
      SwiGLU all in tree, the missing piece is the H-head wrapper
      with Q/K/V projections and an output projection.
- [ ] Tiny GPT char-level example on CPU — pinned across every
      lucky-day batch. Unblocked the moment MHSA lands.
- [ ] Mixture-of-Experts routing layer — best done after MHSA so it
      has a real host architecture; TNNetTopK above is the
      sub-primitive that needs to land first either way.


### Lucky-day batch — 2026-05-15 (seed 186808, post Neg + Sin/Cos + README batch)

Three serial opus agents dispatched on a self-described lucky day.
Landed:

- `a4f6109` — TNNetNeg (`y = -x`): parameter-free TNNetReLUBase
  descendant, derivative `-1` cached in FOutputErrorDeriv. Four tests
  (forward, gradient check, involution-via-double-negate, serialization
  round-trip).
- `270f7a8` — TNNetSin and TNNetCos periodic activations. Both
  parameter-free TNNetReLUBase descendants. TNNetSin caches `cos(x)`,
  TNNetCos caches `-sin(x)` into FOutputErrorDeriv. Eight tests (four
  per layer): forward, gradient check, periodicity (Sin) / phase-shift-
  from-Sin (Cos), serialization round-trip.
- `d80b449` — README activation-table rows for the recently-landed
  transcendental + periodic + negation layers: TNNetSqrt, TNNetExp,
  TNNetLog, TNNetReciprocal, TNNetNeg, TNNetSin, TNNetCos. Closes
  several open README follow-up entries from previous batches.

Test suite: 419 -> 431, all passing in ~1.0s. No bugs surfaced. The
Sqrt/Exp/Log/Reciprocal/Neg/Sin/Cos elementwise family is now complete
and uniformly documented.

#### Natural follow-ups
- [ ] TNNetSnake (`y = x + (1/α) sin(αx)^2`) — completes the periodic
      activation trio (Sin + Cos + Snake). Configurable α via
      FFloatSt[0]; closed-form derivative `1 + sin(2αx)`. Re-pinned
      from seed 141347 batch.
- [ ] TNNetErf — closed-form GELU partner. Verify FPC math.erf exists
      on the supported FPC versions before claiming portability.
- [ ] SIREN-flavored 1D function fit example unblocked by TNNetSin
      landing: train 3-layer Sin MLP on `f(x) = sin(8x) + sin(3x)`
      and compare to a ReLU MLP of equal width. One chart, two configs.
- [ ] Trig identity composition test: `Sin(x)^2 + Cos(x)^2 = 1` to
      within fp tolerance on a random input — pairs naturally with the
      already-landed periodicity and phase-shift tests.

### Lucky-day batch — 2026-05-15 (seed 231116)

A fresh draw on a self-described lucky day. Before pinning, I scanned the
existing tasklist for what is already implemented (Sin, Cos, Neg, Sqrt,
Exp, Log, Reciprocal, RoPE, MaskedFill, SDPA, LayerNorm, RMSNorm,
GroupNorm, DropPath, GLU/GEGLU/SwiGLU, LayerScale, SoftPlus, Gaussian,
SquaredReLU, AddPositionalEmbedding, Concat/Split/Sum gradient audits,
exact SoftMax Jacobian) and what is already pinned but unbuilt (Snake,
Erf, PReLU/PReLUChannel/SwishLearnable, TopK, STE, GumbelSoftmax,
CumSum, Roll, L2Normalize, MHSA, Tiny GPT, MoE, SwiGLU FFN builder,
PreNorm/RMSNormResidual builders, deconv/upsample audit, SIREN demo).
The ideas below try to step into corners those lists do not yet cover.
Every TNNet* name has been grep-verified absent from `neural/neuralnetwork.pas`.

#### Layers I'd enjoy authoring
- [x] TNNetMaxOut — already landed in tree (see earlier batch); the
      seed 231116 re-pin was a stale duplicate.
- [x] TNNetSoftSign — already landed in tree.
- [x] TNNetTanhShrink — already landed in tree.
- [ ] TNNetMishLearnable — TNNetMish with a single learnable α scalar
      (`y = x * tanh(softplus(α x))`). Same template as the pinned
      TNNetSwishLearnable; copy-paste with one extra term in
      Backpropagate. Weight-grad central-difference check is the
      important test.
- [x] TNNetChannelShuffle — already landed in tree.
- [ ] TNNetSpaceToDepth / TNNetDepthToSpace — block reshape pair
      (`FStruct[0] = block_size`). Common stem layer in YOLO/SqueezeNet
      variants and the bridge between conv stems and patch-tokenizers.
      Mutual-inverse round-trip is the headline test.
- [~] TNNetSinusoidalPositionalEmbedding — effectively already in tree:
      TNNetAddPositionalEmbedding calls TVolume.PositionalEncoding which
      is the fixed sin/cos formula. Open follow-up if someone wants a
      separately-named alias and/or a "replace, don't add" variant.

#### Composite block builders I'd enjoy authoring
- [ ] AddPostNormResidual(NN, Sublayer) — the post-norm transformer
      pattern (`Sublayer → residual add → LayerNorm`), companion to the
      already-pinned PreNorm and RMSNormResidual builders. One small
      builder test on a random volume; together the three cover the
      common transformer-block normalization placements.
- [ ] AddGEGLUFeedForward(NN, d_model, d_ff) — the GEGLU twin of the
      pinned SwiGLU FFN builder. Same shape, GELU in place of Swish.
      Both will share an internal helper.

#### Correctness / audit work
- [ ] Numerical-gradient checks for the recurrent-style layers
      (TNNetEmbedding, TNNetTokenAndPositionalEmbedding,
      TNNetMaxChannelPool, TNNetMinChannelPool, TNNetMinPool) — the
      recurrent / token-embedding family is the last big uncovered
      audit unit after deconv/upsample. One test per layer in
      TestNeuralNumerical.pas.
- [ ] Cross-layer composition gradient test: build a 3-layer stack
      (LayerNorm → SwiGLU → Dense) and run a single end-to-end
      central-difference check on the input. Sanity-checks that the
      gradient handoff between layers in a real transformer block is
      still tight — the per-layer tests do not see inter-layer
      bookkeeping bugs.
- [ ] Determinism test for the dropout/DropPath family: with the same
      seed, two forward passes on the same input must produce
      bit-identical outputs. Cheap to add; pins the "seeded reproduction"
      property the gradient checks already rely on.

#### Experiments I'm curious about
- [ ] Maxout vs ReLU width-trade study: at matched parameter count
      (Maxout-k=2 with width N vs ReLU with width 2N), is final loss
      meaningfully different on a tiny MLP? One chart, deterministic.
      Unblocked by the Maxout landing above.
- [ ] Sinusoidal vs learned positional embedding head-to-head on a tiny
      sequence task (the binary-addition idea pinned earlier): which
      converges faster, which generalizes to longer sequences? Pure-CPU,
      seconds per run.
- [ ] "Activation derivative cache invariants" sweep: for every
      activation that caches a derivative into FOutputErrorDeriv
      (ReLU, Swish, Mish, GELU, Sin, Cos, Neg, Sqrt, Exp, Log,
      Reciprocal, SquaredReLU, GLU, GEGLU, SwiGLU, SoftPlus, Gaussian,
      LayerScale), assert that running Compute() twice in a row leaves
      the cache consistent with the second input — guards against the
      Sigmoid/LeakyReLU class of bug the audits already turned up.

#### Examples I'd enjoy writing
- [ ] `examples/MaxoutMnist/` — minimum-viable Maxout demo on a
      tiny-MNIST subset (or a synthetic 2D classification task to
      stay CPU-fast). Doubles as the headline use case for the
      Maxout landing.
- [ ] `examples/SpaceToDepthStem/` — show the SpaceToDepth → Conv
      stem replacing a stride-2 conv on a tiny CIFAR stub; one chart
      of param count and first-epoch loss.
- [ ] `examples/PreNormVsPostNorm/` — toy sequence task with the
      same sublayer wired through PreNorm vs PostNorm builders;
      shows the stability gap that motivates pre-norm in practice.

#### Documentation
- [ ] "Bounded activations" README subsection covering Sigmoid, Tanh,
      HardSigmoid, HardTanh, SoftSign — the bounded counterparts to
      the periodic and gated families already documented. One short
      table with output ranges and derivative notes.
- [ ] Top-of-tasklist "implemented vs pinned" index — the file is now
      large enough that the same idea has been re-pinned several
      times across batches. A small header table (layer name → status
      → batch landed) would cut that down without rewriting history.

#### Stretch / re-pinned for visibility
- [ ] TNNetMultiHeadSelfAttention wrapper around the landed SDPA — the
      headline blocker for Tiny GPT. Re-pinned because every batch
      depends on it and it remains the single highest-leverage open
      layer.
- [ ] Tiny GPT char-level CPU example — unblocked the moment MHSA
      lands; remains the most exciting end-to-end demo on the
      tasklist.

### Lucky-day batch — 2026-05-15 (seed 769362)

Three serial opus agents dispatched on a self-described lucky day.
Before pinning, the orchestrator grep-verified each candidate against
`neural/neuralnetwork.pas` — three of the original seed 231116 pins
(SoftSign, TanhShrink, MaxOut, ChannelShuffle) turned out to already
exist in tree, and the seed 231116 list has been corrected in place.

Landed:

- `961f9cc` — TNNetSnake periodic activation
  (`y = x + (1/α) · sin(αx)²`, derivative `1 + sin(2αx)`). Configurable
  α via FFloatSt[0] with α=0 guard, descends from TNNetReLUBase with
  derivative cached in FOutputErrorDeriv. Three tests (forward,
  gradient check at α=1.5, serialization round-trip at α=2.5).
- `beb40cb` — TNNetStraightThroughEstimator (STE) for
  quantization-aware training. Forward `y = round(x/step)·step` with
  configurable step (default 1.0, step>0 guard); backward is identity
  via the inherited TNNetIdentity backward path. Three tests (forward
  on/off-grid for step ∈ {1, 0.5}, direct backward-equality check
  since the discontinuous forward defeats central-difference,
  serialization round-trip at step=0.25).

Test suite: 431 → 437, all passing.

#### Natural follow-ups
- [ ] TNNetSnake α-sweep micro-experiment — fit `f(x) = sin(8x)` with a
      3-layer Snake MLP across α ∈ {0.5, 1, 2, 4} and chart final loss
      vs α. Tiny, deterministic, unblocked by the Snake landing.
- [ ] Trig identity test for Snake at α=1: derivative
      `1 + sin(2x) = 1 + 2 sin(x) cos(x)`. Pairs naturally with the
      existing Sin²+Cos² identity proposal from the seed 186808 batch.
- [ ] STE quantization-aware MNIST demo — train a small classifier
      where one hidden activation is rounded via TNNetStraightThrough
      Estimator(step=0.25); compare test accuracy against the
      unquantized baseline. Unblocked by the STE landing.
- [ ] TNNetStraightThroughEstimator `step ≤ 0` guard test — confirm the
      constructor's FErrorProc fires for step=0 and step=-1. The runtime
      guard exists but is not currently exercised by a test.
- [ ] Add a "non-differentiable forward" pattern note to the future
      `docs/numerical-gradient.md` — STE is the first layer in tree
      where central-difference is provably wrong, so the direct
      backward-equality test pattern deserves a short callout.

### Lucky-day batch — 2026-05-15 (seed 554877)

Random seed 554877 (Python `random.randint(100000,999999)`) on a
self-described lucky day. Before pinning, every TNNet* candidate name
below was grep-verified absent from `neural/neuralnetwork.pas`, and
the existing tasklist was scanned for prior pins. Skipped from the
initial brainstorm because they already exist in tree: TNNetClamp,
TNNetAbs, TNNetSquare, TNNetPower, TNNetGlobalAvgPool,
TNNetGlobalMaxPool, TNNetCELU, TNNetELU, TNNetThreshold, TNNetHardTanh,
TNNetTanhShrink, TNNetSquaredReLU. Skipped because already pinned in a
previous batch: TNNetErf, TNNetTopK, TNNetGumbelSoftmax, TNNetCumSum,
TNNetRoll, TNNetL2Normalize, TNNetMishLearnable, TNNetSpaceToDepth /
TNNetDepthToSpace.

#### Layers I'd enjoy authoring
- [x] TNNetSinc — `y = sin(x)/x` with the `x→0` limit `y=1` handled
      analytically. Derivative `(cos(x) - sin(x)/x) / x` (with the
      `x→0` limit `0`). Landed (774d04b): parameter-free TNNetReLUBase
      descendant with `Abs(x) < 1e-6` guard; three tests in
      TestNeuralNumerical.pas (forward at known points incl. x=0,
      central-difference gradient check, serialization round-trip).
- [x] TNNetBentIdentity — `y = (sqrt(x²+1) - 1)/2 + x`, derivative
      `x / (2·sqrt(x²+1)) + 1`. Landed (9dc8726): parameter-free
      TNNetReLUBase descendant; three tests in TestNeuralNumerical.pas.
- [x] TNNetLisht — `y = x · tanh(x)`, derivative
      `tanh(x) + x · (1 - tanh(x)²)`. Landed (9685af1): parameter-free
      TNNetReLUBase descendant; tanh evaluation shared between forward
      and derivative; three tests in TestNeuralNumerical.pas.
- [x] TNNetESwish — `y = β · x · sigmoid(β · x)` with configurable β via
      FFloatSt[0] (default 1.25, the paper's recommended value).
      Generalizes Swish; β=1 reduces to Swish. Landed in commit 51daaac:
      parameter-free TNNetReLUBase descendant, derivative cached in
      FOutputErrorDeriv, β>0 guard via FErrorProc, registered in both
      CreateLayer dispatch sites. Three tests in TestNeuralNumerical.pas
      (forward equivalence to Swish at β=1; gradient check at β=1.5;
      β=2.5 serialization round-trip).
- [ ] TNNetHardshrink / TNNetSoftshrink — companions to the already-
      landed TNNetTanhShrink. Hardshrink: zero inside `[-λ, λ]`,
      identity outside (non-smooth, STE-like backward through the
      gate). Softshrink: subtract `λ·sign(x)` outside the band, zero
      inside. λ configurable via FFloatSt[0] (default 0.5). Both are
      standard PyTorch activations and complete the "shrink" family.
- [ ] TNNetLogSoftmax — log of softmax with the standard
      max-subtraction trick for numerical stability. Backward is
      `dL/dy_i - softmax_i · sum_j dL/dy_j` (cheaper Jacobian than
      the exact-softmax version that already landed). Pairs naturally
      with NLL loss; the numerically stable form avoids the
      `log(softmax)` underflow that several training paths quietly
      eat today. Add per-(X,Y) and global variants matching the
      TNNetPointwiseSoftMax / TNNetSoftMax split.
- [ ] TNNetSign — `y = sign(x)` with a straight-through-estimator
      backward (gradient passes unchanged on `|x| ≤ 1`, zero
      otherwise — the "saturated STE" used in BinaryConnect). Builds
      directly on the TNNetStraightThroughEstimator pattern from the
      seed 769362 batch; share the direct-backward-equality test
      style since central-difference is provably wrong here too.
- [ ] TNNetGlobalSumPool — companion to TNNetGlobalAvgPool /
      TNNetGlobalMaxPool. Sum reduction over the (X, Y) plane per
      channel; equals AvgPool · (SizeX·SizeY) so the implementation
      and gradient mirror AvgChannel almost line-for-line. Adds the
      missing third reduction; shows up in attention-pooling and
      set-encoder architectures.
- [ ] TNNetMinMaxNorm — per-sample affine rescale of the volume to
      `[0, 1]` (or to a configurable `[a, b]` via FFloatSt[0..1]).
      Forward `y = (x - min) / (max - min + eps)`; backward needs
      careful handling of the min/max gradient routing
      (subgradient: 1/range to all elements except the argmin/argmax,
      which carry the residual). A focused gradient-check test would
      pin the routing.

#### Composite block builders I'd enjoy authoring
- [ ] AddSqueezeExciteBlock(NN, Reduction) — channel-attention block
      (GlobalAvgPool → FullConnectReLU(C/r) → FullConnect(C) →
      Sigmoid → channel-wise multiply). Needs a fresh
      TNNetChannelMul layer if one is not already in tree (verify
      before authoring). Pairs naturally with the existing
      ConvolutionReLU stems; one short builder test on a tiny
      volume + a Conv-stem-with-SE example.
- [ ] AddPostActivationDenseBlock(NN, GrowthRate, Layers) — a tiny
      DenseNet-style builder around the existing TNNetDeepConcat;
      lets the README transformer-builder family grow a CNN sibling
      without an entire new architecture file.

#### Correctness / audit work
- [ ] Activation derivative-cache invariants sweep (re-pin from seed
      231116 batch, narrowed to a concrete test plan): one
      parameterized helper that, for each activation in the cache-
      using set, runs `Compute(A); Compute(B);` and asserts
      `FOutputErrorDeriv` matches a fresh recomputation against `B`.
      Catches the Sigmoid-class stale-cache bug pattern across the
      whole family in a single ~30-line test.
- [ ] Random-architecture forward/backward fuzz — generate ~50 random
      stacks (input → mix of Dense/Conv/LayerNorm/SwiGLU/GEGLU/
      DropPath/RMSNorm/SDPA → linear head), seed-controlled, and
      assert (a) forward produces no NaN/Inf, (b) backward produces
      no NaN/Inf, (c) parameter gradients are finite. Runs in seconds,
      catches integration bugs that per-layer tests miss.
- [ ] Gradient flow through three normalization layouts: build the
      same tiny block under PreNorm, PostNorm, and RMSNormResidual
      (once those builders land) and assert the input gradient
      magnitude ratio across the three matches the well-known PreNorm
      stability ordering. Doubles as a regression pin on the builder
      semantics.

#### Experiments I'm curious about
- [ ] "Activation zoo" loss-landscape figure — train the same 3-layer
      MLP on the existing Hypotenuse task with each activation in the
      shrink/periodic/gated/transcendental families (~20 configs),
      log final loss + epochs-to-converge, dump a CSV. The data point
      the codebase has been quietly accumulating for several batches
      but never plotted in one place.
- [ ] Sinc vs Sin head-to-head on the SIREN-flavored
      `f(x) = sin(8x) + sin(3x)` fit pinned in the seed 186808 batch —
      Sinc's amplitude decay is a real inductive bias for bounded
      regression targets; the two-line config switch is worth the
      data point.
- [ ] STE bit-width sweep follow-on to the pinned MNIST QAT demo:
      same network, vary `step ∈ {1.0, 0.5, 0.25, 0.125, 0.0625}`
      (effectively 1, 2, 3, 4, 5 bits over a fixed range), plot
      accuracy vs bit-width. Builds directly on the seed 769362
      STE landing.
- [ ] Tanh saturation visualization — feed a wide range of inputs
      through TNNetTanh, plot input vs output and input vs gradient,
      save as a small PNG via the existing image utilities. Cheap
      teaching artifact; the kind of figure the README's activation
      table would benefit from.

#### Examples I'd enjoy writing
- [ ] `examples/SqueezeExciteCifar10/` — minimum-viable SE-augmented
      CIFAR-10 classifier; one chart of with-SE vs without-SE
      validation accuracy at matched epoch count. Headline use case
      for the SE builder above.
- [ ] `examples/ActivationGallery/` — a tiny program that constructs
      a single-layer net per activation in the library, sweeps inputs
      `x ∈ [-5, 5]`, prints `(x, y, dy/dx)` columns to stdout. Doubles
      as documentation and as a smoke test that every activation's
      forward + cached derivative agree with their textbook formula.
- [ ] `examples/QuantizationAwareMnist/` — the STE-MNIST demo pinned
      in the seed 769362 batch, fleshed out: train baseline, train
      with STE on the penultimate activation, compare test accuracy
      and final-weight histograms. Gives the STE landing a
      first-class home in the examples tree.

#### Documentation
- [ ] "Shrink family" README subsection covering TNNetTanhShrink and
      (once landed) TNNetHardshrink / TNNetSoftshrink — short table
      with thresholds, output behavior in/out of band, and the
      sub-gradient note for the non-smooth pair.
- [ ] "Reduction layers" README subsection grouping the three global
      pools (Avg/Max/Sum once it lands) with one usage line each;
      the existing per-layer entries are scattered.
- [ ] Top-of-file lucky-day index — small markdown table of the
      `Lucky-day batch — YYYY-MM-DD (seed N)` headers with their
      landed-layer summary, so the file's growing batch history is
      navigable without scrolling. Re-pin from seed 231116 batch,
      narrowed: just an index, not a rewrite.

#### Stretch / re-pinned for visibility
- [ ] TNNetMultiHeadSelfAttention wrapper around the landed SDPA —
      still the headline blocker for Tiny GPT; re-pinned every batch
      for a reason.
- [ ] Tiny GPT char-level CPU example — unblocked the moment MHSA
      lands.

### Lucky-day batch — 2026-05-15 (seed 758867)

Three serial opus agents dispatched on a self-described lucky day.
Each landed a small parameter-free elementwise activation following
the TNNetSnake / TNNetSquaredReLU / TNNetSin template (TNNetReLUBase
descendant, derivative cached in FOutputErrorDeriv, dispatch entry in
both CreateLayer tables, three tests: forward at known points,
central-difference gradient check, serialization round-trip).

Landed:

- `774d04b` — TNNetSinc (`y = sin(x)/x`, derivative `(cos(x) -
  sin(x)/x)/x`). Analytic `x→0` limits (`y=1`, `dy=0`) guarded by
  `Abs(x) < 1e-6`.
- `9dc8726` — TNNetBentIdentity (`y = (sqrt(x²+1) - 1)/2 + x`,
  derivative `x/(2·sqrt(x²+1)) + 1`). Always-smooth, always-positive-
  slope. The agent rediscovered the float32 cancellation noise that
  bites elementwise gradient checks at large shapes — settled on the
  `(2, 2, 3)` shape Snake also uses.
- `9685af1` — TNNetLisht (`y = x · tanh(x)`, derivative `tanh(x) +
  x · (1 - tanh(x)²)`). `tanh(x)` shared between forward and
  derivative.

Test suite: 437 → 446, all passing. No bugs surfaced.

#### Natural follow-ups (small, doable next batch)
- [ ] TNNetSinc-as-SIREN-activation experiment: rerun the pinned
      `f(x) = sin(8x) + sin(3x)` SIREN fit using TNNetSinc in place of
      TNNetSin and compare convergence + bounded-output behaviour.
      Two-line config switch on the existing experiment idea.
- [ ] TNNetBentIdentity vs SoftPlus head-to-head on the existing
      Hypotenuse task — both are smooth, always-positive-slope; one
      chart of loss-vs-epoch. Tiny, deterministic.
- [ ] LiSHT vs Mish vs Swish bake-off on a tiny MLP — all three are
      smooth, self-gated-ish; matched-width comparison feeds straight
      into the already-pinned activation bake-off CSV experiment.
- [x] README activation-table rows for the three new layers (Sinc,
      BentIdentity, LiSHT) plus TNNetESwish. Landed in commit 2293de5:
      Sinc placed after Cos (periodic), BentIdentity and LiSHT after
      SoftPlus (smooth ReLU alternatives), ESwish after HardSwish.
- [x] Trig-identity micro-test: `Sinc(x) · x = Sin(x)` to within fp
      tolerance on a random non-zero input vector. Landed in commit
      18d8d16 as TestSincTimesXEqualsSin.
- [x] LiSHT symmetry test: `Lisht(-x) = Lisht(x)` to within fp
      tolerance on a random vector. Landed in commit 18d8d16 as
      TestLishtSymmetry. Same commit also adds
      TestSnakeDerivativeTrigIdentity pinning the α=1 derivative
      identity `1 + 2·sin(x)·cos(x)`.

### Lucky-day batch — 2026-05-15 (seed 896318)

Python `random.randint(1, 1000000)` rolled 896318 on a self-described
lucky day. Before pinning, every TNNet* candidate name below was
grep-verified absent from `neural/neuralnetwork.pas` and the existing
tasklist was scanned for prior pins (see the seed 554877, 769362,
758867 batches above). The theme of this batch is "the missing
plumbing": loss functions, regularization-as-layers, schedulers, and
introspection — categories the file has accumulated many ideas about
but only a handful of landings in.

#### Layers I'd enjoy authoring

- [ ] TNNetLogCoshLoss — smooth Huber-like regression loss
      `L = mean(log(cosh(y - ŷ)))` with derivative `tanh(y - ŷ) / N`.
      Numerically stable form: `log(cosh(z)) = |z| + log1p(exp(-2|z|)) -
      log(2)` to avoid overflow for large `|z|`. Pairs naturally with
      the pinned TNNetHuberLoss / TNNetSmoothL1Loss family. Two tests:
      direct backward-equality (closed form is exact) and a
      central-difference check on a small batch.

- [ ] TNNetRReLU — Randomized Leaky ReLU. Slope sampled uniformly from
      `[lower, upper]` per neuron per forward pass during training
      (FFloatSt[0..1], defaults 1/8 and 1/3 per the paper), fixed to
      `(lower + upper) / 2` at inference. Backward uses the slope drawn
      on the matching forward. Same template as TNNetLeakyReLU; the
      training-vs-inference branch is the new wrinkle. Three tests
      including a determinism-given-seed test and a slope-average-at-
      inference test.

- [ ] TNNetThresholdedReLU — `y = x if x > θ else 0`, configurable θ
      via FFloatSt[0] (default 1.0). Subgradient zero at the step;
      derivative cached in FOutputErrorDeriv. Classic but missing.
      Three tests (forward across the threshold, central-difference
      gradient check away from the step, serialization round-trip).

- [ ] TNNetISRU / TNNetISRLU — Inverse-Square-Root (Linear) Unit.
      ISRU: `y = x / sqrt(1 + α·x²)`, ISRLU: piecewise (ISRU for x<0,
      identity for x≥0). Derivatives are clean: ISRU's is
      `(1 + α·x²)^(-3/2)`. Bounded smooth alternative to tanh that
      cheaper to compute than tanh on float32. α via FFloatSt[0].

- [ ] TNNetGaussianNoise — adds N(0, σ²) noise per element during
      training, identity at inference. σ via FFloatSt[0]. Backward
      is pass-through (additive noise has zero expected gradient).
      Test the training/inference toggle and the seeded-determinism
      property. Pairs naturally with the dropout family.

- [ ] TNNetGaussianDropout — multiplicative N(1, σ²) noise during
      training, identity at inference. Approximate equivalence to
      standard dropout for matched `σ² = p/(1-p)`. Same plumbing as
      TNNetGaussianNoise; one extra test that asserts the
      input-equivalence at inference.

- [ ] TNNetSerf — `y = x · erf(softplus(x))`. Self-regularized
      smooth activation; sister to Mish (`x · tanh(softplus(x))`).
      Useful data point for the activation-zoo experiment already
      pinned. Erf is available via the existing erf landing pattern;
      the derivative chains cleanly through softplus.

- [ ] TNNetSoftPlusBeta — generalized SoftPlus with learnable-or-fixed
      β: `y = (1/β) · log(1 + exp(β·x))` with the standard `β·x > 20`
      identity shortcut for numerical stability. β via FFloatSt[0]
      (default 1.0 reduces to the existing TNNetSoftPlus). One test
      that asserts β=1 matches TNNetSoftPlus to fp tolerance.

- [ ] TNNetSoftExponential — `y = (exp(α·x) - 1)/α + α` for α>0,
      identity for α=0, `-log(1 - α·(x + α))/α` for α<0. The single
      activation that smoothly interpolates between log, linear, and
      exponential families as α varies. α via FFloatSt[0]. Documenting
      the three regimes in a single comment block is half the value.

- [ ] TNNetCenteredSoftmax — softmax preceded by per-sample mean
      subtraction. The "centering trick" sometimes used in
      contrastive heads. Cheap to add; pairs with the exact-Jacobian
      softmax that already landed.

#### Loss layers I'd enjoy authoring

- [ ] TNNetCosineEmbeddingLoss — `1 - cos(y, ŷ)` for the positive
      pair and `max(0, cos(y, ŷ) - margin)` for the negative pair.
      Standard contrastive-head loss; pairs naturally with the
      pinned TNNetCosineSimilarityAttention.

- [ ] TNNetKLDivergence — `sum(p · log(p/q))` with numerical-stability
      clamps on `q`. Backward is `(log(p/q) + 1) · dp/dq` style; the
      direct closed form avoids any softmax-Jacobian pitfalls.
      Useful for distillation experiments.

- [ ] TNNetDiceLoss — `1 - 2·sum(p·q + ε) / (sum(p²) + sum(q²) + ε)`,
      the IoU-flavored segmentation loss. Cheap addition; one
      gradient-check test on a small (4, 4, 1) volume.

#### Training infrastructure (the "missing plumbing")

- [ ] TNeuralLRScheduler — small interface (`function NextLR(Epoch,
      Step: integer): TNeuralFloat;`) with three concrete
      implementations: TStepLR (multiply by γ every N epochs),
      TCosineAnnealingLR (`η_min + (η_max - η_min) · 0.5 ·
      (1 + cos(π · t / T))`), TWarmupCosineLR (linear warmup for
      W epochs then cosine to η_min). Hooks into the existing
      neuralfit loop in one place. Unblocks the half-dozen
      scheduler ideas already pinned without committing to a
      framework rewrite.

- [ ] StochasticWeightAveraging helper — a TNNet wrapper that
      maintains a running average of the live network's weights
      every N steps after epoch W. Call `nn.UpdateBatchNormStats(...)`
      once before evaluation. Two-screen implementation; consistent
      ~0.5–1% test-accuracy bump on the existing CIFAR examples
      would be a great data point.

- [ ] PolyLR scheduler — `η · (1 - t/T)^p`, the standard
      segmentation-paper schedule. Folds in once the
      TNeuralLRScheduler interface above lands.

- [ ] GradientClipping options on TNeuralFit — both `clip_norm`
      (global) and `clip_value` (element-wise) variants. Currently
      every example that needs it open-codes the same loop. A
      single property toggle would clean up several of them.

#### Introspection / debugging tools

- [ ] TNNet.PrintSummary — Keras-style layer-by-layer table
      (`Layer name | Output shape | Param count | Trainable`) printed
      to stdout. Computes total/trainable param counts. The single
      piece of introspection users ask for most often when they
      hit a layer-shape mismatch.

- [ ] TNNet.CountFLOPsPerLayer — forward-pass FLOP estimate per
      layer (Conv: `2·Cin·Cout·K²·H·W`; Dense: `2·In·Out`; element-
      wise: `H·W·C`; etc). Printed in the same table as PrintSummary
      via an optional flag. Useful for the bench-suite idea already
      pinned and for the model-zoo loader.

- [ ] WeightHistogramDump — given a TNNet, write one CSV per
      trainable layer with 64-bin histograms of the weight values.
      Cheap implementation, immediately useful for the QAT/STE
      experiments and for the "did my init explode" debugging
      pattern.

- [ ] DeadNeuronReport — for each ReLU-family activation in a
      trained net, report the fraction of neurons whose activation
      is zero across a validation batch. One short utility, one
      data point that informs Leaky/ELU/RReLU experiments.

- [ ] TNNet.ToGraphvizDot — emit a `.dot` file describing the layer
      DAG (concat / branch / skip edges included). Static images
      plus the README-level documentation benefit. The TNNet
      already tracks predecessors; the export is mechanical.

#### Examples I'd enjoy writing

- [ ] `examples/ModelSummaryDemo/` — three networks (tiny MLP, small
      CNN, a transformer-flavored stack once MHSA lands) printed
      via PrintSummary. Doubles as a smoke test for the summary
      output format.

- [ ] `examples/SchedulerCompare/` — same network trained four times
      with constant LR, StepLR, CosineLR, WarmupCosineLR; one chart
      of validation accuracy vs epoch. Lands the scheduler suite's
      headline use case.

- [ ] `examples/SWADemo/` — CIFAR-10 baseline vs the same network
      with SWA enabled from epoch 75% on. One chart of final
      test accuracy across three seeds.

- [ ] `examples/LossLandscapeCompare/` — MSE vs LogCosh vs Huber
      vs MAE on the existing Hypotenuse task with a fixed
      handful of outliers injected into training. Highlights why
      smooth-Huber-flavored losses earn their place in the box.

#### Correctness / audit work

- [ ] Loss-layer gradient-check helper — a parameterized helper that
      takes `(LossLayer, BatchSize, Shape)` and runs a single
      central-difference check on the input. Six lines per loss;
      catches all three of the new loss-layer landings at once.

- [ ] Scheduler unit tests — given seed and schedule parameters,
      `NextLR(epoch, step)` must produce a deterministic, finite,
      monotonically-correct sequence. One test per concrete schedule.

- [ ] PrintSummary smoke test — build five canonical networks (the
      ones the README already shows code for), capture the summary
      output, assert the row count and the total-parameter line.
      Pin the format so future changes are deliberate.

#### Experiments I'm curious about

- [ ] RReLU vs LeakyReLU vs PReLU on a tiny CIFAR stub: which one
      wins at matched param count? One chart. The randomized variant
      is the only one of the three not yet measured in the
      codebase's experimental record.

- [ ] LogCosh vs MSE on noisy Hypotenuse — inject 5% label noise,
      train both, compare final test loss. The kind of one-figure
      result that earns the loss its place in the docs.

- [ ] Cosine-LR vs constant-LR on the existing SimpleImageClassifier
      example, three seeds each. The headline benchmark for
      whether the scheduler infrastructure earns its keep.

- [ ] SWA effect-size sweep: vary the SWA start-epoch fraction
      ∈ {0.5, 0.6, 0.7, 0.8, 0.9} and chart final test accuracy.
      Cheap experiment, useful default-tuning data point.

- [ ] "Activation cost" microbenchmark — for each activation in the
      library, measure forward+backward ns/op on a fixed 64x64x32
      volume. Pairs naturally with the Volume microbench already
      pinned; lets users pick activations by speed when accuracy
      is tied.

#### Documentation

- [ ] "Loss functions" README subsection grouping MSE, MAE, CE,
      and (once landed) Huber/SmoothL1/LogCosh/Dice/KL/Focal/
      LabelSmoothingCE/CosineEmbedding into a single short table
      with use-case notes. The README currently mentions losses
      only inline in examples.

- [ ] "Learning-rate schedulers" README subsection — one paragraph
      per schedule with a snippet showing how to wire it into
      TNeuralImageFit. Folds in once the scheduler interface lands.

- [ ] "Introspection" README subsection — `CountLayers`,
      `CountNeurons`, `CountWeights` are already shown in the quick-
      start; group them with the new PrintSummary / FLOPs /
      WeightHistogram / DeadNeuronReport utilities into a single
      subsection so users discover them.

#### Stretch / re-pinned for visibility

- [ ] TNNetMultiHeadSelfAttention wrapper around the landed SDPA —
      still the headline blocker for Tiny GPT. Re-pinned every
      batch for a reason.

- [ ] Tiny GPT char-level CPU example — unblocked the moment MHSA
      lands; the headline end-to-end demo on this entire tasklist.

### Lucky-day batch — 2026-05-15 (seed 735373, post ESwish + identity-tests + README batch)

Three serial opus agents dispatched on a self-described lucky day.
Landed:

- `51daaac` — TNNetESwish (`y = β·x·sigmoid(β·x)`), parameter-free
  TNNetReLUBase descendant. β in FFloatSt[0], default 1.25, β>0 guard.
  Three tests (Swish-equivalence at β=1, gradient check at β=1.5,
  serialization round-trip at β=2.5).
- `18d8d16` — Identity micro-tests for the just-landed periodic /
  smooth activations: `Sinc(x)·x == sin(x)`, `Lisht(-x) == Lisht(x)`
  (evenness), and Snake α=1 derivative trig identity
  `1 + 2·sin(x)·cos(x)` against FOutputErrorDeriv.
- `2293de5` — README activation-table rows for TNNetSinc /
  TNNetBentIdentity / TNNetLisht / TNNetESwish, placed alongside
  their natural siblings.

Test suite: 446 → 452, all passing.

#### Natural follow-ups
- [ ] ESwish vs Swish bake-off on a tiny task: same MLP, β∈{1.0, 1.25,
      1.5, 2.0}, chart final loss. The β=1.25 default is folklore; pin
      whether it actually wins at toy scale.
- [ ] TNNetESwish saturation test at ±extreme inputs (mirroring the
      HardTanh/SoftCapping pattern). Confirms backward doesn't NaN
      when sigmoid(βx) underflows to 0 or saturates to 1.
- [ ] LiSHT/BentIdentity gradient-magnitude sanity at large |x| —
      both grow unboundedly (LiSHT ≈ |x|, BentIdentity ≈ 1.5·x for
      large |x|), so finite-difference eps must scale with input
      magnitude. A focused relative-tolerance test would pin the
      convention for this family.

### Lucky-day batch — 2026-05-15 (seed 528621)

Self-picked wishlist of features I'd enjoy authoring. Verified against
the current layer registry in neural/neuralnetwork.pas and the pinned
TODOs already present in this tasklist — none of the items below
duplicate something already landed or already in-flight.

#### Activations I'd enjoy adding
- [x] TNNetTanhExp — `y = x · tanh(exp(x))` from the TanhExp paper.
      Landed: TNNetReLUBase descendant with FP32-safe clamping at
      |x|>20, gradient + serialization round-trip tests in
      TestNeuralNumerical.pas. Commit fbf1031.
- [x] TNNetSmish — `y = x · tanh(ln(1 + sigmoid(x)))`. Landed: uses
      two-branch sigmoid to avoid overflow, parameter-free TNNetMish-
      style implementation, gradient + serialization round-trip tests.
      Commit 4c627c1.
- [ ] Saturation-safety tests for TNNetTanhExp / TNNetSmish at
      ±extreme inputs, mirroring the HardTanh/SoftCapping pattern.
      Confirms forward stays finite and backward doesn't NaN when
      exp(x) overflows or sigmoid(x) underflows to 0.

#### Layers I'd enjoy adding
- [ ] TNNetSqueeze and TNNetExpandDims — numpy-style single-axis
      shape helpers. Less error-prone than open-coding TNNetReshape
      for the sequence-model code that the pinned Tiny GPT and
      transformer-encoder work will need. Pure shape ops, identity
      forward, identity backward — easiest possible gradient check.
- [ ] TNNetCosineSimilarity — pairwise output layer that splits its
      input in half along the depth axis and produces per-sample
      `cos(x, y) = (x·y) / (||x||·||y||)`. The natural partner for
      the already-pinned TNNetCosineEmbeddingLoss; without it, every
      contrastive head has to bolt on its own normalization. Full
      gradient through the two norms (one numerical-gradient test
      input-side, one against a fixed reference value).
- [x] TNNetTriangularCausalMask(SeqLen) — convenience layer that
      builds its own upper-triangular `-1e9` mask sized to the
      previous-layer output at SetPrevLayer time and adds it during
      forward. Identity backprop (passthrough). Forward, identity-
      gradient, serialization round-trip and SDPA-pairing tests in
      TestNeuralNumerical.pas. Commit 22b2b13.

#### Inference & introspection
- [ ] Inference sampling utilities — new `neuralsampling` unit (or
      additions to neuralvolume) with `Argmax`, `TopKSample`,
      `TopPSample`, `TemperatureScale` operating on a TNNetVolume's
      probability row. Seeded RNG, exhaustively unit-tested. The
      missing inference primitives for the pinned Tiny GPT example;
      landable independently with their own tests today.
- [ ] TNNetEMAWrapper / SetEmaShadow — exponential moving average
      of network weights for inference, sibling to the pinned SWA
      helper. Two-screen implementation (one EMA buffer per
      trainable layer, update each step with `θ_ema ← α·θ_ema +
      (1−α)·θ`, swap in for evaluation). Worth a separate landing
      from SWA so they can be compared head-to-head.

#### Correctness / audit work
- [ ] Activation golden-values regression test — for every
      registered activation layer, evaluate forward (and backward
      where the layer overrides Backpropagate) on a pinned input
      vector at fixed seed and assert against pinned outputs within
      1e-5. Catches silent numerical regressions across
      compilers / AVX paths / future optimization passes. One
      ~20-line table-driven test in TestNeuralNumerical.pas.
- [ ] Per-activation derivative-sign sanity test — for each
      strictly monotone activation (ReLU family, Swish, GELU, Mish,
      ELU, SELU, CELU, Sigmoid, Tanh, SoftPlus, …), assert
      FOutputErrorDeriv has the expected sign on a grid of inputs.
      Cheap audit that "monotone" claims in the README activation
      table survive future edits.
- [ ] TNNet.SaveToString round-trip fuzz — extend the existing
      round-trip sweep to randomly-shaped tiny networks built from
      the full layer-class registry, asserting bit-for-bit weight
      equality after Save → Load → Save. Catches serialization
      gaps in newly-added layers automatically — the project's
      biggest silent-bug hotspot now that the layer count is
      approaching 150.

#### Examples I'd enjoy writing
- [ ] `examples/TinyTransformerFFN/` — train a SwiGLU + RMSNorm +
      residual FFN block on a toy denoising or autoregressive-bit
      task. Already unblocked today (no MHSA needed); demonstrates
      four recently-landed layers (TNNetSwiGLU, TNNetRMSNorm,
      TNNetLayerScale, TNNetDropPath) working together as the
      transformer-FFN half-block that the pinned MHSA task will
      eventually plug into.
- [ ] `examples/ActivationPlayground/` — a tiny program that prints
      one CSV row per activation: name, forward ns/op, backward
      ns/op, output range on `[-8, 8]`, derivative range, and a
      cheap "is monotone?" check. Doubles as the data source for
      the already-pinned activation-cost microbench and as a
      smoke test that every activation in the registry can at
      least round-trip a forward pass.

#### Documentation
- [ ] Group the gated-activation / normalization / attention
      transformer-building-block layers (TNNetGEGLU, TNNetSwiGLU,
      TNNetGLU, TNNetLayerNorm, TNNetRMSNorm, TNNetGroupNorm,
      TNNetRotaryEmbedding, TNNetMaskedFill,
      TNNetScaledDotProductAttention, TNNetLayerScale, TNNetDropPath)
      into a single "Building a transformer block" subsection of the
      README — currently they're scattered across the layer reference
      and a user wanting to assemble a transformer has to hunt for
      them. One short paragraph + one assembled code snippet.

### Lucky-day batch — 2026-05-15 (seed 491990)

Python `random.randint(1, 1000000)` rolled 491990 on a self-described
lucky day. Before pinning, every candidate `TNNet*` name below was
grep-verified absent from `neural/neuralnetwork.pas` and the existing
tasklist was scanned for prior pins (seeds 735373, 528621, 896318,
769362, 758867, 554877, 781449 batches above). Theme of this batch is
"contrastive heads, MaxOut-family neighbours, and the
register/serialize safety net" — areas the file has gestured at
without yet pinning concrete chunks.

#### Layers I'd enjoy authoring

- [x] TNNetL2Normalize — per-sample L2 normalize across the depth axis
      with epsilon for numerical stability. Landed (commit 9c11107):
      descends from TNNetIdentity, eps in FFloatSt[0] (default 1e-8),
      full `(I - ŷŷᵀ) / n` Jacobian backward with per-(x,y) `1/n`
      cached in FInvNorms. Three tests (unit-norm forward, gradient
      check, serialization round-trip). Note: existing
      TNNetPointwiseNorm has an approximate scalar-only backward;
      TNNetL2Normalize is the exact-Jacobian replacement to prefer.

- [x] TNNetPenalizedTanh — `y = tanh(x) if x>0 else 0.25·tanh(x)`.
      Landed (commit 362af6d): TNNetReLUBase descendant, derivative
      cached in FOutputErrorDeriv so the inherited Backpropagate just
      works. Three tests (asymmetry across zero, gradient check,
      serialization round-trip).

- [x] TNNetSoftMin — `softmax(-x)`, the natural sibling of TNNetSoftMax.
      Landed (commit 3e99e06): descends from TNNetPointwiseSoftMax,
      Compute negates a working copy then calls SoftMax; Backpropagate
      uses the same O(N) softmax-Jacobian form but subtracts (sign
      flip from d(-x)/dx = -1). Three tests (sums-to-one,
      softmax(-x) equivalence, gradient check).

- [ ] TNNetGumbelSoftmax — straight-through estimator with temperature
      `τ` (FFloatSt[0]). Forward draws `g_i = -log(-log(U_i))` and
      computes `softmax((x + g)/τ)`; backward bypasses the sampling
      and uses the softmax Jacobian. Inference uses the configurable
      temperature or argmax (FFloatSt[1] flag). Three tests
      (deterministic-given-seed, low-τ approaches one-hot, gradient
      flow check at τ=1).

- [ ] TNNetMaxOut2 — the two-piece special case of the existing
      TNNetMaxOut, with a tighter API (no group-count parameter; just
      "split depth in half, take element-wise max"). Friendlier default
      for users new to the layer; backward routes the gradient to
      whichever half won per element. One forward, one gradient check.

- [ ] TNNetMaxBlurPool — "Making Convolutional Networks Shift-Invariant
      Again" max-pool followed by a fixed (non-trainable) anti-aliasing
      blur. Cheap drop-in over TNNetMaxPool that measurably improves
      shift-equivariance on CIFAR-style nets. Forward composes the two
      ops; backward composes their adjoints. One gradient-check test
      and one shift-equivariance smoke test on a synthetic input.

- [ ] TNNetSqueezeExcitation — channel-attention block: GlobalAvgPool →
      Dense(C/r) → ReLU → Dense(C) → Sigmoid → channel-wise multiply.
      The single most-cited "free 0.5–1% accuracy" architectural trick
      missing from the layer set. Composite layer that owns its inner
      dense weights so users don't have to wire five layers manually.

- [ ] TNNetUnitNormConstraint — projection layer that L2-normalizes
      the *weights* of the previous trainable layer after each step
      (sister of TNNetL2Normalize on activations). Useful for matrix-
      factorization heads and stabilizing GAN discriminators.

#### Loss layers I'd enjoy authoring

- [ ] TNNetFocalLoss — `-(1-p)^γ · log(p)` with α/γ via FFloatSt[0..1]
      (defaults α=0.25, γ=2.0 per the RetinaNet paper). The single
      most-requested loss for class-imbalanced detection/segmentation
      heads. Closed-form backward is `(1-p)^γ · (γ·p·log(p) + p - 1)`,
      so the gradient check is exact.

- [ ] TNNetTripletLoss — `max(0, ||a-p||² - ||a-n||² + margin)`. Input
      depth is split into three equal anchor/positive/negative chunks
      along depth (or sample axis with a flag). Pairs with the pinned
      TNNetL2Normalize and the cosine-head family. Two tests
      (closed-form margin satisfaction, gradient check on small batch).

- [ ] TNNetTverskyLoss — generalized Dice with separate FP and FN
      weights `α, β` (FFloatSt[0..1]). When `α=β=0.5` reduces to the
      pinned TNNetDiceLoss. The standard segmentation-imbalance loss;
      one short closed-form backward; one central-difference test.

- [ ] TNNetWingLoss — facial-landmark-style regression loss with a
      log-shaped "wing" near zero and a linear tail, parameters
      `(w, ε)` from the paper. Underused but well-suited to keypoint
      regression — the only loss in the family that is finite-tail
      *and* sub-quadratic near zero.

#### Inference / training infrastructure

- [ ] Lookahead optimizer wrapper — every `k` inner SGD steps, set the
      "slow" weights `φ ← φ + α·(θ - φ)` and rewind the fast weights
      to `φ`. Two-screen wrapper around the existing optimizer; one
      test that confirms determinism given seed and one that confirms
      the slow-weight average converges. Sister to the SWA helper
      pinned in the seed-896318 batch.

- [ ] NaN/Inf guard hook for TNeuralFit — optional "abort training and
      print the offending layer" check after each forward+backward
      pass. Currently a NaN silently propagates through the loss and
      contaminates weights; the user only finds out when validation
      acc collapses. One short utility, immediately useful for the
      QAT/GAN/transformer experiments already pinned.

- [ ] Layerwise learning-rate multipliers — per-layer `LRMult` field
      (default 1.0) that the optimizer respects. Unlocks discriminative
      fine-tuning ("freeze backbone, train head at 10×") without the
      current pattern of zeroing weights or doing two-pass training.

- [ ] Deterministic-seed reproducibility test — train one of the
      existing examples twice with the same seed, assert bit-for-bit
      weight equality after N steps. The CI-level safety net that
      catches future "hidden RNG draw" bugs in any layer or optimizer
      change. Sister to the SaveToString fuzz pinned above.

#### Correctness / audit work

- [ ] Layer-registry round-trip audit — for every concrete `TNNet*` in
      the LoadFromString/CreateLayer dispatch table, instantiate with
      defaults, save, load, save again, assert bit-for-bit string
      equality. The cheapest possible safety net against the silent
      "added a layer but forgot to register it in CreateLayer" bug —
      the layer count is past 200, so this is the highest-leverage
      single test on the file.

- [ ] Shape-inference smoke test — instantiate every concrete layer at
      a small canonical input shape, assert the declared output shape
      matches the actually-produced output shape. The second-cheapest
      safety net; catches "forgot to call FOutput.ReSize" regressions.

- [ ] Backward-pass sign-correlation test — for every layer that
      overrides Backpropagate, perturb input by `+ε` and `-ε`, assert
      that the gradient direction agrees with the loss-difference
      direction more than 90% of the time across a small grid. A
      coarse-but-fast complement to the per-layer gradient checks;
      catches sign-flip bugs in seconds rather than minutes.

#### Examples I'd enjoy writing

- [ ] `examples/EmbeddingVisualization/` — train a tiny contrastive
      head on a 4-class toy 2D dataset, dump the learned embeddings
      to CSV, and include a one-screen README showing how to plot
      them. End-to-end demo of the L2Normalize + Cosine* +
      TripletLoss layers above.

- [ ] `examples/AutoencoderReconstructionGrid/` — train the existing
      VisualAutoencoder on MNIST and emit a single PNG showing
      `[input | reconstruction]` rows for 16 random samples each
      epoch. The "is it actually learning?" sanity check that every
      autoencoder tutorial has but this repo doesn't.

- [ ] `examples/SqueezeExciteCifar/` — a SimpleImageClassifier-sized
      net with and without TNNetSqueezeExcitation, three seeds each,
      one chart of test accuracy vs epoch. The headline benchmark for
      whether the SE block above earns its keep on this codebase's
      reference task.

- [ ] `examples/SeededReproducibility/` — runs one of the smallest
      examples twice with the same seed and prints PASS/FAIL on
      bit-for-bit weight equality. Doubles as the in-repo home of the
      reproducibility test pinned above.

#### Documentation

- [ ] "Embedding heads" README subsection — group TNNetL2Normalize,
      TNNetCosineSimilarity, TNNetCosineEmbeddingLoss, TNNetTripletLoss
      into one short subsection with a one-paragraph "how to build a
      contrastive head" recipe. Currently embedding-style heads are
      not mentioned anywhere in the README.

- [ ] "Layer index by family" README appendix — alphabetical-within-
      family table (Convolution / Pooling / Activation / Normalization
      / Attention / Loss / Shape / Regularization). The layer count is
      past 200 and discoverability has degraded; a single appendix
      table doubles as the contributor-side checklist for new layers.

### Lucky-day batch — 2026-05-15 (seed 422331)

Three serial opus agents dispatched on a self-described lucky day.
Theme: "small, self-contained layers from the seed-491990 wishlist
that are unblocked today." Landed:

- `3e99e06` — TNNetSoftMin (softmax(-x)) as TNNetPointwiseSoftMax
  descendant. Compute negates a working copy then SoftMax; Backpropagate
  uses the O(N) softmax-Jacobian form and subtracts the result from
  the previous-layer gradient (sign flip from d(-x)/dx = -1). Three
  tests (sums-to-one, equivalence to softmax(-x), central-difference
  gradient check).
- `362af6d` — TNNetPenalizedTanh, asymmetric tanh from "Revise
  Saturated Activations for Deep Networks". TNNetReLUBase descendant,
  derivative cached in FOutputErrorDeriv so the inherited Backpropagate
  multiplies through. Three tests (asymmetry exploiting
  `y(-x) = -0.25·y(x)`, gradient check, serialization round-trip).
- `9c11107` — TNNetL2Normalize, per-sample L2 normalization over the
  depth axis. TNNetIdentity descendant, epsilon in FFloatSt[0]
  (default 1e-8), full `(I - ŷŷᵀ) / n` Jacobian backward with per-(x,y)
  `1/n` cached in FInvNorms. Three tests (unit-norm forward, gradient
  check, serialization round-trip including non-default eps).

Test suite: 460 → 469 (+9), all passing.

#### Natural follow-ups
- [ ] TNNetCosineSimilarity now has a clean partner in TNNetL2Normalize
      — building cosine similarity on top of normalized inputs is the
      obvious next-step embedding-head layer.
- [ ] Deprecation note for TNNetPointwiseNorm: its backward is the
      scalar-only `Mul(1/n)` approximation; TNNetL2Normalize implements
      the exact Jacobian. Either redirect Backpropagate or add a
      "prefer TNNetL2Normalize" comment to the older layer.
- [ ] TNNetSoftMin saturation test on extreme inputs (mirroring the
      SoftCapping pattern) — assert forward outputs stay in [0,1] and
      sum to 1, and backward doesn't NaN, when the most-negative entry
      dominates.
- [ ] PenalizedTanh activation bake-off slot — pair it with the
      existing activation bake-off entry; the 0.25 negative slope is
      a hyperparameter worth surveying (0.1 / 0.25 / 0.5).
- [ ] README activation-table rows for TNNetSoftMin, TNNetPenalizedTanh,
      and TNNetL2Normalize. Follows the pattern just used for
      TNNetSinc / TNNetBentIdentity / TNNetLisht / TNNetESwish.

### Lucky-day batch — 2026-05-15 (seed 755169)

Python `random.randint(1, 1000000)` rolled 755169. Every candidate
`TNNet*` name below was grep-verified absent from
`neural/neuralnetwork.pas`. Theme: "embedding heads + metric-learning
losses + a few audit nets the file keeps gesturing at." None of these
duplicate finished work (cross-checked against `examples/`,
`tests/Test*.pas`, and prior lucky-day batches 422331, 491990, 623999,
735373, 528621, 896318, 769362, 758867, 554877, 781449).

#### Layers I'd enjoy authoring

- [ ] TNNetCosineSimilarity — pairwise cosine similarity between two
      equal-depth halves of the input volume (split along depth).
      Sister to the freshly-landed TNNetL2Normalize: compose
      L2Normalize → CosineSimilarity for a numerically-stable cosine
      head. Forward: per-(x,y) dot(a,b)/(|a||b|+ε). Backward: the
      symmetric Jacobian `(b - cos·â)/|a|` for the a-half (mirrored
      for b). Three tests (closed-form vs hand-computed, gradient
      check, equivalence to dot product after L2Normalize).

- [ ] TNNetCosineEmbeddingLoss — y·(1-cos) + (1-y)·max(0, cos-margin)²
      loss layer, sample-label flag in FIntArr[0]. Pairs with
      TNNetCosineSimilarity above. One gradient check + one
      sign-correctness test (positive pairs pull closer, negatives
      push past margin).

- [ ] TNNetArcFace — additive angular-margin softmax: replaces the
      final FullConnect+SoftMax with the L2-normalized weight ·
      L2-normalized feature dot product, with margin `m` added to the
      target class's angle before scaling by `s`. The single
      most-cited face/embedding-recognition head not yet represented.
      Constructor takes `(NumClasses, s, m)`. Gradient check at small
      class count.

- [ ] TNNetCenterLoss — joint softmax + `λ·||x - c_y||²` with EMA-
      updated class centers `c_y`. Centers stored in the layer's
      weight tensor and updated in Backpropagate with the
      configurable momentum. Two tests: gradient w.r.t. features and
      a "centers converge to class means" smoke test on a toy mixture.

- [ ] TNNetSqueezeExcitation — channel-attention block: GlobalAvgPool
      → Dense(C/r) → ReLU → Dense(C) → Sigmoid → channel-wise
      multiply. Composite layer owning its inner dense weights so
      users don't wire five layers manually. (Listed in the
      seed-491990 batch under loss-layer follow-ups; pinning here as
      a standalone task because the SE block is the highest-leverage
      single missing architectural primitive.)

- [ ] TNNetFocalLoss — `-α·(1-p_t)^γ·log(p_t)` classification loss
      for class-imbalanced detection/segmentation. α/γ in
      FFloatSt[0..1] (defaults 0.25 / 2.0 per the RetinaNet paper).
      Closed-form backward is exact so the gradient check is clean.
      (Cross-referenced from the seed-491990 batch.)

- [ ] TNNetHuberLoss / TNNetSmoothL1Loss — robust regression loss:
      quadratic near zero (`|err|<δ`), linear in the tail. Single
      δ parameter in FFloatSt[0]. Forward + closed-form backward +
      gradient check + "matches MSE for small δ, matches L1 for
      large δ" boundary test.

- [ ] TNNetGumbelSoftmax — straight-through estimator with
      temperature `τ`. Forward draws `g_i = -log(-log(U_i))` and
      returns `softmax((x+g)/τ)`; backward bypasses the sampling and
      uses the softmax Jacobian. Inference mode (FIntArr[0]) selects
      between the temperature-softmax path and argmax. (Already on
      the seed-491990 wishlist; pinning here so the next pass has a
      ready home.)

- [ ] TNNetMaxBlurPool — anti-aliased max-pool from "Making
      Convolutional Networks Shift-Invariant Again": max-pool
      followed by a fixed (non-trainable) blur filter. Composes the
      forward of TNNetMaxPool + a fixed-kernel convolution; backward
      composes the adjoints. Shift-equivariance smoke test on a
      synthetic input.

- [ ] TNNetSReLU — S-shaped ReLU with four learnable knee
      parameters per channel `(t_l, a_l, t_r, a_r)`. Piecewise-linear
      so gradient is exact and trivially checkable. Slots in next to
      TNNetVeryLeakyReLU / TNNetReLUL in the family.

- [ ] TNNetAconC — "Activate Or Not": `(p1-p2)·x·sigmoid(β(p1-p2)x)
      + p2·x` with channel-wise learnable `(p1, p2, β)`. Generalizes
      Swish and is a drop-in for any ReLU. Adds a learnable-weight
      gradient-check path that the activation family otherwise lacks.

- [ ] TNNetPhish — `x · tanh(softplus(x))` (the Mish sibling that
      uses softplus inside tanh instead of outside). Pure activation,
      closed-form derivative, fits into the activation bake-off slot
      already pinned.

- [ ] TNNetErf — Gaussian error function as a standalone activation,
      shared by exact-GELU and CDF-based heads. The cleanest way to
      enable an "exact GELU" path without breaking the existing
      approximate TNNetGELU.

#### Examples I'd enjoy writing

- [ ] `examples/TinyGPT/` — char-level transformer end-to-end demo on
      a short text snippet (Tiny Shakespeare or repeated arithmetic
      patterns) once TNNetMultiHeadSelfAttention lands. The
      highest-value example missing from the repo and the natural
      capstone for the transformer-building-blocks line of work.

- [ ] `examples/BinaryAdder/` — tiny "learns to add two binary
      numbers" sequence example. Self-contained, fast, deterministic
      right answer; an instructive bridge between the XOR example
      and the transformer demos. (Already gestured at in earlier
      lucky-day batches; pinning here with the directory name.)

- [ ] `examples/DeadReLUDiagnostic/` — train a small ReLU net on
      MNIST and print the per-epoch fraction of units that never
      fire; repeat with LeakyReLU/GELU/Swish for comparison. A fun
      teaching example tied to the activation bake-off entries.

- [ ] `examples/AnomalyAutoencoder/` — train an autoencoder on MNIST
      digit "0", evaluate reconstruction error on all 10 digits,
      print AUROC. The smallest possible "on-device anomaly
      detection" demo — a feature the top of the file has asked for.

- [ ] `examples/SpokenDigitKWS/` — 1D-conv keyword-spotting on the
      free-spoken-digit dataset (FSDD): MFCCs → 1D conv stack →
      classification. Closes the "audio: 1D-conv keyword spotting"
      bullet at the top of the file.

- [ ] `examples/TimeSeriesForecast/` — a one-screen forecasting demo
      on a synthetic seasonal+trend series (no external dataset
      needed) with a 1D-conv or tiny attention model. Closes the
      time-series bullet at the top of the file.

- [ ] `examples/GradientFlowVisualizer/` — train a deep MLP with and
      without LayerNorm/RMSNorm and print per-layer gradient-norm
      tables across steps. Concrete teaching example of the
      vanishing-gradient story; reuses landed norm layers.

- [ ] `examples/ActivationBakeoff/` — pure-CPU bake-off of ReLU /
      GELU / Swish / Mish / SELU / SoftPlus / SquaredReLU /
      PenalizedTanh / SoftMin on a fixed small MLP and dataset; one
      table of final loss and epochs-to-converge. Closes the
      activation-bake-off bullet that has lingered since multiple
      earlier batches.

- [ ] `examples/NormalizationBakeoff/` — same idea, comparing no-norm
      / BatchNorm / LayerNorm / RMSNorm / GroupNorm / InstanceNorm on
      a small net with a convergence-speed chart. The natural sibling
      to the activation bake-off; both depend only on landed layers.

- [ ] `examples/OptimizerBakeoff/` — SGD / SGD+momentum / Adam /
      RMSProp on a fixed toy dataset with a loss-vs-epoch table.
      Pure-CPU. Closes the optimizer-bake-off bullet from earlier
      batches.

#### Correctness / audit work

- [ ] Layer-registry round-trip audit — for every concrete `TNNet*`
      in the LoadFromString/CreateLayer dispatch table, instantiate
      with defaults, save, load, save again, assert bit-for-bit
      string equality. The cheapest possible safety net for the
      "added a layer but forgot to register it" bug; the layer count
      is past 200 so a one-time test like this prevents the entire
      class of regression. (Listed in the seed-491990 batch — pinning
      again here because it is the single highest-leverage missing
      test on the file.)

- [ ] Shape-inference smoke test — instantiate every concrete layer
      at a small canonical input shape, assert the declared output
      shape matches the actually-produced output shape. Catches
      "forgot to call FOutput.ReSize" regressions in seconds.

- [ ] Backward audit for TNNetPointwiseNorm — note in the seed-422331
      follow-ups says its backward is the scalar-only `Mul(1/n)`
      approximation. Either replace with the exact L2-normalize
      Jacobian (now that TNNetL2Normalize implements it) or
      explicitly document the approximation with a deprecation
      comment pointing to TNNetL2Normalize.

- [ ] Audit TNNetSoftCapping behaviour under extreme inputs and add
      a saturation test — same shape of test as the TNNetSoftMin
      follow-up already pinned. Together these two cover the
      "saturating-output layer NaN safety" risk that the
      transformer/QAT examples will eventually exercise.

- [ ] Continue the Backpropagate audit into upsampling/deconvolution
      (TNNetUpsample, TNNetDeconvolution, TNNetDeMaxPool,
      TNNetDeAvgPool, TNNetDeLocalConnect, TNNetDeLocalConnectReLU).
      One numerical-gradient test per layer in TestNeuralNumerical.pas;
      the activation audit found two real bugs in this style of pass.

- [ ] Shared numerical-gradient helper in TestNeuralNumerical.pas —
      already pinned in earlier batches; pinning again because every
      new test in this batch would benefit from it. ~3 lines of test
      per layer instead of a copy-pasted block.

#### Documentation

- [ ] "Building a transformer block" README subsection — pull the
      TNNetMaskedFill / TNNetScaledDotProductAttention /
      TNNetRotaryEmbedding / TNNetGEGLU / TNNetSwiGLU /
      TNNetLayerNorm / TNNetRMSNorm references into one walkthrough
      with a single assembled code snippet. (Listed in the
      seed-781449 batch; re-pinning here because it is now blocked
      only on TNNetMultiHeadSelfAttention.)

- [ ] "Embedding heads" README subsection — group TNNetL2Normalize,
      the proposed TNNetCosineSimilarity, the proposed
      TNNetCosineEmbeddingLoss, the proposed TNNetTripletLoss, and
      TNNetArcFace into one short subsection with a "how to build a
      contrastive head" recipe. (Cross-referenced from seed-491990;
      stays relevant once those layers land.)

- [ ] "Layer authoring checklist" one-pager — constructor +
      LoadFromString round-trip, CreateLayer dispatch entry,
      Compute/Backpropagate, and the mandatory numerical-gradient
      test. Captures the recurring pattern every new-layer task in
      this file actually follows.

- [ ] "How numerical-gradient testing works in this repo" short note
      for contributors — the project's main correctness safety net
      isn't explained anywhere outside the test file itself.

#### Experiments I'm curious about

- [ ] Weight-initialization sensitivity sweep — for a fixed
      deep-ish MLP, plot first-epoch gradient magnitudes across all
      available init schemes. Visualizes why init matters; reuses
      the planned gradient-flow visualizer.

- [ ] Label-smoothing sweep — train SimpleImageClassifier with
      label smoothing `ε ∈ {0, 0.05, 0.1, 0.2}` and tabulate test
      accuracy. A single-knob ablation that beginners always ask
      about and the repo has no concrete answer for.

- [ ] Quick-look NaN-injection test — deliberately inject a NaN
      mid-batch and confirm the NaN/Inf guard hook proposed in the
      seed-491990 batch catches it at the right layer. The minimum
      regression test that hook would need.

### Lucky-day batch — seed 959987

#### Layers I'd enjoy building

- [ ] TNNetLogSoftMax — numerically stable log-softmax along depth.
      Pairs naturally with negative-log-likelihood training and avoids
      the log(softmax(x)) underflow path that several examples emulate
      by hand. Forward uses the log-sum-exp trick; backward is
      `dL/dx_i = dL/dy_i - sum_j(dL/dy_j) * softmax(x)_i`. Full
      numerical-gradient test in TestNeuralNumerical.pas.

- [x] TNNetReGLU — the ReLU-gated sibling of TNNetGLU / TNNetGEGLU /
      TNNetSwiGLU: split input in half along depth, gate one half with
      ReLU of the other. Landed: forward + numerical-gradient test in
      tests/TestNeuralNumerical.pas; registered in both CreateLayer
      dispatches.

- [x] TNNetSinusoidalPositionalEmbedding — parameter-free, additive
      sin/cos position table along the X axis. Landed: reuses
      TVolume.PositionalEncoding (Vaswani formula), configurable `base`
      (default 10000), table built once in SetPrevLayer. Forward,
      identity gradient-check, and LoadFromString round-trip tests.

- [ ] TNNetSnake activation — `x + sin(x)^2 / alpha` with a
      configurable scalar alpha. Periodic activation popular for
      signal/time-series tasks, closed-form derivative, fits next to
      TNNetGaussianActivation in the activation family.

- [ ] TNNetHardTanh and TNNetTanhShrink — two cheap activations that
      round out the tanh family (clip-to-[-1,1] and `x - tanh(x)`
      respectively). Closed-form derivatives, trivial gradient checks,
      good filler for the planned activation bake-off table.

- [ ] TNNetCELU — continuously-differentiable ELU,
      `max(0,x) + min(0, alpha*(exp(x/alpha) - 1))`. Differs from the
      existing TNNetELU at the elbow; useful as a comparison point in
      the activation bake-off and a one-line change from the ELU
      implementation.

- [ ] TNNetSpatialDropout2D — drop *whole channels* at train time
      (vs. per-element TNNetDropout). The convolution-friendly dropout
      variant; descend from TNNetAddNoiseBase like TNNetDropPath so
      `TNNet.EnableDropouts` toggles it. Seeded numerical-gradient
      test asserting masked channels' gradients are zero.

- [ ] TNNetSqueezeExcitation — channel-wise attention block (global
      avg pool → 1x1 dense → ReLU → 1x1 dense → sigmoid → broadcast
      multiply). Probably best built as a small composer helper using
      existing layers rather than a monolithic layer, with a
      gradient-check test on the full composed graph.

- [x] TNNetCosineSimilarity — split input in half along depth, return
      `cos(a, b) = a·b / (|a||b|)` per (X,Y) cell. The missing piece
      of the embedding-heads recipe alongside TNNetL2Normalize.
      Full forward + numerical-gradient test; the backward is a small
      but standard expression that benefits from a dedicated check.

- [ ] TNNetGumbelSoftmax — softmax with Gumbel-noise temperature
      annealing for differentiable discrete sampling. Reuses the new
      exact softmax Jacobian; seeded forward test plus
      central-difference gradient check at fixed temperature.

#### Examples I'd enjoy writing

- [ ] `examples/CharTokenizer/` — minimal in-memory char tokenizer +
      trainable embedding lookup, with a t-SNE-free nearest-neighbor
      printout ("nearest 5 chars to 'q'"). Closes the
      "tokenizer + trainable embeddings" bullet at the top of the file
      without needing the full transformer.

- [ ] `examples/EmbeddingHeadDemo/` — train a small net to learn an
      embedding space on a toy 3-class dataset using TNNetL2Normalize
      + a hand-rolled triplet loss, and print the per-class
      cosine-similarity matrix at the end. Concrete payoff for the
      L2Normalize / cosine-similarity layers.

- [ ] `examples/MixUpAblation/` — train SimpleImageClassifier with
      and without MixUp data augmentation on CIFAR-10 and report the
      delta. A small, self-contained answer to "does MixUp actually
      help in this codebase".

#### Correctness / audit work

- [ ] TestExtensions check: every layer that declares
      `FStruct[k]` constructor parameters should be tested for
      LoadFromString round-trip with non-default values, not just
      defaults. The existing register-audit (already pinned)
      catches "did you register the layer"; this catches "did you
      serialize every constructor argument". One table-driven test
      covers them all.

- [ ] TNNetAddPositionalEmbedding scale-factor backward check —
      verify the gradient w.r.t. the learnable positional table
      matches central differences when the input shape is rectangular
      (X≠Y). Square-shape inputs can hide off-by-one indexing bugs.

- [ ] Gradient-flow regression test — train a 12-layer ReLU MLP on a
      tiny synthetic dataset for one epoch with and without a single
      TNNetLayerNorm/RMSNorm inserted at the midpoint; assert that the
      per-layer gradient norms with the norm layer are uniformly
      bounded above the no-norm case. Encodes the "norms fix vanishing
      gradients" promise as a CI check.

#### Experiments I'm curious about

- [ ] Periodic-activation toy benchmark — fit `y = sin(3x) + 0.3 sin(11x)`
      with the proposed TNNetSnake vs. ReLU/GELU/Tanh MLPs of equal
      width/depth. Single-knob comparison that highlights when
      periodic activations matter, and gives Snake an out-of-the-box
      success story.

- [ ] Dropout-vs-DropPath head-to-head — train the same small CIFAR
      model with element-wise TNNetDropout and whole-sample
      TNNetDropPath at matched effective drop rates, report final
      test accuracy. Concrete payoff for the recently-landed
      TNNetDropPath layer.

#### Documentation

- [ ] "Choosing an activation" one-paragraph cheat-sheet in the
      README — for each landed activation, one sentence on when to
      prefer it (e.g. "GELU: transformer FFNs", "SoftPlus: when you
      need a positive smooth output", "PenalizedTanh: dead-ReLU
      hedge"). Cross-references the activation bake-off when it
      lands.

### Lucky-day batch — seed 689126

A fresh sweep, picked to fill gaps left by the transformer-block work:
small, composable layers; clean numerical-gradient targets; and a handful
of self-contained experiments that pay off the layers already on disk.

#### Layers I'd enjoy building

- [ ] TNNetReZero — a single learnable scalar `alpha` initialized to 0,
      applied as `y = alpha * x`. Sibling of TNNetLayerScale but with a
      single shared scalar instead of per-channel; the "ReZero is all you
      need" trick that lets very deep residual nets train without warmup.
      Forward + numerical-gradient + LoadFromString round-trip tests in
      tests/TestNeuralNumerical.pas.

- [ ] TNNetGRN — Global Response Normalization (ConvNeXt v2):
      `y = gamma * (x * (||x||_2,channel / mean(||x||_2,channel))) + beta + x`.
      Channel-wise contrastive normalization with two learnable
      per-channel vectors. Slots in next to LayerNorm/RMSNorm/GroupNorm.

- [ ] TNNetSoftSign — `y = x / (1 + |x|)`. Cheap bounded activation;
      closed-form derivative `1 / (1+|x|)^2`. Fills a real gap in the
      activation family; one-line forward + numerical-gradient test.

- [ ] TNNetMaxOut — split input into `k` groups along depth, output the
      elementwise max per group (output depth = input_depth / k).
      Parameter-free; backward routes gradient only through the argmax
      slice. Configurable `k` via constructor. Forward + grad check.

- [ ] TNNetPixelShuffle — sub-pixel upsampling: rearrange depth-r² →
      (X*r, Y*r, depth) for super-resolution heads. Pair with the
      existing TNNetUpsample as a learnable-free alternative.
      Round-trip test plus a forward shape assertion.

- [ ] TNNetAdaptiveAvgPool — target output (X,Y) regardless of input
      size; computes per-cell averaging windows. Useful for classifier
      heads that need a fixed feature-map size before a dense layer.
      Backward is the standard average-pool transpose.

- [ ] TNNetHuberLoss / TNNetSmoothL1Loss output layers — robust
      regression losses that pair naturally with the Hypotenuse-style
      examples. Quadratic near 0, linear in the tails; closed-form
      gradient. Add to the loss/output-layer family with a numerical-
      gradient sanity test against synthetic targets.

- [ ] TNNetFocalLoss — class-imbalanced classification loss
      `-(1-p_t)^gamma * log(p_t)` with configurable `gamma`. Wraps the
      softmax-cross-entropy path; a single-knob ablation that the
      sentiment/malaria examples could opt into.

- [ ] TNNetALiBiBias — Attention with Linear Biases: add a static linear
      bias `m_h * (i - j)` to attention scores before softmax, with
      head-specific slope `m_h`. Parameter-free, no learnable weights;
      drops in next to TNNetMaskedFill in the attention pipeline.
      Forward + identity-gradient pass-through test.

- [ ] TNNetTalkingHeads — small mixing projection across heads before
      and after the softmax in multi-head attention. Two learnable
      `H x H` matrices, easy gradient check, slots in after the
      MultiHeadSelfAttention task at the top of the file lands.

- [ ] TNNetMishExact audit — TNNetMish exists with a numerical-gradient
      test (per the earlier audit); verify the analytic derivative
      matches the central-difference check at extreme inputs
      (|x| > 20) where the `tanh(softplus(x))` form can saturate.
      Either tighten the existing test or document the saturation
      behavior in the code.

#### Examples I'd enjoy writing

- [ ] `examples/ActivationBakeOff/` — small MLP on a fixed toy dataset
      (e.g. spiral or two-moons synthesized in-memory), trained
      identically with ReLU / GELU / Swish / Mish / SELU / SquaredReLU /
      SoftPlus / GaussianActivation. Prints a table of final loss and
      epochs-to-converge. Pure-CPU, runs in seconds, no external data.
      Closes the "Activation-function bake-off" experiment bullet above.

- [ ] `examples/NormBakeOff/` — same idea but for normalization: no-norm
      / LayerNorm / RMSNorm / GroupNorm on a small CNN. Demonstrates the
      gradient-flow benefit of the normalization layers that have
      landed.

- [ ] `examples/SwiGLUFeedForward/` — the natural follow-up to the
      already-landed gated-activation layers: a tiny FFN block
      `Dense -> SwiGLU -> Dense` trained on a synthetic regression
      task. Demonstrates how to compose the gated activation with the
      surrounding projections, ready for the transformer-encoder task.

- [ ] `examples/BinaryAddition/` — the "learns to add two binary
      numbers" sequence example from an earlier batch. Self-contained,
      no external dataset, finishes in under a minute on CPU. Good
      smoke-test entry point alongside Hypotenuse / XorAndOr.

- [ ] `examples/PositionalEncodingDemo/` — visualize (ASCII heatmap)
      the sin/cos table that TNNetSinusoidalPositionalEmbedding builds
      vs. the learnable table from TNNetAddPositionalEmbedding after a
      few epochs of training. Concrete payoff for the recently-landed
      sinusoidal layer.

#### Correctness / audit work

- [ ] TestExtensions table-driven LoadFromString check — one parametric
      test that builds each layer with a non-default constructor value
      (e.g. `Padding=2` for convolutions, `alpha=0.5` for activations,
      `causal=True` for ScaledDotProductAttention), serializes the net,
      reloads, and asserts forward outputs match within 1e-6. Catches
      "forgot to serialize the new ctor arg" bugs across the entire
      layer registry.

- [ ] Continue the Backpropagate audit into the upsampling/decon family
      (TNNetUpsample, TNNetDeconvolution, TNNetDeMaxPool, TNNetDeAvgPool,
      TNNetDeLocalConnect, TNNetDeLocalConnectReLU) — explicitly pinned
      in two previous batches; pinning again because this is the last
      family without numerical-gradient coverage and the activation
      audit found two real bugs in this style of code path.

- [ ] Shared numerical-gradient helper in tests/TestNeuralNumerical.pas
      that wraps the "build net, set input, central-difference, compare"
      block into a single procedure. Every new-layer task in this file
      copy-pastes ~25 lines today; the helper would drop that to ~3.
      Pinned across three batches now — flagging that the duplication
      cost keeps growing.

- [ ] NaN/Inf guard hook — opt-in `TNNet.AssertFiniteForward := True`
      that scans each layer's output volume after Compute and raises
      with the layer index when a NaN or Inf appears. Plus a one-shot
      regression test that deliberately seeds a NaN and confirms the
      assertion fires at the right layer. Real safety net for the
      transformer/softmax/RMSNorm work.

#### Experiments I'm curious about

- [ ] Lottery-ticket sanity check — train a small MLP on a toy dataset,
      record the top-k% magnitude mask, reset to original init, retrain
      with the mask applied. Print final-loss delta vs. full-density
      retrain. Educational reproduction of the lottery-ticket
      hypothesis using only what the library already ships.

- [ ] Init-scheme sensitivity heatmap — for the deep ReLU MLP from
      seed-959987's gradient-flow regression idea, render an ASCII
      heatmap of per-layer activation magnitudes across each available
      init scheme (Glorot/He/Orthogonal/etc). Visual companion to that
      regression test.

- [ ] Sequence-length scaling micro-benchmark — measure forward+
      backward wall-time for TNNetScaledDotProductAttention at
      seq_len ∈ {16, 32, 64, 128, 256} with d_k fixed. Confirms the
      expected O(n²) scaling and gives a baseline number to compare
      against once TNNetMultiHeadSelfAttention lands.

- [ ] "Does positional encoding actually help" ablation — train the
      planned tiny char-sequence model with and without
      TNNetSinusoidalPositionalEmbedding, report convergence delta.
      Concrete payoff for the layer that just landed.

#### Documentation

- [ ] "Building a transformer block" README walkthrough — pinned in
      seed-959987 and seed-781449; re-pinning because the layers it
      depends on (MaskedFill / ScaledDotProductAttention /
      RotaryEmbedding / GEGLU / SwiGLU / LayerNorm / RMSNorm /
      SinusoidalPositionalEmbedding) have all landed; the only
      remaining blocker is the TNNetMultiHeadSelfAttention task at the
      top of the file.

- [ ] Layer-authoring checklist one-pager — pinned twice already;
      flagging here that every new layer in this batch would benefit
      from it landing first (ctor + LoadFromString round-trip,
      CreateLayer dispatch in *both* places, Compute/Backpropagate,
      numerical-gradient test). The recurring pattern every "Layers
      I'd enjoy building" entry in this file actually follows.

- [ ] CHANGELOG.md (or a "What's new since v2.x" README section) —
      five-plus lucky-day batches have landed, but a contributor
      reading top-down has no compact summary of which layers exist
      now vs. which the README snippets still imply are missing.
      One bullet per landed layer with date + commit short SHA.

### Lucky-day batch — 2026-05-15 (seed 353052)

A fresh draw. Cross-checked each unchecked name below against
`neural/neuralnetwork.pas` to avoid the re-pinning trap noted in
the seed-993208 meta-observation. None of these layer names appear
in the dispatch tables today.

#### Layers I'd enjoy building (each one commit-sized)

- [x] TNNetBias — bias-only "add a learnable per-channel offset" layer.
      Landed (seed-758266 batch, commit b1394cd): descends from
      TNNetChannelTransformBase next to TNNetLayerScale, single neuron
      with a Depth-sized weight vector, Compute uses
      FOutput.AddToChannels, Backpropagate accumulates the true
      per-channel sum into the delta and passes the input gradient
      through unchanged. Registered in both CreateLayer dispatches.
      Four tests: TestBiasForward / TestBiasGradientCheck /
      TestBiasWeightGradientCheck / TestBiasSerializationRoundTrip.

- [x] TNNetReZero — single learnable scalar `alpha` (init 0), forward
      `y = alpha * x`. Landed (seed-758266 batch, commit 18281db):
      descends from TNNetChannelTransformBase, SetPrevLayer forces
      SetNumWeightsForAllNeurons(1,1,1) so the parameter is a true
      scalar, Compute is FOutput.Mul(alpha), Backpropagate uses the
      scalar dot-product sum(OutputError * PrevOutput) for the alpha
      delta and OutputError*alpha for the input gradient. Registered
      via Ft[0] in both CreateLayer dispatches. Four tests:
      TestReZeroForward (default alpha=0 zeroes output) /
      TestReZeroGradientCheck / TestReZeroWeightGradientCheck /
      TestReZeroSerializationRoundTrip.

- [ ] TNNetGatedResidual / TNNetGatedSkip — `y = x + gate * Sublayer(x)`
      where `gate` is a per-channel learnable initialized to zero (the
      ReZero trick at channel granularity). Already listed multiple
      times; pinning the implementation shape: a thin wrapper layer
      that holds the per-channel gate and composes with whatever
      sub-net is built on top.

- [ ] TNNetMaskedMean / TNNetMaskedMax — pooling over a variable-length
      sequence with a {0,1} mask supplied as an extra depth channel.
      Replaces the existing "pad with zeros and hope the average is
      small" workaround for sentence-classification-style heads.
      Forward sums the masked positions then divides by the mask sum;
      backward routes the gradient only to unmasked positions. Two
      tests per layer (forward correctness against a hand-computed
      reference, numerical-gradient on the input).

- [ ] TNNetAdaptiveAvgPool — target output (X,Y) regardless of input
      size; computes per-cell averaging windows. Useful for classifier
      heads that want a fixed dense-input shape independent of the
      backbone's spatial size. Forward is bucket-then-average per cell;
      backward distributes the gradient uniformly back to the source
      bucket. Forward + numerical-gradient test on a small odd shape
      (e.g. input 5x5 → output 3x3) so the non-integer windows are
      actually exercised.

- [ ] TNNetMultiHeadSelfAttention — yes, again. Reading this file
      top-to-bottom, MHSA is the single most-pinned open item across
      the lucky-day batches. SDPA + RoPE + ALiBi + MaskedFill +
      LayerNorm + the gated FFNs are all in tree. The proposed shape
      is a composition layer that internally builds and owns H copies
      of TNNetScaledDotProductAttention plus three Q/K/V dense
      projections and one output projection — gradient flows through
      the owned sub-layers, so Backpropagate has no new math. One
      gradient-check test at d_model=8, n_heads=2, SeqLen=4.

- [x] TNNetTokenShift — the RWKV-style time-mixing primitive:
      `y[t] = mix * x[t] + (1 - mix) * x[t-1]` with a per-channel
      learnable `mix`. Landed (seed-758266 batch, commit b61c9a4):
      descends from TNNetChannelTransformBase, parameter-free Create()
      with mix initialized to 0.5 per channel, SetPrevLayer validates
      SizeY=1 (1D-sequence convention shared with the rest of the
      transformer-block layers). Forward: y[0,c]=mix[c]*x[0,c];
      y[t,c]=mix[c]*x[t,c]+(1-mix[c])*x[t-1,c]. Backward: per-channel
      dL/dmix[c] = sum_t gy[t,c]*(x[t,c]-x[t-1,c]); input gradient
      scatters gy[t,c]*mix[c] to position t and gy[t,c]*(1-mix[c]) to
      position t-1. Four tests: TestTokenShiftForward (mix=1 identity,
      mix=0 strict right-shift, per-channel mid mix hand-computed) /
      TestTokenShiftGradientCheck / TestTokenShiftWeightGradientCheck /
      TestTokenShiftSerializationRoundTrip.

- [ ] TNNetCausalConv1D — depth-axis 1D causal convolution (i.e. the
      kernel sees only `t-k..t`, never `t+1..t+k`). Useful for
      sequence baselines that compare against the SDPA-based attention
      stack at matched parameter count. Forward = standard 1D conv
      with left-padding; backward is the standard conv-transpose
      restricted to the causal window.

- [ ] TNNetHuberLoss / TNNetSmoothL1Loss output layers — robust
      regression losses with `delta` in `FFloatSt[0]`. Quadratic near
      zero, linear in the tails; closed-form gradient. Pairs naturally
      with the Hypotenuse-style examples and the still-open
      time-series-forecasting example. Already pinned in seed-689126;
      re-pinning because the implementation is a half-hour job and
      unblocks the forecasting example.

- [ ] TNNetFocalLoss — class-imbalanced classification loss
      `-(1-p_t)^gamma * log(p_t)`, `gamma` in `FFloatSt[0]`
      (default 2.0). Wraps the softmax-cross-entropy path; one of the
      sentiment / colorectal / malaria examples could opt into it as
      a one-line ablation knob.

- [ ] TNNetLabelSmoothingCE — cross-entropy with `epsilon` smoothing
      (default 0.1). One float param; identical Jacobian to standard
      CE after the smoothing transform applied to the targets.

#### Composite blocks / examples I'd enjoy shipping

- [ ] TNNetPreNormResidual helper — universe pressure is now off the
      charts: this one-liner builder `y = x + Sublayer(LayerNorm(x))`
      has been pinned at least six times across the lucky-day batches.
      Treating that as a personal commitment to actually ship it. Add
      to `TNNet` as a builder method (`AddPreNormResidual(BuildBranch)`)
      so the sub-net is constructed inside the lambda-style callback,
      matching the existing add-block patterns elsewhere in the API.

- [ ] `examples/BinaryAddition/` — re-pinned. The "learn to add two
      binary numbers" sequence task. Self-contained synthetic dataset
      (generate two N-bit numbers, target = their sum bits). Fits in
      one file under 200 lines; pairs nicely with the existing
      Hypotenuse / XorAndOr examples as a "smallest possible sequence
      task" entry point.

- [ ] `examples/SineRegression/` — re-pinned from seed-135518. 1D
      function-fitting toy `y = sin(2*pi*x) + noise` with a tiny
      two-layer MLP that prints train loss per epoch. Smallest possible
      "does the library still train?" demo, suitable as the README
      quick-start snippet.

- [ ] `examples/AttentionViz/` — load a tiny trained SDPA model and
      dump the per-head attention matrix as a PGM image so the
      diagonal / induction patterns are visible without any plotting
      dependency. Re-pinned from seed-135518; would pair beautifully
      with the "echo the previous token" SDPA demo also on the list.

- [ ] `examples/TimeSeriesForecast/` — synthetic AR(1) or sine + noise
      sequence, train a small TNNetCausalConv1D or SDPA model to
      predict `x[t+1]` from `x[t-k..t]`, with TNNetHuberLoss as the
      output. Lands three open items in one example: the
      time-series-forecasting bullet, the Huber loss, and the
      CausalConv1D baseline.

#### Tests / numerical-gradient audit

- [ ] Continue the upsampling / deconvolution audit: TNNetDeconvolution
      input + weight gradients. The Upsample / DeMaxPool / DeAvgPool
      members of the family are already covered with numerical-grad
      tests; Deconvolution is the last remaining member and the most
      interesting one because of the weight-gradient path. Likely
      benefits from the Double-precision SSE accumulator helper
      already pinned earlier in this file.

- [ ] Shared `LayerInputAndWeightGradientCheck` helper in
      `tests/TestNeuralNumerical.pas` — pinned multiple times across
      the batches; calling it out again because every new
      learnable-weight layer in this batch (TNNetBias, TNNetReZero,
      TNNetGatedResidual, TNNetTokenShift, TNNetCausalConv1D) would
      drop from a copy-pasted 50-line block to a 3-line call.

- [ ] Argmax-tie behaviour test for TNNetGlobalMaxPool (open at line
      1763). Cheap to write, and the documented tie-break ("first
      wins" on a sequential scan) deserves a pinning test before any
      future parallel-pool refactor silently changes it.

- [ ] Sanity-check pass for the `tests/Test*` programs other than
      TestNeuralNumerical: list which ones are still actively
      maintained vs which are crufty leftovers from earlier eras of
      the repo. Output a one-paragraph note added to this tasklist
      so future contributors know which test program to extend.

#### Experiments I'd enjoy running

- [ ] Activation menagerie bake-off (re-pinned, but now with the
      latest landings in the menu): same fixed-seed tiny CIFAR net,
      swap the activation each run across ReLU / LeakyReLU / GELU /
      Swish / Mish / ELU / CELU / SELU / SiLU / SoftSign / SquaredReLU
      / Snake / HardTanh / TanhShrink / SoftPlus / ShiftedReLU /
      Threshold / HardShrink / SoftShrink / LogSigmoid / MaxOut. The
      roster is now genuinely large; the experiment is a real
      teaching artifact. Output `docs/experiments/activations.md`
      with the final accuracy table and per-epoch wall-clock.

- [ ] Normalization menagerie bake-off — same idea, narrower roster:
      no-norm / ChannelStdNorm / LayerNorm / RMSNorm / GroupNorm /
      InstanceNorm / PixelNorm. One chart, one CSV, same tiny CIFAR
      net. Settles the "which one wins at this scale?" question and
      ships a reusable benchmark harness for future norm landings.

- [ ] Position-encoding bake-off (re-pinned from multiple earlier
      batches): no position info / sinusoidal AddPositionalEmbedding /
      TNNetSinusoidalPositionalEmbedding / RoPE / ALiBi. Same tiny
      next-token task, four-or-five training runs, one chart. Now
      fully unblocked.

- [ ] Optimizer bake-off: SGD / SGD+momentum / Adam / RMSProp on a
      fixed tiny MLP and dataset. Re-pinned from earlier batches;
      complements the activation/norm/position bake-offs and is the
      simplest of the four to implement because no new layers are
      needed.

- [ ] "Lucky-number reproducibility": run the existing
      TNNet.SaveToString / LoadFromString sweep with the global RNG
      seed pinned to 353052, save outputs of a tiny canonical net,
      reload, and assert bit-identical outputs. Tiny but settles
      whether the repo's loaded networks are truly deterministic
      under a frozen seed — a real concern across the parallel
      reductions and FP32-cancellation paths called out earlier.

#### Tooling / dev experience

- [ ] `scripts/grep_layer.sh <TNNet...>` — pinned across at least
      four batches. I'd actually take it this time. Three blocks of
      output: the `class(...)` declaration with surrounding context,
      the Compute and Backpropagate method bodies, and every test
      method in `tests/` that mentions the class name. Captures the
      first 30 seconds of every audit I keep doing by hand.

- [ ] `scripts/list_activations.sh` — a tiny report: enumerate every
      class descending from TNNetReLUBase plus every class whose
      Compute does an element-wise transform, and cross-reference
      against `Test*Gradient` methods to find activations that lack
      a numerical-gradient test. Useful warm-up for the next audit
      batch.

- [ ] Tiny `bin/` micro-benchmark printing ns/op for Add, Mul,
      DotProduct, AddArea, InterleaveSplit. Pinned multiple times.
      Single short program, no OpenCL/AVX path required.
      Table-formatted output plus a CSV so future regressions are
      visible. Worth landing as a baseline before the
      sequence-length scaling SDPA benchmark above runs.

- [ ] CI shim — re-pinned: a 15-line GitHub Actions workflow that
      runs `tests/RunAll.sh` on push. The script already has the
      right exit semantics; the YAML is essentially boilerplate.

#### Documentation

- [ ] "Building a transformer block" README walkthrough — re-pinned
      from seed-689126 (and earlier). The only remaining true blocker
      is TNNetMultiHeadSelfAttention; everything else the walkthrough
      needs has landed. Worth writing as soon as MHSA above ships.

- [ ] Activation cheat-sheet in `docs/activations.md` — re-pinned
      multiple times across batches. The current activation roster is
      large enough that a flat README list no longer scales; one
      table with formula / derivative / bounded? / smooth-at-zero? /
      typical use is the right shape.

- [ ] Normalization cheat-sheet in `docs/normalization.md` — same
      story for the normalization family. With PixelNorm now landed,
      the family has six members; tabulating them once will save
      every contributor's first 20 minutes of "wait, which norm
      reduces over which axis?".

- [ ] CHANGELOG.md / "What's new" section — re-pinned from seed-689126.
      Five-plus lucky-day batches have landed without any compact
      summary of which layers exist now vs. which the older README
      snippets still imply are missing. One bullet per landed layer
      with date + commit short SHA would close this for good.

#### Stretch / ambitious (would genuinely enjoy)

- [ ] Tiny GPT char-level transformer end-to-end on CPU — the
      headline open item at the very top of this file. Once MHSA
      above lands, all that's left is a tokenizer wrapper and a
      training-loop file. Lucky-day flavor: ship
      `examples/TinyGPT/data/tinyshakespeare.txt` (a few kilobytes),
      a 2-layer model, and a `Generate` routine that prints
      recognizably Shakespeare-flavored output after a few minutes
      of training.

- [ ] Mixture-of-Experts routing layer — top-k softmax gate over N
      expert sub-networks plus a load-balancing auxiliary loss.
      Stretch; well-scoped once MHSA lands and the
      TNNetPreNormResidual helper above lands.

- [ ] ONNX (or simpler JSON) export — re-pinned from earlier batches.
      Minimal viable: dump a forward-only graph for the
      currently-supported subset of layers, enough to run inference
      in `onnxruntime`. Document which layers are out-of-scope for
      v1. Closes the "I trained it, now what?" gap for non-Pascal
      downstream users.

### Lucky-day batch — 2026-05-15 (seed 758266)

This batch landed three "stabilizer-family" / sequence layers:
TNNetBias (per-channel learnable offset, commit b1394cd),
TNNetReZero (single learnable scalar, commit 18281db) and
TNNetTokenShift (RWKV-style per-channel time-mixing, commit b61c9a4).
Each ships with four tests (forward, input grad, weight grad,
LoadFromString round-trip) and both CreateLayer dispatches are wired.
`tests/RunAll.sh` is green at 488 tests / 0 failures.

#### Doable follow-ups directly unblocked by the new layers

- [ ] TNNetMul — multiplicative companion to TNNetBias: learnable
      per-channel multiplier with closed-form gradient
      `dL/dW[c] = sum_pix(dy[x,y,c] * x[x,y,c])`. Same structural shape
      as the just-landed TNNetBias; together they spell out the
      affine `y = Mul * x + Bias` decomposition that TNNetLayerScale
      gives you fused. One commit, four tests modeled on TestBias*.

- [ ] TNNetAffineBlock helper — once TNNetMul ships, expose a builder
      `AddAffine(Depth)` that wires `TNNetMul -> TNNetBias` so callers
      don't have to spell out the pair. Same shape as the existing
      `AddPreNormResidual` family discussed earlier in the file.

- [ ] TNNetGatedResidual — the per-channel-gate twin of the new
      TNNetReZero. Same single-neuron pattern but with a Depth-sized
      weight vector and a different forward (`y = x + gate * x`-style
      composition with a sub-net). The scalar TNNetReZero is now a
      ready template — copy SetPrevLayer / Compute / Backpropagate
      and grow the weight to Depth. Mark line 717-720 of this file
      `[x]` once it lands.

- [ ] `examples/BiasOnlyTuning/` — when fine-tuning, freeze every
      learnable layer except inserted TNNetBias layers and train only
      those. Demonstrates the "bias-only adapter" pattern from
      modern fine-tuning literature and exercises the new layer
      end-to-end. ~80 lines; pairs with the existing Hypotenuse-style
      examples.

- [ ] Token-shift micro-experiment: train a tiny next-token model
      with and without a single TNNetTokenShift in front of an MLP.
      Already pinned earlier in the file (line 741); now fully
      unblocked. Concrete data point on "how much of attention is
      really just mixing-along-time" — fits in one short example.

- [ ] ReZero deep-MLP convergence demo: train a 16-layer residual
      MLP with and without TNNetReZero on the residual branch
      (alpha=0 init means the network starts as the identity).
      Concrete visualization of why the trick stabilizes deep
      stacks. Pure CPU, finishes in seconds.

#### Tiny correctness / audit follow-ups

- [ ] LoadFromString round-trip for non-default ctor values: the
      three new layers' serialization tests use either default or
      perturbed-weight harnesses. Add an explicit case that constructs
      TNNetReZero(0.5), serializes, and re-loads, asserting the
      preserved alpha. Same idea for TNNetBias initialized via
      direct weight write.

- [ ] Shape-edge test for TNNetTokenShift: assert SetPrevLayer raises
      the documented error when SizeY > 1 (mirrors the
      TNNetRotaryEmbedding odd-depth guard test pattern).

- [ ] Two-layer TokenShift composition test: stack TNNetTokenShift
      twice with different mix values and verify analytical
      vs. numerical gradient through both layers (catches any subtle
      double-pass bug in the t-1 / t+1 input-gradient scatter).

#### Meta-observation for the next batch

TNNetBias / TNNetReZero / TNNetTokenShift share the same
"per-channel (or scalar) learnable weight" skeleton. With three
sibling templates now landed, the next learnable-weight layer
(TNNetMul, TNNetGatedResidual, TNNetCausalConv1D's bias path,
TNNetDyT's alpha) should drop from a copy-pasted 40-line block to
a single SetPrevLayer + a one-line Compute + a short Backpropagate.
The shared `LayerInputAndWeightGradientCheck` test helper pinned
multiple times earlier in this file would now save four tests' worth
of duplication per landing — calling that out one more time.

### Lucky-day batch — 2026-05-15 (seed 97846)

Coming in fresh after the Bias / ReZero / TokenShift trio. The batches
above keep re-pinning TNNetMul, TNNetGatedResidual, TNNetDyT, and the
shared `LayerInputAndWeightGradientCheck` helper — I'd genuinely enjoy
closing those threads. Below: a fresh wave of ideas I'd like to take
on, scoped for a single commit each, and double-checked against the
existing source tree so none duplicate a layer that already ships.

Verified absent in `neural/neuralnetwork.pas` at the time of writing:
TNNetMul (the multiplicative twin of TNNetBias), TNNetGatedResidual,
TNNetCausalConv1D, TNNetDyT, TNNetMixtureOfExperts, TNNetPixelShuffle,
TNNetSpaceToDepth / TNNetDepthToSpace, TNNetSwitchableNorm, TNNetGRN,
TNNetReversibleBlock, TNNetStochasticPool, TNNetTalkingHeads,
TNNetSinkAttention, TNNetCosineSimilarityAttention,
TNNetMultiHeadSelfAttention, TNNetCrossAttention,
TNNetRMSNormGated, TNNetBitLinear, TNNetPolynomialActivation,
TNNetHuberLoss / TNNetSmoothL1Loss / TNNetFocalLoss /
TNNetLabelSmoothingLoss. (TNNetCosineSimilarity *does* exist as a
similarity-score layer; the attention variant is distinct.)

#### Quick wins I'd take first

- [ ] TNNetMul — multiplicative per-channel learnable scale, the twin of
      the just-landed TNNetBias. Forward `y[x,y,c] = w[c] * x[x,y,c]`;
      backward `dL/dW[c] = sum_pix(dy[x,y,c] * x_prev[x,y,c])` and
      `dL/dx_prev[x,y,c] = w[c] * dy[x,y,c]`. Mirror TNNetBias' four
      tests (forward, input grad, weight grad, LoadFromString round-trip)
      and wire both CreateLayer dispatches. Smallest non-trivial
      learnable-weight layer left on the list; together with TNNetBias
      it spells out the affine decomposition that TNNetLayerScale
      fuses.

- [ ] TNNetAffineBlock builder helper — once TNNetMul ships, expose
      `TNNet.AddAffine(Depth)` that wires `TNNetMul -> TNNetBias` so
      callers don't have to spell out the pair. One-line builder
      mirroring the existing `AddGroupedFullConnect` style; a single
      end-to-end "AddAffine matches Mul+Bias" test pins the
      composition.

- [ ] Shared `LayerInputAndWeightGradientCheck(layer, inputShape)`
      helper in `tests/TestNeuralNumerical.pas`. The TNNetBias /
      TNNetReZero / TNNetTokenShift landings each duplicated a ~50-line
      block; calling this out one more time and committing to actually
      ship the helper before the next learnable-weight layer lands.
      Three-line tests instead of copy-pasted scaffolding. Should
      handle both the input and weight central-difference checks, with
      a `Tolerance` parameter (default 1e-2) so the DeMaxPool-style
      Double-precision SSE accumulator can be opted into per-test.

#### Layers I'd enjoy building (no MHA dependency)

- [ ] TNNetGatedResidual — per-channel zero-initialised learnable gate
      `y = x + alpha[c] * Sublayer(x)` (the ReZero-with-channel-dim
      variant). The just-landed TNNetReZero is the scalar template;
      grow alpha from a scalar to a Depth-sized vector. Pins line
      717–720 of this file once it lands. Same four tests as TNNetBias
      / TNNetReZero.

- [ ] TNNetDyT (Dynamic Tanh) — `y = gamma[c] * tanh(alpha * x) + beta[c]`,
      the Liu-et-al-2025 LayerNorm replacement. Per-layer learnable
      scalar `alpha` plus per-channel `gamma` and `beta`. Backward is
      one closed-form `(1 - tanh^2)` chain. Template-wise it's a
      LayerNorm body with the norm bookkeeping deleted and a tanh in
      the middle. Numerical-gradient test on a tiny input.

- [ ] TNNetCausalConv1D — 1D convolution with left-only padding so the
      output at position t depends only on positions ≤ t. Wraps the
      existing TNNetConvolution with a TNNetPadXY in front and a crop
      on the back, OR (cleaner) a small dedicated layer that skips the
      future-side multiplications entirely. Backward is the same
      convolution backward minus the masked-future part. Pairs with
      TNNetTokenShift and unblocks the attention-free baseline
      experiments listed several batches above.

- [ ] TNNetReversibleBlock (RevNet additive coupling): given an even
      split of the input depth into `x1 | x2`, forward computes
      `y1 = x1 + F(x2); y2 = x2 + G(y1)` and concatenates. Backward is
      derived from the forward by inverse algebra (`x2 = y2 − G(y1);
      x1 = y1 − F(x2)`), so a forward-then-inverse round-trip test is
      a stronger correctness signal than any numerical-gradient check.
      F and G are user-supplied sub-nets — accept them as constructed
      TNNet instances in the constructor.

- [ ] TNNetGRN (Global Response Normalization from ConvNeXt-V2):
      `y[c] = gamma[c] * (x[c] * ||x[c]||_2 / mean_c(||x[c]||_2)) + beta[c] + x[c]`.
      Per-channel L2 norm computed across (SizeX,SizeY), then divided
      by the across-channel mean and rescaled. Cheap, transformer-
      and conv-friendly, two learnable params per channel. Numerical-
      gradient test plus a "constant-input → identity-ish" sanity check.

- [ ] TNNetSwitchableNorm — single layer that learns a 2-way softmax
      `w` over `LayerNorm` and `RMSNorm` outputs:
      `y = w[0] * LayerNorm(x) + w[1] * RMSNorm(x)`. Two learnable
      scalars summed via softmax (so initial gradient is well-defined).
      Tiny, fun, and a concrete "let the network pick its normalizer"
      experiment without dragging in any new norm math.

- [ ] TNNetStochasticPool — sample one cell per pooling window weighted
      by its activation (softmax of activations over the window) at
      training, take the expectation (`sum(p_i * x_i)`) at inference.
      Backward: gradient flows to the sampled cell only (training) or
      to all cells weighted by `p_i` (inference). Seeded RNG in the
      test (mirrors DropPath's pattern).

- [ ] TNNetPixelShuffle (sub-pixel convolution) — output spatial size
      = input * r, output channels = input / (r*r). Deterministic
      index permutation; backward is its inverse. Numerical-gradient
      test is trivial because the operation is linear; pairs naturally
      with the SuperResolution example for an end-to-end demo
      (also pinned earlier).

- [x] TNNetSpaceToDepth + TNNetDepthToSpace — inverse permutation pair.
      Same shape as PixelShuffle's forward/backward but configured for
      the ViT patchify step (`SpaceToDepth(P) on a (P*Hp, P*Wp, C)
      input → (Hp, Wp, P*P*C)`). Landed: parameter-free pair descending
      from TNNetIdentity with divisibility validation, registered in both
      CreateLayer dispatch sites. 5 tests in TestNeuralNumerical.pas
      (forward each direction, analytic-identity round trip, and
      central-difference gradient check for each).

- [ ] TNNetBitLinear (BitNet ternary-weight FullConnect) — forward
      uses `sign(W) * mean(|W|)`; backward keeps the full-precision
      shadow weights via the straight-through estimator (gradient
      flows straight through the sign step). Self-contained quantization-
      aware-training experiment, no new infra. Numerical-gradient
      test on the STE path will need a tolerance loosened to ~1e-1
      (standard for STE checks); document the choice in the test.

- [ ] TNNetCosineSimilarityAttention — variant of TNNetScaledDotProductAttention
      that replaces `Q·Kᵀ / √d` with `(Q/||Q||)·(K/||K||)ᵀ * scale`,
      where `scale` is either fixed (default 1.0) or learnable. The
      bounded-by-1 logits remove the need for TNNetSoftCapping in the
      attention head. Reuses the existing SDPA softmax-Jacobian
      backward; the only new derivative is the L2-normalisation chain.
      Numerical-gradient test on the same SeqLen=3, d=4 shape as SDPA.

- [x] TNNetPolynomialActivation — per-channel learnable
      `a[c]*x^2 + b[c]*x + c[c]`. Landed: three Depth-long learnable
      vectors (FNeurons[0..2]); default init a=0/b=1/c0=0 so it
      degrades to identity. Analytic input + weight gradients;
      registered in both CreateLayer dispatch sites. Tests:
      identity-at-init, forward formula, central-difference input
      and weight gradient checks in TestNeuralNumerical.pas.

#### Loss-function layer family — pinned at the very top of the file
under "Huber/SmoothL1/Focal/LabelSmoothing losses" but no entries
exist yet in `neuralnetwork.pas`. Sized for one commit each.

- [x] TNNetHuberLoss — smooth-L1 / L2 piecewise loss, `delta`
      configurable (default 1.0). Landed as an identity-passthrough
      output layer (Compute copies input to output so inference still
      works) whose Backpropagate clips the framework-seeded
      `(output - target)` gradient elementwise to `[-delta, +delta]`
      before propagating. delta stored in FFloatSt[0]; registered in
      both CreateLayer dispatch sites. Tests: passthrough forward,
      element-wise clipping (delta in {1.0, 0.5}, samples at well
      below / near / above the knee), Save/Load round-trip.

- [x] TNNetSmoothL1Loss — the `delta=1` special case of Huber, kept as
      a separate class for callsite readability. Landed as a
      TNNetHuberLoss subclass with parameterless `Create` that calls
      `inherited Create(1.0)`. Defaults-match test confirms its
      clipping is identical to TNNetHuberLoss(1.0).

- [ ] TNNetFocalLoss — for class-imbalanced classification:
      `FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)`, with `alpha`
      and `gamma` configurable. Wraps the existing softmax-cross-entropy
      path; gradient is the cross-entropy gradient pre-multiplied by
      the modulating factor. Numerical-gradient test with `gamma=2`.

- [ ] TNNetLabelSmoothingLoss / helper — convert a one-hot target
      volume into `(1 - eps) * one_hot + eps / NumClasses`. Pure target-
      side transformation; no gradient code needed. Pairs with a tiny
      CIFAR-stub training run that prints the convergence delta.

#### Attention / sequence-model layers I'd enjoy

- [ ] TNNetSinkAttention — prepend K learnable "attention sink" key/value
      slots that every query can attend to regardless of the causal
      mask. Implementation-wise: extend the K|V depth axis by K, mask
      everything but those K slots in the upper triangle. Helps long-
      context stability and is a ~30-line addition on top of SDPA.

- [ ] TNNetTalkingHeads — pre/post-softmax linear mix across heads
      (Shazeer et al.). A tiny learnable HxH multiply applied to
      attention logits along the head axis. Drops into the eventual
      MHA helper; standalone test on H=4, SeqLen=3.

- [ ] TNNetMultiHeadSelfAttention (taking the MHA-a/b/c thread closed
      a few batches ago and committing to ship it). Compose
      TNNetSplitChannels(H) → per-head TNNetScaledDotProductAttention
      → TNNetDeepConcat → TNNetFullConnectLinear(d_model) inside a
      single `TNNet.AddMultiHeadSelfAttention(d_model, heads)` builder.
      Numerical-gradient test on H=2, d_k=4, SeqLen=3 — every internal
      piece is already individually gradient-checked, so the new test
      pins the composition.

- [ ] TNNetCrossAttention — same SDPA core but with separate Q vs K|V
      input branches so encoder-decoder is reachable. Likely shape:
      take two prior layers explicitly via a constructor list, project
      Q from one and K|V from the other, then run SDPA on the
      concatenated depth slab. Forward + numerical-gradient on a tiny
      shape.

#### Composite / builder helpers I'd ship next

- [ ] TNNetPreNormResidual builder — listed many times now, treating
      that as universe pressure. `TNNet.AddPreNormResidual(Sublayer)`
      wires `Out = x + Sublayer(LayerNorm(x))` in one line. Concrete
      proposal: take the sublayer as a TNNet builder closure (Pascal
      flavor: a procedure-of-object that adds layers to a passed-in
      net). Tested via a "wrap an Identity sublayer ⇒ y == x" smoke
      test plus a numerical-gradient check on `Sublayer := TNNetMul`.

- [ ] TNNetGLUFeedForward block — `LayerNorm → Dense → GLU → Dense`,
      no attention dependency. Ship today using the existing TNNetGLU
      that already landed; once SwiGLU FFN ships it becomes a one-line
      variant. Concrete short example test on (d_model=8, d_ff=16).

- [ ] TNNetSwiGLUFeedForward block — same shape as above but with
      TNNetSwiGLU. Companion to the GLU variant; pinned multiple
      times. Once both ship, swap one for the other in a future
      transformer encoder block builder.

#### Examples I'd love to actually write

- [ ] `examples/BiasOnlyTuning/` — pinned earlier in the file under
      the seed-758266 batch. Freeze every learnable layer except a
      few inserted TNNetBias layers and fine-tune only those. ~80
      lines on top of an existing pre-trained CIFAR-10 model.
      Demonstrates the modern "bias-only adapter" fine-tuning pattern
      end-to-end and exercises TNNetBias on a realistic task. Verified
      no `examples/BiasOnlyTuning` directory exists yet.

- [ ] `examples/AffineFineTune/` — once TNNetMul + TNNetAffineBlock
      land above, the same pattern but freezing everything except
      the inserted Affine blocks. A teaching contrast to BiasOnlyTuning:
      "does the multiplicative half pay for the extra parameters?"
      Single chart artifact in the example's README.

- [ ] `examples/TokenShiftBaseline/` — train a tiny next-token char
      model with `TNNetEmbedding → TNNetTokenShift → MLP` and compare
      against the eventual MHA-based version on the same data. The
      "how much of attention is really just mixing-along-time?"
      experiment pinned earlier; now fully unblocked thanks to the
      TokenShift landing. ~120 lines.

- [ ] `examples/ReZeroDeepMLP/` — train a 16-layer residual MLP
      with and without TNNetReZero on each residual branch on a
      regression toy (the hypotenuse function is the natural fit).
      Print loss curves for both side by side. Concrete visualisation
      of why ReZero stabilises deep stacks at zero compute cost.
      Pure CPU, finishes in seconds. (`examples/Hypotenuse` already
      exists as a starting template; this would not duplicate it.)

- [ ] `examples/SubPixelSuperRes/` — once TNNetPixelShuffle lands,
      drop it into a 3-layer net that learns to 2x-upsample 8x8
      random checkerboards to 16x16. Smallest possible end-to-end
      use of the new layer; complements (does not replace) the
      existing `examples/SuperResolution/`.

#### Experiments I'd enjoy running

- [ ] TNNetBias-vs-TNNetMul ablation: train a small classifier four
      ways — (a) no affine, (b) bias only, (c) mul only, (d) both
      (the eventual AffineBlock) — print final accuracy and number
      of learnable params per variant. Concrete data point on whether
      the multiplicative half is worth its parameter count at toy
      scale.

- [ ] DyT-vs-LayerNorm bake-off (depends on TNNetDyT above). Re-pinned
      from earlier batches; once DyT lands this is a 30-line swap in
      the existing normalization bake-off harness.

- [ ] Causal-conv vs token-shift vs SDPA on the same toy next-token
      task. Once TNNetCausalConv1D ships, this becomes a three-bar
      chart that frames the "what's the simplest mixer that solves
      this?" question. Pure CPU.

- [ ] GRN-as-drop-in: take the existing CIFAR SimpleImage example,
      swap each `TNNetMovingStdNormalization` for a `TNNetGRN` and
      chart accuracy. ConvNeXt-V2's headline claim, reproduced at
      toy scale.

- [ ] Loss-family bake-off: same regression task (hypotenuse extended
      to noisy targets), four output layers — MSE / TNNetHuberLoss
      / TNNetSmoothL1Loss / robust median (existing utility) — print
      final test loss under increasing outlier ratios. Direct
      motivation for the Huber/SmoothL1 landings.

#### Correctness / audit work I'd take next

- [ ] TNNetDeconvolution numerical-gradient test (input + weight
      paths). Last remaining member of the upsample/deconv family
      and the most interesting because of the weight-gradient path.
      Likely needs the Double-precision SSE accumulator helper from
      the DeMaxPool fix (called out multiple times above).

- [ ] TNNetEmbedding weight-gradient test. The sparse-update pattern
      (only the embedded indices receive gradient) is the easiest
      place for a silent broadcast/reduction bug to hide. Add a
      central-difference check on a tiny vocab=4, dim=3 embedding.

- [ ] Argmax-tie behaviour test for TNNetGlobalMaxPool (already pinned
      at line ~5530). Cheap, and the documented tie-break ("first
      wins") deserves a pinning test before a future parallel-pool
      refactor silently changes it.

- [ ] LoadFromString round-trip for the entire activation menagerie.
      One parameterised test that walks every TNNetReLUBase descendant,
      builds a default instance, serialises and reloads, asserts
      Compute matches. Would catch dispatch-table gaps for any
      activation added in a hurry.

- [ ] Property-based gradient harness v0 — randomise only input shape
      across a frozen list of the six most recently landed layers
      (TNNetMul, TNNetGatedResidual, TNNetTokenShift, TNNetReZero,
      TNNetBias, TNNetChannelShuffle) and run the existing gradient
      check on each. Catches shape-edge bugs the hand-written tests
      miss; pinned multiple times already.

#### Tooling / dev experience

- [ ] `scripts/grep_layer.sh <TNNet...>` — pinned in at least five
      batches now. Print the class declaration with context, the
      Compute and Backpropagate method bodies, and every
      `tests/Test*` method mentioning the class. ~40 lines of bash;
      saves the first 30 seconds of every audit.

- [ ] `scripts/list_activations.sh` — enumerate every TNNetReLUBase
      descendant and cross-reference against `Test*Gradient` methods.
      Two-line grep + comm. Output is the actionable TODO list for
      the next activation-test batch.

- [ ] `scripts/new_layer.sh <Name>` scaffolder — pinned multiple times.
      Drops a Compute/Backpropagate skeleton into the right spot in
      neuralnetwork.pas plus a numerical-gradient test stub.

- [ ] `tests/SmokeTest.lpr` — five fastest gradient checks, exits in
      under a second. Pinned earlier; revisiting because the CI shim
      idea now feels close enough to land that a sub-second smoke
      target would justify pinning the GitHub Actions workflow YAML.

- [ ] `neural-bench` tiny CLI: time forward + backward for a chosen
      layer at a chosen shape, print ns/op. Subsumes the long-pinned
      volume-unit micro-benchmark idea and extends it to whole
      layers. CSV output so future regressions are visible.

#### Documentation I'd enjoy writing

- [ ] One-pager "loss layers in this repo" — once Huber/SmoothL1/Focal/
      LabelSmoothing land, table them alongside the existing softmax-
      cross-entropy and L2-regression paths. Becomes the index entry
      every new contributor needs to pick the right output layer.

- [ ] "What landed in the lucky-day batches" CHANGELOG entry — pinned
      multiple times. Five-plus lucky-day batches now without a
      compact public-facing summary. One bullet per landed layer with
      date + commit short SHA closes this thread for good.

- [ ] "Building a residual transformer block in this repo" walkthrough
      — once MHA and the FFN blocks above land, the encoder-block
      builder is one short page. Companion to the eventual Tiny GPT
      example.

- [ ] Inline shape annotations on TNNetScaledDotProductAttention's
      Compute and Backpropagate (pinned several batches above). Most
      algorithmically dense layer in the repo; deserves the comments
      pass even before the formal walkthrough doc.

#### Meta — what I'd most enjoy taking first

If forced to pick one task for this lucky-day session, it would be
TNNetMul + the `LayerInputAndWeightGradientCheck` helper, taken
together: TNNetMul is the smallest non-trivial learnable-weight layer
left, and shipping the helper alongside it pays dividends on every
subsequent learnable-weight landing (TNNetGatedResidual, TNNetDyT,
TNNetCausalConv1D, TNNetPolynomialActivation, the loss-layer family).
That single pair of changes unblocks roughly half of the layer ideas
in this batch.

### Lucky-day batch seed 845826 — landed
Three features shipped on this lucky-day session (suite 488 → 501,
0 failures):
- [x] TNNetSpaceToDepth + TNNetDepthToSpace (commit 30c3a11)
- [x] TNNetHuberLoss + TNNetSmoothL1Loss (commit 907657d)
- [x] TNNetPolynomialActivation (commit 79928ec)

Sized-for-one-commit follow-ups now unblocked or made easier by the
above:
- [ ] TNNetFocalLoss — wraps the same softmax-CE path the Huber/SmoothL1
      heads now establish a passthrough+clip pattern for; can reuse the
      identity-passthrough output-layer pattern from TNNetHuberLoss.
      Constructor params (alpha, gamma) in FFloatSt[0..1]. Adds the
      `gamma=2` numerical-gradient test pinned multiple times.
- [ ] TNNetLabelSmoothingLoss helper — pure target-side `(1 - eps) *
      one_hot + eps / NumClasses`; no gradient code. Pairs naturally
      with a Focal-loss test that re-uses it.
- [ ] TNNetPixelShuffle — companion of the SpaceToDepth/DepthToSpace
      pair that just landed; reuse the same divisibility-check and
      inverse-permutation pattern in neuralnetwork.pas. Numerical-
      gradient test is one line of variation away from the
      SpaceToDepth one.
- [ ] examples/SubPixelSuperRes — drop the new TNNetPixelShuffle (or,
      provisionally, DepthToSpace) into a 3-layer net that learns to
      2x-upsample 8x8 random checkerboards to 16x16. Smallest possible
      end-to-end use of the inverse-permutation layers.
- [ ] Polynomial-activation bake-off micro-experiment: swap each
      TNNetReLU in a small MLP for a TNNetPolynomialActivation,
      compare final loss on the existing hypotenuse toy. Concrete
      "is a learned activation worth its 3*Depth params?" data point.
- [ ] Loss-family bake-off (re-pinned, now actionable): hypotenuse
      with added outliers, four heads — MSE / TNNetHuberLoss(delta=1)
      / TNNetSmoothL1Loss / robust median — print final test loss
      under increasing outlier ratios. Direct motivation for the
      Huber/SmoothL1 landings.
- [ ] Document the three newly-landed layers in README.md (matching
      the existing one-line + snippet style): TNNetSpaceToDepth /
      TNNetDepthToSpace under a new "Shape / Reshape" or
      "Patchify" subsection; TNNetHuberLoss / TNNetSmoothL1Loss
      under a "Robust regression losses" entry; TNNetPolynomialActivation
      under the activation reference. ~30 lines total.

### Lucky-day batch — 2026-05-15 (seed 488161)

Fresh wave. The recent batches keep pinning attention/MHA, FFN
builders, the loss-layer family, and the shared gradient-check
helper — those are well-covered above and I won't re-pin them. Below
are ideas I'd actually enjoy taking that don't already appear on the
list and that I verified absent from `neural/neuralnetwork.pas` at
the time of writing.

Verified absent as of seed 488161: TNNetSEBlock /
TNNetSqueezeExcitation, TNNetCBAM, TNNetFiLM, TNNetCoordConv,
TNNetMaxBlurPool, TNNetSpectralNorm (as a wrapper layer),
TNNetShakeShake, TNNetShakeDrop, TNNetSinhAct / TNNetCosh,
TNNetGumbelSoftmax, TNNetSinusoidalTimeEmbedding (the diffusion-style
1D timestep encoder — distinct from the existing sequence-axis
TNNetSinusoidalPositionalEmbedding), TNNetCenterLoss, TNNetTripletLoss,
TNNetContrastiveLoss / InfoNCE, TNNetVectorQuantizer / VQ-VAE bottleneck,
TNNetEntropyRegularizer, TNNetHopfieldUpdate (modern Hopfield retrieve),
TNNetInvertedResidual / MBConv builder, TNNetGhostModule,
TNNetSplineActivation (a tiny KAN-flavored learnable activation),
TNNetMaxBlurPool, TNNetAntiAliasedPool. The closest near-collisions
I double-checked and confirmed are *different*: TNNetCosineSimilarity
exists as a similarity-score layer but no SE/CBAM channel-attention
block exists; TNNetSinusoidalPositionalEmbedding exists for sequence
positions but no scalar-timestep variant; TNNetSqrt / TNNetSquare /
TNNetSign-ish activations exist but no sinh/cosh; TNNetSoftCapping
exists but no Lipschitz / spectral-norm wrapper.

#### Layers I'd enjoy building (channel-attention / conditioning)

- [ ] TNNetSEBlock (Squeeze-and-Excitation) — global-avg-pool over
      (SizeX, SizeY), then two FullConnect layers (Depth → Depth/r →
      Depth) with ReLU + Sigmoid producing per-channel multipliers,
      finally broadcast-multiply against the original input. Ships as
      a builder `TNNet.AddSqueezeExcitation(Reduction=16)` composed
      from existing layers (TNNetGlobalAvgPool, TNNetFullConnectReLU,
      TNNetFullConnectSigmoid, TNNetChannelMulByLayer). One end-to-end
      numerical-gradient test on a tiny (4,4,8) input pins the
      composition; no new low-level Compute/Backpropagate needed
      because every constituent already has gradient coverage.

- [ ] TNNetCBAM (Convolutional Block Attention Module) — the SE block
      above plus a spatial-attention sibling: max-pool + avg-pool over
      the channel axis, concatenate to 2 channels, 7x7 conv to 1
      channel, sigmoid, broadcast-multiply against the input. Same
      builder-from-existing-layers approach. Once SE lands, CBAM is a
      ~20-line addition on top.

- [ ] TNNetFiLM (Feature-wise Linear Modulation) — `y = gamma * x +
      beta` where gamma and beta come from a separate conditioning
      input branch (not learned per-channel weights). Constructor
      takes two prior layers (the feature stream and the gamma|beta
      stream); the gamma|beta stream's depth must equal 2 * feature
      depth. Forward broadcasts the two halves over (SizeX, SizeY).
      Backward splits the incoming gradient and forwards it to both
      branches with the standard product/sum chain. Unblocks
      conditional generative experiments and class-conditional batch
      norm replacements.

- [ ] TNNetCoordConv — concatenate two extra channels containing the
      normalized X and Y coordinates of each pixel before a convolution.
      Parameter-free helper layer; forward is a deterministic
      coordinate fill, backward is the identity on the first Depth
      channels and zero on the appended coordinate channels (gradient
      doesn't flow back into the fixed coords). One numerical-gradient
      test on the kept channels; a separate forward-equality test for
      the coordinate-channel values at a few sample pixels.

#### Regularization / pooling layers

- [ ] TNNetMaxBlurPool — anti-aliased max-pool: max-pool first, then
      blur with a fixed binomial filter (no learnable weights), then
      stride. Improves shift-equivariance of CNNs at zero parameter
      cost. Forward = existing TNNetMaxPool composed with a fixed
      depthwise convolution; backward is the chain through both. Ships
      as a builder on top of existing primitives if the fixed-weight
      blur can be expressed as a TNNetDepthwiseConvLinear initialized
      with the binomial kernel and L2/optimizer disabled (similar
      pattern to the existing TNNetIdentityWithoutL2AndOptimizer).

- [ ] TNNetShakeShake — for two parallel residual branches A and B,
      forward outputs `alpha * A + (1 - alpha) * B` with `alpha` drawn
      uniformly per-sample at training, and `0.5` at inference.
      Backward uses an *independent* `beta ~ U[0,1]` for the gradient
      mix (the headline trick of the paper). Seeded-RNG numerical-
      gradient test pinned to a fixed seed.

- [ ] TNNetShakeDrop — generalization of stochastic depth and
      ShakeShake on a single residual branch: forward output is
      `x + (bernoulli(p) + alpha * (1 - bernoulli(p))) * F(x)` with
      uniform `alpha` and `beta` perturbations per-sample. Same RNG
      pattern as TNNetDropPath; the new ingredient is the
      multiplicative perturbation.

- [ ] TNNetSpectralNorm wrapper — wraps an existing FullConnect or
      Convolution layer and divides its weight matrix by its largest
      singular value (estimated via one power-iteration step per
      forward pass). Stores the iterate `u` as a non-trainable buffer.
      Backward is the standard chain through the division. Common
      regularizer in GANs; would pair nicely with the existing
      `examples/VisualGAN`.

- [ ] TNNetGumbelSoftmax — differentiable categorical sampling:
      `softmax((logits + g) / tau)` where `g ~ Gumbel(0,1)`. Two
      modes — "soft" (used by gradients) and "hard" (straight-through
      argmax at the forward pass). Backward is the softmax-Jacobian
      already implemented for TNNetSoftMax. Test with seeded RNG and
      tau=1.0.

#### Loss / output layers

- [ ] TNNetTripletLoss — accepts a single input volume of depth
      `3 * d_embed` interpreted as anchor | positive | negative
      embeddings concatenated along the depth axis. Forward returns
      the scalar `max(0, ||a-p||^2 - ||a-n||^2 + margin)`. Backward
      is the closed-form derivative through the L2 squared norms with
      the active-margin gate. One numerical-gradient test with the
      margin clearly active and one with it inactive (zero-gradient
      case).

- [ ] TNNetContrastiveLoss / InfoNCE — input of depth `2 * d_embed`
      split into query and key. Computes the InfoNCE loss against the
      other samples in the minibatch as negatives. Tricky bit: needs
      access to the full batch at backward, so it fits naturally as
      a `TNNetLayer` whose Compute operates on the batched
      `FOutputErrorDeriv` rather than per-sample. Numerical-gradient
      test on a tiny batch of 4, d_embed=2.

- [ ] TNNetCenterLoss — auxiliary loss `0.5 * ||x - c_y||^2` where
      `c_y` is a per-class learnable centroid. Stores the centroid
      matrix `(NumClasses x d_embed)` as a normal weight matrix.
      Forward computes the per-sample distance to its class centroid;
      backward updates both the embedding and the corresponding
      centroid row. Numerical-gradient test on input + the centroid
      weights.

- [ ] TNNetEntropyRegularizer — passthrough output layer (Compute is
      the identity) that adds `-lambda * sum(p * log(p))` of its
      input's softmax to the gradient at Backpropagate. Encourages
      confident outputs; the negative-lambda variant encourages
      uncertain outputs (useful for RL exploration). One closed-form
      gradient test.

- [ ] TNNetVectorQuantizer (VQ-VAE bottleneck) — codebook of K vectors,
      forward replaces each input vector with its nearest codebook
      entry under the straight-through estimator (gradient flows
      through the assignment as identity). Tracks a commitment loss
      `beta * ||sg[ze] - e||^2` and a codebook loss
      `||ze - sg[e]||^2` accumulated into the backward pass. Tests:
      forward assignment correctness on a hand-rolled codebook,
      numerical-gradient on the STE path with a loosened ~1e-1
      tolerance (standard for STE tests).

#### Activations

- [x] TNNetSinhAct — `y = sinh(x)`, backward `cosh(x) * dy`. Trivial
      element-wise activation; rounds out the trig/hyperbolic family
      alongside the existing TNNetSin/TNNetCos and TNNetHyperbolicTangent.
      Numerical-gradient test only (no weights).
      Landed: TNNetSinhAct subclasses TNNetReLUBase, caches Cosh(x) into
      FOutputErrorDeriv mirroring TNNetSin/TNNetCos, registered in both
      CreateLayer dispatch tables, covered by TestSinhActGradientCheck.

- [ ] TNNetSplineActivation — per-channel learnable piecewise-linear
      activation defined by K+1 control points at fixed knots
      `[-c, -c + 2c/K, ..., +c]`. KAN-flavored but kept tiny: K
      defaults to 4, so a Depth-channel layer has 5*Depth params.
      Backward distributes gradient between the two knots flanking
      each sample. Numerical-gradient test on input + weights.

- [x] TNNetMishlike / TNNetSerf — `y = x * erf(softplus(x))`, the Serf
      activation (Nag, Bhattacharyya 2021). Drop-in for Mish with a
      slightly smoother profile. Tiny element-wise addition to the
      TNNetReLUBase family; numerical-gradient test as usual.
      Landed as TNNetSerf: caches derivative
      `erf(sp) + x * (2/sqrt(pi)) * exp(-sp^2) * sigmoid(x)` into
      FOutputErrorDeriv with stable-softplus guards mirroring
      TNNetSoftPlus / TNNetMish; erf via Abramowitz & Stegun 7.1.26
      since FPC's Math unit ships no erf. Registered in both CreateLayer
      dispatch tables, covered by TestSerfGradientCheck.

#### Examples I'd love to actually write

- [ ] `examples/SEBlockCifar/` — drop one SE block per stage into the
      existing `examples/SimpleImageClassifier` CIFAR-10 net and
      print the before/after validation accuracy delta. The smallest
      end-to-end demonstration of channel-attention on a real task,
      and a natural follow-on the moment TNNetSEBlock lands. Verified
      no `examples/SEBlockCifar` directory exists yet.

- [ ] `examples/FiLMConditional/` — toy "draw a digit of class C"
      generator: a small decoder net whose first two conv blocks are
      FiLM-conditioned on a 10-way one-hot class input. Trains in
      well under a minute on CPU. Shows off the new TNNetFiLM layer
      end-to-end and the way two input branches compose. Pairs with
      the existing `examples/VisualGAN` as a teaching contrast.

- [ ] `examples/TripletEmbedding/` — learn a 2D embedding of MNIST
      digits using TNNetTripletLoss; output a PGM scatter plot of
      the learned embeddings colored by class. Pure CPU; concrete
      demo of the new loss + a visible result artifact.

- [ ] `examples/VQAutoencoder/` — extend the existing
      `examples/VisualAutoencoder` with a TNNetVectorQuantizer
      bottleneck. Print before/after reconstruction PSNR and the
      number of active codebook entries. Verified no
      `examples/VQAutoencoder` directory exists yet.

- [ ] `examples/CoordConvSpiral/` — minimal CoordConv vs plain conv
      comparison on the "predict (x, y) from a one-hot pixel image"
      task from the original CoordConv paper. Two-line difference
      between the two networks, dramatic accuracy gap on a 2-minute
      training run.

- [ ] `examples/AntiAliasedMaxPool/` — train the same tiny CIFAR-10
      net once with TNNetMaxPool and once with TNNetMaxBlurPool;
      report shift-equivariance (run inference on the validation
      set, then on a one-pixel-shifted copy, and print the prediction
      agreement rate). The headline claim of the MaxBlurPool paper
      reproduced in one short artifact.

#### Experiments I'd enjoy running

- [ ] Channel-attention bake-off: a fixed tiny CIFAR backbone,
      identical seed, four variants — (a) no attention, (b) SE block,
      (c) CBAM, (d) hand-rolled "1x1 conv + sigmoid" baseline. One
      validation-accuracy bar chart + a CSV. The point: how much of
      SE/CBAM's gain is "any channel reweighting" vs the specific
      bottleneck architecture?

- [ ] FiLM-vs-concat conditioning bake-off: the same class-conditional
      MNIST decoder, once with FiLM conditioning and once with
      class-one-hot concatenated to the input. Print final
      reconstruction loss and a side-by-side sample grid. Concrete
      data point on whether FiLM's parameter efficiency translates
      to a sample-quality gap at toy scale.

- [ ] Activation family follow-up: the existing menagerie bake-off
      pinned several batches above doesn't include Serf or the new
      sinh-family activations on this batch's list. Once they land,
      add them as two new rows to the same table — no new harness
      needed.

- [ ] VQ codebook collapse stress test: train the VQ-autoencoder
      above with K in {16, 64, 256} and a few different commitment-loss
      weights, print the per-run number of active codebook entries
      after training. Reproduces the well-known "VQ-VAEs love to
      collapse" failure mode without dragging in any heavy
      reconstruction metric.

#### Correctness / audit work I'd take

- [ ] TNNetDotProducts numerical-gradient test — pinned implicitly
      (the SDPA layer subsumes its usage) but the standalone class
      still ships and its weight-gradient path looks like the kind of
      place a silent bug could live. One input + one weight
      central-difference check on a tiny (2, 1, 4) input.

- [ ] TNNetLocalConnect / TNNetDeLocalConnect input + weight gradient
      tests. Older locally-connected family that lacks numerical-
      gradient coverage in `TestNeuralNumerical.pas`. One test pair
      per direction; would have caught the historical bugs the
      activation audit turned up if applied earlier.

- [ ] TNNetMaxPoolWithPosition correctness check: the auxiliary
      "position channels" it emits should round-trip through
      TNNetDeMaxPool to exactly reconstruct the upsample pattern of
      the forward pass. Pure forward-equality test, no gradient
      needed. Pins an invariant that the SuperResolution example
      already depends on but no test currently enforces.

- [ ] TNNetSinusoidalPositionalEmbedding boundary test: assert that
      the embedding values match the closed-form formula at three
      specific (pos, dim) points. The layer ships, the gradient
      passes through as identity, but no forward-value test currently
      pins the formula constants — a refactor could silently change
      them.

#### Tooling / dev experience

- [ ] `scripts/check_layer_dispatch.sh` — grep every
      `TNNet... = class` line in `neural/neuralnetwork.pas`,
      cross-reference against the two `CreateLayer` dispatch tables
      and the `LoadFromString` cascade, and print any class that's
      missing from any of the three. Catches the "I added a layer
      but forgot the LoadFromString branch" bug in one line of bash.

- [ ] `scripts/coverage_gradient_tests.sh` — list every layer class
      whose Backpropagate is overridden (i.e. not `inherited`) and
      cross-reference against `Test*Gradient*` methods in
      `tests/TestNeuralNumerical.pas`. Output the actionable TODO list
      for the next audit batch; pairs with the
      `scripts/list_activations.sh` pinned earlier.

- [ ] A tiny `docs/layer_taxonomy.md` auto-generator that walks the
      class hierarchy of `TNNetLayer` in `neuralnetwork.pas` and
      emits a tree. The current set of ~150 layers is large enough
      that the README's flat list is hard to navigate; a tree view
      organized by base class would land as a single CI artifact.

#### Documentation

- [ ] `docs/channel_attention.md` — once SE / CBAM / FiLM land, one
      page comparing them on the same axes: what they multiply, what
      conditioning they require, parameter cost vs depth. Becomes the
      reference companion to the channel-attention bake-off above.

- [ ] `docs/loss_layers.md` — once Huber / SmoothL1 / Focal /
      LabelSmoothing / Triplet / Contrastive / Center / Entropy /
      VQ losses are all available, table them on one page: input
      shape required, scalar vs per-sample, whether the layer is a
      drop-in for softmax-cross-entropy or an auxiliary loss. Pinned
      under loss-family docs earlier; this batch sharpens the scope
      to "an actual table once the family is complete".

- [ ] Short note in `tests/README` (or, if absent, a new one) on
      "how to add a numerical-gradient test in three lines" — a
      cookbook for the shared `LayerInputAndWeightGradientCheck`
      helper pinned multiple batches above. Lands the moment that
      helper does.

#### Stretch / ambitious

- [ ] `examples/TinyDiffusion/` — a 20-step denoising-diffusion model
      on 8x8 grayscale MNIST patches using a tiny FiLM-conditioned
      U-Net with TNNetSinusoidalTimeEmbedding for the timestep. Pure
      CPU, generates recognizable-ish digit shapes after a few
      minutes. The first generative-modeling example on the project
      that isn't a GAN; depends on FiLM + the timestep embedding
      landings above.

- [ ] `examples/HopfieldRetrieval/` — modern Hopfield network as
      attention (Ramsauer et al.): store K patterns as the K rows of
      a `(K, d)` matrix, retrieve via a single softmax-attention step
      against a query. Tiny, no training loop needed (it's a
      closed-form retrieval), and demonstrates the
      attention-as-associative-memory framing in one short program.

- [ ] Mixed-precision experiment — re-pin from earlier in this file
      but with a concrete first step: add a `TNeuralFloat16 = packed
      record ...` type in `neuralvolume.pas` with conversion helpers
      to/from `TNeuralFloat`, plus a one-layer forward-only test that
      validates the FP16 path matches FP32 to within 1e-2. Doesn't
      require any optimizer changes yet — pure inference-side
      groundwork.

#### Meta — what I'd most enjoy taking first

If I had to pick one task this lucky-day session it would be
TNNetSEBlock as a *builder over existing layers* (no new
Compute/Backpropagate), paired with the `examples/SEBlockCifar/`
demo. The builder is ~15 lines of `AddLayer` calls, every constituent
already has gradient coverage, and the example lands a visible
accuracy delta on the most-loved existing dataset in the repo. Two
small commits, one clear story, and unblocks both CBAM and the
channel-attention bake-off as natural follow-ups.

### Lucky-day batch — 2026-05-15 (seed 732439)

Landed this session:
- TNNetSinhAct (`y = sinh(x)`, derivative cosh, cached into
  FOutputErrorDeriv via TNNetReLUBase) — commit b6dd73a.
- TNNetSerf / TNNetMishlike (`y = x * erf(softplus(x))`, erf via
  Abramowitz & Stegun 7.1.26 since FPC's Math has no erf) —
  commit 81d8676.

Full test suite (503/503) green after each commit.

#### Small follow-ups pinned from this batch

- [ ] Hyperbolic-family bake-off micro-experiment: same toy MLP
      trained with TNNetHyperbolicTangent / TNNetSinhAct /
      TNNetCos / TNNetSinc / TNNetLisht / TNNetBentIdentity and a
      one-table loss/epoch comparison. Sinh is the only unbounded
      member of the trig/hyperbolic family in the repo, so the
      comparison should highlight where unbounded-vs-bounded
      matters. Pure-CPU, runs in seconds.

- [ ] Promote the private `SerfErf` polynomial in
      neural/neuralnetwork.pas to a small shared helper (probably
      in neuralvolume.pas) so future erf-based losses or
      activations (e.g. GELU-exact, Gaussian-loss variants) can
      reuse it. Add a tiny accuracy regression test pinning the
      A&S 7.1.26 |err| < 1.5e-7 bound.

- [ ] TNNetSerf-vs-TNNetMish ablation: same tiny classifier
      trained four times (ReLU / Mish / Serf / GELU) on
      CIFAR-10 or a synthetic toy dataset, report final loss and
      epochs-to-converge. Concrete payoff for the Mishlike
      landing.

### Lucky-day batch — 2026-05-15 (seed 110681)

Fresh batch picked after auditing the file and the source for prior
mentions. Verified absent from both `neural/neuralnetwork.pas` and
the existing tasklist.md at the time of writing: TNNetArcSinh /
TNNetAsinh, TNNetSparsemax, TNNetSoftMaxOne (softmax with a +1 in
the denominator — "attention sink" alternative), TNNetLeCunTanh
(`1.7159 * tanh(2/3 * x)`), TNNetCharbonnierLoss, TNNetTanhGLU
(the missing tanh sibling of GLU/ReGLU/GEGLU/SwiGLU — GLU exists
with sigmoid, ReGLU exists with ReLU, GEGLU exists with GELU,
SwiGLU exists with Swish, but no tanh-gated variant ships),
TNNetLogCoshActivation (activation form — the LogCosh *loss* is
already pinned several batches up, the activation form is not).

#### Activations I'd enjoy building

- [x] TNNetArcSinh — `y = arcsinh(x) = ln(x + sqrt(x^2 + 1))`,
      derivative `1 / sqrt(x^2 + 1)`. Monotonic, smooth, unbounded,
      but grows like `ln|x|` for large |x| — a tanh sibling that
      never saturates and a sinh sibling that never explodes.
      Element-wise activation under TNNetReLUBase; cache the
      derivative into FOutputErrorDeriv mirroring the existing
      sinh/cos/sin pattern. Numerical-gradient test only (no
      weights). Pairs naturally with the hyperbolic-family bake-off
      pinned in the seed-732439 batch above.

- [x] TNNetLeCunTanh — `y = 1.7159 * tanh(2/3 * x)`, the
      classical scaled-tanh activation from LeCun et al.'s
      "Efficient Backprop" tuned so f(±1)=±1 and f'(0)≈1.14.
      Cheap follow-on to TNNetHyperbolicTangent that just plugs
      the scale constants into Compute/Backpropagate. Forward +
      numerical-gradient test plus a value-at-1 pin
      (asserts f(1) ≈ 1.0 within 1e-3).

- [x] TNNetLogCoshActivation — `y = log(cosh(x))`, derivative
      `tanh(x)`. Smooth, symmetric, ≈ x²/2 near 0 and ≈ |x| for
      large |x| — a smooth-L1 *activation* (the loss variant is
      already pinned). Element-wise under TNNetReLUBase; needs a
      numerically stable formulation for large |x| (use
      `|x| + log1p(exp(-2|x|)) - log(2)` mirroring the
      stable-softplus trick in TNNetSoftPlus / TNNetMish / TNNetSerf).
      One forward + numerical-gradient test plus a stability test
      at x = ±30.

- [ ] TNNetTanhGLU — the missing tanh sibling of the existing
      GLU/ReGLU/GEGLU/SwiGLU family. Splits input along the depth
      axis into A | B, outputs `A * tanh(B)`, output depth =
      input depth / 2. Parameter-free; numerical-gradient test
      mirroring the existing TNNetGLU one with a one-line swap
      of the gating function.

#### Probability projections I'd enjoy building

- [ ] TNNetSparsemax — Martins & Astudillo's exact-sparse
      alternative to softmax: `y = max(x - tau, 0)` where `tau`
      is chosen per-sample so the output sums to 1. Yields true
      zeros (not just small numbers) and is the natural softmax
      drop-in for sparse attention. Forward uses the O(d log d)
      sort-based algorithm from the paper (the O(d) randomized
      version is a follow-up); backward is the closed-form
      Jacobian `dy = ds - (sum(ds[support]) / |support|)` on the
      support set (zero elsewhere). Numerical-gradient test on a
      (Depth=8, 1, 1) volume with a deliberately sparse support.

- [ ] TNNetSoftMaxOne — `y_i = exp(x_i) / (1 + sum_j exp(x_j))`,
      i.e. softmax with a constant +1 in the denominator. Outputs
      do not sum to 1 by design — the model can "attend to
      nothing" (Miller, 2023). Cheaper alternative to
      TNNetSinkAttention pinned above: same anti-attention-spike
      property, no new attention-side plumbing. Forward is one
      line off existing softmax; backward is the same Jacobian
      computation with the modified denominator. Numerical-
      gradient test on a Depth=4 volume and a "y.Sum() < 1" pin.

#### Losses I'd enjoy building

- [ ] TNNetCharbonnierLoss — `sqrt((output - target)^2 + eps^2) -
      eps`, the "smooth-L1" loss favored in super-resolution
      papers. Drops straight into the existing
      `examples/SuperResolution` as a one-line swap from MSE.
      Lands as a TNNetHuberLoss-style passthrough output layer
      that clips/reshapes the framework-seeded `(output - target)`
      gradient before propagating: gradient is
      `(output - target) / sqrt((output - target)^2 + eps^2)` —
      always bounded in [-1, 1] regardless of residual size, so
      it's a robust regression head without Huber's piecewise
      knee. `eps` defaults to 1e-3; stored in FFloatSt[0]. Tests:
      passthrough forward, gradient sanity at three residual
      scales, Save/Load round-trip.

#### Examples I'd love to actually write

- [x] `examples/HyperbolicActivationBakeOff/` — landed (commit bc50efc).
      Trains a tiny `2 -> 32 -> 1` MLP on the hypotenuse toy with each
      of TNNetHyperbolicTangent / TNNetLeCunTanh / TNNetSinhAct /
      TNNetArcSinh / TNNetLisht / TNNetBentIdentity / TNNetTanhExp /
      TNNetLogCoshActivation and prints a CSV of final MSE (in
      original target units) and epochs-to-converge. Inputs/targets
      normalized to `[0,1]` so TNNetSinhAct does not overflow.
      Sample run: TNNetLisht wins (1.70 MSE, converged epoch 32),
      TNNetTanhExp second (2.26, epoch 85), TNNetLogCoshActivation
      crosses the threshold late (5.08, epoch 148), saturating tanh
      family stalls. Full run ~42 s on CPU.

- [ ] `examples/CharbonnierSR/` — minimal variant of the existing
      `examples/SuperResolution` that swaps the MSE head for
      TNNetCharbonnierLoss and prints the before/after PSNR on
      the same eval split. The headline use case for the loss;
      one new short main + a shared training helper between the
      two examples. Verified no `examples/CharbonnierSR` directory
      exists.

- [ ] `examples/SparseAttentionDemo/` — toy
      "predict the next character of a periodic sequence" task
      using TNNetSparsemax in place of softmax over a tiny K|V
      bank. Print the attention-weight histogram per step and
      confirm the sparsity (≥ half of the weights exactly zero).
      Visible demonstration of why "exact zeros" matter for
      attention interpretability. Verified no
      `examples/SparseAttentionDemo` directory exists.

#### Experiments I'd enjoy running

- [ ] Softmax-vs-SoftmaxOne-vs-Sparsemax bake-off on the same
      tiny SDPA test: same Q,K,V tensors, three output
      distributions, print the entropy and max-weight of each.
      No training loop needed — pure forward-pass comparison.
      Lands the moment SoftMaxOne and Sparsemax do.

- [ ] LeCun-tanh-vs-tanh ablation: same tiny MLP on the existing
      hypotenuse toy, two runs differing only in the output-of-
      hidden activation. Reproduces the "scaled tanh trains
      faster than plain tanh" claim from "Efficient Backprop"
      at toy scale. ~30 lines on top of the existing example.

- [ ] Charbonnier-vs-Huber-vs-MSE-vs-LogCosh head-to-head on the
      noisy-hypotenuse harness already pinned in the loss-family
      bake-off above. Adds Charbonnier as a fourth row; nothing
      else changes. Closes the smooth-L1 family comparison
      thread once and for all.

#### Correctness / audit work I'd take

- [ ] TNNetMaxOut numerical-gradient test (input + weight paths).
      The class exists at line 992 of `neuralnetwork.pas`,
      stores `FMaxPosArray` for the backward route, and lacks
      gradient coverage in `TestNeuralNumerical.pas` (verified
      by grep). Exactly the kind of "max-position cache" path
      where a silent off-by-one would not surface in forward
      tests. Tiny Depth=4, Units=2 test mirrors the
      TNNetGlobalMaxPool tie-break test pattern.

- [ ] TNNetDigital forward-equality test — round every input
      cell to 0 or 1 against a fixed threshold; pin the
      threshold and the output for three inputs straddling it.
      Pure forward; no gradient (it's non-differentiable). The
      layer ships at line 529 with no coverage I can find.

- [ ] TNNetReZero "starts as identity" forward test — at
      initialization the per-channel scale is zero, so
      `Out = x + 0 * Sublayer(x)` should equal `x` exactly. Pin
      the invariant before any future init-policy change can
      silently break it. The layer ships at line 1741.

#### Tooling / dev experience

- [ ] `scripts/plot_activation.sh <TNNet...>` — sample the named
      activation at 41 points in `[-5, +5]`, print the forward
      values and numerical derivative as a tiny side-by-side
      ASCII chart. Quick visual sanity for any newly added
      activation; pairs with the activation-family taxonomy
      doc below. ~60 lines of bash + a one-shot Pascal helper
      that takes the class name on argv.

- [ ] `tests/TestActivationMenagerie.pas` smoke test — one
      parameterless test that walks every TNNetReLUBase
      descendant (the list comes from `scripts/list_activations.sh`
      pinned several batches above) and confirms forward returns
      finite values on a (-5, +5)-spanning input. Sub-second;
      lands the moment the activation list helper does.

#### Documentation I'd enjoy writing

- [ ] `docs/activation_taxonomy.md` — organise the ~50
      activations now in the repo by mathematical family
      (rectified linear / smooth-rectified / saturating /
      hyperbolic / trig / polynomial / gated / learnable),
      with the closed-form expression and derivative for each.
      One page, becomes the index entry every new contributor
      needs. The flat README list is becoming hard to navigate
      now that the family is this large.

- [x] README one-line entries for the just-landed TNNetSinhAct
      and TNNetSerf (commits b6dd73a, 81d8676) in the activation
      reference. Done in commit 01c595e together with the seed-623401
      batch (TNNetArcSinh, TNNetLeCunTanh, TNNetLogCoshActivation)
      and TNNetTanhExp — six rows added in one doc pass.

#### Meta — what I'd most enjoy taking first

If I had to pick one task this lucky-day session it would be
TNNetArcSinh + the `examples/HyperbolicActivationBakeOff/` demo,
taken together. The activation is ~15 lines of new code that
follows the just-landed TNNetSinhAct pattern exactly (FOutputErrorDeriv
cache of `1/sqrt(x^2+1)`, no weights, one numerical-gradient
test). The example then includes it as one row in a table that
also re-uses TNNetSinhAct + TNNetSerf — a single short commit
that lands a new activation and closes the hyperbolic-family
bake-off pinned in the seed-732439 batch above. Two small commits,
one visible artifact, follow-on payoff for both prior lucky-day
landings.



### Lucky-day batch — 2026-05-15 (seed 623401)

Landed this session (three serial opus agents, three commits):
- TNNetArcSinh (`y = arcsinh(x)`, derivative `1/sqrt(x^2+1)`) — commit 10533e2.
- TNNetLeCunTanh (`y = 1.7159 * tanh(2/3 * x)`, classical
  scaled-tanh from LeCun's "Efficient Backprop") — commit 4fb1de1.
- TNNetLogCoshActivation (`y = log(cosh(x))`, stable formulation
  `|x| + log1p(exp(-2|x|)) - ln(2)`, derivative `tanh(x)`) —
  commit 3fbff4d.

Test suite grew from 503 to 508/508 green (one gradient test each
plus a value-at-1 pin for LeCunTanh and a |x|=30 stability pin for
LogCoshActivation).

#### Small follow-ups pinned from this batch

- [x] `examples/HyperbolicActivationBakeOff/` — landed in commit
      bc50efc (see entry above).

- [x] README one-line entries for TNNetArcSinh, TNNetLeCunTanh and
      TNNetLogCoshActivation in the activation reference — landed
      in commit 01c595e alongside TNNetSinhAct, TNNetSerf and
      TNNetTanhExp.

- [x] TNNetLeCunTanh-vs-TNNetHyperbolicTangent ablation on the
      hypotenuse toy — now subsumed by the
      `examples/HyperbolicActivationBakeOff/` table, which includes
      both as rows trained on the same data with the same seed.
      On the published 150-epoch run, plain Tanh finishes at MSE
      35.67 and LeCunTanh at 50.07; both fail to converge, so the
      "scaled tanh trains faster" claim is NOT reproduced on this
      particular harness/seed. Worth re-running with a sweep over
      seeds and learning rate before drawing a conclusion.

- [ ] TNNetLogCoshActivation as a hidden-layer activation vs a
      log-cosh loss as the output head: blocked — TNNetLogCoshLoss
      does NOT exist in the repo yet (grep finds only
      TNNetLogCoshActivation). To unblock: add a TNNetLogCoshLoss
      output layer in the TNNetHuberLoss-shape (forward
      passthrough, backward clipped gradient `tanh(output-target)`),
      mirror TNNetHuberLoss's Save/Load + LoadFromString dispatch,
      and add a passthrough test + 3-residual-scale gradient
      sanity test. Then the dual experiment becomes a one-line
      swap on top of the existing Hypotenuse example.

### Lucky-day batch — 2026-05-15 (seed 431337)

Landed this session (two serial opus agents, two commits):
- README activation-reference rows for TNNetArcSinh, TNNetLeCunTanh,
  TNNetLogCoshActivation, TNNetSinhAct, TNNetSerf and TNNetTanhExp —
  commit 01c595e.
- `examples/HyperbolicActivationBakeOff/` — 8-activation bake-off on
  the hypotenuse toy with normalized inputs, fixed `RandSeed=42`,
  150 epochs at LR=0.01, prints CSV of MSE in original target units
  and first epoch each activation crosses MSE<5. Total ~42 s on CPU.
  Sample table copied into the example's README.md. Commit bc50efc.

Test suite remains green at 508/508 — both commits are
pure-additions (docs + a new example) with no changes to library
code or existing tests.

#### Small follow-ups pinned from this batch

- [ ] Add TNNetLogCoshLoss as a TNNetHuberLoss-style output head
      (see blocked dual-experiment item above for the shape) so
      the log-cosh-vs-log-cosh dual experiment is unblocked.
      Also makes log-cosh available as a robust regression loss
      paired with the recently-landed TNNetLogCoshActivation
      hidden-layer form.

- [ ] Reproduce the LeCunTanh-vs-Tanh ablation with a small
      seed/LR sweep on top of `examples/HyperbolicActivationBakeOff/`
      — current 150-epoch / LR=0.01 / seed=42 run does not show
      the scaled-tanh advantage and both stall. Either the toy is
      too easy (linear regression bites both into the same basin)
      or the LR is wrong for the saturating regime; worth a
      one-screen follow-up that averages over 5 seeds and reports
      mean+std for each row.

- [ ] Consider a `HasFlipX`/`HasFlipY` audit on the bake-off
      example — currently disabled there because they corrupt
      pure-regression data; document the reason inline so future
      copy-pasters don't re-enable them by accident.
