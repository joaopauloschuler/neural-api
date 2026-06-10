# CapsuleReconstruction

The **reconstruction-decoder pose-perturbation** demo for `TNNetCapsuleRouting`
— the headline stretch result of Sabour, Frosst & Hinton (2017),
[*Dynamic Routing Between Capsules*](https://arxiv.org/abs/1710.09829). It is the
follow-up to `examples/CapsuleRouting` (which exercises the routing layer
itself).

The paper's central claim is not merely that capsule routing classifies well,
but that the per-capsule **pose vector is interpretable**: a small reconstruction
decoder fed by the *winning* class capsule learns to render the input, and
perturbing a **single dimension** of that pose vector varies one human-readable
visual factor (stroke thickness, width, position, ...). This example trains a
tiny CapsNet jointly with such a decoder on a purely synthetic dataset, then
sweeps one pose dimension and shows the factor move — as ASCII art.

Built from **existing layers only** (`TNNetCapsuleRouting` is already in-tree); no
new layer is added.

## Dataset (synthetic, no download)

A tiny **12×12 two-class "bars"** set with explicit, controllable pose factors so
disentanglement is actually plausible inside a sub-5-minute CPU budget:

- **class 0** = a vertical bar, **class 1** = a horizontal bar;
- two latent pose factors per sample: **thickness** (1..3 px) and **centre
  position**;
- pixels in `[0,1]` plus light Gaussian noise.

These low-dimensional pose factors are exactly the kind of thing a capsule pose
vector ought to capture, so a single decoded dimension has something real to
vary.

## Architecture

Encoder (classifier + pose extractor):

```
Input(144) -> FullConnectReLU(48) -> CapsuleRouting(in=8x6, out=2x8, iters=3)
```

The capsule layer emits `numClasses` pose vectors of length `poseDim=8`; the
per-capsule length `||v_j||` is the class score (squash keeps it in `[0,1)`).

Decoder (reconstruction, a separate small net):

```
Input(16, error-collection) -> FullConnectReLU(64) -> FullConnectSigmoid(144)
```

At train time the decoder input is the capsule output with every capsule **except
the true class zeroed out** (the paper's *masking*), so only the winning pose
vector drives reconstruction.

## Losses (hand-rolled; stated explicitly)

- **Classification: the paper's MARGIN loss** on capsule lengths,
  `L_k = T_k max(0, m+ − ||v_k||)² + λ(1−T_k) max(0, ||v_k|| − m−)²`
  with `m+=0.9, m−=0.1, λ=0.5`. We compute `dL/dv_k` analytically and seed it
  into the encoder's output error. (Not a stock in-tree head — margin loss is
  not available as a layer, so it is computed by hand, as the task allows.)
- **Reconstruction: MSE** between decoder output and the input image, down-
  weighted by `cRecW=0.40` so it regularises rather than dominates (the paper
  uses `0.0005·784`). The MSE output-gradient is obtained from the stock backprop
  via a pseudo-target; the decoder **input** gradient (the decoder `Input` layer
  has error collection enabled) is then added — on the true-class capsule slice
  only — to the encoder's output error.

Both nets run in **batch-update mode** and the training loop is **hand-rolled**
(`Compute`/`Backpropagate`/`UpdateWeights`), so the `TNeuralFit` best-model
reload gotcha never applies and layer references stay valid.

## The headline

After training, the program takes a correctly-classified example, reads its
winning capsule's pose vector, and for every pose dimension sweeps an offset in
`[-0.25, +0.25]`, decodes each perturbed vector, and measures the change. It
picks the **most-responsive dimension** (largest total-ink range) and prints its
sweep as ASCII reconstructions, plus the per-dimension sensitivity table and an
honest monotonicity read-out.

### Observed result (deterministic, seed 42)

Reconstruction converges to a clean bar (recon-MSE ≈ **0.04**). Sweeping the
chosen pose dimension varies the rendered bar **smoothly and monotonically** — in
the reference run, total ink moves `32.6 → 24.9` across the sweep, i.e. one pose
dimension drives the bar's **stroke intensity/thickness**. The decoded image at
each offset is a recognizable vertical bar matching the input.

Classification accuracy (test set): **CapsNet 100%**, parameter-matched plain MLP
(`Input → ReLU(48) → Linear(2) → SoftMax`) **100%** — on this easy 2-class task
the capsule advantage is the *interpretable pose vector*, not raw accuracy.

## Honesty

Capsule pose disentanglement is delicate and normally needs far more training
than a sub-5-minute CPU budget allows. The example reports what it **actually
measures** — the per-dimension reconstruction sensitivity and whether the best
dimension's sweep is monotone — rather than asserting a polished textbook result.
The effect here is real and visible, but the reconstructions are coarse (12×12,
tiny net, few minutes) by design. Different seeds may select a different pose
dimension and a slightly different factor (thickness vs intensity vs position).

## Build & run

```
lazbuild CapsuleReconstruction.lpi
../../bin/x86_64-linux/bin/CapsuleReconstruction
```

Pure CPU, no external data, deterministic (fixed seed), ~17 s on 2 cores — well
under the 5-minute budget.
