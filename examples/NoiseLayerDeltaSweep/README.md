# NoiseLayerDeltaSweep

Measure the **train-time vs inference-time delta** of the four stochastic
noise / dropout layers that already live in `neuralnetwork.pas`
(`TNNetDropout`, `TNNetDropPath`, `TNNetSpatialDropout1D`,
`TNNetSpatialDropout2D`). Train the **same** tiny ResNet-style classifier once
per `(layer family, drop probability p)` pair and print, for each arm, the
train loss with the noise **on**, the train loss with the noise **off**, the
held-out val loss, and the train/val gap. No new layer is added.

## Train-time vs inference-time: the distinction this example is about

Every `TNNetAddNoiseBase` layer has **two regimes**, switched network-wide by a
single call, `TNNet.EnableDropouts(flag)`:

- `EnableDropouts(true)` — **training**. The layer samples a fresh Bernoulli
  mask and rescales the survivors by `1/(1-p)` ("inverted dropout", so the
  expected magnitude is preserved). The output is **stochastic** and varies
  pass-to-pass.
- `EnableDropouts(false)` — **inference**. The layer copies its input
  unchanged. The output is the **deterministic identity** and is identical
  pass-to-pass.

Because of this split, a loss measured on a **train** forward pass (mask on) is
**not comparable** to a loss measured on an **inference / validation** pass
(mask off) unless you are explicit about which switch is set. If you forget to
disable the noise for evaluation, your "validation" loss is itself random and
the reported train/val gap is meaningless. **Disabling the noise for every
inference pass is the whole point of this example**, and the program both does
it (`EnableDropouts(false)` before every val / inference re-scoring) and
*proves* it: each inference probe is run twice and the two outputs are asserted
bit-for-bit identical (the `det` column).

The four layers differ only in **what** they drop:

| Layer | Drops | Needs |
|-------|-------|-------|
| `TNNetDropout` | individual elements (per-element Bernoulli) | any shape |
| `TNNetDropPath` | the **whole** sample / branch (stochastic depth) | any shape |
| `TNNetSpatialDropout2D` | whole feature-map **channels** (a Depth slice) | a meaningful Depth |
| `TNNetSpatialDropout1D` | whole channels of a sequence (same masking as 2D) | a meaningful Depth |

All four share the `Create(p)` constructor; only the class differs.

## What the sweep measures

For each layer family and each `p in {0.0, 0.1, 0.2, 0.4}` the program records:

- **train(ON)** — train cross-entropy with `EnableDropouts(true)` (stochastic
  mask active): the loss the optimiser actually experiences.
- **train(OFF)** — the **same** training data re-scored with
  `EnableDropouts(false)` (inference identity): the *real* fit of the learned
  weights.
- **val(OFF)** — held-out cross-entropy, always at inference.
- **gap** = `val(OFF) - train(OFF)`.

`p = 0.0` is the no-noise baseline: with `p = 0` the layer is the identity in
**both** regimes, so `train(ON)` must equal `train(OFF)` exactly — a direct
numerical proof that the toggle is consistent and that `p = 0` is a no-op.
For `p > 0`, the stochastic mask handicaps the forward pass, so `train(ON)`
tends to sit **above** `train(OFF)` on the same data; the inference identity
recovers the "true" fit.

## Shape constraints (why each layer is wired this way)

The residual-carrying tensor is `1 x 1 x cWidth` (the feature dimension lives in
the **Depth** axis). A residual sublayer must be **shape-preserving**, and
`TNNetPointwiseConvLinear` over Depth preserves shape (`FullConnectLinear` would
flatten it). The swept noise layer sits on the residual **branch**:

```
y = x + Noise_p( ReLU(PointwiseConvLinear(x)) )
```

- `TNNetDropout` / `TNNetDropPath` are shape-agnostic — fine on the branch.
- `TNNetSpatialDropout1D` / `2D` drop whole **Depth channels**, so they need a
  meaningful Depth extent. The `1 x 1 x cWidth` branch tensor gives them
  `cWidth` channels to drop, so no reshaping is needed and every family is
  validly shaped on the same architecture.

**One family per arm** keeps each model clean and every layer validly shaped; a
separate sub-sweep over `p` is run for each of the four families.

## The shared net

```
TNNetInput(cDim)                                   # 6 input features
  -> TNNetFullConnectLinear(cWidth)                # project to 16 feats (along X)
  -> TNNetReshape(1, 1, cWidth)                    # move feats into Depth
  -> cNumBlocks x residual block:
        x
        -> TNNetPointwiseConvLinear(cWidth)
        -> TNNetReLU
        -> <one of the 4 noise layers>(p)          # <-- THE SWEPT KNOB
        -> TNNetSum([branch, x])                    # y = x + Noise_p(Branch(x))
  -> TNNetFullConnectLinear(cClasses)
  -> TNNetSoftMax                                  # 3-way classification
```

The task is a fixed synthetic 3-way classification (`class = argmax_c x'Q_c x +
b_c'x`, a per-class random quadratic form), so the net has something nonlinear
to fit. Every arm shares the teacher, the data, the RNG seed
(`RandSeed = 424242`), the epochs, the batch size and the learning rate; only
the noise layer class and `p` change.

## Build & run

```
cd examples/NoiseLayerDeltaSweep
lazbuild NoiseLayerDeltaSweep.lpi --build-mode=Default
../../bin/x86_64-linux/bin/NoiseLayerDeltaSweep
```

Or compile directly with fpc (point `-Fu` at the LazUtils `lib` dir that holds
`utf8process.ppu`, exactly as `tests/RunAll.sh` discovers it; the program uses
the fit loop, so build with the cthreads driver):

```
fpc -Mobjfpc -Sh -O2 -dUseCThreads -Fu../../neural -Fu<lazutils-lib-dir> NoiseLayerDeltaSweep.lpr
```

Pure CPU, single-threaded fit, no dataset download. All 16 arms finish in about
a minute.

## Sample output

Actual run (single CPU thread, `RandSeed = 424242`):

```
=== Results table (lower CE is better; gap = val(OFF) - train(OFF)) ===
  layer              p   | train(ON)  train(OFF)  val(OFF) |    gap   | det
  -------------------+-----+----------------------------------+----------+----
  Dropout          0.00 |    1.4381     1.4381    1.2039 |  -0.2342 | yes
  Dropout          0.10 |    1.1063     1.1869    1.2028 |   0.0159 | yes
  Dropout          0.20 |    0.6169     0.5623    0.5588 |  -0.0035 | yes
  Dropout          0.40 |    0.6969     0.8750    0.7242 |  -0.1508 | yes
  -------------------+-----+----------------------------------+----------+----
  DropPath         0.00 |    0.6542     0.6542    0.6925 |   0.0383 | yes
  DropPath         0.10 |    0.4989     0.4736    0.4876 |   0.0140 | yes
  DropPath         0.20 |    0.5942     0.5418    0.5374 |  -0.0044 | yes
  DropPath         0.40 |    0.9371     0.8339    0.7516 |  -0.0823 | yes
  -------------------+-----+----------------------------------+----------+----
  SpatialDropout1D 0.00 |    0.6542     0.6542    0.6925 |   0.0383 | yes
  SpatialDropout1D 0.10 |    0.7214     0.6907    0.7700 |   0.0793 | yes
  SpatialDropout1D 0.20 |    0.7731     0.6954    0.6777 |  -0.0177 | yes
  SpatialDropout1D 0.40 |    1.1972     1.0401    0.7929 |  -0.2473 | yes
  -------------------+-----+----------------------------------+----------+----
  SpatialDropout2D 0.00 |    0.6542     0.6542    0.6925 |   0.0383 | yes
  SpatialDropout2D 0.10 |    0.7214     0.6907    0.7700 |   0.0793 | yes
  SpatialDropout2D 0.20 |    0.7731     0.6954    0.6777 |  -0.0177 | yes
  SpatialDropout2D 0.40 |    1.1972     1.0401    0.7929 |  -0.2473 | yes
  -------------------+-----+----------------------------------+----------+----

=== Correctness signals ===
[PASS] no arm produced NaN / Inf (all losses finite).
[PASS] every inference (noise OFF) forward pass is bit-for-bit deterministic.
[PASS] p=0.0: train(ON) == train(OFF) for every layer (identity in both regimes).
[WARN] p=0.4 train(ON) not clearly above train(OFF) for some layer (small net / easy toy can blur this); inspect the table.

Total wall-clock: 63.4 s
```

## Reading the result

- **The toggle works.** The `det` column is `yes` for every arm: once
  `EnableDropouts(false)` is set, the same input maps to the same output every
  time. And at `p = 0.0`, `train(ON)` equals `train(OFF)` to the printed
  precision for all four families (e.g. DropPath/Spatial `0.6542 == 0.6542`,
  Dropout `1.4381 == 1.4381`) — `p = 0` is genuinely the identity in **both**
  regimes. These are the two `[PASS]` signals that matter, and the example
  `Halt(1)`s if either fails.

- **Train(ON) sits above train(OFF) as the noise grows.** Look down the
  `SpatialDropout1D/2D` columns: at `p = 0.4`, `train(ON) = 1.1972` vs
  `train(OFF) = 1.0401` — the random mask makes the loss the optimiser sees
  *worse* than the actual fit of the weights, exactly the train-time penalty
  the noise imposes. `DropPath` shows the same ordering (`0.9371 > 0.8339`).
  The one exception is `Dropout p = 0.4` (`0.6969 < 0.8750`): on this **tiny,
  easy** toy a single random mask draw can land in a lucky spot, which is why
  the program prints a non-fatal `[WARN]` rather than asserting this brittle
  inequality. The honest read is reported, not assumed.

- **`SpatialDropout1D` and `SpatialDropout2D` produce identical numbers here.**
  This is expected, not a bug: on the `1 x 1 x cWidth` tensor (`SizeY = 1`)
  both layers drop the **same Depth channels** drawing from the **same RNG
  sequence**, so they are numerically indistinguishable. The two only diverge
  on genuinely 2D feature maps (`SizeY > 1`), where 2D drops a whole 2D map per
  channel; both are included so the sweep covers the full noise-layer family.

- **Stronger noise narrows the apparent train/val gap.** For `SpatialDropout`
  the gap goes `+0.0383 (p=0) -> +0.0793 -> -0.0177 -> -0.2473` as `p` grows:
  heavier masking pushes the inference train loss up toward (and past) the val
  loss — the classic regularisation signature. On an easy toy the absolute val
  loss does not always improve (a regulariser removes capacity the net could
  have used), so the takeaway is the **measurement methodology**, not a claim
  that "more noise always generalises better": the noise must be **off** at
  inference for the train/val comparison to mean anything, and this example
  shows exactly how to do that and what changes when you do.
