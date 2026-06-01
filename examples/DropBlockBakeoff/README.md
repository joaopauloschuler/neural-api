# DropBlock vs plain Dropout bake-off

This example trains one small CNN **three ways** at a **matched drop rate** and
prints the final train/test loss, accuracy and the train/test **gap** for each
arm. It uses only existing in-tree layers — the `TNNetDropBlock` layer already
lives in `neural/neuralnetwork.pas`; no new layer is added.

## The phenomenon

**DropBlock** (Ghiasi et al. 2018, *DropBlock: A regularization method for
convolutional networks*) is a *structured* dropout for convolutional feature
maps. Plain `TNNetDropout` zeroes individual activations independently,
scattered across the map. On a conv feature map that is a weak regulariser:
neighbouring units are spatially correlated, so a dropped pixel's information
still survives in its neighbours and the network just routes around the holes.

`TNNetDropBlock(block_size, rate)` instead samples **one spatial mask** per
`(x, y)` position, zeroes a contiguous `block_size × block_size` square, and
broadcasts that mask across **all channels** (Depth) — so a whole local patch of
the feature map disappears at once. That removes spatially-correlated
information the network cannot trivially recover, which is the intended stronger
regularisation for conv nets.

Both layers use inverted dropout (survivors are rescaled so the expected
activation is preserved) and **both are the exact identity at inference**
(`FEnabled = false`), so eval is deterministic.

## What this example builds

```
Input(12×12×3) -> ConvReLU(24,3) -> [REG]
               -> ConvReLU(24,3) -> MaxPool(2) -> FC(3) -> SoftMax
```

`[REG]` is the regulariser slot, sitting right after the first conv feature map
(`12×12 × 24`) where DropBlock is meant to live — a spatial map with room for a
`3×3` block. The three arms differ **only** in that slot:

| arm         | `[REG]` layer                       | drop style          |
|-------------|-------------------------------------|---------------------|
| `none`      | `TNNetIdentity`                     | no regulariser      |
| `dropout`   | `TNNetDropout(0.15)`                | scattered per-pixel |
| `dropblock` | `TNNetDropBlock(3, 0.15)`           | localized 3×3 patch |

Dropout and DropBlock are matched at the **same** drop rate (`0.15`). The task
is a fixed 3-way image classification: each class owns a random spatial template
that is stamped at a random amplitude into a noisy `12×12×3` image; the label is
the class whose template was stamped. The amplitude is deliberately low, the
training set deliberately tiny (96 images), so the over-capacity CNN over-fits
and a regulariser has room to help. The teacher and both datasets are generated
once, and the net is re-seeded before each build, so all three arms see
identical data and identical weight initialisation (`RandSeed := 424242`); only
the `[REG]` layer and its train-time RNG draws differ.

## Observed results

Single-threaded, `RandSeed := 424242`, 35 epochs, ~144 s wall-clock on CPU:

```
    arm    | initTrnLoss finalTrnLoss | trainLoss trainAcc | testLoss testAcc | gap(test-trn) | diverged
  ---------+--------------------------+--------------------+------------------+---------------+---------
  none     |      1.4040       0.0671 |    0.0430   1.0000 |   0.3026  0.9500 |        0.2596 | no
  dropout  |      1.4040       0.0666 |    0.0213   1.0000 |   0.2196  0.9575 |        0.1983 | no
  dropblock |      1.4040       0.0771 |    0.0281   1.0000 |   0.2489  0.9600 |        0.2207 | no
```

### Did the regularisers help? (honest read)

The unregularised `none` baseline clearly **over-fits**: it reaches 100% train
accuracy but only 95.0% test accuracy, a train/test loss gap of `0.26`. **Both**
regularisers shrink that gap (`0.26 → 0.20` for Dropout, `0.26 → 0.22` for
DropBlock) and **both** improve held-out accuracy over the baseline. On *this*
toy:

- **DropBlock gives the best held-out accuracy** (`0.9600`, vs `0.9575` for
  Dropout and `0.9500` for the baseline) — the localized-patch regulariser
  generalises best on the metric that counts predictions.
- **Plain Dropout shrinks the test-loss gap the most** (`0.1983`) — it makes the
  net least over-confident on the cross-entropy.

So the two regularisers trade blows depending on the metric, and both beat doing
nothing. This is the expected, honest outcome on a small synthetic task: a
regulariser *can* help here, but it does not always sweep every metric, and the
example reports the numbers rather than pretending DropBlock must win outright
(same spirit as the `DropPathAblation` / `OptimizerBakeoff` README caveats).
Change `cDropRate`, `cBlockSize`, the amplitude or the train-set size and the
ranking can flip.

## Self-check (gate)

The program `Halt(1)`s unless all of these hold — invariants that are actually
true, not a brittle "DropBlock always generalises better" claim:

1. **No arm diverges** — every reported loss is finite (no NaN/Inf).
2. **Every arm trains** — final train loss < initial (random-init) loss.
3. **The `none` baseline learns** — train accuracy well above the `1/3` chance
   level, confirming the plain CNN is a healthy classifier.
4. **Inference is deterministic** — two eval passes give bit-identical loss for
   every arm, proving both DropBlock and Dropout are the exact identity at eval.

Evaluation always calls `NN.EnableDropouts(false)` first, so the regularisers
are the identity and the reported numbers are deterministic at inference.

## Build & run

```
cd examples/DropBlockBakeoff
fpc -O3 -Mobjfpc -Sh -Fu../../neural -dRelease -dUseCThreads DropBlockBakeoff.lpr
./DropBlockBakeoff
```

`-dUseCThreads` pulls in the cthreads driver that `neuralfit`'s worker pool
needs (the fit is still single-threaded via `NFit.MaxThreadNum := 1`, which
keeps the reductions deterministic). If `fpc` cannot find `UTF8Process` (used by
`neuralthread`), add the LazUtils unit path, e.g.
`-Fu/usr/share/lazarus/<ver>/components/lazutils/lib/x86_64-linux`.
