# Weight Standardization + GroupNorm vs BatchNorm

A tiny, pure-CPU bake-off that exercises the **`TNNetWeightStandardizationConv`**
layer — the convolution sibling of the dense `TNNetWeightStandardization`.

## The layer

**Weight Standardization** (Qiao et al. 2019, *Micro-Batch Training with
Batch-Channel Normalization and Weight Standardization*, arXiv:1903.10520)
standardizes each convolution **filter**'s weights — all `SizeX*SizeY*Depth`
weights of one output channel — to **zero mean / unit standard deviation**
*before* the convolution:

```
w_hat = (w - mean(w)) / sqrt(var(w) + eps)
```

The raw weights stay the trainable parameters; the standardization is applied on
the fly and its **exact per-output-channel Jacobian** (the same centering +
scaling Jacobian used by batch / layer norm, applied over each filter's weight
tensor) is propagated in the backward pass, so the reported weight gradient is
the gradient w.r.t. the *raw* weights. `TNNetWeightStandardizationConv` is a
drop-in subclass of `TNNetConvolutionLinear`: it behaves like a plain linear
convolution but with standardized filters, and it round-trips through
`SaveStructureToString` / `LoadFromString` (eps in `FFloatSt[0]`).

Unlike BatchNorm — which normalizes **activations** using **per-batch**
mean/variance — WS normalizes the **weights**, so it is completely independent
of the batch and does **not** degrade when the batch is tiny. The headline WS
recipe pairs it with **GroupNorm** (a batch-size-independent activation norm):
WS + GroupNorm is meant to **match or beat BatchNorm** at a matched
architecture, especially at small batch sizes.

## The bake-off

One small CNN backbone is built and trained twice, swapping **only** the
normalization style (same widths, depth, FC head, seed, data, epochs, LR, and a
deliberately **small** batch size):

```
Input(12x12x3)
  -> [Norm-Conv(16,3) ] x2   -> MaxPool(2)
  -> [Norm-Conv(16,3) ]      -> MaxPool(2)
  -> FullConnectLinear(4) -> SoftMax
```

| arm     | normalization block                                                          |
|---------|------------------------------------------------------------------------------|
| `bn`    | `TNNetConvolutionLinear` + `AddMovingNorm` (moving-mean/var BatchNorm) + ReLU |
| `ws_gn` | `TNNetWeightStandardizationConv` + `TNNetGroupNorm` + ReLU (the Qiao recipe)  |

The synthetic task is a low signal-to-noise image classification: each class
owns a random spatial template stamped at a small amplitude into unit Gaussian
noise.

## Running

```
cd examples/WeightStandardizationConv
fpc -dUseCThreads -Fu../../neural -Fu<lazutils-path> -Mobjfpc -Sh -O2 WeightStandardizationConv.lpr
./WeightStandardizationConv
```

(`<lazutils-path>` is the directory containing `utf8process.ppu`, the same one
`tests/RunAll.sh` auto-discovers. Or just open the `.lpi` in Lazarus.)

## Example output

```
=== Results table ===
    arm   | initTrnLoss finalTrnLoss | trainLoss trainAcc | testLoss testAcc | diverged
  --------+--------------------------+--------------------+------------------+---------
  bn     |      1.4650       0.2749 |    0.1715   1.0000 |   1.4075  0.3925 | no
  ws_gn  |      1.9863       0.1211 |    0.0672   1.0000 |   1.4711  0.3613 | no

Head-to-head TEST accuracy: bn=0.3925 vs ws_gn=0.3613 (ws_gn - bn = -0.0313).
=> On this toy/seed WS+GroupNorm MATCHED BatchNorm (within tolerance) ...
=> ALL CHECKS PASS.
```

## Honesty caveat

This is a **small, easy synthetic** task. WS + GroupNorm matching or beating
BatchNorm is a real *small-batch* phenomenon, but on an easy toy the margin is
small and either arm can win on a given seed. So the self-check does **not**
assert "WS+GroupNorm always beats BatchNorm". It asserts invariants that are
actually true:

1. both arms **train** (final train loss < random-init loss, no NaN / Inf),
2. both arms reach a **healthy** classifier (well above the `1/4` chance rate),
3. WS + GroupNorm is **competitive**: its test accuracy is within a small
   tolerance of (or better than) the BatchNorm arm.

The head-to-head test accuracy is **printed and discussed honestly** — on this
particular easy toy the two arms *match* within tolerance; the genuine WS
advantage shows on harder data and smaller batches. The example is deterministic
(`RandSeed := 424242`), single-threaded, downloads no data, and finishes well
under the 5-minute budget (~65 s).
