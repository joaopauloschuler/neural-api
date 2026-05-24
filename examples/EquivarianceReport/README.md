# EquivarianceReport

Tiny example for `TNNet.EquivarianceReport`, the forward-only input-symmetry
(invariance / equivariance) diagnostic.

Given a network and a probe batch of inputs, the report measures how the
forward output reacts to a fixed menu of input-side symmetry transforms and
prints, **per transform**:

- the **invariance error** = mean over the probe batch of
  `||f(T(x)) - f(x)||_2 / ||f(x)||_2` (0 means the model ignores the
  transform, i.e. is invariant to it);
- the **top-1 agreement** rate `mean(argmax(f(T(x))) == argmax(f(x)))`
  (meaningful for classifier-shaped outputs);
- a **10-bin ASCII histogram** of the per-sample invariance error so
  outliers are visible;
- a one-line **verdict**: `invariant` (`err < InvariantTol`, default 1e-3),
  `approximately invariant` (`err < ApproxTol`, default 1e-1) or `sensitive`.

The default transform menu for image-shaped inputs is `TNNetFlipX`
(horizontal mirror), `TNNetFlipY` (vertical mirror), `TNNetReverseChannels`
(channel reversal) and a 1-channel `TNNetRoll` (depth roll). Each `T(x)` is
produced by a tiny `Input -> Transform` forward-only wrapper net; no backward
pass is run and the inspected network's weights are never touched.

## What this demo shows

On a tiny synthetic `8x8x3` 3-class image task it builds and trains two
classifiers, then prints the report for each:

1. **NET A** — a plain conv classifier (`Conv -> MaxPool -> FC -> SoftMax`).
   It has no built-in spatial symmetry, so it is **flip-sensitive**: the
   FlipX / FlipY rows report a large invariance error (verdict `sensitive`).
2. **NET B** — `Input -> TNNetAvgChannel -> FC -> SoftMax`. A global
   per-channel **spatial average** is unchanged by any spatial permutation,
   so this net is **FlipX- and FlipY-invariant by construction**: the FlipX /
   FlipY rows report `~0` invariance error (verdict `invariant`). It is still
   sensitive to the channel permutations (`ReverseChannels` / `Roll`), which
   is visible in the contrast.

This is the built-in correctness check: a net that is invariant to a transform
by construction reads ~0 invariance error on that transform's row.

## Build & run

```
cd examples/EquivarianceReport
lazbuild EquivarianceReport.lpi
../../bin/x86_64-linux/bin/EquivarianceReport
```

Pure CPU, no dataset download. Total runtime is well under a minute.
