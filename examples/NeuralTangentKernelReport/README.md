# NeuralTangentKernelReport

Demonstrates `TNNet.NeuralTangentKernelReport`: a **forward-only** diagnostic
that measures the **empirical Neural Tangent Kernel** (Jacot, Gabriel & Hongler
2018) of a classifier over a small probe batch.

## What is the NTK?

The Neural Tangent Kernel couples training examples through their *parameter*
gradients of the network output:

```
K(x, x') = < grad_theta f(x) , grad_theta f(x') >
```

where `f` is the scalar **target-class logit**. In the infinite-width limit a
net trains as a kernel machine whose kernel `K` is *fixed*, and even at finite
width the **empirical** NTK over a probe batch tells you how the model couples
its examples and how well that coupling is aligned with the labels.

For each probe sample `i` the report runs **one forward + one backward** on a
**frozen** net (`ClearDeltas` before each, `SetBatchUpdate(true)`, **never**
`UpdateWeights`), seeded with a one-hot at the target class, so the recovered
per-parameter weight gradient is exactly `grad_theta` of that logit. The
gradient is read straight out of `Neurons[*].Delta` / `FBiasDelta` (divided back
out by the layer learning rate), the same machinery `FisherImportanceReport`,
`GradientConflictReport` and `GradientNoiseScaleReport` share. `Backpropagate`
stores `Delta = -LearningRate * grad`, so a **global sign flip** is present — but
it **cancels** in the Gram entries `K_ij = <g_i, g_j>` (a dot product of two
gradients), so the eigenvalues and the alignment are unaffected.

## What it does

This demo is entirely **synthetic** (no dataset download) and runs in a few
seconds:

1. **(A) Fresh-init network** on a synthetic 3-class blob task.
2. **(B) Trained network** on the same task.

Training markedly **reshapes** the empirical NTK — the eigenspectrum, condition
number, effective rank and kernel-target alignment all move between the two
reports, which is the point: the NTK is a live picture of how the *current*
weights couple the probe examples.

Pure CPU, forward + backward **read** only — the weights are never stepped.

## What the report shows

```
NeuralTangentKernelReport: empirical NTK over 9 usable probe(s) of 9, ...
Kernel heatmap (rows/cols = probe index, ramp ' .:-=+*#%@' from 0 to |K|max):
       0 1 2 3 4 5 6 7 8
  [ 0] @     =     -   .
  ...
Eigenvalues (N=9): lambda_max=4.25549, lambda_min=0.285782, trace=9.8059.
Condition number lambda_max/lambda_min = 14.89.
Kernel-target alignment =   0.8041  (headline; ...).
Effective rank (participation ratio) = 4.041 / 9 (44.9% ...).
log10(lambda) histogram across 9 eigenvalue(s):
  ...
```

How to read it:

- **Kernel heatmap** — the `N x N` Gram `K_ij` rendered with the `' .:-=+*#%@'`
  glyph ramp (blank = 0, `@` = `|K|max`). Block structure means the probes group
  into kernel-coupled clusters.
- **Eigenspectrum** — the full set of eigenvalues from the same self-contained
  Double-precision cyclic Jacobi eigensolver `WeightSpectralTailReport` /
  `IntrinsicDimensionReport` ship (no new numerical code; PSD, so tiny negative
  numerical noise is clamped to 0).
- **Condition number** `lambda_max / lambda_min` — large numbers mean an
  ill-conditioned kernel: slow gradient-descent / kernel-regression convergence.
  `+inf` when `lambda_min ~ 0` (redundant / collinear probes).
- **Kernel-target alignment** `<K, yy^T>_F / (||K||_F * ||yy^T||_F)`
  (Cristianini, Shawe-Taylor, Elisseeff & Kandola 2001) — the **headline
  number**, where `y` is the **centered** target-class indicator over the probe
  batch. `1` = the kernel's dominant eigen-directions perfectly separate the
  target class; `0` = unaligned.
- **Effective rank / participation ratio** `(sum lambda)^2 / sum lambda^2` — how
  many kernel directions actually carry signal (out of `N`).
- **log10(lambda) histogram** — the `#`-bar spectrum shape at a glance.

Built-in correctness checks (also asserted in the smoke test): the kernel is
symmetric by construction (reported residual `~0`) and the diagonal
`K_ii = ||g_i||^2 > 0` for every usable probe.

A possible **follow-up** (deliberately not done here to keep this first version
focused) is a fresh-init-vs-trained **NTK-drift** contrast: how far the
empirical kernel itself moves during training.

## Running

```
cd examples/NeuralTangentKernelReport
lazbuild NeuralTangentKernelReport.lpi
../../bin/<arch>/bin/NeuralTangentKernelReport
```

Or directly with fpc:

```
cd examples/NeuralTangentKernelReport
fpc -B -Fu../../neural -Mobjfpc -Sh -O2 NeuralTangentKernelReport.lpr
./NeuralTangentKernelReport
```
