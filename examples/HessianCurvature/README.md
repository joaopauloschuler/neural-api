# HessianCurvature

Demonstrates `TNNet.HessianCurvatureReport`: a **loss-surface curvature /
sharpness** diagnostic built on **Hessian-vector products (HVPs)** — the classic
flat-vs-sharp-minimum metric (Keskar et al. 2017; Foret et al. *SAM* 2021),
estimated with **no second-order autograd**.

## What it does

1. Builds a tiny MLP (`FullConnectReLU -> FullConnectLinear`) on a separable
   synthetic 3-class problem.
2. Trains it **twice from the same architecture**:
   - a **sharp** minimum — small batch (4) + high learning rate (0.20);
   - a **flat** minimum — large batch (64) + low learning rate (0.02) + mild
     weight decay (0.01).
3. Prints `TNNet.HessianCurvatureReport` for each over a shared probe batch, so
   the `lambda_max` gap between the two minima is visible.

No GPU, no dataset download, runs in well under a minute. The report **never**
steps the weights — it is a measurement, not training.

## How the curvature is estimated

The Hessian is never formed. An HVP is estimated by **central-differencing the
per-parameter weight gradient**:

```
H v ~= ( grad L(theta + eps*v) - grad L(theta - eps*v) ) / (2*eps)
```

This reuses the existing whole-batch forward+backward gradient machinery (the
same per-parameter `Neurons[*].Delta` / `FBiasDelta` other reports read, divided
back out by the layer learning rate) and the exact
`SaveDataToString` / `LoadDataFromString` whole-net snapshot/restore pattern, so
weights are perturbed by `+/-eps*v` and restored **bit-for-bit** between probes.
The net is frozen: `ClearDeltas` before each pass, never `UpdateWeights`. `Eps`
defaults to `1e-3` (large enough to clear single-precision gradient round-off,
small enough that the central difference still tracks the true HVP).

## What the report shows

- **tr(H)** — the Hessian trace `E_v[v^T H v]` via the **Hutchinson estimator**
  over `NumProbes` (default 16) Rademacher (`+/-1`) probe vectors: the *mean*
  curvature averaged over all directions (each probe = two full-batch
  backprops).
- **lambda_max** — the **top Hessian eigenvalue** via a few power-iteration
  steps on the HVP operator (`v <- Hv/||Hv||`): the canonical sharpness metric.
- **tr(H)/N** and the **curvature concentration** `lambda_max / (tr(H)/N)` —
  how dominant the single sharpest direction is versus the mean.
- A **per-probe `v^T H v` histogram** (the Hutchinson estimator spread = its
  noise).
- A **per-trainable-layer trace breakdown** (each Hutchinson dot-product
  restricted to one layer's parameter slab — which layers carry the curvature).
- A **flat / moderate / sharp verdict** thresholded on `lambda_max`.

```
tr(H)        = ...   (Hutchinson mean curvature over 16 Rademacher probes)
lambda_max   = ...   (top Hessian eigenvalue, 8 power-iteration steps)
...
Per-layer trace breakdown (which layers carry the curvature):
  layer  class                        layer-tr(H)   %of-total
  ...
Verdict: ...
```

## Built-in correctness checks

(Also the smoke-test invariants.)

- A **purely linear net with an MSE head has a constant Hessian**, so `tr(H)`
  must be **probe-count-independent**: re-running with `2x NumProbes` returns the
  same value within Hutchinson noise.
- In the PSD Gauss-Newton regime **`lambda_max <= tr(H)`** must hold (surfaced as
  a PSD check line; a violation flags a negative-curvature / saddle direction or
  finite-difference noise).

## Running

```
cd examples/HessianCurvature
lazbuild HessianCurvature.lpi
../../bin/<arch>/bin/HessianCurvature
```

Or directly with fpc:

```
cd examples/HessianCurvature
fpc -B -Fu../../neural -Mobjfpc -Sh -O2 HessianCurvature.lpr
./HessianCurvature
```
