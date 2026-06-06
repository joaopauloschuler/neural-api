# WeightSpectralTail

Demonstrates `TNNet.WeightSpectralTailReport`: a **forward-only, weight-tensor-only**
*heavy-tailed self-regularization* (HT-SR) diagnostic that predicts per-layer
training quality from the **weights alone** — no probe batch, no labels, no test
set (Martin, Mahoney & Peng 2021, *Nature Communications*; the WeightWatcher
`alpha` metric).

## What it does

1. Builds a small MLP (`FullConnectReLU -> FullConnectReLU -> FullConnectLinear`)
   on a 16-dim input.
2. Prints `TNNet.WeightSpectralTailReport` on the **fresh-init** network (alpha
   sits high — the spectrum is still random-like / under-trained).
3. Trains briefly (80 epochs of batch 32) on the trivial synthetic task
   `y = ||x||` and prints the report again, so you can see alpha move toward the
   well-shaped `[2, 4]` band as the layers actually learn.

No GPU, runs in well under a minute, no dataset download.

## What the report shows

For every trainable layer the weights are treated as a matrix `W` of shape
`[num_neurons (fan-out) x weights_per_neuron (fan-in)]` (biases excluded). The
report forms the **smaller** Gram matrix (`W^T W` when fan-out >= fan-in, else
`W W^T`), computes its **full eigenvalue spectrum** `{lambda_i}` (= the squared
singular values of `W`) with a self-contained symmetric **cyclic Jacobi
eigensolver** in Double precision, and fits a power law `rho(lambda) ~
lambda^(-alpha)` to the upper tail.

```
Idx  Class                      Shape(o,i)        alpha   w-alpha       KS   lambda_max     MP-edge
----------------------------------------------------------------------------------------------------
1    TNNetFullConnectReLU       (48,16)           ...
...
```

- **alpha** — the power-law tail exponent via the Clauset/Hill MLE
  `alpha = 1 + n / sum_i ln(lambda_i / lambda_min)`, swept over candidate
  `lambda_min` cut points (the cut minimising the KS distance to the fitted
  power law is kept). This is the headline HT-SR quality number: well-trained
  layers land in `alpha in [2, 4]`; `alpha > 6` flags **under-trained**
  (still-random-like), `alpha < 2` flags **over-correlated** (memorising).
- **w-alpha** = `alpha * log10(lambda_max)` — the WeightWatcher capacity-weighted
  quality metric.
- **KS** — the Kolmogorov-Smirnov goodness-of-fit distance of the tail fit
  (large => the power law is a poor description, alpha unreliable).
- **lambda_max** — the largest eigenvalue (= `sigma_1(W)^2`).
- **MP-edge** — the Marchenko-Pastur bulk edge
  `(1 + sqrt(in/out))^2 * sigma^2 * fan_out`, the largest eigenvalue a pure
  Gaussian-random matrix of matching std would produce, for reference.
- A per-layer 10-bin ASCII `log10(lambda)` histogram of the spectrum.

Closing summary:

- The **average weighted alpha** across the fitted layers — a single label-free
  model-quality scalar (lower is better-trained).
- An **alpha-across-depth** bar chart with per-layer flags
  (`well-shaped` / `under-trained` / `over-correlated` / `poor power-law fit`).

Built-in correctness checks (surfaced as flags if they ever trip): the Jacobi
eigenvalues must be non-negative (PSD Gram) and sum to `||W||_F^2` within
tolerance (trace invariance), and `lambda_max` must match the power-iteration
`TNNet.EstimateSpectralNorm` value squared.

Over-wide layers are capped at `MaxMatrixDim` (default 512) on the Gram dimension
to bound the `O(d^3)` Jacobi sweep (such layers are listed as `skipped`).

The report is deterministic (no randomness) and pure CPU, forward-only on the
weight tensors — no backward pass, no probe batch, no weight perturbation.

## Running

```
cd examples/WeightSpectralTail
lazbuild WeightSpectralTail.lpi
../../bin/<arch>/bin/WeightSpectralTail
```

Or directly with fpc:

```
cd examples/WeightSpectralTail
fpc -B -Fu../../neural -Mobjfpc -Sh -O2 WeightSpectralTail.lpr
./WeightSpectralTail
```
