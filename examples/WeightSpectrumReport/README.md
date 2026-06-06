# WeightSpectrumReport

Demonstrates `TNNet.WeightSpectrumReport`: a **forward-only, weight-tensor-only**
per-layer spectral diagnostic. It needs **no probe batch** — it inspects the
weight matrices directly.

## What it does

1. Builds a small MLP (`FullConnectReLU -> FullConnectReLU -> FullConnectLinear`)
   on an 8-dim input.
2. Prints `TNNet.WeightSpectrumReport` on the **fresh-init** network.
3. Trains briefly (60 epochs of batch 32) on the trivial synthetic task
   `y = ||x||` and prints the report again, so you can eyeball how training
   pushes the spectrum away from the Gaussian-init baseline.

No GPU, runs in well under a minute.

## What the report shows

For every trainable layer (those with weights) the layer's weights are treated
as a matrix `W` of shape `[num_neurons (fan-out) x weights_per_neuron (fan-in)]`
(biases excluded, exactly as `WeightHistogramReport` selects weights). Per layer:

```
Idx  Class                      Shape(o,i)     fout   fin    sigma_1    ||W||_F  sr-ratio  MP-ratio
--------------------------------------------------------------------------------------------------------------
1    TNNetFullConnectReLU       (32,8)           32     8     ...
...
```

- **sigma_1** — top singular value of `W`, estimated with a handful of
  power-iteration steps (default 10) via the reusable `EstimateSpectralNorm`
  helper. `u = W v; v = W^T u; v := v/||v||`, repeated; `sigma_1 ~= ||W v||`.
- **||W||_F** — Frobenius norm (cheap exact, `sqrt(sum w^2)`).
- **sr-ratio** = `sigma_1 / ||W||_F` — a stable-rank-flavoured signal. Values
  near `1` hint at **rank-1 collapse** (one direction dominates); values near
  `1/sqrt(min(in,out))` hint at a **well-spread** spectrum.
- **MP-ratio** = `sigma_1 / ((sqrt(in)+sqrt(out)) * std(W))` — a
  Marchenko-Pastur baseline ratio answering "is this layer's top mode larger
  than what a Gaussian init of matching std would produce?". `~1` is init-like.

Closing summary:

- A 10-bin ASCII histogram of the per-layer **fan-in baseline ratio**
  (`sigma_1 / (sqrt(in) * std(W))`) across the whole network.
- A **flag list**: "spectral-norm > threshold" layers (Lipschitz risk; default
  threshold `2.0`) and "stable-rank ~= 1" layers (`sr-ratio >= 0.95`,
  representation-collapse risk).

The spectral-norm helper (`TNNet.EstimateSpectralNorm`) is deterministic
(fixed internal seed) and reusable by a future `TNNetSpectralNorm` wrapper.

## Running

```
cd examples/WeightSpectrumReport
lazbuild WeightSpectrumReport.lpi
../../bin/<arch>/bin/WeightSpectrumReport
```

Or directly with fpc:

```
cd examples/WeightSpectrumReport
fpc -B -Fu../../neural -Mobjfpc -Sh -O2 WeightSpectrumReport.lpr
./WeightSpectrumReport
```
