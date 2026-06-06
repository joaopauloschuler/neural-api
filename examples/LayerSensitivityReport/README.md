# LayerSensitivityReport

Tiny example for `TNNet.LayerSensitivityReport`, the forward-only per-layer
weight-noise sensitivity diagnostic.

The report multiplicatively jitters every **trainable** layer's weights (and
biases) by small Gaussian noise (`W *= 1 + eta`, `eta ~ N(0, sigma^2)`),
re-runs the probe batch, and measures how much the network's final-layer
output changes — restoring the weights exactly between trials (the whole net
is snapshotted with `SaveDataToString` and re-applied with
`LoadDataFromString`, so the network is bit-for-bit unchanged when the report
returns). No backward pass is run. For each trainable layer it reports:

- the **mean** and **max** output-delta L2 over `Trials` trials (how much the
  output moves under small weight noise),
- the **mean loss-delta** vs a supplied target list (or `n/a` when no targets
  are given),
- a **normalised sensitivity** = `mean output-delta / param_count`, so
  naturally large layers don't dominate purely by size,
- a 10-bin ASCII histogram of the per-layer mean output-delta across the
  network,
- a flag list of **high-impact** (top 10%) and **low-impact** (bottom 10%)
  layers,
- a one-line **fragility verdict**: the ratio of the maximum to the median
  per-layer sensitivity (a high ratio means a few layers carry the model).

`Sigma` (default `0.01`), `Trials` (default `8`), `MaxSamples` (default `64`)
and `Seed` (default `1234567`, for reproducible Gaussian noise) are
parameters. The global RNG is saved and restored around the run.

This program runs the report across **three model families** so reviewers can
eyeball how the fragility shape shifts:

1. **MLP** (`8 -> 16 -> 16 -> 1`) on a synthetic `y = max(0, w.x)` regression
   target, briefly trained, run **with** a target list so the loss-delta
   column is populated.
2. **CIFAR-style conv stack** (`conv -> pool -> conv -> pool -> fc -> fc`) on
   synthetic `8x8x3` inputs, fresh-init, output-delta only.
3. **Attention stack** (embedding -> sinusoidal positions -> Q|K|V pack ->
   single-head SDPA -> per-position projection -> softmax) on synthetic
   token-ID sequences, fresh-init, output-delta only.

To stay self-contained and fast, the inputs are tiny synthetic volumes of the
right shape (no dataset download); the point is the sensitivity **shape**
across layers, not accuracy.

## Build & run

```
cd examples/LayerSensitivityReport
lazbuild LayerSensitivityReport.lpi
../../bin/x86_64-linux/bin/LayerSensitivityReport
```

Total runtime is well under a minute (in practice ~1 second).
