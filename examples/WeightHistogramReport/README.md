# WeightHistogramReport example

Demonstrates `TNNet.WeightHistogramReport`, a per-trainable-layer
weight-value histogram diagnostic.

For every layer that owns weights this helper reports min, max, mean,
population std, L2 norm, L-inf norm, sparsity (fraction of weights with
`|w| < 1e-6`) and an ASCII bar histogram of the weight distribution
across `Bins` bins over `[-MaxAbs, +MaxAbs]` (per-layer scaling so each
layer fills its own range). Biases are excluded; the closing line
reports total trainable weight count and trainable-layer count.

The example builds a 2-hidden-layer ReLU MLP and trains it briefly on a
synthetic `y = ||x||` regression task. It prints the histogram twice
(before and after training) and a one-line max-abs summary, so the shift
in distribution shape and range is visible.

## Build & run

```
cd examples/WeightHistogramReport
lazbuild WeightHistogramReport.lpi
../../bin/x86_64-linux/bin/WeightHistogramReport
```

Runs well under 30 seconds, pure CPU, no datasets required.
