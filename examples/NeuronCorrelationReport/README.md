# NeuronCorrelationReport

Tiny example for `TNNet.NeuronCorrelationReport`, the intra-layer
redundancy diagnostic.

The program builds a small 2-hidden-layer ReLU MLP (`6 -> 32 -> 32 -> 1`)
and runs `TNNet.NeuronCorrelationReport(NN, Probes)` twice on the same
probe batch:

1. **FRESH INIT** — straight after weight initialisation. Neurons start
   independent: the high-`|rho|` tail of the histogram is empty and no
   near-duplicate-pair flags fire.
2. **AFTER TRAINING** — after a run on a deliberately *single-feature*
   target (`y = max(0, w.x)`: one fixed random ReLU teacher feature). The
   whole label is explained by that one rectified direction, so gradient
   descent drives a large fraction of the 32 hidden units to replicate the
   same direction. The `|rho|` distribution grows a clear `|rho| > 0.8`
   tail, near-duplicate-pair flags fire, and the effective neuron count
   drops sharply (in the run shipped here the second hidden layer falls
   from ~48% to ~25% of its nominal width).

Printing both makes the redundancy contrast visible.

For every trainable layer (those with weights) the report computes, over
the probe batch, the pairwise Pearson correlation `rho_ij` between
neurons **along the neuron axis** (correlation across the batch, not
across samples) and reports:

- a 10-bin ASCII histogram of `|rho_ij|` over the `i<j` pairs
- the top-K most-correlated neuron pairs (a "merge / prune one of each
  pair" candidate list)
- an **effective neuron count** = participation ratio of the correlation
  matrix, `N^2 / sum_ij rho_ij^2` (with the unit diagonal included), which
  lies in `[1, N]`
- per-layer flags: near-duplicate pair present (`|rho| > 0.95`), collapsed
  layer (effective count `< 25%` of nominal width), and the count of
  constant neurons (`std < 1e-6`)

The computation is pure forward-only (no backward pass, no weight
perturbation). `MaxSamples` caps the probe count used (default 128) and
`MaxNeurons` caps the per-layer width that gets the full `O(N^2)`
correlation matrix (default 512); wider layers are skipped with a note.

## Build & run

```
cd examples/NeuronCorrelationReport
lazbuild NeuronCorrelationReport.lpi
../../bin/x86_64-linux/bin/NeuronCorrelationReport
```

Total runtime is well under a minute.
