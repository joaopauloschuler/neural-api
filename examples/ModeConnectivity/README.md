# ModeConnectivity

Tiny example for `TNNet.ModeConnectivityReport`, the linear
mode-connectivity / loss-barrier diagnostic (Garipov et al. 2018;
Frankle et al. 2020, *Linear Mode Connectivity and the Lottery Ticket
Hypothesis*).

## What it does

It trains the **same** tiny MLP (`2 -> 12 -> 12 -> 3`, ReLU) twice on a
synthetic 3-cluster 2D classification task, then reports the loss along
the straight line `theta(alpha) = (1-alpha)*A + alpha*B` connecting the
two trained solutions A and B for `alpha in [0, 1]`. It runs three times:

1. **RUN 1 - same basin**: both nets start from the **same** random init
   and only differ in batch-shuffle order. Expect a **low** barrier
   ("linearly connected", same loss basin).
2. **RUN 2 - two basins**: the two nets start from **different** random
   inits. Expect a **higher** barrier ("weak barrier" / "separated").
3. **CHECK - self-connectivity**: `B := A`. The curve collapses to a flat
   zero-barrier line — the built-in sanity check.

## What the report shows

`TNNet.ModeConnectivityReport(NN, SnapshotB, Samples, K)` prints:

- the loss curve `L(alpha)` over the `K+1` interpolation points as a
  `#`-bar ASCII chart, with the peak and the two endpoints marked;
- the endpoint losses `L(0)`, `L(1)` and the max loss on the path;
- the **barrier height** `max_alpha L(alpha) - max(L(0), L(1))`
  (`>0` = a bump between the basins; `~0` = linearly connected) and its
  relative size vs the endpoint loss;
- a **faithfulness check**: the endpoint losses recomputed on the
  interpolation path must match a direct forward on each snapshot to
  `< 1e-5` (validates the whole-net snapshot arithmetic);
- a `connected` / `weak barrier` / `separated` verdict.

The live net's weights are restored bit-for-bit at the end — it is a
measurement, never a training step.

## Build & run

```
cd examples/ModeConnectivity
lazbuild ModeConnectivity.lpi
../../bin/x86_64-linux/bin/ModeConnectivity
```

Total runtime is well under a minute.
