# WeightDriftReport — per-layer weight-drift diagnostic

A tiny, pure-CPU demo of `TNNet.WeightDriftReport`, the diagnostic that compares
**two snapshots of the same network** and reports, per layer, how far the weights
moved during training. It is the canonical way to confirm a layer was actually
**frozen** (e.g. `LearningRate := 0` on a backbone) or, conversely, that an update
exploded.

```
snapshot A (before training)  ->  train  ->  snapshot B (after training)
  TNNet.WeightDriftReport(A, B)
  -> per-layer L2 drift + fraction-frozen
```

## What it does

`BuildNet` stacks a 6-layer ReLU MLP — `TNNetInput(2,1,1)` ->
four `TNNetFullConnectReLU(16)` -> `TNNetFullConnectLinear(1)` — and trains it for
200 epochs on a tiny synthetic **hypotenuse** task (`y = sqrt(a² + b²)` for two
random inputs). Before training, ONE hidden layer (index 3) is pinned with
`NN.Layers[cFrozenLayerIdx].LearningRate := 0`.

The network is snapshotted with `NN.SaveToString()` **before** and **after**
training, and the two strings are passed to `TNNet.WeightDriftReport(SnapA, SnapB)`,
whose text report is printed. The frozen layer should show **~0 L2 drift and a
~1.0 frozen fraction**, while every surrounding trainable layer shows non-trivial
drift.

## Build / run

No arguments, no download, no dataset — fully offline and runs in seconds.

```
cd examples/WeightDriftReport
# build with lazbuild (or fpc), then:
./WeightDriftReport
```

`RandSeed := 42` makes the run deterministic.

## Expected output

It prints the architecture (`PrintSummary`, noting layer 3 is frozen), a few
training MSE checkpoints (epochs 1/50/100/150/200), then the boxed
`WeightDriftReport` table followed by the reminder that layer 3 should read ~0 L2
drift / ~1.0 frac frozen while the other trainable layers drift.

Coded by Claude (AI).
