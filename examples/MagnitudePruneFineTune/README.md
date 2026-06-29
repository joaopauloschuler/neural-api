# MagnitudePruneFineTune — persistent pruning + fine-tune recovery

Demonstrates **persistent magnitude-based weight pruning** with accuracy recovery
via **fine-tuning**. It trains a small synthetic 4-class classifier, prunes the
smallest-magnitude weights to a target sparsity with a **mask that persists**,
shows the accuracy drop right after pruning, then recovers it by fine-tuning —
because the mask re-applies the zero constraint after *every* weight update
(unlike temporary pruning, which restores weights afterwards).

## What it uses

- A `TNNet` MLP: `TNNetInput` → two `TNNetFullConnectReLU(32)` hidden layers →
  `TNNetFullConnectLinear(4)` → `TNNetSoftMax`.
- `NN.PruneWeightsByMagnitude(sparsity, PerLayer=False)` installs the persistent
  global mask; `NN.GetPruneSparsity` and `NN.CountPrunedWeights` report it.
- Manual training with `NN.Compute` / `NN.Backpropagate` on synthetic data
  (noisy class prototypes with XOR-like nonlinear labeling).

## What it does

For each target sparsity (50%, 80%, 90%) it: trains a fresh dense net (40 epochs),
measures the dense baseline accuracy on a held-out 300-sample probe set, prunes to
the target, measures accuracy immediately after pruning, fine-tunes (30 epochs)
with the mask re-applied each step, and measures the recovered accuracy —
confirming the mask still holds the weights at zero.

## Running

No arguments, no dataset, no download — all data is synthetic. Pure CPU, a few
seconds.

```
cd examples/MagnitudePruneFineTune
# build with lazbuild (or fpc), then:
./MagnitudePruneFineTune
```

For each sparsity level it prints `[1]` dense baseline, `[2]` weights pruned and
realised sparsity, `[3]` accuracy right after pruning (change vs dense), and `[4]`
accuracy after fine-tuning (points recovered), plus a one-line `SUMMARY` of
dense → pruned → fine-tuned.

Coded by Claude (AI).
