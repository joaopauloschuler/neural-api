# PermutationAlign

Tiny example for `TNNet.PermutationAlignReport`, the **"Git Re-Basin"**
weight-space neuron-permutation alignment diagnostic (Ainsworth, Hayase &
Srinivasa 2022, *Git Re-Basin: Merging Models modulo Permutation Symmetries*;
Entezari et al. 2021). It is the **dual** of `TNNet.ModeConnectivityReport`.

## What it does

`ModeConnectivityReport` **measures** the linear-interpolation loss barrier
between two independently-trained nets of the same architecture but does nothing
about it. `PermutationAlignReport` goes one step further and shows that most of
that barrier is an **illusion of neuron-labelling**: a hidden layer's units are
interchangeable up to a permutation (permute the units **and**, in the next
layer, the matching input-weight columns, and the represented **function** is
unchanged). Align net B's hidden units to net A's, re-interpolate, and the
straight-line barrier largely **collapses** — both nets sit in the same loss
basin once the permutation symmetry is quotiented out.

This example trains the **same** tiny MLP (`2 -> 12 -> 12 -> 3`, ReLU) twice on
a synthetic 3-cluster 2D classification task from **different** random inits (so
a real barrier exists pre-alignment), then runs three times:

1. **RUN 1 - weight matching** (`ScoreMode = 0`): aligns hidden units by the
   cosine between their weight rows.
2. **RUN 2 - activation matching** (`ScoreMode = 1`): aligns hidden units by the
   correlation of their per-unit activations over the probe batch.
3. **CHECK - align-to-self**: `SnapshotB := A`. Every permutation is the
   identity and the post-alignment barrier is a flat zero — the built-in sanity
   check.

## What the report shows

`TNNet.PermutationAlignReport(NN, SnapshotB, Samples, ScoreMode, K)` prints:

- the per-hidden-layer permutation **churn** (fraction of units the alignment
  moved);
- the loss **barrier before vs after** alignment as two `#`-bar rows scaled to
  the larger barrier, plus the percentage reduction;
- three built-in **PASS/FAIL** correctness checks:
  1. **permutation invariance** — applying any permutation + its next-layer
     column compensation leaves `B.Compute` bit-for-bit unchanged on the probe
     batch (the foundational identity);
  2. **align-to-self** — `SnapshotB := A` yields identity permutations and a
     flat zero barrier;
  3. **monotonicity** — the post-alignment barrier is `<=` the pre-alignment
     barrier (alignment can only help or tie);
- a `barrier collapsed` / `partially reduced` / `unchanged` verdict.

The interpolation sweep reuses `ModeConnectivityReport`'s whole-net snapshot
arithmetic (`theta(alpha) = (1-alpha)*A + alpha*P(B)` via `TNNetVolume.MulMulAdd`).
The live net's weights are restored bit-for-bit at the end — it is a
measurement, never a training step.

## Build & run

```
cd examples/PermutationAlign
lazbuild PermutationAlign.lpi
../../bin/x86_64-linux/bin/PermutationAlign
```

Total runtime is well under a minute.
