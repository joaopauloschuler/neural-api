# MagnitudePruning

Tiny example for `TNNet.MagnitudePruningReport`, the forward-only **no-retrain
compressibility** diagnostic.

It answers the practitioner's first pruning question directly: *"if I zero the
smallest-magnitude weights, how much can I throw away before the model
breaks?"* — and it answers by **actually pruning and re-running**, never from a
proxy. The recipe is deterministic:

1. snapshot the whole net once with `SaveDataToString`;
2. for each global sparsity level `s` in `{0,10,...,90,95,99}%`, compute the
   magnitude threshold that zeros the smallest `s%` of `|w|` **across all
   trainable layers** (a single **global** percentile — the standard
   global-magnitude criterion), zero every weight with `|w| <= threshold` in
   place, and run **one** forward pass over the probe batch to read the
   resulting loss and (with labels) top-1 accuracy;
3. restore the original weights **bit-for-bit** with `LoadDataFromString`
   before the next level.

The report prints:

- an accuracy-(or loss-)vs-sparsity **ASCII `#`-bar curve**,
- the **prunability knee** (max sparsity whose top-1 drop stays within
  `Tolerance`, default 1%),
- the **per-layer pruned fraction at the knee** (which layers absorb the
  pruning — typically the wide head),
- the **realised-vs-requested** global sparsity at each level (a built-in
  check that the percentile threshold hit its target to within one weight),
- a `highly-compressible` / `moderate` / `fragile` **verdict**.

An optional `PerLayer` flag switches from one global threshold to a per-layer
percentile (the **uniform-per-layer** baseline) so the global-vs-uniform
question is visible side by side. Built-in correctness checks: `s=0%`
reproduces the unpruned loss/accuracy exactly, and the realised sparsity
matches the requested level to within one weight. Weights are restored
bit-for-bit at the end — pure forward-only, never stepped.

This program runs the report on the **same** synthetic 3-class problem three
times so the over-parameterised-is-compressible story shows up in one run:

1. an **over-wide** net (2 hidden layers of width 64) — redundant capacity, so
   it should stay flat to high sparsity (a deep knee → *highly-compressible*);
2. a **tight-fit** net (width 6) — just enough capacity, so accuracy falls
   earlier (a shallower knee);
3. the over-wide net again with the **per-layer (uniform)** criterion, to
   contrast against run (i)'s global threshold.

To stay self-contained and fast, the data is a synthetic linearly-separable
3-class problem (no dataset download); the point is the **shape** of the
accuracy-vs-sparsity curve, not the absolute accuracy.

## Build & run

```
cd examples/MagnitudePruning
lazbuild MagnitudePruning.lpi
../../bin/x86_64-linux/bin/MagnitudePruning
```

Total runtime is well under a minute (in practice ~9 seconds).
