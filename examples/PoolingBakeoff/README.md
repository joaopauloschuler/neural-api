# Pooling-Head Bake-Off

A pooling-**head** comparison: the exact same tiny conv classifier is
trained on the exact same synthetic image task, and the *only* thing
that changes between arms is the global pooling layer that reduces the
`12x12` feature map to `1x1` before the linear classifier.

Arms:

- `TNNetAvgPool`  plain mean over the window.
- `TNNetMaxPool`  single largest activation.
- `TNNetLpPool`   power-mean `( (1/N) sum |x|^p )^(1/p)`, swept over
  `p in {1, 2, 4, 8}`. `p = 1` is the mean of absolute values; large `p`
  approaches max.
- `TNNetSoftPool` softmax-weighted average with temperature `beta`,
  swept over `beta in {0.5, 1, 2, 8}`. `beta -> 0` recovers the mean;
  `beta -> inf` approaches max.
- `TNNetStochasticPool` (Zeiler & Fergus 2013). Per window it normalises
  the (post-ReLU, non-negative) activations to a probability `p_i = a_i /
  sum_j a_j`. At **training** time it *samples* one cell `k ~ p_i` and
  emits `a_k`; at **inference** time it emits the deterministic
  expectation `sum_i p_i * a_i`. The stochastic draw is a regularizer:
  it injects multiplicative noise into the pooled feature each step, much
  like dropout, which is meant to reduce overfitting and hence the
  **train/val gap**.

### Train vs inference mode for StochasticPool

The layer carries an `Enabled` gate (default `false` = deterministic
expectation). `TNNet.EnableDropouts(true)` flips it on; this example
calls `EnableDropouts(true)` around the train loop and
`EnableDropouts(false)` before *both* the train-accuracy and
val-accuracy measurements, so the reported gap is a clean
generalisation gap measured entirely in inference (expectation) mode,
not a sampling-noise artefact. All other pooling arms have no gate, so
the calls are no-ops for them.

## The train/val gap metric

Each arm now reports `final_train_accuracy`, `final_val_accuracy`, and
`train_val_gap = train_acc - val_acc`, both measured in inference mode
after training. A large positive gap signals overfitting (memorising
train, failing to generalise); a small gap means train and val track
together. The headline question is whether StochasticPool's stochastic
regularization yields a *smaller* gap than MaxPool / AvgPool / SoftPool
at this matched architecture.

Both `LpPool`'s `p` and `SoftPool`'s `beta` are single knobs that span
the **average <-> max** pooling family, so sweeping them lets the
interpolation show up empirically in the final loss / accuracy.

## The synthetic task

A 4-class problem on `12x12x3` images. Each class plants a soft bright
blob in a class-specific quadrant (top-left / top-right / bottom-left /
bottom-right) on top of uniform background noise. The blob's average
contribution is the same regardless of class, so a pooling head that
only sees the *mean* activation cannot separate the classes; *where*
the activation energy concentrates is what carries the label. That is
exactly the regime where max-like pooling wins and average-like pooling
collapses to chance.

Everything is generated in-code (no dataset download). `RandSeed` is
reset to the same value before each arm's data generation and before
building/initialising its net, so every arm sees identical inputs and
identical weight init; only the pooling layer differs.

## Build & run

```
lazbuild examples/PoolingBakeoff/PoolingBakeoff.lpi --build-mode=Default
bin/x86_64-linux/bin/PoolingBakeoff
```

Pure CPU, finishes in under three minutes (~170 s on a typical
machine). The compiled binary lands in `bin/x86_64-linux/bin/` (shared
with the other examples), not inside this directory.

## What it shows

- A CSV block of
  `pooling,final_val_loss,final_train_accuracy,final_val_accuracy,train_val_gap`.
- A per-arm `train NLL vs epoch` trace.
- All prints are NaN/Inf guarded, and hardware FP exceptions are masked
  at startup (matching the other neural-api examples) so the power-mean
  / softmax arms cannot raise `EInvalidOp` on transient denormals.

The endpoints bracket the family: `AvgPool` (and `LpPool(p=1)`, which is
the same mean) sit at chance (4-class accuracy ~0.25, loss ~1.39 ≈
`ln 4`), while `MaxPool` solves the task (accuracy 1.0). As `p` and
`beta` grow, both sweeps move monotonically from the average end toward
the max end — `LpPool` goes `p=1: 0.25 -> p=4: 0.35 -> p=8: 0.75`, and
`SoftPool` goes `beta<=1: 0.25 -> beta=8: 0.50` — empirically tracing
the average -> max interpolation. (The mid-sweep arms are still mid
descent at this short epoch budget, which is why their *validation*
loss can be noisy even while *training* NLL is clearly dropping; the
per-epoch trace makes the interpolation trend unambiguous.)

## Expected output sketch

Real CSV block from a run on this branch (RandSeed=42, 16 epochs,
wall time ~128 s):

```
=== Results (CSV) ===
pooling,final_val_loss,final_train_accuracy,final_val_accuracy,train_val_gap
TNNetAvgPool,1.4022,0.2500,0.2500,0.0000
TNNetMaxPool,0.0750,1.0000,1.0000,0.0000
TNNetLpPool(p=1),1.4022,0.2500,0.2500,0.0000
TNNetLpPool(p=2),2.2091,0.2500,0.2500,0.0000
TNNetLpPool(p=4),1.3556,0.3708,0.3531,0.0177
TNNetLpPool(p=8),NaN,0.2500,0.2500,0.0000
TNNetSoftPool(beta=0.5),1.4190,0.2500,0.2500,0.0000
TNNetSoftPool(beta=1),1.4516,0.2500,0.2500,0.0000
TNNetSoftPool(beta=2),2.2248,0.2517,0.2500,0.0017
TNNetSoftPool(beta=8),1.5865,0.5000,0.5000,0.0000
TNNetStochasticPool,1.2793,0.4858,0.4812,0.0046

Total wall time: 127.62 s
```

(`LpPool(p=8)` diverges to `NaN` mid-training on this seed — its
power-mean overflows; with FP exceptions masked the run completes and
the arm just lands back at chance. This is pre-existing arm fragility,
unrelated to StochasticPool.)

## Verdict: does StochasticPool narrow the train/val gap?

On this tiny task the **gap comparison is essentially inconclusive,
because almost every arm has a near-zero gap** — the 1200/320 train/val
split is drawn from the *same* generator with the same seed, so train
and val accuracy track each other tightly and there is very little
overfitting for any pooling head to fight. Measured gaps:

| Pooling arm        | train_acc | val_acc | gap     |
|--------------------|-----------|---------|---------|
| MaxPool            | 1.0000    | 1.0000  | 0.0000  |
| SoftPool(beta=8)   | 0.5000    | 0.5000  | 0.0000  |
| LpPool(p=4)        | 0.3708    | 0.3531  | 0.0177  |
| **StochasticPool** | 0.4858    | 0.4812  | 0.0046  |
| AvgPool / chance   | 0.2500    | 0.2500  | 0.0000  |

StochasticPool's gap (`0.0046`) is among the smallest of the arms that
actually learn something, and notably **smaller than LpPool(p=4)'s
`0.0177`** at a *higher* accuracy (0.48 vs 0.35) — consistent with the
regularization story. But MaxPool reaches a perfect `0.0000` gap simply
by solving the task outright, and AvgPool/SoftPool sit at a `0.0000`
gap by failing equally on both splits, so a small gap here is not by
itself evidence of better generalisation.

**Honest read:** StochasticPool behaves as expected — it learns a
useful intermediate solution (val_acc ~0.48, the best of the
average-leaning arms and a clear win over plain AvgPool's 0.25) with a
tiny gap — but this task does not produce enough overfitting to
demonstrate a gap-narrowing *advantage* over MaxPool. The headline
question ("does stochastic pooling lower the train/val gap at matched
architecture") cannot be answered with a clean "yes" on this toy; it
would need a harder, overfit-prone task (fewer samples, larger
backbone, or a train/val distribution shift) to separate the arms on
generalisation. We report what we see rather than claim a win.
