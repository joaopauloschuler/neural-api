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

- A CSV block of `pooling,final_val_loss,final_val_accuracy`.
- A per-arm `train NLL vs epoch` trace.
- All prints are NaN/Inf guarded.

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

Real fragment from a recent run:

```
=== Results (CSV) ===
pooling,final_val_loss,final_val_accuracy
TNNetAvgPool,1.4023,0.2500
TNNetMaxPool,0.0750,1.0000
TNNetLpPool(p=1),1.4025,0.2500
TNNetLpPool(p=2),2.2054,0.2500
TNNetLpPool(p=4),1.3554,0.3531
TNNetLpPool(p=8),0.8506,0.7500
TNNetSoftPool(beta=0.5),1.4194,0.2500
TNNetSoftPool(beta=1),1.4516,0.2500
TNNetSoftPool(beta=2),2.2281,0.2500
TNNetSoftPool(beta=8),1.5963,0.5000

Total wall time: 169.61 s
```
