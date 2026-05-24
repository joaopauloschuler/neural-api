# LinearProbeReport

Tiny example for `TNNet.LinearProbeReport`, the linear-probe (linear
separability) diagnostic that answers **"where does the model become a
classifier?"**.

The program builds a small softmax classifier
(`6 -> FC8+ReLU -> FC8+ReLU -> FC8+ReLU -> FC4 -> SoftMax`) on a synthetic
4-class problem (an **XOR-quadrant + concentric-rings** task that is
deliberately **not** linearly separable from the raw input — two of the input
coordinates carry the structure, the rest are noise distractors), then prints
`TNNet.LinearProbeReport(NN, Probes, ValProbes)` on the **same** labelled probe
batch for two networks:

1. **RUN 1**: a freshly-initialised network (no training). The input-layer probe
   sits near the `1/NumClasses` random baseline (the task is not linearly
   separable from the raw input), then the **untrained** random ReLU layers
   *scramble* the signal so probe accuracy **degrades** toward the head, ending
   near-random with collapse (`C`) flags.
2. **RUN 2**: the same architecture after a short training run. The probe
   accuracy is now **preserved and climbs** with depth, holding high all the way
   to the head, and the saturation knee (`S` flag) marks the transfer-learning
   cut point. (The first random ReLU layer already lifts separability in both
   runs — a known random-features effect — but only the trained net keeps that
   separability through the rest of the stack.)

## What it reports

For every intermediate layer the report feeds the whole probe batch forward
(one `NN.Compute` per sample), reads that layer's flat `Output` as a feature row
`x` (a bias `1.0` column is appended), stacks the rows into a design matrix `X`
and the one-hot targets into `Y`, then fits a **closed-form** ridge-regularised
linear probe

```
W = (X^T X + Lambda*I)^-1 X^T Y      (default Lambda = 1e-2)
```

with a self-contained **Double-precision Gauss-Jordan** solve (the `Lambda*I`
ridge term also guards against singular Gram matrices). There is **no SGD loop**
and **no backward pass** — the probe is closed-form and the backbone is pure
forward-only and never modified.

It prints, per layer:

- **(a)** top-1 linear-probe accuracy on the probe batch;
- **(b)** top-1 linear-probe accuracy on the held-out batch (the train/val gap
  flags probes that overfit);
- **(c)** the one-hot regression **MSE** (a smoother per-layer signal than the
  discrete top-1);
- **(d)** the per-layer probe-accuracy delta `acc[k]-acc[k-1]` (the largest
  single jump in linear separability is visible);
- **(e)** a 10-bin ASCII bar chart of per-layer probe accuracy across depth;
- **(f)** per-layer flags: `C` representation collapse (probe acc drops > 5 pt
  vs the previous layer), `S` saturation point (shallowest layer within 1 pt of
  the final layer — the transfer-learning cut point), `R` near-random (probe acc
  within 5 pt of `1/NumClasses`).

### Feature-dimension cap

The Gram solve is `O(D^3)`, so a layer whose flat activation exceeds
`MaxFeatDim` (default **256**) is deterministically random-projected down to
`MaxFeatDim` features (a fixed-seed sparse sign projection) before the probe is
fit, keeping memory and runtime bounded for wide layers. (The tiny layers in
this example stay well under the cap, so no projection fires here.)

## Build & run

```
cd examples/LinearProbeReport
lazbuild LinearProbeReport.lpi
../../bin/x86_64-linux/bin/LinearProbeReport
```

Pure CPU, no dataset download, total runtime well under a minute.
