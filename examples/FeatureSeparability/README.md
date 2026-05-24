# FeatureSeparability

Tiny example for `TNNet.FeatureSeparabilityReport`, a label-aware
class-**geometry** / **Neural-Collapse** diagnostic (Papyan, Han & Donoho 2020)
that answers **"how tightly does each layer cluster the samples of a class, and
how far apart are the classes?"** — the *geometry* of the feature space, not its
decodability (that is what the sibling `LinearProbeReport` measures).

The program builds a small softmax classifier
(`6 -> FC10+ReLU -> FC10+ReLU -> FC10+ReLU -> FC3 -> SoftMax`) on a synthetic
3-class **2-D Gaussian-blob** problem (the three class centres sit on an
equilateral triangle; the remaining input coordinates are noise distractors),
trains it briefly, then prints `TNNet.FeatureSeparabilityReport(NN, Probes, 3)`
on a class-balanced probe batch for **two variants run back to back**:

1. **RUN 1 — well-separated blobs**: centres far apart, tight spread. High Fisher
   ratio `tr(Sb)/tr(Sw)`, silhouette near 1, and final-layer off-diagonal
   class-mean cosines approaching the simplex-ETF target `-1/(K-1) = -0.5`.
2. **RUN 2 — overlapping blobs**: centres close, large spread. Low Fisher ratio,
   low silhouette, class means poorly separated.

In both, the Fisher ratio and silhouette **climb toward the penultimate layer**
(the trained ReLU stack progressively tightens the per-class clusters), but the
well-separated variant climbs much higher — the visible contrast.

## What it reports

For every trainable layer (one `NN.Compute` per sample, flat `Output` per layer)
it computes the Fisher-style scatter decomposition over the classes:

```
mu_c        = mean activation of class c,   mu = global mean
tr(Sw)      = mean_c mean_{i in c} ||x_i - mu_c||^2   (within-class, NC1 tightness)
tr(Sb)      = mean_c ||mu_c - mu||^2                  (between-class spread)
tr(Stot)    = mean_i ||x_i - mu||^2  ==  tr(Sw) + tr(Sb)   (balanced batch: exact)
Fisher ratio = tr(Sb) / tr(Sw)                        (higher = cleaner clusters)
```

plus the mean **silhouette** coefficient over the batch (fit-free cohesion-vs-
separation in `[-1,1]`) and the **class-mean pairwise-cosine** matrix (numeric +
glyph heatmap) with the **simplex-ETF check** (NC2): the mean off-diagonal cosine
printed next to its neural-collapse target `-1/(NumClasses-1)`.

Per-layer flags: `C` collapse (`tr(Sw)` near zero — a tightly collapsed cluster),
`S` well-separated (Fisher ratio `>= SeparatedRatio`, default 1.0), `R`
near-random (Fisher ratio `<= RandomRatio`, default 0.05). The report is pure
forward-only — `NN.Compute` only, no classifier fit, no backward pass.

### Feature-dimension cap

A layer whose flat activation exceeds `MaxFeatDim` (default **256**) is
deterministically random-projected down to `MaxFeatDim` features (a fixed-seed
sparse sign projection) before the scatter is accumulated, bounding cost on wide
layers. The tiny layers here stay under the cap, so no projection fires.

## Build & run

```
cd examples/FeatureSeparability
lazbuild FeatureSeparability.lpi
../../bin/x86_64-linux/bin/FeatureSeparability
```

Pure CPU, no dataset download, total runtime well under a minute.
