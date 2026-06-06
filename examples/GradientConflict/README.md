# GradientConflict

Tiny example for `TNNet.GradientConflictReport`, the per-sample gradient-cosine
batch-conflict diagnostic.

The program trains a small softmax classifier
(`2 -> FC12+ReLU -> FC3 -> SoftMax`) on a clean, linearly-separable 3-cluster
2-D problem, then prints `TNNet.GradientConflictReport(NN, Probes)` on **two**
labelled probe batches measured on the **same frozen** trained net:

1. **RUN 1**: a **clean** linearly-separable 3-cluster batch. Each sample pulls
   the weights toward its (unambiguous) class, so same-class gradients agree
   strongly (the per-class-pair **diagonal** is ~+1) and the **strong-conflict
   tail** (cos < -0.5) is empty.
2. **RUN 2**: a deliberately **noisy / overlapping** batch where 40% of the
   labels are corrupted to a random wrong class and the clusters overlap heavily.
   Two samples sitting in the same region now carry contradictory targets, so
   their gradients oppose: a fat **negative-cosine** tail appears near cos=-1 and
   the strong-conflict fraction climbs sharply.

3. **RUN 3**: the same noisy batch, restricted to the **classifier head**
   (`LayerIdx = 2`) only — the conflict is often concentrated there.

> Note: the cross-class pairs of a softmax head are mildly anti-correlated *by
> construction* (the shared classifier head is pushed in different directions),
> so the raw `cos < 0` fraction is a noisy signal that does not separate clean
> from noisy. The report therefore also prints a **strong-conflict fraction**
> (`cos < -0.5`) — the genuinely-opposed tail — which is the statistic that
> actually distinguishes a clean batch (~0) from a label-noised one.

## What it reports

For a frozen classifier and a labelled probe batch, the report runs, per sample,
one forward + one backward (`ClearDeltas` before each, **never** `UpdateWeights`)
and snapshots that sample's full flattened per-parameter weight-gradient vector
`g_i` — reusing exactly the per-parameter gradient tensors `Backpropagate`
already populates (`Neurons[*].Delta` / `FBiasDelta`, divided back by the layer
`LearningRate` to undo the `-LR` scaling), the same discipline as
`FisherImportanceReport` (no input-gradient enablement, unlike the saliency
report). It then computes the pairwise gradient **cosine** similarity

```
cos(g_i, g_j) = <g_i, g_j> / (||g_i|| ||g_j||)
```

across the batch and prints:

- **(a)** an overall 10-bin ASCII histogram of the pairwise cosines over
  `[-1, 1]` (the `#`-bar style of the sibling reports);
- **(b)** the **conflict fraction**: the share of pairs with `cos < 0` —
  gradients that actively undo each other — plus a **strong-conflict fraction**
  (`cos < -0.5`, the genuinely-opposed tail; see the note above);
- **(c)** the **mean** and **median** pairwise cosine;
- **(d)** the **most-conflicting** sample pair (lowest cosine), a
  "these two examples disagree most" pointer by sample index;
- **(e)** when class labels are decodable, a **per-class-pair mean-cosine
  matrix** (printed numerically and as a glyph heatmap) so a pair of classes
  whose gradients systematically oppose stands out; the diagonal is the mean
  within-class cosine.

`LayerIdx` (default `-1` = all trainable layers) optionally restricts the cosine
to a single trainable layer's gradient slab. The trained weights are **never**
stepped. Built-in correctness checks (also pinned by the smoke test): the
self-cosine `cos(g_i, g_i) = 1` on the diagonal and the cosine matrix is
symmetric.

## Build & run

```
cd examples/GradientConflict
lazbuild GradientConflict.lpi
../../bin/x86_64-linux/bin/GradientConflict
```

Pure CPU, no dataset download, total runtime well under a minute.

## Why it is useful

A high conflict fraction is an early warning that a batch (or a class pair, or a
layer) is asking the optimizer to move in incompatible directions — the
gradient-level signature of label noise, class overlap, or a poorly-conditioned
head. It motivates remedies such as gradient-surgery / PCGrad-style projection,
batch re-balancing, label cleaning, or a smaller learning rate, and it is the
diagnostic counterpart to multi-task gradient-conflict analyses (Yu et al. 2020).
The report is a *measurement* on a frozen net — it never trains or mutates the
backbone.
