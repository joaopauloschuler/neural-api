# FisherImportance

Tiny example for `TNNet.FisherImportanceReport`, the diagonal-Fisher parameter
importance diagnostic.

The program builds a small softmax classifier
(`6 -> FC24+ReLU -> FC24+ReLU -> FC4 -> SoftMax`) on a synthetic, deliberately
overlapping 4-class Gaussian problem, then prints
`TNNet.FisherImportanceReport(NN, Probes)` on the **same** labelled probe batch
for two networks:

1. **RUN 1**: a freshly-initialised network (no training). The diagonal Fisher
   information is diffuse and the per-layer Fisher mass is spread fairly evenly.
2. **RUN 2**: the same architecture after a short training run. Importance
   redistributes toward the layers doing the discriminative work, the
   per-parameter max Fisher grows, and the right tail of the `log10(Fisher)`
   histogram sharpens.

## What it reports

For a trained classifier and a labelled probe batch, the report estimates the
diagonal (empirical) Fisher information of every trainable parameter

```
F[theta] = E_x[ (d log p(y|x) / d theta)^2 ]
```

by accumulating per-sample **squared** parameter gradients over the batch (one
forward + one backward per sample). The label the gradient is taken w.r.t. is,
per sample, the **true** label (`UseTrueLabel = true`, the default) or the
model's **predicted** label (`UseTrueLabel = false`) — both give the *empirical*
Fisher. The network is **frozen**: its weights are never stepped; only the
transient per-parameter gradient tensors populated by `Backpropagate`
(`Neurons[*].Delta` / `FBiasDelta`, divided back by the layer `LearningRate` to
undo the `-LR` scaling) are read, and deltas are cleared between samples.

It prints:

- **(a)** per trainable layer: total Fisher mass and its share of the network
  total — the "which layers can I least afford to prune" ranking;
- **(b)** per layer: mean and max per-parameter Fisher and a near-zero count
  (params with `Fisher <= ZeroThreshold` are free to prune/reuse);
- **(c)** a 10-bin ASCII histogram of `log10(Fisher)` across all positive-Fisher
  parameters, showing the heavy-tailed structure;
- **(d)** the **effective parameter count** = participation ratio
  `(sum F)^2 / sum F^2` (in `[1, NumParams]`) — a one-number concentration proxy;
- **(e)** per-layer flags: `H` high-importance (top 10% of layer Fisher mass),
  `P` prunable (> `PrunableFrac` of the layer's params near-zero Fisher),
  `D` dead layer (whole-layer Fisher ~ 0).

## Build & run

```
cd examples/FisherImportance
lazbuild FisherImportance.lpi
../../bin/x86_64-linux/bin/FisherImportance
```

Pure CPU, no dataset download, total runtime well under a minute.

## Downstream use: Elastic Weight Consolidation (EWC) two-task sketch

The accumulated per-parameter Fisher is exactly the curvature term an
**Elastic Weight Consolidation** (Kirkpatrick et al. 2017) penalty consumes to
fight catastrophic forgetting. After finishing task A you snapshot the trained
weights `theta_A*` and the diagonal Fisher `F_A` measured by this report, then
while training task B you add a quadratic anchor that pulls each parameter back
toward `theta_A*` in proportion to how important it was to task A:

```
loss_B_total(theta) = loss_B(theta)
                    + (lambda / 2) * sum_i F_A[i] * (theta[i] - theta_A*[i])^2
```

Parameters with large `F_A[i]` (the `H`-flagged layers, the right tail of the
`log10(Fisher)` histogram) are held nearly rigid — they encode task A — while
the near-zero-Fisher parameters (the `Zero%` column, the left tail) are left
free to be repurposed for task B. Concretely:

1. Train on task A; call `TNNet.FisherImportanceReport` (or accumulate the same
   per-parameter `(Delta/LR)^2` sums it computes) over a task-A probe batch to
   get `F_A`, and `SaveDataToString` to snapshot `theta_A*`.
2. Train on task B with the EWC penalty above added to the gradient of each
   weight: `grad += lambda * F_A[i] * (theta[i] - theta_A*[i])`.
3. Re-run the report after task B to confirm the high-Fisher task-A parameters
   barely moved.

The same diagonal-Fisher tensor also drives **Fisher pruning** (drop the
lowest-Fisher weights first) and **natural-gradient**-flavoured preconditioning.
