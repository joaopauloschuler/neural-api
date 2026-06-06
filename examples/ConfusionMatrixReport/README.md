# ConfusionMatrixReport example

This example demonstrates `TNNet.ConfusionMatrixReport`, a classifier
diagnostic helper that runs a single forward pass over a labeled
validation set and prints:

1. the full `C x C` confusion matrix (rows = true class, cols = predicted
   class) with row sums, and a row-normalized variant (per-row recall);
2. per-class precision, recall, F1, plus macro and micro F1;
3. overall top-1 accuracy and balanced accuracy (mean of per-class recall);
4. the top `K` most-confused class pairs `(true_i, pred_j, count, recall_loss)`
   sorted by off-diagonal mass;
5. (optional) per-class lowest-confidence sample indices, capped at `K`,
   so the report doubles as a hard-example miner.

The example trains a small MLP (2 -> 16 -> 16 -> 3 + SoftMax) on a
synthetic 3-class dataset of overlapping 2D Gaussian clusters, then
prints the report on a held-out validation split. Pure-CPU, runs in
a few seconds.

## Build and run

```
lazbuild ConfusionMatrixReport.lpi
./ConfusionMatrixReport
```
