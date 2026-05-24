# GradientNoiseScale

Demonstrates `TNNet.GradientNoiseScaleReport`: a **forward + backward, no-weight-update**
*gradient signal-to-noise* diagnostic that analytically **predicts the batch-size
sweep** — the critical batch size beyond which larger batches stop buying faster
convergence (McCandlish et al. 2018, *An Empirical Model of Large-Batch Training*).

## What it does

1. Builds a tiny 3-class softmax MLP (`Input -> FullConnectReLU -> FullConnectLinear -> SoftMax`)
   on a synthetic 2-D problem and trains it briefly on a clean, well-separated set
   so the weights are sensible.
2. Prints `TNNet.GradientNoiseScaleReport` on two labelled probe batches:
   - **RUN 1 — clean, linearly-separable**: per-sample gradients agree (high SNR),
     so `B_simple` is tiny and even a batch of 1 is already near-optimal.
   - **RUN 2 — noisy / overlapping + label-corrupted**: gradients scatter (low SNR),
     so `B_simple` is large and a bigger batch genuinely helps.
3. **RUN 3** restricts every statistic to the classifier head (`LayerIdx`) — head and
   stem usually have different noise scales.
4. **RUN 4** runs a quick *empirical* batch-size sweep on the noisy problem (fixed
   compute budget) so the predicted `B_simple` can be eyeballed against reality.

Pure CPU, no dataset download, runs in well under a minute.

## What the report shows

On a **frozen** net (`ClearDeltas` before each sample, never `UpdateWeights`) it
runs one forward + one backward per labelled sample, snapshots that sample's full
flattened per-parameter weight-gradient vector `g_i`, then forms the mean gradient
`g_bar` and the per-parameter gradient variance across samples, and reports:

- the per-parameter **gradient SNR** `|g_bar_k| / (std_k + eps)` as a 10-bin ASCII
  histogram plus a per-layer mean (which layers carry a clean signal vs noise);
- the **simple noise scale** `B_simple = tr(Sigma) / ||g_bar||^2` (`Sigma` = the
  per-sample gradient covariance) — the McCandlish critical batch size;
- the **effective-batch curve** `noise(B) = B_simple / B`, a noise-vs-batch table so
  the sweet-spot batch size is readable directly;
- per-layer flags (signal-dominated / noise-dominated, and the layer with the
  largest noise scale — the one that most wants a bigger batch).

An optional `LayerIdx` restricts every statistic to one trainable layer's gradient
slab. Built-in correctness checks: feeding the **same** sample N times drives the
variance term (and hence `B_simple`) to ~0 (identical gradients = pure signal); a
single-sample batch emits a clear "need >= 2 samples" message rather than dividing
by zero. The weights are never stepped — this is a measurement, not training.

## Running

```
cd examples/GradientNoiseScale
lazbuild GradientNoiseScale.lpi
../../bin/<arch>/bin/GradientNoiseScale
```

Or directly with fpc:

```
cd examples/GradientNoiseScale
fpc -B -Fu../../neural -Mobjfpc -Sh -O2 GradientNoiseScale.lpr
./GradientNoiseScale
```
