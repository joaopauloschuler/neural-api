# Mahalanobis Out-of-Distribution Detector

## What it does

Reproduces, on a TINY pure-CPU synthetic target, the OOD detector of
Lee et al. 2018, *A Simple Unified Framework for Detecting Out-of-Distribution
Samples and Adversarial Attacks* (NeurIPS 2018).

1. **TRAIN** a small softmax MLP
   (`TNNetInput(8) -> FullConnectReLU(16) -> FullConnectReLU(16) ->
   FullConnectLinear(5) -> TNNetSoftMax`) on `K = 5` IN-distribution Gaussian
   blobs in an 8-dim input space. The classifier is then **frozen** — feature
   extraction is forward-only (`NN.Compute`, no backprop).
2. **FIT class-conditional Gaussians** on the **penultimate-layer** feature
   vector `f` (the second `FullConnectReLU`, **layer index 2**, width 16). For
   each training sample we read `NN.Layers[2].Output` after a forward pass and
   accumulate:
   * a per-class mean `mu_c`;
   * a single **shared (tied)** covariance `Sigma`, pooled across all classes
     (one covariance for the whole model, exactly as in the paper).
3. **SCORE** a new point with the maximum **negative squared Mahalanobis
   distance** over classes:
   `score(x) = max_c -(f - mu_c)^T Sigma^-1 (f - mu_c)`.
   IN-distribution points sit close to a class mean (high score); OOD points
   sit far from every mean (very negative score).
4. **REPORT** a single **AUROC** separating held-in test points (positives)
   from OOD points (negatives).

The OOD set is a Gaussian blob shifted far away (`+30` per coordinate) from the
whole in-distribution cloud — a region the classifier never saw.

### Inverting the covariance + the ridge

`Sigma` is a `16x16` SPD matrix. We invert it with a plain **Cholesky**
factorisation (`A = L L^T`, then forward/back-substitution over the columns of
the identity) implemented locally on `array of array of TNeuralFloat`. Before
inverting we add a small **ridge** `+1e-3` on the diagonal
(`Sigma[d][d] += cRidge`) so the factorisation stays well-conditioned even when
a penultimate feature is near-constant after training. This is the standard
shrinkage trick and keeps `Sigma` strictly positive definite.

### Where the AUROC helper lives

There is no AUROC helper anywhere in the repo, and `neuralvolume.pas` exposes
its statistics only as `TVolume` methods (`GetVariance`, `GetStdDeviation`),
with no natural standalone-function home. To avoid touching core, the AUROC
helper is a **local function inside `MahalanobisOOD.lpr`**. It uses the
**Mann-Whitney U / rank-statistic** form: merge positive and negative scores,
sort ascending, assign 1-based **tie-averaged** ranks, then

```
AUROC = (sum of positive ranks - nPos*(nPos+1)/2) / (nPos * nNeg)
```

which equals `P(score(pos) > score(neg))` (ties counted as 1/2).

Because the helper is example-local, this is a **self-checking example**: it
prints `PASS`/`FAIL` and `Halt(1)`s if the observed AUROC on the easy synthetic
split is not `> 0.8`. (There is no separate unit test in `tests/`, since the
helper is not in core.)

## How to run

From this directory:

```
fpc -O3 -Mobjfpc -Sc -Sh -veiq -Fu../../neural -Fu../../neural/pas-core-math MahalanobisOOD.lpr
./MahalanobisOOD
```

(or open `MahalanobisOOD.lpi` in Lazarus and run). Pure CPU, single-threaded
manual training loop, runs in a couple of seconds and a few MB of memory.

## Sample output

```
Architecture (penultimate feature layer = index 2, width 16):
Idx   Layer                            Output Shape                 Params      Neurons
---------------------------------------------------------------------------------------
0     TNNetInput                       (8, 1, 1)                         0            0
1     TNNetFullConnectReLU             (16, 1, 1)                      128           16
2     TNNetFullConnectReLU             (16, 1, 1)                      256           16
3     TNNetFullConnectLinear           (5, 1, 1)                        80            5
4     TNNetSoftMax                     (5, 1, 1)                         0            0
---------------------------------------------------------------------------------------
Totals: 5 layers, 464 weights, 37 neurons

In-distribution classes=5  train/class=200  test/class=200  OOD points=1000

Training the classifier for 60 epochs...
  epoch    1  mean_nll= 0.40385
  epoch   20  mean_nll= 0.00256
  epoch   40  mean_nll= 0.00189
  epoch   60  mean_nll= 0.00182

Mahalanobis OOD detection (Lee et al. 2018, tied covariance)
Feature layer idx=2  dim=16  ridge=0.0010
================================================================
held-in test (positives) n=1000   mean score =     -16.0323
OOD          (negatives) n=1000   mean score =  -10134.5635
----------------------------------------------------------------
AUROC (held-in vs OOD, score = -d_Mahalanobis^2) = 1.0000
================================================================
PASS: AUROC 1.0000 > 0.80 (Mahalanobis cleanly separates in-dist from OOD)
```

On this deliberately easy split the held-in and OOD score distributions do not
overlap at all, so the AUROC is a perfect `1.0000`: the held-in mean Mahalanobis
score (`-16`) is orders of magnitude higher than the OOD mean (`-10134`).

## What the built-in check asserts

* **AUROC > 0.8** on the synthetic split — the program `Halt(1)`s otherwise.
* It also warns if the held-in mean score is not above the OOD mean score
  (a sanity check on the sign of the Mahalanobis score).
