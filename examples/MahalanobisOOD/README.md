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
     (one covariance for the whole model, exactly as in the paper), AND a
     **per-class (untied)** covariance `Sigma_c` for the ablation below.
3. **SCORE** a new point with the maximum **negative squared Mahalanobis
   distance** over classes:
   `score(x) = max_c -(f - mu_c)^T Sigma^-1 (f - mu_c)`.
   IN-distribution points sit close to a class mean (high score); OOD points
   sit far from every mean (very negative score).
4. **REPORT** an **AUROC** separating held-in test points (positives) from OOD
   points (negatives), for each of the arms below.

### Three OOD arms

| Arm | OOD construction | Covariance | Headline AUROC |
|-----|------------------|-----------|----------------|
| **EASY** | far-away Gaussian blob, `+30` per coordinate (region the net never saw) | tied | **1.0000** |
| **HARD** | near-OOD **boundary blob**: a Gaussian at the **midpoint between two kept in-dist class centres** (classes 0/1), spread `×2.2`, sitting *inside* the convex hull on the decision boundary | tied | **0.9741** |
| **HARD** | same near-OOD boundary blob | **untied** (per-class) | **0.9721** |

The **EASY** arm is the original split, kept as a baseline. Because the held-in
and far-OOD score distributions do not overlap at all, its AUROC pins at exactly
`1.0000` — it is trivially separable and the curve cannot move.

The **HARD** arm is a genuine **near-OOD** split. The class-conditional
Gaussians are fit on only **K−1** in-distribution classes (the last class is
held out), and the negatives are boundary blobs whose penultimate features land
*near* the in-dist manifold rather than far from it. That overlap drops the
AUROC into a **discriminating 0.8–0.99 band** (`0.9741` here), so the ROC curve
actually moves.

### Tied vs untied (per-class) covariance — the ablation

On the HARD split we score the same points two ways:

* **TIED** — a single pooled covariance `Sigma` shared across all classes
  (Lee et al.'s default), `score(x) = max_c −(f−mu_c)^T Sigma^-1 (f−mu_c)`.
* **UNTIED** — a separate per-class covariance `Sigma_c`,
  `score(x) = max_c −(f−mu_c)^T Sigma_c^-1 (f−mu_c)`.

The program prints both AUROCs and their delta. Here **tied wins** by a small
margin (`0.9741` vs `0.9721`, Δ = `−0.0020`): with only `200` samples per class
the per-class covariances are estimated from less data and are noisier, so
pooling across classes (the paper's default) is the more robust estimator on
this tiny target. The delta is small because the clusters are near-isotropic;
untied pays off only when class covariances genuinely differ in shape.

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
prints `PASS`/`FAIL` and `Halt(1)`s unless **all three** of the following hold:
the EASY AUROC is `> 0.8`, the HARD (tied) AUROC is `> 0.8`, and the HARD (tied)
AUROC is **strictly below `1.0`** (so the near-OOD split is verified to actually
be harder, not silently separable). (There is no separate unit test in
`tests/`, since the helper is not in core.)

## How to run

From this directory:

```
fpc -O3 -Mobjfpc -Sc -Sh -veiq -Fu../../neural MahalanobisOOD.lpr
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

In-distribution classes=5  train/class=200  test/class=200  far-OOD points=1000
HARD split: Gaussians fit on classes 0..3 (class 4 held out); near-OOD = boundary blobs at the midpoint of classes 0/1

Training the classifier for 60 epochs...
  epoch    1  mean_nll= 0.40385
  epoch   20  mean_nll= 0.00256
  epoch   40  mean_nll= 0.00189
  epoch   60  mean_nll= 0.00182

Mahalanobis OOD detection (Lee et al. 2018)
Feature layer idx=2  dim=16  ridge=0.0010
================================================================
EASY  far-away OOD (+30/coord), tied covariance:
  held-in (pos) n=1000  mean score =       -16.0323
  OOD     (neg) n=1000  mean score =    -10134.5635
  AUROC = 1.0000
----------------------------------------------------------------
HARD  near-OOD boundary blob (midpoint classes 0/1), TIED covariance:
  held-in (pos) n= 800  mean score =       -15.5156
  OOD     (neg) n= 200  mean score =      -100.5866
  AUROC = 0.9741
----------------------------------------------------------------
HARD  near-OOD boundary blob (midpoint classes 0/1), UNTIED (per-class) covariance:
  held-in (pos) n= 800  mean score =       -16.3318
  OOD     (neg) n= 200  mean score =      -300.2333
  AUROC = 0.9721
================================================================
AUROC summary:
  EASY (far-OOD, tied)          = 1.0000
  HARD (near-OOD, tied)         = 0.9741
  HARD (near-OOD, untied)       = 0.9721
  delta (untied - tied), HARD   = -0.0020
  -> TIED (pooled) covariance wins on the HARD split
================================================================
PASS: EASY AUROC 1.0000 > 0.80, HARD AUROC 0.9741 in discriminating band (0.80, 1.0)
```

On the **EASY** split the held-in and far-OOD score distributions do not overlap
at all, so the AUROC is a perfect `1.0000` (held-in mean `−16` vs OOD mean
`−10134`, orders of magnitude apart). On the **HARD** near-OOD split the boundary
blobs sit close to the manifold (OOD mean only `−100`), the distributions
overlap, and the AUROC drops into the discriminating band (`0.9741`). Tied edges
out untied (`0.9741` vs `0.9721`).

## What the built-in check asserts

* **EASY AUROC > 0.8** — the far-OOD split stays clearly separable.
* **HARD (tied) AUROC > 0.8** — the near-OOD split is still discriminating.
* **HARD (tied) AUROC < 1.0** — the near-OOD split is verified to be *actually
  harder* (not silently separable); the program `Halt(1)`s if it pins at 1.0.
* Each arm also warns if its held-in mean score is not above its OOD mean score
  (a sanity check on the sign of the Mahalanobis score).
