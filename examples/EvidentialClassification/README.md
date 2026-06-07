# Evidential (Dirichlet) Classification: uncertainty out of distribution

A tiny self-contained demo of **Evidential Deep Learning for classification**
(Sensoy, Kaplan & Kandemir, NeurIPS 2018,
[*Evidential Deep Learning to Quantify Classification Uncertainty*](https://arxiv.org/abs/1806.01768)) —
the classification sibling of the [EvidentialRegression](../EvidentialRegression)
example. It shows that a single deterministic forward pass — no sampling, no
ensemble — produces a calibrated **uncertainty mass** that rises toward 1 where
the model has no data.

## The idea

Instead of a softmax probability vector, the evidential head treats the network
output as **evidence** for a Dirichlet over the class-probability simplex. Over
`K` classes the previous layer's `K` raw channels become Dirichlet concentrations:

```
e_k     = softplus(raw_k)   (>= 0, non-negative evidence)
alpha_k = e_k + 1
```

From these it reads off, in closed form,

```
strength    S   = sum_k alpha_k
probability p_k = alpha_k / S
UNCERTAINTY u   = K / S        in [0, 1]
```

`u → 0` means strong evidence (a confident, in-distribution prediction); `u → 1`
means **no evidence** — the network abstains. It is the analogue of the
regression head's epistemic variance: high where the network accumulated no
evidence.

## The problem

Two well-separated 2-D Gaussian blobs are the in-distribution classes, placed
**side by side in the upper half-plane**: class 0 at `(-2.5, +2.5)` and class 1
at `(+2.5, +2.5)`. Because they **share the "up" direction**, the discriminative
axis is `x` but the entire **lower half-plane** is off *both* manifolds. This
geometry is deliberate: two *antipodal* blobs would tile the plane into two
confident halves, leaving no provable OOD region, whereas side-by-side blobs make
the lower half-plane (and far shell) unambiguously unsupported. A plain softmax
classifier stays confident everywhere; the evidential head reports `u → 1` there.

## What it does

* **Model**: a **saturating (tanh) trunk** `(x,y) → 32 → 32`, then
  `TNNetFullConnectLinear(1,1,2)` → `TNNetEvidentialClassification(K=2, lambda)`.
  The tanh trunk is deliberate: far inputs drive the hidden units flat, so the
  head sees near-constant features and falls back to its **low-evidence prior**
  (high `u`) instead of extrapolating evidence.
* The head maps the trunk's `K` raw channels to `alpha_k = 1 + softplus(raw_k)`
  and **owns the EDL loss**: the Bayes-risk expected-MSE (Eq. 5 of the paper)
  plus a `lambda·KL` regularizer toward the uniform Dirichlet on the
  *misleading-evidence* term `alphaTilde = y + (1-y)·alpha`. Its `Backpropagate`
  emits the exact `dL/d(raw evidence)`, chaining softplus.
* The evidence bias starts **low** (a high-uncertainty prior); accumulated
  gradient at training points then **grows** evidence locally, leaving the lower
  half-plane at its low-evidence prior.

**Training** is hand-rolled mini-batch (8000 updates, batch 64, well under a
minute on two CPU cores, a few MB RAM). The target volume is the **one-hot
label** (the head recovers it from the alpha channels) and the accumulated
batch-update deltas are scaled to the mean gradient. The analytic backward was
checked against finite differences in `tests/TestNeuralNumerical.pas`
(`TestEvidentialClassificationGradient`).

## Example output

```
==== UNCERTAINTY PROBES ====
     (x, y)        pred    probabilities      u       region
  ( -2.50,  2.50)   class 0   p=[0.924 0.076]   u=0.150   in-dist (blob 0)
  (  2.50,  2.50)   class 1   p=[0.076 0.924]   u=0.151   in-dist (blob 1)
  ( -2.50, -4.00)   class 0   p=[0.635 0.365]   u=0.729   OOD (below blob 0)
  (  0.00, -6.00)   class 0   p=[0.500 0.500]   u=1.000   OOD (far below)
  (  0.00, -9.00)   class 0   p=[0.500 0.500]   u=1.000   OOD (far below)
  ( -2.33, -8.69)   class 0   p=[0.508 0.492]   u=0.985   OOD (lower shell)
  (  2.33, -8.69)   class 1   p=[0.495 0.505]   u=0.989   OOD (lower shell)

==== VERDICT ====
  mean uncertainty   in-distribution=0.151   OOD=0.853
  Uncertainty is 5.6x larger on the OOD probes (single deterministic pass, no ensemble).
```

The uncertainty is low (~0.15) on the two training blobs and reaches **exactly
1.0** directly below them, so the mean OOD uncertainty comes out about **5–6×
larger** than in-distribution — all from one deterministic pass.

## Build & run

```
lazbuild EvidentialClassification.lpi
../../bin/x86_64-linux/bin/EvidentialClassification
```

## Related layer

* `TNNetEvidentialClassification` — the single-forward-pass uncertainty head used
  here: maps the previous layer's `K` raw outputs to the Dirichlet concentrations
  `alpha_k = 1 + softplus(raw_k)`, owns the EDL Bayes-risk-MSE + KL loss, and
  exposes `Alpha` / `Prediction` / `Uncertainty` inference helpers. The
  classification sibling of `TNNetEvidentialRegression` (the NIG epistemic
  *regression* head): together they cover evidential uncertainty for both
  regression and classification from one deterministic pass.
```
