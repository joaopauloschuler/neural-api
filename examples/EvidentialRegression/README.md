# Deep Evidential Regression: epistemic uncertainty out of distribution

A tiny self-contained demo of **Deep Evidential Regression** (Amini et al.,
NeurIPS 2020, [*Deep Evidential Regression*](https://arxiv.org/abs/1910.02600))
on a 1-D function with a **held-out region** the model never sees during training.
It shows that a single deterministic forward pass — no sampling, no ensemble —
produces an **epistemic (model) uncertainty** that spikes where the model has no
data.

## The idea

Instead of predicting a point estimate, the evidential head places a
**Normal-Inverse-Gamma (NIG)** prior over the Gaussian likelihood and emits its 4
parameters per scalar target:

```
gamma (mean),  nu (>0),  alpha (>1),  beta (>0)
```

From these it reads off, in closed form,

```
prediction    = gamma
aleatoric var = beta / (alpha - 1)              (irreducible data noise)
epistemic var = beta / (nu * (alpha - 1))       (model / knowledge uncertainty)
```

The **epistemic** variance is the interesting one: it grows where the evidence
parameter `nu` is small, i.e. where the network accumulated no evidence. This
mirrors Figure 3 of the paper (cubic regression: epistemic uncertainty explodes
outside the training support).

## The problem

We learn `f(x) = sin(1.5·x) + 0.2·x` (plus small noise on `y`) from inputs drawn
**only** from the central band `[-3, +3]`. The outer tails `x < -3` and `x > +3`
are **held out**. A plain regression head gives no hint that the tails are
unsupported; the evidential head reports rising epistemic uncertainty there.

## What it does

* **Model**: a **saturating (tanh) trunk** `x → 64 → 64`, then
  `TNNetFullConnectLinear(1,1,4)` → `TNNetEvidentialRegression(D=1, lambda)`. The
  tanh trunk is deliberate: far out-of-distribution inputs drive the hidden units
  into their flat regions, so the head sees near-constant features and falls back
  to its **low-evidence bias prior** (high epistemic) instead of *linearly
  extrapolating* the NIG params — which would make the OOD signal meaningless.
* The head maps the trunk's 4 raw channels to `(gamma, nu, alpha, beta)` via
  softplus links (`gamma` linear, `nu`/`beta` = softplus, `alpha` = 1 + softplus)
  and **owns the NIG negative-log-likelihood + evidence-regularizer loss**; its
  `Backpropagate` emits the exact `dL/d(raw param)`.
* The `nu` output bias starts **low** (a high-uncertainty prior); accumulated
  gradient at training points then **grows** `nu` locally, leaving the held-out
  tails at their low-evidence prior.

**Training** is hand-rolled mini-batch maximum likelihood (6000 updates, batch 64,
about **1 minute** on two CPU cores, ~4 MB RAM). We build the target volume so its
`gamma` channel holds the true `y` (the head recovers `y` from there) and scale the
accumulated batch-update deltas to the mean gradient. The analytic backward was
checked against finite differences in `tests/TestNeuralNumerical.pas`
(`TestEvidentialRegressionGradient`).

## Example output

```
EvidentialRegression: f(x) = sin(1.5x) + 0.2x, trained on a central band
  trunk 64->64->4 raw NIG params, lambda=0.010

Training deep evidential regression (NIG NLL + evidence reg)...
  epoch    1   avg NIG loss=   2.2378
  epoch 6000   avg NIG loss=  -0.8...

==== EPISTEMIC UNCERTAINTY SWEEP (trained on [-3.0, 3.0]) ====
     x        pred      true      aleatoric   epistemic   region
   -6.000     ...                              0.0119   HELD-OUT
    ...
    0.000    -0.045     0.000      0.0021      0.0013   train
    ...
==== VERDICT ====
  mean epistemic var   in-distribution=0.0025   held-out=0.0053
  Epistemic uncertainty is 2.1x larger in the held-out tails (single
  deterministic pass, no ensemble).
```

The epistemic variance is lowest in the densely observed band centre and rises in
the held-out tails (deepest in the far-left tail, ~5× the band centre), so the
mean held-out epistemic uncertainty comes out about **2× larger** than
in-distribution — all from one deterministic pass.

## Build & run

```
lazbuild EvidentialRegression.lpi
../../bin/x86_64-linux/bin/EvidentialRegression
```

## Related layer

* `TNNetEvidentialRegression` — the single-forward-pass uncertainty head used
  here: maps the previous layer's `4·D` raw outputs to the NIG parameters
  `(gamma, nu, alpha, beta)` per target, owns the NIG NLL + evidence-regularizer
  loss, and exposes `Prediction` / `AleatoricVar` / `EpistemicVar` inference
  helpers. Distinct from `TNNetMixtureDensity` (aleatoric-only Gaussian mixture)
  and `TNNetKalmanFilterCell` (recursive sequential covariance): it is the only
  head producing a **closed-form epistemic** estimate from one deterministic pass.
```
