# Local Learning Coefficient (LLC) report

Demonstrates `TNNet.LocalLearningCoefficientReport`, an empirical estimate of the
**Local Learning Coefficient (LLC)** — the Real Log Canonical Threshold (RLCT)
from **Singular Learning Theory** (Watanabe; Lau, Murfet, Wei et al. 2023,
*"Quantifying Degeneracy in Singular Models via the LLC"*).

The LLC measures the **volume-scaling / effective dimensionality** of the minimum
the optimizer settled into. It counts the *flat, degenerate* directions a
second-order Hessian top-eigenvalue cannot see: a redundant / over-parameterised
solution has `LLC_hat << dim(w)` — far fewer **effective** degrees of freedom
than raw weights.

## Estimator (SGLD-based WBIC / free-energy form)

From the trained weights `w*`, run a short tempered, **anchored** Stochastic
Gradient Langevin Dynamics (SGLD) chain that samples the local posterior pinned
to the basin by a Gaussian anchor `(gamma/2)*||w - w*||^2`, then form

```
LLC_hat = n * beta * ( mean_chain[L(w)] - L(w*) )
```

where `L` is the average training NLL over the probe set, `n` is the sample
count and `beta = 1/ln(n)` is the WBIC inverse temperature. The per-step
anchored-Langevin update over a minibatch loss gradient `g` is

```
w <- w - (eps/2)*( n*beta*g + gamma*(w - w*) ) + N(0, eps)
```

The report **reuses the existing forward+backward gradient machinery** (the only
new piece is the anchored update + chain average). It is **non-destructive**:
`w*` is snapshotted and restored bit-for-bit on return.

## What this example shows

Three nets are fit (or not) to the **same** 3-class toy target:

1. **minimal** net, trained to fit;
2. **over-parameterised** net, trained to fit, then two hidden units are *forced
   redundant* (duplicate-then-halve) so two directions are exactly flat;
3. **random-init** net, never trained.

### Robust, reproducible reading

* Both **trained** nets (1) & (2) report a small `LLC_hat` with
  `LLC_hat << dim(w)`: the basin has far fewer effective degrees of freedom than
  raw weights.
* The **random-init** net (3) reports a large, often **negative** `LLC_hat`. The
  LLC is only defined *at a local minimum*; at a random point the anchored chain
  slides downhill so `mean_chain[L] < L(w*)`. A negative / wild `LLC_hat` is the
  diagnostic's honest **"this is not a minimum"** signal.

Run it (pure CPU, ~3 seconds):

```
lazbuild LocalLearningCoefficient.lpi
../../bin/x86_64-linux/bin/LocalLearningCoefficient
```

## Known pitfall — and what did NOT fit the CPU budget

LLC estimates are **sensitive to `eps`, `gamma` and the chain length**. This
example ships **fixed, documented** values (`chain=300, eps=1e-4, gamma=10`); the
**absolute** `LLC_hat` is calibration-dependent and shifts when you change them.
Only the **ordering** of `LLC_hat` across nets measured with the *same* fixed
hyperparameters is meaningful.

The fine-grained **(1)-vs-(2)** ordering (a minimal net below an
over-parameterised one fitting the same function) is real in expectation, but at
the short fixed chain length affordable on a CPU toy the run-to-run estimator
**noise** on the tiny `mean_chain[L] - L(w*)` gap is the same size as that gap,
so the two trained arms do **not** separate reproducibly across random seeds.
The robust headline here is therefore the **trained-vs-untrained** contrast plus
`LLC_hat << dim(w)`. Resolving (1) vs (2) cleanly would need a much longer chain
(and several chains averaged) — beyond this example's budget.

The report itself never steps the weights (a measurement, not training).
