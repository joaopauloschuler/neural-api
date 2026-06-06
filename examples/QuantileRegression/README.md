# Quantile (Pinball) Regression

This example demonstrates `TNNetQuantileLoss`, a regression output head that
minimizes the **pinball / quantile loss** instead of mean-squared error. By
training one model per target quantile, you get a *prediction interval* whose
width adapts to input-dependent noise, rather than a single point estimate.

## The pinball loss

For a target quantile `q` in `(0, 1)` and residual `e = (target - prediction)`,
the per-element loss is:

```
L_q(e) = max(q*e, (q-1)*e)
       = q*e         if e >= 0   (under-prediction)
       = (q-1)*e     if e <  0   (over-prediction)
```

`q = 0.5` recovers the median (equivalent to mean-absolute error). Asymmetric
`q` values penalize under- vs over-prediction differently, so the fitted curve
tracks the requested conditional quantile of the data.

`TNNetQuantileLoss` is a `TNNetIdentity` descendant: the forward pass is an
identity passthrough (so `Net.Compute` returns the raw regression output).
During `Backpropagate`, the framework has seeded `FOutputError` with
`(prediction - target) = -e`; the layer overwrites each element with the
analytic subgradient w.r.t. the prediction:

```
dL/dprediction = -q       when e > 0   (FOutputError < 0)
               =  (1 - q)  when e < 0   (FOutputError > 0)
               =  0        at e == 0    (subgradient convention)
```

The quantile `q` is a constructor parameter (default `0.5`) stored in
`FFloatSt[0]`, validated to lie strictly in `(0, 1)`, and round-trips through
`SaveToString` / `LoadFromString`.

## What this demo does

- Builds a **heteroscedastic** 1-D dataset
  `y = sin(x) + 0.3*x + N(0, sigma(x))` where `sigma(x) = 0.10 + 0.45*x`
  grows with `x` (noise gets larger to the right).
- Trains three tiny MLPs (one per quantile `q in {0.1, 0.5, 0.9}`),
  single-threaded for determinism.
- Prints an ASCII chart of the `q=0.5` median (`M`) plus the
  `[q=0.1, q=0.9]` band edges (`#`) and the true mean (`.`). The band visibly
  fans out as `x` increases — exactly the heteroscedastic structure the
  quantile heads are meant to capture.
- Measures **empirical coverage** on a held-out test set: the fraction of test
  targets that land inside the predicted `[q=0.1, q=0.9]` band. For a
  well-calibrated 10%/90% pair this is near 80%. A built-in `PASS/FAIL` check
  asserts coverage lands in a tolerant `[65%, 95%]` window (and `Halt(1)`s on
  failure).

A typical run reports coverage near 80% and finishes in well under a minute on
CPU.

## Building

From this directory (adjust the LazUtils path for your install):

```
LAZUTILS_PATH=/usr/share/lazarus/4.4.0/components/lazutils/lib/x86_64-linux
fpc -B -Fu../../neural -Fu"$LAZUTILS_PATH" -Mobjfpc -Sh -O2 QuantileRegression.lpr
./QuantileRegression
```

Or open `QuantileRegression.lpi` in Lazarus and run it.
