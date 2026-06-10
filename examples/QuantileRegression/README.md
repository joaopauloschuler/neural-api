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

## Joint single-model head: `TNNetMultiQuantileLoss`

Training one MLP per quantile is wasteful: you run N forward passes and keep N
models. `TNNetMultiQuantileLoss` is the production form — **one model with an
N-wide output** (`Depth = N`, one channel per target quantile) that predicts
all N quantiles in a **single forward pass**.

- Constructor: `Create(const Quantiles: array of TNeuralFloat)`, e.g.
  `TNNetMultiQuantileLoss.Create([0.1, 0.5, 0.9])`. The parameterless
  `Create()` defaults to `[0.1, 0.5, 0.9]`.
- `N` is stored in `FStruct[0]` and the N quantiles in `FFloatSt[0..N-1]`
  (so `N` is capped at `csNNetMaxParameterIdx + 1 = 8`); the layer round-trips
  through `SaveToString` / `LoadFromString`.
- The SAME scalar target is replicated across all N channels of the target
  volume. `Backpropagate` writes the per-channel pinball subgradient: channel
  `i` (= `Idx mod N`, data is depth-contiguous) uses its own quantile `q_i`,
  identical in form to the single-quantile gradient above.

In this demo the joint head reaches the same `[q=0.1, q=0.9]` interval coverage
(~80%) as the three independent MLPs, but with one model and one forward pass.

## Monotonicity guard (no quantile crossing)

Independently-fit quantiles can **cross** — the predicted `q=0.1` can come out
*above* the predicted `q=0.9`, which is nonsensical. This example ships
**approach (a): a non-differentiable, inference-time sort/clamp helper**:

```pascal
class procedure TNNetMultiQuantileLoss.SortAscending(AOutput: TNNetVolume; N: integer);
```

It sorts each consecutive N-channel group of the output volume ascending in
place, so after the call `out[i] <= out[i+1]` for `q_i < q_{i+1}` — crossings
are *guaranteed* gone. We chose (a) over the soft monotonicity penalty
(approach b) because it is a hard guarantee with zero risk of destabilising the
pinball gradient, it adds no training-time hyperparameter, and it fits the time
budget. The trade-off is honest: it is a post-hoc projection, not learned
monotonicity, and it is non-differentiable (apply it only at inference).

The demo prints the test-set crossing count with and without the guard, and
shows the guard repairing a deliberately crossed triple
`[0.90, 0.40, 0.10] -> [0.10, 0.40, 0.90]`.

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
- Then trains the **joint single-model arm** (`TNNetMultiQuantileLoss`,
  3-wide output) and reports its held-out band coverage, the quantile-crossing
  count with vs without the `SortAscending` guard, and the guard repairing a
  crossed triple.

A typical run reports ~80% coverage for both arms, 0 crossings after the guard,
and finishes in well under a minute on CPU (about 40 s single-threaded).

## Building

From this directory (adjust the LazUtils path for your install):

```
LAZUTILS_PATH=/usr/share/lazarus/4.4.0/components/lazutils/lib/x86_64-linux
fpc -B -Fu../../neural -Fu"$LAZUTILS_PATH" -Mobjfpc -Sh -O2 QuantileRegression.lpr
./QuantileRegression
```

Or open `QuantileRegression.lpi` in Lazarus and run it.
