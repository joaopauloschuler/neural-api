# Conformal Prediction (split / inductive conformal)

## What it does

Trains a tiny softmax MLP classifier (`TNNetInput -> 2x FullConnectReLU ->
FullConnectLinear -> TNNetSoftMax`) on a synthetic 5-class 2D Gaussian-blob
dataset with deliberate cluster overlap, then wraps the **frozen** model in a
**split (inductive) conformal predictor** that turns its softmax scores into
prediction **SETS** with a finite-sample, distribution-free marginal-coverage
guarantee (Vovk; Angelopoulos & Bates, *A Gentle Introduction to Conformal
Prediction*, 2021).

The data is split three ways: **TRAIN** (fit the model), **CALIBRATION**
(compute the conformal threshold), **TEST** (measure coverage). Using the
**LAC / threshold** nonconformity score `s = 1 - softmax[true_class]`:

1. On the calibration split, score every sample `s_i = 1 - softmax[true_i]`.
2. Take the conformal quantile `qhat` = the `ceil((n+1)(1-alpha))/n` empirical
   quantile of `{s_i}`. If that rank exceeds `n`, `qhat = +inf` and the set is
   all labels (the degenerate-but-valid edge case).
3. At test time emit the set `{ k : 1 - softmax[k] <= qhat }`.

It sweeps `alpha in {0.01, 0.05, 0.10, 0.20}` and prints a table of
`alpha | target-coverage | empirical-coverage | mean-set-size | singleton% | empty%`.

## Why conformal (the guarantee)

The other uncertainty examples in this tree give a **scalar / point** signal:

* `CalibrationReport` — how well a single confidence number matches accuracy
  (ECE / MCE / Brier, temperature scaling);
* `MarginReport` — the per-sample top-1-minus-top-2 logit margin;
* `MCDropoutUncertainty` — stochastic dropout entropy / BALD per input;
* `DeepEnsembleUncertainty` — ensemble predictive-entropy decomposition.

None of them **promises** anything about the true label. Conformal prediction
does: it wraps any frozen model and emits a **variable-size label SET** that
provably contains the true label with probability `>= 1 - alpha` (marginal,
over the data distribution), **distribution-free** and **finite-sample** — no
assumption on the model or the data beyond exchangeability of calibration and
test points. The price is a set instead of a single label: harder inputs get
**larger** sets, easy inputs collapse to a singleton. That set size is itself
an honest, calibrated uncertainty signal.

Built-in correctness signals (the program prints `PASS`/`FAIL` and `Halt(1)`s
on a hard failure):

* **the headline guarantee** — empirical test-set coverage must land at
  `>= 1 - alpha - slack` (`slack = 0.06`) for every alpha in the sweep;
  marginal coverage may slightly exceed `1 - alpha`, which is expected;
* **monotonicity** — mean set size must shrink monotonically as alpha grows
  (looser coverage -> smaller sets).

The test split is 4000 points so the coverage band is meaningful.

## How to run

From this directory:

```
fpc -O3 -Mobjfpc -Sc -Sh -veiq -Fu../../neural ConformalPrediction.lpr
./ConformalPrediction
```

(or open `ConformalPrediction.lpi` in Lazarus and run). Pure CPU,
single-threaded manual training loop, runs in well under 10 seconds and a few
MB of memory.

## Sample output

```
Splits: train=1500  calib=1000  test=4000  classes=5

Split-conformal prediction sets (LAC / threshold score)
Calibration n=1000  Test n=4000
==============================================================================
alpha | target-cov | emp-cov | mean-set | singleton% | empty%
------------------------------------------------------------------------------
 0.01 |      0.990 |   0.992 |    4.282 |       0.0% |   0.0%
 0.05 |      0.950 |   0.960 |    3.916 |       2.3% |   0.0%
 0.10 |      0.900 |   0.900 |    3.654 |       7.7% |   0.0%
 0.20 |      0.800 |   0.798 |    3.135 |      18.6% |   0.0%
==============================================================================
PASS: marginal coverage >= 1-alpha-0.06 across all alpha (the conformal guarantee holds)
PASS: mean set size shrinks monotonically as alpha grows
```

Empirical coverage tracks the `1 - alpha` target on every row, the mean set
size shrinks as alpha grows, and the singleton fraction rises as the
guarantee is loosened — exactly the conformal trade-off.
