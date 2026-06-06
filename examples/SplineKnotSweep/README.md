# Spline Knot-Count / Range Sweep (TNNetSplineActivation capacity study)

A follow-up to [`examples/SplineActivationKAN/`](../SplineActivationKAN/). That
demo showed the per-channel learnable activation `TNNetSplineActivation` (a
Kolmogorov-Arnold Network style piecewise-linear activation, KAN / Liu et al.
2024) can fit a wiggly 1D target. This example asks the obvious next question:
**how many knots do you need, and what does extra capacity buy?**

## What it does

One tiny net is held completely fixed except for two knobs:

```
Input(1) -> FullConnect(8) -> TNNetSplineActivation(K, Range) -> Linear(1)
```

`TNNetSplineActivation(K, Range)` places `K+1` uniformly-spaced control points
(knots) on `[-Range, +Range]` per channel and linearly interpolates between them
(extrapolating linearly outside the range). `K` is the number of intervals:
larger `K` = more knots = a more flexible per-channel activation, costing
`(K+1)*Depth` extra trainable values.

We sweep `K` in `{2, 4, 8, 16}` crossed with `Range` in `{2.0, 4.0}`. **Every
cell uses the identical seed (`424242`), training data, optimizer, width and
epoch budget** — only `K` and `Range` change.

Target:

```
y = sin(3x) + 0.3*sin(11x),   x in [-2, 2]
```

Training inputs are drawn from `[-2, 2]`; we report a separate **held-out MSE**
on a dense clean grid that extends slightly beyond the training span (into
`[-2.4, 2.4]`) so extra knots can overfit inside the span without necessarily
helping outside it.

## Sample run

```
TRAIN MSE  (rows = Range, cols = K):
   Range \ K |         2         4         8        16
       2.0   |  0.065451  0.055766  0.005918  0.021980
       4.0   |  0.079247  0.065451  0.055767  0.005936

HELD-OUT MSE  (rows = Range, cols = K):
   Range \ K |         2         4         8        16
       2.0   |  0.163320  0.116722  0.021353  0.036289
       4.0   |  0.185913  0.163320  0.116719  0.021350
```

## The story

- **More knots help — up to a point.** Going from `K=2` to `K=8` at `Range=2.0`
  cuts held-out MSE roughly 8x (`0.163 -> 0.021`): a low-K spline simply cannot
  represent the two-frequency wiggle, so adding capacity pays off.
- **Then they stop helping (and hurt).** At `K=16` (Range 2.0) the held-out MSE
  *rises* back to `0.036` — the sweet spot is the intermediate `K=8`. This is
  the capacity / overfitting trade.
- **Train error is not even monotone in K** under a fixed SGD budget: `K=16` has
  *more* parameters to optimise and, at the same number of epochs, does not
  reach a lower train loss than `K=8`. So the experiment asserts the *held-out*
  generalization story, not a train-monotonicity claim.
- **Range trades against K.** Doubling `Range` halves the knot density, so e.g.
  `Range=4.0, K=8` reproduces `Range=2.0, K=4` almost exactly — useful resolution
  is governed by `2*Range/K`, not by `K` alone. Match `Range` to your input span.

## Self-checking gate

The program ends with a PASS/FAIL gate (`Halt(1)` on failure) asserting:

1. **knots-help** — at each Range the best held-out MSE over the sweep beats the
   smallest-K held-out MSE by a margin (a low-K spline genuinely under-fits);
2. **endpoints-fit** — both the smallest- and largest-K models drive train MSE
   below a modest threshold (a fair comparison, not a broken run);
3. **knots-stall** — for at least one Range the largest-K held-out MSE is *not*
   better than the sweep best (extra knots stopped helping / hurt).

## Build and run

```
cd examples/SplineKnotSweep
lazbuild SplineKnotSweep.lpi
../../bin/x86_64-linux/bin/SplineKnotSweep
```

Pure CPU, no dataset download, synthetic data generated in-code, well under a
minute.
