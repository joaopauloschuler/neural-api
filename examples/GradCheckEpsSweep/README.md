# Numerical-gradient epsilon sweep (the finite-difference U-curve)

A tiny, self-contained, **no-download** diagnostic that shows *why* the
gradient tests in
[`tests/TestNeuralNumerical.pas`](../../tests/TestNeuralNumerical.pas) pick the
finite-difference step size (`eps`) they do.

It takes one well-tested layer, runs the **exact same central-difference
gradient check** the test suite uses, and sweeps the step size

```
eps in {1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7}
```

printing, per `eps`, the **max absolute** and **max relative** error between the
finite-difference estimate and the analytic gradient from `Backpropagate`.

## The idea: truncation vs round-off

To check a gradient numerically we approximate the derivative of the loss `L`
with respect to one parameter `w` by a **central difference**:

```
g_num(eps) = ( L(w + eps) - L(w - eps) ) / (2*eps)
```

and compare it to the analytic gradient the network reports. The error in this
estimate is the sum of two competing terms:

- **Truncation (discretisation) error** — the central-difference formula drops
  the Taylor terms beyond first order. The leading one is
  `O(eps^2 * L''')` (it involves the *third* derivative of `L`). It **grows with
  eps**: make the step bigger and the straight-line secant fits the curve worse.
- **Round-off (cancellation) error** — `L(w+eps)` and `L(w-eps)` are nearly
  equal numbers; subtracting them throws away most significant digits, and that
  lost precision is then divided by the tiny `2*eps`. This term scales like
  `O(machine_eps / eps)` and **blows up as eps shrinks**.

Add them up and the total error is a **U-shaped curve**: too-big `eps` is
dominated by truncation, too-small `eps` is dominated by round-off, and there is
a sweet spot in between.

> A purely **linear** layer with a quadratic (MSE) loss has `L''' = 0`, so its
> truncation term vanishes and you only ever see the round-off arm. To make the
> *full* U visible this example puts a **`TNNetHyperbolicTangent`** head after a
> linear layer, so the loss is genuinely nonlinear and the large-`eps`
> truncation arm reappears.

## The FP32 caveat (this is the whole point)

`TNeuralFloat` is **single precision (FP32)** in this repo. Machine epsilon for
FP32 is ~`1.2e-7`, versus ~`2.2e-16` for FP64. The round-off arm is governed by
`machine_eps / eps`, so in FP32 it kicks in **much earlier** — the curve bottoms
out around `eps ~ 1e-2 .. 1e-3` and is already badly contaminated by `1e-5`.
In a float64 build the sweet spot would sit far lower (around `eps ~ 1e-5`).

That early bottom is exactly why the test suite probes at `eps = 1e-4` (still on
the good side of the round-off cliff) and asserts with a generous `~0.01`
tolerance instead of demanding 6 significant digits: in FP32 you simply cannot
do better than a few thousandths of relative error from a central difference, no
matter how you pick the step.

## What the program does

- Net: `TNNetInput(4)` -> `TNNetFullConnectLinear(3, suppressBias=1)` ->
  `TNNetHyperbolicTangent`.
- A fixed, hand-set input and target and hand-set linear weights — **no RNG at
  all**, so the output is bit-reproducible (no `RandSeed` needed).
- Loss: `0.5 * sum((y - d)^2)` (the same MSE the test idiom uses).
- For each `eps` it central-differences the loss with respect to every weight of
  the first neuron **and** every input element, compares against the analytic
  weight `Delta` / input `OutputError` from `Backpropagate`, and records the max
  abs / rel error.

It reuses the established idiom from `CellLayerGradientCheck` in
`tests/TestNeuralNumerical.pas` verbatim (`SetBatchUpdate(true)`,
`ClearDelta`, analytic weight grad = `-Delta`, analytic input grad =
`Layers[0].OutputError`).

This is forward + backward only — no training loop — and finishes in well under
a second.

## How to run

```bash
cd examples/GradCheckEpsSweep
lazbuild GradCheckEpsSweep.lpi
../../bin/x86_64-linux/bin/GradCheckEpsSweep
```

Or compile directly with FPC (point `-Fu` at your LazUtils dir, the one that
contains `utf8process.ppu`):

```bash
cd examples/GradCheckEpsSweep
fpc -B -Mobjfpc -Sh -O2 -Fu../../neural \
    -Fu/usr/share/lazarus/<ver>/components/lazutils/lib/x86_64-linux \
    GradCheckEpsSweep.lpr
./GradCheckEpsSweep
```

## Sample output

Actual output on an FP32 build:

```
Numerical-gradient epsilon sweep (central differences).
Net: FullConnectLinear(3) -> HyperbolicTangent, fixed 4-element input,
     0.5*sum((y-d)^2) MSE loss. Tanh makes the loss genuinely nonlinear,
     so BOTH arms of the U-shape (truncation + round-off) are visible.
TNeuralFloat is FP32 in this build, so the round-off arm appears early.
Formula: g_num(eps) = ( L(w+eps) - L(w-eps) ) / (2*eps), compared to Backprop.

      eps        max_abs_err     max_rel_err
  -----------   -------------   -------------
     1.0E-001    8.78483E-004    1.32186E-001
     1.0E-002    8.27247E-006    1.24476E-003   <-- min (sweet spot)
     1.0E-003    1.62944E-005    1.77778E-003
     1.0E-004    1.31613E-004    1.83525E-002
     1.0E-005    8.92676E-004    2.72941E-001
     1.0E-006    1.17287E-002    1.00000E+000
     1.0E-007    1.25075E-001    5.22535E+000

Minimum max-abs error at eps = 1.0E-002 (abs=8.27E-006, rel=1.24E-003).

Large eps -> O(eps^2) truncation error;  tiny eps -> round-off (cancellation).
The FP32 sweet spot is why TestNeuralNumerical.pas uses eps = 1e-4.
```

(Exact digits vary a little with platform / float build, but the shape is
robust.)

## Reading the result

- **Left arm (large eps):** at `1e-1` the error is `~9e-4` — pure truncation.
  Note it falls by ~100x going to `1e-2`, the textbook `O(eps^2)` behaviour
  (10x smaller step -> ~100x smaller truncation error).
- **The bottom:** the minimum sits at `eps ~ 1e-2 .. 1e-3` (`~8e-6` absolute).
  This is the FP32 sweet spot.
- **Right arm (tiny eps):** from `1e-4` down the error climbs roughly 10x per
  decade — pure cancellation (`machine_eps / eps`). By `1e-6` the relative error
  is `~1.0` (the estimate is essentially noise) and by `1e-7` it is meaningless.
- **Why the tests use `eps = 1e-4`:** it is comfortably on the *flat, accurate*
  side of the U (abs error `~1.3e-4`, well under the `~0.01` assertion
  tolerance) while still being big enough that FP32 cancellation has not yet
  taken over. Push it to `1e-6` and the same test would fail not because the
  analytic gradient is wrong but because the *numerical reference* has gone bad.

That last point is the practical lesson: a failing finite-difference gradient
check at very small `eps` is often the **checker** breaking down, not the
backprop — always confirm your step size is on the good side of this curve.
