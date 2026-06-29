# Legendre Memory Unit — orthogonal sliding-window memory vs a diagonal SSM

This example showcases **`TNNetLegendreMemoryUnit`**, a HiPPO-LegS Legendre
Memory Unit that keeps an order-`N` **orthogonal-polynomial projection of a
sliding window** of each input channel, and contrasts it head-to-head with its
diagonal sibling **`TNNetDiagonalSSM`** (one scalar exponentially-decaying state
per channel) on the task that separates them: a **pure continuous-delay (delay
line) reconstruction**.

## The layer

The LMU maintains, per channel, an `N`-coefficient Legendre expansion of the
recent window of length `theta`. Because that window memory is an orthogonal
basis, the value from exactly `D` steps ago is a **fixed linear read-out** of the
coefficients. A diagonal SSM has no notion of "the value exactly `D` steps ago" —
its scalar leaky state can only approximate a delay with a blurred exponential
trace, so it does markedly worse on a pure lag.

## The task — delayed-signal reconstruction

A smooth random 1-D signal `x_t` (sum of two sinusoids with random
phase/frequency/amplitude) is streamed along the sequence axis. The target at
every step is the signal delayed by `cDelay` steps:

```
target_t = x_{t - cDelay}      (0 for t < cDelay)
```

Reconstructing a **pure delay** is exactly what an orthogonal sliding-window
memory makes trivial — read off the Legendre coefficients at the matching lag —
and exactly what a leaky scalar accumulator cannot represent cleanly. The window
length `cTheta` is chosen `>=` the delay so the lag stays in-window. Error is
the mean squared reconstruction error over the steps where the delayed target is
defined (`t >= cDelay`), averaged over held-out sequences.

## The bake-off

Two arms share the **same I/O contract** and differ only in the recurrent mixer.
Both are built from a `1×1` projection, the recurrent layer, and a `1×1`
read-out:

| arm | mixer | front-end projection |
|-----|-------|----------------------|
| LMU         | `TNNetLegendreMemoryUnit(cOrder, cTheta)` | `TNNetPointwiseConvLinear(cModelDim)` |
| DiagonalSSM | `TNNetDiagonalSSM`                        | `TNNetPointwiseConvLinear(cModelDim * cOrder)` |

The DiagonalSSM arm projects to `cModelDim * cOrder` channels so that its total
number of recurrent state scalars **matches** the LMU arm — the diagonal SSM is
state-budget-matched, not starved. Both arms:

```
Input(cSeqLen,1,1)
 -> PointwiseConvLinear(...)        1x1 projection
 -> [ LegendreMemoryUnit | DiagonalSSM ]
 -> PointwiseConvLinear(1)          1x1 read-out
```

Each arm is trained with per-sample SGD (`SetLearningRate(0.001, 0.9)`,
`Compute` + `Backpropagate`) for `cTrainSteps` on the same delay stream, with
seeds reset so both see identical data.

Dimensions: `cSeqLen=32`, `cDelay=6`, `cModelDim=4`, `cOrder=8`, `cTheta=16.0`,
`cTrainSteps=4000`, `cEvalSeqs=400`.

## Running

```
cd examples/LegendreMemoryUnit
fpc -O3 -Mobjfpc -Sh -Fu../../neural LegendreMemoryUnit.lpr
./LegendreMemoryUnit
```

(or open `LegendreMemoryUnit.lpi` in Lazarus). Pure CPU, tiny dimensions;
finishes in well under a minute on 2 cores.

## Expected output

The program prints its configuration, the weight count of each arm, a note that
both arms train on the same delay stream, and the held-out
delayed-reconstruction MSE for each:

```
=== Legendre Memory Unit: delayed-signal reconstruction ===
seq_len=32  delay=6  model_dim=4  order=8  theta=16.0

LMU         params = ...
DiagonalSSM params = ...

training both arms on the SAME delay stream (4000 steps each)...

eval over 400 held-out sequences (delayed-reconstruction MSE):
  LMU (Legendre window) : MSE = ...
  DiagonalSSM (scalar)  : MSE = ...
```

It then asserts its own headline: when the LMU MSE is below the DiagonalSSM MSE
it prints

```
OK: the Legendre window memory reconstructs the delay better than the state-matched diagonal SSM.
```

otherwise it prints a `WARNING:` line. Exact numbers are seed-dependent; the
**contrast** — the orthogonal window memory beating the state-matched diagonal
SSM on a pure delay — is the point.

Coded by Claude (AI).
