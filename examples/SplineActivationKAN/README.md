# SplineActivationKAN

A **KAN-vs-MLP** toy-fit micro-experiment for the library's per-channel learnable
activation [`TNNetSplineActivation`](../../neural/neuralnetwork.pas) — a
Kolmogorov-Arnold-Network-style piecewise-linear activation (KAN, Liu et al.
2024). It puts the headline KAN claim on a tiny, fully synthetic 1D regression
and reports the numbers:

> A **learnable activation** buys **lower final loss** at a **fixed width /
> matched parameter count**.

## The fit

Target (two incommensurate frequencies so a single ReLU MLP can't nail it cheaply):

```
y = sin(3x) + 0.3*sin(11x),   x in [-2, 2]
```

Two arms, trained the **same** number of epochs on the **same** data with the
**same** optimizer:

```
Arm A (baseline MLP):  Input(1) -> FullConnectReLU(Wa) -> FullConnectLinear(1)
Arm B (KAN-flavored):  Input(1) -> FullConnect(Wb)
                                -> TNNetSplineActivation(K=4, Range=2.0)
                                -> FullConnectLinear(1)
```

## Matched parameter count

`TNNetSplineActivation` adds `(K+1)*Depth` trainable control-point values per
layer (here `5*Wb`) while `ReLU` adds **zero**. To keep this a fair fixed-budget
fight the ReLU arm is made **wider** (`Wa > Wb`) so both arms have ~equal total
trainable weight counts. The example does **not** hand-compute this: it builds
the spline arm, reads its exact `TNNet.CountWeights()`, then searches for the
ReLU hidden width whose `CountWeights()` best matches, and **prints** both counts
so the match is auditable (here 21 vs 20).

## The learned spline shapes

An **untrained** `TNNetSplineActivation` is an **exact identity map** (its control
points initialize on `y = x`). After training, the example reads the spline layer
back **by index** (the robust idiom — `TNeuralFit.Fit` reloads the best model and
invalidates held layer references; this example uses a hand-rolled loop but
indexes anyway) and dumps, for the most-bent channels:

- the learned control-point values `y[i,c]` at the fixed knots `t[i]`;
- the activation sampled over `x in [-Range, +Range]`.

Seeing the curve **bent away from `y = x`** is a clean built-in check that the
activation actually learned a nonlinear shape.

The fit turns out to be **sparse**: only one hidden channel learns a non-trivial
bend and the rest stay flat/dead — yet that single learnable activation is enough
to beat the matched-param ReLU arm by a wide margin.

## Sample output

```
SplineActivationKAN: KAN-vs-MLP toy fit at MATCHED parameter count.
Target  y = sin(3x) + 0.3*sin(11x)  over x in [-2.0, 2.0].
Arm A: ReLU MLP (wider).  Arm B: SAME MLP with ReLU -> TNNetSplineActivation (per-channel learnable piecewise-linear).

Spline arm: FullConnect(8) -> SplineActivation(K=4,Range=2.0) -> Linear(1)   =>  21 trainable weights
ReLU   arm: FullConnectReLU(10) -> Linear(1)                          =>  20 trainable weights  (width chosen to match)
Param-count match: spline=21  relu=20  (delta=1)

Training ReLU arm for 600 epochs...
  [ReLU] epoch    1  grid-MSE = 0.403100
  ...
  [ReLU] epoch  600  grid-MSE = 0.186734

Training spline arm for 600 epochs...
  [SPL ] epoch    1  grid-MSE = 0.511934
  ...
  [SPL ] epoch  600  grid-MSE = 0.060656

================================================================
RESULT (final clean-grid MSE, lower is better):
  ReLU   arm (20 params): 0.186734
  Spline arm (21 params): 0.060656
  => KAN claim HOLDS: learnable activation wins by 67.5% at matched params.
================================================================

Learned per-channel spline shapes (untrained = exact identity y=x):
(1 of 8 hidden channels learned a non-trivial bend; showing the 4 most-bent — the fit is sparse, a few channels carry it)
Control points y[i,c] for sampled channels (knots t[i] fixed on [-2.0, 2.0]):
   t[i] =  -2.000  -1.000   0.000   1.000   2.000
  ch 0   y =  -2.000  -1.248  -0.705   1.920   2.000
  ch 1   y =   0.000   0.000   0.000   0.000   0.000
  ch 2   y =   0.000   0.000   0.000   0.000   0.000
  ch 3   y =   0.000   0.000   0.000   0.000   0.000

Same channels, activation sampled over x (deviation from y=x means the activation learned a nonlinear shape):
      x   |  ch 0  y  ch 1  y  ch 2  y  ch 3  y
  -2.000  |   -2.000    0.000    0.000    0.000
  -1.000  |   -1.248    0.000    0.000    0.000
   0.000  |   -0.705    0.000    0.000    0.000
   1.000  |    1.920    0.000    0.000    0.000
   2.000  |    2.000    0.000    0.000    0.000
```

Read it as: the ReLU arm and the spline arm carry the **same** number of
trainable weights, but the spline arm spends some of them on a **learnable
per-channel activation shape** — `ch 0` above is clearly bent away from the
identity (`y(0) = -0.705`, not `0`) — and it fits the wiggly target to a much
lower MSE. That is the KAN fixed-width / matched-param win, achieved here with a
strikingly **sparse** solution (one active channel).

## See also

[KANLayer](../KANLayer) is the **weight-space** counterpart: instead of a
per-channel learnable *activation*, it uses the true Kolmogorov-Arnold *dense
layer* `TNNetKANLayer`, where every input→output **edge** carries its own learned
Chebyshev univariate function `phi_{ij}(x)`. The two examples run the same wiggly
1D fit at matched parameter count, so activation-KAN and layer-KAN can be read
side by side.

## Build & run

```
cd examples/SplineActivationKAN
lazbuild SplineActivationKAN.lpi
../../bin/x86_64-linux/bin/SplineActivationKAN
```

Pure CPU, no dataset download, synthetic data generated in-code, total runtime a
few seconds.
