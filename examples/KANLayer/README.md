# KANLayer

A **layer-KAN vs param-matched ReLU-MLP** toy-fit micro-experiment for the
library's true Kolmogorov-Arnold *dense layer*
[`TNNetKANLayer`](../../neural/neuralnetwork.pas) (KAN, Liu et al. 2024).

Where the sibling example [SplineActivationKAN](../SplineActivationKAN) puts the
KAN idea in an **activation** (one learnable univariate function per channel,
depth-preserving), this example uses the KAN idea in the **weights**:
`TNNetKANLayer` maps `D_in -> D_out` where **every** input→output edge `(i,j)`
carries its own learned univariate function

```
phi_{ij}(x_i) = sum_{k=0..K} c_{ijk} * T_k(tanh(x_i))     (Chebyshev basis, degree K)
y_j = sum_i phi_{ij}(x_i)
```

There is **no weight matrix** — the "weights" *are* the edge functions. Only the
`D_in*D_out*(K+1)` Chebyshev coefficients train; the basis is fixed and
orthogonal. The layer is initialized near-linear (degree-1 coefficient random,
higher orders ~0) so an untrained `TNNetKANLayer` behaves like a tanh-squashed
linear layer.

## The fit

Target (two incommensurate frequencies so a single ReLU MLP can't nail it cheaply):

```
y = sin(3x) + 0.3*sin(11x),   x in [-2, 2]
```

Two arms, trained the **same** number of epochs on the **same** data with the
**same** optimizer:

```
Arm A (baseline MLP):  Input(1) -> FullConnectReLU(Wa) -> FullConnectLinear(1)
Arm B (KAN-dense MLP): Input(1) -> TNNetKANLayer(Wb, K=4) -> FullConnectLinear(1)
```

## Matched parameter count

The KAN arm spends `D_in*D_out*(K+1)` parameters on its edge coefficients. To
keep this a fair fixed-budget fight the ReLU arm is made **wider** (`Wa > Wb`) so
both arms carry ~equal total trainable weight counts. The example does **not**
hand-compute this: it builds the KAN arm, reads its exact
`TNNet.CountWeights()`, then searches for the ReLU hidden width whose
`CountWeights()` best matches, and **prints** both counts so the match is
auditable (here an exact 48 vs 48).

## Sample output

```
KANLayer: layer-KAN vs ReLU-MLP toy fit at MATCHED parameter count.
Target  y = sin(3x) + 0.3*sin(11x)  over x in [-2.0, 2.0].

KAN  arm: KANLayer(Dout=8,K=4) -> Linear(1)                  =>  48 trainable weights
ReLU arm: FullConnectReLU(24) -> Linear(1)                      =>  48 trainable weights  (width chosen to match)
Param-count match: kan=48  relu=48  (delta=0)

Training ReLU arm for 600 epochs...
  [ReLU] epoch  600  grid-MSE = 0.186131

Training KAN arm for 600 epochs...
  [KAN ] epoch  600  grid-MSE = 0.136113

================================================================
RESULT (final clean-grid MSE, lower is better):
  ReLU arm (48 params): 0.186131
  KAN  arm (48 params): 0.136113
  => KAN claim HOLDS: per-edge learnable function wins by 26.9% at matched params.
================================================================
```

Read it as: the ReLU arm and the KAN arm carry the **same** number of trainable
weights, but the KAN arm spends them on **per-edge learnable Chebyshev
functions** `phi_{ij}(x)` instead of a scalar weight matrix + pointwise ReLU, and
fits the wiggly target to a lower MSE — the layer-KAN matched-param win.

See also [SplineActivationKAN](../SplineActivationKAN) for the **activation-KAN**
counterpart (per-channel learnable activation), so the two KAN flavours can be
read side by side.

## Build & run

```
cd examples/KANLayer
lazbuild KANLayer.lpi
../../bin/x86_64-linux/bin/KANLayer
```

Pure CPU, no dataset download, synthetic data generated in-code, total runtime a
few seconds.
