# Hyperbolic Activation Bake-Off

Compares eight hyperbolic-family activation functions on the toy hypotenuse
task `y = sqrt(X^2 + Y^2)` using the same tiny MLP, the same training data,
and the same RNG seed. Prints a CSV-style table of final validation MSE (in
original hypotenuse units) and the first epoch each activation crosses the
convergence threshold of MSE < 5.

Activations compared:

- `TNNetHyperbolicTangent` — the classical baseline.
- `TNNetLeCunTanh` — `1.7159 * tanh(2/3 * x)` from LeCun's "Efficient Backprop".
- `TNNetSinhAct` — `sinh(x)`; only safe here because inputs are normalized.
- `TNNetArcSinh` — `arcsinh(x)`; unbounded, slow-growing.
- `TNNetLisht` — `x * tanh(x)`; non-monotonic smooth ReLU alternative.
- `TNNetBentIdentity` — `(sqrt(x^2+1) - 1)/2 + x`.
- `TNNetTanhExp` — `x * tanh(exp(x))`.
- `TNNetLogCoshActivation` — `log(cosh(x))`.

## Setup

- Net: `2 -> FullConnectLinear(32) -> activation -> FullConnectLinear(1)`.
- Training set: 1000 pairs of `(X, Y)` drawn from `[0, 100)`, with both
  inputs and targets normalized to `[0, 1]` (inputs divided by 100,
  hypotenuse divided by 200) so that `sinh`/`exp`-based activations do not
  overflow.
- Optimizer: default SGD with momentum, learning rate `0.01`, batch size 32,
  150 epochs.
- `RandSeed := 42` is restored before every activation so they all train on
  the same data with the same weight initialization.

## How to run

```bash
cd examples/HyperbolicActivationBakeOff
lazbuild HyperbolicActivationBakeOff.lpi
../../bin/x86_64-linux/bin/HyperbolicActivationBakeOff
```

## Sample output

```
Hyperbolic-family activation bake-off on the hypotenuse toy task.
Net: 2 -> 32 (activation) -> 1, 150 epochs, 1000 train pairs, LR=0.01, RandSeed=42.
Convergence threshold (validation loss): 5.00

=== Results (CSV) ===
activation,final_val_loss,epochs_to_converge,total_epochs
TNNetHyperbolicTangent,35.6685,NA,150
TNNetLeCunTanh,50.0707,NA,150
TNNetSinhAct,40.4517,NA,150
TNNetArcSinh,51.3354,NA,150
TNNetLisht,1.7030,32,150
TNNetBentIdentity,8.7489,NA,150
TNNetTanhExp,2.2611,85,150
TNNetLogCoshActivation,5.0834,148,150

Total wall time: 41.74 s
```

`final_val_loss` is reported as mean squared error of the predicted
hypotenuse in the original target units (the network output is multiplied
back by 200 before measuring). `epochs_to_converge` is the first epoch at
which validation MSE crossed below 5.0; `NA` means the activation did not
hit that threshold within the 150-epoch budget.

On this small / fixed-seed harness, `TNNetLisht` and `TNNetTanhExp` cross
the threshold by a wide margin, `TNNetLogCoshActivation` just makes it,
and the saturating `tanh` family stalls — a small but visible reminder
that the activation choice matters even on the simplest regression toy.
Run with a different seed to see how much of this is harness variance vs.
real ordering.
