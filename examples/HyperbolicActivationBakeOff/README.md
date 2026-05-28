# Hyperbolic Activation Bake-Off

Compares eight hyperbolic-family activation functions on the toy hypotenuse
task `y = sqrt(X^2 + Y^2)` using the same tiny MLP, the same training data,
and the same RNG seed. After every epoch it prints the current validation
`Error:` (MSE in original hypotenuse units), and at the end it prints a
CSV-style table of final validation MSE and the first epoch each activation
crosses the convergence threshold of MSE < 5.

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
- Training set: 102400 pairs of `(X, Y)` drawn from `[0, 100)`, with both
  inputs and targets normalized to `[0, 1]` (inputs divided by 100,
  hypotenuse divided by 200) so that `sinh`/`exp`-based activations do not
  overflow.
- Optimizer: default SGD with momentum, learning rate `0.01`, batch size 32,
  8 epochs.
- `RandSeed := 42` is restored before every activation so they all train on
  the same data with the same weight initialization.

## How to run

```bash
cd examples/HyperbolicActivationBakeOff
lazbuild HyperbolicActivationBakeOff.lpi
../../bin/x86_64-linux/bin/HyperbolicActivationBakeOff
```

## Sample output

Each activation prints a per-epoch `Error:` (validation MSE in hypotenuse
units) while it trains, e.g. for `TNNetLisht`:

```
Training TNNetLisht ...
Epochs: 1. ... Error:  1.9569
Epochs: 2. ... Error:  1.7180
...
Epochs: 8. ... Error:  1.4915
```

followed by the summary table:

```
Hyperbolic-family activation bake-off on the hypotenuse toy task.
Net: 2 -> 32 (activation) -> 1, 8 epochs, 102400 train pairs, LR=0.01, RandSeed=42.
Convergence threshold (validation loss): 5.00

=== Results (CSV) ===
activation,final_val_loss,epochs_to_converge,total_epochs
TNNetHyperbolicTangent,25.1862,4,8
TNNetLeCunTanh,29.0784,6,8
TNNetSinhAct,43.6967,4,8
TNNetArcSinh,35.2361,6,8
TNNetLisht,1.9569,1,8
TNNetBentIdentity,8.0607,3,8
TNNetTanhExp,2.4302,1,8
TNNetLogCoshActivation,3.0677,2,8

Total wall time: 273.26 s
```

`epochs_to_converge` is the first epoch at which the training network's
validation MSE crossed below 5.0 (the value printed each epoch as `Error:`);
`NA` means the activation never crossed it within the 8-epoch budget. With
102400 training pairs every activation now crosses the threshold within a
few epochs.

`final_val_loss` is the MSE of the predicted hypotenuse in the original
target units (the network output is multiplied back by 200 before
measuring), but it is measured on the *best checkpoint* that `TNeuralFit`
reloads at the end of training — which is selected on `TNeuralFit`'s own
internal validation metric, not on this MSE. That is why the saturating
`tanh`/`sinh` family can show a high `final_val_loss` even though their
last-epoch `Error:` was small: their best checkpoint was an early epoch.

On this small / fixed-seed harness, `TNNetLisht`, `TNNetTanhExp`, and
`TNNetLogCoshActivation` reach the lowest final losses and converge in the
first one or two epochs, while the plain `tanh`-family checkpoints reload to
a much higher loss — a small but visible reminder that the activation choice
matters even on the simplest regression toy. Run with a different seed to
see how much of this is harness variance vs. real ordering.
