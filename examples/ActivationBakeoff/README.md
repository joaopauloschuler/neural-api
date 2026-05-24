# Activation Bake-Off

Compares the mainstream / modern ReLU-family activation functions on the toy
hypotenuse task `y = sqrt(X^2 + Y^2)` using the same tiny MLP, the same training
data, and the same RNG seed. Prints a CSV-style table of final validation MSE
(in original hypotenuse units) and the first epoch each activation crosses the
convergence threshold of MSE < 5.

This is the ReLU-family counterpart to
[`HyperbolicActivationBakeOff`](../HyperbolicActivationBakeOff/), which compares
only the hyperbolic-tanh family. The task, net shape, seed, and convergence
logic are identical; the only variable is the activation.

Activations compared:

- `TNNetReLU` — the rectified-linear baseline `max(0, x)`.
- `TNNetLeakyReLU` — small negative slope instead of a hard zero.
- `TNNetVeryLeakyReLU` — leaky ReLU with a larger negative slope.
- `TNNetReLU6` — ReLU clamped at 6.
- `TNNetPReLU` — leaky ReLU whose negative slope is a learnable parameter.
- `TNNetELU` — exponential linear unit.
- `TNNetSELU` — scaled ELU (self-normalizing).
- `TNNetCELU` — continuously-differentiable ELU.
- `TNNetSwish` — `x * sigmoid(x)`.
- `TNNetSiLU` — sigmoid-weighted linear unit (Swish with beta = 1).
- `TNNetGELU` — Gaussian error linear unit.
- `TNNetHardSwish` — piecewise-linear Swish approximation.
- `TNNetMish` — `x * tanh(softplus(x))`.
- `TNNetSoftPlus` — `log(1 + exp(x))`.

## Setup

- Net: `2 -> FullConnectLinear(32) -> activation -> FullConnectLinear(1)`.
- Training set: 1000 pairs of `(X, Y)` drawn from `[0, 100)`, with both
  inputs and targets normalized to `[0, 1]` (inputs divided by 100,
  hypotenuse divided by 200) so that exponential-tail activations do not
  overflow.
- Optimizer: default SGD with momentum, learning rate `0.01`, batch size 32,
  60 epochs.
- `RandSeed := 42` is restored before every activation so they all train on
  the same data with the same weight initialization.

## How to run

```bash
cd examples/ActivationBakeoff
lazbuild ActivationBakeoff.lpi
../../bin/x86_64-linux/bin/ActivationBakeoff
```

The fit's per-epoch progress is printed to stdout during training; the summary
table below is emitted at the end.

## Sample output

```
Mainstream (ReLU-family) activation bake-off on the hypotenuse toy task.
Net: 2 -> 32 (activation) -> 1, 60 epochs, 1000 train pairs, LR=0.01, RandSeed=42.
Convergence threshold (validation loss): 5.00

=== Results (CSV) ===
activation,final_val_loss,epochs_to_converge,total_epochs
TNNetReLU,1.8878,26,60
TNNetLeakyReLU,0.3277,5,60
TNNetVeryLeakyReLU,0.5076,9,60
TNNetReLU6,1.8714,25,60
TNNetPReLU,2.0186,24,60
TNNetELU,12.2412,NA,60
TNNetSELU,4.2781,37,60
TNNetCELU,17.1129,NA,60
TNNetSwish,58.0939,NA,60
TNNetSiLU,51.7903,NA,60
TNNetGELU,17.4489,NA,60
TNNetHardSwish,38.9073,NA,60
TNNetMish,56.9030,NA,60
TNNetSoftPlus,62.2400,NA,60

Total wall time: 120.18 s
```

(The whole run wall-clocks at about 2 minutes on CPU.)

`final_val_loss` is reported as mean squared error of the predicted
hypotenuse in the original target units (the network output is multiplied
back by 200 before measuring). `epochs_to_converge` is the first epoch at
which validation MSE crossed below 5.0; `NA` means the activation did not
hit that threshold within the 60-epoch budget.

On this small / fixed-seed harness the plain `TNNetLeakyReLU` and
`TNNetVeryLeakyReLU` converge fastest and lowest, the classic ReLU /
ReLU6 / PReLU trio lands close behind, and the smooth gated activations
(`Swish`, `SiLU`, `Mish`, `SoftPlus`) need more than 60 epochs at this
learning rate. That ordering is a property of this tiny toy and seed, not
a general ranking — run with a different seed (or the longer 150-epoch
budget the hyperbolic example uses) to see how much is harness variance.
