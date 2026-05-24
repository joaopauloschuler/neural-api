# DeadReLUDiagnostic

Trains the **same** small classifier **four times** — identical in every
respect except the hidden activation function — and tracks the
**per-epoch fraction of dead hidden units** along the whole training
trajectory, then prints a summary comparing the four activations:

- `TNNetReLU`
- `TNNetLeakyReLU`
- `TNNetGELU`
- `TNNetSwish`

A hidden unit is **dead** when its activation output stays at
(numerically) zero for *every* probe sample: such a unit produces a zero
gradient and can never recover. Plain ReLU is prone to this under an
aggressive learning rate — once a unit's pre-activation is pushed
negative for all inputs it is stuck at zero. LeakyReLU / GELU / Swish keep
a non-zero output (and gradient) for negative inputs, so far fewer of
their units die. Demonstrating that contrast is the whole point.

## How this differs from `examples/DeadNeuronReport/`

`DeadNeuronReport` prints a single **static** report of one
already-trained network. This example reports a **trajectory** (the dead
fraction at every checkpoint epoch) **and** a cross-activation
**comparison** of four otherwise-identical networks.

## Dead-unit measurement

Measured exactly the way `TNNet.DeadNeuronReport` defines it (see
`neural/neuralnetwork.pas`): every probe sample is pushed through the net
and, for each hidden unit, the maximum absolute activation is tracked; a
unit is dead if that maximum is `<= 1e-6`. Here the *number* is read
directly each epoch instead of the formatted string report, so it can be
plotted as a per-epoch trajectory.

## Architecture / task

- Task: 3 Gaussian blobs lifted into 6 dimensions, one-hot 3-class
  classification. The blob centres are fixed once and shared by all four
  arms, so the task is literally identical across activations.
- Net: `6 -> FC(64) -> act -> FC(64) -> act -> FC(3) -> SoftMax`.
- Aggressive learning rate `LR = 0.5` to induce dying ReLU.
- Deterministic: `RandSeed` is reset to the same value before each arm so
  all four start from identical weights and see identical training draws;
  the manual `Compute`/`Backpropagate` loop is single-threaded.

## Build & run

```
cd examples/DeadReLUDiagnostic
lazbuild DeadReLUDiagnostic.lpi
../../bin/x86_64-linux/bin/DeadReLUDiagnostic
```

Fallback build:

```
fpc -B -Fu../../neural -Mobjfpc -Sh -O2 DeadReLUDiagnostic.lpr
```

Total runtime is a few seconds.

## Sample output

Real output from this example (`RandSeed = 2026`, `LR = 0.5`):

```
================================================================
ARM: TNNetReLU  (LR=0.50, 64 hidden units x 2 layers)
================================================================
epoch   mean-loss    dead%
1       0.360744        19.53%
10      0.000075        18.75%
20      0.000008        18.75%
30      0.000005        18.75%
40      0.000005        18.75%
50      0.000003        18.75%
60      0.000003        18.75%

================================================================
ARM: TNNetLeakyReLU  (LR=0.50, 64 hidden units x 2 layers)
================================================================
epoch   mean-loss    dead%
1       0.361917         0.00%
...
60      0.000003         0.00%

(GELU and Swish arms also report 0.00% dead at every epoch.)

================================================================
SUMMARY: dead hidden-unit fraction (lower is healthier)
================================================================
activation       peak dead%   final dead%
----------------------------------------------------------------
TNNetReLU             19.53%      18.75%
TNNetLeakyReLU         0.00%       0.00%
TNNetGELU              0.00%       0.00%
TNNetSwish             0.00%       0.00%
```

Roughly one in five ReLU hidden units dies at epoch 1 and never recovers
(18.75% stay dead through the whole run), whereas the three leaky/smooth
activations keep a non-zero gradient for negative inputs and lose **no**
units. All four reach essentially the same training loss, so the dead
ReLU units are pure wasted capacity — the classic dying-ReLU pathology.
