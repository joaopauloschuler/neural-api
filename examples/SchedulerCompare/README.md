# Learning-Rate Schedule Bake-Off

A head-to-head comparison of **four** learning-rate (LR) schedules. The
exact same tiny regression net is trained four times on the exact same
tiny synthetic task, and the *only* thing that changes between arms is the
learning-rate **schedule** that drives the optimiser.

Arms:

- `constant`     flat `baseLR` every epoch (the control)
- `Step`         `TStepLR(baseLR, stepSize, gamma)` — staircase decay
- `Cosine`       `TCosineAnnealingLR(etaMax, etaMin, T)` — smooth anneal
- `WarmupCosine` `TWarmupCosineLR(etaMax, etaMin, warmup, T)` — ramp + anneal

The scheduler classes live in
[`neural/neuralscheduler.pas`](../../neural/neuralscheduler.pas). They are
**not** wired into `TNeuralFit` (that is a separate open task), so this
example drives the LR **by hand**: a plain epoch loop calls
`Sched.NextLR(epoch, epoch)` at the top of every epoch and pushes the
result into the net with `NN.SetLearningRate(lr, inertia)`. The schedulers
key on the **step** argument (`t := Step`); we pass the epoch index as the
step, so the horizon `T` is measured in epochs.

```
TNNetInput(8, 1, 1)               # 8-feature input vector
  -> TNNetFullConnectReLU(16)     # hidden layer 1
  -> TNNetFullConnectReLU(16)     # hidden layer 2
  -> TNNetFullConnectLinear(1)    # scalar regression head
```

The net is identical across all four arms; only the LR schedule differs.

## The schedules

With `baseLR = 0.05`, `minLR = 0.001`, `Epochs = T = 40`:

- **constant** — holds `0.05` for all 40 epochs.
- **Step** — `TStepLR(0.05, 10, 0.5)`: halves every 10 epochs
  (`0.05 -> 0.025 -> 0.0125 -> 0.00625`), a clean staircase.
- **Cosine** — `TCosineAnnealingLR(0.05, 0.001, 40)`: a smooth cosine
  curve from `0.05` down to `0.001`.
- **WarmupCosine** — `TWarmupCosineLR(0.05, 0.001, 8, 40)`: linear warmup
  from `0` up to `0.05` over the first 8 epochs, then the same cosine
  anneal down to `0.001`.

## The synthetic task

A tiny scalar regression. Each sample is an 8-dim input vector drawn
uniformly from `[-1, 1]`. The target is a fixed, deterministic,
mildly-nonlinear function of those features:

```
y = sin(x0) + 0.5 * x1 * x2 - 0.3 * x3 + 0.2 * x4 * x5
```

(one trig term, two feature products, one linear term), so the small MLP
is exactly the right tool and every arm can learn it.

Everything is generated in-code (no dataset download). `RandSeed` is reset
to the same value before each arm's data generation and before
building/initialising its net, so every arm sees identical inputs and
identical weight init; only the LR schedule differs.

## Build & run

```
lazbuild examples/SchedulerCompare/SchedulerCompare.lpi
bin/x86_64-linux/bin/SchedulerCompare
```

Pure CPU, no external data, single-threaded, finishes in about a second.
The compiled binary lands in `bin/x86_64-linux/bin/` (shared with the
other examples), not inside this directory. The run is non-interactive (no
trailing `ReadLn`).

## What it shows

- An **ASCII chart** of the LR curve for each schedule, sampled across the
  run. You can *see* constant is flat, Step drops in stairs, Cosine curves
  smoothly down, and WarmupCosine ramps up then curves down.
- A comparison table with one row per schedule: initial and final
  **train** MSE, initial and final **validation** MSE, and wall-clock
  seconds.
- Two NaN/Inf-guarded sanity checks printed as `PASS`/`FAIL`:
  1. all four arms produced a finite final loss, and
  2. all four arms reduced validation loss below their pre-training
     baseline (i.e. every schedule actually trained the net).

The headline signal is the **shape of the LR curves** plus *"they all
train this net, here is how they compare"*. The per-arm final-loss ranking
is **seed-dependent**: on this short, easy task all four schedules
converge to a small MSE, and which one posts the lowest final loss will
shuffle if you change the seed, the widths, or the epoch budget. This
program is a comparison *harness*, not a claim that any one schedule is
universally best.

## Expected output sketch

Real output from a recent run (`RandSeed = 42`):

```
Learning-rate schedule bake-off: one tiny net, four LR schedules.
Net: Input(8) -> FullConnectReLU(16) -> FullConnectReLU(16) -> FullConnectLinear(1)
Target: y = sin(x0) + 0.5*x1*x2 - 0.3*x3 + 0.2*x4*x5
Train=400 Val=150  Epochs=40  baseLR=0.050  minLR=0.001  RandSeed=42
Same data, same init, same net; only the LR schedule changes.
LR is driven by hand: Sched.NextLR(epoch, epoch) -> SetLearningRate.

Training constant ... done.  final_val_mse=0.0051  0.28s
Training Step ... done.  final_val_mse=0.0056  0.29s
Training Cosine ... done.  final_val_mse=0.0054  0.20s
Training WarmupCosine ... done.  final_val_mse=0.0068  0.28s

=== Learning-rate schedules (bar = LR for that epoch, full bar = 0.050) ===

constant:
  epoch   0  lr=0.0500  |################################|
  epoch  39  lr=0.0500  |################################|

Step:
  epoch   0  lr=0.0500  |################################|
  epoch  13  lr=0.0250  |################                |
  epoch  22  lr=0.0125  |########                        |
  epoch  30  lr=0.0063  |####                            |

Cosine:
  epoch   0  lr=0.0500  |################################|
  epoch  17  lr=0.0312  |####################            |
  epoch  30  lr=0.0082  |#####                           |
  epoch  39  lr=0.0011  |#                               |

WarmupCosine:
  epoch   0  lr=0.0000  |                                |
  epoch   4  lr=0.0250  |################                |
  epoch   9  lr=0.0499  |################################|
  epoch  39  lr=0.0011  |#                               |

=== Comparison (MSE, wall-clock) ===
schedule        init_train final_train    init_val   final_val   seconds
constant            0.5462      0.0021      0.4987      0.0051      0.28
Step                0.5462      0.0023      0.4987      0.0056      0.29
Cosine              0.5462      0.0021      0.4987      0.0054      0.20
WarmupCosine        0.5462      0.0022      0.4987      0.0068      0.28

=== Sanity checks ===
[PASS] all 4 arms produced a finite (no NaN/Inf) final loss.
[PASS] all 4 arms reduced val loss below their pre-training baseline.
```

(The LR chart above is abridged; the program prints ten evenly-spaced
epoch rows per schedule.) All four final MSEs sit far below their initial
baselines, confirming every schedule trains the net end-to-end.
