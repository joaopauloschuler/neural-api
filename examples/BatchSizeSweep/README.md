# Batch Size Sweep

Trains the **same** tiny MLP on the **same** synthetic hypotenuse task
`y = sqrt(X^2 + Y^2)` across a sweep of batch sizes `{1, 8, 32, 128}` and prints
how the batch-size knob trades off wall-clock-per-epoch against
epochs-to-converge. The net, the data, and the RNG seed are all held fixed, so
the **only** variable is the batch size.

This is a beginner-oriented companion to the activation / optimizer bake-offs in
this suite: instead of comparing layers, it isolates one of the most common
training hyperparameters and makes its cost/quality trade visible on a task that
trains in well under a minute on CPU.

## Setup

- Net: `2 -> FullConnectLinear(16) -> ReLU -> FullConnectLinear(1)`.
- Training set: 1000 pairs of `(X, Y)` drawn from `[0, 100)`, with both inputs
  and targets normalized to `~[0, 1]` (inputs divided by 100, hypotenuse divided
  by 200).
- Optimizer: default SGD with momentum, learning rate `0.01`, no L2 / LR decay,
  up to 80 epochs.
- `RandSeed := 424242` is restored **before generating the data and before each
  fit**, so every batch size sees the same pairs and the same weight
  initialization.
- Convergence threshold: validation MSE `< 8.0` in the original hypotenuse units
  (the network output is multiplied back by 200 before measuring).

## How to run

```bash
cd examples/BatchSizeSweep
lazbuild BatchSizeSweep.lpi
../../bin/x86_64-linux/bin/BatchSizeSweep
```

The fit's per-epoch progress is printed to stdout during training; the summary
table and a self-checking correctness gate are emitted at the end.

## Sample output

```
=== Batch-size trade-off ===
batch  sec/epoch  epochs_to_conv  sec_to_conv  final_train_mse  final_val_mse
    1      0.080               4         0.22           0.7979         0.9977
    8      0.357               4         0.98           0.8172         0.7957
   32      0.090               4         0.56           0.7264         0.6723
  128      0.069              16         0.49           1.7435         1.4725

Total wall time: 47.83 s
Correctness gate: PASS (every batch size converged below MSE 8.00).
```

(Exact numbers vary a little with the platform / float build and the
thread-scheduling noise in `sec/epoch`, but the broad story is stable.)

## Reading the table

- `sec/epoch` tends to **fall** as the batch grows: fewer weight updates per
  epoch and bigger vectorized chunks make each epoch cheaper per sample.
- `epochs_to_conv` tends to **rise** as the batch grows: a small batch takes
  many noisy gradient steps and reaches the threshold in fewer epochs, while the
  large batch (128) here needs noticeably more epochs at the same learning rate.
- That is the knob in one line: **small batch = fewer, noisier, slower-per-sample
  epochs; large batch = more, smoother, cheaper epochs** (a large batch often
  wants a larger learning rate to keep the epoch count down).

`final_val_mse` is the mean squared error of the predicted hypotenuse in the
original target units. `epochs_to_conv` / `sec_to_conv` are the first epoch (and
wall-clock) at which validation MSE crossed below the threshold; `NA` would mean
that batch size never converged within the 80-epoch budget. The correctness gate
at the end asserts that **every** batch size converged below the threshold, so a
regression that breaks training for any batch size turns the run red.
