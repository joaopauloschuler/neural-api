# OptimizerBakeoff

Trains the **same** tiny MLP on the **same** fixed toy dataset four times,
changing **only the optimizer**, and prints a loss-vs-epoch table plus a small
ASCII chart. A minimal, reproducible side-by-side of:

1. **SGD** - plain stochastic gradient descent (no momentum).
2. **SGD + momentum** - classic heavy-ball momentum (inertia `0.9`).
3. **Adam**.
4. **RMSProp**.

The task is hypotenuse regression `y = sqrt(x0^2 + x1^2)` with `x0, x1 ~ U(0, 1)`
(the same toy used by `examples/Hypotenuse/` and `examples/LearningRateFinder/`).

## How each optimizer is selected (library API)

This library (see `neural/neuralnetwork.pas`) exposes two weight-update paths,
both driven directly from the example with manual mini-batch training
(`ClearDeltas` -> per-sample `Compute` + `Backpropagate` -> one update call):

- **SGD / SGD+momentum** - `NN.SetLearningRate(LR, Inertia)` then
  `NN.UpdateWeights()`. With `Inertia = 0` the neuron just adds its
  accumulated delta; with `Inertia > 0` it maintains a velocity buffer
  (`FBackInertia`, the heavy-ball update). Momentum here = inertia = `0.9`.
- **Adam** - per layer `NN.Layers[i].InitAdam(Beta1=0.9, Beta2=0.999, eps=1e-8)`,
  then `NN.CalcAdamDelta()` + `NN.UpdateWeightsAdam()`. This maintains the
  first moment `m` and second moment `v` and applies
  `lr * m_hat / (sqrt(v_hat) + eps)`.
- **RMSProp** - the framework has **no separate RMSProp optimizer**, but RMSProp
  *is* Adam with the first-moment momentum turned off. So we use the same Adam
  path with **`Beta1 = 0`**: then `m = grad`, `FBeta1Decay = 0`,
  `(1 - Beta1Decay) = 1`, and the update collapses to
  `lr * grad / (sqrt(v_hat) + eps)`, i.e. textbook RMSProp (centered second
  moment with bias correction). This was verified by reading
  `TNNetNeuron.CalcAdamDelta` and `TNNetLayer.CalcAdamDelta`.

`NN.SetBatchUpdate(True)` makes the per-sample deltas **accumulate** across the
whole mini-batch, so every arm sees identical mini-batch gradients.

## What is held fixed across all four arms

- Same `RandSeed` (424242), reseeded before each arm so all four start from
  **identical initial weights**.
- Same architecture `2 -> 16 -> 16 -> 1` (ReLU hidden, Linear head).
- Same fixed train/validation data (512 / 256 pairs).
- Same mini-batch size (32), same number of epochs (200), and the same
  per-epoch Fisher-Yates shuffle order (the shared RNG is in the same state for
  every arm).
- Single-threaded, CPU-only. The whole bake-off finishes in well under 10 s.

### Fairness caveat (learning rate)

The "best" learning rate is **not** the same for every optimizer, so a single
shared LR cannot be fair to all four. Here the LR is **held fixed**:
`0.05` for the two SGD variants and `0.01` for Adam/RMSProp (adaptive methods
prefer a smaller step). This isolates the **update rule**, but it is *not* a
tuned-LR shoot-out. In particular, plain SGD at a fixed `0.05` oscillates and
does not converge on this problem - which is itself the instructive point:
momentum and the adaptive methods are far more forgiving of the same LR. To
turn this into a tuned comparison, sweep the LR per arm (see
`examples/LearningRateFinder/`).

## Build & run

Using Lazarus / `lazbuild`:

```bash
cd examples/OptimizerBakeoff
lazbuild OptimizerBakeoff.lpi
../../bin/x86_64-linux/bin/OptimizerBakeoff
```

Using `fpc` directly:

```bash
cd examples/OptimizerBakeoff
fpc -B -Fu../../neural -Mobjfpc -Sh -O2 OptimizerBakeoff.lpr
./OptimizerBakeoff
```

## Sample output

Real output for seed `424242` (note: plain SGD does not converge at the fixed
LR; the other three do):

```
Optimizer Bake-off
  task    : y = sqrt(x0^2 + x1^2), x0,x1 ~ U(0,1)
  network : 2 -> 16 -> 16 -> 1 (ReLU hidden, Linear head)
  data    : 512 train / 256 val pairs, batch 32, 200 epochs
  LR      : SGD family = 0.050, Adam/RMSProp = 0.010 (HELD FIXED across arms)
  momentum: 0.90   adam betas: (0.9, 0.999)   rmsprop beta1: 0.0

Training arm 1/4: SGD ...
Training arm 2/4: SGD+momentum ...
Training arm 3/4: Adam ...
Training arm 4/4: RMSProp ...

Validation loss (MSE) vs epoch
================================================================================
epoch ->               1         5        10        25        50       100       200
--------------------------------------------------------------------------------
SGD              0.08570   0.19157   0.12260   0.09451   0.08106   0.08096   0.11899
SGD+momentum     0.10688   0.00133   0.00051   0.00009   0.00003   0.00002   0.00001
Adam             0.03631   0.00121   0.00017   0.00003   0.00002   0.00003   0.00011
RMSProp          0.01499   0.00301   0.00066   0.00099   0.00011   0.00016   0.00007
--------------------------------------------------------------------------------

Summary
--------------------------------------------------------------------------------
optimizer         final loss      epochs-to-converge
                 (epoch 200)       (val MSE < 0.001)
SGD                 0.118987             not reached
SGD+momentum        0.000011                       7
Adam                0.000107                       6
RMSProp             0.000071                      10
--------------------------------------------------------------------------------

ASCII chart  (x = log10 val-loss, low is better; rows = epoch checkpoints)
  log10 loss range: -4.970 .. -0.718
  ----------------------------------------------------------------------
  ep   1 |                                         R    A   S M   |
  ep   5 |                           A    R                      S|
  ep  10 |                A     MR                            S   |
  ep  25 |     A      M            R                         S    |
  ep  50 |   A  M      R                                    S     |
  ep 100 |   M A         R                                  S     |
  ep 200 |M          R A                                      S   |
  ----------------------------------------------------------------------
  legend: S=SGD  M=SGD+momentum  A=Adam  R=RMSProp   (left = lower loss)
```

## Reading the output

- The **table** lists validation MSE at seven epoch checkpoints for each arm.
- The **summary** reports the final (epoch 200) loss and the first epoch each
  arm drives validation MSE below `0.001` ("epochs-to-converge").
- The **ASCII chart** plots `log10(loss)` for each arm at the checkpoints
  (further **left = lower loss = better**); reading top-to-bottom shows how each
  optimizer marches leftward over training.

For this seed: momentum, Adam, and RMSProp all converge within ~10 epochs and
end near `1e-4..1e-5` MSE, while plain SGD at the fixed `0.05` LR stalls around
`1e-1` and never converges.

## Files

- `OptimizerBakeoff.lpr` - the program.
- `OptimizerBakeoff.lpi` - the Lazarus project file (mirrors
  `examples/LearningRateFinder/LearningRateFinder.lpi`).
