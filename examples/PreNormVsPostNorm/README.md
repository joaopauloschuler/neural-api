# PreNorm vs PostNorm

Compares the **same** deepish residual stack wired three different ways on the
toy hypotenuse task `y = sqrt(X^2 + Y^2)`. The only variable is the residual
builder used for every block, so this is a direct three-way builder swap on one
fixed architecture, seed, learning rate, and epoch budget.

This is the residual-normalisation counterpart to
[`ActivationBakeoff`](../ActivationBakeoff/): same synthetic task and the same
fixed-seed bake-off harness, but here we hold the activation fixed and vary
*where the normalisation sits inside the residual block*.

## The three builders compared

Each builder wraps a shape-preserving sublayer in a residual block:

- `TNNet.AddPreNormResidual`  — `y = x + Sublayer(LayerNorm(x))`
- `TNNet.AddRMSNormResidual`  — `y = x + Sublayer(RMSNorm(x))` (LLaMA-style pre-norm)
- `TNNet.AddPostNormResidual` — `y = LayerNorm(Sublayer(x) + x)` (classic post-norm)

The sublayer is identical in all three arms:
`[TNNetPointwiseConvLinear(WIDTH), TNNetReLU]`. `TNNetPointwiseConvLinear` is
shape-preserving along the feature (Depth) axis, so the residual sum is always
valid — `TNNetFullConnectLinear` is **not** used inside the block because it can
reshape and break the skip connection.

The classic result this example reproduces: as the residual stack gets deep,
**pre-norm** variants train stably and reach low loss, while **post-norm** trains
much more slowly / less stably at the same learning rate. (At deeper stacks or
higher learning rates post-norm can diverge to NaN outright; the program guards
all printing against NaN/Inf and masks FPU exceptions so a diverging arm is
reported cleanly in the table rather than crashing the run.)

## Setup

- Architecture (all three arms): `TNNetInput(2)` →
  `FullConnectLinear(WIDTH)` → `Reshape(1,1,WIDTH)` → **12 residual blocks** of
  width 24 → `FullConnectLinear(1)` regression head.
  - The reshape moves the WIDTH features into the **Depth** axis (shape
    `1 x 1 x WIDTH`), which is what the residual builders normalise and convolve
    over (`LayerNorm` / `RMSNorm` / `PointwiseConvLinear` all act on Depth).
- Training set: 800 pairs of `(X, Y)` drawn from `[0, 100)`, plus a 200-pair
  held-out validation split. Inputs and targets are normalised to `[0, 1]`
  (inputs / 100, hypotenuse / 200).
- Optimizer: default SGD with momentum, learning rate `0.01`, batch size 32,
  30 epochs.
- `RandSeed := 42` is restored before generating the data **and** before every
  fit, so all three arms see identical data and weight initialization.

`final_val_mse` is mean squared error of the predicted hypotenuse in the
original target units (the network output is multiplied back by 200 before
measuring). `final_train_loss` is the fit's last-epoch training error.
`diverged` is `YES` if any reported value was NaN/Inf.

## How to run

```bash
cd examples/PreNormVsPostNorm
lazbuild PreNormVsPostNorm.lpi
../../bin/x86_64-linux/bin/PreNormVsPostNorm
```

The whole run wall-clocks at well under a minute on CPU.

## Sample output

```
PreNorm vs RMSNorm vs PostNorm residual-stack bake-off (hypotenuse toy task).
Same arch wired three ways: 12 residual blocks of width 24.
Block sublayer = [PointwiseConvLinear(24), ReLU]; 30 epochs, 800 train pairs, LR=0.01, RandSeed=42.

Training PreNorm  (AddPreNormResidual) ... done.
Training RMSNorm  (AddRMSNormResidual) ... done.
Training PostNorm (AddPostNormResidual) ... done.

=== Loss-vs-epoch traces ===
  PreNorm  (AddPreNormResidual) loss-vs-epoch trace:
    epoch   1: train_err = 0.553748
    epoch   6: train_err = 0.016399
    epoch  11: train_err = 0.009509
    epoch  16: train_err = 0.002841
    epoch  21: train_err = 0.005191
    epoch  26: train_err = 0.002519
    epoch  30: train_err = 0.002318

  RMSNorm  (AddRMSNormResidual) loss-vs-epoch trace:
    epoch   1: train_err = 0.222269
    epoch   6: train_err = 0.009605
    epoch  11: train_err = 0.007488
    epoch  16: train_err = 0.003166
    epoch  21: train_err = 0.003769
    epoch  26: train_err = 0.003453
    epoch  30: train_err = 0.002975

  PostNorm (AddPostNormResidual) loss-vs-epoch trace:
    epoch   1: train_err = 0.129266
    epoch   6: train_err = 0.080462
    epoch  11: train_err = 0.121064
    epoch  16: train_err = 0.135904
    epoch  21: train_err = 0.111640
    epoch  26: train_err = 0.046802
    epoch  30: train_err = 0.024128

=== Results (CSV) ===
builder,final_train_loss,final_val_mse,diverged
PreNorm  (AddPreNormResidual),0.002318,0.4711,no
RMSNorm  (AddRMSNormResidual),0.002975,0.6949,no
PostNorm (AddPostNormResidual),0.024128,488.7097,no

Verdict: PreNorm  (AddPreNormResidual) converged lowest/most stably (val MSE = 0.4711).

Total wall time: 50.37 s
```

On this 12-block stack the two pre-norm arms (`PreNorm` and `RMSNorm`) drop to a
near-zero training error within a handful of epochs and reach a sub-1 validation
MSE. The post-norm arm's training error oscillates for most of the run and is
still ~10x higher at the end, with a validation MSE three orders of magnitude
worse — the textbook deep-residual pre-norm-vs-post-norm stability gap. Exact
numbers vary a little with the platform / float build, but the ordering
(pre-norm ≈ RMSNorm ≪ post-norm) is stable. Increase `NUM_BLOCKS` or the
learning rate to push post-norm all the way to divergence (`diverged = YES`).
