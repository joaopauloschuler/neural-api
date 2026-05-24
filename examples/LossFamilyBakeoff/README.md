# Loss-family Bake-off (output heads)

Head-to-head comparison of robust regression loss heads on a noisy
hypotenuse task `y = sqrt(X^2 + Y^2)` with deliberately injected
**outliers** in the training targets. Five arms share the SAME trunk,
seed, learning rate and epochs and differ ONLY in the output loss head:

- **MSE baseline** — plain `TNNetFullConnectLinear` output trained with
  the library's default L2/MSE backprop (no special loss layer)
- `TNNetHuberLoss`
- `TNNetSmoothL1Loss`
- `TNNetCharbonnierLoss`
- `TNNetLogCoshLoss`

## What it does

The TRAINING targets get Gaussian noise (`sigma = 4`, original units)
plus a fraction (`10%`) of large additive outliers (`+/- 80`, original
units). The held-out TEST set (500 pairs) is CLEAN — no noise, no
outliers. Every arm sees identical data and identical weight init
(`RandSeed := 42`, single thread), so any difference comes purely from
the loss head.

Targets are scaled by 40 (inputs by 100) so that clean residuals stay
small — well inside the quadratic region of Huber/SmoothL1 where
`|r| << delta = 1` — while injected outlier residuals land far outside
it (`|r| ~ 2`). That is exactly the regime where the robust losses
diverge from plain MSE: MSE keeps weighting the squared outlier
residuals, while the robust heads clip or saturate the outlier gradient.
Reported MSE/MAE are converted back to the original target scale.

## Architecture

Tiny MLP `2 -> 32 -> 32 -> 1` (ReLU hidden), 200 epochs, 1000 noisy
training pairs, batch size 32, LR = 0.001.

## How to run

```bash
cd examples/LossFamilyBakeoff
lazbuild LossFamilyBakeoff.lpi
../../bin/x86_64-linux/bin/LossFamilyBakeoff
```

If `lazbuild` is unavailable:

```bash
cd examples/LossFamilyBakeoff
fpc -B -Fu../../neural -Mobjfpc -Sh -O2 LossFamilyBakeoff.lpr
```

## Sample output (200 epochs, 1000 train pairs, ~36 s wall, seed 42)

```
=== Results (clean-test error, original units) ===
loss_head        clean_test_MSE  clean_test_MAE  final_train_loss
MSE-baseline             4.8608          1.5731          0.278998
HuberLoss                1.7227          1.0270          0.272648
SmoothL1Loss             1.7227          1.0270          0.272648
CharbonnierLoss          0.3388          0.4521          0.269356
LogCoshLoss              1.4571          0.9431          0.272347
```

## Takeaway

All four robust heads beat plain MSE on the clean test set. MSE lands
at clean-test MSE `4.86`, while Huber/SmoothL1 reach `1.72`, LogCosh
`1.46`, and Charbonnier `0.34` — roughly a 3x to 14x reduction in
clean-test error. This is the expected result: MSE over-weights the
squared outlier residuals and bends the fit toward the corrupted
targets, whereas the robust losses bound (LogCosh: `tanh(r)`;
Charbonnier: `r/sqrt(r^2 + eps^2)`) or clip (Huber/SmoothL1, here
`delta = 1`) the per-sample gradient so outliers cannot dominate.

`HuberLoss` and `SmoothL1Loss` print identical numbers because
`TNNetSmoothL1Loss` is exactly `TNNetHuberLoss` with `delta = 1.0`, which
is the default used here.

The exact ranking among the robust heads is somewhat seed-dependent;
the durable point is that the matched harness (same trunk/seed/LR/epochs,
differing only in the loss head) cleanly isolates the loss head's effect,
and the robust family consistently beats MSE under heavy-tailed target
noise.
