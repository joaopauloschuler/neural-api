# Shake-Shake regularization bake-off

This example contrasts **Shake-Shake regularization** against a plain
**deterministic two-branch residual** on a small, noisy, over-parameterised
toy classification problem, and charts the headline regularisation win: a
**narrower train/val gap** (better generalisation) for the shake-shake arm
in the spirit of the Mixup / SAM follow-up demos.

## Shake-Shake

[Shake-Shake regularization (Gastaldi 2017, *Shake-Shake regularization*)](https://arxiv.org/abs/1705.07485)
replaces the deterministic average of two parallel residual branches with a
**stochastic** convex combination:

```
forward (train) : y = skip + alpha * B1(x) + (1-alpha) * B2(x),  alpha ~ U[0,1)
backward (train): branch gradients scaled by an INDEPENDENT beta ~ U[0,1)
eval            : alpha = beta = 0.5  (deterministic expected value)
```

The forward `alpha` and the backward `beta` are sampled *independently* per
pass — this "shake" of both the forward blend and the gradient flow is the
regulariser. The deterministic baseline instead uses the fixed `0.5/0.5`
average `y = skip + 0.5*B1 + 0.5*B2`.

## API used

In `neuralnetwork.pas`:

- `function TNNet.AddShakeShakeBlock(HiddenDim: integer): TNNetLayer;`
  Builds the canonical block: two independent pointwise residual branches
  (`PointwiseConvReLU(HiddenDim) -> PointwiseConvLinear(Depth)`) merged with the
  skip via `TNNetShakeShakeMerge`.
- `TNNetShakeShakeMerge.Create([B1, B2, skip])` — the stochastic combiner
  (three same-shape inputs). Stochastic while `Enabled`; deterministic 0.5/0.5
  at inference. The deterministic baseline reproduces the *same* branch shapes
  but merges with `TNNetMulByConstant(0.5)` on each branch + `TNNetSum`.

## The toy task

- 40-D input: **2 informative dims** (two overlapping Gaussian blobs that carry
  the class signal) + **38 pure-nuisance dims** (no signal).
- Only **64 training samples** with **25% label noise** (flipped labels) — few
  enough, in a high-dim space, that an over-sized net can *memorise* the noisy
  labels via the nuisance dims.
- **600 clean validation samples** measure true generalisation.

Both arms share an identical over-parameterised trunk and identical branch
shapes; the *only* difference is the merge layer.

## Result (reproducible, `RandSeed := 20260607`)

```
arm            |   trAcc  trLoss |   vaAcc  vaLoss |  accGAP lossGAP
---------------+-----------------+-----------------+----------------
Deterministic  |  100.00  0.0366 |   70.17  0.6589 |   29.83   0.6224
Shake-Shake    |  100.00  0.0326 |   74.50  0.6193 |   25.50   0.5866
```

Both arms memorise the training set (100%), but the Shake-Shake arm
generalises better on **every** metric: higher val accuracy (74.5% vs 70.2%),
lower val loss, and a narrower train/val gap (accGAP 25.5 vs 29.8, lossGAP
0.587 vs 0.622). That narrower gap is the regularisation win.

## Running

```
lazbuild ShakeShakeReg.lpi
./../../bin/x86_64-linux/bin/ShakeShakeReg
```

Pure CPU, finishes in well under a minute (~7 s on 2 cores).
