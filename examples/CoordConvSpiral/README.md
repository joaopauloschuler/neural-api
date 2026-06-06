# CoordConv Spiral demo

Smallest possible end-to-end demo of `TNNetCoordConv`
(Liu et al. 2018, "An intriguing failing of convolutional neural
networks and the CoordConv solution",
https://arxiv.org/abs/1807.03247).

## Toy task

The input is an 8x8x1 image whose pixels are all 0 except for exactly
one pixel set to 1.0. The target is the (x, y) coordinate of that
pixel, normalized to `[-1, 1]`. A constant predictor scores MSE
~= 2 * Var(Uniform[-1, 1]) = 2/3 ~= 0.667.

## Architecture

Both networks share the same shape; the only difference is the
leading `TNNetCoordConv`:

```
plain :  Input(8,8,1) -> Conv1x1+ReLU(16) -> Conv1x1+ReLU(16) -> AvgPool -> Linear(2)
coord :  Input(8,8,1) -> TNNetCoordConv -> Conv1x1+ReLU(16) -> Conv1x1+ReLU(16) -> AvgPool -> Linear(2)
```

The head (global-average-pool followed by a tiny linear map) is
exactly translation-INVARIANT: after the avg-pool, the model can only
see what features are present, not where they are. With purely
spatial inputs, the plain stack therefore cannot encode absolute
position and gets stuck near the constant-prediction baseline. The
CoordConv variant gives the first conv two extra channels carrying
the normalized `(x, y)` coordinates, so the conv can learn the
trivial gating `relu(input + coord - 1) ~= coord_of_active_pixel`,
which the average-pool then trivially reads out.

## Expected output

Both networks see the SAME data stream (`RandSeed = 1234` before each
build). After 4000 minibatch steps:

```
plain conv  MSE ~ 0.84  (no better than the constant baseline)
CoordConv   MSE ~ 0.002 (~400x better)
```

Total runtime: ~25 seconds on CPU.

## Build & run

```
fpc -dRelease -dUseCThreads -O3 -Fu../../neural CoordConvSpiral.lpr
./CoordConvSpiral
```

No external data.
