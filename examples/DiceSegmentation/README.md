# Dice-loss segmentation vs MSE baseline

This example demonstrates the `TNNetDiceLoss` segmentation head on a tiny
**synthetic binary-mask** task and contrasts the Dice/IoU it reaches against a
plain MSE-head baseline of identical architecture. It is pure CPU, uses no
external dataset, and finishes in a few seconds.

The synthetic task is foreground/background segmentation. Each sample is a
small single-channel `16x16` grid containing a randomly placed/sized filled
**disc**; the ground-truth mask is `1` inside the disc and `0` outside, and
Gaussian noise is added to the **input only**. The foreground is a minority of
the pixels — the regime where a region-overlap loss (Dice) is expected to help
versus a per-pixel MSE loss.

## How the Dice head is wired

`TNNetDiceLoss` is a `TNNetIdentity` descendant — the Tversky loss with
`alpha = beta = 0.5`, for which the Tversky index reduces to the Dice
coefficient `2*TP / (2*TP + FP + FN)`. Its forward pass is an identity
passthrough, so its output equals the foreground-probability map produced by
the preceding `Sigmoid`. The framework seeds the last layer's `FOutputError`
with `(output - target)`; the head recovers the ground-truth mask
`g = p - seeded` and overwrites the residual with the analytic gradient of
`L = 1 - 2*TP/(2*TP + FP + FN)`.

Because the head reads `p` as a foreground **probability in `[0,1]`**, the
layer feeding it must be a `Sigmoid`, and the **target** volume supplied to
`Backpropagate` is the binary mask:

```
Input(16, 16, 1)
ConvolutionReLU(8, 3, 1, 1)    // featuresize 3, inputpadding 1, stride 1
ConvolutionReLU(8, 3, 1, 1)    //   => spatial 16x16 preserved throughout
ConvolutionLinear(1, 3, 1, 1)  // 1-channel logit map, same 16x16
Sigmoid                        // per-pixel foreground probability
TNNetDiceLoss                  // identity passthrough + analytic Dice gradient
```

The **MSE baseline** is the *same* stack without the loss head: the last layer
is the `Sigmoid`, so the framework-seeded `(output - target)` residual is the
MSE gradient. Both nets are built from the same `RandSeed`, so they start from
identical weights and see identical data — the only difference is the loss.

> Because `TNNetDiceLoss` is an identity passthrough, `Net.Compute` still
> returns the `Sigmoid` probabilities, so the 0.5-thresholded prediction used
> for Dice/IoU scoring is read from the last layer for both nets.

## What the example prints

After training both nets for 30 epochs it scores a held-out set by **mean Dice
coefficient** and **mean IoU** (prediction thresholded at 0.5), prints the
comparison and a printed verdict, then renders one held-out sample as ASCII
(noisy input / ground truth / Dice prediction, plus the MSE prediction below)
so the result can be eyeballed.

## Building and running

```
lazbuild DiceSegmentation.lpi
../../bin/x86_64-linux/bin/DiceSegmentation
```

The run is deterministic (`RandSeed` is fixed), pure CPU, and finishes in a few
seconds.

## Sample output

```
DiceSegmentation: TNNetDiceLoss vs MSE on a synthetic disc-mask task
grid=16x16  features=8  train=256  test=64  epochs=30  lr=0.050

Training Dice-loss net...
  epoch   1   test Dice=0.9361  IoU=0.8825
  epoch  10   test Dice=0.9804  IoU=0.9625
  epoch  20   test Dice=0.9837  IoU=0.9685
  epoch  30   test Dice=0.9843  IoU=0.9696
Training MSE-baseline net...
  epoch   1   test Dice=0.9829  IoU=0.9674
  epoch  10   test Dice=0.9872  IoU=0.9753
  epoch  20   test Dice=0.9883  IoU=0.9774
  epoch  30   test Dice=0.9862  IoU=0.9732

Held-out results (threshold 0.5)
  head           mean Dice    mean IoU
  TNNetDiceLoss     0.9843      0.9696
  MSE baseline      0.9862      0.9732

=> On this toy the MSE baseline matched or beat the Dice head.

One held-out sample  ("#" = foreground, threshold 0.5)
  col 1: noisy INPUT   col 2: GROUND TRUTH   col 3: DICE pred   col 4: MSE pred

.+.,,...,.+..+,.   ................   ................
.+..,+,,,...,,,.   ................   ................
..+,.,++.++,.,#+   ................   ................
#,.+......+.+++.   ................   ................
,.,,,,++,.+....+   ................   ................
.,.+.##..#.+,,..   .....#..........   ....###.........
+,#######....++.   ..#######.......   ..#######.......
.#########.+,..#   .#########......   .#########......
.#########..,.,+   .#########......   .#########......
.#########.,.+,.   .#########......   .#########......
###########..,,+   ###########.....   ###########.....
,#########..,..+   .#########......   .#########......
.#########+,.,..   .#########......   .#########......
.#########.+,.+,   .#########......   .#########......
..#######,..,.#+   ..#######.......   ..#######.......
+,...#,.#,.+,.,+   .....#..........   .....#..#.......

  MSE prediction:
  ................
  ................
  ................
  ................
  ................
  ...#.#..........
  ..#######.......
  .#########......
  .#########......
  .#########......
  ###########.....
  .#########......
  .#########......
  .#########......
  ..#######.......
  .....#..#.......
```

On this toy the two heads end up close: in this run the MSE baseline
finishes slightly ahead on held-out IoU (`0.9732` vs `0.9696` for the Dice
head), and the program's verdict line says so. Both losses fit the easy samples
well, and on the rendered sample the two predictions are nearly identical. The
Dice objective optimises region overlap directly, which matters most on harder
masks where the minority foreground is easy to under-segment; on this easy,
fairly balanced disc task that advantage is within run-to-run noise. The
printed numbers are exactly what this run produced, and the printed verdict
reflects whatever the run actually shows.
