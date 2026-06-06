# Sub-Pixel Super-Resolution (PixelShuffle 2x)

A tiny, self-contained demonstration that a convolutional network with a
**sub-pixel (depth-to-space) upsampling head** — the in-tree layer
[`TNNetPixelShuffle`](../../neural/neuralnetwork.pas) — can **learn a 2x
super-resolution mapping** on purely synthetic data. No external dataset, no
new layers, pure CPU, deterministic.

## The idea: sub-pixel / "pixel shuffle" upsampling

Instead of upsampling with a transposed convolution, a super-resolution network
keeps all of its convolutions at the **low resolution** and produces an output
tensor with `r*r * C` channels. A final, **parameter-free** reshape ("pixel
shuffle" / depth-to-space) folds those `r*r` groups of channels into an
`r`-times-larger spatial grid with `C` channels:

```
(SizeX, SizeY, r*r*C)  --PixelShuffle(r)-->  (r*SizeX, r*SizeY, C)
```

The in-tree `TNNetPixelShuffle(r)` implements exactly the mapping

```
out[r*x+i, r*y+j, c] = in[x, y, c*r*r + i*r + j]
```

This is the upsampling head of Shi et al. (2016), *Real-Time Single Image and
Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural
Network* (ESPCN). Doing the upsample as a deterministic reshape (and letting the
convolutions learn the `r*r` sub-pixel channels) is cheaper than a transposed
convolution and avoids its checkerboard artefacts. This example uses `r = 2`.

## The task (synthetic, deterministic)

- **Inputs**: random low-resolution `8x8` single-channel "blocky" tiles. Each
  tile is a coarse `4x4` grid of random intensities in `[0,1]`,
  nearest-neighbour-expanded to `8x8`, so the patterns are genuinely
  low-frequency (a fair SR target).
- **Targets**: the **fixed, known ground-truth 2x upscale** of each LR tile,
  namely a `16x16` nearest-neighbour expansion (each LR pixel becomes a `2x2`
  output block). This is a deterministic function of the input, so a *perfect*
  mapping exists and "did it learn?" has an unambiguous answer.

## The net (3 trainable conv layers + the reshape head)

```
Input(8,8,1)
  -> TNNetConvolutionReLU(16, 3,1,1)     // low-res feature extractor
  -> TNNetConvolutionReLU(16, 3,1,1)     // low-res feature extractor
  -> TNNetConvolutionLinear(4, 3,1,1)    // produce r*r*C = 4*1 = 4 channels
  -> TNNetPixelShuffle(2)                // (8,8,4) -> (16,16,1) : the 2x upsample
```

All convolutions run at `8x8` (`3024` weights total); only the final
parameter-free `TNNetPixelShuffle(2)` reshape changes the resolution.

## What is printed (the headline)

MSE and PSNR of the net's `16x16` output vs the ground-truth `16x16` target,
measured on a **held-out test set BEFORE** training (random init) and **AFTER**
training. The PixelShuffle head, fed by the learned convolutions, drives the MSE
down by orders of magnitude and the PSNR up by tens of dB — i.e. the network
**learns** the 2x super-resolution mapping.

## Built-in correctness gate (printed PASS/FAIL, `Halt(1)` on failure)

1. **test MSE drops by at least 10x** after training — the net learned the upscale.
2. **test PSNR after training ≥ 25 dB** — clean reconstruction.

## How to run

```
# From this directory, with Free Pascal (fpc) installed:
fpc -O3 -Mobjfpc -Sh -Fu../../neural -dRelease SubPixelSuperRes.lpr
./SubPixelSuperRes
```

Or open `SubPixelSuperRes.lpi` in Lazarus and build the *Release* mode. Pure
CPU, no external data, deterministic (`RandSeed = 424242`, single-threaded via
manual `Compute`/`Backpropagate`), finishes in about 15 seconds.

## Sample output (real, seed 424242)

```
================================================================
SubPixelSuperRes: a PixelShuffle (sub-pixel) net LEARNS a 2x
super-resolution mapping on synthetic data.
================================================================
Task: LR 8x8 (coarse 4x4 grid) -> HR 16x16 (2x upscale), 1 channel.
Train=256, Test=64 (held-out).
Net: Input(8,8,1) -> ConvReLU(16,3,1,1) -> ConvReLU(16,3,1,1)
     -> ConvLinear(4,3,1,1) -> PixelShuffle(2) -> (16,16,1).
Mini-batch SGD  batch=16  LR=0.001  momentum=0.90  epochs=120  RandSeed=424242

Net weights: 3024

BEFORE training:  test MSE = 0.394019   PSNR =   4.04 dB
  epoch  20  train MSE = 0.000840
  epoch  40  train MSE = 0.000389
  epoch  60  train MSE = 0.000280
  epoch  80  train MSE = 0.000184
  epoch 100  train MSE = 0.000137
  epoch 120  train MSE = 0.000113

AFTER  training:  test MSE = 0.000120   PSNR =  39.21 dB

=== Results ===
test MSE  : 0.394019 -> 0.000120  (3286.6x reduction)
test PSNR :   4.04 dB ->  39.21 dB  (+35.17 dB)

=== Correctness gate ===
[PASS] test MSE dropped >= 10x (0.394019 -> 0.000120): the net learned the upscale.
[PASS] test PSNR after training = 39.21 dB (must be >= 25.0 dB): clean reconstruction.

TAKEAWAY: with all convolutions kept at low resolution and a single
parameter-free TNNetPixelShuffle(2) head doing the depth-to-space
reshape, the network learns the 2x super-resolution mapping -- MSE
falls by orders of magnitude and PSNR rises by tens of dB.

=> ALL CHECKS PASS: PixelShuffle net learned the 2x super-resolution.
Total wall-clock: 14.0 s
```

### Reading the result

At random init the net's output is uncorrelated with the target (test MSE
`0.39`, PSNR `4 dB`). After 120 epochs of mini-batch SGD the test MSE has fallen
to `0.00012` — a **~3300x reduction**, lifting PSNR from `4 dB` to `39 dB`. The
held-out test number tracks the training number, so this is the learned mapping
generalising to unseen tiles, not memorisation. The `TNNetPixelShuffle(2)` head
contributes **zero** trainable parameters; the convolutions learn to emit the
four sub-pixel channels that the shuffle folds into the `2x` grid.

## Notes / honesty

- This is a *toy* SR target (a known nearest-neighbour upscale) chosen so a
  perfect mapping provably exists and the "before vs after" improvement is
  unambiguous. The point is the **mechanism** — that `TNNetPixelShuffle` makes a
  learnable, parameter-free `2x` upsampling head — not state-of-the-art SR.
- **Learning rate**: this is a regression task trained against raw squared-error
  gradients, which are much larger than the soft-max/cross-entropy gradients of
  the classification examples. A small LR (`1e-3`) converges cleanly; `1e-2`
  plateaus and `3e-2` diverges.

## See also

- [`examples/SuperResolution`](../SuperResolution) — a full image
  super-resolution app (CIFAR-10) that also uses the pixel-shuffle idea on real
  images.

## References

- W. Shi, J. Caballero, F. Huszár, J. Totz, A. P. Aitken, R. Bishop, D.
  Rueckert, Z. Wang, *Real-Time Single Image and Video Super-Resolution Using an
  Efficient Sub-Pixel Convolutional Neural Network*, CVPR 2016.
  https://arxiv.org/abs/1609.05158
