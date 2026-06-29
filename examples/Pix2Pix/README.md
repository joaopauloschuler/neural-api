# Pix2Pix: paired conditional image-to-image translation

A conditional GAN that maps a 1-channel **grayscale** shapes image to its
3-channel **colorized** version (Isola et al. 2017, *Image-to-Image Translation
with Conditional Adversarial Networks*). Synthetic, generated-in-code task; no
download, pure CPU. The default SMOKE run finishes in well under five minutes.

Distinct from the unconditional `../VisualGAN` (noise → image): the generator here
is **conditioned** on the input image and the output is a deterministic translation
of it, scored by a PatchGAN discriminator.

## What it does

Each sample is one or two random filled shapes on a dark background. The input is
the grayscale rendering; the paired target colorizes the same geometry by a fixed
rule (circle → red, rectangle → green, background → dark blue), so the net must
both reconstruct the silhouettes and infer each shape's color from its geometry.
RGB is in `[-1,1]` to match the generator's Tanh output.

- **Generator** — a U-Net built by one `TNNet.AddUNet` call (same builder as
  `../UNetSegmentation`): `Depth` encoder stages, a bottleneck, `Depth` decoder
  stages with skip-concat, and a 1×1 head to 3 channels + Tanh.
- **Discriminator** — a fully-convolutional **PatchGAN** (composed from existing
  convs) that scores overlapping patches as real/fake. Its input is the condition
  stacked with the image (`[grayscale | R | G | B]`), using the **least-squares
  (LSGAN)** objective.

Training is a hand-rolled adversarial loop: the framework seeds a layer's output
error as `(output − target)` (exactly the LSGAN/MSE gradient). Each step trains D
(real→1, fake→0), then reads `d(adv)/d(pixels)` from D's input-layer gradient
(`EnableErrorCollection`), adds the L1 reconstruction gradient, and backpropagates
G directly. L1 makes the output sharp and correctly colored; the adversarial term
sharpens edges.

## Running

```
cd examples/Pix2Pix
fpc -Fu../../neural Pix2Pix.lpr
./Pix2Pix            # smoke run (fast, default)
./Pix2Pix --full     # longer training for a sharper result
```

## Output

Per-eval mean L1 (in `[0,2]` pixel units) and per-pixel color accuracy on a
held-out set, one held-out sample as an ASCII triptych (input / target /
generated), and `pix2pix_sample.ppm` (input | target | generated).

Coded by Claude (AI).
