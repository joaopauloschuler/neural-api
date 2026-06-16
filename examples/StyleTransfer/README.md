# Neural Style Transfer (Gatys et al. 2016)

A command-line demo of **neural style transfer** ([Gatys, Ecker & Bethge,
*Image Style Transfer Using Convolutional Neural Networks*, CVPR 2016](https://arxiv.org/abs/1508.06576)).
It re-paints a *content* image in the texture/colour *style* of a second image
by optimising the **pixels of a canvas** so that, when both are pushed through a
frozen VGG-16, the canvas matches the content image's deep activations and the
style image's **Gram-matrix** statistics.

This is the perceptual-loss flip-side of the [GradientAscent](../GradientAscent)
example: the same "backprop to the INPUT volume and step the input" machinery,
but the loss is computed from a pretrained VGG feature extractor rather than the
net's own logits.

## What it does

1. **Builds a VGG-16 feature extractor** with `BuildVGGFromSafeTensors` /
   `BuildVGGFromSafeTensorsEx` (see `neural/neuralpretrained.pas`). `FeatureTapStage`
   is forced to `5` so the net is TRUNCATED at `relu5` (the AdaptiveAvgPool + FC
   classifier is dropped). All five per-stage taps `relu1_2, relu2_2, relu3_3,
   relu4_3, relu5_x` are still exposed via the `TapLayerIdx` out-parameter — after
   `NN.Compute(img)` you read `NN.Layers[TapLayerIdx[k]].Output`.
2. **Content loss** — MSE between the canvas and content activations at one deep
   tap (`relu4_3`). Gradient `dL/dF = 2·(F − F_target)`.
3. **Style loss** — MSE between the **Gram matrices** of the canvas and style
   activations at all five taps. The Gram matrix is the new piece of
   code, a plain helper (no new layer class):

   ```
   G[i,j] = (1 / (C·H·W)) · Σ_{x,y} F[x,y,i] · F[x,y,j]
   ```

   with the implied gradient `dL/dF[x,y,i] = (4/(C·H·W)) · Σ_j (G − G_target)[i,j] · F[x,y,j]`.
4. **Optimises the canvas pixels** by gradient descent. The content/style
   gradients are injected into the tap layers' `OutputError`, then a single
   manual `Backpropagate()` from the truncated net's last layer carries them all
   the way back to the input layer (the conv/ReLU backward kernels *add* into the
   previous layer's `OutputError`). The step is `canvas := canvas − lr · dInput`.
   Finally the canvas is de-normalised and written as a PNG.

The VGG is **frozen** while the pixels move: `SetBatchUpdate(true)` makes
`Backpropagate` accumulate weight deltas instead of applying them, and the
example never calls `UpdateWeights()`, so the conv weights never change.

## Running

```
# Self-contained pipeline demo: tiny VGG fixture + synthetic 64x64 images
StyleTransfer --iter=40

# Real artistic transfer (point at torchvision VGG-16 safetensors + your images)
StyleTransfer --vgg vgg16.safetensors --config config.json \
              --content photo.png --style painting.png --out stylized.png \
              --size 128 --iter 300 --styleweight 1e4
```

Options:

| flag | meaning | default |
|------|---------|---------|
| `--vgg <file>` | VGG-16 safetensors | committed tiny fixture |
| `--config <file>` | VGG config JSON | matching fixture config |
| `--content <file>` | content image | synthetic gradient |
| `--style <file>` | style image | synthetic stripes |
| `--out <file>` | output PNG | `stylized.png` |
| `--size <n>` | working square size | from VGG config |
| `--iter <n>` | optimisation steps | 40 |
| `--lr <f>` | step size | 5.0 |
| `--styleweight <f>` | style/content balance | 1e3 |

## Fixture vs. real weights

With **no** `--vgg` the example uses the committed tiny VGG fixture
(`tests/fixtures/tiny_vgg16.safetensors`, image size 64, randomly-initialised
weights) and synthesises a content image (smooth diagonal gradient) and a style
image (high-frequency stripes). This **proves the whole pipeline end-to-end and
is CI-runnable** — the style loss falls from ~9.82 to ~0.68 in 40 steps in about
**0.1 s** on CPU — but it is NOT artistic: the fixture weights carry no learned
visual features.

For genuine artistic results, pass real **torchvision VGG-16** weights exported
to safetensors (`--vgg`/`--config`) and real content/style images, and raise
`--size`, `--iter`, and `--styleweight`. The pipeline is identical; only the
feature extractor's quality changes.

## A note on the importer

A convolution sizes its prev-layer-error scratch buffer from the previous
layer's `OutputError` size at wiring time, so for backprop to reach the input
PIXELS the input layer's error collection must be enabled **before** the first
conv is added. `BuildVGG` now enables input error collection when built with
`pInferenceOnly = false`, so an imported VGG can be used directly for
input-gradient methods (style transfer, perceptual gradient ascent, saliency).
This is harmless for plain forward inference.

## Follow-ups

* A real torchvision VGG-16 artistic run (export + larger images).
* A **total-variation** regulariser to suppress high-frequency noise.
* An **L-BFGS**-style optimiser (the paper's choice) instead of plain SGD.
* **Colour preservation** (luminance-only transfer / colour histogram match).
