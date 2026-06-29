# AdaIN style transfer (`TNNetAdaIN`)

A tiny, CPU-friendly demonstration of **fast arbitrary style transfer with
Adaptive Instance Normalization** (Huang & Belongie, ICCV 2017). Unlike the
optimisation-based `../StyleTransfer` (minutes per image, one fixed style), AdaIN
stylizes a (content, style) pair in a **single feed-forward pass**. No dataset
download; runs in seconds.

## What it does

A shared shallow conv encoder encodes a content image and a style image, the
new **`TNNetAdaIN`** layer transfers the per-channel style statistics onto the
instance-normalized content features, and a conv decoder paints the stylized
RGB image. The network is built in `BuildAdaINNet`:

```pascal
ContentInput := NN.AddLayer(TNNetInput.Create(ImgSize, ImgSize, 3, 1));
StyleInput   := NN.AddLayerAfter(TNNetInput.Create(ImgSize, ImgSize, 3, 1), 0);
// shared conv encoder over each branch -> ContentFeat, StyleFeat
AdaIN := NN.AddLayerAfter(TNNetAdaIN.Create(ContentFeat, StyleFeat), ContentFeat);
// conv decoder back to 3 channels
```

`TNNetAdaIN.Create(ContentFeatures, StyleFeatures)` has **no learnable
parameters**. The decoder is trained for a handful of iterations on a toy
identity-reconstruction objective (`decode(AdaIN(x, x)) ≈ x`, feeding the same
image as content and style) so it learns to invert the encoder; the program then
stylizes a synthetic content image **by** a synthetic style image in one forward
pass and reports the stylized output statistics.

## Running

```
cd examples/AdaINStyleTransfer
lazbuild AdaINStyleTransfer.lpi --build-mode=Release
ulimit -v 3000000
./AdaINStyleTransfer
```

(Or `fpc` the `.lpr` with `-Fu../../neural` as in the other examples.)

## Notes

- Inputs are **synthetic** 16×16 RGB images generated in code (a gradient-striped
  content image, a warm-palette style image) — there is no image loading.
- This is a wiring/forward-and-backward demonstration of the AdaIN path, **not**
  an artistic result: the encoder/decoder are tiny and trained only briefly.
- The program asserts the stylized output is finite.

Coded by Claude (AI).
