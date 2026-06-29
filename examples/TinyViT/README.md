# Tiny Vision Transformer (`TNNet.AddPatchEmbedding`)

A tiny from-scratch Vision Transformer image-classification demo of
**`TNNet.AddPatchEmbedding`**, the reusable ViT-style patchify +
token-projection builder.

Reference: Dosovitskiy et al. 2021, *An Image is Worth 16x16 Words: Transformers
for Image Recognition at Scale* (arXiv:2010.11929).

## The builder

```pascal
function TNNet.AddPatchEmbedding(PatchSize, EmbedDim: integer;
  AddClassToken: boolean = false;
  AddPositionalEmbedding: boolean = true): TNNetLayer;
```

Given a 2D image input it turns the image into a token sequence in one call:

1. **Patchify** — a convolution with `kernel = stride = PatchSize` and
   `EmbedDim` output channels (`TNNetConvolutionLinear`). Each
   `PatchSize × PatchSize` patch becomes one `EmbedDim`-vector token. This is the
   standard ViT linear patch projection.
2. **Flatten** — the `(GridX, GridY, EmbedDim)` patch grid is reshaped into a
   `(SeqLen, 1, EmbedDim)` token sequence (`SeqLen = GridX*GridY`), the layout
   the rest of the transformer builders consume.
3. **Class token** *(optional, `AddClassToken`)* — a learnable `[CLS]` token is
   prepended at sequence position 0 (via `TNNetSoftPrompt`), giving `SeqLen+1`
   tokens. Pool/classify from this token as in ViT.
4. **Positional embedding** *(optional, `AddPositionalEmbedding`, default on)* —
   a learnable absolute positional embedding (`TNNetLearnedPositionalEmbedding`)
   is added over the final token positions (it covers the `[CLS]` slot too).

This replaces the conv-stride-then-`TNNetReshape` boilerplate that patch-
tokenizing examples used to hand-roll inline. The input `SizeX`/`SizeY` must be
divisible by `PatchSize`.

## The demo

A self-contained synthetic **which-quadrant** task: each `8×8` single-channel
image is small noise with one bright spike planted in a random pixel; the label
is the quadrant (top-left / top-right / bottom-left / bottom-right) the spike
lands in. Deciding the class needs comparing token *positions* across the patch
grid — exactly what the positional embedding + self-attention provide.

```
Input(8,8,1)
  -> AddPatchEmbedding(2, 16, AddClassToken=true)   -> (17,1,16)  [16 patches + CLS]
  -> AddTransformerEncoderBlock(2, 32) x 2
  -> LayerNorm
  -> Crop(0,0,1,1)   (take the [CLS] token)         -> (1,1,16)
  -> FullConnectLinear(4) -> SoftMax
```

It converges to 100% train/test accuracy in well under a minute on CPU
(deterministic, fixed `RandSeed`). Printing is NaN/Inf-guarded.

## Build & run

```bash
lazbuild TinyViT.lpi   # or: fpc -Fu../../neural -Mobjfpc -Sh -O2 TinyViT.lpr
./TinyViT
```
