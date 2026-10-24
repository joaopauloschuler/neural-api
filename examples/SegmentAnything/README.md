# Segment Anything (SAM) — promptable click → mask, and "segment everything"

A self-contained, CPU/offline demo of **Segment Anything** (Kirillov et al. 2023,
["Segment Anything"](https://arxiv.org/abs/2304.02643)) running the real
promptable-segmentation pipeline end to end on the committed pico SAM parity
fixture (or any `facebook/sam-vit-*` checkpoint you pass in).

## What it does

SAM is a promptable model: a heavy ViT-det **image encoder** produces a dense
`(Grid, Grid, OutCh)` embedding once per image, and a lightweight
prompt-conditioned **mask decoder** then turns user **clicks** (points) or a box
into a mask cheaply. This example:

1. loads a photo (`--image`) or builds a synthetic 96x64 image;
2. preprocesses it like HF `SamImageProcessor`: PIL bilinear resize of the longest
   side to `image_size` (1024 for sam-vit-*), rescale by 1/255, ImageNet
   mean/std normalize, zero pad right and bottom to `image_size` x `image_size`;
3. runs the image encoder once;
4. rescales the clicks / box from photo pixels to the resized frame and runs the
   mask decoder;
5. upscales the low-res mask logits to `image_size`, crops off the padding,
   resizes to the photo's own size (bilinear, as `post_process_masks`) and
   thresholds at 0;
6. with `--mask-input FILE`, gives the first decode a mask prompt read from a
   mask image (see [Mask prompt from a file](#mask-prompt-from-a-file));
7. with `--refine N`, feeds the kept low-res mask logits back N times as a
   **mask prompt** (SAM's `mask_input`, transformers `input_masks`) together with
   the same clicks / box, one mask per pass, as the SAM notebooks do;
8. writes `<out>_mask.png` (white = mask) and `<out>_overlay.png` (photo with the
   mask tinted red), both at the photo's resolution.

With `--everything` it runs SAM's **automatic mask generator** instead (the
transformers `mask-generation` pipeline): a grid of single-point prompts over
the image (and, with `--crop-layers`, over overlapping crops, each encoded on its
own), 3 masks per point, kept when the predicted IoU and the stability score
pass their thresholds, then box non-maximum suppression. It writes
`<out>_everything.png`: every kept mask blended in its own colour, its box
outlined, and prints each mask's IoU, stability score, box and area.

### Key neural-api pieces (`neuralpretrained`)

- **`TSAMPredictor`** — holds the encoder net, the open checkpoint and the
  decoder weight cache. `SetImage` preprocesses and encodes once; each `Predict`
  (points with labels, and/or a box, and/or a low-res mask prompt, single or 3
  masks + IoU scores) is decoder-only. `LowResLogits` holds the last decode's
  `(4*Grid, 4*Grid, NumMasks)` logits; one channel (`CopyChannels`) is the
  `MaskInput` of the next `Predict`.
- **`BuildSAMMaskEmbedding`** — `prompt_encoder.mask_embed` (conv 2x2 s2 ->
  LayerNorm2d -> GELU -> conv 2x2 s2 -> LayerNorm2d -> GELU -> conv 1x1) as a
  small `TNNet`; the decoder adds its output to the image embedding in place of
  `no_mask_embed`.
- **`SAMPreprocessImage`**, **`SAMScaleCoordinates`**, **`SAMPostprocessMasks`**,
  **`SAMResizedImageSize`** — the processor steps on their own.
- **`SAMMaskImageToInputLogits`** — a mask image at the photo's size -> the
  `(4*Grid, 4*Grid, 1)` low-res logits `MaskInput` of `Predict`.
- **`TSAMPredictor.GenerateMasks`** + **`TSAMGeneratedMasks`** — automatic mask
  generation; the masks are stored run-length coded and decoded on demand
  (`GetMask`). **`SAMBuildPointGrid`**, **`SAMGenerateCropBoxes`** and
  **`SAMStabilityScore`** are its steps; the box NMS is `NeuralGreedyNMS`.
- **`BuildSAMFromSafeTensors`** / **`RunSAMMaskDecoderEx`** — the encoder and
  decoder importers underneath (float64 parity < 1e-4).

## Running

Run from the **repo root** so the committed pico fixture resolves (it is also
probed relative to the example dir):

```
examples/SegmentAnything/SegmentAnything
examples/SegmentAnything/SegmentAnything /path/to/sam-vit-base/model.safetensors \
  --image photo.jpg --point 320,240 --point 100,80,0
examples/SegmentAnything/SegmentAnything /path/to/model.safetensors \
  --image photo.jpg --box 50,40,400,300 --multimask --out photo
examples/SegmentAnything/SegmentAnything /path/to/model.safetensors \
  --image photo.jpg --point 320,240 --multimask --refine 2 --out refined
examples/SegmentAnything/SegmentAnything /path/to/model.safetensors \
  --image photo.jpg --mask-input refined_mask.png --point 320,240 --out again
examples/SegmentAnything/SegmentAnything /path/to/model.safetensors \
  --image photo.jpg --everything --out photo
```

On the random pico fixture the default thresholds keep nothing; this relaxed
run keeps a mask and exercises the crop layer:

```
examples/SegmentAnything/SegmentAnything --everything --points-per-side 4 \
  --crop-layers 1 --crop-points-downscale 2 --pred-iou-thresh 0.5 \
  --stability-thresh 0.7 --stability-offset 0.05 --nms-thresh 0.8
```

Options:

- `--image FILE` — any image the library loader reads (PNG, JPEG, BMP, ...).
  Without it a synthetic 96x64 image is segmented.
- `--point X,Y[,LABEL]` — repeatable; photo pixels, LABEL 1 = foreground
  (default), 0 = background. Without any prompt the click is the image centre.
- `--box X0,Y0,X1,Y1` — top-left / bottom-right corner in photo pixels.
- `--multimask` — ask for SAM's 3 masks and keep the highest predicted IoU.
- `--mask-input FILE` — a mask image with the photo's size (white = object,
  black = background, grey in between) used as the mask prompt of the first
  decode. With no `--point` / `--box` it is a mask-only prompt. A `<out>_mask.png`
  from an earlier run works as is.
- `--refine N` — feed the kept low-res mask back N times as a mask prompt with
  the same points / box (default 0). Each pass prints its predicted IoU.
- `--config FILE` — config.json (default: beside the checkpoint).
- `--out PREFIX` — output file prefix (default `sam`).
- `--everything` — automatic mask generation (prompts are ignored), with:
  - `--points-per-side N` — grid points per side (default 32, so 1024 decoder
    runs per crop);
  - `--crop-layers N` — extra crop layers, layer i has 4^i crops (default 0);
  - `--crop-points-downscale N` — layer i uses N^i fewer points per side (1);
  - `--pred-iou-thresh T` (0.88), `--stability-thresh T` (0.95),
    `--stability-offset T` (1), `--nms-thresh T` (0.7).

### Mask prompt from a file

SAM's mask prompt is not a binary mask: it is a `256 x 256` map of mask
**logits** (`4*Grid` per side, 16 x 16 on the pico fixture) in the padded frame
of the encoder input, normally the `low_res_logits` of an earlier decode.
`SAMMaskImageToInputLogits` builds that map from a mask image:

1. PIL `L` conversion of the image (an RGB mask can be read as is);
2. the same longest-side resize to `image_size` as the photo (PIL bilinear,
   float), with the right / bottom padding set to black;
3. a PIL bilinear (antialiased) resize of that `image_size x image_size` frame to
   the low-res size, so each low-res cell covers the same photo area as the cells
   of the decoder's own `LowResLogits`;
4. the linear map `0..255 -> -L..+L` with **L = 10** (`csSAMMaskInputLogit`):
   white -> +10, black -> -10, mid grey -> 0.

Why logits and why L = 10: SAM was trained with an earlier decode's
unthresholded logits as the mask prompt, so a raw 0/1 map would sit at the
decision boundary. L = 10 is the scale SAM 2 uses to turn a binary mask into
mask logits (`_use_mask_as_output`: `mask * 20 - 10`, `sigmoid(-10) = 4.5e-5`),
and SAM 2 also brings masks to the low-res size with antialiased bilinear
resizing. The original SAM API only takes earlier logits, so it has no fixed
choice. A greyscale image acts as a soft mask: grey levels map to logits in
between. `--refine` feeds the decoder's own logits and skips this conversion.

Build first with the usual conventions:

```
cd examples/SegmentAnything
lazbuild -B SegmentAnything.lpi
```

## Inputs / outputs

- **Default checkpoint:** `tests/fixtures/tiny_sam.safetensors` (+
  `tiny_sam_config.json`), committed — works fully offline. Its weights are
  random, so the mask shape is meaningless; it checks the wiring.
- **Output:** `<out>_mask.png` and `<out>_overlay.png`; the program prints the
  prompts, each mask's predicted IoU and the foreground pixel count.
- **Parity:** `TestSAMProcessorEndToEndParity` compares the preprocessed pixels,
  scaled prompts, IoU scores and full-resolution mask logits with
  `SamImageProcessorPil` + `SamModel` (float64) on the pico fixture.
  `TestSAMMaskInputParity` compares the mask-prompt decodes (point refinement,
  box + mask, mask only, and the `TSAMPredictor` refine round trip) with
  `SamModel` given `input_masks`.
  `TestSAMMaskImageToInputLogits` checks the mask image -> low-res logits
  geometry on a non-square photo (a rectangle lands in the expected cells, the
  padding is -L).
  `TestSAMAutomaticMaskGenerationParity` compares the generated masks (count,
  IoU, stability score, box, every pixel) with transformers' `mask-generation`
  pipeline and its processor functions, with and without a crop layer.
- **Crop layers:** transformers' pipeline feeds crop points to the whole image;
  here each crop is encoded on its own with its grid scaled to the crop and its
  boxes moved to the image frame, as in the original SAM.
- **Status:** coded; not yet tested on a real sam-vit-* checkpoint.

Coded by Claude (AI).
