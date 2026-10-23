# Segment Anything (SAM) — promptable click → mask

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
6. writes `<out>_mask.png` (white = mask) and `<out>_overlay.png` (photo with the
   mask tinted red), both at the photo's resolution.

### Key neural-api pieces (`neuralpretrained`)

- **`TSAMPredictor`** — holds the encoder net, the open checkpoint and the
  decoder weight cache. `SetImage` preprocesses and encodes once; each `Predict`
  (points with labels, and/or a box, single or 3 masks + IoU scores) is
  decoder-only.
- **`SAMPreprocessImage`**, **`SAMScaleCoordinates`**, **`SAMPostprocessMasks`**,
  **`SAMResizedImageSize`** — the processor steps on their own.
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
```

Options:

- `--image FILE` — any image the library loader reads (PNG, JPEG, BMP, ...).
  Without it a synthetic 96x64 image is segmented.
- `--point X,Y[,LABEL]` — repeatable; photo pixels, LABEL 1 = foreground
  (default), 0 = background. Without any prompt the click is the image centre.
- `--box X0,Y0,X1,Y1` — top-left / bottom-right corner in photo pixels.
- `--multimask` — ask for SAM's 3 masks and keep the highest predicted IoU.
- `--config FILE` — config.json (default: beside the checkpoint).
- `--out PREFIX` — output file prefix (default `sam`).

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
- **Status:** coded; not yet tested on a real sam-vit-* checkpoint.

Coded by Claude (AI).
