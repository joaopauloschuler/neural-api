# Segment Anything (SAM) — promptable click → mask

A self-contained, CPU/offline demo of **Segment Anything** (Kirillov et al. 2023,
["Segment Anything"](https://arxiv.org/abs/2304.02643)) running the real
promptable-segmentation pipeline end to end on the committed pico SAM parity
fixture (or any `facebook/sam-vit-*` checkpoint you pass in).

## What it does

SAM is a promptable model: a heavy ViT-det **image encoder** produces a dense
`(Grid, Grid, OutCh)` embedding once per image, and a lightweight
prompt-conditioned **mask decoder** then turns a user **click** (point) into a
binary mask cheaply. This example:

1. builds and runs the image encoder on a synthetic deterministic gradient image;
2. feeds **one positive point click** (image-pixel coordinates) into the mask
   decoder;
3. thresholds the low-res mask logits at 0 (sigmoid 0.5) and writes the binary
   mask as a PPM.

### Key neural-api pieces (both float64-parity importers, `neuralpretrained`)

- **`BuildSAMFromSafeTensors`** — lands the ViT-det image **encoder**, returning
  a `TNNet`; config read via `TSAMConfig` / `ReadSAMMaskDecoderConfig`.
- **`RunSAMMaskDecoder`** — the v1 single-point / single-mask **decoder**: prompt
  encoder + two-way transformer + transposed-conv upscale + hypernetwork dot,
  driven from a `TNNetSafeTensorsReader` over the same checkpoint.

## Running

Run from the **repo root** so the committed pico fixture resolves (it is also
probed relative to the example dir):

```
examples/SegmentAnything/SegmentAnything
examples/SegmentAnything/SegmentAnything /path/to/sam-vit-base/model.safetensors [clickX clickY]
```

Build first with the usual conventions:

```
cd examples/SegmentAnything
lazbuild SegmentAnything.lpi --build-mode=Release
```

(Or compile the `.lpr` with `fpc -Fu../../neural`.)

## Inputs / outputs

- **Default checkpoint:** `tests/fixtures/tiny_sam.safetensors` (+
  `tiny_sam_config.json`), committed — works fully offline.
- **Input image:** synthetic smooth dyadic gradient generated in code (no image
  I/O); the click defaults to the image centre, overridable via `clickX clickY`
  arguments.
- **Output:** `sam_mask.ppm` — a binary mask (P3), upscaled ~4× for visibility;
  the program prints the embedding grid, the click, the mask-logit resolution and
  the foreground pixel count.

Coded by Claude (AI).
