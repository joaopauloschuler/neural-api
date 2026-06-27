# DepthEstimation: monocular depth with DPT / Depth-Anything

The demo for the **DPT / Depth-Anything** importer
(`BuildDPTFromSafeTensors`, `neural/neuralpretrained.pas`). Depth Anything
(Yang et al. 2024) and the DPT family (Ranftl et al. 2021, *Vision Transformers
for Dense Prediction*) produce a dense per-pixel **depth regression** from a
single image.

## What it does

A plain ViT / DINOv2 transformer backbone is paired with a convolutional
"reassemble + fusion" neck and a small depth head: four intermediate encoder
stages are projected and resized into a feature pyramid, RefineNet-style additive
fusion blocks merge them coarse-to-fine, and a 3-conv head emits a
**single-channel** per-pixel depth map at the full input resolution.

The example loads the committed pico fixture with `BuildDPTFromSafeTensors`, runs
it on a small synthetic CPU image, and renders the resulting depth map as an
**ASCII grayscale ramp** (near = bright glyph, far = dark glyph).

## Running

```
cd examples/DepthEstimation
lazbuild DepthEstimation.lpi --build-mode=Release   # or: fpc -Fu../../neural DepthEstimation.lpr
./DepthEstimation
```

The program probes both `../../tests/fixtures/` and `tests/fixtures/`, so it runs
from either the example directory or the repo root.

## Notes

- Default input is the committed Depth-Anything parity fixture
  `tests/fixtures/tiny_dpt.safetensors` (a tiny DINOv2-backbone model) with random
  weights, run on a **synthetic** radial+diagonal gradient image — so the depth
  map is a forward-path smoke, not a meaningful scene depth.
- A real run loads `depth-anything/Depth-Anything-V2-Small-hf` the same way and
  processes a photograph; the math is identical, only the checkpoint and image
  differ.
- Inference only, pure CPU, finishes well under a second.

Coded by Joao Paulo Schwarz Schuler with Claude (AI).
