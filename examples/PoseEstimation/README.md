# PoseEstimation: human-pose keypoints with ViTPose

The demo for the **ViTPose** importer
(`BuildViTPoseFromSafeTensors`, `neural/neuralpretrained.pas`; Xu et al. 2022,
*ViTPose: Simple Vision Transformer Baselines for Human Pose Estimation*). A
top-down single-person keypoint estimator.

## What it does

A plain ViT transformer backbone runs over a cropped person image and a small
deconvolution head turns the patch-grid features into one 2-D **heatmap per
keypoint** (ReLU → bilinear upsample → 3×3 conv to `num_joints` channels). The
`(x, y)` location of each joint is read out by a spatial **argmax** over its
heatmap.

The example loads the committed pico fixture with `BuildViTPoseFromSafeTensors`,
runs it on a small synthetic CPU image, decodes the per-joint peaks with
`DecodeViTPoseKeypoints`, and renders the keypoints as an **ASCII plot** over the
heatmap grid (each joint drawn as a digit at its peak).

## Running

```
cd examples/PoseEstimation
lazbuild PoseEstimation.lpi --build-mode=Release   # or: fpc -Fu../../neural PoseEstimation.lpr
./PoseEstimation
```

The program probes both `../../tests/fixtures/` and `tests/fixtures/`, so it runs
from either the example directory or the repo root.

## Notes

- Default input is the committed parity fixture
  `tests/fixtures/tiny_vitpose.safetensors` (a tiny ViT-backbone model) with random
  weights, run on a **synthetic** radial+diagonal gradient image — so the decoded
  keypoints are a forward-path smoke, not real joints.
- A real run loads `usyd-community/vitpose-base-simple` the same way and processes
  a normalized detector person crop; the math is identical, only the checkpoint and
  image differ.
- Inference only, pure CPU, finishes well under a second.

Coded by Joao Paulo Schwarz Schuler with Claude (AI).
