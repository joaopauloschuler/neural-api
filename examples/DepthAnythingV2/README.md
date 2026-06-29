# Depth Anything V2 → normalized depth-map image

The demo for the named `BuildDepthAnythingV2FromSafeTensors` entry point
(`neural/neuralpretrained.pas`; model_type `depth_anything`:
`depth-anything/Depth-Anything-V2-Small/Base/Large-hf`). Depth Anything V2 (Yang
et al. 2024) is the DPT dense-prediction stack (Ranftl et al. 2021) on a
**DINOv2** ViT backbone (S/B/L): four selected intermediate encoder stages (the
backbone's `out_indices`) are projected and resized into a feature pyramid,
RefineNet-style additive fusion blocks merge them coarse-to-fine, and a 3-conv
head emits a single-channel per-pixel **relative-depth** map at full input
resolution.

This example differs from [DepthEstimation](../DepthEstimation) (which prints an
ASCII grayscale ramp of the generic DPT path): it loads via the named V2 entry
point, uses the committed pico fixture that hooks **non-last-4** stages
(`out_indices=[2,3,5,6]`) to exercise the `out_indices` wiring, and **writes real
image files** — a min/max-normalized 8-bit grayscale **PGM (P5)** plus a color
**PPM (P6)** inverse-depth visualization (near = warm, far = cool), the way the
Depth Anything demos visualize depth — alongside a small ASCII preview.

## Build / run

```
cd examples/DepthAnythingV2
lazbuild DepthAnythingV2.lpi --build-mode=Release
../../bin/x86_64-linux/bin/DepthAnythingV2                   # pico smoke
../../bin/x86_64-linux/bin/DepthAnythingV2 model.safetensors # real checkpoint
```

## Input

Default is a **smoke run** on the committed pico fixture
`tests/fixtures/tiny_depth_anything_v2.safetensors` (+ `..._config.json`). The
input image is generated in-code (a smooth synthetic radial-gradient scene at the
config's `image_size`), so no dataset is needed. Pass a real `.safetensors` path
as the first argument (with `config.json` beside it) to run a real checkpoint on
the same synthetic input.

## Output

`depth_pico.pgm` / `depth_pico.ppm` (or `depth_real.*` for a real checkpoint),
plus the printed depth range and an ASCII preview. Pure CPU, well under a second
on the fixture. Importer parity is asserted to max |diff| < 1e-4 vs the HF
`DepthAnythingForDepthEstimation` float64 forward in `TestDepthAnythingV2Parity`.
