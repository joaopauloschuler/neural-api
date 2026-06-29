# SwinIRRestore: transformer image super-resolution

The demo for `BuildSwinIRFromSafeTensors` (`neural/neuralpretrained.pas`), a
**transformer** image-restoration import. SwinIR (Liang et al. 2021, *SwinIR:
Image Restoration Using Swin Transformer*,
[arXiv:2108.10257](https://arxiv.org/abs/2108.10257)) stacks Residual Swin
Transformer Blocks (RSTB = a few Swin window / shifted-window attention layers + a
`3×3` conv + a residual over the block) on a shallow conv stem, then a
pixel-shuffle upsample tail for classical super-resolution.

Architecturally distinct from the CNN-only RRDBNet/ESRGAN SR path and the
SimpleGate-CNN NAFNet denoiser. The window / shifted-window attention reuses the
landed Swin building blocks (`TNNetWindowAttention` + relative_position_bias +
cyclic-shift mask, `TNNetGatherTokens` partition/reverse); the new pieces are the
conv stem, the RSTB residual conv, and the `TNNetDepthToSpace` pixel-shuffle
upsample.

## Build / run

```
cd examples/SwinIRRestore
lazbuild SwinIRRestore.lpi --build-mode=Release
# run from the repo root so the fixture path resolves:
../../bin/x86_64-linux/bin/SwinIRRestore
../../bin/x86_64-linux/bin/SwinIRRestore model.safetensors [config.json]
```

## Input

The official SwinIR checkpoints are large and not obtainable offline, so — like
the repo's RRDBNet / NAFNet pico fixtures — this falls back to the committed
config-faithful **random** pico SwinIR (`tests/fixtures/tiny_swinir.safetensors`
+ `..._config.json`, parity-checked < 1e-4 against a float64 numpy oracle in
`TestNeuralPretrained.pas`). The pico net is random (not trained), so this is a
**wiring/throughput smoke**: it runs the SR forward on a deterministic synthetic
striped image and reports the upscaled shape. Pass a real `.safetensors`
(+ `config.json` sibling) to upscale with your own trained checkpoint.

## Output

Writes `swinir_input.ppm` and `swinir_upscaled.ppm` (P6 color images) and prints
the config plus a small ASCII intensity preview of input / upscaled (the upscaled
side is `Upscale×` larger). Pure CPU.
