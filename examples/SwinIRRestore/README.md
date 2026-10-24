# SwinIRRestore: transformer image super-resolution

The demo for `BuildSwinIRFromSafeTensors` (`neural/neuralpretrained.pas`), a
**transformer** image-restoration import. SwinIR (Liang et al. 2021, *SwinIR:
Image Restoration Using Swin Transformer*,
[arXiv:2108.10257](https://arxiv.org/abs/2108.10257)) stacks Residual Swin
Transformer Blocks (RSTB = a few Swin window / shifted-window attention layers + a
`3×3` conv + a residual over the block) on a shallow conv stem, then one of
four reconstruction tails, named by the official `upsampler` argument:

| `upsampler` | Task | Tail |
| --- | --- | --- |
| `pixelshuffle` | classical SR (x2/x3/x4/x8) | `conv_before_upsample` → LeakyReLU(0.01) → per x2 step (one x3 step): `upsample.0`, `upsample.2`, … → pixel shuffle → `conv_last` |
| `pixelshuffledirect` | lightweight SR (x2/x3/x4) | `upsample.0` (embed_dim → upscale²·3) → pixel shuffle |
| `nearest+conv` | real-world SR (x2/x4) | `conv_before_upsample` → LeakyReLU(0.01) → per x2 step: nearest x2 → `conv_up1`/`conv_up2` → LeakyReLU(0.2); then `conv_hr` → LeakyReLU(0.2) → `conv_last` |
| `""` (upscale 1) | denoise / JPEG artifact removal | input + `conv_last(features)` |

When `config.json` has no `upsampler` field, `BuildSwinIRFromSafeTensors`
detects the tail from the checkpoint keys: `conv_up1` → `nearest+conv`,
`conv_before_upsample` → `pixelshuffle`, `upsample.0` alone →
`pixelshuffledirect`, none → the denoise tail. Every pixel shuffle follows the
torch `nn.PixelShuffle` channel order.

Architecturally distinct from the CNN-only RRDBNet/ESRGAN SR path and the
SimpleGate-CNN NAFNet denoiser. The window / shifted-window attention reuses the
landed Swin building blocks (`TNNetWindowAttention` + relative_position_bias +
cyclic-shift mask, `TNNetGatherTokens` partition/reverse); the new pieces are the
conv stem, the RSTB residual conv, and the `TNNetPixelShuffle` pixel-shuffle
upsample.

## Build / run

```
cd examples/SwinIRRestore
lazbuild SwinIRRestore.lpi --build-mode=Release
# run from examples/SwinIRRestore so the fixture path resolves:
../../bin/x86_64-linux/bin/SwinIRRestore                      # pixelshuffle x2 fixture
../../bin/x86_64-linux/bin/SwinIRRestore --fixture x4         # pixelshuffle x4 fixture
../../bin/x86_64-linux/bin/SwinIRRestore --fixture direct     # pixelshuffledirect x3 fixture
../../bin/x86_64-linux/bin/SwinIRRestore --fixture nearest    # nearest+conv x4 fixture
../../bin/x86_64-linux/bin/SwinIRRestore --fixture denoise    # upscale-1 denoise fixture
../../bin/x86_64-linux/bin/SwinIRRestore model.safetensors [config.json]
```

## Input

The official SwinIR checkpoints are large and not obtainable offline, so — like
the repo's RRDBNet / NAFNet pico fixtures — this falls back to the committed
config-faithful **random** pico SwinIR (`tests/fixtures/tiny_swinir.safetensors`
+ `..._config.json`; `tiny_swinir_x4.*`, `tiny_swinir_direct.*`,
`tiny_swinir_nearest.*` and `tiny_swinir_denoise.*` for the other tails; each
parity-checked < 1e-4 against a float64 numpy oracle in
`TestNeuralPretrained.pas`). The pico net is random (not trained), so this is a
**wiring/throughput smoke**: it runs the SR forward on a deterministic synthetic
striped image and reports the restored shape. Pass a real `.safetensors`
(+ `config.json` sibling) to upscale with your own trained checkpoint.

## Output

Writes `swinir_input.ppm` and `swinir_upscaled.ppm` (P6 color images; the
restored image) and prints the config plus a small ASCII intensity preview of
input / restored (the restored side is `Upscale×` larger). Pure CPU.
