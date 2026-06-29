# RealESRGANUpscale: Real-ESRGAN / ESRGAN RRDBNet super-resolution

An end-to-end smoke of the **Real-ESRGAN / ESRGAN RRDBNet** super-resolution
importer (`BuildRRDBNetFromSafeTensors`, `neural/neuralpretrained.pas`) — the
convolutional, **no-diffusion** image upscaler in the repo. Pure CPU, no dataset
download, well under a minute.

## What it does

1. Build the pico RRDBNet from the committed parity fixture
   `tests/fixtures/tiny_rrdbnet.{safetensors,_config.json}` (scale ×4) and the ×2
   sibling `tiny_rrdbnet_x2.*` (one upsample stage).
2. Synthesize a tiny 6×6 RGB color-gradient image and save it as a PNG with
   `neuraldatasets.SaveImageFromVolumeIntoFile` (exercising the repo's PNG writer).
3. Load that PNG back with `LoadImageFromFileIntoVolume`, normalise `0..255 →
   [-1,1]`, `Compute`, map the output back to `0..255` and save the upscaled PNG.
4. Report the input/output dimensions for both scales.

Outputs: `esrgan_in_x4.png` / `esrgan_out_x4.png` and `esrgan_in_x2.png` /
`esrgan_out_x2.png`.

## Running

```
cd examples/RealESRGANUpscale
lazbuild RealESRGANUpscale.lpi --build-mode=Release   # or: fpc -Fu../../neural RealESRGANUpscale.lpr
./RealESRGANUpscale
```

The fixture paths are relative to the example directory (`../../tests/fixtures/`).

## Notes

- The committed checkpoint is a **random-weight** pico net — a faithful smoke of
  the import + forward + image-I/O path, **not** a photoreal upscaler.
- Swap in a real `RealESRGAN_x4plus` `.pth` / `.safetensors` + its `config.json`
  to upscale real images via the same `BuildRRDBNetFromSafeTensors` call (a `.pth`
  `params_ema` wrapper is unwrapped automatically by `TNNetTorchBinReader`).
- Inference only, pure CPU.

Coded by Claude (AI).
