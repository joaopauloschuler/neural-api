# RealESRGANUpscale: Real-ESRGAN / ESRGAN RRDBNet super-resolution

Upscales an image with the **Real-ESRGAN / ESRGAN RRDBNet** importer in
`neural/neuralpretrained.pas` (`LoadRRDBNetConfig`, `RRDBNetTileConfig`,
`BuildRRDBNetFromSafeTensorsEx`, `RRDBNetUpscale`). RRDBNet is convolution only,
with no diffusion loop.

## Running

```
cd examples/RealESRGANUpscale
lazbuild -B RealESRGANUpscale.lpi
../../bin/x86_64-linux/bin/RealESRGANUpscale                 # pico fixture smoke
../../bin/x86_64-linux/bin/RealESRGANUpscale --model RealESRGAN_x4plus.pth \
  --input photo.png --output photo_x4.png --tile 256
```

| Option | Meaning |
| --- | --- |
| `--model FILE` | RRDBNet checkpoint: `.pth` (the `params_ema` wrapper is unwrapped) or `.safetensors` (a `params_ema.` key prefix is accepted) |
| `--config FILE` | `config.json`. Default: `config.json` next to `--model`; if there is none, `InferRRDBNetConfigFromTensors` reads the config from the tensor shapes |
| `--input IMG` | image to upscale (PNG / JPEG / BMP) |
| `--output IMG` | upscaled PNG |
| `--tile N` | tile edge in input pixels; `0` (default) runs the whole image at once |
| `--tile-pad N` | context pixels added on each tile side and dropped from the output (basicsr `tile_pad`, default 10) |
| `--tile-overlap N` | pixels feather-blended across tile seams (default 8) |
| `--gpu` | run the net on OpenCL (the `.lpi` builds with `-dOpenCL`) |
| `--gpu-platform N`, `--gpu-device N` | OpenCL platform / device index (default 0) |

Pixels go into the net as RGB in `[0,1]`, and the output is clamped to `[0,1]`,
as in basicsr's `RealESRGANer`.

## Supported checkpoint layouts

- `RealESRGAN_x4plus` (and ESRGAN x4): `conv_first` takes 3 channels, and both
  `conv_up1` and `conv_up2` are present.
- `RealESRGAN_x2plus` / x1: `conv_first` takes 3·2·2 or 3·4·4 channels. The
  importer adds a `TNNetSpaceToDepth` stem for the pixel unshuffle, keeps both
  upsample stages, and reflect-pads the image to a multiple of the unshuffle
  factor, as basicsr does.
- The repo's one-stage x2 pico layout (`conv_up2` absent).

The importer reads the xinntao / basicsr key names (`conv_first`,
`body.N.rdb{1,2,3}.conv{1..5}`, `conv_body`, `conv_up1`, `conv_up2`, `conv_hr`,
`conv_last`). It does not read the old ESRGAN `model.N.*` layout.

## Tiling

`RRDBNetTileConfig` sizes the net input to `tile + 2*tile_pad`, rounded to the
unshuffle factor, or to the whole image when the image fits. `RRDBNetUpscale`
then calls `TiledNetCompute`, which `TiledVaeDecode` also uses. For each tile,
`TiledNetCompute` drops `tile_pad` output pixels on every interior side and
feather-blends the next `tile_overlap` pixels into the neighbouring tile.
Peak memory follows the tile size, not the image size. The real x4plus net
looks far beyond 10 pixels, so tiled output is close to the whole-image
output but not equal to it, as in basicsr.

## Smoke mode (no arguments)

With no arguments, the example builds the committed random-weight pico nets
`tests/fixtures/tiny_rrdbnet` (x4) and `tiny_rrdbnet_x2` (x2). It writes a
6×6 gradient PNG, reads it back, upscales it, and writes `esrgan_out_x4.png`
and `esrgan_out_x2.png`. This checks the import and the image I/O. It is not a
photoreal upscale.

Tests: `TestRRDBNetParity{,Pth,Scale2}` (float64 oracle, `< 1e-4`),
`TestRRDBNetInferConfigFromTensors`, `TestRRDBNetKeyPrefix`,
`TestRRDBNetTiledUpscale` (tiled vs whole on a non-square image, with square
and non-square tiles) and `TestRRDBNetPixelUnshuffleX2` (an x2plus layout
written from Pascal, with a direct PixelUnshuffle + conv oracle for the stem).

Coded by Claude (AI).
