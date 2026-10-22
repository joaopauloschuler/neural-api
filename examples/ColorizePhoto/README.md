# ColorizePhoto: DDColor Photo Colorization

Colorizes a black-and-white (or color) photo with the repo's **DDColor**
importer (`BuildDDColorFromSafeTensors`, `DDColorColorize` in
`neuralpretrained.pas`). DDColor (Kang et al., ICCV 2023,
[*DDColor: Towards Photo-Realistic Image Colorization via Dual
Decoders*](https://arxiv.org/abs/2212.11613), `piddnad/DDColor`) predicts the
a\*b\* chroma of a photo from its lightness L.

**Status: coded, not yet tested on a real checkpoint.** The importer is
parity-tested (< 1e-4) against the official `basicsr/archs/ddcolor_arch.py` on
a random pico net (`TestDDColorParity`, `TestDDColorPthParityInfersConfig`),
and the OpenCL forward matches the CPU forward on the same net
(`TestDDColorOpenCLParity`). No official checkpoint has been run yet.

## How it works

1. `LoadImageFromFileIntoVolume` reads the photo (any size; gray images load as
   RGB with R = G = B).
2. `DDColorPrepareInput` resizes it to the net input (`--size`), keeps only L
   (a\*b\* set to 0) and converts back to RGB in [0, 1].
3. The net (ConvNeXt encoder, pixel decoder, multi-scale color decoder with
   learned color queries) outputs a\*b\* at the net input size.
4. `DDColorComposeOutput` resizes a\*b\* to the photo's own size and recombines
   it with the photo's full-resolution L, so no detail is lost to the net's
   input resolution.

A color photo is treated the same way: its own colors are discarded and
re-predicted.

## Run

```
./ColorizePhoto                                    # pico fixture smoke run
./ColorizePhoto --model pytorch_model.bin --input old.jpg --output old_color.png
./ColorizePhoto --model pytorch_model.bin --input old.jpg --output old_color.png --size 256 --gpu
```

| option | meaning | default |
| --- | --- | --- |
| `--model FILE` | DDColor checkpoint: `.bin` / `.pth` (a plain state_dict or the `{'params': state_dict}` release format) or `.safetensors` | smoke run |
| `--config FILE` | DDColor `config.json` (`encoder_name`, `input_size`, `num_queries`, `dec_layers`, ...) | `config.json` next to `--model`; without one, the widths come from the tensor shapes and the input is 512x512 |
| `--input IMG` | photo to colorize, PNG / JPEG / BMP, color or gray | required with `--model` |
| `--output IMG` | colorized photo; the format comes from the extension (PNG recommended) | required with `--model` |
| `--size N` | net input resolution, square, a multiple of 32. 512 is the official setting; 256 is about 4x less compute | `input_size` from the config, else 512 |
| `--int8` | store the convolution and linear weights in int8 (`TNNet.QuantizeWeightsInt8`): less RAM, small accuracy loss | FP32 |
| `--gpu`, `--gpu-platform N`, `--gpu-device N` | run the net on OpenCL (a build with `-dOpenCL`, the `.lpi` default) | CPU |

Checkpoints: the Hugging Face repositories published by the DDColor authors,
for example `piddnad/ddcolor_paper_tiny` (ConvNeXt-T encoder), hold a
`pytorch_model.bin` and a `config.json`; put both in one folder and pass the
`.bin` to `--model`. The large (ConvNeXt-L) checkpoints load the same way. The
official `.pth` releases need no conversion either: `CreatePretrainedTensorReader`
reads the pickle and unwraps `params`.

With no arguments, the program builds the committed random pico net
(`tests/fixtures/tiny_ddcolor.*`), writes a 40x30 color gradient to
`ddcolor_in.png`, colorizes it and writes `ddcolor_out.png`. The weights are
random, so the colors are meaningless; the smoke run only exercises loading,
the forward pass and PNG I/O. It must be run from this folder (the fixture path
is relative).

## Build

Open `ColorizePhoto.lpi` in Lazarus, or:

```
lazbuild ColorizePhoto.lpi
```

The binary lands in `../../bin/<cpu>-<os>/bin/`. For `--gpu`, the library finds
`neural/neural.cl` relative to this folder or to the binary.
