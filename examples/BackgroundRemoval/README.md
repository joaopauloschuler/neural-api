# BackgroundRemoval: BiRefNet / RMBG-2.0 Background Removal

Cuts the subject out of a photo with the repo's **BiRefNet** importer
(`BuildBiRefNetFromSafeTensors`, `BiRefNetRemoveBackground` in
`neuralpretrained.pas`). BiRefNet (Zheng et al., CAAI AIR 2024,
[*Bilateral Reference for High-Resolution Dichotomous Image
Segmentation*](https://arxiv.org/abs/2401.03407), `ZhengPeng7/BiRefNet`)
predicts one high-resolution foreground probability per pixel. BRIA's
`briaai/RMBG-2.0` is published as the same architecture with its own weights,
so both should load through the same importer (its repository is gated, and its
code was not inspected here; `ZhengPeng7/BiRefNet_lite` differs from BiRefNet
only by its Swin-T backbone).

**Status: coded, not yet tested on a real checkpoint.** The importer is
parity-tested (< 1e-4) against the official `birefnet.py` on a random pico net
(`TestBiRefNetParity`), and the OpenCL forward matches the CPU forward on the
same net (`TestBiRefNetOpenCLParity`). No official checkpoint has been run yet.

## How it works

1. `LoadImageFromFileIntoVolume` reads the image (any size).
2. `BiRefNetPrepareInput` resizes it to the net input (`--size`, default
   1024x1024, bilinear as PIL does) and normalizes it with the ImageNet mean and
   standard deviation.
3. The net runs a Swin Transformer backbone on the image and on a half-size
   copy (both through one set of layers), fuses the four stage features,
   decodes them with deformable-convolution blocks that also look at tiles of
   the input image, and outputs the foreground probability.
4. `BiRefNetComposeMask` resizes the probability back to the image's own size
   (bilinear) as an alpha mask in 0..255; `BiRefNetComposeRGBA` attaches it to
   the original RGB, so the output keeps every detail of the input.

## Run

```
./BackgroundRemoval                                          # pico fixture smoke run
./BackgroundRemoval --model model.safetensors --input photo.jpg --output cutout.png
./BackgroundRemoval --model model.safetensors --input photo.jpg --output cutout.png --mask-output mask.png --size 512 --gpu
```

| option | meaning | default |
| --- | --- | --- |
| `--model FILE` | `model.safetensors` from `ZhengPeng7/BiRefNet`, `ZhengPeng7/BiRefNet_lite` (Swin-T) or `briaai/RMBG-2.0`; `.pth` / `.bin` state dicts load too | smoke run |
| `--config FILE` | a BiRefNet `config.json` (`size`, ...). The Hugging Face `config.json` carries no shapes; every width, depth and window comes from the tensors | `config.json` next to `--model` if present |
| `--input IMG` | image, PNG / JPEG / BMP | required with `--model` |
| `--output IMG` | the image with the mask as alpha, PNG (RGBA) | |
| `--mask-output IMG` | the mask alone, gray PNG | |
| `--size N` | square net input, a multiple of 32. 1024 is the official setting; 512 is about 4x less compute and memory | 1024 |
| `--int8` | store the convolution and linear weights in int8 (`TNNet.QuantizeWeightsInt8`): less RAM, small accuracy loss | FP32 |
| `--gpu`, `--gpu-platform N`, `--gpu-device N` | run the net on OpenCL (a build with `-dOpenCL`, the `.lpi` default) | CPU |

At least one of `--output` and `--mask-output` is needed with `--model`.

Checkpoints: download `model.safetensors` from one of the repositories above
and pass it to `--model`; no conversion is needed. `briaai/RMBG-2.0` is gated
and released for non-commercial use only; read its license first. The
default BiRefNet is a Swin-L backbone (about 220M parameters, ~0.9 GB of FP32
weights). At 1024x1024 the forward keeps many large activations: by
arithmetic (not measured) the Swin stages alone hold on the order of 10 GB of
FP32 activations, so start with `--size 512` (about 4x less) on a small
machine, or use the Swin-T `BiRefNet_lite` checkpoint.

With no arguments, the program builds the committed random pico net
(`tests/fixtures/tiny_birefnet.*`, a 96x64 net input), writes a 50x40 color
gradient to `birefnet_in.png`, and writes `birefnet_out.png` (RGBA) and
`birefnet_mask.png`. The weights are random, so the mask is meaningless; the
smoke run only exercises loading, the forward pass and PNG I/O. It must be run
from this folder (the fixture path is relative).

## Build

Open `BackgroundRemoval.lpi` in Lazarus, or:

```
lazbuild BackgroundRemoval.lpi
```

The binary lands in `../../bin/<cpu>-<os>/bin/`. For `--gpu`, the library finds
`neural/neural.cl` relative to this folder or to the binary.
