# FaceRestoration: GFPGAN Blind Face Restoration

Restores a degraded face photo (blur, noise, compression, low resolution) with
the repo's **GFPGAN** importer (`BuildGFPGANFromSafeTensors`,
`GFPGANRestoreFace` in `neuralpretrained.pas`). GFPGAN (Wang et al., CVPR 2021,
[*Towards Real-World Blind Face Restoration with Generative Facial
Prior*](https://arxiv.org/abs/2101.04061), `TencentARC/GFPGAN`) uses a
pretrained StyleGAN2 face generator as a prior: a U-Net cleans the input face
and steers the generator through its style latents and spatial feature
transforms.

**Status: coded, not yet tested on a real checkpoint.** The importer is
parity-tested (< 1e-4) against the official `gfpganv1_clean_arch.py` on two
random pico nets (`TestGFPGANParity`, `TestGFPGANSharedLatentParityInfersConfig`),
and the OpenCL forward matches the CPU forward on the same net
(`TestGFPGANOpenCLParity`). No official checkpoint has been run yet.

## The input must be an aligned face

This program does **not** detect or align faces. The input must already be a
square crop of one face, aligned the way GFPGAN was trained (the FFHQ
alignment: eyes level, face centered, filling the frame). This is what the
official GFPGAN tool calls `--aligned` input. A face of another size is resized
bilinearly to the net resolution (512x512 for the official checkpoints). A
whole photo with a small face in it, or an unaligned crop, gives poor results.

## How it works

1. `LoadImageFromFileIntoVolume` reads the face.
2. `GFPGANPrepareInput` resizes it to the net resolution and maps 0..255 to
   [-1, 1].
3. The net: a U-Net encoder down to 4x4, a linear layer to the StyleGAN2 style
   latents, a U-Net decoder that produces a scale and a shift per resolution,
   and the StyleGAN2 decoder, whose features are multiplied by the scale and
   shifted after the first convolution of every resolution (CS-SFT, on half of
   the channels).
4. `GFPGANComposeOutput` clamps the output to [-1, 1] and maps it back to
   0..255.

## Run

```
./FaceRestoration                                           # pico fixture smoke run
./FaceRestoration --model GFPGANv1.4.pth --input face.png --output face_restored.png
./FaceRestoration --model GFPGANv1.4.pth --input face.png --output face_restored.png --gpu --noise const
```

| option | meaning | default |
| --- | --- | --- |
| `--model FILE` | GFPGAN checkpoint: the official `GFPGANv1.4.pth` or `GFPGANv1.3.pth` (the `{'params_ema': ...}` release format, read directly), or a `.safetensors` with the same tensor names | smoke run |
| `--config FILE` | optional `config.json` (`out_size`, `num_style_feat`, `channel_multiplier`, `narrow`, `different_w`, `sft_half`, ...) | `config.json` next to `--model`; without one, the v1.4 settings, with every size read from the tensor shapes |
| `--input IMG` | aligned, cropped face, PNG / JPEG / BMP | required with `--model` |
| `--output IMG` | restored face; the format comes from the extension (PNG recommended) | required with `--model` |
| `--keep-size` | resize the restored face back to the input size | output at the net resolution |
| `--noise MODE` | per-pixel StyleGAN2 noise: `random` (fresh noise each run, what the official tool does), `const` (the noise maps stored in the checkpoint: the same output every run) or `none` | `random` |
| `--seed N` | seed of the random noise, for a repeatable `random` run | time based |
| `--int8` | store the convolution and linear weights in int8 (`TNNet.QuantizeWeightsInt8`): less RAM, small accuracy loss | FP32 |
| `--gpu`, `--gpu-platform N`, `--gpu-device N` | run the net on OpenCL (a build with `-dOpenCL`, the `.lpi` default) | CPU |

Checkpoints: `GFPGANv1.3.pth` and `GFPGANv1.4.pth` are published on the
TencentARC/GFPGAN GitHub releases page. Both use the "clean" architecture this
importer implements. The older `GFPGANv1.pth` (the original architecture with
compiled CUDA operators) is a different network and is not supported.

With no arguments, the program builds the committed random pico net
(`tests/fixtures/tiny_gfpgan.*`, a 64x64 net), writes a 40x30 color gradient to
`gfpgan_in.png`, restores it and writes `gfpgan_out.png`. The weights are
random, so the output is meaningless; the smoke run only exercises loading, the
forward pass and PNG I/O. It must be run from this folder (the fixture path is
relative).

## Build

Open `FaceRestoration.lpi` in Lazarus, or:

```
lazbuild FaceRestoration.lpi
```

The binary lands in `../../bin/<cpu>-<os>/bin/`. For `--gpu`, the library finds
`neural/neural.cl` relative to this folder or to the binary.
