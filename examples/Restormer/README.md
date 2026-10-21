# Restormer Image Restoration

Restores rainy, blurred and noisy photos with the repo's **Restormer** importer
(`BuildRestormerFromSafeTensorsEx`, `neuralpretrained.pas`). Restormer (Zamir
et al. 2022, [*Restormer: Efficient Transformer for High-Resolution Image
Restoration*](https://arxiv.org/abs/2111.09881), `swz30/Restormer`) restores an
image at its own resolution. One architecture covers every task; only the
weights and two options (LayerNorm type, input channels) change.

**Status: coded, real-checkpoint run pending on the user's machine.** The
importer is parity-tested against a float64 torch reimplementation of the
official `basicsr/models/archs/restormer_arch.py` on two random pico nets. No
official checkpoint has been run yet.

## Architecture

```
patch_embed 3x3 -> [encoder level L blocks -> Downsample]x3 -> latent blocks
  -> [Upsample -> concat encoder skip -> (1x1 reduce, levels 3 and 2)
      -> decoder level L blocks]x3 -> refinement blocks -> output 3x3 + input
```

TransformerBlock(C):

```
x = x + project_out(MDTA(dwconv3x3(qkv1x1(LayerNorm(x)))))
x = x + project_out(gelu(x1) * x2),  [x1|x2] = dwconv3x3(project_in1x1(LayerNorm(x)))
```

| piece | CAI layer |
| --- | --- |
| LayerNorm over channels, `WithBias` | `TNNetTokenLayerNorm` |
| LayerNorm over channels, `BiasFree` (`x / sqrt(var + 1e-5) * w`) | `TNNetTokenLayerNorm(eps, true, pCenterOutput=false)` |
| MDTA: per-head C x C attention over channels | `TNNetTransposedChannelAttention(heads)` |
| 1x1 convs, 3x3 depthwise convs | `TNNetPointwiseConvLinear`, `TNNetDepthwiseConvLinear` |
| GDFN gate `gelu(x1) * x2` (exact erf) | `TNNetGEGLUErf` (halves swapped at load) |
| Downsample: conv3x3 C -> C/2 + PixelUnshuffle(2) | `TNNetSpaceToDepth(2)` + `TNNetGatherChannels` (torch channel order) |
| Upsample: conv3x3 C -> 2C + PixelShuffle(2) | `TNNetPixelShuffle(2)` (conv neurons permuted at load) |
| skip concat, residuals | `TNNetDeepConcat`, `TNNetSum` |

## Run

```
./Restormer                                          # pico fixture smoke run
./Restormer --task derain --input rainy.png --output clean.png
./Restormer --task deblur --model motion_deblurring.pth --input blurry.jpg --output sharp.png --gpu
```

| option | meaning | default |
| --- | --- | --- |
| `--task TASK` | `derain`, `deblur` (motion), `defocus` (single image), `denoise` (real camera noise), `denoise-gaussian`, `denoise-gray` | smoke run |
| `--model FILE` | checkpoint, `.pth` (the release format) or `.safetensors` | the release file name below, in the current folder |
| `--config FILE` | a JSON with the torch constructor kwargs, overriding the task preset | the preset |
| `--input IMG` / `--output IMG` | PNG / JPEG / BMP / PPM in; the output format comes from the extension | required with `--task` |
| `--tile N` | tile edge in pixels; `0` runs the whole image in one net | 256 |
| `--tile-pad N` | context pixels on each tile side, dropped from interior sides | 16 |
| `--tile-overlap N` | pixels feather-blended across tile seams | 32 |
| `--gpu`, `--gpu-platform N`, `--gpu-device N` | run the net on OpenCL (a build with `-dOpenCL`, the `.lpi` default) | CPU |

| `--task` | official option set | release file | LayerNorm |
| --- | --- | --- | --- |
| `derain` | Deraining | `deraining.pth` | WithBias |
| `deblur` | Motion_Deblurring | `motion_deblurring.pth` | WithBias |
| `defocus` | Defocus_Deblurring (single image) | `single_image_defocus_deblurring.pth` | WithBias |
| `denoise` | Real_Denoising | `real_denoising.pth` | BiasFree |
| `denoise-gaussian` | Gaussian color denoising (blind) | `gaussian_color_denoising_blind.pth` | BiasFree |
| `denoise-gray` | Gaussian gray denoising (blind), 1 channel | `gaussian_gray_denoising_blind.pth` | BiasFree |

The release files come from the links in the official repository's
`*/pretrained_models/README.md`. They are `torch.save({'params': state_dict})`
pickles; `CreatePretrainedTensorReader` reads them directly and unwraps
`params`, so no conversion step is needed. The non-blind Gaussian checkpoints
(`gaussian_color_denoising_sigma15.pth` and so on) load with
`--task denoise-gaussian --model FILE`.

Pixels enter the net as [0, 1] (`/255`) and leave it clamped to [0, 1], as the
official test scripts do. `denoise-gray` converts the input to luma
(0.299 R + 0.587 G + 0.114 B) and writes a gray image. The example builds the
net once at the tile size (`TiledNetInputSide`), reflect-pads the image on the
right and bottom to a multiple of 8 (the three 2x downsamples need it; the
official `check_image_size` pads the same way) and runs
`PaddedTiledNetCompute`, which uses the same `TiledNetCompute` tiling as the
Real-ESRGAN example. Tile size, pad and overlap are rounded up to multiples of
8. Restormer's channel attention sees the whole input, so a tiled result
differs slightly from a whole-image one; a larger `--tile` is closer to the
official full-image output and costs more memory. At `dim` 48 the widest
full-resolution activation is 510 channels (the level-1 GDFN), so a 256x256
tile holds about 134 MB per such tensor. `ShareHostOutputsByLiveness` lets
activations that are no longer needed share storage.

With no arguments, the example runs a smoke test on the committed random pico
fixture (`tests/fixtures/tiny_restormer.*`). A synthetic 37x29 image is
reflect-padded and run in 16x16 tiles, and the result is written to
`restormer_smoke.ppm`. The weights are random, so the output is not a
restoration.

## Not covered

* Dual-pixel defocus deblurring (`dual_pixel_defocus_deblurring.pth`, a
  6-channel left/right pair): the importer builds it (`TRestormerConfig.DualPixelTask`,
  parity-tested), but this example reads one image only.
* No real checkpoint has been run, and no GPU timing exists yet.

## Tests

* `TestRestormerParity`: WithBias LayerNorm, no conv biases, 24x16 RGB,
  `.safetensors`; max |diff| 8.4e-7 against the float64 torch output.
* `TestRestormerDualPixelBiasFreeParity`: BiasFree LayerNorm, conv biases,
  dual-pixel skip conv, 16x24, a `{'params': ...}` `.pth`; max |diff| 1.6e-6,
  and a save/load copy of the net reproduces the output exactly.
* `TestRestormerTiledOpenCL`: the dual-pixel pico net on a 40x30 image through
  `PaddedTiledNetCompute`, OpenCL against CPU, max |diff| 1.9e-6 on PoCL.
