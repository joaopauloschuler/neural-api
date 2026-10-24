# ImageMatting: ViTMatte Trimap-Based Image Matting

Estimates a fine alpha matte (hair, fur, glass edges) for a photo with the
repo's **ViTMatte** importer (`BuildViTMatteFromSafeTensors`,
`ViTMatteEstimateAlpha` in `neuralpretrained.pas`). ViTMatte (Yao et al. 2023,
[*ViTMatte: Boosting Image Matting with Pretrained Plain Vision
Transformers*](https://arxiv.org/abs/2305.15272); transformers
`VitMatteForImageMatting`) reads an RGB image plus a **trimap** and predicts
an alpha value in 0..1 for every pixel. A segmentation model such as BiRefNet
(see [BackgroundRemoval](../BackgroundRemoval)) gives a hard foreground mask;
ViTMatte turns a rough mask into a soft matte along the object's edge.

**Status: coded, not yet tested on a real checkpoint.** The importer is
parity-tested (< 1e-4) against transformers on a random pico net
(`TestViTMatteParity`), and the OpenCL forward matches the CPU forward on the
same net (`TestViTMatteOpenCLParity`). No official checkpoint has been run yet.

## The trimap

The trimap is a gray image of the same size as the photo:

| value | meaning |
| --- | --- |
| 0 | background: alpha is expected to be 0 |
| 128 | unknown: ViTMatte decides the alpha here |
| 255 | foreground: alpha is expected to be 1 |

`--trimap FILE` reads a trimap you drew (only the first channel is used).
`--auto-trimap FILE` builds one instead: BiRefNet / RMBG-2.0
(`BiRefNetRemoveBackground`) predicts a foreground mask, and `TrimapFromMask`
thresholds it at 128, then box-filters it with `TNNetBoxFilter` of radius
`--unknown-band N`. A pixel whose whole (2N+1)x(2N+1) window is foreground
becomes 255, a pixel with any foreground in the window becomes 128, the rest
become 0. The window counts only pixels inside the image, so a subject that
touches the image border is not eroded there.

## How it works

1. `LoadImageFromFileIntoVolume` reads the image (any size); the trimap comes
   from `--trimap` or `--auto-trimap`.
2. `BuildViTMatteFromSafeTensors(..., ImageWidth, ImageHeight)` builds the net
   for the image size padded to a multiple of 32 (`ViTMattePaddedSize`), so the
   program builds the net after it reads the image.
3. `ViTMattePreprocess` scales the RGB to -1..1, appends trimap/255 as a fourth
   channel and pads the right and bottom with zeros.
4. A ViTDet backbone (windowed and global attention blocks with relative
   position tables, plus residual convolution blocks) and a detail-capture
   decoder (ConvStream stages on the input, fusion blocks, a sigmoid head)
   return the alpha; `ViTMatteEstimateAlpha` crops the padding away.
5. The program writes the alpha as a gray PNG and, with `--cutout`, the photo
   with the alpha as its alpha channel (`BiRefNetComposeRGBA`).

## Run

```
./ImageMatting                                                   # pico fixture smoke run
./ImageMatting --model vitmatte-small-composition-1k --image photo.jpg --trimap trimap.png --output alpha.png --cutout cutout.png
./ImageMatting --model vitmatte-small-composition-1k --image photo.jpg --auto-trimap BiRefNet/model.safetensors --birefnet-size 512 --unknown-band 10 --save-trimap trimap.png --cutout cutout.png --gpu
```

| option | meaning | default |
| --- | --- | --- |
| `--model DIR_OR_FILE` | ViTMatte `model.safetensors`, or the folder holding it and `config.json` | smoke run |
| `--config FILE` | ViTMatte `config.json` | next to the model |
| `--image IMG` | image, PNG / JPEG / BMP | required with `--model` |
| `--trimap IMG` | trimap of the image size (0 / 128 / 255) | one of `--trimap`, `--auto-trimap` |
| `--auto-trimap FILE` | BiRefNet / RMBG-2.0 `model.safetensors` that makes the trimap | |
| `--auto-trimap-config FILE` | BiRefNet `config.json` | next to the BiRefNet model if present |
| `--birefnet-size N` | BiRefNet square input, a multiple of 32 | its config (1024) |
| `--unknown-band N` | half-width of the unknown band in pixels | 10 |
| `--output IMG` | the alpha matte, gray PNG | |
| `--cutout IMG` | the image with the alpha matte as alpha, RGBA PNG | |
| `--save-trimap IMG` | the trimap used, gray PNG | |
| `--int8` | store the convolution and linear weights in int8 (`TNNet.QuantizeWeightsInt8`) | FP32 |
| `--gpu`, `--gpu-platform N`, `--gpu-device N` | run the nets on OpenCL (a build with `-dOpenCL`, the `.lpi` default) | CPU |

At least one of `--output` and `--cutout` is needed with `--model`. With
`--auto-trimap`, `--int8` and `--gpu` apply to BiRefNet too; the program frees
BiRefNet before it builds ViTMatte.

## Checkpoints

Download the folder (`config.json` + `model.safetensors`) of
`hustvl/vitmatte-small-composition-1k` (ViT-S backbone) or
`hustvl/vitmatte-base-composition-1k` (ViT-B) from Hugging Face and pass it to
`--model`; no conversion is needed. For the license, see the model card. The
global attention blocks attend over every 16x16 patch of the padded image, so
their memory grows with the square of the patch count (for a 1024x1024 image,
4096 patches: by arithmetic, not measured, about 64 MB of FP32 scores per head).
Start with a small image.

## Pico smoke run

With no arguments the program builds a 45x30 gradient image and an ellipse
mask, writes `vitmatte_in.png` and `vitmatte_trimap.png` (`TrimapFromMask`,
band 3), and mattes the image with the random pico fixture
`tests/fixtures/tiny_vitmatte` (64x32 net). It then mattes it again with
`--auto-trimap` from the pico BiRefNet fixture (`tiny_birefnet`). The weights
are random, so the PNGs show only that the pipeline runs end to end.

The binary lands in `../../bin/<cpu>-<os>/bin/`. For `--gpu`, the library finds
`neural/neural.cl` relative to this folder or to the binary.
