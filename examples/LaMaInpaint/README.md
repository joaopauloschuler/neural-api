# LaMaInpaint: big-lama large-mask inpainting

Removes the masked part of a photo in one forward pass. There is no prompt and no
denoising loop. The generator is big-lama (Suvorov et al. 2021,
[*Resolution-robust Large Mask Inpainting with Fourier Convolutions*](https://arxiv.org/abs/2109.07161)),
imported by `BuildLaMa` / `BuildLaMaFromSafeTensors` in `neural/neuralpretrained.pas`.

**Status: coded, real checkpoint untested.** `TestLaMaParity` checks the importer
to < 1e-4 against a PyTorch re-implementation of the generator on a pico
random-weight checkpoint with big-lama's key names
(`tests/fixtures/tiny_lama.safetensors`). The real `big-lama.safetensors` has
not been run yet.

## Getting the checkpoint

Download `big-lama.safetensors` (about 0.5 GB of F32 weights, 124M parameters
in the generator) from the Hugging Face repository
[anyisalin/big-lama](https://huggingface.co/anyisalin/big-lama). The importer
reads the `generator.model.*` tensors. It ignores the `val_evaluator.*` /
`test_evaluator.*` tensors and `num_batches_tracked`.

## Usage

```
cd examples/LaMaInpaint
lazbuild -B LaMaInpaint.lpi
../../bin/x86_64-linux/bin/LaMaInpaint --model big-lama.safetensors \
  --image photo.png --mask mask.png --out result.png
```

- `--image`: any format `LoadImageFromFileIntoVolume` reads (PNG, JPEG, BMP, ...).
- `--mask`: same size as the image. Like lama's `predict.py`, any non-zero pixel
  is a hole, so draw the region to remove in white on a black background.
- `--out`: written as PNG.

The program follows lama's prediction pipeline:
1. It scales the image to [0, 1].
2. It pads the image and the mask on the bottom and right edges by symmetric
   mirroring (`numpy.pad(mode='symmetric')`) to a multiple of 8.
3. It feeds `image*(1-mask) | mask` (4 channels) to the generator.
4. It composites `mask*prediction + (1-mask)*image`, crops the padding away and
   saves the result.

`BuildLaMa` builds the net for one fixed input size. The program builds it for
the padded image size, so each run builds a new net.

## Memory

The weights take about 0.5 GB. At 512x512 the forward pass needs much more.
Every convolution keeps its im2col column matrix after the forward pass, and
every layer keeps its own output. At 512x512 the estimate is about 8.1 GB on
top of the weights, about 8.6 GB in total:

| Buffer | Size |
| --- | --- |
| Column matrix of the 7x7 64->3 output conv | 3.3 GB |
| Column matrices of the 3x3 convs in the 36 FFC units | 3.4 GB |
| Column matrices of the 7x7 input conv and the stride-2 convs | 0.5 GB |
| Layer outputs | 0.9 GB |

These numbers come from the pico net built at 512x512 (836 MB of non-weight
buffers after one forward pass) and from scaling each layer to big-lama's
channel counts. They were not measured on the real checkpoint. Memory grows
with the pixel count, so a 256x256 image needs about a quarter of the
activation memory.

## Notes

- CPU only. The 3x3 and 1x1 convolutions have OpenCL paths, but the
  `TNNetRealFFT2D` / `TNNetInverseRealFFT2D` pair runs on the host, and this
  example does not enable OpenCL.
- Smoke run without the real checkpoint: pass the pico fixture with
  `--model ../../tests/fixtures/tiny_lama.safetensors`. Its weights are random,
  so the filled region is noise. The run still exercises the full pipeline.
  Images smaller than 16x16 are padded up to 16x16.
