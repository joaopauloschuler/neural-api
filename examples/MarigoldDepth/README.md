# MarigoldDepth

Monocular depth or surface normals with Marigold (Ke et al. 2024,
[arXiv:2312.02145](https://arxiv.org/abs/2312.02145)), through
`TMarigoldPipeline` in `neural/neuralpretrained.pas`.

Marigold is a Stable Diffusion 2 UNet fine-tuned to denoise a depth (or
normals) latent while conditioned on the image latent. The UNet's `conv_in`
takes 8 channels: the image latent and the noisy target latent. One
`TMarigoldPipeline.Predict` call:

1. resizes the image so its long side is the processing resolution, then pads
   it to a square (replicating the edge pixels),
2. encodes it with the VAE encoder,
3. runs the trailing DDIM steps (v-prediction, zero-terminal-SNR schedule from
   `scheduler/scheduler_config.json`) for each ensemble member, conditioned on
   the empty-prompt CLIP states from `text_encoder/`,
4. decodes each member with the VAE decoder, ensembles the members (depth:
   scale/shift-aligned median; normals: the member closest to the mean), and
   resizes the result back to the input size.

`Predict` builds the VAE encoder, the UNet and the VAE decoder one at a time and
frees each one after its stage. Peak memory is therefore the UNet, not the sum
of all components.

## Usage

```
MarigoldDepth                                   pico fixture smoke
MarigoldDepth --model DIR --input IMG --output PNG [options]
  --model DIR          diffusers Marigold folder (model_index.json, unet/,
                       vae/, text_encoder/, scheduler/)
  --input IMG          image (PNG / JPEG / BMP)
  --output PNG         depth: 16-bit grayscale PNG; normals: RGB PNG
  --steps N            denoising steps (default: default_denoising_steps)
  --ensemble N         predictions from N noises, ensembled (default 1)
  --processing-res N   long side the model sees; 0 = native
                       (default: default_processing_resolution)
  --seed N             noise seed (default 0)
  --int8               int8 UNet weights
  --gpu                run the nets on OpenCL (build with -dOpenCL)
  --gpu-platform N     OpenCL platform index (default 0)
  --gpu-device N       OpenCL device index (default 0)
```

The task comes from `model_index.json` (`_class_name`
`MarigoldDepthPipeline` or `MarigoldNormalsPipeline`). Both v1-1 checkpoints
default to 4 steps at a processing resolution of 768.

```
MarigoldDepth --model ~/models/marigold-depth-v1-1 --input room.jpg \
  --output room_depth.png --gpu
MarigoldDepth --model ~/models/marigold-normals-v1-1 --input room.jpg \
  --output room_normals.png --ensemble 5 --gpu
```

Outputs:

* depth: the affine-invariant depth in [0,1] as a 16-bit grayscale PNG (value
  x 65535, as diffusers `export_depth_to_16bit_png`), 0 = near.
* normals: the unit normal `n` as an RGB PNG of `(n + 1) / 2`, as diffusers
  `visualize_normals` with no axis flips.

The processing resolution must give a latent grid divisible by the UNet's
downsampling factor (a multiple of 64 for the SD2 UNet); `Predict` raises
otherwise.

## Pico smoke

With no `--model`, the example assembles a diffusers-layout folder in the temp
directory from the committed random pico fixtures
(`tests/fixtures/tiny_sd_unet_linproj.*` with `conv_in` widened to 8 channels
by `MarigoldWidenUNetConvIn`, and `tiny_vae_encoder` + `tiny_vae_decoder`
merged by `MergeSafeTensorsFiles`). It writes `marigold_in.png` (a 24x16
gradient), then `marigold_depth.png` and `marigold_normals.png` with a
3-member ensemble. The pico CLIP fixture is wider than the pico UNet's
`cross_attention_dim`, so the smoke replaces the text encoder with random
states (`SetEmptyPromptStates`). The weights are random, so the outputs check
wiring and file I/O only. It runs in well under a second on the CPU.

Run it from this folder, so that `../../tests/fixtures/` resolves:

```
cd examples/MarigoldDepth
lazbuild -B MarigoldDepth.lpi
../../bin/x86_64-linux/bin/MarigoldDepth
```

## Status

* `TestMarigoldPipelineMatchesPredict` checks that the pipeline reproduces
  `MarigoldPredict` on the pico folder for depth and normals.
* The pico smoke under `--gpu` (PoCL) matches the CPU output to 1/65535.
* Not run here: the real checkpoints. The UNet has ~865M parameters (3.4 GB
  FP32), above this box's memory cap, so real runs (FP32, `--int8`, `--gpu`)
  are untested.
* Not yet checked: parity against diffusers on a pico fixture.
* `--int8` quantizes the UNet after it loads, so the load itself still peaks
  at the FP32 size.
