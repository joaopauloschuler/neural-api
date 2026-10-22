# StableDiffusion

Text-to-image with a diffusers Stable Diffusion 1.x folder, through
`TStableDiffusionPipeline` in `neural/neuralpretrained.pas`.

One `Generate` call:

1. tokenizes the prompt and the negative prompt with the CLIP tokenizer in
   `tokenizer/` (`vocab.json` + `merges.txt`): `[bos] ids [eos]`, padded to 77
   tokens with the pad token of `tokenizer_config.json`,
2. encodes both with the CLIP text encoder in `text_encoder/` (last hidden
   state),
3. runs the denoising steps with the UNet in `unet/`: per step one pass on the
   prompt and one on the negative prompt (empty by default), mixed by
   classifier-free guidance,
4. decodes the latents with the VAE decoder in `vae/`.

The pipeline builds the text encoder, the UNet and the VAE decoder one at a
time and frees each one after its stage. Peak memory is therefore the UNet, not
the sum of all components.

The betas, the timestep spacing (`leading`, `trailing` or `linspace`, with
`steps_offset`) and the prediction type come from
`scheduler/scheduler_config.json`. Its `_class_name` picks the sampler:

| `_class_name`                     | sampler    | `--sampler` |
|-----------------------------------|------------|-------------|
| `DDIMScheduler`                   | DDIM, eta 0 | `ddim`     |
| `PNDMScheduler` (SD 1.5 default)  | PLMS (`skip_prk_steps`) | `pndm` |
| `EulerDiscreteScheduler`          | Euler      | `euler`     |
| `EulerAncestralDiscreteScheduler` | Euler a    | `euler-a`   |
| `DPMSolverMultistepScheduler`     | DPM-Solver++ 2M | `dpm++2m` |

Any other class runs DDIM and prints a `Notice:` line. The notice also lists
config options that are not implemented and are ignored, e.g.
`use_karras_sigmas`, `solver_order` other than 2, or `skip_prk_steps: false`
(the Runge-Kutta warm-up). `--sampler` overrides the config's choice. PNDM
runs one more UNet step than `--steps`, as diffusers does.

## Usage

```
StableDiffusion [--model DIR] [options]
  --model DIR       diffusers StableDiffusionPipeline folder (default: the pico fixture)
  --prompt TEXT     prompt (default "a photo of an astronaut riding a horse on mars")
  --negative TEXT   negative prompt (default empty)
  --steps N         denoising steps (default 25)
  --guidance G      classifier-free guidance scale (default 7.5; <= 1 turns it off)
  --size N          square image side (default: the checkpoint's, 512 for SD 1.5)
  --seed N          noise seed (default 0)
  --sampler NAME    auto (default: from the scheduler config), ddim, pndm,
                    euler, euler-a or dpm++2m
  --output PNG      output file (default stable_diffusion.png)
  --gpu             run the nets on OpenCL (build with -dOpenCL)
  --gpu-platform N  OpenCL platform index (default 0)
  --gpu-device N    OpenCL device index (default 0)
```

```
StableDiffusion --model ~/models/stable-diffusion-v1-5 \
  --prompt "a watercolor painting of a lighthouse at dawn" --gpu
```

With no `--model` the example runs the committed random pico folder
`tests/fixtures/tiny_sd15` (16x16, random weights): it checks wiring and I/O
only.

## Limits

* Only Stable Diffusion 1.x is covered by tests (pico parity against diffusers
  in `TestStableDiffusionPipelineParity` for DDIM and
  `TestStableDiffusionPipelineSamplers` for PNDM, Euler and DPM-Solver++ 2M;
  Euler a against diffusers with replayed noise in `TestNeuralDiffusion`).
  No real checkpoint has been run yet.
* Euler with `linspace` spacing rounds the timesteps to integers; diffusers
  keeps fractional ones when `T - 1` is not a multiple of `steps - 1`.
* Images are square: the UNet and VAE builders take one latent grid side.
* The weights must be F32 `*.safetensors`; the `*.fp16.safetensors` variant
  files are untested.
* The noise (the initial latents and the Euler a step noise) comes from the
  Free Pascal RNG, so a seed does not reproduce a diffusers image.

## Build

```
lazbuild StableDiffusion.lpi
```
