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

With `--repeat N` (N > 1) the example calls `LoadComponents` instead: the
three nets are built once and stay in memory for all N images, and the prompt
is encoded once. A kept UNet or VAE decoder is rebuilt only when the latent
grid changes (a new image size), and a kept text encoder only when the token
count changes. In the library, `TStableDiffusionPipeline.LoadComponents` /
`UnloadComponents` do the same for any sequence of calls; the images are
bit-identical to the build-per-stage path
(`TestStableDiffusionPipelineKeepLoaded`).

`--int8` stores the UNet and text encoder weights in int8
(`TNNet.QuantizeWeightsInt8`, the `TMarigoldPipeline` path; properties
`UNetInt8` and `TextEncoderInt8`). The VAE decoder stays FP32. On the pico
folder the int8 image is within 0.0092 of the FP32 one (pixels in [0, 1]) and
the prompt embeds within 0.045 (`TestStableDiffusionPipelineInt8`); the
quality on a real checkpoint is not measured yet.

`--embedding FILE[:TOKEN]` (repeatable) loads a textual inversion concept, as
diffusers `load_textual_inversion` does
(`TStableDiffusionPipeline.LoadTextualInversion`). FILE is a diffusers
`learned_embeds.safetensors` / `.bin` (one `{token: [N, hidden]}` tensor) or an
AUTOMATIC1111 `.pt` (`string_to_param['*']`, token from its `name`). TOKEN
replaces the file's token. The concept's N vectors become N new tokens (TOKEN,
TOKEN_1, ..., TOKEN_{N-1}) after the CLIP vocab, and a prompt that names TOKEN
gets all N, as diffusers' `maybe_convert_prompt`. Put the token in `--prompt`
or `--negative`:

```
StableDiffusion --model ~/models/stable-diffusion-v1-5 \
  --embedding ~/embeddings/cat-toy/learned_embeds.safetensors \
  --prompt "a <cat-toy> on a beach"
```

The vector width must equal the text encoder's hidden size (768 for SD 1.x;
an SD 2.x embedding is 1024 wide and is rejected). As in diffusers, the token
matches in any case, but only an all-lowercase token expands to its N vectors:
a mixed-case token such as `EasyNegative` uses its first vector only. Pass a
lowercase TOKEN (`file.pt:easynegative`) to get all of them. Tested against
diffusers float64 on the pico folder (`TestStableDiffusionTextualInversion*`);
no real embedding has been run yet.

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
  --width N         image width (default: the checkpoint's, 512 for SD 1.5)
  --height N        image height (default: the checkpoint's)
  --size N          sets both --width and --height
  --seed N          noise seed (default 0)
  --sampler NAME    auto (default: from the scheduler config), ddim, pndm,
                    euler, euler-a or dpm++2m
  --output PNG      output file (default stable_diffusion.png)
  --repeat N        N images, seeds --seed, --seed+1, ...; the prompt is encoded
                    once. N > 1 keeps the components loaded and numbers the files
                    (stable_diffusion_0001.png, ...; default 1)
  --embedding F[:T] textual inversion embedding (diffusers learned_embeds
                    .safetensors/.bin or A1111 .pt) under token T (default: the
                    file's); use the token in the prompts; repeatable
  --int8            UNet and text encoder weights in int8 (default FP32)
  --stats           after each image, a stage table (wall ms; with --gpu also
                    the host<->OpenCL MB of each stage)
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

## Stage table (`--stats`)

After each image, `--stats` prints the wall time of every stage that ran, from
`TStableDiffusionPipeline.ImageStats`. A load row appears only when that net
was built for this image; with `--repeat` the prompt rows appear only for
image 1, which encoded the prompt. The denoise row gives the UNet passes (two
per step with guidance). With `--gpu` the table also gives the host<->OpenCL
megabytes moved in each stage. Pico folder, `--repeat 2 --stats --steps 2
--gpu` on PoCL (CPU OpenCL), image 2:

```
Stages of image 2 (wall time):
  Stage                      ms      %
  encode prompt               -      -  encoded once, before image 1
  denoise                  19.0   79.2  4 UNet passes x 4.8 ms  up 0.63 MB, down 0.63 MB
  VAE decode                2.0    8.3  up 0.00 MB, down 0.00 MB
  PNG save                  3.0   12.5
  sum of stages            24.0  100.0
```

## Limits

* Only Stable Diffusion 1.x is covered by tests (pico parity against diffusers
  in `TestStableDiffusionPipelineParity` for DDIM and
  `TestStableDiffusionPipelineSamplers` for PNDM, Euler and DPM-Solver++ 2M;
  Euler a against diffusers with replayed noise in `TestNeuralDiffusion`).
  No real checkpoint has been run yet.
* Euler with `linspace` spacing rounds the timesteps to integers; diffusers
  keeps fractional ones when `T - 1` is not a multiple of `steps - 1`.
* Width and height must each be a multiple of the VAE factor (8 for SD 1.5;
  2 for the pico folder). When a latent side is not a multiple of the UNet
  down factor (a 520-pixel side on SD 1.5), `BuildSDUNet` crops each nearest
  2x upsample to the size of its skip, as diffusers does with `upsample_size`.
  Non-square sizes are tested against diffusers at 32x16 on the pico folder
  (`TestStableDiffusionPipelineNonSquare`), odd latent grids at 14x10, a 7x5
  latent grid (`TestStableDiffusionPipelineOddSize`).
* The weights must be F32 `*.safetensors`; the `*.fp16.safetensors` variant
  files are untested.
* The noise (the initial latents and the Euler a step noise) comes from the
  Free Pascal RNG, so a seed does not reproduce a diffusers image.
* Every UNet pass recomputes the cross-attention K/V of the text states,
  which do not change across steps. This costs about 0.4% of a 512x512 SD 1.5
  pass, so the K/V are not cached.

## Build

```
lazbuild StableDiffusion.lpi
```
