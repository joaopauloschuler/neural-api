# StableDiffusion

Text-to-image, image-to-image and InstructPix2Pix image editing with a
diffusers Stable Diffusion 1.x folder, through `TStableDiffusionPipeline` in `neural/neuralpretrained.pas`.

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

`--lora FILE[:SCALE]` (repeatable) merges a LoRA `.safetensors` into the
weights as they load, as diffusers `load_lora_weights` + `fuse_lora` do
(`TStableDiffusionPipeline.AddLoRA`, `TNNetLoRAMergeReader`). Every adapted
weight reads as W + SCALE * (alpha / r) * B @ A; nothing is added to the net,
so the UNet and text encoder run at plain speed. Several `--lora` files add
their deltas. The reader takes these key layouts:

* diffusers / PEFT: `unet.<module>.lora_A.weight` / `lora_B`, and
  `text_encoder.<module>...`. alpha / r comes from the file's
  `lora_adapter_metadata` header when it has one, else it is 1.
* older diffusers files: `<module>.lora.down.weight` / `lora.up`,
  `<module>.lora_linear_layer.down.weight`, and attention-processor keys
  (`attn1.processor.to_q_lora.down.weight`); alpha / r is 1, as in diffusers.
* kohya / AUTOMATIC1111: `lora_unet_<module with _>.lora_down.weight` /
  `lora_up` / `alpha`, and `lora_te_` (or `lora_te1_`) for the text encoder;
  the scale is alpha / r (1 without an alpha).

Linear and convolution modules both merge: a 1x1 `proj_in` / `proj_out` and a
3x3 resnet conv (kohya LoCon: down [r, in, 3, 3], up [out, r, 1, 1]). Text
encoder module names with and without `text_model.` both match. A LoRA whose
module matches no tensor, DoRA, LyCORIS LoHa / LoKr, LoRA biases and SDXL
`lora_te2_` keys are refused or skipped: a module that matches nothing raises,
a tensor that is not part of a down/up pair raises, and text encoder 2 keys
are left out. `--lora ~/loras/style.safetensors:0.7` sets SCALE 0.7.

The merge is tested against diffusers float64 `load_lora_weights` +
`fuse_lora` on the pico folder (`TestStableDiffusionPipelineLoRA`: a PEFT file
with metadata alpha, a kohya file with alphas, both stacked; linear and conv
modules of the UNet and the text encoder), and per key layout against
W + scale * B @ A (`TestLoRAMergeSDUNetLayouts`, `TestLoRAMergeSDTextEncoder`).
No community LoRA has been run on a real checkpoint yet.

`--ip-adapter FILE --ip-image PNG [--ip-scale S]` adds an image prompt, as
diffusers `load_ip_adapter` + `set_ip_adapter_scale` do with
`h94/IP-Adapter` `models/ip-adapter_sd15.safetensors` (or `.bin`)
(`TStableDiffusionIPAdapter`). The CLIP image encoder
(`CLIPVisionModelWithProjection`) is read from the `image_encoder/` folder
beside FILE. Once per image, the encoder turns the CLIP-preprocessed image into
`image_embeds`, and the adapter's `image_proj` turns those into 4 image tokens.
Every cross-attention of the UNet gets a second attention over those tokens
(`to_k_ip` / `to_v_ip`), added to the text attention times S before
`to_out.0`. The unguided half of CFG uses the tokens of a zero embedding, as
diffusers does. S = 0 gives the plain pipeline's image. The image preprocessing
reads `feature_extractor/preprocessor_config.json` from the model folder when
it exists, else uses `CLIPImageProcessor` defaults at the encoder's image size:

```
StableDiffusion --model ~/models/stable-diffusion-v1-5 \
  --ip-adapter ~/models/IP-Adapter/models/ip-adapter_sd15.safetensors \
  --ip-image style.png --ip-scale 0.6 --prompt "a cat, best quality"
```

Tested against diffusers float64 on the pico folder with a random
ip-adapter_sd15 and CLIP vision tower (`TestStableDiffusionIPAdapterParity`,
`TestStableDiffusionIPAdapterScaleZero`). No real IP-Adapter has been run yet.
IP-Adapter-Plus, FaceID and SDXL adapters are not supported.

`--init-image PNG [--strength S]` turns the run into image-to-image, as
diffusers `StableDiffusionImg2ImgPipeline` (`TStableDiffusionImg2Img`). The
image is VAE-encoded (the latents are sampled: mean + std * noise), noised to
the first timestep that runs, and denoised with the prompt. S (default 0.8)
sets how much of the image is repainted: only the last `int(steps * S)` steps
run, so `--steps 25 --strength 0.6` runs 15 UNet steps. S = 1 still starts
from the noised image, not from pure noise. The output takes the image's size,
rounded down to a multiple of the VAE factor; `--width` / `--height` resize
the image instead. With `--ip-adapter` too, the image prompt joins the
img2img run (`TStableDiffusionMultiConditioner`, as `load_ip_adapter` on
`StableDiffusionImg2ImgPipeline`; not compared with diffusers):

```
StableDiffusion --model ~/models/stable-diffusion-v1-5 \
  --init-image sketch.png --strength 0.75 \
  --prompt "a fantasy landscape, trending on artstation"
```

Tested against diffusers float64 on the pico folder at strength 0.6 with DDIM,
PNDM, DPM-Solver++ 2M, Euler and Euler a (replayed step noise), and with DDIM
at strength 1 (`TestStableDiffusionImg2ImgParity`). No real checkpoint has
been run yet.

`--edit-image PNG [--image-guidance S]` edits the image as the prompt
instructs ("make it winter"), as diffusers
`StableDiffusionInstructPix2PixPipeline` (`TStableDiffusionInstructPix2Pix`).
It needs an InstructPix2Pix folder such as `timbrooks/instruct-pix2pix`,
whose UNet takes 8 channels: the latents and the VAE latent mean of the
image. Every step runs three UNet passes (prompt + image, image only, neither)
and combines them as
`uncond + guidance * (text - image) + image_guidance * (image - uncond)`.
S (default 1.5) sets how closely the output keeps the image. Guidance is on
when `--guidance` > 1 and S >= 1; otherwise each step runs one pass. All steps
run from pure noise. The output takes the image's size, rounded down to a
multiple of the VAE factor. `--edit-image` cannot be combined with
`--init-image`. With `--ip-adapter`, the image prompt goes with the prompt
pass only, as diffusers' `[image, zero, zero]` embeds
(`TestStableDiffusionIPAdapterInstructPix2PixParity`):

```
StableDiffusion --model ~/models/instruct-pix2pix \
  --edit-image photo.png --image-guidance 1.5 --steps 20 \
  --prompt "turn him into a statue"
```

Tested against diffusers float64 on a pico 8-channel folder
(`tests/fixtures/tiny_sd15_ip2p`) with Euler a (the checkpoint's scheduler,
replayed step noise) and DDIM at guidance 7.5 / image guidance 1.5, and at
1 / 1 (`TestStableDiffusionInstructPix2PixParity`). No real checkpoint has
been run yet.

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
  --lora FILE[:SCALE] merge a LoRA .safetensors (diffusers/PEFT or kohya/A1111
                    keys; UNet and text encoder) at SCALE (default 1.0);
                    repeatable
  --ip-adapter FILE IP-Adapter ip-adapter_sd15 (.safetensors/.bin) with its
                    image_encoder/ folder beside it; needs --ip-image
  --ip-image PNG    the image prompt of --ip-adapter
  --ip-scale S      IP-Adapter scale (default 1.0; 0 ignores the image)
  --init-image PNG  image-to-image from this image; the output takes its size
                    (rounded down to the VAE factor) unless --width/--height
  --strength S      how much of --init-image to repaint, (0, 1] (default 0.8);
                    runs the last int(steps * S) steps
  --edit-image PNG  InstructPix2Pix: edit this image as the prompt says (needs an
                    8-channel UNet folder, e.g. timbrooks/instruct-pix2pix); the
                    output takes its size (rounded down to the VAE factor)
  --image-guidance S how closely --edit-image is kept (default 1.5; < 1 turns all
                    guidance off)
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
* The noise (the initial latents, the Euler a step noise and the img2img VAE
  sample) comes from the Free Pascal RNG, so a seed does not reproduce a diffusers image.
* Every UNet pass recomputes the cross-attention K/V of the text states,
  which do not change across steps. This costs about 0.4% of a 512x512 SD 1.5
  pass, so the K/V are not cached.

## Build

```
lazbuild StableDiffusion.lpi
```
