# StableDiffusion

Text-to-image, image-to-image, depth-to-image, InstructPix2Pix image
editing and x4 upscaling with a diffusers Stable Diffusion 1.x or 2.x folder, through `TStableDiffusionPipeline` in `neural/neuralpretrained.pas`.

One `Generate` call:

1. tokenizes the prompt and the negative prompt with the CLIP tokenizer in
   `tokenizer/` (`tokenizer.json` when present, else `vocab.json` +
   `merges.txt`): `[bos] ids [eos]`, padded to 77
   tokens with the pad token (`special_tokens_map.json` wins over
   `tokenizer_config.json`, as in transformers; SD 2.x pads with `!`),
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
count changes. A rebuild for a new size reads no weights file: the new net
borrows the weights that the first build read (`TestStableDiffusionPipelineSizeChangeBorrows`,
`WeightFileLoadCount`). In the library, `TStableDiffusionPipeline.LoadComponents` /
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

The vector width must equal the text encoder's hidden size (768 for SD 1.x,
1024 for SD 2.x; an embedding made for the other family is rejected). As in diffusers, the token
matches in any case, but only an all-lowercase token expands to its N vectors:
a mixed-case token such as `EasyNegative` uses its first vector only. Pass a
lowercase TOKEN (`file.pt:easynegative`) to get all of them. Tested against
diffusers float64 on the pico folder (`TestStableDiffusionTextualInversion*`);
no real embedding has been run yet.

### Long prompts and prompt weighting

A prompt longer than 75 tokens is not cut at 75. As diffusers'
`lpw_stable_diffusion` community pipeline and AUTOMATIC1111 do,
`TStableDiffusionPipeline.EncodePrompt` splits the token ids into chunks of 75,
wraps each chunk in `<|startoftext|>` ... `<|endoftext|>` padded to 77, encodes
every chunk through the text encoder and concatenates the chunk states, so the
UNet sees 77, 154 or 231 text tokens. The negative prompt is padded to the
same chunk count (`EncodePrompts`). `--max-prompt-chunks N` (property
`MaxPromptChunkCount`, default 3 as lpw's `max_embeddings_multiples`; 0 for no
limit) caps the chunk count; a longer prompt is cut and the program prints a
`Notice:` line saying how many tokens were dropped (`PromptNotice`).

Prompts take the AUTOMATIC1111 weighting syntax (property `PromptWeighting`,
on by default; `--no-prompt-weighting` reads the brackets and `BREAK` as plain
text):

| Syntax | Weight |
| --- | --- |
| `(word:1.3)` | 1.3 |
| `(word)`, `((word))` | 1.1, 1.21 |
| `[word]` | 1 / 1.1 |
| `\(`, `\)`, `\[`, `\]`, `\\` | the literal character |
| `BREAK` | ends the current chunk (see below) |

Each token's text-encoder output row is multiplied by its weight, then the
whole prompt's states are rescaled to their unweighted mean, as lpw's
`get_weighted_text_embeddings` does. A textual inversion token works inside a
weighted span: `(<cat-toy>:1.3)` weights all of its vectors.

`BREAK` (uppercase, a whole word) works as in AUTOMATIC1111: it closes the
current chunk, which is padded as `<|startoftext|>` ids `<|endoftext|>` pads,
and the text after it starts the next chunk. A `BREAK` at the start of the
prompt or straight after another `BREAK` gives an empty chunk; a `BREAK` right
after a full chunk of 75 tokens, or at the end of the prompt, adds nothing.
The chunks still count toward `--max-prompt-chunks`, and the `Notice:` line
reports chunks cut that way. Lowercase `break` is a word. Unlike A1111, a
`BREAK` inside brackets still breaks (A1111 turns it into the word with a
negative weight).

```
StableDiffusion --prompt "a castle on a hill, sunset BREAK red flags, stone walls"
```

```
StableDiffusion --model ~/models/stable-diffusion-v1-5 \
  --prompt "a (photo:1.2) of an ((astronaut)) riding a [horse] on mars" \
  --negative "(blurry:1.3), low quality"
```

Tested against a reimplementation of lpw's `get_weighted_text_embeddings` in
float64 on the pico folder (`tools/make_pico_sd_prompt_weighting_fixture.py`;
the community pipeline is not packaged with diffusers): token ids and weights
match exactly, and 2-chunk weighted prompts are within 1.7e-6
(`TestStableDiffusionLongPrompt*`, `TestSDPromptWeightParser`). `BREAK` has no
lpw counterpart; `TestSDPromptBreakParser` and
`TestStableDiffusionPromptBreakTokens` check the A1111 chunk layout above.

`--clip-skip N` (property `ClipSkip`, diffusers `clip_skip`) takes the text
encoder's hidden state N layers before the last, through its final layer norm:
`TStableDiffusionPipeline` builds the text encoder with only its first
`num_hidden_layers - N` layers. diffusers' `clip_skip=1` is AUTOMATIC1111's
"Clip skip: 2". As A1111 does, the skip applies to the negative prompt too;
diffusers' `encode_prompt` encodes the negative prompt with every layer. Tested
against diffusers `encode_prompt(clip_skip=1)` on the pico folder within 1e-4
(`TestStableDiffusionClipSkipParity`, `tools/make_pico_sd_clip_skip_fixture.py`).

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

`--init-image PNG` on a depth folder such as
`stabilityai/stable-diffusion-2-depth` (whose UNet takes 5 channels: the
latents and the depth) runs depth-to-image, as diffusers
`StableDiffusionDepth2ImgPipeline` (`TStableDiffusionDepth2Img`). The run is
image-to-image (same `--strength`) with the depth map added to every UNet
pass. By default the folder's own `depth_estimator/` (DPT-Hybrid: a BiT
ResNet + ViT, `BuildDPT`) estimates the depth from the init image, after the
`feature_extractor/` `DPTImageProcessor` resize (bilinear to 384x384, mean and
std 0.5), as diffusers does with `depth_map=None`. `--depth-image PNG`
supplies the depth map instead: it is read as its luma (brighter = nearer, as
the relative inverse depth of DPT / MiDaS; a 16-bit PNG keeps all 16 bits), for example the grayscale PGM of
`examples/DepthAnythingV2` saved as PNG (`examples/MarigoldDepth` writes 0 =
near: invert it first). Either map is resized to the latent grid with torch
bicubic interpolation and normalized to [-1, 1] by its minimum and maximum;
any size works. A depth UNet without `--init-image` stops with an error.

```
StableDiffusion --model ~/models/stable-diffusion-2-depth \
  --init-image photo.png --strength 0.7 \
  --prompt "a marble statue in a museum"
StableDiffusion --model ~/models/stable-diffusion-2-depth \
  --init-image photo.png --depth-image photo_depth.png --strength 0.7 \
  --prompt "a marble statue in a museum"
```

Tested against diffusers float64 on a pico 5-channel SD 2.x folder
(`tests/fixtures/tiny_sd2_depth`) at strength 0.6 with DDIM and Euler a
(replayed step noise), with a depth map smaller and one larger than the
latent grid (`TestStableDiffusion2Depth2ImgParity`, and on OpenCL
`TestStableDiffusion2Depth2ImgOpenCL`), and with the depth estimated by a
pico DPT-Hybrid (`tests/fixtures/tiny_sd2_depth_estimator`,
`TestStableDiffusion2Depth2ImgEstimatorParity` / `...EstimatorOpenCL`). No
real checkpoint has been run yet. The estimator runs at the size its
`feature_extractor` resizes the image to (`keep_aspect_ratio` and
`ensure_multiple_of` included), with the position table interpolated when that
is not `image_size`; the stock processor resizes to `image_size`. With `--repeat N`
(`LoadComponents`) it is built once and kept with the other components; it
still estimates the depth for every image
(`TestStableDiffusion2Depth2ImgEstimatorKept`).

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

## x4 upscaler (`--upscale-image`)

`--upscale-image PNG [--noise-level N]` upscales an image 4x with an x4
upscaler folder such as `stabilityai/stable-diffusion-x4-upscaler`, as
diffusers `StableDiffusionUpscalePipeline` (`TStableDiffusionUpscaler`). The
prompt describes the image; it steers the detail the UNet adds.

* The VAE has 3 blocks (factor 4, `scaling_factor` 0.08333), so the latents
  have the size of the low-res image and the decoded image is 4 times larger.
  The image sides are first rounded down to a multiple of 4 with PIL bicubic,
  as diffusers' `VaeImageProcessor(resample="bicubic")`; a 128x128 image
  gives 512x512.
* `low_res_scheduler/` (a DDPM schedule) noises the low-res image to
  `--noise-level` N (default 20, at most the `max_noise_level` 350 of
  `model_index.json`). More noise lets the UNet invent more detail and keep
  less of the input.
* The UNet takes 7 channels: the latents and the noised RGB image. N is also
  its class label: `num_class_embeds` 1000 adds one learned row per noise
  level into the timestep embedding (`BuildSDUNet` reads `num_class_embeds`).
  A block with `only_cross_attention` in the UNet config builds its first
  attention over the text tokens instead of over every pixel.
* Both passes of classifier-free guidance see the same noised image. The
  defaults follow diffusers: 75 steps and guidance 9.0; the sampler and the
  prediction type come from `scheduler/` as usual. `--width`, `--height` and `--size`
  are refused (the image sets the size); `--init-image` and `--edit-image`
  cannot be combined with it.

```
StableDiffusion --model ~/models/stable-diffusion-x4-upscaler \
  --upscale-image cat_128.png --prompt "a white cat" --output cat_512.png
StableDiffusion --model ~/models/stable-diffusion-x4-upscaler \
  --upscale-image cat_128.png --prompt "a white cat" --noise-level 50 \
  --steps 30 --gpu
```

Tested against diffusers float64 on the pico folder
`tests/fixtures/tiny_sd_x4_upscaler` (`tools/make_pico_sd_upscale_fixture.py`:
a 7-channel UNet with `num_class_embeds` and `only_cross_attention`, a 3-block
VAE, the `tiny_sd2` text encoder): DDIM on a 12x8 image at noise level 20,
Euler on a 13x10 image (resized to 12x8) at noise level 150, and the first
case again with the components kept; the latents after each step match to
1e-5 and the 48x32 image to 4e-6 (`TestStableDiffusionUpscalerParity`; the
UNet alone in `TestSDUNetClassEmbedOnlyCrossParity`). With OpenCL enabled the
same cases match within 1e-4 (`TestStableDiffusionUpscalerOpenCL`; the
class-label embedding runs on the CPU). No real checkpoint has
been run yet. The UNet runs at the low-res size: a 128x128 input is a
128x128 latent grid (SD 1.5 at 512x512 runs 64x64), so plan on a GPU.

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
| `LCMScheduler`                    | LCM (see below) | `lcm`   |

Any other class runs DDIM and prints a `Notice:` line. The notice also lists
config options that are not implemented and are ignored, e.g. `solver_order`
other than 2, or `skip_prk_steps: false` (the Runge-Kutta warm-up).
`--sampler` overrides the config's choice. PNDM runs one more UNet step than
`--steps`, as diffusers does.

Euler, Euler a and DPM-Solver++ 2M follow `use_karras_sigmas` as diffusers
does: the sigmas follow the Karras rho = 7 ramp and stay continuous, and each
sigma maps back to a timestep by log-sigma interpolation. Euler takes the
sigma range from its spacing's first and last timesteps and gives the UNet the
fractional timestep; DPM-Solver++ 2M spans the whole training schedule and
rounds the timestep. Euler and Euler a with `linspace` spacing also give the
UNet fractional timesteps (e.g. 749.25 at 5 steps). `--karras-sigmas` turns
Karras sigmas on for any config (Euler a too, which diffusers does not offer),
and `--no-karras-sigmas` turns them off. Pico parity `< 1e-4` against diffusers
float64 (`TestStableDiffusionPipelineKarrasSigmas`). Image-to-image and the
4-channel inpainting blend still run these samplers at integer timesteps.

## Latent Consistency Models (`--sampler lcm`)

An LCM-LoRA turns a plain SD 1.5 checkpoint into a 2-8 step sampler: about
6-10x fewer UNet passes than 25-50 step sampling. Merge the LoRA with `--lora`
and pick the LCM sampler:

```
StableDiffusion --model ~/models/stable-diffusion-v1-5 \
  --lora ~/models/lcm-lora-sdv1-5/pytorch_lora_weights.safetensors \
  --sampler lcm --prompt "a watercolor painting of a lighthouse at dawn"
```

`--sampler lcm` runs diffusers `LCMScheduler` (a `scheduler_config.json` with
that `_class_name` selects it too). With it, `--steps` defaults to 4 and
`--guidance` to 1.0. Guidance 1.0 runs one UNet pass per step. Guidance above 1
runs the usual two-pass classifier-free guidance; LCM-LoRA works best at
1.0-2.0. The betas come from the scheduler config. The timesteps come from
the LCM origin grid of `original_inference_steps` (50) points, k*i - 1 with
k = 1000 / 50; 4 steps run at t = 999, 759, 499 and 259. Each step maps the
latents to a clean estimate and re-noises it to the next timestep with fresh
noise. The last step returns the clean estimate.

With `--init-image`, the strength shrinks the origin grid to its first
int(50 * strength) points, and all `--steps` steps run on it (strength 0.5,
4 steps: t = 499, 379, 259, 139). This is the schedule of diffusers
`LatentConsistencyModelImg2ImgPipeline`. diffusers
`StableDiffusionImg2ImgPipeline` with an `LCMScheduler` instead keeps the last
int(steps * strength) entries of the full grid (t = 499, 259).

### LCM-distilled UNets

An LCM-distilled folder such as `SimianLuo/LCM_Dreamshaper_v7` needs no LoRA.
Its `scheduler_config.json` already names `LCMScheduler`, so `--sampler lcm` is
optional:

```
StableDiffusion --model ~/models/LCM_Dreamshaper_v7 \
  --prompt "a watercolor painting of a lighthouse at dawn"
```

Its `unet/config.json` sets `time_cond_proj_dim` (256): the UNet takes the
guidance scale as an input instead of running classifier-free guidance. The
pipeline feeds the sinusoidal embedding of (w - 1) * 1000 (w = `--guidance`)
through `time_embedding.cond_proj` and adds it to the timestep embedding,
as diffusers `LatentConsistencyModelPipeline` does. Every step runs one UNet
pass at any guidance, and no negative prompt is encoded (`--negative` has no
effect). `--guidance` defaults to 8.5 for such a UNet, the diffusers default.
`--init-image` works the same way as above
(`LatentConsistencyModelImg2ImgPipeline`).

Tested against diffusers float64 on the pico folder (an `LCMScheduler` from the
DDIM config, replayed step noise): with the base UNet, 4 steps at guidance 1.5
and 1.0 and img2img at strength 0.5 (`TestStableDiffusionLCMParity`); with a
pico LCM-distilled UNet, 4 steps at guidance 8.5 and 1.0 and img2img at
strength 0.5 (`TestStableDiffusionLCMDistilledParity`), plus one UNet call
with the guidance embedding (`TestSDUNetTimeCondProjDimParity`). No real
LCM-LoRA or LCM-distilled checkpoint has been run yet.

## Checkpoints

| Checkpoint | Status |
| --- | --- |
| Stable Diffusion 1.x (e.g. `runwayml/stable-diffusion-v1-5`) | coded; pico parity against diffusers; no real checkpoint run yet |
| Stable Diffusion 2.x (e.g. `stabilityai/stable-diffusion-2-1`, 768x768, `v_prediction`) | coded; pico parity against diffusers only; no real checkpoint run yet, so not supported |
| Stable Diffusion x4 upscaler (`stabilityai/stable-diffusion-x4-upscaler`) | coded; pico parity against diffusers only; no real checkpoint run yet, so not supported |
| SDXL | not supported (the pipeline refuses `addition_embed_type` `text_time`) |

What SD 2.x changes, and how the pipeline reads it from the folder:

* `scheduler_config.json` `prediction_type` `v_prediction`: every sampler
  (DDIM, PNDM, Euler, Euler a, DPM++ 2M, LCM) converts the UNet's v output,
  in text-to-image, img2img and inpainting.
* The text encoder is OpenCLIP ViT-H as a `CLIPTextModel`: `hidden_act`
  `gelu` (exact erf), hidden size 1024, and `num_hidden_layers` 23 of the
  original 24, so its last hidden state is OpenCLIP's penultimate layer.
* The UNet config has `use_linear_projection` and a head count per block
  (`attention_head_dim` `[5, 10, 20, 20]`; diffusers reads it as head counts).
* `special_tokens_map.json` sets the pad token to `!`. transformers also
  treats it as a special token, so each `!` in a prompt becomes its own token
  (id 0); the pipeline does the same.
* The default size is the UNet `sample_size` times the VAE factor:
  96 x 8 = 768 for SD 2.1, 64 x 8 = 512 for SD 2.1-base and SD 1.5.

The pico SD 2.x folder `tests/fixtures/tiny_sd2`
(`tools/make_pico_sd2_pipeline_fixture.py`) has all of the above at random
weights. `TestStableDiffusion2*` match diffusers float64 below 1e-4 on the
tokens, the prompt embeds, the latents after each step and the decoded image,
with CFG, for DDIM, PNDM, Euler, DPM++ 2M and LCM, and for img2img with DDIM,
PNDM, DPM++ 2M, Euler and Euler a. Inpainting with `v_prediction` is not
tested separately: it noises the image latents with the same scheduler call
as img2img, which does not depend on the prediction type.

## FreeU (`--freeu`)

FreeU (Si et al. 2023, arXiv:2309.11497) reweights the two inputs of the first
two UNet up blocks with no extra UNet pass and no new weights.
`--freeu s1,s2,b1,b2` does what diffusers `pipe.enable_freeu(s1, s2, b1, b2)`
does. At each skip connection of up block 0 (and 1), before the concat:

* the first half of the backbone channels is multiplied by `b1` (`b2`);
* the skip is filtered: per channel, the low-frequency box of its `fftshift`ed
  2-D spectrum (the DC, `(-1,0)`, `(0,-1)` and `(-1,-1)` bins) is scaled by
  `s1` (`s2`), as diffusers `fourier_filter(threshold=1)`.

The presets of the FreeU paper are `--freeu sd15` (0.9,0.2,1.5,1.6) and
`--freeu sd21` (0.9,0.2,1.4,1.6). A zero factor turns FreeU off, as in
diffusers.

```
StableDiffusion --model ~/models/stable-diffusion-v1-5 --freeu sd15 \
  --prompt "a watercolor painting of a lighthouse at dawn" --gpu
```

`BuildSDUNet` adds the layers only when `TSDUNetConfig.UseFreeU` is set
(`SDUNetSetFreeU`; `TStableDiffusionPipeline.EnableFreeU` / `DisableFreeU`).
The skip filter is `TNNetFreeUFourierFilter`: it projects the few cosine/sine
components of the box out of each channel in O(H*W), without an FFT, and has
an OpenCL forward, so a `--gpu` UNet keeps its activations in OpenCL memory.
The backbone scale reuses `TNNetSplitChannels`, `TNNetMulByConstant` and
`TNNetDeepConcat`. All of these layers are weightless, so a kept UNet still
borrows the weights it read once.

Tested against diffusers float64 with `enable_freeu(0.9, 0.2, 1.5, 1.6)` on
the pico folder: the filter alone on odd, even and 1-wide sides
(`TestFreeUFourierFilterTorchParity`), and the step-0 UNet output, every DDIM
step and the image at 32x16 and 14x10
(`TestStableDiffusionPipelineFreeUNonSquare`,
`TestStableDiffusionPipelineFreeUOddSize`; OpenCL:
`TestStableDiffusionPipelineFreeUOpenCL`), all < 1e-4. No real checkpoint has
been run with FreeU yet.

## DeepCache (`--deepcache`)

DeepCache (Ma et al. 2023, arXiv:2312.00858; reference code
horseee/DeepCache `DeepCacheSDHelper`) skips most of the UNet on most steps.
The deep features of the UNet change little from one step to the next, so
`--deepcache N` runs the whole UNet on steps 0, N, 2N, ... of each image and,
on the steps between, only the shallow branch: `conv_in`, the time embedding,
the first resnets of down block 0, the last resnets of the last up block and
`conv_out`. The input of those last up-block resnets comes from the last
whole run instead of from the deep layers. This is DeepCache's uniform mode;
it needs no new weights and changes the image (it is an approximation).

`--deepcache-branch B` (DeepCache `cache_branch_id`, 0..`layers_per_block`)
sets the size of the shallow branch: B resnets of down block 0 and B + 1
resnets of the last up block run (with their attention). The default is
`layers_per_block` (2 on SD 1.5): down block 0 and the last up block run
whole. `--deepcache-branch 0` runs the fewest layers and is DeepCache's own
default (`set_params(cache_interval=3, cache_branch_id=0)`).

```
StableDiffusion --model ~/models/stable-diffusion-v1-5 --steps 50 \
  --sampler ddim --deepcache 3 --prompt "a watercolor painting of a lighthouse" --gpu --stats
```

Each kind of UNet pass keeps its own deep features: the prompt and the
unconditional pass of classifier-free guidance, and the three passes of
InstructPix2Pix. img2img and inpainting count the steps from the first step
they run. A ControlNet still runs every step, but on a shallow-only step only
its residuals for the shallow skips reach the UNet; T2I-Adapter features of
deep blocks are likewise those of the last whole run. `--stats` shows how many
passes ran only the shallow branch.

`TStableDiffusionPipeline.DeepCacheInterval` (1 = off) and `DeepCacheBranch`
are the API. `BuildSDUNet` records the split of every branch
(`TSDUNetDeepCacheLayout`) and checks that no shallow layer reads a deep one;
`SDUNetDenoise` with a `TDiTFirstBlockCache` (`Interval`) runs layers up to
the split, writes back the kept output of the cache point and runs the rest.
diffusers has no DeepCache, so the tests compare with a hand-written loop on
the pico folder instead: interval 1 bit-identical to the plain loop, interval
3 bit-identical to a loop that runs the whole UNet and then reruns the tail
from the kept output, on every branch, with CFG
(`TestStableDiffusionDeepCacheReference`); img2img
(`TestStableDiffusionDeepCacheImg2Img`); OpenCL against the CPU < 1e-4
(`TestStableDiffusionDeepCacheOpenCL`). Speed and image quality on a real
SD 1.5 checkpoint have not been measured yet.

## Perturbed-Attention Guidance (`--pag-scale`)

Perturbed-Attention Guidance (PAG; Ahn et al. 2024, arXiv:2403.17377) runs
one more UNet pass per step, on the prompt, with the self-attention (`attn1`)
of the chosen transformer blocks replaced by identity attention: each token
keeps its own value row, so the attention output is `to_out(to_v(x))`. The
step's noise moves away from that perturbed prediction:

* with CFG: `uncond + guidance * (cond - uncond) + pag * (cond - perturbed)`
* without CFG (`--guidance 1`): `cond + pag * (cond - perturbed)`

This is diffusers' `StableDiffusionPAGPipeline` (`PAGMixin`). It runs on
every step; diffusers' default `pag_scale` is 3.0 and the default layers are
`mid` (the self-attention of the mid block).

```
StableDiffusion --model ~/models/stable-diffusion-v1-5 --steps 25 \
  --pag-scale 3 --prompt "a watercolor painting of a lighthouse" --gpu
```

`--pag-layers L1,L2,...` takes diffusers' `pag_applied_layers` patterns. Each
one is a regular expression searched in the module name of every UNet
self-attention, for example
`down_blocks.1.attentions.0.transformer_blocks.0.attn1`,
`mid_block.attentions.0.transformer_blocks.0.attn1` or
`up_blocks.2.attentions.1.transformer_blocks.0.attn1`. So `mid` takes the mid
block, `down_blocks.2` every self-attention of down block 2, and
`up_blocks\.1\.attentions\.0` one transformer of up block 1. A pattern that
matches nothing is an error, as in diffusers. The example prints the
self-attentions it perturbs. `--pag-adaptive-scale A` lowers the scale to
`max(0, pag - A * (1000 - t))` at timestep t (diffusers `pag_adaptive_scale`).

PAG combines with img2img, inpainting, depth2img, ControlNet, T2I-Adapter,
IP-Adapter, LoRA, FreeU and DeepCache (the perturbed pass keeps its own deep
features). It does not combine with `--edit-image` (diffusers has no
InstructPix2Pix PAG pipeline).

`TStableDiffusionPipeline.PerturbedGuidance` (a `TNNetPerturbedGuidance`, the
settings class of the Qwen-Image `--pag-scale`; `PerturbedScale`,
`AdaptiveScale`, `SetPerturbedLayers`) is the API. `SDUNetSelfAttentionNames`
lists the diffusers names in index order and `SDUNetPAGLayerIndices` turns
patterns into the indices `SetPerturbedLayers` takes (empty: `mid`). The
perturbed pass sets `TNNetScaledDotProductAttention.SkipAttentionScores` on
the attention heads of those self-attentions; it allocates nothing and does
not rebuild the UNet. Tests on the pico folder: `--pag-scale 0` bit-identical
to no PAG (`TestStableDiffusionPAGScaleZero`); the perturbed UNet
bit-identical to a forward that writes each head's V rows by hand
(`TestStableDiffusionPAGIdentityReference`); diffusers float64
`StableDiffusionPAGPipeline` with DDIM, with CFG on `mid`, without CFG on two
patterns, and with an adaptive scale on every self-attention, step latents
within 2.2e-5 and images within 1.6e-6 (`TestStableDiffusionPAGParity`);
DeepCache (`TestStableDiffusionPAGDeepCache`); OpenCL against the CPU < 1e-4
(`TestStableDiffusionPAGOpenCL`). A real checkpoint has not been run with PAG
yet.

## Self-Attention Guidance (`--sag-scale`)

Self-Attention Guidance (SAG; Hong et al. 2023, arXiv:2210.00939) blurs the
regions the UNet attends to and steers each step away from what the UNet
predicts for the blurred latents. Per step, after the CFG passes:

1. The mask comes from the self-attention (`attn1`) of the mid block in the
   unconditional pass (the conditional pass without CFG). The attention each
   key receives, averaged over the heads and summed over the queries, is
   compared with 1; the keys above 1 form the mask on the mid-block grid. The
   mask is nearest-upsampled to the latent size.
2. x0 and eps are predicted from that pass's noise (epsilon or
   `v_prediction`, as the scheduler config says) at the step's
   `alphas_cumprod`. x0 is Gaussian-blurred (kernel 9, sigma 1, reflect
   padding) inside the mask and re-noised to the step's timestep with the
   same eps.
3. One more UNet pass runs on the degraded latents, with the negative prompt
   (the prompt without CFG). The step's noise gets
   `sag * (source - degraded)`, where source is the unconditional
   prediction (the conditional one without CFG).

This is diffusers' `StableDiffusionSAGPipeline`; its default `sag_scale` is
0.75.

```
StableDiffusion --model ~/models/stable-diffusion-v1-5 --steps 25 \
  --sag-scale 0.75 --prompt "a watercolor painting of a lighthouse" --gpu
```

diffusers indexes `alphas_cumprod` with integer timesteps only. Here the
fractional timesteps of the sigma schedules (Euler / Euler a with linspace
spacing, Karras sigmas) read `alpha_bar = 1 / (1 + sigma^2)` of the
interpolated sigma. The pipeline keeps the latents in the model-input scale,
so SAG's x0 and re-noise formulas hold for every sampler. SAG combines with
PAG (the SAG term is added after the PAG mix), img2img, inpainting,
depth2img, ControlNet, T2I-Adapter, IP-Adapter, LoRA, FreeU and DeepCache.
Under DeepCache the degraded pass keeps its own deep features, and the steps
that skip the mid block reuse the last mask. SAG does not combine with
`--edit-image`.

`TStableDiffusionPipeline.SAGScale` (0 = off) is the API;
`SelfAttentionMask` holds the last mask (mid-block grid, 0/1). The pieces are
reusable: `SDSelfAttentionGuidanceMask`, `SDSelfAttentionDegrade`,
`TNNetDiffusionScheduler.AlphaBarRootsAt` / `SplitPrediction`, and the
torchvision-style `GaussianBlurImage(Source, Dest, Sigma, KernelSize)` in
`neuralimageresize.pas`. During the denoise loop the mid `attn1` heads set
`KeepAttentionWeights`, so on OpenCL those heads skip the fused flash kernel.
Tests on the pico folder: `--sag-scale 0` (and < 0) bit-identical to no SAG
(`TestStableDiffusionSAGScaleZero`); DeepCache (`TestStableDiffusionSAGDeepCache`);
diffusers float64 `StableDiffusionSAGPipeline` with DDIM, with and without
CFG, the last mask bit for bit and sampled final latents within 5e-5 (about
2e-6 measured; `TestStableDiffusionSAGParity`); a 32x16 image, whose mask is
the 8x4 mid-block grid (`TestStableDiffusionSAGNonSquareMask`); the blur taps
against torchvision's kernel (`TestGaussianBlurImageTorchvisionKernel`). The
reference numbers come from an ad hoc diffusers run and are hardcoded in the
tests, because the committed fixture generator has no SAG mode. A real
checkpoint has not been run with SAG yet.

## Usage

```
StableDiffusion [--model DIR] [options]
  --model DIR       diffusers StableDiffusionPipeline folder (default: the pico fixture)
  --prompt TEXT     prompt (default "a photo of an astronaut riding a horse on mars")
  --negative TEXT   negative prompt (default empty)
                    Both take A1111 weights: (word:1.3), (word) = x1.1, [word] = /1.1,
                    \( \) \[ \] for literal brackets; prompts longer than 75 tokens
                    are encoded in chunks of 75 (lpw_stable_diffusion);
                    a standalone BREAK starts a new chunk (A1111)
  --no-prompt-weighting  read brackets and BREAK in the prompts as plain text
  --max-prompt-chunks N  keep at most N chunks of 75 prompt tokens (default 3;
                    0: no limit); longer prompts are cut with a notice
  --clip-skip N     skip the last N text encoder layers (diffusers clip_skip;
                    A1111 "Clip skip" N+1; default 0)
  --steps N         denoising steps (default 25; 4 with lcm)
  --guidance G      classifier-free guidance scale (default 7.5, 1.0 with lcm;
                    <= 1 turns it off); an LCM-distilled UNet embeds it
                    (default 8.5)
  --width N         image width (default: the checkpoint's, 512 for SD 1.5,
                    768 for SD 2.1)
  --height N        image height (default: the checkpoint's)
  --size N          sets both --width and --height
  --seed N          noise seed (default 0)
  --sampler NAME    auto (default: from the scheduler config), ddim, pndm,
                    euler, euler-a, dpm++2m or lcm (Latent Consistency; an
                    LCM-LoRA via --lora or an LCM-distilled UNet, 2-8 steps)
  --karras-sigmas   Karras sigmas for euler, euler-a and dpm++2m (diffusers
                    use_karras_sigmas); --no-karras-sigmas turns them off
                    (default: the scheduler config's)
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
                    runs the last int(steps * S) steps (lcm: all steps over the
                    first int(50 * S) LCM timesteps)
  --depth-image PNG depth-to-image: the depth map of --init-image (luma,
                    brighter = nearer; any size) for a 5-channel depth UNet
                    folder, e.g. stabilityai/stable-diffusion-2-depth; without
                    it the folder's depth_estimator/ estimates the depth
  --edit-image PNG  InstructPix2Pix: edit this image as the prompt says (needs an
                    8-channel UNet folder, e.g. timbrooks/instruct-pix2pix); the
                    output takes its size (rounded down to the VAE factor)
  --image-guidance S how closely --edit-image is kept (default 1.5; < 1 turns all
                    guidance off)
  --upscale-image PNG x4 upscaler: upscale this image (needs a 7-channel UNet
                    folder, e.g. stabilityai/stable-diffusion-x4-upscaler); the
                    output is VAE factor x its size (sides rounded down to the
                    factor); default 75 steps, guidance 9.0
  --noise-level N   noise added to --upscale-image, 0..max_noise_level (350;
                    default 20); more noise lets the UNet invent more detail
  --freeu S1,S2,B1,B2 FreeU (diffusers enable_freeu): skip scales S1, S2 and
                    backbone scales B1, B2 of the first two up blocks; presets
                    sd15 (0.9,0.2,1.5,1.6) and sd21 (0.9,0.2,1.4,1.6)
  --deepcache N     DeepCache: the whole UNet every N-th step, only its shallow
                    branch in between (default 1 = off; DeepCache's SD example uses 3)
  --deepcache-branch B shallow branch size, 0..layers_per_block: B resnets of
                    down block 0 and B + 1 of the last up block run (default
                    layers_per_block: both blocks whole; 0 is the fastest)
  --pag-scale S     Perturbed-Attention Guidance scale (default 0 = off; diffusers
                    uses 3.0): one more UNet pass per step
  --pag-layers L1,L2 the self-attentions PAG perturbs, diffusers pag_applied_layers
                    patterns (regular expressions over names such as
                    mid_block.attentions.0.transformer_blocks.0.attn1; default mid)
  --pag-adaptive-scale A lowers the PAG scale by A per timestep below 1000
                    (diffusers pag_adaptive_scale, default 0)
  --sag-scale S     Self-Attention Guidance scale (default 0 = off; diffusers
                    uses 0.75): one more UNet pass per step
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

* Stable Diffusion 1.x and 2.x are covered by pico tests only (parity against
  diffusers in `TestStableDiffusionPipelineParity` for DDIM and
  `TestStableDiffusionPipelineSamplers` for PNDM, Euler and DPM-Solver++ 2M,
  `TestStableDiffusion2*` for SD 2.x; Euler a against diffusers with replayed
  noise in `TestNeuralDiffusion`). No real checkpoint has been run yet.
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
* The noise (the initial latents, the Euler a and LCM step noise and the
  img2img VAE sample) comes from the Free Pascal RNG, so a seed does not reproduce a diffusers image.
* Long prompts split at every 75th token, as lpw_stable_diffusion does;
  AUTOMATIC1111 also moves a split back to a nearby comma and reads `BREAK`.
  The weighted states are rescaled to the mean of the whole prompt (lpw), not
  of each chunk (A1111). On SD 2.x (pad token `!`) each chunk is
  `<|startoftext|>` ids `<|endoftext|>` then pads, as A1111 and the diffusers
  pipeline lay out one chunk; lpw puts a single `<|endoftext|>` at the end of
  each chunk instead, so its pad rows differ there.
* Every UNet pass recomputes the cross-attention K/V of the text states,
  which do not change across steps. This costs about 0.4% of a 512x512 SD 1.5
  pass, so the K/V are not cached.

## Build

```
lazbuild StableDiffusion.lpi
```
