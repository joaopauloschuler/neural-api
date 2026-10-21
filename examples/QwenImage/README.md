# QwenImage — text-to-image and image editing with Qwen-Image-2.1

`QwenImage` turns a text prompt into an RGBA PNG image with the
Qwen-Image-2.1 checkpoint, or edits up to 10 condition images by an
instruction (see [Image editing](#image-editing)). The whole pipeline is Pascal code in this library
(`TQwenImage21Pipeline` in `neural/neuralpretrained.pas`); no Python runs at
generation time.

```
prompt
  -> processor/tokenizer.json (Qwen chat template, system prompt dropped)
  -> Qwen3-VL text encoder (text part: 36-layer Qwen3 decoder)
  -> MMDiT transformer prefix pass (the prompt's K/V, computed once)
  -> N flow-matching steps (Euler or DPM-Solver++(2M)) over the image tokens
  -> VAE decoder, tiled
  -> RGBA PNG
```

The pipeline follows the diffusers `QwenImage21Pipeline`:

- **Text encoder** — the text part of Qwen3-VL-8B. The transformer reads the
  last decoder layer's output before the final RMSNorm.
- **Transformer** — the 7B single-stream MMDiT (32 blocks). The prompt
  tokens run once through a prefix pass; each denoising step then runs only
  the image tokens against the cached prompt K/V.
- **Scheduler** — `TNNetFlowMatchEulerScheduler` (flow-matching Euler with
  the dynamic exponential shift and `shift_terminal`, or DPM-Solver++(2M)
  with `--sampler dpm2m`; see [Samplers](#samplers---sampler-eulerdpm2m)).
- **VAE decoder** — `TQwenImage21VaeDecoder`, decoding the 64-channel latents
  to RGBA at 16 times the latent width and height, in overlapping tiles.

Weights load as int8 by default, or as int4 (`--int4`) or FP32 (`--fp32`).
With an OpenCL build (the default), the transformer step pass (not with
`--fp32`) and the VAE decode run on the GPU; the text encoder and the
transformer prefix pass always run on the CPU.
`--cpu` runs everything on the CPU, and the program falls back to the CPU by
itself when it finds no usable OpenCL device.

`--image FILE` (repeatable) gives condition images to edit by the prompt
(see [Image editing](#image-editing)). `--init-image FILE` starts from an
init image instead of pure noise (SDEdit img2img, see
[img2img](#img2img-sdedit)): it re-styles the image. Adding `--mask FILE`
regenerates only the mask's white region and keeps the rest of the init
image (see [Inpainting](#inpainting)); `--outpaint L,T,R,B` instead extends the
init image beyond its borders (see [Outpainting](#outpainting)).

**Flag change:** SDEdit img2img used to be `--image FILE` and `/image FILE`.
It is now `--init-image FILE` and `/init-image FILE|off`; `--image` and
`/image` add edit condition images. A run that still passes
`--image photo.jpg --strength 0.6` now edits (and prints that `--strength`
has no effect without `--init-image`).

## Getting the model

The checkpoint is the Hugging Face repository
[`Qwen/Qwen-Image-2.1`](https://huggingface.co/Qwen/Qwen-Image-2.1)
(see its model card for the license). Download the whole repository, for example with
git (git-lfs installed):

```
git clone https://huggingface.co/Qwen/Qwen-Image-2.1
```

`--model` takes that folder as it is, with no conversion step:

```
Qwen-Image-2.1/
  model_index.json
  processor/      tokenizer.json (the prompt tokenizer),
                  preprocessor_config.json (the vision tower, editing only)
  text_encoder/   Qwen3-VL-8B config.json + safetensors
  transformer/    config.json + safetensors (7B, BF16)
  vae/            config.json + safetensors
  scheduler/      scheduler_config.json
```

`TQwenImage21Pipeline.Create` reads `model_index.json`,
`scheduler/scheduler_config.json` and `transformer/config.json`; each stage
loads its own component folder when it starts.

## Building

```
cd examples/QwenImage
lazbuild -B QwenImage.lpi
```

`lazbuild` writes the binary to `bin/x86_64-linux/bin/QwenImage` at the
**repository root** (not inside `examples/QwenImage`). The default build mode
compiles with `-dAVX2 -dRelease -dOpenCL`.

## Usage

`QwenImage` has two modes: a one-shot run and a prompt loop (REPL).

### One-shot

`-p "prompt"` generates one image, writes `--output` and exits:

```
bin/x86_64-linux/bin/QwenImage --model Qwen-Image-2.1 --int4 \
  -p "A red fox sitting in fresh snow, morning light" --output fox.png
```

In this mode each component loads when its stage starts and is freed when the
stage ends, so only one of the text encoder, the transformer and the VAE is in
memory at a time. `--repeat N` with N > 1 makes N images from one prompt and
keeps the three components loaded, as the REPL does.

`--token-ids` replaces `-p` with already-tokenized input: the comma-separated
ids of the whole templated prompt, plus `--drop-count N` for the number of
leading system-prompt tokens to drop. It needs no `processor/` folder. The
repository's pico checkpoint runs this way in under a second on the CPU (its
weights are random, so the image is noise):

```
bin/x86_64-linux/bin/QwenImage --model tests/fixtures/tiny_qwenimage21 \
  --token-ids 11,48,85,122,159,196,233,270,7,44,81,118,155,192 --drop-count 5 \
  --width 64 --height 64 --steps 2 --cpu --output p.png
```

### REPL

Without `-p` or `--token-ids`, `QwenImage` loads the text encoder, the
transformer and the VAE once, keeps them in memory, prints the resident
memory, and then reads one prompt per line from stdin. Each prompt produces
one image. The end of the input ends the session, so a piped file is a batch:

```
bin/x86_64-linux/bin/QwenImage --model Qwen-Image-2.1 --int4 --steps 20 \
  --output out.png < prompts.txt
```

The REPL numbers the images from the `--output` base: `out.png` gives
`out_0001.png`, `out_0002.png`, ... It skips numbers whose file already
exists, so it never overwrites an image. A base without an extension gets
`.png`.

Lines that start with `/` are commands; they apply to the prompts that
follow:

| Command | Effect |
| --- | --- |
| `/size WxH` | image size, e.g. `/size 1024x768` (each side rounded down to a multiple of 32, 32..8192) |
| `/steps N` | sampler steps |
| `/seed N` | seed of the next image; without it the seed grows by one per image |
| `/tile SIZE[,STRIDE]` | VAE tile, as `--vae-tile` |
| `/image FILE` | add a condition image (edit, as `--image`) for the next prompts, up to 10 |
| `/images` | list the condition images; `/images clear` removes them all (back to text-to-image) and frees the vision tower and the VAE encoder |
| `/init-image FILE` | init image for the next prompts (img2img, as `--init-image`); `/init-image off` clears it |
| `/mask FILE` | inpaint mask for the init image (as `--mask`); `/mask off` clears it |
| `/mask-blur R` | mask blur sigma in pixels (as `--mask-blur`); reloads the current mask with it |
| `/padding-mask-crop N` | inpaint only the mask's box plus N pixels (as `--padding-mask-crop`); `/padding-mask-crop off` clears it |
| `/outpaint L,T,R,B [OVERLAP]` | outpaint the init image (as `--outpaint` and `--outpaint-overlap`); sets strength 1 unless `--strength` or `/strength` set it; `/outpaint off` goes back to img2img |
| `/strength S` | img2img strength for the next images, in (0, 1], as `--strength` |
| `/upscale FILE` | upscale the next images with this Real-ESRGAN checkpoint, as `--upscale`; `/upscale` alone shows the current one, `/upscale off` stops upscaling |
| `/repeat N PROMPT` | N images of PROMPT (1..1000) with consecutive seeds, each to the next numbered file; the text encoder runs once. Everything after N is the prompt. An image that fails is skipped and the rest of the batch runs |
| `/fbc X\|off` | First-Block Cache threshold for the next images, as `--fbc-threshold`; `/fbc` alone shows it, `/fbc off` (or `/fbc 0`) turns it off |
| `/negative-prompt TEXT\|off` | negative prompt for the next prompts, as `--negative-prompt`; `/negative-prompt` alone shows the guidance setting, `/negative-prompt off` clears it |
| `/guidance-scale W` | guidance scale for the next prompts, as `--guidance-scale` |
| `/panorama WxH\|off` | panorama canvas for the next prompts, as `--panorama` (windows of `/size`, stride and wrap from the flags); `/panorama` alone shows it, `/panorama off` turns it off |
| `/profile on\|off` | `--profile` from the next image |
| `/stats on\|off` | `--stats` from the next image |
| `/quit` | end the session |

A prompt file can mix both:

```
/size 1024x1024
/seed 7
A red fox sitting in fresh snow, morning light
A lighthouse on a cliff at dusk, oil painting
/size 768x1024
/steps 30
A street market in the rain, photograph
```

The same seed, size and step count give the same image again: the initial
noise comes from the FPC random generator. For the same reason an image does
not equal the diffusers image for the same seed.

### Image editing

`--image FILE` gives a condition image; repeat it for up to 10 images. The
prompt is the instruction, and it can refer to the images by their order:
the first `--image` is image 1, the next image 2, and so on.

```
bin/x86_64-linux/bin/QwenImage --model Qwen-Image-2.1 --image room.jpg \
  -p "Paint the walls dark green" --output room_green.png

bin/x86_64-linux/bin/QwenImage --model Qwen-Image-2.1 --image person.png \
  --image jacket.png -p "Image 1 wears the jacket from image 2" --output out.png
```

In the REPL, `/image FILE` appends a condition image, `/images` lists them
and `/images clear` removes them all. The images apply to every prompt that
follows. An edit and an init image (`--init-image`, img2img) are exclusive.

This is the diffusers `QwenImage21Pipeline` with `image=[...]` (guidance
only with `--negative-prompt`, see
[Guidance](#guidance---negative-prompt-text---guidance-scale-w)):

1. Each image is resized to its own aspect ratio at the area of `--width` x
   `--height` (default 1024x1024), sides multiples of 32 (diffusers'
   `calculate_dimensions` with `output_resolution`; the resize is
   `ResizeImageLanczos`, as for img2img below).
2. The Qwen3-VL vision tower (`TQwen3VLVisionTower`, CPU) encodes the
   resized image composited over white; the VAE encoder
   (`TQwenImage21VaeEncoder`, CPU, untiled) encodes the same resized image
   with its alpha channel.
3. The text encoder reads the edit template (`<image1><|vision_start|>
   <|image_pad|><|vision_end|> <image2>...` then the prompt), with each
   `<|image_pad|>` expanded to the image's vision rows.
4. The transformer prefix pass holds the prompt and the condition images'
   latents; the denoising steps run only the target image.
5. The output takes the aspect ratio of the **last** condition image at the
   `--width` x `--height` area.

There is no mask input and no strength: a mask, a sketch or an image with
circles drawn on it is just one more condition image, and the prompt says
how to use it ("replace the area circled in red in image 1 with ...").

The condition images are encoded once and reused: by every image of a
`--repeat` batch or `/repeat`, and by later REPL prompts until `/image`,
`/images clear` or `/size` (when the area changes) changes them. The prompt
itself is still encoded for every new prompt line. In keep-loaded mode (the
REPL, `--repeat N` > 1) the first edit loads the vision tower and the VAE
encoder and keeps them beside the other components until `/images clear`
frees them; a one-shot run loads and frees each in turn. The stage table shows "vision encode" and
"VAE encode" rows for the image that encoded them.

Requirements and caveats:

- The vision tower needs `processor/preprocessor_config.json` (or
  `processor/processor_config.json`) of the checkpoint for its mean, std and
  pixel limits. `TQwenImage21Pipeline.CheckEditSupport` checks it (and the
  vision config) when `--image` is given or on the first `/image`, before
  any component loads. Text-to-image does not need it.
- `EncodeConditionImages` refuses an image whose resized size is outside the
  processor's `shortest_edge`..`longest_edge` pixel range, before it frees
  the previous images or loads anything (the Qwen3-VL processor would
  resize it a second time; this program does not). Change `--width` x
  `--height` (or `/size`) then.
- Memory: the vision tower and the VAE encoder run untiled on the CPU, so
  their activations grow with the area. The transformer prefix grows by one
  token per 16x16 pixels of every condition image: one 1024x1024 image adds
  4096 prefix tokens (task C4 in `tasklist.md` has the estimates; the prefix
  pass runs on the CPU). None of this has been measured on the real
  checkpoint. The program keeps every condition image as read in memory
  (16 bytes per pixel: a 12-megapixel photo takes 192 MB).
- The JPEG EXIF orientation is not applied (diffusers' `load_image`
  applies `exif_transpose`), so a rotated phone photo stays as stored.
- `--token-ids` with `--image` takes the template ids with one
  `<|image_pad|>` id per image; the pipeline expands them.

### img2img (SDEdit)

`--init-image FILE` (or `/init-image FILE` in the REPL) makes each image
start from an init image instead of pure noise:

```
bin/x86_64-linux/bin/QwenImage --model Qwen-Image-2.1 --init-image photo.jpg \
  --strength 0.6 -p "a watercolor painting of a harbor" --output harbor.png
```

1. The VAE encoder (`TQwenImage21VaeEncoder`, on the CPU, untiled) encodes
   the init image to latents: the posterior mean, as the 2.1 pipeline
   encodes its condition images. (The v1 img2img pipeline samples the
   posterior instead.)
2. `--strength S` cuts the schedule: of N steps, the first
   `t_start = int(N - N * S)` are skipped. 18 steps at 0.6 run the last 11;
   strength 1 runs all of them from pure noise. A strength that leaves no
   step is an error.
3. Each image mixes the seed's noise into the init latents at the sigma
   where the schedule is cut (`sigma * noise + (1 - sigma) * latents`) and
   runs only the remaining steps.

Steps 2 and 3 are the start-step and noise schedule of the Qwen-Image v1
diffusers pipeline (`QwenImageImg2ImgPipeline`), applied to the 2.1
schedule; diffusers ships no img2img pipeline for 2.1.
The prompt describes the whole picture you want: img2img keeps the init
image's layout and colours to a degree set by the strength, and re-styles
it. It does not follow edit instructions such as "remove the car".

The output keeps the init image's aspect ratio at the area of `--width` x
`--height` (default 1024x1024), each side a multiple of 32 — the
`calculate_dimensions` rule of the diffusers 2.1 pipeline. A 4000x3000
photo gives 1184x896 at the default area. The program resizes the init image
to that size with PIL's Lanczos coefficients in floating point
(`ResizeImageLanczos` in `neural/neuralimageresize.pas`). That equals PIL on
float ('F' mode) images. On the 8-bit images diffusers resizes, PIL uses
fixed-point coefficients, an integer alpha premultiply, and rounds and clips
to 8 bits after each pass, so the results differ by a few levels of 255. An
image with an alpha channel is resized with premultiplied alpha, as PIL
does, and the VAE encodes all four channels; an image without alpha is
opaque. The JPEG EXIF orientation is not applied (diffusers' `load_image`
applies `exif_transpose`), so a rotated phone photo stays as stored.

The init image is encoded once and reused: by every image of a `--repeat`
batch or `/repeat`, and by later REPL prompts until `/init-image`, `/size`
(when the output size changes) or `/init-image off`. The noise still changes with the
seed. In keep-loaded mode (the REPL, `--repeat N` > 1) the first encode
loads the VAE encoder and keeps it beside the other components; a one-shot
run loads it and frees it after the encode.

### Inpainting

`--mask FILE` with `--init-image FILE` (or `/mask FILE` in the REPL while an
init image is set) regenerates only the white region of the mask and keeps
the black region of the init image:

```
bin/x86_64-linux/bin/QwenImage --model Qwen-Image-2.1 --init-image room.png \
  --mask room_mask.png --strength 1 -p "a red armchair by the window" \
  --output armchair.png
```

This is the blended (no-retrain) mode of the diffusers Qwen-Image v1
`QwenImageInpaintPipeline` on top of the img2img path above; diffusers ships
no inpaint pipeline for 2.1.

1. The mask is read as grayscale (PIL's `L` weights, alpha ignored),
   blurred at its own size when `--mask-blur R` is given (see below),
   stretched to the output size (the init image's), binarized at 50% and
   reduced to the latent grid by nearest sampling: one value per 16x16-pixel
   cell, taken from the cell's top-left pixel. In 2.1 a latent cell is one
   transformer token, so a cell is either regenerated or kept as a whole.
2. The init image is noised and the schedule cut by `--strength` exactly as
   for img2img. `--strength 1` fills the hole from pure noise; lower values
   keep the init image's layout and colours inside the hole too.
3. After every step the kept cells are reset to the init latents noised
   with the same noise to the next step's sigma, and to the clean init
   latents after the last step, so the kept latents equal the init latents
   exactly.

The VAE decoder mixes neighbouring latents, so pixels next to the hole (and,
through its attention, a little everywhere) can change slightly even though
the kept latents are exact. A mask of another aspect ratio than the init
image is stretched, not cropped. The mask is encoded once per output size;
`/mask off` (or `/init-image off`) goes back to img2img (or text-to-image).

`--mask-blur R` (or `/mask-blur R`) blurs the mask with a Gaussian of sigma
`R` mask pixels before the 50% threshold, as diffusers'
`mask_processor.blur(mask, blur_factor=R)` does before the mask enters the
pipeline. `GaussianBlurImage` (neural/neuralimageresize.pas) runs a true
separable Gaussian with clamped edges; PIL's `GaussianBlur` approximates the
same Gaussian with three extended box blurs, so edge values can differ
slightly. The threshold then rounds the mask's corners and erases specks and
thin lines narrower than about `R` pixels. Default 0 (no blur).

#### Mask crop (`--padding-mask-crop N`)

Without it, the whole init image is resized to the output size, so a small
hole in a large photo gets only a few latent cells. `--padding-mask-crop N`
(or `/padding-mask-crop N`; `/padding-mask-crop off` clears it) does what
diffusers' `padding_mask_crop` does:

1. The mask is stretched to the init image's size (when it has another
   size) and its bounding box (pixels whose PIL `L` value is above 0) is
   grown by `N` pixels on each side, then grown to the output aspect ratio
   around its centre and shifted back inside the image
   (`GetMaskCropRegion`, diffusers `get_crop_region`).
2. That region of the init image and of the mask is resized to the output
   size (`ResizeImageAndFill`, diffusers `resize_mode="fill"`) and inpainted
   as above.
3. The decoded image is resized back to the region's size
   (`ResizeImageAndCrop`, diffusers `resize_mode="crop"`) and blended into
   the init image by the mask, `init + mask/255 * (generated - init)`
   (`PasteInpaintCrop`, diffusers `apply_overlay`). Outside the region the
   init image is kept exactly.

The output keeps the init image's size, not the output size. The mask used
for the blend is the one after `--mask-blur`, so a blur gives a soft seam.

```
bin/x86_64-linux/bin/QwenImage --model Qwen-Image-2.1 --init-image photo.png \
  --mask hole.png --padding-mask-crop 32 --strength 1 -p "a red armchair" \
  --output armchair.png
```

Differences from diffusers: the resizes run in float, not on 8-bit PIL
images; where the region's aspect differs from the output's (a rounding
pixel, or a region clamped by the image border) the fill repeats the edge
pixels, while PIL samples the edge row with its Lanczos filter; the blend
skips PIL's premultiplied-alpha rounding. All helpers live in
neural/neuralimageresize.pas.

Status: the inpaint path matches diffusers on the pico checkpoint
(`TestQwenImage21InpaintParity`); a real 1024x1024 inpaint on the GPU box
is still pending (user run). The mask crop is tested on its helpers (hand
values checked against diffusers) and a pico smoke run, not against a
diffusers pipeline output.

### Outpainting

`--outpaint L,T,R,B` with `--init-image FILE` (or `/outpaint L,T,R,B` in the
REPL) extends the init image by L, T, R and B pixels on the left, top, right
and bottom, and generates the new border:

```
bin/x86_64-linux/bin/QwenImage --model Qwen-Image-2.1 --init-image beach.png \
  --outpaint 256,0,256,0 -p "a wide sandy beach at sunset" --output wide.png
```

1. `BuildOutpaintCanvas` (neural/neuralimageresize.pas) places the init image
   on a larger canvas and fills the new border by repeating the edge pixels,
   so the VAE does not see a black frame. Its mask covers the border plus
   `--outpaint-overlap N` pixels (default 16, one latent cell) into the image
   on each extended side, so the seam is regenerated too.
2. The canvas grows to multiples of 32 (the pipeline's side multiple): the
   extra pixels go to the right (to the left when only the left side is
   extended), and likewise to the bottom or top. The output is generated at
   that canvas size; `--width`/`--height` do not apply, so the init image
   keeps its own scale.
3. The [inpaint](#inpainting) path regenerates the masked cells,
   at `--strength 1` unless `--strength` is given. `--mask-blur R` blurs the
   outpaint mask the same way.
4. After the decode, the program crops the canvas to exactly the init image
   plus L, T, R, B, and pastes the original pixels back over the result
   outside the overlap band (`PasteOutpaintOriginal`, diffusers'
   `apply_overlay` behaviour), so the kept region is exact rather than VAE
   round-tripped.

`--outpaint` and `--mask` are exclusive. The canvas sides must stay within
32..8192 pixels. `TestQwenImage21OutpaintKeepsOriginal` runs these steps on
the pico checkpoint; a real outpaint on the GPU box is still pending (user
run).

### LoRA adapters

`--lora FILE[:SCALE]` merges a LoRA `.safetensors` file into the transformer
weights while `TQwenImage21Transformer.Create` loads them. Repeat the option to
stack adapters; each one adds its own delta:

```
bin/x86_64-linux/bin/QwenImage --model Qwen-Image-2.1 \
  --lora watercolor.safetensors:0.8 --lora character.safetensors \
  -p "a fox in a forest"
```

Each adapted weight matrix W loads as W + SCALE * s * (B @ A), in FP32,
before `--int8` / `--int4` quantize it. SCALE defaults to 1. The adapter scale
s is set as diffusers' Qwen-Image loader sets it:

| Key layout | Example key | s |
| --- | --- | --- |
| diffusers | `transformer.transformer_blocks.0.attn.to_q.lora_A.weight` | 1 (a `.alpha` beside it is ignored) |
| ComfyUI | `diffusion_model.transformer_blocks.0.attn.to_q.lora_A.weight` | 1 (the same) |
| kohya | `lora_unet_transformer_blocks_0_attn_to_q.lora_down.weight` | `.alpha` / rank; 1 without `.alpha` |

A kohya file without `.alpha` loads with s = 1, the PEFT default
(diffusers' Qwen-Image converter raises on that file instead). The loader
maps a kohya name such as `transformer_blocks_0_attn_to_q` back to its
dotted tensor name through the checkpoint's own tensor names.

The program refuses an adapter, and prints why, when:

- a module has no tensor in the checkpoint. An adapter trained for the
  earlier Qwen-Image releases targets modules such as `txt_mlp` and
  `attn.add_q_proj`, which Qwen-Image-2.1 does not have;
- a factor is 4-D (a convolution LoRA);
- a tensor is not part of a LoRA pair (DoRA magnitudes, LoRA biases).

**Step-distilled adapters.** The "Lightning" LoRAs that cut 40 steps to
4-8 (lightx2v/Qwen-Image-Lightning) target the earlier Qwen-Image and
Qwen-Image-Edit releases. We know of no step-distilled LoRA for
Qwen-Image-2.1, and none has been tested with this program, so
`--lora` does not promise fewer steps.

The adapters are checked only against an oracle computed in the tests
(`TestLoRAMergeImage*`, `TestQwenImage21TransformerLoRA`); no real
community LoRA has run through this program yet.

### Upscaling (`--upscale FILE`)

`--upscale FILE` upscales every image with a Real-ESRGAN / ESRGAN RRDBNet
checkpoint after the VAE decode and before the PNG save. With
`RealESRGAN_x4plus.pth`, a 512x512 generation is written as 2048x2048, and the
transformer only runs on the 1024 image tokens of 512x512:

```
bin/x86_64-linux/bin/QwenImage --model Qwen-Image-2.1 --int4 \
  --width 512 --height 512 --upscale RealESRGAN_x4plus.pth -p "a fox in a forest"
```

The upscaler is `TRRDBNetUpscaler` in `neural/neuralpretrained.pas`, the same
code that `examples/RealESRGANUpscale` runs. It takes the `.pth` as
published (the `params_ema` wrapper is unwrapped) or a `.safetensors`. It
reads `config.json` next to the file when there is one, and otherwise reads
the config from the tensor shapes, so x2plus works too. The image goes in as
RGB in [0,1] and comes out clamped to [0,1], as basicsr's `RealESRGANer`
does. The alpha channel of the VAE output is resized bilinearly, not run
through the net. Upscaling runs after the inpaint-crop paste and the outpaint
crop, so it upscales the finished image.

Memory: the upscaler net is built for each image after the VAE decode, so in
one-shot runs it is the only component in memory, and it is freed before the
PNG save. In the REPL and with `--repeat`, the text encoder, the transformer
and the VAE decoder stay loaded beside it while it runs. The program does not
keep it between images: reading the weights again (67 MB for x4plus) costs
far less than the upscale itself, while a kept net would also keep its tile
buffers.

The net runs tile by tile: `--upscale-tile N` pixels (default 32), plus 10
pixels of context on each side, with 8-pixel feathered seams. On the CPU
path a tile holds about 0.3 GB plus 2.7 MB per pixel of (N + 20)^2,
measured on this machine with the x4plus layout. That is about 3.8 GB at 16,
7.6 GB at 32 and 19 GB at 64. Most of it is each convolution's own
input copy and im2col buffer, which `TNNet.ShareHostOutputsByLiveness` does
not share. A larger tile has less context overhead: (N + 20)^2 / N^2 is 2.6x
at 32 and 1.7x at 64. `--upscale-tile 0` runs the whole image as one tile.
When `--gpu` armed the pipeline, the upscaler runs on the same OpenCL device.
The time and memory of the real x4plus checkpoint have not been measured on
a GPU yet.

With `--stats`, the stage table gets an `upscale` row with the device, the
output size and the net input size.

### First-Block Cache (`--fbc-threshold X`)

`--fbc-threshold X` (or `/fbc X` in the REPL) skips most of the transformer
on steps where little changes. It is a port of diffusers'
`apply_first_block_cache` (`hooks/first_block_cache.py`). Each step runs
`img_in` and transformer block 0, then compares block 0's residual (its
output minus its input) with the residual of the last step that ran every
block:

    change = mean |r - r_prev| / mean |r_prev|

When `change` is at most X, the step skips blocks 1 to N-1. It adds the
residual those blocks produced on the last full step (last block output
minus block 0 output) to block 0's output, and runs `norm_out` and
`proj_out` on the sum. When `change` is above X, the step runs every block
and refreshes both residuals. The first step of every image always runs
every block. A skipped step does not refresh `r_prev`, so the comparison
stays against the last full step and the skips cannot drift without bound.

```
bin/x86_64-linux/bin/QwenImage --model Qwen-Image-2.1 --fbc-threshold 0.1 \
  -p "a lighthouse on a cliff at dusk"
```

The trade-off:

- **Speed.** A skipped step costs `img_in`, one block and `norm_out` +
  `proj_out` instead of all N blocks, so the denoise time falls about in
  proportion to the skipped steps. With `--stats`, the `denoise` row shows
  how many steps skipped (`k with blocks 1..N-1 skipped (FBC)`). diffusers
  and ParaAttention report 1.5x to 2x per image at thresholds 0.05 to 0.12
  on other DiT models. Speed-ups for Qwen-Image-2.1 have not been measured
  yet; the pico checkpoint in the tests is too small to time.
- **Quality.** A skipped step reuses an older residual, so a higher
  threshold skips more steps and can lose fine detail or shift colours and
  composition. diffusers' default is 0.05. The quality at each threshold has
  not been checked on Qwen-Image-2.1 yet: compare the same seed with and
  without the option.
- **Memory.** Three host buffers of (image tokens x hidden) floats each,
  with 4096 image tokens at 1024x1024.
- **OpenCL.** The comparison and the residual add run on the CPU: every
  step downloads `img_in`'s and block 0's outputs, a full step also
  downloads the last block's output, and a skipped step uploads the sum.
  That is three transfers of (image tokens x hidden) floats per step, small
  next to a step's time; a device-side version is open in `tasklist.md`.

`0` (the default) turns it off: every step runs every block, exactly as
without the option. The pipeline property is
`TQwenImage21Pipeline.FirstBlockCacheThreshold`; the cache itself is
`TDiTFirstBlockCache` in `neural/neuralpretrained.pas`, written so other
DiT pipelines can reuse it.

### Guidance (`--negative-prompt TEXT --guidance-scale W`)

Qwen-Image-2.1 is meant to run without classifier-free guidance, so it is
off by default. `--negative-prompt TEXT` with `--guidance-scale W` greater
than 1 turns it on, as diffusers' `QwenImage21Pipeline` does with
`negative_prompt` and `true_cfg_scale` (an empty negative prompt also
counts). Each step then runs the transformer twice on the same latents,
once with the prompt and once with the negative prompt, and mixes the two
velocities:

    v = v_neg + W * (v_prompt - v_neg)

This is the 2.1 pipeline's formula. The v1 `QwenImagePipeline` also
rescales the mix to the norm of `v_prompt` per token; the 2.1 pipeline does
not, and neither does this one.

```
bin/x86_64-linux/bin/QwenImage --model Qwen-Image-2.1 --guidance-scale 4 \
  --negative-prompt "blurry, low quality, text" \
  -p "a lighthouse on a cliff at dusk"
```

- **Prefix caches.** The text encoder encodes the negative prompt too (with
  the same condition images when editing). The transformer keeps one prefix
  K/V cache per prompt (`TQwenImage21Transformer.ActivePrefixCache`), so
  the two passes do not overwrite each other; the block weights are shared.
- **Cost.** Two transformer passes per step: the denoise takes about twice
  as long. Memory grows by the negative prompt's prefix K/V and one
  velocity buffer; the step activations are reused by both passes.
- **APG.** `--apg ETA,MOMENTUM,THRESHOLD` replaces the CFG mix with
  Adaptive Projected Guidance (`TNNetAdaptiveProjectedGuidance`, as in
  `examples/LatentTextToImage`; paper setting e.g. `0,-0.5,15`). It mixes
  the denoised predictions `x0 = x - sigma * v` of both passes, the
  paper's space, and converts the result back to a velocity;
  `--apg-velocity` mixes the raw velocities instead (what a diffusers guider
  does). `THRESHOLD` is a norm over the whole latent, so its scale differs
  from the Stable Diffusion values in the paper.
- **Other modes.** img2img, inpainting and outpainting guide each step
  before the sampler step and the mask blend, as diffusers' inpaint
  pipeline does. A panorama guides each window's prediction before the
  windows are averaged (diffusers' `StableDiffusionPanoramaPipeline` guides
  each view the same way); for the plain CFG mix this equals guiding the
  average, for APG each window keeps its own momentum. With
  `--fbc-threshold`, each pass keeps its own First-Block Cache (diffusers'
  `cache_context("cond")` / `"uncond"`), and the `--stats` row counts
  skipped passes of both. `--sampler dpm2m` still takes one sampler step
  per step, on the mixed velocity.

The quality on the real checkpoint has not been checked yet.

### Panoramas and large canvases (`--panorama WxH`)

`--panorama WxH` (or `/panorama WxH` in the REPL) generates a canvas larger
than the transformer pass with MultiDiffusion (Bar-Tal et al. 2023, the
method of diffusers' `StableDiffusionPanoramaPipeline`). The transformer
always runs on a window of `--width` x `--height` pixels (default
1024x1024). Each step:

1. cuts overlapping windows out of the canvas latents, every
   `--panorama-stride` pixels (default half a window side);
2. runs the transformer on each window, all with the one prompt prefix that
   `EncodePrefix` encoded at the start of the image;
3. averages the windows' velocities per latent cell (a cell covered by k
   windows gets the mean of the k velocities);
4. runs one sampler step on the whole canvas.

`--panorama-wrap` wraps the windows past the right edge around to the left
edge, so the left and right edges blend into each other: use it for
360-degree panoramas. The VAE decodes the canvas with its usual tiles.

```
bin/x86_64-linux/bin/QwenImage --model Qwen-Image-2.1 --panorama 4096x1024 \
  --panorama-wrap -p "a mountain lake at sunrise, wide panorama"
```

The trade-off:

- **Memory.** The transformer's activations and attention grow with the
  window, not the canvas: a 4096x1024 banner needs the memory of a
  1024x1024 image plus the canvas latents (small).
- **Time.** Each step runs the transformer once per window. 4096x1024 with
  1024x1024 windows every 512 pixels is 7 windows, so a step takes about 7
  times a 1024x1024 step. A larger stride runs fewer windows and blends
  less; a stride equal to the window side has no overlap and shows seams.
- **Composition.** Each window sees only its own region and the prompt,
  so the global composition is weaker than one pass over the whole image:
  objects can repeat along a wide canvas, and nothing ties distant regions
  together. Prompts that describe a scene that repeats well (landscapes,
  skylines) work best.
- **Schedule.** The sigma shift (`mu`) is computed from a window's token
  count, the size the transformer sees, not the canvas's.
- **First-Block Cache.** With `--fbc-threshold`, each window keeps its own
  cache and decides on its own; the memory is three (window tokens x hidden)
  buffers per window. The `--stats` row counts skipped window passes.

`--panorama` runs text-to-image only: not with `--init-image` (img2img,
inpainting, outpainting) or `--image` (editing). A canvas no larger than one
window runs exactly one pass, the same as without the option. The quality
and the time per step on the real checkpoint (2048x512, 4096x1024) have not
been measured yet. The pipeline properties are
`TQwenImage21Pipeline.PanoramaWindowWidth`, `PanoramaWindowHeight`,
`PanoramaStride` and `PanoramaCircular`; the window planner and the
averaging are `TMultiDiffusionWindows` in `neural/neuraldiffusion.pas`,
which takes any velocity callback, so other DiT pipelines can reuse it. It
plans its windows with `ComputeTileStarts`, the planner of the tiled VAE
decode.

## Options

`QwenImage --help` prints this list.

### Model and weights

| Option | Meaning | Default |
| --- | --- | --- |
| `--model DIR` | the checkpoint folder with `model_index.json` | required |
| `--int8` | transformer block weights in int8; text encoder weights in int8 | on |
| `--int4` | transformer block weights in Q4_0-style int4 (blocks of 32); text encoder weights in int8 | off |
| `--fp32` | transformer and text encoder weights in FP32 | off |
| `--int8-input` | int8 activations into the transformer's int8/int4 projections (not with `--fp32`) | off |
| `--lora FILE[:SCALE]` | merge a LoRA adapter into the transformer weights at load, times SCALE; repeatable ([LoRA adapters](#lora-adapters)) | none |

With `--fp32`, both the text encoder and the transformer load as FP32, the 7B
transformer runs a slow kernel, and the transformer step pass runs on the CPU
instead of OpenCL. The norms, the embeddings, the transformer's input, output
and timestep nets, and the VAE stay FP32 in every mode. Of `--int8`, `--int4`
and `--fp32`, the last one given wins.

### Prompt and output

| Option | Meaning | Default |
| --- | --- | --- |
| `-p TEXT` | one-shot: generate this prompt, write `--output`, exit | REPL |
| `--token-ids LIST` | one-shot from comma-separated token ids instead of `-p` | — |
| `--drop-count N` | leading system-prompt tokens to drop with `--token-ids` | 0 |
| `--output FILE` | image file; `.png` keeps the alpha channel. The REPL numbers it | `qwenimage.png` |

### Image and sampling

| Option | Meaning | Default |
| --- | --- | --- |
| `--width N`, `--height N` | pixels, rounded down to a multiple of 32 (32..8192) | 1024 |
| `--steps N` | sampler steps | 18 |
| `--sampler NAME` | `euler` or `dpm2m` (DPM-Solver++(2M)); see [Samplers](#samplers---sampler-eulerdpm2m) | `euler` |
| `--seed N` | seed of the initial noise | 42 |
| `--image FILE` | edit: a condition image (PNG, JPEG, ...); repeat for up to 10; see [Image editing](#image-editing) | — |
| `--init-image FILE` | img2img from this init image; see [img2img](#img2img-sdedit) | — |
| `--mask FILE` | with `--init-image`: inpaint, regenerate only the mask's white region; see [Inpainting](#inpainting) | — |
| `--mask-blur R` | blur the mask by a Gaussian of sigma R pixels before the 50% threshold (diffusers `mask_processor.blur`) | 0 |
| `--padding-mask-crop N` | with `--mask`: generate only the mask's bounding box plus N pixels (grown to the output aspect) at the output size and paste it back into the init image, which keeps its size; see [Mask crop](#mask-crop---padding-mask-crop-n) | off |
| `--outpaint L,T,R,B` | with `--init-image`: add L, T, R, B pixels on the left, top, right and bottom and generate them; see [Outpainting](#outpainting). Not with `--mask` | — |
| `--outpaint-overlap N` | pixels of the init image along each extended side that are regenerated too | 16 |
| `--strength S` | img2img strength in (0, 1]: how much of the schedule runs (1 = from pure noise) | 0.6 (1 with `--outpaint`) |
| `--upscale FILE` | upscale each image with this Real-ESRGAN / ESRGAN RRDBNet checkpoint before the save; see [Upscaling](#upscaling---upscale-file) | off |
| `--upscale-tile N` | upscaler tile in input pixels, 0 = the whole image; each tile adds 10 context pixels per side | 32 |
| `--fbc-threshold X` | First-Block Cache: skip transformer blocks 1..N-1 on steps whose block-0 residual changed by at most X; see [First-Block Cache](#first-block-cache---fbc-threshold-x) | 0 (off) |
| `--negative-prompt TEXT` | what the image should not show; used when `--guidance-scale` > 1; see [Guidance](#guidance---negative-prompt-text---guidance-scale-w) | none |
| `--guidance-scale W` | classifier-free guidance scale: `v_neg + W * (v_prompt - v_neg)`, two transformer passes per step; 1 = off | 1 (off) |
| `--apg ETA,MOMENTUM,THRESHOLD` | Adaptive Projected Guidance instead of the CFG mix, on the denoised x0 | off |
| `--apg-velocity` | apply `--apg` to the raw velocities instead of x0 | off |
| `--panorama WxH` | MultiDiffusion: a WxH canvas (rounded down to multiples of 32) denoised in overlapping `--width` x `--height` windows; see [Panoramas](#panoramas-and-large-canvases---panorama-wxh). Not with `--init-image` or `--image` | off |
| `--panorama-stride N` | window step in pixels, a multiple of 16 no larger than a window side | half a window side |
| `--panorama-wrap` | wrap the windows past the right edge to the left edge (360-degree panorama) | off |
| `--repeat N` | with `-p` or `--token-ids`: N images (1..1000) with seeds `--seed`, `--seed`+1, ...; the text encoder runs once. N > 1 keeps every component loaded, as the REPL does, and numbers the files like the REPL (`qwenimage_0001.png`, ...). An error ends the run | 1 |

An image of W x H pixels has (W/16) x (H/16) image tokens: 4096 at
1024x1024. The step time grows with that count.

### Samplers (`--sampler euler|dpm2m`)

Both samplers run one transformer pass per step on the same sigma schedule
(the dynamic shift and `shift_terminal` of `scheduler/scheduler_config.json`).

- `euler` (default): the first-order Euler step of diffusers
  `FlowMatchEulerDiscreteScheduler`, the scheduler the checkpoint ships.
- `dpm2m`: DPM-Solver++(2M), diffusers `DPMSolverMultistepScheduler` with
  `use_flow_sigmas=True`, `prediction_type="flow_prediction"`,
  `solver_order=2` and `final_sigmas_type="zero"`. Each step except the
  first and the last also uses the previous step's clean-image prediction,
  so it is second order and should reach the quality of Euler in fewer
  steps. An img2img, inpainting or outpainting run starts mid-schedule with a
  first-order step; a panorama steps the blended canvas once per step.

`dpm2m` has not been compared against `euler` on real images yet; try
`--sampler dpm2m --steps 10` to `12` against the default 18 Euler steps.

### VAE tiles

| Option | Meaning | Default |
| --- | --- | --- |
| `--vae-tile S[,T]` | decode tiles of S pixels, one every T pixels (multiples of 16); without T, the stride is 3/4 of S | 256,192 |

`TQwenImage21VaeDecoder` decodes the image in overlapping tiles and blends
the overlaps. The tile size trades memory against quality and speed: a
smaller tile needs less memory but adds tiles and overlap, so the decode
does more work. At 1024x1024, 128-pixel tiles showed visible seams between
tiles. The default is 256,192.

### GPU and CPU

| Option | Meaning | Default |
| --- | --- | --- |
| `--gpu` | OpenCL for the transformer step pass and the VAE decode | on with an OpenCL build |
| `--cpu` | run everything on the CPU | off |
| `--gpu-platform N` | OpenCL platform index | 0 |
| `--gpu-device N` | OpenCL device index within the platform | 0 |
| `--no-gpu-shared-kernel` | give each layer private OpenCL kernels and command queue instead of the net-wide shared ones (the shared ones are faster) | shared |

The startup banner prints the OpenCL platform and device names. After the
image, a `Compute :` line says where the transformer step pass and the VAE
decode actually ran.

### Threads

| Option | Meaning | Default |
| --- | --- | --- |
| `--serial` | single-threaded forward passes | parallel |
| `--max-threads N` | cap the parallel forward at N worker threads | every CPU thread |

By default the forward passes use the parallel layer scheduler with
intra-layer threading.

### Profiling

| Option | Meaning |
| --- | --- |
| `--stats` | after each image, print the stage table below |
| `--profile` | after the image, print the stage table, then the per-layer time of the transformer step pass (by block role and by layer class, summed over the blocks and steps), of the prefix pass, and of the VAE decode (by layer class, one table per tile shape) |

`--profile` drains the OpenCL queue after every layer that queued OpenCL
work, so each row includes its kernels and transfers, and the steps run
slower than without it. With `--no-gpu-shared-kernel` the layers have private
queues; the table header says which queues the profiler drains.

The stage table has one row per stage of the image: where it ran, its wall
time and its share of the image's stages. This example is the pico
checkpoint on PoCL (CPU OpenCL), one-shot, with `--stats --vae-tile 32,16`:

```
Stages of image 1 (wall time):
  Stage                Ran on             ms      %
  load text encoder    CPU               6.0    0.0
  encode prompt        CPU               5.0    0.0
  load transformer     CPU+upload      170.0    0.8  uploads 0.0 MB
  transformer prefix   CPU               5.0    0.0
  denoise              OpenCL         2100.0   10.3  2 steps x 1050.0 ms
  load VAE             CPU             115.0    0.6
  VAE decode           OpenCL        17989.0   88.2  16 tile(s) x 1124.3 ms, 4 net(s) built
  PNG save             CPU               2.0    0.0
  sum of stages                      20392.0  100.0
  host<->OpenCL: step pass up 0.01 MB, down 0.01 MB per step; VAE decode up 0.10 MB, down 0.05 MB per tile
  OpenCL memory held: transformer 0.1 MB after the last step, VAE 0.5 MB (largest tile net); sampled, not every allocation
  peak RSS this image: 116 MB
```

- "sum of stages" adds the rows. It leaves out the short gaps between
  stages, so it can differ from the time on the `Wrote ...` line.
- The load rows appear only when a stage loaded its component (one-shot
  runs). "CPU+upload" marks a load that also uploaded weights to OpenCL
  memory, with the uploaded MB.
- In the REPL and with `--repeat`, the prompt is encoded once: its rows count
  in the first image of the prompt, and later images show "encoded once".
  The "VAE encode" row (the init image of img2img, the condition images of
  an edit) and the "vision encode" row of an edit work the same way, and
  "denoise" counts only the steps that ran.
- ms per step and ms per tile include one-time setup: step 1 builds and arms
  the step pass, and the decode builds, arms and uploads the weights of one
  VAE net per tile shape (the "net(s) built" count).
- `--stats` adds no per-layer drain. It turns on the host<->OpenCL transfer
  counting, which costs four integer adds per transfer (two of them atomic)
  and stays on for the rest of the session. The counter sees the buffer
  writes and reads of `TEasyOpenCL` (`WriteBuffer`, `ReadBuffer` and their
  offset forms), not mapped buffers or buffers created from host memory.
  The step figure is the step pass's transfers divided by the steps (step 1
  also uploads the prompt K/V); the tile figure is the decode's transfers
  divided by the tiles.
- "OpenCL memory held" is `TNNet.OpenCLBufferBytes`, sampled after the last
  step (the transformer block weights, the step pass and the prompt K/V) and
  before each VAE tile net is freed (the largest is shown). It counts the
  buffers those routines know about; the OpenCL K/V cache of
  `TNNetFusedSDPA` is not counted. After the steps the pipeline frees the
  step pass and the prompt K/V, so in the REPL only the transformer block
  weights stay in OpenCL memory beside the VAE.
- The peak RSS is per image: `--stats` and `--profile` reset the kernel's
  peak-RSS mark (`/proc/self/clear_refs`) before the prompt is encoded and
  before each later image of a `--repeat` batch, so the "peak" in the phase
  lines is per image too. A side effect: `getrusage` `ru_maxrss` and
  `/usr/bin/time -v` ("Maximum resident set size") then report only the peak
  since the last reset. Where the file is not writable the line says
  "peak RSS of the process".

## Performance and memory

Every run prints the time and the process memory (RSS and peak) of each
phase and of each step.

One run measured by the user on an **NVIDIA L4** (24 GB GPU, machine with
53 GB RAM): one-shot, 1024x1024, `--int4` (int4 transformer, int8 text
encoder), 10 steps, VAE tiles 256,192, 12 CPU threads, OpenCL for the
transformer step pass and the VAE decode. The run predates the latest VAE
memory work (task B2f in `tasklist.md`). The default is int8 weights and 18
steps; this run used `--int4` and 10 steps.

| Phase | Time |
| --- | --- |
| load text encoder | 41.5 s |
| encode prompt | 5.0 s |
| load transformer | 55.3 s |
| transformer prefix | 1.7 s |
| denoising, 10 steps | 135.3 s (mean 13.05 s per step, ~13 s) |
| load VAE | 4.2 s |
| VAE decode | 35.2 s |
| **total** | **279.8 s** |

Peak RSS: 12.7 GB.

The default is 18 steps, 1.8 times the steps of this run. In the REPL the
three loads happen once per session instead of once per image.

In the REPL and with `--repeat N` > 1 the transformer stays in OpenCL memory
while the VAE decodes, so both need GPU memory at once. If they do not fit, `/tile 128` (or
`--vae-tile 128`) lowers the VAE's share, at the cost of the seams described
under [VAE tiles](#vae-tiles).

## Known limitations

- `--steps 1` fails with "Invalid floating point operation":
  `TNNetFlowMatchEulerScheduler.SetSigmas` divides zero by zero in the
  `shift_terminal` stretch. Use 2 or more steps.
- An OpenCL error, such as a failed allocation, is printed but not detected.
  The pipeline cannot tell that a layer failed, so a REPL image can be wrong
  yet reported as written. Watch the log for OpenCL error lines.
- Editing is tested only against diffusers on the pico checkpoint; it has
  not run on the real checkpoint yet. The same holds for inpainting.
- `--upscale` has run only with the repository's random pico RRDBNet
  (`tests/fixtures/tiny_rrdbnet*`); a real `RealESRGAN_x4plus.pth` run is
  still pending.
- The VAE encoder and the vision tower run untiled on the CPU, so their
  memory grows with the output size. They have not been measured at
  1024x1024 on the real checkpoint.

## Advanced: A/B switches

Three environment variables turn off recent memory savings, for A/B
comparisons against the previous behaviour. Each is on unless set to `0`:

| Variable | What `=0` turns off |
| --- | --- |
| `NEURAL_OPENCL_IMPLICIT_CONV` | the implicit-GEMM OpenCL convolution (FP32 inference); the convolutions build the im2col matrix instead (`neural/neuralopencl.pas`) |
| `NEURAL_OPENCL_SHARE_OUTPUTS` | OpenCL output buffer reuse by liveness in the VAE nets (`TNNet.ShareOpenCLOutputsByLiveness`) |
| `NEURAL_SHARE_HOST_OUTPUTS` | host layer output sharing by liveness in the VAE nets (`TNNet.ShareHostOutputsByLiveness`) |

```
NEURAL_OPENCL_SHARE_OUTPUTS=0 bin/x86_64-linux/bin/QwenImage --model Qwen-Image-2.1 --int4 -p "..."
```

## Tests

The pipeline's parity tests compare against diffusers on the pico checkpoint
in `tests/fixtures/tiny_qwenimage21/` (generated by
`tools/make_pico_qwenimage21_fixture.py`): `TestQwenImage21Pipeline*`,
`TestQwenImage21Transformer*`, `TestQwenImage21Vae*` and
`TestFlowMatch*`. The img2img tests (`TestQwenImage21Img2Img*`,
`TestQwenImage21LanczosResizeVsPIL`, `TestQwenImage21PrepareVaeImage`,
`TestQwenImage21SizeForAspect`, `TestFlowMatchImg2ImgStartStepVsOracle`,
`TestFlowMatchScaleNoiseVsOracle`) read the oracle of
`tools/make_pico_qwenimage21_img2img_fixture.py`, as do the inpaint tests
(`TestQwenImage21Inpaint*`: the mask latents, the kept latents bit for bit
after every step, and the step latents against diffusers' blend);
`TestQwenImage21OutpaintKeepsOriginal` runs the outpaint canvas, crop and
paste. `TestLoRAMergeImageAdapter*` write pico adapters in the diffusers and
kohya layouts at test time and check the merged tensors against
W + scale * (B @ A); `TestQwenImage21TransformerLoRA` checks that a
zero-scale adapter leaves the velocity unchanged. The edit tests
(`TestQwenImage21Edit*`) read the oracle of
`tools/make_pico_qwenimage21_edit_pipeline_fixture.py`: the diffusers
pipeline with one and two condition images.
`TestRRDBNetUpscalerMatchesRRDBNetUpscale` checks the `--upscale` code
(`TRRDBNetUpscaler`) against a direct `RRDBNetUpscale` on a tiled pico image,
for RGB and for RGBA with a bilinear alpha. `TestRRDBNetUpscalerOpenCL`
checks it on OpenCL against the CPU. `TestDiTFirstBlockCacheDecision` and
`TestQwenImage21FirstBlockCache*` check the First-Block Cache against a
plain-loop reference: threshold 0 gives the same latents bit for bit, and a
threshold that skips some steps gives the reference's skip pattern and
latents, on the CPU and on OpenCL. No diffusers oracle covers it yet.
`TestQwenImage21GuidanceOffBitIdentical` checks that guidance off (scale 1,
or no negative prompt) gives the plain loop's latents bit for bit;
`TestQwenImage21GuidanceMix` that two prefix caches keep their prompts
apart and that a guided step equals CFG (and APG on x0 and on the velocity)
of two separate predictions, with a First-Block Cache and a one-window
panorama changing nothing; `TestQwenImage21GuidanceEditPrefixCaches` the
same caches with a condition image; `TestQwenImage21GuidanceOpenCL` a
guided int8 denoise on OpenCL against the CPU.
`TestQwenImage21PanoramaOneWindowParity` checks that a canvas of one window
gives the plain denoise bit for bit and plans 3 (4 wrapped) windows on a
wider canvas; `TestMultiDiffusionConstantSeams` and
`TestMultiDiffusionCircularWrap` check the window averaging on stubs: a
constant velocity stays constant at the seams, an identity stub returns the
canvas, and a wrapped canvas blends column 0 with the last window.
