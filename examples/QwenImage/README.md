# QwenImage — text-to-image with Qwen-Image-2.1

`QwenImage` turns a text prompt into an RGBA PNG image with the
Qwen-Image-2.1 checkpoint. The whole pipeline is Pascal code in this library
(`TQwenImage21Pipeline` in `neural/neuralpretrained.pas`); no Python runs at
generation time.

```
prompt
  -> processor/tokenizer.json (Qwen chat template, system prompt dropped)
  -> Qwen3-VL text encoder (text part: 36-layer Qwen3 decoder)
  -> MMDiT transformer prefix pass (the prompt's K/V, computed once)
  -> N flow-matching Euler steps over the image tokens
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
  the dynamic exponential shift and `shift_terminal`).
- **VAE decoder** — `TQwenImage21VaeDecoder`, decoding the 64-channel latents
  to RGBA at 16 times the latent width and height, in overlapping tiles.

Weights load as int8 by default, or as int4 (`--int4`) or FP32 (`--fp32`).
With an OpenCL build (the default), the transformer step pass (not with
`--fp32`) and the VAE decode run on the GPU; the text encoder and the
transformer prefix pass always run on the CPU.
`--cpu` runs everything on the CPU, and the program falls back to the CPU by
itself when it finds no usable OpenCL device.

`--image FILE` starts from an init image instead of pure noise (SDEdit
img2img, see [img2img](#img2img-sdedit)): it re-styles the image. Image
editing that follows instructions, and reference images, are not implemented
yet (see [Known limitations](#known-limitations)).

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
  processor/      tokenizer.json (the prompt tokenizer)
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
| `/steps N` | Euler steps |
| `/seed N` | seed of the next image; without it the seed grows by one per image |
| `/tile SIZE[,STRIDE]` | VAE tile, as `--vae-tile` |
| `/image FILE` | init image for the next prompts (img2img, as `--image`); `/image off` clears it |
| `/strength S` | img2img strength for the next images, in (0, 1], as `--strength` |
| `/repeat N PROMPT` | N images of PROMPT (1..1000) with consecutive seeds, each to the next numbered file; the text encoder runs once. Everything after N is the prompt. An image that fails is skipped and the rest of the batch runs |
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

### img2img (SDEdit)

`--image FILE` (or `/image FILE` in the REPL) makes each image start from an
init image instead of pure noise:

```
bin/x86_64-linux/bin/QwenImage --model Qwen-Image-2.1 --image photo.jpg \
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
batch or `/repeat`, and by later REPL prompts until `/image`, `/size` (when
the output size changes) or `/image off`. The noise still changes with the
seed. In keep-loaded mode (the REPL, `--repeat N` > 1) the first encode
loads the VAE encoder and keeps it beside the other components; a one-shot
run loads it and frees it after the encode.

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
| `--steps N` | Euler steps | 18 |
| `--seed N` | seed of the initial noise | 42 |
| `--image FILE` | img2img from this init image (PNG, JPEG, ...); see [img2img](#img2img-sdedit) | — |
| `--strength S` | img2img strength in (0, 1]: how much of the schedule runs (1 = from pure noise) | 0.6 |
| `--repeat N` | with `-p` or `--token-ids`: N images (1..1000) with seeds `--seed`, `--seed`+1, ...; the text encoder runs once. N > 1 keeps every component loaded, as the REPL does, and numbers the files like the REPL (`qwenimage_0001.png`, ...). An error ends the run | 1 |

An image of W x H pixels has (W/16) x (H/16) image tokens: 4096 at
1024x1024. The step time grows with that count.

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
  The "encode init image" row of img2img works the same way, and
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
- Image editing that follows instructions, reference images and the
  negative prompt (classifier-free guidance) are not implemented yet;
  `--image` is img2img (SDEdit) only.
- The VAE encoder runs untiled on the CPU, so its memory grows with the
  output size. It has not been measured at 1024x1024 on the real checkpoint.

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
`tools/make_pico_qwenimage21_img2img_fixture.py`.
