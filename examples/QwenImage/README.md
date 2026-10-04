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

Image editing and reference images are not implemented yet (see
[Known limitations](#known-limitations)).

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
memory at a time.

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
| `/seed N` | seed of the next prompt; without it the seed grows by one per prompt |
| `/tile SIZE[,STRIDE]` | VAE tile, as `--vae-tile` |
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
| `--profile` | after the image, print the per-layer time of the transformer step pass (by block role and by layer class, summed over the blocks and steps), of the prefix pass, and of the VAE decode (by layer class, one table per tile shape) |

`--profile` drains the OpenCL queue after every layer that queued OpenCL
work, so each row includes its kernels and transfers, and the steps run
slower than without it. With `--no-gpu-shared-kernel` the layers have private
queues; the table header says which queues the profiler drains.

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

In the REPL the transformer stays in OpenCL memory while the VAE decodes, so
both need GPU memory at once. If they do not fit, `/tile 128` (or
`--vae-tile 128`) lowers the VAE's share, at the cost of the seams described
under [VAE tiles](#vae-tiles).

## Known limitations

- `--steps 1` fails with "Invalid floating point operation":
  `TNNetFlowMatchEulerScheduler.SetSigmas` divides zero by zero in the
  `shift_terminal` stretch. Use 2 or more steps.
- An OpenCL error, such as a failed allocation, is printed but not detected.
  The pipeline cannot tell that a layer failed, so a REPL image can be wrong
  yet reported as written. Watch the log for OpenCL error lines.
- Text-to-image only: image editing, reference images and the negative
  prompt (classifier-free guidance) are not implemented yet.

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
`TestFlowMatch*`.
