# Text-to-video (CogVideoX pipeline, offline CPU smoke)

A complete offline, CPU-native **text-to-video** pipeline that chains the landed
**CogVideoX** importer (THUDM/CogVideoX) into one sampling loop — distinct from
AnimateDiff, which bolts a temporal module onto a frozen SD UNet.

## What it does

```
caller-supplied T5 text states  (a real T5 tower is a follow-up)
  -> CogVideoX flat-DiT denoiser (BuildCogVideoXFromSafeTensors): a flat
     MMDiT-style transformer over a flattened (frame x height x width) latent
     token sequence with expert adaLN-Zero modulation + 3D RoPE
  -> multi-step DDIM / DPM-Solver++ reverse loop (TNNetDiffusionScheduler)
  -> sampled (NumFrames, GridH, GridW, in_channels) video latent
  -> 3D-causal-conv VAE decode tail (BuildCogVideoXVaeDecoderFromSafeTensorsEx
     + DecodeCogVideoXVae): a depth-axis CAUSAL temporal conv per spatial cell
     (left-pad time, no future-frame leakage) + SiLU + pointwise conv
  -> per-frame RGB -> a sequence of P6 PPM files (frame_00.ppm, frame_01.ppm, ...)
     or PNG files, plus an optional looping animated GIF
```

Nothing here is a new leaf layer — it is pure plumbing over landed pieces. With
the committed random-weight pico fixture this is a **wiring / throughput smoke**:
it proves the chain runs offline and produces **finite** video frames, not
photorealism. The CogVideoX importer is parity-checked < 1e-4 in
`TestCogVideoXParity`.

### Key neural-api pieces (`neuralpretrained`, `neuraldiffusion`)

- **`BuildCogVideoXFromSafeTensors`** + **`CogVideoXConditioning`** — the flat-DiT
  denoiser and its per-step timestep/text conditioning (`TCogVideoXConfig`).
- **`BuildCogVideoXVaeDecoderFromSafeTensorsEx`** + **`DecodeCogVideoXVae`** — the
  3D-causal-conv VAE decode tail. The two genuinely-new CogVideoX primitives (3D
  causal conv via `TNNetCausalConv1D`, 3D RoPE via `TNNetMRotaryEmbedding`) are
  reused landed leaves.
- **`TNNetDiffusionScheduler`** — the `smDDIM` / `smDPMSolverPP2M` reverse driver.
- **`TNNetAnimatedGIFWriter`** (`neuralgif`) — the `--gif` output: a fixed
  6x7x6 RGB-cube (or 256-gray) palette, LZW-coded frames, NETSCAPE2.0 looping;
  reusable by any example that writes frames.

## Running

```
cd examples/TextToVideo
lazbuild TextToVideo.lpi --build-mode=Release
./TextToVideo                          # committed pico fixture
./TextToVideo cogvideox.safetensors    # your own checkpoint (sibling config.json)
```

Optional trailing flags:

| Flag | Effect |
|------|--------|
| `--steps N` | number of reverse steps (default 4) |
| `--dpm` | use DPM-Solver++(2M) instead of DDIM |
| `--smoke` | run, assert finiteness, print `SMOKE OK`/`SMOKE FAIL` and exit (no PPM) |
| `--gif FILE` | also write all frames as one looping animated GIF (`TNNetAnimatedGIFWriter`, `neuralgif.pas`) |
| `--fps N` | GIF frame rate (default 8, the CogVideoX rate) |
| `--png` | write `frame_<ii>.png` instead of `frame_<ii>.ppm` |
| `--fbc-threshold X` | First-Block Cache: skip blocks 1..N-1 on steps where block 0 changed little (0 = off, the default; see below) |

(Or compile the `.lpr` with `fpc -Fu../../neural`.)

### First-Block Cache (`--fbc-threshold X`)

`--fbc-threshold X` skips most of the denoiser on steps where little changes.
It is a port of diffusers' `apply_first_block_cache`
(`hooks/first_block_cache.py`), shared with examples/QwenImage. Each step runs
`patch_embed`, the time and text embeddings and transformer block 0, then
compares block 0's video residual (its video output minus its video input)
with the residual of the last step that ran every block:

    change = mean |r - r_prev| / mean |r_prev|

When `change` is at most X, the step skips blocks 1 to N-1: it adds the video
residual those blocks produced on the last full step to block 0's video
output and runs `norm_out` and `proj_out` on the sum. When `change` is above
X, the step runs every block and keeps the new residuals. The first step
always runs every block. Each step line says when it skipped, and the run
ends with `First-Block Cache skipped k of N steps.`

```
./TextToVideo cogvideox.safetensors --steps 50 --fbc-threshold 0.1
```

- **Speed.** A skipped step costs the embeddings, one block and the output
  head instead of all N blocks. diffusers and ParaAttention report 1.5x to
  2x at thresholds 0.05 to 0.12 on other DiT models; this port has not been
  timed on a real CogVideoX checkpoint.
- **Quality.** A skipped step reuses an older residual, so a higher threshold
  skips more steps and can lose detail or motion. diffusers' default is 0.05.
- **Memory.** Three buffers of (video tokens x hidden) floats.
- **The pico fixture** has one block, so nothing can be skipped there (the
  example says so). `TestCogVideoXFirstBlockCacheReference` exercises the
  skips on a three-block copy of it.

`0` (the default) runs every block on every step, exactly as without the
option. The step is `TCogVideoXSampler.PredictNoise` and the cache
`TDiTFirstBlockCache`, both in `neural/neuralpretrained.pas`.

## Inputs / outputs

- **Fixture (default):** `tests/fixtures/tiny_cogvideox.*` — fully self-contained,
  no network access.
- **Text states:** deterministic synthetic values stand in for a real T5 tower.
- **Output:** writes `frame_<ii>.ppm` (P6, or `.png` with `--png`) for each
  decoded frame, the `--gif` file when asked, and prints the config, the
  per-step latent trajectory norm, and an ASCII preview of each
  frame. With `--smoke` it prints `SMOKE OK`/`SMOKE FAIL` and sets the exit code
  (the build step exercises this).

Coded by Claude (AI).
