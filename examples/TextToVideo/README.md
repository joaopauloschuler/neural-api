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
  -> AutoencoderKLCogVideoX decoder (TCogVideoXVaeDecoder, 2 latent frames
     per batch with the diffusers conv cache): conv_in -> mid block -> up blocks -> norm_out ->
     SiLU -> conv_out, every conv a causal 3D conv and every norm a
     SpatialNorm3D on the latent; space doubles per up block, time doubles on
     the first log2(temporal_compression_ratio) blocks (frame 0 kept single)
  -> per-frame RGB -> a sequence of P6 PPM files (frame_00.ppm, frame_01.ppm, ...)
     or PNG files, plus an optional looping animated GIF
```

Nothing here is a new leaf layer — it is pure plumbing over landed pieces. With
the committed random-weight pico fixture this is a **wiring / throughput smoke**:
it proves the chain runs offline and produces **finite** video frames, not
photorealism. The denoiser is parity-checked < 1e-4 in `TestCogVideoXParity`
and the VAE decoder < 1e-4 against diffusers' own `AutoencoderKLCogVideoX.decode`
in `TestCogVideoXVaeDecoderParity`.

### Key neural-api pieces (`neuralpretrained`, `neuraldiffusion`)

- **`BuildCogVideoXFromSafeTensors`** + **`CogVideoXConditioning`** — the flat-DiT
  denoiser and its per-step timestep/text conditioning (`TCogVideoXConfig`).
- **`TCogVideoXVaeDecoder`** — the diffusers `AutoencoderKLCogVideoX` decoder
  (`TCogVideoXVaeConfig`, read from `vae/config.json`). Like diffusers'
  `_decode`, it decodes 2 latent frames per batch (the first batch also takes
  the extra frame of an odd clip), and every causal conv is padded with the last
  2 input frames of the batch before (the diffusers conv cache). One net decodes
  the first batch and owns the weights; a second net, which borrows them,
  decodes every later batch. Parity < 1e-4 against diffusers on 2-7 latent
  frames (`TestCogVideoXVaeDecoderParity`,
  `TestCogVideoXVaeBatchedDecodeParity`). The two genuinely-new CogVideoX primitives (3D
  causal conv via `TNNetConvolution3D` after a frame-0 repeat in
  `TNNetGatherChannels`, 3D RoPE via `TNNetMRotaryEmbedding`) are reused landed
  leaves.
- **`TNNetDiffusionScheduler`** — the `smDDIM` / `smDPMSolverPP2M` reverse driver.
- **`TNNetAnimatedGIFWriter`** (`neuralgif`) — the `--gif` output: a fixed
  6x7x6 RGB-cube (or 256-gray) palette, LZW-coded frames, NETSCAPE2.0 looping;
  reusable by any example that writes frames.

## Running

```
cd examples/TextToVideo
lazbuild TextToVideo.lpi --build-mode=Release
./TextToVideo                          # committed pico fixture
./TextToVideo cogvideox.safetensors --vae CogVideoX-2b/vae
                                       # your own denoiser (sibling config.json)
                                       # and a diffusers vae folder
```

Optional trailing flags:

| Flag | Effect |
|------|--------|
| `--vae PATH` | `AutoencoderKLCogVideoX` weights: a diffusers `vae` folder (`config.json` + `diffusion_pytorch_model.safetensors`) or a `.safetensors` file with a sibling `config.json` (default: the pico `tests/fixtures/tiny_cogvideox_vae`) |
| `--steps N` | number of reverse steps (default 4) |
| `--dpm` | use DPM-Solver++(2M) instead of DDIM |
| `--smoke` | run, assert finiteness, print `SMOKE OK`/`SMOKE FAIL` and exit (no PPM) |
| `--gif FILE` | also write all frames as one looping animated GIF (`TNNetAnimatedGIFWriter`, `neuralgif.pas`) |
| `--fps N` | GIF frame rate (default 8, the CogVideoX rate) |
| `--png` | write `frame_<ii>.png` instead of `frame_<ii>.ppm` |
| `--fbc-threshold X` | First-Block Cache: skip blocks 1..N-1 on steps where block 0 changed little (0 = off, the default; see below) |
| `--vae-tiling` | decode in overlapping 240x360-sample tiles, as diffusers `enable_tiling()` (see below) |

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

### VAE tiling (`--vae-tiling`)

`--vae-tiling` ports diffusers `AutoencoderKLCogVideoX.tiled_decode`. A latent
wider than `tile_sample_min_width / 8` or taller than
`tile_sample_min_height / 8` latent cells (defaults 360 and 240 samples, so
45 x 30 cells; 480x720 output is a 60x90 latent) is cut into tiles that start
every `int(tile_latent_min * (1 - overlap_factor))` cells (overlap factors
1/6 in height, 1/5 in width: 3 x 3 tiles for 480x720). Each tile runs its own
frame-batched decode with its own conv cache. The tiles are blended row-major
with diffusers' `blend_v` / `blend_h` linear ramps over
`int(tile_sample_min * overlap_factor)` samples (40 rows, 72 columns) and the
first `tile_sample_min - blend` samples of each are kept.

The decoder nets are sized to a tile shape, not to the frame, and borrow the
weights of one 1x1-latent net, so the activation RAM follows the largest tile.
A smaller latent is decoded in one piece, exactly as without the option (the
pico fixture's 2x2 latent always is). In code:
`TCogVideoXVaeDecoder.EnableTiling(TileSampleMinHeight, TileSampleMinWidth,
TileOverlapFactorHeight, TileOverlapFactorWidth)` with the diffusers defaults;
`TestCogVideoXVaeTiledDecodeParity` checks it against diffusers' own
`enable_tiling()` decode on small tiles.

## Inputs / outputs

- **Fixture (default):** `tests/fixtures/tiny_cogvideox.*` (denoiser) and
  `tests/fixtures/tiny_cogvideox_vae/` (VAE, 2 latent frames of 2x2 decode to
  8 frames of 16x16) — fully self-contained, no network access.
- **Memory:** the decoder holds one batch (2-3 latent frames, 8-9 output
  frames) of one tile, not the whole clip. The pico fixture needs a few MB.
  The batch nets share activation storage by liveness, and only one of them
  holds activations at a time. Measured on random weights of the CogVideoX-2b
  VAE shape, a decode peaks at about 6.7 MB per latent cell (93 MB without
  sharing) on top of the 0.47 GB of FP32 weights, so one 30x45-cell
  `--vae-tiling` tile needs about 9 GB and the untiled 60x90 latent about 36 GB.
- **Text states:** deterministic synthetic values stand in for a real T5 tower.
- **Output:** writes `frame_<ii>.ppm` (P6, or `.png` with `--png`) for each
  decoded frame, the `--gif` file when asked, and prints the config, the
  per-step latent trajectory norm, and an ASCII preview of each
  frame. With `--smoke` it prints `SMOKE OK`/`SMOKE FAIL` and sets the exit code
  (the build step exercises this).

Coded by Claude (AI).
