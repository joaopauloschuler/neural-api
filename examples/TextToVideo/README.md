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

(Or compile the `.lpr` with `fpc -Fu../../neural`.)

## Inputs / outputs

- **Fixture (default):** `tests/fixtures/tiny_cogvideox.*` — fully self-contained,
  no network access.
- **Text states:** deterministic synthetic values stand in for a real T5 tower.
- **Output:** writes `frame_<ii>.ppm` (P6) for each decoded frame and prints the
  config, the per-step latent trajectory norm, and an ASCII preview of each
  frame. With `--smoke` it prints `SMOKE OK`/`SMOKE FAIL` and sets the exit code
  (the build step exercises this).

Coded by Claude (AI).
