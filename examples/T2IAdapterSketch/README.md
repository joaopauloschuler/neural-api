# T2I-Adapter sketch conditioning (base UNet + adapter, single-step smoke)

An offline, CPU smoke that wires an end-to-end **base Stable-Diffusion UNet +
T2I-Adapter** single-step denoise — the `StableDiffusionAdapterPipeline` inner
loop, composed over the repo's already-landed importers.

**T2I-Adapter** is the lighter sibling of ControlNet: a small conv encoder over a
spatial hint (canny / sketch / depth) produces a **pyramid of per-resolution
feature maps** that are *added* into the SD UNet down-block hidden state — no
transformer and no per-block zero-conv, just a lightweight ResNet-ish ladder.

## What it does

```
spatial hint (sketch/canny-style image)
  -> T2I-Adapter (BuildT2IAdapterFromSafeTensors / T2IAdapterFeatures)
     -> per-stage feature pyramid (one feature per UNet down block)
  -> base SD UNet built WITH adapter injection (BuildSDUNetFromSafeTensorsEx, pWithAdapter=true)
  -> SDUNetDenoiseWithAdapter ADDS those features into `sample` at the end of
     each down block (diffusers' down_intrablock_additional_residuals)
  -> predicted noise (the eps a sampler would step on)
```

Nothing here is a new leaf layer — it is pure plumbing over landed pieces. This
is a **wiring / throughput smoke**: it proves the chain runs offline and produces
a **finite** noise prediction, not photorealism (the pico fixtures carry random
weights). The adapter feature pyramid itself is parity-checked < 1e-4 in
`TestT2IAdapterParity`.

### Key neural-api pieces (`neuralpretrained`)

- **`BuildSDUNetFromSafeTensorsEx(..., pWithAdapter=true)`** — base SD UNet with
  the extra zero-default main-path inputs for adapter injection
  (`TSDUNetConfig`).
- **`BuildT2IAdapterFromSafeTensors` / `T2IAdapterFeatures`** — the adapter
  importer and its forward producing the feature pyramid (`TT2IAdapterConfig`).
- **`SDUNetDenoiseWithAdapter`** — one denoise step that adds the features into
  the UNet down-block path. `SDUNetAdapterFeatureCount` is asserted to match the
  adapter's feature count (config compatibility check).

## Running

```
cd examples/T2IAdapterSketch
lazbuild T2IAdapterSketch.lpi --build-mode=Release
./T2IAdapterSketch                                  # committed pico fixtures
./T2IAdapterSketch <unet.st> <unet.cfg> <adapter.st> <adapter.cfg>   # real, config-compatible checkpoints
```

(Or compile the `.lpr` with `fpc -Fu../../neural`.)

## Inputs / outputs

- **Fixtures (default):** `tests/fixtures/tiny_sd_unet.*` and the matched
  `tests/fixtures/tiny_t2i_adapter.*` — fully self-contained, no network access.
- **Inputs:** deterministic synthetic fillers (latent, T5-style text states, and
  a toy bright-lattice "sketch" hint) — no RNG, reproducible.
- **Shapes (pico fixtures):** sketch `(16,16,3)` → 2 adapter features `(16,8,8)`
  + `(32,4,4)`; latent `(8,8,4)`; text `(5,1,12)` → predicted noise `(8,8,4)`.
- **Output:** prints the configs, the produced feature-map shapes, and the
  predicted-noise min/max/mean, then asserts the prediction is finite
  (`OK` / `FAIL` with a non-zero exit code on NaN/Inf). No image file is written.

Coded by Claude (AI).
