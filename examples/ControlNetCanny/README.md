# ControlNetCanny

End-to-end **base-UNet-plus-ControlNet** single-step denoise smoke — the inner
loop of diffusers' `StableDiffusionControlNetPipeline`, wired offline on CPU over
the repo's already-landed importers.

## What it does

```
noisy latent + text states + a (canny-edge-style) control image
  -> ControlNet  (BuildControlNetFromSafeTensors / ControlNetResiduals)
     produces down_block_res_samples + mid_block_res_sample
  -> base SD UNet built WITH control injection
     (BuildSDUNet(..., pWithControl=true))
  -> SDUNetDenoiseWithControl ADDS the residuals into the decoder skip
     connections, exactly as diffusers does:
       down_block_res_samples = [d + c for d, c in zip(down, controlnet_down)]
       mid_block_res_sample  += controlnet_mid
  -> predicted noise (the eps a sampler would step on)
```

Nothing here is a new leaf layer — it is pure plumbing over landed pieces (the
ControlNet importer + `ControlNetResiduals`, the SD UNet importer + the
`SDUNetDenoiseWithControl` driver). It is a **wiring / throughput smoke**: it
proves the chain runs offline and produces a **finite** noise prediction, not
photorealism (the pico fixtures carry random weights).

## Running

```sh
cd examples/ControlNetCanny
LAZUTILS_PATH=$(find /usr/lib/lazarus /usr/share/lazarus -name utf8process.ppu -printf '%h\n' 2>/dev/null | head -1)
fpc -Fu../../neural -Fu"$LAZUTILS_PATH" -Mobjfpc -Sh -O2 ControlNetCanny.lpr
./ControlNetCanny                              # committed pico fixtures
./ControlNetCanny <unet.st> <unet.cfg> <controlnet.st> <controlnet.cfg>
```

With no arguments it falls back to the committed config-compatible pico fixtures
(`tests/fixtures/tiny_sd_unet.*` + `tests/fixtures/tiny_controlnet.*`). To run a
real model pass a config-compatible diffusers base UNet **and** ControlNet
(`block_out_channels`, `layers_per_block`, `cross_attention_dim`, latent grid,
text seq must match between the two).

## Shapes (pico fixtures)

```
latent          (8, 8, in_channels=4)
text states     (text_seq=5, 1, cross_dim=12)
control image   (16, 16, cond_channels=3)
  -> 4 down residuals + 1 mid residual
  -> predicted noise (8, 8, out_channels=4)
```

## Parity

The combined forward (base UNet noise prediction **with** the ControlNet
residuals injected) is parity-checked against a numpy float64 oracle
(`tools/controlnet_combined_fixture.py`) at `max|diff| < 1e-4` by
`TestControlNetCombinedParity` in `tests/TestNeuralPretrained.pas`.
