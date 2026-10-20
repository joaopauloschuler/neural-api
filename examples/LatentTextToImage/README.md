# LatentTextToImage: latent text-to-image pipeline

A complete offline, CPU **latent text-to-image** pipeline that chains the repo's
already-landed generative importers into one sampling loop — the SD3 / Sora /
PixArt-alpha recipe end to end. Nothing here is a new leaf layer; it is pure
plumbing over `BuildPixArtFromSafeTensors` (+ `PixArtConditioning` /
`PixArtDenoise`), `BuildVaeDecoderFromSafeTensors`, and the
`TNNetDiffusionScheduler` DDIM / DPM-Solver++ / CFG driver.

```
caller-supplied T5 text states   (a real T5 tower is a tracked follow-up)
  -> PixArt-alpha transformer denoiser (BuildPixArtFromSafeTensors)
  -> multi-step DDIM / DPM-Solver++ reverse loop (TNNetDiffusionScheduler) with
     classifier-free guidance (cond = prompt T5 states; uncond = ZERO T5 states)
  -> sampled latent
  -> VAE decode (BuildVaeDecoderFromSafeTensors; the /0.18215 latent scaling
     lives inside the decoder's first layer)
  -> RGB image -> P6 PPM.
```

## Build / run

```
cd examples/LatentTextToImage
lazbuild LatentTextToImage.lpi --build-mode=Release
# run from the repo root so the fixture path resolves:
../../bin/x86_64-linux/bin/LatentTextToImage
../../bin/x86_64-linux/bin/LatentTextToImage pixart.safetensors vae.safetensors
```

Optional trailing flags: `--steps N` (reverse steps, default 4), `--cfg W`
(guidance scale, default 4.0), `--dpm` (DPM-Solver++(2M) instead of DDIM),
`--unipc` (UniPC order-2 predictor-corrector), `--lcm` (Latent Consistency Model
few-step sampler — single model pass per step, guidance baked in),
`--vae-tiling` (diffusers `enable_vae_tiling`: decode the latent in OVERLAPPING
spatial tiles through a tile-sized decoder and feather-blend the seams with a
linear ramp, bounding peak VAE memory to one tile instead of the full latent —
the standard way to decode megapixel SD/SDXL latents on a memory-constrained
box; tile latent 32, overlap 0.25, clamped to the latent so it is a 1-tile
no-op on the tiny fixture), `--tiny-vae DIR`, `--preview N` and `--fast-vae`
(see below), `--smoke` (assert finiteness, print `SMOKE OK`/`SMOKE FAIL`, exit
without a PPM).

## Tiny VAE previews: `--tiny-vae DIR`, `--preview N`, `--fast-vae`

`--tiny-vae DIR` loads a diffusers `AutoencoderTiny` (TAESD) decoder from `DIR`
(`diffusion_pytorch_model.safetensors` + `config.json`) through
`BuildTinyVaeDecoderFromSafeTensorsEx`. The example builds it once, sized to the
sampled latent grid. PixArt-alpha samples in the SD VAE latent space, so use
`madebyollin/taesd` (4 latent channels); the example rejects a decoder whose latent
channel count differs from PixArt's `in_channels`. Without checkpoint arguments the
example uses the committed pico fixture `tests/fixtures/tiny_taesd4_decoder.*`
instead.

- `--preview N`: every N sampler steps, `TNNetDiffusionScheduler.PredictX0` turns
  the current latent and the guided model output into the x0 estimate. The tiny
  decoder decodes it to `latent_text_to_image_preview_NNN.ppm` (NNN = step
  number). It works with DDIM, `--dpm`, `--unipc` and `--lcm`. With `--smoke` no
  files are written and each preview is checked for NaN/Inf.
- `--fast-vae`: the final image comes from the tiny decoder, and the example does
  not load the full VAE. Use this for drafts. It overrides `--vae-tiling`.

```
../../bin/x86_64-linux/bin/LatentTextToImage pixart.safetensors vae.safetensors \
  --tiny-vae taesd --preview 5 --steps 20
../../bin/x86_64-linux/bin/LatentTextToImage --smoke --preview 1 --fast-vae
```

## Input

A real T5 encoder + real PixArt/VAE checkpoints are the only missing rung (a
tracked follow-up). By default the demo supplies **deterministic synthetic T5
states** and uses the committed config-faithful pico fixtures
(`tests/fixtures/tiny_pixart.*` and the matched `tiny_vae_decoder_ltt.*`), so it
is a **wiring/throughput smoke** — the nets are random so it proves the chain runs
offline and produces a finite image, not photorealism. Pass your own
`pixart.safetensors vae.safetensors` (with sibling `config.json` files) for real
checkpoints.

## Output

Writes `latent_text_to_image.ppm` (the decoded RGB image), plus
`latent_text_to_image_preview_NNN.ppm` files with `--preview N`, and prints the configs,
the per-step trajectory norm and an ASCII preview. Regression-tested by
`TestLatentTextToImageSmoke`. Pure CPU.
