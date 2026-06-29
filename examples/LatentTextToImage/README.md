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
no-op on the tiny fixture), `--smoke` (assert finiteness, print `SMOKE OK`/
`SMOKE FAIL`, exit without a PPM).

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

Writes `latent_text_to_image.ppm` (the decoded RGB image) and prints the configs,
the per-step trajectory norm and an ASCII preview. Regression-tested by
`TestLatentTextToImageSmoke`. Pure CPU.
