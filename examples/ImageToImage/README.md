# ImageToImage: SDEdit image-to-image / real-image editing

A **latent-diffusion editing** example (Meng et al. 2021, *SDEdit: Guided Image
Synthesis and Editing with Stochastic Differential Equations*,
[arXiv:2108.01073](https://arxiv.org/abs/2108.01073)), structurally distinct from
every other generator in the tree, which all start from pure noise. SDEdit starts
the reverse process from a **real image** that has been encoded to a latent and
then only **partially** noised, so the source image's coarse layout survives
while a new conditioning prompt steers the content — the classic "edit this
picture" workflow.

The whole pipeline reuses already-landed importers with **no new model and no new
layer** — the only new code is the encode → partial-noise → denoise → decode
driver and the `strength` knob:

1. **encode** — `BuildVaeEncoderFromSafeTensors` maps the RGB image to a clean
   latent `z0`;
2. **noise** — `TNNetDiffusionScheduler.AddNoise` noises `z0` up to
   `t_start = round(strength·T)` (`strength 0` keeps the source, `strength 1` is
   full noise = ordinary text→image-from-noise);
3. **denoise** — the landed PixArt denoiser
   (`BuildPixArtFromSafeTensors` + `PixArtDenoise`) runs a truncated reverse DDIM
   trajectory from `t_start` down to 0 with the new prompt's T5 states;
4. **decode** — `BuildVaeDecoderFromSafeTensors` (latent `/0.18215`) returns the
   edited RGB.

The **`--inpaint`** flag turns the same driver into a diffusion inpainting run
(the RePaint trick, Lugmayr et al. 2022,
[arXiv:2201.09865](https://arxiv.org/abs/2201.09865)): a binary mask over the
latent grid marks the region to regenerate (1) vs keep (0), and before each
reverse step the unmasked latent is overwritten with the clean encoded latent
re-noised to that step's timestep — only the masked region follows the denoiser.

## Build / run

```
cd examples/ImageToImage
LAZUTILS_PATH=$(find /usr/lib/lazarus /usr/share/lazarus -name "utf8process.ppu" -printf "%h\n" 2>/dev/null | head -1)
fpc -O3 -Mobjfpc -Sh -Fu../../neural -Fu"$LAZUTILS_PATH" ImageToImage.lpr
# or: lazbuild ImageToImage.lpi --build-mode=Release
./ImageToImage                       # smoke run on the tiny fixtures
./ImageToImage --strength 0.6        # stronger edit (more noise added)
./ImageToImage --steps 8             # more reverse steps
./ImageToImage --inpaint             # regenerate only a masked latent region
```

Real checkpoints: `--vae-encoder enc.safetensors --vae-encoder-config enc.json
--vae-decoder dec.safetensors --vae-decoder-config dec.json --pixart
pix.safetensors --pixart-config pix.json`.

## Input

Real Stable-Diffusion / PixArt checkpoints are far too large for this
environment, so by default the example runs a self-contained smoke on the
committed tiny **random** importer-parity fixtures
(`tests/fixtures/tiny_vae_encoder|decoder|pixart.safetensors`). Those nets are
untrained, so the "edit" is not a meaningful picture — the point is to exercise
the full driver path end to end within a small RAM/time budget. A synthetic
color-ramp source is generated when no `--image` is supplied (PPM reading is
omitted to keep the example dependency-free). The fixture grids differ (8×8×4 VAE
latent vs 6×6×4 PixArt grid); the driver center-crops the latent for the denoise
loop (a no-op with matched real checkpoints).

## Output

`edit_before.ppm` (VAE round-trip of the source, the strength-0 reference) and
`edit_after.ppm` (the edit); `--inpaint` additionally writes `edit_mask.ppm`. The
run asserts no NaN/Inf decoded pixels, and `--inpaint` asserts the unmasked
output latent matches the source within tolerance while the masked region
differs. Pure CPU.
