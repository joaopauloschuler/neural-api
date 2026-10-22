# DiffusionInpainting

Stable Diffusion 1.x inpainting, as diffusers' `StableDiffusionInpaintPipeline`
at strength 1: the prompt regenerates the white part of the mask. The example
drives `TStableDiffusionPipeline` (`neural/neuralpretrained.pas`) with a
`TStableDiffusionInpainter` conditioner, which handles both kinds of UNet:

- **9-channel inpainting UNet** (e.g. `runwayml/stable-diffusion-inpainting`):
  every UNet pass sees `latents | mask | masked-image latents`, the mask
  nearest-resized to the latent grid and the masked image
  (`image * (mask < 0.5)`) VAE-encoded once.
- **Standard 4-channel UNet** (e.g. `runwayml/stable-diffusion-v1-5`): after
  every sampler step the kept region is replaced by the image latents
  re-noised to the next timestep with the initial noise (clean after the last
  step), the blended-diffusion mode diffusers runs for such a UNet.

```
image, mask --diffusers image/mask processors--> VAE encoder (latent mean)
prompt --CLIP tokenizer + text encoder--> text states (and the empty prompt's)
for each sampler step (conditional + unconditional UNet pass, CFG, step):
  9 channels: UNet(latents | mask | masked-image latents)
  4 channels: UNet(latents), then blend the kept region back
VAE decoder -> PNG
```

## Running

```sh
cd examples/DiffusionInpainting
lazbuild -B DiffusionInpainting.lpi
../../bin/x86_64-linux/bin/DiffusionInpainting                     # pico, 9 channels
../../bin/x86_64-linux/bin/DiffusionInpainting --model ../../tests/fixtures/tiny_sd15
../../bin/x86_64-linux/bin/DiffusionInpainting \
  --model /path/to/stable-diffusion-inpainting \
  --image photo.png --mask mask.png --prompt "a red brick wall" \
  --steps 25 --seed 1 --out wall.png --gpu
```

| Option | Meaning |
| --- | --- |
| `--model DIR` | diffusers Stable Diffusion 1.x folder, inpainting or standard (default `tests/fixtures/tiny_sd15_inpaint`) |
| `--image FILE` | image to edit, resized to the output size (Lanczos). Default: a color ramp |
| `--mask FILE` | mask, white = regenerate; PIL 'L' gray, binarized at 0.5. Default: the right half |
| `--prompt`, `--negative` | prompt and negative prompt |
| `--steps N`, `--guidance G`, `--seed N` | 25, 7.5, 0 |
| `--width N`, `--height N` | output size (default: the checkpoint's, 512 for SD 1.5) |
| `--out PNG` | output file (default `inpainted.png`) |
| `--int8` | UNet and text encoder weights in int8 |
| `--gpu`, `--gpu-platform N`, `--gpu-device N` | OpenCL |

With no `--model` the run uses a random-weight pico folder: it checks wiring
and I/O, not image quality.

## Parity

`TestStableDiffusionInpaintParity` (`tests/TestNeuralPretrained.pas`) runs the
pipeline on a 16x16 image and mask for 3 guided steps in three cases: the
4-channel `tiny_sd15` with DDIM and with Euler, and the 9-channel
`tiny_sd15_inpaint` with DDIM. It compares the latents after each step and the
decoded image with a diffusers float64 `StableDiffusionInpaintPipeline` run
whose VAE encode takes the latent mean: max |diff| < 1e-4. Fixtures and oracle:
`tools/make_pico_sd_conditioning_fixtures.py`.

## Limits

- Strength 1 only (start from pure noise); diffusers' `strength < 1` is not
  implemented.
- The VAE encoder returns the latent mean; diffusers samples it by default.
- The output is the decoded image; the kept region is not pasted back.
- Real checkpoints are untested.
