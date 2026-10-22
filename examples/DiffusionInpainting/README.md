# DiffusionInpainting

Stable Diffusion 1.x inpainting, as diffusers' `StableDiffusionInpaintPipeline`:
the prompt regenerates the white part of the mask. The example
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

The VAE latents are sampled as diffusers' `retrieve_latents` does: mean + std
times unit noise, the noise drawn from the global RNG (seeded from `--seed`).
`--strength S` below 1 runs diffusers' `get_timesteps` truncation: only the
last `int(steps * S)` steps run, starting from the image latents re-noised
(`add_noise`, with the initial noise) to the first kept timestep. Strength 1
starts from pure noise.

```
image, mask --diffusers image/mask processors--> VAE encoder (sampled latents)
prompt --CLIP tokenizer + text encoder--> text states (and the empty prompt's)
strength < 1: start latents = image latents re-noised to the first kept step
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
  --steps 25 --strength 0.75 --seed 1 --out wall.png --gpu
```

| Option | Meaning |
| --- | --- |
| `--model DIR` | diffusers Stable Diffusion 1.x folder, inpainting or standard (default `tests/fixtures/tiny_sd15_inpaint`) |
| `--image FILE` | image to edit, resized to the output size (Lanczos). Default: a color ramp |
| `--mask FILE` | mask, white = regenerate; PIL 'L' gray, binarized at 0.5. Default: the right half |
| `--prompt`, `--negative` | prompt and negative prompt |
| `--steps N`, `--guidance G`, `--seed N` | 25, 7.5, 0 |
| `--strength S` | 0 < S <= 1 (default 1): runs `int(steps * S)` steps from the re-noised image latents |
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
whose VAE encode takes the latent mean (zero sample noise): max |diff| < 1e-4.
`TestStableDiffusionInpaintStrengthParity` runs strength 0.6 of 5 steps on the
same three cases plus the 9-channel UNet at strength 1, with the noise and the
VAE sample noise injected into the diffusers run: max |diff| < 1e-4. Fixtures
and oracle: `tools/make_pico_sd_conditioning_fixtures.py`.

## Limits

- Torch's RNG is not reproduced: the same seed gives a different image than
  diffusers.
- The output is the decoded image; the kept region is not pasted back.
- Real checkpoints are untested.
