# ControlNetCanny

Stable Diffusion 1.x text-to-image guided by a ControlNet, as diffusers'
`StableDiffusionControlNetPipeline`. The example drives `TStableDiffusionPipeline`
(`neural/neuralpretrained.pas`) with a `TStableDiffusionControlNet` conditioner:

```
prompt --CLIP tokenizer + text encoder--> text states (and the empty prompt's, for CFG)
for each sampler step, for the conditional and the unconditional pass:
  ControlNet(latents, text states, control image, t)  -> down + mid residuals (x scale)
  UNet(latents, text states, t, + residuals)            -> noise prediction
  CFG mix, scheduler step
VAE decoder -> PNG
```

The sampling loop (CFG, the sampler named by `scheduler_config.json`, the
steps) is the pipeline's own `Denoise`; the conditioner only adds the
ControlNet residuals to each UNet pass.

## Running

```sh
cd examples/ControlNetCanny
lazbuild -B ControlNetCanny.lpi
../../bin/x86_64-linux/bin/ControlNetCanny                       # pico fixtures
../../bin/x86_64-linux/bin/ControlNetCanny \
  --model /path/to/stable-diffusion-v1-5 \
  --controlnet /path/to/sd-controlnet-canny \
  --control edges.png --prompt "a futuristic city" --steps 25 --seed 1 \
  --out city.png --gpu
../../bin/x86_64-linux/bin/ControlNetCanny \
  --model /path/to/stable-diffusion-v1-5 \
  --controlnet /path/to/sd-controlnet-canny \
  --photo photo.jpg --edges-out edges.png --prompt "a futuristic city" --gpu
```

| Option | Meaning |
| --- | --- |
| `--model DIR` | diffusers Stable Diffusion 1.x folder (default `tests/fixtures/tiny_sd15`) |
| `--controlnet DIR` | diffusers `ControlNetModel` folder: `config.json` + `diffusion_pytorch_model.safetensors` (default `tests/fixtures/tiny_sd15_controlnet`) |
| `--control FILE` | control image; for the canny ControlNet white edges on black (e.g. `cv2.Canny` output). Resized to the image size (Lanczos). Default: drawn circle and square outlines |
| `--photo FILE` | photo to take the edges from instead of `--control`: resized to the image size (PIL Lanczos, uint8 rounding), then `CannyEdges` |
| `--canny-low T`, `--canny-high T` | `cv2.Canny` thresholds (100, 200: the diffusers ControlNet recipe) |
| `--canny-blur R` | `GaussianBlurImage` radius (sigma) applied before `CannyEdges` (default 0: no blur, as `cv2.Canny`) |
| `--edges-out PNG` | also save the extracted edge map |
| `--prompt`, `--negative` | prompt and negative prompt |
| `--steps N`, `--guidance G`, `--seed N` | 25, 7.5, 0 |
| `--scale S` | `controlnet_conditioning_scale` (1.0) |
| `--size N` | square image side (default: the checkpoint's, 512 for SD 1.5) |
| `--out PNG` | output file (default `controlnet_canny.png`) |
| `--int8` | UNet, ControlNet and text encoder weights in int8 |
| `--gpu`, `--gpu-platform N`, `--gpu-device N` | OpenCL |

With no `--model` / `--controlnet` the run uses random-weight pico fixtures:
it checks wiring and I/O, not image quality.

## Parity

`TestStableDiffusionControlNetParity` (`tests/TestNeuralPretrained.pas`) runs
the pipeline with the pico ControlNet (a diffusers `ControlNetModel.from_unet`
of `tiny_sd15`, conditioning scale 0.5) for 3 guided DDIM steps and compares
the latents after each step and the decoded image with a diffusers float64
`StableDiffusionControlNetPipeline` run: max |diff| < 1e-4. The fixture and the
oracle come from `tools/make_pico_sd_conditioning_fixtures.py`.

## Canny edges

`CannyEdges` (`neural/neuralimageresize.pas`) ports `cv2.Canny(image, low,
high)` with the defaults the diffusers recipe uses (aperture 3, L1 gradient):
3x3 Sobel with replicated borders, the strongest channel per pixel for RGB
input, non-maximum suppression in four directions with cv2's fixed-point
tan(22.5)/tan(67.5) tests, the thresholds compared with `>`, and hysteresis
over 8-connected neighbours. Like `cv2.Canny`, it does not blur; `--canny-blur`
adds a `GaussianBlurImage` pass first. The tests (`TestCannyEdges*` in
`tests/TestNeuralImagePreprocess.pas`) check hand-computed step, square and
hysteresis cases; no `cv2` comparison has been run.

## Limits

- Square images only (`TControlNetConfig.CondGrid` takes one side).
- `--photo` extracts the edges after resizing the photo to the image size;
  the diffusers recipe runs `cv2.Canny` on the original photo and lets the
  pipeline resize the edge map. Resize the photo yourself and pass it at the
  image size for the same result.
- Real checkpoints (`lllyasviel/sd-controlnet-canny` on SD 1.5) are untested.
- One ControlNet; diffusers' `control_guidance_start/end` and guess mode are
  not implemented.
