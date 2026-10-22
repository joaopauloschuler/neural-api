# T2IAdapterSketch

Stable Diffusion 1.x text-to-image guided by a T2I-Adapter, as diffusers'
`StableDiffusionAdapterPipeline`. The example drives `TStableDiffusionPipeline`
(`neural/neuralpretrained.pas`) with a `TStableDiffusionT2IAdapter` conditioner:

```
hint image --T2I-Adapter (once)--> one feature map per UNet down block (x scale)
prompt --CLIP tokenizer + text encoder--> text states (and the empty prompt's, for CFG)
for each sampler step, for the conditional and the unconditional pass:
  UNet(latents, text states, t, + adapter features) -> noise prediction
  CFG mix, scheduler step
VAE decoder -> PNG
```

A T2I-Adapter is a small conv ladder over the hint (PixelUnshuffle, conv_in,
adapter ResNet blocks with AvgPool between stages). Its features do not depend
on the timestep, so it runs once per image; the UNet adds each feature to the
end of its down block, as diffusers' `down_intrablock_additional_residuals`.

## Running

```sh
cd examples/T2IAdapterSketch
lazbuild -B T2IAdapterSketch.lpi
../../bin/x86_64-linux/bin/T2IAdapterSketch                        # pico fixtures
../../bin/x86_64-linux/bin/T2IAdapterSketch \
  --model /path/to/stable-diffusion-v1-5 \
  --adapter /path/to/t2iadapter_sketch_sd15v2 \
  --hint sketch.png --prompt "a cozy cottage, watercolor" --steps 25 \
  --out cottage.png --gpu
```

| Option | Meaning |
| --- | --- |
| `--model DIR` | diffusers Stable Diffusion 1.x folder (default `tests/fixtures/tiny_sd15`) |
| `--adapter DIR` | diffusers `T2IAdapter` (`full_adapter`) folder: `config.json` + `diffusion_pytorch_model.safetensors` (default `tests/fixtures/tiny_sd15_t2i_adapter`) |
| `--hint FILE` | hint image; for a sketch adapter white strokes on black. A one-channel adapter reads it as PIL 'L' gray. Resized to the image size (Lanczos). Default: a drawn house outline |
| `--prompt`, `--negative` | prompt and negative prompt |
| `--steps N`, `--guidance G`, `--seed N` | 25, 7.5, 0 |
| `--scale S` | `adapter_conditioning_scale` (1.0) |
| `--size N` | square image side (default: the checkpoint's, 512 for SD 1.5) |
| `--out PNG` | output file (default `t2i_adapter_sketch.png`) |
| `--int8` | UNet and text encoder weights in int8 (the adapter stays FP32) |
| `--gpu`, `--gpu-platform N`, `--gpu-device N` | OpenCL |

With no `--model` / `--adapter` the run uses random-weight pico fixtures: it
checks wiring and I/O, not image quality.

## Parity

`TestStableDiffusionT2IAdapterParity` (`tests/TestNeuralPretrained.pas`) runs
the pipeline with the pico one-channel adapter (conditioning scale 0.8) for 3
guided DDIM steps and compares the latents after each step and the decoded
image with a diffusers float64 `StableDiffusionAdapterPipeline` run: max |diff|
< 1e-4. Fixture and oracle: `tools/make_pico_sd_conditioning_fixtures.py`.

## Limits

- Square images only (`TT2IAdapterConfig` takes one side).
- `full_adapter` only: `light_adapter` and the SDXL `full_adapter_xl` are refused.
- One adapter; diffusers' `MultiAdapter` is not implemented.
- Real checkpoints are untested.
