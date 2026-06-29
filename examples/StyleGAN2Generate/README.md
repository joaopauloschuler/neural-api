# StyleGAN2Generate: style-based image synthesis

The demo for `BuildStyleGAN2GeneratorFromSafeTensors`
(`neural/neuralpretrained.pas`), a style-based generative import (Karras et al.
2020, *Analyzing and Improving the Image Quality of StyleGAN*,
[arXiv:1912.04958](https://arxiv.org/abs/1912.04958)). It synthesizes ONE image
on the CPU from a fixed latent.

The synthesis path the importer builds and this example runs:

* an 8-layer **mapping MLP** turns the latent `z` into a `w` latent;
* the **synthesis network** grows the image from a learned constant through a
  tower of resolution blocks, each = nearest-`×2` upsample (after the first) →
  [modulated/demodulated conv (the `TNNetModulatedConv2D` primitive) + per-pixel
  noise injection (scaled by a learned strength) + LeakyReLU(0.2)] → a `toRGB`
  modulated `1×1` conv (no demod) summed into an upsampled RGB skip.

Each modulated conv reads a per-input-channel style vector from a small affine
layer "A" applied to `w`. This is **inference-only** synthesis (no discriminator /
path-length reg / training in v1).

## Build / run

```
cd examples/StyleGAN2Generate
lazbuild StyleGAN2Generate.lpi --build-mode=Release
# run from the repo root so the fixture path resolves:
../../bin/x86_64-linux/bin/StyleGAN2Generate
../../bin/x86_64-linux/bin/StyleGAN2Generate model.safetensors [config.json]
```

## Input

The official StyleGAN2 weights are not redistributable / not obtainable offline,
so — like the repo's RRDBNet / VAE-decoder pico fixtures — this falls back to the
committed config-faithful **random** pico generator
(`tests/fixtures/tiny_stylegan2.safetensors` + `..._config.json`, parity-checked
< 1e-4 against a float64 numpy oracle in `TestNeuralPretrained.pas`). The latent
is a fixed, reproducible deterministic ramp (no RNG). Pass a real `.safetensors`
(+ `config.json` sibling) to synthesize from your own checkpoint.

## Output

Writes `stylegan2_sample.ppm` (a P6 color image) and prints the config plus a
small ASCII intensity preview. Pure CPU.
