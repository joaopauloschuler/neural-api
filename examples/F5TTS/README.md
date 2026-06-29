# F5-TTS — flow-matching voice-clone DiT velocity field

A **flow-matching voice-clone** text-to-speech smoke test. F5-TTS is a
non-autoregressive, non-GAN voice cloner: it learns a **DiT (Diffusion
Transformer) velocity field** and synthesizes a mel-spectrogram by integrating a
conditional **flow-matching ODE** from Gaussian noise to the target mel, with a
reference clip + text conditioning the field.

```
noised mel Xt (S x NMel)  + reference mel Cond  + text ids  + time t
  -> DiT velocity field   v = NN.Compute([Xt, Cond, Text, Time])
  -> Euler ODE step       Xt += dt * v          (8 steps, t: 0 -> 1)
  -> predicted mel        (needs a vocoder for a waveform)
```

## The importer

`BuildF5TTSFromSafeTensors` (config `TF5Config`, printed by `F5ConfigToString`)
loads the DiT velocity field into a `TNNet`. The forward takes four
`TNNetVolume` inputs — the noised mel `Xt`, the reference mel `Cond`, the
character `Text` ids, and the continuous diffusion `Time` (scaled by 1000) — and
returns the velocity. The sampler in this example is a plain **8-step Euler ODE
integrator** over `t in [0,1]`: each step forwards the DiT and advances
`Xt += dt * v`. The internal sinusoidal time embedding
(`TNNetSinusoidalTimeEmbedding`) maps `t` into the trunk.

## Running

With **no arguments** it runs a self-contained **pico smoke test** on the
committed random fixture (`tests/fixtures/tiny_f5.safetensors` +
`tiny_f5_config.json`). It builds the DiT, sets up a synthetic reference mel and
a toy text id sequence, runs the 8 Euler ODE steps, and prints the sampled mel's
min/max/mean plus its first frame. Pure CPU, a fraction of a second, no
download.

```
cd examples/F5TTS
# build with lazbuild F5TTS.lpi (or fpc), then:
./F5TTS
```

The fixture weights are **untrained random**, so the sampled mel is noise, not
speech — this only exercises the importer + flow-matching ODE loop end to end.
If the fixture is missing, the program points you at
`python tools/f5_tiny_fixture.py` to regenerate it.

This v1 outputs the **mel only**; pair it with a vocoder (Vocos / HiFi-GAN) to
get a waveform.

Coded by Claude (AI).
