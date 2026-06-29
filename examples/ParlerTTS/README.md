# Parler-TTS — description-conditioned codec-LM decode

An inference smoke test for the **Parler-TTS** codec-LM decoder importer.
Parler-TTS is a **description-conditioned** TTS model: a free-text *style
description* (encoded by a (By)T5 text encoder and consumed via cross-attention)
and a *transcript prompt* (an autoregressive prefix) jointly condition an
autoregressive **codec language model** that generates **DAC** audio codes. This
example exercises the importable piece — the prefix-prepended, cross-attended
codec decoder with **KV-cache incremental decoding** — on a pico fixture.

```
description hidden states (EncSeq x TextDModel)   [cross-attention]
transcript prompt ids (PromptLen)                 [prepended prefix]
  -> codec-LM decoder (autoregressive, KV-cache)
  -> Codes[NumCodebooks][NumFrames]   (delay-patterned multi-codebook DAC codes)
```

## The importer

`BuildParlerTTSFromSafeTensors` (config `TParlerConfig`, printed by
`ParlerConfigToString`) builds a `TParlerTTSModel` holder whose `Generate(EncStates,
PromptIds, NumFrames, UseCache, out Codes)` runs the codec decoder
autoregressively with a KV-cache incremental fast path. The builder is given the
sequence lengths up front (`EncSeq`, `PromptLen`, `CodecLen`); `NumFrames` is
`CodecLen - NumCodebooks` (the DAC delay pattern). Codes come back as a
`TNNetIntArr2D` of `NumCodebooks x NumFrames`.

## Running

With **no arguments** it runs a self-contained **pico smoke test** on the
committed random fixture (`tests/fixtures/tiny_parler.safetensors` +
`tiny_parler_config.json`). It synthesizes a synthetic sine-wave description
conditioning, a toy transcript prompt, and generates the DAC code stack with the
KV-cache decode path, printing the codes codebook-by-codebook. Pure CPU, a
fraction of a second, no download.

```
cd examples/ParlerTTS
# build with lazbuild ParlerTTS.lpi (or fpc), then:
./ParlerTTS
```

The fixture weights are **untrained random**, so the codes are **not real
speech** — the smoke test only exercises the description-conditioned codec
decode. For a waveform, pair this decoder with the real (By)T5 encoder and a DAC
decoder. If the fixture is missing, the program points you at
`python tools/parler_tiny_fixture.py`.

Coded by Claude (AI).
