# Kokoro / StyleTTS2 — phonemes → waveform

A **phoneme-to-waveform** text-to-speech smoke test for the **Kokoro / StyleTTS2**
importer (`hexgrad/Kokoro-82M`, Apache-2.0). It exercises the full StyleTTS2
forward graph: a style/voice vector conditions the synthesis through **AdaIN**
injection at every stage — text encoding, duration prediction, F0/energy
prediction — and an iSTFTNet decoder turns predicted magnitude+phase into a
waveform via inverse STFT.

```
phoneme ids  + style/voice vector (StyleDim)
  -> text encoder (embed -> conv -> ReLU)
  -> duration predictor (AdaIN(style) -> conv -> proj)   -> per-phoneme durations
  -> length regulator (monotonic time expansion)
  -> F0 / energy predictors (AdaIN(style) -> conv -> proj)
  -> iSTFTNet decoder (AdaIN(style) -> magnitude + phase -> ISTFT)
  -> raw mono waveform -> 16-bit WAV
```

## The importer

`BuildKokoroFromSafeTensors` (config `TKokoroConfig`, printed by
`KokoroConfigToString`) builds a self-contained `TNNetKokoro` holder. The holder
exposes `Synthesize(Ids, Style, out Wave)` and `SynthesizeToWav(Ids, Style,
filename)`; the style vector is a `Config.StyleDim`-long float array (the
per-voice embedding StyleTTS2 calls the reference style).

## Running

With **no arguments** (run it from this directory) it runs a self-contained
**pico smoke test** on the committed random fixture
(`tests/fixtures/tiny_kokoro.safetensors` + `tiny_kokoro_config.json`). It feeds
a fixed 6-phoneme id sequence and a deterministic pseudo-voice style vector,
synthesizes, and writes `kokoro_pico.wav`. Pure CPU, a fraction of a second, no
download.

```
cd examples/KokoroTTS
# build with lazbuild KokoroTTS.lpi (or fpc), then:
./KokoroTTS
```

It prints the config, the phoneme ids, the output sample count and peak
amplitude, and the WAV path. The fixture weights are **untrained random**, so
the audio is **not intelligible speech** — the smoke test only proves the forward
graph (style AdaIN → duration → length-regulate → F0/energy → iSTFT) runs end to
end.

Coded by Claude (AI).
