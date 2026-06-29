# MusicGenMelody — MELODY-conditioned MusicGen generation

The melody-conditioned sibling of [`MusicGenText`](../MusicGenText). Where the
text demo steers generation with a **text prompt** through cross-attention, this
example steers it with a reference **melody** (a 12-bin chromagram) prepended to
the decoder sequence — the `facebook/musicgen-melody` (`model_type
"musicgen_melody"`) architecture.

```
reference melody waveform (synthesized here)
  -> ComputeMusicgenMelodyChroma (neuralaudio) -> one-hot chromagram
  -> audio_enc_to_dec_proj                      -> chroma conditioning
  -> (concat with the projected text condition, CHROMA FIRST)
  -> PREPENDED to the MusicGen Melody causal self-attention decoder
  -> greedy delay-pattern decode                -> [K][frames] code stack
  -> EnCodec audio DECODER (BuildEnCodecFromSafeTensors) -> mono waveform
  -> SaveVolumeToWav16                          -> a .wav clip
```

## What is genuinely new vs text-MusicGen

MusicGen Melody **reuses** the landed text-MusicGen path for the EnCodec codec,
the delay-pattern codebook interleaving, the `K` embedding tables / `K` LM
heads, and the sinusoidal positions. The new pieces:

1. **The chroma front-end** (`ComputeMusicgenMelodyChroma`, `neuralaudio.pas`):
   a power spectrogram (`n_fft=16384`, `hop=4096`, periodic Hann window,
   `center=True` reflect pad, normalized by the window L2 energy, `power=2`)
   projected through a librosa-style `chroma_filter_bank` onto 12 pitch
   classes, per-frame inf-norm normalized, then **argmax one-hot** — matching
   the HF `MusicgenMelodyFeatureExtractor` bit-for-bit.

2. **A decoder-only architecture**: unlike text-MusicGen's cross-attention
   decoder, the melody decoder is a **causal self-attention LM**. The
   conditioning (chroma + text) is **prepended** to the decoder sequence —
   `concat([audio_enc_to_dec_proj(chroma), enc_to_dec_proj(text)])` (chroma
   first), repeat-tiled to `chroma_length`. Logits are read at the decoder-frame
   positions.

`BuildMusicGenMelodyFromSafeTensors` (`neuralpretrained.pas`) builds a
self-contained `TMusicGenMelodyModel` holder
(`BuildConditioningPrefix` / `ComputeLogits` / `Generate`).

## Running (pico smoke demo)

This v1 example runs the self-contained **pico fixtures** (committed random
weights) — pure CPU, a fraction of a second:

```
MusicGenMelody                # synthesize melody -> chroma -> codes -> WAV
MusicGenMelody --frames 6     # explicit decoder frame count
MusicGenMelody --no-text      # chroma-only conditioning (zeroed text)
```

It synthesizes a deterministic A4 (440 Hz) + E5 (660 Hz) reference melody,
extracts its chroma (every frame resolves to pitch class **9** = A), prepends
the chroma + text conditioning to the decoder, greedily decodes a `[K][frames]`
code stack through the delay pattern, decodes it to audio with the EnCodec
decoder, and writes `musicgen_melody_demo.wav`. The fixture weights are
untrained random, so the clip is **noise** — this exercises the full
melody→chroma→audio wiring, not musical output.

A `--download` real-checkpoint mode for `facebook/musicgen-melody` (and the
real-melody-conditioned smoke clip) is a deferred follow-up.

## Parity

`TestMusicGenMelodyParity` (`tests/TestNeuralPretrained.pas`, generator
`tools/make_pico_musicgen_melody_fixture.py`) pins BOTH new pieces against a
float64 HF `MusicgenMelodyForConditionalGeneration` oracle: the chroma
extractor matches **exactly** (one-hot, max |diff| = 0) and one decoder forward
step (chroma + text conditioning prepended) matches the HF logits to
**< 1e-4**.
