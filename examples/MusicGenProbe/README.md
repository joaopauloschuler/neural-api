# MusicGenProbe — EnCodec codec isolation diagnostic

A **component-isolation diagnostic** for the MusicGen text-to-music pipeline. The
full pipeline chains **T5 → MusicGen decoder → EnCodec** codec, so a bad output
could come from any stage. This probe tests the **EnCodec codec alone** — no T5,
no language model — by round-tripping a waveform (encode → RVQ codes → decode) and
reporting **reference-free** metrics that tell you whether the codec is the
culprit or the bug is upstream.

```
waveform (WAV or synthesized tone)
  -> EnCodec encode -> RVQ codes [codebooks x frames]   (range-check each codebook)
  -> EnCodec decode -> reconstruction
  -> metrics: peak, NaN/Inf, energy ratio out/in, input/output Pearson corr
  -> write reconstruction to WAV
```

## The importer

`BuildEnCodecFromSafeTensors` (config `TEnCodecConfig`, loaded by
`ReadEnCodecConfigFromJSONFile`, printed by `EnCodecConfigToString`) builds a
`TEnCodecModel`. The probe uses `EncodeAudioToCodes`, `DecodeCodesToAudio`
(decode only the first `--quantizers K` RVQ stages) and `Reconstruct` (all
stages), with `LoadWav16ToVolume` / `SaveVolumeToWav16` for I/O.

## Running

With **no arguments** it runs on the committed pico fixture
(`tests/fixtures/tiny_musicgen_encodec.safetensors` +
`tiny_musicgen_encodec_config.json`). The reconstruction is noise (random
weights), and the verdict acknowledges that "low correlation here is EXPECTED";
the point is that the pipeline runs without NaN/Inf.

```
cd examples/MusicGenProbe
# build with lazbuild (or fpc), then:
./MusicGenProbe
```

Flags (all optional, `--flag value`):

- `--download` — fetch the real `facebook/encodec_32khz` from the HF Hub
  (cached); `--encodec-repo <repo>` overrides the repo.
- `--in <file>` — input WAV (default: a synthesized 220/277/330 Hz triad +
  rising sweep).
- `--out <file>` — output WAV (default `musicgen_probe_recon.wav`).
- `--seconds <float>` — input duration cap (default 3.0; **memory scales with
  this**).
- `--quantizers <int>` — decode only the first K RVQ stages (default 0 = all).

The verdict block reports peak amplitude, NaN/Inf counts, the **energy ratio**
out/in (healthy ~0.2..5), and the **input/output correlation** (healthy >> 0;
~0 = noise). On real weights, low correlation means the codec is the prime
suspect; structure preserved means the bug is UPSTREAM (LM/T5).

Coded by Claude (AI).
