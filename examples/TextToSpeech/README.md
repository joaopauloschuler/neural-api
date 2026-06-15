# TextToSpeech — VITS / MMS-TTS end-to-end text-to-speech

The first **text-to-speech** demo in the repo. VITS (Kim et al. 2021, the
architecture behind `facebook/mms-tts-*` and `kakao-enterprise/vits-ljs`)
synthesizes a raw waveform end-to-end from a sequence of token ids:

```
token ids
  -> text_encoder (relative-position transformer) -> per-token prior
     mean/log-variance + hidden
  -> deterministic duration_predictor -> frames-per-token
  -> length regulator EXPANDS the prior along time
  -> prior_latents = mean + z*exp(logvar)*noise_scale   (z = prior noise)
  -> normalizing FLOW (RealNVP/Glow additive coupling, run in reverse)
  -> HiFi-GAN DECODER (the same generator as BuildHiFiGANFromSafeTensors)
  -> raw mono waveform -> 16-bit PCM WAV (SaveVolumeToWav16)
```

## The importer

`BuildVitsFromSafeTensors[Ex]` (model_type `vits`, `neuralpretrained.pas`)
builds a `TNNetVits` holder doing the **inference** math directly on
channel-major arrays (the whole pipeline is conv1d / relative-position
attention / WaveNet residual stacks, exactly faithful in direct conv math).
Only the inference path is imported:

- the **conditional-prior FLOW** is RealNVP/Glow additive coupling — VITS's
  `log_stddev` is identically zero, so each coupling layer is a pure additive
  shift `second_half -= mean(first_half)` with the mean from a `conv_pre ->
  WaveNet -> conv_post` stack; run in REVERSE with channel-flip between layers;
- the **decoder IS the HiFi-GAN generator** — the same synthesis code as
  `BuildHiFiGANFromSafeTensors` (`TNNetHiFiGAN`), loaded under the `decoder.`
  key prefix (`conv_post` is bias-free here);
- the **text encoder** uses relative-position attention with `emb_rel_k`/
  `emb_rel_v` windowed bias (HF `VitsAttention`);
- the **deterministic** duration predictor
  (`use_stochastic_duration_prediction=false`, the MMS-TTS default) plus the
  monotonic **length-regulator** expansion.

VITS inference injects noise into the flow prior. For a deterministic result
the prior noise `z` is an **explicit input** to `Synthesize`; the example draws
it from a fixed RNG seed.

## Running

With no arguments this runs a self-contained pico smoke test on the committed
random fixture (`tests/fixtures/tiny_vits.*`): it builds the model, synthesizes
a short utterance from a fixed token sequence and a fixed noise tensor, and
writes the waveform to `/tmp/tts_smoke.wav`. The weights are untrained random,
so the output is NOISE, not intelligible speech; this only exercises the
importer + synthesis end to end.

```
TextToSpeech
```

Pure CPU, a fraction of a second, no download. The smoke is CPU/ulimit-bounded
(run with `ulimit -v 3000000` and `timeout 280`); a real downloaded checkpoint
(`facebook/mms-tts-eng`, `kakao-enterprise/vits-ljs`) synthesizes a sentence
within the ~5 min / ulimit budget.

For real speech, import a downloaded checkpoint with
`BuildVitsFromSafeTensors` and tokenize the text with the model's
`VitsTokenizer`.

## Parity

The pico fixture (`tools/make_pico_vits_fixture.py`) pins the contract; the
posterior encoder (training-only) is dropped from the committed weights:

- `TestVitsSynthesisParity` (`tests/TestNeuralPretrained.pas`) — per stage vs
  the HF `VitsModel` float64 oracle, all `< 1e-4`:
  - text-encoder prior means / log-variances and the deterministic durations;
  - the normalizing flow run in reverse (`FlowReverse`);
  - the end-to-end waveform, with the oracle's `z` fed explicitly so the
    comparison is exact (no RNG matching).

## Deferred follow-ups

- The **stochastic** duration predictor (`VitsStochasticDurationPredictor`,
  the spline-flow training/sampling path) — rejected loudly; the deterministic
  readout is implemented.
- **Multi-speaker** models (`num_speakers>1`, `speaker_embedding_size!=0`,
  global conditioning into the WaveNet/decoder) — rejected loudly.
- The **VitsTokenizer** (phonemizer / char vocab) — text is supplied as ids.
- Real-checkpoint smoke (offline + RAM-gated here, so deferred).

Coded by Claude (AI).
