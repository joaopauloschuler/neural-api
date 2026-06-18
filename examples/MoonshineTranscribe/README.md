# MoonshineTranscribe

The [Moonshine](https://huggingface.co/UsefulSensors/moonshine-tiny)
streaming-ASR **encoder** -- a speech-to-text architecture
deliberately distinct from Whisper.

Whisper pads every clip to a **fixed 30 s log-mel spectrogram** (so a 1 s
utterance costs the same to encode as a 30 s one). Moonshine has **no mel
frontend**: it convolves RoPE-positioned features **directly off the raw
16 kHz waveform** with a small strided-conv stem, so the encoder compute
**scales with the actual audio length**.

## Encoder pipeline

```
raw 16 kHz waveform (samples,1,1)
  -> conv stem   conv1 (1->hidden, k=127, s=64, BIAS-FREE) -> tanh
                 GroupNorm(num_groups=1, hidden)            [over (T,C)]
                 conv2 (hidden->2*hidden, k=7, s=3)         -> erf-GELU
                 conv3 (2*hidden->hidden, k=3, s=2)         -> erf-GELU
  -> PRE-norm BIDIRECTIONAL transformer encoder
       partial RoPE on q/k (partial_rotary_factor), bias-free q/k/v/o,
       bias-free (gain-only) LayerNorm, erf-GELU fc1/fc2 MLP
  -> encoder hidden states (frames,1,hidden)
```

The hidden states feed a seq2seq decoder via `T5EncoderStatesInput` (the
landed two-net convention shared with the T5 / Marian / Pegasus / Whisper
importers). The RoPE + SwiGLU decoder and greedy/beam transcription
(reusing `DecodeSeq2SeqGreedy` / `BeamSearch`) are a documented follow-up;
this example exercises the **encoder import**.

## Running

```
# Deterministic smoke off the committed pico fixture (no download):
MoonshineTranscribe

# Real encoder from a downloaded checkpoint dir
# (model.safetensors / pytorch_model.bin + config.json):
MoonshineTranscribe /path/to/moonshine-tiny
```

With no argument it builds the encoder from the committed pico fixture
(`tests/fixtures/tiny_moonshine.*`), encodes a synthetic tone at two
waveform lengths, and prints the per-length frame count + latency so the
length-proportional-compute contrast with Whisper's fixed 30 s cost is
visible. Pure CPU, a fraction of a second.

Pico parity (`tools/make_pico_moonshine_fixture.py`,
`TestMoonshineEncoderParity`): the encoder hidden states match the HF
`MoonshineModel` float64 oracle to **< 1e-4**.

Coded by Claude (AI).
