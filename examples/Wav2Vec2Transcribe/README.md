# Wav2Vec2Transcribe — CTC speech-to-text

A **speech-to-text (ASR)** demo using a **Wav2Vec2 / HuBERT** checkpoint with a
**CTC** head. Unlike encoder-decoder models there is no autoregressive
generation: raw 16 kHz mono audio runs once through a strided conv feature
extractor + transformer encoder + linear CTC head, and the transcript is read
out by **greedy CTC decoding** (per-frame argmax, collapse repeats, drop blanks).

```
16-bit PCM WAV (16 kHz mono)
  -> samples in [-1,1)  (optionally zero-mean/unit-variance normalized)
  -> Wav2Vec2/HuBERT conv feature extractor + transformer encoder
  -> linear CTC head -> per-frame logits
  -> greedy CTC decode (argmax, collapse repeats, remove blank)
  -> vocab detokenize ('|' -> space)  -> transcription
```

## The importer

`BuildWav2Vec2FromSafeTensorsWithConfig` (config `TWav2Vec2Config`, loaded by
`ReadWav2Vec2ConfigFromJSONFile`) builds the network into a `TNNet`. Audio is
loaded with `LoadWav16ToVolume` (returns the sample rate) and optionally passed
through `NormalizeSamples`. After `NN.Compute`, the CTC logits are decoded by
`DecodeCTCGreedy(Logits, BlankId)` into token ids, then rendered to text against
`vocab.json` (word delimiter `|` → space; `<pad>`/`<s>`/`</s>`/`<unk>` skipped).

## Running

With **no arguments** it runs a self-contained **pico smoke test** on the
committed random fixture (`tests/fixtures/tiny_wav2vec2.safetensors` +
`tiny_wav2vec2_config.json`) over a synthesized 200-sample tone. The output is
gibberish (random weights) but proves the conv → encoder → CTC → decode pipeline.

```
cd examples/Wav2Vec2Transcribe
# build with lazbuild (or fpc), then:
./Wav2Vec2Transcribe
```

With a **real checkpoint** it takes a checkpoint directory and a WAV file:

```
./Wav2Vec2Transcribe <checkpoint-dir> <audio.wav> [--no-normalize]
```

- `checkpoint-dir`: a directory holding `model.safetensors` (or
  `pytorch_model.bin`), `config.json`, and `vocab.json` — e.g.
  `facebook/wav2vec2-base-960h` (~360 MB):
  ```
  mkdir -p /tmp/w2v && cd /tmp/w2v
  wget https://huggingface.co/facebook/wav2vec2-base-960h/resolve/main/model.safetensors
  wget https://huggingface.co/facebook/wav2vec2-base-960h/resolve/main/config.json
  wget https://huggingface.co/facebook/wav2vec2-base-960h/resolve/main/vocab.json
  ```
- `audio.wav`: **16 kHz mono 16-bit PCM** (`ffmpeg -i in.ext -ar 16000 -ac 1 out.wav`).
- `--no-normalize`: skip normalization for checkpoints with `do_normalize=false`.

It prints the loaded sample count/duration, config, build and encode times, the
CTC greedy ids, and the final `Transcription:` line.

Coded by Claude (AI).
