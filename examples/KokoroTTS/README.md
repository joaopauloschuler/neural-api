# KokoroTTS — Kokoro-82M text-to-speech from phonemes

Status: **planned (coded), not yet tested by the user on the real
checkpoint.** The code exists and matches the reference `kokoro` package stage
by stage on the real checkpoint (see Parity below), but no full sentence has
been synthesized and listened to on a real machine yet.

`KokoroTTS` loads a `hexgrad/Kokoro-82M` snapshot folder (Apache-2.0;
StyleTTS2 + iSTFTNet, 81.7M parameters) into `TNNetKokoro`
(`neural/neuralpretrained.pas`) and writes a 16-bit mono WAV at **24 kHz**:

```
phoneme string -> ids (config.json vocab, wrapped in 0s)
  -> stage A  (PL-BERT/ALBERT, text encoder, duration predictor) -> frames per token
  -> stage B  (F0 and energy curves)
  -> stage-C decoder
  -> harmonic source + STFT (host)
  -> stage-C generator -> exp / sin -> inverse STFT (host) -> WAV
```

Stages A, B and C are `TNNet`s run by `TNNet.Compute`; each is built once
from the checkpoint and resized per sentence by a borrowing rebuild (no
second copy of the weights).

## Input: phonemes, not text

Text -> phoneme conversion (G2P) is **not included yet**. You supply the
phoneme string that misaki's English G2P (the front end of the `kokoro`
Python package) produces. Example, "Hello world. This is a test.":

```
həlˈO wˈɜɹld. ðˌɪs ɪz ɐ tˈɛst.
```

Symbols that are not in the checkpoint's vocab are skipped and printed. The
voice row is chosen by the phoneme count (`voices/<name>.pt` row
`len(phonemes) - 1`), as in the package.

Only the English voices (`af_*`, `am_*`, `bf_*`, `bm_*`) are meaningful for
now: the other languages need their own G2P, and the planned Pascal G2P is
English-only.

## Download

From `https://huggingface.co/hexgrad/Kokoro-82M` you need:

- `config.json`
- `kokoro-v1_0.pth` (327 MB)
- `voices/<name>.pt` for each voice you want (e.g. `voices/af_heart.pt`,
  0.5 MB)

```
huggingface-cli download hexgrad/Kokoro-82M config.json kokoro-v1_0.pth voices/af_heart.pt
```

Pass the folder that holds those files (in the HF cache it is
`~/.cache/huggingface/hub/models--hexgrad--Kokoro-82M/snapshots/<commit>/`).

## Running

```
KokoroTTS --model-dir <snapshot folder> \
  --phonemes "həlˈO wˈɜɹld. ðˌɪs ɪz ɐ tˈɛst." \
  --voice af_heart --out hello.wav
```

| Flag | Default | Meaning |
| --- | --- | --- |
| `--model-dir DIR` | required | snapshot folder (config.json, the single `*.pth`, `voices/`) |
| `--phonemes TEXT` | required | phoneme string (misaki English G2P output) |
| `--voice NAME` | `af_heart` | `voices/NAME.pt` |
| `--speed S` | `1` | speaking rate; durations are divided by S |
| `--out FILE` | `kokoro.wav` | output WAV, 16-bit, 24 kHz, mono |
| `--serial` | off | single-threaded forward passes |
| `--max-threads N` | every CPU thread | cap the parallel forward at N workers |
| `--gpu` | off (CPU) | stage nets on OpenCL (build with `-dOpenCL`, as every build mode of `KokoroTTS.lpi` does); host steps stay on the CPU |
| `--gpu-platform N`, `--gpu-device N` | 0, 0 | OpenCL platform / device |
| `--no-gpu-shared-kernel` | off | per-layer OpenCL kernels |

The program prints the skipped symbols, the token count, the frame count
(40 frames per second of audio), the audio length, the load time, the
synthesis time, the WAV write time, the total time and the peak RSS. Per-stage
times are not printed: `TNNetKokoro.Synthesize` does not report them.

**Zero noise:** the hn-NSF harmonic source runs without random noise by
default, so the same input gives the same WAV. The holder has a random-noise
option (`HarmonicSource.RandomNoise`, `NoiseSeed`); this example does not
expose it.

Build with `lazbuild -B KokoroTTS.lpi` (modes Default, Debug, Release; all
three define AVX2 and OpenCL). The binary goes to `bin/<cpu>-<os>/bin/`.

## Memory

No full sentence has been run on the real checkpoint on a measured machine
yet. What was measured on the development VM (CPU, real checkpoint):

- stages A -> B -> stage-C decoder on a 102-frame sentence (2.55 s of audio):
  999 MB peak RSS for the test process;
- the stage-C generator is the largest activation user: about 4.1 MB per
  frame on a stage-C-like test net, so a long input can need several GB. An
  earlier real generator run with host-output sharing off was killed at
  2.7 GB RSS on the 3.9 GB VM.

Keep inputs to a sentence or two until the end-to-end peak is measured.

## Smoke run without the real checkpoint

`tests/fixtures/tiny_kokoro82m` has the same layout with tiny random weights;
the output is noise, which is expected:

```
KokoroTTS --model-dir tests/fixtures/tiny_kokoro82m --phonemes "hˈO." --out tiny.wav
```

## Parity

Against the reference `kokoro` 0.9.4 package (`tools/make_kokoro82m_fixture.py`):
the tiny fixture per stage in the default test suite (`TestKokoro*Tiny*`), and
the real checkpoint per stage in tests that run only when the snapshot and
the reference dump (`tools/make_kokoro82m_fixture.py real`) are present
(`TestKokoroReal*`; the stage-C decoder, stages-chained, generator and
end-to-end ones also need `NEURAL_KOKORO82M_REAL_HEAVY=1`). End to
end the audio cannot match torch bit for bit: on unvoiced spans some STFT
phase bins of the harmonic source are rounding noise (MKL FFT artefacts in
torch), and the generator amplifies them (expected max abs difference about
0.1).
