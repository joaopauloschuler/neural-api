# Speech Commands -- keyword-spotting trainer

The audio analogue of `SimpleImageClassifier`, and the **first from-scratch
(no pretrained-model import) audio training example** in the library. It
proves that the log-mel frontend in `neural/neuralaudio.pas`
(`ComputeWhisperLogMel` -- the very same feature extractor that drives the
Whisper / Wav2Vec2 importers) is usable for ordinary supervised training, not
only for replaying imported checkpoints.

## Pipeline

```
16 kHz mono waveform (TNNetVolume of samples in [-1,1))
  -> ComputeWhisperLogMel        (REAL frontend; (NumFrames,1,NumMelBins))
  -> small 1-D conv stack over the time axis (mel bins = Depth)
  -> global average pool over time
  -> dense softmax over the keyword classes
```

The frontend emits `(NumFrames, 1, NumMelBins)` -- time along `SizeX`, mel
bins along `Depth` -- exactly the `(SeqLen, 1, Channels)` layout the
conv/sequence layers expect, so the conv stack consumes it directly.

Model (`BuildModel`):

| layer |
| --- |
| `TNNetInput(100, 1, 40)` (log-mel) |
| `TNNetMovingStdNormalization` |
| `TNNetConvolutionReLU(24, 5, pad 2)` + `MaxPool(4)` |
| `TNNetConvolutionReLU(32, 3, pad 1)` + `MaxPool(4)` |
| `TNNetConvolutionReLU(48, 3, pad 1)` + `MaxPool(2)` |
| `TNNetFullConnectReLU(64)` |
| `TNNetDropout(0.3)` |
| `TNNetFullConnectLinear(NumClasses)` + `TNNetSoftMax` |

## Default: synthetic smoke (no network, reproducible)

Run with no arguments and it generates a deterministic synthetic keyword set
(fixed `RandSeed = 1234`) of **ten genuinely confusable acoustic "words"**
chosen so the task is *not* trivially separable:

| class | content |
| --- | --- |
| `tone_430` / `tone_470` / `tone_510` | closely-spaced pure tones (~9% apart) |
| `chord_lo` / `chord_hi` | two-tone chords that **overlap** one pure tone (430 / 510 Hz) |
| `am_tone` | 7-9 Hz amplitude-modulated (tremolo) tone at 470 Hz |
| `chirp_up` / `chirp_down` | fast 350↔2600 Hz linear sweeps |
| `noise_bright` / `noise_dark` | colored noise differing only in spectral tilt |

Each clip carries seeded pitch jitter and a comparatively **low SNR**
(`noiseAmp ≈ 0.18`), so the classes overlap in mel space.
**Every clip passes through the real `ComputeWhisperLogMel`.** Because the
task is hard, validation now climbs gradually rather than saturating at
epoch 2 -- roughly 0.30 (epoch 1) → 0.65 (6) → 0.91 (10) → 0.99 (16),
hitting the 100% `TargetAccuracy` early-stop at **epoch 17** -- and the smoke finishes at
**≈99% held-out test accuracy** (observed 98.93%, `RandSeed = 1234`), far
above the 1/10 = 10% chance line. The whole run is well under a minute on
CPU within the 3 GB / 280 s budget.

```bash
cd examples/SpeechCommands
LAZUTILS_PATH=$(find /usr/lib/lazarus /usr/share/lazarus -name utf8process.ppu -printf '%h\n' | head -1)
fpc -B -Fu../../neural -Fu"$LAZUTILS_PATH" -Mobjfpc -Sh -O2 SpeechCommands.lpr
# Run under the project's 3 GB / 280 s budget:
( ulimit -v 3000000; timeout 280 ./SpeechCommands )
```

## Optional: real Google Speech Commands v2 (`--full`)

```bash
./SpeechCommands --full <dir>
```

`<dir>` must contain one subfolder per label, each holding 16 kHz mono 16-bit
PCM `.wav` clips (`<dir>/yes/*.wav`, `<dir>/no/*.wav`, ...). Clips are loaded
with `LoadWavResampledToVolume` (which resamples any sample rate to 16 kHz via
the windowed-sinc resampler in `neural/neuralaudio.pas`; a 16 kHz file passes
through bit-identically) and featurized through the same frontend; the example
does a seeded 80/10/10 split and trains for 30 epochs. WAVs at other rates are
accepted directly -- no `ffmpeg` pre-conversion needed.

A tiny downloader that fetches a few keyword folders and re-encodes them is
provided (NOT run by the smoke -- the full archive is ~2.3 GB):

```bash
scripts/download_speech_commands.sh /path/to/out
./SpeechCommands --full /path/to/out/keywords
```
