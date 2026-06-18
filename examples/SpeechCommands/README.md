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
(fixed `RandSeed = 1234`) of six distinct acoustic "words" -- low/mid/high
pure tones, an up-chirp, a down-chirp, and band-limited noise -- each with a
touch of seeded jitter and additive noise so the classes are non-trivial but
clearly separable. **Every clip passes through the real `ComputeWhisperLogMel`**,
so the smoke exercises the exact frontend+training path end to end on CPU in
well under a minute, and reports a final test accuracy far above the 1/6
(~16.7%) chance line.

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
with `LoadWav16ToVolume` and featurized through the same frontend; the example
does a seeded 80/10/10 split and trains for 30 epochs. There is no resampler
in this v1, so non-16 kHz files raise an exception -- convert first
(`ffmpeg -ar 16000 -ac 1`).

A tiny downloader that fetches a few keyword folders and re-encodes them is
provided (NOT run by the smoke -- the full archive is ~2.3 GB):

```bash
scripts/download_speech_commands.sh /path/to/out
./SpeechCommands --full /path/to/out/keywords
```
