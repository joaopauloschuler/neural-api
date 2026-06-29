# SpeakerDiarization — Pyannote "who spoke when"

A **speaker diarization** demo: a frame-level multi-speaker activity detector
using the **Pyannote segmentation-3.0** importer. Unlike transcription models
(Whisper, Wav2Vec2) it answers *"who was speaking when"*, not *"what was said"*.
A learnable SincNet band-pass front-end feeds a bidirectional minimal-LSTM trunk
and a per-frame **powerset** multilabel head (7 classes covering every subset of
≤3 concurrent speakers).

```
waveform
  -> SincNet band-pass front-end  (TNNetSincConv1D, filters from low-cutoff+bandwidth)
  -> bidirectional TNNetMinLSTM trunk (forward + time-reversed)
  -> per-frame POWERSET multilabel head (7 classes, <=3 concurrent speakers)
  -> per-speaker binary activity timeline + RTTM turns
```

## The importer

`BuildPyannoteSegmentationFromSafeTensorsEx` (config `TPyannoteConfig`, loaded by
`ReadPyannoteConfigFromJSONFile`, printed by `PyannoteConfigToString`) builds a
`TNNet` whose front end is `TNNetSincConv1D` and whose trunk is bidirectional
`TNNetMinLSTM`. `PyannoteFrameCount` gives the frame count for a sample length;
`PyannotePowersetDecode(Logits, frame, Config, out Active)` converts the per-frame
powerset argmax into a per-speaker activity bitmask.

## Running

This example takes **no arguments** — it runs a self-contained smoke test on the
committed pico fixture (`tests/fixtures/tiny_pyannote.safetensors` +
`tiny_pyannote_config.json`), synthesizing a two-speaker waveform (180 Hz then
540 Hz, with brief overlap) in memory. Small, pure-CPU, low-RAM.

```
cd examples/SpeakerDiarization
# build with lazbuild SpeakerDiarization.lpi (or fpc), then:
./SpeakerDiarization
```

It prints the config, a per-frame speaker-activity timeline (`.` silent / `#`
active), and **RTTM** `SPEAKER` lines for each turn, and writes the synthesized
audio to `diarization_demo.wav`. The fixture is random, so the activity is
illustrative, not a real model. If the fixture is missing, run
`tools/make_pico_pyannote_fixture.py` first.
