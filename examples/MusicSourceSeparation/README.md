# Demucs music source separation (4 stems)

An **audio source-separation** demo with an audio
**output** modality: one **mixed** stereo track in, **four** stems out
(drums / bass / other / vocals). Every prior audio demo is either analysis
(Whisper / Wav2Vec2 → text), codec (EnCodec waveform → codes → waveform) or
synthesis (VITS text → waveform); Demucs *splits* one waveform into several.

This is the **time-domain** (waveform) Demucs (Défossez et al. 2019,
*Music Source Separation in the Waveform Domain*, arXiv:1911.13254) — a
symmetric 1-D convolutional **U-Net**:

```
mixed waveform (stereo)
  -> ENCODER: depth blocks of
       Conv1d(in -> out, k=kernel_size, stride=stride) -> ReLU
       Conv1d(out -> 2*out, k=1) -> GLU(channel axis)        (each saves a skip)
  -> bi-LSTM bottleneck (lstm_layers stacked bidirectional) + Linear(2C -> C)
  -> DECODER: depth blocks of
       x += center_trim(skip, len(x))                        (U-Net skip add)
       Conv1d(in -> 2*in, k=context, pad=context//2) -> GLU
       ConvTranspose1d(in -> out, k=kernel_size, stride=stride)
       ReLU   (every block except the last)
  -> reshape to (sources, audio_channels, time), center-trimmed to input length
```

`BuildDemucsFromSafeTensors[Ex]` (`neuralpretrained.pas`, model_type `demucs`
/ `htdemucs`) builds a self-contained **`TNNetDemucs`** holder that runs this
math directly on channel-major arrays — exactly like the `TNNetHiFiGAN` /
`TNNetVits` / `TEnCodecModel` holders — and **reuses `THiFiGANConv` /
`RunHiFiGANConv`** for every Conv1d / ConvTranspose1d (no new leaf layer). The
**bi-LSTM** is the one piece run inline in the holder (there is no
bidirectional-LSTM leaf layer): forward and reverse `nn.LSTM` cells with the
standard `(i, f, g, o)` gate packing, concatenated on the feature axis, then a
`Linear(2C → C)`. Weights load as plain folded tensors:
`encoder.i.{0,2}.{weight,bias}`, `lstm.weight_ih_l*` / `weight_hh_l*` /
`bias_*` (+ `*_reverse`), `lstm_linear.{weight,bias}`,
`decoder.i.{0,2}.{weight,bias}`.

## Running

With **no arguments** it runs a self-contained **pico smoke test** on the
committed random fixture (`tests/fixtures/tiny_demucs.*`) and a procedurally
generated stereo two-tone mix — no download, pure CPU, a fraction of a second
— writing each of the four separated stems to a 16-bit WAV via
`SaveVolumeToWav16` (`neuralaudio.pas`; the stem's stereo channels are averaged
to mono for the mono writer). The fixture weights are untrained random, so the
stems are noise rather than real instruments; the smoke test only exercises the
full encoder → bi-LSTM → decoder separation pipeline end to end.

```
cd examples/MusicSourceSeparation
# build with lazbuild MusicSourceSeparation.lpi (or fpc), then:
ulimit -v 3000000
./MusicSourceSeparation
```

With a **real checkpoint directory** as the first argument (a `model.safetensors`
+ `config.json` time-domain Demucs) it builds the full-width model and
separates a synthesized stereo clip:

```
./MusicSourceSeparation /path/to/demucs
```

## Parity

`TestDemucsSeparationParity` (generator `tools/make_pico_demucs_fixture.py`)
separates two pinned mixed waveforms through the imported pico model and
asserts the four stems match a **self-contained numpy float64 oracle** to
`< 1e-4`. transformers ships no Demucs, so the oracle is hand-built from the
published architecture math (the precedent set by the other `*_tiny_fixture`
oracles), keeping the fixture KB-scale and committed.

## Scope and follow-ups

v1 imports the **time-domain** Demucs (v2/v3) U-Net only, with
`normalize=False` (the bare conv/GLU/LSTM/transpose-conv stack). Documented
follow-ups:

* the **hybrid time+spectral HTDemucs** spectral branch (STFT + a parallel
  2-D spectral U-Net summed with the time branch);
* the **v3 cross-domain transformer** bottleneck (in place of the bi-LSTM);
* input/output **normalization** (centering + std rescale) and weight
  rescaling;
* a **real downloaded checkpoint** end-to-end separation within the
  ~5 min / `ulimit -v 3000000` budget.
