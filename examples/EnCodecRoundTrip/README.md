# EnCodec round-trip (neural audio codec)

An **audio-generative** demo: a neural audio **codec**
that compresses a waveform into a stack of discrete codes and reconstructs it
(waveform → codes → waveform). Every prior audio demo (Whisper, Wav2Vec2 /
HuBERT) is analysis-only (audio → text); EnCodec is the inverse — a streaming
convolutional encoder/decoder that you can run in both directions.

The genuinely new building block is **Residual Vector Quantization (RVQ)**: a
cascade of `num_codebooks` codebooks where each successive codebook quantizes
the **residual** left by the previous one. The single-codebook
`TNNetVectorQuantizer` (used by VQ-VAE / MaskGIT) is exactly the one-stage
special case; RVQ stacks several so the latent is described by a *grid* of
codes (one row per codebook, one column per latent frame).

Pipeline (`BuildEnCodecFromSafeTensors`, `neuralpretrained.pas`, model_type
`encodec`, the `facebook/encodec_24khz` family — causal, weight-norm,
`normalize=false`):

```
waveform (mono)
  -> conv ENCODER: causal weight-norm Conv1d (reflect left-pad (k-1)*dilation)
       + ELU + resnet blocks + strided downsample convs
       + a RESIDUAL 2-layer LSTM bottleneck            -> latent frames
  -> RVQ encode: per frame, argmin-L2 over codebook 0, subtract the chosen
       vector, argmin over codebook 1, ...             -> [codebooks][frames]
  -> RVQ decode: sum of the chosen codebook vectors    -> latent
  -> conv DECODER (mirror): Conv1d + LSTM + ELU + ConvTranspose1d upsamplers
       + resnet blocks                                 -> reconstructed waveform
```

Conv weights are `weight_norm`-parametrized in the checkpoint (stored as
`original0` = g and `original1` = v); the importer reconstructs the effective
weight `w[o] = g[o] · v[o] / ‖v[o]‖` (weight_norm dim 0). The codec is run by
a self-contained `TEnCodecModel` holder (`EncodeAudioToCodes` /
`DecodeCodesToAudio` / `Reconstruct`) rather than a `TNNet` layer graph,
because EnCodec's causal reflect-padding, ceil-to-stride extra padding and the
ConvTranspose right-trim do not map cleanly onto the existing strided-conv
layers.

## Running

With **no arguments** it runs a self-contained **pico smoke test** on the
committed random fixture (`tests/fixtures/tiny_encodec.*`) and a procedurally
generated tone — no download, pure CPU, a couple of seconds. The fixture
weights are untrained, so the reconstruction is *not* meant to resemble the
input; the smoke test only exercises the full encode → RVQ → decode pipeline
end to end and reports the compression ratio.

```
cd examples/EnCodecRoundTrip
# build with lazbuild EnCodecRoundTrip.lpi (or fpc), then:
ulimit -v 3000000
./EnCodecRoundTrip
```

With a **real checkpoint directory** as the first argument (a downloaded
`facebook/encodec_24khz` with `model.safetensors` + `config.json`) it builds
the full-width codec and round-trips a synthesized 24 kHz tone, reporting the
compression ratio and reconstruction error:

```
./EnCodecRoundTrip /path/to/encodec_24khz
```

## Parity

`TestEnCodecRoundTripParity` (generator `tools/encodec_tiny_fixture.py`)
round-trips three pinned waveforms through the imported pico codec and asserts
the RVQ codes match the HF `EncodecModel` oracle **exactly** (integer argmin)
and the reconstructed waveform matches the float64 oracle to `< 1e-4`.

The 48 kHz stereo `normalize=true` variant (per-clip RMS scale) and the
chunked-streaming long-audio path are documented follow-ups. This codec is the
audio decoder the **MusicGen / Bark** text-to-audio follow-ups build on (a
transformer LM predicts the EnCodec code stack, then decodes through this
decoder — the audio analogue of the VQ-image-LM path).
