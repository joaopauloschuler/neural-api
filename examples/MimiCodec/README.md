# Mimi codec round-trip (streaming neural audio codec)

A round-trip demo of **Mimi** (`kyutai/mimi`), the 12.5 Hz neural audio
tokenizer behind **Moshi / Kyutai-TTS / Sesame CSM**. Like EnCodec it is an
**audio-generative** codec — a streaming convolutional encoder/decoder you run
in both directions (waveform → discrete codes → waveform) — but Mimi adds two
pieces on top of the EnCodec SEANet + RVQ design:

1. a small **causal transformer bottleneck** (RoPE self-attention + GELU MLP,
   pre-norm with per-channel LayerScale residuals) inserted after the conv
   encoder and before the conv decoder, with a strided **downsample** Conv1d /
   grouped **upsample** ConvTranspose1d to step between the EnCodec frame rate
   and the 12.5 Hz Mimi frame rate;
2. a **split residual vector quantizer**: a *semantic* RVQ (the first
   `num_semantic_quantizers` codebooks, distilled at train time — here just a
   nearest-centroid lookup) concatenated with an *acoustic* RVQ cascade. Each
   RVQ owns its 1×1 `input_proj` / `output_proj` convs, and each codebook is
   stored as `embed_sum` + `cluster_usage` (the effective centroid is
   `embed_sum / clamp(cluster_usage, eps)`).

Pipeline (`BuildMimiFromSafeTensors`, `neuralpretrained.pas`, model_type
`mimi`):

```
waveform (mono)
  -> conv ENCODER: causal Conv1d (constant/zero left-pad) + ELU + resnet blocks
  -> RoPE TRANSFORMER (pre-norm attention + GELU MLP, LayerScale)
  -> downsample Conv1d (stride 2, replicate pad)          -> 12.5 Hz frames
  -> SPLIT RVQ: semantic codebook(s) + acoustic RVQ cascade
       (per frame argmin-L2, subtract, next stage)        -> [quantizers][frames]
  -> RVQ decode (sum of chosen centroids, per-RVQ output_proj)
  -> upsample ConvTranspose1d (grouped) -> RoPE TRANSFORMER
  -> conv DECODER (mirror): Conv1d + ELU + ConvTranspose1d upsamplers + resnets
                                                           -> reconstructed waveform
```

The codec is run by a self-contained channel-major `TNNetMimi` holder
(`Encode` / `Decode` / `Reconstruct`), mirroring `TEnCodecModel`, because the
causal padding / extra ceil-to-stride pad / ConvTranspose right-trim and the
direct attention math do not map cleanly onto the layer graph. The holder
carries its signal in **double precision** (weights stay F32) so the deep
conv + high-gain transformer + conv pipeline stays inside the parity gate.

## Running

With **no arguments** it runs a self-contained **pico smoke test** on the
committed random fixture (`tests/fixtures/tiny_mimi.*`) and a procedurally
generated tone — no download, pure CPU, a couple of seconds — and writes the
resynthesized clip to a 16-bit WAV (`SaveVolumeToWav16`). The fixture weights
are untrained, so the reconstruction is *not* meant to resemble the input; the
smoke test only exercises the full encode → split-RVQ → decode pipeline.

```
cd examples/MimiCodec
# build with lazbuild MimiCodec.lpi (or fpc), then:
ulimit -v 3000000
./MimiCodec
```

With a **real checkpoint directory** as the first argument (a downloaded
`kyutai/mimi` with `model.safetensors` + `config.json`) it builds the
full-width codec and round-trips a synthesized 24 kHz tone:

```
./MimiCodec /path/to/kyutai_mimi
```

## Parity

`TestMimiParity` (generator `tools/mimi_tiny_fixture.py`) round-trips three
pinned waveforms through the imported pico codec and asserts the split-VQ codes
match the HF `MimiModel` oracle **exactly** and the reconstructed waveform
matches the float64 oracle to `< 1e-4`.

Real-checkpoint parity and the streaming chunk-at-a-time padding-cache path
(`MimiConv1dPaddingCache`, KV-cache transformer decode) are documented
follow-ups.
