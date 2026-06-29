# DACRoundTrip — Descript Audio Codec round-trip

A round-trip demo of the **DAC (Descript Audio Codec)**, an RVQGAN-lineage neural
audio codec: it compresses a waveform to a stack of discrete codes via a
**Residual Vector Quantizer (RVQ)** and reconstructs it (waveform → codes →
waveform), then writes the reconstruction to a 16-bit WAV.

DAC differs from EnCodec / Mimi in three ways: (1) **SNAKE** activations
(`x + (1/(alpha+1e-9))*sin(alpha*x)^2`, learnable per-channel `alpha`);
(2) **symmetric** (non-causal) conv padding; (3) a **factorized, L2-normalized**
RVQ.

```
waveform
  -> conv ENCODER (Snake + symmetric Conv1d + residual units)  -> latent
  -> factorized L2-normalized RVQ        -> [num_codebooks][num_frames] codes
  -> RVQ decode (sum of out_proj(codebook[code]))              -> latent
  -> conv DECODER (Snake + ConvTranspose1d upsamplers + Tanh)  -> reconstructed waveform
```

## The importer

`BuildDACFromSafeTensors` (config `TDACConfig`, printed by `DACConfigToString`)
builds a `TNNetDAC` holder. The round-trip uses `Model.Encode(Wave, out Codes,
out Frames)` and `Model.Decode(Codes, out Recon, 0)`.

## Running

With **no arguments** (run it from this directory) it runs a self-contained
**pico smoke test** on the committed random fixture
(`tests/fixtures/tiny_dac.safetensors` + `tiny_dac_config.json`) over a short
generated tone:

```
cd examples/DACRoundTrip
# build with lazbuild DACRoundTrip.lpi (or fpc), then:
./DACRoundTrip
```

With a **real DAC checkpoint directory** as the first argument it loads
`model.safetensors` + `config.json` from there and round-trips a 440 Hz tone:

```
./DACRoundTrip /path/to/descript_dac
```

It prints the codebook count, latent frame count, sample codes, reconstruction
error (`max|diff|`, RMS), the compression ratio, and writes `dac_recon.wav`. The
pico fixture weights are **untrained**, so the reconstruction is not meant to
resemble the input — it only exercises the full encode → RVQ → decode pipeline.

Coded by Claude (AI).
