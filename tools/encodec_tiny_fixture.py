#!/usr/bin/env python3
"""Generate a tiny RANDOM EnCodec parity fixture for
tests/TestNeuralPretrained.pas (no network access: the model is randomly
initialized from a PICO config, never downloaded).

EnCodec (facebook/encodec_24khz family, model_type "encodec") is a streaming
convolutional neural audio CODEC: a waveform -> conv ENCODER -> Residual
Vector Quantizer (RVQ, a cascade of codebooks each quantizing the residual
left by the previous one) -> discrete CODES, and the inverse RVQ-decode ->
conv DECODER -> reconstructed waveform. This fixture pins the WHOLE round
trip (waveform -> codes -> waveform) of a pico instance.

Fixtures, KB-scale, pinned in tests/fixtures/:

  tiny_encodec.safetensors: an EncodecModel with every trait the importer
      must reproduce, at pico width:
        - CAUSAL Conv1d with reflect left-padding (pad = (k-1)*dilation),
          weight_norm parametrized (the checkpoint stores original0 (g,
          shape [O,1,1]) and original1 (v); the effective weight is
          w[o] = g[o] * v[o] / ||v[o]||_F, weight_norm dim=0);
        - ELU activations between conv stages;
        - residual blocks: ELU -> Conv(dim->dim/compress, k=residual_kernel)
          -> ELU -> Conv(dim/compress->dim, k=1) + conv shortcut(dim->dim,k1);
        - strided downsample convs (k=2r, stride=r) per upsampling ratio;
        - a 2-layer residual LSTM bottleneck (out = LSTM(x) + x);
        - the DECODER mirrors with ConvTranspose1d (causal right-trim,
          trim_right_ratio 1.0);
        - RVQ: per stage argmin-L2 over the codebook (HF stores it as
          quantizer.layers.N.codebook.embed [codebook_size, codebook_dim]),
          decode = embedding lookup, residual subtracted between stages.
      normalize=False (no per-clip scale), use_causal_conv=True.

  tiny_encodec_config.json: the pico HF EncodecConfig.

  tiny_encodec_ref.json: the float64 oracle. For each pinned input clip:
        - "codes": the RVQ code stack [num_codebooks][num_frames] (int);
        - "recon": the reconstructed waveform (round-tripped through the
          full encode->decode), the array the Pascal test gates < 1e-4.
      Computed in float64 (the committed-fixture oracle convention).

Coded by Claude (AI).

Usage (from the repo root):
  /home/bpsa/x/bin/python tools/encodec_tiny_fixture.py
writes tests/fixtures/tiny_encodec{.safetensors,_config.json,_ref.json}.
Needs torch + transformers + safetensors + numpy.
"""
import json
import os

import numpy as np
import torch
from safetensors.torch import save_file
from transformers import EncodecConfig, EncodecModel

HERE = os.path.dirname(os.path.abspath(__file__))
FIX = os.path.join(HERE, "..", "tests", "fixtures")

SEED = 1234


def pico_config():
    return EncodecConfig(
        # 8 kHz with a x4 total downsample (ratios [2,2]) gives frame_rate
        # 2000 Hz, which makes the HF config derive exactly num_quantizers=4
        # at the 24 kbps top bandwidth: a real RVQ cascade at pico size.
        sampling_rate=8000,
        audio_channels=1,
        normalize=False,
        hidden_size=8,           # codebook_dim must equal this (RVQ width)
        num_filters=2,
        num_residual_layers=1,
        upsampling_ratios=[2, 2],
        norm_type="weight_norm",
        kernel_size=7,
        last_kernel_size=7,
        residual_kernel_size=3,
        dilation_growth_rate=2,
        use_causal_conv=True,
        pad_mode="reflect",
        compress=2,
        num_lstm_layers=2,
        trim_right_ratio=1.0,
        codebook_size=8,
        codebook_dim=8,
        use_conv_shortcut=True,
    )


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    cfg = pico_config()
    model = EncodecModel(cfg).eval()  # cfg yields num_quantizers == 4 here

    # Re-randomize the RVQ codebooks to be well separated (the default
    # init can leave near-degenerate rows, making argmin fragile). We pull
    # entries onto a small grid so float32 vs float64 cannot flip a tie.
    with torch.no_grad():
        for layer in model.quantizer.layers:
            cb = layer.codebook
            K, D = cb.embed.shape
            base = torch.randn(K, D, dtype=torch.float64) * 1.5
            cb.embed.copy_(base.to(cb.embed.dtype))
            cb.embed_avg.copy_(cb.embed)
            cb.inited.fill_(True)
            cb.cluster_size.fill_(1.0)

    # ---- Pinned input clips. Length chosen so the encoder produces a few
    # frames after the total downsample factor (prod(ratios)=4 here) and the
    # causal convs all keep positive length. Use dyadic-ish small values.
    total_down = int(np.prod(cfg.upsampling_ratios))
    n_frames = 6
    # encoder out length ~= ceil(L / total_down); pick L a multiple plus pad.
    length = n_frames * total_down + 32
    rng = np.random.RandomState(SEED)
    clips = []
    for _ in range(3):
        x = (rng.randn(length).astype(np.float64) * 0.3)
        # round to a coarse grid so it is exactly representable.
        x = np.round(x * 64.0) / 64.0
        clips.append(x.tolist())

    # ---- float64 oracle round trip.
    model_f64 = EncodecModel(cfg).double().eval()
    model_f64.load_state_dict(model.state_dict())

    refs = []
    with torch.no_grad():
        for x in clips:
            inp = torch.tensor(x, dtype=torch.float64).view(1, 1, -1)
            # bandwidth 24 kbps selects ALL 4 codebooks at this frame rate
            # (bandwidth=None would collapse the cascade to a single stage),
            # so the oracle genuinely exercises the residual VQ.
            enc = model_f64.encode(inp, bandwidth=24.0)
            codes = enc.audio_codes  # [num_chunks, B, num_q, T]
            codes_q = codes[0, 0]    # [num_q, T]
            dec = model_f64.decode(codes, enc.audio_scales)
            recon = dec.audio_values[0, 0]  # [L]
            refs.append({
                "input": x,
                "codes": codes_q.to(torch.int64).tolist(),
                "recon": recon.tolist(),
            })

    # ---- save safetensors (raw HF state dict, F32).
    os.makedirs(FIX, exist_ok=True)
    sd = {}
    for k, v in model.state_dict().items():
        sd[k] = v.to(torch.float32).contiguous()
    save_file(sd, os.path.join(FIX, "tiny_encodec.safetensors"))

    with open(os.path.join(FIX, "tiny_encodec_config.json"), "w") as f:
        json.dump(cfg.to_dict(), f, indent=1, default=str)

    with open(os.path.join(FIX, "tiny_encodec_ref.json"), "w") as f:
        json.dump({
            "num_codebooks": len(model.quantizer.layers),
            "codebook_size": cfg.codebook_size,
            "codebook_dim": cfg.codebook_dim,
            "hidden_size": cfg.hidden_size,
            "clips": refs,
        }, f)

    # report sizes
    st = os.path.getsize(os.path.join(FIX, "tiny_encodec.safetensors"))
    print("wrote tiny_encodec.safetensors %d bytes" % st)
    print("num codebooks", len(model.quantizer.layers),
          "frames", len(refs[0]["codes"][0]),
          "recon len", len(refs[0]["recon"]))


if __name__ == "__main__":
    main()
