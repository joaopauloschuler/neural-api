#!/usr/bin/env python3
"""Generate a tiny RANDOM Mimi parity fixture for
tests/TestNeuralPretrained.pas (no network access: the model is randomly
initialized from a PICO config, never downloaded).

Mimi (kyutai/mimi, model_type "mimi") is the streaming neural audio codec
behind Moshi / Kyutai-TTS / Sesame CSM. It extends the convolutional SEANet
+ RVQ EnCodec topology with TWO new pieces:

  1. a small CAUSAL transformer (RoPE self-attention + GELU MLP, pre-norm
     with LayerScale residuals) inserted AFTER the conv encoder and BEFORE
     the conv decoder, plus a strided downsample conv / grouped transpose
     upsample conv to hit the 12.5 Hz frame rate;
  2. a SPLIT residual vector quantizer: a "semantic" RVQ (the first
     num_semantic_quantizers codebooks, with their own 1x1 in/out conv
     projections) concatenated with an "acoustic" RVQ (the rest, with their
     own projections). Each codebook stores embed_sum + cluster_usage; the
     effective centroid is embed_sum / clamp(cluster_usage, eps).

This fixture pins the WHOLE round trip (waveform -> codes -> waveform) of a
pico instance, computed in float64 (the committed-fixture oracle convention).

Fixtures, KB-scale, pinned in tests/fixtures/:
  tiny_mimi.safetensors    : the raw HF state dict (F32).
  tiny_mimi_config.json    : the pico HF MimiConfig.
  tiny_mimi_ref.json       : per input clip, the int code stack [K][T] and
                             the float64 reconstructed waveform.

Coded by Claude (AI).

Usage (from the repo root):
  /home/bpsa/x/bin/python tools/mimi_tiny_fixture.py
Needs torch + transformers + safetensors + numpy.
"""
import json
import os

import numpy as np
import torch
from safetensors.torch import save_file
from transformers import MimiConfig, MimiModel

HERE = os.path.dirname(os.path.abspath(__file__))
FIX = os.path.join(HERE, "..", "tests", "fixtures")

SEED = 1234


def pico_config():
    # ratios [2,2] -> prod 4; sampling_rate 800 -> encodec_frame_rate 200;
    # frame_rate 100 -> the down/up factor encodec_frame_rate/frame_rate == 2,
    # so the downsample conv is kernel 4 / stride 2 (the real 12.5 Hz wiring).
    return MimiConfig(
        sampling_rate=800,
        frame_rate=100,
        audio_channels=1,
        hidden_size=16,
        num_filters=4,
        num_residual_layers=1,
        upsampling_ratios=[2, 2],
        kernel_size=7,
        last_kernel_size=3,
        residual_kernel_size=3,
        dilation_growth_rate=2,
        compress=2,
        trim_right_ratio=1.0,
        codebook_size=8,
        codebook_dim=8,
        vector_quantization_hidden_dimension=8,  # != hidden_size -> in/out proj
        num_quantizers=4,
        num_semantic_quantizers=1,
        use_conv_shortcut=False,
        use_causal_conv=True,
        pad_mode="constant",
        # transformer bottleneck
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=8,
        intermediate_size=32,
        hidden_act="gelu",
        rope_theta=10000.0,
        sliding_window=250,
        norm_eps=1e-5,
        layer_scale_initial_scale=0.01,
        max_position_embeddings=8000,
        attention_bias=False,
        upsample_groups=16,
    )


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    cfg = pico_config()
    model = MimiModel(cfg).eval()

    with torch.no_grad():
        # Re-randomize the codebooks to be well separated (default init leaves
        # embed_sum all-zero / cluster_usage all-one, a degenerate codebook).
        # Pull entries onto a coarse grid so float32 vs float64 cannot flip a
        # nearest-centroid tie.
        for rvq in (model.quantizer.semantic_residual_vector_quantizer,
                    model.quantizer.acoustic_residual_vector_quantizer):
            for layer in rvq.layers:
                cb = layer.codebook
                K, D = cb.embed_sum.shape
                base = torch.randn(K, D, dtype=torch.float64) * 1.5
                base = torch.round(base * 16.0) / 16.0
                # store as embed_sum with cluster_usage 1 so embed == base.
                cb.cluster_usage.fill_(1.0)
                cb.embed_sum.copy_(base.to(cb.embed_sum.dtype))
                cb.initialized.fill_(True)
                cb._embed = None
        # LayerScale defaults to 0.01 which makes the transformer almost a
        # no-op; bump to 1.0 so the RoPE attention / GELU MLP path genuinely
        # contributes and the parity test actually exercises the bottleneck.
        for tr in (model.encoder_transformer, model.decoder_transformer):
            for lyr in tr.layers:
                lyr.self_attn_layer_scale.scale.fill_(1.0)
                lyr.mlp_layer_scale.scale.fill_(1.0)

    # ---- Pinned input clips.
    total_down = int(np.prod(cfg.upsampling_ratios)) * 2  # conv + downsample
    n_frames = 6
    length = n_frames * total_down + 48
    rng = np.random.RandomState(SEED)
    clips = []
    for _ in range(3):
        x = (rng.randn(length).astype(np.float64) * 0.3)
        x = np.round(x * 64.0) / 64.0
        clips.append(x.tolist())

    # ---- float64 oracle round trip.
    model_f64 = MimiModel(cfg).double().eval()
    model_f64.load_state_dict(model.state_dict())
    # clear cached _embed so the f64 copy recomputes from the f64 buffers
    for rvq in (model_f64.quantizer.semantic_residual_vector_quantizer,
                model_f64.quantizer.acoustic_residual_vector_quantizer):
        for layer in rvq.layers:
            layer.codebook._embed = None

    refs = []
    with torch.no_grad():
        for x in clips:
            inp = torch.tensor(x, dtype=torch.float64).view(1, 1, -1)
            enc = model_f64.encode(inp, num_quantizers=cfg.num_quantizers,
                                   return_dict=True)
            codes = enc.audio_codes  # [B, K, T]
            codes_q = codes[0]       # [K, T]
            dec = model_f64.decode(codes, return_dict=True)
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
    save_file(sd, os.path.join(FIX, "tiny_mimi.safetensors"))

    with open(os.path.join(FIX, "tiny_mimi_config.json"), "w") as f:
        json.dump(cfg.to_dict(), f, indent=1, default=str)

    with open(os.path.join(FIX, "tiny_mimi_ref.json"), "w") as f:
        json.dump({
            "num_quantizers": cfg.num_quantizers,
            "num_semantic_quantizers": cfg.num_semantic_quantizers,
            "codebook_size": cfg.codebook_size,
            "codebook_dim": cfg.codebook_dim,
            "hidden_size": cfg.hidden_size,
            "clips": refs,
        }, f)

    st = os.path.getsize(os.path.join(FIX, "tiny_mimi.safetensors"))
    print("wrote tiny_mimi.safetensors %d bytes" % st)
    print("num_quantizers", cfg.num_quantizers,
          "frames", len(refs[0]["codes"][0]),
          "recon len", len(refs[0]["recon"]))


if __name__ == "__main__":
    main()
