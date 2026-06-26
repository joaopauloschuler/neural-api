#!/usr/bin/env python3
"""Generate a tiny RANDOM DAC (Descript Audio Codec) parity fixture for
tests/TestNeuralPretrained.pas (no network access: the model is randomly
initialized from a PICO config, never downloaded).

DAC (descript/dac_44khz family, model_type "dac", the HF DacModel) is an
RVQGAN-lineage convolutional neural audio CODEC: a waveform -> Snake-conv
ENCODER -> Residual Vector Quantizer (RVQ) -> discrete CODES, and the inverse
RVQ-decode -> ConvTranspose Snake DECODER -> reconstructed waveform. This
fixture pins the WHOLE round trip (waveform -> codes -> waveform) of a pico
instance.

DAC differs from EnCodec/Mimi (the two already-landed codecs) in three ways the
importer must reproduce exactly:

  - SNAKE activations everywhere (x + (1/(alpha+1e-9)) * sin(alpha*x)^2) with a
    LEARNABLE PER-CHANNEL alpha of shape (1, C, 1), not ELU/LeakyReLU;
  - SYMMETRIC, NON-causal conv padding (padding=pad on both sides), unlike
    EnCodec's causal reflect-left-pad / Mimi's causal constant-left-pad;
  - a FACTORIZED, L2-NORMALIZED RVQ: each quantizer projects the latent down to
    a small codebook_dim with a 1x1 in_proj conv, L2-normalizes BOTH the
    projected latent and the codebook, picks the nearest codebook row (= argmax
    cosine), looks up the RAW (un-normalized) codebook embedding, projects it
    back up to hidden_size with a 1x1 out_proj conv, adds it to the running
    quantized sum and subtracts it from the residual (in the full hidden_size
    space). Decode-from-codes skips the lookup: codebook(idx) -> out_proj -> sum.

Fixtures, KB-scale, pinned in tests/fixtures/:

  tiny_dac.safetensors: a DacModel at pico width with every trait above.
  tiny_dac_config.json:  the pico HF DacConfig.
  tiny_dac_ref.json:     the float64 oracle. For each pinned input clip:
        - "codes": the RVQ code stack [num_codebooks][num_frames] (int);
        - "recon": the reconstructed waveform (round-tripped through the full
          encode->decode), the array the Pascal test gates < 1e-4.

Coded by Claude (AI).

Usage (from the repo root):
  /home/bpsa/x/bin/python tools/make_pico_dac_fixture.py
writes tests/fixtures/tiny_dac{.safetensors,_config.json,_ref.json}.
Needs torch + transformers + safetensors + numpy.
"""
import json
import os

import numpy as np
import torch
from safetensors.torch import save_file
from transformers import DacConfig, DacModel

HERE = os.path.dirname(os.path.abspath(__file__))
FIX = os.path.join(HERE, "..", "tests", "fixtures")

SEED = 1234


def pico_config():
    # encoder_hidden_size=2 with downsampling_ratios [2,4] gives:
    #   hidden_size = 2 * 2**2 = 8 (RVQ latent dim),
    #   upsampling_ratios = [4,2], hop_length = 8.
    # n_codebooks=3 codebooks of size 8, projected to codebook_dim=4: a genuine
    # factorized L2-normalized RVQ cascade at pico size. decoder_hidden_size=16
    # so the decoder block channel progression (16 -> 8 -> 4) stays integral.
    return DacConfig(
        encoder_hidden_size=2,
        downsampling_ratios=[2, 4],
        decoder_hidden_size=16,
        n_codebooks=3,
        codebook_size=8,
        codebook_dim=4,
        sampling_rate=8000,
    )


def build():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    cfg = pico_config()
    model = DacModel(cfg).eval()

    # Re-randomize the RVQ codebooks well separated and onto a coarse grid so
    # float32-vs-float64 cannot flip an argmax-cosine tie (DAC L2-normalizes
    # before the nearest-neighbour lookup; the cosine argmax is what the Pascal
    # importer must reproduce EXACTLY). The in_proj/out_proj weights stay random.
    with torch.no_grad():
        for q in model.quantizer.quantizers:
            K, D = q.codebook.weight.shape
            base = torch.randn(K, D, dtype=torch.float64) * 1.5
            base = torch.round(base * 8.0) / 8.0
            q.codebook.weight.copy_(base.to(q.codebook.weight.dtype))

    # ---- Pinned input clips. Length picked so the encoder produces several
    # latent frames after the total downsample (prod(downsampling_ratios)=8).
    total_down = int(np.prod(cfg.downsampling_ratios))
    n_frames = 6
    length = n_frames * total_down + 64
    rng = np.random.RandomState(SEED)
    clips = []
    for _ in range(3):
        x = (rng.randn(length).astype(np.float64) * 0.3)
        x = np.round(x * 64.0) / 64.0   # coarse grid: exactly representable
        clips.append(x.tolist())

    # ---- float64 oracle round trip (HF's own encode/decode in double).
    model_f64 = DacModel(cfg).double().eval()
    model_f64.load_state_dict(model.state_dict())

    refs = []
    with torch.no_grad():
        for x in clips:
            inp = torch.tensor(x, dtype=torch.float64).view(1, 1, -1)
            enc = model_f64.encode(inp)
            codes = enc.audio_codes          # [B, num_codebooks, T]
            codes_q = codes[0]               # [num_codebooks, T]
            # Decode strictly FROM the integer codes (quantizer.from_codes path)
            # so the oracle matches the Pascal holder's codes->waveform decode.
            dec = model_f64.decode(audio_codes=codes)
            recon = dec.audio_values[0]      # [L]
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
    save_file(sd, os.path.join(FIX, "tiny_dac.safetensors"))

    ref_obj = {
        "num_codebooks": cfg.n_codebooks,
        "codebook_size": cfg.codebook_size,
        "codebook_dim": cfg.codebook_dim,
        "hidden_size": cfg.hidden_size,
        "clips": refs,
    }
    with open(os.path.join(FIX, "tiny_dac_config.json"), "w") as f:
        json.dump(cfg.to_dict(), f, indent=1, default=str)
    with open(os.path.join(FIX, "tiny_dac_ref.json"), "w") as f:
        json.dump(ref_obj, f)

    st = os.path.getsize(os.path.join(FIX, "tiny_dac.safetensors"))
    print("wrote tiny_dac.safetensors %d bytes" % st)
    print("  hidden_size", cfg.hidden_size,
          "num codebooks", cfg.n_codebooks,
          "codebook_dim", cfg.codebook_dim,
          "frames", len(refs[0]["codes"][0]),
          "recon len", len(refs[0]["recon"]))


if __name__ == "__main__":
    build()
