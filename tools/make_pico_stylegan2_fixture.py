#!/usr/bin/env python3
"""Generate a tiny RANDOM StyleGAN2 generator parity fixture for
tests/TestNeuralPretrained.pas.

The official StyleGAN2 weights are not redistributable / not obtainable offline,
so (exactly like the repo's RRDBNet / VAE-decoder pico fixtures) this builds a
CONFIG-FAITHFUL random generator and a self-contained float64 numpy oracle of
the EXACT forward the Pascal importer runs (Karras et al. 2020, "Analyzing and
Improving the Image Quality of StyleGAN", arXiv:1912.04958), shrunk to KB scale.

Pieces, matching BuildStyleGAN2Generator exactly:
  - mapping MLP: z -> [Linear(latent->latent) + LeakyReLU(0.2)] * MAPPING_LAYERS
    -> w  (no normalization in the pico to keep the oracle exact);
  - synthesis from a LEARNED CONSTANT (channels, start, start); per block b
    (size = start * 2^b):
      * b>0: nearest x2 upsample (each pixel -> 2x2 block);
      * conv pass (1 conv on block 0, 2 after): for each conv
          s = affine(w)                              # Linear(latent->channels)
          w'_{o,k,i} = s_i * W_{o,k,i}               # modulate
          d_o = 1/sqrt(sum_{k,i} w'^2 + 1e-8)        # demodulate
          y = conv_SAME(x, d_o * w') + bias_o
          y = y + noise_strength * noise_map         # per-pixel noise injection
          y = LeakyReLU(0.2)(y)
      * toRGB: s=affine(w); 1x1 modulated conv WITHOUT demod + bias -> RGB;
        summed with the nearest-x2-upsampled previous-block RGB (skip tower).
  OUTPUT = final-resolution RGB image (num_out_ch, FinalSize, FinalSize).

The fixture writes RAW tensor names the importer reads (mapping.{l}.*, const,
block.{b}.conv.{c}.{affine.*,weight,bias,noise_strength,noise},
block.{b}.torgb.{affine.*,weight,bias}).

Coded by Claude (AI).

Usage (from repo root):
  /home/bpsa/x/bin/python tools/make_pico_stylegan2_fixture.py
writes tests/fixtures/tiny_stylegan2{.safetensors,_config.json,_io.json}.
"""
import json
import os

import numpy as np
from safetensors.numpy import save_file

# ---------------- pico config ----------------
LATENT = 8
MAPPING_LAYERS = 3      # shrunk from 8; oracle handles any depth
START = 4
NUM_BLOCKS = 3          # 4 -> 8 -> 16
CHANNELS = 6
NUM_OUT_CH = 3
DEMOD_EPS = 1e-8
LRELU = 0.2

RNG = np.random.default_rng(20260615)


def randn(*shape, scale=0.3):
    return (RNG.standard_normal(shape) * scale).astype(np.float64)


# ---------------- math helpers (float64) ----------------
def leaky(x):
    return np.where(x > 0, x, LRELU * x)


def linear(x, w, b):
    return x @ w.T + b


def upsample_nn(img):
    # img: (C, H, W) -> (C, 2H, 2W), each pixel replicated into a 2x2 block.
    c, h, w = img.shape
    out = np.zeros((c, 2 * h, 2 * w), dtype=np.float64)
    for i in range(h):
        for j in range(w):
            out[:, 2 * i:2 * i + 2, 2 * j:2 * j + 2] = img[:, i:i + 1, j:j + 1]
    return out


def mod_conv(x, W, bias, style, demod, k):
    # x: (Cin, H, W); W: (Cout, Cin, k, k); style: (Cin,)
    cout, cin, _, _ = W.shape
    h, wdt = x.shape[1], x.shape[2]
    wp = W * style[None, :, None, None]                  # modulate
    if demod:
        g = np.sum(wp ** 2, axis=(1, 2, 3)) + DEMOD_EPS  # (Cout,)
        d = 1.0 / np.sqrt(g)
        wpp = wp * d[:, None, None, None]
    else:
        wpp = wp
    pad = k // 2
    xp = np.pad(x, ((0, 0), (pad, pad), (pad, pad)))
    out = np.zeros((cout, h, wdt), dtype=np.float64)
    for o in range(cout):
        for oy in range(h):
            for ox in range(wdt):
                acc = 0.0
                for ky in range(k):
                    for kx in range(k):
                        acc += np.sum(wpp[o, :, ky, kx] *
                                      xp[:, oy + ky, ox + kx])
                out[o, oy, ox] = acc + bias[o]
    return out


# ---------------- weights ----------------
W = {}
for l in range(MAPPING_LAYERS):
    W[f"mapping.{l}.weight"] = randn(LATENT, LATENT)
    W[f"mapping.{l}.bias"] = randn(LATENT)

W["const"] = randn(CHANNELS, START, START, scale=1.0)

for b in range(NUM_BLOCKS):
    size = START << b
    n_conv = 1 if b == 0 else 2
    for c in range(n_conv):
        p = f"block.{b}.conv.{c}."
        W[p + "affine.weight"] = randn(CHANNELS, LATENT)
        W[p + "affine.bias"] = randn(CHANNELS) + 1.0          # bias ~1 (style init)
        W[p + "weight"] = randn(CHANNELS, CHANNELS, 3, 3)
        W[p + "bias"] = randn(CHANNELS, scale=0.1)
        W[p + "noise_strength"] = randn(1, scale=0.5)
        W[p + "noise"] = randn(CHANNELS, size, size, scale=1.0)
    p = f"block.{b}.torgb."
    W[p + "affine.weight"] = randn(CHANNELS, LATENT)
    W[p + "affine.bias"] = randn(CHANNELS) + 1.0
    W[p + "weight"] = randn(NUM_OUT_CH, CHANNELS, 1, 1)
    W[p + "bias"] = randn(NUM_OUT_CH, scale=0.1)


# ---------------- forward (float64 oracle) ----------------
def synth(z):
    w = z.copy()
    for l in range(MAPPING_LAYERS):
        w = leaky(linear(w, W[f"mapping.{l}.weight"], W[f"mapping.{l}.bias"]))
    feat = W["const"].copy()                                 # (C, START, START)
    prev_rgb = None
    for b in range(NUM_BLOCKS):
        size = START << b
        if b > 0:
            feat = upsample_nn(feat)
        n_conv = 1 if b == 0 else 2
        for c in range(n_conv):
            p = f"block.{b}.conv.{c}."
            s = linear(w, W[p + "affine.weight"], W[p + "affine.bias"])
            feat = mod_conv(feat, W[p + "weight"], W[p + "bias"], s, True, 3)
            feat = feat + float(W[p + "noise_strength"][0]) * W[p + "noise"]
            feat = leaky(feat)
        p = f"block.{b}.torgb."
        s = linear(w, W[p + "affine.weight"], W[p + "affine.bias"])
        rgb = mod_conv(feat, W[p + "weight"], W[p + "bias"], s, False, 1)
        if prev_rgb is not None:
            rgb = rgb + upsample_nn(prev_rgb)
        prev_rgb = rgb
    return prev_rgb                                          # (NUM_OUT_CH, F, F)


cases = []
for case in range(3):
    z = randn(LATENT, scale=1.0)
    out = synth(z)                                           # (C, H, W)
    cases.append({
        "z": z.reshape(-1).tolist(),
        "output": out.reshape(-1).tolist(),                  # channel-major
    })

# ---------------- write ----------------
here = os.path.dirname(os.path.abspath(__file__))
fixtures = os.path.join(here, "..", "tests", "fixtures")
os.makedirs(fixtures, exist_ok=True)

tensors = {k: v.astype(np.float32) for k, v in W.items()}
save_file(tensors, os.path.join(fixtures, "tiny_stylegan2.safetensors"))

config = {
    "model_type": "stylegan2",
    "latent_dim": LATENT,
    "mapping_layers": MAPPING_LAYERS,
    "start_size": START,
    "num_blocks": NUM_BLOCKS,
    "channels": CHANNELS,
    "num_out_ch": NUM_OUT_CH,
}
with open(os.path.join(fixtures, "tiny_stylegan2_config.json"), "w") as f:
    json.dump(config, f, indent=2)

with open(os.path.join(fixtures, "tiny_stylegan2_io.json"), "w") as f:
    json.dump({"cases": cases}, f)

final = START << (NUM_BLOCKS - 1)
print("wrote tiny_stylegan2.safetensors,_config.json,_io.json to", fixtures)
print(f"  latent={LATENT} mapping={MAPPING_LAYERS} start={START} "
      f"blocks={NUM_BLOCKS} channels={CHANNELS} out_ch={NUM_OUT_CH} "
      f"final={final}x{final}")
