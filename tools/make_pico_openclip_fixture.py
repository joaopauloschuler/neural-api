#!/usr/bin/env python3
"""Generate a tiny RANDOM open_clip / timm ViT vision-tower parity fixture for
tests/TestNeuralPretrained.pas (TestOpenClipVisionTowerParity).

No network access and no open_clip dependency: a pico open_clip-style ViT
visual tower is built from a small config and randomly initialized (never
downloaded). The reference forward is a SELF-CONTAINED numpy float64 pre-LN
ViT exactly mirroring open_clip's VisionTransformer (and therefore what the
Pascal BuildOpenClipVisionTower -> BuildClipVisionTower graph computes):

  x = conv1(pixels)                       # bias-free patch embed (visual.conv1)
  x = cat([class_embedding, x], dim=0)    # CLS token prepended
  x = x + positional_embedding            # learned pos table (CLS row 0)
  x = ln_pre(x)
  for blk in resblocks:                   # pre-LN transformer
      x = x + attn_out_proj(MHA(in_proj(ln_1(x))))
      x = x + c_proj(act(c_fc(ln_2(x))))
  x = ln_post(x)                          # applied PER TOKEN (exact for CLS)
  x = x @ proj                            # bias-free image projection

The open_clip key scheme exercised (vs HF/OpenAI vision_model.* keys):
  visual.conv1.weight            (no bias)
  visual.class_embedding         [width]
  visual.positional_embedding    [num_patches+1, width]
  visual.ln_pre.{weight,bias}
  visual.transformer.resblocks.N.ln_1/ln_2.{weight,bias}
  visual.transformer.resblocks.N.attn.in_proj_{weight,bias}   (FUSED qkv)
  visual.transformer.resblocks.N.attn.out_proj.{weight,bias}
  visual.transformer.resblocks.N.mlp.c_fc/c_proj.{weight,bias}
  visual.ln_post.{weight,bias}
  visual.proj                    [width, output_dim]  (used as x @ proj)

Coded by Claude (AI).

Usage (from the repo root):
  /home/bpsa/x/bin/python tools/make_pico_openclip_fixture.py
writes tests/fixtures/tiny_openclip{.safetensors,_config.json,_io.json}.
"""
import json
import os

import numpy as np
from safetensors.numpy import save_file

# ---------------- pico config ----------------
IMAGE = 16
PATCH = 8            # -> 2x2 = 4 patches (+1 CLS = 5 tokens)
WIDTH = 24           # hidden_size
PROJ = 16            # output_dim (image projection)
INTER = 48           # mlp c_fc width
NUM_LAYERS = 2
NUM_HEADS = 3
NUM_CHANNELS = 3
LN_EPS = 1e-5
ACT = "gelu"        # LAION ViT-B/g default; "quick_gelu" for OpenAI-style

GRID = IMAGE // PATCH
NUM_PATCHES = GRID * GRID
NUM_TOKENS = NUM_PATCHES + 1
HEAD_DIM = WIDTH // NUM_HEADS

rng = np.random.default_rng(20260627)


def randn(*shape, scale=0.06):
    return (rng.standard_normal(shape) * scale).astype(np.float32)


# ---------------- random weights (open_clip key scheme) ----------------
W = {}
W["visual.conv1.weight"] = randn(WIDTH, NUM_CHANNELS, PATCH, PATCH)
W["visual.class_embedding"] = randn(WIDTH)
W["visual.positional_embedding"] = randn(NUM_TOKENS, WIDTH)
W["visual.ln_pre.weight"] = (1.0 + randn(WIDTH)).astype(np.float32)
W["visual.ln_pre.bias"] = randn(WIDTH)
for i in range(NUM_LAYERS):
    p = f"visual.transformer.resblocks.{i}."
    W[p + "ln_1.weight"] = (1.0 + randn(WIDTH)).astype(np.float32)
    W[p + "ln_1.bias"] = randn(WIDTH)
    W[p + "attn.in_proj_weight"] = randn(3 * WIDTH, WIDTH)
    W[p + "attn.in_proj_bias"] = randn(3 * WIDTH)
    W[p + "attn.out_proj.weight"] = randn(WIDTH, WIDTH)
    W[p + "attn.out_proj.bias"] = randn(WIDTH)
    W[p + "ln_2.weight"] = (1.0 + randn(WIDTH)).astype(np.float32)
    W[p + "ln_2.bias"] = randn(WIDTH)
    W[p + "mlp.c_fc.weight"] = randn(INTER, WIDTH)
    W[p + "mlp.c_fc.bias"] = randn(INTER)
    W[p + "mlp.c_proj.weight"] = randn(WIDTH, INTER)
    W[p + "mlp.c_proj.bias"] = randn(WIDTH)
W["visual.ln_post.weight"] = (1.0 + randn(WIDTH)).astype(np.float32)
W["visual.ln_post.bias"] = randn(WIDTH)
W["visual.proj"] = randn(WIDTH, PROJ)

os.makedirs("tests/fixtures", exist_ok=True)
save_file(W, "tests/fixtures/tiny_openclip.safetensors")


# ---------------- numpy float64 reference forward ----------------
def layernorm(x, g, b, eps=LN_EPS):
    m = x.mean(-1, keepdims=True)
    v = x.var(-1, keepdims=True)
    return (x - m) / np.sqrt(v + eps) * g + b


def gelu(x):
    from scipy.special import erf  # available in env
    return 0.5 * x * (1.0 + erf(x / np.sqrt(2.0)))


def quick_gelu(x):
    return x * (1.0 / (1.0 + np.exp(-1.702 * x)))


act_fn = gelu if ACT == "gelu" else quick_gelu


def f64(name):
    return W[name].astype(np.float64)


# random pixels (W,H,C in the Pascal volume; here build [C,H,W])
pixels = rng.standard_normal((NUM_CHANNELS, IMAGE, IMAGE)).astype(np.float64)

# bias-free patch conv = stride-patch non-overlapping; flatten row-major (y,x)
conv = f64("visual.conv1.weight")  # [WIDTH, C, PATCH, PATCH]
patches = np.zeros((NUM_PATCHES, WIDTH), dtype=np.float64)
for py in range(GRID):
    for px in range(GRID):
        patch = pixels[:, py * PATCH:(py + 1) * PATCH, px * PATCH:(px + 1) * PATCH]
        idx = py * GRID + px
        for o in range(WIDTH):
            patches[idx, o] = np.sum(conv[o] * patch)

# prepend CLS, add positional table
x = np.zeros((NUM_TOKENS, WIDTH), dtype=np.float64)
x[0] = f64("visual.class_embedding")
x[1:] = patches
x = x + f64("visual.positional_embedding")
x = layernorm(x, f64("visual.ln_pre.weight"), f64("visual.ln_pre.bias"))

for i in range(NUM_LAYERS):
    p = f"visual.transformer.resblocks.{i}."
    # attention
    h = layernorm(x, f64(p + "ln_1.weight"), f64(p + "ln_1.bias"))
    inw = f64(p + "attn.in_proj_weight")  # [3*WIDTH, WIDTH]
    inb = f64(p + "attn.in_proj_bias")
    qkv = h @ inw.T + inb               # [T, 3*WIDTH]
    q, k, v = qkv[:, :WIDTH], qkv[:, WIDTH:2 * WIDTH], qkv[:, 2 * WIDTH:]
    attn_out = np.zeros((NUM_TOKENS, WIDTH), dtype=np.float64)
    scale = 1.0 / np.sqrt(HEAD_DIM)
    for hh in range(NUM_HEADS):
        sl = slice(hh * HEAD_DIM, (hh + 1) * HEAD_DIM)
        qs, ks, vs = q[:, sl], k[:, sl], v[:, sl]
        scores = (qs @ ks.T) * scale
        scores = scores - scores.max(-1, keepdims=True)
        w = np.exp(scores)
        w = w / w.sum(-1, keepdims=True)
        attn_out[:, sl] = w @ vs
    ow = f64(p + "attn.out_proj.weight")
    ob = f64(p + "attn.out_proj.bias")
    x = x + attn_out @ ow.T + ob
    # mlp
    h = layernorm(x, f64(p + "ln_2.weight"), f64(p + "ln_2.bias"))
    fc = h @ f64(p + "mlp.c_fc.weight").T + f64(p + "mlp.c_fc.bias")
    fc = act_fn(fc)
    x = x + fc @ f64(p + "mlp.c_proj.weight").T + f64(p + "mlp.c_proj.bias")

# ln_post per token, then projection (bias-free), per token
x = layernorm(x, f64("visual.ln_post.weight"), f64("visual.ln_post.bias"))
out = x @ f64("visual.proj")  # [T, PROJ]

config = {
    "image_size": IMAGE,
    "patch_size": PATCH,
    "width": WIDTH,
    "output_dim": PROJ,
    "intermediate_size": INTER,
    "num_hidden_layers": NUM_LAYERS,
    "num_attention_heads": NUM_HEADS,
    "num_channels": NUM_CHANNELS,
    "layer_norm_eps": LN_EPS,
    "hidden_act": ACT,
}
with open("tests/fixtures/tiny_openclip_config.json", "w") as f:
    json.dump(config, f, indent=2)

io = {
    "pixels": pixels.tolist(),          # [C, H, W]
    "vision_output": out.tolist(),      # [num_tokens, PROJ]
}
with open("tests/fixtures/tiny_openclip_io.json", "w") as f:
    json.dump(io, f)

print("wrote tiny_openclip fixture: tokens=%d proj=%d act=%s" %
      (NUM_TOKENS, PROJ, ACT))
print("out[0][:4] =", np.round(out[0][:4], 5).tolist())
