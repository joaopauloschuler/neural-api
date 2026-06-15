#!/usr/bin/env python3
"""Generate a tiny RANDOM PixArt-alpha (text-conditioned DiT) parity fixture for
tests/TestNeuralPretrained.pas (no network access: the model is randomly
initialized from a pico config, never downloaded -- the diffusers package is
NOT installed in this environment, so the reference forward is a self-contained
float64 numpy re-implementation that EXACTLY mirrors the diffusers
PixArtTransformer2DModel forward, Chen et al. 2023, "PixArt-alpha: Fast Training
of Diffusion Transformer for Photorealistic Text-to-Image Synthesis",
arXiv:2310.00426).

PixArt differs from the landed class-conditional DiT in exactly three pieces:
  (a) CONDITIONING SOURCE: no class-label table; the model is conditioned on
      caller-supplied T5 encoder_hidden_states (B, L, t5_dim), projected by a
      caption_projection (Linear -> GELU(tanh) -> Linear) to width hidden.
  (b) CROSS-ATTENTION per block, between the (DiT) self-attention and the FFN:
      image tokens (query) attend to the projected text tokens (key/value).
      PixArt's cross-attention is NOT adaLN-modulated and has NO preceding
      LayerNorm (diffusers ada_norm_single passes hidden_states straight in),
      and NO output gate.
  (c) SHARED adaLN-single: ONE global timestep-modulation MLP (adaln_single)
      produces a 6*hidden vector; each block ADDS its own learnable
      scale_shift_table[6, hidden] before chunking into the six
      shift/scale/gate params. The final layer uses a top-level
      scale_shift_table[2, hidden] + embedded_timestep (the hidden-width vector
      BEFORE the 6*hidden linear).

Exact diffusers details mirrored here:
  - timestep: get_timestep_embedding(t, 256, flip_sin_to_cos=True,
    downscale_freq_shift=0) -> TimestepEmbedding(linear_1 256->hidden, SiLU,
    linear_2 hidden->hidden) = embedded_timestep; adaln = linear(SiLU(
    embedded_timestep)) -> 6*hidden;
  - self-attn (attn1): separate to_q/to_k/to_v (hidden->hidden) + to_out.0
    (hidden->hidden), scale 1/sqrt(head_dim);
  - cross-attn (attn2): to_q from image tokens, to_k/to_v from projected text
    tokens, to_out.0; scale 1/sqrt(head_dim);
  - FFN (ff): GEGLU net.0.proj (hidden -> 2*mlp_hidden), x*gelu_erf(gate),
    net.2 (mlp_hidden -> hidden). diffusers GEGLU uses ERF gelu (F.gelu
    default), NOT tanh;
  - LayerNorms norm1/norm2/norm_out are elementwise_affine=False (no gain/bias);
  - patch embed + fixed 2-D sin-cos pos embed: identical to the DiT fixture.

learn_sigma=True: out_channels = 2*in_channels (eps in the first in_channels,
the predicted variance in the second half). The parity test checks the FULL
out_channels output. v1 is the plain PixArt-alpha 512 (no micro-conditioning:
no resolution/aspect_ratio embeddings).

The fixture writes RAW diffusers PixArtTransformer2DModel tensor names; the
Pascal importer BuildPixArtFromSafeTensors reads exactly these.

Coded by Claude (AI).

Usage (from the repo root):
  /home/bpsa/x/bin/python tools/make_pico_pixart_fixture.py
writes tests/fixtures/tiny_pixart{.safetensors,_config.json,_io.json}.
"""
import json
import math
import os

import numpy as np
from safetensors.numpy import save_file

# ---------------- pico config ----------------
HIDDEN = 16
LAYERS = 2
HEADS = 2
HEAD_DIM = HIDDEN // HEADS          # 8
MLP_RATIO = 4
MLP_HIDDEN = HIDDEN * MLP_RATIO     # 64
PATCH = 2
IN_CHANNELS = 4
LEARN_SIGMA = True
OUT_CHANNELS = IN_CHANNELS * 2 if LEARN_SIGMA else IN_CHANNELS
GRID = 3                            # latent is (IN_CHANNELS, GRID*PATCH, GRID*PATCH)
LATENT_HW = GRID * PATCH           # 6
NUM_PATCHES = GRID * GRID          # 9
CAPTION_CHANNELS = 12              # T5 hidden (caption_projection in_features)
TEXT_SEQ_LEN = 5                   # number of T5 text tokens
TIME_PROJ_DIM = 256               # PixArt Timesteps num_channels (FIXED, diffusers)
FREQ_PERIOD = 10000

RNG = np.random.default_rng(20260615)


def randn(*shape, scale=0.2):
    return (RNG.standard_normal(shape) * scale).astype(np.float64)


# ---------------- 2-D sin-cos position embedding (DiT/PixArt get_2d_sincos) ----
def get_1d_sincos(dim, pos):
    omega = np.arange(dim // 2, dtype=np.float64) / (dim / 2.0)
    omega = 1.0 / (10000 ** omega)
    out = pos[:, None] * omega[None, :]
    return np.concatenate([np.sin(out), np.cos(out)], axis=1)


def get_2d_sincos_pos_embed(dim, grid_size):
    g = np.arange(grid_size, dtype=np.float64)
    gw, gh = np.meshgrid(g, g)               # w first then h
    gh = gh.reshape(-1)
    gw = gw.reshape(-1)
    emb_h = get_1d_sincos(dim // 2, gh)
    emb_w = get_1d_sincos(dim // 2, gw)
    return np.concatenate([emb_h, emb_w], axis=1)   # (grid*grid, dim)


POS_EMBED = get_2d_sincos_pos_embed(HIDDEN, GRID)    # (9, 16)


# ---------------- math helpers (float64) ----------------
def silu(x):
    return x / (1.0 + np.exp(-x))


def gelu_tanh(x):
    return 0.5 * x * (1.0 + np.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x ** 3)))


def gelu_erf(x):
    from math import erf
    vec = np.vectorize(lambda v: 0.5 * v * (1.0 + erf(v / math.sqrt(2.0))))
    return vec(x)


def layernorm_noaffine(x, eps=1e-6):
    m = x.mean(axis=-1, keepdims=True)
    v = x.var(axis=-1, keepdims=True)
    return (x - m) / np.sqrt(v + eps)


def linear(x, w, b):
    return x @ w.T + b


def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def get_timestep_embedding(t, dim, flip_sin_to_cos=True, downscale_freq_shift=0.0,
                           max_period=FREQ_PERIOD):
    half = dim // 2
    exponent = -math.log(max_period) * np.arange(half, dtype=np.float64)
    exponent = exponent / (half - downscale_freq_shift)
    emb = np.exp(exponent)
    emb = t * emb                                  # (half,)
    emb = np.concatenate([np.sin(emb), np.cos(emb)])
    if flip_sin_to_cos:
        emb = np.concatenate([emb[half:], emb[:half]])
    return emb


# ---------------- weights ----------------
W = {}
# patch embed conv: (hidden, in_channels, patch, patch) + bias
W["pos_embed.proj.weight"] = randn(HIDDEN, IN_CHANNELS, PATCH, PATCH)
W["pos_embed.proj.bias"] = randn(HIDDEN)
# adaln_single: time_proj (no weights) -> timestep_embedder.linear_1/2 -> linear
W["adaln_single.emb.timestep_embedder.linear_1.weight"] = randn(HIDDEN, TIME_PROJ_DIM)
W["adaln_single.emb.timestep_embedder.linear_1.bias"] = randn(HIDDEN)
W["adaln_single.emb.timestep_embedder.linear_2.weight"] = randn(HIDDEN, HIDDEN)
W["adaln_single.emb.timestep_embedder.linear_2.bias"] = randn(HIDDEN)
W["adaln_single.linear.weight"] = randn(6 * HIDDEN, HIDDEN)
W["adaln_single.linear.bias"] = randn(6 * HIDDEN)
# caption projection: linear_1 (t5 -> hidden) -> gelu_tanh -> linear_2 (hidden->hidden)
W["caption_projection.linear_1.weight"] = randn(HIDDEN, CAPTION_CHANNELS)
W["caption_projection.linear_1.bias"] = randn(HIDDEN)
W["caption_projection.linear_2.weight"] = randn(HIDDEN, HIDDEN)
W["caption_projection.linear_2.bias"] = randn(HIDDEN)
# blocks
for i in range(LAYERS):
    p = f"transformer_blocks.{i}."
    # per-block scale_shift_table [6, hidden]
    W[p + "scale_shift_table"] = randn(6, HIDDEN)
    # self-attention (attn1)
    W[p + "attn1.to_q.weight"] = randn(HIDDEN, HIDDEN)
    W[p + "attn1.to_q.bias"] = randn(HIDDEN)
    W[p + "attn1.to_k.weight"] = randn(HIDDEN, HIDDEN)
    W[p + "attn1.to_k.bias"] = randn(HIDDEN)
    W[p + "attn1.to_v.weight"] = randn(HIDDEN, HIDDEN)
    W[p + "attn1.to_v.bias"] = randn(HIDDEN)
    W[p + "attn1.to_out.0.weight"] = randn(HIDDEN, HIDDEN)
    W[p + "attn1.to_out.0.bias"] = randn(HIDDEN)
    # cross-attention (attn2): K/V from caption (hidden after projection)
    W[p + "attn2.to_q.weight"] = randn(HIDDEN, HIDDEN)
    W[p + "attn2.to_q.bias"] = randn(HIDDEN)
    W[p + "attn2.to_k.weight"] = randn(HIDDEN, HIDDEN)
    W[p + "attn2.to_k.bias"] = randn(HIDDEN)
    W[p + "attn2.to_v.weight"] = randn(HIDDEN, HIDDEN)
    W[p + "attn2.to_v.bias"] = randn(HIDDEN)
    W[p + "attn2.to_out.0.weight"] = randn(HIDDEN, HIDDEN)
    W[p + "attn2.to_out.0.bias"] = randn(HIDDEN)
    # feed-forward (GEGLU): net.0.proj (hidden -> 2*mlp), net.2 (mlp -> hidden)
    W[p + "ff.net.0.proj.weight"] = randn(2 * MLP_HIDDEN, HIDDEN)
    W[p + "ff.net.0.proj.bias"] = randn(2 * MLP_HIDDEN)
    W[p + "ff.net.2.weight"] = randn(HIDDEN, MLP_HIDDEN)
    W[p + "ff.net.2.bias"] = randn(HIDDEN)
# final layer: top-level scale_shift_table [2, hidden] + proj_out
W["scale_shift_table"] = randn(2, HIDDEN)
W["proj_out.weight"] = randn(PATCH * PATCH * OUT_CHANNELS, HIDDEN)
W["proj_out.bias"] = randn(PATCH * PATCH * OUT_CHANNELS)


# ---------------- forward (float64 oracle) ----------------
def modulate(h, shift, scale):
    return h * (1.0 + scale) + shift


def mha(q_in, kv_in, prefix):
    # q_in: (Nq, hidden), kv_in: (Nkv, hidden)
    Nq = q_in.shape[0]
    Nkv = kv_in.shape[0]
    q = linear(q_in, W[prefix + "to_q.weight"], W[prefix + "to_q.bias"])
    k = linear(kv_in, W[prefix + "to_k.weight"], W[prefix + "to_k.bias"])
    v = linear(kv_in, W[prefix + "to_v.weight"], W[prefix + "to_v.bias"])
    q = q.reshape(Nq, HEADS, HEAD_DIM).transpose(1, 0, 2)     # (H, Nq, hd)
    k = k.reshape(Nkv, HEADS, HEAD_DIM).transpose(1, 0, 2)
    v = v.reshape(Nkv, HEADS, HEAD_DIM).transpose(1, 0, 2)
    scale = 1.0 / math.sqrt(HEAD_DIM)
    attn = softmax((q @ k.transpose(0, 2, 1)) * scale, axis=-1)
    out = attn @ v                                            # (H, Nq, hd)
    out = out.transpose(1, 0, 2).reshape(Nq, HIDDEN)
    return linear(out, W[prefix + "to_out.0.weight"], W[prefix + "to_out.0.bias"])


def feedforward(x, p):
    proj = linear(x, W[p + "ff.net.0.proj.weight"], W[p + "ff.net.0.proj.bias"])
    a, gate = np.split(proj, 2, axis=-1)
    h = a * gelu_erf(gate)
    return linear(h, W[p + "ff.net.2.weight"], W[p + "ff.net.2.bias"])


def patchify_conv(latent):
    cw = W["pos_embed.proj.weight"]
    cb = W["pos_embed.proj.bias"]
    toks = np.zeros((NUM_PATCHES, HIDDEN), dtype=np.float64)
    idx = 0
    for gh in range(GRID):
        for gw in range(GRID):
            patch = latent[:, gh * PATCH:(gh + 1) * PATCH, gw * PATCH:(gw + 1) * PATCH]
            for o in range(HIDDEN):
                toks[idx, o] = np.sum(cw[o] * patch) + cb[o]
            idx += 1
    return toks


def unpatchify(tokens):
    c = OUT_CHANNELS
    p = PATCH
    img = np.zeros((c, GRID * p, GRID * p), dtype=np.float64)
    idx = 0
    for gh in range(GRID):
        for gw in range(GRID):
            blk = tokens[idx].reshape(p, p, c)   # (ph, pw, c)
            for ph in range(p):
                for pw in range(p):
                    img[:, gh * p + ph, gw * p + pw] = blk[ph, pw, :]
            idx += 1
    return img


def pixart_forward(latent, t, text_states):
    x = patchify_conv(latent) + POS_EMBED        # (N, hidden)
    # timestep -> embedded_timestep -> adaln (6*hidden)
    te = get_timestep_embedding(t, TIME_PROJ_DIM)
    te = linear(te, W["adaln_single.emb.timestep_embedder.linear_1.weight"],
                W["adaln_single.emb.timestep_embedder.linear_1.bias"])
    te = silu(te)
    embedded_t = linear(te, W["adaln_single.emb.timestep_embedder.linear_2.weight"],
                        W["adaln_single.emb.timestep_embedder.linear_2.bias"])
    adaln = linear(silu(embedded_t), W["adaln_single.linear.weight"],
                   W["adaln_single.linear.bias"])              # (6*hidden,)
    # caption projection (T5 states -> hidden)
    enc = linear(text_states, W["caption_projection.linear_1.weight"],
                 W["caption_projection.linear_1.bias"])
    enc = gelu_tanh(enc)
    enc = linear(enc, W["caption_projection.linear_2.weight"],
                 W["caption_projection.linear_2.bias"])        # (L, hidden)
    for i in range(LAYERS):
        p = f"transformer_blocks.{i}."
        mod = (W[p + "scale_shift_table"] + adaln.reshape(6, HIDDEN))  # (6, hidden)
        sh_msa, sc_msa, g_msa, sh_mlp, sc_mlp, g_mlp = mod
        # self-attention (modulated norm1)
        h = modulate(layernorm_noaffine(x), sh_msa, sc_msa)
        x = x + g_msa * mha(h, h, p + "attn1.")
        # cross-attention (NO norm, NO gate, NO modulation)
        x = x + mha(x, enc, p + "attn2.")
        # feed-forward (modulated norm2)
        h = modulate(layernorm_noaffine(x), sh_mlp, sc_mlp)
        x = x + g_mlp * feedforward(h, p)
    # final layer: top-level scale_shift_table[2,hidden] + embedded_timestep,
    # diffusers: (scale_shift_table[None] + embedded_timestep[:, None]).chunk(2)
    final_mod = W["scale_shift_table"] + embedded_t.reshape(1, HIDDEN)  # (2, hidden)
    sh, sc = final_mod[0], final_mod[1]
    x = modulate(layernorm_noaffine(x), sh, sc)
    x = linear(x, W["proj_out.weight"], W["proj_out.bias"])
    return unpatchify(x)


# ---------------- run a few (latent, t, text) cases ----------------
cases = []
for case in range(3):
    latent = randn(IN_CHANNELS, LATENT_HW, LATENT_HW, scale=1.0)
    # t kept in a moderate range: the framework's float32 sinusoidal timestep
    # table loses precision at large t*freq angles, which would push the deep
    # transformer past the 1e-4 parity gate with float32 accumulation.
    t = float(RNG.integers(0, 100))
    text = randn(TEXT_SEQ_LEN, CAPTION_CHANNELS, scale=1.0)
    out = pixart_forward(latent, t, text)        # (out_ch, H, W)
    cases.append({
        "latent": latent.reshape(-1).tolist(),       # (C, H, W)
        "t": t,
        "text": text.reshape(-1).tolist(),           # (L, t5_dim)
        "output": out.reshape(-1).tolist(),          # (out_ch, H, W)
    })

# ---------------- write ----------------
here = os.path.dirname(os.path.abspath(__file__))
fixtures = os.path.join(here, "..", "tests", "fixtures")
os.makedirs(fixtures, exist_ok=True)

tensors = {k: v.astype(np.float32) for k, v in W.items()}
# fixed 2-D sin-cos pos embed buffer (PixArt PatchEmbed keeps it as a buffer).
tensors["pos_embed.pos_embed"] = POS_EMBED.astype(np.float32).reshape(1, NUM_PATCHES, HIDDEN)
save_file(tensors, os.path.join(fixtures, "tiny_pixart.safetensors"))

config = {
    "_class_name": "PixArtTransformer2DModel",
    "num_attention_heads": HEADS,
    "attention_head_dim": HEAD_DIM,
    "num_layers": LAYERS,
    "patch_size": PATCH,
    "in_channels": IN_CHANNELS,
    "out_channels": OUT_CHANNELS,
    "sample_size": LATENT_HW,
    "cross_attention_dim": HIDDEN,
    "caption_channels": CAPTION_CHANNELS,
    "norm_elementwise_affine": False,
    "norm_eps": 1e-6,
    "use_additional_conditions": False,
}
with open(os.path.join(fixtures, "tiny_pixart_config.json"), "w") as f:
    json.dump(config, f, indent=2)

with open(os.path.join(fixtures, "tiny_pixart_io.json"), "w") as f:
    json.dump({"cases": cases, "text_seq_len": TEXT_SEQ_LEN,
               "caption_channels": CAPTION_CHANNELS}, f)

print("wrote tiny_pixart.safetensors,_config.json,_io.json to", fixtures)
print(f"  hidden={HIDDEN} layers={LAYERS} heads={HEADS} patch={PATCH} "
      f"in_ch={IN_CHANNELS} out_ch={OUT_CHANNELS} grid={GRID} "
      f"t5_dim={CAPTION_CHANNELS} text_len={TEXT_SEQ_LEN}")
