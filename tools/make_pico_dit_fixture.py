#!/usr/bin/env python3
"""Generate a tiny RANDOM class-conditional DiT parity fixture for
tests/TestNeuralPretrained.pas (no network access: the model is randomly
initialized from a pico config, never downloaded -- the diffusers package is
NOT installed in this environment, so the reference forward is a self-contained
float64 numpy re-implementation of the canonical DiT forward, Peebles & Xie
2023, "Scalable Diffusion Models with Transformers", arXiv:2212.09748).

One fixture, KB-scale, pinned in tests/fixtures/:

  tiny_dit.* : a class-conditional DiT (the facebook/DiT-XL-2-256 architecture,
      shrunk). Pieces exercised, matching the reference DiT exactly:
        - patchify: a Conv2d(in_channels -> hidden, kernel=stride=patch_size)
          over the (C, H, W) VAE latent -> (Grid*Grid) tokens row-major (h,w),
          + a FIXED 2-D sin-cos position embedding added to the tokens;
        - TimestepEmbedder: sinusoidal(t) [cos|sin half-split, DiT order] ->
          Linear -> SiLU -> Linear, width hidden;
        - LabelEmbedder: an nn.Embedding table (num_classes rows; the extra
          dropout row for classifier-free guidance is NOT exercised here) ->
          width hidden;  c = t_emb + y_emb;
        - N DiTBlocks, each adaLN-Zero:
            (shift_msa,scale_msa,gate_msa,shift_mlp,scale_mlp,gate_mlp)
              = chunk( Linear(SiLU(c)) , 6 )            # adaLN_modulation.1
            x = x + gate_msa * Attn( modulate(LN(x), shift_msa, scale_msa) )
            x = x + gate_mlp * MLP ( modulate(LN(x), shift_mlp, scale_mlp) )
          where modulate(h,shift,scale) = h*(1+scale) + shift, the two LNs are
          elementwise_affine=False (no learned gain/bias), Attn is standard
          multi-head self-attention (1/sqrt(head_dim)), MLP is
          fc2( gelu_tanh( fc1(.) ) ) (DiT uses nn.GELU(approximate='tanh'));
        - FinalLayer: (shift,scale) = chunk(Linear(SiLU(c)),2) (adaLN_modulation
          .1), x = modulate(LN(x),shift,scale), then Linear(hidden ->
          patch^2 * out_channels), then unpatchify -> (C_out, H, W).
      learn_sigma=True: out_channels = 2*in_channels (noise eps in the first
      in_channels, the predicted variance in the second half). The parity test
      checks the FULL out_channels output; the scheduler smoke test slices the
      eps half.

The fixture writes RAW DiT tensor names (no diffusers prefix); the Pascal
importer BuildDiTFromSafeTensors reads exactly these.

Coded by Claude (AI).

Usage (from the repo root):
  /home/bpsa/x/bin/python tools/make_pico_dit_fixture.py
writes tests/fixtures/tiny_dit{.safetensors,_config.json,_io.json}.
"""
import json
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
NUM_CLASSES = 5
FREQ_PERIOD = 10000

RNG = np.random.default_rng(20260614)


def randn(*shape, scale=0.3):
    return (RNG.standard_normal(shape) * scale).astype(np.float64)


# ---------------- 2-D sin-cos position embedding (DiT get_2d_sincos_pos_embed) -
def get_1d_sincos(dim, pos):
    # pos: (M,), dim even -> (M, dim)
    omega = np.arange(dim // 2, dtype=np.float64) / (dim / 2.0)
    omega = 1.0 / (10000 ** omega)
    out = pos[:, None] * omega[None, :]      # (M, dim/2)
    return np.concatenate([np.sin(out), np.cos(out)], axis=1)


def get_2d_sincos_pos_embed(dim, grid_size):
    g = np.arange(grid_size, dtype=np.float64)
    gw, gh = np.meshgrid(g, g)               # DiT: w first then h in meshgrid
    gh = gh.reshape(-1)
    gw = gw.reshape(-1)
    emb_h = get_1d_sincos(dim // 2, gh)
    emb_w = get_1d_sincos(dim // 2, gw)
    return np.concatenate([emb_h, emb_w], axis=1)   # (grid*grid, dim)


POS_EMBED = get_2d_sincos_pos_embed(HIDDEN, GRID)    # (9, 16), fixed buffer


# ---------------- math helpers (float64) ----------------
def silu(x):
    return x / (1.0 + np.exp(-x))


def gelu_tanh(x):
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)))


def layernorm_noaffine(x, eps=1e-6):
    m = x.mean(axis=-1, keepdims=True)
    v = x.var(axis=-1, keepdims=True)
    return (x - m) / np.sqrt(v + eps)


def linear(x, w, b):
    # w: (out, in) torch convention, b: (out,)
    return x @ w.T + b


def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def timestep_embedding(t, dim, max_period=FREQ_PERIOD):
    half = dim // 2
    freqs = np.exp(-np.log(max_period) * np.arange(half, dtype=np.float64) / half)
    args = t * freqs
    # DiT order: cos first, then sin.
    return np.concatenate([np.cos(args), np.sin(args)])


# ---------------- weights ----------------
W = {}
# patch embed conv: (hidden, in_channels, patch, patch) + bias
W["x_embedder.proj.weight"] = randn(HIDDEN, IN_CHANNELS, PATCH, PATCH)
W["x_embedder.proj.bias"] = randn(HIDDEN)
# timestep MLP
W["t_embedder.mlp.0.weight"] = randn(HIDDEN, HIDDEN)
W["t_embedder.mlp.0.bias"] = randn(HIDDEN)
W["t_embedder.mlp.2.weight"] = randn(HIDDEN, HIDDEN)
W["t_embedder.mlp.2.bias"] = randn(HIDDEN)
# label embedding table (NUM_CLASSES rows; no CFG dropout row in the pico)
W["y_embedder.embedding_table.weight"] = randn(NUM_CLASSES, HIDDEN)
# blocks
for i in range(LAYERS):
    p = f"blocks.{i}."
    W[p + "attn.qkv.weight"] = randn(3 * HIDDEN, HIDDEN)
    W[p + "attn.qkv.bias"] = randn(3 * HIDDEN)
    W[p + "attn.proj.weight"] = randn(HIDDEN, HIDDEN)
    W[p + "attn.proj.bias"] = randn(HIDDEN)
    W[p + "mlp.fc1.weight"] = randn(MLP_HIDDEN, HIDDEN)
    W[p + "mlp.fc1.bias"] = randn(MLP_HIDDEN)
    W[p + "mlp.fc2.weight"] = randn(HIDDEN, MLP_HIDDEN)
    W[p + "mlp.fc2.bias"] = randn(HIDDEN)
    # adaLN_modulation.1 : Linear(hidden -> 6*hidden)
    W[p + "adaLN_modulation.1.weight"] = randn(6 * HIDDEN, HIDDEN)
    W[p + "adaLN_modulation.1.bias"] = randn(6 * HIDDEN)
# final layer
W["final_layer.adaLN_modulation.1.weight"] = randn(2 * HIDDEN, HIDDEN)
W["final_layer.adaLN_modulation.1.bias"] = randn(2 * HIDDEN)
W["final_layer.linear.weight"] = randn(PATCH * PATCH * OUT_CHANNELS, HIDDEN)
W["final_layer.linear.bias"] = randn(PATCH * PATCH * OUT_CHANNELS)


# ---------------- forward (float64 oracle) ----------------
def modulate(h, shift, scale):
    return h * (1.0 + scale) + shift


def attention(x, qkvw, qkvb, projw, projb):
    # x: (N, hidden)
    N = x.shape[0]
    qkv = linear(x, qkvw, qkvb)                  # (N, 3*hidden)
    qkv = qkv.reshape(N, 3, HEADS, HEAD_DIM).transpose(1, 2, 0, 3)
    q, k, v = qkv[0], qkv[1], qkv[2]             # (HEADS, N, HEAD_DIM)
    scale = 1.0 / np.sqrt(HEAD_DIM)
    attn = softmax((q @ k.transpose(0, 2, 1)) * scale, axis=-1)  # (HEADS,N,N)
    out = attn @ v                               # (HEADS, N, HEAD_DIM)
    out = out.transpose(1, 0, 2).reshape(N, HIDDEN)
    return linear(out, projw, projb)


def patchify_conv(latent):
    # latent: (IN_CHANNELS, H, W) -> tokens (NUM_PATCHES, HIDDEN), row-major (h,w)
    cw = W["x_embedder.proj.weight"]             # (hidden, in, p, p)
    cb = W["x_embedder.proj.bias"]
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
    # tokens: (NUM_PATCHES, patch*patch*out_ch) -> (out_ch, H, W)
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


def dit_forward(latent, t, y):
    x = patchify_conv(latent) + POS_EMBED        # (N, hidden)
    t_emb = timestep_embedding(t, HIDDEN)
    t_emb = linear(t_emb, W["t_embedder.mlp.0.weight"], W["t_embedder.mlp.0.bias"])
    t_emb = silu(t_emb)
    t_emb = linear(t_emb, W["t_embedder.mlp.2.weight"], W["t_embedder.mlp.2.bias"])
    y_emb = W["y_embedder.embedding_table.weight"][y]
    c = t_emb + y_emb                            # (hidden,)
    for i in range(LAYERS):
        p = f"blocks.{i}."
        mod = linear(silu(c), W[p + "adaLN_modulation.1.weight"],
                     W[p + "adaLN_modulation.1.bias"])    # (6*hidden,)
        sh_msa, sc_msa, g_msa, sh_mlp, sc_mlp, g_mlp = np.split(mod, 6)
        h = modulate(layernorm_noaffine(x), sh_msa, sc_msa)
        a = attention(h, W[p + "attn.qkv.weight"], W[p + "attn.qkv.bias"],
                      W[p + "attn.proj.weight"], W[p + "attn.proj.bias"])
        x = x + g_msa * a
        h = modulate(layernorm_noaffine(x), sh_mlp, sc_mlp)
        m = linear(gelu_tanh(linear(h, W[p + "mlp.fc1.weight"], W[p + "mlp.fc1.bias"])),
                   W[p + "mlp.fc2.weight"], W[p + "mlp.fc2.bias"])
        x = x + g_mlp * m
    mod = linear(silu(c), W["final_layer.adaLN_modulation.1.weight"],
                 W["final_layer.adaLN_modulation.1.bias"])
    sh, sc = np.split(mod, 2)
    x = modulate(layernorm_noaffine(x), sh, sc)
    x = linear(x, W["final_layer.linear.weight"], W["final_layer.linear.bias"])
    return unpatchify(x)                         # (out_ch, H, W)


# ---------------- run a few (latent, t, class) cases ----------------
cases = []
for case in range(3):
    latent = randn(IN_CHANNELS, LATENT_HW, LATENT_HW, scale=1.0)
    t = float(RNG.integers(0, 1000))
    y = int(RNG.integers(0, NUM_CLASSES))
    out = dit_forward(latent, t, y)              # (out_ch, H, W)
    cases.append({
        "latent": latent.reshape(-1).tolist(),
        "t": t,
        "y": y,
        "output": out.reshape(-1).tolist(),      # channel-major (c, h, w)
    })

# ---------------- write ----------------
here = os.path.dirname(os.path.abspath(__file__))
fixtures = os.path.join(here, "..", "tests", "fixtures")
os.makedirs(fixtures, exist_ok=True)

# safetensors expects same dtype; store float32 to match the importer's read path.
tensors = {k: v.astype(np.float32) for k, v in W.items()}
# also store the fixed 2-D sin-cos pos embed as a buffer (DiT keeps it as a
# registered non-persistent buffer; we persist it so the importer need not
# recompute the get_2d_sincos formula).
tensors["pos_embed"] = POS_EMBED.astype(np.float32).reshape(1, NUM_PATCHES, HIDDEN)
save_file(tensors, os.path.join(fixtures, "tiny_dit.safetensors"))

config = {
    "_class_name": "DiT",
    "hidden_size": HIDDEN,
    "depth": LAYERS,
    "num_heads": HEADS,
    "mlp_ratio": MLP_RATIO,
    "patch_size": PATCH,
    "in_channels": IN_CHANNELS,
    "learn_sigma": LEARN_SIGMA,
    "input_size": LATENT_HW,
    "num_classes": NUM_CLASSES,
    "layer_norm_eps": 1e-6,
}
with open(os.path.join(fixtures, "tiny_dit_config.json"), "w") as f:
    json.dump(config, f, indent=2)

with open(os.path.join(fixtures, "tiny_dit_io.json"), "w") as f:
    json.dump({"cases": cases}, f)

print("wrote tiny_dit.safetensors,_config.json,_io.json to", fixtures)
print(f"  hidden={HIDDEN} depth={LAYERS} heads={HEADS} patch={PATCH} "
      f"in_ch={IN_CHANNELS} out_ch={OUT_CHANNELS} grid={GRID} classes={NUM_CLASSES}")
