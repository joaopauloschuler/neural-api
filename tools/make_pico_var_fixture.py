#!/usr/bin/env python3
"""Generate a tiny RANDOM class-conditional VAR (Visual AutoRegressive,
next-SCALE prediction) parity fixture for tests/TestNeuralPretrained.pas.

No network access: the model is randomly initialized from a pico config, never
downloaded. The FoundationVision/var package is NOT installed in this
environment, so the reference forward is a self-contained float64 numpy
re-implementation of the canonical VAR transformer forward (Tian et al. 2024,
"Visual Autoregressive Modeling: Scalable Image Generation via Next-Scale
Prediction", arXiv:2404.02905). The VQ tokenizer that produces the multi-scale
code indices is NOT modeled here (it is the landed BuildVqModel family); the
fixture's "input" is already the flattened multi-scale token-INDEX sequence.

VAR forward modeled (the genuinely new pieces over the landed DiT backbone):
  - next-scale embedding: x[i] = word_embed[idx[i]] + lvl_embed[scale[i]]
    + pos_embed[i], where the sequence is the per-scale token maps flattened
    scale-by-scale (patch_nums = [1,2,3] -> 1+4+9 = 14 tokens), lvl_embed has
    ONE learnable row per pyramid level (broadcast to every token of that
    level), pos_embed one row per flattened position;
  - AdaLN class conditioning: a SINGLE class token c = class_emb[y] modulates
    every block via the DiT adaLN-Zero recipe (6 chunks, modulate(h)=h*(1+scale)
    +shift, the two LNs are elementwise_affine=False, per-channel gate);
  - SCALE-BLOCK-CAUSAL attention mask: token i (scale s) attends to ALL tokens
    j with scale[j] <= scale[i] (full attention within and across earlier
    scales, none to later scales) - the GPT causal mask at scale-BLOCK
    granularity.
The head is a final adaLN + Linear(hidden -> vocab) producing per-token logits
(the next-scale-prediction head). The parity test compares ONE scale's logits.

The fixture writes RAW VAR tensor names; the Pascal importer
BuildVARFromSafeTensors reads exactly these.

Coded by Claude (AI).

Usage (from the repo root):
  /home/bpsa/x/bin/python tools/make_pico_var_fixture.py
writes tests/fixtures/tiny_var{.safetensors,_config.json,_io.json}.
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
VOCAB = 12                          # codebook size
NUM_CLASSES = 5
PATCH_NUMS = [1, 2, 3]             # pyramid: 1x1, 2x2, 3x3 -> 1+4+9 = 14 tokens
NUM_SCALES = len(PATCH_NUMS)
SEQ_LEN = sum(p * p for p in PATCH_NUMS)   # 14

RNG = np.random.default_rng(20260615)


def randn(*shape, scale=0.3):
    return (RNG.standard_normal(shape) * scale).astype(np.float64)


# per-token scale id (0 for the 1x1 level, 1 for the 2x2 level, ...).
SCALE_IDS = []
for s, p in enumerate(PATCH_NUMS):
    SCALE_IDS.extend([s] * (p * p))
SCALE_IDS = np.array(SCALE_IDS, dtype=np.int64)   # (SEQ_LEN,)


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
    return x @ w.T + b


def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


# ---------------- weights ----------------
W = {}
W["word_embed.weight"] = randn(VOCAB, HIDDEN)
W["lvl_embed.weight"] = randn(NUM_SCALES, HIDDEN)
W["class_emb.weight"] = randn(NUM_CLASSES, HIDDEN)
W["pos_embed.weight"] = randn(SEQ_LEN, HIDDEN)
for i in range(LAYERS):
    p = f"blocks.{i}."
    W[p + "attn.qkv.weight"] = randn(3 * HIDDEN, HIDDEN)
    W[p + "attn.qkv.bias"] = randn(3 * HIDDEN)
    W[p + "attn.proj.weight"] = randn(HIDDEN, HIDDEN)
    W[p + "attn.proj.bias"] = randn(HIDDEN)
    W[p + "ffn.fc1.weight"] = randn(MLP_HIDDEN, HIDDEN)
    W[p + "ffn.fc1.bias"] = randn(MLP_HIDDEN)
    W[p + "ffn.fc2.weight"] = randn(HIDDEN, MLP_HIDDEN)
    W[p + "ffn.fc2.bias"] = randn(HIDDEN)
    W[p + "ada_lin.weight"] = randn(6 * HIDDEN, HIDDEN)
    W[p + "ada_lin.bias"] = randn(6 * HIDDEN)
W["head_ada_lin.weight"] = randn(2 * HIDDEN, HIDDEN)
W["head_ada_lin.bias"] = randn(2 * HIDDEN)
W["head.weight"] = randn(VOCAB, HIDDEN)
W["head.bias"] = randn(VOCAB)


# ---------------- forward (float64 oracle) ----------------
def modulate(h, shift, scale):
    return h * (1.0 + scale) + shift


def block_causal_mask():
    # mask[i, j] = True if query i may attend key j (scale[j] <= scale[i]).
    si = SCALE_IDS[:, None]
    sj = SCALE_IDS[None, :]
    return sj <= si


MASK = block_causal_mask()       # (SEQ_LEN, SEQ_LEN) boolean


def attention(x, qkvw, qkvb, projw, projb):
    N = x.shape[0]
    qkv = linear(x, qkvw, qkvb)                  # (N, 3*hidden)
    qkv = qkv.reshape(N, 3, HEADS, HEAD_DIM).transpose(1, 2, 0, 3)
    q, k, v = qkv[0], qkv[1], qkv[2]             # (HEADS, N, HEAD_DIM)
    scale = 1.0 / np.sqrt(HEAD_DIM)
    scores = (q @ k.transpose(0, 2, 1)) * scale  # (HEADS, N, N)
    neg = np.where(MASK[None, :, :], 0.0, -1e9)
    attn = softmax(scores + neg, axis=-1)        # (HEADS, N, N)
    out = attn @ v                               # (HEADS, N, HEAD_DIM)
    out = out.transpose(1, 0, 2).reshape(N, HIDDEN)
    return linear(out, projw, projb)


def var_forward(idx, y):
    # idx: (SEQ_LEN,) codebook indices; y: class id.
    x = (W["word_embed.weight"][idx]
         + W["lvl_embed.weight"][SCALE_IDS]
         + W["pos_embed.weight"])               # (N, hidden)
    c = W["class_emb.weight"][y]                 # (hidden,)
    for i in range(LAYERS):
        p = f"blocks.{i}."
        mod = linear(silu(c), W[p + "ada_lin.weight"], W[p + "ada_lin.bias"])
        sh_msa, sc_msa, g_msa, sh_mlp, sc_mlp, g_mlp = np.split(mod, 6)
        h = modulate(layernorm_noaffine(x), sh_msa, sc_msa)
        a = attention(h, W[p + "attn.qkv.weight"], W[p + "attn.qkv.bias"],
                      W[p + "attn.proj.weight"], W[p + "attn.proj.bias"])
        x = x + g_msa * a
        h = modulate(layernorm_noaffine(x), sh_mlp, sc_mlp)
        m = linear(gelu_tanh(linear(h, W[p + "ffn.fc1.weight"], W[p + "ffn.fc1.bias"])),
                   W[p + "ffn.fc2.weight"], W[p + "ffn.fc2.bias"])
        x = x + g_mlp * m
    mod = linear(silu(c), W["head_ada_lin.weight"], W["head_ada_lin.bias"])
    sh, sc = np.split(mod, 2)
    x = modulate(layernorm_noaffine(x), sh, sc)
    logits = linear(x, W["head.weight"], W["head.bias"])   # (N, vocab)
    return logits


# ---------------- run a few (token sequence, class) cases ----------------
cases = []
for case in range(3):
    idx = RNG.integers(0, VOCAB, size=SEQ_LEN).astype(np.int64)
    y = int(RNG.integers(0, NUM_CLASSES))
    logits = var_forward(idx, y)                 # (SEQ_LEN, vocab)
    cases.append({
        "idx": idx.tolist(),
        "y": y,
        "logits": logits.reshape(-1).tolist(),   # token-major (pos, vocab)
    })

# ---------------- write ----------------
here = os.path.dirname(os.path.abspath(__file__))
fixtures = os.path.join(here, "..", "tests", "fixtures")
os.makedirs(fixtures, exist_ok=True)

tensors = {k: v.astype(np.float32) for k, v in W.items()}
save_file(tensors, os.path.join(fixtures, "tiny_var.safetensors"))

config = {
    "_class_name": "VAR",
    "hidden_size": HIDDEN,
    "depth": LAYERS,
    "num_heads": HEADS,
    "mlp_ratio": MLP_RATIO,
    "vocab_size": VOCAB,
    "num_classes": NUM_CLASSES,
    "patch_nums": PATCH_NUMS,
    "layer_norm_eps": 1e-6,
}
with open(os.path.join(fixtures, "tiny_var_config.json"), "w") as f:
    json.dump(config, f, indent=2)

with open(os.path.join(fixtures, "tiny_var_io.json"), "w") as f:
    json.dump({"cases": cases}, f)

print("wrote tiny_var.safetensors,_config.json,_io.json to", fixtures)
print(f"  hidden={HIDDEN} depth={LAYERS} heads={HEADS} vocab={VOCAB} "
      f"classes={NUM_CLASSES} patch_nums={PATCH_NUMS} seq_len={SEQ_LEN}")
