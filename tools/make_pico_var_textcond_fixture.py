#!/usr/bin/env python3
"""Generate a tiny RANDOM text-conditioned (Infinity-style) VAR parity fixture
for tests/TestNeuralPretrained.pas (TestVARTextCondParity).

No network access: the model is randomly initialized from a pico config, never
downloaded. This is the TEXT-conditioned analogue of make_pico_var_fixture.py
(the class-conditional VAR oracle), the PixArt-style text-cond step: the single
learned CLASS token is replaced by caller-supplied TEXT-ENCODER states. The
reference forward is a self-contained float64 numpy re-implementation of the
canonical VAR next-scale transformer with two text-conditioning pieces added
(Bytedance Infinity, arXiv:2412.04431, "Infinity: Scaling Bitwise Tokenizers
..."):

  (a) CROSS-ATTENTION per block (between the scale-block-causal SELF-attention
      and the FFN): image scale tokens (query) attend to the caption-projected
      text states (key/value); unmodulated residual, no norm, no gate (the
      landed PixArt attn2 recipe). The SELF-attention block-causal mask is
      UNCHANGED.
  (b) POOLED-TEXT adaLN: the conditioning vector c that drives every block's
      adaLN-Zero modulation is the MEAN-POOL of the caption-projected text
      states instead of class_emb[y].

A caption_projection (text_proj.linear_1: text_dim->hidden, GELU(tanh),
text_proj.linear_2: hidden->hidden) maps the text width to the transformer
width.

POOLING NOTE: the Pascal build pools the projected text with TNNetAvgChannel
over the (text_len,1,hidden) stream, which returns sum / text_len**2 (a square
text_len x text_len pool window over a 1-row volume), NOT sum / text_len. The
oracle mirrors that exact scaling so parity is bit-tight; a downstream linear
(ada_lin) absorbs the constant factor, so this is a faithful pooled-text cond.

The fixture writes RAW VAR tensor names; the Pascal importer
BuildVARFromSafeTensors (with text_cond=true) reads exactly these.

Coded by Claude (AI).

Usage (from the repo root):
  /home/bpsa/x/bin/python tools/make_pico_var_textcond_fixture.py
writes tests/fixtures/tiny_var_textcond{.safetensors,_config.json,_io.json}.
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
PATCH_NUMS = [1, 2]                 # pyramid: 1x1, 2x2 -> 1+4 = 5 tokens
NUM_SCALES = len(PATCH_NUMS)
SEQ_LEN = sum(p * p for p in PATCH_NUMS)   # 5
TEXT_DIM = 10                       # supplied text-encoder width
TEXT_LEN = 3                        # number of supplied text tokens

RNG = np.random.default_rng(20260626)


def randn(*shape, scale=0.3):
    return (RNG.standard_normal(shape) * scale).astype(np.float64)


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
W["pos_embed.weight"] = randn(SEQ_LEN, HIDDEN)
W["text_proj.linear_1.weight"] = randn(HIDDEN, TEXT_DIM)
W["text_proj.linear_1.bias"] = randn(HIDDEN)
W["text_proj.linear_2.weight"] = randn(HIDDEN, HIDDEN)
W["text_proj.linear_2.bias"] = randn(HIDDEN)
for i in range(LAYERS):
    p = f"blocks.{i}."
    W[p + "attn.qkv.weight"] = randn(3 * HIDDEN, HIDDEN)
    W[p + "attn.qkv.bias"] = randn(3 * HIDDEN)
    W[p + "attn.proj.weight"] = randn(HIDDEN, HIDDEN)
    W[p + "attn.proj.bias"] = randn(HIDDEN)
    W[p + "cross_attn.to_q.weight"] = randn(HIDDEN, HIDDEN)
    W[p + "cross_attn.to_q.bias"] = randn(HIDDEN)
    W[p + "cross_attn.to_k.weight"] = randn(HIDDEN, HIDDEN)
    W[p + "cross_attn.to_k.bias"] = randn(HIDDEN)
    W[p + "cross_attn.to_v.weight"] = randn(HIDDEN, HIDDEN)
    W[p + "cross_attn.to_v.bias"] = randn(HIDDEN)
    W[p + "cross_attn.to_out.0.weight"] = randn(HIDDEN, HIDDEN)
    W[p + "cross_attn.to_out.0.bias"] = randn(HIDDEN)
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
    si = SCALE_IDS[:, None]
    sj = SCALE_IDS[None, :]
    return sj <= si


MASK = block_causal_mask()       # (SEQ_LEN, SEQ_LEN) boolean


def self_attention(x):
    N = x.shape[0]
    qkv = linear(x, W_QKV, B_QKV)
    qkv = qkv.reshape(N, 3, HEADS, HEAD_DIM).transpose(1, 2, 0, 3)
    q, k, v = qkv[0], qkv[1], qkv[2]
    scale = 1.0 / np.sqrt(HEAD_DIM)
    scores = (q @ k.transpose(0, 2, 1)) * scale
    neg = np.where(MASK[None, :, :], 0.0, -1e9)
    attn = softmax(scores + neg, axis=-1)
    out = attn @ v
    out = out.transpose(1, 0, 2).reshape(N, HIDDEN)
    return linear(out, W_PROJ, B_PROJ)


def cross_attention(x, enc, p):
    # x: (N, hidden) image queries; enc: (TEXT_LEN, hidden) text key/value.
    N = x.shape[0]
    M = enc.shape[0]
    q = linear(x, W[p + "cross_attn.to_q.weight"], W[p + "cross_attn.to_q.bias"])
    k = linear(enc, W[p + "cross_attn.to_k.weight"], W[p + "cross_attn.to_k.bias"])
    v = linear(enc, W[p + "cross_attn.to_v.weight"], W[p + "cross_attn.to_v.bias"])
    q = q.reshape(N, HEADS, HEAD_DIM).transpose(1, 0, 2)   # (H, N, dk)
    k = k.reshape(M, HEADS, HEAD_DIM).transpose(1, 0, 2)   # (H, M, dk)
    v = v.reshape(M, HEADS, HEAD_DIM).transpose(1, 0, 2)
    scale = 1.0 / np.sqrt(HEAD_DIM)
    scores = (q @ k.transpose(0, 2, 1)) * scale            # (H, N, M)
    attn = softmax(scores, axis=-1)
    out = attn @ v                                         # (H, N, dk)
    out = out.transpose(1, 0, 2).reshape(N, HIDDEN)
    return linear(out, W[p + "cross_attn.to_out.0.weight"],
                  W[p + "cross_attn.to_out.0.bias"])


def var_textcond_forward(idx, text):
    # idx: (SEQ_LEN,) codebook indices; text: (TEXT_LEN, TEXT_DIM) text states.
    global W_QKV, B_QKV, W_PROJ, B_PROJ
    x = (W["word_embed.weight"][idx]
         + W["lvl_embed.weight"][SCALE_IDS]
         + W["pos_embed.weight"])               # (N, hidden)
    # caption_projection: Linear -> gelu_tanh -> Linear -> (TEXT_LEN, hidden).
    enc = linear(text, W["text_proj.linear_1.weight"], W["text_proj.linear_1.bias"])
    enc = gelu_tanh(enc)
    enc = linear(enc, W["text_proj.linear_2.weight"], W["text_proj.linear_2.bias"])
    # pooled-text cond: TNNetAvgChannel returns sum / TEXT_LEN**2 (square pool).
    c = enc.sum(axis=0) / (TEXT_LEN * TEXT_LEN)  # (hidden,)
    for i in range(LAYERS):
        p = f"blocks.{i}."
        W_QKV, B_QKV = W[p + "attn.qkv.weight"], W[p + "attn.qkv.bias"]
        W_PROJ, B_PROJ = W[p + "attn.proj.weight"], W[p + "attn.proj.bias"]
        mod = linear(silu(c), W[p + "ada_lin.weight"], W[p + "ada_lin.bias"])
        sh_msa, sc_msa, g_msa, sh_mlp, sc_mlp, g_mlp = np.split(mod, 6)
        # self-attention (scale-block-causal)
        h = modulate(layernorm_noaffine(x), sh_msa, sc_msa)
        a = self_attention(h)
        x = x + g_msa * a
        # cross-attention to text (unmodulated, no norm, no gate)
        x = x + cross_attention(x, enc, p)
        # FFN
        h = modulate(layernorm_noaffine(x), sh_mlp, sc_mlp)
        m = linear(gelu_tanh(linear(h, W[p + "ffn.fc1.weight"], W[p + "ffn.fc1.bias"])),
                   W[p + "ffn.fc2.weight"], W[p + "ffn.fc2.bias"])
        x = x + g_mlp * m
    mod = linear(silu(c), W["head_ada_lin.weight"], W["head_ada_lin.bias"])
    sh, sc = np.split(mod, 2)
    x = modulate(layernorm_noaffine(x), sh, sc)
    logits = linear(x, W["head.weight"], W["head.bias"])   # (N, vocab)
    return logits


# ---------------- run a few (token sequence, text) cases ----------------
cases = []
for case in range(3):
    idx = RNG.integers(0, VOCAB, size=SEQ_LEN).astype(np.int64)
    text = randn(TEXT_LEN, TEXT_DIM, scale=1.0)
    logits = var_textcond_forward(idx, text)     # (SEQ_LEN, vocab)
    cases.append({
        "idx": idx.tolist(),
        "text": text.reshape(-1).tolist(),       # flat (TEXT_LEN, TEXT_DIM)
        "logits": logits.reshape(-1).tolist(),   # token-major (pos, vocab)
    })

# ---------------- write ----------------
here = os.path.dirname(os.path.abspath(__file__))
fixtures = os.path.join(here, "..", "tests", "fixtures")
os.makedirs(fixtures, exist_ok=True)

tensors = {k: v.astype(np.float32) for k, v in W.items()}
save_file(tensors, os.path.join(fixtures, "tiny_var_textcond.safetensors"))

config = {
    "_class_name": "VAR",
    "hidden_size": HIDDEN,
    "depth": LAYERS,
    "num_heads": HEADS,
    "mlp_ratio": MLP_RATIO,
    "vocab_size": VOCAB,
    "patch_nums": PATCH_NUMS,
    "layer_norm_eps": 1e-6,
    "text_cond": True,
    "text_dim": TEXT_DIM,
    "text_seq_len": TEXT_LEN,
}
with open(os.path.join(fixtures, "tiny_var_textcond_config.json"), "w") as f:
    json.dump(config, f, indent=2)

with open(os.path.join(fixtures, "tiny_var_textcond_io.json"), "w") as f:
    json.dump({"cases": cases}, f)

print("wrote tiny_var_textcond.safetensors,_config.json,_io.json to", fixtures)
print(f"  hidden={HIDDEN} depth={LAYERS} heads={HEADS} vocab={VOCAB} "
      f"patch_nums={PATCH_NUMS} seq_len={SEQ_LEN} text_dim={TEXT_DIM} "
      f"text_len={TEXT_LEN}")
