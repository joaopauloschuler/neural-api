#!/usr/bin/env python3
"""Tiny Llama fixture generator + independent reference forward (the oracle).

Coded by Claude (AI).

Generates, using ONLY the Python standard library (no numpy, no torch):
  tests/fixtures/tiny_llama.safetensors  - a random Llama-shaped checkpoint
      (layers=2, heads=2, kv_heads=1, hidden=8, intermediate=12, vocab=11,
      max_pos=16, rope_theta=10000, rms eps=1e-5, UNTIED lm_head) in the
      exact HF layout: "model."-prefixed nn.Linear [out, in] weights, no
      biases, plus a dummy ignorable "rotary_emb.inv_freq" buffer.
  tests/fixtures/tiny_llama_config.json  - the matching HF config.json.
  tests/fixtures/tiny_llama_logits.json  - reference logits for a few fixed
      token sequences, computed by the Llama forward implemented below
      straight from the math (per-token RMSNorm + rotary GQA with HF's
      rotate_half first-half/second-half pair layout + SwiGLU MLP + untied
      LM head). This forward shares NO code with the Pascal library: it is
      the oracle the importer is verified against
      (tests/TestNeuralPretrained.pas).

Run from the repository root:  python3 tools/llama_tiny_fixture.py
"""
import json
import math
import os
import random
import struct

N_LAYER = 2
N_HEAD = 2
N_KV_HEAD = 1
HIDDEN = 8
HEAD_DIM = HIDDEN // N_HEAD
KV_WIDTH = N_KV_HEAD * HEAD_DIM
INTERMEDIATE = 12
MAX_POS = 16
VOCAB = 11
ROPE_THETA = 10000.0
RMS_EPS = 1e-5
SEED = 20260611

rng = random.Random(SEED)


def randn_matrix(rows, cols, scale):
    return [[rng.gauss(0.0, scale) for _ in range(cols)] for _ in range(rows)]


# ---------------------------------------------------------------------------
# Random Llama-shaped weights. Every projection is nn.Linear: shape
# [out, in], y = x . W^T, NO bias. RMSNorm gains start near 1.
# ---------------------------------------------------------------------------
def make_weights():
    w = {}
    w["model.embed_tokens.weight"] = randn_matrix(VOCAB, HIDDEN, 0.4)
    for b in range(N_LAYER):
        p = "model.layers.%d." % b
        w[p + "input_layernorm.weight"] = \
            [1.0 + rng.gauss(0.0, 0.1) for _ in range(HIDDEN)]
        w[p + "self_attn.q_proj.weight"] = randn_matrix(HIDDEN, HIDDEN, 0.3)
        w[p + "self_attn.k_proj.weight"] = randn_matrix(KV_WIDTH, HIDDEN, 0.3)
        w[p + "self_attn.v_proj.weight"] = randn_matrix(KV_WIDTH, HIDDEN, 0.3)
        w[p + "self_attn.o_proj.weight"] = randn_matrix(HIDDEN, HIDDEN, 0.3)
        w[p + "post_attention_layernorm.weight"] = \
            [1.0 + rng.gauss(0.0, 0.1) for _ in range(HIDDEN)]
        w[p + "mlp.gate_proj.weight"] = randn_matrix(INTERMEDIATE, HIDDEN, 0.3)
        w[p + "mlp.up_proj.weight"] = randn_matrix(INTERMEDIATE, HIDDEN, 0.3)
        w[p + "mlp.down_proj.weight"] = randn_matrix(HIDDEN, INTERMEDIATE, 0.3)
    w["model.norm.weight"] = [1.0 + rng.gauss(0.0, 0.1) for _ in range(HIDDEN)]
    w["lm_head.weight"] = randn_matrix(VOCAB, HIDDEN, 0.4)  # UNTIED
    return w


# ---------------------------------------------------------------------------
# safetensors writer (8-byte LE uint64 header length, JSON header, raw data).
# ---------------------------------------------------------------------------
def flatten(t):
    if t and isinstance(t[0], list):
        out = []
        for row in t:
            out.extend(flatten(row))
        return out
    return list(t)


def shape_of(t):
    s = []
    x = t
    while isinstance(x, list):
        s.append(len(x))
        x = x[0]
    return s


def write_safetensors(path, tensors):
    header = {}
    blobs = []
    offset = 0
    for name in tensors:
        flat = flatten(tensors[name])
        blob = struct.pack("<%df" % len(flat), *flat)
        header[name] = {
            "dtype": "F32",
            "shape": shape_of(tensors[name]),
            "data_offsets": [offset, offset + len(blob)],
        }
        offset += len(blob)
        blobs.append(blob)
    hjson = json.dumps(header, separators=(",", ":")).encode("utf-8")
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hjson)))
        f.write(hjson)
        for blob in blobs:
            f.write(blob)


# ---------------------------------------------------------------------------
# Independent Llama forward (the oracle). HF conventions throughout:
# per-token RMSNorm, rotate_half RoPE (pairs (i, i+hd/2)), GQA, SwiGLU.
# ---------------------------------------------------------------------------
def rms_norm(x, gain, eps=RMS_EPS):
    n = len(x)
    ms = sum(v * v for v in x) / n
    inv = 1.0 / math.sqrt(ms + eps)
    return [x[i] * inv * gain[i] for i in range(n)]


def linear(x, w):
    """nn.Linear: y_j = sum_i x_i * w[j][i] (w stored [out, in], no bias)."""
    return [sum(x[i] * row[i] for i in range(len(x))) for row in w]


def rope_rotate_half(vec, pos):
    """HF Llama RoPE on one head vector (len HEAD_DIM): pair (i, i+hd/2)
    rotated by pos * theta^(-2i/hd)."""
    hd = len(vec)
    half = hd // 2
    out = [0.0] * hd
    for i in range(half):
        freq = ROPE_THETA ** (-2.0 * i / hd)
        c = math.cos(pos * freq)
        s = math.sin(pos * freq)
        out[i] = vec[i] * c - vec[i + half] * s
        out[i + half] = vec[i + half] * c + vec[i] * s
    return out


def silu(v):
    return v / (1.0 + math.exp(-v))


def softmax(scores):
    m = max(scores)
    exps = [math.exp(s - m) for s in scores]
    z = sum(exps)
    return [e / z for e in exps]


def attention(xs, w, prefix):
    """Causal rotary GQA over the whole (already-normalized) sequence xs."""
    seq = len(xs)
    group = N_HEAD // N_KV_HEAD
    qs = [linear(x, w[prefix + "self_attn.q_proj.weight"]) for x in xs]
    ks = [linear(x, w[prefix + "self_attn.k_proj.weight"]) for x in xs]
    vs = [linear(x, w[prefix + "self_attn.v_proj.weight"]) for x in xs]
    # Rotate every head slice of q and k by its position.
    for pos in range(seq):
        for h in range(N_HEAD):
            qs[pos][h * HEAD_DIM:(h + 1) * HEAD_DIM] = rope_rotate_half(
                qs[pos][h * HEAD_DIM:(h + 1) * HEAD_DIM], pos)
        for h in range(N_KV_HEAD):
            ks[pos][h * HEAD_DIM:(h + 1) * HEAD_DIM] = rope_rotate_half(
                ks[pos][h * HEAD_DIM:(h + 1) * HEAD_DIM], pos)
    out = []
    for i in range(seq):
        merged = [0.0] * HIDDEN
        for h in range(N_HEAD):
            kvh = h // group
            q = qs[i][h * HEAD_DIM:(h + 1) * HEAD_DIM]
            scores = []
            for j in range(i + 1):  # causal: keys j <= i
                k = ks[j][kvh * HEAD_DIM:(kvh + 1) * HEAD_DIM]
                scores.append(sum(q[d] * k[d] for d in range(HEAD_DIM)) /
                              math.sqrt(HEAD_DIM))
            probs = softmax(scores)
            for j in range(i + 1):
                v = vs[j][kvh * HEAD_DIM:(kvh + 1) * HEAD_DIM]
                for d in range(HEAD_DIM):
                    merged[h * HEAD_DIM + d] += probs[j] * v[d]
        out.append(linear(merged, w[prefix + "self_attn.o_proj.weight"]))
    return out


def llama_forward(tokens, w):
    """Returns logits[position][vocab] for every position."""
    xs = [list(w["model.embed_tokens.weight"][tok]) for tok in tokens]
    for b in range(N_LAYER):
        p = "model.layers.%d." % b
        normed = [rms_norm(x, w[p + "input_layernorm.weight"]) for x in xs]
        att = attention(normed, w, p)
        xs = [[xs[i][c] + att[i][c] for c in range(HIDDEN)]
              for i in range(len(xs))]
        mlp = []
        for x in xs:
            hcur = rms_norm(x, w[p + "post_attention_layernorm.weight"])
            gate = [silu(v) for v in linear(hcur, w[p + "mlp.gate_proj.weight"])]
            up = linear(hcur, w[p + "mlp.up_proj.weight"])
            fused = [gate[i] * up[i] for i in range(INTERMEDIATE)]
            mlp.append(linear(fused, w[p + "mlp.down_proj.weight"]))
        xs = [[xs[i][c] + mlp[i][c] for c in range(HIDDEN)]
              for i in range(len(xs))]
    logits = []
    for x in xs:
        hcur = rms_norm(x, w["model.norm.weight"])
        logits.append(linear(hcur, w["lm_head.weight"]))
    return logits


def round_f32(t):
    """Round-trips values through float32 so the oracle uses EXACTLY the
    weights stored in the (f32) safetensors file."""
    if isinstance(t, list):
        return [round_f32(v) for v in t]
    return struct.unpack("<f", struct.pack("<f", t))[0]


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    fixdir = os.path.join(root, "tests", "fixtures")
    os.makedirs(fixdir, exist_ok=True)

    w = {name: round_f32(t) for name, t in make_weights().items()}

    # Serialize, adding the ignorable inv_freq buffer some older HF exports
    # carry (the importer must skip it without complaint).
    tensors = dict(w)
    tensors["model.layers.0.self_attn.rotary_emb.inv_freq"] = \
        [ROPE_THETA ** (-2.0 * i / HEAD_DIM) for i in range(HEAD_DIM // 2)]
    st_path = os.path.join(fixdir, "tiny_llama.safetensors")
    write_safetensors(st_path, tensors)

    config = {
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "hidden_size": HIDDEN,
        "intermediate_size": INTERMEDIATE,
        "num_hidden_layers": N_LAYER,
        "num_attention_heads": N_HEAD,
        "num_key_value_heads": N_KV_HEAD,
        "vocab_size": VOCAB,
        "max_position_embeddings": MAX_POS,
        "rms_norm_eps": RMS_EPS,
        "rope_theta": ROPE_THETA,
        "tie_word_embeddings": False,
        "rope_scaling": None,
    }
    cfg_path = os.path.join(fixdir, "tiny_llama_config.json")
    with open(cfg_path, "w") as f:
        json.dump(config, f, indent=1)

    sequences = [
        [rng.randrange(VOCAB) for _ in range(MAX_POS)],
        [rng.randrange(VOCAB) for _ in range(MAX_POS)],
        list(range(MAX_POS // 2)) + [VOCAB - 1] * (MAX_POS - MAX_POS // 2),
    ]
    ref = {
        "config": config,
        "sequences": sequences,
        "logits": [llama_forward(seq, w) for seq in sequences],
    }
    json_path = os.path.join(fixdir, "tiny_llama_logits.json")
    with open(json_path, "w") as f:
        json.dump(ref, f)
    print("wrote", st_path, os.path.getsize(st_path), "bytes")
    print("wrote", cfg_path, os.path.getsize(cfg_path), "bytes")
    print("wrote", json_path, os.path.getsize(json_path), "bytes")


if __name__ == "__main__":
    main()
