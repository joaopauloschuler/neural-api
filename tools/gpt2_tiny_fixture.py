#!/usr/bin/env python3
"""Tiny GPT-2 fixture generator + independent reference forward (the oracle).

Coded by Claude (AI).

Generates, using ONLY the Python standard library (no numpy, no torch):
  tests/fixtures/tiny_gpt2.safetensors  - a random GPT-2-shaped checkpoint
      (n_layer=2, n_head=2, n_embd=8, n_ctx=16, vocab=11) in the exact HF
      safetensors layout, including HF Conv1D's TRANSPOSED [in, out] weight
      storage and a dummy ignorable "h.N.attn.bias" causal-mask buffer.
  tests/fixtures/tiny_gpt2_logits.json  - reference logits for a few fixed
      token sequences, computed by the ~150-line GPT-2 forward implemented
      below straight from the math (embeddings + per-token LayerNorm +
      causal softmax attention + gelu_new MLP + tied wte^T head). This
      forward shares NO code with the Pascal library: it is the oracle the
      Pascal importer is verified against (tests/TestNeuralPretrained.pas).

Run from the repository root:  python3 tools/gpt2_tiny_fixture.py
"""
import json
import math
import os
import random
import struct

N_LAYER = 2
N_HEAD = 2
N_EMBD = 8
N_CTX = 16
VOCAB = 11
SEED = 424242

rng = random.Random(SEED)


def randn_matrix(rows, cols, scale):
    return [[rng.gauss(0.0, scale) for _ in range(cols)] for _ in range(rows)]


def randn_vector(n, scale):
    return [rng.gauss(0.0, scale) for _ in range(n)]


# ---------------------------------------------------------------------------
# Random GPT-2-shaped weights. Matrices ending in ".weight" of Conv1D modules
# (c_attn, c_proj, c_fc) use the HF Conv1D convention: shape [in, out] and
# y = x @ W + b. LayerNorm gains start near 1 so activations stay O(1).
# ---------------------------------------------------------------------------
def make_weights():
    w = {}
    w["wte.weight"] = randn_matrix(VOCAB, N_EMBD, 0.4)          # [vocab, d]
    w["wpe.weight"] = randn_matrix(N_CTX, N_EMBD, 0.2)          # [n_ctx, d]
    for b in range(N_LAYER):
        p = "h.%d." % b
        w[p + "ln_1.weight"] = [1.0 + rng.gauss(0.0, 0.1) for _ in range(N_EMBD)]
        w[p + "ln_1.bias"] = randn_vector(N_EMBD, 0.1)
        w[p + "attn.c_attn.weight"] = randn_matrix(N_EMBD, 3 * N_EMBD, 0.3)
        w[p + "attn.c_attn.bias"] = randn_vector(3 * N_EMBD, 0.1)
        w[p + "attn.c_proj.weight"] = randn_matrix(N_EMBD, N_EMBD, 0.3)
        w[p + "attn.c_proj.bias"] = randn_vector(N_EMBD, 0.1)
        w[p + "ln_2.weight"] = [1.0 + rng.gauss(0.0, 0.1) for _ in range(N_EMBD)]
        w[p + "ln_2.bias"] = randn_vector(N_EMBD, 0.1)
        w[p + "mlp.c_fc.weight"] = randn_matrix(N_EMBD, 4 * N_EMBD, 0.3)
        w[p + "mlp.c_fc.bias"] = randn_vector(4 * N_EMBD, 0.1)
        w[p + "mlp.c_proj.weight"] = randn_matrix(4 * N_EMBD, N_EMBD, 0.3)
        w[p + "mlp.c_proj.bias"] = randn_vector(N_EMBD, 0.1)
    w["ln_f.weight"] = [1.0 + rng.gauss(0.0, 0.1) for _ in range(N_EMBD)]
    w["ln_f.bias"] = randn_vector(N_EMBD, 0.1)
    return w


# ---------------------------------------------------------------------------
# safetensors writer (format: 8-byte LE uint64 header length, JSON header,
# raw little-endian tensor data).
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
    for name in tensors:  # insertion order is stable in py3.7+
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
# Independent GPT-2 forward (the oracle). Written from the math.
# ---------------------------------------------------------------------------
def layer_norm(x, gain, bias, eps=1e-5):
    n = len(x)
    mean = sum(x) / n
    var = sum((v - mean) ** 2 for v in x) / n  # population variance
    inv = 1.0 / math.sqrt(var + eps)
    return [(x[i] - mean) * inv * gain[i] + bias[i] for i in range(n)]


def linear_conv1d(x, w, b):
    """HF Conv1D: y_j = sum_i x_i * w[i][j] + b_j (w stored [in, out])."""
    nin = len(w)
    nout = len(b)
    return [sum(x[i] * w[i][j] for i in range(nin)) + b[j] for j in range(nout)]


def gelu_new(v):
    return 0.5 * v * (1.0 + math.tanh(math.sqrt(2.0 / math.pi) *
                                      (v + 0.044715 * v ** 3)))


def softmax(scores):
    m = max(scores)
    exps = [math.exp(s - m) for s in scores]
    z = sum(exps)
    return [e / z for e in exps]


def attention(xs, w, prefix):
    """Causal multi-head self-attention over the whole sequence xs."""
    seq = len(xs)
    dh = N_EMBD // N_HEAD
    qkv = [linear_conv1d(x, w[prefix + "attn.c_attn.weight"],
                         w[prefix + "attn.c_attn.bias"]) for x in xs]
    out = []
    for i in range(seq):
        merged = [0.0] * N_EMBD
        for h in range(N_HEAD):
            q = qkv[i][h * dh:(h + 1) * dh]
            scores = []
            for j in range(i + 1):  # causal: keys j <= i
                k = qkv[j][N_EMBD + h * dh:N_EMBD + (h + 1) * dh]
                scores.append(sum(q[d] * k[d] for d in range(dh)) /
                              math.sqrt(dh))
            probs = softmax(scores)
            for j in range(i + 1):
                v = qkv[j][2 * N_EMBD + h * dh:2 * N_EMBD + (h + 1) * dh]
                for d in range(dh):
                    merged[h * dh + d] += probs[j] * v[d]
        out.append(linear_conv1d(merged, w[prefix + "attn.c_proj.weight"],
                                 w[prefix + "attn.c_proj.bias"]))
    return out


def gpt2_forward(tokens, w):
    """Returns logits[position][vocab] for every position."""
    xs = []
    for pos, tok in enumerate(tokens):
        xs.append([w["wte.weight"][tok][c] + w["wpe.weight"][pos][c]
                   for c in range(N_EMBD)])
    for b in range(N_LAYER):
        p = "h.%d." % b
        normed = [layer_norm(x, w[p + "ln_1.weight"], w[p + "ln_1.bias"])
                  for x in xs]
        att = attention(normed, w, p)
        xs = [[xs[i][c] + att[i][c] for c in range(N_EMBD)]
              for i in range(len(xs))]
        mlp = []
        for x in xs:
            hcur = layer_norm(x, w[p + "ln_2.weight"], w[p + "ln_2.bias"])
            hcur = [gelu_new(v) for v in
                    linear_conv1d(hcur, w[p + "mlp.c_fc.weight"],
                                  w[p + "mlp.c_fc.bias"])]
            mlp.append(linear_conv1d(hcur, w[p + "mlp.c_proj.weight"],
                                     w[p + "mlp.c_proj.bias"]))
        xs = [[xs[i][c] + mlp[i][c] for c in range(N_EMBD)]
              for i in range(len(xs))]
    # final LayerNorm + tied LM head: logits = h . wte^T
    logits = []
    for x in xs:
        hcur = layer_norm(x, w["ln_f.weight"], w["ln_f.bias"])
        logits.append([sum(hcur[c] * w["wte.weight"][t][c]
                           for c in range(N_EMBD)) for t in range(VOCAB)])
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

    # Serialize, adding the ignorable causal-mask buffers HF checkpoints
    # carry (the importer must skip them without complaint).
    tensors = dict(w)
    for b in range(N_LAYER):
        mask = [[[[1.0 if kj <= qi else 0.0 for kj in range(N_CTX)]
                  for qi in range(N_CTX)]]]
        tensors["h.%d.attn.bias" % b] = mask  # shape [1, 1, n_ctx, n_ctx]
    st_path = os.path.join(fixdir, "tiny_gpt2.safetensors")
    write_safetensors(st_path, tensors)

    sequences = [
        [rng.randrange(VOCAB) for _ in range(N_CTX)],
        [rng.randrange(VOCAB) for _ in range(N_CTX)],
        list(range(N_CTX // 2)) + [VOCAB - 1] * (N_CTX - N_CTX // 2),
    ]
    ref = {
        "config": {"n_layer": N_LAYER, "n_head": N_HEAD, "n_embd": N_EMBD,
                   "n_ctx": N_CTX, "vocab": VOCAB},
        "sequences": sequences,
        "logits": [gpt2_forward(seq, w) for seq in sequences],
    }
    json_path = os.path.join(fixdir, "tiny_gpt2_logits.json")
    with open(json_path, "w") as f:
        json.dump(ref, f)
    print("wrote", st_path, os.path.getsize(st_path), "bytes")
    print("wrote", json_path, os.path.getsize(json_path), "bytes")


if __name__ == "__main__":
    main()
