#!/usr/bin/env python3
# Hand-written numpy float64 reference oracle for the ECAPA-TDNN speaker-
# embedding importer (BuildEcapaTdnnFromSafeTensors). speechbrain is NOT
# installed in this environment, so this script reimplements the EXACT forward
# math of the Pascal net it imports:
#
#   log-mel frames (T,1,NumMel)
#     -> conv_pre:  TDNNConv1D(NumMel->C, k=5, SAME) + bias -> ReLU
#     -> 3x SE-Res2Block (dilation 2,3,4):
#          expand = ReLU(TDNNConv1x1(C->C) blockIn)
#          Res2Net cascade over Scale channel groups (width w=C/Scale):
#            y0 = expand[:, group0]
#            y1 = ReLU(TDNNConv_dil(expand[:, group1]))
#            yi = ReLU(TDNNConv_dil(expand[:, group_i] + y_{i-1}))  (i>=2)
#          res2 = concat(y0..y_{S-1})
#          mix  = ReLU(TDNNConv1x1(C->C) res2)
#          se   = SE(mix): avg-over-time -> FCReLU(C/r) -> FCSigmoid(C)
#                          -> per-channel multiply of mix
#          out  = se + blockIn      (residual)
#     -> MFA:  concat([b1,b2,b3]) (T,1,3C) -> TDNNConv1x1(3C->MFA) + bias -> ReLU
#     -> attention head: TDNNConv1x1(MFA->Att)+bias -> tanh
#                        -> TDNNConv1x1(Att->MFA)+bias  = per-frame logits e
#     -> attentive statistics pooling: per channel softmax_t(e), then
#          mu[c]    = sum_t a[t,c]*h[t,c]
#          sigma[c] = sqrt(max(sum_t a[t,c]*h[t,c]^2 - mu[c]^2, 0) + eps)
#        output [mu | sigma]  (2*MFA,)
#     -> embedding linear: FullConnect(2*MFA -> EmbDim) + bias
#
# Writes a tiny RE-RANDOMIZED O(1)-scale safetensors fixture, a matching
# config.json and the expected embeddings (for two distinct inputs, so the test
# can also check the cosine speaker-verification score), all committed under
# tests/fixtures/. The Pascal TestEcapaParity gates max|diff| < 1e-4.
#
# Run with the shared venv: /home/bpsa/x/bin/python tools/make_pico_ecapa_fixture.py
# Coded by Claude (AI).

import json
import os
import struct
import numpy as np

np.random.seed(20260626)

HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(HERE, "..", "tests", "fixtures")
os.makedirs(OUTDIR, exist_ok=True)

# ---- tiny pico config (small for a fast, low-RAM parity test) ----------------
NUM_MEL = 6
CHANNELS = 8
KERNEL = 3            # Res2Net conv kernel (odd)
SCALE = 4            # CHANNELS % SCALE == 0
SE_REDUCTION = 4
MFA_CHANNELS = 10
ATT_CHANNELS = 5
EMB_DIM = 7
NUM_FRAMES = 12
EPS = 1e-12
DILATIONS = [2, 3, 4]

W = CHANNELS // SCALE
BOTTLENECK = max(CHANNELS // SE_REDUCTION, 1)


def relu(x):
    return np.maximum(x, 0.0)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def tdnn_conv(x, Wt, b, dilation):
    # x: (T, In). Wt: (Out, In, K) (HF Conv1d layout). Centred SAME padding:
    # tap k reads x[t + dilation*(k - K//2)]; out-of-range -> 0. -> (T, Out)
    T, In = x.shape
    Out, _, K = Wt.shape
    half = K // 2
    out = np.zeros((T, Out))
    for t in range(T):
        acc = b.copy().astype(np.float64)
        for k in range(K):
            src = t + dilation * (k - half)
            if 0 <= src < T:
                acc += Wt[:, :, k] @ x[src]
        out[t] = acc
    return out


def fc(x, Wt, b):
    # Linear over the last axis. Wt: (Out, In) row-major.
    return x @ Wt.T + b[None, ...] if x.ndim == 2 else Wt @ x + b


def se_block(mix, w_down, b_down, w_up, b_up):
    # mix: (T, C). Global average over time -> reduce(ReLU) -> expand(Sigmoid)
    # -> per-channel multiply broadcast over time.
    # NOTE: the Pascal SE block uses TNNetAvgChannel, which on an (T,1,C) volume
    # returns sum / T^2 (NOT sum / T) -- see the avgchannel-pool-scaling memo.
    # Match that exactly so the parity oracle equals the Pascal forward.
    T = mix.shape[0]
    desc = mix.sum(axis=0) / (T * T)        # (C,)
    red = relu(w_down @ desc + b_down)      # (bottleneck,)
    gate = sigmoid(w_up @ red + b_up)       # (C,)
    return mix * gate[None, :]


def se_res2_block(block_in, P, prefix, dilation):
    C = CHANNELS
    expand = relu(tdnn_conv(block_in, P[f"{prefix}.conv_expand.weight"],
                            P[f"{prefix}.conv_expand.bias"], 1))
    parts = []
    prev_y = None
    for i in range(SCALE):
        xs = expand[:, i * W:(i + 1) * W]
        if i == 0:
            y = xs
        else:
            add_in = xs if i == 1 else xs + prev_y
            y = relu(tdnn_conv(add_in, P[f"{prefix}.res2_{i}.weight"],
                               P[f"{prefix}.res2_{i}.bias"], dilation))
        parts.append(y)
        prev_y = y
    res2 = np.concatenate(parts, axis=1)    # (T, C)
    mix = relu(tdnn_conv(res2, P[f"{prefix}.conv_mix.weight"],
                         P[f"{prefix}.conv_mix.bias"], 1))
    se = se_block(mix, P[f"{prefix}.se_down.weight"], P[f"{prefix}.se_down.bias"],
                  P[f"{prefix}.se_up.weight"], P[f"{prefix}.se_up.bias"])
    return se + block_in


def attentive_stats_pool(h, e):
    # h, e: (T, C). Per channel softmax over time, weighted mean + std.
    em = e - e.max(axis=0, keepdims=True)
    a = np.exp(em)
    a = a / a.sum(axis=0, keepdims=True)    # (T, C)
    mu = (a * h).sum(axis=0)                # (C,)
    m2 = (a * h * h).sum(axis=0)
    var = np.maximum(m2 - mu * mu, 0.0)
    sigma = np.sqrt(var + EPS)
    return np.concatenate([mu, sigma])      # (2C,)


def forward(x, P):
    # x: (T, NUM_MEL)
    h = relu(tdnn_conv(x, P["conv_pre.weight"], P["conv_pre.bias"], 1))
    b1 = se_res2_block(h, P, "block1", DILATIONS[0])
    b2 = se_res2_block(b1, P, "block2", DILATIONS[1])
    b3 = se_res2_block(b2, P, "block3", DILATIONS[2])
    cat = np.concatenate([b1, b2, b3], axis=1)   # (T, 3C)
    mfa = relu(tdnn_conv(cat, P["mfa.weight"], P["mfa.bias"], 1))  # (T, MFA)
    att = np.tanh(tdnn_conv(mfa, P["att_down.weight"], P["att_down.bias"], 1))
    e = tdnn_conv(att, P["att_up.weight"], P["att_up.bias"], 1)    # (T, MFA)
    pooled = attentive_stats_pool(mfa, e)        # (2*MFA,)
    emb = P["emb.weight"] @ pooled + P["emb.bias"]
    return emb


# ---- random O(1)-scale parameters --------------------------------------------
def randn(*shape, scale=1.0):
    return (np.random.randn(*shape) * scale).astype(np.float64)


P = {}
P["conv_pre.weight"] = randn(CHANNELS, NUM_MEL, 5, scale=0.3)
P["conv_pre.bias"] = randn(CHANNELS, scale=0.1)
for bn, dil in zip(("block1", "block2", "block3"), DILATIONS):
    P[f"{bn}.conv_expand.weight"] = randn(CHANNELS, CHANNELS, 1, scale=0.3)
    P[f"{bn}.conv_expand.bias"] = randn(CHANNELS, scale=0.1)
    for i in range(1, SCALE):
        P[f"{bn}.res2_{i}.weight"] = randn(W, W, KERNEL, scale=0.3)
        P[f"{bn}.res2_{i}.bias"] = randn(W, scale=0.1)
    P[f"{bn}.conv_mix.weight"] = randn(CHANNELS, CHANNELS, 1, scale=0.3)
    P[f"{bn}.conv_mix.bias"] = randn(CHANNELS, scale=0.1)
    P[f"{bn}.se_down.weight"] = randn(BOTTLENECK, CHANNELS, scale=0.4)
    P[f"{bn}.se_down.bias"] = randn(BOTTLENECK, scale=0.1)
    P[f"{bn}.se_up.weight"] = randn(CHANNELS, BOTTLENECK, scale=0.4)
    P[f"{bn}.se_up.bias"] = randn(CHANNELS, scale=0.1)
P["mfa.weight"] = randn(MFA_CHANNELS, 3 * CHANNELS, 1, scale=0.3)
P["mfa.bias"] = randn(MFA_CHANNELS, scale=0.1)
P["att_down.weight"] = randn(ATT_CHANNELS, MFA_CHANNELS, 1, scale=0.3)
P["att_down.bias"] = randn(ATT_CHANNELS, scale=0.1)
P["att_up.weight"] = randn(MFA_CHANNELS, ATT_CHANNELS, 1, scale=0.3)
P["att_up.bias"] = randn(MFA_CHANNELS, scale=0.1)
P["emb.weight"] = randn(EMB_DIM, 2 * MFA_CHANNELS, scale=0.4)
P["emb.bias"] = randn(EMB_DIM, scale=0.1)

# Two distinct synthetic input clips so the test can also exercise the cosine
# speaker-verification score on real embeddings.
t = np.arange(NUM_FRAMES)[:, None]
mel = np.arange(NUM_MEL)[None, :]
X_A = (0.6 * np.sin(0.21 * t + 0.5 * mel) + 0.3 * np.cos(0.13 * t * (mel + 1))).astype(np.float64)
X_B = (0.5 * np.sin(0.37 * t - 0.2 * mel) + 0.4 * np.cos(0.29 * t + mel)).astype(np.float64)

emb_a = forward(X_A, P)
emb_b = forward(X_B, P)


def cosine(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


print("emb_a", emb_a.shape, "range", emb_a.min(), emb_a.max())
print("cos(A,A)=", cosine(emb_a, emb_a), "cos(A,B)=", cosine(emb_a, emb_b))


# ---- safetensors writer (float32, the Pascal reader's native dtype) ----------
def save_safetensors(path, tensors):
    header = {}
    blobs = []
    offset = 0
    for name, arr in tensors.items():
        a = np.ascontiguousarray(arr.astype(np.float32))
        b = a.tobytes()
        header[name] = {
            "dtype": "F32",
            "shape": list(a.shape),
            "data_offsets": [offset, offset + len(b)],
        }
        blobs.append(b)
        offset += len(b)
    hjson = json.dumps(header, separators=(",", ":")).encode("utf-8")
    pad = (8 - (len(hjson) % 8)) % 8
    hjson += b" " * pad
    with open(path, "wb") as fh:
        fh.write(struct.pack("<Q", len(hjson)))
        fh.write(hjson)
        for b in blobs:
            fh.write(b)


save_safetensors(os.path.join(OUTDIR, "tiny_ecapa.safetensors"), P)

config = {
    "model_type": "ecapa_tdnn",
    "num_mel": NUM_MEL,
    "channels": CHANNELS,
    "kernel": KERNEL,
    "scale": SCALE,
    "se_reduction": SE_REDUCTION,
    "mfa_channels": MFA_CHANNELS,
    "att_channels": ATT_CHANNELS,
    "emb_dim": EMB_DIM,
}
with open(os.path.join(OUTDIR, "tiny_ecapa_config.json"), "w") as fh:
    json.dump(config, fh, indent=2)

# Expected file: header (num_frames num_mel emb_dim), then for each of the two
# clips a flattened input row (T*NUM_MEL, row-major t-major) and the embedding.
exp_path = os.path.join(OUTDIR, "tiny_ecapa_expected.txt")
with open(exp_path, "w") as fh:
    fh.write(f"{NUM_FRAMES} {NUM_MEL} {EMB_DIM}\n")
    for X, emb in ((X_A, emb_a), (X_B, emb_b)):
        fh.write(" ".join(f"{v:.8f}" for v in X.reshape(-1)) + "\n")
        fh.write(" ".join(f"{v:.8f}" for v in emb) + "\n")
    fh.write(f"{cosine(emb_a, emb_a):.8f} {cosine(emb_a, emb_b):.8f}\n")

print("wrote fixtures to", os.path.abspath(OUTDIR))
