#!/usr/bin/env python3
# Hand-written numpy float64 reference oracle for the pyannote speaker-
# diarization importer (BuildPyannoteSegmentationFromSafeTensors). The pyannote
# python package is NOT installed in this environment, so this script
# reimplements the EXACT forward math of the Pascal net:
#
#   raw waveform (T,1,1)
#     -> TNNetSincConv1D  (band-pass, 2 scalars per filter, Hamming window)
#     -> abs
#     -> MaxPool1 (size=stride=Pool1)          (clean block-max; L1 % Pool1 == 0)
#     -> TokenLayerNorm  (per-frame over channels, biased var + eps)
#     -> Conv1d (valid, stride 1) + bias
#     -> ReLU
#     -> MaxPool2 (size=stride=Pool2)          (clean block-max; L2 % Pool2 == 0)
#     -> TokenLayerNorm
#     -> BiLSTM (vanilla nn.LSTM forward + reverse, concat over channels)
#     -> Linear head -> NumPowersetClasses logits per frame
#
# It then writes a tiny RE-RANDOMIZED O(1)-scale safetensors fixture, a matching
# config.json and the expected per-frame powerset logits, all committed under
# tests/fixtures/. The Pascal TestPyannoteParity gates max|diff| < 1e-4.
#
# Run with the shared venv: /home/bpsa/x/bin/python tools/make_pico_pyannote_fixture.py
# Coded by Claude (AI).

import json
import os
import struct
import numpy as np

np.random.seed(20260616)

HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(HERE, "..", "tests", "fixtures")
os.makedirs(OUTDIR, exist_ok=True)

# ---- tiny pico config (deliberately small for a fast, low-RAM parity test) ---
SAMPLE_RATE = 4000.0
SINC_FILTERS = 4
SINC_KERNEL = 11        # odd
SINC_STRIDE = 2
POOL1 = 2
CONV_CH = 5
CONV_KERNEL = 3
POOL2 = 2
# TNNetLSTMCell is a same-shape recurrence: hidden size == input depth (CONV_CH).
LSTM_HIDDEN = CONV_CH
MAX_SPEAKERS = 3
POWERSET = 7            # 1 + 3 + C(3,2)
EPS = 1e-5

half = (SINC_KERNEL - 1) // 2


def sinc(z):
    return np.where(np.abs(z) < 1e-12, 1.0, np.sin(z) / z)


def pick_num_samples():
    # Choose T so that L1 (after sinc conv) is a multiple of POOL1 and L2
    # (after conv2) is a multiple of POOL2 -> MaxPool windows are clean,
    # non-overlapping block-maxes (matching TNNetMaxPool's default-stride path
    # exactly without the partial-window edge case).
    for T in range(80, 4000):
        L1 = (T - SINC_KERNEL) // SINC_STRIDE + 1
        if L1 % POOL1 != 0:
            continue
        P1 = L1 // POOL1
        L2 = (P1 - CONV_KERNEL) // 1 + 1
        if L2 % POOL2 != 0:
            continue
        frames = L2 // POOL2
        if frames >= 4:
            return T, frames
    raise RuntimeError("no suitable T")


NUM_SAMPLES, FRAMES = pick_num_samples()


# ---- materialize the SincNet band-pass bank exactly like TNNetSincConv1D ------
def hamming(K):
    return np.array([0.54 - 0.46 * np.cos(2 * np.pi * n / (K - 1)) for n in range(K)])


def sinc_bank(low_hz, band_hz):
    win = hamming(SINC_KERNEL)
    B = np.zeros((SINC_KERNEL, SINC_FILTERS))
    for f in range(SINC_FILTERS):
        fl = abs(low_hz[f]) / SAMPLE_RATE
        fh = fl + abs(band_hz[f]) / SAMPLE_RATE
        for n in range(SINC_KERNEL):
            tap = n - half
            g = 2 * fh * sinc(2 * np.pi * fh * tap) - 2 * fl * sinc(2 * np.pi * fl * tap)
            B[n, f] = g * win[n]
    return B


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def token_layernorm(x, gamma, beta):
    # x: (T, C); normalize each frame over channels (biased var + eps).
    mu = x.mean(axis=1, keepdims=True)
    var = ((x - mu) ** 2).mean(axis=1, keepdims=True)
    xhat = (x - mu) / np.sqrt(var + EPS)
    return xhat * gamma[None, :] + beta[None, :]


def block_maxpool(x, p):
    # x: (T, C) -> (T//p, C), non-overlapping block max.
    T, C = x.shape
    To = T // p
    out = np.zeros((To, C))
    for j in range(To):
        out[j] = x[j * p:(j + 1) * p].max(axis=0)
    return out


def vanilla_lstm(x, Wih, Whh, bih, bhh):
    # Exact torch nn.LSTM single-layer forward. x: (T, In) -> (T, Hidden).
    #   weight_ih: (4H, In), weight_hh: (4H, H), bias_ih/bias_hh: (4H,)
    # torch gate order along the 4H axis is i, f, g, o:
    #   i_t = sigmoid(W_ii x + W_hi h + b_i)
    #   f_t = sigmoid(W_if x + W_hf h + b_f)
    #   g_t = tanh   (W_ig x + W_hg h + b_g)
    #   o_t = sigmoid(W_io x + W_ho h + b_o)
    #   c_t = f_t * c_{t-1} + i_t * g_t      (c_{-1}=0)
    #   h_t = o_t * tanh(c_t)                (h_{-1}=0)
    # bias_ih and bias_hh simply ADD, so fold their sum.
    T = x.shape[0]
    H = Whh.shape[1]
    b = bih + bhh                     # (4H,)
    h = np.zeros(H)
    c = np.zeros(H)
    out = np.zeros((T, H))
    for t in range(T):
        z = Wih @ x[t] + Whh @ h + b  # (4H,)
        i = sigmoid(z[0:H])
        f = sigmoid(z[H:2 * H])
        g = np.tanh(z[2 * H:3 * H])
        o = sigmoid(z[3 * H:4 * H])
        c = f * c + i * g
        h = o * np.tanh(c)
        out[t] = h
    return out


def forward(P):
    x = P["waveform"]  # (T,)
    # SincConv (valid, stride SINC_STRIDE) -> (L1, SINC_FILTERS)
    B = sinc_bank(P["low_hz"], P["band_hz"])
    L1 = (len(x) - SINC_KERNEL) // SINC_STRIDE + 1
    feat = np.zeros((L1, SINC_FILTERS))
    for ot in range(L1):
        base = ot * SINC_STRIDE
        for f in range(SINC_FILTERS):
            feat[ot, f] = np.dot(B[:, f], x[base:base + SINC_KERNEL])
    # abs -> maxpool1 -> layernorm1
    feat = np.abs(feat)
    feat = block_maxpool(feat, POOL1)
    feat = token_layernorm(feat, P["ln1_w"], P["ln1_b"])
    # conv2 (valid, stride 1) + bias -> relu -> maxpool2 -> layernorm2
    P1 = feat.shape[0]
    L2 = (P1 - CONV_KERNEL) + 1
    conv = np.zeros((L2, CONV_CH))
    W = P["conv_w"]  # (out, in, k)
    for ot in range(L2):
        for o in range(CONV_CH):
            acc = P["conv_b"][o]
            for k in range(CONV_KERNEL):
                acc += np.dot(W[o, :, k], feat[ot + k])
            conv[ot, o] = acc
    conv = np.maximum(conv, 0.0)
    conv = block_maxpool(conv, POOL2)
    conv = token_layernorm(conv, P["ln2_w"], P["ln2_b"])
    # BiLSTM: forward over conv, reverse over time-flipped conv (then flip back).
    fwd = vanilla_lstm(conv, P["fwd_weight_ih"], P["fwd_weight_hh"],
                       P["fwd_bias_ih"], P["fwd_bias_hh"])
    rev_in = conv[::-1]
    rev = vanilla_lstm(rev_in, P["rev_weight_ih"], P["rev_weight_hh"],
                       P["rev_bias_ih"], P["rev_bias_hh"])[::-1]
    h = np.concatenate([fwd, rev], axis=1)  # (frames, 2*Hidden)
    # Linear head.
    logits = h @ P["head_w"].T + P["head_b"][None, :]  # (frames, POWERSET)
    return logits


# ---- random O(1)-scale parameters --------------------------------------------
def randn(*shape, scale=1.0):
    return (np.random.randn(*shape) * scale).astype(np.float64)


P = {}
P["waveform"] = (np.sin(np.arange(NUM_SAMPLES) * 0.11) * 0.6
                 + np.sin(np.arange(NUM_SAMPLES) * 0.37) * 0.4).astype(np.float64)
# SincNet band edges: positive low/band in Hz, spread under Nyquist.
P["low_hz"] = np.array([100.0 + 300.0 * f for f in range(SINC_FILTERS)])
P["band_hz"] = np.array([150.0 + 80.0 * f for f in range(SINC_FILTERS)])
P["ln1_w"] = randn(SINC_FILTERS, scale=0.3) + 1.0
P["ln1_b"] = randn(SINC_FILTERS, scale=0.2)
P["conv_w"] = randn(CONV_CH, SINC_FILTERS, CONV_KERNEL, scale=0.4)
P["conv_b"] = randn(CONV_CH, scale=0.2)
P["ln2_w"] = randn(CONV_CH, scale=0.3) + 1.0
P["ln2_b"] = randn(CONV_CH, scale=0.2)
for d in ("fwd", "rev"):
    # nn.LSTM tensors: weight_ih (4H, In), weight_hh (4H, H), bias_ih/hh (4H,).
    # Gate rows are ordered i, f, g, o along the 4H axis.
    P[f"{d}_weight_ih"] = randn(4 * LSTM_HIDDEN, CONV_CH, scale=0.3)
    P[f"{d}_weight_hh"] = randn(4 * LSTM_HIDDEN, LSTM_HIDDEN, scale=0.3)
    P[f"{d}_bias_ih"] = randn(4 * LSTM_HIDDEN, scale=0.2)
    P[f"{d}_bias_hh"] = randn(4 * LSTM_HIDDEN, scale=0.2)
    # A common default: positive forget-gate bias (rows H..2H) eases pass-through.
    P[f"{d}_bias_ih"][LSTM_HIDDEN:2 * LSTM_HIDDEN] += 0.5
P["head_w"] = randn(POWERSET, 2 * LSTM_HIDDEN, scale=0.4)
P["head_b"] = randn(POWERSET, scale=0.2)

logits = forward(P)
print("NUM_SAMPLES", NUM_SAMPLES, "FRAMES", FRAMES, "logits", logits.shape)
assert logits.shape[0] == FRAMES, (logits.shape, FRAMES)


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


tensors = {
    "sincnet.low_hz": P["low_hz"],
    "sincnet.band_hz": P["band_hz"],
    "ln1.weight": P["ln1_w"],
    "ln1.bias": P["ln1_b"],
    "conv.weight": P["conv_w"],
    "conv.bias": P["conv_b"],
    "ln2.weight": P["ln2_w"],
    "ln2.bias": P["ln2_b"],
    "head.weight": P["head_w"],
    "head.bias": P["head_b"],
}
for d in ("fwd", "rev"):
    for g in ("weight_ih", "weight_hh", "bias_ih", "bias_hh"):
        tensors[f"lstm.{d}.{g}"] = P[f"{d}_{g}"]

save_safetensors(os.path.join(OUTDIR, "tiny_pyannote.safetensors"), tensors)

config = {
    "model_type": "pyannote",
    "sample_rate": SAMPLE_RATE,
    "sinc_filters": SINC_FILTERS,
    "sinc_kernel": SINC_KERNEL,
    "sinc_stride": SINC_STRIDE,
    "pool1": POOL1,
    "conv_channels": CONV_CH,
    "conv_kernel": CONV_KERNEL,
    "pool2": POOL2,
    "lstm_hidden": LSTM_HIDDEN,
    "max_speakers": MAX_SPEAKERS,
    "layer_norm_eps": EPS,
}
with open(os.path.join(OUTDIR, "tiny_pyannote_config.json"), "w") as fh:
    json.dump(config, fh, indent=2)

# Expected outputs: num_samples, frames, powerset, the input waveform, and the
# per-frame logits (row-major frames x powerset). Plain text so the Pascal test
# reads it without a JSON parser.
exp_path = os.path.join(OUTDIR, "tiny_pyannote_expected.txt")
with open(exp_path, "w") as fh:
    fh.write(f"{NUM_SAMPLES} {FRAMES} {POWERSET}\n")
    fh.write(" ".join(f"{v:.8f}" for v in P["waveform"]) + "\n")
    for fr in range(FRAMES):
        fh.write(" ".join(f"{logits[fr, c]:.8f}" for c in range(POWERSET)) + "\n")

print("wrote fixtures to", os.path.abspath(OUTDIR))
print("logit range", logits.min(), logits.max())
