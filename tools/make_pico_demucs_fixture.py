#!/usr/bin/env python3
"""Generate a tiny RANDOM time-domain Demucs (v2) source-separation parity
fixture for tests/TestNeuralPretrained.pas.

transformers has no Demucs, so this is a SELF-CONTAINED numpy float64 oracle
built directly from the published Demucs-v2 architecture (Defossez et al.,
"Music Source Separation in the Waveform Domain", arXiv:1911.13254). It pins
the WHOLE mixed-waveform -> 4-stem pass of a pico instance.

Time-domain Demucs is a symmetric 1-D conv U-Net:

  encoder block i (depth blocks):
    Conv1d(in -> out,  kernel=8, stride=4)  -> ReLU
    Conv1d(out -> 2*out, kernel=1)          -> GLU(dim=channel)   # halves back
  bottleneck:
    BiLSTM(channels, channels, num_layers=2, bidirectional)
      -> Linear(2*channels -> channels)
  decoder block i (reverse order, depth blocks):
    x = x + skip[i]                                   # U-Net skip add
    Conv1d(in -> 2*in, kernel=3, stride=1, pad=1) -> GLU(dim=channel)
    ConvTranspose1d(in -> out, kernel=8, stride=4)
    ReLU   (every block EXCEPT the last, which emits sources*audio_channels)

Channel schedule: encoder channels[i] = channels * 2**i (channels at block 0
takes audio_channels in); the final decoder block emits
sources*audio_channels. Input length must be a multiple of stride**depth so the
strided encoder / transposed decoder line up to the same length (valid_length
padding in the real model); the fixture pins such a length directly.

This v1 oracle omits the input/output normalization (centering + std rescale,
`normalize=False`) and weight rescaling (`rescale=1`) so the math is the bare
conv/GLU/LSTM/transpose-conv stack the Pascal holder reproduces.

Fixtures, KB-scale, pinned in tests/fixtures/:
  tiny_demucs.safetensors  : raw Demucs-style state dict (F32).
  tiny_demucs_config.json  : the pico TDemucsConfig fields.
  tiny_demucs_ref.json     : float64 oracle (input mix + 4 separated stems).

Coded by Claude (AI).

Usage (from the repo root):
  /home/bpsa/x/bin/python tools/make_pico_demucs_fixture.py
Needs numpy + safetensors (NO torch / transformers / network).
"""
import json
import os

import numpy as np
from safetensors.numpy import save_file

HERE = os.path.dirname(os.path.abspath(__file__))
FIX = os.path.join(HERE, "..", "tests", "fixtures")

SEED = 4242

# ---- pico architecture ----------------------------------------------------
AUDIO_CHANNELS = 2     # stereo in / stereo per stem out
SOURCES = 4            # drums / bass / other / vocals
CHANNELS = 4           # base hidden width
DEPTH = 2              # encoder/decoder blocks
KERNEL = 8
STRIDE = 4
CONTEXT = 3            # decoder pre-conv kernel
LSTM_LAYERS = 2
SAMPLING_RATE = 8000


def q(a):
    """Round onto a coarse grid so f32 and f64 cannot disagree."""
    return np.round(np.asarray(a, dtype=np.float64) * 64.0) / 64.0


def glu(x):
    # x: (channels, time); split channels in half, a * sigmoid(b).
    c = x.shape[0] // 2
    a = x[:c]
    b = x[c:]
    return a * (1.0 / (1.0 + np.exp(-b)))


def conv1d(x, w, b, stride=1, pad=0):
    # x: (in, T); w: (out, in, K); b: (out,). Returns (out, Tout).
    cin, T = x.shape
    out, _, K = w.shape
    if pad:
        x = np.concatenate(
            [np.zeros((cin, pad)), x, np.zeros((cin, pad))], axis=1)
        T = x.shape[1]
    Tout = (T - K) // stride + 1
    y = np.empty((out, Tout), dtype=np.float64)
    for o in range(out):
        acc = np.full(Tout, b[o], dtype=np.float64)
        for k in range(K):
            for i in range(cin):
                xs = x[i, k:k + stride * Tout:stride]
                acc += w[o, i, k] * xs[:Tout]
        y[o] = acc
    return y


def conv_transpose1d(x, w, b, stride):
    # x: (in, T); w: (in, out, K); b: (out,). Output len (T-1)*stride + K.
    cin, T = x.shape
    _, out, K = w.shape
    Tout = (T - 1) * stride + K
    y = np.zeros((out, Tout), dtype=np.float64)
    for o in range(out):
        y[o] += b[o]
    for i in range(cin):
        for t in range(T):
            base = t * stride
            for k in range(K):
                y[:, base + k] += w[i, :, k] * x[i, t]
    return y


def lstm_cell(x, params, reverse=False):
    # x: (T, in). params: weight_ih (4H,in), weight_hh (4H,H), bias_ih, bias_hh.
    Wih, Whh, bih, bhh = params
    H = Whh.shape[1]
    T = x.shape[0]
    h = np.zeros(H, dtype=np.float64)
    c = np.zeros(H, dtype=np.float64)
    order = range(T - 1, -1, -1) if reverse else range(T)
    outs = [None] * T
    for t in order:
        g = Wih @ x[t] + bih + Whh @ h + bhh
        i = 1.0 / (1.0 + np.exp(-g[0:H]))
        f = 1.0 / (1.0 + np.exp(-g[H:2 * H]))
        gg = np.tanh(g[2 * H:3 * H])
        o = 1.0 / (1.0 + np.exp(-g[3 * H:4 * H]))
        c = f * c + i * gg
        h = o * np.tanh(c)
        outs[t] = h.copy()
    return np.stack(outs, axis=0)  # (T, H)


def center_trim(x, ref_len):
    # Trim x (C, T) symmetrically to ref_len (Demucs center_trim).
    extra = x.shape[1] - ref_len
    if extra <= 0:
        return x
    left = extra // 2
    return x[:, left:left + ref_len]


def bilstm(x, layers):
    # x: (T, in). layers: list of {fwd:..., bwd:...} param dicts. Returns (T, 2H).
    cur = x
    for lp in layers:
        f = lstm_cell(cur, lp["fwd"], reverse=False)
        b = lstm_cell(cur, lp["bwd"], reverse=True)
        cur = np.concatenate([f, b], axis=1)  # (T, 2H)
    return cur


def main():
    rng = np.random.RandomState(SEED)

    def randw(*shape):
        return q(rng.randn(*shape) * 0.3)

    # channel schedule
    enc_in = [AUDIO_CHANNELS] + [CHANNELS * (2 ** i) for i in range(DEPTH)]
    enc_out = [CHANNELS * (2 ** i) for i in range(DEPTH)]
    enc_in = enc_in[:DEPTH]  # in channels per encoder block
    # block i: in = enc_in[i], out = enc_out[i]
    H = enc_out[-1]  # bottleneck width = top channels

    sd = {}
    enc = []
    for i in range(DEPTH):
        cin = enc_in[i]
        cout = enc_out[i]
        w1 = randw(cout, cin, KERNEL)
        b1 = randw(cout)
        w2 = randw(2 * cout, cout, 1)
        b2 = randw(2 * cout)
        enc.append((w1, b1, w2, b2))
        sd["encoder.%d.0.weight" % i] = w1.astype(np.float32)
        sd["encoder.%d.0.bias" % i] = b1.astype(np.float32)
        sd["encoder.%d.2.weight" % i] = w2.astype(np.float32)
        sd["encoder.%d.2.bias" % i] = b2.astype(np.float32)

    # BiLSTM
    lstm_layers = []
    for L in range(LSTM_LAYERS):
        in_sz = H if L == 0 else 2 * H
        fwd = (randw(4 * H, in_sz), randw(4 * H, H),
               randw(4 * H), randw(4 * H))
        bwd = (randw(4 * H, in_sz), randw(4 * H, H),
               randw(4 * H), randw(4 * H))
        lstm_layers.append({"fwd": fwd, "bwd": bwd})
        sd["lstm.weight_ih_l%d" % L] = fwd[0].astype(np.float32)
        sd["lstm.weight_hh_l%d" % L] = fwd[1].astype(np.float32)
        sd["lstm.bias_ih_l%d" % L] = fwd[2].astype(np.float32)
        sd["lstm.bias_hh_l%d" % L] = fwd[3].astype(np.float32)
        sd["lstm.weight_ih_l%d_reverse" % L] = bwd[0].astype(np.float32)
        sd["lstm.weight_hh_l%d_reverse" % L] = bwd[1].astype(np.float32)
        sd["lstm.bias_ih_l%d_reverse" % L] = bwd[2].astype(np.float32)
        sd["lstm.bias_hh_l%d_reverse" % L] = bwd[3].astype(np.float32)
    lin_w = randw(H, 2 * H)
    lin_b = randw(H)
    sd["lstm_linear.weight"] = lin_w.astype(np.float32)
    sd["lstm_linear.bias"] = lin_b.astype(np.float32)

    # Decoder blocks (reverse order). block j (j=0 is the TOP, run first).
    dec = []
    for j in range(DEPTH):
        # block index counting from top: cin = enc_out[DEPTH-1-j]
        cin = enc_out[DEPTH - 1 - j]
        if j == DEPTH - 1:
            cout = SOURCES * AUDIO_CHANNELS
        else:
            cout = enc_out[DEPTH - 2 - j]
        w1 = randw(2 * cin, cin, CONTEXT)
        b1 = randw(2 * cin)
        w2 = randw(cin, cout, KERNEL)  # ConvTranspose1d weight (in,out,K)
        b2 = randw(cout)
        dec.append((w1, b1, w2, b2, cin, cout))
        sd["decoder.%d.0.weight" % j] = w1.astype(np.float32)
        sd["decoder.%d.0.bias" % j] = b1.astype(np.float32)
        sd["decoder.%d.2.weight" % j] = w2.astype(np.float32)
        sd["decoder.%d.2.bias" % j] = b2.astype(np.float32)

    # ---- forward (float64 oracle) -----------------------------------------
    # Input length: multiple of STRIDE**DEPTH so lengths line up.
    base_T = STRIDE ** DEPTH  # 16
    n_steps = 3
    T_in = base_T * n_steps + KERNEL  # give room: encoder valid-conv shrinks

    clips = []
    for _ in range(2):
        mix = q(rng.randn(AUDIO_CHANNELS, T_in) * 0.5)

        # encoder
        x = mix
        skips = []
        for i in range(DEPTH):
            w1, b1, w2, b2 = enc[i]
            x = conv1d(x, w1, b1, stride=STRIDE, pad=0)
            x = np.maximum(x, 0.0)               # ReLU
            x = conv1d(x, w2, b2, stride=1, pad=0)
            x = glu(x)
            skips.append(x.copy())

        # bottleneck BiLSTM over time: x is (C, T) -> (T, C)
        seq = x.T.copy()
        seq = bilstm(seq, lstm_layers)           # (T, 2H)
        seq = seq @ lin_w.T + lin_b              # (T, H)
        x = seq.T.copy()                         # (C, T)

        # decoder
        for j in range(DEPTH):
            skip = center_trim(skips[DEPTH - 1 - j], x.shape[1])  # Demucs trim
            x = x + skip                         # U-Net skip add
            w1, b1, w2, b2, cin, cout = dec[j]
            x = conv1d(x, w1, b1, stride=1, pad=CONTEXT // 2)
            x = glu(x)
            x = conv_transpose1d(x, w2, b2, stride=STRIDE)
            if j != DEPTH - 1:
                x = np.maximum(x, 0.0)           # ReLU except last

        # trim final output to the input length (Demucs center_trim to L).
        x = center_trim(x, T_in)
        # x: (sources*audio_channels, Tout) -> (sources, audio_channels, Tout)
        Tout = x.shape[1]
        stems = x.reshape(SOURCES, AUDIO_CHANNELS, Tout)

        clips.append({
            "mix": mix.tolist(),                 # [audio_channels][T_in]
            "stems": stems.tolist(),             # [sources][audio_channels][Tout]
        })

    # ---- save -------------------------------------------------------------
    os.makedirs(FIX, exist_ok=True)
    save_file({k: np.ascontiguousarray(v) for k, v in sd.items()},
              os.path.join(FIX, "tiny_demucs.safetensors"))

    cfg = {
        "model_type": "demucs",
        "sources": SOURCES,
        "audio_channels": AUDIO_CHANNELS,
        "channels": CHANNELS,
        "depth": DEPTH,
        "kernel_size": KERNEL,
        "stride": STRIDE,
        "context": CONTEXT,
        "lstm_layers": LSTM_LAYERS,
        "rescale": 1.0,
        "normalize": False,
        "sampling_rate": SAMPLING_RATE,
    }
    with open(os.path.join(FIX, "tiny_demucs_config.json"), "w") as f:
        json.dump(cfg, f, indent=1)

    with open(os.path.join(FIX, "tiny_demucs_ref.json"), "w") as f:
        json.dump({
            "sources": SOURCES,
            "audio_channels": AUDIO_CHANNELS,
            "sampling_rate": SAMPLING_RATE,
            "clips": clips,
        }, f)

    st = os.path.getsize(os.path.join(FIX, "tiny_demucs.safetensors"))
    print("wrote tiny_demucs.safetensors %d bytes" % st)
    print("mix len", T_in, "-> stem len", len(clips[0]["stems"][0][0]))


if __name__ == "__main__":
    main()
