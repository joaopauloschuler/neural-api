#!/usr/bin/env python3
"""Generate a tiny RANDOM HiFi-GAN ResBlock-TYPE-2 parity fixture for
tests/TestNeuralPretrained.pas (no network access: the generator is randomly
initialized from a PICO config, never downloaded).

HiFi-GAN ResBlock2 (jik876 config_v2/v3): IDENTICAL to the ResBlock1 generator
(conv_pre -> per-stage LeakyReLU -> ConvTranspose1d upsample -> Multi-Receptive-
Field = AVERAGE of num_kernels ResBlocks -> LeakyReLU -> conv_post -> tanh ->
waveform) EXCEPT the inner ResBlock is SIMPLER: per dilation tap a SINGLE
dilated Conv1d with a single LeakyReLU pre-activation and a residual add --

    for d in dilation:           # dilation list is length 2 for type 2
        x = x + conv_d(leaky_relu(x, 0.1))

NO second d=1 conv (type 1's convs2). The HF SpeechT5HifiGan module only
implements ResBlock1, so this oracle is a self-contained float64 numpy forward
of the type-2 generator -- the same approach the other pico fixtures use. The
weight key scheme matches jik876/HF: resblocks.{i}.convs.{j}.{weight,bias}.

Fixtures, KB-scale, pinned in tests/fixtures/:
  tiny_hifigan_rb2.safetensors : the random F32 generator state dict.
  tiny_hifigan_rb2_config.json : the pico config (resblock="2").
  tiny_hifigan_rb2_ref.json    : the float64 oracle mel -> waveform clips.

Coded by Claude (AI).

Usage (from the repo root):
  /home/bpsa/x/bin/python tools/make_pico_hifigan_rb2_fixture.py
Needs numpy + torch + safetensors (torch/safetensors only for the writer).
"""
import json
import os

import numpy as np
import torch
from safetensors.torch import save_file

HERE = os.path.dirname(os.path.abspath(__file__))
FIX = os.path.join(HERE, "..", "tests", "fixtures")

SEED = 4242

# ---- pico config (jik876 v2-style: 2-entry dilation lists, single conv/tap).
CFG = dict(
    model_in_dim=4,
    upsample_initial_channel=8,
    upsample_rates=[2, 2],            # total upsample x4
    upsample_kernel_sizes=[4, 4],     # (k-stride)//2 = 1 pad each
    resblock="2",
    resblock_kernel_sizes=[3, 5],     # num_kernels = 2 (MRF average of 2)
    resblock_dilation_sizes=[[1, 3], [1, 3]],
    leaky_relu_slope=0.1,
    normalize_before=False,
    sampling_rate=8000,
    model_type="hifigan",
)


def grid(a):
    """Round onto a coarse grid so f32 and f64 cannot disagree."""
    return np.round(np.asarray(a, dtype=np.float64) * 64.0) / 64.0


def rand(rng, *shape):
    return grid(rng.randn(*shape) * 0.3)


def lrelu(x, slope):
    return np.where(x >= 0, x, slope * x)


def conv1d(x, w, b, stride=1, dilation=1, pad=0):
    """x: [Cin, T] float64. w: [Cout, Cin, K]. Returns [Cout, Tout]."""
    cin, t = x.shape
    cout, _, k = w.shape
    if pad > 0:
        x = np.concatenate(
            [np.zeros((cin, pad)), x, np.zeros((cin, pad))], axis=1)
    tin = x.shape[1]
    eff = (k - 1) * dilation + 1
    tout = (tin - eff) // stride + 1
    out = np.zeros((cout, tout), dtype=np.float64)
    for o in range(cout):
        acc = np.full(tout, b[o], dtype=np.float64)
        for ci in range(cin):
            for kk in range(k):
                seg = x[ci, kk * dilation: kk * dilation + stride * tout: stride]
                acc += w[o, ci, kk] * seg[:tout]
        out[o] = acc
    return out


def conv_transpose1d(x, w, b, stride=1, pad=0):
    """x: [Cin, T]. w: [Cin, Cout, K] (torch ConvTranspose layout)."""
    cin, t = x.shape
    _, cout, k = w.shape
    tfull = (t - 1) * stride + k
    out = np.zeros((cout, tfull), dtype=np.float64)
    for ci in range(cin):
        for ti in range(t):
            base = ti * stride
            for kk in range(k):
                out[:, base + kk] += x[ci, ti] * w[ci, :, kk]
    out += b[:, None]
    if pad > 0:
        out = out[:, pad:tfull - pad]
    return out


def build_weights(rng):
    """Random weights mirroring the importer's expected key scheme."""
    cfg = CFG
    ic = cfg["upsample_initial_channel"]
    sd = {}

    # conv_pre: Conv1d(model_in_dim -> ic, k=7, pad=3)
    sd["conv_pre.weight"] = rand(rng, ic, cfg["model_in_dim"], 7)
    sd["conv_pre.bias"] = rand(rng, ic)

    num_up = len(cfg["upsample_rates"])
    num_kernels = len(cfg["resblock_kernel_sizes"])

    # upsamplers: ConvTranspose1d(C -> C/2, k, stride)
    ch = ic
    for s in range(num_up):
        k = cfg["upsample_kernel_sizes"][s]
        sd[f"upsampler.{s}.weight"] = rand(rng, ch, ch // 2, k)  # [In,Out,K]
        sd[f"upsampler.{s}.bias"] = rand(rng, ch // 2)
        ch = ch // 2

    # resblocks: FLAT list num_up*num_kernels; type-2 single conv per tap.
    ch = ic
    for s in range(num_up):
        ch = ch // 2
        for j in range(num_kernels):
            rb = s * num_kernels + j
            kk = cfg["resblock_kernel_sizes"][j]
            for d in range(len(cfg["resblock_dilation_sizes"][j])):
                sd[f"resblocks.{rb}.convs.{d}.weight"] = rand(rng, ch, ch, kk)
                sd[f"resblocks.{rb}.convs.{d}.bias"] = rand(rng, ch)

    # conv_post: Conv1d(C -> 1, k=7, pad=3)
    sd["conv_post.weight"] = rand(rng, 1, ch, 7)
    sd["conv_post.bias"] = rand(rng, 1)
    return sd


def forward_f64(sd, mel):
    """Self-contained float64 type-2 generator. mel: [frames, bands]."""
    cfg = CFG
    slope = cfg["leaky_relu_slope"]
    num_up = len(cfg["upsample_rates"])
    num_kernels = len(cfg["resblock_kernel_sizes"])

    sig = mel.T.astype(np.float64).copy()  # [bands, frames] channel-major
    sig = conv1d(sig, sd["conv_pre.weight"], sd["conv_pre.bias"], pad=3)

    for s in range(num_up):
        st = cfg["upsample_rates"][s]
        k = cfg["upsample_kernel_sizes"][s]
        pad = (k - st) // 2
        sig = lrelu(sig, slope)
        sig = conv_transpose1d(
            sig, sd[f"upsampler.{s}.weight"], sd[f"upsampler.{s}.bias"],
            stride=st, pad=pad)
        # MRF = AVERAGE of num_kernels type-2 resblocks.
        ressum = None
        for j in range(num_kernels):
            rb = s * num_kernels + j
            kk = cfg["resblock_kernel_sizes"][j]
            x = sig.copy()
            for d, dil in enumerate(cfg["resblock_dilation_sizes"][j]):
                pad_d = (kk * dil - dil) // 2
                xt = lrelu(x, slope)
                xt = conv1d(
                    xt, sd[f"resblocks.{rb}.convs.{d}.weight"],
                    sd[f"resblocks.{rb}.convs.{d}.bias"], dilation=dil, pad=pad_d)
                x = x + xt
            ressum = x if ressum is None else ressum + x
        sig = ressum / num_kernels

    sig = lrelu(sig, 0.01)  # final LeakyReLU at PyTorch DEFAULT slope.
    sig = conv1d(sig, sd["conv_post.weight"], sd["conv_post.bias"], pad=3)
    return np.tanh(sig[0])  # single channel -> waveform


def main():
    rng = np.random.RandomState(SEED)
    sd = build_weights(rng)

    clip_rng = np.random.RandomState(SEED + 1)
    n_frames = 5
    clips = []
    for _ in range(3):
        mel = grid(clip_rng.randn(n_frames, CFG["model_in_dim"]) * 0.5)
        wav = forward_f64(sd, mel)
        clips.append({"mel": mel.tolist(), "waveform": wav.tolist()})

    os.makedirs(FIX, exist_ok=True)
    tsd = {k: torch.tensor(v, dtype=torch.float32).contiguous()
           for k, v in sd.items()}
    save_file(tsd, os.path.join(FIX, "tiny_hifigan_rb2.safetensors"))

    with open(os.path.join(FIX, "tiny_hifigan_rb2_config.json"), "w") as f:
        json.dump(CFG, f, indent=1)

    with open(os.path.join(FIX, "tiny_hifigan_rb2_ref.json"), "w") as f:
        json.dump({
            "model_in_dim": CFG["model_in_dim"],
            "sampling_rate": CFG["sampling_rate"],
            "clips": clips,
        }, f)

    st = os.path.getsize(os.path.join(FIX, "tiny_hifigan_rb2.safetensors"))
    print("wrote tiny_hifigan_rb2.safetensors %d bytes" % st)
    print("frames", n_frames, "-> waveform len", len(clips[0]["waveform"]))


if __name__ == "__main__":
    main()
