#!/usr/bin/env python3
# Generates a tiny committed NF4 dequant parity fixture for the Pascal
# DequantizeNF4 unit test (tests/TestNeuralPretrained.pas).
#
# bitsandbytes is NOT importable in the project venv (CPU-only torch, the bnb
# CUDA build is absent), so this reconstructs the bitsandbytes 4-bit NormalFloat
# (NF4) dequant in pure numpy from the published, deterministic definition:
#
#   * 16-level NF4 codebook (the fixed nonlinear quantile values baked into
#     bitsandbytes functional.py create_normal_map(); the exact constants below
#     are the values bitsandbytes uses for the "nf4" quant_type).
#   * Weights are quantized in BLOCKS of `blocksize` (default 64) elements.
#     Each block stores one FP32 `absmax` scale = max(|w|) over the block.
#   * Per element: q = round-to-nearest NF4 code index of (w / absmax) using the
#     codebook as the reconstruction levels; dequant = codebook[q] * absmax.
#   * Two 4-bit indices are packed per byte, HIGH nibble first: for packed byte
#     B, element 2*i comes from (B >> 4) and element 2*i+1 from (B & 0x0F)
#     (matches bitsandbytes' kQuantizeBlockwise / kDequantizeBlockwise nibble
#     order and the HF Linear4bit storage).
#
# We emit single-quant absmax only (double-quant deferred / guarded in Pascal).
#
# Coded by Claude (AI).
import json
import numpy as np

OUT_JSON = "tests/fixtures/pico_nf4.json"

# bitsandbytes NF4 codebook (functional.py, quant_type="nf4"). 16 levels,
# index 0..15, ascending. Index 7 is exactly 0.0.
NF4_CODE = np.array([
    -1.0,
    -0.6961928009986877,
    -0.5250730514526367,
    -0.39491748809814453,
    -0.28444138169288635,
    -0.18477343022823334,
    -0.09105003625154495,
    0.0,
    0.07958029955625534,
    0.16093020141124725,
    0.24611230194568634,
    0.33791524171829224,
    0.44070982933044434,
    0.5626170039176941,
    0.7229568362236023,
    1.0,
], dtype=np.float64)


def quantize_nf4(w, blocksize):
    n = w.size
    nblocks = (n + blocksize - 1) // blocksize
    idx = np.zeros(n, dtype=np.uint8)
    absmax = np.zeros(nblocks, dtype=np.float32)
    for b in range(nblocks):
        lo = b * blocksize
        hi = min(lo + blocksize, n)
        blk = w[lo:hi].astype(np.float64)
        am = np.max(np.abs(blk)) if blk.size else 0.0
        absmax[b] = np.float32(am)
        am32 = float(absmax[b])  # quantize against the stored fp32 absmax
        if am32 == 0.0:
            idx[lo:hi] = 7  # all-zero block -> code 0.0
            continue
        normed = blk / am32
        # nearest codebook level
        d = np.abs(normed[:, None] - NF4_CODE[None, :])
        idx[lo:hi] = np.argmin(d, axis=1).astype(np.uint8)
    return idx, absmax, nblocks


def pack_high_first(idx):
    # Pad to even length, pack 2 nibbles/byte, HIGH nibble = even element.
    n = idx.size
    if n % 2 == 1:
        idx = np.concatenate([idx, np.zeros(1, dtype=np.uint8)])
    hi = idx[0::2].astype(np.uint16)
    lo = idx[1::2].astype(np.uint16)
    return ((hi << 4) | lo).astype(np.uint8)


def dequantize_nf4(packed, absmax, n, blocksize):
    # Reference dequant (what Pascal must reproduce).
    out = np.zeros(n, dtype=np.float64)
    idx = np.zeros(n + (n & 1), dtype=np.uint8)
    idx[0::2] = packed >> 4
    idx[1::2] = packed & 0x0F
    idx = idx[:n]
    for i in range(n):
        b = i // blocksize
        out[i] = NF4_CODE[idx[i]] * float(absmax[b])
    return out


def main():
    rng = np.random.default_rng(424242)
    blocksize = 64
    n = 200  # 4 blocks (64,64,64,8) -> exercises a ragged last block
    w = rng.standard_normal(n).astype(np.float32)

    idx, absmax, nblocks = quantize_nf4(w, blocksize)
    packed = pack_high_first(idx)
    deq = dequantize_nf4(packed, absmax, n, blocksize)

    fixture = {
        "blocksize": blocksize,
        "n": n,
        "nblocks": int(nblocks),
        "codebook": NF4_CODE.tolist(),
        "packed_bytes": packed.tolist(),          # uint8, len = ceil(n/2)
        "absmax": [float(x) for x in absmax],     # fp32 per block
        "dequant": deq.tolist(),                  # fp64 reference result
        "reference": "numpy-reconstructed bitsandbytes NF4 (bnb not importable)",
    }
    import os
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(fixture, f)
    print("wrote", OUT_JSON, "n=", n, "nblocks=", nblocks,
          "packed_bytes=", packed.size, "max|deq|=", float(np.max(np.abs(deq))))


if __name__ == "__main__":
    main()
