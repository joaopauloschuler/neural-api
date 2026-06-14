#!/usr/bin/env python3
"""K-quant GGUF fixtures for the Pascal GGUF reader k-quant dequant tests.

Coded by Claude (AI).

ggml's k-quants pack 256-element super-blocks: a block-level f16 scale (and
a second f16 min for Q4_K) plus 6-bit packed sub-scales/sub-mins and 4/6-bit
packed quants across 8 sub-blocks of 32. The Pascal TNNetGGUFReader mirrors
ggml's reference dequant_row_q4_K / dequant_row_q6_K bit-unpacking exactly.

This script quantizes a seeded tensor into each k-quant type with the `gguf`
Python package (the llama.cpp reference quantizer), and writes:

  tests/fixtures/tiny_kquant.gguf
      One GGUF file holding, for every implemented k-quant type T, a tensor
      named "kq.T.weight" carrying the SAME source values quantized to T,
      PLUS "ref.f32.weight" carrying the source values verbatim as F32.
      The Pascal parity test dequantizes each kq.T.weight and compares it to
      ref.f32.weight (NOT to the original floats: ggml's quantizer rounds, so
      the F32 tensor IS the source of truth) AND to the reference
      gguf.quants.dequantize output captured in the JSON below.

  tests/fixtures/tiny_kquant_ref.json
      { "block_size": 256, "shape": [rows, cols],
        "ref_f32": [...],                       # the exact source floats
        "dequant": { "Q4_K": [...], "Q6_K": [...] } }  # reference dequant
      so the Pascal test can assert against the independent Python reference
      values directly (max |diff| within the f16 block-scale tolerance).

Run from the repository root:
  /home/bpsa/x/bin/python tools/make_kquant_gguf_fixture.py
"""
import json
import os

import numpy as np

import gguf
from gguf import quants

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIX = os.path.join(ROOT, "tests", "fixtures")

F32 = gguf.GGMLQuantizationType.F32

# The k-quant types this fixture (and the Pascal reader) covers.
KQUANT_TYPES = {
    "Q4_K": gguf.GGMLQuantizationType.Q4_K,
    "Q5_K": gguf.GGMLQuantizationType.Q5_K,
    "Q6_K": gguf.GGMLQuantizationType.Q6_K,
    "Q2_K": gguf.GGMLQuantizationType.Q2_K,
}

# Source tensor: rows x cols, cols a multiple of the 256-element super-block.
# 4 rows x 256 cols => 4 super-blocks total, enough to exercise sub-block and
# scale-group geometry without a large fixture. O(1)-scale values (NOT
# std-0.02) so the quantization error and dequant values are non-vacuous.
ROWS, COLS = 4, 256
rng = np.random.RandomState(20260613)
# O(1)-scale data with per-sub-block scale variation (so the 6-bit packed
# sub-scales/sub-mins are non-trivial) but WITHOUT a global ramp that would
# make a single super-block scale span two orders of magnitude (which a
# naive quantizer fits poorly). Each row gets its own overall scale; each
# 32-wide sub-block gets its own scale+offset on top.
src = rng.randn(ROWS, COLS).astype(np.float32)
row_scale = np.array([1.0, 2.0, 0.5, 1.5], dtype=np.float32)
for r in range(ROWS):
    sub = src[r].reshape(8, 32)
    sub *= (0.5 + rng.rand(8, 1)).astype(np.float32)   # per-sub-block scale
    sub += (rng.randn(8, 1)).astype(np.float32)        # per-sub-block offset
    src[r] = (sub.reshape(COLS) * row_scale[r])


def f16(x):
    """Round a float array through f16 (the storage precision of d/d_min)."""
    return np.asarray(x, dtype=np.float32).astype(np.float16).astype(np.float32)


def pack_q4_k(src):
    """Hand-rolled Q4_K packer -> 144-byte super-blocks in the exact ggml
    byte layout (d f16, d_min f16, 12 bytes packed 6-bit sc/min, 128 bytes
    4-bit quants). Reconstruction: x = d*sc[j]*q - d_min*min[j], q in [0,15].
    Per sub-block of 32 we pick an affine (scale, min) fit
    (min = sub-block min, scale = (max-min)/15) so q rounds into [0,15]."""
    rows = src.reshape(-1, 256)
    out = []
    for blk in rows:
        sub = blk.reshape(8, 32)
        smin = sub.min(axis=1)              # per sub-block min (<= 0 usually)
        smax = sub.max(axis=1)
        sscale = (smax - smin) / 15.0
        sscale = np.where(sscale == 0, 1.0, sscale)
        # 6-bit super-scale for the scales and mins: d * sc6, d_min * min6.
        # sc6/min6 in [0,63]; pick d, d_min from the largest sub value.
        d = float(sscale.max() / 63.0)
        dmin = float((-smin).max() / 63.0)
        if d == 0:
            d = 1.0
        if dmin == 0:
            dmin = 1.0
        d = float(f16(d)); dmin = float(f16(dmin))
        sc6 = np.clip(np.round(sscale / d), 0, 63).astype(np.uint8)
        min6 = np.clip(np.round((-smin) / dmin), 0, 63).astype(np.uint8)
        # Quantize each element with the EFFECTIVE (f16-rounded) scale/min.
        eff_scale = d * sc6.astype(np.float32)
        eff_min = dmin * min6.astype(np.float32)
        q = np.zeros((8, 32), dtype=np.uint8)
        for j in range(8):
            s = eff_scale[j] if eff_scale[j] != 0 else 1.0
            q[j] = np.clip(np.round((sub[j] + eff_min[j]) / s), 0, 15).astype(np.uint8)
        # Pack the 6-bit sc/min into 12 bytes (inverse of get_scale_min).
        scales = np.zeros(12, dtype=np.uint8)
        for j in range(4):
            scales[j] = sc6[j] & 0x3F
            scales[j + 4] = min6[j] & 0x3F
            scales[j + 8] = (sc6[j + 4] & 0x0F) | ((min6[j + 4] & 0x0F) << 4)
            scales[j] |= (sc6[j + 4] & 0x30) << 2
            scales[j + 4] |= (min6[j + 4] & 0x30) << 2
        # Pack 4-bit quants: byte-chunk c holds sub-block 2c (low) + 2c+1 (high).
        qs = np.zeros(128, dtype=np.uint8)
        for c in range(4):
            lo = q[2 * c]
            hi = q[2 * c + 1]
            qs[c * 32:(c + 1) * 32] = (lo & 0x0F) | ((hi & 0x0F) << 4)
        d16 = np.array([d], dtype=np.float16).view(np.uint8)
        dmin16 = np.array([dmin], dtype=np.float16).view(np.uint8)
        out.append(np.concatenate([d16, dmin16, scales, qs]))
    return np.concatenate(out).astype(np.uint8)


def pack_q6_k(src):
    """Hand-rolled Q6_K packer -> 210-byte super-blocks (128 bytes ql 4-bit
    low, 64 bytes qh 2-bit high, 16 int8 scales, f16 d). Reconstruction:
    x = d*scales[g]*q, q = (ql | (qh<<4)) - 32 in [-32,31], 16 scale groups
    of 16 elements each."""
    rows = src.reshape(-1, 256)
    out = []
    for blk in rows:
        grp = blk.reshape(16, 16)
        # Per 16-element group: symmetric scale so q = round(x/s) in [-32,31].
        amax = np.abs(grp).max(axis=1)
        gscale = amax / 32.0
        gscale = np.where(gscale == 0, 1.0, gscale)
        d = float(gscale.max() / 127.0)
        if d == 0:
            d = 1.0
        d = float(f16(d))
        sc8 = np.clip(np.round(gscale / d), -127, 127).astype(np.int8)
        eff = d * sc8.astype(np.float32)
        q = np.zeros((16, 16), dtype=np.int32)
        for g in range(16):
            s = eff[g] if eff[g] != 0 else 1.0
            q[g] = np.clip(np.round(grp[g] / s), -32, 31)
        # u in [0,63]
        u = (q + 32).astype(np.uint8).reshape(256)
        ql4 = u & 0x0F
        qh2 = (u >> 4) & 0x03
        # ql: 256 nibbles -> 128 bytes, chunk c holds sub-block 2c(low)+2c+1(high)
        # where sub-blocks are 64 wide (ql reshaped (-1,1,64)>>[0,4]).
        ql = np.zeros(128, dtype=np.uint8)
        ql4s = ql4.reshape(4, 64)  # 4 sub-blocks of 64
        for c in range(2):
            ql[c * 64:(c + 1) * 64] = (ql4s[2 * c] & 0x0F) | ((ql4s[2 * c + 1] & 0x0F) << 4)
        # qh: 256 2-bit -> 64 bytes, chunk c (32 bytes) holds 4 sub-blocks of 32
        # via shifts [0,2,4,6] (qh reshaped (-1,1,32)>>[0,2,4,6]).
        qh = np.zeros(64, dtype=np.uint8)
        qh2s = qh2.reshape(8, 32)  # 8 sub-blocks of 32
        for c in range(2):
            qh[c * 32:(c + 1) * 32] = (
                (qh2s[4 * c] & 0x03)
                | ((qh2s[4 * c + 1] & 0x03) << 2)
                | ((qh2s[4 * c + 2] & 0x03) << 4)
                | ((qh2s[4 * c + 3] & 0x03) << 6)
            )
        d16 = np.array([d], dtype=np.float16).view(np.uint8)
        out.append(np.concatenate([ql, qh, sc8.view(np.uint8), d16]))
    return np.concatenate(out).astype(np.uint8)


def pack_q5_k(src):
    """Hand-rolled Q5_K packer -> 176-byte super-blocks in the exact ggml
    byte layout (d f16, d_min f16, 12 bytes packed 6-bit sc/min, 32 bytes qh
    5th-bit plane, 128 bytes 4-bit low quants). Like Q4_K but each quant is
    5-bit: q = ql | (qh_bit << 4) in [0,31]. Reconstruction:
    x = d*sc[j]*q - d_min*min[j], over 8 sub-blocks of 32."""
    rows = src.reshape(-1, 256)
    out = []
    for blk in rows:
        sub = blk.reshape(8, 32)
        smin = sub.min(axis=1)
        smax = sub.max(axis=1)
        sscale = (smax - smin) / 31.0           # 5-bit range
        sscale = np.where(sscale == 0, 1.0, sscale)
        d = float(sscale.max() / 63.0)
        dmin = float((-smin).max() / 63.0)
        if d == 0:
            d = 1.0
        if dmin == 0:
            dmin = 1.0
        d = float(f16(d)); dmin = float(f16(dmin))
        sc6 = np.clip(np.round(sscale / d), 0, 63).astype(np.uint8)
        min6 = np.clip(np.round((-smin) / dmin), 0, 63).astype(np.uint8)
        eff_scale = d * sc6.astype(np.float32)
        eff_min = dmin * min6.astype(np.float32)
        q = np.zeros((8, 32), dtype=np.uint8)
        for j in range(8):
            s = eff_scale[j] if eff_scale[j] != 0 else 1.0
            q[j] = np.clip(np.round((sub[j] + eff_min[j]) / s), 0, 31).astype(np.uint8)
        # Pack the 6-bit sc/min into 12 bytes (same as Q4_K).
        scales = np.zeros(12, dtype=np.uint8)
        for j in range(4):
            scales[j] = sc6[j] & 0x3F
            scales[j + 4] = min6[j] & 0x3F
            scales[j + 8] = (sc6[j + 4] & 0x0F) | ((min6[j + 4] & 0x0F) << 4)
            scales[j] |= (sc6[j + 4] & 0x30) << 2
            scales[j + 4] |= (min6[j + 4] & 0x30) << 2
        ql4 = q & 0x0F                # low 4 bits
        qh1 = (q >> 4) & 0x01         # 5th bit
        # qs: 4-bit low, byte-chunk c holds sub-block 2c(low)+2c+1(high).
        qs = np.zeros(128, dtype=np.uint8)
        for c in range(4):
            qs[c * 32:(c + 1) * 32] = (ql4[2 * c] & 0x0F) | ((ql4[2 * c + 1] & 0x0F) << 4)
        # qh: 32 bytes, sub-block j occupies bit j of each of the 32 bytes
        # (qh.reshape(-1,1,32) >> [0..7] & 1).
        qh = np.zeros(32, dtype=np.uint8)
        for j in range(8):
            qh |= (qh1[j] & 0x01) << j
        d16 = np.array([d], dtype=np.float16).view(np.uint8)
        dmin16 = np.array([dmin], dtype=np.float16).view(np.uint8)
        out.append(np.concatenate([d16, dmin16, scales, qh, qs]))
    return np.concatenate(out).astype(np.uint8)


def pack_q2_k(src):
    """Hand-rolled Q2_K packer -> 84-byte super-blocks in the exact ggml byte
    layout (16 bytes packed 4-bit scale|min, 64 bytes 2-bit quants, f16 d,
    f16 d_min). 16 sub-blocks of 16: x = d*(sc&0xF)*q - d_min*(sc>>4),
    q in [0,3]. The 4-bit scale and 4-bit min share one byte per sub-block."""
    rows = src.reshape(-1, 256)
    out = []
    for blk in rows:
        sub = blk.reshape(16, 16)           # 16 sub-blocks of 16
        smin = sub.min(axis=1)
        smax = sub.max(axis=1)
        sscale = (smax - smin) / 3.0        # 2-bit range
        sscale = np.where(sscale == 0, 1.0, sscale)
        # 4-bit super-scale for the per-sub-block scales and mins.
        d = float(sscale.max() / 15.0)
        dmin = float((-smin).max() / 15.0)
        if d == 0:
            d = 1.0
        if dmin == 0:
            dmin = 1.0
        d = float(f16(d)); dmin = float(f16(dmin))
        sc4 = np.clip(np.round(sscale / d), 0, 15).astype(np.uint8)
        min4 = np.clip(np.round((-smin) / dmin), 0, 15).astype(np.uint8)
        eff_scale = d * sc4.astype(np.float32)
        eff_min = dmin * min4.astype(np.float32)
        q = np.zeros((16, 16), dtype=np.uint8)
        for sb in range(16):
            s = eff_scale[sb] if eff_scale[sb] != 0 else 1.0
            q[sb] = np.clip(np.round((sub[sb] + eff_min[sb]) / s), 0, 3).astype(np.uint8)
        # scales: 1 byte per sub-block, (min4 << 4) | sc4.
        scales = ((min4 & 0x0F) << 4) | (sc4 & 0x0F)
        scales = scales.astype(np.uint8)
        # qs: 64 bytes. Output element e (0..255): c=e//128, s=(e%128)//32,
        # p=e%32; byte c*32+p, 2-bit field at shift 2*s. Build the inverse:
        # element index in (16 sub-blocks x 16) order is e = sb*16 + k.
        qflat = q.reshape(256)              # e -> 2-bit value
        qs = np.zeros(64, dtype=np.uint8)
        for e in range(256):
            c = e // 128
            s = (e % 128) // 32
            p = e % 32
            qs[c * 32 + p] |= (qflat[e] & 0x03) << (2 * s)
        d16 = np.array([d], dtype=np.float16).view(np.uint8)
        dmin16 = np.array([dmin], dtype=np.float16).view(np.uint8)
        out.append(np.concatenate([scales, qs, d16, dmin16]))
    return np.concatenate(out).astype(np.uint8)


def main():
    w = gguf.GGUFWriter(os.path.join(FIX, "tiny_kquant.gguf"), "kquanttest")
    w.add_uint32("kquant.block_size", 256)

    ref = {
        "block_size": 256,
        "shape": [ROWS, COLS],
        "ref_f32": src.reshape(-1).tolist(),
        "dequant": {},
    }

    w.add_tensor("ref.f32.weight", np.ascontiguousarray(src, dtype=np.float32))
    packers = {"Q4_K": pack_q4_k, "Q5_K": pack_q5_k,
               "Q6_K": pack_q6_k, "Q2_K": pack_q2_k}
    for name, qt in KQUANT_TYPES.items():
        # The gguf package only implements dequantize for k-quants, not
        # quantize. We pack valid block bytes ourselves (hand-rolled
        # round-to-nearest in the EXACT ggml byte layout), then use the
        # package's reference dequant_blocks to produce the parity values.
        q = packers[name](src)
        w.add_tensor("kq.%s.weight" % name, q, raw_dtype=qt)
        # Reference dequant via the gguf package (bit-exact to ggml-quants.c).
        deq = quants.dequantize(q, qt).astype(np.float32).reshape(src.shape)
        ref["dequant"][name] = deq.reshape(-1).tolist()
        md = float(np.abs(deq - src).max())
        print("%s: type_size ok, max|dequant-src| = %.6f" % (name, md))

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    path = os.path.join(FIX, "tiny_kquant.gguf")
    print("wrote %s (%d bytes)" % (path, os.path.getsize(path)))

    rpath = os.path.join(FIX, "tiny_kquant_ref.json")
    with open(rpath, "w") as f:
        json.dump(ref, f)
    print("wrote %s (%d bytes)" % (rpath, os.path.getsize(rpath)))


if __name__ == "__main__":
    main()
