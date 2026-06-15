#!/usr/bin/env python3
"""Tiny HF PEFT LoRA adapter fixture + independent reference forward (oracle).

Coded by Claude (AI).

Generates, using ONLY the Python standard library (no numpy, no torch), a
minimal parity fixture for the PEFT LoRA loader (LoadPEFTLoRAModule /
ReadPEFTAdapterConfig in neural/neuralpretrained.pas) and the MergeLoRA fold:

  tests/fixtures/tiny_lora_base.safetensors      - one frozen base nn.Linear
      weight "proj.weight" [d_out, d_in], NO bias. This is the base layer the
      adapter wraps (in Pascal: a TNNetPointwiseConvLinear(d_out)).
  tests/fixtures/tiny_lora_adapter.safetensors   - a PEFT adapter the way
      peft saves it: "base_model.model.proj.lora_A.default.weight" [r, d_in]
      and "base_model.model.proj.lora_B.default.weight" [d_out, r], NO bias.
  tests/fixtures/tiny_lora_adapter_config.json   - the matching PEFT
      adapter_config.json (lora_alpha, r, target_modules, ...).
  tests/fixtures/tiny_lora_logits.json           - reference outputs for a few
      fixed input vectors, computed straight from the math:
          y = W_base . x  +  (lora_alpha / r) * (B . (A . x))
      This forward shares NO code with the Pascal library: it is the oracle the
      loader+merge are verified against (tests/TestNeuralPretrained.pas).

Run from the repository root:  python3 tools/lora_peft_tiny_fixture.py
"""
import json
import os
import random
import struct

D_IN = 5
D_OUT = 4
RANK = 2
LORA_ALPHA = 8.0   # PEFT scale = LORA_ALPHA / RANK = 4.0
SEED = 20260613

rng = random.Random(SEED)


def randn_matrix(rows, cols, scale):
    return [[rng.gauss(0.0, scale) for _ in range(cols)] for _ in range(rows)]


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
# Independent forward (the oracle): adapted = base.x + (alpha/r) * B.(A.x).
# Every matrix is nn.Linear [out, in], y_j = sum_i x_i * w[j][i], no bias.
# ---------------------------------------------------------------------------
def linear(x, w):
    return [sum(x[i] * row[i] for i in range(len(x))) for row in w]


def adapted_forward(x, base, a, b, scale):
    base_y = linear(x, base)          # W_base . x          -> d_out
    a_y = linear(x, a)                # A . x               -> r
    b_y = linear(a_y, b)              # B . (A . x)         -> d_out
    return [base_y[j] + scale * b_y[j] for j in range(D_OUT)]


def round_f32(t):
    if isinstance(t, list):
        return [round_f32(v) for v in t]
    return struct.unpack("<f", struct.pack("<f", t))[0]


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    fixdir = os.path.join(root, "tests", "fixtures")
    os.makedirs(fixdir, exist_ok=True)

    base = round_f32(randn_matrix(D_OUT, D_IN, 0.5))   # [d_out, d_in]
    lora_a = round_f32(randn_matrix(RANK, D_IN, 0.4))  # [r, d_in]
    lora_b = round_f32(randn_matrix(D_OUT, RANK, 0.4)) # [d_out, r] (trained off 0)
    scale = LORA_ALPHA / RANK

    write_safetensors(os.path.join(fixdir, "tiny_lora_base.safetensors"),
                      {"proj.weight": base})
    # PEFT layout: base_model.model.<module>.lora_{A,B}.default.weight
    adapter = {
        "base_model.model.proj.lora_A.default.weight": lora_a,
        "base_model.model.proj.lora_B.default.weight": lora_b,
    }
    write_safetensors(os.path.join(fixdir, "tiny_lora_adapter.safetensors"),
                      adapter)

    config = {
        "peft_type": "LORA",
        "r": RANK,
        "lora_alpha": LORA_ALPHA,
        "lora_dropout": 0.0,
        "bias": "none",
        "target_modules": ["proj"],
        "task_type": "FEATURE_EXTRACTION",
    }
    with open(os.path.join(fixdir, "tiny_lora_adapter_config.json"), "w") as f:
        json.dump(config, f, indent=1)

    inputs = [
        round_f32([rng.gauss(0.0, 1.0) for _ in range(D_IN)]),
        round_f32([rng.gauss(0.0, 1.0) for _ in range(D_IN)]),
        round_f32([1.0, -0.5, 0.25, 2.0, -1.5]),
    ]
    ref = {
        "d_in": D_IN, "d_out": D_OUT, "rank": RANK,
        "lora_alpha": LORA_ALPHA, "scale": scale,
        "inputs": inputs,
        "outputs": [adapted_forward(x, base, lora_a, lora_b, scale)
                    for x in inputs],
    }
    with open(os.path.join(fixdir, "tiny_lora_logits.json"), "w") as f:
        json.dump(ref, f)

    for fn in ("tiny_lora_base.safetensors", "tiny_lora_adapter.safetensors",
               "tiny_lora_adapter_config.json", "tiny_lora_logits.json"):
        p = os.path.join(fixdir, fn)
        print("wrote", p, os.path.getsize(p), "bytes")


if __name__ == "__main__":
    main()
