#!/usr/bin/env python3
"""Fixtures for the restricted PyTorch ``pytorch_model.bin`` reader
(neural/neuraltorchbin.pas, TNNetTorchBinReader).

Generates, into tests/fixtures/:

  tiny_torch_state.bin          torch.save'd pico state_dict: a few small
                                tensors of varied dtypes (f32/f16/bf16/i64)
                                including a 3-D one and a 0-d scalar.
  tiny_torch_state.safetensors  the SAME tensors as a safetensors twin
                                (written by hand - no safetensors package
                                needed) for the bit-for-bit parity test.
  evil_torch.bin                a torch.save zip whose data.pkl REDUCEs the
                                non-whitelisted GLOBAL `os system` - the
                                Pascal reader must raise, never execute.
  tiny_gpt2.bin                 the existing tests/fixtures/tiny_gpt2
                                .safetensors checkpoint re-saved through
                                torch.save, for GPT-2 logit parity through
                                the .bin path against tiny_gpt2_logits.json.

Run from the repo root:  python3 tools/torch_bin_fixture.py
Needs torch only (CPU build is fine).
"""
import json
import os
import struct

import torch

FIXDIR = os.path.join(os.path.dirname(__file__), "..", "tests", "fixtures")

DTYPE_ST = {
    torch.float32: ("F32", 4, "<f"),
    torch.float16: ("F16", 2, "<e"),
    torch.bfloat16: ("BF16", 2, None),
    torch.int64: ("I64", 8, "<q"),
}


def tensor_bytes(t):
    """Raw little-endian bytes of a contiguous tensor (incl. bfloat16)."""
    t = t.contiguous()
    if t.dtype == torch.bfloat16:
        return t.view(torch.uint16).numpy().tobytes()
    return t.numpy().tobytes()


def write_safetensors(path, tensors):
    """Minimal by-hand safetensors writer (avoids the safetensors dep)."""
    header = {}
    payload = b""
    for name, t in tensors.items():
        data = tensor_bytes(t)
        header[name] = {
            "dtype": DTYPE_ST[t.dtype][0],
            "shape": list(t.shape),
            "data_offsets": [len(payload), len(payload) + len(data)],
        }
        payload += data
    hjson = json.dumps(header).encode("utf-8")
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hjson)))
        f.write(hjson)
        f.write(payload)


def parse_safetensors(path):
    """Minimal by-hand safetensors reader -> {name: torch tensor}."""
    with open(path, "rb") as f:
        (hlen,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(hlen))
        data = f.read()
    out = {}
    revmap = {v[0]: k for k, v in DTYPE_ST.items()}
    for name, info in header.items():
        if name == "__metadata__":
            continue
        b, e = info["data_offsets"]
        dtype = revmap[info["dtype"]]
        if dtype == torch.bfloat16:
            t = torch.frombuffer(bytearray(data[b:e]), dtype=torch.uint16)
            t = t.view(torch.bfloat16)
        else:
            t = torch.frombuffer(bytearray(data[b:e]), dtype=dtype)
        out[name] = t.reshape(info["shape"]).clone()
    return out


class Evil:
    """Pickles to REDUCE(os.system, ('echo pwned',)) - the classic torch
    .bin code-execution payload the restricted unpickler must reject."""

    def __reduce__(self):
        return (os.system, ("echo pwned",))


def main():
    torch.manual_seed(20260612)

    # ---- varied-dtype pico state dict + safetensors twin ----
    state = {
        "w_f32": torch.randn(3, 4),
        "w_f16": torch.randn(2, 5).to(torch.float16),
        "w_bf16": torch.randn(4, 2).to(torch.bfloat16),
        "ids_i64": torch.arange(-3, 4, dtype=torch.int64),
        "cube_f32": torch.randn(2, 3, 5),
        "scalar_f32": torch.tensor(3.5),
    }
    torch.save(state, os.path.join(FIXDIR, "tiny_torch_state.bin"))
    write_safetensors(os.path.join(FIXDIR, "tiny_torch_state.safetensors"),
                      state)

    # ---- malicious pickle inside an otherwise valid torch zip ----
    torch.save({"x": Evil()}, os.path.join(FIXDIR, "evil_torch.bin"))

    # ---- tiny GPT-2 checkpoint re-saved as .bin (bit-identical f32) ----
    gpt2 = parse_safetensors(os.path.join(FIXDIR, "tiny_gpt2.safetensors"))
    torch.save(gpt2, os.path.join(FIXDIR, "tiny_gpt2.bin"))

    for name in ("tiny_torch_state.bin", "tiny_torch_state.safetensors",
                 "evil_torch.bin", "tiny_gpt2.bin"):
        p = os.path.join(FIXDIR, name)
        print(f"{name}: {os.path.getsize(p)} bytes")


if __name__ == "__main__":
    main()
