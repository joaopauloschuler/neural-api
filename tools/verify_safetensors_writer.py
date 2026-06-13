#!/usr/bin/env python3
"""Cross-framework check for the Pascal safetensors WRITER
(TNNetSafeTensorsWriter in neural/neuralsafetensors.pas).

The Pascal test TTestNeuralPretrained.TestSafeTensorsWriterRoundTrip writes
a demo file plus a JSON sidecar describing the expected tensors:

    $TMPDIR/cai_safetensors_writer_demo.safetensors
    $TMPDIR/cai_safetensors_writer_demo.json   (names/shapes/values/metadata)

This script loads the .safetensors file with the Python "safetensors"
library (i.e. the reference implementation, NOT the Pascal reader) and
verifies names, shapes, dtypes, exact float32 values and the __metadata__
block against the sidecar.

It also runs a second pass over the half-precision file written by
TTestNeuralPretrained.TestSafeTensorsWriterF16BF16RoundTrip:

    $TMPDIR/cai_st_writer_half.safetensors
    $TMPDIR/cai_st_writer_half.json

asserting the on-disk dtype is exactly F16/BF16 and that the EncodeF16/
EncodeBF16 encode-on-write values match the F32 source within the rounding
tolerance of each format (F16 ~1e-3, BF16 ~1e-2). BF16 is decoded by a manual
uint16->float32 bit expansion (no ml_dtypes dependency).

How to run (not wired into tests/RunAll.sh on purpose):
    bash tests/RunAll.sh                       # writes both file pairs
    /home/bpsa/x/bin/python tools/verify_safetensors_writer.py
Optionally pass the F32 sidecar path as argv[1] and the half sidecar as
argv[2] if TMPDIR differs.
"""
import json
import os
import sys
import tempfile

import numpy as np
from safetensors import safe_open


def _bf16_to_f32(raw_u16):
    """Decode BF16 (numpy uint16) to float32 by bit-expansion.

    BF16 is simply the high 16 bits of an IEEE-754 float32, so left-shifting
    each value by 16 into a uint32 and reinterpreting as float32 is exact (this
    avoids a hard dependency on ml_dtypes, which is absent from this venv)."""
    u32 = raw_u16.astype(np.uint32) << 16
    return u32.view(np.float32)


def verify_half(sidecar_path):
    """Cross-check the EncodeF16/EncodeBF16 encode-on-write output against the
    F32 source values in the sidecar. Returns (failures, n_tensors)."""
    with open(sidecar_path) as f:
        sidecar = json.load(f)
    st_path = sidecar["file"]
    if not os.path.exists(st_path):
        sys.exit(f"safetensors file not found: {st_path}")

    # F16 keeps ~10 mantissa bits (~3-4 sig digits); BF16 only 8.
    tol = {"F32": (0.0, 0.0), "F16": (1e-3, 1e-4), "BF16": (1e-2, 1e-3)}
    header = _read_header(st_path)
    failures = 0
    n = 0
    with safe_open(st_path, framework="numpy") as f:
        names = set(f.keys())
        for name, spec in sidecar["tensors"].items():
            n += 1
            want_dtype = spec["dtype"]
            if name not in names:
                print(f"FAIL {name}: missing from file")
                failures += 1
                continue
            # Read the on-disk dtype straight from the header JSON (safe_open
            # decodes F16, but cannot materialize BF16 without ml_dtypes).
            hdr_dtype = header[name]["dtype"]
            if hdr_dtype != want_dtype:
                print(f"FAIL {name}: header dtype {hdr_dtype!r}, "
                      f"expected {want_dtype!r}")
                failures += 1
            expected = np.array(spec["values"], dtype=np.float32)
            if want_dtype == "BF16":
                # numpy/safe_open cannot materialize bfloat16 without ml_dtypes;
                # read the raw uint16 payload and bit-expand it ourselves.
                got = _bf16_to_f32(_read_raw_u16(st_path, name))
            else:
                t = f.get_tensor(name)
                got = t.astype(np.float32).reshape(-1)
                exp_np = {"F32": np.float32, "F16": np.float16}[want_dtype]
                if t.dtype != exp_np:
                    print(f"FAIL {name}: numpy dtype {t.dtype}, "
                          f"expected {exp_np}")
                    failures += 1
            rtol, atol = tol[want_dtype]
            tols = np.abs(expected) * rtol + atol
            diff = np.abs(got - expected)
            bad = np.flatnonzero(diff > tols)
            if len(bad):
                j = bad[0]
                print(f"FAIL {name} ({want_dtype}): {len(bad)} value(s) "
                      f"out of tol, first at {j}: got {got[j]!r} vs "
                      f"{expected[j]!r} (diff {diff[j]:.3g} > {tols[j]:.3g})")
                failures += 1
    return failures, n


import struct


def _read_header(st_path):
    """Parse the safetensors header JSON (dtype/shape/data_offsets per name)."""
    with open(st_path, "rb") as fh:
        (hlen,) = struct.unpack("<Q", fh.read(8))
        header = json.loads(fh.read(hlen).decode("utf-8"))
    header.pop("__metadata__", None)
    return header


def _read_raw_u16(st_path, name):
    """Read a tensor's raw little-endian uint16 payload by parsing the header
    offsets directly (used for BF16, which the safetensors python loader cannot
    materialize without ml_dtypes)."""
    with open(st_path, "rb") as fh:
        (hlen,) = struct.unpack("<Q", fh.read(8))
        header = json.loads(fh.read(hlen).decode("utf-8"))
        begin, end = header[name]["data_offsets"]
        fh.seek(8 + hlen + begin)
        raw = fh.read(end - begin)
    return np.frombuffer(raw, dtype="<u2")


def main():
    # Optional second pass: half-precision (F16/BF16) cross-check.
    half_sidecar = os.path.join(
        tempfile.gettempdir(), "cai_st_writer_half.json")
    if len(sys.argv) > 2 and sys.argv[2]:
        half_sidecar = sys.argv[2]
    if os.path.exists(half_sidecar):
        hf, hn = verify_half(half_sidecar)
        if hf:
            sys.exit(f"{hf} half-precision failure(s)")
        print(f"OK: {hn} F16/BF16 tensors in the half-precision file match the "
              "F32 source within rounding tolerance (header dtype + values).")

    if len(sys.argv) > 1:
        sidecar_path = sys.argv[1]
    else:
        sidecar_path = os.path.join(
            tempfile.gettempdir(), "cai_safetensors_writer_demo.json")
    if not os.path.exists(sidecar_path):
        sys.exit(f"sidecar not found: {sidecar_path} "
                 "(run `bash tests/RunAll.sh` first)")
    with open(sidecar_path) as f:
        sidecar = json.load(f)

    st_path = sidecar["file"]
    if not os.path.exists(st_path):
        sys.exit(f"safetensors file not found: {st_path}")

    failures = 0
    with safe_open(st_path, framework="numpy") as f:
        meta = f.metadata() or {}
        for k, v in sidecar["metadata"].items():
            if meta.get(k) != v:
                print(f"FAIL metadata[{k}]: expected {v!r}, got {meta.get(k)!r}")
                failures += 1
        names = set(f.keys())
        expected_names = set(sidecar["tensors"].keys())
        if names != expected_names:
            print(f"FAIL tensor names: expected {sorted(expected_names)}, "
                  f"got {sorted(names)}")
            failures += 1
        for name, spec in sidecar["tensors"].items():
            if name not in names:
                continue
            t = f.get_tensor(name)
            if t.dtype != np.float32:
                print(f"FAIL {name}: dtype {t.dtype}, expected float32")
                failures += 1
            if list(t.shape) != spec["shape"]:
                print(f"FAIL {name}: shape {list(t.shape)}, "
                      f"expected {spec['shape']}")
                failures += 1
                continue
            # The sidecar prints 9 significant digits, which round-trips any
            # float32 exactly: the comparison is bit-exact, no tolerance.
            expected = np.array(spec["values"], dtype=np.float32)
            got = t.reshape(-1)
            if not np.array_equal(got, expected):
                bad = np.flatnonzero(got != expected)
                print(f"FAIL {name}: {len(bad)} mismatched elements, "
                      f"first at {bad[0]}: {got[bad[0]]!r} != "
                      f"{expected[bad[0]]!r}")
                failures += 1

    if failures:
        sys.exit(f"{failures} failure(s)")
    n = len(sidecar["tensors"])
    print(f"OK: {n} tensors in {st_path} match the Pascal sidecar bit-exactly "
          "(names, shapes, float32 values, metadata).")


if __name__ == "__main__":
    main()
