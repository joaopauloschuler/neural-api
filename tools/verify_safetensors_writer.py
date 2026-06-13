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

How to run (not wired into tests/RunAll.sh on purpose):
    bash tests/RunAll.sh                       # writes the demo pair
    /home/bpsa/x/bin/python tools/verify_safetensors_writer.py
Optionally pass the sidecar path as argv[1] if TMPDIR differs.
"""
import json
import os
import sys
import tempfile

import numpy as np
from safetensors import safe_open


def main():
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
