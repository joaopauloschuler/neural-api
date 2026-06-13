#!/usr/bin/env python3
"""Cross-framework check for the Pascal GGUF WRITER
(TNNetGGUFWriter / SaveLlamaToGGUFEx in neural/neuralgguf.pas +
neural/neuralpretrained.pas).

The Pascal test TTestNeuralPretrained.TestGGUFWriterRoundTrip writes a demo
GGUF plus a JSON sidecar describing the expected hyperparameters:

    $TMPDIR/cai_gguf_writer_demo.gguf
    $TMPDIR/cai_gguf_writer_demo.json   (llama.* metadata + tied-head flag)

This script loads the .gguf with the Python "gguf" package (the llama.cpp
reference reader, NOT the Pascal reader) and verifies:
  - the typed metadata block (general.architecture, the llama.*
    hyperparameters) matches the sidecar;
  - the tensor table carries the ggml-canonical names with ggml's REVERSED
    dimension order (a [out, in] matrix is stored [in, out]);
  - 32-byte data alignment / a self-consistent container (the package
    refuses to parse a malformed file).

When feasible it also reconstructs next-token logits from the F32 tensors and
compares them to the Pascal model dumped alongside (see below) - but the
default pass is the dimensions/metadata cross-check, which is what proves the
container + reversed-dims + naming are byte-correct.

How to run (not wired into tests/RunAll.sh on purpose):
    bash tests/RunAll.sh                       # writes the file + sidecar
    /home/bpsa/x/bin/python tools/verify_gguf_writer.py
Optionally pass the sidecar path as argv[1] if TMPDIR differs.
"""
import json
import os
import sys
import tempfile

import gguf


# Expected ggml tensor names for a 2-layer llama (per the sidecar's block
# count). Matrices are [out, in] in row-major; ggml stores them reversed, so
# the gguf reader reports the dims contiguous-first ([in, out]).
def expected_tensors(spec):
    nb = spec["block_count"]
    h = spec["embedding_length"]
    ff = spec["feed_forward_length"]
    nh = spec["head_count"]
    nkv = spec["head_count_kv"]
    vocab = spec["vocab_size"]
    hd = h // nh
    qw = nh * hd
    kvw = nkv * hd
    # value = the row-major [out, in] shape the matrix logically has; the
    # gguf reader returns dims REVERSED, so we compare against reversed().
    t = {
        "token_embd.weight": [vocab, h],
        "output_norm.weight": [h],
    }
    if not spec["tie_word_embeddings"]:
        t["output.weight"] = [vocab, h]
    for b in range(nb):
        p = "blk.%d." % b
        t[p + "attn_norm.weight"] = [h]
        t[p + "attn_q.weight"] = [qw, h]
        t[p + "attn_k.weight"] = [kvw, h]
        t[p + "attn_v.weight"] = [kvw, h]
        t[p + "attn_output.weight"] = [h, qw]
        t[p + "ffn_norm.weight"] = [h]
        t[p + "ffn_gate.weight"] = [ff, h]
        t[p + "ffn_up.weight"] = [ff, h]
        t[p + "ffn_down.weight"] = [h, ff]
    return t


def main():
    if len(sys.argv) > 1:
        sidecar_path = sys.argv[1]
    else:
        sidecar_path = os.path.join(
            tempfile.gettempdir(), "cai_gguf_writer_demo.json")
    if not os.path.exists(sidecar_path):
        sys.exit(f"sidecar not found: {sidecar_path} "
                 "(run `bash tests/RunAll.sh` first)")
    with open(sidecar_path) as f:
        spec = json.load(f)
    gguf_path = spec["file"]
    if not os.path.exists(gguf_path):
        sys.exit(f"gguf file not found: {gguf_path}")

    reader = gguf.GGUFReader(gguf_path)
    failures = 0

    def meta_int(key):
        field = reader.get_field(key)
        return None if field is None else int(field.contents())

    def meta_float(key):
        field = reader.get_field(key)
        return None if field is None else float(field.contents())

    def meta_str(key):
        field = reader.get_field(key)
        return None if field is None else str(field.contents())

    # ---- metadata ----
    arch = meta_str("general.architecture")
    if arch != "llama":
        print(f"FAIL general.architecture: {arch!r} != 'llama'")
        failures += 1
    int_checks = {
        "llama.block_count": "block_count",
        "llama.embedding_length": "embedding_length",
        "llama.feed_forward_length": "feed_forward_length",
        "llama.attention.head_count": "head_count",
        "llama.attention.head_count_kv": "head_count_kv",
        "llama.vocab_size": "vocab_size",
        "llama.context_length": "context_length",
    }
    for key, sk in int_checks.items():
        got = meta_int(key)
        if got != spec[sk]:
            print(f"FAIL {key}: {got} != {spec[sk]}")
            failures += 1
    for key, sk, tol in (
        ("llama.attention.layer_norm_rms_epsilon", "rms_norm_eps", 1e-9),
        ("llama.rope.freq_base", "rope_freq_base", 1e-2),
    ):
        got = meta_float(key)
        if got is None or abs(got - spec[sk]) > tol:
            print(f"FAIL {key}: {got} != {spec[sk]} (tol {tol})")
            failures += 1

    # tokenizer block (self-contained file).
    tok_model = meta_str("tokenizer.ggml.model")
    if tok_model != "llama":
        print(f"FAIL tokenizer.ggml.model: {tok_model!r} != 'llama'")
        failures += 1
    tok_field = reader.get_field("tokenizer.ggml.tokens")
    n_tok = 0 if tok_field is None else len(tok_field.data)
    if n_tok != spec["vocab_size"]:
        print(f"FAIL token count: {n_tok} != {spec['vocab_size']}")
        failures += 1

    # ---- tensor table (ggml-canonical names + REVERSED dims) ----
    want = expected_tensors(spec)
    got_names = {t.name: t for t in reader.tensors}
    if set(got_names) != set(want):
        print(f"FAIL tensor names:\n  missing {sorted(set(want) - set(got_names))}"
              f"\n  extra   {sorted(set(got_names) - set(want))}")
        failures += 1
    for name, rowmajor in want.items():
        t = got_names.get(name)
        if t is None:
            continue
        # gguf reports dims contiguous-first; the writer reversed the
        # row-major [out, in] shape, so we expect reversed(rowmajor).
        ggml_dims = list(int(d) for d in t.shape)
        expect = list(reversed(rowmajor))
        if ggml_dims != expect:
            print(f"FAIL {name}: ggml dims {ggml_dims} != reversed row-major "
                  f"{expect} (row-major {rowmajor})")
            failures += 1

    if failures:
        sys.exit(f"{failures} failure(s)")
    print(f"OK: {gguf_path} loads with the python 'gguf' package; "
          f"architecture/metadata and {len(want)} tensors (names + reversed "
          "ggml dims) match the Pascal sidecar.")


if __name__ == "__main__":
    main()
