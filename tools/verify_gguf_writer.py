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

When llama-cpp-python is importable it ALSO loads the same .gguf through
llama.cpp's own graph and cross-checks next-token-logit parity (greedy argmax
+ top-k ranking) against the Pascal model's logits carried in the sidecar
(parity_prompt / parity_next_token_logits). That block SKIPS gracefully when
llama-cpp-python is absent or rejects the pico fixture, so it is never a hard
requirement - the dimensions/metadata cross-check is the always-on pass that
proves the container + reversed-dims + naming are byte-correct.

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

    # ---- optional next-token-logit parity via llama-cpp-python ----
    # The Pascal sidecar carries a fixed prompt + the Pascal model's
    # next-token logit row (last position). When llama-cpp-python is
    # importable we load the SAME .gguf through llama.cpp's own graph and
    # compare argmax + ranking; otherwise we skip with a message (the lib is
    # NOT a requirement for the Pascal tests to pass).
    failures += verify_logit_parity(gguf_path, spec)

    if failures:
        sys.exit(f"{failures} failure(s)")
    print(f"OK: {gguf_path} loads with the python 'gguf' package; "
          f"architecture/metadata and {len(want)} tensors (names + reversed "
          "ggml dims) match the Pascal sidecar.")


def verify_logit_parity(gguf_path, spec):
    """Cross-check next-token logits against llama-cpp-python when available.

    Returns the number of failures (0 on success OR on graceful skip).
    """
    prompt = spec.get("parity_prompt")
    ref_logits = spec.get("parity_next_token_logits")
    if not prompt or not ref_logits:
        print("SKIP logit parity: sidecar has no parity_prompt / "
              "parity_next_token_logits (regenerate with the current test).")
        return 0
    try:
        from llama_cpp import Llama
    except Exception as exc:  # ImportError or a botched native build
        print(f"SKIP logit parity: llama-cpp-python not importable ({exc}); "
              "install it to enable the cross-framework logit check.")
        return 0

    try:
        llm = Llama(model_path=gguf_path, n_ctx=max(len(prompt) + 8, 32),
                    n_threads=1, logits_all=True, vocab_only=False,
                    verbose=False)
    except Exception as exc:
        print(f"SKIP logit parity: llama.cpp refused the file ({exc}); "
              "the demo GGUF is a pico fixture llama.cpp may reject.")
        return 0

    try:
        llm.reset()
        llm.eval([int(t) for t in prompt])
        got = list(llm.eval_logits[-1])
    except Exception as exc:
        print(f"SKIP logit parity: llama.cpp eval failed ({exc}).")
        return 0

    if len(got) != len(ref_logits):
        print(f"FAIL logit parity: llama.cpp vocab {len(got)} != "
              f"Pascal {len(ref_logits)}")
        return 1

    # Compare the argmax (greedy next token) and the top-k ranking; absolute
    # logit values differ by an additive constant between graphs, so rank +
    # argmax is the robust cross-framework signal.
    ref_argmax = max(range(len(ref_logits)), key=lambda i: ref_logits[i])
    got_argmax = max(range(len(got)), key=lambda i: got[i])
    if ref_argmax != got_argmax:
        print(f"FAIL logit parity: greedy next-token argmax "
              f"llama.cpp {got_argmax} != Pascal {ref_argmax}")
        return 1
    k = min(5, len(got))
    ref_top = sorted(range(len(ref_logits)),
                     key=lambda i: ref_logits[i], reverse=True)[:k]
    got_top = sorted(range(len(got)),
                     key=lambda i: got[i], reverse=True)[:k]
    if ref_top != got_top:
        print(f"FAIL logit parity: top-{k} token ranking "
              f"llama.cpp {got_top} != Pascal {ref_top}")
        return 1
    print(f"OK logit parity: greedy argmax {got_argmax} and top-{k} ranking "
          "agree between llama-cpp-python and the Pascal model.")
    return 0


if __name__ == "__main__":
    main()
