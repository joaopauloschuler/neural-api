#!/usr/bin/env python3
"""Generate a TINY raw SentencePiece `.model` fixture plus pinned encode/
decode parity cases for the raw-.model reader in neuralhftokenizer.pas
(LoadSentencePieceModel), exercised by tests/TestNeuralHFTokenizer.pas.

Many checkpoints (T5 / ALBERT / XLNet / DeBERTa-v3 / mBART-Unigram) ship a
raw `sentencepiece.bpe.model` / `spm.model` / `tokenizer.model` ModelProto
protobuf and NO tokenizer.json. This tool trains a tiny UNIGRAM SentencePiece
model with `sentencepiece`, then dumps the reference token ids straight from
the SentencePiece encoder (the oracle the Pascal reader must match).

To keep the committed fixture pico-sized we train with
`normalization_rule_name=identity`, which drops the multi-hundred-KB NFKC
precompiled_charsmap from normalizer_spec (the Pascal reader ignores that
field anyway, and the pinned cases are lowercase ASCII so NFKC would be a
no-op). The resulting .model is < 1 KB.

Cases are lowercase ASCII / single-spaced so the spm normalizer (identity +
remove_extra_whitespaces) and the Pascal Metaspace path agree exactly.

Existing .model fixture is REUSED when present (cases recomputed from the
committed deterministic model) so re-running only refreshes the pins.

Run from the repo root:  /home/bpsa/x/bin/python tools/make_pico_spm_fixture.py

The byte_fallback variant (tiny_spm_bytefallback.model) carries the 256
<0x00>..<0xFF> BYTE pieces (type 6) so the Pascal <0xNN> byte-fallback
encode/decode path is exercised by a round-trip parity test.

The BPE variant (tiny_spm_bpe.model, model_type=BPE) has NO explicit merge
list -- only scored pieces -- so the Pascal reader reconstructs merges from
the pieces (HF generate_merges algorithm) and routes through the metaspace
byte-level-BPE machinery. byte_fallback is forced on so non-ASCII inputs hit
the <0xNN> pieces. NLLB / mBART-BPE / DeBERTa-v3 ship this flavour.

Coded by Claude (AI).
"""
import json
import os

import sentencepiece as spm

FIXDIR = os.path.join(os.path.dirname(__file__), '..', 'tests', 'fixtures')

# Repetitive corpus so multi-character pieces survive the Unigram pruning
# (so the Pascal Viterbi DP is actually exercised, not a per-char split).
CORPUS = [
    "the cat sat on the mat and the dog sat on the log",
    "the rat ran to the cat and the cat ran to the mat",
    "running jumping sitting eating sleeping reading writing",
    "the morning the evening the meeting the greeting",
    "she is reading and he is writing and they are eating",
    "a cat a hat a bat a rat a mat a sat a fat cat",
    "nation station relation creation foundation information",
] * 4

CASES = [
    "the cat sat",
    "the cat sat on the mat",
    "running and jumping",
    "the morning meeting",
    "nation station",
    "a fat cat",
    "eating sleeping",
    "the dog",
    "cat",
    "the qxz cat",   # qxz -> fused unk between known pieces (lowercase)
    "",
]

# byte_fallback cases: characters that are NOT in the tiny vocab and so MUST
# decompose to <0xNN> BYTE pieces -- a 2-byte UTF-8 'e-acute', a 3-byte CJK
# char, a 4-byte emoji and the raw control byte 0x07. The round-trip
# (encode -> decode == original) proves the byte pieces reassemble into the
# correct multi-byte UTF-8 string AND that normal pieces are unaffected.
BYTE_FALLBACK_CASES = [
    "the cat sat",          # all-ASCII, normal pieces (no fallback)
    "café",            # 2-byte UTF-8 (0xC3 0xA9) -> two byte pieces
    "the 你好 cat",  # 3-byte CJK chars mixed with normal pieces
    "smile \U0001f600 now",  # 4-byte emoji
    "bell\x07ring",         # a raw control byte 0x07
    "naïve résumé café",
    "the dog",              # normal
    "",
]


# BPE .model cases: ASCII pieces exercise the reconstructed-merge ranking;
# the non-ASCII tail (cafe / CJK / emoji / control byte) exercises the BPE
# path's <0xNN> byte fallback (byte_fallback is forced on for BPE models).
BPE_CASES = [
    "the cat sat",
    "the cat sat on the mat",
    "running and jumping",
    "the morning meeting",
    "nation station",
    "a fat cat",
    "the dog",
    "cat",
    "the qxz cat",
    "café",
    "the 你好 cat",
    "smile \U0001f600 now",
    "bell\x07ring",
    "",
]


def build_model(path, byte_fallback=False, model_type="unigram"):
    corpus_path = path + '.corpus.txt'
    with open(corpus_path, 'w') as f:
        f.write("\n".join(CORPUS))
    prefix = path[:-len('.model')]
    # BPE models always need byte_fallback to cover the 256-byte alphabet so
    # non-ASCII test cases (cafe / CJK / emoji) decompose to <0xNN> pieces --
    # exactly like the real NLLB/mBART-BPE sentencepiece.bpe.model exports.
    bf = byte_fallback or (model_type == "bpe")
    # vocab budget: byte_fallback adds the 256 <0x00>..<0xFF> BYTE pieces
    # (type 6) on top of the ~23-char learned alphabet, so it must clear 282.
    if bf:
        vocab_size = 320
    else:
        vocab_size = 60
    kwargs = dict(
        input=corpus_path,
        model_prefix=prefix,
        vocab_size=vocab_size,
        model_type=model_type,
        character_coverage=1.0,
        normalization_rule_name="identity",  # no NFKC charsmap -> tiny .model
        unk_id=0, bos_id=1, eos_id=2, pad_id=-1,
        unk_piece="<unk>", bos_piece="<s>", eos_piece="</s>")
    if bf:
        kwargs['byte_fallback'] = True
    spm.SentencePieceTrainer.train(**kwargs)
    os.remove(corpus_path)
    # spm also writes a .vocab sibling we do not need.
    vocab_sibling = prefix + '.vocab'
    if os.path.exists(vocab_sibling):
        os.remove(vocab_sibling)


def dump_cases(sp, texts=CASES):
    cases = []
    for text in texts:
        ids = sp.encode(text)
        cases.append({
            "text": text,
            "ids": ids,
            "decoded": sp.decode(ids),
        })
    return cases


def main():
    path = os.path.join(FIXDIR, 'tiny_spm.model')
    if not os.path.exists(path):
        build_model(path)
    sp = spm.SentencePieceProcessor(model_file=path)  # verify (re)load
    cases = dump_cases(sp)

    bf_path = os.path.join(FIXDIR, 'tiny_spm_bytefallback.model')
    if not os.path.exists(bf_path):
        build_model(bf_path, byte_fallback=True)
    bf_sp = spm.SentencePieceProcessor(model_file=bf_path)  # verify (re)load
    bf_cases = dump_cases(bf_sp, BYTE_FALLBACK_CASES)

    # BPE SentencePiece .model (model_type=BPE). NLLB/mBART-BPE and DeBERTa-v3
    # ship this flavour. The .model stores only scored pieces -- no explicit
    # merge list -- so the Pascal reader reconstructs merges from the pieces
    # the same way HF's transformers.tokenization_utils_base.generate_merges
    # does for BPE SpmConverter models. We verified (tools comment / commit)
    # that SentencePiece's own encode is id-identical to a tokenizers BPE model
    # built from those reconstructed merges over ASCII *and* non-ASCII (byte
    # fallback) inputs, so the sentencepiece encoder is a faithful oracle here.
    bpe_path = os.path.join(FIXDIR, 'tiny_spm_bpe.model')
    if not os.path.exists(bpe_path):
        build_model(bpe_path, model_type="bpe")
    bpe_sp = spm.SentencePieceProcessor(model_file=bpe_path)  # verify (re)load
    bpe_cases = dump_cases(bpe_sp, BPE_CASES)

    cases_path = os.path.join(FIXDIR, 'hf_tokenizer_cases.json')
    out = {}
    if os.path.exists(cases_path):
        with open(cases_path) as f:
            out = json.load(f)
    out["spm_model"] = {"tokenizer": os.path.basename(path), "cases": cases}
    out["spm_bytefallback"] = {"tokenizer": os.path.basename(bf_path),
                               "cases": bf_cases}
    out["spm_bpe_model"] = {"tokenizer": os.path.basename(bpe_path),
                            "cases": bpe_cases}
    with open(cases_path, 'w') as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(path, os.path.getsize(path), 'bytes')
    print('vocab', sp.get_piece_size(),
          'unk/bos/eos', sp.unk_id(), sp.bos_id(), sp.eos_id())
    print(bf_path, os.path.getsize(bf_path), 'bytes',
          'vocab', bf_sp.get_piece_size())
    print(bpe_path, os.path.getsize(bpe_path), 'bytes',
          'vocab', bpe_sp.get_piece_size())
    print(cases_path, os.path.getsize(cases_path), 'bytes')


if __name__ == '__main__':
    main()
