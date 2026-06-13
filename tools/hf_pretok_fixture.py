#!/usr/bin/env python3
r"""Generate tiny tokenizer.json fixtures for the Split-regex and Metaspace
PRE_TOKENIZER support in neural/neuralhftokenizer.pas, plus pinned
encode/decode parity cases (appended to tests/fixtures/hf_tokenizer_cases.json
next to the groups produced by tools/hf_tokenizer_fixture.py).

Three tokenizers are produced (all a few KB):
  1. tiny_bpe_split_qwen2_tokenizer.json  - Qwen2/Qwen3 style: byte-level
     BPE with pre_tokenizer Sequence[Split(qwen2 cl100k-style regex,
     behavior=Isolated), ByteLevel(add_prefix_space=False,
     use_regex=False)], ByteLevel decoder, <|endoftext|>/<|im_end|>
     specials.
  2. tiny_bpe_split_cl100k_tokenizer.json - Llama-3/cl100k style: same but
     with \p{N}{1,3} (digit runs capped at 3).
  3. tiny_bpe_metaspace_pretok_tokenizer.json - Mistral / legacy=false
     Llama style: BPE with byte_fallback, NO normalizer, pre_tokenizer
     Metaspace(replacement="▁", prepend_scheme="first", split=True) and
     the Metaspace decoder.

Existing tokenizer fixtures are REUSED when present (re-running only
recomputes the deterministic cases instead of re-training).

Run from the repo root:  python3 tools/hf_pretok_fixture.py
"""
import json
import os

from tokenizers import Tokenizer, models, trainers, pre_tokenizers, \
    decoders, Regex

FIXDIR = os.path.join(os.path.dirname(__file__), '..', 'tests', 'fixtures')

QWEN2_PATTERN = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}"
    r"| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+")
CL100K_PATTERN = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}"
    r"| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+")

CORPUS = [
    "The quick brown fox jumps over the lazy dog.",
    "Pack my box with five dozen liquor jugs!",
    "How vexingly quick daft zebras jump?",
    "She sells seashells by the seashore, and the shells she sells are surely seashells.",
    "Don't count your chickens before they hatch; it's never wise.",
    "In 2026 there were 365 days, 12 months and 52 weeks.",
    "A naive cafe patron ordered a souffle and a crepe.",
    "the cat sat on the mat. the dog sat on the log.",
    "to be or not to be, that is the question.",
    "numbers like 1, 22, 333 and 4444 appear here.",
]

CASES = [
    "Hello world",
    "Hello, world!",
    "the cat sat on the mat",
    "  leading and  multiple   spaces",
    "trailing space ",
    "tabs\tand\nnewlines",
    "lines\n\n  with blanks\r\nand CRLF\n",
    "Don't stop, DON'T ever STOP! 'Tis it'LL be",
    "numbers 12345 and 3.14",
    "café naïve résumé",
    "emoji \U0001f600 test",
    "mixed: café 42 \U0001f680!",
    "punct!!?\n\nthen text",
    "",
]


def dump_cases(tok, texts, extra):
    cases = []
    for text in texts + extra:
        ids = tok.encode(text, add_special_tokens=False).ids
        cases.append({
            "text": text,
            "ids": ids,
            "decoded": tok.decode(ids, skip_special_tokens=True),
        })
    return cases


def build_split(name, pattern):
    path = os.path.join(FIXDIR, name)
    if not os.path.exists(path):
        tok = Tokenizer(models.BPE())
        tok.pre_tokenizer = pre_tokenizers.Sequence([
            pre_tokenizers.Split(Regex(pattern), behavior="isolated",
                                 invert=False),
            pre_tokenizers.ByteLevel(add_prefix_space=False,
                                     use_regex=False),
        ])
        tok.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(
            vocab_size=400, special_tokens=["<|endoftext|>", "<|im_end|>"],
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
        tok.train_from_iterator(CORPUS, trainer)
        tok.save(path)
    tok = Tokenizer.from_file(path)  # verify (re)load
    return path, dump_cases(tok, CASES, ["x<|im_end|>y"])


def build_metaspace_pretok():
    path = os.path.join(FIXDIR, 'tiny_bpe_metaspace_pretok_tokenizer.json')
    if not os.path.exists(path):
        byte_tokens = ["<0x%02X>" % i for i in range(256)]
        tok = Tokenizer(models.BPE(unk_token="<unk>", byte_fallback=True,
                                   fuse_unk=True))
        tok.pre_tokenizer = pre_tokenizers.Metaspace(
            replacement="▁", prepend_scheme="first", split=True)
        tok.decoder = decoders.Sequence([
            decoders.Metaspace(replacement="▁", prepend_scheme="first",
                               split=True),
            decoders.ByteFallback(),
            decoders.Fuse(),
        ])
        trainer = trainers.BpeTrainer(
            vocab_size=560,
            special_tokens=["<unk>", "<s>", "</s>"] + byte_tokens)
        tok.train_from_iterator(CORPUS, trainer)
        tok.save(path)
        # byte tokens live in the model vocab, not added_tokens (real
        # Llama/Mistral layout) -- same edit as tools/hf_tokenizer_fixture.py
        with open(path) as f:
            data = json.load(f)
        data['added_tokens'] = [t for t in data['added_tokens']
                                if not t['content'].startswith('<0x')]
        with open(path, 'w') as f:
            json.dump(data, f, ensure_ascii=False)
    tok = Tokenizer.from_file(path)  # verify (re)load
    return path, dump_cases(tok, CASES, ["<s>hello world</s>",
                                         "hello<s>not first segment"])


def main():
    q_path, q_cases = build_split('tiny_bpe_split_qwen2_tokenizer.json',
                                  QWEN2_PATTERN)
    c_path, c_cases = build_split('tiny_bpe_split_cl100k_tokenizer.json',
                                  CL100K_PATTERN)
    m_path, m_cases = build_metaspace_pretok()
    cases_path = os.path.join(FIXDIR, 'hf_tokenizer_cases.json')
    with open(cases_path) as f:
        out = json.load(f)
    out["split_qwen2"] = {"tokenizer": os.path.basename(q_path),
                          "cases": q_cases}
    out["split_cl100k"] = {"tokenizer": os.path.basename(c_path),
                           "cases": c_cases}
    out["metaspace_pretok"] = {"tokenizer": os.path.basename(m_path),
                               "cases": m_cases}
    with open(cases_path, 'w') as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    for p in (q_path, c_path, m_path, cases_path):
        print(p, os.path.getsize(p), 'bytes')


if __name__ == '__main__':
    main()
