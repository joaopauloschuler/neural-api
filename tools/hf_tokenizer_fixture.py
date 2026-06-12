#!/usr/bin/env python3
"""Generate tiny-but-realistic HuggingFace tokenizer.json fixtures plus
pinned encode/decode parity cases for tests/TestNeuralHFTokenizer.pas.

Three tokenizers are produced (all a few KB):
  1. tiny_bpe_bytelevel_tokenizer.json  - GPT-2/distilgpt2/SmolLM2 style:
     byte-level BPE (ByteLevel pre-tokenizer with the GPT-2 regex,
     bytes-to-unicode alphabet, <|endoftext|> special token).
  2. tiny_bpe_metaspace_tokenizer.json  - Llama/TinyLlama style: BPE with
     byte_fallback, Prepend("▁")+Replace(" ","▁") normalizer, no
     pre-tokenizer, <unk>/<s>/</s> specials, Llama's decoder chain.
  3. tiny_wordpiece_tokenizer.json      - BERT/MiniLM style: WordPiece with
     BertNormalizer (lowercase + strip accents), BertPreTokenizer,
     [PAD]/[UNK]/[CLS]/[SEP]/[MASK] specials and the WordPiece decoder.

Existing tokenizer fixtures are REUSED when present (so re-running only
ADDS new groups instead of re-training and churning the pinned ids); their
cases are recomputed from the committed tokenizer.json, which is
deterministic.

The expected ids/decoded strings for a battery of inputs (ASCII,
punctuation, multi-space, accents, emoji, specials) are pinned into
tests/fixtures/hf_tokenizer_cases.json. The Pascal TNeuralHFTokenizer must
match them exactly.

Run from the repo root:  python3 tools/hf_tokenizer_fixture.py
"""
import json
import os

from tokenizers import Tokenizer, models, trainers, pre_tokenizers, \
    decoders, normalizers

FIXDIR = os.path.join(os.path.dirname(__file__), '..', 'tests', 'fixtures')

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
    "Don't stop, don't ever stop!",
    "numbers 12345 and 3.14",
    "café naïve résumé",
    "emoji \U0001f600 test",
    "mixed: café 42 \U0001f680!",
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


def build_bytelevel():
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=380, special_tokens=["<|endoftext|>"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    tok.train_from_iterator(CORPUS, trainer)
    path = os.path.join(FIXDIR, 'tiny_bpe_bytelevel_tokenizer.json')
    tok.save(path)
    tok = Tokenizer.from_file(path)  # verify reload
    return path, dump_cases(tok, CASES, ["x<|endoftext|>y"])


def build_metaspace():
    byte_tokens = ["<0x%02X>" % i for i in range(256)]
    tok = Tokenizer(models.BPE(unk_token="<unk>", byte_fallback=True,
                               fuse_unk=True))
    tok.normalizer = normalizers.Sequence([
        normalizers.Prepend("▁"),
        normalizers.Replace(" ", "▁"),
    ])
    tok.decoder = decoders.Sequence([
        decoders.Replace("▁", " "),
        decoders.ByteFallback(),
        decoders.Fuse(),
        decoders.Strip(" ", 1, 0),
    ])
    trainer = trainers.BpeTrainer(
        vocab_size=560,
        special_tokens=["<unk>", "<s>", "</s>"] + byte_tokens)
    tok.train_from_iterator(CORPUS, trainer)
    path = os.path.join(FIXDIR, 'tiny_bpe_metaspace_tokenizer.json')
    tok.save(path)

    # The byte tokens belong in the *model* vocab (byte_fallback reads them
    # there) but real Llama tokenizer.json files do NOT list them under
    # added_tokens. Strip them out so the fixture matches the real layout.
    with open(path) as f:
        data = json.load(f)
    data['added_tokens'] = [t for t in data['added_tokens']
                            if not t['content'].startswith('<0x')]
    with open(path, 'w') as f:
        json.dump(data, f, ensure_ascii=False)
    tok = Tokenizer.from_file(path)  # verify reload after edit
    return path, dump_cases(tok, CASES, ["<s>hello</s>"])


def build_wordpiece():
    path = os.path.join(FIXDIR, 'tiny_wordpiece_tokenizer.json')
    if not os.path.exists(path):
        tok = Tokenizer(models.WordPiece(unk_token="[UNK]"))
        tok.normalizer = normalizers.BertNormalizer(lowercase=True)
        tok.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
        tok.decoder = decoders.WordPiece(prefix="##", cleanup=True)
        trainer = trainers.WordPieceTrainer(
            vocab_size=420,
            special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"])
        tok.train_from_iterator(CORPUS, trainer)
        tok.save(path)
    tok = Tokenizer.from_file(path)  # verify (re)load
    return path, dump_cases(tok, CASES, [
        "[CLS] hello world [SEP]",
        "supercalifragilisticexpialidocious",
        "UPPER Case AND aCCenTs: CAFÉ",
    ])


def build_reusing(builder, name):
    # reuse a committed tokenizer fixture instead of re-training (cases
    # recomputed from the file are deterministic; only missing fixtures
    # get trained)
    path = os.path.join(FIXDIR, name)
    if os.path.exists(path):
        tok = Tokenizer.from_file(path)
        extra = (["x<|endoftext|>y"] if 'bytelevel' in name
                 else ["<s>hello</s>"])
        return path, dump_cases(tok, CASES, extra)
    return builder()


def main():
    bl_path, bl_cases = build_reusing(build_bytelevel,
                                      'tiny_bpe_bytelevel_tokenizer.json')
    ms_path, ms_cases = build_reusing(build_metaspace,
                                      'tiny_bpe_metaspace_tokenizer.json')
    wp_path, wp_cases = build_wordpiece()
    out = {
        "byte_level": {"tokenizer": os.path.basename(bl_path),
                       "cases": bl_cases},
        "metaspace": {"tokenizer": os.path.basename(ms_path),
                      "cases": ms_cases},
        "wordpiece": {"tokenizer": os.path.basename(wp_path),
                      "cases": wp_cases},
    }
    cases_path = os.path.join(FIXDIR, 'hf_tokenizer_cases.json')
    with open(cases_path, 'w') as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    for p in (bl_path, ms_path, wp_path, cases_path):
        print(p, os.path.getsize(p), 'bytes')


if __name__ == '__main__':
    main()
