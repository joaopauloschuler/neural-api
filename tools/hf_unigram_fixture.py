#!/usr/bin/env python3
"""Generate a tiny-but-real HuggingFace Unigram tokenizer.json fixture plus
pinned encode/decode parity cases for tests/TestNeuralHFTokenizer.pas.

Produces tiny_unigram_tokenizer.json - a SentencePiece-Unigram style
tokenizer (ALBERT / T5 / XLNet / DeBERTa-v3 family): the Unigram model
(vocab of [piece, log_prob] pairs + unk_id) with a Metaspace pre_tokenizer
and the Metaspace decoder. Viterbi segmentation maximises the total piece
log-probability; runs of characters that no vocab piece covers collapse to
a single fused <unk>.

The corpus is deliberately repetitive so the trainer keeps several
multi-character pieces (e.g. "the", "at", "ing") in the vocab -- that is
what exercises the Viterbi DP rather than a trivial per-character split.

Existing fixture is REUSED when present (cases recomputed from the
committed tokenizer.json, which is deterministic) so re-running only
recomputes the pins instead of churning the trained vocab.

Run from the repo root:  python3 tools/hf_unigram_fixture.py
"""
import json
import os

from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

FIXDIR = os.path.join(os.path.dirname(__file__), '..', 'tests', 'fixtures')

# Repetitive corpus so multi-character pieces survive the Unigram pruning.
CORPUS = [
    "the cat sat on the mat and the dog sat on the log",
    "the rat ran to the cat and the cat ran to the mat",
    "running jumping sitting eating sleeping reading writing",
    "the morning the evening the meeting the greeting",
    "she is reading and he is writing and they are eating",
    "a cat a hat a bat a rat a mat a sat a fat cat",
    "the the the the cat cat cat sat sat on on the the mat",
    "nation station relation creation foundation information",
] * 4

CASES = [
    "the cat sat",
    "the cat sat on the mat",
    "running and jumping",
    "the morning meeting",
    "nation station",
    "a fat cat",
    "Hello world",      # capital H / W exercise the unk / per-char paths
    "the QXZ cat",      # QXZ -> fused unk between known pieces
    "eating sleeping",
    "the dog",
    "cat",
    "",
]


def dump_cases(tok, texts):
    cases = []
    for text in texts:
        ids = tok.encode(text, add_special_tokens=False).ids
        cases.append({
            "text": text,
            "ids": ids,
            "decoded": tok.decode(ids, skip_special_tokens=True),
        })
    return cases


def build_unigram(path):
    tok = Tokenizer(models.Unigram())
    tok.pre_tokenizer = pre_tokenizers.Metaspace()
    tok.decoder = decoders.Metaspace()
    trainer = trainers.UnigramTrainer(
        vocab_size=200,
        special_tokens=["<unk>", "<s>", "</s>"],
        unk_token="<unk>")
    tok.train_from_iterator(CORPUS, trainer)
    tok.save(path)


def main():
    path = os.path.join(FIXDIR, 'tiny_unigram_tokenizer.json')
    if not os.path.exists(path):
        build_unigram(path)
    tok = Tokenizer.from_file(path)  # verify (re)load
    cases = dump_cases(tok, CASES)

    cases_path = os.path.join(FIXDIR, 'hf_tokenizer_cases.json')
    out = {}
    if os.path.exists(cases_path):
        with open(cases_path) as f:
            out = json.load(f)
    out["unigram"] = {"tokenizer": os.path.basename(path), "cases": cases}
    with open(cases_path, 'w') as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(path, os.path.getsize(path), 'bytes')
    print(cases_path, os.path.getsize(cases_path), 'bytes')


if __name__ == '__main__':
    main()
