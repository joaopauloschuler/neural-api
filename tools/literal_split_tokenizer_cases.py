#!/usr/bin/env python3
"""Expected ids for TTestNeuralHFTokenizer.TestLiteralSplitParityWithHF.

Builds the tiny metaspace-BPE tokenizer.json the Pascal test writes (the
Gemma-3/4 / EmbeddingGemma-2 form: Replace(" " -> U+2581) normalizer +
Split({"String": " "}, MergedWithPrevious), and literal "-" splits with every
behavior, no normalizer) and prints the HF `tokenizers` ids for each case so
they can be pinned in the test.

Coded by Claude (AI).

Usage: python3 tools/literal_split_tokenizer_cases.py
"""
import json

from tokenizers import Tokenizer

VOCAB = {'a': 0, 'b': 1, 'c': 2, '▁': 3, '-': 4, ' ': 5,
         'ab': 6, 'abc': 7, '▁a': 8, '▁ab': 9, 'c-': 10,
         '-a': 11, 'b▁': 12}
MERGES = [['a', 'b'], ['ab', 'c'], ['▁', 'a'], ['▁a', 'b'],
          ['c', '-'], ['-', 'a'], ['b', '▁']]
TEXTS = ['ab c-abc', 'abc--ab', '-ab-', 'a b  c', 'ab']


def make(normalizer, behavior, pattern):
    return {
        'version': '1.0',
        'model': {'type': 'BPE', 'vocab': VOCAB, 'merges': MERGES,
                  'byte_fallback': False, 'fuse_unk': False,
                  'ignore_merges': False, 'dropout': None,
                  'unk_token': None, 'continuing_subword_prefix': None,
                  'end_of_word_suffix': None},
        'normalizer': normalizer,
        'pre_tokenizer': {'type': 'Split', 'pattern': {'String': pattern},
                          'behavior': behavior, 'invert': False},
    }


gemma_norm = {'type': 'Replace', 'pattern': {'String': ' '},
              'content': '▁'}
cases = [('gemma', make(gemma_norm, 'MergedWithPrevious', ' '))]
for behavior in ('MergedWithPrevious', 'MergedWithNext', 'Isolated',
                 'Removed'):
    cases.append((behavior, make(None, behavior, '-')))
for name, spec in cases:
    tok = Tokenizer.from_str(json.dumps(spec))
    print(name, [tok.encode(t, add_special_tokens=False).ids for t in TEXTS])
