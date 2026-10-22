#!/usr/bin/env python3
"""Generate the CLIP tokenizer.json (fast tokenizer) fixture for
TNeuralHFTokenizer.LoadFromFile, from tiny_sd15's vocab.json + merges.txt.

  tests/fixtures/tiny_sd15_fast_tokenizer/
    tokenizer.json, tokenizer_config.json   CLIPTokenizerFast.save_pretrained
                                            of tests/fixtures/tiny_sd15/tokenizer
    added_tokens/tokenizer.json             the same plus ADDED_TOKENS
                                            (add_tokens: normalized=True)
  tests/fixtures/tiny_sd15_fast_tokenizer_io.json
    cases           per text: ids (add_special_tokens=False) and
                    ids_with_special (the RobertaProcessing bos/eos)
    added_tokens    ADDED_TOKENS as given to add_tokens
    added_cases     per text: ids of the added_tokens tokenizer

Run: /home/bpsa/x/bin/python tools/make_clip_fast_tokenizer_fixture.py
"""
import json
import os
import shutil

from transformers import CLIPTokenizerFast

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
FIXTURES = os.path.join(ROOT, "tests", "fixtures")
SOURCE_DIR = os.path.join(FIXTURES, "tiny_sd15", "tokenizer")
OUT_DIR = os.path.join(FIXTURES, "tiny_sd15_fast_tokenizer")
ADDED_DIR = os.path.join(OUT_DIR, "added_tokens")
IO_FILE = os.path.join(FIXTURES, "tiny_sd15_fast_tokenizer_io.json")

TEXTS = [
    "a photo of an astronaut riding a horse on mars",
    "  Hello,   WORLD!!  it's 2024 \t naïve café ",
    "they'll say we've done it, I'm sure; you're right... isn't it?",
    "IT'S THEY'RE DON'T WE'VE I'M YOU'LL HE'D",
    "'s 't 're 've 'm 'll 'd '' 'x rock'n'roll o'clock",
    "3.14159 + 42 = 45.14159 (approx) $100 #1 @user 2,000,000",
    "!",
    "!!! hello!world! a!b ?!",
    "Ünïcödé: Ñandú, Ærøskøbing, Œuvre, Ÿ ÿ, Łódź ĀĒĪ",
    "日本語のテキスト and emoji 🚀🔥 ok",
    "decomposed e\u0301 vs composed \u00e9",
    "tabs\tand\nnewlines\r\nmixed  \u00a0 nbsp\u3000ideographic",
    "<|startoftext|>A Photo<|endoftext|> of THE moon",
    "<|STARTOFTEXT|> x <|endoftext|>y",
    "MiXeD CaSe WoRdS",
    "",
]

ADDED_TOKENS = ["<cat-toy>", "<Él-Ñandú>"]

ADDED_TEXTS = [
    "a <cat-toy> b",
    "a <CAT-TOY> b <Cat-Toy>",
    "x <Él-Ñandú> <él-ñandú> <ÉL-ÑANDÚ>",
    "<cat-toy><cat-toy> near <|endoftext|>",
]


def main():
    tok = CLIPTokenizerFast.from_pretrained(SOURCE_DIR)
    if os.path.isdir(OUT_DIR):
        shutil.rmtree(OUT_DIR)
    tok.save_pretrained(OUT_DIR)
    cases = []
    for text in TEXTS:
        cases.append({
            "text": text,
            "ids": tok.encode(text, add_special_tokens=False),
            "ids_with_special": tok.encode(text, add_special_tokens=True),
        })
    tok.add_tokens(ADDED_TOKENS)
    tok.save_pretrained(ADDED_DIR)
    # Only tokenizer.json is read from the added-tokens variant.
    for name in os.listdir(ADDED_DIR):
        if name != "tokenizer.json":
            os.remove(os.path.join(ADDED_DIR, name))
    added_cases = [{"text": text,
                    "ids": tok.encode(text, add_special_tokens=False)}
                   for text in ADDED_TEXTS]
    with open(IO_FILE, "w", encoding="utf-8") as f:
        json.dump({"cases": cases, "added_tokens": ADDED_TOKENS,
                   "added_cases": added_cases}, f, ensure_ascii=False,
                  indent=1)
        f.write("\n")


if __name__ == "__main__":
    main()
