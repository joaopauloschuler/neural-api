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
# Qwen3.5/3.6: the Qwen2 pattern with [\p{L}\p{M}]+ letter runs (combining
# marks JOIN the letter run) and a \p{M}-excluding punct class.
QWEN35_PATTERN = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}"
    r"| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+")
# o200k_base / GPT-4o family (gpt-4o, gpt-4o-mini, o1/o3). The canonical
# tiktoken o200k_base pat_str, ported to the HF `tokenizers` Split form.
O200K_PATTERN = (
    r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*"
    r"[\p{Ll}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|"
    r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+"
    r"[\p{Ll}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|"
    r"\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+")

# The DeepSeek-V2 / V2-Lite pre_tokenizer is a Sequence of several Split
# stages with huge explicit Unicode letter/punct/CJK ranges + a
# Digits(individual) stage + ByteLevel(use_regex=False). The exact letter
# range string is the one shipped by the real checkpoint; it is embedded
# verbatim so the fixture is self-contained (a few KB of text, no download
# needed to regenerate the cases). Mirror of
# deepseek-ai/DeepSeek-V2-Lite tokenizer.json pre_tokenizer.
DEEPSEEK_LETTER_CLASS = (
    "A-Za-zµÀ-ÖØ-öø-ƺƼ-ƿ"
    "Ǆ-ʓʕ-ʯͰ-ͳͶͷͻ-ͽ"
    "ͿΆΈ-ΊΌΎ-ΡΣ-ϵ"
    "Ϸ-ҁҊ-ԯԱ-ՖႠ-Ⴥ"
    "Ꭰ-Ᏽᏸ-ᏽᲐ-ᲺᲽ-Ჿ"
    "ᴀ-ᴫᵫ-ᵷᵹ-ᶚḀ-ἕ"
    "Ἐ-Ἕἠ-ὅὈ-Ὅὐ-ὗὙ"
    "ὛὝὟ-ώᾀ-ᾴᾶ-ᾼι"
    "ῂ-ῄῆ-ῌῐ-ΐῖ-Ί"
    "ῠ-Ῥῲ-ῴῶ-ῼℂℇ"
    "ℊ-ℓℕℙ-ℝℤΩℨ"
    "K-ℭℯ-ℴℹℼ-ℿⅅ-ⅉ"
    "ⅎↃↄⰀ-ⱻⱾ-ⳤⳫ-ⳮ"
    "ⳲⳳꙀ-ꙭꚀ-ꚛꜢ-ꝯ"
    "ꝱ-ꞇꞋ-ꞎꭰ-ꮿﬀ-ﬆ"
    "ﬓ-ﬗＡ-Ｚａ-ｚ"
    "\U00010400-\U0001044f\U000104b0-\U000104d3\U000104d8-\U000104fb"
    "\U00010c80-\U00010cb2\U00010cc0-\U00010cf2\U000118a0-\U000118df"
    "\U0001e900-\U0001e943")
DEEPSEEK_PRE_TOKENIZER = {
    "type": "Sequence",
    "pretokenizers": [
        {"type": "Split", "pattern": {"Regex": "[\r\n]"},
         "behavior": "Isolated", "invert": False},
        {"type": "Split",
         "pattern": {"Regex": r"\s?[" + DEEPSEEK_LETTER_CLASS + "]+"},
         "behavior": "Isolated", "invert": False},
        {"type": "Split",
         "pattern": {"Regex": "\\s?[!-/:-~！-／：-～"
                     "‘-‟　-。]+"},
         "behavior": "Isolated", "invert": False},
        {"type": "Split", "pattern": {"Regex": r"\s+$"},
         "behavior": "Isolated", "invert": False},
        {"type": "Split",
         "pattern": {"Regex": "[一-龥ࠀ-一"
                     "가-퟿]+"},
         "behavior": "Isolated", "invert": False},
        {"type": "Digits", "individual_digits": True},
        {"type": "ByteLevel", "add_prefix_space": False,
         "trim_offsets": True, "use_regex": False},
    ],
}

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


def build_split(name, pattern, extra_corpus=(), cases=None):
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
        tok.train_from_iterator(CORPUS + list(extra_corpus), trainer)
        tok.save(path)
    tok = Tokenizer.from_file(path)  # verify (re)load
    return path, dump_cases(tok, cases if cases is not None else CASES,
                            ["x<|im_end|>y"])


# Qwen3.5 marks-join corpus/cases: ASCII plus combining marks --
# Devanagari consonants+vowel signs/virama, Hebrew niqqud, Arabic harakat,
# and Latin + combining acute (incl. x+U+0301 which has NO precomposed NFC
# form). The corpus makes the trainer learn merges that SPAN letter<->mark
# boundaries, so a splitter that (incorrectly) isolates marks produces
# different ids. The Bengali/Gujarati/Thai entries pin the real-vocab
# fuzzing regressions found against the actual Qwen3.6-27B tokenizer: the
# pattern's ANY-char letter-run prefix ([^\r\n\p{L}\p{N}]?, e.g. '?s' after
# a Bengali letter) and exotic-script \p{N} digits (a space before Gujarati
# U+0AE8 must stay a standalone token) -- both require the EXACT Unicode
# category tables (neuralunicodeclasses.inc), not a major-script subset.
QWEN35_MARK_CORPUS = [
    ("\u0928\u092E\u0938\u094D\u0924\u0947 "      # namaste
     "\u0926\u0941\u0928\u093F\u092F\u093E "      # duniya
     "\u0928\u092E\u0938\u094D\u0924\u0947"),
    ("\u0939\u093F\u0928\u094D\u0926\u0940 "      # hindi
     "\u092E\u0947\u0902 "                       # mein
     "\u0939\u0948"),                             # hai
    ("\u05E9\u05B8\u05C1\u05DC\u05D5\u05B9\u05DD "  # shalom
     "\u05E2\u05D5\u05B9\u05DC\u05B8\u05DD "        # olam
     "\u05E9\u05B8\u05C1\u05DC\u05D5\u05B9\u05DD"),
    ("\u0645\u064E\u0631\u0652\u062D\u064E\u0628\u064B\u0627 "  # marhaban
     "\u0628\u0650\u0643\u064F\u0645\u0652"),                    # bikum
    "combining x\u0301 y\u0301 x\u0301x\u0301 letters",
    "cafe\u0301 re\u0301sume\u0301 cafe\u0301 naive",
    # real-vocab regression material (see comment above): Bengali letters
    # followed by ?s / !s (teaches the '?s'-spanning merge), Gujarati
    # digits, Thai letters
    "\u09aa?s \u0995?s ?s !s \u09aa?s ?s",
    "\u09aa\u09be\u09a8\u09bf \u09aa\u09be\u09a8\u09bf \u0995\u09b2",
    "\u0ae7\u0ae8\u0ae9 \u0ae8 \u0ae7\u0ae8\u0ae9",
    "\u0e2a\u0e27\u0e31\u0e2a\u0e14\u0e35 \u0e44\u0e17\u0e22 "
    "\u0e2a\u0e27\u0e31\u0e2a\u0e14\u0e35",
]

QWEN35_CASES = CASES + [
    "\u0928\u092E\u0938\u094D\u0924\u0947",                     # namaste
    ("\u0928\u092E\u0938\u094D\u0924\u0947 "
     "\u0926\u0941\u0928\u093F\u092F\u093E"),
    "\u0939\u093F\u0928\u094D\u0926\u0940 123 \u0939\u0948",
    ("\u05E9\u05B8\u05C1\u05DC\u05D5\u05B9\u05DD "
     "\u05E2\u05D5\u05B9\u05DC\u05B8\u05DD"),
    ("\u0645\u064E\u0631\u0652\u062D\u064E\u0628\u064B\u0627 "
     "\u0628\u0650\u0643\u064F\u0645\u0652"),
    "x\u0301 marks x\u0301x\u0301 join",
    "cafe\u0301 and re\u0301sume\u0301",
    "\u0967\u0968\u0969 and \u0663\u0664\u0665",
    "mark after punct !\u0301 ok",
    "1\u0301 digit then mark",
    # real-vocab fuzzing regressions (Qwen3.6-27B): any-char letter-run
    # prefix after an exotic-script letter, exotic \p{N} after space,
    # exotic-script letters next to ASCII
    "\u09aa?s",                                  # Bengali PA + '?s'
    "\u8e5f\u09aa?s",                            # CJK + Bengali + '?s'
    "\u0995?s and !s",                           # Bengali KA + '?s'
    " \u0ae8",                                   # space + Gujarati digit 2
    "\t) \u0ae8",                                # tab, paren, space, digit
    "\u0ae7\u0ae8\u0ae9 then 5\u0ae8",           # Gujarati digit runs
    "\u0e2a\u0e27\u0e31\u0e2a\u0e14\u0e35 Thai", # Thai swasdee
    "\u09aa\u09be\u09a8\u09bf water",            # Bengali pani + ASCII
]


# DeepSeek cases: ASCII letters/digits/punct, whitespace (incl tabs/CR/LF)
# and CJK -- the classes the Pascal SplitDeepSeekPieces reproduces EXACTLY.
# (Non-ASCII Latin accents are intentionally avoided: the HF onig engine
# isolates some precomposed Latin-1 letters that the Pascal \p{L} table
# groups, so those would not be byte-parity.)
DEEPSEEK_CASES = [
    "Hello world",
    "Hello, world!",
    "the cat sat on the mat",
    "  leading and  multiple   spaces",
    "trailing space ",
    "tabs\tand\nnewlines",
    "lines\n\n  with blanks\r\nand CRLF\n",
    "Don't stop, DON'T ever STOP!",
    "numbers 12345 and 3.14",
    "100,000 and 9 9 99",
    "a1b2 word123word",
    "punct!!?\n\nthen text",
    "a !!! b  .  c",
    "你好 world 5",
    "word你好world",
    "",
]


# o200k cases: ASCII letters (with case transitions to exercise the two
# case-aware letter alternations), digits, punct (incl '/' for the [\r\n/]*
# trailing run) and whitespace -- the classes SplitO200kPieces reproduces
# EXACTLY. Non-ASCII (accented Latin AND CJK) is intentionally avoided: the
# HF onig engine distinguishes \p{Lu}/\p{Ll}/\p{Lo} per script while the
# Pascal classifier treats every non-ASCII letter as the Ll/Lo class, so byte
# parity is only guaranteed over ASCII letters/digits/punct/whitespace (the
# same approximation stance as the cl100k/DeepSeek splitters).
O200K_CASES = [
    "Hello world",
    "HelloWorld and ABCdef",
    "iPhone XMLParser HTTPSConnection",
    "the cat SAT on the MAT",
    "Don't stop, DON'T ever STOP! it'LL be",
    "numbers 12345 and 3.14 and 1000000",
    "path/to/file and a/b/c/d",
    "  leading and  multiple   spaces",
    "trailing space ",
    "tabs\tand\nnewlines",
    "lines\n\n  with blanks\r\nand CRLF\n",
    "punct!!?\n\nthen text",
    "slash//then/text and ///",
    "a !!! b  .  c",
    "MixedCASE123and456Words",
    "ALLCAPS then lower then Title Case",
    "",
]


def build_o200k_split():
    name = 'tiny_bpe_split_o200k_tokenizer.json'
    path = os.path.join(FIXDIR, name)
    if not os.path.exists(path):
        tok = Tokenizer(models.BPE())
        tok.pre_tokenizer = pre_tokenizers.Sequence([
            pre_tokenizers.Split(Regex(O200K_PATTERN), behavior="isolated",
                                 invert=False),
            pre_tokenizers.ByteLevel(add_prefix_space=False,
                                     use_regex=False),
        ])
        tok.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(
            vocab_size=400, special_tokens=["<|endoftext|>", "<|im_end|>"],
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
        tok.train_from_iterator(CORPUS + ["你好世界", "中文测试"], trainer)
        tok.save(path)
    tok = Tokenizer.from_file(path)  # verify (re)load
    return path, dump_cases(tok, O200K_CASES, ["x<|im_end|>y"])


def build_deepseek_split():
    path = os.path.join(FIXDIR, 'tiny_bpe_split_deepseek_tokenizer.json')
    if not os.path.exists(path):
        # Build via raw json so the exact DeepSeek pre_tokenizer Sequence is
        # used (tokenizers' python Split() does not round-trip the huge
        # class strings as conveniently as feeding the json directly).
        tok = Tokenizer(models.BPE())
        tok.pre_tokenizer = pre_tokenizers.Sequence([
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
        ])  # placeholder; overwritten below
        tok.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(
            vocab_size=400, special_tokens=["<|endoftext|>", "<|im_end|>"],
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
        tok.train_from_iterator(CORPUS + ["你好世界", "中文测试"], trainer)
        data = json.loads(tok.to_str())
        data['pre_tokenizer'] = DEEPSEEK_PRE_TOKENIZER
        with open(path, 'w') as f:
            json.dump(data, f, ensure_ascii=False)
    tok = Tokenizer.from_file(path)  # verify (re)load with real pre-tok
    return path, dump_cases(tok, DEEPSEEK_CASES, ["x<|im_end|>y"])


# Standalone ByteLevel(use_regex=False): NO Split upstream, so the GPT-2
# regex MUST be skipped and the whole segment fed straight to the byte
# alphabet. add_prefix_space=True exercises the prefix-space path too.
BYTELEVEL_NOREGEX_CASES = [
    "Hello world",
    "Hello, world!",
    "  leading spaces",
    "trailing ",
    "tabs\tand\nnewlines",
    "Don't stop!",
    "numbers 12345 and 3.14",
    "café 42!",
    "你好 world",
    "",
]


def build_bytelevel_noregex():
    path = os.path.join(FIXDIR,
                        'tiny_bpe_bytelevel_noregex_tokenizer.json')
    if not os.path.exists(path):
        tok = Tokenizer(models.BPE())
        tok.pre_tokenizer = pre_tokenizers.ByteLevel(
            add_prefix_space=True, use_regex=False)
        tok.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(
            vocab_size=400, special_tokens=["<|endoftext|>"],
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
        tok.train_from_iterator(CORPUS, trainer)
        tok.save(path)
    tok = Tokenizer.from_file(path)  # verify (re)load
    return path, dump_cases(tok, BYTELEVEL_NOREGEX_CASES,
                            ["x<|endoftext|>y"])


# StarCoder2/SantaCoder style: Sequence[Digits(individual_digits=True),
# ByteLevel(use_regex=True)] -- every numeric codepoint isolated on the RAW
# text BEFORE the GPT-2 regex split. Numerals are restricted to the ranges
# the Pascal IsNumberCP table covers (ASCII, ²³¹¼½¾, Arabic-Indic,
# Devanagari, fullwidth); rarer \p{N} scripts would not be byte-parity
# (same approximation stance as the cl100k/DeepSeek splitters).
DIGITS_CASES = [
    "Hello world",
    "abc 123456 def",
    "price: 12,345.67",
    "a  12",
    " 999",
    "1234567",
    "x²y and 2³=8",
    "٣٤٥ and १२३",
    "v1.2.3-rc4",
    "  leading and  multiple   spaces",
    "trailing 12   ",
    "tabs\there\nnewline 42",
    "Don't stop at 3.14!",
    "a1b22c333d4444e",
    "",
]


def build_digits_bytelevel():
    path = os.path.join(FIXDIR, 'tiny_bpe_digits_bytelevel_tokenizer.json')
    if not os.path.exists(path):
        tok = Tokenizer(models.BPE())
        tok.pre_tokenizer = pre_tokenizers.Sequence([
            pre_tokenizers.Digits(individual_digits=True),
            pre_tokenizers.ByteLevel(add_prefix_space=False,
                                     use_regex=True),
        ])
        tok.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(
            vocab_size=400, special_tokens=["<|endoftext|>"],
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
        tok.train_from_iterator(CORPUS, trainer)
        tok.save(path)
    tok = Tokenizer.from_file(path)  # verify (re)load
    return path, dump_cases(tok, DIGITS_CASES, ["x<|endoftext|>y"])


# The Falcon-family 4-stage Sequence (falcon-7b/40b/rw and falcon-mamba):
#   Punctuation(Contiguous) -> ByteLevel(use_regex=True) ->
#   Digits(individual_digits=False) -> Split("[0-9][0-9][0-9]", Isolated).
# The digit stages run on the byte-MAPPED text, so self-mapped latin-1
# bytes that render as numeric superscripts/fractions (B2/B3/B9/BC/BD/BE)
# split too -- 'м' (D0 BC) -> 'Ð' + '¼' is real HF behavior, pinned below.
FALCON_PRE_TOKENIZER = {
    "type": "Sequence",
    "pretokenizers": [
        {"type": "Punctuation", "behavior": "Contiguous"},
        {"type": "ByteLevel", "add_prefix_space": False,
         "trim_offsets": True, "use_regex": True},
        {"type": "Digits", "individual_digits": False},
        {"type": "Split", "pattern": {"Regex": "[0-9][0-9][0-9]"},
         "behavior": "Isolated", "invert": False},
    ],
}

FALCON_CASES = [
    "Hello world",
    "hello, world!! ...ok",
    "abc 123456 def",
    "price: 12,345.67",
    "a  12",
    " 999",
    "1234567",
    "12½3456",
    "½6789",
    "м and мир 999",
    "x²y",
    "٣٤٥",
    "Don't stop at 3.14159265!",
    "(555) 123-4567 ext. 89",
    "  leading and  multiple   spaces",
    "trailing 12   ",
    "tabs\there\nnewline 42",
    "",
]


def build_falcon_seq():
    path = os.path.join(FIXDIR, 'tiny_bpe_falcon_seq_tokenizer.json')
    if not os.path.exists(path):
        # Same raw-json route as the DeepSeek fixture: train with a
        # placeholder, then install the exact 4-stage Sequence.
        tok = Tokenizer(models.BPE())
        tok.pre_tokenizer = pre_tokenizers.Sequence([
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
        ])  # placeholder; overwritten below
        tok.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(
            vocab_size=400, special_tokens=["<|endoftext|>"],
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
        tok.train_from_iterator(CORPUS, trainer)
        data = json.loads(tok.to_str())
        data['pre_tokenizer'] = FALCON_PRE_TOKENIZER
        with open(path, 'w') as f:
            json.dump(data, f, ensure_ascii=False)
    tok = Tokenizer.from_file(path)  # verify (re)load with real pre-tok
    return path, dump_cases(tok, FALCON_CASES, ["x<|endoftext|>y"])


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


# Qwen3.5 ASCII-equivalence fixture: the EXACT qwen2 fixture model (same
# vocab/merges) with only the Split pattern swapped to QWEN35_PATTERN. On
# the pinned ASCII cases the two patterns must split identically, so the
# expected ids equal the split_qwen2 ids -- asserted here at generation
# time; the Pascal test then pins OUR splitter to the same ids.
def build_qwen35_ascii(qwen2_path):
    path = os.path.join(FIXDIR, 'tiny_bpe_split_qwen35_ascii_tokenizer.json')
    with open(qwen2_path) as f:
        data = json.load(f)
    data['pre_tokenizer']['pretokenizers'][0]['pattern']['Regex'] = \
        QWEN35_PATTERN
    with open(path, 'w') as f:
        json.dump(data, f, ensure_ascii=False)
    tok35 = Tokenizer.from_file(path)
    tok2 = Tokenizer.from_file(qwen2_path)
    for text in CASES:
        assert tok35.encode(text, add_special_tokens=False).ids == \
            tok2.encode(text, add_special_tokens=False).ids, \
            'qwen35 pattern diverges from qwen2 on ASCII: %r' % text
    return path, dump_cases(tok35, CASES, ["x<|im_end|>y"])


def main():
    q_path, q_cases = build_split('tiny_bpe_split_qwen2_tokenizer.json',
                                  QWEN2_PATTERN)
    c_path, c_cases = build_split('tiny_bpe_split_cl100k_tokenizer.json',
                                  CL100K_PATTERN)
    q35_path, q35_cases = build_split(
        'tiny_bpe_split_qwen35_tokenizer.json', QWEN35_PATTERN,
        extra_corpus=QWEN35_MARK_CORPUS, cases=QWEN35_CASES)
    qa_path, qa_cases = build_qwen35_ascii(q_path)
    o_path, o_cases = build_o200k_split()
    d_path, d_cases = build_deepseek_split()
    b_path, b_cases = build_bytelevel_noregex()
    m_path, m_cases = build_metaspace_pretok()
    g_path, g_cases = build_digits_bytelevel()
    f_path, f_cases = build_falcon_seq()
    cases_path = os.path.join(FIXDIR, 'hf_tokenizer_cases.json')
    with open(cases_path) as f:
        out = json.load(f)
    out["split_qwen2"] = {"tokenizer": os.path.basename(q_path),
                          "cases": q_cases}
    out["split_cl100k"] = {"tokenizer": os.path.basename(c_path),
                           "cases": c_cases}
    out["split_qwen35"] = {"tokenizer": os.path.basename(q35_path),
                           "cases": q35_cases}
    out["split_qwen35_ascii"] = {"tokenizer": os.path.basename(qa_path),
                                 "cases": qa_cases}
    out["split_o200k"] = {"tokenizer": os.path.basename(o_path),
                          "cases": o_cases}
    out["split_deepseek"] = {"tokenizer": os.path.basename(d_path),
                             "cases": d_cases}
    out["bytelevel_noregex"] = {"tokenizer": os.path.basename(b_path),
                                "cases": b_cases}
    out["metaspace_pretok"] = {"tokenizer": os.path.basename(m_path),
                               "cases": m_cases}
    out["digits_bytelevel"] = {"tokenizer": os.path.basename(g_path),
                               "cases": g_cases}
    out["falcon_seq"] = {"tokenizer": os.path.basename(f_path),
                         "cases": f_cases}
    with open(cases_path, 'w') as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    for p in (q_path, c_path, q35_path, qa_path, o_path, d_path, b_path,
              m_path, g_path, f_path, cases_path):
        print(p, os.path.getsize(p), 'bytes')


if __name__ == '__main__':
    main()
