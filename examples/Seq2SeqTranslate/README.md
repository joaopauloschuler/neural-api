# Seq2SeqTranslate — offline beam-decode + BLEU/ROUGE plumbing

This example is a **self-contained, offline** demonstration of the landed
sequence-to-sequence decode-and-evaluate pipeline:

```
encoder/decoder import  ->  DecodeSeq2SeqBeamSearch  ->  BLEU / ROUGE
```

It needs **no network access and no multi-GB checkpoint**: it imports the
committed pico **Marian** fixture (`tests/fixtures/tiny_marian.*`, 13-token
vocab, 2 layers, `d_model 8` — the same fixture the importer parity test
uses) via `BuildMarianFromSafeTensors` (neural/neuralpretrained.pas), runs
token-id beam search (`DecodeSeq2SeqBeamSearch`, neural/neuraldecode.pas)
over a fixed source-id sequence, and scores the decoded ids against a
reference id sequence with corpus **BLEU** + **ROUGE-1/2/L**
(neural/neuralnlpmetrics.pas).

## This is a PLUMBING demo, not a quality demo

The pico fixture is **randomly initialized**, so the decoded "translation"
is gibberish and the decoded-vs-reference BLEU/ROUGE numbers are meaningless
as a measure of quality (they come out near 0). What this example proves
end-to-end is that:

1. a real two-net encoder-decoder **imports and beam-decodes offline**;
2. the decoded token ids **flow into the BLEU/ROUGE metrics** (the metrics
   have token-id overloads, so no detokenizer is needed here); and
3. the pipeline is **deterministic** (re-running beam search yields the
   exact same ids) and the metrics are **well-formed** (a reference scored
   against *itself* gives BLEU = ROUGE-1 F1 = 1.0).

Points 2–3 are checked with assertions; the program halts non-zero if either
fails.

## Translating real text

To translate real text, point the same `DecodeSeq2SeqBeamSearch` call at a
real `Helsinki-NLP/opus-mt-*` checkpoint (`model.safetensors` +
`config.json` + `tokenizer.json`) and feed it ids from that checkpoint's
tokenizer — the Marian importer and beam search are unchanged; only the
tokenizer (a SentencePiece/Unigram `tokenizer.json` read by
`TNeuralHFTokenizer`) and the detokenize step differ. See
[`examples/Summarize`](../Summarize/README.md) for the BART **text** path
over a downloaded checkpoint; this example deliberately stays offline so it
runs in CI on the committed fixture.

## Build & run

From the repository root (cap memory; the pico run fits comfortably in 3GB):

```bash
LAZUTILS_PATH=$(find /usr/lib/lazarus /usr/share/lazarus -name utf8process.ppu -printf '%h\n' | head -1)
ulimit -v 3000000
fpc -O2 -Mobjfpc -Sh -Funeural -Fu"$LAZUTILS_PATH" \
    -FUexamples/Seq2SeqTranslate examples/Seq2SeqTranslate/Seq2SeqTranslate.lpr
./examples/Seq2SeqTranslate/Seq2SeqTranslate
```

(Run from the repo root so the default `tests/fixtures/tiny_marian.*` paths
resolve; or pass `<fixture.safetensors> <config.json>` as arguments.) Or
open `Seq2SeqTranslate.lpi` in Lazarus.

Expected output (the gibberish ids are deterministic for the committed
fixture):

```
Source ids: [1 2 3 4 5 6 7 8 9 1]
Decoded ids (beam search): [6 6 6 6 6]
OK: beam decode is deterministic (identical re-run).
...
BLEU      : 0.0000
ROUGE-1 F1: 0.0000
...
Sanity (reference vs itself):
  BLEU      : 1.0000
  ROUGE-1 F1: 1.0000
OK: self-reference BLEU = ROUGE-1 F1 = 1.0.
```
