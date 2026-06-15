# Extractive Question Answering (SQuAD span head)

The extractive-QA path through `neuralpretrained.pas`: given a `question` and a
`context` passage, find the **span of the context** that answers the question.
This is the SQuAD task and the sibling of the sequence-classification importers
— the same BERT-family encoder backbone, a different head.

## What it shows

A `*ForQuestionAnswering` checkpoint is the stock BERT encoder plus a single
`[hidden → 2]` `qa_outputs` dense that produces, **per token**, a start logit and
an end logit. The predicted answer is the span that maximizes
`start[s] + end[e]` over `s ≤ e ≤ s + maxlen`, restricted to the context tokens.

New `neuralpretrained.pas` helpers exercised here:

- **`BuildBertForQuestionAnsweringFromSafeTensors[Ex]`** — builds the BERT-family
  encoder (bert / distilbert / roberta), appends `TNNet.AddQuestionAnsweringHead`
  (two per-token `TNNetPointwiseConvLinear(1)` projections concatenated to
  `(SeqLen, 1, 2)`: channel 0 = start, channel 1 = end), and loads HF's
  `qa_outputs.weight` `[2, hidden]` row 0 onto the **start** projection and row 1
  onto the **end** projection (likewise `qa_outputs.bias`).
- **`AnswerSpan(question, context, …)`** — runs `[CLS] question [SEP] context [SEP]`,
  masks the question / special / pad positions, picks the best valid span over
  the **context** tokens only, and recovers the answer substring through the
  tokenizer offset map (`EncodeWithOffsets`). Returns the SQuAD2 null-answer
  baseline `start[CLS] + end[CLS]` so callers can threshold abstention.
- **`QAReport`** — SQuAD **Exact-Match + macro token-F1** over a planted
  `{question, context, gold}` set, with the official answer normalization
  (lowercase, drop `a`/`an`/`the`, strip punctuation, collapse whitespace).
  Mirrors `STSReport` / `RetrievalReport`. The `NormalizeSquadAnswer` /
  `SquadTokenF1` primitives are exposed too.

## Running

```
# from the repo root, so the default tokenizer fixture path resolves:
ExtractiveQA [tokenizer.json]
```

The released QA checkpoints (`distilbert-base-cased-distilled-squad`,
`deepset/roberta-base-squad2`) are ~250 MB, too large to ship. To stay
self-contained and CPU-fast (<1 s, no download) this demo **hand-wires** a span
head over the committed WordPiece tokenizer fixture: an embedding whose row value
is each token's "score", read directly as the start/end logit, so the answer is
the highest-scoring context word. This exercises the whole public pipeline
(tokenization with offsets, context-only masking, span argmax, substring
recovery, EM/F1 scoring) without any pretrained weights.

For a **real** model, replace the demo-net block with:

```pascal
NN := BuildBertForQuestionAnsweringFromSafeTensors(
        'model.safetensors', {pSeqLen=}384, {pInferenceOnly=}true);
Tok.LoadFromFile('tokenizer.json');
```

`AnswerSpan` and `QAReport` work unchanged.

## Parity

The importer's span logits are pinned against an HF `transformers` float64 oracle
on a synthesized pico `DistilBertForQuestionAnswering`
(`tools/distilbert_qa_tiny_fixture.py`, `TestDistilBertQALogitParity`) to
**< 1e-4**. The fixture boosts the `qa_outputs` head so a start/end row swap in
the importer would fail the parity gate loudly.

Coded by Claude (AI).
