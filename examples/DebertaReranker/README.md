# DebertaReranker — RAG cross-encoder reranking with a DeBERTa-v3 checkpoint

The canonical **second-stage reranker** of a RAG pipeline. A fast bi-encoder /
BM25 retriever returns the top-N candidate passages; this example re-scores each
candidate *jointly* against the query with a pretrained **DeBERTa-v3
`ForSequenceClassification`** checkpoint (the ms-marco family —
`cross-encoder/ms-marco-...`, `naver/trecdl...`).

A **cross-encoder** concatenates the query and one candidate passage into a
single sequence

```
[CLS] query [SEP] passage [SEP]
```

and the classification head emits a single relevance **logit** — higher means
more relevant. This is far more accurate than cosine similarity of independent
embeddings, because every query token can attend to every passage token.

## The model

The encoder is imported with **`BuildDebertaV2FromSafeTensorsEx`** passing
`pSeqClsHead=true`, so the network ends in the sequence-classification head and
the **row-0 (`[CLS]`) logits are the relevance scores**. The returned config is
printed with `DebertaV2ConfigToString`.

DeBERTa-v3's **disentangled attention** (`TNNetDisentangledAttention`) is the
encoder backbone; see the DEBERTA-V2 IMPORT section of `neuralpretrained.pas`.

Scoring loop per candidate:

```
assemble [CLS] q [SEP] p [SEP] ids (pad id 0, truncate/pad to SeqLen)
  -> Input volume (SeqLen, 1, 2)   ch0 = token ids, ch1 = token-type 0
  -> NN.Compute
  -> NN.GetOutput
  -> score = Output.FData[0]        row 0 ([CLS]), label 0
```

(type_vocab_size is 0 in v3, so the token-type channel is all zeros.) Candidates
are then ranked descending by score.

## Inputs and tokenization

Two ways to supply candidates:

- **Text mode** (`-q "query" -p "passage" [-p ...]`): needs `tokenizer.json`
  beside the checkpoint. DeBERTa-v3 ships a **Unigram** `tokenizer.json` that the
  `TNeuralHFTokenizer` Unigram reader handles. The query and each passage are
  encoded and assembled into `[CLS] q [SEP] p [SEP]`.
- **Raw-id mode** (`-ids <id list> [-ids ...]`): for the committed pico fixture
  (which has no tokenizer). Each `-ids` list is **one already-assembled** id
  sequence; repeat `-ids` for several passages.

Other flags: `SeqLen` (optional leading number, default 64), `-cfg <path>`
(config.json path; `''` = next to the checkpoint), `-cls N` (`[CLS]` id,
default 1), `-sep N` (`[SEP]` id, default 2).

## How to run

```
cd examples/DebertaReranker
fpc -O3 -Mobjfpc -Sh -Fu../../neural DebertaReranker.lpr
./DebertaReranker
```

(or open `DebertaReranker.lpi` in Lazarus).

With no arguments the program prints its usage and exits. Try the tiny committed
seq-cls fixture (`num_labels` 2; the demo uses `logit[0]` as the score),
assembling two raw id sequences:

```
./DebertaReranker tests/fixtures/tiny_debertav2_seqcls.safetensors 16 \
  -cfg tests/fixtures/tiny_debertav2_seqcls_config.json \
  -ids 1 5 2 7 9 2 0 0 0 0 0 0 0 0 0 0 \
  -ids 1 5 2 3 4 2 0 0 0 0 0 0 0 0 0 0
```

Or with a real ms-marco DeBERTa-v3 reranker (`tokenizer.json` beside it):

```
./DebertaReranker /tmp/ms-marco/model.safetensors -q "what is rag?" \
  -p "Retrieval augmented generation..." -p "An unrelated sentence."
```

## Expected output

The program announces the load, prints the parsed config, the number of output
labels (`Labels (output depth): ...`), the tokenizer path if one was loaded, the
query and how many candidates it is scoring, and finally the ranked list:

```
Reranked passages (most relevant first):
  #1  score=...  "..."
  #2  score=...  "..."
```

Each line shows the `[CLS]` relevance logit and either the passage text (text
mode) or `(passage N)` (raw-id mode), ordered from most to least relevant. Exact
scores depend on the checkpoint; the *ordering* is the point.

Coded by Claude (AI).
