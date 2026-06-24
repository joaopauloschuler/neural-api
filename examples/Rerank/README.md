# Rerank: two-stage retrieve-then-rerank RAG pipeline on CPU

The single biggest quality lever in retrieval-augmented generation is the
**retrieve-then-rerank** pipeline: a fast first stage recalls candidates, then a
slow-but-accurate **cross-encoder** re-scores them for precision. This demo wires
both stages with library helpers:

1. **Stage 1 — bi-encoder recall.** The query and every corpus passage are
   embedded *independently* (`BertEncodeSentence`, the
   [SemanticSearch](../SemanticSearch) recipe) and ranked by cosine similarity.
   Cheap: passages are embedded once, the query once.
2. **Stage 2 — cross-encoder rerank.** Each top-K candidate is re-scored
   *jointly* with the query — `[CLS] query [SEP] passage [SEP]`, with
   token_type/segment id 0 on the query span and **1 on the passage span** — by
   a BERT-family `*ForSequenceClassification` with `num_labels=1`. The `[CLS]`
   relevance logit (sigmoid) reorders the candidates. Because every query token
   attends to every passage token, this is far more accurate than the
   bi-encoder's independent dot product.

This completes the RAG retrieval family in the repo next to the **bi-encoder**
([SemanticSearch](../SemanticSearch)), the **DeBERTa cross-encoder**
([DebertaReranker](../DebertaReranker)) and **ColBERT late interaction**
([ColBERTSearch](../ColBERTSearch)). The new piece here is the **sentence-PAIR
encoding** with a live `token_type`/segment-id=1 path and a batch rerank scorer.

## New `neuralpretrained.pas` helpers (CROSS-ENCODER RERANKER section)

* `BertTokenizePair(Tok, A, B, out ids, out segments)` — lays out
  `[CLS] A [SEP] B [SEP]` and the parallel segment ids (0 over `[CLS] A [SEP]`,
  1 over `B [SEP]`), HF's `token_type_ids` convention, with HF
  `longest_first` truncation to a token budget.
* `CrossEncoderScore(Net, Tok, Query, Passage)` — one joint forward; returns the
  `[CLS]` relevance logit (sigmoid by default). Feeds the segment ids into
  channel 1 of the net's `(SeqLen,1,2)` input, which the BERT importer wires into
  the `token_type_embeddings` table.
* `RerankPassages(Net, Tok, Query, Passages, out Order, out Scores)` — scores a
  query against a list of candidates (one forward each, optionally int8 via the
  backbone's `pQuantizeInt8`) and returns them most-relevant first.
* `RerankReport(Net, Tok, Query, Passages, Relevant, KList)` — MRR / nDCG@k
  **before vs after** reranking, quantifying the precision lift over the initial
  (bi-encoder) order.

## Run it

Offline demo on the committed pico fixture (no download):

```
cd examples/Rerank
lazbuild Rerank.lpi   # or: fpc -dRelease -dAVX2 -Fu../../neural Rerank.lpr
./Rerank -demo
```

prints the `RerankReport` lift (a hand-wired tiny cross-encoder that only the
gold candidate's joint sequence scores highest, moving it from last place to
first — MRR 0.3333 → 1.0000, nDCG@1 0.0000 → 1.0000).

Real two-stage pipeline (needs two checkpoint downloads):

```
./Rerank -bi <all-MiniLM-L6-v2 dir> -ce <ms-marco-MiniLM-L-6-v2 dir> \
         -q "your query" -k 5
```

Each dir holds `model.safetensors` + `config.json` + `tokenizer.json`. Stage 1
prints the cosine ranking; stage 2 re-scores the top-K with the cross-encoder and
prints the reordered relevance scores. `-int8` quantizes the reranker backbone.

The pair/segment-id path is pinned against the HF float64
`AutoModelForSequenceClassification` logit to **<1e-4** on a synthesized pico
reranker (`tools/bert_reranker_tiny_fixture.py`, `TestRerankerPairLogitParity`).
Always built `pTrainable=false`; pure CPU.
