# ColBERTSearch: ColBERT late-interaction (MaxSim) retrieval on CPU

Pre-encodes a small passage corpus with an imported
[colbert-ir/colbertv2.0](https://huggingface.co/colbert-ir/colbertv2.0)-class
checkpoint and ranks it against a query by the **MaxSim** late-interaction
score. ColBERT is the third RAG retrieval paradigm in this repo, next to the
**bi-encoder** ([examples/SemanticSearch](../SemanticSearch): one pooled vector
+ cosine) and the **cross-encoder**
([examples/DebertaReranker](../DebertaReranker): a joint query+passage scorer).

Unlike either, ColBERT keeps the **per-token** contextual embeddings of query
and document (no pooling), projects each token to a small dim (128) and
L2-normalizes, then scores a `(query, doc)` pair by

```
score = sum_{q in query} max_{d in doc} <E_q, E_d>
```

every query token matched to its single best document token, then summed. This
gives cross-encoder-grade accuracy at bi-encoder cost: documents are encoded
**once** (their per-token matrices cached) and every query is scored by MaxSim
against the cache.

## Pipeline

Helpers in `neural/neuralpretrained.pas`, COLBERT LATE INTERACTION section:

1. `tokenizer.json` WordPiece encode (`neuralhftokenizer.pas`), wrapped per the
   ColBERT marker convention by `ColBERTBuildInput`:
   - documents: `[CLS] [D] tokens... [SEP]` then `[PAD]`-filled (pad rows
     skipped from MaxSim);
   - queries: `[CLS] [Q] tokens... [SEP]` then `[MASK]`-padded to the net's
     SeqLen (query **augmentation** -- those `[MASK]` rows are real inputs and
     DO contribute to MaxSim);
2. `BuildColBERTFromSafeTensors` builds the stock BERT encoder + the ColBERT
   `linear` head (a bias-free `[hidden -> 128]` dense applied per token).
   Pass `pQuantizeInt8=true` to store the WHOLE net -- backbone **and**
   projection head -- as weight-only int8;
3. `ColBERTEmbedTokens` returns the per-token L2-normalized projected matrix
   `(RealTokens, 1, 128)` -- NO pooling;
4. `ColBERTMaxSimScore(query, doc)` ranks the corpus.

The encode-corpus / cache / score-by-MaxSim loop is wrapped by the library
class **`TColBERTIndex`** (also in `neuralpretrained.pas`): construct it with a
built ColBERT net + tokenizer, call `AddCorpus`/`AddDocument` to pre-encode and
cache the per-token doc matrices once, then `Search(query, TopK)` returns the
ranked `TColBERTHit` list (doc index + MaxSim score + text). This example drives
that class.

The imported encoder carries NO attention padding mask (like SemanticSearch):
documents are `[PAD]`-filled to the net's SeqLen and the pad rows are skipped
from MaxSim, but real tokens still attend to pad positions, so a short doc in a
long net is an approximation. Pass `-seqlen` close to the real corpus length
for best fidelity.

## Run

Download a ColBERT checkpoint (the released `colbertv2.0` carries the
`linear.weight` `[128, hidden]` projection head this importer needs):

```bash
mkdir colbert && cd colbert
curl -LO https://huggingface.co/colbert-ir/colbertv2.0/resolve/main/model.safetensors
curl -LO https://huggingface.co/colbert-ir/colbertv2.0/resolve/main/config.json
curl -LO https://huggingface.co/colbert-ir/colbertv2.0/resolve/main/tokenizer.json
cd ..
```

Build and run:

```bash
lazbuild examples/ColBERTSearch/ColBERTSearch.lpi
./bin/x86_64-linux/bin/ColBERTSearch colbert -q "A kitty rests on the window ledge in the sun."
```

The program prints the corpus ranked by MaxSim. Use `-seqlen N` to set the
encoder context length (default 32).

## Parity

The ColBERT forward (encoder + bias-free projection + per-row L2 norm + MaxSim)
is pinned against an HF float64 oracle on a synthesized pico checkpoint by
`tools/colbert_tiny_fixture.py` and `tests/TestNeuralPretrained.pas`
(`TestColBERTParity`): the per-token projected+normalized query/doc matrices and
the MaxSim score match within 1e-4.
