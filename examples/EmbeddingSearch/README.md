# EmbeddingSearch: E5/BGE pooling-mode + instruction-prefix retrieval

The self-contained sibling of [SemanticSearch](../SemanticSearch). Where
SemanticSearch needs a ~90 MB MiniLM download and only covers the
mean-pool / no-prefix MiniLM case, this demo exercises the two pieces that
distinguish the published **E5 / BGE / GTE** retrievers — the
**pooling-mode selector** and the **instruction-prefix table** — entirely
on the committed pico fixture `tests/fixtures/tiny_e5.*`, so it runs in
under a second on CPU with no network.

## What it shows (beyond the MiniLM demo)

1. **Instruction prefixes** (`neuralpretrained.pas`, `EmbedInstructionPrefix`
   / `ApplyEmbedInstruction`): the family-specific strings that are
   MANDATORY for parity — E5's `"query: "`/`"passage: "`, BGE's
   `"Represent this sentence for searching relevant passages: "` query
   instruction, the gte-Qwen2 `"Instruct: …\nQuery: "` template. They
   change the vector because they change the token ids.
2. **Pooling-mode selector** (`PoolSentenceEmbedding` + `TNNetEmbedPooling`):
   wraps `{forward → pool(mode) → optional L2-normalize}` for any of
   `epCLS` (BGE), `epMean` (E5/GTE), `epLastToken` (e5-mistral, EOS/left-pad
   convention) — generalizing the mean-only `BertPoolSentenceEmbedding`.

The pico fixture ships no `tokenizer.json`, so the instruction prefix is
**baked into the leading token ids** of each sequence (ids 1/2/3 stand in
for the prefix tokens) — exactly the sequences the HF float64 parity oracle
was computed from (`tools/e5_embed_tiny_fixture.py`,
`tiny_e5_embed.json`, `TestE5EmbeddingParity`). With a real downloaded E5
checkpoint you would instead tokenize
`ApplyEmbedInstruction(efE5, IsQuery, Text)` (see `SemanticSearch` for the
`tokenizer.json` path).

## Parity

The mean-pooled + L2-normalized query and passage vectors match the HF
`transformers` float64 oracle within **1e-4** — pinned by
`TestE5EmbeddingParity` in `tests/TestNeuralPretrained.pas`. (No
`sentence-transformers` install is needed: for E5/BGE its
`SentenceTransformer` is exactly `AutoModel forward → mean/CLS pool → L2
normalize` in float64, reproduced directly by the fixture maker.)

## Run

```bash
fpc -Fuexamples/EmbeddingSearch -Funeural -Mobjfpc -Sh -O2 \
    examples/EmbeddingSearch/EmbeddingSearch.lpr
examples/EmbeddingSearch/EmbeddingSearch        # uses tests/fixtures
```

Output:

```
Instruction-prefix table (neuralpretrained.pas):
  E5  query   : "query: "
  E5  passage : "passage: "
  BGE query   : "Represent this sentence for searching relevant passages: "
  efE5 applied: "query: how tall is mount everest"

Imported pico E5 encoder: 2 layers, hidden 8. Pooling = mean + L2-normalize (E5 recipe).
...
Ranking passages against the query (cosine = dot, unit vectors):
  1.  0.9958  passage0 (baked "passage: " prefix, shares the query body)
  2.  0.9922  passage1 (baked "passage: " prefix, unrelated body)
OK: the shared-body passage ranks first.
```

To regenerate the fixture (needs `torch` + `transformers` + `safetensors`):

```bash
python3 tools/e5_embed_tiny_fixture.py
```
