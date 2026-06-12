# SemanticSearch: sentence embeddings + semantic search on an imported MiniLM

Embeds a small text corpus with an imported
[sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
checkpoint and ranks it by cosine similarity against a query. The paraphrase
("A kitty rests on the window ledge enjoying the sunshine." vs "A cat is
sitting on the windowsill in the sun.") wins by a wide margin over the
distractors.

## Pipeline

Per sentence (helpers in `neural/neuralpretrained.pas`, SENTENCE EMBEDDINGS
section):

1. `tokenizer.json` WordPiece encode (`neuralhftokenizer.pas`), wrapped in
   `[CLS] ... [SEP]` by `BertTokenizeSentence`;
2. the `BuildBertFromSafeTensors` encoder produces `(SeqLen, 1, hidden)`
   final hidden states;
3. `BertPoolSentenceEmbedding` mean-pools over the REAL tokens only and
   L2-normalizes -- exactly the sentence-transformers
   "mean pooling + normalize" head. Cosine similarity is then a plain dot
   product.

The imported encoder carries NO attention padding mask, so the program never
pads: it builds one inference-only net per distinct token length in the
corpus (cached). That keeps the Pascal embeddings exactly equal to
sentence-transformers' -- cosine > 0.999 per sentence, see the parity check
below.

## Run

Download the checkpoint (three files, ~90 MB):

```bash
mkdir minilm && cd minilm
curl -LO https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/model.safetensors
curl -LO https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/config.json
curl -LO https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/tokenizer.json
```

Build (`lazbuild examples/SemanticSearch/SemanticSearch.lpi` or the fpc line
below) and run:

```bash
fpc -Fuexamples/SemanticSearch -Funeural -Mobjfpc -Sh -O2 examples/SemanticSearch/SemanticSearch.lpr
examples/SemanticSearch/SemanticSearch minilm
```

Output (abbreviated):

```
Query: A kitty rests on the window ledge enjoying the sunshine.
Ranking (cosine similarity):
   1.  0.7240  A cat is sitting on the windowsill in the sun.
   2.  0.3317  A dog is sleeping on the porch.
   3.  0.1359  The orchestra performed a beautiful symphony last night.
   ...
   8. -0.0583  He fixed the leaking kitchen faucet himself.
```

`-q "your query"` ranks the corpus against your own query.

## Parity with sentence-transformers

```bash
examples/SemanticSearch/SemanticSearch minilm -dump pas_embeddings.json
python3 examples/SemanticSearch/compare_st_embeddings.py minilm pas_embeddings.json
```

`compare_st_embeddings.py` compares against `sentence_transformers`
`model.encode()` when installed, else against plain `transformers`
mean-pooled hidden states (the same computation for this model). Verified
result on all-MiniLM-L6-v2: cosine = 0.9999999 for every corpus + query
sentence (bar: > 0.999).

Any HF `BertModel` checkpoint with a WordPiece `tokenizer.json` works, not
just MiniLM -- though only sentence-transformers models are TRAINED so that
mean-pooled cosine similarity means semantic similarity.
