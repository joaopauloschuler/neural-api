# RAG: end-to-end retrieval-augmented generation

Ties the already-landed NLP pieces into the canonical **RAG** application with
**no new core library code**:

1. **Chunk** a small built-in knowledge base (8 short passages).
2. **Embed** each chunk and the question with the
   [SemanticSearch](../SemanticSearch) sentence-embedding path
   (`BertTokenizeSentence` → `BuildBertFromSafeTensors` encoder →
   `BertPoolSentenceEmbedding`: mean-pool over the real tokens + L2
   normalize). One inference-only encoder is cached per distinct token
   length (the encoder carries no attention pad mask, so never padding keeps
   embeddings at sentence-transformers parity).
3. **Retrieve** the top-k chunks by cosine similarity — a plain dot product,
   since the embeddings are unit vectors.
4. **Splice** the retrieved chunks into the canonical RAG prompt template:

   ```
   Context:
   - {chunk 1}
   - {chunk 2}

   Question: {q}
   Answer:
   ```

5. **Generate** a grounded answer with an imported decoder through the
   [ChatTerminal](../ChatTerminal) chat-template + streaming-decode infra
   (`BuildFromPretrained` inference-only → `EncodeChat` /
   `ApplyChatTemplate` → full-recompute greedy decode, streamed token by
   token).

## The headline RAG property

The default question asks for the **launch date of "Project Halcyon"** — an
**invented fact that no pretrained model can know**. Corpus chunk #5 states
it (`...scheduled to launch on the 14th of November, 2027.`). Without
retrieval the bare model can only hallucinate a date; **with** retrieval the
relevant chunk is placed in the context and a grounded decoder reads the
answer off the supplied passage. That gap — wrong without retrieval, right
with it — is the point of RAG.

## Run (no download, retrieval + prompt-splice half)

Both models are **optional command-line paths**, so the example builds and
the retrieval/splice half runs with **no network access**:

```bash
cd examples/RAG
fpc -B -Fu../../neural -Mobjfpc -Sh -O2 RAG.lpr
./RAG                 # built-in fallback embedder, prints the spliced prompt
./RAG --selftest      # 16 offline unit checks, no model files
```

With no `--embed-model`, a deterministic hashing-bag-of-words embedder stands
in for the real sentence encoder (lower quality, but it ranks the Halcyon
chunk first for the Halcyon question and wires the pipeline end to end). With
no `--gen-model`, the program stops after assembling and printing the spliced
prompt.

## Run (full pipeline with real models)

Download a small sentence encoder and a small instruct decoder, then:

```bash
# sentence encoder (~90 MB) — see examples/SemanticSearch/README.md
mkdir minilm && cd minilm
curl -LO https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/model.safetensors
curl -LO https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/config.json
curl -LO https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/tokenizer.json
cd ..

# any instruct decoder ChatTerminal understands (config.json +
# model.safetensors [+ index] + tokenizer.json [+ tokenizer_config.json]);
# a small one (e.g. a ~0.5B Qwen2/3-Instruct) keeps it CPU/RAM-bounded.

./RAG --embed-model ./minilm --gen-model /path/to/decoder --int8
```

The decoder is always built `pTrainable=false` and `--int8` (weight-only int8)
is the default — pass `--fp32` for full precision (much more RAM). Keep
`--ctx` modest (default 1024): the full-recompute decode allocates a
`SeqLen×SeqLen` attention buffer per head per layer, so build memory grows
~O(ctx²).

## Options

```
--embed-model DIR    HF sentence encoder dir (omit for the built-in fallback)
--gen-model DIR      HF decoder LM dir (omit to stop after the spliced prompt)
--question "text"    question to answer (default: the Project Halcyon probe)
--top-k N            chunks to retrieve and splice (default 2)
--max-new-tokens N   answer length cap (default 48)
--ctx N              decoder context window (default 1024)
--fp32               full-precision decoder weights (default int8)
--format NAME        chat-format override (chatml|llama3|gemma|...)
--selftest           offline unit checks, then exit (no model)
--help               usage
```

All the heavy lifting reuses existing helpers — `neuralpretrained.pas`
(BERT encoder + `BuildFromPretrained` decoder dispatch + sentence-embedding
pooling), `neuralhftokenizer.pas`, `neuralchat.pas`
(`EncodeChat`/`ApplyChatTemplate`) and `neuraldecode.pas`. This example adds
no core code; it is the wiring that turns those pieces into RAG.

Coded by Claude (AI).
