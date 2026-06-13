# HubFetch: HuggingFace Hub download -> imported TNNet in one command

Every importer example so far (GPT2Import, LlamaImport, SemanticSearch)
starts with "hand-download these files with python / huggingface-cli". This
example uses the **opt-in** `neural/neuralhfhub.pas` unit to remove that
step:

```pascal
Net := BuildFromPretrained(HubFetchModel('sentence-transformers/all-MiniLM-L6-v2'));
```

`HubFetchModel(repo [, revision [, token]])` downloads, from
`https://huggingface.co/{repo}/resolve/{rev}/{file}`:

* `config.json` (mandatory),
* `tokenizer.json` (when the repo has one — optional, 404 tolerated),
* the safetensors weights: `model.safetensors` when it exists, otherwise
  `model.safetensors.index.json` plus **every shard** its `weight_map`
  references (the sharded reader in `neuralsafetensors.pas` then takes the
  index path directly),

into a local cache — `~/.cache/neural-api/hub/{repo}/{rev}/...` by default,
overridable with `HubSetCacheDir` or the `NEURAL_API_HUB_CACHE` environment
variable — and returns the snapshot **directory**, which is exactly what
`BuildFromPretrained` accepts. Files already present are **never
re-downloaded** (the second run is instant and fully offline). Downloads go
to a `.part` file renamed only on success, so an interrupted download never
poisons the cache. For **gated repos** pass a token or set `HF_TOKEN`; it is
sent as an `Authorization: Bearer` header. Redirects (resolve URLs bounce to
the CDN) and HTTPS are handled by `fphttpclient` + `opensslsockets`.

The point of the unit split: `neuralpretrained.pas` stays strictly
offline — only programs that explicitly `uses neuralhfhub` link
HTTP/OpenSSL.

## Running

```
lazbuild HubFetch.lpi
../../bin/x86_64-linux/bin/HubFetch
```

With no arguments it fetches
`hf-internal-testing/tiny-random-bert-sharded-safetensors` — a ~100KB
**five-shard** checkpoint that exercises the index-json fallback — and
imports it (123 layers). Pass a repo id (and optionally a revision) to fetch
something else, e.g. the ~90MB
`sentence-transformers/all-MiniLM-L6-v2` (22.5M weights, the SemanticSearch
checkpoint, byte-identical to what `huggingface_hub` downloads).

Lower-level API (all in `neuralhfhub.pas`): `HubFetchFile` /
`HubTryFetchFile` (single file, 404-tolerant variant), `HubResolveURL` /
`HubLocalPath` (pure helpers), `HubShardListFromIndexJson` (offline
index parsing), `HubGetCacheDir` / `HubSetCacheDir`.

This example is coded by Claude (AI).
