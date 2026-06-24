# LlamaImport: load pretrained Llama-architecture weights into a TNNet

This example verifies the CAI **Llama** safetensors importer
(`BuildLlamaFromSafeTensors` in `neural/neuralpretrained.pas`) against
HuggingFace `transformers`' `LlamaForCausalLM`, logit by logit. It covers
the standard Llama-2/TinyLlama decoder: token embedding (positions come
only from RoPE), N blocks of pre-RMSNorm rotary **GQA** + pre-RMSNorm
**SwiGLU** MLP with residuals, final RMSNorm and the (optionally tied) LM
head. No biases anywhere.

Importer subtleties handled for you (see the `neuralpretrained.pas` unit
header for the full discussion):

* **nn.Linear layout** — HF Llama stores `[out, in]` (the opposite of
  GPT-2's transposed Conv1D), so rows load straight into
  `TNNetPointwiseConvLinear` neurons with no transpose.
* **rotate_half RoPE** — HF rotates pairs `(i, i + head_dim/2)`;
  `TNNetRotaryEmbedding` rotates interleaved pairs `(2k, 2k+1)`. The
  importer permutes the `q_proj`/`k_proj` rows within each head so the two
  conventions become numerically identical.
* **SwiGLU** — `TNNetSwiGLU` computes `A * Swish(B)` on depth halves, so
  the fused projection holds `up_proj` first and `gate_proj` second.
* **config.json** — `hidden_size`, `intermediate_size`,
  `num_hidden_layers`, `num_attention_heads`, `num_key_value_heads`,
  `rms_norm_eps`, `rope_theta`, `vocab_size`, `max_position_embeddings`
  and `tie_word_embeddings` are read from the `config.json` next to the
  checkpoint.
* **Sharded checkpoints** — for repos shipping
  `model-00001-of-000NN.safetensors` shards, pass the path of the
  `model.safetensors.index.json` instead of a `.safetensors` file:
  `TNNetSafeTensorsReader` parses the `weight_map`, opens every shard and
  serves tensors transparently (no importer changes needed).

Tokenizers: `neural/neuralhftokenizer.pas` (`TNeuralHFTokenizer`) reads the
HF `tokenizer.json` that ships with Llama-family repos (BPE with metaspace
normalizer + `<0xNN>` byte fallback), so text can be encoded to ids and
decoded back (`LoadFromFile` / `Encode` / `Decode`; see
`examples/GPT2Import` for end-to-end text-prompt usage). The logits dumper
below still runs on raw token ids. The raw SentencePiece `.model` protobuf
is not parsed -- use the repo's `tokenizer.json`.

## Files

* `LlamaLogitsDump.lpr/.lpi` — imports a checkpoint, runs one forward pass
  on the given token ids and dumps per-position full-vocab logits as JSON.
* `make_tiny_llama.py` — builds a **random** tiny `LlamaForCausalLM`
  (2 layers, hidden 64, 4 heads, 2 kv heads, vocab 256; `--tie` for tied
  embeddings) with `transformers` — no download needed.
* `slice_llama.py` — alternatively, slices a **real** checkpoint
  (e.g. TinyLlama-1.1B) down to its first N blocks and a vocab prefix so
  RAM-limited boxes can check genuine weights.
* `make_pico_llama_fixture.py <src_dir> <out_prefix>` — slices a real
  checkpoint down EVERY dimension (layers, hidden, heads/kv-heads,
  head_dim, MLP, vocab; GQA group structure preserved) into a ~10 KB
  committable parity fixture of genuine weights plus float64 reference
  logits from `LlamaForCausalLM` (used to pin
  `tests/fixtures/tiny_smollm2.*`).
* `compare_hf_logits.py` — loads the same checkpoint into
  `LlamaForCausalLM` and diffs every logit against the Pascal dump
  (gate 1e-3).

## Workflow

```bash
examples/LlamaImport$ lazbuild --build-mode=Release LlamaLogitsDump.lpi

# random tiny model (needs torch + transformers, no download):
$ python3 examples/LlamaImport/make_tiny_llama.py /tmp/tiny_llama
$ bin/x86_64-linux/bin/LlamaLogitsDump /tmp/tiny_llama/model.safetensors \
    16 1 5 99 250 7 42 > /tmp/cai.json
$ python3 examples/LlamaImport/compare_hf_logits.py \
    /tmp/tiny_llama/model.safetensors /tmp/cai.json
```

Measured (f32 end-to-end, 2026-06): random tiny model max |logit diff|
**1.8e-7**, tied variant **2.1e-7**, 1-layer/128-vocab slice **1.5e-7**;
argmax agrees at every position.

**Verified import targets** (logit parity vs HF transformers on real
weights): `TinyLlama/TinyLlama-1.1B` and `HuggingFaceTB/SmolLM2-135M` —
both load with `BuildLlamaFromSafeTensors` unchanged. SmolLM2-135M (tied
embeddings — no `lm_head` tensor in the checkpoint — BF16, rope_theta
100000, GQA 9 query / 3 kv heads) measured 2026-06: 2-layer/4096-vocab
slice max |logit diff| **6.5e-5** over 53,248 logits, argmax agrees at
every position. A committed pico fixture of REAL SmolLM2 weights
(`tests/fixtures/tiny_smollm2.*`, ~4 KB BF16, dimension-sliced by
`make_pico_llama_fixture.py` — 2 of 30 layers, 2 q-heads sharing 1 kv head
× 4 dims, hidden 8, vocab 12, reference logits from `LlamaForCausalLM` in
float64) is asserted by `TestSmolLM2LogitParity` in every test run; the
tied-head path and the BF16 decode are thereby covered end-to-end on
genuine weights.

For a real checkpoint, download e.g.
`TinyLlama/TinyLlama-1.1B-Chat-v1.0` (`model.safetensors` +
`config.json`), then:

```bash
$ python3 examples/LlamaImport/slice_llama.py /path/to/TinyLlama /tmp/tl_slice 2 8192
$ bin/x86_64-linux/bin/LlamaLogitsDump /tmp/tl_slice/model.safetensors \
    16 1 15043 3186 > /tmp/cai.json
$ python3 examples/LlamaImport/compare_hf_logits.py \
    /tmp/tl_slice/model.safetensors /tmp/cai.json
```

The committed-fixture variant of this check (pure-Python oracle, no torch)
runs in the test suite: `TestLlamaLogitParity` in
`tests/TestNeuralPretrained.pas`, against
`tests/fixtures/tiny_llama.safetensors` generated by
`tools/llama_tiny_fixture.py` (whose oracle is itself verified to match
`LlamaForCausalLM` to ~9e-7).

**Mistral and Qwen2** import through the same path:
`BuildMistralFromSafeTensors` (Llama skeleton + sliding-window attention
from the config's `sliding_window`; `null` = full attention) and
`BuildQwen2FromSafeTensors` (Llama skeleton + q/k/v projection biases,
permuted along with the rotate_half rows), both in
`neural/neuralpretrained.pas`. Or skip the per-family choice with
`BuildFromPretrained(path)`, which reads `config.json`'s `model_type` and
dispatches gpt2 / llama / mistral / qwen2 (pinned HF-parity fixtures
`tests/fixtures/tiny_mistral.*` / `tiny_qwen2.*`, generated by
`tools/mistral_qwen2_tiny_fixture.py`).

Memory note: `LlamaLogitsDump` passes `pTrainable=false`, which frees
every layer's training volumes during construction (`SetTrainable`),
cutting peak memory to roughly one third. Keep `SeqLen` small (e.g. 16) on
real checkpoints — attention cost grows with the square of it.
