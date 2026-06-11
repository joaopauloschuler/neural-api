# GPT2Import: load pretrained HuggingFace GPT-2 weights into a TNNet

This example imports a real, pretrained **GPT-2** checkpoint in the
HuggingFace **safetensors** format into a CAI `TNNet`, prints the inferred
configuration and architecture, and greedily generates a continuation from a
raw token-id prompt.

It is built on two reusable units (both pure Pascal, no external
dependencies):

* `neural/neuralsafetensors.pas` — a reader for the
  [safetensors](https://github.com/huggingface/safetensors) format
  (F32 natively; F16/BF16 decoded to single by bit manipulation; I64).
* `neural/neuralpretrained.pas` — `BuildGPT2FromSafeTensors()`, which infers
  the GPT-2 config from the tensor shapes (vocab/d_model from `wte`, n_ctx
  from `wpe`, n_layer by counting `h.N.` blocks), rebuilds the architecture
  from CAI layers and loads every weight. It fails loudly on missing or
  unexpected tensors and on any shape mismatch.

## Usage

```
GPT2Import <model.safetensors> [SeqLen] [NumHeads] [t0 t1 t2 ...]
```

* `SeqLen` — context window to build (default 64, `0` = the checkpoint's
  full `n_ctx`; the real GPT-2's 1024-token context is slow on CPU, so keep
  this small for a smoke test).
* `NumHeads` — attention head count, which is **not stored** in the
  safetensors. `0` (default) applies the GPT-2 family rule
  `n_head = n_embd/64` (gpt2 768→12, medium 1024→16, large 1280→20,
  xl 1600→25). The tiny test fixture needs an explicit `2`.
* `t0 t1 ...` — prompt token ids (default `464`, which is `"The"` in the
  real GPT-2 BPE vocabulary).

Try it immediately with the tiny committed fixture (from the repo root):

```
examples/GPT2Import$ lazbuild GPT2Import.lpi
$ bin/x86_64-linux/bin/GPT2Import tests/fixtures/tiny_gpt2.safetensors 16 2 0 1 2
```

## Fetching the real weights

```
curl -L -o /tmp/gpt2.safetensors \
  https://huggingface.co/openai-community/gpt2/resolve/main/model.safetensors
$ bin/x86_64-linux/bin/GPT2Import /tmp/gpt2.safetensors 64 0 464
```

(~500 MB; the larger `gpt2-medium` / `gpt2-large` / `gpt2-xl` checkpoints
work too — same tensor naming, the config is inferred from the shapes.)

The program prints token ids only: the repo's `TNeuralTokenizer` cannot read
HF `vocab.json`/`merges.txt` yet (byte-level BPE support is a noted
follow-up), so decode the ids with any GPT-2 tokenizer.

## How correctness is verified (logit parity)

Because the real checkpoint is half a gigabyte, the test suite does **not**
download it. Instead, `tools/gpt2_tiny_fixture.py` (pure Python stdlib — no
numpy/torch) generates:

* `tests/fixtures/tiny_gpt2.safetensors` — a random GPT-2-shaped checkpoint
  (n_layer=2, n_head=2, n_embd=8, n_ctx=16, vocab=11) in the exact HF
  layout, including HF Conv1D's transposed `[in, out]` weight storage and
  the ignorable `h.N.attn.bias` mask buffers;
* `tests/fixtures/tiny_gpt2_logits.json` — reference logits for fixed token
  sequences computed by an **independent** ~150-line GPT-2 forward written
  from the math (learned embeddings + per-token LayerNorm eps=1e-5 + causal
  softmax attention + `gelu_new` tanh-approximation MLP + tied `wte^T`
  head).

`tests/TestNeuralPretrained.pas` imports the fixture and asserts the TNNet's
logits match the oracle: **max |logit diff| ≈ 1.4e-6** (gate: 1e-4; hard
ceiling 1e-3). The same test unit covers F16/BF16 decoding against
hand-computed bit patterns, rejection of truncated/garbage safetensors
files, and the loud failure on missing tensors.

## Real-weight parity vs HuggingFace transformers

The fixture test above proves the import math on a random tiny checkpoint.
For the **real pretrained weights** there is a manual end-to-end check
against `transformers.GPT2LMHeadModel` (PyTorch), built from three tools in
this folder:

* `slice_gpt2.py <src> <dst> [layers] [vocab]` — slices a real checkpoint
  into a smaller-but-genuine one: keeps the first N transformer blocks and
  the first V rows of `wte` (a row-major prefix slice), drops the tied
  `lm_head.weight` and mask buffers. Lets RAM-limited machines load real
  weights (the importer triples weight memory with `Delta`/`BackInertia`).
* `GPT2LogitsDump` — imports a checkpoint and prints per-position
  full-vocab logits as JSON for a given token-id prompt.
* `compare_hf_logits.py <model.safetensors> <cai_logits.json>` — loads the
  **same** checkpoint into `GPT2LMHeadModel` (config inferred from tensor
  shapes, `tie_weights()` mirroring the importer's wte copy) and diffs
  every logit of every position. Needs `torch`, `transformers`,
  `safetensors`.

```
python3 slice_gpt2.py /tmp/gpt2.safetensors /tmp/gpt2_12L.safetensors 12 8192
GPT2LogitsDump /tmp/gpt2_12L.safetensors 16 0 464 262 976 ... > /tmp/cai.json
python3 compare_hf_logits.py /tmp/gpt2_12L.safetensors /tmp/cai.json
```

Measured on the real GPT-2 124M weights (12-token prompt): the FULL
unsliced model max |logit diff| = **2.7e-4** over all 603,084 logits with
logits spanning down to −287 (relative ≈ 1e-6, plain f32 accumulation),
2-layer/8192-vocab slice **3.2e-5**; the greedy argmax of every position
agrees, and 16-step greedy generation from "The" is token-for-token
identical to HF's. Gate: 1e-3.

Both example programs pass `pInferenceOnly=True` to the importer, which
calls `TNNet.MakeInferenceOnly` while building: every neuron's
`Delta`/`BackInertia` training volumes are shrunk to one element, cutting
weight memory to ~1/3. That is what lets the full 124M checkpoint run in
~2.3 GB peak RSS (it needed >3.8 GB before and OOM'd small machines); the
returned net can only `Compute()`, never train. Slicing remains useful for
fast iteration (82 MB / ~6 s vs 523 MB / ~16 s).

## Architecture mapping notes

* Per-token LayerNorm uses the new `TNNetTokenLayerNorm` (normalizes each
  token over d_model; the pre-existing `TNNetLayerNorm` normalizes over the
  whole volume including the sequence axis and is left untouched).
* GPT-2's learned absolute positions use the new
  `TNNetLearnedPositionalEmbedding` (`wpe`).
* The fused `c_attn` (d→3d with bias) maps to one
  `TNNetPointwiseConvLinear(3*d)`; `TNNet.AddMultiHeadSelfAttention` splits
  the Q|K|V slab per head exactly like HF does (contiguous d_head slices)
  and its out-projection receives `c_proj`.
* `TNNetGELU` already implements the tanh approximation = HF `gelu_new`.
* The LM head is tied to `wte` in GPT-2; CAI has no shared-storage tying
  layer yet, so the importer **copies** `wte` into a
  `TNNetPointwiseConvLinear(vocab)` head (exact for inference; a tying
  layer is a planned follow-up).
