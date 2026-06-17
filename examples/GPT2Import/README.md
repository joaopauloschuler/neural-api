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
* `-temp X` — sampling temperature: each step softmaxes `logits/X` and
  draws from the distribution instead of taking the greedy argmax.
* `-topk K` — keep only the `K` most probable tokens, renormalize and draw
  proportionally (the standard HF top-k; combines with `-temp`, and
  `-topk` alone samples at temperature 1.0). Without either flag the
  generation stays deterministic greedy argmax.

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

**Verified import targets** (logit parity vs HF transformers on real
weights): `openai-community/gpt2` (124M, see below) and
`distilbert/distilgpt2` (82M, 6 layers, `transformer.`-prefixed tensors)
— both load with `BuildGPT2FromSafeTensors` unchanged. distilgpt2 measured
2026-06: 2-layer/8192-vocab slice max |logit diff| **4.2e-5** over 98,304
logits, argmax agrees at every position. A committed pico fixture of REAL
distilgpt2 weights (`tests/fixtures/tiny_distilgpt2.*`, ~10 KB,
dimension-sliced by `make_pico_gpt2_fixture.py` — 2 of 6 layers, 2 heads × 4
dims, d_model 8, vocab 12, reference logits from `GPT2LMHeadModel` in
float64) is asserted by `TestDistilGPT2LogitParity` in every test run.

**Cerebras-GPT** (the truest open GPT-3 reproduction — exact GPT-3 recipe,
Chinchilla-scaled, `model_type: "gpt2"`) also loads via this GPT-2 path:
`cerebras/Cerebras-GPT-111M` verified 2026-06 with max |logit diff|
**7.5e-5** vs HF transformers float64 and identical argmax at every
position. Two deviations from the OpenAI checkpoints: `activation_function`
is `"gelu"` (the EXACT erf form — pass `pExactGelu=true` to
`BuildGPT2FromSafeTensors`, or use `BuildFromPretrained` with the
checkpoint's `config.json`, which reads it), and upstream ships only
`pytorch_model.bin` — which now loads DIRECTLY: every importer accepts a
`.bin` path via the restricted torch.save reader in
`neural/neuraltorchbin.pas` (no Python conversion needed; see
`tools/cerebras_gpt_fixture.py` for the committed real-weight pico fixture
`tests/fixtures/tiny_cerebras_gpt.*` asserted by
`TestCerebrasGPTLogitParity`). SHARDED `.bin` checkpoints work too: pass
the `pytorch_model.bin.index.json` path (the same `weight_map` index
layout as sharded safetensors) and the reader opens every referenced
`.bin` shard transparently — `BuildFromPretrained` on a repo directory
probes it automatically after `pytorch_model.bin`, and `HubFetchModel`
falls back to it when downloading.

## GPT-Neo / TinyStories

`neuralpretrained.pas` also imports **GPT-Neo** checkpoints
(`BuildGPTNeoFromSafeTensors`, or `BuildFromPretrained` on a config with
`model_type: "gpt_neo"`): the GPT-2 skeleton with ALTERNATING global /
locally-banded (window_size) causal attention, UNSCALED attention scores
(folded into `W_q` at load) and plain `nn.Linear` weight orientation. The
**roneneldan/TinyStories-1M..33M** reference checkpoints are GPT-Neo and
load through this path — verified 2026-06 on the real TinyStories-1M
weights: max |logit diff| **6.3e-5** vs HF transformers float64, and a
10-token greedy continuation of "Once upon a time there was a little girl
named" is token-for-token identical to HF's (" Lily. She loved to play
outside in the sunshine"). These repos ship `pytorch_model.bin` only — pass
the `.bin` path directly (the restricted torch.save reader in
`neural/neuraltorchbin.pas` handles it; `BuildFromPretrained` on the repo
directory also probes `pytorch_model.bin` automatically). The pico parity
fixture is pinned by `tools/gptneo_tiny_fixture.py` →
`TestGPTNeoLogitParity`.

## GPT-OSS (OpenAI gpt-oss MoE)

`neuralpretrained.pas` imports **GPT-OSS** — OpenAI's open-weight Mixture-of-
Experts decoder (`openai/gpt-oss-20b` and `openai/gpt-oss-120b`,
`model_type: "gpt_oss"`) — via `BuildGptOssFromSafeTensors[Ex]` (or
`BuildFromPretrained` on the config). It crosses several subsystems no other
importer combines: attention with a **learned per-head sink logit** appended to
the softmax denominator (`TNNetGptOssSinkAttention`), **alternating
sliding-window / full** attention per layer (`layer_types`), **YaRN** RoPE
(including gpt-oss's `truncate: false`) on every layer, and a **top-k routed
MoE** FFN with gpt-oss's **clamped-SwiGLU** expert activation
(`TNNetGptOssGatedSwiGLU`, interleaved gate|up, biased experts + router). The
real checkpoints ship the expert matrices as **MXFP4** 4-bit blocks
(`...experts.gate_up_proj_blocks`/`_scales` + `...down_proj_blocks`/`_scales`,
uint8); the importer **dequantizes them at load** via `DequantizeMXFP4`
(`neural/neuralmxfp4.pas`). A checkpoint that ships the experts already dense
(`...experts.gate_up_proj`/`down_proj`, batched `[E,in,out]`) loads straight.
Multi-shard `model.safetensors.index.json` checkpoints (the real 20B/120B are
sharded) are handled transparently by the reader.

Parity is pinned by a pico fixture (`tools/gpt_oss_tiny_fixture.py` →
`tests/fixtures/tiny_gpt_oss.*`) asserted in float64 by `TestGptOssLogitParity`,
with a second MXFP4-packed copy (`tiny_gpt_oss_mxfp4.*`) driving the MXFP4
dequant-at-load path end-to-end in `TestGptOssMXFP4LogitParity`.

**`gpt-oss-120b` is import-capable but RAM-gated** like the other large
checkpoints: pass `pInferenceOnly=True` and a sharded checkpoint so peak memory
stays near the quantized weight size. The 20B fits comfortably with
`pInferenceOnly`; the 120B needs a large-memory host (the importer expands the
MXFP4 experts to FP32 at load — the same RAM caveat as the other multi-billion
checkpoints documented above).

With a HuggingFace `tokenizer.json` sitting next to the checkpoint (every
GPT-2-family repo ships one), prompts can be plain text instead of ids:

```
GPT2Import /tmp/model.safetensors 64 0 -t "The meaning of life is"
GPT2Import /tmp/model.safetensors 64 0 -t "The meaning of life is" -temp 0.8 -topk 40
```

`neural/neuralhftokenizer.pas` (`TNeuralHFTokenizer`) loads the byte-level
BPE (vocab + ranked merges + the GPT-2 pre-tokenization regex and
bytes-to-unicode table), encodes the prompt and decodes the generated
continuation back to text. Raw-id mode keeps working without any tokenizer
file. Exact-id parity with the HF `tokenizers` library is pinned by
`tests/TestNeuralHFTokenizer.pas`.

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

* `make_pico_gpt2_fixture.py <src.safetensors> <out_prefix>` — slices a
  real checkpoint down EVERY dimension (layers, hidden, heads, head_dim,
  MLP, vocab) into a ~10 KB committable parity fixture of genuine weights
  plus float64 reference logits from `GPT2LMHeadModel` (needs
  torch/transformers; used to pin `tests/fixtures/tiny_distilgpt2.*`).
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
calls `TNNet.SetInferenceOnly` while building: every neuron's
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
