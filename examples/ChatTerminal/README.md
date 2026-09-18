# ChatTerminal and ChatServer: two front ends over one chat engine

This folder holds two programs that run an imported instruct checkpoint
through the same shared engine (`TChatEngine` in
`neural/neuralchatengine.pas`):

- [**ChatTerminal**](ChatTerminal.md) — an interactive chat REPL in the
  terminal, with a one-shot `-p "prompt"` mode for scripts.
- [**ChatServer**](ChatServer.md) — a minimal OpenAI-style HTTP server
  (`/v1/chat/completions`, `/v1/completions`, `/v1/models`, SSE streaming).

Both take the same command line: a model directory followed by the flags
listed on this page (`ParseArgs` in `neural/neuralchatengine.pas` parses them
for both). ChatTerminal adds `-p`; ChatServer adds `--host` and `--port`.
Each program's page documents only what is specific to it.

Both programs drive the generic model importer dispatch
(`BuildFromPretrained` in `neural/neuralpretrained.pas`). It supports: qwen2, qwen2.5, qwen3,
qwen3_moe, qwen3_5, mamba, gpt2, llama, mistral, phi3, olmoe (see the
tested-models list below).
It is planned (coded) to support: mixtral,
gemma/2/3, recurrent_gemma, phi, gpt_oss, gpt_neo(x), gptj,
gpt_bigcode, starcoder2, opt, cohere/cohere2, olmo2,
granite/granitemoe, glm4, minicpm, bitnet, internlm2, falcon, rwkv,
falcon_mamba, mamba2, nemotron_h, jamba, bloom, deepseek_v2,
`.gguf` llama.cpp models and llama4/llama4_text (Llama 4 text-only —
iRoPE + MoE, e.g. Llama-4-Scout; the vision tower is out of scope). Point it at a
HuggingFace-style model directory (`config.json` + `model.safetensors` [or
sharded index / pytorch_model.bin] + `tokenizer.json`
[+ `tokenizer_config.json`]) and run either program:

```
lazbuild neural-api/examples/ChatTerminal/ChatTerminal.lpi
lazbuild neural-api/examples/ChatTerminal/ChatServer.lpi
git clone https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct q2
neural-api/bin/x86_64-linux/bin/ChatTerminal q2/ --gpu
neural-api/bin/x86_64-linux/bin/ChatServer q2/ --gpu --port 8080
```

### Tested models

These models have been run/verified through ChatTerminal:

| Model | model_type |
|---|---|
| Qwen/Qwen2.5-{0.5B,1.5B,3B,7B,14B,32B}-Instruct | qwen2 |
| Qwen/Qwen3-0.6B | qwen3 |
| Qwen/Qwen3-30B-A3B-Thinking-2507 | qwen3_moe |
| Qwen/Qwen3.5-0.8B | qwen3_5 |
| Qwen/Qwen3.5-4B | qwen3_5 |
| Qwen/Qwen3.6-27B | qwen3_5 |
| Qwen/Qwen3.8-27B | qwen3_5 |
| TinyLlama/TinyLlama-1.1B-Chat-v1.0 | llama |
| mistralai/Mistral-7B-Instruct-v0.3 | mistral |
| HuggingFaceTB/SmolLM2-1.7B-Instruct | llama |
| microsoft/Phi-3-mini-4k-instruct | phi3 |
| allenai/OLMoE-1B-7B-0125-Instruct | olmoe |
| state-spaces/mamba-130m-hf | mamba |
| openai-community/gpt2 | gpt2 |

## Common flags

Sampling defaults resolve **per parameter** as: explicit flag >
the model's `generation_config.json` (the checkpoint author's recommended
`temperature`/`top_p`/`top_k`/`repetition_penalty`; `do_sample: false` is
honored as greedy) > the built-in fallback **top-p 0.2 +
repetition-penalty 1.05**. A config `top_k` maps to the **weighted** top-k
(and `top_p` is preferred over `top_k`) because this library's plain top-k
draws uniformly. `--greedy` hard-overrides everything.

| Flag | Meaning | Default |
| --- | --- | --- |
| `--greedy` | deterministic argmax: no sampler, no temperature, no penalties — overrides all sampling flags **and** `generation_config.json` (the CPU/GPU parity + debugging mode) | off |
| `--temperature X` | sampling temperature (probability-domain `TNNetTemperatureProcessor`) | config, else 1.0 (off) |
| `--top-k N` | `TNNetSamplerTopK` — NOTE: draws **uniformly** among the top K | off |
| `--weighted-top-k N` | `TNNetSamplerWeightedTopK` — HF semantics: draws **proportionally** to the renormalized top-K probabilities | config `top_k`, else off |
| `--top-p X` | `TNNetSamplerTopP` nucleus sampling (weighted draw) | config `top_p`, else 0.2 |
| `--min-p X` | `TNNetSamplerMinP` (weighted draw) | off |
| `--repetition-penalty X` | CTRL repetition penalty (`TNNetTokenHistoryPenalty`) | config, else 1.05 |
| `--frequency-penalty X` | frequency penalty | 0 (off) |
| `--presence-penalty X` | presence penalty | 0 (off) |
| `--max-new-tokens N` | reply length cap | 8192 |
| `--seed N` | RNG seed (reproducible sampling) | randomize |
| `--ctx N` | context window to build (`pSeqLen`) — KV-cache memory grows ~O(ctx), and the cache is allocated in full when the session opens | model max, capped at 32768 (the startup banner says so; go past the cap, or below it to save RAM, with `--ctx`) |
| `--format NAME` | `chatml`/`llama2`/`llama3`/`zephyr`/`gemma`/`phi3`/`mistral` override, or `raw` (no template: each program's page says what raw means for it) | autodetect |
| `--system "msg"` | initial system prompt | none |
| `--int8` | int8 weight-only quantized inference (`pQuantizeInt8`) — less RAM **and** faster than fp32 on both CPU (fused AVX2 int8 kernels) and GPU: the quantized codes stay resident on the device (see below) | **on** |
| `--fp32` | full-precision fp32 weights — more RAM, slower. Also switches the KV-cache default to fp32 | off |
| `--int4` | int4 (Q4_0, blocks of 32) weights on the convolution/projection layers, int8 elsewhere — half the weight RAM of `--int8`. A Q4_0 tensor of a `.gguf` checkpoint loads straight into the int4 rows (same codes, same block scales, no FP32 and no int8 row in between) whenever one call fills the whole layer; every other tensor streams into int8 rows and `TNNet.QuantizeWeightsInt4` requantizes it after the load. On `--gpu` the packed codes stay resident on the device (`cai_dot_product_int4_splitk`, FP32 activations). Output quality below `--int8` | off |
| `--kv-int8` | int8-quantized KV cache (per-row scale = max\|row\|/127): ~1/4 the KV RAM at long context, identical on CPU and GPU. Slightly lossy logits (drift on the order of e-2, greedy argmax stable); the FP32 K/V buffers are never allocated | **on** whenever the weights are int8 |
| `--kv-fp32` | keep the bit-exact FP32 KV cache while the weights stay int8 | off |
| `--low-memory` | drop each conv/linear layer's concatenated weight cache (`FConcatedWeights`) and compute per-neuron straight from the weights — less RAM, somewhat slower forward (`pLowMemory`). **Overridden by `--gpu`** (see below) | **on** |
| `--max-fast-memory` | keep the concatenated weight cache for a faster forward at the cost of more RAM — required for GPU offload | off |
| `--gpu` | OpenCL offload of the conv/linear matmuls (only when built with `-dOpenCL`) — overrides `--low-memory` (see below) | **on** when built with `-dOpenCL`, else off |
| `--cpu` | force CPU even when built with `-dOpenCL` | — |
| `--gpu-platform N` | OpenCL platform index | 0 |
| `--gpu-device N` | OpenCL device index within the platform | 0 |
| `--experimental-fp16` | **experimental and under construction.** Half-precision activations in the int8 OpenCL matmuls (`cai_dot_product_int8_h` and its split-K twin): the weights stay int8 and the layer still hands the CPU a `Single`, only the column matrix inside OpenCL memory narrows. Logits are not bit-exact. Needs `--gpu` and int8 weights — `--cpu`, `--fp32` or `--int4` ignores it, and a device that rejects the half kernel keeps the FP32 activations | off |
| `--experimental-int8-input` | **experimental and under construction.** `TNNet.EnableInt8Input` after the weights are quantized: every int8-weight layer keeps an int8 copy of its input with one scale per tensor. Today only `TNNetConvolution` has an int8 x int8 CPU kernel (`ComputeInt8Int8CPU`); the fully connected blocks of an LLM arm the copy but still run int8 x FP32, so on ChatTerminal's models this changes nothing yet. Needs int8 or int4 weights — `--fp32` ignores it. With `--int4` the printed count includes the int4 layers, which arm the copy themselves | off |
| `--no-gpu-shared-kernel` | give every layer its own OpenCL kernel handles and command queue instead of the net-wide shared ones (see below) | shared on |
| `--stats` | per-turn timing to **stderr**. `input:` prompt tokens, `(reused K, prefix P of C cached)` — K tokens resumed from the KV cache or a cache checkpoint, P the length of the token-id prefix the prompt shares with the C ids cached from the previous turn (a small K next to a large P says the divergence fell below every checkpoint), TTFT (prefill + first token) and `prefill X tok/s`; `output:` reply tokens, their time from the end of prefill, and the steady-state decode tok/s; a per-decode-step phase split; and `total input` / `total output` lines accumulated for as long as the process runs (`cached` = prompt tokens the reuse skipped) | off |
| `--profile` | per-layer-class forward timing to **stderr** after each turn, one `[profile] prefill:` report (one table per net that ran: the windows on the `--prefill-window` twin, the windows on the tail twin, the single steps on the main net, each under a header line with its window count) and one for the decode steps, each followed by a `[sched]` line with the layer-graph scheduler stats (graph width, parallel vs serial passes, peak in-flight) | off |
| `--mtp` | Qwen3.5/3.8 **multi-token-prediction self-speculative decoding**: the checkpoint's own `mtp.*` module drafts the next token and a width-2 trunk window verifies it, so an accepted draft commits two tokens for one trunk forward. **Greedy only** — pass `--greedy` too. See below | off |
| `--no-mtp` | ordinary token-by-token decoding | **on** |
| `--no-cache-reuse` | re-prefill the whole prompt every turn instead of reusing the shared KV-cache prefix (A/B + debugging) | reuse on |
| `--cache-checkpoints N` | hybrid/recurrent nets only (`qwen3_5`, `qwen3_8`, mamba, ...): keep up to N **cache checkpoints** of the recurrent state (see *cache checkpoints* below), captured after every prefill window and at the end of the prompt and of the reply, kept geometrically denser near the newest token; a prompt resumes from the deepest checkpoint at or below its shared token prefix and prefills only the tail. 0 turns the route off (full re-prefill every turn); 1 and values above 2048 stop the program with an error before loading. Inert with a notice on pure-attention nets, whose KV cache is truncated to the prefix instead | 16 with `--gpu`, 8 on the CPU |
| `--prefill-window N` | prefill the prompt N tokens per forward on a width-N twin of the net (`TChatEngine.WindowNN`); the state crosses to the width-1 net with a session snapshot before the tail and the decode loop. The tail that does not fill a window is fed one token at a time — nothing is padded. The twin borrows the loaded net's weights (`BuildFromPretrained` with `pWeightOwner`: the int8/int4 tables in RAM and the resident codes on the device are shared, the checkpoint is read once, and the twin allocates no weight storage — it costs its activations). Model families outside the Llama builder (`PretrainedModelTypeCanBorrowWeights`: llama, mistral, qwen*, gemma*, phi3, olmo*, mixtral, glm4, granite*, minicpm, bitnet) fall back to a full second build — checkpoint read twice, weights held twice — and the startup notice says so. N must be 0 or at least 2, and below the context length (`--ctx`), otherwise the program stops with an error before loading | 0 (one token per forward) |
| `--prefill-tail-window T` | width of a second, width-T twin (`TChatEngine.TailNN`) that feeds what the width-N windows leave over T tokens per forward, so at most T-1 tokens go one at a time: the prompt runs down a ladder of widths N, then T, then 1. On a 7880-token prompt with N=256 the 199-token leftover cost 199 single steps, about a fifth of the time-to-first-token; with T=16 it costs 12 tail windows and 7 single steps. The tail twin borrows the weights like the width-N twin (it costs its activations) and is not built on the full-second-build fallback. T must be below N and needs `--prefill-window` (otherwise the program stops with an error before loading); 0 picks 16 when that is below N, else a notice and no tail twin; 1 builds none | 0 (auto) |
| `--serial` | classic in-order serial layer loop, fully single-threaded, instead of the layer-graph parallel forward that also threads large conv/linear layers internally (see below) | parallel on |
| `--max-threads N` | cap the parallel forward at N worker threads (the pool becomes `Min(N, cpu threads)`, and per-layer chunk counts follow it); ignored with `--serial` | all CPU threads |
| `--selftest` | run the program's own offline unit checks and exit (see the program's page) | — |

The model is always built with `pTrainable=false` — neither program trains,
so the per-layer error buffers and each neuron's optimizer-state volumes
(delta/inertia) are freed outright, not just shrunk (on a multi-billion-
parameter model the per-neuron object overhead alone is gigabytes).
**Memory vs. speed** is controlled by two orthogonal axes on top of that:
trainability gates the backprop buffers, while
`--low-memory`/`--max-fast-memory` toggles the *forward* weight cache.
Low memory is the default — each conv/linear layer drops its persistent
concatenated weight cache and computes per-neuron from the raw weights
(less resident RAM, a somewhat slower forward); `--max-fast-memory` keeps
the cache for a faster forward at the cost of more RAM. Orthogonally, the
weight storage is int8 by default — quantized at construction time (no FP32
weight copy is ever allocated; large checkpoints stream row-by-row straight
into the int8 codes, so loading never spikes to the FP32 size) and run
through fused int8 kernels that are both smaller *and* faster than fp32 on
CPU and GPU; `--fp32` opts back into full-precision storage, and `--int4`
quantizes the convolution/projection layers one step further (Q4_0 blocks of
32, half the weight RAM of int8 on CPU and GPU, lower output quality).

The decode-time KV cache follows the weight mode: with int8 weights (the
default) each attention layer's K/V rows are quantized to int8 with a
per-row scale as they are appended — ~1/4 the KV RAM, the full-size FP32
K/V buffers are never allocated, and the fused int8 kernels read the codes
directly. The drift is small (logits within ~e-2, greedy argmax stable —
see `TestKVCacheInt8DriftWithinTolerance`) but decode is not bit-exact vs
the FP32 cache; `--kv-fp32` opts back into the exact cache, and `--fp32`
weights default to it. The KV cache behaves identically on CPU and GPU
(the cached decode path is the same code).

**OpenCL / GPU offload.** When the binary is built with `-dOpenCL` (the
default compilation), the conv/linear matmuls are offloaded to the GPU by
default; `--cpu` forces CPU, and `--gpu-platform N` / `--gpu-device N`
select the OpenCL device. A binary built without `-dOpenCL` is CPU-only and
ignores the `--gpu*` flags.

Every accelerated layer shares one net-wide OpenCL program and kernel cache
(`TNNet.EnableOpenCL`'s `pHasSharedKernel`), so a kernel is compiled once and
the layers submit to a shared command queue. `--no-gpu-shared-kernel` opts
out: each layer builds its own kernel handles and gets its own queue. That is
measurably *slower* on the devices tested here (it also switches the scheduler
off worker-0 routing) — it exists as a performance A/B knob, an escape hatch
for drivers that mishandle sharing, and a profiling mode.

The profiling use follows from the queues. On the shared queue, a layer that
enqueues a kernel returns before the kernel runs, so `--profile` charges it the
enqueue only and every kernel's real cost lands in the `OpenCL queue drain`
line under the table. With private queues, a consumer whose source sits on
another queue calls `TNNetLayer.OpenCLWaitOutputIfAnotherQueue`, which blocks
until that source is done — so the GPU time moves out of the drain and into the
per-layer rows, charged to the layer that waited rather than the layer that
computed. Read the ranking, not the total: the private-queue run is a slower
program than the one you are profiling.

GPU offload of an fp32 layer needs its concatenated weight cache, which
`--low-memory` (the default) drops. Combining it with `--gpu` therefore
*overrides* it (`[--low-memory ignored: incompatible with --gpu]`): the cache
is rebuilt and the low-memory forward is turned off on the accelerated layers
(more RAM, the GPU's cost of entry). Since both `--low-memory` and `--gpu`
default to on, the default GPU run keeps the cache; pass `--cpu` to honor
low-memory on CPU, or `--max-fast-memory` to keep the cache explicitly.

**int8 + `--gpu`** (both defaults) run together: quantized layers use a
dedicated int8 device forward (`cai_dot_product_int8`) instead of the fp32
cache. The interleaved int8 codes and per-row scales are uploaded **once** as
resident immutable device buffers (quantized layers are inference-only, so
there is no re-upload) and only each step's input travels to the GPU — 1/4 of
the fp32 weight traffic, with the same fused bias/activation tail. So int8
wins on both paths: less host RAM and a faster forward on CPU, less host
*and* device memory plus less weight traffic on GPU.

**Parallel execution (CPU).** One switch, `--serial`, selects between two
forward paths; each path drives *both* levels of parallelism together:

- **Parallel (the default; `--serial` opts out)** runs each token step through
  `TNNet.ComputeParallel`, the dependency-graph scheduler: independent layers —
  e.g. the q/k/v projections off one RMSNorm, or an MHA block's sibling
  attention heads — are computed concurrently by a worker pool, while dependent
  layers still wait for their inputs. The same path also turns on **intra-layer
  threading**: each *large* conv/linear layer (above the ~4M-MAC work
  threshold) additionally splits its own forward across the pool via worker 0;
  smaller layers stay serial because the pool dispatch costs more than it saves.
  Output is bit-identical to the serial loop (only the order *between
  independent layers* changes, and the intra-layer range split preserves the
  per-neuron reduction order). Straight-line graph regions and graphs whose
  parallel gain cannot repay the scheduler overhead fall back to the serial
  loop automatically; `--profile`'s `[sched]` line shows the parallel/serial
  pass split actually achieved. Intra-layer threading is what helps on
  multi-billion-parameter checkpoints whose big projections dominate; on sub-1B
  models no layer crosses the threshold, so it costs nothing.
- **Serial (`--serial`)** runs the classic in-order layer loop through
  `TNNet.ComputeSerial`, fully single-threaded — both layer-graph parallelism
  and intra-layer threading are off.

`--max-threads N` caps the worker pool on the parallel path (`TNNet.MaxThreadNum`,
the inference twin of `TNeuralFit.MaxThreadNum`): the pool is sized
`Min(N, cpu threads)` and each threaded layer splits into that many chunks.
Useful when the machine is shared, or with `--gpu`, where fewer host workers
leave more cores to the OpenCL driver.

Temperature and the penalties run through a
`TNNetLogitsProcessorChain` in the `TGenerationConfig` pipeline order
(penalty -> temperature -> sampler); the effective settings come from the
flag > `generation_config.json` > fallback resolution above (the startup
banner prints what was resolved), and `--greedy` forces plain argmax.
Generation stops on the tokenizer's EOS id, on the chat
format's end-of-turn marker (`<|im_end|>`, `<|eot_id|>`, `<end_of_turn>`,
`<|end|>`, `</s>` — matched as a token-id stop sequence in the generated
region and trimmed from the reply), or at `--max-new-tokens`.

**KV-cache reuse across turns.** Each turn (a REPL turn, or an HTTP request
that re-sends the conversation) re-renders the whole history, but
its token prefix is almost always identical to what is already resident in
the KV cache (last turn's prompt + reply). The session keeps the cache,
diffs the new prompt against it (`CommonPrefixLen`), `TruncateTo`s the
divergent tail and prefills only the new tokens — so time-to-first-token
stays roughly flat instead of growing with the transcript. This is correct
regardless of tokenizer round-tripping (the diff always finds the true
shared prefix; `/system` and `/reset` simply diverge earlier and re-prefill
more), and it works the same with the int8 KV cache (truncation only
rewinds the cache length). Truncation applies to pure-attention models
only: a recurrent (SSM/Mamba/RWKV) state cannot be truncated by position.

**Cache checkpoints (hybrid/recurrent models).** A net with recurrent
layers (`qwen3_5`, `qwen3_8`, mamba, ...) splits its state in two halves:
the attention K/V, which `TruncateTo` rewinds by position like on a
pure-attention net, and the recurrent state, a fixed-size summary with no
per-position history. `TChatEngine` therefore keeps a store of up to N
**cache checkpoints** (`--cache-checkpoints N`; `TNNetDecoderStateCheckpoint`:
one copy of every recurrent layer's state and step count, no K/V), each
tagged with the number of tokens fed when it was captured. Captures happen
after every window the `--prefill-window` twins feed, at the end of the
prompt and at the end of the reply; the store is sized once at load and a
capture allocates nothing. The next prompt is diffed against the cached ids
(`CommonPrefixLen`), the deepest checkpoint at or below that prefix is
picked, the K/V is truncated to its position, its recurrent state is put
back (`RestoreStateFrom`) and only the tokens after it are prefilled —
bit-identical to a fresh prefill. So a client that echoes the reply resumes
at the end of the reply, one that re-renders the assistant turn resumes at
the end of the prompt, and an agent that edits or appends to a message deep
inside the history resumes at the last checkpoint before the edit instead of
re-prefilling everything.

Retention: with W the finest capture spacing (the tail window, else the
prefill window, else 256 capped at half the context), context C and N
slots, `r = (C / W)^(1 / N)` and band k covers distances `[W r^k, W r^(k+1))`
from the newest fed token (distances below W count as band 0); on every
capture the store keeps the deepest checkpoint per band (denser near the
end, where prompts usually diverge) and drops the rest, so the end-of-prompt
and end-of-reply checkpoints of the newest request, its two deepest, always
survive. A divergence at distance d therefore costs at most about `(r - 1) d`
extra prefill on top of the unavoidable d; with `--prefill-window 256
--prefill-tail-window 16` at C = 32768 that is 0.61 d for N = 16 (r = 1.61)
and 1.6 d for N = 8 (r = 2.59). Each
checkpoint costs the recurrent state only (about 4 MB per GatedDeltaNet
layer on a 27B Qwen3.8, so about 190 MB per checkpoint, 3 GB for N = 16),
held in OpenCL memory under `--gpu` (a capture is a copy between resident
buffers) and in host RAM otherwise; the load notice prints the figure. The
attention K/V is never copied. `--no-cache-reuse` turns both routes off (use
`--stats` to compare: watch `prompt N (reused K, prefix P of C cached)` and
TTFT).

**`--mtp` — multi-token-prediction self-speculative decoding (Qwen3.5/3.8).**
Qwen3.5 and Qwen3.8 checkpoints ship an extra one-block module under the
`mtp.*` tensor prefix that predicts the token at *t+2* from the trunk's hidden
state at *t* and the embedding of the committed token at *t+1*. With `--mtp`
the engine loads that module as a second net and decodes through
`GenerateTokensMTPSpeculative`: the module drafts, and a **width-2** trunk
window verifies the draft and produces the next token in the same forward, so
an accepted draft commits two tokens for one trunk forward. It is exact — every
committed token is the trunk's own argmax over fully committed context — and
the payoff is `2/(2-a)` tokens per trunk forward at acceptance rate `a`, 1.0 at
the worst case and 2.0 at the best. `--stats` prints the measured acceptance
rate and tokens per trunk forward, which is how you decide whether it pays on
your GPU.

The flag is **off by default** and it never fails a chat: whenever it cannot be
honoured the engine prints a one-line notice and decodes normally. The reasons
are a non-Qwen3.5 checkpoint, a `config.json` with no `mtp_num_hidden_layers`,
a weights file that is not safetensors, an `mtp.*` import error, and — the one
that hits by default — **sampling**. The driver is greedy only, and
ChatTerminal's sampling defaults (top-p 0.2 + repetition-penalty 1.05) count as
sampling, so `--mtp` on its own is ignored with a notice. Use it as:

```
$ ChatTerminal qwen3.8-27b/ --gpu --greedy --mtp --stats -p "Explain RoPE."
```

`--mtp` also turns `--prefill-window` off with a notice: the driver prefills
through its own width-2 trunk, so the twins would be built and never fed.

Two limits worth knowing before you measure. First, there is **no cache reuse
across turns** under `--mtp`: the driver resets both sessions on every call, so
each turn re-prefills the whole transcript. Second, the driver stops only on
token ids below 2, so it always decodes the full `--max-new-tokens` budget and
the reply is trimmed at the first EOS / end-of-turn token afterwards — keep
`--max-new-tokens` small (e.g. 128) or every turn costs the whole budget.
Extra memory while it is on: the trunk is built at input width 2 (the layer
activations double; weights and KV cache are unchanged) plus one transformer
block for the module and its own small KV cache.
