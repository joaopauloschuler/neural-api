# ChatTerminal: interactive chat REPL over any imported instruct checkpoint

A terminal chat program for the generic model importer dispatch
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
[+ `tokenizer_config.json`]) and chat:

```
lazbuild neural-api/examples/ChatTerminal/ChatTerminal.lpi
git clone https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct q2
neural-api/bin/x86_64-linux/bin/ChatTerminal q2/ --gpu
```

### Tested models

These models have been run/verified through this REPL:

| Model | model_type |
|---|---|
| Qwen/Qwen2.5-{0.5B,1.5B,3B,7B,14B,32B}-Instruct | qwen2 |
| Qwen/Qwen3-0.6B | qwen3 |
| Qwen/Qwen3-30B-A3B-Thinking-2507 | qwen3_moe |
| Qwen/Qwen3.6-27B | qwen3_5 |
| TinyLlama/TinyLlama-1.1B-Chat-v1.0 | llama |
| mistralai/Mistral-7B-Instruct-v0.3 | mistral |
| HuggingFaceTB/SmolLM2-1.7B-Instruct | llama |
| microsoft/Phi-3-mini-4k-instruct | phi3 |
| allenai/OLMoE-1B-7B-0125-Instruct | olmoe |
| state-spaces/mamba-130m-hf | mamba |
| openai-community/gpt2 | gpt2 |

### Multilingual generation (Cohere Command-R / Aya)

The Cohere family (`BuildCohereFromSafeTensors`, model_type `cohere` /
`cohere2`) is the leading **open multilingual** instruct family
(C4AI Command-R, Aya-Expanse-8B, Command-R7B). The same REPL drives it -
the importer handles Cohere's parallel residual, mean-subtracting bias-free
LayerNorm, interleaved RoPE, tied embeddings with `logit_scale` folded into
the LM head, and (cohere2) the alternating sliding/global attention with
NoPE on the global layers:

```
ChatTerminal /path/to/aya-expanse-8b --temperature 0.3
> Traduce al espanol: "The cat sits on the windowsill."
El gato esta sentado en el alfeizar de la ventana.
> Continue en francais.
Le chat est assis sur le rebord de la fenetre.
```

Aya / Command-R are tuned for cross-lingual instruction following, so a
single session can switch languages turn to turn. The chat format is
fingerprinted from the Cohere `tokenizer_config.json` like every other
family; the default int8 weights save memory *and* run faster than fp32
(and combine with `--gpu`).

The conversation is kept as a multi-turn history rendered through the
chat-template engine (`neural/neuralchat.pas`): the chat format is
auto-detected by fingerprinting `tokenizer_config.json`'s `chat_template`
(`DetectChatFormatFromConfigFile`) and each turn re-renders the whole
history (system prompt + user/assistant turns + generation prompt) and
encodes it with the HF tokenizer (`EncodeChat`). The assistant reply
**streams** to stdout as it decodes (delta printing with a BPE/UTF-8 prefix
guard, flushed per token so piped output streams too).

**`--format raw` — completion mode for base models.** BASE (non-instruct)
checkpoints such as `gpt2`, `mamba-130m` or the pythias have no chat
template; wrapping them in ChatML markup makes greedy decoding parrot the
markup back (the model has never seen it). `--format raw` drops templates
entirely: the REPL becomes a completion notebook over one running
transcript — each typed line is appended verbatim (no roles, no markup, no
BOS) and the model continues it; the continuation is appended back, so the
next turn extends the same document (and reuses the KV cache, since each
turn's token ids strictly extend the previous turn's). There is no
end-of-turn marker: generation stops on the tokenizer's EOS id or at
`--max-new-tokens` only, and base models rarely emit EOS — pass a small
cap (e.g. `--max-new-tokens 128`). `/reset` clears the transcript;
`/system` is ignored with a notice (there is no system role). Raw is never
autodetected — explicit flag only.

```
ChatTerminal gpt2/ --format raw --greedy --max-new-tokens 25
> Hello, I'm a language model,
 not a programming language. I'm a language model. ...
```

## Flags

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
| `--format NAME` | `chatml`/`llama2`/`llama3`/`zephyr`/`gemma`/`phi3`/`mistral` override, or `raw` (see below) | autodetect |
| `--system "msg"` | initial system prompt | none |
| `-p "prompt"` | one-shot: answer this single prompt, print the reply and exit without opening the REPL (see below) | interactive REPL |
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
| `--stats` | per-turn timing to **stderr**: TTFT (prefill + first token), steady-state decode tok/s, and `prompt N (reused K)` from the KV-cache reuse | off |
| `--profile` | per-layer-class forward timing to **stderr** after each turn (decode steps only — prefill is excluded), plus a `[sched]` line with the layer-graph scheduler stats (graph width, parallel vs serial passes, peak in-flight) | off |
| `--no-cache-reuse` | re-prefill the whole prompt every turn instead of reusing the shared KV-cache prefix (A/B + debugging) | reuse on |
| `--serial` | classic in-order serial layer loop, fully single-threaded, instead of the layer-graph parallel forward that also threads large conv/linear layers internally (see below) | parallel on |
| `--max-threads N` | cap the parallel forward at N worker threads (the pool becomes `Min(N, cpu threads)`, and per-layer chunk counts follow it); ignored with `--serial` | all CPU threads |
| `--selftest` | run the offline unit checks and exit | — |

The model is always built with `pTrainable=false` — the REPL never trains,
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

**KV-cache reuse across turns.** Each turn re-renders the whole history, but
its token prefix is almost always identical to what is already resident in
the KV cache (last turn's prompt + reply). The session keeps the cache,
diffs the new prompt against it (`CommonPrefixLen`), `TruncateTo`s the
divergent tail and prefills only the new tokens — so time-to-first-token
stays roughly flat instead of growing with the transcript. This is correct
regardless of tokenizer round-tripping (the diff always finds the true
shared prefix; `/system` and `/reset` simply diverge earlier and re-prefill
more), and it works the same with the int8 KV cache (truncation only
rewinds the cache length). It applies to pure-attention models only: a recurrent (SSM/Mamba/RWKV)
state cannot be truncated by position, so those fall back to a full
re-prefill each turn. `--no-cache-reuse` forces the full re-prefill (use
`--stats` to compare: watch `prompt N (reused K)` and TTFT).

**`-p "prompt"` — one-shot mode.** With `-p` the program answers that single
prompt and exits instead of opening the REPL: stdin is never read, so it
composes with scripts, pipes and benchmark harnesses. The reply streams to
stdout exactly as in interactive use (same token sink), `--system` still
applies, and under `--format raw` the prompt *is* the document and the model
completes it verbatim. There is no history and no second turn, so the KV
cache is filled once and never reused. The exit code is 0, or 1 when the
chat template rejects the turn (e.g. `--system` on a format without a system
role, such as gemma/mistral).

```
$ ChatTerminal q2/ --gpu --greedy -p "What is the capital of France?"
...
The capital of France is Paris.
$ ChatTerminal q2/ --greedy --stats -p "Hi!" > /dev/null   # timings only
```

## REPL commands

```
/exit            quit (EOF / Ctrl-D also exits cleanly)
/reset           clear the conversation history (the transcript in raw mode)
/system <msg>    set the system prompt (formats without a system role,
                 e.g. gemma/mistral, raise a template error - the turn is
                 dropped and the history stays consistent; ignored with a
                 notice in --format raw)
```

## Sample session

```
$ ChatTerminal /path/to/model --temperature 0.7 --top-p 0.9 --seed 42
Loading /path/to/model ...
Model: qwen2, 494032768 params, vocab 151936, context 1024, chat format chatml, int8 weights.
Type your message; /exit quits, /reset clears the history,
/system <msg> sets the system prompt.
> /system You are a terse assistant.
[system prompt set]
> Hi! What is the capital of France?
The capital of France is Paris.
> /exit
Bye.
```

Decoding streams through a `TNNetStreamingDecoder` KV cache: the model is
built at input width 1 and each token costs one width-1 forward over the
cached past (cache memory grows O(ctx), not the O(ctx²) score buffers of a
full-recompute decode). Expect it to be CPU-slow on multi-billion-parameter
checkpoints; small instruct models (0.5B-1B, `--ctx 512`) are the
comfortable range.

## ChatServer: the same engine over HTTP

`ChatServer` (in this folder) is a minimal OpenAI-style HTTP server over the
same shared engine (`neural/neuralchatengine.pas`, `TChatEngine`), so
neural-api models can be called from any codebase that speaks the OpenAI
REST shape. It takes the SAME command line as ChatTerminal (minus the
terminal-only one-shot `-p`) plus `--host` (default `127.0.0.1`, loopback
only) and `--port` (default `8080`):

```
$ ChatServer /path/to/model --temperature 0.7 --top-p 0.9 --port 8080
...
Serving model on http://127.0.0.1:8080/v1 (SSE streaming with "stream":true; Ctrl+C stops)

$ curl http://127.0.0.1:8080/v1/chat/completions \
    -d '{"messages":[{"role":"user","content":"Hi!"}],"max_tokens":64}'

$ curl -N http://127.0.0.1:8080/v1/chat/completions \
    -d '{"messages":[{"role":"user","content":"Hi!"}],"stream":true}'
data: {"id":"chatcmpl-1","object":"chat.completion.chunk", ... "delta":{"role":"assistant","content":""} ...}
data: {"id":"chatcmpl-1","object":"chat.completion.chunk", ... "delta":{"content":"Hello"} ...}
...
data: [DONE]
```

Endpoints: `POST /v1/chat/completions` (messages rendered through the
model's chat template), `POST /v1/completions` (plain completion, no
template - the `--format raw` path), `GET /v1/models`. A message
`content` is either a string or the OpenAI content-parts array
(`[{"type":"text","text":"..."}, ...]`) that current SDKs such as
openai-python and smolagents send by default; text parts are joined
with newlines and any non-text part (`image_url`, ...) is a 400, since
the server is text-only. With
`"stream": true` both POST endpoints stream the reply as OpenAI-style
Server-Sent Events - one `data:` chunk per decoded token, a
`finish_reason` chunk, then `data: [DONE]`.
`"stream_options": {"include_usage": true}` appends the usage chunk.
Only a literal JSON boolean is accepted for `stream` (a `"true"` string
is a 400, never a mis-parsed hang), and `"n"` other than 1 is rejected.
Response headers go out with the first token, so pre-generation failures
(bad template, context overflow) are still ordinary JSON 400s; if the
client disconnects mid-stream, generation aborts and the engine
invalidates its KV cache so the next request decodes cleanly.
Request fields `temperature`, `top_p`,
`top_k`, `min_p`, `repetition_penalty`, `frequency_penalty`,
`presence_penalty` and `max_tokens`/`max_completion_tokens` override the
launch defaults per request; absent fields fall back to them.

Requests are handled strictly one at a time (one model, one KV-cache
session; the non-threaded accept loop is the serialization). The KV-cache
prefix reuse still applies across requests: a growing conversation re-sent
in full each turn only prefills the new tail, so time-to-first-token stays
roughly flat. `ChatServer --selftest` runs the offline request-parsing and
parameter-overlay checks.

## Testing

`--selftest` runs 39 offline checks (argument parsing, prompt assembly
against the byte-exact ChatML render, end-of-turn markers, REPL command
parsing, the KV-cache-reuse prefix diff) without needing any model files. For an end-to-end plumbing check,
any directory with a pico-sized random checkpoint plus a tokenizer works —
output is gibberish by construction, but loading, templating, streaming and
the stop paths are real.
