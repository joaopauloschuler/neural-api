# ChatTerminal: interactive chat REPL over any imported instruct checkpoint

A terminal chat program for the checkpoints the generic importer dispatch
(`BuildFromPretrained` in `neural/neuralpretrained.pas`) supports: qwen2, quen2.5.
It is planned (coded) to support: quen3, gpt2, llama, mistral, /3, gemma/2/3, phi/phi3, gpt_neo(x), 
gptj, cohere/cohere2, rwkv, mamba, bloom, deepseek_v2. Point it at a
HuggingFace-style model directory (`config.json` + `model.safetensors` [or
sharded index / pytorch_model.bin] + `tokenizer.json`
[+ `tokenizer_config.json`]) and chat:

```
lazbuild neural-api/examples/ChatTerminal/ChatTerminal.lpi
git clone https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct q2
neural-api/bin/x86_64-linux/bin/ChatTerminal q2/ --gpu
```

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
family; `--int8` trades CPU speed for memory (and combines with `--gpu`).

The conversation is kept as a multi-turn history rendered through the
chat-template engine (`neural/neuralchat.pas`): the chat format is
auto-detected by fingerprinting `tokenizer_config.json`'s `chat_template`
(`DetectChatFormatFromConfigFile`) and each turn re-renders the whole
history (system prompt + user/assistant turns + generation prompt) and
encodes it with the HF tokenizer (`EncodeChat`). The assistant reply
**streams** to stdout as it decodes (delta printing with a BPE/UTF-8 prefix
guard, flushed per token so piped output streams too).

## Flags

| Flag | Meaning | Default |
| --- | --- | --- |
| `--temperature X` | sampling temperature (probability-domain `TNNetTemperatureProcessor`) | 1.0 (off) |
| `--top-k N` | `TNNetSamplerTopK` — NOTE: draws **uniformly** among the top K | off |
| `--weighted-top-k N` | `TNNetSamplerWeightedTopK` — HF semantics: draws **proportionally** to the renormalized top-K probabilities | off |
| `--top-p X` | `TNNetSamplerTopP` nucleus sampling (weighted draw) | off |
| `--min-p X` | `TNNetSamplerMinP` (weighted draw) | off |
| `--repetition-penalty X` | CTRL repetition penalty (`TNNetTokenHistoryPenalty`) | 1.0 (off) |
| `--frequency-penalty X` | frequency penalty | 0 (off) |
| `--presence-penalty X` | presence penalty | 0 (off) |
| `--max-new-tokens N` | reply length cap | 8192 |
| `--seed N` | RNG seed (reproducible sampling) | randomize |
| `--ctx N` | context window to build (`pSeqLen`) | model max |
| `--format NAME` | `chatml`/`llama2`/`llama3`/`zephyr`/`gemma`/`phi3`/`mistral` override | autodetect |
| `--system "msg"` | initial system prompt | none |
| `--fp32` | full-precision fp32 weights — faster, more RAM | **on** |
| `--int8` | int8 weight-only quantized inference (`pQuantizeInt8`) — less RAM, slower on CPU. Works with `--gpu`: the quantized codes stay resident on the device (see below) | fp32 (faster, more RAM) |
| `--low-memory` | drop each conv/linear layer's concatenated weight cache (`FConcatedWeights`) and compute per-neuron straight from the weights — less RAM, somewhat slower forward (`pLowMemory`). **Overridden by `--gpu`** (see below) | **on** |
| `--max-fast-memory` | keep the concatenated weight cache for a faster forward at the cost of more RAM — required for GPU offload | off |
| `--gpu` | OpenCL offload of the conv/linear matmuls (only when built with `-dOpenCL`) — overrides `--low-memory` (see below) | **on** when built with `-dOpenCL`, else off |
| `--cpu` | force CPU even when built with `-dOpenCL` | — |
| `--gpu-platform N` | OpenCL platform index | 0 |
| `--gpu-device N` | OpenCL device index within the platform | 0 |
| `--stats` | per-turn timing to **stderr**: TTFT (prefill + first token), steady-state decode tok/s, and `prompt N (reused K)` from the KV-cache reuse | off |
| `--profile` | per-layer-class forward timing to **stderr** after each turn (decode steps only — prefill is excluded), plus a `[sched]` line with the layer-graph scheduler stats (graph width, parallel vs serial passes, peak in-flight) | off |
| `--no-cache-reuse` | re-prefill the whole prompt every turn instead of reusing the shared KV-cache prefix (A/B + debugging) | reuse on |
| `--serial` | classic in-order serial layer loop, fully single-threaded, instead of the layer-graph parallel forward that also threads large conv/linear layers internally (see below) | parallel on |
| `--selftest` | run the offline unit checks and exit | — |

The model is always built with `pTrainable=false` (the REPL never
trains; ~1/3 the memory). **Memory vs. speed** is controlled by two
orthogonal axes on top of that: trainability gates the backprop buffers,
while `--low-memory`/`--max-fast-memory` toggles the *forward* weight cache.
Low memory is the default — each conv/linear layer drops its persistent
concatenated weight cache and computes per-neuron from the raw weights
(less resident RAM, a somewhat slower forward); `--max-fast-memory` keeps
the cache for a faster forward at the cost of more RAM. Orthogonally,
`--int8` swaps fp32 storage for weight-only int8 (less RAM, dequantized on
the fly).

**OpenCL / GPU offload.** When the binary is built with `-dOpenCL` (the
default compilation), the conv/linear matmuls are offloaded to the GPU by
default; `--cpu` forces CPU, and `--gpu-platform N` / `--gpu-device N`
select the OpenCL device. A binary built without `-dOpenCL` is CPU-only and
ignores the `--gpu*` flags.

GPU offload of an fp32 layer needs its concatenated weight cache, which
`--low-memory` (the default) drops. Combining it with `--gpu` therefore
*overrides* it (`[--low-memory ignored: incompatible with --gpu]`): the cache
is rebuilt and the low-memory forward is turned off on the accelerated layers
(more RAM, the GPU's cost of entry). Since both `--low-memory` and `--gpu`
default to on, the default GPU run keeps the cache; pass `--cpu` to honor
low-memory on CPU, or `--max-fast-memory` to keep the cache explicitly.

**`--int8` + `--gpu`** run together: quantized layers use a dedicated int8
device forward (`cai_dot_product_int8`) instead of the fp32 cache. The
interleaved int8 codes and per-row scales are uploaded **once** as resident
immutable device buffers (quantized layers are inference-only, so there is no
re-upload) and only each step's input travels to the GPU — 1/4 of the fp32
weight traffic, with the same fused bias/activation tail. So `--int8` saves
RAM on both paths: host RAM on CPU, host *and* device memory on GPU.

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

Temperature and the penalties run through a
`TNNetLogitsProcessorChain` in the `TGenerationConfig` pipeline order
(penalty -> temperature -> sampler); without a sampler flag decoding is
greedy argmax. Generation stops on the tokenizer's EOS id, on the chat
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
more). It applies to pure-attention models only: a recurrent (SSM/Mamba/RWKV)
state cannot be truncated by position, so those fall back to a full
re-prefill each turn. `--no-cache-reuse` forces the full re-prefill (use
`--stats` to compare: watch `prompt N (reused K)` and TTFT).

## REPL commands

```
/exit            quit (EOF / Ctrl-D also exits cleanly)
/reset           clear the conversation history
/system <msg>    set the system prompt (formats without a system role,
                 e.g. gemma/mistral, raise a template error - the turn is
                 dropped and the history stays consistent)
```

## Sample session

```
$ ChatTerminal /path/to/model --temperature 0.7 --top-p 0.9 --seed 42
Loading /path/to/model ...
Model: qwen2, 494032768 params, vocab 151936, context 1024, chat format chatml, fp32 weights.
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

## Testing

`--selftest` runs 55 offline checks (argument parsing, prompt assembly
against the byte-exact ChatML render, end-of-turn markers, REPL command
parsing, the KV-cache-reuse prefix diff) without needing any model files. For an end-to-end plumbing check,
any directory with a pico-sized random checkpoint plus a tokenizer works —
output is gibberish by construction, but loading, templating, streaming and
the stop paths are real.
