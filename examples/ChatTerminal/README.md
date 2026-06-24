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
neural-api/bin/x86_64-linux/bin/ChatTerminal q2/ --ctx 64 --fp32
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
family; `--int8` trades speed for memory.

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
| `--max-new-tokens N` | reply length cap | 128 |
| `--seed N` | RNG seed (reproducible sampling) | randomize |
| `--ctx N` | context window to build (`pSeqLen`) | model max |
| `--format NAME` | `chatml`/`llama2`/`llama3`/`zephyr`/`gemma`/`phi3`/`mistral` override | autodetect |
| `--system "msg"` | initial system prompt | none |
| `--int8` | int8 weight-only quantized inference (`pQuantizeInt8`) — slower, less RAM. **Not GPU-compatible** (see below) | fp32 (faster, more RAM) |
| `--low-memory` | drop each conv/linear layer's concatenated weight cache (`FConcatedWeights`) and compute per-neuron straight from the weights — less RAM, somewhat slower forward (`pLowMemory`). **Not GPU-compatible** (see below) | **on** |
| `--max-fast-memory` | keep the concatenated weight cache for a faster forward at the cost of more RAM — required for GPU offload | off |
| `--gpu` | OpenCL offload of the conv/linear matmuls (only when built with `-dOpenCL`) — incompatible with `--int8` and `--low-memory` (see below) | **on** when built with `-dOpenCL`, else off |
| `--no-gpu` | force CPU even when built with `-dOpenCL` | — |
| `--gpu-platform N` | OpenCL platform index | 0 |
| `--gpu-device N` | OpenCL device index within the platform | 0 |
| `--stats` | per-turn timing to **stderr**: TTFT (prefill + first token), steady-state decode tok/s, and `prompt N (reused K)` from the KV-cache reuse | off |
| `--no-cache-reuse` | re-prefill the whole prompt every turn instead of reusing the shared KV-cache prefix (A/B + debugging) | reuse on |
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
default; `--no-gpu` forces CPU, and `--gpu-platform N` / `--gpu-device N`
select the OpenCL device. A binary built without `-dOpenCL` is CPU-only and
ignores the `--gpu*` flags.

GPU offload is **incompatible with both `--int8` and `--low-memory`**, because
the OpenCL kernel consumes each accelerated layer's concatenated weight cache,
which neither path provides:

- **`--int8`** — the int8 path never builds the interleaved cache the kernel
  reads, so combining it with `--gpu` is rejected: the GPU is disabled and the
  model runs int8 on CPU (`[--gpu ignored: incompatible with --int8 - running
  int8 on CPU]`). Use `--fp32` (the default) for GPU.
- **`--low-memory`** (the default) drops exactly that weight cache. Enabling
  `--gpu` therefore *overrides* it: the cache is rebuilt and the low-memory
  forward is turned off on the accelerated layers (more RAM, the GPU's cost of
  entry). Since both `--low-memory` and `--gpu` default to on, the default GPU
  run keeps the cache; pass `--no-gpu` to honor low-memory on CPU, or
  `--max-fast-memory` to keep the cache explicitly.

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

Decoding is one full fixed-width forward per token (the `GPT2Import`
convention), so it works unchanged across every imported family — including
the ones whose normalization layers are not KV-cache streamable. Expect it
to be CPU-slow on multi-billion-parameter checkpoints; small instruct
models (0.5B-1B, `--ctx 512`) are the comfortable range.

## Testing

`--selftest` runs 31 offline checks (argument parsing, prompt assembly
against the byte-exact ChatML render, end-of-turn markers, REPL command
parsing) without needing any model files. For an end-to-end plumbing check,
any directory with a pico-sized random checkpoint plus a tokenizer works —
output is gibberish by construction, but loading, templating, streaming and
the stop paths are real.
