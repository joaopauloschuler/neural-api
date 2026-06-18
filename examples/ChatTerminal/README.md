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
| `--int8` | int8 weight-only quantized inference (`pQuantizeInt8`) — slower, less RAM | fp32 (faster, more RAM) |
| `--stats` | per-turn timing to **stderr**: TTFT (prefill + first token) and steady-state decode tok/s | off |
| `--selftest` | run the offline unit checks and exit | — |

The model is always built with `pInferenceOnly=true` (the REPL never
trains; ~1/3 the memory). Temperature and the penalties run through a
`TNNetLogitsProcessorChain` in the `TGenerationConfig` pipeline order
(penalty -> temperature -> sampler); without a sampler flag decoding is
greedy argmax. Generation stops on the tokenizer's EOS id, on the chat
format's end-of-turn marker (`<|im_end|>`, `<|eot_id|>`, `<end_of_turn>`,
`<|end|>`, `</s>` — matched as a token-id stop sequence in the generated
region and trimmed from the reply), or at `--max-new-tokens`.

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
