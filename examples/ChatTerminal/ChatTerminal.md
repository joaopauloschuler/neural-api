# ChatTerminal: interactive chat REPL over any imported instruct checkpoint

A terminal chat program over the shared chat engine (`TChatEngine` in
`neural/neuralchatengine.pas`). The supported model families, the build
line and every sampling/memory/GPU flag are documented once in
[README.md](README.md); this page covers what is specific to the REPL.

```
lazbuild neural-api/examples/ChatTerminal/ChatTerminal.lpi
git clone https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct q2
neural-api/bin/x86_64-linux/bin/ChatTerminal q2/ --gpu
```

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

## Terminal-only flag

| Flag | Meaning | Default |
| --- | --- | --- |
| `-p "prompt"` | one-shot: answer this single prompt, print the reply and exit without opening the REPL (see below) | interactive REPL |

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

## Testing

`--selftest` runs 39 offline checks (argument parsing, prompt assembly
against the byte-exact ChatML render, end-of-turn markers, REPL command
parsing, the KV-cache-reuse prefix diff) without needing any model files. For an end-to-end plumbing check,
any directory with a pico-sized random checkpoint plus a tokenizer works —
output is gibberish by construction, but loading, templating, streaming and
the stop paths are real.
