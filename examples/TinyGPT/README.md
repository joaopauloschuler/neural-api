# TinyGPT — char-level GPT-style transformer, end-to-end

A tiny, fully self-contained **decoder-only (GPT-style) transformer** that learns
to predict the next character on a small embedded text corpus, then
**autoregressively generates** a short sample from a seed prompt. It is the
capstone demo for the transformer-building-blocks line of work in this repo: it
wires the existing causal-attention / SwiGLU-FFN transformer block builder
(`TNNet.AddTransformerEncoderBlock` with `CausalMask=true`) into a complete
char-level language model, trained with `TNeuralDataLoadingFit`.

Everything is hardcoded — **no downloads, no external data files** — and the
configuration is sized to train well under five minutes on a pure-CPU machine
without exhausting memory.

## Architecture

```
Input(SeqLen=24, 1, VocabSize=128)            one-hot char
  -> PointwiseConvLinear(d_model=64)          per-token token projection
  -> AddPositionalEmbedding                   absolute positional encoding
  -> AddTransformerEncoderBlock(Heads=4, d_ff=64, CausalMask=true)   x2
       (causal multi-head self-attention + SwiGLU FFN, pre-norm residuals)
  -> PointwiseConvReLU(64)
  -> FullConnectReLU(64)
  -> FullConnectLinear(VocabSize=128)
  -> SoftMax                                  next-char distribution
```

The assembled network is 43 layers / ~188k weights; each transformer block
expands to 4 heads of `TNNetScaledDotProductAttention(d_k=16)` over
`TNNetSplitChannels` slices, re-concatenated and out-projected.

Notes on the design choices (all dictated by existing, already-tested library
building blocks):

- **Decoder-only via the encoder block + causal mask.** A GPT block is just
  causal self-attention + FFN with no cross-attention.
  `AddTransformerEncoderBlock(..., CausalMask=true)` is exactly that. The repo's
  `AddTransformerDecoderBlock` is an *encoder-decoder* block (it requires an
  `EncoderOutput` for cross-attention), which is not what a GPT needs.
- **One-hot input, not `TNNetEmbedding`.** Char-level input is one-hot encoded so
  the stock `GenerateStringFromChars` sampler (which calls
  `OneHotEncodingReversed`) works unchanged. A `PointwiseConvLinear` then projects
  each one-hot token to `d_model` — the learnable token-embedding equivalent.
- **Absolute positional embedding** (`TNNetAddPositionalEmbedding`), *not* RoPE.
  The library's RoPE flag must be paired with a token-only `TNNetEmbedding`; we
  use a single, consistent positional scheme (absolute) here.
- **Pointwise (1x1) projections everywhere inside the blocks** keep the
  `(SeqLen, 1, d_model)` token axis intact; the only flattening is the final LM
  head, which collapses the context window into one next-token distribution.

## Corpus

A small set of 8 hardcoded "quick brown fox / lazy dog"-style sentences,
lower-cased, each terminated with `chr(1)` (the end-of-sequence marker that stops
autoregressive sampling), repeated to give the trainer enough samples. Honest
headline: with this tiny corpus the model **memorizes the corpus structure** — it
learns to continue a seed prompt into corpus-style text (the right words, in
plausible order). This is the expected, acceptable outcome for a sub-5-minute
char-level capstone; it is not claimed to generalize to unseen English.

## Build & run

```
cd examples/TinyGPT
lazbuild --build-mode=Release TinyGPT.lpi
cd ../../bin/x86_64-linux/bin
stdbuf -oL -eL ./TinyGPT          # stdbuf keeps the streamed loss visible
```

The `stdbuf -oL -eL` wrapper is important: FPC block-buffers stdout when it is
redirected, so without line-buffering the streamed per-batch loss is lost.

## Final configuration

| Hyperparameter | Value |
|---|---|
| context window (SeqLen) | 24 |
| d_model | 64 |
| heads | 4 (d_k = 16) |
| d_ff | 64 |
| transformer blocks | 2 (causal) |
| vocab | 128 (char-level, one-hot) |
| optimizer LR / inertia | 0.01 / 0.9 |
| batch size | 32 |
| training volumes / epoch | 1024 |
| epochs | 8 (8192 training examples seen) |

This was tuned down from a larger run to fit the pure-CPU budget. On the
2-thread box used here, per-batch wall time is extremely noisy (~0.05s when the
box is idle to several seconds under external load; see the project memory note
on the NLP harness), so compare progress on the **examples-seen** axis, not
wall-clock. A clean run finishes in well under two minutes; longer/larger
configs train better but become unreliable when the shared box is loaded.

## Observed training trajectory

Training loss drops cleanly and accuracy climbs as the model starts to memorize
the corpus. Numbers below are from one clean end-to-end run (exact values vary
with the noisy RNG/scheduling):

```
 epoch 1   1024 examples   val Loss 4.49   val Accuracy 0.145   (first batch Loss 5.18)
 epoch 3   3072 examples   val Loss 4.09   val Accuracy 0.145
 epoch 5   5120 examples   val Loss 3.76   val Accuracy 0.145
 epoch 6   6144 examples   val Loss 3.49   val Accuracy 0.242
 epoch 8   8192 examples   val Loss 3.47   val Accuracy 0.290   (total time 1.28 min)
```

Start -> end: **next-char accuracy ~0.01 -> ~0.29**, **loss ~5.2 -> ~3.5** over
the run, dropping monotonically. The model is intentionally under-trained to fit
the budget; with more epochs (when the box is idle) the loss continues down past
~1.0 and the samples become near-verbatim corpus text — but 8 epochs already
shows the loss clearly and steadily falling, which is the headline.

## Generated sample

After training, the program autoregressively continues several seed prompts,
e.g. `"the quick"`, `"the lazy"`, `"in the for"`, sampling with `TNNetSamplerTopP(0.6)`,
plus one continuation with `TNNetSamplerMinP(0.15)` (min-p sampling: keep tokens
with `p >= MinP * max(p)`, renormalize, weighted draw — the candidate set adapts
to the model's confidence).
At the budgeted 8-epoch checkpoint the continuations are still rough but already
reproduce corpus character n-grams (`og`, `ox`, `fog`, `for`, `and`), e.g.:

```
  "the quick" -> the quick ax aaog ox anx
  "the lazy"  -> the lazy tog anx griog d
  "in the for"-> in the for ad fog agd xd
```

The model has clearly picked up the corpus's local structure (the fox/dog/forest
vocabulary fragments) rather than random characters. Training longer (more
epochs) sharpens these into whole corpus words. Output is stochastic via the
Top-P sampler, so run the binary to see your machine's exact sample.

## Honest caveats

- The corpus is intentionally tiny, so success is **memorization**, not
  generalization — the right and honest headline for a capstone demo.
- Char-level + a 5-minute CPU budget means the model is small; longer training or
  a bigger corpus would be needed for genuine language modeling.
- Per-batch loss and wall-time are noisy on a 2-thread machine; judge progress by
  the examples-seen trend, not any single batch.

---

Coded by Claude (AI). Sibling of [SimpleNLP](../SimpleNLP) (conv/transformer
char-LM trained from a file) — this is the smallest end-to-end *pure-transformer*
GPT demo built only from existing library blocks.
