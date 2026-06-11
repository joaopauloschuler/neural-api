# Sequence Packing for LM Pretraining

Demo for `TNNetSequencePacker` (`neural/neuraldatasets.pas`): packing multiple
short documents into each fixed-length context window instead of padding each
document — the GPT-2/GPT-3 pretraining recipe and a large throughput win on
pad-heavy corpora.

## What it shows

The same tiny causal transformer (token embedding -> 1 causal RoPE transformer
block -> per-position softmax head) is trained three times for the **same 400
optimizer steps** on the same corpus (56 templated five-word sentences; 8
sentences held out), differing only in how windows are built:

| feeding | mode | utilization | held-out perplexity |
|---|---|---|---|
| padded (baseline) | `pmOneDocPerWindow` | 33.3% | 5.41 |
| packed, no-split | `pmNoSplitGreedy` | 73.3% | 5.22 |
| packed, GPT-style | `pmSplitAcrossWindows` | **100%** | **4.35** |

Utilization = % of next-token target slots that are real (non-pad) tokens.
A training step costs the same wall-clock regardless of how much of the window
is padding, so higher utilization means more learning signal per step *and*
per second.

## Conventions

- pad = `0`, end-of-document separator = `1`, real tokens >= `2` (ids < 2 are
  special across the repo's NLP pipeline). Every document is followed by one
  separator (separator count = document count); the separator IS predicted.
- `pmSplitAcrossWindows` concatenates all documents into one stream and cuts
  it into consecutive windows (documents may split across boundaries; only the
  final partial window is padded). `pmNoSplitGreedy` never splits a document.
- Loss masking: only positions whose target is the pad token are excluded.
  `ApplyLossMask` copies the actual output into the desired output at masked
  positions; with the framework's `e = Output - Desired` error convention the
  masked error is exactly zero (verified to 0 in `tests/TestNeuralPacking.pas`).

## Notable gotchas reproduced here

- **Shuffle + re-pack every epoch.** With a fixed document order the packed
  stream is deterministic and the model memorizes cross-document order instead
  of the language (held-out PPL ~500 without shuffling).
- **Per-element norms.** `TNNetLayerNorm` normalizes over the whole sample
  including the sequence axis, so trailing pad rows at evaluation time would
  shift early rows — the blocks use `TNNetDyT` (per-element) instead.
- **Cross-document attention masking is intentionally absent**:
  `TNNetScaledDotProductAttention` only supports a static causal flag, not
  per-sample dynamic masks, so attention may cross document boundaries inside
  a packed window. This is standard GPT-2/GPT-3 behaviour and works fine; a
  per-sample block-diagonal mask is a noted follow-up in `tasklist.md`.

## Build & run

```
fpc -B -Fu../../neural -Fu$LAZUTILS_PATH -Mobjfpc -Sh -O2 SequencePacking.lpr
./SequencePacking
```

Pure CPU, finishes in about a second.
