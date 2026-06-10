# SequenceReverse

A tiny self-attention model that learns to output the **reverse** of its input
token sequence:

```
input  [a, b, c, d, e, f, g, h]
target [h, g, f, e, d, c, b, a]
```

This is the classic **copy/reverse** probe: a clean teaching demo that
self-attention can learn a **pure positional permutation**. Reversal is the fixed
map `output[i] = input[SeqLen-1-i]` — it depends only on POSITION, not on token
values. The winning strategy for the attention head is "route query position `i`
to key position `SeqLen-1-i`, and copy that token's content forward".

## Architecture (composed from existing layers — no new layer)

```
Input(SeqLen, 1, 1)                       token ids on the X axis
 -> Embedding(Vocab, d_model)             value content per token
 -> SinusoidalPositionalEmbedding         parameter-free position signal
 == one NON-causal self-attention block with a residual ====================
 -> PointwiseConvLinear(3*d_k)            per-token Q|K|V  (NOT FullConnect!)
 -> ScaledDotProductAttention(d_k)        full (non-causal) attention
 -> PointwiseConvLinear(d_model)          project head output to d_model
 -> Sum([resid_in, block_out])            residual connection
 == per-position readout ===================================================
 -> PointwiseConvLinear(Vocab)            per-position vocab logits
 -> PointwiseSoftMax(1)                   softmax across depth, per position
```

Two wiring rules make this work:

* **Per-token projections must be `TNNetPointwiseConvLinear` (1x1 conv).** A
  `TNNetFullConnect` would flatten/mix the whole `SeqLen*d_model` tensor into a
  single vector, collapsing the token axis and destroying the per-position
  structure this task needs.
* **A positional signal is essential.** Without positions every token looks the
  same to the head and reversal (a position permutation) is unlearnable. Here a
  parameter-free `TNNetSinusoidalPositionalEmbedding` supplies it.

Training is a manual single-threaded `Compute`/`Backpropagate` loop with
per-position cross-entropy and a per-position softmax head (the same idiom as
`examples/InductionHeads`). Random sequences are drawn on the fly; the target is
the reversed sequence.

## Build & run

```
cd examples/SequenceReverse
lazbuild --build-mode=Release SequenceReverse.lpi
cd ../../bin/x86_64-linux/bin && ./SequenceReverse
```

## Results (honest)

Everything is tiny (vocab 12, seqlen 8, d_model 24, one head, ~2.9k weights), so
the demo trains and self-reports in **about 2-3 seconds** on CPU — far under the
5-minute budget — and never exhausts memory.

Reversal is a fixed permutation, so a correctly wired attention model learns it
essentially perfectly. A representative run reaches:

```
Chance per-token accuracy (1/vocab) :   8.33%
Per-token accuracy                  : 100.00%
Sequence-level (exact reversal) acc : 100.00%
```

with mean cross-entropy collapsing from ~2.58 to <0.001 by epoch 100. The program
also prints one concrete worked example (input vs predicted vs target) so the
result is easy to read off by hand.
