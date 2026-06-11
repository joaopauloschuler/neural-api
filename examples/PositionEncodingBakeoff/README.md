# PositionEncodingBakeoff

Train the **same** tiny attention-based next-token model **five times**,
differing only in the position-encoding scheme, and print a side-by-side
comparison. One `BuildNet(scheme)` helper assembles an identical
embedding -> attention -> head stack and switches only the position
component, so the five runs (same seed, epochs, learning rate and data)
are apples-to-apples.

The five schemes:

- **(a) NONE** - attention only, no positional layer at all.
- **(b) SINUSOIDAL** - fixed sin/cos additive position embedding
  (`TNNetAddPositionalEmbedding`).
- **(c) RoPE** - rotary position embedding (`TNNetRotaryEmbedding`),
  applied to the embedded sequence before the Q/K/V projection.
- **(d) ALiBi** - attention with linear biases (`TNNetALiBi`): a
  per-distance bias added to the raw attention scores.
- **(e) T5 REL-POS** - T5 bucketed relative position bias
  (`TNNetT5RelPosBiasAttention`): a **learnable** scalar `b[bucket(j-i)]`
  added to the pre-softmax scores. This arm swaps the hand-rolled
  attention block for the equivalent in-tree layer (same SDPA core,
  verified to match, plus the trained bias table).

## The task (position matters)

A causal **predict-the-previous-token** model: the target at position `i`
is the input token at position `i-1` (the target at position 0 is a fixed
begin token, 0). Self-attention is permutation-invariant over the key
positions, so **without** position information the model literally cannot
tell which key is "the previous one" and must fail. This makes the
no-position arm the clear loser - the teaching point of the bake-off.

Vocabulary 8, sequence length 12, embedding dim 32 (even, as RoPE
requires). Tiny enough that all five runs finish in well under half a
minute on a single CPU thread, with no external dataset.

## The shared stack

```
TNNetInput(SeqLen, 1, 1)                  # token IDs along X
  -> TNNetEmbedding(Vocab, d_model)       # learned token vectors
  -> [ TNNetAddPositionalEmbedding ]      # scheme = SINUSOIDAL
  -> [ TNNetRotaryEmbedding        ]      # scheme = RoPE
  -> single-head CAUSAL attention (same wiring as the in-tree
     TNNet.AddSingleHeadSelfAttention helper):
        Q | K | V via three TNNetSplitChannels on a packed projection
        ValueT = TransposeXD(V)
        scores = DotProducts(Q, K) / sqrt(d_k)   # (key, 1, query)
        reshape -> (key, query, 1)
        -> [ TNNetALiBi ]                          # scheme = ALiBi
        -> TNNetMaskedFill                         # causal upper-triangle
        reshape -> (key, 1, query) -> ReLUL -> softmax (depth axis)
        -> DotProducts(ValueT, W)                  # weighted sum of V
  -> TNNetPointwiseConvLinear(Vocab)      # per-position logits
  -> TNNetPointwiseSoftMax(1)             # softmax across depth
```

The hand-rolled attention block was checked offline to reproduce
`TNNetScaledDotProductAttention` to floating-point exactness on the
non-causal case, and to train to ~0 cross-entropy on this task, so the
only thing that differs between arms is the position scheme.

## Build & run

```
cd examples/PositionEncodingBakeoff
lazbuild PositionEncodingBakeoff.lpi --build-mode=Default
../../bin/x86_64-linux/bin/PositionEncodingBakeoff
```

Pure CPU, all five arms combined finish in ~20-30 seconds.

## Expected output sketch (actual numbers from a run)

```
COMPARISON TABLE (final training cross-entropy, lower is better):
  NONE (no position)          final-CE=1.69953   sample-acc=5/12
  SINUSOIDAL (sin/cos add)    final-CE=0.00026   sample-acc=12/12
  RoPE (rotary)               final-CE=0.00018   sample-acc=12/12
  ALiBi (score bias)          final-CE=1.69751   sample-acc=5/12
  T5 REL-POS (learned bias)   final-CE=0.30360   sample-acc=11/12

SAMPLE PREDICTIONS (same probe sequence for every scheme):
  [SINUSOIDAL (sin/cos add)]
    INPUT     : 5 6 4 6 6 6 1 5 6 2 4 7
    PREDICTED : 0 5 6 4 6 6 6 1 5 6 2 4   <- exactly the previous token
    TARGET    : 0 5 6 4 6 6 6 1 5 6 2 4
  [NONE (no position)]
    INPUT     : 5 6 4 6 6 6 1 5 6 2 4 7
    PREDICTED : 5 5 5 4 4 4 6 6 5 6 0 6   <- guesses, cannot locate i-1
    TARGET    : 0 5 6 4 6 6 6 1 5 6 2 4
```

## Teaching takeaway

- **NONE is the worst** (highest final CE): attention alone is
  permutation-invariant over keys, so with no position signal it cannot
  pick out "the previous token".
- **SINUSOIDAL and RoPE win** (CE ~0, 12/12): they inject genuine
  positional structure into the token stream, which lets the attention
  address the key one step back and copy it.
- **ALiBi lands just above the no-position baseline** on this task. ALiBi
  only adds a per-distance bias to the attention *scores* and injects no
  positional content into the *values*; with a single head its slope is
  `2^-8`, a weak recency bias that under the causal mask actually prefers
  the query's own position. So it cannot perform precise fixed-offset
  (`-1`) retrieval. ALiBi is a relative *recency/locality* prior, not an
  absolute or fixed-offset addressing mechanism - the right inductive
  bias for "attend to nearby tokens", the wrong one for "fetch exactly
  the token one step back". This is itself the instructive result of the
  bake-off.
- **T5 REL-POS nearly wins (11/12)** despite ALSO being a score-only
  bias: because its per-distance bias is *learned*, the head simply puts
  a large bias on the distance-1 bucket and concentrates the softmax on
  the previous token. Only the special position-0 begin token stays out
  of reach (score-bias schemes put no positional content into the
  *values*, so position 0 cannot signal "I am first"), which is exactly
  the residual CE the table shows. Learned relative bias = precise
  offset addressing; fixed slope = recency prior only.
