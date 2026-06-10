# SlidingWindowBakeoff

Train the **same** tiny causal-attention next-token model **three times**,
differing only in the **masking scheme**, and chart final loss against the
**per-query key count** each arm must inspect. One `BuildNet(arm)` helper
assembles an identical embedding -> attention -> head stack and swaps only
the single mask layer, so the three runs (same seed, steps, learning rate
and data) are apples-to-apples.

The three arms (all causal — a query never sees the future):

- **(a) W=2** — sliding window of width 2 (`TNNetSlidingWindowMaskedFill(2)`):
  each query attends only to itself and the 1 key before it.
- **(b) W=4** — sliding window of width 4 (`TNNetSlidingWindowMaskedFill(4)`):
  each query attends to itself and the 3 keys before it.
- **(c) FULL** — full causal attention (`TNNetMaskedFill`): each query
  attends to **all** keys at or before it (count grows with position).

## The task (the answer lives inside a short window)

A causal **content-gated copy** rule that depends only on the last two
tokens:

```
if input[i] is EVEN -> target[i] = input[i]     # copy the current token
if input[i] is ODD  -> target[i] = input[i-1]   # copy the previous token
position 0          -> target[0] = input[0]     # no predecessor
```

Because the rule reads at most the current token and its immediate
predecessor, the necessary context is **fully contained in a width-2
window**. So both sliding arms (W=2 and W=4) already have everything they
need, and the FULL arm has the same information plus a lot of irrelevant
far-past keys. The point: the sliding window solves the task at the **same
quality** as full causal attention while inspecting **fewer keys per
query** — exactly the long-context cost saving the layer exists for.

It is a retrieval/copy task (which single-head attention handles well),
unlike an arithmetic rule a single linear-value head cannot compute.

Vocabulary 8, sequence length 12, embedding dim 32. Tiny enough that all
three arms finish in ~10-15 seconds on a single CPU thread, no external
dataset.

## Per-query key count (the cost knob)

- **FULL causal**: query at position `p` attends to `p+1` keys; the mean
  over a length-`L` sequence is `(L+1)/2` and the **last** query inspects
  all `L` keys. Cost grows with sequence length.
- **SLIDING W**: query at position `p` attends to `min(p+1, W)` keys;
  bounded by `W` no matter how long the sequence gets.

## The shared stack

```
TNNetInput(SeqLen, 1, 1)                  # token IDs along X
  -> TNNetEmbedding(Vocab, d_model)       # learned token vectors
  -> TNNetAddPositionalEmbedding          # fixed sin/cos position
  -> single-head CAUSAL attention (same wiring as the in-tree
     TNNet.AddSingleHeadSelfAttention helper):
        Q | K | V via three TNNetSplitChannels on a packed projection
        ValueT = TransposeXD(V)
        scores = DotProducts(Q, K) / sqrt(d_k)   # (key, 1, query)
        reshape -> (key, query, 1)
        -> MASK: TNNetSlidingWindowMaskedFill(W)  # arms W=2, W=4
               or TNNetMaskedFill                 # arm FULL
        reshape -> (key, 1, query) -> ReLUL -> softmax (depth axis)
        -> DotProducts(ValueT, W)                 # weighted sum of V
  -> TNNetPointwiseConvLinear(Vocab)      # per-position logits
  -> TNNetPointwiseSoftMax(1)             # softmax across depth
```

The only difference between arms is the single mask layer.

## Build & run

```
cd examples/SlidingWindowBakeoff
lazbuild SlidingWindowBakeoff.lpi --build-mode=Default
../../bin/x86_64-linux/bin/SlidingWindowBakeoff
```

Pure CPU, all three arms combined finish in ~10-15 seconds.

## Expected output (actual numbers from a run)

```
RESULTS: loss vs per-query key count (the long-context cost/quality trade)
  arm              W    train-CE      val-CE  val-acc  mean-keys  last-key
  --------------------------------------------------------------------------
  SLIDING W=2      2     0.00003     0.00003   100.0%       1.92         2
  SLIDING W=4      4     0.00004     0.00004   100.0%       3.50         4
  FULL causal   full     0.00003     0.00004   100.0%       6.50        12

GRADING (sliding window must not hurt — the answer is inside the window):
  [PASS] SLIDING W=2  val-CE=0.00003  within 0.05002 of FULL (0.00004)
  [PASS] SLIDING W=4  val-CE=0.00004  within 0.05002 of FULL (0.00004)

RESULT: PASS
```

## Teaching takeaway

Because the rule reads only the last two tokens, the **W=2** and **W=4**
sliding windows reach the **same loss / 100% accuracy** as full causal
attention while each query inspects **far fewer keys** (mean `1.92` vs
`6.50`, last query `2` vs `12`). When the relevant context is local, a
bounded sliding window buys the long-context cost saving for free — and
`TNNetSlidingWindowMaskedFill` collapses to plain `TNNetMaskedFill` once
`W >= SeqLen`, so it is a strict generalization of full causal masking.

The example is **graded**: each sliding arm must land within a tolerance of
the FULL-causal validation loss, and every arm must actually learn the task
(val-CE < 0.5); it prints `RESULT: PASS` when both hold.
