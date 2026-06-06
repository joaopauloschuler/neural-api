# Set Attention Block (SAB)

A small, fast demo of the `TNNet.AddSAB(InducingPoints, Heads, DFF)` builder --
the Set Attention Block of the Set Transformer (Lee et al. 2019,
[arXiv:1810.00825](https://arxiv.org/abs/1810.00825)).

The SAB wraps a Multihead Attention Block (MAB) in two post-norm residual
sublayers over a `(N, 1, d_model)` set:

```
H   = LayerNorm(X + MAB(X, X))      // self-attention over the set
out = LayerNorm(H + FFN(H))         // token-wise 2-layer MLP
```

In this fork the MAB is the multi-head ISAB block (`AddInducedSetAttention`):
`Heads` independent heads, each a per-token input projection feeding a
single-head `TNNetInducedSetAttention` with its own inducing bank, concatenated
along Depth and run through a per-token out-projection. Every op is a 1x1
(pointwise) conv over Depth except the symmetric softmax-over-inputs, so the
block is **permutation-equivariant**: shuffle the input rows and the output rows
follow the same permutation.

## What the demo shows

- **Part 1 -- permutation equivariance (structural).** The SAB stack is run on a
  random bag and a shuffled copy; the shuffled output rows match the
  correspondingly-permuted original output to `< 1e-5`, before any training.

- **Part 2 -- above-the-mean classification.** For each element of a bag, predict
  `+1` if its feature 0 is above the bag mean, else `-1`. This REQUIRES
  cross-element interaction (the mean depends on every element). A 2-block SAB
  stack learns it; a per-element MLP baseline (no cross-element path) is stuck
  near chance.

Pure CPU, single thread, tiny dimensions -- runs in well under a minute.

## Build & run

```
lazbuild examples/SetAttentionBlock/SetAttentionBlock.lpi
./bin/x86_64-linux/bin/SetAttentionBlock
```

(or open the `.lpi` in Lazarus and Run.)
