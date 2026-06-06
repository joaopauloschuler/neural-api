# Set Transformer (ISAB + PMA)

A permutation-**invariant** set-learning demo built on the two Set-Transformer
primitives (Lee et al. 2019, *Set Transformer*, ICML / arXiv:1810.00825):

- **`TNNetInducedSetAttention` (ISAB)** — a set-to-set block that replaces the
  `O(N^2)` self-attention over `N` tokens with an `O(N*M)` bottleneck through `M`
  **learnable inducing points** `I`:
  - `H = MAB(I, X)` — the `M` inducing points attend over the `N` inputs → `(M,d)`
  - `Y = MAB(X, H)` — the `N` inputs attend back over the `M` summaries → `(N,d)`

  Output shape `== input (N,1,d)`. The largest score matrix is `N×M` (stage 2),
  not `N×N`. The inducing points are trainable Neurons (read **from** the layer),
  so this is **not** `TNNetCrossAttention` (two external sources).

- **`TNNetAttentionPooling` (PMA)** — a learnable, permutation-invariant pooler
  that collapses a variable-length set `(N,1,d)` to a **fixed** `(k,1,d)` by
  letting `k` **learnable seed vectors** `S` cross-attend over the inputs:
  `PMA = MAB(S, X)`. `k=1` is a single learned-query weighted-sum pool — unlike
  the parameter-free `TNNetAvgChannel` / `TNNetMaxChannel`.

`MAB(Q,KV)` here is a **single-head** scaled-dot-product cross-attention block:
`scores = Q·Kᵀ/√d`, softmax over keys, weighted sum of values. **v1 uses identity
Q/K/V projections** — the only trainable parameters are the inducing-point / seed
banks — which keeps the two-stage softmax-Jacobian backward exact and
numerically gradient-checkable. Multi-head MABs and per-MAB learnable
`W_Q/W_K/W_V` are documented follow-ups.

## What the demo shows

1. **Permutation invariance (structural).** A tiny `ISAB → PMA(k=1)` network is
   evaluated on a random bag and on a shuffled copy of the *same* bag. Because
   every softmax-over-inputs is symmetric in the input rows, the pooled output is
   identical to `< 1e-5` — **before any training**.

2. **Max-of-set regression.** Learning `f(X) = max_i X[i,0]`, `ISAB + PMA(k=1)`
   (which can place almost all of its softmax mass on the single largest element)
   beats a parameter-free **mean-pool** baseline (`TNNetAvgChannel`), which is
   forced to average and cannot single out the max.

3. **Score-matrix size.** ISAB routes `N` inputs through `M` inducing points
   (`N×M` scores) vs a full self-attention pool (`N×N`); the demo prints both to
   show the `O(N*M)` vs `O(N^2)` saving.

## Running

```
lazbuild SetTransformer.lpi
../../bin/x86_64-linux/bin/SetTransformer
```

Pure CPU, deterministic, runs in seconds.

### Example output

```
PART 1 -- permutation invariance
  ISAB->PMA output max abs diff (original vs shuffled bag): 0.000000011
  PASS: pooled output is invariant to input order (< 1e-5).

PART 2 -- max-of-set regression (MSE after training, lower = better)
  ISAB + PMA(k=1) MSE : 0.01387
  mean-pool baseline  : 0.04556
  ISAB+PMA beats the mean-pool baseline on max-of-set.

PART 3 -- score-matrix size: ISAB bottleneck vs full self-attention
  full self-attention pool over N inputs: N x N = 64 scores
  ISAB through M inducing points:         N x M = 16 scores
```
