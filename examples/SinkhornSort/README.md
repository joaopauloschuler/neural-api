# SinkhornSort — learning to sort through a doubly-stochastic relaxation with `TNNetSinkhorn`

This example showcases **`TNNetSinkhorn`**, a differentiable **optimal-transport /
doubly-stochastic normalization** layer, by training a small network to **sort a
short list of scalars end-to-end** — a *discrete* operation learned purely through
its *continuous* relaxation (Mena et al. 2018, *"Learning Latent Permutations with
Gumbel-Sinkhorn Networks"*, arXiv:1802.08665; Adams & Zemel 2011).

## The idea

Where `TNNetSoftMax` / sparsemax / entmax normalize **one** axis, `TNNetSinkhorn`
normalizes a square `N × N` score matrix to be **doubly stochastic** (every row
**and** every column sums to 1) by iterating the **Sinkhorn–Knopp** algorithm in
log-space:

```
L := score / tau
repeat KIter times:
  row-normalize:  L[i,j] -= logsumexp_j L[i,:]     (each row  -> log-simplex)
  col-normalize:  L[i,j] -= logsumexp_i L[:,j]     (each col  -> log-simplex)
Output := exp(L)
```

The doubly-stochastic matrices are the convex hull of **permutation matrices**, and
as the temperature `tau → 0` the output sharpens towards a single **hard
permutation**. So a permutation — the thing a sort produces — becomes a smooth,
fully differentiable function of a score matrix.

## The network

```
Input(N) -> FullConnectReLU(N*N) -> FullConnectLinear(N*N) -> Reshape(N,1,N)
         -> TNNetSinkhorn(KIter, tau)  ->  P   (an N x N soft permutation)
```

The soft permutation `P` is applied to the input vector,
`yhat[i] = sum_j P[i,j]·x[j]`, and trained with plain **MSE** against the ascending
sort of `x`. The loss gradient w.r.t. `P`,
`dL/dP[i,j] = (yhat[i] − sorted[i])·x[j]`, is set by hand on the Sinkhorn output and
back-propagated through the **entire unrolled Sinkhorn iteration** (each step is a
differentiable subtract-logsumexp; the layer caches every step's input so its
backward replays the exact softmax-style adjoint).

## Annealing

`tau` starts **warm** (smooth, easy gradients, blurry `P`) and is **annealed down**
geometrically across training, sharpening `P` towards a true permutation. The
example reports **exact-sort accuracy** (round `P` to a hard permutation by greedy
arg-max, then check the result is the exact ascending sort) before and after
annealing.

## Headline result

Sorting **all 5** elements exactly is demanding (1/120 ≈ 0.8% by chance). A typical
run on 2 CPU cores (under ~1 minute, modest memory, no committed binaries):

```
Exact-sort accuracy BEFORE training (tau=1.000):  2.5%
  ... tau annealed 1.00 -> 0.07 ...
Exact-sort accuracy AFTER annealing  (tau=0.070): 62.5%
```

Accuracy climbs **monotonically** as `tau` cools — the net learns the **discrete**
argsort operation through a **continuous** Sinkhorn relaxation. (Cooling `tau`
below the sweet spot saturates the internal softmaxes and gradients vanish, so the
schedule stops at a small-but-finite final temperature.)

## Build & run

```
lazbuild --bm=Release SinkhornSort.lpi
../../bin/x86_64-linux/bin/SinkhornSort
```
