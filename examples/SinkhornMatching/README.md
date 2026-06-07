# SinkhornMatching — solving a linear-assignment problem with a soft permutation from `TNNetSinkhorn`

This example showcases **`TNNetSinkhorn`**, a differentiable **optimal-transport /
doubly-stochastic normalization** layer, on the canonical **bipartite MATCHING /
linear-assignment** task — distinct from the sorting demo in
[`../SinkhornSort/`](../SinkhornSort). A small network learns to **assign workers
to tasks at minimum cost** end-to-end, by emitting a soft **permutation matrix**
and being trained on the *continuous* optimal-transport cost of that soft
assignment (Mena et al. 2018, *"Learning Latent Permutations with Gumbel-Sinkhorn
Networks"*, arXiv:1802.08665).

## The problem

The **assignment problem**: given an `N × N` **cost matrix** `C` (`C[i,j]` = cost
of assigning worker `i` to task `j`), choose a **permutation** `pi` (one task per
worker, one worker per task) that **minimizes** the total cost
`sum_i C[i, pi(i)]`. Permutation matrices are exactly the 0/1 doubly-stochastic
matrices, and the **Sinkhorn–Knopp** projection maps any square score matrix onto
the doubly-stochastic polytope — their convex hull — sharpening towards a single
hard permutation as the temperature `tau → 0`:

```
L := score / tau
repeat KIter times:
  row-normalize:  L[i,j] -= logsumexp_j L[i,:]     (each row -> log-simplex)
  col-normalize:  L[i,j] -= logsumexp_i L[:,j]     (each col -> log-simplex)
Output := exp(L)
```

This is a **different use** of `TNNetSinkhorn` from the sort demo: there the target
is the fixed ascending sort of the input *values*; here it is the cost-minimizing
*matching* of a cost *matrix* — the optimal-transport / assignment task.

## The network

```
Input(N,1,N)  =  cost matrix C  (row i = worker i, depth j = task j)
   -> PointwiseConvReLU(N*N) -> PointwiseConvLinear(N)   (token-wise score head)
   -> TNNetSinkhorn(KIter, tau)  ->  P   (an N x N soft permutation)
```

The score head is **token-wise** (`PointwiseConvLinear` keeps the `(N,1,N)` row
axis; `FullConnect` would flatten/mix the rows).

## Training objective

Pure optimal-transport cost of the soft assignment:

```
L = sum_{i,j} P[i,j] · C[i,j]        dL/dP[i,j] = C[i,j]
```

The trivial gradient is set by hand on the Sinkhorn output and back-propagated
through the **entire unrolled Sinkhorn iteration** into the score head. The net is
**never told the optimal permutation** — only its own soft cost — yet learns to
produce assignments whose hard-decoded cost approaches the brute-force optimum.

## Annealing & evaluation

`tau` starts **warm** and is **annealed down** geometrically. On held-out test
cost matrices the example reports the **exact-match rate** (fraction solved to the
optimal permutation, decoded greedily from `P`) and the **mean normalized
optimality gap** (decoded cost − brute-force optimal cost, ÷ N) before and after
annealing.

## Headline result

Matching all 4 workers optimally by chance is `1/24 ≈ 4%`. A typical run on 2 CPU
cores (**~18 seconds**, modest memory, no committed binaries):

```
Exact-match BEFORE training (tau=1.000):   0.5%  mean optimality gap= 0.3165
  epoch     0  tau=1.0000  exact=  0.5%  gap= 0.3165
  epoch  1000  tau=0.4681  exact= 86.8%  gap= 0.0029
  epoch  2000  tau=0.2191  exact= 95.5%  gap= 0.0009
Exact-match AFTER annealing  (tau=0.150):  94.8%  mean optimality gap= 0.0015
```

Exact-match climbs from chance to **~95%** and the optimality gap shrinks by
**~200×** as `tau` cools — the net learns the **discrete** assignment problem
through a **continuous** Sinkhorn relaxation. (Cooling `tau` below the sweet spot
saturates the internal softmaxes and gradients vanish, so the schedule stops at a
small-but-finite final temperature; `TAU_LO = 0.15` is where exact-match peaks.)

## Build & run

```
lazbuild --bm=Release SinkhornMatching.lpi
../../bin/x86_64-linux/bin/SinkhornMatching
```
