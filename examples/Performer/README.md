# Performer / FAVOR+ — softmax attention via positive random features

Demonstrates `TNNetPerformerAttention`, **Performer self-attention**
(Choromanski et al. 2020, "Rethinking Attention with Performers",
[arXiv:2009.14794](https://arxiv.org/abs/2009.14794)) — the FAVOR+ mechanism
that gives an **unbiased, low-variance** estimate of the softmax attention kernel
at **linear** cost.

## The layer

Performer keeps the *softmax* kernel `exp(q·k)` but never materializes the
`SeqLen × SeqLen` score matrix. It maps queries and keys through **positive
random features** for an `m × d_k` **frozen** (non-trainable) random projection
`W`:

```
φ(x) = exp(W·x − ‖x‖²/2) / √m            (m positive features)
```

chosen so that `E[ φ(q)·φ(k) ] = exp(q·k)` — i.e. the random features are an
*unbiased estimator of the softmax numerator kernel*. Attention then reassociates
exactly like the kernel linear-attention family, but in the `m`-dimensional
feature space:

```
q'_t = φ(Q_t),  k'_s = φ(K_s)            (each length m)
S = Σ_s k'_s ⊗ V_s     (m × d_v)
Z = Σ_s k'_s           (m,)
Out_t = (q'_t · S) / (q'_t · Z)
```

so attention costs **O(SeqLen · m · d_v)** — linear in the sequence length, with
no quadratic score matrix.

The rows of `W` are drawn i.i.d. `N(0,1)` and, when `m ≥ d_k`, **orthogonalized**
block by block (Gram–Schmidt, then rescaled to a chi-distributed norm). Orthogonal
random features further reduce the estimator variance — the "+" in FAVOR+.

`W` is **frozen**: it receives no weight gradient. The input gradients `dL/dQ`,
`dL/dK`, `dL/dV` *are* backpropagated through `φ` (which depends on `x` via both
`W·x` and the `−‖x‖²/2` term); the exact input gradient is finite-difference
checked in the test suite. `d_k`, `m` and the RNG seed are stored in `FStruct[]`
so the frozen `W` reloads **bit-identically** through `SaveToString` /
`LoadFromString` (also verified by a round-trip test).

### How it differs from the other linear-attention layers

| Layer | Feature map | Approximates softmax? |
|---|---|---|
| `TNNetLinearAttention` | `φ(x)=elu(x)+1` (deterministic) | no — a *different* kernel |
| `TNNetLinformerAttention` | keeps softmax, low-rank-projects the sequence axis | yes, low-rank assumption |
| **`TNNetPerformerAttention`** | **positive random features** `exp(W·x−‖x‖²/2)` | **yes — unbiased softmax estimate** |

Performer is the only one of these that provides an *unbiased estimate of the
exact softmax kernel*.

## Part 1 — the approximation shrinks as `m` grows

For one fixed random sequence the demo computes the exact full-softmax
`TNNetScaledDotProductAttention` output, then the Performer output at increasing
`m`, averaged over many random projection seeds, and reports the mean RMS error.
(Because SDPA scales scores by `1/√d_k`, the demo pre-scales `Q` and `K` by
`d_k^{−1/4}` so both arms use the *same* kernel.)

```
   m   |  mean RMS error vs SDPA
  -----+------------------------
     4 |             0.118361
     8 |             0.095336
    16 |             0.097153
    32 |             0.077522
    64 |             0.069738
   128 |             0.060198
```

More random features → tighter softmax approximation: the headline FAVOR+ claim,
reproduced. The residual floor is the finite-`m` Monte-Carlo variance.

## Part 2 — it trains as a drop-in attention layer

A tiny **majority-value** task: a sequence of one-hot value tokens; the target read
out at a dedicated query position is the most frequent value class. The arm is
`1×1` projection (→ `3·d_k` = `Q|K|V`) → Performer attention → `1×1` readout →
per-position softmax.

```
eval over 600 held-out sequences (chance = 20.0%):
  Performer : majority-class accuracy = 55.7%
```

Performer learns the task well above chance using linear-cost random-feature
attention.

## Running

```
lazbuild Performer.lpi
../../bin/x86_64-linux/bin/Performer
```

Pure CPU, tiny dimensions; finishes in well under a minute.
