# Latent Attention

A tiny demonstration of **Multi-head Latent Attention (MLA)** — the attention
mechanism of **DeepSeek-V2** (Liu et al. 2024) — built with the
`TNNet.AddMultiHeadLatentAttention` builder, plus a head-to-head next-token
bake-off against a parameter-matched plain multi-head self-attention (MHA).

## What MLA is (and how it differs from MHA / GQA)

| Variant | KV cache idea | Cacheable state per token |
|---------|---------------|---------------------------|
| MHA | cache full-width K **and** V | `2·d_model` |
| GQA | **share** K/V across query-head groups | `2·KVHeads·d_k` |
| **MLA** | **low-rank-factor** the K/V projection | `d_c` (latent width) |

Plain MHA projects each token to a full-width Key and Value and — in an
incremental decoder — must cache both, so the KV cache grows with `d_model` and
head count. GQA shrinks that cache by *sharing* K/V across query-head groups.
MLA takes an orthogonal route: it **low-rank-factors** the K/V projection. Each
token is first **down-projected** to a single tiny shared latent

```
c_KV := x · W_DKV        (width d_c = LatentDim, with d_c << d_model)
```

and K and V for every head are then reconstructed by **up-projections**
`K := c_KV · W_UK`, `V := c_KV · W_UV`. In a decoder, `c_KV` (width `d_c`) is
the *only* per-token state that needs caching, so the cacheable state shrinks
from `2·d_model` (full K and V) down to just `d_c` — a **rank-based** saving of

```
d_c / (2·d_model)
```

that is independent of the number of heads. Query is projected from `d_model`
directly (per head), exactly as in MHA.

## The builder

```pascal
NN.AddMultiHeadLatentAttention(d_model, Heads, LatentDim, CausalMask);
```

over a `(SeqLen, 1, d_model)` token tensor (source is `GetLastLayer()`). It
performs its own token-wise projections, all `TNNetPointwiseConvLinear` (1×1
convs) so the sequence axis is preserved (a `TNNetFullConnect*` would flatten
and mix tokens):

- `Q := PointwiseConvLinear(d_model)` from the input (per head, `d_k = d_model/Heads`);
- `c_KV := PointwiseConvLinear(LatentDim)` from the input — the shared latent;
- `K := PointwiseConvLinear(d_model)` and `V := PointwiseConvLinear(d_model)`
  from `c_KV` (the up-projections);
- per head, `[Q_h | K_h | V_h]` (width `3·d_k`) is fed to one
  `TNNetScaledDotProductAttention`; head outputs are `TNNetDeepConcat`-ed and
  out-projected token-wise back to `d_model`.

## What this program shows

```
TNNetInput(16, 1, 1)                      # 16 token ids along X
  -> TNNetEmbedding(vocab=6, d_model=24)
  -> TNNetSinusoidalPositionalEmbedding   # parameter-free positions (both arms)
  -> [ ATTENTION ]                        # MLA  or  (QKV slab -> MHA)
  -> TNNetPointwiseConvLinear(32)         # per-token MLP hidden (d_ff)
  -> TNNetReLU
  -> TNNetPointwiseConvLinear(6)          # per-token vocab logits
  -> TNNetPointwiseSoftMax(1)
```

Two arms share an identical embedding front-end, MLP read-out, data and weight
init; only the attention block differs:

- **MLA:** `AddMultiHeadLatentAttention(d_model, heads, d_c, causal=true)`.
- **MHA:** `PointwiseConvLinear(3·d_model)` slab → `AddMultiHeadSelfAttention(d_model, heads, true)`.

The task is a fixed-offset **copy**: `target[t] = S[t-4]` for `t ≥ 4` (and
`S[t-1]` for the early positions), reachable by attention via the sinusoidal
positions. The program prints:

1. the cacheable-state saving `d_c/(2·d_model)`;
2. each arm's parameter count, final cross-entropy and next-token accuracy.

### Sample output (seed 424242)

```
=== MLA cacheable-state saving ===
Plain MHA caches full-width K AND V: 2*d_model = 48 floats per token.
MLA caches only the shared latent c_KV: d_c = 8 floats per token.
Cacheable-state saving d_c/(2*d_model) = 8/48 = 0.1667 (16.7% of the MHA cache).

=== Next-token bake-off ===
arm        params   init_CE  final_CE      acc  acc_long     sec
MLA          2832     1.858     0.000    1.000     1.000    8.46
MHA          3408     1.873     0.000    1.000     1.000    7.40
```

MLA solves the copy task with a smaller parameter count *and* advertises a
cacheable state of only 16.7 % of MHA's. Per-arm numbers are seed-dependent;
the point is the mechanism plus the cache arithmetic.

## Honest scope (v1)

- This is **training-time** behaviour. The KV-cache *win* only materialises with
  an incremental-decode path (still open in this repo); here we prove the shapes
  are correct and **report** the cacheable-state saving rather than measure cache
  bytes at decode time.
- v1 MLA is **NoPE** on the attention scores (the builder carries no positional
  info); the shared `TNNetSinusoidalPositionalEmbedding` front-end gives both
  arms identical positions. The paper's **decoupled RoPE** slice (a separate
  small RoPE-only channel group on Q and K) is **not yet implemented** — a
  documented follow-up.

## Build & run

```
fpc -Fu../../neural -Fu<lazutils> -Mobjfpc -Sh -O2 LatentAttention.lpr
./LatentAttention
```

Pure CPU, single-threaded, no external dataset; finishes in well under a minute.
