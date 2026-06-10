# Latent Attention

A tiny demonstration of **Multi-head Latent Attention (MLA)** — the attention
mechanism of **DeepSeek-V2** (Liu et al. 2024) — built with the
`TNNet.AddMultiHeadLatentAttention` builder: a three-arm next-token bake-off
(NoPE MLA, **decoupled-RoPE** MLA, and a plain MHA baseline) plus the headline
**KV-cache incremental-decode win**, demonstrated and measured two ways
(SDPA-cache machinery and a *true latent-only cache* of `d_c` floats/token).

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
NN.AddMultiHeadLatentAttention(d_model, Heads, LatentDim, CausalMask, RopeDim);
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

### The decoupled-RoPE slice (`RopeDim > 0`)

RoPE cannot be applied to the compressed latent `c_KV`: the per-head
up-projection would smear positions across channels. The paper's fix — carried
by the builder's optional `RopeDim` parameter (default `0` = NoPE, bit-for-bit
the original wiring) — keeps the content path position-free and adds a small
extra rope dimension:

- **rope-Q**: a *per-head* token-wise projection of `x` (width `Heads·RopeDim`),
  each head's `RopeDim` slice rotated by `TNNetRotaryEmbedding`;
- **rope-K**: **one** token-wise projection of `x` (width `RopeDim`) **shared
  by all heads**, rotated *once* — so the per-token decode state grows by only
  `RopeDim`, independent of head count.

Each head then attends with `concat(Q_h, ropeQ_h) · concat(K_h, ropeK)` over
width `d_k + RopeDim` (V is zero-padded to satisfy the SDPA equal-width
`Q|K|V` pack via an exact `x + (−x)` zero block whose gradients cancel; the
head output is sliced back to its `d_k` content channels).

## What this program shows

```
TNNetInput(16, 1, 1)                      # 16 token ids along X
  -> TNNetEmbedding(vocab=6, d_model=24)
  -> TNNetSinusoidalPositionalEmbedding   # absolute positions (NOT the RoPE arm)
  -> [ ATTENTION ]                        # MLA | MLA+RoPE | (QKV slab -> MHA)
  -> TNNetPointwiseConvLinear(32)         # per-token MLP hidden (d_ff)
  -> TNNetReLU
  -> TNNetPointwiseConvLinear(6)          # per-token vocab logits
  -> TNNetPointwiseSoftMax(1)
```

Three arms share the embedding front-end, MLP read-out, data and weight init;
only the attention block (and positional pathway) differs:

- **MLA:** `AddMultiHeadLatentAttention(d_model, heads, d_c, causal=true)` after
  `SinusoidalPositionalEmbedding` (NoPE scores, absolute input positions).
- **MLA+RoPE:** `AddMultiHeadLatentAttention(..., causal=true, RopeDim=4)` with
  **no** absolute positional embedding — position enters *only* through the
  decoupled rope slice, proving it carries position.
- **MHA:** `PointwiseConvLinear(3·d_model)` slab → `AddMultiHeadSelfAttention(heads, true)`.

The task is a fixed-offset **copy**: `target[t] = S[t-4]` for `t ≥ 4` (and
`S[t-1]` for the early positions), reachable only via positional information.

### KV-cache incremental decode (the headline win)

After the bake-off, a random-weight causal MLA stack decodes 24 tokens one at
a time, three ways, each checked against the full re-encode forward:

1. **(a) SDPA cache machinery, NoPE and `RopeDim=4`:** the per-head
   `TNNetScaledDotProductAttention` layers run in
   `BeginIncrementalDecode(MaxContext)` mode (commit `2dacc95`). For the RoPE
   arm every `TNNetRotaryEmbedding` gets `PositionOffset := t` before each step,
   so the streamed length-1 token is rotated with its **absolute** position.
   This is the O(1)-per-step compute path, but it caches post-up-projection
   per-head K/V — MHA-sized memory.
2. **(b) TRUE latent-only cache:** a decode loop whose *only* per-token state is
   the `c_KV` stream (`d_c` floats/token). Each step re-runs the K/V
   up-projections over the cached latents (O(t) recompute) and performs one
   query row of attention manually. This proves `d_c` floats/token is genuinely
   sufficient decode state.

### Sample output (seed 424242)

```
=== Next-token bake-off (lower CE is better; acc is argmax over vocab) ===
arm        params   init_CE  final_CE      acc  acc_long     sec
MLA          2832     1.858     0.000    1.000     1.000   11.27
MLA+RoPE     3312     1.792     0.166    0.920     0.971   12.22
MHA          3408     1.873     0.000    1.000     1.000   10.30

=== KV-cache incremental decode (the MLA headline win) ===
(a) SDPA cache machinery, NoPE:           max |full - cached| = 0.000000000
(a) SDPA cache machinery, RopeDim=4:      max |full - cached| = 0.000000000
(b) TRUE latent-only cache (d_c floats/token): max |full - latent| = 0.000000119

KV-cache memory per token (4-byte floats):
  MHA full K+V (2*d_model):                        48 floats =  192 bytes
  MLA via SDPA caches, arm (a) NoPE (2*d_model):   48 floats =  192 bytes
  MLA via SDPA caches, arm (a) RopeDim=4:          80 floats =  320 bytes
  MLA latent-only, arm (b) NoPE (d_c):              8 floats =   32 bytes
  MLA latent-only + shared rope-K (d_c+4):         12 floats =   48 bytes
Arm (b) stores 8 vs MHA's 48 floats/token: 16.7% of the MHA cache
```

The MLA+RoPE arm reaches 0.971 long-range copy accuracy with **no absolute
positional embedding** — the decoupled rope slice alone carries position. The
latent-only decode matches the full forward to ~1e-7 while storing 16.7 % of
the MHA cache (and `d_c + RopeDim = 12` floats/token with the rope slice, since
rope-K is one shared projection). Per-arm bake-off numbers are seed-dependent;
the point is the mechanism plus the cache arithmetic.

## Honest scope

- Arm (a) caches post-up-projection per-head K/V (the existing SDPA cache
  machinery), so its **memory** matches MHA; it demonstrates the O(1)-per-step
  compute path. Arm (b) is the **memory** win: latent-only state at `d_c`
  floats/token, paying an O(t) up-projection recompute per step. A production
  decoder would fold the up-projections into the attention math (the paper's
  absorbed-matrices trick) to get both at once.
- Per-arm bake-off numbers are seed-dependent.

## Build & run

```
fpc -Fu../../neural -Fu<lazutils> -Mobjfpc -Sh -O2 LatentAttention.lpr
./LatentAttention
```

Pure CPU, single-threaded, no external dataset; finishes in well under a minute.
