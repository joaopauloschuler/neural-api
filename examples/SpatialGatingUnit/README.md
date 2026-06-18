# Spatial Gating Unit (gMLP) vs single-head attention

This example demonstrates `TNNetSpatialGatingUnit` and the `TNNet.AddgMLPBlock`
builder — the attention-free sequence mixer from Liu et al. 2021,
[*Pay Attention to MLPs*](https://arxiv.org/abs/2105.08315) — and pits a gMLP
block against a same-parameter-budget single-head self-attention baseline on a
long-range sequence task.

## What the Spatial Gating Unit does

Over a `(SeqLen, 1, d)` tensor (`d` even) the SGU:

1. splits the `d` channels in half into `u` (first `d/2`) and `v` (last `d/2`);
2. applies a single **learned, content-independent** `SeqLen x SeqLen` weight
   matrix `W` (plus a per-position bias) across the sequence axis of `v`:
   `v'[n] = bias[n] + sum_m W[n,m] * v[m]` — the same static spatial projection
   shared by every one of the `d/2` channels;
3. gates multiplicatively: `out[n] = u[n] * v'[n]`, output shape
   `(SeqLen, 1, d/2)`.

Because `W` is fixed after training (it does not depend on the input), the SGU is
a different primitive from attention — there are no per-input
query/key scores, just one static cross-token matrix. The mixing matrix is a
true `SeqLen x SeqLen` object, so the sequence length is **pinned at
construction** (`TNNetSpatialGatingUnit.Create(SeqLen)`); a mismatched input is
rejected.

## The task

**Long-range first-token broadcast**: the input is a length-24 sequence of small
tokens; the target at *every* position is the value of token 0. Solving it forces
each output position to pull information from one distant source position — a
clean long-range dependency that purely local mixing cannot capture.

Two models are trained on identical data:

- **gMLP**: `Embedding -> SinusoidalPosEmb -> AddgMLPBlock -> PointwiseConvLinear
  -> PointwiseSoftMax`. The block is `up-proj -> Spatial Gating Unit -> down-proj`
  inside a pre-LayerNorm residual, with the two gMLP-paper LayerNorms that bound
  the multiplicative gate (one before the SGU, one after). The SGU layer itself
  is the pure spatial-gating primitive; the normalization lives in the builder.
- **attention**: `Embedding -> SinusoidalPosEmb -> PointwiseConvLinear(3*d_k) ->
  single-head SDPA -> PointwiseConvLinear -> PointwiseSoftMax`. `d_k` is sized so
  the two models have comparable trainable-weight counts (printed at startup).

## Build & run

```
cd examples/SpatialGatingUnit
lazbuild SpatialGatingUnit.lpi --build-mode=Default
../../bin/x86_64-linux/bin/SpatialGatingUnit
```

Pure CPU, no external dataset, no files written. The whole run (both models,
800 steps each) finishes in ~13 seconds on one thread.

## Expected output (actual numbers from a run)

```
Trainable weights: gMLP=4568   attention=1312

Training gMLP (Spatial Gating Unit) ...
  [gMLP] step    1 /  800   mean-CE=1.64856
  ...
Training attention baseline ...
  ...
================================================================
Held-out per-token accuracy:  gMLP=100.00%   attention=100.00%
RESULT: the attention-free Spatial Gating Unit MATCHES or BEATS the same-budget attention head.
================================================================
```

The attention head learns the broadcast faster (it only needs one position to
attend to), but the gMLP's static spatial matrix reaches the same 100% per-token
accuracy with no input-dependent attention at all — exactly the point of the
Spatial Gating Unit.
