# DifferentialAttentionNoise

The headline **noise-cancellation** micro-experiment for
`TNNetDifferentialAttention` — the Differential Transformer (Ye et al.,
Microsoft 2024, *Differential Transformer*,
<https://arxiv.org/abs/2410.05258>).

## The idea

Softmax attention must spend a full unit of probability mass on the keys even
when none is relevant; on such an "all-keys-irrelevant" query position that
mass is pure noise spread over the keys. Differential attention computes
**two** independent softmax maps from two halves of the `Q|K` channels and
outputs their scaled **difference** applied to the shared `V`:

```
Attn_eff[j] = softmax1(Q1.K1 / sqrt(d_k/2))[j]
            - lambda * softmax2(Q2.K2 / sqrt(d_k/2))[j]
```

The two maps carry the **same common-mode attention noise**, so subtracting
them cancels it. `lambda` is a single learnable scalar initialised to the
paper's `lambda_init ~= 0.8`. On a fully-irrelevant probe row both softmaxes
are the same near-uniform noise, so the residual effective mass is
`~ (1 - lambda)` — well below SDPA's `1.0`.

`TNNetDifferentialAttention` exposes map 1 via the inherited
`AttentionWeights` property and map 2 via `AttentionWeights2` (both with
layout `X` = key `j`, `Y` = query `i`), and the current `lambda` via the
`Lambda` property. `d_k` must be **even** (it is split into two `d_k/2`
sub-heads).

## What this probe does

Builds a tiny **causal** next-token setup, `TNNetInput(SeqLen, 1, 3*d_k)`
(the depth axis packs `Q | K | V`). Every position except the last gets a
query strongly aligned with its own key (a clear real target). The **last**
position is the **probe row**: its query is ~orthogonal to every real key, so
all keys are irrelevant to it and the whole probe row is attention noise.

A forward pass runs through (a) plain `TNNetScaledDotProductAttention` and
(b) `TNNetDifferentialAttention`, both causal. We read the probe row's
post-softmax map(s) straight off the layers and print the noise mass on the
irrelevant keys:

- **plain SDPA**: the softmax row sum (`= 1.0` — all mass is noise);
- **differential (net)**: `sum_j (a1[j] - lambda*a2[j])`;
- **differential (abs)**: `sum_j |a1[j] - lambda*a2[j]|`.

```
TNNetInput(SeqLen, 1, 3*d_k)         # depth packs Q | K | V
  -> TNNetScaledDotProductAttention(d_k, causal)   # baseline
  -> TNNetDifferentialAttention(d_k, causal)        # two maps, subtracted
```

Forward-only, deterministic, single Compute per model — runs in well under a
second on one CPU thread, modest memory.

## Build & run

```
lazbuild DifferentialAttentionNoise.lpi
../../bin/x86_64-linux/bin/DifferentialAttentionNoise
```

## Observed output

```
TNNetDifferentialAttention noise-cancellation micro-experiment
  d_k = 8, SeqLen = 6 (causal)
  probe = last query row; its query is ~orthogonal to all keys
  (so EVERY real key is irrelevant -> the whole probe row is noise).

----------------------------------------------------------------
model                      probe noise mass
----------------------------------------------------------------
plain SDPA                         1.0000
differential (net)                 0.2000
differential (abs)                 0.2000
----------------------------------------------------------------
  lambda = 0.8000 (paper lambda_init)
```

## Reading the result

Plain SDPA spends the full unit of mass (`1.0000`) on the irrelevant keys —
it has no way to abstain. Differential attention leaves only `0.2000`
effective mass on those keys, **strictly below** SDPA's `1.0`. Because the
probe query is identical across the two sub-heads, both softmax maps produce
the same near-uniform noise, so `lambda` (here `0.8`) of it cancels exactly
and the residual is `1 - lambda = 0.2`. Net and absolute masses coincide here
(the effective weight stays non-negative on every key). This is the
Differential Transformer noise-cancellation claim, observed directly: the
second softmax map subtracts the common-mode attention noise the first map is
forced to emit.

The result uses the untrained `lambda_init` (`0.8`). Training would let
`lambda` adapt, but the inequality (`differential < SDPA`) already holds at
initialisation by construction.
