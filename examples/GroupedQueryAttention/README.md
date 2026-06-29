# GroupedQueryAttention — sharing K/V heads across query heads

Demonstrates `TNNet.AddMultiHeadGroupedQueryAttention`, **Grouped-Query
Attention** (Ainslie et al. 2023, *"GQA: Training Generalized Multi-Query
Transformer Models from Multi-Head Checkpoints"*,
[arXiv:2305.13245](https://arxiv.org/abs/2305.13245)), and runs it head-to-head
in **three configurations** that differ *only* in how many key/value heads the
query heads share.

## The idea

In plain multi-head attention every query head owns a private key/value head. GQA
lets **several query heads share one K/V head**, shrinking the K and V
projections from `QueryHeads · d_k` to `KVHeads · d_k` output channels — a factor
`QueryHeads / KVHeads` fewer K/V projection parameters (and, in a KV-cached
decoder, proportionally less cache state per shared projection). The example
spans the full range with a fixed `QueryHeads = 4`:

| arm | `KVHeads` | meaning |
|-----|-----------|---------|
| MHA | `4` (= `QueryHeads`) | every query head has its own K/V head — plain multi-head attention (the degenerate full case) |
| GQA | `2` | pairs of query heads share one K/V head |
| MQA | `1` | all query heads share a single K/V head — Multi-Query Attention (Shazeer 2019, [arXiv:1911.02150](https://arxiv.org/abs/1911.02150)) |

The same builder, `AddMultiHeadGroupedQueryAttention(QueryHeads, KVHeads,
CausalMask=false)`, produces all three — only the `KVHeads` argument changes.

## The task (content-based recall, position-free on purpose)

Each sequence presents `cNumPairs = 4` WRITE tokens
`[key_onehot | value_vec | flag=0]` for distinct keys with random value vectors,
then one QUERY token that re-presents one written key with `flag=1`. The target
at the query position is that key's stored value vector. This is pure content
addressing — exactly what softmax attention does — so **no positional encoding is
needed**, and the comparison isolates the K/V head sharing.

A fixed `ValueBank` of `cNumVals = 6` value vectors (each `cValueDim = 4`-wide)
supplies the discrete "contents" the network must recall. `Evaluate` reports both
the recall MSE at the query position and an **exact-recall accuracy** via
nearest-neighbour decode over the value bank.

## The model

Each arm is a tiny three-layer net; the only difference between arms is
`KVHeads`:

```
Input(SeqLen, 1, cInDim)
 -> PointwiseConvLinear(cModelDim)              token-wise projection to d_model
 -> AddMultiHeadGroupedQueryAttention(QueryHeads=4, KVHeads, CausalMask=false)
 -> PointwiseConvLinear(cValueDim)              token-wise readout to value dim
```

All projections are token-wise `TNNetPointwiseConvLinear` so they preserve the
sequence axis (`TNNetFullConnect` would flatten/mix the whole sequence). The
input is built by `TNNetInput.Create(cSeqLen, 1, cInDim)` where
`cInDim = cNumKeys + cValueDim + 1` (`key one-hot | value | query flag`).
Dimensions: `d_model = 16`, `QueryHeads = 4`, `SeqLen = 5` (`cNumPairs + 1`).

All three arms are trained on the **same** sample stream (`RandSeed := 999`
replayed before each `Train`, 20000 steps, `lr = 0.03`, momentum 0.9) via direct
`Compute` / `Backpropagate`, then evaluated on the same `RandSeed := 7` held-out
stream of 400 sequences.

## Running

```
cd examples/GroupedQueryAttention
fpc -O3 -Mobjfpc -Sh -Fu../../neural GroupedQueryAttention.lpr
./GroupedQueryAttention
```

(or open `GroupedQueryAttention.lpi` in Lazarus). Pure CPU, tiny dims, finishes
in well under a minute.

## What it prints

A header echoing the configuration, then the per-arm parameter counts
(`CountWeights`) — making the K/V projection shrink explicit:

```
=== Grouped-Query Attention: sharing K/V heads across query heads ===
...
MHA (KVHeads=4) params = ...
GQA (KVHeads=2) params = ...
MQA (KVHeads=1) params = ...
(the difference is exactly the K and V projection shrink: 2 * (QueryHeads-KVHeads) * d_k * d_model weights)
```

followed by the held-out recall table (recall MSE and exact-recall accuracy for
each arm) and the headline:

```
Headline: recall quality stays competitive as KVHeads shrinks, while
the K/V projection parameter count drops by QueryHeads/KVHeads.
```

The point — the GQA paper's result reproduced on a toy — is the **contrast**:
recall stays competitive as `KVHeads` drops from 4 to 1 while the K/V projection
parameter count shrinks. Exact numbers are seed-dependent.

Coded by Claude (AI).
