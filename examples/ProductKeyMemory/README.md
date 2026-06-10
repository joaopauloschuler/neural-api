# Product-Key Memory

A tiny synthetic **key &rarr; value retrieval** demo for the
`TNNetProductKeyMemory` layer, following Lample et al., NeurIPS 2019,
*"Large Memory Layers with Product Keys"*
([arXiv:1907.05242](https://arxiv.org/abs/1907.05242)).

## The idea

A flat memory of `|K|` slots addressed by a full softmax costs `O(|K|)` per
query. **Product-key memory** factorizes the `|K|` keys into the Cartesian
product of **two small half-key banks** `K1`, `K2`, each of
`sqrt(|K|)` keys with half the query dimension. The effective key set is
`K1 x K2` of size `sqrt(|K|)^2 = |K|`.

For each query:

1. Split the query into two half-queries `q1`, `q2`.
2. Score `q1` against `K1` and `q2` against `K2` &mdash; each `O(sqrt(|K|))`.
3. Take the **top-`TopK`** per half, form the `TopK x TopK` candidate
   combinations, re-score each as `s1[a] + s2[b]`, and pick the **global
   top-`TopK`** product keys.
4. Softmax over the selected scores and use the weights to gate a **sparse**
   weighted sum over the corresponding learned **value** rows (an
   EmbeddingBag-style lookup) &rarr; the output.

So a memory of `|K|` slots is addressed in `O(sqrt(|K|))` work while touching
only `TopK` value rows per query.

## What this demo does

It builds `NumPairs` random `(query, target-value)` associations and trains
two models on the **identical** data:

| Model | Layer | Value rows read per query |
|---|---|---|
| **Product-key memory** | `TNNet.AddProductKeyMemory(NumKeys, ValueDim, TopK, 1)` | `TopK` (sparse) |
| **Flat-memory baseline** | one dense softmax readout over all `NumKeys` keys (`AddModernHopfieldRetrieval` + linear) | `NumKeys` (dense) |

It reports the final mean-squared retrieval error for both and a per-slot
read-count histogram for the product-key memory.

### Headline result (fixed seed)

```
  [product-key] final MSE = 0.000000
  [flat-memory ] final MSE = 0.000019

Per-query value rows TOUCHED:
  product-key : 4   (sparse)
  flat-memory : 64  (dense)

Key-usage spread (product-key memory):
  distinct slots used = 49 / 64
  busiest slot read in 6 / 24 queries
  Usage is reasonably spread across the memory.
```

The product-key memory **matches** the flat baseline's retrieval accuracy
while touching only `TopK = 4` rows per query instead of all `64` &mdash; the
whole point of the factorization.

Pure CPU, single-threaded, tiny dims, runs in a few seconds.

## Known failure mode: key-usage collapse

The classic pathology of product-key (and any learned-routing) memory is
**key-usage collapse**: a handful of slots hog all the reads. Every query
routes to the same few product keys, so most of the memory's capacity is
wasted and gradients only ever reach the over-used rows (which then stay
ahead in a self-reinforcing loop).

The demo prints a read-count histogram so collapse is visible
(`distinct slots used` and `busiest slot`). On this small, well-separated
synthetic task usage stays spread; on larger tasks it does not.

**The paper's fix:** put a **batch-norm on the query** *before* scoring it
against the half-keys. Batch-norm re-centres and de-correlates the query
distribution across the batch, so no slot is permanently favoured by a
biased query mean; this empirically spreads reads across the whole memory
and is what makes product-key memory usable at the million-slot scale. It is
described here but **not applied** in this small demo (the synthetic task is
small enough to avoid severe collapse). Wiring a `TNNetMovingScale` /
batch-norm in front of `AddProductKeyMemory` is the natural extension.

## Build & run

With Lazarus:

```
lazbuild ProductKeyMemory.lpi
```

Or directly with FPC:

```
fpc -O3 -Mobjfpc -Sh -dRelease -dAVX2 -Fu../../neural ProductKeyMemory.lpr
./ProductKeyMemory
```
