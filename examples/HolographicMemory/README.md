# Holographic Memory (HRR cleanup memory)

A cleanup-memory demo of **`TNNetHolographicBinding`**, the Holographic Reduced
Representation (HRR) vector-symbolic *binding* layer — the associative-binding
sibling of the exotic-algebra family (Complex / Quaternion / Octonion /
Tropical / Hyperbolic).

## The layer

`TNNetHolographicBinding` reads two equal-length Depth vectors `a` and `b`
packed as adjacent halves of the input (input Depth must be even `= 2n`; first
`n` channels are `a`, last `n` are `b`, the same packing idiom
`TNNetComplexLinear` uses for adjacent Re/Im pairs) and outputs an `n`-vector:

- **Bind** (`Unbind=0`, default): the circular **convolution**
  `c = a ⊛ b`, `c[k] = Σ_j a[j]·b[(k-j) mod n]`. This composes a role–filler
  pair into a new vector that is dissimilar to both operands, so many pairs can
  be **superposed** (simply added) into one fixed-width trace.
- **Unbind** (`Unbind=1`): the circular **correlation**
  `c = a ⊛ involution(b)`, `c[k] = Σ_j a[j]·b[(j-k) mod n]`, the approximate
  inverse used to query a bound trace.

The layer has **no trainable weights**; the forward is a direct `O(n²)` cyclic
sum and the backward is the exact bilinear adjoint. The `Unbind` flag round-trips
on save/load.

## What this example does

1. Samples a codebook of random unit `key`/`value` atoms (dimension `n=256`).
2. For a growing number `P` of stored pairs, binds `P` `key→value` pairs into
   **one** superposed trace `t = Σ_i key_i ⊛ value_i`, then for each stored key
   queries the trace with the unbind op and snaps the noisy result to the nearest
   codebook value (a "cleanup memory" by cosine similarity).
3. Prints the **HRR capacity curve**: recall accuracy vs. number of superposed
   pairs.

Querying a trace recovers the bound value via `unbind(a = trace, b = key)`
= `trace ⊛ involution(key)` = `involution(key) ⊛ trace`, the standard HRR
inverse.

## Running

```
lazbuild HolographicMemory.lpi
../../bin/x86_64-linux/bin/HolographicMemory
```

Pure CPU, no training (binding is weightless) — runs in a couple of seconds and
a few MB.

## Expected output

Recall is ~100% while the trace is lightly loaded and degrades **gracefully**
(not catastrophically) as more pairs are crammed into the single fixed-width
trace — the hallmark HRR capacity tradeoff:

```
   pairs P |  recall accuracy
   --------+-----------------
         1 |   100.00%
         ...
        10 |    99.75%
        ...
        24 |    88.54%
```

The single-pair sanity check confirms the algebra: the bound trace is
dissimilar to its own filler (`cos ≈ 0`), yet unbinding by the correct key
returns an approximate copy of the value (`cos ≈ 0.72`) whose nearest codebook
entry is the right one.
