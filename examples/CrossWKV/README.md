# CrossWKV — two-source RWKV WKV external-memory recall

Demonstrates `TNNetCrossWKV`, a **two-source** variant of the RWKV-4
weighted-key-value (WKV) time-mixing recurrence (Peng et al. 2023, "RWKV:
Reinventing RNNs for the Transformer Era",
[arXiv:2305.13048](https://arxiv.org/abs/2305.13048)), on a cross-copy task that
the single-source `TNNetWKV` **cannot** express.

## The layer

`TNNetWKV` splits its own input channels into the `key|value` pair that drives
its running numerator/denominator WKV state — so the memory it accumulates and
the stream that reads it are **one** sequence. `TNNetCrossWKV` reads the
`key|value` stream from a **separate source** than the receptance/query stream,
exactly as `TNNetCrossAttention` generalises self-attention's packed `Q|K|V` to
two sources. Per channel `d`, per timestep `t` (`k_t,v_t` from the key|value
source B, `r_t` from the receptance source A):

```
wkv_t = ( a_{t-1} + e^{u+k_t} v_t ) / ( b_{t-1} + e^{u+k_t} )
a_t   = e^{-w} a_{t-1} + e^{k_t} v_t
b_t   = e^{-w} b_{t-1} + e^{k_t}
y_t   = sigmoid(r_t) · wkv_t
```

The decay/bonus core is the **exact** log-space-stable RWKV-v4 kernel of
`TNNetWKV` (per-channel positive `w = softplus(w_raw)`, per-channel bonus `u`,
running-max stabiliser). The only additions are the second source and a sigmoid
receptance gate read from source A. The key|value source layer index is
serialized like `TNNetConcat` / `TNNetCrossAttention`, so the layer round-trips
through `SaveToString` / `LoadFromString`.

### Sequence-length contract (v1)

This v1 requires **equal sequence length** on both sources. The read-out at time
`t` uses the WKV state accumulated over the key|value source up to `t` (with the
current-token bonus from that source's `k_t`), gated by the receptance from the
query source at the same index `t`. The asymmetric / full-context variant
(summarise the memory once, query it with a different-length stream) is a
documented follow-up, not implemented here.

## The task (cross-copy)

A **memory** sequence carries a stream of random value vectors (`position
one-hot | value`); a **separate query** sequence carries a per-position read
pulse. The read-out at position `t` must reproduce the memory value at position
`t` — a value that lives **only** in the memory tensor, never in the query
tensor. The value stream is re-randomised every sequence, so the answer cannot
be memorised: it must be read out of the cross-built WKV memory.

The **contrast arm** is a single-source `TNNetWKV` that can only see the query
stream; with no access to the memory tensor it is structurally blind to the
stored values and stays at chance.

## Headline result

```
eval over 400 held-out cross-recall sequences:
  CrossWKV (two-source)  : recall MSE = 0.00486   exact-recall acc = 100.0%
  blind WKV (query-only) : recall MSE = 0.17761   exact-recall acc = 17.0%
  chance accuracy        : 16.7%
```

The two-source `TNNetCrossWKV` recalls values out of the cross-built memory with
perfect exact recall; the memory-blind single-source `TNNetWKV` cannot beat
chance — the capability the two-source layer adds.

## Running

```
lazbuild CrossWKV.lpi
../../bin/x86_64-linux/bin/CrossWKV
```

Pure CPU, tiny dimensions; finishes in well under a minute on 2 cores.
