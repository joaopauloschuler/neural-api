# Gated Linear Attention — data-dependent per-channel forget gate

Demonstrates `TNNetGatedLinearAttention` (GLA, Yang et al. 2023, "Gated Linear
Attention Transformers with Hardware-Efficient Training",
[arXiv:2312.06635](https://arxiv.org/abs/2312.06635)) on an **overwrite**
key→value recall task, head-to-head with the delta-rule layer `TNNetDeltaNet`
and a single-scalar fixed-decay baseline (`TNNetRetention`).

## The layer

`TNNetGatedLinearAttention` carries a `(d × d)` matrix memory `S` over the
sequence/time axis and updates it per timestep with a **data-dependent,
per-channel (vector) diagonal forget gate**:

```
α_t   = sigmoid(W_a x_t)                       (d forget gates, one per key channel)
S_t[d,e] = α_t[d] · S_{t-1}[d,e] + k_t[d] · v_t[e]   (gated rank-1 write)
y_t   = Sᵀ_t q_t                               (read-out)
```

Each key channel `d` decays *its own* slice of the memory by *its own*
input-dependent factor `α_t[d] ∈ (0,1)` before the new outer-product write. This
is what makes GLA distinct from its siblings in this library:

| layer | forget mechanism |
|---|---|
| `TNNetWKV` (RWKV) | **fixed-learned** per-channel exp decay (not input-dependent) |
| `TNNetRetention` | a single **scalar** γ |
| `TNNetMLSTMCell` | scalar exp gates + running-max stabilizer |
| `TNNetDeltaNet` | scalar **write** gate β (no multiplicative forget) |
| **`TNNetGatedLinearAttention`** | **data-dependent VECTOR forget gate on a 2-D state** |

The keys are L2-normalized (the DeltaNet idiom); the query is `1/√d`-scaled.
Forward is the exact left-to-right scan; backward is exact BPTT carrying the full
`dL/dS_t` (the per-channel gate row-scales the carry).

## The task (overwrite recall)

Each sequence presents several distinct `(key, value)` write tokens; one key is
then **re-written with a new value** (an overwrite); finally that key is queried
and the network must return its **most recent** value. GLA's per-channel gate can
selectively forget the channels carrying the stale value while retaining the
rest. All three arms share the same I/O contract and a comparable parameter
budget: a per-token `1×1` projection → memory mixer → `1×1` readout.

## Headline result

```
eval over 400 held-out recall sequences:
  GLA (per-channel gate) : recall MSE = 0.00851   exact-recall acc = 100.0%
  DeltaNet (delta rule)  : recall MSE = 0.00765   exact-recall acc = 100.0%
  Retention (fixed decay): recall MSE = 0.01912   exact-recall acc = 96.3%
```

GLA reaches perfect exact recall — competitive with the delta rule and clearly
ahead of the single-scalar fixed-decay baseline, which blends the stale and fresh
values and misses a fraction of overwrites.

## Running

```
lazbuild GatedLinearAttention.lpi
../../bin/x86_64-linux/bin/GatedLinearAttention
```

Pure CPU, tiny dimensions; finishes in a few seconds.

## Follow-ups

- Chunked/parallel (sub-quadratic, hardware-efficient) forward — the paper's main
  contribution on the systems side; v1 ships the exact per-token scan.
- An `AddGatedLinearAttention` builder composing token-shift + projections + the
  cell into a drop-in time-mixing block.
- A rectangular state `d_k ≠ d_v` variant (`FStruct[0]`/`FStruct[1]` already carry
  `d_k`/`d_v` for this).
