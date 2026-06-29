# RWKV — WKV time-mixing vs. the delta rule on overwrite recall

Demonstrates **`TNNetWKV`**, the RWKV-4 time-mixing weighted-key-value
recurrence, wired by the **`TNNet.AddRWKVTimeMix`** builder, and contrasts it
head-to-head with the delta-rule linear-attention layer **`TNNetDeltaNet`** on
the *same* associative-recall task.

## The layer

`AddRWKVTimeMix` assembles a complete RWKV time-mix block:

```
TokenShift -> {r, k, v} pointwise projections
           -> TNNetWKV (EMA numerator/denominator recurrence)
           -> receptance gate
           -> output projection
```

The core is `TNNetWKV`: a softmax-free, linear-time "attention". Every past
token receives an **exponentially-decaying** weight controlled by a learnable
per-channel decay `w`, plus a per-token **bonus** `u` for the current position,
and the whole sum is **self-normalised** by its running denominator. Conceptually
WKV maintains running numerator/denominator EMAs of the value stream keyed by
`exp(k)`, so it is an EMA-style content mixer rather than an explicit memory.

`TNNetDeltaNet`, by contrast, maintains a `(d, d)` **matrix memory** and applies
the delta (Widrow-Hoff) rule: it writes back only the read-then-corrected
residual gated by a learnable write strength `beta`, so it cleanly **overwrites**
an existing association instead of blending it into a decaying average.

Both are linear-time recurrences and both share the same `(SeqLen, 1, Depth)`
sequence contract.

## The task (overwrite recall)

Each sequence presents a series of WRITE tokens, then a QUERY. One key is written
**twice with two different values** (an overwrite); the QUERY re-presents that key
and the target is its **most recent** value:

```
write t :  [ key_onehot(k_t) | value_vec | flag=0 ]
query   :  [ key_onehot(k_q) | 0...0     | flag=1 ]
```

`cNumPairs` distinct keys are written with random value-ids drawn from a fixed
sinusoidal `ValueBank`; one of them is then re-written with a different value-id;
finally that key is queried and the target read out at the last position is its
**latest** value vector. This overwrite regime is exactly where an error-correcting
matrix memory is expected to have an edge over an EMA mixer.

## The bake-off

Two arms share an identical I/O contract and a comparable parameter budget,
differing only in the sequence-mixing core:

| arm | mixer | block |
|-----|-------|-------|
| RWKV     | `TNNetWKV` via `AddRWKVTimeMix` | TokenShift + WKV + gating |
| DeltaNet | `TNNetDeltaNet`                 | delta-rule matrix memory  |

Shared front-end / read-out, built with `TNNetPointwiseConvLinear`:

```
Input(SeqLen,1,cInDim)
 -> PointwiseConvLinear(cModelDim)      1x1 projection into memory width
 -> [MIXER]                             AddRWKVTimeMix  |  TNNetDeltaNet
 -> PointwiseConvLinear(cValueDim)      1x1 readout to the value vector
```

The DeltaNet arm warm-starts its write gate to `beta ~= 0.5` (one weight of the
cell) so the memory writes from step one. Both arms are trained with per-sample
SGD (`SetLearningRate(0.01, 0.9)`) on the *same* recall stream (same `RandSeed`)
for `cTrainSteps` steps each, then evaluated over `cEvalSeqs` held-out
sequences. `Evaluate` reports both the recall **MSE** at the query position and
the **exact-recall accuracy** (nearest neighbour of the prediction over the value
bank).

## Headline result

```
=== RWKV: associative recall, WKV time-mix vs delta-rule ===
keys=6  value_dim=4  pairs/seq=4  model_dim=16  seq_len=6
...
eval over 400 held-out recall sequences:
  RWKV     (WKV time-mix): recall MSE = ...   exact-recall acc = ...%
  DeltaNet (delta rule)  : recall MSE = ...   exact-recall acc = ...%
```

Trained on the same data for the same number of steps, both linear-time
recurrences learn the recall task well above chance (`100 / cNumVals` %). The
delta rule — an explicit error-correcting matrix memory — tends to edge out the
EMA-style WKV mixer on the hard *overwrite* regime, while WKV is the cheaper
softmax-free attention surrogate. The program closes with a self-check that the
WKV arm clears chance:

```
OK: the WKV time-mixing recurrence learns associative recall well above chance (...%).
```

(Exact numbers are seed-dependent; the *contrast* is the point.)

## Running

```
cd examples/RWKV
fpc -O3 -Mobjfpc -Sh -Fu../../neural RWKV.lpr
./RWKV
```

(or open `RWKV.lpi` in Lazarus). Pure CPU, tiny dimensions; finishes in a couple
of minutes on 2 cores.

Coded by Claude (AI).
