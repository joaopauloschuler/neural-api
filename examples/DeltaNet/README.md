# DeltaNet — delta-rule associative recall

Demonstrates `TNNetDeltaNet`, the delta-rule linear-attention recurrence
(Yang et al. 2024, "Parallelizing Linear Transformers with the Delta Rule over
Sequence Length", [arXiv:2406.06484](https://arxiv.org/abs/2406.06484)), on an
**overwrite** key→value recall task and contrasts it with a parameter-matched
fixed-decay linear-attention baseline (`TNNetRetention`).

## The layer

`TNNetDeltaNet` carries a `(d × d)` matrix memory `S` over the sequence/time
axis and updates it per timestep with the classic delta (Widrow–Hoff) rule:

```
pred_t = Sᵀ_{t-1} k_t            (read the current value prediction for k_t)
err_t  = v_t − pred_t            (delta-rule error vs the target value)
S_t    = S_{t-1} + β_t · k_t ⊗ err_t   (write back ONLY the correction)
y_t    = Sᵀ_t q_t               (read-out)
```

The per-token write strength `β_t = sigmoid(…) ∈ (0,1)` and the read-then-correct
write make `S` a **true editable associative memory**: writing a fresh value for
a key first removes the stale association, unlike `TNNetRetention` (fixed/learned
exponential decay) or `TNNetMLSTMCell` (unbounded outer-product accumulation),
which can only *blend* associations by recency.

## The task (overwrite recall)

Each sequence presents several distinct `(key, value)` write tokens; one key is
then **re-written with a new value** (an overwrite); finally that key is queried
and the network must return its **most recent** value. Retrieving it requires
removing the stale association and keeping the fresh one — exactly what the delta
rule does, and exactly what a fixed-decay accumulator cannot do cleanly.

Both arms share the same I/O contract and a matched parameter budget:
a per-token `1×1` projection → memory mixer (`TNNetDeltaNet` or `TNNetRetention`)
→ `1×1` readout.

## Headline result

```
eval over 400 held-out recall sequences:
  DeltaNet (delta rule)  : recall MSE = 0.00607   exact-recall acc = 100.0%
  Retention (fixed decay): recall MSE = 0.02254   exact-recall acc = 93.0%
```

The delta-rule layer reaches lower recall error and perfect exact recall, while
the fixed-decay baseline blends the stale and fresh values and misses a fraction
of overwrites.

## Running

```
lazbuild DeltaNet.lpi
../../bin/x86_64-linux/bin/DeltaNet
```

Pure CPU, tiny dimensions; finishes in a couple of seconds.
