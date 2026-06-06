# AttentionEntropyReport

Demonstrates `TNNet.AttentionEntropyReport`, a per-row Shannon-entropy diagnostic
for every `TNNetScaledDotProductAttention` layer in a network.

The example builds a tiny two-block SDPA model (one non-causal block then one
causal block, each preceded by a pointwise linear projection into a Q | K | V
concatenation) and trains it briefly on a synthetic "broadcast from position 0"
task: at every output position `i`, the target is `tanh(input[0, :])`. The
attention-optimal solution is to route every query almost entirely to key 0,
which collapses the per-row entropy.

The example prints the report twice:

1. **Before training** — weights are random, so attention rows should sit near
   `log(SeqLen)` and the `dead` column should be non-zero.
2. **After training** — rows should concentrate at low entropy values, raising
   the `spike` column and shifting the normalised-entropy histogram to the
   left.

## Build & run

```
cd examples/AttentionEntropyReport
lazbuild AttentionEntropyReport.lpi
./AttentionEntropyReport
```

Runtime is well under a minute on CPU.

## What the report shows

For each SDPA layer:

- `meanH`, `stdH` of per-row entropy (natural log, in nats),
- the layer's `log(K)` reference (max possible entropy if all keys were
  uniformly weighted),
- `meanH/lK` ratio (normalised entropy in `[0, 1]`),
- `dead` count — rows whose entropy is within `DeadEpsilon` of `log(K)`
  (nearly-uniform attention, no useful routing),
- `spike` count — rows whose entropy is below `SpikeThreshold` (attending
  to essentially one key),
- a 10-bin ASCII histogram of normalised entropy across all probe rows.

Use it as a single-number-per-layer health check across training runs:
many dead rows late in training signals an under-utilised attention block;
all spikes signals collapse onto a single key.
