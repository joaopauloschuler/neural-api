# SinkAttentionStability

Attention-**sink** stability micro-experiment for `TNNetSinkAttention` — the
"attention sink" idea from StreamingLLM (Xiao et al. 2023, *Efficient
Streaming Language Models with Attention Sinks*,
<https://arxiv.org/abs/2309.17453>).

## The idea

Softmax attention must distribute a full unit of probability mass over the
keys **even when no key is relevant** — the softmax denominator can never be
zero. On such an "all-keys-irrelevant" query position the mass gets dumped
onto whatever key scores least-badly, i.e. it becomes noise. StreamingLLM's
fix is to append a few always-available, never-masked, learnable **sink**
slots. Softmax can park the otherwise-misplaced mass on the sinks, so the
mass landing on the **real** keys drops.

`TNNetSinkAttention` augments the key set with `NumSinks` learnable slots
(stored as extra neurons: sink keys + sink values). Its post-softmax map is
the augmented row `[sink slots ++ real keys]`, exposed read-only via the
`SinkAttentionWeights` property (layout `X` = augmented key index, `0..K-1` =
sinks, `K..K+SeqLen-1` = real keys; `Y` = query index; rows sum to 1).

## What this probe does

Builds a tiny **causal** next-token setup, `TNNetInput(SeqLen, 1, 3*d_k)`
(the depth axis packs `Q | K | V`, the same layout SDPA uses). Every position
except the last gets a query strongly aligned with its own key (a clear real
target). The **last** position is the **probe row**: its query is
~orthogonal to every real key, so *all* keys are irrelevant to it.

A forward pass runs through:

- (a) plain `TNNetScaledDotProductAttention` (causal), and
- (b) `TNNetSinkAttention` (causal) with `NumSinks` in `{1, 2, 4}`,

and the probe row's post-softmax mass is read straight off the layer. We
print the fraction of mass on the **sink** slot(s) vs the **real** keys.

```
TNNetInput(SeqLen, 1, 3*d_k)         # depth packs Q | K | V
  -> TNNetScaledDotProductAttention(d_k, causal)   # baseline, no sink
  -> TNNetSinkAttention(d_k, causal, NumSinks)      # augmented key set
```

Forward-only, deterministic, single Compute per model — runs in well under a
second on one CPU thread, modest memory.

## Build & run

```
lazbuild SinkAttentionStability.lpi
../../bin/x86_64-linux/bin/SinkAttentionStability
```

## Observed output

```
TNNetSinkAttention attention-sink stability micro-experiment
  d_k = 8, SeqLen = 6 (causal)
  probe = last query row; its query is ~orthogonal to all keys
  (so EVERY real key is irrelevant to it).

plain SDPA  : real-key mass on probe row = 1.0000  (sink mass = n/a, no sink slot)

----------------------------------------------------------------
NumSinks        sink mass      real mass   real - plainSDPA
----------------------------------------------------------------
1                  0.1408         0.8592            -0.1408
2                  0.2467         0.7533            -0.2467
4                  0.3957         0.6043            -0.3957
----------------------------------------------------------------
```

## Reading the result

Plain SDPA has no outlet, so it dumps the full unit of mass (`1.0000`) onto
the irrelevant real keys. With sink slots the never-masked sinks absorb a
share of that mass, and the real-key mass on the probe row falls strictly
below SDPA's `1.0` every time: `0.86` (1 sink), `0.75` (2 sinks), `0.60`
(4 sinks). More sinks absorb more mass and leave less real-key noise — the
sink mass rises monotonically `0.14 -> 0.25 -> 0.40` while real-key mass
falls monotonically. This is exactly the StreamingLLM attention-sink
stabilisation claim: the sink slot absorbs the otherwise-misplaced attention
mass.

These are the untrained sink slots (sink keys initialised small-random, sink
values zero). Because the probe query is near-orthogonal to every real key,
all scores are ~0 and the softmax is roughly uniform over the augmented row,
giving sink mass ~ `K / (K + SeqLen)` — which matches the table. Training the
sinks would let them learn to absorb even more on genuinely-irrelevant rows;
the absorption is already monotone in `NumSinks` here without any training.
