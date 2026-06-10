# Linformer — low-rank linear-complexity attention

Demonstrates `TNNetLinformerAttention`, **Linformer self-attention**
(Wang et al. 2020, "Linformer: Self-Attention with Linear Complexity",
[arXiv:2006.04768](https://arxiv.org/abs/2006.04768)), and contrasts it with a
full quadratic `TNNetScaledDotProductAttention` baseline at the **same sequence
length**.

## The layer

Standard single-head self-attention forms a `SeqLen × SeqLen` score matrix —
quadratic in the sequence length. Linformer first projects the Key and Value
sequences **down along the sequence axis** from `SeqLen` to a small fixed rank
`k ≪ SeqLen` using two learnable projection matrices `E`, `F` (each `k × SeqLen`):

```
K' = E · K          (k × d_k)
V' = F · V          (k × d_v)
Attn = softmax( Q · K'ᵀ / √d_k )    (SeqLen × k)   ← not SeqLen × SeqLen
Out  = Attn · V'                    (SeqLen × d_v)
```

so attention costs **O(SeqLen · k)** instead of O(SeqLen²).

`TNNetLinformerAttention` uses the same `Q|K|V` input contract as
`TNNetScaledDotProductAttention` / `TNNetLinearAttention`: `SizeY = 1`, input
depth `3·d_k` laid out as `[Q | K | V]` along the depth axis, output depth
`d_v = d_k`. The two projection matrices `E`, `F` are stored as trainable neurons
(`FNeurons[0]=E`, `FNeurons[1]=F`) with an exact backward pass — both the input
gradient and the `E`/`F` weight gradients are finite-difference checked in the
test suite.

Because `E` and `F` carry a fixed `SeqLen` dimension, the layer requires a
**FIXED sequence length** (asserted in `SetPrevLayer`). This is the standard,
expected Linformer constraint.

It is distinct from the kernel / feature-map linear-attention family
(`TNNetLinearAttention` with `φ(x)=elu(x)+1`, `TNNetGatedLinearAttention`,
`TNNetDeltaNet`, `TNNetWKV`): those drop the softmax and reassociate the
products, whereas Linformer **keeps the softmax** and instead low-rank-projects
the sequence axis.

## The task (majority value)

Each sequence carries `cNumTokens` one-hot value tokens drawn from `cNumVals`
classes, followed by a dedicated query position; the target read out at the query
is the **most frequent value class** in the sequence. Answering it requires
pooling evidence across the whole sequence — a global, permutation-friendly
aggregate, which is exactly the regime where attention is approximately low-rank
and Linformer's `k`-rank projection loses little.

Both arms share an identical I/O contract and differ **only** in the attention
layer: `1×1` projection (→ `3·d_k` = `Q|K|V`) → attention → `1×1` readout →
per-position softmax.

## Headline result

```
attention score-matrix size (per head, per sequence):
  Full SDPA  : 17 x 17 = 289 scores
  Linformer  : 17 x 4 = 68 scores  (23.5% of full)

eval over 600 held-out sequences (chance = 20.0%):
  Linformer (rank 4) : majority-class accuracy = 38.0%
  Full SDPA          : majority-class accuracy = 57.5%
```

At the same sequence length the Linformer arm learns the task well above chance
using a score matrix of just **23.5%** the size of full SDPA — the classic
Linformer trade: a little accuracy for **linear** attention cost.

## Running

```
lazbuild Linformer.lpi
../../bin/x86_64-linux/bin/Linformer
```

Pure CPU, tiny dimensions; finishes in a few seconds.
