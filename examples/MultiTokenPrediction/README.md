# Multi-Token Prediction (MTP)

Multi-Token Prediction (Gloeckle et al. 2024, *"Better & Faster Large Language
Models via Multi-token Prediction"*; popularized at scale by **DeepSeek-V3**)
trains a language model to predict **several future tokens at once** from each
position, instead of only the next one. Each position `t` then carries
`NumFuture` cross-entropy signals (`t+1, t+2, ..., t+NumFuture`) rather than a
single one, which **densifies the training signal** and tends to make the primary
next-token (`t+1`) head **converge faster**. The extra heads are also reusable at
inference time for **self-speculative decoding** (see below).

This example demonstrates the convergence-speed benefit on a tiny, fully
deterministic next-token rule, and exercises the new library builder
`TNNet.AddMultiTokenPrediction`.

## The builder: `TNNet.AddMultiTokenPrediction`

```pascal
function TNNet.AddMultiTokenPrediction(
  NumFuture, VocabSize: integer;
  ProjHidden: boolean = false;
  pActFn: TNNetActivationFunctionClass = nil): TNNetLayer;
```

It taps the current **shared trunk** hidden state — a `(SeqLen, 1, d_model)`
sequence tensor — and attaches `NumFuture` **parallel** prediction heads, head `h`
forecasting the token at `t+1+h` at every position `t`. It is a *builder* composed
purely from existing primitives (no new leaf class):

```
trunk -> [optional PointwiseConvLinear(d_model) + activation] ->
         PointwiseConvLinear(VocabSize) -> TNNetPointwiseSoftMax     (per head)
```

The `NumFuture` per-head, per-token softmax distributions are `TNNetDeepConcat`'d
along the depth axis into a single

```
(SeqLen, 1, NumFuture * VocabSize)
```

output, where head `h` occupies the depth slab `[h*VocabSize .. (h+1)*VocabSize-1]`.
The `t+1` head is always slab `h = 0`.

Every projection is a **pointwise (1×1) conv** so the sequence/token axis is
preserved — a `TNNetFullConnect*` would flatten and mix the whole sequence (see
the note in `AddMultiHeadSelfAttention`).

### Training target

Supervise it with a matching `(SeqLen, 1, NumFuture*VocabSize)` target tensor
whose slab `h` at position `t` is the **one-hot of the token at `t+1+h`**.
Positions whose future token falls past the end of the sequence are left zero (no
supervision there). Because the output is a per-token softmax, the framework's
default `(output − target)` seed gives the standard per-head cross-entropy
gradient — just call `NN.Backpropagate(Desired)` as usual.

`NumFuture = 1` degenerates to an ordinary single next-token head, which this
example uses as the baseline arm.

`ProjHidden = True` inserts an extra per-head hidden transform (`pActFn`, default
`TNNetReLU`) before the vocabulary projection; `False` (default) makes each head a
single linear vocabulary projection.

## The toy task

Each sequence is an **arithmetic progression** over a small vocabulary:

```
token[t] = (start + t*step) mod V        start ~ U{0..V-1},  step in {1,2,3,4}
```

After the first couple of tokens the rule is fully determined, so a **causal**
trunk at position `t` can predict not just `t+1` but `t+2`, `t+3`, … — exactly the
regime where MTP's denser signal helps.

Two arms share an **identical trunk** (one-hot → pointwise projection →
positional embedding → one causal transformer block) and the **same** training
and evaluation RNG streams per seed:

* **MTP arm** — `AddMultiTokenPrediction(NumFuture = 3)` (predicts `t+1, t+2, t+3`).
* **Baseline arm** — `AddMultiTokenPrediction(1)` (predicts `t+1` only).

The only difference is the MTP arm's extra `t+2`/`t+3` future heads.

## What it shows

We compare the **next-token (`t+1`) accuracy** at a deliberately **early
checkpoint** (a short, fixed step budget) averaged over several seeds — because
the comparison is about convergence *speed*, not the final converged accuracy
(train long enough and this easy rule saturates for both arms, erasing the gap).

A representative run:

```
 seed | MTP t+1 acc | baseline t+1 acc | delta
 -----+-------------+------------------+-------
   11 |     72.6%   |       71.2%      |    1.4
   23 |     78.1%   |       67.7%      |   10.3
   ...
 -----+-------------+------------------+-------
 mean |     75.2%   |       69.9%      |    5.3

MTP arm ahead in 7 / 8 seeds.
```

The MTP arm reaches higher `t+1` accuracy with the same number of training steps:
the auxiliary future-token losses give the primary head a head start.

## Inference-time reuse: self-speculative decoding

The extra heads are not wasted at inference. Because the same forward pass already
predicts `t+1, t+2, …, t+NumFuture`, the model can **draft** several tokens in one
pass and then **verify** them with subsequent passes (accepting the longest
matching prefix) — *self-speculative decoding*, which can speed up generation
**without a second draft network**. This contrasts with the
[`SpeculativeDecoding`](../SpeculativeDecoding/) example, which uses two *separate*
nets (a small draft + a large verifier), and with `TNNetTokenHistoryPenalty`,
which is a decode-time repetition penalty rather than a training objective. At
inference here you simply read the `t+1` slab (`depth 0 .. VocabSize-1`) for plain
greedy/sampled decoding, and the further slabs when speculating.

## Build & run

```
cd examples/MultiTokenPrediction
lazbuild --build-mode=Release MultiTokenPrediction.lpi
../../bin/x86_64-linux/bin/MultiTokenPrediction
```

Pure CPU, tiny dimensions; finishes in well under a minute on two cores.
