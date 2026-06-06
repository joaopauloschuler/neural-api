# LoRAFineTune — parameter-efficient adaptation with `TNNet.AddLoRAAdapter`

## What it does

Trains a small softmax classifier on a **base** synthetic task (four 2D Gaussian
blobs), then **freezes every base weight** and adapts the model to a
**related-but-shifted target** task by training only a low-rank **LoRA** bypass
added to one frozen projection of the trunk. This is a minimal demonstration of
**LoRA** (Hu et al. 2021, [arXiv:2106.09685](https://arxiv.org/abs/2106.09685)):
instead of fine-tuning a whole backbone, adapt a frozen model by training a tiny
low-rank residual on top of a frozen linear layer.

The adaptation capacity is `TNNet.AddLoRAAdapter`, which wraps a frozen
`d_in -> d_out` projection with a rank-`r` residual bypass

```
adapted(x) = base(x) + (alpha/r) * B*A*x
```

where `A: d_in -> r` (the *down* projection) and `B: r -> d_out` (the *up*
projection) are the **only** trainable parameters. `B` is **zero-initialised**,
so at step 0 the bypass contributes nothing and the adapted forward equals the
frozen base **bit-for-bit** (the LoRA "starts as identity" property — `CHECK 1`).

### Shape rule (important)

Both projections are `TNNetPointwiseConvLinear` (over `Depth`), **not**
`TNNetFullConnect`. For a per-token `(SeqLen, 1, d_model)` stream a FullConnect
would flatten and mix the whole sequence and zero the per-token gradient; the
pointwise projections act per-token so the residual is **shape-preserving** and
the `TNNetSum([frozenProjection, scaledUp])` is valid. (This demo uses a
`(1,1,2)` single-token stream, the degenerate `SeqLen=1` case of the same rule.)

The builder does **not** freeze the base — the caller does (see below). It also
must not be followed by `NN.InitWeights()`, which would re-run `InitDefault` on
every layer and overwrite `B`'s zero-init; layers are already initialised at
`AddLayer` time (the base randomly, `B` at zero).

## The experiment

The example runs the textbook LoRA rank sweep `r in {1, 2, 4, 8}` and charts, for
each rank, the number of trainable adapter params (a small fraction of the full
model) against the recovered **target** accuracy, bracketed by two references:

- **frozen base** (no adaptation) — the lower bound, and
- **full fine-tune** (every weight trainable) — the upper bound.

Every rank starts from the **same** pretrained base (copied in via per-layer
`CopyWeights`), so the comparison is apples-to-apples. The expected shape is
"recover most of the full-fine-tune accuracy at a few % of the parameters".

## How to run

```
cd examples/LoRAFineTune
fpc -B -Fu../../neural -Fu"$LAZUTILS_PATH" -Mobjfpc -Sh -O2 LoRAFineTune.lpr
./LoRAFineTune
```

(`LAZUTILS_PATH` defaults to
`/usr/share/lazarus/4.4.0/components/lazutils/lib/x86_64-linux`.) Pure CPU,
manual training loop, runs in under a minute.

## Freezing mechanism

Freezing is done per layer via the writable `TNNetLayer.LearningRate` property.
A frozen layer has `LearningRate := 0`, so its weight delta `-LR*grad` is `0`
and its weights never move, while the output error still backpropagates THROUGH
it (so the gradient reaches the LoRA `A`/`B` layers). The net-wide
`NN.SetLearningRate(LR, 0.0)` is called first, which also sets **inertia to 0**
net-wide; with inertia 0 a frozen layer has no momentum term either, so it
provably cannot drift (`UpdateWeights` reduces to `FWeights.Add(FDelta)` with
`FDelta == 0`).

During each rank's adaptation, **everything except the two adapter layers**
(`A` and `B`) is frozen. The example asserts the sampled frozen projection
weight is **bit-identical** before and after adaptation (hard `Halt(1)` on
failure), proving the base weights truly do not move.

## Sample output

```
=== Pretraining base classifier on BASE task ===
  base accuracy on BASE   test: 100.00 %
  base accuracy on TARGET test:  90.25 %   (distribution-shifted -> degraded)

=== Reference: full fine-tune (every weight trainable) ===
  full fine-tune TARGET accuracy:  99.88 %  (244 trainable params)

=== LoRA rank sweep (only A/B adapter params trainable) ===

---------------------------------------------------------------
Rank sweep: trainable adapter params vs recovered TARGET accuracy
  full network (upper bound):   244 params  ->  99.88 %
  frozen base (lower bound) :     0 params  ->  90.25 %
  ---------------------------------------------------
   rank | adapter params | % of full | TARGET acc
  ------+----------------+-----------+-----------
      1 |             37 |    15.16 % |   98.88 %
      2 |             62 |    25.41 % |   99.88 %
      4 |            112 |    45.90 % |   99.94 %
      8 |            212 |    86.89 % |   99.88 %
---------------------------------------------------------------

CHECK 1 PASS: B zero-init -> adapted == frozen base before training (max diff 0.000000000).
CHECK 2 PASS: rank-1 adapter uses far fewer params (37 < 244).
CHECK 3 PASS: best LoRA recovers accuracy (99.94 % > base 90.25 % + margin, and >  70 %).

ALL CHECKS PASSED.
```

Even rank 1 (15 % of the full-fine-tune params) recovers most of the gap, and
rank 2+ matches the full fine-tune — the canonical LoRA result.

## What the checks assert (PASS/FAIL, `Halt(1)` on failure)

1. **CHECK 1** — with `B` zero-init the adapted forward equals the frozen base
   forward to `< 1e-6` BEFORE any training (the "starts as identity" property).
2. **CHECK 2** — the rank-1 adapter trains far fewer params than the full
   network, i.e. LoRA is parameter-efficient.
3. **CHECK 3** — the best LoRA rank's **target** accuracy clearly beats the
   frozen-base target accuracy (`best_acc > base_acc + 0.05`) and exceeds `0.70`.

(A fourth, inline check inside the sweep `Halt(1)`s if any frozen base weight
moves during a rank's adaptation.)

## Deferred

The "merge for inference" property (fold `W + (alpha/r)*B*A` into a single dense
layer and verify it reproduces the adapted forward) is **not** demonstrated here:
folding two `TNNetPointwiseConvLinear` projections plus the scale into the base
projection's weights is fiddly and orthogonal to the parameter-efficiency point
this example makes. It is left as a future addition.
