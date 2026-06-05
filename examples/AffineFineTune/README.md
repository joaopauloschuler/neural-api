# AffineFineTune — BitFit-style cheap adaptation with `TNNet.AddAffineBlock`

## What it does

Trains a small softmax classifier on a **base** synthetic task (four 2D
Gaussian blobs), then **freezes every base weight** and adapts the model to a
**related-but-shifted target** task by fine-tuning only a couple of per-channel
affine blocks. This is a minimal demonstration of **BitFit-style** adaptation
(Ben Zaken, Goldberg & Ravfogel 2021): instead of fine-tuning a whole backbone,
adapt a frozen model by training a tiny number of parameters.

The adaptation capacity is `TNNet.AddAffineBlock`, a learnable per-channel
affine transform

```
y[d] = gamma[d] * x[d] + beta[d]
```

built as `TNNetChannelMul` (per-channel scale `gamma`, initialised to 1.0)
followed by `TNNetChannelBias` (per-channel shift `beta`, initialised to 0.0).
At init the block is the **exact identity** (`gamma=1`, `beta=0`), so inserting
it never changes the base model until it is trained. It is FullConnect-separable:
only `2*Depth` parameters, no cross-channel mixing.

The **target** task is the base blobs with each coordinate scaled and shifted by
a fixed affine map (`x' = 1.6*x + 3.0`, `y' = 0.7*y - 2.5`). A per-channel
affine on the hidden features is exactly the right capacity to undo this kind of
input distribution shift, so the affine-only fine-tune recovers most of the
accuracy a full fine-tune would, at a tiny fraction of the trainable params.

Two affine blocks are inserted into the trunk (one after each hidden ReLU layer)
and are present from the start, frozen to identity during base training. Note:
`TNNetFullConnect*` emits its units on the `SizeX` axis (shape `(16,1,1)`,
`Depth=1`), while `TNNetChannelMul`/`TNNetChannelBias` scale per-`Depth`, so a
`TNNetReshape(1,1,16)` precedes each affine block to move the features onto the
`Depth` axis. The affine then learns a genuine per-feature `gamma[16]`,
`beta[16]`.

## How to run

```
cd examples/AffineFineTune
fpc -O3 -Mobjfpc -Sc -Sh -veiq -Fu../../neural AffineFineTune.lpr
./AffineFineTune
```

Pure CPU, single-threaded (manual training loop, no thread pool), runs in a few
seconds.

## Freezing mechanism

Freezing is done per layer via the writable `TNNetLayer.LearningRate` property.
A frozen layer has `LearningRate := 0`, so its weight delta `-LR*grad` is `0`
and its weights never move, while the output error still backpropagates THROUGH
it (so the gradient reaches the affine blocks). The net-wide
`NN.SetLearningRate(LR, 0.0)` is called first, which also sets **inertia to 0**
net-wide; with inertia 0 a frozen layer has no momentum term either, so it
provably cannot drift (`UpdateWeights` reduces to `FWeights.Add(FDelta)` with
`FDelta == 0`).

The roles flip between phases:
- **Base training:** the affine blocks are frozen (kept at identity); the
  trunk + head train.
- **Affine-only adaptation:** the trunk + head are frozen; only the affine
  blocks train.

`CHECK 3` asserts a sampled base-trunk weight is **bit-identical** before and
after adaptation, proving the base weights truly do not move.

## Trainable-parameter comparison

| path                  | trainable params | fraction |
|-----------------------|------------------|----------|
| full network (trunk + head) | 388        | 100 %    |
| affine-only fine-tune | 68               | 17.53 %  |

Each affine block is a `TNNetChannelMul` + `TNNetChannelBias` operating over
`Depth=16`. The example's param counter scores `weights + 1 bias` per neuron, so
each affine layer contributes `16 + 1 = 17` and the 4 affine layers across the 2
blocks total `4 * 17 = 68` trainable params — a small fraction of the 388-param
trunk + head, which stay frozen.

## Sample output

```
Frozen base model accuracy:
  on BASE   test: 100.00 %
  on TARGET test:  40.69 %   (distribution-shifted -> degraded)

Trainable-parameter comparison:
  full network        :    388 params
  affine-only fine-tune:     68 params  (17.53 % of full)

Accuracy on TARGET test set:
  frozen base          :  40.69 %
  affine-only fine-tune:  99.56 %

CHECK 1 PASS: affine path is a small fraction of trainable params (17.53 %).
CHECK 2 PASS: affine fine-tune adapts (99.56 % > base 40.69 % + margin, and >  70 %).
CHECK 3 PASS: frozen base weight unchanged (0.75188994).

ALL CHECKS PASSED.
```

## What the checks assert (PASS/FAIL, `Halt(1)` on failure)

1. **CHECK 1** — the affine-only trainable-param count is a small fraction
   (< 20 %, ~17.5 % here) of the full network's trainable params.
2. **CHECK 2** — the affine-only fine-tuned **target** accuracy clearly beats the
   frozen-base target accuracy (`affine_acc > base_acc + 0.05`) and exceeds
   `0.70`, i.e. the affine blocks genuinely adapt the frozen model.
3. **CHECK 3** — a sampled frozen base weight is unchanged after adaptation,
   proving the freezing mechanism holds.
