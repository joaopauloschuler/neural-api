# Center loss + softmax (joint discriminative feature learning)

This example reproduces the headline result of **Wen et al. 2016**,
*A Discriminative Feature Learning Approach for Deep Face Recognition*, on a
tiny synthetic multi-class task. It is pure CPU, single-threaded, uses no
external dataset, and finishes in about two seconds.

## The headline (Wen et al. 2016)

Training a feature extractor with **softmax cross-entropy alone** makes the
classes *separable* but leaves the per-class features *spread out*. **Adding the
center-loss penalty** jointly (lambda-weighted) pulls every sample toward its
learned class center, giving visibly **tighter intra-class clusters at the same
(or better) accuracy**. The center-loss objective is

```
L = L_softmax  +  (lambda/2) * sum_i || x_i - c_{y_i} ||^2
```

where `x_i` is the embedding of sample `i`, `y_i` its class and `c_{y_i}` the
learnable center of that class.

## What the example shows

Two arms train on the **same architecture, same seed, same data**; the only
difference is whether the joint center-loss penalty is active:

* **ARM A** — softmax cross-entropy only (center loss OFF, lambda effectively 0).
* **ARM B** — softmax + center loss joint (center loss ON, `lambda = 0.30`).

For each arm it reports final accuracy, an **intra-class tightness** metric
(`mean within-class feature radius` = `sqrt(mean ||x_i - mean_c||^2)`), the
**inter-class mean separation**, their **intra/inter ratio** (smaller = cleaner
clusters), and an **ASCII scatter of the 2-D embedding** so the tightening is
visible with no external plotting tools. Representative numbers:

```
ARM A: softmax cross-entropy ONLY (center loss OFF)
  accuracy                       : 1.0000
  intra-class radius (TIGHTNESS) : 1.9666   (smaller = tighter)
  inter-class mean separation    : 13.9238
  intra/inter ratio              : 0.1412   (smaller = cleaner)

ARM B: softmax + center loss JOINT (center loss ON)
  accuracy                       : 1.0000
  intra-class radius (TIGHTNESS) : 0.1142   (smaller = tighter)
  inter-class mean separation    : 3.9970
  intra/inter ratio              : 0.0286   (smaller = cleaner)
```

Both arms classify perfectly, but the center-loss arm's intra-class radius drops
~17x and its intra/inter ratio is ~5x smaller — exactly the Wen et al. result.
The ARM B ASCII scatter shows the four class blobs collapsed into compact knots,
while ARM A's blobs are sprawling.

## How the two heads share one embedding (joint wiring)

`TNNetCenterLoss` is an identity-passthrough **penalty head**: it consumes a
depth layout `feature | label` (the last channel is the integer class label,
channels `0..D-1` are the embedding) and in `Backpropagate` it **overwrites its
own output error** with `lambda*(x - center_c)` while pulling `center_c` toward
`x`. It ignores whatever residual the framework seeds into it. The
classification head needs the standard cross-entropy gradient instead. To make
**both heads share the same 2-D embedding inside one trainable network** we
branch the graph and rejoin it at a final `DeepConcat` that a single
`Backpropagate` call walks back through:

```
Input(1, 1, 3)                                  // [x, y, label]
 |
 |- SplitChannels([0,1]) -> PointwiseConvReLU(16) x2 -> PointwiseConvLinear(2)
 |       = EMB  (the shared 2-D embedding; pointwise = per spatial cell)
 |          |
 |          |- PointwiseConvLinear(K)            = LOGITS  (classification head)
 |          |
 |          |- DeepConcat([EMB, LABEL]) -> CenterLoss(K, lambda) = CENTER
 |
 |- SplitChannels([2])                           = LABEL  (untouched)
 |
 DeepConcat([LOGITS, CENTER])                    = final (1, 1, K + (2+1))
```

`EMB` feeds **two** consumers (the logits head AND the center head), so the
framework's departing-branch counter makes `EMB` wait for and **accumulate** the
gradients from both before propagating — exactly the gradient sharing the joint
objective needs.

**Backpropagate seeding.** The framework computes the last-layer residual as
`output_error = output - target`, so to inject the cross-entropy gradient we
compute `softmax + cross-entropy` externally on the `LOGITS` sub-region and pass
`target = output - desired_residual`, where the desired residual is
`softmax(logits) - onehot` on the K logits channels and `0` on the center
channels (the center head self-generates its own gradient and ignores its
residual). The final `DeepConcat` splits that back to each head. ARM A omits the
center head entirely (softmax only); ARM B includes it with `lambda > 0`.
Gradients from the center penalty reach `EMB` only in ARM B, which is precisely
why ARM B's clusters come out tighter.

## Pairs with

* [`FeatureSeparability`](../FeatureSeparability) — the fit-free
  Neural-Collapse cluster-geometry report; here we *cause* the tightening rather
  than just measure it.
* [`ArcFaceEmbedding`](../ArcFaceEmbedding) — the angular-margin alternative to
  center loss for the same intra-class-tightening goal.

## Building and running

```
lazbuild CenterLossSoftmaxJoint.lpi
../../bin/x86_64-linux/bin/CenterLossSoftmaxJoint
```

Or directly with `fpc` (mirrors how the test suite is built):

```
fpc -O3 -Mobjfpc -Sh -Fu../../neural CenterLossSoftmaxJoint.lpr
./CenterLossSoftmaxJoint
```

The run is deterministic (`RandSeed` is fixed and reset per arm for a fair
contrast), pure CPU, single-threaded, and finishes in about two seconds with a
tiny memory footprint.
