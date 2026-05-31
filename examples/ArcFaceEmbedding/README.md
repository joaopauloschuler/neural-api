# ArcFace embedding (angular-margin face/identity recognition)

This micro-example shows how the **ArcFace** angular-margin head
(`TNNetArcFace`) tightens the intra-class cosine clusters of a learned
embedding. It is pure CPU, single-threaded, uses no external dataset, and
finishes in under two seconds.

Four classes of 2D Gaussian blobs (think "identities") are mapped to a small
3-D embedding by a shared MLP. The embedding feeds an ArcFace classification
head that uses the **additive angular margin softmax**:

```
logit_k = s * cos(theta_k)      for k != y
logit_y = s * cos(theta_y + m)  for the true class y     (the angular margin)
loss    = -log softmax(logit)_y
```

`theta_k` is the angle between the (L2-normalized) embedding and the
(L2-normalized) class-`k` weight vector. Adding the margin `m` to the true
class's angle forces that angle to be **smaller** (more confident) before the
loss is satisfied, which pulls same-class embeddings into a tighter cone.

## What the example shows

It sweeps the angular margin `m` in `{0, 0.3, 0.5}` and, after training each
net, prints same-class vs different-class mean cosine similarity. `m = 0` is a
plain normalized (cosine) softmax classifier; larger `m` adds the angular
margin. The **same-class cosine rises** and the **separation grows** with `m`:

```
  margin    same     diff   separation   mean_loss
  ------   ------   ------   ----------   ---------
   0.00      0.9573     -0.3173      1.2747     0.02232
   0.30      0.9673     -0.3084      1.2758     0.04439
   0.50      0.9695     -0.3122      1.2817     0.05893
```

`same` is the mean cosine similarity of same-class embedding pairs, `diff` is
the mean over different-class pairs, and `separation = same - diff`. The mean
loss rises with `m` because the margin deliberately makes the objective harder
(the true class must clear a larger angular margin), not because the embedding
is worse — the rising same-class cosine confirms the clusters are tighter.

## How the ArcFace head is wired (label passthrough via depth concat)

`TNNetArcFace` consumes an input depth laid out as `embedding | label`: the last
depth channel is the integer class label and channels `0..D-1` are the
embedding. The forward pass is an identity passthrough; the loss and gradient
are produced in `Backpropagate` from that layout (so there is **no external
target** — the label rides in the input). To feed it inside one trainable
network we split the raw input into a feature branch and a label-passthrough
branch and re-concat along depth just before the head:

```
Input(1, 1, 3)                              // [x, y, label]
 |- SplitChannels([0,1]) -> PointwiseConvReLU(16) x2 -> PointwiseConvLinear(D)
 |       (the shared embedding MLP; pointwise = per spatial cell)
 |- SplitChannels([2])                       // the label channel, untouched
 DeepConcat([embedding, label]) -> (1, 1, D+1)
 ArcFace(K=4, m, s=8)
```

Gradients from the margin flow back through the embedding branch only (the
label branch carries no gradient), so the margin genuinely shapes the
embedding.

The `m = 0, s = 1` degenerate case of `TNNetArcFace` reduces exactly to a plain
normalized-softmax (cosine) classifier; this is pinned by
`TestArcFaceDegenerateIsCosineSoftmax` in `tests/TestNeuralNumerical.pas`.

## Building and running

```
lazbuild ArcFaceEmbedding.lpi
../../bin/x86_64-linux/bin/ArcFaceEmbedding
```

Or directly with `fpc` (mirrors how the test suite is built):

```
fpc -O3 -Mobjfpc -Sh -Fu../../neural -Fu../../neural/pas-core-math ArcFaceEmbedding.lpr
./ArcFaceEmbedding
```

The run is deterministic (`RandSeed` is fixed and reset per margin for a fair
sweep), pure CPU, single-threaded, and finishes in under two seconds with a
tiny memory footprint.
