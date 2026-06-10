# Cosine-Attention Learnable Scale

A tiny, self-contained experiment (no downloads, no data files) that turns the
`scale` scalar of `TNNetCosineSimilarityAttention` into a **single learnable
parameter** and watches where training drives it.

## Why

Cosine attention replaces the raw `Q.K^T` logit with a bounded cosine score:

```
score[i,j] = scale * (Q[i]/||Q[i]||) . (K[j]/||K[j]||)
```

Because the cosine term lives in `[-1, +1]`, the pre-softmax logits are bounded
to `[-scale, +scale]`. Cosine-attention papers therefore hard-code
`scale = 1/tau` with a small temperature `tau ~ 0.05..0.1`, i.e. `scale ~ 10..20`
-- otherwise a `scale` of `1.0` keeps the softmax too flat to ever become
confident. This example asks: **if we just make `scale` learnable (init 1.0),
does gradient descent rediscover that large temperature on its own?**

## Setup

One `TNNetCosineSimilarityAttention(d_k=6, learnable scale, init 1.0)` layer over
a `4 x 1 x (3*d_k)` packed `Q|K|V` tensor. The synthetic task forces each query
`q` to copy the value of exactly one key `(q+1) mod SeqLen`: that query and that
key are given the same one-hot direction (cos ~ 1), every other key points
elsewhere (cos ~ 0). The target can only be matched by a **sharp** softmax, which
is reachable only as `scale` grows.

The layer is driven directly (Compute / Backpropagate, per-sample update) so the
scale trajectory is easy to read.

## Result

```
epoch       scale        loss
    0       1.0000     2.637767
  400       5.5399     0.001299
 2000       6.3611     0.000255
 4000       6.7118     0.000127

Final learned scale = 6.7119
```

Training drives `scale` from `1.0` up past `6.7` (still climbing, loss
monotonically shrinking) -- i.e. it rediscovers the large `1/tau`-style
temperature on its own when the task needs peaked attention. Conversely, on a
task where sharpening does not help (smooth, near-collinear keys), the gradient
instead pushes `scale` toward `0` (uniform attention) -- so the learnable scale
adapts to the data rather than to a cargo-culted constant.

## Build & run

```
lazbuild --build-mode=Release CosineAttentionLearnableScale.lpi
../../bin/x86_64-linux/bin/CosineAttentionLearnableScale
```
