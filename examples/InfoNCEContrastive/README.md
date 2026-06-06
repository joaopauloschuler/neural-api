# InfoNCE contrastive embedding

This example learns a **contrastive representation** of a synthetic multi-class
toy task using the `TNNetInfoNCELoss` head and `TNNetL2Normalize`. It is pure
CPU, uses no external dataset, and finishes in a couple of seconds.

Each latent class owns a random prototype vector in input space; an *augmented
view* of a class is its prototype plus Gaussian noise. A single training sample
packs:

```
slab 0          q     = a view of class c            (the query)
slab 1          k_0   = ANOTHER view of class c       (the positive)
slab 2..K       k_1.. = views of OTHER classes        (the negatives)
```

A weight-shared encoder maps every view to a `d`-dim **unit-norm** embedding.
InfoNCE (van den Oord et al. 2018) trains the encoder so the query's embedding
is closer (higher cosine) to its positive than to any of the `K-1` negatives,
via a temperature-scaled softmax cross-entropy that selects the positive:

```
s_j = <q, k_j> / tau           // similarity of query to key j
L   = -s_0 + logsumexp_j(s_j)  // pick the positive (slab 1, j=0) among all keys
```

## How the InfoNCE head is wired

`TNNetInfoNCELoss` has **no external target** — supervision is implicit in the
input **depth** layout, exactly like `TNNetTripletLoss`. Construct it with
`TNNetInfoNCELoss.Create(EmbeddingDim, Temperature)`. It requires the input
depth to be divisible by the embedding dim `d`, splitting it into
`NumSlabs = Depth div d` slabs (at least 3: one query + at least two keys). Per
spatial cell, slab `0` is the query `q`, slab `1` is the positive `k_0`, and
slabs `2..K` are the negatives. The forward pass is a pure identity passthrough
(so `Net.Compute` returns the raw packed embeddings); `Backpropagate` writes the
analytic InfoNCE gradient and **seeds it itself** — no target, no manual
gradient surgery.

To produce that layout, one network embeds all `K+1` views at once with a
genuinely shared encoder:

```
Input(SizeX=K+1, SizeY=1, Depth=in_dim)  // K+1 views at X=0..K; raw coords
PointwiseConvReLU(16)                     // featuresize=1 => the SAME weights
PointwiseConvReLU(16)                     // are applied at every X position:
PointwiseConvLinear(embed_dim)            // a SHARED encoder MLP over the views
L2Normalize                              // each view's embedding -> unit sphere
  -> output shape (K+1, 1, embed_dim)
Reshape(1, 1, (K+1)*embed_dim)            // pure reinterpretation -> q|k_0|.. depth
InfoNCELoss(embed_dim, tau)
```

Because `TNNetVolume` is stored depth-major (`pos = ((SizeX*y)+x)*Depth + d`),
the per-X embeddings land as consecutive depth chunks after the reshape — exactly
the `q | k_0 | k_1 | .. | k_{K-1}` layout the loss head consumes. No transpose
is needed.

## Contrast with the other embedding examples

- **`examples/TripletEmbedding`** uses `TNNetTripletLoss`: a margin/hinge loss
  over a single `(anchor, positive, negative)` triplet
  (`L = max(0, ||a-p||^2 - ||a-n||^2 + margin)`), i.e. exactly **one** negative
  and a hard margin. InfoNCE contrasts the positive against **K negatives at
  once** through a temperature-scaled softmax (a soft, multi-negative
  generalization; no margin, a temperature `tau`).
- A `TNNetCosineEmbeddingLoss`-style head scores **one** pair as
  similar/dissimilar against a per-pair target label. InfoNCE needs no labels:
  the positive is fixed by position (slab 1) and the contrast set is the other
  slabs.

More negatives generally give a tighter, more informative contrastive signal,
which is why InfoNCE underpins modern self-supervised methods (SimCLR, MoCo).

## What the example prints

It reports, **before and after** training, the mean positive-pair cosine, the
mean negative-pair cosine, and the Wang & Isola (2020) **alignment** (mean
`||z_q - z_k+||^2` over positives) and **uniformity** (`log mean exp(-2 ||z_i -
z_j||^2)` over negatives), plus the InfoNCE loss. Sample output:

```
BEFORE training:
  mean pos cosine =   0.7163   mean neg cosine =   0.1261   gap =   0.5902
  alignment       =   0.5674   uniformity      =  -1.8768   InfoNCE loss =   1.0432

AFTER training:
  mean pos cosine =   0.9916   mean neg cosine =  -0.2076   gap =   1.1992
  alignment       =   0.0168   uniformity      =  -4.4304   InfoNCE loss =   0.0185

Summary (Wang & Isola 2020 alignment/uniformity):
  pos-vs-neg cosine gap : 0.5902 -> 1.1992   (wider is better)
  alignment             : 0.5674 -> 0.0168   (lower is better)
  uniformity            : -1.8768 -> -4.4304
  InfoNCE loss          : 1.0432 -> 0.0185   (lower is better)

Correctness signals:
  [PASS] pos-vs-neg cosine gap widened (positives pulled together, negatives pushed apart)
  [PASS] final InfoNCE loss < initial InfoNCE loss
  [PASS] alignment decreased (positive pairs collapsed closer)
```

The cosine gap widens (positives pulled to ~1, negatives pushed below 0),
alignment drops toward 0, uniformity becomes more negative (embeddings spread
out), and the loss collapses by ~50x. The program exits non-zero if any
correctness signal fails.

## Building and running

```
lazbuild InfoNCEContrastive.lpi
../../bin/x86_64-linux/bin/InfoNCEContrastive
```

The run is deterministic (`RandSeed` is fixed), pure CPU, and finishes in
about two seconds.
