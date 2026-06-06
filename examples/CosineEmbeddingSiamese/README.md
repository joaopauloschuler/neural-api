# Cosine-embedding siamese (same vs different pairs)

This tiny example demonstrates the `TNNetCosineEmbeddingLoss` head on a
synthetic **"same vs different class"** pair task, using a **shared-weight
(siamese)** embedding MLP. It is pure CPU, uses no external dataset, and
finishes in well under a second.

A handful of nD Gaussian class blobs are mapped, by **one shared MLP applied to
both members of every pair**, to a low-dimensional embedding on the unit sphere.
We sample same-class pairs (label `y=1`) and different-class pairs (`y=0`) and
train with the PyTorch-style pairwise cosine-embedding loss

```
L = y*(1 - cos(a,b)) + (1 - y)*sqr(max(0, cos(a,b) - margin))
```

so same-class pairs are pulled toward `cos = +1` while different-class pairs are
pushed down to (or below) the `margin`.

## How the cosine-embedding head is wired

`TNNetCosineEmbeddingLoss` is a **self-contained** metric head: there is **no
external target tensor**. The per-pair similarity label `y` is packed INTO the
input. Per spatial `(X,Y)` cell it splits the input depth as

```
[ a (d channels) | b (d channels) | y (1 channel) ]   => Depth = 2*d + 1
```

(validated odd and `>= 3`). `a` and `b` are the two embeddings to compare; `y`
is the similarity label (`y=1` => similar, `y=0` => dissimilar). The forward
pass is an identity passthrough; `Backpropagate` writes the cosine-loss
`+gradient` into the `a` and `b` channels and `0` into the `y` channel. The
`margin` (default `0.0`, must be in `[-1,1]`) round-trips via Save/Load.

To produce that `a|b|y` layout natively the example builds a **two-input** net:

```
Input0(SizeX=2, SizeY=1, Depth=in_dim)   // the pair: point a at X=0, b at X=1
Input1(SizeX=1, SizeY=1, Depth=1)        // the scalar label y

  branch off Input0:
    PointwiseConvReLU(16)        // featuresize=1 => the SAME weights are
    PointwiseConvReLU(16)        // applied at every X position: a genuine
    PointwiseConvLinear(d)       // SHARED embedding MLP (siamese over a,b)
    L2Normalize                  // each point's embedding -> unit sphere
      -> shape (2,1,d)
    Reshape(1,1,2*d)             // pure reinterpretation -> a|b in depth

DeepConcat([Reshape, Input1])    // append the y channel -> (1,1,2*d+1) = a|b|y
CosineEmbeddingLoss(margin)
```

Because `TNNetVolume` is stored depth-major (`pos = ((SizeX*y)+x)*Depth + d`),
the two per-X embeddings land as consecutive depth chunks after the reshape —
exactly the `a|b` layout the head consumes — and the depth concat then tacks on
`y`. The two `TNNetInput` layers are added first so the multi-input
`Compute([Pair, YVol])` / `Backpropagate` overloads can feed them by index.

Both members of a pair pass through the **same** pointwise-conv weights, so the
embedding function is genuinely shared (a true siamese net): a single learned
encoder is trained purely by the relational `same/different` supervision.

## What the example prints

After training it prints cosine-similarity **histograms** for held-out
same-class vs different-class pairs (with the per-group mean ± spread):

```
SAME-class pairs (target cos -> +1)  (n=600)  mean cos = 1.000  +/- 0.000
  ...
  [  0.80,  1.00)   600 |########################################

DIFFERENT-class pairs (pushed below margin)  (n=600)  mean cos = -0.250  +/- 0.382
  [ -1.00, -0.80)   102 |##################
  ...
  [  0.00,  0.20)   226 |########################################
```

Same-class pairs collapse onto `cos = +1`; different-class pairs are all driven
below the `margin` (0.2 here), exactly the headline behavior of the
cosine-embedding loss.

## Building and running

```
lazbuild CosineEmbeddingSiamese.lpi
../../bin/x86_64-linux/bin/CosineEmbeddingSiamese
```

The run is deterministic (`RandSeed` is fixed), pure CPU, and finishes in well
under a second.
