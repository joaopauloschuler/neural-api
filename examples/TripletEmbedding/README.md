# Triplet embedding (metric learning)

This example learns a low-dimensional **metric-learning embedding** of a
synthetic multi-class toy dataset using the `TNNetTripletLoss` head and
`TNNetL2Normalize`. It is pure CPU, uses no external dataset, and finishes in
well under a second.

Four classes of 2D Gaussian blobs (centered at the corners of a square) are
mapped to a **3-D embedding constrained to the unit sphere**. Training uses the
triplet hinge

```
L = max(0, ||a - p||^2 - ||a - n||^2 + margin)
```

to pull same-class samples (anchor `a`, positive `p`) together while pushing
different-class samples (negative `n`) apart.

## How the triplet head is wired

`TNNetTripletLoss` has **no external target** — supervision is implicit in the
input depth layout. The head splits its input depth into 3 equal
`anchor | positive | negative` chunks (requires `Depth mod 3 = 0`) and computes
the hinge per spatial cell. To feed it, this example uses **one weight-shared
siamese network** that processes a whole triplet in a single forward pass:

```
Input(SizeX=3, SizeY=1, Depth=2)    // 3 points (a,p,n) at X=0,1,2; 2 coords
PointwiseConvReLU(16)               // featuresize=1 => the SAME weights are
PointwiseConvReLU(16)               // applied at every X position: a genuine
PointwiseConvLinear(embed_dim)      // SHARED embedding MLP over the 3 points
L2Normalize                         // each point's embedding -> unit sphere
  -> output shape (3, 1, embed_dim)
Reshape(1, 1, 3*embed_dim)          // pure reinterpretation -> a|p|n in depth
TripletLoss(margin)
```

Because `TNNetVolume` is stored depth-major
(`pos = ((SizeX*y)+x)*Depth + d`), the three per-X embeddings are already laid
out as consecutive depth chunks after the reshape — exactly the `a|p|n` layout
the loss head consumes. No transpose is needed. The pointwise convolutions
share weights across the three X positions, so the embedding function applied
to the anchor, positive, and negative is identical (a true siamese net).

## What the example prints

After training it prints a **per-class mean pairwise cosine-similarity matrix**.
Embeddings are unit-norm, so cosine similarity is just the dot product:

```
          class0  class1  class2  class3
class0     0.987  -0.614   0.219  -0.342
class1    -0.614   0.993  -0.420  -0.187
class2     0.219  -0.420   0.992  -0.594
class3    -0.342  -0.187  -0.594   0.991
```

Within-class entries (diagonal) come out near `1.0`; cross-class entries are
much lower. (With four classes on a 3-D unit sphere it is geometrically
impossible for all pairs to be maximally far apart, so an occasional small
positive off-diagonal is expected.)

## Plotting the embeddings

The run also writes `embeddings.csv` (columns `class,e0,e1,e2`). A quick 3-D
scatter, colored by class, shows the four clusters on the sphere:

```python
import pandas as pd, matplotlib.pyplot as plt
df = pd.read_csv("embeddings.csv")
ax = plt.figure().add_subplot(projection="3d")
ax.scatter(df.e0, df.e1, df.e2, c=df["class"], cmap="tab10")
plt.show()
```

## Building and running

```
lazbuild TripletEmbedding.lpi
../../bin/x86_64-linux/bin/TripletEmbedding
```

The run is deterministic (`RandSeed` is fixed), pure CPU, and finishes in
seconds. `embeddings.csv` is generated at runtime and is git-ignored.
