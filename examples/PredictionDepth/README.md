# PredictionDepth

Tiny example for `TNNet.PredictionDepthReport`, the per-**example** difficulty
diagnostic based on the **prediction depth** of Baldock, Maennel & Neyshabur 2021
(*Deep Learning Through the Lens of Example Difficulty*). It answers
**"at how deep a layer does the network actually make up its mind about THIS
example?"** — a per-*sample* resolution depth, not a per-layer aggregate. Easy,
well-separated examples are decided early (shallow depth); hard, ambiguous or
mislabelled examples stay contested until the last layers (deep depth).

The program builds a small softmax classifier
(`2 -> FC12+ReLU -> FC12+ReLU -> FC12+ReLU -> FC4 -> SoftMax`) on a synthetic
4-class 2D-blob problem and prints `TNNet.PredictionDepthReport` on the **same**
labelled support batch + query batch for two networks:

1. **RUN 1**: a freshly-initialised network (no training). Nothing is decided
   early, so the prediction depths pile up at the **last** layer (a right-skewed
   histogram).
2. **RUN 2**: the same architecture after a short training run. The mass shifts
   **shallow** for the well-separated clusters, while the **ambiguous
   between-blob** query subset keeps a **deep tail**, and the
   incorrect-minus-correct mean-depth gap is positive (errors decide deeper —
   the literature's headline result).

It then runs a **built-in correctness check**: feeding the support set as its
**own** queries. Each point is its own nearest neighbour (cosine distance 0), so
the final-layer k-NN vote matches the network argmax for ~every sample
(agreement ~1.0) and every depth is 0.

## How it works (forward-only, non-parametric)

For each query the report:

- runs one forward pass over the support and query batches and **snapshots**
  every trainable layer's flattened activation;
- at every layer, takes a **k-NN vote** (default `K=5`, **cosine** distance) over
  the support activations at that same layer — the predicted class at that depth;
- defines the **prediction depth** of a query as the index of the **shallowest
  layer after which the k-NN vote agrees with the network's final argmax and
  never disagrees again** (all deeper layers confirm it).

There is **no probe fit and no matrix solve** — only distances. The backbone is
pure forward-only and never modified.

## What it reports

- a 10-bin ASCII histogram of prediction depth across the query batch, plus
  **mean / median** depth;
- the per-layer **newly-resolved** count — how many queries first lock in their
  final class at each layer (a depth-vs-layer profile of where examples get
  decided);
- the `K` **deepest (= hardest)** query indices, a ready-made hard-example /
  relabel-candidate queue;
- with query labels supplied, a **correctness cross-tab**: the mean prediction
  depth of correctly vs incorrectly classified queries.

### Feature-dimension cap

A layer whose flat activation exceeds `MaxFeatDim` (default **256**) is
deterministically random-projected down to `MaxFeatDim` features (a fixed-seed
sparse sign projection — the same trick `LinearProbeReport` uses) before the
cosine distances are computed, bounding the k-NN cost on wide layers. (The tiny
layers in this example stay well under the cap, so no projection fires here.)

## Build & run

```
cd examples/PredictionDepth
lazbuild PredictionDepth.lpi
../../bin/x86_64-linux/bin/PredictionDepth
```

Pure CPU, no dataset download, total runtime well under a minute.
