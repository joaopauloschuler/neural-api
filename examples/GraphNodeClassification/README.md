# Graph Node Classification (GCN vs. feature-only MLP)

Semi-supervised, **transductive** node classification on a tiny synthetic
two-community graph, demonstrating the `TNNetGraphConvolution` layer (a spectral
Graph Convolutional Network, Kipf & Welling 2017).

## What it does

A two-community **Stochastic Block Model (SBM)** graph is generated on the fly:
60 nodes (30 per class), with edges **dense within** a community and **sparse
between** communities. Each node carries a short, deliberately **weak and noisy**
feature vector (a faint community hint buried in noise), and only a **handful of
nodes are labelled** (4 per class). The goal is to classify all the held-out
(unlabelled) nodes.

Because the labels are hidden but the features and the graph are fully visible at
forward time, this is the classic *transductive* setting.

## The headline: message passing carries the signal

Two models are trained head-to-head with the **same layer stack, same shapes,
same parameter budget, and same training loop**. The only difference is the
adjacency handed to the two `TNNetGraphConvolution` layers:

| Model | Layers | Adjacency `Ahat` | Message passing |
|-------|--------|------------------|-----------------|
| **GCN** | `Input → GraphConvolution(8) → ReLU → GraphConvolution(2) → per-node SoftMax` | symmetrically-normalized SBM adjacency | **ON** |
| **MLP baseline** | identical | **identity** (`Ahat = I`) | **OFF** (per-node features only) |

With `Ahat = I` the GCN's neighbour-aggregation step is a no-op, so each node is
classified purely from its own (weak) features — a per-node MLP. The contrast
isolates exactly one thing: whether mixing a node's features with its neighbours'
recovers the class.

## Result

```
Transductive accuracy on held-out nodes
  GCN (message passing ON) : 100.00 %
  MLP (features only)      :  50.00 %
  gap (GCN - MLP)          :  50.00 pp
```

The per-node features alone are too weak to separate the classes (the MLP sits at
chance). Once the GCN aggregates over the graph, a node's noisy features are
averaged with its (same-class) neighbours' features, the noise cancels, and the
community structure becomes linearly separable — the GCN reaches 100 % on the
held-out nodes. **The graph structure, not the features, carries the signal.**

## The layer

`TNNetGraphConvolution` implements the GCN propagation rule

```
H' = Ahat · (H · W) (+ bias)
```

in two steps: a per-node pointwise linear map `H·W` over the feature axis (the
same weight layout / gradient path as a 1×1 / pointwise convolution, so nodes are
never mixed by `W`), followed by a left-multiply by the constant
symmetrically-normalized adjacency `Ahat = D^-1/2 (A + I) D^-1/2` (the only step
that mixes across nodes). The caller supplies the raw 0/1 adjacency `A` (no
self-loops) via `SetAdjacency`; the layer adds the `+I` self-loop and normalizes
internally. The adjacency is a fixed buffer (no gradient) and is **not**
serialized — re-call `SetAdjacency` after loading a saved network.

## Running

```
cd examples/GraphNodeClassification
fpc -Fu../../neural -Mobjfpc -Sh -O2 GraphNodeClassification.lpr
./GraphNodeClassification
```

Pure CPU, single thread, runs in a couple of seconds.
