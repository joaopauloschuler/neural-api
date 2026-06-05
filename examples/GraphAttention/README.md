# Graph Attention (GAT vs. GCN on a noisy graph)

Semi-supervised, **transductive** node classification on a tiny synthetic
two-community graph, demonstrating the `TNNetGraphAttention` layer (a single-head
Graph Attention Network, Velickovic et al. 2018) and contrasting it head-to-head
against the fixed-weight `TNNetGraphConvolution` (GCN, Kipf & Welling 2017) on the
**same graph**.

## The question

A plain **GCN** aggregates every neighbour with **fixed**, degree-normalized
weights: it cannot tell a *good* (same-community) edge from a *bad*
(cross-community) one. A **GAT** instead **learns** a per-edge attention
coefficient from the node features, so it can **down-weight** misleading edges.
Does that help when the graph is noisy / partially **heterophilous**?

## The setup

A two-community **Stochastic Block Model (SBM)** graph is generated on the fly
(50 nodes, 25 per class). On top of the clean SBM edges we **inject many noisy
cross-community edges** — the heterophilous corruption — so that the wrong-class
neighbours actually outnumber the right-class ones for most nodes:

```
Same-class edges     : 135
Cross-class edges    : 209  (noisy / heterophilous)
```

Each node carries a short **class-indicative** feature vector (a clear class
signal buried in some noise). The features are informative enough that attention
can learn to attend to neighbours that "look like me" and ignore the wrong-class
ones — but the GCN's fixed edge weights **cannot** exploit that; it must average
every neighbour in regardless. Only **a handful of nodes are labelled** (5 per
class); the goal is to classify all the held-out (unlabelled) nodes.

Two models are trained head-to-head with the **same input, same shapes, same
training loop, and the same adjacency**. The only difference is the aggregator:

| Model | Layers | Edge weights |
|-------|--------|--------------|
| **GAT** | `Input → GraphAttention(8) → ReLU → GraphAttention(2) → per-node SoftMax` | **learned** per-edge attention |
| **GCN** | `Input → GraphConvolution(8) → ReLU → GraphConvolution(2) → per-node SoftMax` | **fixed** `Ahat = D^-1/2 (A+I) D^-1/2` |

## Result

```
Transductive accuracy on held-out nodes
  GAT (learned edge weights) :  90.00 %
  GCN (fixed edge weights)   :  85.00 %
  gap (GAT - GCN)            :   5.00 pp
```

With the injected cross-community edges, the fixed-weight GCN is forced to average
in a lot of wrong-class signal and loses accuracy. The GAT learns to **attend more
strongly to the same-class neighbours** (whose transformed features resemble the
node's own), down-weighting the noisy edges, and comes out ahead. **Learned edge
weighting buys robustness to noisy / heterophilous edges.**

(This is the regime where attention helps. When the graph is clean and the
features are weak, plain symmetric-normalized averaging is already excellent and a
GCN is the simpler, stronger choice — see the sibling `GraphNodeClassification`
example.)

## Multi-head GAT

The paper uses **K independent attention heads**: each head runs the whole
single-head mechanism with its own `W` and attention vector. Hidden layers
**concatenate** the heads along the feature axis (eq. 5); the output layer
**averages** them (eq. 6). This tree has no head-axis tensor, so multi-head GAT is
a **builder** that composes K independent `TNNetGraphAttention` layers fed from the
same source and sharing the same adjacency:

```pascal
NN.AddMultiHeadGraphAttention(Heads, cHidden, GAdj, {Concat=}true);  // hidden: concat
NN.AddLayer(TNNetReLU.Create());
NN.AddMultiHeadGraphAttention(Heads, 2, GAdj, {Concat=}false);       // output: average
```

More heads = more independent edge-weighting views averaged together, which is
more robust on the noisy graph:

```
Multi-head GAT (concat hidden, averaged output)
  1 head                     :  85.00 %
  4 heads                    :  92.50 %
  gain (4 heads - 1 head)    :   7.50 pp
```

## Attention-dropout

`TNNetGraphAttention.Create(Features, AttentionDropout, SuppressBias)` (and the
`pAttentionDropout` knob on the multi-head builder) enable the paper's **edge /
attention dropout** (Sec 2.2): the **normalized** per-edge coefficients
`alpha[i,j]` are randomly dropped at **training time only** (inverted-dropout
scaled by `1/(1-p)`, so the expected aggregation is unchanged). At inference the
layer is fully deterministic. `TNNet.EnableDropouts(true/false)` (called by `Fit`,
or manually around the training loop) gates it, exactly like `TNNetDropout`.

On the noisy-edge SBM it discourages over-committing to any single (possibly
cross-community) edge:

```
Attention-dropout ablation (4-head GAT on the noisy graph)
  dropout OFF (p=0.0)        :  92.50 %
  dropout ON  (p=0.3)        :  97.50 %
  effect (on - off)          :   5.00 pp
```

## The layer

`TNNetGraphAttention` is a **single-head** GAT aggregator over a
`(NumNodes, 1, FeatureDim)` node tensor (the same layout as
`TNNetGraphConvolution`). Per the GAT paper it computes, for every masked edge
`i ← j` (including the self-loop):

```
Z[n]      = H[n] · W (+ bias)                       (per-node pointwise map)
e[i,j]    = LeakyReLU( a_src · Z[i] + a_dst · Z[j] ) (shared attention vector a)
alpha[i,j]= softmax over j in N(i) of e[i,j]         (masked to graph edges)
Y[i]      = sum_j alpha[i,j] · Z[j]                  (attentional aggregation)
```

The caller supplies the raw 0/1 adjacency (no self-loops) via `SetAdjacency`,
exactly as the GCN does; the layer adds the `+I` self-loop and from then on
attends only over each node's masked neighbourhood. The pointwise weights `W` are
stored in the first `OutFeat` neurons (1×1-conv layout); the shared attention
vector `a = [a_src | a_dst]` (length `2·OutFeat`) lives in one extra appended
neuron. Both are trainable and round-trip through the ordinary serialization. The
adjacency mask is **not** serialized — re-call `SetAdjacency` after loading.

## Running

```
cd examples/GraphAttention
fpc -Fu../../neural -Mobjfpc -Sh -O2 GraphAttention.lpr
./GraphAttention
```

Pure CPU, single thread, runs in well under a second.
