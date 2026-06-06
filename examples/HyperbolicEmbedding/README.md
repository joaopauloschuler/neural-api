# Hyperbolic Embedding: recovering tree distances in the Poincaré ball

A tiny self-contained demo that embeds the nodes of a small balanced binary
**tree** and shows that a **hyperbolic** (Poincaré-ball) embedding recovers the
tree's shortest-path distances markedly better than a **parameter-matched
Euclidean** embedding.

## Why hyperbolic?

A tree's node count grows *exponentially* with depth, and so does the volume of
a hyperbolic ball with its radius. Trees therefore embed into hyperbolic space
with almost no distortion, while Euclidean space (polynomial volume growth)
cannot hold an exponentially branching tree without crowding the leaves
together. This is the classic result of Nickel & Kiela 2017,
[*Poincaré Embeddings for Learning Hierarchical Representations*](https://arxiv.org/abs/1705.08039),
reproduced here on a toy tree small enough to train in a few seconds on CPU.

## What it does

* **Tree**: a complete binary tree of depth 4 (31 nodes). Ground-truth distance
  between two nodes = the number of edges on the tree path between them
  (lowest-common-ancestor on heap-style node ids).
* **Model A (hyperbolic)**: `TNNetInput(N) -> TNNetHyperbolicLinear(2, c)`. Each
  node id (one-hot) maps to a point inside the Poincaré ball; the distance
  between two embeddings is the curvature-`c` Poincaré distance
  `dist_c(a,b) = (2/√c)·atanh(√c·‖(-a) ⊕_c b‖)` — the same formula
  `TNNetHyperbolicDistance` computes against its prototype bank (here we apply it
  between two *learned* embeddings and hand-seed the gradient through the same
  Möbius math).
* **Model B (Euclidean baseline)**: `TNNetInput(N) -> TNNetFullConnectLinear(2)`,
  plain `‖a−b‖`. Same parameter budget (one `N×2` weight matrix; the hyperbolic
  Möbius bias is suppressed so the counts match exactly).
* **Training**: sample node pairs, regress the embedded distance to the (scaled)
  tree path length by MSE. The loss couples two inputs, so training is hand-rolled
  (two forward passes per pair, seed the embedder's `OutputError` with the analytic
  `dMSE/dembedding`, backprop each) — the custom-loss pattern from
  `examples/SparseAutoencoder`. `SetBatchUpdate(True)` is required.

Both models get the **same** data, mini-batch size and epoch budget; only the
optimiser step differs (hyperbolic SGD is boundary-sensitive and uses a gentler
schedule than flat Euclidean SGD). The analytic Poincaré-distance gradient was
checked against finite differences during development.

## Example output

```
HyperbolicEmbedding: tree-distance recovery, hyperbolic vs Euclidean
  complete binary tree depth=4  nodes=31  embed dim=2  c=1.00

Training hyperbolic model...
  [HYP] epoch    1   MSE=  0.9125   corr=-0.031
  [HYP] epoch  300   MSE=  0.0514   corr= 0.944
  [HYP] epoch 1500   MSE=  0.0516   corr= 0.942
Training Euclidean model...
  [EUC] epoch    1   MSE=  2.4398   corr= 0.106
  [EUC] epoch  300   MSE=  0.2473   corr= 0.714
  [EUC] epoch 1500   MSE=  0.1550   corr= 0.833

==== FINAL (all node pairs, embedded distance vs tree path length) ====
  Hyperbolic : MSE=  0.0516   Pearson corr=0.942
  Euclidean  : MSE=  0.1550   Pearson corr=0.833

  VERDICT: hyperbolic WINS - lower MSE (0.0516 vs 0.1550) and higher/equal
           correlation (0.942 vs 0.833) at the SAME param budget.
```

The hyperbolic embedding fits the tree distances roughly **3× lower MSE** and a
substantially higher correlation than the Euclidean one at the identical 2-D
parameter budget — the expected headline. The whole run takes a few seconds on
2 CPU cores.

## Build & run

```
lazbuild HyperbolicEmbedding.lpi
../../bin/x86_64-linux/bin/HyperbolicEmbedding
```

## Trainable curvature (optional)

This demo fixes the ball curvature `c` to `CURVATURE`. `TNNetHyperbolicLinear`
also supports learning `c` as a single trainable scalar — pass the extra
`pLearnCurvature` flag:

```pascal
// fixed c (default, exactly as used in this demo):
TNNetHyperbolicLinear.Create(DIM, CURVATURE, 1);
// trainable c, starting from CURVATURE:
TNNetHyperbolicLinear.Create(DIM, CURVATURE, 1, {pLearnCurvature=}true);
```

In learnable mode `c = CMIN + (CMAX-CMIN)·sigmoid(raw)` (bounded to `(0.01, 4.0)`
so it stays strictly positive), stored as one extra 1-weight neuron that the
ordinary optimizer / save-load machinery handles transparently. The exact
`dL/draw` is back-propagated through the `log₀ → matmul → exp₀ → Möbius-add`
chain, so curvature is learned jointly with the matrix and bias. (This demo
keeps `c` fixed because its hand-coded distance loss reads `CURVATURE` as a
constant; in a model where the layer's own output drives the loss, just flip the
flag.)

## Related layers

* `TNNetHyperbolicLinear` — the Poincaré-ball hyperbolic dense layer used as the
  embedder here.
* `TNNetHyperbolicDistance` — a readout head producing Poincaré distances from an
  input point to `K` learnable prototypes (same distance formula used here).
