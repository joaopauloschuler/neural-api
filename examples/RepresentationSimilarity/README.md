# RepresentationSimilarity

Tiny example for `TNNet.RepresentationSimilarityReport`, a forward-only
**linear-CKA** (Centered Kernel Alignment, Kornblith et al. 2019)
representation-geometry diagnostic.

It answers *"how does the representation reshape itself with depth, and which
layers do redundant work?"* by computing the linear CKA similarity between
every pair of layer activations over a shared probe batch.

## What is linear CKA?

For each probeable layer `l` the report runs the probe batch (`N` probes)
through the net and flattens each probe's activation at layer `l` to a row of
an `N x D_l` matrix `X_l`. It **column-centers** the features (subtracts each
feature's mean across the `N` rows), forms the `N x N` Gram matrix
`K_l = X_l X_l^T` (cheap when `D_l` is large and `N` small), double-centers it,
and then for two layers `i`, `j`:

```
CKA(i, j) = <K_i, K_j>_F / ( ||K_i||_F * ||K_j||_F )
```

The result is in `[0, 1]`, **invariant to orthogonal rotation and isotropic
scaling** of the features — so it compares *representations*, not raw
coordinates. The self-CKA diagonal is `1.0` by construction (the built-in
correctness check) and the matrix is symmetric.

## What the report shows

- the full `L x L` CKA matrix as an ASCII heatmap, glyph-shaded by similarity
  band (`' .:-=+*#%@'`, brightest = most similar);
- the **adjacent-layer** similarity vector `CKA(l, l+1)` down the stack — a
  high value flags a near-pass-through *redundant depth* layer, a sharp dip
  flags where the representation genuinely reorganizes;
- the single **most-redundant layer pair** (highest off-diagonal CKA — a
  merge / prune candidate);
- the **block structure**: contiguous runs of layers whose mutual CKA stays
  above `BlockThreshold` (the "representational stages" of the net);
- a one-line **verdict**: *"K of L layers are near-duplicates of a neighbour"*.

Supplying a second net (`RepresentationSimilarityReport(NN, Probes, OtherNet)`)
switches to **cross-CKA** `CKA(layer_i of NN, layer_j of OtherNet)` over the
same probe batch — *"do these two trained nets learn the same intermediate
features?"* (the headline use case in the CKA paper: comparing widths / depths
/ seeds). The cross matrix is rectangular and is **not** `1.0` on its diagonal.

## What this example demonstrates

The program builds a deliberately **over-deep** ReLU MLP
(`16 -> 16 x6 -> 1`) on a task that needs essentially one feature
(`y = max(0, w.x)`, a single rectified-linear teacher direction) and runs
three contrasts:

1. **FRESH INIT** — untrained layers barely transform their input, so the
   adjacent-layer CKA is already high.
2. **AFTER TRAINING** — a clearer block structure emerges, yet the over-deep
   middle layers stay near-duplicates of one another (in the shipped run the
   verdict reports *6 of 8 layers* are near-duplicates of their successor):
   the *"depth is wasted"* signal lights up.
3. **CROSS-CKA** — the trained net against a second fresh-init net of the same
   shape: the input layers match (`CKA = 1.0`, same input) while the learned
   intermediate features diverge.

The computation is **pure forward-only**: weights are never touched and no
backward pass is run. `MaxSamples` caps the probe count used (default 64) and
over-wide layers (flat `Output` larger than `MaxFeatDim`, default 2048) are
skipped to bound the Gram build.

### How this differs from the neighbours

- **`NeuronCorrelationReport`** measures *intra-layer* neuron-pair Pearson
  redundancy *within one* layer. This measures *whole-layer representation*
  similarity *between* layers/nets and is rotation-invariant (raw Pearson is
  not).
- **`LinearProbeReport`** asks "is the label linearly decodable here?" against
  a target. This needs no labels — it asks "are these two feature spaces the
  same?".
- **`DiffArchitecture`** diffs the static layer list. This compares the
  *learned activations*.

## Build & run

```
cd examples/RepresentationSimilarity
lazbuild RepresentationSimilarity.lpi
../../bin/x86_64-linux/bin/RepresentationSimilarity
```

Total runtime is well under a minute (pure CPU, synthetic data, no download).
