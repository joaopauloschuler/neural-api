# Matryoshka Embedding (nested-prefix representation learning)

A CPU micro-example of **Matryoshka Representation Learning** (Kusupati et al.,
*NeurIPS 2022*). A single encoder produces one `d = 64` embedding whose nested
**prefixes** `{8, 16, 32, 64}` are *each* independently usable as a classifier
feature. The total training loss is the **sum of K softmax cross-entropy
losses**, one computed on each truncated prefix, so the early coordinates are
forced to pack the coarsest / most important information and the later
coordinates only refine it.

## Why this is different from the other embedding examples

`TripletEmbedding`, `InfoNCEContrastive`, `ArcFaceEmbedding` and
`CosineEmbeddingSiamese` all learn **one fixed-width vector**. Matryoshka learns
an **elastic, sub-dimension-addressable** representation: from a *single* model
you get a whole accuracy-vs-width curve, and at retrieval time you can simply
**truncate** the vector to trade accuracy for cost with **zero retraining**
(adaptive-cost retrieval for free).

## How the nested prefixes are wired

```
Input(1,1,6)
  -> PointwiseConvReLU(48) -> PointwiseConvReLU(48)
  -> PointwiseConvLinear(64)                         [the shared embedding]
  for each prefix p in {8,16,32,64}:
     SplitChannels(0, p) -> FullConnectLinear(8) -> SoftMax   [head p]
  DeepConcat([head_8, head_16, head_32, head_64])    [final output, 4*8 wide]
```

The training target is the **concatenation of K identical one-hot labels**. The
network's softmax + cross-entropy backward (`OutputError = output - target` on
the concatenated heads) yields exactly the **sum of the per-prefix softmax CE
gradients**. The key asymmetry: embedding coordinate `0` receives gradient from
**all four** heads while coordinate `63` receives gradient only from the
64-head. That asymmetry *is* the Matryoshka nesting that makes the early
coordinates dominant.

`TNNetSplitChannels(0, p)` is the clean prefix-slice primitive — no new layer
class was needed.

## The prefix-loss-weighting pitfall

The per-prefix losses must be combined so the **smallest prefix does not
dominate the gradient**. Here every head emits the *same* number of classes (8
logits), so the logit-level gradient magnitudes are comparable and **uniform**
weighting (`cHeadWeight = (1,1,1,1)`) is correct. The trap is weighting a
prefix's loss by its **dimension**, or summing *unnormalised* losses of heads
with very different output sizes: then the small prefix's gradient swamps
coordinates `0..7`, over-training that head while the wide prefix barely learns.
Keep the heads' loss scale comparable.

## Dataset choice — honest budget note

This uses a **tiny synthetic 8-class Gaussian-blob task** (blobs in 6-D,
projected through the encoder). **MNIST was not used**: the demo trains *three*
models (the Matryoshka model plus a dedicated fixed-8 and a dedicated fixed-16
baseline) single-threaded from scratch, and a from-scratch MNIST loop would blow
the ~5-minute budget. The synthetic task lands the nested-prefix headline
cleanly and deterministically (fixed `RandSeed`).

## Result actually observed

Wall clock: **~6.5 s** on 2 CPU cores, single-threaded numerics.

```
=== Accuracy vs embedding width (test set) ===

  width  matryoshka-prefix   dedicated-baseline
  -----  -----------------   ------------------
     8        0.913              0.913
    16        0.925              0.909
    32        0.916                (none)
    64        0.919                (none)

  ASCII accuracy-vs-width curve (Matryoshka prefixes):
   d= 8 |##############################################  91.3%
   d=16 |##############################################  92.5%
   d=32 |##############################################  91.6%
   d=64 |##############################################  91.9%
```

**Headline:** the Matryoshka 8-dim and 16-dim **prefixes** (0.913 / 0.925) are
right on top of the **dedicated** fixed-8 / fixed-16 baselines (0.913 / 0.909) —
the nested prefixes lose essentially nothing versus models trained specifically
at those widths, so you get adaptive-cost retrieval for free from one model.

Two honesty notes:

* The Matryoshka mean training loss prints ~5.6 because it is the **sum of 4
  head losses** (~1.4 each); the single-head baselines print ~0.1. Different
  scales, both converged — do not compare the raw loss numbers across models.
* On this easy 6-D blob task even an 8-dim prefix nearly saturates, so the
  curve is fairly **flat** (it does rise 8 -> 16). On a harder dataset the
  width curve would be steeper; the close-to-baseline property is the point
  being demonstrated here, and it holds.

## Build & run

```
lazbuild MatryoshkaEmbedding.lpi
stdbuf -oL -eL ../../bin/x86_64-linux/bin/MatryoshkaEmbedding
```
