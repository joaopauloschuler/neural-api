# Conformer (convolution-augmented transformer)

A tiny, self-contained, pure-CPU demo of `TNNet.AddConformerBlock`, the
convolution-augmented transformer block of Gulati et al. 2020,
[*Conformer: Convolution-augmented Transformer for Speech Recognition*](https://arxiv.org/abs/2005.08100).

## The block

A Conformer block is a "macaron" sandwich: two **half-step** feed-forward
modules wrap a multi-head self-attention module and a convolution module, every
sub-module a pre-norm residual, with a final LayerNorm:

```
x := x + 0.5 * FFN(x)        first half-step feed-forward
x := x + MHSA(x)             multi-head self-attention   (GLOBAL mixing)
x := x + ConvModule(x)       1-D convolution over time   (LOCAL mixing)
x := x + 0.5 * FFN(x)        second half-step feed-forward
x := LayerNorm(x)
```

`AddConformerBlock(Heads, d_ff, ConvKernelSize)` is a **builder**: it composes
the block entirely from existing serializable primitives, so it needs no new
layer class and a net containing it round-trips through
`SaveToString`/`LoadFromString`. The composed primitives are:

* `TNNetLayerNorm` for every pre-norm / final norm slot;
* `TNNetPointwiseConvLinear` for all per-token (1x1-over-depth) projections
  (the FFN expand/project, the QKV slab, the conv-module pointwise convs);
* `AddMultiHeadSelfAttention` for the self-attention module;
* `TNNetSwish` (SiLU) as the FFN/conv activation;
* `TNNetGLU` for the conv-module gating;
* `TNNetCausalConv1D` for the 1-D convolution over the time axis (causal/SAME,
  length-preserving);
* `TNNetSum` for the residual adds and `TNNetMulByConstant(0.5)` for the macaron
  half-step scaling.

> **Note on the convolution.** The original Conformer uses a *depthwise*
> (per-channel) 1-D convolution. This library has no per-channel
> 1-D-over-sequence primitive, so the closest available shape-preserving 1-D
> sequence conv — `TNNetCausalConv1D`, a full conv that **does** mix channels
> across the kernel window — is used instead. This is documented in the builder's
> source comment.

The block is **shape-preserving** over a `(SeqLen, 1, d_model)` input (`SizeY=1`),
so blocks stack directly.

## The toy task

To show the block actually uses **both** its conv and its attention pathway, the
demo is a per-token tagging task whose label at position `t` is the **XOR** of:

* **a LOCAL feature** — does the adjacent bigram `(S[t], S[t+1])` equal `(3, 4)`?
  A 1-D conv over the time axis detects this two-token pattern directly.
* **a GLOBAL feature** — is the far-away first token `S[0] == 1`? A single
  long-range bit every position must condition on; attention routes it to every
  token in one hop.

`target[t] = LocalBigram(t) XOR (S[0] == 1)`. Because it is an XOR, a model that
recovers only the local feature flips its answer on every sequence with `S[0]==1`,
and an attention-only model can never spot the local bigram — so high accuracy
**requires both pathways**. A per-token softmax readout (no pooling) keeps every
position's gradient alive.

## Running

```
lazbuild examples/Conformer/Conformer.lpi
bin/x86_64-linux/bin/Conformer
```

It trains in well under a minute on two CPU cores and self-checks with a PASS/FAIL
gate (overall per-token accuracy > 90%). A representative run:

```
Per-token test accuracy (overall)        :  97.18%
  ... at LOCAL bigram-firing positions   :  93.00%   (conv pathway)
  ... on GLOBAL-bit-set sequences        :  84.44%   (attention pathway)
GATE (acc > 90%) : PASS  -- the Conformer used BOTH conv and attention.
```

The high accuracy at *both* the local-bigram positions and the global-bit-set
sequences is the point: the convolution module and the self-attention module are
each pulling their weight.
