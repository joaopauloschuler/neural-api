# Hyena operator (attention-free implicit long convolution)

A long-range recall bake-off of the attention-free **Hyena operator**
(`TNNet.AddHyenaOperator`, built on the leaf layer `TNNetImplicitLongConv`)
against a single-head self-attention block, on a synthetic task where the
implicit long convolution's *global* receptive field should shine.

Reference: Poli et al. 2023, *Hyena Hierarchy: Towards Larger Convolutional
Language Models* (arXiv:2302.10866).

## The primitive

`TNNetImplicitLongConv` is a **causal depthwise long convolution** over a
`(SeqLen, 1, Depth)` sequence whose per-channel filter spans the WHOLE sequence
(length `SeqLen`). Crucially the filter is not stored tap-by-tap — it is
generated IMPLICITLY by a tiny shared MLP over positional features and then
multiplied by a learnable exponential-decay window:

```
phi[p]      = [1, p_norm, sin(2*pi*p_norm)]          (frozen positional features)
act[p]      = tanh(W1 * phi[p] + b1)                 (shared hidden layer, width H)
base[p,c]   = W2[c] . act[p] + b2[c]
decay[p,c]  = exp(-softplus(logDecay[c]) * p_norm)
h[p,c]      = base[p,c] * decay[p,c]                 (the implicit filter)
y[t,c]      = sum_{p=0..t} h[p,c] * x[t-p,c]         (causal long conv)
```

Because the filter is parametrized FROM POSITIONS rather than per tap, one set
of weights covers any `SeqLen` — so the parameter count does **not** grow with
the sequence length. This is what distinguishes it from `TNNetCausalConv1D`
(a short, fixed-length learned kernel) and `TNNetDiagonalSSM` (a per-channel
linear recurrence).

## The operator (builder)

`TNNet.AddHyenaOperator(d_model, Hidden)` assembles the order-2 Hyena recurrence
from existing primitives, all token-wise (1x1 conv) so the sequence axis is
preserved:

```
v, g1, g2 = three PointwiseConvLinear projections of the block input
y = OutProj( g2 .* LongConv( g1 .* v ) )      (.* = data-controlled gating)
```

## The task

Each example is a length-`SeqLen` sequence. Channel 0 carries a random payload
that is non-zero at exactly ONE early random position; channel 1 is a query
marker that is 1 only at the LAST position. The target is to copy the early
payload to the last position's channel-0 output. Solving it requires moving
information across the whole sequence.

## Running

```
lazbuild HyenaOperator.lpi      # or: fpc -O3 -Mobjfpc -Sh -Fu../../neural HyenaOperator.lpr
./HyenaOperator
```

Pure CPU, ~11 s. Example output:

```
arm          weights   train recall MSE   test recall MSE
HYENA            152     0.001137         0.001355
ATTENTION        688     0.071599         0.089982
```

The attention-free Hyena operator generalises better on this long-range recall
toy while using fewer trainable weights. It is a drop-in attention-free block
for the downstream `../gpt-3-for-pascal` decoder.
