# Gated Linear Attention Block — residual + LayerNorm + SwiGLU FFN tower

Demonstrates `TNNet.AddGatedLinearAttentionBlock`, a full transformer-style
**block** that wraps the gated-linear-attention time mixer
(`TNNet.AddGatedLinearAttention`, itself around the `TNNetGatedLinearAttention`
leaf — Yang et al. 2023, [arXiv:2312.06635](https://arxiv.org/abs/2312.06635))
in a pre-norm residual structure plus a token-wise SwiGLU feed-forward residual.
It mirrors `AddTransformerEncoderBlock` but swaps the multi-head self-attention
arm for gated linear attention.

## The builder

```
function TNNet.AddGatedLinearAttentionBlock(d_ff: integer;
  PreNorm: boolean = true; NormClass: TNNetLayerClass = nil): TNNetLayer;
```

Over a `(SeqLen, 1, d_model)` sequence (`SizeY=1`) it wires two residual
sub-blocks (PreNorm form):

```
x := x + GLA(LayerNorm(x))                                    (time-mixing residual)
x := x + FFN(LayerNorm(x))                                    (token-wise SwiGLU FFN)
  FFN(z) = PointwiseConvLinear(d_model)( SwiGLU( PointwiseConvLinear(2·d_ff)(z) ) )
```

`PreNorm=False` places the norm *after* each residual sum instead. `d_model` is
inferred from the input depth; the output shape matches the input so blocks
**stack into a deep tower**. Pointwise (`1×1`) convs are used everywhere so the
token axis is preserved (a `FullConnect` would flatten/mix the whole sequence).
`NormClass` selects the norm class at every norm slot (defaults to
`TNNetLayerNorm`; e.g. `TNNetRMSNorm`, `TNNetDyT`).

This is a pure **builder** composing existing layers — it adds no new leaf class
and so needs no new save/load format.

## The task (overwrite recall)

Each sequence presents several distinct `(key, value)` write tokens; one key is
then **re-written with a new value** (an overwrite); finally that key is queried
and the network must return its **most recent** value — a natural fit for GLA's
data-dependent per-channel forget gate. Two arms on the same data / step budget:

- **Bare mixer**: `1×1` projection → `AddGatedLinearAttention` → `1×1` readout.
- **Block tower**: `1×1` projection → `AddGatedLinearAttentionBlock` × 3 →
  `1×1` readout. The same mixer, but each layer is residual-wrapped with
  LayerNorm + a SwiGLU FFN and several are stacked for depth.

## Headline result

```
bare GLA mixer       params = 2064
GLA block tower (x3) params = 11472

eval over 400 held-out recall sequences:
  bare GLA mixer        : recall MSE = 0.03317   exact-recall acc = 81.0%
  GLA block tower (x3) : recall MSE = 0.00743   exact-recall acc = 100.0%
```

The residual + LayerNorm + FFN scaffolding the block adds is what lets the mixer
stack deep and train stably: the 3-block tower reaches perfect exact recall and
~4.5× lower MSE than the single bare mixer.

## Running

```
lazbuild GatedLinearAttentionBlock.lpi
../../bin/x86_64-linux/bin/GatedLinearAttentionBlock
```

Pure CPU, tiny dimensions; finishes in well under a minute (~8 s, ~6 MB on
2 cores).
