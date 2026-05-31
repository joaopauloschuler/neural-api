# TinyTransformerFFN

The **feed-forward half** of a transformer block, trained end-to-end on a
tiny pure-CPU synthetic task. There is **no multi-head attention** here on
purpose: this demo isolates the FFN sub-block — the part of a transformer
layer that mixes *features within each token*, not information *across
tokens* — and shows that a deep pre-norm residual stack of those sub-blocks
composes and trains stably.

## What it shows

Each transformer FFN sub-block is the LLaMA-style pre-norm residual

```
y = x + SwiGLU_FFN( RMSNorm(x) )
```

assembled from two existing library builders:

- `TNNet.AddRMSNormResidual([...])` — `y = x + Sublayer(RMSNorm(x))`
- `TNNet.AddSwiGLUFeedForward(D, H, D)` — `Dense(2H) -> SwiGLU -> Dense(D)`

The example stacks `NUM_BLOCKS = 4` of these blocks on a `d_model = 16`,
`hidden = 32` tensor and a small per-token output head:

```
TNNetInput(1, 1, 16)                         # one token = 16 features, in Depth
  4 x [                                       # FFN-only transformer stack
    RMSNorm                                   #   pre-norm
    FullConnectLinear(1,1,2*32) -> SwiGLU     #   gate||value -> value*SiLU(gate)
    FullConnectLinear(16) -> Reshape(1,1,16)  #   back to d_model, into Depth
    + residual                                #   x + branch
  ]
  TNNetPointwiseConvLinear(16)               # per-token output head over Depth
```

### Shape gotcha (handled here)

The residual `RMSNorm` and the residual `TNNetSum` operate on the **Depth**
axis, so the block-carrying tensor is `1 x 1 x 16` (features in Depth). But
`AddSwiGLUFeedForward` ends in a `TNNetFullConnectLinear`, which lays its
output along the **X** axis (shape `16 x 1 x 1`). So a
`TNNetReshape(1, 1, 16)` is appended **inside** the residual sublayer list to
move the feature vector back into Depth before the residual Sum. Without it
you get `Size doesn't match ... Should be:(1 1 16) It is:(16 1 1)`.

## The task — per-token denoising

Each token is a feature vector `v` in `R^16`. The **clean** target for that
token is a fixed deterministic nonlinear elementwise map of two of its own
coordinates:

```
target[d] = 0.6 * tanh(1.5 * v[d]) + 0.4 * v[d] * v[d+1]
```

The network sees a **noised** copy of `v` (Gaussian-ish noise, std 0.15) and
must reconstruct the clean target. Because every token's target depends only
on that token's own features, a per-position FFN solves it with **no
attention**. The trivial "do nothing, echo the noised input" baseline is
printed for comparison.

## Build & run

```
lazbuild TinyTransformerFFN.lpi
../../bin/x86_64-linux/bin/TinyTransformerFFN
```

Pure CPU, no external data, runs in well under a minute (~15 s on a modern
laptop, single-threaded for determinism). The run is non-interactive.

## Expected output sketch

```
TinyTransformerFFN: FFN-only (no-attention) transformer half-block.
Stack: 4 x [ x + SwiGLU_FFN(RMSNorm(x)) ],  d_model=16, hidden=32.
...
Trainable parameter count: 6464

Baseline val MSE (echo noised input, no learning): 0.074623

Train-loss trace (every 5th epoch + final):
  epoch   1: train_err = 2.443710
  ...
  epoch  60: train_err = 1.283895

Final validation MSE (clean target): 0.012639
  vs baseline echo MSE             : 0.074623   (5.90x better)

Sample token (first 6 of 16 channels):
  ch | noised_in  predicted   clean_tgt
   0 |    0.6612     0.5139     0.3346
   ...
```

The deep pre-norm residual stack converges stably to roughly **6x lower MSE
than the do-nothing baseline**, with predictions tracking the clean target
rather than the noised input — demonstrating that the transformer FFN
half-block trains end-to-end on its own.
