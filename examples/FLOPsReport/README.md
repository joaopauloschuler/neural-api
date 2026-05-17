# FLOPsReport

Demonstrates `TNNet.CountFLOPsPerLayer`: a static, per-layer estimator of
forward-pass FLOPs for a `TNNet`.

## What it does

1. Builds a tiny MLP (`64 -> 128 -> 64 -> 10 + softmax`) and prints its
   per-layer FLOPs table.
2. Builds a tiny CIFAR-style ConvNet (3 conv+pool stages, global avg
   channel, softmax head) and prints the same table.
3. Prints a one-line side-by-side total comparison.

No training, no forward pass: the estimator walks `NN.Layers` and uses each
layer's recorded output shape plus the convolutional kernel/stride/depth
parameters.

## What is counted

| Layer family | Approximation |
| --- | --- |
| `TNNetFullConnect*` | `2 * In * Out` (mul + add per weight), plus `~1*Out` for ReLU, `~4*Out` for sigmoid/tanh |
| `TNNetConvolution*`, `TNNetPointwiseConv*` | `2 * Hout * Wout * Dout * (Fy * Fx * Din)` plus activation cost |
| `TNNetDepthwiseConv*` | `2 * Hout * Wout * Dout * Fy * Fx` (no channel sum) |
| `TNNetMaxPool` / `TNNetAvgPool` / `TNNetMinPool` | `Hout * Wout * Dout * PoolSize^2` |
| `TNNetMaxChannel` / `TNNetAvgChannel` | `InputSize` (one op per input element) |
| Elementwise activations (`TNNetReLU`, `TNNetReLU6`, ...) | `Output.Size`, or `4 * Output.Size` for exp/tanh-family (`TNNetSwish`, `TNNetGELU`, `TNNetSigmoid`, `TNNetTanhExp`) |
| `TNNetLayerNorm` / `TNNetRMSNorm` / `*StdNormalization` | `3 * Output.Size` (mean, var, normalize) |
| `TNNetScaledDotProductAttention` | `4 * S^2 * d_k + 5 * S^2` (QK^T + softmax + PV) |
| `TNNetDropout` | `Output.Size` (mask multiply) |
| `TNNetEmbedding`, `TNNetIdentity`, `TNNetPad*`, `TNNetReshape`, `TNNetSplit*`, `TNNetConcat*`, `TNNetInput*` | `0` |
| Anything else | `0` and incremented in the `uncovered classes` tally |

## Output

For each network the report prints one row per layer:

```
Idx   Class                              OutShape                    FLOPs    % total
------------------------------------------------------------------------------
0     TNNetInput                         (32,32,3)                       0       0.0%
1     TNNetConvolutionReLU               (32,32,16)                 901120      45.5%
...
------------------------------------------------------------------------------
TOTAL FLOPs: 1981440   (uncovered classes: 0)
```

The `uncovered classes` count flags layer types the estimator does not
model precisely - useful when adding new layers to the library.

## Running

```
cd examples/FLOPsReport
lazbuild FLOPsReport.lpi
../../bin/<arch>/bin/FLOPsReport
```

Runs in under a second on CPU.
