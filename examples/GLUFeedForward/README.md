# GLUFeedForward

Smallest possible end-to-end demo of the transformer feed-forward
block: a `Dense -> GLU -> Dense` sandwich trained on a synthetic
regression target. No attention, no embedding, no positional
encoding. Just `TNNetGLU` wired between two fully-connected layers
so you can see the gated activation work in isolation.

## What it shows

The model is the original "GLU FFN" from the gated-activations
paper (https://arxiv.org/abs/2002.05202), shrunk down to CPU-friendly
sizes:

```
TNNetInput(4)                            # 4 raw features
  -> TNNetFullConnectLinear(1, 1, 32)    # pack gate || value (2 * d_hidden)
  -> TNNetGLU                            # value * Sigmoid(gate) -> depth 16
  -> TNNetFullConnectLinear(1)           # regression head
```

`TNNetGLU` splits its input depth in half: the second half is passed
through `Sigmoid(x)` and used to gate the first half via element-wise
multiplication, i.e. `GLU(a, b) = a * sigmoid(b)`. Output depth is
therefore half of input depth, which is why the projection above
produces `2 * d_hidden = 32` channels.

This is the plain GLU gate. The sibling examples use richer gates:
`SwiGLU` replaces the sigmoid with the Swish gate
`Swish(b) = b * sigmoid(b)`, and `GEGLU` uses a GELU gate. The wiring
and depth bookkeeping are identical across all three.

The synthetic target is

```
y = sin(x0) + 0.5 * x1 * x2 - 0.3 * x3
```

with `x0..x3` drawn uniformly from `[-1, 1]`. It is smooth, mildly
nonlinear (one trig term plus a product of two inputs), and easily
learnable by a single GLU FFN block.

## Build & run

```
lazbuild GLUFeedForward.lpi
../../bin/x86_64-linux/bin/GLUFeedForward
```

Or directly with fpc:

```
fpc -dRelease -dUseCThreads -O3 -Fu../../neural GLUFeedForward.lpr
./GLUFeedForward
```

Pure CPU, no external data. A few seconds on a single low-core
machine; faster on anything modern. The run is non-interactive (no
trailing `ReadLn`).

## Expected output sketch

```
Layers:
 Layers: 4
 Neurons:33
 ...
Layer 0 ... TNNetInput          Output:4,1,1
Layer 1 ... TNNetFullConnectLinear Output:1,1,32
Layer 2 ... TNNetGLU            Output:1,1,16
Layer 3 ... TNNetFullConnectLinear Output:1,1,1

Training Dense -> GLU -> Dense FFN on synthetic regression...
Epochs:  1 ... Validation Error: 0.42 ...
Epochs: 10 ... Validation Error: 0.13 ...
Epochs: 20 ... Validation Error: 0.03 ... Test Accuracy: ~0.98

Sample predictions on held-out test data:
  inputs=( 0.83, -0.92,  0.05, -0.50)  predicted= 0.86  target= 0.86
  inputs=( 0.45,  0.38, -0.13,  0.70)  predicted= 0.21  target= 0.20
  ...
```

A prediction counts as a "hit" when it is within 0.1 of the target.
Validation error should fall from ~0.45 at random init to under 0.05
by the end of training.
