# Reversible Block (RevNet additive coupling)

This example demonstrates `TNNet.AddReversibleBlock`, a builder that wires a
RevNet-style **reversible additive-coupling** block
([Gomez et al. 2017, *The Reversible Residual Network*](https://arxiv.org/abs/1707.04585)).
It is pure CPU, single-threaded, uses only synthetic data, and finishes in a
few seconds.

## The additive-coupling math

The block splits its input depth into two equal halves `x1 | x2` and applies two
arbitrary **shape-preserving** functions `F` and `G` (here each is a small
pointwise stack `PointwiseConvReLU(HiddenDim) -> PointwiseConvLinear(halfDepth)`):

```
y1 = x1 + F(x2)
y2 = x2 + G(y1)
output = Concat(y1, y2)   # same shape as the input
```

This forward map is invertible **no matter what `F` and `G` are**, because each
step only ADDS a function of an already-known quantity. Running it backwards:

```
x2 = y2 - G(y1)     # y1 is the first output half, available directly
x1 = y1 - F(x2)     # x2 was just recovered above
```

`F` and `G` are never themselves inverted — only the two additions are undone.
That is the property RevNets exploit to discard layer activations during the
forward pass and recompute them on demand during backprop, giving training
memory that is (asymptotically) constant in network depth.

This example does **not** use that activation-recompute memory trick — standard
backprop through the composed layers is used, which is exactly correct. The
point is to demonstrate the round-trip the trick depends on.

## What the program does

1. **Trains end-to-end without NaN.** It builds
   `Input(2,2,4) -> AddReversibleBlock -> PointwiseConvLinear(4)` and trains it
   briefly on a tiny synthetic reconstruction task. The MSE drops smoothly,
   showing the block trains cleanly.
2. **Demonstrates the analytic inverse.** After a forward pass on a fresh probe
   input it applies the inverse formulas above (reusing the SAME `F`/`G` layer
   outputs captured from the forward pass) and reports the maximum
   reconstruction error between the recovered input and the original. The error
   is at floating-point noise level (well under `1e-4`), proving the coupling is
   exactly invertible.

## Build & run

From this directory, with FPC on your PATH:

```
fpc -B -Fu../../neural -Mobjfpc -Sh -O2 ReversibleBlock.lpr
./ReversibleBlock
```

(or open `ReversibleBlock.lpi` in Lazarus and run).

Expected tail of the output:

```
Analytic inverse round-trip on a fresh probe input:
  max reconstruction error = 8.15E-010
  ROUND-TRIP OK: input recovered from output to fp tolerance.
```
