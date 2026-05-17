# Involution Demo

A function `f` is an **involution** when applying it twice returns the
original input: `f(f(x)) = x`. Reflecting a 2D image about an axis is
the textbook example; reversing the order of elements in a list is
another.

This example checks numerically that four shape-preserving layers in
the library are involutions:

- `TNNetReverseChannels` — flips the channel order at each (x, y).
- `TNNetReverseXY` — reflects the spatial grid through its centre.
- `TNNetFlipX` — mirrors columns left-to-right.
- `TNNetFlipY` — mirrors rows top-to-bottom.

For each layer it builds a two-layer net `[Input(4,4,3), L, L]`, feeds a
fixed random 4x4x3 input, and asserts that the output matches the input
under the L1 distance with a `1e-6` tolerance. One `PASS` / `FAIL` line
is printed per layer; the program exits with a non-zero status if any
check fails, so it doubles as a regression test for the inverse property
of these layers.

## Build and run

```
cd examples/InvolutionDemo
lazbuild InvolutionDemo.lpi
../../bin/x86_64-linux/bin/InvolutionDemo
```
