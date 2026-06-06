# MaxBlurPool — anti-aliased, shift-invariant max pooling

This example demonstrates `TNNetMaxBlurPool`, an anti-aliased max-pool from
Zhang 2019, *"Making Convolutional Networks Shift-Invariant Again"*.

## The idea

A naive strided max-pool **aliases**: it subsamples right after a non-linear
max, so the down-sampled output can jump sharply when the input is shifted by
a single pixel. `TNNetMaxBlurPool` fixes this by splitting the operation:

1. **Dense max** — a max-pool with window `PoolSize` and **stride 1**, so the
   spatial size is preserved (no subsampling yet).
2. **Blur + subsample** — a **fixed (non-trainable)** separable binomial
   low-pass filter `[1,2,1] x [1,2,1] / 16` applied with the downsampling
   stride.

Low-pass filtering before subsampling is the classic anti-aliasing recipe, so
the result approximately commutes with small input translations.

The blur weights are constant (no gradient flows to them), but the layer's
backward pass still routes the input gradient correctly through both the fixed
blur (a transposed convolution that scatters each output error to its taps) and
the max selection (each blurred contribution is routed to the arg-max cell of
its dense-max window).

## What the demo does

It builds a smooth `16x16x3` image and measures the **mean absolute output
change** under small 1..3 px horizontal input shifts, for a plain strided
`TNNetMaxPool(2)` versus `TNNetMaxBlurPool(2)`. It prints a per-shift table and
ends with a **self-checking PASS/FAIL gate**: `MaxBlurPool` must change *less*
than `MaxPool`. No training, CPU-only, runs in well under a second.

## Constraint

Like `TNNetMaxPool`, the implementation assumes **square** feature maps
(`SizeX = SizeY`). Use square inputs.

## Build and run

```bash
cd examples/MaxBlurPool
lazbuild MaxBlurPool.lpi
../../bin/x86_64-linux/bin/MaxBlurPool
```

## Constructor

```pascal
TNNetMaxBlurPool.Create(pPoolSize: integer = 2; pStride: integer = 0;
  pPadding: integer = 0);
```

`pStride = 0` defaults the stride to `pPoolSize` (the usual down-sampling
behaviour). The integer pool/stride/padding parameters round-trip through
`SaveToString` / `LoadFromString` exactly like `TNNetMaxPool`.
