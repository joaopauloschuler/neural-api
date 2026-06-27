# OctonionConv — when octonion weight sharing pays off

A **parameter-matched bake-off** showing when the structured weight sharing of
`TNNetOctonionConv` beats an ordinary convolution. The synthetic task is exactly
what an octonion conv computes: a fixed **octonion left-multiplication** applied
to the 3×3 spatial average of an 8-component input field (one octonion per pixel),
so the octonion layer's inductive bias matches the target symmetry.

## What it uses

- `TNNetOctonionConv(8, 3, 1, 1)` (~72 weights) vs a parameter-matched
  `TNNetConvolutionLinear` 8→1→8 bottleneck baseline.
- `TNNetInput`, `TNNet.Create/AddLayer`, manual `Compute` / `Backpropagate`
  training (LR 0.01, momentum 0.9, 120 epochs).
- Procedurally generated data: 128 train + 32 validation 8×8 images with 8
  channels; the target is the ground-truth octonion (`OMul`) applied to the
  blurred input.

## Running

No arguments, no dataset, no download — data is generated in code. Pure CPU,
under a minute.

```
cd examples/OctonionConv
# build with lazbuild OctonionConv.lpi (or fpc), then:
./OctonionConv
```

It prints each model's weight count and final validation MSE. Expected: the
octonion conv reaches the lowest error because its weight sharing matches the task
symmetry — fewer effective parameters for the same job.

Coded by Claude (AI).
