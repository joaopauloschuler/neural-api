# QuaternionConv — when quaternion weight sharing pays off

A **parameter-matched bake-off** showing when the structured weight sharing of
`TNNetQuaternionConv` beats an ordinary convolution. The synthetic task is
exactly what a quaternion conv computes: a fixed **quaternion rotation** (Hamilton
product) applied to the 3×3 spatial average of a 4-component input field (one
quaternion / RGBA-style pixel), so the quaternion layer's inductive bias matches
the target symmetry.

## What it uses

- `TNNetQuaternionConv(4, 3, 1, 1)` (~36 weights) vs a parameter-matched
  `TNNetConvolutionLinear` 4→1→4 bottleneck baseline.
- `TNNetInput`, `TNNet.Create/AddLayer`, manual `Compute` / `Backpropagate`
  training (LR 0.01, momentum 0.9, 120 epochs).
- Procedurally generated data: 128 train + 32 validation 8×8 images with 4
  channels; the target is the ground-truth quaternion (`QMul`) applied to the
  blurred input.

## Running

No arguments, no dataset, no download — data is generated in code. Pure CPU,
under a minute.

```
cd examples/QuaternionConv
# build with lazbuild QuaternionConv.lpi (or fpc), then:
./QuaternionConv
```

It prints each model's weight count and final validation MSE. Expected: the
quaternion conv reaches the lowest error because its weight sharing matches the
task symmetry.

Coded by Claude (AI).
