# QuaternionLinear — when quaternion weight sharing pays off

A **parameter-matched bake-off** showing when the structured weight sharing of
`TNNetQuaternionLinear` (a hypercomplex fully-connected layer) beats dense and
grouped baselines. The synthetic task is exactly a quaternion-linear map: rotate
4 input quaternions (16 real channels) by a fixed ground-truth quaternion plus a
small fixed cross-quaternion coupling, matching the layer's inductive bias.

## What it uses

- `TNNetQuaternionLinear(16)` (64 weights) vs two parameter-matched baselines:
  a `TNNetFullConnectLinear(4)` → `TNNetFullConnectLinear(16)` bottleneck (bias
  off) and an `AddGroupedFullConnect(4 groups)` block-diagonal layer.
- `TNNetInput`, `TNNet.Create/AddLayer`, manual `Compute` / `Backpropagate`
  training (LR 0.02, momentum 0.9, 200 epochs).
- Procedurally generated data: 256 train + 64 validation 16-channel vectors; the
  target applies `QMul(gRot, q_i)` per quaternion plus small cross-terms.

## Running

No arguments, no dataset, no download — data is generated in code. Pure CPU,
under a minute.

```
cd examples/QuaternionLinear
# build with lazbuild QuaternionLinear.lpi (or fpc), then:
./QuaternionLinear
```

It prints each model's weight count and final validation MSE. Expected: the
quaternion layer reaches the lowest error because its weight sharing matches the
task symmetry.

Coded by Claude (AI).
