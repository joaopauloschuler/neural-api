# BitLinear Bake-Off

A ternary-vs-full-precision bake-off, the `TNNetBitLinear` (BitNet) follow-up.
The same tiny classifier architecture is trained twice on the same small
synthetic multi-class task with the same fixed `RandSeed` (424242), same data,
and same init order — the only thing that changes is the **head type**:

- **Full precision (FP32):** `TNNetFullConnectLinear` heads — 32 bits/weight.
- **Ternary {-1, 0, +1}:** `TNNetBitLinear` heads — the BitNet-style quantized
  drop-in replacement. `TNNetBitLinear` subclasses `TNNetFullConnectLinear` and
  quantizes its weights to `{-1, 0, +1}` in the forward pass, so it wires in
  identically and at matched layer sizes.

Both nets share the exact same shape:

```
Input(2) -> Head(24) -> ReLU -> Head(4) -> SoftMax
```

## The task

A 4-class 2D Gaussian-blob classification problem: each class is a noisy blob
around one of the four corners `(±1, ±1)`. The four corners are not linearly
separable into the four classes, so the hidden layer actually does work. The
problem is small enough that the whole bake-off (both variants, 40 epochs each
with early stopping) runs single-threaded in well under a second on CPU.

## What it reports

For each variant:

- final **train / validation / test accuracy**, and
- the **effective weight memory**: the exact weight count from
  `TNNet.CountWeights` times the bits-per-weight assumption, where
  - FP32 = **32** bits/weight, and
  - ternary = **log2(3) ≈ 1.585** bits/weight (the information-theoretic cost
    of one ternary symbol — the headline BitNet "1.58-bit" figure).

It then prints the **compression ratio** (FP bytes / ternary bytes) and a
self-checking **PASS/FAIL gate** encoding the headline claim:

> *near-FP accuracy at a fraction of the weight memory.*

The gate requires all three of:

1. BitLinear test accuracy is within **8 points** of the FP head,
2. BitLinear weight memory is **< 10%** of the FP head, and
3. BitLinear is still a useful classifier (test accuracy **≥ 80%**).

The weight count is identical for both variants (144 weights), so the memory
win comes purely from bits-per-weight; the gate is non-trivial because it also
demands the ternary net does not lose meaningful accuracy. If the gate fails
the program exits with a non-zero status.

## Example run

```
=== Accuracy ===
head                            train     val      test
TNNetFullConnectLinear (FP32)    99.75%  100.00%   99.67%
TNNetBitLinear (ternary)        100.00%  100.00%  100.00%

=== Effective weight memory ===
head                            weights  bits/wt  bytes
TNNetFullConnectLinear (FP32)       144   32.00       576.0
TNNetBitLinear (ternary)            144    1.58        28.5

Weight count is matched (144 vs 144).
Compression ratio (FP bytes / ternary bytes): 20.19x

GATE: PASS -- near-FP accuracy at a fraction of the weight memory.
```

On this task the ternary head **matches** the full-precision head (it even
edges it out on the held-out test set, well within noise) while using only
~5% of the weight memory — a **~20x** compression.

## Build & run

```
cd examples/BitLinearBakeoff
lazbuild BitLinearBakeoff.lpi
../../bin/x86_64-linux/bin/BitLinearBakeoff
```
