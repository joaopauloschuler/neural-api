# BitLinear Bake-Off

A ternary-vs-full-precision bake-off, the `TNNetBitLinear` (BitNet) follow-up.
The same tiny classifier architecture is trained three times on the same small
synthetic multi-class task with the same fixed `RandSeed` (424242), same data,
and same init order — the only thing that changes is the **head type**:

- **Full precision (FP32):** `TNNetFullConnectLinear` heads — 32 bits/weight.
- **Ternary {-1, 0, +1} weights:** `TNNetBitLinear` heads — the BitNet-style
  quantized drop-in replacement. `TNNetBitLinear` subclasses
  `TNNetFullConnectLinear` and quantizes its **weights** to `{-1, 0, +1}` in the
  forward pass, so it wires in identically and at matched layer sizes.
- **Fully quantized (ternary weights + int8 activations):** `TNNetBitLinear`
  with its **activation-quantization flag ON** — the BitNet b1.58
  "fully-quantized linear" path. In addition to the ternary weights, the layer
  **input** is rounded through a per-token absmax int8 quantization
  (`scale = absmax(x)/127`, `x_q = round(clip(x/scale, -127, +127)) * scale`)
  *before* the ternary matmul. The backward pass uses the straight-through
  estimator (STE) for the activation round/clip, exactly like the weight-STE.
  The flag is constructed via
  `TNNetBitLinear.Create(SizeX, SizeY, Depth, SuppressBias, QuantizeActivation)`
  and defaults to OFF (so the un-flagged class is unchanged).

All three nets share the exact same shape:

```
Input(2) -> Head(24) -> ReLU -> Head(4) -> SoftMax
```

## The task

A 4-class 2D Gaussian-blob classification problem: each class is a noisy blob
around one of the four corners `(±1, ±1)`. The four corners are not linearly
separable into the four classes, so the hidden layer actually does work. The
problem is small enough that the whole bake-off (all three variants, 40 epochs
each with early stopping) runs single-threaded in ~1 second on CPU.

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

The gate requires all of:

1. ternary-weight BitLinear test accuracy is within **8 points** of the FP head,
2. ternary-weight BitLinear weight memory is **< 10%** of the FP head,
3. ternary-weight BitLinear is still a useful classifier (test accuracy **≥ 80%**),
4. fully-quantized (ternary W + int8 act) test accuracy is within **15 points**
   of the FP head (a looser bar, since quantizing the activations too costs a
   little more), and
5. fully-quantized is still a useful classifier (test accuracy **≥ 70%**).

The weight count is identical for all variants (144 weights), so the memory
win comes purely from bits-per-weight; both ternary variants store the same
ternary weights (the int8 activation quantization is a runtime transform of the
input and adds no stored weight memory). The gate is non-trivial because it also
demands the quantized nets do not lose meaningful accuracy. If the gate fails
the program exits with a non-zero status.

## Example run

```
=== Accuracy ===
head                                    train     val      test
TNNetFullConnectLinear (FP32)            99.75%  100.00%   99.67%
TNNetBitLinear (ternary W)              100.00%  100.00%  100.00%
TNNetBitLinear (ternary W + int8 act)    99.67%  100.00%   99.67%

=== Effective weight memory ===
head                                    weights  bits/wt  bytes
TNNetFullConnectLinear (FP32)               144   32.00       576.0
TNNetBitLinear (ternary W)                  144    1.58        28.5
TNNetBitLinear (ternary W + int8 act)       144    1.58        28.5

Weight count is matched (144 vs 144).
Compression ratio (FP bytes / ternary bytes): 20.19x

GATE: PASS -- ternary weights AND fully-quantized stay near FP at a fraction of the weight memory.
```

On this task the ternary-weight head **matches** the full-precision head (it
even edges it out on the held-out test set, well within noise) while using only
~5% of the weight memory — a **~20x** compression. The **fully-quantized** head
(ternary weights *and* absmax-int8 activations) lands at **99.67%** test
accuracy — a **0.00-point** gap vs FP on this run — showing the BitNet b1.58
"fully-quantized linear" path trains end-to-end through the activation-STE and
loses essentially nothing on this task.

## Build & run

```
cd examples/BitLinearBakeoff
lazbuild BitLinearBakeoff.lpi
../../bin/x86_64-linux/bin/BitLinearBakeoff
```
