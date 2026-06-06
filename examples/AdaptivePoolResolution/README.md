# AdaptivePoolResolution

A tiny demo of the headline property of **adaptive pooling**: one
fully-convolutional stack can accept inputs of **different spatial sizes**
and still produce a **fixed-size feature head** (and therefore a fixed-size
classifier output). This is the classic *"adaptive global pool -> FC
classifier"* pattern (PyTorch's `AdaptiveAvgPool2d` / `AdaptiveMaxPool2d`).

It exercises **both** library layers:

- [`TNNetAdaptiveAvgPool`](../../neural/neuralnetwork.pas) — mean over the
  adaptive window.
- [`TNNetAdaptiveMaxPool`](../../neural/neuralnetwork.pas) — maximum over the
  adaptive window.

Constructors are `Create(pSize)` (square) or `Create(pSizeX, pSizeY)`.

## The backbone

```
Input(NxNx3) --> Conv8(3x3,pad1)+ReLU --> Conv8(3x3,pad1)+ReLU
             --> MaxPool(2) --> [adaptive pool head]
```

Nothing before the adaptive head hard-codes a spatial size: the two padded
3x3 convolutions preserve the spatial dimensions and the stride-2 max pool
just halves them. So the **only** thing that changes between the two input
resolutions is the `TNNetInput` size, and the conv stack maps:

- `16x16x3` input  -> `8x8x8`  post-conv map
- `24x24x3` input  -> `12x12x8` post-conv map

The adaptive head then collapses **either** map to the same fixed output.

## What it does

1. **Variable in, fixed out.** For each of `TNNetAdaptiveAvgPool` and
   `TNNetAdaptiveMaxPool`, feed the same network the two resolutions and
   print `input -> post-conv -> post-adaptive-pool -> output` for each. With
   a global `Create(1)` head both resolutions yield a `1x1x8` output; with a
   `Create(2)` head both yield `2x2x8`.

2. **Degeneracy correctness checks** (built-in assertions):
   - **`Create(1)` == global pooling.** The adaptive(1) output is compared
     element-for-element against an independent per-channel global avg / max
     over the post-conv map.
   - **`Create(N)` == identity** when `N` equals the post-conv spatial size:
     the head output is compared against the post-conv map itself.

3. **Training sanity.** Train the global-head classifier
   (`...-> AdaptiveAvgPool(1) -> FC(2) -> SoftMax`) for a few epochs on a
   trivial synthetic 2-class task (bright top half vs. bright bottom half) at
   `16x16`, then run inference at the unseen `24x24` resolution. Because the
   convolution / FC weights are spatial-size independent, a same-architecture
   sibling net pinned to `24x24` (built and fed the trained weights via
   `CopyWeights`) classifies the larger images correctly.

Pure CPU, no dataset download, all data synthesised in-code; runs in well
under two seconds.

## Sample output

```
AdaptivePoolResolution: variable-resolution in, fixed-size out.
Backbone: Input(NxNx3) -> Conv8(3x3,pad1)+ReLU x2 -> MaxPool(2) -> [adaptive head]
Feeding the SAME net inputs at 16x16 and 24x24.

--- TNNetAdaptiveAvgPool with global (1x1) head ---
  res A: in 16x16x3 -> post-conv 8x8x8 -> adaptive 1x1x8 -> out 1x1x8
    degeneracy Create(1)==global-pool: max|diff|=0.0E+000 OK
  res B: in 24x24x3 -> post-conv 12x12x8 -> adaptive 1x1x8 -> out 1x1x8
  => variable in, FIXED out: both resolutions yield 1x1x8

--- TNNetAdaptiveMaxPool with global (1x1) head ---
  res A: in 16x16x3 -> post-conv 8x8x8 -> adaptive 1x1x8 -> out 1x1x8
    degeneracy Create(1)==global-pool: max|diff|=0.0E+000 OK
  res B: in 24x24x3 -> post-conv 12x12x8 -> adaptive 1x1x8 -> out 1x1x8
  => variable in, FIXED out: both resolutions yield 1x1x8

--- TNNetAdaptiveAvgPool with 2x2 head ---
  res A: in 16x16x3 -> post-conv 8x8x8 -> out 2x2x8
  res B: in 24x24x3 -> post-conv 12x12x8 -> out 2x2x8
  => variable in, FIXED out: both resolutions yield 2x2x8
  degeneracy Create(8)==identity: max|diff|=0.0E+000 OK

--- TNNetAdaptiveMaxPool with 2x2 head ---
  res A: in 16x16x3 -> post-conv 8x8x8 -> out 2x2x8
  res B: in 24x24x3 -> post-conv 12x12x8 -> out 2x2x8
  => variable in, FIXED out: both resolutions yield 2x2x8
  degeneracy Create(8)==identity: max|diff|=0.0E+000 OK

--- training sanity (TNNetAdaptiveAvgPool global head) ---
  epoch  1  train NLL=0.6822
  epoch 30  train NLL=0.0959
  train acc @ res 16x16 : 1.000
  infer acc @ res 24x24 (unseen size, weight-shared): 1.000
  => trained fixed-size head runs at a resolution it never saw.

=== summary ===
  degeneracy/shape checks: ALL PASSED
  identity check ran: yes
OK.
```

## Build & run

```
lazbuild AdaptivePoolResolution.lpi
../../bin/x86_64-linux/bin/AdaptivePoolResolution
```
