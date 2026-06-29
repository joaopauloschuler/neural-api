# VideoActionTiny — 3-D convolution motion-direction classifier

Demonstrates **`TNNetConvolution3D`** (spatiotemporal / volumetric convolution)
on a tiny synthetic Moving-MNIST-style action-recognition task. No external
dataset, pure CPU, finishes in a few seconds.

## What it shows

Each sample is a short grayscale **clip**: `cT = 6` frames of a `12×12` image in
which a small bright blob slides across the grid in one of **four** directions
(right / left / down / up). The task is to classify the **motion direction**. A
single frame is ambiguous, so the network must integrate information **across
time** — exactly what a 3-D convolution buys you over a per-frame 2-D conv.

## Clip packing

`TNNetVolume` has three axes (`SizeX, SizeY, Depth`). The `T` frames are packed
contiguously along the **Depth** axis as `T` blocks of `C` channels (here `C = 1`,
so `Depth = cT`, frame `t` in depth slot `t`). `TNNetConvolution3D` slides a
`(FeatureSizeT × FeatureSizeXY × FeatureSizeXY)` kernel over the spatial grid
**and** over the time blocks within the depth axis, producing
`Depth = OutputT * NumFeatures` packed the same way.

## Model

```
Input(12, 12, cT*1)
  -> Convolution3D(F=8, T, K=3, pad=1, stride=1, C=1)   # mixes space AND time
  -> ReLU
  -> Convolution3D(F=8, T, K=3, pad=1, stride=1, C=8)   # OutputT shrinks by 2/conv
  -> ReLU
  -> FullConnectLinear(4)                               # 4 motion classes
  -> SoftMax
```

To prove the temporal mixing matters, the example trains **two** models on the
identical data and weight init: the spatiotemporal model (`FeatureSizeT = 3`) and
a **per-frame baseline** (`FeatureSizeT = 1`, a 2-D conv shared across frames, no
cross-frame coupling). It prints the held-out accuracy each actually reaches; the
3-D conv is expected to win because direction needs time.

> Related: `examples/VideoAction` is the *imported-transformer* counterpart — it
> runs a pre-built VideoMAE checkpoint in inference only. This example trains a
> from-scratch `TNNetConvolution3D` stack end-to-end.

## Build & run

```
lazbuild examples/VideoActionTiny/VideoActionTiny.lpi
./bin/x86_64-linux/bin/VideoActionTiny
```

## Output

Per-epoch held-out accuracy for each model, then a final comparison (chance =
0.25):

```
  3-D conv (T=3)        <acc>
  per-frame (T=1)       <acc>
```

with a verdict line stating whether the 3-D conv beat the per-frame baseline on
this run.
