# CBAMAttention

Minimal demo of `TNNet.AddCBAM`, the Convolutional Block Attention Module
(Woo et al., 2018, https://arxiv.org/abs/1807.06521).

CBAM refines a conv feature map `(SizeX, SizeY, C)` with two sequential
attention sub-modules, both shape-preserving:

## 1. Channel attention

Re-weights channels (which feature maps matter):

1. Pool the feature map over space with BOTH global average (`TNNetAvgChannel`)
   AND global max (`TNNetMaxChannel`), giving two `(1,1,C)` descriptors.
2. Each descriptor goes through a reduce(`C/r`) -> ReLU -> expand(`C`) MLP
   (`TNNetFullConnectReLU` -> `TNNetFullConnectLinear`).
3. The two are summed (`TNNetSum`), `TNNetSigmoid`'d to per-channel weights
   `Mc` in `(0,1)`, and broadcast-multiplied over space onto the input with
   `TNNetChannelMulByLayer`.

## 2. Spatial attention

Re-weights spatial positions (where to look) on the channel-refined map:

1. Reduce over the channel axis to a small 2-channel descriptor.
2. A `SpatialKernelSize` x `SpatialKernelSize` convolution (`TNNetConvolutionLinear`,
   symmetrically padded to keep the spatial size) followed by `TNNetSigmoid`
   yields a per-`(x,y)` gate `Ms` in `(0,1)` of shape `(SizeX, SizeY, 1)`.
3. `Ms` is replicated across depth (`TNNetDeepConcat.Replicate`) and
   element-wise multiplied with the channel-refined map (`TNNetCellMulByCell`).

## Signature

```pascal
function AddCBAM(InputLayer: TNNetLayer; ReductionRatio: integer = 16;
  SpatialKernelSize: integer = 7): TNNetLayer;
```

Returns the final refined layer (same shape as `InputLayer`).
`ReductionRatio` is clamped so the bottleneck width is >= 1; `SpatialKernelSize`
is forced odd and >= 1 so the padding keeps the spatial size exactly.

## v1 simplifications (vs. the paper)

This builder is wired purely from existing, tested layers — no new gradient
code. Two faithful simplifications:

- **Separate channel MLPs.** The paper shares ONE MLP between the avg and max
  channel branches; here each branch has its own reduce/expand MLP (summed
  before the sigmoid). This is a common, defensible simplification and is
  otherwise identical to the paper's channel module.
- **Learned channel descriptor for the spatial branch.** The paper reduces the
  channel axis with a FIXED avg-over-depth and max-over-depth into a 2-channel
  map. The library has no fixed avg-over-depth / max-over-depth primitive that
  produces a `(SizeX, SizeY, 1)` map, so the descriptor is instead a LEARNED
  pointwise (`1x1`) conv `C -> 2` (`TNNetPointwiseConvLinear`). It plays the same
  structural role (a 2-channel spatial descriptor feeding the spatial conv) but
  the two channels are learned linear projections rather than fixed avg/max.

Nothing is otherwise deferred: both the channel and spatial sub-modules are
fully implemented and trained end-to-end.

> Note: the channel sub-module's `TNNetMaxChannel` assumes a SQUARE feature map
> (`SizeX = SizeY`). Keep CBAM inputs square (this demo uses `16x16`).

## What the demo does

Wires `Input(16,16,3) -> ConvReLU(8) -> AddCBAM(r=4, k=7) -> AvgChannel ->
FullConnectLinear(2) -> SoftMax` and trains on a tiny synthetic 2-class problem
(low-mean vs high-mean noise). It prints the training loss after short rounds so
you can see it DECREASE while accuracy rises to 1.0. CPU-only, runs in seconds.

Example output:

```
Before training   train loss=0.27488  train acc=0.5195
After  2 epochs   train loss=0.25443  train acc=0.5195
After  4 epochs   train loss=0.23842  train acc=1.0000
After  6 epochs   train loss=0.23126  train acc=1.0000
After  8 epochs   train loss=0.22606  train acc=1.0000
After 10 epochs   train loss=0.22067  train acc=1.0000
```

## Build (FPC)

```
fpc -Mobjfpc -Sh -O2 -dUseCThreads -Fu../../neural \
    -Fu/usr/share/lazarus/<ver>/components/lazutils/lib/<arch> \
    CBAMAttention.lpr
./CBAMAttention
```

Or open `CBAMAttention.lpi` in Lazarus and press Run.
