# DeformableConv — Deformable Convolution

A small bake-off demonstrating `TNNetDeformableConv` (Dai et al. 2017, ICCV,
[arXiv:1703.06211](https://arxiv.org/abs/1703.06211)): a convolution whose K×K
sampling grid is shifted by **learnable, per-output-location and per-tap 2-D
offsets**, so the receptive field adapts to image content — and can reach
**beyond** the rigid K×K window.

## The idea

A regular convolution samples a fixed K×K axis-aligned window at every output
location; a 3×3 conv can therefore only ever "see" one pixel in each direction.
`TNNetDeformableConv` adds a small **offset head** — an ordinary convolution over
the same input — that predicts, for every output location `(ox,oy)` and every tap
`(fx,fy)`, a 2-D offset `(dx,dy)`. Each tap is then gathered by **bilinear
interpolation** at

```
px = ox*Stride + fx - Padding + dx
py = oy*Stride + fy - Padding + dy
```

and the usual weighted sum over taps and input channels produces each output
channel. Gradients flow into the conv weights, the input, **and** the offset head
(through the bilinear interpolation coefficients' dependence on `px`,`py` — the
interesting/hard part of the backward pass). The offset head is
**zero-initialised**, so the layer starts identical to a plain convolution and
only learns to deform if it helps.

It is distinct from its fork siblings:
- a regular / dilated conv has a **fixed** grid (dilation just spaces it out by a
  constant);
- `TNNetGroupConvP4` is rotation-equivariant by a fixed group, not content-adaptive;
- **DeformableConv learns a per-location, per-tap, floating-point offset** and
  samples by bilinear interpolation.

## The task

A 14×14 single-channel field of a few smooth Gaussian bumps. The **target is the
input translated by (+3,+3)**: `target(x,y) = input(x-3, y-3)`. The answer at each
pixel lies **3 pixels away — outside any 3×3 window** — so a rigid 3×3 conv (taps
span only ±1 pixel) *structurally cannot* reach it and floors out. A deformable
3×3 conv can learn a constant +3 sampling offset and copy the value almost
exactly. This isolates the headline property: the offset head turns a 3×3 conv
into a learned long-range gather.

Two contenders, both a single 3×3 conv (pad 1, stride 1, 1 output map) mapping
14×14×1 → 14×14×1:
- **(A)** `TNNetConvolutionLinear(1,3,1,1)` — rigid grid;
- **(B)** `TNNetDeformableConv(1,3,1,1)` — adaptive grid (adds the offset head).

## Running

```
lazbuild examples/DeformableConv/DeformableConv.lpi
./bin/x86_64-linux/bin/DeformableConv
```

Pure CPU, tiny data, runs in well under a minute (~40 s on 2 cores). Typical
result (lower val-MSE is better):

```
  TNNetConvolutionLinear (rigid 3x3)     weights=   9   val-MSE=0.040235
  TNNetDeformableConv (adaptive 3x3)     weights= 190   val-MSE=0.000001
```

The rigid conv is bounded by its 3×3 footprint and cannot copy a value 3 pixels
away, so its error floors out around 0.04. **The deformable conv learns to shift
its sampling grid and reaches the answer** — MSE drops to ~1e-6, a ~40000×
improvement, on a task the rigid conv structurally cannot solve. The deformable
model does carry extra parameters (the offset head); the honest headline is that
it solves a problem the rigid conv physically cannot.

## A note on training stability

The predicted offsets are unbounded, and the bilinear-interpolation gradient
scales with the surrounding pixel magnitudes, so an aggressive learning rate can
make the offset head diverge (NaN). The example uses a modest `LR=0.004` with
momentum; if you push the LR much higher, expect instability — the standard cure
in the literature is a smaller LR (or a dedicated, smaller LR) for the offset head.
