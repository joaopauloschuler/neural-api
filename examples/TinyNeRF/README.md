# TinyNeRF — a differentiable volume renderer

A tiny Neural Radiance Field (Mildenhall et al. 2020, *"NeRF"*) learns an
implicit 3-D scene as a coordinate MLP `F(x,y,z) -> (r,g,b,sigma)`, and an image
is produced by **casting rays** from a pinhole camera, **sampling** points along
each ray, evaluating `F` at every sample, and **alpha-compositing** the samples
into a pixel colour:

```
C   = sum_i  T_i * (1 - exp(-sigma_i * delta_i)) * c_i
T_i = exp( -sum_{j<i} sigma_j * delta_j )            (transmittance)
```

The new code here is that compositing step **and its hand-derived backward**, so
the whole render is trainable end-to-end. Fully **synthetic** (no dataset
download), pure CPU.

## Scene (analytic, in-code)

A single coloured sphere floating in front of a faint graded background, with a
**known** emission/density field. Training views are ray-marched from this
analytic field at a handful of camera poses (evenly spaced azimuths); a **held-out**
pose (new azimuth + elevation) is rendered to test novel-view synthesis. Because
the analytic ground truth uses the same compositing equation as the NeRF, the
target is exactly reachable.

## Model

```
Input(3) -> TNNetFourierFeatures(M, sigma)   # positional encoding (reused layer)
         -> FullConnectReLU(W) -> FullConnectReLU(W)
         -> FullConnectLinear(4)             # raw r,g,b,sigma per sample
rgb  = sigmoid(raw_rgb)   in (0,1)
sig  = softplus(raw_s)    >= 0
C    = alpha-composite over the per-ray samples (above)
```

The MLP forward runs once per ray sample. Per-ray loss is the squared colour
error vs the ground-truth pixel; `BackwardRay` computes the composite gradient
w.r.t. each sample's `(r,g,b,sigma)` by hand (a single far→near pass with running
suffix sums for the behind-attenuation term) and feeds it into the linear layer's
`OutputError`, then `Backpropagate()`. The last layer is primed once with
`IncDepartingBranchesCnt` (the manual-backprop convention).

## Build & run

```
lazbuild examples/TinyNeRF/TinyNeRF.lpi
./bin/x86_64-linux/bin/TinyNeRF
```

Smoke-sized by default (`24×24` render, 5 train poses, 8 ray samples, ~1000 ray
batches); finishes comfortably under a couple of minutes on CPU. Scale up by
raising `ImgRes` / `NumTrainPoses` / `NumSamples` / `NumIters` at the top of
`RunAlgo`.

## Output

- The held-out **PSNR before vs after** training. A flat/untrained render would
  mean the compositing gradient is wrong; the example checks that PSNR improves
  by at least ~2 dB and prints `OK: the volume renderer learned the scene.`
- `tinynerf_gt.ppm` (ground-truth held-out view) and `tinynerf_pred.ppm` (the
  trained NeRF's render of the same pose), written next to the binary at runtime.

A representative smoke run improves the held-out novel view from ~7.9 dB to
~16.4 dB.
