# FrameInterpolation: predict the middle frame (RIFE / FILM)

Video **frame interpolation** — synthesize the unseen middle frame `t+1` that sits
between two given endpoint frames `t` and `t+2` (the RIFE / FILM task). This is
distinct from `../VideoPrediction`, which *extrapolates* the next frame. Synthetic
data, pure CPU; the default SMOKE run finishes well under five minutes.

## What it does

Reuses a self-contained Moving-MNIST-style blob world: a bright blob translates
across a `16x16` grid at constant velocity, bouncing off walls. The model sees the
two endpoints (`t`, `t+2`) stacked as two channels and is supervised on the hidden
middle frame `t+1`. Two models are trained and compared:

- **(a) Direct** conv encoder-decoder that directly synthesizes the middle frame's
  pixels.
- **(b) Flow-based warping**: the same encoder instead predicts, per pixel, a dense
  optical-flow field for each endpoint (`F0 = mid→t`, `F1 = mid→t+2`); the new
  **`TNNetFlowWarp`** primitive backward-warps each endpoint, and the two warps are
  averaged with a symmetric `0.5/0.5` blend. On this rigid-motion data the flow
  path reaches lower held-out error than direct synthesis.

The reconstruction loss is **L1 + (1 − SSIM)**, using the
`neuralimagemetrics.ComputeSSIMLossAndGradient` helper. The custom per-pixel
gradient is injected through the standard `TNNet.Backpropagate` path via the
pseudo-target identity `Desired = Output − GradOut` (the library's last-layer rule
is `OutputError = Output − Desired`).

## Running

```
cd examples/FrameInterpolation
lazbuild FrameInterpolation.lpi --build-mode=Release   # or: fpc -Fu../../neural FrameInterpolation.lpr
./FrameInterpolation            # smoke run (fast, default)
./FrameInterpolation --full     # longer, sharper run
```

## Output

Held-out L1 / SSIM for both models, an ASCII panel
(before | predicted | truth | after) and a PPM triplet per model
(`frameinterp_direct.ppm`, `frameinterp_flow.ppm`).

## Notes

- The SSIM window is 11×11, so the grid is fixed at 16×16.
- All data is generated in code — no download, fully offline.

Coded by Claude (AI).
