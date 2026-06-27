# VideoFrameInterpolation — RIFE intermediate-frame synthesis

Synthesises **one intermediate frame** (t = 0.5) between two input frames on the
CPU end-to-end with the repo's RIFE importer
`BuildRIFEFromSafeTensors` (`neuralpretrained.pas`) — a video-generative import.
RIFE (Huang et al. 2022, *"Real-Time Intermediate Flow Estimation for Video Frame
Interpolation"*, [arXiv:2011.06294](https://arxiv.org/abs/2011.06294)) estimates a
bidirectional intermediate flow with a coarse-to-fine stack of IFBlocks,
**backward-warps** both frames, and blends them with a learned soft fusion mask.

Unlike the landed RAFT path (which estimates optical **flow** but does not
synthesise frames) and `examples/FrameInterpolation` (a from-scratch toy on
`TNNetFlowWarp`), this example imports the full RIFE synthesis network.

## The new primitive

The differentiable backward-warp `TNNetBackwardWarp` (RIFE convention:
pixel-unit flow, bilinear sampling, border-clamp — the integer-pixel equivalent
of `grid_sample(padding_mode='border', align_corners=True)`). Everything else
(3×3 conv, per-channel PReLU, sigmoid, channel-broadcast multiply, residual sum)
reuses landed layers.

## Self-contained fixture (no network access)

The official Practical-RIFE checkpoints are not obtainable offline, so — exactly
like the repo's NAFNet / SwinIR pico fixtures — this falls back to a committed
config-faithful **random pico RIFE** (`tests/fixtures/tiny_rife.*`, built by
`tools/rife_tiny_fixture.py` and parity-checked < 1e-4 against a float64 numpy
oracle in `tests/TestNeuralPretrained.pas`). The pico net is **random, not
trained**, so this is a wiring / throughput **smoke** demo: it builds the net,
makes two synthetic frames (a bright blob translating across the grid), packs
them on the depth axis as `[frame0 | frame1]`, runs the interpolation forward
pass, and writes the result. Pass a real `.safetensors` (+ `config.json` sibling)
to interpolate with your own trained checkpoint.

## Build & run

```
lazbuild examples/VideoFrameInterpolation/VideoFrameInterpolation.lpi
cd examples/VideoFrameInterpolation
./VideoFrameInterpolation                          # committed pico fixture
./VideoFrameInterpolation model.safetensors [config.json]   # real checkpoint
```

(The default fixture paths are relative to the example folder, so run it from
`examples/VideoFrameInterpolation`.)

## Output

- The imported RIFE config (`RIFEConfigToString`) and the interpolated frame's
  shape.
- `rife_frame0.ppm`, `rife_middle.ppm`, `rife_frame1.ppm` (P6 colour images) plus
  a small ASCII preview of `frame0 / middle / frame1`.

With the random pico fixture the middle frame is not a meaningful interpolation;
the example demonstrates that the importer and the backward-warp forward path run
end-to-end on CPU.
