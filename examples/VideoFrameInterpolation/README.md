# VideoFrameInterpolation — RIFE intermediate-frame synthesis

Synthesises **intermediate frames** (a 2x/4x/8x/... schedule, or one frame at any
time `t`) between two input frames on the CPU end-to-end with the repo's RIFE importer
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

## Timesteps and multi-frame schedules

How a frame at time `t` is produced depends on the checkpoint, which the importer
inspects (`DetectRIFETimestepInput`: the first IFBlock's input-plane count):

- **Timestep models** (Practical-RIFE v4: `2*in_channel + 1` input planes) read
  a constant plane filled with `t` after the two frames, so every `t` is one
  direct forward pass and `--factor` may be any integer >= 2.
- **Models without a timestep plane** (older RIFE: `2*in_channel` planes) only
  predict `t = 0.5`. `RIFEInterpolateFrames` recurses — the middle frame, then
  the middles of each half — so `--factor` must be a power of two, and
  `RIFEInterpolateAt` bisects toward `--t`, which must be dyadic (`k/2^n`, e.g.
  0.25, 0.375).

Either way the net is built once and reused for every frame.

## Self-contained fixture (no network access)

The official Practical-RIFE checkpoints are not obtainable offline, so — exactly
like the repo's NAFNet / SwinIR pico fixtures — this falls back to a committed
config-faithful **random pico RIFE** (`tests/fixtures/tiny_rife.*`, built by
`tools/rife_tiny_fixture.py` and parity-checked < 1e-4 against a float64 numpy
oracle in `tests/TestNeuralPretrained.pas`; `tests/fixtures/tiny_rife_t.*` is the
timestep-plane variant, checked at t = 0.25). The pico net is **random, not
trained**, so this is a wiring / throughput **smoke** demo: it builds the net,
makes two synthetic frames (a bright blob translating across the grid), packs
them on the depth axis as `[frame0 | frame1]`, runs the interpolation forward
pass, and writes the result. Pass a real `.safetensors` (+ `config.json` sibling)
to interpolate with your own trained checkpoint.

## Build & run

```
lazbuild examples/VideoFrameInterpolation/VideoFrameInterpolation.lpi
cd examples/VideoFrameInterpolation
../../bin/x86_64-linux/bin/VideoFrameInterpolation              # pico fixture, 2x
../../bin/x86_64-linux/bin/VideoFrameInterpolation --factor 4 --gif rife.gif
../../bin/x86_64-linux/bin/VideoFrameInterpolation --t 0.375    # one frame
../../bin/x86_64-linux/bin/VideoFrameInterpolation model.safetensors [config.json] --factor 8
```

| Option | Meaning |
| --- | --- |
| `--factor N` | write the `N-1` frames at `t = k/N` (default 2) |
| `--t VALUE` | write one frame at time `VALUE` in (0, 1) instead |
| `--gif FILE` | also write every frame, inputs included, as a looping animated GIF |

(The default fixture paths are relative to the example folder, so run it from
`examples/VideoFrameInterpolation`.)

## Output

- The imported RIFE config (`RIFEConfigToString`) and the interpolated frame's
  shape.
- `rife_frame_00.ppm` (frame0), the intermediate frames in time order, and
  frame1 last (P6 colour images), plus an ASCII preview of each frame for grids
  up to 32x32.

With the random pico fixture the middle frame is not a meaningful interpolation;
the example demonstrates that the importer and the backward-warp forward path run
end-to-end on CPU.
