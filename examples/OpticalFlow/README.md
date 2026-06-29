# OpticalFlow: dense flow with RAFT

The demo for the **RAFT** optical-flow importer
(`BuildRaftFromSafeTensors`, `neural/neuralpretrained.pas`; Teed & Deng 2020,
*RAFT: Recurrent All-Pairs Field Transforms for Optical Flow*,
[arXiv:2003.12039](https://arxiv.org/abs/2003.12039)). Two frames in, a dense
per-pixel `(dx, dy)` motion field out.

## What it does

The RAFT pipeline: a shared feature encoder over both frames → an all-pairs
**correlation volume** (`TNNetCorrelationVolume`, dot-products between every pair
of feature locations) → an iterative **ConvGRU** update operator
(`TNNetConvGRUCell`) that, via a local correlation lookup
(`TNNetCorrelationLookup`) around the current flow, refines the flow over N steps.

The example synthesizes a bright square translated by a known `(SHIFT_X, SHIFT_Y)`
between frame 1 and frame 2, loads the pico `raft_small` fixture with
`BuildRaftFromSafeTensors`, runs `RaftPredictFlow`, and writes two PPMs:

- `opticalflow_field.ppm` — the predicted flow color-coded Middlebury-style
  (hue = direction, brightness = magnitude).
- `opticalflow_warp.ppm` — frame-1 | frame-1 warped toward frame-2 by the predicted
  flow (`TNNetFlowWarp`) | frame-2, side by side at the `/4` flow grid.

## Running

```
cd examples/OpticalFlow
fpc -Fu../../neural OpticalFlow.lpr
./OpticalFlow
```

The program probes both `../../tests/fixtures/` and `tests/fixtures/` for the
fixture, so it runs from either the example directory or the repo root.

## Notes

- Default input is the committed `tests/fixtures/tiny_raft.safetensors` fixture
  (built by `tools/raft_small_tiny_fixture.py`) with **random** weights, so the
  predicted field is **not** the ground-truth flow — this is a forward / plumbing
  and flow-visualisation + warping demonstration.
- Point `BuildRaftFromSafeTensors` at a real torchvision `raft_small` export for
  real flow. Inference only, pure CPU.

Coded by Claude (AI).
