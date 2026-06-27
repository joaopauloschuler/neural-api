# SDPAOpenCLParity — CPU vs OpenCL SDPA parity test

A numeric **parity test** for the OpenCL Phase-1 offload of **Scaled Dot-Product
Attention**. For several attention configurations it runs the same single
`TNNetScaledDotProductAttention` layer once on the CPU and once on the GPU and
asserts the outputs match within a small tolerance (reordered GPU reductions
cause tiny FP differences).

## What it uses

- A one-layer net: `TNNetInput` → `TNNetScaledDotProductAttention`.
- `TEasyOpenCL` to enumerate platforms/devices, then `NN.EnableOpenCL(PlatformId,
  DeviceId)` to switch the forward to the GPU path (which fires once SeqLen ≥
  `csSDPAOpenCLMinSeqLen`).
- Deterministic input (`RandSeed := 424242`, shape `(64, 1, 3*48)`).

It checks 5 cases: full attention, causal, sliding-window 8, soft-cap 50, and a
causal window-1 edge case (near-single-key rows). Each reports
`max|diff|` against `cTolerance = 1e-3`.

## Running

Must be built **with `-dOpenCL`** to actually run the GPU path:

```
cd examples/SDPAOpenCLParity
fpc -Mobjfpc -Sh -O3 -dAVX2 -dRelease -dOpenCL -Fu../../neural SDPAOpenCLParity.lpr
./SDPAOpenCLParity
```

Behavior and exit codes:

- Built **without** `-dOpenCL`: prints a SKIP message and exits 0.
- No OpenCL platform / device available at runtime: prints `SKIP:` and exits 0.
- Otherwise prints `PASS`/`FAIL` per case and a final verdict —
  `SDPA OpenCL PARITY OK` (exit 0) or `SDPA OpenCL PARITY FAILED: N case(s)`
  (exit 1).

Coded by Claude (AI).
