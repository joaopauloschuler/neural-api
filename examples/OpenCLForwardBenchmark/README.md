# OpenCL Forward Benchmark

Forward-pass CPU-vs-OpenCL timing sweep across every **single-input** layer type
that has an OpenCL forward path (~40 layers).

For each layer the program builds a two-layer net (`TNNetInput -> TheLayer`),
times the forward pass on the CPU, then calls `EnableOpenCL` and times the same
forward on the device, and prints one table row:

```
layer                            out shape        cpu us/fwd   gpu us/fwd    speedup  gpu?   verdict
--------------------------------------------------------------------------------------------------------
TNNetConvolution                 32x32x1280          85125.0      20281.3      4.20x  yes    yes
TNNetRMSNorm                     256x1x512              42.1         98.7           -  NO-cpu no
...
```

The `verdict` column is separate from `gpu?`: every row is forced onto the device
(`NN.ForceOpenCL(True)`) so both timings are always charted, while `verdict`
reports each layer's own `FShouldOpenCL` size decision — what production dispatch
would choose at that shape. The gap between where `verdict` flips to `yes` and
where `speedup` crosses `1x` shows whether a layer's threshold is tuned well.

## The `gpu?` column matters

Several layers only dispatch to the device above a size threshold (e.g. SDPA
needs `SeqLen >= csSDPAOpenCLMinSeqLen`) and otherwise silently run on the CPU.
The benchmark reads back each layer's `ForwardGPUCnt` dispatch counter after the
timed run: `yes` means the device path actually fired; `NO-cpu` means it fell
back to the host, in which case the "speedup" is meaningless and is omitted
(`-`). This keeps a CPU-vs-CPU comparison from masquerading as a GPU win.

## Timing method

Wall-clock, auto-scaled: each measurement repeats the forward pass in doubling
batches until at least 0.4 s of work has run (capped at 8192 forwards), so the
coarse resolution of `Now()` does not dominate the small layers. A warmup pass
is run and discarded before each measurement; the first device call additionally
pays kernel compilation and buffer upload, which the warmup absorbs.

Input profiles: `SEQ = (256, 1, D)` for sequence/transformer layers,
`VIS = (32, 32, C)` for spatial layers; `d_k = 64`, `d_model = 512`. These are
the base sizes at the top of the `.lpr` (`cSeqLen`, `cVisX/Y/C`, `cDk`, ...).

To sweep larger without recompiling, pass an **integer scale factor** as the
first argument. It does **not** multiply every dimension at once; instead, for
each layer, it multiplies only the ONE dimension that drives that layer's
CPU-vs-GPU crossover — the sequence length for the per-token SEQ ops and
attention, the spatial width for the VIS spatial ops, the output-feature count
for the convolutions (spatial and pointwise), the output width for the dense
layers, and the cell width for the recurrent cells. Every other dimension stays
at base, so the sweep grows near-linearly instead of the `k^3..k^4` blow-up of
scaling all dimensions together:

```
OpenCLForwardBenchmark        # base sizes (default == scale 1)
OpenCLForwardBenchmark 8      # each layer's primary dimension x8
```

An integer factor is used deliberately: the base sizes satisfy each layer's
divisibility invariants (channels `/8` for GroupNorm and `/4` for GroupConvP4,
even gate/RoPE depths, `3*d_k` packing) and integer multiplication preserves
them, so no factor can produce an illegal shape. A non-numeric or `<1` argument
falls back to 1. The resolved base sizes (and the factor) are echoed in the banner.

The base sizes are kept moderate on purpose: the benchmark lowers the device
dispatch gate `NeuralConvOpenCLMinWork` to `2^16` (`cMinWorkMACs`), so they
already push the conv/norm/gate/softmax family onto the device without needing
giant tensors. (The depthwise/KAN paths whose result buffer once grew with the
square of the group count - and could reach tens of GB - have been fixed; see
the GEMV-diagonal note in `tasklist.md`, so large factors are now safe.)

This is a microbenchmark of isolated single layers, not a model — the numbers
show where the device beats the host for each operator in isolation. Wall-clock
numbers are inherently machine/run dependent.

## Reading the results

On a **GPU** device the big matmul layers (convolutions, `TNNetFullConnect`, the
softmax attentions) should show `speedup > 1`. On a **CPU OpenCL device** (e.g.
POCL) the native AVX host path usually wins, so speedups come out `< 1` - that is
expected and still useful: it confirms the offload fires (`gpu? = yes`) and shows
the per-operator device/host ratio for that machine.

Many elementwise/normalization layers report `NO-cpu`: they override
`EnableOpenCL` but do not run a standalone device kernel at these shapes
(`ForwardGPUCnt = 0`), so the host path is used and no speedup is shown.

## Out of scope (v1)

Multi-input layers need a second source branch and are not benchmarked here:
`TNNetCrossAttention`, `TNNetGridSample`, `TNNetAffineGridSample`,
`TNNetBackwardWarp`, `TNNetFlowWarp`, `TNNetAdaIN`, `TNNetCorrelationVolume`,
`TNNetCorrelationLookup`.

One further single-input layer is excluded by name:
- `TNNetMRotaryEmbedding` - requires `SetPositions` (a multimodal T/H/W position
  grid) before any forward, which this generic harness does not supply.

(`TNNetKANConv` used to be excluded too - its OpenCL forward requested a ~81 GB
buffer and segfaulted inside the driver. That has been fixed: it now contracts
the combined tap axis into a single dense GEMM, so it is back in the sweep.)

## Build & run

Requires a build with `-dOpenCL` and a linkable `libOpenCL`. With no OpenCL
platform/device available the program prints `SKIP` and exits 0 (harmless in a
CPU-only CI). A CPU OpenCL runtime such as POCL is enough to exercise the device
path end to end.

**Run it from this directory.** The OpenCL kernel source (`neural/neural.cl`) is
located by relative-path fallbacks; from `examples/<x>/` the `../../neural/neural.cl`
fallback resolves it. Running from the repo root will print
`File neural.cl could not be found.` and the device path will fail.

If linking complains `cannot find -lOpenCL`, your system has the runtime
`libOpenCL.so.1` but not the `libOpenCL.so` dev symlink. Either install the ICD
loader `-dev` package, or point the linker at a local symlink:

```
mkdir -p /tmp/cllib && ln -sf $(ldconfig -p | awk '/libOpenCL.so.1/{print $NF; exit}') /tmp/cllib/libOpenCL.so
```

then add `-Fl/tmp/cllib` to the build's custom options.

From this directory, with Lazarus:

```
lazbuild OpenCLForwardBenchmark.lpi
../../bin/x86_64-linux/bin/OpenCLForwardBenchmark   # run from HERE, not the repo root
```

Or directly (needs the `multithreadprocslaz` unit path on the command line):

```
fpc -Mobjfpc -Sh -O3 -dAVX2 -dRelease -dOpenCL -Fu../../neural OpenCLForwardBenchmark.lpr
```
