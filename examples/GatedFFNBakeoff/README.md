# Gated FFN Bake-Off

A head-to-head comparison of the **five** in-tree gated feed-forward
activation layers. The exact same transformer-style feed-forward block
is built five times on the exact same tiny synthetic sequence task, and
the *only* thing that changes between arms is which gating layer sits in
the middle of the `Dense -> GATE -> Dense` sandwich.

Arms:

- `TNNetGLU`     `value * sigmoid(gate)` (https://arxiv.org/abs/1612.08083)
- `TNNetReGLU`   `ReLU(value) * gate`    (https://arxiv.org/abs/2002.05202)
- `TNNetGEGLU`   `value * GELU(gate)`    (https://arxiv.org/abs/2002.05202)
- `TNNetSwiGLU`  `value * Swish(gate)`   (https://arxiv.org/abs/2002.05202)
- `TNNetTanhGLU` `value * tanh(gate)`

All five gates share the **depth-doubling** convention used by the
single-layer `GLUFeedForward` / `GEGLUFeedForward` / `SwiGLUFeedForward`
demos: the projection in front of the gate emits `2 * d_ff` channels
(gate `||` value packed on the depth axis), the gate splits that in half
and combines the two halves down to `d_ff`, and the read-out projects
`d_ff -> d_model`. The gates are all **parameter-free**, so the five arms
have **identical parameter counts** and identical wiring — only the gate
non-linearity moves. There is no unavoidable asymmetry.

```
TNNetInput(6, 1, 4)                     # 6-position sequence, 4 feats/pos
  -> TNNetPointwiseConvLinear(32)       # per-position: gate || value (2 * d_ff)
  -> [ GATE ]                           # halves depth -> 16
  -> TNNetPointwiseConvLinear(1)        # per-position regression head
```

`TNNetPointwiseConvLinear` is used instead of `TNNetFullConnectLinear`
(which the single-layer demos use) because here the input is a real
length-6 sequence: a pointwise (1x1) projection is applied independently
at every position, which is exactly what a transformer FFN block does.

## The synthetic task

A tiny sequence-to-sequence regression. Each sample is a length-6
sequence of 4-dim vectors drawn uniformly from `[-1, 1]`. The target at
each position is a fixed, deterministic, mildly-nonlinear function of
that position's own features:

```
y = sin(x0) + 0.5 * x1 * x2 - 0.3 * x3
```

It is applied independently per position (one trig term, one feature
product, one linear term), so the per-position FFN block is exactly the
right tool and every arm can learn it.

Everything is generated in-code (no dataset download). `RandSeed` is
reset to the same value before each arm's data generation and before
building/initialising its net, so every arm sees identical inputs and
identical weight init; only the gate layer differs.

## Build & run

```
lazbuild examples/GatedFFNBakeoff/GatedFFNBakeoff.lpi
bin/x86_64-linux/bin/GatedFFNBakeoff
```

Pure CPU, no external data, finishes in a few seconds (~4 s on a typical
machine). The compiled binary lands in `bin/x86_64-linux/bin/` (shared
with the other examples), not inside this directory. The run is
non-interactive (no trailing `ReadLn`).

## What it shows

- A comparison table with one row per gate: initial MSE, final
  validation MSE, wall-clock seconds, and epochs-to-converge (first
  epoch whose validation MSE drops below `0.02`, or `>25` if it never
  does).
- Two NaN/Inf-guarded sanity checks printed as `PASS`/`FAIL`:
  1. all five arms produced a finite final loss, and
  2. all five arms reduced loss below their pre-training baseline
     (i.e. every gate actually learned).

The headline signal is **"they all work, here is how they compare"**.
The per-arm ranking is **seed-dependent**: on this short, easy task all
five gates converge almost immediately, and which one posts the lowest
final MSE or the fastest wall-clock will shuffle if you change the seed,
the widths, or the epoch budget. This program is a comparison *harness*,
not a claim that any one gate is universally best.

## Expected output sketch

Real fragment from a recent run (`RandSeed = 42`):

```
=== Comparison (val MSE, wall-clock, epochs-to-converge < 0.02) ===
gate             init_mse  final_mse   seconds  epochs_conv
TNNetGLU           0.1046     0.0006      0.73            1
TNNetReGLU         0.3931     0.0009      0.37            1
TNNetGEGLU         0.2626     0.0005      0.69            1
TNNetSwiGLU        0.2899     0.0006      0.80            1
TNNetTanhGLU       0.4388     0.0003      1.00            1

=== Sanity checks ===
[PASS] all 5 arms produced a finite (no NaN/Inf) final loss.
[PASS] all 5 arms reduced loss below their pre-training baseline.
```

All five final MSEs sit far below their initial baselines, confirming
every gate trains end-to-end on the task.
