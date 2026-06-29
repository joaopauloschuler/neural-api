# Consistency distillation (few-step generative sampling)

A small, CPU-friendly demonstration of **consistency distillation** (Song et al.
2023, ["Consistency Models"](https://arxiv.org/abs/2303.01469)). The landed
diffusion examples (`../DiffusionMNIST`, `../ConditionalDiffusion`,
`../FlowMatching`) all sample with many reverse steps (25–200); this example
distils a trained MNIST diffusion **teacher** into a **consistency model** that
generates a digit in **1, 2 or 4 steps**.

## What it does

A consistency model learns a function `f(x_t, t)` that maps any point on a
probability-flow ODE trajectory directly back to that trajectory's clean origin
`x_0`, so a single network call already yields a (rough) sample. The example:

1. pretrains an **eps-prediction DDPM teacher** (the usual noise-prediction
   objective) on MNIST;
2. distils it into a **student** `F_theta` using the consistency loss — for
   adjacent timesteps on a sub-grid it forward-noises a clean digit, takes one
   deterministic teacher **DDIM** ODE step, and minimises
   `|| f_theta(x_hi, t_hi) − f_target(x_lo, t_lo) ||²` against a stop-gradient
   **EMA target net**;
3. samples in 1/2/4 steps and reports fidelity against the multi-step teacher.

The boundary condition `f(x, 0) = x` is enforced with Karras-style skip/out
scalings (`f = c_skip·x + c_out·F_theta`), computed as plain example-side
arithmetic — no new layer is needed. The `c_out(t_hi)` factor is folded into the
backprop target so the raw student head trains on the right residual.

### Key neural-api pieces

- **`TNNetEMAWrapper`** (`neuralnetwork.pas`) — its shadow net *is* the
  stop-gradient target `f_target`; `Update()` folds the live student weights with
  decay.
- **`TNNetDiffusionScheduler`** (`neuraldiffusion.pas`) — linear-beta schedule
  (`AlphaBar`, `AddNoise`, `Sample` with `smDDIM`) used for both teacher noising
  and the multi-step teacher reference.
- The teacher and student share a tiny **time-conditioned U-Net** (reused from
  `../DiffusionMNIST`): `TNNetSinusoidalTimeEmbedding` → shared cond MLP →
  `AddFiLMConditioned` into each `TNNetConvolutionReLU` + `TNNetGroupNorm` block,
  with `TNNetDeepConcat` skips and `TNNetUpsample` / `TNNetMaxPool` for the
  encoder/decoder.

## Running

```
cd examples/ConsistencyDistill
lazbuild ConsistencyDistill.lpi --build-mode=Release
./ConsistencyDistill            # SMOKE: finishes well under 5 min on one CPU
./ConsistencyDistill --full     # many more steps for sharper digits
```

(Or compile the `.lpr` directly with `fpc -Fu../../neural` as in the other
examples.)

## Inputs / outputs

- **Data:** standard MNIST `*-idx*-ubyte` files in the working directory
  (symlinked from `../DiffusionMNIST`). If they are absent the program falls back
  to a **synthetic** bar dataset so the demo still runs in CI; the pipeline and
  metrics are identical.
- **Output:** writes `consistency_samples.png`, a sample grid whose rows are
  *teacher multistep / 1-step / 2-step / 4-step* consistency samples.
- **Metric:** for each sampler it prints the mean per-sample MSE of generated
  digits to their nearest training digit (lower = closer to the data manifold)
  plus a NaN/Inf pixel count. The headline is that the 1/2/4-step consistency
  samples approach the multi-step teacher quality at a fraction of the steps.
  Smoke-mode samples are rough; `--full` sharpens them.

Coded by Claude (AI).
