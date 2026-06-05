# Echo State Network (Reservoir Computing)

## What it is

A self-contained implementation of an **Echo State Network** (ESN), the
canonical *Reservoir Computing* model (Jaeger, 2001). The ESN is a genuinely
different training paradigm from everything else in this repository: **there is
no backpropagation-through-time**. The recurrent core is a *fixed, random,
sparse* matrix; only a single linear readout is ever trained.

For a 1-D driving signal `x_t` the reservoir of `N` units evolves as a leaky
integrator:

```
h_t = (1 - a) * h_{t-1} + a * tanh(W_in * x_t + W * h_{t-1})
```

- `a` is the leak rate.
- `W_in` is a random input vector.
- `W` is a sparse random `N x N` matrix **rescaled to a chosen spectral radius
  `rho < 1`**. That `rho < 1` condition is the **echo-state property**: the
  reservoir asymptotically forgets its initial state, so the same input drives
  it to the same state regardless of where it started. That is what makes a
  *fixed random* recurrence usable as a feature generator.

### Spectral-radius rescaling

We reuse the library's power-iteration helper
[`TNNet.EstimateSpectralNorm`](../../neural/neuralnetwork.pas) to **measure**
the scale of `W` instead of running a full eigensolver. Note that helper
returns the spectral **norm** (largest singular value `sigma_1`), not the
spectral **radius** (`|lambda|_max`). For a general non-symmetric `W` these
differ, but `sigma_1` is always an *upper bound* on `|lambda|_max`, so
rescaling `W := W * (rho_target / sigma_1)` leaves the *true* radius
`<= rho_target` — conservatively on the safe side of the echo-state property.
This is why the working `rho_target` in the source is set above 1.0 (the true
radius still lands below 1). See the comments in `EchoStateNetwork.lpr`.

### Pipeline

1. Build `W_in` and a sparse `W` as plain Pascal arrays (hand-rolled
   recurrence — never touched by a gradient).
2. Rescale `W` to the target spectral radius using `EstimateSpectralNorm`.
3. Run the reservoir **forward** (no gradient) over a training sequence and
   collect each state `h_t` into a `TNNetVolumePair` (input = `h_t`,
   target = `x_{t+1}`).
4. Train **only** a `TNNetFullConnectLinear(1)` readout on those collected
   pairs with a tiny L2-regularised (ridge-style) SGD loop.

### Headline task

One-step-ahead prediction of the deterministic series
`sin(0.2 t) + 0.3 sin(0.31 t)`. After teacher-forced fitting the network
**free-runs**: its own prediction is fed back as the next input, and it
autonomously continues the waveform. An ASCII plot shows predicted (`o`) vs
true (`.`) over the free-run window.

### Built-in correctness signals (PASS/FAIL, `Halt(1)` on failure)

1. Teacher-forced one-step NRMSE well below the **persistence baseline**
   (the trivial `x_{t+1} = x_t` predictor).
2. The good (`rho < 1`) reservoir free-runs accurately.
3. **Echo-state ablation:** rebuilding the reservoir with `rho` driven above 1
   makes the free-run prediction *diverge* (NRMSE explodes, often to
   NaN/Inf) — proving that `rho < 1` is what makes the method work.

## How it differs from the other sequence examples

- **`examples/DiagonalSSM`** *trains* its diagonal linear recurrence
  `h_t = a*h_{t-1} + b*x_t` end-to-end by gradient descent; the learned decay
  spectrum is the whole point. The ESN does the opposite: it **freezes** the
  recurrence at random init and trains only the readout.
- The **causal-conv / attention** baselines likewise *learn* their sequence
  mixer's parameters via backprop. The ESN's mixer (`W`, `W_in`) is never
  trained — its richness comes purely from being a large random nonlinear
  dynamical system held just inside the edge of stability (`rho < 1`).

## How to run

This is a Lazarus project (`.lpi` + `.lpr`). It is pure CPU and dependency-free
and finishes in a few seconds on one thread.

Compile directly with FPC (units live in `../../neural`):

```
fpc -O3 -Mobjfpc -Sc -Sh -Fu../../neural EchoStateNetwork.lpr
./EchoStateNetwork
```

(Or open `EchoStateNetwork.lpi` in Lazarus and run.)

Do **not** commit the compiled `EchoStateNetwork` binary or `.o` files — they
are covered by `examples/.gitignore` and the root `.gitignore`.

## Sample output

```
Echo State Network (Reservoir Computing, Jaeger 2001)
Reservoir N=100  leak=0.30  sparsity=0.10  target rho=1.50
Task: one-step prediction of sin(0.2 t) + 0.3 sin(0.31 t).
================================================================

[1] Building reservoir at rho=1.50 ...
    measured spectral norm sigma_1 of raw W = 3.6656  -> W rescaled so rho <= 1.50
    training the linear readout (600 epochs)...
    teacher-forced one-step NRMSE = 0.0161
    persistence baseline   NRMSE = 0.2136
    free-run (autonomous)  NRMSE = 0.2061

Free-run waveform   ( . = true   o = predicted   * = overlap ):
  step |---------------------------------------------------|
     0 |                                        *          |
     3 |                             o.                    |
     6 |                   o .                             |
    ...

================================================================
[2] ABLATION - rebuilding reservoir at rho=3.00 (> 1, echo-state property BROKEN)
    measured spectral norm sigma_1 of raw W = 3.9123
    free-run (autonomous)  NRMSE = Nan  (expected to explode)

================================================================
Correctness checks:
  PASS  teacher-forced NRMSE 0.0161 < 0.5 x persistence 0.2136
  PASS  rho<1 free-run NRMSE 0.2061 < 0.5
  PASS  rho>1 free-run NRMSE Nan explodes vs rho<1 0.2061
================================================================
ALL CHECKS PASSED
```

(NRMSE = RMSE normalised by the target standard deviation; `1.0` means "no
better than predicting the mean".)
