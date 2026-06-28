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
- `W_in` is a random input matrix.
- `W` is a sparse random `N x N` matrix **rescaled to a chosen spectral radius
  `rho < 1`**. That `rho < 1` condition is the **echo-state property**: the
  reservoir asymptotically forgets its initial state, so the same input drives
  it to the same state regardless of where it started. That is what makes a
  *fixed random* recurrence usable as a feature generator.

### The reusable reservoir layer

The recurrent core is the reusable
[`TNNetEchoStateReservoir`](../../neural/neuralnetwork.pas) layer — a
shape-`(SeqLen,1,1)` → `(SeqLen,1,N)` sequence map. The layer **owns** the two
frozen matrices `W_in` and `W`, owns the leak rate `a`, runs the leaky-integrator
recurrence above over the whole driving sequence in one `Compute`, and does the
one-shot spectral-radius rescale of `W` at build time. The matrices are
**non-trainable** (regenerated deterministically from a seed, so they round-trip
through serialization for free) and are never touched by a gradient; only a
linear read-out **downstream** is ever trained. The example wraps the layer in a
one-layer `Input(1) → TNNetEchoStateReservoir(N)` net and reads each output
column `h_t` as a reservoir state.

### Spectral-radius rescaling

At build time the layer reuses the library's power-iteration helper
[`TNNet.EstimateSpectralRadius`](../../neural/neuralnetwork.pas) to **measure**
the scale of `W` instead of running a full eigensolver. Unlike its sibling
`TNNet.EstimateSpectralNorm` — which estimates the spectral **norm** (largest
singular value `sigma_1`) by alternating `W*v` and `W^T*u` steps —
`EstimateSpectralRadius` iterates **only** `v := W*v / ‖W*v‖` (no transpose
step) and returns the Rayleigh-flavoured ratio `rho ≈ ‖W*v‖` at convergence,
i.e. the true spectral **radius** `|lambda|_max` that actually governs the
echo-state property. The layer then scales `W := W * (rho_target / rho)`, which
targets the radius **directly and exactly** — so `rho_target < 1` can be set
straight (here `0.9`). The example prints the measured raw radius via the layer's
`MeasuredRho` property.

### Pipeline

1. Build an `Input(1) → TNNetEchoStateReservoir(N)` net; the layer builds and
   spectral-rescales its own `W_in` and `W` (never touched by a gradient).
2. Run the reservoir **forward** (no gradient) over a training sequence and
   collect each output state `h_t` into a `TNNetVolumePair` (input = `h_t`,
   target = `x_{t+1}`).
3. Train **only** a `TNNetFullConnectLinear(1)` readout on those collected
   pairs. Two arms are trained and compared on the **same** reservoir, **same**
   collected states and **same** error metric:
   - an iterative, LR-sensitive **SGD** loop (a tiny L2-regularised linear fit);
   - the classic **closed-form ridge (Tikhonov) solve** — one shot, no LR.

### Closed-form ridge readout (the classic ESN training)

Because the readout is *linear* in the reservoir state, its optimal weights are
not something to chase with SGD — they are the one-shot ridge-regression
solution. Collect the state matrix `S` (rows = training timesteps, columns = the
`N` reservoir units **plus a bias/intercept column of 1s**) and the target
matrix `Y` (one column, `x_{t+1}`). The ridge readout minimises
`||S·Wout − Y||² + lambda·||Wout||²`, whose normal equations are

```
(Sᵀ S + lambda·I) · Wout = Sᵀ Y          ->   A · Wout = B
```

The example forms `A` (size `(N+1)×(N+1)`) and `B` (`(N+1)×1`) and solves the
small dense system directly via the shared library routine
`NeuralLinearSolve` in `neuralvolume.pas` — Gauss-Jordan elimination with
partial pivoting — the single reusable dense solver also used by the library's
closed-form least-squares head, exact for this reservoir size. The solved
`Wout` is then packed back into the **same**
`Input(N)→FullConnectLinear(1)` net shape as the SGD arm (reservoir weights into
the neuron's `Weights`, the intercept into its `BiasWeight`), so both arms are
evaluated by identical code. No learning rate, no epochs, no shuffling — it is a
single linear solve, deterministic and not LR-sensitive.

#### Lambda sweep

The ridge arm runs a small regularisation sweep `lambda ∈ {0, 1e-6, 1e-4, 1e-2}`
and prints the teacher-forced and free-run NRMSE for each. This shows the
regularisation behaviour directly: at `lambda = 0` the readout nails the
teacher-forced one-step prediction but a tiny unregularised readout *amplifies*
error in the autonomous feedback loop, so its free-run NRMSE explodes; a modest
`lambda` damps the readout and stabilises the free-run. The headline picks the
lambda with the best **free-run** NRMSE (the metric that matters for autonomous
generation) and contrasts it with the SGD arm:

> **The closed-form ridge readout matches or beats the SGD readout in one shot,
> with no learning rate to tune.**

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
Reservoir N=100  leak=0.30  sparsity=0.10  target rho=0.90
Recurrent core: reusable TNNetEchoStateReservoir layer.
Task: one-step prediction of sin(0.2 t) + 0.3 sin(0.31 t).
================================================================

[1] Building reservoir at rho=0.90 ...
    measured raw W: spectral RADIUS rho = 2.0081
    -> W rescaled by the layer so its true spectral radius = 0.90
    training the linear readout (600 epochs)...
    teacher-forced one-step NRMSE = 0.0189
    persistence baseline   NRMSE = 0.2136
    free-run (autonomous)  NRMSE = 0.0948

Free-run waveform   ( . = true   o = predicted   * = overlap ):
  step |---------------------------------------------------|
     0 |                                        *          |
    ...

----------------------------------------------------------------
[1b] Closed-form RIDGE readout  Wout = (S^T S + lambda I)^-1 S^T Y
     one-shot solve (no LR, no epochs); lambda regularisation sweep:
       lambda     teacher-NRMSE   free-run-NRMSE
       0.0E+000         0.0020         6.4692
       1.0E-006         0.0015         0.4024
       1.0E-004         0.0336        12.2650
       1.0E-002         0.0071         0.0492

     SGD-vs-ridge headline (same reservoir, same task):
       SGD readout   (600 epochs, LR=0.02): teacher 0.0189  free-run 0.0948
       ridge readout (one-shot, lambda=1.0E-002):  teacher 0.0071  free-run 0.0492

================================================================
[2] ABLATION - rebuilding reservoir at rho=1.80 (> 1, echo-state property BROKEN)
    measured raw W: spectral RADIUS rho = 2.0081
    free-run (autonomous)  NRMSE = Nan  (expected to explode)

================================================================
Correctness checks:
  PASS  teacher-forced NRMSE 0.0189 < 0.5 x persistence 0.2136
  PASS  rho<1 free-run NRMSE 0.0948 < 0.5
  PASS  rho>1 free-run NRMSE Nan explodes vs rho<1 0.0948
================================================================
ALL CHECKS PASSED
```

(NRMSE = RMSE normalised by the target standard deviation; `1.0` means "no
better than predicting the mean".)
