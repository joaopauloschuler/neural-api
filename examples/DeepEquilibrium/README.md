# Deep Equilibrium Model (implicit fixed-point depth)

A tiny, synthetic, **no-download** demonstration of the Bai, Kolter & Koltun 2019
[*Deep Equilibrium Models*](https://arxiv.org/abs/1909.01377) (DEQ) idea, built
on `TNNet.AddDeepEquilibriumBlock`.

## The idea

A residual stack `x_{n+1} = x_n + f(x_n)` has a **fixed** depth. A DEQ instead
defines its output as the **fixed point** `z* = f(z*; x)` of *one* weight-tied
transform `f`, reached by iterating the map until it stops moving:

```
z_0 := 0
repeat:   z_{k+1} := f(z_k + x)     until ||z_{k+1} - z_k|| < tol  (or MaxIters)
```

The input `x` is **injected additively** every iteration. Because the **same**
`f` (the same weights) is reused at every step, the parameter count is
**independent of the iteration count** — an *"infinite-depth, weight-tied"*
network whose **effective depth adapts to the input** (the number of iterations
the solve needs is data- and weight-dependent).

Contrast with the explicit-unroll cousin
[`AddNeuralODEBlock`](../NeuralODE/README.md): a Neural ODE runs a **fixed**
number of explicit Euler steps `y := y + h*f(y)` (a known-length forward graph);
a DEQ runs `f` to **convergence** at a depth decided by the fixed-point solve.

## The builder

```pascal
function TNNet.AddDeepEquilibriumBlock(InputLayer: TNNetLayer;
  HiddenDim, MaxIters: integer): TNNetLayer;
```

- `f` is a shape-preserving pointwise 2-layer function over Depth:
  `PointwiseConvReLU(HiddenDim)` → `PointwiseConvLinear(d_model)` (with
  `d_model = InputLayer.Output.Depth`), exactly as in `AddNeuralODEBlock`.
  1×1/pointwise convs keep the sequence axis intact (`FullConnectLinear` would
  flatten it and zero the input gradient).
- **Weight sharing is the whole point.** Iteration 1 creates the two real
  convolution layers; every later iteration reuses their weights via
  `TNNetDeepEquilibriumSharedConv` — a weight-tied convolution that **rebuilds
  its weight cache on every forward pass** so each application of `f` is
  byte-identical (a true fixed-point iteration requires the *same* map every
  step; the plain `TNNetConvolutionSharedWeights` snapshots its cache at build
  time, which goes stale under the usual init-after-build flow).
- **Contraction.** A fixed-point iteration only converges when `f` is a
  contraction. `f`'s output is multiplied by a fixed `BETA ∈ (0,1)` to bound its
  Lipschitz constant and the iterates are **under-relaxed (damped)**
  `z_k := (1-ALPHA)·z_{k-1} + ALPHA·f(u_k)` to damp period-2 orbits. This is *not*
  a guarantee at arbitrary weights — real DEQs use spectral constraints /
  root-finders that this v1 does not implement — but it converges on this task.

## The backward pass — honest disclosure

The exact DEQ gradient needs the **implicit-function theorem** (an
inverse-Jacobian solve), which is awkward under this library's per-layer
`Backpropagate` contract. This builder ships the tractable **jacobian-free /
phantom gradient** (Geng et al. 2021,
[arXiv:2103.12803](https://arxiv.org/abs/2103.12803)): the forward iteration is
run to the fixed point, but **every iterate except the last is detached** (its
injected input passes through `TNNetIdentityWithoutBackprop`), so gradients flow
through only the **final** application of `f`. This is an **approximation**, not
the exact implicit gradient. It trains fine here but is not state of the art; the
exact inverse-Jacobian backward is logged as a follow-up.

## What this example shows

The only trainable trunk of a tiny classifier is **one**
`AddDeepEquilibriumBlock`, trained on a synthetic, nonlinearly-separable **two
concentric rings** task. Per epoch it reports:

- **(a)** the validation accuracy at a **constant** parameter count, and
- **(b)** the **mean forward iteration-count-to-convergence** — the adaptive-depth
  signal. This is data/weight-dependent: on this run it *rises* with training,
  because an untrained `f` is near the zero map (trivial fixed point `z*≈0`,
  reached in very few steps) while a trained `f` has a richer, more distant
  equilibrium that takes more iterations to settle. Either direction is the
  honest adaptive-depth story — the depth is decided by the solve, not fixed up
  front.

It also runs a **param-matched** `AddNeuralODEBlock` (same `f` shape, same weight
count) as **(c)** a side-by-side, to make the explicit-unroll vs
implicit-fixed-point contrast concrete: identical weight budget, but the DEQ's
depth is data-dependent rather than a fixed hyperparameter.

## Setup

- Architecture: `TNNetInput(2)` → `FullConnectLinear(8)` → `Reshape(1,1,8)` →
  **`AddDeepEquilibriumBlock(HiddenDim=16, MaxIters=30)`** →
  `FullConnectLinear(2)` → `SoftMax`. The reshape moves the 8 features into the
  **Depth** axis (`1×1×8`), which the pointwise convs inside the block act on.
- Data: 600 train / 200 validation points on two noisy concentric rings
  (inner = class 0, outer = class 1).
- Optimizer: SGD with momentum, `LR = 0.05`, batch size 32, 25 epochs,
  single-threaded (`MaxThreadNum := 1`) for determinism. Convergence tolerance
  `tol = 1e-2`.

## How to run

```bash
cd examples/DeepEquilibrium
lazbuild DeepEquilibrium.lpi
../../bin/x86_64-linux/bin/DeepEquilibrium
```

or directly with `fpc` (single file, ~35 s on one CPU thread):

```bash
fpc -dUseCThreads -Fu../../neural -Fu<lazutils> -Mobjfpc -Sh -O2 DeepEquilibrium.lpr
./DeepEquilibrium
```

## Sample output

```
Deep Equilibrium Model (implicit fixed-point depth) demo.
Trunk = ONE TNNet.AddDeepEquilibriumBlock(HiddenDim=16), d_model=8, MaxIters=30, tol=0.01.
Task = synthetic 2-class concentric rings. 25 epochs, 600 train / 200 val.
Backward = jacobian-free phantom gradient (detach all but last f).

=== DEQ (adaptive implicit depth) ===
  epoch  val_acc  mean_iters_to_converge
      1    0.760        13.23
      2    0.990        21.69
      3    1.000        23.64
      5    0.990        24.16
     10    0.995        25.11
     15    1.000        25.81
     20    1.000        26.31
     25    1.000        26.60

=== Param-matched Neural ODE (explicit 4-step Euler unroll) ===

=== Summary ===
model                neurons  weights  val_acc
DEQ (implicit)           34      288   1.000
NeuralODE (explicit)     34      288   0.790

DEQ adaptive depth: mean iters-to-converge went from 3.64 (untrained, near-zero f, trivial fixed point) to 26.60 (trained, richer equilibrium).
Both trunks carry the SAME weight count; the DEQ reaches its answer at
a data-dependent depth (the implicit fixed point) instead of a fixed
hand-picked number of explicit steps.

Total wall time: 37.41 s
```

The DEQ trunk and the param-matched Neural ODE carry the **same** weight count
(34 neurons / 288 weights); the DEQ solves the rings task perfectly while
reaching its answer at a **data-dependent depth** (the implicit fixed point).
Exact numbers vary a little with platform / float build, and the Neural ODE
accuracy in particular is sensitive to the (deliberately un-tuned, shared) LR.
