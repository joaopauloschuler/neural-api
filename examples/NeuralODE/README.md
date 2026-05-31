# Neural ODE (continuous-depth residual)

A tiny, synthetic, **no-download** demonstration of the Chen et al. 2018
[*Neural Ordinary Differential Equations*](https://arxiv.org/abs/1806.07366)
idea, built on `TNNet.AddNeuralODEBlock`.

## The idea

A residual block `x_{n+1} = x_n + f(x_n)` is exactly one explicit **Euler step**
of the ODE `dx/dt = f(x, t)`. A Neural ODE replaces a *stack of distinct
residual blocks* with **one shared** residual function `f` integrated forward
over `Steps` Euler sub-steps of fixed size `h = 1/Steps`:

```
y := x
repeat Steps times:   y := y + h * f(y)
```

Because the **same** `f` (the same weights) is reused at every step, the
parameter count is **independent of `Steps`** — the headline *"depth for free"*
property.

## The builder

```pascal
function TNNet.AddNeuralODEBlock(InputLayer: TNNetLayer;
  HiddenDim, Steps: integer): TNNetLayer;
```

- `f` is a shape-preserving pointwise 2-layer residual function over Depth:
  `PointwiseConvReLU(HiddenDim)` → `PointwiseConvLinear(d_model)`, where
  `d_model = InputLayer.Output.Depth`. 1×1/pointwise convs keep the sequence
  axis intact (`FullConnectLinear` would flatten it and break the residual sum),
  so the block is shape-preserving and can be wrapped in
  `AddPreNormResidual([...])` like any other FFN block.
- **Weight sharing is the whole point.** Step 1 creates the two real
  convolution layers; every later step reuses their weights via
  `TNNetConvolutionSharedWeights` (which links to the step-1 convs and shares
  their `FNeurons`). Without sharing the block would just be an ordinary
  residual stack with `Steps`-many distinct weight sets.
- Each Euler step scales `f`'s output by `h = 1/Steps`
  (`TNNetMulByConstant(h)`) and adds it residually (`TNNetSum`).
- v1 is **Euler-only**; an RK2/midpoint integrator and the O(1)-memory
  adjoint-sensitivity backward pass are logged as follow-ups (see the builder
  doc-comment and `tasklist.md`). Training here uses ordinary stored-activation
  backprop through the unrolled steps.

## What this example shows

The only trainable trunk of a tiny classifier is **one** `AddNeuralODEBlock`.
We sweep `Steps ∈ {1, 2, 4}` (averaging each arm over a few seeds) on a
synthetic, nonlinearly-separable **two concentric rings** task and print, per
`Steps`:

- the trainable parameter count (`CountNeurons` / `CountWeights`), which stays
  **constant** across the sweep, and
- the validation accuracy, which stays **roughly flat and high**.

The learning rate is scaled by `Steps`, because each Euler step multiplies `f`
by `h = 1/Steps`, shrinking the effective gradient reaching `f`'s shared weights
by ~`1/Steps`. Scaling the LR keeps the fixed-epoch comparison fair **without
changing the parameter count**.

## Setup

- Architecture: `TNNetInput(2)` → `FullConnectLinear(8)` → `Reshape(1,1,8)` →
  **`AddNeuralODEBlock(HiddenDim=16, Steps)`** → `FullConnectLinear(2)` →
  `SoftMax`.
  - The reshape moves the 8 features into the **Depth** axis (`1×1×8`), which is
    what the pointwise convs inside the ODE block act on.
- Data: 600 train / 200 validation `(x, y)` points on two noisy concentric rings
  (inner ring = class 0, outer ring = class 1).
- Optimizer: SGD with momentum, `LR = 0.05 * Steps`, batch size 32, 60 epochs,
  single-threaded (`MaxThreadNum := 1`) for determinism.

## How to run

```bash
cd examples/NeuralODE
lazbuild NeuralODE.lpi
../../bin/x86_64-linux/bin/NeuralODE
```

The whole sweep wall-clocks in a few seconds on CPU.

## Sample output

```
Neural ODE (continuous-depth residual) constant-parameter sweep.
Trunk = ONE TNNet.AddNeuralODEBlock(HiddenDim=16), d_model=8, on synthetic 2-class concentric rings.
Sweep Steps in {1,2,4}: parameter count is INDEPENDENT of Steps (one shared f).
60 epochs, 600 train / 200 val pairs, LR=0.05*Steps, RandSeed=42.

Training Steps=1 ... done.
Training Steps=2 ... done.
Training Steps=4 ... done.

=== Steps vs accuracy vs parameter count ===
steps  neurons  weights  val_accuracy (mean of 5 seeds)
    1       34      288      1.000
    2       34      288      0.971
    4       34      288      0.947

Note: weights/neurons are CONSTANT across the sweep (shared f);
accuracy stays roughly flat as Steps grows -- continuous depth at a
fixed parameter budget. This is the Neural ODE "depth for free" point.

Total wall time: 6.99 s
```

The parameter count (34 neurons / 288 weights) is **identical** for every
`Steps`, and validation accuracy stays high and roughly flat as the integration
depth grows — continuous depth at a fixed parameter budget, the Neural ODE
"depth for free" point. Exact numbers vary a little with platform / float build.
