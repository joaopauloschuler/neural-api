# MuonOptimizer

A self-contained demo of the **Muon** optimizer (Newton-Schulz orthogonalized
momentum, Jordan et al. 2024, <https://kellerjordan.github.io/posts/muon/>) on a
tiny synthetic regression toy. The Muon arm drives the **library optimizer**
(`TNeuralOptimizerMuon`, via `TNNet.CalcMuonDelta` → `UpdateWeightsAdam`) from a
hand-rolled per-step training loop (no `TNeuralFit`), in the same idiom as
[`SharpnessAwareMinimization`](../SharpnessAwareMinimization). The standalone
`NewtonSchulz5` in the source is kept only for the headline orthogonality probe;
the weight update itself is the real facility.

## The library optimizer

The orthogonalized-momentum step lives in `neuralfit.pas` as
**`TNeuralOptimizerMuon`** (a `TNeuralOptimizer` subclass, sibling of
`TNeuralOptimizerLion` / `TNeuralOptimizerAdafactor`). With `TNeuralFit` you just
set it as the optimizer; it keeps a **single** momentum buffer (like Lion),
applies the Newton-Schulz orthogonalized step to genuine 2-D weight matrices
(`FullConnect` / linear layers) and falls back to **SGD-momentum** on non-matrix
params (biases, conv kernels) — mirroring real Muon, which routes vector params
to a scalar optimizer. Like the other custom optimizers it requires
`SetBatchUpdate(True)` (`TNeuralFit` already sets it).

## What it is

Muon turns the raw gradient of each 2-D weight matrix into an **orthogonalized
momentum** step. Per dense layer, per step:

1. momentum buffer `M <- mu*M + G` (`G` = the layer's accumulated gradient);
2. **orthogonalize** the update: `O <- NewtonSchulz5(M)`, i.e. replace `M` by
   the (approximately) nearest orthogonal matrix using ~5 fixed quintic
   Newton-Schulz iterations on the Frobenius-normalized `X = M / ||M||_F`:

   ```
   X <- a*X + b*(X X^T X) + c*(X X^T)^2 X      (a,b,c) ~ (3.4445, -4.7750, 2.0315)
   ```

3. apply `W <- W - lr * sqrt(max(rows,cols)) * O`. The `sqrt(max(rows,cols))`
   factor makes the per-element update RMS roughly match Adam's.

The 5-step quintic is a *deliberately approximate* orthogonalizer: its stable
fixed points are `sigma ~ 0.868` and `~1.264` (`f(1) = 0.701`, so `sigma = 1`
is **not** a fixed point). It squeezes every singular value into roughly
`[0.7, 1.3]` — "semi-orthogonal", which is all Muon needs to make the update
directions near-isotropic.

## How it maps onto this library

- Training runs in **batch-update mode** (`NN.SetBatchUpdate(True)`) so
  `Backpropagate` *accumulates* the gradient into `Neurons[].Delta`
  (`FDelta = -lr*grad`) instead of applying it per sample. This is a known
  gotcha in this repo — without it the deltas are zeroed and there is nothing to
  orthogonalize. We read `G = -Delta` straight out of that tensor (the constant
  `lr` folds into the step scale).
- For a `TNNetFullConnectLinear` layer, **neuron `n` owns row `n`** of the
  weight matrix (`Weights` = `FanIn` values) — exactly the `(FanOut x FanIn)`
  layout `TNNet.CalcMuonDelta` packs, orthogonalizes (Newton-Schulz through
  `TNNetVolume.DotProducts`), and writes back into every neuron's `Delta` as
  `-lr*sqrt(max(rows,cols))*O`. The momentum buffer is each neuron's
  `FBackInertia` (zeroed by `NN.ClearInertia()` at setup) — a single buffer,
  like Lion.
- **Biases are suppressed model-wide** (`pSuppressBias = 1` on every dense
  layer). The library's only public bias accessor is read-only, and Muon's paper
  routes vector params to a scalar optimizer anyway; dropping biases keeps the
  three-way comparison apples-to-apples and lets Muon fully own every trainable
  tensor.

### Not a forward-weight reparametrization

Muon normalizes the **update direction** each step; it never changes the forward
pass. This is distinct from the differentiable forward-weight normalizers in
this library — `TNNetWeightNormLinear` and `TNNetWeightStandardization` — which
reparametrize the **weights the forward pass reads** (and are trained *through*).
Here `W` is an ordinary dense weight matrix; we only intercept how its gradient
is turned into a step.

## What it reports

1. **Headline orthogonality check (PASS/FAIL, `Halt(1)` on failure).** A random
   probe matrix is orthogonalized; the example asserts the singular values land
   in the Muon band — `||O^T O - I||_F` is *bounded* (not ~0) and the top
   singular value (cross-checked with `TNNet.EstimateSpectralNorm`) is in
   `[0.65, 1.35]`.
2. **Bake-off:** the same tiny `3 -> 16 -> 16 -> 1` MLP trained three times at
   matched seed/data/epochs — **SGD-momentum vs Adam vs Muon** — with a printed
   train-MSE-vs-epoch table, final MSE, wall-clock and per-step timing, plus
   ASCII loss curves. (Muon uses its own smaller `lr` so its post-`sqrt`-scaled
   step matches the others.)

## How to run

```
lazbuild examples/MuonOptimizer/MuonOptimizer.lpi --build-mode=Default
./bin/x86_64-linux/bin/MuonOptimizer
```

or directly with fpc (mirroring the sibling examples' unit paths):

```
cd examples/MuonOptimizer
fpc -dRelease -dAVX2 -O3 -Mobjfpc -Sh \
    -Fu../../neural \
    -Fu<lazarus>/components/lazutils/lib/x86_64-linux \
    MuonOptimizer.lpr
./MuonOptimizer
```

Pure CPU, no external data, deterministic (fixed seed), tiny dims, finishes in
about 1 second.

## Sample output

```
Headline check: Newton-Schulz drives the singular values into ~[0.7,1.3].
  probe matrix 16x3:  ||O^T O - I||_F = 6.155890E-001   top sigma = 1.132982
  orthogonality (semi, Muon band): PASS

Bake-off (final train MSE, lower is better):
  epoch    SGD-momentum         Adam            Muon
      1       0.352790       0.115187       0.283280
     10       0.004544       0.002078       0.003270
     40       0.001877       0.001156       0.001796

Final train MSE:  SGD-momentum=0.001877   Adam=0.001156   Muon=0.001796
```

On this small noise-free regression all three optimizers converge to comparable
loss; Muon tracks SGD-momentum closely while Adam edges slightly ahead. The toy
is too small for Muon's orthogonalization to pay off — the point of the example
is the *mechanic* (and the orthogonality guarantee), not winning the bake-off.
```
