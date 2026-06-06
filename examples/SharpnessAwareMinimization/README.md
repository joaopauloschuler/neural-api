# SharpnessAwareMinimization

A self-contained demo of **Sharpness-Aware Minimization** (SAM, Foret et al.
2021, <https://arxiv.org/abs/2010.01412>) on a tiny noisy-label 2D
classification toy, with a hand-rolled mini-batch training loop (no
`TNeuralFit` surgery).

## What SAM is

SAM replaces the usual single gradient step with a **two-pass** step that
minimises the loss of the *worst point in a `rho`-ball* around the current
weights `w`, biasing training toward **flat minima** (which generalise better
and are more robust to label noise):

1. forward+backward on a batch to get the gradient `g = dL/dw`;
2. climb to the worst-case neighbour `w_adv = w + rho * g/||g||` (the *ascent*
   step: snapshot + perturb the whole net);
3. a SECOND forward+backward AT `w_adv` to get the perturbed gradient `g_adv`;
4. restore `w` and apply `g_adv` with the normal (plain-SGD) optimizer.

## How it maps onto this library

- The classifier runs in **batch-update mode** (`NN.SetBatchUpdate(True)`), so
  `Backpropagate` accumulates the gradient into `Neurons[].Delta` as
  `FDelta = -learningRate * grad` instead of applying it per sample. SAM reads
  that tensor directly: the ascent direction `g/||g||` equals
  `-Delta/||Delta||`, so the perturb is `w_i += -rho * Delta_i / ||Delta||`.
  `||g||` is the **global L2 norm** over every weight-gradient tensor.
- The whole-net snapshot for step (4) is a deep copy of each neuron's `Weights`
  volume (the same snapshot/restore trick used by `LossLandscapeProbe` and
  `LayerSensitivityReport`); restore is `Weights.Copy(snapshot)`.
- Only the **weight** tensors are perturbed (a common SAM variant). Biases are
  never perturbed — so they need no snapshot — and still receive their normal
  optimizer update. This keeps everything inside the example with no changes to
  `neural/*.pas`.
- Updates are **plain SGD** (momentum = 0), so the extra SAM pass has no
  momentum/Adam state to desync.

## What it reports

1. Final train/val loss + accuracy for **plain SGD vs SAM** at matched
   LR/epochs/seed.
2. The headline **flatness contrast**: `TNNet.LossLandscapeProbe` is run on
   both trained nets (cross-entropy, same random direction via a fixed probe
   seed) and its sharpness scalar + loss-doubling radius are parsed and
   printed. SAM should be flatter/wider.
3. A sweep of `rho in {0.0, 0.01, 0.05, 0.1, 0.2}` with ASCII charts of
   sharpness-vs-rho and val-accuracy-vs-rho, plus the per-rho train loss so the
   fit/flatness trade-off is visible.

## Built-in invariants (acceptance tests)

- **(1) `rho = 0` reproduces plain SGD bit-for-bit.** At `rho=0` the perturb is
  a no-op, the second pass reproduces the first gradient, the restore is a
  no-op, and the applied update is a single plain-SGD step. The example asserts
  the `rho=0` SAM-arm final weights equal the plain-SGD-arm weights with **max
  abs weight+bias diff == 0** and prints `rho=0 == plain SGD: PASS/FAIL`.
- **(2) Higher `rho` trades a little train fit for a flatter minimum.** The
  sweep verifies the sharpness scalar generally **decreases** as `rho` grows
  and documents the observed trend.

## Running

```
lazbuild examples/SharpnessAwareMinimization/SharpnessAwareMinimization.lpi --build-mode=Default
./bin/x86_64-linux/bin/SharpnessAwareMinimization
```

Pure CPU, no external data, deterministic (fixed seed), finishes in well under
a minute.

## Expected reading

On the bundled overlapping-blobs toy with 12% label noise, SAM lands clearly
flatter than plain SGD (smaller sharpness, equal-or-wider loss-doubling
radius), the per-rho train loss rises monotonically with `rho` (the fit/flatness
trade-off), and sharpness falls overall as `rho` rises. Clean validation
accuracy is saturated (the clean clusters are easily separable), so the
flatness and train-loss signals — not val accuracy — carry the story here.
