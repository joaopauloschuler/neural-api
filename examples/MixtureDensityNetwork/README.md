# Mixture Density Network

A self-contained, pure-CPU reproduction of **Bishop's Mixture Density Network**
(Bishop 1994, *Mixture Density Networks*, NCRG/4288 — also Bishop, *PRML* §5.6)
on the classic **inverse-problem toy**. It contrasts a plain MSE regressor
(which provably collapses to the conditional mean) against an MDN head that
recovers the full **multimodal** conditional `p(y|x)`.

The MDN head is the **library layer `TNNetMixtureDensity`** (with its own
log-sum-exp negative-log-likelihood backward) on top of a small
`TNNetFullConnect` MLP trunk — no hand-coded mixture math.

## The phenomenon

Generate a **forward** map

```
x = y + 0.3*sin(2*pi*y) + noise,   y ~ U[0,1]
```

then train a network to predict `y` **from** `x` (the *inverse* map). The
forward curve folds back on itself, so over the middle band a single `x` is
reached by **three different `y`** — the inverse is one-to-many and the
conditional `p(y|x)` is genuinely **multimodal**.

A plain MSE regressor minimises `E[(y - f(x))^2]`, whose minimiser is the
**conditional mean** `E[y|x]`. In the fold region that mean lands in the
**low-density gap between the modes** — a `y` value the data almost never takes.
This is the textbook failure that MDNs exist to fix.

## The fix (Bishop 1994)

Model the conditional as a **Gaussian mixture** whose parameters are functions
of `x` emitted by the net:

```
p(y|x) = sum_k  pi_k(x) * Normal(y ; mu_k(x), sigma_k(x)^2)
```

The trunk emits `K*(1+2*D)` **raw** outputs along the **depth axis**, which the
library head `TNNetMixtureDensity(K, D)` turns into the mixture parameters in
place. For `D = 1` the depth packing is
`[ a_0..a_{K-1} | m_0..m_{K-1} | s_0..s_{K-1} ]`:

| parameter | from raw output | activation               | constraint        |
|-----------|-----------------|--------------------------|-------------------|
| `pi_k`    | `a_k`           | softmax over the `a`     | `sum_k pi_k = 1`  |
| `mu_k`    | `m_k`           | identity (linear)        | —                 |
| `sigma_k` | `s_k`           | `softplus(s_k)`          | `sigma_k > 0`     |

trained on the mixture **negative log-likelihood**

```
NLL = -log( sum_k pi_k * Normal(y ; mu_k, sigma_k) ).
```

## How it trains (the library head owns the loss)

The MDN arm is `Input(1) -> Tanh -> Tanh -> Linear(K*(1+2*D)) ->
TNNetMixtureDensity(K, D)`. The library head applies the softmax/softplus
transforms in `Compute()` and emits the **exact** responsibility-weighted
`dNLL/dparam` (stable log-sum-exp) in its own `Backpropagate()` — there is no
hand-coded mixture gradient in this example.

We drive that backward the way the layer expects. The framework seeds the head
error as `(output - target)`, and the head recovers the regression target `y`
from its **first `D` channels** as `output - (output - target)`. So we build a
target volume that copies the head output (every channel has zero residual) and
overwrites the first `D` channels with the true `y`:

```
Tgt.Copy(head.Output);  Tgt.FData[0] := y;  NN.Backpropagate(Tgt);
```

The loop runs in **batch-update mode** (`NN.SetBatchUpdate(True)` makes
`Backpropagate` *accumulate* into `Neurons[].Delta`); after each mini-batch we
`MulDeltas(1/batch)` so the applied step is the **mean** gradient and
`UpdateWeights` applies it once. We never call `TNeuralFit.Fit`, so layer
references never go stale. The MSE arm uses the stock `(out - y)` gradient on its
single linear output.

The `K` mean biases are spread across `[0.1, 0.9]` at init (and the scale biases
set so `sigma ~ 0.15`) to break the symmetry that would otherwise collapse every
component onto the conditional mean.

## What it reports

- **Per-arm scores**: mean NLL and point-prediction MSE for the MSE arm, the
  `K=3` MDN, and a `K=1` MDN.
- **Fold-region collapse**: at the probe `x = 0.5` the MSE prediction is shown
  sitting in the low-density gap *between* the MDN's outer component means.
- **Learned `(pi, mu, sigma)`** at several probe `x` — three distinct modes
  appear in the fold band.
- An **ASCII scatter** of sampled MDN predictions (`o`) and the MSE prediction
  (`M`) over the true data (`.`): the `M` curve snakes through the empty middle
  while the `o` samples recover all three branches.

## Built-in correctness invariants (the program `HALT(1)`s if any fail)

1. **K=1 reduction** — a `K=1` MDN's NLL is a homoscedastic Gaussian NLL whose
   only mean is `mu_0`; that `mu` must match the independently-trained MSE arm's
   prediction over a probe grid (both recover `E[y|x]`).
2. **Simplex** — the mixture weights (read from the head output) must sum to 1
   (within `1e-5`) at every probe `x`.

## Run

```
lazbuild examples/MixtureDensityNetwork/MixtureDensityNetwork.lpi --build-mode=Default
./bin/x86_64-linux/bin/MixtureDensityNetwork
```

Deterministic (`RandSeed := 424242`), pure CPU, no external data, finishes in
well under a minute.

## How this is DISTINCT from other in-tree uncertainty work

- **`examples/MCDropoutUncertainty/`** models **epistemic** uncertainty (what
  the model doesn't know) by sampling dropout masks at inference and looking at
  the spread of predictions. It produces an *error bar around a single
  prediction*; it does **not** model a target that is genuinely multi-valued.
  MDN here models **aleatoric** uncertainty — irreducible ambiguity in the data
  itself — and specifically its **multimodal** form.
- The pointwise regression loss heads **`TNNetHuberLoss` / `TNNetCharbonnierLoss`
  / `TNNetLogCoshLoss` / `TNNetWingLoss`** all reshape the *residual penalty* of
  a **unimodal** prediction (robustness to outliers, etc.) but still emit **one
  `y` per `x`** and so collapse in exactly the same way the plain MSE arm does on
  this toy.

**MDN is the only thing in-tree that models aleatoric MULTIMODALity** — several
valid `y` for a single `x`.
