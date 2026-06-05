# TunedLens

Tiny example for `TNNet.TunedLensReport`, the **learned sibling** of the logit
lens — the **tuned lens** (Belrose et al. 2023, *"Eliciting Latent Predictions
with the Tuned Lens"*). Where the logit lens splices a raw hidden activation
straight into the model's **own frozen head**, the tuned lens first passes that
activation through a small **per-layer learned affine translator** (one
`TNNetFullConnectLinear` of the head-input width, plus bias) that is **trained**
to map the layer's residual state into the **final-layer basis** *before* the
frozen head decodes it — correcting the representation drift that makes the raw
logit lens biased / mis-calibrated at early depths.

The program builds the **same** small constant-width softmax classifier as the
LogitLens example (`6 -> FC10+ReLU x4 -> FC4 -> SoftMax`) on the synthetic
4-class **XOR-quadrant + concentric-rings** task, trains it briefly, then prints
**both** `TNNet.LogitLensReport` and `TNNet.TunedLensReport` on the **same
UNLABELLED** probe batch so the two lenses are directly comparable in one run.

## How the tuned lens works

The trunk and head are **frozen**. For every lens-compatible layer `L` (flat
activation shape-compatible with the head input), a private throw-away mini-net
`Input -> Translator(FCLinear, identity-seeded) -> clone of the frozen head` is
fit by minimising `KL` to the model's **own** final output distribution on the
unlabelled probe batch (**distillation-to-self** — no ground-truth labels; with
a softmax head, backpropagating `p_final` as the soft target *is* exactly that
KL gradient). The model's own weights are **never** stepped.

## What it reports

Per compatible layer, **side by side** with the raw logit-lens columns:

- the tuned-lens **`KL-to-final`** and **entropy** next to the logit-lens ones;
- the tuned-lens **agreement** with the final argmax;
- a paired **`KL-to-final` curve** (logit `.` vs tuned `#`) — the tuned curve
  should sit **lower** at the early/middle layers (it **commits earlier** and
  tracks the final answer more faithfully — the headline Belrose result);
- the aggregate **mean `KL-to-final`** for the logit lens, the **untrained**
  (identity) tuned lens, and the **trained** tuned lens.

## Built-in correctness checks (printed as PASS/FAIL)

1. An **untrained** (identity-seeded) translator does **no better** than the raw
   logit lens — its mean `KL-to-final` **ties** the logit lens (no free lunch
   before fitting).
2. **Fitting** the translators **lowers** the mean `KL-to-final`.
3. At the **head input** the translator collapses to the **identity**, so there
   `tuned == logit == final` (max `|dp| ~ 0`).

## Distinct from the neighbours

- `LogitLensReport` fits **zero** parameters (the tuned lens fits one affine
  translator per layer).
- `LinearProbeReport` fits a fresh **label-supervised** ridge probe per layer;
  the tuned lens is label-**free** and reuses the model's own **frozen** head.

## Build & run

```
cd examples/TunedLens
lazbuild TunedLens.lpi
../../bin/x86_64-linux/bin/TunedLens
```

Pure CPU, no dataset download, total runtime well under a minute (~1 s here).
