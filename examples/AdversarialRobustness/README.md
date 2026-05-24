# AdversarialRobustness

Tiny example for `TNNet.AdversarialRobustnessReport`, a forward+backward
adversarial-robustness diagnostic based on the **fast gradient sign method**
(FGSM, Goodfellow et al. 2015).

## What is FGSM?

FGSM crafts a worst-case input perturbation in a single backward step. Given a
labelled sample `(x, y)` it computes the gradient of the loss with respect to
the *input* and steps the input along its sign:

```
x_adv = x + eps * sign(d loss / d x)
```

The sign-step puts equal-magnitude pressure on every input element in the
direction that most increases the loss, so a small `eps` is often enough to flip
a confidently-correct prediction. FGSM is the canonical cheap probe of how close
a model's decision boundary sits to its real inputs.

The diagnostic needs the gradient that lands on the **input layer**, which is
off by default (the input never needs a gradient for ordinary training), so the
report calls `TNNet.EnableInputGradient` once before the backward passes. The
last-layer output error is the cross-entropy gradient `softmax_output -
one_hot(label)` — exactly what `TNNet.Backpropagate(target)` produces.

## What the report shows

For a labelled probe batch and a menu of epsilons it reports:

- **(a)** top-1 accuracy at each `eps` as a degradation curve (`eps=0` is the
  clean baseline — a built-in sanity check that it matches a plain evaluation
  pass);
- **(b)** the per-sample **critical epsilon** (smallest `eps` that flips the
  prediction away from the clean argmax) as a 10-bin ASCII histogram, so the
  spread between fragile and robust inputs is visible;
- **(c)** the mean clean-confidence (max softmax probability) of the samples
  that flip **earliest** vs those that survive **longest** — the
  high-confidence-yet-fragile failure mode;
- **(d)** per-class top-1 accuracy at the median `eps`, so a class whose
  decision boundary sits unusually close to its inputs stands out;
- **(e)** a one-line verdict `robust` / `moderately fragile` / `fragile` from
  the accuracy drop at the median `eps`.

The inspected network is **frozen**: deltas are cleared and `UpdateWeights` is
never called, so the trained weights are never modified. This is an *evaluation*
of robustness, not adversarial *training*.

`AdversarialRobustnessReport` is distinct from `SaliencyReport` (per-sample
input attribution — *where* the model looks, not *how far* an input can be
pushed), `EquivarianceReport` (fixed symmetry transforms — invariance, not
worst-case accuracy), `LayerSensitivityReport` (jitters *weights*, not inputs)
and `DecisionBoundaryReport` (renders the 2-D learned function over a grid).

## What this demo shows

On a synthetic 3-class 2-D problem (three Gaussian blobs) it trains a small
`FC -> FC -> SoftMax` classifier, then runs the report over a small labelled
probe batch. For contrast it trains a **second** model with input-noise
augmentation (Gaussian jitter on every training input — a cheap robustness
regulariser) and runs the report again; its degradation curve should fall off
more slowly, the eyeballable expected effect.

## Build & run

```
cd examples/AdversarialRobustness
lazbuild AdversarialRobustness.lpi
../../bin/x86_64-linux/bin/AdversarialRobustness
```

Pure CPU, no dataset download. Total runtime is well under a minute.
