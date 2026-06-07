# EWCContinualLearning

A tiny, pure-CPU reproduction of the classic **catastrophic-forgetting** result and
its **Elastic Weight Consolidation (EWC)** cure — Kirkpatrick et al., *"Overcoming
catastrophic forgetting in neural networks"*, PNAS 2017.

A neural network trained sequentially on Task A then Task B tends to **forget** Task A:
plain SGD on B overwrites the weights A relied on. EWC fixes this by adding a quadratic
penalty that pulls the parameters **most important for Task A** back toward their
Task-A values, where "importance" is the **diagonal Fisher information** of each
parameter — exactly the quantity `TNNet.FisherImportanceReport` computes.

## What it does

1. Builds a small MLP classifier `Input(2) → FullConnectReLU(32) → FullConnectLinear(4) → SoftMax`.
2. Trains it to convergence on **Task A**: four overlapping 2-D Gaussian clusters, one per class.
3. Snapshots two things from the converged model:
   * the **Task-A-optimal weights** `w_A` (a flat copy of every trainable parameter);
   * the **diagonal empirical Fisher** `F_i = E_x[(∂ log p(y|x)/∂θ_i)²]` of every
     parameter, computed the same way as `TNNet.FisherImportanceReport` — accumulate
     the **squared per-parameter gradient** over the Task-A training set on a **frozen**
     net (`SetBatchUpdate(true)` so the gradient lands in `Neuron.Delta`/`BiasDelta`;
     divide by the layer learning rate to undo the `-LR` scaling; never `UpdateWeights`).
4. Continues training on **Task B**: the **same** clusters and labels but with the input
   coordinates **rotated** (a "perturbed-input" task in the spirit of the paper's
   permuted MNIST). Both tasks share the hidden layer and are individually learnable,
   but they demand overlapping-yet-different features, so their solutions **compete** for
   the same weights — which is what makes the forgetting real and the EWC fix meaningful.
   Two arms:
   * **PLAIN** — ordinary sequential fine-tuning, no penalty.
   * **EWC** — after each data weight step, a decoupled penalty step pulls each parameter
     toward `w_A` with a Fisher-weighted, clamped strength.
5. Prints the headline 2×2 table (Task-A and Task-B accuracy, PLAIN vs EWC) and runs
   built-in PASS/FAIL sanity gates.

## The EWC penalty

The EWC objective adds to the Task-B loss the quadratic penalty

```
  L_EWC = Σ_i (λ/2) · F_i · (w_i − w_A_i)²
```

whose gradient is `λ · F_i · (w_i − w_A_i)`, i.e. a per-parameter pull toward the
Task-A value with strength proportional to that parameter's Fisher importance. We apply
it as a **decoupled** step (like AdamW decouples weight decay) after each data
`UpdateWeights`:

```
  s_i = clamp(LR · λ · F_i, 0, 1)
  w_i ← w_i − s_i · (w_i − w_A_i)
```

The **clamp** matters: the empirical Fisher of a confident, converged model is *tiny*
(here `max F ≈ 1e-6`), so a useful penalty needs a very large `λ` (`~1e8`), and an
unclamped step would then overshoot and oscillate. The clamped pull is unconditionally
stable and monotone in `λ`: **high-Fisher** parameters get `s_i → 1` and are pinned
essentially **at** `w_A`, while the many **low-Fisher** parameters get `s_i → 0` and stay
free to learn Task B — exactly the selectivity EWC is built on.

## Example output

```
After Task A:  Task-A acc = 1.000   Task-B acc = 0.324

  arm     | Task-A acc | Task-B acc
  --------+------------+-----------
  PLAIN   |    0.285   |    1.000
  EWC     |    0.535   |    0.984

Task-A retention gain (EWC - PLAIN): 0.250
Task-B cost          (PLAIN - EWC): 0.016
```

PLAIN fine-tuning learns Task B perfectly but **forgets Task A** (1.000 → 0.285). EWC
**retains Task A far better** (0.535, a +0.25 gain over PLAIN) while still learning Task B
almost perfectly (0.984) — the textbook "retain the old task at a modest cost to the new
one" result.

### Honest notes

* The toy is deliberately small and the clusters **overlap**, so accuracies are not a
  clean 100% and the exact numbers depend on the (fixed) seed. The point is the *contrast*
  between the arms, which the built-in gates enforce: PLAIN must drop Task A by >0.20 and
  EWC must retain Task A >0.15 better than PLAIN, or the program exits non-zero.
* `λ` is large purely because the empirical Fisher of a confident model is tiny; this is a
  property of EWC on well-fit models, not a quirk of this implementation. The clamped step
  makes the result robust to the exact `λ` over a wide range.
* The penalty here is the standard **single-task** EWC (one Fisher snapshot, one anchor
  `w_A`). The multi-task accumulation `Σ_tasks λ_t F_i^t (w_i − w_{t,i})²` is a
  straightforward extension left out to keep the demo small.

## Build & run

```
cd examples/EWCContinualLearning
lazbuild --build-mode=Release EWCContinualLearning.lpi
../../bin/x86_64-linux/bin/EWCContinualLearning
```

Pure CPU, single-threaded determinism (fixed `RandSeed`), no dataset download, ~3 seconds.
