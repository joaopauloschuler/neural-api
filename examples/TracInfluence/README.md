# TracIn training-data attribution (TracInLast)

A self-contained demo of **TracIn** (Pruthi, Liu, Sundararajan & Yan, 2020,
*"Estimating Training Data Influence by Tracing Gradient Descent"*). TracIn
answers a different attribution question from the rest of this repo's
explainability family: instead of attributing a prediction to **input features**
(`SaliencyReport`, `GradCAMReport`) or to **layers/activations**
(`ActivationPatchingReport`), it attributes the prediction back to the
**training examples** that shaped the model.

The influence of a training point `z_train` on a test point `z_test` is the dot
product of their per-sample loss gradients:

```
influence(z_train, z_test) = < grad_loss(z_train), grad_loss(z_test) >
```

- **Positive** influence → a **PROPONENT**: this training point pushed the model
  *toward* the test prediction.
- **Negative** influence → an **OPPONENT**: this training point pushed *against*
  the test prediction.

## What this example does (the graded self-check)

1. Builds a clearly-separable 2-D, 2-class blob task (300 training points,
   60 test points).
2. **Plants exactly ONE mislabelled training example**: a class-0 point sitting
   near the class-1 boundary, with its one-hot label flipped to class 1.
3. Trains a tiny MLP (`Input -> FC+ReLU -> FC -> SoftMax`).
4. Picks a test point the mislabel corrupts, computes the TracIn influence of
   **every** training point on it, and ranks them.
5. Prints the top-K **proponents** and top-K **opponents**, then **asserts** the
   planted mislabel lands among the top-K most-negative opponents — the paper's
   headline result that *TracIn surfaces mislabelled data*. Prints a clear
   `PASS`/`FAIL` line and exits non-zero on FAIL.

## Single-checkpoint form ("TracInLast")

This uses only the **final** trained weights — a single gradient-dot similarity,
**no checkpoint summation**. The full paper sums the gradient dot product over
several training checkpoints (TracIn-CP), each scaled by that interval's learning
rate, to reduce variance. On this toy the single-checkpoint ranking is already
clean (the planted mislabel is the single most-negative opponent, the *only*
training point with negative influence, by a ~2x margin), so multi-checkpoint
summation was **not** needed and is **deferred**.

## Mechanics in this library

Per-sample weight gradients are read out the same way `FisherImportanceReport` /
`GradientConflictReport` do:

- `NN.SetBatchUpdate(true)` so each sample's gradient **accumulates into the
  neuron's `Delta`** instead of being consumed inline.
- `NN.ClearDeltas()` before each backward pass; one `Compute` + one
  `Backpropagate` per example; **never** `UpdateWeights` — the trained net is
  frozen, this is a measurement.
- Each layer's `Neuron.Delta` is flattened (and divided back out by the layer
  learning rate, since `Delta = -LR * grad`) into one vector. The per-sample
  `-LR` sign cancels in the dot product. The per-neuron bias delta is private
  (a single scalar per neuron) and is omitted; the weight gradients alone rank
  TracIn cleanly.

## Cost

`O(N_train)` backward passes **per test point**. `N_train` is deliberately kept
to a few hundred (300 here). The example runs in well under a second on CPU.

## Run it

```bash
cd examples/TracInfluence
LAZUTILS_PATH=$(find /usr/lib/lazarus /usr/share/lazarus -name utf8process.ppu -printf '%h\n' | head -1)
fpc -B -Fu../../neural -Fu"$LAZUTILS_PATH" -Mobjfpc -Sh -O2 TracInfluence.lpr
./TracInfluence
```

or open `TracInfluence.lpi` in Lazarus and run.

## Sample output

```
Planted ONE mislabel: train point #132 (true class 0) relabelled as class 1.

Test point #53: true class 1, predicted class 1, features (1.123, 1.141).

Top-5 OPPONENTS (most NEGATIVE influence):
  #132   influence=-1.90E-005  (label class 1)   <== PLANTED MISLABEL
  #298   influence=+0.00E+000  (label class 0)
  ...

Planted mislabel #132 influence=-1.90E-005, opponent-rank=0 (0 = most negative).

PASS: planted mislabel is among the top-5 most-negative opponents (TracIn surfaced it).
```
