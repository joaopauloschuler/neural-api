# Forward-Forward

Reproduces **Geoffrey Hinton's 2022 *Forward-Forward (FF) Algorithm***
("The Forward-Forward Algorithm: Some Preliminary Investigations") on a
pure-CPU toy, using only existing in-tree layers (no new layer class is added).

## Why this example is unique in the tree

This is the **first and only example that does NOT learn by end-to-end
backpropagation.** Every other example forms **one global loss** and
backpropagates it through the whole stack. FF does neither:

- It replaces the forward+backward pair with **two forward passes** — one on
  **positive** (real) data and one on **negative** (fake) data.
- It trains **each layer greedily by its own LOCAL objective**. No gradient
  ever flows from one layer into the layer below.

It is explicitly **distinct** from the other "unusual training" examples here:

- **Not the SAM / Lookahead gradient-surgery demos.** Those still form a global
  loss and backpropagate it; they only *manipulate* the global gradient (ascent
  step, slow/fast weights). FF never forms a global loss and never chains a
  backward pass between layers.
- **Not layerwise *unsupervised* pretraining** (greedy autoencoder / RBM
  stacking). FF's per-layer objective is **supervised**: it is the positive vs
  negative *goodness contrast* using the label embedded in the input, not a
  reconstruction or a generic representation objective.
- **Unrelated to the activation / optimiser bake-offs**, which are ordinary
  backprop training runs that vary one knob.

## The algorithm (reusing only existing layers)

Network: `Input -> [TNNetFullConnectReLU -> TNNetL2Normalize] x2`.

- **Length normalisation is essential.** Each `TNNetL2Normalize` (full-volume
  mode, `Create(1)`) rescales a hidden vector to unit length, so a layer feeds
  only the **direction** of its activity to the next layer, never its
  **magnitude**. Without it, every layer could trivially win its own goodness
  game just by reading and re-scaling the previous layer's length, and no real
  features would be learned.
- **Goodness** of a layer is `G = sum(activation^2)` over its units
  (`TNNetVolume.GetSumSqr` on that layer's `Output`).
- **Local objective per layer** — push `G` *above* a threshold `theta` on
  positives and *below* it on negatives, via the logistic loss
  `log(1+exp(-(G-theta)))` (positives) / `log(1+exp(+(G-theta)))` (negatives).
- **Local gradient.** The per-unit gradient w.r.t. the layer's
  (pre-normalisation) activation `a_j` is `dL/dG * 2 * a_j`, with
  `dL/dG = -sigmoid(-(G-theta))` for positives and `+sigmoid(+(G-theta))` for
  negatives. We write that vector into the layer's `OutputError` and call the
  layer's **own** `BackpropagateCPU` — which accumulates the weight gradient
  into `Neurons[].Delta` using the ReLU mask and the normalised input feeding
  the layer, and crucially does **not** pass any error to the layer below. That
  "no downward error" is the defining FF property.

### The critical implementation gotcha

The accumulated-gradient idiom needs **`NN.SetBatchUpdate(True)`**. Under the
library's per-sample default, `Backpropagate`/`BackpropagateCPU` applies the
weight update immediately and leaves `Neurons[].Delta` at zero, so the FF
mini-batch accumulation would be a silent no-op. With batch mode on, deltas are
**summed** (not averaged) across the batch, so each per-sample local gradient is
pre-scaled by `1/batch` to get the batch mean. One `NN.UpdateWeights()` per
mini-batch then takes the step (momentum `0.9`).

## Classification — Hinton's label-in-input trick

The one-hot class label is overlaid in the **first `cClasses` input slots**
(raw features follow). A **positive** sample carries the **correct** label; a
**negative** sample carries a **wrong** label (same features). At **inference**
the net is run once per candidate label and the label whose **accumulated
goodness** (summed over both FF layers) is highest is chosen. The net never
emits a class directly — classification falls out of "which label makes the
features look real".

## The task

A tiny synthetic **4-way Gaussian-blob** problem in 2D (`cFeat=2`, input dim
`6`), two `30`-unit FF layers, `400` train / `400` test points, `60` greedy
epochs. Single-threaded with a fixed `RandSeed = 424242`, so it is fully
deterministic and finishes in about a second.

## Built-in correctness gates (`Halt(1)` on failure)

1. **GATE 1** — after training, mean **positive** goodness must exceed mean
   **negative** goodness at **every** FF layer (the local contrast actually
   separated the two streams).
2. **GATE 2** — the goodness-argmax classifier must **beat chance** on the
   held-out set by a clear margin (`> 1/cClasses + 0.15`).

## How to run

```
# From this directory, with Free Pascal (fpc) installed:
LAZUTILS_PATH=/usr/share/lazarus/4.4.0/components/lazutils/lib/x86_64-linux
fpc -B -Fu../../neural -Fu"$LAZUTILS_PATH" -Mobjfpc -Sh -O2 ForwardForward.lpr
./ForwardForward
```

(Adjust the LazUtils path for your install; it is only needed for the
`UTF8Process` unit pulled in by `neuralthread`.) Or open `ForwardForward.lpi`
in Lazarus and build the *Release* mode. Pure CPU, no external data,
deterministic.

## Sample output (real, seed 424242)

```
================================================================
Forward-Forward: per-layer LOCAL goodness training (NO backprop).
================================================================
Task: 4 Gaussian blobs in 2D; label one-hot overlaid in the
first 4 input slots (input dim=6).  Net: Input -> [FCReLU(30)->L2Norm]
-> [FCReLU(30)->L2Norm].  theta=2.00  LR=0.020  mom=0.90  epochs=60
batch=20  train=400  test=400  RandSeed=424242
POSITIVE = correct label; NEGATIVE = a wrong label. Each FF layer is
trained ONLY by its own goodness contrast; no error crosses layers.

Pre-training goodness (untrained net):
  FF layer 0: pos=15.485  neg=14.221
  FF layer 1: pos=0.654  neg=0.601

Training (FF, two forward passes per sample) .......

=== Per-layer goodness on held-out set (after training) ===
theta = 2.00.  POSITIVE should sit clearly ABOVE NEGATIVE.
  layer    posG    negG  margin | pos bar / neg bar (scaled)
  FF 0    4.742   0.653   4.089 | ########################
                                  | ###
  FF 1    2.438   1.752   0.686 | ############
                                  | #########

=== Goodness-argmax classifier (held-out) ===
  accuracy = 0.897   (chance = 0.250 for 4 classes)

=== Correctness gates ===
[PASS] GATE 1: positive goodness exceeds negative goodness at EVERY FF layer.
[PASS] GATE 2: goodness-argmax accuracy 0.897 beats chance 0.250 by a clear margin.

Total wall-clock: 1.0 s
=> ALL GATES PASS: Forward-Forward learned features by a purely local objective.
```

### Reading the output

The untrained net's goodness is dominated by raw input magnitude (layer 0 sits
near `15`), and positive/negative are barely distinguishable. After 60 greedy
FF epochs the **local** objective alone has driven a clean separation —
positive `4.74` vs negative `0.65` at layer 0, positive `2.44` vs negative
`1.75` at layer 1 — and the goodness-argmax read-out reaches **89.7%** on
held-out data versus a **25%** four-class chance baseline. No global loss was
ever formed and no error ever crossed a layer boundary.

## Notes / honesty (tuning caveats)

In the spirit of this repo's `Grokking` / `RandomLabelMemorization` caveats: FF
is **sensitive to `theta`, the per-layer learning rate, and how negatives are
generated.** The shipped settings are tuned to pass deterministically at the
fixed seed on a small task. If you change the task and accuracy refuses to clear
chance within budget, the honest move is to **report which knob failed** rather
than weaken a gate:

- **`theta` too high** relative to the post-normalisation activation scale →
  positives can't get `G` above it; both streams collapse below threshold and
  GATE 1's margin shrinks to noise.
- **Learning rate too high** → the logistic gradient saturates / oscillates and
  goodness diverges; too low and 60 epochs are not enough to separate the
  streams.
- **Negatives too easy** (e.g. always the same wrong label) → the layer learns a
  shortcut that does not generalise, GATE 1 passes but GATE 2 (accuracy) lags.
  Here negatives are a *uniformly random* wrong label per pass.
- This is a *toy* reproduction; the exact goodness values are seed-dependent.
  The robust, reproducible claim is the **shape**: a purely local pos/neg
  contrast, with no backprop, separates the streams at every layer and yields a
  goodness-argmax classifier well above chance.

## References

- G. Hinton, *The Forward-Forward Algorithm: Some Preliminary Investigations*,
  2022. https://arxiv.org/abs/2212.13345
