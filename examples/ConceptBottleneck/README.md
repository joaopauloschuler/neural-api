# ConceptBottleneck

Self-contained, **interpretable-by-design** Concept Bottleneck Model (CBM; Koh
et al. 2020, [*Concept Bottleneck Models*](https://arxiv.org/abs/2007.04612))
demo built from **existing layers only** (no new layer type). The network is
forced to route **all** information about the label through a narrow layer of
`K` **human-meaningful concepts**, so the label can only be predicted from those
concepts — which makes the model **editable at test time** by overwriting the
concept values (*concept intervention*).

## The task

Each sample is a 6-D feature vector. Three known boolean **concepts** are read
off the raw features (each pushes one signal feature up or down, the rest is
noise):

- `c0` = *"is bright"*
- `c1` = *"is in top half"*
- `c2` = *"is round"*

The 4-way **label** is a fixed, known function of the concept bits:

```
class = c0 + 2*c1      (c2 is a DECOY concept that does NOT drive the label)
```

so the label is a deterministic boolean/linear function of two of the three
concepts. The third concept is a decoy, present to show the bottleneck still
learns it accurately even though it is irrelevant to the label.

## Architecture (two-stage sequential bottleneck)

```
Input(6)
  -> Trunk: FC+ReLU(16) -> FC+ReLU(16)
  -> TNNetFullConnectSigmoid(K=3)   <- the CONCEPT BOTTLENECK (sole path to label)
  -> TNNetFullConnectLinear(4)      <- the LABEL head reads ONLY the concepts
  -> TNNetSoftMax
Concat([SoftMax, Sigmoid]) -> packed output [ label probs | concept acts ] = width 7
```

`NN.Compute(x)` exposes **both heads at once**: channels `[0..3]` are the label
probabilities, channels `[4..6]` are the sigmoid concept activations. The label
head reads **only** the 3-unit bottleneck, so the model is interpretable by
construction.

## Joint training (deep supervision, manual two-head loss loop)

Automatic `Fit` seeds the gradient only at the last layer, so we seed **both**
heads ourselves — the same packed-target idiom as `examples/EarlyExitNetwork`:

- the **SoftMax block** of the packed output holds label probabilities `p`; a
  one-hot label target there makes `ComputeOutputErrorWith` form `(p - onehot)`,
  the softmax-cross-entropy gradient;
- the **Sigmoid block** holds concept activations `s`; a target equal to the
  ground-truth concept bits makes it form `(s - c)·s'`, the sigmoid
  concept-prediction gradient.

A **single** `Backpropagate` splits the packed error through the `Concat`: the
label gradient flows `softmax → label-linear → bottleneck → trunk`, and the
concept gradient flows directly into the bottleneck (and trunk). The
**concept-loss weight** `lambda` is a knob: with `lambda = 0` we set the concept
target equal to the current activation (`s − s = 0`), i.e. **no concept
supervision** — the bottleneck is free to drift.

**Manual-gradient gotcha:** multi-head accumulation needs `SetBatchUpdate(True)`
— the per-sample default zeroes `Neurons[].Delta` between samples. We use the
batch idiom: `ClearDeltas` at batch start, accumulate over the minibatch,
`UpdateWeights`, repeat (mirrors `examples/EarlyExitNetwork` and
`examples/GradientNoiseScale`).

## The headline payoff: test-time concept intervention

At inference we `Compute` once, then **overwrite the predicted concept vector**
at the sigmoid bottleneck and recompute **only** the downstream label head — the
same `CopyNoChecks`-then-recompute machinery as `examples/ActivationSteering` /
`examples/ActivationPatching`:

```pascal
NN.Compute(Input);                                  // clean forward
NN.Layers[ConceptIdx].Output.CopyNoChecks(Inject);  // overwrite the bottleneck
for i := ConceptIdx+1 to LastLayer do
  NN.Layers[i].Compute();                            // recompute downstream only
```

We show:

- **(a)** injecting the **ground-truth** concepts raises label accuracy over the
  end-to-end prediction (the model's residual mistakes are attributable to
  concept errors, not the label head);
- **(b)** flipping a **single** concept bit by hand deterministically flips the
  predicted class in the direction that concept controls.

## The two built-in invariants (asserted & printed)

1. **No-op overwrite is exact.** Intervening with the model's **own** predicted
   concepts (a no-op overwrite) reproduces the un-intervened logits
   **bit-for-bit** (`max|dlogit| = 0`).
2. **`lambda = 0` ⇒ the bottleneck drifts.** A second net trained with the
   concept-loss weight set to 0 has a bottleneck free to drift, so the concepts
   no longer align with the ground-truth bits — the *"leaky"* joint-vs-
   independent CBM failure mode. Mean concept accuracy collapses toward chance.

Both print `PASS`.

## Sample output

```
Per-concept accuracy (sigmoid bottleneck thresholded at 0.5):
  concept             test-acc
  c0 "is bright"        0.9900
  c1 "is top half"      0.9933
  c2 "is round"(decoy)  0.9933
  mean concept acc      0.9922

Label accuracy:
  clean (end-to-end)                 0.9833
  intervened (inject TRUE concepts)  1.0000
  -> intervention gain               0.0167

INVARIANT 1 (no-op overwrite with OWN concepts reproduces logits bit-for-bit): max|dlogit|=0.00E+000  -> PASS

Worked single-concept flip (set concept c1 "is top half" := 1):
  baseline concepts [1 0 1] -> predicted class 1
  flipped  concepts [1 1 1] -> predicted class 3
  delta = 2  (c1 carries weight +2 in label=c0+2*c1; flipping it moves the class deterministically)

  lambda=1 mean concept acc = 0.9922  (concepts ALIGN)
  lambda=0 mean concept acc = 0.4983  (concepts DRIFT - leaky CBM)
INVARIANT 2 (lambda=0 bottleneck drifts: alignment drops vs lambda=1): -> PASS

ALL CHECKS PASS
```

## Contrast with neighbouring examples

- **`examples/LinearProbeReport`** fits a **post-hoc, frozen** linear probe that
  only **reads** what some layer already encodes — it never changes the network
  or its predictions. Here the concept layer is **trained into** the forward
  path and is causally **editable**.
- **`examples/ActivationSteering`** **edits** raw hidden activations with a
  diff-of-means direction but has **no concept supervision** and no
  interpretable, named bottleneck — the edited coordinates have no a-priori
  meaning. Here every bottleneck unit **is** a named concept by construction, so
  the intervention is "set concept `c1` := true", not "add `1.7·v`".
- **`examples/DomainAdversarial`** uses gradient reversal to **remove**
  information (it makes a feature undecodable). Here we do the opposite: we
  **force** a specific, human-meaningful set of concepts to be decodable **and**
  to be the sole carrier of the label.

## Build & run

```
cd examples/ConceptBottleneck
lazbuild ConceptBottleneck.lpi
../../bin/x86_64-linux/bin/ConceptBottleneck
```

Pure CPU, no dataset download, deterministic (seeded `RandSeed := 424242`).
Total runtime ~10 s (it trains two nets) and a few MB of RAM.
