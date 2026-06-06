# TestTimeAugmentation

Tiny example for `TNNet.TTAReport`, the forward-only test-time augmentation
(TTA) evaluator.

The program builds a small synthetic 3-class image task (8x8x3 colored
patterns):

- **class 0**: a bright red vertical stripe on the left columns,
- **class 1**: a bright green horizontal stripe on the top rows,
- **class 2**: a bright blue checkerboard,

each plus light per-pixel noise. It trains a small conv classifier
(`Input -> ConvReLU(8,3) -> MaxPool(2) -> FCReLU(16) -> FCLinear(3) -> SoftMax`)
for a few dozen epochs, builds a held-out probe batch with integer labels, then
prints `TNNet.TTAReport(NN, Probes, Labels)`.

The report runs only forward passes over a fixed transform menu — identity
(baseline), `TNNetFlipX`, `TNNetFlipY`, `TNNetReverseChannels` and
`TNNetRoll(+1)` — each produced by a tiny `Input -> Transform` wrapper net, and
reports:

- baseline top-1 accuracy on the untransformed inputs;
- per-transform top-1 accuracy (each transform applied alone — a
  near-invariance check);
- full-ensemble TTA top-1 accuracy = argmax of the **averaged** outputs across
  all transforms, plus the signed delta vs baseline;
- per-class accuracy delta (baseline -> ensemble) so classes that lose under
  TTA are visible;
- the per-sample agreement rate `mean(argmax(avg) == argmax(baseline))`;
- a one-line verdict `TTA helps` / `TTA neutral` / `TTA hurts` from a
  configurable threshold on the accuracy delta.

The example prints the report twice: once averaging **raw logits** (the
default, arithmetic mean) and once averaging **post-softmax probabilities**
(soft voting — the linear-vs-geometric-mean question), so the two averaging
spaces can be compared side by side.

The patterns are deliberately not flip/channel symmetric, so each transform
genuinely perturbs the input and the per-transform rows are informative. The
report is pure forward-only — the trained weights are never touched and no
backward pass is run.

## Build & run

```
cd examples/TestTimeAugmentation
lazbuild TestTimeAugmentation.lpi
../../bin/x86_64-linux/bin/TestTimeAugmentation
```

Total runtime is well under a minute.
