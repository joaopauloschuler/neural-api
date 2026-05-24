# MarginReport

Tiny example for `TNNet.TopLogitMarginReport`, the classifier
confidence-margin diagnostic.

The program builds a small MLP classifier (`2 -> 16 -> 16 -> 4` + softmax)
and trains it briefly on a self-contained synthetic dataset of four 2D
Gaussian clusters. Three clusters are well separated; the fourth is wider
and sits between two of the others, so the model is systematically less
certain about it.

After training it prints
`TNNet.TopLogitMarginReport(NN, ValSet, NumClasses)`, which runs one
forward pass per validation sample and computes the per-sample margin
`top1_logit - top2_logit` on the raw final-layer output. The report shows:

- overall mean / median / min / max margin and the total sample count
- a 10-bin ASCII histogram of the margin over `[min, max]`
- per-class mean and median margin, grouped by the sample's **true**
  class (the wide class 3 stands out with a smaller margin)
- the lowest-margin sample indices per class (a ready-made "hard
  examples" pool for active learning, label-noise auditing or curriculum
  work)

A large margin means the model is confidently separating the top class
from the runner-up; a margin near zero means the top two logits are
nearly tied — exactly the samples worth re-checking.

## Build & run

```
cd examples/MarginReport
lazbuild MarginReport.lpi
../../bin/x86_64-linux/bin/MarginReport
```

Total runtime is well under a minute.
