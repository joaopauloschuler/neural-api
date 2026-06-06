# SaliencyReport

Tiny example for `TNNet.SaliencyReport`, the forward+backward input-attribution
(saliency) diagnostic.

Given a trained classifier and a single probe sample (a `TNNetVolume` input),
the report picks the predicted class `c = argmax(f(x))` and computes three
flavours of per-input-element attribution, printing each as a compact ASCII
heatmap over the input plane (one block per channel, one cell per pixel,
`~10` intensity buckets ``" .:-=+*#%@"``, brightest = highest `|attr|`):

- **(a) vanilla input-gradient saliency** `|d logit_c / d x|` — one forward
  pass, then one backward pass with the final-layer error set to the one-hot
  `e_c`, so the gradient that reaches the input layer is exactly
  `d logit_c / d x`;
- **(b) SmoothGrad** — (a) averaged over `N` noisy copies `x + eta`,
  `eta ~ N(0, sigma^2)`, `sigma = SmoothNoiseFrac * (max(x) - min(x))`
  (denoises the raw gradient);
- **(c) Integrated Gradients** — the straight-line path integral from a zero
  baseline to `x` in `K` steps:
  `IG_i ~= (x_i - 0) * (1/K) * sum_{k=1..K} d logit_c / d x_i` at `(k/K)*x`.

Per channel it also prints the total attribution mass for each variant and the
top-K most-attributing pixel coordinates of the vanilla map.

### Built-in correctness check: the IG completeness gap

Integrated Gradients satisfies the **completeness axiom**
`sum_i IG_i ~= logit_c(x) - logit_c(0)`. The report prints the
**completeness gap** `|sum(IG) - (logit_c(x) - logit_c(0))|` (absolute and
relative). A small relative gap means the path integration is faithful — this
is the one-number sanity/regression check for the report.

Pure CPU; reuses the existing forward/backward path. No optimiser step is
applied, so the inspected network's weights are left untouched.

## What this demo shows

On a synthetic `8x8x2` two-class image task (class 0 = bright `3x3` blob in the
top-left of channel 0; class 1 = bright blob in the bottom-right of channel 1,
plus background noise) it trains a small conv classifier
(`Conv -> MaxPool -> FC -> SoftMax`), then runs the report on:

1. a **clean, correctly-classified** class-0 probe — the three heatmaps should
   concentrate around the discriminative top-left/channel-0 blob, and the IG
   completeness gap is small;
2. a **noisier (harder)** class-1 probe — attribution spreads out more.

The raw input is printed as its own ASCII heatmap above each report so the
attribution maps can be eyeballed against the actual input content.

## Build & run

```
cd examples/SaliencyReport
lazbuild SaliencyReport.lpi
../../bin/x86_64-linux/bin/SaliencyReport
```

Pure CPU, no dataset download. Total runtime is well under a minute.
