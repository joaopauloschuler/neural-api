# Decision-boundary report

This example demonstrates `TNNet.DecisionBoundaryReport`, a forward-only
introspection diagnostic that visualizes the **learned function of a 2-input
classifier head over its whole input domain**.

Given a trained net whose input layer is 2-D (`TNNetInput.Create(2)`) and a
classifier head with `NumClasses` outputs, the report sweeps a `Gx x Gy` grid
(default 41x41) over an axis-aligned bounding box of the `(x, y)` input plane
and runs **one forward pass per grid cell**, then renders:

* **(a) class map** — one glyph per grid cell = the `argmax` class at that
  point (`0..9`, `A..Z`), so the carved-up decision regions are directly
  visible;
* **(b) confidence overlay** — the same grid but each cell's glyph intensity
  (`.:-=+*#%@`, ~10 buckets) is set by the top-1 softmax probability (when the
  head sums to ~1) or by a normalised top1-top2 logit margin (for a linear
  head), so low-confidence boundary bands show up as faint seams;
* **(c) boundary length** — a single scalar = the number of grid cells whose
  4-neighbours disagree on `argmax`. This is a cheap proxy for how convoluted
  the learned boundary is: a near-linear separator gives a short boundary, an
  overfit wiggly boundary gives a long one;
* **(d) probe overlay** (optional) — when a probe set is supplied each sample
  is stamped onto the grid by its **true** class, so misclassified points stand
  out against the boundary;
* **(e) CSV side-output** (optional, `EmitCsv=True`) — a clearly-delimited
  `x,y,argmax,top1prob` block, one row per grid cell, for downstream plotting.

The bounding box is auto-fitted (with a small margin) from the probe set when
one is supplied, otherwise it uses the caller-supplied `(xMin,xMax,yMin,yMax)`,
falling back to `[-3,3]^2`. A **guard** returns a clear error message (never
crashes) when the input layer is not 2-D.

## What the example shows

All data is synthetic (no download); the whole run finishes in well under a
minute on CPU.

1. **Clean 3-cluster 2D Gaussian.** The class map is printed *before* training
   (weights shrunk toward zero so the softmax collapses to a near-constant
   single-class plane, boundary length 0) and *after* a short training run
   (clean separated regions, ~2% of the probe points misclassified).

2. **Tiny noisy two-moons, well-fit vs overfit.** A small `1x6` net and a
   deliberately-oversized `3x64` net are trained on the same 40 heavily-noised
   samples. The oversized net trained for 400 epochs memorises the noise: it
   prints wigglier class-map art *and* a larger boundary-length scalar than the
   small net — the overfitting pathology visible both as art and as a number.

## Building and running

```
lazbuild DecisionBoundary.lpi
../../bin/x86_64-linux/bin/DecisionBoundary
```

Pure forward-only — no backward pass, no training-time changes, no new layer
types; the inspected weights are never touched.
