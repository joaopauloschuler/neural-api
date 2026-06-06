# LossLandscapeProbe

Demonstrates `TNNet.LossLandscapeProbe`: a forward-only, filter-normalised
1D probe of the loss surface around a trained network's weights.

## What it does

1. Builds a tiny 2-hidden-layer MLP softmax classifier for a 2D
   two-Gaussians toy.
2. Trains briefly (5 epochs), measures accuracy, then runs the probe with
   `K=21, R=1.0` (cross-entropy loss).
3. Repeats with 40 epochs of training so the two curves can be compared
   directly.

The probe samples loss along a random direction `d` in weight space, where
`d` is drawn N(0,1) per neuron and then *filter-normalised* so
`||d_neuron|| = ||W_neuron||` (Li et al., 2018). This is the standard fix
that neutralises ReLU scale-invariance and turns the curve into a property
of the landscape rather than of weight magnitude.

## Output

For each checkpoint, the report prints:

- the K alpha/loss pairs as a table,
- a one-line ASCII curve over `[-R, +R]` with `*` at the centre, `!` at
  the minimum and `#` at the other samples,
- per-alpha relative loss bars,
- the central-difference sharpness scalar
  `(L(+h) - 2*L(0) + L(-h)) / h^2`,
- the loss-doubling radius — the smallest `|alpha|` with `L(alpha) > 2*L(0)`
  (or `">R"` if it never doubles inside `[-R, +R]`).

The original weights are restored bit-for-bit at the end of the probe
(try/finally protected).

## Running

```
cd examples/LossLandscapeProbe
lazbuild LossLandscapeProbe.lpi
../../bin/<arch>/bin/LossLandscapeProbe
```

Total runtime is under a minute on CPU.

## Expected reading

A flatter minimum should produce a smaller sharpness scalar and a larger
(or `">R"`) loss-doubling radius. On the bundled two-Gaussians toy the
40-epoch run is consistently lower-sharpness than the 5-epoch run,
matching the "flat-minima generalise better" folklore.
