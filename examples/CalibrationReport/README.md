# CalibrationReport example

This example demonstrates the forward-only model-calibration / reliability
unit `neuralcalibration`, which measures **how well a classifier's
confidence matches its accuracy**. It runs a single forward pass over a
labeled validation set and reports:

1. **Expected Calibration Error (ECE)** — the sample-weighted average over
   confidence bins of `|accuracy(bin) - confidence(bin)|` (0 == perfect);
2. **Maximum Calibration Error (MCE)** — the worst bin gap;
3. **Brier score** — mean squared error between the predicted probability
   vector and the one-hot label (a proper scoring rule);
4. a **reliability diagram** — per-bin `(mean confidence, accuracy, count)`
   rendered as an ASCII chart (accuracy `#` bars vs confidence `-` bars) and
   also written out as a P2 (ASCII) **PGM image** (`reliability.pgm`), with
   per-bin accuracy bars plotted against the `y = x` perfect-calibration line.

It then fits a single **temperature-scaling** scalar `T`
(`FitTemperature`, Guo et al. 2017) by a 1-D grid scan over validation NLL
and prints the report **before and after** scaling so the ECE drop is
visible.

The example trains a small MLP (`2 -> 16 -> 16 -> 4 + SoftMax`) on a
synthetic 4-class dataset of overlapping 2D Gaussian clusters with a
deliberately aggressive learning rate, which makes the raw model
over-confident (large ECE). Temperature scaling softens the over-confident
probabilities and shrinks the ECE. Pure-CPU, runs in a few seconds.

## Logit / temperature-scaling assumption

Calibration and temperature scaling need pre-softmax **logits**. A net that
ends in `TNNetSoftMax` only exposes probabilities, so the unit reconstructs
pseudo-logits as the elementwise log of the probability vector
(`z_i := ln(max(p_i, eps))`). Up to an additive constant (which softmax
ignores) this recovers the true logits of a plain softmax-over-linear head,
so `softmax(z / T)` is exactly temperature scaling on the original logits.
The "after" report is produced by feeding the trained net's softmax output
through a tiny stateless wrapper net `Log -> *(1/T) -> SoftMax`. The trained
backbone is never re-trained, never back-propagated through and its weights
are never mutated.

## Build and run

```
lazbuild CalibrationReport.lpi
./CalibrationReport
```
