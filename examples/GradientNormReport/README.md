# GradientNormReport example

Runs `TNNet.GradientNormReport` on a deep 12-layer ReLU MLP using a single
forward + backward pass over a tiny hypotenuse-like regression probe, then
repeats the run with a `TNNetLayerNorm` inserted at the midpoint of the
stack.

The report prints, per layer:

- `||dL/dx_in||` — L2 norm of the input-error tensor entering the layer.
- `||dL/dW||`    — L2 norm of the weight-gradient tensor across the layer.
- `ratio`        — per-step gradient-amplification factor versus the
  previous reported layer.

It also flags vanishing (`V`), exploding (`E`), and outside-ratio (`R`)
rows, and renders a 10-bin ASCII histogram of `log10(||dL/dx_in||)` across
the network.

Useful for spotting where a deep stack is collapsing or blowing up its
gradients without any training-time changes — pure CPU, no API additions.

## Build

```
lazbuild GradientNormReport.lpi
./GradientNormReport
```
