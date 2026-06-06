# Sine Regression

The smallest possible "does the library still train?" demo.

A two-layer MLP learns `y = sin(pi*x) + small Gaussian noise` on 256
random samples drawn uniformly from `x in [-1, 1]`. Designed to finish
well under a minute on a single CPU and to serve as a quick smoke test
for any change to the core training path.

## Architecture

```
  TNNetInput.Create(1)              // scalar x
  TNNetFullConnectReLU.Create(64)   // single hidden layer
  TNNetFullConnectLinear.Create(1)  // scalar y
```

The output layer is linear because `sin` ranges over `[-1, 1]` and we
do not want a non-linearity squashing the target.

## Training loop

This example deliberately does NOT use `TNeuralFit`. It calls
`NN.Compute`, `NN.Backpropagate`, `NN.UpdateWeights` and
`NN.ClearDeltas` directly so that the example doubles as a tiny
reference for hand-rolled mini-batch SGD.

- 256 training pairs (regenerated once at startup)
- batch size 16
- 300 epochs of plain SGD with momentum 0.9
- learning rate 0.002
- L2 decay disabled

Mean-squared error on a clean (no-noise) 100-point test grid is printed
every 25 epochs. After training, a side-by-side table compares
predictions to ground truth at 11 evenly-spaced points in `[-1, 1]`.

## Build and run

```
cd examples/SineRegression
lazbuild SineRegression.lpi
../../bin/x86_64-linux/bin/SineRegression
```
