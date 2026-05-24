# Fourier Features and Spectral Bias

The headline Tancik et al. 2020 "spectral bias" micro-experiment for the
`TNNetFourierFeatures` layer (a fixed random Fourier-feature coordinate
embedding, Rahimi & Recht 2007 / Tancik et al. 2020,
<https://arxiv.org/abs/2006.10739>).

A small ReLU coordinate-MLP is asked to fit a high-frequency 1D target

```
  y = sin(20*x) + 0.5*sin(53*x),   x in [-1, 1]
```

TWICE, with the **identical** hidden head:

```
  raw     : Input(1) ->                            FullConnectReLU(64) ->
            FullConnectReLU(64) -> FullConnectLinear(1)
  fourier : Input(1) -> FourierFeatures(M=64, sigma) -> FullConnectReLU(64) ->
            FullConnectReLU(64) -> FullConnectLinear(1)
```

The Fourier front-end lifts the scalar `x` into a `2*M`-dimensional cos/sin
basis (`cos(2*pi*B^T x)` concatenated with `sin(2*pi*B^T x)`, where `B` is a
fixed random Gaussian frequency matrix of std-dev `sigma`). That lift is the
only difference between the two networks.

## The headline result

Plain coordinate MLPs suffer from **spectral bias**: they learn low
frequencies fast and high frequencies barely at all, so the raw model cannot
fit the `sin(53*x)` component. The Fourier mapping injects the high
frequencies directly, so the same head fits the target almost exactly.

```
=== HEADLINE: final clean-grid MSE ===
  raw-coordinate MLP   MSE = 0.389365
  fourier-feature MLP  MSE = 0.000567
  improvement factor (raw / fourier) = 687.27x
```

A side-by-side predicted-vs-truth table at 11 points makes the failure
obvious: the raw MLP flattens toward a low-frequency average while the
Fourier MLP tracks every wiggle.

## The sigma bandwidth sweep — the single-knob story

`sigma` sets the spread of the sampled frequencies and is the one knob that
matters. Too small and the mapping is still low-pass (underfit); too large
and the features are so high-frequency they fit noise (overfit). The sweep
prints final MSE vs `sigma in {0.5, 2, 8, 32}`:

```
=== SIGMA BANDWIDTH SWEEP (Fourier front-end, M=64) ===
  too small = low-pass / underfit;  too large = noisy / overfit
     sigma        final MSE
       0.5       0.001540
       2.0       0.000143
       8.0       0.008059
      32.0       0.356886
```

A clear U-shape: `sigma = 2` is the sweet spot here, `0.5` underfits and `32`
is back up near the raw-MLP error.

## Training loop

Like `SineRegression`, this example does NOT use `TNeuralFit`. It calls
`NN.Compute`, `NN.Backpropagate`, `NN.UpdateWeights` and `NN.ClearDeltas`
directly for hand-rolled shuffled mini-batch SGD:

- 384 training pairs (generated once at startup, shared by every model)
- batch size 16
- 400 epochs of plain SGD with momentum 0.9
- learning rate 0.01, L2 decay disabled
- MSE is measured on a clean 200-point test grid over `[-1, 1]`

Pure CPU, no dataset download, finishes in under two minutes.

## Build and run

```
cd examples/FourierFeaturesSpectralBias
lazbuild FourierFeaturesSpectralBias.lpi
../../bin/x86_64-linux/bin/FourierFeaturesSpectralBias
```
