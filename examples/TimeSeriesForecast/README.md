# TimeSeriesForecast — causal-conv univariate forecaster

A one-screen forecasting demo on a **synthetic** seasonal + trend + noise time
series (generated in-code; no external data files). A small causal 1-D
convolutional stack predicts the next sample from a sliding window, and is then
rolled out **auto-regressively** to produce a multi-step forecast over the
held-out tail.

## The signal

```
x[t] = 0.004·t                          (linear trend)
     + 1.0·sin(2π·t/40)                 (slow seasonality)
     + 0.5·sin(2π·t/13 + 0.7)           (fast seasonality)
     + N(0, 0.05²)                      (observation noise)
```

900 samples; the first 700 are training, the rest validation/test. The series is
standardised using **train-only** mean/std (no scale leakage).

## The model

```
Input (window=32, 1, 1)
  -> TNNetCausalConv1D(16, k=3)              + ReLU   (local patterns, no future leak)
  -> TNNetCausalConv1D(16, k=3, dilation=2)  + ReLU   (wider receptive field)
  -> TNNetPointwiseConvLinear(8)             + ReLU   (per-timestep channel mix)
  -> TNNetFullConnectReLU(16)
  -> TNNetFullConnectLinear(1)                        (one-step-ahead scalar)
```

`TNNetCausalConv1D` left-pads so output position `t` sees only inputs `≤ t`. The
window is regressed to the next sample; a `cHorizon`-step forecast is produced by
feeding each prediction back into the window.

All layers are **existing** library layers — nothing is added to the core unit.

## Build & run

```
lazbuild examples/TimeSeriesForecast/TimeSeriesForecast.lpi
./bin/x86_64-linux/bin/TimeSeriesForecast
```

Pure CPU, ~5,000 weights, finishes in a few seconds.

## Sample output

```
=== TimeSeriesForecast: causal-conv 1-step forecaster ===
series_len=900  window=32  train_end=700  horizon=24
signal = trend + sin(period 40) + sin(period 13) + noise(std 0.05)

model params = 5056

training (40 epochs)...
  epoch    train_MSE      val_MSE
      1     4.252160     1.011888
      5     0.037562     0.032070
     10     0.012267     0.022812
     15     0.006361     0.019763
     20     0.005146     0.017180
     25     0.004672     0.015352
     30     0.004329     0.014236
     35     0.004017     0.013713
     40     0.003805     0.013150

multi-step forecast vs truth (last 24 samples, original scale):
  step      forecast        truth        abs_err
     1        2.75907      2.94180        0.18273
     2        2.72192      2.80669        0.08476
     3        2.71430      2.78999        0.07568
     4        2.77748      2.82497        0.04749
     5        2.97577      3.09140        0.11564
     6        3.30525      3.34455        0.03930
     7        3.66878      3.69828        0.02949
     8        4.00940      4.15035        0.14095
    24        3.85145      4.19087        0.33942

horizon MAE  = 0.22183
horizon RMSE = 0.27106

naive persistence MAE = 1.00364  (last-value baseline)
```

## How to read it

- The **train/val MSE** columns are a short loss curve (standardised units);
  both drop sharply and the gap stays small — the model fits the seasonal +
  trend structure without overfitting.
- The **forecast-vs-truth** block compares the 24-step auto-regressive rollout
  against the true held-out tail in the original signal scale, with per-step
  absolute error. Error grows slowly with horizon (expected for auto-regressive
  rollout) but stays well-behaved.
- The **horizon MAE/RMSE** summarise forecast accuracy; the model beats the
  naive last-value persistence baseline by roughly 5× on MAE.
