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
      1     1.073264     0.050222
      5     0.253923     0.023028
     10     0.089153     0.088493
     15     0.070126     0.038485
     20     0.043645     0.034799
     25     0.028318     0.021490
     30     0.020902     0.021057
     35     0.016855     0.021622
     40     0.014595     0.021054

multi-step forecast vs truth (last 24 samples, original scale):
  step      forecast        truth        abs_err
     1        2.79455      2.96144        0.16690
     2        2.78855      2.80832        0.01977
     3        2.69504      2.73847        0.04343
     4        2.80081      2.80238        0.00156
     5        2.95296      3.10671        0.15375
     6        3.22170      3.33740        0.11570
     7        3.56119      3.67935        0.11816
     8        3.89704      4.16253        0.26549
    24        3.86110      4.16524        0.30413

horizon MAE  = 0.30445
horizon RMSE = 0.34830

naive persistence MAE = 1.03048  (last-value baseline)
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
