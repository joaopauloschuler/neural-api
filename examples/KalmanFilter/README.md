# KalmanFilter — differentiable diagonal Kalman filtering

Tracks a 1-D constant-velocity signal corrupted by additive Gaussian
measurement noise and recovers the latent track with a
`TNNetKalmanFilterCell` — a layer that propagates
**uncertainty** (a per-channel state covariance `P`) rather than just a
deterministic state.

## What it does

The cell sweeps the time axis running the classic two-phase Kalman recurrence
per channel (here a single channel), with `x_0 = 0`, `P_0 = 1`:

```
PREDICT: xm_t = a·x_{t-1}            Pm_t = a²·P_{t-1} + Q
UPDATE:  g_t  = Pm_t / (Pm_t + R)    x_t  = xm_t + g_t·(z_t − xm_t)
                                     P_t  = (1 − g_t)·Pm_t
```

The transition `a = tanh(a_raw)` is bounded to `(−1,1)` for stability and the
noises `Q = softplus(Q_raw)`, `R = softplus(R_raw)` are kept positive, so the
Kalman gain `g_t ∈ (0,1)` by construction. All three per-channel scalars are
learned end-to-end via full BPTT (the covariance carries its own adjoint scan
alongside the mean adjoint).

The observation `z_t` is the noisy signal; the supervised target is the clean
signal, so training drives `Q`, `R` (hence the gain) to the values that best
denoise this track.

## Contrast arm

A parameter-matched `TNNetDiagonalSSM` (a linear time-invariant diagonal
state-space cell, 4 scalars vs the Kalman cell's 3) is trained on the same task.
The SSM has **no covariance** and therefore applies a fixed linear smoothing
kernel; the Kalman cell forms an uncertainty-aware blend `g = P/(P+R)`.

## Running

```
lazbuild examples/KalmanFilter/KalmanFilter.lpi
./bin/x86_64-linux/bin/KalmanFilter
```

Runs in well under a minute on 2 CPU cores; tiny memory footprint.

## Headline result (seed 424242, noise std 0.45)

```
MSE(noisy   , clean) = 0.21877   (the observation)
MSE(Kalman  , clean) = 0.04632   reduction 78.8%
MSE(DiagSSM , clean) = 0.04572   reduction 79.1%

Learned Kalman params (channel 0):
  a = tanh(a_raw)     = 0.9447
  Q = softplus(Q_raw) = 0.1641   (process noise)
  R = softplus(R_raw) = 1.2795   (measurement noise)
```

Both sequence mixers denoise to essentially the same MSE on this 1-channel
track (the DiagonalSSM, given a smaller learning rate, is a strong linear
smoother here too). The Kalman cell matches it with **one fewer parameter** and
additionally exposes interpretable learned `a`, `Q`, `R` and the implied
steady-state gain `g = P/(P+R)` — a principled, uncertainty-aware blend the LTI
SSM cannot form. (Exact numbers print at runtime.)
