# ActivationStatsReport

Demonstrates `TNNet.ActivationStatsReport`: a **forward-only** per-layer
activation-distribution diagnostic. Given a probe batch it runs one
`NN.Compute` per sample, walks every layer's `Output` volume and prints a
per-layer statistics table plus a flag list and a network-wide std histogram.

## What it does

1. Builds a small MLP (`Linear -> ReLU -> Linear -> tanh -> Linear`) on an
   8-dim input.
2. Builds a tiny probe batch (48 random `TNNetVolume` samples).
3. Prints `TNNet.ActivationStatsReport` on the **fresh-init** network.
4. Trains briefly (40 epochs of batch 32) on the trivial synthetic task
   `y = ||x||` and prints the report again, so you can eyeball how training
   reshapes the activation distribution.

No GPU, runs in well under a minute.

## What the report shows

Per layer (one table row + a one-line histogram):

```
Idx  Class                      OutShape           mean       std       min       max     |med|  |skew|     kurt   sat-   sat+   neg%    ~0%
------------------------------------------------------------------------------------------------------------------------------------------------------
0    TNNetInput                 (8,1,1)          ...
     hist[-x..+x] .._oO#Oo_.....
1    TNNetFullConnectLinear     (32,1,1)         ...
...
```

- **mean / std / min / max** — the headline distribution summary.
- **|median|** — magnitude of the median, approximated from the last probe
  sample to keep memory bounded.
- **|skew| / kurt** — standard sample skewness magnitude and *excess* kurtosis
  (a normal distribution has kurt = 0).
- **sat- / sat+** — fraction of activations below `-SaturationThreshold` /
  above `+SaturationThreshold`.
- **neg%** — fraction of negative activations.
- **~0%** — fraction with `|x| < 1e-6`.
- **hist** — compact 16-bin ASCII histogram over `[-MaxAbs, +MaxAbs]`
  (per-layer scaling; `.` empty, then `_ o O #` for increasing density).

Closing summary:

- A 10-bin ASCII histogram of **per-layer std** across the whole network, so
  vanishing (all std near 0) or exploding (one huge std) activation patterns
  jump out at a glance.
- A **flag list**: "near-collapsed" layers (`std < 1e-4`) and "saturating"
  layers (`>50%` saturated on either side).

## Saturation threshold

The spec distinguishes bounded activations (saturate near `±0.99·OutputRange`)
from unbounded ones (`|x| > 6`). This implementation simplifies that to a
single configurable `SaturationThreshold` parameter (default `6.0`, the
unbounded case). For bounded layers (e.g. tanh in `[-1, 1]`) the default
threshold of 6 never trips, which is the intended conservative behaviour —
pass a smaller threshold (e.g. `0.99`) if you want to flag bounded layers that
crowd their limits.

## Running

```
cd examples/ActivationStatsReport
lazbuild ActivationStatsReport.lpi
../../bin/<arch>/bin/ActivationStatsReport
```

Or directly with fpc:

```
cd examples/ActivationStatsReport
fpc -B -Fu../../neural -Mobjfpc -Sh -O2 ActivationStatsReport.lpr
./ActivationStatsReport
```
