# EffectiveReceptiveField

Sweeps a conv stem over several **kernel-size / stack-depth** configurations
and charts how the **empirical** (gradient-measured) effective receptive field
grows against the **theoretical** receptive field, using
`TNNet.EffectiveReceptiveFieldReport` — the empirical counterpart of the
analytical [`TNNet.ReceptiveFieldReport`](../ReceptiveFieldReport). The
analytical report asks *"what input region COULD a deep output unit see?"*;
this one asks *"what input region does that unit ACTUALLY WEIGHT?"*.

**Headline (Luo et al. 2016, "Understanding the Effective Receptive Field in
Deep Convolutional Neural Networks"):** the effective RF grows only
**sub-linearly** in the theoretical RF. Composing many small kernels
concentrates the centre-unit gradient into a tight, roughly Gaussian blob, so a
deep unit COULD see its whole theoretical window but only really WEIGHTS the
middle of it — and that gap **widens** as the stack gets deeper.

## What it does

For each stem configuration the program:

1. builds an **untrained**, single-channel conv stem on a small 21x21x1 grid;
2. runs `TNNet.EffectiveReceptiveFieldReport` over a deterministic 16-sample
   synthetic probe batch. The report picks the **centre output unit** of the
   final spatial layer, enables input-gradient flow with
   `TNNet.EnableInputGradient`, back-propagates a one-hot `e_centre` error per
   probe, and accumulates `|d out_centre / d input|` into a per-`(x,y)`
   input-plane heatmap (the net is **frozen** — `ClearDeltas` before each pass,
   never `UpdateWeights`);
3. writes the **`(radius, cumulative-mass-fraction)` CSV side-output** (the
   optional `CsvFile` argument) to `<tmp>/erf_sweep_<i>.csv` so the growth
   curve can be plotted outside the terminal;
4. extracts the effective-RF diameter and the theoretical RF from the report.

It then prints a compact table of `(config, theoretical_RF, effective_RF_diam,
eff/theo ratio)`.

The configs are sized so the **theoretical RF roughly doubles** down the table
(`theo_RF = 1 + N*(K-1)` for `N` stacked stride-1 convs of side `K`) while the
**effective diameter grows much more slowly** — the sub-linear story.

No training, no dataset download, tiny tensors: runs in well under a second on
2 CPU cores.

## Output

```
=== Effective receptive-field growth sweep (Luo et al. 2016) ===
Input grid 21x21x1, 16 synthetic probes, untrained frozen stems.

config          theo_RF    eff_RF_diam   eff/theo
--------------------------------------------------
1x 3x3                3              3       1.00
2x 3x3                5              5       1.00
4x 3x3                9              7       0.78
1x 9x9                9              9       1.00
4x 5x5               17             13       0.76
--------------------------------------------------
```

As the theoretical RF climbs from 3 to 17, the effective diameter lags further
and further behind: a **single** large kernel weights its whole window
(`eff/theo ≈ 1.0`), but **stacked** small kernels concentrate the gradient and
the ratio drops below 1 and keeps shrinking (`0.78` at 4×3×3, `0.76` at 4×5×5).
That is the effective RF growing **sub-linearly** in the theoretical receptive
field.

The per-config CSV (e.g. `erf_sweep_2.csv` for the 4×3×3 stem) holds the
cumulative gradient-mass curve, monotonically rising to 1.0:

```
radius,mass_fraction
0,0.02585155
1,0.21686888
2,0.46353251
3,0.78151649
4,1.00000000
```

## How it differs from the siblings

- **`ReceptiveFieldReport`** is purely analytical ("could see"); this measures
  what the unit actually weights ("does weight").
- **`SaliencyReport`** is per-sample input attribution for a *class logit*;
  this is a batch-averaged spatial-extent measurement of a *spatial output
  unit*.

## Running

```
cd examples/EffectiveReceptiveField
lazbuild EffectiveReceptiveField.lpi
../../bin/<arch>/bin/EffectiveReceptiveField
```

Or directly with fpc:

```
cd examples/EffectiveReceptiveField
fpc -B -Fu../../neural -Mobjfpc -Sh -O2 EffectiveReceptiveField.lpr
./EffectiveReceptiveField
```

Runs in under a second on CPU.
