# EffectiveReceptiveField

Demonstrates `TNNet.EffectiveReceptiveFieldReport`: the **empirical**
(gradient-measured) counterpart of the analytical
[`TNNet.ReceptiveFieldReport`](../ReceptiveFieldReport). The analytical report
asks *"what input region COULD a deep output unit see?"*; this one asks *"what
input region does that unit ACTUALLY WEIGHT?"*. Luo et al. 2016 show the
**effective** receptive field is typically far smaller — and more Gaussian —
than the theoretical one: a deep stem often only really uses the middle of its
theoretical window.

## What it does

It picks the **centre output unit** of the final spatial layer, enables
input-gradient flow with `TNNet.EnableInputGradient` (the same helper the
saliency and adversarial reports use), and for every probe input runs one
`NN.Compute`, injects a one-hot output error `e_centre`, back-propagates, and
reads `d out_centre / d input` off the input layer. It accumulates `|gradient|`
(summed over input depth) into a per-`(x,y)` input-plane heatmap. The net is
**frozen** — `ClearDeltas` before each pass, never `UpdateWeights`.

The example builds two **untrained** nets on the same 15x15x1 input, both with
a **theoretical RF of 9x9**:

1. A **stack of four 3x3 stride-1 convs** (RF grows by 2 per conv: `1 + 4*2 =
   9`). Many composed 3x3 kernels make the effective RF concentrate sharply
   near the centre, so the effective/theoretical ratio is **well below 1**.
2. A **single 9x9 conv** (RF = 9 directly). One flat kernel weights its whole
   window much more evenly, so the effective RF **fills** the theoretical
   window (ratio ≈ 1).

No training and no dataset download: a small deterministic synthetic probe
batch drives the centre-unit gradient. Runs in well under a second.

## What it reports

- a **2-D ASCII heatmap** of the input-plane sensitivity (per-`(x,y)`, summed
  over depth, shaded `' .:-=+*#%@'` by its own max);
- the **effective RF**: the smallest centred square holding 90% of the
  gradient mass (per-axis half-widths and diameters), the mass centroid, and
  the mass-weighted spatial std `sigma_x`, `sigma_y`;
- the **theoretical RF side-by-side**: it calls `TNNet.ReceptiveFieldReport`
  internally for the analytical final-RF number and prints the
  effective/theoretical ratio per axis.

## Output (abridged)

```
=== (i) Stack of four 3x3 stride-1 convs (15x15x1 input) ===
...
       ......
     ...:.:::.
     ..:--=-:.
     .:-==+=-.
     .:=-@+=-.
     .:-+=+=:.
     .:---*=-.
     ...::::.
...
Mass-weighted spatial std: sigma_x= 2.001  sigma_y= 2.012.
EFFECTIVE RF (90% of gradient mass): ... per-axis half-width x=3 y=3 ...
Theoretical RF (analytical, ReceptiveFieldReport): 9 x 9 on 15x15 input.
Effective / theoretical RF: x =   7.0 / 9 =  0.78   y =   7.0 / 9 =  0.78.

=== (ii) Single 9x9 conv (15x15x1 input) ===
...
Effective / theoretical RF: x =   9.0 / 9 =  1.00   y =   9.0 / 9 =  1.00.
```

The 3x3 stack's effective RF is a tight central blob (ratio ≈ 0.78), while the
single large kernel weights its full window (ratio ≈ 1.0) — the
effective-vs-theoretical gap made visible in one run.

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
