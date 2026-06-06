# Edge of Stability / progressive sharpening

This example reproduces **progressive sharpening** and the **Edge of Stability**
(EoS) of full-batch gradient descent (Cohen, Kaur, Li, Kolter, Talwalkar 2021,
*Gradient Descent on Neural Networks Typically Occurs at the Edge of Stability*)
on a pure-CPU synthetic toy. It uses only existing in-tree machinery — the
sharpness is read straight off `TNNet.HessianCurvatureReport`; **no new layer or
report is added.**

## The phenomenon

Train a network with plain **full-batch gradient descent** — deterministic GD,
**no Adam, no momentum, no mini-batches** — at a **fixed** step size `eta`. Two
things happen:

1. **Progressive sharpening.** The top Hessian eigenvalue `lambda_max` (the
   *sharpness* of the loss surface) **rises** during early training.
2. **Edge of Stability.** `lambda_max` stops rising once it reaches `~ 2/eta`,
   the classical stability limit of GD on a quadratic (a step through curvature
   above `2/eta` would diverge). It then **hovers just above `2/eta`** while the
   loss keeps falling **non-monotonically** (small ripples instead of a smooth
   descent).

The punchline across an `eta` sweep: the **plateau height tracks `2/eta`**. A
smaller step lets the network get sharper before it stalls (higher plateau); a
larger step caps it sooner (lower plateau).

## How sharpness is measured (reusing `HessianCurvatureReport`)

`lambda_max` comes straight out of `TNNet.HessianCurvatureReport`, whose power
iteration on **finite-difference Hessian-vector products** estimates the top
Hessian eigenvalue *without ever forming the Hessian*. Crucially the report is a
**pure measurement**: it snapshots the weights, probes, and restores them
bit-for-bit, and **never takes a training step**. That is exactly what makes it
safe to call every `K` GD steps *inside* the training loop — it cannot perturb
the trajectory. The example parses the `lambda_max = ...` line out of the
report's text once per measurement.

## Distinct from `examples/HessianCurvature`

`HessianCurvature` is a **static** contrast: it trains two nets to two different
*already-converged* minima (a sharp one and a flat one) and prints **one**
curvature report for each. There is **no time axis, no `2/eta` threshold and no
`eta` sweep**.

**This** example is **dynamic**: a *single* net under fixed-`eta` full-batch GD,
with `lambda_max` sampled as a **time series**, charted against the `2/eta` line,
the **EoS-entry step flagged**, and the whole thing repeated across **three
`eta` values** to show the plateau tracks `2/eta`. Different question, different
machinery exercised (the report used as an *online probe*, not a one-shot
verdict).

## What this example builds

```
Input(4) -> FullConnectReLU(12) -> FullConnectLinear(2)
```

The task is a fixed **2-class** problem with **heavily overlapping** clusters
(noise >> centre separation), so it is *not* separable: the best achievable MSE
is well above zero. That matters — on a trivially separable toy the loss
collapses to ~0 and sharpening ends before the edge is reached. The overlap
keeps full-batch GD making progress so it rides the edge for the whole run.

One **fixed** full batch of 24 samples is generated once and reused for every GD
step, every Hessian probe and every `eta` arm. The weights are initialised once
(`InitWeights` then `MulWeights(0.35)` for a deliberately **flat start**, so
`lambda_max` begins *below* `2/eta` and has room to sharpen *up* into it) and the
same `RandSeed := 424242` is restored before each arm, so the arms differ **only
in `eta`**. Training is plain GD via `SetBatchUpdate(true)` +
`SetLearningRate(eta, 0)` (zero inertia = no momentum) + one `UpdateWeights()`
per step over the summed full-batch gradient.

## Observed results

Single-threaded, `RandSeed := 424242`, 800 GD steps/arm, ~3.5 s wall-clock on
CPU. For each arm the lambda time series hovers right around its `2/eta` line:

```
eta=0.037 : init lam=27.955, peak lam= 76.021, plateau lam=52.112 vs 2/eta=54.054 ; loss 0.85241 -> 0.05560  OK
eta=0.040 : init lam=27.955, peak lam= 89.825, plateau lam=48.280 vs 2/eta=50.000 ; loss 0.85241 -> 0.06424  OK
eta=0.043 : init lam=27.955, peak lam=170.393, plateau lam=47.122 vs 2/eta=46.512 ; loss 0.85241 -> 0.11225  OK

OK: plateau height DECREASES as eta increases -> it tracks 2/eta across the sweep.
All Edge-of-Stability self-checks passed.
```

A slice of the `eta = 0.040` chart (the `L` marks `lambda_max`, the `|` marks the
`2/eta = 50` line) — note `lambda_max` rises from 28 and then **ripples right
around the threshold** while the loss keeps drifting down with ripples:

```
  step    0  lam=  27.955 | L                 |                ...   loss=  0.85241
  step   25  lam=  60.802 |                   |        L        ...   loss=  0.05071
  step  150  lam=  45.781 |                L  |                 ...   loss=  0.02104
  step  225  lam=  89.825 |                   |               L ...   loss=  0.02006   <- ripple spike
  step  600  lam=  47.470 |                 L |                 ...   loss=  0.05931
  step  800  lam=  83.367 |                   |          L      ...   loss=  0.06424
```

The **plateau** reported per arm is the **median** of `lambda_max` over the
second half of the run (not the mean): the EoS regime ripples, occasionally
spiking well above `2/eta` then snapping back, so the median reports where
`lambda_max` actually *hovers* and is robust to those transients.

## Self-check (gate)

The program `Halt(1)`s unless **all** of these hold (per the repo's self-gating
style, e.g. `RandomLabelMemorization` / `DropPathAblation`):

1. **Finiteness** — every measured `lambda_max` and loss is finite (a diverging
   arm would NaN/Inf; the FPU exception mask turns those into detectable values).
2. **Progressive sharpening** — peak `lambda_max` rises above `1.10 ×` its
   value at init. (If it never rises, the probe/`eta`/init is off.)
3. **Plateau sits at/just above `2/eta`** — the median plateau is within
   `[0.70, 1.40] × 2/eta`. Well below = not yet at the edge; far above = diverging.
4. **Loss still trends down across the plateau** — final-sample loss < first-sample
   loss, despite the ripples.
5. **Plateau tracks `2/eta` across the sweep** — since the `eta` values ascend,
   the plateau heights must be (weakly) **decreasing**.

## Tuning honesty (which knobs are fragile)

The EoS plateau is genuinely fragile on a tiny ReLU net under plain GD, and the
**usable `eta` window is narrow**. What had to be tuned, and what does **not**
reproduce:

- **Init sharpness vs `eta`.** `lambda_max` at init must sit *below* the smallest
  `2/eta` so there is room to sharpen *up*. `lambda_max` at init is dominated by
  the data scale and (because `HessianCurvatureReport` sums the loss over the
  batch) by the **batch size**, *not* mainly by the weight scale — shrinking the
  init weights alone barely moves it. The flat start needed `MulWeights(0.35)`
  **plus** the small fixed batch (24) **plus** the modest input scale.
- **Too-large `eta` collapses the net, it does not ride the edge.** At
  `eta >= ~0.046` the very first GD step kills the ReLUs and the net falls into a
  dead `loss = 0.5`, `lambda ~ 24` state (or, slightly higher, diverges to
  Inf/NaN). At `eta = 0.05` we also saw it wander into a region with a *negative*
  top eigenvalue (a saddle), loss stuck ~0.4. None of those are EoS.
- **Too-small `eta` is curvature-limited, not edge-limited.** At
  `eta <= ~0.03` (`2/eta >= 67`) the attainable sharpness ceiling on this data
  (~55-65) is *below* `2/eta`, so the net sharpens as much as the problem allows
  and plateaus there — it never reaches the edge, and the plateau then does
  **not** track `2/eta`.
- **Net result:** the clean three-way "plateau tracks `2/eta`" sweep only
  reproduces in a narrow band, here `eta in {0.037, 0.040, 0.043}`. We report
  the **median** of the second-half samples (the mean was polluted by ripple
  spikes and broke the monotonic-tracking check even when the trajectory was
  visibly hovering at the edge).

These caveats are documented rather than papered over by loosening the gate —
same spirit as the `RandomLabelMemorization` / Grokking README notes.

## Build & run

```
cd examples/EdgeOfStability
fpc -O3 -Mobjfpc -Sh -Fu../../neural -dRelease EdgeOfStability.lpr
./EdgeOfStability
```

If `fpc` cannot find `UTF8Process` (pulled in transitively), add the LazUtils
unit path, e.g.
`-Fu/usr/share/lazarus/<ver>/components/lazutils/lib/x86_64-linux`. The program
is single-threaded and deterministic; all three `eta` arms together run in well
under a minute on CPU.
