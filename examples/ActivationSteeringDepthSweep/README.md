# ActivationSteeringDepthSweep

The **depth-sweep** follow-up to [`examples/ActivationSteering`](../ActivationSteering).
`ActivationSteering` injects a concept direction into **one** fixed hidden layer
(`k=2`) and shows the injection **causally** controls the softmax output (ActAdd /
activation addition / representation engineering, Turner et al. 2023). This
example asks the natural next question:

> **Where does a concept vector bite hardest?**

It sweeps the steering layer `k` across **all** steerable hidden layers (not just
one) and, per layer, reports how cleanly the diff-of-class-means direction
controls `P(target)` and how small an `alpha` is needed to flip the prediction.

The classifier is the same synthetic two-cluster task as `ActivationSteering`,
but with **four** steerable `FC+ReLU` hidden layers so a depth sweep is
meaningful:

```
Input(6) --> FC12+ReLU(k=1) --> FC12+ReLU(k=2) --> FC12+ReLU(k=3)
         --> FC12+ReLU(k=4) --> FC2 --> SoftMax
```

The class is decided *entirely* by `sign(x0)` (`x0 < 0` -> class 0, `x0 > 0` ->
class 1); the rest of the input coordinates are pure noise distractors. After a
short training run the weights are **frozen** — nothing below is a training step.

## What it does

For **each** steerable hidden layer `k = 1..4`:

1. **Steering vector.** Computes a diff-of-class-means direction over fresh
   training activations at that layer

   ```
   v_k = mean(act_k | class 1) - mean(act_k | class 0)
   ```

   No extra training: `v_k` is just a difference of two activation averages.

2. **Alpha sweep.** Snapshots the unsteered activation at `k` for a fixed probe
   input, then for `alpha in {-3,-2,-1,0,1,2,3}` runs a forward pass, overwrites
   layer `k`'s activation with `Output_k.MulAdd(alpha, v_k)`, recomputes layers
   `k+1..last`, and records `P(target class)` vs `alpha`. This reuses the exact
   recompute machinery the landed `TNNet.ActivationPatchingReport` /
   `ActivationSteering` drive: overwrite a cached activation
   (`CopyNoChecks`, same shape by construction), then call `FLayers[i].Compute()`
   for `i = k+1..last`.

## What it reports

Per layer `k`:

- the `P(target)`-vs-`alpha` **ASCII curve** (plus the predicted argmax per row);
- a **monotonicity** measure: the fraction of adjacent-`alpha` steps that
  increase (`1.0` = perfectly monotone up in `alpha`);
- the **alpha-to-flip**: the smallest `|alpha|` at which the predicted argmax
  leaves the plain-forward class (or `none in range` if it never flips).

Then a **summary** naming which layer `k` gave the cleanest monotone control and
which gave the smallest alpha-to-flip.

### Built-in correctness check

Carried over from `ActivationSteering` and applied at **every** layer `k`:
`alpha = 0` reproduces the unsteered forward pass **bit-for-bit** (`P(target)`
identical to the plain forward pass). The summary prints a single PASS/FAIL over
all layers.

## Forward-only

No backward pass and no weight steps run during the sweep. The **only** mutation
is the transient activation shift at layer `k`, reverted by the next clean
forward pass — the trained weights are never touched.

## Sample output

```
Probe input: plain forward P(class 1)=0.025631 (argmax=0). Steering TOWARD class 1.

==============================================================================
LAYER k=1 (TNNetFullConnectReLU, size=12, ||v_k||=2.3411): P(target) vs alpha
==============================================================================
  alpha   P(v)      argmax | P(v) bar (0..1)
 -3.0000  0.012018      0  | #
 ...
  3.0000  1.000000      1  | ################################################
  monotonicity(up-frac)=1.000   alpha-to-flip=  1.0000   alpha=0 bit-for-bit: PASS

==============================================================================
LAYER k=2 (TNNetFullConnectReLU, size=12, ||v_k||=3.0156): P(target) vs alpha
==============================================================================
  ...
  monotonicity(up-frac)=0.500   alpha-to-flip=  1.0000   alpha=0 bit-for-bit: PASS

... (k=3, k=4) ...

==============================================================================
SUMMARY
==============================================================================
Cleanest monotone P(target)-vs-alpha control: layer k=1 (up-fraction=1.000).
Smallest alpha-to-flip (concept bites hardest): layer k=1 (|alpha|=1.0000).
CHECK (alpha=0 reproduces plain forward BIT-FOR-BIT at EVERY layer k): PASS
```

Read it as: the same diff-of-means concept direction is a **causal knob at every
depth**, but it does **not** bite equally hard everywhere. Here every layer
flips the prediction by `alpha=1`, but layer `k=2`'s curve is **non-monotone**
(its diff-of-means direction pushes `P(target)` the *wrong* way for negative
`alpha`), while `k=1/3/4` give perfectly monotone control — sweeping `k` is how
you discover that. `alpha=0` is a bit-for-bit no-op at every `k`, and weights are
never stepped.

## Build & run

```
cd examples/ActivationSteeringDepthSweep
lazbuild ActivationSteeringDepthSweep.lpi
../../bin/x86_64-linux/bin/ActivationSteeringDepthSweep
```

Pure CPU, no dataset download, total runtime well under a second.
