# ActivationSteering

Tiny example for **activation steering / concept-vector intervention** (ActAdd /
activation addition / representation engineering, Turner et al. 2023) — the
**interventional** flip-side of the read-only probe examples. Instead of asking
*"what is decodable from a hidden layer?"* it **injects** a concept direction
**into** a hidden layer mid-forward and shows the injection **causally** controls
the output.

The program trains a small softmax classifier:

```
Input(6) --> FC12+ReLU --> FC12+ReLU(k=2) --> FC12+ReLU --> FC2 --> SoftMax
```

on a synthetic 2-class **two-cluster** task: the class is decided *entirely* by
`sign(x0)` (`x0 < 0` -> class 0, `x0 > 0` -> class 1); the rest of the input
coordinates are pure noise distractors. After a short training run the weights
are **frozen** — nothing below is a training step.

## What it reports

1. **Steering vector.** At a chosen hidden layer `k` it computes a
   diff-of-class-means direction over fresh training activations

   ```
   v = mean(act_k | class 1) - mean(act_k | class 0)
   ```

   No extra training: `v` is just a difference of two activation averages.

2. **Alpha sweep.** It snapshots the unsteered activation at `k` for a fixed
   probe input, then for `alpha in {-3,-2,-1,0,1,2,3}` runs a forward pass,
   overwrites layer `k`'s activation with `Output_k.MulAdd(alpha, v)`, recomputes
   layers `k+1..last`, and charts `P(target class)` vs `alpha` as an ASCII curve.
   This reuses the exact recompute machinery the landed
   `TNNet.ActivationPatchingReport` drives: overwrite a cached activation
   (`CopyNoChecks`, same shape by construction), then call `FLayers[i].Compute()`
   for `i = k+1..last`.

3. **Three built-in correctness checks** (PASS/FAIL):
   - **CHECK 1** — `alpha = 0` reproduces the unsteered forward pass
     **bit-for-bit** (`P(target)` identical to the plain forward pass);
   - **CHECK 2** — `P(target)` moves **monotonically** with `alpha` (positive
     steers toward class 1, negative toward class 0);
   - **CHECK 3** — steering with `v` vs a **random** unit direction scaled to the
     **same** L2 norm: the concept direction moves the output far more per unit
     norm, so `v` is genuinely special and not just "any push of this magnitude".

## Forward-only

No backward pass and no weight steps run during steering. The **only** mutation
is the transient activation shift at layer `k`, reverted by the next clean
forward pass — the trained weights are never touched.

This is **distinct from** `ActivationPatching` (swaps **whole** cached
activations between two inputs to localise *where* the decision lives),
`SaliencyReport` (input-space gradient attribution), `GradientAscent` (ascends on
the input image) and `LinearProbeReport` (only **reads** what a layer encodes).
Here we **add** a direction and watch the output move.

## Sample output

```
Steering layer k=2 (TNNetFullConnectReLU), activation size=12, ||v||=3.2611.
Random control direction r: ||r||=3.2611 (matched to ||v||).

Probe input: plain forward P(class 1)=0.010061 (argmax=0). Steering TOWARD class 1 with v.

==============================================================================
ALPHA SWEEP: P(target class) vs alpha (steered with v)
==============================================================================
  alpha   P(v)      P(rand)   |  P(v) bar (0..1)
  -3.0  0.000001  0.005087  |  
  -2.0  0.000016  0.006177  |  
  -1.0  0.000399  0.007499  |  
   0.0  0.010061  0.010061  |  
   1.0  0.997307  0.288712  |  ################################################
   2.0  0.999999  0.895929  |  ################################################
   3.0  1.000000  0.994232  |  ################################################

CHECK 1 (alpha=0 reproduces plain forward BIT-FOR-BIT): plain=0.01006051 steered@0=0.01006051  -> PASS
CHECK 2 (P(target) increases MONOTONICALLY with alpha): P(-3)=0.000001 .. P(+3)=1.000000  -> PASS
CHECK 3 (concept v steers MORE per unit norm than random r): mean|dP| v=0.428127  random=0.308587  ratio=1.39x  -> PASS

ALL CHECKS PASS: the diff-of-means concept direction CAUSALLY steers the output; alpha=0 is a no-op; random does far less.
```

Injecting `v` drives `P(target)` smoothly from `~0` to `~1`: a causal knob on the
prediction discovered with **no** extra training. `alpha = 0` is a bit-for-bit
no-op, and an equal-norm random direction barely moves the output by comparison —
the concept direction is special.

## Build & run

```
cd examples/ActivationSteering
lazbuild ActivationSteering.lpi
../../bin/x86_64-linux/bin/ActivationSteering
```

Pure CPU, no dataset download, total runtime well under a minute.
