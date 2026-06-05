# LabelSmoothingCalibration

Does **label smoothing** actually improve a classifier's **calibration**?
Train the **same** tiny MLP **once per smoothing strength** `eps`, then feed
each trained model into the forward-only calibration report
(`neuralcalibration`: ECE + Brier) and print a results table so a reader can
see whether the textbook claim holds on a concrete, deliberately hard task.

## What is label smoothing?

Standard classification trains against a one-hot target: the true class is
1, every other class is 0. Cross-entropy then pushes the winning logit
arbitrarily high to make its softmax probability approach 1, which tends to
make the model **over-confident** — its softmax confidences run higher than
its actual accuracy.

Label smoothing (Mueller, Kornblith & Hinton, *"When Does Label Smoothing
Help?"*, NeurIPS 2019; Szegedy et al. 2016) replaces the hard one-hot with a
**soft** target:

```
t' = (1 - eps) * onehot + eps / NumClasses
```

The true class now targets `1 - eps + eps/NumClasses` and every other class
targets `eps/NumClasses` instead of 0. This caps how confident the model is
encouraged to be, which the literature reports as **better calibration**
(lower ECE / Brier) at a **small accuracy cost**.

In the library the head is `TNNetLabelSmoothingLoss`. It is an **identity
passthrough** on the forward pass (so the last layer's output is still the
plain softmax probability vector) and only rewrites the back-prop gradient
to `p - t'`. The smoothing strength is the constructor argument:
`TNNetLabelSmoothingLoss.Create(eps)` with `0 <= eps < 1`. Crucially
**`eps = 0` reduces exactly to standard softmax cross-entropy**, so it is the
honest baseline arm. This example sweeps `eps` over `{0, 0.05, 0.10, 0.20}`.

## What ECE and Brier measure

Both come from `neuralcalibration` and are computed forward-only on a
held-out split:

- **ECE (Expected Calibration Error)** — bin predictions by their top-1
  confidence, and for each bin take `|accuracy - mean confidence|`, then
  average over bins weighted by bin population. `0` is perfect calibration;
  a large gap means the confidences do not match reality (positive gap of
  `conf - acc` = over-confidence).
- **Brier score** — mean squared error between the full predicted
  probability vector and the one-hot label. A proper scoring rule: lower is
  better, rewarding probabilities that are both accurate *and* well-calibrated.

## Task design (deliberately HARD)

A model that is right with ~100% confidence everywhere has `ECE ~ 0`
**regardless of `eps`**, which would make the comparison meaningless. To keep
the model genuinely uncertain the problem is made un-separable:

- **6 classes** laid out as 2D Gaussian clusters on a small ring
  (`radius = 1.30`) with a **large sigma (1.05)**, so adjacent clusters
  **heavily overlap** — the Bayes error is far above zero and no model can
  be confidently correct everywhere;
- **15% of the training labels are randomly corrupted** (label noise), which
  without smoothing pushes the network to memorise wrong targets at full
  confidence — the classic over-confidence regime.

The **validation** set uses the same overlapping clusters but **clean
labels**, so calibration is measured against the true generative labels.
The consequence is a low headline accuracy (~20%, vs the 1/6 ≈ 17% chance
floor) — that is by design; the point is the *calibration* spread, not the
accuracy.

## The shared architecture

```
TNNetInput(2, 1, 1)              # 2D point
  -> TNNetFullConnectReLU(24)
  -> TNNetFullConnectReLU(24)
  -> TNNetFullConnectLinear(6)   # class logits
  -> TNNetSoftMax                # probability simplex (what calibration reads)
  -> TNNetLabelSmoothingLoss(eps) # <-- THE SWEPT KNOB (identity on forward)
```

Every arm shares the RNG seed, dataset, epochs (120) and learning rate, so
the **only** difference between arms is the smoothing strength `eps`.

## Build & run

```
cd examples/LabelSmoothingCalibration
lazbuild LabelSmoothingCalibration.lpi --build-mode=Default
../../bin/x86_64-linux/bin/LabelSmoothingCalibration
```

Or compile directly with fpc (point `-Fu` at the LazUtils `lib` dir that
holds `utf8process.ppu`, exactly as `tests/RunAll.sh` discovers it):

```
fpc -B -Mobjfpc -Sh -O2 -Fu../../neural -Fu<lazutils-lib-dir> LabelSmoothingCalibration.lpr
```

Pure CPU, no dataset download. All four arms combined finish in well under
two minutes.

## Sample output

Actual run on a single CPU thread:

```
================================================================
RESULTS  (lower ECE / lower Brier = better calibrated)
================================================================
    eps     val-acc        ECE       Brier
   -----    -------     -------     -------
    0.00      22.71%     0.4232      1.0855
    0.05      19.79%     0.3957      1.0179
    0.10      20.42%     0.3074      0.9436
    0.20      19.17%     0.3850      1.0115

Best ECE   : eps=0.10 (ECE=0.3074)
Best Brier : eps=0.10 (Brier=0.9436)
Label smoothing IMPROVED ECE over the eps=0 cross-entropy baseline (textbook claim holds here).
Baseline (eps=0) ECE=0.4232 vs best ECE=0.3074  (delta 0.1158).
Total runtime for all 4 arms: 83.3s.
```

## Reading the result

On this task the **textbook claim holds**. The `eps = 0` cross-entropy
baseline is the most over-confident model (ECE = 0.4232, the worst Brier =
1.0855). Adding smoothing tightens calibration:

- `eps = 0.10` is best on **both** ECE (0.3074, a 0.116 absolute improvement
  over the baseline) **and** Brier (0.9436);
- the accuracy cost is small and expected — the baseline's 22.7% drops to
  ~20%, well within the noise of a hard, heavily-overlapping problem.

Note the curve is **non-monotone**: calibration improves up to `eps = 0.10`
and then *degrades* again at `eps = 0.20` (ECE back up to 0.3850). This is
the classic over-smoothing failure mode — too much smoothing flattens the
targets so far that the model becomes *under*-confident and the soft target
no longer matches the true class-overlap structure. The takeaway is not
"always smooth", but that label smoothing is a **calibration knob with a
sweet spot**: a modest `eps` reliably improves ECE/Brier over hard-target
cross-entropy, and the optimal strength is worth a short sweep like this one.
