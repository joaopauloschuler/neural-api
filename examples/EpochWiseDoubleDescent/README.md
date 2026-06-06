# Epoch-Wise (Temporal) Double Descent

Reproduces the **third axis** of double descent from Nakkiran et al. 2020
(*Deep Double Descent: Where Bigger Models and More Data Hurt*) — the
**epoch-wise** figure — on a pure-CPU toy, using only existing in-tree layers
(no new layer is added).

A **fixed**, mildly over-parameterised MLP is trained on a **small,
label-noisy** classification set, and held-out **test error is charted against
training EPOCH**. The only swept axis is **time**.

## The phenomenon

Over training time the test-error curve is **non-monotone**:

1. **FALLS** — classical learning. The net first captures the clean,
   low-complexity signal (the blob structure) that *generalises*; test error
   drops to a low **valley**.
2. **RISES to an interior PEAK** — around the epoch where **train error hits
   ~0** and the net is forced to **interpolate (memorise) the noisy labels**.
   Fitting the flipped labels is high-complexity and wrecks generalisation, so
   test error spikes.
3. **FALLS AGAIN** — with continued training the interpolating solution drifts
   toward a flatter, slightly better-generalising minimum; test error settles
   back down below the peak.

The signature is the **down → up → down** test-error trajectory *in time*.

## Contrast with the siblings (this is the important distinction)

| | swept axis | held fixed | driver |
|---|---|---|---|
| `examples/DoubleDescent/` | **model CAPACITY** (width `H`), measured at end of training | time (train to convergence) | label NOISE; peak sits at the interpolation **threshold** |
| **this example** | **EPOCH count** (time) | **CAPACITY and weight decay** | label NOISE; peak sits at the interpolation **epoch** |
| grokking | EPOCH count (time) | capacity | **WEIGHT DECAY on CLEAN labels** — late *sudden* generalisation, not a noise-driven down-up-down |

- `DoubleDescent/` (the **model-wise** sibling) sweeps **capacity** at the end of
  training and finds the test-error peak at the interpolation *threshold* — the
  smallest width that fits the noisy set. Time is fixed; capacity is the
  variable. **This example is a fork in spirit**: same kind of synthetic task
  and label noise, but here **one net is fixed and EPOCHS are swept**.
- **Grokking** is also a *time* axis at fixed capacity, but it is driven by
  **weight decay on CLEAN labels** and shows up as *delayed sudden*
  generalisation (test error stays bad, then abruptly drops). Here labels carry
  **NOISE**, weight decay is **fixed** (`0`), and the signature is the
  noise-driven **down-up-down**, not a late jump.

So: in this example **capacity AND weight decay are FIXED, the labels carry
NOISE, and the only swept axis is the epoch count.**

## The experiment

- **Teacher**: two well-separated `D=6` Gaussian **blobs**, one per class. Blob
  membership is a clean, low-complexity, *easily-learnable* target — the net
  captures it early, which is what lets the early test-error valley get
  genuinely low (`~0.09`). (The model-wise sibling uses a `sign(x'Qx+b'x)`
  teacher; here an *easy* signal is essential so a real "down" precedes the
  "up".)
- **Train set**: a SMALL fixed set of `50` points with **18% of labels
  flipped** (label noise — the peak driver).
- **Test set**: a LARGE `4000`-point **clean** held-out set (no flips).
- **Model (FIXED)**:
  `Input(6) -> FullConnectReLU(64) -> FullConnectReLU(64) ->
  FullConnectLinear(2) -> SoftMax`, trained as a SoftMax classifier on one-hot
  targets. `CountWeights = 4608` vs `50` training points — mildly
  over-parameterised, so it can eventually interpolate the noisy labels. Class
  is read back as `argmax` of the softmax output.
- **Optimiser**: plain **mini-batch SGD**, batch `5`, LR `0.01`, **momentum
  `0`**, **weight decay `0`**, for `3000` epochs. **NO `NeuralFit`** — a
  hand-rolled per-epoch loop logs train and test 0/1 error every `20` epochs
  deterministically (`RandSeed = 424242`, single-threaded per-sample
  Compute/Backpropagate).

## Self-gate (asserts the GENUINE invariants; `Halt(1)` on failure)

1. **Interpolation actually happens** — train 0/1 error first reaches `~0`.
2. **The test trajectory is NON-MONOTONE with an INTERIOR peak** strictly above
   **both** its earlier valley **and** its final value (the down-up-down
   signature). The peak must be a true interior point (not the first or last
   logged epoch).

Each prints a `[PASS]`/`[FAIL]` line; if either fails the program exits non-zero
and **reports which knob to turn** rather than silently passing.

## How to run

From inside `examples/EpochWiseDoubleDescent/`, with Free Pascal (`fpc`):

```
LAZUTILS_PATH=/usr/share/lazarus/4.4.0/components/lazutils/lib/x86_64-linux
fpc -B -Fu../../neural -Fu"$LAZUTILS_PATH" -Mobjfpc -Sh -O2 EpochWiseDoubleDescent.lpr
./EpochWiseDoubleDescent
```

(Adjust the lazutils path to your Lazarus install if `4.4.0` is absent;
`tests/RunAll.sh` documents the override.) Or open `EpochWiseDoubleDescent.lpi`
in Lazarus and build the *Release* mode. Pure CPU, no external data,
deterministic; finishes in about **12 seconds**.

## Sample output (real, seed 424242)

```
================================================================
Epoch-wise (TEMPORAL) Double Descent: test error vs EPOCH.
================================================================
Teacher: 2 separated Gaussian blobs, D=6.  Train=50 (18% label noise),
Test=4000 (clean).  FIXED net: Input(6)->FCReLU(64)->FCReLU(64)
->FCLinear(2)->SoftMax.  Only swept axis = EPOCH.
Epochs=3000  log every 20  LR=0.010  momentum=0.00  wd=0.000  seed=424242
Capacity and weight decay are FIXED (contrast: DoubleDescent/ sweeps
CAPACITY; grokking is weight-decay-driven on CLEAN labels).

Net parameters (CountWeights) = 4608  vs  train points = 50

Training (epoch-wise): ...............................................

  (V=early valley, ^=interior PEAK, I=interpolation epoch)
     epoch  trErr  testErr | test-error curve (bar scaled x2, 0.5=full)
         0  0.520   0.541  |##############################
 V      20  0.120   0.092  |######
        80  0.040   0.153  |#########
       160  0.020   0.209  |#############
I      220  0.000   0.228  |##############
  ^    240  0.020   0.294  |##################
       320  0.000   0.238  |##############
       400  0.000   0.235  |##############
       480  0.000   0.238  |##############
       560  0.000   0.237  |##############
       640  0.000   0.243  |###############
       ...
      2960  0.000   0.250  |###############
      3000  0.000   0.250  |###############

=== Self-gate: the GENUINE epoch-wise invariants ===
[PASS] interpolation: train 0/1 error first hits ~0 at epoch 220.
[PASS] interior peak: test err valley=0.092 (epoch 20) -> PEAK=0.294 (epoch 240) -> final=0.250 (epoch 3000). Down-up-down confirmed.

=> ALL GATES PASS: epoch-wise (temporal) double descent reproduced.
Total wall-clock: 11.8 s
```

### Reading the curve

```
test err  0.541  initial (epoch 0, untrained)
          0.092  early VALLEY (epoch 20) -- clean blob signal learned, generalises
          0.294  interior PEAK (epoch 240) -- noisy labels just interpolated <-- the spike
          0.250  continued training (epoch 3000), descended off the peak
```

Train 0/1 error confirms the mechanism: it collapses `0.52 -> 0.12 -> 0.04 ->
0.00` and first hits `0` at epoch `220`; the **test-error peak lands right
there** (epoch `240`), exactly when the net is forced to fit the flipped
labels. After the peak the over-fit solution relaxes and test error descends.

## Notes / honesty (the peak is FRAGILE)

This is a *toy* reproduction tuned to be fast and deterministic on a single CPU.
The epoch-wise peak is **delicate** and depends on three knobs being in balance
(mirroring the caveats in `RandomLabelMemorization/` and `DropPathAblation/`):

- **Enough label noise (`~15–20%`)** to *force* memorisation. With too little
  noise there is nothing high-complexity to memorise, the curve stays ~monotone,
  and `peak ≈ valley` (gate 2 fails — *raise the noise*).
- **A large-enough net** to *eventually interpolate*. If the net never drives
  train error to `0`, no interpolation peak forms (gate 1 fails — *raise width
  or epochs*).
- **A small-enough LR (and an EASY clean signal)** so the noise-fitting phase is
  **temporally resolvable**. An earlier all-linear-teacher / full-batch /
  high-LR configuration interpolated within ~50 epochs and the clean-signal and
  noise-fitting phases *collapsed onto each other*: train error hit `0`
  immediately and the test curve was nearly flat at `~0.29` (no valley, no
  peak). Two fixes were decisive: **(a)** switch the teacher to well-separated
  Gaussian blobs so the clean signal is learned *fast and well* (a low valley
  appears), and **(b)** use **mini-batch SGD with no momentum** at a small LR so
  the noise-memorisation phase is spread over hundreds of epochs and the peak is
  visible. Momentum or a large LR re-collapses the phases.

If you retune and the peak refuses to appear within the CPU budget, the gate
**reports which knob failed** (peak≈valley → noise; peak at an edge → epochs;
peak≈final → LR) and still `Halt(1)`s — it is **not** weakened to always pass.

The *shape* (valley → peak-at-interpolation → second descent, present only with
noise on a fixed over-parameterised net) is the robust, reproducible claim; the
exact valley/peak/final values are seed-dependent. Note the final error
(`0.250`) sits **above** the early valley (`0.092`): on a tiny noisy set the
memorised flips do permanent damage, so the second descent only partially
recovers — the *interior peak above both ends* is the signature the gate tests,
not a return to the valley.

## References

- P. Nakkiran, G. Kaplun, Y. Bansal, T. Yang, B. Barak, I. Sutskever,
  *Deep Double Descent: Where Bigger Models and More Data Hurt*, ICLR 2020.
  https://arxiv.org/abs/1912.02292
- M. Belkin, D. Hsu, S. Ma, S. Mandal, *Reconciling modern machine-learning
  practice and the classical bias-variance trade-off*, PNAS 2019.
  https://arxiv.org/abs/1812.11118

## Pairs naturally with

- `examples/DoubleDescent/` — the **model-wise** sibling (capacity axis).
- `examples/RandomLabelMemorization/` — the same fit-the-noise machinery, viewed
  as memorisation capacity rather than as a temporal curve.
