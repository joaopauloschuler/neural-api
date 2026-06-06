# Random Label Memorization

Reproduces the headline result of Zhang, Bengio, Hardt, Recht & Vinyals,
*Understanding deep learning requires rethinking generalization* (ICLR 2017) on
a pure-CPU toy, using only existing in-tree layers (no new layer is added).

## The phenomenon

A sufficiently over-parameterised network can fit **anything** — including a
training set whose labels have been replaced by **pure noise**. Zhang et al.
showed that the *same* network that learns a real task to ~100% train accuracy
*also* drives train accuracy to ~100% on the very same inputs with their labels
**randomly shuffled**. Fitting random labels is genuine **memorisation**: there
is no signal left to learn, so the network simply stores the answers.

The shock is what this does to **generalisation**:

- with **true** labels the network generalises — held-out test accuracy is far
  above chance;
- with **random** labels it *cannot possibly* generalise (the held-out labels
  are independent of the inputs), so test accuracy sits at **chance** (`1/K`).

The training loss/accuracy is **identical** in both regimes (both ~100%), yet
one generalises and one does not.

> **Takeaway: train error alone says NOTHING about generalisation.** Capacity to
> fit the training data is not, by itself, evidence of having learned anything.

This is **distinct** from [`examples/DoubleDescent`](../DoubleDescent), which
sweeps model **capacity** under a *fixed* amount of label noise and charts the
non-monotone test-error curve. Here **capacity is fixed** and the contrast is
**true labels vs random labels**.

## The experiment

- **Task**: `K=5` Gaussian blobs with fixed random centres in `D=20`
  dimensions. The TRUE label is which blob a point was drawn from — a clean,
  learnable target. Chance test accuracy is `1/K = 20%`.
- **Train set**: a SMALL fixed set of `200` points.
- **Test set**: a LARGE `1000`-point held-out set (always TRUE labels).
- **The FIXED over-parameterised net** (identical for both runs):
  `Input(20) -> TNNetFullConnectReLU(64) -> TNNetFullConnectReLU(64) ->
  TNNetFullConnectLinear(5) -> TNNetSoftMax` — `5696` weights for `200`
  training points, so it is heavily over-parameterised.
- **Two runs, same net, same inputs, same weight-init seed**:
  1. trained on the **TRUE** labels;
  2. trained on the **RANDOMLY SHUFFLED** labels (the label column is permuted,
     so the marginal label distribution is preserved but all input→label signal
     is destroyed; the inputs are untouched).
- **Optimiser**: mini-batch SGD (`SetBatchUpdate(True)`, accumulate the
  cross-entropy gradient over a shuffled mini-batch of `25`, one
  `UpdateWeights` step per batch), LR `0.02`, momentum `0.9`. The random-label
  run is allowed more epochs (up to `4000`) because memorising noise is harder
  than learning real structure, with an early-stop once train accuracy is
  perfect. The high input dimension makes the `200` points individually
  separable, so the net can drive **random-label train accuracy all the way to
  100%** — the capacity-to-memorise claim.

## Built-in correctness gate (printed PASS/FAIL, `Halt(1)` on failure)

1. **random-label TRAIN acc ≥ 99%** — the net memorises pure noise.
2. **random-label TEST acc ≤ chance + 5%** — memorising noise does not
   generalise (test accuracy stays at `~1/K`).
3. **true-label TEST acc ≫ chance (≥ 60%)** — the same net on true labels really
   does generalise.
4. **BOTH runs reach ~100% TRAIN acc** — identical train error, opposite
   generalisation.

## How to run

```
# From this directory, with Free Pascal (fpc) installed:
fpc -O3 -Mobjfpc -Sh -Fu../../neural -dRelease RandomLabelMemorization.lpr
./RandomLabelMemorization
```

Or open `RandomLabelMemorization.lpi` in Lazarus and build the *Release* mode.
Pure CPU, no external data, deterministic (`RandSeed = 424242`), finishes in
under 10 seconds.

## Sample output (real, seed 424242)

```
================================================================
Random-Label Memorization (Zhang et al. 2017): train error says
NOTHING about generalization.
================================================================
Task: 5-class Gaussian blobs, D=20.  Train=200, Test=1000 (clean).
Chance test accuracy = 1/K = 0.200.
FIXED net: Input(20)->FullConnectReLU(64)->FullConnectReLU(64)->FullConnectLinear(5)->SoftMax.
Mini-batch SGD  batch=25  LR=0.020  momentum=0.90  RandSeed=424242
Same net + same inputs, trained on TRUE labels vs RANDOMLY SHUFFLED labels.

Training on TRUE labels   done (50 epochs).
Training on RANDOM labels done (625 epochs).

=== Results ===
run            params  epochs   TRAIN acc   TEST acc
TRUE labels      5696      50    100.00%     100.00%
RANDOM labels    5696     625    100.00%      19.30%
chance (1/K) test accuracy = 20.00%

=== Correctness gate ===
[PASS] random-label TRAIN acc = 100.00% (must be >= 99%): the net memorised pure noise.
[PASS] random-label TEST  acc = 19.30% (must be <= chance+5% = 25.00%): no generalisation.
[PASS] true-label   TEST  acc = 100.00% (must be >> chance, >= 60.00%): real generalisation.
[PASS] BOTH runs reach ~100% TRAIN acc (true=100.00%, random=100.00%): same train error, opposite generalisation.

TAKEAWAY: both runs fit the training set perfectly (train error ~0),
yet only the true-label run generalises. TRAIN ERROR ALONE SAYS
NOTHING ABOUT GENERALIZATION -- capacity to fit the data is not
evidence of having learned anything (Zhang et al. 2017).

=> ALL CHECKS PASS: random-label memorization reproduced.
Total wall-clock: 8.5 s
```

### Reading the result

Both runs reach **100% train accuracy** on the same inputs and the same fixed
net — the only difference is the labels. With true labels test accuracy is also
**100%**; with random labels it is **19.3%**, indistinguishable from the `20%`
chance rate. The random-label run takes more epochs (625 vs 50) to memorise,
exactly as Zhang et al. report, but it gets there. The train number is the same
in both cases and tells you nothing about which one learned anything.

## Notes / honesty

- This is a *toy* reproduction tuned to be fast and deterministic on a single
  CPU. The robust, reproducible claim is the *shape*: equal ~100% train accuracy
  with true vs random labels, but generalisation only for true labels.
- A high input dimension (`D=20`) is used so the `200` training points are
  individually separable; that lets the over-parameterised net drive
  random-label train accuracy cleanly to 100% within the time budget.

## Pairs naturally with

- [`examples/DoubleDescent`](../DoubleDescent) — fixes the *labels* and sweeps
  *capacity* (the complementary slice through the same memorisation story).

## References

- C. Zhang, S. Bengio, M. Hardt, B. Recht, O. Vinyals, *Understanding deep
  learning requires rethinking generalization*, ICLR 2017.
  https://arxiv.org/abs/1611.03530
