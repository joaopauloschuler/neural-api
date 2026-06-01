# Tversky alpha/beta asymmetry sweep

This example is a pure **alpha/beta knob study** on the `TNNetTverskyLoss`
segmentation head. It forks `examples/DiceSegmentation/` but, instead of
contrasting Tversky against an MSE baseline, it trains the **same tiny net**
three times on the **same deliberately class-imbalanced** synthetic mask, once
per `(alpha, beta)` pair, and shows how `beta > alpha` trades precision for
recall. It is pure CPU, uses no external dataset, and finishes in ~1–2 seconds.

## What it does

The Tversky loss (Salehi et al. 2017) generalises Dice:

```
TI = TP / (TP + alpha*FP + beta*FN),   loss = 1 - TI
```

`alpha` weights false positives, `beta` weights false negatives. The sweep is:

| (alpha, beta) | meaning                                              |
|---------------|------------------------------------------------------|
| (0.5, 0.5)    | reduces to **Dice** (balanced FP/FN penalty)         |
| (0.3, 0.7)    | `beta > alpha`: false **negatives** penalised harder |
| (0.7, 0.3)    | `alpha > beta`: false **positives** penalised harder |

Raising `beta` above `alpha` makes a *missed* foreground pixel (FN) more
expensive than a *spurious* one (FP), so the trained model is pushed to predict
MORE foreground: recall rises and FN falls, at the cost of precision (more FP).

The task is intentionally hard and imbalanced so the trade is **visible**
rather than saturating: a **small** disc (radius 2–3, a minority of pixels) on a
`12x12` grid, with **heavy** input noise (`stddev 0.85`) so the net cannot
trivially fit every sample. The foreground is ~15% of the pixels.

> **Tuning note (documented honestly):** an easy/separable toy makes all three
> settings saturate near perfect overlap, hiding the trade. Two knobs were
> turned to keep the recall/FN trend monotone and visible: the disc is kept
> small (clear minority class) and the input noise is high (`cNoise = 0.85`), so
> the model genuinely has to choose how aggressively to call foreground.

## How the Tversky head is wired

The wiring mirrors `DiceSegmentation` exactly — only the head differs.
`TNNetTverskyLoss` is a `TNNetIdentity` descendant: its forward pass is an
identity passthrough, so `Net.Compute` still returns the `Sigmoid`
probabilities. The framework seeds the last layer's `FOutputError` with
`(output - target)`; the head recovers the binary ground-truth mask and
overwrites the residual with the analytic Tversky gradient driven by the
configured `(alpha, beta)`. Because the head reads `p` as a foreground
**probability in `[0,1]`**, the feeding layer must be a `Sigmoid`, and the
**target** supplied to `Backpropagate` is the binary mask:

```
Input(12, 12, 1)
ConvolutionReLU(6, 3, 1, 1)            // featuresize 3, pad 1, stride 1
ConvolutionReLU(6, 3, 1, 1)            //   => spatial 12x12 preserved
ConvolutionLinear(1, 3, 1, 1)         // 1-channel logit map, same 12x12
Sigmoid                               // per-pixel foreground probability
TNNetTverskyLoss(alpha, beta, 1.0)    // identity passthrough + analytic grad
```

`(0.5, 0.5)` is exactly `TNNetDiceLoss`. All three runs start from the same
`RandSeed`, so they see identical data and identical initial weights — the only
thing that changes between runs is `(alpha, beta)`.

## How to run

```
fpc -B -Funeural -Mobjfpc -Sh -O2 TverskySweep.lpr
./TverskySweep
```

or with Lazarus:

```
lazbuild TverskySweep.lpi
../../bin/x86_64-linux/bin/TverskySweep
```

The run is deterministic (`RandSeed` is fixed), pure CPU, and finishes in
~1–2 seconds total for all three runs.

## Sample output

```
TverskySweep: alpha/beta knob study on TNNetTverskyLoss
grid=12x12  features=6  train=192  test=96  epochs=24  lr=0.040  noise=0.85

Held-out class balance: foreground 2048 / 13824 pixels (14.8%),  background 11776 (85.2%)
=> foreground is a deliberate minority (class-imbalanced mask).

Training (alpha=0.5, beta=0.5)  [= Dice] ...
Training (alpha=0.3, beta=0.7) ...
Training (alpha=0.7, beta=0.3) ...

Held-out results (threshold 0.5, counts summed over all test pixels)
  (alpha,beta)   note      precision   recall      F1/Dice      FP      FN
  (0.5, 0.5)   Dice        0.7225     0.8872     0.7964     698     231
  (0.3, 0.7)               0.6780     0.9160     0.7792     891     172
  (0.7, 0.3)               0.7745     0.8335     0.8029     497     341

Trend as beta rises relative to alpha (beta-alpha: -0.4 -> 0 -> +0.4):
  recall:  0.8335 -> 0.8872 -> 0.9160
  FN:        341 ->   231 ->   172
  precision:0.7745 -> 0.7225 -> 0.6780
  FP:        497 ->   698 ->   891

=> CONFIRMED: beta>alpha raises RECALL and lowers FN (it trades precision/FP for recall).
```

## Reading the result

Order the three runs by `beta - alpha` (`-0.4 -> 0 -> +0.4`, i.e.
`(0.7,0.3) -> (0.5,0.5) -> (0.3,0.7)`) and the headline trade is monotone and
clear:

- **Recall rises**: `0.8335 -> 0.8872 -> 0.9160`. As `beta` grows the model is
  penalised more for missing foreground, so it labels more pixels foreground.
- **False negatives fall**: `341 -> 231 -> 172`. Fewer true-foreground pixels
  are missed — exactly what a higher `beta` rewards.
- **Precision drops / false positives rise**: precision `0.7745 -> 0.7225 ->
  0.6780`, FP `497 -> 698 -> 891`. The extra recall is bought with more
  spurious foreground calls.

`(0.5, 0.5)` sits in the middle and is identical to `TNNetDiceLoss`. So the
single `(alpha, beta)` knob on the landed `TNNetTverskyLoss` directly controls
the precision/recall operating point: **push `beta` up to chase recall (fewer
missed regions), push `alpha` up to chase precision (fewer false alarms).** The
printed numbers are exactly what this run produced, and the printed verdict
reflects whatever the run actually shows.
