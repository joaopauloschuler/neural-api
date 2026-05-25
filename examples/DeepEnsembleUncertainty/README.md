# Deep Ensembles predictive-uncertainty baseline

## What it does

This example implements **Deep Ensembles** (Lakshminarayanan, Pritzel &
Blundell, *Simple and Scalable Predictive Uncertainty Estimation using Deep
Ensembles*, NeurIPS 2017) — the gold-standard predictive-uncertainty baseline —
on a pure-CPU toy using **only existing layers** (no new layer type).

It trains **M = 5 INDEPENDENT** small softmax MLP classifiers. Every member has
the *identical* architecture (`Input(2) -> FC-ReLU(32) -> FC-ReLU(32) ->
FC-Linear(3) -> SoftMax`) but a **different `RandSeed`** per member, so each
member lands in a different weight basin. They are trained on the **same
synthetic 3-cluster 2D task** that
[`examples/MCDropoutUncertainty/`](../MCDropoutUncertainty/) uses — the data
generator and the three probe groups are copied verbatim so the two epistemic
estimates are apples-to-apples.

At inference it **averages the M post-softmax probability vectors** and reports,
on three probe groups (cluster cores / an out-of-distribution band in the empty
space *between* clusters / a labelled validation split):

* **(a) accuracy + calibration** — the *average single-member* top-1 accuracy
  and ECE/Brier versus the *ensemble's*, computed through
  `neuralcalibration.ComputeCalibration` / `CalibrationReport` (we do **not**
  hand-roll calibration; the ensemble-mean probability vectors are fed through a
  trivial `Input -> Identity` passthrough net so the existing forward-only
  calibrator sees them as a softmax head);
* **(b) the predictive-entropy decomposition** (nats):
  `total H[mean_p] = aleatoric (mean_m H[p_m]) + epistemic (mutual information)`,
  with the epistemic / mutual-information term `I = H[mean_p] - mean_m H[p_m] >= 0`
  by Jensen;
* **(c) an ASCII bar** of per-group epistemic uncertainty.

## Built-in correctness signals

The program asserts (and raises on failure):

* **M=1 reproduces member 0 bit-for-bit** — the ensemble-of-one is exactly that
  one model's probability vector (`max|diff| = 0`); the averaging code adds no
  drift;
* **ensemble accuracy >= mean single-member accuracy** on every labelled group;
* **epistemic MI >= 0 everywhere**, and **strictly higher on the OOD band than
  on the cluster cores**.

## How this differs from the sibling examples

* [`examples/MCDropoutUncertainty/`](../MCDropoutUncertainty/) estimates
  epistemic uncertainty by **sampling dropout masks inside ONE trained
  network** (MC-dropout, Gal & Ghahramani 2016). Deep ensembles instead use
  **M genuinely INDEPENDENT networks**. Both run on the **same** 3-cluster task,
  so the two epistemic estimates can be compared head-to-head: dropout samples
  *one* posterior mode, an ensemble explores *several*. Both light up on the OOD
  band and read ~0 on the cores.
* [`examples/KnowledgeDistillation/`](../KnowledgeDistillation/) treats an
  ensemble as the natural **teacher** to be compressed into one student. Here we
  **keep** the ensemble and **quantify its uncertainty** instead of distilling
  it away.
* **TestTimeAugmentation** averages predictions over input **transforms** of a
  *single* model — that captures input sensitivity, not the
  model-disagreement (epistemic) signal that M independent models give.

## Build & run

```
cd examples/DeepEnsembleUncertainty
lazbuild DeepEnsembleUncertainty.lpi
../../bin/x86_64-linux/bin/DeepEnsembleUncertainty
```

Pure CPU, synthetic data (no download), deterministic per seed list, runs in a
few seconds.

## Sample output

```
DeepEnsembleUncertainty: 5 INDEPENDENT softmax MLPs on a synthetic 3-cluster 2D task.
Training each member (different RandSeed) for 300 epochs of batch 48 ...
  member 0 (seed 2026) trained.
  ...
==============================================================================
(a) ACCURACY + CALIBRATION: average single member vs the ENSEMBLE
    (ECE/Brier via neuralcalibration.ComputeCalibration).
==============================================================================
  CLUSTER CORES (in-distribution):
    mean member:  acc=1.0000  ECE=0.0068  Brier=0.0001
    ENSEMBLE   :  acc=1.0000  ECE=0.0068  Brier=0.0001
  VALIDATION SPLIT (clean + hard boundary points):
    mean member:  acc=0.8125  ECE=0.1507  Brier=0.2497
    ENSEMBLE   :  acc=0.8125  ECE=0.1507  Brier=0.2470
  ... full CalibrationReport for the ENSEMBLE on the validation split ...

==============================================================================
(b) PREDICTIVE-ENTROPY DECOMPOSITION (nats): total = aleatoric + epistemic
    epistemic = mutual information I = H[mean_p] - mean_m H[p_m] >= 0
==============================================================================
  group           total(H)   aleatoric   epistemic(MI)   MI range
  cluster-cores     0.0453     0.0452        0.0001    [0.0000, 0.0003]
  OOD-band          0.9550     0.9278        0.0271    [0.0037, 0.0579]
  validation        0.1566     0.1519        0.0048    [0.0000, 0.0390]

==============================================================================
(c) PER-GROUP EPISTEMIC UNCERTAINTY (mean mutual information, nats)
==============================================================================
  cluster-cores  | 0.0001
  OOD-band       |################################################## 0.0271
  validation     |######### 0.0048
  (bar full-scale = OOD-band mean MI)

==============================================================================
CORRECTNESS SIGNALS
==============================================================================
  [PASS] M=1 reproduces member 0 bit-for-bit (max|diff|=0.00E+000).
  [PASS] ensemble acc >= mean-member acc on cores (1.0000 >= 1.0000).
  [PASS] ensemble acc >= mean-member acc on val   (0.8125 >= 0.8125).
  [PASS] epistemic MI >= 0 everywhere (min over groups = 0.0000).
  [PASS] OOD epistemic > cores epistemic (0.0271 > 0.0001).
Total runtime: ~5 s.
```

The contrast is the whole point. The **epistemic (mutual-information) term**
reads `0.0001` on the cluster cores (every member agrees) but `0.0271` on the
OOD band — **~270x higher**, peaking mid-band where the members most disagree —
and the ensemble's **Brier score** on the validation split (`0.2470`) beats the
average member's (`0.2497`) while matching its accuracy. On this saturated toy
the accuracy headroom is small, but the calibration gain and the OOD epistemic
spike are exactly the Deep-Ensembles signal: *the ensemble knows what it does
not know*.
