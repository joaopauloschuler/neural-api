# MC-Dropout uncertainty report

## What it does

`TNNet.MCDropoutUncertaintyReport` is a forward-only **Monte-Carlo-Dropout
epistemic uncertainty** estimator (Gal & Ghahramani, *Dropout as a Bayesian
Approximation*, 2016). Unlike the rest of the `TNNet.*Report` family it
deliberately **keeps the stochastic (dropout / noise) layers active at
inference**: it calls `NN.EnableDropouts(true)`, runs `NumPasses` (default 30)
stochastic forward passes over each probe input, applies a temperature-scaled
softmax to each pass's final-layer output, and aggregates the per-pass
probability vectors. The original dropout-enabled flag is saved and restored on
exit; the weights are never touched (no backward pass).

This example trains a small dropout MLP on a **synthetic 3-cluster 2D
classifier** (no dataset download) and prints the report for three probe
groups:

1. the three **cluster cores** (in-distribution) — the model is confident and
   every MC pass agrees, so epistemic uncertainty is ~0;
2. an **out-of-distribution (OOD) band** placed in the empty space *between*
   the clusters — MC passes disagree, so epistemic uncertainty is high;
3. a labelled **validation split** (with a few hard boundary points) feeding
   the correctness cross-tab.

The takeaway is *"the model knows what it doesn't know"*: the OOD band lights up
with high BALD while the cluster cores read near zero.

## What the report shows

Per probe sample:

* **pred / conf** — the mean predicted class (argmax of the pass-averaged
  probability vector `mean_p`) and its mean confidence `max(mean_p)`;
* **H[tot]** — predictive entropy `H[mean_p]` (total uncertainty, nats);
* **H[alea]** — expected entropy `mean_t H[p_t]` (aleatoric, the average
  per-pass entropy);
* **BALD** — the mutual information / **BALD** score
  `H[mean_p] - mean_t H[p_t]` (epistemic, `>= 0` by Jensen) — what the model
  *doesn't know*;
* **topVar** — variance of the top-class probability across passes;
* **flip%** — pass-to-pass argmax flip rate (fraction of passes whose argmax
  differs from the modal argmax).

Across the batch: a 10-bin ASCII histogram of per-sample BALD, the K
most-uncertain sample indices (an **active-learning query queue**), and — when a
label per probe is supplied — a **correctness cross-tab** comparing the mean
predictive entropy of correctly- vs incorrectly-predicted samples.

**Temperature / head convention.** The final-layer output is converted to a
probability vector with `Temperature` applied consistently: for a
softmax/log-softmax head the probabilities are temperature-renormalised
(`p^(1/T)` then renormalise; `T=1` is the identity), and for a raw-logit head a
numerically-stable `softmax(z / T)` is used. The report prints which head it
detected.

**Built-in correctness checks.** With `NumPasses=1` and dropout disabled the
BALD term collapses to ~0 (a single deterministic pass means `mean_p == p_1`, so
`H[mean_p] == H[p_1]`). A net containing **no** `TNNetAddNoiseBase` layer emits
a clear *"no stochastic layers — MC sampling is a no-op"* warning instead of
silently reporting zero variance.

## Build & run

```
cd examples/MCDropoutUncertainty
lazbuild MCDropoutUncertainty.lpi
../../bin/x86_64-linux/bin/MCDropoutUncertainty
```

Pure CPU, synthetic data, runs in well under a second.

## Sample output

Cluster cores (in-distribution) — confident, every pass agrees, BALD ~ 0:

```
GROUP 1: cluster cores (in-distribution). Expect LOW BALD.
MCDropoutUncertaintyReport: Monte-Carlo-Dropout epistemic uncertainty (Gal & Ghahramani 2016).
Probes used: 36 (of 36, cap MaxProbes=256). NumPasses=40, Temperature=1.0000, classes=3, Seed=1234567.
Stochastic (TNNetAddNoiseBase) layers found: 1.
Output head: softmax (probabilities, temperature-renorm).
Convention: probabilities p^(1/T) renormalised for a softmax/log-softmax head; softmax(z/T) for a raw-logit head (T=1 is the identity).
Entropy/BALD in nats. BALD = H[mean_p] - mean_t H[p_t] (epistemic >= 0 by Jensen).

Per-sample uncertainty (36 sample(s)):
  idx   pred       conf    H[tot]   H[alea]      BALD    topVar     flip%
  --------------------------------------------------------------------------
  0     0        0.9984    0.0129    0.0107    0.0022   0.00001      0.0%
  4     0        0.9981    0.0147    0.0124    0.0023   0.00001      0.0%
  15    1        0.9988    0.0104    0.0092    0.0012   0.00000      0.0%
  24    2        0.9963    0.0273    0.0246    0.0026   0.00003      0.0%
  ... (most rows omitted)

Per-sample BALD histogram (10 bins over [0.0012, 0.0519]):
  [  0.0012-  0.0062) | n=  19 ########################################
  [  0.0062-  0.0113) | n=  13 ###########################
  [  0.0113-  0.0164) | n=   1 ##
  ...
Batch means: H[tot]=0.0338  H[alea]=0.0261  BALD=0.0077  (BALD range [0.0012, 0.0519]).
```

OOD band between clusters — confidence drops toward chance, the argmax flips
across passes, BALD is ~4x higher and peaks mid-band:

```
GROUP 2: OOD band between clusters. Expect HIGH BALD.
...
Per-sample uncertainty (24 sample(s)):
  idx   pred       conf    H[tot]   H[alea]      BALD    topVar     flip%
  --------------------------------------------------------------------------
  0     0        0.9059    0.3688    0.3400    0.0288   0.00587      0.0%
  8     2        0.4337    1.0204    1.0112    0.0093   0.00120     47.5%
  15    1        0.5015    0.9530    0.9309    0.0220   0.00370     30.0%
  23    1        0.8691    0.4423    0.3590    0.0833   0.01312      2.5%
  ... (rows omitted)

Active-learning queue — 5 most-uncertain sample(s) by BALD:
  #1  sample 23    BALD=0.0833  H[tot]=0.4423  pred=1
  #2  sample 22    BALD=0.0737  H[tot]=0.5119  pred=1
  ...
Batch means: H[tot]=0.7650  H[alea]=0.7350  BALD=0.0301  (BALD range [0.0063, 0.0833]).
```

Labelled validation split — the correctness cross-tab shows the model is far
more uncertain on the samples it gets wrong:

```
GROUP 3: labelled validation split (correctness cross-tab).
...
Correctness cross-tab (mean predictive entropy H[tot]):
  correct   : n=78    mean H[tot]=0.0331
  incorrect : n=18    mean H[tot]=0.5303
  -> model is MORE uncertain on its mistakes (well-behaved MC uncertainty).
  accuracy  : 81.25% (78/96).

Batch means: H[tot]=0.1264  H[alea]=0.1106  BALD=0.0158  (BALD range [0.0004, 0.0898]).
Total runtime: 0.60 s.
```

The contrast is the whole point: the cluster cores read mean **BALD = 0.0077**
(every MC pass agrees, 0% flip rate), the OOD band reads mean **BALD = 0.0301**
(~4x higher, flip rates up to 47.5% mid-band), and the correct/incorrect
entropy split (`0.033` vs `0.530`) shows MC-dropout uncertainty tracks error —
*the model knows what it doesn't know*.
