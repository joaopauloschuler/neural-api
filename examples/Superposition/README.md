# Superposition

A self-contained, pure-CPU reproduction of the headline result of Anthropic's
**Toy Models of Superposition** (Elhage et al. 2022,
<https://transformer-circuits.pub/2022/toy_model/index.html>): a network packs
**more sparse features than it has dimensions** by storing them in
*superposition* — non-orthogonal, mutually-interfering directions — and the
geometry it picks is governed by feature **sparsity**.

No new layer is needed; the demo uses only existing layers
(`TNNetFullConnectLinear`, `TNNetFullConnectReLU`).

## The phenomenon

A linear bottleneck of width `M` can store at most `M` features *orthogonally*
(one clean direction each, "monosemantic"). When features are **dense** (active
on most samples) they collide on every example, so the optimum is to keep only
the `M` most-*important* features orthogonally and drop the rest. When features
are **sparse** (rarely co-active), the model can afford to **cram in more than
`M`** of them, packing extra features into shared, slightly-interfering
directions ("polysemantic" superposition) and paying only a small interference
cost because two packed features rarely fire together.

## The toy

- Synthetic feature vectors of width `N = 20`. Each feature `i` is independently
  **active** with probability `(1 - S)` (sparsity `S`); when active its value is
  `~ U[0,1]`, else `0`.
- Per-feature **importance** `I_i = r^i` (`r = 0.8`, geometric decay): early
  features matter more, so the model must *choose which* to represent.
- Autoencoder with an `M < N` bottleneck:

  ```
  Input(N) -> TNNetFullConnectLinear(M)   {encoder W, M x N}
            -> TNNetFullConnectReLU(N)     {decoder D + bias + ReLU}
  ```

- Loss: **importance-weighted MSE** `L = sum_i I_i * (out_i - in_i)^2`. The
  importance weighting is the crux — it is what makes the model spend its
  limited `M`-dim capacity on the features that matter.
- Sweep `S ∈ {0.0, 0.7, 0.9, 0.99}`.

## Importance-weighted MSE with the stock backprop (no library changes)

The framework seeds the output layer's error as `(output − target)` and then
multiplies by the activation derivative. The gradient of the *weighted* MSE
w.r.t. the output is `I_i · (out_i − in_i)` (the constant 2 folds into the
learning rate). We obtain exactly that with the stock `Backpropagate` by feeding
a **pseudo-target**

```
pseudo_i = out_i − w_i · (out_i − in_i),     so   (output − pseudo)_i = w_i · (out_i − in_i)
```

with `w_i = I_i / batchSize`. The division by the batch size matters: in
batch-update mode (`SetBatchUpdate(True)`) `Backpropagate` **sums** the
per-sample error into `Neurons[].Delta`, so dividing by the batch size makes the
applied step the **mean** weighted-MSE gradient — this keeps the learning rate
batch-size-independent and the dense regime numerically stable.

The training loop is **hand-rolled** (`NN.Compute` / `NN.Backpropagate` over
freshly drawn random samples, `NN.UpdateWeights` once per mini-batch). Because we
hand-roll it, the post-training layer references never go stale (the
`TNeuralFit.Fit` best-model-reload gotcha does not apply here). Plain SGD,
momentum 0, no weight decay.

## Tied vs untied weights

The classic toy **ties** `decoder = encoder^T` and studies the Gram matrix
`W^T W` (`N x N`): its diagonal is the represented norm per feature and its
off-diagonals are the cross-feature interference. Here the two
`TNNetFullConnect` layers are **untied**, so we read the **effective pre-ReLU
linear map** instead:

```
G = D · W          (N x N,  D is the N x M decoder,  W is the M x N encoder)
```

`G` plays the role of `W^T W`: `G[i][i]` is how strongly feature `i` is
represented, the column norm `||G[:,i]||` is feature `i`'s **represented norm**,
and the off-diagonals are cross-feature **interference**. (Reading the layer
weights: encoder neuron `m` owns a size-`N` weight vector `W[m][:]`; decoder
neuron `i` owns a size-`M` weight vector `D[i][:]`; `G[i][k] = Σ_m D[i][m]·W[m][k]`.)

## What it reports, per sparsity level

- the per-feature **represented norm** `||G[:,i]||` as an ASCII bar chart (which
  features the model kept vs dropped to ~zero);
- the `N x N` effective map `G` as a **glyph-shaded heatmap** (near-identity =
  monosemantic / one-feature-per-direction; growing off-diagonal structure =
  polysemantic superposition);
- the scalar **superposition ratio** = (#features with non-trivial norm) / `M`
  (`≈1` dense → only `M` features represented orthogonally; `>1` sparse → extras
  crammed in);
- the **mean off-diagonal `|interference|`**.

## Built-in correctness signals (printed PASS / FAIL)

1. **Dense (S=0):** kept-feature count `≈ M` and `G` near-diagonal (small mean
   off-diagonal). No superposition is optimal when features collide every sample.
2. **Importance tracking:** at every sparsity level the kept features are more
   important on average (mean `I`) than the dropped ones — high-`I` features are
   kept first. (A robust mean-importance test is used rather than a strict
   prefix test, which is brittle near the keep threshold in the untied geometry.)
3. **Growth:** the number of features packed into the `M`-dim bottleneck grows
   as `S` rises (the superposition phenomenon itself). Raw `G` magnitudes are not
   comparable across `S` — the per-active-feature reconstruction scale changes
   with sparsity — so total represented norm is reported only as a diagnostic.

## Running

```
lazbuild examples/Superposition/Superposition.lpi --build-mode=Default
./bin/x86_64-linux/bin/Superposition
```

or directly with the compiler (the same unit paths the `.lpi` encodes):

```
cd examples/Superposition
fpc -B -Fu../../neural -Fu<lazutils-lib-path> -Mobjfpc -Sh -O3 Superposition.lpr
./Superposition
```

Pure CPU, no external data, fully deterministic (fixed seed), finishes in about
**1.5–2 minutes**.

## Sample output

```
========================================================================
Toy Models of Superposition (Elhage et al. 2022) -- pure-CPU reproduction
========================================================================
Features N=20, bottleneck M=5, importance I_i = 0.80^i.
Autoencoder: Input(20) -> Linear(5){encoder} -> ReLU(20){decoder}.
Importance-weighted MSE via pseudo-target; 4000 steps x batch 768, lr=1.000.
Effective map G = D*W (N x N); column-norm = represented norm, off-diag = interference.
"Kept" = represented norm >= 20% of the max column norm.
```

**S = 0.00 (dense)** — exactly `M = 5` features represented, a clean diagonal map
(monosemantic, ratio 1.00):

```
SPARSITY S = 0.00   (feature-active prob = 1.00)
  kept features = 5 / 20     superposition ratio = 1.00  (kept/M)
  total represented norm =   5.028   mean |off-diag interference| = 0.0009

  Per-feature represented norm  ||G[:,i]||  (decreasing importance left->right):
    f 0 I=1.000 |################################|  1.001  [kept]
    f 1 I=0.800 |################################|  1.002  [kept]
    f 2 I=0.640 |################################|  1.001  [kept]
    f 3 I=0.512 |################################|  1.001  [kept]
    f 4 I=0.410 |################################|  1.000  [kept]
    f 5 I=0.328 |                                |  0.008
    ...
    f19 I=0.014 |                                |  0.000

  Effective map G = D*W  (rows=outputs, cols=features; glyphs scaled to |max|):
         0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19
   o 0  #
   o 1     #
   o 2        #
   o 3           #
   o 4              #
   o 5
   ...
```

**S = 0.99 (sparse)** — 19 of 20 features packed into `M = 5` dims (ratio 3.80),
with structured off-diagonal interference:

```
SPARSITY S = 0.99   (feature-active prob = 0.01)
  kept features = 19 / 20     superposition ratio = 3.80  (kept/M)
  total represented norm =  17.267   mean |off-diag interference| = 0.1451

  Effective map G = D*W  (rows=outputs, cols=features; glyphs scaled to |max|):
         0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19
   o 0        .  .  .  :  .  .        .     .           .     .
   o 3  .  .     +  .  :  .  .  .        :        .     .  .  .  .
   o 7     .  .  .  +     .  +     .  .  .  :  .     :  .  .  .  .
   o12  .  .  .  .  .  .  .  :     .  .     :  .     .  .  .     .
   ...
```

**Correctness signals:**

```
(1) DENSE (S=0): kept-feature count should be ~ M and G near-diagonal.
    dense kept = 5  (M = 5);  mean |off-diag| = 0.0009
    kept ~ M : PASS
    G near-diagonal at S=0 (mean |off-diag| < 0.10) : PASS

(2) IMPORTANCE TRACKING: kept features more important (mean-I) than dropped.
    S=0.00 : mean-I(kept) > mean-I(dropped) = yes
    S=0.70 : mean-I(kept) > mean-I(dropped) = yes
    S=0.90 : mean-I(kept) > mean-I(dropped) = yes
    S=0.99 : mean-I(kept) > mean-I(dropped) = yes
    importance tracking : PASS

(3) GROWTH: # features packed into the M-dim bottleneck grows as S rises.
    S=0.00 : kept =  5  (ratio 1.00)   [diag: total norm =   5.028]
    S=0.70 : kept =  5  (ratio 1.00)   [diag: total norm =  27.026]
    S=0.90 : kept =  9  (ratio 1.80)   [diag: total norm =  16.149]
    S=0.99 : kept = 19  (ratio 3.80)   [diag: total norm =  17.267]
    represented-feature count grows (sparse > dense) : PASS
```

## Expected reading

The **superposition ratio** climbs monotonically with sparsity — `1.00` (dense:
exactly `M` features, orthogonal, monosemantic) → `1.80` → `3.80` (sparse: nearly
all `N` features packed into `M` dims, polysemantic) — which is the textbook
phase transition from the paper. At `S=0` the effective map `G` is an almost
perfect rank-`M` diagonal on the top-`M` most-important features; as `S → 1` the
off-diagonal interference fills in with structure while the model keeps
representing more and more features. The exact identity of the marginal kept
features at intermediate sparsities is mildly noisy (untied ReLU geometry with
near-tied importances), which is why signal (2) uses a robust mean-importance
criterion rather than an exact-ordering test.

## References

- N. Elhage et al., *Toy Models of Superposition*, Transformer Circuits Thread,
  2022. <https://transformer-circuits.pub/2022/toy_model/index.html>

## See also

This demo reproduces the sparse-feature-packing *phenomenon* and sweeps sparsity
to surface the monosemantic↔polysemantic transition — distinct from the in-tree
diagnostic reports (`FeatureSeparabilityReport`, `NeuronCorrelationReport`,
`RepresentationSimilarity`, `IntrinsicDimension`) which analyse a single trained
net's activations, and from the grokking / double-descent demos which vary
training time / capacity rather than feature geometry. It pairs naturally with
`WeightSpectrumReport` (the bottleneck spectrum fills in as features enter
superposition).
