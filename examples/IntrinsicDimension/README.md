# IntrinsicDimension

Demonstrates `TNNet.IntrinsicDimensionReport`: a **forward-only**
representation-geometry diagnostic answering *"how many effective dimensions
does each layer's activation cloud actually occupy?"* — the dimensionality of
the data **manifold** the probe batch traces out at each depth, not its class
structure or its neuron redundancy.

For every trainable layer it runs one forward pass over an **unlabelled** probe
batch, flattens each sample to a row of an `N x D_l` matrix, and reports **two
complementary intrinsic-dimension (ID) estimates** side by side.

## What it does

1. **(A) Ground-truth recovery** — feeds a probe batch lying on a known
   `k=3`-dimensional linear subspace (embedded in a higher ambient dimension via
   a fixed random map) through a fresh wide linear layer. Both `PCA_ID` and
   `TwoNN_ID` should land near `k` (the faithfulness check).
2. **(B) Fresh-init network** — the ID profile is flat / near-input at every
   layer.
3. **(C) Trained network** — the famous "hunchback" of Ansuini et al. 2019: the
   ID first **expands** in the early layers then **contracts** monotonically
   toward the output as the representation compresses onto the task manifold.

Pure CPU, forward-only (`NN.Compute` only — weights are never touched), runs in
well under a minute.

## What the report shows

Per trainable layer:

```
Idx  Class                         D_l    PCA_ID  TwoNN_ID      gap      comp
------------------------------------------------------------------------------
1    TNNetFullConnectReLU           48    18.565    15.134    3.431     0.315
...
```

- **PCA_ID** — the **linear / PCA** intrinsic dimension via the participation
  ratio of the activation covariance eigenspectrum `PR = (sum lambda)^2 / sum
  lambda^2` ("how many principal components hold the variance"). Eigenvalues
  come from the smaller of the `N x N` Gram or `D x D` covariance matrix via the
  same self-contained Double-precision cyclic Jacobi eigensolver
  `WeightSpectralTailReport` ships (no new numerical code). Non-negative (PSD).
- **TwoNN_ID** — the **TwoNN** nonlinear manifold estimator (Facco, Rodriguez,
  Glielmo & Laio 2017): for each sample take `mu = r2/r1`, the ratio of its
  2nd-to-1st nearest-neighbour Euclidean distances; sort the `mu`; read the
  manifold dimension `d` off the least-squares slope of `-log(1 - F(mu))`
  against `log(mu)` (a single line fit through the origin).
- **gap** = `PCA_ID - TwoNN_ID` — a positive gap means the representation sits on
  a low-dimensional **curved** manifold that PCA over-counts.
- **comp** = `TwoNN_ID / D_l` — the `D_l`-normalised compression ratio.
- Flags: `expanded` (`comp >= 0.50`), `compressed` (`comp <= 0.10`), and
  near-`full-rank` (`PCA_ID >= 0.90 * min(N-1, D_l)`).

Closing summary: a `TwoNN_ID`-across-depth ASCII bar chart (the expand-then-
contract "hunchback") and a PSD-violation flag list.

Built-in correctness checks (visible in block A / asserted in the smoke test):
a known-`k`-dim subspace recovers `PCA_ID ~ k` and `TwoNN_ID ~ k`; identical
samples drive both IDs to ~0; PCA eigenvalues are non-negative.

## Running

```
cd examples/IntrinsicDimension
lazbuild IntrinsicDimension.lpi
../../bin/<arch>/bin/IntrinsicDimension
```

Or directly with fpc:

```
cd examples/IntrinsicDimension
fpc -B -Fu../../neural -Mobjfpc -Sh -O2 IntrinsicDimension.lpr
./IntrinsicDimension
```
