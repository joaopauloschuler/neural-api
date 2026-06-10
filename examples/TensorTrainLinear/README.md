# Tensor-Train structured-linear parameter/accuracy bake-off

This example showcases **`TNNetTensorTrain`**, a sub-quadratic **structured**
dense layer. A Tensor-Train (TT / Matrix-Product-State / MPO) layer factorises
an `n x n` linear map as a **chain of `d` small cores**. The flat dimension is
split `n = prod_k f_k`; core `k` has shape

```
G_k : r_{k-1} x f_k x f_k x r_k
```

with the boundary TT-ranks pinned to `r_0 = r_d = 1` and every interior rank
equal to a tunable rank `r`. The cores are contracted **left-to-right** (one
input leg consumed and one output leg emitted per core, carrying the TT-rank
index) so the dense `n x n` matrix is **never materialised**. Backward is the
standard *"freeze the other cores, contract"* rule applied as the exact reverse
sweep.

Total params `~ sum_k r_{k-1}*f_k*f_k*r_k = O(d * f^2 * r^2)` versus `n^2` for
the dense map. For the square case `n = 64` with `d = 2` cores
(`f_0 = f_1 = 8`) and interior rank `r`, the TT map stores only
`1*8*8*r + r*8*8*1 = 128*r` core weights. At `r = 4` that is **512** core
weights (`+64` bias `= 576`) versus **4096** for a dense
`TNNetFullConnectLinear` — a ~7x saving.

The square map **infers `n` from the previous layer** (no `N` arg), mirroring
`TNNetMonarchLinear` / `TNNetKroneckerLinear`. The constructor is
`Create(pSuppressBias, pCores, pRank)` — `pCores = 0` auto-picks `d = 2`,
`pRank = 0` defaults to `r = 2`. The number of cores `d` and the interior rank
`r` round-trip through serialization.

## The bake-off

Three square `64 -> 64` mixing layers are trained on the same regression task
(a fixed random linear-then-`tanh` teacher applied to random 64-vectors), each
followed by an **identical** tiny linear read-out head (`64 -> 8`). We print the
trainable weight count of just the mixing layer and its final training MSE:

| Arm | Mixer | Mixing weights |
| --- | --- | --- |
| Tensor-Train (structured) | `TNNetTensorTrain(0, d=2, r=4)` | **576** (`128*r + n`) |
| Dense baseline | `TNNetFullConnectLinear(64)` | 4096 (`n^2`) |
| Kronecker (structured) | `TNNetKroneckerLinear` (A⊗B, p=q=8) | 192 (`p^2 + q^2 + n`) |

## Headline result

```
=== Mixing-layer weight count vs final training MSE ===
  TensorTrain :   576 weights   final MSE 0.000680
  Dense FC    :  4096 weights   final MSE 0.000397
  Kronecker   :   192 weights   final MSE 0.009778

HEADLINE: TensorTrain uses 7.1x FEWER mixing weights than the
dense arm yet reaches a comparable training MSE.
```

The Tensor-Train arm uses **7.1x fewer** mixing weights than the dense arm yet
lands in the same accuracy ballpark — *structured = fewer params, comparable
accuracy*. Kronecker is shown as a second (even leaner) structured point on the
params-vs-accuracy curve; it is a strictly smaller factorisation and here trades
some accuracy for its smaller footprint.

Pure CPU, single-threaded-friendly, tiny dataset, **runs in a few seconds**. No
binaries are committed.

## Build & run

```
lazbuild TensorTrainLinear.lpi
./TensorTrainLinear      # or wherever lazbuild dropped the binary
```
