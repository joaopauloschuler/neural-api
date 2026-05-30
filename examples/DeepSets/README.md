# DeepSets — permutation-invariant set learning

This example reproduces the **Deep Sets** recipe (Zaheer et al. 2017,
*"Deep Sets"*) on a tiny pure-CPU target: learn to regress the **MAX** of a
fixed-size bag of `N` scalars. The headline is the architectural
**invariant** — once trained, the network's output is **unchanged** when the
`N` inputs are randomly permuted, yet **changes** when an element's value is
edited.

## The idea

A function defined on a *set* must not depend on the order of its elements:
`f({a,b,c}) = f({c,a,b})`. Zaheer et al. prove that any permutation-invariant
set function can be written as

```
f(X) = rho( POOL_{x in X} phi(x) )
```

where `phi` is a **shared** per-element encoder, `POOL` is a **symmetric**
reduction (sum / mean / max), and `rho` is a head applied to the pooled
summary. The invariance is **structural**: it holds for *every* weight setting,
before and after training, because a symmetric pool is by definition
order-agnostic.

## Architecture (existing layers only)

```
Input(N, 1, 1)                  -- N set elements laid along the X axis
TNNetPointwiseConvReLU(H)       -- shared phi: featuresize-1 conv...
TNNetConvolutionLinear(H,1,0,1) --   ...identical weights on every element
TNNetMaxChannel                 -- symmetric pool: (N,1,H) -> (1,1,H)
TNNetFullConnectReLU(H)         -- rho head
TNNetFullConnectLinear(1)       -- scalar prediction
```

A pointwise (featuresize-1) convolution is the natural "shared MLP over
elements": the same `H`-wide filter bank slides over the X axis, so element `k`
is encoded by exactly the same weights as element `0`. `TNNetMaxChannel` then
collapses the X axis to one value per channel. Shared `phi` + symmetric pool =
permutation invariance by construction.

## Why a plain flatten -> dense net fails

A flatten -> dense network assigns a **distinct** weight to "the element in
slot 0", "the element in slot 1", and so on. Reordering the inputs lands
different values on different weights, so the output moves — it is *not*
invariant. It also **hard-codes `N`**: a dense layer over `N` flattened inputs
literally cannot accept a bag of a different size. The Deep Sets net has
neither problem: the shared encoder is per-element and the pool accepts any
X width.

## Why not self-attention

Self-attention is also permutation **equivariant** (permuting the inputs
permutes the outputs identically) and, with a final symmetric pool, permutation
**invariant** — so it *could* solve this task too. But it pays `O(N^2)` for the
pairwise attention scores plus a stack of query/key/value projections. Deep
Sets buys the same invariance for `O(N)` with a single shared encoder and one
pooling op: it is the cheapest member of the permutation-invariant family, and
all this demo needs.

## Feasibility note: `TNNetMaxChannel` vs `TNNetAvgChannel` on `(N,1,F)`

Settled with a forward/backward probe before building the demo. For an
`(N,1,F)` input, **both** channel pools set `FPoolSize := SizeX = N` and
`stride = N`, reducing `(N,1,F) -> (1,1,F)` (one number per channel):

* **`TNNetMaxChannel`** returns the exact per-channel **MAX** over the `N`
  elements (verified: a `(5,1,2)` bag of `1..5` / `10..50` → `5` / `50`).
* **`TNNetAvgChannel`** returns `sum / (FPoolSize*FPoolSize) = sum / N^2`, i.e.
  a **scaled mean** (`1/N^2`, *not* `1/N`), because the avg-pool divides by
  `PoolSize^2` while only `N` (`= PoolSize * 1`) cells are non-empty along
  `Y = 1` (verified: the same bag → `0.6` / `6.0`, i.e. `15/25` / `150/25`).
  It is still perfectly symmetric (hence still permutation invariant); only the
  scale differs, which a linear `rho` head absorbs.

This demo pairs **MAX-pool with a MAX target** so the pool computes *exactly*
the symmetric statistic being regressed — the cleanest, fastest-learning
pairing, with mathematically exact invariance (`max|dy| = 0`).

## What the demo does

1. Trains the net (manual mini-batch SGD, ~60 epochs, a few seconds) to regress
   the MAX of `N = 5` scalars in `[-1, 1]`.
2. **Headline 1 — permutation invariance:** shuffles one bag 200 times and
   reports `max |dy|` (≈ 0).
3. **Headline 2 — value sensitivity:** edits one element above the whole bag
   and reports the resulting output change (large).
4. **Stretch — set-size generalization:** copies the trained weights into a net
   built for an **unseen** larger bag (`N = 8`) via `CopyWeights` and reports
   the RMSE. This works precisely because no trainable layer depends on `N`; a
   flatten -> dense baseline could not even be fed the `N = 8` bag.

It ends with a **self-checking PASS/FAIL gate** (`Halt(1)` on failure):
invariant to permutation, sensitive to value, and it actually learned MAX.
Pure CPU, single-threaded, deterministic (`RandSeed := 424242`).

## Sample output

```
Trained. RMSE on fresh N=5 bags: 0.007602

HEADLINE 1 - PERMUTATION INVARIANCE
  baseline output           : 0.82963985
  max |dy| over 200 shuffles : 0.00E+000  (tol 1.0E-005)
HEADLINE 2 - VALUE SENSITIVITY
  |dy| after editing one element: 1.824569  (tol 1.0E-003)

STRETCH - SET-SIZE GENERALIZATION (weights trained ONLY on N=5)
  RMSE on N=5 (train size)   : 0.012304
  RMSE on N=8 (UNSEEN size)  : 0.004120

GATE: PASS - invariant to permutation, sensitive to value, and it learned MAX.
```

## Build and run

```bash
cd examples/DeepSets
lazbuild DeepSets.lpi
../../bin/x86_64-linux/bin/DeepSets
```

Or directly with `fpc` (as the test suite builds examples):

```bash
fpc -B -Fu../../neural -Mobjfpc -Sh -O2 DeepSets.lpr && ./DeepSets
```
