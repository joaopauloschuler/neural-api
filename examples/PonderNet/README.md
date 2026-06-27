# PonderNet — adaptive computation time with learned probabilistic halting

Demonstrates **`TNNet.AddPonderNetBlock`** and the **`TNNetPonderCostLoss`** head,
a PonderNet-style adaptive-compute block (Banino, Balaguer & Blundell 2021,
"PonderNet: Learning to Ponder", [arXiv:2107.05407](https://arxiv.org/abs/2107.05407)).
PonderNet learns **how long to think** per input: a weight-tied step function is
applied up to `MaxSteps` times, and a tiny halting head decides — probabilistically
— when to stop, so the model spends extra compute only where the task forces it.

## The block

A single weight-tied step function `f` is unrolled `MaxSteps` times. At each step a
shared halting head emits `λ_n ∈ (0,1)`, giving the **geometric halting
distribution**

```
p_n = λ_n · ∏_{k<n} (1 − λ_k)
```

and the block output is the **smooth, `p_n`-weighted sum** of the per-step outputs
(no hard argmax). `TNNetPonderCostLoss` adds a `KL(p ‖ geometric(prior))`
regularizer that prefers halting **early**, so the model only ponders longer when
the input is harder.

`AddPonderNetBlock(nil, cMaxSteps, Halting, cHidden, cPrior)` builds the
weight-tied core, a shared halting head, and the running `p_n` accumulator, and
returns the `p_n`-weighted block output. It hands back `Halting`, a
`(1, 1, MaxSteps)` layer carrying the halting distribution, already wired into the
graph.

Because the weights are tied across steps, the parameter count is **independent of
`MaxSteps`** (the program prints this).

## The task — parity of a variable-length bit string

A textbook "harder = needs more sequential computation" problem. Each sample has a
random number of **active leading bits** `L ∈ [1..cMaxLen]` (encoded as `+1`/`−1`,
one channel per position; the rest zeroed = inactive). The label is the **parity
(XOR)** of the active bits. Difficulty is exactly `L`: a longer active prefix needs
more iterative XOR steps to resolve, so an adaptive-compute model should ponder
**longer** as `L` grows, while a fixed-depth net must pay the worst case on every
input.

## Architecture

```
Input(1,1,cInDim)
 -> AddPonderNetBlock(nil, MaxSteps, Halting, hidden, prior)   weight-tied f x MaxSteps, smooth p_n output
 == parity head (on the p_n-weighted block output) ==========================
 -> PointwiseConvLinear(2)
 -> SoftMax                                                    2-way parity softmax
 == ponder-cost head (on the halting branch) ================================
 -> PonderCostLoss(prior)   on `Halting`                       KL ponder-cost gradient passthrough
 == combined output =========================================================
 -> DeepConcat([ParityHead, CostHead])                         [ parity softmax (2) | ponder-cost passthrough (MaxSteps) ]
```

Both losses are trained in **one backward pass**: the two heads are
`TNNetDeepConcat`-ed into a single output of width `2 + MaxSteps`. The parity head
uses the framework-default cross-entropy; the halting branch uses
`TNNetPonderCostLoss` (an identity passthrough whose gradient is rewritten to the
KL ponder-cost). Consuming `Halting` through the cost head also keeps the halting
branch on the backward path (no dangling branch). The per-token projections are
`TNNetPointwiseConvLinear` (not `TNNetFullConnect`).

Dimensions: `cMaxLen=6`, `cMaxSteps=6`, `cHidden=24`, `cPrior=0.5`; trained for
`cTrainSteps=24000` per-sample SGD steps at `lr=0.001`, momentum `0.9`. Evaluation
draws `cEvalPer=500` samples per difficulty bucket.

The expected number of ponder steps for a forward pass is
`E[n] = Σ_n (n+1)·p_n` (1-based), read directly off the `Halting` layer's output.

> **Inference note (from the source):** this build always unrolls `MaxSteps`
> applications (static tensor shapes) and returns the `p_n`-weighted expectation;
> `E[n]` is the adaptive-depth signal. A true threshold-on-cumulative-`p_n`
> early-exit would need dynamic shapes the unrolled-graph API does not support, so
> it is intentionally not done — the expected **output** is identical, only compute
> is not saved.

## The headline

The program asserts that the **expected number of ponder steps rises with
difficulty** `L`. It prints a per-difficulty table of parity accuracy and mean
expected ponder steps, then a self-check:

```
=== PonderNet: adaptive computation time on variable-length parity ===
...
difficulty L | parity acc | mean expected ponder steps E[n]
-------------+------------+-------------------------------
...
E[n] at L=1 (easiest) = ...   E[n] at L=6 (hardest) = ...
OK: expected ponder steps RISE with difficulty (adaptive computation time).
```

The run is red (`Halt(1)`) unless `E[n]` is monotone non-decreasing in `L` (within
a `0.02` tolerance) **and** rises by more than `0.1` from the easiest to the
hardest bucket. Exact numbers are seed-dependent; the *rise* is the point.

## Running

```
cd examples/PonderNet
fpc -O3 -Mobjfpc -Sh -Fu../../neural PonderNet.lpr
./PonderNet
```

(or open `PonderNet.lpi` in Lazarus / `lazbuild PonderNet.lpi`, which writes the
binary to `../../bin/$(TargetCPU)-$(TargetOS)/bin/PonderNet`). Pure CPU, tiny
dimensions; finishes well under three minutes on two cores.

Coded by Claude (AI).
