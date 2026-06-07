# Bit Processing Shows Its Work

A small hybrid network that learns the clean arithmetic relation **y = a − b**,
**generalizes it far outside the box it was trained on**, and — unlike an
ordinary dense net — can **print the human-readable rule it induced**.

The star is `TNNetBitProcessing` (in `neural/neuralnetwork.pas`). It
affine-quantizes *each* input scalar to one whole byte over `[-25.6, +25.6]`
(≈0.2 step), runs a **symbolic byte engine** (`TEasyLearnAndPredictClass`) whose
rule grammar literally contains `A-B`, `A+B`, `AND` and comparisons, then
affine-decodes one float back per byte. `decode∘encode ≈ identity`, and a
straight-through estimator lets the layer sit inside a gradient-trained `TNNet`.
So the layer turns `(a, b)` into a discrete **affine code** that a tiny linear
readout combines — a pipeline that is genuinely **scale-free**, which is exactly
why it extrapolates a clean linear rule perfectly.

## Problem encoding

Two continuous inputs `a, b`; the target is `y = a - b`.

| split        | distribution of a, b | y = a − b | inside ±25.6? |
|--------------|----------------------|-----------|---------------|
| **train**    | `[0, 10]`            | `[-10,10]`| yes           |
| **extrapol.**| `[10, 20]` (unseen)  | `[-10,10]`| yes (no clip) |

The extrapolation box is the whole point: the symbolic **rule** is range-free
and must generalize, while a dense net fits the training box and bends away once
the inputs leave it.

## The two models

1. **Symbolic** (shows its work):
   `TNNetInput(2)` → `TNNetBitProcessing(0, 16, 40, 0)` → `TNNetFullConnectLinear(1)`.
2. **Dense baseline** (same-ish size, nonlinear):
   `TNNetInput(2)` → `TNNetFullConnect(8)` [tanh] → `TNNetFullConnect(8)` [tanh] → `TNNetFullConnectLinear(1)`.

Both are trained with the same manual, single-thread, deterministic SGD loop
(fixed `RandSeed`, batch 16, 40 epochs).

## Headline result

```
1) IN-RANGE error (a,b in [0,10])
   symbolic (TNNetBitProcessing)  RMSE =  0.0991
   dense baseline (tanh)          RMSE =  0.2317

2) EXTRAPOLATION error (a,b in [10,20])  -- the money shot
   symbolic (TNNetBitProcessing)  RMSE =  0.1140
   dense baseline (tanh)          RMSE =  0.5788
   -> symbolic extrapolates 5.1x better than the dense net.
```

Both fit the training box comparably; **out of the box the symbolic model barely
degrades (0.099 → 0.114) while the dense net's error jumps ~2.5×** (0.232 →
0.579). The symbolic edge is the affine-quantize + linear-readout pipeline being
exactly scale-free.

## Reading the rule

The layer keeps its engine in a private field (`FByteLearning`), unreachable
from a separate program unit. So — exactly as the sibling examples
[ByteProcessingRelationTable](../ByteProcessingRelationTable/README.md) and
[ByteRuleInduction](../ByteRuleInduction/README.md) do — this demo drives an
**independent `TEasyLearnAndPredictClass` mirror**, configured identically to the
layer's engine (same 16 neuron groups / 40 searches, same affine byte encoding),
feeds it the subtraction target in byte space, and calls `printRelationTable`.
What it prints is what the layer's engine can and does induce for `a − b`:

```
B=1  (A[B] < A[B-1]) ...  =>  fE[B] := (A[0] - A[1])  [ f=1  Vit=909  n=910 ]
└┬─┘ └──────┬───────┘      └────────┬────────┘         └┬┘  └──┬──┘
out      condition         THEN emit A[0]-A[1]        conf.  rule wins
byte     on inputs         (the csSub operation)      (=1)   this prediction
```

- `A[0]` is the encoded `a`, `A[1]` the encoded `b`.
- Read the **winning rows** (`f=1`, `Vit>0`); rows with `n=0` are unused neuron
  slots. Several neuron groups **independently converge** on the same effect,
  `fE[B] := (A[0] - A[1])` — the engine's native `csSub`.
- You will also see a near-equivalent alias such as `fE[B] := (S[0] mod S[1])`
  guarded by `(S[0] > S[1])`: on this training range it computes the same value,
  a faithful illustration (à la `ByteRuleInduction`) that the engine's grammar is
  *richer and messier* than the minimal human description — yet the dominant,
  highest-victory rule is exactly subtraction.

The dense baseline has **nothing comparable to print**: its knowledge is a tangle
of tanh weights, not a readable rule — which is also why it cannot extrapolate
the clean linear relation outside its training box.

## Build and run

```
cd examples/BitProcessingShowsItsWork
lazbuild BitProcessingShowsItsWork.lpi
../../bin/x86_64-linux/bin/BitProcessingShowsItsWork
```

Deterministic, no external data, finishes in well under a second on two cores.
