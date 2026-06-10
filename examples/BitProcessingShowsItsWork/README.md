# Bit Processing Shows Its Work

A small hybrid network that learns the pair of comparison relations
**y₁ = (a > b)** and **y₂ = (a < b)** on two continuous inputs and — unlike an
ordinary dense net — can **print the human-readable rule it induced**.

The star is `TNNetBitProcessing` (in `neural/neuralnetwork.pas`). It
affine-quantizes *each* input scalar to one whole byte over `[-25.6, +25.6]`
(≈0.2 step), runs a **symbolic byte engine** (`TEasyLearnAndPredictClass`) whose
rule grammar's **conditions are comparisons** (`S[i] < S[j]`, …), then
affine-decodes one float back per byte. The engine is trained online during
backprop, and a straight-through estimator lets the layer sit inside a
gradient-trained `TNNet`. Because the affine encoding is **monotone**,
`encByte(a) > encByte(b) ⇔ a > b` holds identically across the whole range — so
the rule the engine induces is genuinely **scale-free**.

## Why comparison (and not, say, `a − b`)

A linear relation like `a − b` is trivial for a ReLU net: ReLU is
piecewise-linear, so the net simply *is* the rule and extrapolates it for free —
there is no gap to win. Comparison is more honest, but the honesty cuts both
ways: the decision boundary `a = b` is **also linearly separable**, so a dense
net is not at a disadvantage either.

**So this example is not a claim that the symbolic model is more accurate — it
is not.** Both models solve the task and tie on accuracy. The point is
**interpretability**: only the symbolic model can *print the exact, scale-free
rule it is using* (see [Reading the rule](#reading-the-rule)).

## Problem encoding

Two continuous inputs `a, b`; the targets are the two booleans `(a > b)`,
`(a < b)` as `1.0 / 0.0`.

| split        | distribution of a, b | inside ±25.6? |
|--------------|----------------------|---------------|
| **train**    | `[0, 10]`            | yes           |
| **extrapol.**| `[10, 20]` (unseen)  | yes (no clip) |

A near-boundary **hard band** (`|a − b| ≤ 0.5`) is also evaluated in each box —
that is where a blurred or drifted decision boundary actually shows up.

## The two models

1. **Symbolic** (shows its work):
   `TNNetInput(2)` → `TNNetBitProcessing(0, 16, 40, 0, -25.6, 25.6)` → `TNNetFullConnectLinear(2)`.
   The byte engine itself supplies the nonlinearity (the comparison condition);
   the linear readout only scales its high/low branch output into the two
   booleans.
2. **Dense baseline** (same-ish size, nonlinear):
   `TNNetInput(2)` → `TNNetFullConnectReLU(8)` → `TNNetFullConnectReLU(8)` → `TNNetFullConnectLinear(2)`.

Both are trained with the same manual, single-thread, deterministic SGD loop
(fixed `RandSeed`, batch 16, 40 epochs), and scored by **classification
accuracy** (arg-max over the two outputs).

## Headline result

```
1) IN-RANGE accuracy (a,b in [0,10])
   symbolic (TNNetBitProcessing)  acc =  99.25%
   dense baseline (ReLU)          acc =  99.25%

2) EXTRAPOLATION accuracy (a,b in [10,20])
   symbolic (TNNetBitProcessing)  acc =  97.75%
   dense baseline (ReLU)          acc =  98.50%

3) HARD-BAND accuracy (|a-b| <= 0.5, right on the a=b boundary)
   symbolic  in-range =  88.50%   extrap =  79.50%
   dense     in-range =  92.75%   extrap =  76.25%
```

The accuracy is a **tie** — exactly what you should expect on a linearly
separable task. The hard-band row is *not* a symbolic win: the byte quantizer
has a ~0.2 step, so pairs closer than that collapse to equal bytes and become
**unresolvable by construction** — a built-in floor at the `a ≈ b` tie line
that the dense net's smooth ramp does not have (which is why the dense net is in
fact a little better in-range there). The symbolic model's edge, if any, is only
in the extrapolation band.

**The actual payoff is the next section: only the symbolic model can show its
work.**

## Reading the rule

The layer keeps its engine in a private field (`FByteLearning`), unreachable
from a separate program unit. So — exactly as the sibling examples
[ByteProcessingRelationTable](../ByteProcessingRelationTable/README.md) and
[ByteRuleInduction](../ByteRuleInduction/README.md) do — this demo drives an
**independent `TEasyLearnAndPredictClass` mirror**, configured identically to the
layer's engine (same 16 neuron groups / 40 searches, same affine byte encoding),
feeds it the comparison target in byte space (high byte when the relation holds,
low byte otherwise), and calls `printRelationTable`. What it prints is what the
layer's engine can and does induce — comparison **conditions** gating a
high/low output:

```
B=0  (A[B+1] < A[B]) ...  =>  fE[B] := 255  [ f=1  Vit=952  n=953 ]
└┬─┘ └──────┬───────┘      └─────┬──────┘     └┬┘  └──┬──┘
out      condition         THEN emit "high"   conf.  rule wins
byte     (a > b)           (the > channel)    (=1)   this prediction
```

- `A[0]` is the encoded `a`, `A[1]` the encoded `b`.
- Read the **winning rows** (`f=1`, `Vit>0`); rows with `n=0` are unused neuron
  slots.
- `B=0` (the “>” channel) fires its high output on the condition `(A[1] < A[0])`,
  i.e. `a > b`; `B=1` (the “<” channel) fires on `(A[0] < A[1])`, i.e. `a < b`.
  Several neuron groups **independently converge** on the same comparison —
  exactly the two relations, made explicit.

The dense baseline has **nothing comparable to print**: its knowledge is a tangle
of ReLU weights, not a readable comparison rule.

## Build and run

```
cd examples/BitProcessingShowsItsWork
lazbuild BitProcessingShowsItsWork.lpi
../../bin/x86_64-linux/bin/BitProcessingShowsItsWork
```

Deterministic, no external data, finishes in a few seconds on two cores.
