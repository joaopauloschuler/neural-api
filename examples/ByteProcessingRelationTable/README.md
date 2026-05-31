# Byte Processing Relation Table

A tiny, readable peek inside the **symbolic byte engine**
(`TEasyLearnAndPredictClass`) that powers `TNNetByteProcessing`. Unlike the
gradient layers in the rest of the library, this engine *induces discrete
cause→effect rules* ("relations") that map an input byte pattern to an output
byte. This demo drives it exactly the way `TNNetByteProcessing` does internally
(`Predict()` then `newStateFound()`), then prints the rules with
`printRelationTable`.

## Problem encoding

Four 1-byte "codewords" (the kind a binarizing encoder emits) are each mapped
to a fixed 1-byte target:

| codeword   | target     |
|------------|------------|
| `00000001` | `11110000` |
| `00000010` | `00001111` |
| `00000100` | `10101010` |
| `00001000` | `01010101` |

The engine is trained for 300 passes, then we print its prediction per class
and dump the relation table.

## Reading a rule

```
B=0  (1 = A[0])  =>  fE[B] := 240  [ f=1  Vit=298  n=299 ]
└┬─┘ └────┬────┘     └────┬─────┘    └┬┘  └───┬───┘  └─┬─┘
 │     condition       THEN emit    conf.  rule wins  samples
out    on input        output byte  (correct/  this    seen
byte   byte A[0]       240=11110000  total)  prediction
```

- `B=0` — the output byte position (a single byte here).
- `A[0]` is the action byte, `S[0]` the state byte. The layer feeds the same
  bits as both, so each rule shows up in `A[0]` and `S[0]` twins.
- `f` is the rule's confidence; `Vit` how often it won the prediction; `n` the
  sample count.

A small neuron-group budget (16) and `FGeneralize := False` keep the table
tiny, so the four winning rules — one per class, `f=1`, `Vit≈298` — are easy to
read.

## What to notice

With a **distinct codeword per class**, every rule reaches `f=1.0`: the engine
finds one crisp logical rule per class. If two classes were forced to **share a
codeword** but demanded different targets, the contended rule could never reach
certainty — it would get stuck near `f=0.5`, the symbolic signature of an
unresolvable collision (and exactly the kind of error a binarizing encoder
learns to avoid when trained through `TNNetByteProcessing`'s straight-through
estimator).

## Build and run

```
cd examples/ByteProcessingRelationTable
lazbuild ByteProcessingRelationTable.lpi
../../bin/x86_64-linux/bin/ByteProcessingRelationTable
```

Finishes instantly on a single CPU.
