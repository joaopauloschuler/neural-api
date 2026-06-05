# ByteRuleInduction

A small **discovery experiment** (not a demo) on the symbolic byte engine
(`TEasyLearnAndPredictClass`) that powers `TNNetByteProcessing`.

## The question

The byte engine is not a gradient network — it induces discrete, *human-readable*
cause→effect relations (see `printRelationTable`). That lets us ask something
most ML frameworks can't answer cleanly:

> When we train the engine on only **half** of the 256 single-byte inputs, does
> it get the **held-out** half right — i.e. does it *induce a rule*, or just
> *memorize* the examples it saw?

## Setup

- 256 single-byte inputs, deterministic 50/50 train/test split (fixed `RandSeed`).
- Train on the train half only, then measure accuracy on the train half (did it
  fit?) and the held-out half (did it generalize?).
- Chance level for guessing a held-out byte is 1/256 ≈ **0.4%**, so any sizable
  test accuracy is real generalization.

Three independent variables:

| Variable | Levels |
|---|---|
| **Encoding** | `BYTE` (input = 1 action byte) vs `BITS` (input unpacked into 8 bit-positions) |
| **Function** | `XOR $5A` and `NOT` (bit-local) vs `INC +1` (carry-coupled) |
| **FGeneralize** | engine flag: `False` (memorize seen patterns) vs `True` (clone neurons onto broader patterns) |

## Findings

Representative run (`test acc` is the held-out half):

```
enc    function                   FGeneralize   train acc   test acc
BYTE   XOR $5A (bit-local)        False            100.0%       0.0%
BYTE   ... (all BYTE rows)        ...              100.0%       0.0%
BITS   XOR $5A (bit-local)        False            100.0%      34.4%
BITS   NOT     (bit-local)        False             89.8%      39.8%
BITS   INC +1  (carry-coupled)    True             100.0%      46.1%
```

1. **Encoding is decisive.** Under `BYTE` the engine forms whole-byte equality
   relations (`A[0] = 82 => 8`) — a **lookup table**: perfect train fit, **0%
   held-out** for every function. Pure memorization. Under `BITS`, exposing the
   individual bits unlocks generalization far above chance (~20–45%).

2. **The bit-locality hypothesis fails.** We expected bit-local `XOR`/`NOT` to
   generalize and carry-coupled `INC` to collapse. It doesn't: `INC` generalizes
   about as well as the others. The relation table shows why — the engine's rule
   grammar is far richer than per-bit equality. It mixes per-bit tests (`A[3]=0`)
   with **inter-position relations** (`A[i] < A[j]`) and **arithmetic effects**
   (`fE[B] := inc S[B]`, `dec S[B]`, `S[i] xor S[j]`). It has a native `inc`
   operator, so increment is directly expressible. The flip side: those extra
   degrees of freedom let it bolt spurious conditions onto correct rules, so it
   overfits and held-out accuracy plateaus well below 100%.

## The lesson for a discovery library

The symbolic engine **can** induce genuinely generalizing rules, but only when
the **input encoding exposes the relevant structure**, and the rules it finds
are **richer and messier** than the minimal human description of the target
function. The value of this engine for science is precisely that you can *read*
the rules and see this — which is what the printed relation table is for.

## Run it

```bash
cd examples/ByteRuleInduction
fpc -Mobjfpc -Sh -O2 -Fu../../neural ByteRuleInduction.lpr
./ByteRuleInduction
```

Reproducible (fixed seeds), single CPU, no external data. Finishes in well under
a second.
