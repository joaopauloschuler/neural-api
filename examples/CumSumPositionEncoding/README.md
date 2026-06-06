# CumSumPositionEncoding

Forward-only demonstration of how `TNNetCumSum` can manufacture a
**parameter-free linear position feature** from a constant input.

## The trick

`TNNetCumSum` computes the prefix-sum along the depth (channel) axis:

```
Output[x, y, c] := sum_{k=0..c} Input[x, y, k]
```

If the input is constant `1.0` along depth, the output is just the
running position index plus one:

```
Input  = [1, 1, 1, 1, 1, 1, 1, 1]
Output = [1, 2, 3, 4, 5, 6, 7, 8]
```

That is a strictly increasing per-position signal that downstream
permutation-invariant layers can use as a "where am I in the sequence?"
cue — at zero parameter cost.

## What this example shows

Three quick forward passes through a 2-layer network
(`TNNetInput -> TNNetCumSum`):

1. **Linear ramp** — all-ones input produces the position index `+1`.
2. **Arbitrary prefix-sum** — sanity-check that the math is the
   standard cumulative sum.
3. **Per-row independence** — for a `(1, Rows, Depth)` volume, each
   row accumulates only its own depth column; rows do not bleed into
   each other.

No training is performed. The whole program runs in well under a
second.

## Build & run

```
lazbuild CumSumPositionEncoding.lpi
../../bin/x86_64-linux/bin/CumSumPositionEncoding
```

## Expected output sketch

```
--- Demo 1: constant-1 input  ->  linear position ramp ---
  input  = [  1.00,   1.00,   1.00, ...,   1.00]
  cumsum = [  1.00,   2.00,   3.00, ...,   8.00]
  OK: CumSum of all-ones is the position index + 1.

--- Demo 2: arbitrary input   ->  standard prefix-sum ---
  input  = [  1.00,   2.00,   3.00,   4.00,   5.00,   6.00]
  cumsum = [  1.00,   3.00,   6.00,  10.00,  15.00,  21.00]
  OK: prefix-sum matches the textbook result.

--- Demo 3: multi-row input   ->  each row sums independently ---
  row 0  in= 1.0,...   out=  1.0,  2.0,  3.0,  4.0,  5.0
  row 1  in= 2.0,...   out=  2.0,  4.0,  6.0,  8.0, 10.0
  row 2  in= 3.0,...   out=  3.0,  6.0,  9.0, 12.0, 15.0
  OK: each row accumulates only its own column.
```

## Putting it into a real model

To use this as a position encoding, concatenate a CumSum-of-constant
branch alongside the real feature channels:

```
Real    : TNNetInput(1, 1, Depth)
PosFeat : TNNetInput(1, 1, Depth, FillValue=1) -> TNNetCumSum
Merged  : TNNetConcat([Real, PosFeat])  -> downstream layers
```

The downstream model now sees a strictly-increasing position index in
the extra channels, with no trainable parameters added.
