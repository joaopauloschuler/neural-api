# CumSumPositionEncoding

Demonstration of how `TNNetCumSum` can manufacture a **parameter-free
linear position feature** from a constant input, in two parts:

- **Part 1 (forward-only demo)** — the original landing: three forward
  passes proving the CumSum-of-constant ramp math.
- **Part 2 (train-time bake-off)** — the follow-up: actually *train* a
  tiny permutation-invariant model on a position-dependent task, once
  **without** the position feature and once **with** the CumSum ramp
  concatenated, and chart the loss/accuracy delta. The original
  forward-only demo is retained and runs first.

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

## Part 1 — what the forward demo shows

Three quick forward passes through a 2-layer network
(`TNNetInput -> TNNetCumSum`):

1. **Linear ramp** — all-ones input produces the position index `+1`.
2. **Arbitrary prefix-sum** — sanity-check that the math is the
   standard cumulative sum.
3. **Per-row independence** — for a `(1, Rows, Depth)` volume, each
   row accumulates only its own depth column; rows do not bleed into
   each other.

No training is performed in Part 1; it runs in well under a second.

## Part 2 — train-time bake-off (does the feature actually help?)

**Task "find the marker".** A length-8 sequence of one-hot tokens over a
4-symbol vocab. Exactly one position holds the special MARKER token
(vocab index 0); the rest are random non-marker tokens. The target class
is the **position index** of the marker (8 classes, chance = 1/8 = 12.5%).

**The model is permutation-invariant on purpose:**

```
Input(8, 1, 5)
  -> TNNetPointwiseConvReLU(48)   // SHARED transform, same weights per token
  -> TNNetMaxChannel              // order-agnostic pool over positions
  -> TNNetFullConnectLinear(8)
  -> TNNetSoftMax
```

Because the per-token transform shares weights across positions and the
pool is order-agnostic, the network has **no way to tell which slot a
token sits in from token identity alone** — it must read the position out
of each token's own feature vector.

The input's 5th channel (index `cVocab`) is the only difference between
the two arms:

| arm    | extra channel content | meaning                          |
|--------|------------------------|----------------------------------|
| NoPos  | constant `1.0`         | positions invisible              |
| CumSum | CumSum ramp (`pos+1`)  | each slot stamped with its index |

Everything else is identical: same architecture, same parameter count
(the constant pad keeps shapes equal), same `RandSeed`, same 300 epochs of
mini-batch SGD, same data (the marker positions and distractor tokens come
from the same RNG stream). A small `L2Decay(0.001)` keeps the SoftMax
logits from running away so the reported cross-entropy reflects how well
each arm *solves* the task rather than overconfidence on its mistakes.

### Built-in correctness gate (`Halt(1)` on failure)

- CumSum test loss is at least `0.20` below the NoPos loss,
- CumSum test accuracy `>= 70%` (it solves the task),
- NoPos test accuracy `<= chance + 10%` (positions truly invisible).

## Build & run

```
fpc -O3 -Mobjfpc -Sh -Fu../../neural -Fu../../neural/pas-core-math \
    -dRelease CumSumPositionEncoding.lpr
./CumSumPositionEncoding
```

(or `lazbuild CumSumPositionEncoding.lpi` then run the binary under
`../../bin/x86_64-linux/bin/`). The whole program — both parts — finishes
in well under half a minute on one CPU core.

## Expected output sketch — Part 1

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

## Sample output — Part 2 (representative run)

```
================================================================
PART 2: train-time bake-off on a POSITION-DEPENDENT task.
================================================================
Task "find the marker": SeqLen=8, vocab=4, 8 classes (chance=0.125).
Net: Input(8,1,5)->PointwiseConvReLU(48)->MaxChannel->FullConnectLinear(8)->SoftMax  (shared per-token + order-agnostic pool).
Mini-batch SGD  batch=20  LR=0.020  mom=0.90  epochs=300  seed=424242.

=== Bake-off results (held-out test set) ===
arm                params    TEST loss    TEST acc
NoPos (constant)      624      2.0799     13.00%
CumSum (ramp)         624      0.7997     89.13%
chance accuracy (1/SeqLen) = 12.50%
loss delta (NoPos - CumSum) = 1.2802   (positive => CumSum wins)

=== Correctness gate ===
[PASS] CumSum loss 0.7997 is >= 0.20 below NoPos loss 2.0799.
[PASS] CumSum test acc = 89.13% (must be >= 70%): solves the task.
[PASS] NoPos test acc = 13.00% (must be <= chance+10% = 22.50%): positions invisible.

=> BAKE-OFF CHECKS PASS: the CumSum position feature measurably helps.
```

The NoPos arm is pinned at chance (it literally cannot see position); the
SAME network, given the parameter-free CumSum ramp, jumps to ~89%
accuracy and cuts its loss by ~1.3 nats. That is the whole point: a
position-dependent task is unsolvable for an order-agnostic model until
you hand it a position signal — and `TNNetCumSum`-of-a-constant is a free
one.

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
