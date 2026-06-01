# SoftCapping Logit-Stability Micro-Experiment

A tiny experiment showing that a single `TNNetSoftCapping(c)` layer in front of
the final SoftMax tames pre-softmax logit blow-ups under a deliberately
aggressive learning rate.

## What `TNNetSoftCapping` does

`TNNetSoftCapping(c)` maps every pre-softmax logit `x` through

```
y = c * tanh(x / c)
```

which smoothly squashes the logit into the open interval `(-c, +c)`. This is the
Gemma-style "soft capping" trick: it stops classification logits from running
away, where a large logit makes `exp(logit)` inside the SoftMax overflow float32
and the forward/backward pass can turn into Inf/NaN. The layer has **no trainable
parameters** and its derivative is `dy/dx = 1 - tanh(x/c)^2`.

## The experiment

The **same** tiny classifier is trained **twice** on the **same** small synthetic
2D Gaussian-blob task, with the **same** fixed `RandSeed` (424242), the same data,
the same init order and the same **deliberately-aggressive** learning rate. The
two arms differ in exactly one layer:

```
NoCap  arm : Input(2) -> Head(24) -> ReLU -> Head(4) ->                    SoftMax
Capped arm : Input(2) -> Head(24) -> ReLU -> Head(4) -> SoftCapping(8.0) -> SoftMax
```

The task is a 4-class problem where each class is a noisy Gaussian blob around one
of the four corners `(±1, ±1)`. With a sane learning rate either net solves it
trivially; the point here is what happens at `LR = 5.0` (about 50x a sane `0.1`,
with momentum 0.9).

## Counting blow-ups

After **every weight update** the program inspects the logits feeding the SoftMax.
A **SoftMax-overflow event** is defined precisely as:

- a logit whose magnitude exceeds **88.7** — the float32 threshold above which
  `exp(logit)` is no longer finite, i.e. a logit the SoftMax cannot exponentiate
  without overflowing — **or**
- an already non-finite logit/weight (checked with `IsNan` / `IsInfinite` from the
  `Math` unit, as a hard backstop).

Any epoch containing such an event is counted as a **blow-up epoch**.

> **Why the exp-overflow threshold rather than waiting for a literal `Inf`?**
> The framework's SoftMax is internally hardened (it subtracts the row maximum and
> clamps its input range) and the cross-entropy gradient is bounded, so in a few
> dozen epochs the raw weights rarely reach a literal float32 `Inf`. The
> *physically meaningful* blow-up is the logit growing past the point where a naive
> `exp(logit)` overflows — and the bounded SoftMax is exactly the band-aid that
> soft-capping makes unnecessary by fixing the cause. The capped arm
> (`|logit| < 8`) can never trip this threshold; the uncapped arm trips it
> immediately and stays tripped.

## What it reports

A summary table per arm — aggressive LR, blow-up epoch count, epoch of first
onset, the largest `|logit|` reached, final test loss and final test accuracy —
followed by a self-checking **PASS/FAIL gate**. The gate requires all four of:

1. the **NoCap** arm actually blew up (`> 0` blow-up epochs),
2. the **Capped** arm has **strictly fewer** blow-up epochs than NoCap,
3. the **Capped** arm stayed **completely clean** (`0` blow-up epochs), and
4. the **Capped** arm is still a useful classifier (test accuracy `≥ 85%`).

If the gate fails the program exits with a non-zero status (`Halt(1)`). The whole
experiment runs single-threaded, pure CPU, with no external data, in ~2 seconds.

## Example run

```
=== Numerical-stability summary ===
arm     LR     blow-up epochs  first onset   max |logit|   final loss   final acc
NoCap    5.00        60 /60               1      11849.07      22.5000     25.00%
Capped   5.00         0 /60              --          8.00       0.0000    100.00%

Blow-up epochs: NoCap=60  Capped=0  (lower is better).
Largest |logit| reached: NoCap=11849.07  Capped=8.00  (cap=8.0, exp-overflow=88.7).

GATE: PASS -- SoftCapping tamed the logit blow-up.
```

The uncapped head's logits explode to ~`11849` (far past the exp-overflow point)
on **every** epoch and its accuracy collapses to chance (25%). The capped head
pins every logit to exactly `8.0`, never trips a single overflow event, and trains
to **100%** accuracy at the very same aggressive learning rate.

## Build & run

From this directory:

```
fpc -O3 -Mobjfpc -Sh -Fu../../neural -Fu../../neural/pas-core-math -dRelease SoftCappingStability.lpr
./SoftCappingStability
```

Or with Lazarus:

```
cd examples/SoftCappingStability
lazbuild SoftCappingStability.lpi
../../bin/x86_64-linux/bin/SoftCappingStability
```
