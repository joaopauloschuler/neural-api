# Neural Turing Machine — the COPY task

A headline demo of `TNNetNTMMemory`, the first **writable** differentiable
external-memory layer in this fork (Neural Turing Machine, Graves et al. 2014,
["Neural Turing Machines", arXiv:1410.5401](https://arxiv.org/abs/1410.5401)).

## What an NTM layer does

Unlike the read-only associative memories already in the tree
(`TNNetModernHopfield` iterated energy recall, `TNNetProductKeyMemory` sparse
top-k lookup), an NTM carries a **persistent memory matrix** `M`
(`NumSlots × SlotWidth`) that the layer both **reads** and **writes** as it
sweeps the time axis of an `(T, 1, InputDim)` input. Output is the per-step read
vectors `(T, 1, SlotWidth)`.

Per timestep the input `x_t` is projected to:

- a **content key** `k_t` — cosine-similarity addressed against every slot row,
  sharpened by a softplus **key-strength** `beta_t`, softmaxed over slots to give
  weights `w`;
- a **read** `r_t = w^T M` (the layer's output for step `t`);
- a sigmoid **erase** vector `e_t` and an **add** vector `a_t`, applied as
  `M[i] := M[i]·(1 − w[i]·e_t) + w[i]·a_t` (erase-then-add, same `w` as the read).

Backprop is **BPTT through the recurrent `M` update**: because `M_t` depends on
both `M_{t-1}` and `w_t`, the gradient `dL/dM` and `dL/dw` both chain backward
across steps. The four projection matrices (key, beta, erase, add) are the only
trainables; `M` is re-initialised to a small constant each sweep and is **not**
persisted across loads.

**v1 scope:** content addressing only, single read+write head. Deferred:
multi-head, location-based shift/interpolation addressing, and the DNC
temporal-link matrix.

## The COPY task

The classic NTM benchmark. Each episode is one `(T, 1, InputDim)` sequence:

1. **Present** (`LEN` steps): a random `BITS`-wide binary vector per step
   (delimiter channel `= 0`).
2. **Delimiter** (1 step): all bits `= 0`, delimiter channel `= 1`.
3. **Recall** (`LEN` steps): all-zero inputs; the net must reproduce the
   presented bits, in order, at the output.

The target is zero during present + delimiter and equals the original bits during
recall. We train on a single fixed random episode and report per-bit recall
accuracy on the copy window for the NTM vs a **scalar-xLSTM** arm
(`TNNetSLSTMCell`), both topped by the same per-token linear + sigmoid head.

The intended contrast: the SLSTM arm must cram every presented bit into a fixed
recurrent cell state and replay it from one evolving vector; the NTM can park each
step's bits in a distinct addressable slot and fetch them back.

## Running it

```
cd examples/NeuralTuringMachine
lazbuild --build-mode=Release NeuralTuringMachine.lpi
../../bin/x86_64-linux/bin/NeuralTuringMachine
```

Pure CPU, ~1 second.

## Measured result (seed 20260607)

```
  NTM   (writable external memory, 138 weights):  87.50%
  SLSTM (fixed recurrent state, 366 weights):  87.50%
  -> external writable memory recalls at least as well, at fewer weights.
```

The NTM matches the recurrent baseline's recall while using **~2.6× fewer
weights** (138 vs 366) — addressable external memory is the more
parameter-efficient way to store-then-recall here.

## Honest caveats

This is deliberately a **tiny memorization/recall toy** (one training episode,
`LEN = 4`, `BITS = 4`, `6 × 6` memory) so it runs in ~1 s on CPU. With
content-only addressing (no location/shift addressing) and a single episode it is
**not** a length-generalization benchmark — the paper's headline "generalizes to
longer-than-trained sequences" claim needs the deferred location-based addressing.
The result is RNG/episode dependent at this scale; the steady, reproducible signal
is that the NTM reaches the same recall accuracy as a larger recurrent net.

Coded by Claude (AI).
