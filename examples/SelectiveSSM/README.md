# SelectiveSSM — input-dependent (Mamba/S6-style) selective state-space mixer

This example showcases **`TNNetSelectiveSSM`**, an **input-dependent**
("selective", Mamba / S6-style, Gu & Dao 2023, *"Mamba: Linear-Time Sequence
Modeling with Selective State Spaces"*, arXiv:2312.00752) diagonal state-space
sequence mixer, and contrasts it head-to-head with its **linear-time-invariant**
sibling **`TNNetDiagonalSSM`** on the canonical task that separates them: a
**content-addressed selective copy**.

Both layers operate over a `(SeqLen, 1, Depth)` sequence (`SizeY = 1`) with the
same depth-parallel causal sweep and output shape `== input shape`. The
difference is the recurrence:

| | per-channel decay `a` | input gain `b` | output gain `c` |
|---|---|---|---|
| `TNNetDiagonalSSM` (LTI) | **fixed** learned scalar | **fixed** | **fixed** |
| `TNNetSelectiveSSM` (selective) | `a_t = exp(-δ_t·exp(A_raw))` | `b_t = x_t·W_B` | `c_t = x_t·W_C` |

where the per-channel timestep `δ_t = softplus(x_t·W_d + b_d)` is itself a
function of the input. So the selective layer's gates **condition on content** at
every position:

```
δ_t   = softplus(x_t·W_d + b_d)        (per-channel positive step)
a_t   = exp(-δ_t · exp(A_raw))         (discretized decay in (0,1))
b_t   = x_t·W_B,   c_t = x_t·W_C       (input-dependent gains)
h_t   = a_t ⊙ h_{t-1} + (δ_t·b_t) ⊙ x_t
y_t   = c_t ⊙ h_t + e ⊙ x_t            (e = S4D feedthrough)
```

This is exactly the mechanism behind Mamba's selective-copy / induction-head
wins that an LTI SSM **provably cannot** do: an LTI recurrence applies the *same*
decay/gain at every step, so it can only build a smeared decaying average of the
stream, never a content-triggered latch.

## The task — content-addressed long-hold copy

A small vocabulary (`QUERY`, `MARKER`, `BLANK`, plus ordinary payload symbols).
Each sample plants exactly one `MARKER` at a **random** position `p`; the symbol
at `p+1` is the **payload**. The trailing query window is overwritten with
`QUERY` tokens, and the model must emit the payload **only in that window**:

```
layout:  [ .. distractors .. MARKER payload .. distractors .. QUERY QUERY ]
target:  [ BLANK ...........................................  pay   pay  ]
```

Two ingredients make this **provably LTI-defeating**:

1. **`p` is random**, so the payload sits at a data-dependent position among
   same-type distractors — a time-invariant gate cannot know *which* symbol to
   keep.
2. **Only the trailing query window is scored**, and the read-out is a *shallow
   per-token linear* map (no nonlinear MLP, no cross-position mixing), so the
   payload cannot be decoded locally near the marker — it must **survive in the
   recurrent state** all the way to the query.

## The bake-off

Two models, identical except for the single sequence-mixing layer, trained on
the same data with the same init and LR schedule:

| arm | mixer | mixing width |
|-----|-------|--------------|
| selective | `TNNetSelectiveSSM` | `d = 24` |
| LTI       | `TNNetDiagonalSSM`  | `d = 64` (**wider** — not starved of capacity) |

Shared front-end / read-out:
`Input → Embedding(d) → [MIXER] → PointwiseConvLinear(vocab) → PointwiseSoftMax`.
Training is mini-batch SGD with gradient clipping and a stepped LR decay.

The example **asserts its own headline** with self-check gates (`Halt(1)` on
failure), so a regression in `TNNetSelectiveSSM` turns the run red instead of
printing a quietly-wrong table.

## Running

```
cd examples/SelectiveSSM
fpc -Mobjfpc -Sh -O3 -Fu../../neural -dRelease -dAVX2 SelectiveSSM.lpr
./SelectiveSSM
```

(or open `SelectiveSSM.lpi` in Lazarus). Pure CPU, single thread, finishes in
about a minute — comfortably under the 5-minute budget, a few MB of RAM.

## Representative output

```
  model              params    init_CE   final_CE   recall_acc     secs
  SelectiveSSM         2136      1.944      0.143        0.796    46.76
  DiagonalSSM          1152      1.940      0.335        0.479     9.14

  SELECTIVITY clears the content-addressed copy (79.6% recall);
  the wider LTI sibling PLATEAUS (47.9% recall, chance 25.0%).

Self-check gates:
  [PASS] LTI not starved (LTI width 64 >= selective width 24)
  [PASS] selective recall >= 0.60  (got 0.796)
  [PASS] selective beats LTI by >= 0.12  (0.796 vs 0.479)
```

The selective mixer clears the content-addressed recall (~0.80, CE ~0.14) while
the **wider** LTI sibling plateaus near the chance/blend floor (~0.48, CE ~0.34),
never improving past the first few hundred epochs — a clean mechanistic
demonstration of selectivity. Exact numbers are seed-dependent; the *contrast*
is the point.

## Notes

`TNNetSelectiveSSM` stores six learnable tensors: the three `Depth×Depth`
projections `W_d / W_B / W_C`, and the `Depth`-long per-channel vectors `b_d`,
`A_raw` and `e`. Forward is a direct `O(SeqLen·Depth + SeqLen·Depth²)` causal
sweep; backward is backprop-through-time (a right-to-left `dL/dh` sweep) that
also scatters into every weight set and chains through `softplus(δ)` and the
`exp` decay. The recurrence is `O(1)`-per-step by nature and composes with an
incremental / KV-cache-style decode follow-up.

Coded by Claude (AI).
