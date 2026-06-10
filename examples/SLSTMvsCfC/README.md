# sLSTM scalar cell (xLSTM) vs CfC vs DiagonalSSM

A one-screen **copy-with-state-reset** task that contrasts the new
`TNNetSLSTMCell` (the **scalar sLSTM** of xLSTM) against the closed-form
continuous-time `TNNetClosedFormContinuous` (CfC) cell and the linear
time-invariant `TNNetDiagonalSSM` diagonal state-space mixer, at a comparable
parameter budget.

Reference: Beck et al. 2024,
[*xLSTM: Extended Long Short-Term Memory*](https://arxiv.org/abs/2405.04517).

## What the scalar sLSTM is

The sLSTM is a classic LSTM-style multiplicative-gate recurrence, but with two
xLSTM twists: **EXPONENTIAL** input/forget gates instead of sigmoid, and a
running-max **stabilizer** so the unbounded exp gates never overflow. Per
timestep `t`, per channel `d` (each gate reads the whole input `x_t` through a
Depth×Depth projection plus the previous hidden `h_{t-1}` through a Depth×Depth
recurrent projection):

```
li_t = b_i + W_i·x_t + r_i·h_{t-1}          (log input gate)
lf_t = b_f + W_f·x_t + r_f·h_{t-1}          (log forget gate)
z_t  = tanh(b_z + W_z·x_t + r_z·h_{t-1})    (cell input)
o_t  = sigmoid(b_o + W_o·x_t + r_o·h_{t-1}) (output gate)

m_t  = max(lf_t + m_{t-1}, li_t)            (STABILIZER, stop-grad running max)
i'_t = exp(li_t - m_t)                      (renormalized input gate)
f'_t = exp(lf_t + m_{t-1} - m_t)            (renormalized forget gate)

c_t  = f'_t·c_{t-1} + i'_t·z_t              (cell state,  c_{-1}=0)
n_t  = f'_t·n_{t-1} + i'_t                  (normalizer,  n_{-1}=0)
h_t  = o_t·(c_t / n_t)                      (hidden / output, h_{-1}=0)
```

The exp forget gate can collapse the retained state to **~0 in a single step**,
so the cell can FORGET SHARPLY on command — the defining ingredient that a plain
sigmoid LSTM (forget gate bounded in `(0,1)`, decays only gradually) lacks. The
running-max `m_t` (treated as a stop-gradient constant in backward) is the paper's
key trick that makes exp gating numerically trainable.

This is genuinely distinct from the other recurrent/sequence mixers in tree —
all of which are **linear-state or decay-based** and have no multiplicative
input/forget/output gates:

| Layer | Mechanism | Forgetting |
| --- | --- | --- |
| `TNNetClosedFormContinuous` | closed-form liquid ODE recurrence | input-dependent convex `(1-gate)` leak |
| `TNNetDiagonalSSM` | linear time-invariant recurrence | fixed per-channel `sigmoid(a_raw)` decay |
| `TNNetSelectiveSSM` | Mamba/S6 selective state space | input-dependent decay |
| `TNNetRetention` | softmax-free attention with decay mask | fixed/learned `γ^(n-m)` |
| **`TNNetSLSTMCell`** | **exp-gated LSTM with stabilizer** | **multiplicative exp gate (sharp, on-command)** |

## The task

A sequence of symbols is shown. Symbols `1..cNumSym` are **cues**, symbol `0` is
filler, and symbol `cClear` is a **"clear memory" pulse**. The target at every
position is the **most recent cue seen since the last clear** (class `0` = none).
Cues are dense and clears are frequent, so the running state is constantly
overwritten and wiped. Accuracy is read at the **last** position (the longest
recall horizon).

Three models share the same `Embedding -> (mixer) -> per-position readout ->
softmax` skeleton; only the mixer differs.

## Library API

```pascal
// Leaf layer over a (SeqLen, 1, Depth) sequence (SizeY must be 1).
// Output shape == input shape. Depth is inferred from the previous layer.
NN.AddLayer(TNNetSLSTMCell.Create());

// Or the pre-norm residual builder: y = x + sLSTM(RMSNorm(x)).
NN.AddSLSTM();
```

Storage is twelve learnable tensors: `W_z/W_i/W_f/W_o` and `r_z/r_i/r_f/r_o`
(Depth×Depth projections) plus `b_z/b_i/b_f/b_o` (Depth-long biases, with `b_f`
initialised to `+1` so the cell starts as a near-pass-through accumulator).
Nothing has to be passed to `Create`; the layer round-trips through save/load with
no constructor arguments.

The forward pass is the explicit per-timestep recurrence; the backward pass is
backprop-through-time (a right-to-left sweep carrying `dL/dc`, `dL/dn`, `dL/dh`)
that differentiates the stabilized exp gates with `m_t` treated as a
stop-gradient constant. The gradients are pinned by finite-difference checks in
`tests/TestNeuralNumerical.pas` (`TestSLSTMCellInputGradientCheck`,
`TestSLSTMCellWeightGradientCheck`, max error ≈ 2·10⁻³) plus a save/load
round-trip (`TestSLSTMCellSerializationRoundTrip`) and the builder test
(`TestAddSLSTMBuilder`).

## Build & run

```bash
cd examples/SLSTMvsCfC
lazbuild SLSTMvsCfC.lpi
../../bin/x86_64-linux/bin/SLSTMvsCfC
```

Pure CPU, single-threaded, no external dataset; finishes in well under a minute
with a small memory footprint.

## Sample output

```
==================================================================
  Last-position recall accuracy (chance ~ 0.200):
    sLSTM       (exp-gated) : 100.0%   (950 weights)
    CfC         (liquid)    : 100.0%   (330 weights)
    DiagonalSSM (LTI decay) : 100.0%   (150 weights)
==================================================================
```

(Numbers are seed-dependent. All three mixers learn the clean state-reset; the
point is that the sLSTM matches the gated/decay recurrences at a comparable
budget while using a fundamentally different mechanism — an exponential forget
gate with a running-max stabilizer that collapses the retained state in a single
step. The four gates make the sLSTM the heaviest of the three per channel.)
