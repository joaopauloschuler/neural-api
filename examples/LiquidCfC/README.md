# Liquid CfC cell (Closed-form Continuous-time)

A one-screen **remember-then-recall** toy that contrasts the new
`TNNetClosedFormContinuous` "liquid" recurrent cell against a single
scaled-dot-product attention (**SDPA**) head at a **matched parameter count**.

Reference: Hasani et al. 2022,
[*Closed-form continuous-time neural networks*](https://www.nature.com/articles/s42256-022-00556-7),
Nature Machine Intelligence.

## What a CfC cell is

A *liquid time-constant* (LTC) network is an ODE whose per-channel time constant
is itself a **learned function of the input**. The CfC contribution is the
**analytic closed-form solution** of that ODE, so there is **no numerical ODE
solver**. Per timestep `t`, per channel `d`:

```
tau_t[d]  = b_t[d] + Σ_j Wt[d,j]·x_t[j]          (input-dependent rate)
gate_t[d] = sigmoid(-tau_t[d] · tnorm_t)         (closed-form time gate)
g_t[d]    = tanh(b_g[d] + Σ_j Wg[d,j]·x_t[j])    (fast input pathway)
h_t[d]    = gate_t[d]·g_t[d] + (1 - gate_t[d])·h_{t-1}[d]      (h_{-1}=0)
y_t[d]    = h_t[d]
```

`tnorm_t = (t+1)/SeqLen` is the elapsed continuous time. The per-channel learned
rate `tau`, computed from the **current input**, decides how fast each channel
forgets — the defining "liquid" behaviour. This is distinct from the
other sequence mixers in the library:

| Layer | Mechanism | Decay |
| --- | --- | --- |
| `AddNeuralODEBlock` | numerically integrated residual field | n/a (RK/Euler step) |
| `TNNetRetention` | softmax-free attention with fixed decay mask | fixed `γ^(n-m)` |
| `TNNetDiagonalSSM` | linear recurrence | fixed per-channel `sigmoid(a_raw)` |
| `TNNetSelectiveSSM` | Mamba/S6 selective state space | input-dependent |
| **`TNNetClosedFormContinuous`** | **closed-form liquid recurrence** | **input-dependent per-channel time constant** |

Highway / GatedResidual gates are depthwise but **not** recurrent over the time
axis; the CfC carries a hidden state through time.

## The task

A **cue symbol** in `{1..4}` is shown at position `0`; the rest of the sequence
is filler (symbol `0`). Every position must report the cue, so the model has to
carry one piece of information across the whole sequence. Accuracy is read at the
**last** position (the longest recall horizon). A recurrence solves this by
holding the cue in its hidden state; an attention head by attending back to
position `0` — two different mechanisms at the same parameter budget.

Both models share the same `Embedding -> (mixer) -> per-position readout ->
softmax` skeleton; only the mixer differs. The SDPA head width `d_k` is chosen so
the two learnable-weight counts are close (~220 each).

## Library API

```pascal
// Leaf layer over a (SeqLen, 1, Depth) sequence (SizeY must be 1).
// Output shape == input shape.
NN.AddLayer(TNNetClosedFormContinuous.Create());
```

Storage is four learnable tensors: `Wt`, `Wg` (Depth×Depth projections) and
`b_t`, `b_g` (Depth-long biases). `Depth` is inferred from the previous layer, so
nothing has to be passed to `Create`; the layer round-trips through save/load
with no constructor arguments (the recurrence reuses the standard neuron weight
storage).

The forward pass is the explicit per-timestep recurrence; the backward pass is
backprop-through-time (a right-to-left `dL/dh` sweep). The gradients are pinned by
finite-difference checks in `tests/TestNeuralNumerical.pas`
(`TestClosedFormContinuousInputGradientCheck`,
`TestClosedFormContinuousWeightGradientCheck`, max error ≈ 2·10⁻³) plus a
save/load round-trip (`TestClosedFormContinuousSerializationRoundTrip`).

## Build & run

```bash
cd examples/LiquidCfC
lazbuild LiquidCfC.lpi
../../bin/x86_64-linux/bin/LiquidCfC
```

Pure CPU, single-threaded, no external dataset; finishes in a few seconds with a
modest memory footprint.

## Sample output

```
================================================================
  Last-position recall accuracy (chance = 0.250):
    CfC  liquid cell : 100.0%   (224 weights)
    SDPA attention   : 100.0%   (214 weights)
================================================================
```

(Numbers are seed-dependent. Both mixers reach 100% recall at a matched
parameter budget; the CfC converges cleanly to a near-zero cross-entropy.)
