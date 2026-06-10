# Liquid CfC vs fixed-decay diagonal SSM (last-write-wins)

A longer-horizon, multi-cue **last-write-wins** toy that isolates the one thing a
CfC liquid cell (`TNNetClosedFormContinuous`) has and a fixed-decay diagonal SSM
(`TNNetDiagonalSSM`) does not — an **input-dependent time constant** — and shows
it pays off at a (more than) matched parameter budget.

This is the sibling of [`examples/LiquidCfC`](../LiquidCfC), which contrasts the
CfC against an attention head on a simpler single-cue recall task. Here the
baseline is a *fixed-decay* recurrence, and the task is deliberately chosen to
require **input-dependent forgetting**.

Reference: Hasani et al. 2022,
[*Closed-form continuous-time neural networks*](https://www.nature.com/articles/s42256-022-00556-7),
Nature Machine Intelligence.

## CfC vs diagonal SSM: the difference that matters here

```
CfC :  tau_t[d] = b_t[d] + Σ_j Wt[d,j]·x_t[j]            (rate depends on input)
       gate_t[d]= sigmoid(-tau_t[d]·tnorm_t)
       h_t[d]   = gate_t[d]·g_t[d] + (1 - gate_t[d])·h_{t-1}[d]

SSM :  h_t[d]   = a[d]·h_{t-1}[d] + b[d]·x_t[d]          (a[d] learned ONCE)
       y_t[d]   = c[d]·h_t[d] + e[d]·x_t[d]
```

The CfC's `gate_t` is a function of the **current input** `x_t`, so the cell can
slam the gate shut on a marker and **reset/overwrite** its state. The diagonal
SSM's decay `a[d]` is fixed after training: it can only **blend** the new value
into the old one, never selectively forget.

## The task

`SeqLen = 16`. Several cue symbols in `{1..4}` (2–4 of them) are written at
random positions; a dedicated **WRITE marker channel** pulses on at every cue;
the rest is filler. At the **last** position the model must output the
**most-recently written** cue. Because each new write must *overwrite* the
previous value, the model needs input-dependent forgetting — exactly the CfC's
strength and the fixed-decay SSM's weakness.

Both models share the same skeleton:

```
Input(SeqLen,1,2)            // channel 0 = symbol id, channel 1 = write pulse
  -> embed channel 0, concat raw write pulse back
  -> (TNNetClosedFormContinuous  |  TNNetDiagonalSSM)
  -> LayerNorm -> per-position PointwiseConvLinear -> softmax
```

Only the mixer differs. The SSM is given a wider working width because a diagonal
SSM is cheap per channel (`4·d` weights vs the CfC's `2·d² + 2·d`); even so it is
given **more** total weights than the CfC and still loses.

## Build & run

```bash
cd examples/LiquidCfCvsSSM
lazbuild LiquidCfCvsSSM.lpi
../../bin/x86_64-linux/bin/LiquidCfCvsSSM
```

Pure CPU, single-threaded, no external dataset; finishes in ~10 s, well under the
5-minute budget.

## Sample output

```
================================================================
  Last-position recall accuracy (chance = 0.250):
    CfC liquid cell (input-dependent decay): 100.0%   (475 weights)
    Diagonal SSM    (fixed decay)          : 93.8%   (639 weights)
================================================================
```

(Numbers are seed-dependent.) The CfC drives the cross-entropy to ~0 and recalls
the last write perfectly; the fixed-decay SSM plateaus at a clearly higher loss
and a few points below 100% **despite having more parameters** — earlier cues
bleed into its answer because it cannot reset on a write pulse.

## Related

- [`examples/LiquidCfC`](../LiquidCfC) — the single-cue CfC-vs-attention toy.
- `TNNet.AddClosedFormContinuous` — RMSNorm pre-norm residual CfC block builder.
- `TNNet.AddBidirectionalClosedFormContinuous` — forward + reverse CfC
  concatenated along Depth, for non-causal sequence tasks.
