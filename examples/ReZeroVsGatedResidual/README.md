# ReZero vs GatedResidual depth ablation

Trains the **same** deepish residual MLP on the toy hypotenuse task
(`y = sqrt(X^2 + Y^2)`) twice, changing **only** the residual gate placed on
each block's branch, then dumps the learned gate values so you can see how the
gates open during training.

```
block input x ──────────────────────────────┐ (skip)
   │                                          │
   └─► PointwiseConvLinear(WIDTH) ─► ReLU ─► GATE ─► Sum(+x) ─► block output
```

* **Arm A — ReZero** (`y = x + alpha * Sublayer(x)`): the gate is `TNNetReZero`,
  a **single learnable scalar** `alpha` per block. Wired manually, mirroring the
  body of `TNNet.AddGatedResidual` but substituting `TNNetReZero` for the gate.
* **Arm B — GatedResidual** (`y = x + alpha[d] * Sublayer(x)`): the gate is
  `TNNetGatedResidual`, **one learnable scalar per channel** (Depth). Wired with
  a single `TNNet.AddGatedResidual([...])` builder call.

Both gates initialise to `0.0`, so every block starts as the identity and the
gate "opens" as training proceeds. Everything else (architecture, synthetic
data, `RandSeed`, learning rate, epochs, batch size) is identical between arms.

## What `TNNetReZero` is in this repo (scalar vs per-channel)

`TNNetReZero` here is **scalar**: `SetPrevLayer` calls
`SetNumWeightsForAllNeurons(1, 1, 1)`, so the layer holds exactly one learnable
weight regardless of Depth (`Output[x,y,d] = alpha * Input[x,y,d]`).
`TNNetGatedResidual` is its **per-channel generalisation**:
`SetNumWeightsForAllNeurons(1, 1, Depth)` gives one `alpha[d]` per channel
(`Output[x,y,d] = alpha[d] * Input[x,y,d]`). The contrast is therefore
init-0 single-scalar gate vs init-0 per-channel gated-residual sum.

## How the gate values are read

Both gates store their learnable parameter(s) in `Layer.Neurons[0].Weights`
(a `TNNetVolume`): one element for ReZero (`.Raw[0]`), `Depth` elements for
GatedResidual (`.Raw[d]`). We capture each block's gate-layer **index** at build
time and read `NN.Layers[idx].Neurons[0].Weights` after `Fit` returns. Indices
(not layer references) are used because `TNeuralFit.Fit` reloads the best model
at the end via `FNN.LoadFromFile`, which rebuilds every layer instance — a saved
layer reference would be stale, but the structural index stays valid.

## Build & run

```
lazbuild examples/ReZeroVsGatedResidual/ReZeroVsGatedResidual.lpi --build-mode=Default
./bin/x86_64-linux/bin/ReZeroVsGatedResidual
```

Pure CPU, no external data (the hypotenuse pairs are generated in-code). Runs in
well under a minute. All prints are guarded against NaN / Inf.

## What the dump shows

The headline output is a per-block gate dump:

* **ReZero** prints the single scalar `alpha` per block.
* **GatedResidual** prints `min / mean / max`, an "open" count
  (`|gate| >= 0.01`), and an ASCII bar chart of every channel's gate.

The per-channel GatedResidual gate opens **unevenly**: within each block some
channels grow (positive or negative) while many sit at exactly `0.0`. The single
ReZero scalar, by contrast, concentrates the whole branch's gradient into one
parameter and opens faster/larger. At identical LR/epochs on this toy task the
ReZero arm therefore reaches a much lower validation MSE, while the per-channel
gates stay small and the GatedResidual arm is still close to identity — a clean
illustration that a scalar gate trains faster but a per-channel gate gives the
network the *option* to open channels independently (visible in the chart),
albeit slowly under a shared learning rate.
