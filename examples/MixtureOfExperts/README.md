# MixtureOfExperts — soft (dense) mixture-of-experts feed-forward block

Demonstrates **`TNNet.AddMixtureOfExperts`**, a **soft (dense) mixture-of-experts**
feed-forward block, on a synthetic **multi-task regression** toy. Every input
carries a one-hot *task* selector plus a payload, each task applies a different
non-linear function to the payload, and the block's gating network has an
incentive to route different tasks to different experts.

## The block

A soft MoE feed-forward block runs **every** expert on the input and blends their
outputs with weights produced by a small gating network — unlike a hard / top-k
router, all experts are evaluated and the mixture is fully differentiable. The
block is **shape-preserving**: output shape `==` input shape, so it drops in
anywhere a plain FFN would go.

```
gate(x)        = softmax over the NumExperts experts        (per sample)
expert_e(x)    = small non-linear sub-network               (e = 0 .. NumExperts-1)
MoE(x)         = Σ_e gate_e(x) · expert_e(x)
```

In this example the block is built with:

```pascal
NN.AddMixtureOfExperts(nil, NumExperts, ExpertHidden);
```

`NumExperts = 4`, `ExpertHidden = 12` (the per-expert hidden width), and the first
argument `nil` lets the builder construct the default per-expert sub-network.

## The model

A minimal three-layer network over a `(1, 1, d_model)` input
(`d_model = NumTasks + PayloadDim = 5`):

```
TNNetInput(1, 1, d_model)
 -> AddMixtureOfExperts(nil, NumExperts=4, ExpertHidden=12)   soft MoE FFN (shape-preserving)
 -> TNNetFullConnectLinear(1)                                 linear regression head
```

The structure is printed at start-up via `NN.DebugStructure()`.

## The task (multi-task regression)

`SampleCount = 96` synthetic samples. Each sample's first `NumTasks = 3` channels
hold a one-hot task selector (set with `TNNetVolume.OneHotEncodingOnPixel`); the
remaining `PayloadDim = 2` channels hold two random payload values `p0, p1` in
`[-1, 1]`. The target is a **different** non-linear function of the payload per
task:

```
task 0:  y = p0 * p1            (product)
task 1:  y = p0*p0 - p1         (square minus)
task 2:  y = 0.5 * (p0 + p1)    (average)
```

Because each task needs a different computation, the gating network is rewarded
for routing different tasks to different experts.

Training is per-sample SGD (`SetLearningRate(0.02, 0.9)`) for `Epochs = 300`,
driven directly via `NN.Compute` / `NN.GetOutput` / `NN.Backpropagate`. The
per-epoch mean squared error is accumulated by hand and printed every 50 epochs.

## How to run

```
cd examples/MixtureOfExperts
fpc -O3 -Mobjfpc -Sh -Fu../../neural MixtureOfExperts.lpr
./MixtureOfExperts
```

(or open `MixtureOfExperts.lpi` in Lazarus / `lazbuild MixtureOfExperts.lpi`).
Pure CPU, tiny dimensions; finishes in seconds.

## Expected output

The program first prints the network structure, then the MSE at epoch 1 and every
50th epoch, and finally an initial-vs-final summary with a self-check:

```
Mixture-of-Experts network layers:
... (DebugStructure dump) ...
Epoch    1  mean sq error: ...
Epoch   50  mean sq error: ...
...
Epoch  300  mean sq error: ...
Initial MSE: ...   Final MSE: ...
OK: loss decreased - the MoE block trained.
Done.
```

The headline check is that the final MSE is below the initial MSE; the program
prints `OK: loss decreased - the MoE block trained.` on success (or a `WARNING:`
line otherwise). Exact numbers are seed-dependent (`RandSeed := 12345`); the
decreasing loss is the point.

Coded by Claude (AI).
