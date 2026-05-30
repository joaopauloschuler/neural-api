# HardConcrete L0 Sparsity

Demonstrates `TNNetHardConcrete`, the learnable **L0-sparsity gate** of
Louizos, Welling & Kingma (2018), *"Learning Sparse Neural Networks through L0
Regularization"* (https://arxiv.org/abs/1712.01312).

## The gate

`TNNetHardConcrete` is a per-DEPTH-channel multiplicative gate `z in [0,1]`:

```
s         = sigmoid((logit(u) + log_alpha) / beta)   // u ~ Uniform(0,1), training only
s_stretched = s * (zeta - gamma) + gamma             // gamma < 0 < zeta
z         = clip(s_stretched, 0, 1)                  // a fraction of gates hit EXACTLY 0
y[x,y,d]  = x[x,y,d] * z[d]                           // broadcast over the spatial axes
```

`log_alpha` is a **trained** per-channel weight (one per depth channel); `beta`,
`gamma`, `zeta` are constructor constants (paper defaults `beta=2/3`,
`gamma=-0.1`, `zeta=1.1`). Because `gamma < 0` the stretched gate can fall below
0 and clip to **exactly 0**, so channels can be *hard*-pruned rather than merely
shrunk.

- **Training** (`Enabled` / `EnableDropouts(true)`): a fresh noise `u` is drawn
  per channel per forward and held fixed for the matching backward
  (reparameterization). The gradient is zero in the hard-clipped regions and
  flows through the sigmoid/stretch elsewhere.
- **Inference** (`Enabled = false`): the deterministic gate
  `z = clip(sigmoid(log_alpha) * (zeta - gamma) + gamma, 0, 1)`, no noise.

`TNNet.EnableDropouts(flag)` toggles the layer's `Enabled` flag (alongside the
`TNNetAddNoiseBase` dropouts), so `Fit`/manual training loops switch the
stochastic gate on for training and off for inference automatically.

## What this example shows

Two identical tiny nets gate the 12 input **features** (only features 0 and 1
carry signal; the other 10 are pure noise):

```
Input(1,1,12) -> HardConcrete gate(12) -> FullConnectReLU(16) -> FullConnect(2) -> SoftMax
```

- The **L0 variant** adds the expected-L0 penalty gradient
  `lambda * d/dlog_alpha sigmoid(log_alpha - beta*ln(-gamma/zeta))`
  to each gate's `log_alpha`. The loss gradient anchors the 2 informative gates
  open while the penalty prunes the 10 noise gates to hard zero.
- The **L2 baseline** trains the SAME gate with only a small L2 weight decay and
  no L0 pressure, so all gates stay nominally alive.

The self-gate (`Halt(1)` on failure) asserts the L0 variant reaches >=80%
accuracy AND a strictly higher hard-zero sparsity than the baseline.

## Reported result (RandSeed=424242, CPU, single-threaded)

```
  L0 per-feature log_alpha:   5.46   6.12  -6.61  -6.61 ...  (features 0,1 open; rest closed)

  L0 variant : train acc =  97.27%   hard-zero gates =  83.33%   (10 of 12 == exactly 0)
  L2 baseline: train acc =  97.66%   hard-zero gates =   0.00%
```

The L0 gate prunes precisely the 10 noise features (83.33%) while keeping the 2
informative ones, at accuracy matched to the dense baseline.

## Tuning sensitivity (be honest)

Like most L0-regularized setups, the result depends on the penalty strength
`cL0Lambda` relative to the task-loss gradient on the gates, and on the
RNG seed. The behaviour is **bistable around a knee**: at `cL0Lambda = 0.05`
exactly the 10 noise gates are pruned (the demo value); push it to `~0.1+` and
the L0 pressure overwhelms even the informative gates, collapsing *all* gates to
zero and dropping accuracy to chance (~64%); drop it well below and *no* gate is
pruned. A longer warm-up before applying the penalty, or annealing `lambda`,
widens this window in practice. The shipped constants are tuned to land on the
clean 10/12 outcome with the fixed seed; treat the exact numbers as
seed/hyperparameter-specific, not a guarantee.

## Build & run

```
fpc -O2 -Mobjfpc -Sh -Fu../../neural HardConcreteSparsity.lpr
./HardConcreteSparsity
```

(or open `HardConcreteSparsity.lpi` in Lazarus). Runs in a couple of seconds on
CPU. The compiled binary is git-ignored.
