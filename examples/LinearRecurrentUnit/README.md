# LinearRecurrentUnit — long-range integration, recurrence vs memoryless

This example showcases **`TNNetLRU`**, the **Linear Recurrent Unit**
(Orvieto et al. 2023, *"Resurrecting Recurrent Neural Networks for Long
Sequences"*, [arXiv:2303.06349](https://arxiv.org/abs/2303.06349)) — a **stable
complex-diagonal linear recurrence** — and contrasts it head-to-head with a
**memoryless per-token baseline** on the canonical task that separates them: a
**causal prefix sum** (running integral).

## The layer

`TNNetLRU` is a diagonal linear recurrence over the channel axis whose
eigenvalue is exp-parameterised so it can **never leave the unit disk**:

```
lambda = exp(-exp(nu) + i*exp(theta))     (|lambda| < 1 by construction)
gamma  = sqrt(1 - |lambda|^2)             (input normaliser)
h_t    = lambda * h_{t-1} + gamma * B * x_t
```

A channel whose `|lambda|` is driven near `1` becomes a **leak-free
accumulator**: `h_t ≈ h_{t-1} + gamma*B*x_t`, i.e. a running sum the readout can
rescale into the target. The three per-channel parameter sets live on the layer's
neurons (`Neurons[0]` = `nu` → eigenvalue magnitude, `Neurons[1]` = `theta` →
rotation, `Neurons[2]` = input gain `B`). The example seeds them with a
**long-range warm start** (`|lambda| ≈ 0.999`, near-zero rotation, `B` chosen so
`gamma*B ≈ 1`) so each channel forms a true running sum from step one; all
parameters remain fully trainable from there. The LRU runs **directly on the
input channels** (depth-preserving) so the clean signal channel is not attenuated
by a random front projection.

## The task (causal prefix sum)

Each sequence has length `cSeqLen = 24` and a single signal channel carrying a
random value `s_t` at every position, plus `cNoiseDim = 3` irrelevant noise
channels. The target at **every** position is the running cumulative sum of the
signal so far:

```
pos t    : [ signal=s_t | noise ]
target@t : sum_{tau=0..t} s_tau
```

Evaluation reports the MSE at the **final position** — the hardest, since it
integrates the whole window. A correct answer there needs information from every
past step: exactly what a linear recurrence does and what a memoryless per-token
map cannot. The memoryless arm only ever sees the current token, so at the last
position it has access to `s_{N-1}` alone; its error floors at the variance of
the unseen partial sum.

## The bake-off

Two arms share the same I/O contract and a matched parameter budget, differing
only in the sequence-mixing core:

| arm | core |
|-----|------|
| LRU | `TNNetLRU` (over input channels) → `TNNetPointwiseConvLinear(1)` readout |
| memoryless | `TNNetPointwiseConv(cModelDim)` (tanh) → `TNNetPointwiseConvLinear(1)`, per token |

Both start from `TNNetInput(cSeqLen, 1, cInDim)`. Both are trained on the **same
RNG-replayed stream** (`cTrainSteps = 12000` per-sample SGD steps, `lr = 0.01`,
momentum `0.9`) and evaluated on `cEvalSeqs = 1000` held-out integration
sequences. The example **asserts its own headline**: it checks that the LRU MSE
is below half the memoryless MSE and prints `OK` / `NOTE` accordingly.

## Running

```
cd examples/LinearRecurrentUnit
fpc -O3 -Mobjfpc -Sh -Fu../../neural LinearRecurrentUnit.lpr
./LinearRecurrentUnit
```

(or open `LinearRecurrentUnit.lpi` in Lazarus). Pure CPU, tiny dims; finishes in
well under 5 minutes on 2 cores.

## Expected output

The program prints a header, the per-arm parameter counts, a training notice, and
the two final-position integration MSEs:

```
=== LRU: long-range integration (cumulative sum), recurrence vs memoryless ===
seq_len=24  model_dim=16  integrate over 24 steps

LRU         params = ...
Memoryless  params = ...

training both arms on the SAME stream (12000 steps each)...

eval over 1000 held-out integration sequences:
  LRU (stable complex recurrence): integration MSE = ...
  Memoryless (per-token MLP)     : integration MSE = ...

OK: the LRU integrates the signal across the whole window; the memoryless
baseline cannot see past the current token.
```

The LRU arm drives the integration MSE far below the memoryless arm, a clean
demonstration that the LRU's stable near-unit eigenvalue genuinely integrates
information across the whole time axis. Exact numbers are seed-dependent; the
*contrast* is the point.

Coded by Claude (AI).
