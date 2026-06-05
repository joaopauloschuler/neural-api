# HyperNetwork — context-generated weights with `TNNetHyperLinear`

This example reproduces the core idea of Ha, Dai & Le 2016,
[*HyperNetworks*](https://arxiv.org/abs/1609.09106), on a tiny pure-CPU
multi-task target. A small **generator** network consumes a per-task **context**
vector and **emits the weights** of a *main* layer, which then applies those
generated weights to the actual input. One shared main network thus implements a
whole **family** of input→output maps; the per-task behaviour is carried
entirely by the context-conditioned generated weights.

## The new mechanism: `TNNetHyperLinear`

Every other layer in this library **owns** its weights in `Neurons[].Weights`,
fixed at construction. **`TNNetHyperLinear` owns no trainable weights at all** —
its weight matrix is read from a *second input tensor* (the generated weights):

* **Two sources** (wired like `TNNetCrossAttention` / `TNNetAffineGridSample`):
  * `PrevLayer` — the main feature vector (`Din` values).
  * `WeightsSource` — a flat generated-weights vector of size
    `Din*Dout` (no bias) or `Din*Dout + Dout` (with bias), read row-major as the
    weight matrix `W[o,i]` followed by the optional bias `b[o]`.
* **Forward:** `y[o] = sum_i W[o,i]·x[i] (+ b[o])`.
* **Backward** (both paths, so the generator trains end-to-end):
  * `dL/dx[i] = sum_o W[o,i]·dy[o]` → into the main features
  * `dL/dW[o,i] = dy[o]·x[i]`, `dL/db[o] = dy[o]` → into the generated-weights tensor

The `WeightsSource` layer index is serialized (injected into the source slot, like
`TNNetConcat`), so the layer round-trips through `SaveToString` / `LoadFromString`.

## The multi-task target

`cTasks` distinct 2D→2D linear maps. Task `t` rotates the input by `theta_t` and
scales it by `s_t` (a different 2×2 matrix per task), selected only by the task
id. The network has two input branches joined at the hyper layer:

```
context Input(one-hot task id) -> FullConnect(hidden)            (task embedding)
                               -> FullConnectLinear(Din*Dout+Dout) = GENERATED WEIGHTS
feature Input(2D point)        -> TNNetHyperLinear(Dout, WeightsSource=generated)
```

A single shared `TNNetFullConnectLinear` trained on the same mixed stream (no task
conditioning) is the contrast baseline — it has one fixed matrix and can only
average the tasks.

## Running

```
cd examples/HyperNetwork
fpc -Mobjfpc -Sh -O3 -Fu../../neural -dRelease -dAVX2 HyperNetwork.lpr
./HyperNetwork
```

(or open `HyperNetwork.lpi` in Lazarus). Pure CPU, single thread, finishes in
about a second — comfortably under the 5-minute budget.

## Representative output (4 tasks)

```
HyperNetwork structure:
  context Input(4) -> FullConnect(8) -> FullConnectLinear(6) = generated weights
  feature Input(2) -> TNNetHyperLinear(2, WeightsSource=generated)
  (the hyper layer itself owns ZERO trainable weights: 0)

RESULTS (per-task held-out test MSE)
  task            hyper    shared-linear
  0           3.80E-013        1.07E-001
  1           1.64E-012        2.36E-002
  2           1.86E-013        1.99E-002
  3           1.80E-012        1.68E-001
  mean        1.00E-012        7.94E-002
  hyper mean-MSE 1.0E-012  vs  shared-linear mean-MSE 7.9E-002

Save/load round-trip: reloaded net task-0 MSE=3.82E-013 (original 3.80E-013).
```

The one shared hyper layer drives every task's MSE to ~1e-12 (it has zero owned
weights), while the fixed shared linear layer is stuck averaging the family.

Coded by Claude (AI).
