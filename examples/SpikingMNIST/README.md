# SpikingMNIST — a spiking net learns through a surrogate gradient with `TNNetLIFNeuron`

This example showcases **`TNNetLIFNeuron`**, an **event-driven, stateful spiking
leaky-integrate-and-fire (LIF) neuron** layer trained with a **surrogate
gradient** (Neftci, Mostafa & Zenke 2019, *"Surrogate Gradient Learning in
Spiking Neural Networks"*; Zenke & Ganguli 2018, *SuperSpike*). It is a new
computational paradigm for this library: instead of a smooth activation, the
layer integrates an input current into a **membrane potential over a time axis**
and emits a **binary {0,1} spike** when the potential crosses a threshold, then
**resets**.

> **This is a TINY SYNTHETIC rate-coded task, not real MNIST.** The "MNIST" name
> nods to the canonical spiking-net benchmark; to stay inside a small pure-CPU
> budget we use a handful of synthetic classes rate-encoded as Bernoulli spike
> trains. The point is the *learning paradigm*, not the dataset.

## The layer

Per neuron (one independent LIF unit per channel) over input shape `(T, 1, D)` —
`T` time steps on `SizeX`, `D` channels on `Depth` (the same time-on-X convention
as the SSM / attention / `TNNetCausalConv1D` layers):

```
V[t] = beta*V[t-1]*(1 - S[t-1]) + I[t]      membrane integrate + leak + hard reset
S[t] = 1 if V[t] >= V_th else 0             hard Heaviside spike
```

`beta = exp(-1/tau)` is the membrane leak, `V_th` the firing threshold. The layer
**output is the binary spike train `S`** (faithfully binary at inference). By
default it has **no trainable parameters of its own** — it is a pointwise neuron
model over an upstream linear/conv layer, like an activation.

### Opt-in learnable per-channel dynamics

The constructor's 4th flag `pLearnDynamics` (default `false`) makes the
per-channel firing threshold `V_th[d]` and leak `beta[d]` **trainable**
parameters instead of fixed scalars. Two weight neurons of width `Depth` are
installed: an unconstrained raw threshold (`V_th[d] = raw`) and a constrained
raw leak (`beta[d] = sigmoid(raw)`, always in `(0,1)`), seeded so the layer
starts at the scalar `(tau, V_th)` it was created with. The exact surrogate
backward accumulates `dL/dV_th[d]` and `dL/dbeta[d]` (chaining
`dbeta/draw = beta*(1-beta)`). With the flag **off** the layer is byte-identical
to the original parameter-free layer. This example trains with
`LearnDynamics=true`.

### One-line builder: `AddSpikingBlock`

`TNNet.AddSpikingBlock(pHidden, tau, V_th, alpha, LearnDynamics)` wires the
canonical **linear -> LIF -> rate-readout** pipeline (a per-timestep
`TNNetPointwiseConvLinear` projection, a `TNNetLIFNeuron`, then a
`TNNetAvgChannel` rate readout averaging spikes over the time axis to a
`(1,1,pHidden)` firing-rate vector) and returns the rate-readout layer. This
example builds its spiking stage with that single call.

### Why a surrogate gradient

The forward spike is a hard Heaviside whose derivative `dS/dV` is **zero almost
everywhere**, so plain backprop gives no gradient. The backward pass instead
substitutes the **fast-sigmoid / SuperSpike surrogate**

```
sigma'(V) = 1 / (1 + alpha*|V - V_th|)^2
```

and back-propagates **through time** across the `T` unrolled steps: `dL/dV[t]`
receives the direct spike-path term plus the membrane-carry term from `t+1`
(through `beta` and the reset coupling). `alpha` sets the surrogate sharpness.

## The network

```
Input(T,1,DIN) spike trains
  -> AddSpikingBlock(HIDDEN, tau, V_th, alpha, LearnDynamics=true)
       PointwiseConvLinear(HIDDEN)        per-timestep synaptic current (pointwise)
       TNNetLIFNeuron(tau,V_th,alpha,...) HIDDEN spiking neurons, trainable V_th/leak
       TNNetAvgChannel                    rate readout -> (1,1,HIDDEN)
  -> FullConnectLinear(NCLASS) -> SoftMax
```

Each class has a prototype per-feature firing probability; a sample is the
prototype rate-encoded as `T` independent Bernoulli spike trains. Trained with
plain SoftMax + cross-entropy — the whole gradient passes **through** the LIF
layer via the surrogate.

## Headline result

A typical run on 2 CPU cores (~2 seconds, ~4 MB RAM, no committed binaries):

```
Test accuracy BEFORE training: 47.1%  (chance = 25.0%)   hidden spike rate: 11.6%
  epoch    0  test-acc= 47.1%  hidden-spike-rate= 11.6%
  epoch  100  test-acc= 26.6%  hidden-spike-rate= 11.8%
  epoch  300  test-acc= 66.8%  hidden-spike-rate= 12.6%
  epoch  500  test-acc= 94.1%  hidden-spike-rate= 13.7%
  epoch  599  test-acc= 99.0%  hidden-spike-rate= 14.3%
Test accuracy AFTER training: 99.0%
Mean hidden SPIKE RATE: 14.3%  (sparsity = 85.7% silent).
```

The headline payoff is the **spike rate reported alongside accuracy**: the net
reaches **99%** accuracy while its hidden neurons fire on only **~14%** of
(time, neuron) sites — i.e. ~86% of the spiking compute is *silent*. Low,
event-driven sparsity is exactly what makes spiking networks attractive, and the
surrogate gradient still trains the net to high accuracy.

## Honest caveat (forward-vs-backward mismatch)

The forward pass is the **hard** reset + Heaviside spike, but the backward pass
uses a **smooth** surrogate — so the gradient is a **biased estimator** of the
true (almost-everywhere zero) gradient. There is a real **forward-vs-backward
mismatch**. In practice this means a spiking net needs a **gentler learning rate
and/or more steps** than an equivalent ReLU MLP on the same task, and training is
noisier (note the dip around epoch 100 before the net climbs). We pick a modest
LR (0.02) and a small task accordingly — do not expect ReLU-MLP convergence
speed. The `BEFORE` accuracy can also land above chance purely from a lucky
random init; what matters is the climb to near-perfect accuracy at a low,
event-driven spike rate.

## Build & run

```
lazbuild --bm=Release SpikingMNIST.lpi
../../bin/x86_64-linux/bin/SpikingMNIST
```
