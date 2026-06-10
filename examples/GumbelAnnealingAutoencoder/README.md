# GumbelAnnealingAutoencoder

A temperature-annealing micro-experiment for the `TNNetGumbelSoftmax` bottleneck.

This is **not** the same experiment as
[`GumbelSoftmaxDemo`](../GumbelSoftmaxDemo/), which only sweeps `tau` on a *fixed*
logit vector at inference (no training). Here we **train a tiny discrete-latent
autoencoder end-to-end** and **anneal `tau` across training**.

## The model

```
input (D=8 dims)
  -> encoder: FullConnectReLU(16) -> FullConnectLinear(K=6)   [K logits]
  -> TNNetGumbelSoftmax(tau)                                   [discrete latent]
  -> decoder: FullConnectReLU(16) -> FullConnectLinear(8)      [reconstruction]
```

The synthetic dataset is genuinely **K-category structured**: `K=6`
well-separated cluster prototypes in `D=8` dimensions plus small Gaussian noise
(`std=0.10`). A categorical (one-of-K) bottleneck is the right inductive bias —
each sample should be routed to its cluster's latent code and the decoder
reconstructs the prototype. With clean routing the achievable reconstruction MSE
floor is roughly the noise variance, `0.10^2 = 0.01`.

## How `tau` is annealed

The network is built **once**. At the start of each phase we lower the Gumbel
bottleneck's temperature **in place** with
`TNNetGumbelSoftmax.SetTemperature(tau)`. `tau` is read live by `Compute` on
every forward pass, so the learned encoder/decoder weights carry forward
automatically — no rebuild, no `CopyWeights`.

Schedule: `tau = 2.0 -> 1.0 -> 0.5 -> 0.25 -> 0.1`, 30 SGD epochs per phase.

## Headline output

For each phase we report, on the **deterministic inference path** (Gumbel noise
disabled), the reconstruction MSE and the **mean Shannon entropy** (nats) of the
bottleneck's categorical output. Maximum possible entropy is `ln(K) = 1.79`.

Representative run (`RandSeed = 424242`):

```
  phase    tau    recon-MSE    entropy   entropy/ln(K)
      0   2.00     0.010323    0.06444         0.0360
      1   1.00     0.010374    0.00062         0.0003
      2   0.50     0.010372    0.00000         0.0000
      3   0.25     0.010362    0.00000         0.0000
      4   0.10     0.010352    0.00000         0.0000

VERDICT: PASS - annealing sharpened the categorical while keeping recon.
```

As `tau` drops the categorical **sharpens toward one-hot** — the bottleneck
entropy collapses from `0.064` nats to exactly `0`. Reconstruction stays at the
noise floor (`~0.0103`, essentially the `0.01` lower bound), confirming the
discrete latent reliably routes each sample to its cluster.

## Honest caveat on the absolute entropy

The entropy at the *highest* `tau` is already modest (`~0.06`, far below
`ln(K)=1.79`), not close to uniform. That is **expected and honest** for this
dataset: the clusters are well separated, so correct routing demands very
**confident** logits (large pairwise gaps). Even dividing those large logits by
`tau=2.0` leaves a near-one-hot softmax. The annealing *trend* (entropy strictly
decreasing toward 0 as `tau` falls, recon held at the noise floor) is exactly the
predicted behaviour; the experiment simply lives in the confident-routing regime
where the highest-`tau` distribution is already fairly peaked. The verdict checks
the trend (final entropy well below the initial entropy and a small fraction of
`ln(K)`, with reconstruction not degrading), not an unrealistically high starting
entropy.

## Build / run

```
cd examples/GumbelAnnealingAutoencoder
fpc -O3 -Mobjfpc -Sh -dRelease -dAVX2 -Fu../../neural -Fi../../neural GumbelAnnealingAutoencoder.lpr
./GumbelAnnealingAutoencoder
```

Pure CPU, single-threaded, runs in about 2 seconds.
