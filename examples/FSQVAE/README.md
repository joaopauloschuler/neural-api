# FSQVAE — codebook-free Finite Scalar Quantization on MNIST

A Finite Scalar Quantization autoencoder (Mentzer et al. 2023,
[*Finite Scalar Quantization: VQ-VAE Made Simple*](https://arxiv.org/abs/2309.15505);
a port of the lucidrains `vector-quantize-pytorch` FSQ) built on the new layer
**`TNNetFiniteScalarQuant`**. It is the **collapse-free** counterpart to the
learned-codebook examples [`VQVAE`](../VQVAE) and
[`VQCodebookCollapse`](../VQCodebookCollapse).

## Why FSQ

The classic VQ-VAE discretizes a latent by nearest-neighbour lookup into a
**learned codebook**, which famously **collapses**: only a handful of entries
ever win the argmin and the rest are dead weight, needing hacks (EMA updates,
commitment loss, dead-code re-init — see [`VQCodebookCollapse`](../VQCodebookCollapse)).

FSQ removes the codebook entirely. Each of `D` latent channels is independently
bounded and rounded:

```
half_l = (L_i - 1) / 2
offset = 0.5 if L_i even else 0
shift  = atanh(offset / half_l)
f(z)   = tanh(z + shift) * half_l - offset      (the bounded value)
zhat   = round(f(z))                            (nearest of L_i levels)
```

The implicit codebook is the **product** of the `L_i` (here `5⁶ = 15625` codes),
reachable without ever being stored. Because the round is deterministic and
every level is always in range, the codebook **cannot collapse by
construction**. Gradients flow through the non-differentiable round via the
**straight-through estimator** (backward = the analytic derivative of the tanh
bound, `half_l·(1 − tanh(z+shift)²)`).

## What the program does

* Builds a tiny MLP autoencoder `784 → … → 6 FSQ channels (bottleneck) → … → 784`
  and trains it on reconstruction MSE over a small MNIST subset (hand-rolled
  mini-batch SGD).
* After training, probes the bottleneck over a fresh batch and reports:
  * **per-channel level utilization** — of each channel's `L_i` levels, how many
    are actually hit. FSQ being collapse-free, this climbs to **~100 %**;
  * the number of **distinct full codes** seen, read via the public
    `LFSQ.CodeIndex(X, Y)` accessor (the discrete token a downstream
    embedding/transformer prior would consume).
* Prints a graded `VERDICT: PASS` once mean per-channel utilization clears a high
  bar.

## Gotcha — keep latents in the informative tanh region

FSQ's one training pitfall: if the raw encoder latents **collapse toward 0** they
all map to the single central level, and if they **saturate past the tanh knee**
they all map to a single outer level — either way utilization sticks low. This
example fixes it the standard way: a per-channel
`TNNetChannelStdNormalization` (unit-std latents) followed by a fixed gain
(`TNNetMulByConstant`) so each latent spans the full `±` range of a 5-level
channel. With that in place all six channels reach 100 % level utilization.

## Run

```
# from a directory containing the MNIST idx-ubyte files (e.g. examples/VQVAE):
../FSQVAE/FSQVAE          # SMOKE: ~33 s on one CPU, VERDICT: PASS
../FSQVAE/FSQVAE --full   # longer run
```

Needs the standard MNIST `*-idx?-ubyte` files in the working directory (the same
files every MNIST example here uses). If absent the program prints a hint and
exits cleanly. Pure CPU.
