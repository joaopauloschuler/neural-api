# LFQVAE — Lookup-Free (binary) Quantization

A self-contained Lookup-Free Quantization (LFQ) autoencoder demo (Yu et al.
2023, MagViT-v2,
[*Language Model Beats Diffusion: Tokenizer is Key to Visual Generation*](https://arxiv.org/abs/2310.05737);
a port of the lucidrains `vector-quantize-pytorch` LFQ) built on the new layer
**`TNNetLookupFreeQuant`**. It is the **binary sibling** of
[`FSQVAE`](../FSQVAE).

## Why LFQ

FSQ ([`FSQVAE`](../FSQVAE)) removes the learned codebook of a VQ-VAE by rounding
each latent channel to one of `L_i` integer levels. LFQ takes that to the limit
`L_i = 2`: each channel is just `sign(z)` in `{-1,+1}`, so the implicit codebook
is the **product set `{-1,+1}^D`** of size `2^D` — reachable with **NO codebook
lookup at all** (no learned vectors, no argmin, unlike `TNNetVectorQuantizer`).

```
zhat_i = +1 if z_i > 0 else -1          (sign, with the lucidrains 0 -> +1)
```

The discrete token at a position is the **bit-packed sign pattern** (bit `i` = 1
iff `z_i > 0`, channel 0 most significant), read via the public
`LFQ.CodeIndex(X, Y)`. Gradients flow through the non-differentiable `sign` via
the **straight-through estimator** clipped to the `|z| <= 1` band (the gradient
passes inside the band and is zeroed outside — the lucidrains LFQ math, exactly).

## The entropy objective

Because the codebook factorizes per channel, LFQ's `entropy_aux_loss` has a
tractable **binary** form. Per channel the soft assignment to `{-1,+1}` is

```
logits = -t * [ (z+1)^2, (z-1)^2 ]      (inverse-temperature t)
p      = softmax(logits)
EntropyAuxLoss = PerSampleEntropy - DiversityWeight * CodebookEntropy
```

where **PerSampleEntropy** (minimized) drives each assignment to be *confident*
and **CodebookEntropy** (maximized) drives the batch to use codes *diversely*.
The three terms are exposed as PUBLIC methods on `TNNetLookupFreeQuant`
(`PerSampleEntropy` / `CodebookEntropy` / `EntropyAuxLoss`), computed on the last
`Compute()`. The layer injects **no** entropy gradient — the STE is the only
gradient path — so a real tokenizer **adds** `EntropyAuxLoss` to its
reconstruction loss to keep the codebook diverse, exactly the way
`TNNetLoadBalanceLoss` is added to a Mixture-of-Experts objective. This demo only
**reads** the accessors to show they track code usage.

## What the program does

* Builds a tiny MLP autoencoder `16 → … → 6 LFQ channels (bottleneck) → … → 16`
  and trains it on reconstruction MSE over synthetic prototype data (hand-rolled
  mini-batch SGD).
* Reports, before and after training, the number of **distinct binary codes**
  seen (`LFQ.CodeIndex`) and the LFQ **entropy terms**.
* Prints a graded `VERDICT: PASS` once the binary bottleneck cleanly separates
  the data prototypes into distinct codes and the entropy accessors return sane
  values.

## Run

```
./LFQVAE        # ~2 s on one CPU, VERDICT: PASS — NO external data needed
```

Pure CPU, no data files required.
