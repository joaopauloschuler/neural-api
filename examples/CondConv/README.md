# CondConv — Conditionally-Parameterized ("dynamic") Convolution

A small bake-off demonstrating `TNNetCondConv` (Yang et al. 2019, NeurIPS,
[arXiv:1904.04971](https://arxiv.org/abs/1904.04971)): a *dynamic* convolution
that matches a much **wider** plain convolution while keeping the inference cost
of a **single** conv.

## The idea

`TNNetCondConv` owns a **bank of K expert kernels** `W_1..W_K` (each a normal
`Features × FeatureSize × FeatureSize × InChannels` kernel) plus a tiny
**per-sample routing head**: global-average-pool over the spatial map → a small
FullConnect → sigmoid, emitting K mixing coefficients `alpha_1..alpha_K` **per
input sample**. The effective kernel is the per-sample blend

```
W_eff = sum_k alpha_k * W_k
```

applied as **one ordinary convolution**. So inference cost stays that of a single
conv regardless of K, while model capacity grows with the bank.

It is distinct from its fork siblings:
- `TNNetHyperConv` *generates* the whole kernel from a second tensor in one shot.
- `TNNet.AddMixtureOfExperts` mixes K expert **outputs** post-hoc (K forward passes).
- **CondConv mixes K kernels *before* the conv** (one forward pass / one conv).

## The task

8×8 single-channel fields. The **sign of each sample's global mean** selects which
of two ground-truth 3×3 filters (a horizontal-edge detector vs a blur) produced
the target. So the *right* kernel **depends on the input** — exactly what
per-sample routing buys. The routing signal (global mean) is precisely what the
global-avg-pool head sees.

Three contenders map an 8×8×1 field → 8×8×1 field with a 3×3 kernel:
- **(A)** one plain narrow conv `TNNetConvolutionLinear(1,3,1,1)`;
- **(B)** a much wider plain conv (`cWide` feature maps) + a 1×1 reduce — more
  weights **and** more inference FLOPs;
- **(C)** `TNNetCondConv(K,1,3,1,1)` — K-expert dynamic conv, one conv at inference.

## Running

```
lazbuild examples/CondConv/CondConv.lpi
./bin/x86_64-linux/bin/CondConv
```

Pure CPU, tiny data, runs in well under a minute. Typical result (lower val-MSE
is better; `infer` is wall-clock over the validation set):

```
  TNNetConvolutionLinear (1 plain conv)     weights=   9   infer= ~16 ms   val-MSE~1.18
  TNNetConvolutionLinear (wide=8) + 1x1     weights=  80   infer= ~29 ms   val-MSE~1.96
  TNNetCondConv (K=2 experts)               weights=  22   infer= ~14 ms   val-MSE~0.02
```

The single plain conv cannot switch behaviour; the wide plain conv costs more
weights and more inference FLOPs; **CondConv matches/beats them at the inference
cost of one narrow conv**, because it routes each sample to the right blend of its
expert kernels.
