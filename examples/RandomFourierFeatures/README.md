# RandomFourierFeatures — an RBF-kernel random-feature layer with `TNNetRandomFourierFeatures`

This example showcases **`TNNetRandomFourierFeatures`**, a kernel-approximation
projection layer (Rahimi & Recht 2007, *"Random Features for Large-Scale Kernel
Machines"*). It maps a `Din`-vector input `x` to a `2·D`-vector feature map

```
phi_k(x) = sqrt(1/D) · [ cos(w_k · x) , sin(w_k · x) ]   for k = 1..D
```

where the projection rows `w_k` (the `D × Din` matrix `W`) are drawn **once**,
i.i.d. from `N(0, 1/sigma²)`, and **frozen by default**. The remarkable fact:

```
<phi(x), phi(y)>  →  exp(-‖x-y‖² / (2·sigma²))   as D → ∞
```

i.e. the dot product of the random features **approximates the RBF / Gaussian
kernel**. So a plain **linear** classifier over `phi(x)` approximates an
**RBF-kernel machine** (a kernel SVM / kernel ridge regressor) — *without* ever
forming the `N×N` Gram matrix, and with a feature map that is fixed up front.

This is mathematically **distinct** from the learnable FFT layers in neural-api
(`TNNetFourierMixFFT`, `TNNetSpectralConv1D/2D`, the `TNNetCirculantLinear` FFT
path): RFF is a **fixed random Gaussian projection** approximating a
shift-invariant kernel, *not* a transform along a signal axis.

## How it works

* **Forward**: one matmul `z = W·x`, then the cos/sin lift scaled by `sqrt(1/D)`.
  Output `Depth = 2·D` — the `D` cosine channels followed by the `D` sine channels.
* **Frozen mode** (default, classic RFF): backward propagates only `dL/dx`; `W`
  stays the fixed random map.
* **Trainable mode** (constructor flag → "deep kernel learning"): also
  accumulates `dL/dW`, with the exact chain rule
  `dz_k = sqrt(1/D)·(−sin(z_k)·dCos_k + cos(z_k)·dSin_k)`, `dW row_k += dz_k·x`.
  `sigma` is kept **fixed** (a frozen scalar — `dL/dsigma` is **not** propagated)
  to keep the scope tight.

`D` round-trips through `FStruct[0]`, the draw `RandSeed` through `FStruct[5]`,
the trainable flag through `FStruct[6]`, and `sigma` through `FFloatSt[0]`. `W`
is also written out by the ordinary per-neuron weight serialization, so a
save/load round-trip reloads the **same** random map exactly.

## The task

The canonical **concentric rings**: an inner disk (class 0) surrounded by an
outer ring (class 1). No straight line separates them. We compare three models
on a held-out test set:

| model | architecture | trainable params | what learns |
|-------|--------------|------------------|-------------|
| **(A) RFF** | `RFF(D=256, frozen) → FullConnectLinear(2) → SoftMax` | 1536 | only the linear head |
| (B) linear | `FullConnectLinear(2) → SoftMax` on raw (x,y) | 4 | everything (but can't bend) |
| (C) MLP | 2×`FullConnect(32)+ReLU → FullConnectLinear(2) → SoftMax` | 1152 | learned features |

The RFF model's feature map is **frozen** — only the tiny 2-class linear head
trains — yet it carves out the curved decision boundary, because the random
features make the rings linearly separable (an explicit RBF-kernel machine).

## Running

```
cd examples/RandomFourierFeatures
lazbuild RandomFourierFeatures.lpi
../../bin/x86_64-linux/bin/RandomFourierFeatures
```

(or `fpc -Mobjfpc -Sh -O3 -Fu../../neural -dRelease -dAVX2 RandomFourierFeatures.lpr`,
or open `RandomFourierFeatures.lpi` in Lazarus). Pure CPU, finishes in a **few
seconds** — well under the 5-minute budget. No binaries are committed.

## Representative output

```
=== Held-out test accuracy ===
  (A) RFF + linear head : 1.000
  (B) raw linear        : 0.468
  (C) ReLU MLP          : 1.000
```

The **frozen random-feature layer + linear head separates the rings perfectly**
(matching the ReLU MLP), while the same linear classifier on the raw coordinates
is stuck near chance (≈0.5 — it can only draw one straight boundary). One
random-feature layer = an explicit RBF-kernel machine.

Coded by Claude (AI).
