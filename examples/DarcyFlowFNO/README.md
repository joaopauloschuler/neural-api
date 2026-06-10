# DarcyFlowFNO — a PDE surrogate with the Fourier Neural Operator

This example demonstrates the **headline use case** of the **Fourier Neural
Operator** (FNO; Li et al. 2021, *"Fourier Neural Operator for Parametric
PDEs"*, arXiv:2010.08895): learning a parametric-PDE **coefficient → solution
map** in a single forward pass, replacing an iterative numerical solve. It is
built with the library builder **`TNNet.AddFourierNeuralOperator2D`** on top of
`TNNetSpectralConv2D`.

## The operator we learn

On the periodic unit square we learn the linear, FNO-tractable member of the
Darcy-flow family — a **Poisson solve whose source is driven by a
spatially-varying coefficient field** `a(x,y)`:

```
    -Laplacian u(x,y) = (a(x,y) - 1)        (zero-mean source, periodic BCs)
```

i.e. the operator `G : a(x,y) -> u(x,y)`. This is a genuine, resolution-invariant
elliptic **solution operator**: in Fourier space its Green's function is the
smooth low-pass gain `1/|k|^2`, which the spectral conv's learnable low-mode
complex weights reproduce almost exactly. Because those weights live in **mode
space, not grid space**, the same weights describe the same continuous operator
at any resolution.

### Data generation (pure Pascal, no external dependency)

For each sample (seeded deterministically by index, so the *same continuous*
`a` is reproducible and re-samplable on any grid):

1. `a(x,y) = exp( smooth band-limited random field )` — a strictly-positive
   permeability in roughly `0.5 .. 2`, built from a few low Fourier modes.
2. `u(x,y)` is the deterministic **solution** of `-Laplacian u = (a-1)`, computed
   by Jacobi sweeps of the 5-point finite-difference Laplacian with periodic
   wrap-around. The grid spacing `h` is folded in (the source is multiplied by
   `h^2`) so the discrete operator approximates the **same continuous PDE** at
   every resolution — this is what makes the resolution-transfer headline
   meaningful. The iteration count scales with the grid (`3·L·L` sweeps) so the
   solve converges at every resolution.

## Architecture

A canonical FNO built with the library builder:

```
Input(L, L, 1)
  -> TNNetPointwiseConvLinear(WIDTH)              // lift  1 -> WIDTH channels
  -> AddFourierNeuralOperator2D(WIDTH, 6, 6, nil) // spectral conv + 1x1 residual
  -> TNNetPointwiseConvLinear(1)                  // project WIDTH -> 1
```

`AddFourierNeuralOperator2D(pOutDepth, pModesX, pModesY, pActFn)` adds the
canonical FNO block `y = Act( SpectralConv2D(x) + W_1x1(x) )` — a spectral branch
(low-pass 2-D mode mix with learnable complex weights) plus a per-pixel pointwise
residual, summed. The target operator is **linear**, so we pass `pActFn = nil`
(no activation). `WIDTH = 8`, `MODESX = MODESY = 6`.

Training is plain mini-batch SGD with momentum (MSE loss, `LR = 0.02`,
mini-batch 12, 150 epochs, 96 training fields).

## Headlines

* **HEADLINE 1 — the FNO learns the operator.** Held-out relative-L2 error drops
  steadily during training.
* **HEADLINE 2 — resolution invariance.** The SAME trained weights are evaluated
  with **no retraining** on a finer `32×32` grid generated from the same
  continuous operator (weights copied across with `CopyWeights`). The error stays
  bounded, because the spectral conv's weights are resolution-independent.

## Running

```
cd examples/DarcyFlowFNO
lazbuild DarcyFlowFNO.lpi
../../bin/x86_64-linux/bin/DarcyFlowFNO
```

(or open `DarcyFlowFNO.lpi` in Lazarus). Pure CPU, small grids / few modes /
modest epochs — finishes in about **3 minutes** on 2 cores and uses little
memory. The data-generation Jacobi solve (especially the `32×32` evaluation set)
dominates the runtime. No binaries are committed.

> **Grid must be a power of two.** `TNNetSpectralConv2D` uses a separable radix-2
> FFT, so both spatial dimensions must be powers of two (here `16` and `32`). A
> non-power-of-two grid trips the FFT assertion.

## Representative output

```
Darcy-flow FNO surrogate: learning the coefficient a(x,y) -> solution u(x,y)
of  -Laplacian u = (a-1)  (periodic Poisson), via AddFourierNeuralOperator2D.
Train grid 16x16, eval grid 32x32, modes 6x6, width 8

Training FNO surrogate (96 coefficient fields)...
  epoch    train relL2    test relL2
      0        0.6408        0.6387
     25        0.0578        0.0597
     50        0.0297        0.0298
     75        0.0268        0.0266
    100        0.0261        0.0260
    125        0.0257        0.0255
    149        0.0252        0.0251

=== Held-out relative-L2 error of the learned solution operator ===
  test on TRAINED grid (16x16)         : 0.0251
  test on FINER UNSEEN grid (32x32)     : 0.0284
```

* **HEADLINE 1:** held-out relative-L2 error falls from `≈0.64` to `≈0.025` — the
  FNO has learned the coefficient → solution operator.
* **HEADLINE 2:** the trained weights transfer to the unseen finer grid with
  **bounded** error (`0.0251` at `16×16` → `0.0284` at `32×32`), confirming the
  resolution-invariance of mode-space spectral weights.

## Honest scope note

The **fully nonlinear** Darcy operator `-div(a·grad u) = f` (where `u` depends on
`a` through the inverse of the `a`-weighted Laplacian) is a *non-diagonal*,
nonlinear-in-`a` map. In this fork's pure-CPU setting with plain SGD it does
**not** converge within the ~4-minute budget — a single linear spectral conv
cannot represent it, and deeper / activated FNO stacks train far too slowly
(the FFT-domain weight gradients move glacially at a learning rate that keeps the
pointwise branches stable). We therefore use the **linear Poisson member**
`-Laplacian u = (a-1)`, which is a genuine elliptic PDE solution operator (smooth,
diagonal-in-mode-space Green's function), trains cleanly, and showcases both FNO
headlines honestly. Extending to the nonlinear operator would need either a
longer training budget, per-layer learning-rate scaling for the spectral weights,
or normalization layers between FNO blocks.

Coded by Claude (AI).
