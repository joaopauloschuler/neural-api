# SpectralConv2D — learning a 2-D solution operator with `TNNetSpectralConv2D`

This example showcases **`TNNetSpectralConv2D`**, the **2-D** spectral
convolution of the **Fourier Neural Operator** (Li et al. 2021,
*"Fourier Neural Operator for Parametric PDEs"*, arXiv:2010.08895). It is the
natural 2-D extension of `TNNetSpectralConv1D`: instead of mixing modes along a
single sequence axis it mixes the lowest **2-D** Fourier modes of an image with
learnable **complex** weights.

Over a `(SizeX, SizeY, InDepth)` image the layer performs, per channel:

1. a **2-D FFT** — a real radix-2 FFT along X for every row, then along Y for
   every column (both reuse the proven `FourierMixFFT` radix-2 helper — not a
   second hand-rolled FFT);
2. a spectral **low-pass truncation** to the lowest `ModesX × ModesY` 2-D
   frequency modes (all higher modes are zeroed, never learned);
3. per kept mode `(mx, my)`, a learnable per-`(in-channel, out-channel)`
   **complex weight** `R[mx,my]` — an `InDepth × OutDepth` complex matmul mixing
   real/imag parts via the 2×2 complex-multiply block (the same hypercomplex
   weight-packing idiom as `TNNetQuaternionLinear` / `TNNetOctonionLinear`);
4. an inverse **2-D FFT** (IFFT along Y then X) back to the spatial domain; the
   real part is the output.

Because the learned weights live in **2-D mode space**, not grid space, the
*same* weights describe the *same* continuous operator at **any** resolution —
the property that makes an FNO **resolution-invariant**.

## The task

We learn a smooth **2-D low-pass diffusion operator**: each kept 2-D Fourier
mode of a band-limited source field `f(x,y)` is scaled by a smooth gain
`1 / (1 + c·(kx² + ky²))`. Each training field is generated directly in 2-D mode
space (with Hermitian symmetry, so the sampled grid is real), so both the field
and its target live entirely inside the kept low band — the operator is exactly
representable by the spectral conv and identical on any grid.

The headline experiment:

* both models are trained **only** on a coarse `16×16` grid;
* both are then evaluated, **with no retraining**, on a finer `32×32` grid they
  never saw (weights copied across with `CopyWeights`).

| model | mixer | resolution-invariant? |
|-------|-------|-----------------------|
| **2-D FNO** | `TNNetSpectralConv2D` (2-D mode-space weights) + pointwise lifts | **yes** |
| baseline | 2× `TNNetConvolutionReLU` 3×3 (local grid-spacing taps) + lifts | no |

## Running

```
cd examples/SpectralConv2D
lazbuild SpectralConv2D.lpi
../../bin/x86_64-linux/bin/SpectralConv2D
```

(or open `SpectralConv2D.lpi` in Lazarus). Pure CPU, small grids and channel
counts, finishes in about **2 minutes** — comfortably under the 5-minute budget
and uses little memory. No binaries are committed.

## Representative output

```
=== Relative L2 error on held-out 2-D fields ===
               train grid (16x16)   eval grid (32x32, UNSEEN)
  2-D FNO              0.0001            0.0001
  local conv          0.2935            0.5768
```

The **2-D FNO keeps essentially zero error on the finer, never-trained grid**
(0.01% → 0.01%), because its 2-D mode-space weights are resolution invariant.
The local-conv baseline, whose 3×3 taps encode a fixed grid spacing, **degrades
sharply** when the grid is refined (29% → 58%) — exactly the contrast an
ordinary local convolution stack cannot avoid.

Coded by Claude (AI).
