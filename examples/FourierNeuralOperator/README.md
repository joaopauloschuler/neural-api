# FourierNeuralOperator — learning a 1-D PDE solution operator with `TNNetSpectralConv1D`

This example showcases **`TNNetSpectralConv1D`**, the core layer of the
**Fourier Neural Operator** (Li et al. 2021,
*"Fourier Neural Operator for Parametric PDEs"*, arXiv:2010.08895). It is the
**first** layer in neural-api with **learnable complex spectral weights**,
distinct from the parameter-free `TNNetFourierMix` (fixed `Re(DFT)` token mixer)
and the fixed-random `TNNetFourierFeatures`.

Over a `(SeqLen, 1, InDepth)` sequence the layer performs, per channel:

1. a real radix-2 **FFT** along `SeqLen` (reusing the proven `FourierMixFFT`
   radix-2 Cooley-Tukey helper — not a second hand-rolled FFT);
2. a spectral **low-pass truncation** to the lowest `Modes` frequency modes
   (high modes are zeroed, never learned);
3. per kept mode `m`, a learnable per-`(in-channel, out-channel)` **complex
   weight** `R[m]` — an `InDepth × OutDepth` complex matmul mixing the real/imag
   parts via the 2×2 complex-multiply block (the same hypercomplex
   weight-packing idiom as `TNNetQuaternionLinear` / `TNNetOctonionLinear`);
4. an inverse **FFT** back to the `SeqLen` domain.

Because the learned weights live in **mode space**, not grid space, the *same*
weights describe the *same* continuous operator at **any** resolution — this is
the property that makes an FNO **resolution-invariant**.

## The task

We learn the **antiderivative operator** of a 1-D periodic PDE,
`(G f)(x) = ∫₀ˣ f(t) dt` (i.e. `u' = f`, a 1-D Poisson / diffusion-step style
linear solution operator). Inputs are random smooth band-limited functions
`f = Σₖ aₖ cos(2πkx + φₖ)`; the exact target is
`u = Σₖ aₖ/(2πk) · sin(2πkx + φₖ)`.

The headline experiment:

* both models are trained **only** on a coarse `32`-point grid;
* both are then evaluated, **with no retraining**, on a finer `64`-point grid
  they never saw (weights copied across with `CopyWeights`).

| model | mixer | resolution-invariant? |
|-------|-------|-----------------------|
| **FNO** | 2× `TNNetSpectralConv1D` (mode-space weights) + pointwise lifts | **yes** |
| baseline | 2× `TNNetCausalConv1D` (local grid-spacing taps) + pointwise lifts | no |

## Running

```
cd examples/FourierNeuralOperator
fpc -Mobjfpc -Sh -O3 -Fu../../neural -dRelease -dAVX2 FourierNeuralOperator.lpr
./FourierNeuralOperator
```

(or open `FourierNeuralOperator.lpi` in Lazarus). Pure CPU, finishes in about
**90 seconds** — comfortably under the 5-minute budget. No binaries are
committed.

## Representative output

```
=== Relative L2 error on held-out functions ===
               train grid (32)   eval grid (64, UNSEEN)
  FNO                0.0792            0.0786
  local conv         0.6329            0.8569
```

The **FNO keeps essentially the same error on the finer, never-trained grid**
(7.9% → 7.9%), because its mode-space weights are resolution invariant. The
local-conv baseline, whose 1-D taps encode a fixed grid spacing, **degrades
sharply** when the grid is refined (63% → 86%) — exactly the contrast an
ordinary local convolution stack cannot avoid.

Coded by Claude (AI).
