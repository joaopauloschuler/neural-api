# WaveletDenoise — Donoho-Johnstone wavelet shrinkage with `TNNetDWT1D`

This example showcases **`TNNetDWT1D`**, the lifting-scheme single-level 1-D
discrete wavelet transform, used for the classic **Donoho-Johnstone wavelet
shrinkage** denoiser (Donoho & Johnstone 1994, *"Ideal spatial adaptation by
wavelet shrinkage"*). It also exercises the new **`TNNet.AddWaveletPacketTransform`**
builder.

## The idea

A `TNNetDWT1D` over a `(SeqLen, 1, Depth)` sequence maps one level
`(L, 1, D) → (L div 2, 1, 2·D)`: the first `D` channels are the **approximation**
(low-pass) band, the next `D` are the **detail** (high-pass) band. Forward and
inverse share one lifting step list, so it is **exactly invertible** for any taps
(`InverseChannel` reconstructs a channel from packed `[approx | detail]` bands).

The denoiser:

1. **Decompose** — run a *Mallat pyramid*: apply the single-level DWT (Haar)
   repeatedly to the **approximation band only** for `LEVELS` levels, collecting
   one detail band per level.
2. **Shrink** — **soft-threshold** the detail coefficients at every level with
   the universal threshold `λ = σ · sqrt(2 ln M)`, where `σ` is robustly
   estimated **per level** by the MAD estimator `median(|d|)/0.6745` and `M` is
   that level's band length (per-level because the unnormalised lifting rescales
   the coefficients between levels).
3. **Reconstruct** — invert the pyramid level by level with
   `TNNetDWT1D.InverseChannel`.

The test signal is the canonical Donoho-Johnstone **"Blocks"** signal — a
piecewise-constant staircase of signed steps with **sharp edges** — corrupted by
white Gaussian noise. Blocks is **sparse in the wavelet domain** (the jump energy
collapses onto a few large detail coefficients that survive the threshold) but
**dense in the local-average domain**, so it is the textbook case where wavelet
shrinkage beats a linear low-pass filter.

## Baseline

A param-free **moving-average low-pass filter** (window 9). Being a *single fixed
scale*, it must blur the very edges that define the signal — it cannot separate
"sharp edge" from "noise" the way a multi-resolution basis can.

| method | basis | keeps sharp edges? |
|--------|-------|--------------------|
| **wavelet soft-threshold** | `TNNetDWT1D` multi-resolution (Haar) | **yes** |
| baseline | moving average (single fixed scale) | no |

## The `AddWaveletPacketTransform` builder

The example also prints the shape of the network-builder companion:

```pascal
NN.AddWaveletPacketTransform({Levels=}5, {Filter=}csDWT1DHaar);
// (1024,1,1) -> (32,1,32)
```

`AddWaveletPacketTransform(Levels, Filter, Learnable)` stacks `Levels`
single-level `TNNetDWT1D` layers. Because each DWT transforms **every** input
channel, re-applying it recursively decomposes **both** the approximation and the
detail subbands — i.e. the **full balanced wavelet-packet tree**
(Coifman-Wickerhauser), *not* the Mallat pyramid. After `Levels` levels the shape
is `(SeqLen div 2^Levels, 1, (2^Levels)·Depth)`, a single dense tensor with no
channel split/concat plumbing. (The denoiser above instead recurses the
approximation band only — the standard *decomposition* tree — which is done by
hand precisely because it is a different, asymmetric recursion.)

## Running

```
cd examples/WaveletDenoise
lazbuild WaveletDenoise.lpi
../../bin/x86_64-linux/bin/WaveletDenoise
```

(or open `WaveletDenoise.lpi` in Lazarus). Pure CPU, a 1024-sample signal, no
training — finishes in **well under a second** and uses almost no memory. No
binaries are committed.

## Representative output

```
1-D wavelet shrinkage denoising demo (Donoho-Johnstone soft threshold)
Signal length = 1024, levels = 5, filter = Haar, noise sd = 0.30

AddWaveletPacketTransform(5) packet tree: (1024,1,1) -> (32,1,32)

=== Reconstruction SNR (dB, higher is better) ===
  noisy input                          18.29 dB
  low-pass baseline (moving avg 9)     16.60 dB
  wavelet soft-threshold (DWT)         21.20 dB

  wavelet gain over noisy    :   2.91 dB
  wavelet gain over low-pass :   4.60 dB
```

The wavelet shrinkage reaches **21.2 dB**, beating the moving-average baseline by
**+4.6 dB** and the noisy input by **+2.9 dB**: the multi-resolution basis keeps
the sharp block edges that the single-scale low-pass filter is forced to blur.

Coded by Claude (AI).
