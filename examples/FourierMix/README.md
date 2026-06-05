# FourierMix — FNet parameter-free Fourier token mixer vs self-attention

This example showcases **`TNNetFourierMix`**, the FNet-style
**parameter-free** token mixer of Lee-Thorp et al. 2021,
*"FNet: Mixing Tokens with Fourier Transforms"*. Over a `(SeqLen, 1, d)`
sequence tensor it replaces self-attention with an **unparameterised** 2D
discrete Fourier transform applied across the sequence and hidden axes, keeping
only the real part:

```
y = Re( DFT_seq( DFT_hidden( x ) ) )
y[a,b] = sum_{s,h} x[s,h] * cos( 2*pi*(a*s/L + b*h/D) )
```

The layer owns **no trainable weights at all** — mixing is a fixed linear
operator. Because `Re(DFT)` is a fixed **real** linear operator `M` with
`M[(a,b),(s,h)] = cos(2*pi*(a*s/L + b*h/D))`, and `M` is symmetric under
swapping the output `(a,b)` index with the input `(s,h)` index, the adjoint
`M^T` equals `M`. Hence the exact input gradient is the **same** real 2D-DFT
operator applied to `dL/dy` — clean and verified against finite differences.

## The bake-off

The regression target is a tiny `8`-token × `8`-dim sequence task where each
output token must combine information from **every** input token (a fixed global
mixing of the sequence followed by a smooth `tanh` nonlinearity). Solving it
*requires* token mixing — a position-wise MLP alone cannot do it. Two
equal-depth models train head-to-head on the same data:

| model | mixer | followed by |
|-------|-------|-------------|
| FNet | `TNNetFourierMix` (**0** mixing weights) | shared per-token MLP |
| attention | `AddMultiHeadSelfAttention(2)` (Q\|K\|V + out projection) | the same per-token MLP |

The headline makes the FNet **expressiveness-vs-cost trade** concrete on a short
sequence: dropping the entire learned mixing block for a fixed Fourier basis
removes the attention model's whole Q\|K\|V\|out projection (hundreds of weights,
~half the model) and trades only a modest amount of accuracy — the paper's
selling point being that on short sequences the mix is nearly free.

## Running

```
cd examples/FourierMix
fpc -Mobjfpc -Sh -O3 -Fu../../neural -dRelease -dAVX2 FourierMix.lpr
./FourierMix
```

(or open `FourierMix.lpi` in Lazarus). Pure CPU, single thread, finishes in
about 13 seconds — comfortably under the 5-minute budget.

## Representative output (SeqLen = 8, d = 8)

```
Total trainable parameters:  FNet=280   Attention=568
  (the Fourier mixer itself adds ZERO parameters; the attention block adds 288)

  model                            params       test-MSE     train-secs
  FNet (Fourier, free mix)            280      1.08E-001          10.09
  attention (learned mix)             568      4.40E-002           2.29
  FNet drops the entire learned mixing block: 288 FEWER parameters (51% smaller).
  test-MSE ratio (fnet/attn) = 2.46: the FIXED Fourier basis trades some accuracy
  for ZERO mixing weights.
```

## Opt-in FFT fast path (`UseFFT`)

By default both forward and backward use the direct **O(n²)** DFT sum, which
works for **arbitrary** `SeqLen` and `d` and is the numerical source of truth.
Setting

```pascal
FM := TNNetFourierMix.Create;  // SeqLen and d MUST be powers of two for FFT
FM.UseFFT := true;             // opt-in separable radix-2 FFT path
```

switches the mixing to a separable 2D FFT (FFT along the hidden axis, then along
the sequence axis, keep the real part), backed by a self-contained radix-2
Cooley-Tukey FFT in double precision. A test
(`TestFourierMixFFTMatchesDirect` in `tests/TestNeuralNumerical.pas`) asserts the
FFT output reproduces the direct output to **< 1e-5**; `UseFFT` errors clearly if
either axis is not a power of two (disable it to use the direct path for any
shape). `UseFFT` round-trips through `SaveToString` / `LoadFromString`.

Coded by Claude (AI).
