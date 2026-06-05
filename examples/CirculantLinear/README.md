# CirculantLinear — structured-matrix dense layer parameter-efficiency bake-off

This example showcases **`TNNetCirculantLinear`**, a structured-matrix dense
layer whose square weight matrix `W` (n × n, n = Depth) is **circulant**: every
row is a cyclic shift of a single learned length-n kernel `c`. The layer's map is
the circular convolution of `c` with the input,

```
y[i] = bias[i] + sum_k c[(i-k) mod n] * x[k]
```

so it stores only **O(n)** weights (the kernel `c` plus an optional length-n
bias) instead of the **O(n²)** of a full dense layer.

This is genuinely distinct from the other structured dense layers in the library:
LoRA is *low-rank*, `AddGroupedFullConnect` is *block-diagonal*,
`TNNetBitLinear` *quantizes* a full matrix and `TNNetSpectralNorm` *rescales*
one — none impose a shift-invariant Toeplitz/circulant structure.

## The bake-off

The regression target is a **genuine circular convolution** of the input with a
fixed, unknown-to-the-model teacher kernel `c_true` (plus a small per-output
bias). This is exactly the function class a circulant layer represents, so it
can fit the task with just `2n` learned numbers. Two models train head-to-head
on the same data:

| model | layer | trainable weights |
|-------|-------|-------------------|
| circulant (structured) | `TNNetCirculantLinear(n)` | `2n` (kernel `c` + bias) |
| dense (param-matched)  | `TNNetFullConnectLinear(n)` | `n·n + n` |

The headline metric is **accuracy-per-weight** = `(1 / test-MSE) / params`.
Both models drive the test MSE to ~0, but the circulant layer gets there with an
order of magnitude fewer weights, so its accuracy-per-weight is dramatically
higher. The example also prints the **recovered kernel vs the teacher kernel**,
showing the circulant layer learns the true circular-convolution operator
essentially exactly.

## Running

```
cd examples/CirculantLinear
fpc -Mobjfpc -Sh -O3 -Fu../../neural -dRelease -dAVX2 CirculantLinear.lpr
./CirculantLinear
```

(or open `CirculantLinear.lpi` in Lazarus). Pure CPU, single thread, finishes in
well under a second — comfortably under the 5-minute budget.

## Representative output (n = 16)

```
Trainable parameters:
  circulant  TNNetCirculantLinear(16) :    32   (= 2n)
  dense      TNNetFullConnectLinear(16):   272   (= n*n + n)

  model                              params       test-MSE       acc/weight
  circulant (structured)                 32      2.15E-015        1.46E+013
  dense (param-matched)                 272      1.33E-013        2.76E+010
  Circulant uses 8.5x FEWER weights and is 526.7x more ACCURATE-PER-WEIGHT.
```

The recovered kernel matches the teacher kernel to ~5 decimal places.

## Opt-in FFT fast path (`UseFFT`)

By default the forward and backward passes use the direct **O(n²)** circular sum
(clear and easy to verify against finite differences). Setting

```pascal
CL := TNNetCirculantLinear.Create(n);  // n MUST be a power of two for FFT
CL.UseFFT := true;                     // opt-in O(n log n) path
```

switches **both the forward and the backward** to a frequency-domain
implementation backed by a self-contained **radix-2 Cooley-Tukey FFT**:

- forward: `y = IFFT( FFT(c) .* FFT(x) ) + bias`,
- input gradient: `dx = IFFT( conj(FFT(c)) .* FFT(e) )` (a circular correlation),
- kernel gradient: `dc = IFFT( conj(FFT(x)) .* FFT(e) )`, bias gradient `= e`.

This is O(n log n) instead of O(n²). The direct path stays the **default** and is
the numerical source of truth; a test (`TestCirculantLinearFFTEquivalence` in
`tests/TestNeuralNumerical.pas`) asserts the FFT path reproduces the direct
forward output, input gradient and kernel/bias deltas to **< 1e-5** on a random
kernel/input/error. The radix-2 transform requires `n` to be a power of two;
`UseFFT` errors clearly otherwise (disable it to use the direct path for any
`n`). The FFT works in double precision internally so the round-trip stays
faithful even when `TNeuralFloat` is single precision. A direct-vs-FFT
wall-clock chart as `n` grows is not included here.

Coded by Claude (AI).
