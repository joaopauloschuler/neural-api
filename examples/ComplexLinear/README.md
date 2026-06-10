# ComplexLinear

A small, fast, parameter-matched bake-off demonstrating `TNNetComplexLinear`,
the 2-dimensional hypercomplex (complex) dense layer — the **base rung** of the
same Cayley–Dickson ladder as `TNNetQuaternionLinear` (4D) and
`TNNetOctonionLinear` (8D).

## The layer

`TNNetComplexLinear` reinterprets the input/output `Depth` (both multiples of 2)
as packed **complex numbers** (group `g` holds `Re = chan[2g]`,
`Im = chan[2g+1]`) and learns an `(OutC x InC)` grid of complex-valued weights
`w = a + b·i`. The forward pass is the complex product `y = w · x`, i.e. a real
2×2-block matrix where each learned complex's 2 reals drive an entire 2×2 block:

```
M(w) = [[a, -b],
        [b,  a]]      ->   Re' = a·Re − b·Im,   Im' = a·Im + b·Re
```

Written in the same table form used by the octonion layer this is
`M(w)[i][j] = CPX_SGN[i][j] · w[i xor j]` with `CPX_SGN = ((1,-1),(1,1))`.
Because one complex's 2 reals control 4 matrix entries, the layer stores
**~1/2** the weights of a plain dense layer of equal real width while still
mixing the real and imaginary parts of every input complex (the 2× analogue of
the quaternion layer's 4× and the octonion layer's 8× saving).

## The task

Each sample is `X` = 2 input complex numbers (4 real channels). The target
left-multiplies every input complex by the **same** fixed complex `g` — a pure
**phase rotation** (50°) with gain 1.3 — plus a small fixed complex-valued cross
coupling between the two complex numbers. The target is therefore *exactly* a
complex-linear map, so a model whose inductive bias matches that structure
should win at equal parameter count.

Three param-matched contenders map 4 → 4 channels:

| Model                          | Trainable weights | Notes                              |
|--------------------------------|-------------------|------------------------------------|
| `TNNetComplexLinear(4)`        | 8 (+4 bias)       | OutC·InC·2 = 2·2·2 complex reals   |
| `TNNetFullConnectLinear` 4→1→4 bottleneck | 8      | low-rank dense factorisation       |
| `AddGroupedFullConnect(2)`     | 8                 | block-diagonal, cannot mix groups  |

A full unstructured 4×4 dense layer would need **16** multiply weights, so the
complex layer uses 2× fewer.

## Norm / phase guarantee

After the bake-off the example verifies the algebraic identity the layer is
built on — a single complex multiply scales magnitude by `|w|` and rotates phase
by `arg(w)`:

```
|w · x| = |w| · |x|        arg(w · x) = arg(w) + arg(x)
```

exactly the way the `OctonionLinear` example checks octonion norm
multiplicativity.

## Running

```
lazbuild ComplexLinear.lpi
../../bin/x86_64-linux/bin/ComplexLinear
```

Pure CPU, tiny data, ~200 epochs — runs in well under a minute. Expected: the
complex model reaches a near-zero validation MSE (the task is in its hypothesis
class) while the param-matched dense and grouped baselines cannot, at a fraction
of the parameter budget.

Coded by Claude (AI).
