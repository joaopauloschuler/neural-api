# OctonionLinear

A small, fast, parameter-matched bake-off demonstrating `TNNetOctonionLinear`,
the 8-dimensional hypercomplex (octonion / Cayley-Dickson) dense layer — the
8-component sibling of `TNNetQuaternionLinear`.

## The layer

`TNNetOctonionLinear` reinterprets the input/output `Depth` (both multiples of
8) as packed **octonions** and learns an `(OutO x InO)` grid of octonion-valued
weights `w = o0 + o1·e1 + ... + o7·e7`. The forward pass is the octonion product
`y = W · X`, i.e. a real 8×8-block matrix where each learned octonion's 8 reals
drive an entire 8×8 block via the fixed octonion multiplication table

```
M(W)[i][j] = SGN[i][j] · W[i xor j]
```

The table is the standard Cayley–Dickson doubling of quaternions,
`(a,b)(c,d) = (a·c − d*·b, d·a + b·c*)`, and is verified independently by the
norm-multiplicativity identity `|W·X| = |W|·|X|` in the unit tests. Because one
octonion's 8 reals control 64 matrix entries, the layer stores **~1/8** the
weights of a plain dense layer of equal real width while still mixing all eight
components of every input octonion (the 8× analogue of the quaternion layer's 4×
saving).

## The task

Each sample is `X` = 2 input octonions (16 real channels). The target
left-multiplies every input octonion by the **same** fixed unit octonion `g`,
plus a small fixed octonion-valued cross coupling between the two octonions.
The target is therefore *exactly* an octonion-linear map, so a model whose
inductive bias matches that structure should win at equal parameter count.

Three param-matched contenders map 16 → 16 channels:

| Model                          | Trainable weights | Notes                              |
|--------------------------------|-------------------|------------------------------------|
| `TNNetOctonionLinear(16)`      | 32 (+16 bias)     | OutO·InO·8 = 2·2·8 octonion reals  |
| `TNNetFullConnectLinear` 16→2→16 bottleneck | 64   | low-rank dense factorisation       |
| `AddGroupedFullConnect(2)`     | 128               | block-diagonal, cannot mix groups  |

A full unstructured 16×16 dense layer would need **256** multiply weights, so
the octonion layer uses 8× fewer.

## Running

```
lazbuild OctonionLinear.lpi
../../bin/x86_64-linux/bin/OctonionLinear
```

Pure CPU, tiny data, ~200 epochs — runs in well under a minute. Expected: the
octonion model reaches a near-zero validation MSE (the task is in its hypothesis
class) while the param-matched dense and grouped baselines cannot, at a fraction
of the parameter budget.

Coded by Claude (AI).
