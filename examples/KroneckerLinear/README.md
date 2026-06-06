# Kronecker structured-linear parameter/accuracy bake-off

This example showcases **`TNNetKroneckerLinear`**, a sub-quadratic **structured**
dense layer. The `n x n` weight matrix is a single **Kronecker product** of two
small learned factors:

```
W = A (x) B          A is p x p,  B is q x q,  n = p*q
y = W x  (+ bias)
```

The dense `n x n` matrix is **never materialized**. The flat input `x` (length
`n`) is reshaped to a `q x p` matrix `X` with `X[i,j] = x[i*p + j]`, and the
matvec is the two small GEMMs

```
Y = B * X * A^T          y[i*p + j] = Y[i,j]
```

which costs `O(n*(p+q)) = O(n^1.5)` time. Under this row-major `vec` convention
`y = vec(B X A^T)` exactly equals `(A (x) B) x`. Backward is the exact transpose
chain — `dX = B^T dY A`, `dA = dY^T (B X)`, `dB = dY (X A^T)^T` — all small
GEMMs, every gradient numerically gradient-checked.

The layer stores only `p^2 + q^2` factor weights (`≈ 2n` for the square case
`p = q = sqrt(n)`) instead of the dense `n^2`. The square map **infers `n` from
the previous layer** (no `N` arg): the constructor is `Create(pSuppressBias, pP)`
where `pP = 0` auto-picks the factor split (`p = round(sqrt(n))`, or the largest
divisor `<= sqrt(n)` when `n` is not a perfect square) and `q = n div p`.

## The bake-off

Three square `256 -> 256` mixing layers are trained on a tiny **MNIST-shaped**
10-class task — each `16x16` sample is one of 10 fixed random class prototypes
plus per-pixel Gaussian noise — each followed by an **identical** head
(`ReLU -> linear(10) -> softmax`). We print the trainable weight count of just
the mixing layer and its final train/test accuracy:

| Arm | Mixer | Mixing weights |
| --- | --- | --- |
| Kronecker (structured) | `TNNetKroneckerLinear` (p=q=16) | **768** (`p^2 + q^2 + n`) |
| Dense baseline | `TNNetFullConnectLinear(256)` | 65536 (`n^2`) |
| Monarch (structured) | `TNNetMonarchLinear` (b=m=16) | 8448 (`2*n*sqrt(n) + n`) |

## Headline result

```
=== Mixing-layer weight count vs accuracy (train / test) ===
  Kronecker  :    768 weights   train 1.000   test 1.000
  Dense FC   :  65536 weights   train 1.000   test 1.000
  Monarch    :   8448 weights   train 1.000   test 1.000

HEADLINE: Kronecker uses 85x FEWER mixing weights than the dense
arm yet reaches comparable accuracy.
```

The Kronecker arm uses **85x fewer** mixing weights than the dense arm yet
reaches the same test accuracy — *structured = far fewer params, comparable
accuracy*. Monarch is an intermediate structured point (`O(n^1.5)` weights) on
the params-vs-accuracy curve.

Pure CPU, single-threaded-friendly, tiny dataset, **runs in a few seconds**. No
binaries or datasets are committed.

## Build & run

```
lazbuild KroneckerLinear.lpi
./KroneckerLinear      # or wherever lazbuild dropped the binary
```
