# Euclidean-Norm-Reciprocal Head (composing Square / Sqrt / Reciprocal)

This example shows that the in-tree, parameter-free elementwise transcendental
layers compose into a recognisable analytic function: the **reciprocal of the
Euclidean (L2) norm** of a vector,

```
1 / ||x||_2  =  Reciprocal( Sqrt( Sum_i( Square(x_i) ) ) )
```

Every piece already exists as a layer:

| Layer             | maps              | derivative           | guard        |
|-------------------|-------------------|----------------------|--------------|
| `TNNetSquare`     | `x -> x^2`        | `2x`                 | none         |
| `TNNetSqrt`       | `s -> sqrt(s)`    | `1/(2y)`             | `s>=1e-6`    |
| `TNNetReciprocal` | `n -> 1/n`        | `-y^2`               | `|n|>=1e-6`  |

The only non-elementwise step is the **sum over the features**. We get an
*exact* sum from an existing layer too: a `TNNetFullConnectLinear(1)` whose single
neuron has all-ones weights and zero bias computes `y = sum_i x_i`.

## Layer composition (shapes; `F` features on the Depth axis)

```
Input(1,1,F)                       x                      (1,1,F)
  -> TNNetSquare                   x_i^2                  (1,1,F)
  -> TNNetFullConnectLinear(1)     sum_i x_i^2  [W=1,b=0] (1,1,1)
  -> TNNetSqrt                     ||x||_2                (1,1,1)
  -> TNNetReciprocal               1/||x||_2              (1,1,1)
```

The sum layer is frozen after `InitWeights()`:

```pascal
SumLayer.Neurons[0].Weights.Fill(1.0);  // all-ones -> exact sum over F
SumLayer.ClearBias();                    // zero bias
SumLayer.LearningRate := 0.0;            // never disturbed by training
```

### Why the all-ones FullConnectLinear (and not a channel pool)

The repo's channel pools are a trap for this job. `TNNetAvgChannel` pools over the
`X*Y` plane (**not** the Depth axis where our vector lives) and, on an `(N,1,F)`
bag, divides by `PoolSize^2` rather than `PoolSize` — so its scale is not a clean
sum. A `FullConnectLinear(1)` with weights filled to `1.0` and `ClearBias()` is an
**exact** sum over the `F` features with no hidden scaling, and its gradient (a
fan of ones) flows straight through. That is exactly what a reduction should do,
which is why this demo uses it.

## What the example verifies (self-checking gate, `Halt(1)` on failure)

1. **Forward match** — for 200 random vectors, the composed head output equals the
   analytic `1/sqrt(sum x_i^2)` to within `1e-4` (observed: exact, `0.00e+000`).
2. **Unit-norm sanity** — feeding `e_0` (norm 1) returns `~1.0`.
3. **L2-normalize extension** — scaling `x` by the head's reciprocal yields
   `x/||x||`, whose own L2 norm must equal 1 (observed `~1.2e-7`). This is the
   natural use of the head as the reciprocal factor of a full L2 normalizer.
4. **Gradient flow** — a downstream `FullConnectLinear(1)` regresses the analytic
   `1/||x||` target through the `Square/Sum/Sqrt/Reciprocal` stack; the MSE must
   drop at least 10x with no NaN, proving the backward chain rule flows.

## Build & run

With Lazarus:

```
lazbuild EuclideanNormHead.lpi
```

Directly with FPC (pure CPU, single-threaded, deterministic, < 1 s):

```
fpc -B -Fu../../neural -Mobjfpc -Sh -O2 EuclideanNormHead.lpr && ./EuclideanNormHead
```

## Sample output

```
EuclideanNormHead: composing Reciprocal(Sqrt(Sum(Square(x)))) = 1/||x||_2.
Head: Input(1,1,8) -> TNNetSquare -> FullConnectLinear(1)[W=1,b=0 sum] -> TNNetSqrt -> TNNetReciprocal -> (1,1,1)

Check 1 - forward match vs analytic 1/||x||_2 over 200 vectors:
  max abs error  = 0.00E+000
  mean abs error = 0.00E+000
  example: head=0.186520  analytic=0.186520
  ...
Check 2 - unit-norm input e_0 -> head = 1.000000 (want ~1.0)
Check 3 - L2-normalize extension: max | ||x*(1/||x||)|| - 1 | = 1.19E-007
Check 4 - gradient flow: MSE 0.115135 -> 0.002109 over 600 steps (NaN seen: False)
GATE: PASS - composed Reciprocal(Sqrt(Sum(Square(x)))) matches 1/||x||_2, ...
```

## Why `TNNetL2Normalize` exists

This composed head is a **teaching artifact**: it demonstrates that the
transcendental layers chain into a recognisable analytic function, and that the
gradient flows end-to-end. For production L2 normalization, reach for the
dedicated `TNNetL2Normalize` layer instead. It:

- applies the **exact Jacobian** `(I - y y^T) / n` over a chosen axis (per-depth,
  per-channel or full-volume) in **one fused backward pass**, rather than chaining
  three separate elementwise backward passes;
- carries a tunable **epsilon guard** (`FFloatSt[0]`, default `1e-8`) that
  round-trips through `SaveToString` / `LoadFromString`;
- normalizes the vector directly (`x/||x||`), whereas the composed head only
  produces the scalar reciprocal `1/||x||` and leaves the broadcast multiply to
  you.

The composed head leans on the per-layer `1e-6` clamps of `Sqrt`/`Reciprocal` and
is correct and instructive — but it is not the fused, axis-aware, save-safe
primitive you would wire into a real model.
