# HouseholderOrthogonal — exact orthogonality kills deep-stack gradient blow-up

This example demonstrates `TNNetHouseholderLinear`, an **exactly-orthogonal**
dense layer. Its `n × n` weight is parameterized as a product of `K` Householder
reflections

```
Q = H_1 · H_2 · … · H_K,   H_i = I − 2·(v_i·v_iᵀ) / (v_iᵀ·v_i)
```

so `Q` is orthogonal for **any** reflection vectors `v_i` — no constrained
optimization, no re-projection, no normalization. An orthogonal Jacobian is an
isometry (`‖Q·x‖ = ‖x‖` with bias off, `‖Qᵀ·g‖ = ‖g‖`), and a stack of these
layers (with no nonlinearity) has an end-to-end Jacobian that is *still*
orthogonal. A gradient pushed back from the top of the stack therefore reaches
the input with its norm **unchanged at any depth**.

## What it does

For depths `D = 1, 2, 4, 8, 16, 32` it builds two deep plain linear stacks of
width `n = 16` (no activation, no normalization, no residuals):

- **(A)** `D ×` `TNNetHouseholderLinear` (exactly orthogonal blocks)
- **(B)** `D ×` `TNNetFullConnectLinear` (unconstrained dense blocks, mildly
  scaled so their spectral radius sits a touch above 1 — a realistic init)

It plants a unit gradient at the top, backpropagates, and prints the L2 norm of
the gradient that reaches the **input**.

```
  depth | Householder (K=n, orthogonal) | FullConnectLinear (unconstrained)
  ------+-------------------------------+----------------------------------
      1 |                    1.000000   |                    1.324771
      2 |                    1.000000   |                    1.219459
      4 |                    1.000000   |                    1.286019
      8 |                    1.000000   |                    9.783570
     16 |                    1.000000   |                   35.279118
     32 |                    1.000000   |                 1046.688232
```

The Householder column stays **exactly 1.0** at every depth; the unconstrained
column explodes geometrically with depth (it would collapse toward 0 for a
spectral radius below 1). This is exactly the exploding/vanishing-gradient
pathology that exact orthogonality removes.

## K vs cost / expressivity

A second sweep varies the number of reflections `K ∈ {1, n/2, n}` at depth 32:

- Cost is `O(K·n)` per layer — linear in `K`.
- The gradient norm is preserved for **every** `K`, because any product of
  reflections is exactly orthogonal. `K = n` spans the full orthogonal group
  `O(n)`; `K < n` is a cheaper sub-group with less representational reach but the
  **same** exact gradient stability.

So `K` trades expressivity for compute, not gradient stability. Use the
`TNNet.AddHouseholderLinear(N, NumReflections, UseBias)` builder; `NumReflections
<= 0` defaults to `K = n`.

## Build & run

```
lazbuild HouseholderOrthogonal.lpi
../../bin/x86_64-linux/bin/HouseholderOrthogonal
```

CPU-only, no data files, runs in well under a minute.

## Why it matters

Exact orthogonality is the building block for orthogonal/unitary RNNs (no
exploding/vanishing gradients across long sequences) and for invertible,
norm-preserving blocks in normalizing flows and reversible networks. Unlike
`TNNetSpectralNorm` (which only bounds the *largest* singular value) or the
structured-matrix layers (which constrain the matrix *form*), this layer makes
**all** singular values exactly 1.
