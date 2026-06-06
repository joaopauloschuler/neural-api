# Soft Decision Tree vs matched MLP (two-moons)

A tiny, pure-CPU demo of the new **`TNNetSoftDecisionTree`** layer — a single
differentiable **soft (oblique) decision tree** (Kontschieder et al. 2015,
*Deep Neural Decision Forests*; Frosst & Hinton 2017, *Distilling a Neural
Network Into a Soft Decision Tree*). This is a structurally new paradigm for the
library: **hierarchical soft routing**, distinct from matrix factorization,
attention, recurrence and kernel methods.

## The layer

A balanced binary tree of depth `D` has `2^D - 1` inner nodes and `2^D` leaves.

- Each inner node `i` is a learnable linear gate producing a routing probability
  `p_i = sigmoid(beta * (w_i . x + b_i))` (probability of going **left**); `beta`
  is a fixed inverse-temperature hyperparameter.
- A sample reaches leaf `l` with probability `P_l = product` of the left/right
  gate decisions (`p_i` left, `1 - p_i` right) along its root-to-leaf path.
- Each leaf `l` holds a learnable output vector `phi_l` (length `OutputDepth`).
- The output is the path-probability-weighted mixture `y = sum_l P_l * phi_l`.

The backward pass is **exact** (no diagonal/approximate path): the
product-of-gates path probabilities give clean analytic node responsibilities.
For inner node `i`, with left/right subtree responsibility sums `A_i`/`B_i`, the
`p_i / (1 - p_i)` divisions cancel against `dp_i/dz_i = beta*p_i*(1-p_i)` and the
node gradient collapses to

```
dL/dz_i = beta * ( A_i*(1 - p_i) - B_i*p_i )
```

Finite-difference verified for the input gradient, the gate weights/biases AND
the leaf vectors (`TestSoftDecisionTree*` in `tests/TestNeuralNumerical.pas`).

## The demo

A non-linearly-separable **two-moons** binary classification. We compare, behind
an **identical softmax** head and on the **same raw 2-D input**:

- **Soft tree:** `TNNetSoftDecisionTree(D=3, OutputDepth=2, beta=2.0)` — 7 inner
  gates (2 weights + 1 bias each) + 8 leaf logit-vectors (2 each) = **37
  parameters**.
- **Matched MLP:** `ReLU(7) -> linear(2)` = `3*7 + (7+1)*2` = **37 parameters**.

Both are trained with manual mini-batch softmax cross-entropy (the framework
seeds the softmax error with `pred - one_hot`), mean-gradient batch updates, a
few thousand steps. Runs in ~15 s on two CPU cores.

## Headline result

```
Tree parameters: 37   MLP parameters: 37 (matched)
...
=== Held-out accuracy (4000 fresh samples) ===
  Soft decision tree :  99.93%
  Matched MLP        :  95.05%
  HEADLINE: the soft tree TIES OR BEATS the matched MLP.

=== Interpretable decision path (tree only) ===
  probe point ( 0.80,  0.60) -> predicted class 0
    node 0: p(left)=0.060 -> RIGHT
    node 2: p(left)=0.937 -> LEFT
    node 5: p(left)=0.928 -> LEFT
    => dominant leaf 4, logits [ 2.083, -2.054]
  probe point ( 0.20, -0.20) -> predicted class 1
    node 0: p(left)=0.171 -> RIGHT
    node 2: p(left)=0.084 -> RIGHT
    node 6: p(left)=0.979 -> LEFT
    => dominant leaf 6, logits [-1.619,  1.475]
```

At matched capacity the soft tree ties or beats the MLP on this toy **and** —
unlike the MLP — exposes a **human-readable decision path**: which gate sends the
point left/right, with what confidence, down to the dominant leaf and its class
logits. (Exact numbers vary slightly with the RNG seed.)

## Build & run

```
cd examples/SoftDecisionTree
lazbuild SoftDecisionTree.lpi
../../bin/x86_64-linux/bin/SoftDecisionTree
```
