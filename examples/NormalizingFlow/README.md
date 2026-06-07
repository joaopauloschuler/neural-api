# Normalizing Flow (exact-likelihood density estimation)

Fits a 2-D **two-moons** density with a small RealNVP/Glow-style
**normalizing flow** built from stacked `TNNetAffineCoupling` layers — the
library's first exact-likelihood generative primitive — and shows that
interleaving the couplings with Glow's **learnable invertible 1x1 convolution**
(`TNNetInvertible1x1Conv`) reaches a **higher** mean log-likelihood than the
fixed-permute baseline on the same target.

## What a flow is

A normalizing flow is an **invertible** neural network `F` that maps a data
point `x` to a latent `z` whose distribution is a simple base (here a unit
Gaussian). Because every layer is a bijection with a tractable Jacobian, the
change-of-variables formula gives the **exact** log-likelihood of the data —
no ELBO, no adversary, no sampling approximation:

```
log p_X(x) = log p_Z(F(x)) + sum_layers log|det J|
           = -0.5*||z||^2 - 0.5*D*log(2*pi) + sum_couplings(LogDetJacobian)
```

## The affine coupling layer

`TNNetAffineCoupling` splits the channel (Depth) axis into two halves. One half
passes through **unchanged** and conditions an affine transform of the other:

```
forward:  y_a = x_a ;  y_b = x_b * exp(s) + t      (s, t = conditioner(x_a))
inverse:  x_a = y_a ;  x_b = (y_b - t) * exp(-s)
```

* The log-scale `s` is **tanh-clamped** (Glow's stability trick) and the
  per-call sum of `s` over the transformed channels is exposed as the public
  `LogDetJacobian` property — exactly the Jacobian log-determinant.
* The map is **analytically invertible**, so the same trained weights run
  forward (density) and backward (sampling). Pass `pInverse=true` to the
  constructor for the sampling direction.
* The constructor flag `pTransformSecond` chooses which half is transformed;
  stacking couplings that **alternate** the flag updates every channel.
* During training the `-sum(s)` (negative log-det) gradient is **folded into
  each layer's backward pass** (`LogDetLossWeight=1`, the default), so the whole
  flow trains end-to-end under one maximum-likelihood objective by injecting
  only the data-loss gradient `dL/dz = z`.

## The learnable invertible 1x1 convolution (the real Glow step)

The alternating `pTransformSecond` flag is a **fixed** channel permutation — the
network cannot adapt how channels are mixed. Glow replaces it with a **learnable
`C x C` matrix `W`** applied per spatial position across the Depth axis,
`y[x,y,:] = W * x[x,y,:]` (`TNNetInvertible1x1Conv`):

* `W` is parametrized by its LU decomposition `W = P * L * (U + diag(s))` with
  `P` a **fixed** permutation chosen at init, `L` unit-lower-triangular, `U`
  strictly-upper-triangular, and `s` the log-scale vector. This makes the
  per-position log-det the **cheap** `sum(log|s|)` (`O(C)`) instead of a full
  `O(C^3)` determinant.
* `LogDetJacobian` returns `SizeX*SizeY*sum(log|s|)` for the current forward, so
  it composes additively with the couplings' log-dets.
* The map is exactly invertible via triangular solves (no matrix inverse); pass
  `pInverse=true` for the sampling direction, reusing the same `L/U/s`.

## What this example does

1. Builds a **baseline** flow of 6 alternating affine couplings (fixed permute)
   and a **glow** flow with a learnable `TNNetInvertible1x1Conv` between each
   consecutive pair of couplings.
2. Trains both by maximum likelihood on two-moons mini-batches (pure SGD, CPU).
   The glow flow uses a smaller learning rate plus light weight decay because the
   unconstrained matrix + the `-1/s` log-det gradient are less forgiving.
3. Prints the **mean log-likelihood climbing** for each, then reports that the
   learnable mixing reaches a **higher** final log-likelihood.
4. Builds the inverse (sampling) flow of the glow model with the same weights,
   draws `z ~ N(0, I)`, and pushes them through `z -> x` to **generate** new
   points on the data manifold.
5. Round-trips real points `x -> z -> x` to confirm exact invertibility.

## Running

```
lazbuild NormalizingFlow.lpi
../../bin/x86_64-linux/bin/NormalizingFlow
```

Runs in a few minutes on 2 cores. Expected: both mean log-likelihoods rise from
roughly `-2.6`, the **glow flow ends higher** than the fixed-permute baseline
(e.g. `-1.16` vs `-1.56`), the forward->inverse reconstruction error is ~0 (exact
bijection), and the generated-sample statistics match the data cloud.
