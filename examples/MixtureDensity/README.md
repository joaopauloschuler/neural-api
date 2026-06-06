# Mixture Density Network: recovering a one-to-many inverse map

A tiny self-contained demo of a **Mixture Density Network** (Bishop 1994,
[*Mixture Density Networks*](https://publications.aston.ac.uk/id/eprint/373/))
on the classic multi-valued inverse-mapping problem. It shows that a probabilistic
mixture head recovers the **several valid branches** of an inverse map while a plain
mean-squared-error (MSE) regression head **collapses to the gap** between them.

## The problem

Data is generated from the forward map

```
x = y + 0.3·sin(2π·y) + small noise,    y ~ Uniform(0,1)
```

and we try to learn the **inverse**: predict `y` given `x`. The forward map is
non-monotonic, so for many `x` there are **several** valid `y` branches. A plain
MSE regressor must predict a *single* `y`, and its least-squares optimum is the
conditional **mean** of those branches — a value that falls in the gap between
them and that the true process never produces. A Mixture Density Network instead
predicts a full conditional distribution `p(y|x)` as a mixture of Gaussians, so
it can put one component on each branch.

## What it does

Both models share the same small trunk `x → 32 → 32`:

* **MDN**: trunk → `TNNetFullConnectLinear(1,1,K·(1+2·D))` → `TNNetMixtureDensity(K, D)`
  with `K = 5` components over a `D = 1` target. The head turns the trunk output
  into `(π, μ, σ)` (softmax mixing weights, raw means, softplus scales) and **owns
  the negative-log-likelihood loss**; its `Backpropagate` emits the exact
  responsibility-weighted `dNLL/dparam`. The `SampleMixture` helper draws samples
  at inference (pick a component by `π`, then sample that diagonal Gaussian).
* **MSE baseline**: trunk → `TNNetFullConnectLinear(1)`, trained with plain squared
  error to the same targets.

**Training** is hand-rolled mini-batch maximum likelihood (8000 updates, batch 64,
well under five minutes on two CPU cores). For the MDN we build the target volume
so its first `D` channels hold the true `y` (the head recovers `y` from there) and
scale the accumulated batch-update deltas to the **mean** gradient — the clean mean
gradient is what stops a single component from collapsing and lets the trunk learn.
The MDN head's mean biases are initialised spread across the target range to break
symmetry. The analytic backward was checked against finite differences in
`tests/TestNeuralNumerical.pas` (`TestMixtureDensityGradient`).

## Example output

```
MixtureDensity: one-to-many inverse map  x = y + 0.3*sin(2*pi*y) + noise
  K=5 Gaussian components, D=1 target, trunk 32->32->15

Training mixture density network (maximum likelihood)...
  [MDN] epoch    1   avg NLL=  0.1549
  [MDN] epoch 8000   avg NLL=  0.0990
Training MSE regression baseline...
  [MSE] epoch    1   avg MSE=0.119727
  [MSE] epoch 8000   avg MSE=0.024315

==== INVERSE-MAP RECOVERY ON THE MULTI-VALUED REGION ====
x=0.450  true y branches: 0.179 0.559 0.750
   MDN components (pi: mu):  0.53:0.298  0.47:0.830  0.00:-1.065  0.00:-2.653  0.00:-1.663
   MDN samples: 0.715 0.244 -0.034 -0.075 0.834 0.679 0.325 0.480
   MSE prediction: 0.567   (dist to nearest branch=0.007)
   branch-coverage error:  MDN means=0.153   MSE point=0.193   (avg MDN-sample dist=0.118)
...
==== VERDICT ====
  mean branch-coverage error   MDN component means=0.110   MSE single point=0.195
  The MDN places a component on EACH branch (low coverage error); the single MSE
  point cannot, so it collapses to the gap-filling mean.
```

The MDN spreads its active components across the valid `y` branches, so its
**branch-coverage error** (mean distance from each true branch to the nearest
active component mean) is roughly **half** that of the single MSE point, which can
only ever sit near one branch (and lands in the gap on average). Samples drawn
from the MDN scatter across the branches; the MSE prediction is a single value.
The whole run takes about 25 s on 2 CPU cores.

## Build & run

```
lazbuild MixtureDensity.lpi
../../bin/x86_64-linux/bin/MixtureDensity
```

## Related layer

* `TNNetMixtureDensity` — the probabilistic regression head used here: maps the
  previous layer's `K·(1+2·D)` raw outputs to a `K`-component diagonal-Gaussian
  mixture over a `D`-dim target, owns the mixture NLL loss, and provides a
  `SampleMixture` inference helper.
```
