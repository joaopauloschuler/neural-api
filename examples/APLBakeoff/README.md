# APL Bake-off (APL vs PReLU vs ReLU)

This example answers a small but concrete question:

> Does the extra piecewise capacity of an **Adaptive Piecewise Linear (APL)**
> activation buy a lower final loss compared to plain **ReLU** and **PReLU**,
> and how does that change as you sweep the number of hinges `S`?

It is a follow-up to `TNNetAPL` (Agostinelli et al. 2015,
<https://arxiv.org/abs/1412.6830>). APL learns, **per channel**, a set of `S`
hinges - each with its own slope and knee - on top of a ReLU:

```
h(x) = max(0, x) + sum_{s=1..S} a[s] * max(0, -x + b[s])
```

so APL adds `2*S` learnable parameters per unit.

## What it does

All arms train the **same** fixed MLP on the hypotenuse regression toy
(predict `sqrt(a^2 + b^2)` from `(a, b)`):

```
Input(2) -> FullConnect(32) -> activation -> FullConnect(32) -> activation -> FullConnectLinear(1)
```

Only the hidden-layer activation changes between arms:

| Arm       | Activation                              | Extra params / unit |
|-----------|-----------------------------------------|---------------------|
| `ReLU`    | `TNNetReLU`                             | 0                   |
| `PReLU`   | `TNNetPReLU` (learnable negative slope) | 1                   |
| `APL S=1` | `TNNetAPL.Create(1)`                    | 2                   |
| `APL S=2` | `TNNetAPL.Create(2)`                    | 4                   |
| `APL S=4` | `TNNetAPL.Create(4)`                    | 8                   |

Each arm reseeds `RandSeed` to the same fixed value before generating its data
and building its net, so the data, weight init and training schedule are
identical across arms - the only difference is the activation. After training,
the example measures the final **train** and **validation** MSE directly and
prints a comparison table together with the trainable-parameter count
(`TNNet.CountWeights`) of each arm.

**On matched param count:** the arms are *not* exactly param-matched - they all
share the same width (32) so the only difference in parameter count comes from
the activation itself. This keeps the architecture identical and isolates the
activation's effect; the param column is printed so the (small) mismatch is
visible and honest rather than hidden. APL's higher count is exactly
`2 * S * Hidden` extra scalars summed over the two hidden activations.

It is tiny on purpose: 2000 train / 500 test samples, 60 epochs, batch 1000.
The whole run finishes in about 2 s on CPU (well under the 5-minute budget) and
uses negligible memory.

## How to run

From this directory, with FPC available:

```
fpc -Mobjfpc -Sh -O2 -Fu../../neural -Fu/usr/share/lazarus/4.4.0/components/lazutils/lib/x86_64-linux APLBakeoff.lpr
./APLBakeoff
```

The second `-Fu` points FPC at the Lazarus `lazutils` units that
`neuralthread` (pulled in by `neuralfit`) depends on; adjust the path to your
Lazarus install (this is the same `LAZUTILS_PATH` the repo's `tests/RunAll.sh`
uses). Or open `APLBakeoff.lpi` in Lazarus and run it.

## Sample output

Real output from the dev machine (FPC 3.2.2, x86_64-linux):

```
APL vs PReLU vs ReLU bake-off on the hypotenuse toy (regression)
Same MLP: Input(2) -> FC(32) -> act -> FC(32) -> act -> FCLinear(1)
2000 train / 500 test samples, 60 epochs, seed=424242

  done: ReLU      train MSE=0.000663  val MSE=0.000620  params=1153
  done: PReLU     train MSE=0.000523  val MSE=0.000492  params=1217
  done: APL S=1   train MSE=0.000546  val MSE=0.000517  params=1281
  done: APL S=2   train MSE=0.000546  val MSE=0.000515  params=1345
  done: APL S=4   train MSE=0.000546  val MSE=0.000536  params=1473

=== Activation bake-off results ===
Arm        Train MSE     Val MSE     Params
-------------------------------------------
ReLU         0.000663    0.000620      1153
PReLU        0.000523    0.000492      1217
APL S=1      0.000546    0.000517      1281
APL S=2      0.000546    0.000515      1345
APL S=4      0.000546    0.000536      1473
-------------------------------------------
Note: APL adds 2*S learnable params per hidden unit (slope+knee per
hinge), so its param count is higher than ReLU/PReLU at equal width.
```

## Reading the result

On this toy and budget the **answer is "not really, not for free"**. Any
learnable activation beats plain `ReLU` (val MSE 0.000620), but the cheapest
one wins: `PReLU` (one extra param per unit) reaches the lowest val MSE
(0.000492). The APL arms all land in a tight band (val MSE ~0.000515-0.000536)
that beats ReLU but does **not** beat PReLU, and adding hinges does not help -
`S=2` is marginally best of the three and `S=4` is marginally worst, i.e. the
extra `2*S` params per unit buy more parameters and a harder optimisation
landscape without buying lower loss. That is exactly the kind of honest
negative result the param column is there to make visible.

All losses are tiny on this easy problem and the differences are small, so
treat this as a sanity-checkable demonstration of the capacity-vs-loss
trade-off rather than a general verdict on APL. APL is designed to shine on
harder, non-convex per-channel response shapes (e.g. image features), not on a
two-input smooth regression. Bump `NUM_EPOCHS`, `HIDDEN`, or the sample counts
at the top of `APLBakeoff.lpr`, or swap in a harder target, to explore further.
