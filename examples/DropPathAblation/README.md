# DropPath / Stochastic-Depth ablation

This example sweeps the **DropPath** (Stochastic Depth) probability `p ∈ {0.0, 0.1, 0.2}`
on a small ResNet-style classifier and prints the final train/test loss and
accuracy for each arm. It uses only existing in-tree layers — the `TNNetDropPath`
layer already lives in `neural/neuralnetwork.pas`; no new layer is added.

## The phenomenon

Stochastic Depth ("DropPath", Huang et al. 2016, *Deep Networks with Stochastic
Depth*) is a residual-network regulariser. Each residual block computes

```
y = x + Branch(x)
```

`TNNetDropPath(p)` sits on the residual **branch**, right before the closing
`Sum`. At **training** time it zeroes the *entire branch for a whole sample*
with probability `p` (and rescales the survivors by `1/(1-p)` — inverted dropout,
so the expected magnitude is preserved). At **inference** the branch is always
kept and the layer is the exact identity. Dropping the branch turns that block
into a plain skip for that sample, so the network's *effective depth varies
sample-to-sample* during training. Like dropout, it is a **regulariser**: it
tends to help on hard / over-fitting-prone problems and is often neutral — or
mildly hurtful — on tiny easy ones.

## What this example builds

```
Input(6) -> FullConnectLinear(16) -> Reshape(1,1,16)
          -> 6 × [ residual block ]
          -> FullConnectLinear(3) -> SoftMax
```

Each residual block is wired manually so the DropPath layer lands on the branch:

```pascal
BranchInput := NN.GetLastLayer();
NN.AddLayer( TNNetPointwiseConvLinear.Create(16) );   // branch, shape-preserving over Depth
NN.AddLayer( TNNetReLU.Create() );
NN.AddLayer( TNNetDropPath.Create(DropProb) );        // stochastic depth on the branch
NN.AddLayer( TNNetSum.Create([NN.GetLastLayer(), BranchInput]) );
```

The residual sublayer is `TNNetPointwiseConvLinear` (preserves the
`1×1×Width` shape so the `TNNetSum` skip is valid — `TNNetFullConnectLinear`
would flatten the tensor and break the add). The feature vector lives in the
**Depth** axis, which is why the projection is followed by a `Reshape(1,1,16)`.

The task is a fixed 3-way classification: a random quadratic teacher
`class = argmax_c (xᵀQ_c x + b_cᵀx)` over 6-D Gaussian inputs. The teacher and
both datasets are generated once, so all three arms see identical data and
identical weight initialisation (same `RandSeed`); only the DropPath
probability and its train-time RNG draws differ.

`p = 0.0` is the **plain no-drop baseline**: `TNNetDropPath(0)` is the identity
at both train and inference, so that arm is exactly the unregularised residual
net.

## Observed results

Single-threaded, `RandSeed := 424242`, 40 epochs, ~12 s wall-clock on CPU:

```
   p   | initTrnLoss  finalTrnLoss  trainAcc | testLoss  testAcc | diverged
  -----+-----------------------------------------+-------------------+---------
  0.00 |      8.5230        0.3750   0.7734 |   3.0274  0.8030 | no
  0.10 |      8.5230        0.5001   0.7734 |   2.8012  0.8030 | no
  0.20 |      8.5230        0.5000   0.7734 |   2.3684  0.8030 | no
```

### Did DropPath help? (honest read)

On this small/easy toy DropPath did **not** improve test *accuracy* — all three
arms land at the same 0.8030 held-out accuracy (the argmax decision boundary
barely moves). This is the expected, honest outcome for a regulariser on an
easy task, and the example says so rather than pretending otherwise (same spirit
as the SAM / OptimizerBakeoff README caveats).

There is, however, a real and consistent regularisation signal in the held-out
**loss**: it falls monotonically as `p` grows (`3.03 → 2.80 → 2.37`). DropPath
makes the network *less over-confident* on the test set — it lowers the
cross-entropy even where it does not flip enough predictions to change accuracy.
The training loss rises slightly with `p` (`0.375 → 0.500`), as expected when
you regularise away train-set fit.

## Self-check (gate)

The program `Halt(1)`s unless all of these hold (invariants that are actually
true, not a brittle "more DropPath always generalises better" claim):

1. **No arm diverges** — every loss is finite (no NaN/Inf).
2. **Every arm trains** — final train loss < initial (random-init) loss.
3. **The `p=0.0` baseline learns** — train accuracy well above the `1/3` chance
   level, confirming the no-drop residual net is a healthy classifier.

Evaluation always calls `NN.EnableDropouts(false)` first, so DropPath is the
identity and the reported numbers are deterministic at inference.

## Build & run

```
cd examples/DropPathAblation
fpc -O3 -Mobjfpc -Sh -Fu../../neural -dRelease -dUseCThreads DropPathAblation.lpr
./DropPathAblation
```

`-dUseCThreads` pulls in the cthreads driver that `neuralfit`'s worker pool
needs (the fit is still single-threaded via `NFit.MaxThreadNum := 1`, which
keeps the reductions deterministic). If `fpc` cannot find `UTF8Process` (used by
`neuralthread`), add the LazUtils unit path, e.g.
`-Fu/usr/share/lazarus/<ver>/components/lazutils/lib/x86_64-linux`.
