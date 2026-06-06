# Dead-ReLU learning-rate sweep

This is the learning-rate-sweep follow-up to
[`DeadReLUDiagnostic`](../DeadReLUDiagnostic). Where that demo **pins one
aggressive learning rate** (LR = 0.5) and watches plain ReLU strand ~19% of its
hidden units, this demo **sweeps the learning rate over a grid** and, for each
LR, trains the same small classifier with each of `TNNetReLU` /
`TNNetLeakyReLU` / `TNNetGELU` / `TNNetSwish`. The output is a table — rows are
LRs, columns are activations, cells are the peak dead-unit fraction — that
isolates the learning rate as the cause of dying ReLU.

## The phenomenon

A hidden unit is **dead** when its activation output stays at (numerically)
zero for *every* probe sample: such a unit produces a zero gradient and can
never recover. Plain ReLU is prone to this, and the larger the learning rate,
the more readily a unit's pre-activation is shoved negative for all inputs and
stuck there. `LeakyReLU` / `GELU` / `Swish` keep a non-zero output (and
gradient) for negative inputs, so essentially none of their units die at any LR.

The dead-unit fraction is measured exactly as `TNNet.DeadNeuronReport` defines
it (see `neural/neuralnetwork.pas`): push every probe sample through the net
and, for each hidden unit, track the maximum absolute activation; a unit is
dead if that maximum is `<= 1e-6`. Here we read the number directly per epoch
(across the two hidden activation layers) and keep the **peak** over the
trajectory.

## What the sweep shows

Two hidden `FullConnectLinear(64)` blocks + softmax head, a shared synthetic
3-Gaussian-blob task, 60 epochs per arm, `RandSeed := 424242` reseeded
identically before every `(LR, activation)` arm so the only differences are the
LR and the activation. Observed peak dead hidden-unit fraction:

| LR    | TNNetReLU | TNNetLeakyReLU | TNNetGELU | TNNetSwish |
|-------|-----------|----------------|-----------|------------|
| 0.02  | 10.16%    | 0.00%          | 0.00%     | 0.00%      |
| 0.05  | 11.72%    | 0.00%          | 0.00%     | 0.00%      |
| 0.10  | 13.28%    | 0.00%          | 0.00%     | 0.00%      |
| 0.20  | 14.06%    | 0.00%          | 0.00%     | 0.00%      |
| 0.50  | 17.19%    | 0.00%          | 0.00%     | 0.00%      |

Read down the `TNNetReLU` column: the dead fraction climbs monotonically from
10.16% to 17.19% as the LR grows. The other three columns stay flat at 0% across
the entire sweep. That is the cleanest statement of the point: **dying ReLU is
a learning-rate pathology**, not an intrinsic property of the task.

## Self-gate

The program halts with exit code 1 unless both invariants hold:

1. ReLU's dead fraction at the highest LR exceeds its dead fraction at the
   lowest LR by a meaningful margin (the monotone-ish climb).
2. At the highest LR, ReLU strands strictly more units than each of LeakyReLU,
   GELU and Swish.

Both hold robustly with the numbers above (gap 10.16% -> 17.19%).

## How this differs from `DeadReLUDiagnostic`

| | `DeadReLUDiagnostic` | `DeadReLULRSweep` (this demo) |
|---|---|---|
| Learning rate | **one** fixed LR (0.5) | a **grid** of LRs |
| Output | per-epoch dead-fraction trajectory for 4 activations | LR x activation table of peak dead fractions |
| Question answered | *how* the four activations differ at a hard LR | *that the LR itself* is what kills ReLU units |

## Build and run

```
cd examples/DeadReLULRSweep
fpc -O3 -Mobjfpc -Sh -Fu../../neural -dRelease DeadReLULRSweep.lpr
./DeadReLULRSweep
```

If the build fails with `Can't find unit UTF8Process`, add the LazUtils unit
path and the C-threads define (mirrors the `DeadReLUDiagnostic` build):

```
fpc -O3 -Mobjfpc -Sh -Fu../../neural \
    -Fu/usr/share/lazarus/<ver>/components/lazutils/lib/x86_64-linux \
    -dUseCThreads -dRelease DeadReLULRSweep.lpr
```

Pure CPU, single-threaded, fully reproducible, finishes in ~11 seconds.
