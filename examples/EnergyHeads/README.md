# EnergyHeads

Side-by-side comparison of two final-feature "energy" heads on a
tiny regression task:

- `TNNetAbs`    -> L1-energy head (elementwise `|x|`)
- `TNNetSquare` -> L2-energy head (elementwise `x^2`)

Both networks share an identical body and the same training data,
so the final test-MSE gap isolates the effect of the energy layer.

## Architecture

```
TNNetInput(4)
  -> TNNetFullConnectReLU(16)
  -> TNNetFullConnectReLU(8)
  -> [ TNNetAbs | TNNetSquare ]    # the only difference
  -> TNNetFullConnectLinear(1)     # regression head
```

## Target task

The target is the Euclidean norm of the input vector:

```
y = sqrt(x0^2 + x1^2 + x2^2 + x3^2)
```

with `x0..x3` drawn uniformly from `[-1, 1]`. This is a natural fit
for an energy head: `TNNetAbs` exposes per-feature magnitudes
(L1-style), and `TNNetSquare` exposes per-feature squared values
(L2-style) that a linear head can combine to approximate the
squared norm.

## Build & run

```
lazbuild EnergyHeads.lpi
../../bin/x86_64-linux/bin/EnergyHeads
```

Or directly with fpc:

```
fpc -dRelease -dUseCThreads -O3 -Fu../../neural EnergyHeads.lpr
./EnergyHeads
```

Pure CPU, no external data. Runs in well under three minutes on a
modern machine. The run is non-interactive.

## Expected output sketch

```
=========================================================
Energy head: TNNetAbs    (L1-energy)
=========================================================
... training log ...
=========================================================
Energy head: TNNetSquare (L2-energy)
=========================================================
... training log ...

=========================================================
 Final test-MSE comparison (Euclidean-norm regression)
=========================================================
 Head                      |   Test MSE
 --------------------------+--------------
 TNNetAbs    (L1-energy) |   0.00xx
 TNNetSquare (L2-energy) |   0.00xx

Sample predictions on held-out test data (first 5):
  inputs                            |   target |    Abs |  Square
  ----------------------------------+----------+--------+--------
  ( 0.83, -0.92,  0.05, -0.50) |  1.34xx | 1.34x | 1.33x
  ...
```

Both heads typically converge to test MSE well under 0.01 on this
task. Which one wins depends on seed and optimisation noise; the
point of the example is to see both energy layers train end-to-end
on the same problem.
