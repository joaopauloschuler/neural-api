# LogCosh Dual Experiment

Pairs `TNNetLogCoshActivation` (hidden layer) with the matching
`TNNetLogCoshLoss` output head vs. a plain MSE head on the toy
hypotenuse task `y = sqrt(X^2 + Y^2)`. Both runs share the same
data, weight init (`RandSeed := 42`), and 2 -> 32 -> 1 MLP; only
the output head differs.

Inputs are normalized by 100 and the target by 200; reported MSE
is in the original target scale (output multiplied back by 200).

## How to run

```bash
cd examples/LogCoshDualExperiment
lazbuild LogCoshDualExperiment.lpi
../../bin/x86_64-linux/bin/LogCoshDualExperiment
```

## Sample output (50 epochs, 1000 train pairs, ~113 s wall)

```
=== Results (CSV) ===
config,final_val_mse,epochs_to_converge,total_epochs
LogCoshHidden+PlainMSE,22.7371,NA,50
LogCoshHidden+LogCoshLoss,24.2739,NA,50
```

`NA` means neither configuration crossed the MSE < 5 threshold in
50 epochs. The two heads land in the same ballpark — the LogCosh
head's `tanh(residual)` gradient is bounded in `[-1, 1]`, so it
trades raw convergence speed for robustness to outliers.
