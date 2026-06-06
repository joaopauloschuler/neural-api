# LogCosh Dual Experiment

Pairs `TNNetLogCoshActivation` (hidden layer) with the matching
`TNNetLogCoshLoss` output head vs. a plain MSE head on the toy
hypotenuse task `y = sqrt(X^2 + Y^2)`. Both runs share the same
data, weight init (`RandSeed := 42`), and 2 -> 32 -> 1 MLP; only
the output head differs.

Inputs are normalized by 100 and the target by 200; the regression
metrics (MSE, RMSE, MAE, max error, R²) are reported in the original
target scale (output multiplied back by 200). `LogCosh` is the mean
`log(cosh(residual))` on the normalized scale the network actually
optimizes. Both a held-out validation set and a separate test set
are scored, and each configuration's training wall time is recorded.

## How to run

```bash
cd examples/LogCoshDualExperiment
lazbuild LogCoshDualExperiment.lpi
../../bin/x86_64-linux/bin/LogCoshDualExperiment
```

## Sample output (50 epochs, 1000 train pairs)

A wide CSV table (one row per configuration) is followed by a
per-config human-readable summary:

```
=== Results (CSV) ===
config,val_mse,val_rmse,val_mae,val_max_err,val_r2,val_logcosh,test_mse,test_rmse,test_mae,test_r2,epochs_to_converge,total_epochs,seconds
LogCoshHidden+PlainMSE,25.8285,5.0822,3.8627,16.7810,0.9688,0.000323,19.2160,4.3836,3.2797,0.9730,NA,50,23.39
LogCoshHidden+LogCoshLoss,18.8089,4.3369,3.4007,13.1568,0.9793,0.000235,19.0206,4.3613,3.2870,0.9744,NA,50,170.33

=== Per-config summary ===
LogCoshHidden+PlainMSE:
  validation : MSE=25.8285  RMSE=5.0822  MAE=3.8627  maxErr=16.7810  R2=0.9688  LogCosh=0.000323
  test       : MSE=19.2160  RMSE=4.3836  MAE=3.2797  R2=0.9730
  converged  : NA (MSE < 5.00 not reached in 50 epochs)
  train time : 23.39 s
LogCoshHidden+LogCoshLoss:
  validation : MSE=18.8089  RMSE=4.3369  MAE=3.4007  maxErr=13.1568  R2=0.9793  LogCosh=0.000235
  test       : MSE=19.0206  RMSE=4.3613  MAE=3.2870  R2=0.9744
  converged  : NA (MSE < 5.00 not reached in 50 epochs)
  train time : 170.33 s
```

Columns: `*_mse` / `*_rmse` / `*_mae` are the mean-squared, root-mean-
squared and mean-absolute error in original hypotenuse units;
`val_max_err` is the worst single-sample absolute error; `*_r2` is the
coefficient of determination (1.0 is perfect); `val_logcosh` is the
mean `log(cosh(residual))` on the normalized scale; `epochs_to_converge`
is the first epoch validation MSE crossed below 5.0 (`NA` if never).

`NA` means neither configuration crossed the MSE < 5 threshold in
50 epochs. The two heads land in the same ballpark — the LogCosh
head's `tanh(residual)` gradient is bounded in `[-1, 1]`, so it
trades raw convergence speed for robustness to outliers (note the
lower `val_max_err` for the LogCosh head). Exact numbers vary a
little with platform / float build.
