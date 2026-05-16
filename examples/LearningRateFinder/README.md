# LearningRateFinder

A self-contained implementation of Leslie Smith's **learning-rate range
test** ("LR finder", from
[*Cyclical Learning Rates for Training Neural Networks*, 2017](https://arxiv.org/abs/1506.01186))
applied to the hypotenuse toy problem `y = sqrt(x1^2 + x2^2)` with
`x1, x2 ~ U(0, 1)`.

The point of the LR-range test is to discover, *before* committing to a
real training run, the order of magnitude of learning rate at which the
network actually learns.  You sweep the learning rate exponentially from
absurdly small to absurdly large across a handful of mini-batches and
plot the resulting (smoothed) loss versus `log10(LR)`.  The curve has a
characteristic shape:

```
loss
 |  __________            <-- LR too small: nothing happens
 |             \
 |              \         <-- steepest descent: the useful LR
 |               \____
 |                    \__/\___/\
 |                              \    <-- LR too large: loss explodes
 +-------------------------------> log10(LR)
```

A reasonable rule of thumb is to pick a learning rate on the order of
the point of **steepest negative slope** of the smoothed curve (some
practitioners prefer roughly one decade below the explosion point; both
land in the same neighbourhood for this toy problem).

## What this example does

- Builds a tiny MLP: `2 -> 16 -> 16 -> 1` with ReLU hidden layers and
  a linear head.
- Generates one fresh mini-batch of size 32 per step (input pairs are
  drawn i.i.d. from `[0, 1]^2`).
- Sweeps the learning rate exponentially from `1e-6` to `1e+1` across
  `100` mini-batches.
- Trains with manual mini-batch SGD: `ClearDeltas` -> per-sample
  `Compute` + `Backpropagate` -> `UpdateWeights`.  This is the only
  way to change the LR every single step, which `TNeuralFit` doesn't
  allow.
- Smooths the per-batch loss with an exponential moving average
  (`beta = 0.98`) and applies the standard bias correction
  `EMA / (1 - beta^t)`.
- Prints a CSV-style table to stdout and to `lr_finder.csv`.
- Renders a 60-column ASCII chart of `log10(LR)` versus smoothed loss
  and marks the steepest-descent row with `*`.
- Reports the suggested LR (the LR at the row of steepest negative
  slope, after a short warm-up window).
- Stops early if the smoothed loss diverges (currently `> 4x` best) or
  becomes `NaN/Inf`.

The whole thing runs single-threaded on the CPU and finishes in a
second or two.

## Files

- `LearningRateFinder.lpr` - the program.
- `LearningRateFinder.lpi` - the Lazarus project file (mirrors
  `examples/Hypotenuse/Hypotenuse.lpi`).
- `lr_finder.csv` - written at runtime; columns
  `step, log10_lr, raw_loss, smoothed_loss`.

## Build & run

Using `fpc` directly (the simplest path):

```bash
cd examples/LearningRateFinder
mkdir -p build
fpc -O2 -Mobjfpc -Sh -Fu../../neural/ -FUbuild/ LearningRateFinder.lpr
./LearningRateFinder
```

Using Lazarus / `lazbuild` if you have it installed:

```bash
cd examples/LearningRateFinder
lazbuild LearningRateFinder.lpi
../../bin/x86_64-linux/bin/LearningRateFinder
```

## Reading the output

Each row of the CSV / table is one mini-batch:

```
step  log10(LR)      raw_loss   smooth_loss
  0    -6.0000      1.123456      1.441717
  1    -5.9293      0.987654      1.288452
  ...
```

The ASCII chart that follows plots one mini-batch per line, with
`log10(LR)` on the left and the smoothed loss bar laid out across 60
columns.  The single `*` marks the row used as the LR suggestion:

```
  -2.889 |                                         *                  | 0.2507
```

The last line of the output is the takeaway:

```
Suggested LR (steepest descent): 0.00129  (log10 = -2.89, slope = -0.099 / decade)
```

For this toy problem with seed `25557` you should see a flat plateau
out to `log10(LR) ~ -3`, a clean descent through `log10(LR) ~ -3` to
`-1`, and divergence shortly after `log10(LR) = -0.5`.  The suggestion
lands near `1e-3`, which is the right ballpark for a real training run
on the same model.

## Caveats

- The "best" LR coming out of the range test is an *upper-bound
  estimate* meant for use with a warm-up / one-cycle schedule.  For a
  flat learning rate, divide it by 3-10x.
- A single sweep is noisy; for serious use, average a handful of runs
  with different seeds.
- The example deliberately uses raw inputs in `[0, 1]` and no
  normalisation so that the divergence point is well-separated from
  the useful range on a log scale.
