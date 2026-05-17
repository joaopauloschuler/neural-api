# SeededReproducibility

Tiny CI-style check that confirms a training run is fully
deterministic when re-seeded. The program trains the same small
network twice with `RandSeed := 42` before each run, then compares
*every* trainable weight and bias element-by-element. Anything other
than an exact match exits with code `1`.

## What it shows

Reproducibility is easy to lose: an unseeded RNG path, a parallel
reduction with non-deterministic order, or a stray global state can
all make "the same training" produce different weights on the second
run. This example pins that down with the strictest possible probe:

- same `RandSeed`,
- same data generation,
- same architecture,
- single fit worker (`NFit.MaxThreadNum := 1`) so reductions are
  order-deterministic,

and asserts **max abs diff == 0** across all 105 trainable scalars
(weights + biases) in the tiny ReLU MLP:

```
TNNetInput(2)
  -> TNNetFullConnectReLU(8)
  -> TNNetFullConnectReLU(8)
  -> TNNetFullConnectLinear(1)
```

trained for 3 epochs on 500 synthetic hypotenuse samples.

## Build & run

```
lazbuild SeededReproducibility.lpi
../../bin/x86_64-linux/bin/SeededReproducibility
```

Pure CPU, runs in well under a second.

## Expected output sketch

```
SeededReproducibility: training twice with RandSeed=42...
Run 1 of 2:
  ...fit logs...
Run 2 of 2:
  ...fit logs...

Weights compared : 105
Mismatching      : 0
Max abs diff     : 0.00000000000000000000
PASS: bit-for-bit identical weights across runs.
```

On FAIL the process exits with code `1`, so this binary can be plugged
straight into a CI job as a guard against future non-determinism
regressions.

## Note on threading

The fit is run single-threaded on purpose. With `MaxThreadNum > 1`,
parallel gradient accumulation can sum partials in a non-deterministic
order, which makes bit-for-bit equality fail even when every RNG call
is seeded. If you remove the `NFit.MaxThreadNum := 1` line, expect
FAIL on multi-core machines.
