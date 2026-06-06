# SIREN: periodic-activation coordinate-MLP 1D fit

A tiny, pure-CPU reproduction of the headline of Sitzmann et al. 2020,
*"Implicit Neural Representations with Periodic Activation Functions"* (SIREN),
using the existing `TNNetSin` activation. No new layer class is added.

## The idea

An implicit neural representation maps a coordinate `x` to a signal value
`f(x)`. A plain ReLU/Tanh coordinate-MLP has a strong **spectral bias**: it
learns low-frequency content quickly but struggles to represent fine
high-frequency detail. Swapping the hidden activation for a plain **sine**
(`TNNetSin`) and initializing so the pre-activations land in the sine's useful
regime removes that bias — the *same* width/depth net fits high-frequency
content far better.

## Target

```
y = sin(3x) + 0.3*sin(11x),   x in [-1, 1]
```

The `0.3*sin(11x)` term is the fine detail a tiny Tanh MLP fights to reproduce
while a SIREN of the same size captures it.

## Bake-off

The **same** architecture, width, depth, seed, epoch budget, batch size,
learning rate and optimizer is trained twice; only the hidden activation
differs:

```
Arm SIREN : Input(1) -> [FullConnectLinear(24) -> TNNetSin]x3                 -> FullConnectLinear(1)
Arm TANH  : Input(1) -> [FullConnectLinear(24) -> TNNetHyperbolicTangent]x3   -> FullConnectLinear(1)
```

Both report final dense-grid MSE; the headline is that the SIREN arm reaches a
substantially lower MSE.

## Initialization / scaling choice (important)

SIREN's whole trick is its init. The paper uses first-layer weights
`~ U(-1/n, 1/n)` with the pre-activation multiplied by a frequency
`omega_0 ~ 30`, and hidden layers `~ U(-sqrt(6/fan_in)/omega_0, +sqrt(6/fan_in)/omega_0)`.

The library does not expose this init directly, but `TNNetLayer.InitUniform(s)`
sets every weight `~ U(-s, +s)`, which is exactly the building block needed.
After the default `InitWeights()`, this example reproduces the scheme **by
hand** (`ApplySirenInit`):

- The input dim is `n = 1`, so the paper's first-layer `U(-1/n, 1/n) = U(-1, 1)`.
  We fold the `omega_0` frequency straight into that first layer by initializing
  it to `U(-omega_0, +omega_0)` — mathematically identical to "`U(-1,1)` weights,
  then multiply the pre-activation by `omega_0`", and needs no input rescaling or
  extra layer.
- Each subsequent sine-feeding layer gets `U(-sqrt(6/fan_in)/omega_0, +...)`.

We use `omega_0 = 12` (a touch below the paper's 30; with `x in [-1,1]` and this
tiny net, 12 keeps the high-frequency advantage while staying numerically calm
and deterministic). The Tanh arm uses the library's default `InitWeights()` — a
fair, conventional baseline. Biases stay at their default (0) for both arms.

## Self-gate

Prints final MSE for both arms and asserts the SIREN arm's MSE is meaningfully
lower (`< 0.5 * Tanh MSE` **and** `< 0.05` absolute). `Halt(1)` on failure,
mirroring `examples/DeepSets`, `examples/MaxBlurPool` and
`examples/BitLinearBakeoff`.

## Build & run

```
lazbuild SIREN.lpi
../../bin/x86_64-linux/bin/SIREN
```

or directly with fpc:

```
fpc -B -Fu../../neural -Mobjfpc -Sh -O2 SIREN.lpr && ./SIREN
```

Pure CPU, single-threaded, deterministic (fixed `RandSeed`), runs in ~12s.

## Sample output

```
RESULT (final dense-grid MSE, lower is better):
  TANH  baseline arm : 0.053259
  SIREN (TNNetSin)   : 0.000016
  SIREN reduces MSE by 100.0% vs the Tanh baseline.
GATE: PASS - the periodic-activation (SIREN) arm fits the high-frequency target substantially better than the Tanh baseline.
```
