# ReceptiveFieldReport

Demonstrates `TNNet.ReceptiveFieldReport`: a purely **analytical** per-layer
receptive-field (RF) walk of a `TNNet`. No probe batch, no forward pass, no
backward pass — it reads each layer's kernel/stride/padding geometry and
propagates the textbook RF recurrence.

## What it does

1. Builds a plain **VGG-style 3x3 stride-1** stack (32x32 input) and prints
   its per-layer RF table. Each 3x3 conv grows the RF by 2 and the jump stays
   at 1.
2. Builds a **stride-2 downsampling** stack (32x32 input) and prints the same
   table. Every stride-2 layer (and the 2x2 max-pool) **doubles the jump**, so
   the RF grows much faster and covers the whole input far sooner.

No training and no data: the report walks `NN.Layers` and uses each layer's
recorded kernel size, stride and padding.

## The recurrence

Tracked independently for X and Y (so rectangular kernels/strides work):

```
r_k     = r_{k-1} + (kernel_k - 1) * jump_{k-1}     // RF size on the input
jump_k  = jump_{k-1} * stride_k                      // effective stride
start_k = start_{k-1} + ((kernel_k - 1)/2 - pad_k) * jump_{k-1}
```

- **Convolution / Deconvolution** read `FeatureSizeX/Y`, `Stride`, `Padding`.
  A deconvolution that grows the map is treated as a fractional stride.
- **Pool family** (`TNNetMaxPool`, `TNNetAvgPool`, `TNNetMinPool`, ...) read
  `PoolSize`, `Stride`, `Padding`.
- **Global channel reductions** (`TNNetMaxChannel`, `TNNetAvgChannel`,
  `TNNetGlobalSumPool`) span the whole current feature map → RF jumps to the
  full input.
- **Upsample / DeMaxPool / DeAvgPool** divide the jump (fractional stride).
- **Pad layers** shift the `start` offset.
- Everything else (FullConnect, activations, norms, ...) is treated as
  kernel=1, stride=1, padding=0 (pass-through).

## Output

For each network the report prints one row per layer plus a summary:

```
Idx   Class                          OutShape       k/s/p(x,y)        RF(x,y)  jump(x,y)     cover%
--------------------------------------------------------------------------------------------------------
0     TNNetInput                     (32,32,3)      1/1/0,1/1/0          1x1        1x1       0.1%
1     TNNetConvolutionReLU           (32,32,16)     3/1/1,3/1/1          3x3        1x1       0.9%
...
--------------------------------------------------------------------------------------------------------
Final receptive field: ... on a 32 x 32 input.
Final jump (effective stride): ...
First layer whose RF covers the whole input: ... — the rest is global mixing.
Flags:
  ...
```

`cover%` is the fraction of the input plane a single output unit of that layer
can see; it is flagged `>100%` once the RF exceeds the input (the
"already global" point). The flag list also calls out any layer whose RF
stopped growing (the pointwise / 1x1 tail).

## Running

```
cd examples/ReceptiveFieldReport
lazbuild ReceptiveFieldReport.lpi
../../bin/<arch>/bin/ReceptiveFieldReport
```

Or directly with fpc:

```
cd examples/ReceptiveFieldReport
fpc -B -Fu../../neural -Mobjfpc -Sh -O2 ReceptiveFieldReport.lpr
./ReceptiveFieldReport
```

Runs in under a second on CPU.
