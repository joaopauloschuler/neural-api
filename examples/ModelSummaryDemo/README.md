# ModelSummaryDemo — `TNNet.PrintSummary()` across three networks (+ format smoke test)

A tiny, fast (well under a second), no-training demo that builds **three
structurally-distinct** networks and prints each via `TNNet.PrintSummary()`. It
doubles as a **self-checking smoke test** of the summary table format.

## The idea

`TNNet.SummaryString()` (which `PrintSummary()` simply `WriteLn`s) renders a
Keras-style table:

- a **header row**: `Idx  Layer  Output Shape  Params  Neurons`;
- a separator rule (`-----`);
- **one body row per layer**, carrying the layer index, the layer class name,
  its output shape `(SizeX, SizeY, Depth)`, its weight count and its neuron count;
- a closing separator rule;
- a **footer**: `Totals: <L> layers, <W> weights, <N> neurons`.

The demo constructs three nets that exercise this table across very different
layer types, prints all three, and asserts the output is well-formed.

## The three networks

1. **Small MLP** — `Input(8) -> FullConnectReLU(16) -> FullConnectReLU(12) ->
   FullConnectLinear(4)`. Dense layers, flat shapes. (4 layers, 368 weights.)
2. **Tiny conv net** — `Input(16,16,3) -> ConvReLU(8) -> MaxPool -> ConvReLU(12)
   -> MaxPool -> FullConnectReLU(8) -> FullConnectLinear(3)`. Shows spatial output
   shapes and convolution parameter counts. (7 layers, 2640 weights.)
3. **Pre-norm residual** — `Input(1,1,16) -> [LayerNorm -> PointwiseConvLinear(16)
   -> Sum] -> FullConnectLinear(4)`, built via `TNNet.AddPreNormResidual`. Shows a
   normalization layer and a non-trivial **multi-input** `TNNetSum` node. The
   residual sublayer is a depth-wise `TNNetPointwiseConvLinear` so it is
   shape-preserving (a `FullConnect` would change the shape and break the residual
   sum). (5 layers, 352 weights.)

All three are tiny and untrained: just `Create` + `AddLayer` + `InitWeights` +
`PrintSummary`.

## How the smoke test works

`PrintSummary` writes to stdout, but the library also exposes the
string-returning `TNNet.SummaryString()`. For each net the demo captures that
string, prints it, and parses it — then **cross-checks the parsed numbers against
the network object computed independently**:

- the header row contains the expected column titles;
- there are exactly `NN.CountLayers()` body rows (counted strictly between the
  two separator rules);
- the per-row `Params` summed equal `NN.CountWeights()`, and the per-row
  `Neurons` summed equal `NN.CountNeurons()`;
- the `Totals:` footer reports `CountLayers()`/`CountWeights()`/`CountNeurons()`
  matching the object;
- every net has `layers > 0` and `weights >= 0`;
- the three nets are structurally distinct (different layer counts).

Any failure prints a diagnostic and `Halt(1)`s; otherwise a final `GATE: PASS`
line is printed. This mirrors the self-checking gate idiom of `examples/SIREN`
and `examples/DeepSets`.

## Build & run

With Lazarus:

```
lazbuild examples/ModelSummaryDemo/ModelSummaryDemo.lpi
./bin/$(fpc -iTP)-$(fpc -iTO)/bin/ModelSummaryDemo
```

Or directly with FPC:

```
cd examples/ModelSummaryDemo
fpc -B -Fu../../neural -Mobjfpc -Sh -O2 ModelSummaryDemo.lpr && ./ModelSummaryDemo
```

It must compile clean and exit 0 with a `GATE: PASS` line.

## Sample output

```
================================================================
Net 1: small MLP
  Input(8) -> FullConnectReLU(16) -> FullConnectReLU(12) -> FullConnectLinear(4)
----------------------------------------------------------------
Idx   Layer                            Output Shape                 Params      Neurons
---------------------------------------------------------------------------------------
0     TNNetInput                       (8, 1, 1)                         0            0
1     TNNetFullConnectReLU             (16, 1, 1)                      128           16
2     TNNetFullConnectReLU             (12, 1, 1)                      192           12
3     TNNetFullConnectLinear           (4, 1, 1)                        48            4
---------------------------------------------------------------------------------------
Totals: 4 layers, 368 weights, 32 neurons

================================================================
Net 2: tiny conv net
  Input(16,16,3) -> ConvReLU(8) -> MaxPool -> ConvReLU(12) -> MaxPool
    -> FullConnectReLU(8) -> FullConnectLinear(3)
----------------------------------------------------------------
Idx   Layer                            Output Shape                 Params      Neurons
---------------------------------------------------------------------------------------
0     TNNetInput                       (16, 16, 3)                       0            0
1     TNNetConvolutionReLU             (16, 16, 8)                     216            8
2     TNNetMaxPool                     (8, 8, 8)                         0            0
3     TNNetConvolutionReLU             (8, 8, 12)                      864           12
4     TNNetMaxPool                     (4, 4, 12)                        0            0
5     TNNetFullConnectReLU             (8, 1, 1)                      1536            8
6     TNNetFullConnectLinear           (3, 1, 1)                        24            3
---------------------------------------------------------------------------------------
Totals: 7 layers, 2640 weights, 31 neurons

================================================================
Net 3: pre-norm residual (LayerNorm + multi-input Sum)
  Input(1,1,16) -> [LayerNorm -> PointwiseConvLinear(16) -> Sum]
    -> FullConnectLinear(4)
----------------------------------------------------------------
Idx   Layer                            Output Shape                 Params      Neurons
---------------------------------------------------------------------------------------
0     TNNetInput                       (1, 1, 16)                        0            0
1     TNNetLayerNorm                   (1, 1, 16)                       32            2
2     TNNetPointwiseConvLinear         (1, 1, 16)                      256           16
3     TNNetSum                         (1, 1, 16)                        0            0
4     TNNetFullConnectLinear           (4, 1, 1)                        64            4
---------------------------------------------------------------------------------------
Totals: 5 layers, 352 weights, 22 neurons

================================================================
GATE: PASS - all three summaries are well-formed and their row/total counts match CountLayers()/CountWeights()/CountNeurons().
```
