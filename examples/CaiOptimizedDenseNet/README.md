# CaiOptimizedDenseNet — CAI DenseNet on CIFAR-10

Trains **CAI Optimized DenseNet** image classifiers on **CIFAR-10**, inspired by
the original [DenseNet](https://github.com/liuzhuang13/DenseNet). The folder ships
three variants:

- **`CaiOptimizedDenseNet.lpr`** — 32×32 input with padding/cropping
  augmentation.
- **`CaiOptimizedDenseNet48.lpr`** — input resized to 48×48 (no crop
  augmentation).
- **`kOptimizedDenseNet.lpr`** — the *k* variant: grouped pointwise convolutions
  with interleaved channels and an alternative compression path.

## What they use

- `THistoricalNets` builders: `AddDenseNetBlockCAI` / `AddkDenseNetBlock`, plus
  `TNNetInput`, `TNNetConvolutionReLU`/`TNNetConvolutionLinear`,
  `TNNetCompression` / `TNNetGroupedCompression`,
  `TNNetGroupedPointwiseConvLinear`, `TNNetInterleaveChannels`, `TNNetMaxPool`,
  `TNNetMaxChannel`, `TNNetDropout`, `TNNetFullConnectLinear`, `TNNetSoftMax`.
- Training with `TNeuralImageFit` (`Fit`), batch size 64, up to 300 epochs,
  cyclical learning rate (length 100), staircase decay every 15 epochs.
- Data via `CreateCifar10Volumes` (and `CheckCIFARFile`); the 32×32 variant pads
  and crops (`AddPadding`, `MaxCropSize=8`), the 48 variant resizes.

## Running

CIFAR-10 must be available (checked/loaded by `CheckCIFARFile` /
`CreateCifar10Volumes`). Build a variant and run it; the trained network is saved
under a `FileNameBase` (`CaiOptimizedDenseNet*`, `CaiOptimizedDenseNet48*`,
`kCaiOptimizedDenseNet*`).

```
cd examples/CaiOptimizedDenseNet
# build with lazbuild CaiOptimizedDenseNet.lpi (or fpc), then:
./CaiOptimizedDenseNet [options]
```

Command-line options:

- `-l, --learningrate` (default 0.001)
- `-i, --inertia` momentum (default 0.9)
- `-t, --target` target accuracy (default 1.0)
- `-c, --convolutions` inner convolutions (default 12)
- `-b, --bottleneck` (default 32; 128 in the k variant)
- `-n, --neurons` convolutional growth rate (default 32)
- `-p, --padding` enable padding/crop augmentation (k variant)
- `-h, --help`

It prints the network structure (layers / neurons / weights), the hyperparameters
in effect, and per-epoch training/validation/test accuracy from `TNeuralImageFit`.

Coded by Joao Paulo Schwarz Schuler.
