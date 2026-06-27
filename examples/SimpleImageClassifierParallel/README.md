# CIFAR-10 Image Classifier with Parallel Convolutions

This example trains a CIFAR-10 image classifier built from **parallel
inception-style convolution branches** rather than a plain sequential conv
stack. It is the parallel-branch companion to the
[Simple CIFAR-10 Image Classifier](../SimpleImageClassifier/README.md).

It has interesting aspects to look at:
* The network is a `THistoricalNets` (a `TNNet` subclass that adds the
  branch/residual builder helpers).
* Most of the depth is added with the `AddParallelConvs` helper, which builds
  a multi-path block of `1x1`, `3x3`, `5x5` and `7x7` convolutions in parallel
  (each behind a bottleneck) and concatenates the results.
* Training is multi-threaded via `TNeuralImageFit.MaxThreadNum`.

## The network

After an initial `TNNetConvolutionReLU` and a `TNNetMulLearning` scaling layer,
the body is built by calling `AddParallelConvs` four times (two blocks before a
`TNNetMaxPool`, two after). Each call adds a parallel block like this:

```
    NN.AddParallelConvs(
        {PointWiseConv}TNNetConvolutionLinear,
        {IsSeparable}false,
        {CopyInput}false,
        {pBeforeBottleNeck}nil,
        {pAfterBottleNeck}nil,
        {pBeforeConv}nil,
        {pAfterConv}TNNetReLU,
        {PreviousLayer}nil,
        {BottleNeck}16,
        {p11ConvCount}2,  {p11FilterCount}16,
        {p33ConvCount}2,  {p33FilterCount}16,
        {p55ConvCount}2,  {p55FilterCount}16,
        {p77ConvCount}2,  {p77FilterCount}16,
        {maxPool}0
    );
```

The head is a `TNNetDropout(0.5)`, a `TNNetMaxPool`, a
`TNNetFullConnectLinear(10)` and a `TNNetSoftMax`. The full structure is printed
at startup with `NN.DebugStructure()`.

## Fitting

Training is driven by `TNeuralImageFit`, with parallel training enabled:

```
    NeuralFit := TNeuralImageFit.Create;
    NeuralFit.FileNameBase := 'SimpleImageClassifierParallel';
    NeuralFit.InitialLearningRate := 0.001;
    NeuralFit.LearningRateDecay := 0.01;
    NeuralFit.StaircaseEpochs := 10;
    NeuralFit.Inertia := 0.9;
    NeuralFit.L2Decay := 0.00001;
    NeuralFit.MaxThreadNum := 32;
    NeuralFit.Fit(NN, ImgTrainingVolumes, ImgValidationVolumes, ImgTestVolumes, {NumClasses=}10, {batchsize=}64, {epochs=}50);
```

`MaxThreadNum := 32` caps the number of worker threads used during training; the
fit code will use up to that many (limited by the available cores).

## Running

The CIFAR-10 dataset is fetched/checked automatically by `CheckCIFARFile()` on
first run. Build the program and run it:

```
cd examples/SimpleImageClassifierParallel
# build with lazbuild (or fpc), then run:
./SimpleImageClassifierParallel
```

It prints the network structure, then per-epoch training/validation progress
through `TNeuralImageFit` for 50 epochs, and saves the best model to files based
on `FileNameBase`.

Coded by Joao Paulo Schwarz Schuler.
