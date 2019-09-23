# CAI NEURAL API
CAI NEURAL API is a pascal based neural network API optimized for AVX, AVX2 and AVX512 instruction sets plus
OpenCL capable devices including AMD, Intel and NVIDIA. This API has been tested under Windows and Linux.

This project is a subproject from a bigger and older project called [CAI](https://sourceforge.net/projects/cai/).

## Why Pascal?
* Compiled pascal code is super fast! This API can outperform some major APIs in some architectures.
* Pascal is easy to learn and easy to make a bug free code. You'll be able to make super fast **native** code.

## Prerequisites
You'll need [Lazarus](https://www.lazarus-ide.org/) development environment. If you have an OpenCL capable device, you'll need its OpenCL drivers.

## Installation
Clone this project and add the **neural** folder to your Lazarus unit search path and you are ready to go!

## How does The Code Look Like for a CIFAR-10 Classification Example?
```
    NN := TNNet.Create();
    NN.AddLayer(TNNetInput.Create(32, 32, 3));
    NN.AddLayer(TNNetConvolutionReLU.Create(16, 5, 0, 0));
    NN.AddLayer(TNNetMaxPool.Create(2));
    NN.AddLayer(TNNetConvolutionReLU.Create(32, 5, 0, 0));
    NN.AddLayer(TNNetMaxPool.Create(2));
    NN.AddLayer(TNNetConvolutionReLU.Create(32, 5, 0, 0));
    NN.AddLayer(TNNetLayerFullConnectReLU.Create(32));
    NN.AddLayer(TNNetFullConnectLinear.Create(NumClasses));
    NN.AddLayer(TNNetSoftMax.Create());
    CreateCifar10Volumes(ImgTrainingVolumes, ImgValidationVolumes, ImgTestVolumes);

    WriteLn('Neural Network will minimize error with:');
    WriteLn(' Layers: ', NN.CountLayers());
    WriteLn(' Neurons:', NN.CountNeurons());
    WriteLn(' Weights:' ,NN.CountWeights());
    NN.DebugWeights();
    NN.DebugStructure();

    NeuralFit := TNeuralImageFit.Create;
    NeuralFit.InitialLearningRate := fLearningRate;
    NeuralFit.Inertia := fInertia;
    NeuralFit.Fit(NN, ImgTrainingVolumes, ImgValidationVolumes, ImgTestVolumes, NumClasses, {batchsize}128, {epochs}100);
 ```
 
## Documentation

The documentation is currently under construction. There are some available videos:
* [Increasing Image Resolution with Neural Networks](https://www.youtube.com/watch?v=jdFixaZ2P4w)
* [Ultra Fast Single Precision Floating Point Computing](https://www.youtube.com/watch?v=qGnfwpKUTIQ)

There are also some [older code examples](https://sourceforge.net/p/cai/svncode/HEAD/tree/trunk/lazarus/experiments/) that you can look at. 

## Quick View about the API
This API is really big. The following list gives a general idea about this API but it doesn't contain everything.

### Input Layer
* TNNetInput (input/output: 1D, 2D or 3D).

### Convolutional Layers
* TNNetConvolution (input/output: 1D, 2D or 3D - feature size: 1D or 2D).
* TNNetConvolutionReLU (input/output: 1D, 2D or 3D - feature size: 1D or 2D).
* TNNetConvolutionLinear (input/output: 1D, 2D or 3D - feature size: 1D or 2D).
* TNNetPointwiseConvReLU (input/output: 1D, 2D or 3D).
* TNNetPointwiseConvLinear (input/output: 1D, 2D or 3D).
* TNNetDepthwiseConv (input/output: 1D, 2D or 3D).
* TNNetDepthwiseConvReLU (input/output: 1D, 2D or 3D).
* TNNetDepthwiseConvLinear (input/output: 1D, 2D or 3D).
* TNNet.AddSeparableConvReLU (input/output: 1D, 2D or 3D - separable convolution).
* TNNet.AddSeparableConvLinear (input/output: 1D, 2D or 3D - separable convolution).
* TNNet.AddConvOrSeparableConv (input/output: 1D, 2D or 3D). Adds a convolution or a separable convolutions with/without ReLU and normalization.

### Fully Connected Layers
* TNNetFullConnect (input/output: 1D, 2D or 3D).
* TNNetFullConnectReLU (input/output: 1D, 2D or 3D).
* TNNetFullConnectLinear (input/output: 1D, 2D or 3D).
* TNNetFullConnectSigmoid (input/output: 1D, 2D or 3D).

### Locally Connected Layers
* TNNetLocalConnect (input/output: 1D, 2D or 3D - feature size: 1D or 2D). Similar to full connect with individual neurons.
* TNNetLocalConnectReLU (input/output: 1D, 2D or 3D - feature size: 1D or 2D).

### Min / Max / Avg pools
* TNNetAvgPool (input/output: 1D, 2D or 3D).
* TNNetMaxPool (input/output: 1D, 2D or 3D).
* TNNetMinPool (input/output: 1D, 2D or 3D).
* TNNet.AddMinMaxPool (input/output: 1D, 2D or 3D - min and max pools and then concatenates results).
* TNNet.AddAvgMaxPool (input/output: 1D, 2D or 3D - average and max pools and then concatenates results).

### Min / Max / Avg layers that Operate an Entire Channel and Produce Only One Result per Channel
* TNNetAvgChannel (input: 2D or 3D - output: 1D). Calculates the channel average.
* TNNetMaxChannel (input: 2D or 3D - output: 1D). Calculates the channel max.
* TNNetMinChannel (input: 2D or 3D - output: 1D). Calculates the channel min.
* TNNet.AddMinMaxChannel (input/output: 1D, 2D or 3D - min and max channel and then concatenates results).
* TNNet.AddAvgMaxChannel (input/output: 1D, 2D or 3D - average and max channel and then concatenates results).

### Trainable Normalization Layers Allowing Faster Learning/Convergence
* TNNetChannelZeroCenter (input/output: 1D, 2D or 3D). Trainable zero centering.
* TNNetMovingStdNormalization (input/output: 1D, 2D or 3D). Trainable std. normalization.
* TNNetChannelStdNormalization (input/output: 1D, 2D or 3D). Trainable per channel std. normalization.
* TNNet.AddMovingNorm (input/output: 1D, 2D or 3D). Possible replacement for batch normalization.
* TNNet.AddChannelMovingNorm (input/output: 1D, 2D or 3D). Possible replacement for per batch normalization.

### Non Trainable and per Sample Normalization Layers
* TNNetLayerMaxNormalization (input/output: 1D, 2D or 3D).
* TNNetLayerStdNormalization (input/output: 1D, 2D or 3D).
* TNNetLocalResponseNorm2D (input/output: 2D or 3D).
* TNNetLocalResponseNormDepth (input/output: 2D or 3D).
* TNNetRandomMulAdd (input/output: 1D, 2D or 3D). Adds a random multiplication and random bias (shift).
* TNNetChannelRandomMulAdd (input/output: 1D, 2D or 3D). Adds a random multiplication and random bias (shift) per channel.

### Concatenation, Summation and Reshaping Layers
* TNNetConcat (input/output: 1D, 2D or 3D). Allows concatenating the result from previous layers.
* TNNetDeepConcat (input/output: 1D, 2D or 3D). Concatenates into the Depth axis. This is useful with DenseNet like architectures.
* TNNetIdentity (input/output: 1D, 2D or 3D).
* TNNetIdentityWithoutBackprop (input/output: 1D, 2D or 3D). Allows the forward pass to proceed but prevents backpropagation.
* TNNetReshape (input/output: 1D, 2D or 3D).
* TNNetSplitChannels (input: 1D, 2D or 3D / output: 1D, 2D or 3D). Splits layers/channels from input.
* TNNetSum (input/output: 1D, 2D or 3D). Sums outputs from parallel layers allowing ResNet style networks.

### Layers with Activation Functions and no Trainable Parameter
* TNNetReLU (input/output: 1D, 2D or 3D).
* TNNetSigmoid (input/output: 1D, 2D or 3D).
* TNNetSoftMax (input/output: 1D, 2D or 3D).

### Trainable Bias (Shift) and Multiplication (Scaling) per Cell or Channel Allowing Faster Learning and Convergence
* TNNetCellBias (input/output: 1D, 2D or 3D).
* TNNetCellMul (input/output: 1D, 2D or 3D).
* TNNetChannelBias (input/output: 1D, 2D or 3D).
* TNNetChannelMul (input/output: 1D, 2D or 3D).

### Opposing Operations
* TNNetDeLocalConnect (input/output: 1D, 2D or 3D - feature size: 1D or 2D). Similar to full connect with individual neurons.
* TNNetDeLocalConnectReLU (input/output: 1D, 2D or 3D - feature size: 1D or 2D).
* TNNetDeconvolution (input/output: 1D, 2D or 3D - feature size: 1D or 2D).
* TNNetDeconvolutionReLU (input/output: 1D, 2D or 3D - feature size: 1D or 2D).
* TNNetDeMaxPool (input/output: 1D, 2D or 3D - max is done on a single layer).

### Weight Initializers
* InitUniform(Value: TNeuralFloat = 1).
* InitLeCunUniform(Value: TNeuralFloat = 1).
* InitHeUniform(Value: TNeuralFloat = 1).
* InitHeUniformDepthwise(Value: TNeuralFloat = 1).
* InitHeGaussian(Value: TNeuralFloat = 0.5).
* InitHeGaussianDepthwise(Value: TNeuralFloat = 0.5).
* InitGlorotBengioUniform(Value: TNeuralFloat = 1).

### Data Augmentation Methods Implemented at TVolume
* procedure FlipX();
* procedure FlipY();
* procedure CopyCropping(Original: TVolume; StartX, StartY, pSizeX, pSizeY: integer);
* procedure CopyResizing(Original: TVolume; NewSizeX, NewSizeY: integer);
* procedure AddGaussianNoise(pMul: TNeuralFloat);
* procedure AddSaltAndPepper(pNum: integer; pSalt: integer = 2; pPepper: integer = -2);

## Contributing
Pull requests are welcome. Having requests accepted might be hard.
