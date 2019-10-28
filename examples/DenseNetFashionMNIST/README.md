# CAI Optimized DenseNet Fashion MNIST Image Classifier

[Fashion MNIST](https://github.com/zalandoresearch/fashion-mnist) is a dataset commonly used for testing computer vision algorithms.
It contains these classes:
* T-shirt/top
* Trouser
* Pullover
* Dress
* Coat
* Sandal
* Shirt
* Sneaker
* Bag
* Ankle boot

This example contains a [CAI Optimized DenseNet](https://github.com/liuzhuang13/DenseNet) architecture:
```
NN.AddLayer( TNNetInput.Create(28, 28, 1).EnableErrorCollection() );
// First block shouldn't be separable.
NN.AddDenseNetBlockCAI(iInnerConvNum div 6, iConvNeuronCount, {supressBias=}0, TNNetConvolutionReLU, {IsSeparable=}false, {HasMovingNorm=}HasMovingNorm, {pBeforeNorm=}nil, {pAfterNorm=}nil, {BottleNeck=}iBottleneck, {Compression=}0, {Dropout=}0, {RandomAdd=}1, {RandomMul=}1);
NN.AddDenseNetBlockCAI(iInnerConvNum div 6, iConvNeuronCount, {supressBias=}0, TNNetConvolutionReLU, {IsSeparable=}bSeparable, {HasMovingNorm=}HasMovingNorm, {pBeforeNorm=}nil, {pAfterNorm=}nil, {BottleNeck=}iBottleneck, {Compression=}0, {Dropout=}0, {RandomAdd=}1, {RandomMul=}1);
NN.AddLayer( TNNetMaxPool.Create(2) );
NN.AddDenseNetBlockCAI(iInnerConvNum div 3, iConvNeuronCount, {supressBias=}0, TNNetConvolutionReLU, {IsSeparable=}bSeparable, {HasMovingNorm=}HasMovingNorm, {pBeforeNorm=}nil, {pAfterNorm=}nil, {BottleNeck=}iBottleneck, {Compression=}0, {Dropout=}0, {RandomAdd=}1, {RandomMul=}1);
NN.AddLayer( TNNetMaxPool.Create(2) );
NN.AddDenseNetBlockCAI(iInnerConvNum div 3, iConvNeuronCount, {IsSeparable=}0, TNNetConvolutionReLU, {IsSeparable=}bSeparable, {HasMovingNorm=}HasMovingNorm, {pBeforeNorm=}nil, {pAfterNorm=}nil, {BottleNeck=}iBottleneck, {Compression=}0, {Dropout=}0, {RandomAdd=}1, {RandomMul=}1);
NN.AddLayer( TNNetDropout.Create(0.10) );
NN.AddLayer( TNNetMaxChannel.Create() );
NN.AddLayer( TNNetFullConnectLinear.Create(NumClasses) );
NN.AddLayer( TNNetSoftMax.Create() );
```

With CAI, Fashion MNIST dataset is loaded with:
```
CreateMNISTVolumes(ImgTrainingVolumes, ImgValidationVolumes, ImgTestVolumes, 'train', 't10k', {Verbose=}true, {IsFashion=}true);
```
