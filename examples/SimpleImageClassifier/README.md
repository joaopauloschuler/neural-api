# Simple CIFAR-10 Image Classifier
This example has interesting aspects to look at:
* Its source code is **very** small.
* Layers are added sequentially.
* Training parameters are defined before calling the `fit` method.

This is how a sequential array of layers is added:
```
    NN := TNNet.Create();
    NN.AddLayer([
      TNNetInput.Create(32, 32, 3),
      TNNetConvolutionLinear.Create({Features=}64, {FeatureSize=}5, {Padding=}2, {Stride=}1, {SuppressBias=}1),
      TNNetMaxPool.Create(4),
      TNNetMovingStdNormalization.Create(),
      TNNetConvolutionReLU.Create({Features=}64, {FeatureSize=}3, {Padding=}1, {Stride=}1, {SuppressBias=}1),
      TNNetConvolutionReLU.Create({Features=}64, {FeatureSize=}3, {Padding=}1, {Stride=}1, {SuppressBias=}1),
      TNNetConvolutionReLU.Create({Features=}64, {FeatureSize=}3, {Padding=}1, {Stride=}1, {SuppressBias=}1),
      TNNetConvolutionReLU.Create({Features=}64, {FeatureSize=}3, {Padding=}1, {Stride=}1, {SuppressBias=}1),
      TNNetDropout.Create(0.5),
      TNNetMaxPool.Create(2),
      TNNetFullConnectLinear.Create(10),
      TNNetSoftMax.Create()
    ]);
```

Later on, this is how the training/fitting is called:

```
    NeuralFit := TNeuralImageFit.Create;
    NeuralFit.FileNameBase := 'SimpleImageClassifier';
    NeuralFit.InitialLearningRate := 0.001;
    NeuralFit.LearningRateDecay := 0.005;
    NeuralFit.StaircaseEpochs := 17;
    NeuralFit.Inertia := 0.9;
    NeuralFit.L2Decay := 0.00001;
    NeuralFit.Fit(NN, ImgTrainingVolumes, ImgValidationVolumes, ImgTestVolumes, {NumClasses=}10, {batchsize=}128, {epochs=}100);
```

There is a trick that you can do with this API or any other API when working with image classification: **you can increase the input image size**.

As per the following example, by increasing CIFAR-10 input image sizes from 32x32 to 48x48, you can gain up to 2% in classification accuracy.

You can change image sizes with:
```
ImgTrainingVolumes.ResizeImage(48, 48);
ImgValidationVolumes.ResizeImage(48, 48);
ImgTestVolumes.ResizeImage(48, 48);
```

You can find an implementation with this trick at the [SimpleImageClassifierResize48.lpr](https://github.com/joaopauloschuler/neural-api/blob/master/examples/SimpleImageClassifier/SimpleImageClassifierResize48.lpr) file. There is also another implementation resizing to CIFAR-10 to 64x64 pixels but the gain won't be too big.
