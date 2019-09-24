# Simple CIFAR-10 Image Classifier
This example has interesting aspects to look at:
* Its source code is **very** small.
* Layers are added sequentially.
* Training parameters are defined before calling the `fit` method.

This is how a sequential array of layers is added:
```
    NN := THistoricalNets.Create();
    NN.AddLayer([
      TNNetInput.Create(32, 32, 3),
      TNNetConvolutionLinear.Create(64, 5, 2, 1, 1).InitBasicPatterns(),
      TNNetMaxPool.Create(4),
      TNNetConvolutionReLU.Create(64, 3, 0, 1, 1),
      TNNetConvolutionReLU.Create(64, 3, 0, 1, 1),
      TNNetConvolutionReLU.Create(64, 3, 0, 1, 1),
      TNNetConvolutionReLU.Create(64, 3, 0, 1, 1),
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
    NeuralFit.LearningRateDecay := 0.995;
    NeuralFit.StaircaseEpochs := 17;
    NeuralFit.Inertia := 0.9;
    NeuralFit.L2Decay := 0.00001;
    NeuralFit.Fit(NN, ImgTrainingVolumes, ImgValidationVolumes, ImgTestVolumes, {NumClasses=}10, {batchsize=}128, {epochs=}100);
```

Looks pretty simple!
