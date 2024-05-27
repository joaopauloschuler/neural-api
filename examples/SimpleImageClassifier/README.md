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
    NeuralFit.LearningRateDecay := 0.01;
    NeuralFit.StaircaseEpochs := 10;
    NeuralFit.Inertia := 0.9;
    NeuralFit.L2Decay := 0;
    NeuralFit.Fit(NN, ImgTrainingVolumes, ImgValidationVolumes, ImgTestVolumes, {NumClasses=}10, {batchsize=}64, {epochs=}50);   
```

## Beyond ReLU Activation Function

The paper [Searching for Activation Functions](https://arxiv.org/abs/1710.05941) describes the search for a better activation function than **ReLU**. In their work, the authors found **Swish** to be the best replacement for **ReLU**. The downside of Swish is: it requires a lot of computation to calculate it. Later, the paper [Searching for MobileNetV3](https://arxiv.org/pdf/1905.02244v5.pdf) introduces the **Hard Swish** activation function. The **Hard Swish** gives similar results to **Swish** with a lot less computation.

The same neural network shown above could be implemented with **Swish** as
```
    NN.AddLayer([
      TNNetInput.Create(32, 32, 3),
      TNNetConvolutionLinear.Create({Features=}64, {FeatureSize=}5, {Padding=}2, {Stride=}1, {SuppressBias=}1),
      TNNetMaxPool.Create(4),
      TNNetMovingStdNormalization.Create(),
      TNNetConvolutionLinear.Create({Features=}64, {FeatureSize=}3, {Padding=}1, {Stride=}1, {SuppressBias=}1),
      TNNetSwish.Create(),
      TNNetConvolutionLinear.Create({Features=}64, {FeatureSize=}3, {Padding=}1, {Stride=}1, {SuppressBias=}1),
      TNNetSwish.Create(),
      TNNetConvolutionLinear.Create({Features=}64, {FeatureSize=}3, {Padding=}1, {Stride=}1, {SuppressBias=}1),
      TNNetSwish.Create(),
      TNNetConvolutionLinear.Create({Features=}64, {FeatureSize=}3, {Padding=}1, {Stride=}1, {SuppressBias=}1),
      TNNetSwish.Create(),
      TNNetDropout.Create(0.5),
      TNNetMaxPool.Create(2),
      TNNetFullConnectLinear.Create(10),
      TNNetSoftMax.Create()
    ]); 
```

or as
```
    NN.AddLayer([
      TNNetInput.Create(32, 32, 3),
      TNNetConvolutionLinear.Create({Features=}64, {FeatureSize=}5, {Padding=}2, {Stride=}1, {SuppressBias=}1),
      TNNetMaxPool.Create(4),
      TNNetMovingStdNormalization.Create(),
      TNNetConvolutionSwish.Create({Features=}64, {FeatureSize=}3, {Padding=}1, {Stride=}1, {SuppressBias=}1),
      TNNetConvolutionSwish.Create({Features=}64, {FeatureSize=}3, {Padding=}1, {Stride=}1, {SuppressBias=}1),
      TNNetConvolutionSwish.Create({Features=}64, {FeatureSize=}3, {Padding=}1, {Stride=}1, {SuppressBias=}1),
      TNNetConvolutionSwish.Create({Features=}64, {FeatureSize=}3, {Padding=}1, {Stride=}1, {SuppressBias=}1),
      TNNetDropout.Create(0.5),
      TNNetMaxPool.Create(2),
      TNNetFullConnectLinear.Create(10),
      TNNetSoftMax.Create()
    ]); 
```

The Hard Swish variant is implemented with:
```
    NN.AddLayer([
      TNNetInput.Create(32, 32, 3),
      TNNetConvolutionLinear.Create({Features=}64, {FeatureSize=}5, {Padding=}2, {Stride=}1, {SuppressBias=}1),
      TNNetMaxPool.Create(4),
      TNNetMovingStdNormalization.Create(),
      TNNetConvolutionHardSwish.Create({Features=}64, {FeatureSize=}3, {Padding=}1, {Stride=}1, {SuppressBias=}1),
      TNNetConvolutionHardSwish.Create({Features=}64, {FeatureSize=}3, {Padding=}1, {Stride=}1, {SuppressBias=}1),
      TNNetConvolutionHardSwish.Create({Features=}64, {FeatureSize=}3, {Padding=}1, {Stride=}1, {SuppressBias=}1),
      TNNetConvolutionHardSwish.Create({Features=}64, {FeatureSize=}3, {Padding=}1, {Stride=}1, {SuppressBias=}1),
      TNNetDropout.Create(0.5),
      TNNetMaxPool.Create(2),
      TNNetFullConnectLinear.Create(10),
      TNNetSoftMax.Create()
    ]);
```


These are the CIFAR-10 classification accuracies with ReLU, Swish and HardSwish activation functions:

Activation Function (source) | Test Classification Accuracy (%)
---------------------------- | -------------------------------------
[ReLU](https://github.com/joaopauloschuler/neural-api/blob/master/examples/SimpleImageClassifier/SimpleImageClassifier.lpr) | [85.53%](https://github.com/joaopauloschuler/neural-api/blob/master/examples/SimpleImageClassifier/results/SimpleImageClassifier20221206.csv)
[Swish](https://github.com/joaopauloschuler/neural-api/blob/master/examples/SimpleImageClassifier/SimpleImageClassifierSwish.lpr) | [86.55%](https://github.com/joaopauloschuler/neural-api/blob/master/examples/SimpleImageClassifier/results/SimpleImageClassifierSwish20221207.csv)
[Hard Swish](https://github.com/joaopauloschuler/neural-api/blob/master/examples/SimpleImageClassifier/SimpleImageClassifierHardSwish.lpr) | [86.82%](https://github.com/joaopauloschuler/neural-api/blob/master/examples/SimpleImageClassifier/results/SimpleImageClassifierHardSwish20221208.csv) 
