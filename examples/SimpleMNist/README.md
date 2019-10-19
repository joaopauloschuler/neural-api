# Simple MNIST Image Classifier

[MNIST](http://yann.lecun.com/exdb/mnist/) is a dataset commonly used for testing computer vision algorithms.
It contains images with handwritten digits. With CAI, MNIST dataset is loaded with:
```
CreateMNISTVolumes(ImgTrainingVolumes, ImgValidationVolumes, ImgTestVolumes, 'train', 't10k');
```
It doesn't make sense using FlipX data augmentation method with MNIST. This is disabled with:
```
NeuralFit.HasFlipX := false;
```
This simple neural network classifies MNIST with above 99% accuracy as shown in the [results](https://github.com/joaopauloschuler/neural-api/tree/master/examples/SimpleMNist/results)
folder.
