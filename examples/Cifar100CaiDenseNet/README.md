# CAI Optimized DenseNet CIFAR-100 Image Classifier

[CIFAR-100](https://www.cs.toronto.edu/~kriz/cifar.html) is an 100 classes dataset commonly used for testing computer vision algorithms.

This example contains a CAI Optimized [DenseNet](https://github.com/liuzhuang13/DenseNet) architecture.

With CAI, CIFAR-100 dataset is loaded with:
```
    CreateCifar100Volumes(ImgTrainingVolumes, ImgValidationVolumes, ImgTestVolumes);
```
