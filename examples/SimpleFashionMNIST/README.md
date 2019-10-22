# Simple Fashion MNIST Image Classifier

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

With CAI, Fashion MNIST dataset is loaded with:
```
CreateMNISTVolumes(ImgTrainingVolumes, ImgValidationVolumes, ImgTestVolumes, 'train', 't10k', {Verbose=}true, {IsFashion=}true);
```

In this example, the presence of required dataset files is tested with
```
if Not(CheckMNISTFile('train', {IsFashion=}true)) or
   Not(CheckMNISTFile('t10k', {IsFashion=}true)) then exit;
```
