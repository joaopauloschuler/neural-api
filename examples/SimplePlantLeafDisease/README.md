# Simple Plant Leaf Disease Image Classifier for the PlantVillage Dataset

This example has interesting aspects to look at:
* An easy and fast way to load image datasets where each folder represents one class of images: **CreateVolumesFromImagesFromFolder**.
* An image classification example that identifies plant diseases based on a photo taken from a leaf.
* A [jupyter notebook](https://github.com/joaopauloschuler/neural-api/blob/master/examples/SimplePlantLeafDisease/SimplePlantLeafDisease.ipynb)
that trains a neural network coded in **PASCAL** for plant disease classification.

There are plenty of image datasets where each folder represents a class of images.
This is exactly what **CreateVolumesFromImagesFromFolder** does as exemplified below:
```
    // change ProportionToLoad to a smaller number if you don't have available 32GB of RAM.
    ProportionToLoad := 1;
    WriteLn('Loading ', Round(ProportionToLoad*100), '% of the Plant leave disease dataset into memory.');
    CreateVolumesFromImagesFromFolder
    (
      ImgTrainingVolumes, ImgValidationVolumes, ImgTestVolumes,
      {FolderName=}'plant', {pImageSubFolder=}'',
      {color_encoding=}0{RGB},
      {TrainingProp=}0.9*ProportionToLoad,
      {ValidationProp=}0.05*ProportionToLoad,
      {TestProp=}0.05*ProportionToLoad,
      {NewSizeX=}128, {NewSizeY=}128
    );
```
The example above shows how to load the dataset used with the paper
[Identification of Plant Leaf Diseases Using a 9-layer Deep Convolutional Neural Network](https://data.mendeley.com/datasets/tywbtsjrjv/1).
90% is loaded as training data while 5% is loaded for each validation and testing. Images are being resized to 128x128.

As you can see on these [raw results](https://github.com/joaopauloschuler/neural-api/blob/master/examples/SimplePlantLeafDisease/results/SimplePlantLeafDisease-20200412.csv),
the test classification (or the plant disease diagnostic) accuracy is 98.95%.

The dataset can be downloaded from: https://data.mendeley.com/datasets/tywbtsjrjv/1 .
