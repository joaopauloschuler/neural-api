# Tiny ImageNet 200 Simple Image Classifier

This example loads the [Tiny ImageNet 200](https://tiny-imagenet.herokuapp.com/) dataset into RAM memory.

In the below example, 90% of the dataset is loaded as training data while 5% is loaded for each validation and testing:
```
    // change ProportionToLoad to a smaller number if you don't have available 6GB of RAM.
    ProportionToLoad := 1;
    WriteLn('Loading ', Round(ProportionToLoad*100), '% of the Tiny ImageNet 200 dataset into memory.');
    CreateVolumesFromImagesFromFolder
    (
      ImgTrainingVolumes, ImgValidationVolumes, ImgTestVolumes,
      {FolderName=}'tiny-imagenet-200/train', {pImageSubFolder=}'images',
      {color_encoding=}0{RGB},
      {TrainingProp=}0.9*ProportionToLoad,
      {ValidationProp=}0.05*ProportionToLoad,
      {TestProp=}0.05*ProportionToLoad
    );
```

There is a [jupyter notebook](https://github.com/joaopauloschuler/neural-api/blob/master/examples/SimpleTinyImageNet/tiny-imagenet-200.ipynb) that runs this example.
