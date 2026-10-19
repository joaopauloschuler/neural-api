# CIFAR-10 Resized
This folder contains a program that resizes CIFAR-10 and CIFAR-100 images to 64x64 and 128x128 pixels. 

## Original Work
The [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html) is a fantastic [contribution](https://paperswithcode.com/dataset/cifar-10) to computer science made by the original authors.
Please refer to https://www.cs.toronto.edu/~kriz/cifar.html and to [Learning Multiple Layers of Features from Tiny Images, Alex Krizhevsky, 2009](https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf) for the original work.

## Resizing Images
Plenty of modern neural network architectures wont work with small images. 
You may look at this [API example](https://github.com/joaopauloschuler/neural-api/tree/master/examples/SuperResolution) if you are curious to know more about how the resizing internally works. There is also [this bug report](https://github.com/joaopauloschuler/neural-api/issues/26) that gives insights about how it was built.
<p>
  <img src="https://github.com/joaopauloschuler/neural-api/blob/master/examples/SuperResolution/results/bird.png?raw=true"> </img>
</p>
<p>
  <img src="https://github.com/joaopauloschuler/neural-api/blob/master/examples/SuperResolution/results/stealth.png?raw=true"> </img>
</p>

## Download
You can download resized images from:
* https://www.kaggle.com/joaopauloschuler/cifar10-64x64-resized-via-cai-super-resolution
* https://www.kaggle.com/joaopauloschuler/cifar10-128x128-resized-via-cai-super-resolution
* https://www.kaggle.com/joaopauloschuler/cifar100-128x128-resized-via-cai-super-resolution

## Citing this Work
You can cite this work in BibTeX format with:
```
@software{cai_neural_api_2021_5810077,
  author       = {Joao Paulo Schwarz Schuler},
  title        = {CAI NEURAL API},
  month        = dec,
  year         = 2021,
  publisher    = {Zenodo},
  version      = {v1.0.6},
  doi          = {10.5281/zenodo.5810077},
  url          = {https://doi.org/10.5281/zenodo.5810077}
}
