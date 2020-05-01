# Super Resolution Example is Under Construction
<img align="right" src="results/street_result.png"></img>
## Introduction
The image at the right side shows an example. The smaller image is the original image while the bigger image is the image processed twice by a neural network trained to increase the resolution.
This example has been created via the **SuperResolution.lpi** command line tool with:

```
#SuperResolution -i street.png -o street2.png
Loading input file: street.png
Input image size: 79x107x3
Creating Neural Network...
Resizing...
Neural network file found at ../../../examples/SuperResolution : super-resolution-7-64-sep.nn
Saving output file: street2.png

#SuperResolution -i street2.png -o street3.png
Loading input file: street2.png
Input image size: 158x214x3
Creating Neural Network...
Resizing with tiles...
Neural network file found at ../../../examples/SuperResolution : super-resolution-7-64-sep.nn
Padding input image.
Resizing with tiles to: 288x416x3
Saving output file: street3.png
```
Besides the command line tool above, the **SuperResolution** folder has these main components:
* **SuperResolutionTrain.lpi**: trains a neural network for increasing image resolution with the CIFAR-10 dataset.
* **Cifar10ImageClassifierSuperResolution.lpi**: can we better classify 32x32 images if we first upscale to 64x64? This experiment shows it!
* **super-resolution-7-64-sep.nn**: ready to use already trained neural network.
* **SuperResolutionApp.lpi**: visually tests the trained neural network with CIFAR-10 images upscaling 32x32 images up 256x256 images.

## SuperResolutionTrain
Under construction.
## Cifar10ImageClassifierSuperResolution
Under construction.
## SuperResolution Command Line Tool
Under construction.
## SuperResolutionApp
<p>
  <img src="results/bird.png"> </img>
</p>
Under construction.
<p>
  <img src="results/stealth.png"> </img>
</p>
