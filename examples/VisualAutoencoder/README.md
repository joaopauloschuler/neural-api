# Autoencoder Example
<p><img src="../../docs/autoencoder_small.png"></img></p>

Autoencoders have 2 pieces: an **encoder** and a **decoder**. In this example, the encoder encodes an image
from the [TinyImageNet dataset](https://paperswithcode.com/dataset/tiny-imagenet) with 64x64x3 pixels (12,228 numbers per image) into a 4x4x128 representation (2,048 numbers per image). Then, the decoder transforms the 4x4x128 representation back to 64x64x3 pixels. The neural network is trained to produce an output that matches the input image.

This is how the **encoder** is implemented:
```
      TNNetConvolution.Create({Features=}32 * NeuronMultiplier,{FeatureSize=}3,{Padding=}1,{Stride=}2,{SuppressBias=}1), //32x32
      TNNetConvolution.Create({Features=}32 * NeuronMultiplier,{FeatureSize=}3,{Padding=}1,{Stride=}1,{SuppressBias=}1),
      TNNetConvolution.Create({Features=}32 * NeuronMultiplier,{FeatureSize=}3,{Padding=}1,{Stride=}2,{SuppressBias=}1), //16x16
      TNNetConvolution.Create({Features=}32 * NeuronMultiplier,{FeatureSize=}3,{Padding=}1,{Stride=}1,{SuppressBias=}1),
      TNNetConvolution.Create({Features=}64 * NeuronMultiplier,{FeatureSize=}3,{Padding=}1,{Stride=}2,{SuppressBias=}0), //8x8
      TNNetConvolution.Create({Features=}64 * NeuronMultiplier,{FeatureSize=}3,{Padding=}1,{Stride=}1,{SuppressBias=}0),
      TNNetConvolution.Create({Features=}128 * NeuronMultiplier,{FeatureSize=}3,{Padding=}1,{Stride=}2,{SuppressBias=}1), //4x4
      TNNetConvolution.Create({Features=}128 * NeuronMultiplier,{FeatureSize=}3,{Padding=}1,{Stride=}1,{SuppressBias=}1),
```
In the encoder shown above, the resolution is decreased via `{Stride=}2`.

This is how the **decoder** is implemented:
```
      TNNetUpsample.Create(), //8x8
      TNNetConvolution.Create({Features=}128 * NeuronMultiplier,{FeatureSize=}3,{Padding=}1,{Stride=}1,{SuppressBias=}1),
      TNNetConvolution.Create({Features=}128 * NeuronMultiplier,{FeatureSize=}3,{Padding=}1,{Stride=}1,{SuppressBias=}1),
      TNNetUpsample.Create(), //16x16
      TNNetConvolution.Create({Features=}32 * NeuronMultiplier,{FeatureSize=}3,{Padding=}1,{Stride=}1,{SuppressBias=}1),
      TNNetConvolution.Create({Features=}128 * NeuronMultiplier,{FeatureSize=}3,{Padding=}1,{Stride=}1,{SuppressBias=}1),
      TNNetUpsample.Create(), //32x32
      TNNetConvolution.Create({Features=}32 * NeuronMultiplier,{FeatureSize=}3,{Padding=}1,{Stride=}1,{SuppressBias=}1),
      TNNetConvolution.Create({Features=}128 * NeuronMultiplier,{FeatureSize=}3,{Padding=}1,{Stride=}1,{SuppressBias=}1),
      TNNetUpsample.Create(), //64x64
      TNNetConvolution.Create({Features=}32 * NeuronMultiplier,{FeatureSize=}3,{Padding=}1,{Stride=}1,{SuppressBias=}1),
      TNNetConvolution.Create({Features=}32 * NeuronMultiplier,{FeatureSize=}3,{Padding=}1,{Stride=}1,{SuppressBias=}1),
      TNNetConvolutionLinear.Create({Features=}3,{FeatureSize=}1,{Padding=}0,{Stride=}1,{SuppressBias=}0),
      TNNetReLUL.Create(-40, +40, 0) // Protection against overflow
```

In the implementation above, it’s important to note the layer `TNNetUpsample`. `TNNetUpsample` is used to increase the resolution of the activation map. It converts channels (depth) into spatial data. For example, the 4x4x128 activation map is converted to 8x8x32. The number of channels is always divided by 4 while the resolution increases.

For this example, the Tiny ImageNet Dataset can be downloaded from http://cs231n.stanford.edu/tiny-imagenet-200.zip.
