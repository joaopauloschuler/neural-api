# Autoencoder Example
<p><img src="../../docs/autoencoder_small.png"></img></p>

Autoencoders have 2 pieces: an **encoder** and a **decoder**. In this example, the encoder encodes an image
from the [TinyImageNet dataset](https://paperswithcode.com/dataset/tiny-imagenet) with 64x64x3 pixels (12,228 numbers per image) into a 4x4x128 representation (2,048 numbers per image). The neural network is trained to produce an output that matches the input image.

This is how the **encoder** is implemented:
```
      TNNetConvolution.Create(32 * NeuronMultiplier,3,1,2,1), //32x32
      TNNetConvolution.Create(32 * NeuronMultiplier,3,1,1,1),
      TNNetConvolution.Create(32 * NeuronMultiplier,3,1,2,1), //16x16
      TNNetConvolution.Create(32 * NeuronMultiplier,3,1,1,1),
      TNNetConvolution.Create(64 * NeuronMultiplier,3,1,2,0), //8x8
      TNNetConvolution.Create(64 * NeuronMultiplier,3,1,1,0),
      TNNetConvolution.Create(128 * NeuronMultiplier,3,1,2,1), //4x4
      TNNetConvolution.Create(128 * NeuronMultiplier,3,1,1,1),
```

This is how the **decoder** is implemented:
```
      TNNetUpsample.Create(), //8x8
      TNNetConvolution.Create(128 * NeuronMultiplier,3,1,1,1),
      TNNetConvolution.Create(128 * NeuronMultiplier,3,1,1,1),
      TNNetUpsample.Create(), //16x16
      TNNetConvolution.Create(32 * NeuronMultiplier,3,1,1,1),
      TNNetConvolution.Create(128 * NeuronMultiplier,3,1,1,1),
      TNNetUpsample.Create(), //32x32
      TNNetConvolution.Create(32 * NeuronMultiplier,3,1,1,1),
      TNNetConvolution.Create(128 * NeuronMultiplier,3,1,1,1),
      TNNetUpsample.Create(), //64x64
      TNNetConvolution.Create(32 * NeuronMultiplier,3,1,1,1),
      TNNetConvolution.Create(32 * NeuronMultiplier,3,1,1,1),
      TNNetConvolutionLinear.Create(3,1,0,1,0),
      TNNetReLUL.Create(-40, +40, 0) // Protection against overflow
```

In the implementation above, it’s important to note the layer `TNNetUpsample`. `TNNetUpsample` is used to increase the resolution of the activation map. It converts channels (depth) into spatial data. For example, the 4x4x128 activation map is converted to 8x8x32. The number of channels is always divided by 4 while the resolution increases.

For this example, the Tiny ImageNet Dataset can be downloaded from http://cs231n.stanford.edu/tiny-imagenet-200.zip.
