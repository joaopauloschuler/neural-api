# Separable Convolutions
Adding neurons and neuronal layers is surely a possible way to improve artificial neural networks if you have enough hardware
and computing time. In the case that you can’t afford time and hardware, you’ll look for efficiency. There is an inspirational
paper [MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications](https://arxiv.org/abs/1704.04861) . They base
their work on [separable convolutions](https://towardsdatascience.com/a-basic-introduction-to-separable-convolutions-b99ec3102728) .

As can be seen on above links, a separable convolution is a composition of 2 building blocks: 
* A depth-wise convolution followed by 
* a point-wise convolution. 

Although this readme file won't explain here what these 2 building blocks do, it's important to comment that these 2 building blocks
allow artificial neural network architectures to solve classification problems with a fraction of trainable parameters and computing power.

These 2 separable convolution building blocks (depth-wise and point-wise convolutions) are ready to use from CAI API:
```
TNNetDepthwiseConv.Create(pMultiplier, pFeatureSize, pInputPadding, pStride: integer);
TNNetDepthwiseConvLinear.Create(pMultiplier, pFeatureSize, pInputPadding, pStride: integer);
TNNetDepthwiseConvReLU.Create(pMultiplier, pFeatureSize, pInputPadding, pStride: integer);
TNNetPointwiseConvReLU.Create(pNumFeatures: integer; pSuppressBias: integer = 0);
TNNetPointwiseConvLinear.Create(pNumFeatures: integer; pSuppressBias: integer = 0);
```

You can also add separable convolutions to your network with:
```
TNNet.AddSeparableConvReLU(pNumFeatures{filters}, pFeatureSize, pInputPadding, pStride: integer; pDepthMultiplier: integer = 1; pSuppressBias: integer = 0; pAfterLayer: TNNetLayer = nil): TNNetLayer;
TNNet.AddSeparableConvLinear(pNumFeatures{filters}, pFeatureSize, pInputPadding, pStride: integer; pDepthMultiplier: integer = 1; pSuppressBias: integer = 0; pAfterLayer: TNNetLayer = nil): TNNetLayer;
```

You can also decide if you want a separable convolution or a traditional convolution at run time with:
```
function TNNet.AddConvOrSeparableConv(IsSeparable, HasReLU, HasNorm: boolean;
  pNumFeatures, pFeatureSize, pInputPadding, pStride: integer;
  PerCell: boolean; pSuppressBias: integer; RandomBias: integer;
  RandomAmplifier: integer; pAfterLayer: TNNetLayer): TNNetLayer;
```
