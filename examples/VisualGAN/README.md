# Artificial Art Example
<img src="https://github.com/joaopauloschuler/neural-api/blob/master/docs/art2.jpg" height="400"></img>

In this example, 2 neural networks will enter an arms race via a [Generative Adversarial Neural Network](https://en.wikipedia.org/wiki/Generative_adversarial_network): `FGenerative` and `FDiscriminator`. The Generative network will learn how to create images while the discriminator will compare these images against the CIFAR-10 dataset.

## The Source Code
The discriminator `FDiscriminator` is trained via `TNeuralDataLoadingFit`:
```
  FFit.OnAfterEpoch := @Self.DiscriminatorOnAfterEpoch;
  FFit.OnAfterStep := @Self.DiscriminatorOnAfterStep;
  FFit.OnStart := @Self.DiscriminatorOnStart;
  FFit.DataAugmentationFn := @Self.DiscriminatorAugmentation;
  ...
  {$ifdef OpenCL}
  if FHasOpenCL then
  begin
    FFit.EnableOpenCL(FEasyOpenCL.PlatformIds[0], FEasyOpenCL.Devices[0]);
    Generative.EnableOpenCL(FEasyOpenCL.PlatformIds[0], FEasyOpenCL.Devices[0]);
  end;
  {$endif}
  //FFit.FitLoading(FDiscriminator, 64*10, 500, 500, 64, 35000, @GetDiscriminatorTrainingPair, nil, nil); // This line does the same as below
  FFit.FitLoading(FDiscriminator, 64*10, 500, 500, 64, 35000, @GetDiscriminatorTrainingProc, nil, nil); // This line does the same as above
```

From the above code, some events are used and are interesting to note:
* `FFit.OnStart := @Self.DiscriminatorOnStart;` is responsible for building required memory structures.
* `FFit.OnAfterEpoch := @Self.DiscriminatorOnAfterEpoch;` calls the generative training. Therefore, the generative training is done after each discriminator epoch.

In order to show how `TNeuralDataLoadingFit` loads training data, above implementation has 2 equivalent calls (the first is commented) with both `GetDiscriminatorTrainingPair` and `GetDiscriminatorTrainingProc`.

## Results
Can you spot face like characteristics such as eyes and mouths?

<img src="https://github.com/joaopauloschuler/neural-api/blob/master/docs/art3.png"></img>

<img src="https://github.com/joaopauloschuler/neural-api/blob/master/docs/art4.png"></img>

<img src="https://github.com/joaopauloschuler/neural-api/blob/master/docs/art5.png"></img>



