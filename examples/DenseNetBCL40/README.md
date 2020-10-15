# DenseNetBC L40 Example
This source code example shows:
* How to implement a DenseNet.
* How to define a custom learning rate schedule.

## Creating a DenseNet
You can find more about DenseNets reading the paper [Densely Connected Convolutional Networks](https://arxiv.org/abs/1608.06993) and
having a look [here](https://towardsdatascience.com/review-densenet-image-classification-b6631a8ef803).
DenseNet building blocks are:
* `THistoricalNets.AddDenseNetBlock`.
* `THistoricalNets.AddDenseNetTransition`.

If you read the paper [Densely Connected Convolutional Networks](https://arxiv.org/abs/1608.06993), the following code will make sense
to you:
```
function THistoricalNets.AddDenseNetBlock(pUnits, k: integer;
  BottleNeck: integer = 0;
  supressBias: integer = 1): TNNetLayer;
var
  UnitCnt: integer;
  PreviousLayer: TNNetLayer;
begin
  if pUnits > 0 then
  begin
    for UnitCnt := 1 to pUnits do
    begin
      PreviousLayer := GetLastLayer();
      if BottleNeck > 0 then
      begin
        AddMovingNorm(false, 0, 0);
        AddLayer( TNNetReLU.Create() );
        AddLayer( TNNetPointwiseConvLinear.Create(BottleNeck, supressBias) );
      end;
      AddMovingNorm(false, 0, 0);
      AddLayer( TNNetReLU.Create() );
      AddLayer( TNNetConvolutionLinear.Create(k, {featuresize}3, {padding}1, {stride}1, supressBias) );
      AddLayer( TNNetDeepConcat.Create([PreviousLayer, GetLastLayer()]) );
    end;
  end;
  Result := GetLastLayer();
end;

function THistoricalNets.AddDenseNetTransition(
  Compression: TNeuralFloat = 0.5;
  supressBias: integer = 1;
  HasAvgPool: boolean = true): TNNetLayer;
begin
  AddChannelMovingNorm(false, 0, 0);
  AddLayer( TNNetReLU.Create() );
  AddCompression(Compression, supressBias);
  if HasAvgPool
    then Result := AddLayer( TNNetAvgPool.Create(2) )
    else Result := AddLayer( TNNetMaxPool.Create(2) );
end;
```

## Custom Learning Rate Schedule
In this example, besides showing how to implement a [DenseNet](https://github.com/liuzhuang13/DenseNet), 
this example also shows how CAI users can implement custom learning rate schedules. This is done via `CustomLearningRateScheduleObjFn` 
or `CustomLearningRateScheduleFn` depending if your schedule belongs to an object or not.

This is an example from a custom learning rate schedule:
```
  function TTestCNNAlgo.DenseNetLearningRateSchedule(Epoch: integer): single;
  begin
    if Epoch < 150
      then Result := 0.001
    else if epoch < 225
      then Result :=  0.0001
    else
      Result := 0.00001;
  end;
```

And this is how you can link your custom schedule to the fitting method:
```
    NeuralFit := TNeuralImageFit.Create;
    ...
    NeuralFit.CustomLearningRateScheduleObjFn := @Self.DenseNetLearningRateSchedule;
    NeuralFit.Fit(NN, ImgTrainingVolumes, ImgValidationVolumes, ImgTestVolumes, NumClasses, {batchsize=}64, {epochs=}300);
```
