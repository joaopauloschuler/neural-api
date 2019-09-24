# DenseNetBC L40
In this example, besides showing how to implement a [DenseNet](https://github.com/liuzhuang13/DenseNet), 
this example shows how CAI users can implement custom learning rate schedules. This is done via `CustomLearningRateScheduleObjFn` 
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
