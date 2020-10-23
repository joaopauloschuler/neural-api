# Learning Hypotenuse Function with TNeuralDataLoadingFit.FitLoading

In the case that your dataset is too large for RAM, you can call `TNeuralDataLoadingFit.FitLoading` shown in this example.

This example has these main steps:
* Preparing training data
* Creating the neural network
* Fitting
* Printing a test result

Training, validation and testing pairs data are created with (in this example, the method is the same):
```
  procedure TTestFitLoading.GetTrainingPair(Idx: integer; ThreadId: integer;
    pInput, pOutput: TNNetVolume);
  var
    LocalX, LocalY, Hypotenuse: TNeuralFloat;
  begin
    LocalX := Random(100);
    LocalY := Random(100);
    Hypotenuse := sqrt(LocalX*LocalX + LocalY*LocalY);
    pInput.ReSize(2,1,1);
    pInput.FData[0] := LocalX;
    pInput.FData[1] := LocalY;
    pOutput.ReSize(1,1,1);
    pOutput.FData[0] := Hypotenuse;
  end;

  procedure TTestFitLoading.GetValidationPair(Idx: integer; ThreadId: integer;
    pInput, pOutput: TNNetVolume);
  begin
    GetTrainingPair(Idx, ThreadId, pInput, pOutput);
  end;

  procedure TTestFitLoading.GetTestPair(Idx: integer; ThreadId: integer;
    pInput, pOutput: TNNetVolume);
  begin
    GetTrainingPair(Idx, ThreadId, pInput, pOutput);
  end;
```

This is how the neural network is created:
```
    NN.AddLayer([
      TNNetInput.Create(2),
      TNNetFullConnectReLU.Create(32),
      TNNetFullConnectReLU.Create(32),
      TNNetFullConnectReLU.Create(1)
    ]);
```

As you can see, there is one input layer followed by 2 fully connected layers with 32 neurons each. The last layer contains only one neuron for only one output.

This is how the fitting object is created and run:
```
    WriteLn('Computing...');
    NFit.InitialLearningRate := 0.00001;
    NFit.LearningRateDecay := 0;
    NFit.L2Decay := 0;
    NFit.InferHitFn := @LocalFloatCompare;
    NFit.FitLoading(
      NN,
      {TrainingVolumesCount=}10000,
      {ValidationVolumesCount=}1000,
      {TestVolumesCount=}1000,
      {batchsize=}32,
      {epochs=}50,
      @GetTrainingPair, @GetValidationPair, @GetTestPair
    );
```

After fitting, the neural network is then tested for 10 input values with:
```
    for Cnt := 0 to 9 do
    begin
      GetTestPair({Idx=}0, {ThreadId=}0, TestInput, TestOutput);
      NN.Compute(TestInput);
      NN.GetOutput(pOutPut);
      WriteLn
      ( 'Inputs:',
        TestInput.FData[0]:5:2,', ',
        TestInput.FData[1]:5:2,' - ',
        'Output:',
        pOutPut.Raw[0]:5:2,' ',
        ' Desired Output:',
        TestOutput.FData[0]:5:2
      );
    end;
```
