# Learning Hypotenuse Function

This example has these main steps:
* Preparing training data
* Creating the neural network
* Fitting
* Printing a test result

Training, validation and testing pairs data are created with:
```
  function CreateHypotenusePairList(MaxCnt: integer): TNNetVolumePairList;
  var
    Cnt: integer;
    LocalX, LocalY, Hypotenuse: TNeuralFloat;
  begin
    Result := TNNetVolumePairList.Create();
    for Cnt := 1 to MaxCnt do
    begin
      LocalX := Random(100);
      LocalY := Random(100);
      Hypotenuse := sqrt(LocalX*LocalX + LocalY*LocalY);

      Result.Add(
        TNNetVolumePair.Create(
          TNNetVolume.Create([LocalX, LocalY]),
          TNNetVolume.Create([Hypotenuse])
        )
      );
    end;
  end;
...
    TrainingPairs := CreateHypotenusePairList(10000);
    ValidationPairs := CreateHypotenusePairList(1000);
    TestPairs := CreateHypotenusePairList(1000);
```

This is how the neural network is created:
```
    NN.AddLayer([
      TNNetInput.Create(2),
      TNNetFullConnectReLU.Create(32),
      TNNetFullConnectReLU.Create(32),
      TNNetFullConnectLinear.Create(1)
    ]);
```

As you can see, there is one input layer followed by 2 fully connected layers `TNNetFullConnectReLU` with 32 neurons each. The last layer `TNNetFullConnectLinear` contains only one neuron for only one output. This last layer doesn't have a ReLU as the hypotenuse results into a positive value.

This is how the fitting object is created and run:
```
    WriteLn('Computing...');
    NFit.InitialLearningRate := 0.00001;
    NFit.LearningRateDecay := 0;
    NFit.L2Decay := 0;
    NFit.InferHitFn := @LocalFloatCompare;
    NFit.Fit(NN, TrainingPairs, ValidationPairs, TestPairs, {batchsize=}32, {epochs=}50);
```

After fitting, the neural network is then tested for 10 input values with:
```
    for Cnt := 0 to 9 do
    begin
      NN.Compute(TestPairs[Cnt].I);
      NN.GetOutput(pOutPut);
      WriteLn
      ( 'Inputs:',
        TestPairs[Cnt].I.FData[0]:5:2,', ',
        TestPairs[Cnt].I.FData[1]:5:2,' - ',
        'Output:',
        pOutPut.Raw[0]:5:2,' ',
        ' Desired Output:',
        TestPairs[Cnt].O.FData[0]:5:2
      );
    end;
```
