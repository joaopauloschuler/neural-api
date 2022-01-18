# Learning boolean functions Xor And Or

This example has these main steps:
* Preparing training data
* Creating the neural network
* Fitting
* Printing a test result

These are the inputs and expected outputs:
```
const cs_inputs : TBackInput =
  ( // x1,   x2
    ( 0.1,  0.1), // False, False
    ( 0.1,  0.8), // False, True
    ( 0.8,  0.1), // True,  False
    ( 0.8,  0.8)  // True,  True
  );

const cs_outputs : TBackOutput =
  (// XOR, AND,   OR
    ( 0.1, 0.1, 0.1),
    ( 0.8, 0.1, 0.8),
    ( 0.8, 0.1, 0.8),
    ( 0.1, 0.8, 0.8)
  );
```
The first row in `reluoutputs` has expected outputs for **XOR**, **AND** and **OR** boolean functions while the first row of
`inputs` contains inputs for these 3 boolean functions. All 3 boolean functions will be trained together in this example.

This is how the training data is prepared with training pairs (input,output):
```
    TrainingPairs := TNNetVolumePairList.Create();
    ...
    for Cnt := Low(cs_inputs) to High(cs_inputs) do
    begin
      TrainingPairs.Add(
        TNNetVolumePair.Create(
          TNNetVolume.Create(vInputs[Cnt]),
          TNNetVolume.Create(vOutput[Cnt])
        )
      );
    end;
```
In this example, values smaller than 0.5 mean **false** while values bigger than 0.5 mean **true**. This is called **monopolar** encoding.

CAI also supports bipolar encoding (-1, +1). Please have a look directly into the source code at these 2 methods:
* `EnableMonopolarHitComparison()`
* `EnableBipolarHitComparison()`

This is how the neural network is created:
```
    NN := TNNet.Create();
    ...
    NN.AddLayer( TNNetInput.Create(2) );
    NN.AddLayer( TNNetFullConnect.Create(3) );
    NN.AddLayer( TNNetFullConnectLinear.Create(3) );
```

As you can see, there is one input layer followed by 2 fully connected layers with 3 neurons each. The furst fully connected layer has hyperbolic tangent as activation function while se second layer has no activation function.

This is how the fitting object is created and run:
```
    NFit := TNeuralFit.Create();
    ...
    NFit.InitialLearningRate := 0.01;
    NFit.LearningRateDecay := 0;
    NFit.L2Decay := 0;
    NFit.Verbose := false;
    NFit.HideMessages();
    NFit.Fit(NN, TrainingPairs, nil, nil, {batchsize=}4, {epochs=}6000);
```

The neural network is then tested for each input with:
```
    // tests the learning
    for Cnt := Low(inputs) to High(inputs) do
    begin
      NN.Compute(vInputs[Cnt]);
      NN.GetOutput(pOutPut);
      WriteLn
      (
        ' Output:',
        pOutPut.Raw[0]:5:2,' ',
        pOutPut.Raw[1]:5:2,' ',
        pOutPut.Raw[2]:5:2,
        ' - Training/Desired Output:',
        vOutput[cnt][0]:5:2,' ',
        vOutput[cnt][1]:5:2,' ' ,
        vOutput[cnt][2]:5:2,' '
      );
    end;
```
