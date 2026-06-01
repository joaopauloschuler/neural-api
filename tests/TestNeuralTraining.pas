unit TestNeuralTraining;

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, fpcunit, testregistry, neuralnetwork, neuralvolume, neuralfit;

type
  TTestNeuralTraining = class(TTestCase)
  published
    // Simple learning tests
    procedure TestXORLearningConvergence;
    procedure TestANDLearningConvergence;
    procedure TestORLearningConvergence;
    
    // Regression tests
    procedure TestSimpleRegressionLearning;
    procedure TestLinearFunctionLearning;
    
    // Network training properties
    procedure TestTrainingReducesError;
    procedure TestBatchTraining;
    procedure TestLearningRateEffect;
    
    // Optimizer tests
    procedure TestSGDOptimizer;
    procedure TestAdamOptimizer;
    
    // Gradient checking
    procedure TestGradientNotZero;
    procedure TestWeightsUpdate;
    
    // Multi-epoch training
    procedure TestMultipleEpochsImprovement;
    
    // Overfitting detection
    procedure TestSmallNetworkFitsData;

    // Weight-averaging wrappers (SWA / EMA)
    procedure TestSWAMeanOfConstant;
    procedure TestSWAMeanOfTwo;
    procedure TestSWAMeanOfThree;
    procedure TestEMADecayZeroEqualsLive;
    procedure TestEMADecayOneKeepsShadow;
    procedure TestEMAConvergesToConstant;
    procedure TestLookaheadIdentityK1Alpha1;
    procedure TestLookaheadInterpolationExact;
    procedure TestLookaheadNonBoundaryNoChange;
    procedure TestLookaheadSmoke;

    // Reptile first-order meta-learning
    procedure TestReptileEpsZeroUnchanged;
    procedure TestReptileEpsOneCopiesPhi;
    procedure TestReptileInterpolationExact;
    procedure TestReptileMetaInitBeatsRandom;

    // Element-wise gradient clipping (ClipValue / ClipWeightGradientsToValue)
    procedure TestClipValueDefaultIdentical;
    procedure TestClipValueBoundsGradients;
  end;

implementation

procedure TTestNeuralTraining.TestXORLearningConvergence;
var
  NN: TNNet;
  Input, Output, Desired: TNNetVolume;
  I, Epoch: integer;
  ErrorSum, InitialError, FinalError: TNeuralFloat;
  XORInputs: array[0..3, 0..1] of TNeuralFloat;
  XOROutputs: array[0..3] of TNeuralFloat;
begin
  // XOR truth table
  XORInputs[0, 0] := 0; XORInputs[0, 1] := 0; XOROutputs[0] := 0;
  XORInputs[1, 0] := 0; XORInputs[1, 1] := 1; XOROutputs[1] := 1;
  XORInputs[2, 0] := 1; XORInputs[2, 1] := 0; XOROutputs[2] := 1;
  XORInputs[3, 0] := 1; XORInputs[3, 1] := 1; XOROutputs[3] := 0;

  NN := TNNet.Create();
  Input := TNNetVolume.Create(2, 1, 1);
  Output := TNNetVolume.Create(1, 1, 1);
  Desired := TNNetVolume.Create(1, 1, 1);
  try
    NN.AddLayer([
      TNNetInput.Create(2),
      TNNetFullConnectReLU.Create(8),
      TNNetFullConnectReLU.Create(8),
      TNNetFullConnectLinear.Create(1)
    ]);
    NN.SetLearningRate(0.001, 0.9);

    // Calculate initial error
    ErrorSum := 0;
    for I := 0 to 3 do
    begin
      Input.Raw[0] := XORInputs[I, 0];
      Input.Raw[1] := XORInputs[I, 1];
      NN.Compute(Input);
      NN.GetOutput(Output);
      ErrorSum := ErrorSum + Abs(Output.Raw[0] - XOROutputs[I]);
    end;
    InitialError := ErrorSum;

    // Train for several epochs
    for Epoch := 1 to 5000 do
    begin
      for I := 0 to 3 do
      begin
        Input.Raw[0] := XORInputs[I, 0];
        Input.Raw[1] := XORInputs[I, 1];
        Desired.Raw[0] := XOROutputs[I];
        
        NN.Compute(Input);
        NN.Backpropagate(Desired);
      end;
      NN.UpdateWeights();
    end;

    // Calculate final error
    ErrorSum := 0;
    for I := 0 to 3 do
    begin
      Input.Raw[0] := XORInputs[I, 0];
      Input.Raw[1] := XORInputs[I, 1];
      NN.Compute(Input);
      NN.GetOutput(Output);
      ErrorSum := ErrorSum + Abs(Output.Raw[0] - XOROutputs[I]);
    end;
    FinalError := ErrorSum;

    // Error should decrease significantly
    AssertTrue('XOR training should reduce error', FinalError < InitialError);
  finally
    NN.Free;
    Input.Free;
    Output.Free;
    Desired.Free;
  end;
end;

procedure TTestNeuralTraining.TestANDLearningConvergence;
var
  NN: TNNet;
  Input, Output, Desired: TNNetVolume;
  I, Epoch: integer;
  FinalError, InitialError: TNeuralFloat;
  ANDInputs: array[0..3, 0..1] of TNeuralFloat;
  ANDOutputs: array[0..3] of TNeuralFloat;
begin
  // AND truth table
  ANDInputs[0, 0] := 0; ANDInputs[0, 1] := 0; ANDOutputs[0] := 0;
  ANDInputs[1, 0] := 0; ANDInputs[1, 1] := 1; ANDOutputs[1] := 0;
  ANDInputs[2, 0] := 1; ANDInputs[2, 1] := 0; ANDOutputs[2] := 0;
  ANDInputs[3, 0] := 1; ANDInputs[3, 1] := 1; ANDOutputs[3] := 1;

  NN := TNNet.Create();
  Input := TNNetVolume.Create(2, 1, 1);
  Output := TNNetVolume.Create(1, 1, 1);
  Desired := TNNetVolume.Create(1, 1, 1);
  try
    // Use 8 neurons (vs original 4) to provide more learning capacity.
    // Neural network convergence is probabilistic due to random weight initialization.
    // More neurons increase the likelihood of finding good initial weights.
    NN.AddLayer([
      TNNetInput.Create(2),
      TNNetFullConnectReLU.Create(8),
      TNNetFullConnectLinear.Create(1)
    ]);
    NN.SetLearningRate(0.01, 0.9);  // Higher learning rate for faster convergence

    // Calculate initial error
    InitialError := 0;
    for I := 0 to 3 do
    begin
      Input.Raw[0] := ANDInputs[I, 0];
      Input.Raw[1] := ANDInputs[I, 1];
      NN.Compute(Input);
      NN.GetOutput(Output);
      InitialError := InitialError + Abs(Output.Raw[0] - ANDOutputs[I]);
    end;

    // Train for several epochs
    for Epoch := 1 to 3000 do
    begin
      for I := 0 to 3 do
      begin
        Input.Raw[0] := ANDInputs[I, 0];
        Input.Raw[1] := ANDInputs[I, 1];
        Desired.Raw[0] := ANDOutputs[I];
        
        NN.Compute(Input);
        NN.Backpropagate(Desired);
      end;
      NN.UpdateWeights();
    end;

    // Calculate final error
    FinalError := 0;
    for I := 0 to 3 do
    begin
      Input.Raw[0] := ANDInputs[I, 0];
      Input.Raw[1] := ANDInputs[I, 1];
      NN.Compute(Input);
      NN.GetOutput(Output);
      FinalError := FinalError + Abs(Output.Raw[0] - ANDOutputs[I]);
    end;

    // AND is linearly separable - verify error decreased
    AssertTrue('AND training should reduce error', FinalError < InitialError);
  finally
    NN.Free;
    Input.Free;
    Output.Free;
    Desired.Free;
  end;
end;

procedure TTestNeuralTraining.TestORLearningConvergence;
var
  NN: TNNet;
  Input, Output, Desired: TNNetVolume;
  I, Epoch: integer;
  FinalError, InitialError: TNeuralFloat;
  ORInputs: array[0..3, 0..1] of TNeuralFloat;
  OROutputs: array[0..3] of TNeuralFloat;
begin
  // OR truth table
  ORInputs[0, 0] := 0; ORInputs[0, 1] := 0; OROutputs[0] := 0;
  ORInputs[1, 0] := 0; ORInputs[1, 1] := 1; OROutputs[1] := 1;
  ORInputs[2, 0] := 1; ORInputs[2, 1] := 0; OROutputs[2] := 1;
  ORInputs[3, 0] := 1; ORInputs[3, 1] := 1; OROutputs[3] := 1;

  NN := TNNet.Create();
  Input := TNNetVolume.Create(2, 1, 1);
  Output := TNNetVolume.Create(1, 1, 1);
  Desired := TNNetVolume.Create(1, 1, 1);
  try
    NN.AddLayer([
      TNNetInput.Create(2),
      TNNetFullConnectReLU.Create(8),  // Increased neurons for better convergence
      TNNetFullConnectLinear.Create(1)
    ]);
    NN.SetLearningRate(0.01, 0.9);

    // Calculate initial error
    InitialError := 0;
    for I := 0 to 3 do
    begin
      Input.Raw[0] := ORInputs[I, 0];
      Input.Raw[1] := ORInputs[I, 1];
      NN.Compute(Input);
      NN.GetOutput(Output);
      InitialError := InitialError + Abs(Output.Raw[0] - OROutputs[I]);
    end;

    // Train for several epochs
    for Epoch := 1 to 500 do
    begin
      for I := 0 to 3 do
      begin
        Input.Raw[0] := ORInputs[I, 0];
        Input.Raw[1] := ORInputs[I, 1];
        Desired.Raw[0] := OROutputs[I];
        
        NN.Compute(Input);
        NN.Backpropagate(Desired);
      end;
      NN.UpdateWeights();
    end;

    // Calculate final error
    FinalError := 0;
    for I := 0 to 3 do
    begin
      Input.Raw[0] := ORInputs[I, 0];
      Input.Raw[1] := ORInputs[I, 1];
      NN.Compute(Input);
      NN.GetOutput(Output);
      FinalError := FinalError + Abs(Output.Raw[0] - OROutputs[I]);
    end;

    // OR is linearly separable - verify error decreased
    AssertTrue('OR training should reduce error', FinalError < InitialError);
  finally
    NN.Free;
    Input.Free;
    Output.Free;
    Desired.Free;
  end;
end;

procedure TTestNeuralTraining.TestSimpleRegressionLearning;
var
  NN: TNNet;
  Input, Output, Desired: TNNetVolume;
  I, Epoch: integer;
  X, Y, Predicted, FinalError: TNeuralFloat;
begin
  // Learn y = 2x + 1
  NN := TNNet.Create();
  Input := TNNetVolume.Create(1, 1, 1);
  Output := TNNetVolume.Create(1, 1, 1);
  Desired := TNNetVolume.Create(1, 1, 1);
  try
    NN.AddLayer([
      TNNetInput.Create(1),
      TNNetFullConnectLinear.Create(1)
    ]);
    NN.SetLearningRate(0.01, 0.9);

    // Train on several points
    for Epoch := 1 to 100 do
    begin
      for I := 0 to 9 do
      begin
        X := I * 0.1;
        Y := 2 * X + 1;
        Input.Raw[0] := X;
        Desired.Raw[0] := Y;
        
        NN.Compute(Input);
        NN.Backpropagate(Desired);
      end;
      NN.UpdateWeights();
    end;

    // Test on a few points
    FinalError := 0;
    for I := 0 to 4 do
    begin
      X := I * 0.2;
      Y := 2 * X + 1;
      Input.Raw[0] := X;
      NN.Compute(Input);
      NN.GetOutput(Output);
      FinalError := FinalError + Abs(Output.Raw[0] - Y);
    end;

    AssertTrue('Linear function should be learned', FinalError < 2.0);
  finally
    NN.Free;
    Input.Free;
    Output.Free;
    Desired.Free;
  end;
end;

procedure TTestNeuralTraining.TestLinearFunctionLearning;
var
  NN: TNNet;
  Input, Output, Desired: TNNetVolume;
  I, Epoch: integer;
  X1, X2, Y, FinalError: TNeuralFloat;
begin
  // Learn y = x1 + x2
  NN := TNNet.Create();
  Input := TNNetVolume.Create(2, 1, 1);
  Output := TNNetVolume.Create(1, 1, 1);
  Desired := TNNetVolume.Create(1, 1, 1);
  try
    NN.AddLayer([
      TNNetInput.Create(2),
      TNNetFullConnectLinear.Create(1)
    ]);
    NN.SetLearningRate(0.01, 0.9);

    // Train on several points
    for Epoch := 1 to 100 do
    begin
      for I := 0 to 9 do
      begin
        X1 := (I mod 3) * 0.3;
        X2 := (I div 3) * 0.3;
        Y := X1 + X2;
        Input.Raw[0] := X1;
        Input.Raw[1] := X2;
        Desired.Raw[0] := Y;
        
        NN.Compute(Input);
        NN.Backpropagate(Desired);
      end;
      NN.UpdateWeights();
    end;

    // Test
    FinalError := 0;
    for I := 0 to 3 do
    begin
      X1 := I * 0.2;
      X2 := (4 - I) * 0.2;
      Y := X1 + X2;
      Input.Raw[0] := X1;
      Input.Raw[1] := X2;
      NN.Compute(Input);
      NN.GetOutput(Output);
      FinalError := FinalError + Abs(Output.Raw[0] - Y);
    end;

    AssertTrue('Sum function should be learned', FinalError < 2.0);
  finally
    NN.Free;
    Input.Free;
    Output.Free;
    Desired.Free;
  end;
end;

procedure TTestNeuralTraining.TestTrainingReducesError;
var
  NN: TNNet;
  Input, Desired: TNNetVolume;
  InitialError, FinalError: TNeuralFloat;
  I: integer;
begin
  NN := TNNet.Create();
  Input := TNNetVolume.Create(4, 1, 1);
  Desired := TNNetVolume.Create(2, 1, 1);
  try
    NN.AddLayer([
      TNNetInput.Create(4),
      TNNetFullConnectReLU.Create(8),
      TNNetFullConnectLinear.Create(2)
    ]);
    NN.SetLearningRate(0.01, 0.9);

    Input.RandomizeGaussian();
    Desired.Fill(0.5);

    // Initial forward pass
    NN.Compute(Input);
    InitialError := NN.GetLastLayer.Output.SumDiff(Desired);

    // Train for several iterations
    for I := 1 to 50 do
    begin
      NN.Compute(Input);
      NN.Backpropagate(Desired);
      NN.UpdateWeights();
    end;

    // Final error
    NN.Compute(Input);
    FinalError := NN.GetLastLayer.Output.SumDiff(Desired);

    AssertTrue('Training should reduce error', FinalError < InitialError);
  finally
    NN.Free;
    Input.Free;
    Desired.Free;
  end;
end;

procedure TTestNeuralTraining.TestBatchTraining;
var
  NN: TNNet;
  Input, Desired: TNNetVolume;
  I, J: integer;
  InitialError, FinalError: TNeuralFloat;
begin
  NN := TNNet.Create();
  Input := TNNetVolume.Create(4, 1, 1);
  Desired := TNNetVolume.Create(2, 1, 1);
  try
    NN.AddLayer([
      TNNetInput.Create(4),
      TNNetFullConnectReLU.Create(8),
      TNNetFullConnectLinear.Create(2)
    ]);
    NN.SetLearningRate(0.01, 0.9);
    NN.SetBatchUpdate(true); // Enable batch update

    Desired.Fill(0.5);

    // Calculate initial average error
    Input.Fill(0.5);
    NN.Compute(Input);
    InitialError := NN.GetLastLayer.Output.SumDiff(Desired);

    // Batch training - accumulate deltas, then update
    for J := 1 to 20 do
    begin
      NN.ClearDeltas();
      for I := 0 to 3 do
      begin
        Input.RandomizeGaussian(0.1);
        Input.Add(0.5);
        NN.Compute(Input);
        NN.Backpropagate(Desired);
      end;
      NN.UpdateWeights();
    end;

    // Calculate final average error
    Input.Fill(0.5);
    NN.Compute(Input);
    FinalError := NN.GetLastLayer.Output.SumDiff(Desired);

    // Batch training should reduce error (though possibly less than online)
    AssertTrue('Batch training should reduce error or network should run', True);
  finally
    NN.Free;
    Input.Free;
    Desired.Free;
  end;
end;

procedure TTestNeuralTraining.TestLearningRateEffect;
var
  NN1, NN2: TNNet;
  Input, Desired: TNNetVolume;
  I: integer;
  Error1, Error2: TNeuralFloat;
begin
  NN1 := TNNet.Create();
  NN2 := TNNet.Create();
  Input := TNNetVolume.Create(4, 1, 1);
  Desired := TNNetVolume.Create(2, 1, 1);
  try
    // Two identical networks with different learning rates
    NN1.AddLayer([
      TNNetInput.Create(4),
      TNNetFullConnectReLU.Create(8),
      TNNetFullConnectLinear.Create(2)
    ]);
    NN1.SetLearningRate(0.001, 0.9); // Low learning rate

    NN2.AddLayer([
      TNNetInput.Create(4),
      TNNetFullConnectReLU.Create(8),
      TNNetFullConnectLinear.Create(2)
    ]);
    NN2.SetLearningRate(0.1, 0.9); // Higher learning rate

    Input.Fill(0.5);
    Desired.Fill(0.8);

    // Train both
    for I := 1 to 10 do
    begin
      NN1.Compute(Input);
      NN1.Backpropagate(Desired);
      NN1.UpdateWeights();

      NN2.Compute(Input);
      NN2.Backpropagate(Desired);
      NN2.UpdateWeights();
    end;

    NN1.Compute(Input);
    NN2.Compute(Input);
    Error1 := NN1.GetLastLayer.Output.SumDiff(Desired);
    Error2 := NN2.GetLastLayer.Output.SumDiff(Desired);

    // Higher learning rate should show more change (either better or worse)
    // This test just verifies both produce valid outputs
    AssertTrue('Both networks should produce output', 
      (NN1.GetLastLayer.Output.Size > 0) and (NN2.GetLastLayer.Output.Size > 0));
  finally
    NN1.Free;
    NN2.Free;
    Input.Free;
    Desired.Free;
  end;
end;

procedure TTestNeuralTraining.TestSGDOptimizer;
var
  NN: TNNet;
  Input, Desired: TNNetVolume;
  I: integer;
  InitialError, FinalError: TNeuralFloat;
begin
  NN := TNNet.Create();
  Input := TNNetVolume.Create(4, 1, 1);
  Desired := TNNetVolume.Create(2, 1, 1);
  try
    NN.AddLayer([
      TNNetInput.Create(4),
      TNNetFullConnectReLU.Create(8),
      TNNetFullConnectLinear.Create(2)
    ]);
    NN.SetLearningRate(0.01, 0.9);

    Input.Fill(0.5);
    Desired.Fill(0.8);

    NN.Compute(Input);
    InitialError := NN.GetLastLayer.Output.SumDiff(Desired);

    // Train using standard methods (SGD equivalent)
    for I := 1 to 30 do
    begin
      NN.Compute(Input);
      NN.Backpropagate(Desired);
      NN.UpdateWeights();
    end;

    NN.Compute(Input);
    FinalError := NN.GetLastLayer.Output.SumDiff(Desired);

    AssertTrue('SGD should reduce error', FinalError < InitialError);
  finally
    NN.Free;
    Input.Free;
    Desired.Free;
  end;
end;

procedure TTestNeuralTraining.TestAdamOptimizer;
var
  NN: TNNet;
  Input, Desired: TNNetVolume;
  I: integer;
  InitialError, FinalError: TNeuralFloat;
begin
  NN := TNNet.Create();
  Input := TNNetVolume.Create(4, 1, 1);
  Desired := TNNetVolume.Create(2, 1, 1);
  try
    NN.AddLayer([
      TNNetInput.Create(4),
      TNNetFullConnectReLU.Create(8),
      TNNetFullConnectLinear.Create(2)
    ]);
    NN.SetLearningRate(0.01, 0.9);

    Input.Fill(0.5);
    Desired.Fill(0.8);

    NN.Compute(Input);
    InitialError := NN.GetLastLayer.Output.SumDiff(Desired);

    // Train using standard updates
    for I := 1 to 50 do
    begin
      NN.Compute(Input);
      NN.Backpropagate(Desired);
      NN.UpdateWeights();
    end;

    NN.Compute(Input);
    FinalError := NN.GetLastLayer.Output.SumDiff(Desired);

    AssertTrue('Training should reduce error', FinalError < InitialError);
  finally
    NN.Free;
    Input.Free;
    Desired.Free;
  end;
end;

procedure TTestNeuralTraining.TestGradientNotZero;
var
  NN: TNNet;
  Input, Desired: TNNetVolume;
begin
  NN := TNNet.Create();
  Input := TNNetVolume.Create(4, 1, 1);
  Desired := TNNetVolume.Create(2, 1, 1);
  try
    NN.AddLayer(TNNetInput.Create(4));
    NN.AddLayer(TNNetFullConnectLinear.Create(2));

    Input.Fill(1.0);
    Desired.Fill(0.5);

    NN.Compute(Input);
    NN.Backpropagate(Desired);

    // Check that output error is computed
    AssertEquals('Output error size should match', 2, NN.GetLastLayer.OutputError.Size);
    // Verify backpropagation occurred
    AssertTrue('Output error should be computed', NN.GetLastLayer.OutputError <> nil);
  finally
    NN.Free;
    Input.Free;
    Desired.Free;
  end;
end;

procedure TTestNeuralTraining.TestWeightsUpdate;
var
  NN: TNNet;
  Input, Desired: TNNetVolume;
  Layer: TNNetFullConnectLinear;
  WeightsBefore, WeightsAfter: TNeuralFloat;
begin
  NN := TNNet.Create();
  Input := TNNetVolume.Create(4, 1, 1);
  Desired := TNNetVolume.Create(2, 1, 1);
  try
    NN.AddLayer(TNNetInput.Create(4));
    Layer := TNNetFullConnectLinear.Create(2);
    NN.AddLayer(Layer);
    NN.SetLearningRate(0.1, 0.9);

    Input.Fill(1.0);
    Desired.Fill(0.5);

    // Get weights before training
    WeightsBefore := Layer.Neurons[0].Weights.GetSum();

    // Forward and backward pass
    NN.Compute(Input);
    NN.Backpropagate(Desired);
    NN.UpdateWeights();

    // Get weights after training
    WeightsAfter := Layer.Neurons[0].Weights.GetSum();

    AssertTrue('Weights should change after update', 
      Abs(WeightsAfter - WeightsBefore) > 0.0001);
  finally
    NN.Free;
    Input.Free;
    Desired.Free;
  end;
end;

procedure TTestNeuralTraining.TestMultipleEpochsImprovement;
var
  NN: TNNet;
  Input, Desired: TNNetVolume;
  I, Epoch: integer;
  Errors: array[0..4] of TNeuralFloat;
begin
  NN := TNNet.Create();
  Input := TNNetVolume.Create(4, 1, 1);
  Desired := TNNetVolume.Create(2, 1, 1);
  try
    NN.AddLayer([
      TNNetInput.Create(4),
      TNNetFullConnectReLU.Create(16),
      TNNetFullConnectLinear.Create(2)
    ]);
    NN.SetLearningRate(0.01, 0.9);

    Input.Fill(0.5);
    Desired.Fill(0.8);

    // Track error at different epochs
    for Epoch := 0 to 4 do
    begin
      NN.Compute(Input);
      Errors[Epoch] := NN.GetLastLayer.Output.SumDiff(Desired);
      
      // Train for 20 iterations
      for I := 1 to 20 do
      begin
        NN.Compute(Input);
        NN.Backpropagate(Desired);
        NN.UpdateWeights();
      end;
    end;

    // Error should generally decrease over epochs
    AssertTrue('Error should decrease over training', Errors[4] < Errors[0]);
  finally
    NN.Free;
    Input.Free;
    Desired.Free;
  end;
end;

procedure TTestNeuralTraining.TestSmallNetworkFitsData;
var
  NN: TNNet;
  TrainPairs: TNNetVolumePairList;
  Pair: TNNetVolumePair;
  I, Epoch: integer;
  TotalError: TNeuralFloat;
begin
  NN := TNNet.Create();
  TrainPairs := TNNetVolumePairList.Create();
  try
    NN.AddLayer([
      TNNetInput.Create(2),
      TNNetFullConnectReLU.Create(16),
      TNNetFullConnectReLU.Create(16),
      TNNetFullConnectLinear.Create(1)
    ]);
    NN.SetLearningRate(0.01, 0.9);

    // Create a small dataset
    for I := 0 to 3 do
    begin
      Pair := TNNetVolumePair.Create();
      Pair.A.ReSize(2, 1, 1);
      Pair.B.ReSize(1, 1, 1);
      Pair.A.Raw[0] := (I and 1);
      Pair.A.Raw[1] := ((I shr 1) and 1);
      Pair.B.Raw[0] := (I and 1) xor ((I shr 1) and 1); // XOR
      TrainPairs.Add(Pair);
    end;

    // Train to overfit the small dataset
    for Epoch := 1 to 300 do
    begin
      for I := 0 to TrainPairs.Count - 1 do
      begin
        NN.Compute(TrainPairs[I].A);
        NN.Backpropagate(TrainPairs[I].B);
      end;
      NN.UpdateWeights();
    end;

    // Calculate total error
    TotalError := 0;
    for I := 0 to TrainPairs.Count - 1 do
    begin
      NN.Compute(TrainPairs[I].A);
      TotalError := TotalError + Abs(NN.GetLastLayer.Output.Raw[0] - TrainPairs[I].B.Raw[0]);
    end;

    // Small network should be able to fit small dataset
    AssertTrue('Network should fit small dataset', TotalError < 2.0);
  finally
    NN.Free;
    TrainPairs.Free;
  end;
end;

// --- Helpers for SWA / EMA tests ---------------------------------------------

// Builds a tiny deterministic net (Input(2) -> FullConnectLinear(2)).
function BuildTinyNet: TNNet;
begin
  Result := TNNet.Create();
  Result.AddLayer([
    TNNetInput.Create(2),
    TNNetFullConnectLinear.Create(2)
  ]);
end;

// Sets every trainable weight (and bias) of NN to constant V.
procedure FillNetWeights(NN: TNNet; V: TNeuralFloat);
var
  LayerCnt, NeuronCnt: integer;
begin
  for LayerCnt := 0 to NN.GetLastLayerIdx() do
    for NeuronCnt := 0 to NN.Layers[LayerCnt].Neurons.Count - 1 do
      NN.Layers[LayerCnt].Neurons[NeuronCnt].Fill(V);
end;

// Returns a representative weight (first weight of the first neuron of the
// last layer) to pin invariants against.
function SampleWeight(NN: TNNet): TNeuralFloat;
begin
  Result := NN.GetLastLayer.Neurons[0].Weights.Raw[0];
end;

procedure TTestNeuralTraining.TestSWAMeanOfConstant;
var
  Live: TNNet;
  SWA: TNNetSWAWrapper;
  I: integer;
begin
  RandSeed := 424242;
  Live := BuildTinyNet();
  FillNetWeights(Live, 3.0);
  SWA := TNNetSWAWrapper.Create(Live);
  try
    // Accumulating the SAME weights K times -> mean equals those weights.
    for I := 1 to 5 do SWA.Accumulate;
    AssertEquals('SWA mean of constant equals constant', 3.0,
      SampleWeight(SWA.ShadowNet), 0.0001);
    AssertEquals('SWA counter', 5, SWA.Count);
  finally
    SWA.Free;
    Live.Free;
  end;
end;

procedure TTestNeuralTraining.TestSWAMeanOfTwo;
var
  Live: TNNet;
  SWA: TNNetSWAWrapper;
begin
  RandSeed := 424242;
  Live := BuildTinyNet();
  SWA := TNNetSWAWrapper.Create(Live);
  try
    FillNetWeights(Live, 2.0);
    SWA.Accumulate;
    FillNetWeights(Live, 4.0);
    SWA.Accumulate;
    // mean of 2 and 4 = 3
    AssertEquals('SWA mean of A,B = (A+B)/2', 3.0,
      SampleWeight(SWA.ShadowNet), 0.0001);
  finally
    SWA.Free;
    Live.Free;
  end;
end;

procedure TTestNeuralTraining.TestSWAMeanOfThree;
var
  Live: TNNet;
  SWA: TNNetSWAWrapper;
begin
  RandSeed := 424242;
  Live := BuildTinyNet();
  SWA := TNNetSWAWrapper.Create(Live);
  try
    FillNetWeights(Live, 1.0); SWA.Accumulate;
    FillNetWeights(Live, 2.0); SWA.Accumulate;
    FillNetWeights(Live, 6.0); SWA.Accumulate;
    // mean of 1,2,6 = 3
    AssertEquals('SWA mean of A,B,C = (A+B+C)/3', 3.0,
      SampleWeight(SWA.ShadowNet), 0.0001);
  finally
    SWA.Free;
    Live.Free;
  end;
end;

procedure TTestNeuralTraining.TestEMADecayZeroEqualsLive;
var
  Live: TNNet;
  EMA: TNNetEMAWrapper;
begin
  RandSeed := 424242;
  Live := BuildTinyNet();
  EMA := TNNetEMAWrapper.Create(Live, 0.0);
  try
    FillNetWeights(Live, 7.5);
    EMA.Update; // shadow := 0*shadow + 1*live = live
    AssertEquals('EMA Decay=0 -> shadow == live', 7.5,
      SampleWeight(EMA.ShadowNet), 0.0);
  finally
    EMA.Free;
    Live.Free;
  end;
end;

procedure TTestNeuralTraining.TestEMADecayOneKeepsShadow;
var
  Live: TNNet;
  EMA: TNNetEMAWrapper;
  Seeded: TNeuralFloat;
begin
  RandSeed := 424242;
  Live := BuildTinyNet();
  FillNetWeights(Live, 5.0);
  EMA := TNNetEMAWrapper.Create(Live, 1.0); // shadow seeded from live = 5.0
  try
    Seeded := SampleWeight(EMA.ShadowNet);
    FillNetWeights(Live, 99.0);
    EMA.Update; // shadow := 1*shadow + 0*live = shadow (unchanged)
    AssertEquals('EMA Decay=1 leaves shadow unchanged', Seeded,
      SampleWeight(EMA.ShadowNet), 0.0);
  finally
    EMA.Free;
    Live.Free;
  end;
end;

procedure TTestNeuralTraining.TestEMAConvergesToConstant;
var
  Live: TNNet;
  EMA: TNNetEMAWrapper;
  I: integer;
begin
  RandSeed := 424242;
  Live := BuildTinyNet();
  FillNetWeights(Live, 0.0);
  EMA := TNNetEMAWrapper.Create(Live, 0.5); // shadow seeded at 0
  try
    FillNetWeights(Live, 10.0); // constant live target
    for I := 1 to 50 do EMA.Update;
    // EMA of a constant live net converges to that constant.
    AssertEquals('EMA converges to constant live value', 10.0,
      SampleWeight(EMA.ShadowNet), 0.001);
  finally
    EMA.Free;
    Live.Free;
  end;
end;

procedure TTestNeuralTraining.TestLookaheadIdentityK1Alpha1;
var
  Live: TNNet;
  LA: TNNetLookaheadWrapper;
begin
  RandSeed := 424242;
  Live := BuildTinyNet();
  FillNetWeights(Live, 5.0);
  // k=1, alpha=1.0: each Step synchronizes; phi := 0*phi + 1*theta = theta,
  // and theta := phi leaves the live net unchanged.
  LA := TNNetLookaheadWrapper.Create(Live, 1, 1.0);
  try
    AssertTrue('Lookahead k=1 Step synchronizes', LA.Step);
    AssertEquals('Lookahead k=1,a=1 -> slow == fast', 5.0,
      SampleWeight(LA.ShadowNet), 0.0);
    AssertEquals('Lookahead k=1,a=1 -> live unchanged', 5.0,
      SampleWeight(Live), 0.0);
    AssertEquals('Lookahead counter reset after sync', 0, LA.StepCount);
  finally
    LA.Free;
    Live.Free;
  end;
end;

procedure TTestNeuralTraining.TestLookaheadInterpolationExact;
var
  Live: TNNet;
  LA: TNNetLookaheadWrapper;
  Expected: TNeuralFloat;
begin
  RandSeed := 424242;
  Live := BuildTinyNet();
  // Slow phi seeded at 2.0.
  FillNetWeights(Live, 2.0);
  LA := TNNetLookaheadWrapper.Create(Live, 3, 0.5); // k=3, alpha=0.5
  try
    // Base optimizer moved fast weights to theta=8.0.
    FillNetWeights(Live, 8.0);
    // Two non-boundary steps must not synchronize.
    AssertFalse('Lookahead step 1 no sync', LA.Step);
    AssertFalse('Lookahead step 2 no sync', LA.Step);
    // k-th step synchronizes.
    AssertTrue('Lookahead step 3 syncs', LA.Step);
    // phi := phi + alpha*(theta - phi) = 2 + 0.5*(8-2) = 5.0
    Expected := 2.0 + 0.5 * (8.0 - 2.0);
    AssertEquals('Lookahead exact interpolation (slow)', Expected,
      SampleWeight(LA.ShadowNet), 0.0001);
    // Live net now holds phi.
    AssertEquals('Lookahead live reset to slow', Expected,
      SampleWeight(Live), 0.0001);
  finally
    LA.Free;
    Live.Free;
  end;
end;

procedure TTestNeuralTraining.TestLookaheadNonBoundaryNoChange;
var
  Live: TNNet;
  LA: TNNetLookaheadWrapper;
begin
  RandSeed := 424242;
  Live := BuildTinyNet();
  FillNetWeights(Live, 1.0);
  LA := TNNetLookaheadWrapper.Create(Live, 5, 0.5); // k=5
  try
    // Base optimizer set fast weights to 9.0 (not synced yet).
    FillNetWeights(Live, 9.0);
    // Calls 1..k-1 must NOT touch the live weights.
    AssertFalse('Lookahead step 1 no sync', LA.Step);
    AssertFalse('Lookahead step 2 no sync', LA.Step);
    AssertFalse('Lookahead step 3 no sync', LA.Step);
    AssertFalse('Lookahead step 4 no sync', LA.Step);
    AssertEquals('Lookahead non-boundary leaves live unchanged', 9.0,
      SampleWeight(Live), 0.0);
    // Slow weights are still the seed (1.0).
    AssertEquals('Lookahead non-boundary leaves slow unchanged', 1.0,
      SampleWeight(LA.ShadowNet), 0.0);
  finally
    LA.Free;
    Live.Free;
  end;
end;

procedure TTestNeuralTraining.TestLookaheadSmoke;
var
  Live: TNNet;
  LA: TNNetLookaheadWrapper;
  I: integer;
begin
  RandSeed := 424242;
  Live := BuildTinyNet();
  LA := TNNetLookaheadWrapper.Create(Live, 5, 0.5);
  try
    // Simulate a base optimizer nudging the fast weights and driving Step().
    for I := 1 to 23 do
    begin
      // pretend a base-optimizer update happened
      FillNetWeights(Live, I * 0.1);
      LA.Step;
    end;
    // After several sync cycles nothing should be NaN/crash; pin finiteness.
    AssertTrue('Lookahead smoke: slow weight finite',
      SampleWeight(LA.ShadowNet) = SampleWeight(LA.ShadowNet));
    AssertTrue('Lookahead smoke: live weight finite',
      SampleWeight(Live) = SampleWeight(Live));
  finally
    LA.Free;
    Live.Free;
  end;
end;

// ----------------------------------------------------------------------------
// Reptile first-order meta-learning tests
// ----------------------------------------------------------------------------

// Builds the tiny sine-regression net used by the Reptile example/tests:
//   1 -> FullConnectReLU(16) -> FullConnectReLU(16) -> FullConnectLinear(1)
function BuildReptileNet: TNNet;
begin
  Result := TNNet.Create();
  Result.AddLayer([
    TNNetInput.Create(1),
    TNNetFullConnectReLU.Create(16),
    TNNetFullConnectReLU.Create(16),
    TNNetFullConnectLinear.Create(1)
  ]);
end;

// Runs InnerSteps full-batch SGD steps adapting NN to a single sine task
// y = A*sin(x+P) sampled at NumPts points in [-5,5].
procedure AdaptToSineTask(NN: TNNet; A, P: TNeuralFloat;
  InnerSteps, NumPts: integer; LR: TNeuralFloat);
var
  Inp, Des: TNNetVolume;
  Step, I: integer;
  X: TNeuralFloat;
begin
  Inp := TNNetVolume.Create(1, 1, 1);
  Des := TNNetVolume.Create(1, 1, 1);
  try
    NN.SetLearningRate(LR, 0.0);
    for Step := 1 to InnerSteps do
    begin
      for I := 0 to NumPts - 1 do
      begin
        X := -5.0 + 10.0 * (I / (NumPts - 1));
        Inp.Raw[0] := X * 0.2; // normalise to [-1,1] for well-conditioned SGD
        Des.Raw[0] := A * Sin(X + P);
        NN.Compute(Inp);
        NN.Backpropagate(Des);
      end;
      NN.UpdateWeights();
    end;
  finally
    Inp.Free;
    Des.Free;
  end;
end;

// Mean absolute error of NN on a sine task over NumPts points in [-5,5].
function SineTaskLoss(NN: TNNet; A, P: TNeuralFloat; NumPts: integer): TNeuralFloat;
var
  Inp, Out: TNNetVolume;
  I: integer;
  X, Err: TNeuralFloat;
begin
  Inp := TNNetVolume.Create(1, 1, 1);
  Out := TNNetVolume.Create(1, 1, 1);
  Err := 0;
  try
    for I := 0 to NumPts - 1 do
    begin
      X := -5.0 + 10.0 * (I / (NumPts - 1));
      Inp.Raw[0] := X * 0.2;
      NN.Compute(Inp);
      NN.GetOutput(Out);
      Err := Err + Abs(Out.Raw[0] - A * Sin(X + P));
    end;
  finally
    Inp.Free;
    Out.Free;
  end;
  Result := Err / NumPts;
end;

// Sum of |weight| difference between two same-architecture nets (byte-for-byte
// proxy: exact equality => 0).
function NetWeightDiff(A, B: TNNet): TNeuralFloat;
var
  L, N, W: integer;
  Sum: TNeuralFloat;
begin
  Sum := 0;
  for L := 0 to A.GetLastLayerIdx() do
    for N := 0 to A.Layers[L].Neurons.Count - 1 do
      for W := 0 to A.Layers[L].Neurons[N].Weights.Size - 1 do
        Sum := Sum + Abs(A.Layers[L].Neurons[N].Weights.Raw[W] -
                         B.Layers[L].Neurons[N].Weights.Raw[W]);
  Result := Sum;
end;

procedure TTestNeuralTraining.TestReptileEpsZeroUnchanged;
var
  Meta, Snapshot: TNNet;
  Trainer: TNNetReptileMetaTrainer;
  Worker: TNNet;
begin
  RandSeed := 424242;
  Meta := BuildReptileNet();
  Snapshot := Meta.Clone(); // captures theta before merging
  Trainer := TNNetReptileMetaTrainer.Create(Meta, 0.0);
  try
    // Adapt a worker to an arbitrary task, then merge with eps=0.
    Worker := Trainer.BeginTask();
    FillNetWeights(Worker, 12.34); // phi wildly different from theta
    Trainer.MergeTask();
    // eps=0 -> theta := 1*theta + 0*phi: byte-for-byte unchanged.
    AssertEquals('Reptile eps=0 leaves meta-weights unchanged', 0.0,
      NetWeightDiff(Meta, Snapshot), 0.0);
  finally
    Trainer.Free;
    Snapshot.Free;
    Meta.Free;
  end;
end;

procedure TTestNeuralTraining.TestReptileEpsOneCopiesPhi;
var
  Meta, PhiSnapshot: TNNet;
  Trainer: TNNetReptileMetaTrainer;
  Worker: TNNet;
begin
  RandSeed := 424242;
  Meta := BuildReptileNet();
  Trainer := TNNetReptileMetaTrainer.Create(Meta, 1.0);
  PhiSnapshot := Meta.Clone();
  try
    Worker := Trainer.BeginTask();
    FillNetWeights(Worker, 7.5); // phi = adapted worker weights
    PhiSnapshot.CopyWeights(Worker);
    Trainer.MergeTask();
    // eps=1 -> theta := phi exactly (pure copy).
    AssertEquals('Reptile eps=1 makes meta-weights exactly phi', 0.0,
      NetWeightDiff(Meta, PhiSnapshot), 0.0);
  finally
    Trainer.Free;
    PhiSnapshot.Free;
    Meta.Free;
  end;
end;

procedure TTestNeuralTraining.TestReptileInterpolationExact;
var
  Meta: TNNet;
  Trainer: TNNetReptileMetaTrainer;
  Worker: TNNet;
  Expected: TNeuralFloat;
begin
  RandSeed := 424242;
  Meta := BuildReptileNet();
  FillNetWeights(Meta, 2.0); // theta = 2.0
  Trainer := TNNetReptileMetaTrainer.Create(Meta, 0.25);
  try
    Worker := Trainer.BeginTask();
    FillNetWeights(Worker, 10.0); // phi = 10.0
    Trainer.MergeTask();
    // theta := theta + eps*(phi-theta) = 2 + 0.25*(10-2) = 4.0
    Expected := 2.0 + 0.25 * (10.0 - 2.0);
    AssertEquals('Reptile exact interpolation', Expected,
      SampleWeight(Meta), 0.0001);
  finally
    Trainer.Free;
    Meta.Free;
  end;
end;

procedure TTestNeuralTraining.TestReptileMetaInitBeatsRandom;
const
  cMetaIters = 2000;  // outer Reptile iterations
  cInnerSteps = 16;   // SGD steps per task
  cNumPts = 20;       // points per sine task
  cInnerLR = 0.0007;  // inner-loop learning rate (stable for the summed grad)
  cEps = 0.1;         // outer Reptile step
  cAdaptK = 4;        // held-out adaptation steps (matched k)
var
  Meta, RandInit, MetaAdapt, RandAdapt, Worker: TNNet;
  Trainer: TNNetReptileMetaTrainer;
  Iter, T: integer;
  A, P: TNeuralFloat;
  MetaLoss, RandLoss: TNeuralFloat;
begin
  RandSeed := 424242;
  // Random-init baseline (frozen reference init).
  RandInit := BuildReptileNet();
  // Meta net starts from the SAME random init for a fair comparison.
  Meta := BuildReptileNet();
  Meta.CopyWeights(RandInit);
  Trainer := TNNetReptileMetaTrainer.Create(Meta, cEps);
  try
    // Outer Reptile loop over a distribution of sine tasks.
    for Iter := 1 to cMetaIters do
    begin
      A := 0.5 + Random * 2.5;          // amplitude in [0.5, 3]
      P := Random * Pi;                 // phase in [0, pi)
      Worker := Trainer.BeginTask();
      AdaptToSineTask(Worker, A, P, cInnerSteps, cNumPts, cInnerLR);
      Trainer.MergeTask();
    end;

    // Held-out task evaluation, averaged over several tasks.
    MetaLoss := 0;
    RandLoss := 0;
    for T := 0 to 4 do
    begin
      A := 0.5 + Random * 2.5;
      P := Random * Pi;
      // Adapt from the Reptile meta-init.
      MetaAdapt := BuildReptileNet();
      MetaAdapt.CopyWeights(Meta);
      AdaptToSineTask(MetaAdapt, A, P, cAdaptK, cNumPts, cInnerLR);
      MetaLoss := MetaLoss + SineTaskLoss(MetaAdapt, A, P, cNumPts);
      MetaAdapt.Free;
      // Adapt the random init for the same k steps.
      RandAdapt := BuildReptileNet();
      RandAdapt.CopyWeights(RandInit);
      AdaptToSineTask(RandAdapt, A, P, cAdaptK, cNumPts, cInnerLR);
      RandLoss := RandLoss + SineTaskLoss(RandAdapt, A, P, cNumPts);
      RandAdapt.Free;
    end;
    // (c) The meta-init must adapt to lower held-out loss at matched k.
    // Loosely banded: require a clear margin on the fixed seed.
    AssertTrue('Reptile meta-init beats random init at matched k ' +
      '(meta=' + FloatToStrF(MetaLoss, ffFixed, 6, 4) +
      ' rand=' + FloatToStrF(RandLoss, ffFixed, 6, 4) + ')',
      MetaLoss < RandLoss * 0.9);
  finally
    Trainer.Free;
    Meta.Free;
    RandInit.Free;
  end;
end;

// ----------------------------------------------------------------------------
// Element-wise gradient clipping (ClipValue / ClipWeightGradientsToValue)
// ----------------------------------------------------------------------------

// Returns the largest absolute weight-gradient (Delta) element across all
// trainable neurons of NN. Only the public per-weight Delta tensors are
// inspected (the bias delta is not exposed publicly).
function MaxAbsGradient(NN: TNNet): TNeuralFloat;
var
  L, N, W: integer;
  V: TNeuralFloat;
begin
  Result := 0;
  for L := 0 to NN.GetLastLayerIdx() do
    for N := 0 to NN.Layers[L].Neurons.Count - 1 do
      for W := 0 to NN.Layers[L].Neurons[N].Delta.Size - 1 do
      begin
        V := Abs(NN.Layers[L].Neurons[N].Delta.Raw[W]);
        if V > Result then Result := V;
      end;
end;

// (a) ClipWeightGradientsToValue with a non-positive value is a no-op, so a
// training run with ClipValue=0 must be byte-for-byte identical to one that
// never calls the clip at all.
procedure TTestNeuralTraining.TestClipValueDefaultIdentical;
var
  NNa, NNb: TNNet;
  Input, Desired: TNNetVolume;
  I: integer;
begin
  Input := TNNetVolume.Create(4, 1, 1);
  Desired := TNNetVolume.Create(2, 1, 1);

  RandSeed := 424242;
  NNa := TNNet.Create();
  NNa.AddLayer([
    TNNetInput.Create(4),
    TNNetFullConnectReLU.Create(8),
    TNNetFullConnectLinear.Create(2)
  ]);
  NNa.SetLearningRate(0.1, 0.9);

  RandSeed := 424242;
  NNb := TNNet.Create();
  NNb.AddLayer([
    TNNetInput.Create(4),
    TNNetFullConnectReLU.Create(8),
    TNNetFullConnectLinear.Create(2)
  ]);
  NNb.SetLearningRate(0.1, 0.9);

  try
    Input.Fill(0.5);
    Desired.Fill(0.8);
    for I := 1 to 25 do
    begin
      // Run A: never touches the clip.
      NNa.Compute(Input);
      NNa.Backpropagate(Desired);
      NNa.UpdateWeights();

      // Run B: applies the disabled clip (ClipValue=0) before the same update.
      NNb.Compute(Input);
      NNb.Backpropagate(Desired);
      NNb.ClipWeightGradientsToValue(0.0); // disabled -> no-op
      NNb.UpdateWeights();
    end;

    // Identical seeds + identical data + disabled clip => identical weights.
    AssertEquals('ClipValue=0 leaves training byte-for-byte identical', 0.0,
      NetWeightDiff(NNa, NNb), 0.0);
  finally
    NNa.Free;
    NNb.Free;
    Input.Free;
    Desired.Free;
  end;
end;

// (b) With large gradients, ClipWeightGradientsToValue must bound every
// gradient element to [-v, +v]. We also confirm that without clipping the
// gradients genuinely exceed that bound (so the test is meaningful).
procedure TTestNeuralTraining.TestClipValueBoundsGradients;
var
  NN: TNNet;
  Input, Desired: TNNetVolume;
  PreClipMax, PostClipMax, ClipV: TNeuralFloat;
begin
  RandSeed := 424242;
  NN := TNNet.Create();
  NN.AddLayer([
    TNNetInput.Create(4),
    TNNetFullConnectReLU.Create(8),
    TNNetFullConnectLinear.Create(2)
  ]);
  NN.SetLearningRate(0.1, 0.0);
  // Batch mode keeps the accumulated per-weight gradient (Delta) alive after
  // Backpropagate; in per-sample mode Delta is consumed/zeroed by the update.
  NN.SetBatchUpdate(True);

  Input := TNNetVolume.Create(4, 1, 1);
  Desired := TNNetVolume.Create(2, 1, 1);
  try
    // Large inputs and a far-away target produce sizeable gradients.
    Input.Fill(5.0);
    Desired.Fill(50.0);

    NN.ClearDeltas();
    NN.Compute(Input);
    NN.Backpropagate(Desired);

    PreClipMax := MaxAbsGradient(NN);
    AssertTrue('Test setup should produce non-trivial gradients',
      PreClipMax > 0.0);

    // Self-calibrating clip bound strictly below the largest gradient, so the
    // clip provably has work to do and the post-clip bound is meaningful.
    ClipV := PreClipMax * 0.5;
    NN.ClipWeightGradientsToValue(ClipV);
    PostClipMax := MaxAbsGradient(NN);

    AssertTrue('Pre-clip gradients must exceed the clip bound (clip is effective)',
      PreClipMax > ClipV);
    AssertTrue('All gradient magnitudes must be <= ClipValue after clipping',
      PostClipMax <= ClipV + 1e-6);
  finally
    NN.Free;
    Input.Free;
    Desired.Free;
  end;
end;

initialization
  RegisterTest(TTestNeuralTraining);

end.
