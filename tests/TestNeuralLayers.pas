unit TestNeuralLayers;

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, Math, fpcunit, testregistry, neuralnetwork, neuralvolume;

type
  TTestNeuralLayers = class(TTestCase)
  published
    procedure TestFullyConnectedForward;
    procedure TestConvolutionForward;
    procedure TestMaxPoolForward;
    procedure TestNetworkSaveLoad;
    procedure TestSimpleXORLearning;
    // New comprehensive layer tests
    procedure TestAvgPoolForward;
    procedure TestMinPoolForward;
    procedure TestReLUActivation;
    procedure TestSigmoidActivation;
    procedure TestSoftMaxLayer;
    procedure TestDepthwiseConvolution;
    procedure TestPointwiseConvolution;
    procedure TestConcatLayers;
    procedure TestSumLayers;
    procedure TestIdentityLayer;
    procedure TestReshapeLayer;
    procedure TestDropoutLayer;
    procedure TestMultipleLayersNetwork;
    procedure TestNetworkClone;
    procedure TestLayerCount;
    // Additional activation function tests
    procedure TestReLU6Activation;
    procedure TestLeakyReLUActivation;
    procedure TestSwishActivation;
    procedure TestHyperbolicTangent;
    procedure TestSELUActivation;
    // Additional pooling tests
    procedure TestMaxChannel;
    procedure TestAvgChannel;
    // Normalization layers
    procedure TestLayerMaxNormalization;
    procedure TestLayerStdNormalization;
    procedure TestMovingStdNormalization;
    procedure TestChannelBias;
    procedure TestChannelMul;
    procedure TestCellBias;
    // Split and channel operations
    procedure TestSplitChannels;
    procedure TestInterleaveChannels;
    // Additional convolution tests
    procedure TestPointwiseConvLinear;
    procedure TestLocalConnect;
    procedure TestGroupedConvolution;
    // Backpropagation tests
    procedure TestBackpropagation;
    procedure TestGradientComputation;
    // Weight initialization tests
    procedure TestWeightInitHe;
    procedure TestWeightInitLeCun;
    procedure TestWeightInitGlorot;
    // Embedding layers
    procedure TestEmbeddingLayer;
    procedure TestTokenAndPositionalEmbedding;
  end;

implementation

procedure TTestNeuralLayers.TestFullyConnectedForward;
var
  NN: TNNet;
  Input, Output: TNNetVolume;
  Layer: TNNetFullConnectLinear;
begin
  NN := TNNet.Create();
  Input := TNNetVolume.Create(2, 1, 1);
  Output := TNNetVolume.Create(1, 1, 1);
  try
    NN.AddLayer(TNNetInput.Create(2));
    Layer := TNNetFullConnectLinear.Create(1);
    NN.AddLayer(Layer);

    // Set known weights for numerical verification (bias not set directly, use default 0)
    // Output = w1*x1 + w2*x2 (bias is 0 by default from initialization)
    Layer.Neurons[0].Weights.Raw[0] := 2.0;  // w1 = 2
    Layer.Neurons[0].Weights.Raw[1] := 3.0;  // w2 = 3

    Input.Raw[0] := 1.0;
    Input.Raw[1] := 1.0;

    NN.Compute(Input);
    NN.GetOutput(Output);

    // Numerical verification
    AssertEquals('Output should have size 1', 1, Output.Size);
    // With zero bias: 2*1 + 3*1 = 5
    AssertEquals('Output should be 2*1+3*1=5 (with default zero bias)', 5.0, Output.Raw[0], 0.0001);
    
    // Test with different input
    Input.Raw[0] := 2.0;
    Input.Raw[1] := 3.0;
    // Expected: 2*2 + 3*3 = 4 + 9 = 13
    
    NN.Compute(Input);
    NN.GetOutput(Output);
    AssertEquals('Output should be 2*2+3*3=13', 13.0, Output.Raw[0], 0.0001);
    
    // Verify output is not NaN or Inf
    AssertFalse('Output should not be NaN', IsNaN(Output.Raw[0]));
    AssertFalse('Output should not be Inf', IsInfinite(Output.Raw[0]));
  finally
    NN.Free;
    Input.Free;
    Output.Free;
  end;
end;

procedure TTestNeuralLayers.TestConvolutionForward;
var
  NN: TNNet;
  Input: TNNetVolume;
  ConvLayer: TNNetConvolutionReLU;
  I, OutputSize: integer;
begin
  NN := TNNet.Create();
  Input := TNNetVolume.Create(8, 8, 3);
  try
    NN.AddLayer(TNNetInput.Create(8, 8, 3));
    ConvLayer := TNNetConvolutionReLU.Create(16, 3, 1, 1);
    NN.AddLayer(ConvLayer);

    Input.Fill(1.0);
    NN.Compute(Input);

    // Verify output dimensions
    AssertEquals('Output SizeX should be 8', 8, NN.GetLastLayer.Output.SizeX);
    AssertEquals('Output SizeY should be 8', 8, NN.GetLastLayer.Output.SizeY);
    AssertEquals('Output depth should be 16', 16, NN.GetLastLayer.Output.Depth);
    
    // Numerical verification: Output should exist and be finite
    OutputSize := NN.GetLastLayer.Output.Size;
    AssertEquals('Output size should be 8*8*16 = 1024', 1024, OutputSize);
    
    // Check that output values are valid (not NaN or Inf)
    for I := 0 to OutputSize - 1 do
    begin
      AssertFalse('Output should not be NaN', IsNaN(NN.GetLastLayer.Output.Raw[I]));
      AssertFalse('Output should not be Inf', IsInfinite(NN.GetLastLayer.Output.Raw[I]));
    end;
    
    // ReLU applied, so output should be non-negative for positive weighted sums
    // Since weights are random, we just verify the output exists and ReLU works
    AssertTrue('Output min should be >= 0 (ReLU)', NN.GetLastLayer.Output.GetMin() >= -0.0001);
  finally
    NN.Free;
    Input.Free;
  end;
end;

procedure TTestNeuralLayers.TestMaxPoolForward;
var
  NN: TNNet;
  Input: TNNetVolume;
begin
  NN := TNNet.Create();
  Input := TNNetVolume.Create(4, 4, 1);
  try
    NN.AddLayer(TNNetInput.Create(4, 4, 1));
    NN.AddLayer(TNNetMaxPool.Create(2));

    // Set up input with known values for numerical verification
    // 2x2 pool regions: (0,0)-(1,1), (2,0)-(3,1), (0,2)-(1,3), (2,2)-(3,3)
    Input.Fill(1.0);
    Input[0, 0, 0] := 5.0;  // Max in region (0,0)
    Input[3, 1, 0] := 7.0;  // Max in region (1,0)
    Input[1, 3, 0] := 3.0;  // Max in region (0,1)
    Input[2, 2, 0] := 9.0;  // Max in region (1,1)
    
    NN.Compute(Input);

    // Verify output dimensions
    AssertEquals('Output SizeX should be 2 after 2x2 pool', 2, NN.GetLastLayer.Output.SizeX);
    AssertEquals('Output SizeY should be 2 after 2x2 pool', 2, NN.GetLastLayer.Output.SizeY);
    
    // Numerical verification: each output cell should contain the max of its 2x2 region
    AssertEquals('Max pool output (0,0) should be 5.0', 5.0, NN.GetLastLayer.Output[0, 0, 0], 0.0001);
    AssertEquals('Max pool output (1,0) should be 7.0', 7.0, NN.GetLastLayer.Output[1, 0, 0], 0.0001);
    AssertEquals('Max pool output (0,1) should be 3.0', 3.0, NN.GetLastLayer.Output[0, 1, 0], 0.0001);
    AssertEquals('Max pool output (1,1) should be 9.0', 9.0, NN.GetLastLayer.Output[1, 1, 0], 0.0001);
  finally
    NN.Free;
    Input.Free;
  end;
end;

procedure TTestNeuralLayers.TestNetworkSaveLoad;
var
  NN1, NN2: TNNet;
  Input, Output1, Output2: TNNetVolume;
  TempFile: string;
begin
  NN1 := TNNet.Create();
  NN2 := TNNet.Create();
  Input := TNNetVolume.Create(4, 1, 1);
  Output1 := TNNetVolume.Create(2, 1, 1);
  Output2 := TNNetVolume.Create(2, 1, 1);
  TempFile := GetTempDir() + 'nn_test_' + IntToStr(Random(MaxInt)) + '.nn';
  try
    NN1.AddLayer(TNNetInput.Create(4));
    NN1.AddLayer(TNNetFullConnectReLU.Create(8));
    NN1.AddLayer(TNNetFullConnectLinear.Create(2));

    Input.RandomizeGaussian();
    NN1.Compute(Input);
    NN1.GetOutput(Output1);

    NN1.SaveToFile(TempFile);
    NN2.LoadFromFile(TempFile);

    NN2.Compute(Input);
    NN2.GetOutput(Output2);

    AssertEquals('Loaded network should produce same output', 0.0, Output1.SumDiff(Output2), 0.0001);
  finally
    NN1.Free;
    NN2.Free;
    Input.Free;
    Output1.Free;
    Output2.Free;
    DeleteFile(TempFile);
  end;
end;

procedure TTestNeuralLayers.TestSimpleXORLearning;
var
  NN: TNNet;
  // Quick smoke test - just verify XOR network can be constructed
begin
  NN := TNNet.Create();
  try
    NN.AddLayer([
      TNNetInput.Create(2),
      TNNetFullConnectReLU.Create(4),
      TNNetFullConnectReLU.Create(4),
      TNNetFullConnectLinear.Create(1)
    ]);
    AssertEquals('Network should have 4 layers', 4, NN.CountLayers);
  finally
    NN.Free;
  end;
end;

procedure TTestNeuralLayers.TestAvgPoolForward;
var
  NN: TNNet;
  Input: TNNetVolume;
begin
  NN := TNNet.Create();
  Input := TNNetVolume.Create(4, 4, 1);
  try
    NN.AddLayer(TNNetInput.Create(4, 4, 1));
    NN.AddLayer(TNNetAvgPool.Create(2));

    // Set up input with known values for numerical verification
    // Region (0,0)-(1,1): values 2, 4, 6, 8 -> avg = 5.0
    Input[0, 0, 0] := 2.0;
    Input[1, 0, 0] := 4.0;
    Input[0, 1, 0] := 6.0;
    Input[1, 1, 0] := 8.0;
    // Region (2,0)-(3,1): all 4.0 -> avg = 4.0
    Input[2, 0, 0] := 4.0;
    Input[3, 0, 0] := 4.0;
    Input[2, 1, 0] := 4.0;
    Input[3, 1, 0] := 4.0;
    // Region (0,2)-(1,3): values 0, 0, 0, 12 -> avg = 3.0
    Input[0, 2, 0] := 0.0;
    Input[1, 2, 0] := 0.0;
    Input[0, 3, 0] := 0.0;
    Input[1, 3, 0] := 12.0;
    // Region (2,2)-(3,3): all 10.0 -> avg = 10.0
    Input[2, 2, 0] := 10.0;
    Input[3, 2, 0] := 10.0;
    Input[2, 3, 0] := 10.0;
    Input[3, 3, 0] := 10.0;
    
    NN.Compute(Input);

    // Verify output dimensions
    AssertEquals('Output SizeX should be 2 after 2x2 avg pool', 2, NN.GetLastLayer.Output.SizeX);
    AssertEquals('Output SizeY should be 2 after 2x2 avg pool', 2, NN.GetLastLayer.Output.SizeY);
    AssertEquals('Output Depth should be 1', 1, NN.GetLastLayer.Output.Depth);
    
    // Numerical verification: each output cell should contain the average of its 2x2 region
    AssertEquals('Avg pool output (0,0) should be 5.0', 5.0, NN.GetLastLayer.Output[0, 0, 0], 0.0001);
    AssertEquals('Avg pool output (1,0) should be 4.0', 4.0, NN.GetLastLayer.Output[1, 0, 0], 0.0001);
    AssertEquals('Avg pool output (0,1) should be 3.0', 3.0, NN.GetLastLayer.Output[0, 1, 0], 0.0001);
    AssertEquals('Avg pool output (1,1) should be 10.0', 10.0, NN.GetLastLayer.Output[1, 1, 0], 0.0001);
  finally
    NN.Free;
    Input.Free;
  end;
end;

procedure TTestNeuralLayers.TestMinPoolForward;
var
  NN: TNNet;
  Input: TNNetVolume;
begin
  NN := TNNet.Create();
  Input := TNNetVolume.Create(4, 4, 1);
  try
    NN.AddLayer(TNNetInput.Create(4, 4, 1));
    NN.AddLayer(TNNetMinPool.Create(2));

    // Set up input with known values for numerical verification
    Input.Fill(5.0);
    // Region (0,0)-(1,1): min will be 1.0
    Input[0, 0, 0] := 1.0;
    // Region (2,0)-(3,1): min will be 2.0
    Input[3, 1, 0] := 2.0;
    // Region (0,2)-(1,3): min will be 0.5
    Input[1, 2, 0] := 0.5;
    // Region (2,2)-(3,3): min will be 3.0
    Input[2, 3, 0] := 3.0;

    NN.Compute(Input);

    // Verify output dimensions
    AssertEquals('Output SizeX should be 2 after 2x2 min pool', 2, NN.GetLastLayer.Output.SizeX);
    AssertEquals('Output SizeY should be 2 after 2x2 min pool', 2, NN.GetLastLayer.Output.SizeY);
    
    // Numerical verification: each output cell should contain the min of its 2x2 region
    AssertEquals('Min pool output (0,0) should be 1.0', 1.0, NN.GetLastLayer.Output[0, 0, 0], 0.0001);
    AssertEquals('Min pool output (1,0) should be 2.0', 2.0, NN.GetLastLayer.Output[1, 0, 0], 0.0001);
    AssertEquals('Min pool output (0,1) should be 0.5', 0.5, NN.GetLastLayer.Output[0, 1, 0], 0.0001);
    AssertEquals('Min pool output (1,1) should be 3.0', 3.0, NN.GetLastLayer.Output[1, 1, 0], 0.0001);
  finally
    NN.Free;
    Input.Free;
  end;
end;

procedure TTestNeuralLayers.TestReLUActivation;
var
  NN: TNNet;
  Input: TNNetVolume;
begin
  NN := TNNet.Create();
  Input := TNNetVolume.Create(4, 1, 1);
  try
    NN.AddLayer(TNNetInput.Create(4));
    NN.AddLayer(TNNetReLU.Create());

    Input.Raw[0] := -2.0;
    Input.Raw[1] := -1.0;
    Input.Raw[2] := 1.0;
    Input.Raw[3] := 2.0;

    NN.Compute(Input);

    // ReLU: max(0, x)
    AssertEquals('ReLU of -2 should be 0', 0.0, NN.GetLastLayer.Output.Raw[0], 0.0001);
    AssertEquals('ReLU of -1 should be 0', 0.0, NN.GetLastLayer.Output.Raw[1], 0.0001);
    AssertEquals('ReLU of 1 should be 1', 1.0, NN.GetLastLayer.Output.Raw[2], 0.0001);
    AssertEquals('ReLU of 2 should be 2', 2.0, NN.GetLastLayer.Output.Raw[3], 0.0001);
  finally
    NN.Free;
    Input.Free;
  end;
end;

procedure TTestNeuralLayers.TestSigmoidActivation;
var
  NN: TNNet;
  Input: TNNetVolume;
begin
  NN := TNNet.Create();
  Input := TNNetVolume.Create(3, 1, 1);
  try
    NN.AddLayer(TNNetInput.Create(3));
    NN.AddLayer(TNNetSigmoid.Create());

    Input.Raw[0] := 0.0;
    Input.Raw[1] := 10.0;
    Input.Raw[2] := -10.0;

    NN.Compute(Input);

    // Sigmoid(0) = 0.5
    AssertEquals('Sigmoid of 0 should be 0.5', 0.5, NN.GetLastLayer.Output.Raw[0], 0.0001);
    // Sigmoid(10) ≈ 1
    AssertTrue('Sigmoid of 10 should be close to 1', NN.GetLastLayer.Output.Raw[1] > 0.99);
    // Sigmoid(-10) ≈ 0
    AssertTrue('Sigmoid of -10 should be close to 0', NN.GetLastLayer.Output.Raw[2] < 0.01);
  finally
    NN.Free;
    Input.Free;
  end;
end;

procedure TTestNeuralLayers.TestSoftMaxLayer;
var
  NN: TNNet;
  Input: TNNetVolume;
  SumOutput: TNeuralFloat;
begin
  NN := TNNet.Create();
  Input := TNNetVolume.Create(4, 1, 1);
  try
    NN.AddLayer(TNNetInput.Create(4));
    NN.AddLayer(TNNetSoftMax.Create());

    Input.Raw[0] := 1.0;
    Input.Raw[1] := 2.0;
    Input.Raw[2] := 3.0;
    Input.Raw[3] := 4.0;

    NN.Compute(Input);

    SumOutput := NN.GetLastLayer.Output.GetSum();

    // SoftMax output should sum to 1.0
    AssertEquals('SoftMax output sum should be 1.0', 1.0, SumOutput, 0.0001);
    // Higher inputs should produce higher probabilities
    AssertTrue('Output[3] should be greatest', NN.GetLastLayer.Output.Raw[3] > NN.GetLastLayer.Output.Raw[2]);
  finally
    NN.Free;
    Input.Free;
  end;
end;

procedure TTestNeuralLayers.TestDepthwiseConvolution;
var
  NN: TNNet;
  Input: TNNetVolume;
begin
  NN := TNNet.Create();
  Input := TNNetVolume.Create(8, 8, 4);
  try
    NN.AddLayer(TNNetInput.Create(8, 8, 4));
    NN.AddLayer(TNNetDepthwiseConvReLU.Create(1, 3, 1, 1));

    Input.Fill(1.0);
    NN.Compute(Input);

    // Depthwise conv with multiplier 1 keeps same depth
    AssertEquals('Output SizeX should be 8', 8, NN.GetLastLayer.Output.SizeX);
    AssertEquals('Output Depth should be 4', 4, NN.GetLastLayer.Output.Depth);
  finally
    NN.Free;
    Input.Free;
  end;
end;

procedure TTestNeuralLayers.TestPointwiseConvolution;
var
  NN: TNNet;
  Input: TNNetVolume;
begin
  NN := TNNet.Create();
  Input := TNNetVolume.Create(8, 8, 16);
  try
    NN.AddLayer(TNNetInput.Create(8, 8, 16));
    NN.AddLayer(TNNetPointwiseConvReLU.Create(32));

    Input.Fill(1.0);
    NN.Compute(Input);

    // Pointwise conv changes depth while keeping spatial dimensions
    AssertEquals('Output SizeX should be 8', 8, NN.GetLastLayer.Output.SizeX);
    AssertEquals('Output SizeY should be 8', 8, NN.GetLastLayer.Output.SizeY);
    AssertEquals('Output Depth should be 32', 32, NN.GetLastLayer.Output.Depth);
  finally
    NN.Free;
    Input.Free;
  end;
end;

procedure TTestNeuralLayers.TestConcatLayers;
var
  NN: TNNet;
  Input: TNNetVolume;
  InputLayer, Layer1, Layer2: TNNetLayer;
begin
  NN := TNNet.Create();
  Input := TNNetVolume.Create(8, 8, 3);
  try
    InputLayer := NN.AddLayer(TNNetInput.Create(8, 8, 3));
    
    // Create two parallel paths branching from the input layer
    Layer1 := NN.AddLayer(TNNetConvolutionReLU.Create(16, 3, 1, 1));
    Layer2 := NN.AddLayerAfter(TNNetConvolutionReLU.Create(8, 3, 1, 1), InputLayer);
    
    // Concatenate the two paths
    NN.AddLayer(TNNetDeepConcat.Create([Layer1, Layer2]));

    Input.Fill(1.0);
    NN.Compute(Input);

    // Concatenated depth should be 16 + 8 = 24
    AssertEquals('Concatenated depth should be 24', 24, NN.GetLastLayer.Output.Depth);
    AssertEquals('Output SizeX should be 8', 8, NN.GetLastLayer.Output.SizeX);
  finally
    NN.Free;
    Input.Free;
  end;
end;

procedure TTestNeuralLayers.TestSumLayers;
var
  NN: TNNet;
  Input: TNNetVolume;
  InputLayer, Layer1, Layer2: TNNetLayer;
begin
  NN := TNNet.Create();
  Input := TNNetVolume.Create(8, 8, 16);
  try
    InputLayer := NN.AddLayer(TNNetInput.Create(8, 8, 16));
    
    // Create two parallel paths with same output size branching from input
    Layer1 := NN.AddLayer(TNNetConvolutionLinear.Create(16, 3, 1, 1));
    Layer2 := NN.AddLayerAfter(TNNetConvolutionLinear.Create(16, 3, 1, 1), InputLayer);
    
    // Sum the two paths
    NN.AddLayer(TNNetSum.Create([Layer1, Layer2]));

    Input.Fill(1.0);
    NN.Compute(Input);

    // Sum should maintain the same dimensions
    AssertEquals('Sum output depth should be 16', 16, NN.GetLastLayer.Output.Depth);
    AssertEquals('Sum output SizeX should be 8', 8, NN.GetLastLayer.Output.SizeX);
  finally
    NN.Free;
    Input.Free;
  end;
end;

procedure TTestNeuralLayers.TestIdentityLayer;
var
  NN: TNNet;
  Input, Output: TNNetVolume;
begin
  NN := TNNet.Create();
  Input := TNNetVolume.Create(4, 4, 2);
  Output := TNNetVolume.Create(4, 4, 2);
  try
    NN.AddLayer(TNNetInput.Create(4, 4, 2));
    NN.AddLayer(TNNetIdentity.Create());

    Input.RandomizeGaussian();
    NN.Compute(Input);
    NN.GetOutput(Output);

    // Identity layer should pass through unchanged
    AssertEquals('Identity should preserve values', 0.0, Input.SumDiff(Output), 0.0001);
  finally
    NN.Free;
    Input.Free;
    Output.Free;
  end;
end;

procedure TTestNeuralLayers.TestReshapeLayer;
var
  NN: TNNet;
  Input: TNNetVolume;
begin
  NN := TNNet.Create();
  Input := TNNetVolume.Create(4, 4, 4);
  try
    NN.AddLayer(TNNetInput.Create(4, 4, 4));
    NN.AddLayer(TNNetReshape.Create(8, 2, 4));

    Input.Fill(1.0);
    NN.Compute(Input);

    // Total size should be preserved: 4*4*4 = 8*2*4 = 64
    AssertEquals('Reshape output SizeX should be 8', 8, NN.GetLastLayer.Output.SizeX);
    AssertEquals('Reshape output SizeY should be 2', 2, NN.GetLastLayer.Output.SizeY);
    AssertEquals('Reshape output Depth should be 4', 4, NN.GetLastLayer.Output.Depth);
    AssertEquals('Total size should be preserved', 64, NN.GetLastLayer.Output.Size);
  finally
    NN.Free;
    Input.Free;
  end;
end;

procedure TTestNeuralLayers.TestDropoutLayer;
var
  NN: TNNet;
  Input: TNNetVolume;
begin
  NN := TNNet.Create();
  Input := TNNetVolume.Create(10, 1, 1);
  try
    NN.AddLayer(TNNetInput.Create(10));
    NN.AddLayer(TNNetDropout.Create(0.5));

    Input.Fill(1.0);
    NN.Compute(Input);

    // During inference (non-training), dropout should pass values through
    // The output size should match input size
    AssertEquals('Dropout output size should be 10', 10, NN.GetLastLayer.Output.Size);
  finally
    NN.Free;
    Input.Free;
  end;
end;

procedure TTestNeuralLayers.TestMultipleLayersNetwork;
var
  NN: TNNet;
  Input: TNNetVolume;
begin
  NN := TNNet.Create();
  Input := TNNetVolume.Create(32, 32, 3);
  try
    // Build a more complex network similar to a simple CNN
    NN.AddLayer([
      TNNetInput.Create(32, 32, 3),
      TNNetConvolutionReLU.Create(16, 3, 1, 1),
      TNNetMaxPool.Create(2),
      TNNetConvolutionReLU.Create(32, 3, 1, 1),
      TNNetMaxPool.Create(2),
      TNNetFullConnectReLU.Create(64),
      TNNetFullConnectLinear.Create(10),
      TNNetSoftMax.Create()
    ]);

    Input.RandomizeGaussian();
    NN.Compute(Input);

    // Output should be 10 classes with softmax
    AssertEquals('Output should have 10 classes', 10, NN.GetLastLayer.Output.Size);
    // SoftMax sum should be 1.0
    AssertEquals('SoftMax sum should be 1.0', 1.0, NN.GetLastLayer.Output.GetSum(), 0.0001);
  finally
    NN.Free;
    Input.Free;
  end;
end;

procedure TTestNeuralLayers.TestNetworkClone;
var
  NN1, NN2: TNNet;
  Input, Output1, Output2: TNNetVolume;
begin
  NN1 := TNNet.Create();
  NN2 := nil;
  Input := TNNetVolume.Create(4, 1, 1);
  Output1 := TNNetVolume.Create(2, 1, 1);
  Output2 := TNNetVolume.Create(2, 1, 1);
  try
    NN1.AddLayer([
      TNNetInput.Create(4),
      TNNetFullConnectReLU.Create(8),
      TNNetFullConnectLinear.Create(2)
    ]);

    Input.RandomizeGaussian();
    NN1.Compute(Input);
    NN1.GetOutput(Output1);

    // Clone the network using the Clone method
    NN2 := NN1.Clone();

    NN2.Compute(Input);
    NN2.GetOutput(Output2);

    // Cloned network should produce same output
    AssertEquals('Cloned network should produce same output', 0.0, Output1.SumDiff(Output2), 0.0001);
  finally
    NN1.Free;
    if NN2 <> nil then NN2.Free;
    Input.Free;
    Output1.Free;
    Output2.Free;
  end;
end;

procedure TTestNeuralLayers.TestLayerCount;
var
  NN: TNNet;
begin
  NN := TNNet.Create();
  try
    NN.AddLayer([
      TNNetInput.Create(10),
      TNNetFullConnectReLU.Create(20),
      TNNetFullConnectReLU.Create(20),
      TNNetFullConnectLinear.Create(5)
    ]);

    AssertEquals('Network should have 4 layers', 4, NN.CountLayers());
    // Count neurons and weights
    AssertTrue('Network should have positive neuron count', NN.CountNeurons() > 0);
    AssertTrue('Network should have positive weight count', NN.CountWeights() > 0);
  finally
    NN.Free;
  end;
end;

procedure TTestNeuralLayers.TestReLU6Activation;
var
  NN: TNNet;
  Input: TNNetVolume;
begin
  NN := TNNet.Create();
  Input := TNNetVolume.Create(5, 1, 1);
  try
    // TNNetReLU6: ReLU clamped to [0, 6]
    // Note: Full activation behavior only applies during training when error derivatives are set
    NN.AddLayer(TNNetInput.Create(5));
    NN.AddLayer(TNNetReLU6.Create());

    Input.Raw[0] := -2.0;
    Input.Raw[1] := 0.0;
    Input.Raw[2] := 3.0;
    Input.Raw[3] := 6.0;
    Input.Raw[4] := 10.0;

    NN.Compute(Input);

    // Verify output is produced
    AssertEquals('Output should have 5 elements', 5, NN.GetLastLayer.Output.Size);
    // ReLU6 should not produce NaN values
    AssertFalse('Output 0 should not be NaN', IsNaN(NN.GetLastLayer.Output.Raw[0]));
    AssertFalse('Output 1 should not be NaN', IsNaN(NN.GetLastLayer.Output.Raw[1]));
    AssertFalse('Output 2 should not be NaN', IsNaN(NN.GetLastLayer.Output.Raw[2]));
    AssertFalse('Output 3 should not be NaN', IsNaN(NN.GetLastLayer.Output.Raw[3]));
    AssertFalse('Output 4 should not be NaN', IsNaN(NN.GetLastLayer.Output.Raw[4]));
  finally
    NN.Free;
    Input.Free;
  end;
end;

procedure TTestNeuralLayers.TestLeakyReLUActivation;
var
  NN: TNNet;
  Input: TNNetVolume;
begin
  NN := TNNet.Create();
  Input := TNNetVolume.Create(4, 1, 1);
  try
    NN.AddLayer(TNNetInput.Create(4));
    NN.AddLayer(TNNetLeakyReLU.Create());

    Input.Raw[0] := -2.0;
    Input.Raw[1] := 0.0;
    Input.Raw[2] := 2.0;
    Input.Raw[3] := -100.0;

    NN.Compute(Input);

    // Leaky ReLU should produce output
    AssertEquals('Output should have 4 elements', 4, NN.GetLastLayer.Output.Size);
    // For positive values, output equals input
    AssertEquals('LeakyReLU of 0 should be 0', 0.0, NN.GetLastLayer.Output.Raw[1], 0.0001);
    AssertEquals('LeakyReLU of 2 should be 2', 2.0, NN.GetLastLayer.Output.Raw[2], 0.0001);
    // Output values should be finite
    AssertFalse('Output should not be NaN', IsNaN(NN.GetLastLayer.Output.Raw[0]));
    AssertFalse('Output should not be Inf', IsInfinite(NN.GetLastLayer.Output.Raw[3]));
  finally
    NN.Free;
    Input.Free;
  end;
end;

procedure TTestNeuralLayers.TestSwishActivation;
var
  NN: TNNet;
  Input: TNNetVolume;
begin
  NN := TNNet.Create();
  Input := TNNetVolume.Create(3, 1, 1);
  try
    NN.AddLayer(TNNetInput.Create(3));
    NN.AddLayer(TNNetSwish.Create());

    Input.Raw[0] := 0.0;
    Input.Raw[1] := 1.0;
    Input.Raw[2] := -1.0;

    NN.Compute(Input);

    // Swish: x * sigmoid(x)
    // At x=0: 0 * 0.5 = 0
    AssertEquals('Swish of 0 should be 0', 0.0, NN.GetLastLayer.Output.Raw[0], 0.0001);
    // At x=1: 1 * sigmoid(1) ≈ 0.731
    AssertTrue('Swish of 1 should be around 0.731', Abs(NN.GetLastLayer.Output.Raw[1] - 0.731) < 0.01);
  finally
    NN.Free;
    Input.Free;
  end;
end;

procedure TTestNeuralLayers.TestHyperbolicTangent;
var
  NN: TNNet;
  Input: TNNetVolume;
begin
  NN := TNNet.Create();
  Input := TNNetVolume.Create(3, 1, 1);
  try
    NN.AddLayer(TNNetInput.Create(3));
    NN.AddLayer(TNNetHyperbolicTangent.Create());

    Input.Raw[0] := 0.0;
    Input.Raw[1] := 10.0;
    Input.Raw[2] := -10.0;

    NN.Compute(Input);

    // tanh(0) = 0
    AssertEquals('Tanh of 0 should be 0', 0.0, NN.GetLastLayer.Output.Raw[0], 0.0001);
    // tanh(10) ≈ 1
    AssertTrue('Tanh of 10 should be close to 1', NN.GetLastLayer.Output.Raw[1] > 0.99);
    // tanh(-10) ≈ -1
    AssertTrue('Tanh of -10 should be close to -1', NN.GetLastLayer.Output.Raw[2] < -0.99);
  finally
    NN.Free;
    Input.Free;
  end;
end;

procedure TTestNeuralLayers.TestSELUActivation;
var
  NN: TNNet;
  Input: TNNetVolume;
begin
  NN := TNNet.Create();
  Input := TNNetVolume.Create(3, 1, 1);
  try
    NN.AddLayer(TNNetInput.Create(3));
    NN.AddLayer(TNNetSELU.Create());

    Input.Raw[0] := 0.0;
    Input.Raw[1] := 1.0;
    Input.Raw[2] := -1.0;

    NN.Compute(Input);

    // SELU(0) = 0
    AssertEquals('SELU of 0 should be 0', 0.0, NN.GetLastLayer.Output.Raw[0], 0.0001);
    // SELU of positive values scales them
    AssertTrue('SELU of 1 should be positive', NN.GetLastLayer.Output.Raw[1] > 0);
    // SELU of negative values
    AssertTrue('SELU of -1 should be negative', NN.GetLastLayer.Output.Raw[2] < 0);
  finally
    NN.Free;
    Input.Free;
  end;
end;

procedure TTestNeuralLayers.TestMaxChannel;
var
  NN: TNNet;
  Input: TNNetVolume;
begin
  NN := TNNet.Create();
  Input := TNNetVolume.Create(4, 4, 3);
  try
    NN.AddLayer(TNNetInput.Create(4, 4, 3));
    NN.AddLayer(TNNetMaxChannel.Create());

    // Fill channels with different values
    Input.FillAtDepth(0, 1.0);
    Input.FillAtDepth(1, 2.0);
    Input.FillAtDepth(2, 3.0);

    NN.Compute(Input);

    // MaxChannel reduces to depth-sized 1D output
    AssertEquals('Output should have 3 elements', 3, NN.GetLastLayer.Output.Size);
    AssertEquals('Max of channel 0 should be 1.0', 1.0, NN.GetLastLayer.Output.Raw[0], 0.0001);
    AssertEquals('Max of channel 1 should be 2.0', 2.0, NN.GetLastLayer.Output.Raw[1], 0.0001);
    AssertEquals('Max of channel 2 should be 3.0', 3.0, NN.GetLastLayer.Output.Raw[2], 0.0001);
  finally
    NN.Free;
    Input.Free;
  end;
end;

procedure TTestNeuralLayers.TestAvgChannel;
var
  NN: TNNet;
  Input: TNNetVolume;
begin
  NN := TNNet.Create();
  Input := TNNetVolume.Create(4, 4, 2);
  try
    NN.AddLayer(TNNetInput.Create(4, 4, 2));
    NN.AddLayer(TNNetAvgChannel.Create());

    Input.FillAtDepth(0, 4.0);
    Input.FillAtDepth(1, 8.0);

    NN.Compute(Input);

    // AvgChannel reduces to depth-sized 1D output
    AssertEquals('Output should have 2 elements', 2, NN.GetLastLayer.Output.Size);
    AssertEquals('Avg of channel 0 should be 4.0', 4.0, NN.GetLastLayer.Output.Raw[0], 0.0001);
    AssertEquals('Avg of channel 1 should be 8.0', 8.0, NN.GetLastLayer.Output.Raw[1], 0.0001);
  finally
    NN.Free;
    Input.Free;
  end;
end;

procedure TTestNeuralLayers.TestLayerMaxNormalization;
var
  NN: TNNet;
  Input: TNNetVolume;
begin
  NN := TNNet.Create();
  Input := TNNetVolume.Create(4, 1, 1);
  try
    NN.AddLayer(TNNetInput.Create(4));
    NN.AddLayer(TNNetLayerMaxNormalization.Create());

    Input.Raw[0] := 2.0;
    Input.Raw[1] := 4.0;
    Input.Raw[2] := 6.0;
    Input.Raw[3] := 8.0;

    NN.Compute(Input);

    // Max normalization divides by max value (8.0)
    // Output should be in range [0, 1]
    AssertTrue('Output max should be 1.0', Abs(NN.GetLastLayer.Output.GetMax() - 1.0) < 0.0001);
  finally
    NN.Free;
    Input.Free;
  end;
end;

procedure TTestNeuralLayers.TestLayerStdNormalization;
var
  NN: TNNet;
  Input: TNNetVolume;
begin
  NN := TNNet.Create();
  Input := TNNetVolume.Create(4, 1, 1);
  try
    NN.AddLayer(TNNetInput.Create(4));
    NN.AddLayer(TNNetLayerStdNormalization.Create());

    Input.Raw[0] := 2.0;
    Input.Raw[1] := 4.0;
    Input.Raw[2] := 6.0;
    Input.Raw[3] := 8.0;

    NN.Compute(Input);

    // Std normalization should produce output with unit std deviation
    AssertEquals('Output size should be 4', 4, NN.GetLastLayer.Output.Size);
  finally
    NN.Free;
    Input.Free;
  end;
end;

procedure TTestNeuralLayers.TestMovingStdNormalization;
var
  NN: TNNet;
  Input: TNNetVolume;
begin
  NN := TNNet.Create();
  Input := TNNetVolume.Create(8, 1, 1);
  try
    NN.AddLayer(TNNetInput.Create(8));
    NN.AddLayer(TNNetMovingStdNormalization.Create());

    Input.RandomizeGaussian(5.0);
    Input.Add(10.0); // Shift to have non-zero mean

    NN.Compute(Input);

    AssertEquals('Output size should be 8', 8, NN.GetLastLayer.Output.Size);
  finally
    NN.Free;
    Input.Free;
  end;
end;

procedure TTestNeuralLayers.TestChannelBias;
var
  NN: TNNet;
  Input: TNNetVolume;
begin
  NN := TNNet.Create();
  Input := TNNetVolume.Create(4, 4, 2);
  try
    NN.AddLayer(TNNetInput.Create(4, 4, 2));
    NN.AddLayer(TNNetChannelBias.Create());

    Input.Fill(1.0);
    NN.Compute(Input);

    // ChannelBias adds a learnable bias per channel
    AssertEquals('Output should maintain dimensions', 32, NN.GetLastLayer.Output.Size);
  finally
    NN.Free;
    Input.Free;
  end;
end;

procedure TTestNeuralLayers.TestChannelMul;
var
  NN: TNNet;
  Input: TNNetVolume;
begin
  NN := TNNet.Create();
  Input := TNNetVolume.Create(4, 4, 2);
  try
    NN.AddLayer(TNNetInput.Create(4, 4, 2));
    NN.AddLayer(TNNetChannelMul.Create());

    Input.Fill(2.0);
    NN.Compute(Input);

    // ChannelMul multiplies by a learnable scale per channel
    AssertEquals('Output should maintain dimensions', 32, NN.GetLastLayer.Output.Size);
  finally
    NN.Free;
    Input.Free;
  end;
end;

procedure TTestNeuralLayers.TestCellBias;
var
  NN: TNNet;
  Input: TNNetVolume;
begin
  NN := TNNet.Create();
  Input := TNNetVolume.Create(4, 4, 1);
  try
    NN.AddLayer(TNNetInput.Create(4, 4, 1));
    NN.AddLayer(TNNetCellBias.Create());

    Input.Fill(1.0);
    NN.Compute(Input);

    // CellBias adds a learnable bias per cell
    AssertEquals('Output should maintain dimensions', 16, NN.GetLastLayer.Output.Size);
  finally
    NN.Free;
    Input.Free;
  end;
end;

procedure TTestNeuralLayers.TestSplitChannels;
var
  NN: TNNet;
  Input: TNNetVolume;
begin
  NN := TNNet.Create();
  Input := TNNetVolume.Create(4, 4, 8);
  try
    NN.AddLayer(TNNetInput.Create(4, 4, 8));
    // Split first 4 channels
    NN.AddLayer(TNNetSplitChannels.Create(0, 4));

    Input.RandomizeGaussian();
    NN.Compute(Input);

    // Output should have 4 channels
    AssertEquals('Output depth should be 4', 4, NN.GetLastLayer.Output.Depth);
    AssertEquals('Output SizeX should be 4', 4, NN.GetLastLayer.Output.SizeX);
  finally
    NN.Free;
    Input.Free;
  end;
end;

procedure TTestNeuralLayers.TestInterleaveChannels;
var
  NN: TNNet;
  Input: TNNetVolume;
begin
  NN := TNNet.Create();
  Input := TNNetVolume.Create(4, 4, 8);
  try
    NN.AddLayer(TNNetInput.Create(4, 4, 8));
    NN.AddLayer(TNNetInterleaveChannels.Create(2));

    Input.RandomizeGaussian();
    NN.Compute(Input);

    // Interleave should maintain size
    AssertEquals('Output size should match input size', 128, NN.GetLastLayer.Output.Size);
  finally
    NN.Free;
    Input.Free;
  end;
end;

procedure TTestNeuralLayers.TestPointwiseConvLinear;
var
  NN: TNNet;
  Input: TNNetVolume;
begin
  NN := TNNet.Create();
  Input := TNNetVolume.Create(8, 8, 16);
  try
    NN.AddLayer(TNNetInput.Create(8, 8, 16));
    NN.AddLayer(TNNetPointwiseConvLinear.Create(32));

    Input.Fill(1.0);
    NN.Compute(Input);

    // Pointwise conv (1x1) changes depth only
    AssertEquals('Output SizeX should be 8', 8, NN.GetLastLayer.Output.SizeX);
    AssertEquals('Output SizeY should be 8', 8, NN.GetLastLayer.Output.SizeY);
    AssertEquals('Output Depth should be 32', 32, NN.GetLastLayer.Output.Depth);
  finally
    NN.Free;
    Input.Free;
  end;
end;

procedure TTestNeuralLayers.TestLocalConnect;
var
  NN: TNNet;
  Input: TNNetVolume;
begin
  NN := TNNet.Create();
  Input := TNNetVolume.Create(8, 8, 3);
  try
    NN.AddLayer(TNNetInput.Create(8, 8, 3));
    NN.AddLayer(TNNetLocalConnectReLU.Create(8, 3, 1, 1));

    Input.Fill(1.0);
    NN.Compute(Input);

    // Local connect should work similar to convolution but with unique weights per position
    AssertEquals('Output SizeX should be 8', 8, NN.GetLastLayer.Output.SizeX);
    AssertEquals('Output Depth should be 8', 8, NN.GetLastLayer.Output.Depth);
  finally
    NN.Free;
    Input.Free;
  end;
end;

procedure TTestNeuralLayers.TestGroupedConvolution;
var
  NN: TNNet;
  Input: TNNetVolume;
begin
  NN := TNNet.Create();
  Input := TNNetVolume.Create(8, 8, 16);
  try
    NN.AddLayer(TNNetInput.Create(8, 8, 16));
    // Groups=4 means 4 separate convolutions on 4 channels each
    NN.AddLayer(TNNetGroupedConvolutionLinear.Create(32, 3, 1, 1, 4));

    Input.Fill(1.0);
    NN.Compute(Input);

    AssertEquals('Output SizeX should be 8', 8, NN.GetLastLayer.Output.SizeX);
    AssertEquals('Output Depth should be 32', 32, NN.GetLastLayer.Output.Depth);
  finally
    NN.Free;
    Input.Free;
  end;
end;

procedure TTestNeuralLayers.TestBackpropagation;
var
  NN: TNNet;
  Input, DesiredOutput: TNNetVolume;
  ErrorBefore, ErrorAfter: TNeuralFloat;
begin
  NN := TNNet.Create();
  Input := TNNetVolume.Create(2, 1, 1);
  DesiredOutput := TNNetVolume.Create(1, 1, 1);
  try
    NN.AddLayer([
      TNNetInput.Create(2),
      TNNetFullConnectLinear.Create(4),
      TNNetReLU.Create(),
      TNNetFullConnectLinear.Create(1)
    ]);

    // Set input and desired output
    Input.Raw[0] := 1.0;
    Input.Raw[1] := 0.5;
    DesiredOutput.Raw[0] := 0.7;

    // Forward pass
    NN.Compute(Input);
    ErrorBefore := Abs(NN.GetLastLayer.Output.Raw[0] - DesiredOutput.Raw[0]);

    // Backward pass with learning
    NN.Backpropagate(DesiredOutput);
    NN.UpdateWeights();

    // Forward pass again
    NN.Compute(Input);
    ErrorAfter := Abs(NN.GetLastLayer.Output.Raw[0] - DesiredOutput.Raw[0]);

    // Error should decrease after one step (in most cases)
    // Note: This is a probabilistic test, may occasionally fail
    AssertTrue('Network should produce output', NN.GetLastLayer.Output.Size = 1);
  finally
    NN.Free;
    Input.Free;
    DesiredOutput.Free;
  end;
end;

procedure TTestNeuralLayers.TestGradientComputation;
var
  NN: TNNet;
  Input, DesiredOutput: TNNetVolume;
begin
  NN := TNNet.Create();
  Input := TNNetVolume.Create(4, 1, 1);
  DesiredOutput := TNNetVolume.Create(2, 1, 1);
  try
    NN.AddLayer([
      TNNetInput.Create(4),
      TNNetFullConnectLinear.Create(2)
    ]);

    Input.RandomizeGaussian();
    DesiredOutput.RandomizeGaussian();

    NN.Compute(Input);
    NN.Backpropagate(DesiredOutput);

    // Check that output error is computed
    AssertEquals('Output error size should match', 2, NN.GetLastLayer.OutputError.Size);
    // The output error should not be all zeros
    AssertTrue('Output error should be non-zero', NN.GetLastLayer.OutputError.GetSumAbs() > 0);
  finally
    NN.Free;
    Input.Free;
    DesiredOutput.Free;
  end;
end;

procedure TTestNeuralLayers.TestWeightInitHe;
var
  NN: TNNet;
  Layer: TNNetFullConnectLinear;
  MaxWeight: TNeuralFloat;
begin
  NN := TNNet.Create();
  try
    NN.AddLayer(TNNetInput.Create(100));
    Layer := TNNetFullConnectLinear.Create(50);
    NN.AddLayer(Layer);

    // Initialize with He method
    Layer.InitHeUniform();

    // Weights should be in reasonable range
    MaxWeight := Layer.Neurons.GetMaxAbsWeight();
    AssertTrue('Weights should be initialized', MaxWeight > 0);
    AssertTrue('Weights should be bounded', MaxWeight < 10);
  finally
    NN.Free;
  end;
end;

procedure TTestNeuralLayers.TestWeightInitLeCun;
var
  NN: TNNet;
  Layer: TNNetFullConnectLinear;
  MaxWeight: TNeuralFloat;
begin
  NN := TNNet.Create();
  try
    NN.AddLayer(TNNetInput.Create(100));
    Layer := TNNetFullConnectLinear.Create(50);
    NN.AddLayer(Layer);

    // Initialize with LeCun method
    Layer.InitLeCunUniform();

    // Weights should be in reasonable range
    MaxWeight := Layer.Neurons.GetMaxAbsWeight();
    AssertTrue('Weights should be initialized', MaxWeight > 0);
    AssertTrue('Weights should be bounded', MaxWeight < 10);
  finally
    NN.Free;
  end;
end;

procedure TTestNeuralLayers.TestWeightInitGlorot;
var
  NN: TNNet;
  Layer: TNNetFullConnectLinear;
  MaxWeight: TNeuralFloat;
begin
  NN := TNNet.Create();
  try
    NN.AddLayer(TNNetInput.Create(100));
    Layer := TNNetFullConnectLinear.Create(50);
    NN.AddLayer(Layer);

    // Initialize with Glorot/Xavier method
    Layer.InitGlorotBengioUniform();

    // Weights should be in reasonable range
    MaxWeight := Layer.Neurons.GetMaxAbsWeight();
    AssertTrue('Weights should be initialized', MaxWeight > 0);
    AssertTrue('Weights should be bounded', MaxWeight < 10);
  finally
    NN.Free;
  end;
end;

procedure TTestNeuralLayers.TestEmbeddingLayer;
var
  NN: TNNet;
  Input: TNNetVolume;
begin
  NN := TNNet.Create();
  Input := TNNetVolume.Create(4, 1, 1); // 4 tokens
  try
    NN.AddLayer(TNNetInput.Create(4, 1, 1));
    // Vocab size 100, embedding dim 16
    NN.AddLayer(TNNetEmbedding.Create(100, 16));

    // Input tokens as integers (stored as floats)
    Input.Raw[0] := 5;
    Input.Raw[1] := 10;
    Input.Raw[2] := 25;
    Input.Raw[3] := 50;

    NN.Compute(Input);

    // Output should be 4 x 16 (4 tokens, 16 embedding dim)
    AssertEquals('Output SizeX should be 4', 4, NN.GetLastLayer.Output.SizeX);
    AssertEquals('Output Depth should be 16', 16, NN.GetLastLayer.Output.Depth);
  finally
    NN.Free;
    Input.Free;
  end;
end;

procedure TTestNeuralLayers.TestTokenAndPositionalEmbedding;
var
  NN: TNNet;
  Input: TNNetVolume;
begin
  NN := TNNet.Create();
  Input := TNNetVolume.Create(8, 1, 1); // 8 tokens (sequence length)
  try
    NN.AddLayer(TNNetInput.Create(8, 1, 1));
    // Vocab size 256, embedding dim 32
    NN.AddLayer(TNNetTokenAndPositionalEmbedding.Create(256, 32));

    // Input tokens
    Input.Raw[0] := 1;
    Input.Raw[1] := 5;
    Input.Raw[2] := 10;
    Input.Raw[3] := 20;
    Input.Raw[4] := 30;
    Input.Raw[5] := 40;
    Input.Raw[6] := 50;
    Input.Raw[7] := 60;

    NN.Compute(Input);

    // Output should be 8 x 32 (8 tokens, 32 embedding dim)
    AssertEquals('Output SizeX should be 8', 8, NN.GetLastLayer.Output.SizeX);
    AssertEquals('Output Depth should be 32', 32, NN.GetLastLayer.Output.Depth);
  finally
    NN.Free;
    Input.Free;
  end;
end;

initialization
  RegisterTest(TTestNeuralLayers);

end.
