unit TestNeuralLayers;

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, fpcunit, testregistry, neuralnetwork, neuralvolume;

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
  end;

implementation

procedure TTestNeuralLayers.TestFullyConnectedForward;
var
  NN: TNNet;
  Input, Output: TNNetVolume;
begin
  NN := TNNet.Create();
  Input := TNNetVolume.Create(2, 1, 1);
  Output := TNNetVolume.Create(1, 1, 1);
  try
    NN.AddLayer(TNNetInput.Create(2));
    NN.AddLayer(TNNetFullConnectLinear.Create(1));

    Input.Raw[0] := 1.0;
    Input.Raw[1] := 1.0;

    NN.Compute(Input);
    NN.GetOutput(Output);

    // Just verify it runs without error and produces output
    AssertTrue('Output should have size 1', Output.Size = 1);
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
begin
  NN := TNNet.Create();
  Input := TNNetVolume.Create(8, 8, 3);
  try
    NN.AddLayer(TNNetInput.Create(8, 8, 3));
    NN.AddLayer(TNNetConvolutionReLU.Create(16, 3, 1, 1));

    Input.Fill(1.0);
    NN.Compute(Input);

    AssertEquals('Output SizeX should be 8', 8, NN.GetLastLayer.Output.SizeX);
    AssertEquals('Output depth should be 16', 16, NN.GetLastLayer.Output.Depth);
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
  Input := TNNetVolume.Create(8, 8, 3);
  try
    NN.AddLayer(TNNetInput.Create(8, 8, 3));
    NN.AddLayer(TNNetMaxPool.Create(2));

    Input.Fill(1.0);
    NN.Compute(Input);

    AssertEquals('Output SizeX should be 4 after 2x2 pool', 4, NN.GetLastLayer.Output.SizeX);
    AssertEquals('Output SizeY should be 4 after 2x2 pool', 4, NN.GetLastLayer.Output.SizeY);
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
  Input := TNNetVolume.Create(8, 8, 3);
  try
    NN.AddLayer(TNNetInput.Create(8, 8, 3));
    NN.AddLayer(TNNetAvgPool.Create(2));

    Input.Fill(4.0);
    NN.Compute(Input);

    AssertEquals('Output SizeX should be 4 after 2x2 avg pool', 4, NN.GetLastLayer.Output.SizeX);
    AssertEquals('Output SizeY should be 4 after 2x2 avg pool', 4, NN.GetLastLayer.Output.SizeY);
    AssertEquals('Output Depth should be 3', 3, NN.GetLastLayer.Output.Depth);
    // Average of all 4.0 values should still be 4.0
    AssertEquals('Average pooled value should be 4.0', 4.0, NN.GetLastLayer.Output.Raw[0], 0.0001);
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

    // Set values so we can test min pooling
    Input.Fill(5.0);
    Input[0, 0, 0] := 1.0; // This should be the min in its 2x2 region

    NN.Compute(Input);

    AssertEquals('Output SizeX should be 2 after 2x2 min pool', 2, NN.GetLastLayer.Output.SizeX);
    AssertEquals('Output SizeY should be 2 after 2x2 min pool', 2, NN.GetLastLayer.Output.SizeY);
    // The min value 1.0 should appear in output
    AssertEquals('Min pooled value should be 1.0', 1.0, NN.GetLastLayer.Output[0, 0, 0], 0.0001);
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
  Layer1, Layer2: TNNetLayer;
begin
  NN := TNNet.Create();
  Input := TNNetVolume.Create(8, 8, 3);
  try
    NN.AddLayer(TNNetInput.Create(8, 8, 3));
    
    // Create two parallel paths
    Layer1 := NN.AddLayer(TNNetConvolutionReLU.Create(16, 3, 1, 1));
    Layer2 := NN.AddLayerAfter(TNNetConvolutionReLU.Create(8, 3, 1, 1), NN.Layers[0]);
    
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
  Layer1, Layer2: TNNetLayer;
begin
  NN := TNNet.Create();
  Input := TNNetVolume.Create(8, 8, 16);
  try
    NN.AddLayer(TNNetInput.Create(8, 8, 16));
    
    // Create two parallel paths with same output size
    Layer1 := NN.AddLayer(TNNetConvolutionLinear.Create(16, 3, 1, 1));
    Layer2 := NN.AddLayerAfter(TNNetConvolutionLinear.Create(16, 3, 1, 1), NN.Layers[0]);
    
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

initialization
  RegisterTest(TTestNeuralLayers);

end.
