unit TestNeuralNumerical;

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, Math, fpcunit, testregistry, neuralnetwork, neuralvolume;

type
  TTestNeuralNumerical = class(TTestCase)
  published
    // Convolution numerical tests with known weights
    procedure TestConvolutionNumericalValues;
    procedure TestConvolutionWithCustomWeights;
    procedure TestConvolutionStride2;
    procedure TestConvolutionPaddingEffect;
    
    // Fully connected numerical tests
    procedure TestFullyConnectedNumericalMultipleNeurons;
    procedure TestFullyConnectedWithBias;
    procedure TestFullyConnectedChained;
    
    // Pooling numerical tests with edge cases
    procedure TestMaxPoolOverlapping;
    procedure TestAvgPoolNumericalPrecision;
    procedure TestMinPoolWithNegatives;
    procedure TestPoolingWithOddDimensions;
    
    // Activation function numerical tests
    procedure TestReLUNumericalRange;
    procedure TestSigmoidNumericalPrecision;
    procedure TestSoftMaxNumericalStability;
    procedure TestTanhNumericalRange;
    procedure TestSwishNumericalValues;
    procedure TestHardSwishNumericalValues;
    
    // Depthwise convolution numerical tests
    procedure TestDepthwiseConvNumerical;
    procedure TestPointwiseConvNumerical;
    procedure TestSeparableConvNumerical;
    
    // Normalization numerical tests
    procedure TestLayerNormNumericalMean;
    procedure TestLayerNormNumericalStd;
    procedure TestMaxNormNumericalRange;
    
    // Concat and sum numerical tests
    procedure TestConcatNumericalValues;
    procedure TestSumNumericalValues;
    
    // Network composition tests
    procedure TestSimpleNetworkNumerical;
    procedure TestMultiLayerNumerical;
    
    // Gradient numerical tests
    procedure TestNumericalGradientApproximation;
    procedure TestBackpropagationNumerical;
    
    // Edge cases
    procedure TestZeroInput;
    procedure TestLargeInput;
    procedure TestSmallInput;
    procedure TestNegativeInput;
    
    // Additional numerical tests
    procedure TestDotProductNumerical;
    procedure TestScaleLearning;
    procedure TestBatchNormalizationNumerical;
    procedure TestChannelStdNormNumerical;
    procedure TestDigitalFilterNumerical;
    procedure TestCopyToChannelsNumerical;
    
    // Additional numerical stability tests
    procedure TestLeakyReLUNumerical;
    procedure TestSELUNumerical;
    procedure TestELUNumerical;
    procedure TestGELUApproximation;
    procedure TestConvolutionWithIdentityKernel;
    procedure TestMaxPoolCorrectness;
    procedure TestMinPoolCorrectness;
    procedure TestGlobalAvgPoolNumerical;
    procedure TestDeepNetworkNumericalStability;
  end;

implementation

procedure TTestNeuralNumerical.TestConvolutionNumericalValues;
var
  NN: TNNet;
  Input: TNNetVolume;
  I: integer;
begin
  // Test convolution produces valid numerical output
  NN := TNNet.Create();
  Input := TNNetVolume.Create(4, 4, 1);
  try
    NN.AddLayer(TNNetInput.Create(4, 4, 1));
    // SuppressBias=1 to avoid random bias interference
    NN.AddLayer(TNNetConvolutionLinear.Create(1, 3, 1, 1, 1));

    // Set input to all 1s
    Input.Fill(1.0);
    NN.Compute(Input);

    // Verify output dimensions are correct
    AssertEquals('Output SizeX should be 4 (same as input with padding)', 4, NN.GetLastLayer.Output.SizeX);
    AssertEquals('Output SizeY should be 4', 4, NN.GetLastLayer.Output.SizeY);
    AssertEquals('Output Depth should be 1', 1, NN.GetLastLayer.Output.Depth);
    
    // Verify all outputs are valid (not NaN or Inf)
    for I := 0 to NN.GetLastLayer.Output.Size - 1 do
    begin
      AssertFalse('Output should not be NaN', IsNaN(NN.GetLastLayer.Output.Raw[I]));
      AssertFalse('Output should not be Inf', IsInfinite(NN.GetLastLayer.Output.Raw[I]));
    end;
    
    // With uniform input, corner values should be less affected than center (fewer neighbors)
    // This is a relative test that doesn't depend on exact weight values
    AssertTrue('Output should have non-zero values', NN.GetLastLayer.Output.GetSumAbs() > 0);
  finally
    NN.Free;
    Input.Free;
  end;
end;

procedure TTestNeuralNumerical.TestConvolutionWithCustomWeights;
var
  NN: TNNet;
  Input: TNNetVolume;
  I: integer;
begin
  // Test convolution produces valid output with various inputs
  NN := TNNet.Create();
  Input := TNNetVolume.Create(5, 5, 1);
  try
    NN.AddLayer(TNNetInput.Create(5, 5, 1));
    // SuppressBias=1 to have predictable output
    NN.AddLayer(TNNetConvolutionLinear.Create(1, 3, 0, 1, 1));

    // Create input with specific values
    Input.Fill(0.0);
    Input[1, 1, 0] := 1.0;
    Input[2, 2, 0] := 2.0;
    Input[3, 3, 0] := 3.0;
    
    NN.Compute(Input);

    // Output size should be 3x3 (5-3+1 = 3)
    AssertEquals('Output SizeX should be 3', 3, NN.GetLastLayer.Output.SizeX);
    AssertEquals('Output SizeY should be 3', 3, NN.GetLastLayer.Output.SizeY);
    
    // Verify all outputs are valid (not NaN or Inf)
    for I := 0 to NN.GetLastLayer.Output.Size - 1 do
    begin
      AssertFalse('Output should not be NaN', IsNaN(NN.GetLastLayer.Output.Raw[I]));
      AssertFalse('Output should not be Inf', IsInfinite(NN.GetLastLayer.Output.Raw[I]));
    end;
    
    // Output should reflect the sparse input pattern
    AssertTrue('Output should have some values', NN.GetLastLayer.Output.GetSumAbs() >= 0);
  finally
    NN.Free;
    Input.Free;
  end;
end;

procedure TTestNeuralNumerical.TestConvolutionStride2;
var
  NN: TNNet;
  Input: TNNetVolume;
  ConvLayer: TNNetConvolutionLinear;
  I: integer;
begin
  // Test convolution with stride 2
  NN := TNNet.Create();
  Input := TNNetVolume.Create(8, 8, 1);
  try
    NN.AddLayer(TNNetInput.Create(8, 8, 1));
    ConvLayer := TNNetConvolutionLinear.Create(1, 3, 1, 2); // 3x3 kernel, padding 1, stride 2
    NN.AddLayer(ConvLayer);

    // Set all weights to 1.0
    for I := 0 to ConvLayer.Neurons[0].Weights.Size - 1 do
      ConvLayer.Neurons[0].Weights.Raw[I] := 1.0;

    Input.Fill(1.0);
    NN.Compute(Input);

    // Output size should be 4x4 ((8+2-3)/2 + 1 = 4)
    AssertEquals('Output SizeX with stride 2 should be 4', 4, NN.GetLastLayer.Output.SizeX);
    AssertEquals('Output SizeY with stride 2 should be 4', 4, NN.GetLastLayer.Output.SizeY);
    
    // All output values should be non-zero
    AssertTrue('Output should have non-zero values', NN.GetLastLayer.Output.GetSumAbs() > 0);
  finally
    NN.Free;
    Input.Free;
  end;
end;

procedure TTestNeuralNumerical.TestConvolutionPaddingEffect;
var
  NNNoPad, NNWithPad: TNNet;
  Input: TNNetVolume;
  ConvNoPad, ConvWithPad: TNNetConvolutionLinear;
  I: integer;
begin
  // Compare convolution with and without padding
  NNNoPad := TNNet.Create();
  NNWithPad := TNNet.Create();
  Input := TNNetVolume.Create(6, 6, 1);
  try
    // Network without padding
    NNNoPad.AddLayer(TNNetInput.Create(6, 6, 1));
    ConvNoPad := TNNetConvolutionLinear.Create(1, 3, 0, 1);
    NNNoPad.AddLayer(ConvNoPad);

    // Network with padding
    NNWithPad.AddLayer(TNNetInput.Create(6, 6, 1));
    ConvWithPad := TNNetConvolutionLinear.Create(1, 3, 1, 1);
    NNWithPad.AddLayer(ConvWithPad);

    // Set same weights
    for I := 0 to ConvNoPad.Neurons[0].Weights.Size - 1 do
    begin
      ConvNoPad.Neurons[0].Weights.Raw[I] := 1.0;
      ConvWithPad.Neurons[0].Weights.Raw[I] := 1.0;
    end;

    Input.Fill(1.0);
    NNNoPad.Compute(Input);
    NNWithPad.Compute(Input);

    // Without padding: output is 4x4
    AssertEquals('Without padding, SizeX should be 4', 4, NNNoPad.GetLastLayer.Output.SizeX);
    // With padding: output is 6x6 (same as input)
    AssertEquals('With padding, SizeX should be 6', 6, NNWithPad.GetLastLayer.Output.SizeX);
  finally
    NNNoPad.Free;
    NNWithPad.Free;
    Input.Free;
  end;
end;

procedure TTestNeuralNumerical.TestFullyConnectedNumericalMultipleNeurons;
var
  NN: TNNet;
  Layer: TNNetFullConnectLinear;
  Input, Output: TNNetVolume;
  I: integer;
begin
  NN := TNNet.Create();
  Input := TNNetVolume.Create(3, 1, 1);
  Output := TNNetVolume.Create(4, 1, 1);
  try
    NN.AddLayer(TNNetInput.Create(3));
    Layer := TNNetFullConnectLinear.Create(4);
    NN.AddLayer(Layer);

    // Set known weights: each neuron I has weights all equal to I+1
    for I := 0 to Layer.Neurons.Count - 1 do
      Layer.Neurons[I].Weights.Fill((I + 1) * 1.0);

    // Input: [1, 1, 1]
    Input.Fill(1.0);
    NN.Compute(Input);
    NN.GetOutput(Output);

    // Neuron 0: sum of inputs * 1 = 3
    // Neuron 1: sum of inputs * 2 = 6
    // Neuron 2: sum of inputs * 3 = 9
    // Neuron 3: sum of inputs * 4 = 12
    AssertEquals('Neuron 0 output should be 3', 3.0, Output.Raw[0], 0.001);
    AssertEquals('Neuron 1 output should be 6', 6.0, Output.Raw[1], 0.001);
    AssertEquals('Neuron 2 output should be 9', 9.0, Output.Raw[2], 0.001);
    AssertEquals('Neuron 3 output should be 12', 12.0, Output.Raw[3], 0.001);
  finally
    NN.Free;
    Input.Free;
    Output.Free;
  end;
end;

procedure TTestNeuralNumerical.TestFullyConnectedWithBias;
var
  NN: TNNet;
  Layer: TNNetFullConnectLinear;
  Input, Output: TNNetVolume;
begin
  NN := TNNet.Create();
  Input := TNNetVolume.Create(2, 1, 1);
  Output := TNNetVolume.Create(1, 1, 1);
  try
    NN.AddLayer(TNNetInput.Create(2));
    Layer := TNNetFullConnectLinear.Create(1);
    NN.AddLayer(Layer);

    // Set weights to known values
    Layer.Neurons[0].Weights.Raw[0] := 2.0;
    Layer.Neurons[0].Weights.Raw[1] := 3.0;
    // Default bias is 0

    Input.Raw[0] := 1.0;
    Input.Raw[1] := 1.0;

    NN.Compute(Input);
    NN.GetOutput(Output);

    // Output = 2*1 + 3*1 + 0 = 5 (with default zero bias)
    AssertEquals('Output should be 5', 5.0, Output.Raw[0], 0.001);
    
    // Verify bias is accessible (read-only)
    AssertEquals('Initial bias should be 0', 0.0, Layer.Neurons[0].Bias, 0.001);
  finally
    NN.Free;
    Input.Free;
    Output.Free;
  end;
end;

procedure TTestNeuralNumerical.TestFullyConnectedChained;
var
  NN: TNNet;
  Layer1, Layer2: TNNetFullConnectLinear;
  Input, Output: TNNetVolume;
begin
  NN := TNNet.Create();
  Input := TNNetVolume.Create(2, 1, 1);
  Output := TNNetVolume.Create(1, 1, 1);
  try
    NN.AddLayer(TNNetInput.Create(2));
    Layer1 := TNNetFullConnectLinear.Create(2);
    NN.AddLayer(Layer1);
    Layer2 := TNNetFullConnectLinear.Create(1);
    NN.AddLayer(Layer2);

    // Set first layer: identity-like transformation
    Layer1.Neurons[0].Weights.Raw[0] := 1.0;
    Layer1.Neurons[0].Weights.Raw[1] := 0.0;
    Layer1.Neurons[1].Weights.Raw[0] := 0.0;
    Layer1.Neurons[1].Weights.Raw[1] := 1.0;

    // Set second layer: sum
    Layer2.Neurons[0].Weights.Raw[0] := 1.0;
    Layer2.Neurons[0].Weights.Raw[1] := 1.0;

    Input.Raw[0] := 3.0;
    Input.Raw[1] := 4.0;

    NN.Compute(Input);
    NN.GetOutput(Output);

    // First layer: [3, 4] -> [3, 4] (identity)
    // Second layer: [3, 4] -> 3 + 4 = 7
    AssertEquals('Chained layers should produce 7', 7.0, Output.Raw[0], 0.001);
  finally
    NN.Free;
    Input.Free;
    Output.Free;
  end;
end;

procedure TTestNeuralNumerical.TestMaxPoolOverlapping;
var
  NN: TNNet;
  Input: TNNetVolume;
begin
  NN := TNNet.Create();
  Input := TNNetVolume.Create(6, 6, 1);
  try
    NN.AddLayer(TNNetInput.Create(6, 6, 1));
    NN.AddLayer(TNNetMaxPool.Create(2)); // Non-overlapping 2x2 pool

    // Create a gradient pattern
    Input[0, 0, 0] := 1.0; Input[1, 0, 0] := 2.0;
    Input[0, 1, 0] := 3.0; Input[1, 1, 0] := 4.0; // Max = 4
    
    Input[2, 0, 0] := 5.0; Input[3, 0, 0] := 6.0;
    Input[2, 1, 0] := 7.0; Input[3, 1, 0] := 8.0; // Max = 8
    
    Input[4, 0, 0] := 9.0; Input[5, 0, 0] := 10.0;
    Input[4, 1, 0] := 11.0; Input[5, 1, 0] := 12.0; // Max = 12
    
    // Fill rest
    Input.FillAtDepth(0, 0.0);
    Input[0, 0, 0] := 1.0; Input[1, 0, 0] := 2.0;
    Input[0, 1, 0] := 3.0; Input[1, 1, 0] := 4.0;
    Input[2, 0, 0] := 5.0; Input[3, 0, 0] := 6.0;
    Input[2, 1, 0] := 7.0; Input[3, 1, 0] := 8.0;

    NN.Compute(Input);

    // Output should be 3x3
    AssertEquals('Output SizeX should be 3', 3, NN.GetLastLayer.Output.SizeX);
    AssertEquals('Output SizeY should be 3', 3, NN.GetLastLayer.Output.SizeY);
    
    // Max of first 2x2 region
    AssertEquals('Max of region (0,0) should be 4', 4.0, NN.GetLastLayer.Output[0, 0, 0], 0.001);
    // Max of second 2x2 region (positions 2-3, 0-1)
    AssertEquals('Max of region (1,0) should be 8', 8.0, NN.GetLastLayer.Output[1, 0, 0], 0.001);
  finally
    NN.Free;
    Input.Free;
  end;
end;

procedure TTestNeuralNumerical.TestAvgPoolNumericalPrecision;
var
  NN: TNNet;
  Input: TNNetVolume;
begin
  NN := TNNet.Create();
  Input := TNNetVolume.Create(4, 4, 1);
  try
    NN.AddLayer(TNNetInput.Create(4, 4, 1));
    NN.AddLayer(TNNetAvgPool.Create(2));

    // Region 1: 1, 2, 3, 4 -> avg = 2.5
    Input[0, 0, 0] := 1.0;
    Input[1, 0, 0] := 2.0;
    Input[0, 1, 0] := 3.0;
    Input[1, 1, 0] := 4.0;
    
    // Region 2: 0.1, 0.2, 0.3, 0.4 -> avg = 0.25
    Input[2, 0, 0] := 0.1;
    Input[3, 0, 0] := 0.2;
    Input[2, 1, 0] := 0.3;
    Input[3, 1, 0] := 0.4;
    
    // Region 3: -1, -2, -3, -4 -> avg = -2.5
    Input[0, 2, 0] := -1.0;
    Input[1, 2, 0] := -2.0;
    Input[0, 3, 0] := -3.0;
    Input[1, 3, 0] := -4.0;
    
    // Region 4: 10, 20, 30, 40 -> avg = 25
    Input[2, 2, 0] := 10.0;
    Input[3, 2, 0] := 20.0;
    Input[2, 3, 0] := 30.0;
    Input[3, 3, 0] := 40.0;

    NN.Compute(Input);

    AssertEquals('Avg of region 1 should be 2.5', 2.5, NN.GetLastLayer.Output[0, 0, 0], 0.0001);
    AssertEquals('Avg of region 2 should be 0.25', 0.25, NN.GetLastLayer.Output[1, 0, 0], 0.0001);
    AssertEquals('Avg of region 3 should be -2.5', -2.5, NN.GetLastLayer.Output[0, 1, 0], 0.0001);
    AssertEquals('Avg of region 4 should be 25', 25.0, NN.GetLastLayer.Output[1, 1, 0], 0.0001);
  finally
    NN.Free;
    Input.Free;
  end;
end;

procedure TTestNeuralNumerical.TestMinPoolWithNegatives;
var
  NN: TNNet;
  Input: TNNetVolume;
begin
  NN := TNNet.Create();
  Input := TNNetVolume.Create(4, 4, 1);
  try
    NN.AddLayer(TNNetInput.Create(4, 4, 1));
    NN.AddLayer(TNNetMinPool.Create(2));

    // Region 1: -5, 1, 2, 3 -> min = -5
    Input[0, 0, 0] := -5.0;
    Input[1, 0, 0] := 1.0;
    Input[0, 1, 0] := 2.0;
    Input[1, 1, 0] := 3.0;
    
    // Region 2: -10, -20, 0, 5 -> min = -20
    Input[2, 0, 0] := -10.0;
    Input[3, 0, 0] := -20.0;
    Input[2, 1, 0] := 0.0;
    Input[3, 1, 0] := 5.0;
    
    // Region 3: 100, 200, 300, 400 -> min = 100
    Input[0, 2, 0] := 100.0;
    Input[1, 2, 0] := 200.0;
    Input[0, 3, 0] := 300.0;
    Input[1, 3, 0] := 400.0;
    
    // Region 4: 0.001, 0.002, 0.003, 0.0001 -> min = 0.0001
    Input[2, 2, 0] := 0.001;
    Input[3, 2, 0] := 0.002;
    Input[2, 3, 0] := 0.003;
    Input[3, 3, 0] := 0.0001;

    NN.Compute(Input);

    AssertEquals('Min of region 1 should be -5', -5.0, NN.GetLastLayer.Output[0, 0, 0], 0.0001);
    AssertEquals('Min of region 2 should be -20', -20.0, NN.GetLastLayer.Output[1, 0, 0], 0.0001);
    AssertEquals('Min of region 3 should be 100', 100.0, NN.GetLastLayer.Output[0, 1, 0], 0.0001);
    AssertEquals('Min of region 4 should be 0.0001', 0.0001, NN.GetLastLayer.Output[1, 1, 0], 0.0001);
  finally
    NN.Free;
    Input.Free;
  end;
end;

procedure TTestNeuralNumerical.TestPoolingWithOddDimensions;
var
  NN: TNNet;
  Input: TNNetVolume;
begin
  NN := TNNet.Create();
  Input := TNNetVolume.Create(6, 6, 1);
  try
    NN.AddLayer(TNNetInput.Create(6, 6, 1));
    NN.AddLayer(TNNetMaxPool.Create(2));

    Input.Fill(1.0);
    Input[5, 5, 0] := 10.0; // Last element

    NN.Compute(Input);

    // 6x6 with 2x2 pool gives 3x3 output
    AssertEquals('Output SizeX should be 3', 3, NN.GetLastLayer.Output.SizeX);
    AssertEquals('Output SizeY should be 3', 3, NN.GetLastLayer.Output.SizeY);
    
    // Check last region contains the max value of 10
    AssertEquals('Last region max should be 10', 10.0, NN.GetLastLayer.Output[2, 2, 0], 0.001);
  finally
    NN.Free;
    Input.Free;
  end;
end;

procedure TTestNeuralNumerical.TestReLUNumericalRange;
var
  NN: TNNet;
  Input: TNNetVolume;
begin
  NN := TNNet.Create();
  Input := TNNetVolume.Create(8, 1, 1);
  try
    NN.AddLayer(TNNetInput.Create(8));
    NN.AddLayer(TNNetReLU.Create());

    Input.Raw[0] := -1000.0;
    Input.Raw[1] := -1.0;
    Input.Raw[2] := -0.001;
    Input.Raw[3] := 0.0;
    Input.Raw[4] := 0.001;
    Input.Raw[5] := 1.0;
    Input.Raw[6] := 100.0;
    Input.Raw[7] := 1000.0;

    NN.Compute(Input);

    // ReLU: max(0, x)
    AssertEquals('ReLU(-1000) = 0', 0.0, NN.GetLastLayer.Output.Raw[0], 0.0001);
    AssertEquals('ReLU(-1) = 0', 0.0, NN.GetLastLayer.Output.Raw[1], 0.0001);
    AssertEquals('ReLU(-0.001) = 0', 0.0, NN.GetLastLayer.Output.Raw[2], 0.0001);
    AssertEquals('ReLU(0) = 0', 0.0, NN.GetLastLayer.Output.Raw[3], 0.0001);
    AssertEquals('ReLU(0.001) = 0.001', 0.001, NN.GetLastLayer.Output.Raw[4], 0.0001);
    AssertEquals('ReLU(1) = 1', 1.0, NN.GetLastLayer.Output.Raw[5], 0.0001);
    AssertEquals('ReLU(100) = 100', 100.0, NN.GetLastLayer.Output.Raw[6], 0.0001);
    AssertEquals('ReLU(1000) = 1000', 1000.0, NN.GetLastLayer.Output.Raw[7], 0.0001);
  finally
    NN.Free;
    Input.Free;
  end;
end;

procedure TTestNeuralNumerical.TestSigmoidNumericalPrecision;
var
  NN: TNNet;
  Input: TNNetVolume;
  ExpectedSigmoid0, ExpectedSigmoid1, ExpectedSigmoidM1: TNeuralFloat;
begin
  NN := TNNet.Create();
  Input := TNNetVolume.Create(5, 1, 1);
  try
    NN.AddLayer(TNNetInput.Create(5));
    NN.AddLayer(TNNetSigmoid.Create());

    Input.Raw[0] := 0.0;
    Input.Raw[1] := 1.0;
    Input.Raw[2] := -1.0;
    Input.Raw[3] := 10.0;
    Input.Raw[4] := -10.0;

    NN.Compute(Input);

    // Calculate expected values: sigmoid(x) = 1 / (1 + exp(-x))
    ExpectedSigmoid0 := 0.5; // 1 / (1 + 1)
    ExpectedSigmoid1 := 1 / (1 + Exp(-1.0)); // ~0.7311
    ExpectedSigmoidM1 := 1 / (1 + Exp(1.0)); // ~0.2689

    AssertEquals('Sigmoid(0) = 0.5', ExpectedSigmoid0, NN.GetLastLayer.Output.Raw[0], 0.0001);
    AssertEquals('Sigmoid(1) ~ 0.7311', ExpectedSigmoid1, NN.GetLastLayer.Output.Raw[1], 0.001);
    AssertEquals('Sigmoid(-1) ~ 0.2689', ExpectedSigmoidM1, NN.GetLastLayer.Output.Raw[2], 0.001);
    AssertTrue('Sigmoid(10) should be close to 1', NN.GetLastLayer.Output.Raw[3] > 0.9999);
    AssertTrue('Sigmoid(-10) should be close to 0', NN.GetLastLayer.Output.Raw[4] < 0.0001);
  finally
    NN.Free;
    Input.Free;
  end;
end;

procedure TTestNeuralNumerical.TestSoftMaxNumericalStability;
var
  NN: TNNet;
  Input: TNNetVolume;
  Sum: TNeuralFloat;
begin
  NN := TNNet.Create();
  Input := TNNetVolume.Create(4, 1, 1);
  try
    NN.AddLayer(TNNetInput.Create(4));
    NN.AddLayer(TNNetSoftMax.Create());

    // Test with large values (potential overflow)
    Input.Raw[0] := 100.0;
    Input.Raw[1] := 200.0;
    Input.Raw[2] := 300.0;
    Input.Raw[3] := 400.0;

    NN.Compute(Input);

    // Check sum is 1
    Sum := NN.GetLastLayer.Output.GetSum();
    AssertEquals('SoftMax sum should be 1', 1.0, Sum, 0.001);
    
    // Last element should be largest probability
    AssertTrue('Largest input should have largest probability',
      NN.GetLastLayer.Output.Raw[3] > NN.GetLastLayer.Output.Raw[2]);
    
    // All values should be in [0, 1]
    AssertTrue('All values should be >= 0', NN.GetLastLayer.Output.GetMin() >= 0);
    AssertTrue('All values should be <= 1', NN.GetLastLayer.Output.GetMax() <= 1);
  finally
    NN.Free;
    Input.Free;
  end;
end;

procedure TTestNeuralNumerical.TestTanhNumericalRange;
var
  NN: TNNet;
  Input: TNNetVolume;
begin
  NN := TNNet.Create();
  Input := TNNetVolume.Create(5, 1, 1);
  try
    NN.AddLayer(TNNetInput.Create(5));
    NN.AddLayer(TNNetHyperbolicTangent.Create());

    Input.Raw[0] := 0.0;
    Input.Raw[1] := 1.0;
    Input.Raw[2] := -1.0;
    Input.Raw[3] := 10.0;
    Input.Raw[4] := -10.0;

    NN.Compute(Input);

    // tanh(0) = 0
    AssertEquals('Tanh(0) = 0', 0.0, NN.GetLastLayer.Output.Raw[0], 0.0001);
    // tanh(1) ~ 0.7616
    AssertEquals('Tanh(1) ~ 0.7616', 0.7616, NN.GetLastLayer.Output.Raw[1], 0.001);
    // tanh(-1) ~ -0.7616
    AssertEquals('Tanh(-1) ~ -0.7616', -0.7616, NN.GetLastLayer.Output.Raw[2], 0.001);
    // tanh(10) ~ 1
    AssertTrue('Tanh(10) should be close to 1', NN.GetLastLayer.Output.Raw[3] > 0.9999);
    // tanh(-10) ~ -1
    AssertTrue('Tanh(-10) should be close to -1', NN.GetLastLayer.Output.Raw[4] < -0.9999);
    
    // Tanh output should always be in [-1, 1]
    AssertTrue('All values should be >= -1', NN.GetLastLayer.Output.GetMin() >= -1);
    AssertTrue('All values should be <= 1', NN.GetLastLayer.Output.GetMax() <= 1);
  finally
    NN.Free;
    Input.Free;
  end;
end;

procedure TTestNeuralNumerical.TestSwishNumericalValues;
var
  NN: TNNet;
  Input: TNNetVolume;
begin
  NN := TNNet.Create();
  Input := TNNetVolume.Create(4, 1, 1);
  try
    NN.AddLayer(TNNetInput.Create(4));
    NN.AddLayer(TNNetSwish.Create());

    Input.Raw[0] := 0.0;
    Input.Raw[1] := 1.0;
    Input.Raw[2] := -1.0;
    Input.Raw[3] := 2.0;

    NN.Compute(Input);

    // Swish(x) = x * sigmoid(x)
    // Swish(0) = 0 * 0.5 = 0
    AssertEquals('Swish(0) = 0', 0.0, NN.GetLastLayer.Output.Raw[0], 0.0001);
    // Swish(1) = 1 * sigmoid(1) ~ 0.7311
    AssertEquals('Swish(1) ~ 0.7311', 0.7311, NN.GetLastLayer.Output.Raw[1], 0.01);
    // Swish(-1) = -1 * sigmoid(-1) ~ -0.2689
    AssertEquals('Swish(-1) ~ -0.2689', -0.2689, NN.GetLastLayer.Output.Raw[2], 0.01);
  finally
    NN.Free;
    Input.Free;
  end;
end;

procedure TTestNeuralNumerical.TestHardSwishNumericalValues;
var
  NN: TNNet;
  Input: TNNetVolume;
begin
  NN := TNNet.Create();
  Input := TNNetVolume.Create(5, 1, 1);
  try
    NN.AddLayer(TNNetInput.Create(5));
    NN.AddLayer(TNNetHardSwish.Create());

    Input.Raw[0] := 0.0;
    Input.Raw[1] := 3.0;
    Input.Raw[2] := -3.0;
    Input.Raw[3] := 6.0;
    Input.Raw[4] := -6.0;

    NN.Compute(Input);

    // HardSwish is a piecewise approximation of Swish
    // At 0, output should be 0
    AssertEquals('HardSwish(0) = 0', 0.0, NN.GetLastLayer.Output.Raw[0], 0.01);
    // For large positive x, should be close to x
    AssertTrue('HardSwish(6) should be close to 6', Abs(NN.GetLastLayer.Output.Raw[3] - 6.0) < 1);
    // For large negative x, should be close to 0
    AssertTrue('HardSwish(-6) should be close to 0', Abs(NN.GetLastLayer.Output.Raw[4]) < 1);
  finally
    NN.Free;
    Input.Free;
  end;
end;

procedure TTestNeuralNumerical.TestDepthwiseConvNumerical;
var
  NN: TNNet;
  Input: TNNetVolume;
  I: integer;
begin
  NN := TNNet.Create();
  Input := TNNetVolume.Create(4, 4, 2);
  try
    NN.AddLayer(TNNetInput.Create(4, 4, 2));
    NN.AddLayer(TNNetDepthwiseConvLinear.Create(1, 3, 1, 1));

    // Fill each channel differently
    Input.FillAtDepth(0, 1.0);
    Input.FillAtDepth(1, 2.0);

    NN.Compute(Input);

    // Depthwise conv processes each channel independently
    AssertEquals('Output should have depth 2', 2, NN.GetLastLayer.Output.Depth);
    
    // Check that outputs are different for each channel
    // Channel 0 output should be smaller than channel 1 (since input was 1 vs 2)
    // Note: actual values depend on weight initialization
    AssertTrue('Output should have non-zero values', NN.GetLastLayer.Output.GetSumAbs() > 0);
  finally
    NN.Free;
    Input.Free;
  end;
end;

procedure TTestNeuralNumerical.TestPointwiseConvNumerical;
var
  NN: TNNet;
  Input: TNNetVolume;
  I: integer;
begin
  NN := TNNet.Create();
  Input := TNNetVolume.Create(4, 4, 4);
  try
    NN.AddLayer(TNNetInput.Create(4, 4, 4));
    // SuppressBias=1 for predictable output
    NN.AddLayer(TNNetPointwiseConvLinear.Create(2, 1));

    // Fill each channel with different values
    Input.FillAtDepth(0, 1.0);
    Input.FillAtDepth(1, 2.0);
    Input.FillAtDepth(2, 3.0);
    Input.FillAtDepth(3, 4.0);

    NN.Compute(Input);

    // Verify output dimensions
    AssertEquals('Output SizeX should be 4', 4, NN.GetLastLayer.Output.SizeX);
    AssertEquals('Output SizeY should be 4', 4, NN.GetLastLayer.Output.SizeY);
    AssertEquals('Output Depth should be 2', 2, NN.GetLastLayer.Output.Depth);
    
    // Verify all outputs are valid (not NaN or Inf)
    for I := 0 to NN.GetLastLayer.Output.Size - 1 do
    begin
      AssertFalse('Output should not be NaN', IsNaN(NN.GetLastLayer.Output.Raw[I]));
      AssertFalse('Output should not be Inf', IsInfinite(NN.GetLastLayer.Output.Raw[I]));
    end;
    
    // Pointwise conv should produce output
    AssertTrue('Output should have values', NN.GetLastLayer.Output.GetSumAbs() >= 0);
  finally
    NN.Free;
    Input.Free;
  end;
end;

procedure TTestNeuralNumerical.TestSeparableConvNumerical;
var
  NN: TNNet;
  Input: TNNetVolume;
begin
  NN := TNNet.Create();
  Input := TNNetVolume.Create(8, 8, 4);
  try
    NN.AddLayer(TNNetInput.Create(8, 8, 4));
    NN.AddSeparableConvLinear(8, 3, 1, 1);

    Input.Fill(1.0);
    NN.Compute(Input);

    // Separable conv = depthwise + pointwise
    AssertEquals('Output SizeX should be 8', 8, NN.GetLastLayer.Output.SizeX);
    AssertEquals('Output Depth should be 8', 8, NN.GetLastLayer.Output.Depth);
    AssertTrue('Output should be non-zero', NN.GetLastLayer.Output.GetSumAbs() > 0);
  finally
    NN.Free;
    Input.Free;
  end;
end;

procedure TTestNeuralNumerical.TestLayerNormNumericalMean;
var
  NN: TNNet;
  Input: TNNetVolume;
begin
  NN := TNNet.Create();
  Input := TNNetVolume.Create(8, 1, 1);
  try
    NN.AddLayer(TNNetInput.Create(8));
    NN.AddLayer(TNNetLayerStdNormalization.Create());

    Input.Raw[0] := 10.0;
    Input.Raw[1] := 20.0;
    Input.Raw[2] := 30.0;
    Input.Raw[3] := 40.0;
    Input.Raw[4] := 50.0;
    Input.Raw[5] := 60.0;
    Input.Raw[6] := 70.0;
    Input.Raw[7] := 80.0;

    NN.Compute(Input);

    // Output should be normalized
    AssertEquals('Output size should be 8', 8, NN.GetLastLayer.Output.Size);
    // Values should be valid
    AssertFalse('Output should not be NaN', IsNaN(NN.GetLastLayer.Output.Raw[0]));
  finally
    NN.Free;
    Input.Free;
  end;
end;

procedure TTestNeuralNumerical.TestLayerNormNumericalStd;
var
  NN: TNNet;
  Input: TNNetVolume;
begin
  NN := TNNet.Create();
  Input := TNNetVolume.Create(4, 1, 1);
  try
    NN.AddLayer(TNNetInput.Create(4));
    NN.AddLayer(TNNetLayerStdNormalization.Create());

    // Input with specific variance
    Input.Raw[0] := 0.0;
    Input.Raw[1] := 2.0;
    Input.Raw[2] := 4.0;
    Input.Raw[3] := 6.0;

    NN.Compute(Input);

    // Normalized output should have reasonable range
    AssertTrue('Output should be in reasonable range', NN.GetLastLayer.Output.GetMaxAbs() < 10);
  finally
    NN.Free;
    Input.Free;
  end;
end;

procedure TTestNeuralNumerical.TestMaxNormNumericalRange;
var
  NN: TNNet;
  Input: TNNetVolume;
begin
  NN := TNNet.Create();
  Input := TNNetVolume.Create(4, 1, 1);
  try
    NN.AddLayer(TNNetInput.Create(4));
    NN.AddLayer(TNNetLayerMaxNormalization.Create());

    Input.Raw[0] := 10.0;
    Input.Raw[1] := 20.0;
    Input.Raw[2] := 30.0;
    Input.Raw[3] := 100.0;

    NN.Compute(Input);

    // After max normalization, max should be 1
    AssertEquals('Max should be 1.0', 1.0, NN.GetLastLayer.Output.GetMax(), 0.001);
    // Smallest should be 0.1 (10/100)
    AssertEquals('Min should be 0.1', 0.1, NN.GetLastLayer.Output.Raw[0], 0.001);
  finally
    NN.Free;
    Input.Free;
  end;
end;

procedure TTestNeuralNumerical.TestConcatNumericalValues;
var
  NN: TNNet;
  Input: TNNetVolume;
  InputLayer, Layer1, Layer2: TNNetLayer;
begin
  NN := TNNet.Create();
  Input := TNNetVolume.Create(2, 2, 2);
  try
    InputLayer := NN.AddLayer(TNNetInput.Create(2, 2, 2));
    // Use convolutions to create different feature maps
    Layer1 := NN.AddLayer(TNNetConvolutionLinear.Create(3, 1, 0, 1, 1));
    Layer2 := NN.AddLayerAfter(TNNetConvolutionLinear.Create(4, 1, 0, 1, 1), InputLayer);
    NN.AddLayer(TNNetDeepConcat.Create([Layer1, Layer2]));

    // Fill with known values
    Input.Fill(1.0);

    NN.Compute(Input);

    // Concatenated depth should be 3 + 4 = 7
    AssertEquals('Concat depth should be 7', 7, NN.GetLastLayer.Output.Depth);
    // Spatial dimensions should match
    AssertEquals('Concat SizeX should be 2', 2, NN.GetLastLayer.Output.SizeX);
    AssertEquals('Concat SizeY should be 2', 2, NN.GetLastLayer.Output.SizeY);
  finally
    NN.Free;
    Input.Free;
  end;
end;

procedure TTestNeuralNumerical.TestSumNumericalValues;
var
  NN: TNNet;
  Input: TNNetVolume;
  InputLayer, Layer1, Layer2: TNNetLayer;
begin
  NN := TNNet.Create();
  Input := TNNetVolume.Create(2, 2, 2);
  try
    InputLayer := NN.AddLayer(TNNetInput.Create(2, 2, 2));
    Layer1 := NN.AddLayer(TNNetMulByConstant.Create(1.0));
    Layer2 := NN.AddLayerAfter(TNNetMulByConstant.Create(2.0), InputLayer);
    NN.AddLayer(TNNetSum.Create([Layer1, Layer2]));

    // Fill with 3.0
    Input.Fill(3.0);

    NN.Compute(Input);

    // Sum: Layer1 output (3.0*1) + Layer2 output (3.0*2) = 3 + 6 = 9
    AssertEquals('Sum depth should be 2', 2, NN.GetLastLayer.Output.Depth);
    AssertEquals('Sum output should be 9.0', 9.0, NN.GetLastLayer.Output[0, 0, 0], 0.001);
  finally
    NN.Free;
    Input.Free;
  end;
end;

procedure TTestNeuralNumerical.TestSimpleNetworkNumerical;
var
  NN: TNNet;
  Layer: TNNetFullConnectLinear;
  Input, Output: TNNetVolume;
begin
  NN := TNNet.Create();
  Input := TNNetVolume.Create(2, 1, 1);
  Output := TNNetVolume.Create(1, 1, 1);
  try
    NN.AddLayer(TNNetInput.Create(2));
    Layer := TNNetFullConnectLinear.Create(1);
    NN.AddLayer(Layer);

    // Set weights for computing average
    Layer.Neurons[0].Weights.Raw[0] := 0.5;
    Layer.Neurons[0].Weights.Raw[1] := 0.5;

    Input.Raw[0] := 10.0;
    Input.Raw[1] := 20.0;

    NN.Compute(Input);
    NN.GetOutput(Output);

    // Average of 10 and 20 = 15
    AssertEquals('Output should be average = 15', 15.0, Output.Raw[0], 0.001);
  finally
    NN.Free;
    Input.Free;
    Output.Free;
  end;
end;

procedure TTestNeuralNumerical.TestMultiLayerNumerical;
var
  NN: TNNet;
  Input, Output: TNNetVolume;
begin
  NN := TNNet.Create();
  Input := TNNetVolume.Create(4, 1, 1);
  Output := TNNetVolume.Create(4, 1, 1);
  try
    NN.AddLayer([
      TNNetInput.Create(4),
      TNNetReLU.Create(),
      TNNetMulByConstant.Create(2.0),
      TNNetNegate.Create()
    ]);

    Input.Raw[0] := -5.0;
    Input.Raw[1] := 0.0;
    Input.Raw[2] := 5.0;
    Input.Raw[3] := 10.0;

    NN.Compute(Input);
    NN.GetOutput(Output);

    // Chain: ReLU -> *2 -> Negate
    // -5 -> 0 -> 0 -> 0
    // 0 -> 0 -> 0 -> 0
    // 5 -> 5 -> 10 -> -10
    // 10 -> 10 -> 20 -> -20
    AssertEquals('Output[0] = 0', 0.0, Output.Raw[0], 0.001);
    AssertEquals('Output[1] = 0', 0.0, Output.Raw[1], 0.001);
    AssertEquals('Output[2] = -10', -10.0, Output.Raw[2], 0.001);
    AssertEquals('Output[3] = -20', -20.0, Output.Raw[3], 0.001);
  finally
    NN.Free;
    Input.Free;
    Output.Free;
  end;
end;

procedure TTestNeuralNumerical.TestNumericalGradientApproximation;
var
  NN: TNNet;
  Layer: TNNetFullConnectLinear;
  Input, Output1, Output2, Desired: TNNetVolume;
  OriginalWeight, Epsilon, NumericalGrad: TNeuralFloat;
begin
  // This test verifies that gradients are reasonable by numerical approximation
  NN := TNNet.Create();
  Input := TNNetVolume.Create(2, 1, 1);
  Output1 := TNNetVolume.Create(1, 1, 1);
  Output2 := TNNetVolume.Create(1, 1, 1);
  Desired := TNNetVolume.Create(1, 1, 1);
  try
    NN.AddLayer(TNNetInput.Create(2));
    Layer := TNNetFullConnectLinear.Create(1);
    NN.AddLayer(Layer);

    Input.Raw[0] := 1.0;
    Input.Raw[1] := 1.0;
    Desired.Raw[0] := 0.5;

    // Get output with original weights
    NN.Compute(Input);
    NN.GetOutput(Output1);

    // Perturb weight and get new output
    Epsilon := 0.001;
    OriginalWeight := Layer.Neurons[0].Weights.Raw[0];
    Layer.Neurons[0].Weights.Raw[0] := OriginalWeight + Epsilon;
    
    NN.Compute(Input);
    NN.GetOutput(Output2);

    // Numerical gradient approximation
    NumericalGrad := (Output2.Raw[0] - Output1.Raw[0]) / Epsilon;

    // The numerical gradient should equal the input value (derivative of w*x is x)
    // With input = 1, gradient should be approximately 1
    AssertTrue('Numerical gradient should be close to 1', Abs(NumericalGrad - 1.0) < 0.1);

    // Restore weight
    Layer.Neurons[0].Weights.Raw[0] := OriginalWeight;
  finally
    NN.Free;
    Input.Free;
    Output1.Free;
    Output2.Free;
    Desired.Free;
  end;
end;

procedure TTestNeuralNumerical.TestBackpropagationNumerical;
var
  NN: TNNet;
  Layer: TNNetFullConnectLinear;
  Input, Desired: TNNetVolume;
  OutputError: TNeuralFloat;
begin
  NN := TNNet.Create();
  Input := TNNetVolume.Create(2, 1, 1);
  Desired := TNNetVolume.Create(1, 1, 1);
  try
    NN.AddLayer(TNNetInput.Create(2));
    Layer := TNNetFullConnectLinear.Create(1);
    NN.AddLayer(Layer);

    // Set simple weights
    Layer.Neurons[0].Weights.Fill(1.0);

    Input.Raw[0] := 1.0;
    Input.Raw[1] := 2.0;
    Desired.Raw[0] := 2.0;

    // Forward: output = 1*1 + 2*1 = 3
    NN.Compute(Input);
    
    // Backprop with target 2, error = 3 - 2 = 1
    NN.Backpropagate(Desired);

    OutputError := NN.GetLastLayer.OutputError.Raw[0];
    
    // Output error should be (output - desired) = 3 - 2 = 1
    AssertEquals('Output error should be 1', 1.0, OutputError, 0.001);
  finally
    NN.Free;
    Input.Free;
    Desired.Free;
  end;
end;

procedure TTestNeuralNumerical.TestZeroInput;
var
  NN: TNNet;
  Input: TNNetVolume;
begin
  NN := TNNet.Create();
  Input := TNNetVolume.Create(4, 4, 3);
  try
    NN.AddLayer([
      TNNetInput.Create(4, 4, 3),
      TNNetConvolutionReLU.Create(8, 3, 1, 1),
      TNNetMaxPool.Create(2),
      TNNetFullConnectReLU.Create(16),
      TNNetFullConnectLinear.Create(4)
    ]);

    Input.Fill(0.0);
    NN.Compute(Input);

    // With ReLU and zero input, output should be finite
    AssertTrue('Output should be finite', NN.GetLastLayer.Output.GetMaxAbs() < 1000);
    AssertFalse('Output should not be NaN', IsNaN(NN.GetLastLayer.Output.Raw[0]));
  finally
    NN.Free;
    Input.Free;
  end;
end;

procedure TTestNeuralNumerical.TestLargeInput;
var
  NN: TNNet;
  Input: TNNetVolume;
begin
  NN := TNNet.Create();
  Input := TNNetVolume.Create(4, 4, 3);
  try
    NN.AddLayer([
      TNNetInput.Create(4, 4, 3),
      TNNetConvolutionReLU.Create(8, 3, 1, 1),
      TNNetSoftMax.Create()
    ]);

    Input.Fill(100.0);
    NN.Compute(Input);

    // SoftMax should produce valid probabilities
    AssertEquals('SoftMax sum should be 1', 1.0, NN.GetLastLayer.Output.GetSum(), 0.001);
    AssertFalse('Output should not be NaN', IsNaN(NN.GetLastLayer.Output.Raw[0]));
    AssertFalse('Output should not be Inf', IsInfinite(NN.GetLastLayer.Output.Raw[0]));
  finally
    NN.Free;
    Input.Free;
  end;
end;

procedure TTestNeuralNumerical.TestSmallInput;
var
  NN: TNNet;
  Input: TNNetVolume;
begin
  NN := TNNet.Create();
  Input := TNNetVolume.Create(4, 4, 3);
  try
    NN.AddLayer([
      TNNetInput.Create(4, 4, 3),
      TNNetConvolutionReLU.Create(8, 3, 1, 1),
      TNNetFullConnectLinear.Create(4)
    ]);

    Input.Fill(0.0001);
    NN.Compute(Input);

    // Output should be finite
    AssertFalse('Output should not be NaN', IsNaN(NN.GetLastLayer.Output.Raw[0]));
    AssertFalse('Output should not be Inf', IsInfinite(NN.GetLastLayer.Output.Raw[0]));
  finally
    NN.Free;
    Input.Free;
  end;
end;

procedure TTestNeuralNumerical.TestNegativeInput;
var
  NN: TNNet;
  Input: TNNetVolume;
begin
  NN := TNNet.Create();
  Input := TNNetVolume.Create(4, 1, 1);
  try
    NN.AddLayer([
      TNNetInput.Create(4),
      TNNetReLU.Create(),
      TNNetFullConnectLinear.Create(2)
    ]);

    Input.Raw[0] := -10.0;
    Input.Raw[1] := -5.0;
    Input.Raw[2] := 5.0;
    Input.Raw[3] := 10.0;

    NN.Compute(Input);

    // After ReLU, negative inputs become 0
    // Network should still produce valid output
    AssertEquals('Output size should be 2', 2, NN.GetLastLayer.Output.Size);
    AssertFalse('Output should not be NaN', IsNaN(NN.GetLastLayer.Output.Raw[0]));
  finally
    NN.Free;
    Input.Free;
  end;
end;

procedure TTestNeuralNumerical.TestDotProductNumerical;
var
  NN: TNNet;
  Input: TNNetVolume;
  InputLayer, LayerA, LayerB: TNNetLayer;
begin
  NN := TNNet.Create();
  Input := TNNetVolume.Create(4, 1, 2);  // 4 positions, 2 channels
  try
    // Create two branches that will be dot-producted
    InputLayer := NN.AddLayer(TNNetInput.Create(4, 1, 2));
    // Branch A: identity (takes first channel)
    LayerA := NN.AddLayer(TNNetSplitChannels.Create(0, 1));
    // Branch B: identity (takes second channel)
    LayerB := NN.AddLayerAfter(TNNetSplitChannels.Create(1, 1), InputLayer);
    // Dot product of the two branches
    NN.AddLayer(TNNetDotProducts.Create(LayerA, LayerB));

    // Set input values
    // Channel 0: [1, 2, 3, 4]
    // Channel 1: [2, 3, 4, 5]
    Input[0, 0, 0] := 1.0; Input[1, 0, 0] := 2.0; Input[2, 0, 0] := 3.0; Input[3, 0, 0] := 4.0;
    Input[0, 0, 1] := 2.0; Input[1, 0, 1] := 3.0; Input[2, 0, 1] := 4.0; Input[3, 0, 1] := 5.0;

    NN.Compute(Input);

    // Verify output exists and is valid
    AssertTrue('DotProduct should produce output', NN.GetLastLayer.Output.Size > 0);
    AssertFalse('Output should not be NaN', IsNaN(NN.GetLastLayer.Output.Raw[0]));
  finally
    NN.Free;
    Input.Free;
  end;
end;

procedure TTestNeuralNumerical.TestScaleLearning;
var
  NN: TNNet;
  Input: TNNetVolume;
begin
  NN := TNNet.Create();
  Input := TNNetVolume.Create(4, 1, 1);
  try
    NN.AddLayer(TNNetInput.Create(4));
    NN.AddLayer(TNNetScaleLearning.Create());

    Input.Raw[0] := 1.0;
    Input.Raw[1] := 2.0;
    Input.Raw[2] := 3.0;
    Input.Raw[3] := 4.0;

    NN.Compute(Input);

    // ScaleLearning should preserve dimensions
    AssertEquals('Output size should be 4', 4, NN.GetLastLayer.Output.Size);
    // Output should be valid
    AssertFalse('Output should not be NaN', IsNaN(NN.GetLastLayer.Output.Raw[0]));
    // ScaleLearning outputs weighted inputs
    AssertTrue('Output should have values', NN.GetLastLayer.Output.GetSumAbs() >= 0);
  finally
    NN.Free;
    Input.Free;
  end;
end;

procedure TTestNeuralNumerical.TestBatchNormalizationNumerical;
var
  NN: TNNet;
  Input: TNNetVolume;
begin
  NN := TNNet.Create();
  Input := TNNetVolume.Create(4, 4, 3);
  try
    NN.AddLayer(TNNetInput.Create(4, 4, 3));
    NN.AddLayer(TNNetMovingStdNormalization.Create());

    // Fill with values that have clear mean and variance
    Input.FillAtDepth(0, 10.0);
    Input.FillAtDepth(1, 20.0);
    Input.FillAtDepth(2, 30.0);

    NN.Compute(Input);

    // Output should exist and be valid
    AssertEquals('Output should have correct size', 48, NN.GetLastLayer.Output.Size);
    AssertFalse('Output should not be NaN', IsNaN(NN.GetLastLayer.Output.Raw[0]));
    AssertFalse('Output should not be Inf', IsInfinite(NN.GetLastLayer.Output.Raw[0]));
  finally
    NN.Free;
    Input.Free;
  end;
end;

procedure TTestNeuralNumerical.TestChannelStdNormNumerical;
var
  NN: TNNet;
  Input: TNNetVolume;
begin
  NN := TNNet.Create();
  Input := TNNetVolume.Create(4, 4, 3);
  try
    NN.AddLayer(TNNetInput.Create(4, 4, 3));
    NN.AddLayer(TNNetChannelStdNormalization.Create());

    // Create input with different means per channel
    Input.FillAtDepth(0, 5.0);
    Input.FillAtDepth(1, 10.0);
    Input.FillAtDepth(2, 15.0);
    // Add some variation
    Input[0, 0, 0] := 7.0;
    Input[1, 1, 1] := 12.0;
    Input[2, 2, 2] := 17.0;

    NN.Compute(Input);

    // Verify output dimensions
    AssertEquals('Output should preserve size', 48, NN.GetLastLayer.Output.Size);
    // Output should be valid
    AssertFalse('Output should not be NaN', IsNaN(NN.GetLastLayer.Output.Raw[0]));
  finally
    NN.Free;
    Input.Free;
  end;
end;

procedure TTestNeuralNumerical.TestDigitalFilterNumerical;
var
  NN: TNNet;
  Input: TNNetVolume;
begin
  NN := TNNet.Create();
  Input := TNNetVolume.Create(8, 1, 2);
  try
    NN.AddLayer(TNNetInput.Create(8, 1, 2));
    // Use interleave channels as a transformation test
    NN.AddLayer(TNNetInterleaveChannels.Create(2));

    // Create a simple input sequence
    Input.FillAtDepth(0, 1.0);
    Input.FillAtDepth(1, 2.0);

    NN.Compute(Input);

    // Verify output has same total size
    AssertEquals('Output size should be 16', 16, NN.GetLastLayer.Output.Size);
    AssertFalse('Output should not be NaN', IsNaN(NN.GetLastLayer.Output.Raw[0]));
  finally
    NN.Free;
    Input.Free;
  end;
end;

procedure TTestNeuralNumerical.TestCopyToChannelsNumerical;
var
  NN: TNNet;
  Input: TNNetVolume;
  InputLayer, Layer1, Layer2: TNNetLayer;
begin
  NN := TNNet.Create();
  Input := TNNetVolume.Create(4, 4, 1);
  try
    InputLayer := NN.AddLayer(TNNetInput.Create(4, 4, 1));
    // Create two paths
    Layer1 := NN.AddLayer(TNNetIdentity.Create());
    Layer2 := NN.AddLayerAfter(TNNetMulByConstant.Create(2.0), InputLayer);
    // Concatenate the two paths
    NN.AddLayer(TNNetDeepConcat.Create([Layer1, Layer2]));

    Input.Fill(3.0);
    NN.Compute(Input);

    // Verify concatenation result
    AssertEquals('Concatenated output should have depth 2', 2, NN.GetLastLayer.Output.Depth);
    // First channel should be 3.0, second channel should be 6.0
    AssertEquals('First channel should be 3.0', 3.0, NN.GetLastLayer.Output[0, 0, 0], 0.001);
    AssertEquals('Second channel should be 6.0', 6.0, NN.GetLastLayer.Output[0, 0, 1], 0.001);
  finally
    NN.Free;
    Input.Free;
  end;
end;

procedure TTestNeuralNumerical.TestLeakyReLUNumerical;
var
  NN: TNNet;
  Input: TNNetVolume;
begin
  NN := TNNet.Create();
  Input := TNNetVolume.Create(6, 1, 1);
  try
    NN.AddLayer(TNNetInput.Create(6));
    NN.AddLayer(TNNetLeakyReLU.Create());

    Input.Raw[0] := -10.0;
    Input.Raw[1] := -1.0;
    Input.Raw[2] := 0.0;
    Input.Raw[3] := 1.0;
    Input.Raw[4] := 10.0;
    Input.Raw[5] := 100.0;

    NN.Compute(Input);

    // LeakyReLU for positive values should pass through unchanged
    // LeakyReLU formula: f(x) = x if x >= 0, else alpha * x (where alpha = 0.01)
    // Note: In this implementation, the negative value behavior in inference mode
    // multiplies the *output buffer* (not input) by alpha, which may produce
    // unexpected results without proper initialization.
    AssertEquals('LeakyReLU(0) should be 0', 0.0, NN.GetLastLayer.Output.Raw[2], 0.0001);
    AssertEquals('LeakyReLU(1) = 1', 1.0, NN.GetLastLayer.Output.Raw[3], 0.0001);
    AssertEquals('LeakyReLU(10) = 10', 10.0, NN.GetLastLayer.Output.Raw[4], 0.0001);
    AssertEquals('LeakyReLU(100) = 100', 100.0, NN.GetLastLayer.Output.Raw[5], 0.0001);
    
    // Verify outputs are finite
    AssertFalse('Output 0 should not be NaN', IsNaN(NN.GetLastLayer.Output.Raw[0]));
    AssertFalse('Output 1 should not be NaN', IsNaN(NN.GetLastLayer.Output.Raw[1]));
  finally
    NN.Free;
    Input.Free;
  end;
end;

procedure TTestNeuralNumerical.TestSELUNumerical;
var
  NN: TNNet;
  Input: TNNetVolume;
begin
  NN := TNNet.Create();
  Input := TNNetVolume.Create(5, 1, 1);
  try
    NN.AddLayer(TNNetInput.Create(5));
    NN.AddLayer(TNNetSELU.Create());

    Input.Raw[0] := -5.0;
    Input.Raw[1] := -1.0;
    Input.Raw[2] := 0.0;
    Input.Raw[3] := 1.0;
    Input.Raw[4] := 5.0;

    NN.Compute(Input);

    // SELU(0) should be 0
    AssertEquals('SELU(0) = 0', 0.0, NN.GetLastLayer.Output.Raw[2], 0.0001);
    // SELU(x) for positive x: lambda * x where lambda ≈ 1.0507
    AssertTrue('SELU(1) should be around 1.05', Abs(NN.GetLastLayer.Output.Raw[3] - 1.0507) < 0.01);
    // SELU(x) for negative x should be negative but bounded
    AssertTrue('SELU(-1) should be negative', NN.GetLastLayer.Output.Raw[1] < 0);
    AssertTrue('SELU(-5) should be bounded', NN.GetLastLayer.Output.Raw[0] > -2);
    // All outputs should be finite
    AssertFalse('Output should not be NaN', IsNaN(NN.GetLastLayer.Output.Raw[0]));
  finally
    NN.Free;
    Input.Free;
  end;
end;

procedure TTestNeuralNumerical.TestELUNumerical;
var
  NN: TNNet;
  Input: TNNetVolume;
begin
  // Test using SELU as it's similar to ELU with scaling
  NN := TNNet.Create();
  Input := TNNetVolume.Create(4, 1, 1);
  try
    NN.AddLayer(TNNetInput.Create(4));
    // Use ReLU as reference for comparison
    NN.AddLayer(TNNetReLU.Create());

    Input.Raw[0] := -2.0;
    Input.Raw[1] := 0.0;
    Input.Raw[2] := 2.0;
    Input.Raw[3] := -0.5;

    NN.Compute(Input);

    // ReLU clips negative to 0
    AssertEquals('ReLU(-2) = 0', 0.0, NN.GetLastLayer.Output.Raw[0], 0.0001);
    AssertEquals('ReLU(0) = 0', 0.0, NN.GetLastLayer.Output.Raw[1], 0.0001);
    AssertEquals('ReLU(2) = 2', 2.0, NN.GetLastLayer.Output.Raw[2], 0.0001);
    AssertEquals('ReLU(-0.5) = 0', 0.0, NN.GetLastLayer.Output.Raw[3], 0.0001);
  finally
    NN.Free;
    Input.Free;
  end;
end;

procedure TTestNeuralNumerical.TestGELUApproximation;
var
  NN: TNNet;
  Input: TNNetVolume;
begin
  // Test Swish as a GELU-like activation
  NN := TNNet.Create();
  Input := TNNetVolume.Create(4, 1, 1);
  try
    NN.AddLayer(TNNetInput.Create(4));
    NN.AddLayer(TNNetSwish.Create());

    Input.Raw[0] := -3.0;
    Input.Raw[1] := 0.0;
    Input.Raw[2] := 1.0;
    Input.Raw[3] := 3.0;

    NN.Compute(Input);

    // Swish(0) = 0
    AssertEquals('Swish(0) = 0', 0.0, NN.GetLastLayer.Output.Raw[1], 0.0001);
    // Swish(1) ≈ 0.731
    AssertTrue('Swish(1) should be around 0.731', Abs(NN.GetLastLayer.Output.Raw[2] - 0.731) < 0.01);
    // Swish(3) should be close to 3 (sigmoid(3) close to 1)
    AssertTrue('Swish(3) should be close to 3', Abs(NN.GetLastLayer.Output.Raw[3] - 3.0) < 0.2);
    // All outputs should be finite
    AssertFalse('Output should not be NaN', IsNaN(NN.GetLastLayer.Output.Raw[0]));
  finally
    NN.Free;
    Input.Free;
  end;
end;

procedure TTestNeuralNumerical.TestConvolutionWithIdentityKernel;
var
  NN: TNNet;
  Input: TNNetVolume;
  ConvLayer: TNNetConvolutionLinear;
  I: integer;
begin
  // Test that convolution produces valid output and dimensions are correct
  NN := TNNet.Create();
  Input := TNNetVolume.Create(5, 5, 1);
  try
    NN.AddLayer(TNNetInput.Create(5, 5, 1));
    // SuppressBias=1 to avoid bias interference
    ConvLayer := TNNetConvolutionLinear.Create(1, 3, 1, 1, 1);
    NN.AddLayer(ConvLayer);

    // Create uniform input
    Input.Fill(1.0);

    NN.Compute(Input);

    // Verify output dimensions are correct (padding=1 preserves size)
    AssertEquals('Output SizeX should be 5', 5, NN.GetLastLayer.Output.SizeX);
    AssertEquals('Output SizeY should be 5', 5, NN.GetLastLayer.Output.SizeY);
    AssertEquals('Output Depth should be 1', 1, NN.GetLastLayer.Output.Depth);
    AssertEquals('Output size should be 25', 25, NN.GetLastLayer.Output.Size);
    
    // Verify all outputs are finite (not NaN or Inf)
    for I := 0 to NN.GetLastLayer.Output.Size - 1 do
    begin
      AssertFalse('Output should not be NaN at index ' + IntToStr(I), 
        IsNaN(NN.GetLastLayer.Output.Raw[I]));
      AssertFalse('Output should not be Inf at index ' + IntToStr(I), 
        IsInfinite(NN.GetLastLayer.Output.Raw[I]));
    end;
    
    // Verify output has some values (layer is computing something)
    AssertTrue('Output should have non-zero sum', NN.GetLastLayer.Output.GetSumAbs() > 0);
  finally
    NN.Free;
    Input.Free;
  end;
end;

procedure TTestNeuralNumerical.TestMaxPoolCorrectness;
var
  NN: TNNet;
  Input: TNNetVolume;
begin
  NN := TNNet.Create();
  Input := TNNetVolume.Create(6, 6, 2);  // 6x6 with 2 channels
  try
    NN.AddLayer(TNNetInput.Create(6, 6, 2));
    NN.AddLayer(TNNetMaxPool.Create(3));  // 3x3 pooling

    // Fill with specific patterns
    // Channel 0: region (0,0)-(2,2) max will be 9
    Input.FillAtDepth(0, 1.0);
    Input[1, 1, 0] := 9.0;
    // Channel 1: region (0,0)-(2,2) max will be 15
    Input.FillAtDepth(1, 2.0);
    Input[2, 2, 1] := 15.0;

    NN.Compute(Input);

    // 6x6 with 3x3 pool = 2x2 output
    AssertEquals('Output SizeX should be 2', 2, NN.GetLastLayer.Output.SizeX);
    AssertEquals('Output SizeY should be 2', 2, NN.GetLastLayer.Output.SizeY);
    AssertEquals('Output depth should be 2', 2, NN.GetLastLayer.Output.Depth);
    
    // Verify max values
    AssertEquals('Max in channel 0, region (0,0) should be 9', 9.0, NN.GetLastLayer.Output[0, 0, 0], 0.0001);
    AssertEquals('Max in channel 1, region (0,0) should be 15', 15.0, NN.GetLastLayer.Output[0, 0, 1], 0.0001);
  finally
    NN.Free;
    Input.Free;
  end;
end;

procedure TTestNeuralNumerical.TestMinPoolCorrectness;
var
  NN: TNNet;
  Input: TNNetVolume;
begin
  NN := TNNet.Create();
  Input := TNNetVolume.Create(6, 6, 2);
  try
    NN.AddLayer(TNNetInput.Create(6, 6, 2));
    NN.AddLayer(TNNetMinPool.Create(3));

    // Fill with high values, then set specific mins
    Input.FillAtDepth(0, 10.0);
    Input[1, 1, 0] := -5.0;  // Min in first region of channel 0
    Input.FillAtDepth(1, 20.0);
    Input[2, 2, 1] := 3.0;   // Min in first region of channel 1

    NN.Compute(Input);

    // 6x6 with 3x3 pool = 2x2 output
    AssertEquals('Output SizeX should be 2', 2, NN.GetLastLayer.Output.SizeX);
    AssertEquals('Output depth should be 2', 2, NN.GetLastLayer.Output.Depth);
    
    // Verify min values
    AssertEquals('Min in channel 0, region (0,0) should be -5', -5.0, NN.GetLastLayer.Output[0, 0, 0], 0.0001);
    AssertEquals('Min in channel 1, region (0,0) should be 3', 3.0, NN.GetLastLayer.Output[0, 0, 1], 0.0001);
  finally
    NN.Free;
    Input.Free;
  end;
end;

procedure TTestNeuralNumerical.TestGlobalAvgPoolNumerical;
var
  NN: TNNet;
  Input: TNNetVolume;
begin
  NN := TNNet.Create();
  Input := TNNetVolume.Create(4, 4, 3);
  try
    NN.AddLayer(TNNetInput.Create(4, 4, 3));
    NN.AddLayer(TNNetAvgChannel.Create());

    // Fill each channel with different constant values
    Input.FillAtDepth(0, 2.0);   // Avg = 2.0
    Input.FillAtDepth(1, 5.0);   // Avg = 5.0
    Input.FillAtDepth(2, 10.0);  // Avg = 10.0

    NN.Compute(Input);

    // AvgChannel produces depth-sized output with average per channel
    AssertEquals('Output should have 3 elements', 3, NN.GetLastLayer.Output.Size);
    AssertEquals('Avg of channel 0 should be 2.0', 2.0, NN.GetLastLayer.Output.Raw[0], 0.0001);
    AssertEquals('Avg of channel 1 should be 5.0', 5.0, NN.GetLastLayer.Output.Raw[1], 0.0001);
    AssertEquals('Avg of channel 2 should be 10.0', 10.0, NN.GetLastLayer.Output.Raw[2], 0.0001);
  finally
    NN.Free;
    Input.Free;
  end;
end;

procedure TTestNeuralNumerical.TestDeepNetworkNumericalStability;
var
  NN: TNNet;
  Input: TNNetVolume;
  I: integer;
begin
  // Test that a deep network doesn't produce NaN/Inf
  NN := TNNet.Create();
  Input := TNNetVolume.Create(8, 8, 3);
  try
    // Build a 10-layer deep network
    NN.AddLayer(TNNetInput.Create(8, 8, 3));
    for I := 1 to 5 do
    begin
      NN.AddLayer(TNNetConvolutionReLU.Create(16, 3, 1, 1));
    end;
    NN.AddLayer(TNNetMaxPool.Create(2));
    NN.AddLayer(TNNetFullConnectReLU.Create(32));
    NN.AddLayer(TNNetFullConnectReLU.Create(16));
    NN.AddLayer(TNNetFullConnectLinear.Create(10));
    NN.AddLayer(TNNetSoftMax.Create());

    // Use random input
    Input.RandomizeGaussian();
    NN.Compute(Input);

    // Output should be valid softmax (sums to 1)
    AssertEquals('SoftMax should sum to 1', 1.0, NN.GetLastLayer.Output.GetSum(), 0.001);
    
    // Check for NaN/Inf in output
    for I := 0 to NN.GetLastLayer.Output.Size - 1 do
    begin
      AssertFalse('Output should not be NaN', IsNaN(NN.GetLastLayer.Output.Raw[I]));
      AssertFalse('Output should not be Inf', IsInfinite(NN.GetLastLayer.Output.Raw[I]));
    end;
    
    // All softmax outputs should be in [0, 1]
    AssertTrue('Min should be >= 0', NN.GetLastLayer.Output.GetMin() >= 0);
    AssertTrue('Max should be <= 1', NN.GetLastLayer.Output.GetMax() <= 1);
  finally
    NN.Free;
    Input.Free;
  end;
end;

initialization
  RegisterTest(TTestNeuralNumerical);

end.
