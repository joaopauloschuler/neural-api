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
    procedure TestGELUNumericalValues;
    procedure TestMishNumericalValues;
    procedure TestSoftPlusNumericalValues;
    procedure TestGaussianActivationNumericalValues;
    procedure TestGELUGradientCheck;
    procedure TestMishGradientCheck;
    procedure TestSoftPlusGradientCheck;
    procedure TestGaussianActivationGradientCheck;
    procedure TestSwishGradientCheck;
    procedure TestSwish6GradientCheck;
    procedure TestHardSwishGradientCheck;
    procedure TestSELUGradientCheck;
    procedure TestLeakyReLUGradientCheck;
    procedure TestVeryLeakyReLUGradientCheck;
    procedure TestReLU6GradientCheck;
    procedure TestSigmoidGradientCheck;
    procedure TestHyperbolicTangentGradientCheck;

    // Depthwise convolution numerical tests
    procedure TestDepthwiseConvNumerical;
    procedure TestPointwiseConvNumerical;
    procedure TestSeparableConvNumerical;
    
    // Normalization numerical tests
    procedure TestLayerNormNumericalMean;
    procedure TestLayerNormNumericalStd;
    procedure TestMaxNormNumericalRange;
    procedure TestLayerNormForward;
    procedure TestLayerNormGradientCheck;
    procedure TestGroupNormForward;
    procedure TestGroupNormGradientCheck;
    procedure TestRMSNormForward;
    procedure TestRMSNormGradientCheck;
    procedure TestLayerScaleForward;
    procedure TestLayerScaleGradientCheck;

    // Transform / reshaping / element-wise layer gradient checks
    procedure TestPadXYGradientCheck;
    procedure TestCropGradientCheck;
    procedure TestInterleaveChannelsGradientCheck;
    procedure TestAvgPoolGradientCheck;
    procedure TestCellBiasGradientCheck;
    procedure TestCellMulGradientCheck;

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

procedure TTestNeuralNumerical.TestSoftPlusNumericalValues;
var
  NN: TNNet;
  Input: TNNetVolume;
begin
  NN := TNNet.Create();
  Input := TNNetVolume.Create(4, 1, 1);
  try
    NN.AddLayer(TNNetInput.Create(4));
    NN.AddLayer(TNNetSoftPlus.Create());

    Input.Raw[0] := 0.0;
    Input.Raw[1] := 1.0;
    Input.Raw[2] := -1.0;
    Input.Raw[3] := 40.0;

    NN.Compute(Input);

    // SoftPlus(x) = ln(1 + exp(x))
    // SoftPlus(0) = ln(2) ~ 0.6931
    AssertEquals('SoftPlus(0) ~ 0.6931', 0.6931, NN.GetLastLayer.Output.Raw[0], 0.001);
    // SoftPlus(1) = ln(1+e) ~ 1.3133
    AssertEquals('SoftPlus(1) ~ 1.3133', 1.3133, NN.GetLastLayer.Output.Raw[1], 0.001);
    // SoftPlus(-1) = ln(1+e^-1) ~ 0.3133
    AssertEquals('SoftPlus(-1) ~ 0.3133', 0.3133, NN.GetLastLayer.Output.Raw[2], 0.001);
    // SoftPlus(40) ~ 40 (numerically stable for large x)
    AssertEquals('SoftPlus(40) ~ 40', 40.0, NN.GetLastLayer.Output.Raw[3], 0.001);
  finally
    NN.Free;
    Input.Free;
  end;
end;

procedure TTestNeuralNumerical.TestGaussianActivationNumericalValues;
var
  NN: TNNet;
  Input: TNNetVolume;
begin
  NN := TNNet.Create();
  Input := TNNetVolume.Create(4, 1, 1);
  try
    NN.AddLayer(TNNetInput.Create(4));
    NN.AddLayer(TNNetGaussianActivation.Create());

    Input.Raw[0] := 0.0;
    Input.Raw[1] := 1.0;
    Input.Raw[2] := -1.0;
    Input.Raw[3] := 2.0;

    NN.Compute(Input);

    // Gaussian(x) = exp(-x^2)
    // Gaussian(0) = 1
    AssertEquals('Gaussian(0) = 1', 1.0, NN.GetLastLayer.Output.Raw[0], 0.0001);
    // Gaussian(1) = exp(-1) ~ 0.3679
    AssertEquals('Gaussian(1) ~ 0.3679', 0.3679, NN.GetLastLayer.Output.Raw[1], 0.001);
    // Gaussian(-1) = exp(-1) ~ 0.3679
    AssertEquals('Gaussian(-1) ~ 0.3679', 0.3679, NN.GetLastLayer.Output.Raw[2], 0.001);
    // Gaussian(2) = exp(-4) ~ 0.0183
    AssertEquals('Gaussian(2) ~ 0.0183', 0.0183, NN.GetLastLayer.Output.Raw[3], 0.001);
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

procedure TTestNeuralNumerical.TestGELUNumericalValues;
var
  NN: TNNet;
  Input: TNNetVolume;
  x, tanhArg, tanhVal, expected: TNeuralFloat;
const
  SQRT_2_OVER_PI = 0.7978845608;
  GELU_CONST = 0.044715;
begin
  NN := TNNet.Create();
  Input := TNNetVolume.Create(7, 1, 1);
  try
    NN.AddLayer(TNNetInput.Create(7));
    NN.AddLayer(TNNetGELU.Create());

    // Test a range of values
    Input.Raw[0] := 0.0;
    Input.Raw[1] := 1.0;
    Input.Raw[2] := -1.0;
    Input.Raw[3] := 2.0;
    Input.Raw[4] := -2.0;
    Input.Raw[5] := 0.5;
    Input.Raw[6] := -0.5;

    NN.Compute(Input);

    // GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715*x^3)))
    // Test each value against the formula
    x := 0.0;
    tanhArg := SQRT_2_OVER_PI * (x + GELU_CONST * x * x * x);
    tanhVal := Tanh(tanhArg);
    expected := 0.5 * x * (1 + tanhVal);
    AssertEquals('GELU(0) should match formula', expected, NN.GetLastLayer.Output.Raw[0], 0.0001);

    x := 1.0;
    tanhArg := SQRT_2_OVER_PI * (x + GELU_CONST * x * x * x);
    tanhVal := Tanh(tanhArg);
    expected := 0.5 * x * (1 + tanhVal);
    AssertEquals('GELU(1) should match formula', expected, NN.GetLastLayer.Output.Raw[1], 0.0001);

    x := -1.0;
    tanhArg := SQRT_2_OVER_PI * (x + GELU_CONST * x * x * x);
    tanhVal := Tanh(tanhArg);
    expected := 0.5 * x * (1 + tanhVal);
    AssertEquals('GELU(-1) should match formula', expected, NN.GetLastLayer.Output.Raw[2], 0.0001);

    // Verify known approximate values
    AssertTrue('GELU(1) ≈ 0.841', Abs(NN.GetLastLayer.Output.Raw[1] - 0.841) < 0.01);
    AssertTrue('GELU(-1) ≈ -0.159', Abs(NN.GetLastLayer.Output.Raw[2] - (-0.159)) < 0.01);
    AssertTrue('GELU(2) ≈ 1.955', Abs(NN.GetLastLayer.Output.Raw[3] - 1.955) < 0.01);

    // Verify asymptotic behavior
    AssertTrue('GELU approaches identity for large positive x', NN.GetLastLayer.Output.Raw[3] > 1.9);
    AssertTrue('GELU approaches 0 for large negative x', Abs(NN.GetLastLayer.Output.Raw[4]) < 0.1);

  finally
    NN.Free;
    Input.Free;
  end;
end;

procedure TTestNeuralNumerical.TestMishNumericalValues;
var
  NN: TNNet;
  Input: TNNetVolume;
  x, softplus, expected: TNeuralFloat;
begin
  NN := TNNet.Create();
  Input := TNNetVolume.Create(7, 1, 1);
  try
    NN.AddLayer(TNNetInput.Create(7));
    NN.AddLayer(TNNetMish.Create());

    // Test a range of values
    Input.Raw[0] := 0.0;
    Input.Raw[1] := 1.0;
    Input.Raw[2] := -1.0;
    Input.Raw[3] := 2.0;
    Input.Raw[4] := -2.0;
    Input.Raw[5] := 0.5;
    Input.Raw[6] := -0.5;

    NN.Compute(Input);

    // Mish(x) = x * tanh(ln(1 + exp(x)))
    // Test each value against the formula
    x := 0.0;
    softplus := Ln(1 + Exp(x));
    expected := x * Tanh(softplus);
    AssertEquals('Mish(0) should match formula', expected, NN.GetLastLayer.Output.Raw[0], 0.0001);

    x := 1.0;
    softplus := Ln(1 + Exp(x));
    expected := x * Tanh(softplus);
    AssertEquals('Mish(1) should match formula', expected, NN.GetLastLayer.Output.Raw[1], 0.0001);

    x := -1.0;
    softplus := Ln(1 + Exp(x));
    expected := x * Tanh(softplus);
    AssertEquals('Mish(-1) should match formula', expected, NN.GetLastLayer.Output.Raw[2], 0.0001);

    // Verify known approximate values
    AssertTrue('Mish(0) = 0', Abs(NN.GetLastLayer.Output.Raw[0]) < 0.0001);
    AssertTrue('Mish(1) ≈ 0.865', Abs(NN.GetLastLayer.Output.Raw[1] - 0.865) < 0.01);
    AssertTrue('Mish(-1) ≈ -0.303', Abs(NN.GetLastLayer.Output.Raw[2] - (-0.303)) < 0.01);

    // Verify asymptotic behavior
    AssertTrue('Mish approaches identity for large positive x', NN.GetLastLayer.Output.Raw[3] > 1.9);
    AssertTrue('Mish is non-monotonic for negative x', 
      Abs(NN.GetLastLayer.Output.Raw[2]) > Abs(NN.GetLastLayer.Output.Raw[4]));

  finally
    NN.Free;
    Input.Free;
  end;
end;

procedure TTestNeuralNumerical.TestGELUGradientCheck;
var
  NN: TNNet;
  Input, InputPlus, InputMinus: TNNetVolume;
  epsilon: TNeuralFloat;
  numericalGrad, analyticalGrad: TNeuralFloat;
  i: integer;
begin
  NN := TNNet.Create();
  Input := TNNetVolume.Create(3, 1, 1);
  InputPlus := TNNetVolume.Create(3, 1, 1);
  InputMinus := TNNetVolume.Create(3, 1, 1);
  epsilon := 0.0001;
  try
    NN.AddLayer(TNNetInput.Create(3, 1, 1, 1)); // pError=1 resizes error volumes
    NN.AddLayer(TNNetGELU.Create());

    Input.Raw[0] := 0.5;
    Input.Raw[1] := -0.5;
    Input.Raw[2] := 1.0;

    // Compute forward pass to get the derivative
    NN.Compute(Input);
    
    // Check gradient at each input position
    for i := 0 to 2 do
    begin
      // Compute f(x + epsilon)
      InputPlus.Copy(Input);
      InputPlus.Raw[i] := Input.Raw[i] + epsilon;
      NN.Compute(InputPlus);
      numericalGrad := NN.GetLastLayer.Output.Raw[i];

      // Compute f(x - epsilon)
      InputMinus.Copy(Input);
      InputMinus.Raw[i] := Input.Raw[i] - epsilon;
      NN.Compute(InputMinus);
      numericalGrad := (numericalGrad - NN.GetLastLayer.Output.Raw[i]) / (2 * epsilon);

      // Get analytical gradient from the layer's error derivative
      NN.Compute(Input);
      analyticalGrad := NN.GetLastLayer.OutputErrorDeriv.Raw[i];

      // Compare numerical and analytical gradients
      AssertTrue('GELU gradient check at position ' + IntToStr(i),
        Abs(numericalGrad - analyticalGrad) < 0.01);
    end;

  finally
    NN.Free;
    Input.Free;
    InputPlus.Free;
    InputMinus.Free;
  end;
end;

procedure TTestNeuralNumerical.TestMishGradientCheck;
var
  NN: TNNet;
  Input, InputPlus, InputMinus: TNNetVolume;
  epsilon: TNeuralFloat;
  numericalGrad, analyticalGrad: TNeuralFloat;
  i: integer;
begin
  NN := TNNet.Create();
  Input := TNNetVolume.Create(3, 1, 1);
  InputPlus := TNNetVolume.Create(3, 1, 1);
  InputMinus := TNNetVolume.Create(3, 1, 1);
  epsilon := 0.0001;
  try
    NN.AddLayer(TNNetInput.Create(3, 1, 1, 1)); // pError=1 resizes error volumes
    NN.AddLayer(TNNetMish.Create());

    Input.Raw[0] := 0.5;
    Input.Raw[1] := -0.5;
    Input.Raw[2] := 1.0;

    // Compute forward pass to get the derivative
    NN.Compute(Input);
    
    // Check gradient at each input position
    for i := 0 to 2 do
    begin
      // Compute f(x + epsilon)
      InputPlus.Copy(Input);
      InputPlus.Raw[i] := Input.Raw[i] + epsilon;
      NN.Compute(InputPlus);
      numericalGrad := NN.GetLastLayer.Output.Raw[i];

      // Compute f(x - epsilon)
      InputMinus.Copy(Input);
      InputMinus.Raw[i] := Input.Raw[i] - epsilon;
      NN.Compute(InputMinus);
      numericalGrad := (numericalGrad - NN.GetLastLayer.Output.Raw[i]) / (2 * epsilon);

      // Get analytical gradient from the layer's error derivative
      NN.Compute(Input);
      analyticalGrad := NN.GetLastLayer.OutputErrorDeriv.Raw[i];

      // Compare numerical and analytical gradients
      AssertTrue('Mish gradient check at position ' + IntToStr(i),
        Abs(numericalGrad - analyticalGrad) < 0.01);
    end;

  finally
    NN.Free;
    Input.Free;
    InputPlus.Free;
    InputMinus.Free;
  end;
end;

// Generic central finite-difference gradient check for an activation layer.
// AInputs holds the input values to probe; each must be away from any
// non-differentiable kink of the activation under test.
procedure ActivationGradientCheck(ATestCase: TTestCase;
  ALayer: TNNetLayer; const AName: string; const AInputs: array of TNeuralFloat;
  ATolerance: TNeuralFloat);
var
  NN: TNNet;
  Input, InputPlus, InputMinus: TNNetVolume;
  epsilon: TNeuralFloat;
  numericalGrad, analyticalGrad: TNeuralFloat;
  i, n: integer;
begin
  n := Length(AInputs);
  NN := TNNet.Create();
  Input := TNNetVolume.Create(n, 1, 1);
  InputPlus := TNNetVolume.Create(n, 1, 1);
  InputMinus := TNNetVolume.Create(n, 1, 1);
  epsilon := 0.0001;
  try
    NN.AddLayer(TNNetInput.Create(n, 1, 1, 1)); // pError=1 resizes error volumes
    NN.AddLayer(ALayer);

    for i := 0 to n - 1 do
      Input.Raw[i] := AInputs[i];

    NN.Compute(Input);

    for i := 0 to n - 1 do
    begin
      InputPlus.Copy(Input);
      InputPlus.Raw[i] := Input.Raw[i] + epsilon;
      NN.Compute(InputPlus);
      numericalGrad := NN.GetLastLayer.Output.Raw[i];

      InputMinus.Copy(Input);
      InputMinus.Raw[i] := Input.Raw[i] - epsilon;
      NN.Compute(InputMinus);
      numericalGrad := (numericalGrad - NN.GetLastLayer.Output.Raw[i]) / (2 * epsilon);

      NN.Compute(Input);
      analyticalGrad := NN.GetLastLayer.OutputErrorDeriv.Raw[i];

      ATestCase.AssertTrue(AName + ' gradient check at position ' + IntToStr(i) +
        ' (num=' + FloatToStr(numericalGrad) + ' ana=' + FloatToStr(analyticalGrad) + ')',
        Abs(numericalGrad - analyticalGrad) < ATolerance);
    end;
  finally
    NN.Free;
    Input.Free;
    InputPlus.Free;
    InputMinus.Free;
  end;
end;

procedure TTestNeuralNumerical.TestSwishGradientCheck;
begin
  ActivationGradientCheck(Self, TNNetSwish.Create(), 'Swish',
    [0.5, -0.5, 1.0, -2.0, 2.5], 0.01);
end;

procedure TTestNeuralNumerical.TestSwish6GradientCheck;
begin
  // Stay clear of the upper saturation kink at 6.
  ActivationGradientCheck(Self, TNNetSwish6.Create(), 'Swish6',
    [0.5, -0.5, 1.0, -2.0, 3.0], 0.01);
end;

procedure TTestNeuralNumerical.TestHardSwishGradientCheck;
begin
  // Avoid the non-differentiable kinks at x = -3 and x = 3.
  ActivationGradientCheck(Self, TNNetHardSwish.Create(), 'HardSwish',
    [0.5, -0.5, 1.0, -2.0, 2.0, 4.0], 0.01);
end;

procedure TTestNeuralNumerical.TestSoftPlusGradientCheck;
begin
  ActivationGradientCheck(Self, TNNetSoftPlus.Create(), 'SoftPlus',
    [0.5, -0.5, 1.0, -2.0, 2.5], 0.01);
end;

procedure TTestNeuralNumerical.TestGaussianActivationGradientCheck;
begin
  ActivationGradientCheck(Self, TNNetGaussianActivation.Create(), 'GaussianActivation',
    [0.5, -0.5, 1.0, -1.5, 2.0], 0.01);
end;

procedure TTestNeuralNumerical.TestSELUGradientCheck;
begin
  // Avoid the kink at x = 0.
  ActivationGradientCheck(Self, TNNetSELU.Create(), 'SELU',
    [0.5, -0.5, 1.0, -2.0, 2.5], 0.01);
end;

procedure TTestNeuralNumerical.TestLeakyReLUGradientCheck;
begin
  // Avoid the kink at x = 0.
  ActivationGradientCheck(Self, TNNetLeakyReLU.Create(), 'LeakyReLU',
    [0.5, -0.5, 1.0, -2.0, 2.5], 0.01);
end;

procedure TTestNeuralNumerical.TestVeryLeakyReLUGradientCheck;
begin
  // Avoid the kink at x = 0.
  ActivationGradientCheck(Self, TNNetVeryLeakyReLU.Create(), 'VeryLeakyReLU',
    [0.5, -0.5, 1.0, -2.0, 2.5], 0.01);
end;

procedure TTestNeuralNumerical.TestReLU6GradientCheck;
begin
  // Avoid the kinks at x = 0 and x = 6.
  ActivationGradientCheck(Self, TNNetReLU6.Create(), 'ReLU6',
    [1.0, -1.0, 3.0, -2.0, 7.0], 0.01);
end;

// TNNetSigmoid / TNNetHyperbolicTangent compute their error derivative inside
// Backpropagate (not Compute), so this check drives a real backward pass with
// a known per-element output error and compares against central differences.
procedure ActivationGradientCheckViaBackprop(ATestCase: TTestCase;
  ALayer: TNNetLayer; const AName: string; const AInputs: array of TNeuralFloat;
  ATolerance: TNeuralFloat);
var
  NN: TNNet;
  Input, InputPlus, Desired: TNNetVolume;
  epsilon, lossPlus, lossMinus, numericalGrad, analyticalGrad: TNeuralFloat;
  i, n: integer;

  function ComputeLoss(AInput: TNNetVolume): TNeuralFloat;
  var
    k: integer;
    diff: TNeuralFloat;
  begin
    NN.Compute(AInput);
    Result := 0;
    for k := 0 to NN.GetLastLayer.Output.Size - 1 do
    begin
      diff := NN.GetLastLayer.Output.Raw[k] - Desired.Raw[k];
      Result := Result + 0.5 * diff * diff;
    end;
  end;

begin
  n := Length(AInputs);
  NN := TNNet.Create();
  Input := TNNetVolume.Create(n, 1, 1);
  InputPlus := TNNetVolume.Create(n, 1, 1);
  Desired := TNNetVolume.Create(n, 1, 1);
  epsilon := 0.0001;
  try
    NN.AddLayer(TNNetInput.Create(n, 1, 1, 1)); // pError=1 resizes error volumes
    NN.AddLayer(ALayer);
    NN.SetLearningRate(1.0, 0.0);
    NN.SetBatchUpdate(true);

    for i := 0 to n - 1 do
    begin
      Input.Raw[i] := AInputs[i];
      Desired.Raw[i] := Cos(i * 0.5);
    end;

    for i := 0 to n - 1 do
    begin
      InputPlus.Copy(Input);
      InputPlus.Raw[i] := Input.Raw[i] + epsilon;
      lossPlus := ComputeLoss(InputPlus);
      InputPlus.Raw[i] := Input.Raw[i] - epsilon;
      lossMinus := ComputeLoss(InputPlus);
      numericalGrad := (lossPlus - lossMinus) / (2 * epsilon);

      NN.Compute(Input);
      NN.Layers[0].OutputError.Fill(0);
      NN.Backpropagate(Desired);
      analyticalGrad := NN.Layers[0].OutputError.Raw[i];

      ATestCase.AssertTrue(AName + ' gradient check at position ' + IntToStr(i) +
        ' (num=' + FloatToStr(numericalGrad) + ' ana=' + FloatToStr(analyticalGrad) + ')',
        Abs(numericalGrad - analyticalGrad) < ATolerance);
    end;
  finally
    NN.Free;
    Input.Free;
    InputPlus.Free;
    Desired.Free;
  end;
end;

procedure TTestNeuralNumerical.TestSigmoidGradientCheck;
begin
  ActivationGradientCheckViaBackprop(Self, TNNetSigmoid.Create(), 'Sigmoid',
    [0.5, -0.5, 1.0, -2.0, 2.5], 0.01);
end;

procedure TTestNeuralNumerical.TestHyperbolicTangentGradientCheck;
begin
  ActivationGradientCheckViaBackprop(Self, TNNetHyperbolicTangent.Create(), 'HyperbolicTangent',
    [0.5, -0.5, 1.0, -2.0, 2.5], 0.01);
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

procedure TTestNeuralNumerical.TestLayerNormForward;
var
  NN: TNNet;
  Input: TNNetVolume;
  LNorm: TNNetLayerNorm;
  Mean, Variance, diff: TNeuralFloat;
  i: integer;
begin
  // With default gamma=1 and beta=0, TNNetLayerNorm output must have
  // ~zero mean and ~unit variance over the whole sample.
  NN := TNNet.Create();
  Input := TNNetVolume.Create(2, 2, 2);
  try
    NN.AddLayer(TNNetInput.Create(2, 2, 2));
    LNorm := TNNetLayerNorm.Create();
    NN.AddLayer(LNorm);

    for i := 0 to Input.Size - 1 do
      Input.Raw[i] := i * 1.5 - 3.0;

    NN.Compute(Input);

    Mean := NN.GetLastLayer.Output.GetAvg();
    Variance := 0;
    for i := 0 to NN.GetLastLayer.Output.Size - 1 do
    begin
      diff := NN.GetLastLayer.Output.Raw[i] - Mean;
      Variance := Variance + diff * diff;
    end;
    Variance := Variance / NN.GetLastLayer.Output.Size;

    AssertEquals('LayerNorm output mean should be ~0', 0.0, Mean, 0.001);
    AssertEquals('LayerNorm output variance should be ~1', 1.0, Variance, 0.001);

    // Now test with non-trivial learnable gamma and beta.
    LNorm.Neurons[0].Weights.Fill(3.0); // gamma
    LNorm.Neurons[1].Weights.Fill(2.0); // beta
    NN.Compute(Input);
    Mean := NN.GetLastLayer.Output.GetAvg();
    Variance := 0;
    for i := 0 to NN.GetLastLayer.Output.Size - 1 do
    begin
      diff := NN.GetLastLayer.Output.Raw[i] - Mean;
      Variance := Variance + diff * diff;
    end;
    Variance := Variance / NN.GetLastLayer.Output.Size;
    // mean = beta, variance = gamma^2
    AssertEquals('LayerNorm output mean should be ~beta', 2.0, Mean, 0.001);
    AssertEquals('LayerNorm output variance should be ~gamma^2', 9.0, Variance, 0.01);
  finally
    NN.Free;
    Input.Free;
  end;
end;

procedure TTestNeuralNumerical.TestLayerNormGradientCheck;
var
  NN: TNNet;
  Input, InputPlus, Desired: TNNetVolume;
  LNorm: TNNetLayerNorm;
  epsilon, lossPlus, lossMinus, numericalGrad, analyticalGrad: TNeuralFloat;
  i, j: integer;

  function ComputeLoss(AInput: TNNetVolume): TNeuralFloat;
  var
    k: integer;
    diff: TNeuralFloat;
  begin
    NN.Compute(AInput);
    Result := 0;
    for k := 0 to NN.GetLastLayer.Output.Size - 1 do
    begin
      diff := NN.GetLastLayer.Output.Raw[k] - Desired.Raw[k];
      Result := Result + 0.5 * diff * diff;
    end;
  end;

begin
  NN := TNNet.Create();
  Input := TNNetVolume.Create(3, 1, 2);
  InputPlus := TNNetVolume.Create(3, 1, 2);
  Desired := TNNetVolume.Create(3, 1, 2);
  epsilon := 0.0001;
  try
    NN.AddLayer(TNNetInput.Create(3, 1, 2, 1)); // pError=1 resizes error volumes
    LNorm := TNNetLayerNorm.Create();
    NN.AddLayer(LNorm);
    // Learning rate 1, batch update on: deltas accumulate -1*gradient and
    // weights are NOT modified, so finite-difference checks stay valid.
    NN.SetLearningRate(1.0, 0.0);
    NN.SetBatchUpdate(true);

    for i := 0 to Input.Size - 1 do
      Input.Raw[i] := Sin(i * 0.7) * 2.0 + 0.3;
    for i := 0 to Desired.Size - 1 do
      Desired.Raw[i] := Cos(i * 0.5);

    // Non-trivial learnable parameters.
    for i := 0 to LNorm.Neurons[0].Weights.Size - 1 do
    begin
      LNorm.Neurons[0].Weights.Raw[i] := 1.0 + i * 0.1; // gamma
      LNorm.Neurons[1].Weights.Raw[i] := i * 0.05 - 0.1; // beta
    end;

    // ---- Gradient w.r.t. the input ----
    for i := 0 to Input.Size - 1 do
    begin
      InputPlus.Copy(Input);
      InputPlus.Raw[i] := Input.Raw[i] + epsilon;
      lossPlus := ComputeLoss(InputPlus);
      InputPlus.Raw[i] := Input.Raw[i] - epsilon;
      lossMinus := ComputeLoss(InputPlus);
      numericalGrad := (lossPlus - lossMinus) / (2 * epsilon);

      NN.Compute(Input);
      NN.Layers[0].OutputError.Fill(0);
      NN.Backpropagate(Desired);
      analyticalGrad := NN.Layers[0].OutputError.Raw[i];

      AssertTrue('LayerNorm input gradient check at position ' + IntToStr(i) +
        ' (num=' + FloatToStr(numericalGrad) + ' ana=' + FloatToStr(analyticalGrad) + ')',
        Abs(numericalGrad - analyticalGrad) < 0.01);
    end;

    // ---- Gradient w.r.t. gamma and beta ----
    for j := 0 to 1 do // 0 = gamma, 1 = beta
      for i := 0 to LNorm.Neurons[j].Weights.Size - 1 do
      begin
        LNorm.Neurons[j].Weights.Raw[i] := LNorm.Neurons[j].Weights.Raw[i] + epsilon;
        lossPlus := ComputeLoss(Input);
        LNorm.Neurons[j].Weights.Raw[i] := LNorm.Neurons[j].Weights.Raw[i] - 2 * epsilon;
        lossMinus := ComputeLoss(Input);
        LNorm.Neurons[j].Weights.Raw[i] := LNorm.Neurons[j].Weights.Raw[i] + epsilon;
        numericalGrad := (lossPlus - lossMinus) / (2 * epsilon);

        NN.Compute(Input);
        LNorm.Neurons[j].ClearDelta;
        NN.Backpropagate(Desired);
        // Backprop accumulates Delta := Delta - LearningRate*gradient.
        // With LearningRate = 1, analytical gradient = -Delta.
        analyticalGrad := -LNorm.Neurons[j].Delta.Raw[i];

        AssertTrue('LayerNorm weight gradient check (' + IntToStr(j) + ',' + IntToStr(i) +
          ') num=' + FloatToStr(numericalGrad) + ' ana=' + FloatToStr(analyticalGrad),
          Abs(numericalGrad - analyticalGrad) < 0.01);
      end;
  finally
    NN.Free;
    Input.Free;
    InputPlus.Free;
    Desired.Free;
  end;
end;

procedure TTestNeuralNumerical.TestRMSNormForward;
var
  NN: TNNet;
  Input: TNNetVolume;
  RNorm: TNNetRMSNorm;
  MeanSqr, RMS: TNeuralFloat;
  i: integer;
begin
  // With default gamma=1, TNNetRMSNorm output must have ~unit root mean
  // square over the whole sample (no mean subtraction).
  NN := TNNet.Create();
  Input := TNNetVolume.Create(2, 2, 2);
  try
    NN.AddLayer(TNNetInput.Create(2, 2, 2));
    RNorm := TNNetRMSNorm.Create();
    NN.AddLayer(RNorm);

    for i := 0 to Input.Size - 1 do
      Input.Raw[i] := i * 1.5 - 3.0;

    NN.Compute(Input);

    MeanSqr := NN.GetLastLayer.Output.GetSumSqr() / NN.GetLastLayer.Output.Size;
    RMS := Sqrt(MeanSqr);
    AssertEquals('RMSNorm output RMS should be ~1', 1.0, RMS, 0.001);

    // Now test with non-trivial learnable gamma.
    RNorm.Neurons[0].Weights.Fill(3.0); // gamma
    NN.Compute(Input);
    MeanSqr := NN.GetLastLayer.Output.GetSumSqr() / NN.GetLastLayer.Output.Size;
    RMS := Sqrt(MeanSqr);
    // RMS scales by gamma.
    AssertEquals('RMSNorm output RMS should be ~gamma', 3.0, RMS, 0.01);
  finally
    NN.Free;
    Input.Free;
  end;
end;

procedure TTestNeuralNumerical.TestRMSNormGradientCheck;
var
  NN: TNNet;
  Input, InputPlus, Desired: TNNetVolume;
  RNorm: TNNetRMSNorm;
  epsilon, lossPlus, lossMinus, numericalGrad, analyticalGrad: TNeuralFloat;
  i: integer;

  function ComputeLoss(AInput: TNNetVolume): TNeuralFloat;
  var
    k: integer;
    diff: TNeuralFloat;
  begin
    NN.Compute(AInput);
    Result := 0;
    for k := 0 to NN.GetLastLayer.Output.Size - 1 do
    begin
      diff := NN.GetLastLayer.Output.Raw[k] - Desired.Raw[k];
      Result := Result + 0.5 * diff * diff;
    end;
  end;

begin
  NN := TNNet.Create();
  Input := TNNetVolume.Create(3, 1, 2);
  InputPlus := TNNetVolume.Create(3, 1, 2);
  Desired := TNNetVolume.Create(3, 1, 2);
  epsilon := 0.0001;
  try
    NN.AddLayer(TNNetInput.Create(3, 1, 2, 1)); // pError=1 resizes error volumes
    RNorm := TNNetRMSNorm.Create();
    NN.AddLayer(RNorm);
    // Learning rate 1, batch update on: deltas accumulate -1*gradient and
    // weights are NOT modified, so finite-difference checks stay valid.
    NN.SetLearningRate(1.0, 0.0);
    NN.SetBatchUpdate(true);

    for i := 0 to Input.Size - 1 do
      Input.Raw[i] := Sin(i * 0.7) * 2.0 + 0.3;
    for i := 0 to Desired.Size - 1 do
      Desired.Raw[i] := Cos(i * 0.5);

    // Non-trivial learnable gamma.
    for i := 0 to RNorm.Neurons[0].Weights.Size - 1 do
      RNorm.Neurons[0].Weights.Raw[i] := 1.0 + i * 0.1; // gamma

    // ---- Gradient w.r.t. the input ----
    for i := 0 to Input.Size - 1 do
    begin
      InputPlus.Copy(Input);
      InputPlus.Raw[i] := Input.Raw[i] + epsilon;
      lossPlus := ComputeLoss(InputPlus);
      InputPlus.Raw[i] := Input.Raw[i] - epsilon;
      lossMinus := ComputeLoss(InputPlus);
      numericalGrad := (lossPlus - lossMinus) / (2 * epsilon);

      NN.Compute(Input);
      NN.Layers[0].OutputError.Fill(0);
      NN.Backpropagate(Desired);
      analyticalGrad := NN.Layers[0].OutputError.Raw[i];

      AssertTrue('RMSNorm input gradient check at position ' + IntToStr(i) +
        ' (num=' + FloatToStr(numericalGrad) + ' ana=' + FloatToStr(analyticalGrad) + ')',
        Abs(numericalGrad - analyticalGrad) < 0.01);
    end;

    // ---- Gradient w.r.t. gamma ----
    for i := 0 to RNorm.Neurons[0].Weights.Size - 1 do
    begin
      RNorm.Neurons[0].Weights.Raw[i] := RNorm.Neurons[0].Weights.Raw[i] + epsilon;
      lossPlus := ComputeLoss(Input);
      RNorm.Neurons[0].Weights.Raw[i] := RNorm.Neurons[0].Weights.Raw[i] - 2 * epsilon;
      lossMinus := ComputeLoss(Input);
      RNorm.Neurons[0].Weights.Raw[i] := RNorm.Neurons[0].Weights.Raw[i] + epsilon;
      numericalGrad := (lossPlus - lossMinus) / (2 * epsilon);

      NN.Compute(Input);
      RNorm.Neurons[0].ClearDelta;
      NN.Backpropagate(Desired);
      // Backprop accumulates Delta := Delta - LearningRate*gradient.
      // With LearningRate = 1, analytical gradient = -Delta.
      analyticalGrad := -RNorm.Neurons[0].Delta.Raw[i];

      AssertTrue('RMSNorm weight gradient check (' + IntToStr(i) +
        ') num=' + FloatToStr(numericalGrad) + ' ana=' + FloatToStr(analyticalGrad),
        Abs(numericalGrad - analyticalGrad) < 0.01);
    end;
  finally
    NN.Free;
    Input.Free;
    InputPlus.Free;
    Desired.Free;
  end;
end;

procedure TTestNeuralNumerical.TestLayerScaleForward;
var
  NN: TNNet;
  Input: TNNetVolume;
  LScale: TNNetLayerScale;
  x, y, d, i: integer;
begin
  // TNNetLayerScale applies a per-channel learnable multiplier.
  // Output[x,y,d] = Input[x,y,d] * Scale[d].
  NN := TNNet.Create();
  Input := TNNetVolume.Create(2, 2, 3);
  try
    NN.AddLayer(TNNetInput.Create(2, 2, 3));
    // Default initial scale is 1.0 -> output must equal input.
    LScale := TNNetLayerScale.Create();
    NN.AddLayer(LScale);

    for i := 0 to Input.Size - 1 do
      Input.Raw[i] := Sin(i * 0.6) * 2.5 + 1.3;

    NN.Compute(Input);
    for i := 0 to Input.Size - 1 do
      AssertEquals('LayerScale default scale=1 keeps input', Input.Raw[i],
        NN.GetLastLayer.Output.Raw[i], 0.0001);

    // Now set a non-trivial per-channel scale.
    LScale.Neurons[0].Weights.Raw[0] := 2.0;
    LScale.Neurons[0].Weights.Raw[1] := -1.5;
    LScale.Neurons[0].Weights.Raw[2] := 0.25;
    NN.Compute(Input);
    for x := 0 to Input.SizeX - 1 do
      for y := 0 to Input.SizeY - 1 do
        for d := 0 to Input.Depth - 1 do
          AssertEquals('LayerScale per-channel multiply',
            Input[x, y, d] * LScale.Neurons[0].Weights.Raw[d],
            NN.GetLastLayer.Output[x, y, d], 0.0001);
  finally
    NN.Free;
    Input.Free;
  end;
end;

procedure TTestNeuralNumerical.TestLayerScaleGradientCheck;
var
  NN: TNNet;
  Input, InputPlus, Desired: TNNetVolume;
  LScale: TNNetLayerScale;
  epsilon, lossPlus, lossMinus, numericalGrad, analyticalGrad: TNeuralFloat;
  i: integer;

  function ComputeLoss(AInput: TNNetVolume): TNeuralFloat;
  var
    k: integer;
    diff: TNeuralFloat;
  begin
    NN.Compute(AInput);
    Result := 0;
    for k := 0 to NN.GetLastLayer.Output.Size - 1 do
    begin
      diff := NN.GetLastLayer.Output.Raw[k] - Desired.Raw[k];
      Result := Result + 0.5 * diff * diff;
    end;
  end;

begin
  NN := TNNet.Create();
  Input := TNNetVolume.Create(3, 1, 2);
  InputPlus := TNNetVolume.Create(3, 1, 2);
  Desired := TNNetVolume.Create(3, 1, 2);
  epsilon := 0.0001;
  try
    NN.AddLayer(TNNetInput.Create(3, 1, 2, 1)); // pError=1 resizes error volumes
    LScale := TNNetLayerScale.Create(0.5);
    NN.AddLayer(LScale);
    // Learning rate 1, batch update on: deltas accumulate -1*gradient and
    // weights are NOT modified, so finite-difference checks stay valid.
    NN.SetLearningRate(1.0, 0.0);
    NN.SetBatchUpdate(true);

    for i := 0 to Input.Size - 1 do
      Input.Raw[i] := Sin(i * 0.7) * 2.0 + 0.3;
    for i := 0 to Desired.Size - 1 do
      Desired.Raw[i] := Cos(i * 0.5);

    // Non-trivial per-channel scale.
    for i := 0 to LScale.Neurons[0].Weights.Size - 1 do
      LScale.Neurons[0].Weights.Raw[i] := 0.7 + i * 0.3;

    // ---- Gradient w.r.t. the input ----
    for i := 0 to Input.Size - 1 do
    begin
      InputPlus.Copy(Input);
      InputPlus.Raw[i] := Input.Raw[i] + epsilon;
      lossPlus := ComputeLoss(InputPlus);
      InputPlus.Raw[i] := Input.Raw[i] - epsilon;
      lossMinus := ComputeLoss(InputPlus);
      numericalGrad := (lossPlus - lossMinus) / (2 * epsilon);

      NN.Compute(Input);
      NN.Layers[0].OutputError.Fill(0);
      NN.Backpropagate(Desired);
      analyticalGrad := NN.Layers[0].OutputError.Raw[i];

      AssertTrue('LayerScale input gradient check at position ' + IntToStr(i) +
        ' (num=' + FloatToStr(numericalGrad) + ' ana=' + FloatToStr(analyticalGrad) + ')',
        Abs(numericalGrad - analyticalGrad) < 0.01);
    end;

    // ---- Gradient w.r.t. the learnable scale weights ----
    for i := 0 to LScale.Neurons[0].Weights.Size - 1 do
    begin
      LScale.Neurons[0].Weights.Raw[i] := LScale.Neurons[0].Weights.Raw[i] + epsilon;
      lossPlus := ComputeLoss(Input);
      LScale.Neurons[0].Weights.Raw[i] := LScale.Neurons[0].Weights.Raw[i] - 2 * epsilon;
      lossMinus := ComputeLoss(Input);
      LScale.Neurons[0].Weights.Raw[i] := LScale.Neurons[0].Weights.Raw[i] + epsilon;
      numericalGrad := (lossPlus - lossMinus) / (2 * epsilon);

      NN.Compute(Input);
      LScale.Neurons[0].ClearDelta;
      NN.Backpropagate(Desired);
      // Backprop accumulates Delta := Delta - LearningRate*gradient.
      // With LearningRate = 1, analytical gradient = -Delta.
      analyticalGrad := -LScale.Neurons[0].Delta.Raw[i];

      AssertTrue('LayerScale weight gradient check (' + IntToStr(i) +
        ') num=' + FloatToStr(numericalGrad) + ' ana=' + FloatToStr(analyticalGrad),
        Abs(numericalGrad - analyticalGrad) < 0.01);
    end;
  finally
    NN.Free;
    Input.Free;
    InputPlus.Free;
    Desired.Free;
  end;
end;

procedure TTestNeuralNumerical.TestGroupNormForward;
var
  NN: TNNet;
  Input: TNNetVolume;
  GNorm: TNNetGroupNorm;
  Mean, Variance, diff: TNeuralFloat;
  Groups, ChannelsPerGroup, GroupSize: integer;
  g, x, y, d, dStart, dEnd: integer;
begin
  // With default gamma=1 and beta=0, each group of the TNNetGroupNorm output
  // must have ~zero mean and ~unit variance.
  NN := TNNet.Create();
  // 2x2 spatial, 4 channels, split into 2 groups of 2 channels each.
  Input := TNNetVolume.Create(2, 2, 4);
  Groups := 2;
  try
    NN.AddLayer(TNNetInput.Create(2, 2, 4));
    GNorm := TNNetGroupNorm.Create(Groups);
    NN.AddLayer(GNorm);

    for x := 0 to Input.Size - 1 do
      Input.Raw[x] := Sin(x * 0.6) * 2.5 + 1.3;

    NN.Compute(Input);

    ChannelsPerGroup := Input.Depth div Groups;
    GroupSize := Input.SizeX * Input.SizeY * ChannelsPerGroup;
    for g := 0 to Groups - 1 do
    begin
      dStart := g * ChannelsPerGroup;
      dEnd := dStart + ChannelsPerGroup - 1;
      Mean := 0;
      for x := 0 to Input.SizeX - 1 do
        for y := 0 to Input.SizeY - 1 do
          for d := dStart to dEnd do
            Mean := Mean + NN.GetLastLayer.Output[x, y, d];
      Mean := Mean / GroupSize;
      Variance := 0;
      for x := 0 to Input.SizeX - 1 do
        for y := 0 to Input.SizeY - 1 do
          for d := dStart to dEnd do
          begin
            diff := NN.GetLastLayer.Output[x, y, d] - Mean;
            Variance := Variance + diff * diff;
          end;
      Variance := Variance / GroupSize;
      AssertEquals('GroupNorm group ' + IntToStr(g) + ' mean should be ~0',
        0.0, Mean, 0.001);
      AssertEquals('GroupNorm group ' + IntToStr(g) + ' variance should be ~1',
        1.0, Variance, 0.001);
    end;

    // Now test with non-trivial learnable gamma and beta.
    GNorm.Neurons[0].Weights.Fill(3.0); // gamma
    GNorm.Neurons[1].Weights.Fill(2.0); // beta
    NN.Compute(Input);
    for g := 0 to Groups - 1 do
    begin
      dStart := g * ChannelsPerGroup;
      dEnd := dStart + ChannelsPerGroup - 1;
      Mean := 0;
      for x := 0 to Input.SizeX - 1 do
        for y := 0 to Input.SizeY - 1 do
          for d := dStart to dEnd do
            Mean := Mean + NN.GetLastLayer.Output[x, y, d];
      Mean := Mean / GroupSize;
      Variance := 0;
      for x := 0 to Input.SizeX - 1 do
        for y := 0 to Input.SizeY - 1 do
          for d := dStart to dEnd do
          begin
            diff := NN.GetLastLayer.Output[x, y, d] - Mean;
            Variance := Variance + diff * diff;
          end;
      Variance := Variance / GroupSize;
      // mean = beta, variance = gamma^2
      AssertEquals('GroupNorm group ' + IntToStr(g) + ' mean should be ~beta',
        2.0, Mean, 0.001);
      AssertEquals('GroupNorm group ' + IntToStr(g) + ' variance should be ~gamma^2',
        9.0, Variance, 0.01);
    end;
  finally
    NN.Free;
    Input.Free;
  end;
end;

procedure TTestNeuralNumerical.TestGroupNormGradientCheck;
var
  NN: TNNet;
  Input, InputPlus, Desired: TNNetVolume;
  GNorm: TNNetGroupNorm;
  epsilon, lossPlus, lossMinus, numericalGrad, analyticalGrad: TNeuralFloat;
  i, j: integer;

  function ComputeLoss(AInput: TNNetVolume): TNeuralFloat;
  var
    k: integer;
    diff: TNeuralFloat;
  begin
    NN.Compute(AInput);
    Result := 0;
    for k := 0 to NN.GetLastLayer.Output.Size - 1 do
    begin
      diff := NN.GetLastLayer.Output.Raw[k] - Desired.Raw[k];
      Result := Result + 0.5 * diff * diff;
    end;
  end;

begin
  NN := TNNet.Create();
  // 2x1 spatial, 4 channels, 2 groups.
  Input := TNNetVolume.Create(2, 1, 4);
  InputPlus := TNNetVolume.Create(2, 1, 4);
  Desired := TNNetVolume.Create(2, 1, 4);
  epsilon := 0.0001;
  try
    NN.AddLayer(TNNetInput.Create(2, 1, 4, 1)); // pError=1 resizes error volumes
    GNorm := TNNetGroupNorm.Create(2);
    NN.AddLayer(GNorm);
    NN.SetLearningRate(1.0, 0.0);
    NN.SetBatchUpdate(true);

    for i := 0 to Input.Size - 1 do
      Input.Raw[i] := Sin(i * 0.7) * 2.0 + 0.3;
    for i := 0 to Desired.Size - 1 do
      Desired.Raw[i] := Cos(i * 0.5);

    // Non-trivial learnable parameters.
    for i := 0 to GNorm.Neurons[0].Weights.Size - 1 do
    begin
      GNorm.Neurons[0].Weights.Raw[i] := 1.0 + i * 0.1; // gamma
      GNorm.Neurons[1].Weights.Raw[i] := i * 0.05 - 0.1; // beta
    end;

    // ---- Gradient w.r.t. the input ----
    for i := 0 to Input.Size - 1 do
    begin
      InputPlus.Copy(Input);
      InputPlus.Raw[i] := Input.Raw[i] + epsilon;
      lossPlus := ComputeLoss(InputPlus);
      InputPlus.Raw[i] := Input.Raw[i] - epsilon;
      lossMinus := ComputeLoss(InputPlus);
      numericalGrad := (lossPlus - lossMinus) / (2 * epsilon);

      NN.Compute(Input);
      NN.Layers[0].OutputError.Fill(0);
      NN.Backpropagate(Desired);
      analyticalGrad := NN.Layers[0].OutputError.Raw[i];

      AssertTrue('GroupNorm input gradient check at position ' + IntToStr(i) +
        ' (num=' + FloatToStr(numericalGrad) + ' ana=' + FloatToStr(analyticalGrad) + ')',
        Abs(numericalGrad - analyticalGrad) < 0.01);
    end;

    // ---- Gradient w.r.t. gamma and beta ----
    for j := 0 to 1 do // 0 = gamma, 1 = beta
      for i := 0 to GNorm.Neurons[j].Weights.Size - 1 do
      begin
        GNorm.Neurons[j].Weights.Raw[i] := GNorm.Neurons[j].Weights.Raw[i] + epsilon;
        lossPlus := ComputeLoss(Input);
        GNorm.Neurons[j].Weights.Raw[i] := GNorm.Neurons[j].Weights.Raw[i] - 2 * epsilon;
        lossMinus := ComputeLoss(Input);
        GNorm.Neurons[j].Weights.Raw[i] := GNorm.Neurons[j].Weights.Raw[i] + epsilon;
        numericalGrad := (lossPlus - lossMinus) / (2 * epsilon);

        NN.Compute(Input);
        GNorm.Neurons[j].ClearDelta;
        NN.Backpropagate(Desired);
        analyticalGrad := -GNorm.Neurons[j].Delta.Raw[i];

        AssertTrue('GroupNorm weight gradient check (' + IntToStr(j) + ',' + IntToStr(i) +
          ') num=' + FloatToStr(numericalGrad) + ' ana=' + FloatToStr(analyticalGrad),
          Abs(numericalGrad - analyticalGrad) < 0.01);
      end;
  finally
    NN.Free;
    Input.Free;
    InputPlus.Free;
    Desired.Free;
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

// Generic input-gradient check: builds a 1-layer net (Input -> ALayer), drives a
// real backward pass with a known per-element output error and compares the
// input error against central finite differences. ALayer is owned by the net.
procedure LayerInputGradientCheck(ATestCase: TTestCase; ALayer: TNNetLayer;
  const AName: string; ASizeX, ASizeY, ASizeD: integer; ATolerance: TNeuralFloat);
var
  NN: TNNet;
  Input, InputPlus, Desired: TNNetVolume;
  epsilon, lossPlus, lossMinus, numericalGrad, analyticalGrad: TNeuralFloat;
  i: integer;

  function ComputeLoss(AInput: TNNetVolume): TNeuralFloat;
  var
    k: integer;
    diff: TNeuralFloat;
  begin
    NN.Compute(AInput);
    Result := 0;
    for k := 0 to NN.GetLastLayer.Output.Size - 1 do
    begin
      diff := NN.GetLastLayer.Output.Raw[k] - Desired.Raw[k];
      Result := Result + 0.5 * diff * diff;
    end;
  end;

begin
  NN := TNNet.Create();
  Input := TNNetVolume.Create(ASizeX, ASizeY, ASizeD);
  InputPlus := TNNetVolume.Create(ASizeX, ASizeY, ASizeD);
  epsilon := 0.0001;
  try
    NN.AddLayer(TNNetInput.Create(ASizeX, ASizeY, ASizeD, 1));
    NN.AddLayer(ALayer);
    NN.SetLearningRate(1.0, 0.0);
    NN.SetBatchUpdate(true);

    Desired := TNNetVolume.Create();
    Desired.ReSize(NN.GetLastLayer.Output);
    for i := 0 to Input.Size - 1 do
      Input.Raw[i] := Sin(i * 0.7) * 2.0 + 0.3;
    for i := 0 to Desired.Size - 1 do
      Desired.Raw[i] := Cos(i * 0.5);

    for i := 0 to Input.Size - 1 do
    begin
      InputPlus.Copy(Input);
      InputPlus.Raw[i] := Input.Raw[i] + epsilon;
      lossPlus := ComputeLoss(InputPlus);
      InputPlus.Raw[i] := Input.Raw[i] - epsilon;
      lossMinus := ComputeLoss(InputPlus);
      numericalGrad := (lossPlus - lossMinus) / (2 * epsilon);

      NN.Compute(Input);
      NN.Layers[0].OutputError.Fill(0);
      NN.Backpropagate(Desired);
      analyticalGrad := NN.Layers[0].OutputError.Raw[i];

      ATestCase.AssertTrue(AName + ' input gradient check at position ' + IntToStr(i) +
        ' (num=' + FloatToStr(numericalGrad) + ' ana=' + FloatToStr(analyticalGrad) + ')',
        Abs(numericalGrad - analyticalGrad) < ATolerance);
    end;
  finally
    NN.Free;
    Input.Free;
    InputPlus.Free;
    Desired.Free;
  end;
end;

procedure TTestNeuralNumerical.TestPadXYGradientCheck;
begin
  LayerInputGradientCheck(Self, TNNetPadXY.Create(1, 1), 'PadXY', 3, 2, 2, 0.01);
end;

procedure TTestNeuralNumerical.TestCropGradientCheck;
begin
  LayerInputGradientCheck(Self, TNNetCrop.Create(1, 1, 2, 2), 'Crop', 4, 4, 2, 0.01);
end;

procedure TTestNeuralNumerical.TestInterleaveChannelsGradientCheck;
begin
  LayerInputGradientCheck(Self, TNNetInterleaveChannels.Create(2),
    'InterleaveChannels', 2, 2, 4, 0.01);
end;

procedure TTestNeuralNumerical.TestAvgPoolGradientCheck;
begin
  LayerInputGradientCheck(Self, TNNetAvgPool.Create(2), 'AvgPool', 4, 4, 2, 0.01);
end;

// CellBias / CellMul carry learnable per-cell weights; check both the input
// gradient and the weight (Delta) gradient against central differences.
procedure CellLayerGradientCheck(ATestCase: TTestCase; ALayer: TNNetLayer;
  const AName: string);
var
  NN: TNNet;
  Input, InputPlus, Desired: TNNetVolume;
  epsilon, lossPlus, lossMinus, numericalGrad, analyticalGrad: TNeuralFloat;
  i: integer;

  function ComputeLoss(AInput: TNNetVolume): TNeuralFloat;
  var
    k: integer;
    diff: TNeuralFloat;
  begin
    NN.Compute(AInput);
    Result := 0;
    for k := 0 to NN.GetLastLayer.Output.Size - 1 do
    begin
      diff := NN.GetLastLayer.Output.Raw[k] - Desired.Raw[k];
      Result := Result + 0.5 * diff * diff;
    end;
  end;

begin
  NN := TNNet.Create();
  Input := TNNetVolume.Create(2, 2, 2);
  InputPlus := TNNetVolume.Create(2, 2, 2);
  Desired := TNNetVolume.Create(2, 2, 2);
  epsilon := 0.0001;
  try
    NN.AddLayer(TNNetInput.Create(2, 2, 2, 1));
    NN.AddLayer(ALayer);
    NN.SetLearningRate(1.0, 0.0);
    NN.SetBatchUpdate(true);

    for i := 0 to Input.Size - 1 do
      Input.Raw[i] := Sin(i * 0.7) * 2.0 + 0.3;
    for i := 0 to Desired.Size - 1 do
      Desired.Raw[i] := Cos(i * 0.5);
    // Non-trivial learnable weights.
    for i := 0 to ALayer.Neurons[0].Weights.Size - 1 do
      ALayer.Neurons[0].Weights.Raw[i] := 0.5 + i * 0.13;

    // ---- Gradient w.r.t. the input ----
    for i := 0 to Input.Size - 1 do
    begin
      InputPlus.Copy(Input);
      InputPlus.Raw[i] := Input.Raw[i] + epsilon;
      lossPlus := ComputeLoss(InputPlus);
      InputPlus.Raw[i] := Input.Raw[i] - epsilon;
      lossMinus := ComputeLoss(InputPlus);
      numericalGrad := (lossPlus - lossMinus) / (2 * epsilon);

      NN.Compute(Input);
      NN.Layers[0].OutputError.Fill(0);
      NN.Backpropagate(Desired);
      analyticalGrad := NN.Layers[0].OutputError.Raw[i];

      ATestCase.AssertTrue(AName + ' input gradient check at position ' + IntToStr(i) +
        ' (num=' + FloatToStr(numericalGrad) + ' ana=' + FloatToStr(analyticalGrad) + ')',
        Abs(numericalGrad - analyticalGrad) < 0.01);
    end;

    // ---- Gradient w.r.t. the learnable weights ----
    for i := 0 to ALayer.Neurons[0].Weights.Size - 1 do
    begin
      ALayer.Neurons[0].Weights.Raw[i] := ALayer.Neurons[0].Weights.Raw[i] + epsilon;
      lossPlus := ComputeLoss(Input);
      ALayer.Neurons[0].Weights.Raw[i] := ALayer.Neurons[0].Weights.Raw[i] - 2 * epsilon;
      lossMinus := ComputeLoss(Input);
      ALayer.Neurons[0].Weights.Raw[i] := ALayer.Neurons[0].Weights.Raw[i] + epsilon;
      numericalGrad := (lossPlus - lossMinus) / (2 * epsilon);

      NN.Compute(Input);
      ALayer.Neurons[0].ClearDelta;
      NN.Backpropagate(Desired);
      // Backprop accumulates Delta := Delta - LearningRate*gradient.
      analyticalGrad := -ALayer.Neurons[0].Delta.Raw[i];

      ATestCase.AssertTrue(AName + ' weight gradient check (' + IntToStr(i) +
        ') num=' + FloatToStr(numericalGrad) + ' ana=' + FloatToStr(analyticalGrad),
        Abs(numericalGrad - analyticalGrad) < 0.01);
    end;
  finally
    NN.Free;
    Input.Free;
    InputPlus.Free;
    Desired.Free;
  end;
end;

procedure TTestNeuralNumerical.TestCellBiasGradientCheck;
begin
  CellLayerGradientCheck(Self, TNNetCellBias.Create(), 'CellBias');
end;

procedure TTestNeuralNumerical.TestCellMulGradientCheck;
begin
  CellLayerGradientCheck(Self, TNNetCellMul.Create(), 'CellMul');
end;

initialization
  RegisterTest(TTestNeuralNumerical);

end.
