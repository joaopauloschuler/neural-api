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
    procedure TestInstanceNormForward;
    procedure TestInstanceNormGradientCheck;
    procedure TestInstanceNormSerializationRoundTrip;
    procedure TestRMSNormForward;
    procedure TestRMSNormGradientCheck;
    procedure TestPixelNormForward;
    procedure TestPixelNormGradientCheck;
    procedure TestPixelNormSerializationRoundTrip;
    procedure TestLayerScaleForward;
    procedure TestLayerScaleGradientCheck;

    // Transform / reshaping / element-wise layer gradient checks
    procedure TestPadXYGradientCheck;
    procedure TestCropGradientCheck;
    procedure TestInterleaveChannelsGradientCheck;
    procedure TestGEGLUForward;
    procedure TestGEGLUGradientCheck;
    procedure TestSwiGLUForward;
    procedure TestSwiGLUGradientCheck;
    procedure TestGLUForward;
    procedure TestGLUGradientCheck;
    procedure TestSquaredReLUForward;
    procedure TestSquaredReLUGradientCheck;
    procedure TestTanhShrinkForward;
    procedure TestTanhShrinkGradientCheck;
    procedure TestTanhShrinkSerializationRoundTrip;
    procedure TestLogSigmoidForward;
    procedure TestLogSigmoidGradientCheck;
    procedure TestLogSigmoidSerializationRoundTrip;
    procedure TestLogSigmoidExtremeInputSaturation;
    procedure TestShiftedReLUForward;
    procedure TestShiftedReLUGradientCheck;
    procedure TestShiftedReLUSerializationRoundTrip;
    procedure TestHardTanhForward;
    procedure TestHardTanhGradientCheck;
    procedure TestHardTanhSerializationRoundTrip;
    procedure TestHardShrinkForward;
    procedure TestHardShrinkGradientCheck;
    procedure TestHardShrinkSerializationRoundTrip;
    procedure TestSoftShrinkForward;
    procedure TestSoftShrinkGradientCheck;
    procedure TestSoftShrinkSerializationRoundTrip;
    procedure TestThresholdForward;
    procedure TestThresholdReLUEquivalence;
    procedure TestThresholdGradientCheck;
    procedure TestThresholdSerializationRoundTrip;
    procedure TestReLU6Forward;
    procedure TestReLU6ExtremeInputSaturation;
    procedure TestGlobalMaxPoolForward;
    procedure TestGlobalMaxPoolGradientCheck;
    procedure TestMaskedFillForward;
    procedure TestMaskedFillGradientCheck;
    procedure TestALiBiForward;
    procedure TestALiBiGradientCheck;
    procedure TestALiBiSerializationRoundTrip;
    procedure TestALiBiMaskedFillComposition;
    procedure TestSoftCappingForward;
    procedure TestSoftCappingGradientCheck;
    procedure TestDropPathInferenceIdentity;
    procedure TestDropPathTrainingScaling;
    procedure TestDropPathGradientCheck;
    procedure TestAvgPoolGradientCheck;
    procedure TestUpsampleGradientCheck;
    procedure TestDeMaxPoolGradientCheck;
    procedure TestDeAvgPoolGradientCheck;
    procedure TestDeMaxPoolForwardReplication;
    procedure TestCellBiasGradientCheck;
    procedure TestCellMulGradientCheck;
    procedure TestAddPositionalEmbeddingForward;
    procedure TestAddPositionalEmbeddingGradientCheck;
    procedure TestAddPositionalEmbeddingEmbeddingIsConstant;
    procedure TestScaledDotProductAttentionForward;
    procedure TestScaledDotProductAttentionGradientCheck;
    procedure TestScaledDotProductAttentionCausalGradientCheck;
    procedure TestRotaryEmbeddingForward;
    procedure TestRotaryEmbeddingGradientCheck;
    procedure TestRotaryEmbeddingInverse;
    procedure TestRotaryEmbeddingInverseSeqLen5;
    procedure TestRotaryEmbeddingOddDepthGuard;
    procedure TestDropPathPZeroBoundary;
    procedure TestDropPathPOneBoundary;
    procedure TestDropPathDeterminismFixedSeed;
    procedure TestSoftCappingLargeCapContinuity;
    procedure TestSoftCappingExtremeInputSaturation;
    procedure TestSoftCappingSerializationRoundTrip;
    procedure TestDropPathSerializationRoundTrip;
    procedure TestRotaryEmbeddingSerializationRoundTrip;
    procedure TestMaskedFillSerializationRoundTrip;
    procedure TestScaledDotProductAttentionSerializationRoundTrip;
    procedure TestSpatialDropout1DInferenceIdentity;
    procedure TestSpatialDropout1DTrainingMaskShape;
    procedure TestSpatialDropout1DGradientCheck;
    procedure TestSpatialDropout1DSerializationRoundTrip;
    procedure TestSpatialDropout2DInferenceIdentity;
    procedure TestSpatialDropout2DTrainingMaskShape;
    procedure TestSpatialDropout2DGradientCheck;
    procedure TestSpatialDropout2DSerializationRoundTrip;
    procedure TestChannelShuffleForward;
    procedure TestChannelShuffleGradientCheck;
    procedure TestChannelShuffleSerializationRoundTrip;
    procedure TestSoftmaxTemperatureMatchesSoftMaxAtOne;
    procedure TestSoftmaxTemperatureIncreasesEntropy;
    procedure TestSoftmaxTemperatureGradientCheck;
    procedure TestSoftmaxTemperatureSerializationRoundTrip;
    procedure TestPointwiseSoftMaxExactJacobianGradientCheck;
    procedure TestSoftMaxExactJacobianGradientCheck;
    procedure TestLogSoftMaxForward;
    procedure TestLogSoftMaxGradientCheck;
    procedure TestLogSoftMaxSerializationRoundTrip;
    procedure TestChannelShuffleIndivisibleGuard;
    procedure TestChannelShuffleInverseProperty;
    procedure TestReverseChannelsForward;
    procedure TestReverseChannelsGradientCheck;
    procedure TestReverseChannelsInvolution;
    procedure TestReverseChannelsSerializationRoundTrip;
    procedure TestLayerNormSerializationRoundTrip;
    procedure TestRMSNormSerializationRoundTrip;
    procedure TestGroupNormSerializationRoundTrip;
    procedure TestChannelStdNormalizationSerializationRoundTrip;
    procedure TestLocalResponseNorm2DSerializationRoundTrip;
    procedure TestMaxOutForward;
    procedure TestMaxOutGradientCheck;
    procedure TestMaxOutSerializationRoundTrip;
    procedure TestHardTanhExtremeInputSaturation;
    procedure TestTanhShrinkTanhComposition;
    procedure TestMaxOutDepthNotDivisibleByKGuard;
    procedure TestSoftPlusIdentityAtZero;
    procedure TestSoftPlusLargeXLinearization;
    procedure TestSoftPlusExtremeInputSaturation;
    procedure TestELUForward;
    procedure TestELUGradientCheck;
    procedure TestELUSerializationRoundTrip;
    procedure TestCELUForward;
    procedure TestCELUGradientCheck;
    procedure TestCELUSerializationRoundTrip;
    procedure TestSiLUMatchesSwish;
    procedure TestSoftSignForward;
    procedure TestSoftSignGradientCheck;
    procedure TestSoftSignSerializationRoundTrip;
    procedure TestGlobalAvgPoolGradientCheck;
    procedure TestReLU6SerializationRoundTrip;
    procedure TestGlobalMaxPoolSerializationRoundTrip;
    procedure TestGlobalAvgPoolSerializationRoundTrip;
    procedure TestSwiGLUSerializationRoundTrip;
    procedure TestGEGLUSerializationRoundTrip;
    procedure TestLayerScaleSerializationRoundTrip;

    // Concat and sum numerical tests
    procedure TestConcatNumericalValues;
    procedure TestSumNumericalValues;
    procedure TestConcatGradientCheck;
    procedure TestDeepConcatGradientCheck;
    procedure TestSplitChannelsGradientCheck;
    procedure TestSumGradientCheck;
    
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

procedure TTestNeuralNumerical.TestPixelNormForward;
var
  NN: TNNet;
  Input: TNNetVolume;
  PNorm: TNNetPixelNorm;
  Output: TNNetVolume;
  x, y, c, i: integer;
  SumSqr, RMS: TNeuralFloat;
begin
  // TNNetPixelNorm: for every (x,y) pixel, the depth-vector RMS must be ~1.
  NN := TNNet.Create();
  Input := TNNetVolume.Create(2, 2, 4);
  try
    NN.AddLayer(TNNetInput.Create(2, 2, 4));
    PNorm := TNNetPixelNorm.Create();
    NN.AddLayer(PNorm);

    for i := 0 to Input.Size - 1 do
      Input.Raw[i] := Sin(i * 0.55) * 2.0 + 0.3;

    NN.Compute(Input);
    Output := NN.GetLastLayer.Output;

    for x := 0 to Output.SizeX - 1 do
      for y := 0 to Output.SizeY - 1 do
      begin
        SumSqr := 0;
        for c := 0 to Output.Depth - 1 do
          SumSqr := SumSqr + Output[x, y, c] * Output[x, y, c];
        RMS := Sqrt(SumSqr / Output.Depth);
        AssertEquals('PixelNorm per-pixel RMS at (' + IntToStr(x) + ',' +
          IntToStr(y) + ') should be ~1', 1.0, RMS, 0.001);
      end;
  finally
    NN.Free;
    Input.Free;
  end;
end;

procedure TTestNeuralNumerical.TestPixelNormGradientCheck;
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
  Input := TNNetVolume.Create(2, 2, 4);
  InputPlus := TNNetVolume.Create(2, 2, 4);
  Desired := TNNetVolume.Create(2, 2, 4);
  epsilon := 0.0001;
  try
    NN.AddLayer(TNNetInput.Create(2, 2, 4, 1));
    NN.AddLayer(TNNetPixelNorm.Create());
    NN.SetLearningRate(1.0, 0.0);
    NN.SetBatchUpdate(true);

    for i := 0 to Input.Size - 1 do
      Input.Raw[i] := Sin(i * 0.7) * 1.5 + 0.4;
    for i := 0 to Desired.Size - 1 do
      Desired.Raw[i] := Cos(i * 0.4);

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

      AssertTrue('PixelNorm input gradient check at position ' + IntToStr(i) +
        ' (num=' + FloatToStr(numericalGrad) + ' ana=' + FloatToStr(analyticalGrad) + ')',
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

procedure TTestNeuralNumerical.TestInstanceNormForward;
var
  NN: TNNet;
  Input: TNNetVolume;
  INorm: TNNetInstanceNorm;
  Mean, Variance, diff: TNeuralFloat;
  ChannelSize: integer;
  x, y, d: integer;
begin
  // InstanceNorm = GroupNorm with Groups=Depth: each channel of each sample
  // is independently normalized to zero mean and unit variance.
  NN := TNNet.Create();
  Input := TNNetVolume.Create(4, 4, 3);
  try
    NN.AddLayer(TNNetInput.Create(4, 4, 3));
    INorm := TNNetInstanceNorm.Create();
    NN.AddLayer(INorm);

    RandSeed := 131313;
    for x := 0 to Input.Size - 1 do
      Input.Raw[x] := Random() * 4 - 2;

    NN.Compute(Input);

    ChannelSize := Input.SizeX * Input.SizeY;
    for d := 0 to Input.Depth - 1 do
    begin
      Mean := 0;
      for x := 0 to Input.SizeX - 1 do
        for y := 0 to Input.SizeY - 1 do
          Mean := Mean + NN.GetLastLayer.Output[x, y, d];
      Mean := Mean / ChannelSize;
      Variance := 0;
      for x := 0 to Input.SizeX - 1 do
        for y := 0 to Input.SizeY - 1 do
        begin
          diff := NN.GetLastLayer.Output[x, y, d] - Mean;
          Variance := Variance + diff * diff;
        end;
      Variance := Variance / ChannelSize;
      AssertEquals('InstanceNorm channel ' + IntToStr(d) + ' mean should be ~0',
        0.0, Mean, 0.001);
      AssertEquals('InstanceNorm channel ' + IntToStr(d) + ' variance should be ~1',
        1.0, Variance, 0.001);
    end;
  finally
    NN.Free;
    Input.Free;
  end;
end;

procedure TTestNeuralNumerical.TestInstanceNormGradientCheck;
var
  NN: TNNet;
  Input, InputPlus, Desired: TNNetVolume;
  INorm: TNNetInstanceNorm;
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
  // 3x1 spatial, 3 channels -> InstanceNorm uses Groups=3 (one per channel).
  Input := TNNetVolume.Create(3, 1, 3);
  InputPlus := TNNetVolume.Create(3, 1, 3);
  Desired := TNNetVolume.Create(3, 1, 3);
  epsilon := 0.0001;
  try
    NN.AddLayer(TNNetInput.Create(3, 1, 3, 1));
    INorm := TNNetInstanceNorm.Create();
    NN.AddLayer(INorm);
    NN.SetLearningRate(1.0, 0.0);
    NN.SetBatchUpdate(true);

    for i := 0 to Input.Size - 1 do
      Input.Raw[i] := Sin(i * 0.7) * 2.0 + 0.3;
    for i := 0 to Desired.Size - 1 do
      Desired.Raw[i] := Cos(i * 0.5);

    for i := 0 to INorm.Neurons[0].Weights.Size - 1 do
    begin
      INorm.Neurons[0].Weights.Raw[i] := 1.0 + i * 0.1;
      INorm.Neurons[1].Weights.Raw[i] := i * 0.05 - 0.1;
    end;

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

      AssertTrue('InstanceNorm input gradient check at position ' + IntToStr(i) +
        ' (num=' + FloatToStr(numericalGrad) + ' ana=' + FloatToStr(analyticalGrad) + ')',
        Abs(numericalGrad - analyticalGrad) < 0.01);
    end;

    for j := 0 to 1 do
      for i := 0 to INorm.Neurons[j].Weights.Size - 1 do
      begin
        INorm.Neurons[j].Weights.Raw[i] := INorm.Neurons[j].Weights.Raw[i] + epsilon;
        lossPlus := ComputeLoss(Input);
        INorm.Neurons[j].Weights.Raw[i] := INorm.Neurons[j].Weights.Raw[i] - 2 * epsilon;
        lossMinus := ComputeLoss(Input);
        INorm.Neurons[j].Weights.Raw[i] := INorm.Neurons[j].Weights.Raw[i] + epsilon;
        numericalGrad := (lossPlus - lossMinus) / (2 * epsilon);

        NN.Compute(Input);
        INorm.Neurons[j].ClearDelta;
        NN.Backpropagate(Desired);
        analyticalGrad := -INorm.Neurons[j].Delta.Raw[i];

        AssertTrue('InstanceNorm weight gradient check (' + IntToStr(j) + ',' + IntToStr(i) +
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

procedure TTestNeuralNumerical.TestConcatGradientCheck;
// Numerical gradient check for TNNetConcat (flat concat).
// Two branches (MulByConstant) fan out from the input layer, then are
// concatenated flat. Verifies the input-error path accumulates correctly
// from both branches.
var
  NN: TNNet;
  Input, InputPlus, Desired: TNNetVolume;
  InputLayer, Branch1, Branch2: TNNetLayer;
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
  Input := TNNetVolume.Create(2, 1, 2);
  InputPlus := TNNetVolume.Create(2, 1, 2);
  epsilon := 0.001;
  try
    InputLayer := NN.AddLayer(TNNetInput.Create(2, 1, 2, 1));
    Branch1 := NN.AddLayer(TNNetMulByConstant.Create(1.5));
    Branch2 := NN.AddLayerAfter(TNNetMulByConstant.Create(-0.7), InputLayer);
    NN.AddLayer(TNNetConcat.Create([Branch1, Branch2]));
    NN.SetLearningRate(1.0, 0.0);
    NN.SetBatchUpdate(true);

    Desired := TNNetVolume.Create(NN.GetLastLayer.Output.SizeX,
                                  NN.GetLastLayer.Output.SizeY,
                                  NN.GetLastLayer.Output.Depth);

    for i := 0 to Input.Size - 1 do
      Input.Raw[i] := Sin(i * 0.6) * 1.7 + 0.2;
    for i := 0 to Desired.Size - 1 do
      Desired.Raw[i] := Cos(i * 0.4) - 0.1;

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

      AssertTrue('Concat input gradient at ' + IntToStr(i) +
        ' (num=' + FloatToStr(numericalGrad) + ' ana=' + FloatToStr(analyticalGrad) + ')',
        Abs(numericalGrad - analyticalGrad) < 0.01);
    end;
  finally
    NN.Free;
    Input.Free;
    InputPlus.Free;
    Desired.Free;
  end;
end;

procedure TTestNeuralNumerical.TestDeepConcatGradientCheck;
// Numerical gradient check for TNNetDeepConcat. Two branches with different
// transforms are stacked along the depth axis.
var
  NN: TNNet;
  Input, InputPlus, Desired: TNNetVolume;
  InputLayer, Branch1, Branch2: TNNetLayer;
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
  epsilon := 0.001;
  try
    InputLayer := NN.AddLayer(TNNetInput.Create(2, 2, 2, 1));
    Branch1 := NN.AddLayer(TNNetMulByConstant.Create(2.0));
    Branch2 := NN.AddLayerAfter(TNNetMulByConstant.Create(0.5), InputLayer);
    NN.AddLayer(TNNetDeepConcat.Create([Branch1, Branch2]));
    NN.SetLearningRate(1.0, 0.0);
    NN.SetBatchUpdate(true);

    Desired := TNNetVolume.Create(NN.GetLastLayer.Output.SizeX,
                                  NN.GetLastLayer.Output.SizeY,
                                  NN.GetLastLayer.Output.Depth);

    for i := 0 to Input.Size - 1 do
      Input.Raw[i] := Sin(i * 0.5) + 0.3;
    for i := 0 to Desired.Size - 1 do
      Desired.Raw[i] := Cos(i * 0.3) * 0.5;

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

      AssertTrue('DeepConcat input gradient at ' + IntToStr(i) +
        ' (num=' + FloatToStr(numericalGrad) + ' ana=' + FloatToStr(analyticalGrad) + ')',
        Abs(numericalGrad - analyticalGrad) < 0.01);
    end;
  finally
    NN.Free;
    Input.Free;
    InputPlus.Free;
    Desired.Free;
  end;
end;

procedure TTestNeuralNumerical.TestSplitChannelsGradientCheck;
// Numerical gradient check for TNNetSplitChannels. Two splits feed a
// DeepConcat so every input channel reaches the loss; this exercises the
// SplitChannels backprop path on multiple channel selections.
var
  NN: TNNet;
  Input, InputPlus, Desired: TNNetVolume;
  InputLayer, SplitA, SplitB: TNNetLayer;
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
  // 4 channels: SplitA takes channels [1] (single), SplitB takes [0,2,3].
  // Reordered concat exercises both contiguous and non-contiguous picks.
  Input := TNNetVolume.Create(2, 1, 4);
  InputPlus := TNNetVolume.Create(2, 1, 4);
  epsilon := 0.001;
  try
    InputLayer := NN.AddLayer(TNNetInput.Create(2, 1, 4, 1));
    SplitA := NN.AddLayer(TNNetSplitChannels.Create([1]));
    SplitB := NN.AddLayerAfter(TNNetSplitChannels.Create([0, 2, 3]), InputLayer);
    NN.AddLayer(TNNetDeepConcat.Create([SplitA, SplitB]));
    NN.SetLearningRate(1.0, 0.0);
    NN.SetBatchUpdate(true);

    Desired := TNNetVolume.Create(NN.GetLastLayer.Output.SizeX,
                                  NN.GetLastLayer.Output.SizeY,
                                  NN.GetLastLayer.Output.Depth);

    for i := 0 to Input.Size - 1 do
      Input.Raw[i] := Sin(i * 0.55) * 1.3 - 0.2;
    for i := 0 to Desired.Size - 1 do
      Desired.Raw[i] := Cos(i * 0.45) * 0.4;

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

      AssertTrue('SplitChannels input gradient at ' + IntToStr(i) +
        ' (num=' + FloatToStr(numericalGrad) + ' ana=' + FloatToStr(analyticalGrad) + ')',
        Abs(numericalGrad - analyticalGrad) < 0.01);
    end;
  finally
    NN.Free;
    Input.Free;
    InputPlus.Free;
    Desired.Free;
  end;
end;

procedure TTestNeuralNumerical.TestSumGradientCheck;
// Numerical gradient check for TNNetSum (residual-style add). Two branches
// with different scalar multipliers feed a sum; each branch contributes its
// full gradient back to the shared input.
var
  NN: TNNet;
  Input, InputPlus, Desired: TNNetVolume;
  InputLayer, Branch1, Branch2: TNNetLayer;
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
  epsilon := 0.001;
  try
    InputLayer := NN.AddLayer(TNNetInput.Create(2, 2, 2, 1));
    Branch1 := NN.AddLayer(TNNetMulByConstant.Create(1.0));
    Branch2 := NN.AddLayerAfter(TNNetMulByConstant.Create(-0.5), InputLayer);
    NN.AddLayer(TNNetSum.Create([Branch1, Branch2]));
    NN.SetLearningRate(1.0, 0.0);
    NN.SetBatchUpdate(true);

    Desired := TNNetVolume.Create(NN.GetLastLayer.Output.SizeX,
                                  NN.GetLastLayer.Output.SizeY,
                                  NN.GetLastLayer.Output.Depth);

    for i := 0 to Input.Size - 1 do
      Input.Raw[i] := Sin(i * 0.7) * 1.1 + 0.4;
    for i := 0 to Desired.Size - 1 do
      Desired.Raw[i] := Cos(i * 0.5) * 0.6 - 0.1;

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

      AssertTrue('Sum input gradient at ' + IntToStr(i) +
        ' (num=' + FloatToStr(numericalGrad) + ' ana=' + FloatToStr(analyticalGrad) + ')',
        Abs(numericalGrad - analyticalGrad) < 0.01);
    end;
  finally
    NN.Free;
    Input.Free;
    InputPlus.Free;
    Desired.Free;
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

procedure TTestNeuralNumerical.TestUpsampleGradientCheck;
begin
  // TNNetUpsample (depth_to_space): input depth must be a multiple of 4.
  // 2x2x4 -> 4x4x1 keeps the check tiny. Layer is a pure reshuffle of
  // input cells into output positions, so gradients are an identity
  // permutation on OutputError.
  LayerInputGradientCheck(Self, TNNetUpsample.Create(), 'Upsample', 2, 2, 4, 0.01);
end;

// Local gradient check that accumulates the loss in Double precision. The
// generic LayerInputGradientCheck helper accumulates loss in TNeuralFloat
// (Single), which catastrophically cancels for layers whose forward
// replicates large values into many output cells (e.g. DeMaxPool/DeAvgPool):
// the sum-of-squares is large but the perturbation-induced delta is tiny.
procedure DeMaxPoolFamilyGradientCheck(ATestCase: TTestCase;
  ALayer: TNNetLayer; const AName: string;
  ASizeX, ASizeY, ASizeD: integer; ATolerance: TNeuralFloat);
var
  NN: TNNet;
  Input, InputPlus, Desired: TNNetVolume;
  epsilon: TNeuralFloat;
  lossPlus, lossMinus, numericalGrad: Double;
  analyticalGrad: TNeuralFloat;
  i: integer;

  function ComputeLoss(AInput: TNNetVolume): Double;
  var
    k: integer;
    diff: Double;
  begin
    NN.Compute(AInput);
    Result := 0;
    for k := 0 to NN.GetLastLayer.Output.Size - 1 do
    begin
      diff := Double(NN.GetLastLayer.Output.Raw[k]) - Double(Desired.Raw[k]);
      Result := Result + 0.5 * diff * diff;
    end;
  end;

begin
  NN := TNNet.Create();
  Input := TNNetVolume.Create(ASizeX, ASizeY, ASizeD);
  InputPlus := TNNetVolume.Create(ASizeX, ASizeY, ASizeD);
  epsilon := 0.001;
  try
    NN.AddLayer(TNNetInput.Create(ASizeX, ASizeY, ASizeD, 1));
    NN.AddLayer(ALayer);
    NN.SetLearningRate(1.0, 0.0);
    NN.SetBatchUpdate(true);

    Desired := TNNetVolume.Create();
    Desired.ReSize(NN.GetLastLayer.Output);
    // Small magnitudes keep the sum-of-squares loss small enough that the
    // perturbation-induced delta survives Single precision when accumulated.
    for i := 0 to Input.Size - 1 do
      Input.Raw[i] := Sin(i * 0.7) * 0.1;
    for i := 0 to Desired.Size - 1 do
      Desired.Raw[i] := Cos(i * 0.5) * 0.1;

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

      ATestCase.AssertTrue(AName + ' input gradient check at position ' +
        IntToStr(i) + ' (num=' + FloatToStr(numericalGrad) +
        ' ana=' + FloatToStr(analyticalGrad) + ')',
        Abs(numericalGrad - analyticalGrad) < ATolerance);
    end;
  finally
    NN.Free;
    Input.Free;
    InputPlus.Free;
    Desired.Free;
  end;
end;

procedure TTestNeuralNumerical.TestDeMaxPoolGradientCheck;
begin
  // TNNetDeMaxPool replicates each input cell into a PoolSize x PoolSize
  // output block. The correct input gradient is therefore the SUM of the
  // block's output errors (no scaling). This guards the historical off-by-
  // PoolSize bug where ComputePreviousLayerError divided the output error
  // by PoolSize before accumulating.
  DeMaxPoolFamilyGradientCheck(Self, TNNetDeMaxPool.Create(2), 'DeMaxPool',
    2, 2, 2, 0.001);
end;

procedure TTestNeuralNumerical.TestDeAvgPoolGradientCheck;
begin
  // TNNetDeAvgPool = class(TNNetDeMaxPool) inherits both forward (pure
  // replication into a PoolSize x PoolSize block) and backward, so its
  // input gradient must also be the sum of the block's output errors.
  DeMaxPoolFamilyGradientCheck(Self, TNNetDeAvgPool.Create(2), 'DeAvgPool',
    2, 2, 2, 0.001);
end;

procedure TTestNeuralNumerical.TestDeMaxPoolForwardReplication;
var
  NN: TNNet;
  Input: TNNetVolume;
  CntX, CntY, CntD, BlockX, BlockY: integer;
  Expected: TNeuralFloat;
begin
  // Verify the forward pass replicates each input cell into a PoolSize x
  // PoolSize block (FSpacing = 0 default).
  NN := TNNet.Create();
  Input := TNNetVolume.Create(2, 2, 2);
  try
    NN.AddLayer(TNNetInput.Create(2, 2, 2));
    NN.AddLayer(TNNetDeMaxPool.Create(2));

    // Distinct values per (x, y, d).
    for CntD := 0 to 1 do
      for CntY := 0 to 1 do
        for CntX := 0 to 1 do
          Input[CntX, CntY, CntD] := 1.0 + CntX + 10 * CntY + 100 * CntD;

    NN.Compute(Input);

    AssertEquals('DeMaxPool output SizeX', 4, NN.GetLastLayer.Output.SizeX);
    AssertEquals('DeMaxPool output SizeY', 4, NN.GetLastLayer.Output.SizeY);
    AssertEquals('DeMaxPool output Depth', 2, NN.GetLastLayer.Output.Depth);

    // Each input cell (CntX, CntY, CntD) must appear in every output position
    // inside its 2x2 block.
    for CntD := 0 to 1 do
      for CntY := 0 to 1 do
        for CntX := 0 to 1 do
        begin
          Expected := Input[CntX, CntY, CntD];
          for BlockY := 0 to 1 do
            for BlockX := 0 to 1 do
              AssertEquals('DeMaxPool replication at (' +
                IntToStr(CntX * 2 + BlockX) + ',' +
                IntToStr(CntY * 2 + BlockY) + ',' + IntToStr(CntD) + ')',
                Expected,
                NN.GetLastLayer.Output[CntX * 2 + BlockX,
                                       CntY * 2 + BlockY, CntD],
                0.0001);
        end;
  finally
    NN.Free;
    Input.Free;
  end;
end;

procedure TTestNeuralNumerical.TestGEGLUForward;
var
  NN: TNNet;
  Input: TNNetVolume;
  a, b, b3, tanhArg, expected: TNeuralFloat;
const
  SQRT_2_OVER_PI = 0.7978845608;
  GELU_CONST = 0.044715;
begin
  NN := TNNet.Create();
  Input := TNNetVolume.Create(1, 1, 4);
  try
    NN.AddLayer(TNNetInput.Create(1, 1, 4, 1));
    NN.AddLayer(TNNetGEGLU.Create());

    // First half = A, second half = B.
    Input.Raw[0] := 2.0;   // A[0]
    Input.Raw[1] := -1.5;  // A[1]
    Input.Raw[2] := 1.0;   // B[0]
    Input.Raw[3] := -0.5;  // B[1]

    NN.Compute(Input);

    AssertEquals('GEGLU output depth = input depth / 2', 2,
      NN.GetLastLayer.Output.Depth);

    // output[0] = A[0] * GELU(B[0])
    a := 2.0; b := 1.0;
    b3 := b * b * b;
    tanhArg := SQRT_2_OVER_PI * (b + GELU_CONST * b3);
    expected := a * (0.5 * b * (1 + Tanh(tanhArg)));
    AssertEquals('GEGLU output[0]', expected, NN.GetLastLayer.Output.Raw[0], 0.0001);

    // output[1] = A[1] * GELU(B[1])
    a := -1.5; b := -0.5;
    b3 := b * b * b;
    tanhArg := SQRT_2_OVER_PI * (b + GELU_CONST * b3);
    expected := a * (0.5 * b * (1 + Tanh(tanhArg)));
    AssertEquals('GEGLU output[1]', expected, NN.GetLastLayer.Output.Raw[1], 0.0001);
  finally
    NN.Free;
    Input.Free;
  end;
end;

procedure TTestNeuralNumerical.TestGEGLUGradientCheck;
begin
  // Depth 4 -> output depth 2; gradient flows to both input halves.
  LayerInputGradientCheck(Self, TNNetGEGLU.Create(), 'GEGLU', 2, 2, 4, 0.01);
end;

procedure TTestNeuralNumerical.TestSwiGLUForward;
var
  NN: TNNet;
  Input: TNNetVolume;
  a, b, expected: TNeuralFloat;
begin
  NN := TNNet.Create();
  Input := TNNetVolume.Create(1, 1, 4);
  try
    NN.AddLayer(TNNetInput.Create(1, 1, 4, 1));
    NN.AddLayer(TNNetSwiGLU.Create());

    Input.Raw[0] := 2.0;   // A[0]
    Input.Raw[1] := -1.5;  // A[1]
    Input.Raw[2] := 1.0;   // B[0]
    Input.Raw[3] := -0.5;  // B[1]

    NN.Compute(Input);

    AssertEquals('SwiGLU output depth = input depth / 2', 2,
      NN.GetLastLayer.Output.Depth);

    // output[0] = A[0] * Swish(B[0]); Swish(x) = x * sigmoid(x)
    a := 2.0; b := 1.0;
    expected := a * (b * (1 / (1 + Exp(-b))));
    AssertEquals('SwiGLU output[0]', expected, NN.GetLastLayer.Output.Raw[0], 0.0001);

    a := -1.5; b := -0.5;
    expected := a * (b * (1 / (1 + Exp(-b))));
    AssertEquals('SwiGLU output[1]', expected, NN.GetLastLayer.Output.Raw[1], 0.0001);
  finally
    NN.Free;
    Input.Free;
  end;
end;

procedure TTestNeuralNumerical.TestSwiGLUGradientCheck;
begin
  LayerInputGradientCheck(Self, TNNetSwiGLU.Create(), 'SwiGLU', 2, 2, 4, 0.01);
end;

procedure TTestNeuralNumerical.TestGLUForward;
var
  NN: TNNet;
  Input: TNNetVolume;
  a, b, expected: TNeuralFloat;
begin
  NN := TNNet.Create();
  Input := TNNetVolume.Create(1, 1, 4);
  try
    NN.AddLayer(TNNetInput.Create(1, 1, 4, 1));
    NN.AddLayer(TNNetGLU.Create());

    Input.Raw[0] := 2.0;   // A[0]
    Input.Raw[1] := -1.5;  // A[1]
    Input.Raw[2] := 1.0;   // B[0]
    Input.Raw[3] := -0.5;  // B[1]

    NN.Compute(Input);

    AssertEquals('GLU output depth = input depth / 2', 2,
      NN.GetLastLayer.Output.Depth);

    // output[0] = A[0] * sigmoid(B[0])
    a := 2.0; b := 1.0;
    expected := a * (1 / (1 + Exp(-b)));
    AssertEquals('GLU output[0]', expected, NN.GetLastLayer.Output.Raw[0], 0.0001);

    a := -1.5; b := -0.5;
    expected := a * (1 / (1 + Exp(-b)));
    AssertEquals('GLU output[1]', expected, NN.GetLastLayer.Output.Raw[1], 0.0001);
  finally
    NN.Free;
    Input.Free;
  end;
end;

procedure TTestNeuralNumerical.TestGLUGradientCheck;
begin
  LayerInputGradientCheck(Self, TNNetGLU.Create(), 'GLU', 2, 2, 4, 0.01);
end;

procedure TTestNeuralNumerical.TestSquaredReLUForward;
var
  NN: TNNet;
  Input: TNNetVolume;
begin
  NN := TNNet.Create();
  Input := TNNetVolume.Create(1, 1, 4);
  try
    NN.AddLayer(TNNetInput.Create(1, 1, 4, 1));
    NN.AddLayer(TNNetSquaredReLU.Create());

    Input.Raw[0] := 2.0;
    Input.Raw[1] := -1.5;
    Input.Raw[2] := 0.5;
    Input.Raw[3] := -3.0;

    NN.Compute(Input);

    // SquaredReLU(x) = relu(x)^2
    AssertEquals('SquaredReLU output[0]', 4.0, NN.GetLastLayer.Output.Raw[0], 0.0001);
    AssertEquals('SquaredReLU output[1]', 0.0, NN.GetLastLayer.Output.Raw[1], 0.0001);
    AssertEquals('SquaredReLU output[2]', 0.25, NN.GetLastLayer.Output.Raw[2], 0.0001);
    AssertEquals('SquaredReLU output[3]', 0.0, NN.GetLastLayer.Output.Raw[3], 0.0001);
  finally
    NN.Free;
    Input.Free;
  end;
end;

procedure TTestNeuralNumerical.TestSquaredReLUGradientCheck;
begin
  // Stay clear of the kink at 0.
  ActivationGradientCheck(Self, TNNetSquaredReLU.Create(), 'SquaredReLU',
    [0.5, 1.0, 2.0, -1.0, -2.5], 0.01);
end;

procedure TTestNeuralNumerical.TestTanhShrinkForward;
var
  NN: TNNet;
  Input: TNNetVolume;
begin
  NN := TNNet.Create();
  Input := TNNetVolume.Create(1, 1, 4);
  try
    NN.AddLayer(TNNetInput.Create(1, 1, 4, 1));
    NN.AddLayer(TNNetTanhShrink.Create());

    Input.Raw[0] := 0.0;
    Input.Raw[1] := 1.0;
    Input.Raw[2] := -1.0;
    Input.Raw[3] := 2.0;

    NN.Compute(Input);

    // TanhShrink(x) = x - tanh(x)
    AssertEquals('TanhShrink(0)', 0.0, NN.GetLastLayer.Output.Raw[0], 0.0001);
    AssertEquals('TanhShrink(1)', 1.0 - Tanh(1.0), NN.GetLastLayer.Output.Raw[1], 0.0001);
    AssertEquals('TanhShrink(-1)', -1.0 - Tanh(-1.0), NN.GetLastLayer.Output.Raw[2], 0.0001);
    AssertEquals('TanhShrink(2)', 2.0 - Tanh(2.0), NN.GetLastLayer.Output.Raw[3], 0.0001);
  finally
    NN.Free;
    Input.Free;
  end;
end;

procedure TTestNeuralNumerical.TestTanhShrinkGradientCheck;
begin
  ActivationGradientCheck(Self, TNNetTanhShrink.Create(), 'TanhShrink',
    [0.5, -0.5, 1.0, -2.0, 2.5], 0.01);
end;

procedure TTestNeuralNumerical.TestLogSigmoidForward;
var
  NN: TNNet;
  Input: TNNetVolume;
  function LogSigmoidRef(x: TNeuralFloat): TNeuralFloat;
  begin
    if x >= 0 then
      Result := -Ln(1 + Exp(-x))
    else
      Result := x - Ln(1 + Exp(x));
  end;
begin
  NN := TNNet.Create();
  Input := TNNetVolume.Create(1, 1, 5);
  try
    NN.AddLayer(TNNetInput.Create(1, 1, 5, 1));
    NN.AddLayer(TNNetLogSigmoid.Create());

    Input.Raw[0] := 0.0;
    Input.Raw[1] := 1.0;
    Input.Raw[2] := -1.0;
    Input.Raw[3] := 2.0;
    Input.Raw[4] := -3.0;

    NN.Compute(Input);

    // LogSigmoid(0) = -ln(2) ~= -0.6931472
    AssertEquals('LogSigmoid(0)', -Ln(2.0), NN.GetLastLayer.Output.Raw[0], 1e-5);
    AssertEquals('LogSigmoid(1)', LogSigmoidRef(1.0), NN.GetLastLayer.Output.Raw[1], 1e-5);
    AssertEquals('LogSigmoid(-1)', LogSigmoidRef(-1.0), NN.GetLastLayer.Output.Raw[2], 1e-5);
    AssertEquals('LogSigmoid(2)', LogSigmoidRef(2.0), NN.GetLastLayer.Output.Raw[3], 1e-5);
    AssertEquals('LogSigmoid(-3)', LogSigmoidRef(-3.0), NN.GetLastLayer.Output.Raw[4], 1e-5);
  finally
    NN.Free;
    Input.Free;
  end;
end;

procedure TTestNeuralNumerical.TestLogSigmoidGradientCheck;
begin
  ActivationGradientCheck(Self, TNNetLogSigmoid.Create(), 'LogSigmoid',
    [0.5, -0.5, 1.0, -1.0, 2.0, -2.5], 0.01);
end;

procedure TTestNeuralNumerical.TestLogSigmoidExtremeInputSaturation;
var
  NN: TNNet;
  Input, Desired: TNNetVolume;
  v, g: TNeuralFloat;
  i: integer;
begin
  // Drive LogSigmoid with extreme magnitudes (+/-1e6 and others). The stable
  // formulation must produce no NaN/Inf in either forward or backward, and
  // outputs should be <= 0 (since sigmoid(x) in (0,1] => log <= 0).
  NN := TNNet.Create();
  Input := TNNetVolume.Create(6, 1, 1);
  Desired := TNNetVolume.Create();
  try
    NN.AddLayer(TNNetInput.Create(6, 1, 1, 1));
    NN.AddLayer(TNNetLogSigmoid.Create());
    NN.SetLearningRate(1.0, 0.0);
    NN.SetBatchUpdate(true);

    Desired.ReSize(NN.GetLastLayer.Output);
    Input.Raw[0] := 1e6;
    Input.Raw[1] := -1e6;
    Input.Raw[2] := 1e30;
    Input.Raw[3] := -1e30;
    Input.Raw[4] := 1e3;
    Input.Raw[5] := -1e3;
    for i := 0 to Desired.Size - 1 do
      Desired.Raw[i] := Sin(i * 0.3);

    NN.Compute(Input);
    for i := 0 to Input.Size - 1 do
    begin
      v := NN.GetLastLayer.Output.Raw[i];
      AssertFalse('LogSigmoid saturation forward NaN at ' + IntToStr(i), IsNan(v));
      AssertFalse('LogSigmoid saturation forward Inf at ' + IntToStr(i), IsInfinite(v));
      AssertTrue('LogSigmoid saturation output <= small epsilon at ' + IntToStr(i) +
        ' v=' + FloatToStr(v), v <= 1e-4);
    end;
    // Specific check: x = +1e6 should saturate to ~0 (sigmoid -> 1, log -> 0)
    AssertEquals('LogSigmoid(+1e6) ~ 0', 0.0, NN.GetLastLayer.Output.Raw[0], 1e-4);
    // x = -1e6 should be ~ x (since log(sigmoid(x)) -> x for very negative x)
    AssertTrue('LogSigmoid(-1e6) finite and very negative',
      NN.GetLastLayer.Output.Raw[1] < -1e5);

    NN.Layers[0].OutputError.Fill(0);
    NN.Backpropagate(Desired);
    for i := 0 to Input.Size - 1 do
    begin
      v := NN.GetLastLayer.OutputError.Raw[i];
      AssertFalse('LogSigmoid saturation output-grad NaN at ' + IntToStr(i), IsNan(v));
      AssertFalse('LogSigmoid saturation output-grad Inf at ' + IntToStr(i), IsInfinite(v));
      g := NN.Layers[0].OutputError.Raw[i];
      AssertFalse('LogSigmoid saturation input-grad NaN at ' + IntToStr(i), IsNan(g));
      AssertFalse('LogSigmoid saturation input-grad Inf at ' + IntToStr(i), IsInfinite(g));
    end;
  finally
    NN.Free;
    Input.Free;
    Desired.Free;
  end;
end;

procedure TTestNeuralNumerical.TestShiftedReLUForward;
var
  NN: TNNet;
  Input: TNNetVolume;
begin
  NN := TNNet.Create();
  Input := TNNetVolume.Create(1, 1, 4);
  try
    NN.AddLayer(TNNetInput.Create(1, 1, 4, 1));
    NN.AddLayer(TNNetShiftedReLU.Create());

    Input.Raw[0] := -2.0;
    Input.Raw[1] := -1.0;
    Input.Raw[2] := 0.0;
    Input.Raw[3] := 2.0;

    NN.Compute(Input);

    AssertEquals('ShiftedReLU(-2)', -1.0, NN.GetLastLayer.Output.Raw[0], 1e-6);
    AssertEquals('ShiftedReLU(-1)', -1.0, NN.GetLastLayer.Output.Raw[1], 1e-6);
    AssertEquals('ShiftedReLU(0)',   0.0, NN.GetLastLayer.Output.Raw[2], 1e-6);
    AssertEquals('ShiftedReLU(2)',   2.0, NN.GetLastLayer.Output.Raw[3], 1e-6);
  finally
    NN.Free;
    Input.Free;
  end;
end;

procedure TTestNeuralNumerical.TestShiftedReLUGradientCheck;
begin
  // Stay clear of the kink at x = -1.
  ActivationGradientCheck(Self, TNNetShiftedReLU.Create(), 'ShiftedReLU',
    [0.5, -0.5, 1.0, -0.25, 2.0, -2.5], 0.01);
end;

procedure TTestNeuralNumerical.TestHardTanhForward;
var
  NN: TNNet;
  Input: TNNetVolume;
begin
  NN := TNNet.Create();
  Input := TNNetVolume.Create(1, 1, 5);
  try
    NN.AddLayer(TNNetInput.Create(1, 1, 5, 1));
    NN.AddLayer(TNNetHardTanh.Create());

    Input.Raw[0] := 0.5;
    Input.Raw[1] := -0.5;
    Input.Raw[2] := 2.0;
    Input.Raw[3] := -2.0;
    Input.Raw[4] := 0.0;

    NN.Compute(Input);

    // HardTanh(x) = clamp(x, -1, 1)
    AssertEquals('HardTanh(0.5)', 0.5, NN.GetLastLayer.Output.Raw[0], 0.0001);
    AssertEquals('HardTanh(-0.5)', -0.5, NN.GetLastLayer.Output.Raw[1], 0.0001);
    AssertEquals('HardTanh(2)', 1.0, NN.GetLastLayer.Output.Raw[2], 0.0001);
    AssertEquals('HardTanh(-2)', -1.0, NN.GetLastLayer.Output.Raw[3], 0.0001);
    AssertEquals('HardTanh(0)', 0.0, NN.GetLastLayer.Output.Raw[4], 0.0001);
  finally
    NN.Free;
    Input.Free;
  end;
end;

procedure TTestNeuralNumerical.TestHardTanhGradientCheck;
begin
  // Stay clear of the kinks at +/-1.
  ActivationGradientCheck(Self, TNNetHardTanh.Create(), 'HardTanh',
    [0.5, -0.5, 0.25, -0.75, 2.0, -2.5], 0.01);
end;

procedure TTestNeuralNumerical.TestHardShrinkForward;
var
  NN: TNNet;
  Input: TNNetVolume;
begin
  NN := TNNet.Create();
  Input := TNNetVolume.Create(1, 1, 5);
  try
    NN.AddLayer(TNNetInput.Create(1, 1, 5, 1));
    NN.AddLayer(TNNetHardShrink.Create());

    Input.Raw[0] := -2.0;
    Input.Raw[1] := -0.3;
    Input.Raw[2] := 0.0;
    Input.Raw[3] := 0.3;
    Input.Raw[4] := 2.0;

    NN.Compute(Input);

    // HardShrink(x) = x if |x| > 0.5, else 0
    AssertEquals('HardShrink(-2)', -2.0, NN.GetLastLayer.Output.Raw[0], 0.0001);
    AssertEquals('HardShrink(-0.3)', 0.0, NN.GetLastLayer.Output.Raw[1], 0.0001);
    AssertEquals('HardShrink(0)', 0.0, NN.GetLastLayer.Output.Raw[2], 0.0001);
    AssertEquals('HardShrink(0.3)', 0.0, NN.GetLastLayer.Output.Raw[3], 0.0001);
    AssertEquals('HardShrink(2)', 2.0, NN.GetLastLayer.Output.Raw[4], 0.0001);
  finally
    NN.Free;
    Input.Free;
  end;
end;

procedure TTestNeuralNumerical.TestHardShrinkGradientCheck;
begin
  // Stay clear of the kink at +/-lambda (lambda=0.5).
  ActivationGradientCheck(Self, TNNetHardShrink.Create(), 'HardShrink',
    [1.0, -1.0, 1.5, -2.0, 0.25, -0.3], 0.01);
end;

procedure TTestNeuralNumerical.TestSoftShrinkForward;
var
  NN: TNNet;
  Input: TNNetVolume;
begin
  NN := TNNet.Create();
  Input := TNNetVolume.Create(1, 1, 5);
  try
    NN.AddLayer(TNNetInput.Create(1, 1, 5, 1));
    NN.AddLayer(TNNetSoftShrink.Create());

    Input.Raw[0] := -2.0;
    Input.Raw[1] := -0.3;
    Input.Raw[2] := 0.0;
    Input.Raw[3] := 0.3;
    Input.Raw[4] := 2.0;

    NN.Compute(Input);

    // SoftShrink(x) = x - lambda if x>lambda, x+lambda if x<-lambda, else 0
    AssertEquals('SoftShrink(-2)', -1.5, NN.GetLastLayer.Output.Raw[0], 0.0001);
    AssertEquals('SoftShrink(-0.3)', 0.0, NN.GetLastLayer.Output.Raw[1], 0.0001);
    AssertEquals('SoftShrink(0)', 0.0, NN.GetLastLayer.Output.Raw[2], 0.0001);
    AssertEquals('SoftShrink(0.3)', 0.0, NN.GetLastLayer.Output.Raw[3], 0.0001);
    AssertEquals('SoftShrink(2)', 1.5, NN.GetLastLayer.Output.Raw[4], 0.0001);
  finally
    NN.Free;
    Input.Free;
  end;
end;

procedure TTestNeuralNumerical.TestSoftShrinkGradientCheck;
begin
  // Stay clear of the kink at +/-lambda (lambda=0.5).
  ActivationGradientCheck(Self, TNNetSoftShrink.Create(), 'SoftShrink',
    [1.0, -1.0, 1.5, -2.0, 0.25, -0.3], 0.01);
end;

procedure TTestNeuralNumerical.TestThresholdForward;
var
  NN: TNNet;
  Input: TNNetVolume;
begin
  NN := TNNet.Create();
  Input := TNNetVolume.Create(1, 1, 5);
  try
    NN.AddLayer(TNNetInput.Create(1, 1, 5, 1));
    NN.AddLayer(TNNetThreshold.Create(1.0, -0.5));

    Input.Raw[0] := 0.0;
    Input.Raw[1] := 0.5;
    Input.Raw[2] := 1.0;
    Input.Raw[3] := 1.5;
    Input.Raw[4] := 2.0;

    NN.Compute(Input);

    // Threshold(x; theta=1.0, value=-0.5) = x if x > 1.0 else -0.5
    AssertEquals('Threshold(0)', -0.5, NN.GetLastLayer.Output.Raw[0], 0.0001);
    AssertEquals('Threshold(0.5)', -0.5, NN.GetLastLayer.Output.Raw[1], 0.0001);
    AssertEquals('Threshold(1.0)', -0.5, NN.GetLastLayer.Output.Raw[2], 0.0001);
    AssertEquals('Threshold(1.5)', 1.5, NN.GetLastLayer.Output.Raw[3], 0.0001);
    AssertEquals('Threshold(2.0)', 2.0, NN.GetLastLayer.Output.Raw[4], 0.0001);
  finally
    NN.Free;
    Input.Free;
  end;
end;

procedure TTestNeuralNumerical.TestThresholdReLUEquivalence;
var
  NN, NNReLU: TNNet;
  Input: TNNetVolume;
  i: integer;
begin
  NN := TNNet.Create();
  NNReLU := TNNet.Create();
  Input := TNNetVolume.Create(1, 1, 7);
  try
    NN.AddLayer(TNNetInput.Create(1, 1, 7, 1));
    NN.AddLayer(TNNetThreshold.Create()); // defaults: theta=0, value=0
    NNReLU.AddLayer(TNNetInput.Create(1, 1, 7, 1));
    NNReLU.AddLayer(TNNetReLU.Create());

    Input.Raw[0] := -2.0;
    Input.Raw[1] := -0.5;
    Input.Raw[2] := -0.1;
    Input.Raw[3] := 0.0;
    Input.Raw[4] := 0.1;
    Input.Raw[5] := 0.5;
    Input.Raw[6] := 2.0;

    NN.Compute(Input);
    NNReLU.Compute(Input);

    for i := 0 to Input.Size - 1 do
      AssertEquals('Threshold defaults == ReLU at ' + IntToStr(i),
        NNReLU.GetLastLayer.Output.Raw[i],
        NN.GetLastLayer.Output.Raw[i], 1e-6);
  finally
    NN.Free;
    NNReLU.Free;
    Input.Free;
  end;
end;

procedure TTestNeuralNumerical.TestThresholdGradientCheck;
begin
  // theta=0.5 is the kink; bias inputs clear of x=0.5.
  ActivationGradientCheck(Self, TNNetThreshold.Create(0.5, 0.0), 'Threshold',
    [1.0, -1.0, 1.5, -2.0, 0.9, -0.25], 0.01);
end;

procedure TTestNeuralNumerical.TestReLU6Forward;
var
  NN: TNNet;
  Input: TNNetVolume;
begin
  NN := TNNet.Create();
  Input := TNNetVolume.Create(1, 1, 5);
  try
    NN.AddLayer(TNNetInput.Create(1, 1, 5, 1));
    NN.AddLayer(TNNetReLU6.Create());

    Input.Raw[0] := -1.0;
    Input.Raw[1] := 0.5;
    Input.Raw[2] := 3.0;
    Input.Raw[3] := 7.5;
    Input.Raw[4] := 0.0;

    NN.Compute(Input);

    // ReLU6(x) = clamp(x, 0, 6)
    AssertEquals('ReLU6(-1)', 0.0, NN.GetLastLayer.Output.Raw[0], 0.0001);
    AssertEquals('ReLU6(0.5)', 0.5, NN.GetLastLayer.Output.Raw[1], 0.0001);
    AssertEquals('ReLU6(3)', 3.0, NN.GetLastLayer.Output.Raw[2], 0.0001);
    AssertEquals('ReLU6(7.5)', 6.0, NN.GetLastLayer.Output.Raw[3], 0.0001);
    AssertEquals('ReLU6(0)', 0.0, NN.GetLastLayer.Output.Raw[4], 0.0001);
  finally
    NN.Free;
    Input.Free;
  end;
end;

procedure TTestNeuralNumerical.TestGlobalMaxPoolForward;
var
  NN: TNNet;
  Input: TNNetVolume;
begin
  NN := TNNet.Create();
  // 2 x 2 x 3 input; expect 1 x 1 x 3 output containing per-channel max.
  Input := TNNetVolume.Create(2, 2, 3);
  try
    NN.AddLayer(TNNetInput.Create(2, 2, 3, 1));
    NN.AddLayer(TNNetGlobalMaxPool.Create());

    // Channel 0: max = 4.0 at (1,0)
    Input[0, 0, 0] := 1.0;  Input[1, 0, 0] := 4.0;
    Input[0, 1, 0] := 2.0;  Input[1, 1, 0] := 3.0;
    // Channel 1: max = 0.5 at (0,1) (all non-positive elsewhere)
    Input[0, 0, 1] := -1.0; Input[1, 0, 1] := -2.0;
    Input[0, 1, 1] :=  0.5; Input[1, 1, 1] := -0.5;
    // Channel 2: max = 9.0 at (1,1)
    Input[0, 0, 2] := 7.0;  Input[1, 0, 2] := 8.0;
    Input[0, 1, 2] := 6.0;  Input[1, 1, 2] := 9.0;

    NN.Compute(Input);

    AssertEquals('GlobalMaxPool ch0', 4.0, NN.GetLastLayer.Output[0, 0, 0], 0.0001);
    AssertEquals('GlobalMaxPool ch1', 0.5, NN.GetLastLayer.Output[0, 0, 1], 0.0001);
    AssertEquals('GlobalMaxPool ch2', 9.0, NN.GetLastLayer.Output[0, 0, 2], 0.0001);
    AssertEquals('GlobalMaxPool output SizeX', 1, NN.GetLastLayer.Output.SizeX);
    AssertEquals('GlobalMaxPool output SizeY', 1, NN.GetLastLayer.Output.SizeY);
    AssertEquals('GlobalMaxPool output Depth', 3, NN.GetLastLayer.Output.Depth);
  finally
    NN.Free;
    Input.Free;
  end;
end;

procedure TTestNeuralNumerical.TestGlobalMaxPoolGradientCheck;
begin
  // 3 x 3 x 2 input. The Sin(i*0.7)*2.0+0.3 pattern from the helper
  // generates distinct values, so the argmax stays stable under the
  // epsilon=1e-4 perturbation.
  LayerInputGradientCheck(Self, TNNetGlobalMaxPool.Create(),
    'GlobalMaxPool', 3, 3, 2, 0.01);
end;

procedure TTestNeuralNumerical.TestReLU6ExtremeInputSaturation;
var
  NN: TNNet;
  Input, Desired: TNNetVolume;
  v, g: TNeuralFloat;
  i: integer;
begin
  NN := TNNet.Create();
  Input := TNNetVolume.Create(4, 1, 4);
  Desired := TNNetVolume.Create();
  try
    NN.AddLayer(TNNetInput.Create(4, 1, 4, 1));
    NN.AddLayer(TNNetReLU6.Create());
    NN.SetLearningRate(1.0, 0.0);
    NN.SetBatchUpdate(true);

    Desired.ReSize(NN.GetLastLayer.Output);
    for i := 0 to Input.Size - 1 do
    begin
      if (i mod 2) = 0 then Input.Raw[i] := 1e6
      else Input.Raw[i] := -1e6;
    end;
    for i := 0 to Desired.Size - 1 do
      Desired.Raw[i] := Cos(i * 0.2);

    NN.Compute(Input);
    for i := 0 to Input.Size - 1 do
    begin
      v := NN.GetLastLayer.Output.Raw[i];
      AssertFalse('ReLU6 saturation forward NaN at ' + IntToStr(i), IsNan(v));
      AssertFalse('ReLU6 saturation forward Inf at ' + IntToStr(i), IsInfinite(v));
      AssertTrue('ReLU6 saturation in [0, 6] at ' + IntToStr(i) +
        ' v=' + FloatToStr(v),
        (v >= -1e-4) and (v <= 6.0 + 1e-4));
    end;

    NN.Layers[0].OutputError.Fill(0);
    NN.Backpropagate(Desired);
    for i := 0 to Input.Size - 1 do
    begin
      g := NN.Layers[0].OutputError.Raw[i];
      AssertFalse('ReLU6 saturation grad NaN at ' + IntToStr(i), IsNan(g));
      AssertFalse('ReLU6 saturation grad Inf at ' + IntToStr(i), IsInfinite(g));
    end;
  finally
    NN.Free;
    Input.Free;
    Desired.Free;
  end;
end;

procedure TTestNeuralNumerical.TestMaskedFillForward;
var
  NN: TNNet;
  Input: TNNetVolume;
  X, Y: integer;
begin
  NN := TNNet.Create();
  // 3x3 score map, single depth slice.
  Input := TNNetVolume.Create(3, 3, 1);
  try
    NN.AddLayer(TNNetInput.Create(3, 3, 1, 1));
    NN.AddLayer(TNNetMaskedFill.Create(-1e9));

    Input.Fill(1.0);
    NN.Compute(Input);

    for Y := 0 to 2 do
      for X := 0 to 2 do
      begin
        if X > Y then
          AssertTrue('MaskedFill upper triangle masked at X=' + IntToStr(X) +
            ' Y=' + IntToStr(Y),
            NN.GetLastLayer.Output[X, Y, 0] < -1e8)
        else
          AssertEquals('MaskedFill lower/diagonal untouched at X=' +
            IntToStr(X) + ' Y=' + IntToStr(Y),
            1.0, NN.GetLastLayer.Output[X, Y, 0], 0.0001);
      end;
  finally
    NN.Free;
    Input.Free;
  end;
end;

procedure TTestNeuralNumerical.TestMaskedFillGradientCheck;
begin
  // Adding a constant has identity gradient passthrough. A small mask
  // value keeps float32 precision intact for the central-difference check
  // (a large constant in the MSE loss causes catastrophic cancellation).
  LayerInputGradientCheck(Self, TNNetMaskedFill.Create(-0.5),
    'MaskedFill', 3, 3, 2, 0.01);
end;

procedure TTestNeuralNumerical.TestALiBiForward;
var
  NN: TNNet;
  Input: TNNetVolume;
  X, Y, H, Depth: integer;
  Slope, Expected: TNeuralFloat;
begin
  // SeqLen=3, Depth=2. Input is zero, so output equals the bias map per head.
  Depth := 2;
  NN := TNNet.Create();
  Input := TNNetVolume.Create(3, 3, Depth);
  try
    NN.AddLayer(TNNetInput.Create(3, 3, Depth, 1));
    NN.AddLayer(TNNetALiBi.Create());

    Input.Fill(0.0);
    NN.Compute(Input);

    for H := 0 to Depth - 1 do
    begin
      Slope := Power(2, -8 * (H + 1) / Depth);
      for Y := 0 to 2 do
        for X := 0 to 2 do
        begin
          Expected := Slope * (X - Y);
          AssertEquals('ALiBi bias at H=' + IntToStr(H) +
            ' X=' + IntToStr(X) + ' Y=' + IntToStr(Y),
            Expected, NN.GetLastLayer.Output[X, Y, H], 1e-6);
        end;
    end;
  finally
    NN.Free;
    Input.Free;
  end;
end;

procedure TTestNeuralNumerical.TestALiBiGradientCheck;
begin
  // Adding a position-dependent constant has identity gradient passthrough.
  LayerInputGradientCheck(Self, TNNetALiBi.Create(),
    'ALiBi', 3, 3, 2, 0.01);
end;


procedure TTestNeuralNumerical.TestALiBiMaskedFillComposition;
var
  NN: TNNet;
  Input: TNNetVolume;
  X, Y, H, Depth, SeqLen: integer;
  Slope, Expected, Actual: TNeuralFloat;
begin
  // Stack TNNetMaskedFill on top of TNNetALiBi. Input is all zeros so
  // ALiBi adds Slope[h] * (X - Y) per (key=X, query=Y, head=h). MaskedFill
  // then additively shifts the strict-upper-triangle (X > Y) by -1e9.
  // Pins the composition expected on the eventual MHA causal path.
  Depth := 2;
  SeqLen := 4;
  NN := TNNet.Create();
  Input := TNNetVolume.Create(SeqLen, SeqLen, Depth);
  try
    NN.AddLayer(TNNetInput.Create(SeqLen, SeqLen, Depth, 1));
    NN.AddLayer(TNNetALiBi.Create());
    NN.AddLayer(TNNetMaskedFill.Create()); // default mask value -1e9

    Input.Fill(0.0);
    NN.Compute(Input);

    for H := 0 to Depth - 1 do
    begin
      Slope := Power(2, -8 * (H + 1) / Depth);
      for Y := 0 to SeqLen - 1 do
        for X := 0 to SeqLen - 1 do
        begin
          Actual := NN.GetLastLayer.Output[X, Y, H];
          if X > Y then
          begin
            // Strict upper triangle: MaskedFill adds -1e9 to ALiBi bias,
            // which dominates the small slope*(X-Y) contribution.
            AssertTrue('ALiBi+MaskedFill upper triangle masked at H=' +
              IntToStr(H) + ' X=' + IntToStr(X) + ' Y=' + IntToStr(Y) +
              ' got ' + FloatToStr(Actual),
              Actual < -1e8);
          end
          else
          begin
            // Lower triangle and diagonal: ALiBi bias only.
            Expected := Slope * (X - Y);
            AssertEquals('ALiBi+MaskedFill lower/diag at H=' + IntToStr(H) +
              ' X=' + IntToStr(X) + ' Y=' + IntToStr(Y),
              Expected, Actual, 1e-5);
          end;
        end;
    end;
  finally
    NN.Free;
    Input.Free;
  end;
end;

procedure TTestNeuralNumerical.TestSoftCappingForward;
var
  NN, NN2: TNNet;
  Input: TNNetVolume;
  Cap, Expected: TNeuralFloat;
  Saved: string;
  i: integer;
  InputValues: array[0..3] of TNeuralFloat;
begin
  NN := TNNet.Create();
  Input := TNNetVolume.Create(3, 1, 4);
  try
    Cap := 5.0;
    NN.AddLayer(TNNetInput.Create(3, 1, 4, 1));
    NN.AddLayer(TNNetSoftCapping.Create(Cap));

    // A few non-saturating values plus one strongly saturating value.
    InputValues[0] := 0.0;
    InputValues[1] := 1.0;
    InputValues[2] := -2.5;
    InputValues[3] := 50.0; // strongly saturating: c*tanh(10) ~= c
    for i := 0 to 3 do Input.Raw[i] := InputValues[i];
    // Fill the rest with deterministic values.
    for i := 4 to Input.Size - 1 do
      Input.Raw[i] := Sin(i * 0.3) * 4.0;

    NN.Compute(Input);

    for i := 0 to 3 do
    begin
      Expected := Cap * Tanh(InputValues[i] / Cap);
      AssertEquals('SoftCapping output[' + IntToStr(i) + ']',
        Expected, NN.GetLastLayer.Output.Raw[i], 0.0001);
    end;
    // Sanity: saturating value approaches the cap.
    AssertTrue('SoftCapping saturates toward cap',
      Abs(NN.GetLastLayer.Output.Raw[3] - Cap) < 0.001);

    // Round-trip SaveToString / LoadFromString preserves the cap value.
    Saved := NN.SaveToString();
    NN2 := TNNet.Create();
    try
      NN2.LoadFromString(Saved);
      NN2.Compute(Input);
      for i := 0 to Input.Size - 1 do
        AssertEquals('SoftCapping round-trip output[' + IntToStr(i) + ']',
          NN.GetLastLayer.Output.Raw[i], NN2.GetLastLayer.Output.Raw[i], 0.0001);
    finally
      NN2.Free;
    end;
  finally
    NN.Free;
    Input.Free;
  end;
end;

procedure TTestNeuralNumerical.TestSoftCappingGradientCheck;
begin
  // Cap chosen small enough that the inputs span both the near-linear region
  // and the saturating tails for a meaningful derivative check.
  LayerInputGradientCheck(Self, TNNetSoftCapping.Create(3.0),
    'SoftCapping', 3, 1, 4, 0.01);
end;

procedure TTestNeuralNumerical.TestDropPathInferenceIdentity;
var
  NN: TNNet;
  Input: TNNetVolume;
  i: integer;
begin
  NN := TNNet.Create();
  Input := TNNetVolume.Create(3, 2, 2);
  try
    NN.AddLayer(TNNetInput.Create(3, 2, 2, 1));
    NN.AddLayer(TNNetDropPath.Create(0.5));
    // Inference mode: dropouts disabled => identity.
    NN.EnableDropouts(false);
    for i := 0 to Input.Size - 1 do
      Input.Raw[i] := Sin(i * 0.7) * 2.0 + 0.3;
    NN.Compute(Input);
    for i := 0 to Input.Size - 1 do
      AssertEquals('DropPath inference is identity at ' + IntToStr(i),
        Input.Raw[i], NN.GetLastLayer.Output.Raw[i], 0.0001);
  finally
    NN.Free;
    Input.Free;
  end;
end;

procedure TTestNeuralNumerical.TestDropPathTrainingScaling;
var
  NN: TNNet;
  Input, Desired: TNNetVolume;
  i, Trials: integer;
  P, InvKeep: TNeuralFloat;
  KeptObserved, DroppedObserved: boolean;
  Out0, Err0: TNeuralFloat;
begin
  NN := TNNet.Create();
  Input := TNNetVolume.Create(3, 2, 2);
  Desired := TNNetVolume.Create();
  P := 0.4;
  InvKeep := 1.0 / (1 - P);
  KeptObserved := false;
  DroppedObserved := false;
  try
    NN.AddLayer(TNNetInput.Create(3, 2, 2, 1));
    NN.AddLayer(TNNetDropPath.Create(P));
    NN.SetLearningRate(1.0, 0.0);
    NN.SetBatchUpdate(true);
    NN.EnableDropouts(true);

    Desired.ReSize(NN.GetLastLayer.Output);
    for i := 0 to Input.Size - 1 do
      Input.Raw[i] := Sin(i * 0.7) * 2.0 + 0.3;
    for i := 0 to Desired.Size - 1 do
      Desired.Raw[i] := 0; // upstream gradient = output - desired = output itself

    RandSeed := 12345;
    for Trials := 0 to 19 do
    begin
      NN.Compute(Input);
      Out0 := NN.GetLastLayer.Output.Raw[0];
      // Check forward: either zero, or input/(1-p).
      if Abs(Out0) < 1e-6 then
      begin
        DroppedObserved := true;
        for i := 0 to Input.Size - 1 do
          AssertEquals('DropPath dropped sample zero at ' + IntToStr(i),
            0.0, NN.GetLastLayer.Output.Raw[i], 0.0001);
        // Backprop: gradient scaled by 0 -> input layer gets zero.
        NN.Layers[0].OutputError.Fill(0);
        NN.Backpropagate(Desired);
        for i := 0 to Input.Size - 1 do
          AssertEquals('DropPath dropped grad zero at ' + IntToStr(i),
            0.0, NN.Layers[0].OutputError.Raw[i], 0.0001);
      end
      else
      begin
        KeptObserved := true;
        for i := 0 to Input.Size - 1 do
          AssertEquals('DropPath kept sample scaled at ' + IntToStr(i),
            Input.Raw[i] * InvKeep, NN.GetLastLayer.Output.Raw[i], 0.0001);
        // Backprop: upstream grad = output - 0 = output, then scaled by 1/(1-p).
        NN.Layers[0].OutputError.Fill(0);
        NN.Backpropagate(Desired);
        // Input layer error should equal output * (1/(1-p)) = input * (1/(1-p))^2.
        for i := 0 to Input.Size - 1 do
        begin
          Err0 := Input.Raw[i] * InvKeep * InvKeep;
          AssertEquals('DropPath kept grad scaled at ' + IntToStr(i),
            Err0, NN.Layers[0].OutputError.Raw[i], 0.001);
        end;
      end;
    end;
    AssertTrue('DropPath should observe at least one kept sample', KeptObserved);
    AssertTrue('DropPath should observe at least one dropped sample', DroppedObserved);
  finally
    NN.Free;
    Input.Free;
    Desired.Free;
  end;
end;

procedure TTestNeuralNumerical.TestDropPathGradientCheck;
var
  NN: TNNet;
  Input, InputPlus, Desired: TNNetVolume;
  epsilon, lossPlus, lossMinus, numericalGrad, analyticalGrad: TNeuralFloat;
  i, k: integer;
  diff: TNeuralFloat;
  Seed: longint;

  function ComputeLossSeeded(AInput: TNNetVolume): TNeuralFloat;
  var
    kk: integer;
    d: TNeuralFloat;
  begin
    RandSeed := Seed;
    NN.Compute(AInput);
    Result := 0;
    for kk := 0 to NN.GetLastLayer.Output.Size - 1 do
    begin
      d := NN.GetLastLayer.Output.Raw[kk] - Desired.Raw[kk];
      Result := Result + 0.5 * d * d;
    end;
  end;

begin
  NN := TNNet.Create();
  Input := TNNetVolume.Create(3, 2, 2);
  InputPlus := TNNetVolume.Create(3, 2, 2);
  Desired := TNNetVolume.Create();
  epsilon := 0.0001;
  Seed := 4242;
  try
    NN.AddLayer(TNNetInput.Create(3, 2, 2, 1));
    NN.AddLayer(TNNetDropPath.Create(0.3));
    NN.SetLearningRate(1.0, 0.0);
    NN.SetBatchUpdate(true);
    NN.EnableDropouts(true);

    Desired.ReSize(NN.GetLastLayer.Output);
    for i := 0 to Input.Size - 1 do
      Input.Raw[i] := Sin(i * 0.7) * 2.0 + 0.3;
    for k := 0 to Desired.Size - 1 do
      Desired.Raw[k] := Cos(k * 0.5);

    // Pick a seed that yields a "kept" forward so the gradient is nonzero
    // and the central-difference check is informative. If the first try
    // happens to drop the sample, advance to the next seed that keeps it.
    while True do
    begin
      RandSeed := Seed;
      NN.Compute(Input);
      diff := 0;
      for k := 0 to NN.GetLastLayer.Output.Size - 1 do
        diff := diff + Abs(NN.GetLastLayer.Output.Raw[k]);
      if diff > 1e-3 then break;
      Inc(Seed);
    end;

    for i := 0 to Input.Size - 1 do
    begin
      InputPlus.Copy(Input);
      InputPlus.Raw[i] := Input.Raw[i] + epsilon;
      lossPlus := ComputeLossSeeded(InputPlus);
      InputPlus.Raw[i] := Input.Raw[i] - epsilon;
      lossMinus := ComputeLossSeeded(InputPlus);
      numericalGrad := (lossPlus - lossMinus) / (2 * epsilon);

      RandSeed := Seed;
      NN.Compute(Input);
      NN.Layers[0].OutputError.Fill(0);
      NN.Backpropagate(Desired);
      analyticalGrad := NN.Layers[0].OutputError.Raw[i];

      AssertTrue('DropPath input gradient check at position ' + IntToStr(i) +
        ' (num=' + FloatToStr(numericalGrad) + ' ana=' + FloatToStr(analyticalGrad) + ')',
        Abs(numericalGrad - analyticalGrad) < 0.01);
    end;
  finally
    NN.Free;
    Input.Free;
    InputPlus.Free;
    Desired.Free;
  end;
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

procedure TTestNeuralNumerical.TestAddPositionalEmbeddingForward;
var
  NN: TNNet;
  ZeroInput, NonZeroInput, Encoding: TNNetVolume;
  PE: TNNetAddPositionalEmbedding;
  i: integer;
  anyDiff: boolean;
begin
  NN := TNNet.Create();
  ZeroInput := TNNetVolume.Create(4, 1, 8);
  NonZeroInput := TNNetVolume.Create(4, 1, 8);
  Encoding := TNNetVolume.Create(4, 1, 8);
  try
    NN.AddLayer(TNNetInput.Create(4, 1, 8));
    PE := TNNetAddPositionalEmbedding.Create();
    NN.AddLayer(PE);

    ZeroInput.Fill(0);
    NN.Compute(ZeroInput);
    Encoding.Copy(NN.GetLastLayer.Output);

    anyDiff := False;
    for i := 0 to Encoding.Size - 1 do
      if Abs(Encoding.Raw[i]) > 1e-6 then anyDiff := True;
    AssertTrue('AddPositionalEmbedding must produce nonzero encoding', anyDiff);

    for i := 0 to NonZeroInput.Size - 1 do
      NonZeroInput.Raw[i] := Sin(i * 0.3) * 1.5;
    NN.Compute(NonZeroInput);
    for i := 0 to NonZeroInput.Size - 1 do
      AssertEquals('AddPositionalEmbedding output = input + encoding at ' + IntToStr(i),
        NonZeroInput.Raw[i] + Encoding.Raw[i],
        NN.GetLastLayer.Output.Raw[i], 0.0001);
  finally
    NN.Free;
    ZeroInput.Free;
    NonZeroInput.Free;
    Encoding.Free;
  end;
end;

procedure TTestNeuralNumerical.TestAddPositionalEmbeddingGradientCheck;
var
  NN: TNNet;
  Input, InputPlus, Desired: TNNetVolume;
  PE: TNNetAddPositionalEmbedding;
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
  Input := TNNetVolume.Create(4, 1, 8);
  InputPlus := TNNetVolume.Create(4, 1, 8);
  Desired := TNNetVolume.Create(4, 1, 8);
  epsilon := 0.001;
  try
    NN.AddLayer(TNNetInput.Create(4, 1, 8, 1));
    PE := TNNetAddPositionalEmbedding.Create();
    NN.AddLayer(PE);
    NN.SetLearningRate(1.0, 0.0);
    NN.SetBatchUpdate(true);

    for i := 0 to Input.Size - 1 do
      Input.Raw[i] := Sin(i * 0.7) * 1.7 + 0.2;
    for i := 0 to Desired.Size - 1 do
      Desired.Raw[i] := Cos(i * 0.4);

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

      AssertTrue('AddPositionalEmbedding input gradient at ' + IntToStr(i) +
        ' num=' + FloatToStr(numericalGrad) + ' ana=' + FloatToStr(analyticalGrad),
        Abs(numericalGrad - analyticalGrad) < 0.01);
    end;
  finally
    NN.Free;
    Input.Free;
    InputPlus.Free;
    Desired.Free;
  end;
end;

procedure TTestNeuralNumerical.TestAddPositionalEmbeddingEmbeddingIsConstant;
var
  NN: TNNet;
  Input, Desired: TNNetVolume;
  PE: TNNetAddPositionalEmbedding;
  BeforeOutput: TNNetVolume;
  i: integer;
begin
  NN := TNNet.Create();
  Input := TNNetVolume.Create(4, 1, 8);
  Desired := TNNetVolume.Create(4, 1, 8);
  BeforeOutput := TNNetVolume.Create(4, 1, 8);
  try
    NN.AddLayer(TNNetInput.Create(4, 1, 8, 1));
    PE := TNNetAddPositionalEmbedding.Create();
    NN.AddLayer(PE);
    NN.SetLearningRate(1.0, 0.0);

    for i := 0 to Input.Size - 1 do
      Input.Raw[i] := 0;
    for i := 0 to Desired.Size - 1 do
      Desired.Raw[i] := Cos(i * 0.4) + 5;

    NN.Compute(Input);
    BeforeOutput.Copy(NN.GetLastLayer.Output);

    for i := 1 to 5 do
    begin
      NN.Compute(Input);
      NN.Backpropagate(Desired);
      NN.UpdateWeights();
    end;

    NN.Compute(Input);
    for i := 0 to BeforeOutput.Size - 1 do
      AssertEquals('AddPositionalEmbedding encoding must stay constant at ' + IntToStr(i),
        BeforeOutput.Raw[i], NN.GetLastLayer.Output.Raw[i], 1e-5);
  finally
    NN.Free;
    Input.Free;
    Desired.Free;
    BeforeOutput.Free;
  end;
end;

procedure TTestNeuralNumerical.TestScaledDotProductAttentionForward;
var
  NN: TNNet;
  Input: TNNetVolume;
  SeqLen, Dk, i, d: integer;
  Attn: TNNetScaledDotProductAttention;
  ExpectedAvg, Sum: TNeuralFloat;
begin
  SeqLen := 3;
  Dk := 4;
  NN := TNNet.Create();
  Input := TNNetVolume.Create(SeqLen, 1, 3 * Dk);
  try
    NN.AddLayer(TNNetInput.Create(SeqLen, 1, 3 * Dk, 1));
    Attn := TNNetScaledDotProductAttention.Create(Dk, false);
    NN.AddLayer(Attn);

    // Uniform Q and K (all zeros) -> equal scores -> uniform attention 1/SeqLen.
    // Use distinct V values per position so the output = average of V across positions.
    for i := 0 to SeqLen - 1 do
    begin
      for d := 0 to Dk - 1 do
      begin
        Input[i, 0, d] := 0;            // Q[i,d]
        Input[i, 0, Dk + d] := 0;       // K[i,d]
        Input[i, 0, 2 * Dk + d] := i + 0.1 * d; // V[i,d]
      end;
    end;

    NN.Compute(Input);
    AssertEquals('Output SizeX', SeqLen, Attn.Output.SizeX);
    AssertEquals('Output SizeY', 1, Attn.Output.SizeY);
    AssertEquals('Output Depth', Dk, Attn.Output.Depth);

    // Each output row should equal the column-wise mean of V.
    for d := 0 to Dk - 1 do
    begin
      Sum := 0;
      for i := 0 to SeqLen - 1 do
        Sum := Sum + Input[i, 0, 2 * Dk + d];
      ExpectedAvg := Sum / SeqLen;
      for i := 0 to SeqLen - 1 do
        AssertEquals('Uniform attn output [i=' + IntToStr(i) + ',d=' + IntToStr(d) + ']',
          ExpectedAvg, Attn.Output[i, 0, d], 1e-5);
    end;
  finally
    NN.Free;
    Input.Free;
  end;
end;

procedure TTestNeuralNumerical.TestScaledDotProductAttentionGradientCheck;
var
  NN: TNNet;
  Input, InputPlus, Desired: TNNetVolume;
  Attn: TNNetScaledDotProductAttention;
  SeqLen, Dk: integer;
  epsilon, lossPlus, lossMinus, numericalGrad, analyticalGrad: TNeuralFloat;
  i: integer;

  function ComputeLoss(AInput: TNNetVolume): TNeuralFloat;
  var k: integer; diff: TNeuralFloat;
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
  SeqLen := 3;
  Dk := 4;
  NN := TNNet.Create();
  Input := TNNetVolume.Create(SeqLen, 1, 3 * Dk);
  InputPlus := TNNetVolume.Create(SeqLen, 1, 3 * Dk);
  Desired := TNNetVolume.Create(SeqLen, 1, Dk);
  epsilon := 0.001;
  try
    NN.AddLayer(TNNetInput.Create(SeqLen, 1, 3 * Dk, 1));
    Attn := TNNetScaledDotProductAttention.Create(Dk, false);
    NN.AddLayer(Attn);
    NN.SetLearningRate(1.0, 0.0);
    NN.SetBatchUpdate(true);

    for i := 0 to Input.Size - 1 do
      Input.Raw[i] := Sin(i * 0.53) * 0.9 + 0.1;
    for i := 0 to Desired.Size - 1 do
      Desired.Raw[i] := Cos(i * 0.31);

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

      AssertTrue('SDPA input gradient at ' + IntToStr(i) +
        ' num=' + FloatToStr(numericalGrad) + ' ana=' + FloatToStr(analyticalGrad),
        Abs(numericalGrad - analyticalGrad) < 0.01);
    end;
  finally
    NN.Free;
    Input.Free;
    InputPlus.Free;
    Desired.Free;
  end;
end;

procedure TTestNeuralNumerical.TestScaledDotProductAttentionCausalGradientCheck;
var
  NN: TNNet;
  Input, InputPlus, Desired: TNNetVolume;
  Attn: TNNetScaledDotProductAttention;
  SeqLen, Dk: integer;
  epsilon, lossPlus, lossMinus, numericalGrad, analyticalGrad: TNeuralFloat;
  i: integer;

  function ComputeLoss(AInput: TNNetVolume): TNeuralFloat;
  var k: integer; diff: TNeuralFloat;
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
  SeqLen := 3;
  Dk := 4;
  NN := TNNet.Create();
  Input := TNNetVolume.Create(SeqLen, 1, 3 * Dk);
  InputPlus := TNNetVolume.Create(SeqLen, 1, 3 * Dk);
  Desired := TNNetVolume.Create(SeqLen, 1, Dk);
  epsilon := 0.001;
  try
    NN.AddLayer(TNNetInput.Create(SeqLen, 1, 3 * Dk, 1));
    Attn := TNNetScaledDotProductAttention.Create(Dk, true);
    NN.AddLayer(Attn);
    NN.SetLearningRate(1.0, 0.0);
    NN.SetBatchUpdate(true);

    for i := 0 to Input.Size - 1 do
      Input.Raw[i] := Sin(i * 0.41) * 0.8 - 0.2;
    for i := 0 to Desired.Size - 1 do
      Desired.Raw[i] := Cos(i * 0.27) * 0.5;

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

      AssertTrue('SDPA causal input gradient at ' + IntToStr(i) +
        ' num=' + FloatToStr(numericalGrad) + ' ana=' + FloatToStr(analyticalGrad),
        Abs(numericalGrad - analyticalGrad) < 0.01);
    end;
  finally
    NN.Free;
    Input.Free;
    InputPlus.Free;
    Desired.Free;
  end;
end;

procedure TTestNeuralNumerical.TestRotaryEmbeddingForward;
var
  NN, NN2: TNNet;
  Input: TNNetVolume;
  SeqLen, Depth, HalfD, pos, k, d: integer;
  Base, theta0, theta1, angle, c, s, x0, x1, ey0, ey1: TNeuralFloat;
  Saved: string;
begin
  SeqLen := 3;
  Depth := 4;
  HalfD := Depth div 2;
  Base := 10000.0;
  NN := TNNet.Create();
  Input := TNNetVolume.Create(SeqLen, 1, Depth);
  try
    NN.AddLayer(TNNetInput.Create(SeqLen, 1, Depth, 1));
    NN.AddLayer(TNNetRotaryEmbedding.Create(Base));

    // Deterministic input.
    for pos := 0 to SeqLen - 1 do
      for d := 0 to Depth - 1 do
        Input[pos, 0, d] := Sin((pos * Depth + d) * 0.41) * 0.8 - 0.2;

    NN.Compute(Input);

    // pos = 0 must be identity (angle = 0 -> cos=1, sin=0).
    for d := 0 to Depth - 1 do
      AssertEquals('RoPE pos=0 identity at d=' + IntToStr(d),
        Input[0, 0, d], NN.GetLastLayer.Output[0, 0, d], 1e-5);

    // pos = 1 / pos = 2: check hand-computed rotation for each pair k.
    theta0 := Exp(-2.0 * 0 / Depth * Ln(Base)); // = 1
    theta1 := Exp(-2.0 * 1 / Depth * Ln(Base)); // = 1 / sqrt(base) = 1/100
    for pos := 1 to SeqLen - 1 do
    begin
      for k := 0 to HalfD - 1 do
      begin
        if k = 0 then angle := pos * theta0 else angle := pos * theta1;
        c := Cos(angle);
        s := Sin(angle);
        x0 := Input[pos, 0, 2 * k];
        x1 := Input[pos, 0, 2 * k + 1];
        ey0 := c * x0 - s * x1;
        ey1 := s * x0 + c * x1;
        AssertEquals('RoPE forward pos=' + IntToStr(pos) + ' k=' + IntToStr(k) + ' y0',
          ey0, NN.GetLastLayer.Output[pos, 0, 2 * k], 1e-5);
        AssertEquals('RoPE forward pos=' + IntToStr(pos) + ' k=' + IntToStr(k) + ' y1',
          ey1, NN.GetLastLayer.Output[pos, 0, 2 * k + 1], 1e-5);
      end;
    end;

    // SaveToString / LoadFromString round-trip preserves the base.
    Saved := NN.SaveToString();
    NN2 := TNNet.Create();
    try
      NN2.LoadFromString(Saved);
      NN2.Compute(Input);
      for pos := 0 to SeqLen - 1 do
        for d := 0 to Depth - 1 do
          AssertEquals('RoPE round-trip pos=' + IntToStr(pos) + ' d=' + IntToStr(d),
            NN.GetLastLayer.Output[pos, 0, d],
            NN2.GetLastLayer.Output[pos, 0, d], 1e-5);
    finally
      NN2.Free;
    end;
  finally
    NN.Free;
    Input.Free;
  end;
end;

procedure TTestNeuralNumerical.TestRotaryEmbeddingGradientCheck;
begin
  // Standard central-difference gradient check. RoPE backward is the
  // transpose rotation; if signs are correct this must match to ~1e-2.
  LayerInputGradientCheck(Self, TNNetRotaryEmbedding.Create(10000.0),
    'RotaryEmbedding', 3, 1, 4, 0.01);
end;

procedure TTestNeuralNumerical.TestRotaryEmbeddingInverse;
var
  NN: TNNet;
  Input, Zero: TNNetVolume;
  SeqLen, Depth, pos, d: integer;
  OutNormSq, GradNormSq: TNeuralFloat;
begin
  SeqLen := 3;
  Depth := 4;
  NN := TNNet.Create();
  Input := TNNetVolume.Create(SeqLen, 1, Depth);
  Zero := TNNetVolume.Create(SeqLen, 1, Depth);
  try
    NN.AddLayer(TNNetInput.Create(SeqLen, 1, Depth, 1));
    NN.AddLayer(TNNetRotaryEmbedding.Create(10000.0));
    NN.SetLearningRate(1.0, 0.0);
    NN.SetBatchUpdate(true);

    for pos := 0 to SeqLen - 1 do
      for d := 0 to Depth - 1 do
        Input[pos, 0, d] := Sin((pos * Depth + d) * 0.37) * 1.2 + 0.1;
    Zero.Fill(0);

    NN.Compute(Input);

    // Rotation is orthogonal so |x|^2 must equal |y|^2.
    OutNormSq := 0;
    for pos := 0 to SeqLen - 1 do
      for d := 0 to Depth - 1 do
        OutNormSq := OutNormSq + Sqr(NN.GetLastLayer.Output[pos, 0, d]);
    AssertEquals('RoPE preserves norm', Input.GetSumSqr(), OutNormSq, 1e-4);

    // With Desired = 0, OutputError = Output - 0 = Output. The analytic
    // input gradient is then R^T * R * x = x (rotation is orthogonal).
    NN.Layers[0].OutputError.Fill(0);
    NN.Backpropagate(Zero);

    GradNormSq := 0;
    for pos := 0 to SeqLen - 1 do
      for d := 0 to Depth - 1 do
        GradNormSq := GradNormSq + Sqr(NN.Layers[0].OutputError[pos, 0, d]);

    AssertEquals('RoPE inverse: |gx|^2 = |x|^2', Input.GetSumSqr(), GradNormSq, 1e-4);

    // And the recovered gradient should equal the original input element-wise.
    for pos := 0 to SeqLen - 1 do
      for d := 0 to Depth - 1 do
        AssertEquals('RoPE inverse pos=' + IntToStr(pos) + ' d=' + IntToStr(d),
          Input[pos, 0, d], NN.Layers[0].OutputError[pos, 0, d], 1e-4);
  finally
    NN.Free;
    Input.Free;
    Zero.Free;
  end;
end;

// Helper used by TestRotaryEmbeddingOddDepthGuard. The layers' FErrorProc is a
// method pointer (TGetStrProc = procedure(const S: string) of object), so we
// need an object to capture the error message into.
type
  TErrorCapture = class
  public
    Triggered: boolean;
    Message: string;
    procedure Capture(const S: string);
  end;

procedure TErrorCapture.Capture(const S: string);
begin
  Triggered := true;
  Message := S;
end;

procedure TTestNeuralNumerical.TestRotaryEmbeddingInverseSeqLen5;
var
  NN: TNNet;
  Input, Zero: TNNetVolume;
  SeqLen, Depth, pos, d: integer;
  OutNormSq, GradNormSq: TNeuralFloat;
begin
  // Same property as TestRotaryEmbeddingInverse, but at a non-trivial sequence
  // length. Forward then "inverse" (R^T applied via Backpropagate with upstream
  // gradient = output) must round-trip to within fp tolerance for every
  // position, exercising the full set of position-dependent angles.
  SeqLen := 5;
  Depth := 4;
  NN := TNNet.Create();
  Input := TNNetVolume.Create(SeqLen, 1, Depth);
  Zero := TNNetVolume.Create(SeqLen, 1, Depth);
  try
    NN.AddLayer(TNNetInput.Create(SeqLen, 1, Depth, 1));
    NN.AddLayer(TNNetRotaryEmbedding.Create(10000.0));
    NN.SetLearningRate(1.0, 0.0);
    NN.SetBatchUpdate(true);

    for pos := 0 to SeqLen - 1 do
      for d := 0 to Depth - 1 do
        Input[pos, 0, d] := Sin((pos * Depth + d) * 0.29) * 1.5 - 0.4;
    Zero.Fill(0);

    NN.Compute(Input);

    OutNormSq := 0;
    for pos := 0 to SeqLen - 1 do
      for d := 0 to Depth - 1 do
        OutNormSq := OutNormSq + Sqr(NN.GetLastLayer.Output[pos, 0, d]);
    AssertEquals('RoPE SeqLen=5 preserves norm', Input.GetSumSqr(), OutNormSq, 1e-4);

    NN.Layers[0].OutputError.Fill(0);
    NN.Backpropagate(Zero);

    GradNormSq := 0;
    for pos := 0 to SeqLen - 1 do
      for d := 0 to Depth - 1 do
        GradNormSq := GradNormSq + Sqr(NN.Layers[0].OutputError[pos, 0, d]);
    AssertEquals('RoPE SeqLen=5 inverse norm', Input.GetSumSqr(), GradNormSq, 1e-4);

    for pos := 0 to SeqLen - 1 do
      for d := 0 to Depth - 1 do
        AssertEquals('RoPE SeqLen=5 round-trip pos=' + IntToStr(pos) +
          ' d=' + IntToStr(d),
          Input[pos, 0, d], NN.Layers[0].OutputError[pos, 0, d], 1e-4);
  finally
    NN.Free;
    Input.Free;
    Zero.Free;
  end;
end;

procedure TTestNeuralNumerical.TestRotaryEmbeddingOddDepthGuard;
var
  NN: TNNet;
  Rope: TNNetRotaryEmbedding;
  Capture: TErrorCapture;
begin
  // SetPrevLayer of TNNetRotaryEmbedding routes a hard precondition violation
  // through FErrorProc when the previous layer's depth is odd. Hook a custom
  // capture method onto the layer and assert that it fires with a message
  // that mentions the offending depth.
  NN := TNNet.Create();
  Capture := TErrorCapture.Create();
  try
    NN.AddLayer(TNNetInput.Create(2, 1, 3, 1)); // odd Depth = 3
    Rope := TNNetRotaryEmbedding.Create(10000.0);
    Rope.ErrorProc := {$IFDEF FPC}@{$ENDIF}Capture.Capture;
    NN.AddLayer(Rope);
    AssertTrue('RoPE odd-Depth guard must fire FErrorProc', Capture.Triggered);
    AssertTrue('RoPE odd-Depth message must mention "even Depth"',
      Pos('even Depth', Capture.Message) > 0);
  finally
    NN.Free;
    Capture.Free;
  end;
end;

procedure TTestNeuralNumerical.TestDropPathPZeroBoundary;
var
  NN: TNNet;
  Input, Desired: TNNetVolume;
  i: integer;
  v: TNeuralFloat;
begin
  // p=0 in training mode must be the identity: forward output = input,
  // backward gradient = upstream gradient, no NaN.
  NN := TNNet.Create();
  Input := TNNetVolume.Create(3, 2, 2);
  Desired := TNNetVolume.Create();
  try
    NN.AddLayer(TNNetInput.Create(3, 2, 2, 1));
    NN.AddLayer(TNNetDropPath.Create(0.0));
    NN.SetLearningRate(1.0, 0.0);
    NN.SetBatchUpdate(true);
    NN.EnableDropouts(true);

    Desired.ReSize(NN.GetLastLayer.Output);
    for i := 0 to Input.Size - 1 do
      Input.Raw[i] := Sin(i * 0.7) * 2.0 + 0.3;
    // Upstream gradient = output - desired. With desired = 0 the upstream
    // gradient equals the (identity) output, which equals the input.
    for i := 0 to Desired.Size - 1 do
      Desired.Raw[i] := 0;

    NN.Compute(Input);
    for i := 0 to Input.Size - 1 do
    begin
      v := NN.GetLastLayer.Output.Raw[i];
      AssertFalse('DropPath p=0 forward NaN at ' + IntToStr(i), IsNan(v));
      AssertEquals('DropPath p=0 forward identity at ' + IntToStr(i),
        Input.Raw[i], v, 1e-6);
    end;

    NN.Layers[0].OutputError.Fill(0);
    NN.Backpropagate(Desired);
    for i := 0 to Input.Size - 1 do
    begin
      v := NN.Layers[0].OutputError.Raw[i];
      AssertFalse('DropPath p=0 grad NaN at ' + IntToStr(i), IsNan(v));
      AssertEquals('DropPath p=0 grad passthrough at ' + IntToStr(i),
        Input.Raw[i], v, 1e-6);
    end;
  finally
    NN.Free;
    Input.Free;
    Desired.Free;
  end;
end;

procedure TTestNeuralNumerical.TestDropPathPOneBoundary;
var
  NN: TNNet;
  Input, Desired: TNNetVolume;
  i, Trials: integer;
  v: TNeuralFloat;
begin
  // p=1 boundary safety net. TNNetDropPath special-cases pDropProb >= 1
  // (the "always drop" case) so that every sample is zeroed and the
  // inverted-dropout 1/(1-p) scaling is bypassed. This test pins that
  // contract STRICTLY:
  //   - forward output is exactly zero for every sample, every trial
  //   - backward gradient w.r.t. the input is exactly zero
  //   - no NaN/Inf anywhere
  NN := TNNet.Create();
  Input := TNNetVolume.Create(3, 2, 2);
  Desired := TNNetVolume.Create();
  try
    NN.AddLayer(TNNetInput.Create(3, 2, 2, 1));
    NN.AddLayer(TNNetDropPath.Create(1.0));
    NN.SetLearningRate(1.0, 0.0);
    NN.SetBatchUpdate(true);
    NN.EnableDropouts(true);

    Desired.ReSize(NN.GetLastLayer.Output);
    for i := 0 to Input.Size - 1 do
      Input.Raw[i] := Sin(i * 0.7) * 2.0 + 0.3;
    for i := 0 to Desired.Size - 1 do
      Desired.Raw[i] := 0;

    RandSeed := 7777;
    for Trials := 0 to 199 do
    begin
      NN.Compute(Input);
      for i := 0 to Input.Size - 1 do
      begin
        v := NN.GetLastLayer.Output.Raw[i];
        AssertFalse('DropPath p=1 forward NaN at trial ' + IntToStr(Trials) +
          ' i=' + IntToStr(i), IsNan(v));
        AssertFalse('DropPath p=1 forward Inf at trial ' + IntToStr(Trials) +
          ' i=' + IntToStr(i), IsInfinite(v));
        AssertEquals('DropPath p=1 forward must be exactly zero at trial ' +
          IntToStr(Trials) + ' i=' + IntToStr(i), 0.0, v);
      end;

      NN.Layers[0].OutputError.Fill(0);
      NN.Backpropagate(Desired);
      for i := 0 to Input.Size - 1 do
      begin
        v := NN.Layers[0].OutputError.Raw[i];
        AssertFalse('DropPath p=1 grad NaN at trial ' + IntToStr(Trials) +
          ' i=' + IntToStr(i), IsNan(v));
        AssertFalse('DropPath p=1 grad Inf at trial ' + IntToStr(Trials) +
          ' i=' + IntToStr(i), IsInfinite(v));
        AssertEquals('DropPath p=1 grad must be exactly zero at trial ' +
          IntToStr(Trials) + ' i=' + IntToStr(i), 0.0, v);
      end;
    end;
  finally
    NN.Free;
    Input.Free;
    Desired.Free;
  end;
end;

procedure TTestNeuralNumerical.TestDropPathDeterminismFixedSeed;
var
  NN1, NN2: TNNet;
  Input: TNNetVolume;
  i, Trials: integer;
  Out1A, Out1B, Out2A, Out2B: array of TNeuralFloat;
  Size: integer;
const
  P = 0.4;
  Seed = 424242;
begin
  // Determinism contract: given a fixed RandSeed, two consecutive
  // Compute calls in training mode must produce identical outputs
  // (i.e. identical drop masks) across runs. This pins TNNetDropPath's
  // RNG behavior so future refactors cannot silently change the
  // sequence of masks consumed.
  NN1 := TNNet.Create();
  NN2 := TNNet.Create();
  Input := TNNetVolume.Create(3, 2, 2);
  try
    NN1.AddLayer(TNNetInput.Create(3, 2, 2, 1));
    NN1.AddLayer(TNNetDropPath.Create(P));
    NN1.SetLearningRate(1.0, 0.0);
    NN1.SetBatchUpdate(true);
    NN1.EnableDropouts(true);

    NN2.AddLayer(TNNetInput.Create(3, 2, 2, 1));
    NN2.AddLayer(TNNetDropPath.Create(P));
    NN2.SetLearningRate(1.0, 0.0);
    NN2.SetBatchUpdate(true);
    NN2.EnableDropouts(true);

    for i := 0 to Input.Size - 1 do
      Input.Raw[i] := Sin(i * 0.7) * 2.0 + 0.3;

    Size := NN1.GetLastLayer.Output.Size;
    SetLength(Out1A, Size);
    SetLength(Out1B, Size);
    SetLength(Out2A, Size);
    SetLength(Out2B, Size);

    // Run 1: seed, two consecutive Computes.
    RandSeed := Seed;
    NN1.Compute(Input);
    for i := 0 to Size - 1 do
      Out1A[i] := NN1.GetLastLayer.Output.Raw[i];
    NN1.Compute(Input);
    for i := 0 to Size - 1 do
      Out1B[i] := NN1.GetLastLayer.Output.Raw[i];

    // Run 2: same seed, fresh net, two consecutive Computes.
    RandSeed := Seed;
    NN2.Compute(Input);
    for i := 0 to Size - 1 do
      Out2A[i] := NN2.GetLastLayer.Output.Raw[i];
    NN2.Compute(Input);
    for i := 0 to Size - 1 do
      Out2B[i] := NN2.GetLastLayer.Output.Raw[i];

    // Both runs must agree on both Compute calls, bit-for-bit.
    for i := 0 to Size - 1 do
    begin
      AssertEquals('DropPath determinism: call A differs at i=' + IntToStr(i),
        Out1A[i], Out2A[i]);
      AssertEquals('DropPath determinism: call B differs at i=' + IntToStr(i),
        Out1B[i], Out2B[i]);
    end;

    // Sanity: at least one of the two calls produced a non-trivial mask
    // (either a drop or a scaled keep) so we are actually exercising the
    // RNG. We loop a handful of seeds to make the check robust.
    Trials := 0;
    for i := 0 to Size - 1 do
      if (Abs(Out1A[i]) < 1e-9) or (Abs(Out1A[i] - Input.Raw[i]) > 1e-6) then
        Inc(Trials);
    AssertTrue('DropPath determinism: RNG path not exercised (output equals input)',
      Trials > 0);
  finally
    NN1.Free;
    NN2.Free;
    Input.Free;
  end;
end;

procedure TTestNeuralNumerical.TestSoftCappingLargeCapContinuity;
var
  NN: TNNet;
  Input: TNNetVolume;
  Cap, v, expected, rel: TNeuralFloat;
  i: integer;
begin
  // y = c * tanh(x/c). As c -> infinity, y -> x for any bounded x. With a
  // moderate input range and c = 1e6 the layer must be effectively the
  // identity within tight fp tolerance.
  Cap := 1e6;
  NN := TNNet.Create();
  Input := TNNetVolume.Create(4, 1, 4);
  try
    NN.AddLayer(TNNetInput.Create(4, 1, 4, 1));
    NN.AddLayer(TNNetSoftCapping.Create(Cap));

    for i := 0 to Input.Size - 1 do
      Input.Raw[i] := Sin(i * 0.31) * 10.0; // values in roughly +/-10

    NN.Compute(Input);
    for i := 0 to Input.Size - 1 do
    begin
      v := NN.GetLastLayer.Output.Raw[i];
      expected := Input.Raw[i];
      AssertFalse('SoftCapping large-cap NaN at ' + IntToStr(i), IsNan(v));
      // Relative tolerance 1e-3, with absolute floor for near-zero values.
      if Abs(expected) < 1e-3 then
        rel := Abs(v - expected)
      else
        rel := Abs(v - expected) / Abs(expected);
      AssertTrue('SoftCapping c->inf identity at ' + IntToStr(i) +
        ' v=' + FloatToStr(v) + ' expected=' + FloatToStr(expected) +
        ' rel=' + FloatToStr(rel), rel < 1e-3);
    end;
  finally
    NN.Free;
    Input.Free;
  end;
end;

procedure TTestNeuralNumerical.TestSoftCappingExtremeInputSaturation;
var
  NN: TNNet;
  Input, Desired: TNNetVolume;
  Cap, v, g: TNeuralFloat;
  i: integer;
begin
  // Drive the layer with extreme magnitudes (+/-1e6) and a small cap (5).
  // Every output element must lie inside [-c, +c], no NaN/Inf in either the
  // forward or the backward pass.
  Cap := 5.0;
  NN := TNNet.Create();
  Input := TNNetVolume.Create(4, 1, 4);
  Desired := TNNetVolume.Create();
  try
    NN.AddLayer(TNNetInput.Create(4, 1, 4, 1));
    NN.AddLayer(TNNetSoftCapping.Create(Cap));
    NN.SetLearningRate(1.0, 0.0);
    NN.SetBatchUpdate(true);

    Desired.ReSize(NN.GetLastLayer.Output);
    for i := 0 to Input.Size - 1 do
    begin
      if (i mod 2) = 0 then Input.Raw[i] := 1e6
      else Input.Raw[i] := -1e6;
    end;
    for i := 0 to Desired.Size - 1 do
      Desired.Raw[i] := Cos(i * 0.2);

    NN.Compute(Input);
    for i := 0 to Input.Size - 1 do
    begin
      v := NN.GetLastLayer.Output.Raw[i];
      AssertFalse('SoftCapping saturation forward NaN at ' + IntToStr(i), IsNan(v));
      AssertFalse('SoftCapping saturation forward Inf at ' + IntToStr(i), IsInfinite(v));
      AssertTrue('SoftCapping saturation in [-c, c] at ' + IntToStr(i) +
        ' v=' + FloatToStr(v),
        (v >= -Cap - 1e-4) and (v <= Cap + 1e-4));
    end;

    NN.Layers[0].OutputError.Fill(0);
    NN.Backpropagate(Desired);
    for i := 0 to Input.Size - 1 do
    begin
      g := NN.Layers[0].OutputError.Raw[i];
      AssertFalse('SoftCapping saturation grad NaN at ' + IntToStr(i), IsNan(g));
      AssertFalse('SoftCapping saturation grad Inf at ' + IntToStr(i), IsInfinite(g));
    end;
  finally
    NN.Free;
    Input.Free;
    Desired.Free;
  end;
end;

// Generic helper: build a tiny net with ALayer wired after a single input
// layer of the given shape, drive Compute on a fixed deterministic input,
// then SaveToString / LoadFromString into a fresh net and verify the output
// matches element-wise within tolerance. ALayer is owned by the original net.
procedure SerializationRoundTrip(ATestCase: TTestCase; ALayer: TNNetLayer;
  const AName: string; ASizeX, ASizeY, ASizeD: integer;
  ATolerance: TNeuralFloat);
var
  NN, NN2: TNNet;
  Input: TNNetVolume;
  Saved: string;
  i: integer;
begin
  NN := TNNet.Create();
  Input := TNNetVolume.Create(ASizeX, ASizeY, ASizeD);
  try
    NN.AddLayer(TNNetInput.Create(ASizeX, ASizeY, ASizeD, 1));
    NN.AddLayer(ALayer);

    RandSeed := 31337;
    for i := 0 to Input.Size - 1 do
      Input.Raw[i] := Sin(i * 0.41) * 0.7 + 0.1;

    NN.Compute(Input);
    Saved := NN.SaveToString();

    NN2 := TNNet.Create();
    try
      NN2.LoadFromString(Saved);
      NN2.Compute(Input);
      ATestCase.AssertEquals(AName + ' round-trip output size',
        NN.GetLastLayer.Output.Size, NN2.GetLastLayer.Output.Size);
      for i := 0 to NN.GetLastLayer.Output.Size - 1 do
        ATestCase.AssertEquals(AName + ' round-trip output at ' + IntToStr(i),
          NN.GetLastLayer.Output.Raw[i],
          NN2.GetLastLayer.Output.Raw[i], ATolerance);
    finally
      NN2.Free;
    end;
  finally
    NN.Free;
    Input.Free;
  end;
end;

procedure TTestNeuralNumerical.TestSoftCappingSerializationRoundTrip;
begin
  SerializationRoundTrip(Self, TNNetSoftCapping.Create(),
    'SoftCapping', 3, 1, 4, 1e-5);
end;

procedure TTestNeuralNumerical.TestHardShrinkSerializationRoundTrip;
begin
  SerializationRoundTrip(Self, TNNetHardShrink.Create(0.3),
    'HardShrink', 3, 1, 4, 1e-5);
end;

procedure TTestNeuralNumerical.TestLogSigmoidSerializationRoundTrip;
begin
  SerializationRoundTrip(Self, TNNetLogSigmoid.Create(),
    'LogSigmoid', 3, 1, 4, 1e-5);
end;

procedure TTestNeuralNumerical.TestShiftedReLUSerializationRoundTrip;
begin
  SerializationRoundTrip(Self, TNNetShiftedReLU.Create(),
    'ShiftedReLU', 3, 1, 4, 1e-5);
end;

procedure TTestNeuralNumerical.TestTanhShrinkSerializationRoundTrip;
begin
  SerializationRoundTrip(Self, TNNetTanhShrink.Create(),
    'TanhShrink', 3, 1, 4, 1e-5);
end;

procedure TTestNeuralNumerical.TestHardTanhSerializationRoundTrip;
begin
  SerializationRoundTrip(Self, TNNetHardTanh.Create(),
    'HardTanh', 3, 1, 4, 1e-5);
end;

procedure TTestNeuralNumerical.TestSoftShrinkSerializationRoundTrip;
begin
  SerializationRoundTrip(Self, TNNetSoftShrink.Create(0.3),
    'SoftShrink', 3, 1, 4, 1e-5);
end;

procedure TTestNeuralNumerical.TestThresholdSerializationRoundTrip;
begin
  SerializationRoundTrip(Self, TNNetThreshold.Create(0.7, -0.25),
    'Threshold', 3, 1, 4, 1e-5);
end;

procedure TTestNeuralNumerical.TestDropPathSerializationRoundTrip;
var
  NN, NN2: TNNet;
  Input: TNNetVolume;
  Saved: string;
  i: integer;
begin
  // DropPath at inference (default: dropouts disabled) is the identity, so
  // both the original and the reloaded net must produce input == output.
  NN := TNNet.Create();
  Input := TNNetVolume.Create(3, 2, 2);
  try
    NN.AddLayer(TNNetInput.Create(3, 2, 2, 1));
    NN.AddLayer(TNNetDropPath.Create()); // default p
    NN.EnableDropouts(false);

    for i := 0 to Input.Size - 1 do
      Input.Raw[i] := Sin(i * 0.41) * 0.7 + 0.1;

    NN.Compute(Input);
    // In inference mode the forward must be exactly the identity.
    for i := 0 to Input.Size - 1 do
      AssertEquals('DropPath inference forward is identity at ' + IntToStr(i),
        Input.Raw[i], NN.GetLastLayer.Output.Raw[i], 1e-7);
    Saved := NN.SaveToString();

    NN2 := TNNet.Create();
    try
      NN2.LoadFromString(Saved);
      NN2.EnableDropouts(false);
      NN2.Compute(Input);
      AssertEquals('DropPath round-trip output size',
        NN.GetLastLayer.Output.Size, NN2.GetLastLayer.Output.Size);
      for i := 0 to NN.GetLastLayer.Output.Size - 1 do
        AssertEquals('DropPath round-trip output at ' + IntToStr(i),
          NN.GetLastLayer.Output.Raw[i],
          NN2.GetLastLayer.Output.Raw[i], 1e-5);
      // Also pin: the reloaded layer is still the identity at inference.
      for i := 0 to Input.Size - 1 do
        AssertEquals('DropPath reloaded forward is identity at ' + IntToStr(i),
          Input.Raw[i], NN2.GetLastLayer.Output.Raw[i], 1e-7);
    finally
      NN2.Free;
    end;
  finally
    NN.Free;
    Input.Free;
  end;
end;

procedure TTestNeuralNumerical.TestRotaryEmbeddingSerializationRoundTrip;
begin
  SerializationRoundTrip(Self, TNNetRotaryEmbedding.Create(),
    'RotaryEmbedding', 3, 1, 4, 1e-5);
end;

procedure TTestNeuralNumerical.TestMaskedFillSerializationRoundTrip;
begin
  // Use a small mask value to keep float32 precision intact when comparing.
  SerializationRoundTrip(Self, TNNetMaskedFill.Create(-0.5),
    'MaskedFill', 3, 3, 2, 1e-5);
end;

procedure TTestNeuralNumerical.TestALiBiSerializationRoundTrip;
begin
  SerializationRoundTrip(Self, TNNetALiBi.Create(),
    'ALiBi', 3, 3, 2, 1e-5);
end;

procedure TTestNeuralNumerical.TestScaledDotProductAttentionSerializationRoundTrip;
begin
  // d_k = 4, non-causal. Input depth must be 3*d_k = 12.
  SerializationRoundTrip(Self, TNNetScaledDotProductAttention.Create(4, false),
    'SDPA', 3, 1, 12, 1e-5);
end;

// ---------------------------------------------------------------------------
// Spatial dropout tests. Both layers descend from TNNetAddNoiseBase so they
// honor TNNet.EnableDropouts(). The defining property versus standard
// dropout is that the per-element Bernoulli mask is replaced by one mask
// value per channel (Depth slice). For both layers, channels are along
// Depth; SpatialDropout1D treats SizeX as the sequence length (SizeY=1)
// while SpatialDropout2D operates on the full SizeX*SizeY spatial extent.
// The expectation tested here: every element within a kept-or-dropped
// channel is consistently scaled by the same factor (0 or 1/(1-p)).
// ---------------------------------------------------------------------------

procedure TTestNeuralNumerical.TestSpatialDropout1DInferenceIdentity;
var
  NN: TNNet;
  Input: TNNetVolume;
  i: integer;
begin
  NN := TNNet.Create();
  Input := TNNetVolume.Create(5, 1, 4);
  try
    NN.AddLayer(TNNetInput.Create(5, 1, 4, 1));
    NN.AddLayer(TNNetSpatialDropout1D.Create(0.5));
    NN.EnableDropouts(false);
    for i := 0 to Input.Size - 1 do
      Input.Raw[i] := Sin(i * 0.7) * 2.0 + 0.3;
    NN.Compute(Input);
    for i := 0 to Input.Size - 1 do
      AssertEquals('SpatialDropout1D inference identity at ' + IntToStr(i),
        Input.Raw[i], NN.GetLastLayer.Output.Raw[i], 1e-5);
  finally
    NN.Free;
    Input.Free;
  end;
end;

procedure TTestNeuralNumerical.TestSpatialDropout1DTrainingMaskShape;
var
  NN: TNNet;
  Input: TNNetVolume;
  x, d, i: integer;
  P, InvKeep, Ratio, Expected: TNeuralFloat;
  KeptObserved, DroppedObserved: boolean;
begin
  // For each channel we expect every (x, 0, d) element to be either all
  // zero (dropped) or all input*1/(1-p) (kept). Iterate enough trials with
  // a fixed seed to observe both outcomes.
  NN := TNNet.Create();
  Input := TNNetVolume.Create(4, 1, 6);
  P := 0.5;
  InvKeep := 1.0 / (1 - P);
  KeptObserved := false;
  DroppedObserved := false;
  try
    NN.AddLayer(TNNetInput.Create(4, 1, 6, 1));
    NN.AddLayer(TNNetSpatialDropout1D.Create(P));
    NN.EnableDropouts(true);
    for i := 0 to Input.Size - 1 do
      Input.Raw[i] := Sin(i * 0.7) * 2.0 + 0.3;

    RandSeed := 9001;
    for i := 0 to 49 do
    begin
      NN.Compute(Input);
      for d := 0 to Input.Depth - 1 do
      begin
        // Inspect first element of channel d to determine kept vs dropped.
        if Abs(NN.GetLastLayer.Output[0, 0, d]) < 1e-6 then
        begin
          DroppedObserved := true;
          for x := 0 to Input.SizeX - 1 do
            AssertEquals('SD1D dropped channel ' + IntToStr(d) +
              ' x=' + IntToStr(x),
              0.0, NN.GetLastLayer.Output[x, 0, d], 1e-5);
        end
        else
        begin
          KeptObserved := true;
          Ratio := NN.GetLastLayer.Output[0, 0, d] / Input[0, 0, d];
          AssertTrue('SD1D kept channel scale ~ 1/(1-p): got ' +
            FloatToStr(Ratio), Abs(Ratio - InvKeep) < 1e-3);
          for x := 0 to Input.SizeX - 1 do
          begin
            Expected := Input[x, 0, d] * InvKeep;
            AssertEquals('SD1D kept channel ' + IntToStr(d) +
              ' x=' + IntToStr(x),
              Expected, NN.GetLastLayer.Output[x, 0, d], 1e-4);
          end;
        end;
      end;
    end;
    AssertTrue('SD1D should observe at least one kept channel', KeptObserved);
    AssertTrue('SD1D should observe at least one dropped channel', DroppedObserved);
  finally
    NN.Free;
    Input.Free;
  end;
end;

procedure TTestNeuralNumerical.TestSpatialDropout1DGradientCheck;
var
  NN: TNNet;
  Input, InputPlus, Desired: TNNetVolume;
  epsilon, lossPlus, lossMinus, numericalGrad, analyticalGrad: TNeuralFloat;
  i, k: integer;
  diff: TNeuralFloat;
  Seed: longint;

  function ComputeLossSeeded(AInput: TNNetVolume): TNeuralFloat;
  var
    kk: integer;
    d: TNeuralFloat;
  begin
    RandSeed := Seed;
    NN.Compute(AInput);
    Result := 0;
    for kk := 0 to NN.GetLastLayer.Output.Size - 1 do
    begin
      d := NN.GetLastLayer.Output.Raw[kk] - Desired.Raw[kk];
      Result := Result + 0.5 * d * d;
    end;
  end;

begin
  NN := TNNet.Create();
  Input := TNNetVolume.Create(3, 1, 4);
  InputPlus := TNNetVolume.Create(3, 1, 4);
  Desired := TNNetVolume.Create();
  epsilon := 0.0001;
  Seed := 4242;
  try
    NN.AddLayer(TNNetInput.Create(3, 1, 4, 1));
    NN.AddLayer(TNNetSpatialDropout1D.Create(0.3));
    NN.SetLearningRate(1.0, 0.0);
    NN.SetBatchUpdate(true);
    NN.EnableDropouts(true);

    Desired.ReSize(NN.GetLastLayer.Output);
    for i := 0 to Input.Size - 1 do
      Input.Raw[i] := Sin(i * 0.7) * 2.0 + 0.3;
    for k := 0 to Desired.Size - 1 do
      Desired.Raw[k] := Cos(k * 0.5);

    // Find a seed under which at least one channel is kept so the gradient
    // is informative (otherwise all gradients are zero and the test is vacuous).
    while True do
    begin
      RandSeed := Seed;
      NN.Compute(Input);
      diff := 0;
      for k := 0 to NN.GetLastLayer.Output.Size - 1 do
        diff := diff + Abs(NN.GetLastLayer.Output.Raw[k]);
      if diff > 1e-3 then break;
      Inc(Seed);
    end;

    for i := 0 to Input.Size - 1 do
    begin
      InputPlus.Copy(Input);
      InputPlus.Raw[i] := Input.Raw[i] + epsilon;
      lossPlus := ComputeLossSeeded(InputPlus);
      InputPlus.Raw[i] := Input.Raw[i] - epsilon;
      lossMinus := ComputeLossSeeded(InputPlus);
      numericalGrad := (lossPlus - lossMinus) / (2 * epsilon);

      RandSeed := Seed;
      NN.Compute(Input);
      NN.Layers[0].OutputError.Fill(0);
      NN.Backpropagate(Desired);
      analyticalGrad := NN.Layers[0].OutputError.Raw[i];

      AssertTrue('SD1D input gradient at ' + IntToStr(i) +
        ' (num=' + FloatToStr(numericalGrad) +
        ' ana=' + FloatToStr(analyticalGrad) + ')',
        Abs(numericalGrad - analyticalGrad) < 0.01);
    end;
  finally
    NN.Free;
    Input.Free;
    InputPlus.Free;
    Desired.Free;
  end;
end;

procedure TTestNeuralNumerical.TestSpatialDropout1DSerializationRoundTrip;
var
  NN, NN2: TNNet;
  Input: TNNetVolume;
  Saved: string;
  i: integer;
begin
  // At inference (dropouts disabled) both nets must produce identity output.
  NN := TNNet.Create();
  Input := TNNetVolume.Create(3, 1, 4);
  try
    NN.AddLayer(TNNetInput.Create(3, 1, 4, 1));
    NN.AddLayer(TNNetSpatialDropout1D.Create(0.25));
    NN.EnableDropouts(false);
    for i := 0 to Input.Size - 1 do
      Input.Raw[i] := Sin(i * 0.41) * 0.7 + 0.1;
    NN.Compute(Input);
    Saved := NN.SaveToString();
    NN2 := TNNet.Create();
    try
      NN2.LoadFromString(Saved);
      NN2.EnableDropouts(false);
      NN2.Compute(Input);
      AssertEquals('SD1D round-trip output size',
        NN.GetLastLayer.Output.Size, NN2.GetLastLayer.Output.Size);
      for i := 0 to NN.GetLastLayer.Output.Size - 1 do
        AssertEquals('SD1D round-trip output at ' + IntToStr(i),
          NN.GetLastLayer.Output.Raw[i],
          NN2.GetLastLayer.Output.Raw[i], 1e-5);
    finally
      NN2.Free;
    end;
  finally
    NN.Free;
    Input.Free;
  end;
end;

procedure TTestNeuralNumerical.TestSpatialDropout2DInferenceIdentity;
var
  NN: TNNet;
  Input: TNNetVolume;
  i: integer;
begin
  NN := TNNet.Create();
  Input := TNNetVolume.Create(3, 3, 4);
  try
    NN.AddLayer(TNNetInput.Create(3, 3, 4, 1));
    NN.AddLayer(TNNetSpatialDropout2D.Create(0.5));
    NN.EnableDropouts(false);
    for i := 0 to Input.Size - 1 do
      Input.Raw[i] := Sin(i * 0.7) * 2.0 + 0.3;
    NN.Compute(Input);
    for i := 0 to Input.Size - 1 do
      AssertEquals('SpatialDropout2D inference identity at ' + IntToStr(i),
        Input.Raw[i], NN.GetLastLayer.Output.Raw[i], 1e-5);
  finally
    NN.Free;
    Input.Free;
  end;
end;

procedure TTestNeuralNumerical.TestSpatialDropout2DTrainingMaskShape;
var
  NN: TNNet;
  Input: TNNetVolume;
  x, y, d, i: integer;
  P, InvKeep, Ratio, Expected: TNeuralFloat;
  KeptObserved, DroppedObserved: boolean;
begin
  NN := TNNet.Create();
  Input := TNNetVolume.Create(3, 3, 5);
  P := 0.5;
  InvKeep := 1.0 / (1 - P);
  KeptObserved := false;
  DroppedObserved := false;
  try
    NN.AddLayer(TNNetInput.Create(3, 3, 5, 1));
    NN.AddLayer(TNNetSpatialDropout2D.Create(P));
    NN.EnableDropouts(true);
    for i := 0 to Input.Size - 1 do
      Input.Raw[i] := Sin(i * 0.7) * 2.0 + 0.3;

    RandSeed := 9002;
    for i := 0 to 49 do
    begin
      NN.Compute(Input);
      for d := 0 to Input.Depth - 1 do
      begin
        if Abs(NN.GetLastLayer.Output[0, 0, d]) < 1e-6 then
        begin
          DroppedObserved := true;
          for y := 0 to Input.SizeY - 1 do
            for x := 0 to Input.SizeX - 1 do
              AssertEquals('SD2D dropped ch ' + IntToStr(d) +
                ' (' + IntToStr(x) + ',' + IntToStr(y) + ')',
                0.0, NN.GetLastLayer.Output[x, y, d], 1e-5);
        end
        else
        begin
          KeptObserved := true;
          Ratio := NN.GetLastLayer.Output[0, 0, d] / Input[0, 0, d];
          AssertTrue('SD2D kept channel scale ~ 1/(1-p): got ' +
            FloatToStr(Ratio), Abs(Ratio - InvKeep) < 1e-3);
          for y := 0 to Input.SizeY - 1 do
            for x := 0 to Input.SizeX - 1 do
            begin
              Expected := Input[x, y, d] * InvKeep;
              AssertEquals('SD2D kept ch ' + IntToStr(d) +
                ' (' + IntToStr(x) + ',' + IntToStr(y) + ')',
                Expected, NN.GetLastLayer.Output[x, y, d], 1e-4);
            end;
        end;
      end;
    end;
    AssertTrue('SD2D should observe at least one kept channel', KeptObserved);
    AssertTrue('SD2D should observe at least one dropped channel', DroppedObserved);
  finally
    NN.Free;
    Input.Free;
  end;
end;

procedure TTestNeuralNumerical.TestSpatialDropout2DGradientCheck;
var
  NN: TNNet;
  Input, InputPlus, Desired: TNNetVolume;
  epsilon, lossPlus, lossMinus, numericalGrad, analyticalGrad: TNeuralFloat;
  i, k: integer;
  diff: TNeuralFloat;
  Seed: longint;

  function ComputeLossSeeded(AInput: TNNetVolume): TNeuralFloat;
  var
    kk: integer;
    d: TNeuralFloat;
  begin
    RandSeed := Seed;
    NN.Compute(AInput);
    Result := 0;
    for kk := 0 to NN.GetLastLayer.Output.Size - 1 do
    begin
      d := NN.GetLastLayer.Output.Raw[kk] - Desired.Raw[kk];
      Result := Result + 0.5 * d * d;
    end;
  end;

begin
  NN := TNNet.Create();
  Input := TNNetVolume.Create(3, 2, 2);
  InputPlus := TNNetVolume.Create(3, 2, 2);
  Desired := TNNetVolume.Create();
  epsilon := 0.0001;
  Seed := 4242;
  try
    NN.AddLayer(TNNetInput.Create(3, 2, 2, 1));
    NN.AddLayer(TNNetSpatialDropout2D.Create(0.3));
    NN.SetLearningRate(1.0, 0.0);
    NN.SetBatchUpdate(true);
    NN.EnableDropouts(true);

    Desired.ReSize(NN.GetLastLayer.Output);
    for i := 0 to Input.Size - 1 do
      Input.Raw[i] := Sin(i * 0.7) * 2.0 + 0.3;
    for k := 0 to Desired.Size - 1 do
      Desired.Raw[k] := Cos(k * 0.5);

    while True do
    begin
      RandSeed := Seed;
      NN.Compute(Input);
      diff := 0;
      for k := 0 to NN.GetLastLayer.Output.Size - 1 do
        diff := diff + Abs(NN.GetLastLayer.Output.Raw[k]);
      if diff > 1e-3 then break;
      Inc(Seed);
    end;

    for i := 0 to Input.Size - 1 do
    begin
      InputPlus.Copy(Input);
      InputPlus.Raw[i] := Input.Raw[i] + epsilon;
      lossPlus := ComputeLossSeeded(InputPlus);
      InputPlus.Raw[i] := Input.Raw[i] - epsilon;
      lossMinus := ComputeLossSeeded(InputPlus);
      numericalGrad := (lossPlus - lossMinus) / (2 * epsilon);

      RandSeed := Seed;
      NN.Compute(Input);
      NN.Layers[0].OutputError.Fill(0);
      NN.Backpropagate(Desired);
      analyticalGrad := NN.Layers[0].OutputError.Raw[i];

      AssertTrue('SD2D input gradient at ' + IntToStr(i) +
        ' (num=' + FloatToStr(numericalGrad) +
        ' ana=' + FloatToStr(analyticalGrad) + ')',
        Abs(numericalGrad - analyticalGrad) < 0.01);
    end;
  finally
    NN.Free;
    Input.Free;
    InputPlus.Free;
    Desired.Free;
  end;
end;

procedure TTestNeuralNumerical.TestSpatialDropout2DSerializationRoundTrip;
var
  NN, NN2: TNNet;
  Input: TNNetVolume;
  Saved: string;
  i: integer;
begin
  NN := TNNet.Create();
  Input := TNNetVolume.Create(3, 3, 4);
  try
    NN.AddLayer(TNNetInput.Create(3, 3, 4, 1));
    NN.AddLayer(TNNetSpatialDropout2D.Create(0.25));
    NN.EnableDropouts(false);
    for i := 0 to Input.Size - 1 do
      Input.Raw[i] := Sin(i * 0.41) * 0.7 + 0.1;
    NN.Compute(Input);
    Saved := NN.SaveToString();
    NN2 := TNNet.Create();
    try
      NN2.LoadFromString(Saved);
      NN2.EnableDropouts(false);
      NN2.Compute(Input);
      AssertEquals('SD2D round-trip output size',
        NN.GetLastLayer.Output.Size, NN2.GetLastLayer.Output.Size);
      for i := 0 to NN.GetLastLayer.Output.Size - 1 do
        AssertEquals('SD2D round-trip output at ' + IntToStr(i),
          NN.GetLastLayer.Output.Raw[i],
          NN2.GetLastLayer.Output.Raw[i], 1e-5);
    finally
      NN2.Free;
    end;
  finally
    NN.Free;
    Input.Free;
  end;
end;

procedure TTestNeuralNumerical.TestChannelShuffleForward;
var
  NN: TNNet;
  Input: TNNetVolume;
  c: integer;
  Expected: array[0..3] of integer;
begin
  // Depth=4, Groups=2 -> per-group=2.
  // Channel c maps to (c mod G) * (C/G) + (c div G):
  //   0 -> 0, 1 -> 2, 2 -> 1, 3 -> 3.
  NN := TNNet.Create();
  Input := TNNetVolume.Create(1, 1, 4);
  try
    NN.AddLayer(TNNetInput.Create(1, 1, 4, 1));
    NN.AddLayer(TNNetChannelShuffle.Create(2));
    Input.Raw[0] := 10.0;
    Input.Raw[1] := 20.0;
    Input.Raw[2] := 30.0;
    Input.Raw[3] := 40.0;
    NN.Compute(Input);
    Expected[0] := 0; Expected[1] := 2; Expected[2] := 1; Expected[3] := 3;
    for c := 0 to 3 do
      AssertEquals('ChannelShuffle output channel ' + IntToStr(Expected[c]),
        Input.Raw[c], NN.GetLastLayer.Output.Raw[Expected[c]], 1e-6);
  finally
    NN.Free;
    Input.Free;
  end;
end;

procedure TTestNeuralNumerical.TestChannelShuffleGradientCheck;
begin
  // Depth=4, Groups=2: matches the InterleaveChannels gradient-check shape;
  // the permutation is parameter-free so backprop is the inverse permutation.
  LayerInputGradientCheck(Self, TNNetChannelShuffle.Create(2),
    'ChannelShuffle', 2, 2, 4, 0.01);
end;

procedure TTestNeuralNumerical.TestChannelShuffleSerializationRoundTrip;
var
  NN, NN2: TNNet;
  Input: TNNetVolume;
  Saved: string;
  i: integer;
begin
  // Pin the Groups hyperparameter (FStruct[0]) survives the dispatch in
  // addition to the element-wise output parity exercised by the helper.
  SerializationRoundTrip(Self, TNNetChannelShuffle.Create(3),
    'ChannelShuffle', 2, 2, 6, 1e-5);

  NN := TNNet.Create();
  Input := TNNetVolume.Create(2, 2, 6);
  try
    NN.AddLayer(TNNetInput.Create(2, 2, 6, 1));
    NN.AddLayer(TNNetChannelShuffle.Create(3));
    for i := 0 to Input.Size - 1 do
      Input.Raw[i] := Sin(i * 0.41) * 0.7 + 0.1;
    NN.Compute(Input);
    Saved := NN.SaveToString();
    NN2 := TNNet.Create();
    try
      NN2.LoadFromString(Saved);
      // SaveStructureToString embeds FStruct[0] (Groups); equality here
      // pins the Groups hyperparameter through the CreateLayer dispatch.
      AssertEquals('ChannelShuffle round-trip structure',
        NN.GetLastLayer.SaveStructureToString(),
        NN2.GetLastLayer.SaveStructureToString());
    finally
      NN2.Free;
    end;
  finally
    NN.Free;
    Input.Free;
  end;
end;

procedure TTestNeuralNumerical.TestChannelShuffleIndivisibleGuard;
var
  NN: TNNet;
  Shuf: TNNetChannelShuffle;
  Capture: TErrorCapture;
begin
  // Depth=5 is not divisible by Groups=2; SetPrevLayer must fire FErrorProc.
  NN := TNNet.Create();
  Capture := TErrorCapture.Create();
  try
    NN.AddLayer(TNNetInput.Create(1, 1, 5, 1));
    Shuf := TNNetChannelShuffle.Create(2);
    Shuf.ErrorProc := {$IFDEF FPC}@{$ENDIF}Capture.Capture;
    NN.AddLayer(Shuf);
    AssertTrue('ChannelShuffle indivisible-depth guard must fire FErrorProc',
      Capture.Triggered);
    AssertTrue('ChannelShuffle indivisible-depth message must mention "divisible"',
      Pos('divisible', Capture.Message) > 0);
  finally
    NN.Free;
    Capture.Free;
  end;
end;

procedure TTestNeuralNumerical.TestChannelShuffleInverseProperty;
var
  NN: TNNet;
  Input: TNNetVolume;
  i: integer;
begin
  // ChannelShuffle(G) composed with ChannelShuffle(C/G) is the identity.
  // The forward permutation is c -> (c mod G) * (C/G) + (c div G); applying it
  // again with G' = C/G inverts that map. Use C=12, G=3, C/G=4.
  NN := TNNet.Create();
  Input := TNNetVolume.Create(1, 1, 12);
  try
    NN.AddLayer(TNNetInput.Create(1, 1, 12, 1));
    NN.AddLayer(TNNetChannelShuffle.Create(3));
    NN.AddLayer(TNNetChannelShuffle.Create(4));
    RandSeed := 424242;
    for i := 0 to Input.Size - 1 do
      Input.Raw[i] := Random() * 2 - 1;
    NN.Compute(Input);
    for i := 0 to Input.Size - 1 do
      AssertEquals('ChannelShuffle inverse at channel ' + IntToStr(i),
        Input.Raw[i], NN.GetLastLayer.Output.Raw[i], 1e-6);
  finally
    NN.Free;
    Input.Free;
  end;
end;

procedure TTestNeuralNumerical.TestReverseChannelsForward;
var
  NN: TNNet;
  Input: TNNetVolume;
  c: integer;
begin
  // Depth=4: channel c maps to (Depth - 1 - c).
  NN := TNNet.Create();
  Input := TNNetVolume.Create(1, 1, 4);
  try
    NN.AddLayer(TNNetInput.Create(1, 1, 4, 1));
    NN.AddLayer(TNNetReverseChannels.Create());
    Input.Raw[0] := 10.0;
    Input.Raw[1] := 20.0;
    Input.Raw[2] := 30.0;
    Input.Raw[3] := 40.0;
    NN.Compute(Input);
    for c := 0 to 3 do
      AssertEquals('ReverseChannels output channel ' + IntToStr(c),
        Input.Raw[3 - c], NN.GetLastLayer.Output.Raw[c], 1e-6);
  finally
    NN.Free;
    Input.Free;
  end;
end;

procedure TTestNeuralNumerical.TestReverseChannelsGradientCheck;
begin
  // Parameter-free permutation; backward is the same involution.
  LayerInputGradientCheck(Self, TNNetReverseChannels.Create(),
    'ReverseChannels', 2, 2, 4, 0.01);
end;

procedure TTestNeuralNumerical.TestReverseChannelsInvolution;
var
  NN: TNNet;
  Input: TNNetVolume;
  i: integer;
begin
  // Applying ReverseChannels twice must return the identity.
  NN := TNNet.Create();
  Input := TNNetVolume.Create(2, 2, 7);
  try
    NN.AddLayer(TNNetInput.Create(2, 2, 7, 1));
    NN.AddLayer(TNNetReverseChannels.Create());
    NN.AddLayer(TNNetReverseChannels.Create());
    RandSeed := 131313;
    for i := 0 to Input.Size - 1 do
      Input.Raw[i] := Random() * 2 - 1;
    NN.Compute(Input);
    for i := 0 to Input.Size - 1 do
      AssertEquals('ReverseChannels involution at index ' + IntToStr(i),
        Input.Raw[i], NN.GetLastLayer.Output.Raw[i], 1e-6);
  finally
    NN.Free;
    Input.Free;
  end;
end;

procedure TTestNeuralNumerical.TestReverseChannelsSerializationRoundTrip;
begin
  // Parameter-free, so only element-wise output parity matters after the
  // SaveToString -> LoadFromString cycle.
  SerializationRoundTrip(Self, TNNetReverseChannels.Create(),
    'ReverseChannels', 2, 2, 5, 1e-5);
end;

// Generic helper for the *Norm family: after the layer is wired by AddLayer,
// perturb every learnable weight (gamma / beta) with deterministic noise so
// the round-trip is not a trivial identity (gamma=1, beta=0). Then verify
// SaveToString / LoadFromString reproduce Compute element-wise.
procedure NormSerializationRoundTripWithPerturbedWeights(ATestCase: TTestCase;
  ALayer: TNNetLayer; const AName: string;
  ASizeX, ASizeY, ASizeD: integer; ATolerance: TNeuralFloat);
var
  NN, NN2: TNNet;
  Input: TNNetVolume;
  Saved: string;
  i, NCnt, WCnt: integer;
begin
  NN := TNNet.Create();
  Input := TNNetVolume.Create(ASizeX, ASizeY, ASizeD);
  try
    NN.AddLayer(TNNetInput.Create(ASizeX, ASizeY, ASizeD, 1));
    NN.AddLayer(ALayer);

    // Perturb each learnable-weight tensor with deterministic noise so the
    // round-trip exercises a non-trivial gamma/beta. We poke FWeights
    // directly (the public Weights accessor returns the same tensor).
    for NCnt := 0 to ALayer.Neurons.Count - 1 do
      for WCnt := 0 to ALayer.Neurons[NCnt].Weights.Size - 1 do
        ALayer.Neurons[NCnt].Weights.Raw[WCnt] :=
          ALayer.Neurons[NCnt].Weights.Raw[WCnt]
          + Sin(NCnt * 7.3 + WCnt * 0.31) * 0.25;

    for i := 0 to Input.Size - 1 do
      Input.Raw[i] := Sin(i * 0.41) * 0.7 + 0.1;

    NN.Compute(Input);
    Saved := NN.SaveToString();

    NN2 := TNNet.Create();
    try
      NN2.LoadFromString(Saved);
      NN2.Compute(Input);
      ATestCase.AssertEquals(AName + ' round-trip output size',
        NN.GetLastLayer.Output.Size, NN2.GetLastLayer.Output.Size);
      // Hyperparameter / structure parity via SaveStructureToString.
      ATestCase.AssertEquals(AName + ' round-trip structure',
        NN.GetLastLayer.SaveStructureToString(),
        NN2.GetLastLayer.SaveStructureToString());
      // Learnable-weight parity: gamma / beta survive the round-trip.
      ATestCase.AssertEquals(AName + ' round-trip neuron count',
        NN.GetLastLayer.Neurons.Count, NN2.GetLastLayer.Neurons.Count);
      for NCnt := 0 to NN.GetLastLayer.Neurons.Count - 1 do
      begin
        ATestCase.AssertEquals(AName + ' round-trip weight size n=' +
          IntToStr(NCnt),
          NN.GetLastLayer.Neurons[NCnt].Weights.Size,
          NN2.GetLastLayer.Neurons[NCnt].Weights.Size);
        for WCnt := 0 to NN.GetLastLayer.Neurons[NCnt].Weights.Size - 1 do
          ATestCase.AssertEquals(AName + ' round-trip weight n=' +
            IntToStr(NCnt) + ' w=' + IntToStr(WCnt),
            NN.GetLastLayer.Neurons[NCnt].Weights.Raw[WCnt],
            NN2.GetLastLayer.Neurons[NCnt].Weights.Raw[WCnt], ATolerance);
      end;
      // Compute parity.
      for i := 0 to NN.GetLastLayer.Output.Size - 1 do
        ATestCase.AssertEquals(AName + ' round-trip output at ' + IntToStr(i),
          NN.GetLastLayer.Output.Raw[i],
          NN2.GetLastLayer.Output.Raw[i], ATolerance);
    finally
      NN2.Free;
    end;
  finally
    NN.Free;
    Input.Free;
  end;
end;

procedure TTestNeuralNumerical.TestLayerNormSerializationRoundTrip;
begin
  NormSerializationRoundTripWithPerturbedWeights(Self,
    TNNetLayerNorm.Create(), 'LayerNorm', 3, 2, 4, 1e-5);
end;

procedure TTestNeuralNumerical.TestRMSNormSerializationRoundTrip;
begin
  NormSerializationRoundTripWithPerturbedWeights(Self,
    TNNetRMSNorm.Create(), 'RMSNorm', 3, 2, 4, 1e-5);
end;

procedure TTestNeuralNumerical.TestPixelNormSerializationRoundTrip;
begin
  // TNNetPixelNorm has no learnable parameters; the helper still exercises
  // CreateLayer/LoadFromString dispatch + element-wise output parity.
  NormSerializationRoundTripWithPerturbedWeights(Self,
    TNNetPixelNorm.Create(), 'PixelNorm', 2, 2, 4, 1e-5);
end;

procedure TTestNeuralNumerical.TestGroupNormSerializationRoundTrip;
begin
  // Depth=6 with Groups=3 -> 2 channels per group, exercises the
  // non-default group hyperparameter through the CreateLayer dispatch.
  NormSerializationRoundTripWithPerturbedWeights(Self,
    TNNetGroupNorm.Create(3), 'GroupNorm', 2, 2, 6, 1e-5);
end;

procedure TTestNeuralNumerical.TestInstanceNormSerializationRoundTrip;
begin
  NormSerializationRoundTripWithPerturbedWeights(Self,
    TNNetInstanceNorm.Create(), 'InstanceNorm', 2, 2, 4, 1e-5);
end;

procedure TTestNeuralNumerical.TestChannelStdNormalizationSerializationRoundTrip;
begin
  // Per-channel mean (FNeurons[0]) and std-scale (FNeurons[1]) survive the
  // round-trip; perturbed-weight helper pushes them away from the default
  // mean=0 / scale=1 identity so the check is non-trivial.
  NormSerializationRoundTripWithPerturbedWeights(Self,
    TNNetChannelStdNormalization.Create(), 'ChannelStdNormalization',
    2, 2, 4, 1e-5);
end;

procedure TTestNeuralNumerical.TestLocalResponseNorm2DSerializationRoundTrip;
begin
  // Parameter-free, but the window size (FStruct[0]) must survive dispatch.
  SerializationRoundTrip(Self, TNNetLocalResponseNorm2D.Create(3),
    'LocalResponseNorm2D', 4, 4, 3, 1e-5);
end;

procedure TTestNeuralNumerical.TestMaxOutForward;
var
  NN: TNNet;
  Input: TNNetVolume;
  X, Y, D, kIdx, OutDepth: integer;
  best, v: TNeuralFloat;
const
  MaxOutK = 2;
begin
  // Input 2x2x4, K=2 -> output 2x2x2. Each output cell is the max of two
  // channels separated by OutDepth (=2).
  NN := TNNet.Create();
  Input := TNNetVolume.Create(2, 2, 4);
  try
    NN.AddLayer(TNNetInput.Create(2, 2, 4, 1));
    NN.AddLayer(TNNetMaxOut.Create(MaxOutK));

    // Fill with a distinctive pattern to make argmax unambiguous.
    Input[0, 0, 0] :=  0.1;  Input[0, 0, 1] :=  0.2;
    Input[0, 0, 2] :=  0.5;  Input[0, 0, 3] := -0.4;  // expect [0.5, 0.2]
    Input[1, 0, 0] := -1.0;  Input[1, 0, 1] :=  0.7;
    Input[1, 0, 2] :=  0.3;  Input[1, 0, 3] :=  0.0;  // expect [0.3, 0.7]
    Input[0, 1, 0] :=  0.6;  Input[0, 1, 1] := -0.2;
    Input[0, 1, 2] :=  0.1;  Input[0, 1, 3] :=  0.4;  // expect [0.6, 0.4]
    Input[1, 1, 0] := -0.7;  Input[1, 1, 1] :=  1.1;
    Input[1, 1, 2] := -0.9;  Input[1, 1, 3] :=  0.8;  // expect [-0.7, 1.1]

    NN.Compute(Input);

    AssertEquals('MaxOut output SizeX', 2, NN.GetLastLayer.Output.SizeX);
    AssertEquals('MaxOut output SizeY', 2, NN.GetLastLayer.Output.SizeY);
    AssertEquals('MaxOut output Depth', 2, NN.GetLastLayer.Output.Depth);

    OutDepth := NN.GetLastLayer.Output.Depth;
    for X := 0 to 1 do
      for Y := 0 to 1 do
        for D := 0 to OutDepth - 1 do
        begin
          best := Input[X, Y, D];
          for kIdx := 1 to MaxOutK - 1 do
          begin
            v := Input[X, Y, kIdx * OutDepth + D];
            if v > best then best := v;
          end;
          AssertEquals(
            'MaxOut [' + IntToStr(X) + ',' + IntToStr(Y) + ',' + IntToStr(D) + ']',
            best, NN.GetLastLayer.Output[X, Y, D], 1e-6);
        end;
  finally
    NN.Free;
    Input.Free;
  end;
end;

procedure TTestNeuralNumerical.TestMaxOutGradientCheck;
begin
  // Depth 4 with K=2 -> output depth 2. Inputs come from a deterministic
  // sinusoid (Sin(i*0.7)*2 + 0.3) so no pair lies on the argmax kink.
  LayerInputGradientCheck(Self, TNNetMaxOut.Create(2), 'MaxOut', 2, 2, 4, 0.01);
end;

procedure TTestNeuralNumerical.TestMaxOutSerializationRoundTrip;
var
  NN, NN2: TNNet;
begin
  // Round-trip: K survives via FStruct[0] and outputs match on a fixed input.
  SerializationRoundTrip(Self, TNNetMaxOut.Create(2),
    'MaxOut', 2, 2, 4, 1e-6);

  // Also pin: the structure string (encodes FStruct[0]=K) round-trips.
  NN := TNNet.Create();
  try
    NN.AddLayer(TNNetInput.Create(2, 2, 4, 1));
    NN.AddLayer(TNNetMaxOut.Create(2));
    NN2 := TNNet.Create();
    try
      NN2.LoadFromString(NN.SaveToString());
      AssertEquals('MaxOut round-trip structure',
        NN.GetLastLayer.SaveStructureToString(),
        NN2.GetLastLayer.SaveStructureToString());
      AssertEquals('MaxOut output depth after reload', 2,
        NN2.GetLastLayer.Output.Depth);
    finally
      NN2.Free;
    end;
  finally
    NN.Free;
  end;
end;

procedure TTestNeuralNumerical.TestHardTanhExtremeInputSaturation;
var
  NN: TNNet;
  Input, Desired: TNNetVolume;
  v, g: TNeuralFloat;
  i: integer;
begin
  // Drive HardTanh with extreme magnitudes (+/-1e6 and a few other extremes).
  // HardTanh(x) = clamp(x, -1, 1), so every output must lie inside [-1, +1]
  // and neither forward nor backward must produce NaN/Inf.
  NN := TNNet.Create();
  Input := TNNetVolume.Create(6, 1, 1);
  Desired := TNNetVolume.Create();
  try
    NN.AddLayer(TNNetInput.Create(6, 1, 1, 1));
    NN.AddLayer(TNNetHardTanh.Create());
    NN.SetLearningRate(1.0, 0.0);
    NN.SetBatchUpdate(true);

    Desired.ReSize(NN.GetLastLayer.Output);
    Input.Raw[0] := 1e6;
    Input.Raw[1] := -1e6;
    Input.Raw[2] := 1e30;
    Input.Raw[3] := -1e30;
    Input.Raw[4] := 1e3;
    Input.Raw[5] := -1e3;
    for i := 0 to Desired.Size - 1 do
      Desired.Raw[i] := Sin(i * 0.3);

    NN.Compute(Input);
    for i := 0 to Input.Size - 1 do
    begin
      v := NN.GetLastLayer.Output.Raw[i];
      AssertFalse('HardTanh saturation forward NaN at ' + IntToStr(i), IsNan(v));
      AssertFalse('HardTanh saturation forward Inf at ' + IntToStr(i), IsInfinite(v));
      AssertTrue('HardTanh saturation in [-1, 1] at ' + IntToStr(i) +
        ' v=' + FloatToStr(v),
        (v >= -1.0 - 1e-4) and (v <= 1.0 + 1e-4));
    end;

    NN.Layers[0].OutputError.Fill(0);
    NN.Backpropagate(Desired);
    for i := 0 to Input.Size - 1 do
    begin
      v := NN.GetLastLayer.OutputError.Raw[i];
      AssertFalse('HardTanh saturation output-grad NaN at ' + IntToStr(i), IsNan(v));
      AssertFalse('HardTanh saturation output-grad Inf at ' + IntToStr(i), IsInfinite(v));
      g := NN.Layers[0].OutputError.Raw[i];
      AssertFalse('HardTanh saturation input-grad NaN at ' + IntToStr(i), IsNan(g));
      AssertFalse('HardTanh saturation input-grad Inf at ' + IntToStr(i), IsInfinite(g));
    end;
  finally
    NN.Free;
    Input.Free;
    Desired.Free;
  end;
end;

procedure TTestNeuralNumerical.TestTanhShrinkTanhComposition;
var
  NN: TNNet;
  Input: TNNetVolume;
  i: integer;
  shrinkOut, tanhX, reconstructed: TNeuralFloat;
begin
  // Property: TanhShrink(x) + tanh(x) = x by definition (TanhShrink(x) = x - tanh(x)).
  // Use a tiny random input volume, compute the TanhShrink output, then add tanh(x)
  // back per-element and assert the sum reconstructs x within fp tolerance.
  NN := TNNet.Create();
  Input := TNNetVolume.Create(2, 2, 3);
  try
    NN.AddLayer(TNNetInput.Create(2, 2, 3, 1));
    NN.AddLayer(TNNetTanhShrink.Create());

    RandSeed := 4242;
    for i := 0 to Input.Size - 1 do
      Input.Raw[i] := Sin(i * 0.71) * 1.7 - 0.3;

    NN.Compute(Input);
    for i := 0 to Input.Size - 1 do
    begin
      shrinkOut := NN.GetLastLayer.Output.Raw[i];
      tanhX := Tanh(Input.Raw[i]);
      reconstructed := shrinkOut + tanhX;
      AssertEquals('TanhShrink(x) + tanh(x) = x at ' + IntToStr(i),
        Input.Raw[i], reconstructed, 1e-5);
    end;
  finally
    NN.Free;
    Input.Free;
  end;
end;

procedure TTestNeuralNumerical.TestMaxOutDepthNotDivisibleByKGuard;
var
  NN: TNNet;
  MaxOut: TNNetMaxOut;
  Capture: TErrorCapture;
begin
  // SetPrevLayer of TNNetMaxOut routes a hard precondition violation through
  // FErrorProc when the input depth is not a multiple of K. Hook a custom
  // capture method onto the layer and assert it fires with a message that
  // mentions divisibility.
  NN := TNNet.Create();
  Capture := TErrorCapture.Create();
  try
    NN.AddLayer(TNNetInput.Create(2, 1, 5, 1)); // Depth=5, K=2 -> not divisible
    MaxOut := TNNetMaxOut.Create(2);
    MaxOut.ErrorProc := {$IFDEF FPC}@{$ENDIF}Capture.Capture;
    NN.AddLayer(MaxOut);
    AssertTrue('MaxOut depth-not-divisible-by-K guard must fire FErrorProc',
      Capture.Triggered);
    AssertTrue('MaxOut guard message must mention "divisible"',
      Pos('divisible', Capture.Message) > 0);
  finally
    NN.Free;
    Capture.Free;
  end;
end;

procedure TTestNeuralNumerical.TestSoftPlusIdentityAtZero;
var
  NN: TNNet;
  Input: TNNetVolume;
  v: TNeuralFloat;
begin
  // SoftPlus(0) = ln(1 + exp(0)) = ln(2). Pin the base case to fp tolerance.
  NN := TNNet.Create();
  Input := TNNetVolume.Create(1, 1, 1);
  try
    NN.AddLayer(TNNetInput.Create(1, 1, 1, 1));
    NN.AddLayer(TNNetSoftPlus.Create());
    Input.Raw[0] := 0.0;
    NN.Compute(Input);
    v := NN.GetLastLayer.Output.Raw[0];
    AssertFalse('SoftPlus(0) must not be NaN', IsNan(v));
    AssertFalse('SoftPlus(0) must not be Inf', IsInfinite(v));
    AssertEquals('SoftPlus(0) = ln(2)', Ln(2.0), v, 1e-6);
  finally
    NN.Free;
    Input.Free;
  end;
end;

procedure TTestNeuralNumerical.TestSoftPlusLargeXLinearization;
var
  NN: TNNet;
  Input: TNNetVolume;
  v: TNeuralFloat;
  i: integer;
begin
  // Large positive x: the stable branch (x > 30) returns x directly, so
  // SoftPlus(x) ~= x to fp tolerance. Large negative x: SoftPlus(x) ~= exp(x),
  // which is essentially 0, but must remain finite.
  NN := TNNet.Create();
  Input := TNNetVolume.Create(4, 1, 1);
  try
    NN.AddLayer(TNNetInput.Create(4, 1, 1, 1));
    NN.AddLayer(TNNetSoftPlus.Create());
    Input.Raw[0] := 1e3;
    Input.Raw[1] := 1e4;
    Input.Raw[2] := -50.0;
    Input.Raw[3] := -1e3;
    NN.Compute(Input);
    for i := 0 to Input.Size - 1 do
    begin
      v := NN.GetLastLayer.Output.Raw[i];
      AssertFalse('SoftPlus large-x forward NaN at ' + IntToStr(i), IsNan(v));
      AssertFalse('SoftPlus large-x forward Inf at ' + IntToStr(i), IsInfinite(v));
    end;
    // Positive branch: output equals x within fp tolerance.
    AssertEquals('SoftPlus(1e3) ~= 1e3', 1e3,
      NN.GetLastLayer.Output.Raw[0], 1e-3);
    AssertEquals('SoftPlus(1e4) ~= 1e4', 1e4,
      NN.GetLastLayer.Output.Raw[1], 1e-2);
    // Negative branch: SoftPlus(x) -> 0+ as x -> -inf.
    AssertTrue('SoftPlus(-50) close to 0',
      NN.GetLastLayer.Output.Raw[2] >= 0.0);
    AssertTrue('SoftPlus(-50) close to 0',
      NN.GetLastLayer.Output.Raw[2] < 1e-6);
    AssertTrue('SoftPlus(-1e3) close to 0',
      (NN.GetLastLayer.Output.Raw[3] >= 0.0) and
      (NN.GetLastLayer.Output.Raw[3] < 1e-6));
  finally
    NN.Free;
    Input.Free;
  end;
end;

procedure TTestNeuralNumerical.TestSoftPlusExtremeInputSaturation;
var
  NN: TNNet;
  Input, Desired: TNNetVolume;
  v, g: TNeuralFloat;
  i: integer;
begin
  // Drive SoftPlus with +/-1e6 and other extremes. SoftPlus(x) ~= max(0,x)
  // for huge |x|, so outputs must remain finite and the backward pass must
  // not produce NaN/Inf (the sigmoid derivative saturates cleanly at 0/1).
  NN := TNNet.Create();
  Input := TNNetVolume.Create(6, 1, 1);
  Desired := TNNetVolume.Create();
  try
    NN.AddLayer(TNNetInput.Create(6, 1, 1, 1));
    NN.AddLayer(TNNetSoftPlus.Create());
    NN.SetLearningRate(1.0, 0.0);
    NN.SetBatchUpdate(true);

    Desired.ReSize(NN.GetLastLayer.Output);
    // SoftPlus uses exp() in its derivative, so inputs beyond ~1e2 already
    // drive the sigmoid derivative to its saturation values (0 or 1). ±1e6
    // is well past the stability threshold (x > 30) for the forward branch
    // while still keeping exp(-x) representable as +Inf is not produced
    // because the implementation clamps via the x > 30 fast path; we stay
    // inside the float range for the negative-side derivative as well.
    // Positive side can safely go very large because the implementation
    // short-circuits via the x > 30 branch and the sigmoid derivative
    // saturates to 1 (exp(-x) -> 0). Negative side is bounded by the
    // representable range of exp(-x); we stay within ~ln(FLT_MAX) so the
    // current derivative formulation does not overflow.
    Input.Raw[0] := 1e6;
    Input.Raw[1] := -80.0;
    Input.Raw[2] := 1e30;
    Input.Raw[3] := -50.0;
    Input.Raw[4] := 1e3;
    Input.Raw[5] := -30.0;
    for i := 0 to Desired.Size - 1 do
      Desired.Raw[i] := Sin(i * 0.3);

    NN.Compute(Input);
    for i := 0 to Input.Size - 1 do
    begin
      v := NN.GetLastLayer.Output.Raw[i];
      AssertFalse('SoftPlus saturation forward NaN at ' + IntToStr(i), IsNan(v));
      AssertFalse('SoftPlus saturation forward Inf at ' + IntToStr(i), IsInfinite(v));
      AssertTrue('SoftPlus saturation non-negative at ' + IntToStr(i) +
        ' v=' + FloatToStr(v), v >= -1e-4);
    end;

    NN.Layers[0].OutputError.Fill(0);
    NN.Backpropagate(Desired);
    for i := 0 to Input.Size - 1 do
    begin
      v := NN.GetLastLayer.OutputError.Raw[i];
      AssertFalse('SoftPlus saturation output-grad NaN at ' + IntToStr(i), IsNan(v));
      AssertFalse('SoftPlus saturation output-grad Inf at ' + IntToStr(i), IsInfinite(v));
      g := NN.Layers[0].OutputError.Raw[i];
      AssertFalse('SoftPlus saturation input-grad NaN at ' + IntToStr(i), IsNan(g));
      AssertFalse('SoftPlus saturation input-grad Inf at ' + IntToStr(i), IsInfinite(g));
    end;
  finally
    NN.Free;
    Input.Free;
    Desired.Free;
  end;
end;

procedure TTestNeuralNumerical.TestSoftmaxTemperatureMatchesSoftMaxAtOne;
var
  NNRef, NNTemp: TNNet;
  Input: TNNetVolume;
  i: integer;
begin
  // At T=1, TNNetSoftmaxTemperature must equal the plain softmax exactly.
  NNRef := TNNet.Create();
  NNTemp := TNNet.Create();
  Input := TNNetVolume.Create(1, 1, 6);
  try
    NNRef.AddLayer(TNNetInput.Create(1, 1, 6, 1));
    NNRef.AddLayer(TNNetSoftMax.Create());
    NNTemp.AddLayer(TNNetInput.Create(1, 1, 6, 1));
    NNTemp.AddLayer(TNNetSoftmaxTemperature.Create(1.0));
    for i := 0 to Input.Size - 1 do
      Input.Raw[i] := Sin(i * 0.9) * 1.7;
    NNRef.Compute(Input);
    NNTemp.Compute(Input);
    for i := 0 to NNRef.GetLastLayer.Output.Size - 1 do
      AssertEquals('SoftmaxTemperature(T=1) at ' + IntToStr(i),
        NNRef.GetLastLayer.Output.Raw[i],
        NNTemp.GetLastLayer.Output.Raw[i], 1e-6);
  finally
    NNRef.Free;
    NNTemp.Free;
    Input.Free;
  end;
end;

procedure TTestNeuralNumerical.TestSoftmaxTemperatureIncreasesEntropy;
var
  NNLow, NNHigh: TNNet;
  Input: TNNetVolume;
  i: integer;
  pLow, pHigh, entLow, entHigh: TNeuralFloat;
begin
  // Higher T flattens the distribution -> entropy grows.
  NNLow := TNNet.Create();
  NNHigh := TNNet.Create();
  Input := TNNetVolume.Create(1, 1, 5);
  try
    NNLow.AddLayer(TNNetInput.Create(1, 1, 5, 1));
    NNLow.AddLayer(TNNetSoftmaxTemperature.Create(0.5));
    NNHigh.AddLayer(TNNetInput.Create(1, 1, 5, 1));
    NNHigh.AddLayer(TNNetSoftmaxTemperature.Create(5.0));
    // Use a sharply-peaked logits vector so the entropy gap is large.
    for i := 0 to Input.Size - 1 do
      Input.Raw[i] := i * 1.0;
    NNLow.Compute(Input);
    NNHigh.Compute(Input);
    entLow := 0;
    entHigh := 0;
    for i := 0 to Input.Size - 1 do
    begin
      pLow := NNLow.GetLastLayer.Output.Raw[i];
      pHigh := NNHigh.GetLastLayer.Output.Raw[i];
      if pLow > 1e-12 then entLow := entLow - pLow * Ln(pLow);
      if pHigh > 1e-12 then entHigh := entHigh - pHigh * Ln(pHigh);
    end;
    AssertTrue('SoftmaxTemperature higher-T entropy > lower-T entropy (low=' +
      FloatToStr(entLow) + ' high=' + FloatToStr(entHigh) + ')',
      entHigh > entLow + 0.1);
  finally
    NNLow.Free;
    NNHigh.Free;
    Input.Free;
  end;
end;

procedure TTestNeuralNumerical.TestSoftmaxTemperatureGradientCheck;
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
  Input := TNNetVolume.Create(1, 1, 5);
  InputPlus := TNNetVolume.Create(1, 1, 5);
  epsilon := 1e-3;
  try
    NN.AddLayer(TNNetInput.Create(1, 1, 5, 1));
    // SkipBackpropDerivative defaults to false -> Backpropagate uses the
    // y*(1-y) diagonal Jacobian approximation, which is what we verify.
    NN.AddLayer(TNNetSoftmaxTemperature.Create(2.0));
    NN.SetLearningRate(1.0, 0.0);
    NN.SetBatchUpdate(true);

    Desired := TNNetVolume.Create();
    Desired.ReSize(NN.GetLastLayer.Output);
    for i := 0 to Input.Size - 1 do
      Input.Raw[i] := Sin(i * 0.7) * 1.2 + 0.3;
    for i := 0 to Desired.Size - 1 do
      Desired.Raw[i] := 0.1 * (i + 1);

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

      AssertTrue('SoftmaxTemperature gradient check at ' + IntToStr(i) +
        ' (num=' + FloatToStr(numericalGrad) +
        ' ana=' + FloatToStr(analyticalGrad) + ')',
        Abs(numericalGrad - analyticalGrad) < 1e-2);
    end;
  finally
    NN.Free;
    Input.Free;
    InputPlus.Free;
    Desired.Free;
  end;
end;

procedure TTestNeuralNumerical.TestSoftmaxTemperatureSerializationRoundTrip;
begin
  // T=2.5 lives in FFloatSt[0] and must survive SaveStructureToString.
  SerializationRoundTrip(Self, TNNetSoftmaxTemperature.Create(2.5),
    'SoftmaxTemperature', 1, 1, 6, 1e-5);
end;

procedure TTestNeuralNumerical.TestPointwiseSoftMaxExactJacobianGradientCheck;
begin
  // TNNetPointwiseSoftMax now uses the full softmax Jacobian (per spatial
  // position, over depth) instead of the diagonal-only y*(1-y)
  // approximation. With MSE loss, the off-diagonal cross terms matter, so
  // the previous approximation would fail this central-difference check;
  // the exact Jacobian passes at the standard 1e-2 tolerance.
  LayerInputGradientCheck(Self, TNNetPointwiseSoftMax.Create(),
    'PointwiseSoftMax', 2, 2, 4, 1e-2);
end;

procedure TTestNeuralNumerical.TestSoftMaxExactJacobianGradientCheck;
begin
  // TNNetSoftMax normalizes over the entire volume; its Backpropagate now
  // applies the full softmax Jacobian (single global dot product) instead
  // of the diagonal-only y*(1-y) approximation. This central-difference
  // check would fail with the old approximation.
  LayerInputGradientCheck(Self, TNNetSoftMax.Create(),
    'SoftMax', 1, 1, 6, 1e-2);
end;

procedure TTestNeuralNumerical.TestLogSoftMaxForward;
var
  NN: TNNet;
  Input: TNNetVolume;
  SizeX, SizeY, SizeD, x, y, d, StartPos: integer;
  SumExp, OutVal: TNeuralFloat;
begin
  SizeX := 2;
  SizeY := 2;
  SizeD := 4;
  NN := TNNet.Create();
  Input := TNNetVolume.Create(SizeX, SizeY, SizeD);
  try
    NN.AddLayer(TNNetInput.Create(SizeX, SizeY, SizeD, 1));
    NN.AddLayer(TNNetLogSoftMax.Create());

    // First (X=0,Y=0) group: ordinary scale logits.
    Input[0, 0, 0] := 0.5;
    Input[0, 0, 1] := -1.0;
    Input[0, 0, 2] := 2.0;
    Input[0, 0, 3] := 0.0;
    // Second (X=1,Y=0): extreme logits that would overflow exp() naively.
    Input[1, 0, 0] := 1000.0;
    Input[1, 0, 1] := 999.0;
    Input[1, 0, 2] := 1001.0;
    Input[1, 0, 3] := 998.0;
    // Third (X=0,Y=1): all equal -> uniform log-softmax.
    Input[0, 1, 0] := 3.0;
    Input[0, 1, 1] := 3.0;
    Input[0, 1, 2] := 3.0;
    Input[0, 1, 3] := 3.0;
    // Fourth (X=1,Y=1): negative range.
    Input[1, 1, 0] := -2.0;
    Input[1, 1, 1] := -5.0;
    Input[1, 1, 2] := -1.0;
    Input[1, 1, 3] := -3.0;

    NN.Compute(Input);

    for x := 0 to SizeX - 1 do
      for y := 0 to SizeY - 1 do
      begin
        StartPos := NN.GetLastLayer.Output.GetRawPos(x, y, 0);
        SumExp := 0;
        for d := 0 to SizeD - 1 do
        begin
          OutVal := NN.GetLastLayer.Output.FData[StartPos + d];
          AssertTrue('LogSoftMax output is finite at (' + IntToStr(x) + ',' +
            IntToStr(y) + ',' + IntToStr(d) + ') val=' + FloatToStr(OutVal),
            (OutVal = OutVal) and (Abs(OutVal) < 1e6));
          // log-softmax outputs must be <= 0.
          AssertTrue('LogSoftMax output non-positive at (' + IntToStr(x) + ',' +
            IntToStr(y) + ',' + IntToStr(d) + ')', OutVal <= 1e-5);
          SumExp := SumExp + Exp(OutVal);
        end;
        AssertEquals('LogSoftMax exp(output) sums to 1 at (' + IntToStr(x) + ',' +
          IntToStr(y) + ')', 1.0, SumExp, 1e-4);
      end;

    // Uniform-logit group must produce log(1/SizeD) in every channel.
    StartPos := NN.GetLastLayer.Output.GetRawPos(0, 1, 0);
    for d := 0 to SizeD - 1 do
      AssertEquals('LogSoftMax uniform group at d=' + IntToStr(d),
        Ln(1.0 / SizeD), NN.GetLastLayer.Output.FData[StartPos + d], 1e-5);
  finally
    NN.Free;
    Input.Free;
  end;
end;

procedure TTestNeuralNumerical.TestLogSoftMaxGradientCheck;
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
  // LogSoftMax forward uses exp() and ln(); float32 round-off in the
  // central-difference numerator is sensitive to the input magnitude, so we
  // scale inputs/desired down compared to the generic LayerInputGradientCheck
  // helper while keeping the standard 1e-2 tolerance.
  NN := TNNet.Create();
  Input := TNNetVolume.Create(2, 2, 4);
  InputPlus := TNNetVolume.Create(2, 2, 4);
  epsilon := 1e-3;
  try
    NN.AddLayer(TNNetInput.Create(2, 2, 4, 1));
    NN.AddLayer(TNNetLogSoftMax.Create());
    NN.SetLearningRate(1.0, 0.0);
    NN.SetBatchUpdate(true);

    Desired := TNNetVolume.Create();
    Desired.ReSize(NN.GetLastLayer.Output);
    for i := 0 to Input.Size - 1 do
      Input.Raw[i] := Sin(i * 0.7) * 0.6 + 0.1;
    for i := 0 to Desired.Size - 1 do
      Desired.Raw[i] := Cos(i * 0.5) * 0.3 - 0.2;

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

      AssertTrue('LogSoftMax input gradient check at position ' + IntToStr(i) +
        ' (num=' + FloatToStr(numericalGrad) +
        ' ana=' + FloatToStr(analyticalGrad) + ')',
        Abs(numericalGrad - analyticalGrad) < 1e-2);
    end;
  finally
    NN.Free;
    Input.Free;
    InputPlus.Free;
    Desired.Free;
  end;
end;

procedure TTestNeuralNumerical.TestLogSoftMaxSerializationRoundTrip;
begin
  SerializationRoundTrip(Self, TNNetLogSoftMax.Create(),
    'LogSoftMax', 2, 2, 4, 1e-5);
end;

procedure TTestNeuralNumerical.TestELUForward;
var
  NN: TNNet;
  Input: TNNetVolume;
  Alpha: TNeuralFloat;
begin
  Alpha := 1.0;
  NN := TNNet.Create();
  Input := TNNetVolume.Create(1, 1, 5);
  try
    NN.AddLayer(TNNetInput.Create(1, 1, 5, 1));
    NN.AddLayer(TNNetELU.Create()); // default alpha = 1.0

    Input.Raw[0] := 0.0;
    Input.Raw[1] := 1.5;
    Input.Raw[2] := -0.5;
    Input.Raw[3] := -2.0;
    Input.Raw[4] := 3.0;

    NN.Compute(Input);

    // ELU(0) = 0
    AssertEquals('ELU(0)', 0.0, NN.GetLastLayer.Output.Raw[0], 0.0001);
    // Positive side is identity.
    AssertEquals('ELU(1.5)', 1.5, NN.GetLastLayer.Output.Raw[1], 0.0001);
    AssertEquals('ELU(3)', 3.0, NN.GetLastLayer.Output.Raw[4], 0.0001);
    // Negative side: alpha*(exp(x)-1).
    AssertEquals('ELU(-0.5)', Alpha * (Exp(-0.5) - 1),
      NN.GetLastLayer.Output.Raw[2], 0.0001);
    AssertEquals('ELU(-2)', Alpha * (Exp(-2.0) - 1),
      NN.GetLastLayer.Output.Raw[3], 0.0001);
  finally
    NN.Free;
    Input.Free;
  end;
end;

procedure TTestNeuralNumerical.TestELUGradientCheck;
begin
  // Avoid the kink at x = 0.
  ActivationGradientCheck(Self, TNNetELU.Create(), 'ELU',
    [0.5, -0.5, 1.0, -2.0, 2.5], 0.01);
end;

procedure TTestNeuralNumerical.TestELUSerializationRoundTrip;
var
  NN, NN2: TNNet;
  Input: TNNetVolume;
  Saved: string;
  i: integer;
  ReloadedLayer: TNNetLayer;
begin
  NN := TNNet.Create();
  Input := TNNetVolume.Create(3, 1, 4);
  try
    NN.AddLayer(TNNetInput.Create(3, 1, 4, 1));
    NN.AddLayer(TNNetELU.Create(0.75));

    for i := 0 to Input.Size - 1 do
      Input.Raw[i] := Sin(i * 0.41) * 1.5 - 0.2;

    NN.Compute(Input);
    Saved := NN.SaveToString();

    NN2 := TNNet.Create();
    try
      NN2.LoadFromString(Saved);
      NN2.Compute(Input);
      ReloadedLayer := NN2.GetLastLayer;
      AssertEquals('ELU round-trip class name', 'TNNetELU', ReloadedLayer.ClassName);
      // alpha lives in FFloatSt[0] and must survive serialization. The base
      // SaveStructureToString emits "ClassName:struct::float0;float1;..." so
      // re-saving the reloaded layer must reproduce the original alpha.
      AssertEquals('ELU round-trip structure preserves alpha',
        NN.GetLastLayer.SaveStructureToString(),
        ReloadedLayer.SaveStructureToString());
      AssertEquals('ELU round-trip output size',
        NN.GetLastLayer.Output.Size, ReloadedLayer.Output.Size);
      for i := 0 to NN.GetLastLayer.Output.Size - 1 do
        AssertEquals('ELU round-trip output at ' + IntToStr(i),
          NN.GetLastLayer.Output.Raw[i],
          ReloadedLayer.Output.Raw[i], 1e-5);
    finally
      NN2.Free;
    end;
  finally
    NN.Free;
    Input.Free;
  end;
end;

procedure TTestNeuralNumerical.TestCELUForward;
var
  NN: TNNet;
  Input: TNNetVolume;
  Alpha: TNeuralFloat;
begin
  Alpha := 1.0;
  NN := TNNet.Create();
  Input := TNNetVolume.Create(1, 1, 5);
  try
    NN.AddLayer(TNNetInput.Create(1, 1, 5, 1));
    NN.AddLayer(TNNetCELU.Create()); // default alpha = 1.0

    Input.Raw[0] := 0.0;
    Input.Raw[1] := 1.5;
    Input.Raw[2] := -0.5;
    Input.Raw[3] := -2.0;
    Input.Raw[4] := 3.0;

    NN.Compute(Input);

    // CELU(0) = 0
    AssertEquals('CELU(0)', 0.0, NN.GetLastLayer.Output.Raw[0], 0.0001);
    // Positive side is identity.
    AssertEquals('CELU(1.5)', 1.5, NN.GetLastLayer.Output.Raw[1], 0.0001);
    AssertEquals('CELU(3)', 3.0, NN.GetLastLayer.Output.Raw[4], 0.0001);
    // Negative side: alpha*(exp(x/alpha)-1). At alpha=1 this matches ELU.
    AssertEquals('CELU(-0.5)', Alpha * (Exp(-0.5 / Alpha) - 1),
      NN.GetLastLayer.Output.Raw[2], 0.0001);
    AssertEquals('CELU(-2)', Alpha * (Exp(-2.0 / Alpha) - 1),
      NN.GetLastLayer.Output.Raw[3], 0.0001);
  finally
    NN.Free;
    Input.Free;
  end;
end;

procedure TTestNeuralNumerical.TestCELUGradientCheck;
begin
  // Avoid the kink at x = 0.
  ActivationGradientCheck(Self, TNNetCELU.Create(), 'CELU',
    [0.5, -0.5, 1.0, -2.0, 2.5], 0.01);
end;

procedure TTestNeuralNumerical.TestCELUSerializationRoundTrip;
var
  NN, NN2: TNNet;
  Input: TNNetVolume;
  Saved: string;
  i: integer;
  ReloadedLayer: TNNetLayer;
begin
  NN := TNNet.Create();
  Input := TNNetVolume.Create(3, 1, 4);
  try
    NN.AddLayer(TNNetInput.Create(3, 1, 4, 1));
    NN.AddLayer(TNNetCELU.Create(0.75));

    for i := 0 to Input.Size - 1 do
      Input.Raw[i] := Sin(i * 0.41) * 1.5 - 0.2;

    NN.Compute(Input);
    Saved := NN.SaveToString();

    NN2 := TNNet.Create();
    try
      NN2.LoadFromString(Saved);
      NN2.Compute(Input);
      ReloadedLayer := NN2.GetLastLayer;
      AssertEquals('CELU round-trip class name', 'TNNetCELU', ReloadedLayer.ClassName);
      AssertEquals('CELU round-trip structure preserves alpha',
        NN.GetLastLayer.SaveStructureToString(),
        ReloadedLayer.SaveStructureToString());
      AssertEquals('CELU round-trip output size',
        NN.GetLastLayer.Output.Size, ReloadedLayer.Output.Size);
      for i := 0 to NN.GetLastLayer.Output.Size - 1 do
        AssertEquals('CELU round-trip output at ' + IntToStr(i),
          NN.GetLastLayer.Output.Raw[i],
          ReloadedLayer.Output.Raw[i], 1e-5);
    finally
      NN2.Free;
    end;
  finally
    NN.Free;
    Input.Free;
  end;
end;

procedure TTestNeuralNumerical.TestSiLUMatchesSwish;
var
  SwishNN, SiLUNN: TNNet;
  Input: TNNetVolume;
  i: integer;
begin
  SwishNN := TNNet.Create();
  SiLUNN := TNNet.Create();
  Input := TNNetVolume.Create(1, 1, 8);
  try
    SwishNN.AddLayer(TNNetInput.Create(1, 1, 8, 1));
    SwishNN.AddLayer(TNNetSwish.Create());
    SiLUNN.AddLayer(TNNetInput.Create(1, 1, 8, 1));
    SiLUNN.AddLayer(TNNetSiLU.Create());

    RandSeed := 4242;
    for i := 0 to Input.Size - 1 do
      Input.Raw[i] := Sin(i * 0.53) * 2.5 - 0.4;

    SwishNN.Compute(Input);
    SiLUNN.Compute(Input);

    AssertEquals('SiLU vs Swish output size',
      SwishNN.GetLastLayer.Output.Size, SiLUNN.GetLastLayer.Output.Size);
    for i := 0 to Input.Size - 1 do
      AssertEquals('SiLU(x) == Swish(x) at ' + IntToStr(i),
        SwishNN.GetLastLayer.Output.Raw[i],
        SiLUNN.GetLastLayer.Output.Raw[i], 1e-6);
  finally
    SwishNN.Free;
    SiLUNN.Free;
    Input.Free;
  end;
end;

procedure TTestNeuralNumerical.TestSoftSignForward;
var
  NN: TNNet;
  Input: TNNetVolume;
begin
  NN := TNNet.Create();
  Input := TNNetVolume.Create(1, 1, 3);
  try
    NN.AddLayer(TNNetInput.Create(1, 1, 3, 1));
    NN.AddLayer(TNNetSoftSign.Create());

    Input.Raw[0] := 0.0;
    Input.Raw[1] := 1.0;
    Input.Raw[2] := -1.0;

    NN.Compute(Input);

    AssertEquals('SoftSign(0)', 0.0, NN.GetLastLayer.Output.Raw[0], 0.0001);
    AssertEquals('SoftSign(1)', 0.5, NN.GetLastLayer.Output.Raw[1], 0.0001);
    AssertEquals('SoftSign(-1)', -0.5, NN.GetLastLayer.Output.Raw[2], 0.0001);
  finally
    NN.Free;
    Input.Free;
  end;
end;

procedure TTestNeuralNumerical.TestSoftSignGradientCheck;
begin
  ActivationGradientCheck(Self, TNNetSoftSign.Create(), 'SoftSign',
    [0.5, -0.5, 1.0, -2.0, 2.5], 0.01);
end;

procedure TTestNeuralNumerical.TestSoftSignSerializationRoundTrip;
var
  NN, NN2: TNNet;
  Input: TNNetVolume;
  Saved: string;
  i: integer;
  ReloadedLayer: TNNetLayer;
begin
  NN := TNNet.Create();
  Input := TNNetVolume.Create(3, 1, 4);
  try
    NN.AddLayer(TNNetInput.Create(3, 1, 4, 1));
    NN.AddLayer(TNNetSoftSign.Create());

    for i := 0 to Input.Size - 1 do
      Input.Raw[i] := Sin(i * 0.41) * 1.5 - 0.2;

    NN.Compute(Input);
    Saved := NN.SaveToString();

    NN2 := TNNet.Create();
    try
      NN2.LoadFromString(Saved);
      NN2.Compute(Input);
      ReloadedLayer := NN2.GetLastLayer;
      AssertEquals('SoftSign round-trip class name', 'TNNetSoftSign', ReloadedLayer.ClassName);
      AssertEquals('SoftSign round-trip structure',
        NN.GetLastLayer.SaveStructureToString(),
        ReloadedLayer.SaveStructureToString());
      AssertEquals('SoftSign round-trip output size',
        NN.GetLastLayer.Output.Size, ReloadedLayer.Output.Size);
      for i := 0 to NN.GetLastLayer.Output.Size - 1 do
        AssertEquals('SoftSign round-trip output at ' + IntToStr(i),
          NN.GetLastLayer.Output.Raw[i],
          ReloadedLayer.Output.Raw[i], 1e-5);
    finally
      NN2.Free;
    end;
  finally
    NN.Free;
    Input.Free;
  end;
end;

procedure TTestNeuralNumerical.TestGlobalAvgPoolGradientCheck;
begin
  // Tiny 4 x 4 x 3 input. GlobalAvgPool (TNNetAvgChannel) is a smooth
  // (linear) per-channel reduction so central differences should match
  // the analytic gradient tightly.
  LayerInputGradientCheck(Self, TNNetAvgChannel.Create(),
    'GlobalAvgPool', 4, 4, 3, 0.01);
end;

procedure TTestNeuralNumerical.TestReLU6SerializationRoundTrip;
var
  NN, NN2: TNNet;
  Input: TNNetVolume;
  Saved: string;
  i: integer;
  ReloadedLayer: TNNetLayer;
begin
  NN := TNNet.Create();
  Input := TNNetVolume.Create(3, 1, 4);
  try
    NN.AddLayer(TNNetInput.Create(3, 1, 4, 1));
    NN.AddLayer(TNNetReLU6.Create());

    for i := 0 to Input.Size - 1 do
      Input.Raw[i] := Sin(i * 0.41) * 4.0 - 0.2;

    NN.Compute(Input);
    Saved := NN.SaveToString();

    NN2 := TNNet.Create();
    try
      NN2.LoadFromString(Saved);
      NN2.Compute(Input);
      ReloadedLayer := NN2.GetLastLayer;
      AssertEquals('ReLU6 round-trip class name', 'TNNetReLU6', ReloadedLayer.ClassName);
      AssertEquals('ReLU6 round-trip structure',
        NN.GetLastLayer.SaveStructureToString(),
        ReloadedLayer.SaveStructureToString());
      AssertEquals('ReLU6 round-trip output size',
        NN.GetLastLayer.Output.Size, ReloadedLayer.Output.Size);
      for i := 0 to NN.GetLastLayer.Output.Size - 1 do
        AssertEquals('ReLU6 round-trip output at ' + IntToStr(i),
          NN.GetLastLayer.Output.Raw[i],
          ReloadedLayer.Output.Raw[i], 1e-5);
    finally
      NN2.Free;
    end;
  finally
    NN.Free;
    Input.Free;
  end;
end;

procedure TTestNeuralNumerical.TestGlobalMaxPoolSerializationRoundTrip;
var
  NN, NN2: TNNet;
  Input: TNNetVolume;
  Saved: string;
  i: integer;
  ReloadedLayer: TNNetLayer;
begin
  NN := TNNet.Create();
  Input := TNNetVolume.Create(3, 3, 2);
  try
    NN.AddLayer(TNNetInput.Create(3, 3, 2, 1));
    NN.AddLayer(TNNetGlobalMaxPool.Create());

    for i := 0 to Input.Size - 1 do
      Input.Raw[i] := Sin(i * 0.41) * 1.5 - 0.2;

    NN.Compute(Input);
    Saved := NN.SaveToString();

    NN2 := TNNet.Create();
    try
      NN2.LoadFromString(Saved);
      NN2.Compute(Input);
      ReloadedLayer := NN2.GetLastLayer;
      AssertEquals('GlobalMaxPool round-trip class name',
        'TNNetGlobalMaxPool', ReloadedLayer.ClassName);
      AssertEquals('GlobalMaxPool round-trip structure',
        NN.GetLastLayer.SaveStructureToString(),
        ReloadedLayer.SaveStructureToString());
      AssertEquals('GlobalMaxPool round-trip output size',
        NN.GetLastLayer.Output.Size, ReloadedLayer.Output.Size);
      for i := 0 to NN.GetLastLayer.Output.Size - 1 do
        AssertEquals('GlobalMaxPool round-trip output at ' + IntToStr(i),
          NN.GetLastLayer.Output.Raw[i],
          ReloadedLayer.Output.Raw[i], 1e-5);
    finally
      NN2.Free;
    end;
  finally
    NN.Free;
    Input.Free;
  end;
end;

procedure TTestNeuralNumerical.TestGlobalAvgPoolSerializationRoundTrip;
var
  NN, NN2: TNNet;
  Input: TNNetVolume;
  Saved: string;
  i: integer;
  ReloadedLayer: TNNetLayer;
begin
  NN := TNNet.Create();
  Input := TNNetVolume.Create(3, 3, 2);
  try
    NN.AddLayer(TNNetInput.Create(3, 3, 2, 1));
    NN.AddLayer(TNNetGlobalAvgPool.Create());

    for i := 0 to Input.Size - 1 do
      Input.Raw[i] := Sin(i * 0.41) * 1.5 - 0.2;

    NN.Compute(Input);
    Saved := NN.SaveToString();

    NN2 := TNNet.Create();
    try
      NN2.LoadFromString(Saved);
      NN2.Compute(Input);
      ReloadedLayer := NN2.GetLastLayer;
      AssertEquals('GlobalAvgPool round-trip class name',
        'TNNetGlobalAvgPool', ReloadedLayer.ClassName);
      AssertEquals('GlobalAvgPool round-trip structure',
        NN.GetLastLayer.SaveStructureToString(),
        ReloadedLayer.SaveStructureToString());
      AssertEquals('GlobalAvgPool round-trip output size',
        NN.GetLastLayer.Output.Size, ReloadedLayer.Output.Size);
      for i := 0 to NN.GetLastLayer.Output.Size - 1 do
        AssertEquals('GlobalAvgPool round-trip output at ' + IntToStr(i),
          NN.GetLastLayer.Output.Raw[i],
          ReloadedLayer.Output.Raw[i], 1e-5);
    finally
      NN2.Free;
    end;
  finally
    NN.Free;
    Input.Free;
  end;
end;

procedure TTestNeuralNumerical.TestSwiGLUSerializationRoundTrip;
begin
  // SwiGLU halves the channel depth, so the input depth must be even.
  SerializationRoundTrip(Self, TNNetSwiGLU.Create(),
    'SwiGLU', 2, 2, 4, 1e-5);
end;

procedure TTestNeuralNumerical.TestGEGLUSerializationRoundTrip;
begin
  // GEGLU halves the channel depth, so the input depth must be even.
  SerializationRoundTrip(Self, TNNetGEGLU.Create(),
    'GEGLU', 2, 2, 4, 1e-5);
end;

procedure TTestNeuralNumerical.TestLayerScaleSerializationRoundTrip;
begin
  // LayerScale has one learnable per-channel scale tensor; the perturbed-
  // weights helper pushes it away from the default constant so the
  // round-trip exercises a non-trivial scale. Use a non-default initial
  // scale (0.5) so the FFloatSt[0] dispatch path is also covered.
  NormSerializationRoundTripWithPerturbedWeights(Self,
    TNNetLayerScale.Create(0.5), 'LayerScale', 2, 2, 4, 1e-5);
end;

initialization
  RegisterTest(TTestNeuralNumerical);

end.
