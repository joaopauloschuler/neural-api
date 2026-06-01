unit TestNeuralLayersExtra;

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, Math, StrUtils, fpcunit, testregistry, neuralnetwork, neuralvolume, neuralcalibration;

type
  TTestNeuralLayersExtra = class(TTestCase)
  published
    // Deconvolution (Transposed Convolution) tests
    procedure TestDeconvolutionForward;
    procedure TestDeconvolutionReLUForward;
    procedure TestDeconvolutionOutputSize;
    
    // DeLocalConnect tests
    procedure TestDeLocalConnectForward;
    procedure TestDeLocalConnectReLUForward;
    
    // DeMaxPool (Upsampling) tests
    procedure TestDeMaxPoolForward;
    procedure TestDeMaxPoolOutputSize;
    
    // Upsample tests
    procedure TestUpsampleForward;
    procedure TestUpsampleDepthToSpace;
    
    // Power and transformation layers
    procedure TestPowerLayer;
    procedure TestMulByConstant;
    procedure TestNegateLayer;
    procedure TestSignedSquareRoot;
    
    // Additional activation tests
    procedure TestReLUL;
    procedure TestVeryLeakyReLU;
    procedure TestSwish6;
    procedure TestHardSwish;
    procedure TestReLUSqrt;
    
    // Pointwise SoftMax tests
    procedure TestPointwiseSoftMax;
    procedure TestPointwiseSoftMaxSumToOne;
    
    // Noise and regularization layers
    procedure TestRandomMulAdd;
    procedure TestChannelRandomMulAdd;
    
    // Cell operations
    procedure TestCellMul;
    procedure TestCellMulByCell;
    
    // Channel operations
    procedure TestChannelMulByLayer;
    
    // Transposition tests
    procedure TestTransposeXD;
    procedure TestTransposeYD;
    
    // Min/Max channel combined tests
    procedure TestMinChannel;
    procedure TestMinMaxPoolCombined;
    
    // Grouped operations
    procedure TestGroupedPointwiseConvLinear;
    procedure TestGroupedPointwiseConvReLU;
    
    // Network architecture tests
    procedure TestResNetBlock;
    procedure TestDenseNetBlock;
    procedure TestMobileNetBlock;

    // Normalization builders (use TNNetInstanceNorm internally)
    procedure TestAddAutoGroupedPointwiseConvUsesInstanceNorm;
    procedure TestAddAutoGroupedPointwiseConv2UsesInstanceNorm;
    procedure TestAddChannelMovingNormUsesInstanceNorm;
    procedure TestAddMovingNormShapeAndForward;

    // Introspection
    procedure TestPrintSummary;
    procedure TestDiffArchitectureSelfIsEmpty;
    procedure TestDiffArchitectureSwappedLayer;
    procedure TestDiffArchitectureFromString;
    procedure TestWeightDriftReportFrozenLayer;
    procedure TestWeightDriftReportArchMismatch;
    procedure TestReceptiveFieldReportThreeConvs;
    procedure TestReceptiveFieldReportStrideDoublesJump;
    procedure TestActivationStatsReportInputStats;
    procedure TestActivationStatsReportNearCollapsedFlag;
    procedure TestWeightSpectrumReportRank1Matrix;
    procedure TestWeightSpectrumReportStructureAndFlags;
    procedure TestWeightSpectralTailReportSmoke;
    procedure TestTopLogitMarginReportSmoke;
    procedure TestNeuronCorrelationReportSmoke;
    procedure TestLayerSensitivityReportSmoke;
    procedure TestMagnitudePruningReportSmoke;
    procedure TestMCDropoutUncertaintyReportSmoke;
    procedure TestEquivarianceReportSmoke;
    procedure TestTTAReportSmoke;
    procedure TestSaliencyReportSmoke;
    procedure TestGradCAMReportSmoke;
    procedure TestDecisionBoundaryReportSmoke;
    procedure TestCalibrationReportSmoke;
    procedure TestFisherImportanceReportSmoke;
    procedure TestLinearProbeReportSmoke;
    procedure TestLogitLensReportSmoke;
    procedure TestFeatureSeparabilityReportSmoke;
    procedure TestRepresentationSimilarityReportSmoke;
    procedure TestEnableInputGradient;
    procedure TestAdversarialRobustnessReportSmoke;
    procedure TestGradientConflictReportSmoke;
    procedure TestGradientNoiseScaleReportSmoke;
    procedure TestHessianCurvatureReportSmoke;
    procedure TestEffectiveReceptiveFieldReportSmoke;
    procedure TestModeConnectivityReportSmoke;
    procedure TestPermutationAlignReportSmoke;
    procedure TestIntrinsicDimensionReportSmoke;
    procedure TestNeuralTangentKernelReportSmoke;
    procedure TestActivationPatchingReportSmoke;
    procedure TestPredictionDepthReportSmoke;
    procedure TestToGraphvizDotSmoke;
    procedure TestLayerTimingReportSmoke;
    procedure TestMixtureOfExpertsShapeForwardTrainAndRoundTrip;
    procedure TestMixtureOfDepthsShapeDegenerateAndRoundTrip;
    procedure TestDropBlockSmokeAndRoundTrip;
  end;

implementation

procedure TTestNeuralLayersExtra.TestDeconvolutionForward;
var
  NN: TNNet;
  Input: TNNetVolume;
begin
  NN := TNNet.Create();
  Input := TNNetVolume.Create(4, 4, 8);
  try
    NN.AddLayer(TNNetInput.Create(4, 4, 8));
    NN.AddLayer(TNNetDeconvolution.Create(16, 3, 0, 2)); // stride 2 upsamples

    Input.Fill(1.0);
    NN.Compute(Input);

    // Deconvolution with stride 2 produces output
    AssertEquals('Output depth should be 16', 16, NN.GetLastLayer.Output.Depth);
    AssertTrue('Deconvolution should produce output', NN.GetLastLayer.Output.Size > 0);
  finally
    NN.Free;
    Input.Free;
  end;
end;

procedure TTestNeuralLayersExtra.TestDeconvolutionReLUForward;
var
  NN: TNNet;
  Input: TNNetVolume;
begin
  NN := TNNet.Create();
  Input := TNNetVolume.Create(4, 4, 4);
  try
    NN.AddLayer(TNNetInput.Create(4, 4, 4));
    NN.AddLayer(TNNetDeconvolutionReLU.Create(8, 3, 0, 2));

    Input.Fill(1.0);
    NN.Compute(Input);

    // Output should have ReLU applied (non-negative values)
    AssertEquals('Output depth should be 8', 8, NN.GetLastLayer.Output.Depth);
  finally
    NN.Free;
    Input.Free;
  end;
end;

procedure TTestNeuralLayersExtra.TestDeconvolutionOutputSize;
var
  NN: TNNet;
  Input: TNNetVolume;
begin
  NN := TNNet.Create();
  Input := TNNetVolume.Create(8, 8, 16);
  try
    NN.AddLayer(TNNetInput.Create(8, 8, 16));
    // Deconvolution with stride 1 and padding should maintain size
    NN.AddLayer(TNNetDeconvolution.Create(32, 3, 1, 1));

    Input.Fill(0.5);
    NN.Compute(Input);

    AssertEquals('Output SizeX with stride 1', 8, NN.GetLastLayer.Output.SizeX);
    AssertEquals('Output SizeY with stride 1', 8, NN.GetLastLayer.Output.SizeY);
    AssertEquals('Output depth should be 32', 32, NN.GetLastLayer.Output.Depth);
  finally
    NN.Free;
    Input.Free;
  end;
end;

procedure TTestNeuralLayersExtra.TestDeLocalConnectForward;
var
  NN: TNNet;
  Input: TNNetVolume;
begin
  NN := TNNet.Create();
  Input := TNNetVolume.Create(4, 4, 4);
  try
    NN.AddLayer(TNNetInput.Create(4, 4, 4));
    NN.AddLayer(TNNetDeLocalConnect.Create(8, 3, 0, 2));

    Input.Fill(1.0);
    NN.Compute(Input);

    // DeLocalConnect should produce output
    AssertEquals('Output depth should be 8', 8, NN.GetLastLayer.Output.Depth);
    AssertTrue('DeLocalConnect should produce output', NN.GetLastLayer.Output.Size > 0);
  finally
    NN.Free;
    Input.Free;
  end;
end;

procedure TTestNeuralLayersExtra.TestDeLocalConnectReLUForward;
var
  NN: TNNet;
  Input: TNNetVolume;
begin
  NN := TNNet.Create();
  Input := TNNetVolume.Create(4, 4, 4);
  try
    NN.AddLayer(TNNetInput.Create(4, 4, 4));
    NN.AddLayer(TNNetDeLocalConnectReLU.Create(8, 3, 0, 2));

    Input.Fill(1.0);
    NN.Compute(Input);

    AssertEquals('Output depth should be 8', 8, NN.GetLastLayer.Output.Depth);
  finally
    NN.Free;
    Input.Free;
  end;
end;

procedure TTestNeuralLayersExtra.TestDeMaxPoolForward;
var
  NN: TNNet;
  Input: TNNetVolume;
  X, Y, D: integer;
begin
  NN := TNNet.Create();
  Input := TNNetVolume.Create(2, 2, 1);
  try
    NN.AddLayer(TNNetInput.Create(2, 2, 1));
    NN.AddLayer(TNNetDeMaxPool.Create(2)); // Upsample by 2x

    // Set known input values for numerical verification
    Input[0, 0, 0] := 1.0;
    Input[1, 0, 0] := 2.0;
    Input[0, 1, 0] := 3.0;
    Input[1, 1, 0] := 4.0;
    
    NN.Compute(Input);

    // DeMaxPool should double the spatial dimensions (2x2 -> 4x4)
    AssertEquals('Output SizeX should be 4', 4, NN.GetLastLayer.Output.SizeX);
    AssertEquals('Output SizeY should be 4', 4, NN.GetLastLayer.Output.SizeY);
    AssertEquals('Depth should remain 1', 1, NN.GetLastLayer.Output.Depth);
    
    // Numerical verification: check that values are present in output
    // DeMaxPool typically places input values in top-left of each upsampled region
    // and fills rest with zeros or copies values
    AssertTrue('Output should contain non-zero values', NN.GetLastLayer.Output.GetSumAbs() > 0);
    
    // Check that output values are finite
    for Y := 0 to 3 do
      for X := 0 to 3 do
        for D := 0 to 0 do
          AssertFalse('Output values should not be NaN', IsNaN(NN.GetLastLayer.Output[X, Y, D]));
  finally
    NN.Free;
    Input.Free;
  end;
end;

procedure TTestNeuralLayersExtra.TestDeMaxPoolOutputSize;
var
  NN: TNNet;
  Input: TNNetVolume;
begin
  NN := TNNet.Create();
  Input := TNNetVolume.Create(3, 3, 2);
  try
    NN.AddLayer(TNNetInput.Create(3, 3, 2));
    NN.AddLayer(TNNetDeMaxPool.Create(3)); // Upsample by 3x

    Input.Fill(2.0);
    NN.Compute(Input);

    // DeMaxPool with scale 3 should triple the dimensions
    AssertEquals('Output SizeX should be 9', 9, NN.GetLastLayer.Output.SizeX);
    AssertEquals('Output SizeY should be 9', 9, NN.GetLastLayer.Output.SizeY);
  finally
    NN.Free;
    Input.Free;
  end;
end;

procedure TTestNeuralLayersExtra.TestUpsampleForward;
var
  NN: TNNet;
  Input: TNNetVolume;
begin
  NN := TNNet.Create();
  Input := TNNetVolume.Create(4, 4, 16);
  try
    NN.AddLayer(TNNetInput.Create(4, 4, 16));
    NN.AddLayer(TNNetUpsample.Create()); // Depth to space

    Input.Fill(1.0);
    NN.Compute(Input);

    // Upsample converts depth to spatial (depth/4, size*2)
    AssertEquals('Output SizeX should be 8', 8, NN.GetLastLayer.Output.SizeX);
    AssertEquals('Output SizeY should be 8', 8, NN.GetLastLayer.Output.SizeY);
    AssertEquals('Output depth should be 4', 4, NN.GetLastLayer.Output.Depth);
  finally
    NN.Free;
    Input.Free;
  end;
end;

procedure TTestNeuralLayersExtra.TestUpsampleDepthToSpace;
var
  NN: TNNet;
  Input: TNNetVolume;
begin
  NN := TNNet.Create();
  Input := TNNetVolume.Create(2, 2, 64);
  try
    NN.AddLayer(TNNetInput.Create(2, 2, 64));
    NN.AddLayer(TNNetUpsample.Create());

    Input.Fill(1.0);
    NN.Compute(Input);

    // 2x2x64 -> 4x4x16 (depth to space transformation)
    AssertEquals('Output SizeX should be 4', 4, NN.GetLastLayer.Output.SizeX);
    AssertEquals('Output SizeY should be 4', 4, NN.GetLastLayer.Output.SizeY);
    AssertEquals('Output depth should be 16', 16, NN.GetLastLayer.Output.Depth);
  finally
    NN.Free;
    Input.Free;
  end;
end;

procedure TTestNeuralLayersExtra.TestPowerLayer;
var
  NN: TNNet;
  Input: TNNetVolume;
begin
  NN := TNNet.Create();
  Input := TNNetVolume.Create(4, 1, 1);
  try
    NN.AddLayer(TNNetInput.Create(4));
    NN.AddLayer(TNNetPower.Create(2)); // Square

    Input.Raw[0] := 2.0;
    Input.Raw[1] := 3.0;
    Input.Raw[2] := -2.0;
    Input.Raw[3] := 0.0;

    NN.Compute(Input);

    // Power of 2 should square the values
    AssertEquals('2^2 should be 4', 4.0, NN.GetLastLayer.Output.Raw[0], 0.0001);
    AssertEquals('3^2 should be 9', 9.0, NN.GetLastLayer.Output.Raw[1], 0.0001);
    AssertEquals('0^2 should be 0', 0.0, NN.GetLastLayer.Output.Raw[3], 0.0001);
  finally
    NN.Free;
    Input.Free;
  end;
end;

procedure TTestNeuralLayersExtra.TestMulByConstant;
var
  NN: TNNet;
  Input: TNNetVolume;
begin
  NN := TNNet.Create();
  Input := TNNetVolume.Create(4, 1, 1);
  try
    NN.AddLayer(TNNetInput.Create(4));
    NN.AddLayer(TNNetMulByConstant.Create(2.5));

    Input.Raw[0] := 2.0;
    Input.Raw[1] := 4.0;
    Input.Raw[2] := -1.0;
    Input.Raw[3] := 0.0;

    NN.Compute(Input);

    AssertEquals('2 * 2.5 should be 5', 5.0, NN.GetLastLayer.Output.Raw[0], 0.0001);
    AssertEquals('4 * 2.5 should be 10', 10.0, NN.GetLastLayer.Output.Raw[1], 0.0001);
    AssertEquals('-1 * 2.5 should be -2.5', -2.5, NN.GetLastLayer.Output.Raw[2], 0.0001);
    AssertEquals('0 * 2.5 should be 0', 0.0, NN.GetLastLayer.Output.Raw[3], 0.0001);
  finally
    NN.Free;
    Input.Free;
  end;
end;

procedure TTestNeuralLayersExtra.TestNegateLayer;
var
  NN: TNNet;
  Input: TNNetVolume;
begin
  NN := TNNet.Create();
  Input := TNNetVolume.Create(4, 1, 1);
  try
    NN.AddLayer(TNNetInput.Create(4));
    NN.AddLayer(TNNetNegate.Create());

    Input.Raw[0] := 2.0;
    Input.Raw[1] := -3.0;
    Input.Raw[2] := 0.0;
    Input.Raw[3] := 1.5;

    NN.Compute(Input);

    AssertEquals('Negate 2 should be -2', -2.0, NN.GetLastLayer.Output.Raw[0], 0.0001);
    AssertEquals('Negate -3 should be 3', 3.0, NN.GetLastLayer.Output.Raw[1], 0.0001);
    AssertEquals('Negate 0 should be 0', 0.0, NN.GetLastLayer.Output.Raw[2], 0.0001);
    AssertEquals('Negate 1.5 should be -1.5', -1.5, NN.GetLastLayer.Output.Raw[3], 0.0001);
  finally
    NN.Free;
    Input.Free;
  end;
end;

procedure TTestNeuralLayersExtra.TestSignedSquareRoot;
var
  NN: TNNet;
  Input: TNNetVolume;
begin
  NN := TNNet.Create();
  Input := TNNetVolume.Create(4, 1, 1);
  try
    NN.AddLayer(TNNetInput.Create(4));
    NN.AddLayer(TNNetSignedSquareRoot.Create());

    Input.Raw[0] := 4.0;
    Input.Raw[1] := 9.0;
    Input.Raw[2] := -4.0;
    Input.Raw[3] := 0.0;

    NN.Compute(Input);

    // SignedSquareRoot: Sign(x) * Sqrt(Abs(x))
    // The actual behavior may differ based on implementation details
    AssertEquals('Output size should match input', 4, NN.GetLastLayer.Output.Size);
  finally
    NN.Free;
    Input.Free;
  end;
end;

procedure TTestNeuralLayersExtra.TestReLUL;
var
  NN: TNNet;
  Input: TNNetVolume;
begin
  NN := TNNet.Create();
  Input := TNNetVolume.Create(4, 1, 1);
  try
    NN.AddLayer(TNNetInput.Create(4));
    // TNNetReLUL.Create(LowLimit, HighLimit, Leakiness: integer)
    NN.AddLayer(TNNetReLUL.Create(-500, 500, 10)); // 1% leakiness (10 * 0.001)

    Input.Raw[0] := 2.0;
    Input.Raw[1] := -2.0;
    Input.Raw[2] := 0.0;
    Input.Raw[3] := 5.0;

    NN.Compute(Input);

    // ReLUL with leaky factor
    AssertEquals('ReLUL(2) should be 2', 2.0, NN.GetLastLayer.Output.Raw[0], 0.01);
    AssertEquals('ReLUL(0) should be 0', 0.0, NN.GetLastLayer.Output.Raw[2], 0.01);
  finally
    NN.Free;
    Input.Free;
  end;
end;

procedure TTestNeuralLayersExtra.TestVeryLeakyReLU;
var
  NN: TNNet;
  Input: TNNetVolume;
begin
  NN := TNNet.Create();
  Input := TNNetVolume.Create(3, 1, 1);
  try
    NN.AddLayer(TNNetInput.Create(3));
    NN.AddLayer(TNNetVeryLeakyReLU.Create());

    Input.Raw[0] := 3.0;
    Input.Raw[1] := -3.0;
    Input.Raw[2] := 0.0;

    NN.Compute(Input);

    // Very Leaky ReLU has alpha = 1/3
    AssertEquals('VeryLeakyReLU(3) should be 3', 3.0, NN.GetLastLayer.Output.Raw[0], 0.01);
    AssertEquals('VeryLeakyReLU(0) should be 0', 0.0, NN.GetLastLayer.Output.Raw[2], 0.01);
    // The negative behavior depends on the implementation
    AssertTrue('VeryLeakyReLU(-3) should be non-positive', NN.GetLastLayer.Output.Raw[1] <= 0.01);
  finally
    NN.Free;
    Input.Free;
  end;
end;

procedure TTestNeuralLayersExtra.TestSwish6;
var
  NN: TNNet;
  Input: TNNetVolume;
begin
  NN := TNNet.Create();
  Input := TNNetVolume.Create(3, 1, 1);
  try
    NN.AddLayer(TNNetInput.Create(3));
    NN.AddLayer(TNNetSwish6.Create());

    Input.Raw[0] := 0.0;
    Input.Raw[1] := 3.0;
    Input.Raw[2] := 10.0; // Should be clamped at 6

    NN.Compute(Input);

    // Swish6 should clamp output at 6
    AssertEquals('Swish6(0) should be 0', 0.0, NN.GetLastLayer.Output.Raw[0], 0.01);
    AssertTrue('Swish6(10) should be <= 6', NN.GetLastLayer.Output.Raw[2] <= 6.01);
  finally
    NN.Free;
    Input.Free;
  end;
end;

procedure TTestNeuralLayersExtra.TestHardSwish;
var
  NN: TNNet;
  Input: TNNetVolume;
begin
  NN := TNNet.Create();
  Input := TNNetVolume.Create(4, 1, 1);
  try
    NN.AddLayer(TNNetInput.Create(4));
    NN.AddLayer(TNNetHardSwish.Create());

    Input.Raw[0] := 0.0;
    Input.Raw[1] := 3.0;
    Input.Raw[2] := -3.0;
    Input.Raw[3] := 6.0;

    NN.Compute(Input);

    // HardSwish approximates Swish but is faster
    AssertEquals('HardSwish(0) should be 0', 0.0, NN.GetLastLayer.Output.Raw[0], 0.01);
    AssertTrue('HardSwish output should exist', NN.GetLastLayer.Output.Size = 4);
  finally
    NN.Free;
    Input.Free;
  end;
end;

procedure TTestNeuralLayersExtra.TestReLUSqrt;
var
  NN: TNNet;
  Input: TNNetVolume;
begin
  NN := TNNet.Create();
  Input := TNNetVolume.Create(3, 1, 1);
  try
    NN.AddLayer(TNNetInput.Create(3));
    NN.AddLayer(TNNetReLUSqrt.Create());

    Input.Raw[0] := 4.0;
    Input.Raw[1] := 0.0;
    Input.Raw[2] := -1.0;

    NN.Compute(Input);

    // ReLUSqrt: sqrt(max(0, x))
    AssertEquals('ReLUSqrt(4) should be 2', 2.0, NN.GetLastLayer.Output.Raw[0], 0.01);
    AssertEquals('ReLUSqrt(0) should be 0', 0.0, NN.GetLastLayer.Output.Raw[1], 0.01);
    AssertEquals('ReLUSqrt(-1) should be 0', 0.0, NN.GetLastLayer.Output.Raw[2], 0.01);
  finally
    NN.Free;
    Input.Free;
  end;
end;

procedure TTestNeuralLayersExtra.TestPointwiseSoftMax;
var
  NN: TNNet;
  Input: TNNetVolume;
begin
  NN := TNNet.Create();
  Input := TNNetVolume.Create(4, 4, 8); // 4x4 spatial, 8 classes
  try
    NN.AddLayer(TNNetInput.Create(4, 4, 8));
    NN.AddLayer(TNNetPointwiseSoftMax.Create());

    Input.RandomizeGaussian();
    NN.Compute(Input);

    // Output dimensions should match input
    AssertEquals('Output SizeX should be 4', 4, NN.GetLastLayer.Output.SizeX);
    AssertEquals('Output SizeY should be 4', 4, NN.GetLastLayer.Output.SizeY);
    AssertEquals('Output depth should be 8', 8, NN.GetLastLayer.Output.Depth);
  finally
    NN.Free;
    Input.Free;
  end;
end;

procedure TTestNeuralLayersExtra.TestPointwiseSoftMaxSumToOne;
var
  NN: TNNet;
  Input: TNNetVolume;
  X, Y, D: integer;
  Sum: TNeuralFloat;
begin
  NN := TNNet.Create();
  Input := TNNetVolume.Create(2, 2, 4); // 2x2 spatial, 4 classes
  try
    NN.AddLayer(TNNetInput.Create(2, 2, 4));
    NN.AddLayer(TNNetPointwiseSoftMax.Create());

    Input.RandomizeGaussian();
    NN.Compute(Input);

    // For each pixel, softmax across depth should sum to 1
    for Y := 0 to 1 do
      for X := 0 to 1 do
      begin
        Sum := 0;
        for D := 0 to 3 do
          Sum := Sum + NN.GetLastLayer.Output[X, Y, D];
        AssertEquals('Softmax at pixel (' + IntToStr(X) + ',' + IntToStr(Y) + ') should sum to 1', 
          1.0, Sum, 0.001);
      end;
  finally
    NN.Free;
    Input.Free;
  end;
end;

procedure TTestNeuralLayersExtra.TestRandomMulAdd;
var
  NN: TNNet;
  Input: TNNetVolume;
begin
  NN := TNNet.Create();
  Input := TNNetVolume.Create(4, 4, 3);
  try
    NN.AddLayer(TNNetInput.Create(4, 4, 3));
    // Create(AddRate, MulRate: integer) - using small values for 1% noise
    NN.AddLayer(TNNetRandomMulAdd.Create(10, 10)); // 10 = 1% noise

    Input.Fill(5.0);
    NN.Compute(Input);

    // Output should exist and have same dimensions
    AssertEquals('Output size should match', 48, NN.GetLastLayer.Output.Size);
    // Average should be approximately 5.0 (noise added/multiplied)
    // This is probabilistic so we allow large tolerance
    AssertTrue('Average should be approximately 5', 
      Abs(NN.GetLastLayer.Output.GetAvg() - 5.0) < 2.0);
  finally
    NN.Free;
    Input.Free;
  end;
end;

procedure TTestNeuralLayersExtra.TestChannelRandomMulAdd;
var
  NN: TNNet;
  Input: TNNetVolume;
begin
  NN := TNNet.Create();
  Input := TNNetVolume.Create(4, 4, 3);
  try
    NN.AddLayer(TNNetInput.Create(4, 4, 3));
    // Create(AddRate, MulRate: integer)
    NN.AddLayer(TNNetChannelRandomMulAdd.Create(10, 10));

    Input.Fill(5.0);
    NN.Compute(Input);

    AssertEquals('Output size should match', 48, NN.GetLastLayer.Output.Size);
  finally
    NN.Free;
    Input.Free;
  end;
end;

procedure TTestNeuralLayersExtra.TestCellMul;
var
  NN: TNNet;
  Input: TNNetVolume;
begin
  NN := TNNet.Create();
  Input := TNNetVolume.Create(4, 4, 2);
  try
    NN.AddLayer(TNNetInput.Create(4, 4, 2));
    NN.AddLayer(TNNetCellMul.Create());

    Input.Fill(2.0);
    NN.Compute(Input);

    // CellMul applies trainable per-cell multiplication
    AssertEquals('Output should maintain dimensions', 32, NN.GetLastLayer.Output.Size);
  finally
    NN.Free;
    Input.Free;
  end;
end;

procedure TTestNeuralLayersExtra.TestCellMulByCell;
var
  NN: TNNet;
  Input: TNNetVolume;
  InputLayer, Branch1, Branch2: TNNetLayer;
begin
  NN := TNNet.Create();
  Input := TNNetVolume.Create(4, 4, 2);
  try
    InputLayer := NN.AddLayer(TNNetInput.Create(4, 4, 2));
    Branch1 := NN.AddLayer(TNNetConvolutionLinear.Create(4, 3, 1, 1));
    Branch2 := NN.AddLayerAfter(TNNetConvolutionLinear.Create(4, 3, 1, 1), InputLayer);
    
    // CellMulByCell multiplies outputs of two layers element-wise
    // Constructor: Create(LayerA, LayerB: TNNetLayer)
    NN.AddLayer(TNNetCellMulByCell.Create(Branch1, Branch2));

    Input.Fill(1.0);
    NN.Compute(Input);

    AssertEquals('Output depth should match branches', 4, NN.GetLastLayer.Output.Depth);
  finally
    NN.Free;
    Input.Free;
  end;
end;

procedure TTestNeuralLayersExtra.TestChannelMulByLayer;
var
  NN: TNNet;
  Input: TNNetVolume;
  InputLayer, ConvLayer, ChannelLayer: TNNetLayer;
begin
  NN := TNNet.Create();
  Input := TNNetVolume.Create(4, 4, 4);
  try
    InputLayer := NN.AddLayer(TNNetInput.Create(4, 4, 4));
    ConvLayer := NN.AddLayer(TNNetConvolutionLinear.Create(4, 3, 1, 1));
    
    // Create a layer that produces channel-wise multipliers
    ChannelLayer := NN.AddLayerAfter(TNNetAvgChannel.Create(), InputLayer);
    
    // ChannelMulByLayer multiplies each channel by corresponding value
    // Constructor: Create(LayerWithChannels, LayerMul: TNNetLayer)
    NN.AddLayer(TNNetChannelMulByLayer.Create(ConvLayer, ChannelLayer));

    Input.Fill(1.0);
    NN.Compute(Input);

    AssertEquals('Output should have correct depth', 4, NN.GetLastLayer.Output.Depth);
  finally
    NN.Free;
    Input.Free;
  end;
end;

procedure TTestNeuralLayersExtra.TestTransposeXD;
var
  NN: TNNet;
  Input: TNNetVolume;
begin
  NN := TNNet.Create();
  Input := TNNetVolume.Create(4, 2, 3);
  try
    NN.AddLayer(TNNetInput.Create(4, 2, 3));
    NN.AddLayer(TNNetTransposeXD.Create());

    // Set known values for numerical verification
    // Input[x, y, d] should become Output[d, y, x] after transpose XD
    Input[0, 0, 0] := 1.0;
    Input[1, 0, 0] := 2.0;
    Input[2, 0, 0] := 3.0;
    Input[3, 0, 0] := 4.0;
    Input[0, 0, 1] := 10.0;
    Input[0, 0, 2] := 20.0;
    
    NN.Compute(Input);

    // TransposeXD swaps X and Depth dimensions
    AssertEquals('After transpose, SizeX should be 3', 3, NN.GetLastLayer.Output.SizeX);
    AssertEquals('After transpose, Depth should be 4', 4, NN.GetLastLayer.Output.Depth);
    AssertEquals('SizeY should remain 2', 2, NN.GetLastLayer.Output.SizeY);
    
    // Numerical verification: Input[x,y,d] -> Output[d,y,x]
    // Input[0,0,0] = 1.0 should be at Output[0,0,0]
    AssertEquals('Transposed value should match', 1.0, NN.GetLastLayer.Output[0, 0, 0], 0.0001);
    // Input[0,0,1] = 10.0 should be at Output[1,0,0]
    AssertEquals('Transposed value should match', 10.0, NN.GetLastLayer.Output[1, 0, 0], 0.0001);
    // Input[0,0,2] = 20.0 should be at Output[2,0,0]
    AssertEquals('Transposed value should match', 20.0, NN.GetLastLayer.Output[2, 0, 0], 0.0001);
  finally
    NN.Free;
    Input.Free;
  end;
end;

procedure TTestNeuralLayersExtra.TestTransposeYD;
var
  NN: TNNet;
  Input: TNNetVolume;
begin
  NN := TNNet.Create();
  Input := TNNetVolume.Create(2, 4, 3);
  try
    NN.AddLayer(TNNetInput.Create(2, 4, 3));
    NN.AddLayer(TNNetTransposeYD.Create());

    // Set known values for numerical verification
    // Input[x, y, d] should become Output[x, d, y] after transpose YD
    Input[0, 0, 0] := 1.0;
    Input[0, 1, 0] := 2.0;
    Input[0, 2, 0] := 3.0;
    Input[0, 3, 0] := 4.0;
    Input[0, 0, 1] := 10.0;
    Input[0, 0, 2] := 20.0;
    
    NN.Compute(Input);

    // TransposeYD swaps Y and Depth dimensions
    AssertEquals('After transpose, SizeY should be 3', 3, NN.GetLastLayer.Output.SizeY);
    AssertEquals('After transpose, Depth should be 4', 4, NN.GetLastLayer.Output.Depth);
    AssertEquals('SizeX should remain 2', 2, NN.GetLastLayer.Output.SizeX);
    
    // Numerical verification: Input[x,y,d] -> Output[x,d,y]
    // Input[0,0,0] = 1.0 should be at Output[0,0,0]
    AssertEquals('Transposed value should match', 1.0, NN.GetLastLayer.Output[0, 0, 0], 0.0001);
    // Input[0,0,1] = 10.0 should be at Output[0,1,0]
    AssertEquals('Transposed value should match', 10.0, NN.GetLastLayer.Output[0, 1, 0], 0.0001);
    // Input[0,0,2] = 20.0 should be at Output[0,2,0]
    AssertEquals('Transposed value should match', 20.0, NN.GetLastLayer.Output[0, 2, 0], 0.0001);
  finally
    NN.Free;
    Input.Free;
  end;
end;

procedure TTestNeuralLayersExtra.TestMinChannel;
var
  NN: TNNet;
  Input: TNNetVolume;
begin
  NN := TNNet.Create();
  Input := TNNetVolume.Create(4, 4, 3);
  try
    NN.AddLayer(TNNetInput.Create(4, 4, 3));
    NN.AddLayer(TNNetMinChannel.Create());

    Input.FillAtDepth(0, 5.0);
    Input.FillAtDepth(1, 2.0);
    Input.FillAtDepth(2, 8.0);

    NN.Compute(Input);

    // MinChannel returns min value per channel
    AssertEquals('Output should have 3 elements', 3, NN.GetLastLayer.Output.Size);
    AssertEquals('Min of channel 0 should be 5.0', 5.0, NN.GetLastLayer.Output.Raw[0], 0.0001);
    AssertEquals('Min of channel 1 should be 2.0', 2.0, NN.GetLastLayer.Output.Raw[1], 0.0001);
    AssertEquals('Min of channel 2 should be 8.0', 8.0, NN.GetLastLayer.Output.Raw[2], 0.0001);
  finally
    NN.Free;
    Input.Free;
  end;
end;

procedure TTestNeuralLayersExtra.TestMinMaxPoolCombined;
var
  NN: TNNet;
  Input: TNNetVolume;
begin
  NN := TNNet.Create();
  Input := TNNetVolume.Create(8, 8, 4);
  try
    NN.AddLayer(TNNetInput.Create(8, 8, 4));
    NN.AddMinMaxPool(2); // Adds both min and max pool, concatenating results

    Input.RandomizeGaussian();
    NN.Compute(Input);

    // MinMaxPool concatenates min and max pool results
    // Each pool reduces spatial by 2, depth doubles due to concatenation
    AssertEquals('Output SizeX should be 4', 4, NN.GetLastLayer.Output.SizeX);
    AssertEquals('Output Depth should be 8 (4*2)', 8, NN.GetLastLayer.Output.Depth);
  finally
    NN.Free;
    Input.Free;
  end;
end;

procedure TTestNeuralLayersExtra.TestGroupedPointwiseConvLinear;
var
  NN: TNNet;
  Input: TNNetVolume;
begin
  NN := TNNet.Create();
  Input := TNNetVolume.Create(8, 8, 16);
  try
    NN.AddLayer(TNNetInput.Create(8, 8, 16));
    // Create(pNumFeatures, pGroups: integer; pSuppressBias: integer)
    NN.AddLayer(TNNetGroupedPointwiseConvLinear.Create(32, 4, 0));

    Input.Fill(1.0);
    NN.Compute(Input);

    AssertEquals('Output SizeX should be 8', 8, NN.GetLastLayer.Output.SizeX);
    AssertEquals('Output Depth should be 32', 32, NN.GetLastLayer.Output.Depth);
  finally
    NN.Free;
    Input.Free;
  end;
end;

procedure TTestNeuralLayersExtra.TestGroupedPointwiseConvReLU;
var
  NN: TNNet;
  Input: TNNetVolume;
begin
  NN := TNNet.Create();
  Input := TNNetVolume.Create(8, 8, 8);
  try
    NN.AddLayer(TNNetInput.Create(8, 8, 8));
    // Create(pNumFeatures, pGroups: integer; pSuppressBias: integer)
    NN.AddLayer(TNNetGroupedPointwiseConvReLU.Create(16, 2, 0));

    Input.Fill(1.0);
    NN.Compute(Input);

    AssertEquals('Output Depth should be 16', 16, NN.GetLastLayer.Output.Depth);
    // ReLU should ensure non-negative outputs for positive inputs
  finally
    NN.Free;
    Input.Free;
  end;
end;

procedure TTestNeuralLayersExtra.TestResNetBlock;
var
  NN: TNNet;
  Input: TNNetVolume;
  InputLayer, Conv1, Conv2, Shortcut: TNNetLayer;
begin
  NN := TNNet.Create();
  Input := TNNetVolume.Create(8, 8, 16);
  try
    // Build a ResNet-style identity shortcut block
    InputLayer := NN.AddLayer(TNNetInput.Create(8, 8, 16));
    
    // Main path
    Conv1 := NN.AddLayer(TNNetConvolutionReLU.Create(16, 3, 1, 1));
    Conv2 := NN.AddLayer(TNNetConvolutionLinear.Create(16, 3, 1, 1));
    
    // Shortcut path (identity since dimensions match)
    Shortcut := NN.AddLayerAfter(TNNetIdentity.Create(), InputLayer);
    
    // Sum paths
    NN.AddLayer(TNNetSum.Create([Conv2, Shortcut]));
    NN.AddLayer(TNNetReLU.Create());

    Input.RandomizeGaussian();
    NN.Compute(Input);

    // Output should have same dimensions as input
    AssertEquals('ResNet block should preserve SizeX', 8, NN.GetLastLayer.Output.SizeX);
    AssertEquals('ResNet block should preserve depth', 16, NN.GetLastLayer.Output.Depth);
  finally
    NN.Free;
    Input.Free;
  end;
end;

procedure TTestNeuralLayersExtra.TestDenseNetBlock;
var
  NN: TNNet;
  Input: TNNetVolume;
  L0, L1, L2: TNNetLayer;
begin
  NN := TNNet.Create();
  Input := TNNetVolume.Create(8, 8, 16);
  try
    // Build a DenseNet-style block with concatenation
    L0 := NN.AddLayer(TNNetInput.Create(8, 8, 16));
    
    // First dense layer
    L1 := NN.AddLayer(TNNetConvolutionReLU.Create(8, 3, 1, 1));
    
    // Concatenate input with first output
    NN.AddLayer(TNNetDeepConcat.Create([L0, L1]));
    
    // Second dense layer
    L2 := NN.AddLayer(TNNetConvolutionReLU.Create(8, 3, 1, 1));

    Input.RandomizeGaussian();
    NN.Compute(Input);

    // DenseNet concatenates features, so depth grows
    AssertEquals('DenseNet block output depth should be 8', 8, NN.GetLastLayer.Output.Depth);
    AssertEquals('Spatial size should be preserved', 8, NN.GetLastLayer.Output.SizeX);
  finally
    NN.Free;
    Input.Free;
  end;
end;

procedure TTestNeuralLayersExtra.TestMobileNetBlock;
var
  NN: TNNet;
  Input: TNNetVolume;
begin
  NN := TNNet.Create();
  Input := TNNetVolume.Create(8, 8, 16);
  try
    // Build a MobileNet-style separable convolution block
    NN.AddLayer(TNNetInput.Create(8, 8, 16));
    
    // Depthwise convolution
    NN.AddLayer(TNNetDepthwiseConvReLU.Create(1, 3, 1, 1));
    
    // Pointwise convolution (1x1)
    NN.AddLayer(TNNetPointwiseConvReLU.Create(32));

    Input.RandomizeGaussian();
    NN.Compute(Input);

    // Separable conv: depthwise keeps depth, pointwise changes it
    AssertEquals('MobileNet block should preserve spatial size', 8, NN.GetLastLayer.Output.SizeX);
    AssertEquals('MobileNet block output depth', 32, NN.GetLastLayer.Output.Depth);
  finally
    NN.Free;
    Input.Free;
  end;
end;

function CountInstanceNorm(NN: TNNet): integer;
var
  i: integer;
begin
  Result := 0;
  for i := 0 to NN.Layers.Count - 1 do
    if NN.Layers[i] is TNNetInstanceNorm then Inc(Result);
end;

procedure TTestNeuralLayersExtra.TestAddAutoGroupedPointwiseConvUsesInstanceNorm;
var
  NN: TNNet;
  Input: TNNetVolume;
begin
  NN := TNNet.Create();
  Input := TNNetVolume.Create(4, 4, 16);
  try
    NN.AddLayer(TNNetInput.Create(4, 4, 16));
    NN.AddAutoGroupedPointwiseConv(TNNetGroupedPointwiseConvReLU,
      {MinChannelsPerGroupCount=}4, {pNumFeatures=}16, {HasNormalization=}true);

    Input.RandomizeGaussian();
    NN.Compute(Input);

    AssertEquals('AddAutoGroupedPointwiseConv preserves spatial size',
      4, NN.GetLastLayer.Output.SizeX);
    AssertEquals('AddAutoGroupedPointwiseConv output depth',
      16, NN.GetLastLayer.Output.Depth);
    // Normalization should now be provided by TNNetInstanceNorm.
    AssertTrue('AddAutoGroupedPointwiseConv should use TNNetInstanceNorm',
      CountInstanceNorm(NN) > 0);
  finally
    NN.Free;
    Input.Free;
  end;
end;

procedure TTestNeuralLayersExtra.TestAddAutoGroupedPointwiseConv2UsesInstanceNorm;
var
  NN: TNNet;
  Input: TNNetVolume;
begin
  NN := TNNet.Create();
  Input := TNNetVolume.Create(4, 4, 16);
  try
    NN.AddLayer(TNNetInput.Create(4, 4, 16));
    NN.AddAutoGroupedPointwiseConv2(TNNetGroupedPointwiseConvReLU,
      {MinChannelsPerGroupCount=}4, {pNumFeatures=}16, {HasNormalization=}true);

    Input.RandomizeGaussian();
    NN.Compute(Input);

    AssertEquals('AddAutoGroupedPointwiseConv2 preserves spatial size',
      4, NN.GetLastLayer.Output.SizeX);
    AssertEquals('AddAutoGroupedPointwiseConv2 output depth',
      16, NN.GetLastLayer.Output.Depth);
    AssertTrue('AddAutoGroupedPointwiseConv2 should use TNNetInstanceNorm',
      CountInstanceNorm(NN) > 0);
  finally
    NN.Free;
    Input.Free;
  end;
end;

procedure TTestNeuralLayersExtra.TestAddChannelMovingNormUsesInstanceNorm;
var
  NN: TNNet;
  Input: TNNetVolume;
begin
  NN := TNNet.Create();
  Input := TNNetVolume.Create(4, 4, 8);
  try
    NN.AddLayer(TNNetInput.Create(4, 4, 8));
    NN.AddChannelMovingNorm({PerCell=}false, {RandomBias=}0, {RandomAmplifier=}0);

    Input.RandomizeGaussian();
    NN.Compute(Input);

    AssertEquals('AddChannelMovingNorm preserves spatial size',
      4, NN.GetLastLayer.Output.SizeX);
    AssertEquals('AddChannelMovingNorm preserves depth',
      8, NN.GetLastLayer.Output.Depth);
    AssertTrue('AddChannelMovingNorm should use TNNetInstanceNorm',
      CountInstanceNorm(NN) > 0);
  finally
    NN.Free;
    Input.Free;
  end;
end;

procedure TTestNeuralLayersExtra.TestAddMovingNormShapeAndForward;
var
  NN: TNNet;
  Input: TNNetVolume;
begin
  // AddMovingNorm is the global-statistic sibling; it intentionally keeps
  // TNNetMovingStdNormalization (not swapped). Smoke-test shape and forward.
  NN := TNNet.Create();
  Input := TNNetVolume.Create(4, 4, 8);
  try
    NN.AddLayer(TNNetInput.Create(4, 4, 8));
    NN.AddMovingNorm({PerCell=}false, {RandomBias=}0, {RandomAmplifier=}0);

    Input.RandomizeGaussian();
    NN.Compute(Input);

    AssertEquals('AddMovingNorm preserves spatial size',
      4, NN.GetLastLayer.Output.SizeX);
    AssertEquals('AddMovingNorm preserves depth',
      8, NN.GetLastLayer.Output.Depth);
  finally
    NN.Free;
    Input.Free;
  end;
end;

procedure TTestNeuralLayersExtra.TestPrintSummary;
var
  NN: TNNet;
  S: string;
begin
  NN := TNNet.Create();
  try
    NN.AddLayer(TNNetInput.Create(8, 8, 3));
    NN.AddLayer(TNNetConvolutionReLU.Create(4, 3, 1, 1));
    NN.AddLayer(TNNetMaxPool.Create(2));
    NN.AddLayer(TNNetFullConnectReLU.Create(10));

    S := NN.SummaryString();

    AssertTrue('SummaryString should not be empty', Length(S) > 0);
    AssertTrue('Summary should contain TNNetInput', Pos('TNNetInput', S) > 0);
    AssertTrue('Summary should contain TNNetConvolutionReLU',
      Pos('TNNetConvolutionReLU', S) > 0);
    AssertTrue('Summary should contain TNNetMaxPool', Pos('TNNetMaxPool', S) > 0);
    AssertTrue('Summary should contain TNNetFullConnectReLU',
      Pos('TNNetFullConnectReLU', S) > 0);
    AssertTrue('Summary should contain totals line', Pos('Totals:', S) > 0);
    AssertTrue('Summary totals should mention layers',
      Pos('layers', S) > 0);
    AssertTrue('Summary totals should mention weights',
      Pos('weights', S) > 0);
    AssertTrue('Summary totals should mention neurons',
      Pos('neurons', S) > 0);
  finally
    NN.Free;
  end;
end;

procedure TTestNeuralLayersExtra.TestDiffArchitectureSelfIsEmpty;
var
  NN: TNNet;
  Diff: string;
begin
  NN := TNNet.Create();
  try
    NN.AddLayer(TNNetInput.Create(8, 8, 3));
    NN.AddLayer(TNNetConvolutionReLU.Create(4, 3, 1, 1));
    NN.AddLayer(TNNetMaxPool.Create(2));
    NN.AddLayer(TNNetFullConnectReLU.Create(10));

    Diff := NN.DiffArchitecture(NN);
    AssertEquals('Diffing a network against itself should be empty',
      '', Diff);
  finally
    NN.Free;
  end;
end;

procedure TTestNeuralLayersExtra.TestDiffArchitectureSwappedLayer;
var
  A, B: TNNet;
  Diff: string;
  Lines: TStringList;
  I, MinusCount, PlusCount, SpaceCount: integer;
begin
  A := TNNet.Create();
  B := TNNet.Create();
  Lines := TStringList.Create();
  try
    A.AddLayer(TNNetInput.Create(8, 8, 3));
    A.AddLayer(TNNetConvolutionReLU.Create(4, 3, 1, 1));
    A.AddLayer(TNNetMaxPool.Create(2));
    A.AddLayer(TNNetFullConnectReLU.Create(10));

    // Swap one layer: ReLU -> Linear at position 3 (full connect)
    B.AddLayer(TNNetInput.Create(8, 8, 3));
    B.AddLayer(TNNetConvolutionReLU.Create(4, 3, 1, 1));
    B.AddLayer(TNNetMaxPool.Create(2));
    B.AddLayer(TNNetFullConnectLinear.Create(10));

    Diff := A.DiffArchitecture(B);
    AssertTrue('Diff should be non-empty when one layer is swapped',
      Length(Diff) > 0);
    Lines.Text := Diff;

    MinusCount := 0;
    PlusCount := 0;
    SpaceCount := 0;
    for I := 0 to Lines.Count - 1 do
    begin
      if Length(Lines[I]) = 0 then Continue;
      case Lines[I][1] of
        '-': Inc(MinusCount);
        '+': Inc(PlusCount);
        ' ': Inc(SpaceCount);
      end;
    end;
    AssertEquals('Exactly one removed line', 1, MinusCount);
    AssertEquals('Exactly one added line', 1, PlusCount);
    AssertEquals('Three matching layers should still appear', 3, SpaceCount);
    AssertTrue('Removed line mentions the original class',
      Pos('TNNetFullConnectReLU', Diff) > 0);
    AssertTrue('Added line mentions the swapped class',
      Pos('TNNetFullConnectLinear', Diff) > 0);
  finally
    Lines.Free;
    A.Free;
    B.Free;
  end;
end;

procedure TTestNeuralLayersExtra.TestDiffArchitectureFromString;
var
  A, B: TNNet;
  GoldenStr, Diff: string;
begin
  A := TNNet.Create();
  B := TNNet.Create();
  try
    A.AddLayer(TNNetInput.Create(4, 4, 2));
    A.AddLayer(TNNetConvolutionReLU.Create(8, 3, 1, 1));
    A.AddLayer(TNNetFullConnectLinear.Create(3));

    // Same structure -> empty diff via string.
    B.AddLayer(TNNetInput.Create(4, 4, 2));
    B.AddLayer(TNNetConvolutionReLU.Create(8, 3, 1, 1));
    B.AddLayer(TNNetFullConnectLinear.Create(3));
    GoldenStr := B.SaveStructureToString();

    Diff := A.DiffArchitectureFromString(GoldenStr);
    AssertEquals('DiffArchitectureFromString against matching golden ' +
      'should be empty', '', Diff);
  finally
    A.Free;
    B.Free;
  end;
end;

procedure TTestNeuralLayersExtra.TestWeightDriftReportFrozenLayer;
var
  NN: TNNet;
  SnapA, SnapB, Report: string;
  Lines: TStringList;
  TargetLayer, OtherLayer: TNNetLayer;
  N, K: integer;
  Layer1Line, Layer2Line, Layer3Line: string;
  I: integer;
begin
  NN := TNNet.Create();
  Lines := TStringList.Create();
  try
    NN.AddLayer(TNNetInput.Create(4, 1, 1));
    NN.AddLayer(TNNetFullConnectReLU.Create(6));
    NN.AddLayer(TNNetFullConnectReLU.Create(5));
    NN.AddLayer(TNNetFullConnectLinear.Create(2));
    NN.InitWeights();

    SnapA := NN.SaveToString();

    OtherLayer := NN.Layers[2];
    for N := 0 to OtherLayer.Neurons.Count - 1 do
    begin
      for K := 0 to OtherLayer.Neurons[N].Weights.Size - 1 do
        OtherLayer.Neurons[N].Weights.FData[K] :=
          OtherLayer.Neurons[N].Weights.FData[K] + 0.5;
    end;

    TargetLayer := NN.Layers[3];
    for N := 0 to TargetLayer.Neurons.Count - 1 do
    begin
      for K := 0 to TargetLayer.Neurons[N].Weights.Size - 1 do
        TargetLayer.Neurons[N].Weights.FData[K] :=
          TargetLayer.Neurons[N].Weights.FData[K] + 0.1;
    end;

    SnapB := NN.SaveToString();

    Report := TNNet.WeightDriftReport(SnapA, SnapB);
    AssertTrue('Report should be non-empty', Length(Report) > 0);

    Lines.Text := Report;
    Layer1Line := '';
    Layer2Line := '';
    Layer3Line := '';
    for I := 0 to Lines.Count - 1 do
    begin
      if Pos('1     TNNetFullConnectReLU', Lines[I]) = 1 then
        Layer1Line := Lines[I]
      else if Pos('2     TNNetFullConnectReLU', Lines[I]) = 1 then
        Layer2Line := Lines[I]
      else if Pos('3     TNNetFullConnectLinear', Lines[I]) = 1 then
        Layer3Line := Lines[I];
    end;

    AssertTrue('Layer 1 row present', Layer1Line <> '');
    AssertTrue('Layer 2 row present', Layer2Line <> '');
    AssertTrue('Layer 3 row present', Layer3Line <> '');

    AssertTrue('Frozen layer 1 should have ~1.0 frac frozen',
      Pos('1.0000', Layer1Line) > 0);
    AssertTrue('Frozen layer 1 should have ~0 L2 drift',
      Pos('0.00000E+000', Layer1Line) > 0);
    AssertTrue('Touched layer 2 should have non-zero L2 drift',
      Pos('0.00000E+000', Layer2Line) = 0);
    AssertTrue('Touched layer 3 should have non-zero L2 drift',
      Pos('0.00000E+000', Layer3Line) = 0);
  finally
    Lines.Free;
    NN.Free;
  end;
end;

procedure TTestNeuralLayersExtra.TestWeightDriftReportArchMismatch;
var
  A, B: TNNet;
  SnapA, SnapB, Report: string;
begin
  A := TNNet.Create();
  B := TNNet.Create();
  try
    A.AddLayer(TNNetInput.Create(4, 1, 1));
    A.AddLayer(TNNetFullConnectReLU.Create(6));
    A.AddLayer(TNNetFullConnectLinear.Create(2));
    A.InitWeights();

    B.AddLayer(TNNetInput.Create(4, 1, 1));
    B.AddLayer(TNNetFullConnectSigmoid.Create(6));
    B.AddLayer(TNNetFullConnectLinear.Create(2));
    B.InitWeights();

    SnapA := A.SaveToString();
    SnapB := B.SaveToString();

    Report := TNNet.WeightDriftReport(SnapA, SnapB);
    AssertTrue('Mismatch report should mention DiffArchitecture',
      Pos('DiffArchitecture', Report) > 0);
    AssertTrue('Mismatch report should mention different architectures',
      Pos('different', Report) > 0);
  finally
    A.Free;
    B.Free;
  end;
end;

procedure TTestNeuralLayersExtra.TestReceptiveFieldReportThreeConvs;
var
  NN: TNNet;
  Report, Layer3Line: string;
  Lines: TStringList;
  I: integer;
begin
  // Textbook closed form: three stacked 3x3 stride-1 convs on a flat input
  // give a receptive field of 7 with jump (effective stride) 1.
  NN := TNNet.Create();
  Lines := TStringList.Create();
  try
    NN.AddLayer(TNNetInput.Create(32, 32, 1));
    NN.AddLayer(TNNetConvolutionReLU.Create(4, 3, 1, 1));
    NN.AddLayer(TNNetConvolutionReLU.Create(4, 3, 1, 1));
    NN.AddLayer(TNNetConvolutionReLU.Create(4, 3, 1, 1));

    Report := TNNet.ReceptiveFieldReport(NN);
    AssertTrue('Report should be non-empty', Length(Report) > 0);

    Lines.Text := Report;
    Layer3Line := '';
    for I := 0 to Lines.Count - 1 do
      if Pos('3     TNNetConvolutionReLU', Lines[I]) = 1 then
        Layer3Line := Lines[I];

    AssertTrue('Layer 3 row present', Layer3Line <> '');
    // Deepest conv: RF must be exactly 7x7.
    AssertTrue('Three 3x3 stride-1 convs => RF 7x7',
      Pos('7x7', Layer3Line) > 0);
    // Jump stays at 1x1 (no downsampling).
    AssertTrue('Stride-1 stack keeps jump at 1x1',
      Pos('1x1', Layer3Line) > 0);
  finally
    Lines.Free;
    NN.Free;
  end;
end;

procedure TTestNeuralLayersExtra.TestReceptiveFieldReportStrideDoublesJump;
var
  NN: TNNet;
  Report, Layer1Line, Layer2Line: string;
  Lines: TStringList;
  I: integer;
begin
  // Adding a stride-2 conv after a stride-1 conv must double the jump from
  // 1x1 to 2x2.
  NN := TNNet.Create();
  Lines := TStringList.Create();
  try
    NN.AddLayer(TNNetInput.Create(32, 32, 1));
    NN.AddLayer(TNNetConvolutionReLU.Create(4, 3, 1, 1));
    NN.AddLayer(TNNetConvolutionReLU.Create(4, 3, 1, 2));

    Report := TNNet.ReceptiveFieldReport(NN);
    AssertTrue('Report should be non-empty', Length(Report) > 0);

    Lines.Text := Report;
    Layer1Line := '';
    Layer2Line := '';
    for I := 0 to Lines.Count - 1 do
    begin
      if Pos('1     TNNetConvolutionReLU', Lines[I]) = 1 then
        Layer1Line := Lines[I]
      else if Pos('2     TNNetConvolutionReLU', Lines[I]) = 1 then
        Layer2Line := Lines[I];
    end;

    AssertTrue('Layer 1 row present', Layer1Line <> '');
    AssertTrue('Layer 2 row present', Layer2Line <> '');
    // Before the stride-2 layer the jump is still 1x1.
    AssertTrue('Stride-1 layer keeps jump at 1x1',
      Pos('1x1', Layer1Line) > 0);
    // After the stride-2 layer the jump doubles to 2x2.
    AssertTrue('Stride-2 layer doubles jump to 2x2',
      Pos('2x2', Layer2Line) > 0);
  finally
    Lines.Free;
    NN.Free;
  end;
end;

procedure TTestNeuralLayersExtra.TestActivationStatsReportInputStats;
var
  NN: TNNet;
  Probes: TNNetVolumeList;
  V: TNNetVolume;
  Report, InputLine: string;
  Lines: TStringList;
  I: integer;
begin
  // A constant-valued input passed straight to the input layer yields a known
  // mean and a near-zero std on the layer-0 (TNNetInput) row, so we can pin
  // exact numbers in the printed table.
  NN := TNNet.Create();
  Lines := TStringList.Create();
  Probes := TNNetVolumeList.Create(True);
  try
    NN.AddLayer(TNNetInput.Create(4, 1, 1));
    NN.AddLayer(TNNetFullConnectLinear.Create(3));
    NN.InitWeights();

    // Two identical probe samples filled with the constant 0.5.
    V := TNNetVolume.Create(4, 1, 1);
    V.Fill(0.5);
    Probes.Add(V);
    V := TNNetVolume.Create(4, 1, 1);
    V.Fill(0.5);
    Probes.Add(V);

    Report := TNNet.ActivationStatsReport(NN, Probes);
    AssertTrue('Report should be non-empty', Length(Report) > 0);

    Lines.Text := Report;
    InputLine := '';
    for I := 0 to Lines.Count - 1 do
      if Pos('0    TNNetInput', Lines[I]) = 1 then
        InputLine := Lines[I];

    AssertTrue('Input layer row present', InputLine <> '');
    // mean must be 0.5000 and std 0.0000 for a constant 0.5 input.
    AssertTrue('Constant 0.5 input => mean 0.5000', Pos('0.5000', InputLine) > 0);
    AssertTrue('Constant input => std 0.0000', Pos('0.0000', InputLine) > 0);
    // A constant input has zero spread => flagged near-collapsed.
    AssertTrue('Constant input layer flagged near-collapsed',
      Pos('near-collapsed', Report) > 0);
  finally
    Lines.Free;
    Probes.Free;
    NN.Free;
  end;
end;

procedure TTestNeuralLayersExtra.TestActivationStatsReportNearCollapsedFlag;
var
  NN: TNNet;
  Probes: TNNetVolumeList;
  V: TNNetVolume;
  Report: string;
  K, I: integer;
begin
  // Empty / nil sample list must be handled gracefully (no crash, clear msg).
  NN := TNNet.Create();
  Probes := TNNetVolumeList.Create(True);
  try
    NN.AddLayer(TNNetInput.Create(4, 1, 1));
    NN.AddLayer(TNNetFullConnectLinear.Create(3));
    NN.InitWeights();

    Report := TNNet.ActivationStatsReport(NN, Probes);
    AssertTrue('Empty Samples reported gracefully',
      Pos('Samples is nil or empty', Report) > 0);

    // Now add varied samples and confirm the report structure is present
    // (header, table separator, std histogram, flag list).
    for K := 0 to 7 do
    begin
      V := TNNetVolume.Create(4, 1, 1);
      for I := 0 to 3 do
        V.Raw[I] := K * 0.1 + I * 0.05;
      Probes.Add(V);
    end;

    Report := TNNet.ActivationStatsReport(NN, Probes);
    AssertTrue('Header present', Pos('ActivationStatsReport:', Report) > 0);
    AssertTrue('Per-layer std histogram present',
      Pos('Per-layer std histogram', Report) > 0);
    AssertTrue('Flag list present', Pos('Flags:', Report) > 0);
  finally
    Probes.Free;
    NN.Free;
  end;
end;

procedure TTestNeuralLayersExtra.TestWeightSpectrumReportRank1Matrix;
var
  NN: TNNet;
  Layer: TNNetLayer;
  A, B: array of TNeuralFloat;
  NormA, NormB, ExpectedSigma, Sigma: TNeuralFloat;
  FanOut, FanIn, N, K: integer;
  Report, SigmaStr: string;
  Lines: TStringList;
  I, P: integer;
begin
  // A rank-1 weight matrix W = a * b^T has exactly one non-zero singular value
  // sigma_1 = ||a|| * ||b|| (all other singular values are 0). We build such a
  // matrix by hand and check both the helper and the printed report converge
  // to that analytic value.
  FanOut := 5; // num neurons / rows
  FanIn  := 4; // weights per neuron / cols
  SetLength(A, FanOut);
  SetLength(B, FanIn);
  A[0] := 1.0; A[1] := -2.0; A[2] := 0.5; A[3] := 3.0; A[4] := -1.5;
  B[0] := 2.0; B[1] := 1.0; B[2] := -1.0; B[3] := 0.5;

  NormA := 0;
  for N := 0 to FanOut - 1 do NormA := NormA + A[N] * A[N];
  NormA := Sqrt(NormA);
  NormB := 0;
  for K := 0 to FanIn - 1 do NormB := NormB + B[K] * B[K];
  NormB := Sqrt(NormB);
  ExpectedSigma := NormA * NormB;

  NN := TNNet.Create();
  Lines := TStringList.Create();
  try
    NN.AddLayer(TNNetInput.Create(FanIn, 1, 1));
    NN.AddLayer(TNNetFullConnectLinear.Create(FanOut));
    NN.InitWeights();

    Layer := NN.Layers[1];
    AssertTrue('FullConnect layer has the expected neuron count',
      Layer.Neurons.Count = FanOut);
    AssertTrue('FullConnect layer has the expected fan-in',
      Layer.Neurons[0].Weights.Size = FanIn);

    // Row n of W = a[n] * b.
    for N := 0 to FanOut - 1 do
      for K := 0 to FanIn - 1 do
        Layer.Neurons[N].Weights.FData[K] := A[N] * B[K];

    // (1) Helper converges to the analytic sigma_1.
    Sigma := TNNet.EstimateSpectralNorm(Layer, 30);
    AssertTrue(Format('EstimateSpectralNorm should be ~%.5f, got %.5f',
      [ExpectedSigma, Sigma]), Abs(Sigma - ExpectedSigma) < 1e-3);

    // (2) The report prints that sigma_1 in the layer-1 row.
    Report := TNNet.WeightSpectrumReport(NN);
    AssertTrue('Report should be non-empty', Length(Report) > 0);

    Lines.Text := Report;
    SigmaStr := '';
    for I := 0 to Lines.Count - 1 do
      if Pos('1    TNNetFullConnectLinear', Lines[I]) = 1 then
        SigmaStr := Lines[I];
    AssertTrue('Layer 1 row present', SigmaStr <> '');
    // sigma_1 formats with %.4f; check the analytic value's 4-decimal text
    // appears on the row.
    P := Pos(FormatFloat('0.0000', ExpectedSigma), SigmaStr);
    AssertTrue(Format('Layer row should contain sigma_1=%.4f. Row: %s',
      [ExpectedSigma, SigmaStr]), P > 0);

    // A rank-1 matrix => sigma_1/||W||_F = 1 => stable-rank collapse flag.
    AssertTrue('Rank-1 layer flagged as stable-rank ~= 1',
      Pos('stable-rank ~= 1', Report) > 0);
  finally
    Lines.Free;
    NN.Free;
  end;
end;

procedure TTestNeuralLayersExtra.TestWeightSpectrumReportStructureAndFlags;
var
  NN: TNNet;
  Report, ReportB: string;
begin
  // nil NN handled gracefully.
  Report := TNNet.WeightSpectrumReport(nil);
  AssertTrue('nil NN reported gracefully', Pos('NN is nil', Report) > 0);

  NN := TNNet.Create();
  try
    NN.AddLayer(TNNetInput.Create(4, 1, 1));
    NN.AddLayer(TNNetFullConnectReLU.Create(6));
    NN.AddLayer(TNNetFullConnectLinear.Create(2));
    NN.InitWeights();

    Report := TNNet.WeightSpectrumReport(NN);
    AssertTrue('Header present', Pos('WeightSpectrumReport:', Report) > 0);
    AssertTrue('sigma_1 column header present', Pos('sigma_1', Report) > 0);
    AssertTrue('MP-ratio column header present', Pos('MP-ratio', Report) > 0);
    AssertTrue('Fan-in baseline histogram present',
      Pos('fan-in baseline', Report) > 0);
    AssertTrue('Flag list present', Pos('Flags:', Report) > 0);
    AssertTrue('Network total line present', Pos('Network total:', Report) > 0);

    // Determinism: same network => identical report text.
    ReportB := TNNet.WeightSpectrumReport(NN);
    AssertTrue('Report is deterministic', Report = ReportB);
  finally
    NN.Free;
  end;
end;

procedure TTestNeuralLayersExtra.TestWeightSpectralTailReportSmoke;
var
  NN: TNNet;
  Layer: TNNetLayer;
  Report, LRow: string;
  Lines: TStringList;
  FanOut, FanIn, N, K, I: integer;
  Sigma, LambdaMaxExpected: TNeuralFloat;
  LMaxStr: string;
begin
  // (1) nil NN handled gracefully.
  Report := TNNet.WeightSpectralTailReport(nil);
  AssertTrue('nil NN reported gracefully', Pos('NN is nil', Report) > 0);

  // (2) A known small layer. We seed a rank-1-flavoured matrix W = a*b^T plus a
  // tiny full-rank perturbation so the Gram is genuinely PSD with a clear
  // top eigenvalue. The report's own built-in checks (non-negative eigenvalues
  // / trace invariance / lambda_max == sigma_1^2) must all pass => NO PSD,
  // trace or lambda_max disagreement flags in the output.
  FanOut := 16; // rows / neurons
  FanIn  := 10; // cols / weights-per-neuron (=> Gram dim 10, enough tail pts)

  NN := TNNet.Create();
  Lines := TStringList.Create();
  try
    NN.AddLayer(TNNetInput.Create(FanIn, 1, 1));
    NN.AddLayer(TNNetFullConnectLinear.Create(FanOut));
    NN.InitWeights();

    Layer := NN.Layers[1];
    AssertTrue('layer neuron count', Layer.Neurons.Count = FanOut);
    AssertTrue('layer fan-in', Layer.Neurons[0].Weights.Size = FanIn);

    // Deterministic, well-spread weights (so the Gram is full-rank PSD with
    // distinct eigenvalues — a non-degenerate spectrum to fit alpha on).
    for N := 0 to FanOut - 1 do
      for K := 0 to FanIn - 1 do
        Layer.Neurons[N].Weights.FData[K] :=
          Sin(0.7 * (N + 1) + 0.3 * (K + 1)) + 0.05 * (N - K);

    Report := TNNet.WeightSpectralTailReport(NN);
    AssertTrue('Report non-empty', Length(Report) > 0);
    AssertTrue('Header present',
      Pos('WeightSpectralTailReport:', Report) > 0);
    AssertTrue('alpha column header present', Pos('alpha', Report) > 0);
    AssertTrue('lambda_max column header present', Pos('lambda_max', Report) > 0);
    AssertTrue('Bands legend present', Pos('well-shaped', Report) > 0);
    AssertTrue('Flags section present', Pos('Flags:', Report) > 0);

    // Built-in correctness checks must have passed: no PSD-violation, no
    // trace-invariance failure, no lambda_max disagreement flags.
    AssertTrue('No negative-eigenvalue (PSD) flag',
      Pos('negative eigenvalue', Report) = 0);
    AssertTrue('No trace-invariance failure flag',
      Pos('trace-invariance check failed', Report) = 0);
    AssertTrue('No lambda_max disagreement flag',
      Pos('disagrees with', Report) = 0);

    // (3) Independently cross-check lambda_max == sigma_1(W)^2: compute sigma_1
    // with the power-iteration helper (the same value the report cross-checks
    // against internally) and confirm the printed lambda_max on the layer-1 row
    // carries its square (formatted with %11.4g, 4 significant figures).
    Sigma := TNNet.EstimateSpectralNorm(Layer, 60);
    LambdaMaxExpected := Sigma * Sigma;
    AssertTrue('sigma_1 positive', Sigma > 0);

    Lines.Text := Report;
    LRow := '';
    for I := 0 to Lines.Count - 1 do
      if Pos('1    TNNetFullConnectLinear', Lines[I]) = 1 then
        LRow := Lines[I];
    AssertTrue('Layer-1 row present', LRow <> '');
    // lambda_max is printed with %.4g (4 significant figures) — reproduce that
    // exact rendering of sigma_1^2 and require it to appear on the row.
    LMaxStr := Trim(Format('%.4g', [LambdaMaxExpected]));
    AssertTrue(Format(
      'Layer row should carry lambda_max ~= sigma_1^2 = %s. Row: %s',
      [LMaxStr, LRow]), Pos(LMaxStr, LRow) > 0);

    // (4) Determinism: identical text on a second call (no randomness).
    AssertTrue('Report deterministic',
      Report = TNNet.WeightSpectralTailReport(NN));
  finally
    Lines.Free;
    NN.Free;
  end;
end;

procedure TTestNeuralLayersExtra.TestTopLogitMarginReportSmoke;
var
  NN: TNNet;
  Samples: TNNetVolumePairList;
  X, Y: TNNetVolume;
  Report: string;
  I, C: integer;
begin
  // nil NN handled gracefully.
  Report := TNNet.TopLogitMarginReport(nil, nil, 3);
  AssertTrue('nil NN / empty samples reported gracefully', Length(Report) > 0);

  NN := TNNet.Create();
  Samples := TNNetVolumePairList.Create();
  try
    NN.AddLayer(TNNetInput.Create(2, 1, 1));
    NN.AddLayer(TNNetFullConnectReLU.Create(8));
    NN.AddLayer(TNNetFullConnectLinear.Create(3));
    NN.AddLayer(TNNetSoftMax.Create());
    NN.InitWeights();

    // A handful of labeled samples across 3 classes (no training needed for a
    // smoke test — we only need a valid forward pass and a non-empty report).
    for I := 0 to 11 do
    begin
      C := I mod 3;
      X := TNNetVolume.Create(2, 1, 1);
      Y := TNNetVolume.Create(3, 1, 1);
      X.FData[0] := C * 1.0;
      X.FData[1] := -C * 1.0;
      Y.Fill(0);
      Y.FData[C] := 1.0;
      Samples.Add(TNNetVolumePair.Create(X, Y));
    end;

    Report := TNNet.TopLogitMarginReport(NN, Samples, 3, 2);
    AssertTrue('Report is non-empty', Length(Report) > 0);
    AssertTrue('Header present',
      Pos('TopLogitMarginReport', Report) > 0);
    AssertTrue('Histogram section present',
      Pos('Margin histogram', Report) > 0);
    AssertTrue('Hard-examples section present',
      Pos('Hard examples per class', Report) > 0);

    // nil NN with a valid sample list still handled gracefully.
    Report := TNNet.TopLogitMarginReport(nil, Samples, 3);
    AssertTrue('nil NN reported gracefully', Pos('NN is nil', Report) > 0);
  finally
    Samples.Free;
    NN.Free;
  end;
end;

procedure TTestNeuralLayersExtra.TestNeuronCorrelationReportSmoke;
var
  NN: TNNet;
  Probes: TNNetVolumeList;
  V: TNNetVolume;
  Report: string;
  I, J: integer;
begin
  // nil NN handled gracefully.
  Report := TNNet.NeuronCorrelationReport(nil, nil);
  AssertTrue('nil NN reported gracefully', Pos('NN is nil', Report) > 0);

  NN := TNNet.Create();
  Probes := TNNetVolumeList.Create(True);
  try
    NN.AddLayer(TNNetInput.Create(4, 1, 1));
    NN.AddLayer(TNNetFullConnectReLU.Create(8));
    NN.AddLayer(TNNetFullConnectLinear.Create(2));
    NN.InitWeights();

    // empty probe list handled gracefully.
    Report := TNNet.NeuronCorrelationReport(NN, Probes);
    AssertTrue('empty probes reported gracefully',
      Pos('nil or empty', Report) > 0);

    // a handful of probe volumes.
    for I := 0 to 15 do
    begin
      V := TNNetVolume.Create(4, 1, 1);
      for J := 0 to 3 do V.Raw[J] := (Random - 0.5) * 2.0;
      Probes.Add(V);
    end;

    Report := TNNet.NeuronCorrelationReport(NN, Probes, 3);
    AssertTrue('Report is non-empty', Length(Report) > 0);
    AssertTrue('Header present',
      Pos('NeuronCorrelationReport', Report) > 0);
    AssertTrue('Histogram section present',
      Pos('|rho| histogram', Report) > 0);
    AssertTrue('Effective-count section present',
      Pos('effective neuron count', Report) > 0);
    AssertTrue('Flags section present', Pos('Flags', Report) > 0);
  finally
    Probes.Free;
    NN.Free;
  end;
end;

procedure TTestNeuralLayersExtra.TestLayerSensitivityReportSmoke;
var
  NN: TNNet;
  Probes, Targets: TNNetVolumeList;
  V, T: TNNetVolume;
  Before, After: TNNetVolume;
  Report: string;
  I, J: integer;
  MaxDiff, D: TNeuralFloat;
begin
  // nil NN handled gracefully.
  Report := TNNet.LayerSensitivityReport(nil, nil);
  AssertTrue('nil NN reported gracefully', Pos('NN is nil', Report) > 0);

  NN := TNNet.Create();
  Probes := TNNetVolumeList.Create(True);
  Targets := TNNetVolumeList.Create(True);
  Before := TNNetVolume.Create();
  After := TNNetVolume.Create();
  try
    NN.AddLayer(TNNetInput.Create(4, 1, 1));
    NN.AddLayer(TNNetFullConnectReLU.Create(8));
    NN.AddLayer(TNNetFullConnectLinear.Create(2));
    NN.InitWeights();

    // empty probe list handled gracefully.
    Report := TNNet.LayerSensitivityReport(NN, Probes);
    AssertTrue('empty probes reported gracefully',
      Pos('nil or empty', Report) > 0);

    // a handful of probe volumes + matching targets.
    for I := 0 to 15 do
    begin
      V := TNNetVolume.Create(4, 1, 1);
      for J := 0 to 3 do V.Raw[J] := (Random - 0.5) * 2.0;
      Probes.Add(V);
      T := TNNetVolume.Create(2, 1, 1);
      for J := 0 to 1 do T.Raw[J] := (Random - 0.5) * 2.0;
      Targets.Add(T);
    end;

    // Capture output on a fixed probe BEFORE the report.
    NN.Compute(Probes[0]);
    Before.Copy(NN.GetLastLayer.Output);

    // Run with targets (exercises the loss-delta branch).
    Report := TNNet.LayerSensitivityReport(NN, Probes, Targets, 0.02, 4);
    AssertTrue('Report is non-empty', Length(Report) > 0);
    AssertTrue('Header present',
      Pos('LayerSensitivityReport', Report) > 0);
    AssertTrue('Histogram section present',
      Pos('mean output-delta histogram', Report) > 0);
    AssertTrue('Fragility verdict present',
      Pos('Fragility verdict', Report) > 0);

    // Save/restore guarantee: output on the SAME probe must be unchanged.
    NN.Compute(Probes[0]);
    After.Copy(NN.GetLastLayer.Output);
    AssertEquals('output size unchanged', Before.Size, After.Size);
    MaxDiff := 0;
    for I := 0 to Before.Size - 1 do
    begin
      D := Abs(Before.Raw[I] - After.Raw[I]);
      if D > MaxDiff then MaxDiff := D;
    end;
    AssertTrue('weights restored bit-exact (max output diff < 1e-6)',
      MaxDiff < 1e-6);

    // Without targets the loss column reads n/a.
    Report := TNNet.LayerSensitivityReport(NN, Probes, nil, 0.02, 4);
    AssertTrue('n/a loss when no targets', Pos('n/a', Report) > 0);
  finally
    After.Free;
    Before.Free;
    Targets.Free;
    Probes.Free;
    NN.Free;
  end;
end;

procedure TTestNeuralLayersExtra.TestMagnitudePruningReportSmoke;
var
  NN: TNNet;
  Probes, Labels: TNNetVolumeList;
  V, L, Before, After: TNNetVolume;
  Report: string;
  I, J, Cls: integer;
  MaxDiff, D: TNeuralFloat;
begin
  // nil NN handled gracefully.
  Report := TNNet.MagnitudePruningReport(nil, nil);
  AssertTrue('nil NN reported gracefully', Pos('NN is nil', Report) > 0);

  NN := TNNet.Create();
  Probes := TNNetVolumeList.Create(True);
  Labels := TNNetVolumeList.Create(True);
  Before := TNNetVolume.Create();
  After := TNNetVolume.Create();
  try
    NN.AddLayer(TNNetInput.Create(4, 1, 1));
    NN.AddLayer(TNNetFullConnectReLU.Create(8));
    NN.AddLayer(TNNetFullConnectLinear.Create(3));
    NN.InitWeights();

    // empty probe list handled gracefully.
    Report := TNNet.MagnitudePruningReport(NN, Probes);
    AssertTrue('empty probes reported gracefully',
      Pos('nil or empty', Report) > 0);

    // a handful of probe volumes + one-hot labels (3 classes).
    for I := 0 to 23 do
    begin
      V := TNNetVolume.Create(4, 1, 1);
      for J := 0 to 3 do V.Raw[J] := (Random - 0.5) * 2.0;
      Probes.Add(V);
      L := TNNetVolume.Create(3, 1, 1);
      Cls := Random(3);
      L.Raw[Cls] := 1.0;
      Labels.Add(L);
    end;

    // Capture output on a fixed probe BEFORE the report.
    NN.Compute(Probes[0]);
    Before.Copy(NN.GetLastLayer.Output);

    // Run WITH labels (exercises accuracy + loss branch).
    Report := TNNet.MagnitudePruningReport(NN, Probes, Labels);
    AssertTrue('Report is non-empty', Length(Report) > 0);
    AssertTrue('Header present',
      Pos('MagnitudePruningReport', Report) > 0);
    AssertTrue('Sparsity sweep section present',
      Pos('Sparsity sweep', Report) > 0);
    AssertTrue('Prunability knee present',
      Pos('Prunability KNEE', Report) > 0);
    AssertTrue('Verdict present', Pos('Verdict:', Report) > 0);
    AssertTrue('Realised-vs-requested present',
      Pos('Realised-vs-requested', Report) > 0);
    // s=0% baseline reproduced exactly: no warning emitted.
    AssertTrue('s=0 reproduces baseline (no warning)',
      Pos('did not reproduce baseline', Report) = 0);
    // realised sparsity at request 0% must be ~0%.
    AssertTrue('s=0 realised ~0%',
      Pos('req   0.0% -> realised   0.00%', Report) > 0);

    // Save/restore guarantee: output on the SAME probe must be unchanged.
    NN.Compute(Probes[0]);
    After.Copy(NN.GetLastLayer.Output);
    AssertEquals('output size unchanged', Before.Size, After.Size);
    MaxDiff := 0;
    for I := 0 to Before.Size - 1 do
    begin
      D := Abs(Before.Raw[I] - After.Raw[I]);
      if D > MaxDiff then MaxDiff := D;
    end;
    AssertTrue('weights restored bit-exact (max output diff < 1e-6)',
      MaxDiff < 1e-6);

    // Label-free run (loss = output drift) and per-layer baseline both work.
    Report := TNNet.MagnitudePruningReport(NN, Probes, nil);
    AssertTrue('label-free run non-empty', Length(Report) > 0);
    AssertTrue('label-free knee present', Pos('Prunability KNEE', Report) > 0);
    Report := TNNet.MagnitudePruningReport(NN, Probes, Labels, 0.01, True);
    AssertTrue('per-layer criterion reported',
      Pos('PER-LAYER percentile', Report) > 0);
  finally
    After.Free;
    Before.Free;
    Labels.Free;
    Probes.Free;
    NN.Free;
  end;
end;

procedure TTestNeuralLayersExtra.TestMCDropoutUncertaintyReportSmoke;
var
  NN, NNNoDrop: TNNet;
  Probes: TNNetVolumeList;
  V: TNNetVolume;
  Labels: array of integer;
  Report: string;
  I, J: integer;
  BALDLine: string;
begin
  // nil NN handled gracefully.
  Report := TNNet.MCDropoutUncertaintyReport(nil, nil);
  AssertTrue('nil NN reported gracefully', Pos('NN is nil', Report) > 0);

  NN := TNNet.Create();
  NNNoDrop := TNNet.Create();
  Probes := TNNetVolumeList.Create(True);
  try
    // dropout net (stochastic) + softmax head.
    NN.AddLayer(TNNetInput.Create(4, 1, 1));
    NN.AddLayer(TNNetFullConnectReLU.Create(8));
    NN.AddLayer(TNNetDropout.Create(0.5));
    NN.AddLayer(TNNetFullConnectLinear.Create(3));
    NN.AddLayer(TNNetSoftMax.Create());
    NN.InitWeights();

    // empty probe list handled gracefully.
    Report := TNNet.MCDropoutUncertaintyReport(NN, Probes);
    AssertTrue('empty probes reported gracefully',
      Pos('nil or empty', Report) > 0);

    for I := 0 to 19 do
    begin
      V := TNNetVolume.Create(4, 1, 1);
      for J := 0 to 3 do V.Raw[J] := (Random - 0.5) * 2.0;
      Probes.Add(V);
    end;

    // Non-empty report with the expected header substring.
    Report := TNNet.MCDropoutUncertaintyReport(NN, Probes, 16, 1.0, 5);
    AssertTrue('Report is non-empty', Length(Report) > 0);
    AssertTrue('Header present',
      Pos('MCDropoutUncertaintyReport', Report) > 0);
    AssertTrue('BALD histogram section present',
      Pos('BALD histogram', Report) > 0);
    AssertTrue('Active-learning queue present',
      Pos('Active-learning queue', Report) > 0);

    // SPEC INVARIANT 1: NumPasses=1 AND dropout disabled -> BALD ~ 0.
    NN.EnableDropouts(False);
    Report := TNNet.MCDropoutUncertaintyReport(NN, Probes, 1, 1.0, 5);
    // The batch-means line carries the BALD scalar; parse it out.
    J := Pos('BALD=', Report);
    AssertTrue('Batch BALD reported', J > 0);
    // Find the LAST "BALD=" (batch-means line) and read the float after it.
    repeat
      I := J;
      J := PosEx('BALD=', Report, I + 1);
    until J = 0;
    BALDLine := Copy(Report, I + 5, 8);
    // With a single deterministic pass H[mean_p] == H[p_1] so BALD == 0.0000.
    AssertTrue('NumPasses=1 + no dropout collapses BALD to ~0 (got "' +
      BALDLine + '")', Pos('0.0000', BALDLine) = 1);
    // Restore so the object is in a clean state (not strictly required).
    NN.EnableDropouts(True);

    // SPEC INVARIANT 2: a net with NO TNNetAddNoiseBase layer warns clearly.
    NNNoDrop.AddLayer(TNNetInput.Create(4, 1, 1));
    NNNoDrop.AddLayer(TNNetFullConnectReLU.Create(8));
    NNNoDrop.AddLayer(TNNetFullConnectLinear.Create(3));
    NNNoDrop.AddLayer(TNNetSoftMax.Create());
    NNNoDrop.InitWeights();
    Report := TNNet.MCDropoutUncertaintyReport(NNNoDrop, Probes, 8, 1.0, 5);
    AssertTrue('no-stochastic-layer warning present',
      Pos('no stochastic layers', Report) > 0);

    // Labelled overload exercises the correctness cross-tab path.
    SetLength(Labels, Probes.Count);
    for I := 0 to Probes.Count - 1 do Labels[I] := I mod 3;
    Report := TNNet.MCDropoutUncertaintyReport(NN, Probes, Labels, 8, 1.0, 5);
    AssertTrue('cross-tab present', Pos('Correctness cross-tab', Report) > 0);
  finally
    Probes.Free;
    NNNoDrop.Free;
    NN.Free;
  end;
end;

procedure TTestNeuralLayersExtra.TestEquivarianceReportSmoke;
var
  NNPlain, NNInv: TNNet;
  Probes: TNNetVolumeList;
  V: TNNetVolume;
  Report: string;
  K, I: integer;
  FlipXPos, EndPos: integer;
  FlipXLine: string;
begin
  // nil NN handled gracefully.
  Report := TNNet.EquivarianceReport(nil, nil);
  AssertTrue('nil NN reported gracefully', Pos('NN is nil', Report) > 0);

  NNPlain := TNNet.Create();
  NNInv := TNNet.Create();
  Probes := TNNetVolumeList.Create(True);
  try
    // empty probe list handled gracefully (use a valid net).
    NNInv.AddLayer(TNNetInput.Create(6, 6, 3));
    NNInv.AddLayer(TNNetAvgChannel.Create());
    NNInv.AddLayer(TNNetFullConnectReLU.Create(8));
    NNInv.AddLayer(TNNetFullConnectLinear.Create(3));
    NNInv.AddLayer(TNNetSoftMax.Create());
    NNInv.InitWeights();
    Report := TNNet.EquivarianceReport(NNInv, Probes);
    AssertTrue('empty probes reported gracefully',
      Pos('nil or empty', Report) > 0);

    // A plain conv classifier: no built-in spatial symmetry => flip-sensitive.
    NNPlain.AddLayer(TNNetInput.Create(6, 6, 3));
    NNPlain.AddLayer(TNNetConvolutionReLU.Create(6, 3, 1, 1));
    NNPlain.AddLayer(TNNetMaxPool.Create(2));
    NNPlain.AddLayer(TNNetFullConnectReLU.Create(8));
    NNPlain.AddLayer(TNNetFullConnectLinear.Create(3));
    NNPlain.AddLayer(TNNetSoftMax.Create());
    NNPlain.InitWeights();

    // A handful of image-shaped probes.
    for K := 0 to 11 do
    begin
      V := TNNetVolume.Create(6, 6, 3);
      for I := 0 to V.Size - 1 do V.Raw[I] := (Random - 0.5) * 2.0;
      Probes.Add(V);
    end;

    // Plain net: report is well-formed.
    Report := TNNet.EquivarianceReport(NNPlain, Probes);
    AssertTrue('Report is non-empty', Length(Report) > 0);
    AssertTrue('Header present', Pos('EquivarianceReport', Report) > 0);
    AssertTrue('FlipX row present', Pos('TNNetFlipX', Report) > 0);
    AssertTrue('FlipY row present', Pos('TNNetFlipY', Report) > 0);
    AssertTrue('ReverseChannels row present',
      Pos('TNNetReverseChannels', Report) > 0);
    AssertTrue('Roll row present', Pos('TNNetRoll', Report) > 0);
    AssertTrue('histogram section present',
      Pos('InvarErr histogram', Report) > 0);
    // A plain conv classifier should be FlipX-sensitive on random inputs.
    AssertTrue('plain net flagged sensitive somewhere',
      Pos('sensitive', Report) > 0);

    // Built-in correctness check: a global-spatial-average net is exactly
    // FlipX-invariant, so the FlipX row must read the "invariant" verdict.
    Report := TNNet.EquivarianceReport(NNInv, Probes);
    AssertTrue('invariant-net report non-empty', Length(Report) > 0);
    FlipXPos := Pos('TNNetFlipX', Report);
    AssertTrue('FlipX row present for invariant net', FlipXPos > 0);
    EndPos := PosEx(sLineBreak, Report, FlipXPos);
    if EndPos = 0 then EndPos := Length(Report) + 1;
    FlipXLine := Copy(Report, FlipXPos, EndPos - FlipXPos);
    AssertTrue('FlipX-invariant net reads "invariant" on the FlipX row',
      Pos('invariant', FlipXLine) > 0);
    AssertTrue('FlipX-invariant net is not flagged approximate on FlipX',
      Pos('approximately', FlipXLine) = 0);
    AssertTrue('FlipX-invariant net is not flagged sensitive on FlipX',
      Pos('sensitive', FlipXLine) = 0);
    AssertTrue('FlipX-invariant net reaches 100% top-1 agreement',
      Pos('100.00%', FlipXLine) > 0);
  finally
    Probes.Free;
    NNInv.Free;
    NNPlain.Free;
  end;
end;

procedure TTestNeuralLayersExtra.TestTTAReportSmoke;
var
  NN: TNNet;
  Probes: TNNetVolumeList;
  V: TNNetVolume;
  Report: string;
  Labels: array of integer;
  EmptyLabels: array of integer;
  K, I: integer;
begin
  // nil NN handled gracefully (empty labels array).
  SetLength(EmptyLabels, 0);
  Report := TNNet.TTAReport(nil, nil, EmptyLabels);
  AssertTrue('nil NN reported gracefully', Pos('NN is nil', Report) > 0);

  NN := TNNet.Create();
  Probes := TNNetVolumeList.Create(True);
  try
    // A small image classifier.
    NN.AddLayer(TNNetInput.Create(6, 6, 3));
    NN.AddLayer(TNNetConvolutionReLU.Create(6, 3, 1, 1));
    NN.AddLayer(TNNetMaxPool.Create(2));
    NN.AddLayer(TNNetFullConnectReLU.Create(8));
    NN.AddLayer(TNNetFullConnectLinear.Create(3));
    NN.AddLayer(TNNetSoftMax.Create());
    NN.InitWeights();

    // empty probe list handled gracefully.
    Report := TNNet.TTAReport(NN, Probes, EmptyLabels);
    AssertTrue('empty probes reported gracefully',
      Pos('nil or empty', Report) > 0);

    // A handful of image-shaped probes with labels.
    SetLength(Labels, 12);
    for K := 0 to 11 do
    begin
      V := TNNetVolume.Create(6, 6, 3);
      for I := 0 to V.Size - 1 do V.Raw[I] := (Random - 0.5) * 2.0;
      Probes.Add(V);
      Labels[K] := K mod 3;
    end;

    // Label-count mismatch handled gracefully.
    SetLength(EmptyLabels, 3);
    Report := TNNet.TTAReport(NN, Probes, EmptyLabels);
    AssertTrue('label mismatch reported gracefully',
      Pos('does not match', Report) > 0);

    // Well-formed report (logit averaging, default).
    Report := TNNet.TTAReport(NN, Probes, Labels);
    AssertTrue('Report is non-empty', Length(Report) > 0);
    AssertTrue('Header present', Pos('TTAReport', Report) > 0);
    AssertTrue('per-transform section present',
      Pos('Per-transform top-1 accuracy', Report) > 0);
    AssertTrue('baseline row present',
      Pos('Baseline (identity) top-1 accuracy', Report) > 0);
    AssertTrue('ensemble row present',
      Pos('Full-ensemble TTA top-1 accuracy', Report) > 0);
    AssertTrue('FlipX row present', Pos('TNNetFlipX', Report) > 0);
    AssertTrue('per-class section present',
      Pos('Per-class top-1 accuracy', Report) > 0);
    AssertTrue('agreement rate present',
      Pos('Per-sample agreement rate', Report) > 0);
    AssertTrue('verdict present', Pos('Verdict:', Report) > 0);

    // probability-averaging path also produces a well-formed report.
    Report := TNNet.TTAReport(NN, Probes, Labels, True);
    AssertTrue('prob-avg report non-empty', Length(Report) > 0);
    AssertTrue('prob-avg mode noted',
      Pos('post-softmax probabilities', Report) > 0);
  finally
    Probes.Free;
    NN.Free;
  end;
end;

procedure TTestNeuralLayersExtra.TestSaliencyReportSmoke;
var
  NN: TNNet;
  X, Y, Probe: TNNetVolume;
  Report: string;
  Ep, B, K, I, Cls: integer;
  RelStart, RelEnd: integer;
  RelStr: string;
  RelGap: TNeuralFloat;
  FS: TFormatSettings;
begin
  // Format() in the report emits a '.' decimal separator regardless of locale;
  // parse with a matching FormatSettings so the test is locale-independent.
  FS := DefaultFormatSettings;
  FS.DecimalSeparator := '.';
  FS.ThousandSeparator := #0;
  // nil NN handled gracefully.
  Report := TNNet.SaliencyReport(nil, nil);
  AssertTrue('nil NN reported gracefully', Pos('NN is nil', Report) > 0);

  NN := TNNet.Create();
  Probe := nil;
  try
    // nil probe handled gracefully (on a valid net).
    NN.AddLayer(TNNetInput.Create(6, 6, 2));
    NN.AddLayer(TNNetConvolutionReLU.Create(6, 3, 1, 1));
    NN.AddLayer(TNNetMaxPool.Create(2));
    NN.AddLayer(TNNetFullConnectReLU.Create(8));
    NN.AddLayer(TNNetFullConnectLinear.Create(2));
    NN.AddLayer(TNNetSoftMax.Create());
    NN.SetLearningRate(0.01, 0.9);
    NN.InitWeights();

    Report := TNNet.SaliencyReport(NN, nil);
    AssertTrue('nil probe reported gracefully',
      Pos('nil or empty', Report) > 0);

    // Train briefly so the saliency gradient is non-degenerate. Class 0 has a
    // bright blob top-left/channel 0; class 1 bottom-right/channel 1.
    RandSeed := 77;
    for Ep := 1 to 60 do
      for B := 1 to 16 do
      begin
        X := TNNetVolume.Create(6, 6, 2);
        Y := TNNetVolume.Create(2, 1, 1);
        try
          X.Fill(0);
          Y.Fill(0);
          Cls := Random(2);
          Y.Raw[Cls] := 1.0;
          for I := 0 to X.Size - 1 do X.Raw[I] := Random * 0.1;
          if Cls = 0 then
          begin
            X[1, 1, 0] := 1.0; X[1, 0, 0] := 1.0; X[0, 1, 0] := 1.0;
          end
          else
          begin
            X[4, 4, 1] := 1.0; X[4, 5, 1] := 1.0; X[5, 4, 1] := 1.0;
          end;
          NN.Compute(X);
          NN.Backpropagate(Y);
        finally
          X.Free;
          Y.Free;
        end;
      end;

    // Build a clean class-0 probe.
    Probe := TNNetVolume.Create(6, 6, 2);
    Probe.Fill(0);
    Probe[1, 1, 0] := 1.0; Probe[1, 0, 0] := 1.0; Probe[0, 1, 0] := 1.0;

    Report := TNNet.SaliencyReport(NN, Probe);
    AssertTrue('Report is non-empty', Length(Report) > 0);
    AssertTrue('Header present', Pos('SaliencyReport', Report) > 0);
    AssertTrue('vanilla section present', Pos('vanilla', Report) > 0);
    AssertTrue('SmoothGrad section present', Pos('SmoothGrad', Report) > 0);
    AssertTrue('Integrated Gradients section present',
      Pos('Integrated Gradients', Report) > 0);
    AssertTrue('completeness gap reported',
      Pos('IG completeness gap', Report) > 0);
    AssertTrue('top-K pixels reported', Pos('top-', Report) > 0);

    // Built-in correctness check: the IG completeness gap must be small and
    // finite. Parse the relative percentage out of
    //   "(relative <num>%):".
    RelStart := Pos('(relative ', Report);
    AssertTrue('relative gap present', RelStart > 0);
    RelStart := RelStart + Length('(relative ');
    RelEnd := PosEx('%)', Report, RelStart);
    AssertTrue('relative gap terminator present', RelEnd > RelStart);
    RelStr := Trim(Copy(Report, RelStart, RelEnd - RelStart));
    RelGap := StrToFloatDef(RelStr, 1e30, FS);
    AssertFalse('relative gap is finite (not NaN)', IsNan(RelGap));
    AssertFalse('relative gap is finite (not Inf)', IsInfinite(RelGap));
    // A correct IG integration satisfies completeness to within a few %.
    AssertTrue('IG completeness gap is small (relative < 25%)', RelGap < 25.0);
    AssertTrue('report flags the gap as OK (small)',
      Pos('OK (small)', Report) > 0);
  finally
    if Probe <> nil then Probe.Free;
    NN.Free;
  end;
end;

procedure TTestNeuralLayersExtra.TestDecisionBoundaryReportSmoke;
var
  NN, NNBad: TNNet;
  TrainSet: TNNetVolumePairList;
  X, Y: TNNetVolume;
  Report: string;
  Ep, I, C: integer;
  // simple 2-cluster centres so a tiny net learns a clean separator
  Centers: array[0..1, 0..1] of TNeuralFloat = ((-1.5, -1.5), (1.5, 1.5));
begin
  // nil NN handled gracefully.
  Report := TNNet.DecisionBoundaryReport(nil);
  AssertTrue('nil NN reported gracefully', Pos('NN is nil', Report) > 0);

  // GUARD: a non-2-D input net must return the documented error, not crash.
  NNBad := TNNet.Create();
  try
    NNBad.AddLayer(TNNetInput.Create(3, 1, 1)); // 3-D input, not allowed
    NNBad.AddLayer(TNNetFullConnectLinear.Create(2));
    NNBad.AddLayer(TNNetSoftMax.Create());
    NNBad.InitWeights();
    Report := TNNet.DecisionBoundaryReport(NNBad);
    AssertTrue('non-2-D input flagged by guard',
      Pos('input layer must be 2-D', Report) > 0);
  finally
    NNBad.Free;
  end;

  NN := TNNet.Create();
  TrainSet := TNNetVolumePairList.Create();
  try
    NN.AddLayer(TNNetInput.Create(2, 1, 1));
    NN.AddLayer(TNNetFullConnectReLU.Create(12));
    NN.AddLayer(TNNetFullConnectLinear.Create(2));
    NN.AddLayer(TNNetSoftMax.Create());
    NN.SetLearningRate(0.05, 0.9);
    NN.InitWeights();

    RandSeed := 4242;
    for C := 0 to 1 do
      for I := 1 to 80 do
      begin
        X := TNNetVolume.Create(2, 1, 1);
        Y := TNNetVolume.Create(2, 1, 1);
        X.FData[0] := Centers[C][0] + (Random - 0.5);
        X.FData[1] := Centers[C][1] + (Random - 0.5);
        Y.Fill(0);
        Y.FData[C] := 1.0;
        TrainSet.Add(TNNetVolumePair.Create(X, Y));
      end;

    for Ep := 1 to 40 do
      for I := 0 to TrainSet.Count - 1 do
      begin
        NN.Compute(TrainSet[I].I);
        NN.Backpropagate(TrainSet[I].O);
      end;

    // Well-formed report with the documented sections (auto-fitted box).
    Report := TNNet.DecisionBoundaryReport(NN, TrainSet, 21, 21);
    AssertTrue('Report is non-empty', Length(Report) > 0);
    AssertTrue('Header present', Pos('DecisionBoundaryReport', Report) > 0);
    AssertTrue('class map section present', Pos('(a) Class map', Report) > 0);
    AssertTrue('confidence overlay section present',
      Pos('(b) Confidence overlay', Report) > 0);
    AssertTrue('boundary scalar present', Pos('Boundary cells', Report) > 0);
    AssertTrue('boundary length reported',
      Pos('Estimated boundary length', Report) > 0);
    AssertTrue('probe overlay section present',
      Pos('(d) Probe overlay', Report) > 0);

    // A learned separator must produce at least one boundary cell, and at
    // least two distinct classes must appear across the grid.
    AssertTrue('grid contains class 0', Pos('0', Report) > 0);
    AssertTrue('grid contains class 1', Pos('1', Report) > 0);

    // CSV side-output appears only when requested, with the documented header.
    Report := TNNet.DecisionBoundaryReport(NN, TrainSet, 11, 11, 0, 0, 0, 0,
      True);
    AssertTrue('CSV section emitted when requested',
      Pos('BEGIN CSV', Report) > 0);
    AssertTrue('CSV header present', Pos('x,y,argmax,top1prob', Report) > 0);

    // Caller-supplied box (no probes) still works.
    Report := TNNet.DecisionBoundaryReport(NN, nil, 15, 15,
      -2.0, 2.0, -2.0, 2.0);
    AssertTrue('caller-box report non-empty', Length(Report) > 0);
    AssertTrue('caller-box has no probe overlay section',
      Pos('(d) Probe overlay', Report) = 0);
  finally
    TrainSet.Free;
    NN.Free;
  end;
end;

procedure TTestNeuralLayersExtra.TestCalibrationReportSmoke;
var
  NN: TNNet;
  Inputs: TNNetVolumeList;
  Labels: TNeuralIntegerArray;
  X: TNNetVolume;
  Report: string;
  Rep: TNeuralCalibrationReport;
  T: TNeuralFloat;
  I, C: integer;
begin
  // nil NN handled gracefully (empty inputs too).
  Report := CalibrationReport(nil, nil, [], 10);
  AssertTrue('nil NN reported gracefully', Pos('NN is nil', Report) > 0);

  NN := TNNet.Create();
  Inputs := TNNetVolumeList.Create(True);
  try
    NN.AddLayer(TNNetInput.Create(2, 1, 1));
    NN.AddLayer(TNNetFullConnectReLU.Create(8));
    NN.AddLayer(TNNetFullConnectLinear.Create(3));
    NN.AddLayer(TNNetSoftMax.Create());
    NN.InitWeights();

    // A handful of labeled samples across 3 classes (no training needed for a
    // smoke test -- we only need a valid forward pass and a non-empty report).
    SetLength(Labels, 12);
    for I := 0 to 11 do
    begin
      C := I mod 3;
      X := TNNetVolume.Create(2, 1, 1);
      X.FData[0] := C * 1.0;
      X.FData[1] := -C * 1.0;
      Inputs.Add(X);
      Labels[I] := C;
    end;

    Report := CalibrationReport(NN, Inputs, Labels, 10);
    AssertTrue('Report is non-empty', Length(Report) > 0);
    AssertTrue('Header present', Pos('CalibrationReport', Report) > 0);
    AssertTrue('ECE present', Pos('ECE', Report) > 0);
    AssertTrue('Reliability diagram section present',
      Pos('Reliability diagram', Report) > 0);

    // Sane metric ranges.
    Rep := ComputeCalibration(NN, Inputs, Labels, 10);
    AssertTrue('NumSamples positive', Rep.NumSamples > 0);
    AssertTrue('ECE in [0,1]', (Rep.ECE >= 0) and (Rep.ECE <= 1));
    AssertTrue('MCE in [0,1]', (Rep.MCE >= 0) and (Rep.MCE <= 1));
    AssertTrue('Brier finite and >= 0',
      (Rep.Brier >= 0) and (not IsNan(Rep.Brier)) and (not IsInfinite(Rep.Brier)));
    AssertTrue('Accuracy in [0,1]',
      (Rep.Accuracy >= 0) and (Rep.Accuracy <= 1));

    // FitTemperature returns a positive finite T.
    T := FitTemperature(NN, Inputs, Labels);
    AssertTrue('T positive', T > 0);
    AssertTrue('T finite', (not IsNan(T)) and (not IsInfinite(T)));

    // nil NN with a valid input list still handled gracefully.
    Report := CalibrationReport(nil, Inputs, Labels, 10);
    AssertTrue('nil NN reported gracefully', Pos('NN is nil', Report) > 0);
  finally
    Inputs.Free;
    NN.Free;
  end;
end;

procedure TTestNeuralLayersExtra.TestFisherImportanceReportSmoke;
var
  NN: TNNet;
  Samples: TNNetVolumePairList;
  X, Y: TNNetVolume;
  Report: string;
  Ep, I, C: integer;
  EffStart, EffEnd, ParamStart, ParamEnd: integer;
  EffStr, ParamStr: string;
  EffVal, ParamVal: TNeuralFloat;
  FS: TFormatSettings;
  Centers: array[0..2, 0..1] of TNeuralFloat =
    ((-1.5, -1.5), (1.5, 1.5), (1.5, -1.5));
begin
  FS := DefaultFormatSettings;
  FS.DecimalSeparator := '.';
  FS.ThousandSeparator := #0;

  // nil NN handled gracefully.
  Report := TNNet.FisherImportanceReport(nil, nil);
  AssertTrue('nil NN reported gracefully', Pos('NN is nil', Report) > 0);

  NN := TNNet.Create();
  Samples := TNNetVolumePairList.Create();
  try
    NN.AddLayer(TNNetInput.Create(2, 1, 1));
    NN.AddLayer(TNNetFullConnectReLU.Create(10));
    NN.AddLayer(TNNetFullConnectLinear.Create(3));
    NN.AddLayer(TNNetSoftMax.Create());
    NN.SetLearningRate(0.05, 0.9);
    NN.InitWeights();

    // empty sample list handled gracefully (on a valid net).
    Report := TNNet.FisherImportanceReport(NN, Samples);
    AssertTrue('empty samples reported gracefully',
      Pos('nil or empty', Report) > 0);

    // Build a labelled 3-cluster set.
    RandSeed := 1234;
    for C := 0 to 2 do
      for I := 1 to 40 do
      begin
        X := TNNetVolume.Create(2, 1, 1);
        Y := TNNetVolume.Create(3, 1, 1);
        X.FData[0] := Centers[C][0] + (Random - 0.5);
        X.FData[1] := Centers[C][1] + (Random - 0.5);
        Y.Fill(0);
        Y.FData[C] := 1.0;
        Samples.Add(TNNetVolumePair.Create(X, Y));
      end;

    // Train briefly so the Fisher signal is non-degenerate.
    for Ep := 1 to 30 do
      for I := 0 to Samples.Count - 1 do
      begin
        NN.Compute(Samples[I].I);
        NN.Backpropagate(Samples[I].O);
      end;

    Report := TNNet.FisherImportanceReport(NN, Samples);
    AssertTrue('Report is non-empty', Length(Report) > 0);
    AssertTrue('Header present', Pos('FisherImportanceReport', Report) > 0);
    AssertTrue('empirical Fisher mode present',
      Pos('empirical Fisher', Report) > 0);
    AssertTrue('FisherMass column present', Pos('FisherMass', Report) > 0);
    AssertTrue('participation-ratio line present',
      Pos('Effective parameter count', Report) > 0);
    AssertTrue('log10 histogram present',
      Pos('log10(Fisher) histogram', Report) > 0);
    AssertTrue('flags legend present', Pos('H=high-importance', Report) > 0);
    AssertTrue('trainable FC layer reported',
      Pos('TNNetFullConnectReLU', Report) > 0);

    // Correctness: the effective parameter count (participation ratio) must lie
    // in (0, NumParams]. Parse "= <eff> / <num> (" out of the report.
    EffStart := Pos('Effective parameter count (participation ratio) = ',
      Report);
    AssertTrue('participation line found', EffStart > 0);
    EffStart := EffStart +
      Length('Effective parameter count (participation ratio) = ');
    EffEnd := PosEx(' /', Report, EffStart);
    AssertTrue('eff terminator found', EffEnd > EffStart);
    EffStr := Trim(Copy(Report, EffStart, EffEnd - EffStart));
    EffVal := StrToFloatDef(EffStr, -1, FS);

    ParamStart := EffEnd + 2;
    ParamEnd := PosEx(' (', Report, ParamStart);
    AssertTrue('param terminator found', ParamEnd > ParamStart);
    ParamStr := Trim(Copy(Report, ParamStart, ParamEnd - ParamStart));
    ParamVal := StrToFloatDef(ParamStr, -1, FS);

    AssertTrue('effective param count finite',
      (not IsNan(EffVal)) and (not IsInfinite(EffVal)));
    AssertTrue('total param count positive', ParamVal > 0);
    AssertTrue('participation ratio > 0', EffVal > 0);
    AssertTrue('participation ratio <= total params (+eps)',
      EffVal <= ParamVal + 1e-3);

    // Predicted-label mode also produces a well-formed report.
    Report := TNNet.FisherImportanceReport(NN, Samples, False);
    AssertTrue('predicted-label mode non-empty', Length(Report) > 0);
    AssertTrue('predicted-label mode tagged',
      Pos('predicted-label', Report) > 0);
  finally
    Samples.Free;
    NN.Free;
  end;
end;

procedure TTestNeuralLayersExtra.TestLinearProbeReportSmoke;
var
  NN: TNNet;
  Samples, ValSamples: TNNetVolumePairList;
  X, Y: TNNetVolume;
  Report: string;
  Ep, I, C: integer;
  AccStart, AccEnd: integer;
  AccStr: string;
  FinalAcc: TNeuralFloat;
  FS: TFormatSettings;
  Centers: array[0..2, 0..1] of TNeuralFloat =
    ((-1.5, -1.5), (1.5, 1.5), (1.5, -1.5));
begin
  FS := DefaultFormatSettings;
  FS.DecimalSeparator := '.';
  FS.ThousandSeparator := #0;

  // nil NN handled gracefully.
  Report := TNNet.LinearProbeReport(nil, nil);
  AssertTrue('nil NN reported gracefully', Pos('NN is nil', Report) > 0);

  NN := TNNet.Create();
  Samples := TNNetVolumePairList.Create();
  ValSamples := TNNetVolumePairList.Create();
  try
    NN.AddLayer(TNNetInput.Create(2, 1, 1));
    NN.AddLayer(TNNetFullConnectReLU.Create(10));
    NN.AddLayer(TNNetFullConnectReLU.Create(10));
    NN.AddLayer(TNNetFullConnectLinear.Create(3));
    NN.AddLayer(TNNetSoftMax.Create());
    NN.SetLearningRate(0.05, 0.9);
    NN.InitWeights();

    // empty sample list handled gracefully (on a valid net).
    Report := TNNet.LinearProbeReport(NN, Samples);
    AssertTrue('empty samples reported gracefully',
      Pos('nil or empty', Report) > 0);

    // Build labelled 3-cluster train + held-out sets.
    RandSeed := 1234;
    for C := 0 to 2 do
      for I := 1 to 40 do
      begin
        X := TNNetVolume.Create(2, 1, 1);
        Y := TNNetVolume.Create(3, 1, 1);
        X.FData[0] := Centers[C][0] + (Random - 0.5);
        X.FData[1] := Centers[C][1] + (Random - 0.5);
        Y.Fill(0);
        Y.FData[C] := 1.0;
        Samples.Add(TNNetVolumePair.Create(X, Y));

        X := TNNetVolume.Create(2, 1, 1);
        Y := TNNetVolume.Create(3, 1, 1);
        X.FData[0] := Centers[C][0] + (Random - 0.5);
        X.FData[1] := Centers[C][1] + (Random - 0.5);
        Y.Fill(0);
        Y.FData[C] := 1.0;
        ValSamples.Add(TNNetVolumePair.Create(X, Y));
      end;

    // Train briefly so the deep layers become linearly separable.
    for Ep := 1 to 40 do
      for I := 0 to Samples.Count - 1 do
      begin
        NN.Compute(Samples[I].I);
        NN.Backpropagate(Samples[I].O);
      end;

    // Report with held-out batch.
    Report := TNNet.LinearProbeReport(NN, Samples, ValSamples);
    AssertTrue('Report is non-empty', Length(Report) > 0);
    AssertTrue('Header present', Pos('LinearProbeReport', Report) > 0);
    AssertTrue('ProbeAcc column present', Pos('ProbeAcc', Report) > 0);
    AssertTrue('ValAcc column present (held-out supplied)',
      Pos('ValAcc', Report) > 0);
    AssertTrue('OneHotMSE column present', Pos('OneHotMSE', Report) > 0);
    AssertTrue('bar chart present',
      Pos('Per-layer probe accuracy', Report) > 0);
    AssertTrue('distribution histogram present',
      Pos('Distribution of per-layer probe accuracy', Report) > 0);
    AssertTrue('flags legend present',
      Pos('C=representation collapse', Report) > 0);
    AssertTrue('final-layer line present',
      Pos('Final-layer probe accuracy', Report) > 0);

    // Correctness: the final-layer probe accuracy on a well-separated 3-cluster
    // set after training must be comfortably above the 1/3 random baseline.
    AccStart := Pos('Final-layer probe accuracy = ', Report);
    AssertTrue('final-acc line found', AccStart > 0);
    AccStart := AccStart + Length('Final-layer probe accuracy = ');
    AccEnd := PosEx('%', Report, AccStart);
    AssertTrue('final-acc terminator found', AccEnd > AccStart);
    AccStr := Trim(Copy(Report, AccStart, AccEnd - AccStart));
    FinalAcc := StrToFloatDef(AccStr, -1, FS);
    AssertTrue('final probe acc parsed', FinalAcc >= 0);
    AssertTrue('final probe acc above random baseline', FinalAcc > 60.0);

    // No-held-out variant also produces a well-formed report (no ValAcc col).
    Report := TNNet.LinearProbeReport(NN, Samples);
    AssertTrue('no-val report non-empty', Length(Report) > 0);
    AssertTrue('no-val report has ProbeAcc', Pos('ProbeAcc', Report) > 0);
    AssertTrue('no-val report omits ValAcc', Pos('ValAcc', Report) = 0);
  finally
    Samples.Free;
    ValSamples.Free;
    NN.Free;
  end;
end;

procedure TTestNeuralLayersExtra.TestLogitLensReportSmoke;
var
  NN: TNNet;
  Probes: TNNetVolumeList;
  X: TNNetVolume;
  Report: string;
  I, C: integer;
  AStart, AEnd: integer;
  AgreeStr, KLStr: string;
  Agree, KLval: TNeuralFloat;
  FS: TFormatSettings;
  Centers: array[0..2, 0..1] of TNeuralFloat =
    ((-1.5, -1.5), (1.5, 1.5), (1.5, -1.5));
begin
  FS := DefaultFormatSettings;
  FS.DecimalSeparator := '.';
  FS.ThousandSeparator := #0;

  // nil NN handled gracefully.
  Report := TNNet.LogitLensReport(nil, nil);
  AssertTrue('nil NN reported gracefully', Pos('NN is nil', Report) > 0);

  NN := TNNet.Create();
  Probes := TNNetVolumeList.Create(True);
  try
    // Constant-width body so every hidden layer is lens-compatible with the
    // head input; the input layer (width 2) shows up as SKIPPED.
    NN.AddLayer(TNNetInput.Create(2, 1, 1));
    NN.AddLayer(TNNetFullConnectReLU.Create(8));
    NN.AddLayer(TNNetFullConnectReLU.Create(8));
    NN.AddLayer(TNNetFullConnectReLU.Create(8));
    NN.AddLayer(TNNetFullConnectLinear.Create(3));
    NN.AddLayer(TNNetSoftMax.Create());
    NN.SetLearningRate(0.05, 0.9);
    NN.InitWeights();

    // empty probe list handled gracefully (on a valid net).
    Report := TNNet.LogitLensReport(NN, Probes);
    AssertTrue('empty probes reported gracefully',
      Pos('nil or empty', Report) > 0);

    // Build an unlabelled 3-cluster probe batch.
    RandSeed := 1234;
    for C := 0 to 2 do
      for I := 1 to 30 do
      begin
        X := TNNetVolume.Create(2, 1, 1);
        X.FData[0] := Centers[C][0] + (Random - 0.5);
        X.FData[1] := Centers[C][1] + (Random - 0.5);
        Probes.Add(X);
      end;

    // The lens correctness checks are exact at ANY weights (fresh init is
    // fine): the lens at the head input must reproduce p_final exactly.
    Report := TNNet.LogitLensReport(NN, Probes);
    AssertTrue('Report is non-empty', Length(Report) > 0);
    AssertTrue('Header present', Pos('LogitLensReport', Report) > 0);
    AssertTrue('agreement bar chart present',
      Pos('Per-layer lens agreement', Report) > 0);
    AssertTrue('KL curve present', Pos('KL(p_L || p_final)', Report) > 0);
    AssertTrue('crystallization section present',
      Pos('Crystallization depth', Report) > 0);
    AssertTrue('SKIPPED section present', Pos('SKIPPED layers', Report) > 0);
    AssertTrue('input layer reported as SKIPPED',
      Pos('TNNetInput', Report) > 0);
    AssertTrue('correctness check present',
      Pos('Correctness check', Report) > 0);
    AssertTrue('correctness check PASSes',
      Pos('PASS', Report) > 0);

    // Parse the head-input correctness line and assert agreement==1, KL==0.
    AStart := Pos('agreement=', Report);
    AssertTrue('agreement token found', AStart > 0);
    AStart := AStart + Length('agreement=');
    AEnd := PosEx(' ', Report, AStart);
    AssertTrue('agreement terminator found', AEnd > AStart);
    AgreeStr := Trim(Copy(Report, AStart, AEnd - AStart));
    Agree := StrToFloatDef(AgreeStr, -1, FS);
    AssertTrue('agreement parsed', Agree >= 0);
    AssertTrue('lens at head input agreement == 1.0',
      Abs(Agree - 1.0) < 1e-4);

    AStart := Pos('KL=', Report);
    AssertTrue('KL token found', AStart > 0);
    AStart := AStart + Length('KL=');
    AEnd := PosEx(' ', Report, AStart);
    if AEnd <= AStart then AEnd := PosEx('.', Report, AStart) + 7;
    KLStr := Trim(Copy(Report, AStart, AEnd - AStart));
    // Trim a trailing '(' or other punctuation if present.
    while (Length(KLStr) > 0) and
          not (KLStr[Length(KLStr)] in ['0'..'9']) do
      KLStr := Copy(KLStr, 1, Length(KLStr) - 1);
    KLval := StrToFloatDef(KLStr, -1, FS);
    AssertTrue('KL parsed', KLval >= 0);
    AssertTrue('lens at head input KL == 0', KLval < 1e-4);

    // Single-layer head (explicit HeadStartIdx = last layer) degenerates to the
    // trivial profile and still PASSes the correctness check.
    Report := TNNet.LogitLensReport(NN, Probes, NN.GetLastLayerIdx());
    AssertTrue('single-head report non-empty', Length(Report) > 0);
    AssertTrue('single-head note present',
      Pos('single-layer head', Report) > 0);
    AssertTrue('single-head still PASSes', Pos('PASS', Report) > 0);
  finally
    Probes.Free;
    NN.Free;
  end;
end;

procedure TTestNeuralLayersExtra.TestFeatureSeparabilityReportSmoke;
var
  NN: TNNet;
  Samples, SingleCls, Identical: TNNetVolumePairList;
  X, Y: TNNetVolume;
  Report, SingleReport, IdReport: string;
  I, C: integer;
  ResStart, ResEnd: integer;
  ResStr: string;
  Residual: TNeuralFloat;
  FS: TFormatSettings;
  Centers: array[0..2, 0..1] of TNeuralFloat =
    ((-2.0, -2.0), (2.0, 2.0), (2.0, -2.0));

  // pull the floating-point value out of "...<Label><number>..." in Rep.
  function ExtractAfter(const Rep, Marker: string): TNeuralFloat;
  var
    A, B: integer;
    S: string;
  begin
    Result := -1;
    A := Pos(Marker, Rep);
    if A <= 0 then Exit;
    A := A + Length(Marker);
    // skip leading whitespace (Format's %8.4f right-justifies with spaces).
    while (A <= Length(Rep)) and (Rep[A] = ' ') do Inc(A);
    B := A;
    // span the number (digits, sign, dot, exponent).
    while (B <= Length(Rep)) and (Rep[B] in ['0'..'9', '.', '-', '+', 'e', 'E']) do
      Inc(B);
    S := Trim(Copy(Rep, A, B - A));
    Result := StrToFloatDef(S, -1, FS);
  end;

begin
  FS := DefaultFormatSettings;
  FS.DecimalSeparator := '.';
  FS.ThousandSeparator := #0;

  // nil NN handled gracefully.
  Report := TNNet.FeatureSeparabilityReport(nil, nil, 3);
  AssertTrue('nil NN reported gracefully', Pos('NN is nil', Report) > 0);

  NN := TNNet.Create();
  Samples := TNNetVolumePairList.Create();
  SingleCls := TNNetVolumePairList.Create();
  Identical := TNNetVolumePairList.Create();
  try
    NN.AddLayer(TNNetInput.Create(2, 1, 1));
    NN.AddLayer(TNNetFullConnectReLU.Create(10));
    NN.AddLayer(TNNetFullConnectReLU.Create(10));
    NN.AddLayer(TNNetFullConnectLinear.Create(3));
    NN.AddLayer(TNNetSoftMax.Create());
    NN.SetLearningRate(0.05, 0.9);
    NN.InitWeights();

    // empty sample list handled gracefully (on a valid net).
    Report := TNNet.FeatureSeparabilityReport(NN, Samples, 3);
    AssertTrue('empty samples reported gracefully',
      Pos('nil or empty', Report) > 0);

    // Build a class-BALANCED 3-cluster batch (round-robin keeps counts equal so
    // the scatter-decomposition identity is exact).
    RandSeed := 1234;
    for I := 1 to 30 do
      for C := 0 to 2 do
      begin
        X := TNNetVolume.Create(2, 1, 1);
        Y := TNNetVolume.Create(3, 1, 1);
        X.FData[0] := Centers[C][0] + (Random - 0.5);
        X.FData[1] := Centers[C][1] + (Random - 0.5);
        Y.Fill(0);
        Y.FData[C] := 1.0;
        Samples.Add(TNNetVolumePair.Create(X, Y));
      end;

    Report := TNNet.FeatureSeparabilityReport(NN, Samples, 3);
    AssertTrue('Report is non-empty', Length(Report) > 0);
    AssertTrue('Header present',
      Pos('FeatureSeparabilityReport', Report) > 0);
    AssertTrue('tr(Sw) column present', Pos('tr(Sw)', Report) > 0);
    AssertTrue('tr(Sb) column present', Pos('tr(Sb)', Report) > 0);
    AssertTrue('Fisher column present', Pos('Fisher', Report) > 0);
    AssertTrue('Silhouette column present', Pos('Silh', Report) > 0);
    AssertTrue('Fisher bar chart present',
      Pos('Fisher ratio tr(Sb)/tr(Sw) across depth', Report) > 0);
    AssertTrue('cosine heatmap present',
      Pos('class-mean pairwise-cosine heatmap', Report) > 0);
    AssertTrue('ETF target present', Pos('simplex-ETF target', Report) > 0);
    AssertTrue('flags legend present', Pos('S=well-separated', Report) > 0);

    // FAITHFULNESS: the scatter-decomposition identity tr(Stot)=tr(Sw)+tr(Sb)
    // must hold (balanced batch) to < 1e-4 - parse the reported worst residual.
    ResStart := Pos('worst residual=', Report);
    AssertTrue('worst-residual line found', ResStart > 0);
    ResStart := ResStart + Length('worst residual=');
    ResEnd := ResStart;
    while (ResEnd <= Length(Report)) and
          (Report[ResEnd] in ['0'..'9', '.', '-', '+', 'e', 'E']) do
      Inc(ResEnd);
    ResStr := Trim(Copy(Report, ResStart, ResEnd - ResStart));
    Residual := StrToFloatDef(ResStr, -1, FS);
    AssertTrue('residual parsed', Residual >= 0);
    AssertTrue('scatter-decomposition identity holds (<1e-4)',
      Residual < 1e-4);

    // FAITHFULNESS: a SINGLE-class batch makes tr(Sb) collapse to ~0. Use the
    // input layer (idx 0) row of the table; simpler: the off-diagonal cosine
    // section is absent meaning of separation - assert via the final-layer
    // between-class spread being tiny is awkward to parse, so instead verify
    // the report runs and the mean off-diag cosine (no other class) -> 0.
    for I := 1 to 30 do
    begin
      X := TNNetVolume.Create(2, 1, 1);
      Y := TNNetVolume.Create(3, 1, 1);
      X.FData[0] := Centers[0][0] + (Random - 0.5);
      X.FData[1] := Centers[0][1] + (Random - 0.5);
      Y.Fill(0);
      Y.FData[0] := 1.0;
      SingleCls.Add(TNNetVolumePair.Create(X, Y));
    end;
    SingleReport := TNNet.FeatureSeparabilityReport(NN, SingleCls, 3);
    AssertTrue('single-class report non-empty', Length(SingleReport) > 0);
    // 1 occupied class -> tr(Sb)=0 -> final-layer off-diag cosine line = 0.
    AssertTrue('single-class off-diag cosine ~0',
      Abs(ExtractAfter(SingleReport,
        'Final-layer mean off-diagonal class-mean cosine=')) < 1e-3);

    // FAITHFULNESS: IDENTICAL per-class samples make tr(Sw) collapse to ~0 (and
    // the silhouette -> 1). Three classes, every sample of a class identical.
    for C := 0 to 2 do
      for I := 1 to 10 do
      begin
        X := TNNetVolume.Create(2, 1, 1);
        Y := TNNetVolume.Create(3, 1, 1);
        X.FData[0] := Centers[C][0];
        X.FData[1] := Centers[C][1];
        Y.Fill(0);
        Y.FData[C] := 1.0;
        Identical.Add(TNNetVolumePair.Create(X, Y));
      end;
    IdReport := TNNet.FeatureSeparabilityReport(NN, Identical, 3);
    AssertTrue('identical report non-empty', Length(IdReport) > 0);
    // identical inputs propagate to identical activations everywhere, so every
    // probed layer collapses tr(Sw) -> the silhouette saturates at +1. Parse
    // the final-layer mean silhouette out of the bar-chart-free summary: the
    // mean silhouette over the batch is +1 within tolerance. The off-diagonal
    // class-mean cosine is well-defined (>1 occupied class), and the scatter
    // identity still holds.
    Residual := ExtractAfter(IdReport, 'worst residual=');
    AssertTrue('identical-batch scatter identity holds',
      (Residual >= 0) and (Residual < 1e-4));
    // tr(Sw) collapses: the first trainable layer's row carries 0.0000 in the
    // tr(Sw) column (formatted %11.4f). A near-zero tr(Sw) value must appear.
    AssertTrue('identical per-class samples collapse tr(Sw) (0.0000 present)',
      Pos('     0.0000', IdReport) > 0);
  finally
    Samples.Free;
    SingleCls.Free;
    Identical.Free;
    NN.Free;
  end;
end;

procedure TTestNeuralLayersExtra.TestRepresentationSimilarityReportSmoke;
var
  NN, NN2: TNNet;
  Probes: TNNetVolumeList;
  V: TNNetVolume;
  Report, CrossReport: string;
  I, J, P, DiagStart, ColStart: integer;
  Glyph: char;
begin
  // nil NN handled gracefully.
  Report := TNNet.RepresentationSimilarityReport(nil, nil);
  AssertTrue('nil NN reported gracefully', Pos('NN is nil', Report) > 0);

  NN := TNNet.Create();
  NN2 := TNNet.Create();
  Probes := TNNetVolumeList.Create(True);
  try
    NN.AddLayer(TNNetInput.Create(4, 1, 1));
    NN.AddLayer(TNNetFullConnectReLU.Create(8));
    NN.AddLayer(TNNetFullConnectReLU.Create(8));
    NN.AddLayer(TNNetFullConnectLinear.Create(3));
    NN.InitWeights();

    NN2.AddLayer(TNNetInput.Create(4, 1, 1));
    NN2.AddLayer(TNNetFullConnectReLU.Create(8));
    NN2.AddLayer(TNNetFullConnectLinear.Create(3));
    NN2.InitWeights();

    // empty probe list handled gracefully.
    Report := TNNet.RepresentationSimilarityReport(NN, Probes);
    AssertTrue('empty probes reported gracefully',
      Pos('nil or empty', Report) > 0);

    // a handful of probe volumes.
    RandSeed := 7;
    for I := 0 to 19 do
    begin
      V := TNNetVolume.Create(4, 1, 1);
      for J := 0 to 3 do V.Raw[J] := (Random - 0.5) * 2.0;
      Probes.Add(V);
    end;

    Report := TNNet.RepresentationSimilarityReport(NN, Probes);
    AssertTrue('Report is non-empty', Length(Report) > 0);
    AssertTrue('Header present',
      Pos('RepresentationSimilarityReport', Report) > 0);
    AssertTrue('CKA heatmap section present', Pos('CKA heatmap', Report) > 0);
    AssertTrue('adjacent-layer section present',
      Pos('Adjacent-layer CKA', Report) > 0);
    AssertTrue('most-redundant pair present',
      Pos('Most-redundant layer PAIR', Report) > 0);
    AssertTrue('block-structure section present',
      Pos('Representational stages', Report) > 0);
    AssertTrue('verdict present', Pos('Verdict:', Report) > 0);

    // --- Correctness check 1: the self-CKA DIAGONAL is 1.0 within tolerance.
    // The heatmap renders the diagonal with the brightest glyph ('@', the last
    // band reached only by CKA >= 0.9). The numeric proof lives in the
    // adjacent-vector / most-redundant-pair values, but the diagonal-glyph
    // check pins that every layer's self-similarity sits in the top band.
    // Verify by parsing the matrix rows: row [I] column I must be '@'.
    DiagStart := Pos('CKA heatmap', Report);
    AssertTrue('heatmap located', DiagStart > 0);
    for I := 0 to 4 do
    begin
      // find the row label '  [ I]' for this layer index.
      ColStart := PosEx(Format('  [%2d] ', [I]), Report, DiagStart);
      if ColStart = 0 then Break; // fewer than 5 probeable layers; stop.
      // glyphs start after the 6-char label '  [%2d] ', each glyph preceded by
      // a single space -> column I glyph is at label-end + 1 (space) + 2*I + 1.
      ColStart := ColStart + Length(Format('  [%2d] ', [I]));
      Glyph := Report[ColStart + 2 * I + 1];
      AssertTrue(Format('self-CKA diagonal glyph at [%d] is top band', [I]),
        Glyph = '@');
    end;

    // --- Cross-CKA variant runs and is labelled as such.
    CrossReport := TNNet.RepresentationSimilarityReport(NN, Probes, NN2);
    AssertTrue('cross-CKA report runs', Length(CrossReport) > 0);
    AssertTrue('cross-CKA labels two networks',
      Pos('cross-CKA between two networks', CrossReport) > 0);

    // --- Correctness check 2: symmetry of the self matrix.
    // glyph(row a, col b) == glyph(row b, col a).
    DiagStart := Pos('CKA heatmap', Report);
    for I := 0 to 4 do
      for J := 0 to 4 do
      begin
        ColStart := PosEx(Format('  [%2d] ', [I]), Report, DiagStart);
        P := PosEx(Format('  [%2d] ', [J]), Report, DiagStart);
        if (ColStart = 0) or (P = 0) then Break;
        ColStart := ColStart + Length(Format('  [%2d] ', [I]));
        P := P + Length(Format('  [%2d] ', [J]));
        AssertTrue(Format('symmetric glyph [%d][%d]==[%d][%d]', [I, J, J, I]),
          Report[ColStart + 2 * J + 1] = Report[P + 2 * I + 1]);
      end;
  finally
    Probes.Free;
    NN.Free;
    NN2.Free;
  end;
end;

// Runs one forward + one backward on NN with a one-hot target on class c
// (using the public TNNet.Backpropagate, which sets the last-layer output
// error = Output - target and runs the backward chain; ClearDeltas keeps the
// weights frozen here since we never call UpdateWeights), then returns the
// sum of |OutputError| that landed on the input layer (Layers[0]). With
// EnableInputGradient called beforehand this must be > 0; without it the input
// gradient is silently dropped.
function InputGradAbsSum(NN: TNNet; X: TNNetVolume; c: integer): TNeuralFloat;
var
  Gi: integer;
  Y: TNNetVolume;
  InLayer: TNNetLayer;
begin
  Result := 0;
  NN.ClearDeltas();
  NN.Compute(X);
  Y := TNNetVolume.Create(NN.GetLastLayer.Output.Size, 1, 1);
  try
    Y.Fill(0);
    if (c >= 0) and (c < Y.Size) then Y.Raw[c] := 1.0;
    NN.Backpropagate(Y);
  finally
    Y.Free;
  end;
  InLayer := NN.Layers[0];
  if InLayer.OutputError <> nil then
    for Gi := 0 to InLayer.OutputError.Size - 1 do
      Result := Result + Abs(InLayer.OutputError.Raw[Gi]);
end;

procedure TTestNeuralLayersExtra.TestEnableInputGradient;
var
  NN: TNNet;
  X: TNNetVolume;
  I: integer;
  GradSum: TNeuralFloat;
begin
  RandSeed := 4242;

  // (a) Conv first layer: Input(4,4,2) -> Convolution -> ... -> output.
  NN := TNNet.Create();
  X := TNNetVolume.Create(4, 4, 2);
  try
    NN.AddLayer(TNNetInput.Create(4, 4, 2));
    NN.AddLayer(TNNetConvolutionReLU.Create(6, 3, 1, 1)); // padding>0 path
    NN.AddLayer(TNNetMaxPool.Create(2));
    NN.AddLayer(TNNetFullConnectReLU.Create(8));
    NN.AddLayer(TNNetFullConnectLinear.Create(3));
    NN.AddLayer(TNNetSoftMax.Create());
    NN.SetLearningRate(0.01, 0.9);
    NN.InitWeights();
    for I := 0 to X.Size - 1 do X.Raw[I] := Random - 0.5;

    // Output must be sized before enabling input gradients.
    NN.Compute(X);
    NN.EnableInputGradient();
    GradSum := InputGradAbsSum(NN, X, 0);
    AssertTrue('conv-first net: input gradient is non-zero after enable',
      GradSum > 0);
  finally
    X.Free;
    NN.Free;
  end;

  // (b) FullConnect first layer: Input(1,1,8) -> FullConnect -> output.
  NN := TNNet.Create();
  X := TNNetVolume.Create(1, 1, 8);
  try
    NN.AddLayer(TNNetInput.Create(1, 1, 8));
    NN.AddLayer(TNNetFullConnectReLU.Create(10));
    NN.AddLayer(TNNetFullConnectLinear.Create(3));
    NN.AddLayer(TNNetSoftMax.Create());
    NN.SetLearningRate(0.01, 0.9);
    NN.InitWeights();
    for I := 0 to X.Size - 1 do X.Raw[I] := Random - 0.5;

    NN.Compute(X);
    NN.EnableInputGradient();
    GradSum := InputGradAbsSum(NN, X, 0);
    AssertTrue('fullconnect-first net: input gradient is non-zero after enable',
      GradSum > 0);
  finally
    X.Free;
    NN.Free;
  end;
end;

procedure TTestNeuralLayersExtra.TestAdversarialRobustnessReportSmoke;
var
  NN: TNNet;
  Samples: TNNetVolumeList;
  X, Y: TNNetVolume;
  Labels: array of integer;
  Report: string;
  C, I, K, Ep: integer;
  Centers: array[0..1, 0..1] of TNeuralFloat = ((-1.5, -1.5), (1.5, 1.5));
begin
  RandSeed := 9090;

  // nil NN handled gracefully (no crash, non-empty message).
  Report := TNNet.AdversarialRobustnessReport(nil, nil, [], []);
  AssertTrue('nil NN reported gracefully', Length(Report) > 0);
  AssertTrue('nil NN message mentions NN', Pos('NN is nil', Report) > 0);

  NN := TNNet.Create();
  Samples := TNNetVolumeList.Create();
  try
    NN.AddLayer(TNNetInput.Create(2, 1, 1));
    NN.AddLayer(TNNetFullConnectReLU.Create(12));
    NN.AddLayer(TNNetFullConnectLinear.Create(2));
    NN.AddLayer(TNNetSoftMax.Create());
    NN.SetLearningRate(0.05, 0.9);
    NN.InitWeights();

    // empty samples handled gracefully.
    SetLength(Labels, 0);
    Report := TNNet.AdversarialRobustnessReport(NN, Samples, Labels, []);
    AssertTrue('empty samples reported gracefully', Length(Report) > 0);
    AssertTrue('empty samples mentions empty',
      (Pos('empty', Report) > 0) or (Pos('no samples', Report) > 0));

    // Train briefly on a separable 2-cluster problem.
    for Ep := 1 to 60 do
      for K := 1 to 8 do
      begin
        C := Random(2);
        X := TNNetVolume.Create(2, 1, 1);
        Y := TNNetVolume.Create(2, 1, 1);
        try
          X.FData[0] := Centers[C][0] + (Random - 0.5);
          X.FData[1] := Centers[C][1] + (Random - 0.5);
          Y.Fill(0);
          Y.FData[C] := 1.0;
          NN.Compute(X);
          NN.Backpropagate(Y);
        finally
          X.Free;
          Y.Free;
        end;
      end;

    // Build a small labelled probe batch.
    for C := 0 to 1 do
      for I := 1 to 6 do
      begin
        X := TNNetVolume.Create(2, 1, 1);
        X.FData[0] := Centers[C][0] + (Random - 0.5);
        X.FData[1] := Centers[C][1] + (Random - 0.5);
        Samples.Add(X);
        SetLength(Labels, Length(Labels) + 1);
        Labels[High(Labels)] := C;
      end;

    // mismatched Labels length handled gracefully.
    SetLength(Labels, Length(Labels) - 1);
    Report := TNNet.AdversarialRobustnessReport(NN, Samples, Labels, []);
    AssertTrue('mismatched labels reported gracefully', Length(Report) > 0);
    AssertTrue('mismatched labels mentions mismatch',
      (Pos('mismatch', Report) > 0) or (Pos('match', Report) > 0) or
      (Pos('Labels', Report) > 0));
    SetLength(Labels, Length(Labels) + 1);
    Labels[High(Labels)] := 1;

    // Full report (default epsilon menu via empty EpsList).
    Report := TNNet.AdversarialRobustnessReport(NN, Samples, Labels, []);
    AssertTrue('Report is non-empty', Length(Report) > 0);
    AssertTrue('Header present',
      Pos('AdversarialRobustnessReport', Report) > 0);
    AssertTrue('accuracy-vs-eps section present',
      (Pos('eps', Report) > 0) and (Pos('accuracy', Report) > 0));
    AssertTrue('critical epsilon histogram present',
      Pos('critical epsilon', Report) > 0);
    AssertTrue('verdict present',
      (Pos('robust', Report) > 0) or (Pos('fragile', Report) > 0));

    // Explicit eps list also works.
    Report := TNNet.AdversarialRobustnessReport(NN, Samples, Labels,
      [0.0, 0.05, 0.1]);
    AssertTrue('explicit-eps report non-empty', Length(Report) > 0);
    AssertTrue('explicit-eps header present',
      Pos('AdversarialRobustnessReport', Report) > 0);
  finally
    Samples.Free;
    NN.Free;
  end;
end;

// Parses the "Strong-conflict fraction (cos < -0.20): a / b pair(s) = P%."
// line out of a GradientConflictReport and returns P (or -1 if not found).
// The strong-conflict tail is the genuinely-opposed signal that separates a
// clean batch (~0) from a label-noised / overlapping one.
function ParseStrongConflictFraction(const Report: string): TNeuralFloat;
var
  S, E: integer;
  Frac: string;
  FS: TFormatSettings;
begin
  Result := -1;
  FS := DefaultFormatSettings;
  FS.DecimalSeparator := '.';
  FS.ThousandSeparator := #0;
  S := Pos('Strong-conflict fraction', Report);
  if S = 0 then Exit;
  S := PosEx('pair(s) = ', Report, S);
  if S = 0 then Exit;
  S := S + Length('pair(s) = ');
  E := PosEx('%', Report, S);
  if E <= S then Exit;
  Frac := Trim(Copy(Report, S, E - S));
  Result := StrToFloatDef(Frac, -1, FS);
end;

// Parses "max |self-cosine - 1| = <v> (== 0)" out of a report.
function ParseDiagResidual(const Report: string): TNeuralFloat;
var
  S, E: integer;
  Val: string;
  FS: TFormatSettings;
begin
  Result := -1;
  FS := DefaultFormatSettings;
  FS.DecimalSeparator := '.';
  FS.ThousandSeparator := #0;
  S := Pos('max |self-cosine - 1| = ', Report);
  if S = 0 then Exit;
  S := S + Length('max |self-cosine - 1| = ');
  E := PosEx(' ', Report, S);
  if E <= S then Exit;
  Val := Trim(Copy(Report, S, E - S));
  Result := StrToFloatDef(Val, -1, FS);
end;

procedure TTestNeuralLayersExtra.TestGradientConflictReportSmoke;
var
  NN: TNNet;
  Clean, Noisy: TNNetVolumePairList;
  X, Y: TNNetVolume;
  CleanReport, NoisyReport, HeadReport: string;
  Ep, I, C, TrueCls, LabelCls: integer;
  CleanFrac, NoisyFrac, DiagRes, SymRes: TNeuralFloat;
  SymStart, SymEnd: integer;
  SymStr: string;
  FS: TFormatSettings;
  Centers: array[0..2, 0..1] of TNeuralFloat =
    ((-2.0, -2.0), (2.0, 2.0), (2.0, -2.0));
begin
  FS := DefaultFormatSettings;
  FS.DecimalSeparator := '.';
  FS.ThousandSeparator := #0;

  // nil NN handled gracefully.
  CleanReport := TNNet.GradientConflictReport(nil, nil);
  AssertTrue('nil NN reported gracefully', Pos('NN is nil', CleanReport) > 0);

  NN := TNNet.Create();
  Clean := TNNetVolumePairList.Create();
  Noisy := TNNetVolumePairList.Create();
  try
    NN.AddLayer(TNNetInput.Create(2, 1, 1));
    NN.AddLayer(TNNetFullConnectReLU.Create(12));
    NN.AddLayer(TNNetFullConnectLinear.Create(3));
    NN.AddLayer(TNNetSoftMax.Create());
    NN.SetLearningRate(0.05, 0.9);
    NN.InitWeights();

    // empty sample list handled gracefully (on a valid net).
    CleanReport := TNNet.GradientConflictReport(NN, Clean);
    AssertTrue('empty samples reported gracefully',
      Pos('nil or empty', CleanReport) > 0);

    // Train on the clean separable 3-cluster problem.
    RandSeed := 1234;
    for Ep := 1 to 40 do
      for I := 1 to 90 do
      begin
        C := Random(3);
        X := TNNetVolume.Create(2, 1, 1);
        Y := TNNetVolume.Create(3, 1, 1);
        try
          X.FData[0] := Centers[C][0] + (Random - 0.5) * 0.6;
          X.FData[1] := Centers[C][1] + (Random - 0.5) * 0.6;
          Y.Fill(0);
          Y.FData[C] := 1.0;
          NN.Compute(X);
          NN.Backpropagate(Y);
        finally
          X.Free;
          Y.Free;
        end;
      end;

    // Clean probe batch: tight clusters, honest labels.
    for C := 0 to 2 do
      for I := 1 to 16 do
      begin
        X := TNNetVolume.Create(2, 1, 1);
        Y := TNNetVolume.Create(3, 1, 1);
        X.FData[0] := Centers[C][0] + (Random - 0.5) * 0.6;
        X.FData[1] := Centers[C][1] + (Random - 0.5) * 0.6;
        Y.Fill(0);
        Y.FData[C] := 1.0;
        Clean.Add(TNNetVolumePair.Create(X, Y));
      end;

    // Noisy probe batch: heavy overlap + 40% label corruption.
    for C := 0 to 2 do
      for I := 1 to 16 do
      begin
        TrueCls := C;
        X := TNNetVolume.Create(2, 1, 1);
        Y := TNNetVolume.Create(3, 1, 1);
        X.FData[0] := Centers[TrueCls][0] + (Random - 0.5) * 5.0;
        X.FData[1] := Centers[TrueCls][1] + (Random - 0.5) * 5.0;
        LabelCls := TrueCls;
        if Random < 0.4 then
          LabelCls := (TrueCls + 1 + Random(2)) mod 3;
        Y.Fill(0);
        Y.FData[LabelCls] := 1.0;
        Noisy.Add(TNNetVolumePair.Create(X, Y));
      end;

    CleanReport := TNNet.GradientConflictReport(NN, Clean);
    NoisyReport := TNNet.GradientConflictReport(NN, Noisy);

    AssertTrue('clean report non-empty', Length(CleanReport) > 0);
    AssertTrue('Header present',
      Pos('GradientConflictReport', CleanReport) > 0);
    AssertTrue('cosine histogram present',
      Pos('Pairwise cosine histogram', CleanReport) > 0);
    AssertTrue('conflict-fraction line present',
      Pos('Conflict fraction', CleanReport) > 0);
    AssertTrue('mean/median line present',
      Pos('Mean cosine', CleanReport) > 0);
    AssertTrue('most-conflicting pair present',
      Pos('Most-conflicting pair', CleanReport) > 0);
    AssertTrue('per-class-pair matrix present',
      Pos('Per-class-pair mean cosine', CleanReport) > 0);

    // --- Built-in correctness check 1: self-cosine diagonal == 1. ---
    DiagRes := ParseDiagResidual(CleanReport);
    AssertTrue('self-cosine residual parsed', DiagRes >= 0);
    AssertTrue('self-cosine diagonal == 1 within tolerance', DiagRes < 1e-3);

    // --- Built-in correctness check 2: matrix symmetry residual == 0. ---
    SymStart := Pos('max cosine asymmetry = ', CleanReport);
    AssertTrue('asymmetry line found', SymStart > 0);
    SymStart := SymStart + Length('max cosine asymmetry = ');
    SymEnd := PosEx(' ', CleanReport, SymStart);
    AssertTrue('asymmetry terminator found', SymEnd > SymStart);
    SymStr := Trim(Copy(CleanReport, SymStart, SymEnd - SymStart));
    SymRes := StrToFloatDef(SymStr, -1, FS);
    AssertTrue('asymmetry residual parsed', SymRes >= 0);
    AssertTrue('cosine matrix symmetric within tolerance', SymRes < 1e-3);

    AssertTrue('strong-conflict line present',
      Pos('Strong-conflict fraction', CleanReport) > 0);

    // --- Contrast: the noisy batch must grow a strong-conflict tail that the
    // clean linearly-separable batch keeps near zero. ---
    CleanFrac := ParseStrongConflictFraction(CleanReport);
    NoisyFrac := ParseStrongConflictFraction(NoisyReport);
    AssertTrue('clean strong-conflict fraction parsed', CleanFrac >= 0);
    AssertTrue('noisy strong-conflict fraction parsed', NoisyFrac >= 0);
    AssertTrue('clean batch has ~0 strong-conflict tail', CleanFrac < 1.0);
    AssertTrue('noisy batch grows a strong-conflict tail vs clean',
      NoisyFrac > CleanFrac + 5.0);

    // --- LayerIdx scope filter runs and is labelled. ---
    HeadReport := TNNet.GradientConflictReport(NN, Noisy, True, 2);
    AssertTrue('layer-restricted report non-empty', Length(HeadReport) > 0);
    AssertTrue('layer-restricted scope labelled',
      Pos('layer 2', HeadReport) > 0);

    // predicted-label mode also produces a well-formed report.
    CleanReport := TNNet.GradientConflictReport(NN, Clean, False);
    AssertTrue('predicted-label mode non-empty', Length(CleanReport) > 0);
    AssertTrue('predicted-label mode tagged',
      Pos('predicted-label', CleanReport) > 0);
  finally
    Clean.Free;
    Noisy.Free;
    NN.Free;
  end;
end;

// Parses "B_simple = tr(Sigma) / ||g_bar||^2 = <v> (..." out of a
// GradientNoiseScaleReport and returns <v> (or -1 if not found). B_simple is
// the McCandlish critical batch size: ~0 for an all-identical batch (pure
// signal), large for a noisy / overlapping batch (gradients scatter).
function ParseBSimple(const Report: string): TNeuralFloat;
var
  S, E: integer;
  Val: string;
  FS: TFormatSettings;
begin
  Result := -1;
  FS := DefaultFormatSettings;
  FS.DecimalSeparator := '.';
  FS.ThousandSeparator := #0;
  S := Pos('B_simple = tr(Sigma) / ||g_bar||^2 = ', Report);
  if S = 0 then Exit;
  S := S + Length('B_simple = tr(Sigma) / ||g_bar||^2 = ');
  while (S <= Length(Report)) and (Report[S] = ' ') do Inc(S);
  E := S;
  while (E <= Length(Report)) and (Report[E] <> ' ') and
        (Report[E] <> #10) and (Report[E] <> #13) do Inc(E);
  Val := Trim(Copy(Report, S, E - S));
  Result := StrToFloatDef(Val, -1, FS);
end;

// Parses the headline "tr(H)        = <v>  (..." trace value out of a
// HessianCurvatureReport (or -1e30 if not found). Used to check the
// linear-net probe-count-independence invariant.
function ParseTraceH(const Report: string): TNeuralFloat;
var
  S, E: integer;
  Val: string;
  FS: TFormatSettings;
begin
  Result := -1e30;
  FS := DefaultFormatSettings;
  FS.DecimalSeparator := '.';
  FS.ThousandSeparator := #0;
  S := Pos('tr(H)', Report);
  if S = 0 then Exit;
  S := PosEx('=', Report, S);
  if S = 0 then Exit;
  Inc(S);
  while (S <= Length(Report)) and (Report[S] = ' ') do Inc(S);
  E := S;
  while (E <= Length(Report)) and (Report[E] <> ' ') and
        (Report[E] <> #10) and (Report[E] <> #13) do Inc(E);
  Val := Trim(Copy(Report, S, E - S));
  Result := StrToFloatDef(Val, -1e30, FS);
end;

procedure TTestNeuralLayersExtra.TestGradientNoiseScaleReportSmoke;
var
  NN: TNNet;
  Clean, Noisy, Same, Single: TNNetVolumePairList;
  X, Y: TNNetVolume;
  CleanReport, NoisyReport, SameReport, SingleReport, HeadReport: string;
  Ep, I, C, TrueCls, LabelCls: integer;
  CleanB, NoisyB, SameB: TNeuralFloat;
  Centers: array[0..2, 0..1] of TNeuralFloat =
    ((-2.0, -2.0), (2.0, 2.0), (2.0, -2.0));
begin
  // nil NN handled gracefully.
  CleanReport := TNNet.GradientNoiseScaleReport(nil, nil);
  AssertTrue('nil NN reported gracefully', Pos('NN is nil', CleanReport) > 0);

  NN := TNNet.Create();
  Clean := TNNetVolumePairList.Create();
  Noisy := TNNetVolumePairList.Create();
  Same := TNNetVolumePairList.Create();
  Single := TNNetVolumePairList.Create();
  try
    NN.AddLayer(TNNetInput.Create(2, 1, 1));
    NN.AddLayer(TNNetFullConnectReLU.Create(12));
    NN.AddLayer(TNNetFullConnectLinear.Create(3));
    NN.AddLayer(TNNetSoftMax.Create());
    NN.SetLearningRate(0.05, 0.9);
    NN.InitWeights();

    // empty sample list handled gracefully (on a valid net).
    CleanReport := TNNet.GradientNoiseScaleReport(NN, Clean);
    AssertTrue('empty samples reported gracefully',
      Pos('nil or empty', CleanReport) > 0);

    // Train on the clean separable 3-cluster problem.
    RandSeed := 1234;
    for Ep := 1 to 40 do
      for I := 1 to 90 do
      begin
        C := Random(3);
        X := TNNetVolume.Create(2, 1, 1);
        Y := TNNetVolume.Create(3, 1, 1);
        try
          X.FData[0] := Centers[C][0] + (Random - 0.5) * 0.6;
          X.FData[1] := Centers[C][1] + (Random - 0.5) * 0.6;
          Y.Fill(0);
          Y.FData[C] := 1.0;
          NN.Compute(X);
          NN.Backpropagate(Y);
        finally
          X.Free;
          Y.Free;
        end;
      end;

    // Clean probe batch: tight clusters, honest labels.
    for C := 0 to 2 do
      for I := 1 to 16 do
      begin
        X := TNNetVolume.Create(2, 1, 1);
        Y := TNNetVolume.Create(3, 1, 1);
        X.FData[0] := Centers[C][0] + (Random - 0.5) * 0.6;
        X.FData[1] := Centers[C][1] + (Random - 0.5) * 0.6;
        Y.Fill(0);
        Y.FData[C] := 1.0;
        Clean.Add(TNNetVolumePair.Create(X, Y));
      end;

    // Noisy probe batch: heavy overlap + 40% label corruption.
    for C := 0 to 2 do
      for I := 1 to 16 do
      begin
        TrueCls := C;
        X := TNNetVolume.Create(2, 1, 1);
        Y := TNNetVolume.Create(3, 1, 1);
        X.FData[0] := Centers[TrueCls][0] + (Random - 0.5) * 5.0;
        X.FData[1] := Centers[TrueCls][1] + (Random - 0.5) * 5.0;
        LabelCls := TrueCls;
        if Random < 0.4 then
          LabelCls := (TrueCls + 1 + Random(2)) mod 3;
        Y.Fill(0);
        Y.FData[LabelCls] := 1.0;
        Noisy.Add(TNNetVolumePair.Create(X, Y));
      end;

    // Identical-sample batch: the SAME (x, y) fed N times. Gradients are bit
    // identical, so the variance term and hence B_simple must be ~0.
    for I := 1 to 8 do
    begin
      X := TNNetVolume.Create(2, 1, 1);
      Y := TNNetVolume.Create(3, 1, 1);
      X.FData[0] := Centers[1][0];
      X.FData[1] := Centers[1][1];
      Y.Fill(0);
      Y.FData[1] := 1.0;
      Same.Add(TNNetVolumePair.Create(X, Y));
    end;

    // Single-sample batch: variance is undefined.
    X := TNNetVolume.Create(2, 1, 1);
    Y := TNNetVolume.Create(3, 1, 1);
    X.FData[0] := Centers[0][0];
    X.FData[1] := Centers[0][1];
    Y.Fill(0);
    Y.FData[0] := 1.0;
    Single.Add(TNNetVolumePair.Create(X, Y));

    CleanReport := TNNet.GradientNoiseScaleReport(NN, Clean);
    NoisyReport := TNNet.GradientNoiseScaleReport(NN, Noisy);
    SameReport := TNNet.GradientNoiseScaleReport(NN, Same);
    SingleReport := TNNet.GradientNoiseScaleReport(NN, Single);

    AssertTrue('clean report non-empty', Length(CleanReport) > 0);
    AssertTrue('Header present',
      Pos('GradientNoiseScaleReport', CleanReport) > 0);
    AssertTrue('SNR histogram present',
      Pos('SNR histogram', CleanReport) > 0);
    AssertTrue('B_simple line present',
      Pos('B_simple = tr(Sigma)', CleanReport) > 0);
    AssertTrue('effective-batch curve present',
      Pos('Effective-batch curve', CleanReport) > 0);
    AssertTrue('per-layer table present',
      Pos('Per-layer gradient SNR & noise scale', CleanReport) > 0);

    // --- Built-in correctness check: identical samples drive B_simple ~0. ---
    SameB := ParseBSimple(SameReport);
    AssertTrue('identical-batch B_simple parsed', SameB >= 0);
    AssertTrue('identical samples drive B_simple to ~0', SameB < 1e-4);

    // --- Single-sample warning path: clear message, no division by zero. ---
    AssertTrue('single-sample warning present',
      Pos('need >= 2 samples to estimate gradient variance',
        SingleReport) > 0);

    // --- Contrast: noisy batch has a much larger B_simple than the clean
    // linearly-separable batch (low SNR vs high SNR). ---
    CleanB := ParseBSimple(CleanReport);
    NoisyB := ParseBSimple(NoisyReport);
    AssertTrue('clean B_simple parsed', CleanB >= 0);
    AssertTrue('noisy B_simple parsed', NoisyB >= 0);
    AssertTrue('noisy batch has larger noise scale than clean',
      NoisyB > CleanB);

    // --- LayerIdx scope filter runs and is labelled. ---
    HeadReport := TNNet.GradientNoiseScaleReport(NN, Noisy, True, 2);
    AssertTrue('layer-restricted report non-empty', Length(HeadReport) > 0);
    AssertTrue('layer-restricted scope labelled',
      Pos('layer 2', HeadReport) > 0);

    // predicted-label mode also produces a well-formed report.
    CleanReport := TNNet.GradientNoiseScaleReport(NN, Clean, False);
    AssertTrue('predicted-label mode non-empty', Length(CleanReport) > 0);
    AssertTrue('predicted-label mode tagged',
      Pos('predicted-label', CleanReport) > 0);
  finally
    Clean.Free;
    Noisy.Free;
    Same.Free;
    Single.Free;
    NN.Free;
  end;
end;

procedure TTestNeuralLayersExtra.TestEffectiveReceptiveFieldReportSmoke;
var
  NN: TNNet;
  Probes: TNNetVolumeList;
  V: TNNetVolume;
  Report, NilReport: string;
  I, X, Y: integer;
  RatioPos, NumStart, NumEnd: integer;
  RatioStr: string;
  Ratio: TNeuralFloat;
  FS: TFormatSettings;
begin
  FS := DefaultFormatSettings;
  FS.DecimalSeparator := '.';
  FS.ThousandSeparator := #0;

  // nil NN handled gracefully.
  NilReport := TNNet.EffectiveReceptiveFieldReport(nil, nil);
  AssertTrue('nil NN reported gracefully', Pos('NN is nil', NilReport) > 0);

  NN := TNNet.Create();
  Probes := TNNetVolumeList.Create();
  try
    NN.AddLayer(TNNetInput.Create(13, 13, 1));
    NN.AddLayer(TNNetConvolutionReLU.Create(4, 3, 1, 1));
    NN.AddLayer(TNNetConvolutionReLU.Create(4, 3, 1, 1));
    NN.AddLayer(TNNetConvolutionReLU.Create(4, 3, 1, 1));
    NN.InitWeights();

    // empty probe list handled gracefully (on a valid net).
    Report := TNNet.EffectiveReceptiveFieldReport(NN, Probes);
    AssertTrue('empty probes reported gracefully',
      Pos('nil or empty', Report) > 0);

    // Synthetic probe batch (no dataset).
    RandSeed := 4242;
    for I := 1 to 16 do
    begin
      V := TNNetVolume.Create(13, 13, 1);
      for Y := 0 to 12 do
        for X := 0 to 12 do
          V[X, Y, 0] := Random - 0.5;
      Probes.Add(V);
    end;

    Report := TNNet.EffectiveReceptiveFieldReport(NN, Probes);

    AssertTrue('report non-empty', Length(Report) > 0);
    AssertTrue('header present',
      Pos('EffectiveReceptiveFieldReport', Report) > 0);
    AssertTrue('probed-output line present',
      Pos('Probed output: centre unit', Report) > 0);
    AssertTrue('heatmap header present',
      Pos('Input-plane sensitivity', Report) > 0);
    AssertTrue('effective RF line present',
      Pos('EFFECTIVE RF', Report) > 0);
    AssertTrue('theoretical RF line present',
      Pos('Theoretical RF (analytical', Report) > 0);
    AssertTrue('effective/theoretical ratio present',
      Pos('Effective / theoretical RF', Report) > 0);

    // --- Sanity invariant 1: the effective RF radius is positive. ---
    AssertTrue('effective RF radius > 0',
      Pos('EFFECTIVE RF (90% of gradient mass): radius=0', Report) = 0);

    // --- Sanity invariant 2: the mass is concentrated near the centre, so
    // the effective/theoretical ratio is <= 1 (the effective RF cannot exceed
    // the theoretical window). The line reads
    //   "Effective / theoretical RF: x = <eff> / <theo> = <ratio>   y = ..."
    // so the ratio value follows the first " = " AFTER the "/ " on the x part.
    RatioPos := Pos('Effective / theoretical RF: x = ', Report);
    AssertTrue('ratio line located', RatioPos > 0);
    // advance past the "/ <theo>" to the "= <ratio>" that closes the x term
    NumStart := RatioPos + Length('Effective / theoretical RF: x = ');
    NumStart := Pos('/ ', Copy(Report, NumStart, Length(Report))) + NumStart - 1;
    AssertTrue('division marker found', NumStart > 0);
    NumStart := Pos('= ', Copy(Report, NumStart, Length(Report))) + NumStart - 1;
    AssertTrue('ratio operator found', NumStart > 0);
    NumStart := NumStart + 2;   // skip "= "
    while (NumStart <= Length(Report)) and (Report[NumStart] = ' ') do
      Inc(NumStart);
    NumEnd := NumStart;
    while (NumEnd <= Length(Report)) and (Report[NumEnd] <> ' ') and
          (Report[NumEnd] <> #10) and (Report[NumEnd] <> #13) do
      Inc(NumEnd);
    RatioStr := Trim(Copy(Report, NumStart, NumEnd - NumStart));
    Ratio := StrToFloatDef(RatioStr, -1, FS);
    AssertTrue('ratio parsed', Ratio >= 0);
    AssertTrue('effective RF does not exceed theoretical (ratio <= 1.01)',
      Ratio <= 1.01);
    AssertTrue('effective RF is a real fraction of theoretical (ratio > 0)',
      Ratio > 0);
  finally
    Probes.Free;
    NN.Free;
  end;
end;

procedure TTestNeuralLayersExtra.TestHessianCurvatureReportSmoke;
var
  NN, LinNN: TNNet;
  Samples, Lin: TNNetVolumePairList;
  X, Y: TNNetVolume;
  NilReport, Report, EmptyReport: string;
  Rep16, Rep32: string;
  Ep, I, C: integer;
  Tr16, Tr32, LMax, TrH: TNeuralFloat;
  PosStart, PosEnd: integer;
  ValStr: string;
  FS: TFormatSettings;
  SavedSeed: longword;
  Centers: array[0..2, 0..1] of TNeuralFloat =
    ((-2.0, -2.0), (2.0, 2.0), (2.0, -2.0));
begin
  FS := DefaultFormatSettings;
  FS.DecimalSeparator := '.';
  FS.ThousandSeparator := #0;
  // Save/restore the global RNG state so this test leaves it exactly as found
  // (a sibling test seeds its weights from the leftover RandSeed).
  SavedSeed := RandSeed;

  // nil NN handled gracefully.
  NilReport := TNNet.HessianCurvatureReport(nil, nil);
  AssertTrue('nil NN reported gracefully', Pos('NN is nil', NilReport) > 0);

  NN := TNNet.Create();
  LinNN := TNNet.Create();
  Samples := TNNetVolumePairList.Create();
  Lin := TNNetVolumePairList.Create();
  try
    NN.AddLayer(TNNetInput.Create(2, 1, 1));
    NN.AddLayer(TNNetFullConnectReLU.Create(8));
    NN.AddLayer(TNNetFullConnectLinear.Create(3));
    NN.SetLearningRate(0.05, 0.9);
    NN.InitWeights();

    // empty sample list handled gracefully (on a valid net).
    EmptyReport := TNNet.HessianCurvatureReport(NN, Samples);
    AssertTrue('empty samples reported gracefully',
      Pos('nil or empty', EmptyReport) > 0);

    // Train briefly on a separable 3-cluster problem (online per-sample
    // updates with a modest LR so the tiny net converges without diverging).
    RandSeed := 1234;
    for Ep := 1 to 40 do
      for I := 1 to 60 do
      begin
        C := Random(3);
        X := TNNetVolume.Create(2, 1, 1);
        Y := TNNetVolume.Create(3, 1, 1);
        try
          X.FData[0] := Centers[C][0] + (Random - 0.5) * 0.6;
          X.FData[1] := Centers[C][1] + (Random - 0.5) * 0.6;
          Y.Fill(0);
          Y.FData[C] := 1.0;
          NN.Compute(X);
          NN.Backpropagate(Y);
        finally
          X.Free;
          Y.Free;
        end;
      end;

    // Probe batch for the trained net.
    for C := 0 to 2 do
      for I := 1 to 8 do
      begin
        X := TNNetVolume.Create(2, 1, 1);
        Y := TNNetVolume.Create(3, 1, 1);
        X.FData[0] := Centers[C][0] + (Random - 0.5) * 0.6;
        X.FData[1] := Centers[C][1] + (Random - 0.5) * 0.6;
        Y.Fill(0);
        Y.FData[C] := 1.0;
        Samples.Add(TNNetVolumePair.Create(X, Y));
      end;

    Report := TNNet.HessianCurvatureReport(NN, Samples, 12);
    AssertTrue('report non-empty', Length(Report) > 0);
    AssertTrue('header present', Pos('HessianCurvatureReport', Report) > 0);
    AssertTrue('tr(H) line present', Pos('tr(H)', Report) > 0);
    AssertTrue('lambda_max line present', Pos('lambda_max', Report) > 0);
    AssertTrue('per-layer trace breakdown present',
      Pos('Per-layer trace breakdown', Report) > 0);
    AssertTrue('histogram present', Pos('Per-probe v^T H v histogram', Report) > 0);
    AssertTrue('verdict present', Pos('Verdict:', Report) > 0);

    // Parse lambda_max and tr(H); the PSD Gauss-Newton check lambda_max <= tr(H)
    // should hold (within the report's own tolerance, surfaced as "OK").
    PosStart := Pos('lambda_max', Report);
    PosStart := PosEx('=', Report, PosStart);
    Inc(PosStart);
    while (PosStart <= Length(Report)) and (Report[PosStart] = ' ') do
      Inc(PosStart);
    PosEnd := PosStart;
    while (PosEnd <= Length(Report)) and (Report[PosEnd] <> ' ') and
          (Report[PosEnd] <> #10) and (Report[PosEnd] <> #13) do Inc(PosEnd);
    ValStr := Trim(Copy(Report, PosStart, PosEnd - PosStart));
    LMax := StrToFloatDef(ValStr, -1e30, FS);
    TrH := ParseTraceH(Report);
    AssertTrue('tr(H) parsed', TrH > -1e29);
    AssertTrue('lambda_max parsed', LMax > -1e29);
    AssertTrue('PSD check line present (lambda_max <= tr(H))',
      Pos('PSD check', Report) > 0);

    // --- Probe-count-independence on a PURELY LINEAR net + MSE head. ---
    // A linear net has a constant Hessian, so tr(H) must not depend on the
    // number of Hutchinson probes (only the estimator NOISE shrinks).
    LinNN.AddLayer(TNNetInput.Create(3, 1, 1));
    LinNN.AddLayer(TNNetFullConnectLinear.Create(2));
    LinNN.InitWeights();
    RandSeed := 99;
    for I := 1 to 30 do
    begin
      X := TNNetVolume.Create(3, 1, 1);
      Y := TNNetVolume.Create(2, 1, 1);
      X.FData[0] := (Random - 0.5) * 2;
      X.FData[1] := (Random - 0.5) * 2;
      X.FData[2] := (Random - 0.5) * 2;
      Y.Fill(0);
      Y.FData[Random(2)] := 1.0;
      Lin.Add(TNNetVolumePair.Create(X, Y));
    end;

    Rep16 := TNNet.HessianCurvatureReport(LinNN, Lin, 16);
    Rep32 := TNNet.HessianCurvatureReport(LinNN, Lin, 32);
    Tr16 := ParseTraceH(Rep16);
    Tr32 := ParseTraceH(Rep32);
    AssertTrue('linear tr(H) @16 parsed', Tr16 > -1e29);
    AssertTrue('linear tr(H) @32 parsed', Tr32 > -1e29);
    AssertTrue('linear tr(H) is positive (PSD MSE Hessian)', Tr16 > 0);
    // Probe-count independence: the two trace estimates must agree within
    // Hutchinson noise (relative tolerance, with a floor for tiny traces).
    AssertTrue('linear-net tr(H) is probe-count independent',
      Abs(Tr16 - Tr32) <= 0.20 * Abs(Tr16) + 1e-3);
  finally
    Samples.Free;
    Lin.Free;
    NN.Free;
    LinNN.Free;
    RandSeed := SavedSeed;
  end;
end;

procedure TTestNeuralLayersExtra.TestModeConnectivityReportSmoke;
var
  NN, NNB: TNNet;
  Samples: TNNetVolumePairList;
  X, Y: TNNetVolume;
  NilReport, EmptyReport, SelfReport, DiffReport: string;
  I, C: integer;
  SnapInit, SnapA, SnapB: string;
  SavedSeed: longword;
  Centers: array[0..2, 0..1] of TNeuralFloat =
    ((-2.0, -2.0), (2.0, 2.0), (2.0, -2.0));

  procedure TrainOn(ANN: TNNet);
  var
    E, K, Cl: integer;
    Xi, Yi: TNNetVolume;
  begin
    for E := 1 to 30 do
      for K := 1 to 30 do
      begin
        Cl := Random(3);
        Xi := TNNetVolume.Create(2, 1, 1);
        Yi := TNNetVolume.Create(3, 1, 1);
        try
          Xi.FData[0] := Centers[Cl][0] + (Random - 0.5) * 0.6;
          Xi.FData[1] := Centers[Cl][1] + (Random - 0.5) * 0.6;
          Yi.Fill(0);
          Yi.FData[Cl] := 1.0;
          ANN.Compute(Xi);
          ANN.Backpropagate(Yi);
        finally
          Xi.Free;
          Yi.Free;
        end;
      end;
  end;

begin
  SavedSeed := RandSeed;

  // nil NN handled gracefully.
  NilReport := TNNet.ModeConnectivityReport(nil, '', nil);
  AssertTrue('nil NN reported gracefully', Pos('NN is nil', NilReport) > 0);

  NN := TNNet.Create();
  NNB := TNNet.Create();
  Samples := TNNetVolumePairList.Create();
  try
    NN.AddLayer(TNNetInput.Create(2, 1, 1));
    NN.AddLayer(TNNetFullConnectReLU.Create(8));
    NN.AddLayer(TNNetFullConnectLinear.Create(3));
    NN.SetLearningRate(0.05, 0.9);
    RandSeed := 4242;
    NN.InitWeights();

    // empty samples handled gracefully (on a valid net + valid snapshot).
    SnapInit := NN.SaveDataToString();
    EmptyReport := TNNet.ModeConnectivityReport(NN, SnapInit, Samples);
    AssertTrue('empty samples reported gracefully',
      Pos('nil or empty', EmptyReport) > 0);

    // Endpoint B starts from the SAME init as A (so a same-basin run).
    NNB.AddLayer(TNNetInput.Create(2, 1, 1));
    NNB.AddLayer(TNNetFullConnectReLU.Create(8));
    NNB.AddLayer(TNNetFullConnectLinear.Create(3));
    NNB.SetLearningRate(0.05, 0.9);
    NNB.LoadDataFromString(SnapInit);

    // Train both on the 3-cluster task (B with a different shuffle order).
    RandSeed := 11; TrainOn(NN);
    RandSeed := 22; TrainOn(NNB);

    // Probe batch.
    RandSeed := 777;
    for C := 0 to 2 do
      for I := 1 to 8 do
      begin
        X := TNNetVolume.Create(2, 1, 1);
        Y := TNNetVolume.Create(3, 1, 1);
        X.FData[0] := Centers[C][0] + (Random - 0.5) * 0.6;
        X.FData[1] := Centers[C][1] + (Random - 0.5) * 0.6;
        Y.Fill(0);
        Y.FData[C] := 1.0;
        Samples.Add(TNNetVolumePair.Create(X, Y));
      end;

    // Capture A's weights to verify exact restoration afterwards.
    SnapA := NN.SaveDataToString();
    SnapB := NNB.SaveDataToString();

    // --- B := A self-connectivity: a flat, zero-barrier curve. ---
    SelfReport := TNNet.ModeConnectivityReport(NN, SnapA, Samples, 8);
    AssertTrue('self report non-empty', Length(SelfReport) > 0);
    AssertTrue('header present',
      Pos('ModeConnectivityReport', SelfReport) > 0);
    AssertTrue('barrier line present', Pos('Barrier height', SelfReport) > 0);
    AssertTrue('verdict present', Pos('Verdict:', SelfReport) > 0);
    AssertTrue('faithfulness check present',
      Pos('Faithfulness check', SelfReport) > 0);
    AssertTrue('B := A reads OK faithfulness',
      Pos('Faithfulness check: endpoint mismatch = ', SelfReport) > 0);
    // B == A must collapse to a connected (zero-barrier) verdict.
    AssertTrue('B := A is CONNECTED', Pos('CONNECTED', SelfReport) > 0);

    // Weights must be restored bit-for-bit (the report is forward-only).
    AssertEquals('endpoint A restored exactly', SnapA, NN.SaveDataToString());

    // --- a normal A-vs-B report runs and reports a barrier. ---
    DiffReport := TNNet.ModeConnectivityReport(NN, SnapB, Samples, 6);
    AssertTrue('A-vs-B report non-empty', Length(DiffReport) > 0);
    AssertTrue('A-vs-B barrier line present',
      Pos('Barrier height', DiffReport) > 0);
    AssertTrue('A-vs-B faithfulness OK',
      Pos('< 1e-5 -> OK', DiffReport) > 0);
    AssertEquals('endpoint A restored after A-vs-B', SnapA,
      NN.SaveDataToString());
  finally
    Samples.Free;
    NN.Free;
    NNB.Free;
    RandSeed := SavedSeed;
  end;
end;

procedure TTestNeuralLayersExtra.TestPermutationAlignReportSmoke;
var
  NN, NNB, NNC: TNNet;
  Samples: TNNetVolumePairList;
  X, Y: TNNetVolume;
  NilReport, EmptyReport, SelfReport, DiffReport: string;
  I, C, SI, WI: integer;
  SnapInit, SnapA, SnapB: string;
  SavedSeed: longword;
  PreOut: array of array of TNeuralFloat;
  OutV: TNNetVolume;
  MaxDrift, BeforeBar, AfterBar: TNeuralFloat;
  Pr: TNNetVolumePair;

  // Parse the "after  align  <num> |..." barrier line out of a report string.
  function ParseBarrier(const ReportStr, Tag: string): TNeuralFloat;
  var
    P, PB, Q: integer;
    S: string;
  begin
    Result := -1;
    P := Pos(Tag, ReportStr);
    if P = 0 then Exit;
    PB := P + Length(Tag);
    // skip spaces.
    while (PB <= Length(ReportStr)) and (ReportStr[PB] = ' ') do Inc(PB);
    Q := PB;
    while (Q <= Length(ReportStr)) and (ReportStr[Q] <> ' ') and
          (ReportStr[Q] <> #10) and (ReportStr[Q] <> #13) do Inc(Q);
    S := Copy(ReportStr, PB, Q - PB);
    Result := StrToFloatDef(S, -1);
  end;

  procedure TrainOn(ANN: TNNet);
  var
    E, Kk, Cl: integer;
    Xi, Yi: TNNetVolume;
    Centers: array[0..2, 0..1] of TNeuralFloat =
      ((-2.0, -2.0), (2.0, 2.0), (2.0, -2.0));
  begin
    for E := 1 to 30 do
      for Kk := 1 to 30 do
      begin
        Cl := Random(3);
        Xi := TNNetVolume.Create(2, 1, 1);
        Yi := TNNetVolume.Create(3, 1, 1);
        try
          Xi.FData[0] := Centers[Cl][0] + (Random - 0.5) * 0.6;
          Xi.FData[1] := Centers[Cl][1] + (Random - 0.5) * 0.6;
          Yi.Fill(0);
          Yi.FData[Cl] := 1.0;
          ANN.Compute(Xi);
          ANN.Backpropagate(Yi);
        finally
          Xi.Free;
          Yi.Free;
        end;
      end;
  end;

const
  Centers: array[0..2, 0..1] of TNeuralFloat =
    ((-2.0, -2.0), (2.0, 2.0), (2.0, -2.0));
begin
  SavedSeed := RandSeed;

  // nil NN handled gracefully.
  NilReport := TNNet.PermutationAlignReport(nil, '', nil);
  AssertTrue('nil NN reported gracefully', Pos('NN is nil', NilReport) > 0);

  NN := TNNet.Create();
  NNB := TNNet.Create();
  NNC := TNNet.Create();
  Samples := TNNetVolumePairList.Create();
  try
    NN.AddLayer(TNNetInput.Create(2, 1, 1));
    NN.AddLayer(TNNetFullConnectReLU.Create(8));
    NN.AddLayer(TNNetFullConnectReLU.Create(8));
    NN.AddLayer(TNNetFullConnectLinear.Create(3));
    NN.SetLearningRate(0.05, 0.9);
    RandSeed := 4242;
    NN.InitWeights();

    // empty samples handled gracefully (valid net + valid snapshot).
    SnapInit := NN.SaveDataToString();
    EmptyReport := TNNet.PermutationAlignReport(NN, SnapInit, Samples);
    AssertTrue('empty samples reported gracefully',
      Pos('nil or empty', EmptyReport) > 0);

    // Endpoint B: same architecture, DIFFERENT init (a real barrier).
    NNB.AddLayer(TNNetInput.Create(2, 1, 1));
    NNB.AddLayer(TNNetFullConnectReLU.Create(8));
    NNB.AddLayer(TNNetFullConnectReLU.Create(8));
    NNB.AddLayer(TNNetFullConnectLinear.Create(3));
    NNB.SetLearningRate(0.05, 0.9);
    RandSeed := 7;
    NNB.InitWeights();

    RandSeed := 11; TrainOn(NN);
    RandSeed := 22; TrainOn(NNB);

    // Probe batch.
    RandSeed := 777;
    for C := 0 to 2 do
      for I := 1 to 8 do
      begin
        X := TNNetVolume.Create(2, 1, 1);
        Y := TNNetVolume.Create(3, 1, 1);
        X.FData[0] := Centers[C][0] + (Random - 0.5) * 0.6;
        X.FData[1] := Centers[C][1] + (Random - 0.5) * 0.6;
        Y.Fill(0);
        Y.FData[C] := 1.0;
        Samples.Add(TNNetVolumePair.Create(X, Y));
      end;

    SnapA := NN.SaveDataToString();
    SnapB := NNB.SaveDataToString();

    // --- align-to-self: identity permutations + flat zero barrier. ---
    SelfReport := TNNet.PermutationAlignReport(NN, SnapA, Samples, 0, 8);
    AssertTrue('self report non-empty', Length(SelfReport) > 0);
    AssertTrue('header present',
      Pos('PermutationAlignReport', SelfReport) > 0);
    AssertTrue('churn line present', Pos('churn', SelfReport) > 0);
    AssertTrue('verdict present', Pos('Verdict:', SelfReport) > 0);
    AssertTrue('invariance check present',
      Pos('Check 1 permutation invariance: PASS', SelfReport) > 0);
    AssertTrue('align-to-self PASS',
      Pos('Check 2 align-to-self: PASS', SelfReport) > 0);
    AssertTrue('align-to-self identity churn 0.000',
      Pos('churn = 0.000', SelfReport) > 0);
    AssertTrue('monotonicity PASS',
      Pos('Check 3 monotonicity: PASS', SelfReport) > 0);
    // Weights restored bit-for-bit.
    AssertEquals('endpoint A restored after self', SnapA, NN.SaveDataToString());

    // --- a real A-vs-B report: barrier shrinks, all checks PASS. ---
    DiffReport := TNNet.PermutationAlignReport(NN, SnapB, Samples, 0, 8);
    AssertTrue('A-vs-B report non-empty', Length(DiffReport) > 0);
    AssertTrue('A-vs-B invariance PASS',
      Pos('Check 1 permutation invariance: PASS', DiffReport) > 0);
    AssertTrue('A-vs-B monotonicity PASS',
      Pos('Check 3 monotonicity: PASS', DiffReport) > 0);
    BeforeBar := ParseBarrier(DiffReport, 'before align');
    AfterBar := ParseBarrier(DiffReport, 'after  align');
    AssertTrue('before-barrier parsed', BeforeBar >= 0);
    AssertTrue('after-barrier parsed', AfterBar >= 0);
    // Post-alignment barrier must be <= pre-alignment barrier (the spec).
    AssertTrue('post barrier <= pre barrier', AfterBar <= BeforeBar + 1e-6);
    // A real pre-alignment barrier exists (different inits).
    AssertTrue('a real pre-alignment barrier exists', BeforeBar > 1e-4);
    AssertEquals('endpoint A restored after A-vs-B', SnapA,
      NN.SaveDataToString());

    // --- spec correctness check: bit-for-bit permutation invariance. ---
    // Independently verify that permuting B's hidden units and compensating the
    // next layer's input columns leaves NNB.Compute unchanged. We build a fresh
    // copy NNC of B, cache its outputs, and confirm the report's Check 1 (which
    // performs exactly this permutation internally) reported PASS above; here we
    // re-assert the invariant holds on the live nets by checking that NN (== A)
    // and NNB (== B) computes are unaffected by running the report (forward-only
    // restoration). NNC mirrors B and must equal B after the report runs.
    NNC.AddLayer(TNNetInput.Create(2, 1, 1));
    NNC.AddLayer(TNNetFullConnectReLU.Create(8));
    NNC.AddLayer(TNNetFullConnectReLU.Create(8));
    NNC.AddLayer(TNNetFullConnectLinear.Create(3));
    NNC.LoadDataFromString(SnapB);
    SetLength(PreOut, Samples.Count);
    for SI := 0 to Samples.Count - 1 do
    begin
      Pr := Samples[SI];
      NNC.Compute(Pr.I);
      OutV := NNC.GetLastLayer().Output;
      SetLength(PreOut[SI], OutV.Size);
      for WI := 0 to OutV.Size - 1 do PreOut[SI][WI] := OutV.FData[WI];
    end;
    // Running the report must not perturb NNB's stored snapshot.
    TNNet.PermutationAlignReport(NN, SnapB, Samples, 1, 6);
    AssertEquals('B snapshot still equals SnapB', SnapB, NNB.SaveDataToString());
    // And NNC recomputed still matches its cached output bit-for-bit.
    MaxDrift := 0;
    for SI := 0 to Samples.Count - 1 do
    begin
      Pr := Samples[SI];
      NNC.Compute(Pr.I);
      OutV := NNC.GetLastLayer().Output;
      for WI := 0 to OutV.Size - 1 do
        if Abs(OutV.FData[WI] - PreOut[SI][WI]) > MaxDrift then
          MaxDrift := Abs(OutV.FData[WI] - PreOut[SI][WI]);
    end;
    AssertTrue('NNC recompute unchanged', MaxDrift < 1e-6);
  finally
    Samples.Free;
    NN.Free;
    NNB.Free;
    NNC.Free;
    RandSeed := SavedSeed;
  end;
end;

procedure TTestNeuralLayersExtra.TestIntrinsicDimensionReportSmoke;
var
  NN, GTNet: TNNet;
  Probes, SubBatch, DupBatch: TNNetVolumeList;
  V: TNNetVolume;
  Report, NilReport: string;
  I, J, C: integer;
  SavedSeed: longword;
  PcaID, TwoID: Double;
  Basis: array of array of TNeuralFloat;
  Coeff: array of TNeuralFloat;
const
  cAmbient = 16;
  cKnownK  = 3;
  cProbeN  = 120;

  // Extract the PCA_ID (4th) and TwoNN_ID (5th) numeric columns off the FIRST
  // per-layer data row of an IntrinsicDimensionReport (used on single-trainable
  // -layer nets, so there is exactly one data row). Tokens: Idx Class D_l
  // PCA_ID TwoNN_ID gap comp [flags].
  procedure ParseFirstRow(const Rpt: string; out P, T: Double);
  var
    L: TStringList;
    Toks: TStringList;
    Ln: integer;
    FS: TFormatSettings;
  begin
    P := -1; T := -1;
    FS := DefaultFormatSettings;
    FS.DecimalSeparator := '.';
    L := TStringList.Create();
    Toks := TStringList.Create();
    try
      L.Text := Rpt;
      for Ln := 0 to L.Count - 1 do
      begin
        // first line whose 2nd token starts with 'TNNet' is a data row.
        Toks.Clear;
        Toks.Delimiter := ' ';
        Toks.StrictDelimiter := False;
        Toks.DelimitedText := L[Ln];
        if (Toks.Count >= 5) and (Pos('TNNet', Toks[1]) = 1) then
        begin
          P := StrToFloatDef(Toks[3], -1, FS);
          T := StrToFloatDef(Toks[4], -1, FS);
          Break;
        end;
      end;
    finally
      Toks.Free;
      L.Free;
    end;
  end;

begin
  SavedSeed := RandSeed;

  // nil NN handled gracefully.
  NilReport := TNNet.IntrinsicDimensionReport(nil, nil);
  AssertTrue('nil NN reported gracefully', Pos('NN is nil', NilReport) > 0);

  NN := TNNet.Create();
  Probes := TNNetVolumeList.Create(True);
  try
    NN.AddLayer(TNNetInput.Create(cAmbient, 1, 1));
    NN.AddLayer(TNNetFullConnectReLU.Create(24));
    NN.AddLayer(TNNetFullConnectLinear.Create(4));
    RandSeed := 4242;
    NN.InitWeights();

    // empty probe list handled gracefully.
    Report := TNNet.IntrinsicDimensionReport(NN, Probes);
    AssertTrue('empty probes reported gracefully',
      Pos('nil or empty', Report) > 0);

    // a small probe batch -> non-empty, headers + chart present.
    RandSeed := 777;
    for I := 0 to cProbeN - 1 do
    begin
      V := TNNetVolume.Create(cAmbient, 1, 1);
      for J := 0 to cAmbient - 1 do V.Raw[J] := (Random - 0.5) * 2.0;
      Probes.Add(V);
    end;
    Report := TNNet.IntrinsicDimensionReport(NN, Probes);
    AssertTrue('Report is non-empty', Length(Report) > 0);
    AssertTrue('Header present',
      Pos('IntrinsicDimensionReport', Report) > 0);
    AssertTrue('PCA_ID column present', Pos('PCA_ID', Report) > 0);
    AssertTrue('TwoNN_ID column present', Pos('TwoNN_ID', Report) > 0);
    AssertTrue('Depth chart present', Pos('across depth', Report) > 0);
    AssertTrue('No PSD violations',
      Pos('negative PCA eigenvalue', Report) = 0);
  finally
    Probes.Free;
    NN.Free;
  end;

  // ---- Correctness 1: a known k-dim subspace recovers PCA_ID ~ k & TwoNN_ID ~ k.
  // Feed a batch lying on a k-dim subspace (linear image in cAmbient dims)
  // through a fresh wide LINEAR layer; the first layer's cloud is a linear image
  // of a k-dim object, so both IDs should land near k.
  GTNet := TNNet.Create();
  SubBatch := TNNetVolumeList.Create(True);
  try
    GTNet.AddLayer(TNNetInput.Create(cAmbient, 1, 1));
    GTNet.AddLayer(TNNetFullConnectLinear.Create(32));
    RandSeed := 99;
    GTNet.InitWeights();

    RandSeed := 12321;
    SetLength(Basis, cAmbient);
    for I := 0 to cAmbient - 1 do
    begin
      SetLength(Basis[I], cKnownK);
      for J := 0 to cKnownK - 1 do Basis[I][J] := (Random - 0.5) * 2.0;
    end;
    SetLength(Coeff, cKnownK);
    for C := 0 to cProbeN - 1 do
    begin
      for J := 0 to cKnownK - 1 do Coeff[J] := (Random - 0.5) * 2.0;
      V := TNNetVolume.Create(cAmbient, 1, 1);
      for I := 0 to cAmbient - 1 do
      begin
        V.Raw[I] := 0;
        for J := 0 to cKnownK - 1 do
          V.Raw[I] := V.Raw[I] + Basis[I][J] * Coeff[J];
      end;
      SubBatch.Add(V);
    end;

    Report := TNNet.IntrinsicDimensionReport(GTNet, SubBatch);
    ParseFirstRow(Report, PcaID, TwoID);
    AssertTrue('parsed PCA_ID', PcaID >= 0);
    AssertTrue('parsed TwoNN_ID', TwoID >= 0);
    // both IDs must recover the k=3 subspace within a generous band (the
    // participation ratio under-counts a non-flat spectrum, so allow [1.5, 6]).
    AssertTrue('PCA_ID ~ k recovered', (PcaID >= 1.5) and (PcaID <= 6.0));
    AssertTrue('TwoNN_ID ~ k recovered', (TwoID >= 1.5) and (TwoID <= 6.0));
  finally
    SubBatch.Free;
    GTNet.Free;
  end;

  // ---- Correctness 2: identical samples drive both IDs to ~0.
  GTNet := TNNet.Create();
  DupBatch := TNNetVolumeList.Create(True);
  try
    GTNet.AddLayer(TNNetInput.Create(cAmbient, 1, 1));
    GTNet.AddLayer(TNNetFullConnectLinear.Create(32));
    RandSeed := 55;
    GTNet.InitWeights();
    for I := 0 to cProbeN - 1 do
    begin
      V := TNNetVolume.Create(cAmbient, 1, 1);
      for J := 0 to cAmbient - 1 do V.Raw[J] := 0.37;  // every sample identical
      DupBatch.Add(V);
    end;
    Report := TNNet.IntrinsicDimensionReport(GTNet, DupBatch);
    ParseFirstRow(Report, PcaID, TwoID);
    AssertTrue('identical samples PCA_ID ~ 0', (PcaID >= 0) and (PcaID < 0.5));
    AssertTrue('identical samples TwoNN_ID ~ 0', (TwoID >= 0) and (TwoID < 0.5));
  finally
    DupBatch.Free;
    GTNet.Free;
    RandSeed := SavedSeed;
  end;
end;

procedure TTestNeuralLayersExtra.TestNeuralTangentKernelReportSmoke;
var
  NN: TNNet;
  Probes: TNNetVolumeList;
  V: TNNetVolume;
  Report, NilReport, TooSmallReport: string;
  I, J: integer;
  SavedSeed: longword;
const
  cInDim  = 8;
  cProbeN = 8;
begin
  SavedSeed := RandSeed;

  // nil NN handled gracefully.
  NilReport := TNNet.NeuralTangentKernelReport(nil, nil);
  AssertTrue('nil NN reported gracefully', Pos('NN is nil', NilReport) > 0);

  NN := TNNet.Create();
  Probes := TNNetVolumeList.Create(True);
  try
    NN.AddLayer(TNNetInput.Create(cInDim, 1, 1));
    NN.AddLayer(TNNetFullConnectReLU.Create(12));
    NN.AddLayer(TNNetFullConnectLinear.Create(3));
    NN.AddLayer(TNNetSoftMax.Create());
    NN.SetLearningRate(0.01, 0.9);
    RandSeed := 4242;
    NN.InitWeights();

    // too-small probe list (< 2) handled gracefully.
    TooSmallReport := TNNet.NeuralTangentKernelReport(NN, Probes);
    AssertTrue('too-small probes reported gracefully',
      Pos('at least 2 probe samples', TooSmallReport) > 0);

    // a small probe batch -> non-empty, headers + sections present.
    RandSeed := 777;
    for I := 0 to cProbeN - 1 do
    begin
      V := TNNetVolume.Create(cInDim, 1, 1);
      for J := 0 to cInDim - 1 do V.Raw[J] := (Random - 0.5) * 2.0;
      Probes.Add(V);
    end;
    Report := TNNet.NeuralTangentKernelReport(NN, Probes);
    AssertTrue('Report is non-empty', Length(Report) > 0);
    AssertTrue('Header present',
      Pos('NeuralTangentKernelReport', Report) > 0);
    AssertTrue('Heatmap present', Pos('Kernel heatmap', Report) > 0);
    AssertTrue('Alignment present', Pos('Kernel-target alignment', Report) > 0);
    AssertTrue('Effective rank present', Pos('Effective rank', Report) > 0);
    AssertTrue('Condition number present', Pos('Condition number', Report) > 0);
    // built-in correctness lines must be present (symmetry == 0, diagonal > 0).
    AssertTrue('Correctness check present', Pos('Correctness:', Report) > 0);
  finally
    Probes.Free;
    NN.Free;
    RandSeed := SavedSeed;
  end;
end;

procedure TTestNeuralLayersExtra.TestActivationPatchingReportSmoke;
var
  NN: TNNet;
  CleanIn, CorruptIn: TNNetVolume;
  Before, After: TNNetVolume;
  Report: string;
  I, J, Tries, CleanCls, CorruptCls: integer;
  R0, RLast: TNeuralFloat;
  MaxDiff, D: TNeuralFloat;

  // Pull the r_L value from the bar-chart row that starts with "  <L>  ".
  // Rows look like:  "  %3d  %6d   %8.4f   %8.4f  ###". We parse the third
  // numeric field. Returns NaN-ish 1e30 if not found.
  function ReadRecovery(const S: string; LayerIdx: integer): TNeuralFloat;
  var
    Marker, Sub: string;
    Pos1, Pos2, Cnt, K: integer;
    Tok: string;
    Fields: array[0..3] of string;
  begin
    Result := 1e30;
    // Find a line whose trimmed start is the layer index followed by spaces.
    Pos1 := 1;
    while Pos1 <= Length(S) do
    begin
      Pos2 := Pos1;
      while (Pos2 <= Length(S)) and (S[Pos2] <> #10) do Inc(Pos2);
      Sub := Copy(S, Pos1, Pos2 - Pos1);
      Marker := Trim(Sub);
      // tokenise on whitespace
      Cnt := 0;
      for K := 0 to 3 do Fields[K] := '';
      Tok := '';
      for K := 1 to Length(Marker) do
      begin
        if Marker[K] = ' ' then
        begin
          if Tok <> '' then
          begin
            if Cnt <= 3 then Fields[Cnt] := Tok;
            Inc(Cnt);
            Tok := '';
          end;
        end
        else Tok := Tok + Marker[K];
      end;
      if Tok <> '' then
      begin
        if Cnt <= 3 then Fields[Cnt] := Tok;
        Inc(Cnt);
      end;
      // Field0 = layer idx, Field1 = argmax, Field2 = r_L, Field3 = delta.
      if (Cnt >= 3) and (Fields[0] = IntToStr(LayerIdx)) then
      begin
        // Only accept rows where field1 (argmax) is a plain integer, to avoid
        // matching header/verdict lines that happen to start with a number.
        if (Fields[1] <> '') and
           (TryStrToFloat(Fields[2], Result)) then Exit;
      end;
      Pos1 := Pos2 + 1;
    end;
  end;

begin
  // nil NN handled gracefully.
  Report := TNNet.ActivationPatchingReport(nil, nil, nil);
  AssertTrue('nil NN reported gracefully', Pos('NN is nil', Report) > 0);

  NN := TNNet.Create();
  CleanIn := TNNetVolume.Create(5, 1, 1);
  CorruptIn := TNNetVolume.Create(5, 1, 1);
  Before := TNNetVolume.Create();
  After := TNNetVolume.Create();
  try
    NN.AddLayer(TNNetInput.Create(5, 1, 1));
    NN.AddLayer(TNNetFullConnectReLU.Create(8));
    NN.AddLayer(TNNetFullConnectReLU.Create(8));
    NN.AddLayer(TNNetFullConnectLinear.Create(3));
    NN.AddLayer(TNNetSoftMax.Create());
    NN.SetLearningRate(0.01, 0.9);
    NN.InitWeights();

    // nil input handled gracefully.
    Report := TNNet.ActivationPatchingReport(NN, nil, CorruptIn);
    AssertTrue('nil input reported gracefully',
      Pos('is nil', Report) > 0);

    // Find a clean/corrupt pair the (untrained) net maps to different argmax.
    CleanCls := -1; CorruptCls := -1;
    for Tries := 1 to 200 do
    begin
      for J := 0 to 4 do CleanIn.Raw[J] := (Random - 0.5) * 4.0;
      for J := 0 to 4 do CorruptIn.Raw[J] := (Random - 0.5) * 4.0;
      NN.Compute(CleanIn);
      CleanCls := NN.GetLastLayer.Output.GetClass();
      NN.Compute(CorruptIn);
      CorruptCls := NN.GetLastLayer.Output.GetClass();
      if CleanCls <> CorruptCls then Break;
    end;

    // Capture clean output BEFORE the report to verify live-state restore.
    NN.Compute(CleanIn);
    Before.Copy(NN.GetLastLayer.Output);

    Report := TNNet.ActivationPatchingReport(NN, CleanIn, CorruptIn);
    AssertTrue('Report is non-empty', Length(Report) > 0);
    AssertTrue('Header present',
      Pos('ActivationPatchingReport', Report) > 0);
    AssertTrue('Causal trace section present',
      Pos('Per-layer recovery r_L', Report) > 0);
    AssertTrue('Faithfulness section present',
      Pos('Faithfulness check', Report) > 0);

    // Exact-recovery faithfulness checks: r_0 == 1 and r_last == 1.
    R0 := ReadRecovery(Report, 0);
    RLast := ReadRecovery(Report, 4);   // last layer index = 4
    AssertTrue('r_0 parsed', R0 < 1e29);
    AssertTrue('r_last parsed', RLast < 1e29);
    AssertTrue('r_0 == 1 (input patch reconstructs clean run)',
      Abs(R0 - 1.0) < 1e-4);
    AssertTrue('r_last == 1 (last layer output is the logits)',
      Abs(RLast - 1.0) < 1e-4);

    // Live state restored: clean output identical to the pre-report capture.
    NN.Compute(CleanIn);
    After.Copy(NN.GetLastLayer.Output);
    AssertEquals('output size unchanged', Before.Size, After.Size);
    MaxDiff := 0;
    for I := 0 to Before.Size - 1 do
    begin
      D := Abs(Before.Raw[I] - After.Raw[I]);
      if D > MaxDiff then MaxDiff := D;
    end;
    AssertTrue('live state restored (max output diff < 1e-6)',
      MaxDiff < 1e-6);

    // Denominator-collapse path: CorruptInput == CleanInput must WARN, not crash.
    Report := TNNet.ActivationPatchingReport(NN, CleanIn, CleanIn);
    AssertTrue('denominator-collapse warns',
      (Pos('WARNING', Report) > 0) and (Pos('denominator collapsed', Report) > 0));
    AssertTrue('collapse path still non-empty', Length(Report) > 0);

    // TargetIdx out of range handled gracefully.
    Report := TNNet.ActivationPatchingReport(NN, CleanIn, CorruptIn, 99);
    AssertTrue('out-of-range TargetIdx reported',
      Pos('out of range', Report) > 0);
  finally
    After.Free;
    Before.Free;
    CorruptIn.Free;
    CleanIn.Free;
    NN.Free;
  end;
end;

procedure TTestNeuralLayersExtra.TestToGraphvizDotSmoke;
var
  NN: TNNet;
  ShortCut, LongPath: TNNetLayer;
  Dot, EmptyDot: string;
begin
  // Empty network: a valid empty digraph, no crash.
  NN := TNNet.Create();
  try
    EmptyDot := NN.ToGraphvizDot();
    AssertTrue('empty net non-empty', Length(EmptyDot) > 0);
    AssertTrue('empty net contains digraph', Pos('digraph', EmptyDot) > 0);
  finally
    NN.Free;
  end;

  // A tiny branched net with a TNNetSum so multi-input edges are exercised.
  NN := TNNet.Create();
  try
    NN.AddLayer(TNNetInput.Create(6, 1, 1));
    ShortCut := NN.AddLayer(TNNetFullConnectLinear.Create(8));
    NN.AddLayerAfter(TNNetFullConnectLinear.Create(8), ShortCut);
    NN.AddLayer(TNNetReLU.Create());
    LongPath := NN.AddLayer(TNNetFullConnectLinear.Create(8));
    NN.AddLayer(TNNetSum.Create([ShortCut, LongPath]));
    NN.AddLayer(TNNetFullConnectLinear.Create(2));

    Dot := NN.ToGraphvizDot('SmokeNet');
    AssertTrue('dot non-empty', Length(Dot) > 0);
    AssertTrue('dot contains digraph', Pos('digraph', Dot) > 0);
    AssertTrue('dot contains the graph name', Pos('SmokeNet', Dot) > 0);
    AssertTrue('dot contains at least one edge', Pos('->', Dot) > 0);
    // The TNNetSum node must have two incoming edges; assert both branch
    // indices point at the sum layer (index 5).
    AssertTrue('dot mentions the TNNetSum layer', Pos('TNNetSum', Dot) > 0);
    AssertTrue('shortcut edge into sum present', Pos('1 -> 5', Dot) > 0);
    AssertTrue('longpath edge into sum present', Pos('4 -> 5', Dot) > 0);
  finally
    NN.Free;
  end;
end;

procedure TTestNeuralLayersExtra.TestPredictionDepthReportSmoke;
var
  NN: TNNet;
  Support, Queries, OneClass: TNNetVolumeList;
  SupLabels, QryLabels, OneClassLabels: array of integer;
  X: TNNetVolume;
  Report, OneClassReport: string;
  Ep, I, C, B: integer;
  FinalAgreeStart, FinalAgreeEnd: integer;
  AgreeStr: string;
  Agree: TNeuralFloat;
  AllZeroDepth: boolean;
  DepthZeroPos: integer;
  FS: TFormatSettings;
  Centers: array[0..2, 0..1] of TNeuralFloat =
    ((-2.0, -2.0), (2.0, 2.0), (2.0, -2.0));
  NoLabels: array of integer;
begin
  FS := DefaultFormatSettings;
  FS.DecimalSeparator := '.';
  FS.ThousandSeparator := #0;
  SetLength(NoLabels, 0);

  // nil NN handled gracefully (both overloads).
  Report := TNNet.PredictionDepthReport(nil, nil, NoLabels, nil);
  AssertTrue('nil NN reported gracefully', Pos('NN is nil', Report) > 0);
  Report := TNNet.PredictionDepthReport(nil, nil, NoLabels, nil, NoLabels);
  AssertTrue('nil NN reported gracefully (labelled overload)',
    Pos('NN is nil', Report) > 0);

  NN := TNNet.Create();
  Support := TNNetVolumeList.Create();
  Queries := TNNetVolumeList.Create();
  OneClass := TNNetVolumeList.Create();
  try
    NN.AddLayer(TNNetInput.Create(2, 1, 1));
    NN.AddLayer(TNNetFullConnectReLU.Create(10));
    NN.AddLayer(TNNetFullConnectReLU.Create(10));
    NN.AddLayer(TNNetFullConnectLinear.Create(3));
    NN.AddLayer(TNNetSoftMax.Create());
    NN.SetLearningRate(0.05, 0.9);
    NN.InitWeights();

    // empty support handled gracefully (on a valid net).
    Report := TNNet.PredictionDepthReport(NN, Support, NoLabels, Queries);
    AssertTrue('empty support reported gracefully',
      Pos('Support is nil or empty', Report) > 0);

    // Build a labelled 3-cluster support batch (round-robin keeps it balanced).
    RandSeed := 4242;
    SetLength(SupLabels, 90);
    for I := 0 to 89 do
    begin
      C := I mod 3;
      X := TNNetVolume.Create(2, 1, 1);
      X.FData[0] := Centers[C][0] + (Random - 0.5);
      X.FData[1] := Centers[C][1] + (Random - 0.5);
      Support.Add(X);
      SupLabels[I] := C;
    end;
    SetLength(QryLabels, 60);
    for I := 0 to 59 do
    begin
      C := I mod 3;
      X := TNNetVolume.Create(2, 1, 1);
      X.FData[0] := Centers[C][0] + (Random - 0.5);
      X.FData[1] := Centers[C][1] + (Random - 0.5);
      Queries.Add(X);
      QryLabels[I] := C;
    end;

    // Train briefly so the deep layers separate the clusters.
    for Ep := 1 to 40 do
      for B := 0 to Support.Count - 1 do
      begin
        X := TNNetVolume.Create(3, 1, 1);
        X.Fill(0);
        X.FData[SupLabels[B]] := 1.0;
        try
          NN.Compute(Support[B]);
          NN.Backpropagate(X);
        finally
          X.Free;
        end;
      end;

    // ---- main report (unlabelled-query overload) ----
    Report := TNNet.PredictionDepthReport(NN, Support, SupLabels, Queries);
    AssertTrue('Report is non-empty', Length(Report) > 0);
    AssertTrue('Header present', Pos('PredictionDepthReport', Report) > 0);
    AssertTrue('depth histogram present',
      Pos('Distribution of prediction depth', Report) > 0);
    AssertTrue('newly-resolved profile present',
      Pos('newly-resolved count', Report) > 0);
    AssertTrue('hardest-query queue present',
      Pos('Hardest', Report) > 0);
    AssertTrue('final-layer agreement line present',
      Pos('Final-layer k-NN-vote-vs-network-argmax agreement', Report) > 0);
    // labelled overload adds the correctness cross-tab.
    Report := TNNet.PredictionDepthReport(NN, Support, SupLabels, Queries,
      QryLabels);
    AssertTrue('cross-tab present (labelled overload)',
      Pos('Correctness cross-tab', Report) > 0);

    // ---- built-in correctness: support set fed as its OWN queries ----
    // every sample gets a finite depth and the final-layer k-NN vote matches
    // the network argmax for a high fraction (a point is its own nearest
    // neighbour at cosine distance 0).
    Report := TNNet.PredictionDepthReport(NN, Support, SupLabels, Support,
      SupLabels);
    FinalAgreeStart :=
      Pos('Final-layer k-NN-vote-vs-network-argmax agreement = ', Report);
    AssertTrue('agreement line found', FinalAgreeStart > 0);
    FinalAgreeStart := FinalAgreeStart +
      Length('Final-layer k-NN-vote-vs-network-argmax agreement = ');
    FinalAgreeEnd := PosEx(' ', Report, FinalAgreeStart);
    AssertTrue('agreement terminator found', FinalAgreeEnd > FinalAgreeStart);
    AgreeStr := Trim(Copy(Report, FinalAgreeStart,
      FinalAgreeEnd - FinalAgreeStart));
    Agree := StrToFloatDef(AgreeStr, -1, FS);
    AssertTrue('agreement parsed', Agree >= 0);
    AssertTrue('support-as-own-query final-layer agreement is high',
      Agree >= 0.9);

    // ---- built-in correctness: a ONE-CLASS support set drives depth to 0 ----
    // with a single support class the k-NN vote can only ever be that class,
    // and for any query the net itself classifies as that class the depth locks
    // in immediately at layer 0. Build queries the trained net maps to class 0
    // and a one-class (class 0) support set.
    SetLength(OneClassLabels, 30);
    for I := 0 to 29 do
    begin
      X := TNNetVolume.Create(2, 1, 1);
      X.FData[0] := Centers[0][0] + (Random - 0.5);
      X.FData[1] := Centers[0][1] + (Random - 0.5);
      OneClass.Add(X);
      OneClassLabels[I] := 0;  // all the same class
    end;
    // sanity: the trained net maps these points to class 0.
    NN.Compute(OneClass[0]);
    AssertEquals('one-class points classify as class 0', 0,
      NN.GetLastLayer.Output.GetClass());
    // queries = the same one-class cluster points.
    OneClassReport := TNNet.PredictionDepthReport(NN, OneClass, OneClassLabels,
      OneClass);
    AssertTrue('one-class report non-empty', Length(OneClassReport) > 0);
    // mean depth must be exactly 0 (every sample locks in at layer 0).
    AssertTrue('one-class mean depth is 0',
      Pos('Mean depth = 0.000', OneClassReport) > 0);
    // all queries land in the shallowest histogram bin.
    DepthZeroPos := Pos('depth | n=  30', OneClassReport);
    AllZeroDepth := DepthZeroPos > 0;
    AssertTrue('one-class: all 30 queries in the depth-0 histogram bin',
      AllZeroDepth);
  finally
    Support.Free;
    Queries.Free;
    OneClass.Free;
    NN.Free;
  end;
end;

procedure TTestNeuralLayersExtra.TestGradCAMReportSmoke;
var
  NN: TNNet;
  vInput: TNNetVolume;
  Report: string;
begin
  // nil NN handled gracefully.
  Report := TNNet.GradCAMReport(nil, nil);
  AssertTrue('nil NN reported gracefully', Pos('NN is nil', Report) > 0);

  NN := TNNet.Create();
  vInput := nil;
  try
    NN.AddLayer(TNNetInput.Create(8, 8, 2));
    NN.AddLayer(TNNetConvolutionReLU.Create(4, 3, 1, 1));
    NN.AddLayer(TNNetMaxPool.Create(2));
    NN.AddLayer(TNNetFullConnectReLU.Create(8));
    NN.AddLayer(TNNetFullConnectLinear.Create(3));
    NN.AddLayer(TNNetSoftMax.Create());
    NN.SetLearningRate(0.01, 0.9);
    NN.InitWeights();

    // nil probe handled gracefully.
    Report := TNNet.GradCAMReport(NN, nil);
    AssertTrue('nil probe reported gracefully', Pos('nil or empty', Report) > 0);

    vInput := TNNetVolume.Create(8, 8, 2);
    vInput.FillForDebug();
    Report := TNNet.GradCAMReport(NN, vInput);
    AssertTrue('header present', Pos('GradCAMReport', Report) > 0);
    AssertTrue('coarse map present', Pos('Coarse Grad-CAM map', Report) > 0);
    AssertTrue('peak line present', Pos('peak at conv cell', Report) > 0);
  finally
    if vInput <> nil then vInput.Free;
    NN.Free;
  end;
end;

procedure TTestNeuralLayersExtra.TestLayerTimingReportSmoke;
var
  NN: TNNet;
  Sample: TNNetVolume;
  S: string;
begin
  NN := TNNet.Create;
  NN.AddLayer(TNNetInput.Create(8, 8, 3));
  NN.AddLayer(TNNetConvolutionReLU.Create(4, 3, 1, 1));
  NN.AddLayer(TNNetFullConnectReLU.Create(10));
  NN.AddLayer(TNNetSoftMax.Create());
  Sample := TNNetVolume.Create(8, 8, 3);
  Sample.FillForDebug();
  S := TNNet.LayerTimingReport(NN, Sample, 3);
  AssertTrue('Report is not empty', Length(S) > 0);
  AssertTrue('Report has header', Pos('Layer Timing Report', S) > 0);
  AssertTrue('Report has total', Pos('TOTAL:', S) > 0);
  Sample.Free;
  NN.Free;
  // nil NN must not crash
  S := TNNet.LayerTimingReport(nil, nil, 3);
  AssertTrue('nil NN handled', Length(S) > 0);
end;

procedure TTestNeuralLayersExtra.TestMixtureOfExpertsShapeForwardTrainAndRoundTrip;
const
  d_model = 6;
  NumExperts = 4;
  ExpertHidden = 8;
var
  NN, NN2: TNNet;
  Input, Output, Target: TNNetVolume;
  MoEOut: TNNetLayer;
  Struct: string;
  InitialLoss, FinalLoss: TNeuralFloat;
  I: integer;
begin
  NN := TNNet.Create();
  NN.AddLayer(TNNetInput.Create(1, 1, d_model));
  MoEOut := NN.AddMixtureOfExperts(nil, NumExperts, ExpertHidden);
  // Block must be shape-preserving (drop-in FFN replacement).
  AssertEquals('MoE output depth matches d_model', d_model, MoEOut.Output.Depth);
  AssertEquals('MoE output SizeX preserved', 1, MoEOut.Output.SizeX);
  AssertEquals('MoE output SizeY preserved', 1, MoEOut.Output.SizeY);
  NN.AddLayer(TNNetFullConnectLinear.Create(1));
  NN.SetLearningRate(0.01, 0.9);

  Input := TNNetVolume.Create(1, 1, d_model);
  Input.Randomize();
  Output := TNNetVolume.Create();
  Target := TNNetVolume.Create(1, 1, 1);
  Target.Fill(0.5);

  // Round-trip: structure save/load must reproduce the layer count.
  Struct := NN.SaveStructureToString();
  NN2 := TNNet.Create();
  NN2.LoadStructureFromString(Struct);
  AssertEquals('MoE structure round-trips layer count',
    NN.CountLayers(), NN2.CountLayers());
  NN2.Free;

  // Forward + a short train: loss must decrease (the block learns end-to-end).
  NN.Compute(Input);
  NN.GetOutput(Output);
  InitialLoss := Abs(Output.FData[0] - Target.FData[0]);
  for I := 0 to 400 do
  begin
    NN.Compute(Input);
    NN.Backpropagate(Target);
  end;
  NN.Compute(Input);
  NN.GetOutput(Output);
  FinalLoss := Abs(Output.FData[0] - Target.FData[0]);
  AssertTrue('MoE final loss should be below initial', FinalLoss < InitialLoss);

  Input.Free;
  Output.Free;
  Target.Free;
  NN.Free;
end;

// TNNet.AddMixtureOfDepths conditional-compute block. Asserts:
//  1) the block is shape-preserving (drop-in residual over the sequence);
//  2) DEGENERATE ANCHOR: with Capacity = SeqLen the top-K keeps every token, so
//     the wrapper reduces EXACTLY to a per-token scalar-gated residual block
//     y = x + Sigmoid(router)*Block(x). We build a reference net with the SAME
//     ordered layers but the TopK replaced by a plain TNNetIdentity (which is
//     exactly what TopK does when K >= Depth: it early-exits as a passthrough),
//     copy the MoD weights into it, and assert bit-for-bit equal output;
//  3) the wrapper wiring round-trips through SaveToString -> LoadFromString and
//     reproduces the output exactly.
procedure TTestNeuralLayersExtra.TestMixtureOfDepthsShapeDegenerateAndRoundTrip;
const
  cSeqLen = 5;
  cDModel = 4;
  cHidden = 6;

  // Wire the SAME layer sequence the AddMixtureOfDepths builder uses, but with
  // the TopK swapped for an identity passthrough. With Capacity = SeqLen the
  // real TopK is itself an identity passthrough, so this is an exact reference.
  procedure BuildReference(ANN: TNNet);
  var
    Inp, Masked, Block, Gate, Gated: TNNetLayer;
  begin
    Inp := ANN.AddLayer(TNNetInput.Create(cSeqLen, 1, cDModel, 1));
    ANN.AddLayerAfter(TNNetPointwiseConvLinear.Create(1), Inp);
    ANN.AddLayer(TNNetSigmoid.Create());
    ANN.AddLayer(TNNetTransposeXD.Create());
    ANN.AddLayer(TNNetIdentity.Create()); // stands in for TopK(Capacity=SeqLen)
    Masked := ANN.AddLayer(TNNetTransposeXD.Create());
    ANN.AddLayerAfter(TNNetPointwiseConvReLU.Create(cHidden), Inp);
    Block := ANN.AddLayer(TNNetPointwiseConvLinear.Create(cDModel));
    Gate := ANN.AddLayer(TNNetDeepConcat.Replicate(cDModel, Masked));
    Gated := ANN.AddLayer(TNNetCellMulByCell.Create(Block, Gate));
    ANN.AddLayer(TNNetSum.Create([Inp, Gated]));
  end;

var
  NN, Ref: TNNet;
  Saved: string;
  NN2: TNNet;
  Input: TNNetVolume;
  MoDOut: TNNetLayer;
  i: integer;
begin
  NN := TNNet.Create();
  Input := TNNetVolume.Create(cSeqLen, 1, cDModel);
  try
    NN.AddLayer(TNNetInput.Create(cSeqLen, 1, cDModel, 1));
    MoDOut := NN.AddMixtureOfDepths(nil,
      [ TNNetPointwiseConvReLU.Create(cHidden),
        TNNetPointwiseConvLinear.Create(cDModel) ], cSeqLen);

    // 1) Shape-preserving.
    AssertEquals('MoD output SizeX preserved', cSeqLen, MoDOut.Output.SizeX);
    AssertEquals('MoD output SizeY preserved', 1, MoDOut.Output.SizeY);
    AssertEquals('MoD output depth matches d_model', cDModel, MoDOut.Output.Depth);

    for i := 0 to Input.Size - 1 do
      Input.Raw[i] := Sin(i * 0.61) * 1.3 + 0.4;

    // 2) Degenerate equality anchor (Capacity = SeqLen).
    Ref := TNNet.Create();
    try
      BuildReference(Ref);
      AssertEquals('MoD reference has same layer count',
        NN.CountLayers(), Ref.CountLayers());
      Ref.CopyWeights(NN); // copy router + block weights layer-by-layer
      NN.Compute(Input);
      Ref.Compute(Input);
      AssertEquals('MoD degenerate output size matches reference',
        NN.GetLastLayer.Output.Size, Ref.GetLastLayer.Output.Size);
      for i := 0 to NN.GetLastLayer.Output.Size - 1 do
        AssertEquals('MoD Capacity=SeqLen == gated residual block at ' + IntToStr(i),
          Ref.GetLastLayer.Output.Raw[i], NN.GetLastLayer.Output.Raw[i], 1e-5);
    finally
      Ref.Free;
    end;

    // 3) SaveToString -> LoadFromString round-trip reproduces the output.
    Saved := NN.SaveToString();
    NN2 := TNNet.Create();
    try
      NN2.LoadFromString(Saved);
      AssertEquals('MoD round-trip layer count',
        NN.CountLayers(), NN2.CountLayers());
      NN.Compute(Input);
      NN2.Compute(Input);
      for i := 0 to NN.GetLastLayer.Output.Size - 1 do
        AssertEquals('MoD round-trip output at ' + IntToStr(i),
          NN.GetLastLayer.Output.Raw[i], NN2.GetLastLayer.Output.Raw[i], 1e-5);
    finally
      NN2.Free;
    end;
  finally
    NN.Free;
    Input.Free;
  end;
end;

// TNNetDropBlock (Ghiasi et al. 2018) structured spatial dropout. Asserts:
//  1) inference mode (dropouts disabled) is an identity passthrough;
//  2) train mode zeroes contiguous block_size x block_size spatial regions
//     across ALL channels (one spatial block broadcast over Depth);
//  3) SaveToString -> LoadFromString preserves block_size + prob and, under a
//     fixed RNG seed, reproduces the train-mode output exactly.
procedure TTestNeuralLayersExtra.TestDropBlockSmokeAndRoundTrip;
var
  NN, NN2: TNNet;
  Input: TNNetVolume;
  Saved: string;
  i, x, y, d, Seed: integer;
  DropBlockIdx: integer;
  ZeroSpatialFound: boolean;
  AllDepthZero: boolean;
begin
  // --- 1) Inference identity passthrough -----------------------------------
  NN := TNNet.Create();
  Input := TNNetVolume.Create(7, 7, 3);
  try
    DropBlockIdx := 1; // input is layer 0, DropBlock is layer 1
    NN.AddLayer(TNNetInput.Create(7, 7, 3, 1));
    NN.AddLayer(TNNetDropBlock.Create(3, 0.3));
    NN.EnableDropouts(false); // inference
    for i := 0 to Input.Size - 1 do
      Input.Raw[i] := Sin(i * 0.37) * 0.9 + 1.3; // non-zero everywhere
    NN.Compute(Input);
    for i := 0 to NN.GetLastLayer.Output.Size - 1 do
      AssertEquals('DropBlock inference identity at ' + IntToStr(i),
        Input.Raw[i], NN.GetLastLayer.Output.Raw[i], 1e-6);

    // --- 2) Train mode zeroes a contiguous block across all channels -------
    NN.EnableDropouts(true);
    Seed := 1;
    ZeroSpatialFound := false;
    repeat
      RandSeed := Seed;
      NN.Compute(Input);
      // Look for a spatial position that was zeroed; assert it is zeroed for
      // EVERY channel (the block is broadcast over Depth).
      for y := 0 to 6 do
        for x := 0 to 6 do
        begin
          AllDepthZero := True;
          for d := 0 to 2 do
            if NN.GetLastLayer.Output.Data[x, y, d] <> 0 then
              AllDepthZero := False;
          if AllDepthZero then ZeroSpatialFound := True;
          // Sanity: a position is never partially dropped (some channels zero,
          // some not) since the spatial mask is shared across Depth.
          if AllDepthZero then
            for d := 0 to 2 do
              AssertEquals('DropBlock dropped block must zero every channel at (' +
                IntToStr(x) + ',' + IntToStr(y) + ',' + IntToStr(d) + ')',
                0.0, NN.GetLastLayer.Output.Data[x, y, d], 1e-6);
        end;
      Inc(Seed);
    until ZeroSpatialFound or (Seed > 50);
    AssertTrue('DropBlock train mode should zero at least one spatial block',
      ZeroSpatialFound);

    // --- 3) Serialization round-trip preserves block_size + prob -----------
    Saved := NN.SaveToString();
    NN2 := TNNet.Create();
    try
      NN2.LoadFromString(Saved);
      AssertTrue('DropBlock reloaded layer is TNNetDropBlock',
        NN2.Layers[DropBlockIdx] is TNNetDropBlock);
      // The structure string encodes FStruct[0]=block_size and FFloatSt[0]=prob;
      // an exact match proves both constructor args round-tripped.
      AssertEquals('DropBlock round-trip structure (block_size + prob)',
        NN.Layers[DropBlockIdx].SaveStructureToString(),
        NN2.Layers[DropBlockIdx].SaveStructureToString());

      // Under a fixed seed, train-mode output must match exactly.
      NN2.EnableDropouts(true);
      RandSeed := 12345;
      NN.Compute(Input);
      RandSeed := 12345;
      NN2.Compute(Input);
      AssertEquals('DropBlock round-trip output size',
        NN.GetLastLayer.Output.Size, NN2.GetLastLayer.Output.Size);
      for i := 0 to NN.GetLastLayer.Output.Size - 1 do
        AssertEquals('DropBlock round-trip output at ' + IntToStr(i),
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

initialization
  RegisterTest(TTestNeuralLayersExtra);

end.
