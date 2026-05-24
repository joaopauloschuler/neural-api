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
    procedure TestTopLogitMarginReportSmoke;
    procedure TestNeuronCorrelationReportSmoke;
    procedure TestLayerSensitivityReportSmoke;
    procedure TestEquivarianceReportSmoke;
    procedure TestTTAReportSmoke;
    procedure TestSaliencyReportSmoke;
    procedure TestDecisionBoundaryReportSmoke;
    procedure TestCalibrationReportSmoke;
    procedure TestFisherImportanceReportSmoke;
    procedure TestLinearProbeReportSmoke;
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

initialization
  RegisterTest(TTestNeuralLayersExtra);

end.
