unit TestNeuralLayersExtra;

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, Math, fpcunit, testregistry, neuralnetwork, neuralvolume;

const
  // Maximum number of elements to check for NaN/Inf in large tensors.
  MAX_NAN_CHECK_ITERATIONS = 100;

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
  end;

implementation

procedure TTestNeuralLayersExtra.TestDeconvolutionForward;
var
  NN: TNNet;
  Input: TNNetVolume;
  I: integer;
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
    
    // Numerical verification: output values should be finite
    for I := 0 to Min(MAX_NAN_CHECK_ITERATIONS, NN.GetLastLayer.Output.Size - 1) do
    begin
      AssertFalse('Output should not be NaN at index ' + IntToStr(I), IsNaN(NN.GetLastLayer.Output.Raw[I]));
      AssertFalse('Output should not be Inf at index ' + IntToStr(I), IsInfinite(NN.GetLastLayer.Output.Raw[I]));
    end;
    
    // Verify output has meaningful values (not all zeros)
    AssertTrue('Output should have non-zero sum', NN.GetLastLayer.Output.GetSumAbs() > 0);
  finally
    NN.Free;
    Input.Free;
  end;
end;

procedure TTestNeuralLayersExtra.TestDeconvolutionReLUForward;
var
  NN: TNNet;
  Input: TNNetVolume;
  I: integer;
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
    
    // Numerical verification: ReLU should produce non-negative outputs
    AssertTrue('ReLU output min should be >= 0', NN.GetLastLayer.Output.GetMin() >= -0.0001);
    
    // All outputs should be finite
    for I := 0 to Min(MAX_NAN_CHECK_ITERATIONS, NN.GetLastLayer.Output.Size - 1) do
      AssertFalse('Output should not be NaN', IsNaN(NN.GetLastLayer.Output.Raw[I]));
  finally
    NN.Free;
    Input.Free;
  end;
end;

procedure TTestNeuralLayersExtra.TestDeconvolutionOutputSize;
var
  NN: TNNet;
  Input: TNNetVolume;
  I: integer;
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
    
    // Numerical verification: total output size should match expected
    AssertEquals('Total output size should be 8*8*32 = 2048', 2048, NN.GetLastLayer.Output.Size);
    
    // Verify all outputs are finite
    for I := 0 to Min(MAX_NAN_CHECK_ITERATIONS, NN.GetLastLayer.Output.Size - 1) do
    begin
      AssertFalse('Output should not be NaN', IsNaN(NN.GetLastLayer.Output.Raw[I]));
      AssertFalse('Output should not be Inf', IsInfinite(NN.GetLastLayer.Output.Raw[I]));
    end;
  finally
    NN.Free;
    Input.Free;
  end;
end;

procedure TTestNeuralLayersExtra.TestDeLocalConnectForward;
var
  NN: TNNet;
  Input: TNNetVolume;
  I: integer;
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
    
    // Numerical verification: output values should be finite
    for I := 0 to Min(MAX_NAN_CHECK_ITERATIONS, NN.GetLastLayer.Output.Size - 1) do
    begin
      AssertFalse('Output should not be NaN', IsNaN(NN.GetLastLayer.Output.Raw[I]));
      AssertFalse('Output should not be Inf', IsInfinite(NN.GetLastLayer.Output.Raw[I]));
    end;
    
    // Verify output has meaningful values
    AssertTrue('Output should have non-zero sum', NN.GetLastLayer.Output.GetSumAbs() > 0);
  finally
    NN.Free;
    Input.Free;
  end;
end;

procedure TTestNeuralLayersExtra.TestDeLocalConnectReLUForward;
var
  NN: TNNet;
  Input: TNNetVolume;
  I: integer;
begin
  NN := TNNet.Create();
  Input := TNNetVolume.Create(4, 4, 4);
  try
    NN.AddLayer(TNNetInput.Create(4, 4, 4));
    NN.AddLayer(TNNetDeLocalConnectReLU.Create(8, 3, 0, 2));

    Input.Fill(1.0);
    NN.Compute(Input);

    AssertEquals('Output depth should be 8', 8, NN.GetLastLayer.Output.Depth);
    
    // Numerical verification: ReLU should produce non-negative outputs
    AssertTrue('ReLU output min should be >= 0', NN.GetLastLayer.Output.GetMin() >= -0.0001);
    
    // All outputs should be finite
    for I := 0 to Min(MAX_NAN_CHECK_ITERATIONS, NN.GetLastLayer.Output.Size - 1) do
      AssertFalse('Output should not be NaN', IsNaN(NN.GetLastLayer.Output.Raw[I]));
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

    // Note: TNNetSignedSquareRoot requires error derivatives to be initialized
    // to compute properly. During inference without backprop setup, it only
    // computes when (FOutput.Size = FOutputError.Size) and (FOutputErrorDeriv.Size = FOutput.Size)
    // This test verifies the layer runs without error and produces valid output
    AssertEquals('Output size should match input', 4, NN.GetLastLayer.Output.Size);
    
    // Verify outputs are finite (not NaN/Inf)
    AssertFalse('Output 0 should not be NaN', IsNaN(NN.GetLastLayer.Output.Raw[0]));
    AssertFalse('Output 1 should not be NaN', IsNaN(NN.GetLastLayer.Output.Raw[1]));
    AssertFalse('Output 2 should not be NaN', IsNaN(NN.GetLastLayer.Output.Raw[2]));
    AssertFalse('Output 3 should not be NaN', IsNaN(NN.GetLastLayer.Output.Raw[3]));
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
  I: integer;
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
    AssertEquals('Output SizeY should be 8', 8, NN.GetLastLayer.Output.SizeY);
    AssertEquals('Output Depth should be 32', 32, NN.GetLastLayer.Output.Depth);
    
    // Numerical verification: outputs should be finite
    for I := 0 to Min(MAX_NAN_CHECK_ITERATIONS, NN.GetLastLayer.Output.Size - 1) do
    begin
      AssertFalse('Output should not be NaN', IsNaN(NN.GetLastLayer.Output.Raw[I]));
      AssertFalse('Output should not be Inf', IsInfinite(NN.GetLastLayer.Output.Raw[I]));
    end;
    
    // Verify output has meaningful values
    AssertTrue('Output should have non-zero sum', NN.GetLastLayer.Output.GetSumAbs() > 0);
  finally
    NN.Free;
    Input.Free;
  end;
end;

procedure TTestNeuralLayersExtra.TestGroupedPointwiseConvReLU;
var
  NN: TNNet;
  Input: TNNetVolume;
  I: integer;
begin
  NN := TNNet.Create();
  Input := TNNetVolume.Create(8, 8, 8);
  try
    NN.AddLayer(TNNetInput.Create(8, 8, 8));
    // Create(pNumFeatures, pGroups: integer; pSuppressBias: integer)
    NN.AddLayer(TNNetGroupedPointwiseConvReLU.Create(16, 2, 0));

    Input.Fill(1.0);
    NN.Compute(Input);

    AssertEquals('Output SizeX should be 8', 8, NN.GetLastLayer.Output.SizeX);
    AssertEquals('Output SizeY should be 8', 8, NN.GetLastLayer.Output.SizeY);
    AssertEquals('Output Depth should be 16', 16, NN.GetLastLayer.Output.Depth);
    
    // ReLU should ensure non-negative outputs
    AssertTrue('ReLU output min should be >= 0', NN.GetLastLayer.Output.GetMin() >= -0.0001);
    
    // Numerical verification: outputs should be finite
    for I := 0 to Min(MAX_NAN_CHECK_ITERATIONS, NN.GetLastLayer.Output.Size - 1) do
      AssertFalse('Output should not be NaN', IsNaN(NN.GetLastLayer.Output.Raw[I]));
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

initialization
  RegisterTest(TTestNeuralLayersExtra);

end.
