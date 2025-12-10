unit TestNeuralVolume;

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, fpcunit, testregistry, neuralvolume;

type
  TTestNeuralVolume = class(TTestCase)
  published
    procedure TestVolumeCreation;
    procedure TestVolumeFill;
    procedure TestVolumeDotProduct;
    procedure TestVolumeAddSub;
    procedure TestVolumeCopy;
    procedure TestVolumeSaveLoad;
    // New comprehensive tests
    procedure TestVolumeMul;
    procedure TestVolumeDiv;
    procedure TestVolumeResize;
    procedure TestVolumeStatistics;
    procedure TestVolumeMinMax;
    procedure TestVolumeFlip;
    procedure TestVolumeClassification;
    procedure TestVolumeSoftMax;
    procedure TestVolumePadding;
    procedure TestVolumeTranspose;
    // Additional volume tests
    procedure TestVolumeNormalization;
    procedure TestVolumeMagnitude;
    procedure TestVolumeEntropy;
    procedure TestVolumeOneHotEncoding;
    procedure TestVolumePositionalEncoding;
    procedure TestVolumeColorConversions;
    procedure TestVolumeGaussianNoise;
    procedure TestVolumeCopyResizing;
    procedure TestVolumeCopyCropping;
    procedure TestVolumeShift;
    procedure TestVolumeRawPosAndPtr;
    procedure TestVolumeDepthOperations;
  end;

implementation

procedure TTestNeuralVolume.TestVolumeCreation;
var
  V: TNNetVolume;
begin
  V := TNNetVolume.Create(32, 32, 3);
  try
    AssertEquals('SizeX should be 32', 32, V.SizeX);
    AssertEquals('SizeY should be 32', 32, V.SizeY);
    AssertEquals('Depth should be 3', 3, V.Depth);
    AssertEquals('Total size should be 3072', 3072, V.Size);
  finally
    V.Free;
  end;
end;

procedure TTestNeuralVolume.TestVolumeFill;
var
  V: TNNetVolume;
  I: integer;
begin
  V := TNNetVolume.Create(10, 10, 1);
  try
    V.Fill(5.0);
    for I := 0 to V.Size - 1 do
      AssertEquals('All values should be 5.0', 5.0, V.Raw[I], 0.0001);
  finally
    V.Free;
  end;
end;

procedure TTestNeuralVolume.TestVolumeDotProduct;
var
  V1, V2: TNNetVolume;
  DotProd: TNeuralFloat;
begin
  V1 := TNNetVolume.Create(4, 1, 1);
  V2 := TNNetVolume.Create(4, 1, 1);
  try
    V1.Fill(2.0);
    V2.Fill(3.0);
    DotProd := V1.DotProduct(V2);
    AssertEquals('Dot product of [2,2,2,2] and [3,3,3,3] should be 24', 24.0, DotProd, 0.0001);
  finally
    V1.Free;
    V2.Free;
  end;
end;

procedure TTestNeuralVolume.TestVolumeAddSub;
var
  V1, V2: TNNetVolume;
begin
  V1 := TNNetVolume.Create(4, 1, 1);
  V2 := TNNetVolume.Create(4, 1, 1);
  try
    V1.Fill(5.0);
    V2.Fill(3.0);
    V1.Add(V2);
    AssertEquals('After adding, values should be 8.0', 8.0, V1.Raw[0], 0.0001);
    V1.Sub(V2);
    AssertEquals('After subtracting, values should be 5.0', 5.0, V1.Raw[0], 0.0001);
  finally
    V1.Free;
    V2.Free;
  end;
end;

procedure TTestNeuralVolume.TestVolumeCopy;
var
  V1, V2: TNNetVolume;
begin
  V1 := TNNetVolume.Create(10, 10, 3);
  V2 := TNNetVolume.Create(10, 10, 3);
  try
    V1.RandomizeGaussian();
    V2.Copy(V1);
    AssertEquals('Copied volume should match', 0.0, V1.SumDiff(V2), 0.0001);
  finally
    V1.Free;
    V2.Free;
  end;
end;

procedure TTestNeuralVolume.TestVolumeSaveLoad;
var
  V1, V2: TNNetVolume;
  SavedStr: string;
begin
  V1 := TNNetVolume.Create(5, 5, 2);
  V2 := TNNetVolume.Create(1, 1, 1);
  try
    V1.RandomizeGaussian();
    SavedStr := V1.SaveToString();
    V2.LoadFromString(SavedStr);
    AssertEquals('Loaded volume should match saved', 0.0, V1.SumDiff(V2), 0.0001);
  finally
    V1.Free;
    V2.Free;
  end;
end;

procedure TTestNeuralVolume.TestVolumeMul;
var
  V1, V2: TNNetVolume;
begin
  V1 := TNNetVolume.Create(4, 1, 1);
  V2 := TNNetVolume.Create(4, 1, 1);
  try
    V1.Fill(5.0);
    V1.Mul(2.0);
    AssertEquals('After multiplying by 2, values should be 10.0', 10.0, V1.Raw[0], 0.0001);
    
    V1.Fill(3.0);
    V2.Fill(4.0);
    V1.Mul(V2);
    AssertEquals('After element-wise multiplication, values should be 12.0', 12.0, V1.Raw[0], 0.0001);
  finally
    V1.Free;
    V2.Free;
  end;
end;

procedure TTestNeuralVolume.TestVolumeDiv;
var
  V1, V2: TNNetVolume;
begin
  V1 := TNNetVolume.Create(4, 1, 1);
  V2 := TNNetVolume.Create(4, 1, 1);
  try
    V1.Fill(10.0);
    V1.Divi(2.0);
    AssertEquals('After dividing by 2, values should be 5.0', 5.0, V1.Raw[0], 0.0001);
    
    V1.Fill(12.0);
    V2.Fill(4.0);
    V1.Divi(V2);
    AssertEquals('After element-wise division, values should be 3.0', 3.0, V1.Raw[0], 0.0001);
  finally
    V1.Free;
    V2.Free;
  end;
end;

procedure TTestNeuralVolume.TestVolumeResize;
var
  V: TNNetVolume;
begin
  V := TNNetVolume.Create(10, 10, 3);
  try
    AssertEquals('Initial SizeX should be 10', 10, V.SizeX);
    AssertEquals('Initial SizeY should be 10', 10, V.SizeY);
    AssertEquals('Initial Depth should be 3', 3, V.Depth);
    
    V.ReSize(20, 15, 5);
    AssertEquals('After resize SizeX should be 20', 20, V.SizeX);
    AssertEquals('After resize SizeY should be 15', 15, V.SizeY);
    AssertEquals('After resize Depth should be 5', 5, V.Depth);
    AssertEquals('After resize total size should be 1500', 1500, V.Size);
  finally
    V.Free;
  end;
end;

procedure TTestNeuralVolume.TestVolumeStatistics;
var
  V: TNNetVolume;
  Avg, Sum, Variance, StdDev: TNeuralFloat;
begin
  V := TNNetVolume.Create(4, 1, 1);
  try
    // Set values: 2, 4, 6, 8
    V.Raw[0] := 2.0;
    V.Raw[1] := 4.0;
    V.Raw[2] := 6.0;
    V.Raw[3] := 8.0;
    
    Sum := V.GetSum();
    AssertEquals('Sum should be 20.0', 20.0, Sum, 0.0001);
    
    Avg := V.GetAvg();
    AssertEquals('Average should be 5.0', 5.0, Avg, 0.0001);
    
    Variance := V.GetVariance();
    // Variance of [2,4,6,8] = E[(X-5)^2] = (9+1+1+9)/4 = 5
    AssertEquals('Variance should be 5.0', 5.0, Variance, 0.0001);
    
    StdDev := V.GetStdDeviation();
    // StdDev = sqrt(5) ≈ 2.236
    AssertEquals('StdDeviation should be ~2.236', 2.236, StdDev, 0.01);
  finally
    V.Free;
  end;
end;

procedure TTestNeuralVolume.TestVolumeMinMax;
var
  V: TNNetVolume;
  MinVal, MaxVal, MaxAbsVal: TNeuralFloat;
begin
  V := TNNetVolume.Create(5, 1, 1);
  try
    V.Raw[0] := -3.0;
    V.Raw[1] := 1.0;
    V.Raw[2] := 5.0;
    V.Raw[3] := -7.0;
    V.Raw[4] := 2.0;
    
    MinVal := V.GetMin();
    MaxVal := V.GetMax();
    MaxAbsVal := V.GetMaxAbs();
    
    AssertEquals('Min should be -7.0', -7.0, MinVal, 0.0001);
    AssertEquals('Max should be 5.0', 5.0, MaxVal, 0.0001);
    AssertEquals('MaxAbs should be 7.0', 7.0, MaxAbsVal, 0.0001);
  finally
    V.Free;
  end;
end;

procedure TTestNeuralVolume.TestVolumeFlip;
var
  V: TNNetVolume;
begin
  V := TNNetVolume.Create(3, 1, 1);
  try
    V.Raw[0] := 1.0;
    V.Raw[1] := 2.0;
    V.Raw[2] := 3.0;
    
    V.FlipX();
    
    AssertEquals('After FlipX, first value should be 3.0', 3.0, V.Raw[0], 0.0001);
    AssertEquals('After FlipX, middle value should be 2.0', 2.0, V.Raw[1], 0.0001);
    AssertEquals('After FlipX, last value should be 1.0', 1.0, V.Raw[2], 0.0001);
  finally
    V.Free;
  end;
end;

procedure TTestNeuralVolume.TestVolumeClassification;
var
  V: TNNetVolume;
  PredictedClass: integer;
begin
  V := TNNetVolume.Create(5, 1, 1);
  try
    // Simulate classification output with class 2 having highest probability
    V.Raw[0] := 0.1;
    V.Raw[1] := 0.2;
    V.Raw[2] := 0.5;
    V.Raw[3] := 0.15;
    V.Raw[4] := 0.05;
    
    PredictedClass := V.GetClass();
    AssertEquals('Predicted class should be 2', 2, PredictedClass);
    
    // Test SetClass with single value parameter
    // SetClass(class, value) fills non-class elements with -value
    // This is useful for hyperbolic tangent activations (-1 to +1 range)
    V.SetClass(3, 1.0);
    AssertEquals('After SetClass(3), class 3 should be 1.0', 1.0, V.Raw[3], 0.0001);
    AssertEquals('After SetClass(3), class 0 should be -1.0', -1.0, V.Raw[0], 0.0001);
    
    // Test SetClass with explicit true/false values (two-parameter overload)
    // This allows standard one-hot encoding (0 for false, 1 for true)
    V.SetClass(2, 1.0, 0.0);
    AssertEquals('After SetClass(2, 1.0, 0.0), class 2 should be 1.0', 1.0, V.Raw[2], 0.0001);
    AssertEquals('After SetClass(2, 1.0, 0.0), class 0 should be 0.0', 0.0, V.Raw[0], 0.0001);
  finally
    V.Free;
  end;
end;

procedure TTestNeuralVolume.TestVolumeSoftMax;
var
  V: TNNetVolume;
  SumAfterSoftMax: TNeuralFloat;
begin
  V := TNNetVolume.Create(4, 1, 1);
  try
    V.Raw[0] := 1.0;
    V.Raw[1] := 2.0;
    V.Raw[2] := 3.0;
    V.Raw[3] := 4.0;
    
    V.SoftMax();
    
    SumAfterSoftMax := V.GetSum();
    // SoftMax output should sum to 1.0
    AssertEquals('SoftMax output should sum to 1.0', 1.0, SumAfterSoftMax, 0.0001);
    
    // Higher input values should have higher probabilities
    AssertTrue('V[3] should be greater than V[0]', V.Raw[3] > V.Raw[0]);
    AssertTrue('V[3] should be greater than V[1]', V.Raw[3] > V.Raw[1]);
    AssertTrue('V[3] should be greater than V[2]', V.Raw[3] > V.Raw[2]);
  finally
    V.Free;
  end;
end;

procedure TTestNeuralVolume.TestVolumePadding;
var
  Original, Padded: TNNetVolume;
begin
  Original := TNNetVolume.Create(3, 3, 1);
  Padded := TNNetVolume.Create(1, 1, 1);
  try
    Original.Fill(1.0);
    
    Padded.CopyPadding(Original, 1);
    
    // After padding by 1, size should be 5x5
    AssertEquals('Padded SizeX should be 5', 5, Padded.SizeX);
    AssertEquals('Padded SizeY should be 5', 5, Padded.SizeY);
    
    // Center should have original values
    AssertEquals('Center value should be 1.0', 1.0, Padded[1, 1, 0], 0.0001);
    
    // Padding areas should be 0
    AssertEquals('Top-left corner should be 0.0', 0.0, Padded[0, 0, 0], 0.0001);
    AssertEquals('Bottom-right corner should be 0.0', 0.0, Padded[4, 4, 0], 0.0001);
  finally
    Original.Free;
    Padded.Free;
  end;
end;

procedure TTestNeuralVolume.TestVolumeTranspose;
var
  Original, Transposed: TNNetVolume;
begin
  // Test transpose of X and D dimensions
  Original := TNNetVolume.Create(4, 1, 2);
  Transposed := TNNetVolume.Create(1, 1, 1);
  try
    // Set some values
    Original[0, 0, 0] := 1.0;
    Original[1, 0, 0] := 2.0;
    Original[2, 0, 0] := 3.0;
    Original[3, 0, 0] := 4.0;
    Original[0, 0, 1] := 5.0;
    Original[1, 0, 1] := 6.0;
    Original[2, 0, 1] := 7.0;
    Original[3, 0, 1] := 8.0;
    
    Transposed.CopyTransposingXD(Original);
    
    // After transposing X and D, SizeX and Depth should be swapped
    AssertEquals('Transposed SizeX should be 2', 2, Transposed.SizeX);
    AssertEquals('Transposed Depth should be 4', 4, Transposed.Depth);
  finally
    Original.Free;
    Transposed.Free;
  end;
end;

procedure TTestNeuralVolume.TestVolumeNormalization;
var
  V: TNNetVolume;
  Magnitude: TNeuralFloat;
begin
  V := TNNetVolume.Create(4, 1, 1);
  try
    V.Raw[0] := 3.0;
    V.Raw[1] := 4.0;
    V.Raw[2] := 0.0;
    V.Raw[3] := 0.0;
    
    Magnitude := V.GetMagnitude();
    // Magnitude of [3, 4, 0, 0] = 5
    AssertEquals('Magnitude should be 5.0', 5.0, Magnitude, 0.0001);
  finally
    V.Free;
  end;
end;

procedure TTestNeuralVolume.TestVolumeMagnitude;
var
  V: TNNetVolume;
  Magnitude: TNeuralFloat;
begin
  V := TNNetVolume.Create(3, 1, 1);
  try
    V.Raw[0] := 1.0;
    V.Raw[1] := 2.0;
    V.Raw[2] := 2.0;
    
    Magnitude := V.GetMagnitude();
    // Magnitude of [1, 2, 2] = sqrt(1 + 4 + 4) = 3
    AssertEquals('Magnitude should be 3.0', 3.0, Magnitude, 0.0001);
  finally
    V.Free;
  end;
end;

procedure TTestNeuralVolume.TestVolumeEntropy;
var
  V: TNNetVolume;
  Entropy: TNeuralFloat;
begin
  V := TNNetVolume.Create(4, 1, 1);
  try
    // Uniform distribution
    V.Raw[0] := 0.25;
    V.Raw[1] := 0.25;
    V.Raw[2] := 0.25;
    V.Raw[3] := 0.25;
    
    Entropy := V.GetEntropy();
    // Entropy of uniform distribution over 4 elements = log2(4) = 2
    AssertTrue('Entropy of uniform dist should be around 2', Abs(Entropy - 2.0) < 0.1);
    
    // Deterministic distribution
    V.Raw[0] := 1.0;
    V.Raw[1] := 0.0;
    V.Raw[2] := 0.0;
    V.Raw[3] := 0.0;
    
    Entropy := V.GetEntropy();
    // Entropy of deterministic distribution = 0
    AssertEquals('Entropy of deterministic dist should be 0', 0.0, Entropy, 0.001);
  finally
    V.Free;
  end;
end;

procedure TTestNeuralVolume.TestVolumeOneHotEncoding;
var
  V: TNNetVolume;
  Tokens: array[0..2] of integer;
begin
  // OneHotEncoding: SizeX = number of tokens, Depth = vocab size
  // It sets Self[TokenIndex, 0, TokenValue] := 1
  V := TNNetVolume.Create(3, 1, 10);  // 3 tokens, vocab size 10
  try
    Tokens[0] := 1;
    Tokens[1] := 5;
    Tokens[2] := 8;
    
    V.Fill(0);
    V.OneHotEncoding(Tokens);
    
    // Check that the correct positions are set
    // Token 0 has value 1, so V[0, 0, 1] = 1
    AssertEquals('V[0,0,1] should be 1.0', 1.0, V[0, 0, 1], 0.0001);
    // Token 1 has value 5, so V[1, 0, 5] = 1
    AssertEquals('V[1,0,5] should be 1.0', 1.0, V[1, 0, 5], 0.0001);
    // Token 2 has value 8, so V[2, 0, 8] = 1
    AssertEquals('V[2,0,8] should be 1.0', 1.0, V[2, 0, 8], 0.0001);
    // Other positions should be 0
    AssertEquals('V[0,0,0] should be 0.0', 0.0, V[0, 0, 0], 0.0001);
    AssertEquals('V[1,0,0] should be 0.0', 0.0, V[1, 0, 0], 0.0001);
  finally
    V.Free;
  end;
end;

procedure TTestNeuralVolume.TestVolumePositionalEncoding;
var
  V: TNNetVolume;
begin
  V := TNNetVolume.Create(8, 1, 16);
  try
    V.Fill(0);
    V.PositionalEncoding();
    
    // Positional encoding should produce values in [-1, 1] range
    AssertTrue('Max value should be <= 1', V.GetMax() <= 1.001);
    AssertTrue('Min value should be >= -1', V.GetMin() >= -1.001);
    // Should have non-zero values
    AssertTrue('Should have non-zero values', V.GetSumAbs() > 0);
  finally
    V.Free;
  end;
end;

procedure TTestNeuralVolume.TestVolumeColorConversions;
var
  V: TNNetVolume;
begin
  V := TNNetVolume.Create(1, 1, 3);
  try
    // Test RGB to HSV and back
    // Pure red: RGB(1, 0, 0)
    V[0, 0, 0] := 1.0;
    V[0, 0, 1] := 0.0;
    V[0, 0, 2] := 0.0;
    
    V.RgbToHsv();
    // After conversion, we should have HSV values
    AssertTrue('HSV converted values should be valid', V.GetSum() >= 0);
    
    V.HsvToRgb();
    // After conversion back, should approximately be red
    AssertEquals('R should be approximately 1.0', 1.0, V[0, 0, 0], 0.01);
    AssertEquals('G should be approximately 0.0', 0.0, V[0, 0, 1], 0.01);
    AssertEquals('B should be approximately 0.0', 0.0, V[0, 0, 2], 0.01);
  finally
    V.Free;
  end;
end;

procedure TTestNeuralVolume.TestVolumeGaussianNoise;
var
  V: TNNetVolume;
  SumBefore, SumAfter: TNeuralFloat;
begin
  V := TNNetVolume.Create(100, 1, 1);
  try
    V.Fill(5.0);
    SumBefore := V.GetSum();
    
    V.AddGaussianNoise(0.1);
    SumAfter := V.GetSum();
    
    // Sum should change after adding noise
    AssertTrue('Values should change after adding noise', Abs(SumAfter - SumBefore) > 0.001);
    // Average should still be around 5.0 with small noise
    AssertTrue('Average should be approximately 5.0', Abs(V.GetAvg() - 5.0) < 1.0);
  finally
    V.Free;
  end;
end;

procedure TTestNeuralVolume.TestVolumeCopyResizing;
var
  Original, Resized: TNNetVolume;
begin
  Original := TNNetVolume.Create(4, 4, 1);
  Resized := TNNetVolume.Create(1, 1, 1);
  try
    Original.Fill(1.0);
    
    Resized.CopyResizing(Original, 8, 8);
    
    AssertEquals('Resized SizeX should be 8', 8, Resized.SizeX);
    AssertEquals('Resized SizeY should be 8', 8, Resized.SizeY);
  finally
    Original.Free;
    Resized.Free;
  end;
end;

procedure TTestNeuralVolume.TestVolumeCopyCropping;
var
  Original, Cropped: TNNetVolume;
begin
  Original := TNNetVolume.Create(8, 8, 1);
  Cropped := TNNetVolume.Create(1, 1, 1);
  try
    Original.Fill(1.0);
    // Set a specific value in the center
    Original[3, 3, 0] := 5.0;
    
    Cropped.CopyCropping(Original, 2, 2, 4, 4);
    
    AssertEquals('Cropped SizeX should be 4', 4, Cropped.SizeX);
    AssertEquals('Cropped SizeY should be 4', 4, Cropped.SizeY);
    // The value at (3,3) in original should be at (1,1) in cropped
    AssertEquals('Cropped value at (1,1) should be 5.0', 5.0, Cropped[1, 1, 0], 0.0001);
  finally
    Original.Free;
    Cropped.Free;
  end;
end;

procedure TTestNeuralVolume.TestVolumeShift;
var
  V: TNNetVolume;
begin
  V := TNNetVolume.Create(4, 1, 1);
  try
    V.Raw[0] := 1.0;
    V.Raw[1] := 2.0;
    V.Raw[2] := 3.0;
    V.Raw[3] := 4.0;
    
    V.ShiftRight(1);
    
    // After shifting right by 1: [0, 1, 2, 3]
    AssertEquals('Position 0 should be 0 after shift', 0.0, V.Raw[0], 0.0001);
    AssertEquals('Position 1 should be 1.0', 1.0, V.Raw[1], 0.0001);
    AssertEquals('Position 2 should be 2.0', 2.0, V.Raw[2], 0.0001);
    AssertEquals('Position 3 should be 3.0', 3.0, V.Raw[3], 0.0001);
  finally
    V.Free;
  end;
end;

procedure TTestNeuralVolume.TestVolumeRawPosAndPtr;
var
  V: TNNetVolume;
  Pos: integer;
  Ptr: pointer;
begin
  V := TNNetVolume.Create(4, 4, 3);
  try
    // Test GetRawPos - storage is (x, y, depth) in interleaved format
    // The data is stored as: for each (x,y) position, all depths are consecutive
    Pos := V.GetRawPos(0, 0, 0);
    AssertEquals('RawPos(0,0,0) should be 0', 0, Pos);
    
    // Depth is interleaved, so (0,0,1) is at position 1
    Pos := V.GetRawPos(0, 0, 1);
    AssertEquals('RawPos(0,0,1) should be 1', 1, Pos);
    
    // Position (1,0,0) is at position Depth (which is 3)
    Pos := V.GetRawPos(1, 0, 0);
    AssertEquals('RawPos(1,0,0) should be 3 (depth)', 3, Pos);
    
    // Test GetRawPtr
    Ptr := V.GetRawPtr(0, 0, 0);
    AssertTrue('RawPtr should not be nil', Ptr <> nil);
    
    Ptr := V.GetRawPtr();
    AssertTrue('RawPtr() should not be nil', Ptr <> nil);
  finally
    V.Free;
  end;
end;

procedure TTestNeuralVolume.TestVolumeDepthOperations;
var
  V: TNNetVolume;
  SumD0, AvgD0: TNeuralFloat;
begin
  V := TNNetVolume.Create(2, 2, 3);
  try
    // Fill each depth with different values
    V.FillAtDepth(0, 2.0);
    V.FillAtDepth(1, 4.0);
    V.FillAtDepth(2, 6.0);
    
    // Test SumAtDepth
    SumD0 := V.SumAtDepth(0);
    AssertEquals('Sum at depth 0 should be 8.0 (4 * 2.0)', 8.0, SumD0, 0.0001);
    
    // Test AvgAtDepth
    AvgD0 := V.AvgAtDepth(0);
    AssertEquals('Avg at depth 0 should be 2.0', 2.0, AvgD0, 0.0001);
    
    // Test AddAtDepth
    V.AddAtDepth(0, 1.0);
    AssertEquals('After AddAtDepth, value should be 3.0', 3.0, V[0, 0, 0], 0.0001);
    
    // Test MulAtDepth
    V.MulAtDepth(1, 0.5);
    AssertEquals('After MulAtDepth, value should be 2.0', 2.0, V[0, 0, 1], 0.0001);
  finally
    V.Free;
  end;
end;

initialization
  RegisterTest(TTestNeuralVolume);

end.
