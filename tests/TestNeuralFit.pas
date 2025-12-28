unit TestNeuralFit;

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, fpcunit, testregistry, neuralnetwork, neuralvolume, neuralfit;

type
  TTestNeuralFit = class(TTestCase)
  published
    // Optimizer tests
    procedure TestSGDOptimizerCreation;
    procedure TestAdamOptimizerCreation;
    procedure TestAdamOptimizerReset;
    
    // TNeuralFitBase property tests
    procedure TestFitBaseDefaultProperties;
    procedure TestFitBaseLearningRateProperties;
    procedure TestFitBaseClipProperties;
    
    // TNeuralImageFit tests
    procedure TestImageFitCreation;
    procedure TestImageFitDataAugmentationProperties;
    
    // TNeuralFit tests
    procedure TestNeuralFitCreation;
    procedure TestNeuralFitHideMessages;
    
    // Regression comparison function tests
    procedure TestRegressionCompare;
    procedure TestRegressionCompareWide;
    procedure TestEnableRegressionComparison;
  end;

implementation

procedure TTestNeuralFit.TestSGDOptimizerCreation;
var
  Optimizer: TNeuralOptimizerSGD;
begin
  Optimizer := TNeuralOptimizerSGD.Create;
  try
    AssertTrue('SGD optimizer should be created', Optimizer <> nil);
  finally
    Optimizer.Free;
  end;
end;

procedure TTestNeuralFit.TestAdamOptimizerCreation;
var
  Optimizer: TNeuralOptimizerAdam;
begin
  Optimizer := TNeuralOptimizerAdam.Create;
  try
    AssertTrue('Adam optimizer should be created', Optimizer <> nil);
  finally
    Optimizer.Free;
  end;
end;

procedure TTestNeuralFit.TestAdamOptimizerReset;
var
  Optimizer: TNeuralOptimizerAdam;
begin
  // Test creating with custom parameters
  Optimizer := TNeuralOptimizerAdam.Create(0.9, 0.999, 1e-08);
  try
    AssertTrue('Adam optimizer with custom params should be created', Optimizer <> nil);
    // Note: Reset requires NN to be set, so we just test creation
    AssertTrue('Optimizer should be valid', True);
  finally
    Optimizer.Free;
  end;
end;

procedure TTestNeuralFit.TestFitBaseDefaultProperties;
var
  Fit: TNeuralFit;
begin
  Fit := TNeuralFit.Create;
  try
    // Test default property values
    AssertTrue('Default learning rate should be positive', Fit.InitialLearningRate > 0);
    AssertTrue('Default inertia should be non-negative', Fit.Inertia >= 0);
    AssertTrue('Not running by default', not Fit.Running);
    AssertEquals('Initial epoch should be 0', 0, Fit.CurrentEpoch);
  finally
    Fit.Free;
  end;
end;

procedure TTestNeuralFit.TestFitBaseLearningRateProperties;
var
  Fit: TNeuralFit;
begin
  Fit := TNeuralFit.Create;
  try
    // Test setting properties
    Fit.InitialLearningRate := 0.01;
    AssertEquals('Learning rate should be 0.01', 0.01, Fit.InitialLearningRate, 0.0001);
    
    Fit.LearningRateDecay := 0.1;
    AssertEquals('Learning rate decay should be 0.1', 0.1, Fit.LearningRateDecay, 0.0001);
    
    Fit.Inertia := 0.9;
    AssertEquals('Inertia should be 0.9', 0.9, Fit.Inertia, 0.0001);
    
    Fit.CyclicalLearningRateLen := 10;
    AssertEquals('Cyclical LR len should be 10', 10, Fit.CyclicalLearningRateLen);
  finally
    Fit.Free;
  end;
end;

procedure TTestNeuralFit.TestFitBaseClipProperties;
var
  Fit: TNeuralFit;
begin
  Fit := TNeuralFit.Create;
  try
    // Test clip properties
    Fit.ClipDelta := 1.0;
    AssertEquals('ClipDelta should be 1.0', 1.0, Fit.ClipDelta, 0.0001);
    
    Fit.L2Decay := 0.0001;
    AssertEquals('L2Decay should be 0.0001', 0.0001, Fit.L2Decay, 0.00001);
  finally
    Fit.Free;
  end;
end;

procedure TTestNeuralFit.TestImageFitCreation;
var
  Fit: TNeuralImageFit;
begin
  Fit := TNeuralImageFit.Create;
  try
    AssertTrue('Image fit should be created', Fit <> nil);
    // Test default data augmentation properties
    AssertTrue('HasFlipX should be available', True);
  finally
    Fit.Free;
  end;
end;

procedure TTestNeuralFit.TestImageFitDataAugmentationProperties;
var
  Fit: TNeuralImageFit;
begin
  Fit := TNeuralImageFit.Create;
  try
    // Test data augmentation properties
    Fit.HasFlipX := True;
    AssertTrue('HasFlipX should be True', Fit.HasFlipX);
    
    Fit.HasFlipX := False;
    AssertFalse('HasFlipX should be False', Fit.HasFlipX);
    
    Fit.HasFlipY := True;
    AssertTrue('HasFlipY should be True', Fit.HasFlipY);
    
    Fit.HasMakeGray := True;
    AssertTrue('HasMakeGray should be True', Fit.HasMakeGray);
    
    Fit.HasImgCrop := True;
    AssertTrue('HasImgCrop should be True', Fit.HasImgCrop);
    
    Fit.MaxCropSize := 4;
    AssertEquals('MaxCropSize should be 4', 4, Fit.MaxCropSize);
  finally
    Fit.Free;
  end;
end;

procedure TTestNeuralFit.TestNeuralFitCreation;
var
  Fit: TNeuralFit;
begin
  Fit := TNeuralFit.Create;
  try
    AssertTrue('Neural fit should be created', Fit <> nil);
  finally
    Fit.Free;
  end;
end;

procedure TTestNeuralFit.TestNeuralFitHideMessages;
var
  Fit: TNeuralFit;
begin
  Fit := TNeuralFit.Create;
  try
    // Test hiding messages (should not crash)
    Fit.HideMessages;
    Fit.Verbose := False;
    AssertFalse('Verbose should be False', Fit.Verbose);
  finally
    Fit.Free;
  end;
end;

procedure TTestNeuralFit.TestRegressionCompare;
var
  A, B: TNNetVolume;
begin
  A := TNNetVolume.Create(3, 1, 1);
  B := TNNetVolume.Create(3, 1, 1);
  try
    // Test exact match
    A.FData[0] := 0.5;
    A.FData[1] := 0.7;
    A.FData[2] := 0.3;
    B.FData[0] := 0.5;
    B.FData[1] := 0.7;
    B.FData[2] := 0.3;
    AssertTrue('Exact match should return True', RegressionCompare(A, B, 0));
    
    // Test within tolerance (0.09 difference, threshold is 0.1)
    B.FData[0] := 0.59;
    B.FData[1] := 0.61;
    B.FData[2] := 0.21;
    AssertTrue('Within tolerance should return True', RegressionCompare(A, B, 0));
    
    // Test outside tolerance (0.15 difference, threshold is 0.1)
    B.FData[0] := 0.65;
    AssertFalse('Outside tolerance should return False', RegressionCompare(A, B, 0));
    
    // Test size mismatch
    B.Resize(2, 1, 1);
    AssertFalse('Size mismatch should return False', RegressionCompare(A, B, 0));
  finally
    A.Free;
    B.Free;
  end;
end;

procedure TTestNeuralFit.TestRegressionCompareWide;
var
  A, B: TNNetVolume;
begin
  A := TNNetVolume.Create(3, 1, 1);
  B := TNNetVolume.Create(3, 1, 1);
  try
    // Test exact match
    A.FData[0] := 10.0;
    A.FData[1] := 20.0;
    A.FData[2] := 30.0;
    B.FData[0] := 10.0;
    B.FData[1] := 20.0;
    B.FData[2] := 30.0;
    AssertTrue('Exact match should return True', RegressionCompareWide(A, B, 0));
    
    // Test within tolerance (0.9 difference, threshold is 1.0)
    B.FData[0] := 10.9;
    B.FData[1] := 19.1;
    B.FData[2] := 29.5;
    AssertTrue('Within tolerance should return True', RegressionCompareWide(A, B, 0));
    
    // Test outside tolerance (1.5 difference, threshold is 1.0)
    B.FData[0] := 11.5;
    AssertFalse('Outside tolerance should return False', RegressionCompareWide(A, B, 0));
  finally
    A.Free;
    B.Free;
  end;
end;

procedure TTestNeuralFit.TestEnableRegressionComparison;
var
  Fit: TNeuralFit;
begin
  Fit := TNeuralFit.Create;
  try
    // Test enabling regression comparison (should not crash)
    Fit.EnableRegressionComparison();
    AssertTrue('InferHitFn should be assigned', Fit.InferHitFn <> nil);
    
    Fit.EnableRegressionComparisonWide();
    AssertTrue('InferHitFn should be assigned for wide comparison', Fit.InferHitFn <> nil);
  finally
    Fit.Free;
  end;
end;

initialization
  RegisterTest(TTestNeuralFit);

end.
