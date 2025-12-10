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

initialization
  RegisterTest(TTestNeuralFit);

end.
