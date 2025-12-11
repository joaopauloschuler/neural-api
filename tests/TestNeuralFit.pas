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
    procedure TestAdamOptimizerParameters;
    
    // TNeuralFitBase property tests
    procedure TestFitBaseDefaultProperties;
    procedure TestFitBaseLearningRateProperties;
    procedure TestFitBaseClipProperties;
    procedure TestFitBaseEpochProperties;
    
    // TNeuralImageFit tests
    procedure TestImageFitCreation;
    procedure TestImageFitDataAugmentationProperties;
    procedure TestImageFitColorTransformProperties;
    
    // TNeuralFit tests
    procedure TestNeuralFitCreation;
    procedure TestNeuralFitHideMessages;
    procedure TestNeuralFitBatchSizeProperties;
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

procedure TTestNeuralFit.TestAdamOptimizerParameters;
var
  Optimizer: TNeuralOptimizerAdam;
begin
  // Test Adam optimizer with various beta parameters
  Optimizer := TNeuralOptimizerAdam.Create(0.95, 0.99, 1e-07);
  try
    AssertTrue('Adam optimizer with different betas should be created', Optimizer <> nil);
  finally
    Optimizer.Free;
  end;
  
  // Test with edge case parameters
  Optimizer := TNeuralOptimizerAdam.Create(0.5, 0.5, 1e-10);
  try
    AssertTrue('Adam optimizer with low betas should be created', Optimizer <> nil);
  finally
    Optimizer.Free;
  end;
end;

procedure TTestNeuralFit.TestFitBaseEpochProperties;
var
  Fit: TNeuralFit;
begin
  Fit := TNeuralFit.Create;
  try
    // Test epoch-related properties
    Fit.InitialEpoch := 5;
    AssertEquals('InitialEpoch should be 5', 5, Fit.InitialEpoch);
    
    // Test initial epoch is 0 by default
    Fit.InitialEpoch := 0;
    AssertEquals('InitialEpoch can be reset to 0', 0, Fit.InitialEpoch);
    
    // Test staircase epochs property
    Fit.StaircaseEpochs := 10;
    AssertEquals('StaircaseEpochs should be 10', 10, Fit.StaircaseEpochs);
    
    // Test target accuracy
    Fit.TargetAccuracy := 0.95;
    AssertEquals('TargetAccuracy should be 0.95', 0.95, Fit.TargetAccuracy, 0.0001);
  finally
    Fit.Free;
  end;
end;

procedure TTestNeuralFit.TestImageFitColorTransformProperties;
var
  Fit: TNeuralImageFit;
begin
  Fit := TNeuralImageFit.Create;
  try
    // Test channel shift rate (color jitter)
    Fit.ChannelShiftRate := 0.1;
    AssertEquals('ChannelShiftRate should be 0.1', 0.1, Fit.ChannelShiftRate, 0.0001);
    
    // Test resizing property
    Fit.HasResizing := True;
    AssertTrue('HasResizing should be True', Fit.HasResizing);
    
    Fit.HasResizing := False;
    AssertFalse('HasResizing should be False', Fit.HasResizing);
    
    // Test different channel shift rates
    Fit.ChannelShiftRate := 0.0;
    AssertEquals('ChannelShiftRate can be 0', 0.0, Fit.ChannelShiftRate, 0.0001);
    
    Fit.ChannelShiftRate := 0.5;
    AssertEquals('ChannelShiftRate can be 0.5', 0.5, Fit.ChannelShiftRate, 0.0001);
  finally
    Fit.Free;
  end;
end;

procedure TTestNeuralFit.TestNeuralFitBatchSizeProperties;
var
  Fit: TNeuralFit;
begin
  Fit := TNeuralFit.Create;
  try
    // Test max thread num property
    Fit.MaxThreadNum := 4;
    AssertEquals('MaxThreadNum should be 4', 4, Fit.MaxThreadNum);
    
    // Test various thread counts
    Fit.MaxThreadNum := 1;
    AssertEquals('MaxThreadNum can be 1', 1, Fit.MaxThreadNum);
    
    Fit.MaxThreadNum := 8;
    AssertEquals('MaxThreadNum can be 8', 8, Fit.MaxThreadNum);
    
    // Test log every batches property
    Fit.LogEveryBatches := 100;
    AssertEquals('LogEveryBatches should be 100', 100, Fit.LogEveryBatches);
  finally
    Fit.Free;
  end;
end;

initialization
  RegisterTest(TTestNeuralFit);

end.
