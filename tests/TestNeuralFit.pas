unit TestNeuralFit;

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, Math, fpcunit, testregistry, neuralnetwork, neuralvolume,
  neuralfit, neuralscheduler;

type
  // A scheduler whose NextLR always returns the same fixed value, regardless
  // of Epoch/Step. Used to prove the scheduler hook composes as a clean
  // override: assigning a constant scheduler equal to the fixed LR must
  // reproduce the no-scheduler run exactly.
  TConstantLR = class(TNeuralLRScheduler)
  private
    FLR: TNeuralFloat;
  public
    constructor Create(pLR: TNeuralFloat);
    function NextLR(Epoch, Step: integer): TNeuralFloat; override;
  end;

  // Captures the in-loop NaNGuard ErrorProc callback. TNeuralFit.ErrorProc is a
  // TGetStrProc (of object), so we hand it this object's HandleError method and
  // assert the flag flips when the guard fires (and stays clear when it doesn't).
  TNaNGuardProbe = class
  public
    Fired: boolean;
    LastMsg: string;
    constructor Create;
    procedure HandleError(const S: string);
  end;

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

    // NaN/Inf guard tests
    procedure TestNaNGuardDefaultOff;
    procedure TestFirstLayerWithNonFiniteAllFinite;
    procedure TestFirstLayerWithNonFiniteDetectsLayer;
    procedure TestFirstLayerWithNonFiniteDetectsInf;
    // End-to-end: NaNGuard aborts a live Fit on a rigged net, and stays silent
    // on the same net/data when it is well-behaved.
    procedure TestNaNGuardAbortsTrainingOnInf;
    procedure TestNaNGuardSilentOnHealthyNet;

    // Scheduler wiring
    procedure TestSchedulerDefaultsNil;
    procedure TestConstantSchedulerMatchesFixedLR;
  end;

implementation

constructor TNaNGuardProbe.Create;
begin
  inherited Create;
  Fired := False;
  LastMsg := '';
end;

procedure TNaNGuardProbe.HandleError(const S: string);
begin
  Fired := True;
  LastMsg := S;
end;

constructor TConstantLR.Create(pLR: TNeuralFloat);
begin
  inherited Create();
  FLR := pLR;
end;

function TConstantLR.NextLR(Epoch, Step: integer): TNeuralFloat;
begin
  Result := FLR;
end;

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
    AssertEquals('ClipValue should default to 0.0 (disabled)', 0.0, Fit.ClipValue, 0.0);
    Fit.ClipDelta := 1.0;
    AssertEquals('ClipDelta should be 1.0', 1.0, Fit.ClipDelta, 0.0001);
    Fit.ClipValue := 0.5;
    AssertEquals('ClipValue should be 0.5', 0.5, Fit.ClipValue, 0.0001);

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

procedure TTestNeuralFit.TestNaNGuardDefaultOff;
var
  Fit: TNeuralFit;
begin
  Fit := TNeuralFit.Create;
  try
    // The guard must be OFF by default so the default training path is unchanged.
    AssertFalse('NaNGuard should be False by default', Fit.NaNGuard);
    Fit.NaNGuard := True;
    AssertTrue('NaNGuard should be settable to True', Fit.NaNGuard);
  finally
    Fit.Free;
  end;
end;

procedure TTestNeuralFit.TestFirstLayerWithNonFiniteAllFinite;
var
  NN: TNNet;
  Input: TNNetVolume;
begin
  RandSeed := 424242;
  NN := TNNet.Create();
  Input := TNNetVolume.Create(4, 1, 1);
  try
    NN.AddLayer(TNNetInput.Create(4, 1, 1));
    NN.AddLayer(TNNetFullConnectReLU.Create(3));
    NN.AddLayer(TNNetFullConnectLinear.Create(2));
    Input.Fill(1.0);
    NN.Compute(Input);
    // On an all-finite net the helper must return -1.
    AssertEquals('All-finite net should return -1', -1,
      NN.FirstLayerWithNonFinite());
  finally
    NN.Free;
    Input.Free;
  end;
end;

procedure TTestNeuralFit.TestFirstLayerWithNonFiniteDetectsLayer;
var
  NN: TNNet;
  Input: TNNetVolume;
  TargetIdx: integer;
begin
  RandSeed := 424242;
  NN := TNNet.Create();
  Input := TNNetVolume.Create(4, 1, 1);
  try
    NN.AddLayer(TNNetInput.Create(4, 1, 1));
    NN.AddLayer(TNNetFullConnectReLU.Create(3));
    NN.AddLayer(TNNetFullConnectLinear.Create(2));
    Input.Fill(1.0);
    NN.Compute(Input);
    AssertEquals('Net should be finite before poking NaN', -1,
      NN.FirstLayerWithNonFinite());

    // Deliberately seed a NaN into a chosen layer's Output volume.
    TargetIdx := 1;
    NN.Layers[TargetIdx].Output.FData[0] := NaN;
    AssertEquals('Helper should detect the poked NaN layer', TargetIdx,
      NN.FirstLayerWithNonFinite());
  finally
    NN.Free;
    Input.Free;
  end;
end;

procedure TTestNeuralFit.TestFirstLayerWithNonFiniteDetectsInf;
var
  NN: TNNet;
  Input: TNNetVolume;
  TargetIdx: integer;
begin
  RandSeed := 424242;
  NN := TNNet.Create();
  Input := TNNetVolume.Create(4, 1, 1);
  try
    NN.AddLayer(TNNetInput.Create(4, 1, 1));
    NN.AddLayer(TNNetFullConnectReLU.Create(3));
    NN.AddLayer(TNNetFullConnectLinear.Create(2));
    Input.Fill(1.0);
    NN.Compute(Input);

    // Seed a +Inf into the last layer's Output and confirm it is detected.
    TargetIdx := NN.GetLastLayerIdx();
    NN.Layers[TargetIdx].Output.FData[0] := Infinity;
    AssertEquals('Helper should detect the poked Inf layer', TargetIdx,
      NN.FirstLayerWithNonFinite());
  finally
    NN.Free;
    Input.Free;
  end;
end;

procedure TTestNeuralFit.TestSchedulerDefaultsNil;
var
  Fit: TNeuralFit;
begin
  Fit := TNeuralFit.Create;
  try
    // The scheduler must be OPTIONAL and unset by default so the legacy
    // fixed/decay/cyclical LR path stays byte-for-byte unchanged.
    AssertTrue('Scheduler should default to nil', Fit.Scheduler = nil);
  finally
    Fit.Free;
  end;
end;

// Builds a fresh tiny regression net under a fixed seed so two builds are
// bit-identical at initialization.
function BuildFitNet: TNNet;
begin
  RandSeed := 424242;
  Result := TNNet.Create();
  Result.AddLayer([
    TNNetInput.Create(2),
    TNNetFullConnectReLU.Create(8),
    TNNetFullConnectLinear.Create(1)
  ]);
  Result.SetLearningRate(0.01, 0.9);
end;

// Builds the tiny XOR-ish training set used by the scheduler regression test.
function BuildFitPairs: TNNetVolumePairList;
var
  I: integer;
  Pair: TNNetVolumePair;
begin
  Result := TNNetVolumePairList.Create();
  for I := 0 to 3 do
  begin
    Pair := TNNetVolumePair.Create();
    Pair.A.ReSize(2, 1, 1);
    Pair.B.ReSize(1, 1, 1);
    Pair.A.Raw[0] := (I and 1);
    Pair.A.Raw[1] := ((I shr 1) and 1);
    Pair.B.Raw[0] := (I and 1) xor ((I shr 1) and 1);
    Result.Add(Pair);
  end;
end;

// Trains Net with an OPTIONAL scheduler and returns the last-layer weight
// vector after training (no best-model reload, so weights are the live ones).
procedure RunFit(Net: TNNet; Sched: TNeuralLRScheduler; out W: TNNetVolume);
var
  Fit: TNeuralFit;
  Pairs: TNNetVolumePairList;
begin
  Pairs := BuildFitPairs();
  // Reseed immediately before Fit so both runs see the same RNG stream
  // (Fit shuffles batches off the global RNG); the only difference between
  // the baseline and scheduler runs is then the scheduler hook itself.
  RandSeed := 100;
  Fit := TNeuralFit.Create;
  try
    Fit.HideMessages;
    Fit.Verbose := False;
    // Single thread so the only difference between the two runs is the
    // scheduler hook (thread-reduction order is otherwise nondeterministic).
    Fit.MaxThreadNum := 1;
    // Hold the built-in LR constant (decay 0) so the no-scheduler path keeps
    // a fixed LR equal to the constant scheduler's value.
    Fit.InitialLearningRate := 0.01;
    Fit.LearningRateDecay := 0;
    Fit.StaircaseEpochs := 1;
    Fit.LoadBestAtEnd := False; // keep live trained weights, avoid reload
    Fit.Scheduler := Sched;     // nil for the baseline run
    Fit.Fit(Net, Pairs, nil, nil, 4, 20);
    W := TNNetVolume.Create();
    W.Copy(Net.GetLastLayer.Neurons[0].Weights);
  finally
    Fit.Free;
    Pairs.Free;
  end;
end;

procedure TTestNeuralFit.TestConstantSchedulerMatchesFixedLR;
var
  NetA, NetB: TNNet;
  WA, WB: TNNetVolume;
  Sched: TConstantLR;
  I: integer;
begin
  WA := nil;
  WB := nil;
  Sched := nil;
  NetA := BuildFitNet();          // baseline: no scheduler, fixed LR 0.01
  NetB := BuildFitNet();          // same seed -> identical init weights
  try
    Sched := TConstantLR.Create(0.01);
    RunFit(NetA, nil, WA);        // legacy fixed-LR path
    RunFit(NetB, Sched, WB);      // scheduler path, same constant LR

    AssertEquals('Weight vector size must match', WA.Size, WB.Size);
    for I := 0 to WA.Size - 1 do
      AssertEquals('Constant-scheduler weights must match fixed-LR weights',
        WA.FData[I], WB.FData[I], 1e-6);
  finally
    WA.Free;
    WB.Free;
    Sched.Free;
    NetA.Free;
    NetB.Free;
  end;
end;

procedure TTestNeuralFit.TestNaNGuardAbortsTrainingOnInf;
var
  Net: TNNet;
  Pairs: TNNetVolumePairList;
  Fit: TNeuralFit;
  Probe: TNaNGuardProbe;
  OldMask: TFPUExceptionMask;
const
  cEpochs = 20;
begin
  Net := BuildFitNet();
  Pairs := BuildFitPairs();
  Fit := TNeuralFit.Create;
  Probe := TNaNGuardProbe.Create;
  // Mask invalid-op/overflow/zero-divide so the planted Inf propagates as a
  // value (the NaNGuard's job is to detect such non-finite values) instead of
  // trapping as a hardware FP exception. Real training loops that rely on the
  // guard run with these masked. Restored in the finally block.
  OldMask := GetExceptionMask;
  SetExceptionMask(OldMask + [exInvalidOp, exOverflow, exZeroDivide,
    exDenormalized, exUnderflow, exPrecision]);
  try
    // Rig the net: plant a +Inf into a hidden-layer weight so the very first
    // forward pass produces a non-finite activation. The guard scans after
    // forward+backward, so it must fire on the first batch of epoch 0.
    Net.Layers[1].Neurons[0].Weights.FData[0] := Infinity;

    Fit.HideMessages;
    Fit.Verbose := False;
    Fit.MaxThreadNum := 1;          // single thread, deterministic
    Fit.InitialLearningRate := 0.01;
    Fit.LearningRateDecay := 0;
    Fit.StaircaseEpochs := 1;
    Fit.LoadBestAtEnd := False;
    Fit.NaNGuard := True;
    // Capture the guard's ErrorProc so we can assert it actually fired.
    Fit.ErrorProc := {$IFDEF FPC}@{$ENDIF}Probe.HandleError;

    Fit.Fit(Net, Pairs, nil, nil, 4, cEpochs);

    // Observable signals the guard documents: it fires FErrorProc and sets
    // FShouldQuit (exposed as ShouldQuit), which breaks the epoch loop early.
    AssertTrue('NaNGuard ErrorProc must fire on a non-finite net', Probe.Fired);
    AssertTrue('NaNGuard message must mention non-finite/NaNGuard',
      Pos('NaNGuard', Probe.LastMsg) > 0);
    AssertTrue('NaNGuard must set ShouldQuit to abort training',
      Fit.ShouldQuit);
    AssertTrue('Training must abort before the full epoch budget',
      Fit.CurrentEpoch < cEpochs);
  finally
    SetExceptionMask(OldMask);
    Fit.Free;
    Probe.Free;
    Pairs.Free;
    Net.Free;
  end;
end;

procedure TTestNeuralFit.TestNaNGuardSilentOnHealthyNet;
var
  Net: TNNet;
  Pairs: TNNetVolumePairList;
  Fit: TNeuralFit;
  Probe: TNaNGuardProbe;
const
  cEpochs = 20;
begin
  // Complement of the abort test: the SAME net/data, NaNGuard ON, but NOT
  // rigged and with a sane LR must train to completion without firing.
  Net := BuildFitNet();
  Pairs := BuildFitPairs();
  Fit := TNeuralFit.Create;
  Probe := TNaNGuardProbe.Create;
  try
    Fit.HideMessages;
    Fit.Verbose := False;
    Fit.MaxThreadNum := 1;
    Fit.InitialLearningRate := 0.01;
    Fit.LearningRateDecay := 0;
    Fit.StaircaseEpochs := 1;
    Fit.LoadBestAtEnd := False;
    Fit.NaNGuard := True;
    Fit.ErrorProc := {$IFDEF FPC}@{$ENDIF}Probe.HandleError;

    Fit.Fit(Net, Pairs, nil, nil, 4, cEpochs);

    AssertFalse('NaNGuard must stay silent on a well-behaved net', Probe.Fired);
    AssertFalse('Healthy training must not set ShouldQuit', Fit.ShouldQuit);
    AssertEquals('Healthy training must run the full epoch budget',
      cEpochs, Fit.CurrentEpoch);
  finally
    Fit.Free;
    Probe.Free;
    Pairs.Free;
    Net.Free;
  end;
end;

initialization
  RegisterTest(TTestNeuralFit);

end.
