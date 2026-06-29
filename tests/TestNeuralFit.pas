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

  // Deterministic, counter-addressed training-data source for the gradient
  // accumulation equivalence test. It owns a fixed pool of (input,target)
  // samples and, given the within-micro-batch index and the owning Fit's
  // CurrentAccumulationStep, returns the GLOBAL sample of an effective batch.
  // This makes "N micro-batches of M" draw the SAME M*N samples in the SAME
  // order as "one true batch of M*N", which is what the equivalence proof needs.
  TAccumDataSource = class
  public
    FFit: TNeuralFitBase;
    FMicroBatch: integer;          // M, the per-micro-batch size
    FPoolIn, FPoolOut: array of TNNetVolume;
    constructor Create(pMicroBatch, pPoolSize, pInDepth, pOutDepth: integer);
    destructor Destroy; override;
    procedure GetPair(Idx, ThreadId: integer; vInput, vOutput: TNNetVolume);
  end;

  // OnAfterStep probe for the EMA tests. Fires inside the epoch loop right
  // after each optimizer step (and thus after the per-step EMA update), BEFORE
  // the end-of-Fit FAvgWeight overwrite of the live net. On every step it
  // snapshots the LIVE last-layer weights and the EMA SHADOW last-layer weights,
  // so after Fit the snapshots hold the final-step values that must satisfy the
  // EMA invariants (which the post-Fit FNN, overwritten by FAvgWeight, does not).
  TEMAStepProbe = class
  public
    FFit: TNeuralFitBase;
    Live, Shadow: TNNetVolume;   // last-step snapshots (owned)
    HadShadow: boolean;
    constructor Create(pFit: TNeuralFitBase);
    destructor Destroy; override;
    procedure AfterStep(Sender: TObject);
  end;

  TTestNeuralFit = class(TTestCase)
  private
    function RunAccumFit(MicroBatch, AccumSteps, Epochs: integer;
      out FinalLoss: TNeuralFloat): TNNetVolume;
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

    // AdamW optimizer (decoupled weight decay) tests
    procedure TestAdamWOptimizerCreation;
    procedure TestDecoupledWeightDecayExactMath;
    procedure TestAdamWZeroDecayMatchesAdam;
    procedure TestAdamWShrinksWeightsVsAdam;
    procedure TestAdamWTrainingLearns;

    // Label smoothing tests
    procedure TestLabelSmoothingDefaultZero;
    procedure TestLabelSmoothingTargetMath;
    procedure TestLabelSmoothingZeroIsBitIdentical;

    // Gradient accumulation tests
    procedure TestAccumulationStepsDefaultsToOne;
    procedure TestAccumulationStepsClampsBelowOne;
    procedure TestAccumulationEqualsBigBatch;

    // EMA shadow-weights tests
    procedure TestEMADefaultsOff;
    procedure TestEMADecayZeroEqualsLive;
    procedure TestEMAConvexBlend;
    procedure TestEMAApplyRestoreRoundTrip;
    procedure TestEMADisabledIsBitIdentical;

    // Parameter groups (PyTorch param_groups port)
    procedure TestParamGroupDefaultsOff;
    procedure TestL2DecayExcludeBiasKeepsBias;
    procedure TestL2DecayDefaultDecaysBias;
    procedure TestNormLayerLearningRateMultiplier;

    // Batch-level mixup / CutMix augmentation
    procedure TestBatchAugDefaultsOff;
    procedure TestMixedSoftTargetSumsToOne;
    procedure TestMixedSoftTargetLambdaOneIsOneHot;
    procedure TestMixedSoftTargetSameTag;
    procedure TestMixupImageFitConverges;
    procedure TestCutMixImageFitConverges;
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

// Trains Net with the given OPTIONAL optimizer (nil = default SGD path) and
// returns the last-layer weight vector after training. Mirrors RunFit but
// exercises the Optimizer property instead of the Scheduler property.
procedure RunFitWithOptimizer(Net: TNNet; Opt: TNeuralOptimizer;
  out W: TNNetVolume);
var
  Fit: TNeuralFit;
  Pairs: TNNetVolumePairList;
begin
  Pairs := BuildFitPairs();
  // Reseed immediately before Fit so runs see the same RNG stream; the only
  // difference between two runs is then the optimizer itself.
  RandSeed := 100;
  Fit := TNeuralFit.Create;
  try
    Fit.HideMessages;
    Fit.Verbose := False;
    Fit.MaxThreadNum := 1;
    Fit.InitialLearningRate := 0.01;
    Fit.LearningRateDecay := 0;
    Fit.StaircaseEpochs := 1;
    Fit.LoadBestAtEnd := False;
    if Opt <> nil then Fit.Optimizer := Opt; // caller keeps ownership
    Fit.Fit(Net, Pairs, nil, nil, 4, 20);
    W := TNNetVolume.Create();
    W.Copy(Net.GetLastLayer.Neurons[0].Weights);
  finally
    Fit.Free;
    Pairs.Free;
  end;
end;

// Mean squared error of Net over the tiny XOR-ish set from BuildFitPairs.
function FitNetMSE(Net: TNNet): TNeuralFloat;
var
  Pairs: TNNetVolumePairList;
  Output: TNNetVolume;
  I: integer;
  Diff: TNeuralFloat;
begin
  Result := 0;
  Output := TNNetVolume.Create(1, 1, 1);
  Pairs := BuildFitPairs();
  try
    for I := 0 to Pairs.Count - 1 do
    begin
      Net.Compute(Pairs[I].A);
      Net.GetOutput(Output);
      Diff := Output.FData[0] - Pairs[I].B.FData[0];
      Result := Result + Diff * Diff;
    end;
    Result := Result / Pairs.Count;
  finally
    Pairs.Free;
    Output.Free;
  end;
end;

procedure TTestNeuralFit.TestAdamWOptimizerCreation;
var
  Optimizer: TNeuralOptimizerAdamW;
begin
  Optimizer := TNeuralOptimizerAdamW.Create(0.9, 0.999, 1e-08, 0.01);
  try
    AssertTrue('AdamW optimizer should be created', Optimizer <> nil);
    AssertEquals('WeightDecay should round-trip', 0.01,
      Optimizer.WeightDecay, 1e-9);
    Optimizer.WeightDecay := 0.1;
    AssertEquals('WeightDecay should be writable', 0.1,
      Optimizer.WeightDecay, 1e-6);
  finally
    Optimizer.Free;
  end;
end;

procedure TTestNeuralFit.TestDecoupledWeightDecayExactMath;
var
  NN: TNNet;
  OldFC, OldLN: TNNetVolume;
  OldBias: TNeuralFloat;
  I: integer;
const
  cRate = 0.01; // = LearningRate * WeightDecay in an AdamW step
begin
  // Controlled single decay step: weights of a neuron-bearing trainable
  // layer must scale EXACTLY by (1 - rate); its biases and any
  // normalization layer's parameters must be left untouched.
  RandSeed := 424242;
  NN := TNNet.Create();
  OldFC := TNNetVolume.Create();
  OldLN := TNNetVolume.Create();
  try
    NN.AddLayer(TNNetInput.Create(4, 1, 1));
    NN.AddLayer(TNNetFullConnectReLU.Create(3));
    NN.AddLayer(TNNetLayerNorm.Create());
    NN.AddLayer(TNNetFullConnectLinear.Create(2));

    OldFC.Copy(NN.Layers[1].Neurons[0].Weights);
    OldBias := NN.Layers[1].Neurons[0].BiasWeight;
    OldLN.Copy(NN.Layers[2].Neurons[0].Weights);

    NN.ApplyDecoupledWeightDecay(cRate);

    // (a) decayable weights: exact (1 - rate) scaling, up to single-precision
    // rounding of the multiply (weights are singles; 1 ulp ~ 6e-8 here).
    for I := 0 to OldFC.Size - 1 do
      AssertEquals('FC weight must scale exactly by (1-rate)',
        OldFC.FData[I] * (1 - cRate),
        NN.Layers[1].Neurons[0].Weights.FData[I], 1e-7);
    // (b) biases: never decayed.
    AssertEquals('Bias must not be decayed', OldBias,
      NN.Layers[1].Neurons[0].BiasWeight, 0);
    // (c) normalization layer (TNNetIdentityWithoutL2 descendant): skipped.
    for I := 0 to OldLN.Size - 1 do
      AssertEquals('LayerNorm parameter must not be decayed',
        OldLN.FData[I], NN.Layers[2].Neurons[0].Weights.FData[I], 0);
  finally
    OldFC.Free;
    OldLN.Free;
    NN.Free;
  end;
end;

procedure TTestNeuralFit.TestAdamWZeroDecayMatchesAdam;
var
  NetA, NetB: TNNet;
  WA, WB: TNNetVolume;
  Adam: TNeuralOptimizerAdam;
  AdamW: TNeuralOptimizerAdamW;
  I: integer;
begin
  WA := nil;
  WB := nil;
  NetA := BuildFitNet();          // same seed -> identical init weights
  NetB := BuildFitNet();
  Adam := TNeuralOptimizerAdam.Create(0.9, 0.999, 1e-08);
  AdamW := TNeuralOptimizerAdamW.Create(0.9, 0.999, 1e-08, 0.0);
  try
    RunFitWithOptimizer(NetA, Adam, WA);
    RunFitWithOptimizer(NetB, AdamW, WB);

    AssertEquals('Weight vector size must match', WA.Size, WB.Size);
    // With WeightDecay = 0 the decay branch is skipped entirely, so the
    // computation sequence is identical to plain Adam: exact match.
    for I := 0 to WA.Size - 1 do
      AssertEquals('AdamW(wd=0) weights must match plain Adam exactly',
        WA.FData[I], WB.FData[I], 0);
  finally
    WA.Free;
    WB.Free;
    Adam.Free;
    AdamW.Free;
    NetA.Free;
    NetB.Free;
  end;
end;

procedure TTestNeuralFit.TestAdamWShrinksWeightsVsAdam;
var
  NetA, NetB: TNNet;
  WA, WB: TNNetVolume;
  Adam: TNeuralOptimizerAdam;
  AdamW: TNeuralOptimizerAdamW;
  NormA, NormB: TNeuralFloat;
  I: integer;
begin
  WA := nil;
  WB := nil;
  NetA := BuildFitNet();
  NetB := BuildFitNet();
  Adam := TNeuralOptimizerAdam.Create(0.9, 0.999, 1e-08);
  // Strong decay so the multiplicative shrinkage dominates the (identical)
  // gradient dynamics on this tiny run: per step factor (1 - 0.01*2.0).
  AdamW := TNeuralOptimizerAdamW.Create(0.9, 0.999, 1e-08, 2.0);
  try
    RunFitWithOptimizer(NetA, Adam, WA);
    RunFitWithOptimizer(NetB, AdamW, WB);

    NormA := 0;
    NormB := 0;
    for I := 0 to WA.Size - 1 do
      NormA := NormA + WA.FData[I] * WA.FData[I];
    for I := 0 to WB.Size - 1 do
      NormB := NormB + WB.FData[I] * WB.FData[I];
    AssertTrue('AdamW with wd>0 must end with smaller weight norm than Adam'
      + ' (' + FloatToStr(NormB) + ' vs ' + FloatToStr(NormA) + ')',
      NormB < NormA);
  finally
    WA.Free;
    WB.Free;
    Adam.Free;
    AdamW.Free;
    NetA.Free;
    NetB.Free;
  end;
end;

procedure TTestNeuralFit.TestAdamWTrainingLearns;
var
  Net: TNNet;
  W: TNNetVolume;
  AdamW: TNeuralOptimizerAdamW;
  ErrBefore, ErrAfter: TNeuralFloat;
begin
  W := nil;
  Net := BuildFitNet();
  AdamW := TNeuralOptimizerAdamW.Create(0.9, 0.999, 1e-08, 0.01);
  try
    ErrBefore := FitNetMSE(Net);
    RunFitWithOptimizer(Net, AdamW, W);
    ErrAfter := FitNetMSE(Net);
    AssertTrue('AdamW training must reduce the error ('
      + FloatToStr(ErrAfter) + ' vs ' + FloatToStr(ErrBefore) + ')',
      ErrAfter < ErrBefore);
  finally
    W.Free;
    AdamW.Free;
    Net.Free;
  end;
end;

procedure TTestNeuralFit.TestLabelSmoothingDefaultZero;
var
  Fit: TNeuralImageFit;
begin
  Fit := TNeuralImageFit.Create;
  try
    // Must default to 0 so existing training paths are unchanged.
    AssertEquals('LabelSmoothing should default to 0', 0.0,
      Fit.LabelSmoothing, 0);
    Fit.LabelSmoothing := 0.1;
    AssertEquals('LabelSmoothing should be settable', 0.1,
      Fit.LabelSmoothing, 1e-7);
  finally
    Fit.Free;
  end;
end;

procedure TTestNeuralFit.TestLabelSmoothingTargetMath;
var
  Fit: TNeuralImageFit;
  Target: TNNetVolume;
  I: integer;
const
  cEps: single = 0.1;
  cNumClasses = 4;
  cClassId = 2;
begin
  Fit := TNeuralImageFit.Create;
  Target := TNNetVolume.Create(cNumClasses, 1, 1);
  try
    Fit.LabelSmoothing := cEps;
    Target.SetClassForSoftMax(cClassId);
    Fit.ApplyLabelSmoothing(Target);
    // target = (1-eps)*onehot + eps/NumClasses
    for I := 0 to cNumClasses - 1 do
    begin
      if I = cClassId
        then AssertEquals('Smoothed true-class target',
          (1 - cEps) * 1 + cEps / cNumClasses, Target.FData[I], 1e-6)
        else AssertEquals('Smoothed off-class target',
          cEps / cNumClasses, Target.FData[I], 1e-6);
    end;
  finally
    Target.Free;
    Fit.Free;
  end;
end;

procedure TTestNeuralFit.TestLabelSmoothingZeroIsBitIdentical;
var
  Fit: TNeuralImageFit;
  Target, Reference: TNNetVolume;
  I: integer;
const
  cNumClasses = 10;
begin
  Fit := TNeuralImageFit.Create;
  Target := TNNetVolume.Create(cNumClasses, 1, 1);
  Reference := TNNetVolume.Create(cNumClasses, 1, 1);
  try
    Target.SetClassForSoftMax(7);
    Reference.Copy(Target);
    // LabelSmoothing = 0 (the default) must leave the target bit-for-bit
    // identical to the plain one-hot construction.
    Fit.ApplyLabelSmoothing(Target);
    for I := 0 to cNumClasses - 1 do
      AssertEquals('LabelSmoothing=0 must not change the target at all',
        Reference.FData[I], Target.FData[I], 0);
  finally
    Reference.Free;
    Target.Free;
    Fit.Free;
  end;
end;

constructor TAccumDataSource.Create(pMicroBatch, pPoolSize, pInDepth,
  pOutDepth: integer);
var
  S, D: integer;
begin
  inherited Create;
  FMicroBatch := pMicroBatch;
  SetLength(FPoolIn, pPoolSize);
  SetLength(FPoolOut, pPoolSize);
  // Deterministic content: independent of the global RNG so both runs see the
  // exact same samples regardless of how many times RefreshDropoutMask draws.
  for S := 0 to pPoolSize - 1 do
  begin
    FPoolIn[S] := TNNetVolume.Create(pInDepth, 1, 1);
    FPoolOut[S] := TNNetVolume.Create(pOutDepth, 1, 1);
    for D := 0 to pInDepth - 1 do
      FPoolIn[S].FData[D] := Sin(0.7 * S + 1.3 * D) * 0.5 + 0.5;
    for D := 0 to pOutDepth - 1 do
      FPoolOut[S].FData[D] := Cos(0.9 * S + 0.5 * D) * 0.5 + 0.5;
  end;
end;

destructor TAccumDataSource.Destroy;
var
  S: integer;
begin
  for S := 0 to High(FPoolIn) do
  begin
    FPoolIn[S].Free;
    FPoolOut[S].Free;
  end;
  inherited Destroy;
end;

procedure TAccumDataSource.GetPair(Idx, ThreadId: integer;
  vInput, vOutput: TNNetVolume);
var
  Global: integer;
begin
  // Idx is the 1-based within-micro-batch index; CurrentAccumulationStep is the
  // 0-based micro-batch number inside the accumulation group. Together they
  // address the global sample of an effective FMicroBatch*AccumulationSteps
  // batch. With AccumulationSteps = 1 this is just (Idx - 1), so a true big
  // batch and an accumulated batch enumerate the identical sample sequence.
  Global := FFit.CurrentAccumulationStep * FMicroBatch + (Idx - 1);
  vInput.Copy(FPoolIn[Global]);
  vOutput.Copy(FPoolOut[Global]);
end;

// Trains a fresh, deterministically-initialized net with the given micro-batch
// size and AccumulationSteps for Epochs epochs, returning the last-layer weight
// vector (live, no best-model reload) and the final training loss. The
// effective batch is always MicroBatch*AccumSteps and the pool/order is fixed,
// so two configs with the same product MUST converge to the same weights.
function TTestNeuralFit.RunAccumFit(MicroBatch, AccumSteps, Epochs: integer;
  out FinalLoss: TNeuralFloat): TNNetVolume;
var
  Net: TNNet;
  Fit: TNeuralFit;
  Source: TAccumDataSource;
  EffectiveBatch: integer;
begin
  EffectiveBatch := MicroBatch * AccumSteps;
  // Identical initial weights for every call.
  RandSeed := 424242;
  Net := TNNet.Create();
  Net.AddLayer([
    TNNetInput.Create(3),
    TNNetFullConnectLinear.Create(4),
    TNNetFullConnectLinear.Create(2)
  ]);
  // Plain SGD with no inertia keeps the step a pure sum of per-sample deltas,
  // so the summed-delta equivalence is exercised without momentum book-keeping.
  Net.SetLearningRate(0.01, 0.0);

  Source := TAccumDataSource.Create(MicroBatch, EffectiveBatch, 3, 2);
  Fit := TNeuralFit.Create;
  try
    Fit.HideMessages;
    Fit.Verbose := False;
    Fit.MaxThreadNum := 1;            // deterministic reduction order
    Fit.InitialLearningRate := 0.01;
    Fit.LearningRateDecay := 0;       // hold LR fixed
    Fit.StaircaseEpochs := 1;
    Fit.Inertia := 0.0;               // pure SGD, no momentum
    Fit.L2Decay := 0;                 // no weight decay term
    Fit.ClipNorm := 0;                // disable per-layer norm clipping...
    Fit.ClipDelta := 0;               // ...and delta clipping so deltas pass through raw
    Fit.MinBackpropagationError := 0;
    Fit.MinBackpropagationErrorProportion := 0; // backprop EVERY sample (no skip)
    Fit.LoadBestAtEnd := False;       // keep live trained weights
    Fit.AccumulationSteps := AccumSteps;
    Source.FFit := Fit;
    Fit.EnableDefaultLoss();
    // One optimizer step per epoch: the per-epoch loop runs
    // (TrainingCnt div BatchSize) = 1 RunTrainingBatch, and each RunTrainingBatch
    // runs AccumSteps micro-batches => exactly EffectiveBatch samples per step.
    Fit.FitLoading(Net, {TrainingCnt=}MicroBatch, 0, 0, {BatchSize=}MicroBatch,
      Epochs, {$IFDEF FPC}@{$ENDIF}Source.GetPair, nil, nil);
    FinalLoss := Fit.CurrentTrainingError;
    Result := TNNetVolume.Create();
    Result.Copy(Net.GetLastLayer.Neurons[0].Weights);
  finally
    Fit.Free;
    Source.Free;
    Net.Free;
  end;
end;

procedure TTestNeuralFit.TestAccumulationStepsDefaultsToOne;
var
  Fit: TNeuralFit;
begin
  Fit := TNeuralFit.Create;
  try
    AssertEquals('AccumulationSteps must default to 1 (today''s behaviour)',
      1, Fit.AccumulationSteps);
    AssertEquals('CurrentAccumulationStep must start at 0',
      0, Fit.CurrentAccumulationStep);
  finally
    Fit.Free;
  end;
end;

procedure TTestNeuralFit.TestAccumulationStepsClampsBelowOne;
var
  Fit: TNeuralFit;
begin
  Fit := TNeuralFit.Create;
  try
    Fit.AccumulationSteps := 0;
    AssertEquals('AccumulationSteps <= 0 must clamp to 1', 1,
      Fit.AccumulationSteps);
    Fit.AccumulationSteps := -5;
    AssertEquals('Negative AccumulationSteps must clamp to 1', 1,
      Fit.AccumulationSteps);
    Fit.AccumulationSteps := 4;
    AssertEquals('Valid AccumulationSteps must round-trip', 4,
      Fit.AccumulationSteps);
  finally
    Fit.Free;
  end;
end;

procedure TTestNeuralFit.TestAccumulationEqualsBigBatch;
var
  WAccum, WBig: TNNetVolume;
  LossAccum, LossBig: TNeuralFloat;
  I: integer;
const
  cMicroBatch = 3;
  cAccumSteps = 4;   // effective batch = 12
  cEpochs = 8;
begin
  WAccum := nil;
  WBig := nil;
  try
    // Accumulated: 4 micro-batches of 3 => effective batch 12 per optimizer step.
    WAccum := RunAccumFit(cMicroBatch, cAccumSteps, cEpochs, LossAccum);
    // True big batch: one batch of 12, AccumulationSteps = 1.
    WBig := RunAccumFit(cMicroBatch * cAccumSteps, 1, cEpochs, LossBig);

    AssertEquals('Weight vector size must match', WBig.Size, WAccum.Size);
    // The framework SUMS deltas across a batch (it never averages), so N
    // micro-batches of M produce the identical summed delta as one batch of
    // N*M. After many compounding steps the weights must coincide to tight
    // single-precision tolerance (only float-add ordering differs).
    for I := 0 to WBig.Size - 1 do
      AssertEquals('Accumulated weights must match true big-batch weights',
        WBig.FData[I], WAccum.FData[I], 1e-5);
    // Loss bookkeeping must also match: accumulation reports the loss over the
    // full effective batch, exactly like the true big batch.
    AssertEquals('Final training error must match the big-batch run',
      LossBig, LossAccum, 1e-5);
  finally
    WAccum.Free;
    WBig.Free;
  end;
end;

constructor TEMAStepProbe.Create(pFit: TNeuralFitBase);
begin
  inherited Create;
  FFit := pFit;
  Live := TNNetVolume.Create();
  Shadow := TNNetVolume.Create();
  HadShadow := False;
end;

destructor TEMAStepProbe.Destroy;
begin
  Live.Free;
  Shadow.Free;
  inherited Destroy;
end;

procedure TEMAStepProbe.AfterStep(Sender: TObject);
var
  N: TNNet;
begin
  // Snapshot the LIVE weights as they stand right after this step's update.
  Live.Copy(FFit.NN.GetLastLayer.Neurons[0].Weights);
  N := FFit.EMAShadowNet();
  if Assigned(N) then
  begin
    Shadow.Copy(N.GetLastLayer.Neurons[0].Weights);
    HadShadow := True;
  end;
end;

// Runs a short, deterministic Fit on Net with EMA configured by pEnable/pDecay.
// Captures the LAST-step LIVE and EMA SHADOW last-layer weights via an
// OnAfterStep probe (BEFORE the end-of-Fit FAvgWeight overwrite). Returns the
// live snapshot in WLive and, when EMA produced a shadow, the shadow in WShadow
// (nil otherwise). Mirrors RunFit's determinism recipe.
procedure RunFitEMA(Net: TNNet; pEnable: boolean; pDecay: TNeuralFloat;
  out WLive, WShadow: TNNetVolume);
var
  Fit: TNeuralFit;
  Pairs: TNNetVolumePairList;
  Probe: TEMAStepProbe;
begin
  WShadow := nil;
  Pairs := BuildFitPairs();
  RandSeed := 100;
  Fit := TNeuralFit.Create;
  Probe := TEMAStepProbe.Create(Fit);
  try
    Fit.HideMessages;
    Fit.Verbose := False;
    Fit.MaxThreadNum := 1;
    Fit.InitialLearningRate := 0.01;
    Fit.LearningRateDecay := 0;
    Fit.StaircaseEpochs := 1;
    Fit.LoadBestAtEnd := False;
    Fit.EnableEMA := pEnable;
    Fit.EMADecay := pDecay;
    Fit.OnAfterStep := {$IFDEF FPC}@{$ENDIF}Probe.AfterStep;
    Fit.Fit(Net, Pairs, nil, nil, 4, 20);
    WLive := TNNetVolume.Create();
    WLive.Copy(Probe.Live);
    if Probe.HadShadow then
    begin
      WShadow := TNNetVolume.Create();
      WShadow.Copy(Probe.Shadow);
    end;
  finally
    Probe.Free;
    Fit.Free;
    Pairs.Free;
  end;
end;

procedure TTestNeuralFit.TestEMADefaultsOff;
var
  Fit: TNeuralFit;
begin
  Fit := TNeuralFit.Create;
  try
    // EMA must be OPT-IN so the default training path is unchanged.
    AssertFalse('EnableEMA must default to false', Fit.EnableEMA);
    AssertTrue('EMAShadowNet must be nil before any update',
      Fit.EMAShadowNet() = nil);
    // ApplyEMAWeights is a no-op when EMA is disabled.
    AssertFalse('ApplyEMAWeights must be a no-op when EMA is off',
      Fit.ApplyEMAWeights());
    Fit.EnableEMA := True;
    AssertTrue('EnableEMA must be settable', Fit.EnableEMA);
    Fit.EMADecay := 0.9999;
    // EMADecay is a single, so compare at single precision.
    AssertEquals('EMADecay must round-trip', 0.9999, Fit.EMADecay, 1e-6);
  finally
    Fit.Free;
  end;
end;

procedure TTestNeuralFit.TestEMADecayZeroEqualsLive;
var
  Net: TNNet;
  WLive, WShadow: TNNetVolume;
  I: integer;
begin
  WLive := nil;
  WShadow := nil;
  Net := BuildFitNet();
  try
    // With decay = 0 the EMA degenerates to a plain copy: after the LAST update
    // shadow := 0*shadow + 1*live = live. So the shadow must equal the final
    // live weights exactly.
    RunFitEMA(Net, True, 0.0, WLive, WShadow);
    AssertTrue('Shadow net must exist after training with EMA on',
      Assigned(WShadow));
    AssertEquals('Shadow size must match live size', WLive.Size, WShadow.Size);
    for I := 0 to WLive.Size - 1 do
      AssertEquals('decay=0 shadow must equal the live weights exactly',
        WLive.FData[I], WShadow.FData[I], 0);
  finally
    WLive.Free;
    WShadow.Free;
    Net.Free;
  end;
end;

procedure TTestNeuralFit.TestEMAConvexBlend;
var
  NetInit, Net: TNNet;
  WInit, WLive, WShadow: TNNetVolume;
  I: integer;
  Lo, Hi, Sh: TNeuralFloat;
  SawStrictlyBetween: boolean;
begin
  WInit := nil;
  WLive := nil;
  WShadow := nil;
  // Snapshot the initialization weights (same seed as the trained net).
  NetInit := BuildFitNet();
  Net := BuildFitNet();
  try
    WInit := TNNetVolume.Create();
    WInit.Copy(NetInit.GetLastLayer.Neurons[0].Weights);

    // 0 < decay < 1 over several updates: the shadow is a convex blend of the
    // initial seed and the live weights. It must lie BETWEEN them and equal
    // neither (for at least one coordinate that actually moved).
    RunFitEMA(Net, True, 0.9, WLive, WShadow);
    AssertTrue('Shadow net must exist', Assigned(WShadow));

    SawStrictlyBetween := False;
    for I := 0 to WLive.Size - 1 do
    begin
      Lo := Min(WInit.FData[I], WLive.FData[I]);
      Hi := Max(WInit.FData[I], WLive.FData[I]);
      Sh := WShadow.FData[I];
      // Shadow stays within the [init, live] interval (small epsilon for fp).
      AssertTrue('EMA shadow must lie within [init, live]',
        (Sh >= Lo - 1e-6) and (Sh <= Hi + 1e-6));
      // Strict blend: for a coordinate that moved, the shadow differs from BOTH
      // endpoints (it lags the live weights but has left the init).
      if (Abs(WLive.FData[I] - WInit.FData[I]) > 1e-4) then
        if (Abs(Sh - WLive.FData[I]) > 1e-7) and
           (Abs(Sh - WInit.FData[I]) > 1e-7) then
          SawStrictlyBetween := True;
    end;
    AssertTrue('EMA shadow must be a STRICT blend (not equal to init or live) '
      + 'for at least one moved weight', SawStrictlyBetween);
  finally
    WInit.Free;
    WLive.Free;
    WShadow.Free;
    NetInit.Free;
    Net.Free;
  end;
end;

procedure TTestNeuralFit.TestEMAApplyRestoreRoundTrip;
var
  Net: TNNet;
  Pairs: TNNetVolumePairList;
  Fit: TNeuralFit;
  Before, Swapped, After: TNNetVolume;
  ShadowRef: TNNetVolume;
  I: integer;
  Applied: boolean;
begin
  Before := nil;
  Swapped := nil;
  After := nil;
  ShadowRef := nil;
  Net := BuildFitNet();
  Pairs := BuildFitPairs();
  Fit := TNeuralFit.Create;
  try
    Fit.HideMessages;
    Fit.Verbose := False;
    Fit.MaxThreadNum := 1;
    Fit.InitialLearningRate := 0.01;
    Fit.LearningRateDecay := 0;
    Fit.StaircaseEpochs := 1;
    Fit.LoadBestAtEnd := False;
    Fit.EnableEMA := True;
    Fit.EMADecay := 0.9; // shadow != live so the swap is observable
    RandSeed := 100;
    Fit.Fit(Net, Pairs, nil, nil, 4, 20);

    // Capture the live training weights before any swap.
    Before := TNNetVolume.Create();
    Before.Copy(Net.GetLastLayer.Neurons[0].Weights);
    // Capture what the shadow holds (the expected post-swap content).
    ShadowRef := TNNetVolume.Create();
    ShadowRef.Copy(Fit.EMAShadowNet().GetLastLayer.Neurons[0].Weights);

    // Swap EMA weights into the live net.
    Applied := Fit.ApplyEMAWeights();
    AssertTrue('ApplyEMAWeights must succeed when EMA is initialized', Applied);
    Swapped := TNNetVolume.Create();
    Swapped.Copy(Net.GetLastLayer.Neurons[0].Weights);
    for I := 0 to Swapped.Size - 1 do
      AssertEquals('After swap the live net must hold the EMA shadow weights',
        ShadowRef.FData[I], Swapped.FData[I], 0);

    // A nested ApplyEMAWeights must be ignored (returns false, no change).
    AssertFalse('Nested ApplyEMAWeights must be ignored',
      Fit.ApplyEMAWeights());

    // Restore: the live training weights must be bit-identical to before.
    Fit.RestoreLiveWeights();
    After := TNNetVolume.Create();
    After.Copy(Net.GetLastLayer.Neurons[0].Weights);
    AssertEquals('Restore size must match', Before.Size, After.Size);
    for I := 0 to Before.Size - 1 do
      AssertEquals('Apply-then-restore must leave LIVE weights bit-identical',
        Before.FData[I], After.FData[I], 0);
  finally
    Before.Free;
    Swapped.Free;
    After.Free;
    ShadowRef.Free;
    Fit.Free;
    Pairs.Free;
    Net.Free;
  end;
end;

procedure TTestNeuralFit.TestEMADisabledIsBitIdentical;
var
  NetA, NetB, NetOn: TNNet;
  WA, WB, WOn, DummyA, DummyB, ShadowOn: TNNetVolume;
  I: integer;
  ErrBeforeOn, ErrAfterOn: TNeuralFloat;
begin
  WA := nil; WB := nil; WOn := nil;
  DummyA := nil; DummyB := nil; ShadowOn := nil;
  NetA := BuildFitNet();
  NetB := BuildFitNet();    // same seed -> identical init
  NetOn := BuildFitNet();
  try
    // (1) Zero behaviour change with EMA OFF (the default): two EMA-disabled
    // runs of the same net/data must be BIT-FOR-BIT identical. The EMA code
    // path is fully short-circuited when EnableEMA is false, so adding the
    // feature does not perturb the default training path at all.
    RunFitEMA(NetA, False, 0.999, WA, DummyA);
    RunFitEMA(NetB, False, 0.999, WB, DummyB);
    AssertTrue('No shadow must exist with EMA off', DummyA = nil);
    AssertTrue('No shadow must exist with EMA off', DummyB = nil);
    AssertEquals('Live weight size must match', WA.Size, WB.Size);
    for I := 0 to WA.Size - 1 do
      AssertEquals('EMA-off must be bit-identical run-to-run '
        + '(default path unchanged by the feature)',
        WA.FData[I], WB.FData[I], 0);

    // (2) EMA is a side-channel that never feeds back into the live weights or
    // deltas (the math is untouched; only the one lazy shadow clone draws from
    // the global RNG, which can reorder the batch shuffle). So an EMA-ON run
    // must still LEARN normally: the live error after training is well below
    // the error before training.
    ErrBeforeOn := FitNetMSE(NetOn);
    RunFitEMA(NetOn, True, 0.999, WOn, ShadowOn);
    ErrAfterOn := FitNetMSE(NetOn);
    AssertTrue('A shadow must exist when EMA is on', Assigned(ShadowOn));
    AssertTrue('EMA-on training must still reduce the error ('
      + FloatToStr(ErrAfterOn) + ' vs ' + FloatToStr(ErrBeforeOn) + ')',
      ErrAfterOn < ErrBeforeOn);
  finally
    WA.Free; WB.Free; WOn.Free;
    DummyA.Free; DummyB.Free; ShadowOn.Free;
    NetA.Free; NetB.Free; NetOn.Free;
  end;
end;

procedure TTestNeuralFit.TestParamGroupDefaultsOff;
var
  Fit: TNeuralFit;
begin
  // The parameter-group feature must be OFF by default so existing training
  // paths are unchanged.
  Fit := TNeuralFit.Create();
  try
    AssertFalse('ExcludeBiasAndNormFromWeightDecay must default OFF',
      Fit.ExcludeBiasAndNormFromWeightDecay);
    AssertEquals('NormAndBiasLearningRateMul must default to 1.0',
      1.0, Fit.NormAndBiasLearningRateMul, 0);
  finally
    Fit.Free;
  end;
end;

procedure TTestNeuralFit.TestL2DecayDefaultDecaysBias;
var
  NN: TNNet;
  OldW, OldB, Factor: TNeuralFloat;
const
  cL2 = 0.1;
  cLR = 0.5;
begin
  // DEFAULT (off) path: ComputeL2Decay() / ComputeL2Decay(False) must be
  // bit-identical to the legacy behaviour, which shrinks BOTH weights AND the
  // bias by (1 - L2Decay*LearningRate).
  RandSeed := 424242;
  NN := TNNet.Create();
  try
    NN.AddLayer(TNNetInput.Create(4, 1, 1));
    NN.AddLayer(TNNetFullConnectLinear.Create(3));
    NN.SetL2Decay(cL2);
    NN.SetLearningRate(cLR, 0.9);
    Factor := 1 - (cL2 * cLR);

    OldW := NN.Layers[1].Neurons[0].Weights.FData[0];
    OldB := NN.Layers[1].Neurons[0].BiasWeight;

    NN.ComputeL2Decay(False); // ExcludeBias = false == legacy path

    AssertEquals('Default path must shrink weight by (1-L2*LR)',
      OldW * Factor, NN.Layers[1].Neurons[0].Weights.FData[0], 1e-7);
    AssertEquals('Default path must ALSO shrink the bias (legacy behaviour)',
      OldB * Factor, NN.Layers[1].Neurons[0].BiasWeight, 1e-7);
  finally
    NN.Free;
  end;
end;

procedure TTestNeuralFit.TestL2DecayExcludeBiasKeepsBias;
var
  NN: TNNet;
  OldW, OldB, OldLN, Factor: TNeuralFloat;
  I: integer;
  OldWAll: TNNetVolume;
const
  cL2 = 0.1;
  cLR = 0.5;
begin
  // PARAMETER-GROUP path: with ExcludeBias = true the matrix weights are still
  // decayed exactly by (1 - L2*LR), but the bias term receives NO shrinkage,
  // and the normalization layer's gain is untouched as well.
  RandSeed := 424242;
  NN := TNNet.Create();
  OldWAll := TNNetVolume.Create();
  try
    NN.AddLayer(TNNetInput.Create(4, 1, 1));
    NN.AddLayer(TNNetFullConnectLinear.Create(3));
    NN.AddLayer(TNNetLayerNorm.Create());
    NN.SetL2Decay(cL2);
    NN.SetLearningRate(cLR, 0.9);
    Factor := 1 - (cL2 * cLR);

    OldWAll.Copy(NN.Layers[1].Neurons[0].Weights);
    OldW := OldWAll.FData[0];
    OldB := NN.Layers[1].Neurons[0].BiasWeight;
    OldLN := NN.Layers[2].Neurons[0].Weights.FData[0];

    NN.ComputeL2Decay(True); // ExcludeBias

    // (a) matrix weights: still decayed exactly.
    for I := 0 to OldWAll.Size - 1 do
      AssertEquals('Excluded path must still decay matrix weights',
        OldWAll.FData[I] * Factor,
        NN.Layers[1].Neurons[0].Weights.FData[I], 1e-7);
    AssertEquals('Weight[0] sanity', OldW * Factor,
      NN.Layers[1].Neurons[0].Weights.FData[0], 1e-7);
    // (b) bias: NO shrinkage (the whole point of the param group).
    AssertEquals('Bias must NOT be decayed when excluded', OldB,
      NN.Layers[1].Neurons[0].BiasWeight, 0);
    // (c) norm-layer gain: untouched.
    AssertEquals('Norm-layer gain must NOT be decayed', OldLN,
      NN.Layers[2].Neurons[0].Weights.FData[0], 0);
  finally
    OldWAll.Free;
    NN.Free;
  end;
end;

procedure TTestNeuralFit.TestNormLayerLearningRateMultiplier;
var
  NN: TNNet;
  Scaled: integer;
const
  cLR = 0.5;
  cMul = 0.1;
begin
  // The per-group LR multiplier scales ONLY normalization layers that carry
  // trainable neurons; ordinary weight layers keep the base learning rate.
  RandSeed := 424242;
  NN := TNNet.Create();
  try
    NN.AddLayer(TNNetInput.Create(4, 1, 1));
    NN.AddLayer(TNNetFullConnectLinear.Create(3));
    NN.AddLayer(TNNetLayerNorm.Create());
    NN.AddLayer(TNNetFullConnectLinear.Create(2));
    NN.SetLearningRate(cLR, 0.9);

    Scaled := NN.ScaleNormLayerLearningRate(cMul);

    AssertEquals('Exactly one norm layer must be scaled', 1, Scaled);
    AssertEquals('FullConnect LR must stay at base', cLR,
      NN.Layers[1].LearningRate, 0);
    AssertEquals('Norm-layer LR must be multiplied', cLR * cMul,
      NN.Layers[2].LearningRate, 1e-7);
    AssertEquals('Second FullConnect LR must stay at base', cLR,
      NN.Layers[3].LearningRate, 0);

    // Mul = 1 is a no-op and reports zero scaled layers.
    NN.SetLearningRate(cLR, 0.9);
    AssertEquals('Mul=1 must be a no-op', 0,
      NN.ScaleNormLayerLearningRate(1.0));
    AssertEquals('Norm-layer LR unchanged by no-op', cLR,
      NN.Layers[2].LearningRate, 0);
  finally
    NN.Free;
  end;
end;

procedure TTestNeuralFit.TestBatchAugDefaultsOff;
var
  Fit: TNeuralImageFit;
begin
  Fit := TNeuralImageFit.Create;
  try
    // The batch-level mix must be OPT-IN so the default training path is
    // unchanged. Defaults: disabled, Beta(1,1) (=Uniform), every sample.
    AssertTrue('BatchAugMode must default to bamNone',
      Fit.BatchAugMode = bamNone);
    AssertEquals('BatchAugAlpha must default to 1.0', 1.0,
      Fit.BatchAugAlpha, 0);
    AssertEquals('BatchAugProb must default to 1.0', 1.0,
      Fit.BatchAugProb, 0);
    Fit.BatchAugMode := bamMixup;
    AssertTrue('BatchAugMode must round-trip', Fit.BatchAugMode = bamMixup);
    Fit.BatchAugMode := bamCutMix;
    AssertTrue('BatchAugMode must round-trip', Fit.BatchAugMode = bamCutMix);
    Fit.BatchAugAlpha := 0.2;
    AssertEquals('BatchAugAlpha must round-trip', 0.2, Fit.BatchAugAlpha, 1e-7);
  finally
    Fit.Free;
  end;
end;

procedure TTestNeuralFit.TestMixedSoftTargetSumsToOne;
var
  Fit: TNeuralImageFit;
  Target: TNNetVolume;
  I, K: integer;
  Sum, Lambda: TNeuralFloat;
const
  cNumClasses = 10;
begin
  // The mixup/CutMix SOFT target is a convex blend of two one-hot rows; under
  // the softmax convention (the default) every produced row must sum to 1, for
  // any lambda in [0,1] and any pair of (distinct) class ids.
  Fit := TNeuralImageFit.Create;
  Target := TNNetVolume.Create(cNumClasses, 1, 1);
  try
    for K := 0 to 6 do
    begin
      Lambda := K / 6.0; // sweep 0, 1/6, ..., 1
      Fit.BuildMixedSoftTarget(Target, 3, 7, Lambda);
      Sum := 0;
      for I := 0 to cNumClasses - 1 do Sum := Sum + Target.FData[I];
      AssertEquals('Soft target row must sum to 1 (lambda=' +
        FloatToStr(Lambda) + ')', 1.0, Sum, 1e-6);
      // The two mixed classes carry exactly lambda / (1-lambda); all others 0.
      AssertEquals('TagA weight must equal lambda', Lambda, Target.FData[3], 1e-6);
      AssertEquals('TagB weight must equal 1-lambda', 1.0 - Lambda,
        Target.FData[7], 1e-6);
      AssertEquals('Unmixed class must be 0', 0.0, Target.FData[0], 0);
    end;
  finally
    Target.Free;
    Fit.Free;
  end;
end;

procedure TTestNeuralFit.TestMixedSoftTargetLambdaOneIsOneHot;
var
  Fit: TNeuralImageFit;
  Mixed, Reference: TNNetVolume;
  I: integer;
const
  cNumClasses = 10;
  cTagA = 4;
  cTagB = 9;
begin
  // lambda = 1 must reduce EXACTLY to the unaugmented one-hot target for TagA,
  // bit-for-bit identical to SetClassForSoftMax(TagA). This is the "lam=1
  // reduces exactly to the unaugmented batch" guarantee for the softmax path.
  Fit := TNeuralImageFit.Create;
  Mixed := TNNetVolume.Create(cNumClasses, 1, 1);
  Reference := TNNetVolume.Create(cNumClasses, 1, 1);
  try
    Reference.SetClassForSoftMax(cTagA);
    Fit.BuildMixedSoftTarget(Mixed, cTagA, cTagB, 1.0);
    for I := 0 to cNumClasses - 1 do
      AssertEquals('lambda=1 soft target must be bit-identical to one-hot(TagA)',
        Reference.FData[I], Mixed.FData[I], 0);
  finally
    Reference.Free;
    Mixed.Free;
    Fit.Free;
  end;
end;

procedure TTestNeuralFit.TestMixedSoftTargetSameTag;
var
  Fit: TNeuralImageFit;
  Mixed, Reference: TNNetVolume;
  I: integer;
  Sum, Lambda: TNeuralFloat;
const
  cNumClasses = 5;
  cTag = 2;
begin
  // CutMix can pair a sample with a partner of the SAME class. The blend
  // lambda*onehot(t)+(1-lambda)*onehot(t) must collapse back to a clean one-hot
  // (the two weights add up on the same index) and still sum to 1.
  Fit := TNeuralImageFit.Create;
  Mixed := TNNetVolume.Create(cNumClasses, 1, 1);
  Reference := TNNetVolume.Create(cNumClasses, 1, 1);
  try
    Reference.SetClassForSoftMax(cTag);
    Lambda := 0.37;
    Fit.BuildMixedSoftTarget(Mixed, cTag, cTag, Lambda);
    Sum := 0;
    for I := 0 to cNumClasses - 1 do Sum := Sum + Mixed.FData[I];
    AssertEquals('Same-tag soft target must still sum to 1', 1.0, Sum, 1e-6);
    for I := 0 to cNumClasses - 1 do
      AssertEquals('Same-tag soft target must collapse to one-hot',
        Reference.FData[I], Mixed.FData[I], 1e-6);
  finally
    Reference.Free;
    Mixed.Free;
    Fit.Free;
  end;
end;

// Builds a tiny, easily separable 2-class image dataset: class 0 images are
// (mostly) dark, class 1 images are (mostly) bright. SizeImgX/Y small so the
// run is fast and fits well under the test memory/time budget.
function BuildTinyImageSet(pCount: integer): TNNetVolumeList;
var
  I, X, Y, D, Cls: integer;
  V: TNNetVolume;
  Base: TNeuralFloat;
const
  // Must exceed TNeuralImageFit's default MaxCropSize (8) so the built-in
  // random crop/resize leaves a non-degenerate region.
  cSize = 16;
begin
  Result := TNNetVolumeList.Create();
  for I := 0 to pCount - 1 do
  begin
    Cls := I mod 2;
    V := TNNetVolume.Create(cSize, cSize, 3);
    if Cls = 0 then Base := -0.6 else Base := 0.6;
    for X := 0 to cSize - 1 do
      for Y := 0 to cSize - 1 do
        for D := 0 to 2 do
          // small deterministic per-pixel jitter keeps classes separable but
          // not perfectly constant.
          V[X, Y, D] := Base + 0.05 * Sin(0.3 * X + 0.7 * Y + 1.1 * D + I);
    V.Tag := Cls;
    Result.Add(V);
  end;
end;

// Accuracy of an image net over a volume list (argmax == Tag).
function ImageSetAccuracy(NN: TNNet; pList: TNNetVolumeList): TNeuralFloat;
var
  I, Hit: integer;
  Output: TNNetVolume;
begin
  Hit := 0;
  Output := TNNetVolume.Create();
  try
    for I := 0 to pList.Count - 1 do
    begin
      NN.Compute(pList[I]);
      NN.GetOutput(Output);
      if Output.GetClass() = pList[I].Tag then Inc(Hit);
    end;
    Result := Hit / pList.Count;
  finally
    Output.Free;
  end;
end;

// Shared body for the end-to-end convergence tests: a tiny TNeuralImageFit run
// with the given batch-level augmentation mode ENABLED must train without
// crashing and reach high accuracy on a trivially separable set. This exercises
// the in-loop batch-level wiring (partner sampling, the mixup MixVolumes blend
// or the CutMix patch paste + area-fraction lambda, and the soft-target
// rewrite) on the real training path.
function RunImageFitConverges(Mode: TNeuralBatchAugMode): TNeuralFloat;
var
  NN: TNNet;
  Training: TNNetVolumeList;
  Fit: TNeuralImageFit;
  OldMask: TFPUExceptionMask;
begin
  // Mask FP exceptions like a real training loop (softmax/log can transiently
  // touch exInvalidOp); restored in the finally block.
  OldMask := GetExceptionMask;
  SetExceptionMask(OldMask + [exInvalidOp, exOverflow, exZeroDivide,
    exDenormalized, exUnderflow, exPrecision]);
  RandSeed := 424242;
  NN := TNNet.Create();
  NN.AddLayer([
    TNNetInput.Create(16, 16, 3),
    TNNetConvolutionReLU.Create(8, 3, 1, 1, 0),
    TNNetMaxPool.Create(2),
    TNNetFullConnectReLU.Create(16),
    TNNetFullConnectLinear.Create(2),
    TNNetSoftMax.Create()
  ]);

  Training := BuildTinyImageSet(40);
  Fit := TNeuralImageFit.Create;
  try
    Fit.HideMessages;
    Fit.Verbose := False;
    Fit.MaxThreadNum := 1;
    Fit.InitialLearningRate := 0.01;
    Fit.LearningRateDecay := 0;
    Fit.StaircaseEpochs := 1;
    Fit.LoadBestAtEnd := False;
    // Turn ON the batch-level augmentation under test.
    Fit.BatchAugMode := Mode;
    Fit.BatchAugAlpha := 0.2; // typical mixup/CutMix alpha
    RandSeed := 100;
    Fit.Fit(NN, Training, nil, nil, 2, 16, 30);

    Result := ImageSetAccuracy(NN, Training);
  finally
    SetExceptionMask(OldMask);
    Fit.Free;
    Training.Free;
    NN.Free;
  end;
end;

procedure TTestNeuralFit.TestMixupImageFitConverges;
var Acc: TNeuralFloat;
begin
  Acc := RunImageFitConverges(bamMixup);
  AssertTrue('Mixup image-fit must reach high accuracy (' + FloatToStr(Acc) + ')',
    Acc >= 0.9);
end;

procedure TTestNeuralFit.TestCutMixImageFitConverges;
var Acc: TNeuralFloat;
begin
  Acc := RunImageFitConverges(bamCutMix);
  AssertTrue('CutMix image-fit must reach high accuracy (' + FloatToStr(Acc) + ')',
    Acc >= 0.9);
end;

initialization
  RegisterTest(TTestNeuralFit);

end.
