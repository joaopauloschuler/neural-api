unit TestNeuralCallbacks;

// Tests for the Trainer callbacks API (transformers TrainerCallback port):
// TNeuralFitCallback hooks dispatched by the fit loop, and the concrete
// TNeuralFitEarlyStopping callback. Coded by Claude (AI).

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, Math, fpcunit, testregistry, neuralnetwork, neuralvolume,
  neuralfit;

type
  // A counting callback that records how many times each hook fired plus the
  // last-seen epoch/step/validation values, so a tiny fit can assert the
  // hooks land at the expected boundaries the expected number of times.
  TCountingCallback = class(TNeuralFitCallback)
  public
    EpochBeginCount, EpochEndCount, StepEndCount, EvaluateCount: integer;
    LastBeginEpoch, LastEndEpoch, LastStep, LastEvalEpoch: integer;
    LastValLoss, LastValAcc: TNeuralFloat;
    SenderMatched: boolean;
    FExpectedSender: TNeuralFitBase;
    constructor Create(pExpectedSender: TNeuralFitBase);
    procedure OnEpochBegin(Sender: TNeuralFitBase; Epoch: integer); override;
    procedure OnEpochEnd(Sender: TNeuralFitBase; Epoch: integer); override;
    procedure OnStepEnd(Sender: TNeuralFitBase; GlobalStep: integer); override;
    procedure OnEvaluate(Sender: TNeuralFitBase; Epoch: integer;
      ValLoss, ValAcc: TNeuralFloat); override;
  end;

  TTestNeuralCallbacks = class(TTestCase)
  published
    procedure TestCallbackDefaultsEmpty;
    procedure TestAddCallbackCount;
    procedure TestCountingCallbackFiresPerEpoch;
    procedure TestEarlyStoppingDefaultsAndReset;
    procedure TestEarlyStoppingHaltsOnNoImprovement;
    procedure TestEarlyStoppingNoHaltWhenImproving;
    procedure TestZeroCallbackPathStillTrains;
    procedure TestOwnsCallbacksFreesThem;
  end;

implementation

constructor TCountingCallback.Create(pExpectedSender: TNeuralFitBase);
begin
  inherited Create();
  FExpectedSender := pExpectedSender;
  SenderMatched := True;
end;

procedure TCountingCallback.OnEpochBegin(Sender: TNeuralFitBase; Epoch: integer);
begin
  Inc(EpochBeginCount);
  LastBeginEpoch := Epoch;
  if Sender <> FExpectedSender then SenderMatched := False;
end;

procedure TCountingCallback.OnEpochEnd(Sender: TNeuralFitBase; Epoch: integer);
begin
  Inc(EpochEndCount);
  LastEndEpoch := Epoch;
  if Sender <> FExpectedSender then SenderMatched := False;
end;

procedure TCountingCallback.OnStepEnd(Sender: TNeuralFitBase; GlobalStep: integer);
begin
  Inc(StepEndCount);
  LastStep := GlobalStep;
  if Sender <> FExpectedSender then SenderMatched := False;
end;

procedure TCountingCallback.OnEvaluate(Sender: TNeuralFitBase; Epoch: integer;
  ValLoss, ValAcc: TNeuralFloat);
begin
  Inc(EvaluateCount);
  LastEvalEpoch := Epoch;
  LastValLoss := ValLoss;
  LastValAcc := ValAcc;
  if Sender <> FExpectedSender then SenderMatched := False;
end;

// Builds a tiny XOR-ish regression net under a fixed seed.
function BuildCbNet: TNNet;
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

// The 4 XOR samples reused for both training and validation.
function BuildCbPairs: TNNetVolumePairList;
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

procedure TTestNeuralCallbacks.TestCallbackDefaultsEmpty;
var
  Fit: TNeuralFit;
begin
  Fit := TNeuralFit.Create;
  try
    // Zero registered callbacks by default; OwnsCallbacks off by default.
    AssertEquals('No callbacks registered by default', 0, Fit.CallbackCount());
    AssertFalse('OwnsCallbacks must default to false', Fit.OwnsCallbacks);
  finally
    Fit.Free;
  end;
end;

procedure TTestNeuralCallbacks.TestAddCallbackCount;
var
  Fit: TNeuralFit;
  Cb: TCountingCallback;
begin
  Fit := TNeuralFit.Create;
  Cb := TCountingCallback.Create(Fit);
  try
    Fit.AddCallback(Cb);
    AssertEquals('One callback registered', 1, Fit.CallbackCount());
    // nil callbacks are ignored.
    Fit.AddCallback(nil);
    AssertEquals('nil callback must be ignored', 1, Fit.CallbackCount());
  finally
    Fit.Free;   // does not free Cb (OwnsCallbacks=false)
    Cb.Free;
  end;
end;

procedure TTestNeuralCallbacks.TestCountingCallbackFiresPerEpoch;
var
  Net: TNNet;
  Pairs, ValPairs: TNNetVolumePairList;
  Fit: TNeuralFit;
  Cb: TCountingCallback;
const
  cEpochs = 2;
begin
  Net := BuildCbNet();
  Pairs := BuildCbPairs();
  ValPairs := BuildCbPairs();
  RandSeed := 100;
  Fit := TNeuralFit.Create;
  Cb := TCountingCallback.Create(Fit);
  try
    Fit.HideMessages;
    Fit.Verbose := False;
    Fit.MaxThreadNum := 1;
    Fit.InitialLearningRate := 0.01;
    Fit.LearningRateDecay := 0;
    Fit.StaircaseEpochs := 1;
    Fit.LoadBestAtEnd := False;
    Fit.AddCallback(Cb);
    // Validation pairs supplied so OnEvaluate fires once per epoch.
    Fit.Fit(Net, Pairs, ValPairs, nil, 4, cEpochs);

    AssertEquals('OnEpochBegin must fire once per epoch',
      cEpochs, Cb.EpochBeginCount);
    AssertEquals('OnEpochEnd must fire once per epoch',
      cEpochs, Cb.EpochEndCount);
    AssertEquals('OnEvaluate must fire once per epoch (validation supplied)',
      cEpochs, Cb.EvaluateCount);
    AssertTrue('OnStepEnd must fire at least once per epoch',
      Cb.StepEndCount >= cEpochs);
    // Epoch indexing: begin is 0-based, end is 1-based (matches CurrentEpoch).
    AssertEquals('Last begin epoch is 0-based last epoch',
      cEpochs - 1, Cb.LastBeginEpoch);
    AssertEquals('Last end epoch matches final CurrentEpoch',
      cEpochs, Cb.LastEndEpoch);
    AssertEquals('GlobalStep matches CurrentStep at the end',
      Fit.CurrentStep, Cb.LastStep);
    AssertTrue('Sender must be the originating fit object on every hook',
      Cb.SenderMatched);
  finally
    Fit.Free;
    Cb.Free;
    Pairs.Free;
    ValPairs.Free;
    Net.Free;
  end;
end;

procedure TTestNeuralCallbacks.TestEarlyStoppingDefaultsAndReset;
var
  Es: TNeuralFitEarlyStopping;
begin
  Es := TNeuralFitEarlyStopping.Create(3, 0.01);
  try
    AssertEquals('Patience round-trips', 3, Es.Patience);
    AssertEquals('MinDelta round-trips', 0.01, Es.MinDelta, 1e-7);
    AssertFalse('Must not be stopped before any evaluation', Es.Stopped);
    // OnEvaluate with no Sender (nil) must not crash and must track best loss.
    Es.OnEvaluate(nil, 0, 1.0, 0.0);
    AssertEquals('Best loss tracks the first measurement', 1.0, Es.BestLoss, 1e-7);
    Es.Reset();
    AssertFalse('Reset clears the stopped flag', Es.Stopped);
  finally
    Es.Free;
  end;
end;

procedure TTestNeuralCallbacks.TestEarlyStoppingHaltsOnNoImprovement;
var
  Es: TNeuralFitEarlyStopping;
  Sender: TNeuralFit;
  E: integer;
begin
  // Synthetic non-improving validation signal: loss never drops, so after
  // Patience evaluations the callback must request a stop.
  Es := TNeuralFitEarlyStopping.Create(2, 0.0);
  Sender := TNeuralFit.Create;
  try
    Sender.HideMessages;
    Sender.Verbose := False;
    // Eval 0: first measurement (best := 1.0), no stop.
    Es.OnEvaluate(Sender, 0, 1.0, 0.5);
    AssertFalse('No stop after the first evaluation', Sender.ShouldQuit);
    // Eval 1: no improvement (wait=1 < patience 2), still no stop.
    Es.OnEvaluate(Sender, 1, 1.0, 0.5);
    AssertFalse('No stop after one non-improving evaluation', Sender.ShouldQuit);
    // Eval 2: no improvement (wait=2 >= patience 2) => stop requested.
    Es.OnEvaluate(Sender, 2, 1.0, 0.5);
    AssertTrue('Early stopping must request a stop after Patience misses',
      Sender.ShouldQuit);
    AssertTrue('Callback reports itself as stopped', Es.Stopped);
  finally
    Sender.Free;
    Es.Free;
  end;
end;

procedure TTestNeuralCallbacks.TestEarlyStoppingNoHaltWhenImproving;
var
  Es: TNeuralFitEarlyStopping;
  Sender: TNeuralFit;
begin
  // A steadily improving validation loss must never trip the stop.
  Es := TNeuralFitEarlyStopping.Create(2, 0.0);
  Sender := TNeuralFit.Create;
  try
    Sender.HideMessages;
    Sender.Verbose := False;
    Es.OnEvaluate(Sender, 0, 1.0, 0.5);
    Es.OnEvaluate(Sender, 1, 0.9, 0.6);
    Es.OnEvaluate(Sender, 2, 0.8, 0.7);
    Es.OnEvaluate(Sender, 3, 0.7, 0.8);
    AssertFalse('Improving loss must never request a stop', Sender.ShouldQuit);
    AssertFalse('Callback must not be stopped while improving', Es.Stopped);
    AssertEquals('Best loss tracks the lowest seen', 0.7, Es.BestLoss, 1e-7);
  finally
    Sender.Free;
    Es.Free;
  end;
end;

procedure TTestNeuralCallbacks.TestZeroCallbackPathStillTrains;
var
  Net: TNNet;
  Pairs, ValPairs: TNNetVolumePairList;
  Fit: TNeuralFit;
const
  cEpochs = 3;
begin
  // With no callbacks registered, training must complete the full budget
  // exactly as before (backward compatibility / cheap no-op dispatch).
  Net := BuildCbNet();
  Pairs := BuildCbPairs();
  ValPairs := BuildCbPairs();
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
    AssertEquals('No callbacks before fit', 0, Fit.CallbackCount());
    Fit.Fit(Net, Pairs, ValPairs, nil, 4, cEpochs);
    AssertEquals('Zero-callback fit must run the full epoch budget',
      cEpochs, Fit.CurrentEpoch);
    AssertFalse('Zero-callback fit must not request an early stop',
      Fit.ShouldQuit);
  finally
    Fit.Free;
    Pairs.Free;
    ValPairs.Free;
    Net.Free;
  end;
end;

procedure TTestNeuralCallbacks.TestOwnsCallbacksFreesThem;
var
  Fit: TNeuralFit;
  Cb: TCountingCallback;
begin
  // When OwnsCallbacks is set, Destroy frees the registered callbacks. We can
  // only assert this does not crash / double-free here (the callback is not
  // touched after Fit.Free).
  Fit := TNeuralFit.Create;
  Cb := TCountingCallback.Create(Fit);
  Fit.OwnsCallbacks := True;
  Fit.AddCallback(Cb);
  AssertEquals('One owned callback', 1, Fit.CallbackCount());
  Fit.Free;   // must free Cb without leaking or crashing
  AssertTrue('OwnsCallbacks teardown completed', True);
end;

initialization
  RegisterTest(TTestNeuralCallbacks);
end.
