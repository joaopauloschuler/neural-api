unit TestNeuralBytePrediction;
(*
Tests for neuralbyteprediction.pas prediction statistics: TNeuronGroupBase.GetD
implements the Laplace-style (m+1)/(n+2) belief documented in the interface and
computed inline by PredProb/SelectBestIndexes, while GetF is the plain m/n
frequency. GetD is what feeds the "> 0.8" confidence gate, so dropping the +1
would let a rule pass the gate only after more hits than intended.
Coded by Claude (AI).
*)

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, fpcunit, testregistry,
  neuralbyteprediction;

type
  TTestNeuralBytePrediction = class(TTestCase)
  private
    FGroup: TNeuronGroupBase;
    // Set the hit/miss counters of the group under test.
    procedure SetCounts(pCorrect, pWrong: integer);
  protected
    procedure SetUp; override;
  published
    // GetD is the (m+1)/(n+2) belief, not m/(n+2).
    procedure TestGetDIsLaplaceEstimator;
    // With no observation at all the belief is the 1/2 prior.
    procedure TestGetDWithoutObservationsIsHalf;
    // GetF stays the raw m/n frequency and is 0 when nothing was observed.
    procedure TestGetFIsPlainFrequency;
    // The "> 0.8" confidence gate turns at the count the estimator implies.
    procedure TestConfidenceGateThreshold;
    // AddState returns the index it wrote and keeps the state usable.
    procedure TestClassifierAddStateReturnsIndex;
    // Past the NumStates capacity AddState refuses instead of writing past the end.
    procedure TestClassifierAddStateBeyondCapacity;
    // EvolveNeuronGroupAtPos stops at the first improving mutation and reports
    // how many attempts that took, or -1 when the budget of 10 ran out.
    procedure TestClassifierEvolveReturnsAttemptCount;
  end;

implementation

procedure TTestNeuralBytePrediction.SetUp;
begin
  inherited SetUp;
  FGroup.Clear();
end;

procedure TTestNeuralBytePrediction.SetCounts(pCorrect, pWrong: integer);
begin
  FGroup.CorrectNeuronPredictionCnt := pCorrect;
  FGroup.WrongNeuronPredictionCnt := pWrong;
end;

procedure TTestNeuralBytePrediction.TestGetDIsLaplaceEstimator;
begin
  // 8 hits out of 10 => (8+1)/(10+2).
  SetCounts(8, 2);
  AssertEquals('GetD with 8 hits and 2 misses', 9/12, FGroup.GetD(), 0.0001);

  // A single hit and nothing else => (1+1)/(1+2).
  SetCounts(1, 0);
  AssertEquals('GetD with 1 hit', 2/3, FGroup.GetD(), 0.0001);

  // Only misses still keeps the prior mass => (0+1)/(3+2).
  SetCounts(0, 3);
  AssertEquals('GetD with 3 misses', 1/5, FGroup.GetD(), 0.0001);
end;

procedure TTestNeuralBytePrediction.TestGetDWithoutObservationsIsHalf;
begin
  AssertEquals('GetD of a cleared group', 0.5, FGroup.GetD(), 0.0001);
end;

procedure TTestNeuralBytePrediction.TestGetFIsPlainFrequency;
begin
  AssertEquals('GetF of a cleared group', 0, FGroup.GetF(), 0.0001);

  SetCounts(8, 2);
  AssertEquals('GetF with 8 hits and 2 misses', 0.8, FGroup.GetF(), 0.0001);
end;

procedure TTestNeuralBytePrediction.TestConfidenceGateThreshold;
begin
  // 2 flawless hits give 3/4 = 0.75, below the gate.
  SetCounts(2, 0);
  AssertFalse('2 flawless hits pass the 0.8 gate', FGroup.GetD() > 0.8);

  // The fourth one gives 5/6 and turns the gate on.
  SetCounts(4, 0);
  AssertTrue('4 flawless hits pass the 0.8 gate', FGroup.GetD() > 0.8);
end;

procedure TTestNeuralBytePrediction.TestClassifierAddStateReturnsIndex;
var
  Classifier: TClassifier;
  State: array[0..3] of byte;
  PredictedClass: byte;
begin
  State[0] := 1; State[1] := 0; State[2] := 1; State[3] := 0;

  Classifier.Init({pZerosIncluded=}False, {operationLayerSize=}8,
    {pNumberOfSearches=}4);
  Classifier.AddClassifier({NumClasses=}3, {NumStates=}2);

  AssertEquals('First AddState index', 0, Classifier.AddState(1, State));
  AssertEquals('Second AddState index', 1, Classifier.AddState(2, State));

  // The stored state has to be usable: the neuron builder draws its tests from
  // the loaded action array of state 0.
  Classifier.CreateRandomNeuronGroup({neuronpos=}0, {pClass=}2);
  PredictedClass := Classifier.PredictClass(State);
  AssertTrue('Predicted class is within NumClasses', PredictedClass < 3);
end;

procedure TTestNeuralBytePrediction.TestClassifierAddStateBeyondCapacity;
var
  Classifier: TClassifier;
  State: array[0..1] of byte;
begin
  State[0] := 1; State[1] := 1;

  Classifier.Init({pZerosIncluded=}False, {operationLayerSize=}8,
    {pNumberOfSearches=}4);
  Classifier.AddClassifier({NumClasses=}2, {NumStates=}1);

  AssertEquals('Only slot', 0, Classifier.AddState(0, State));
  AssertEquals('One past the capacity', -1, Classifier.AddState(1, State));
  AssertEquals('Still refused afterwards', -1, Classifier.AddState(1, State));
end;

procedure TTestNeuralBytePrediction.TestClassifierEvolveReturnsAttemptCount;
var
  Classifier: TClassifier;
  State: array[0..3] of byte;
  StateCount, Attempts, ImprovedCnt: integer;
  OldSeed: longint;
begin
  OldSeed := RandSeed;
  RandSeed := 20260826;
  try
    Classifier.Init({pZerosIncluded=}False, {operationLayerSize=}8,
      {pNumberOfSearches=}4);
    Classifier.AddClassifier({NumClasses=}2, {NumStates=}16);

    // The label is a copy of the first action, so a single binary test on it is
    // enough for a mutation to find a real improvement.
    for StateCount := 0 to 15 do
    begin
      State[0] := StateCount and 1;
      State[1] := (StateCount shr 1) and 1;
      State[2] := (StateCount shr 2) and 1;
      State[3] := (StateCount shr 3) and 1;
      Classifier.AddState(State[0], State);
    end;

    ImprovedCnt := 0;
    for StateCount := 0 to 19 do
    begin
      Classifier.CreateRandomNeuronGroup({neuronpos=}0, {pClass=}1);
      Attempts := Classifier.EvolveNeuronGroupAtPos(0);
      AssertTrue('Evolve result is -1 or an attempt number in 1..10',
        (Attempts = -1) or ((Attempts >= 1) and (Attempts <= 10)));
      if Attempts > 0 then Inc(ImprovedCnt);
    end;

    // With a learnable label the search has to succeed at least once.
    AssertTrue('At least one evolution improved the group', ImprovedCnt > 0);
  finally
    RandSeed := OldSeed;
  end;
end;

initialization
  RegisterTest(TTestNeuralBytePrediction);
end.
