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

initialization
  RegisterTest(TTestNeuralBytePrediction);
end.
