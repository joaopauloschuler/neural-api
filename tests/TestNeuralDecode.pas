unit TestNeuralDecode;

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, fpcunit, testregistry,
  neuralvolume, neuralnetwork, neuraldecode;

type
  TTestNeuralDecode = class(TTestCase)
  private
    // Builds a tiny char-level next-token net: Input(ContextLen,1,Vocab) ->
    // FC ReLU -> FC Linear(Vocab) -> SoftMax. Random init is fine; the decode
    // routines only need a valid SoftMax head of size = Vocab.
    function BuildTinyNet(ContextLen, Vocab: integer): TNNet;
  published
    // Pure helper functions (no network needed).
    procedure TestLengthPenaltyAlphaZeroIsOne;
    procedure TestLengthPenaltyWuFormula;
    procedure TestSafeLogProbOfOneIsZero;
    procedure TestSafeLogProbClampsZero;
    procedure TestSafeLogProbMatchesLn;
    // End-to-end behaviour on a tiny net.
    procedure TestGreedyReturnsBoundedFiniteResult;
    procedure TestBeamSearchAllSortedDescending;
    procedure TestBeamSearchScoreNoWorseThanGreedy;
  end;

implementation

uses Math;

function TTestNeuralDecode.BuildTinyNet(ContextLen, Vocab: integer): TNNet;
begin
  Result := TNNet.Create();
  Result.AddLayer(TNNetInput.Create(ContextLen, 1, Vocab));
  Result.AddLayer(TNNetFullConnectReLU.Create(16));
  Result.AddLayer(TNNetFullConnectLinear.Create(Vocab));
  Result.AddLayer(TNNetSoftMax.Create());
  Result.SetLearningRate(0.01, 0.0);
  Result.InitWeights();
end;

procedure TTestNeuralDecode.TestLengthPenaltyAlphaZeroIsOne;
begin
  // alpha = 0 -> denominator is exactly 1.0 for every length (raw sum ranking).
  AssertEquals('alpha=0,L=1', 1.0, LengthPenaltyDenominator(1, 0.0), 1e-7);
  AssertEquals('alpha=0,L=37', 1.0, LengthPenaltyDenominator(37, 0.0), 1e-7);
end;

procedure TTestNeuralDecode.TestLengthPenaltyWuFormula;
begin
  // Wu et al. 2016: ((5+L)/6)^alpha. L=7,alpha=1 -> (12/6)^1 = 2.0.
  AssertEquals('L=7,alpha=1', 2.0, LengthPenaltyDenominator(7, 1.0), 1e-6);
  // L=1,alpha=1 -> (6/6) = 1.0 ; longer beams get a >1 denominator (lifts them).
  AssertEquals('L=1,alpha=1', 1.0, LengthPenaltyDenominator(1, 1.0), 1e-6);
  AssertTrue('longer L has larger denominator at alpha>0',
    LengthPenaltyDenominator(20, 0.7) > LengthPenaltyDenominator(5, 0.7));
end;

procedure TTestNeuralDecode.TestSafeLogProbOfOneIsZero;
begin
  AssertEquals('ln(1)=0', 0.0, SafeLogProb(1.0), 1e-7);
end;

procedure TTestNeuralDecode.TestSafeLogProbClampsZero;
var
  V: TNeuralFloat;
begin
  // A zero/near-zero prob must clamp instead of returning -Inf, so the value
  // is large-negative but FINITE (never poisons a cumulative sum).
  V := SafeLogProb(0.0);
  AssertTrue('clamped log is finite', not IsInfinite(V) and not IsNan(V));
  AssertTrue('clamped log is large-negative', V < -50.0);
end;

procedure TTestNeuralDecode.TestSafeLogProbMatchesLn;
begin
  // For an ordinary probability SafeLogProb is just Ln.
  AssertEquals('ln(exp(-2))', -2.0, SafeLogProb(Exp(-2.0)), 1e-6);
end;

procedure TTestNeuralDecode.TestGreedyReturnsBoundedFiniteResult;
var
  NN: TNNet;
  R: TNNetDecodeResult;
begin
  NN := BuildTinyNet(4, 8);
  try
    R := DecodeGreedy(NN, 'ab', 6);
    AssertTrue('greedy never exceeds MaxLen', Length(R.Text) <= 6);
    AssertTrue('greedy score finite',
      not IsInfinite(R.Score) and not IsNan(R.Score));
  finally
    NN.Free;
  end;
end;

procedure TTestNeuralDecode.TestBeamSearchAllSortedDescending;
var
  NN: TNNet;
  All: TNNetDecodeResultArray;
  I: integer;
begin
  NN := BuildTinyNet(4, 8);
  try
    All := DecodeBeamSearchAll(NN, 'ab', 6, 4, 0.0);
    AssertTrue('beam returns at least one result', Length(All) >= 1);
    for I := 1 to High(All) do
      AssertTrue('results sorted by descending score',
        All[I - 1].Score >= All[I].Score - 1e-6);
  finally
    NN.Free;
  end;
end;

procedure TTestNeuralDecode.TestBeamSearchScoreNoWorseThanGreedy;
var
  NN: TNNet;
  G, B: TNNetDecodeResult;
begin
  // Beam search explores a superset of greedy's single path, so its best
  // length-penalised score (alpha=0 -> raw sum-log-prob) can never be lower
  // than greedy's. Tolerance covers float re-normalisation noise.
  NN := BuildTinyNet(4, 8);
  try
    G := DecodeGreedy(NN, 'ab', 6);
    B := DecodeBeamSearch(NN, 'ab', 6, 4, 0.0);
    AssertTrue('beam score >= greedy score', B.Score >= G.Score - 1e-4);
  finally
    NN.Free;
  end;
end;

initialization
  RegisterTest(TTestNeuralDecode);

end.
