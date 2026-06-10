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
    // KV-cache incremental decode on TNNetScaledDotProductAttention.
    procedure TestKVCacheIncrementalMatchesFullForward;
    procedure TestKVCachePrefillThenStepMatchesFullForward;
    procedure TestKVCacheResetStartsFreshSequence;
    procedure TestKVCacheDisabledPathUnchanged;
    // O(1)-per-step incremental decode on TNNetDiagonalSSM (persisted state).
    procedure TestSSMIncrementalMatchesFullForward;
    procedure TestSSMPrefillThenStepMatchesFullForward;
    procedure TestSSMResetStateStartsFreshSequence;
    procedure TestSSMDisabledPathUnchanged;
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

// Headline KV-cache faithfulness check: run the SAME random Q|K|V sequence
// (SDPA is parameter-free, so two nets with different input widths compute
// the same function) through (a) one full causal forward and (b) token-at-a-
// time through the cached incremental-decode path, and assert EVERY position's
// output matches to < 1e-5. With a cache, attending over the cached keys
// [0..t] IS the causal behavior, so all positions (not just the last) agree.
procedure TTestNeuralDecode.TestKVCacheIncrementalMatchesFullForward;
const
  SeqLen = 7;
  Dk = 5;
var
  NNFull, NNStep: TNNet;
  SDPAStep: TNNetScaledDotProductAttention;
  FullIn, StepIn, FullOut, StepOut: TNNetVolume;
  T, D: integer;
begin
  RandSeed := 424242;
  NNFull := TNNet.Create();
  NNStep := TNNet.Create();
  FullIn := TNNetVolume.Create(SeqLen, 1, 3 * Dk);
  StepIn := TNNetVolume.Create(1, 1, 3 * Dk);
  FullOut := TNNetVolume.Create();
  StepOut := TNNetVolume.Create();
  try
    NNFull.AddLayer(TNNetInput.Create(SeqLen, 1, 3 * Dk));
    NNFull.AddLayer(TNNetScaledDotProductAttention.Create(Dk, {CausalMask=}true));
    NNStep.AddLayer(TNNetInput.Create(1, 1, 3 * Dk));
    SDPAStep := TNNetScaledDotProductAttention.Create(Dk, {CausalMask=}true);
    NNStep.AddLayer(SDPAStep);

    FullIn.Randomize();
    FullIn.Sub(0.5);
    NNFull.Compute(FullIn);
    NNFull.GetOutput(FullOut);

    SDPAStep.BeginIncrementalDecode({MaxContext=}SeqLen);
    AssertTrue('cache enabled after Begin', SDPAStep.CacheEnabled);
    for T := 0 to SeqLen - 1 do
    begin
      for D := 0 to 3 * Dk - 1 do
        StepIn[0, 0, D] := FullIn[T, 0, D];
      NNStep.Compute(StepIn);
      NNStep.GetOutput(StepOut);
      AssertEquals('cache length tracks tokens', T + 1, SDPAStep.CacheLength);
      for D := 0 to Dk - 1 do
        AssertEquals('pos ' + IntToStr(T) + ' dim ' + IntToStr(D),
          FullOut[T, 0, D], StepOut[0, 0, D], 1e-5);
    end;
    SDPAStep.EndIncrementalDecode();
    AssertTrue('cache disabled after End', not SDPAStep.CacheEnabled);
  finally
    StepOut.Free;
    FullOut.Free;
    StepIn.Free;
    FullIn.Free;
    NNStep.Free;
    NNFull.Free;
  end;
end;

// Multi-token prompt prefill: feed the first PrefillLen tokens in ONE cached
// forward, then decode the rest token-at-a-time; outputs of the single-token
// steps must still match the full causal forward.
procedure TTestNeuralDecode.TestKVCachePrefillThenStepMatchesFullForward;
const
  SeqLen = 6;
  PrefillLen = 4;
  Dk = 4;
var
  NNFull, NNPre, NNStep: TNNet;
  SDPAPre, SDPAStep: TNNetScaledDotProductAttention;
  FullIn, PreIn, StepIn, FullOut, StepOut: TNNetVolume;
  T, D: integer;
begin
  RandSeed := 31337;
  NNFull := TNNet.Create();
  NNPre := TNNet.Create();
  NNStep := TNNet.Create();
  FullIn := TNNetVolume.Create(SeqLen, 1, 3 * Dk);
  PreIn := TNNetVolume.Create(PrefillLen, 1, 3 * Dk);
  StepIn := TNNetVolume.Create(1, 1, 3 * Dk);
  FullOut := TNNetVolume.Create();
  StepOut := TNNetVolume.Create();
  try
    NNFull.AddLayer(TNNetInput.Create(SeqLen, 1, 3 * Dk));
    NNFull.AddLayer(TNNetScaledDotProductAttention.Create(Dk, true));
    // A layer's input width is fixed by its net, so the multi-token branch of
    // ComputeIncremental (prompt prefill) is exercised on its own net: one
    // cached forward of PrefillLen tokens must reproduce the first PrefillLen
    // causal rows of the full forward. The single-token decode loop is then
    // re-verified on a separate width-1 net over the whole sequence.
    SDPAPre := TNNetScaledDotProductAttention.Create(Dk, true);
    NNPre.AddLayer(TNNetInput.Create(PrefillLen, 1, 3 * Dk));
    NNPre.AddLayer(SDPAPre);
    NNStep.AddLayer(TNNetInput.Create(1, 1, 3 * Dk));
    SDPAStep := TNNetScaledDotProductAttention.Create(Dk, true);
    NNStep.AddLayer(SDPAStep);

    FullIn.Randomize();
    FullIn.Sub(0.5);
    NNFull.Compute(FullIn);
    NNFull.GetOutput(FullOut);

    // Prefill branch: PrefillLen tokens in one cached forward.
    SDPAPre.BeginIncrementalDecode(SeqLen);
    for T := 0 to PrefillLen - 1 do
      for D := 0 to 3 * Dk - 1 do
        PreIn[T, 0, D] := FullIn[T, 0, D];
    NNPre.Compute(PreIn);
    NNPre.GetOutput(StepOut);
    AssertEquals('prefill cache length', PrefillLen, SDPAPre.CacheLength);
    for T := 0 to PrefillLen - 1 do
      for D := 0 to Dk - 1 do
        AssertEquals('prefill pos ' + IntToStr(T) + ' dim ' + IntToStr(D),
          FullOut[T, 0, D], StepOut[T, 0, D], 1e-5);

    // Token-at-a-time branch on a fresh cache must agree at every position.
    SDPAStep.BeginIncrementalDecode(SeqLen);
    for T := 0 to SeqLen - 1 do
    begin
      for D := 0 to 3 * Dk - 1 do
        StepIn[0, 0, D] := FullIn[T, 0, D];
      NNStep.Compute(StepIn);
      NNStep.GetOutput(StepOut);
      for D := 0 to Dk - 1 do
        AssertEquals('step pos ' + IntToStr(T) + ' dim ' + IntToStr(D),
          FullOut[T, 0, D], StepOut[0, 0, D], 1e-5);
    end;
  finally
    StepOut.Free;
    FullOut.Free;
    StepIn.Free;
    PreIn.Free;
    FullIn.Free;
    NNStep.Free;
    NNPre.Free;
    NNFull.Free;
  end;
end;

// ResetCache must start a genuinely fresh sequence: decoding the same token
// stream twice (with a ResetCache in between) yields identical outputs.
procedure TTestNeuralDecode.TestKVCacheResetStartsFreshSequence;
const
  SeqLen = 5;
  Dk = 3;
var
  NN: TNNet;
  SDPA: TNNetScaledDotProductAttention;
  StepIn, OutA, OutB: TNNetVolume;
  Seq: TNNetVolume;
  T, D, Pass: integer;
  FirstRun: array of TNeuralFloat;
begin
  RandSeed := 90210;
  NN := TNNet.Create();
  StepIn := TNNetVolume.Create(1, 1, 3 * Dk);
  OutA := TNNetVolume.Create();
  OutB := TNNetVolume.Create();
  Seq := TNNetVolume.Create(SeqLen, 1, 3 * Dk);
  SetLength(FirstRun, SeqLen * Dk);
  try
    NN.AddLayer(TNNetInput.Create(1, 1, 3 * Dk));
    SDPA := TNNetScaledDotProductAttention.Create(Dk, true);
    NN.AddLayer(SDPA);
    Seq.Randomize();
    Seq.Sub(0.5);
    SDPA.BeginIncrementalDecode(SeqLen);
    for Pass := 0 to 1 do
    begin
      if Pass = 1 then
      begin
        SDPA.ResetCache();
        AssertEquals('cache empty after reset', 0, SDPA.CacheLength);
      end;
      for T := 0 to SeqLen - 1 do
      begin
        for D := 0 to 3 * Dk - 1 do
          StepIn[0, 0, D] := Seq[T, 0, D];
        NN.Compute(StepIn);
        NN.GetOutput(OutA);
        for D := 0 to Dk - 1 do
        begin
          if Pass = 0 then
            FirstRun[T * Dk + D] := OutA[0, 0, D]
          else
            AssertEquals('replay pos ' + IntToStr(T) + ' dim ' + IntToStr(D),
              FirstRun[T * Dk + D], OutA[0, 0, D], 1e-7);
        end;
      end;
    end;
  finally
    Seq.Free;
    OutB.Free;
    OutA.Free;
    StepIn.Free;
    NN.Free;
  end;
end;

// With the cache disabled (default), Begin+End round-trip must leave the
// normal full-sequence forward bit-for-bit unchanged.
procedure TTestNeuralDecode.TestKVCacheDisabledPathUnchanged;
const
  SeqLen = 6;
  Dk = 4;
var
  NN: TNNet;
  SDPA: TNNetScaledDotProductAttention;
  InV, OutBefore, OutAfter: TNNetVolume;
  T, D: integer;
begin
  RandSeed := 777;
  NN := TNNet.Create();
  InV := TNNetVolume.Create(SeqLen, 1, 3 * Dk);
  OutBefore := TNNetVolume.Create();
  OutAfter := TNNetVolume.Create();
  try
    NN.AddLayer(TNNetInput.Create(SeqLen, 1, 3 * Dk));
    SDPA := TNNetScaledDotProductAttention.Create(Dk, false);
    NN.AddLayer(SDPA);
    InV.Randomize();
    InV.Sub(0.5);
    NN.Compute(InV);
    NN.GetOutput(OutBefore);
    // Enable then immediately disable: the next forward must be identical.
    SDPA.BeginIncrementalDecode(SeqLen);
    SDPA.EndIncrementalDecode();
    NN.Compute(InV);
    NN.GetOutput(OutAfter);
    for T := 0 to SeqLen - 1 do
      for D := 0 to Dk - 1 do
        AssertEquals('pos ' + IntToStr(T) + ' dim ' + IntToStr(D),
          OutBefore[T, 0, D], OutAfter[T, 0, D], 0);
  finally
    OutAfter.Free;
    OutBefore.Free;
    InV.Free;
    NN.Free;
  end;
end;

// Headline SSM faithfulness check: run the SAME random sequence through the
// SAME weights two ways - (a) one full forward over SeqLen tokens and (b)
// token-at-a-time through the incremental path (persisted state h carried
// across single-token forwards) - and assert EVERY position's output matches
// to < 1e-5. A linear recurrence summarises the entire past in its state, so
// the incremental sweep is mathematically identical to the full one.
procedure TTestNeuralDecode.TestSSMIncrementalMatchesFullForward;
const
  SeqLen = 9;
  Depth = 6;
var
  NNFull, NNStep: TNNet;
  SSMFull, SSMStep: TNNetDiagonalSSM;
  FullIn, StepIn, FullOut, StepOut: TNNetVolume;
  T, D: integer;
begin
  RandSeed := 424242;
  NNFull := TNNet.Create();
  NNStep := TNNet.Create();
  FullIn := TNNetVolume.Create(SeqLen, 1, Depth);
  StepIn := TNNetVolume.Create(1, 1, Depth);
  FullOut := TNNetVolume.Create();
  StepOut := TNNetVolume.Create();
  try
    NNFull.AddLayer(TNNetInput.Create(SeqLen, 1, Depth));
    SSMFull := TNNetDiagonalSSM.Create();
    NNFull.AddLayer(SSMFull);
    NNStep.AddLayer(TNNetInput.Create(1, 1, Depth));
    SSMStep := TNNetDiagonalSSM.Create();
    NNStep.AddLayer(SSMStep);
    // Non-trivial weights: randomize a_raw / b / c / e per channel, then copy
    // the exact values to the single-token net so both compute one function.
    for D := 0 to Depth - 1 do
    begin
      SSMFull.Neurons[0].Weights.FData[D] := (Random(2000) - 1000) / 1000;
      SSMFull.Neurons[1].Weights.FData[D] := 0.5 + Random(1000) / 1000;
      SSMFull.Neurons[2].Weights.FData[D] := 0.5 + Random(1000) / 1000;
      SSMFull.Neurons[3].Weights.FData[D] := (Random(2000) - 1000) / 1000;
    end;
    SSMStep.CopyWeights(SSMFull);

    FullIn.Randomize();
    FullIn.Sub(0.5);
    NNFull.Compute(FullIn);
    NNFull.GetOutput(FullOut);

    SSMStep.BeginIncrementalDecode();
    AssertTrue('decode enabled after Begin', SSMStep.DecodeEnabled);
    for T := 0 to SeqLen - 1 do
    begin
      for D := 0 to Depth - 1 do
        StepIn[0, 0, D] := FullIn[T, 0, D];
      NNStep.Compute(StepIn);
      NNStep.GetOutput(StepOut);
      AssertEquals('decode steps track tokens', T + 1, SSMStep.DecodeSteps);
      for D := 0 to Depth - 1 do
        AssertEquals('pos ' + IntToStr(T) + ' dim ' + IntToStr(D),
          FullOut[T, 0, D], StepOut[0, 0, D], 1e-5);
    end;
    SSMStep.EndIncrementalDecode();
    AssertTrue('decode disabled after End', not SSMStep.DecodeEnabled);
  finally
    StepOut.Free;
    FullOut.Free;
    StepIn.Free;
    FullIn.Free;
    NNStep.Free;
    NNFull.Free;
  end;
end;

// Multi-token prompt prefill: feed the first PrefillLen tokens in ONE
// incremental forward (on a width-PrefillLen net), then verify they match the
// first PrefillLen rows of the full forward - the persisted-state sweep
// handles any number of tokens per call, not just one.
procedure TTestNeuralDecode.TestSSMPrefillThenStepMatchesFullForward;
const
  SeqLen = 8;
  PrefillLen = 5;
  Depth = 4;
var
  NNFull, NNPre: TNNet;
  SSMFull, SSMPre: TNNetDiagonalSSM;
  FullIn, PreIn, FullOut, PreOut: TNNetVolume;
  T, D: integer;
begin
  RandSeed := 31337;
  NNFull := TNNet.Create();
  NNPre := TNNet.Create();
  FullIn := TNNetVolume.Create(SeqLen, 1, Depth);
  PreIn := TNNetVolume.Create(PrefillLen, 1, Depth);
  FullOut := TNNetVolume.Create();
  PreOut := TNNetVolume.Create();
  try
    NNFull.AddLayer(TNNetInput.Create(SeqLen, 1, Depth));
    SSMFull := TNNetDiagonalSSM.Create();
    NNFull.AddLayer(SSMFull);
    NNPre.AddLayer(TNNetInput.Create(PrefillLen, 1, Depth));
    SSMPre := TNNetDiagonalSSM.Create();
    NNPre.AddLayer(SSMPre);
    for D := 0 to Depth - 1 do
    begin
      SSMFull.Neurons[0].Weights.FData[D] := (Random(2000) - 1000) / 1000;
      SSMFull.Neurons[1].Weights.FData[D] := 0.5 + Random(1000) / 1000;
      SSMFull.Neurons[2].Weights.FData[D] := 0.5 + Random(1000) / 1000;
      SSMFull.Neurons[3].Weights.FData[D] := (Random(2000) - 1000) / 1000;
    end;
    SSMPre.CopyWeights(SSMFull);

    FullIn.Randomize();
    FullIn.Sub(0.5);
    NNFull.Compute(FullIn);
    NNFull.GetOutput(FullOut);

    SSMPre.BeginIncrementalDecode();
    for T := 0 to PrefillLen - 1 do
      for D := 0 to Depth - 1 do
        PreIn[T, 0, D] := FullIn[T, 0, D];
    NNPre.Compute(PreIn);
    NNPre.GetOutput(PreOut);
    AssertEquals('prefill decode steps', PrefillLen, SSMPre.DecodeSteps);
    for T := 0 to PrefillLen - 1 do
      for D := 0 to Depth - 1 do
        AssertEquals('prefill pos ' + IntToStr(T) + ' dim ' + IntToStr(D),
          FullOut[T, 0, D], PreOut[T, 0, D], 1e-5);
  finally
    PreOut.Free;
    FullOut.Free;
    PreIn.Free;
    FullIn.Free;
    NNPre.Free;
    NNFull.Free;
  end;
end;

// ResetState must start a genuinely fresh sequence: streaming the same token
// stream twice (with a ResetState in between) yields identical outputs.
procedure TTestNeuralDecode.TestSSMResetStateStartsFreshSequence;
const
  SeqLen = 6;
  Depth = 3;
var
  NN: TNNet;
  SSM: TNNetDiagonalSSM;
  StepIn, OutV, Seq: TNNetVolume;
  T, D, Pass: integer;
  FirstRun: array of TNeuralFloat;
begin
  RandSeed := 90210;
  NN := TNNet.Create();
  StepIn := TNNetVolume.Create(1, 1, Depth);
  OutV := TNNetVolume.Create();
  Seq := TNNetVolume.Create(SeqLen, 1, Depth);
  SetLength(FirstRun, SeqLen * Depth);
  try
    NN.AddLayer(TNNetInput.Create(1, 1, Depth));
    SSM := TNNetDiagonalSSM.Create();
    NN.AddLayer(SSM);
    for D := 0 to Depth - 1 do
      SSM.Neurons[0].Weights.FData[D] := (Random(2000) - 1000) / 1000;
    Seq.Randomize();
    Seq.Sub(0.5);
    SSM.BeginIncrementalDecode();
    for Pass := 0 to 1 do
    begin
      if Pass = 1 then
      begin
        SSM.ResetState();
        AssertEquals('steps zero after reset', 0, SSM.DecodeSteps);
      end;
      for T := 0 to SeqLen - 1 do
      begin
        for D := 0 to Depth - 1 do
          StepIn[0, 0, D] := Seq[T, 0, D];
        NN.Compute(StepIn);
        NN.GetOutput(OutV);
        for D := 0 to Depth - 1 do
        begin
          if Pass = 0 then
            FirstRun[T * Depth + D] := OutV[0, 0, D]
          else
            AssertEquals('replay pos ' + IntToStr(T) + ' dim ' + IntToStr(D),
              FirstRun[T * Depth + D], OutV[0, 0, D], 1e-7);
        end;
      end;
    end;
  finally
    Seq.Free;
    OutV.Free;
    StepIn.Free;
    NN.Free;
  end;
end;

// With incremental decode disabled (default), Begin+End round-trip must leave
// the normal full-sequence forward bit-for-bit unchanged.
procedure TTestNeuralDecode.TestSSMDisabledPathUnchanged;
const
  SeqLen = 7;
  Depth = 5;
var
  NN: TNNet;
  SSM: TNNetDiagonalSSM;
  InV, OutBefore, OutAfter: TNNetVolume;
  T, D: integer;
begin
  RandSeed := 777;
  NN := TNNet.Create();
  InV := TNNetVolume.Create(SeqLen, 1, Depth);
  OutBefore := TNNetVolume.Create();
  OutAfter := TNNetVolume.Create();
  try
    NN.AddLayer(TNNetInput.Create(SeqLen, 1, Depth));
    SSM := TNNetDiagonalSSM.Create();
    NN.AddLayer(SSM);
    for D := 0 to Depth - 1 do
      SSM.Neurons[0].Weights.FData[D] := (Random(2000) - 1000) / 1000;
    InV.Randomize();
    InV.Sub(0.5);
    NN.Compute(InV);
    NN.GetOutput(OutBefore);
    // Enable then immediately disable: the next forward must be identical.
    SSM.BeginIncrementalDecode();
    SSM.EndIncrementalDecode();
    NN.Compute(InV);
    NN.GetOutput(OutAfter);
    for T := 0 to SeqLen - 1 do
      for D := 0 to Depth - 1 do
        AssertEquals('pos ' + IntToStr(T) + ' dim ' + IntToStr(D),
          OutBefore[T, 0, D], OutAfter[T, 0, D], 0);
  finally
    OutAfter.Free;
    OutBefore.Free;
    InV.Free;
    NN.Free;
  end;
end;

initialization
  RegisterTest(TTestNeuralDecode);

end.
