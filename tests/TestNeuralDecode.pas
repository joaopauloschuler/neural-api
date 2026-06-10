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
    // Streamable tiny causal LMs for the TNNetStreamingDecoder tests. All
    // parameter shapes are sequence-length independent, so a width-1 twin can
    // CopyWeights() from the full-width net. Norms are TNNetDyT on purpose
    // (per-element, no cross-token statistics -> streamed decode is exact;
    // TNNetLayerNorm/TNNetRMSNorm normalize over the whole sample including
    // the sequence axis and would break full-vs-incremental equality).
    function BuildTinyCausalLM(ContextLen: integer): TNNet;   // RoPE attention
    function BuildTinySSMLM(ContextLen: integer): TNNet;      // DiagonalSSM
    function BuildTinyHybridLM(ContextLen: integer): TNNet;   // SSM + attention
    // Streams Toks token-at-a-time through Session and asserts every step's
    // output row matches the corresponding row of Full's causal forward.
    procedure AssertStreamMatchesFull(Full: TNNet;
      Session: TNNetStreamingDecoder; const Toks: array of integer;
      const MsgPrefix: string);
    // Hand-rolled full-re-encode greedy reference loop (what a cache-less
    // sampler pays): zero-pad the prefix into the full-width net, argmax the
    // row at the last real position, append; same EOS rule (token < 2 is
    // stored, then generation stops) as GenerateTokensStreamed.
    procedure FullGreedyGenerate(Full: TNNet; var Toks: TNeuralIntegerArray;
      PromptLen, MaxNew: integer; out OutLen: integer);
    // Zeroes the last (logit head) layer's weights and biases and raises one
    // token's bias so greedy argmax always emits exactly that token.
    procedure RigHeadToConstantToken(NN: TNNet; Token: integer);
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
    procedure TestKVCacheTruncateThenReappendMatchesFresh;
    procedure TestKVCacheDisabledPathUnchanged;
    // O(1)-per-step incremental decode on TNNetDiagonalSSM (persisted state).
    procedure TestSSMIncrementalMatchesFullForward;
    procedure TestSSMPrefillThenStepMatchesFullForward;
    procedure TestSSMResetStateStartsFreshSequence;
    procedure TestSSMDisabledPathUnchanged;
    // TNNetStreamingDecoder: the reusable incremental-decode session.
    procedure TestStreamingDecoderTransformerMatchesFullForward;
    procedure TestStreamingDecoderSSMMatchesFullForward;
    procedure TestStreamingDecoderResetStartsFreshSequence;
    procedure TestStreamingDecoderTruncateToRollsBack;
    // GenerateTokensStreamed / GenerateStringStreamed: streamed generation.
    procedure TestGenerateTokensStreamedTransformerMatchesFullGreedy;
    procedure TestGenerateTokensStreamedSSMMatchesFullGreedy;
    procedure TestGenerateTokensStreamedGreedySamplerMatchesNilPath;
    procedure TestGenerateTokensStreamedStopsAtEOS;
    procedure TestGenerateTokensStreamedRespectsCaps;
    procedure TestGenerateTokensStreamedRejectsWideSession;
    procedure TestGenerateStringStreamedMatchesTokenCore;
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

// TruncateCache must implement the speculative-decoding rollback exactly:
// append CommitLen + DraftLen tokens, truncate back to CommitLen (discarding
// the rejected drafts' K/V), append DIFFERENT replacement tokens, and the
// outputs from the truncated cache must match a fresh-cache run over the
// same final sequence at every kept position.
procedure TTestNeuralDecode.TestKVCacheTruncateThenReappendMatchesFresh;
const
  CommitLen = 4;   // tokens committed before the speculative block
  DraftLen = 3;    // speculatively appended, then all rejected
  TailLen = 3;     // replacement tokens appended after the rollback
  Dk = 4;
var
  NN: TNNet;
  SDPA: TNNetScaledDotProductAttention;
  StepIn, OutV: TNNetVolume;
  Committed, Drafts, Tail: TNNetVolume;
  T, D, Pass: integer;
  FirstRun: array of TNeuralFloat;

  procedure FeedToken(Source: TNNetVolume; Index: integer);
  var
    DD: integer;
  begin
    for DD := 0 to 3 * Dk - 1 do
      StepIn[0, 0, DD] := Source[Index, 0, DD];
    NN.Compute(StepIn);
    NN.GetOutput(OutV);
  end;

begin
  RandSeed := 55501;
  NN := TNNet.Create();
  StepIn := TNNetVolume.Create(1, 1, 3 * Dk);
  OutV := TNNetVolume.Create();
  Committed := TNNetVolume.Create(CommitLen, 1, 3 * Dk);
  Drafts := TNNetVolume.Create(DraftLen, 1, 3 * Dk);
  Tail := TNNetVolume.Create(TailLen, 1, 3 * Dk);
  SetLength(FirstRun, (CommitLen + TailLen) * Dk);
  try
    NN.AddLayer(TNNetInput.Create(1, 1, 3 * Dk));
    SDPA := TNNetScaledDotProductAttention.Create(Dk, true);
    NN.AddLayer(SDPA);
    Committed.Randomize(); Committed.Sub(0.5);
    Drafts.Randomize(); Drafts.Sub(0.5);
    Tail.Randomize(); Tail.Sub(0.5);

    SDPA.BeginIncrementalDecode(CommitLen + DraftLen + TailLen);
    // Pass 0: committed prefix + rejected drafts + truncate + real tail.
    // Pass 1 (reference): fresh cache, committed prefix + real tail only.
    for Pass := 0 to 1 do
    begin
      if Pass = 1 then SDPA.ResetCache();
      for T := 0 to CommitLen - 1 do
      begin
        FeedToken(Committed, T);
        if Pass = 0 then
          for D := 0 to Dk - 1 do
            FirstRun[T * Dk + D] := OutV[0, 0, D]
        else
          for D := 0 to Dk - 1 do
            AssertEquals('prefix pos ' + IntToStr(T) + ' dim ' + IntToStr(D),
              FirstRun[T * Dk + D], OutV[0, 0, D], 1e-7);
      end;
      if Pass = 0 then
      begin
        // Speculative block: append drafts, then reject them all.
        for T := 0 to DraftLen - 1 do
          FeedToken(Drafts, T);
        AssertEquals('cache holds prefix+drafts',
          CommitLen + DraftLen, SDPA.CacheLength);
        SDPA.TruncateCache(CommitLen);
        AssertEquals('cache rewound to the committed prefix',
          CommitLen, SDPA.CacheLength);
      end;
      for T := 0 to TailLen - 1 do
      begin
        FeedToken(Tail, T);
        if Pass = 0 then
          for D := 0 to Dk - 1 do
            FirstRun[(CommitLen + T) * Dk + D] := OutV[0, 0, D]
        else
          for D := 0 to Dk - 1 do
            AssertEquals('tail pos ' + IntToStr(T) + ' dim ' + IntToStr(D),
              FirstRun[(CommitLen + T) * Dk + D], OutV[0, 0, D], 1e-7);
      end;
    end;
    // TruncateCache(0) behaves as ResetCache.
    SDPA.TruncateCache(0);
    AssertEquals('truncate to zero empties the cache', 0, SDPA.CacheLength);
  finally
    Tail.Free;
    Drafts.Free;
    Committed.Free;
    OutV.Free;
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

// ---------------------------------------------------------------------------
// TNNetStreamingDecoder tests. Shared tiny-LM shape: token ids in (W,1,1),
// TNNetEmbedding(Vocab=12, Dim=8), a streamable mixer, PointwiseConvLinear
// head (raw logits - matching pre-softmax outputs is the stronger check).
// ---------------------------------------------------------------------------
const
  csStreamVocab = 12;
  csStreamDim = 8;

function TTestNeuralDecode.BuildTinyCausalLM(ContextLen: integer): TNNet;
begin
  Result := TNNet.Create();
  Result.AddLayer(TNNetInput.Create(ContextLen, 1, 1));
  // Token-only embedding: position is injected by RoPE inside attention.
  Result.AddLayer(TNNetEmbedding.Create(csStreamVocab, csStreamDim, 0, 0.02));
  Result.AddTransformerEncoderBlock({Heads=}2, {d_ff=}8,
    {PreNorm=}true, {CausalMask=}true, {UseRoPE=}true, {NormClass=}TNNetDyT);
  Result.AddLayer(TNNetPointwiseConvLinear.Create(csStreamVocab));
end;

function TTestNeuralDecode.BuildTinySSMLM(ContextLen: integer): TNNet;
begin
  // No RoPE and no positional embedding: the left-to-right recurrence carries
  // order by construction.
  Result := TNNet.Create();
  Result.AddLayer(TNNetInput.Create(ContextLen, 1, 1));
  Result.AddLayer(TNNetEmbedding.Create(csStreamVocab, csStreamDim, 0, 0.02));
  Result.AddLayer(TNNetPointwiseConvLinear.Create(csStreamDim));
  Result.AddLayer(TNNetDiagonalSSM.Create());
  Result.AddLayer(TNNetPointwiseConvLinear.Create(csStreamVocab));
end;

function TTestNeuralDecode.BuildTinyHybridLM(ContextLen: integer): TNNet;
begin
  // Both streamable mixer families in ONE net, so a single Reset() must
  // clear BOTH the SSM state and the attention KV caches.
  Result := TNNet.Create();
  Result.AddLayer(TNNetInput.Create(ContextLen, 1, 1));
  Result.AddLayer(TNNetEmbedding.Create(csStreamVocab, csStreamDim, 0, 0.02));
  Result.AddLayer(TNNetDiagonalSSM.Create());
  Result.AddTransformerEncoderBlock({Heads=}2, {d_ff=}8,
    {PreNorm=}true, {CausalMask=}true, {UseRoPE=}true, {NormClass=}TNNetDyT);
  Result.AddLayer(TNNetPointwiseConvLinear.Create(csStreamVocab));
end;

procedure TTestNeuralDecode.AssertStreamMatchesFull(Full: TNNet;
  Session: TNNetStreamingDecoder; const Toks: array of integer;
  const MsgPrefix: string);
var
  FullIn, FullOut, StepIn: TNNetVolume;
  T, D, SeqLen, Depth: integer;
begin
  SeqLen := Length(Toks);
  FullIn := TNNetVolume.Create(SeqLen, 1, 1);
  FullOut := TNNetVolume.Create();
  StepIn := TNNetVolume.Create(1, 1, 1);
  try
    for T := 0 to SeqLen - 1 do FullIn.FData[T] := Toks[T];
    Full.Compute(FullIn);
    Full.GetOutput(FullOut);
    Depth := FullOut.Depth;
    for T := 0 to SeqLen - 1 do
    begin
      StepIn.FData[0] := Toks[T];
      Session.StepForward(StepIn, T);
      for D := 0 to Depth - 1 do
        AssertEquals(MsgPrefix + ' pos ' + IntToStr(T) + ' dim ' + IntToStr(D),
          FullOut[T, 0, D], Session.Output()[0, 0, D], 1e-5);
    end;
  finally
    StepIn.Free;
    FullOut.Free;
    FullIn.Free;
  end;
end;

// Headline session test: a width-1 weight-copied twin of a RoPE causal
// transformer streamed token-at-a-time through TNNetStreamingDecoder must
// reproduce every per-position output of the full-width causal forward.
procedure TTestNeuralDecode.TestStreamingDecoderTransformerMatchesFullForward;
const
  SeqLen = 7;
  Toks: array[0..6] of integer = (3, 7, 1, 9, 4, 11, 2);
var
  Full, Twin: TNNet;
  Session: TNNetStreamingDecoder;
begin
  RandSeed := 424242;
  Full := BuildTinyCausalLM(SeqLen);
  Twin := BuildTinyCausalLM(1);
  Session := nil;
  try
    Twin.CopyWeights(Full);
    Session := TNNetStreamingDecoder.Create(Twin, SeqLen);
    AssertEquals('one SDPA per head', 2, Session.SDPACount);
    AssertTrue('RoPE layers collected', Session.RopeCount > 0);
    AssertEquals('no SSM layers in a transformer', 0, Session.SSMCount);
    AssertTrue('session exposes the twin', Session.Net = Twin);
    Session.Reset();
    AssertStreamMatchesFull(Full, Session, Toks, 'transformer');
  finally
    Session.Free;
    Twin.Free;
    Full.Free;
  end;
end;

// Same exactness gate for the recurrent family: a width-1 DiagonalSSM twin
// (no RoPE, no positional embedding - the state carries order) must stream
// exactly.
procedure TTestNeuralDecode.TestStreamingDecoderSSMMatchesFullForward;
const
  SeqLen = 8;
  Toks: array[0..7] of integer = (5, 2, 9, 9, 1, 7, 3, 10);
var
  Full, Twin: TNNet;
  Session: TNNetStreamingDecoder;
  SSMFull: TNNetDiagonalSSM;
  i, D: integer;
begin
  RandSeed := 424242;
  Full := BuildTinySSMLM(SeqLen);
  Twin := BuildTinySSMLM(1);
  Session := nil;
  try
    // Non-trivial per-channel recurrence (the defaults are uniform across
    // channels); the twin still copies the exact values below.
    SSMFull := nil;
    for i := 0 to Full.Layers.Count - 1 do
      if Full.Layers[i] is TNNetDiagonalSSM then
        SSMFull := TNNetDiagonalSSM(Full.Layers[i]);
    AssertTrue('SSM layer found in full net', SSMFull <> nil);
    for D := 0 to csStreamDim - 1 do
    begin
      SSMFull.Neurons[0].Weights.FData[D] := (Random(2000) - 1000) / 1000;
      SSMFull.Neurons[1].Weights.FData[D] := 0.5 + Random(1000) / 1000;
      SSMFull.Neurons[2].Weights.FData[D] := 0.5 + Random(1000) / 1000;
      SSMFull.Neurons[3].Weights.FData[D] := (Random(2000) - 1000) / 1000;
    end;
    Twin.CopyWeights(Full);
    Session := TNNetStreamingDecoder.Create(Twin, SeqLen);
    AssertEquals('one SSM layer', 1, Session.SSMCount);
    AssertEquals('no attention layers', 0, Session.SDPACount);
    AssertEquals('no rope layers', 0, Session.RopeCount);
    Session.Reset();
    AssertStreamMatchesFull(Full, Session, Toks, 'ssm');
  finally
    Session.Free;
    Twin.Free;
    Full.Free;
  end;
end;

// Reset() must start a genuinely fresh sequence in BOTH state machineries at
// once: stream sequence A through a hybrid (SSM + RoPE attention) twin, then
// Reset() and stream a DIFFERENT sequence B - B's outputs must match B's own
// full forward (no state leaked from A through either the SSM state vector
// or the KV caches).
procedure TTestNeuralDecode.TestStreamingDecoderResetStartsFreshSequence;
const
  SeqLen = 6;
  ToksA: array[0..5] of integer = (4, 8, 2, 11, 6, 1);
  ToksB: array[0..5] of integer = (9, 3, 10, 5, 7, 2);
var
  Full, Twin: TNNet;
  Session: TNNetStreamingDecoder;
  StepIn: TNNetVolume;
  T: integer;
begin
  RandSeed := 424242;
  Full := BuildTinyHybridLM(SeqLen);
  Twin := BuildTinyHybridLM(1);
  Session := nil;
  StepIn := TNNetVolume.Create(1, 1, 1);
  try
    Twin.CopyWeights(Full);
    Session := TNNetStreamingDecoder.Create(Twin, SeqLen);
    AssertTrue('hybrid has SSM layers', Session.SSMCount > 0);
    AssertTrue('hybrid has attention layers', Session.SDPACount > 0);
    // Pollute the session state with sequence A...
    Session.Reset();
    for T := 0 to SeqLen - 1 do
    begin
      StepIn.FData[0] := ToksA[T];
      Session.StepForward(StepIn, T);
    end;
    // ...then Reset and require sequence B to stream exactly.
    Session.Reset();
    AssertStreamMatchesFull(Full, Session, ToksB, 'after reset');
  finally
    StepIn.Free;
    Session.Free;
    Twin.Free;
    Full.Free;
  end;
end;

// TruncateTo() must implement the speculative-decoding rollback: stream the
// first CommitLen clean tokens, append two GARBAGE tokens (a rejected draft),
// TruncateTo(CommitLen) to discard their K/V, then continue with the clean
// sequence - every continued step must still match the clean sequence's full
// causal forward (note the continued steps' AbsPos resumes at CommitLen).
procedure TTestNeuralDecode.TestStreamingDecoderTruncateToRollsBack;
const
  SeqLen = 6;
  CommitLen = 4;
  Toks: array[0..5] of integer = (2, 10, 5, 8, 3, 7);
  Garbage: array[0..1] of integer = (11, 1);
var
  Full, Twin: TNNet;
  Session: TNNetStreamingDecoder;
  FullIn, FullOut, StepIn: TNNetVolume;
  T, D: integer;
begin
  RandSeed := 424242;
  Full := BuildTinyCausalLM(SeqLen);
  Twin := BuildTinyCausalLM(1);
  Session := nil;
  FullIn := TNNetVolume.Create(SeqLen, 1, 1);
  FullOut := TNNetVolume.Create();
  StepIn := TNNetVolume.Create(1, 1, 1);
  try
    Twin.CopyWeights(Full);
    // Cache budget covers the worst transient: clean prefix + draft block.
    Session := TNNetStreamingDecoder.Create(Twin, SeqLen + Length(Garbage));
    for T := 0 to SeqLen - 1 do FullIn.FData[T] := Toks[T];
    Full.Compute(FullIn);
    Full.GetOutput(FullOut);
    Session.Reset();
    // Clean committed prefix.
    for T := 0 to CommitLen - 1 do
    begin
      StepIn.FData[0] := Toks[T];
      Session.StepForward(StepIn, T);
      for D := 0 to FullOut.Depth - 1 do
        AssertEquals('prefix pos ' + IntToStr(T) + ' dim ' + IntToStr(D),
          FullOut[T, 0, D], Session.Output()[0, 0, D], 1e-5);
    end;
    // Speculative block: two garbage tokens at positions CommitLen and
    // CommitLen+1, then reject them all.
    for T := 0 to High(Garbage) do
    begin
      StepIn.FData[0] := Garbage[T];
      Session.StepForward(StepIn, CommitLen + T);
    end;
    Session.TruncateTo(CommitLen);
    // Continue with the clean sequence; positions resume at CommitLen.
    for T := CommitLen to SeqLen - 1 do
    begin
      StepIn.FData[0] := Toks[T];
      Session.StepForward(StepIn, T);
      for D := 0 to FullOut.Depth - 1 do
        AssertEquals('resumed pos ' + IntToStr(T) + ' dim ' + IntToStr(D),
          FullOut[T, 0, D], Session.Output()[0, 0, D], 1e-5);
    end;
  finally
    StepIn.Free;
    FullOut.Free;
    FullIn.Free;
    Session.Free;
    Twin.Free;
    Full.Free;
  end;
end;

// ---------------------------------------------------------------------------
// GenerateTokensStreamed / GenerateStringStreamed tests.
// ---------------------------------------------------------------------------

procedure TTestNeuralDecode.FullGreedyGenerate(Full: TNNet;
  var Toks: TNeuralIntegerArray; PromptLen, MaxNew: integer;
  out OutLen: integer);
var
  InV: TNNetVolume;
  ContextLen, T, NextTok: integer;
begin
  ContextLen := Full.GetFirstLayer().Output.SizeX;
  if Length(Toks) < ContextLen then SetLength(Toks, ContextLen);
  InV := TNNetVolume.Create(ContextLen, 1, 1);
  try
    OutLen := PromptLen;
    while (OutLen - PromptLen < MaxNew) and (OutLen < ContextLen) do
    begin
      // Full re-encode: the whole zero-padded prefix, every step. Trailing
      // zero pads are invisible to the read row: every layer is per-token
      // except the CAUSAL attention/SSM mixers, which never look right.
      InV.Fill(0);
      for T := 0 to OutLen - 1 do InV.FData[T] := Toks[T];
      Full.Compute(InV);
      NextTok := Full.GetLastLayer().Output.GetClassOnPixel(OutLen - 1, 0);
      Toks[OutLen] := NextTok;
      Inc(OutLen);
      if NextTok < 2 then Break; // same EOS rule as the streamed core
    end;
  finally
    InV.Free;
  end;
end;

procedure TTestNeuralDecode.RigHeadToConstantToken(NN: TNNet; Token: integer);
var
  Head: TNNetLayer;
  N: integer;
begin
  Head := NN.GetLastLayer();
  for N := 0 to Head.Neurons.Count - 1 do
  begin
    Head.Neurons[N].Weights.Fill(0);
    Head.Neurons[N].BiasWeight := 0;
  end;
  Head.Neurons[Token].BiasWeight := 5;
end;

// Streamed greedy generation from a width-1 RoPE-transformer twin must equal
// a hand-rolled full-re-encode greedy loop token for token (and in length).
procedure TTestNeuralDecode.TestGenerateTokensStreamedTransformerMatchesFullGreedy;
const
  SeqLen = 10;
  PromptLen = 3;
var
  Full, Twin: TNNet;
  Session: TNNetStreamingDecoder;
  FullToks, StreamToks: TNeuralIntegerArray;
  FullLen, StreamLen, T: integer;
begin
  RandSeed := 424242;
  Full := BuildTinyCausalLM(SeqLen);
  Twin := BuildTinyCausalLM(1);
  Session := nil;
  try
    Twin.CopyWeights(Full);
    Session := TNNetStreamingDecoder.Create(Twin, SeqLen);
    SetLength(FullToks, SeqLen);
    SetLength(StreamToks, PromptLen);
    FullToks[0] := 3; FullToks[1] := 7; FullToks[2] := 5;
    for T := 0 to PromptLen - 1 do StreamToks[T] := FullToks[T];

    FullGreedyGenerate(Full, FullToks, PromptLen, SeqLen - PromptLen, FullLen);
    StreamLen := GenerateTokensStreamed(Session, StreamToks, PromptLen,
      SeqLen - PromptLen, SeqLen);

    AssertEquals('same generated length', FullLen, StreamLen);
    for T := PromptLen to StreamLen - 1 do
      AssertEquals('token at pos ' + IntToStr(T), FullToks[T], StreamToks[T]);
  finally
    Session.Free;
    Twin.Free;
    Full.Free;
  end;
end;

// Same token-for-token gate for the recurrent family (DiagonalSSM twin with
// non-trivial per-channel recurrence weights).
procedure TTestNeuralDecode.TestGenerateTokensStreamedSSMMatchesFullGreedy;
const
  SeqLen = 10;
  PromptLen = 3;
var
  Full, Twin: TNNet;
  Session: TNNetStreamingDecoder;
  SSMFull: TNNetDiagonalSSM;
  FullToks, StreamToks: TNeuralIntegerArray;
  FullLen, StreamLen, T, i, D: integer;
begin
  RandSeed := 424242;
  Full := BuildTinySSMLM(SeqLen);
  Twin := BuildTinySSMLM(1);
  Session := nil;
  try
    // Non-trivial recurrence weights (the defaults are uniform per channel).
    SSMFull := nil;
    for i := 0 to Full.Layers.Count - 1 do
      if Full.Layers[i] is TNNetDiagonalSSM then
        SSMFull := TNNetDiagonalSSM(Full.Layers[i]);
    AssertTrue('SSM layer found', SSMFull <> nil);
    for D := 0 to csStreamDim - 1 do
    begin
      SSMFull.Neurons[0].Weights.FData[D] := (Random(2000) - 1000) / 1000;
      SSMFull.Neurons[1].Weights.FData[D] := 0.5 + Random(1000) / 1000;
      SSMFull.Neurons[2].Weights.FData[D] := 0.5 + Random(1000) / 1000;
      SSMFull.Neurons[3].Weights.FData[D] := (Random(2000) - 1000) / 1000;
    end;
    Twin.CopyWeights(Full);
    Session := TNNetStreamingDecoder.Create(Twin, SeqLen);
    SetLength(FullToks, SeqLen);
    SetLength(StreamToks, PromptLen);
    FullToks[0] := 5; FullToks[1] := 2; FullToks[2] := 9;
    for T := 0 to PromptLen - 1 do StreamToks[T] := FullToks[T];

    FullGreedyGenerate(Full, FullToks, PromptLen, SeqLen - PromptLen, FullLen);
    StreamLen := GenerateTokensStreamed(Session, StreamToks, PromptLen,
      SeqLen - PromptLen, SeqLen);

    AssertEquals('same generated length', FullLen, StreamLen);
    for T := PromptLen to StreamLen - 1 do
      AssertEquals('token at pos ' + IntToStr(T), FullToks[T], StreamToks[T]);
  finally
    Session.Free;
    Twin.Free;
    Full.Free;
  end;
end;

// A TNNetSamplerGreedy (argmax sampler from neuralvolume) must produce the
// exact same stream as the Sampler=nil internal greedy path.
procedure TTestNeuralDecode.TestGenerateTokensStreamedGreedySamplerMatchesNilPath;
const
  SeqLen = 9;
  PromptLen = 2;
var
  Full, Twin: TNNet;
  Session: TNNetStreamingDecoder;
  Sampler: TNNetSamplerGreedy;
  NilToks, SamplerToks: TNeuralIntegerArray;
  NilLen, SamplerLen, T: integer;
begin
  RandSeed := 424242;
  Full := BuildTinyCausalLM(SeqLen);
  Twin := BuildTinyCausalLM(1);
  Session := nil;
  Sampler := TNNetSamplerGreedy.Create();
  try
    Twin.CopyWeights(Full);
    Session := TNNetStreamingDecoder.Create(Twin, SeqLen);
    SetLength(NilToks, PromptLen);
    SetLength(SamplerToks, PromptLen);
    NilToks[0] := 6; NilToks[1] := 11;
    SamplerToks[0] := 6; SamplerToks[1] := 11;
    // The routine Reset()s the session itself, so it can be reused.
    NilLen := GenerateTokensStreamed(Session, NilToks, PromptLen,
      SeqLen - PromptLen, SeqLen);
    SamplerLen := GenerateTokensStreamed(Session, SamplerToks, PromptLen,
      SeqLen - PromptLen, SeqLen, Sampler);
    AssertEquals('same generated length', NilLen, SamplerLen);
    for T := PromptLen to NilLen - 1 do
      AssertEquals('token at pos ' + IntToStr(T), NilToks[T], SamplerToks[T]);
  finally
    Sampler.Free;
    Session.Free;
    Twin.Free;
    Full.Free;
  end;
end;

// EOS rule: rig the logit head so argmax is ALWAYS token 1 (EOS); generation
// must stop after storing exactly that one token.
procedure TTestNeuralDecode.TestGenerateTokensStreamedStopsAtEOS;
const
  PromptLen = 3;
var
  Full, Twin: TNNet;
  Session: TNNetStreamingDecoder;
  Toks: TNeuralIntegerArray;
  OutLen: integer;
begin
  RandSeed := 424242;
  Full := BuildTinySSMLM(16);
  Twin := BuildTinySSMLM(1);
  Session := nil;
  try
    RigHeadToConstantToken(Full, 1);
    Twin.CopyWeights(Full);
    Session := TNNetStreamingDecoder.Create(Twin, 16);
    SetLength(Toks, PromptLen);
    Toks[0] := 4; Toks[1] := 9; Toks[2] := 6;
    OutLen := GenerateTokensStreamed(Session, Toks, PromptLen, 10, 16);
    AssertEquals('stopped right after EOS', PromptLen + 1, OutLen);
    AssertEquals('EOS token stored', 1, Toks[PromptLen]);
  finally
    Session.Free;
    Twin.Free;
    Full.Free;
  end;
end;

// MaxNewTokens / MaxTotalLen caps, made deterministic by rigging the head to
// a constant non-special token (7) so EOS never fires.
procedure TTestNeuralDecode.TestGenerateTokensStreamedRespectsCaps;
const
  PromptLen = 2;
var
  Full, Twin: TNNet;
  Session: TNNetStreamingDecoder;
  Toks: TNeuralIntegerArray;
  OutLen, T: integer;
begin
  RandSeed := 424242;
  Full := BuildTinySSMLM(32);
  Twin := BuildTinySSMLM(1);
  Session := nil;
  try
    RigHeadToConstantToken(Full, 7);
    Twin.CopyWeights(Full);
    Session := TNNetStreamingDecoder.Create(Twin, 32);
    // MaxNewTokens caps generation: 2 prompt + exactly 3 new tokens.
    SetLength(Toks, PromptLen);
    Toks[0] := 3; Toks[1] := 8;
    OutLen := GenerateTokensStreamed(Session, Toks, PromptLen, 3, 32);
    AssertEquals('MaxNewTokens cap', PromptLen + 3, OutLen);
    for T := PromptLen to OutLen - 1 do
      AssertEquals('constant token at pos ' + IntToStr(T), 7, Toks[T]);
    // MaxTotalLen clips even a generous MaxNewTokens budget.
    SetLength(Toks, 0);
    SetLength(Toks, PromptLen);
    Toks[0] := 3; Toks[1] := 8;
    OutLen := GenerateTokensStreamed(Session, Toks, PromptLen, 50, 4);
    AssertEquals('MaxTotalLen cap', 4, OutLen);
  finally
    Session.Free;
    Twin.Free;
    Full.Free;
  end;
end;

// v1 is width-1 only: a wider (speculative-verify-style) session must be
// rejected with EArgumentException, not silently mis-decoded.
procedure TTestNeuralDecode.TestGenerateTokensStreamedRejectsWideSession;
var
  Twin: TNNet;
  Session: TNNetStreamingDecoder;
  Toks: TNeuralIntegerArray;
  Raised: boolean;
begin
  RandSeed := 424242;
  Twin := BuildTinyCausalLM(2); // width 2, not 1
  Session := nil;
  try
    Session := TNNetStreamingDecoder.Create(Twin, 8);
    SetLength(Toks, 2);
    Toks[0] := 3; Toks[1] := 4;
    Raised := false;
    try
      GenerateTokensStreamed(Session, Toks, 2, 4, 8);
    except
      on EArgumentException do Raised := true;
    end;
    AssertTrue('width-2 session rejected', Raised);
  finally
    Session.Free;
    Twin.Free;
  end;
end;

// String wrapper: tokenizing the prompt, running the core and detokenizing
// the continuation (space-joined: TStringListInt.TokenizerHasSeparator=True;
// display stops at the first token < 2) must match a manual run of
// GenerateTokensStreamed + Dict.DeTokenize.
procedure TTestNeuralDecode.TestGenerateStringStreamedMatchesTokenCore;
const
  SeqLen = 9;
  Words: array[0..11] of string = ('<eos>', '<pad>', 'apple', 'blue', 'cat',
    'dog', 'egg', 'fox', 'gold', 'hat', 'ice', 'jam');
var
  Full, Twin: TNNet;
  Session: TNNetStreamingDecoder;
  Dict: TStringListInt;
  Toks: TNeuralIntegerArray;
  PromptLen, TotalLen, T: integer;
  Got, Expected: string;
begin
  RandSeed := 424242;
  Full := BuildTinyCausalLM(SeqLen);
  Twin := BuildTinyCausalLM(1);
  Session := nil;
  Dict := TStringListInt.Create();
  try
    Dict.Sorted := true;
    for T := 0 to High(Words) do Dict.Add(Words[T]);
    Dict.SaveCurrentPosition(); // token id = sorted index (0='<eos>',...)
    AssertEquals('vocab matches the tiny LM', csStreamVocab,
      Dict.GetVocabCount());
    Twin.CopyWeights(Full);
    Session := TNNetStreamingDecoder.Create(Twin, SeqLen);

    Got := GenerateStringStreamed(Session, Dict, 'cat dog egg',
      SeqLen - 3, SeqLen);

    // Manual reference: same prompt ids, same core, manual detokenize.
    Dict.Tokenize('cat dog egg', Toks);
    PromptLen := Length(Toks);
    AssertEquals('prompt tokenizes to 3 ids', 3, PromptLen);
    TotalLen := GenerateTokensStreamed(Session, Toks, PromptLen,
      SeqLen - PromptLen, SeqLen);
    Expected := 'cat dog egg';
    for T := PromptLen to TotalLen - 1 do
    begin
      if Toks[T] < 2 then Break;
      Expected := Expected + ' ' + Dict.DeTokenize(Toks[T]);
    end;
    AssertEquals('wrapper equals token core + detokenize', Expected, Got);
    AssertTrue('result preserves the prompt prefix',
      Copy(Got, 1, Length('cat dog egg')) = 'cat dog egg');
  finally
    Dict.Free;
    Session.Free;
    Twin.Free;
    Full.Free;
  end;
end;

initialization
  RegisterTest(TTestNeuralDecode);

end.
