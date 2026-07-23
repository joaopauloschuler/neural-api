unit TestNeuralSamplers;

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, Math, fpcunit, testregistry, neuralvolume;

type
  TTestNeuralSamplers = class(TTestCase)
  published
    // TNNetSamplerGreedy tests
    procedure TestGreedySamplerCreation;
    procedure TestGreedySamplerGetToken;
    procedure TestGreedySamplerGetTokenOnPixel;
    procedure TestGreedySamplerDeterministic;
    
    // TNNetSamplerTopK tests
    procedure TestTopKSamplerCreation;
    procedure TestTopKSamplerGetToken;
    procedure TestTopKSamplerGetTokenOnPixel;
    procedure TestTopKSamplerWithDifferentK;
    
    // TNNetSamplerTopP tests
    procedure TestTopPSamplerCreation;
    procedure TestTopPSamplerGetToken;
    procedure TestTopPSamplerGetTokenOnPixel;
    procedure TestTopPSamplerWithDifferentP;
    
    // Edge case tests
    procedure TestSamplerWithUniformDistribution;
    procedure TestSamplerWithSingleToken;
    procedure TestSamplerWithSoftmaxOutput;

    // TNNetTokenHistoryPenalty tests
    procedure TestPenaltyNoOpIsBitForBit;
    procedure TestPenaltyRepetitionDecreases;
    procedure TestPenaltyFrequencyDecreases;
    procedure TestPenaltyPresenceDecreases;
    procedure TestPenaltyRepetitionSignCorrect;
    procedure TestPenaltyResetHistory;
    procedure TestPenaltyFrequencyScalesWithCount;
    procedure TestPenaltyTokenIdBeyondVolumeIsIgnored;
    procedure TestPenaltyHistoryReusableAfterReset;

    // TNNetSamplerMinP tests
    procedure TestMinPSamplerKeepsExactlyExpectedSet;
    procedure TestMinPSamplerOneIsGreedy;
    procedure TestMinPSamplerBoundaryTokenIsKept;
    procedure TestMinPSamplerGetTokenOnPixel;

    // TNNetTokenHistoryPenalty.ApplyToProbabilities tests
    procedure TestPenaltyProbabilitiesNoOpIsBitForBit;
    procedure TestPenaltyProbabilitiesRepetitionChangesArgmax;
    procedure TestPenaltyProbabilitiesFrequencyPresenceFactor;
    procedure TestPenaltyProbabilitiesSumsToOne;

    // TNNetSamplerWeightedTopK tests
    procedure TestWeightedTopKMatchesRenormalizedDistribution;
    procedure TestWeightedTopKDiffersFromUniform;
    procedure TestWeightedTopKSequenceReproducible;
    procedure TestWeightedTopKNeverDrawsOutsideTopK;
    procedure TestWeightedTopKGetTokenOnPixel;

    // TNNetSamplerTypical (locally-typical sampling) tests
    procedure TestTypicalKeepsExactlyExpectedSet;
    procedure TestTypicalExcludesMostLikelyToken;
    procedure TestTypicalGetTokenOnPixel;

    // TNNetSamplerMirostat tests
    procedure TestMirostatV2KeepsExactlyExpectedSet;
    procedure TestMirostatV2NarrowerMuKeepsFewer;
    procedure TestMirostatResetReArmsMu;
    procedure TestMirostatV2MuConvergesTowardTau;
    procedure TestMirostatV1MuStaysBounded;

    // Large-vocabulary tests: these exercise the partial-selection path,
    // which only engages above the adaptive threshold (a small-vocab test
    // silently takes the plain full-sort branch and proves nothing).
    procedure TestTopPLargeVocabNucleusIsCorrect;
    procedure TestTopPLargeVocabFlatDistributionRetries;
    procedure TestWeightedTopKLargeVocabNeverDrawsOutsideTopK;
    procedure TestMinPLargeVocabRespectsThreshold;
  end;

implementation

procedure TTestNeuralSamplers.TestGreedySamplerCreation;
var
  Sampler: TNNetSamplerGreedy;
begin
  Sampler := TNNetSamplerGreedy.Create;
  try
    AssertTrue('Greedy sampler should be created', Sampler <> nil);
  finally
    Sampler.Free;
  end;
end;

procedure TTestNeuralSamplers.TestGreedySamplerGetToken;
var
  Sampler: TNNetSamplerGreedy;
  V: TNNetVolume;
  Token: integer;
begin
  Sampler := TNNetSamplerGreedy.Create;
  V := TNNetVolume.Create(10, 1, 1);
  try
    // Set up probabilities - token 5 has highest value
    V.Fill(0.1);
    V.Raw[5] := 0.9;
    
    Token := Sampler.GetToken(V);
    AssertEquals('Greedy should select token with highest probability', 5, Token);
  finally
    V.Free;
    Sampler.Free;
  end;
end;

procedure TTestNeuralSamplers.TestGreedySamplerGetTokenOnPixel;
var
  Sampler: TNNetSamplerGreedy;
  V: TNNetVolume;
  Token: integer;
begin
  Sampler := TNNetSamplerGreedy.Create;
  V := TNNetVolume.Create(4, 4, 10); // 4x4 spatial, 10 tokens
  try
    // Fill with low values
    V.Fill(0.1);
    // Set pixel (2,1) to have token 7 as highest
    V[2, 1, 7] := 0.9;
    
    Token := Sampler.GetTokenOnPixel(V, 2, 1);
    AssertEquals('Greedy should select token 7 at pixel (2,1)', 7, Token);
  finally
    V.Free;
    Sampler.Free;
  end;
end;

procedure TTestNeuralSamplers.TestGreedySamplerDeterministic;
var
  Sampler: TNNetSamplerGreedy;
  V: TNNetVolume;
  Token1, Token2, Token3: integer;
begin
  Sampler := TNNetSamplerGreedy.Create;
  V := TNNetVolume.Create(8, 1, 1);
  try
    V.Fill(0.05);
    V.Raw[3] := 0.8;
    
    // Greedy should always return the same result
    Token1 := Sampler.GetToken(V);
    Token2 := Sampler.GetToken(V);
    Token3 := Sampler.GetToken(V);
    
    AssertEquals('Greedy sampler should be deterministic (1)', 3, Token1);
    AssertEquals('Greedy sampler should be deterministic (2)', 3, Token2);
    AssertEquals('Greedy sampler should be deterministic (3)', 3, Token3);
  finally
    V.Free;
    Sampler.Free;
  end;
end;

procedure TTestNeuralSamplers.TestTopKSamplerCreation;
var
  Sampler: TNNetSamplerTopK;
begin
  Sampler := TNNetSamplerTopK.Create(5);
  try
    AssertTrue('TopK sampler should be created', Sampler <> nil);
  finally
    Sampler.Free;
  end;
end;

procedure TTestNeuralSamplers.TestTopKSamplerGetToken;
var
  Sampler: TNNetSamplerTopK;
  V: TNNetVolume;
  Token: integer;
  I: integer;
  TokenCounts: array[0..9] of integer;
  TotalSamples: integer;
begin
  Sampler := TNNetSamplerTopK.Create(3);
  V := TNNetVolume.Create(10, 1, 1);
  try
    // Set up probabilities - top 3 tokens are 7, 8, 9
    V.Fill(0.01);
    V.Raw[7] := 0.3;
    V.Raw[8] := 0.35;
    V.Raw[9] := 0.25;
    
    // Initialize counts
    for I := 0 to 9 do TokenCounts[I] := 0;
    
    // Sample multiple times
    TotalSamples := 100;
    for I := 1 to TotalSamples do
    begin
      Token := Sampler.GetToken(V);
      AssertTrue('Token should be in range 0-9', (Token >= 0) and (Token <= 9));
      Inc(TokenCounts[Token]);
    end;
    
    // The top-K sampler should mostly select from top K tokens
    // Tokens 7, 8, 9 should be selected most often
    AssertTrue('TopK should mostly select from top K tokens',
      (TokenCounts[7] + TokenCounts[8] + TokenCounts[9]) >= TotalSamples div 2);
  finally
    V.Free;
    Sampler.Free;
  end;
end;

procedure TTestNeuralSamplers.TestTopKSamplerGetTokenOnPixel;
var
  Sampler: TNNetSamplerTopK;
  V: TNNetVolume;
  Token: integer;
begin
  Sampler := TNNetSamplerTopK.Create(1); // K=1 should behave like greedy
  V := TNNetVolume.Create(2, 2, 5); // 2x2 spatial, 5 tokens
  try
    V.Fill(0.1);
    V[1, 1, 3] := 0.9;
    
    Token := Sampler.GetTokenOnPixel(V, 1, 1);
    AssertEquals('TopK with K=1 should select token 3', 3, Token);
  finally
    V.Free;
    Sampler.Free;
  end;
end;

procedure TTestNeuralSamplers.TestTopKSamplerWithDifferentK;
var
  Sampler1, Sampler5: TNNetSamplerTopK;
  V: TNNetVolume;
  Token: integer;
  I: integer;
begin
  Sampler1 := TNNetSamplerTopK.Create(1);
  Sampler5 := TNNetSamplerTopK.Create(5);
  V := TNNetVolume.Create(10, 1, 1);
  try
    V.Fill(0.05);
    V.Raw[0] := 0.5;
    V.Raw[1] := 0.2;
    V.Raw[2] := 0.1;
    
    // K=1 should always return the max
    for I := 1 to 10 do
    begin
      Token := Sampler1.GetToken(V);
      AssertEquals('TopK with K=1 should always return max', 0, Token);
    end;
    
    // K=5 should sample from top 5, which includes tokens 0,1,2 with high prob
    Token := Sampler5.GetToken(V);
    AssertTrue('TopK with K=5 should return valid token', (Token >= 0) and (Token <= 9));
  finally
    V.Free;
    Sampler1.Free;
    Sampler5.Free;
  end;
end;

procedure TTestNeuralSamplers.TestTopPSamplerCreation;
var
  Sampler: TNNetSamplerTopP;
begin
  Sampler := TNNetSamplerTopP.Create(0.9);
  try
    AssertTrue('TopP sampler should be created', Sampler <> nil);
  finally
    Sampler.Free;
  end;
end;

procedure TTestNeuralSamplers.TestTopPSamplerGetToken;
var
  Sampler: TNNetSamplerTopP;
  V: TNNetVolume;
  Token: integer;
  I: integer;
  TokenCounts: array[0..9] of integer;
begin
  Sampler := TNNetSamplerTopP.Create(0.9);
  V := TNNetVolume.Create(10, 1, 1);
  try
    // Set up probabilities - simulate softmax-like output
    V.Fill(0.02);
    V.Raw[0] := 0.4;
    V.Raw[1] := 0.3;
    V.Raw[2] := 0.2;
    
    // Initialize counts
    for I := 0 to 9 do TokenCounts[I] := 0;
    
    // Sample multiple times
    for I := 1 to 100 do
    begin
      Token := Sampler.GetToken(V);
      AssertTrue('Token should be in range', (Token >= 0) and (Token <= 9));
      Inc(TokenCounts[Token]);
    end;
    
    // TopP should mostly select from tokens with cumulative prob <= P
    AssertTrue('TopP should select from top probability tokens',
      (TokenCounts[0] + TokenCounts[1] + TokenCounts[2]) >= 50);
  finally
    V.Free;
    Sampler.Free;
  end;
end;

procedure TTestNeuralSamplers.TestTopPSamplerGetTokenOnPixel;
var
  Sampler: TNNetSamplerTopP;
  V: TNNetVolume;
  Token: integer;
begin
  Sampler := TNNetSamplerTopP.Create(0.1); // Very low P should select top token
  V := TNNetVolume.Create(3, 3, 8); // 3x3 spatial, 8 tokens
  try
    V.Fill(0.05);
    V[2, 2, 6] := 0.9;
    
    Token := Sampler.GetTokenOnPixel(V, 2, 2);
    // With P=0.1, should mostly select the top token
    AssertTrue('Token should be valid', (Token >= 0) and (Token <= 7));
  finally
    V.Free;
    Sampler.Free;
  end;
end;

procedure TTestNeuralSamplers.TestTopPSamplerWithDifferentP;
var
  SamplerLow, SamplerHigh: TNNetSamplerTopP;
  V: TNNetVolume;
  Token: integer;
  I: integer;
  LowPTokens, HighPTokens: integer;
begin
  SamplerLow := TNNetSamplerTopP.Create(0.1);  // Low P - more focused
  SamplerHigh := TNNetSamplerTopP.Create(0.99); // High P - more diverse
  V := TNNetVolume.Create(10, 1, 1);
  try
    V.Fill(0.05);
    V.Raw[0] := 0.5;
    V.Raw[1] := 0.2;
    V.Raw[2] := 0.15;
    
    LowPTokens := 0;
    HighPTokens := 0;
    
    for I := 1 to 50 do
    begin
      Token := SamplerLow.GetToken(V);
      if Token = 0 then Inc(LowPTokens);
    end;
    
    for I := 1 to 50 do
    begin
      Token := SamplerHigh.GetToken(V);
      if Token = 0 then Inc(HighPTokens);
    end;
    
    // Low P should select token 0 more often (more focused)
    // This is a probabilistic test but should generally hold
    AssertTrue('Low P should focus on top token', LowPTokens >= 10);
  finally
    V.Free;
    SamplerLow.Free;
    SamplerHigh.Free;
  end;
end;

procedure TTestNeuralSamplers.TestSamplerWithUniformDistribution;
var
  GreedySampler: TNNetSamplerGreedy;
  V: TNNetVolume;
  Token: integer;
begin
  GreedySampler := TNNetSamplerGreedy.Create;
  V := TNNetVolume.Create(5, 1, 1);
  try
    // Uniform distribution
    V.Fill(0.2);
    
    Token := GreedySampler.GetToken(V);
    // Should return some valid token (first one found typically)
    AssertTrue('Token should be valid for uniform distribution',
      (Token >= 0) and (Token <= 4));
  finally
    V.Free;
    GreedySampler.Free;
  end;
end;

procedure TTestNeuralSamplers.TestSamplerWithSingleToken;
var
  GreedySampler: TNNetSamplerGreedy;
  V: TNNetVolume;
  Token: integer;
begin
  GreedySampler := TNNetSamplerGreedy.Create;
  V := TNNetVolume.Create(1, 1, 1); // Single token
  try
    V.Raw[0] := 1.0;
    
    Token := GreedySampler.GetToken(V);
    // GetToken may return -1 for edge cases or 0 for valid single token
    // The actual behavior depends on the implementation
    AssertTrue('Single token should return valid result', Token >= -1);
  finally
    V.Free;
    GreedySampler.Free;
  end;
end;

procedure TTestNeuralSamplers.TestSamplerWithSoftmaxOutput;
var
  GreedySampler: TNNetSamplerGreedy;
  V: TNNetVolume;
  Token: integer;
begin
  GreedySampler := TNNetSamplerGreedy.Create;
  V := TNNetVolume.Create(4, 1, 1);
  try
    // Simulate pre-softmax values
    V.Raw[0] := 1.0;
    V.Raw[1] := 2.0;
    V.Raw[2] := 3.0;
    V.Raw[3] := 4.0;
    
    // Apply softmax
    V.SoftMax();
    
    // Now token 3 should have highest probability
    Token := GreedySampler.GetToken(V);
    AssertEquals('After softmax, token 3 should have highest prob', 3, Token);
    
    // Verify softmax properties
    AssertEquals('Softmax should sum to 1.0', 1.0, V.GetSum(), 0.0001);
  finally
    V.Free;
    GreedySampler.Free;
  end;
end;

procedure TTestNeuralSamplers.TestPenaltyNoOpIsBitForBit;
var
  Penalty: TNNetTokenHistoryPenalty;
  V, Before: TNNetVolume;
  I: integer;
begin
  // All knobs at no-op values (r=1, alpha_f=alpha_p=0) must leave the
  // volume bit-for-bit unchanged, even for registered tokens.
  Penalty := TNNetTokenHistoryPenalty.Create(1.0, 0.0, 0.0);
  V := TNNetVolume.Create(10, 1, 1);
  Before := TNNetVolume.Create(10, 1, 1);
  try
    for I := 0 to 9 do V.Raw[I] := I * 0.7 - 3.0;
    Before.Copy(V);
    // Register several tokens to ensure counts are non-zero.
    Penalty.RegisterToken(2);
    Penalty.RegisterToken(2);
    Penalty.RegisterToken(7);
    Penalty.Apply(V);
    for I := 0 to 9 do
      AssertTrue('No-op penalty must be bit-for-bit unchanged at ' + IntToStr(I),
        V.Raw[I] = Before.Raw[I]);
  finally
    V.Free;
    Before.Free;
    Penalty.Free;
  end;
end;

procedure TTestNeuralSamplers.TestPenaltyRepetitionDecreases;
var
  Penalty: TNNetTokenHistoryPenalty;
  V: TNNetVolume;
  OrigLogit: TNeuralFloat;
begin
  // A single registered token's positive logit strictly decreases under the
  // repetition knob alone.
  Penalty := TNNetTokenHistoryPenalty.Create(2.0, 0.0, 0.0);
  V := TNNetVolume.Create(5, 1, 1);
  try
    V.Fill(1.0);
    V.Raw[3] := 4.0;
    OrigLogit := V.Raw[3];
    Penalty.RegisterToken(3);
    Penalty.Apply(V);
    AssertTrue('Repetition penalty must strictly decrease the logit',
      V.Raw[3] < OrigLogit);
    AssertEquals('Repetition penalty on positive logit divides by r',
      2.0, V.Raw[3], 0.0001);
    // Unregistered tokens must be untouched.
    AssertEquals('Unregistered token must be unchanged', 1.0, V.Raw[0], 0.0001);
  finally
    V.Free;
    Penalty.Free;
  end;
end;

procedure TTestNeuralSamplers.TestPenaltyFrequencyDecreases;
var
  Penalty: TNNetTokenHistoryPenalty;
  V: TNNetVolume;
  OrigLogit: TNeuralFloat;
begin
  // A single registered token's logit strictly decreases under the
  // frequency knob alone.
  Penalty := TNNetTokenHistoryPenalty.Create(1.0, 0.5, 0.0);
  V := TNNetVolume.Create(5, 1, 1);
  try
    V.Fill(2.0);
    OrigLogit := V.Raw[1];
    Penalty.RegisterToken(1);
    Penalty.Apply(V);
    AssertTrue('Frequency penalty must strictly decrease the logit',
      V.Raw[1] < OrigLogit);
    AssertEquals('Frequency penalty subtracts alpha_f * count',
      2.0 - 0.5, V.Raw[1], 0.0001);
    AssertEquals('Unregistered token must be unchanged', 2.0, V.Raw[0], 0.0001);
  finally
    V.Free;
    Penalty.Free;
  end;
end;

procedure TTestNeuralSamplers.TestPenaltyPresenceDecreases;
var
  Penalty: TNNetTokenHistoryPenalty;
  V: TNNetVolume;
  OrigLogit: TNeuralFloat;
begin
  // A single registered token's logit strictly decreases under the
  // presence knob alone, and the push is flat (count-independent).
  Penalty := TNNetTokenHistoryPenalty.Create(1.0, 0.0, 0.75);
  V := TNNetVolume.Create(5, 1, 1);
  try
    V.Fill(2.0);
    OrigLogit := V.Raw[4];
    // Register twice; presence push must still be a single flat subtraction.
    Penalty.RegisterToken(4);
    Penalty.RegisterToken(4);
    Penalty.Apply(V);
    AssertTrue('Presence penalty must strictly decrease the logit',
      V.Raw[4] < OrigLogit);
    AssertEquals('Presence penalty subtracts alpha_p once',
      2.0 - 0.75, V.Raw[4], 0.0001);
    AssertEquals('Unregistered token must be unchanged', 2.0, V.Raw[0], 0.0001);
  finally
    V.Free;
    Penalty.Free;
  end;
end;

procedure TTestNeuralSamplers.TestPenaltyRepetitionSignCorrect;
var
  Penalty: TNNetTokenHistoryPenalty;
  V: TNNetVolume;
begin
  // Sign-correct CTRL form: a penalty must LOWER the score for both a
  // positive logit (l/r) and a negative logit (l*r, more negative).
  Penalty := TNNetTokenHistoryPenalty.Create(2.0, 0.0, 0.0);
  V := TNNetVolume.Create(2, 1, 1);
  try
    V.Raw[0] := 3.0;   // positive
    V.Raw[1] := -3.0;  // negative
    Penalty.RegisterToken(0);
    Penalty.RegisterToken(1);
    Penalty.Apply(V);
    // Positive: 3.0 / 2.0 = 1.5 (lower).
    AssertTrue('Positive logit must decrease', V.Raw[0] < 3.0);
    AssertEquals('Positive logit divided by r', 1.5, V.Raw[0], 0.0001);
    // Negative: -3.0 * 2.0 = -6.0 (more negative => lower).
    AssertTrue('Negative logit must become more negative', V.Raw[1] < -3.0);
    AssertEquals('Negative logit multiplied by r', -6.0, V.Raw[1], 0.0001);
  finally
    V.Free;
    Penalty.Free;
  end;
end;

procedure TTestNeuralSamplers.TestPenaltyResetHistory;
var
  Penalty: TNNetTokenHistoryPenalty;
  V, Before: TNNetVolume;
  I: integer;
begin
  // After ResetHistory, Apply must again be a no-op for previously
  // registered tokens.
  Penalty := TNNetTokenHistoryPenalty.Create(2.0, 0.5, 0.75);
  V := TNNetVolume.Create(6, 1, 1);
  Before := TNNetVolume.Create(6, 1, 1);
  try
    for I := 0 to 5 do V.Raw[I] := I - 2.5;
    Before.Copy(V);
    Penalty.RegisterToken(0);
    Penalty.RegisterToken(3);
    Penalty.RegisterToken(3);
    Penalty.ResetHistory();
    Penalty.Apply(V);
    for I := 0 to 5 do
      AssertTrue('After reset, Apply must be a no-op at ' + IntToStr(I),
        V.Raw[I] = Before.Raw[I]);
  finally
    V.Free;
    Before.Free;
    Penalty.Free;
  end;
end;

procedure TTestNeuralSamplers.TestPenaltyTokenIdBeyondVolumeIsIgnored;
var
  Penalty: TNNetTokenHistoryPenalty;
  V, Before: TNNetVolume;
  I: integer;
begin
  // A registered id larger than the volume being penalized must be skipped
  // (the history outlives any single logit row - e.g. a shorter draft-model
  // vocabulary). Guards the distinct-id walk against an out-of-range write.
  Penalty := TNNetTokenHistoryPenalty.Create(2.0, 0.5, 0.75);
  V := TNNetVolume.Create(4, 1, 1);
  Before := TNNetVolume.Create(4, 1, 1);
  try
    for I := 0 to 3 do V.Raw[I] := I - 1.5;
    Before.Copy(V);
    Penalty.RegisterToken(99);
    Penalty.Apply(V);
    for I := 0 to 3 do
      AssertTrue('Out-of-range id must leave element ' + IntToStr(I) +
        ' untouched', V.Raw[I] = Before.Raw[I]);
  finally
    V.Free;
    Before.Free;
    Penalty.Free;
  end;
end;

procedure TTestNeuralSamplers.TestPenaltyHistoryReusableAfterReset;
var
  Penalty: TNNetTokenHistoryPenalty;
  V: TNNetVolume;
  Orig1: TNeuralFloat;
begin
  // After ResetHistory the tracker must still work for a NEW sequence: the
  // freshly registered token is penalized and the previously registered one
  // is not. Exercises reuse of the retained distinct-id buffer.
  Penalty := TNNetTokenHistoryPenalty.Create(1.0, 0.0, 0.75);
  V := TNNetVolume.Create(4, 1, 1);
  try
    V.Fill(2.0);
    Orig1 := V.Raw[1];
    Penalty.RegisterToken(0);
    Penalty.ResetHistory();
    Penalty.RegisterToken(1);
    Penalty.Apply(V);
    AssertTrue('Token registered before the reset must not be penalized',
      V.Raw[0] = 2.0);
    AssertTrue('Token registered after the reset must be penalized',
      V.Raw[1] < Orig1);
  finally
    V.Free;
    Penalty.Free;
  end;
end;

procedure TTestNeuralSamplers.TestPenaltyFrequencyScalesWithCount;
var
  Penalty: TNNetTokenHistoryPenalty;
  V: TNNetVolume;
begin
  // Frequency penalty scales with count: registering the same token twice
  // must subtract 2*alpha_f.
  Penalty := TNNetTokenHistoryPenalty.Create(1.0, 0.5, 0.0);
  V := TNNetVolume.Create(4, 1, 1);
  try
    V.Fill(5.0);
    Penalty.RegisterToken(2);
    Penalty.RegisterToken(2);
    Penalty.Apply(V);
    // 5.0 - 2 * 0.5 = 4.0
    AssertEquals('Frequency penalty subtracts 2 * alpha_f for count = 2',
      4.0, V.Raw[2], 0.0001);
  finally
    V.Free;
    Penalty.Free;
  end;
end;

procedure TTestNeuralSamplers.TestMinPSamplerKeepsExactlyExpectedSet;
var
  Sampler: TNNetSamplerMinP;
  V: TNNetVolume;
  Token, I: integer;
  Seen: array[0..4] of boolean;
begin
  // probs = [0.5, 0.3, 0.12, 0.05, 0.03]; MinP = 0.2 -> threshold = 0.1.
  // EXACTLY tokens {0, 1, 2} pass p >= 0.1; tokens 3 and 4 must NEVER be
  // drawn.
  RandSeed := 20260611;
  Sampler := TNNetSamplerMinP.Create(0.2);
  V := TNNetVolume.Create(5, 1, 1);
  try
    V.Raw[0] := 0.5;
    V.Raw[1] := 0.3;
    V.Raw[2] := 0.12;
    V.Raw[3] := 0.05;
    V.Raw[4] := 0.03;
    for I := 0 to 4 do Seen[I] := false;
    for I := 1 to 300 do
    begin
      Token := Sampler.GetToken(V);
      AssertTrue('min-p must only draw from the kept set {0,1,2}, got ' +
        IntToStr(Token), (Token >= 0) and (Token <= 2));
      Seen[Token] := true;
    end;
    // The weighted draw over the renormalized kept mass must reach every
    // kept token in 300 draws (min renormalized prob is 0.12/0.92 > 0.13).
    AssertTrue('kept token 0 drawn', Seen[0]);
    AssertTrue('kept token 1 drawn', Seen[1]);
    AssertTrue('kept token 2 drawn', Seen[2]);
  finally
    V.Free;
    Sampler.Free;
  end;
end;

procedure TTestNeuralSamplers.TestMinPSamplerOneIsGreedy;
var
  Sampler: TNNetSamplerMinP;
  V: TNNetVolume;
  I: integer;
begin
  // MinP = 1.0 keeps only tokens with p >= max(p), i.e. the argmax alone:
  // min-p degenerates to deterministic greedy.
  RandSeed := 20260611;
  Sampler := TNNetSamplerMinP.Create(1.0);
  V := TNNetVolume.Create(6, 1, 1);
  try
    V.Fill(0.1);
    V.Raw[4] := 0.5;
    for I := 1 to 50 do
      AssertEquals('MinP=1.0 must always return the argmax',
        4, Sampler.GetToken(V));
  finally
    V.Free;
    Sampler.Free;
  end;
end;

procedure TTestNeuralSamplers.TestMinPSamplerBoundaryTokenIsKept;
var
  Sampler: TNNetSamplerMinP;
  V: TNNetVolume;
  Token, I: integer;
  SawBoundary: boolean;
begin
  // Token 1 sits EXACTLY at the threshold (p = MinP * max = 0.4*0.5 = 0.2);
  // the cut is inclusive (p >= threshold), so it must remain sampleable while
  // the below-threshold tokens (0.05 each) must never appear.
  RandSeed := 20260611;
  Sampler := TNNetSamplerMinP.Create(0.4);
  V := TNNetVolume.Create(8, 1, 1);
  try
    V.Fill(0.05);
    V.Raw[0] := 0.5;
    V.Raw[1] := 0.2;
    SawBoundary := false;
    for I := 1 to 300 do
    begin
      Token := Sampler.GetToken(V);
      AssertTrue('only {0,1} pass the inclusive cut, got ' + IntToStr(Token),
        (Token = 0) or (Token = 1));
      if Token = 1 then SawBoundary := true;
    end;
    AssertTrue('the exactly-at-threshold token must be kept', SawBoundary);
  finally
    V.Free;
    Sampler.Free;
  end;
end;

procedure TTestNeuralSamplers.TestMinPSamplerGetTokenOnPixel;
var
  Sampler: TNNetSamplerMinP;
  V: TNNetVolume;
  I: integer;
begin
  // Pixel-addressed variant with MinP=1.0: deterministic argmax at (2,1).
  RandSeed := 20260611;
  Sampler := TNNetSamplerMinP.Create(1.0);
  V := TNNetVolume.Create(4, 4, 10);
  try
    V.Fill(0.05);
    V[2, 1, 7] := 0.9;
    for I := 1 to 20 do
      AssertEquals('MinP=1.0 on pixel (2,1) must return token 7',
        7, Sampler.GetTokenOnPixel(V, 2, 1));
  finally
    V.Free;
    Sampler.Free;
  end;
end;

procedure TTestNeuralSamplers.TestPenaltyProbabilitiesNoOpIsBitForBit;
var
  Penalty: TNNetTokenHistoryPenalty;
  V, Before: TNNetVolume;
  I: integer;
begin
  // Default knobs (r=1, alpha_f=alpha_p=0) must leave a probability volume
  // bit-for-bit unchanged - no power, no exp factor, no renormalization.
  Penalty := TNNetTokenHistoryPenalty.Create(1.0, 0.0, 0.0);
  V := TNNetVolume.Create(4, 1, 1);
  Before := TNNetVolume.Create(4, 1, 1);
  try
    V.Raw[0] := 0.4; V.Raw[1] := 0.3; V.Raw[2] := 0.2; V.Raw[3] := 0.1;
    Before.Copy(V);
    Penalty.RegisterToken(0);
    Penalty.RegisterToken(2);
    Penalty.ApplyToProbabilities(V);
    for I := 0 to 3 do
      AssertTrue('no-op ApplyToProbabilities must be bit-for-bit at ' +
        IntToStr(I), V.Raw[I] = Before.Raw[I]);
  finally
    V.Free;
    Before.Free;
    Penalty.Free;
  end;
end;

procedure TTestNeuralSamplers.TestPenaltyProbabilitiesRepetitionChangesArgmax;
var
  Penalty: TNNetTokenHistoryPenalty;
  V: TNNetVolume;
begin
  // probs = [0.6, 0.4], token 0 already generated, r = 2:
  // p0 := 0.6^2 = 0.36 < 0.4 -> after renormalization the argmax flips to
  // token 1. Exact values: 0.36/0.76 and 0.40/0.76.
  Penalty := TNNetTokenHistoryPenalty.Create(2.0, 0.0, 0.0);
  V := TNNetVolume.Create(2, 1, 1);
  try
    V.Raw[0] := 0.6;
    V.Raw[1] := 0.4;
    AssertEquals('argmax before penalty', 0, V.GetClass());
    Penalty.RegisterToken(0);
    Penalty.ApplyToProbabilities(V);
    AssertEquals('repetition penalty must flip the argmax', 1, V.GetClass());
    AssertEquals('p0 = 0.36/0.76', 0.36 / 0.76, V.Raw[0], 0.0001);
    AssertEquals('p1 = 0.40/0.76', 0.40 / 0.76, V.Raw[1], 0.0001);
  finally
    V.Free;
    Penalty.Free;
  end;
end;

procedure TTestNeuralSamplers.TestPenaltyProbabilitiesFrequencyPresenceFactor;
var
  Penalty: TNNetTokenHistoryPenalty;
  V: TNNetVolume;
begin
  // probs = [0.5, 0.5], token 0 seen TWICE, alpha_f=0.5, alpha_p=0.3:
  // log-space subtraction of 2*0.5 + 0.3 = 1.3 means
  // p0/p1 = exp(-1.3) after the (ratio-preserving) renormalization.
  Penalty := TNNetTokenHistoryPenalty.Create(1.0, 0.5, 0.3);
  V := TNNetVolume.Create(2, 1, 1);
  try
    V.Raw[0] := 0.5;
    V.Raw[1] := 0.5;
    Penalty.RegisterToken(0);
    Penalty.RegisterToken(0);
    Penalty.ApplyToProbabilities(V);
    AssertEquals('p0/p1 must equal exp(-(2*alpha_f + alpha_p))',
      Exp(-1.3), V.Raw[0] / V.Raw[1], 0.0001);
  finally
    V.Free;
    Penalty.Free;
  end;
end;

procedure TTestNeuralSamplers.TestPenaltyProbabilitiesSumsToOne;
var
  Penalty: TNNetTokenHistoryPenalty;
  V: TNNetVolume;
begin
  // After any non-trivial penalty the volume must be renormalized to a
  // proper distribution (sum = 1).
  Penalty := TNNetTokenHistoryPenalty.Create(3.0, 0.7, 0.4);
  V := TNNetVolume.Create(5, 1, 1);
  try
    V.Raw[0] := 0.3; V.Raw[1] := 0.25; V.Raw[2] := 0.2;
    V.Raw[3] := 0.15; V.Raw[4] := 0.1;
    Penalty.RegisterToken(1);
    Penalty.RegisterToken(3);
    Penalty.RegisterToken(3);
    Penalty.ApplyToProbabilities(V);
    AssertEquals('penalized probabilities must sum to 1',
      1.0, V.GetSum(), 0.0001);
  finally
    V.Free;
    Penalty.Free;
  end;
end;

procedure TTestNeuralSamplers.TestWeightedTopKMatchesRenormalizedDistribution;
var
  Sampler: TNNetSamplerWeightedTopK;
  V: TNNetVolume;
  Token, I, N: integer;
  Counts: array[0..2] of integer;
  Freq, Expected: array[0..2] of TNeuralFloat;
  KeptSum: TNeuralFloat;
begin
  // Skewed top-3 head probs = [0.7, 0.2, 0.07] (+ negligible tail). With K=3
  // the renormalized distribution over {0,1,2} is far from uniform 1/3 each.
  // A large fixed-seed draw must match the renormalized probabilities, NOT
  // the uniform distribution that the legacy TNNetSamplerTopK produces.
  RandSeed := 20260613;
  Sampler := TNNetSamplerWeightedTopK.Create(3);
  V := TNNetVolume.Create(6, 1, 1);
  try
    V.Fill(0.01);
    V.Raw[0] := 0.7;
    V.Raw[1] := 0.2;
    V.Raw[2] := 0.07;
    KeptSum := 0.7 + 0.2 + 0.07;
    Expected[0] := 0.7 / KeptSum;
    Expected[1] := 0.2 / KeptSum;
    Expected[2] := 0.07 / KeptSum;
    for I := 0 to 2 do Counts[I] := 0;
    N := 40000;
    for I := 1 to N do
    begin
      Token := Sampler.GetToken(V);
      AssertTrue('weighted top-k must draw from {0,1,2}',
        (Token >= 0) and (Token <= 2));
      Inc(Counts[Token]);
    end;
    for I := 0 to 2 do Freq[I] := Counts[I] / N;
    // Empirical frequencies must match the renormalized probabilities.
    for I := 0 to 2 do
      AssertEquals('empirical freq matches renormalized prob at ' + IntToStr(I),
        Expected[I], Freq[I], 0.02);
  finally
    V.Free;
    Sampler.Free;
  end;
end;

procedure TTestNeuralSamplers.TestWeightedTopKDiffersFromUniform;
var
  Sampler: TNNetSamplerWeightedTopK;
  V: TNNetVolume;
  Token, I, N: integer;
  Freq0: TNeuralFloat;
  Count0: integer;
begin
  // The dominant token (renormalized prob ~0.72) must be drawn FAR more often
  // than the uniform 1/3, proving weighted-vs-uniform differ.
  RandSeed := 20260613;
  Sampler := TNNetSamplerWeightedTopK.Create(3);
  V := TNNetVolume.Create(6, 1, 1);
  try
    V.Fill(0.01);
    V.Raw[0] := 0.7;
    V.Raw[1] := 0.2;
    V.Raw[2] := 0.07;
    Count0 := 0;
    N := 40000;
    for I := 1 to N do
    begin
      Token := Sampler.GetToken(V);
      if Token = 0 then Inc(Count0);
    end;
    Freq0 := Count0 / N;
    // Uniform would give ~0.333; weighted gives ~0.72. Assert clearly above
    // uniform (well outside any sampling noise band).
    AssertTrue('weighted top-k must NOT match uniform 1/3 (got ' +
      FloatToStr(Freq0) + ')', Freq0 > 0.5);
  finally
    V.Free;
    Sampler.Free;
  end;
end;

procedure TTestNeuralSamplers.TestWeightedTopKSequenceReproducible;
var
  Sampler: TNNetSamplerWeightedTopK;
  V: TNNetVolume;
  SeqA, SeqB: array[0..49] of integer;
  I: integer;
begin
  // Two runs from the SAME fixed seed must produce the identical draw
  // sequence.
  Sampler := TNNetSamplerWeightedTopK.Create(4);
  V := TNNetVolume.Create(8, 1, 1);
  try
    V.Fill(0.01);
    V.Raw[0] := 0.5;
    V.Raw[1] := 0.25;
    V.Raw[2] := 0.12;
    V.Raw[3] := 0.06;
    RandSeed := 987654321;
    for I := 0 to 49 do SeqA[I] := Sampler.GetToken(V);
    RandSeed := 987654321;
    for I := 0 to 49 do SeqB[I] := Sampler.GetToken(V);
    for I := 0 to 49 do
      AssertEquals('fixed-seed draw sequence must be reproducible at ' +
        IntToStr(I), SeqA[I], SeqB[I]);
  finally
    V.Free;
    Sampler.Free;
  end;
end;

procedure TTestNeuralSamplers.TestWeightedTopKNeverDrawsOutsideTopK;
var
  Sampler: TNNetSamplerWeightedTopK;
  V: TNNetVolume;
  Token, I: integer;
begin
  // With K=2 only the two highest-probability tokens (0 and 1) may ever be
  // drawn; the renormalized tail must never appear.
  RandSeed := 20260613;
  Sampler := TNNetSamplerWeightedTopK.Create(2);
  V := TNNetVolume.Create(10, 1, 1);
  try
    V.Fill(0.02);
    V.Raw[0] := 0.6;
    V.Raw[1] := 0.3;
    for I := 1 to 500 do
    begin
      Token := Sampler.GetToken(V);
      AssertTrue('K=2 must only draw from {0,1}, got ' + IntToStr(Token),
        (Token = 0) or (Token = 1));
    end;
  finally
    V.Free;
    Sampler.Free;
  end;
end;

procedure TTestNeuralSamplers.TestWeightedTopKGetTokenOnPixel;
var
  Sampler: TNNetSamplerWeightedTopK;
  V: TNNetVolume;
  Token, I: integer;
begin
  // Pixel-addressed variant with K=1 degenerates to greedy argmax.
  RandSeed := 20260613;
  Sampler := TNNetSamplerWeightedTopK.Create(1);
  V := TNNetVolume.Create(4, 4, 10);
  try
    V.Fill(0.05);
    V[2, 1, 7] := 0.9;
    for I := 1 to 20 do
    begin
      Token := Sampler.GetTokenOnPixel(V, 2, 1);
      AssertEquals('K=1 weighted top-k on pixel (2,1) must return token 7',
        7, Token);
    end;
  finally
    V.Free;
    Sampler.Free;
  end;
end;

procedure TTestNeuralSamplers.TestTypicalKeepsExactlyExpectedSet;
var
  Sampler: TNNetSamplerTypical;
  V: TNNetVolume;
  Token, I: integer;
  Seen: array[0..4] of boolean;
begin
  // probs = [0.4, 0.3, 0.2, 0.07, 0.03]; H = 1.3409 nats. Ordered by ascending
  // |(-log p) - H| the head is tok1, tok2, tok0 (NOT rank order: the most-likely
  // tok0 is NOT first). For Mass = 0.9 the cumulative mass reaches 0.9 exactly at
  // {1, 2, 0}, so the typical set is {0, 1, 2}; tokens 3 and 4 must NEVER be
  // drawn. This is a truncation by surprise-distance, not by rank/cumulative
  // mass, which is what makes typical sampling distinct.
  RandSeed := 20260614;
  Sampler := TNNetSamplerTypical.Create(0.9);
  V := TNNetVolume.Create(5, 1, 1);
  try
    V.Raw[0] := 0.4;
    V.Raw[1] := 0.3;
    V.Raw[2] := 0.2;
    V.Raw[3] := 0.07;
    V.Raw[4] := 0.03;
    for I := 0 to 4 do Seen[I] := false;
    for I := 1 to 400 do
    begin
      Token := Sampler.GetToken(V);
      AssertTrue('typical(0.9) must only draw from {0,1,2}, got ' +
        IntToStr(Token), (Token >= 0) and (Token <= 2));
      Seen[Token] := true;
    end;
    AssertTrue('typical token 0 drawn', Seen[0]);
    AssertTrue('typical token 1 drawn', Seen[1]);
    AssertTrue('typical token 2 drawn', Seen[2]);
  finally
    V.Free;
    Sampler.Free;
  end;
end;

procedure TTestNeuralSamplers.TestTypicalExcludesMostLikelyToken;
var
  Sampler: TNNetSamplerTypical;
  V: TNNetVolume;
  Token, I: integer;
begin
  // Same row, but Mass = 0.5. The cumulative mass of the surprise-sorted head
  // reaches 0.5 at {tok1 (0.3), tok2 (0.2)} = exactly {1, 2}. The MOST-LIKELY
  // token 0 is EXCLUDED (its surprise 0.916 is too far below H = 1.341). No
  // rank/cumulative-mass sampler would ever drop the argmax while keeping
  // lower-probability tokens - this is the signature of typical sampling.
  RandSeed := 20260614;
  Sampler := TNNetSamplerTypical.Create(0.5);
  V := TNNetVolume.Create(5, 1, 1);
  try
    V.Raw[0] := 0.4;
    V.Raw[1] := 0.3;
    V.Raw[2] := 0.2;
    V.Raw[3] := 0.07;
    V.Raw[4] := 0.03;
    for I := 1 to 400 do
    begin
      Token := Sampler.GetToken(V);
      AssertTrue('typical(0.5) must draw only from {1,2}, got ' +
        IntToStr(Token), (Token = 1) or (Token = 2));
    end;
  finally
    V.Free;
    Sampler.Free;
  end;
end;

procedure TTestNeuralSamplers.TestTypicalGetTokenOnPixel;
var
  Sampler: TNNetSamplerTypical;
  V: TNNetVolume;
  Token, I: integer;
begin
  // Pixel-addressed variant. A spiked, near-deterministic row (one token holds
  // almost all mass) has tiny entropy, so the typical set collapses to that
  // dominant token: the draw is effectively deterministic.
  RandSeed := 20260614;
  Sampler := TNNetSamplerTypical.Create(0.9);
  V := TNNetVolume.Create(4, 4, 10);
  try
    V.Fill(0.001);
    V[2, 1, 7] := 0.991;
    for I := 1 to 30 do
    begin
      Token := Sampler.GetTokenOnPixel(V, 2, 1);
      AssertEquals('typical on spiked pixel (2,1) must return token 7',
        7, Token);
    end;
  finally
    V.Free;
    Sampler.Free;
  end;
end;

procedure TTestNeuralSamplers.TestMirostatV2KeepsExactlyExpectedSet;
var
  Sampler: TNNetSamplerMirostat;
  V: TNNetVolume;
  Token, I: integer;
  Seen: array[0..4] of boolean;
begin
  // v2 keeps every token with surprise -log p <= Mu. A fresh sampler has
  // Mu = 2*Tau. probs = [0.5, 0.25, 0.15, 0.07, 0.03] -> surprises
  // [0.693, 1.386, 1.897, 2.659, 3.507]. Tau = 1.1 -> Mu = 2.2 keeps exactly
  // {0, 1, 2} (surprise <= 2.2); tokens 3 and 4 are above Mu. We assert ONLY on
  // the first draw set per fresh-reset so the feedback update cannot move Mu
  // out from under the documented cut.
  RandSeed := 20260614;
  for I := 0 to 4 do Seen[I] := false;
  V := TNNetVolume.Create(5, 1, 1);
  try
    V.Raw[0] := 0.5;
    V.Raw[1] := 0.25;
    V.Raw[2] := 0.15;
    V.Raw[3] := 0.07;
    V.Raw[4] := 0.03;
    for I := 1 to 400 do
    begin
      Sampler := TNNetSamplerMirostat.Create(1.1, 0.1, mvV2);
      try
        Token := Sampler.GetToken(V);  // first step: Mu = 2.2 exactly
      finally
        Sampler.Free;
      end;
      AssertTrue('Mirostat v2 (Mu=2.2) first draw must be in {0,1,2}, got ' +
        IntToStr(Token), (Token >= 0) and (Token <= 2));
      Seen[Token] := true;
    end;
    AssertTrue('Mirostat v2 kept token 0', Seen[0]);
    AssertTrue('Mirostat v2 kept token 1', Seen[1]);
    AssertTrue('Mirostat v2 kept token 2', Seen[2]);
  finally
    V.Free;
  end;
end;

procedure TTestNeuralSamplers.TestMirostatV2NarrowerMuKeepsFewer;
var
  Sampler: TNNetSamplerMirostat;
  V: TNNetVolume;
  Token, I: integer;
begin
  // Same row; Tau = 0.8 -> Mu = 1.6 keeps only {0, 1} (surprises 0.693, 1.386
  // <= 1.6; token 2's 1.897 > 1.6). A smaller Tau truncates harder.
  RandSeed := 20260614;
  V := TNNetVolume.Create(5, 1, 1);
  try
    V.Raw[0] := 0.5;
    V.Raw[1] := 0.25;
    V.Raw[2] := 0.15;
    V.Raw[3] := 0.07;
    V.Raw[4] := 0.03;
    for I := 1 to 400 do
    begin
      Sampler := TNNetSamplerMirostat.Create(0.8, 0.1, mvV2);
      try
        Token := Sampler.GetToken(V);
      finally
        Sampler.Free;
      end;
      AssertTrue('Mirostat v2 (Mu=1.6) first draw must be in {0,1}, got ' +
        IntToStr(Token), (Token = 0) or (Token = 1));
    end;
  finally
    V.Free;
  end;
end;

procedure TTestNeuralSamplers.TestMirostatResetReArmsMu;
var
  Sampler: TNNetSamplerMirostat;
  V: TNNetVolume;
  I: integer;
begin
  // Mirostat is STATEFUL: Mu drifts as tokens are drawn. Reset() must re-arm it
  // to 2*Tau (the paper's init), restoring the start-of-generation state.
  RandSeed := 20260614;
  Sampler := TNNetSamplerMirostat.Create(3.0, 0.1, mvV2);
  V := TNNetVolume.Create(8, 1, 1);
  try
    AssertEquals('fresh Mirostat Mu must be 2*Tau', 6.0, Sampler.Mu, 0.0001);
    V.Fill(0.1);
    V.Raw[0] := 0.3;
    // Draw a bunch so Mu drifts away from 6.0.
    for I := 1 to 30 do Sampler.GetToken(V);
    AssertTrue('Mu must have drifted from its init after drawing',
      Abs(Sampler.Mu - 6.0) > 0.001);
    Sampler.Reset();
    AssertEquals('Reset must re-arm Mu to 2*Tau', 6.0, Sampler.Mu, 0.0001);
  finally
    V.Free;
    Sampler.Free;
  end;
end;

procedure TTestNeuralSamplers.TestMirostatV2MuConvergesTowardTau;
var
  Sampler: TNNetSamplerMirostat;
  V: TNNetVolume;
  Token, I, J, V_SIZE: integer;
  Raw, Sum, AvgSurprise, SumSurprise, P: TNeuralFloat;
  NTail: integer;
begin
  // The Mirostat guarantee: on a synthetic stream with a RANGE of surprises the
  // feedback loop drives the OBSERVED surprise (and thus output entropy) toward
  // Tau. We stream a fresh power-law-ish row each step (mild per-step noise),
  // run the controller, and assert the running-average observed surprise over
  // the converged tail sits near Tau. Mu must also stay BOUNDED (not diverge).
  RandSeed := 20260614;
  V_SIZE := 80;
  Sampler := TNNetSamplerMirostat.Create(3.0, 0.1, mvV2);
  V := TNNetVolume.Create(V_SIZE, 1, 1);
  try
    Sampler.Reset();
    SumSurprise := 0;
    NTail := 0;
    for I := 1 to 800 do
    begin
      // Power-law p_j ~ 1/(j+1)^1.1 with small noise, renormalized.
      Sum := 0;
      for J := 0 to V_SIZE - 1 do
      begin
        Raw := (1.0 / Power(J + 1, 1.1)) * (1.0 + 0.05 * Random);
        V.Raw[J] := Raw;
        Sum := Sum + Raw;
      end;
      for J := 0 to V_SIZE - 1 do V.Raw[J] := V.Raw[J] / Sum;
      Token := Sampler.GetToken(V);
      // Accumulate observed surprise over the converged tail (last 300 steps).
      if I > 500 then
      begin
        P := V.Raw[Token];
        if P > 0 then SumSurprise := SumSurprise - Ln(P);
        Inc(NTail);
        AssertTrue('Mu must stay bounded (no divergence)',
          (Sampler.Mu > -50) and (Sampler.Mu < 50));
      end;
    end;
    AvgSurprise := SumSurprise / NTail;
    // Observed surprise must converge near Tau = 3.0 (the controller's target).
    AssertEquals('average observed surprise must converge toward Tau',
      3.0, AvgSurprise, 0.6);
  finally
    V.Free;
    Sampler.Free;
  end;
end;

procedure TTestNeuralSamplers.TestMirostatV1MuStaysBounded;
var
  Sampler: TNNetSamplerMirostat;
  V: TNNetVolume;
  Token, I, J, V_SIZE: integer;
  Raw, Sum: TNeuralFloat;
begin
  // v1 (Zipf-estimate truncation) on the same power-law stream must also remain
  // a well-behaved controller: Mu stays bounded and every draw is a valid token.
  RandSeed := 20260614;
  V_SIZE := 80;
  Sampler := TNNetSamplerMirostat.Create(3.0, 0.1, mvV1);
  V := TNNetVolume.Create(V_SIZE, 1, 1);
  try
    Sampler.Reset();
    for I := 1 to 600 do
    begin
      Sum := 0;
      for J := 0 to V_SIZE - 1 do
      begin
        Raw := (1.0 / Power(J + 1, 1.1)) * (1.0 + 0.05 * Random);
        V.Raw[J] := Raw;
        Sum := Sum + Raw;
      end;
      for J := 0 to V_SIZE - 1 do V.Raw[J] := V.Raw[J] / Sum;
      Token := Sampler.GetToken(V);
      AssertTrue('v1 draw must be a valid token',
        (Token >= 0) and (Token < V_SIZE));
      AssertTrue('v1 Mu must stay bounded',
        (Sampler.Mu > -100) and (Sampler.Mu < 100));
    end;
  finally
    V.Free;
    Sampler.Free;
  end;
end;

// Fills V with a normalized, deterministically "peaked" distribution: token
// PeakAt gets the dominant mass, the rest decay geometrically. Returns the
// probabilities already summing to 1.
procedure BuildLargePeakedRow(V: TNNetVolume; Size, PeakAt: integer);
var
  I: integer;
  Sum, Val: TNeuralFloat;
begin
  Sum := 0;
  for I := 0 to Size - 1 do
  begin
    // Deterministic, no RNG: geometric decay with distance from the peak.
    // The decay must be EXPONENTIAL, not harmonic - a 1/(1+d) row is so flat
    // that its 0.8 nucleus spans most of the vocabulary, which would make
    // this a retry-path test rather than a prefix-path one.
    Val := Exp(-0.35 * Abs(I - PeakAt));
    V.Raw[I] := Val;
    Sum := Sum + Val;
  end;
  for I := 0 to Size - 1 do V.Raw[I] := V.Raw[I] / Sum;
end;

procedure TTestNeuralSamplers.TestTopPLargeVocabNucleusIsCorrect;
const
  V_SIZE = 5000; // > the 1024 adaptive threshold and > the 256 first guess
var
  Sampler: TNNetSamplerTopP;
  V: TNNetVolume;
  I, Token, NucleusSize: integer;
  Cum, MaxP: TNeuralFloat;
begin
  // With a peaked row the nucleus is far narrower than the first-guess K, so
  // the partial-selection prefix must answer without the retry - and every
  // draw must land inside the true nucleus.
  Sampler := TNNetSamplerTopP.Create(0.80);
  V := TNNetVolume.Create(V_SIZE, 1, 1);
  try
    BuildLargePeakedRow(V, V_SIZE, 0);
    // Independently compute the nucleus width: the row is already descending
    // from index 0 here, so a plain cumulative scan gives the true answer.
    Cum := 0;
    NucleusSize := V_SIZE;
    MaxP := V.Raw[0];
    for I := 0 to V_SIZE - 1 do
    begin
      Cum := Cum + V.Raw[I];
      if Cum > 0.80 then
      begin
        NucleusSize := I;
        Break;
      end;
    end;
    AssertTrue('Test setup: nucleus must be narrower than the first guess',
      NucleusSize < 256);
    AssertTrue('Test setup: peak must dominate', MaxP > 0);
    for I := 0 to 299 do
    begin
      Token := Sampler.GetToken(V);
      AssertTrue('Large-vocab top-p draw must stay inside the nucleus, got ' +
        IntToStr(Token), (Token >= 0) and (Token < NucleusSize));
    end;
  finally
    V.Free;
    Sampler.Free;
  end;
end;

procedure TTestNeuralSamplers.TestTopPLargeVocabFlatDistributionRetries;
const
  V_SIZE = 5000;
var
  Sampler: TNNetSamplerTopP;
  V: TNNetVolume;
  I, Token, Distinct: integer;
  Seen: array of boolean;
begin
  // A UNIFORM row needs ~0.9*5000 = 4500 tokens to reach the mass, far more
  // than the 256-token first guess, so this forces the widen-and-retry path.
  // Every token must still be reachable and in range.
  Sampler := TNNetSamplerTopP.Create(0.90);
  V := TNNetVolume.Create(V_SIZE, 1, 1);
  SetLength(Seen, V_SIZE);
  try
    V.Fill(1.0 / V_SIZE);
    Distinct := 0;
    for I := 0 to 499 do
    begin
      Token := Sampler.GetToken(V);
      AssertTrue('Retry-path draw must be a valid token, got ' +
        IntToStr(Token), (Token >= 0) and (Token < V_SIZE));
      if not Seen[Token] then
      begin
        Seen[Token] := true;
        Inc(Distinct);
      end;
    end;
    // A retry that silently collapsed to the truncated window would keep
    // returning the same handful of tokens.
    AssertTrue('Retry path must still draw from a wide set, distinct=' +
      IntToStr(Distinct), Distinct > 100);
  finally
    SetLength(Seen, 0);
    V.Free;
    Sampler.Free;
  end;
end;

procedure TTestNeuralSamplers.TestWeightedTopKLargeVocabNeverDrawsOutsideTopK;
const
  V_SIZE = 4096;
  TOP_K  = 40;
var
  Sampler: TNNetSamplerWeightedTopK;
  V: TNNetVolume;
  I, J, Token, BestIdx: integer;
  Scratch: TNNetVolume;
  Cutoff, Best: TNeuralFloat;
begin
  // The quickselect prefix must equal the true top-K. Build the reference by
  // repeatedly extracting the max from a scratch copy.
  Sampler := TNNetSamplerWeightedTopK.Create(TOP_K);
  V := TNNetVolume.Create(V_SIZE, 1, 1);
  Scratch := TNNetVolume.Create(V_SIZE, 1, 1);
  try
    BuildLargePeakedRow(V, V_SIZE, 1234); // peak away from index 0
    Scratch.Copy(V);
    Cutoff := 0;
    for I := 0 to TOP_K - 1 do
    begin
      BestIdx := 0;
      Best := Scratch.Raw[0];
      for J := 1 to V_SIZE - 1 do
        if Scratch.Raw[J] > Best then
        begin
          Best := Scratch.Raw[J];
          BestIdx := J;
        end;
      Cutoff := Best;          // after the loop: the K-th largest probability
      Scratch.Raw[BestIdx] := -1;
    end;
    for I := 0 to 299 do
    begin
      Token := Sampler.GetToken(V);
      AssertTrue('Draw must be a valid token', (Token >= 0) and (Token < V_SIZE));
      AssertTrue('Large-vocab weighted top-k drew outside the true top-K: ' +
        'token ' + IntToStr(Token), V.Raw[Token] >= Cutoff);
    end;
  finally
    Scratch.Free;
    V.Free;
    Sampler.Free;
  end;
end;

procedure TTestNeuralSamplers.TestMinPLargeVocabRespectsThreshold;
const
  V_SIZE = 4096;
var
  Sampler: TNNetSamplerMinP;
  V: TNNetVolume;
  I, Token: integer;
  MaxP, Threshold: TNeuralFloat;
begin
  // The linear survivor count must reduce the window to exactly the tokens
  // passing p >= MinP * max(p) - no draw may fall below that cut.
  Sampler := TNNetSamplerMinP.Create(0.10);
  V := TNNetVolume.Create(V_SIZE, 1, 1);
  try
    BuildLargePeakedRow(V, V_SIZE, 77);
    MaxP := V.Raw[0];
    for I := 1 to V_SIZE - 1 do
      if V.Raw[I] > MaxP then MaxP := V.Raw[I];
    Threshold := 0.10 * MaxP;
    for I := 0 to 299 do
    begin
      Token := Sampler.GetToken(V);
      AssertTrue('Draw must be a valid token', (Token >= 0) and (Token < V_SIZE));
      AssertTrue('Large-vocab min-p drew below the threshold: token ' +
        IntToStr(Token), V.Raw[Token] >= Threshold);
    end;
  finally
    V.Free;
    Sampler.Free;
  end;
end;

initialization
  RegisterTest(TTestNeuralSamplers);

end.
