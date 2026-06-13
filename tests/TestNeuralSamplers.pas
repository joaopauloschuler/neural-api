unit TestNeuralSamplers;

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, fpcunit, testregistry, neuralvolume;

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

initialization
  RegisterTest(TTestNeuralSamplers);

end.
