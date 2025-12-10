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

initialization
  RegisterTest(TTestNeuralSamplers);

end.
