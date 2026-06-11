unit TestNeuralNLPMetrics;
(*
Tests for neuralnlpmetrics.pas: held-out perplexity, corpus BLEU and ROUGE.
All BLEU/ROUGE expectations are hand-computed on tiny classic examples; the
perplexity tests use fixed-weight models whose perplexity is analytic
(uniform over V -> PPL = V; near-deterministic -> PPL ~ 1).
Coded by Claude (AI).
*)

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, Math, fpcunit, testregistry,
  neuralnetwork, neuralvolume, neuralnlpmetrics;

type
  TTestNeuralNLPMetrics = class(TTestCase)
  private
    // Per-position LM: Input(Context,1,1) token ids -> PointwiseConvLinear(V)
    // -> PointwiseSoftMax. With zero weights the logits are pure biases, so
    // the next-token distribution is input-independent and analytic.
    function BuildPerPositionLM(ContextLen, Vocab: integer): TNNet;
    // Single next-char LM (TinyGPT family): Input(Context,1,Vocab) one-hot
    // reversed -> FullConnectLinear(Vocab) -> SoftMax.
    function BuildCharLM(ContextLen, Vocab: integer): TNNet;
    // 12-word sorted vocabulary; ids 0..11 by sort order
    // ('<eos>'=0, '<pad>'=1, 'apple'=2, ..., 'jam'=11).
    function BuildDict(): TStringListInt;
    // Zero every weight, set every bias to BiasValue, then re-pack the conv
    // weight caches (PointwiseConv keeps packed copies of its weights).
    procedure SetUniformBiases(NN: TNNet; BiasValue: TNeuralFloat);
  published
    // Identical candidate/reference -> BLEU exactly 1.0, smoothed or not.
    procedure TestBLEUIdenticalIsOne;
    // The classic "the the the the the the the" candidate: modified
    // (clipped) unigram precision is 2/7, and higher unsmoothed orders are 0.
    procedure TestBLEUClippedRepeatedUnigrams;
    // A 2-token candidate against a 6-token reference: every measurable
    // precision is 1, so BLEU reduces to the brevity penalty exp(1-6/2).
    procedure TestBLEUBrevityPenalty;
    // The token-id API and the whitespace string overload agree exactly.
    procedure TestBLEUTokenAndStringAPIsAgree;
    // ROUGE-1 P/R/F1 on the classic "found under the bed" pair.
    procedure TestRouge1HandComputed;
    // ROUGE-2 P/R/F1 on the same pair.
    procedure TestRouge2HandComputed;
    // ROUGE-L (LCS) on the same pair + the dual API agreement.
    procedure TestRougeLHandComputed;
    // ROUGE-1 is order-blind (score 1) while ROUGE-L sees the reordering.
    procedure TestRougeLOrderSensitivity;
    // Corpus-level ROUGE is the macro average of the per-pair scores.
    procedure TestCorpusRougeAverages;
    // Uniform next-token distribution over V -> PPL = V, MeanNLL = ln V,
    // BitsPerToken = log2 V.
    procedure TestPerplexityUniformModelEqualsVocab;
    // A near-one-hot model (one large bias) -> PPL ~ 1.
    procedure TestPerplexityDeterministicModelNearOne;
    // Special targets (< 2) are excluded from the average but counted.
    procedure TestPerplexitySkipsSpecialTokens;
    // Char-level path: uniform over 128 -> PPL = 128 and the EOS char chr(1)
    // is skipped as special.
    procedure TestPerplexityFromCharsUniform;
  end;

implementation

const
  csCtx = 8;
  csVocab = 12;
  csCharVocab = 128;

function TTestNeuralNLPMetrics.BuildPerPositionLM(ContextLen,
  Vocab: integer): TNNet;
begin
  Result := TNNet.Create();
  Result.AddLayer(TNNetInput.Create(ContextLen, 1, 1));
  Result.AddLayer(TNNetPointwiseConvLinear.Create(Vocab));
  Result.AddLayer(TNNetPointwiseSoftMax.Create());
end;

function TTestNeuralNLPMetrics.BuildCharLM(ContextLen, Vocab: integer): TNNet;
begin
  Result := TNNet.Create();
  Result.AddLayer(TNNetInput.Create(ContextLen, 1, Vocab));
  Result.AddLayer(TNNetFullConnectLinear.Create(Vocab));
  Result.AddLayer(TNNetSoftMax.Create());
end;

function TTestNeuralNLPMetrics.BuildDict(): TStringListInt;
const
  Words: array[0..11] of string = ('<eos>', '<pad>', 'apple', 'blue', 'cat',
    'dog', 'egg', 'fox', 'gold', 'hat', 'ice', 'jam');
var
  W: integer;
begin
  Result := TStringListInt.Create();
  // Byte-exact compare: the default case-insensitive AnsiCompareText is
  // locale/widestring-manager dependent (LazUtils swaps in UTF-8 collation in
  // the test binary), which would reorder '<eos>'/'<pad>' vs the words.
  Result.CaseSensitive := true;
  Result.Sorted := true;
  for W := 0 to High(Words) do Result.Add(Words[W]);
  Result.SaveCurrentPosition(); // token id = sorted index ('<eos>'=0, '<pad>'=1)
end;

procedure TTestNeuralNLPMetrics.SetUniformBiases(NN: TNNet;
  BiasValue: TNeuralFloat);
var
  L, N: integer;
begin
  for L := 0 to NN.CountLayers() - 1 do
    for N := 0 to NN.Layers[L].Neurons.Count - 1 do
    begin
      NN.Layers[L].Neurons[N].Weights.Fill(0);
      NN.Layers[L].Neurons[N].BiasWeight := BiasValue;
    end;
  // Re-pack the conv weight caches (deltas are zero, so this only re-packs).
  NN.ClearDeltas();
  NN.UpdateWeights();
end;

procedure TTestNeuralNLPMetrics.TestBLEUIdenticalIsOne;
const
  S = 'the cat sat on the mat';
begin
  AssertEquals('identical pair, smoothed', 1.0,
    CorpusBLEU([S], [S], 4, true), 1e-9);
  AssertEquals('identical pair, unsmoothed', 1.0,
    CorpusBLEU([S], [S], 4, false), 1e-9);
end;

procedure TTestNeuralNLPMetrics.TestBLEUClippedRepeatedUnigrams;
const
  Cand = 'the the the the the the the';
  Ref = 'the cat is on the mat';
var
  Expected: TNeuralFloat;
begin
  // Modified unigram precision: 'the' appears 7 times in the candidate but
  // only 2 times in the reference -> clipped to 2/7 (Papineni's example).
  // Candidate (7) is longer than the reference (6) -> BP = 1.
  AssertEquals('clipped unigram precision', 2.0 / 7.0,
    CorpusBLEU([Cand], [Ref], 1, false), 1e-9);
  // No bigram (or higher) matches -> unsmoothed BLEU-4 collapses to 0.
  AssertEquals('unsmoothed BLEU-4 is zero', 0.0,
    CorpusBLEU([Cand], [Ref], 4, false), 1e-9);
  // Lin & Och smoothing-1: p1 = 2/7 (unigrams never smoothed),
  // p2 = (0+1)/(6+1), p3 = (0+1)/(5+1), p4 = (0+1)/(4+1).
  Expected := Power((2.0 / 7.0) * (1.0 / 7.0) * (1.0 / 6.0) * (1.0 / 5.0), 0.25);
  AssertEquals('smoothed BLEU-4', Expected,
    CorpusBLEU([Cand], [Ref], 4, true), 1e-9);
end;

procedure TTestNeuralNLPMetrics.TestBLEUBrevityPenalty;
const
  Cand = 'the cat';
  Ref = 'the cat sat on the mat';
begin
  // p1 = 2/2 = 1; p2 = 1/1 -> smoothed (1+1)/(1+1) = 1; the candidate has no
  // 3-/4-grams so those orders are excluded. BLEU = BP = exp(1 - 6/2).
  AssertEquals('BLEU equals the brevity penalty', Exp(1.0 - 6.0 / 2.0),
    CorpusBLEU([Cand], [Ref], 4, true), 1e-7);
  AssertTrue('short candidate is penalised below 0.14',
    CorpusBLEU([Cand], [Ref], 4, true) < 0.14);
end;

procedure TTestNeuralNLPMetrics.TestBLEUTokenAndStringAPIsAgree;
var
  CandIds, RefIds: array of TNeuralIntegerArray;
begin
  // 'a b a c' vs 'a b c c' with shared first-seen ids a=0, b=1, c=2.
  SetLength(CandIds, 1);
  SetLength(RefIds, 1);
  CandIds[0] := TNeuralIntegerArray.Create(0, 1, 0, 2);
  RefIds[0] := TNeuralIntegerArray.Create(0, 1, 2, 2);
  AssertEquals('token-id and string APIs agree (smoothed)',
    CorpusBLEU(CandIds, RefIds, 4, true),
    CorpusBLEU(['a b a c'], ['a b c c'], 4, true), 1e-9);
  AssertEquals('token-id and string APIs agree (unsmoothed BLEU-2)',
    CorpusBLEU(CandIds, RefIds, 2, false),
    CorpusBLEU(['a b a c'], ['a b c c'], 2, false), 1e-9);
end;

const
  csRougeCand = 'the cat was found under the bed';
  csRougeRef = 'the cat was under the bed';

procedure TTestNeuralNLPMetrics.TestRouge1HandComputed;
var
  Score: TNNetRougeScore;
begin
  // Clipped unigram overlap = 6 (the:2, cat, was, under, bed); candidate has
  // 7 tokens, reference 6 -> P = 6/7, R = 6/6 = 1, F1 = 12/13.
  Score := RougeN(csRougeCand, csRougeRef, 1);
  AssertEquals('ROUGE-1 precision', 6.0 / 7.0, Score.Precision, 1e-9);
  AssertEquals('ROUGE-1 recall', 1.0, Score.Recall, 1e-9);
  AssertEquals('ROUGE-1 F1', 12.0 / 13.0, Score.F1, 1e-9);
end;

procedure TTestNeuralNLPMetrics.TestRouge2HandComputed;
var
  Score: TNNetRougeScore;
begin
  // Matching bigrams: 'the cat', 'cat was', 'under the', 'the bed' = 4.
  // Candidate has 6 bigrams, reference 5 -> P = 4/6, R = 4/5, F1 = 8/11.
  Score := RougeN(csRougeCand, csRougeRef, 2);
  AssertEquals('ROUGE-2 precision', 4.0 / 6.0, Score.Precision, 1e-9);
  AssertEquals('ROUGE-2 recall', 4.0 / 5.0, Score.Recall, 1e-9);
  AssertEquals('ROUGE-2 F1', 8.0 / 11.0, Score.F1, 1e-9);
end;

procedure TTestNeuralNLPMetrics.TestRougeLHandComputed;
var
  Score, TokScore: TNNetRougeScore;
  CandIds, RefIds: TNeuralIntegerArray;
begin
  // The reference is a subsequence of the candidate -> LCS = 6.
  Score := RougeL(csRougeCand, csRougeRef);
  AssertEquals('ROUGE-L precision', 6.0 / 7.0, Score.Precision, 1e-9);
  AssertEquals('ROUGE-L recall', 1.0, Score.Recall, 1e-9);
  AssertEquals('ROUGE-L F1', 12.0 / 13.0, Score.F1, 1e-9);
  // Dual API agreement (first-seen ids: the=0 cat=1 was=2 found=3 under=4
  // bed=5).
  CandIds := TNeuralIntegerArray.Create(0, 1, 2, 3, 4, 0, 5);
  RefIds := TNeuralIntegerArray.Create(0, 1, 2, 4, 0, 5);
  TokScore := RougeL(CandIds, RefIds);
  AssertEquals('token/string ROUGE-L F1 agree', Score.F1, TokScore.F1, 1e-9);
end;

procedure TTestNeuralNLPMetrics.TestRougeLOrderSensitivity;
var
  Bag, Lcs: TNNetRougeScore;
begin
  // Same bag of words, different order: ROUGE-1 is blind (1.0) while
  // ROUGE-L only finds a 2-token common subsequence -> 2/3.
  Bag := RougeN('the fox brown', 'the brown fox', 1);
  Lcs := RougeL('the fox brown', 'the brown fox');
  AssertEquals('ROUGE-1 is order-blind', 1.0, Bag.F1, 1e-9);
  AssertEquals('ROUGE-L sees the reorder', 2.0 / 3.0, Lcs.F1, 1e-9);
end;

procedure TTestNeuralNLPMetrics.TestCorpusRougeAverages;
var
  Pair1, Pair2, Avg: TNNetRougeScore;
begin
  Pair1 := RougeN(csRougeCand, csRougeRef, 1);
  Pair2 := RougeN('a b c', 'a b c', 1); // identical -> all 1.0
  Avg := CorpusRougeN([csRougeCand, 'a b c'], [csRougeRef, 'a b c'], 1);
  AssertEquals('corpus ROUGE-1 F1 is the macro average',
    (Pair1.F1 + Pair2.F1) / 2.0, Avg.F1, 1e-9);
  Pair1 := RougeL(csRougeCand, csRougeRef);
  Avg := CorpusRougeL([csRougeCand, 'a b c'], [csRougeRef, 'a b c']);
  AssertEquals('corpus ROUGE-L F1 is the macro average',
    (Pair1.F1 + 1.0) / 2.0, Avg.F1, 1e-9);
end;

procedure TTestNeuralNLPMetrics.TestPerplexityUniformModelEqualsVocab;
var
  NN: TNNet;
  Dict: TStringListInt;
  Corpus: TStringList;
  Stats: TNNetPerplexityStats;
begin
  NN := BuildPerPositionLM(csCtx, csVocab);
  Dict := BuildDict();
  Corpus := TStringList.Create();
  try
    // All biases 1 (NOT 0: an all-zero pre-softmax row is the documented
    // TVolume.SoftMax gotcha) -> equal logits -> uniform p = 1/V everywhere.
    SetUniformBiases(NN, 1.0);
    Corpus.Add('cat dog egg fox');
    Stats := Perplexity(NN, Dict, Corpus);
    AssertEquals('3 predicted positions', 3, Stats.PredictedTokens);
    AssertEquals('no skipped tokens', 0, Stats.SkippedTokens);
    AssertEquals('uniform PPL = V', csVocab, Stats.Perplexity, 1e-3);
    AssertEquals('MeanNLL = ln V', Ln(csVocab), Stats.MeanNLL, 1e-4);
    AssertEquals('BitsPerToken = log2 V', Ln(csVocab) / Ln(2.0),
      Stats.BitsPerToken, 1e-4);
  finally
    Corpus.Free;
    Dict.Free;
    NN.Free;
  end;
end;

procedure TTestNeuralNLPMetrics.TestPerplexityDeterministicModelNearOne;
var
  NN: TNNet;
  Dict: TStringListInt;
  Corpus: TStringList;
  Stats: TNNetPerplexityStats;
  Head: TNNetLayer;
  CatId: integer;
begin
  NN := BuildPerPositionLM(csCtx, csVocab);
  Dict := BuildDict();
  Corpus := TStringList.Create();
  try
    SetUniformBiases(NN, 0.0);
    // One dominant bias on the 'cat' neuron: p(cat) ~ 1 - 11*e^-20. The id
    // is looked up (NOT hardcoded): the sorted position of '<eos>'/'<pad>'
    // depends on the active widestring manager's collation.
    Head := NN.Layers[1];
    CatId := Dict.WordToInteger('cat');
    AssertTrue('cat id is a regular (non-special) token', CatId >= 2);
    Head.Neurons[CatId].BiasWeight := 20.0;
    NN.ClearDeltas();
    NN.UpdateWeights();
    Corpus.Add('cat cat cat cat cat');
    Stats := Perplexity(NN, Dict, Corpus);
    AssertEquals('4 predicted positions', 4, Stats.PredictedTokens);
    AssertTrue('deterministic PPL >= 1', Stats.Perplexity >= 1.0);
    AssertEquals('deterministic PPL ~ 1', 1.0, Stats.Perplexity, 1e-3);
  finally
    Corpus.Free;
    Dict.Free;
    NN.Free;
  end;
end;

procedure TTestNeuralNLPMetrics.TestPerplexitySkipsSpecialTokens;
var
  NN: TNNet;
  Dict: TStringListInt;
  Corpus: TStringList;
  Stats: TNNetPerplexityStats;
begin
  NN := BuildPerPositionLM(csCtx, csVocab);
  Dict := BuildDict();
  Corpus := TStringList.Create();
  try
    SetUniformBiases(NN, 1.0);
    // The last word is WHATEVER word carries token id 0 (collation-proof):
    // a target id < 2 is special and must be excluded but counted.
    Corpus.Add('cat dog ' + Dict.IntegerToWord(0));
    Stats := Perplexity(NN, Dict, Corpus);
    AssertEquals('one scored position', 1, Stats.PredictedTokens);
    AssertEquals('one skipped special target', 1, Stats.SkippedTokens);
    AssertEquals('uniform PPL = V over scored positions', csVocab,
      Stats.Perplexity, 1e-3);
  finally
    Corpus.Free;
    Dict.Free;
    NN.Free;
  end;
end;

procedure TTestNeuralNLPMetrics.TestPerplexityFromCharsUniform;
var
  NN: TNNet;
  Corpus: TStringList;
  Stats: TNNetPerplexityStats;
begin
  NN := BuildCharLM(6, csCharVocab);
  Corpus := TStringList.Create();
  try
    // A perfectly FLAT logit row would hit the documented TVolume.SoftMax
    // gotcha (after max-subtraction an all-zero row is left unchanged), so
    // make the row non-flat with one analytically-negligible outlier:
    // biases 0 everywhere except -20 on the never-targeted neuron 127.
    // Then p(target) = 1/(127 + e^-20) ~ 1/127 for every real char.
    SetUniformBiases(NN, 0.0);
    NN.Layers[1].Neurons[csCharVocab - 1].BiasWeight := -20.0;
    NN.ClearDeltas();
    NN.UpdateWeights();
    Corpus.Add('abcd' + chr(1)); // TinyGPT-style EOS terminator
    Stats := PerplexityFromChars(NN, Corpus);
    // Predicted: 'b', 'c', 'd' (positions 2..4); chr(1) is special-skipped.
    AssertEquals('3 predicted chars', 3, Stats.PredictedTokens);
    AssertEquals('EOS char skipped', 1, Stats.SkippedTokens);
    AssertEquals('near-uniform char PPL = 127', csCharVocab - 1,
      Stats.Perplexity, 1e-2);
    AssertEquals('BPC = log2(127)', Ln(csCharVocab - 1) / Ln(2.0),
      Stats.BitsPerToken, 1e-4);
  finally
    Corpus.Free;
    NN.Free;
  end;
end;

initialization
  RegisterTest(TTestNeuralNLPMetrics);
end.
