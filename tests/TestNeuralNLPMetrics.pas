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
    // Input-DEPENDENT per-position scorer: neuron J gets weight 0.02*J and
    // bias 0.03*J, so the logit row at position P is 0.02*J*TokenId(P)+0.03*J
    // - the distribution genuinely depends on the PREVIOUS token, pinning the
    // causal shift (row P-1 scores token P).
    procedure SetLinearScorerWeights(NN: TNNet);
    // Analytic softmax log-prob of target Tgt in the row conditioned on
    // PrevId under SetLinearScorerWeights.
    function ExpectedRowLogProb(PrevId, Tgt: integer): TNeuralFloat;
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
    // ScoreSequence matches hand-computed softmax math (input-dependent
    // logits pin the causal shift); Result[0] = 0; context overflow raises.
    procedure TestScoreSequenceHandComputedSoftmax;
    // Summing ScoreSequence logprobs over a corpus reproduces Perplexity's
    // MeanNLL / exp(-mean) on the same (truncation-free) windows.
    procedure TestScoreSequenceMatchesPerplexity;
    // ScoreCompletion scores ONLY completion tokens: a non-boundary context
    // edit leaves the score unchanged, editing the LAST context token changes
    // it, and shifting the boundary moves exactly one ScoreSequence term.
    procedure TestScoreCompletionScoresOnlyCompletion;
    // Pinned multiple-choice fixture: the gold answer wins, and acc vs
    // acc_norm disagree on a length-confounded item (p_cat^2 < p_dog < p_cat).
    procedure TestMultipleChoiceAccVsAccNorm;
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

procedure TTestNeuralNLPMetrics.SetLinearScorerWeights(NN: TNNet);
var
  J: integer;
  Head: TNNetLayer;
begin
  Head := NN.Layers[1]; // PointwiseConvLinear(V) over depth-1 token-id input
  for J := 0 to Head.Neurons.Count - 1 do
  begin
    Head.Neurons[J].Weights.Fill(0.02 * J); // single weight (1x1x1 kernel)
    Head.Neurons[J].BiasWeight := 0.03 * J;
  end;
  NN.ClearDeltas();
  NN.UpdateWeights(); // re-pack the pointwise-conv weight caches
end;

function TTestNeuralNLPMetrics.ExpectedRowLogProb(PrevId,
  Tgt: integer): TNeuralFloat;
var
  J: integer;
  SumExp: TNeuralFloat;
begin
  SumExp := 0;
  for J := 0 to csVocab - 1 do
    SumExp := SumExp + Exp(0.02 * J * PrevId + 0.03 * J);
  Result := (0.02 * Tgt * PrevId + 0.03 * Tgt) - Ln(SumExp);
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

procedure TTestNeuralNLPMetrics.TestScoreSequenceHandComputedSoftmax;
var
  NN: TNNet;
  Tokens, TooLong: TNeuralIntegerArray;
  LogProbs: TNeuralFloatDynArr;
  Pos: integer;
  Raised: boolean;
begin
  NN := BuildPerPositionLM(csCtx, csVocab);
  try
    SetLinearScorerWeights(NN);
    Tokens := TNeuralIntegerArray.Create(2, 5, 3, 7);
    LogProbs := ScoreSequence(NN, Tokens);
    AssertEquals('one logprob per token', 4, Length(LogProbs));
    AssertEquals('first token is never scored', 0.0, LogProbs[0], 1e-9);
    // Causal shift: LogProbs[Pos] is the softmax row conditioned on the
    // PREVIOUS token id, hand-computed.
    for Pos := 1 to 3 do
      AssertEquals('hand-computed logprob at position ' + IntToStr(Pos),
        ExpectedRowLogProb(Tokens[Pos - 1], Tokens[Pos]), LogProbs[Pos], 1e-5);
    // Out-of-vocab target: clamped (SafeLogProb(0)), never -Inf.
    LogProbs := ScoreSequence(NN, TNeuralIntegerArray.Create(2, csVocab + 5));
    AssertTrue('OOV target is clamped, finite',
      (LogProbs[1] < -30) and (LogProbs[1] > -1e6));
    // v1 context policy: longer than the window raises clearly.
    SetLength(TooLong, csCtx + 1);
    for Pos := 0 to High(TooLong) do TooLong[Pos] := 2;
    Raised := false;
    try
      ScoreSequence(NN, TooLong);
    except
      on EArgumentException do Raised := true;
    end;
    AssertTrue('over-context sequence raises EArgumentException', Raised);
  finally
    NN.Free;
  end;
end;

procedure TTestNeuralNLPMetrics.TestScoreSequenceMatchesPerplexity;
var
  NN: TNNet;
  Dict: TStringListInt;
  Corpus: TStringList;
  Stats: TNNetPerplexityStats;
  Toks: TNeuralIntegerArray;
  LogProbs: TNeuralFloatDynArr;
  LineIdx, Pos, NumScored: integer;
  SumLP: TNeuralFloat;
begin
  NN := BuildPerPositionLM(csCtx, csVocab);
  Dict := BuildDict();
  Corpus := TStringList.Create();
  try
    SetLinearScorerWeights(NN);
    // Every line fits the window, so the Perplexity windows and the
    // ScoreSequence windows are identical. ExcludeSpecialTokens MUST be
    // false: ScoreSequence never skips tokens, and which word carries a
    // special id (<2) is collation-dependent here (the test binary's
    // UTF-8 widestring manager interleaves '<eos>'/'<pad>' with the words,
    // handing id 1 to a regular word).
    Corpus.Add('cat dog egg fox');
    Corpus.Add('gold hat ice jam blue');
    SumLP := 0;
    NumScored := 0;
    for LineIdx := 0 to Corpus.Count - 1 do
    begin
      Dict.Tokenize(Corpus[LineIdx], Toks);
      LogProbs := ScoreSequence(NN, Toks);
      for Pos := 1 to High(LogProbs) do
      begin
        SumLP := SumLP + LogProbs[Pos];
        Inc(NumScored);
      end;
    end;
    Stats := Perplexity(NN, Dict, Corpus, false);
    AssertEquals('same number of scored positions',
      NumScored, Stats.PredictedTokens);
    AssertEquals('sum of ScoreSequence logprobs reproduces MeanNLL',
      -SumLP / NumScored, Stats.MeanNLL, 1e-5);
    AssertEquals('and therefore the perplexity',
      Exp(-SumLP / NumScored), Stats.Perplexity, 1e-4);
  finally
    Corpus.Free;
    Dict.Free;
    NN.Free;
  end;
end;

procedure TTestNeuralNLPMetrics.TestScoreCompletionScoresOnlyCompletion;
var
  NN: TNNet;
  Ctx, CtxMidEdit, CtxLastEdit, CtxShort, Comp, CompLong,
    EmptyCtx, Full: TNeuralIntegerArray;
  Base, MidEdit, LastEdit, Shifted: TNNetCompletionScore;
  LogProbs: TNeuralFloatDynArr;
  Raised: boolean;
begin
  NN := BuildPerPositionLM(csCtx, csVocab);
  try
    SetLinearScorerWeights(NN);
    Ctx := TNeuralIntegerArray.Create(2, 4, 5);
    Comp := TNeuralIntegerArray.Create(6, 7);
    Base := ScoreCompletion(NN, Ctx, Comp);
    AssertEquals('two completion tokens scored', 2, Base.TokenCount);
    AssertEquals('mean is sum / count', Base.SumLogProb / 2.0,
      Base.MeanLogProb, 1e-6);
    // Pinned values: completion rows are conditioned on tokens 5 and 6.
    AssertEquals('completion sum is hand-computable',
      ExpectedRowLogProb(5, 6) + ExpectedRowLogProb(6, 7),
      Base.SumLogProb, 1e-5);
    // (3a) Editing a NON-boundary context token (its row only scores another
    // CONTEXT token) leaves the completion score unchanged.
    CtxMidEdit := TNeuralIntegerArray.Create(2, 9, 5);
    MidEdit := ScoreCompletion(NN, CtxMidEdit, Comp);
    AssertEquals('non-boundary context edit does not change the score',
      Base.SumLogProb, MidEdit.SumLogProb, 1e-6);
    // (3b) Editing the LAST context token changes the conditioning of the
    // first completion token - predictably.
    CtxLastEdit := TNeuralIntegerArray.Create(2, 4, 9);
    LastEdit := ScoreCompletion(NN, CtxLastEdit, Comp);
    AssertTrue('last-context edit changes the score',
      Abs(LastEdit.SumLogProb - Base.SumLogProb) > 1e-3);
    AssertEquals('and lands exactly on the new conditioning',
      ExpectedRowLogProb(9, 6) + ExpectedRowLogProb(6, 7),
      LastEdit.SumLogProb, 1e-5);
    // (3c) Shifting the boundary by one adds exactly the ScoreSequence term
    // of the token that crossed from context into completion.
    CtxShort := TNeuralIntegerArray.Create(2, 4);
    CompLong := TNeuralIntegerArray.Create(5, 6, 7);
    Shifted := ScoreCompletion(NN, CtxShort, CompLong);
    Full := TNeuralIntegerArray.Create(2, 4, 5, 6, 7);
    LogProbs := ScoreSequence(NN, Full);
    AssertEquals('boundary shift moves exactly one term',
      Base.SumLogProb + LogProbs[2], Shifted.SumLogProb, 1e-5);
    // Empty context is rejected (nothing to condition the first token on).
    SetLength(EmptyCtx, 0);
    Raised := false;
    try
      ScoreCompletion(NN, EmptyCtx, Comp);
    except
      on EArgumentException do Raised := true;
    end;
    AssertTrue('empty context raises EArgumentException', Raised);
  finally
    NN.Free;
  end;
end;

procedure TTestNeuralNLPMetrics.TestMultipleChoiceAccVsAccNorm;
var
  NN: TNNet;
  Dict: TStringListInt;
  Items: array of TNNetMultipleChoiceItem;
  Stats: TNNetMultipleChoiceStats;
  Head: TNNetLayer;
  CatId, DogId, HatId: integer;
  SumExp, LpCat, LpDog: TNeuralFloat;
  GoldScore, OtherScore: TNNetCompletionScore;
begin
  NN := BuildPerPositionLM(csCtx, csVocab);
  Dict := BuildDict();
  try
    // Zero weights -> input-independent logits: p(cat) = e^3 / S and
    // p(dog) = e^2.6 / S with S = e^3 + e^2.6 + 10. Chosen so that
    // p_cat^2 < p_dog < p_cat: the LONG gold candidate [cat,cat] loses on
    // sum logprob (acc) but wins on mean logprob (acc_norm).
    SetUniformBiases(NN, 0.0);
    Head := NN.Layers[1];
    CatId := Dict.WordToInteger('cat');
    DogId := Dict.WordToInteger('dog');
    HatId := Dict.WordToInteger('hat');
    Head.Neurons[CatId].BiasWeight := 3.0;
    Head.Neurons[DogId].BiasWeight := 2.6;
    NN.ClearDeltas();
    NN.UpdateWeights();
    SumExp := Exp(3.0) + Exp(2.6) + (csVocab - 2) * Exp(0.0);
    LpCat := 3.0 - Ln(SumExp);
    LpDog := 2.6 - Ln(SumExp);
    AssertTrue('fixture is length-confounded: p_cat^2 < p_dog < p_cat',
      (2 * LpCat < LpDog) and (LpDog < LpCat));
    SetLength(Items, 2);
    // Item 0: gold [cat,cat] vs distractor [dog] - acc misses, acc_norm hits.
    Items[0].ContextTokens := TNeuralIntegerArray.Create(HatId);
    SetLength(Items[0].Candidates, 2);
    Items[0].Candidates[0] := TNeuralIntegerArray.Create(CatId, CatId);
    Items[0].Candidates[1] := TNeuralIntegerArray.Create(DogId);
    Items[0].GoldIndex := 0;
    // Item 1: gold [cat] vs distractor [dog] - the gold answer wins BOTH ways.
    Items[1].ContextTokens := TNeuralIntegerArray.Create(HatId);
    SetLength(Items[1].Candidates, 2);
    Items[1].Candidates[0] := TNeuralIntegerArray.Create(CatId);
    Items[1].Candidates[1] := TNeuralIntegerArray.Create(DogId);
    Items[1].GoldIndex := 0;
    // Pin the per-candidate scores against the analytic softmax first.
    GoldScore := ScoreCompletion(NN, Items[0].ContextTokens,
      Items[0].Candidates[0]);
    OtherScore := ScoreCompletion(NN, Items[0].ContextTokens,
      Items[0].Candidates[1]);
    AssertEquals('gold [cat,cat] sum = 2 ln p_cat', 2 * LpCat,
      GoldScore.SumLogProb, 1e-4);
    AssertEquals('distractor [dog] sum = ln p_dog', LpDog,
      OtherScore.SumLogProb, 1e-4);
    Stats := EvaluateMultipleChoice(NN, Items);
    AssertEquals('two items evaluated', 2, Stats.ItemCount);
    AssertEquals('acc: only the unconfounded item is right', 1,
      Stats.CorrectCount);
    AssertEquals('acc_norm: the gold answer wins both items', 2,
      Stats.CorrectNormCount);
    AssertEquals('Accuracy = 0.5', 0.5, Stats.Accuracy, 1e-9);
    AssertEquals('AccuracyNorm = 1.0', 1.0, Stats.AccuracyNorm, 1e-9);
  finally
    Dict.Free;
    NN.Free;
  end;
end;

initialization
  RegisterTest(TTestNeuralNLPMetrics);
end.
