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
    // Deterministic "copy the previous token" next-token head for a char LM:
    // neuron J reads ONLY input slot (X=0, depth=J) with a large weight. Under
    // the reversed one-hot encoding slot 0 holds the MOST RECENT prefix token,
    // so the argmax next token equals the last prefix token. If the prefix were
    // NOT reversed the most-recent token would land at the wrong slot and the
    // prediction would be constant - this pins the reversed-prefix encoding.
    procedure SetCopyPreviousWeights(NN: TNNet);
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
    // PerplexityStrided with Stride = window reproduces the disjoint
    // Perplexity() baseline EXACTLY (same scored count + MeanNLL bit-for-bit).
    procedure TestPerplexityStridedStrideEqualsWindowMatchesDisjoint;
    // A smaller stride scores every stream position past the first window
    // exactly once (coverage = StreamLen-1, a SUPERSET of the disjoint set
    // that skips each window's first token), with no double-counting.
    procedure TestPerplexityStridedTokenCoverage;
    // On a model with genuine long-range structure (next token = first stream
    // token) a smaller stride gives strictly LOWER MeanNLL than the disjoint
    // baseline, because the extra-context tokens are now predictable.
    procedure TestPerplexityStridedLowerNLLWithLongRange;
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
    // Shared-prefix batch scoring gives IDENTICAL scores to per-candidate
    // ScoreCompletion (and EvaluateMultipleChoice still agrees).
    procedure TestScoreCompletionsBatchMatchesPerCandidate;
    // Over-context sequences raise by default but score under LastWindow, and
    // the trailing positions match scoring the trailing window standalone.
    procedure TestScoreSequenceLastWindowScoresOverContext;
    // Pinned multiple-choice fixture: the gold answer wins, and acc vs
    // acc_norm disagree on a length-confounded item (p_cat^2 < p_dog < p_cat).
    procedure TestMultipleChoiceAccVsAccNorm;
    // chrF of an identical pair is exactly 1.0 (every order matches).
    procedure TestChrFIdenticalIsOne;
    // chrF / chrF++ pinned against a hand-written char-ngram reference that
    // reproduces sacrebleu's CHRF aggregation (sacrebleu not installed here).
    procedure TestChrFPinnedValues;
    // chrF is tokenizer/whitespace-independent by default: "a b"/"ab" tie.
    procedure TestChrFWhitespaceStripped;
    // CorpusChrF is the macro average of the per-pair scores.
    procedure TestCorpusChrFAverages;
    // distinct-n and repetition-rate on hand-computable fixtures
    // ("a a a a" -> distinct-1 = 1/4, repetition-1 = 3/4).
    procedure TestDistinctNAndRepetition;
    // distinct-n dual API (token-id vs whitespace string) agrees.
    procedure TestDistinctNTokenAndStringAPIsAgree;
    // self-BLEU: identical generations -> 1.0; reuses CorpusBLEU and equals
    // the hand-computed mean of pairwise BLEUs.
    procedure TestSelfBLEU;
    // Entity-level NER metrics (Task A): BIO decoding + span P/R/F1.
    procedure TestDecodeBIOEntities;
    procedure TestEntityScorePerfectMatch;
    procedure TestEntityScoreBoundaryError;
    procedure TestEntityScoreTypeMismatch;
    procedure TestCorpusEntityScoreMicroAverage;
    // QA span extraction (Task B): pinned logits -> pinned n-best spans.
    procedure TestExtractQASpansPinned;
    procedure TestExtractQASpansMaxLenAndOrder;
    // MMLU harness: with the monotone linear scorer the largest answer-letter
    // token id always wins, so per-question predictions (hence per-subject /
    // macro / micro accuracy) are analytic. Verifies the answer-letter argmax,
    // the per-subject buckets, and macro != micro on unbalanced subjects.
    procedure TestMMLUAnswerLetterArgmaxAndAggregation;
    // MMLUReport formatting contains the headline macro/micro lines.
    procedure TestMMLUReportFormatting;
    // ARC/PIQA/WinoGrande reuse EvaluateMultipleChoice (acc vs acc_norm).
    procedure TestARCPIQAWinoGrandeReuseMultipleChoiceCore;
    procedure TestMultipleChoiceReportFormatting;
    // LAMBADA greedy last-word reproduction (next-token-head reversed prefix).
    procedure TestLambadaGreedyLastWordAccuracy;
    procedure TestLambadaPerPositionHeadAndPerplexity;
    procedure TestLambadaReportFormatting;
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

procedure TTestNeuralNLPMetrics.SetCopyPreviousWeights(NN: TNNet);
var
  J: integer;
  Head: TNNetLayer;
begin
  Head := NN.Layers[1]; // FullConnectLinear(Vocab) over (Context,1,Vocab) one-hot
  for J := 0 to Head.Neurons.Count - 1 do
  begin
    Head.Neurons[J].Weights.Fill(0);
    // Read only the most-recent (reversed slot 0) one-hot of token J.
    Head.Neurons[J].Weights[0, 0, J] := 10.0;
    Head.Neurons[J].BiasWeight := 0;
  end;
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

// Joins token ids into a space-separated line of dictionary words (ids must
// be in 0..11). Lets the strided/disjoint tests pin an exact token stream.
function IdsToLine(Dict: TStringListInt; const Ids: array of integer): string;
var I: integer;
begin
  Result := '';
  for I := 0 to High(Ids) do
  begin
    if I > 0 then Result := Result + ' ';
    Result := Result + Dict.IntegerToWord(Ids[I]);
  end;
end;

// Same, for the slice Ids[Start .. Start+Count-1] of a (fixed) array.
function IdsRangeToLine(Dict: TStringListInt; const Ids: array of integer;
  Start, Count: integer): string;
var I: integer;
begin
  Result := '';
  for I := 0 to Count - 1 do
  begin
    if I > 0 then Result := Result + ' ';
    Result := Result + Dict.IntegerToWord(Ids[Start + I]);
  end;
end;

procedure TTestNeuralNLPMetrics.TestPerplexityStridedStrideEqualsWindowMatchesDisjoint;
var
  NN: TNNet;
  Dict: TStringListInt;
  StreamCorpus, ChoppedCorpus: TStringList;
  Strided, Disjoint: TNNetPerplexityStats;
  Stream: array[0..23] of integer;
  I, W: integer;
begin
  W := csCtx; // window = 8
  NN := BuildPerPositionLM(W, csVocab);
  Dict := BuildDict();
  StreamCorpus := TStringList.Create();
  ChoppedCorpus := TStringList.Create();
  try
    SetLinearScorerWeights(NN); // genuinely input-dependent per-position rows
    // A 24-token stream over the regular (non-special) ids 2..11.
    for I := 0 to 23 do Stream[I] := 2 + (I mod 10);
    // The whole stream as ONE corpus line (PerplexityStrided concatenates).
    StreamCorpus.Add(IdsToLine(Dict, Stream));
    // The SAME stream chopped into disjoint W-token lines: Perplexity() scores
    // positions 1..W-1 of each line - exactly the disjoint chop a Stride = W
    // slide produces (each window's first token is unscored).
    ChoppedCorpus.Add(IdsRangeToLine(Dict, Stream, 0, W));
    ChoppedCorpus.Add(IdsRangeToLine(Dict, Stream, W, W));
    ChoppedCorpus.Add(IdsRangeToLine(Dict, Stream, 2 * W, W));
    Disjoint := Perplexity(NN, Dict, ChoppedCorpus, false);
    Strided := PerplexityStrided(NN, Dict, StreamCorpus, W, false);
    AssertEquals('Stride=W scores the same token count as the disjoint chop',
      Disjoint.PredictedTokens, Strided.PredictedTokens);
    AssertEquals('Stride=W reproduces the disjoint MeanNLL bit-for-bit',
      Disjoint.MeanNLL, Strided.MeanNLL, 1e-5);
    AssertEquals('and therefore the disjoint perplexity',
      Disjoint.Perplexity, Strided.Perplexity, 1e-5);
  finally
    ChoppedCorpus.Free;
    StreamCorpus.Free;
    Dict.Free;
    NN.Free;
  end;
end;

procedure TTestNeuralNLPMetrics.TestPerplexityStridedTokenCoverage;
var
  NN: TNNet;
  Dict: TStringListInt;
  Corpus: TStringList;
  Full, Strided: TNNetPerplexityStats;
  Stream: array[0..23] of integer;
  I, W: integer;
begin
  W := csCtx; // 8
  NN := BuildPerPositionLM(W, csVocab);
  Dict := BuildDict();
  Corpus := TStringList.Create();
  try
    SetLinearScorerWeights(NN);
    for I := 0 to 23 do Stream[I] := 2 + (I mod 10);
    Corpus.Add(IdsToLine(Dict, Stream));
    // Stride < W: every stream position 1..StreamLen-1 is scored EXACTLY once
    // (FirstTgt/PrevEndAbs bookkeeping forbids double-counting and leaves no
    // gap), so coverage = 23 - a SUPERSET of the disjoint chop (which skips
    // each non-first window's first token). No skipped tokens here (all ids
    // 2..11 are regular).
    Strided := PerplexityStrided(NN, Dict, Corpus, 4, false);
    AssertEquals('every position past index 0 scored once', 23,
      Strided.PredictedTokens);
    AssertEquals('no skipped tokens', 0, Strided.SkippedTokens);
    // Sanity: Stride = StreamLen (>= W, clamped to W) is the disjoint chop,
    // which DOES skip the two interior window-first tokens (positions 8, 16).
    Full := PerplexityStrided(NN, Dict, Corpus, 1000, false);
    AssertEquals('disjoint chop scores fewer tokens (skips window-firsts)',
      21, Full.PredictedTokens);
    AssertTrue('strided covers strictly more tokens than the disjoint chop',
      Strided.PredictedTokens > Full.PredictedTokens);
  finally
    Corpus.Free;
    Dict.Free;
    NN.Free;
  end;
end;

procedure TTestNeuralNLPMetrics.TestPerplexityStridedLowerNLLWithLongRange;
var
  NN: TNNet;
  Dict: TStringListInt;
  Corpus: TStringList;
  Strided, Disjoint: TNNetPerplexityStats;
  Stream: array[0..23] of integer;
  I, W, HiId: integer;
begin
  W := csCtx; // 8
  NN := BuildPerPositionLM(W, csVocab);
  Dict := BuildDict();
  Corpus := TStringList.Create();
  try
    // The linear scorer's softmax favours the HIGHEST token id (logit(J) rises
    // with J). The disjoint chop NEVER scores the first token of an interior
    // window (stream positions 8 and 16); a smaller stride DOES, each with its
    // in-window predecessor as real left context. By placing the easiest-to-
    // predict token (id 11) at exactly those boundary positions, the tokens
    // the disjoint chop omits are the most predictable ones, so folding them
    // into the average can only LOWER the MeanNLL: the bounded-left-context
    // re-scoring beats the chop. (A pointwise head cannot mix across
    // positions, so this is the honest small-net analogue of "real long-range
    // structure" - the extra-context tokens are the ones that become
    // scorable; documented choice.)
    SetLinearScorerWeights(NN);
    HiId := csVocab - 1; // 11, the argmax-favoured token
    for I := 0 to 23 do Stream[I] := 2 + (I mod 10);
    Stream[8] := HiId;
    Stream[16] := HiId;
    Corpus.Add(IdsToLine(Dict, Stream));
    Disjoint := PerplexityStrided(NN, Dict, Corpus, W, false);     // = chop
    Strided := PerplexityStrided(NN, Dict, Corpus, 4, false);      // S < W
    AssertTrue('smaller stride scores more tokens',
      Strided.PredictedTokens > Disjoint.PredictedTokens);
    AssertTrue('smaller stride gives <= MeanNLL than the disjoint baseline',
      Strided.MeanNLL <= Disjoint.MeanNLL + 1e-6);
    AssertTrue('and the drop is real (boundary tokens are easy)',
      Strided.MeanNLL < Disjoint.MeanNLL);
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

procedure TTestNeuralNLPMetrics.TestScoreCompletionsBatchMatchesPerCandidate;
var
  NN: TNNet;
  Ctx: TNeuralIntegerArray;
  Cands: array of TNeuralIntegerArray;
  Batch: TNNetCompletionScoreArray;
  Single: TNNetCompletionScore;
  Cand: integer;
begin
  // Single next-token head (BuildCharLM): the batch path scores ONLY the
  // completion positions per candidate, skipping the shared-context forwards.
  // Result MUST be bit-identical to per-candidate ScoreCompletion.
  RandSeed := 424242;
  NN := BuildCharLM(csCtx, csVocab);
  try
    NN.InitWeights();
    Ctx := TNeuralIntegerArray.Create(2, 4, 5);
    SetLength(Cands, 3);
    Cands[0] := TNeuralIntegerArray.Create(6, 7);
    Cands[1] := TNeuralIntegerArray.Create(3);
    Cands[2] := TNeuralIntegerArray.Create(8, 9, 10);
    Batch := ScoreCompletionsBatch(NN, Ctx, Cands);
    AssertEquals('one score per candidate', 3, Length(Batch));
    for Cand := 0 to High(Cands) do
    begin
      Single := ScoreCompletion(NN, Ctx, Cands[Cand]);
      AssertEquals('batch SumLogProb == per-candidate, cand ' + IntToStr(Cand),
        Single.SumLogProb, Batch[Cand].SumLogProb, 1e-9);
      AssertEquals('batch MeanLogProb == per-candidate, cand ' + IntToStr(Cand),
        Single.MeanLogProb, Batch[Cand].MeanLogProb, 1e-9);
      AssertEquals('batch TokenCount == per-candidate, cand ' + IntToStr(Cand),
        Single.TokenCount, Batch[Cand].TokenCount);
    end;
  finally
    NN.Free;
  end;
end;

procedure TTestNeuralNLPMetrics.TestScoreSequenceLastWindowScoresOverContext;
var
  NN, CharNN: TNNet;
  Tokens, Window: TNeuralIntegerArray;
  LogProbs, WinProbs: TNeuralFloatDynArr;
  Pos: integer;
  Raised: boolean;
begin
  // Per-position LM with deterministic weights: it conditions only on the
  // immediately previous token, so last-window scoring of an over-context
  // sequence reproduces the hand-computed per-token logprob exactly.
  NN := BuildPerPositionLM(csCtx, csVocab);
  try
    SetLinearScorerWeights(NN);
    // csCtx+2 tokens: longer than the window (csCtx). Default policy raises.
    SetLength(Tokens, csCtx + 2);
    for Pos := 0 to High(Tokens) do Tokens[Pos] := (Pos * 3 + 2) mod csVocab;
    Raised := false;
    try
      ScoreSequence(NN, Tokens);
    except
      on EArgumentException do Raised := true;
    end;
    AssertTrue('over-context raises without LastWindow', Raised);
    // With LastWindow it scores instead of raising.
    LogProbs := ScoreSequence(NN, Tokens, true);
    AssertEquals('one logprob per token', Length(Tokens), Length(LogProbs));
    AssertEquals('first token never scored', 0.0, LogProbs[0], 1e-9);
    for Pos := 1 to High(Tokens) do
      AssertEquals('last-window logprob conditions on previous token, pos ' +
        IntToStr(Pos),
        ExpectedRowLogProb(Tokens[Pos - 1], Tokens[Pos]), LogProbs[Pos], 1e-5);
  finally
    NN.Free;
  end;
  // Single next-token head: the last scored position's window equals scoring
  // the trailing context-window standalone (LastWindow truncates the prefix).
  RandSeed := 424242;
  CharNN := BuildCharLM(csCtx, csVocab);
  try
    CharNN.InitWeights();
    SetLength(Tokens, csCtx + 3);
    for Pos := 0 to High(Tokens) do Tokens[Pos] := (Pos * 5 + 1) mod csVocab;
    LogProbs := ScoreSequence(CharNN, Tokens, true);
    AssertEquals('char-LM over-context scores under LastWindow',
      Length(Tokens), Length(LogProbs));
    // Last position: window = the csCtx tokens immediately before it plus it.
    // Build that standalone sequence (length csCtx+1) and score its last token.
    Window := Copy(Tokens, High(Tokens) - csCtx, csCtx + 1);
    WinProbs := ScoreSequence(CharNN, Window); // fits exactly, no flag needed
    AssertEquals('last-window tail matches standalone trailing-window score',
      WinProbs[High(WinProbs)], LogProbs[High(LogProbs)], 1e-5);
  finally
    CharNN.Free;
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

procedure TTestNeuralNLPMetrics.TestChrFIdenticalIsOne;
const
  S = 'the cat sat on the mat';
begin
  // Every char (and word) n-gram order matches perfectly -> F = 1 each order.
  AssertEquals('identical pair chrF', 1.0, ChrF(S, S), 1e-9);
  AssertEquals('identical pair chrF++', 1.0, ChrFpp(S, S), 1e-9);
end;

procedure TTestNeuralNLPMetrics.TestChrFPinnedValues;
begin
  // Reference values from a hand-written char-ngram script reproducing
  // sacrebleu's CHRF aggregation (sacrebleu is not installed in this env):
  // per-order F_beta=2 averaged over effective orders, whitespace stripped.
  // "abc"/"abd": char n=1 P=R=2/3 (F=2/3); n=2 match 1/2 (F=1/2); n=3 match 0
  // (F=0); n=4..6 empty (skipped) -> chrF = (2/3+1/2+0)/3 = 7/18.
  AssertEquals('chrF abc/abd', 7.0 / 18.0, ChrF('abc', 'abd'), 1e-5);
  // chrF++ adds word unigram (no match, F=0) - bigrams empty/skipped -> the
  // 3 char orders + 1 word order = (7/6 + 0)/4 = 7/24.
  AssertEquals('chrF++ abc/abd', 7.0 / 24.0, ChrFpp('abc', 'abd'), 1e-5);
  // A realistic sentence pair, pinned to the reference script's 0..1 output.
  AssertEquals('chrF sentence pair', 0.6010827407886231,
    ChrF('the cat sat on the mat', 'a cat sat upon the mat'), 1e-5);
  AssertEquals('chrF++ sentence pair', 0.5841453889248007,
    ChrFpp('the cat sat on the mat', 'a cat sat upon the mat'), 1e-5);
end;

procedure TTestNeuralNLPMetrics.TestChrFWhitespaceStripped;
begin
  // sacrebleu default strips whitespace before char n-grams, so spacing does
  // not matter: "a b" and "ab" share identical char n-grams -> chrF = 1.
  AssertEquals('whitespace is stripped by default', 1.0,
    ChrF('a b', 'ab'), 1e-9);
  // With IncludeWhitespace the space participates and the score drops below 1.
  AssertTrue('whitespace-included chrF sees the space',
    ChrF('a b', 'ab', 6, 2.0, 0, true) < 1.0);
end;

procedure TTestNeuralNLPMetrics.TestCorpusChrFAverages;
var
  P1, P2: TNeuralFloat;
begin
  P1 := ChrF('abc', 'abd');
  P2 := ChrF('the cat', 'the cat'); // identical -> 1.0
  AssertEquals('corpus chrF is the macro average', (P1 + P2) / 2.0,
    CorpusChrF(['abc', 'the cat'], ['abd', 'the cat']), 1e-5);
end;

procedure TTestNeuralNLPMetrics.TestDistinctNAndRepetition;
begin
  // "a a a a": 4 unigrams, 1 distinct -> distinct-1 = 1/4, repetition = 3/4.
  AssertEquals('distinct-1 of "a a a a"', 1.0 / 4.0, DistinctN('a a a a', 1), 1e-9);
  AssertEquals('repetition-1 of "a a a a"', 3.0 / 4.0,
    RepetitionRate('a a a a', 1), 1e-9);
  AssertEquals('repeated-token-rate alias', 3.0 / 4.0,
    RepeatedTokenRate('a a a a'), 1e-9);
  // "a a a a": 3 bigrams ("a a" x3), 1 distinct -> distinct-2 = 1/3.
  AssertEquals('distinct-2 of "a a a a"', 1.0 / 3.0, DistinctN('a a a a', 2), 1e-9);
  // All-distinct generation -> distinct-1 = 1, repetition = 0.
  AssertEquals('distinct-1 of "a b c d"', 1.0, DistinctN('a b c d', 1), 1e-9);
  AssertEquals('repetition-1 of "a b c d"', 0.0,
    RepetitionRate('a b c d', 1), 1e-9);
  // "the cat the dog the": 5 unigrams, 3 distinct (the,cat,dog) -> 3/5.
  AssertEquals('distinct-1 of repeated-"the"', 3.0 / 5.0,
    DistinctN('the cat the dog the', 1), 1e-9);
  // Fewer than N tokens -> 0 (no n-grams to count).
  AssertEquals('distinct-2 of a single token', 0.0, DistinctN('a', 2), 1e-9);
  AssertEquals('repetition-2 of a single token', 0.0,
    RepetitionRate('a', 2), 1e-9);
end;

procedure TTestNeuralNLPMetrics.TestDistinctNTokenAndStringAPIsAgree;
var
  Ids: TNeuralIntegerArray;
begin
  // "a a a a" -> ids [0,0,0,0]; both APIs must agree.
  Ids := TNeuralIntegerArray.Create(0, 0, 0, 0);
  AssertEquals('token/string distinct-1 agree', DistinctN('a a a a', 1),
    DistinctN(Ids, 1), 1e-9);
  AssertEquals('token/string distinct-2 agree', DistinctN('a a a a', 2),
    DistinctN(Ids, 2), 1e-9);
end;

procedure TTestNeuralNLPMetrics.TestSelfBLEU;
var
  Gens: array of TNeuralIntegerArray;
  Expected: TNeuralFloat;
begin
  // Three IDENTICAL generations: every candidate-vs-other BLEU is 1.0, so
  // self-BLEU = 1.0 (maximal redundancy / zero diversity).
  AssertEquals('self-BLEU of identical generations',
    1.0, SelfBLEU(['a b c a', 'a b c a', 'a b c a'], 4, true), 1e-9);
  // Two generations: self-BLEU is symmetric, = mean of the two single-ref
  // BLEUs, which here equals CorpusBLEU(one vs the other) both directions.
  // Hand-anchor against the reused CorpusBLEU machinery directly.
  Expected := (CorpusBLEU(['the cat sat'], ['the cat ran'], 4, true) +
               CorpusBLEU(['the cat ran'], ['the cat sat'], 4, true)) / 2.0;
  AssertEquals('self-BLEU reuses CorpusBLEU (two generations)',
    Expected, SelfBLEU(['the cat sat', 'the cat ran'], 4, true), 1e-9);
  // Token-id and string overloads agree on the same content.
  SetLength(Gens, 2);
  Gens[0] := TNeuralIntegerArray.Create(0, 1, 2); // 'the cat sat'
  Gens[1] := TNeuralIntegerArray.Create(0, 1, 3); // 'the cat ran'
  AssertEquals('token/string self-BLEU agree',
    SelfBLEU(['the cat sat', 'the cat ran'], 4, true),
    SelfBLEU(Gens, 4, true), 1e-9);
  // Fewer than two generations -> 0 (nothing to compare against).
  AssertEquals('self-BLEU of a single generation is 0',
    0.0, SelfBLEU(['a b c'], 4, true), 1e-9);
end;

procedure TTestNeuralNLPMetrics.TestDecodeBIOEntities;
var
  Spans: TNNetEntitySpanArray;
begin
  // 'B-PER I-PER O B-LOC' -> PER[0..1], LOC[3..3].
  Spans := DecodeBIOEntities(['B-PER', 'I-PER', 'O', 'B-LOC']);
  AssertEquals('two entities decoded', 2, Length(Spans));
  AssertEquals('first type', 'PER', Spans[0].EntityType);
  AssertEquals('first start', 0, Spans[0].TokenStart);
  AssertEquals('first end', 1, Spans[0].TokenEnd);
  AssertEquals('second type', 'LOC', Spans[1].EntityType);
  AssertEquals('second start', 3, Spans[1].TokenStart);
  AssertEquals('second end', 3, Spans[1].TokenEnd);
  // Two adjacent B- of the same type are TWO separate entities (B closes B).
  Spans := DecodeBIOEntities(['B-PER', 'B-PER']);
  AssertEquals('adjacent B-PER are two spans', 2, Length(Spans));
  // Lenient IOB2: a leading I- opens a span.
  Spans := DecodeBIOEntities(['I-ORG', 'I-ORG']);
  AssertEquals('leading I-ORG opens one span', 1, Length(Spans));
  AssertEquals('lenient start', 0, Spans[0].TokenStart);
  AssertEquals('lenient end', 1, Spans[0].TokenEnd);
end;

procedure TTestNeuralNLPMetrics.TestEntityScorePerfectMatch;
var
  S: TNNetEntityScore;
begin
  S := EntityScore(['B-PER', 'I-PER', 'O', 'B-LOC'],
                   ['B-PER', 'I-PER', 'O', 'B-LOC']);
  AssertEquals('TP', 2, S.TruePos);
  AssertEquals('FP', 0, S.FalsePos);
  AssertEquals('FN', 0, S.FalseNeg);
  AssertEquals('P', 1.0, S.Precision, 1e-9);
  AssertEquals('R', 1.0, S.Recall, 1e-9);
  AssertEquals('F1', 1.0, S.F1, 1e-9);
end;

procedure TTestNeuralNLPMetrics.TestEntityScoreBoundaryError;
var
  S: TNNetEntityScore;
begin
  // Gold one PER span [0..1]; pred only [0..0] (boundary/length error).
  // Spans differ -> 0 TP, 1 FP (pred), 1 FN (gold). P=R=F1=0.
  S := EntityScore(['B-PER', 'O'], ['B-PER', 'I-PER']);
  AssertEquals('TP', 0, S.TruePos);
  AssertEquals('FP', 1, S.FalsePos);
  AssertEquals('FN', 1, S.FalseNeg);
  AssertEquals('F1', 0.0, S.F1, 1e-9);
  // Split entity: gold one [0..2], pred two [0..0] and [2..2] (the seqeval
  // split-error case). 0 TP, 2 FP, 1 FN -> P=0, R=0.
  S := EntityScore(['B-PER', 'O', 'B-PER'], ['B-PER', 'I-PER', 'I-PER']);
  AssertEquals('split TP', 0, S.TruePos);
  AssertEquals('split FP', 2, S.FalsePos);
  AssertEquals('split FN', 1, S.FalseNeg);
end;

procedure TTestNeuralNLPMetrics.TestEntityScoreTypeMismatch;
var
  S: TNNetEntityScore;
begin
  // Right boundary, wrong type: gold LOC[0..0], pred PER[0..0]. Not a match.
  S := EntityScore(['B-PER'], ['B-LOC']);
  AssertEquals('type-mismatch TP', 0, S.TruePos);
  AssertEquals('type-mismatch FP', 1, S.FalsePos);
  AssertEquals('type-mismatch FN', 1, S.FalseNeg);
  // One correct + one type error: gold PER[0..0],LOC[2..2]; pred PER[0..0],
  // ORG[2..2]. TP=1, FP=1, FN=1 -> P=R=0.5, F1=0.5.
  S := EntityScore(['B-PER', 'O', 'B-ORG'], ['B-PER', 'O', 'B-LOC']);
  AssertEquals('mixed TP', 1, S.TruePos);
  AssertEquals('mixed FP', 1, S.FalsePos);
  AssertEquals('mixed FN', 1, S.FalseNeg);
  AssertEquals('mixed P', 0.5, S.Precision, 1e-9);
  AssertEquals('mixed R', 0.5, S.Recall, 1e-9);
  AssertEquals('mixed F1', 0.5, S.F1, 1e-9);
end;

procedure TTestNeuralNLPMetrics.TestCorpusEntityScoreMicroAverage;
var
  Pred, Gold: array of TStringArray;
  S: TNNetEntityScore;
begin
  SetLength(Pred, 2);
  SetLength(Gold, 2);
  // Sentence 0: perfect single PER. Sentence 1: type error (1 FP, 1 FN).
  Pred[0] := TStringArray.Create('B-PER', 'I-PER');
  Gold[0] := TStringArray.Create('B-PER', 'I-PER');
  Pred[1] := TStringArray.Create('B-ORG');
  Gold[1] := TStringArray.Create('B-LOC');
  S := CorpusEntityScore(Pred, Gold);
  // Pooled: TP=1, FP=1, FN=1 -> P=R=F1=0.5.
  AssertEquals('micro TP', 1, S.TruePos);
  AssertEquals('micro FP', 1, S.FalsePos);
  AssertEquals('micro FN', 1, S.FalseNeg);
  AssertEquals('micro F1', 0.5, S.F1, 1e-9);
end;

procedure TTestNeuralNLPMetrics.TestExtractQASpansPinned;
var
  StartLogits, EndLogits: TNeuralFloatDynArr;
  Spans: TNNetQASpanArray;
begin
  // 5 tokens. Best start at idx 1 (3.0), best end at idx 3 (4.0): span [1..3].
  StartLogits := TNeuralFloatDynArr.Create(0.5, 3.0, 0.1, 0.2, 0.0);
  EndLogits   := TNeuralFloatDynArr.Create(0.0, 0.3, 1.0, 4.0, 0.1);
  Spans := ExtractQASpans(StartLogits, EndLogits, 20, 30, 20);
  AssertTrue('at least one span', Length(Spans) >= 1);
  AssertEquals('best start', 1, Spans[0].TokenStart);
  AssertEquals('best end', 3, Spans[0].TokenEnd);
  AssertEquals('best score', 7.0, Spans[0].Score, 1e-6);
  // Spans are ranked descending by score.
  AssertTrue('descending order', Spans[0].Score >= Spans[1].Score);
end;

procedure TTestNeuralNLPMetrics.TestExtractQASpansMaxLenAndOrder;
var
  StartLogits, EndLogits: TNeuralFloatDynArr;
  Spans: TNNetQASpanArray;
begin
  // Best raw pair would be start 0, end 4 (len 5), but MaxAnswerLen=2 forbids
  // it. With cap 2 the valid top pair is start 0 (5.0) end 1 (5.0) = 10.0.
  StartLogits := TNeuralFloatDynArr.Create(5.0, 0.0, 0.0, 0.0, 0.0);
  EndLogits   := TNeuralFloatDynArr.Create(0.0, 5.0, 0.0, 0.0, 4.0);
  Spans := ExtractQASpans(StartLogits, EndLogits, 20, 2, 20);
  AssertEquals('capped start', 0, Spans[0].TokenStart);
  AssertEquals('capped end', 1, Spans[0].TokenEnd);
  AssertEquals('capped len <= 2', true,
    (Spans[0].TokenEnd - Spans[0].TokenStart + 1) <= 2);
  // end >= start enforced.
  AssertTrue('end>=start', Spans[0].TokenEnd >= Spans[0].TokenStart);
  // NBest cap limits the list length.
  Spans := ExtractQASpans(StartLogits, EndLogits, 20, 30, 3);
  AssertTrue('n-best capped at 3', Length(Spans) <= 3);
end;

procedure TTestNeuralNLPMetrics.TestMMLUAnswerLetterArgmaxAndAggregation;
var
  NN: TNNet;
  Questions: array of TNNetMMLUQuestion;
  LetterTokens: array[0..3] of integer;
  Stats: TNNetMMLUStats;
begin
  // Per-position monotone linear scorer: with a non-negative previous token the
  // single-token logit logit(J) = 0.02*J*PrevId + 0.03*J is strictly increasing
  // in the token id J, so among the four answer-letter tokens the one with the
  // LARGEST id always has the highest log-prob -> it is the harness prediction.
  NN := BuildPerPositionLM(csCtx, csVocab);
  try
    SetLinearScorerWeights(NN);

    // Letter tokens A/B/C/D mapped to ids 2,3,4,5 (D has the largest id, so the
    // monotone scorer ALWAYS predicts D = index 3).
    LetterTokens[0] := 2; LetterTokens[1] := 3;
    LetterTokens[2] := 4; LetterTokens[3] := 5;

    // 3 questions in subject 0 (gold D twice -> 2/3) and 1 in subject 1 (gold D
    // -> 1/1). The prompt tokens just need a non-negative last token (PrevId).
    SetLength(Questions, 4);
    Questions[0].PromptTokens := TNeuralIntegerArray.Create(2, 6);
    Questions[0].GoldLetter := 3; Questions[0].SubjectIndex := 0; // D, correct
    Questions[1].PromptTokens := TNeuralIntegerArray.Create(2, 7);
    Questions[1].GoldLetter := 0; Questions[1].SubjectIndex := 0; // A, wrong
    Questions[2].PromptTokens := TNeuralIntegerArray.Create(3, 5);
    Questions[2].GoldLetter := 3; Questions[2].SubjectIndex := 0; // D, correct
    Questions[3].PromptTokens := TNeuralIntegerArray.Create(2, 4);
    Questions[3].GoldLetter := 3; Questions[3].SubjectIndex := 1; // D, correct

    Stats := EvaluateMMLU(NN, Questions, LetterTokens, 2);

    AssertEquals('items scored', 4, Stats.ItemCount);
    AssertEquals('subjects scored', 2, Stats.SubjectCount);
    AssertEquals('total correct (3 of 4 are D)', 3, Stats.CorrectCount);
    // Subject 0: 2/3 ; subject 1: 1/1.
    AssertEquals('subject0 total', 3, Stats.PerSubject[0].Total);
    AssertEquals('subject0 correct', 2, Stats.PerSubject[0].Correct);
    AssertEquals('subject0 acc', 2.0 / 3.0, Stats.PerSubject[0].Accuracy, 1e-6);
    AssertEquals('subject1 acc', 1.0, Stats.PerSubject[1].Accuracy, 1e-6);
    // Macro = mean(2/3, 1) = 5/6 ; micro = 3/4 -> they DISAGREE (unbalanced).
    AssertEquals('macro = mean over subjects', 5.0 / 6.0,
      Stats.MacroAccuracy, 1e-6);
    AssertEquals('micro = pooled', 0.75, Stats.MicroAccuracy, 1e-6);
    AssertTrue('macro != micro on unbalanced subjects',
      Abs(Stats.MacroAccuracy - Stats.MicroAccuracy) > 1e-3);

    // Out-of-range subject index is skipped (not scored, not counted).
    SetLength(Questions, 1);
    Questions[0].PromptTokens := TNeuralIntegerArray.Create(2, 6);
    Questions[0].GoldLetter := 3; Questions[0].SubjectIndex := 9; // >= NumSubjects
    Stats := EvaluateMMLU(NN, Questions, LetterTokens, 2);
    AssertEquals('out-of-range subject skipped', 0, Stats.ItemCount);
  finally
    NN.Free;
  end;
end;

procedure TTestNeuralNLPMetrics.TestMMLUReportFormatting;
var
  Stats: TNNetMMLUStats;
  Report: string;
begin
  Stats.MacroAccuracy := 0.5;
  Stats.MicroAccuracy := 0.4;
  Stats.ItemCount := 5;
  Stats.CorrectCount := 2;
  Stats.SubjectCount := 2;
  SetLength(Stats.PerSubject, 2);
  Stats.PerSubject[0].Correct := 1; Stats.PerSubject[0].Total := 2;
  Stats.PerSubject[0].Accuracy := 0.5;
  Stats.PerSubject[1].Correct := 1; Stats.PerSubject[1].Total := 3;
  Stats.PerSubject[1].Accuracy := 1.0 / 3.0;

  Report := MMLUReport(Stats, ['algebra', 'history'], 5);
  AssertTrue('header mentions 5-shot', Pos('5-shot', Report) > 0);
  AssertTrue('lists named subject', Pos('algebra', Report) > 0);
  AssertTrue('has macro-average line', Pos('macro-average', Report) > 0);
  AssertTrue('has micro-average line', Pos('micro-average', Report) > 0);
end;

procedure TTestNeuralNLPMetrics.TestARCPIQAWinoGrandeReuseMultipleChoiceCore;
var
  NN: TNNet;
  Dict: TStringListInt;
  Items: array of TNNetMultipleChoiceItem;
  MCStats, ArcStats, PiqaStats, WinoStats: TNNetMultipleChoiceStats;
  Head: TNNetLayer;
  CatId, DogId, HatId: integer;
  SumExp, LpCat, LpDog: TNeuralFloat;
begin
  // Same length-confounded fixture as TestMultipleChoiceAccVsAccNorm: the long
  // gold candidate loses on acc (sum logprob) but wins on acc_norm (mean). This
  // exercises the ARC/PIQA acc_norm headline vs the WinoGrande acc headline.
  NN := BuildPerPositionLM(csCtx, csVocab);
  Dict := BuildDict();
  try
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
    AssertTrue('fixture is length-confounded', (2 * LpCat < LpDog) and (LpDog < LpCat));
    SetLength(Items, 2);
    // ARC-style 3-choice item (variable choice count is fine per item).
    Items[0].ContextTokens := TNeuralIntegerArray.Create(HatId);
    SetLength(Items[0].Candidates, 3);
    Items[0].Candidates[0] := TNeuralIntegerArray.Create(CatId, CatId); // gold (long)
    Items[0].Candidates[1] := TNeuralIntegerArray.Create(DogId);
    Items[0].Candidates[2] := TNeuralIntegerArray.Create(DogId, DogId);
    Items[0].GoldIndex := 0;
    // 2-choice item where gold wins both ways (PIQA/WinoGrande shape).
    Items[1].ContextTokens := TNeuralIntegerArray.Create(HatId);
    SetLength(Items[1].Candidates, 2);
    Items[1].Candidates[0] := TNeuralIntegerArray.Create(CatId);
    Items[1].Candidates[1] := TNeuralIntegerArray.Create(DogId);
    Items[1].GoldIndex := 0;

    MCStats := EvaluateMultipleChoice(NN, Items);
    ArcStats := EvaluateARC(NN, Items);
    PiqaStats := EvaluatePIQA(NN, Items);
    WinoStats := EvaluateWinoGrande(NN, Items);

    // The benchmark wrappers are EXACT pass-throughs to the shared core.
    AssertEquals('ARC acc == core acc', MCStats.Accuracy, ArcStats.Accuracy, 1e-12);
    AssertEquals('ARC acc_norm == core', MCStats.AccuracyNorm, ArcStats.AccuracyNorm, 1e-12);
    AssertEquals('PIQA acc == core', MCStats.Accuracy, PiqaStats.Accuracy, 1e-12);
    AssertEquals('PIQA acc_norm == core', MCStats.AccuracyNorm, PiqaStats.AccuracyNorm, 1e-12);
    AssertEquals('Wino acc == core', MCStats.Accuracy, WinoStats.Accuracy, 1e-12);
    AssertEquals('Wino acc_norm == core', MCStats.AccuracyNorm, WinoStats.AccuracyNorm, 1e-12);

    // Headline semantics: acc misses the confounded item (0.5) but acc_norm hits
    // both (1.0) - ARC/PIQA report acc_norm, WinoGrande reports acc.
    AssertEquals('acc = 0.5 (confounded item missed)', 0.5, ArcStats.Accuracy, 1e-9);
    AssertEquals('acc_norm = 1.0 (gold wins normalized)', 1.0, ArcStats.AccuracyNorm, 1e-9);
    AssertEquals('two items scored', 2, ArcStats.ItemCount);
  finally
    Dict.Free;
    NN.Free;
  end;
end;

procedure TTestNeuralNLPMetrics.TestMultipleChoiceReportFormatting;
var
  Stats: TNNetMultipleChoiceStats;
  ArcReport, WinoReport: string;
begin
  Stats.Accuracy := 0.4;
  Stats.AccuracyNorm := 0.6;
  Stats.ItemCount := 5;
  Stats.CorrectCount := 2;
  Stats.CorrectNormCount := 3;
  ArcReport := MultipleChoiceReport(Stats, 'ARC-Challenge', true);
  WinoReport := MultipleChoiceReport(Stats, 'WinoGrande', false);
  AssertTrue('ARC report names benchmark', Pos('ARC-Challenge', ArcReport) > 0);
  AssertTrue('ARC report has acc_norm line', Pos('acc_norm', ArcReport) > 0);
  // The headline annotation tracks the convention (acc_norm for ARC, acc for Wino).
  AssertTrue('ARC headline is on acc_norm',
    Pos('acc_norm', ArcReport) < Pos('headline', ArcReport));
  AssertTrue('Wino names benchmark', Pos('WinoGrande', WinoReport) > 0);
  AssertTrue('Wino headline is on acc',
    (Pos('headline', WinoReport) > 0) and
    (Pos('headline', WinoReport) < Pos('acc_norm', WinoReport)));
end;

procedure TTestNeuralNLPMetrics.TestLambadaGreedyLastWordAccuracy;
var
  CharNN: TNNet;
  Examples: array of TNNetLambadaExample;
  Stats: TNNetLambadaStats;
begin
  // Single next-token head that COPIES the most-recent prefix token (via the
  // reversed one-hot slot 0). So the greedy final-word prediction equals the
  // last context token - LAMBADA is correct iff every final-word token equals
  // the token immediately before it. This pins the reversed-prefix encoding:
  // were it not reversed the prediction would be constant and accuracy chance.
  CharNN := BuildCharLM(csCtx, csVocab);
  try
    SetCopyPreviousWeights(CharNN);
    SetLength(Examples, 3);
    // Ex0: context ends in token 7; final word [7] -> copy reproduces it (HIT).
    Examples[0].ContextTokens := TNeuralIntegerArray.Create(2, 3, 7);
    Examples[0].FinalWordTokens := TNeuralIntegerArray.Create(7);
    // Ex1: context ends in 5; final word [9] -> copy predicts 5 != 9 (MISS).
    Examples[1].ContextTokens := TNeuralIntegerArray.Create(4, 5);
    Examples[1].FinalWordTokens := TNeuralIntegerArray.Create(9);
    // Ex2: multi-token final word [8,8]: pos1 copies ctx-last 8 (ok), pos2 copies
    // the gold 8 just placed (ok) -> whole word reproduced (HIT).
    Examples[2].ContextTokens := TNeuralIntegerArray.Create(2, 8);
    Examples[2].FinalWordTokens := TNeuralIntegerArray.Create(8, 8);

    Stats := EvaluateLAMBADA(CharNN, Examples);
    AssertEquals('three examples scored', 3, Stats.ItemCount);
    AssertEquals('two greedy hits (ex0, ex2)', 2, Stats.CorrectCount);
    AssertEquals('accuracy = 2/3', 2.0 / 3.0, Stats.Accuracy, 1e-9);
    AssertTrue('not pinned at chance (reversed prefix matched)',
      Stats.Accuracy > 0.5);
  finally
    CharNN.Free;
  end;
end;

procedure TTestNeuralNLPMetrics.TestLambadaPerPositionHeadAndPerplexity;
var
  NN: TNNet;
  Examples: array of TNNetLambadaExample;
  Stats: TNNetLambadaStats;
  ExpNLL: TNeuralFloat;
begin
  // Per-position monotone scorer: argmax next token is always the LARGEST id
  // (csVocab-1 = 11) for a non-negative previous token. So a final word [11]
  // is reproduced; any other final token misses. Perplexity is the analytic
  // teacher-forced last-word NLL.
  NN := BuildPerPositionLM(csCtx, csVocab);
  try
    SetLinearScorerWeights(NN);
    SetLength(Examples, 2);
    // Ex0: ctx ends in 6; final word [11] -> argmax is 11 (HIT).
    Examples[0].ContextTokens := TNeuralIntegerArray.Create(2, 6);
    Examples[0].FinalWordTokens := TNeuralIntegerArray.Create(csVocab - 1);
    // Ex1: ctx ends in 7; final word [4] -> argmax is 11 != 4 (MISS).
    Examples[1].ContextTokens := TNeuralIntegerArray.Create(2, 7);
    Examples[1].FinalWordTokens := TNeuralIntegerArray.Create(4);

    Stats := EvaluateLAMBADA(NN, Examples);
    AssertEquals('two examples', 2, Stats.ItemCount);
    AssertEquals('one greedy hit (the id-11 word)', 1, Stats.CorrectCount);
    AssertEquals('accuracy = 0.5', 0.5, Stats.Accuracy, 1e-9);
    AssertEquals('final-word tokens scored', 2, Stats.TokenCount);

    // Perplexity: mean of -logprob over the two final tokens, analytic.
    // Ex0 final token 11 conditioned on PrevId=6; Ex1 final token 4 on PrevId=7.
    ExpNLL := (-ExpectedRowLogProb(6, csVocab - 1) - ExpectedRowLogProb(7, 4)) / 2;
    AssertEquals('last-word mean NLL is analytic', ExpNLL, Stats.MeanNLL, 1e-4);
    AssertEquals('perplexity = exp(meanNLL)', Exp(ExpNLL), Stats.Perplexity, 1e-3);
  finally
    NN.Free;
  end;
end;

procedure TTestNeuralNLPMetrics.TestLambadaReportFormatting;
var
  Stats: TNNetLambadaStats;
  Report: string;
begin
  Stats.Accuracy := 0.42;
  Stats.ItemCount := 50;
  Stats.CorrectCount := 21;
  Stats.MeanNLL := 1.5;
  Stats.Perplexity := Exp(1.5);
  Stats.TokenCount := 70;
  Report := LambadaReport(Stats);
  AssertTrue('header mentions LAMBADA', Pos('LAMBADA', Report) > 0);
  AssertTrue('has accuracy line', Pos('accuracy', Report) > 0);
  AssertTrue('reports last-word perplexity', Pos('PPL', Report) > 0);
end;

initialization
  RegisterTest(TTestNeuralNLPMetrics);
end.
