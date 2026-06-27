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
    function BuildTinyGQALM(ContextLen: integer): TNNet;      // grouped-query
    function BuildTinyLearnedPosLM(ContextLen: integer): TNNet; // GPT-2 wpe
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
    // DoLa toy LM with a PLANTED shallow-layer bias. Layers:
    //   0 Input(ContextLen,1,Vocab)
    //   1 FullConnectLinear(Vocab)  <- the premature candidate; zero weights,
    //        bias[TBias]=4, bias[TGood]=2 (input-independent logits).
    //   2 FullConnectLinear(Vocab)  <- head input; IDENTITY weights + a bias
    //        correction that lowers TBias and raises TGood, so the FINAL argmax
    //        is still TBias (greedy emits it) but the premature distribution is
    //        even more peaked on TBias.
    //   3 FullConnectLinear(Vocab)  <- LM head; IDENTITY.
    //   4 SoftMax
    // log p_final - log p_premature is then MAXIMISED at TGood, so DoLa flips
    // the completion from TBias to TGood while greedy never does.
    function BuildDoLaBiasedNet(ContextLen, Vocab, TBias, TGood: integer): TNNet;
    // DoLa toy LM with TWO lens-compatible premature layers (a SHALLOW and a
    // DEEP candidate) that peak on DIFFERENT tokens, so DoLa-low and DoLa-high
    // select different premature layers and contrast to different next tokens.
    function BuildDoLaTwoCandidateNet(ContextLen, Vocab,
      TShallow, TDeep, TFinal: integer): TNNet;
    // Locates the committed encoder-decoder fixtures (same probing as
    // TestNeuralPretrained.FixturePath: RunAll.sh runs from tests/).
    function FixturePath(const FileName: string): string;
    // Hand-rigged MARKOV-CHAIN two-net pair for the seq2seq beam tests: the
    // decoder is Input(DecSeqLen,1,1) tokens -> TNNetEmbedding(Vocab,Vocab)
    // whose output row r IS the next-token logits given the token at
    // position r (logits depend only on the previous token - trivially
    // causal), plus the second TNNetInput the two-net convention requires
    // (filled but unused). Each embedding row is set to ln(P(next|prev)),
    // so the routines' softmax recovers the exact transition probabilities
    // and every beam's cumulative log-prob is known in closed form.
    procedure BuildMarkovSeq2SeqPair(out Enc, Dec: TNNet;
      out Emb: TNNetEmbedding; Vocab, EncSeqLen, DecSeqLen: integer);
    // Sets Emb's row of PrevToken to ln of the given next-token probs.
    procedure SetMarkovRow(Emb: TNNetEmbedding; PrevToken: integer;
      const Probs: array of TNeuralFloat);
    // BIGRAM pair for the cache-forking beam test. Both nets are
    // Input(ContextLen,1,Vocab)->FullConnectLinear(Vocab)->SoftMax with the
    // SAME hand-set lookup table wired into the position-0 weight block (all
    // other positions zeroed). The one-hot-REVERSED cache-less forward and the
    // width-1 streamed forward therefore both compute softmax(Table[lastTok]),
    // so DecodeBeamSearchAll(Full) and DecodeBeamSearchCachedAll(Twin session)
    // must agree bit-for-bit. A deterministic seeded table gives a non-trivial
    // (multi-branch, EOS-bearing) beam tree.
    procedure BuildBigramBeamPair(out Full, Twin: TNNet; Vocab,
      FullContextLen: integer);
    // Full-re-encode beam-search reference for a CAUSAL streamable net (zero-pad
    // the prefix, read the last-real-position softmax row). Same algorithm and
    // ranking as DecodeBeamSearchAll, but driven by the SAME causal net the
    // cache-forking variant streams, so the two are bit-comparable on a genuine
    // KV-cache topology (RoPE attention).
    function CausalReEncodeBeamAll(Full: TNNet; const PromptToks: array of integer;
      MaxLen, BeamWidth: integer; LengthPenalty: TNeuralFloat
      ): TNNetDecodeResultArray;
    // Synthetic CROSS-ATTENTION encoder-decoder pair for the forced-prefix
    // cached-decode tests. The encoder is token-ids -> embedding -> a causal
    // transformer block -> (EncSeqLen,1,Dim) hidden states. The decoder is built
    // TWICE off the SAME weights: Enc and the wide DecFull share an encoder-
    // states input filled by the caller; DecCached is the width-1 incremental
    // twin (DecSeqLen=1) used by DecodeSeq2SeqForcedPrefixCached. CopyWeights
    // keeps the two decoders bit-identical so the cached vs full paths agree.
    procedure BuildCrossAttnSeq2SeqPair(out Enc, DecFull, DecCached: TNNet;
      EncSeqLen, DecFullSeqLen, Vocab: integer);
    // Naive O(L^2) reference: re-run the FULL decoder over the whole growing
    // (ForcedPrefix ++ generated) prefix every step, reading the last-real-row
    // argmax. EncStates is pre-computed and copied into DecFull's second input.
    function ForcedPrefixFullReDecode(DecFull: TNNet; EncStates: TNNetVolume;
      const ForcedPrefix: array of integer;
      EOSTokenId, MaxNewTokens: integer): TNeuralIntegerArray;
  published
    // Pure helper functions (no network needed).
    procedure TestLengthPenaltyAlphaZeroIsOne;
    procedure TestLengthPenaltyWuFormula;
    procedure TestSafeLogProbOfOneIsZero;
    procedure TestSafeLogProbClampsZero;
    procedure TestSafeLogProbMatchesLn;
    // End-to-end behaviour on a tiny net.
    procedure TestGreedyReturnsBoundedFiniteResult;
    procedure TestBatchGreedyMatchesPerPromptGreedy;
    procedure TestBatchGreedyStopStringsPerRowIndependent;
    procedure TestBatchGreedyEmptyAndSingle;
    procedure TestBeamSearchAllSortedDescending;
    procedure TestBeamSearchScoreNoWorseThanGreedy;
    // KV-cache (cache-forking) beam search must be bit-identical to the
    // re-encoding DecodeBeamSearchAll, full ranked beam, best-first.
    procedure TestBeamSearchCachedMatchesReEncodeBigram;
    procedure TestBeamSearchCachedMatchesReEncodeCausalReference;
    procedure TestBeamSearchCachedRejectsWideSession;
    // Diverse beam search (Hamming-diversity groups).
    procedure TestDiverseBeamLambdaZeroMatchesBeam;
    procedure TestDiverseBeamGroupsDifferInFirstToken;
    // Constrained beam search (force_words_ids).
    procedure TestConstrainedBeamNoForceMatchesBeam;
    procedure TestConstrainedBeamForcesPhrasePresence;
    procedure TestConstrainedBeamForcesMultiplePhrases;
    // Contrastive search (penalty_alpha) decoding.
    procedure TestContrastiveAlphaZeroMatchesGreedy;
    procedure TestContrastiveAlphaChangesSelection;
    // Token-level / KV-cache contrastive search (TNNetStreamingDecoder).
    procedure TestContrastiveStreamedAlphaZeroMatchesStreamedGreedy;
    procedure TestContrastiveStreamedDeterministic;
    procedure TestContrastiveStreamedAlphaChangesSelection;
    // DoLa (Decoding by Contrasting Layers).
    procedure TestDoLaAlphaZeroMatchesGreedy;
    procedure TestDoLaEmptyBucketMatchesGreedy;
    procedure TestDoLaFlipsShallowBiasedCompletion;
    procedure TestDoLaLowHighBucketsSelectDifferentLayers;
    // Early-exit / self-speculative decode (LayerSkip/CALM, single-model).
    procedure TestEarlyExitMatchesGreedyBitIdentical;
    procedure TestEarlyExitHighConfidenceMatchesGreedy;
    procedure TestEarlyExitAcceptCountsAreConsistent;
    // Best-of-N / self-consistency reranking.
    procedure TestBestOfNReturnsHighestScoringCandidate;
    procedure TestBestOfNExternalScorerPicksItsTop;
    procedure TestSelfConsistencyReturnsModalAnswer;
    // Prompt-lookup / n-gram speculative decoding (training-free, no draft net).
    procedure TestPromptLookupMatchesGreedyBitIdentical;
    procedure TestPromptLookupNoMatchDegradesToGreedy;
    // KV-cache incremental decode on TNNetScaledDotProductAttention.
    procedure TestKVCacheIncrementalMatchesFullForward;
    procedure TestKVCacheSlidingWindowMatchesFullForward;
    procedure TestKVCacheT5RelPosBiasMatchesFullForward;
    procedure TestKVCachePrefillThenStepMatchesFullForward;
    procedure TestKVCacheResetStartsFreshSequence;
    procedure TestKVCacheTruncateThenReappendMatchesFresh;
    procedure TestKVCacheDisabledPathUnchanged;
    procedure TestKVCacheInt8DriftWithinTolerance;
    // FlashAttention-style tiled online-softmax forward parity.
    procedure TestSDPATiledOnlineSoftmaxParity;
    // O(1)-per-step incremental decode on TNNetDiagonalSSM (persisted state).
    procedure TestSSMIncrementalMatchesFullForward;
    procedure TestSSMPrefillThenStepMatchesFullForward;
    procedure TestSSMResetStateStartsFreshSequence;
    procedure TestSSMDisabledPathUnchanged;
    // TNNetStreamingDecoder: the reusable incremental-decode session.
    procedure TestStreamingDecoderTransformerMatchesFullForward;
    procedure TestStreamingDecoderGQAMatchesFullForward;
    procedure TestStreamingDecoderLearnedPosMatchesFullForward;
    procedure TestStreamingDecoderSSMMatchesFullForward;
    procedure TestStreamingDecoderResetStartsFreshSequence;
    procedure TestStreamingDecoderTruncateToRollsBack;
    // Prefix-cache reuse / cache fork (Snapshot / RestoreSnapshot).
    procedure TestStreamingDecoderForkContinuationBitIdenticalTransformer;
    procedure TestStreamingDecoderForkContinuationBitIdenticalSSM;
    procedure TestStreamingDecoderSnapshotForksManyIndependentSessions;
    // StreamingLLM KV-cache eviction (attention sinks + rolling window).
    procedure TestStreamingEvictionWithinWindowBitIdenticalToUnbounded;
    procedure TestStreamingEvictionCapsCacheLengthPastWindow;
    // GenerateTokensStreamed / GenerateStringStreamed: streamed generation.
    procedure TestGenerateTokensStreamedTransformerMatchesFullGreedy;
    procedure TestGenerateTokensStreamedSSMMatchesFullGreedy;
    procedure TestGenerateTokensStreamedGreedySamplerMatchesNilPath;
    procedure TestGenerateTokensStreamedStopsAtEOS;
    procedure TestGenerateTokensStreamedRespectsCaps;
    procedure TestGenerateTokensStreamedRejectsWideSession;
    procedure TestGenerateStringStreamedMatchesTokenCore;
    // Modern sampling controls: penalties, stop sequences, stop strings.
    procedure TestGenerateTokensStreamedExtendedOverloadDefaultsMatchPlain;
    procedure TestGenerateTokensStreamedStopSequenceTrims;
    procedure TestGenerateTokensStreamedStopSequenceStaysInGeneratedRegion;
    procedure TestGenerateTokensStreamedPenaltyAvoidsRepetition;
    procedure TestDecodeGreedyStopStringTrimsAndFinishes;
    procedure TestGenerateStringStreamedStopStringTrims;
    // Logits-processor chain + temperature + generation config.
    procedure TestTemperatureProcessorProbabilityDomainMath;
    procedure TestProcessorChainOrderMatters;
    procedure TestNoRepeatNGramBansSeenBigramAndTrigram;
    procedure TestNoOpChainAndConfigBitIdenticalToPlainPath;
    procedure TestGenerateWithConfigMatchesHandAssembled;
    procedure TestTemperatureNearZeroWithSamplerMatchesGreedy;
    procedure TestStringConfigDefaultsMatchPlainWrapper;
    // Classifier-free guidance (CFG) processor.
    procedure TestCFGScaleOneMatchesPlainDecoding;
    procedure TestCFGScaleZeroIgnoresConditionalPrompt;
    procedure TestCFGRejectsInvalidArguments;
    procedure TestConfigCFGScaleOneMatchesNoCFGConfig;
    procedure TestConfigCFGScaleChangesOutput;
    procedure TestMakeUnconditionalTwinMatchesSourceLogits;
    // LLM output watermarking (Kirchenbauer green-list scheme).
    procedure TestWatermarkGreenListReproducibleFromKeyAndPrefix;
    procedure TestWatermarkDetectsWatermarkedAndRejectsRandom;
    procedure TestWatermarkProcessorBoostsGreenInProbabilityDomain;
    // Sequence-bias / bad-words processor (HF SequenceBiasLogitsProcessor).
    procedure TestSequenceBiasSingleTokenIsUnconditionalBias;
    procedure TestSequenceBiasMultiTokenFiresOnlyOnPrefixMatch;
    procedure TestSequenceBiasBannedWordNeverAppearsInGreedyOutput;
    // Constrained (structured) decoding: TNNetTokenConstraint and friends.
    procedure TestAllowedTokensConstraintMasksAndRenormalizes;
    procedure TestConstraintMaskFallbackLeavesRowUntouched;
    procedure TestGenerateTokensStreamedNilConstraintMatchesPlain;
    procedure TestGenerateTokensStreamedWhitelistOnlyEmitsAllowed;
    procedure TestForcedSequenceConstraintFollowsCandidate;
    // Token healing (guidance-style BPE boundary repair).
    procedure TestPrepareTokenHealingBuildsPrefixSetAndTrims;
    procedure TestTokenHealingChangesFirstTokenVsUnhealed;
    procedure TestTokenHealingMultiTokenRollback;
    // JSON automaton unit tests (no model).
    procedure TestJSONStateMachineObjectTransitions;
    procedure TestJSONStateMachineStringAndEscapes;
    procedure TestJSONStateMachineNumbers;
    procedure TestJSONStateMachineTopLevelCompletionAllowsOnlyWS;
    procedure TestJSONStateMachineBalancedStackClosing;
    procedure TestJSONConstraintValidatesMultiCharTokens;
    procedure TestJSONStateMachineFuzzRandomWalksStayValid;
    // GBNF grammar engine + constraint (no model).
    procedure TestGrammarArithmeticAcceptsOnlyValidExpressions;
    procedure TestGrammarMachineForkKeepsIndependentState;
    procedure TestGrammarConstraintMasksViolationsAndGatesEOS;
    procedure TestGrammarAlternationGroupingRepetitionOptional;
    procedure TestGrammarConstraintGreedyDrivenWalkStaysValid;
    procedure TestGrammarConstraintSampledDrivenWalkStaysValid;
    // Model-integration: constrained generation emits parseable JSON.
    procedure TestGenerateTokensStreamedJSONConstraintEmitsParseableJSON;
    procedure TestDecodeGreedyJSONConstraintEmitsParseableJSON;
    // JSON-Schema -> GBNF compiler (CompileJSONSchemaToGBNF) + the
    // CreateJSONSchemaConstraint convenience wrapper.
    procedure TestSchemaGBNFObjectRequiredAcceptsAndRejects;
    procedure TestSchemaGBNFEnumArrayRefRecursion;
    procedure TestSchemaGBNFConstraintGreedyToolCallIsByteLegal;
    // Seq2seq (encoder-decoder) generation on the committed T5/Marian pico
    // fixture pairs (BuildT5FromSafeTensors / BuildMarianFromSafeTensors).
    procedure TestDecodeSeq2SeqGreedyT5DeterministicEOSAndOracle;
    procedure TestDecodeSeq2SeqSampledTempZeroMatchesGreedy;
    procedure TestDecodeSeq2SeqGreedyMarianPairAndCapacity;
    procedure TestDecodeSeq2SeqRejectsInvalidArguments;
    // Seq2seq token-id beam search (DecodeSeq2SeqBeamSearch[All]) on the
    // hand-rigged Markov pair (exact closed-form log-probs) + T5 fixture.
    procedure TestSeq2SeqBeamWidth1MatchesGreedy;
    procedure TestSeq2SeqBeamBeatsGreedyFirstTokenTrap;
    procedure TestSeq2SeqBeamFinishedPoolLengthPenaltyFlipsRanking;
    procedure TestSeq2SeqBeamDeterministicCapsAndValidation;
    // Forced-prefix KV-cached seq2seq decode (DecodeSeq2SeqForcedPrefixCached):
    // a synthetic cross-attention encoder-decoder pair; the cached path must be
    // bit-identical to the naive full-re-decode loop, and a multi-token forced
    // prologue must be honored verbatim and steer the greedy continuation.
    procedure TestForcedPrefixCachedMatchesFullReDecode;
    procedure TestForcedPrefixSteersGreedyContinuation;
    procedure TestForcedPrefixCachedRejectsInvalidArguments;
    // Needle-in-a-haystack long-context eval harness (no model: deterministic
    // string stand-ins exercise insertion/grid/accuracy mechanics).
    procedure TestNeedleHaystackGridShapeMatchesAxes;
    procedure TestNeedleHaystackPerfectRetrieverIs100Percent;
    procedure TestNeedleHaystackNeverRetrieverIsZeroPercent;
    procedure TestNeedleHaystackInsertsNeedleAtRequestedDepth;
  end;

implementation

uses Math, fpjson, jsonparser, neuralpretrained;

// External Best-of-N scorer fixture: rewards LONGER completions (matches the
// TNNetSequenceScorer signature). Overrides the default logprob ranking so the
// test can assert the scorer, not logprob, decides the winner.
function ScoreByLength(const Prompt, Generated: string): TNeuralFloat;
begin
  Result := Length(Generated) * 1.0;
end;

// Self-consistency answer-extractor fixture: the answer is the FIRST character
// of the completion (matches the TNNetAnswerExtractor signature).
function FirstChar(const Generated: string): string;
begin
  if Generated = '' then Result := '' else Result := Generated[1];
end;

// ---- Needle-in-a-haystack deterministic stand-in callbacks (no model) ----
// Filler: deterministic, space-separated, never contains the needle answer.
function NeedleTestFiller(CharCount: integer; Data: Pointer): string;
const cWord = 'word ';
begin
  Result := '';
  while Length(Result) < CharCount do Result := Result + cWord;
  Result := Copy(Result, 1, CharCount);
end;

// Perfect retriever: echoes the needle answer ("42") only when the retrieval
// question is present in the prompt, simulating an oracle model. This exercises
// the full pipeline (insertion -> prompt assembly -> hit detection) at 100%.
function NeedlePerfectGen(const Prompt: string; Data: Pointer): string;
begin
  if Pos('magic number', LowerCase(Prompt)) > 0
  then Result := 'The answer is 42.'
  else Result := 'I do not know.';
end;

// Never retriever: never emits the answer -> 0% accuracy.
function NeedleNeverGen(const Prompt: string; Data: Pointer): string;
begin
  Result := 'I do not know.';
end;

// Depth-probe generator: returns whatever 8 characters follow the needle fact
// in the prompt, so the test can confirm the needle was actually spliced in
// (and the question still trails it). Reports the answer iff the splice is
// intact ("42" appears).
function NeedleDepthProbeGen(const Prompt: string; Data: Pointer): string;
var P: integer;
begin
  P := Pos('magic number is 42', LowerCase(Prompt));
  if P > 0 then Result := '42' else Result := 'missing';
end;

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

procedure TTestNeuralDecode.TestBatchGreedyMatchesPerPromptGreedy;
var
  NN: TNNet;
  Prompts: array[0..2] of string;
  Batch: TNNetDecodeResultArray;
  Single: TNNetDecodeResult;
  R: integer;
begin
  // HEADLINE GUARANTEE: batched greedy for N prompts is token-for-token
  // identical to running each prompt independently through DecodeGreedy.
  // The three prompts have DIFFERENT lengths so the (implicit) per-row
  // left-padding of the reversed one-hot encoding is exercised.
  RandSeed := 424242;
  NN := BuildTinyNet(4, 8);
  Prompts[0] := 'a';        // length 1
  Prompts[1] := 'abc';      // length 3
  Prompts[2] := 'abcdef';   // length 6 (> ContextLen, forces window truncation)
  try
    Batch := DecodeBatchGreedy(NN, Prompts, 10);
    AssertEquals('batch returns one result per prompt', 3, Length(Batch));
    for R := 0 to 2 do
    begin
      Single := DecodeGreedy(NN, Prompts[R], 10);
      AssertEquals('row '+IntToStr(R)+' text bit-identical',
        Single.Text, Batch[R].Text);
      AssertEquals('row '+IntToStr(R)+' finished bit-identical',
        Ord(Single.Finished), Ord(Batch[R].Finished));
      AssertEquals('row '+IntToStr(R)+' SumLogProb bit-identical',
        Single.SumLogProb, Batch[R].SumLogProb, 0.0);
      AssertEquals('row '+IntToStr(R)+' Score bit-identical',
        Single.Score, Batch[R].Score, 0.0);
    end;
  finally
    NN.Free;
  end;
end;

procedure TTestNeuralDecode.TestBatchGreedyStopStringsPerRowIndependent;
var
  NN: TNNet;
  Prompts: array[0..2] of string;
  Stops: array[0..0] of string;
  Batch: TNNetDecodeResultArray;
  Single: TNNetDecodeResult;
  R, MinLen: integer;
begin
  // PER-ROW STOP HANDLING: with a stop string the rows terminate at DIFFERENT
  // steps (the generated continuations differ by prompt window). Each row must
  // be byte-identical to the single-sample DecodeGreedy with the SAME stop
  // string -- proving an early-finishing row neither extends nor truncates the
  // others. Equivalence to the independent single-sample path IS the proof of
  // non-interference.
  RandSeed := 424242;
  NN := BuildTinyNet(4, 8);
  Prompts[0] := 'a';
  Prompts[1] := 'abc';
  Prompts[2] := 'abcdef';
  // Find a 1-char stop that actually fires somewhere so rows finish at
  // different lengths: use the first emitted char of the longest run.
  Single := DecodeGreedy(NN, Prompts[2], 12);
  if Length(Single.Text) >= 2 then Stops[0] := Single.Text[2]
  else Stops[0] := 'q';
  try
    Batch := DecodeBatchGreedy(NN, Prompts, 12, Stops);
    MinLen := MaxInt;
    for R := 0 to 2 do
    begin
      Single := DecodeGreedy(NN, Prompts[R], 12, Stops);
      AssertEquals('stop row '+IntToStr(R)+' text bit-identical',
        Single.Text, Batch[R].Text);
      AssertEquals('stop row '+IntToStr(R)+' finished bit-identical',
        Ord(Single.Finished), Ord(Batch[R].Finished));
      AssertEquals('stop row '+IntToStr(R)+' SumLogProb bit-identical',
        Single.SumLogProb, Batch[R].SumLogProb, 0.0);
      if Length(Batch[R].Text) < MinLen then MinLen := Length(Batch[R].Text);
    end;
    // Sanity: at least one row actually stopped early (shorter than the cap),
    // otherwise the per-row independence is not really exercised.
    AssertTrue('at least one row stopped before the cap', MinLen < 12);
  finally
    NN.Free;
  end;
end;

procedure TTestNeuralDecode.TestBatchGreedyEmptyAndSingle;
var
  NN: TNNet;
  Empty: array of string;
  One: array[0..0] of string;
  Batch: TNNetDecodeResultArray;
  Single: TNNetDecodeResult;
begin
  // Degenerate batch sizes: 0 prompts -> empty result; 1 prompt -> identical
  // to the single-sample call.
  RandSeed := 424242;
  NN := BuildTinyNet(4, 8);
  SetLength(Empty, 0);
  One[0] := 'ab';
  try
    Batch := DecodeBatchGreedy(NN, Empty, 6);
    AssertEquals('empty batch -> empty result', 0, Length(Batch));
    Batch := DecodeBatchGreedy(NN, One, 6);
    AssertEquals('single batch -> one result', 1, Length(Batch));
    Single := DecodeGreedy(NN, One[0], 6);
    AssertEquals('single batch text bit-identical', Single.Text, Batch[0].Text);
    AssertEquals('single batch SumLogProb bit-identical',
      Single.SumLogProb, Batch[0].SumLogProb, 0.0);
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

// Constant-logit net: zeroed weights -> input-independent logits set purely by
// the LM head bias, peaked on token PeakTok. Used for the diverse / constrained
// beam planted setups where the model "prefers" one specific continuation.
function BuildPeakedLogitNet(ContextLen, Vocab, PeakTok: integer): TNNet;
var
  L1: TNNetLayer;
  N: integer;
begin
  Result := TNNet.Create();
  Result.AddLayer(TNNetInput.Create(ContextLen, 1, Vocab));
  L1 := Result.AddLayer(TNNetFullConnectLinear.Create(Vocab));
  Result.AddLayer(TNNetSoftMax.Create());
  Result.InitWeights();
  for N := 0 to L1.Neurons.Count - 1 do
  begin
    L1.Neurons[N].Weights.Fill(0);
    // A mild monotone ramp so the runner-up ordering is deterministic, with a
    // strong peak on PeakTok (the token the unconstrained beam will emit).
    L1.Neurons[N].BiasWeight := 0.01 * N;
  end;
  L1.Neurons[PeakTok].BiasWeight := 8.0;
  L1.MulWeights(1.0);
end;

procedure TTestNeuralDecode.TestDiverseBeamLambdaZeroMatchesBeam;
var
  NN: TNNet;
  Beam, Dvb: TNNetDecodeResultArray;
  I: integer;
begin
  // Degrade-to-beam invariant: one group (NumGroups=1) with Diversity=0 is
  // BIT-FOR-BIT ordinary beam search, element for element.
  RandSeed := 424242;
  NN := BuildTinyNet(4, 8);
  try
    Beam := DecodeBeamSearchAll(NN, 'ab', 6, 4, 0.0);
    Dvb := DecodeDiverseBeamSearchAll(NN, 'ab', 6, 4, 1, 0.0, 0.0);
    AssertEquals('same beam count', Length(Beam), Length(Dvb));
    for I := 0 to High(Beam) do
    begin
      AssertEquals('text matches', Beam[I].Text, Dvb[I].Text);
      AssertEquals('score matches', Beam[I].Score, Dvb[I].Score, 1e-7);
      AssertEquals('sumlogprob matches', Beam[I].SumLogProb,
        Dvb[I].SumLogProb, 1e-7);
      AssertEquals('finished matches', Ord(Beam[I].Finished),
        Ord(Dvb[I].Finished));
    end;
  finally
    NN.Free;
  end;
end;

procedure TTestNeuralDecode.TestDiverseBeamGroupsDifferInFirstToken;
var
  NN: TNNet;
  D0, D1: TNNetDecodeResultArray;
  I, NDistinct0, NDistinct1: integer;
  Seen: array[0..255] of boolean;
  C: char;
begin
  // With Diversity=0 the two groups, each width 1, both follow the single most
  // probable first token (peaked on token 5) -> their first chars coincide.
  // A strong Diversity penalty pushes the SECOND group off token 5 onto a
  // different first token, so the kept beams now start with >1 distinct char.
  RandSeed := 424242;
  NN := BuildPeakedLogitNet(4, 8, 5);
  try
    D0 := DecodeDiverseBeamSearchAll(NN, 'ab', 4, 2, 2, 0.0, 0.0);
    D1 := DecodeDiverseBeamSearchAll(NN, 'ab', 4, 2, 2, 50.0, 0.0);

    for I := 0 to 255 do Seen[I] := False;
    NDistinct0 := 0;
    for I := 0 to High(D0) do
      if Length(D0[I].Text) >= 1 then
      begin
        C := D0[I].Text[1];
        if not Seen[Ord(C)] then begin Seen[Ord(C)] := True; Inc(NDistinct0); end;
      end;

    for I := 0 to 255 do Seen[I] := False;
    NDistinct1 := 0;
    for I := 0 to High(D1) do
      if Length(D1[I].Text) >= 1 then
      begin
        C := D1[I].Text[1];
        if not Seen[Ord(C)] then begin Seen[Ord(C)] := True; Inc(NDistinct1); end;
      end;

    AssertEquals('lambda=0 groups share the single best first token',
      1, NDistinct0);
    AssertTrue('lambda>0 groups produce different first tokens',
      NDistinct1 > NDistinct0);
  finally
    NN.Free;
  end;
end;

procedure TTestNeuralDecode.TestConstrainedBeamNoForceMatchesBeam;
var
  NN: TNNet;
  Beam, Con: TNNetDecodeResultArray;
  NoForce: array of string;
  I: integer;
begin
  // With no forced phrases the constrained routine is bit-for-bit ordinary beam.
  RandSeed := 424242;
  SetLength(NoForce, 0);
  NN := BuildTinyNet(4, 8);
  try
    Beam := DecodeBeamSearchAll(NN, 'ab', 6, 4, 0.0);
    Con := DecodeConstrainedBeamSearchAll(NN, 'ab', 6, 4, NoForce, 0.0);
    AssertEquals('same beam count', Length(Beam), Length(Con));
    for I := 0 to High(Beam) do
    begin
      AssertEquals('text matches', Beam[I].Text, Con[I].Text);
      AssertEquals('score matches', Beam[I].Score, Con[I].Score, 1e-7);
    end;
  finally
    NN.Free;
  end;
end;

procedure TTestNeuralDecode.TestConstrainedBeamForcesPhrasePresence;
var
  NN: TNNet;
  Uncon, Con: TNNetDecodeResult;
  Force, NoForce: array of string;
  Phrase: string;
begin
  // Planted setup: the net is peaked on token 'Z'=Ord 90; an unconstrained beam
  // emits an all-'Z' run and would NEVER produce the phrase 'xyz'. The
  // constrained beam MUST contain 'xyz' somewhere in its output.
  RandSeed := 424242;
  Phrase := 'xyz';
  SetLength(NoForce, 0);
  SetLength(Force, 1);
  Force[0] := Phrase;
  NN := BuildPeakedLogitNet(8, 128, Ord('Z'));
  try
    Uncon := DecodeBeamSearch(NN, 'ab', 10, 4, 0.0);
    AssertEquals('unconstrained beam never emits the phrase',
      0, Pos(Phrase, Uncon.Text));
    Con := DecodeConstrainedBeamSearch(NN, 'ab', 10, 4, Force, 0.0);
    AssertTrue('constrained beam emits the forced phrase',
      Pos(Phrase, Con.Text) > 0);
  finally
    NN.Free;
  end;
end;

procedure TTestNeuralDecode.TestConstrainedBeamForcesMultiplePhrases;
var
  NN: TNNet;
  Con: TNNetDecodeResult;
  Force: array of string;
begin
  // Two disjoint forced phrases must BOTH appear in the single best output.
  RandSeed := 424242;
  SetLength(Force, 2);
  Force[0] := 'xy';
  Force[1] := 'qrs';
  NN := BuildPeakedLogitNet(12, 128, Ord('Z'));
  try
    Con := DecodeConstrainedBeamSearch(NN, 'ab', 16, 4, Force, 0.0);
    AssertTrue('first forced phrase present', Pos('xy', Con.Text) > 0);
    AssertTrue('second forced phrase present', Pos('qrs', Con.Text) > 0);
  finally
    NN.Free;
  end;
end;

procedure TTestNeuralDecode.TestContrastiveAlphaZeroMatchesGreedy;
var
  NN: TNNet;
  G, C: TNNetDecodeResult;
  NoStops: array of string;
begin
  // Degrade-to-greedy invariant: PenaltyAlpha=0 keeps only (1-alpha)*p(v), so
  // the top-probability candidate (always the global argmax) wins every step.
  // Output must be bit-identical to plain greedy argmax over the same net.
  RandSeed := 424242;
  SetLength(NoStops, 0);
  NN := BuildTinyNet(4, 8);
  try
    G := DecodeGreedy(NN, 'ab', 8);
    C := DecodeContrastiveSearch(NN, 'ab', 8, 4, 0.0, NoStops);
    AssertEquals('alpha=0 contrastive text == greedy text', G.Text, C.Text);
    AssertEquals('alpha=0 contrastive finished == greedy',
      Ord(G.Finished), Ord(C.Finished));
  finally
    NN.Free;
  end;
end;

procedure TTestNeuralDecode.TestContrastiveAlphaChangesSelection;
var
  NN: TNNet;
  G, C1, C2: TNNetDecodeResult;
  NoStops: array of string;
begin
  // With a non-zero degeneration penalty the re-rank must (a) be deterministic
  // and (b) be able to steer away from greedy's choice. We pin a seed so the
  // tiny net's random head gives top-k candidates with distinct hidden states;
  // alpha=1 (pure similarity penalty) then re-orders them, changing the output.
  RandSeed := 1234567;
  SetLength(NoStops, 0);
  NN := BuildTinyNet(4, 8);
  try
    G := DecodeGreedy(NN, 'ab', 8);
    C1 := DecodeContrastiveSearch(NN, 'ab', 8, 6, 1.0, NoStops);
    C2 := DecodeContrastiveSearch(NN, 'ab', 8, 6, 1.0, NoStops);
    AssertEquals('contrastive is deterministic', C1.Text, C2.Text);
    AssertTrue('contrastive result finite/bounded',
      (Length(C1.Text) <= 8) and not IsNan(C1.Score) and
      not IsInfinite(C1.Score));
    AssertTrue('alpha=1 penalty changes the greedy selection',
      C1.Text <> G.Text);
  finally
    NN.Free;
  end;
end;

procedure TTestNeuralDecode.TestContrastiveStreamedAlphaZeroMatchesStreamedGreedy;
const
  SeqLen = 10;
  PromptLen = 3;
var
  Full, Twin: TNNet;
  Session: TNNetStreamingDecoder;
  GreedyToks, CSToks: TNeuralIntegerArray;
  GreedyLen, CSLen, T: integer;
  NoStops: TNNetTokenSequences;
begin
  // Degrade-to-greedy invariant for the KV-cache variant: PenaltyAlpha=0 keeps
  // only (1-alpha)*p(v), so the top-probability candidate (= streamed greedy
  // argmax) wins every step. Output must be BIT-FOR-BIT the streamed greedy
  // GenerateTokensStreamed (nil sampler) on the same session.
  RandSeed := 424242;
  SetLength(NoStops, 0);
  Full := BuildTinyCausalLM(SeqLen);
  Twin := BuildTinyCausalLM(1);
  Session := nil;
  try
    Twin.CopyWeights(Full);
    Session := TNNetStreamingDecoder.Create(Twin, SeqLen);
    SetLength(GreedyToks, PromptLen);
    SetLength(CSToks, PromptLen);
    GreedyToks[0] := 3; GreedyToks[1] := 7; GreedyToks[2] := 5;
    for T := 0 to PromptLen - 1 do CSToks[T] := GreedyToks[T];

    GreedyLen := GenerateTokensStreamed(Session, GreedyToks, PromptLen,
      SeqLen - PromptLen, SeqLen);
    CSLen := DecodeContrastiveSearchStreamed(Session, CSToks, PromptLen,
      SeqLen - PromptLen, SeqLen, {TopK=}4, {alpha=}0.0, NoStops);

    AssertEquals('alpha=0 streamed CS length == streamed greedy length',
      GreedyLen, CSLen);
    for T := PromptLen to CSLen - 1 do
      AssertEquals('alpha=0 streamed CS token at ' + IntToStr(T),
        GreedyToks[T], CSToks[T]);
  finally
    Session.Free;
    Twin.Free;
    Full.Free;
  end;
end;

procedure TTestNeuralDecode.TestContrastiveStreamedDeterministic;
const
  SeqLen = 10;
  PromptLen = 3;
var
  Full, Twin: TNNet;
  Session: TNNetStreamingDecoder;
  A, B: TNeuralIntegerArray;
  LenA, LenB, T: integer;
  NoStops: TNNetTokenSequences;
begin
  // Same session, same prompt, twice -> identical output (no hidden RNG / state
  // leak across the fork/rollback candidate evaluation).
  RandSeed := 1234567;
  SetLength(NoStops, 0);
  Full := BuildTinyCausalLM(SeqLen);
  Twin := BuildTinyCausalLM(1);
  Session := nil;
  try
    Twin.CopyWeights(Full);
    Session := TNNetStreamingDecoder.Create(Twin, SeqLen);
    SetLength(A, PromptLen);
    SetLength(B, PromptLen);
    A[0] := 4; A[1] := 2; A[2] := 9;
    for T := 0 to PromptLen - 1 do B[T] := A[T];

    LenA := DecodeContrastiveSearchStreamed(Session, A, PromptLen,
      SeqLen - PromptLen, SeqLen, {TopK=}6, {alpha=}0.6, NoStops);
    LenB := DecodeContrastiveSearchStreamed(Session, B, PromptLen,
      SeqLen - PromptLen, SeqLen, {TopK=}6, {alpha=}0.6, NoStops);

    AssertEquals('streamed CS is deterministic (length)', LenA, LenB);
    for T := PromptLen to LenA - 1 do
      AssertEquals('streamed CS deterministic token at ' + IntToStr(T),
        A[T], B[T]);
  finally
    Session.Free;
    Twin.Free;
    Full.Free;
  end;
end;

procedure TTestNeuralDecode.TestContrastiveStreamedAlphaChangesSelection;
const
  SeqLen = 10;
  PromptLen = 3;
var
  Full, Twin: TNNet;
  Session: TNNetStreamingDecoder;
  GreedyToks, CSToks: TNeuralIntegerArray;
  GreedyLen, CSLen, T: integer;
  Differs: boolean;
  NoStops: TNNetTokenSequences;
begin
  // A non-zero degeneration penalty must be able to steer away from the greedy
  // choice (the contrast path actually does something on the cached state).
  RandSeed := 99999;
  SetLength(NoStops, 0);
  Full := BuildTinyCausalLM(SeqLen);
  Twin := BuildTinyCausalLM(1);
  Session := nil;
  try
    Twin.CopyWeights(Full);
    Session := TNNetStreamingDecoder.Create(Twin, SeqLen);
    SetLength(GreedyToks, PromptLen);
    SetLength(CSToks, PromptLen);
    GreedyToks[0] := 6; GreedyToks[1] := 1; GreedyToks[2] := 8;
    for T := 0 to PromptLen - 1 do CSToks[T] := GreedyToks[T];

    GreedyLen := GenerateTokensStreamed(Session, GreedyToks, PromptLen,
      SeqLen - PromptLen, SeqLen);
    CSLen := DecodeContrastiveSearchStreamed(Session, CSToks, PromptLen,
      SeqLen - PromptLen, SeqLen, {TopK=}8, {alpha=}1.0, NoStops);

    Differs := (GreedyLen <> CSLen);
    if not Differs then
      for T := PromptLen to CSLen - 1 do
        if CSToks[T] <> GreedyToks[T] then Differs := True;
    AssertTrue('alpha=1 streamed penalty changes the greedy selection',
      Differs);
  finally
    Session.Free;
    Twin.Free;
    Full.Free;
  end;
end;

function TTestNeuralDecode.BuildDoLaBiasedNet(
  ContextLen, Vocab, TBias, TGood: integer): TNNet;
var
  L1, L2, L3: TNNetLayer;
  N: integer;
begin
  Result := TNNet.Create();
  Result.AddLayer(TNNetInput.Create(ContextLen, 1, Vocab));
  L1 := Result.AddLayer(TNNetFullConnectLinear.Create(Vocab)); // premature
  L2 := Result.AddLayer(TNNetFullConnectLinear.Create(Vocab)); // head input
  L3 := Result.AddLayer(TNNetFullConnectLinear.Create(Vocab)); // LM head
  Result.AddLayer(TNNetSoftMax.Create());
  Result.InitWeights();
  // Layer 1: zero weights -> input-independent constant logits via bias only.
  for N := 0 to L1.Neurons.Count - 1 do
  begin
    L1.Neurons[N].Weights.Fill(0);
    L1.Neurons[N].BiasWeight := 0;
  end;
  L1.Neurons[TBias].BiasWeight := 4;
  L1.Neurons[TGood].BiasWeight := 2;
  L1.MulWeights(1.0);
  // Layer 2: identity weights + bias correction (lower TBias, raise TGood) so
  // FINAL argmax stays TBias but premature is even more peaked on TBias.
  for N := 0 to L2.Neurons.Count - 1 do
  begin
    L2.Neurons[N].Weights.Fill(0);
    L2.Neurons[N].Weights.FData[N] := 1; // identity
    L2.Neurons[N].BiasWeight := 0;
  end;
  L2.Neurons[TBias].BiasWeight := -1.0;  // L2[TBias] = 4 - 1.0 = 3.0
  L2.Neurons[TGood].BiasWeight := 0.8;   // L2[TGood] = 2 + 0.8 = 2.8  (< 3.0)
  L2.MulWeights(1.0);
  // Layer 3: identity LM head.
  for N := 0 to L3.Neurons.Count - 1 do
  begin
    L3.Neurons[N].Weights.Fill(0);
    L3.Neurons[N].Weights.FData[N] := 1;
    L3.Neurons[N].BiasWeight := 0;
  end;
  L3.MulWeights(1.0);
end;

procedure TTestNeuralDecode.TestDoLaAlphaZeroMatchesGreedy;
var
  NN: TNNet;
  G, D: TNNetDecodeResult;
  NoStops: array of string;
begin
  // Degrade-to-greedy invariant: Alpha<=0 disables the layer contrast, so DoLa
  // must reproduce plain greedy argmax bit-for-bit.
  RandSeed := 424242;
  SetLength(NoStops, 0);
  NN := BuildTinyNet(4, 8);
  try
    G := DecodeGreedy(NN, 'ab', 8);
    D := DecodeDoLa(NN, 'ab', 8, 0.0, NoStops);
    AssertEquals('alpha=0 DoLa text == greedy text', G.Text, D.Text);
    AssertEquals('alpha=0 DoLa finished == greedy',
      Ord(G.Finished), Ord(D.Finished));
    AssertEquals('alpha=0 DoLa sumlogprob == greedy', G.SumLogProb,
      D.SumLogProb, 1e-6);
  finally
    NN.Free;
  end;
end;

procedure TTestNeuralDecode.TestDoLaEmptyBucketMatchesGreedy;
var
  NN: TNNet;
  G, D: TNNetDecodeResult;
  NoStops: array of string;
begin
  // BuildTinyNet has no earlier layer the size of the head input (FC ReLU(16)
  // feeds an FCLinear(Vocab) head), so the candidate bucket is EMPTY and DoLa
  // must fall back to greedy even at a non-zero Alpha.
  RandSeed := 424242;
  SetLength(NoStops, 0);
  NN := BuildTinyNet(4, 8);
  try
    G := DecodeGreedy(NN, 'ab', 8);
    D := DecodeDoLa(NN, 'ab', 8, 0.1, NoStops);
    AssertEquals('empty-bucket DoLa text == greedy text', G.Text, D.Text);
    AssertEquals('empty-bucket DoLa sumlogprob == greedy', G.SumLogProb,
      D.SumLogProb, 1e-6);
  finally
    NN.Free;
  end;
end;

procedure TTestNeuralDecode.TestDoLaFlipsShallowBiasedCompletion;
const
  TBias = 5;
  TGood = 6;
var
  NN: TNNet;
  G, D: TNNetDecodeResult;
  NoStops: array of string;
  I: integer;
begin
  // Planted shallow-layer bias: greedy emits the biased token TBias every step,
  // but the layer contrast (log p_final - log p_premature) is maximised at
  // TGood, so DoLa flips the whole completion to TGood.
  SetLength(NoStops, 0);
  NN := BuildDoLaBiasedNet(4, 8, TBias, TGood);
  try
    G := DecodeGreedy(NN, 'ab', 5);
    D := DecodeDoLa(NN, 'ab', 5, 0.1, NoStops);
    AssertTrue('greedy emitted at least one token', Length(G.Text) >= 1);
    AssertEquals('DoLa and greedy same length', Length(G.Text), Length(D.Text));
    for I := 1 to Length(G.Text) do
    begin
      AssertEquals('greedy emits the biased token', Chr(TBias), G.Text[I]);
      AssertEquals('DoLa flips to the good token', Chr(TGood), D.Text[I]);
    end;
    AssertTrue('DoLa actually changed the completion', D.Text <> G.Text);
  finally
    NN.Free;
  end;
end;

function TTestNeuralDecode.BuildDoLaTwoCandidateNet(ContextLen, Vocab,
  TShallow, TDeep, TFinal: integer): TNNet;
var
  L1, L2, L3, L4: TNNetLayer;
  N: integer;
begin
  Result := TNNet.Create();
  Result.AddLayer(TNNetInput.Create(ContextLen, 1, Vocab));
  L1 := Result.AddLayer(TNNetFullConnectLinear.Create(Vocab)); // shallow premature
  L2 := Result.AddLayer(TNNetFullConnectLinear.Create(Vocab)); // deep premature
  L3 := Result.AddLayer(TNNetFullConnectLinear.Create(Vocab)); // head input
  L4 := Result.AddLayer(TNNetFullConnectLinear.Create(Vocab)); // LM head
  Result.AddLayer(TNNetSoftMax.Create());
  Result.InitWeights();
  // L1 (shallow lens): zero weights -> input-independent logits peaked on
  // TShallow. lens_low(t) = softmax over [.. peak at TShallow ..].
  for N := 0 to L1.Neurons.Count - 1 do
  begin
    L1.Neurons[N].Weights.Fill(0);
    L1.Neurons[N].BiasWeight := 0;
  end;
  L1.Neurons[TShallow].BiasWeight := 5;
  L1.MulWeights(1.0);
  // L2 (deep lens): zero weights -> input-independent logits peaked on TDeep.
  for N := 0 to L2.Neurons.Count - 1 do
  begin
    L2.Neurons[N].Weights.Fill(0);
    L2.Neurons[N].BiasWeight := 0;
  end;
  L2.Neurons[TDeep].BiasWeight := 5;
  L2.MulWeights(1.0);
  // L3 (head input): IDENTITY from L2 (logit 5 at TDeep) plus a bias so the
  // FINAL logits are EQUAL-high at BOTH TShallow and TDeep (the plausible set),
  // making the contrast term log p_final - log p_lens decide the winner.
  // Final logits: TShallow=6, TDeep=6 (L2 had 5 at TDeep so +1 there, +6 at
  // TShallow which L2 left at 0). TFinal is left low here on purpose - the
  // contrast, not the raw argmax, drives the choice.
  for N := 0 to L3.Neurons.Count - 1 do
  begin
    L3.Neurons[N].Weights.Fill(0);
    L3.Neurons[N].Weights.FData[N] := 1; // identity from L2
    L3.Neurons[N].BiasWeight := 0;
  end;
  L3.Neurons[TShallow].BiasWeight := 6;  // final[TShallow] = 0 + 6 = 6
  L3.Neurons[TDeep].BiasWeight := 1;     // final[TDeep]    = 5 + 1 = 6
  L3.MulWeights(1.0);
  // L4 (LM head): identity.
  for N := 0 to L4.Neurons.Count - 1 do
  begin
    L4.Neurons[N].Weights.Fill(0);
    L4.Neurons[N].Weights.FData[N] := 1;
    L4.Neurons[N].BiasWeight := 0;
  end;
  L4.MulWeights(1.0);
  if TFinal < 0 then; // TFinal unused as a separate peak; kept for documentation
end;

procedure TTestNeuralDecode.TestDoLaLowHighBucketsSelectDifferentLayers;
const
  TShallow = 3;
  TDeep    = 4;
var
  NN: TNNet;
  DLow, DHigh: TNNetDecodeResult;
  NoStops: array of string;
  I: integer;
begin
  // Two lens-compatible premature layers (L1 shallow peaks on TShallow, L2 deep
  // peaks on TDeep). The plausible set is {TShallow, TDeep} (equal-high final
  // logits). The contrast argmax(final_logit - lens_logit) then:
  //   * DoLa-LOW (lens = shallow, peaks TShallow) suppresses TShallow -> TDeep.
  //   * DoLa-HIGH (lens = deep,  peaks TDeep)     suppresses TDeep    -> TShallow.
  // So restricting the candidate bucket to the low vs high half flips the
  // emitted token EVERY step - proof the bucket restriction changes which
  // premature layer is chosen and thus the contrasted logits.
  SetLength(NoStops, 0);
  NN := BuildDoLaTwoCandidateNet(4, 8, TShallow, TDeep, -1);
  try
    DLow  := DecodeDoLa(NN, 'ab', 4, 0.1, NoStops, dlbLow);
    DHigh := DecodeDoLa(NN, 'ab', 4, 0.1, NoStops, dlbHigh);
    AssertEquals('DoLa-low and DoLa-high same length',
      Length(DLow.Text), Length(DHigh.Text));
    AssertTrue('DoLa-low produced tokens', Length(DLow.Text) >= 1);
    AssertTrue('low/high buckets pick different layers -> different text',
      DLow.Text <> DHigh.Text);
    for I := 1 to Length(DLow.Text) do
    begin
      AssertEquals('DoLa-low emits TDeep (shallow lens suppresses TShallow)',
        Chr(TDeep), DLow.Text[I]);
      AssertEquals('DoLa-high emits TShallow (deep lens suppresses TDeep)',
        Chr(TShallow), DHigh.Text[I]);
    end;
  finally
    NN.Free;
  end;
end;

procedure TTestNeuralDecode.TestEarlyExitMatchesGreedyBitIdentical;
var
  NN: TNNet;
  G, S: TNNetDecodeResult;
  Stats: TNNetEarlyExitStats;
  NoStops: array of string;
begin
  // CORRECTNESS GUARANTEE of exact-greedy self-speculative decode: every
  // emitted token is the full-depth argmax, so the accepted sequence MUST equal
  // plain greedy DecodeGreedy bit-for-bit, regardless of the early-exit draft.
  // Confidence=0 means the draft is consulted EVERY step (maximum exercise of
  // the splice path); the output still has to match greedy exactly.
  RandSeed := 424242;
  SetLength(NoStops, 0);
  NN := BuildDoLaTwoCandidateNet(4, 8, 5, 6, 7);
  try
    G := DecodeGreedy(NN, 'ab', 8);
    // ExitLayer = 1 (the shallow intermediate), auto head-start detection.
    S := DecodeEarlyExitSelfSpeculative(NN, 'ab', 8, Stats, NoStops,
      {ExitLayer=}1, {Confidence=}0.0);
    AssertEquals('early-exit text == greedy text (bit-identical)',
      G.Text, S.Text);
    AssertEquals('early-exit finished == greedy', Ord(G.Finished),
      Ord(S.Finished));
    AssertEquals('early-exit sumlogprob == greedy', G.SumLogProb,
      S.SumLogProb, 1e-6);
    AssertEquals('one full-depth forward per emitted token',
      Length(G.Text), Stats.Steps);
  finally
    NN.Free;
  end;
end;

procedure TTestNeuralDecode.TestEarlyExitHighConfidenceMatchesGreedy;
var
  NN: TNNet;
  G, S: TNNetDecodeResult;
  Stats: TNNetEarlyExitStats;
  NoStops: array of string;
begin
  // Confidence > 1 disables the draft path entirely (pure greedy fallback). The
  // output must STILL be bit-identical and NO drafts may be proposed.
  RandSeed := 424242;
  SetLength(NoStops, 0);
  NN := BuildDoLaTwoCandidateNet(4, 8, 5, 6, 7);
  try
    G := DecodeGreedy(NN, 'ab', 8);
    S := DecodeEarlyExitSelfSpeculative(NN, 'ab', 8, Stats, NoStops,
      {ExitLayer=}1, {Confidence=}2.0);
    AssertEquals('draft-disabled text == greedy text', G.Text, S.Text);
    AssertEquals('draft-disabled sumlogprob == greedy', G.SumLogProb,
      S.SumLogProb, 1e-6);
    AssertEquals('no draft proposed when Confidence>1', 0, Stats.DraftProposals);
  finally
    NN.Free;
  end;
end;

procedure TTestNeuralDecode.TestEarlyExitAcceptCountsAreConsistent;
var
  NN: TNNet;
  S: TNNetDecodeResult;
  Stats: TNNetEarlyExitStats;
  NoStops: array of string;
begin
  // Accounting invariant: Accepted + Rejected = DraftProposals <= Steps, and the
  // acceptance rate matches Accepted/DraftProposals. (Output identity is covered
  // by the dedicated test above; here we exercise the fallback bookkeeping.)
  RandSeed := 424242;
  SetLength(NoStops, 0);
  NN := BuildDoLaTwoCandidateNet(4, 8, 5, 6, 7);
  try
    S := DecodeEarlyExitSelfSpeculative(NN, 'ab', 8, Stats, NoStops,
      {ExitLayer=}1, {Confidence=}0.0);
    AssertTrue('at least one step ran', Stats.Steps >= 1);
    AssertEquals('accepted + rejected == proposals',
      Stats.DraftProposals, Stats.Accepted + Stats.Rejected);
    AssertTrue('proposals never exceed steps',
      Stats.DraftProposals <= Stats.Steps);
    if Stats.DraftProposals > 0 then
      AssertEquals('acceptance rate == accepted/proposals',
        Stats.Accepted / Stats.DraftProposals, Stats.AcceptanceRate, 1e-6);
    AssertTrue('emitted length does not exceed steps',
      Length(S.Text) <= Stats.Steps);
  finally
    NN.Free;
  end;
end;

procedure TTestNeuralDecode.TestBestOfNReturnsHighestScoringCandidate;
var
  NN: TNNet;
  Cands: TNNetDecodeResultArray;
  Best: TNNetDecodeResult;
  Sampler: TNNetSamplerTopK;
  NoStops: array of string;
  I, MaxIdx: integer;
begin
  // Best-of-N must return the candidate with the highest length-normalized
  // score among the SAME N draws (re-run with the same seed to reproduce them).
  SetLength(NoStops, 0);
  NN := BuildTinyNet(4, 8);
  Sampler := TNNetSamplerTopK.Create(4);
  try
    RandSeed := 7777;
    Cands := SampleNCompletions(NN, 'ab', 8, 5, Sampler, 0.0, NoStops, nil);
    MaxIdx := 0;
    for I := 1 to High(Cands) do
      if Cands[I].Score > Cands[MaxIdx].Score then MaxIdx := I;
    RandSeed := 7777;
    Best := DecodeBestOfN(NN, 'ab', 8, 5, Sampler, 0.0, NoStops, nil);
    AssertEquals('best-of-N returns the modal max-score text',
      Cands[MaxIdx].Text, Best.Text);
    AssertEquals('best-of-N score == that candidate score',
      Cands[MaxIdx].Score, Best.Score, 1e-6);
  finally
    Sampler.Free;
    NN.Free;
  end;
end;

procedure TTestNeuralDecode.TestBestOfNExternalScorerPicksItsTop;
var
  NN: TNNet;
  Cands: TNNetDecodeResultArray;
  Best: TNNetDecodeResult;
  Sampler: TNNetSamplerTopK;
  NoStops: array of string;
  I, WantIdx: integer;
begin
  // The external scorer overrides logprob ranking: ScoreByLength rewards LONGER
  // completions, so best-of-N must return the longest of the N draws.
  SetLength(NoStops, 0);
  NN := BuildTinyNet(4, 8);
  Sampler := TNNetSamplerTopK.Create(4);
  try
    RandSeed := 31337;
    Cands := SampleNCompletions(NN, 'ab', 8, 6, Sampler, 0.0, NoStops,
      @ScoreByLength);
    WantIdx := 0;
    for I := 1 to High(Cands) do
      if Cands[I].Score > Cands[WantIdx].Score then WantIdx := I;
    RandSeed := 31337;
    Best := DecodeBestOfN(NN, 'ab', 8, 6, Sampler, 0.0, NoStops, @ScoreByLength);
    AssertEquals('external scorer picks its top-ranked candidate',
      Cands[WantIdx].Text, Best.Text);
    AssertEquals('returned score is the scorer value',
      Length(Best.Text) * 1.0, Best.Score, 1e-6);
  finally
    Sampler.Free;
    NN.Free;
  end;
end;

procedure TTestNeuralDecode.TestSelfConsistencyReturnsModalAnswer;
var
  NN: TNNet;
  Sampler: TNNetSamplerTopK;
  NoStops: array of string;
  Cands: TNNetDecodeResultArray;
  Answer, A: string;
  Distinct: array of string;
  Counts: array of integer;
  I, J, K, NumDistinct, BestCount, BestIdx: integer;
  Found: boolean;
begin
  // Self-consistency = majority vote over extracted answers. We compute the
  // expected modal answer independently from the SAME N draws (FirstChar
  // extractor) and assert DecodeSelfConsistency agrees, deterministically.
  SetLength(NoStops, 0);
  NN := BuildTinyNet(4, 8);
  Sampler := TNNetSamplerTopK.Create(4);
  try
    RandSeed := 9001;
    Cands := SampleNCompletions(NN, 'ab', 8, 9, Sampler, 0.0, NoStops, nil);
    SetLength(Distinct, 0);
    SetLength(Counts, 0);
    NumDistinct := 0;
    for I := 0 to High(Cands) do
    begin
      A := FirstChar(Cands[I].Text);
      if A = '' then Continue;
      Found := False;
      for J := 0 to NumDistinct - 1 do
        if Distinct[J] = A then begin Inc(Counts[J]); Found := True; Break; end;
      if not Found then
      begin
        SetLength(Distinct, NumDistinct + 1);
        SetLength(Counts, NumDistinct + 1);
        Distinct[NumDistinct] := A; Counts[NumDistinct] := 1;
        Inc(NumDistinct);
      end;
    end;
    BestIdx := 0; BestCount := -1;
    for K := 0 to NumDistinct - 1 do
      if Counts[K] > BestCount then begin BestCount := Counts[K]; BestIdx := K; end;
    RandSeed := 9001;
    Answer := DecodeSelfConsistency(NN, 'ab', 8, 9, Sampler, @FirstChar, NoStops);
    AssertTrue('there is at least one extractable answer', NumDistinct >= 1);
    AssertEquals('self-consistency returns the modal answer',
      Distinct[BestIdx], Answer);
  finally
    Sampler.Free;
    NN.Free;
  end;
end;

procedure TTestNeuralDecode.TestPromptLookupMatchesGreedyBitIdentical;
var
  NN: TNNet;
  G, P: TNNetDecodeResult;
  NoStops: array of string;
  Prompt: string;
begin
  // CORE INVARIANT: prompt-lookup speculative decoding emits exactly the greedy
  // argmax at every position; acceptance is a speedup, never a quality change.
  // We use a REPETITION-HEAVY prompt so the suffix n-gram lookup fires every
  // step (forcing several accepted AND rejected drafts), and assert the output
  // is BIT-IDENTICAL to plain DecodeGreedy on the same net. A small ContextLen
  // (the input window is the LAST few chars) makes the tiny net behave like a
  // short-context Markov model, so repeated context yields repeated drafts.
  RandSeed := 424242;
  SetLength(NoStops, 0);
  Prompt := 'abcabcabcabc';
  NN := BuildTinyNet(4, 8);
  try
    G := DecodeGreedy(NN, Prompt, 24);
    // MatchLen=2, NumDraft=3 exercises multi-token accept windows + rejects.
    P := DecodePromptLookup(NN, Prompt, 24, 2, 3, NoStops);
    AssertEquals('prompt-lookup text == greedy text (bit-identical)',
      G.Text, P.Text);
    AssertEquals('prompt-lookup finished == greedy',
      Ord(G.Finished), Ord(P.Finished));
    // A second, different (MatchLen,NumDraft) must ALSO match greedy exactly.
    P := DecodePromptLookup(NN, Prompt, 24, 3, 5, NoStops);
    AssertEquals('prompt-lookup text == greedy text (other params)',
      G.Text, P.Text);
  finally
    NN.Free;
  end;
end;

procedure TTestNeuralDecode.TestPromptLookupNoMatchDegradesToGreedy;
var
  NN: TNNet;
  G, P: TNNetDecodeResult;
  NoStops: array of string;
begin
  // DEGRADE-TO-GREEDY GUARANTEE: a MatchLen LONGER than the whole context can
  // never find an earlier n-gram occurrence, so no draft is ever produced and
  // every step emits exactly one greedy token. Output must be bit-identical to
  // DecodeGreedy.
  RandSeed := 424242;
  SetLength(NoStops, 0);
  NN := BuildTinyNet(4, 8);
  try
    G := DecodeGreedy(NN, 'ab', 8);
    P := DecodePromptLookup(NN, 'ab', 8, 64, 4, NoStops);
    AssertEquals('no-match prompt-lookup text == greedy text', G.Text, P.Text);
    AssertEquals('no-match prompt-lookup finished == greedy',
      Ord(G.Finished), Ord(P.Finished));
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
// FlashAttention-1 tiled online-softmax forward must be numerically equivalent
// (< 1e-5) to the naive full-score forward. Same fixed input fed through a
// naive SDPA and a tiled SDPA across a few SeqLen / Dk / mask / tile configs,
// including a causal case and a tile width that does not divide SeqLen.
procedure TTestNeuralDecode.TestSDPATiledOnlineSoftmaxParity;

  procedure RunOne(SeqLen, Dk, Window: integer; Causal: boolean; TileBc: integer);
  var
    NNNaive, NNTiled: TNNet;
    SDPATiled: TNNetScaledDotProductAttention;
    InV, NaiveOut, TiledOut: TNNetVolume;
    T, D: integer;
    Diff, MaxDiff: TNeuralFloat;
    Tag: string;
  begin
    RandSeed := 424242;
    NNNaive := TNNet.Create();
    NNTiled := TNNet.Create();
    InV := TNNetVolume.Create(SeqLen, 1, 3 * Dk);
    NaiveOut := TNNetVolume.Create();
    TiledOut := TNNetVolume.Create();
    try
      Tag := Format('L=%d Dk=%d W=%d C=%d Bc=%d',
        [SeqLen, Dk, Window, Ord(Causal), TileBc]);
      NNNaive.AddLayer(TNNetInput.Create(SeqLen, 1, 3 * Dk));
      NNNaive.AddLayer(TNNetScaledDotProductAttention.Create(Dk, Causal, Window));
      NNTiled.AddLayer(TNNetInput.Create(SeqLen, 1, 3 * Dk));
      SDPATiled := TNNetScaledDotProductAttention.Create(Dk, Causal, Window);
      NNTiled.AddLayer(SDPATiled);
      SDPATiled.EnableTiledForward(TileBc);
      AssertTrue('tiled flag on ' + Tag, SDPATiled.TiledForward);

      InV.Randomize();
      InV.Sub(0.5);
      NNNaive.Compute(InV);
      NNNaive.GetOutput(NaiveOut);
      NNTiled.Compute(InV);
      NNTiled.GetOutput(TiledOut);

      MaxDiff := 0;
      for T := 0 to SeqLen - 1 do
        for D := 0 to Dk - 1 do
        begin
          Diff := Abs(NaiveOut[T, 0, D] - TiledOut[T, 0, D]);
          if Diff > MaxDiff then MaxDiff := Diff;
        end;
      AssertTrue('tiled parity max|diff| < 1e-5 (' + Tag + ') got '
        + FloatToStr(MaxDiff), MaxDiff < 1e-5);
    finally
      TiledOut.Free;
      NaiveOut.Free;
      InV.Free;
      NNTiled.Free;
      NNNaive.Free;
    end;
  end;

begin
  // Plain bidirectional, tile divides and does not divide SeqLen, single-tile.
  RunOne({SeqLen=}16, {Dk=}8,  {Window=}0, {Causal=}false, {TileBc=}4);
  RunOne({SeqLen=}13, {Dk=}5,  {Window=}0, {Causal=}false, {TileBc=}4);
  RunOne({SeqLen=}10, {Dk=}7,  {Window=}0, {Causal=}false, {TileBc=}64);
  // Causal mask.
  RunOne({SeqLen=}16, {Dk=}8,  {Window=}0, {Causal=}true,  {TileBc=}4);
  RunOne({SeqLen=}31, {Dk=}6,  {Window=}0, {Causal=}true,  {TileBc=}8);
  // Sliding-window causal and a larger depth.
  RunOne({SeqLen=}20, {Dk=}12, {Window=}5, {Causal=}true,  {TileBc=}3);
  RunOne({SeqLen=}64, {Dk=}16, {Window=}0, {Causal=}true,  {TileBc=}16);
end;

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

// Sliding-window (Gemma-2 / Mistral local) attention: the KV-cache
// incremental path must reproduce the full windowed causal forward -- each
// decode step only attends over the last Window cached positions, matching
// the banded mask of the non-cached Compute.
procedure TTestNeuralDecode.TestKVCacheSlidingWindowMatchesFullForward;
const
  SeqLen = 7;
  Dk = 5;
  Window = 3;
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
    NNFull.AddLayer(TNNetScaledDotProductAttention.Create(Dk,
      {CausalMask=}true, Window));
    NNStep.AddLayer(TNNetInput.Create(1, 1, 3 * Dk));
    SDPAStep := TNNetScaledDotProductAttention.Create(Dk,
      {CausalMask=}true, Window);
    NNStep.AddLayer(SDPAStep);

    FullIn.Randomize();
    FullIn.Sub(0.5);
    NNFull.Compute(FullIn);
    NNFull.GetOutput(FullOut);

    SDPAStep.BeginIncrementalDecode({MaxContext=}SeqLen);
    for T := 0 to SeqLen - 1 do
    begin
      for D := 0 to 3 * Dk - 1 do
        StepIn[0, 0, D] := FullIn[T, 0, D];
      NNStep.Compute(StepIn);
      NNStep.GetOutput(StepOut);
      for D := 0 to Dk - 1 do
        AssertEquals('windowed pos ' + IntToStr(T) + ' dim ' + IntToStr(D),
          FullOut[T, 0, D], StepOut[0, 0, D], 1e-5);
    end;
    SDPAStep.EndIncrementalDecode();
  finally
    StepOut.Free;
    FullOut.Free;
    StepIn.Free;
    FullIn.Free;
    NNStep.Free;
    NNFull.Free;
  end;
end;

// TNNetT5RelPosBiasAttention supports the KV-cache incremental decode path:
// the relative position bias depends only on j - i, so adding b[bucket(j-t)]
// at the current absolute position t over the cached keys must reproduce the
// full causal forward token-exactly. Uses a NON-zero trained bias table, a
// sliding window AND a small NumBuckets/MaxDistance so the exact, log and
// cap bucket regions all participate.
procedure TTestNeuralDecode.TestKVCacheT5RelPosBiasMatchesFullForward;
const
  SeqLen = 9;
  Dk = 5;
  Window = 4;
  NumBuckets = 8;
  MaxDistance = 6;
var
  NNFull, NNStep: TNNet;
  AttnFull, AttnStep: TNNetT5RelPosBiasAttention;
  FullIn, StepIn, FullOut, StepOut: TNNetVolume;
  T, D, W: integer;
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
    AttnFull := TNNetT5RelPosBiasAttention.Create(Dk, {CausalMask=}true,
      NumBuckets, MaxDistance, Window);
    NNFull.AddLayer(AttnFull);
    NNStep.AddLayer(TNNetInput.Create(1, 1, 3 * Dk));
    AttnStep := TNNetT5RelPosBiasAttention.Create(Dk, {CausalMask=}true,
      NumBuckets, MaxDistance, Window);
    NNStep.AddLayer(AttnStep);

    // Identical NON-zero bias tables in both layers (a zero table would not
    // exercise the bias term of the cached path at all).
    for W := 0 to NumBuckets - 1 do
    begin
      AttnFull.Neurons[0].Weights.Raw[W] := Sin((W + 1) * 0.57) * 0.6;
      AttnStep.Neurons[0].Weights.Raw[W] := AttnFull.Neurons[0].Weights.Raw[W];
    end;

    FullIn.Randomize();
    FullIn.Sub(0.5);
    NNFull.Compute(FullIn);
    NNFull.GetOutput(FullOut);

    AttnStep.BeginIncrementalDecode({MaxContext=}SeqLen);
    for T := 0 to SeqLen - 1 do
    begin
      for D := 0 to 3 * Dk - 1 do
        StepIn[0, 0, D] := FullIn[T, 0, D];
      NNStep.Compute(StepIn);
      NNStep.GetOutput(StepOut);
      for D := 0 to Dk - 1 do
        AssertEquals('T5 rel-pos cached pos ' + IntToStr(T) + ' dim ' +
          IntToStr(D), FullOut[T, 0, D], StepOut[0, 0, D], 1e-5);
    end;
    AttnStep.EndIncrementalDecode();
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

// int8 KV-cache quantization (opt-in, long-context memory win): stream the
// SAME pinned prompt through a width-1 twin twice - once with the default
// FP32 KV cache, once with int8 KV (per-row scale = maxabs/127, dequant on
// read) - and assert the per-position logit DRIFT stays within a documented
// tolerance AND the next-token argmax agrees at every position. int8 quant is
// LOSSY, so the logits are NOT bit-exact; the assertion is the bound, not
// equality. TOLERANCE: max |logit_int8 - logit_fp32| < 5e-2 over all positions
// and all vocab logits on this pinned prompt, with the argmax (greedy next
// token) identical at every step. The bound is HEADROOM: the measured drift on
// this prompt is ~8.4e-5 (per-row int8 quant is accurate), so 5e-2 is a very
// safe ceiling that absorbs weight/seed variation while still being a real
// (non-vacuous) cap on the lossy quantization.
procedure TTestNeuralDecode.TestKVCacheInt8DriftWithinTolerance;
const
  SeqLen = 7;
  Toks: array[0..6] of integer = (3, 7, 1, 9, 4, 11, 2);
  Tol = 5e-2;
var
  Net, Twin: TNNet;
  SessFP32, SessInt8: TNNetStreamingDecoder;
  StepIn: TNNetVolume;
  T, D, Vocab: integer;
  LogitFP32: array of array of TNeuralFloat;
  ArgMaxFP32: array of integer;
  MaxDrift, Drift, BestVal: TNeuralFloat;
  BestIdx: integer;
begin
  RandSeed := 424242;
  Net := BuildTinyCausalLM(SeqLen);
  Twin := BuildTinyCausalLM(1);
  SessFP32 := nil;
  SessInt8 := nil;
  StepIn := TNNetVolume.Create(1, 1, 1);
  try
    Twin.CopyWeights(Net);
    Vocab := Twin.GetLastLayer().Output.Depth; // = csStreamVocab (12) of the LM head
    SetLength(LogitFP32, SeqLen, Vocab);
    SetLength(ArgMaxFP32, SeqLen);

    // Pass 1: default FP32 KV cache. Record per-position logits + argmax.
    SessFP32 := TNNetStreamingDecoder.Create(Twin, SeqLen);
    AssertTrue('session has attention layers to quantize', SessFP32.SDPACount > 0);
    SessFP32.Reset();
    for T := 0 to SeqLen - 1 do
    begin
      StepIn.FData[0] := Toks[T];
      SessFP32.StepForward(StepIn, T);
      BestIdx := 0; BestVal := SessFP32.Output()[0, 0, 0];
      for D := 0 to Vocab - 1 do
      begin
        LogitFP32[T][D] := SessFP32.Output()[0, 0, D];
        if LogitFP32[T][D] > BestVal then
        begin
          BestVal := LogitFP32[T][D];
          BestIdx := D;
        end;
      end;
      ArgMaxFP32[T] := BestIdx;
    end;
    SessFP32.Free; SessFP32 := nil;

    // Pass 2: int8 KV cache enabled on a FRESH session. Compare to pass 1.
    SessInt8 := TNNetStreamingDecoder.Create(Twin, SeqLen);
    SessInt8.Reset();
    SessInt8.EnableInt8KVCache();
    MaxDrift := 0;
    for T := 0 to SeqLen - 1 do
    begin
      StepIn.FData[0] := Toks[T];
      SessInt8.StepForward(StepIn, T);
      BestIdx := 0; BestVal := SessInt8.Output()[0, 0, 0];
      for D := 0 to Vocab - 1 do
      begin
        Drift := Abs(SessInt8.Output()[0, 0, D] - LogitFP32[T][D]);
        if Drift > MaxDrift then MaxDrift := Drift;
        if SessInt8.Output()[0, 0, D] > BestVal then
        begin
          BestVal := SessInt8.Output()[0, 0, D];
          BestIdx := D;
        end;
      end;
      // Argmax (greedy next token) must be stable under int8 quantization.
      AssertEquals('int8 argmax stable at pos ' + IntToStr(T),
        ArgMaxFP32[T], BestIdx);
    end;
    // Documented tolerance: max logit drift across the whole pinned prompt.
    AssertTrue('int8 KV logit drift ' + FloatToStr(MaxDrift) +
      ' within tolerance ' + FloatToStr(Tol), MaxDrift < Tol);
  finally
    StepIn.Free;
    SessInt8.Free;
    SessFP32.Free;
    Twin.Free;
    Net.Free;
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

function TTestNeuralDecode.BuildTinyLearnedPosLM(ContextLen: integer): TNNet;
begin
  // GPT-2 style: token embedding + LEARNED absolute positions (wpe), then a
  // causal transformer block with NO RoPE (UseRoPE=false). The wpe table holds
  // 16 rows (>= any SeqLen these tests stream), so the width-1 twin and the
  // full net share an identically shaped table for CopyWeights.
  Result := TNNet.Create();
  Result.AddLayer(TNNetInput.Create(ContextLen, 1, 1));
  Result.AddLayer(TNNetEmbedding.Create(csStreamVocab, csStreamDim, 0, 0.02));
  Result.AddLayer(TNNetLearnedPositionalEmbedding.Create({MaxPosition=}16, 0.02));
  Result.AddTransformerEncoderBlock({Heads=}2, {d_ff=}8,
    {PreNorm=}true, {CausalMask=}true, {UseRoPE=}false, {NormClass=}TNNetDyT);
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

function TTestNeuralDecode.BuildTinyGQALM(ContextLen: integer): TNNet;
begin
  // Grouped-Query Attention mixer (4 query heads sharing 2 K/V heads). The
  // builder composes token-wise projections + plain SDPA heads, all
  // sequence-length independent, so it must stream through the same KV-cache
  // path as standard MHA. No RoPE/positional signal on purpose (keeps the
  // exactness check focused on the GQA wiring itself).
  Result := TNNet.Create();
  Result.AddLayer(TNNetInput.Create(ContextLen, 1, 1));
  Result.AddLayer(TNNetEmbedding.Create(csStreamVocab, csStreamDim, 0, 0.02));
  Result.AddMultiHeadGroupedQueryAttention({QueryHeads=}4, {KVHeads=}2,
    {CausalMask=}true);
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

// Grouped-Query Attention nets must stream exactly through the same KV-cache
// path: GQA emits one plain TNNetScaledDotProductAttention per QUERY head (the
// K/V projection weights are shared per group), so the session collects all
// 4 of them and a width-1 weight-copied twin must reproduce every
// per-position output of the full-width causal forward.
procedure TTestNeuralDecode.TestStreamingDecoderGQAMatchesFullForward;
const
  SeqLen = 7;
  Toks: array[0..6] of integer = (6, 2, 10, 3, 8, 1, 5);
var
  Full, Twin: TNNet;
  Session: TNNetStreamingDecoder;
begin
  RandSeed := 424242;
  Full := BuildTinyGQALM(SeqLen);
  Twin := BuildTinyGQALM(1);
  Session := nil;
  try
    Twin.CopyWeights(Full);
    Session := TNNetStreamingDecoder.Create(Twin, SeqLen);
    AssertEquals('one SDPA per QUERY head (KV sharing is in the projections)',
      4, Session.SDPACount);
    AssertEquals('no SSM layers in a GQA transformer', 0, Session.SSMCount);
    Session.Reset();
    AssertStreamMatchesFull(Full, Session, Toks, 'gqa');
  finally
    Session.Free;
    Twin.Free;
    Full.Free;
  end;
end;

// GPT-2-style LEARNED absolute positions (TNNetLearnedPositionalEmbedding /
// "wpe") must stream exactly: the session advances the layer's PositionOffset
// per StepForward so a width-1 step at absolute position p reads wpe[p], the
// same row the full-width forward adds at position p. No RoPE in this net -
// position lives entirely in the learned table.
procedure TTestNeuralDecode.TestStreamingDecoderLearnedPosMatchesFullForward;
const
  SeqLen = 7;
  Toks: array[0..6] of integer = (4, 1, 9, 2, 11, 3, 8);
var
  Full, Twin: TNNet;
  Session: TNNetStreamingDecoder;
begin
  RandSeed := 424242;
  Full := BuildTinyLearnedPosLM(SeqLen);
  Twin := BuildTinyLearnedPosLM(1);
  Session := nil;
  try
    Twin.CopyWeights(Full);
    Session := TNNetStreamingDecoder.Create(Twin, SeqLen);
    AssertEquals('one SDPA per head', 2, Session.SDPACount);
    AssertEquals('no RoPE layers (learned absolute positions)',
      0, Session.RopeCount);
    Session.Reset();
    AssertStreamMatchesFull(Full, Session, Toks, 'learnedpos');
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

// Prefix-cache reuse / cache fork, the headline correctness gate: prefill a
// shared prompt P, Snapshot, then RestoreSnapshot into a FRESH session and
// generate continuation C. Every continuation output row must be BIT-IDENTICAL
// (exact float equality, tolerance 0) to streaming the WHOLE P+C through a
// session that never forked - i.e. the fork is indistinguishable from a fresh
// prefill of the whole prefix. RoPE transformer, so the absolute-position
// contract is exercised across the snapshot boundary too.
procedure TTestNeuralDecode.TestStreamingDecoderForkContinuationBitIdenticalTransformer;
const
  PromptLen = 4;
  TotalLen  = 9;
  Toks: array[0..8] of integer = (3, 7, 1, 9, 4, 11, 2, 6, 8);
var
  Twin: TNNet;
  RefSession, ForkSession: TNNetStreamingDecoder;
  Snap: TNNetDecoderSessionSnapshot;
  StepIn: TNNetVolume;
  RefOut: array[PromptLen..TotalLen - 1] of array of TNeuralFloat;
  T, D: integer;
begin
  RandSeed := 424242;
  Twin := BuildTinyCausalLM(1);
  RefSession := nil; ForkSession := nil; Snap := nil;
  StepIn := TNNetVolume.Create(1, 1, 1);
  try
    // --- Reference: one session prefills P then continues C, recording every
    // continuation output row (the "fresh prefill of the whole prefix" path). ---
    RefSession := TNNetStreamingDecoder.Create(Twin, TotalLen);
    RefSession.Reset();
    for T := 0 to TotalLen - 1 do
    begin
      StepIn.FData[0] := Toks[T];
      RefSession.StepForward(StepIn, T);
      if T >= PromptLen then
      begin
        SetLength(RefOut[T], RefSession.Output().Size);
        for D := 0 to RefSession.Output().Size - 1 do
          RefOut[T][D] := RefSession.Output().FData[D];
      end;
    end;
    RefSession.Free; RefSession := nil;

    // --- Fork: prefill ONLY P, Snapshot, then restore into a brand-new session
    // and generate C. Must reproduce the recorded rows bit-for-bit. ---
    ForkSession := TNNetStreamingDecoder.Create(Twin, TotalLen);
    ForkSession.Reset();
    for T := 0 to PromptLen - 1 do
    begin
      StepIn.FData[0] := Toks[T];
      ForkSession.StepForward(StepIn, T);
    end;
    Snap := ForkSession.Snapshot();
    AssertEquals('snapshot captured every attention layer',
      ForkSession.SDPACount, Snap.SDPACount);
    ForkSession.Free; ForkSession := nil;

    ForkSession := TNNetStreamingDecoder.Create(Twin, TotalLen);
    ForkSession.Reset();           // deliberately a different live state...
    ForkSession.RestoreSnapshot(Snap);   // ...overwritten by the snapshot.
    for T := PromptLen to TotalLen - 1 do
    begin
      StepIn.FData[0] := Toks[T];
      ForkSession.StepForward(StepIn, T);
      AssertEquals('forked continuation size pos ' + IntToStr(T),
        Length(RefOut[T]), ForkSession.Output().Size);
      for D := 0 to ForkSession.Output().Size - 1 do
        AssertTrue('BIT-IDENTICAL pos ' + IntToStr(T) + ' dim ' + IntToStr(D) +
          ' ref=' + FloatToStr(RefOut[T][D]) + ' fork=' +
          FloatToStr(ForkSession.Output().FData[D]),
          RefOut[T][D] = ForkSession.Output().FData[D]);
    end;
  finally
    Snap.Free;
    ForkSession.Free;
    RefSession.Free;
    StepIn.Free;
    Twin.Free;
  end;
end;

// Same bit-identical fork gate for the recurrent (SSM) family: the snapshot must
// also carry the persisted recurrent state h, so continuation from a fork equals
// continuation from a session that streamed the whole prefix.
procedure TTestNeuralDecode.TestStreamingDecoderForkContinuationBitIdenticalSSM;
const
  PromptLen = 4;
  TotalLen  = 9;
  Toks: array[0..8] of integer = (5, 2, 9, 9, 1, 7, 3, 10, 6);
var
  Twin: TNNet;
  RefSession, ForkSession: TNNetStreamingDecoder;
  Snap: TNNetDecoderSessionSnapshot;
  StepIn: TNNetVolume;
  RefOut: array[PromptLen..TotalLen - 1] of array of TNeuralFloat;
  T, D: integer;
begin
  RandSeed := 424242;
  Twin := BuildTinySSMLM(1);
  RefSession := nil; ForkSession := nil; Snap := nil;
  StepIn := TNNetVolume.Create(1, 1, 1);
  try
    RefSession := TNNetStreamingDecoder.Create(Twin, TotalLen);
    RefSession.Reset();
    for T := 0 to TotalLen - 1 do
    begin
      StepIn.FData[0] := Toks[T];
      RefSession.StepForward(StepIn, T);
      if T >= PromptLen then
      begin
        SetLength(RefOut[T], RefSession.Output().Size);
        for D := 0 to RefSession.Output().Size - 1 do
          RefOut[T][D] := RefSession.Output().FData[D];
      end;
    end;
    RefSession.Free; RefSession := nil;

    ForkSession := TNNetStreamingDecoder.Create(Twin, TotalLen);
    ForkSession.Reset();
    for T := 0 to PromptLen - 1 do
    begin
      StepIn.FData[0] := Toks[T];
      ForkSession.StepForward(StepIn, T);
    end;
    Snap := ForkSession.Snapshot();
    AssertEquals('snapshot captured the SSM layer', ForkSession.SSMCount,
      Snap.SSMCount);
    ForkSession.Free; ForkSession := nil;

    ForkSession := TNNetStreamingDecoder.Create(Twin, TotalLen);
    ForkSession.Reset();
    ForkSession.RestoreSnapshot(Snap);
    for T := PromptLen to TotalLen - 1 do
    begin
      StepIn.FData[0] := Toks[T];
      ForkSession.StepForward(StepIn, T);
      for D := 0 to ForkSession.Output().Size - 1 do
        AssertTrue('SSM BIT-IDENTICAL pos ' + IntToStr(T) + ' dim ' + IntToStr(D),
          RefOut[T][D] = ForkSession.Output().FData[D]);
    end;
  finally
    Snap.Free;
    ForkSession.Free;
    RefSession.Free;
    StepIn.Free;
    Twin.Free;
  end;
end;

// One snapshot forks MANY independent sessions: prefill a shared prompt once,
// snapshot it, then restore it into the SAME session twice with DIFFERENT
// continuations. The snapshot must be untouched by the first continuation (the
// fork is a deep copy, not a move), so the second restore reproduces the first
// continuation's outputs exactly when fed the same tokens - proving the prompt
// was prefilled once and reused, not consumed.
procedure TTestNeuralDecode.TestStreamingDecoderSnapshotForksManyIndependentSessions;
const
  PromptLen = 4;
  ContLen   = 3;
  Toks:  array[0..6] of integer = (3, 7, 1, 9, 4, 11, 2);
  ContB: array[0..2] of integer = (8, 5, 6);   // a DIFFERENT continuation
var
  Twin: TNNet;
  Session: TNNetStreamingDecoder;
  Snap: TNNetDecoderSessionSnapshot;
  StepIn: TNNetVolume;
  RunA: array[PromptLen..PromptLen + ContLen - 1] of array of TNeuralFloat;
  T, D: integer;
begin
  RandSeed := 424242;
  Twin := BuildTinyCausalLM(1);
  Session := nil; Snap := nil;
  StepIn := TNNetVolume.Create(1, 1, 1);
  try
    Session := TNNetStreamingDecoder.Create(Twin, PromptLen + ContLen);
    Session.Reset();
    for T := 0 to PromptLen - 1 do
    begin
      StepIn.FData[0] := Toks[T];
      Session.StepForward(StepIn, T);
    end;
    Snap := Session.Snapshot();

    // Run A: continuation = the prompt's natural tail tokens.
    Session.RestoreSnapshot(Snap);
    for T := PromptLen to PromptLen + ContLen - 1 do
    begin
      StepIn.FData[0] := Toks[T];
      Session.StepForward(StepIn, T);
      SetLength(RunA[T], Session.Output().Size);
      for D := 0 to Session.Output().Size - 1 do
        RunA[T][D] := Session.Output().FData[D];
    end;

    // Run B: a DIFFERENT continuation off the SAME snapshot (perturbs the
    // live cache well past the snapshot length).
    Session.RestoreSnapshot(Snap);
    for T := 0 to ContLen - 1 do
    begin
      StepIn.FData[0] := ContB[T];
      Session.StepForward(StepIn, PromptLen + T);
    end;

    // Run A again off the same snapshot: must reproduce RunA bit-for-bit,
    // proving the snapshot survived both prior restores untouched.
    Session.RestoreSnapshot(Snap);
    for T := PromptLen to PromptLen + ContLen - 1 do
    begin
      StepIn.FData[0] := Toks[T];
      Session.StepForward(StepIn, T);
      for D := 0 to Session.Output().Size - 1 do
        AssertTrue('reusable snapshot pos ' + IntToStr(T) + ' dim ' + IntToStr(D),
          RunA[T][D] = Session.Output().FData[D]);
    end;
  finally
    Snap.Free;
    Session.Free;
    StepIn.Free;
    Twin.Free;
  end;
end;

// StreamingLLM eviction: while the streamed length stays within the sink+window
// budget, eviction never fires, so an eviction-enabled session must be
// BIT-IDENTICAL (not just close) to the unbounded-cache session AND to the
// full-width causal forward. Uses a RoPE causal transformer so the test also
// exercises the "keep original positions" RoPE-after-eviction contract.
procedure TTestNeuralDecode.TestStreamingEvictionWithinWindowBitIdenticalToUnbounded;
const
  Sinks = 2;
  Window = 4;
  // Streamed length (6) equals Sinks+Window exactly: still no eviction (we
  // evict only when CacheLength would EXCEED the cap), so it must match.
  SeqLen = 6;
  Toks: array[0..5] of integer = (3, 7, 1, 9, 4, 11);
var
  Full, TwinA, TwinB: TNNet;
  Plain, Evict: TNNetStreamingDecoder;
  FullIn, FullOut, StepIn: TNNetVolume;
  T, D: integer;
begin
  RandSeed := 424242;
  Full := BuildTinyCausalLM(SeqLen);
  TwinA := BuildTinyCausalLM(1);
  TwinB := BuildTinyCausalLM(1);
  Plain := nil; Evict := nil;
  FullIn := TNNetVolume.Create(SeqLen, 1, 1);
  FullOut := TNNetVolume.Create();
  StepIn := TNNetVolume.Create(1, 1, 1);
  try
    TwinA.CopyWeights(Full);
    TwinB.CopyWeights(Full);
    for T := 0 to SeqLen - 1 do FullIn.FData[T] := Toks[T];
    Full.Compute(FullIn);
    Full.GetOutput(FullOut);

    Plain := TNNetStreamingDecoder.Create(TwinA, SeqLen);
    Evict := TNNetStreamingDecoder.Create(TwinB, SeqLen);
    Evict.EnableEviction(Sinks, Window);
    AssertTrue('has attention layers', Evict.SDPACount > 0);
    Plain.Reset();
    Evict.Reset();
    for T := 0 to SeqLen - 1 do
    begin
      StepIn.FData[0] := Toks[T];
      Plain.StepForward(StepIn, T);
      StepIn.FData[0] := Toks[T];
      Evict.StepForward(StepIn, T);
      // (1) Bit-identical to the unbounded session.
      for D := 0 to FullOut.Depth - 1 do
        AssertTrue('evict==plain pos ' + IntToStr(T) + ' dim ' + IntToStr(D),
          Plain.Output()[0, 0, D] = Evict.Output()[0, 0, D]);
      // ... and both reproduce the full causal forward (sanity).
      for D := 0 to FullOut.Depth - 1 do
        AssertEquals('evict==full pos ' + IntToStr(T) + ' dim ' + IntToStr(D),
          FullOut[T, 0, D], Evict.Output()[0, 0, D], 1e-5);
    end;
  finally
    StepIn.Free;
    FullOut.Free;
    FullIn.Free;
    Plain.Free;
    Evict.Free;
    TwinA.Free;
    TwinB.Free;
    Full.Free;
  end;
end;

// StreamingLLM eviction: stream far PAST the sink+window budget and assert every
// attention layer's CacheLength is pinned at Sinks+Window (constant memory),
// while the unbounded session grows linearly. The first Sinks slots are
// untouched and the middle is evicted, so generation continues forever.
procedure TTestNeuralDecode.TestStreamingEvictionCapsCacheLengthPastWindow;
const
  Sinks = 2;
  Window = 4;
  Cap = Sinks + Window;   // 6
  Steps = 30;             // well past the cap
  Budget = 64;            // MaxCacheLen >= Cap; covers the unbounded session too
var
  Twin: TNNet;
  Evict: TNNetStreamingDecoder;
  StepIn: TNNetVolume;
  T, L: integer;
begin
  RandSeed := 424242;
  Twin := BuildTinyCausalLM(1);
  Evict := nil;
  StepIn := TNNetVolume.Create(1, 1, 1);
  try
    Evict := TNNetStreamingDecoder.Create(Twin, Budget);
    Evict.EnableEviction(Sinks, Window);
    Evict.Reset();
    for T := 0 to Steps - 1 do
    begin
      StepIn.FData[0] := (T * 7 + 3) mod csStreamVocab;
      Evict.StepForward(StepIn, T);
      // Every attention layer's live cache is capped at the budget once we pass
      // it, and never below min(T+1, Cap) before.
      for L := 0 to Evict.SDPACount - 1 do
      begin
        if T + 1 <= Cap then
          AssertEquals('pre-cap len layer ' + IntToStr(L) + ' step ' + IntToStr(T),
            T + 1, Evict.SDPACacheLength(L))
        else
          AssertEquals('capped len layer ' + IntToStr(L) + ' step ' + IntToStr(T),
            Cap, Evict.SDPACacheLength(L));
      end;
    end;
    // Final state: pinned at the cap, not Steps.
    for L := 0 to Evict.SDPACount - 1 do
      AssertEquals('flat memory layer ' + IntToStr(L), Cap,
        Evict.SDPACacheLength(L));
  finally
    StepIn.Free;
    Evict.Free;
    Twin.Free;
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
  // MulWeights(1.0) is a no-op on the values but runs AfterWeightUpdate, so
  // layers with packed/prepared weight copies see the hand-edited neurons
  // even when the rigged net is computed directly (without a CopyWeights
  // refresh on a twin).
  Head.MulWeights(1.0);
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

// The extended overload with Penalty=nil and no stop sequences must produce
// exactly the same stream (length and tokens) as the plain overload.
procedure TTestNeuralDecode.TestGenerateTokensStreamedExtendedOverloadDefaultsMatchPlain;
const
  SeqLen = 10;
  PromptLen = 3;
var
  Full, Twin: TNNet;
  Session: TNNetStreamingDecoder;
  PlainToks, ExtToks: TNeuralIntegerArray;
  PlainLen, ExtLen, T: integer;
begin
  RandSeed := 424242;
  Full := BuildTinySSMLM(SeqLen);
  Twin := BuildTinySSMLM(1);
  Session := nil;
  try
    Twin.CopyWeights(Full);
    Session := TNNetStreamingDecoder.Create(Twin, SeqLen);
    SetLength(PlainToks, PromptLen);
    SetLength(ExtToks, PromptLen);
    PlainToks[0] := 5; PlainToks[1] := 2; PlainToks[2] := 9;
    for T := 0 to PromptLen - 1 do ExtToks[T] := PlainToks[T];

    PlainLen := GenerateTokensStreamed(Session, PlainToks, PromptLen,
      SeqLen - PromptLen, SeqLen);
    ExtLen := GenerateTokensStreamed(Session, ExtToks, PromptLen,
      SeqLen - PromptLen, SeqLen, nil, nil, nil);

    AssertEquals('same length with default extended args', PlainLen, ExtLen);
    for T := PromptLen to PlainLen - 1 do
      AssertEquals('token at pos ' + IntToStr(T), PlainToks[T], ExtToks[T]);
  finally
    Session.Free;
    Twin.Free;
    Full.Free;
  end;
end;

// A matched stop sequence terminates generation and is TRIMMED from the
// returned length. Deterministic via a head rigged to a constant token 7:
// a [7] stop trims immediately; a [7,7] stop trims after two emissions.
procedure TTestNeuralDecode.TestGenerateTokensStreamedStopSequenceTrims;
const
  PromptLen = 2;
var
  Full, Twin: TNNet;
  Session: TNNetStreamingDecoder;
  Toks: TNeuralIntegerArray;
  Stops: TNNetTokenSequences;
  OutLen: integer;
begin
  RandSeed := 424242;
  Full := BuildTinySSMLM(16);
  Twin := BuildTinySSMLM(1);
  Session := nil;
  try
    RigHeadToConstantToken(Full, 7);
    Twin.CopyWeights(Full);
    Session := TNNetStreamingDecoder.Create(Twin, 16);

    // Single-token stop [7]: the first emitted 7 matches and is trimmed.
    SetLength(Stops, 1);
    SetLength(Stops[0], 1);
    Stops[0][0] := 7;
    SetLength(Toks, PromptLen);
    Toks[0] := 3; Toks[1] := 8;
    OutLen := GenerateTokensStreamed(Session, Toks, PromptLen, 10, 16,
      nil, nil, Stops);
    AssertEquals('single-token stop trims to the prompt', PromptLen, OutLen);

    // Two-token stop [7,7]: emitted 7,7 then both trimmed.
    SetLength(Stops[0], 2);
    Stops[0][0] := 7; Stops[0][1] := 7;
    SetLength(Toks, 0);
    SetLength(Toks, PromptLen);
    Toks[0] := 3; Toks[1] := 8;
    OutLen := GenerateTokensStreamed(Session, Toks, PromptLen, 10, 16,
      nil, nil, Stops);
    AssertEquals('two-token stop trims both matched tokens',
      PromptLen, OutLen);
  finally
    Session.Free;
    Twin.Free;
    Full.Free;
  end;
end;

// A stop sequence must match entirely inside the GENERATED region: with a
// prompt ENDING in 7 and a [7,7] stop, the first generated 7 must NOT match
// (it would span the prompt boundary); only the second generated 7 completes
// a legal match, trimming back exactly to the prompt.
procedure TTestNeuralDecode.TestGenerateTokensStreamedStopSequenceStaysInGeneratedRegion;
const
  PromptLen = 2;
var
  Full, Twin: TNNet;
  Session: TNNetStreamingDecoder;
  Toks: TNeuralIntegerArray;
  Stops: TNNetTokenSequences;
  OutLen: integer;
begin
  RandSeed := 424242;
  Full := BuildTinySSMLM(16);
  Twin := BuildTinySSMLM(1);
  Session := nil;
  try
    RigHeadToConstantToken(Full, 7);
    Twin.CopyWeights(Full);
    Session := TNNetStreamingDecoder.Create(Twin, 16);
    SetLength(Stops, 1);
    SetLength(Stops[0], 2);
    Stops[0][0] := 7; Stops[0][1] := 7;
    SetLength(Toks, PromptLen);
    Toks[0] := 3; Toks[1] := 7; // prompt ENDS in 7
    OutLen := GenerateTokensStreamed(Session, Toks, PromptLen, 10, 16,
      nil, nil, Stops);
    // A boundary-spanning (buggy) match after the FIRST generated 7 would
    // trim into the prompt (OutLen = PromptLen - 1). The correct match needs
    // TWO generated 7s and trims back exactly to the prompt.
    AssertEquals('match confined to the generated region', PromptLen, OutLen);
    AssertEquals('prompt left untouched', 7, Toks[PromptLen - 1]);
  finally
    Session.Free;
    Twin.Free;
    Full.Free;
  end;
end;

// With a SOFTMAX-headed LM and a huge frequency penalty, every seen token's
// probability collapses to ~0, so greedy decode can never emit the same
// token twice (nor re-emit a prompt token): the whole stream is pairwise
// distinct. This also exercises the probability-domain penalty path inside
// the generation loop end to end.
procedure TTestNeuralDecode.TestGenerateTokensStreamedPenaltyAvoidsRepetition;
const
  SeqLen = 12;
  PromptLen = 3;
var
  Full, Twin: TNNet;
  Session: TNNetStreamingDecoder;
  Penalty: TNNetTokenHistoryPenalty;
  Toks: TNeuralIntegerArray;
  OutLen, I, J: integer;
begin
  RandSeed := 424242;
  Full := BuildTinySSMLM(SeqLen);
  Twin := BuildTinySSMLM(1);
  // The streamed penalty path expects POST-SOFTMAX probabilities: cap both
  // twins with a per-position softmax (weightless, so CopyWeights is exact).
  Full.AddLayer(TNNetPointwiseSoftMax.Create());
  Twin.AddLayer(TNNetPointwiseSoftMax.Create());
  Session := nil;
  Penalty := TNNetTokenHistoryPenalty.Create(1.0, {Frequency=}100.0, 0.0);
  try
    Twin.CopyWeights(Full);
    Session := TNNetStreamingDecoder.Create(Twin, SeqLen);
    SetLength(Toks, PromptLen);
    Toks[0] := 4; Toks[1] := 9; Toks[2] := 6; // distinct prompt tokens
    OutLen := GenerateTokensStreamed(Session, Toks, PromptLen,
      SeqLen - PromptLen, SeqLen, nil, Penalty, nil);
    AssertTrue('something was generated', OutLen > PromptLen);
    for I := 0 to OutLen - 2 do
      for J := I + 1 to OutLen - 1 do
        AssertTrue('tokens at ' + IntToStr(I) + ' and ' + IntToStr(J) +
          ' must differ under a huge frequency penalty',
          Toks[I] <> Toks[J]);
  finally
    Penalty.Free;
    Session.Free;
    Twin.Free;
    Full.Free;
  end;
end;

// DecodeGreedy stop strings: with the logit layer rigged to a constant
// char, a matching stop string terminates generation, is trimmed from the
// text and marks the result Finished; a non-occurring stop string leaves
// the output identical to the plain overload.
procedure TTestNeuralDecode.TestDecodeGreedyStopStringTrimsAndFinishes;
const
  Vocab = 16;
  MaxLen = 6;
var
  NN: TNNet;
  Logit: TNNetLayer;
  Plain, Stopped, Unmatched: TNNetDecodeResult;
  N: integer;
begin
  RandSeed := 424242;
  NN := BuildTinyNet(8, Vocab);
  try
    // Rig the LOGIT layer (the SoftMax head has no neurons) to constant 7.
    Logit := NN.Layers[NN.Layers.Count - 2];
    for N := 0 to Logit.Neurons.Count - 1 do
    begin
      Logit.Neurons[N].Weights.Fill(0);
      Logit.Neurons[N].BiasWeight := 0;
    end;
    Logit.Neurons[7].BiasWeight := 5;

    Plain := DecodeGreedy(NN, 'ab', MaxLen);
    AssertEquals('rigged plain decode emits MaxLen constant chars',
      StringOfChar(Chr(7), MaxLen), Plain.Text);
    AssertEquals('plain decode does not finish', False, Plain.Finished);

    Stopped := DecodeGreedy(NN, 'ab', MaxLen, [Chr(7) + Chr(7)]);
    AssertEquals('stop string trimmed from the output', '', Stopped.Text);
    AssertEquals('stop string marks the result finished',
      True, Stopped.Finished);

    Unmatched := DecodeGreedy(NN, 'ab', MaxLen, [Chr(9)]);
    AssertEquals('non-occurring stop leaves the text unchanged',
      Plain.Text, Unmatched.Text);
    AssertEquals('non-occurring stop does not finish',
      False, Unmatched.Finished);
  finally
    NN.Free;
  end;
end;

// String-level wrapper: a stop string is tokenized for early termination and
// never appears in the returned string; without stops the rigged net keeps
// appending the constant word.
procedure TTestNeuralDecode.TestGenerateStringStreamedStopStringTrims;
const
  SeqLen = 9;
  Words: array[0..11] of string = ('<eos>', '<pad>', 'apple', 'blue', 'cat',
    'dog', 'egg', 'fox', 'gold', 'hat', 'ice', 'jam');
var
  Full, Twin: TNNet;
  Session: TNNetStreamingDecoder;
  Dict: TStringListInt;
  DogToks: TNeuralIntegerArray;
  T: integer;
  NoStops, WithStops: string;
begin
  RandSeed := 424242;
  Full := BuildTinyCausalLM(SeqLen);
  Twin := BuildTinyCausalLM(1);
  Session := nil;
  Dict := TStringListInt.Create();
  try
    Dict.Sorted := true;
    for T := 0 to High(Words) do Dict.Add(Words[T]);
    Dict.SaveCurrentPosition();
    Dict.Tokenize('dog', DogToks);
    AssertEquals('stop word tokenizes to one id', 1, Length(DogToks));
    RigHeadToConstantToken(Full, DogToks[0]);
    Twin.CopyWeights(Full);
    Session := TNNetStreamingDecoder.Create(Twin, SeqLen);

    NoStops := GenerateStringStreamed(Session, Dict, 'cat egg',
      SeqLen - 2, SeqLen);
    AssertEquals('rigged net keeps emitting the constant word',
      'cat egg dog dog dog dog dog dog dog', NoStops);

    WithStops := GenerateStringStreamed(Session, Dict, 'cat egg',
      SeqLen - 2, SeqLen, nil, nil, ['dog']);
    AssertEquals('stop string trimmed: prompt returned unchanged',
      'cat egg', WithStops);
  finally
    Dict.Free;
    Session.Free;
    Twin.Free;
    Full.Free;
  end;
end;

// TNNetTemperatureProcessor probability-domain math: T=1 is a bitwise no-op,
// T=0.5 equals p^2 renormalized (the exact image of logits/T), and T at/below
// the clamp concentrates all mass on the argmax (greedy degeneration - the
// DecodeSeq2SeqSampled convention).
procedure TTestNeuralDecode.TestTemperatureProcessorProbabilityDomainMath;
var
  Row: TNNetVolume;
  Proc: TNNetTemperatureProcessor;
  I: integer;
  SumSq, Total: TNeuralFloat;
const
  P: array[0..3] of TNeuralFloat = (0.1, 0.2, 0.3, 0.4);
begin
  Row := TNNetVolume.Create(4, 1, 1);
  Proc := TNNetTemperatureProcessor.Create(1.0);
  try
    AssertTrue('declares the probability domain',
      Proc.ExpectsProbabilities());
    // T = 1.0: bitwise no-op.
    for I := 0 to 3 do Row.Raw[I] := P[I];
    Proc.ProcessRow(Row);
    for I := 0 to 3 do
      AssertEquals('T=1 leaves p[' + IntToStr(I) + '] bitwise unchanged',
        P[I], Row.Raw[I], 0.0);
    // T = 0.5: p^(1/T) = p^2, renormalized.
    Proc.Temperature := 0.5;
    SumSq := 0;
    for I := 0 to 3 do SumSq := SumSq + P[I] * P[I];
    for I := 0 to 3 do Row.Raw[I] := P[I];
    Proc.ProcessRow(Row);
    for I := 0 to 3 do
      AssertEquals('T=0.5 equals p^2 renormalized at ' + IntToStr(I),
        P[I] * P[I] / SumSq, Row.Raw[I], 1e-6);
    Total := 0;
    for I := 0 to 3 do Total := Total + Row.Raw[I];
    AssertEquals('T=0.5 row sums to 1', 1.0, Total, 1e-6);
    // T -> 0 (below the clamp): one-hot at the argmax.
    Proc.Temperature := 1e-12;
    for I := 0 to 3 do Row.Raw[I] := P[I];
    Proc.ProcessRow(Row);
    AssertEquals('T->0 puts all mass on the argmax', 1.0, Row.Raw[3], 1e-6);
    for I := 0 to 2 do
      AssertEquals('T->0 zeroes non-argmax token ' + IntToStr(I),
        0.0, Row.Raw[I], 1e-6);
  finally
    Proc.Free;
    Row.Free;
  end;
end;

// TNNetNoRepeatNGramProcessor: EXACT n-gram blocking (HF no_repeat_ngram_size).
// With size=2, after a context ending in token "B" whose only prior
// occurrence was followed by "C", the bigram (B,C) cannot be re-formed: prob
// of C is zeroed and surviving mass renormalized. With size=3 only the FULL
// 2-token suffix keys the ban (a token sharing just the 1-token suffix is NOT
// blocked). Banning is in the probability domain (the post-softmax image of
// logit -> -inf); a uniform row over the survivors integrates to 1.
procedure TTestNeuralDecode.TestNoRepeatNGramBansSeenBigramAndTrigram;
var
  Proc: TNNetNoRepeatNGramProcessor;
  Row: TNNetVolume;
  I: integer;
  Total: TNeuralFloat;
begin
  // ---- size=2: bigram blocking. Context: A B C B  (vocab 0..4: A=0..C=2).
  // Seen bigram (B,C) = (3->4 indices): the only B at index 1 was followed by
  // C. Current suffix is the trailing "B" (index 3); its prior occurrence was
  // followed by C, so C (=2) must be banned for the next token.
  Proc := TNNetNoRepeatNGramProcessor.Create(2);
  Row := TNNetVolume.Create(5, 1, 1);
  try
    AssertTrue('declares the probability domain', Proc.ExpectsProbabilities());
    Proc.Reset([0, 1, 2, 1]); // A B C B  (B=1, C=2)
    for I := 0 to 4 do Row.Raw[I] := 0.2; // uniform
    Proc.ProcessRow(Row);
    AssertEquals('size=2 bans C (re-forms seen bigram B,C)', 0.0, Row.Raw[2],
      0.0);
    Total := 0;
    for I := 0 to 4 do Total := Total + Row.Raw[I];
    AssertEquals('survivors renormalize to 1', 1.0, Total, 1e-6);
    AssertEquals('an unrelated token keeps non-zero mass (>0)', 0.25,
      Row.Raw[0], 1e-6); // 0.2 / (4*0.2) = 0.25
  finally
    Row.Free;
    Proc.Free;
  end;

  // ---- size=3: trigram blocking keys on the 2-token suffix. Context:
  // A B C  A B  -> suffix "A B". The prior "A B" (indices 0,1) was followed by
  // C, so C must be banned. A token that only shares the 1-token suffix "B"
  // (e.g. via a (X,B,*) elsewhere) is NOT blocked - only the full (A,B,*).
  Proc := TNNetNoRepeatNGramProcessor.Create(3);
  Row := TNNetVolume.Create(5, 1, 1);
  try
    Proc.Reset([0, 1, 2, 0, 1]); // A B C A B
    for I := 0 to 4 do Row.Raw[I] := 0.2;
    Proc.ProcessRow(Row);
    AssertEquals('size=3 bans C (re-forms seen trigram A,B,C)', 0.0,
      Row.Raw[2], 0.0);
    Total := 0;
    for I := 0 to 4 do Total := Total + Row.Raw[I];
    AssertEquals('trigram survivors renormalize to 1', 1.0, Total, 1e-6);

    // Negative: with the SAME suffix "B" but a DIFFERENT preceding token, the
    // (A,B,C) ban does not fire. Context: D B (suffix "B" only, no full 2-gram
    // suffix repeated) -> with size=3 and FLen<3 nothing is banned.
    Proc.Reset([3, 1]); // D B  (len 2 < 3)
    for I := 0 to 4 do Row.Raw[I] := 0.2;
    Proc.ProcessRow(Row);
    for I := 0 to 4 do
      AssertEquals('size=3 with short context bans nothing at ' + IntToStr(I),
        0.2, Row.Raw[I], 1e-6);
  finally
    Row.Free;
    Proc.Free;
  end;

  // ---- Commit advances state: after emitting a token the new suffix governs
  // the next ban. Start "A B", emit C -> context "A B C"; with size=2 the
  // suffix is now "C" (no prior C), so nothing banned; then the prior B,C is
  // recorded so a later B would ban C.
  Proc := TNNetNoRepeatNGramProcessor.Create(2);
  Row := TNNetVolume.Create(5, 1, 1);
  try
    Proc.Reset([0, 1]); // A B
    Proc.Commit(2);     // emit C -> A B C
    Proc.Commit(1);     // emit B -> A B C B ; suffix B, prior B followed by C
    for I := 0 to 4 do Row.Raw[I] := 0.2;
    Proc.ProcessRow(Row);
    AssertEquals('after Commit, suffix B bans C', 0.0, Row.Raw[2], 0.0);
  finally
    Row.Free;
    Proc.Free;
  end;
end;

// Order matters: penalty-then-temperature sharpens the PENALIZED distribution
// while temperature-then-penalty penalizes the SHARPENED one - the frequency
// factor enters once as exp(-a)^(1/T) vs exp(-a). Both expected rows are
// computed in closed form here.
procedure TTestNeuralDecode.TestProcessorChainOrderMatters;
const
  Alpha = 0.7;
  Temp = 0.5;
var
  Row: TNNetVolume;
  ChainA, ChainB: TNNetLogitsProcessorChain;
  PenA, PenB: TNNetTokenHistoryPenalty;
  A0, A1, B0, B1, Norm: TNeuralFloat;
begin
  Row := TNNetVolume.Create(2, 1, 1);
  PenA := TNNetTokenHistoryPenalty.Create(1.0, Alpha, 0.0);
  PenB := TNNetTokenHistoryPenalty.Create(1.0, Alpha, 0.0);
  // Chain A: penalty THEN temperature; chain B: temperature THEN penalty.
  ChainA := TNNetLogitsProcessorChain.Create();
  ChainA.Add(TNNetPenaltyProcessor.Create(PenA), true)
        .Add(TNNetTemperatureProcessor.Create(Temp), true);
  ChainB := TNNetLogitsProcessorChain.Create();
  ChainB.Add(TNNetTemperatureProcessor.Create(Temp), true)
        .Add(TNNetPenaltyProcessor.Create(PenB), true);
  try
    AssertEquals('chain A holds two processors', 2, ChainA.Count);
    // Token 0 was seen once (the "prompt"); the row is (0.6, 0.4).
    ChainA.Reset([0]);
    Row.Raw[0] := 0.6; Row.Raw[1] := 0.4;
    ChainA.ProcessRow(Row);
    // Expected: penalize (0.6*exp(-a), 0.4)/Z, then square and renormalize.
    A0 := 0.6 * Exp(-Alpha); A1 := 0.4;
    Norm := A0 + A1; A0 := A0 / Norm; A1 := A1 / Norm;
    A0 := A0 * A0; A1 := A1 * A1;
    Norm := A0 + A1; A0 := A0 / Norm; A1 := A1 / Norm;
    AssertEquals('penalty->temperature p0', A0, Row.Raw[0], 1e-6);
    AssertEquals('penalty->temperature p1', A1, Row.Raw[1], 1e-6);

    ChainB.Reset([0]);
    Row.Raw[0] := 0.6; Row.Raw[1] := 0.4;
    ChainB.ProcessRow(Row);
    // Expected: square and renormalize, THEN penalize once.
    B0 := 0.6 * 0.6; B1 := 0.4 * 0.4;
    Norm := B0 + B1; B0 := B0 / Norm; B1 := B1 / Norm;
    B0 := B0 * Exp(-Alpha);
    Norm := B0 + B1; B0 := B0 / Norm; B1 := B1 / Norm;
    AssertEquals('temperature->penalty p0', B0, Row.Raw[0], 1e-6);
    AssertEquals('temperature->penalty p1', B1, Row.Raw[1], 1e-6);

    // The two orders differ materially (the sharpened chain A penalizes
    // token 0 harder: exp(-a)^2 after squaring vs exp(-a) once).
    AssertTrue('processor order changes the distribution',
      Abs(A0 - B0) > 0.05);
    AssertTrue('penalty-then-temperature suppresses the seen token harder',
      A0 < B0);
  finally
    ChainB.Free;
    ChainA.Free;
    PenB.Free;
    PenA.Free;
    Row.Free;
  end;
end;

// REQUIRED BIT-IDENTITY: a nil chain, an EMPTY chain, the Temperature=1.0
// overload and an all-defaults TGenerationConfig all reproduce the plain
// GenerateTokensStreamed path token-for-token.
procedure TTestNeuralDecode.TestNoOpChainAndConfigBitIdenticalToPlainPath;
const
  SeqLen = 10;
  PromptLen = 3;
var
  Full, Twin: TNNet;
  Session: TNNetStreamingDecoder;
  PlainToks, OtherToks: TNeuralIntegerArray;
  EmptyChain: TNNetLogitsProcessorChain;
  Config: TGenerationConfig;
  PlainLen, OtherLen, T, VariantIdx: integer;
  VariantName: string;
begin
  RandSeed := 424242;
  Full := BuildTinySSMLM(SeqLen);
  Twin := BuildTinySSMLM(1);
  Session := nil;
  EmptyChain := TNNetLogitsProcessorChain.Create();
  try
    Twin.CopyWeights(Full);
    Session := TNNetStreamingDecoder.Create(Twin, SeqLen);
    SetLength(PlainToks, PromptLen);
    PlainToks[0] := 5; PlainToks[1] := 2; PlainToks[2] := 9;
    PlainLen := GenerateTokensStreamed(Session, PlainToks, PromptLen,
      SeqLen - PromptLen, SeqLen);
    AssertTrue('plain path generated something', PlainLen > PromptLen);

    for VariantIdx := 0 to 3 do
    begin
      SetLength(OtherToks, 0);
      SetLength(OtherToks, PromptLen);
      OtherToks[0] := 5; OtherToks[1] := 2; OtherToks[2] := 9;
      case VariantIdx of
        0: begin
             VariantName := 'nil chain';
             OtherLen := GenerateTokensStreamedWithProcessors(Session,
               OtherToks, PromptLen, SeqLen - PromptLen, SeqLen, nil, nil,
               nil, nil);
           end;
        1: begin
             VariantName := 'empty chain';
             OtherLen := GenerateTokensStreamedWithProcessors(Session,
               OtherToks, PromptLen, SeqLen - PromptLen, SeqLen, nil,
               EmptyChain, nil, nil);
           end;
        2: begin
             VariantName := 'Temperature=1.0 overload';
             OtherLen := GenerateTokensStreamed(Session, OtherToks,
               PromptLen, SeqLen - PromptLen, SeqLen, nil, nil, nil, nil,
               1.0);
           end;
        else
           begin
             VariantName := 'default TGenerationConfig';
             Config := DefaultGenerationConfig(SeqLen - PromptLen, SeqLen);
             OtherLen := GenerateTokensWithConfig(Session, OtherToks,
               PromptLen, Config);
           end;
      end;
      AssertEquals(VariantName + ': same length', PlainLen, OtherLen);
      for T := PromptLen to PlainLen - 1 do
        AssertEquals(VariantName + ': token at ' + IntToStr(T),
          PlainToks[T], OtherToks[T]);
    end;
  finally
    EmptyChain.Free;
    Session.Free;
    Twin.Free;
    Full.Free;
  end;
end;

// REQUIRED EQUIVALENCE: a TGenerationConfig with penalty + temperature +
// top-k sampler reproduces the hand-assembled GenerateTokensStreamed
// Temperature-overload call (same RNG seed -> same stochastic draws).
procedure TTestNeuralDecode.TestGenerateWithConfigMatchesHandAssembled;
const
  SeqLen = 12;
  PromptLen = 3;
var
  Full, Twin: TNNet;
  Session: TNNetStreamingDecoder;
  Penalty: TNNetTokenHistoryPenalty;
  Sampler: TNNetSamplerBase;
  HandToks, CfgToks: TNeuralIntegerArray;
  Config: TGenerationConfig;
  HandLen, CfgLen, T: integer;
begin
  RandSeed := 424242;
  Full := BuildTinySSMLM(SeqLen);
  Twin := BuildTinySSMLM(1);
  // The pipeline operates on POST-SOFTMAX probabilities: cap both twins
  // with a per-position softmax (weightless, so CopyWeights stays exact).
  Full.AddLayer(TNNetPointwiseSoftMax.Create());
  Twin.AddLayer(TNNetPointwiseSoftMax.Create());
  Session := nil;
  Penalty := TNNetTokenHistoryPenalty.Create(1.0, 0.5, 0.0);
  Sampler := TNNetSamplerTopK.Create(3);
  try
    Twin.CopyWeights(Full);
    Session := TNNetStreamingDecoder.Create(Twin, SeqLen);

    SetLength(HandToks, PromptLen);
    HandToks[0] := 4; HandToks[1] := 9; HandToks[2] := 6;
    RandSeed := 5000; // fix the sampler's draws
    HandLen := GenerateTokensStreamed(Session, HandToks, PromptLen,
      SeqLen - PromptLen, SeqLen, Sampler, Penalty, nil, nil, 0.7);

    SetLength(CfgToks, PromptLen);
    CfgToks[0] := 4; CfgToks[1] := 9; CfgToks[2] := 6;
    Config := DefaultGenerationConfig(SeqLen - PromptLen, SeqLen);
    Config.Penalty := Penalty;
    Config.Temperature := 0.7;
    Config.Sampler := Sampler;
    RandSeed := 5000; // same draws for the config-driven run
    CfgLen := GenerateTokensWithConfig(Session, CfgToks, PromptLen, Config);

    AssertTrue('something was generated', HandLen > PromptLen);
    AssertEquals('config-driven length equals hand-assembled',
      HandLen, CfgLen);
    for T := PromptLen to HandLen - 1 do
      AssertEquals('token at ' + IntToStr(T), HandToks[T], CfgToks[T]);
  finally
    Sampler.Free;
    Penalty.Free;
    Session.Free;
    Twin.Free;
    Full.Free;
  end;
end;

// REQUIRED CONVENTION: Temperature -> 0 with a STOCHASTIC sampler matches the
// greedy argmax path (the distribution degenerates to one-hot, so a weighted
// draw becomes deterministic) - the DecodeSeq2SeqSampled temp-zero behavior
// on the causal-LM side. The sampler is min-p (a TRUE weighted draw over the
// kept mass); top-k would NOT work here - it draws UNIFORMLY among the top K
// tokens, so even a one-hot row stays stochastic under it.
procedure TTestNeuralDecode.TestTemperatureNearZeroWithSamplerMatchesGreedy;
const
  SeqLen = 10;
  PromptLen = 3;
var
  Full, Twin: TNNet;
  Session: TNNetStreamingDecoder;
  Sampler: TNNetSamplerBase;
  GreedyToks, ColdToks: TNeuralIntegerArray;
  GreedyLen, ColdLen, T: integer;
begin
  RandSeed := 424242;
  Full := BuildTinySSMLM(SeqLen);
  Twin := BuildTinySSMLM(1);
  Full.AddLayer(TNNetPointwiseSoftMax.Create());
  Twin.AddLayer(TNNetPointwiseSoftMax.Create());
  Session := nil;
  Sampler := TNNetSamplerMinP.Create(0.5);
  try
    Twin.CopyWeights(Full);
    Session := TNNetStreamingDecoder.Create(Twin, SeqLen);

    SetLength(GreedyToks, PromptLen);
    GreedyToks[0] := 5; GreedyToks[1] := 2; GreedyToks[2] := 9;
    GreedyLen := GenerateTokensStreamed(Session, GreedyToks, PromptLen,
      SeqLen - PromptLen, SeqLen);

    SetLength(ColdToks, PromptLen);
    ColdToks[0] := 5; ColdToks[1] := 2; ColdToks[2] := 9;
    ColdLen := GenerateTokensStreamed(Session, ColdToks, PromptLen,
      SeqLen - PromptLen, SeqLen, Sampler, nil, nil, nil, 1e-12);

    AssertEquals('temp->0 sampled length equals greedy', GreedyLen, ColdLen);
    for T := PromptLen to GreedyLen - 1 do
      AssertEquals('token at ' + IntToStr(T), GreedyToks[T], ColdToks[T]);
  finally
    Sampler.Free;
    Session.Free;
    Twin.Free;
    Full.Free;
  end;
end;

// String-level config path: an all-defaults config reproduces the plain
// GenerateStringStreamed wrapper exactly (same dict, same prompt).
procedure TTestNeuralDecode.TestStringConfigDefaultsMatchPlainWrapper;
const
  SeqLen = 9;
  Words: array[0..11] of string = ('<eos>', '<pad>', 'apple', 'blue', 'cat',
    'dog', 'egg', 'fox', 'gold', 'hat', 'ice', 'jam');
var
  Full, Twin: TNNet;
  Session: TNNetStreamingDecoder;
  Dict: TStringListInt;
  Config: TGenerationConfig;
  T: integer;
  Plain, Got: string;
begin
  RandSeed := 424242;
  Full := BuildTinyCausalLM(SeqLen);
  Twin := BuildTinyCausalLM(1);
  Session := nil;
  Dict := TStringListInt.Create();
  try
    Dict.Sorted := true;
    for T := 0 to High(Words) do Dict.Add(Words[T]);
    Dict.SaveCurrentPosition();
    Twin.CopyWeights(Full);
    Session := TNNetStreamingDecoder.Create(Twin, SeqLen);

    Plain := GenerateStringStreamed(Session, Dict, 'cat dog egg',
      SeqLen - 3, SeqLen);
    Config := DefaultGenerationConfig(SeqLen - 3, SeqLen);
    Got := GenerateStringWithConfig(Session, Dict, 'cat dog egg', Config);
    AssertEquals('config wrapper equals plain wrapper', Plain, Got);
  finally
    Dict.Free;
    Session.Free;
    Twin.Free;
    Full.Free;
  end;
end;

// CFG INVARIANT 1: GuidanceScale = 1 collapses to the conditional-only
// distribution (uncond + 1*(cond-uncond) = cond), so a CFG-guided greedy run
// must reproduce a plain (no-processor) greedy run token-for-token. The
// conditional loop and the unconditional branch use SEPARATE width-1 twins of
// the same weights (one session cannot be driven by two loops at once).
procedure TTestNeuralDecode.TestCFGScaleOneMatchesPlainDecoding;
const
  SeqLen = 12;
  PromptLen = 4;
var
  Full, TwinPlain, TwinCond, TwinUncond: TNNet;
  PlainSession, CondSession, UncondSession: TNNetStreamingDecoder;
  CFG: TNNetCFGProcessor;
  PlainToks, CFGToks: TNeuralIntegerArray;
  NegPrompt: array[0..1] of integer;
  PlainLen, CFGLen, T: integer;
begin
  RandSeed := 424242;
  Full := BuildTinySSMLM(SeqLen);
  // The pipeline operates on POST-SOFTMAX probabilities: cap with a
  // per-position softmax (weightless, so CopyWeights stays exact).
  Full.AddLayer(TNNetPointwiseSoftMax.Create());
  TwinPlain := BuildTinySSMLM(1); TwinPlain.AddLayer(TNNetPointwiseSoftMax.Create());
  TwinCond := BuildTinySSMLM(1); TwinCond.AddLayer(TNNetPointwiseSoftMax.Create());
  TwinUncond := BuildTinySSMLM(1); TwinUncond.AddLayer(TNNetPointwiseSoftMax.Create());
  PlainSession := nil; CondSession := nil; UncondSession := nil; CFG := nil;
  try
    TwinPlain.CopyWeights(Full);
    TwinCond.CopyWeights(Full);
    TwinUncond.CopyWeights(Full);
    PlainSession := TNNetStreamingDecoder.Create(TwinPlain, SeqLen);
    CondSession := TNNetStreamingDecoder.Create(TwinCond, SeqLen);
    UncondSession := TNNetStreamingDecoder.Create(TwinUncond, SeqLen);

    SetLength(PlainToks, PromptLen);
    PlainToks[0] := 5; PlainToks[1] := 2; PlainToks[2] := 9; PlainToks[3] := 4;
    PlainLen := GenerateTokensStreamed(PlainSession, PlainToks, PromptLen,
      SeqLen - PromptLen, SeqLen);
    AssertTrue('plain path generated something', PlainLen > PromptLen);

    NegPrompt[0] := 7; NegPrompt[1] := 3; // arbitrary - ignored at scale 1
    CFG := TNNetCFGProcessor.Create(UncondSession, NegPrompt, 1.0);
    SetLength(CFGToks, PromptLen);
    CFGToks[0] := 5; CFGToks[1] := 2; CFGToks[2] := 9; CFGToks[3] := 4;
    CFGLen := GenerateTokensStreamedWithProcessors(CondSession, CFGToks,
      PromptLen, SeqLen - PromptLen, SeqLen, nil, CFG, nil, nil);

    AssertEquals('scale=1 same generated length', PlainLen, CFGLen);
    for T := PromptLen to PlainLen - 1 do
      AssertEquals('scale=1 token at ' + IntToStr(T), PlainToks[T], CFGToks[T]);
  finally
    CFG.Free;
    UncondSession.Free; CondSession.Free; PlainSession.Free;
    TwinUncond.Free; TwinCond.Free; TwinPlain.Free; Full.Free;
  end;
end;

// CFG INVARIANT 2: GuidanceScale = 0 collapses to the unconditional branch
// (uncond + 0*(cond-uncond) = uncond), so the conditional prompt has NO
// influence: two runs with DIFFERENT conditional prompts but the SAME negative
// prompt must produce identical output. (The model is a stateful SSM, so its
// output genuinely depends on the conditioning context - the test would be
// vacuous on a context-free head.)
procedure TTestNeuralDecode.TestCFGScaleZeroIgnoresConditionalPrompt;
const
  SeqLen = 12;
  PromptLen = 4;
var
  Full, TwinA, TwinB, TwinUA, TwinUB: TNNet;
  SessA, SessB, USessA, USessB: TNNetStreamingDecoder;
  CFGa, CFGb: TNNetCFGProcessor;
  ToksA, ToksB: TNeuralIntegerArray;
  NegPrompt: array[0..2] of integer;
  LenA, LenB, T: integer;
begin
  RandSeed := 424242;
  Full := BuildTinySSMLM(SeqLen);
  Full.AddLayer(TNNetPointwiseSoftMax.Create());
  TwinA := BuildTinySSMLM(1); TwinA.AddLayer(TNNetPointwiseSoftMax.Create());
  TwinB := BuildTinySSMLM(1); TwinB.AddLayer(TNNetPointwiseSoftMax.Create());
  TwinUA := BuildTinySSMLM(1); TwinUA.AddLayer(TNNetPointwiseSoftMax.Create());
  TwinUB := BuildTinySSMLM(1); TwinUB.AddLayer(TNNetPointwiseSoftMax.Create());
  SessA := nil; SessB := nil; USessA := nil; USessB := nil; CFGa := nil; CFGb := nil;
  try
    TwinA.CopyWeights(Full); TwinB.CopyWeights(Full);
    TwinUA.CopyWeights(Full); TwinUB.CopyWeights(Full);
    SessA := TNNetStreamingDecoder.Create(TwinA, SeqLen);
    SessB := TNNetStreamingDecoder.Create(TwinB, SeqLen);
    USessA := TNNetStreamingDecoder.Create(TwinUA, SeqLen);
    USessB := TNNetStreamingDecoder.Create(TwinUB, SeqLen);

    NegPrompt[0] := 6; NegPrompt[1] := 1; NegPrompt[2] := 10; // shared uncond ctx

    // Run A: conditional prompt #1.
    CFGa := TNNetCFGProcessor.Create(USessA, NegPrompt, 0.0);
    SetLength(ToksA, PromptLen);
    ToksA[0] := 5; ToksA[1] := 2; ToksA[2] := 9; ToksA[3] := 4;
    LenA := GenerateTokensStreamedWithProcessors(SessA, ToksA, PromptLen,
      SeqLen - PromptLen, SeqLen, nil, CFGa, nil, nil);

    // Run B: a COMPLETELY DIFFERENT conditional prompt, same negative prompt.
    CFGb := TNNetCFGProcessor.Create(USessB, NegPrompt, 0.0);
    SetLength(ToksB, PromptLen);
    ToksB[0] := 11; ToksB[1] := 8; ToksB[2] := 3; ToksB[3] := 7;
    LenB := GenerateTokensStreamedWithProcessors(SessB, ToksB, PromptLen,
      SeqLen - PromptLen, SeqLen, nil, CFGb, nil, nil);

    AssertTrue('scale=0 generated something', LenA > PromptLen);
    AssertEquals('scale=0 output length is prompt-independent', LenA, LenB);
    for T := PromptLen to LenA - 1 do
      AssertEquals('scale=0 token at ' + IntToStr(T) +
        ' independent of conditional prompt', ToksA[T], ToksB[T]);
  finally
    CFGa.Free; CFGb.Free;
    USessB.Free; USessA.Free; SessB.Free; SessA.Free;
    TwinUB.Free; TwinUA.Free; TwinB.Free; TwinA.Free; Full.Free;
  end;
end;

// CFG construction validation: an unassigned session, an empty negative
// prompt, and a wide (SizeX>1) session are each rejected.
procedure TTestNeuralDecode.TestCFGRejectsInvalidArguments;
var
  Twin, Wide: TNNet;
  Session, WideSession: TNNetStreamingDecoder;
  Neg: array[0..0] of integer;
  Empty: array of integer;
  Raised: boolean;
begin
  RandSeed := 424242;
  Twin := BuildTinySSMLM(1); Twin.AddLayer(TNNetPointwiseSoftMax.Create());
  Wide := BuildTinySSMLM(4); Wide.AddLayer(TNNetPointwiseSoftMax.Create());
  Session := nil; WideSession := nil;
  Neg[0] := 5;
  SetLength(Empty, 0);
  try
    Session := TNNetStreamingDecoder.Create(Twin, 8);
    WideSession := TNNetStreamingDecoder.Create(Wide, 8);

    Raised := false;
    try TNNetCFGProcessor.Create(nil, Neg, 1.5).Free;
    except on EArgumentException do Raised := true; end;
    AssertTrue('nil session rejected', Raised);

    Raised := false;
    try TNNetCFGProcessor.Create(Session, Empty, 1.5).Free;
    except on EArgumentException do Raised := true; end;
    AssertTrue('empty negative prompt rejected', Raised);

    Raised := false;
    try TNNetCFGProcessor.Create(WideSession, Neg, 1.5).Free;
    except on EArgumentException do Raised := true; end;
    AssertTrue('wide session rejected', Raised);
  finally
    WideSession.Free; Session.Free;
    Wide.Free; Twin.Free;
  end;
end;

// TGenerationConfig CFG wiring, INVARIANT 1: a config with GuidanceScale = 1
// (the default) must be bit-for-bit identical to a config that leaves CFG off
// entirely (CFGUncond unassigned) - the CFG branch is never constructed and
// never steps. Both run through GenerateTokensWithConfig.
procedure TTestNeuralDecode.TestConfigCFGScaleOneMatchesNoCFGConfig;
const
  SeqLen = 12;
  PromptLen = 4;
var
  Full, TwinA, TwinB, TwinU, TwinUClone: TNNet;
  SessA, SessB, USess: TNNetStreamingDecoder;
  NoCFG, WithCFG: TGenerationConfig;
  ToksA, ToksB: TNeuralIntegerArray;
  LenA, LenB, T: integer;
begin
  RandSeed := 424242;
  Full := BuildTinySSMLM(SeqLen);
  Full.AddLayer(TNNetPointwiseSoftMax.Create());
  TwinA := BuildTinySSMLM(1); TwinA.AddLayer(TNNetPointwiseSoftMax.Create());
  TwinB := BuildTinySSMLM(1); TwinB.AddLayer(TNNetPointwiseSoftMax.Create());
  TwinU := BuildTinySSMLM(1); TwinU.AddLayer(TNNetPointwiseSoftMax.Create());
  SessA := nil; SessB := nil; USess := nil; TwinUClone := nil;
  try
    TwinA.CopyWeights(Full); TwinB.CopyWeights(Full); TwinU.CopyWeights(Full);
    SessA := TNNetStreamingDecoder.Create(TwinA, SeqLen);
    SessB := TNNetStreamingDecoder.Create(TwinB, SeqLen);
    // Derive the unconditional twin automatically (part (b)).
    USess := MakeUnconditionalTwin(TwinU, SeqLen, TwinUClone);

    SetLength(ToksA, PromptLen);
    ToksA[0] := 5; ToksA[1] := 2; ToksA[2] := 9; ToksA[3] := 4;
    NoCFG := DefaultGenerationConfig(SeqLen - PromptLen, SeqLen);
    LenA := GenerateTokensWithConfig(SessA, ToksA, PromptLen, NoCFG);
    AssertTrue('no-CFG config generated something', LenA > PromptLen);

    SetLength(ToksB, PromptLen);
    ToksB[0] := 5; ToksB[1] := 2; ToksB[2] := 9; ToksB[3] := 4;
    WithCFG := DefaultGenerationConfig(SeqLen - PromptLen, SeqLen);
    WithCFG.GuidanceScale := 1.0; // explicit no-op
    WithCFG.CFGUncond := USess;
    SetLength(WithCFG.NegativePrompt, 2);
    WithCFG.NegativePrompt[0] := 7; WithCFG.NegativePrompt[1] := 3;
    LenB := GenerateTokensWithConfig(SessB, ToksB, PromptLen, WithCFG);

    AssertEquals('scale=1 config same length', LenA, LenB);
    for T := PromptLen to LenA - 1 do
      AssertEquals('scale=1 config token at ' + IntToStr(T),
        ToksA[T], ToksB[T]);
  finally
    USess.Free; SessB.Free; SessA.Free;
    TwinUClone.Free; TwinU.Free; TwinB.Free; TwinA.Free; Full.Free;
  end;
end;

// TGenerationConfig CFG wiring, INVARIANT 2: a config with GuidanceScale <> 1
// must actually engage the processor and CHANGE the result versus an otherwise
// identical scale=1 config (sanity that the per-step pipeline really runs the
// CFG combine). A negative prompt that differs from the conditional prompt is
// used so the two branches disagree.
procedure TTestNeuralDecode.TestConfigCFGScaleChangesOutput;
const
  SeqLen = 16;
  PromptLen = 4;
var
  Full, TwinA, TwinB, TwinUA, TwinUB, CloneA, CloneB: TNNet;
  SessA, SessB, USessA, USessB: TNNetStreamingDecoder;
  CfgOne, CfgGuided: TGenerationConfig;
  ToksOne, ToksGuided: TNeuralIntegerArray;
  LenOne, LenGuided, T: integer;
  Differs: boolean;
begin
  RandSeed := 424242;
  Full := BuildTinySSMLM(SeqLen);
  Full.AddLayer(TNNetPointwiseSoftMax.Create());
  TwinA := BuildTinySSMLM(1); TwinA.AddLayer(TNNetPointwiseSoftMax.Create());
  TwinB := BuildTinySSMLM(1); TwinB.AddLayer(TNNetPointwiseSoftMax.Create());
  TwinUA := BuildTinySSMLM(1); TwinUA.AddLayer(TNNetPointwiseSoftMax.Create());
  TwinUB := BuildTinySSMLM(1); TwinUB.AddLayer(TNNetPointwiseSoftMax.Create());
  SessA := nil; SessB := nil; USessA := nil; USessB := nil;
  CloneA := nil; CloneB := nil;
  try
    TwinA.CopyWeights(Full); TwinB.CopyWeights(Full);
    TwinUA.CopyWeights(Full); TwinUB.CopyWeights(Full);
    SessA := TNNetStreamingDecoder.Create(TwinA, SeqLen);
    SessB := TNNetStreamingDecoder.Create(TwinB, SeqLen);
    USessA := MakeUnconditionalTwin(TwinUA, SeqLen, CloneA);
    USessB := MakeUnconditionalTwin(TwinUB, SeqLen, CloneB);

    // Baseline: scale=1 (CFG is a no-op even though wired).
    SetLength(ToksOne, PromptLen);
    ToksOne[0] := 5; ToksOne[1] := 2; ToksOne[2] := 9; ToksOne[3] := 4;
    CfgOne := DefaultGenerationConfig(SeqLen - PromptLen, SeqLen);
    CfgOne.GuidanceScale := 1.0;
    CfgOne.CFGUncond := USessA;
    SetLength(CfgOne.NegativePrompt, 2);
    CfgOne.NegativePrompt[0] := 11; CfgOne.NegativePrompt[1] := 8;
    LenOne := GenerateTokensWithConfig(SessA, ToksOne, PromptLen, CfgOne);

    // Guided: scale=3, same prompts/negative prompt.
    SetLength(ToksGuided, PromptLen);
    ToksGuided[0] := 5; ToksGuided[1] := 2; ToksGuided[2] := 9; ToksGuided[3] := 4;
    CfgGuided := DefaultGenerationConfig(SeqLen - PromptLen, SeqLen);
    CfgGuided.GuidanceScale := 3.0;
    CfgGuided.CFGUncond := USessB;
    SetLength(CfgGuided.NegativePrompt, 2);
    CfgGuided.NegativePrompt[0] := 11; CfgGuided.NegativePrompt[1] := 8;
    LenGuided := GenerateTokensWithConfig(SessB, ToksGuided, PromptLen,
      CfgGuided);

    AssertTrue('guided run generated something', LenGuided > PromptLen);
    Differs := (LenOne <> LenGuided);
    if not Differs then
      for T := PromptLen to LenOne - 1 do
        if ToksOne[T] <> ToksGuided[T] then Differs := true;
    AssertTrue('GuidanceScale<>1 changes the generated tokens', Differs);
  finally
    USessB.Free; USessA.Free; SessB.Free; SessA.Free;
    CloneB.Free; CloneA.Free;
    TwinUB.Free; TwinUA.Free; TwinB.Free; TwinA.Free; Full.Free;
  end;
end;

// MakeUnconditionalTwin part (b): the auto-derived twin must produce the SAME
// unconditional next-token distribution as the source net on a pinned input
// sequence (the SaveToString -> LoadFromString clone shares the trained
// weights exactly; only the cache state is independent).
procedure TTestNeuralDecode.TestMakeUnconditionalTwinMatchesSourceLogits;
const
  SeqLen = 10;
  Toks: array[0..3] of integer = (6, 1, 9, 4);
var
  Full, Source, Clone: TNNet;
  SrcSess, TwinSess: TNNetStreamingDecoder;
  InV: TNNetVolume;
  SrcRow: array of TNeuralFloat;
  I, P: integer;
begin
  RandSeed := 424242;
  Full := BuildTinySSMLM(SeqLen);
  Full.AddLayer(TNNetPointwiseSoftMax.Create());
  Source := BuildTinySSMLM(1); Source.AddLayer(TNNetPointwiseSoftMax.Create());
  SrcSess := nil; TwinSess := nil; Clone := nil; InV := nil;
  try
    Source.CopyWeights(Full);
    SrcSess := TNNetStreamingDecoder.Create(Source, SeqLen);
    TwinSess := MakeUnconditionalTwin(Source, SeqLen, Clone);

    InV := TNNetVolume.Create(Source.GetFirstLayer().Output);
    InV.Fill(0);

    // Drive both sessions over the SAME pinned input; the final step's output
    // row is the unconditional distribution to compare.
    SrcSess.Reset();
    for P := 0 to High(Toks) do
    begin
      InV.FData[0] := Toks[P];
      SrcSess.StepForward(InV, P);
    end;
    SetLength(SrcRow, SrcSess.Output().Size);
    for I := 0 to SrcSess.Output().Size - 1 do
      SrcRow[I] := SrcSess.Output().Raw[I];

    TwinSess.Reset();
    for P := 0 to High(Toks) do
    begin
      InV.FData[0] := Toks[P];
      TwinSess.StepForward(InV, P);
    end;

    AssertEquals('clone output size matches source',
      Length(SrcRow), TwinSess.Output().Size);
    for I := 0 to High(SrcRow) do
      AssertEquals('clone logit ' + IntToStr(I) + ' matches source',
        SrcRow[I], TwinSess.Output().Raw[I], 1e-6);
  finally
    InV.Free;
    TwinSess.Free; SrcSess.Free;
    Clone.Free; Source.Free; Full.Free;
  end;
end;

// The green-list partition must be a deterministic, bit-reproducible function
// of (Key, PrevToken): the same key+prefix yields the identical green set, a
// different prefix or a different key yields a different one, and the green
// fraction over the vocab is close to Gamma.
procedure TTestNeuralDecode.TestWatermarkGreenListReproducibleFromKeyAndPrefix;
const
  Vocab = 4000;
  Key1: UInt64 = $C0FFEE123456789A;
  Key2: UInt64 = $0123456789ABCDEF;
  Gamma = 0.25;
var
  Tok: integer;
  G_k1_p5a, G_k1_p5b, G_k1_p7, G_k2_p5: boolean;
  GreenCount, DiffPrefix, DiffKey: integer;
begin
  GreenCount := 0;
  DiffPrefix := 0;
  DiffKey := 0;
  for Tok := 0 to Vocab - 1 do
  begin
    // Reproducibility: two independent calls with identical args must agree.
    G_k1_p5a := TNNetWatermarkLogitsProcessor.IsGreen(Key1, 5, Tok, Gamma);
    G_k1_p5b := TNNetWatermarkLogitsProcessor.IsGreen(Key1, 5, Tok, Gamma);
    AssertEquals('green membership reproducible at token ' + IntToStr(Tok),
      G_k1_p5a, G_k1_p5b);
    if G_k1_p5a then Inc(GreenCount);
    // Different prefix token => generally a different partition.
    G_k1_p7 := TNNetWatermarkLogitsProcessor.IsGreen(Key1, 7, Tok, Gamma);
    if G_k1_p7 <> G_k1_p5a then Inc(DiffPrefix);
    // Different key => generally a different partition.
    G_k2_p5 := TNNetWatermarkLogitsProcessor.IsGreen(Key2, 5, Tok, Gamma);
    if G_k2_p5 <> G_k1_p5a then Inc(DiffKey);
  end;
  // Green fraction tracks Gamma (within sampling noise over 4000 draws).
  AssertTrue('green fraction near Gamma (got ' + IntToStr(GreenCount) + '/' +
    IntToStr(Vocab) + ')',
    (GreenCount > Round(0.20 * Vocab)) and (GreenCount < Round(0.30 * Vocab)));
  // A changed prefix and a changed key each flip a substantial fraction of
  // the partition (decorrelated), not a handful of tokens.
  AssertTrue('different prefix changes the partition (' + IntToStr(DiffPrefix) +
    ')', DiffPrefix > Round(0.10 * Vocab));
  AssertTrue('different key changes the partition (' + IntToStr(DiffKey) + ')',
    DiffKey > Round(0.10 * Vocab));
end;

// End-to-end statistical test: a token stream synthesized to follow the green
// list (each token drawn green w.r.t. its predecessor) scores well above the
// detection threshold, while a uniform-random stream scores around zero.
// Detection also depends on the key: the WRONG key does not detect.
procedure TTestNeuralDecode.TestWatermarkDetectsWatermarkedAndRejectsRandom;
const
  Vocab = 500;
  Len = 400;
  Key: UInt64 = $A5A5A5A5DEADBEEF;
  WrongKey: UInt64 = $1111111122222222;
  Gamma = 0.25;
  Threshold = 4.0;
var
  Watermarked, Random: array of integer;
  I, Tok, Prev: integer;
  ZMarked, ZRandom, ZWrongKey: TNeuralFloat;
begin
  RandSeed := 424242;
  SetLength(Watermarked, Len);
  SetLength(Random, Len);
  // First token is unconstrained (no predecessor inside the scored window).
  Watermarked[0] := System.Random(Vocab);
  for I := 1 to Len - 1 do
  begin
    Prev := Watermarked[I - 1];
    // Pick the first green token at a random offset (a greedy watermark would
    // do the same in expectation); guaranteed to find one since ~Gamma*Vocab
    // tokens are green.
    Tok := System.Random(Vocab);
    while not TNNetWatermarkLogitsProcessor.IsGreen(Key, Prev, Tok, Gamma) do
      Tok := (Tok + 1) mod Vocab;
    Watermarked[I] := Tok;
  end;
  for I := 0 to Len - 1 do Random[I] := System.Random(Vocab);

  ZMarked := DetectWatermark(Watermarked, Key, Gamma);
  ZRandom := DetectWatermark(Random, Key, Gamma);
  ZWrongKey := DetectWatermark(Watermarked, WrongKey, Gamma);

  AssertTrue('watermarked text scores above threshold (z=' +
    FloatToStrF(ZMarked, ffFixed, 8, 3) + ')', ZMarked > Threshold);
  AssertTrue('uniform-random text scores below threshold (z=' +
    FloatToStrF(ZRandom, ffFixed, 8, 3) + ')', ZRandom < Threshold);
  AssertTrue('watermark not detectable with the wrong key (z=' +
    FloatToStrF(ZWrongKey, ffFixed, 8, 3) + ')', ZWrongKey < Threshold);
end;

// The processor, in the post-softmax probability domain, must multiply every
// green probability by exp(Delta) and renormalize (the exact image of
// logit += Delta), and ProcessRow must use the PREVIOUS token (seeded in
// Reset, advanced in Commit) to pick the green list.
procedure TTestNeuralDecode.TestWatermarkProcessorBoostsGreenInProbabilityDomain;
const
  Vocab = 64;
  Delta = 2.0;
  Gamma = 0.25;
  Key: UInt64 = $DEADC0DE;
  Prompt: array[0..1] of integer = (3, 9);
var
  Row, Base: TNNetVolume;
  Proc: TNNetWatermarkLogitsProcessor;
  I: integer;
  ExpDelta, Sum, Expected: TNeuralFloat;
begin
  Row := TNNetVolume.Create(Vocab, 1, 1);
  Base := TNNetVolume.Create(Vocab, 1, 1);
  Proc := TNNetWatermarkLogitsProcessor.Create(Key, Gamma, Delta);
  try
    AssertTrue('declares the probability domain', Proc.ExpectsProbabilities());
    // A normalized starting distribution.
    Sum := 0;
    for I := 0 to Vocab - 1 do
    begin
      Base.Raw[I] := 0.5 + I; // arbitrary positive weights
      Sum := Sum + Base.Raw[I];
    end;
    for I := 0 to Vocab - 1 do Base.Raw[I] := Base.Raw[I] / Sum;

    Proc.Reset(Prompt);          // FPrevToken := 9 (last prompt token)
    for I := 0 to Vocab - 1 do Row.Raw[I] := Base.Raw[I];
    Proc.ProcessRow(Row);

    // Verify against the hand-computed image: green *= exp(Delta), renormalize.
    ExpDelta := Exp(Delta);
    Sum := 0;
    for I := 0 to Vocab - 1 do
      if TNNetWatermarkLogitsProcessor.IsGreen(Key, 9, I, Gamma) then
        Sum := Sum + Base.Raw[I] * ExpDelta
      else
        Sum := Sum + Base.Raw[I];
    for I := 0 to Vocab - 1 do
    begin
      if TNNetWatermarkLogitsProcessor.IsGreen(Key, 9, I, Gamma) then
        Expected := Base.Raw[I] * ExpDelta / Sum
      else
        Expected := Base.Raw[I] / Sum;
      AssertEquals('green-boosted prob at token ' + IntToStr(I),
        Expected, Row.Raw[I], 1e-6);
    end;
    // Row remains a valid distribution.
    Sum := 0;
    for I := 0 to Vocab - 1 do Sum := Sum + Row.Raw[I];
    AssertEquals('row sums to 1', 1.0, Sum, 1e-6);

    // Commit advances the predecessor: after emitting token 17 the green list
    // is now seeded by 17, not 9.
    Proc.Commit(17);
    for I := 0 to Vocab - 1 do Row.Raw[I] := Base.Raw[I];
    Proc.ProcessRow(Row);
    for I := 0 to Vocab - 1 do
    begin
      if TNNetWatermarkLogitsProcessor.IsGreen(Key, 17, I, Gamma) then
        Expected := Base.Raw[I] * ExpDelta
      else
        Expected := Base.Raw[I];
      // (unnormalized comparison up to the shared constant is enough to prove
      // the predecessor switched; check the boost direction)
      if TNNetWatermarkLogitsProcessor.IsGreen(Key, 17, I, Gamma) then
        AssertTrue('after Commit green token ' + IntToStr(I) + ' boosted',
          Row.Raw[I] >= Base.Raw[I] - 1e-9)
      else
        AssertTrue('after Commit red token ' + IntToStr(I) + ' not boosted up',
          Row.Raw[I] <= Base.Raw[I] + 1e-9);
    end;
  finally
    Proc.Free;
    Base.Free;
    Row.Free;
  end;
end;

// A single-token bias has an empty prefix, so it fires UNCONDITIONALLY every
// step (transformers' documented degradation). The probability-domain image is
// p *= exp(bias), renormalized.
procedure TTestNeuralDecode.TestSequenceBiasSingleTokenIsUnconditionalBias;
const
  Vocab = 5;
  Bias = 1.5;
var
  Proc: TNNetSequenceBiasProcessor;
  Row: TNNetVolume;
  I: integer;
  ExpBias, Sum, Expected: TNeuralFloat;
begin
  Proc := TNNetSequenceBiasProcessor.Create();
  Row := TNNetVolume.Create(Vocab, 1, 1);
  try
    AssertTrue('declares the probability domain', Proc.ExpectsProbabilities());
    Proc.AddSequenceBias([2], Bias); // single-token: unconditional bias on 2
    AssertEquals('one entry registered', 1, Proc.Count);
    Proc.Reset([0, 1]); // arbitrary history; the empty prefix matches anyway
    for I := 0 to Vocab - 1 do Row.Raw[I] := 0.2;
    Proc.ProcessRow(Row);
    ExpBias := Exp(Bias);
    Sum := (Vocab - 1) * 0.2 + 0.2 * ExpBias;
    for I := 0 to Vocab - 1 do
    begin
      if I = 2 then Expected := 0.2 * ExpBias / Sum
      else Expected := 0.2 / Sum;
      AssertEquals('single-token bias prob at ' + IntToStr(I), Expected,
        Row.Raw[I], 1e-6);
    end;
    Sum := 0;
    for I := 0 to Vocab - 1 do Sum := Sum + Row.Raw[I];
    AssertEquals('row sums to 1', 1.0, Sum, 1e-6);
  finally
    Row.Free;
    Proc.Free;
  end;
end;

// A multi-token entry biases its FINAL token only when the leading tokens are
// the tail of the generated history (prefix match). It must NOT fire when the
// lead-in does not match.
procedure TTestNeuralDecode.TestSequenceBiasMultiTokenFiresOnlyOnPrefixMatch;
const
  Vocab = 5;
var
  Proc: TNNetSequenceBiasProcessor;
  Row: TNNetVolume;
  I: integer;
begin
  Proc := TNNetSequenceBiasProcessor.Create();
  Row := TNNetVolume.Create(Vocab, 1, 1);
  try
    // Bias token 4 only after the bigram (1,3) has been generated.
    Proc.AddSequenceBias([1, 3, 4], 2.0);

    // -- NO match: history tail is (0,2); the (1,3) prefix is absent.
    Proc.Reset([0, 2]);
    for I := 0 to Vocab - 1 do Row.Raw[I] := 0.2;
    Proc.ProcessRow(Row);
    for I := 0 to Vocab - 1 do
      AssertEquals('no prefix match leaves token ' + IntToStr(I) + ' untouched',
        0.2, Row.Raw[I], 1e-6);

    // -- MATCH: history tail is exactly (1,3) -> token 4 gets boosted.
    Proc.Reset([9, 1, 3]); // suffix (1,3) matches the entry's prefix
    for I := 0 to Vocab - 1 do Row.Raw[I] := 0.2;
    Proc.ProcessRow(Row);
    AssertTrue('matched prefix boosts the final token 4 above baseline',
      Row.Raw[4] > 0.2 + 1e-9);
    for I := 0 to 3 do
      AssertTrue('non-final tokens are NOT boosted ' + IntToStr(I),
        Row.Raw[I] < 0.2 + 1e-9);

    // -- MATCH BY COMMIT: prefix completed via emitted tokens, not the prompt.
    Proc.Reset([7]);
    Proc.Commit(1);
    Proc.Commit(3); // history now (7,1,3); suffix (1,3) matches
    for I := 0 to Vocab - 1 do Row.Raw[I] := 0.2;
    Proc.ProcessRow(Row);
    AssertTrue('prefix completed by Commit boosts token 4',
      Row.Raw[4] > 0.2 + 1e-9);
  finally
    Row.Free;
    Proc.Free;
  end;
end;

// End-to-end greedy check: a hard-banned multi-token word never completes under
// argmax decoding. We simulate the streamed greedy loop directly on a model
// distribution that WOULD otherwise emit the banned word, and assert the banned
// final token never wins once its lead-in has been generated.
procedure TTestNeuralDecode.TestSequenceBiasBannedWordNeverAppearsInGreedyOutput;
const
  Vocab = 4;
  Steps = 12;
var
  Proc: TNNetSequenceBiasProcessor;
  Row: TNNetVolume;
  History: array[0..Steps - 1] of integer;
  I, S, Argmax, Prev: integer;
  Best: TNeuralFloat;
begin
  Proc := TNNetSequenceBiasProcessor.Create();
  Row := TNNetVolume.Create(Vocab, 1, 1);
  try
    // Hard-ban the two-token word (1,2): once a 1 is emitted, a 2 must never
    // follow it under greedy decoding.
    Proc.AddBadWord([1, 2]);
    Proc.Reset([0]); // BOS-like prompt
    Prev := -1;
    for S := 0 to Steps - 1 do
    begin
      // A degenerate model that ALWAYS most-favours emitting 1 then 2 (so that
      // without the ban, greedy WOULD produce ...1,2,1,2...). After a 1, token 2
      // has the largest base probability.
      if Prev = 1 then
      begin
        Row.Raw[0] := 0.1; Row.Raw[1] := 0.1; Row.Raw[2] := 0.7; Row.Raw[3] := 0.1;
      end
      else
      begin
        Row.Raw[0] := 0.1; Row.Raw[1] := 0.7; Row.Raw[2] := 0.1; Row.Raw[3] := 0.1;
      end;
      Proc.ProcessRow(Row);
      // Greedy argmax.
      Argmax := 0; Best := Row.Raw[0];
      for I := 1 to Vocab - 1 do
        if Row.Raw[I] > Best then begin Best := Row.Raw[I]; Argmax := I; end;
      History[S] := Argmax;
      Proc.Commit(Argmax);
      Prev := Argmax;
    end;
    // Assert the banned bigram (1,2) never appears in the greedy output.
    for S := 0 to Steps - 2 do
      AssertTrue('banned word (1,2) must not appear at position ' + IntToStr(S),
        not ((History[S] = 1) and (History[S + 1] = 2)));
    // Sanity: token 1 WAS emitted at least once (so the ban genuinely fired on
    // its completion rather than the word never being attempted).
    Argmax := 0;
    for S := 0 to Steps - 1 do if History[S] = 1 then Inc(Argmax);
    AssertTrue('the banned word lead-in (1) was actually generated', Argmax > 0);
  finally
    Row.Free;
    Proc.Free;
  end;
end;

// MaskAllowed must zero every disallowed token and renormalize the allowed
// ones to sum 1, preserving their relative proportions.
procedure TTestNeuralDecode.TestAllowedTokensConstraintMasksAndRenormalizes;
var
  V: TNNetVolume;
  C: TNNetAllowedTokensConstraint;
  I: integer;
  Total: TNeuralFloat;
begin
  V := TNNetVolume.Create(8, 1, 1);
  C := TNNetAllowedTokensConstraint.Create([1, 2]);
  try
    for I := 0 to 7 do V.Raw[I] := 0.125;
    V.Raw[1] := 0.1; V.Raw[2] := 0.2; // allowed mass 0.3 before masking
    C.MaskAllowed(V);
    for I := 0 to 7 do
      if (I <> 1) and (I <> 2) then
        AssertEquals('disallowed token ' + IntToStr(I) + ' zeroed',
          0.0, V.Raw[I], 1e-9);
    AssertEquals('allowed token 1 renormalized', 0.1 / 0.3, V.Raw[1], 1e-6);
    AssertEquals('allowed token 2 renormalized', 0.2 / 0.3, V.Raw[2], 1e-6);
    Total := 0;
    for I := 0 to 7 do Total := Total + V.Raw[I];
    AssertEquals('row sums to 1 after masking', 1.0, Total, 1e-6);
    AssertTrue('whitelist membership', C.TokenAllowed(2));
    AssertTrue('out-of-whitelist id', not C.TokenAllowed(5));
    AssertTrue('out-of-range id', not C.TokenAllowed(123));
  finally
    C.Free;
    V.Free;
  end;
end;

// The documented fallback: when the allowed probability mass is zero (no
// token allowed, or every allowed token at probability 0), the row is left
// UNTOUCHED rather than zeroed (whose argmax would degenerate to token 0).
procedure TTestNeuralDecode.TestConstraintMaskFallbackLeavesRowUntouched;
var
  V: TNNetVolume;
  CEmpty, CZeroMass: TNNetAllowedTokensConstraint;
  I: integer;
begin
  V := TNNetVolume.Create(6, 1, 1);
  CEmpty := TNNetAllowedTokensConstraint.Create([]);
  CZeroMass := TNNetAllowedTokensConstraint.Create([5]);
  try
    for I := 0 to 4 do V.Raw[I] := 0.2;
    V.Raw[5] := 0; // the only whitelisted token has zero probability
    CEmpty.MaskAllowed(V);
    for I := 0 to 4 do
      AssertEquals('empty whitelist leaves element ' + IntToStr(I),
        0.2, V.Raw[I], 1e-6);
    CZeroMass.MaskAllowed(V);
    for I := 0 to 4 do
      AssertEquals('zero allowed mass leaves element ' + IntToStr(I),
        0.2, V.Raw[I], 1e-6);
    AssertEquals('zero allowed mass leaves the allowed element too',
      0.0, V.Raw[5], 1e-9);
  finally
    CZeroMass.Free;
    CEmpty.Free;
    V.Free;
  end;
end;

// The constrained overload with Constraint=nil must produce exactly the same
// stream (length and tokens) as the plain overload - fixed seed, same net.
procedure TTestNeuralDecode.TestGenerateTokensStreamedNilConstraintMatchesPlain;
const
  SeqLen = 10;
  PromptLen = 3;
var
  Full, Twin: TNNet;
  Session: TNNetStreamingDecoder;
  PlainToks, ConToks: TNeuralIntegerArray;
  PlainLen, ConLen, T: integer;
begin
  RandSeed := 424242;
  Full := BuildTinySSMLM(SeqLen);
  Twin := BuildTinySSMLM(1);
  Session := nil;
  try
    Twin.CopyWeights(Full);
    Session := TNNetStreamingDecoder.Create(Twin, SeqLen);
    SetLength(PlainToks, PromptLen);
    SetLength(ConToks, PromptLen);
    PlainToks[0] := 5; PlainToks[1] := 2; PlainToks[2] := 9;
    for T := 0 to PromptLen - 1 do ConToks[T] := PlainToks[T];

    PlainLen := GenerateTokensStreamed(Session, PlainToks, PromptLen,
      SeqLen - PromptLen, SeqLen);
    ConLen := GenerateTokensStreamed(Session, ConToks, PromptLen,
      SeqLen - PromptLen, SeqLen, nil, nil, nil, nil);

    AssertEquals('same length with nil constraint', PlainLen, ConLen);
    for T := PromptLen to PlainLen - 1 do
      AssertEquals('token at pos ' + IntToStr(T), PlainToks[T], ConToks[T]);
  finally
    Session.Free;
    Twin.Free;
    Full.Free;
  end;
end;

// A static whitelist on the streamed path: every generated token must come
// from the whitelist; EOS is NOT whitelisted, so generation runs to the cap.
procedure TTestNeuralDecode.TestGenerateTokensStreamedWhitelistOnlyEmitsAllowed;
const
  SeqLen = 12;
  PromptLen = 2;
var
  Full, Twin: TNNet;
  Session: TNNetStreamingDecoder;
  Whitelist: TNNetAllowedTokensConstraint;
  Toks: TNeuralIntegerArray;
  OutLen, T: integer;
begin
  RandSeed := 424242;
  Full := BuildTinySSMLM(SeqLen);
  Twin := BuildTinySSMLM(1);
  // The constraint path expects POST-SOFTMAX probabilities (the masking
  // renormalizes a probability row): cap both twins with a per-position
  // softmax, exactly like the penalty test.
  Full.AddLayer(TNNetPointwiseSoftMax.Create());
  Twin.AddLayer(TNNetPointwiseSoftMax.Create());
  Session := nil;
  Whitelist := TNNetAllowedTokensConstraint.Create([5, 8]);
  try
    Twin.CopyWeights(Full);
    Session := TNNetStreamingDecoder.Create(Twin, SeqLen);
    SetLength(Toks, PromptLen);
    Toks[0] := 3; Toks[1] := 9;
    OutLen := GenerateTokensStreamed(Session, Toks, PromptLen,
      SeqLen - PromptLen, SeqLen, nil, nil, nil, Whitelist);
    AssertEquals('EOS blocked: generation runs to the cap', SeqLen, OutLen);
    for T := PromptLen to OutLen - 1 do
      AssertTrue('token at pos ' + IntToStr(T) + ' is whitelisted (got ' +
        IntToStr(Toks[T]) + ')', (Toks[T] = 5) or (Toks[T] = 8));
  finally
    Whitelist.Free;
    Session.Free;
    Twin.Free;
    Full.Free;
  end;
end;

// Forced-sequence (trie) constraint: the head is rigged so unconstrained
// greedy always emits 'dog' - which starts NO candidate. The constraint must
// force generation down a candidate it would never pick ('cat apple'),
// complete exactly that one candidate, and then allow EOS.
procedure TTestNeuralDecode.TestForcedSequenceConstraintFollowsCandidate;
const
  SeqLen = 10;
  Words: array[0..11] of string = ('<eos>', '<pad>', 'apple', 'blue', 'cat',
    'dog', 'egg', 'fox', 'gold', 'hat', 'ice', 'jam');
var
  Full, Twin: TNNet;
  Session: TNNetStreamingDecoder;
  Dict: TStringListInt;
  Forced: TNNetForcedSequenceConstraint;
  Toks, FreeToks, CatToks, AppleToks, DogToks: TNeuralIntegerArray;
  PromptLen, OutLen, FreeLen, T: integer;
begin
  RandSeed := 424242;
  Full := BuildTinySSMLM(SeqLen);
  Twin := BuildTinySSMLM(1);
  Session := nil;
  Forced := nil;
  Dict := TStringListInt.Create();
  try
    // With LazUtils linked, locale-aware collation would sort '<eos>'/'<pad>'
    // AFTER the alphabetic words, breaking the "special ids < 2" convention
    // this test (and the constraint's EOS handling) relies on.
    Dict.UseLocale := false;
    Dict.Sorted := true;
    for T := 0 to High(Words) do Dict.Add(Words[T]);
    Dict.SaveCurrentPosition();
    Dict.Tokenize('dog', DogToks);
    Dict.Tokenize('cat', CatToks);
    Dict.Tokenize('apple', AppleToks);
    AssertEquals('special tokens keep ids < 2 (UseLocale=false)',
      2, AppleToks[0]);
    // Rig the logit head to constant 'dog', then cap with softmax (the
    // constraint renormalizes a probability row).
    RigHeadToConstantToken(Full, DogToks[0]);
    Full.AddLayer(TNNetPointwiseSoftMax.Create());
    Twin.AddLayer(TNNetPointwiseSoftMax.Create());
    Twin.CopyWeights(Full);
    Session := TNNetStreamingDecoder.Create(Twin, SeqLen);
    Forced := TNNetForcedSequenceConstraint.Create(Dict,
      ['cat apple', 'egg jam']);

    // Unconstrained reference: greedy emits 'dog' (not any candidate start).
    Dict.Tokenize('blue hat', FreeToks);
    PromptLen := Length(FreeToks);
    FreeLen := GenerateTokensStreamed(Session, FreeToks, PromptLen,
      SeqLen - PromptLen, SeqLen);
    AssertTrue('reference generated something', FreeLen > PromptLen);
    AssertEquals('unconstrained greedy picks the rigged token',
      DogToks[0], FreeToks[PromptLen]);

    // Constrained run: forced down 'cat apple' ('cat' < 'egg' in id order,
    // probabilities tie at the rigged head), then EOS.
    Dict.Tokenize('blue hat', Toks);
    OutLen := GenerateTokensStreamed(Session, Toks, PromptLen,
      SeqLen - PromptLen, SeqLen, nil, nil, nil, Forced);
    AssertEquals('candidate + EOS generated', PromptLen + 3, OutLen);
    AssertEquals('first forced token is cat', CatToks[0], Toks[PromptLen]);
    AssertEquals('second forced token is apple',
      AppleToks[0], Toks[PromptLen + 1]);
    AssertTrue('then a special/EOS token', Toks[PromptLen + 2] < 2);
    AssertTrue('constraint reports completion', Forced.Completed());
  finally
    Forced.Free;
    Dict.Free;
    Session.Free;
    Twin.Free;
    Full.Free;
  end;
end;

// PrepareTokenHealing without a model: on a vocabulary holding 'egg',
// 'egging' and 'eggs', a prompt ending in 'egg' must yield a one-shot
// constraint allowing EXACTLY the egg-prefixed tokens, decrement PromptLen
// and report the dropped id; Commit lifts the restriction and Reset re-arms
// it. Healing must be SKIPPED (nil, PromptLen untouched) for a 1-token
// prompt and for a last token without any strict extension.
procedure TTestNeuralDecode.TestPrepareTokenHealingBuildsPrefixSetAndTrims;
const
  Words: array[0..11] of string = ('<eos>', '<pad>', 'blue', 'cat', 'dog',
    'egg', 'egging', 'eggs', 'fox', 'gold', 'hat', 'ice');
var
  Dict: TStringListInt;
  Toks: TNeuralIntegerArray;
  C: TNNetTokenHealingConstraint;
  PromptLen, Dropped, T: integer;
begin
  Dict := TStringListInt.Create();
  C := nil;
  try
    Dict.Sorted := true;
    for T := 0 to High(Words) do Dict.Add(Words[T]);
    Dict.SaveCurrentPosition();

    Dict.Tokenize('cat egg', Toks);
    PromptLen := Length(Toks);
    AssertEquals('prompt tokenizes to 2 ids', 2, PromptLen);
    C := PrepareTokenHealing(Dict, Toks, PromptLen, Dropped);
    AssertTrue('healing applies', C <> nil);
    AssertEquals('prompt len trimmed by one', 1, PromptLen);
    AssertEquals('dropped id is egg', Dict.WordToIndex('egg'), Dropped);
    for T := 0 to Dict.GetVocabCount() - 1 do
      AssertEquals('step-1 allowance of "' + Dict.DeTokenize(T) + '"',
        Copy(Dict.DeTokenize(T), 1, 3) = 'egg', C.TokenAllowed(T));
    C.Commit(Dict.WordToIndex('eggs'));
    AssertTrue('constraint lifted after the first emission',
      C.TokenAllowed(Dict.WordToIndex('dog')));
    C.Reset([]);
    AssertTrue('Reset re-arms the one-shot restriction',
      not C.TokenAllowed(Dict.WordToIndex('dog')));
    FreeAndNil(C);

    // 1-token prompt: healing would empty it - skipped.
    Dict.Tokenize('egg', Toks);
    PromptLen := Length(Toks);
    C := PrepareTokenHealing(Dict, Toks, PromptLen, Dropped);
    AssertTrue('1-token prompt is not healed', C = nil);
    AssertEquals('prompt len untouched', 1, PromptLen);

    // Last token with no strict extension: provable no-op - skipped.
    Dict.Tokenize('cat hat', Toks);
    PromptLen := Length(Toks);
    C := PrepareTokenHealing(Dict, Toks, PromptLen, Dropped);
    AssertTrue('no strict extension - not healed', C = nil);
    AssertEquals('prompt len untouched without extensions', 2, PromptLen);
  finally
    C.Free;
    Dict.Free;
  end;
end;

// Token-healing follow-up (b) on the dict path: guidance-style MULTI-TOKEN
// rollback. On a vocab holding 'ab','c','abc','abcd', a prompt ending in the
// two tokens 'ab','c' is NOT healed by the default single-token rollback
// ('c' alone has no strict extension - provable no-op), but a 2-token
// rollback rebuilds the boundary fragment 'abc', which 'abcd' strictly
// extends -> healing applies, allows exactly {'abc','abcd'}, trims PromptLen
// by two and reports the first dropped id ('ab'). The default path stays
// bit-identical to v1.
procedure TTestNeuralDecode.TestTokenHealingMultiTokenRollback;
const
  Words: array[0..6] of string = ('<eos>', '<pad>', 'ab', 'abc', 'abcd',
    'c', 'z');
var
  Dict: TStringListInt;
  Toks: TNeuralIntegerArray;
  C: TNNetTokenHealingConstraint;
  PromptLen, Dropped, T: integer;
begin
  Dict := TStringListInt.Create();
  C := nil;
  try
    Dict.Sorted := true;
    for T := 0 to High(Words) do Dict.Add(Words[T]);
    Dict.SaveCurrentPosition();

    // Prompt 'z ab c' -> ids [z, ab, c].
    Dict.Tokenize('z ab c', Toks);
    PromptLen := Length(Toks);
    AssertEquals('prompt tokenizes to 3 ids', 3, PromptLen);

    // Default single-token rollback: 'c' has no strict extension -> skipped.
    C := PrepareTokenHealing(Dict, Toks, PromptLen, Dropped);
    AssertTrue('single-token rollback is a no-op', C = nil);
    AssertEquals('single-token leaves PromptLen', 3, PromptLen);

    // Two-token rollback heals 'ab'+'c' -> 'abc' (extended by 'abcd').
    PromptLen := Length(Toks);
    C := PrepareTokenHealing(Dict, Toks, PromptLen, Dropped, 2);
    AssertTrue('two-token rollback heals', C <> nil);
    AssertEquals('PromptLen trimmed by two', 1, PromptLen);
    AssertEquals('first dropped id is ab', Dict.WordToIndex('ab'), Dropped);
    AssertTrue('allows abc', C.TokenAllowed(Dict.WordToIndex('abc')));
    AssertTrue('allows abcd', C.TokenAllowed(Dict.WordToIndex('abcd')));
    AssertTrue('disallows c',
      not C.TokenAllowed(Dict.WordToIndex('c')));
    AssertTrue('disallows z',
      not C.TokenAllowed(Dict.WordToIndex('z')));
  finally
    C.Free;
    Dict.Free;
  end;
end;

// End-to-end token healing on a rigged head whose healed and unhealed
// first-token distributions PROVABLY differ: probabilities are ordered
// dog > eggs > egg > egging, so an unhealed greedy run from 'cat egg' emits
// 'dog' first, while the healed run drops 'egg', masks step 1 to the
// egg-prefixed tokens and emits 'eggs' (the argmax of the allowed set),
// then continues unconstrained with 'dog'. Pinned both at the token level
// (PrepareTokenHealing + Constraint overload) and at the string level
// (TGenerationConfig.TokenHealing through GenerateStringWithConfig).
procedure TTestNeuralDecode.TestTokenHealingChangesFirstTokenVsUnhealed;
const
  SeqLen = 9;
  Words: array[0..11] of string = ('<eos>', '<pad>', 'blue', 'cat', 'dog',
    'egg', 'egging', 'eggs', 'fox', 'gold', 'hat', 'ice');
var
  Full, Twin: TNNet;
  Session: TNNetStreamingDecoder;
  Dict: TStringListInt;
  Head: TNNetLayer;
  UToks, HToks: TNeuralIntegerArray;
  C: TNNetTokenHealingConstraint;
  Config: TGenerationConfig;
  PromptLen, Dropped, T, ULen, HLen: integer;
  Unhealed, Healed: string;
begin
  RandSeed := 424242;
  Full := BuildTinyCausalLM(SeqLen);
  Twin := BuildTinyCausalLM(1);
  Session := nil;
  C := nil;
  Dict := TStringListInt.Create();
  try
    Dict.Sorted := true;
    for T := 0 to High(Words) do Dict.Add(Words[T]);
    Dict.SaveCurrentPosition();
    AssertEquals('vocab matches the tiny LM', csStreamVocab,
      Dict.GetVocabCount());
    // Rig the head to a CONSTANT graded distribution:
    // dog > eggs > egg > egging > everything else.
    Head := Full.GetLastLayer();
    for T := 0 to Head.Neurons.Count - 1 do
    begin
      Head.Neurons[T].Weights.Fill(0);
      Head.Neurons[T].BiasWeight := 0;
    end;
    Head.Neurons[Dict.WordToIndex('dog')].BiasWeight := 5;
    Head.Neurons[Dict.WordToIndex('eggs')].BiasWeight := 3;
    Head.Neurons[Dict.WordToIndex('egg')].BiasWeight := 2;
    Head.Neurons[Dict.WordToIndex('egging')].BiasWeight := 1;
    Head.MulWeights(1.0); // no-op refresh of packed weight copies
    Full.AddLayer(TNNetPointwiseSoftMax.Create());
    Twin.AddLayer(TNNetPointwiseSoftMax.Create());
    Twin.CopyWeights(Full);
    Session := TNNetStreamingDecoder.Create(Twin, SeqLen);

    // Token level. Unhealed greedy baseline: 'dog' twice.
    Dict.Tokenize('cat egg', UToks);
    PromptLen := Length(UToks);
    AssertEquals('prompt tokenizes to 2 ids', 2, PromptLen);
    ULen := GenerateTokensStreamed(Session, UToks, PromptLen, 2, SeqLen);
    AssertEquals('unhealed generates 2 tokens', PromptLen + 2, ULen);
    AssertEquals('unhealed first token is dog',
      Dict.WordToIndex('dog'), UToks[PromptLen]);
    // Healed: drop 'egg', constrain step 1 to its extensions.
    Dict.Tokenize('cat egg', HToks);
    PromptLen := Length(HToks);
    C := PrepareTokenHealing(Dict, HToks, PromptLen, Dropped);
    AssertTrue('healing constraint built', C <> nil);
    AssertEquals('healed prompt is 1 token', 1, PromptLen);
    HLen := GenerateTokensStreamed(Session, HToks, PromptLen, 2, SeqLen,
      nil, nil, nil, C);
    AssertEquals('healed generates 2 tokens', PromptLen + 2, HLen);
    AssertEquals('healed first token is eggs (argmax of the allowed set)',
      Dict.WordToIndex('eggs'), HToks[PromptLen]);
    AssertTrue('healed and unhealed first tokens differ',
      HToks[1] <> UToks[2]);
    AssertEquals('healed second step is unconstrained again (dog)',
      Dict.WordToIndex('dog'), HToks[PromptLen + 1]);

    // String level: TGenerationConfig.TokenHealing, pinned results.
    Config := DefaultGenerationConfig(2, SeqLen);
    Unhealed := GenerateStringWithConfig(Session, Dict, 'cat egg', Config);
    AssertEquals('unhealed string pinned', 'cat egg dog dog', Unhealed);
    Config.TokenHealing := true;
    Healed := GenerateStringWithConfig(Session, Dict, 'cat egg', Config);
    AssertEquals('healed string pinned', 'cat eggs dog', Healed);
    AssertTrue('healed text differs from the unhealed run',
      Healed <> Unhealed);
  finally
    C.Free;
    Dict.Free;
    Session.Free;
    Twin.Free;
    Full.Free;
  end;
end;

// Driving the automaton directly: object context transitions. After '{' only
// a key string, '}' or whitespace may follow - never ']' or a bare value.
procedure TTestNeuralDecode.TestJSONStateMachineObjectTransitions;
var
  M: TNNetJSONStateMachine;
begin
  M := TNNetJSONStateMachine.Create();
  try
    AssertTrue('open object', M.FeedChar('{'));
    AssertTrue('after { a key quote is allowed', M.CharAllowed('"'));
    AssertTrue('after { an immediate } is allowed', M.CharAllowed('}'));
    AssertTrue('after { whitespace is allowed', M.CharAllowed(' '));
    AssertTrue('after { ] is NOT allowed', not M.CharAllowed(']'));
    AssertTrue('after { a digit is NOT allowed', not M.CharAllowed('5'));
    AssertTrue('after { a comma is NOT allowed', not M.CharAllowed(','));
    AssertTrue('key string', M.FeedString('"a"'));
    AssertTrue('after key only a colon', M.CharAllowed(':'));
    AssertTrue('no } right after a key', not M.CharAllowed('}'));
    AssertTrue('no , right after a key', not M.CharAllowed(','));
    AssertTrue('colon', M.FeedChar(':'));
    AssertTrue('value position accepts a digit', M.CharAllowed('5'));
    AssertTrue('value position accepts a string', M.CharAllowed('"'));
    AssertTrue('no } right after the colon', not M.CharAllowed('}'));
    AssertTrue('number value', M.FeedChar('1'));
    AssertTrue('} closes (and first completes the number)',
      M.FeedChar('}'));
    AssertTrue('complete top-level object', M.IsComplete());
    AssertEquals('stack fully popped', 0, M.StackDepth());
    // No trailing comma: '{"a":1,' must require another key.
    M.Reset();
    AssertTrue('reopen', M.FeedString('{"a":1,'));
    AssertTrue('after , a key is required', M.CharAllowed('"'));
    AssertTrue('no } after a trailing comma', not M.CharAllowed('}'));
  finally
    M.Free;
  end;
end;

// String state: most characters are content; an unescaped quote closes;
// raw control characters are rejected; escapes are validated.
procedure TTestNeuralDecode.TestJSONStateMachineStringAndEscapes;
var
  M: TNNetJSONStateMachine;
begin
  M := TNNetJSONStateMachine.Create();
  try
    AssertTrue('open string', M.FeedChar('"'));
    AssertTrue('plain char allowed', M.CharAllowed('x'));
    AssertTrue('structural chars are plain content in a string',
      M.CharAllowed('{'));
    AssertTrue('closing quote allowed', M.CharAllowed('"'));
    AssertTrue('raw newline rejected in a string', not M.CharAllowed(#10));
    AssertTrue('raw control char rejected', not M.CharAllowed(#1));
    AssertTrue('backslash starts an escape', M.FeedChar('\'));
    AssertTrue('\n escape allowed', M.CharAllowed('n'));
    AssertTrue('escaped quote allowed', M.CharAllowed('"'));
    AssertTrue('unicode escape allowed', M.CharAllowed('u'));
    AssertTrue('invalid escape char rejected', not M.CharAllowed('x'));
    AssertTrue('take the unicode escape', M.FeedChar('u'));
    AssertTrue('hex digit required', not M.CharAllowed('g'));
    AssertTrue('4 hex digits', M.FeedString('0Ab9'));
    AssertTrue('back in the string: content again', M.CharAllowed('x'));
    AssertTrue('unescaped quote closes', M.FeedChar('"'));
    AssertTrue('top-level string value complete', M.IsComplete());
  finally
    M.Free;
  end;
end;

// Number grammar: [-] int frac? exp? with no leading zeros, mandatory digits
// after '-', '.' and 'e'; an unterminated complete number at top level
// counts as a complete value.
procedure TTestNeuralDecode.TestJSONStateMachineNumbers;
var
  M: TNNetJSONStateMachine;
begin
  M := TNNetJSONStateMachine.Create();
  try
    AssertTrue('minus', M.FeedChar('-'));
    AssertTrue('- alone is not complete', not M.IsComplete());
    AssertTrue('-- rejected', not M.CharAllowed('-'));
    AssertTrue('-] rejected', not M.CharAllowed(']'));
    AssertTrue('digit after minus', M.FeedChar('0'));
    AssertTrue('-0 is complete', M.IsComplete());
    AssertTrue('no digits after a leading zero', not M.CharAllowed('1'));
    AssertTrue('fraction allowed after 0', M.FeedChar('.'));
    AssertTrue('. needs a digit, not e', not M.CharAllowed('e'));
    AssertTrue('-0. is not complete', not M.IsComplete());
    AssertTrue('fraction digit', M.FeedChar('5'));
    AssertTrue('-0.5 is complete', M.IsComplete());
    AssertTrue('exponent marker', M.FeedChar('e'));
    AssertTrue('-0.5e is not complete', not M.IsComplete());
    AssertTrue('exponent sign allowed', M.FeedChar('+'));
    AssertTrue('second sign rejected', not M.CharAllowed('+'));
    AssertTrue('exponent digit', M.FeedChar('2'));
    AssertTrue('-0.5e+2 is complete', M.IsComplete());
    // Plain int and the leading-zero rule from a fresh start.
    M.Reset();
    AssertTrue('42', M.FeedString('42'));
    AssertTrue('42 is a complete top-level value', M.IsComplete());
    AssertTrue('no second top-level value', not M.CharAllowed('"'));
    M.Reset();
    AssertTrue('0', M.FeedChar('0'));
    AssertTrue('01 is rejected (leading zero)', not M.CharAllowed('1'));
  finally
    M.Free;
  end;
end;

// After a complete top-level value, only whitespace may follow (EOS/nothing
// at the token level): no new value, no closer, no digits.
procedure TTestNeuralDecode.TestJSONStateMachineTopLevelCompletionAllowsOnlyWS;
var
  M: TNNetJSONStateMachine;
begin
  M := TNNetJSONStateMachine.Create();
  try
    AssertTrue('top-level string', M.FeedString('"hi"'));
    AssertTrue('complete', M.IsComplete());
    AssertTrue('trailing whitespace ok', M.CharAllowed(' '));
    AssertTrue('no second value {', not M.CharAllowed('{'));
    AssertTrue('no second value "', not M.CharAllowed('"'));
    AssertTrue('no digit', not M.CharAllowed('1'));
    AssertTrue('no stray ]', not M.CharAllowed(']'));
    AssertTrue('no stray }', not M.CharAllowed('}'));
    AssertTrue('feeding whitespace keeps it complete',
      M.FeedChar(' ') and M.IsComplete());
  finally
    M.Free;
  end;
end;

// Balanced-stack closing: each close must match the innermost opener.
procedure TTestNeuralDecode.TestJSONStateMachineBalancedStackClosing;
var
  M: TNNetJSONStateMachine;
begin
  M := TNNetJSONStateMachine.Create();
  try
    AssertTrue('[[{', M.FeedString('[[{'));
    AssertEquals('three open containers', 3, M.StackDepth());
    AssertTrue('innermost is an object: ] rejected', not M.CharAllowed(']'));
    AssertTrue('close the object', M.FeedChar('}'));
    AssertEquals('two left', 2, M.StackDepth());
    AssertTrue('now } is rejected (innermost is an array)',
      not M.CharAllowed('}'));
    AssertTrue('comma then another value is fine', M.CharAllowed(','));
    AssertTrue('close inner array', M.FeedChar(']'));
    AssertTrue('not complete yet', not M.IsComplete());
    AssertTrue('close outer array', M.FeedChar(']'));
    AssertTrue('balanced: complete', M.IsComplete());
    AssertEquals('stack empty', 0, M.StackDepth());
    AssertTrue('no further ]', not M.CharAllowed(']'));
  finally
    M.Free;
  end;
end;

// Token-level constraint over MULTI-CHARACTER (BPE-style) tokens: a token is
// allowed exactly when ALL its characters are legal continuations, validated
// transitively through a cloned automaton; EOS (< 2) only at completion.
procedure TTestNeuralDecode.TestJSONConstraintValidatesMultiCharTokens;
var
  Dict: TStringListInt;
  C: TNNetJSONConstraint;
begin
  Dict := TStringListInt.Create();
  C := nil;
  try
    // Insertion order = token id (no sorting; SaveCurrentPosition snapshots
    // the id->string map the constraint reads via DeTokenize).
    Dict.Add('<eos>');   // 0
    Dict.Add('<pad>');   // 1
    Dict.Add('{"k":');   // 2
    Dict.Add('true');    // 3
    Dict.Add('}');       // 4
    Dict.Add(']');       // 5
    Dict.Add('[1,');     // 6
    Dict.SaveCurrentPosition();
    C := TNNetJSONConstraint.Create(Dict);
    C.Reset([]);
    AssertTrue('multi-char object opener allowed at start', C.TokenAllowed(2));
    AssertTrue('literal value allowed at start', C.TokenAllowed(3));
    AssertTrue('array opener with content allowed at start',
      C.TokenAllowed(6));
    AssertTrue('lone } not allowed at start', not C.TokenAllowed(4));
    AssertTrue('lone ] not allowed at start', not C.TokenAllowed(5));
    AssertTrue('EOS not allowed before any value', not C.TokenAllowed(0));
    C.Commit(2); // '{"k":' - expecting a value now
    AssertTrue('value token allowed after the colon', C.TokenAllowed(3));
    AssertTrue('} not allowed right after the colon', not C.TokenAllowed(4));
    AssertTrue('] never matches an object', not C.TokenAllowed(5));
    AssertTrue('EOS not allowed mid-object', not C.TokenAllowed(0));
    C.Commit(3); // 'true' - value done, object still open
    AssertTrue('} closes the object now', C.TokenAllowed(4));
    AssertTrue('] still illegal', not C.TokenAllowed(5));
    C.Commit(4); // '}' - complete top-level value
    AssertTrue('EOS allowed once complete', C.TokenAllowed(0));
    AssertTrue('PAD allowed once complete', C.TokenAllowed(1));
    AssertTrue('no second value', not C.TokenAllowed(2));
    AssertTrue('machine agrees', C.Machine.IsComplete());
  finally
    C.Free;
    Dict.Free;
  end;
end;

// Fuzz-ish: random walks over the ALLOWED transitions must always yield
// strings fpjson parses once the automaton reports completion (and must
// never reject a character it just reported as allowed).
procedure TTestNeuralDecode.TestJSONStateMachineFuzzRandomWalksStayValid;
const
  Alphabet = '{}[]",:0123456789.eE+-truefalsn x';
  Closers = '"}]1:';
var
  M: TNNetJSONStateMachine;
  Parsed: TJSONData;
  S: string;
  AllowedList: array[1..64] of char;
  Walk, Step, I, AllowedCnt, CompletedCnt: integer;
  C: char;
begin
  RandSeed := 31337;
  M := TNNetJSONStateMachine.Create();
  CompletedCnt := 0;
  try
    for Walk := 1 to 25 do
    begin
      M.Reset();
      S := '';
      for Step := 1 to 300 do
      begin
        if M.IsComplete() then break;
        C := #0;
        // Closing phase (always after step 60, half the time before): take
        // the first allowed char from a closing-priority list so walks
        // terminate instead of nesting forever.
        if (Step > 60) or (Random(2) = 0) then
          for I := 1 to Length(Closers) do
            if M.CharAllowed(Closers[I]) then
            begin
              C := Closers[I];
              break;
            end;
        if C = #0 then
        begin
          // Random pick among the allowed alphabet characters. In a literal
          // state exactly one character is allowed, so walks always advance.
          AllowedCnt := 0;
          for I := 1 to Length(Alphabet) do
            if M.CharAllowed(Alphabet[I]) then
            begin
              Inc(AllowedCnt);
              AllowedList[AllowedCnt] := Alphabet[I];
            end;
          AssertTrue('walk ' + IntToStr(Walk) + ' step ' + IntToStr(Step) +
            ': some character must be allowed (got stuck after "' + S + '")',
            AllowedCnt > 0);
          C := AllowedList[1 + Random(AllowedCnt)];
        end;
        AssertTrue('walk ' + IntToStr(Walk) + ': allowed char ''' + C +
          ''' must be accepted after "' + S + '"', M.FeedChar(C));
        S := S + C;
      end;
      if M.IsComplete() and (S <> '') then
      begin
        Inc(CompletedCnt);
        try
          Parsed := GetJSON(S);
          Parsed.Free;
        except
          on E: Exception do
            Fail('walk ' + IntToStr(Walk) + ' produced invalid JSON "' + S +
              '": ' + E.Message);
        end;
      end;
    end;
    AssertTrue('most walks reach a complete value (got ' +
      IntToStr(CompletedCnt) + '/25)', CompletedCnt >= 15);
  finally
    M.Free;
  end;
end;

// The REQUIRED arithmetic grammar from the tasklist:
//   root ::= term (("+"|"-") term)*   term ::= num   num ::= [0-9]+
// must accept exactly the valid expressions and reject everything else; EOS
// (IsComplete) only after a complete term.
const
  ArithGrammar =
    'root ::= term (("+"|"-") term)*'#10 +
    'term ::= num'#10 +
    'num ::= [0-9]+';

procedure TTestNeuralDecode.TestGrammarArithmeticAcceptsOnlyValidExpressions;
var
  G: TNNetGrammar;
  M: TNNetGrammarMachine;

  function AcceptsComplete(const S: string): boolean;
  begin
    M.Reset();
    Result := M.FeedString(S) and M.IsComplete();
  end;

  function AcceptsPrefix(const S: string): boolean;
  begin
    M.Reset();
    Result := M.FeedString(S);
  end;

begin
  G := TNNetGrammar.Create(ArithGrammar);
  M := TNNetGrammarMachine.Create(G);
  try
    // Valid complete expressions.
    AssertTrue('"1" valid', AcceptsComplete('1'));
    AssertTrue('"42" valid', AcceptsComplete('42'));
    AssertTrue('"12+3" valid', AcceptsComplete('12+3'));
    AssertTrue('"1-2+34" valid', AcceptsComplete('1-2+34'));
    AssertTrue('"0-0-0" valid', AcceptsComplete('0-0-0'));
    // Incomplete but legal prefixes (accepted by FeedString, not complete).
    AssertTrue('"1+" is a legal prefix', AcceptsPrefix('1+'));
    AssertTrue('"1+" is NOT complete', not AcceptsComplete('1+'));
    AssertTrue('"" is NOT complete', not AcceptsComplete(''));
    // Hard rejections.
    AssertTrue('"+1" rejected (no leading op)', not AcceptsPrefix('+1'));
    AssertTrue('"1++2" rejected (double op)', not AcceptsPrefix('1++2'));
    AssertTrue('"1a" rejected (non-grammar char)', not AcceptsPrefix('1a'));
    AssertTrue('"1 + 2" rejected (no whitespace in grammar)',
      not AcceptsPrefix('1 + 2'));
  finally
    M.Free;
    G.Free;
  end;
end;

// A forked machine (CopyFrom) must keep parse progress fully independent of the
// machine it was copied from - the beam copy-on-fork guarantee.
procedure TTestNeuralDecode.TestGrammarMachineForkKeepsIndependentState;
var
  G: TNNetGrammar;
  A, B: TNNetGrammarMachine;
begin
  G := TNNetGrammar.Create(ArithGrammar);
  A := TNNetGrammarMachine.Create(G);
  B := TNNetGrammarMachine.Create(G);
  try
    A.Reset();
    AssertTrue('feed "12"', A.FeedString('12'));
    AssertTrue('"12" is a complete value', A.IsComplete());
    B.CopyFrom(A);                          // fork at "12"
    AssertTrue('A advances past the fork ("+")', A.FeedChar('+'));
    AssertTrue('A is now incomplete after "12+"', not A.IsComplete());
    // B must be untouched by A's advance.
    AssertTrue('B still complete at "12"', B.IsComplete());
    AssertTrue('B (at "12") can take a "+"', B.CharAllowed('+'));
    AssertTrue('A (at "12+") needs a digit, not a "+"',
      A.CharAllowed('7') and (not A.CharAllowed('+')));
    // Drive them to different complete strings.
    AssertTrue('A -> "12+5"', A.FeedString('5') and A.IsComplete());
    AssertTrue('B -> "12-9"', B.FeedString('-9') and B.IsComplete());
  finally
    A.Free;
    B.Free;
    G.Free;
  end;
end;

// TNNetGrammarConstraint over a char-level vocab (id = char code, ids < 2
// special): violating chars are zeroed by MaskAllowed and EOS is allowed only
// at a complete state.
procedure TTestNeuralDecode.TestGrammarConstraintMasksViolationsAndGatesEOS;
var
  C: TNNetGrammarConstraint;
  P: TNNetVolume;
begin
  C := TNNetGrammarConstraint.CreateCharLevel(ArithGrammar, 128);
  P := TNNetVolume.Create(128, 1, 1);
  try
    C.Reset([]);
    // At the start only digits are legal; ops, letters, EOS are not.
    AssertTrue('digit allowed at start', C.TokenAllowed(Ord('5')));
    AssertTrue('"+" not allowed at start', not C.TokenAllowed(Ord('+')));
    AssertTrue('letter not allowed at start', not C.TokenAllowed(Ord('a')));
    AssertTrue('EOS not allowed at start', not C.TokenAllowed(0));
    C.Commit(Ord('1'));
    // After one digit the expression is complete -> EOS allowed; digit/op too.
    AssertTrue('EOS allowed after a complete term', C.TokenAllowed(0));
    AssertTrue('"+" allowed after a term', C.TokenAllowed(Ord('+')));
    AssertTrue('another digit allowed (multi-digit num)',
      C.TokenAllowed(Ord('2')));
    AssertTrue('letter still illegal', not C.TokenAllowed(Ord('a')));
    C.Commit(Ord('+'));
    // After an operator a digit is mandatory; EOS now illegal (incomplete).
    AssertTrue('EOS illegal mid-expression', not C.TokenAllowed(0));
    AssertTrue('digit mandatory after "+"', C.TokenAllowed(Ord('9')));
    AssertTrue('"+" illegal after "+"', not C.TokenAllowed(Ord('+')));
    // MaskAllowed zeroes the illegal mass and keeps the legal one.
    P.Fill(1.0 / 128);
    C.MaskAllowed(P);
    AssertEquals('illegal "a" zeroed', 0.0, P.Raw[Ord('a')], 0.0);
    AssertEquals('illegal "+" zeroed', 0.0, P.Raw[Ord('+')], 0.0);
    AssertEquals('illegal EOS zeroed', 0.0, P.Raw[0], 0.0);
    AssertTrue('legal digit "7" keeps positive mass', P.Raw[Ord('7')] > 0);
  finally
    P.Free;
    C.Free;
  end;
end;

// Exercises every GBNF-subset construct: alternation, grouping, '*', '+', '?',
// classes (incl. negation), '.' and rule references.
procedure TTestNeuralDecode.TestGrammarAlternationGroupingRepetitionOptional;
var
  G: TNNetGrammar;
  M: TNNetGrammarMachine;

  function OK(const S: string): boolean;
  begin
    M.Reset();
    Result := M.FeedString(S) and M.IsComplete();
  end;

begin
  // Optional sign, '+' digits, optional fractional group.
  G := TNNetGrammar.Create('root ::= "-"? [0-9]+ ("." [0-9]+)?');
  M := TNNetGrammarMachine.Create(G);
  try
    AssertTrue('5', OK('5'));
    AssertTrue('-5', OK('-5'));
    AssertTrue('5.25', OK('5.25'));
    AssertTrue('-12.5', OK('-12.5'));
    AssertTrue('"5." rejected (frac needs a digit)', not OK('5.'));
    AssertTrue('"--5" rejected (one optional sign)', not OK('--5'));
    AssertTrue('"" rejected ([0-9]+ needs one)', not OK(''));
  finally
    M.Free;
    G.Free;
  end;
  // Plain alternation of literals.
  G := TNNetGrammar.Create('root ::= "ab" | "cd" | "e"');
  M := TNNetGrammarMachine.Create(G);
  try
    AssertTrue('ab', OK('ab'));
    AssertTrue('cd', OK('cd'));
    AssertTrue('e', OK('e'));
    AssertTrue('"a" incomplete', not OK('a'));
    AssertTrue('"ax" rejected', not OK('ax'));
  finally
    M.Free;
    G.Free;
  end;
  // Negated class and '.' any-char inside delimiters.
  G := TNNetGrammar.Create('root ::= "<" [^>]+ ">" "." ');
  M := TNNetGrammarMachine.Create(G);
  try
    AssertTrue('<hi>.', OK('<hi>.'));
    AssertTrue('"<>." rejected ([^>]+ needs one)', not OK('<>.'));
    AssertTrue('">" inside class rejected', not OK('<a>b>.'));
  finally
    M.Free;
    G.Free;
  end;
end;

// Greedy decode driven by the constraint: at every step pick the highest-prob
// token that survives MaskAllowed, Commit it, stop at EOS. The emitted string
// must always parse, and a synthetic head biased toward an ILLEGAL char must be
// steered to a legal one by the constraint.
procedure TTestNeuralDecode.TestGrammarConstraintGreedyDrivenWalkStaysValid;
var
  C: TNNetGrammarConstraint;
  Verify: TNNetGrammarMachine;
  VG: TNNetGrammar;
  P: TNNetVolume;
  Step, Best, I: integer;
  S: string;
begin
  C := TNNetGrammarConstraint.CreateCharLevel(ArithGrammar, 128);
  VG := TNNetGrammar.Create(ArithGrammar);
  Verify := TNNetGrammarMachine.Create(VG);
  P := TNNetVolume.Create(128, 1, 1);
  try
    C.Reset([]);
    S := '';
    for Step := 1 to 11 do  // odd cap: a "d+d+..." walk ends on a digit
    begin
      // Head that ADORES '+' (an often-illegal char) over the digit, with EOS
      // last - so the unconstrained argmax would emit '+' first (invalid). The
      // constraint must steer every step to a legal token, producing the
      // alternating "7+7+7..." expression.
      P.Fill(0.001);
      P.Raw[Ord('+')] := 5.0;
      P.Raw[Ord('7')] := 4.0;
      P.Raw[0] := 1.0;            // EOS (never wins here while '+'/'7' legal)
      C.MaskAllowed(P);
      Best := 0;
      for I := 0 to P.Size - 1 do
        if P.Raw[I] > P.Raw[Best] then Best := I;
      if Best < 2 then break;     // EOS (only reachable at a complete state)
      AssertTrue('greedy step ' + IntToStr(Step) + ' picked a legal token',
        C.TokenAllowed(Best));
      C.Commit(Best);
      S := S + Chr(Best);
    end;
    // The constrained greedy walk must never have emitted an illegal char; the
    // accumulated string always feeds the grammar (it is at worst a legal
    // prefix). With the odd cap the walk ends after a digit -> complete.
    Verify.Reset();
    AssertTrue('greedy output "' + S + '" feeds the grammar',
      Verify.FeedString(S));
    AssertTrue('greedy output "' + S + '" is a complete expression',
      Verify.IsComplete());
    AssertEquals('greedy steered to the "7+7+..." form', '7+7+7+7+7+7', S);
  finally
    P.Free;
    Verify.Free;
    VG.Free;
    C.Free;
  end;
end;

// Same idea under SAMPLED decoding: a probability-weighted sampler over the
// masked row never draws a zeroed (illegal) token, and every completed run
// parses. (TNNetSamplerWeightedTopK samples proportional to mass, so masked-out
// tokens have zero draw probability - unlike TNNetSamplerTopK's uniform top-K
// draw, which would ignore the mask.)
procedure TTestNeuralDecode.TestGrammarConstraintSampledDrivenWalkStaysValid;
var
  C: TNNetGrammarConstraint;
  Verify: TNNetGrammarMachine;
  VG: TNNetGrammar;
  P: TNNetVolume;
  Sampler: TNNetSamplerWeightedTopK;
  Trial, Step, Tok: integer;
  S: string;
begin
  RandSeed := 20260613;
  C := TNNetGrammarConstraint.CreateCharLevel(ArithGrammar, 128);
  VG := TNNetGrammar.Create(ArithGrammar);
  Verify := TNNetGrammarMachine.Create(VG);
  P := TNNetVolume.Create(128, 1, 1);
  Sampler := TNNetSamplerWeightedTopK.Create(40);
  try
    for Trial := 1 to 20 do
    begin
      C.Reset([]);
      S := '';
      for Step := 1 to 16 do
      begin
        // Roughly uniform head; the grammar does all the constraining. Include
        // a chance of EOS so runs terminate.
        P.Fill(1.0);
        P.Raw[0] := 3.0;       // EOS bias
        C.MaskAllowed(P);
        Tok := Sampler.GetToken(P);
        if Tok < 2 then break; // EOS - allowed only at a complete state
        AssertTrue('run ' + IntToStr(Trial) + ' step ' + IntToStr(Step) +
          ': sampled token is grammar-legal', C.TokenAllowed(Tok));
        C.Commit(Tok);
        S := S + Chr(Tok);
      end;
      // Whatever was produced (terminated by EOS or the cap) must at least be a
      // legal prefix; if it stopped at EOS it must be a complete expression.
      Verify.Reset();
      AssertTrue('run ' + IntToStr(Trial) + ' output "' + S +
        '" is a legal grammar prefix', Verify.FeedString(S));
    end;
  finally
    Sampler.Free;
    P.Free;
    Verify.Free;
    VG.Free;
    C.Free;
  end;
end;

// Model-integration: a tiny char-level streamed LM whose head LOVES EOS
// (then '}', '{', '"'). Unconstrained greedy emits EOS immediately (no JSON
// at all); JSON-constrained greedy is forced to open and close an object
// before EOS becomes legal - and the result parses.
procedure TTestNeuralDecode.TestGenerateTokensStreamedJSONConstraintEmitsParseableJSON;
const
  CharVocab = 128;
var
  Net: TNNet;
  Head: TNNetLayer;
  Session: TNNetStreamingDecoder;
  Constraint: TNNetJSONConstraint;
  Toks: TNeuralIntegerArray;
  Parsed: TJSONData;
  OutLen, FreeLen, T, N: integer;
  Emitted: string;
begin
  RandSeed := 424242;
  // Width-1 char-level LM (token id = character code), built directly at the
  // decode width - no full-width twin needed because nothing is compared
  // against a full forward here.
  Net := TNNet.Create();
  Net.AddLayer(TNNetInput.Create(1, 1, 1));
  Net.AddLayer(TNNetEmbedding.Create(CharVocab, 8, 0, 0.02));
  Net.AddLayer(TNNetPointwiseConvLinear.Create(8));
  Net.AddLayer(TNNetDiagonalSSM.Create());
  Net.AddLayer(TNNetPointwiseConvLinear.Create(CharVocab));
  Head := Net.GetLastLayer();
  Net.AddLayer(TNNetPointwiseSoftMax.Create());
  Session := nil;
  Constraint := TNNetJSONConstraint.CreateCharLevel(CharVocab);
  try
    // Rig the logit head: EOS(1) > '}' > '{' > '"' > everything else.
    for N := 0 to Head.Neurons.Count - 1 do
    begin
      Head.Neurons[N].Weights.Fill(0);
      Head.Neurons[N].BiasWeight := 0;
    end;
    Head.Neurons[1].BiasWeight := 10;
    Head.Neurons[Ord('}')].BiasWeight := 9;
    Head.Neurons[Ord('{')].BiasWeight := 8;
    Head.Neurons[Ord('"')].BiasWeight := 7;
    // No-op refresh so the conv layer's packed weights see the hand edits
    // (this net is computed directly, with no CopyWeights to refresh it).
    Head.MulWeights(1.0);
    Session := TNNetStreamingDecoder.Create(Net, 32);

    // Unconstrained reference: EOS on the very first step.
    SetLength(Toks, 1);
    Toks[0] := Ord('a');
    FreeLen := GenerateTokensStreamed(Session, Toks, 1, 10, 32);
    AssertEquals('unconstrained greedy stops at EOS immediately', 2, FreeLen);
    AssertEquals('the free token IS the EOS', 1, Toks[1]);

    // JSON-constrained: EOS is masked until a complete value stands, so the
    // model is walked through '{' (best allowed) then '}' (best allowed in
    // the object) and only THEN may stop.
    SetLength(Toks, 0);
    SetLength(Toks, 1);
    Toks[0] := Ord('a');
    OutLen := GenerateTokensStreamed(Session, Toks, 1, 10, 32,
      nil, nil, nil, Constraint);
    Emitted := '';
    for T := 1 to OutLen - 1 do
    begin
      if Toks[T] < 2 then break;
      Emitted := Emitted + Chr(Toks[T]);
    end;
    AssertEquals('constrained generation emits an empty object',
      '{}', Emitted);
    AssertEquals('then stops at EOS', 1, Toks[OutLen - 1]);
    try
      Parsed := GetJSON(Emitted);
      Parsed.Free;
    except
      on E: Exception do
        Fail('constrained output "' + Emitted + '" is not parseable JSON: ' +
          E.Message);
    end;
  finally
    Constraint.Free;
    Session.Free;
    Net.Free;
  end;
end;

// Same JSON-mode gate on the full-re-encode DecodeGreedy path (char-level
// one-hot net): unconstrained = immediate EOS, constrained = '{}'.
procedure TTestNeuralDecode.TestDecodeGreedyJSONConstraintEmitsParseableJSON;
const
  CharVocab = 128;
var
  NN: TNNet;
  Logit: TNNetLayer;
  Constraint: TNNetJSONConstraint;
  Plain, Constrained: TNNetDecodeResult;
  Parsed: TJSONData;
  N: integer;
begin
  RandSeed := 424242;
  NN := BuildTinyNet(8, CharVocab);
  Constraint := TNNetJSONConstraint.CreateCharLevel(CharVocab);
  try
    // Rig the LOGIT layer (the SoftMax head has no neurons): EOS first.
    Logit := NN.Layers[NN.Layers.Count - 2];
    for N := 0 to Logit.Neurons.Count - 1 do
    begin
      Logit.Neurons[N].Weights.Fill(0);
      Logit.Neurons[N].BiasWeight := 0;
    end;
    Logit.Neurons[1].BiasWeight := 10;
    Logit.Neurons[Ord('}')].BiasWeight := 9;
    Logit.Neurons[Ord('{')].BiasWeight := 8;
    Logit.Neurons[Ord('"')].BiasWeight := 7;

    Plain := DecodeGreedy(NN, 'q', 6);
    AssertEquals('unconstrained: immediate EOS, empty text', '', Plain.Text);
    AssertEquals('unconstrained finishes on EOS', True, Plain.Finished);

    Constrained := DecodeGreedy(NN, 'q', 6, [], Constraint);
    AssertEquals('constrained text is an empty object', '{}',
      Constrained.Text);
    AssertEquals('constrained run finishes on the now-legal EOS',
      True, Constrained.Finished);
    try
      Parsed := GetJSON(Constrained.Text);
      Parsed.Free;
    except
      on E: Exception do
        Fail('constrained DecodeGreedy output "' + Constrained.Text +
          '" is not parseable JSON: ' + E.Message);
    end;
  finally
    Constraint.Free;
    NN.Free;
  end;
end;

function TTestNeuralDecode.FixturePath(const FileName: string): string;
begin
  Result := 'fixtures' + DirectorySeparator + FileName;
  if not FileExists(Result) then
    Result := 'tests' + DirectorySeparator + 'fixtures' + DirectorySeparator +
      FileName;
  if not FileExists(Result) then
    Fail('Fixture not found: ' + FileName +
      ' (run python3 tools/t5_tiny_fixture.py from the repo root).');
end;

// Greedy seq2seq decode on the committed T5 v1.0 pico pair: (a) two runs
// produce identical token ids (deterministic) within the caps and the vocab;
// (b) EOS may only ever be the LAST emitted token, and rigging EOS to the
// run's own first token forces an immediate single-token EOS stop; (c) the
// first generated token equals the argmax of row 0 of a single RunT5 call
// whose decoder input is the start token (the consistency oracle - causal
// self-attention makes the padded suffix invisible to row 0).
procedure TTestNeuralDecode.TestDecodeSeq2SeqGreedyT5DeterministicEOSAndOracle;
const
  Src: array[0..9] of integer = (3, 7, 1, 11, 4, 9, 2, 8, 5, 12);
  StartId = 0; // T5 decoder_start_token_id
  EOSId = 1;   // T5 eos_token_id
var
  Enc, Dec: TNNet;
  Config: TT5Config;
  G1, G2, ForcedEOS: TNeuralIntegerArray;
  EncToks, DecToks, Logits: TNNetVolume;
  I, OracleArgmax: integer;
begin
  RandSeed := 424242;
  BuildT5FromSafeTensors(FixturePath('tiny_t5v10.safetensors'),
    Enc, Dec, Config, {EncSeqLen=}10, {DecSeqLen=}6,
    {pTrainable=}true, FixturePath('tiny_t5v10_config.json'));
  EncToks := TNNetVolume.Create(10, 1, 1);
  DecToks := TNNetVolume.Create(6, 1, 1);
  Logits := TNNetVolume.Create();
  try
    G1 := DecodeSeq2SeqGreedy(Enc, Dec, Src, StartId, EOSId, 5);
    G2 := DecodeSeq2SeqGreedy(Enc, Dec, Src, StartId, EOSId, 5);
    AssertTrue('greedy generated something', Length(G1) >= 1);
    AssertTrue('greedy respects MaxNewTokens', Length(G1) <= 5);
    AssertEquals('greedy is deterministic (length)', Length(G1), Length(G2));
    for I := 0 to High(G1) do
    begin
      AssertEquals('greedy is deterministic (token ' + IntToStr(I) + ')',
        G1[I], G2[I]);
      AssertTrue('token id in vocab',
        (G1[I] >= 0) and (G1[I] < Config.VocabSize));
      // EOS terminates generation, so it can only be the last element.
      if G1[I] = EOSId then
        AssertEquals('EOS only at the last position', High(G1), I);
    end;
    AssertTrue('stops at EOS or MaxNewTokens',
      (G1[High(G1)] = EOSId) or (Length(G1) = 5));
    // (b) Forced EOS stop: declare the run's own first token to be EOS - the
    // decode must emit exactly that one token and stop.
    ForcedEOS := DecodeSeq2SeqGreedy(Enc, Dec, Src, StartId, G1[0], 5);
    AssertEquals('EOS-on-first-token stops immediately', 1, Length(ForcedEOS));
    AssertEquals('the EOS token is appended', G1[0], ForcedEOS[0]);
    // (c) Consistency oracle: one RunT5 call with the start-token-padded
    // decoder input; row 0's argmax must equal the first generated token.
    for I := 0 to 9 do EncToks.FData[I] := Src[I];
    for I := 0 to 5 do DecToks.FData[I] := StartId;
    RunT5(Enc, Dec, EncToks, DecToks, Logits);
    OracleArgmax := 0;
    for I := 1 to Config.VocabSize - 1 do
      if Logits.FData[I] > Logits.FData[OracleArgmax] then OracleArgmax := I;
    AssertEquals('first greedy step matches the RunT5 row-0 argmax oracle',
      OracleArgmax, G1[0]);
  finally
    Logits.Free;
    DecToks.Free;
    EncToks.Free;
    Dec.Free;
    Enc.Free;
  end;
end;

// Sampling at Temperature -> 0 turns the softmaxed row into a one-hot at the
// argmax, so probability-weighted samplers must reproduce the greedy
// sequence token for token (the "temperature ~0 matches greedy" gate): TopP
// falls back to the argmax once the top token alone exceeds P, and MinP's
// weighted draw collapses onto the one-hot. TNNetSamplerTopK draws UNIFORMLY
// among the top K (not probability-weighted), so only K=1 matches greedy -
// covered as the degenerate deterministic case. A nil sampler must equal
// greedy at ANY temperature (pure argmax path, no softmax computed).
procedure TTestNeuralDecode.TestDecodeSeq2SeqSampledTempZeroMatchesGreedy;
const
  Src: array[0..9] of integer = (3, 7, 1, 11, 4, 9, 2, 8, 5, 12);
var
  Enc, Dec: TNNet;
  Config: TT5Config;
  Greedy, Sampled: TNeuralIntegerArray;
  Sampler: TNNetSamplerBase;
  I: integer;
begin
  RandSeed := 424242;
  BuildT5FromSafeTensors(FixturePath('tiny_t5v10.safetensors'),
    Enc, Dec, Config, {EncSeqLen=}10, {DecSeqLen=}6,
    {pTrainable=}true, FixturePath('tiny_t5v10_config.json'));
  try
    Greedy := DecodeSeq2SeqGreedy(Enc, Dec, Src, 0, 1, 5);
    Sampler := TNNetSamplerTopK.Create(1);
    try
      Sampled := DecodeSeq2SeqSampled(Enc, Dec, Src, 0, 1, 5, Sampler, 1.0);
    finally
      Sampler.Free;
    end;
    AssertEquals('TopK(1) length matches greedy',
      Length(Greedy), Length(Sampled));
    for I := 0 to High(Greedy) do
      AssertEquals('TopK(1) token ' + IntToStr(I), Greedy[I], Sampled[I]);
    Sampler := TNNetSamplerTopP.Create(0.9);
    try
      Sampled := DecodeSeq2SeqSampled(Enc, Dec, Src, 0, 1, 5, Sampler, 1e-8);
    finally
      Sampler.Free;
    end;
    AssertEquals('TopP temp~0 length matches greedy',
      Length(Greedy), Length(Sampled));
    for I := 0 to High(Greedy) do
      AssertEquals('TopP temp~0 token ' + IntToStr(I), Greedy[I], Sampled[I]);
    Sampler := TNNetSamplerMinP.Create(0.5);
    try
      Sampled := DecodeSeq2SeqSampled(Enc, Dec, Src, 0, 1, 5, Sampler, 1e-8);
    finally
      Sampler.Free;
    end;
    AssertEquals('MinP temp~0 length matches greedy',
      Length(Greedy), Length(Sampled));
    for I := 0 to High(Greedy) do
      AssertEquals('MinP temp~0 token ' + IntToStr(I), Greedy[I], Sampled[I]);
    // nil sampler = greedy at any temperature (the argmax path ignores it).
    Sampled := DecodeSeq2SeqSampled(Enc, Dec, Src, 0, 1, 5, nil, 7.5);
    AssertEquals('nil sampler length matches greedy',
      Length(Greedy), Length(Sampled));
    for I := 0 to High(Greedy) do
      AssertEquals('nil sampler token ' + IntToStr(I), Greedy[I], Sampled[I]);
  finally
    Dec.Free;
    Enc.Free;
  end;
end;

// The harness is GENERIC over the two-net convention: the post-norm Marian
// pico pair (decoder_start = pad = 12, eos = 0) decodes deterministically,
// matches the same RunT5 row-0 argmax oracle, and a MaxNewTokens larger
// than the decoder's build-time DecSeqLen is capped by the capacity.
procedure TTestNeuralDecode.TestDecodeSeq2SeqGreedyMarianPairAndCapacity;
const
  Src: array[0..9] of integer = (5, 2, 9, 0, 7, 11, 3, 8, 1, 10);
  StartId = 12; // Marian decoder_start_token_id = pad_token_id
  EOSId = 0;    // Marian eos_token_id
var
  Enc, Dec: TNNet;
  Config: TMarianConfig;
  G1, G2: TNeuralIntegerArray;
  EncToks, DecToks, Logits: TNNetVolume;
  I, OracleArgmax: integer;
begin
  RandSeed := 424242;
  BuildMarianFromSafeTensors(FixturePath('tiny_marian.safetensors'),
    Enc, Dec, Config, {EncSeqLen=}10, {DecSeqLen=}6,
    {pTrainable=}true, FixturePath('tiny_marian_config.json'));
  EncToks := TNNetVolume.Create(10, 1, 1);
  DecToks := TNNetVolume.Create(6, 1, 1);
  Logits := TNNetVolume.Create();
  try
    // MaxNewTokens 99 >> DecSeqLen 6: capped by the build-time capacity
    // (6 tokens at most - the 6th is generated from the full-width prefix).
    G1 := DecodeSeq2SeqGreedy(Enc, Dec, Src, StartId, EOSId, 99);
    G2 := DecodeSeq2SeqGreedy(Enc, Dec, Src, StartId, EOSId, 99);
    AssertTrue('marian greedy generated something', Length(G1) >= 1);
    AssertTrue('capacity caps generation at DecSeqLen', Length(G1) <= 6);
    AssertEquals('marian greedy deterministic (length)',
      Length(G1), Length(G2));
    for I := 0 to High(G1) do
    begin
      AssertEquals('marian greedy deterministic (token ' + IntToStr(I) + ')',
        G1[I], G2[I]);
      AssertTrue('marian token id in vocab',
        (G1[I] >= 0) and (G1[I] < Config.VocabSize));
      if G1[I] = EOSId then
        AssertEquals('marian EOS only at the last position', High(G1), I);
    end;
    // Same consistency oracle as the T5 test, through the shared RunT5 path.
    for I := 0 to 9 do EncToks.FData[I] := Src[I];
    for I := 0 to 5 do DecToks.FData[I] := StartId;
    RunT5(Enc, Dec, EncToks, DecToks, Logits);
    OracleArgmax := 0;
    for I := 1 to Config.VocabSize - 1 do
      if Logits.FData[I] > Logits.FData[OracleArgmax] then OracleArgmax := I;
    AssertEquals('marian first greedy step matches the RunT5 oracle',
      OracleArgmax, G1[0]);
  finally
    Logits.Free;
    DecToks.Free;
    EncToks.Free;
    Dec.Free;
    Enc.Free;
  end;
end;

// Validation: a wrong source length and a decoder without a second
// TNNetInput raise EArgumentException; MaxNewTokens < 1 returns empty.
procedure TTestNeuralDecode.TestDecodeSeq2SeqRejectsInvalidArguments;
const
  Src: array[0..9] of integer = (3, 7, 1, 11, 4, 9, 2, 8, 5, 12);
var
  Enc, Dec, SingleInput: TNNet;
  Config: TT5Config;
  G: TNeuralIntegerArray;
begin
  RandSeed := 424242;
  BuildT5FromSafeTensors(FixturePath('tiny_t5v10.safetensors'),
    Enc, Dec, Config, {EncSeqLen=}10, {DecSeqLen=}6,
    {pTrainable=}true, FixturePath('tiny_t5v10_config.json'));
  try
    try
      DecodeSeq2SeqGreedy(Enc, Dec, [3, 7, 1], 0, 1, 5);
      Fail('a 3-token source must be rejected (encoder built at 10)');
    except
      on EArgumentException do ; // expected
    end;
    SingleInput := BuildTinyNet(4, 8);
    try
      try
        DecodeSeq2SeqGreedy(Enc, SingleInput, Src, 0, 1, 5);
        Fail('a decoder without a second TNNetInput must be rejected');
      except
        on EArgumentException do ; // expected
      end;
    finally
      SingleInput.Free;
    end;
    G := DecodeSeq2SeqGreedy(Enc, Dec, Src, 0, 1, 0);
    AssertEquals('MaxNewTokens=0 returns empty', 0, Length(G));
  finally
    Dec.Free;
    Enc.Free;
  end;
end;

procedure TTestNeuralDecode.BuildMarkovSeq2SeqPair(out Enc, Dec: TNNet;
  out Emb: TNNetEmbedding; Vocab, EncSeqLen, DecSeqLen: integer);
var
  TokIn: TNNetLayer;
begin
  // Identity "encoder": (EncSeqLen,1,1) in = out, so the decoder's second
  // input is sized EncSeqLen to match.
  Enc := TNNet.Create();
  Enc.AddLayer(TNNetInput.Create(EncSeqLen, 1, 1));
  Enc.AddLayer(TNNetIdentity.Create());
  // Markov decoder: token ids -> embedding row = next-token logits given the
  // PREVIOUS token only (row r reads the token at position r, so causality
  // holds trivially and the padded suffix never influences earlier rows).
  Dec := TNNet.Create();
  TokIn := Dec.AddLayer(TNNetInput.Create(DecSeqLen, 1, 1));
  // The second TNNetInput the two-net convention requires (filled by the
  // decode routines with the cached encoder states; unused by the chain).
  Dec.AddLayerAfter(TNNetInput.Create(EncSeqLen, 1, 1, 1), 0);
  Emb := TNNetEmbedding(Dec.AddLayerAfter(
    TNNetEmbedding.Create(Vocab, Vocab, {EncodeZero=}1, 1.0), TokIn));
end;

procedure TTestNeuralDecode.SetMarkovRow(Emb: TNNetEmbedding;
  PrevToken: integer; const Probs: array of TNeuralFloat);
var
  T: integer;
begin
  // Logits = ln(P): the decode routines' stable softmax recovers exactly
  // these transition probabilities (up to the row's normalisation).
  for T := 0 to High(Probs) do
    Emb.Neurons[0].Weights[PrevToken, 0, T] := Ln(Probs[T]);
end;

// "Trap" transition table shared by the two beam-vs-greedy tests (vocab 6,
// start=0, EOS=1). After start, token 2 (p=.50) narrowly beats token 3
// (p=.45) - but 2 leads to a mediocre continuation (best: EOS at p=.40)
// while 3 is followed by EOS at p=.99. Greedy locks in [2,1] with
// ln(.50)+ln(.40) = -1.609; the global best is [3,1] with
// ln(.45)+ln(.99) = -0.809.
procedure FillTrapTable(Test: TTestNeuralDecode; Emb: TNNetEmbedding);
begin
  Test.SetMarkovRow(Emb, 0, [1e-6, 0.02, 0.50, 0.45, 0.02, 0.01]);
  Test.SetMarkovRow(Emb, 1, [1/6, 1/6, 1/6, 1/6, 1/6, 1/6]);
  Test.SetMarkovRow(Emb, 2, [0.05, 0.40, 0.15, 0.15, 0.15, 0.10]);
  Test.SetMarkovRow(Emb, 3, [0.002, 0.99, 0.002, 0.002, 0.002, 0.002]);
  Test.SetMarkovRow(Emb, 4, [1/6, 1/6, 1/6, 1/6, 1/6, 1/6]);
  Test.SetMarkovRow(Emb, 5, [1/6, 1/6, 1/6, 1/6, 1/6, 1/6]);
end;

// BeamWidth=1 with LengthPenalty=0 reproduces DecodeSeq2SeqGreedy exactly,
// token for token (EOS included): the single beam follows the argmax path
// and its EOS-terminated beam outranks the step-1 EOS truncation
// (ln(.02) = -3.9 < -1.609) in the finished pool.
procedure TTestNeuralDecode.TestSeq2SeqBeamWidth1MatchesGreedy;
const
  Src: array[0..1] of integer = (1, 2);
var
  Enc, Dec: TNNet;
  Emb: TNNetEmbedding;
  G, Bm: TNeuralIntegerArray;
  I: integer;
begin
  BuildMarkovSeq2SeqPair(Enc, Dec, Emb, {Vocab=}6, {EncSeqLen=}2,
    {DecSeqLen=}6);
  try
    FillTrapTable(Self, Emb);
    G := DecodeSeq2SeqGreedy(Enc, Dec, Src, 0, 1, 6);
    // Self-check the scenario: greedy falls into the trap, [2, EOS].
    AssertEquals('greedy emits two tokens', 2, Length(G));
    AssertEquals('greedy first token is the trap token 2', 2, G[0]);
    AssertEquals('greedy second token is EOS', 1, G[1]);
    Bm := DecodeSeq2SeqBeamSearch(Enc, Dec, Src, 0, 1, 6,
      {BeamWidth=}1, {LengthPenalty=}0.0);
    AssertEquals('B=1/alpha=0 beam length equals greedy',
      Length(G), Length(Bm));
    for I := 0 to High(G) do
      AssertEquals('B=1/alpha=0 beam token ' + IntToStr(I), G[I], Bm[I]);
  finally
    Dec.Free;
    Enc.Free;
  end;
end;

// BeamWidth=2 recovers from the locally-greedy first-token mistake: the
// beam keeps runner-up token 3 alive, finds [3, EOS] with cumulative
// log-prob ln(.45)+ln(.99) = -0.809 > greedy's ln(.50)+ln(.40) = -1.609,
// and ranks it first in the returned pool.
procedure TTestNeuralDecode.TestSeq2SeqBeamBeatsGreedyFirstTokenTrap;
const
  Src: array[0..1] of integer = (1, 2);
var
  Enc, Dec: TNNet;
  Emb: TNNetEmbedding;
  G: TNeuralIntegerArray;
  All: TNNetTokenDecodeResultArray;
  GreedySum: TNeuralFloat;
  I: integer;
begin
  BuildMarkovSeq2SeqPair(Enc, Dec, Emb, 6, 2, 6);
  try
    FillTrapTable(Self, Emb);
    G := DecodeSeq2SeqGreedy(Enc, Dec, Src, 0, 1, 6);
    AssertEquals('greedy locks in the trap path [2,1]', 2, G[0]);
    All := DecodeSeq2SeqBeamSearchAll(Enc, Dec, Src, 0, 1, 6,
      {BeamWidth=}2, {LengthPenalty=}0.0);
    AssertTrue('beam returns a ranked pool', Length(All) >= 2);
    for I := 1 to High(All) do
      AssertTrue('pool sorted by descending score',
        All[I - 1].Score >= All[I].Score - 1e-6);
    AssertEquals('best beam has two tokens', 2, Length(All[0].Tokens));
    AssertEquals('best beam recovers token 3', 3, All[0].Tokens[0]);
    AssertEquals('best beam then EOS', 1, All[0].Tokens[1]);
    AssertTrue('best beam is finished', All[0].Finished);
    AssertEquals('best beam sum-log-prob is ln(.45)+ln(.99)',
      Ln(0.45) + Ln(0.99), All[0].SumLogProb, 1e-3);
    GreedySum := Ln(0.50) + Ln(0.40);
    AssertTrue('beam best beats the greedy path''s cumulative log-prob',
      All[0].SumLogProb > GreedySum + 0.5);
  finally
    Dec.Free;
    Enc.Free;
  end;
end;

// Finished-pool ranking: with alpha=0 the short finished beam [2,EOS]
// (sum -0.998) outranks the longer [3,4,EOS] (sum -1.203); alpha=2 lifts
// longer beams (denominator (5+L)/6)^2 grows with L) and flips the
// ranking - the verifiable direction of the Wu et al. penalty. Both
// winners come from the finished pool (EOS-terminated), ranked against
// the still-growing beams that were pruned/dominated along the way.
procedure TTestNeuralDecode.TestSeq2SeqBeamFinishedPoolLengthPenaltyFlipsRanking;
const
  Src: array[0..1] of integer = (3, 0);
var
  Enc, Dec: TNNet;
  Emb: TNNetEmbedding;
  All0, All2: TNNetTokenDecodeResultArray;
  I: integer;
  FoundLong: boolean;
begin
  BuildMarkovSeq2SeqPair(Enc, Dec, Emb, 6, 2, 6);
  try
    // [2,1]: ln(.55)+ln(.67) = -0.998 ; [3,4,1]: ln(.40)+ln(.95)+ln(.79)
    // = -1.203. Flip needs (8/7)^alpha > 1.203/0.998: alpha=2 gives 1.306.
    SetMarkovRow(Emb, 0, [1e-6, 0.01, 0.55, 0.40, 0.02, 0.02]);
    SetMarkovRow(Emb, 1, [1/6, 1/6, 1/6, 1/6, 1/6, 1/6]);
    SetMarkovRow(Emb, 2, [0.03, 0.67, 0.10, 0.10, 0.05, 0.05]);
    SetMarkovRow(Emb, 3, [0.002, 0.02, 0.01, 0.013, 0.95, 0.005]);
    SetMarkovRow(Emb, 4, [0.05, 0.79, 0.04, 0.04, 0.04, 0.04]);
    SetMarkovRow(Emb, 5, [1/6, 1/6, 1/6, 1/6, 1/6, 1/6]);

    All0 := DecodeSeq2SeqBeamSearchAll(Enc, Dec, Src, 0, 1, 6, 2, 0.0);
    AssertTrue('alpha=0 pool non-empty', Length(All0) >= 2);
    AssertEquals('alpha=0: short beam [2,1] wins', 2, Length(All0[0].Tokens));
    AssertEquals('alpha=0 winner first token', 2, All0[0].Tokens[0]);
    AssertEquals('alpha=0 winner ends in EOS', 1, All0[0].Tokens[1]);
    AssertTrue('alpha=0 winner is from the finished pool', All0[0].Finished);
    // The longer rival IS in the ranked pool (finished), just outranked.
    FoundLong := false;
    for I := 0 to High(All0) do
      if (Length(All0[I].Tokens) = 3) and (All0[I].Tokens[0] = 3) and
        (All0[I].Tokens[1] = 4) and (All0[I].Tokens[2] = 1) and
        All0[I].Finished then FoundLong := true;
    AssertTrue('alpha=0 pool contains the finished [3,4,1] rival', FoundLong);

    All2 := DecodeSeq2SeqBeamSearchAll(Enc, Dec, Src, 0, 1, 6, 2, 2.0);
    AssertTrue('alpha=2 pool non-empty', Length(All2) >= 2);
    AssertEquals('alpha=2: longer beam [3,4,1] wins',
      3, Length(All2[0].Tokens));
    AssertEquals('alpha=2 winner token 0', 3, All2[0].Tokens[0]);
    AssertEquals('alpha=2 winner token 1', 4, All2[0].Tokens[1]);
    AssertEquals('alpha=2 winner ends in EOS', 1, All2[0].Tokens[2]);
    AssertTrue('alpha=2 winner is finished', All2[0].Finished);
    AssertEquals('alpha=2 runner-up is the short beam [2,1]',
      2, Length(All2[1].Tokens));
    AssertEquals('alpha=2 runner-up first token', 2, All2[1].Tokens[0]);
    // Score = SumLogProb / ((5+L)/6)^alpha with L counting the EOS token.
    AssertEquals('alpha=2 winner score applies the Wu denominator',
      All2[0].SumLogProb / Sqr(8.0 / 6.0), All2[0].Score, 1e-5);
  finally
    Dec.Free;
    Enc.Free;
  end;
end;

// Pure beam search has NO RNG: two runs on the committed T5 pico pair are
// identical beam-for-beam (tokens, sums, scores). Caps and validation match
// the greedy routine: generation never exceeds min(MaxNewTokens, DecSeqLen),
// EOS only ever ends a finished beam, a wrong source length raises, and
// MaxNewTokens < 1 returns an empty pool (BeamWidth < 1 clamps to 1).
procedure TTestNeuralDecode.TestSeq2SeqBeamDeterministicCapsAndValidation;
const
  Src: array[0..9] of integer = (3, 7, 1, 11, 4, 9, 2, 8, 5, 12);
  StartId = 0;
  EOSId = 1;
var
  Enc, Dec: TNNet;
  Config: TT5Config;
  A1, A2: TNNetTokenDecodeResultArray;
  B1, BClamped: TNeuralIntegerArray;
  I, T: integer;
begin
  RandSeed := 424242;
  BuildT5FromSafeTensors(FixturePath('tiny_t5v10.safetensors'),
    Enc, Dec, Config, {EncSeqLen=}10, {DecSeqLen=}6,
    {pTrainable=}true, FixturePath('tiny_t5v10_config.json'));
  try
    A1 := DecodeSeq2SeqBeamSearchAll(Enc, Dec, Src, StartId, EOSId, 99,
      {BeamWidth=}3, {LengthPenalty=}1.0);
    A2 := DecodeSeq2SeqBeamSearchAll(Enc, Dec, Src, StartId, EOSId, 99,
      {BeamWidth=}3, {LengthPenalty=}1.0);
    AssertTrue('beam pool non-empty', Length(A1) >= 1);
    AssertEquals('deterministic pool size', Length(A1), Length(A2));
    for I := 0 to High(A1) do
    begin
      AssertEquals('deterministic beam ' + IntToStr(I) + ' length',
        Length(A1[I].Tokens), Length(A2[I].Tokens));
      for T := 0 to High(A1[I].Tokens) do
        AssertEquals('deterministic beam ' + IntToStr(I) + ' token ' +
          IntToStr(T), A1[I].Tokens[T], A2[I].Tokens[T]);
      AssertEquals('deterministic beam ' + IntToStr(I) + ' score',
        A1[I].Score, A2[I].Score, 0.0);
      // Caps and EOS placement, per beam.
      AssertTrue('beam capped at DecSeqLen', Length(A1[I].Tokens) <= 6);
      for T := 0 to High(A1[I].Tokens) do
        if A1[I].Tokens[T] = EOSId then
        begin
          AssertEquals('EOS only at the last position of a beam',
            High(A1[I].Tokens), T);
          AssertTrue('EOS-ending beam is marked finished', A1[I].Finished);
        end;
      if I > 0 then
        AssertTrue('pool sorted by descending score',
          A1[I - 1].Score >= A1[I].Score);
    end;
    // BeamWidth < 1 clamps to 1 (same single-best path as BeamWidth = 1).
    B1 := DecodeSeq2SeqBeamSearch(Enc, Dec, Src, StartId, EOSId, 5, 1, 0.0);
    BClamped := DecodeSeq2SeqBeamSearch(Enc, Dec, Src, StartId, EOSId, 5,
      0, 0.0);
    AssertEquals('BeamWidth=0 clamps to 1 (length)',
      Length(B1), Length(BClamped));
    for T := 0 to High(B1) do
      AssertEquals('BeamWidth=0 clamps to 1 (token ' + IntToStr(T) + ')',
        B1[T], BClamped[T]);
    // Validation mirrors the greedy/sampled routines.
    try
      DecodeSeq2SeqBeamSearch(Enc, Dec, [3, 7, 1], StartId, EOSId, 5, 2, 0.0);
      Fail('a 3-token source must be rejected (encoder built at 10)');
    except
      on EArgumentException do ; // expected
    end;
    AssertEquals('MaxNewTokens=0 returns an empty pool', 0,
      Length(DecodeSeq2SeqBeamSearchAll(Enc, Dec, Src, StartId, EOSId, 0,
        2, 0.0)));
  finally
    Dec.Free;
    Enc.Free;
  end;
end;

// ---------------------------------------------------------------------------
// Needle-in-a-haystack harness tests (deterministic, no trained model).

procedure TTestNeuralDecode.TestNeedleHaystackGridShapeMatchesAxes;
const
  Lengths: array[0..2] of integer = (50, 100, 200);
  Depths: array[0..3] of TNeuralFloat = (0.0, 0.33, 0.66, 1.0);
var
  R: TNeedleInHaystackResult;
begin
  R := NeedleInHaystackReport(Lengths, Depths,
    'The magic number is 42.', '42', 'What is the magic number?',
    @NeedleTestFiller, @NeedlePerfectGen, nil);
  AssertEquals('rows = number of depth fractions', Length(Depths),
    Length(R.Cells));
  AssertEquals('cols = number of context lengths', Length(Lengths),
    Length(R.Cells[0]));
  AssertEquals('total cells = rows * cols',
    Length(Depths) * Length(Lengths), R.TotalCount);
  AssertEquals('depth axis preserved', Length(Depths),
    Length(R.DepthFractions));
  AssertEquals('length axis preserved', Length(Lengths),
    Length(R.ContextLengths));
end;

procedure TTestNeuralDecode.TestNeedleHaystackPerfectRetrieverIs100Percent;
const
  Lengths: array[0..1] of integer = (60, 120);
  Depths: array[0..2] of TNeuralFloat = (0.0, 0.5, 1.0);
var
  R: TNeedleInHaystackResult;
begin
  R := NeedleInHaystackReport(Lengths, Depths,
    'The magic number is 42.', '42', 'What is the magic number?',
    @NeedleTestFiller, @NeedlePerfectGen, nil);
  AssertEquals('oracle retrieves every cell', R.TotalCount, R.HitCount);
  AssertEquals('accuracy is 1.0', 1.0, R.Accuracy, 1e-6);
  // Rendered report must show only hit markers and the 100% summary.
  AssertTrue('report contains a hit marker (X)', Pos('X', R.Report) > 0);
  AssertTrue('report contains no miss marker (.)',
    Pos(' .', R.Report) = 0);
  AssertTrue('report shows 100.0% accuracy',
    Pos('100.0% accuracy', R.Report) > 0);
end;

procedure TTestNeuralDecode.TestNeedleHaystackNeverRetrieverIsZeroPercent;
const
  Lengths: array[0..1] of integer = (60, 120);
  Depths: array[0..2] of TNeuralFloat = (0.0, 0.5, 1.0);
var
  R: TNeedleInHaystackResult;
begin
  R := NeedleInHaystackReport(Lengths, Depths,
    'The magic number is 42.', '42', 'What is the magic number?',
    @NeedleTestFiller, @NeedleNeverGen, nil);
  AssertEquals('never-retriever scores no hits', 0, R.HitCount);
  AssertEquals('accuracy is 0.0', 0.0, R.Accuracy, 1e-6);
  AssertTrue('report contains a miss marker (.)', Pos('.', R.Report) > 0);
  AssertTrue('report shows 0.0% accuracy',
    Pos('0.0% accuracy', R.Report) > 0);
end;

procedure TTestNeuralDecode.TestNeedleHaystackInsertsNeedleAtRequestedDepth;
const
  Lengths: array[0..0] of integer = (80);
  Depths: array[0..2] of TNeuralFloat = (0.0, 0.5, 1.0);
var
  R: TNeedleInHaystackResult;
  d: integer;
  NeedlePos: integer;
begin
  // The depth-probe generator only answers if "magic number is 42" survived
  // splicing intact at EVERY depth -> all hits proves correct insertion.
  R := NeedleInHaystackReport(Lengths, Depths,
    'The magic number is 42.', '42', 'What is the magic number?',
    @NeedleTestFiller, @NeedleDepthProbeGen, nil);
  AssertEquals('needle intact at all depths', R.TotalCount, R.HitCount);
  // Splice position must MOVE with depth: start prompt has the needle near the
  // front, end prompt near the back.
  for d := 0 to High(Depths) do
  begin
    NeedlePos := Pos('magic number is 42', LowerCase(R.Cells[d][0].Prompt));
    AssertTrue('needle present in cell prompt', NeedlePos > 0);
  end;
  AssertTrue('depth 0 needle is earlier than depth 1 needle',
    Pos('magic number is 42', LowerCase(R.Cells[0][0].Prompt)) <
    Pos('magic number is 42', LowerCase(R.Cells[High(Depths)][0].Prompt)));
end;

// --- JSON-Schema -> GBNF compiler ------------------------------------------

// A two-field tool-call arguments schema: both properties required, a strict
// (additionalProperties:false) object. The compiled grammar must ACCEPT the
// well-formed argument objects and REJECT malformed / extra-field ones via the
// TNNetGrammar pushdown (same accept = FeedString and IsComplete idiom the
// arithmetic-grammar test uses).
procedure TTestNeuralDecode.TestSchemaGBNFObjectRequiredAcceptsAndRejects;
const
  Schema =
    '{"type":"object",' +
    '"properties":{' +
    '"location":{"type":"string"},' +
    '"days":{"type":"integer"}},' +
    '"required":["location","days"],' +
    '"additionalProperties":false}';
var
  GBNF: string;
  G: TNNetGrammar;
  M: TNNetGrammarMachine;

  function OK(const S: string): boolean;
  begin
    M.Reset();
    Result := M.FeedString(S) and M.IsComplete();
  end;

begin
  GBNF := CompileJSONSchemaToGBNF(Schema);
  G := TNNetGrammar.Create(GBNF);
  M := TNNetGrammarMachine.Create(G);
  try
    // Both required props present in declared order -> accepted.
    AssertTrue('both fields valid',
      OK('{"location":"Paris","days":3}'));
    AssertTrue('whitespace between tokens tolerated',
      OK('{ "location" : "Paris" , "days" : 12 }'));
    AssertTrue('negative integer day',
      OK('{"location":"Rio","days":-1}'));
    // Missing a required field -> rejected.
    AssertTrue('missing required "days" rejected',
      not OK('{"location":"Paris"}'));
    AssertTrue('empty object rejected (fields required)',
      not OK('{}'));
    // Wrong order (root is declared-order) -> rejected.
    AssertTrue('swapped field order rejected',
      not OK('{"days":3,"location":"Paris"}'));
    // additionalProperties:false -> an extra field is rejected.
    AssertTrue('extra field rejected (additionalProperties:false)',
      not OK('{"location":"Paris","days":3,"unit":"c"}'));
    // Type violation: days must be an integer, not a string.
    AssertTrue('string where integer expected rejected',
      not OK('{"location":"Paris","days":"three"}'));
    AssertTrue('fractional where integer expected rejected',
      not OK('{"location":"Paris","days":3.5}'));
  finally
    M.Free;
    G.Free;
  end;
end;

// Exercises enum -> literal alternation, array + minItems/maxItems, and
// $ref/$defs recursion (a tree node referencing itself through $defs).
procedure TTestNeuralDecode.TestSchemaGBNFEnumArrayRefRecursion;
const
  EnumSchema =
    '{"type":"object",' +
    '"properties":{"unit":{"enum":["celsius","fahrenheit"]}},' +
    '"required":["unit"],"additionalProperties":false}';
  ArraySchema =
    '{"type":"array","items":{"type":"integer"},' +
    '"minItems":2,"maxItems":3}';
  RefSchema =
    '{"$ref":"#/$defs/node",' +
    '"$defs":{"node":{"type":"object",' +
    '"properties":{"children":{"type":"array",' +
    '"items":{"$ref":"#/$defs/node"}}},' +
    '"required":["children"],"additionalProperties":false}}}';
var
  G: TNNetGrammar;
  M: TNNetGrammarMachine;

  function OK(const S: string): boolean;
  begin
    M.Reset();
    Result := M.FeedString(S) and M.IsComplete();
  end;

begin
  // enum -> alternation of the two exact string literals.
  G := TNNetGrammar.Create(CompileJSONSchemaToGBNF(EnumSchema));
  M := TNNetGrammarMachine.Create(G);
  try
    AssertTrue('enum value "celsius"', OK('{"unit":"celsius"}'));
    AssertTrue('enum value "fahrenheit"', OK('{"unit":"fahrenheit"}'));
    AssertTrue('non-enum value rejected', not OK('{"unit":"kelvin"}'));
  finally
    M.Free; G.Free;
  end;
  // array minItems=2 maxItems=3.
  G := TNNetGrammar.Create(CompileJSONSchemaToGBNF(ArraySchema));
  M := TNNetGrammarMachine.Create(G);
  try
    AssertTrue('two items', OK('[1,2]'));
    AssertTrue('three items', OK('[1,2,3]'));
    AssertTrue('one item rejected (minItems 2)', not OK('[1]'));
    AssertTrue('four items rejected (maxItems 3)', not OK('[1,2,3,4]'));
    AssertTrue('empty array rejected', not OK('[]'));
  finally
    M.Free; G.Free;
  end;
  // $ref/$defs recursion: a self-referential tree.
  G := TNNetGrammar.Create(CompileJSONSchemaToGBNF(RefSchema));
  M := TNNetGrammarMachine.Create(G);
  try
    AssertTrue('leaf node (empty children)', OK('{"children":[]}'));
    AssertTrue('one level of nesting',
      OK('{"children":[{"children":[]}]}'));
    AssertTrue('two levels of nesting',
      OK('{"children":[{"children":[{"children":[]}]}]}'));
    AssertTrue('missing required children rejected', not OK('{}'));
  finally
    M.Free; G.Free;
  end;
end;

// A tiny greedy decode under CreateJSONSchemaConstraint must produce
// byte-legal output for a two-field tool-call schema: every emitted character
// keeps the grammar machine in a legal state and the run ends in a complete
// (accepting) parse. Char-level Dict (id = char code, ids < 2 special) mirrors
// the existing JSON-constraint greedy-decode test.
procedure TTestNeuralDecode.TestSchemaGBNFConstraintGreedyToolCallIsByteLegal;
const
  CharVocab = 128;
  Schema =
    '{"type":"object",' +
    '"properties":{' +
    '"name":{"type":"string"},' +
    '"value":{"type":"integer"}},' +
    '"required":["name","value"],"additionalProperties":false}';
var
  NN: TNNet;
  Logit: TNNetLayer;
  Dict: TStringListInt;
  Constraint: TNNetGrammarConstraint;
  Constrained: TNNetDecodeResult;
  Probe, Fresh: TNNetGrammarMachine;
  FreshG: TNNetGrammar;
  N, I: integer;
begin
  RandSeed := 424242;
  // Context wide enough to hold the whole emitted tool call (no reversed-
  // one-hot truncation warnings).
  NN := BuildTinyNet(64, CharVocab);
  // Char-level Dict: id i (>= 2) detokenizes to Chr(i); ids 0/1 are special.
  Dict := TStringListInt.Create();
  Constraint := nil;
  try
    Dict.Sorted := false;
    Dict.Add('<eos>');  // id 0
    Dict.Add('<pad>');  // id 1
    for I := 2 to CharVocab - 1 do Dict.Add(Chr(I));
    Dict.SaveCurrentPosition();
    AssertEquals('char-level vocab', CharVocab, Dict.GetVocabCount());

    Constraint := CreateJSONSchemaConstraint(Schema, Dict);

    // Rig the LOGIT layer so the unconstrained argmax would wander, but the
    // constraint forces every step to a grammar-legal char. Bias toward EOS
    // and the structural chars; the constraint masks illegal mass each step.
    Logit := NN.Layers[NN.Layers.Count - 2];
    for N := 0 to Logit.Neurons.Count - 1 do
    begin
      Logit.Neurons[N].Weights.Fill(0);
      Logit.Neurons[N].BiasWeight := 0;
    end;
    Logit.Neurons[1].BiasWeight := 14;          // EOS most wanted (gated)
    Logit.Neurons[Ord('"')].BiasWeight := 11;   // quote beats other string chars
    Logit.Neurons[Ord('}')].BiasWeight := 10;
    Logit.Neurons[Ord('{')].BiasWeight := 9;
    Logit.Neurons[Ord(':')].BiasWeight := 8;
    Logit.Neurons[Ord(',')].BiasWeight := 7;
    Logit.Neurons[Ord('a')].BiasWeight := 6;
    Logit.Neurons[Ord('1')].BiasWeight := 5;

    Constrained := DecodeGreedy(NN, 'q', 64, [], Constraint);

    // (a) Every byte of the produced text must keep an independent fresh
    //     machine legal, and the full string must parse (IsComplete).
    Probe := Constraint.Machine; // live machine after the committed run
    AssertTrue('decode ended in a complete (accepting) parse',
      Probe.IsComplete());
    AssertTrue('non-empty structured output', Length(Constrained.Text) > 0);

    // (b) Independent re-validation: feed the text through a FRESH machine
    //     compiled straight from the schema; it must be byte-legal end to end.
    FreshG := TNNetGrammar.Create(CompileJSONSchemaToGBNF(Schema));
    Fresh := TNNetGrammarMachine.Create(FreshG);
    try
      AssertTrue('emitted text is byte-legal under a fresh machine',
        Fresh.FeedString(Constrained.Text));
      AssertTrue('emitted text is a complete parse', Fresh.IsComplete());
    finally
      Fresh.Free;
      FreshG.Free;
    end;
  finally
    Constraint.Free;
    NN.Free;
    Dict.Free;
  end;
end;

procedure TTestNeuralDecode.BuildBigramBeamPair(out Full, Twin: TNNet;
  Vocab, FullContextLen: integer);
var
  HF: TNNetLayer;
  Emb: TNNetEmbedding;
  T, O: integer;
  Table: array of array of TNeuralFloat;
begin
  // Deterministic bigram logit table Table[prevTok][outTok].
  RandSeed := 991733;
  SetLength(Table, Vocab);
  for T := 0 to Vocab - 1 do
  begin
    SetLength(Table[T], Vocab);
    for O := 0 to Vocab - 1 do
      // A spread of values (some negative) so the beam tree actually branches
      // and EOS (token 1) sometimes wins, exercising the finished pool.
      Table[T][O] := (Random - 0.5) * 6.0;
  end;

  // CACHE-LESS net (DecodeBeamSearchAll): ONE-HOT input fed REVERSED, so the
  // current (last) token lands at position 0. A FullConnect head collapses the
  // whole sequence to one Vocab-sized distribution; we wire Table into ONLY the
  // position-0 weight block (every other position zeroed), making the output
  // exactly softmax(Table[currentToken]).
  Full := TNNet.Create();
  Full.AddLayer(TNNetInput.Create(FullContextLen, 1, Vocab));
  HF := Full.AddLayer(TNNetFullConnectLinear.Create(Vocab));
  Full.AddLayer(TNNetSoftMax.Create());
  Full.InitWeights();
  for O := 0 to Vocab - 1 do
  begin
    HF.Neurons[O].Weights.Fill(0);
    HF.Neurons[O].BiasWeight := 0;
    for T := 0 to Vocab - 1 do
      HF.Neurons[O].Weights[0, 0, T] := Table[T][O];
  end;
  HF.MulWeights(1.0);

  // STREAMING twin (DecodeBeamSearchCachedAll): TOKEN-ID input (FData[0]=token,
  // the streamed-decoder convention). An Embedding whose row t IS the logit
  // vector Table[t], then SoftMax, computes the SAME softmax(Table[token]) - so
  // both nets realise the identical bigram next-token distribution.
  Twin := TNNet.Create();
  Twin.AddLayer(TNNetInput.Create(1, 1, 1));
  // EncodeZero=1: token 0 is a REAL embedding row (not a zero pad), so the
  // bigram covers the full 0..Vocab-1 alphabet like the one-hot net does.
  Emb := TNNetEmbedding(Twin.AddLayer(TNNetEmbedding.Create(Vocab, Vocab, 1, 1.0)));
  Twin.AddLayer(TNNetSoftMax.Create());
  Twin.InitWeights();
  for T := 0 to Vocab - 1 do
    for O := 0 to Vocab - 1 do
      Emb.Neurons[0].Weights[T, 0, O] := Table[T][O];
  Emb.MulWeights(1.0);
end;

function TTestNeuralDecode.CausalReEncodeBeamAll(Full: TNNet;
  const PromptToks: array of integer; MaxLen, BeamWidth: integer;
  LengthPenalty: TNeuralFloat): TNNetDecodeResultArray;
type
  TLBeam = record
    Text: string; SumLogProb, Score: TNeuralFloat; Finished: boolean;
  end;
const
  csEOS = 1;
var
  InV: TNNetVolume; Row: TNNetVolume;
  ContextLen, VocabSize, Step, I, T, B, PromptLen: integer;
  Total, CutScore: TNeuralFloat;
  Live, Finished, Cand: array of TLBeam;
  NB: TLBeam;
  AllDominated: boolean;

  procedure SortDesc(var A: array of TLBeam);
  var ii, jj: integer; tt: TLBeam;
  begin
    for ii := 1 to High(A) do
    begin
      tt := A[ii]; jj := ii - 1;
      while (jj >= 0) and (A[jj].Score < tt.Score) do
      begin A[jj + 1] := A[jj]; Dec(jj); end;
      A[jj + 1] := tt;
    end;
  end;

  // Full re-encode: zero-pad PromptToks ++ generated chars, read the softmax
  // row at the last real position (the next-token distribution).
  procedure NextLP(const Gen: string; var LP: array of TNeuralFloat);
  var k, last: integer;
  begin
    InV.Fill(0);
    PromptLen := Length(PromptToks);
    for k := 0 to PromptLen - 1 do InV.FData[k] := PromptToks[k];
    for k := 1 to Length(Gen) do InV.FData[PromptLen + k - 1] := Ord(Gen[k]);
    last := PromptLen + Length(Gen) - 1;
    Full.Compute(InV);
    Row := Full.GetLastLayer().Output;
    Total := 0;
    for k := 0 to VocabSize - 1 do Total := Total + Row[last, 0, k];
    if Total <= 0 then Total := 1.0;
    for k := 0 to VocabSize - 1 do LP[k] := SafeLogProb(Row[last, 0, k] / Total);
  end;

var
  LP: array of TNeuralFloat;
begin
  ContextLen := Full.GetFirstLayer().Output.SizeX;
  // Token-id input (Depth=1); the vocabulary is the LM head's output depth.
  VocabSize := Full.GetLastLayer().Output.Depth;
  SetLength(LP, VocabSize);
  if BeamWidth < 1 then BeamWidth := 1;
  // Token-id input volume (Depth=1), zero-padded; matches the streamed twin.
  InV := TNNetVolume.Create(ContextLen, 1, 1);
  try
    SetLength(Live, 1);
    Live[0].Text := ''; Live[0].SumLogProb := 0; Live[0].Score := 0;
    Live[0].Finished := False;
    SetLength(Finished, 0);
    for Step := 1 to MaxLen do
    begin
      if Length(Live) = 0 then Break;
      if Length(Finished) >= BeamWidth then
      begin
        SortDesc(Finished);
        CutScore := Finished[BeamWidth - 1].Score;
        AllDominated := True;
        for I := 0 to High(Live) do
          if Live[I].Score > CutScore then AllDominated := False;
        if AllDominated then Break;
      end;
      SetLength(Cand, 0);
      for B := 0 to High(Live) do
      begin
        NextLP(Live[B].Text, LP);
        for T := 0 to VocabSize - 1 do
        begin
          NB.SumLogProb := Live[B].SumLogProb + LP[T];
          if T = csEOS then
          begin
            NB.Text := Live[B].Text; NB.Finished := True;
            NB.Score := NB.SumLogProb /
              LengthPenaltyDenominator(Length(NB.Text), LengthPenalty);
            SetLength(Finished, Length(Finished) + 1);
            Finished[High(Finished)] := NB;
          end
          else
          begin
            NB.Text := Live[B].Text + Chr(T); NB.Finished := False;
            NB.Score := NB.SumLogProb /
              LengthPenaltyDenominator(Length(NB.Text), LengthPenalty);
            SetLength(Cand, Length(Cand) + 1);
            Cand[High(Cand)] := NB;
          end;
        end;
      end;
      SortDesc(Cand);
      if Length(Cand) > BeamWidth then SetLength(Cand, BeamWidth);
      SetLength(Live, Length(Cand));
      for I := 0 to High(Cand) do Live[I] := Cand[I];
    end;
    for B := 0 to High(Live) do
    begin
      SetLength(Finished, Length(Finished) + 1);
      Finished[High(Finished)] := Live[B];
    end;
    SortDesc(Finished);
    SetLength(Result, Length(Finished));
    for I := 0 to High(Finished) do
    begin
      Result[I].Text := Finished[I].Text;
      Result[I].SumLogProb := Finished[I].SumLogProb;
      Result[I].Score := Finished[I].Score;
      Result[I].Finished := Finished[I].Finished;
    end;
  finally
    InV.Free;
  end;
end;

procedure TTestNeuralDecode.TestBeamSearchCachedMatchesReEncodeBigram;
const
  Vocab = 8;
  ContextLen = 12;
var
  Full, Twin: TNNet;
  Session: TNNetStreamingDecoder;
  RefAll, CacAll: TNNetDecodeResultArray;
  Prompt: string;
  I: integer;
begin
  // Prompt chars must be >= 2 (0/1 are EOS/terminal) and < Vocab.
  Prompt := Chr(3) + Chr(5);
  BuildBigramBeamPair(Full, Twin, Vocab, ContextLen);
  Session := nil;
  try
    Session := TNNetStreamingDecoder.Create(Twin, ContextLen);
    RefAll := DecodeBeamSearchAll(Full, Prompt, 6, 4, 0.0);
    CacAll := DecodeBeamSearchCachedAll(Session, Prompt, 6, 4, 0.0);
    AssertEquals('same beam count', Length(RefAll), Length(CacAll));
    for I := 0 to High(RefAll) do
    begin
      AssertEquals('text rank ' + IntToStr(I), RefAll[I].Text, CacAll[I].Text);
      AssertEquals('sumlogprob rank ' + IntToStr(I),
        RefAll[I].SumLogProb, CacAll[I].SumLogProb, 0.0);
      AssertEquals('score rank ' + IntToStr(I),
        RefAll[I].Score, CacAll[I].Score, 0.0);
      AssertTrue('finished flag rank ' + IntToStr(I),
        RefAll[I].Finished = CacAll[I].Finished);
    end;
    // Also exercise the length-penalised, single-best wrapper.
    RefAll := DecodeBeamSearchAll(Full, Prompt, 6, 4, 0.7);
    CacAll := DecodeBeamSearchCachedAll(Session, Prompt, 6, 4, 0.7);
    AssertEquals('alpha=0.7 same beam count', Length(RefAll), Length(CacAll));
    for I := 0 to High(RefAll) do
    begin
      AssertEquals('alpha=0.7 text rank ' + IntToStr(I),
        RefAll[I].Text, CacAll[I].Text);
      AssertEquals('alpha=0.7 score rank ' + IntToStr(I),
        RefAll[I].Score, CacAll[I].Score, 0.0);
    end;
  finally
    Session.Free;
    Twin.Free;
    Full.Free;
  end;
end;

procedure TTestNeuralDecode.TestBeamSearchCachedMatchesReEncodeCausalReference;
const
  SeqLen = 12;
var
  Full, Twin: TNNet;
  Session: TNNetStreamingDecoder;
  RefAll, CacAll: TNNetDecodeResultArray;
  PromptToks: array of integer;
  Prompt: string;
  I: integer;
begin
  // Genuine KV-cache topology: RoPE causal attention + DyT norms. The full net
  // is driven by a zero-pad re-encode beam reference; the twin streams with
  // per-beam cache forking. SoftMax appended to both so the rows are proper
  // probabilities (CopyWeights still maps 1:1 - identical topology).
  RandSeed := 424242;
  Full := BuildTinyCausalLM(SeqLen);
  Full.AddLayer(TNNetPointwiseSoftMax.Create());
  Twin := BuildTinyCausalLM(1);
  Twin.AddLayer(TNNetPointwiseSoftMax.Create());
  Session := nil;
  try
    Twin.CopyWeights(Full);
    Session := TNNetStreamingDecoder.Create(Twin, SeqLen);
    SetLength(PromptToks, 2);
    PromptToks[0] := 3; PromptToks[1] := 7;
    Prompt := Chr(PromptToks[0]) + Chr(PromptToks[1]);

    RefAll := CausalReEncodeBeamAll(Full, PromptToks, 5, 3, 0.0);
    CacAll := DecodeBeamSearchCachedAll(Session, Prompt, 5, 3, 0.0);

    AssertEquals('same beam count', Length(RefAll), Length(CacAll));
    for I := 0 to High(RefAll) do
    begin
      AssertEquals('text rank ' + IntToStr(I), RefAll[I].Text, CacAll[I].Text);
      AssertEquals('sumlogprob rank ' + IntToStr(I),
        RefAll[I].SumLogProb, CacAll[I].SumLogProb, 1e-5);
      AssertEquals('score rank ' + IntToStr(I),
        RefAll[I].Score, CacAll[I].Score, 1e-5);
      AssertTrue('finished flag rank ' + IntToStr(I),
        RefAll[I].Finished = CacAll[I].Finished);
    end;
  finally
    Session.Free;
    Twin.Free;
    Full.Free;
  end;
end;

procedure TTestNeuralDecode.TestBeamSearchCachedRejectsWideSession;
var
  Wide: TNNet;
  Session: TNNetStreamingDecoder;
  Raised: boolean;
begin
  // A non-width-1 session must be rejected (the cache-forking driver needs a
  // width-1 twin, like the other streamed decoders).
  Wide := BuildTinyCausalLM(4);
  Session := nil;
  Raised := False;
  try
    Session := TNNetStreamingDecoder.Create(Wide, 8);
    try
      DecodeBeamSearchCachedAll(Session, Chr(3) + Chr(5), 4, 2, 0.0);
    except
      on EArgumentException do Raised := True;
    end;
    AssertTrue('wide session rejected', Raised);
  finally
    Session.Free;
    Wide.Free;
  end;
end;

// ---------------------------------------------------------------------------
// Forced-prefix KV-cached seq2seq decode tests.
// ---------------------------------------------------------------------------

procedure TTestNeuralDecode.BuildCrossAttnSeq2SeqPair(
  out Enc, DecFull, DecCached: TNNet;
  EncSeqLen, DecFullSeqLen, Vocab: integer);
const Dim = 8;

  // Builds one cross-attention decoder at the given query width. The token
  // input is FIRST (GetFirstLayer), the encoder-states input SECOND (the
  // two-net convention); the embedding stream branches off the token input,
  // then a full decoder block cross-attends to the encoder-states input.
  function BuildDec(DecSeqLen: integer): TNNet;
  var TokIn, EncStatesIn: TNNetLayer;
  begin
    Result := TNNet.Create();
    TokIn := Result.AddLayer(TNNetInput.Create(DecSeqLen, 1, 1));
    EncStatesIn := Result.AddLayer(TNNetInput.Create(EncSeqLen, 1, Dim));
    Result.AddLayerAfter(
      TNNetEmbedding.Create(Vocab, Dim, {EncodeZero=}0, 0.02), TokIn);
    Result.AddTransformerDecoderBlock({Heads=}2, {d_ff=}8, EncStatesIn,
      {PreNorm=}true, {UseRoPE=}true, {NormClass=}TNNetDyT);
    Result.AddLayer(TNNetPointwiseConvLinear.Create(Vocab));
  end;

begin
  // Encoder: token ids -> embedding -> causal transformer block -> hidden
  // states (EncSeqLen,1,Dim), matching the decoder's encoder-states input.
  RandSeed := 424242;
  Enc := TNNet.Create();
  Enc.AddLayer(TNNetInput.Create(EncSeqLen, 1, 1));
  Enc.AddLayer(TNNetEmbedding.Create(Vocab, Dim, 0, 0.02));
  Enc.AddTransformerEncoderBlock({Heads=}2, {d_ff=}8,
    {PreNorm=}true, {CausalMask=}false, {UseRoPE=}true, {NormClass=}TNNetDyT);
  Enc.InitWeights();

  // Two decoders off the SAME random init, then CopyWeights makes the width-1
  // cached twin bit-identical to the wide full-re-decode reference.
  RandSeed := 987654;
  DecFull := BuildDec(DecFullSeqLen);
  DecFull.InitWeights();
  DecCached := BuildDec(1);
  DecCached.CopyWeights(DecFull);
end;

function TTestNeuralDecode.ForcedPrefixFullReDecode(DecFull: TNNet;
  EncStates: TNNetVolume; const ForcedPrefix: array of integer;
  EOSTokenId, MaxNewTokens: integer): TNeuralIntegerArray;
var
  EncStatesIn: TNNetLayer;
  DecToks, Logits: TNNetVolume;
  Prefix: TNeuralIntegerArray;
  DecSeqLen, CurLen, Pos, Next, PfxLen, i: integer;
begin
  SetLength(Result, 0);
  if MaxNewTokens < 1 then exit;
  // Second TNNetInput holds the (constant) encoder states.
  EncStatesIn := Seq2SeqEncoderStatesInput(DecFull);
  EncStatesIn.Output.Copy(EncStates);
  DecSeqLen := DecFull.GetFirstLayer().Output.Size;
  Logits := DecFull.GetLastLayer().Output;
  PfxLen := Length(ForcedPrefix);
  // Working prefix = forced prologue ++ generated, padded to DecSeqLen with the
  // last token (causal masking makes the pad invisible to earlier rows).
  SetLength(Prefix, PfxLen);
  for i := 0 to PfxLen - 1 do Prefix[i] := ForcedPrefix[i];
  CurLen := PfxLen;
  DecToks := TNNetVolume.Create(DecSeqLen, 1, 1);
  try
    while True do
    begin
      for Pos := 0 to DecSeqLen - 1 do
        if Pos < CurLen
        then DecToks.FData[Pos] := Prefix[Pos]
        else DecToks.FData[Pos] := Prefix[CurLen - 1];
      DecFull.Compute(DecToks);
      Next := Logits.GetClassOnPixel(CurLen - 1, 0);
      SetLength(Result, Length(Result) + 1);
      Result[High(Result)] := Next;
      if Next = EOSTokenId then break;
      if Length(Result) >= MaxNewTokens then break;
      if CurLen >= DecSeqLen then break;
      SetLength(Prefix, CurLen + 1);
      Prefix[CurLen] := Next;
      Inc(CurLen);
    end;
  finally
    DecToks.Free;
  end;
end;

procedure TTestNeuralDecode.TestForcedPrefixCachedMatchesFullReDecode;
const
  EncSeqLen = 5; Vocab = 12; MaxNew = 14; DecFullSeqLen = 20;
var
  Enc, DecFull, DecCached: TNNet;
  EncToks, EncStates: TNNetVolume;
  Prefix, Cached, Full: TNeuralIntegerArray;
  i: integer;
begin
  BuildCrossAttnSeq2SeqPair(Enc, DecFull, DecCached, EncSeqLen, DecFullSeqLen,
    Vocab);
  EncToks := TNNetVolume.Create(EncSeqLen, 1, 1);
  EncStates := TNNetVolume.Create();
  try
    // Run the encoder ONCE; both decode paths consume these fixed states.
    for i := 0 to EncSeqLen - 1 do EncToks.FData[i] := (i * 3 + 2) mod Vocab;
    Enc.Compute(EncToks);
    EncStates.Copy(Enc.GetLastLayer().Output);

    // A multi-token forced prologue (Whisper-style), EOS = 11.
    SetLength(Prefix, 4);
    Prefix[0] := 1; Prefix[1] := 6; Prefix[2] := 3; Prefix[3] := 9;

    Cached := DecodeSeq2SeqForcedPrefixCached(DecCached, EncStates, Prefix,
      {EOS=}11, MaxNew);
    Full := ForcedPrefixFullReDecode(DecFull, EncStates, Prefix, {EOS=}11,
      MaxNew);

    AssertEquals('cached and full lengths match', Length(Full), Length(Cached));
    for i := 0 to High(Full) do
      AssertEquals('token ' + IntToStr(i) + ' bit-identical',
        Full[i], Cached[i]);
    AssertTrue('at least one token generated', Length(Cached) > 0);
  finally
    EncStates.Free; EncToks.Free;
    DecCached.Free; DecFull.Free; Enc.Free;
  end;
end;

procedure TTestNeuralDecode.TestForcedPrefixSteersGreedyContinuation;
const
  EncSeqLen = 5; Vocab = 12; MaxNew = 10; DecFullSeqLen = 20;
var
  Enc, DecFull, DecCached, Dec2, DecFull2: TNNet;
  EncToks, EncStates: TNNetVolume;
  PrefixA, PrefixB, OutA, OutB: TNeuralIntegerArray;
  i: integer;
  Differ: boolean;
begin
  BuildCrossAttnSeq2SeqPair(Enc, DecFull, DecCached, EncSeqLen, DecFullSeqLen,
    Vocab);
  // A second cached twin off the same weights for the B run (a session resets
  // its own cache, so one twin would suffice, but two keeps the runs isolated).
  BuildCrossAttnSeq2SeqPair(Enc, DecFull2, Dec2, EncSeqLen, DecFullSeqLen,
    Vocab);
  EncToks := TNNetVolume.Create(EncSeqLen, 1, 1);
  EncStates := TNNetVolume.Create();
  try
    for i := 0 to EncSeqLen - 1 do EncToks.FData[i] := (i * 3 + 2) mod Vocab;
    Enc.Compute(EncToks);
    EncStates.Copy(Enc.GetLastLayer().Output);

    // Two DIFFERENT forced prologues from the SAME encoder states. EOS unset
    // (use an id never produced) so the full MaxNew window is generated.
    SetLength(PrefixA, 3); PrefixA[0] := 2; PrefixA[1] := 4; PrefixA[2] := 7;
    SetLength(PrefixB, 3); PrefixB[0] := 8; PrefixB[1] := 5; PrefixB[2] := 1;

    OutA := DecodeSeq2SeqForcedPrefixCached(DecCached, EncStates, PrefixA,
      {EOS=}-1, MaxNew);
    OutB := DecodeSeq2SeqForcedPrefixCached(Dec2, EncStates, PrefixB,
      {EOS=}-1, MaxNew);

    // The forced tokens are NOT in the result (excluded by convention), yet the
    // distinct prologues must steer the greedy continuation to differ.
    AssertEquals('A full window', MaxNew, Length(OutA));
    AssertEquals('B full window', MaxNew, Length(OutB));
    Differ := False;
    for i := 0 to MaxNew - 1 do
      if OutA[i] <> OutB[i] then Differ := True;
    AssertTrue('distinct forced prologues steer different continuations',
      Differ);
  finally
    EncStates.Free; EncToks.Free;
    Dec2.Free; DecFull2.Free;
    DecCached.Free; DecFull.Free; Enc.Free;
  end;
end;

procedure TTestNeuralDecode.TestForcedPrefixCachedRejectsInvalidArguments;
const
  EncSeqLen = 5; Vocab = 12; DecFullSeqLen = 20;
var
  Enc, DecFull, DecCached: TNNet;
  EncStates, BadStates: TNNetVolume;
  EmptyPrefix, Prefix: TNeuralIntegerArray;
  Raised: boolean;
begin
  BuildCrossAttnSeq2SeqPair(Enc, DecFull, DecCached, EncSeqLen, DecFullSeqLen,
    Vocab);
  EncStates := TNNetVolume.Create(EncSeqLen, 1, 8);
  BadStates := TNNetVolume.Create(EncSeqLen + 1, 1, 8);
  try
    SetLength(EmptyPrefix, 0);
    SetLength(Prefix, 1); Prefix[0] := 1;

    // Empty prefix is rejected.
    Raised := False;
    try
      DecodeSeq2SeqForcedPrefixCached(DecCached, EncStates, EmptyPrefix, 11, 4);
    except on EArgumentException do Raised := True; end;
    AssertTrue('empty prefix rejected', Raised);

    // Encoder-states size mismatch is rejected.
    Raised := False;
    try
      DecodeSeq2SeqForcedPrefixCached(DecCached, BadStates, Prefix, 11, 4);
    except on EArgumentException do Raised := True; end;
    AssertTrue('encoder-states size mismatch rejected', Raised);

    // A wide (DecSeqLen > 1) decoder cannot drive incremental decode.
    Raised := False;
    try
      DecodeSeq2SeqForcedPrefixCached(DecFull, EncStates, Prefix, 11, 4);
    except on EArgumentException do Raised := True; end;
    AssertTrue('wide decoder rejected', Raised);

    // MaxNewTokens < 1 returns empty (no raise).
    AssertEquals('MaxNewTokens<1 empty', 0,
      Length(DecodeSeq2SeqForcedPrefixCached(DecCached, EncStates, Prefix,
        11, 0)));
  finally
    BadStates.Free; EncStates.Free;
    DecCached.Free; DecFull.Free; Enc.Free;
  end;
end;

initialization
  RegisterTest(TTestNeuralDecode);

end.
