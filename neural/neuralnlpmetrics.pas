unit neuralnlpmetrics;

(*
neuralnlpmetrics
NLP evaluation utilities: held-out PERPLEXITY for autoregressive language
models plus the two classic text-overlap metrics, corpus-level BLEU
(Papineni et al. 2002) and ROUGE-N / ROUGE-L (Lin 2004).

Everything here is pure forward-pass / counting code: no training
dependencies, no new layer types, no state.

PERPLEXITY. Two model families are covered, matching the two generation
conventions already in the codebase:

  Perplexity(NN, Dict, Corpus): token models driven through a
  TStringListInt / TNeuralTokenizer vocabulary (Dict.Tokenize). Two head
  shapes are auto-detected:
    (a) per-position heads (GenerateStringFromCasualNN convention): the
        last layer output has SizeX = input SizeX and Depth = vocab, so ONE
        teacher-forced forward scores every next-token position of a sample
        at once. Input is encoded left-aligned: raw token ids
        (CopyNoChecksIntArr, zero/pad-token padded) when the input depth is
        1 (TNNetEmbedding front-ends), one-hot (OneHotEncoding) otherwise.
    (b) single next-token heads (GenerateStringFromTokens convention,
        output is one vocab-sized distribution): one forward per predicted
        position over the growing RIGHT-aligned prefix
        (CopyReversedNoChecksIntArr / OneHotEncodingReversed).

  PerplexityFromChars(NN, Corpus): char models in the
  GenerateStringFromChars / TinyGPT / SimpleNLP convention (one-hot
  right-aligned via OneHotEncodingReversed(string), single next-char
  softmax). Tokens are Ord(ch); for these char-level models
  bits-per-token IS bits-per-character (BPC).

  Shared conventions (both functions):
  - TRUNCATION (v1): each sample is truncated to the model's context window
    (input SizeX); positions beyond the window are NOT scored (no sliding
    re-evaluation). Documented trade-off: simple, cheap, and unbiased for
    samples that fit the window.
  - SPECIAL TOKENS: target tokens < 2 (the codebase-wide pad=0 / EOS=1
    convention, "NextTokenInt < 2") and out-of-vocab targets are EXCLUDED
    from the average (counted in SkippedTokens) when ExcludeSpecialTokens
    is true (default).
  - The model head is read as POST-SOFTMAX probabilities (the
    neuralcalibration convention); each row is defensively re-normalised
    and clamped through neuraldecode.SafeLogProb so a dead-but-not-
    impossible token never produces -Inf.
  - Returned stats: Perplexity = exp(MeanNLL); MeanNLL is the mean negative
    log-likelihood in NATS per predicted token; BitsPerToken =
    MeanNLL / ln(2). For char-level models BitsPerToken = BPC; for word/BPE
    models bits-per-character = BitsPerToken * (#tokens / #characters).

TOKEN-LEVEL LOGPROB SCORING (mini lm-eval harness). ScoreSequence(NN,
Tokens) returns the per-token log-probabilities ln p(t_i | t_0..t_{i-1})
of an already-tokenized sequence from teacher-forced forward passes (NO
generation, NO samplers/logit processors: the raw model distribution).
Result[i] scores Tokens[i]; Result[0] is always 0 (the first token has no
conditioning context and is never scored - the lm-evaluation-harness
convention). The same two head shapes as Perplexity are auto-detected
(per-position heads: ONE forward for the whole sequence, the row at
position i-1 scores token i; single next-token heads: one forward per
scored position over the growing right-aligned prefix). Rows are
defensively re-normalised and clamped through SafeLogProb exactly like
Perplexity, so summing ScoreSequence over a corpus reproduces
Perplexity's MeanNLL on the same windows (with ExcludeSpecialTokens =
false; ScoreSequence itself never skips tokens - out-of-vocab targets
score SafeLogProb(0)). CONTEXT LENGTH (v1, documented choice): sequences
longer than the model context window raise a clear exception instead of
silently scoring a sub-window.

ScoreCompletion(NN, ContextTokens, CompletionTokens) concatenates the two
and sums ScoreSequence over the COMPLETION tokens only (context tokens
are conditioned on, never scored; the first completion token is scored
from the last context position, so ContextTokens must be non-empty).
Returns the sum, the length-normalized mean (sum / completion length) and
the token count.

EvaluateMultipleChoice(NN, Items) is the HellaSwag/ARC/PIQA pattern from
lm-evaluation-harness: each item is a context, N candidate completions
and a gold index; every candidate is scored with ScoreCompletion and the
argmax wins (first-max tie-break). Accuracy ranks by SumLogProb (lm-eval
"acc", short-biased) and AccuracyNorm by MeanLogProb (lm-eval "acc_norm",
length-normalized) - the two disagree on length-confounded items.
One full forward per candidate (v1); batching candidates that share a
context prefix is a possible follow-up.

BLEU. CorpusBLEU implements Papineni et al. 2002: corpus-pooled MODIFIED
(clipped) n-gram precision up to MaxN (default 4), geometric mean with
uniform weights, multiplied by the brevity penalty
BP = min(1, exp(1 - refLen/candLen)) over corpus-total lengths. ONE
reference per candidate (v1). Smoothing (default on) is Lin & Och (2004)
smoothing-1 (= Chen & Cherry 2014 method 2): add 1 to BOTH the clipped
match count and the candidate n-gram total for every order n >= 2
(unigrams are never smoothed). Orders with zero candidate n-grams (all
candidates shorter than n) are excluded from the geometric mean. Identical
candidate/reference pairs score exactly 1.0 with or without smoothing.

chrF. ChrF implements Popovic (2015): the character n-gram F-score, a
tokenizer-INDEPENDENT metric that operates directly on characters and so
sidesteps BLEU's tokenization sensitivity. For each character n-gram order
n in 1..CharOrder (default 6) the clipped (min-count) multiset overlap
gives a per-order precision (match/hyp) and recall (match/ref); a per-order
F_beta = (1+beta^2)*P*R / (beta^2*P + R) is formed, and the metric is the
ARITHMETIC MEAN of the per-order F_beta over all "effective" orders (orders
where the hypothesis OR reference has at least one n-gram of that length;
empty orders are skipped, never counted as zero). beta default 2 weights
recall twice precision. This matches sacrebleu's CHRF aggregation (average
of per-order F, NOT F of the averaged P/R).

WHITESPACE (sacrebleu default, documented): sacrebleu's default chrF has
whitespace=False, which STRIPS all whitespace before extracting character
n-grams (so "a b" and "ab" share the same char n-grams). ChrF reproduces
that default; pass IncludeWhitespace=true for the spaces-in-n-grams variant.

chrF++ (ChrFpp / the WordOrder argument): adds WORD n-grams (whitespace-
tokenized, BEFORE whitespace stripping) of orders 1..WordOrder to the same
per-order F average. The canonical chrF++ uses WordOrder=2 (word unigrams +
bigrams); WordOrder=0 is plain chrF.

SCALE: the functions here return a 0..1 fraction (consistent with the
CorpusBLEU/ROUGE convention in this unit); sacrebleu reports the same number
multiplied by 100. CorpusChrF macro-averages the per-pair score.

ROUGE. RougeN returns clipped n-gram overlap precision/recall/F1
(beta = 1); RougeL returns the LCS-based variant (P = LCS/|cand|,
R = LCS/|ref|, F1 = harmonic mean). CorpusRougeN / CorpusRougeL average the
per-pair scores (macro average over the corpus).

DEGENERATION / GENERATION-QUALITY metrics (the standard suite for judging
sampler/decoding diversity).

  DistinctN (Li et al. 2016): the ratio (number of DISTINCT n-grams) /
  (total number of n-grams) of a single generation. distinct-1 / distinct-2
  are the canonical reported values; the n argument generalises. "a a a a"
  has 4 unigrams but 1 distinct type -> distinct-1 = 1/4. An empty / too-short
  generation (fewer than n tokens) scores 0.

  RepetitionRate is the complementary degeneration signal: the fraction of
  n-gram occurrences that are REPEATS of an already-seen type,
  = 1 - DistinctN (so "a a a a" has repetition-1 = 3/4). Sequence-level
  repetition: it is the standard rep-n used in the neural-text-degeneration
  literature (Welleck et al. 2020 / Holtzman et al. 2020). RepeatedTokenRate
  is the rep-1 special case spelled out for convenience.

  SelfBLEU (Zhu et al. 2018, "Texec"/diversity): for a set of generations,
  each generation is scored with sentence-BLEU against ALL THE OTHERS as
  references, and the scores are averaged. High self-BLEU = the generations
  resemble each other = low diversity. This REUSES the corpus-BLEU machinery
  above (one generation = candidate, every other = a reference): because
  CorpusBLEU here is single-reference (v1), each candidate is scored against
  every other generation separately and the per-candidate BLEU is the MEAN
  over those single-reference BLEUs (the multi-reference clip is a documented
  v1 simplification). Needs at least two generations.

Both BLEU/ROUGE metrics expose a dual API: integer TOKEN-ID arrays
(TNeuralIntegerArray, e.g. straight out of Dict.Tokenize) and a string
convenience overload that whitespace-tokenizes (case-sensitive; words are
mapped to ids through a vocabulary shared between candidate and
reference, so the two APIs agree exactly).

Coded by Claude (AI).
*)

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, neuralvolume, neuralnetwork;

type
  // Aggregate result of a perplexity evaluation.
  TNNetPerplexityStats = record
    Perplexity: TNeuralFloat;   // exp(MeanNLL)
    MeanNLL: TNeuralFloat;      // mean -ln p(target) in nats / predicted token
    BitsPerToken: TNeuralFloat; // MeanNLL / ln(2); = BPC for char models
    PredictedTokens: integer;   // positions actually scored
    SkippedTokens: integer;     // special (<2) / out-of-vocab targets excluded
  end;

  // Result of scoring one candidate completion given a context.
  TNNetCompletionScore = record
    SumLogProb: TNeuralFloat;   // sum ln p over completion tokens only
    MeanLogProb: TNeuralFloat;  // SumLogProb / TokenCount (length-normalized)
    TokenCount: integer;        // number of completion tokens scored
  end;

  // One multiple-choice evaluation item (lm-eval request style): a shared
  // context, N candidate completions and the index of the gold candidate.
  TNNetMultipleChoiceItem = record
    ContextTokens: TNeuralIntegerArray;
    Candidates: array of TNeuralIntegerArray;
    GoldIndex: integer;
  end;

  // Aggregate multiple-choice results: acc (SumLogProb argmax) vs acc_norm
  // (MeanLogProb argmax) - lm-eval's accuracy / length-normalized accuracy.
  TNNetMultipleChoiceStats = record
    Accuracy: TNeuralFloat;       // CorrectCount / ItemCount
    AccuracyNorm: TNeuralFloat;   // CorrectNormCount / ItemCount
    ItemCount: integer;
    CorrectCount: integer;        // gold wins by sum logprob
    CorrectNormCount: integer;    // gold wins by mean (length-normalized) logprob
  end;

  // Precision / recall / F1 triple returned by the ROUGE functions.
  TNNetRougeScore = record
    Precision: TNeuralFloat;
    Recall: TNeuralFloat;
    F1: TNeuralFloat;
  end;

  // One decoded entity span from a BIO/IOB2 tag sequence: the entity TYPE
  // (the part after the "B-"/"I-" prefix, e.g. 'PER') and the inclusive token
  // index range [TokenStart..TokenEnd] it covers.
  TNNetEntitySpan = record
    EntityType: string;
    TokenStart: integer;
    TokenEnd: integer;
  end;
  TNNetEntitySpanArray = array of TNNetEntitySpan;

  // seqeval-style entity-level evaluation result (an exact-match span set
  // comparison: a predicted span counts as correct only when its type AND its
  // [start..end] range both match a gold span).
  TNNetEntityScore = record
    Precision: TNeuralFloat;  // TruePos / (TruePos + FalsePos)
    Recall: TNeuralFloat;     // TruePos / (TruePos + FalseNeg)
    F1: TNeuralFloat;         // harmonic mean
    TruePos: integer;
    FalsePos: integer;
    FalseNeg: integer;
  end;

  // One QA candidate answer span (token-index range, inclusive) and its score
  // (start_logit + end_logit), produced by ExtractQASpans as an n-best list.
  TNNetQASpan = record
    TokenStart: integer;
    TokenEnd: integer;
    Score: TNeuralFloat;
  end;
  TNNetQASpanArray = array of TNNetQASpan;

// Teacher-forced held-out perplexity of a token-level autoregressive LM over
// a corpus of text samples (one sample per Corpus line). See the unit header
// for head-shape auto-detection, truncation and special-token rules.
function Perplexity(NN: TNNet; Dict: TStringListInt; Corpus: TStrings;
  ExcludeSpecialTokens: boolean = true): TNNetPerplexityStats;

// Char-level sibling for GenerateStringFromChars-style models (one-hot
// right-aligned input, single next-char softmax). MinContext is the shortest
// prefix length used to predict (TinyGPT trains with prefixes >= 3).
function PerplexityFromChars(NN: TNNet; Corpus: TStrings;
  MinContext: integer = 1;
  ExcludeSpecialTokens: boolean = true): TNNetPerplexityStats;

// Per-token log-probabilities ln p(Tokens[i] | Tokens[0..i-1]) of an
// already-tokenized sequence (teacher-forced; no generation). Result has the
// same length as Tokens; Result[0] is 0 (never scored). Raises EArgumentException
// when the sequence exceeds the model context window (v1; see unit header).
function ScoreSequence(NN: TNNet;
  const Tokens: TNeuralIntegerArray): TNeuralFloatDynArr;

// Sum + length-normalized logprob of CompletionTokens given ContextTokens.
// Only completion tokens are scored; ContextTokens must be non-empty.
function ScoreCompletion(NN: TNNet;
  const ContextTokens, CompletionTokens: TNeuralIntegerArray): TNNetCompletionScore;

// Multiple-choice harness: scores every candidate completion of every item
// with ScoreCompletion and reports acc (sum-logprob argmax) and acc_norm
// (mean-logprob argmax). See unit header.
function EvaluateMultipleChoice(NN: TNNet;
  const Items: array of TNNetMultipleChoiceItem): TNNetMultipleChoiceStats;

// Corpus-level BLEU (one reference per candidate; Candidates[i] is scored
// against References[i]; both arrays must have the same length).
function CorpusBLEU(const Candidates, References: array of TNeuralIntegerArray;
  MaxN: integer = 4; Smooth: boolean = true): TNeuralFloat; overload;
function CorpusBLEU(const Candidates, References: array of string;
  MaxN: integer = 4; Smooth: boolean = true): TNeuralFloat; overload;

// ROUGE-N (clipped n-gram overlap) for one candidate/reference pair.
function RougeN(const Candidate, Reference: TNeuralIntegerArray;
  N: integer): TNNetRougeScore; overload;
function RougeN(const Candidate, Reference: string;
  N: integer): TNNetRougeScore; overload;

// ROUGE-L (longest-common-subsequence) for one candidate/reference pair.
function RougeL(const Candidate, Reference: TNeuralIntegerArray): TNNetRougeScore; overload;
function RougeL(const Candidate, Reference: string): TNNetRougeScore; overload;

// Corpus-level (macro-averaged) ROUGE over aligned candidate/reference lists.
function CorpusRougeN(const Candidates, References: array of string;
  N: integer): TNNetRougeScore;
function CorpusRougeL(const Candidates, References: array of string): TNNetRougeScore;

// chrF / chrF++ (Popovic 2015) for one hypothesis/reference string pair.
// CharOrder = max character n-gram length (default 6); Beta weights recall
// (default 2); WordOrder > 0 turns on chrF++ (word n-grams 1..WordOrder, the
// canonical chrF++ uses 2); IncludeWhitespace = false (sacrebleu default)
// strips all whitespace before char n-gram extraction. Returns a 0..1
// fraction (= sacrebleu's report / 100).
function ChrF(const Hypothesis, Reference: string;
  CharOrder: integer = 6; Beta: TNeuralFloat = 2.0;
  WordOrder: integer = 0; IncludeWhitespace: boolean = false): TNeuralFloat;

// Convenience chrF++ wrapper: ChrF with WordOrder = 2.
function ChrFpp(const Hypothesis, Reference: string;
  CharOrder: integer = 6; Beta: TNeuralFloat = 2.0): TNeuralFloat;

// Corpus-level (macro-averaged) chrF over aligned hypothesis/reference lists.
function CorpusChrF(const Hypotheses, References: array of string;
  CharOrder: integer = 6; Beta: TNeuralFloat = 2.0;
  WordOrder: integer = 0; IncludeWhitespace: boolean = false): TNeuralFloat;

// distinct-n (Li et al. 2016): distinct n-grams / total n-grams of one
// generation. Returns 0 when the generation has fewer than N tokens.
function DistinctN(const Tokens: TNeuralIntegerArray; N: integer): TNeuralFloat; overload;
function DistinctN(const Text: string; N: integer): TNeuralFloat; overload;

// repetition rate = 1 - distinct-n (fraction of n-gram occurrences that
// repeat an already-seen type); 0 for a generation shorter than N tokens.
function RepetitionRate(const Tokens: TNeuralIntegerArray; N: integer): TNeuralFloat; overload;
function RepetitionRate(const Text: string; N: integer): TNeuralFloat; overload;

// Convenience rep-1 (repeated-token rate = 1 - distinct-1).
function RepeatedTokenRate(const Tokens: TNeuralIntegerArray): TNeuralFloat; overload;
function RepeatedTokenRate(const Text: string): TNeuralFloat; overload;

// self-BLEU (Zhu et al. 2018): mean sentence-BLEU of each generation against
// the OTHER generations (diversity / mode-collapse signal). Reuses CorpusBLEU.
// Needs >= 2 generations; returns 0 otherwise.
function SelfBLEU(const Generations: array of TNeuralIntegerArray;
  MaxN: integer = 4; Smooth: boolean = true): TNeuralFloat; overload;
function SelfBLEU(const Generations: array of string;
  MaxN: integer = 4; Smooth: boolean = true): TNeuralFloat; overload;

// ---------------------------------------------------------------------------
// Token classification (NER): BIO/IOB2 entity decoding + entity-level P/R/F1
// ---------------------------------------------------------------------------

// Decodes a BIO/IOB2 tag sequence into entity spans (the seqeval get_entities
// algorithm). Tags are strings: 'O' (outside), 'B-TYPE' (begin) or 'I-TYPE'
// (inside). A span opens on a 'B-TYPE' tag (or on an 'I-TYPE' that follows a
// non-matching tag - the IOB2 lenient-start rule seqeval uses) and extends
// over consecutive 'I-TYPE' of the SAME type; any 'O', a 'B-' or a type switch
// closes the current span. Returns the spans in left-to-right order.
function DecodeBIOEntities(const Tags: array of string): TNNetEntitySpanArray;

// Entity-level (span exact-match) precision / recall / F1 for one sentence,
// the seqeval default. A predicted span is a true positive only when an
// identical (type + [start..end]) gold span exists; unmatched predictions are
// false positives, unmatched gold spans false negatives. PredTags and GoldTags
// must have the same length.
function EntityScore(const PredTags, GoldTags: array of string): TNNetEntityScore; overload;

// Corpus-level (micro-averaged) entity P/R/F1 over aligned predicted/gold tag
// sentences: TP/FP/FN are pooled across all sentences, then P/R/F1 computed
// once (the seqeval classification_report micro-average). Pred[i] and Gold[i]
// are the tag sequences of sentence i and must be equal length.
function CorpusEntityScore(const Pred, Gold: array of TStringArray): TNNetEntityScore;

// ---------------------------------------------------------------------------
// QA span extraction (SQuAD-style postprocessing)
// ---------------------------------------------------------------------------

// SQuAD-style n-best span extraction from per-token start/end logits. Considers
// the TopK highest start positions and TopK highest end positions, forms every
// (start,end) pair with end >= start and (end - start + 1) <= MaxAnswerLen, and
// returns them ranked by start_logit + end_logit (descending), keeping at most
// NBest candidates. StartLogits and EndLogits must have the same length (the
// sequence length); MaxAnswerLen <= 0 means no length cap. The top result is
// Result[0] (empty array only when no valid pair exists, e.g. zero-length input).
function ExtractQASpans(const StartLogits, EndLogits: TNeuralFloatDynArr;
  TopK: integer = 20; MaxAnswerLen: integer = 30;
  NBest: integer = 20): TNNetQASpanArray;

implementation

uses
  Math, neuraldecode;

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

procedure ZeroStats(out Stats: TNNetPerplexityStats);
begin
  Stats.Perplexity := 0;
  Stats.MeanNLL := 0;
  Stats.BitsPerToken := 0;
  Stats.PredictedTokens := 0;
  Stats.SkippedTokens := 0;
end;

procedure FinishStats(SumNLL: TNeuralFloat; var Stats: TNNetPerplexityStats);
begin
  if Stats.PredictedTokens > 0 then
  begin
    Stats.MeanNLL := SumNLL / Stats.PredictedTokens;
    Stats.BitsPerToken := Stats.MeanNLL / Ln(2.0);
    Stats.Perplexity := Exp(Stats.MeanNLL);
  end;
end;

// True when the target token must be skipped (special/pad or out-of-vocab).
function SkipTarget(Tgt, VocabSize: integer; ExcludeSpecial: boolean): boolean;
begin
  Result := (Tgt < 0) or (Tgt >= VocabSize) or (ExcludeSpecial and (Tgt < 2));
end;

// ---------------------------------------------------------------------------
// Perplexity
// ---------------------------------------------------------------------------

function Perplexity(NN: TNNet; Dict: TStringListInt; Corpus: TStrings;
  ExcludeSpecialTokens: boolean = true): TNNetPerplexityStats;
var
  InV: TNNetVolume;
  Last: TNNetLayer;
  Toks, Prefix: TNeuralIntegerArray;
  ContextLen, InDepth, VocabSize: integer;
  PerPosition: boolean;
  LineIdx, SampleLen, ClippedLen, Pos, D, Tgt: integer;
  RowSum, Prob, SumNLL: TNeuralFloat;
begin
  ZeroStats(Result);
  if (NN = nil) or (Dict = nil) or (Corpus = nil) then Exit;
  if NN.CountLayers() < 2 then Exit;
  ContextLen := NN.GetFirstLayer().Output.SizeX;
  InDepth := NN.GetFirstLayer().Output.Depth;
  Last := NN.GetLastLayer();
  // Head-shape auto-detection (see unit header): per-position teacher-forced
  // heads carry one vocab distribution per input position.
  PerPosition := (ContextLen > 1) and (Last.Output.SizeX = ContextLen) and
    (Last.Output.Depth >= 2);
  if PerPosition
  then VocabSize := Last.Output.Depth
  else VocabSize := Last.Output.Size;
  if VocabSize < 2 then Exit;
  SumNLL := 0;
  InV := TNNetVolume.Create(NN.GetFirstLayer().Output);
  try
    for LineIdx := 0 to Corpus.Count - 1 do
    begin
      Dict.Tokenize(Corpus[LineIdx], Toks);
      SampleLen := Length(Toks);
      if SampleLen < 2 then continue; // nothing to predict
      if PerPosition then
      begin
        // v1 truncation: only the first ContextLen tokens are scored.
        ClippedLen := Min(SampleLen, ContextLen);
        Prefix := Copy(Toks, 0, ClippedLen);
        InV.Fill(0);
        if InDepth = 1
        then InV.CopyNoChecksIntArr(Prefix)       // token ids -> embedding
        else InV.OneHotEncoding(Prefix);          // one-hot, left-aligned
        NN.Compute(InV);
        for Pos := 1 to ClippedLen - 1 do
        begin
          Tgt := Toks[Pos];
          if SkipTarget(Tgt, VocabSize, ExcludeSpecialTokens) then
          begin
            Inc(Result.SkippedTokens);
            continue;
          end;
          // Output row Pos-1 predicts token Pos. Defensive re-normalisation
          // keeps the math honest even for near-softmax heads.
          RowSum := 0;
          for D := 0 to VocabSize - 1 do
            RowSum := RowSum + Last.Output[Pos - 1, 0, D];
          if RowSum <= 0 then RowSum := 1.0;
          Prob := Last.Output[Pos - 1, 0, Tgt] / RowSum;
          SumNLL := SumNLL - SafeLogProb(Prob);
          Inc(Result.PredictedTokens);
        end;
      end
      else
      begin
        // Single next-token head: one forward per predicted position over
        // the growing right-aligned prefix (truncated to the window).
        for Pos := 1 to Min(SampleLen - 1, ContextLen) do
        begin
          Tgt := Toks[Pos];
          if SkipTarget(Tgt, VocabSize, ExcludeSpecialTokens) then
          begin
            Inc(Result.SkippedTokens);
            continue;
          end;
          Prefix := Copy(Toks, 0, Pos);
          if InDepth = 1 then
          begin
            InV.Fill(0);
            InV.CopyReversedNoChecksIntArr(Prefix);
          end
          else InV.OneHotEncodingReversed(Prefix);
          NN.Compute(InV);
          RowSum := Last.Output.GetSum();
          if RowSum <= 0 then RowSum := 1.0;
          Prob := Last.Output.FData[Tgt] / RowSum;
          SumNLL := SumNLL - SafeLogProb(Prob);
          Inc(Result.PredictedTokens);
        end;
      end;
    end;
  finally
    InV.Free;
  end;
  SetLength(Toks, 0);
  SetLength(Prefix, 0);
  FinishStats(SumNLL, Result);
end;

function PerplexityFromChars(NN: TNNet; Corpus: TStrings;
  MinContext: integer = 1;
  ExcludeSpecialTokens: boolean = true): TNNetPerplexityStats;
var
  InV: TNNetVolume;
  Last: TNNetLayer;
  ContextLen, VocabSize: integer;
  LineIdx, SampleLen, Pos, Tgt: integer;
  Line: string;
  RowSum, Prob, SumNLL: TNeuralFloat;
begin
  ZeroStats(Result);
  if (NN = nil) or (Corpus = nil) then Exit;
  if NN.CountLayers() < 2 then Exit;
  if MinContext < 1 then MinContext := 1;
  ContextLen := NN.GetFirstLayer().Output.SizeX;
  Last := NN.GetLastLayer();
  VocabSize := Last.Output.Size; // single next-char distribution (v1)
  if VocabSize < 2 then Exit;
  SumNLL := 0;
  InV := TNNetVolume.Create(NN.GetFirstLayer().Output);
  try
    for LineIdx := 0 to Corpus.Count - 1 do
    begin
      Line := Corpus[LineIdx];
      SampleLen := Length(Line);
      // Predict char at (1-based) Pos from the prefix 1..Pos-1; the prefix
      // must fit the window (v1 truncation: Pos-1 <= ContextLen) and be at
      // least MinContext chars long.
      for Pos := MinContext + 1 to Min(SampleLen, ContextLen + 1) do
      begin
        Tgt := Ord(Line[Pos]);
        if SkipTarget(Tgt, VocabSize, ExcludeSpecialTokens) then
        begin
          Inc(Result.SkippedTokens);
          continue;
        end;
        InV.OneHotEncodingReversed(Copy(Line, 1, Pos - 1));
        NN.Compute(InV);
        RowSum := Last.Output.GetSum();
        if RowSum <= 0 then RowSum := 1.0;
        Prob := Last.Output.FData[Tgt] / RowSum;
        SumNLL := SumNLL - SafeLogProb(Prob);
        Inc(Result.PredictedTokens);
      end;
    end;
  finally
    InV.Free;
  end;
  FinishStats(SumNLL, Result);
end;

// ---------------------------------------------------------------------------
// Token-level logprob scoring + multiple-choice harness
// ---------------------------------------------------------------------------

function ScoreSequence(NN: TNNet;
  const Tokens: TNeuralIntegerArray): TNeuralFloatDynArr;
var
  InV: TNNetVolume;
  Last: TNNetLayer;
  Prefix: TNeuralIntegerArray;
  ContextLen, InDepth, VocabSize: integer;
  PerPosition: boolean;
  SampleLen, Pos, D, Tgt: integer;
  RowSum, Prob: TNeuralFloat;
begin
  SetLength(Result, 0);
  if NN = nil then Exit;
  if NN.CountLayers() < 2 then Exit;
  SampleLen := Length(Tokens);
  if SampleLen = 0 then Exit;
  ContextLen := NN.GetFirstLayer().Output.SizeX;
  InDepth := NN.GetFirstLayer().Output.Depth;
  Last := NN.GetLastLayer();
  // Same head-shape auto-detection as Perplexity (see unit header).
  PerPosition := (ContextLen > 1) and (Last.Output.SizeX = ContextLen) and
    (Last.Output.Depth >= 2);
  if PerPosition
  then VocabSize := Last.Output.Depth
  else VocabSize := Last.Output.Size;
  if VocabSize < 2 then Exit;
  // v1 context policy: error clearly instead of silently sub-windowing.
  // Per-position heads need the whole sequence in one window; single
  // next-token heads need the longest scored prefix (SampleLen-1) to fit.
  if (PerPosition and (SampleLen > ContextLen)) or
     ((not PerPosition) and (SampleLen - 1 > ContextLen)) then
    raise EArgumentException.CreateFmt(
      'ScoreSequence: sequence length %d exceeds the model context window %d',
      [SampleLen, ContextLen]);
  SetLength(Result, SampleLen);
  Result[0] := 0; // first token has no conditioning context - never scored
  if SampleLen < 2 then Exit;
  InV := TNNetVolume.Create(NN.GetFirstLayer().Output);
  try
    if PerPosition then
    begin
      // ONE teacher-forced forward scores every position at once.
      InV.Fill(0);
      if InDepth = 1
      then InV.CopyNoChecksIntArr(Tokens)       // token ids -> embedding
      else InV.OneHotEncoding(Tokens);          // one-hot, left-aligned
      NN.Compute(InV);
      for Pos := 1 to SampleLen - 1 do
      begin
        Tgt := Tokens[Pos];
        if (Tgt < 0) or (Tgt >= VocabSize) then
        begin
          Result[Pos] := SafeLogProb(0); // out-of-vocab: clamped, not skipped
          continue;
        end;
        // Output row Pos-1 predicts token Pos (the causal shift). Defensive
        // re-normalisation, exactly like Perplexity.
        RowSum := 0;
        for D := 0 to VocabSize - 1 do
          RowSum := RowSum + Last.Output[Pos - 1, 0, D];
        if RowSum <= 0 then RowSum := 1.0;
        Prob := Last.Output[Pos - 1, 0, Tgt] / RowSum;
        Result[Pos] := SafeLogProb(Prob);
      end;
    end
    else
    begin
      // Single next-token head: one forward per scored position over the
      // growing right-aligned prefix.
      for Pos := 1 to SampleLen - 1 do
      begin
        Tgt := Tokens[Pos];
        if (Tgt < 0) or (Tgt >= VocabSize) then
        begin
          Result[Pos] := SafeLogProb(0);
          continue;
        end;
        Prefix := Copy(Tokens, 0, Pos);
        if InDepth = 1 then
        begin
          InV.Fill(0);
          InV.CopyReversedNoChecksIntArr(Prefix);
        end
        else InV.OneHotEncodingReversed(Prefix);
        NN.Compute(InV);
        RowSum := Last.Output.GetSum();
        if RowSum <= 0 then RowSum := 1.0;
        Prob := Last.Output.FData[Tgt] / RowSum;
        Result[Pos] := SafeLogProb(Prob);
      end;
    end;
  finally
    InV.Free;
  end;
  SetLength(Prefix, 0);
end;

function ScoreCompletion(NN: TNNet;
  const ContextTokens, CompletionTokens: TNeuralIntegerArray): TNNetCompletionScore;
var
  Full: TNeuralIntegerArray;
  LogProbs: TNeuralFloatDynArr;
  CtxLen, CompLen, I: integer;
begin
  Result.SumLogProb := 0;
  Result.MeanLogProb := 0;
  Result.TokenCount := 0;
  CtxLen := Length(ContextTokens);
  CompLen := Length(CompletionTokens);
  if CtxLen < 1 then
    raise EArgumentException.Create(
      'ScoreCompletion: ContextTokens must be non-empty (the first ' +
      'completion token is scored from the last context position)');
  if CompLen = 0 then Exit;
  SetLength(Full, CtxLen + CompLen);
  for I := 0 to CtxLen - 1 do Full[I] := ContextTokens[I];
  for I := 0 to CompLen - 1 do Full[CtxLen + I] := CompletionTokens[I];
  LogProbs := ScoreSequence(NN, Full);
  if Length(LogProbs) <> CtxLen + CompLen then Exit; // degenerate model
  // Completion tokens ONLY: indices CtxLen .. CtxLen+CompLen-1.
  for I := CtxLen to CtxLen + CompLen - 1 do
    Result.SumLogProb := Result.SumLogProb + LogProbs[I];
  Result.TokenCount := CompLen;
  Result.MeanLogProb := Result.SumLogProb / CompLen;
  SetLength(Full, 0);
  SetLength(LogProbs, 0);
end;

function EvaluateMultipleChoice(NN: TNNet;
  const Items: array of TNNetMultipleChoiceItem): TNNetMultipleChoiceStats;
var
  ItemIdx, Cand, BestSum, BestNorm: integer;
  Score: TNNetCompletionScore;
  BestSumLP, BestNormLP: TNeuralFloat;
begin
  Result.Accuracy := 0;
  Result.AccuracyNorm := 0;
  Result.ItemCount := 0;
  Result.CorrectCount := 0;
  Result.CorrectNormCount := 0;
  if NN = nil then Exit;
  for ItemIdx := 0 to High(Items) do
  begin
    if Length(Items[ItemIdx].Candidates) = 0 then continue;
    BestSum := 0;
    BestNorm := 0;
    BestSumLP := 0;
    BestNormLP := 0;
    for Cand := 0 to High(Items[ItemIdx].Candidates) do
    begin
      Score := ScoreCompletion(NN, Items[ItemIdx].ContextTokens,
        Items[ItemIdx].Candidates[Cand]);
      // First-max tie-break: a later candidate must be STRICTLY better.
      if (Cand = 0) or (Score.SumLogProb > BestSumLP) then
      begin
        BestSumLP := Score.SumLogProb;
        BestSum := Cand;
      end;
      if (Cand = 0) or (Score.MeanLogProb > BestNormLP) then
      begin
        BestNormLP := Score.MeanLogProb;
        BestNorm := Cand;
      end;
    end;
    Inc(Result.ItemCount);
    if BestSum = Items[ItemIdx].GoldIndex then Inc(Result.CorrectCount);
    if BestNorm = Items[ItemIdx].GoldIndex then Inc(Result.CorrectNormCount);
  end;
  if Result.ItemCount > 0 then
  begin
    Result.Accuracy := Result.CorrectCount / Result.ItemCount;
    Result.AccuracyNorm := Result.CorrectNormCount / Result.ItemCount;
  end;
end;

// ---------------------------------------------------------------------------
// n-gram machinery (shared by BLEU and ROUGE-N)
// ---------------------------------------------------------------------------

// Builds a sorted "ngram-key -> count" map for all N-grams of Tokens. Keys
// are the ids joined with commas; counts live in Objects[] as PtrInt.
function CountNGrams(const Tokens: TNeuralIntegerArray; N: integer): TStringList;
var
  Start, Idx, KeyPos: integer;
  Key: string;
begin
  Result := TStringList.Create();
  Result.Sorted := true;
  Result.CaseSensitive := true;
  for Start := 0 to Length(Tokens) - N do
  begin
    Key := IntToStr(Tokens[Start]);
    for Idx := 1 to N - 1 do
      Key := Key + ',' + IntToStr(Tokens[Start + Idx]);
    if Result.Find(Key, KeyPos)
    then Result.Objects[KeyPos] := TObject(PtrInt(Result.Objects[KeyPos]) + 1)
    else Result.AddObject(Key, TObject(PtrInt(1)));
  end;
end;

// Clipped (modified-precision) overlap: sum over candidate n-gram types of
// min(count in candidate, count in reference).
function ClippedOverlap(CandCounts, RefCounts: TStringList): integer;
var
  Idx, RefPos: integer;
begin
  Result := 0;
  for Idx := 0 to CandCounts.Count - 1 do
    if RefCounts.Find(CandCounts[Idx], RefPos) then
      Result := Result + Min(PtrInt(CandCounts.Objects[Idx]),
        PtrInt(RefCounts.Objects[RefPos]));
end;

// Whitespace tokenization into ids through a shared (growing) vocabulary, so
// candidate and reference words map to the same ids. Vocab is sorted; the id
// of each word is stored in Objects[] when first seen.
procedure TokenizeWithVocab(const Text: string; Vocab: TStringList;
  var Ids: TNeuralIntegerArray);
var
  CharIdx, WordPos, IdCount: integer;
  CurWord: string;
  procedure PushWord();
  begin
    if CurWord = '' then Exit;
    if not Vocab.Find(CurWord, WordPos) then
      // New word: its id is the number of distinct words seen so far.
      WordPos := Vocab.AddObject(CurWord, TObject(PtrInt(Vocab.Count)));
    IdCount := Length(Ids);
    SetLength(Ids, IdCount + 1);
    Ids[IdCount] := PtrInt(Vocab.Objects[WordPos]);
    CurWord := '';
  end;
begin
  SetLength(Ids, 0);
  CurWord := '';
  for CharIdx := 1 to Length(Text) do
    if Text[CharIdx] <= ' ' then PushWord()
    else CurWord := CurWord + Text[CharIdx];
  PushWord();
end;

// ---------------------------------------------------------------------------
// BLEU
// ---------------------------------------------------------------------------

function CorpusBLEU(const Candidates, References: array of TNeuralIntegerArray;
  MaxN: integer = 4; Smooth: boolean = true): TNeuralFloat;
var
  Matches, Totals: array of int64;
  CandCounts, RefCounts: TStringList;
  PairIdx, Order, UsedOrders: integer;
  CandLen, RefLen: int64;
  Precision, SumLogP, BrevityPenalty: TNeuralFloat;
begin
  Result := 0;
  if (Length(Candidates) = 0) or
     (Length(Candidates) <> Length(References)) or (MaxN < 1) then Exit;
  SetLength(Matches, MaxN + 1);
  SetLength(Totals, MaxN + 1);
  for Order := 1 to MaxN do
  begin
    Matches[Order] := 0;
    Totals[Order] := 0;
  end;
  CandLen := 0;
  RefLen := 0;
  // Pool clipped matches and candidate n-gram totals over the whole corpus
  // (corpus BLEU is NOT an average of sentence BLEUs).
  for PairIdx := 0 to High(Candidates) do
  begin
    CandLen := CandLen + Length(Candidates[PairIdx]);
    RefLen := RefLen + Length(References[PairIdx]);
    for Order := 1 to MaxN do
    begin
      if Length(Candidates[PairIdx]) < Order then continue;
      CandCounts := CountNGrams(Candidates[PairIdx], Order);
      RefCounts := CountNGrams(References[PairIdx], Order);
      try
        Matches[Order] := Matches[Order] + ClippedOverlap(CandCounts, RefCounts);
        Totals[Order] := Totals[Order] +
          (Length(Candidates[PairIdx]) - Order + 1);
      finally
        CandCounts.Free;
        RefCounts.Free;
      end;
    end;
  end;
  if CandLen = 0 then Exit;
  SumLogP := 0;
  UsedOrders := 0;
  for Order := 1 to MaxN do
  begin
    if Totals[Order] = 0 then continue; // order not measurable -> excluded
    if Smooth and (Order >= 2) then
      // Lin & Och (2004) smoothing-1: add-1 to numerator and denominator
      // for every order above unigrams.
      Precision := (Matches[Order] + 1) / (Totals[Order] + 1)
    else
      Precision := Matches[Order] / Totals[Order];
    if Precision <= 0 then Exit; // unsmoothed zero precision -> BLEU = 0
    SumLogP := SumLogP + Ln(Precision);
    Inc(UsedOrders);
  end;
  if UsedOrders = 0 then Exit;
  if CandLen >= RefLen
  then BrevityPenalty := 1.0
  else BrevityPenalty := Exp(1.0 - RefLen / CandLen);
  Result := BrevityPenalty * Exp(SumLogP / UsedOrders);
end;

function CorpusBLEU(const Candidates, References: array of string;
  MaxN: integer = 4; Smooth: boolean = true): TNeuralFloat;
var
  Vocab: TStringList;
  CandIds, RefIds: array of TNeuralIntegerArray;
  PairIdx: integer;
begin
  Result := 0;
  if (Length(Candidates) = 0) or
     (Length(Candidates) <> Length(References)) then Exit;
  Vocab := TStringList.Create();
  Vocab.Sorted := true;
  Vocab.CaseSensitive := true;
  try
    SetLength(CandIds, Length(Candidates));
    SetLength(RefIds, Length(References));
    for PairIdx := 0 to High(Candidates) do
    begin
      TokenizeWithVocab(Candidates[PairIdx], Vocab, CandIds[PairIdx]);
      TokenizeWithVocab(References[PairIdx], Vocab, RefIds[PairIdx]);
    end;
    Result := CorpusBLEU(CandIds, RefIds, MaxN, Smooth);
  finally
    Vocab.Free;
  end;
end;

// ---------------------------------------------------------------------------
// ROUGE
// ---------------------------------------------------------------------------

function MakeRouge(MatchCount, CandTotal, RefTotal: integer): TNNetRougeScore;
begin
  if CandTotal > 0
  then Result.Precision := MatchCount / CandTotal
  else Result.Precision := 0;
  if RefTotal > 0
  then Result.Recall := MatchCount / RefTotal
  else Result.Recall := 0;
  if Result.Precision + Result.Recall > 0
  then Result.F1 := 2 * Result.Precision * Result.Recall /
    (Result.Precision + Result.Recall)
  else Result.F1 := 0;
end;

function RougeN(const Candidate, Reference: TNeuralIntegerArray;
  N: integer): TNNetRougeScore;
var
  CandCounts, RefCounts: TStringList;
  Overlap, CandTotal, RefTotal: integer;
begin
  Result := MakeRouge(0, 0, 0);
  if N < 1 then Exit;
  CandTotal := Max(0, Length(Candidate) - N + 1);
  RefTotal := Max(0, Length(Reference) - N + 1);
  if (CandTotal = 0) or (RefTotal = 0) then
  begin
    Result := MakeRouge(0, CandTotal, RefTotal);
    Exit;
  end;
  CandCounts := CountNGrams(Candidate, N);
  RefCounts := CountNGrams(Reference, N);
  try
    Overlap := ClippedOverlap(CandCounts, RefCounts);
  finally
    CandCounts.Free;
    RefCounts.Free;
  end;
  Result := MakeRouge(Overlap, CandTotal, RefTotal);
end;

function RougeN(const Candidate, Reference: string;
  N: integer): TNNetRougeScore;
var
  Vocab: TStringList;
  CandIds, RefIds: TNeuralIntegerArray;
begin
  Vocab := TStringList.Create();
  Vocab.Sorted := true;
  Vocab.CaseSensitive := true;
  try
    TokenizeWithVocab(Candidate, Vocab, CandIds);
    TokenizeWithVocab(Reference, Vocab, RefIds);
    Result := RougeN(CandIds, RefIds, N);
  finally
    Vocab.Free;
  end;
end;

// Classic O(|cand|*|ref|) two-row LCS length.
function LCSLength(const A, B: TNeuralIntegerArray): integer;
var
  Prev, Curr: array of integer;
  RowIdx, ColIdx: integer;
begin
  Result := 0;
  if (Length(A) = 0) or (Length(B) = 0) then Exit;
  SetLength(Prev, Length(B) + 1);
  SetLength(Curr, Length(B) + 1);
  for ColIdx := 0 to Length(B) do Prev[ColIdx] := 0;
  for RowIdx := 1 to Length(A) do
  begin
    Curr[0] := 0;
    for ColIdx := 1 to Length(B) do
      if A[RowIdx - 1] = B[ColIdx - 1]
      then Curr[ColIdx] := Prev[ColIdx - 1] + 1
      else Curr[ColIdx] := Max(Prev[ColIdx], Curr[ColIdx - 1]);
    Prev := Copy(Curr, 0, Length(Curr));
  end;
  Result := Prev[Length(B)];
end;

function RougeL(const Candidate, Reference: TNeuralIntegerArray): TNNetRougeScore;
begin
  Result := MakeRouge(LCSLength(Candidate, Reference),
    Length(Candidate), Length(Reference));
end;

function RougeL(const Candidate, Reference: string): TNNetRougeScore;
var
  Vocab: TStringList;
  CandIds, RefIds: TNeuralIntegerArray;
begin
  Vocab := TStringList.Create();
  Vocab.Sorted := true;
  Vocab.CaseSensitive := true;
  try
    TokenizeWithVocab(Candidate, Vocab, CandIds);
    TokenizeWithVocab(Reference, Vocab, RefIds);
    Result := RougeL(CandIds, RefIds);
  finally
    Vocab.Free;
  end;
end;

function CorpusRougeN(const Candidates, References: array of string;
  N: integer): TNNetRougeScore;
var
  PairIdx: integer;
  PairScore: TNNetRougeScore;
begin
  Result := MakeRouge(0, 0, 0);
  if (Length(Candidates) = 0) or
     (Length(Candidates) <> Length(References)) then Exit;
  for PairIdx := 0 to High(Candidates) do
  begin
    PairScore := RougeN(Candidates[PairIdx], References[PairIdx], N);
    Result.Precision := Result.Precision + PairScore.Precision;
    Result.Recall := Result.Recall + PairScore.Recall;
    Result.F1 := Result.F1 + PairScore.F1;
  end;
  Result.Precision := Result.Precision / Length(Candidates);
  Result.Recall := Result.Recall / Length(Candidates);
  Result.F1 := Result.F1 / Length(Candidates);
end;

function CorpusRougeL(const Candidates, References: array of string): TNNetRougeScore;
var
  PairIdx: integer;
  PairScore: TNNetRougeScore;
begin
  Result := MakeRouge(0, 0, 0);
  if (Length(Candidates) = 0) or
     (Length(Candidates) <> Length(References)) then Exit;
  for PairIdx := 0 to High(Candidates) do
  begin
    PairScore := RougeL(Candidates[PairIdx], References[PairIdx]);
    Result.Precision := Result.Precision + PairScore.Precision;
    Result.Recall := Result.Recall + PairScore.Recall;
    Result.F1 := Result.F1 + PairScore.F1;
  end;
  Result.Precision := Result.Precision / Length(Candidates);
  Result.Recall := Result.Recall / Length(Candidates);
  Result.F1 := Result.F1 / Length(Candidates);
end;

// ---------------------------------------------------------------------------
// chrF / chrF++
// ---------------------------------------------------------------------------

// Sorted "ngram -> count" multiset of all length-N character substrings of S.
function CountCharNGrams(const S: string; N: integer): TStringList;
var
  Start, KeyPos: integer;
  Key: string;
begin
  Result := TStringList.Create();
  Result.Sorted := true;
  Result.CaseSensitive := true;
  for Start := 1 to Length(S) - N + 1 do
  begin
    Key := Copy(S, Start, N);
    if Result.Find(Key, KeyPos)
    then Result.Objects[KeyPos] := TObject(PtrInt(Result.Objects[KeyPos]) + 1)
    else Result.AddObject(Key, TObject(PtrInt(1)));
  end;
end;

// Accumulates one effective order into the running F-score sum: given the
// clipped match count and the hypothesis / reference n-gram totals, adds the
// per-order F_beta (skipping the order entirely when both totals are zero).
procedure AddOrderF(Match, HypTotal, RefTotal: integer; Beta2: TNeuralFloat;
  var ScoreSum: TNeuralFloat; var EffOrders: integer);
var
  P, R, Denom: TNeuralFloat;
begin
  if (HypTotal = 0) and (RefTotal = 0) then Exit; // order not present at all
  if HypTotal > 0 then P := Match / HypTotal else P := 0;
  if RefTotal > 0 then R := Match / RefTotal else R := 0;
  Denom := Beta2 * P + R;
  if Denom > 0
  then ScoreSum := ScoreSum + (1 + Beta2) * P * R / Denom;
  // (Denom = 0 contributes a zero F but still counts as an effective order.)
  Inc(EffOrders);
end;

function ChrF(const Hypothesis, Reference: string;
  CharOrder: integer = 6; Beta: TNeuralFloat = 2.0;
  WordOrder: integer = 0; IncludeWhitespace: boolean = false): TNeuralFloat;
var
  H, R: string;
  HypW, RefW: TNeuralIntegerArray; // word-id sequences for chrF++
  Vocab: TStringList;
  N, ScoreSumOrders: integer;
  HypNG, RefNG: TStringList;
  HypWNG, RefWNG: TStringList;
  ScoreSum, Beta2: TNeuralFloat;

  // Strip whitespace (sacrebleu default) or keep it, per IncludeWhitespace.
  function Prep(const Src: string): string;
  var I: integer;
  begin
    if IncludeWhitespace then Exit(Src);
    Result := '';
    for I := 1 to Length(Src) do
      if Src[I] > ' ' then Result := Result + Src[I];
  end;

begin
  Result := 0;
  if CharOrder < 1 then CharOrder := 1;
  if Beta <= 0 then Beta := 2.0;
  Beta2 := Beta * Beta;
  H := Prep(Hypothesis);
  R := Prep(Reference);
  ScoreSum := 0;
  ScoreSumOrders := 0;
  // Character n-gram orders.
  for N := 1 to CharOrder do
  begin
    HypNG := CountCharNGrams(H, N);
    RefNG := CountCharNGrams(R, N);
    try
      AddOrderF(ClippedOverlap(HypNG, RefNG),
        // total counts = sum over types, but a multiset's total is just the
        // number of n-gram occurrences = max(0, len - n + 1).
        Max(0, Length(H) - N + 1), Max(0, Length(R) - N + 1),
        Beta2, ScoreSum, ScoreSumOrders);
    finally
      HypNG.Free;
      RefNG.Free;
    end;
  end;
  // Word n-gram orders (chrF++): tokenize through a shared vocab so identical
  // words map to identical ids, then reuse the BLEU/ROUGE n-gram machinery.
  if WordOrder > 0 then
  begin
    Vocab := TStringList.Create();
    Vocab.Sorted := true;
    Vocab.CaseSensitive := true;
    try
      // Word ids come from the ORIGINAL strings (before whitespace stripping).
      TokenizeWithVocab(Hypothesis, Vocab, HypW);
      TokenizeWithVocab(Reference, Vocab, RefW);
      for N := 1 to WordOrder do
      begin
        HypWNG := CountNGrams(HypW, N);
        RefWNG := CountNGrams(RefW, N);
        try
          AddOrderF(ClippedOverlap(HypWNG, RefWNG),
            Max(0, Length(HypW) - N + 1), Max(0, Length(RefW) - N + 1),
            Beta2, ScoreSum, ScoreSumOrders);
        finally
          HypWNG.Free;
          RefWNG.Free;
        end;
      end;
    finally
      Vocab.Free;
    end;
  end;
  if ScoreSumOrders > 0 then Result := ScoreSum / ScoreSumOrders;
end;

function ChrFpp(const Hypothesis, Reference: string;
  CharOrder: integer = 6; Beta: TNeuralFloat = 2.0): TNeuralFloat;
begin
  Result := ChrF(Hypothesis, Reference, CharOrder, Beta, 2, false);
end;

function CorpusChrF(const Hypotheses, References: array of string;
  CharOrder: integer = 6; Beta: TNeuralFloat = 2.0;
  WordOrder: integer = 0; IncludeWhitespace: boolean = false): TNeuralFloat;
var
  PairIdx: integer;
begin
  Result := 0;
  if (Length(Hypotheses) = 0) or
     (Length(Hypotheses) <> Length(References)) then Exit;
  for PairIdx := 0 to High(Hypotheses) do
    Result := Result + ChrF(Hypotheses[PairIdx], References[PairIdx],
      CharOrder, Beta, WordOrder, IncludeWhitespace);
  Result := Result / Length(Hypotheses);
end;

// ---------------------------------------------------------------------------
// Degeneration / generation-quality metrics
// ---------------------------------------------------------------------------

function DistinctN(const Tokens: TNeuralIntegerArray; N: integer): TNeuralFloat;
var
  Counts: TStringList;
  Total: integer;
begin
  Result := 0;
  if N < 1 then Exit;
  Total := Length(Tokens) - N + 1;
  if Total <= 0 then Exit;
  Counts := CountNGrams(Tokens, N); // sorted "ngram -> count" map
  try
    // distinct = number of TYPES (one StringList entry per distinct n-gram).
    Result := Counts.Count / Total;
  finally
    Counts.Free;
  end;
end;

function DistinctN(const Text: string; N: integer): TNeuralFloat;
var
  Vocab: TStringList;
  Ids: TNeuralIntegerArray;
begin
  Vocab := TStringList.Create();
  Vocab.Sorted := true;
  Vocab.CaseSensitive := true;
  try
    TokenizeWithVocab(Text, Vocab, Ids);
    Result := DistinctN(Ids, N);
  finally
    Vocab.Free;
  end;
end;

function RepetitionRate(const Tokens: TNeuralIntegerArray; N: integer): TNeuralFloat;
begin
  if Length(Tokens) - N + 1 <= 0 then Result := 0
  else Result := 1.0 - DistinctN(Tokens, N);
end;

function RepetitionRate(const Text: string; N: integer): TNeuralFloat;
var
  Vocab: TStringList;
  Ids: TNeuralIntegerArray;
begin
  Vocab := TStringList.Create();
  Vocab.Sorted := true;
  Vocab.CaseSensitive := true;
  try
    TokenizeWithVocab(Text, Vocab, Ids);
    Result := RepetitionRate(Ids, N);
  finally
    Vocab.Free;
  end;
end;

function RepeatedTokenRate(const Tokens: TNeuralIntegerArray): TNeuralFloat;
begin
  Result := RepetitionRate(Tokens, 1);
end;

function RepeatedTokenRate(const Text: string): TNeuralFloat;
begin
  Result := RepetitionRate(Text, 1);
end;

function SelfBLEU(const Generations: array of TNeuralIntegerArray;
  MaxN: integer = 4; Smooth: boolean = true): TNeuralFloat;
var
  I, J, NumOthers: integer;
  PerCand, OtherBleu: TNeuralFloat;
  Cand, Ref: array of TNeuralIntegerArray;
begin
  Result := 0;
  if Length(Generations) < 2 then Exit; // need at least one "other" reference
  SetLength(Cand, 1);
  SetLength(Ref, 1);
  for I := 0 to High(Generations) do
  begin
    // Mean single-reference BLEU of generation I against every OTHER one
    // (v1: CorpusBLEU is single-reference, so average instead of multi-ref
    // clipping - documented in the unit header).
    PerCand := 0;
    NumOthers := 0;
    Cand[0] := Generations[I];
    for J := 0 to High(Generations) do
    begin
      if J = I then continue;
      Ref[0] := Generations[J];
      OtherBleu := CorpusBLEU(Cand, Ref, MaxN, Smooth);
      PerCand := PerCand + OtherBleu;
      Inc(NumOthers);
    end;
    if NumOthers > 0 then Result := Result + PerCand / NumOthers;
  end;
  Result := Result / Length(Generations);
end;

function SelfBLEU(const Generations: array of string;
  MaxN: integer = 4; Smooth: boolean = true): TNeuralFloat;
var
  Vocab: TStringList;
  Ids: array of TNeuralIntegerArray;
  I: integer;
begin
  Result := 0;
  if Length(Generations) < 2 then Exit;
  Vocab := TStringList.Create();
  Vocab.Sorted := true;
  Vocab.CaseSensitive := true;
  try
    SetLength(Ids, Length(Generations));
    for I := 0 to High(Generations) do
      TokenizeWithVocab(Generations[I], Vocab, Ids[I]);
    Result := SelfBLEU(Ids, MaxN, Smooth);
  finally
    Vocab.Free;
  end;
end;

// ---------------------------------------------------------------------------
// Token classification (NER) entity-level metrics
// ---------------------------------------------------------------------------

// Splits a tag into prefix ('B'/'I'/'O') and type (after the first '-').
procedure SplitTag(const Tag: string; out Prefix, EntType: string);
var
  DashPos: integer;
begin
  if Tag = '' then begin Prefix := 'O'; EntType := ''; Exit; end;
  DashPos := Pos('-', Tag);
  if DashPos = 0 then
  begin
    Prefix := Tag;       // 'O' (or a bare tag treated as outside)
    EntType := '';
  end
  else
  begin
    Prefix := Copy(Tag, 1, DashPos - 1);
    EntType := Copy(Tag, DashPos + 1, MaxInt);
  end;
end;

function DecodeBIOEntities(const Tags: array of string): TNNetEntitySpanArray;
var
  I, Count: integer;
  Prefix, EntType: string;
  OpenType: string;
  OpenStart: integer;

  procedure CloseOpen(EndIdx: integer);
  begin
    if OpenStart >= 0 then
    begin
      SetLength(Result, Count + 1);
      Result[Count].EntityType := OpenType;
      Result[Count].TokenStart := OpenStart;
      Result[Count].TokenEnd := EndIdx;
      Inc(Count);
      OpenStart := -1;
      OpenType := '';
    end;
  end;

begin
  SetLength(Result, 0);
  Count := 0;
  OpenStart := -1;
  OpenType := '';
  for I := 0 to High(Tags) do
  begin
    SplitTag(Tags[I], Prefix, EntType);
    if Prefix = 'B' then
    begin
      CloseOpen(I - 1);
      OpenType := EntType;
      OpenStart := I;
    end
    else if Prefix = 'I' then
    begin
      // Continue only when the type matches the open span; otherwise the
      // IOB2 lenient rule opens a fresh span on this 'I-'.
      if (OpenStart >= 0) and (EntType = OpenType) then
        // extend (handled implicitly: span end advances on close)
      else
      begin
        CloseOpen(I - 1);
        OpenType := EntType;
        OpenStart := I;
      end;
    end
    else // 'O' or unknown -> outside
      CloseOpen(I - 1);
  end;
  CloseOpen(High(Tags));
end;

function SpanInArray(const Span: TNNetEntitySpan;
  const Arr: TNNetEntitySpanArray; const Used: array of boolean): integer;
var
  I: integer;
begin
  Result := -1;
  for I := 0 to High(Arr) do
    if (not Used[I]) and (Arr[I].EntityType = Span.EntityType) and
       (Arr[I].TokenStart = Span.TokenStart) and
       (Arr[I].TokenEnd = Span.TokenEnd) then
      Exit(I);
end;

procedure CountEntityMatches(const PredTags, GoldTags: array of string;
  var TP, FP, FN: integer);
var
  Pred, Gold: TNNetEntitySpanArray;
  Used: array of boolean;
  I, MatchIdx: integer;
begin
  Pred := DecodeBIOEntities(PredTags);
  Gold := DecodeBIOEntities(GoldTags);
  SetLength(Used, Length(Gold));
  for I := 0 to High(Used) do Used[I] := false;
  for I := 0 to High(Pred) do
  begin
    MatchIdx := SpanInArray(Pred[I], Gold, Used);
    if MatchIdx >= 0 then
    begin
      Inc(TP);
      Used[MatchIdx] := true;
    end
    else
      Inc(FP);
  end;
  for I := 0 to High(Used) do
    if not Used[I] then Inc(FN);
end;

procedure FinishEntityScore(var S: TNNetEntityScore);
begin
  if (S.TruePos + S.FalsePos) > 0 then
    S.Precision := S.TruePos / (S.TruePos + S.FalsePos)
  else S.Precision := 0;
  if (S.TruePos + S.FalseNeg) > 0 then
    S.Recall := S.TruePos / (S.TruePos + S.FalseNeg)
  else S.Recall := 0;
  if (S.Precision + S.Recall) > 0 then
    S.F1 := 2 * S.Precision * S.Recall / (S.Precision + S.Recall)
  else S.F1 := 0;
end;

function EntityScore(const PredTags, GoldTags: array of string): TNNetEntityScore;
begin
  if Length(PredTags) <> Length(GoldTags) then
    raise EArgumentException.Create(
      'EntityScore: PredTags and GoldTags must have the same length.');
  Result.TruePos := 0;
  Result.FalsePos := 0;
  Result.FalseNeg := 0;
  CountEntityMatches(PredTags, GoldTags,
    Result.TruePos, Result.FalsePos, Result.FalseNeg);
  FinishEntityScore(Result);
end;

function CorpusEntityScore(const Pred, Gold: array of TStringArray): TNNetEntityScore;
var
  I: integer;
begin
  if Length(Pred) <> Length(Gold) then
    raise EArgumentException.Create(
      'CorpusEntityScore: Pred and Gold must have the same number of sentences.');
  Result.TruePos := 0;
  Result.FalsePos := 0;
  Result.FalseNeg := 0;
  for I := 0 to High(Pred) do
  begin
    if Length(Pred[I]) <> Length(Gold[I]) then
      raise EArgumentException.Create(
        'CorpusEntityScore: sentence ' + IntToStr(I) +
        ' Pred/Gold length mismatch.');
    CountEntityMatches(Pred[I], Gold[I],
      Result.TruePos, Result.FalsePos, Result.FalseNeg);
  end;
  FinishEntityScore(Result);
end;

// ---------------------------------------------------------------------------
// QA span extraction
// ---------------------------------------------------------------------------

// Returns the indices of the TopK largest values in Logits (descending by
// value; stable first-max tie-break by index). Fewer than TopK when the input
// is shorter.
function TopKIndices(const Logits: TNeuralFloatDynArr; TopK: integer): TNeuralIntegerArray;
var
  Order: TNeuralIntegerArray;
  I, J, Tmp, N: integer;
begin
  N := Length(Logits);
  SetLength(Order, N);
  for I := 0 to N - 1 do Order[I] := I;
  // simple selection sort by descending logit (N is small: TopK candidates)
  for I := 0 to N - 1 do
    for J := I + 1 to N - 1 do
      if Logits[Order[J]] > Logits[Order[I]] then
      begin
        Tmp := Order[I]; Order[I] := Order[J]; Order[J] := Tmp;
      end;
  if TopK > N then TopK := N;
  if TopK < 0 then TopK := 0;
  SetLength(Result, TopK);
  for I := 0 to TopK - 1 do Result[I] := Order[I];
end;

function ExtractQASpans(const StartLogits, EndLogits: TNeuralFloatDynArr;
  TopK: integer = 20; MaxAnswerLen: integer = 30;
  NBest: integer = 20): TNNetQASpanArray;
var
  StartIdx, EndIdx: TNeuralIntegerArray;
  I, J, S, E, Count, A, B, Tmp: integer;
  Cand: TNNetQASpanArray;
  TmpSpan: TNNetQASpan;
begin
  SetLength(Result, 0);
  if Length(StartLogits) <> Length(EndLogits) then
    raise EArgumentException.Create(
      'ExtractQASpans: StartLogits and EndLogits must have the same length.');
  if Length(StartLogits) = 0 then Exit;

  StartIdx := TopKIndices(StartLogits, TopK);
  EndIdx := TopKIndices(EndLogits, TopK);

  Count := 0;
  SetLength(Cand, Length(StartIdx) * Length(EndIdx));
  for I := 0 to High(StartIdx) do
    for J := 0 to High(EndIdx) do
    begin
      S := StartIdx[I];
      E := EndIdx[J];
      if E < S then Continue;
      if (MaxAnswerLen > 0) and ((E - S + 1) > MaxAnswerLen) then Continue;
      Cand[Count].TokenStart := S;
      Cand[Count].TokenEnd := E;
      Cand[Count].Score := StartLogits[S] + EndLogits[E];
      Inc(Count);
    end;
  SetLength(Cand, Count);

  // descending by score (stable selection sort; small candidate set)
  for A := 0 to Count - 1 do
    for B := A + 1 to Count - 1 do
      if Cand[B].Score > Cand[A].Score then
      begin
        TmpSpan := Cand[A]; Cand[A] := Cand[B]; Cand[B] := TmpSpan;
      end;

  Tmp := Count;
  if (NBest > 0) and (NBest < Tmp) then Tmp := NBest;
  SetLength(Result, Tmp);
  for A := 0 to Tmp - 1 do Result[A] := Cand[A];
end;

end.
