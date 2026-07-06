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

  PerplexityStrided(NN, Dict, Corpus, Stride): the HF "Perplexity of fixed-
  length models" recipe for the per-position head. The corpus is tokenized and
  CONCATENATED into one stream, then a window of length W (input SizeX) slides
  over it with the given Stride. Each window runs one forward but only scores
  the targets the previous window did not, so every token past the first
  window carries up to W-1 tokens of real left context instead of the disjoint
  chop. Stride = W reproduces the disjoint Perplexity() baseline EXACTLY (each
  window's first token unscored); Stride < W re-scores those window-first
  tokens too (a SUPERSET of the disjoint set - every stream position is scored
  exactly once), so MeanNLL can only drop for a model with genuine long-range
  structure. Per-position heads only (single next-token heads already see the
  full prefix per position).

  Shared conventions (both functions):
  - TRUNCATION (v1): each sample is truncated to the model's context window
    (input SizeX); positions beyond the window are NOT scored (no sliding
    re-evaluation - PerplexityStrided is the sliding-window remedy). Documented
    trade-off: simple, cheap, and unbiased for samples that fit the window.
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
It scores each item's candidates through ScoreCompletionsBatch, which shares
the common context prefix: for single next-token heads only the completion
positions are forwarded per candidate (the shared-context forwards are not
re-run), giving scores IDENTICAL to per-candidate ScoreCompletion.

CONTEXT OVERFLOW. ScoreSequence / ScoreCompletion take an optional
LastWindow flag (default false = the v1 raise-on-overflow policy). With
LastWindow=true an over-context sequence is scored over the trailing
context-window ending at each position (the standard sliding-window LM eval)
instead of raising; sequences that already fit score identically either way.

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

  TNNetCompletionScoreArray = array of TNNetCompletionScore;

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

// Strided sliding-window perplexity (HF "Perplexity of fixed-length models"
// recipe) for per-position teacher-forced LMs. The whole Corpus is tokenized
// and CONCATENATED into one token stream, then a window of length W (the model
// context = input SizeX) is slid over the stream with the given Stride. Each
// window runs ONE forward; only the target positions NOT already scored by the
// previous window are counted, so every scored token after the first window
// carries up to W-1 tokens of real left context (instead of the disjoint-
// window chop the per-line truncation in Perplexity() implies). With
// Stride = W the windows are disjoint and this reproduces the disjoint
// baseline EXACTLY (each window's first token is unscored, exactly the chop);
// with Stride < W every token past the first window is re-scored with MORE
// left context, so the MeanNLL can only drop for a model with genuine long-
// range structure. Each stream token is scored EXACTLY once (no double-
// counting): the scored set is identical to the disjoint baseline, only the
// available left context differs. Per-position heads only (single next-token
// heads already condition on the full prefix per position, so striding is a
// no-op); Stride is clamped to [1, W].
function PerplexityStrided(NN: TNNet; Dict: TStringListInt; Corpus: TStrings;
  Stride: integer; ExcludeSpecialTokens: boolean = true): TNNetPerplexityStats;

// Per-token log-probabilities ln p(Tokens[i] | Tokens[0..i-1]) of an
// already-tokenized sequence (teacher-forced; no generation). Result has the
// same length as Tokens; Result[0] is 0 (never scored).
//   LastWindow=false (default): a sequence that exceeds the model context
//     window raises EArgumentException (the v1 policy; see unit header).
//   LastWindow=true: instead of raising, each position is scored over the LAST
//     context-window of tokens ending at it (the standard sliding-window LM
//     eval) - position Pos conditions on tokens (Pos-ContextLen+1 .. Pos-1)
//     when its full left context would overflow. Sequences that already fit are
//     scored identically with or without the flag.
function ScoreSequence(NN: TNNet;
  const Tokens: TNeuralIntegerArray;
  LastWindow: boolean = false): TNeuralFloatDynArr;

// Sum + length-normalized logprob of CompletionTokens given ContextTokens.
// Only completion tokens are scored; ContextTokens must be non-empty.
// LastWindow forwards to ScoreSequence (over-context -> last-window scoring
// instead of raising).
function ScoreCompletion(NN: TNNet;
  const ContextTokens, CompletionTokens: TNeuralIntegerArray;
  LastWindow: boolean = false): TNNetCompletionScore;

// SHARED-PREFIX batch scoring: scores every candidate completion of ONE shared
// context, returning the same TNNetCompletionScore array as calling
// ScoreCompletion(NN, ContextTokens, Candidates[i]) candidate-by-candidate -
// but the single next-token head path scores ONLY the completion positions
// (the unused context-position forwards ScoreSequence would run are skipped, so
// the shared context is not re-forwarded for every candidate). Scores are
// IDENTICAL to the per-candidate path. LastWindow is honored per candidate.
function ScoreCompletionsBatch(NN: TNNet;
  const ContextTokens: TNeuralIntegerArray;
  const Candidates: array of TNeuralIntegerArray;
  LastWindow: boolean = false): TNNetCompletionScoreArray;

// Multiple-choice harness: scores every candidate completion of every item
// with ScoreCompletion and reports acc (sum-logprob argmax) and acc_norm
// (mean-logprob argmax). See unit header.
function EvaluateMultipleChoice(NN: TNNet;
  const Items: array of TNNetMultipleChoiceItem): TNNetMultipleChoiceStats;

// ---------------------------------------------------------------------------
// MMLU few-shot accuracy harness (single-token answer-letter scoring)
// ---------------------------------------------------------------------------
//
// MMLU (Hendrycks et al. 2021) is a 4-choice (A/B/C/D) knowledge benchmark.
// This harness follows the HF lm-evaluation-harness CONVENTION, which is
// DISTINCT from the HellaSwag full-continuation scoring above
// (EvaluateMultipleChoice): for each question the standard k-shot prompt
//
//   <k same-subject demos, each "Question..\nA. ..\nB. ..\nC. ..\nD. ..\nAnswer: X\n\n">
//   Question ..\nA. ..\nB. ..\nC. ..\nD. ..\nAnswer:
//
// is built and the model is scored by the log-probability of the SINGLE
// answer-letter token (" A" / " B" / " C" / " D") that would come next - NOT
// by the perplexity of the whole answer string. The four letter tokens are
// supplied by the caller (tokenizer-specific: each letter is encoded once and
// the harness uses the LAST token of that encoding as the answer token), so
// the harness itself is checkpoint-/tokenizer-agnostic.
//
// Each scored letter is one single-token completion through ScoreCompletion,
// so the prompt prefix carries the causal-shift / reversed-prefix encoding
// exactly like every other scorer in this unit; the argmax letter is the
// prediction (first-max tie-break). Per-subject accuracy is reported plus the
// MACRO average (mean over subjects, the headline MMLU number) and the MICRO
// average (pooled over all questions). 0-shot (no demos) and k-shot both run
// through the same path - the demo prompt is just prepended to PromptTokens by
// the caller / the example's prompt builder.

// One MMLU question, already tokenized by the caller. PromptTokens is the full
// k-shot prompt up to and including "Answer:" (or "Answer: " with the trailing
// space the caller's tokenizer prefers); the four answer-letter tokens are
// passed once in EvaluateMMLU. GoldLetter is 0..3 (A/B/C/D). SubjectIndex
// selects the per-subject bucket (0 .. number-of-subjects-1).
type
  TNNetMMLUQuestion = record
    PromptTokens: TNeuralIntegerArray;
    GoldLetter: integer;   // 0=A 1=B 2=C 3=D
    SubjectIndex: integer; // bucket for per-subject accuracy
  end;

  // Per-subject tally and accuracy.
  TNNetMMLUSubjectStat = record
    Correct: integer;
    Total: integer;
    Accuracy: TNeuralFloat; // Correct / Total (0 when Total = 0)
  end;
  TNNetMMLUSubjectStatArray = array of TNNetMMLUSubjectStat;

  // Aggregate MMLU result. MacroAccuracy is the mean of the per-subject
  // accuracies over the subjects that actually have questions (the headline
  // MMLU number); MicroAccuracy pools all questions (CorrectCount/ItemCount).
  TNNetMMLUStats = record
    MacroAccuracy: TNeuralFloat;
    MicroAccuracy: TNeuralFloat;
    ItemCount: integer;
    CorrectCount: integer;
    SubjectCount: integer;            // subjects with at least one question
    PerSubject: TNNetMMLUSubjectStatArray; // length = NumSubjects (see below)
  end;

// MMLU answer-letter scoring harness. For every question the four answer-letter
// tokens (LetterTokens[0..3], typically the ids of " A".." D") are each scored
// as a single-token completion of PromptTokens via ScoreCompletion; the highest
// log-probability letter is the prediction. NumSubjects sizes the per-subject
// table (questions with SubjectIndex outside 0..NumSubjects-1 are skipped).
// LastWindow forwards to ScoreCompletion (over-context prompts score over the
// trailing window instead of raising). Reports per-subject accuracy plus the
// macro (mean-over-subjects) and micro (pooled) averages.
function EvaluateMMLU(NN: TNNet;
  const Questions: array of TNNetMMLUQuestion;
  const LetterTokens: array of integer;
  NumSubjects: integer;
  LastWindow: boolean = false): TNNetMMLUStats;

// Formats a TNNetMMLUStats into a small multi-line report (the PerplexityReport
// / *Report idiom). SubjectNames must have NumSubjects entries (or be empty, in
// which case subjects are labelled "subject <i>"); ShotsK is printed in the
// header (0 = zero-shot, 5 = the headline five-shot setting).
function MMLUReport(const Stats: TNNetMMLUStats;
  const SubjectNames: array of string; ShotsK: integer): string;

// ---------------------------------------------------------------------------
// ARC / PIQA / WinoGrande - answer-letter / full-continuation scoring core
// ---------------------------------------------------------------------------
//
// These three benchmarks are plain multiple-choice tasks scored EXACTLY like
// HellaSwag/MMLU above: each item is a shared context plus N candidate answer
// CONTINUATIONS, every candidate is scored with ScoreCompletion (per-choice
// sequence log-likelihood), and the highest-scoring candidate is the
// prediction. They therefore reuse TNNetMultipleChoiceItem /
// EvaluateMultipleChoice wholesale - the only thing that differs per benchmark
// is the candidate count and WHICH accuracy field is the lm-eval-harness
// headline number:
//
//   * PIQA       - 2 choices (sol1 / sol2); headline = acc_norm (length-
//                  normalized; goal+solution continuations vary in length).
//   * WinoGrande - 2 choices (option1 / option2 substituted for the "_" blank);
//                  headline = acc (lm-eval scores the SUFFIX after the blank,
//                  so the continuations are equal length and acc == acc_norm in
//                  practice; acc is reported as the canonical number).
//   * ARC        - 4 labeled choices typically (some questions have 3 or 5);
//                  headline = acc_norm. Each item just supplies its own
//                  Candidates[] (variable length is fine - EvaluateMultipleChoice
//                  scores per item, so a 3- or 5-choice item works unchanged).
//
// The caller tokenizes the context + each candidate continuation following the
// benchmark's prompt convention (PIQA: "Goal: <goal>\nSolution: <sol>";
// WinoGrande: the sentence with option substituted, scored over the suffix
// after the blank; ARC: "Question: <q>\nAnswer: <choice text>"). These thin
// wrappers forward to EvaluateMultipleChoice and exist so call sites read as
// the benchmark they evaluate; the per-benchmark headline accuracy is the
// documented field of the returned TNNetMultipleChoiceStats. Coded by Claude (AI).
function EvaluateARC(NN: TNNet;
  const Items: array of TNNetMultipleChoiceItem): TNNetMultipleChoiceStats;
function EvaluatePIQA(NN: TNNet;
  const Items: array of TNNetMultipleChoiceItem): TNNetMultipleChoiceStats;
function EvaluateWinoGrande(NN: TNNet;
  const Items: array of TNNetMultipleChoiceItem): TNNetMultipleChoiceStats;

// Formats a TNNetMultipleChoiceStats into a small report (the *Report idiom).
// Title heads the block (e.g. 'PIQA', 'ARC-Challenge'); HeadlineNorm selects
// which line is annotated as the headline number (true = acc_norm, the
// ARC/PIQA convention; false = acc, the WinoGrande convention).
function MultipleChoiceReport(const Stats: TNNetMultipleChoiceStats;
  const Title: string; HeadlineNorm: boolean): string;

// ---------------------------------------------------------------------------
// LAMBADA - last-word prediction accuracy (greedy reproduction)
// ---------------------------------------------------------------------------
//
// LAMBADA (Paperno et al. 2016) measures whether a model, given a passage,
// predicts the EXACT final word. This is NOT a likelihood ranking task: the
// metric is greedy (argmax) reproduction of the target word token-by-token -
// every token of the final word must be the model's argmax continuation of the
// gold prefix (the standard lm-eval "acc" for lambada). FinalWordTokens is the
// tokenization of the final word (one or more tokens); ContextTokens is the
// passage up to but excluding the final word and MUST be non-empty (the first
// final-word token is predicted from the last context position). Both the
// argmax prediction path and the log-prob path go through the SAME forward used
// by ScoreSequence, so the next-token-head reversed-prefix encoding is matched
// automatically (a mismatch would pin accuracy at chance).
type
  TNNetLambadaExample = record
    ContextTokens: TNeuralIntegerArray;   // passage minus the final word
    FinalWordTokens: TNeuralIntegerArray; // tokens of the target final word
  end;

  // Aggregate LAMBADA result. Accuracy is the fraction of examples whose ENTIRE
  // final word was reproduced greedily (all final-word tokens argmax-correct).
  // Perplexity / MeanNLL are over the final-word tokens (teacher-forced), the
  // commonly co-reported lambada perplexity.
  TNNetLambadaStats = record
    Accuracy: TNeuralFloat;      // CorrectCount / ItemCount
    ItemCount: integer;
    CorrectCount: integer;       // whole final word greedily reproduced
    MeanNLL: TNeuralFloat;       // mean -ln p over final-word tokens
    Perplexity: TNeuralFloat;    // exp(MeanNLL)
    TokenCount: integer;         // final-word tokens scored (for perplexity)
  end;

// LAMBADA last-word accuracy harness. For every example the final word is
// predicted token-by-token greedily (teacher-forced on the gold tokens): an
// example counts correct only if EVERY final-word token is the model's argmax
// next token. Also accumulates the teacher-forced NLL of the final-word tokens
// for the co-reported last-word perplexity. LastWindow forwards to the scorer
// (over-context passages score over the trailing window instead of raising).
// Coded by Claude (AI).
function EvaluateLAMBADA(NN: TNNet;
  const Examples: array of TNNetLambadaExample;
  LastWindow: boolean = false): TNNetLambadaStats;

// Formats a TNNetLambadaStats into a small report (the *Report idiom).
function LambadaReport(const Stats: TNNetLambadaStats): string;

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

// Teacher-forced forward of one window of token ids through a per-position
// head, accumulating the NLL of the targets at window-relative positions
// FirstTgt..LastTgt (output row Pos-1 predicts token Pos). Shared by Perplexity
// and PerplexityStrided so the per-row softmax / re-normalisation / special-
// token math lives in exactly one place. InV is reused (caller owns it).
procedure ScorePerPositionWindow(NN: TNNet; InV: TNNetVolume; Last: TNNetLayer;
  const WindowToks: TNeuralIntegerArray; FirstTgt, LastTgt, InDepth,
  VocabSize: integer; ExcludeSpecial: boolean; var SumNLL: TNeuralFloat;
  var Stats: TNNetPerplexityStats);
var
  Pos, D, Tgt, VocabSizeM1: integer;
  RowSum, Prob: TNeuralFloat;
begin
  InV.Fill(0);
  if InDepth = 1
  then InV.CopyNoChecksIntArr(WindowToks)   // token ids -> embedding
  else InV.OneHotEncoding(WindowToks);       // one-hot, left-aligned
  NN.Compute(InV);
  VocabSizeM1 := VocabSize - 1;
  for Pos := FirstTgt to LastTgt do
  begin
    Tgt := WindowToks[Pos];
    if SkipTarget(Tgt, VocabSize, ExcludeSpecial) then
    begin
      Inc(Stats.SkippedTokens);
      continue;
    end;
    // Output row Pos-1 predicts token Pos. Defensive re-normalisation keeps
    // the math honest even for near-softmax heads.
    RowSum := 0;
    for D := 0 to VocabSizeM1 do
      RowSum := RowSum + Last.Output[Pos - 1, 0, D];
    if RowSum <= 0 then RowSum := 1.0;
    Prob := Last.Output[Pos - 1, 0, Tgt] / RowSum;
    SumNLL := SumNLL - SafeLogProb(Prob);
    Inc(Stats.PredictedTokens);
  end;
end;

function Perplexity(NN: TNNet; Dict: TStringListInt; Corpus: TStrings;
  ExcludeSpecialTokens: boolean = true): TNNetPerplexityStats;
var
  InV: TNNetVolume;
  Last: TNNetLayer;
  Toks, Prefix: TNeuralIntegerArray;
  ContextLen, InDepth, VocabSize: integer;
  PerPosition: boolean;
  LineIdx, SampleLen, ClippedLen, Pos, Tgt, CorpusCount: integer;
  CorpusCountM1, LastPos: integer;
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
    CorpusCount := Corpus.Count;
    CorpusCountM1 := CorpusCount - 1;
    for LineIdx := 0 to CorpusCountM1 do
    begin
      Dict.Tokenize(Corpus[LineIdx], Toks);
      SampleLen := Length(Toks);
      if SampleLen < 2 then continue; // nothing to predict
      if PerPosition then
      begin
        // v1 truncation: only the first ContextLen tokens are scored.
        ClippedLen := Min(SampleLen, ContextLen);
        Prefix := Copy(Toks, 0, ClippedLen);
        ScorePerPositionWindow(NN, InV, Last, Prefix, 1, ClippedLen - 1,
          InDepth, VocabSize, ExcludeSpecialTokens, SumNLL, Result);
      end
      else
      begin
        // Single next-token head: one forward per predicted position over
        // the growing right-aligned prefix (truncated to the window).
        LastPos := Min(SampleLen - 1, ContextLen);
        for Pos := 1 to LastPos do
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
  LineIdx, SampleLen, Pos, Tgt, CorpusCount: integer;
  CorpusCountM1, LastPos: integer;
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
    CorpusCount := Corpus.Count;
    CorpusCountM1 := CorpusCount - 1;
    for LineIdx := 0 to CorpusCountM1 do
    begin
      Line := Corpus[LineIdx];
      SampleLen := Length(Line);
      // Predict char at (1-based) Pos from the prefix 1..Pos-1; the prefix
      // must fit the window (v1 truncation: Pos-1 <= ContextLen) and be at
      // least MinContext chars long.
      LastPos := Min(SampleLen, ContextLen + 1);
      for Pos := MinContext + 1 to LastPos do
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

function PerplexityStrided(NN: TNNet; Dict: TStringListInt; Corpus: TStrings;
  Stride: integer; ExcludeSpecialTokens: boolean = true): TNNetPerplexityStats;
var
  InV: TNNetVolume;
  Last: TNNetLayer;
  Stream, Toks, Window: TNeuralIntegerArray;
  ContextLen, InDepth, VocabSize: integer;
  PerPosition: boolean;
  LineIdx, StreamLen, WinStart, WinLen, FirstTgt, LastTgt: integer;
  CorpusCount, CorpusCountM1: integer;
  PrevEndAbs: integer; // last ABSOLUTE stream position already scored
  SumNLL: TNeuralFloat;
begin
  ZeroStats(Result);
  if (NN = nil) or (Dict = nil) or (Corpus = nil) then Exit;
  if NN.CountLayers() < 2 then Exit;
  ContextLen := NN.GetFirstLayer().Output.SizeX;
  InDepth := NN.GetFirstLayer().Output.Depth;
  Last := NN.GetLastLayer();
  // Strided sliding window only applies to per-position teacher-forced heads;
  // see the unit header for the head-shape auto-detection.
  PerPosition := (ContextLen > 1) and (Last.Output.SizeX = ContextLen) and
    (Last.Output.Depth >= 2);
  if not PerPosition then Exit;
  VocabSize := Last.Output.Depth;
  if VocabSize < 2 then Exit;
  // Clamp the stride into [1, W]: Stride = W is the disjoint baseline.
  if Stride < 1 then Stride := 1;
  if Stride > ContextLen then Stride := ContextLen;
  // Concatenate the whole corpus into one token stream (the HF recipe scores
  // the corpus as a single sequence, not per line).
  StreamLen := 0;
  CorpusCount := Corpus.Count;
  CorpusCountM1 := CorpusCount - 1;
  for LineIdx := 0 to CorpusCountM1 do
  begin
    Dict.Tokenize(Corpus[LineIdx], Toks);
    if Length(Toks) = 0 then continue;
    SetLength(Stream, StreamLen + Length(Toks));
    Move(Toks[0], Stream[StreamLen], Length(Toks) * csIntegerSize);
    StreamLen := StreamLen + Length(Toks);
  end;
  if StreamLen < 2 then Exit;
  SumNLL := 0;
  // PrevEndAbs tracks the last stream position already scored. The first
  // window scores all its predictable positions (1..WinLen-1); each later
  // window scores only positions strictly past PrevEndAbs, so every token is
  // scored exactly once and (after the first window) with bounded left
  // context (its window start sits up to W-1 tokens before it).
  PrevEndAbs := 0; // position 0 has no left context and is never a target
  InV := TNNetVolume.Create(NN.GetFirstLayer().Output);
  try
    WinStart := 0;
    while WinStart < StreamLen - 1 do
    begin
      WinLen := Min(ContextLen, StreamLen - WinStart);
      Window := Copy(Stream, WinStart, WinLen);
      // Window-relative target positions to score (1..WinLen-1 absolute
      // WinStart+1..WinStart+WinLen-1). Skip the ones already scored by a
      // previous window so no token is double-counted.
      FirstTgt := Max(1, PrevEndAbs + 1 - WinStart);
      LastTgt := WinLen - 1;
      if LastTgt >= FirstTgt then
      begin
        ScorePerPositionWindow(NN, InV, Last, Window, FirstTgt, LastTgt,
          InDepth, VocabSize, ExcludeSpecialTokens, SumNLL, Result);
        PrevEndAbs := WinStart + LastTgt;
      end;
      if WinStart + WinLen >= StreamLen then break; // last window reached
      WinStart := WinStart + Stride;
    end;
  finally
    InV.Free;
  end;
  SetLength(Stream, 0);
  SetLength(Toks, 0);
  SetLength(Window, 0);
  FinishStats(SumNLL, Result);
end;

// ---------------------------------------------------------------------------
// Token-level logprob scoring + multiple-choice harness
// ---------------------------------------------------------------------------

// Internal worker behind ScoreSequence / ScoreCompletionsBatch. Fills Result
// for scored positions in [max(1,FirstScored) .. High(Tokens)] and leaves all
// earlier entries at 0. FirstScored lets the shared-prefix batch path skip the
// (unused) context positions for single next-token heads, scoring ONLY the
// completion tokens while producing identical per-token log-probs. LastWindow
// switches over-context sequences from raising to trailing-window scoring.
function ScoreSequenceFrom(NN: TNNet;
  const Tokens: TNeuralIntegerArray; FirstScored: integer;
  LastWindow: boolean): TNeuralFloatDynArr;
var
  InV: TNNetVolume;
  Last: TNNetLayer;
  Prefix: TNeuralIntegerArray;
  ContextLen, InDepth, VocabSize: integer;
  PerPosition, Overflows: boolean;
  SampleLen, Pos, D, Tgt, FirstPos, WinStart, WinLen, Row: integer;
  SampleLenM1, VocabSizeM1: integer;
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
  // Per-position heads need the whole sequence in one window; single next-token
  // heads need the longest scored prefix (SampleLen-1) to fit.
  Overflows := (PerPosition and (SampleLen > ContextLen)) or
               ((not PerPosition) and (SampleLen - 1 > ContextLen));
  // v1 context policy: error clearly instead of silently sub-windowing -
  // UNLESS the caller opted into last-window scoring.
  if Overflows and (not LastWindow) then
    raise EArgumentException.CreateFmt(
      'ScoreSequence: sequence length %d exceeds the model context window %d',
      [SampleLen, ContextLen]);
  SetLength(Result, SampleLen);
  SampleLenM1 := SampleLen - 1;
  VocabSizeM1 := VocabSize - 1;
  for Pos := 0 to SampleLenM1 do Result[Pos] := 0;
  if SampleLen < 2 then Exit;
  FirstPos := FirstScored;
  if FirstPos < 1 then FirstPos := 1;
  InV := TNNetVolume.Create(NN.GetFirstLayer().Output);
  try
    if PerPosition and (not Overflows) then
    begin
      // ONE teacher-forced forward scores every position at once.
      InV.Fill(0);
      if InDepth = 1
      then InV.CopyNoChecksIntArr(Tokens)       // token ids -> embedding
      else InV.OneHotEncoding(Tokens);          // one-hot, left-aligned
      NN.Compute(InV);
      for Pos := FirstPos to SampleLenM1 do
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
        for D := 0 to VocabSizeM1 do
          RowSum := RowSum + Last.Output[Pos - 1, 0, D];
        if RowSum <= 0 then RowSum := 1.0;
        Prob := Last.Output[Pos - 1, 0, Tgt] / RowSum;
        Result[Pos] := SafeLogProb(Prob);
      end;
    end
    else if PerPosition then
    begin
      // Over-context per-position head + LastWindow: per scored position, run a
      // ContextLen window ENDING at Pos and read the row predicting Pos.
      for Pos := FirstPos to SampleLenM1 do
      begin
        Tgt := Tokens[Pos];
        if (Tgt < 0) or (Tgt >= VocabSize) then
        begin
          Result[Pos] := SafeLogProb(0);
          continue;
        end;
        WinStart := Pos - ContextLen + 1;
        if WinStart < 0 then WinStart := 0;
        WinLen := Pos - WinStart + 1;           // tokens [WinStart..Pos]
        Prefix := Copy(Tokens, WinStart, WinLen);
        InV.Fill(0);
        if InDepth = 1
        then InV.CopyNoChecksIntArr(Prefix)
        else InV.OneHotEncoding(Prefix);
        NN.Compute(InV);
        Row := WinLen - 2;                       // row predicting the last token
        RowSum := 0;
        for D := 0 to VocabSizeM1 do
          RowSum := RowSum + Last.Output[Row, 0, D];
        if RowSum <= 0 then RowSum := 1.0;
        Prob := Last.Output[Row, 0, Tgt] / RowSum;
        Result[Pos] := SafeLogProb(Prob);
      end;
    end
    else
    begin
      // Single next-token head: one forward per scored position over the
      // right-aligned prefix. LastWindow caps the prefix at ContextLen tokens.
      for Pos := FirstPos to SampleLenM1 do
      begin
        Tgt := Tokens[Pos];
        if (Tgt < 0) or (Tgt >= VocabSize) then
        begin
          Result[Pos] := SafeLogProb(0);
          continue;
        end;
        // Prefix = tokens [WinStart..Pos-1] (the target Tokens[Pos] excluded).
        WinStart := 0;
        if LastWindow and (Pos > ContextLen) then WinStart := Pos - ContextLen;
        Prefix := Copy(Tokens, WinStart, Pos - WinStart);
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

function ScoreSequence(NN: TNNet;
  const Tokens: TNeuralIntegerArray;
  LastWindow: boolean): TNeuralFloatDynArr;
begin
  Result := ScoreSequenceFrom(NN, Tokens, 1, LastWindow);
end;

// Shared core: scores CompletionTokens given ContextTokens. FirstScored=CtxLen
// asks ScoreSequenceFrom to forward only the completion positions (the unused
// context positions are skipped for single next-token heads). Identical scores
// to scoring the whole sequence and summing the completion indices.
function ScoreCompletionCore(NN: TNNet;
  const ContextTokens, CompletionTokens: TNeuralIntegerArray;
  LastWindow: boolean): TNNetCompletionScore;
var
  Full: TNeuralIntegerArray;
  LogProbs: TNeuralFloatDynArr;
  CtxLen, CompLen, I, CtxLenM1, CompLenM1, FullLastIdx: integer;
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
  CtxLenM1 := CtxLen - 1;
  CompLenM1 := CompLen - 1;
  for I := 0 to CtxLenM1 do Full[I] := ContextTokens[I];
  for I := 0 to CompLenM1 do Full[CtxLen + I] := CompletionTokens[I];
  // Only the completion positions (CtxLen ..) are summed below, so scoring can
  // start there - the shared context forwards are skipped for single-head nets.
  LogProbs := ScoreSequenceFrom(NN, Full, CtxLen, LastWindow);
  if Length(LogProbs) <> CtxLen + CompLen then Exit; // degenerate model
  // Completion tokens ONLY: indices CtxLen .. CtxLen+CompLen-1.
  FullLastIdx := CtxLen + CompLen - 1;
  for I := CtxLen to FullLastIdx do
    Result.SumLogProb := Result.SumLogProb + LogProbs[I];
  Result.TokenCount := CompLen;
  Result.MeanLogProb := Result.SumLogProb / CompLen;
  SetLength(Full, 0);
  SetLength(LogProbs, 0);
end;

function ScoreCompletion(NN: TNNet;
  const ContextTokens, CompletionTokens: TNeuralIntegerArray;
  LastWindow: boolean): TNNetCompletionScore;
begin
  Result := ScoreCompletionCore(NN, ContextTokens, CompletionTokens, LastWindow);
end;

function ScoreCompletionsBatch(NN: TNNet;
  const ContextTokens: TNeuralIntegerArray;
  const Candidates: array of TNeuralIntegerArray;
  LastWindow: boolean): TNNetCompletionScoreArray;
var
  Cand, CandHi: integer;
begin
  SetLength(Result, Length(Candidates));
  CandHi := High(Candidates);
  for Cand := 0 to CandHi do
    Result[Cand] := ScoreCompletionCore(NN, ContextTokens,
      Candidates[Cand], LastWindow);
end;

function EvaluateMultipleChoice(NN: TNNet;
  const Items: array of TNNetMultipleChoiceItem): TNNetMultipleChoiceStats;
var
  ItemIdx, Cand, BestSum, BestNorm, ItemsHi, CandHigh: integer;
  Scores: TNNetCompletionScoreArray;
  Score: TNNetCompletionScore;
  BestSumLP, BestNormLP: TNeuralFloat;
begin
  Result.Accuracy := 0;
  Result.AccuracyNorm := 0;
  Result.ItemCount := 0;
  Result.CorrectCount := 0;
  Result.CorrectNormCount := 0;
  if NN = nil then Exit;
  ItemsHi := High(Items);
  for ItemIdx := 0 to ItemsHi do
  begin
    if Length(Items[ItemIdx].Candidates) = 0 then continue;
    BestSum := 0;
    BestNorm := 0;
    BestSumLP := 0;
    BestNormLP := 0;
    // Shared-prefix batch: all candidates of an item share ContextTokens, so
    // the shared context is not re-forwarded per candidate (single-head nets).
    Scores := ScoreCompletionsBatch(NN, Items[ItemIdx].ContextTokens,
      Items[ItemIdx].Candidates);
    CandHigh := High(Items[ItemIdx].Candidates);
    for Cand := 0 to CandHigh do
    begin
      Score := Scores[Cand];
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
// MMLU few-shot accuracy harness
// ---------------------------------------------------------------------------

function EvaluateMMLU(NN: TNNet;
  const Questions: array of TNNetMMLUQuestion;
  const LetterTokens: array of integer;
  NumSubjects: integer;
  LastWindow: boolean): TNNetMMLUStats;
var
  QIdx, L, NumLetters, Best, Subj, NonEmpty: integer;
  NumSubjectsM1, QuestionsHi, NumLettersM1: integer;
  Letter: TNeuralIntegerArray;
  Score: TNNetCompletionScore;
  BestLP, MacroSum: TNeuralFloat;
begin
  Result.MacroAccuracy := 0;
  Result.MicroAccuracy := 0;
  Result.ItemCount := 0;
  Result.CorrectCount := 0;
  Result.SubjectCount := 0;
  if NumSubjects < 0 then NumSubjects := 0;
  SetLength(Result.PerSubject, NumSubjects);
  NumSubjectsM1 := NumSubjects - 1;
  for Subj := 0 to NumSubjectsM1 do
  begin
    Result.PerSubject[Subj].Correct := 0;
    Result.PerSubject[Subj].Total := 0;
    Result.PerSubject[Subj].Accuracy := 0;
  end;
  if NN = nil then Exit;
  NumLetters := Length(LetterTokens);
  if NumLetters = 0 then Exit;
  NumLettersM1 := NumLetters - 1;
  SetLength(Letter, 1); // each candidate is the SINGLE answer-letter token

  QuestionsHi := High(Questions);
  for QIdx := 0 to QuestionsHi do
  begin
    Subj := Questions[QIdx].SubjectIndex;
    if (Subj < 0) or (Subj >= NumSubjects) then continue;
    if Length(Questions[QIdx].PromptTokens) = 0 then continue;
    // Score every answer-letter token as a single-token completion; the
    // highest single-token log-prob letter is the prediction (first-max).
    Best := 0;
    BestLP := 0;
    for L := 0 to NumLettersM1 do
    begin
      Letter[0] := LetterTokens[L];
      Score := ScoreCompletionCore(NN, Questions[QIdx].PromptTokens,
        Letter, LastWindow);
      if (L = 0) or (Score.SumLogProb > BestLP) then
      begin
        BestLP := Score.SumLogProb;
        Best := L;
      end;
    end;
    Inc(Result.ItemCount);
    Inc(Result.PerSubject[Subj].Total);
    if Best = Questions[QIdx].GoldLetter then
    begin
      Inc(Result.CorrectCount);
      Inc(Result.PerSubject[Subj].Correct);
    end;
  end;

  // Per-subject accuracy + macro (mean over non-empty subjects) and micro.
  MacroSum := 0;
  NonEmpty := 0;
  for Subj := 0 to NumSubjectsM1 do
    if Result.PerSubject[Subj].Total > 0 then
    begin
      Result.PerSubject[Subj].Accuracy :=
        Result.PerSubject[Subj].Correct / Result.PerSubject[Subj].Total;
      MacroSum := MacroSum + Result.PerSubject[Subj].Accuracy;
      Inc(NonEmpty);
    end;
  Result.SubjectCount := NonEmpty;
  if NonEmpty > 0 then Result.MacroAccuracy := MacroSum / NonEmpty;
  if Result.ItemCount > 0 then
    Result.MicroAccuracy := Result.CorrectCount / Result.ItemCount;
end;

function MMLUReport(const Stats: TNNetMMLUStats;
  const SubjectNames: array of string; ShotsK: integer): string;
var
  Subj, PerSubjectHi: integer;
  Name: string;
  SL: TStringList;
begin
  SL := TStringList.Create();
  try
    SL.Add(Format('MMLU %d-shot accuracy (single-token answer-letter scoring)',
      [ShotsK]));
    SL.Add(Format('  questions scored : %d', [Stats.ItemCount]));
    SL.Add(Format('  subjects scored  : %d', [Stats.SubjectCount]));
    SL.Add('  per-subject:');
    PerSubjectHi := High(Stats.PerSubject);
    for Subj := 0 to PerSubjectHi do
    begin
      if Stats.PerSubject[Subj].Total = 0 then continue;
      if Subj <= High(SubjectNames) then Name := SubjectNames[Subj]
      else Name := Format('subject %d', [Subj]);
      SL.Add(Format('    %-28s %.4f  (%d / %d)',
        [Name, Stats.PerSubject[Subj].Accuracy,
         Stats.PerSubject[Subj].Correct, Stats.PerSubject[Subj].Total]));
    end;
    SL.Add(Format('  macro-average    : %.4f  (mean over %d subjects)',
      [Stats.MacroAccuracy, Stats.SubjectCount]));
    SL.Add(Format('  micro-average    : %.4f  (%d / %d pooled)',
      [Stats.MicroAccuracy, Stats.CorrectCount, Stats.ItemCount]));
    Result := SL.Text;
  finally
    SL.Free;
  end;
end;

// ---------------------------------------------------------------------------
// ARC / PIQA / WinoGrande (multiple-choice wrappers over EvaluateMultipleChoice)
// ---------------------------------------------------------------------------

function EvaluateARC(NN: TNNet;
  const Items: array of TNNetMultipleChoiceItem): TNNetMultipleChoiceStats;
begin
  // ARC: per-choice full-continuation log-likelihood; headline = acc_norm.
  Result := EvaluateMultipleChoice(NN, Items);
end;

function EvaluatePIQA(NN: TNNet;
  const Items: array of TNNetMultipleChoiceItem): TNNetMultipleChoiceStats;
begin
  // PIQA: 2 candidate solutions; headline = acc_norm (length-normalized).
  Result := EvaluateMultipleChoice(NN, Items);
end;

function EvaluateWinoGrande(NN: TNNet;
  const Items: array of TNNetMultipleChoiceItem): TNNetMultipleChoiceStats;
begin
  // WinoGrande: 2 substituted options scored over the suffix; headline = acc.
  Result := EvaluateMultipleChoice(NN, Items);
end;

function MultipleChoiceReport(const Stats: TNNetMultipleChoiceStats;
  const Title: string; HeadlineNorm: boolean): string;
var
  SL: TStringList;
  AccLine, NormLine: string;
begin
  SL := TStringList.Create();
  try
    SL.Add(Format('%s multiple-choice accuracy', [Title]));
    SL.Add(Format('  items scored : %d', [Stats.ItemCount]));
    AccLine := Format('  acc          : %.4f  (%d / %d)',
      [Stats.Accuracy, Stats.CorrectCount, Stats.ItemCount]);
    NormLine := Format('  acc_norm     : %.4f  (%d / %d)',
      [Stats.AccuracyNorm, Stats.CorrectNormCount, Stats.ItemCount]);
    if HeadlineNorm then NormLine := NormLine + '  <- headline'
    else AccLine := AccLine + '  <- headline';
    SL.Add(AccLine);
    SL.Add(NormLine);
    Result := SL.Text;
  finally
    SL.Free;
  end;
end;

// ---------------------------------------------------------------------------
// LAMBADA last-word accuracy
// ---------------------------------------------------------------------------

// Internal: model's argmax next-token prediction for position Pos of Tokens
// (conditioned on Tokens[0..Pos-1]). Mirrors the forward in ScoreSequenceFrom
// EXACTLY - including the reversed-prefix encoding for single next-token heads -
// so the greedy LAMBADA prediction matches the scorer's conditioning. Returns
// -1 on a degenerate model / out-of-range Pos.
function PredictArgmaxAt(NN: TNNet;
  const Tokens: TNeuralIntegerArray; Pos: integer;
  LastWindow: boolean): integer;
var
  InV: TNNetVolume;
  Last: TNNetLayer;
  Prefix: TNeuralIntegerArray;
  ContextLen, InDepth, VocabSize, WinStart, WinLen, Row, D, Best: integer;
  VocabSizeM1: integer;
  PerPosition, Overflows: boolean;
  BestVal, V: TNeuralFloat;
begin
  Result := -1;
  if NN = nil then Exit;
  if NN.CountLayers() < 2 then Exit;
  if (Pos < 1) or (Pos > High(Tokens)) then Exit;
  ContextLen := NN.GetFirstLayer().Output.SizeX;
  InDepth := NN.GetFirstLayer().Output.Depth;
  Last := NN.GetLastLayer();
  PerPosition := (ContextLen > 1) and (Last.Output.SizeX = ContextLen) and
    (Last.Output.Depth >= 2);
  if PerPosition
  then VocabSize := Last.Output.Depth
  else VocabSize := Last.Output.Size;
  if VocabSize < 2 then Exit;
  VocabSizeM1 := VocabSize - 1;
  InV := TNNetVolume.Create(NN.GetFirstLayer().Output);
  try
    if PerPosition then
    begin
      // Window of context tokens [WinStart..Pos-1]; the row predicting Pos is
      // the last row of that window (length-1). Over-context: trailing window.
      WinStart := 0;
      if Pos > ContextLen then
      begin
        if not LastWindow then
          raise EArgumentException.CreateFmt(
            'EvaluateLAMBADA: position %d exceeds the model context window %d',
            [Pos, ContextLen]);
        WinStart := Pos - ContextLen;
      end;
      WinLen := Pos - WinStart;                 // tokens [WinStart..Pos-1]
      Prefix := Copy(Tokens, WinStart, WinLen);
      InV.Fill(0);
      if InDepth = 1
      then InV.CopyNoChecksIntArr(Prefix)
      else InV.OneHotEncoding(Prefix);
      NN.Compute(InV);
      Row := WinLen - 1;                         // row predicting token at Pos
      Best := 0;
      BestVal := Last.Output[Row, 0, 0];
      for D := 1 to VocabSizeM1 do
      begin
        V := Last.Output[Row, 0, D];
        if V > BestVal then begin BestVal := V; Best := D; end;
      end;
      Result := Best;
    end
    else
    begin
      // Single next-token head: reversed right-aligned prefix [WinStart..Pos-1].
      WinStart := 0;
      if Pos > ContextLen then
      begin
        if not LastWindow then
          raise EArgumentException.CreateFmt(
            'EvaluateLAMBADA: position %d exceeds the model context window %d',
            [Pos, ContextLen]);
        WinStart := Pos - ContextLen;
      end;
      Prefix := Copy(Tokens, WinStart, Pos - WinStart);
      if InDepth = 1 then
      begin
        InV.Fill(0);
        InV.CopyReversedNoChecksIntArr(Prefix);
      end
      else InV.OneHotEncodingReversed(Prefix);
      NN.Compute(InV);
      Best := 0;
      BestVal := Last.Output.FData[0];
      for D := 1 to VocabSizeM1 do
      begin
        V := Last.Output.FData[D];
        if V > BestVal then begin BestVal := V; Best := D; end;
      end;
      Result := Best;
    end;
  finally
    InV.Free;
  end;
  SetLength(Prefix, 0);
end;

function EvaluateLAMBADA(NN: TNNet;
  const Examples: array of TNNetLambadaExample;
  LastWindow: boolean): TNNetLambadaStats;
var
  ExIdx, CtxLen, WordLen, I, Pos, Pred: integer;
  ExamplesHi, CtxLenM1, WordLenM1, FullLastIdx: integer;
  Full: TNeuralIntegerArray;
  LogProbs: TNeuralFloatDynArr;
  AllCorrect: boolean;
  SumNLL: TNeuralFloat;
begin
  Result.Accuracy := 0;
  Result.ItemCount := 0;
  Result.CorrectCount := 0;
  Result.MeanNLL := 0;
  Result.Perplexity := 0;
  Result.TokenCount := 0;
  if NN = nil then Exit;
  SumNLL := 0;
  ExamplesHi := High(Examples);
  for ExIdx := 0 to ExamplesHi do
  begin
    CtxLen := Length(Examples[ExIdx].ContextTokens);
    WordLen := Length(Examples[ExIdx].FinalWordTokens);
    if (CtxLen < 1) or (WordLen < 1) then continue;
    CtxLenM1 := CtxLen - 1;
    WordLenM1 := WordLen - 1;
    // Build the full passage (context + gold final word). Greedy prediction is
    // teacher-forced on the gold tokens: position CtxLen+i predicts the i-th
    // final-word token from Full[0..CtxLen+i-1].
    SetLength(Full, CtxLen + WordLen);
    for I := 0 to CtxLenM1 do Full[I] := Examples[ExIdx].ContextTokens[I];
    for I := 0 to WordLenM1 do
      Full[CtxLen + I] := Examples[ExIdx].FinalWordTokens[I];
    AllCorrect := true;
    for I := 0 to WordLenM1 do
    begin
      Pos := CtxLen + I;
      Pred := PredictArgmaxAt(NN, Full, Pos, LastWindow);
      if Pred <> Full[Pos] then
      begin
        AllCorrect := false;
        break; // word is already wrong; remaining argmaxes don't matter
      end;
    end;
    // Teacher-forced final-word NLL for the co-reported perplexity.
    LogProbs := ScoreSequenceFrom(NN, Full, CtxLen, LastWindow);
    FullLastIdx := CtxLen + WordLen - 1;
    if Length(LogProbs) = CtxLen + WordLen then
      for I := CtxLen to FullLastIdx do
      begin
        SumNLL := SumNLL - LogProbs[I];
        Inc(Result.TokenCount);
      end;
    Inc(Result.ItemCount);
    if AllCorrect then Inc(Result.CorrectCount);
  end;
  if Result.ItemCount > 0 then
    Result.Accuracy := Result.CorrectCount / Result.ItemCount;
  if Result.TokenCount > 0 then
  begin
    Result.MeanNLL := SumNLL / Result.TokenCount;
    Result.Perplexity := Exp(Result.MeanNLL);
  end;
  SetLength(Full, 0);
  SetLength(LogProbs, 0);
end;

function LambadaReport(const Stats: TNNetLambadaStats): string;
var
  SL: TStringList;
begin
  SL := TStringList.Create();
  try
    SL.Add('LAMBADA last-word accuracy (greedy reproduction)');
    SL.Add(Format('  examples     : %d', [Stats.ItemCount]));
    SL.Add(Format('  accuracy     : %.4f  (%d / %d)',
      [Stats.Accuracy, Stats.CorrectCount, Stats.ItemCount]));
    SL.Add(Format('  last-word PPL : %.4f  (mean NLL %.4f over %d tokens)',
      [Stats.Perplexity, Stats.MeanNLL, Stats.TokenCount]));
    Result := SL.Text;
  finally
    SL.Free;
  end;
end;

// ---------------------------------------------------------------------------
// n-gram machinery (shared by BLEU and ROUGE-N)
// ---------------------------------------------------------------------------

// Builds a sorted "ngram-key -> count" map for all N-grams of Tokens. Keys
// are the ids joined with commas; counts live in Objects[] as PtrInt.
function CountNGrams(const Tokens: TNeuralIntegerArray; N: integer): TStringList;
var
  Start, Idx, KeyPos, LastStart, NM1: integer;
  Key: string;
begin
  Result := TStringList.Create();
  Result.Sorted := true;
  Result.CaseSensitive := true;
  LastStart := Length(Tokens) - N;
  NM1 := N - 1;
  for Start := 0 to LastStart do
  begin
    Key := IntToStr(Tokens[Start]);
    for Idx := 1 to NM1 do
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
  Idx, RefPos, CandCount, CandCountM1: integer;
begin
  Result := 0;
  CandCount := CandCounts.Count;
  CandCountM1 := CandCount - 1;
  for Idx := 0 to CandCountM1 do
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
  CharIdx, WordPos, IdCount, TextLen: integer;
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
  TextLen := Length(Text);
  for CharIdx := 1 to TextLen do
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
  PairIdx, Order, UsedOrders, CandidatesHi: integer;
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
  CandidatesHi := High(Candidates);
  for PairIdx := 0 to CandidatesHi do
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
  PairIdx, CandidatesHi: integer;
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
    CandidatesHi := High(Candidates);
    for PairIdx := 0 to CandidatesHi do
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
  RowIdx, ColIdx, LenA, LenB: integer;
begin
  Result := 0;
  if (Length(A) = 0) or (Length(B) = 0) then Exit;
  LenA := Length(A);
  LenB := Length(B);
  SetLength(Prev, Length(B) + 1);
  SetLength(Curr, Length(B) + 1);
  for ColIdx := 0 to LenB do Prev[ColIdx] := 0;
  for RowIdx := 1 to LenA do
  begin
    Curr[0] := 0;
    for ColIdx := 1 to LenB do
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
  PairIdx, CandidatesHi: integer;
  PairScore: TNNetRougeScore;
begin
  Result := MakeRouge(0, 0, 0);
  if (Length(Candidates) = 0) or
     (Length(Candidates) <> Length(References)) then Exit;
  CandidatesHi := High(Candidates);
  for PairIdx := 0 to CandidatesHi do
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
  PairIdx, CandidatesHi: integer;
  PairScore: TNNetRougeScore;
begin
  Result := MakeRouge(0, 0, 0);
  if (Length(Candidates) = 0) or
     (Length(Candidates) <> Length(References)) then Exit;
  CandidatesHi := High(Candidates);
  for PairIdx := 0 to CandidatesHi do
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
  Start, KeyPos, LastStart: integer;
  Key: string;
begin
  Result := TStringList.Create();
  Result.Sorted := true;
  Result.CaseSensitive := true;
  LastStart := Length(S) - N + 1;
  for Start := 1 to LastStart do
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
  var I, SrcLen: integer;
  begin
    if IncludeWhitespace then Exit(Src);
    Result := '';
    SrcLen := Length(Src);
    for I := 1 to SrcLen do
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
  PairIdx, HypothesesHi: integer;
begin
  Result := 0;
  if (Length(Hypotheses) = 0) or
     (Length(Hypotheses) <> Length(References)) then Exit;
  HypothesesHi := High(Hypotheses);
  for PairIdx := 0 to HypothesesHi do
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
  I, J, NumOthers, GenerationsHi: integer;
  PerCand, OtherBleu: TNeuralFloat;
  Cand, Ref: array of TNeuralIntegerArray;
begin
  Result := 0;
  if Length(Generations) < 2 then Exit; // need at least one "other" reference
  SetLength(Cand, 1);
  SetLength(Ref, 1);
  GenerationsHi := High(Generations);
  for I := 0 to GenerationsHi do
  begin
    // Mean single-reference BLEU of generation I against every OTHER one
    // (v1: CorpusBLEU is single-reference, so average instead of multi-ref
    // clipping - documented in the unit header).
    PerCand := 0;
    NumOthers := 0;
    Cand[0] := Generations[I];
    for J := 0 to GenerationsHi do
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
  I, GenerationsHi: integer;
begin
  Result := 0;
  if Length(Generations) < 2 then Exit;
  Vocab := TStringList.Create();
  Vocab.Sorted := true;
  Vocab.CaseSensitive := true;
  try
    SetLength(Ids, Length(Generations));
    GenerationsHi := High(Generations);
    for I := 0 to GenerationsHi do
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
  I, Count, TagsHi: integer;
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
  TagsHi := High(Tags);
  for I := 0 to TagsHi do
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
  I, ArrHi: integer;
begin
  Result := -1;
  ArrHi := High(Arr);
  for I := 0 to ArrHi do
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
  I, MatchIdx, UsedHi, PredHi: integer;
begin
  Pred := DecodeBIOEntities(PredTags);
  Gold := DecodeBIOEntities(GoldTags);
  SetLength(Used, Length(Gold));
  UsedHi := High(Used);
  PredHi := High(Pred);
  for I := 0 to UsedHi do Used[I] := false;
  for I := 0 to PredHi do
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
  for I := 0 to UsedHi do
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
  I, PredHi: integer;
begin
  if Length(Pred) <> Length(Gold) then
    raise EArgumentException.Create(
      'CorpusEntityScore: Pred and Gold must have the same number of sentences.');
  Result.TruePos := 0;
  Result.FalsePos := 0;
  Result.FalseNeg := 0;
  PredHi := High(Pred);
  for I := 0 to PredHi do
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
  I, J, Tmp, N, NM1, TopKM1: integer;
begin
  N := Length(Logits);
  SetLength(Order, N);
  NM1 := N - 1;
  for I := 0 to NM1 do Order[I] := I;
  // simple selection sort by descending logit (N is small: TopK candidates)
  for I := 0 to NM1 do
    for J := I + 1 to NM1 do
      if Logits[Order[J]] > Logits[Order[I]] then
      begin
        Tmp := Order[I]; Order[I] := Order[J]; Order[J] := Tmp;
      end;
  if TopK > N then TopK := N;
  if TopK < 0 then TopK := 0;
  SetLength(Result, TopK);
  TopKM1 := TopK - 1;
  for I := 0 to TopKM1 do Result[I] := Order[I];
end;

function ExtractQASpans(const StartLogits, EndLogits: TNeuralFloatDynArr;
  TopK: integer = 20; MaxAnswerLen: integer = 30;
  NBest: integer = 20): TNNetQASpanArray;
var
  StartIdx, EndIdx: TNeuralIntegerArray;
  I, J, S, E, Count, A, B, Tmp: integer;
  StartIdxHi, EndIdxHi, CountM1, TmpM1: integer;
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
  StartIdxHi := High(StartIdx);
  EndIdxHi := High(EndIdx);
  for I := 0 to StartIdxHi do
    for J := 0 to EndIdxHi do
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
  CountM1 := Count - 1;
  for A := 0 to CountM1 do
    for B := A + 1 to CountM1 do
      if Cand[B].Score > Cand[A].Score then
      begin
        TmpSpan := Cand[A]; Cand[A] := Cand[B]; Cand[B] := TmpSpan;
      end;

  Tmp := Count;
  if (NBest > 0) and (NBest < Tmp) then Tmp := NBest;
  SetLength(Result, Tmp);
  TmpM1 := Tmp - 1;
  for A := 0 to TmpM1 do Result[A] := Cand[A];
end;

end.
