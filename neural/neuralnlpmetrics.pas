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

ROUGE. RougeN returns clipped n-gram overlap precision/recall/F1
(beta = 1); RougeL returns the LCS-based variant (P = LCS/|cand|,
R = LCS/|ref|, F1 = harmonic mean). CorpusRougeN / CorpusRougeL average the
per-pair scores (macro average over the corpus).

Both metrics expose a dual API: integer TOKEN-ID arrays
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

  // Precision / recall / F1 triple returned by the ROUGE functions.
  TNNetRougeScore = record
    Precision: TNeuralFloat;
    Recall: TNeuralFloat;
    F1: TNeuralFloat;
  end;

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

end.
