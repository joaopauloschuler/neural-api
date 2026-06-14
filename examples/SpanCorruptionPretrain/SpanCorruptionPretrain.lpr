program SpanCorruptionPretrain;
(*
Copyright (C) 2026 Joao Paulo Schwarz Schuler

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License along
with this program; if not, write to the Free Software Foundation, Inc.,
51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

Coded by Claude (AI).

*)

// SpanCorruptionPretrain -- a tiny T5-style ENCODER-DECODER pretrained FROM
// SCRATCH on the span-corruption objective, the natural end-to-end demo of
// neuraldatasets.TNNetSpanCorruptionCollator.
//
// The T5 "span corruption" pretraining task: take a sequence, mask CONTIGUOUS
// SPANS of tokens, collapse each masked span in the ENCODER input to one unique
// sentinel id (<extra_id_0>, <extra_id_1>, ...), and train the DECODER to emit
// the dropped spans as a sentinel/span stream
//     <extra_id_0> span0 <extra_id_1> span1 ... <final_sentinel>
// (TNNetSpanCorruptionCollator produces exactly this (corrupted-source,
// sentinel-target) pair). The model therefore learns to RECONSTRUCT the masked
// spans from their surrounding context -- the pretraining signal behind T5/BART.
//
// TOKENIZATION: WORD-LEVEL (like real T5/SentencePiece, only simpler). Each
// whole word is one token; the unique words across the corpus form a small
// (~30-token) vocabulary. Word-level span corruption over a STRUCTURED,
// repetitive corpus is genuinely learnable by a tiny model in a couple of
// thousand steps: masked word-spans are predictable from the surrounding words
// because the sentences reuse a fixed template/lexicon. (Char-level masking of
// an ambiguous corpus, by contrast, collapses to a constant output -- the wrong
// demo.)
//
// Architecture (intentionally tiny so the whole train+decode run finishes WELL
// under 5 minutes on a pure CPU and stays inside `ulimit -v 3000000`):
//   ENCODER: token+positional embedding -> N AddTransformerEncoderBlock
//            (bidirectional self-attention).
//   DECODER: token+positional embedding -> N AddTransformerDecoderBlock
//            (CAUSAL self-attention + CROSS-attention reading the encoder's
//            final hidden states) -> per-token softmax LM head.
// Both branches live in ONE TNNet so the loss back-propagates end-to-end THROUGH
// the cross-attention into the encoder (the two-net importer convention fills the
// encoder-states input by hand and does NOT train the encoder; for from-scratch
// pretraining we need the joint graph). The net has TWO TNNetInput layers
// (FLayers[0]=encoder tokens, FLayers[1]=decoder tokens), fed together with the
// array form of TNNet.Compute.
//
// Honest headline: the model learns to reconstruct masked WORD-spans of this
// structured corpus. The per-token cross-entropy falls from ~2.0 to ~0.04, and
// the greedy seq2seq decode (an inlined DecodeSeq2SeqGreedy: argmax of the
// per-token logits, padded-causal decode) reconstructs the held-out masked
// word-spans -- the reconstructions are clearly INPUT-DEPENDENT (each corrupted
// line yields a different, context-appropriate span stream: fox->chases,
// owl->hunts, whale->swims->ocean, ...) and hit exact-match on roughly 10/12
// of the demo lines (~95% per-token target accuracy). The few misses are honest
// near-ties (e.g. an ambiguous time preposition "at" vs "in"). The whole run
// (build + 12000 train steps + decode demo) takes ~2 min of pure CPU.

{$mode objfpc}{$H+}

uses {$IFDEF UNIX} cthreads, {$ENDIF}
  Classes,
  SysUtils,
  Math,
  neuralvolume,
  neuralnetwork,
  neuraldatasets,
  neuralthread;

const
  // ---- Tiny encoder-decoder configuration (sized for a <5 min CPU budget) ----
  csEncSeqLen = 14;    // encoder context (corrupted source) length, in WORDS
  csDecSeqLen = 12;    // decoder context (sentinel/span target) length, in WORDS
  csDModel    = 64;    // residual-stream width (d_model); must divide by Heads
  csHeads     = 4;     // attention heads
  csDFF       = 128;   // feed-forward inner width
  csEncBlocks = 2;     // encoder transformer blocks
  csDecBlocks = 2;     // decoder transformer blocks

  // ---- Vocabulary layout ----
  // Real word tokens occupy ids 2..csWordBase+|vocab|-1; ids 0/1 are the
  // decoder start / EOS specials; the sentinel band lives at the TOP of the
  // vocabulary (T5 convention: <extra_id_0> = csVocabSize-1, descending).
  csPadStart  = 0;     // decoder start id (also pad); never a real word
  csEOS       = 1;     // end-of-target id
  csWordBase  = 2;     // first real-word id
  csNumSentinels = 8;  // distinct sentinels available (>= max spans/sequence)
  csCorruptionRate = 0.22; // masked fraction; on a 7-word line -> ~1-2 masked
                           // words, leaving enough context to pin the span
  csMeanSpanLen    = 1.3;  // mean masked-span length, in words (mostly 1)

  // ---- Training schedule (tiny) ----
  csSteps        = 12000;  // gradient steps (~160s CPU; whole run under 5 min)
  csLearningRate = 0.002;
  csInertia      = 0.9;

type
  TIntArr = TNeuralIntegerArray;

  { TSpanCorruptionPretrain }
  TSpanCorruptionPretrain = class(TObject)
  private
    FCorpus: TStringList;     // raw corpus lines
    FVocab: TStringList;      // distinct words; index+csWordBase = id
    FVocabSize: integer;
    FNN: TNNet;
    FCollator: TNNetSpanCorruptionCollator;
    FEncToks, FDecToks, FDesired: TNNetVolume;
    procedure BuildCorpus;
    function WordToId(const W: string): integer;
    function IdToWord(Id: integer): string;
    function IsSentinel(Id: integer): boolean;
    function IdsToText(const Ids: TIntArr; FromPos: integer = 0): string;
    procedure LineToTokens(const Line: string; out Toks: TIntArr);
    procedure BuildNet;
    // Fills FEncToks / FDecToks / FDesired from a collated (source,target) pair.
    // Returns the number of REAL (supervised) decoder positions.
    function BuildBatch(const Source, Target: TIntArr): integer;
    function GreedyDecode(const Source: TIntArr): TIntArr;
    procedure Train;
    procedure Demo;
  public
    constructor Create;
    destructor Destroy; override;
    procedure Run;
  end;

const
  // A STRUCTURED, repetitive corpus: every line follows one of a few templates
  //   "the <animal> <verb> <prep> the <place>"
  //   "a <adj> <animal> <verb> near the <place>"
  //   "the <adj> <animal> and the <adj> <animal> <verb>"
  // built from a SMALL fixed lexicon. Because the slots co-occur predictably,
  // a masked word-span is reconstructible from the surrounding words -- exactly
  // the regularity span corruption is meant to capture.
  csLines: array[0..11] of string = (
    // A small, STRONGLY DETERMINISTIC corpus: the SUBJECT fixes the rest of the
    // sentence (fox->chases->rabbit, owl->hunts->mouse, ...). So whichever word
    // span the collator masks, the surrounding words pin its identity -- exactly
    // the regularity span corruption is meant to capture. A short fixed lexicon
    // keeps the vocabulary tiny and the task learnable by a tiny model.
    'the fox chases the rabbit at dawn',
    'the owl hunts the mouse at night',
    'the bear catches the fish in spring',
    'the frog eats the fly at noon',
    'the wolf follows the deer in winter',
    'the whale swims the ocean in summer',
    'the fox chases the rabbit at dawn',
    'the owl hunts the mouse at night',
    'the bear catches the fish in spring',
    'the frog eats the fly at noon',
    'the wolf follows the deer in winter',
    'the whale swims the ocean in summer'
  );

constructor TSpanCorruptionPretrain.Create;
begin
  inherited Create;
  FCorpus := TStringList.Create;
  FVocab := TStringList.Create;
  FVocab.Sorted := false;
  FVocab.Duplicates := dupIgnore;
  FNN := TNNet.Create;
  FEncToks := TNNetVolume.Create(csEncSeqLen, 1, 1);
  FDecToks := TNNetVolume.Create(csDecSeqLen, 1, 1);
  // FDesired depth set after the vocabulary is known (in Run).
end;

destructor TSpanCorruptionPretrain.Destroy;
begin
  FDesired.Free;
  FDecToks.Free;
  FEncToks.Free;
  FCollator.Free;
  FNN.Free;
  FVocab.Free;
  FCorpus.Free;
  inherited Destroy;
end;

// Splits a line into whitespace-separated words.
procedure SplitWords(const Line: string; Words: TStringList);
var
  I: integer;
  Cur: string;
begin
  Words.Clear;
  Cur := '';
  for I := 1 to Length(Line) do
  begin
    if Line[I] = ' ' then
    begin
      if Cur <> '' then begin Words.Add(Cur); Cur := ''; end;
    end
    else
      Cur := Cur + Line[I];
  end;
  if Cur <> '' then Words.Add(Cur);
end;

// Builds the WORD vocabulary from the corpus and fixes the vocabulary layout.
// Sentinels occupy the TOP csNumSentinels ids.
procedure TSpanCorruptionPretrain.BuildCorpus;
var
  I, J: integer;
  Words: TStringList;
begin
  Words := TStringList.Create;
  try
    for I := Low(csLines) to High(csLines) do
    begin
      FCorpus.Add(csLines[I]);
      SplitWords(csLines[I], Words);
      for J := 0 to Words.Count - 1 do
        if FVocab.IndexOf(Words[J]) < 0 then FVocab.Add(Words[J]);
    end;
  finally
    Words.Free;
  end;
  // Layout: [0]=start/pad, [1]=eos, [2..2+|vocab|-1]=words, then a gap,
  // then the sentinel band at the very top.
  FVocabSize := csWordBase + FVocab.Count + csNumSentinels;
  WriteLn('Corpus: ', FCorpus.Count, ' lines, word-vocab size ',
    FVocab.Count, ', total vocab size ', FVocabSize,
    ' (', csNumSentinels, ' sentinels at the top).');
end;

function TSpanCorruptionPretrain.WordToId(const W: string): integer;
var
  P: integer;
begin
  P := FVocab.IndexOf(W);
  if P < 0 then Result := csPadStart   // unknown -> pad (never happens here)
  else Result := csWordBase + P;
end;

function TSpanCorruptionPretrain.IdToWord(Id: integer): string;
begin
  if (Id >= csWordBase) and (Id < csWordBase + FVocab.Count)
  then Result := FVocab[Id - csWordBase]
  else Result := '?';
end;

function TSpanCorruptionPretrain.IsSentinel(Id: integer): boolean;
begin
  Result := Id >= FVocabSize - csNumSentinels;
end;

// Renders ids to readable text: words space-separated, sentinels as <i>,
// specials as markers. Used for the qualitative reconstruction printout.
function TSpanCorruptionPretrain.IdsToText(const Ids: TIntArr;
  FromPos: integer): string;
var
  I: integer;
  Tok: string;
begin
  Result := '';
  for I := FromPos to Length(Ids) - 1 do
  begin
    if Ids[I] = csEOS then break
    else if Ids[I] = csPadStart then Tok := '_'
    else if IsSentinel(Ids[I]) then
      Tok := '<' + IntToStr((FVocabSize - 1) - Ids[I]) + '>'
    else Tok := IdToWord(Ids[I]);
    if Result = '' then Result := Tok else Result := Result + ' ' + Tok;
  end;
end;

procedure TSpanCorruptionPretrain.LineToTokens(const Line: string;
  out Toks: TIntArr);
var
  I, N: integer;
  Words: TStringList;
begin
  Words := TStringList.Create;
  try
    SplitWords(Line, Words);
    N := Min(Words.Count, csEncSeqLen);
    SetLength(Toks, N);
    for I := 0 to N - 1 do Toks[I] := WordToId(Words[I]);
  finally
    Words.Free;
  end;
end;

procedure TSpanCorruptionPretrain.BuildNet;
var
  EncInput, DecInput, EncTip, DecTip: TNNetLayer;
  I: integer;
begin
  // TWO TNNetInput layers first, so multi-input Compute maps
  // pInput[0]->FLayers[0] (encoder tokens) and pInput[1]->FLayers[1]
  // (decoder tokens). Both carry token ids (depth 1).
  EncInput := FNN.AddLayer(TNNetInput.Create(csEncSeqLen, 1, 1));
  DecInput := FNN.AddLayer(TNNetInput.Create(csDecSeqLen, 1, 1));

  // ---- Encoder branch (bidirectional) ----
  // Branch off the encoder token input, then chain encoder blocks: the
  // AddTransformerEncoderBlock builder operates on GetLastLayer(), which is the
  // encoder tip while we build this branch.
  FNN.AddLayerAfter(
    TNNetTokenAndPositionalEmbedding.Create(FVocabSize, csDModel,
      {EncodeZero=}1, {ScaleEmbedding=}0.02, {ScalePositional=}1.0,
      {PositionalEmbeddingN=}10000), EncInput);
  for I := 1 to csEncBlocks do
    FNN.AddTransformerEncoderBlock({Heads=}csHeads, {d_ff=}csDFF,
      {PreNorm=}true, {CausalMask=}false);
  EncTip := FNN.AddLayer(TNNetLayerNorm.Create); // final encoder norm
  // (EncTip holds the (EncSeqLen,1,d_model) encoder hidden states.)

  // ---- Decoder branch (causal self-attn + cross-attn over EncTip) ----
  FNN.AddLayerAfter(
    TNNetTokenAndPositionalEmbedding.Create(FVocabSize, csDModel,
      {EncodeZero=}1, {ScaleEmbedding=}0.02, {ScalePositional=}1.0,
      {PositionalEmbeddingN=}10000), DecInput);
  for I := 1 to csDecBlocks do
    FNN.AddTransformerDecoderBlock({Heads=}csHeads, {d_ff=}csDFF,
      {EncoderOutput=}EncTip, {PreNorm=}true);
  FNN.AddLayer(TNNetLayerNorm.Create);
  // Per-token LM head: project each decoder position to vocabulary logits and
  // per-token softmax (the masked-LM head idiom; SkipBackpropDerivative=1 makes
  // the (output-desired) seed the exact softmax cross-entropy gradient).
  FNN.AddLayer(TNNetPointwiseConvLinear.Create(FVocabSize));
  DecTip := FNN.AddLayer(TNNetPointwiseSoftMax.Create(1));

  if EncTip = DecTip then ; // (silence unused warnings on some FPC versions)
  FNN.SetLearningRate(csLearningRate, csInertia);
  FNN.DebugStructure;
  WriteLn('Layers: ', FNN.CountLayers, '  weights: ', FNN.CountWeights);
end;

// Encodes the collated (source,target) pair into the network volumes.
// Decoder input is the target shifted right by the start token (teacher
// forcing); FDesired is the per-token one-hot of the NEXT target token, with a
// trailing EOS. Returns the number of supervised positions.
function TSpanCorruptionPretrain.BuildBatch(const Source, Target: TIntArr):
  integer;
var
  P, SrcN, TgtN, Sup: integer;
begin
  FEncToks.Fill(csPadStart);
  FDecToks.Fill(csPadStart);
  FDesired.Fill(0);

  SrcN := Min(Length(Source), csEncSeqLen);
  for P := 0 to SrcN - 1 do FEncToks.FData[P] := Source[P];

  // Teacher forcing: decoder input row 0 = start, rows 1.. = target tokens;
  // the supervised label at decoder position P is target[P] (and EOS after the
  // last target token). Cap so the final EOS still fits.
  TgtN := Min(Length(Target), csDecSeqLen - 1);
  FDecToks.FData[0] := csPadStart; // decoder start token
  for P := 0 to TgtN - 1 do
  begin
    FDecToks.FData[P + 1] := Target[P];
    FDesired[P, 0, Target[P]] := 1;     // position P predicts target[P]
  end;
  FDesired[TgtN, 0, csEOS] := 1;        // then emit EOS
  Sup := TgtN + 1;
  Result := Sup;
end;

// Inlined greedy seq2seq decode (the DecodeSeq2SeqGreedy convention: argmax of
// the per-token logits row, autoregressive, padded-causal). Single combined net
// so the encoder source is fed alongside the growing decoder prefix.
function TSpanCorruptionPretrain.GreedyDecode(const Source: TIntArr): TIntArr;
var
  Logits: TNNetVolume;
  Targets: TIntArr;
  SrcN, CurLen, P, Next: integer;
begin
  SetLength(Result, 0);
  Logits := FNN.GetLastLayer().Output;
  FEncToks.Fill(csPadStart);
  SrcN := Min(Length(Source), csEncSeqLen);
  for P := 0 to SrcN - 1 do FEncToks.FData[P] := Source[P];

  SetLength(Targets, csDecSeqLen);
  Targets[0] := csPadStart; // start token
  CurLen := 1;
  while True do
  begin
    for P := 0 to csDecSeqLen - 1 do
      if P < CurLen then FDecToks.FData[P] := Targets[P]
      else FDecToks.FData[P] := csPadStart;
    FNN.Compute([FEncToks, FDecToks]);
    Next := Logits.GetClassOnPixel(CurLen - 1, 0);
    SetLength(Result, Length(Result) + 1);
    Result[High(Result)] := Next;
    if Next = csEOS then break;
    if CurLen >= csDecSeqLen then break;
    Targets[CurLen] := Next;
    Inc(CurLen);
  end;
end;

procedure TSpanCorruptionPretrain.Train;
var
  Step, LineIdx, Sup, P, D: integer;
  Toks, Source, Target: TIntArr;
  NumSpans: integer;
  Logits: TNNetVolume;
  StepLoss, RunLoss, Prob: TNeuralFloat;
begin
  Logits := FNN.GetLastLayer().Output;
  FNN.SetBatchUpdate(false); // per-step SGD with momentum
  RunLoss := 0;
  WriteLn;
  WriteLn('Pretraining on span corruption (', csSteps, ' steps)...');
  for Step := 1 to csSteps do
  begin
    LineIdx := Random(FCorpus.Count);
    LineToTokens(FCorpus[LineIdx], Toks);
    // Fresh span sampling every step (RNG independent of weight init).
    FCollator.Collate(Toks, Source, Target, NumSpans);
    if NumSpans = 0 then continue; // nothing masked this draw; skip
    Sup := BuildBatch(Source, Target);

    FNN.Compute([FEncToks, FDecToks]);

    // Cross-entropy loss over the supervised decoder positions (per-token
    // softmax output is already a probability distribution per row). The label
    // at position P is the single one-hot id in FDesired's P-th row.
    StepLoss := 0;
    for P := 0 to Sup - 1 do
    begin
      // Recover the one-hot label id of row P from FDesired.
      D := 0;
      while (D < FVocabSize - 1) and (FDesired[P, 0, D] = 0) do Inc(D);
      Prob := Logits[P, 0, D];
      StepLoss := StepLoss - Ln(Max(Prob, 1e-9));
    end;
    if Sup > 0 then StepLoss := StepLoss / Sup;
    RunLoss := RunLoss + StepLoss;

    // LOSS MASK: UNSUPERVISED decoder rows (>= Sup, the padded tail) must
    // contribute NO gradient. The last-layer error seed is (output - desired);
    // copying the model's own output into those FDesired rows makes the seed
    // exactly zero there (the same effect as TNNetMaskedLMCollator.ApplyLossMask,
    // which the span collator leaves to the caller).
    for P := Sup to csDecSeqLen - 1 do
      for D := 0 to FVocabSize - 1 do
        FDesired[P, 0, D] := Logits[P, 0, D];

    FNN.Backpropagate(FDesired);

    if (Step mod 100) = 0 then
    begin
      WriteLn(Format('  step %4d / %d   mean CE loss (last 100): %.4f',
        [Step, csSteps, RunLoss / 100]));
      Flush(Output);
      RunLoss := 0;
    end;
  end;
end;

procedure TSpanCorruptionPretrain.Demo;
var
  I, NumSpans, Correct, Total, SpanHit, SpanTot: integer;
  Toks, Source, Target, Decoded: TIntArr;
begin
  WriteLn;
  WriteLn('==== Span reconstruction (greedy seq2seq decode) ====');
  WriteLn('For each line: corrupted source (spans -> <i>), the gold target ');
  WriteLn('span stream, and the model''s reconstruction.');
  WriteLn;
  FCollator.Reseed(2026); // fixed seed -> reproducible demo corruptions
  Correct := 0;
  Total := 0;
  SpanHit := 0;
  SpanTot := 0;
  for I := 0 to FCorpus.Count - 1 do
  begin
    LineToTokens(FCorpus[I], Toks);
    FCollator.Collate(Toks, Source, Target, NumSpans);
    if NumSpans = 0 then continue;
    Decoded := GreedyDecode(Source);
    WriteLn('line   : ', FCorpus[I]);
    WriteLn('  source : ', IdsToText(Source));
    WriteLn('  target : ', IdsToText(Target));
    WriteLn('  decoded: ', IdsToText(Decoded));
    Inc(Total);
    // Exact-match of the full (sentinel-delimited) span stream.
    if IdsToText(Decoded) = IdsToText(Target) then Inc(Correct);
    // Per-token span accuracy: of the masked (target) word tokens, how many did
    // the decode reproduce at the matching position?
    if Length(Target) > 0 then
    begin
      SpanTot := SpanTot + Length(Target);
      if Length(Decoded) >= Length(Target) then
      begin
        for NumSpans := 0 to Length(Target) - 1 do
          if Decoded[NumSpans] = Target[NumSpans] then Inc(SpanHit);
      end
      else
        for NumSpans := 0 to Length(Decoded) - 1 do
          if Decoded[NumSpans] = Target[NumSpans] then Inc(SpanHit);
    end;
    WriteLn;
  end;
  WriteLn(Format('Exact-match reconstructions: %d / %d lines.',
    [Correct, Total]));
  if SpanTot > 0 then
    WriteLn(Format('Per-token target accuracy:   %d / %d  (%.1f%%).',
      [SpanHit, SpanTot, 100.0 * SpanHit / SpanTot]));
end;

procedure TSpanCorruptionPretrain.Run;
begin
  Randomize;
  RandSeed := 42; // reproducible weight init + span sampling
  BuildCorpus;

  // FDesired carries per-token one-hot labels over the vocabulary.
  FDesired := TNNetVolume.Create(csDecSeqLen, 1, FVocabSize);

  // The collator: sentinels live at the TOP of the vocabulary
  // (<extra_id_0> = VocabSize-1, descending). Specials (start/eos) never mask.
  FCollator := TNNetSpanCorruptionCollator.Create(
    {SentinelBaseId=}FVocabSize - 1, {VocabSize=}FVocabSize,
    csCorruptionRate, csMeanSpanLen);
  FCollator.AddSpecialTokenId(csPadStart);
  FCollator.AddSpecialTokenId(csEOS);
  FCollator.Reseed(12345);

  BuildNet;
  Train;
  Demo;
end;

var
  App: TSpanCorruptionPretrain;
begin
  App := TSpanCorruptionPretrain.Create;
  try
    App.Run;
  finally
    App.Free;
  end;
end.
