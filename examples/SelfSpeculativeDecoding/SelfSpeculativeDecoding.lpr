program SelfSpeculativeDecoding;
(*
SelfSpeculativeDecoding: SELF-speculative greedy decoding from the extra heads
of ONE Multi-Token-Prediction model (Gloeckle et al. 2024 "Better & Faster
LLMs via Multi-token Prediction"; deployed at scale by DeepSeek-V3).

examples/SpeculativeDecoding needs TWO networks: a small DRAFT proposes a block
of tokens and the big TARGET verifies them in one pass. This example drops the
second net entirely: a model trained with TNNet.AddMultiTokenPrediction already
emits, at every position t, NumFuture parallel softmax heads forecasting the
tokens at t+1, t+2, ..., t+NumFuture. At inference, head 1..NumFuture-1 ARE the
draft -- the model speculates about its own future, for free, in the SAME
forward pass that commits the next token.

THE LOOP (one model forward per pass; commits 1..NumFuture tokens):
  Each pass forwards the committed prefix PLUS the pending draft tokens.
  Reading row r, head h gives the model's forecast for position r+1+h.
    1. VERIFY the pending drafts left-to-right: draft j (at position L+j-1,
       proposed by head j last pass) is ACCEPTED iff it equals head-0's greedy
       argmax at row L+j-2 -- which is EXACTLY the token plain greedy decoding
       would emit there, because every row left of the first mismatch sees only
       committed-or-accepted (i.e. greedy-identical) context.
    2. On the FIRST mismatch, head-0's argmax at that row is the CORRECT greedy
       token, so commit it (the standard "bonus" of speculative decoding: a
       rejection still yields one token) and discard the remaining drafts.
       If ALL drafts are accepted, head-0 at the last draft row yields a bonus
       token beyond the block.
    3. NEW drafts for the next pass come from heads 1..NumFuture-1 at the SAME
       row that produced the last committed token -- a row whose context is
       fully committed, so the proposals are well-defined.

WHY THIS IS EXACT (the headline gate, Halt(1) on failure). Every committed
token is head-0's argmax at a row whose causal context consists only of
already-committed tokens; the verification pass recomputes that argmax
bit-for-bit identically to what plain one-token-at-a-time greedy decoding
computes. Greedy speculative decoding therefore reproduces plain greedy
decoding EXACTLY -- we assert sequence equality on every benchmark run.

WHY IT IS FASTER. Plain greedy spends one full forward per token. The
self-speculative loop commits (1 + #accepted drafts) tokens per forward, so a
model that agrees with its own future heads (easy on an overfit tiny corpus)
commits up to NumFuture tokens per pass. The input window here is fixed-size,
so every forward costs the same -- the wall-clock ratio tracks the
forwards-per-token ratio directly.

THE TOY. A tiny char-level corpus (a few sentences with repeated clauses), a
causal attention trunk, and AddMultiTokenPrediction(NumFuture=4): head 0 is the
ordinary next-char head; heads 1..3 forecast 2..4 chars ahead and double as the
draft. Heavily overfit on purpose: the accept-rate signal (and the speedup) is
the point, not generalization.

Pure CPU, single-threaded manual Compute loop, deterministic under the fixed
RandSeed; runs in well under five minutes on two cores.

Copyright (C) 2026 Joao Paulo Schwarz Schuler

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

Coded by Claude (AI).
*)

{$mode objfpc}{$H+}

uses {$IFDEF UNIX} cthreads, BaseUnix, Unix, {$ENDIF}
  Classes, SysUtils, Math,
  neuralnetwork,
  neuralvolume;

const
  // Tiny char-level corpus. Repeated clause structure makes the future heads
  // easy to satisfy once overfit -> high accept rates and a clean speedup.
  cCorpus =
    'in the town of altdorf the clockmaker wound every clock at dawn. ' +
    'the clocks ticked, the bells rang, and the town woke. ' +
    'at noon the clockmaker oiled the gears of every clock in the town. ' +
    'at dusk the clockmaker wound every clock again. ' +
    'the clocks ticked, the bells rang, and the town slept.';

  cSeqLen     = 64;     // model context window
  cDModel     = 32;     // trunk width
  cHeads      = 2;      // causal attention heads
  cDFF        = 64;     // trunk feed-forward width
  cNumFuture  = 4;      // MTP heads: t+1 (committed) + t+2..t+4 (the draft)
  cTrainSteps = 9000;   // enough to overfit the tiny corpus
  cFineSteps  = 3000;   // extra steps at a low LR to sharpen the heads
  cLR         = 0.01;
  cFineLR     = 0.002;
  cInertia    = 0.9;
  cSeed       = 424242; // repo idiom

  cPrefixLen  = 8;      // chars of corpus used as the decode prompt
  cGenTokens  = 48;     // chars generated per benchmark run
  cRuns       = 12;     // benchmark prompts (different corpus offsets)
  cTimeReps   = 4;      // repeat the timed benchmark to smooth the clock

type
  TSeq = array[0..cSeqLen - 1] of integer;

var
  gVocab: integer;
  gCharToTok: array[char] of integer;
  gTokToChar: array[0..63] of char;
  gCorpusTok: array of integer;

// ---------------------------------------------------------------------------
// Microsecond-resolution wall clock in milliseconds since the first call.
// SysUtils.Now ticks at ~1 ms on Linux; rebasing to the first call keeps the
// value small (FPC types the 1000.0 literal as SINGLE -- at an absolute
// Unix-epoch scale single-precision quantization would freeze the clock).
// ---------------------------------------------------------------------------
{$IFDEF UNIX}
var
  GBaseSec: int64 = -1;

function NowMs(): double;
var
  tv: TTimeVal;
begin
  fpGetTimeOfDay(@tv, nil);
  if GBaseSec < 0 then GBaseSec := tv.tv_sec;
  Result := (tv.tv_sec - GBaseSec) * 1000.0 + tv.tv_usec / 1000.0;
end;
{$ELSE}
var
  GBaseMs: double = -1;

function NowMs(): double;
begin
  if GBaseMs < 0 then GBaseMs := Now() * 24 * 3600 * 1000;
  Result := Now() * 24 * 3600 * 1000 - GBaseMs;
end;
{$ENDIF}

// ---------------------------------------------------------------------------
// Char-level tokenization of the fixed corpus.
// ---------------------------------------------------------------------------
procedure BuildVocab();
var
  c: char;
  i: integer;
begin
  gVocab := 0;
  for c := Low(char) to High(char) do gCharToTok[c] := -1;
  for i := 1 to Length(cCorpus) do
  begin
    c := cCorpus[i];
    if gCharToTok[c] < 0 then
    begin
      gCharToTok[c] := gVocab;
      gTokToChar[gVocab] := c;
      Inc(gVocab);
    end;
  end;
  SetLength(gCorpusTok, Length(cCorpus));
  for i := 1 to Length(cCorpus) do
    gCorpusTok[i - 1] := gCharToTok[cCorpus[i]];
end;

function SeqToStr(const S: TSeq; FromPos, Len: integer): string;
var t: integer;
begin
  Result := '';
  for t := FromPos to FromPos + Len - 1 do
    Result := Result + gTokToChar[S[t]];
end;

// ---------------------------------------------------------------------------
// Model: causal next-char decoder with NumFuture parallel MTP heads.
// Output (cSeqLen,1,cNumFuture*gVocab); slab h at row t = p(token at t+1+h).
// ---------------------------------------------------------------------------
function BuildNet(): TNNet;
begin
  Result := TNNet.Create();
  Result.AddLayer(TNNetInput.Create(cSeqLen, 1, 1));
  Result.AddLayer(TNNetEmbedding.Create(gVocab, cDModel, 1));
  Result.AddLayer(TNNetSinusoidalPositionalEmbedding.Create());
  Result.AddLayer(TNNetPointwiseConvLinear.Create(3 * cDModel)); // Q|K|V slab
  Result.AddMultiHeadSelfAttention(cHeads, {CausalMask=}True);
  Result.AddLayer(TNNetPointwiseConvLinear.Create(cDFF));
  Result.AddLayer(TNNetReLU.Create());
  Result.AddLayer(TNNetPointwiseConvLinear.Create(cDModel));
  Result.AddMultiTokenPrediction(cNumFuture, gVocab);
  Result.SetLearningRate(cLR, cInertia);
  Result.SetL2Decay(0.0);
end;

// Train on random corpus windows. Target slab h at row t supervises the char
// at corpus position (start+t+1+h) -- reading past the window end is fine, the
// corpus continues; positions past the corpus end carry no supervision.
procedure Train(NN: TNNet; Steps: integer);
var
  InputV, TargetV: TNNetVolume;
  i, t, h, start, fut: integer;
begin
  InputV  := TNNetVolume.Create(cSeqLen, 1, 1);
  TargetV := TNNetVolume.Create(cSeqLen, 1, cNumFuture * gVocab);
  try
    for i := 1 to Steps do
    begin
      start := Random(Length(gCorpusTok) - cSeqLen);
      TargetV.Fill(0);
      for t := 0 to cSeqLen - 1 do
      begin
        InputV.FData[t] := gCorpusTok[start + t];
        for h := 0 to cNumFuture - 1 do
        begin
          fut := start + t + 1 + h;
          if fut < Length(gCorpusTok) then
            TargetV[t, 0, h * gVocab + gCorpusTok[fut]] := 1.0;
        end;
      end;
      NN.Compute(InputV);
      NN.Backpropagate(TargetV);
    end;
  finally
    InputV.Free; TargetV.Free;
  end;
end;

// ---------------------------------------------------------------------------
// Inference helpers. Inputs are zero-padded to cSeqLen; the causal mask keeps
// rows < Len independent of the padding.
// ---------------------------------------------------------------------------
procedure Forward(NN: TNNet; const S: TSeq; Len: integer; InputV: TNNetVolume);
var t: integer;
begin
  InputV.Fill(0);
  for t := 0 to Len - 1 do InputV.FData[t] := S[t];
  NN.Compute(InputV);
end;

// Greedy argmax of head h's softmax slab at row Row: the model's forecast for
// the token at position Row+1+h.
function HeadArgmax(NN: TNNet; Row, Head: integer): integer;
var
  v: integer;
  p, bestP: TNeuralFloat;
begin
  Result := 0;
  bestP := -1;
  for v := 0 to gVocab - 1 do
  begin
    p := NN.GetLastLayer.Output[Row, 0, Head * gVocab + v];
    if p > bestP then begin bestP := p; Result := v; end;
  end;
end;

// ---------------------------------------------------------------------------
// PLAIN greedy decoding: one full forward per token, head 0 only.
// ---------------------------------------------------------------------------
procedure PlainGreedy(NN: TNNet; var S: TSeq; PrefixLen, NTokens: integer;
  InputV: TNNetVolume; out Forwards: integer);
var
  len, i: integer;
begin
  len := PrefixLen;
  Forwards := 0;
  for i := 1 to NTokens do
  begin
    if len >= cSeqLen then Break;
    Forward(NN, S, len, InputV);
    Inc(Forwards);
    S[len] := HeadArgmax(NN, len - 1, 0);
    Inc(len);
  end;
end;

// ---------------------------------------------------------------------------
// SELF-SPECULATIVE greedy decoding. One forward per pass; the SAME pass
// verifies the previous block's drafts (heads 1..N-1) and drafts the next
// block. Exact-greedy by construction (see header comment).
//
// Per-head bookkeeping: draft slot j (0-based; position len+j) was proposed by
// head j+1; Verified[j]/Accepted[j] count how often that head-distance got
// checked / matched. Drafts past the first mismatch are discarded unverified.
// ---------------------------------------------------------------------------
procedure SelfSpeculative(NN: TNNet; var S: TSeq; PrefixLen, NTokens: integer;
  InputV: TNNetVolume; out Passes, Committed: integer;
  var Verified, Accepted: array of integer; out Discarded: integer);
var
  len, m, j, h, lastRow, newLen, pos, g: integer;
  mismatch: boolean;
begin
  len := PrefixLen;     // committed length
  m := 0;               // pending drafts at positions len..len+m-1
  Passes := 0; Committed := 0; Discarded := 0;

  while (len - PrefixLen < NTokens) and (len < cSeqLen) do
  begin
    Forward(NN, S, len + m, InputV);  // committed prefix + pending drafts
    Inc(Passes);

    // --- verify pending drafts left-to-right --------------------------------
    j := 0;             // drafts accepted so far this pass
    mismatch := false;
    while (j < m) and (not mismatch) do
    begin
      // Head-0 argmax at row len+j-1 is EXACTLY plain greedy's token for
      // position len+j (its causal context is committed/accepted only).
      g := HeadArgmax(NN, len + j - 1, 0);
      Inc(Verified[j]);
      if g = S[len + j] then
      begin
        Inc(Accepted[j]);
        Inc(j);
      end
      else
      begin
        S[len + j] := g;            // corrected token: still exact greedy
        Discarded := Discarded + (m - j - 1);
        mismatch := true;
      end;
    end;

    if mismatch then
    begin
      lastRow := len + j - 1;       // row that produced the corrected token
      newLen := len + j + 1;        // j accepted drafts + 1 corrected token
    end
    else
    begin
      // All m drafts accepted; head 0 at the last draft row yields a BONUS
      // token (position len+m). With m=0 this is the plain first-pass commit.
      lastRow := len + m - 1;
      if len + m < cSeqLen then
      begin
        S[len + m] := HeadArgmax(NN, lastRow, 0);
        newLen := len + m + 1;
      end
      else
        newLen := len + m;
    end;
    Committed := Committed + (newLen - len);

    // --- draft the next block from heads 1..N-1 at the last committed row ---
    m := 0;
    for h := 1 to cNumFuture - 1 do
    begin
      pos := lastRow + 1 + h;       // head h at lastRow forecasts position
      if pos >= cSeqLen then Break; // = newLen-1+h, i.e. newLen..newLen+N-2
      S[pos] := HeadArgmax(NN, lastRow, h);
      Inc(m);
    end;
    len := newLen;
  end;
end;

function SeqEqual(const A, B: TSeq; Len: integer): boolean;
var t: integer;
begin
  Result := true;
  for t := 0 to Len - 1 do
    if A[t] <> B[t] then Exit(false);
end;

var
  NN: TNNet;
  InputV: TNNetVolume;
  PlainSeq, SpecSeq: TSeq;
  Verified, Accepted: array[0..cNumFuture - 2] of integer;
  ScratchV, ScratchA: array[0..cNumFuture - 2] of integer;
  run, rep, h: integer;
  plainFwd, passes, committed, discarded: integer;
  totPlainFwd, totPasses, totCommitted, totDiscarded, totTokens: integer;
  tA, plainMs, specMs: double;
  allExact: boolean;

// Prompt for benchmark run `Run`: cPrefixLen corpus chars at a per-run offset.
procedure LoadPrompt(Run: integer; var S: TSeq);
var t, start: integer;
begin
  start := (Run * 17) mod (Length(gCorpusTok) - cPrefixLen - 1);
  for t := 0 to cPrefixLen - 1 do S[t] := gCorpusTok[start + t];
  for t := cPrefixLen to cSeqLen - 1 do S[t] := 0;
end;

begin
  // Mask FPU exceptions so transient early-training underflows don't abort
  // (matches sibling examples).
  SetExceptionMask([exInvalidOp, exDenormalized, exZeroDivide,
                    exOverflow, exUnderflow, exPrecision]);
  RandSeed := cSeed;
  BuildVocab();

  WriteLn('SelfSpeculativeDecoding: the extra heads of ONE Multi-Token-Prediction');
  WriteLn('model act as their own draft -- no second draft network.');
  WriteLn(Format('corpus=%d chars  vocab=%d  ctx=%d  d_model=%d  NumFuture=%d ' +
    '(draft block = %d)', [Length(cCorpus), gVocab, cSeqLen, cDModel,
    cNumFuture, cNumFuture - 1]));
  WriteLn('Headline: greedy self-speculative output is IDENTICAL to plain greedy');
  WriteLn('decoding (asserted per run) while running far fewer forward passes.');
  WriteLn;

  InputV := TNNetVolume.Create(cSeqLen, 1, 1);
  Write('Training (', cTrainSteps, ' steps @ ', cLR: 5: 3, ' + ', cFineSteps,
    ' @ ', cFineLR: 5: 3, ', overfit on purpose) ...');
  NN := BuildNet();
  Train(NN, cTrainSteps);
  NN.SetLearningRate(cFineLR, cInertia);
  Train(NN, cFineSteps);
  WriteLn(' done.  params=', NN.CountWeights());
  WriteLn;

  for h := 0 to cNumFuture - 2 do begin Verified[h] := 0; Accepted[h] := 0; end;
  totPlainFwd := 0; totPasses := 0; totCommitted := 0; totDiscarded := 0;
  totTokens := 0;
  allExact := true;

  // ===== Phase 1: EXACTNESS + accept-rate statistics (counted once) =========
  WriteLn(StringOfChar('=', 72));
  WriteLn('GATE  Exactness: self-speculative greedy == plain greedy, per run');
  WriteLn(StringOfChar('=', 72));
  for run := 0 to cRuns - 1 do
  begin
    LoadPrompt(run, PlainSeq);
    LoadPrompt(run, SpecSeq);
    PlainGreedy(NN, PlainSeq, cPrefixLen, cGenTokens, InputV, plainFwd);
    SelfSpeculative(NN, SpecSeq, cPrefixLen, cGenTokens, InputV,
      passes, committed, Verified, Accepted, discarded);
    totPlainFwd  := totPlainFwd + plainFwd;
    totPasses    := totPasses + passes;
    totCommitted := totCommitted + committed;
    totDiscarded := totDiscarded + discarded;
    totTokens    := totTokens + cGenTokens;
    if SeqEqual(PlainSeq, SpecSeq, cPrefixLen + cGenTokens) then
      WriteLn(Format('  run %2d : EXACT   forwards plain=%2d  spec=%2d',
        [run, plainFwd, passes]))
    else
    begin
      allExact := false;
      WriteLn('  run ', run, ' : EXACTNESS FAILURE!');
      WriteLn('    plain: "', SeqToStr(PlainSeq, 0, cPrefixLen + cGenTokens), '"');
      WriteLn('    spec : "', SeqToStr(SpecSeq, 0, cPrefixLen + cGenTokens), '"');
    end;
  end;
  WriteLn;

  WriteLn('Sample (prompt | continuation), run 0:');
  LoadPrompt(0, SpecSeq);
  SelfSpeculative(NN, SpecSeq, cPrefixLen, cGenTokens, InputV,
    passes, committed, ScratchV, ScratchA, discarded);
  WriteLn('  "', SeqToStr(SpecSeq, 0, cPrefixLen), '" | "',
    SeqToStr(SpecSeq, cPrefixLen, cGenTokens), '"');
  WriteLn;

  // ===== Phase 2: accept rates and forward-pass accounting ==================
  WriteLn(StringOfChar('=', 72));
  WriteLn('ACCEPT RATES per head distance (head h forecasts t+1+h; h>=1 drafts)');
  WriteLn(StringOfChar('=', 72));
  WriteLn('  head  dist   verified   accepted   accept-rate');
  for h := 1 to cNumFuture - 1 do
    WriteLn(Format('   %2d    +%d    %7d    %7d      %6.1f%%',
      [h, h + 1, Verified[h - 1], Accepted[h - 1],
       100.0 * Accepted[h - 1] / Max(1, Verified[h - 1])]));
  WriteLn(Format('  drafts discarded unverified (past first mismatch): %d',
    [totDiscarded]));
  WriteLn;
  WriteLn(Format('  tokens generated (truncated)      : %d', [totTokens]));
  WriteLn(Format('  tokens committed (incl. overshoot): %d', [totCommitted]));
  WriteLn(Format('  plain forward passes              : %d', [totPlainFwd]));
  WriteLn(Format('  speculative forward passes        : %d', [totPasses]));
  WriteLn(Format('  mean tokens committed per pass    : %.2f',
    [totCommitted / Max(1, totPasses)]));
  WriteLn(Format('  forward passes saved              : %.1f%%',
    [100.0 * (1.0 - totPasses / Max(1.0, totPlainFwd * 1.0))]));
  WriteLn;

  // ===== Phase 3: wall clock (cTimeReps repetitions of all runs) ============
  plainMs := 0; specMs := 0;
  for rep := 1 to cTimeReps do
    for run := 0 to cRuns - 1 do
    begin
      LoadPrompt(run, PlainSeq);
      tA := NowMs();
      PlainGreedy(NN, PlainSeq, cPrefixLen, cGenTokens, InputV, plainFwd);
      plainMs := plainMs + (NowMs() - tA);

      LoadPrompt(run, SpecSeq);
      tA := NowMs();
      SelfSpeculative(NN, SpecSeq, cPrefixLen, cGenTokens, InputV,
        passes, committed, ScratchV, ScratchA, discarded);
      specMs := specMs + (NowMs() - tA);
    end;
  WriteLn(StringOfChar('=', 72));
  WriteLn('WALL CLOCK (', cTimeReps, ' reps x ', cRuns, ' runs x ', cGenTokens,
    ' tokens; fixed-size window, so');
  WriteLn('every forward costs the same -- speedup tracks forwards-per-token)');
  WriteLn(StringOfChar('=', 72));
  WriteLn(Format('  plain greedy       : %8.1f ms  (%.3f ms/token)',
    [plainMs, plainMs / (cTimeReps * cRuns * cGenTokens)]));
  WriteLn(Format('  self-speculative   : %8.1f ms  (%.3f ms/token)',
    [specMs, specMs / (cTimeReps * cRuns * cGenTokens)]));
  WriteLn(Format('  wall-clock speedup : %.2fx   (forward-pass ratio %.2fx)',
    [plainMs / Max(0.001, specMs),
     totPlainFwd / Max(1.0, totPasses * 1.0)]));
  WriteLn;

  if allExact then
    WriteLn('GATE : PASS  (all ', cRuns,
      ' speculative continuations identical to plain greedy)')
  else
    WriteLn('GATE : FAIL  (speculative output diverged from plain greedy!)');

  NN.Free;
  InputV.Free;
  if not allExact then Halt(1);
end.
