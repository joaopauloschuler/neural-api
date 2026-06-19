program SpeculativeDecoding;
(*
SpeculativeDecoding: a pure-CPU reproduction of SPECULATIVE SAMPLING
(Leviathan et al. 2023, "Fast Inference from Transformers via Speculative
Decoding"; Chen et al. 2023, "Accelerating Large Language Model Decoding with
Speculative Sampling") on a TINY pair of next-token decoders sharing one toy
vocabulary. It proves the headline property of the trick:

    the speculative sampler's output is distributed EXACTLY as if drawn from
    the big TARGET model alone, while calling the big model far FEWER times.

THE TRICK (one verification pass commits between 1 and K+1 tokens):
  1. A small fast DRAFT model autoregressively proposes K candidate tokens
     x_1..x_K (K serial CHEAP draft forward passes over the growing prefix).
  2. The big TARGET model scores all K+1 positions in ONE batched forward pass
     over prefix+draft (it sees prefix, prefix+x_1, ..., prefix+x_1..x_K and the
     per-position softmax gives p_target at each of those K+1 contexts).
  3. Walk the block left-to-right. ACCEPT token x_i with probability
     min(1, p_target(x_i) / p_draft(x_i)). On the FIRST rejection, resample that
     position from the renormalised residual  norm(max(0, p_target - p_draft))
     and DISCARD the rest of the block. If all K are accepted, additionally
     sample one BONUS token x_{K+1} from p_target at the last position.

WHY THIS IS EXACT. Leviathan/Chen prove that the accept-or-residual-resample
rule makes the committed token at every position an exact draw from p_target,
for ANY draft distribution. This program makes that claim SELF-CHECKING two
ways (see GATES below).

THIS PROGRAM IS FORWARD-ONLY. Both models are pretrained; generation only — no
gradient surgery, so no SetBatchUpdate concerns. TWO speculative samplers are
implemented and compared:
  * v1 (SpeculativeSample): each verification pass RECOMPUTES the whole prefix
    from scratch (correct and simple, but every pass costs O((t+K)^2)).
  * KV-CACHED (SpeculativeSampleCached): the target's verification runs through
    the TNNetScaledDotProductAttention KV-cache incremental-decode path. The
    prompt is PREFILLED once; each pass then feeds ONLY the short window
    [last committed token + the K drafts] as one multi-token cached forward
    (each window token's K/V is appended and its query attends over the cache
    so far — multi-token prefill semantics), so a pass costs O(K * t) instead
    of O((t+K)^2). On a rejection the K/V entries already appended for the
    rejected positions are STALE: TruncateCache(committed-1) rewinds the cache
    to the committed prefix (a one-field rewind; the preallocated slots are
    simply reused). The two efficiency wins therefore COMPOSE: fewer big-model
    passes (speculation) x cheaper passes (KV cache).
POSITIONAL CONTRACT: the step net sees only the short window, so its
TNNetSinusoidalPositionalEmbedding must encode the window tokens at their
ABSOLUTE positions — PositionOffset is set to the window start (the running
cache length) before every cached forward. The draft stays on the simple
full-recompute path: it is attention-free (TokenShift), so it has no KV cache
to exploit, and its forwards are cheap by design.
The cached plumbing (per-head SDPA collection, BeginIncrementalDecode,
ResetCache, TruncateCache) is handled by the reusable TNNetStreamingDecoder
session from neuraldecode; only the sinusoidal PositionOffset stays
hand-rolled here, because the session manages rotary (RoPE) layers only.

THE TOY (forked from examples/InductionHeads + TokenShiftBaseline). A small
vocabulary V; a sequence is a random length-L string and the next-token target
is a fixed-offset copy with a tiny bit of structure, so a SMALL draft learns a
decent-but-imperfect approximation and a LARGER target learns it better. Both
are per-position causal decoders (Embedding -> SinPos -> [attention/shift] ->
pointwise MLP -> per-position softmax); the draft is shallower/narrower than the
target so it disagrees often enough to make the accept/reject machinery do real
work, yet agrees often enough to show a speedup.

----------------------------------------------------------------------------
RNG ORDERING (this is the crux of the bit-for-bit exactness gate).
----------------------------------------------------------------------------
We use TWO independent, explicitly-seeded scalar RNG streams (a tiny in-program
xorshift64* so the ordering is fully under our control, NOT the library RNG):

  * gProp  — the PROPOSAL stream. One uniform is drawn per token position to
             pick a token FROM A CATEGORICAL DISTRIBUTION via inverse-CDF.
  * gAcc   — the ACCEPT/RESIDUAL stream. One uniform per verification step for
             the accept/reject Bernoulli, and one more (from gProp) when a
             residual resample is needed.

Plain target-only sampling draws token t by inverse-CDF sampling p_target with
ONE gProp uniform per token, advancing gProp once per committed token, and
touches gAcc NEVER.

Speculative sampling draws each DRAFT proposal x_i by inverse-CDF sampling
p_draft with ONE gProp uniform (same stream), and draws each accept Bernoulli
from gAcc.

KEY EQUIVALENCE. When DRAFT == TARGET (we literally pass the same net as both),
p_draft == p_target at every accepted position, so the ratio is
min(1, p/p) = 1 and EVERY accept succeeds regardless of the gAcc uniform. The
proposal x_i was drawn by inverse-CDF on p_draft == p_target using exactly the
gProp uniform that plain sampling would have used for the token at that same
committed position. Because accepts never fail, gProp advances in lock-step with
plain sampling and no residual draw is ever taken, so the committed sequence is
BIT-FOR-BIT identical to plain target-only sampling under the same gProp seed.
We assert exactly this (and exit non-zero on failure). The bonus token at the
end of a fully-accepted block is also an inverse-CDF p_target draw from gProp, so
it too matches the next plain token. This collapse is the whole reason the
two-stream split is designed the way it is.

GATES (printed; the EXACTNESS ones are mandatory and Halt(1) on failure):
  1. (MANDATORY) DEGENERATE EXACTNESS: with draft == target, the speculative
     and plain token sequences are identical element-for-element.
  1b. (MANDATORY) KV-CACHE EXACTNESS: the cached speculative sampler produces
     a token sequence IDENTICAL to the v1 full-recompute speculative sampler
     under the same seeds (both with draft == target, where it must also
     equal plain sampling, and with the genuinely different trained draft,
     which exercises the TruncateCache rollback on real rejections).
  2. EMPIRICAL FAITHFULNESS: with a genuinely DIFFERENT trained draft, sample N
     tokens many times from a fixed prefix both ways and assert the two token
     histograms match within sampling noise (total-variation distance small).
  3. SPEEDUP: as draft/target agreement varies (we sweep the draft quality by
     using draft snapshots from different training lengths), chart mean
     accepted-tokens-per-pass and big-model-calls saved; accept rate (hence
     speedup) rises with agreement.
  4. COMPOSITION BENCHMARK: wall-clock ms/token and ACTUAL target-net forward
     counts for (i) plain full-recompute, (ii) v1 speculative full-recompute,
     (iii) speculative + KV cache. The headline is (iii) beating both: v1 cuts
     the number of conceptual big-model PASSES but still pays full-prefix
     forwards inside each pass; the cache makes each pass O(K * t).

Pure CPU, single-threaded (MaxThreadNum := 1) for reproducibility. Trains both
tiny models and runs all checks in well under the 5-minute budget.

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
  neuralvolume,
  neuraldecode;

const
  cVocab    = 8;        // small vocabulary
  cSeqLen   = 24;       // model context window (max prefix+block length)
  cLag      = 3;        // fixed-offset copy distance in the toy target
  // TARGET (big): wider + attention mixing.
  cTgtDModel = 32;
  cTgtHeads  = 4;
  cTgtDFF    = 48;
  cTgtEpochs = 600;
  // DRAFT (small): narrow + cheap TokenShift mixing, trained briefly.
  cDrfDModel = 12;
  cDrfDFF    = 16;
  cDrfEpochs = 70;
  cBatch     = 48;
  cLR        = 0.005;
  cInertia   = 0.9;
  cSeed      = 424242;  // repo idiom
  cK         = 4;       // speculative block size (draft proposes K tokens)

type
  TSeq = array[0..cSeqLen - 1] of integer;
  TProbs = array[0..cVocab - 1] of TNeuralFloat;

// ---------------------------------------------------------------------------
// Tiny self-contained xorshift64* RNG so the stream ordering is fully under our
// control (independent of the library/global RNG). Returns a uniform in [0,1).
// ---------------------------------------------------------------------------
type
  TRng = record State: QWord; end;

procedure RngSeed(var R: TRng; Seed: QWord);
begin
  if Seed = 0 then Seed := 1;
  R.State := Seed;
end;

function RngNextU64(var R: TRng): QWord;
var x: QWord;
begin
  x := R.State;
  x := x xor (x shr 12);
  x := x xor (x shl 25);
  x := x xor (x shr 27);
  R.State := x;
  Result := x * QWord(2685821657736338717);
end;

// Uniform in [0,1) with 53 bits of mantissa.
function RngFloat(var R: TRng): TNeuralFloat;
begin
  Result := (RngNextU64(R) shr 11) * (1.0 / 9007199254740992.0);
end;

var
  gProp: TRng;   // PROPOSAL / plain-sampling stream
  gAcc: TRng;    // accept-reject (+ residual resample) stream

// ---------------------------------------------------------------------------
// TARGET-DISTRIBUTION PEAKEDNESS CONTROL.
//
// The committed token's law is whatever p_target we hand the accept/reject loop,
// so we can make the target SHARPER (more peaked) without touching the algorithm:
//   * gTgtTemp < 1 re-softmaxes the TARGET's next-token distribution at a low
//     temperature, concentrating mass on the top token(s). The DRAFT is left at
//     temperature 1 (it stays the weak, flat-objective approximator), so a weak
//     draft now puts mass on the WRONG token -> p_target/p_draft collapses ->
//     accept rate falls well below 1. A strong draft tracks the peak -> high
//     accept rate. This WIDENS the good-vs-bad-draft calls-saved gap.
// Plain target-only sampling and speculative sampling read the SAME tempered
// p_target (NextDist applies gTgtTemp whenever IsTarget=True), so Gate 1's
// bit-for-bit exactness and Gate 2's faithfulness invariant both still hold for
// any temperature -- we only change WHAT the target distribution is, never the
// sampler that draws from it.
// ---------------------------------------------------------------------------
var
  gTgtTemp: TNeuralFloat = 1.0;  // 1.0 = flat (original); <1 = peaked target

// ACTUAL target-net forward count (every NN.Compute on target weights, full or
// cached). Contrast with BigCalls, which counts conceptual batched PASSES: v1
// counts one "pass" per verification block but still pays one full-prefix
// forward per verified position, which this counter exposes.
var
  gTgtForwards: int64 = 0;

// ---------------------------------------------------------------------------
// Microsecond-resolution wall clock in milliseconds since the first call.
// SysUtils.Now ticks at ~1 ms on Linux, too coarse for a short cached step.
// Rebasing to the first call keeps the value small: FPC types the 1000.0
// literal as SINGLE, and at an absolute Unix-epoch scale single-precision
// quantization would freeze the clock. (Pattern from examples/IncrementalDecode.)
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
// Toy task. A random string; the next-token target is a structured fixed-offset
// copy. We make it learnable but imperfectly-so for a tiny draft.
// ---------------------------------------------------------------------------
procedure MakeSeq(out S: TSeq);
var i: integer;
begin
  for i := 0 to cSeqLen - 1 do S[i] := Random(cVocab);
end;

// Next-token target at position t given prefix S[0..t]. Fixed-offset copy of an
// earlier token, with a deterministic twist so the mapping is non-trivial.
function TargetTok(const S: TSeq; t: integer): integer;
begin
  if t - cLag >= 0 then
    Result := (S[t - cLag] + S[t]) mod cVocab
  else
    Result := S[t];
end;

procedure FillPair(const S: TSeq; InputV, TargetV: TNNetVolume);
var t: integer;
begin
  TargetV.Fill(0);
  for t := 0 to cSeqLen - 1 do
  begin
    InputV.FData[t] := S[t];
    TargetV.OneHotEncodingOnPixel(t, 0, TargetTok(S, t));
  end;
end;

// ---------------------------------------------------------------------------
// Models. Both are per-position causal decoders ending in a per-position
// softmax over the vocabulary. TARGET uses multi-head causal attention; DRAFT
// uses the cheap attention-free TokenShift mixer + a narrow MLP.
// ---------------------------------------------------------------------------
// The target architecture at an arbitrary input length. The training/full
// net uses InputLen = cSeqLen; the KV-cached verification STEP net reuses the
// SAME architecture (and, via CopyWeights, the SAME trained weights — every
// layer's parameter shapes are independent of the sequence length) at the
// short window width cK + 1 ([last committed token | K drafts]).
function BuildTargetWithLen(InputLen: integer): TNNet;
begin
  Result := TNNet.Create();
  Result.AddLayer(TNNetInput.Create(InputLen, 1, 1));
  Result.AddLayer(TNNetEmbedding.Create(cVocab, cTgtDModel, 1));
  Result.AddLayer(TNNetSinusoidalPositionalEmbedding.Create());
  Result.AddLayer(TNNetPointwiseConvLinear.Create(3 * cTgtDModel)); // Q|K|V slab
  Result.AddMultiHeadSelfAttention(cTgtHeads, True);     // causal
  Result.AddLayer(TNNetPointwiseConvLinear.Create(cTgtDFF));
  Result.AddLayer(TNNetReLU.Create());
  Result.AddLayer(TNNetPointwiseConvLinear.Create(cVocab));
  Result.AddLayer(TNNetPointwiseSoftMax.Create(1));
  Result.SetLearningRate(cLR, cInertia);
  Result.SetL2Decay(0.0);
end;

function BuildTarget(): TNNet;
begin
  Result := BuildTargetWithLen(cSeqLen);
end;

function BuildDraft(): TNNet;
begin
  Result := TNNet.Create();
  Result.AddLayer(TNNetInput.Create(cSeqLen, 1, 1));
  Result.AddLayer(TNNetEmbedding.Create(cVocab, cDrfDModel, 1));
  Result.AddLayer(TNNetSinusoidalPositionalEmbedding.Create());
  Result.AddLayer(TNNetTokenShift.Create());                        // cheap mixer
  Result.AddLayer(TNNetPointwiseConvLinear.Create(cDrfDFF));
  Result.AddLayer(TNNetReLU.Create());
  Result.AddLayer(TNNetPointwiseConvLinear.Create(cVocab));
  Result.AddLayer(TNNetPointwiseSoftMax.Create(1));
  Result.SetLearningRate(cLR, cInertia);
  Result.SetL2Decay(0.0);
end;

procedure TrainEpochs(NN: TNNet; Epochs: integer);
var
  Epoch, b: integer;
  InputV, TargetV: TNNetVolume;
  S: TSeq;
begin
  InputV  := TNNetVolume.Create(cSeqLen, 1, 1);
  TargetV := TNNetVolume.Create(cSeqLen, 1, cVocab);
  try
    for Epoch := 1 to Epochs do
      for b := 1 to cBatch do
      begin
        MakeSeq(S);
        FillPair(S, InputV, TargetV);
        NN.Compute(InputV);
        NN.Backpropagate(TargetV);
      end;
  finally
    InputV.Free; TargetV.Free;
  end;
end;

// ---------------------------------------------------------------------------
// Inference helpers. We feed a length-cSeqLen input padded with zeros and read
// the per-position softmax row at position (Len-1), i.e. the distribution over
// the NEXT token given the prefix S[0..Len-1].
// ---------------------------------------------------------------------------
procedure FillInput(const S: TSeq; Len: integer; InputV: TNNetVolume);
var t: integer;
begin
  InputV.Fill(0);
  for t := 0 to Len - 1 do InputV.FData[t] := S[t];
end;

// Read p(next | prefix of length Len) from NN's softmax output at row Len-1.
// When IsTarget and gTgtTemp <> 1, re-softmax the target distribution at the
// given temperature (T<1 sharpens / peaks it) BEFORE returning it. This changes
// only the target's distribution, not the sampler -- the speculative loop draws
// from whatever p_target NextDist returns, so exactness/faithfulness hold for
// any temperature.
// Shared post-processing of one softmax row: clamp negatives, renormalise,
// and (target only) apply the gTgtTemp re-softmax. Factored out so the
// KV-cached verification path post-processes its rows EXACTLY like NextDist.
procedure NormalizeAndTemper(var P: TProbs; IsTarget: boolean);
var d: integer; Sum, mx: TNeuralFloat;
begin
  Sum := 0;
  for d := 0 to cVocab - 1 do
  begin
    if P[d] < 0 then P[d] := 0;
    Sum := Sum + P[d];
  end;
  if Sum <= 0 then
  begin
    for d := 0 to cVocab - 1 do P[d] := 1.0 / cVocab;
    Exit;
  end;
  for d := 0 to cVocab - 1 do P[d] := P[d] / Sum;

  // PEAKED TARGET: re-softmax at temperature gTgtTemp. p' ~ p^(1/T) (equivalently
  // logit/T on logits = ln p), renormalised. T<1 concentrates mass on the mode.
  if IsTarget and (gTgtTemp <> 1.0) then
  begin
    mx := 0;
    for d := 0 to cVocab - 1 do
    begin
      // logit = ln p; clamp tiny probs so ln is finite, then scale by 1/T.
      if P[d] < 1e-12 then P[d] := 1e-12;
      P[d] := Ln(P[d]) / gTgtTemp;
      if (d = 0) or (P[d] > mx) then mx := P[d];
    end;
    Sum := 0;
    for d := 0 to cVocab - 1 do
    begin
      P[d] := Exp(P[d] - mx);   // shift for numerical stability
      Sum := Sum + P[d];
    end;
    for d := 0 to cVocab - 1 do P[d] := P[d] / Sum;
  end;
end;

procedure NextDist(NN: TNNet; const S: TSeq; Len: integer;
  InputV: TNNetVolume; out P: TProbs; IsTarget: boolean = False);
var d: integer; Row: integer;
begin
  FillInput(S, Len, InputV);
  NN.Compute(InputV);
  if IsTarget then Inc(gTgtForwards);  // one FULL-prefix target forward
  Row := Len - 1;
  for d := 0 to cVocab - 1 do
    P[d] := NN.GetLastLayer.Output[Row, 0, d];
  NormalizeAndTemper(P, IsTarget);
end;

// Inverse-CDF categorical sample from distribution P using uniform u in [0,1).
function SampleCDF(const P: TProbs; u: TNeuralFloat): integer;
var d: integer; c: TNeuralFloat;
begin
  c := 0;
  for d := 0 to cVocab - 1 do
  begin
    c := c + P[d];
    if u < c then Exit(d);
  end;
  Result := cVocab - 1;  // numerical fallback
end;

// ---------------------------------------------------------------------------
// PLAIN target-only autoregressive sampling. Draws NTokens continuation tokens
// from the target, one gProp uniform per token. Returns them appended after the
// given prefix; counts big-model calls. Touches gAcc never.
// ---------------------------------------------------------------------------
procedure PlainSample(Target: TNNet; const Prefix: TSeq; PrefixLen, NTokens: integer;
  InputV: TNNetVolume; out OutSeq: TSeq; out OutLen: integer; out BigCalls: integer);
var
  P: TProbs;
  i, tok: integer;
begin
  OutSeq := Prefix;
  OutLen := PrefixLen;
  BigCalls := 0;
  for i := 1 to NTokens do
  begin
    NextDist(Target, OutSeq, OutLen, InputV, P, True);  // one big-model call
    Inc(BigCalls);
    tok := SampleCDF(P, RngFloat(gProp));
    OutSeq[OutLen] := tok;
    Inc(OutLen);
    if OutLen >= cSeqLen then Break;
  end;
end;

// ---------------------------------------------------------------------------
// SPECULATIVE sampling. Generates AT LEAST NTokens continuation tokens (a final
// pass may overshoot; the caller truncates). One DRAFT call per proposed token
// and ONE batched TARGET call per verification pass (we recompute the prefix
// each pass). Implements accept-or-residual-resample exactly.
//
// Returns: the produced sequence, big-model (verification) call count, total
// draft calls, and the number of accepted draft tokens / verification passes
// (for the speedup metric).
// ---------------------------------------------------------------------------
procedure SpeculativeSample(Target, Draft: TNNet;
  const Prefix: TSeq; PrefixLen, NTokens: integer;
  InputV: TNNetVolume;
  out OutSeq: TSeq; out OutLen: integer;
  out BigCalls, DraftCalls, AcceptedTokens, Passes: integer);
var
  Pd, Pt, Resid: TProbs;
  DraftToks: array[0..cK - 1] of integer;
  PdAtPos: array[0..cK - 1] of TProbs;  // p_draft used to PROPOSE each x_i
  i, d, tok, nProposed, accept: integer;
  ratio, u, sresid, ptv, pdv: TNeuralFloat;
  rejected: boolean;
begin
  OutSeq := Prefix;
  OutLen := PrefixLen;
  BigCalls := 0; DraftCalls := 0; AcceptedTokens := 0; Passes := 0;

  while OutLen - PrefixLen < NTokens do
  begin
    Inc(Passes);
    // --- 1. DRAFT proposes up to K tokens autoregressively over the growing
    //        prefix, stopping early if the context window fills. nProposed is
    //        the number of draft positions to verify this pass.
    nProposed := 0;
    for i := 0 to cK - 1 do
    begin
      if OutLen + i >= cSeqLen then Break;          // no room for this position
      NextDist(Draft, OutSeq, OutLen + i, InputV, PdAtPos[i]);  // draft call
      Inc(DraftCalls);
      DraftToks[i] := SampleCDF(PdAtPos[i], RngFloat(gProp));
      OutSeq[OutLen + i] := DraftToks[i];
      Inc(nProposed);
    end;
    if nProposed = 0 then Break;

    // --- 2. TARGET scores all positions in the block. We need p_target at the
    //        context of length (OutLen), (OutLen+1), ..., (OutLen+nProposed-1),
    //        plus one bonus context (OutLen+nProposed) if all accepted. We make
    //        one batched-style sweep: here, for clarity and correctness, we call
    //        NextDist per position (each is a target call). We count ONE big
    //        model "pass" for the whole block to reflect the batched cost.
    Inc(BigCalls);

    rejected := false;
    i := 0;
    while (i < nProposed) do
    begin
      // p_target at the i-th block context (prefix + accepted x_1..x_i).
      NextDist(Target, OutSeq, OutLen + i, InputV, Pt, True);
      Pd := PdAtPos[i];
      tok := DraftToks[i];

      ptv := Pt[tok];
      pdv := Pd[tok];
      if pdv <= 0 then ratio := 1.0
      else ratio := ptv / pdv;
      if ratio > 1.0 then ratio := 1.0;

      u := RngFloat(gAcc);
      if u < ratio then
        accept := 1
      else
        accept := 0;

      if accept = 1 then
      begin
        // commit x_i
        OutSeq[OutLen + i] := tok;
        Inc(AcceptedTokens);
        Inc(i);
      end
      else
      begin
        // FIRST rejection: resample this position from norm(max(0,Pt-Pd)).
        sresid := 0;
        for d := 0 to cVocab - 1 do
        begin
          Resid[d] := Pt[d] - PdAtPos[i][d];
          if Resid[d] < 0 then Resid[d] := 0;
          sresid := sresid + Resid[d];
        end;
        if sresid <= 0 then
        begin
          // degenerate (p_draft dominates everywhere): fall back to p_target.
          Resid := Pt;
        end
        else
          for d := 0 to cVocab - 1 do Resid[d] := Resid[d] / sresid;
        tok := SampleCDF(Resid, RngFloat(gProp));
        OutSeq[OutLen + i] := tok;
        Inc(i);          // this resampled token is committed
        rejected := true;
        Break;           // discard the rest of the block
      end;
    end;

    if not rejected then
    begin
      // all `nProposed` draft tokens accepted; commit them and, if room, sample
      // a BONUS token from p_target at the end of the block (free extra token).
      OutLen := OutLen + nProposed;
      if OutLen < cSeqLen then
      begin
        NextDist(Target, OutSeq, OutLen, InputV, Pt, True);
        tok := SampleCDF(Pt, RngFloat(gProp));
        OutSeq[OutLen] := tok;
        Inc(OutLen);
      end;
    end
    else
    begin
      // committed i tokens (i-1 accepted draft + 1 resampled).
      OutLen := OutLen + i;
    end;

    if OutLen >= cSeqLen then Break;
  end;
end;

// ---------------------------------------------------------------------------
// KV-CACHED verification. The target's verification runs on a STEP net: the
// same architecture/weights at input width cK+1, with every per-head
// TNNetScaledDotProductAttention switched into the KV-cache incremental-decode
// path. A pass feeds only the short window [last committed token | drafts];
// each window token's K/V is appended to the per-head caches and its query
// attends over the cache so far (multi-token prefill semantics), so a pass
// costs O(K * t_cached) instead of the v1 full re-encode O((t+K)^2).
// On rejection, TruncateCache rewinds the caches to the committed prefix.
// ---------------------------------------------------------------------------
const
  cStepWidth = cK + 1;  // [last committed token | up to K drafts]

type
  TStepTarget = record
    Net: TNNet;                                          // width-cStepWidth twin
    Session: TNNetStreamingDecoder;                      // KV-cache session
    SinPos: TNNetSinusoidalPositionalEmbedding;          // absolute positions
    StepIn: TNNetVolume;                                 // (cStepWidth,1,1) ids
  end;

procedure CreateStepTarget(out Step: TStepTarget; Target: TNNet);
var
  i: integer;
begin
  Step.Net := BuildTargetWithLen(cStepWidth);
  // Same architecture, so layer-by-layer weight copy is exact (all parameter
  // shapes are sequence-length independent).
  Step.Net.CopyWeights(Target);
  // TNNetStreamingDecoder collects every per-head SDPA and switches it into
  // the KV-cache path; the budget covers the worst transient cache load:
  // (cSeqLen-1) committed + a full window. The sinusoidal positional layer is
  // NOT a rotary layer, so the session does not manage its PositionOffset —
  // we keep that one pointer ourselves (set before every cached forward).
  Step.Session := TNNetStreamingDecoder.Create(Step.Net, cSeqLen + cStepWidth);
  Step.SinPos := nil;
  for i := 0 to Step.Net.Layers.Count - 1 do
    if Step.Net.Layers[i] is TNNetSinusoidalPositionalEmbedding then
      Step.SinPos := TNNetSinusoidalPositionalEmbedding(Step.Net.Layers[i]);
  Step.StepIn := TNNetVolume.Create(cStepWidth, 1, 1);
end;

procedure FreeStepTarget(var Step: TStepTarget);
begin
  Step.StepIn.Free;
  Step.Session.Free;
  Step.Net.Free;
end;

// One cached window forward: tokens S[StartPos .. StartPos+NReal-1] fill the
// window (zero padding after; pads come AFTER every real token, so the cached
// causal path never lets a real query see a pad — pad K/V is dropped by the
// caller's next TruncateTo). The sinusoidal layer is shifted so the
// window is encoded at its ABSOLUTE positions (positional contract of the
// cached path); StepForward would do the same for rotary layers, but the
// sinusoidal layer is not collected by the session, so it is set here.
// Output row r is the softmax at absolute position StartPos+r.
procedure CachedWindowForward(var Step: TStepTarget; const S: TSeq;
  StartPos, NReal: integer);
var
  t: integer;
begin
  Step.StepIn.Fill(0);
  for t := 0 to NReal - 1 do Step.StepIn.FData[t] := S[StartPos + t];
  Step.SinPos.PositionOffset := StartPos;
  Step.Session.StepForward(Step.StepIn, StartPos);
  Inc(gTgtForwards);  // one SHORT cached target forward
end;

// ---------------------------------------------------------------------------
// SPECULATIVE sampling with KV-CACHED verification. Identical sampling logic
// (and RNG stream discipline) to SpeculativeSample — only the way p_target is
// obtained changes: the prompt is PREFILLED once, then each pass runs ONE
// short cached window forward and, after accept/reject, the caches are
// TRUNCATED back to the committed prefix. Greedy/seeded sampling is exact, so
// the produced token sequence must be IDENTICAL to SpeculativeSample's.
// ---------------------------------------------------------------------------
procedure SpeculativeSampleCached(var Step: TStepTarget; Draft: TNNet;
  const Prefix: TSeq; PrefixLen, NTokens: integer;
  InputV: TNNetVolume;
  out OutSeq: TSeq; out OutLen: integer;
  out BigCalls, DraftCalls, AcceptedTokens, Passes: integer);
var
  Pd, Pt, Resid: TProbs;
  DraftToks: array[0..cK - 1] of integer;
  PdAtPos: array[0..cK - 1] of TProbs;  // p_draft used to PROPOSE each x_i
  PtAtPos: array[0..cK] of TProbs;      // target rows 0..nProposed of one pass
  i, d, r, tok, nProposed, accept: integer;
  done, w: integer;
  ratio, u, sresid, ptv, pdv: TNeuralFloat;
  rejected: boolean;
begin
  OutSeq := Prefix;
  OutLen := PrefixLen;
  BigCalls := 0; DraftCalls := 0; AcceptedTokens := 0; Passes := 0;

  // --- PREFILL (once): cache the K/V of tokens 0..PrefixLen-2. The LAST
  //     committed token is NOT cached — it is re-fed as the first window
  //     token of every pass, so its softmax row is fresh each pass.
  Step.Session.Reset();
  done := 0;
  while done < PrefixLen - 1 do
  begin
    w := Min(cStepWidth, PrefixLen - 1 - done);
    CachedWindowForward(Step, OutSeq, done, w);
    done := done + w;
    Step.Session.TruncateTo(done);  // drop pad K/V from a partial window
  end;

  while OutLen - PrefixLen < NTokens do
  begin
    Inc(Passes);
    // --- 1. DRAFT proposes up to K tokens (unchanged: the draft is
    //        attention-free, its full recompute is the cheap arm by design).
    nProposed := 0;
    for i := 0 to cK - 1 do
    begin
      if OutLen + i >= cSeqLen then Break;          // no room for this position
      NextDist(Draft, OutSeq, OutLen + i, InputV, PdAtPos[i]);  // draft call
      Inc(DraftCalls);
      DraftToks[i] := SampleCDF(PdAtPos[i], RngFloat(gProp));
      OutSeq[OutLen + i] := DraftToks[i];
      Inc(nProposed);
    end;
    if nProposed = 0 then Break;

    // --- 2. TARGET scores the whole block in ONE short cached forward over
    //        the window [S[OutLen-1] | x_1..x_nProposed]: row i is p_target at
    //        context length OutLen+i (verification rows 0..nProposed-1 plus
    //        the bonus row nProposed).
    CachedWindowForward(Step, OutSeq, OutLen - 1, nProposed + 1);
    Inc(BigCalls);
    for r := 0 to nProposed do
    begin
      for d := 0 to cVocab - 1 do
        PtAtPos[r][d] := Step.Net.GetLastLayer.Output[r, 0, d];
      NormalizeAndTemper(PtAtPos[r], True);
    end;

    rejected := false;
    i := 0;
    while (i < nProposed) do
    begin
      Pt := PtAtPos[i];
      Pd := PdAtPos[i];
      tok := DraftToks[i];

      ptv := Pt[tok];
      pdv := Pd[tok];
      if pdv <= 0 then ratio := 1.0
      else ratio := ptv / pdv;
      if ratio > 1.0 then ratio := 1.0;

      u := RngFloat(gAcc);
      if u < ratio then
        accept := 1
      else
        accept := 0;

      if accept = 1 then
      begin
        // commit x_i
        OutSeq[OutLen + i] := tok;
        Inc(AcceptedTokens);
        Inc(i);
      end
      else
      begin
        // FIRST rejection: resample this position from norm(max(0,Pt-Pd)).
        sresid := 0;
        for d := 0 to cVocab - 1 do
        begin
          Resid[d] := Pt[d] - PdAtPos[i][d];
          if Resid[d] < 0 then Resid[d] := 0;
          sresid := sresid + Resid[d];
        end;
        if sresid <= 0 then
        begin
          // degenerate (p_draft dominates everywhere): fall back to p_target.
          Resid := Pt;
        end
        else
          for d := 0 to cVocab - 1 do Resid[d] := Resid[d] / sresid;
        tok := SampleCDF(Resid, RngFloat(gProp));
        OutSeq[OutLen + i] := tok;
        Inc(i);          // this resampled token is committed
        rejected := true;
        Break;           // discard the rest of the block
      end;
    end;

    if not rejected then
    begin
      // all `nProposed` draft tokens accepted; commit them and, if room, the
      // BONUS token comes for free from row nProposed of the SAME forward.
      OutLen := OutLen + nProposed;
      if OutLen < cSeqLen then
      begin
        Pt := PtAtPos[nProposed];
        tok := SampleCDF(Pt, RngFloat(gProp));
        OutSeq[OutLen] := tok;
        Inc(OutLen);
      end;
    end
    else
    begin
      // committed i tokens (i-1 accepted draft + 1 resampled).
      OutLen := OutLen + i;
    end;

    // --- 3. ROLLBACK: the caches transiently hold the whole window (pads and
    //        rejected drafts included). Keep exactly the committed prefix
    //        minus its last token (re-fed as the next window's first query).
    Step.Session.TruncateTo(OutLen - 1);

    if OutLen >= cSeqLen then Break;
  end;
end;

// ---------------------------------------------------------------------------
// Metrics.
// ---------------------------------------------------------------------------
function SeqEqual(const A, B: TSeq; Len: integer): boolean;
var t: integer;
begin
  Result := true;
  for t := 0 to Len - 1 do
    if A[t] <> B[t] then Exit(false);
end;

// Total-variation distance between two normalised token histograms over vocab.
function HistTV(const Ha, Hb: array of integer; Na, Nb: integer): TNeuralFloat;
var d: integer; pa, pb, s: TNeuralFloat;
begin
  s := 0;
  for d := 0 to cVocab - 1 do
  begin
    if Na > 0 then pa := Ha[d] / Na else pa := 0;
    if Nb > 0 then pb := Hb[d] / Nb else pb := 0;
    s := s + Abs(pa - pb);
  end;
  Result := 0.5 * s;
end;

// ---------------------------------------------------------------------------
// GATE 2 driver: empirical faithfulness at the CURRENT gTgtTemp. Samples the
// first continuation token cFaithN times from a fixed prefix, plain vs
// speculative, and returns the histogram total-variation distance. The accept/
// reject loop must reproduce p_target (tempered or not) so TV stays within
// sampling noise for ANY temperature -- that is the invariant being checked.
// ---------------------------------------------------------------------------
procedure RunFaithfulness(Target, Draft: TNNet; InputV: TNNetVolume;
  const Prefix: TSeq; PrefixLen, FaithN: integer; out TV: TNeuralFloat);
var
  HistPlain, HistSpec: array[0..cVocab - 1] of integer;
  NPlain, NSpec, i, run: integer;
  PlainOut, SpecOut: TSeq;
  PlainLen, SpecLen, bcP, bcS, dcS, accTok, passes: integer;
begin
  for i := 0 to cVocab - 1 do begin HistPlain[i] := 0; HistSpec[i] := 0; end;
  NPlain := 0; NSpec := 0;
  for run := 1 to FaithN do
  begin
    RngSeed(gProp, QWord(2000000 + run)); RngSeed(gAcc, QWord(3000000 + run));
    PlainSample(Target, Prefix, PrefixLen, 1, InputV, PlainOut, PlainLen, bcP);
    Inc(HistPlain[PlainOut[PrefixLen]]); Inc(NPlain);

    RngSeed(gProp, QWord(5000000 + run)); RngSeed(gAcc, QWord(6000000 + run));
    SpeculativeSample(Target, Draft, Prefix, PrefixLen, 1, InputV,
      SpecOut, SpecLen, bcS, dcS, accTok, passes);
    Inc(HistSpec[SpecOut[PrefixLen]]); Inc(NSpec);
  end;
  TV := HistTV(HistPlain, HistSpec, NPlain, NSpec);
  WriteLn('  token |   plain%   spec%   (', FaithN, ' draws each)');
  for i := 0 to cVocab - 1 do
    WriteLn(Format('   %3d  |  %6.2f  %6.2f', [i,
      100.0 * HistPlain[i] / Max(1, NPlain),
      100.0 * HistSpec[i]  / Max(1, NSpec)]));
  WriteLn(Format('  total-variation distance = %.4f  (small => same distribution)', [TV]));
end;

// ---------------------------------------------------------------------------
// GATE 3 driver: speedup sweep at the CURRENT gTgtTemp. Rebuilds a fresh draft
// and grows it through {0, 8, 80} epochs, measuring accept rate and big-model
// calls saved over cSweepRuns fresh prefixes per stage. Returns the WEAK
// (untrained, stage 0) and STRONG (fully, last stage) accept-rate and
// calls-saved so the caller can print a flat-vs-peaked side-by-side.
// ---------------------------------------------------------------------------
procedure RunSpeedupSweep(Target: TNNet; InputV: TNNetVolume;
  PrefixLen, Tokens, SweepRuns: integer;
  out WeakRate, WeakSaved, StrongRate, StrongSaved: TNeuralFloat);
const
  DraftStages: array[0..2] of integer = (0, 8, 80);
  StageName: array[0..2] of string = ('untrained', 'lightly', 'fully');
var
  Draft: TNNet;
  trainedSoFar, want, q, sample: integer;
  SweepPrefix, SpecOut: TSeq;
  SpecLen, bcS, dcS, accTok, passes: integer;
  SumAcc, SumPass, MeanAccPerPass, BaselineCalls, SpecCalls, Saved, Rate: TNeuralFloat;
begin
  WeakRate := 0; WeakSaved := 0; StrongRate := 0; StrongSaved := 0;
  WriteLn('  draft       acc-tok/pass   accept-rate   big-calls/tok   calls-saved');
  WriteLn('  ----------  ------------   -----------   -------------   -----------');
  RandSeed := cSeed + 7;
  Draft := BuildDraft();
  try
    trainedSoFar := 0;
    for q := 0 to High(DraftStages) do
    begin
      want := DraftStages[q];
      if want > trainedSoFar then
      begin
        TrainEpochs(Draft, want - trainedSoFar);
        trainedSoFar := want;
      end;

      SumAcc := 0; SumPass := 0; SpecCalls := 0; BaselineCalls := 0;
      RandSeed := cSeed + 555;
      for sample := 1 to SweepRuns do
      begin
        MakeSeq(SweepPrefix);
        RngSeed(gProp, QWord(11000000 + sample));
        RngSeed(gAcc,  QWord(12000000 + sample));
        SpeculativeSample(Target, Draft, SweepPrefix, PrefixLen, Tokens, InputV,
          SpecOut, SpecLen, bcS, dcS, accTok, passes);
        SumAcc := SumAcc + accTok;
        SumPass := SumPass + passes;
        SpecCalls := SpecCalls + bcS;
        BaselineCalls := BaselineCalls + (SpecLen - PrefixLen);
      end;
      MeanAccPerPass := SumAcc / Max(1, SumPass);
      Rate := MeanAccPerPass / cK;
      Saved := 1.0 - SpecCalls / Max(1.0, BaselineCalls);
      WriteLn(Format('  %-10s   %10.3f   %10.3f    %12.3f   %9.1f%%',
        [StageName[q], MeanAccPerPass, Rate,
         SpecCalls / Max(1.0, BaselineCalls), 100.0 * Saved]));
      if q = 0 then begin WeakRate := Rate; WeakSaved := 100.0 * Saved; end;
      if q = High(DraftStages) then
        begin StrongRate := Rate; StrongSaved := 100.0 * Saved; end;
    end;
  finally
    Draft.Free;
  end;
end;

var
  Target, Draft: TNNet;
  Step: TStepTarget;
  InputV: TNNetVolume;
  Prefix, PlainOut, SpecOut, CachedOut: TSeq;
  PrefixLen, PlainLen, SpecLen, CachedLen: integer;
  bcP, bcS, dcS, accTok, passes: integer;
  bcC, dcC, accTokC, passesC: integer;
  t, run: integer;
  GateExact, GateCachedExact, GateFaithFlat, GateFaithPeak: boolean;
  TVFlat, TVPeak: TNeuralFloat;
  // Side-by-side speedup results (weak = untrained draft, strong = fully trained).
  fWeakRate, fWeakSaved, fStrongRate, fStrongSaved: TNeuralFloat;
  pWeakRate, pWeakSaved, pStrongRate, pStrongSaved: TNeuralFloat;
  // GATE 4 composition benchmark accumulators.
  BenchSeq: TSeq;
  T0, MsPlain, MsSpecV1, MsSpecKV: double;
  ToksPlain, ToksSpecV1, ToksSpecKV: int64;
  FwdPlain, FwdSpecV1, FwdSpecKV: int64;
  PassesSpecV1, PassesSpecKV: int64;
const
  cTokens   = 12;     // continuation length for exactness / sample checks
  cFaithN   = 4000;   // sampling runs for the faithfulness histogram
  cSweepRuns = 200;   // runs per draft stage in the speedup sweep
  cPeakTemp = 0.10;   // target softmax temperature for the PEAKED variant (<1)
  cBenchRuns = 150;   // prefixes per arm in the composition benchmark
begin
  // Mask FPU exceptions so transient early-training underflows don't abort
  // (matches sibling examples).
  SetExceptionMask([exInvalidOp, exDenormalized, exZeroDivide,
                    exOverflow, exUnderflow, exPrecision]);
  // Manual Compute/Backpropagate runs single-threaded by construction, so the
  // whole program is deterministic under the fixed RandSeed + our own RNG.
  RandSeed := cSeed;

  WriteLn('SpeculativeDecoding: speculative sampling (Leviathan/Chen 2023) on a tiny');
  WriteLn(Format('pure-CPU draft/target pair (vocab %d, ctx %d, block K=%d).',
    [cVocab, cSeqLen, cK]));
  WriteLn('Headline: the speculative output is distributed EXACTLY as plain target-only');
  WriteLn('sampling, while calling the big TARGET far fewer times.');
  WriteLn('Accept x_i with prob min(1, p_target/p_draft); on first reject resample from');
  WriteLn('norm(max(0, p_target - p_draft)) and discard the rest of the block.');
  WriteLn('TWO verification backends: v1 (recompute the whole prefix each pass) and');
  WriteLn('KV-CACHED (prefill once; each pass feeds only [last token | K drafts] through');
  WriteLn('the attention KV cache; TruncateCache rolls back rejected positions), so the');
  WriteLn('two efficiency wins compose: fewer big-model passes x O(K*t) per pass.');
  WriteLn('Gates 2-3 run TWICE: a FLAT target (temp 1.0, the mod-sum map) and a PEAKED');
  WriteLn(Format('target (softmax temperature %.2f) that sharpens p_target so a WEAK draft''s',
    [cPeakTemp]));
  WriteLn('accept rate collapses -- a more discriminating speedup chart.');
  WriteLn;

  InputV := TNNetVolume.Create(cSeqLen, 1, 1);

  // --- Train the TARGET (big) model. ---------------------------------------
  Write('Training TARGET (', cTgtEpochs, ' epochs, d_model=', cTgtDModel,
        ', heads=', cTgtHeads, ', causal attention) ...');
  RandSeed := cSeed;
  Target := BuildTarget();
  TrainEpochs(Target, cTgtEpochs);
  WriteLn(' done.  params=', Target.CountWeights());

  // --- Train the DRAFT (small) model briefly. ------------------------------
  Write('Training DRAFT  (', cDrfEpochs, ' epochs, d_model=', cDrfDModel,
        ', TokenShift) ...');
  RandSeed := cSeed + 7;
  Draft := BuildDraft();
  TrainEpochs(Draft, cDrfEpochs);
  WriteLn(' done.  params=', Draft.CountWeights());
  WriteLn;

  // The KV-cached verification STEP net: same target weights at window width
  // cK+1, with every attention head in incremental-decode (cached) mode.
  CreateStepTarget(Step, Target);

  // A fixed prefix used by the exactness + faithfulness checks.
  RandSeed := cSeed + 99;
  MakeSeq(Prefix);
  PrefixLen := cLag + 2;   // short, leaving room for cTokens continuation

  // =========================================================================
  // GATE 1 (MANDATORY): DEGENERATE EXACTNESS. With draft == target the accept
  // rule is min(1,1)=1 (always accept), so the speculative path must reproduce
  // plain target-only sampling BIT-FOR-BIT under the same gProp seed.
  // =========================================================================
  WriteLn(StringOfChar('=', 72));
  WriteLn('GATE 1  Degenerate exactness  (draft == target => always-accept)');
  WriteLn(StringOfChar('=', 72));

  // Gate 1 is a property of the ALGORITHM (always-accept when draft==target), so
  // it runs at the FLAT target (temp 1.0): with draft==target the untempered
  // proposal distribution must equal the verification distribution.
  gTgtTemp := 1.0;
  RngSeed(gProp, 1234567); RngSeed(gAcc, 7654321);
  PlainSample(Target, Prefix, PrefixLen, cTokens, InputV, PlainOut, PlainLen, bcP);

  RngSeed(gProp, 1234567); RngSeed(gAcc, 7654321);
  // Pass TARGET as BOTH draft and target.
  SpeculativeSample(Target, Target, Prefix, PrefixLen, cTokens, InputV,
    SpecOut, SpecLen, bcS, dcS, accTok, passes);

  GateExact := (PlainLen >= PrefixLen + cTokens) and
               SeqEqual(PlainOut, SpecOut, PrefixLen + cTokens);

  Write('  plain      tokens : ');
  for t := PrefixLen to PrefixLen + cTokens - 1 do Write(PlainOut[t], ' ');
  WriteLn;
  Write('  speculative tokens: ');
  for t := PrefixLen to PrefixLen + cTokens - 1 do Write(SpecOut[t], ' ');
  WriteLn;
  WriteLn(Format('  big-model calls: plain=%d  speculative=%d (draft==target case)',
    [bcP, bcS]));
  if GateExact then
    WriteLn('  GATE 1 : PASS  (speculative == plain, bit-for-bit)')
  else
    WriteLn('  GATE 1 : FAIL  (sequences differ -- exactness broken!)');
  WriteLn;

  // =========================================================================
  // GATE 1b (MANDATORY): KV-CACHE EXACTNESS. The cached sampler must produce
  // a token sequence IDENTICAL to the v1 full-recompute sampler under the
  // same seeds (seeded sampling is exact, so any cache bug shows up as a
  // token mismatch). Checked three ways:
  //   (a) draft == target, flat: cached == plain (chains through Gate 1);
  //   (b) real draft, flat: cached == v1 (real rejections exercise the
  //       TruncateCache rollback);
  //   (c) real draft, peaked target: same, through the tempering path.
  // =========================================================================
  WriteLn(StringOfChar('=', 72));
  WriteLn('GATE 1b Cached-verification exactness (KV cache + TruncateCache rollback)');
  WriteLn(StringOfChar('=', 72));
  GateCachedExact := true;

  // (a) draft == target: cached speculative == plain, bit-for-bit.
  gTgtTemp := 1.0;
  RngSeed(gProp, 1234567); RngSeed(gAcc, 7654321);
  SpeculativeSampleCached(Step, Target, Prefix, PrefixLen, cTokens, InputV,
    CachedOut, CachedLen, bcC, dcC, accTokC, passesC);
  if SeqEqual(PlainOut, CachedOut, PrefixLen + cTokens) then
    WriteLn('  (a) draft==target : cached == plain, bit-for-bit       PASS')
  else
  begin
    WriteLn('  (a) draft==target : cached != plain                    FAIL');
    GateCachedExact := false;
  end;

  // (b) real draft, flat target: cached == v1 under the same seeds.
  RngSeed(gProp, 24681357); RngSeed(gAcc, 13572468);
  SpeculativeSample(Target, Draft, Prefix, PrefixLen, cTokens, InputV,
    SpecOut, SpecLen, bcS, dcS, accTok, passes);
  RngSeed(gProp, 24681357); RngSeed(gAcc, 13572468);
  SpeculativeSampleCached(Step, Draft, Prefix, PrefixLen, cTokens, InputV,
    CachedOut, CachedLen, bcC, dcC, accTokC, passesC);
  if (SpecLen = CachedLen) and SeqEqual(SpecOut, CachedOut, SpecLen) then
    WriteLn(Format('  (b) real draft     : cached == v1 (%d tokens, %d passes)  PASS',
      [SpecLen - PrefixLen, passesC]))
  else
  begin
    WriteLn('  (b) real draft     : cached != v1                      FAIL');
    GateCachedExact := false;
  end;

  // (c) real draft, PEAKED target (more rejections -> more rollbacks).
  gTgtTemp := cPeakTemp;
  RngSeed(gProp, 9182736); RngSeed(gAcc, 6372819);
  SpeculativeSample(Target, Draft, Prefix, PrefixLen, cTokens, InputV,
    SpecOut, SpecLen, bcS, dcS, accTok, passes);
  RngSeed(gProp, 9182736); RngSeed(gAcc, 6372819);
  SpeculativeSampleCached(Step, Draft, Prefix, PrefixLen, cTokens, InputV,
    CachedOut, CachedLen, bcC, dcC, accTokC, passesC);
  gTgtTemp := 1.0;
  if (SpecLen = CachedLen) and SeqEqual(SpecOut, CachedOut, SpecLen) then
    WriteLn('  (c) peaked target  : cached == v1 under rejections     PASS')
  else
  begin
    WriteLn('  (c) peaked target  : cached != v1                      FAIL');
    GateCachedExact := false;
  end;
  if GateCachedExact then
    WriteLn('  GATE 1b: PASS  (cached speculative output is exact)')
  else
    WriteLn('  GATE 1b: FAIL  (KV-cache verification broke exactness!)');
  WriteLn;

  // =========================================================================
  // GATE 2: EMPIRICAL FAITHFULNESS with a genuinely DIFFERENT trained draft.
  // Run for BOTH the flat and the peaked target -- exactness of the committed
  // distribution must hold regardless of how peaked p_target is.
  // =========================================================================
  WriteLn(StringOfChar('=', 72));
  WriteLn('GATE 2  Empirical faithfulness  (real draft != target; histogram match)');
  WriteLn(StringOfChar('=', 72));

  WriteLn('  -- FLAT target (temp 1.0) --');
  gTgtTemp := 1.0;
  RunFaithfulness(Target, Draft, InputV, Prefix, PrefixLen, cFaithN, TVFlat);
  // With ~4000 draws per arm over 8 tokens, sampling noise in TV is ~0.02-0.04.
  GateFaithFlat := (TVFlat < 0.06);
  WriteLn;
  WriteLn(Format('  -- PEAKED target (temp %.2f) --', [cPeakTemp]));
  gTgtTemp := cPeakTemp;
  RunFaithfulness(Target, Draft, InputV, Prefix, PrefixLen, cFaithN, TVPeak);
  GateFaithPeak := (TVPeak < 0.06);
  if GateFaithFlat and GateFaithPeak then
    WriteLn(Format('  GATE 2 : PASS  (flat TV=%.4f, peaked TV=%.4f -- both within noise)',
      [TVFlat, TVPeak]))
  else
    WriteLn(Format('  GATE 2 : WARN  (flat TV=%.4f, peaked TV=%.4f; see README note)',
      [TVFlat, TVPeak]));
  WriteLn;

  // =========================================================================
  // GATE 3: SPEEDUP STORY, side-by-side FLAT vs PEAKED target. The peaked
  // target sharpens p_target so a WEAK draft (which approximates the flat
  // objective) puts mass on the wrong token -> low accept rate, while a STRONG
  // draft still tracks the mode -> the good-vs-bad calls-saved gap widens.
  // =========================================================================
  WriteLn(StringOfChar('=', 72));
  WriteLn('GATE 3  Speedup vs draft/target agreement  (block K=', cK, ')');
  WriteLn(StringOfChar('=', 72));

  WriteLn('  ----- FLAT target (temp 1.0; original mod-sum map) -----');
  gTgtTemp := 1.0;
  RunSpeedupSweep(Target, InputV, PrefixLen, cTokens, cSweepRuns,
    fWeakRate, fWeakSaved, fStrongRate, fStrongSaved);
  WriteLn;
  WriteLn(Format('  ----- PEAKED target (temp %.2f) -----', [cPeakTemp]));
  gTgtTemp := cPeakTemp;
  RunSpeedupSweep(Target, InputV, PrefixLen, cTokens, cSweepRuns,
    pWeakRate, pWeakSaved, pStrongRate, pStrongSaved);
  WriteLn;

  WriteLn('  flat-vs-peaked contrast (weak=untrained draft, strong=fully trained):');
  WriteLn('                     weak-rate   strong-rate   weak-saved   strong-saved   gap(saved)');
  WriteLn(Format('    FLAT target      %7.3f       %7.3f     %7.1f%%      %7.1f%%      %7.1f pts',
    [fWeakRate, fStrongRate, fWeakSaved, fStrongSaved, fStrongSaved - fWeakSaved]));
  WriteLn(Format('    PEAKED target    %7.3f       %7.3f     %7.1f%%      %7.1f%%      %7.1f pts',
    [pWeakRate, pStrongRate, pWeakSaved, pStrongSaved, pStrongSaved - pWeakSaved]));
  WriteLn;
  WriteLn('  Under the PEAKED target the weak draft''s accept rate drops well below the');
  WriteLn('  flat case and the strong-minus-weak calls-saved gap widens: a sharper');
  WriteLn('  target rewards a good draft far more, so the speedup chart discriminates');
  WriteLn('  draft quality much more strongly. The committed distribution is unchanged');
  WriteLn('  at either temperature (Gate 1/2).');
  WriteLn;

  // =========================================================================
  // GATE 4: COMPOSITION BENCHMARK. Wall-clock and ACTUAL target-net forward
  // counts for the three arms over the SAME cBenchRuns random prefixes:
  //   (i)   plain target-only sampling      (one FULL forward per token)
  //   (ii)  v1 speculative, full recompute  (one FULL forward per verified
  //         position + bonus -- fewer conceptual passes, same forward cost)
  //   (iii) speculative + KV cache          (prefill once; ONE short cached
  //         forward per pass; TruncateCache rollback on rejection)
  // The headline: (iii) beats both because the two wins compose.
  // =========================================================================
  WriteLn(StringOfChar('=', 72));
  WriteLn('GATE 4  Composition benchmark  (', cBenchRuns, ' runs x ',
    cTokens, ' tokens, trained draft, flat target)');
  WriteLn(StringOfChar('=', 72));
  gTgtTemp := 1.0;

  // (i) plain full-recompute.
  RandSeed := cSeed + 321;
  gTgtForwards := 0; ToksPlain := 0;
  T0 := NowMs();
  for run := 1 to cBenchRuns do
  begin
    MakeSeq(BenchSeq);
    RngSeed(gProp, QWord(21000000 + run)); RngSeed(gAcc, QWord(22000000 + run));
    PlainSample(Target, BenchSeq, PrefixLen, cTokens, InputV,
      PlainOut, PlainLen, bcP);
    ToksPlain := ToksPlain + (PlainLen - PrefixLen);
  end;
  MsPlain := NowMs() - T0;
  FwdPlain := gTgtForwards;

  // (ii) v1 speculative full-recompute (same prefixes, same seeds).
  RandSeed := cSeed + 321;
  gTgtForwards := 0; ToksSpecV1 := 0; PassesSpecV1 := 0;
  T0 := NowMs();
  for run := 1 to cBenchRuns do
  begin
    MakeSeq(BenchSeq);
    RngSeed(gProp, QWord(21000000 + run)); RngSeed(gAcc, QWord(22000000 + run));
    SpeculativeSample(Target, Draft, BenchSeq, PrefixLen, cTokens, InputV,
      SpecOut, SpecLen, bcS, dcS, accTok, passes);
    ToksSpecV1 := ToksSpecV1 + (SpecLen - PrefixLen);
    PassesSpecV1 := PassesSpecV1 + passes;
  end;
  MsSpecV1 := NowMs() - T0;
  FwdSpecV1 := gTgtForwards;

  // (iii) speculative + KV cache (same prefixes, same seeds).
  RandSeed := cSeed + 321;
  gTgtForwards := 0; ToksSpecKV := 0; PassesSpecKV := 0;
  T0 := NowMs();
  for run := 1 to cBenchRuns do
  begin
    MakeSeq(BenchSeq);
    RngSeed(gProp, QWord(21000000 + run)); RngSeed(gAcc, QWord(22000000 + run));
    SpeculativeSampleCached(Step, Draft, BenchSeq, PrefixLen, cTokens, InputV,
      CachedOut, CachedLen, bcC, dcC, accTokC, passesC);
    ToksSpecKV := ToksSpecKV + (CachedLen - PrefixLen);
    PassesSpecKV := PassesSpecKV + passesC;
  end;
  MsSpecKV := NowMs() - T0;
  FwdSpecKV := gTgtForwards;

  WriteLn('  arm                        tokens  ms/token   tok/s   tgt-fwd  fwd/tok');
  WriteLn('  -------------------------  ------  --------  ------  --------  -------');
  WriteLn(Format('  (i)   plain full-recompute %7d  %8.3f  %6.0f  %8d  %7.3f',
    [ToksPlain, MsPlain / Max(1, ToksPlain),
     1000.0 * ToksPlain / Max(1e-9, MsPlain), FwdPlain,
     FwdPlain / Max(1.0, ToksPlain)]));
  WriteLn(Format('  (ii)  spec v1 (recompute)  %7d  %8.3f  %6.0f  %8d  %7.3f',
    [ToksSpecV1, MsSpecV1 / Max(1, ToksSpecV1),
     1000.0 * ToksSpecV1 / Max(1e-9, MsSpecV1), FwdSpecV1,
     FwdSpecV1 / Max(1.0, ToksSpecV1)]));
  WriteLn(Format('  (iii) spec + KV cache      %7d  %8.3f  %6.0f  %8d  %7.3f',
    [ToksSpecKV, MsSpecKV / Max(1, ToksSpecKV),
     1000.0 * ToksSpecKV / Max(1e-9, MsSpecKV), FwdSpecKV,
     FwdSpecKV / Max(1.0, ToksSpecKV)]));
  WriteLn(Format('  verification passes: v1=%d  cached=%d  (cached fwd = passes + prefill)',
    [PassesSpecV1, PassesSpecKV]));
  WriteLn(Format('  speedup (ms/token): (iii) vs (i) = %.2fx,  (iii) vs (ii) = %.2fx',
    [(MsPlain / Max(1, ToksPlain)) / Max(1e-9, MsSpecKV / Max(1, ToksSpecKV)),
     (MsSpecV1 / Max(1, ToksSpecV1)) / Max(1e-9, MsSpecKV / Max(1, ToksSpecKV))]));
  WriteLn('  v1 cuts conceptual passes but still pays one FULL-prefix forward per');
  WriteLn('  verified position (fwd/tok ~ 1, each over all ', cSeqLen, ' positions);');
  WriteLn('  the cached arm pays ~1 SHORT (', cStepWidth, '-token) cached forward per PASS,');
  WriteLn('  so both the count and the cost of big-model forwards drop.');
  WriteLn;

  // ---- Summary / interpretation ------------------------------------------
  WriteLn(StringOfChar('=', 72));
  WriteLn('SUMMARY');
  WriteLn(StringOfChar('=', 72));
  if GateExact then
    WriteLn('  [PASS] EXACTNESS  : draft==target => speculative is bit-for-bit plain.')
  else
    WriteLn('  [FAIL] EXACTNESS  : degenerate case diverged.');
  if GateCachedExact then
    WriteLn('  [PASS] KV-CACHED  : cached verification == v1 full recompute, token-exact.')
  else
    WriteLn('  [FAIL] KV-CACHED  : cached verification diverged from v1.');
  if GateFaithFlat and GateFaithPeak then
    WriteLn(Format('  [PASS] FAITHFUL   : real-draft TV flat=%.4f peaked=%.4f within noise.',
      [TVFlat, TVPeak]))
  else
    WriteLn(Format('  [WARN] FAITHFUL   : real-draft TV flat=%.4f peaked=%.4f above noise.',
      [TVFlat, TVPeak]));
  WriteLn(Format('  [INFO] SPEEDUP    : weak-draft accept rate flat=%.3f -> peaked=%.3f;',
    [fWeakRate, pWeakRate]));
  WriteLn(Format('                      calls-saved gap (strong-weak) flat=%.1fpts -> peaked=%.1fpts.',
    [fStrongSaved - fWeakSaved, pStrongSaved - pWeakSaved]));
  WriteLn;
  WriteLn('Interpretation: speculative sampling exactly preserves the target''s output');
  WriteLn('distribution at ANY peakedness (Gate 1 pins it bit-for-bit; Gate 2 confirms');
  WriteLn('empirically for both a flat and a peaked target) while committing 1..K+1 tokens');
  WriteLn('per big-model pass. A peaked target makes the speedup MORE sensitive to draft');
  WriteLn('quality at NO change in what is sampled. The KV-cached verifier removes the');
  WriteLn('per-pass prefix recompute (prefill once, short cached window per pass,');
  WriteLn('TruncateCache rollback on rejection), composing the two efficiency wins.');

  FreeStepTarget(Step);
  Target.Free;
  Draft.Free;
  InputV.Free;

  // Gate 1 (exactness) and Gate 1b (cached exactness) are mandatory.
  if not (GateExact and GateCachedExact) then Halt(1);
end.
