program EarlyExitSelfSpeculative;
(*
EarlyExitSelfSpeculative: single-model self-speculative decoding in the
LayerSkip / CALM style. The SAME trained network is its OWN draft model - there
is NO second checkpoint and NO separate prediction head:

  * a DRAFT token is read from an INTERMEDIATE exit layer's logits, passed
    through the model's OWN LM head via the frozen-body LogitLens splice (copy
    the intermediate activation into the head-input slot, recompute ONLY the
    head sub-stack);
  * that draft is VERIFIED against the full-depth argmax (exact-greedy verify,
    exactly like examples/SpeculativeDecoding does for a SEPARATE draft net);
  * the EMITTED token is ALWAYS the full-depth argmax, so the accepted sequence
    is BIT-IDENTICAL to plain greedy decoding - the early exit only changes how
    much tail-layer work a cached decoder could SKIP, never the output.

This differs from the two existing examples:
  - examples/SelfSpeculativeDecoding drafts from MTP prediction HEADS;
  - examples/EarlyExitNetwork is a BranchyNet CLASSIFICATION demo.
Here the draft is the model's own INTERMEDIATE-LAYER readout.

What this program does in ONE bounded run (pure CPU, well under 5 min):
  1. builds a small constant-width char-level LM and trains it briefly on a
     deterministic repetitive corpus so the early-exit readout becomes a
     USEFUL draft (high acceptance);
  2. decodes a fixed prompt with plain greedy AND with
     DecodeEarlyExitSelfSpeculative, asserts the outputs are IDENTICAL;
  3. prints the early-exit accept/reject counters + acceptance rate, and a
     tokens/sec figure for both paths at MATCHED output.

The tokens/sec headline: because v1 has no cached tail-skip yet, the
self-speculative path here does the SAME full forward PLUS a head-only splice,
so it is intentionally a touch SLOWER per token - the measured ACCEPTANCE RATE
is the real speed signal: in a cached decoder each accepted token would let the
verifier reuse the draft's already-computed prefix and skip the tail layers,
turning the acceptance rate directly into saved compute. The per-token-adaptive
exit + cached tail-skip is the open follow-up tracked in tasklist.md.

Coded by Claude (AI).

Copyright (C) 2026 Joao Paulo Schwarz Schuler
*)
{$mode objfpc}{$H+}

uses {$IFDEF UNIX} cthreads, {$ENDIF}
  Classes, SysUtils, DateUtils,
  neuralnetwork, neuralvolume, neuralfit, neuraldecode;

const
  cContextLen = 32;     // window must hold prompt + full generation (the
                        // char-level decode passes the WHOLE growing context)
  cVocab      = 128;    // ASCII range is enough for the toy corpus
  cWidth      = 48;     // constant body width -> lens-compatible layers
  cExitLayer  = 2;      // intermediate exit layer index
  cDecodeLen  = 16;     // generated tokens for the demo continuation
  cTimedTokens = 80;    // bounded timing run (5 x 16 tokens)

// A deterministic, highly-predictable CYCLIC corpus: next char is a pure
// function of the current char (a -> b -> ... -> j -> a). A tiny readout learns
// this almost perfectly, so the intermediate-layer exit becomes a CONFIDENT and
// ACCURATE draft -> high acceptance. Keeps the run tiny.
function BuildCorpus(): string;
var
  I: integer;
  Pattern: string;
begin
  Pattern := 'abcdefghij';   // 10-char cycle
  Result := '';
  for I := 1 to 400 do Result := Result + Pattern;
end;

// Constant-width char-level LM: embedding -> N constant-width FC ReLU body
// layers -> FC linear LM head -> softmax. The body width equals the head input
// width so every body layer is lens/splice compatible.
function BuildLM(): TNNet;
begin
  Result := TNNet.Create();
  Result.AddLayer(TNNetInput.Create(cContextLen, 1, cVocab));
  Result.AddLayer(TNNetFullConnectReLU.Create(cWidth));   // layer 1
  Result.AddLayer(TNNetFullConnectReLU.Create(cWidth));   // layer 2 (exit)
  Result.AddLayer(TNNetFullConnectReLU.Create(cWidth));   // layer 3
  Result.AddLayer(TNNetFullConnectLinear.Create(cVocab)); // layer 4 (LM head)
  Result.AddLayer(TNNetSoftMax.Create());
  Result.InitWeights();
end;

// Builds (context -> next-char) training pairs from the corpus, char codes
// clamped into [0, cVocab). Reversed one-hot context, matching the decode
// convention (OneHotEncodingReversed).
function BuildTrainingPairs(const Corpus: string): TNNetVolumePairList;
var
  Pos, J, Code, Target: integer;
  InV, OutV: TNNetVolume;
  Window: string;
begin
  Result := TNNetVolumePairList.Create();
  for Pos := cContextLen + 1 to Length(Corpus) - 1 do
  begin
    Window := Copy(Corpus, Pos - cContextLen, cContextLen);
    Target := Ord(Corpus[Pos]) mod cVocab;
    InV := TNNetVolume.Create(cContextLen, 1, cVocab);
    InV.Fill(0);
    // OneHotEncodingReversed: last char of Window at position 0.
    for J := 0 to cContextLen - 1 do
    begin
      Code := Ord(Window[cContextLen - J]) mod cVocab;
      InV[J, 0, Code] := 1;
    end;
    OutV := TNNetVolume.Create(1, 1, cVocab);
    OutV.Fill(0);
    OutV[0, 0, Target] := 1;
    Result.Add(TNNetVolumePair.Create(InV, OutV));
  end;
end;

var
  NN: TNNet;
  Corpus, Prompt: string;
  TrainPairs: TNNetVolumePairList;
  Fit: TNeuralFit;
  Greedy, Spec: TNNetDecodeResult;
  Stats: TNNetEarlyExitStats;
  NoStops: array of string;
  T0: TDateTime;
  GreedyMs, SpecMs: double;
  GreedyTps, SpecTps: double;
  Iter: integer;
begin
  Randomize;
  RandSeed := 424242;
  SetLength(NoStops, 0);

  WriteLn('Early-exit / self-speculative (LayerSkip/CALM) single-model decode');
  WriteLn('================================================================');

  Corpus := BuildCorpus();
  NN := BuildLM();

  TrainPairs := BuildTrainingPairs(Corpus);
  WriteLn(Format('Corpus chars: %d   training pairs: %d',
    [Length(Corpus), TrainPairs.Count]));

  // Brief training so the intermediate-layer readout is a useful draft.
  Fit := TNeuralFit.Create();
  try
    Fit.InitialLearningRate := 0.001;
    Fit.LearningRateDecay := 0.0;
    Fit.L2Decay := 0.0;
    Fit.Verbose := False;
    Fit.EnableClassComparison();
    Fit.Fit(NN, TrainPairs, nil, nil, {batchsize=}32, {epochs=}40);
  finally
    Fit.Free;
  end;
  WriteLn('Training done.');

  Prompt := 'abc';

  // (A) Plain greedy full-depth decode.
  Greedy := DecodeGreedy(NN, Prompt, cDecodeLen);

  // (B) Self-speculative early-exit decode. Confidence=0 -> the early exit is
  //     consulted EVERY step, so acceptance measures how often the
  //     intermediate-layer argmax already agrees with full depth. The OUTPUT is
  //     bit-identical to greedy regardless of the gate.
  Spec := DecodeEarlyExitSelfSpeculative(NN, Prompt, cDecodeLen, Stats, NoStops,
    cExitLayer, {Confidence=}0.0);

  WriteLn;
  WriteLn('Prompt              : "', Prompt, '"');
  WriteLn('Greedy continuation : "', Greedy.Text, '"');
  WriteLn('Self-spec continuation: "', Spec.Text, '"');
  WriteLn;

  if Greedy.Text = Spec.Text then
    WriteLn('CORRECTNESS CHECK: PASS - self-speculative output is BIT-IDENTICAL',
      ' to plain greedy.')
  else
  begin
    WriteLn('CORRECTNESS CHECK: FAIL - outputs differ! (this must never happen)');
    Halt(1);
  end;

  WriteLn(Format('Early-exit stats : steps=%d  drafts=%d  accepted=%d' +
    '  rejected=%d  acceptance=%.1f%%',
    [Stats.Steps, Stats.DraftProposals, Stats.Accepted, Stats.Rejected,
     100.0 * Stats.AcceptanceRate]));
  WriteLn(Format('  => in a CACHED decoder, ~%.0f%% of tokens would skip the' +
    ' tail layers (exit layer %d of %d body layers).',
    [100.0 * Stats.AcceptanceRate, cExitLayer, 3]));

  // ---- Bounded timing: same number of decoded tokens, both paths. ----
  WriteLn;
  WriteLn(Format('Timing %d tokens per path...', [cTimedTokens]));

  T0 := Now();
  for Iter := 1 to 5 do
    Greedy := DecodeGreedy(NN, Prompt, cTimedTokens div 5);
  GreedyMs := MilliSecondsBetween(Now(), T0);

  T0 := Now();
  for Iter := 1 to 5 do
    Spec := DecodeEarlyExitSelfSpeculative(NN, Prompt, cTimedTokens div 5,
      Stats, NoStops, cExitLayer, 0.0);
  SpecMs := MilliSecondsBetween(Now(), T0);

  if GreedyMs <= 0 then GreedyMs := 1;
  if SpecMs <= 0 then SpecMs := 1;
  GreedyTps := cTimedTokens / (GreedyMs / 1000.0);
  SpecTps := cTimedTokens / (SpecMs / 1000.0);

  WriteLn(Format('  plain greedy     : %.1f tokens/sec  (%.0f ms)',
    [GreedyTps, GreedyMs]));
  WriteLn(Format('  self-speculative : %.1f tokens/sec  (%.0f ms)',
    [SpecTps, SpecMs]));
  WriteLn(Format('  acceptance rate this run: %.1f%%',
    [100.0 * Stats.AcceptanceRate]));
  WriteLn;
  WriteLn('NOTE: v1 has no cached tail-skip, so the self-speculative path here');
  WriteLn('does the full forward PLUS a head-only splice and is a touch slower');
  WriteLn('per token. The headline is the ACCEPTANCE RATE: with the cached');
  WriteLn('tail-skip (open follow-up) each accepted token skips the tail');
  WriteLn('layers, so acceptance rate translates directly into speedup at');
  WriteLn('bit-identical output.');

  TrainPairs.Free;
  NN.Free;
end.
