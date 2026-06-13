// ConstrainedDecoding example
//
// Constrained (structured) decoding: a caller-supplied "allowed next tokens"
// hook (TNNetTokenConstraint, neuraldecode) that the streamed generation loop
// applies to the post-softmax probability row AFTER the repetition penalties
// and BEFORE the sampler - zeroing every disallowed token and renormalizing.
//
// The demo is deliberately model-free in spirit: the SAME tiny UNTRAINED
// char-level network is decoded twice -
//   1. FREE       : top-p sampling produces character soup (it is untrained);
//   2. JSON MODE  : the same sampler behind TNNetJSONConstraint, whose
//                   character-level pushdown automaton (TNNetJSONStateMachine)
//                   only ever exposes tokens that keep the output a legal
//                   prefix of a JSON value - so even an untrained model can
//                   ONLY emit syntactically valid JSON, and EOS only becomes
//                   legal once a complete top-level value stands.
// A third pass shows TNNetForcedSequenceConstraint (multiple-choice
// answering): generation is forced down one of the candidate strings
// 'yes' / 'no' / 'maybe', characters the free model would never line up.
//
// To keep the JSON samples SHORT (an untrained uniform model closes a string
// with probability ~1/96 per step), the logit head gets a small hand-set bias
// toward structural JSON characters. That is a stand-in for a trained model's
// preferences, NOT part of the constraint mechanism: correctness comes from
// the automaton alone; the bias only shortens the walk. Runs in seconds on
// CPU.
//
// Coded by Claude (AI).
program ConstrainedDecoding;

{$mode objfpc}{$H+}

uses
  {$IFDEF UNIX}cthreads,{$ENDIF}
  Classes, SysUtils,
  neuralnetwork, neuralvolume, neuraldecode;

const
  csVocab   = 128; // char-level: token id = character code, ids < 2 special
  csDim     = 16;
  csMaxNew  = 120;
  csCacheLen= 160;

// Builds the width-1 char-level next-token net used for streamed decoding.
function BuildCharLM(): TNNet;
begin
  Result := TNNet.Create();
  Result.AddLayer(TNNetInput.Create(1, 1, 1));
  Result.AddLayer(TNNetEmbedding.Create(csVocab, csDim, 0, 0.02));
  Result.AddLayer(TNNetPointwiseConvLinear.Create(csDim));
  Result.AddLayer(TNNetDiagonalSSM.Create());
  Result.AddLayer(TNNetPointwiseConvLinear.Create(csVocab));
  Result.AddLayer(TNNetPointwiseSoftMax.Create());
end;

// Mild structural-character preference on the logit head (see the header
// note: a stand-in for a trained model so samples close quickly; validity
// is guaranteed by the constraint either way).
procedure BiasStructuralChars(Net: TNNet);
const
  Structural = '"{}[]:,0123456789';
var
  Head: TNNetLayer;
  I: integer;
begin
  Head := Net.Layers[Net.Layers.Count - 2]; // the logit conv under softmax
  for I := 1 to Length(Structural) do
    Head.Neurons[Ord(Structural[I])].BiasWeight :=
      Head.Neurons[Ord(Structural[I])].BiasWeight + 1.5;
  // EOS mildly attractive so complete values actually stop.
  Head.Neurons[1].BiasWeight := Head.Neurons[1].BiasWeight + 1.5;
  // No-op multiply so the conv layer's packed weights see the hand edits.
  Head.MulWeights(1.0);
end;

// One streamed generation pass from the prompt 'a'; returns the emitted
// characters (stops printing at the first special id).
function Generate(Net: TNNet; Sampler: TNNetSamplerBase;
  Constraint: TNNetTokenConstraint): string;
var
  Session: TNNetStreamingDecoder;
  Toks: TNeuralIntegerArray;
  OutLen, T: integer;
begin
  Result := '';
  Session := TNNetStreamingDecoder.Create(Net, csCacheLen);
  try
    SetLength(Toks, 1);
    Toks[0] := Ord('a');
    OutLen := GenerateTokensStreamed(Session, Toks, 1, csMaxNew, csCacheLen,
      Sampler, nil, nil, Constraint);
    for T := 1 to OutLen - 1 do
    begin
      if Toks[T] < 2 then break;
      Result := Result + Chr(Toks[T]);
    end;
  finally
    Session.Free;
  end;
end;

// Candidate string -> char-level token-id sequence.
function StrToTokenSeq(const S: string): TNeuralIntegerArray;
var
  I: integer;
begin
  SetLength(Result, Length(S));
  for I := 1 to Length(S) do Result[I - 1] := Ord(S[I]);
end;

var
  Net: TNNet;
  Sampler: TNNetSamplerBase;
  JSON: TNNetJSONConstraint;
  Forced: TNNetForcedSequenceConstraint;
  Checker: TNNetJSONStateMachine;
  Candidates: TNNetTokenSequences;
  S: string;
  Round: integer;
begin
  RandSeed := 20260611;
  Net := BuildCharLM();
  BiasStructuralChars(Net);
  Sampler := TNNetSamplerTopP.Create(0.9);
  JSON := TNNetJSONConstraint.CreateCharLevel(csVocab);
  Checker := TNNetJSONStateMachine.Create();
  SetLength(Candidates, 3);
  Candidates[0] := StrToTokenSeq('yes');
  Candidates[1] := StrToTokenSeq('no');
  Candidates[2] := StrToTokenSeq('maybe');
  Forced := TNNetForcedSequenceConstraint.Create(Candidates);
  try
    WriteLn('=== FREE decoding (untrained net, top-p 0.9) ===');
    for Round := 1 to 2 do
      WriteLn('  sample ', Round, ': "', Generate(Net, Sampler, nil), '"');
    WriteLn;
    WriteLn('=== JSON-constrained decoding (same net, same sampler) ===');
    for Round := 1 to 4 do
    begin
      S := Generate(Net, Sampler, JSON);
      Checker.Reset();
      if not Checker.FeedString(S) then
        raise Exception.Create('BUG: the constraint emitted invalid JSON: ' + S);
      WriteLn('  sample ', Round, ': ', S);
      WriteLn('     -> legal JSON prefix: yes; complete value: ',
        Checker.IsComplete());
    end;
    WriteLn;
    WriteLn('=== Forced-sequence decoding (multiple choice yes/no/maybe) ===');
    S := Generate(Net, Sampler, Forced);
    WriteLn('  the untrained net was forced to answer: "', S, '"');
    WriteLn('  completed one candidate: ', Forced.Completed());
  finally
    Forced.Free;
    Checker.Free;
    JSON.Free;
    Sampler.Free;
    Net.Free;
  end;
end.
