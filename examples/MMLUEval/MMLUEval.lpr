program MMLUEval;
(*
MMLUEval: an end-to-end demo of the MMLU few-shot accuracy harness
(EvaluateMMLU / MMLUReport in neuralnlpmetrics.pas). MMLU (Hendrycks et al.
2021) is the canonical 4-choice (A/B/C/D) knowledge benchmark; the HF
lm-evaluation-harness scores it by the log-probability of the SINGLE answer-
letter token that follows the standard k-shot prompt

  <k same-subject demos, each "Question..\nA. ..\nB. ..\nC. ..\nD. ..\nAnswer: X\n\n">
  Question ..\nA. ..\nB. ..\nC. ..\nD. ..\nAnswer:

NOT by the perplexity of the whole answer string (that latter, full-
continuation pattern is what examples/HellaSwagEval demonstrates via
EvaluateMultipleChoice -- the two scoring modes are kept clearly separate).
This example shows BOTH 0-shot and 5-shot modes; 5-shot is the headline MMLU
setting.

To stay within a tiny CPU / memory budget (no multi-GB checkpoint download,
no network fetch) this demo trains a small CHAR-LEVEL autoregressive next-token
model on a TINY EMBEDDED smoke subset (two toy "subjects", a few questions
each) so it compiles and runs in CI under the 3 GB ulimit in seconds. The goal
is the HARNESS MECHANICS, not a real accuracy number. The scoring path is
checkpoint-agnostic: swap BuildAndTrainToyModel for a BuildLlamaFromSafeTensors
(or any importer) and feed TNeuralHFTokenizer-produced token ids into the same
TNNetMMLUQuestion records and the harness below is unchanged.

To run against the REAL cais/mmlu (a.k.a. hendrycks_test) dataset, dump the
dev (few-shot) and test splits to a small text file with the venv-x datasets
package (/home/bpsa/x/bin/python) and feed the questions through the same
prompt builder; this smoke build hard-codes its questions so it needs no fetch.
A "--full <path>" hook is left as a documented follow-up (see README.md).

Pure CPU, well under a minute, well under the 3 GB ulimit.

Copyright (C) 2026 Joao Paulo Schwarz Schuler

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
any later version.

Coded by Claude (AI).
*)

{$mode objfpc}{$H+}

uses {$IFDEF UNIX} cthreads, {$ENDIF}
  Classes, SysUtils, Math,
  neuralnetwork,
  neuralvolume,
  neuralnlpmetrics;

const
  // Printable-ASCII char vocabulary: ids 0..94 = chr(32..126). id 0 (space)
  // doubles as the pad / boundary symbol. Char-level stand-in for a real
  // subword tokenizer (the harness is tokenizer-agnostic).
  cVocabBase    = 32;       // first printable char
  cVocab        = 95;       // chr(32) .. chr(126)
  cContextLen   = 64;       // model receptive field (chars)
  cEmbedDim     = 24;
  cHidden       = 48;
  cTrainPasses  = 400;
  cLearningRate = 0.05;

type
  // One embedded MMLU smoke question: the stem, the four choices, the gold
  // letter (0..3) and the subject bucket.
  TSmokeQ = record
    Subject: integer;
    Stem: string;
    A, B, C, D: string;
    Gold: integer;
  end;

const
  cSubjectNames: array[0..1] of string = ('toy_animals', 'toy_colors');

  // Dev (few-shot demo) pool, grouped by subject. The five-shot demos for a
  // question are drawn from its OWN subject's dev pool (the MMLU convention).
  cDev: array[0..9] of TSmokeQ = (
    (Subject:0; Stem:'What sound does a lion make';
     A:'roar'; B:'meow'; C:'chirp'; D:'moo'; Gold:0),
    (Subject:0; Stem:'What sound does a cat make';
     A:'roar'; B:'meow'; C:'chirp'; D:'moo'; Gold:1),
    (Subject:0; Stem:'What sound does a bird make';
     A:'roar'; B:'meow'; C:'chirp'; D:'moo'; Gold:2),
    (Subject:0; Stem:'What sound does a cow make';
     A:'roar'; B:'meow'; C:'chirp'; D:'moo'; Gold:3),
    (Subject:0; Stem:'What sound does a dog make';
     A:'roar'; B:'bark'; C:'chirp'; D:'moo'; Gold:1),
    (Subject:1; Stem:'What color is the clear sky';
     A:'red'; B:'blue'; C:'green'; D:'yellow'; Gold:1),
    (Subject:1; Stem:'What color is fresh grass';
     A:'red'; B:'blue'; C:'green'; D:'yellow'; Gold:2),
    (Subject:1; Stem:'What color is a ripe banana';
     A:'red'; B:'blue'; C:'green'; D:'yellow'; Gold:3),
    (Subject:1; Stem:'What color is a ripe tomato';
     A:'red'; B:'blue'; C:'green'; D:'yellow'; Gold:0),
    (Subject:1; Stem:'What color is a clear lake';
     A:'red'; B:'blue'; C:'green'; D:'yellow'; Gold:1)
  );

  // Test questions (held out from training in spirit; the toy model is so
  // small this is purely a mechanics demo).
  cTest: array[0..3] of TSmokeQ = (
    (Subject:0; Stem:'What sound does a lion make';
     A:'roar'; B:'meow'; C:'chirp'; D:'moo'; Gold:0),
    (Subject:0; Stem:'What sound does a bird make';
     A:'roar'; B:'meow'; C:'chirp'; D:'moo'; Gold:2),
    (Subject:1; Stem:'What color is the clear sky';
     A:'red'; B:'blue'; C:'green'; D:'yellow'; Gold:1),
    (Subject:1; Stem:'What color is fresh grass';
     A:'red'; B:'blue'; C:'green'; D:'yellow'; Gold:2)
  );

// --- char <-> token id ----------------------------------------------------

function CharToId(C: char): integer;
begin
  if (Ord(C) >= cVocabBase) and (Ord(C) < cVocabBase + cVocab) then
    Result := Ord(C) - cVocabBase
  else
    Result := 0; // map anything else (incl. newline) to the space/boundary id
end;

function Encode(const S: string): TNeuralIntegerArray;
var
  I: integer;
begin
  SetLength(Result, Length(S));
  for I := 1 to Length(S) do
    Result[I - 1] := CharToId(S[I]);
end;

// --- MMLU prompt builder (the standard lm-eval format) ---------------------

// One question's choice block + "Answer: " (with the trailing space, so the
// answer-letter token is the very next token both for training, for the demo
// completion, and for EvaluateMMLU's single-letter scoring - all consistent).
function FormatQuestion(const Q: TSmokeQ): string;
begin
  Result :=
    'Question: ' + Q.Stem + LineEnding +
    'A. ' + Q.A + LineEnding +
    'B. ' + Q.B + LineEnding +
    'C. ' + Q.C + LineEnding +
    'D. ' + Q.D + LineEnding +
    'Answer: ';
end;

// A completed demo: the question block + "<letter>" + blank line.
function FormatDemo(const Q: TSmokeQ): string;
const
  Letters: array[0..3] of char = ('A', 'B', 'C', 'D');
begin
  Result := FormatQuestion(Q) + Letters[Q.Gold] + LineEnding + LineEnding;
end;

// Builds the full k-shot prompt for question Q: ShotsK same-subject demos
// (taken in order from cDev, skipping Q itself if present) then Q's block.
function BuildPrompt(const Q: TSmokeQ; ShotsK: integer): string;
var
  D, Used: integer;
begin
  Result := '';
  Used := 0;
  D := 0;
  while (Used < ShotsK) and (D <= High(cDev)) do
  begin
    if (cDev[D].Subject = Q.Subject) and
       not ((cDev[D].Stem = Q.Stem) and (cDev[D].Gold = Q.Gold)) then
    begin
      Result := Result + FormatDemo(cDev[D]);
      Inc(Used);
    end;
    Inc(D);
  end;
  Result := Result + FormatQuestion(Q);
end;

// --- model ----------------------------------------------------------------

procedure BuildModel(out NN: TNNet);
begin
  NN := TNNet.Create();
  NN.AddLayer(TNNetInput.Create(cContextLen, 1, 1));
  NN.AddLayer(TNNetEmbedding.Create(cVocab, cEmbedDim));
  NN.AddLayer(TNNetPointwiseConvReLU.Create(cHidden));
  // Collapse the sequence to a single next-token distribution over the vocab
  // (single next-token head -> ScoreSequence/ScoreCompletion use the right-
  // aligned-prefix path, exactly like an autoregressive LM scored teacher-
  // forced; EvaluateMMLU scores the single answer-letter token through it).
  NN.AddLayer(TNNetFullConnectReLU.Create(cHidden));
  NN.AddLayer(TNNetFullConnectLinear.Create(cVocab));
  NN.AddLayer(TNNetSoftMax.Create());
  NN.SetLearningRate(cLearningRate, 0.9);
end;

// Train next-char SGD over the completed dev demos (prompt + gold letter). The
// input encoding MUST match neuralnlpmetrics' single-next-token-head path: the
// window is zero-filled and the prefix is laid in REVERSED order (most-recent
// char at index 0) via CopyReversedNoChecksIntArr. We replicate that exactly so
// train and eval see identically-encoded inputs.
procedure TrainModel(NN: TNNet);
var
  Input, Target: TNNetVolume;
  Pass, I, T: integer;
  Toks, Prefix: TNeuralIntegerArray;
  Sample: string;
begin
  Input := TNNetVolume.Create(cContextLen, 1, 1);
  Target := TNNetVolume.Create(NN.GetLastLayer().Output);
  try
    // We only need the head to learn the SINGLE answer-letter prediction that
    // EvaluateMMLU scores: predict the gold letter from the question prompt
    // (everything up to and including "Answer: "). Training only that final
    // position (instead of every char) keeps the smoke run fast while using the
    // IDENTICAL reversed-prefix encoding the harness uses at eval time.
    for Pass := 1 to cTrainPasses do
      for I := 0 to High(cDev) do
      begin
        Sample := FormatQuestion(cDev[I]); // prompt up to and incl "Answer: "
        Toks := Encode(Sample);
        T := Length(Toks); // predict the letter that comes right after Sample
        Prefix := Copy(Toks, Max(0, T - cContextLen), Min(T, cContextLen));
        Input.Fill(0);
        Input.CopyReversedNoChecksIntArr(Prefix);
        Target.Fill(0);
        Target.FData[CharToId(chr(Ord('A') + cDev[I].Gold))] := 1.0;
        NN.Compute(Input);
        NN.Backpropagate(Target);
      end;
  finally
    Target.Free;
    Input.Free;
  end;
end;

// --- build MMLU questions for the harness ----------------------------------

procedure BuildQuestions(ShotsK: integer;
  out Questions: array of TNNetMMLUQuestion);
var
  I: integer;
begin
  for I := 0 to High(cTest) do
  begin
    Questions[I].PromptTokens := Encode(BuildPrompt(cTest[I], ShotsK));
    Questions[I].GoldLetter := cTest[I].Gold;
    Questions[I].SubjectIndex := cTest[I].Subject;
  end;
end;

procedure RunMode(NN: TNNet; ShotsK: integer; const LetterTokens: array of integer);
var
  Questions: array[0..High(cTest)] of TNNetMMLUQuestion;
  Stats: TNNetMMLUStats;
begin
  BuildQuestions(ShotsK, Questions);
  // LastWindow=true: real MMLU k-shot prompts routinely exceed any fixed
  // context window, so over-context prompts are scored over the model's
  // trailing context window (the standard sliding-window LM eval) instead of
  // raising. The trailing window always contains the "Answer: " region that
  // determines the letter, and training clipped its prefix to the same window.
  Stats := EvaluateMMLU(NN, Questions, LetterTokens, Length(cSubjectNames),
    {LastWindow=}true);
  WriteLn(MMLUReport(Stats, cSubjectNames, ShotsK));
end;

var
  NN: TNNet;
  LetterTokens: array[0..3] of integer;
begin
  WriteLn('MMLUEval: MMLU few-shot accuracy harness (EvaluateMMLU / MMLUReport)');
  WriteLn('Single-token A/B/C/D answer-letter scoring (HF lm-eval convention),');
  WriteLn('distinct from HellaSwagEval''s full-continuation EvaluateMultipleChoice.');
  WriteLn('Tiny embedded smoke subset; the scoring path is checkpoint-agnostic.');
  WriteLn;

  RandSeed := 1234;
  BuildModel(NN);
  try
    WriteLn('Architecture:');
    NN.PrintSummary();
    WriteLn;
    WriteLn('Training char-level toy model on ', Length(cDev),
      ' MMLU-style demos (', cTrainPasses, ' passes) ...');
    TrainModel(NN);
    WriteLn('Done.');
    WriteLn;

    // The four answer-letter tokens: the char ids of 'A','B','C','D' (each a
    // single token in this char vocabulary). For a real subword tokenizer the
    // caller would encode " A".." D" and pass the LAST id of each.
    LetterTokens[0] := CharToId('A');
    LetterTokens[1] := CharToId('B');
    LetterTokens[2] := CharToId('C');
    LetterTokens[3] := CharToId('D');

    RunMode(NN, 0, LetterTokens);  // zero-shot
    WriteLn;
    RunMode(NN, 5, LetterTokens);  // five-shot (the headline MMLU setting)
  finally
    NN.Free;
  end;
end.
