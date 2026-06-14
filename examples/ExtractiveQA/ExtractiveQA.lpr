program ExtractiveQA;
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

// ExtractiveQA -- the SQuAD extractive question-answering path through the
// landed neuralpretrained.pas helpers: AnswerSpan (run "[CLS] question
// [SEP] context [SEP]", mask the question/special/pad rows, pick the
// best-scoring span over the CONTEXT tokens, map it back to a substring
// via the tokenizer offset map) and QAReport (SQuAD Exact-Match + token
// F1 over a planted set).
//
// A real distilbert-base-cased-distilled-squad / deepset-roberta-base-
// squad2 checkpoint is ~250 MB - too large to ship. To stay self-contained
// and CPU-fast (<1 s, no download), this demo HAND-WIRES a span head on top
// of the committed WordPiece tokenizer fixture: an Embedding whose row
// value IS each token's logit, then AddQuestionAnsweringHead with the two
// projections set to start = end = hidden. The answer word is therefore the
// highest-"score" context token. This exercises the WHOLE public pipeline
// (tokenization with offsets, context-only masking, span argmax, substring
// recovery, EM/F1 scoring) without any pretrained weights.
//
// To run a REAL checkpoint instead, replace the BuildDemoNet block with:
//   NN := BuildBertForQuestionAnsweringFromSafeTensors(
//           'model.safetensors', {pSeqLen=}384, {pInferenceOnly=}true);
//   Tok.LoadFromFile('tokenizer.json');
// and AnswerSpan / QAReport work unchanged.
//
// Usage:
//   ExtractiveQA [tokenizer.json]
//     tokenizer.json - a BERT-family WordPiece tokenizer
//       (default: tests/fixtures/tiny_wordpiece_tokenizer.json).

{$mode objfpc}{$H+}

uses
  {$IFDEF UNIX}cthreads, cmem,{$ENDIF}
  Classes, SysUtils,
  neuralvolume, neuralnetwork, neuralhftokenizer, neuralpretrained;

const
  csSeqLen = 64;

// Builds the self-contained demo QA net over Tok's vocabulary: every token's
// embedding row holds a per-token "score" (high for a couple of answer-ish
// words, low otherwise); the span head reads that score as both start and
// end logit, so AnswerSpan returns the top-scoring CONTEXT word.
function BuildDemoNet(Tok: TNeuralHFTokenizer): TNNet;
var
  Emb, StartProj, EndProj: TNNetLayer;
  V, i, Id: integer;
begin
  V := Tok.GetVocabSize();
  Result := TNNet.Create();
  Result.AddLayer(TNNetInput.Create(csSeqLen, 1, 2));
  Result.AddLayer(TNNetSplitChannels.Create([0]));
  Emb := Result.AddLayer(TNNetEmbedding.Create(V, 1, {EncodeZero=}1));
  Result.AddQuestionAnsweringHead();
  // The two [hidden -> 1] projections are the layers right before the
  // (SeqLen,1,2) DeepConcat: start = +hidden, end = +hidden, no bias.
  StartProj := Result.Layers[Result.Layers.Count - 3];
  EndProj := Result.Layers[Result.Layers.Count - 2];
  StartProj.Neurons[0].Weights.FData[0] := 1.0;
  StartProj.Neurons[0].BiasWeight := 0;
  EndProj.Neurons[0].Weights.FData[0] := 1.0;
  EndProj.Neurons[0].BiasWeight := 0;
  StartProj.FlushWeightCache();
  EndProj.FlushWeightCache();
  // Default low score for every token; boost a few "salient" answer words.
  for i := 0 to V - 1 do Emb.Neurons[0].Weights.FData[i] := -5.0;
  Id := Tok.TokenToId('fox');   if Id >= 0 then Emb.Neurons[0].Weights.FData[Id] := 9.0;
  Id := Tok.TokenToId('dog');   if Id >= 0 then Emb.Neurons[0].Weights.FData[Id] := 8.0;
  Id := Tok.TokenToId('cat');   if Id >= 0 then Emb.Neurons[0].Weights.FData[Id] := 8.0;
  Id := Tok.TokenToId('brown'); if Id >= 0 then Emb.Neurons[0].Weights.FData[Id] := 6.0;
  Emb.FlushWeightCache();
end;

procedure Ask(NN: TNNet; Tok: TNeuralHFTokenizer;
  const Question, Context: string);
var
  Answer: string;
  SChar, EChar: integer;
  Score, NoAns: TNeuralFloat;
begin
  Answer := AnswerSpan(NN, Tok, Question, Context, SChar, EChar,
    Score, NoAns);
  WriteLn('  Q: ', Question);
  WriteLn('  A: "', Answer, '"  (chars [', SChar, ',', EChar,
    '), score=', FormatFloat('0.000', Score),
    ', null=', FormatFloat('0.000', NoAns), ')');
  WriteLn;
end;

var
  TokPath: string;
  Tok: TNeuralHFTokenizer;
  NN: TNNet;
  Context: string;

begin
  DefaultFormatSettings.DecimalSeparator := '.';
  if ParamCount >= 1 then TokPath := ParamStr(1)
  else TokPath := 'tests' + DirectorySeparator + 'fixtures' +
    DirectorySeparator + 'tiny_wordpiece_tokenizer.json';

  if not FileExists(TokPath) then
  begin
    WriteLn('Tokenizer not found: ', TokPath);
    WriteLn('Pass a BERT-family tokenizer.json as the first argument, or run');
    WriteLn('from the repo root so the default fixture path resolves.');
    Halt(1);
  end;

  Tok := TNeuralHFTokenizer.Create();
  NN := nil;
  try
    Tok.LoadFromFile(TokPath);
    NN := BuildDemoNet(Tok);

    WriteLn('=== Extractive QA (SQuAD span head) ===');
    WriteLn('Demo span head: the answer is the highest-scored CONTEXT word.');
    WriteLn('(Swap in BuildBertForQuestionAnsweringFromSafeTensors for a ',
      'real model.)');
    WriteLn;

    Context := 'the quick brown fox jumps over the lazy dog';
    WriteLn('Context: ', Context);
    WriteLn;
    Ask(NN, Tok, 'what animal jumped', Context);
    Ask(NN, Tok, 'what was lazy', Context);

    // QAReport over a tiny planted set: one right, one wrong gold.
    WriteLn(QAReport(NN, Tok,
      ['what animal jumped', 'what color'],
      [Context, Context],
      ['fox', 'red']));
  finally
    NN.Free;
    Tok.Free;
  end;
end.
