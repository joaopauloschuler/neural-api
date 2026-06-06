program QuickStartSequence;
(*
QuickStartSequence: the shortest possible "it learns a sequence" demo.

A tiny char-level next-token model learns a fixed counting sequence
(0,1,2,...,9,0,1,2,...). Given the last two symbols it predicts the next
one. After a couple of seconds of CPU training it can continue the count
indefinitely from a seed -- a "hello world" for sequence learning, far
simpler than the transformer in examples/SimpleNLP.

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

{$mode objfpc}{$H+}

uses {$IFDEF UNIX} cthreads, {$ENDIF}
  SysUtils,
  neuralnetwork,
  neuralvolume;

const
  // The fixed counting sequence the model has to learn to continue.
  // One cycle of the symbols 0..9 then wrap back to 0 -- a "counting" task.
  cSequence = '0123456789';
  cContext  = 2;   // how many previous symbols the model sees
  cVocabReal = 10; // the ten digits 0..9

// Maps a digit character to its integer index (0..9).
function CharToIndex(c: char): integer;
begin
  Result := Ord(c) - Ord('0');
  if (Result < 0) or (Result > 9) then Result := 0;
end;

function IndexToChar(i: integer): char;
begin
  if (i >= 0) and (i <= 9) then Result := Chr(Ord('0') + i) else Result := '?';
end;

// Fills a context window (cContext one-hot chars) into volume V.
procedure EncodeContext(const Ctx: string; V: TNNetVolume);
var i, idx, base: integer;
begin
  V.Fill(0);
  for i := 1 to cContext do
  begin
    idx := CharToIndex(Ctx[i]);
    base := (i - 1) * cVocabReal;
    V.FData[base + idx] := 1;
  end;
end;

procedure RunAlgo();
const
  cEpochs = 200;
  cLearningRate = 0.05;
var
  NN: TNNet;
  TrainingPairs: TNNetVolumePairList;
  Corpus: string;
  P, Epoch, Cnt, Predicted, Correct: integer;
  Ctx: string;
  InVol: TNNetVolume;
  OutVol: TNNetVolume;
  Generated: string;
  StartTime: TDateTime;
  SumLoss: TNeuralFloat;
begin
  // Build a corpus by repeating the sequence a handful of times so every
  // (context -> next char) transition appears many times.
  Corpus := '';
  for Cnt := 1 to 30 do Corpus := Corpus + cSequence;

  // Build the (input, target) training pairs.
  TrainingPairs := TNNetVolumePairList.Create();
  for P := cContext + 1 to Length(Corpus) do
  begin
    Ctx := Copy(Corpus, P - cContext, cContext);
    InVol := TNNetVolume.Create(cContext * cVocabReal, 1, 1, 0);
    EncodeContext(Ctx, InVol);
    OutVol := TNNetVolume.Create(cVocabReal, 1, 1, 0);
    OutVol.Fill(0);
    OutVol.FData[CharToIndex(Corpus[P])] := 1; // one-hot next char
    TrainingPairs.Add(TNNetVolumePair.Create(InVol, OutVol));
  end;

  // A tiny network: one hidden layer + softmax classifier over the vocab.
  NN := TNNet.Create();
  NN.AddLayer( TNNetInput.Create(cContext * cVocabReal) );
  NN.AddLayer( TNNetFullConnectReLU.Create(16) );
  NN.AddLayer( TNNetFullConnectLinear.Create(cVocabReal) );
  NN.AddLayer( TNNetSoftMax.Create() );

  WriteLn('Training a tiny next-character model on: "', cSequence, '"');
  WriteLn('Training pairs: ', TrainingPairs.Count, '   Network layers: ', NN.CountLayers());
  WriteLn;

  // Plain stochastic-gradient training loop. NN.Backpropagate applies the
  // weight update for us, so each step is just Compute then Backpropagate.
  NN.SetLearningRate(cLearningRate, {inertia=}0.9);
  StartTime := Now;
  for Epoch := 1 to cEpochs do
  begin
    SumLoss := 0;
    for P := 0 to TrainingPairs.Count - 1 do
    begin
      NN.Compute(TrainingPairs[P].I);
      SumLoss := SumLoss +
        NN.GetLastLayer().Output.SumDiff(TrainingPairs[P].O);
      NN.Backpropagate(TrainingPairs[P].O);
    end;
    if (Epoch = 1) or (Epoch mod 40 = 0) or (Epoch = cEpochs) then
      WriteLn('  epoch ', Epoch:3, ' / ', cEpochs,
              '   avg |error| = ',
              FormatFloat('0.0000', SumLoss / TrainingPairs.Count));
  end;

  WriteLn;
  WriteLn('Training finished in ',
          FormatFloat('0.0', (Now - StartTime) * 24 * 60 * 60), ' s.');
  WriteLn;

  // Measure next-char accuracy over the corpus.
  InVol := TNNetVolume.Create(cContext * cVocabReal, 1, 1, 0);
  Correct := 0;
  Cnt := 0;
  for P := cContext + 1 to Length(Corpus) do
  begin
    Ctx := Copy(Corpus, P - cContext, cContext);
    EncodeContext(Ctx, InVol);
    NN.Compute(InVol);
    Predicted := NN.GetLastLayer().Output.GetClass();
    if Predicted = CharToIndex(Corpus[P]) then Inc(Correct);
    Inc(Cnt);
  end;
  WriteLn('Next-character accuracy: ', Correct, '/', Cnt,
          '  (', FormatFloat('0.0', 100 * Correct / Cnt), '%)');
  WriteLn;

  // Demonstration: seed with the first two characters, then let the model
  // GENERATE the rest by feeding its own predictions back in.
  Generated := Copy(cSequence, 1, cContext);
  for Cnt := 1 to 3 * Length(cSequence) - cContext do
  begin
    Ctx := Copy(Generated, Length(Generated) - cContext + 1, cContext);
    EncodeContext(Ctx, InVol);
    NN.Compute(InVol);
    Predicted := NN.GetLastLayer().Output.GetClass();
    Generated := Generated + IndexToChar(Predicted);
  end;

  WriteLn('Seed:      "', Copy(cSequence, 1, cContext), '"');
  WriteLn('Generated: "', Generated, '"');
  WriteLn('Expected:  "', cSequence + cSequence + cSequence, '"');
  WriteLn;
  if Generated = cSequence + cSequence + cSequence then
    WriteLn('SUCCESS: the model learned to continue the counting sequence.')
  else
    WriteLn('Note: pattern not fully reproduced (try more epochs).');

  InVol.Free;
  TrainingPairs.Free;
  NN.Free;
end;

begin
  RunAlgo();
end.
