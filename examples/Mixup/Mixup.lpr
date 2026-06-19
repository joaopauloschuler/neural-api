program Mixup;
(*
Mixup: demonstrates the Mixup data-augmentation helper
(CreateMixedVolumePairList) on a tiny synthetic 2-class toy problem.
It trains the same small classifier WITH and WITHOUT mixup and prints
a short comparison of final validation accuracy.

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
  Classes, SysUtils,
  neuralnetwork,
  neuralvolume;

  // Two interleaved Gaussian blobs in 2D, one-hot 2-class targets.
  function CreateToyPairList(MaxCnt: integer): TNNetVolumePairList;
  var
    Cnt, Cls: integer;
    cx, cy, x, y: TNeuralFloat;
  begin
    Result := TNNetVolumePairList.Create();
    for Cnt := 1 to MaxCnt do
    begin
      Cls := Random(2);
      if Cls = 0 then begin cx := -1.0; cy := -1.0; end
                 else begin cx :=  1.0; cy :=  1.0; end;
      x := cx + (Random()-0.5);
      y := cy + (Random()-0.5);
      if Cls = 0
      then Result.Add(TNNetVolumePair.Create(
             TNNetVolume.Create([x, y]), TNNetVolume.Create([1.0, 0.0])))
      else Result.Add(TNNetVolumePair.Create(
             TNNetVolume.Create([x, y]), TNNetVolume.Create([0.0, 1.0])));
    end;
  end;

  // Manual accuracy over a pair list (argmax match).
  function Accuracy(NN: TNNet; Pairs: TNNetVolumePairList): TNeuralFloat;
  var
    Cnt, Hits: integer;
    Output: TNNetVolume;
  begin
    Hits := 0;
    Output := TNNetVolume.Create();
    for Cnt := 0 to Pairs.Count - 1 do
    begin
      NN.Compute(Pairs[Cnt].I);
      NN.GetOutput(Output);
      if Output.GetClass() = Pairs[Cnt].O.GetClass() then Inc(Hits);
    end;
    Output.Free;
    Result := Hits / Pairs.Count;
  end;

  function BuildNet(): TNNet;
  begin
    Result := TNNet.Create();
    Result.AddLayer([
      TNNetInput.Create(2),
      TNNetFullConnectReLU.Create(16),
      TNNetFullConnectLinear.Create(2),
      TNNetSoftMax.Create()
    ]);
  end;

  function TrainAndEval(TrainPairs, ValPairs: TNNetVolumePairList): TNeuralFloat;
  const
    cEpochs = 40;
  var
    NN: TNNet;
    Epoch, I, Idx: integer;
    Pair: TNNetVolumePair;
  begin
    NN := BuildNet();
    NN.SetLearningRate(0.01, 0.9);
    for Epoch := 1 to cEpochs do
      for I := 0 to TrainPairs.Count - 1 do
      begin
        // Sample with replacement (online SGD).
        Idx := Random(TrainPairs.Count);
        Pair := TrainPairs[Idx];
        NN.Compute(Pair.I);
        NN.Backpropagate(Pair.O);
      end;
    Result := Accuracy(NN, ValPairs);
    NN.Free;
  end;

  procedure RunAlgo();
  var
    BaseTrain, MixedTrain, ValPairs: TNNetVolumePairList;
    AccPlain, AccMixup: TNeuralFloat;
  begin
    RandSeed := 1234;
    BaseTrain := CreateToyPairList(400);
    ValPairs  := CreateToyPairList(200);

    // Mixup-augmented training set: convex combinations with lambda ~ Beta(0.4,0.4).
    MixedTrain := CreateMixedVolumePairList(BaseTrain, {alpha=}0.4);

    WriteLn('Mixup data augmentation demo (synthetic 2-class toy)');
    WriteLn('Training WITHOUT mixup...');
    AccPlain := TrainAndEval(BaseTrain, ValPairs);
    WriteLn('Training WITH mixup...');
    AccMixup := TrainAndEval(MixedTrain, ValPairs);

    WriteLn('');
    WriteLn('=== Validation accuracy comparison ===');
    WriteLn('  No mixup : ', (AccPlain*100):6:2, ' %');
    WriteLn('  Mixup    : ', (AccMixup*100):6:2, ' %');

    MixedTrain.Free;
    ValPairs.Free;
    BaseTrain.Free;
  end;

var
  // Stops Lazarus errors
  Application: record Title:string; end;

begin
  Application.Title:='Mixup Example';
  RunAlgo();
end.
