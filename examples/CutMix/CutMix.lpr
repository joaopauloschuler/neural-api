program CutMix;
(*
CutMix: demonstrates the CutMix data-augmentation helper
(CreateCutMixVolumePairList) on a tiny synthetic image classification toy.
Each class is a solid-color 8x8 patch plus noise; CutMix pastes a random
rectangle of one image into another and mixes the targets by the pasted-area
fraction. The program trains the same small convolutional classifier WITH and
WITHOUT CutMix and prints a short comparison of final validation accuracy.

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

const
  cImgSize = 8;
  cClasses = 2;

  // Each class is a distinct solid base intensity over an 8x8x1 image plus
  // light per-pixel noise, so a spatial CutMix box is meaningful.
  function CreateToyPairList(MaxCnt: integer): TNNetVolumePairList;
  var
    Cnt, Cls, X, Y: integer;
    Base: TNeuralFloat;
    Img, Tgt: TNNetVolume;
  begin
    Result := TNNetVolumePairList.Create();
    for Cnt := 1 to MaxCnt do
    begin
      Cls := Random(cClasses);
      if Cls = 0 then Base := 0.2 else Base := 0.8;
      Img := TNNetVolume.Create(cImgSize, cImgSize, 1);
      for X := 0 to cImgSize - 1 do
        for Y := 0 to cImgSize - 1 do
          Img[X, Y, 0] := Base + (Random() - 0.5) * 0.2;
      Tgt := TNNetVolume.Create(cClasses);
      Tgt.SetClassForSoftMax(Cls);
      Result.Add(TNNetVolumePair.Create(Img, Tgt));
    end;
  end;

  function ArgMax(V: TNNetVolume): integer;
  var I: integer;
  begin
    Result := 0;
    for I := 1 to V.Size - 1 do
      if V.Raw[I] > V.Raw[Result] then Result := I;
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
      if ArgMax(Output) = ArgMax(Pairs[Cnt].O) then Inc(Hits);
    end;
    Output.Free;
    Result := Hits / Pairs.Count;
  end;

  function BuildNet(): TNNet;
  begin
    Result := TNNet.Create();
    Result.AddLayer([
      TNNetInput.Create(cImgSize, cImgSize, 1),
      TNNetConvolutionReLU.Create({features=}8, {featuresize=}3, {padding=}1, {stride=}1),
      TNNetMaxPool.Create(2),
      TNNetFullConnectReLU.Create(16),
      TNNetFullConnectLinear.Create(cClasses),
      TNNetSoftMax.Create()
    ]);
  end;

  function TrainAndEval(TrainPairs, ValPairs: TNNetVolumePairList): TNeuralFloat;
  const
    cEpochs = 20;
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
    BaseTrain, CutTrain, ValPairs: TNNetVolumePairList;
    AccPlain, AccCutMix: TNeuralFloat;
  begin
    RandSeed := 1234;
    BaseTrain := CreateToyPairList(300);
    ValPairs  := CreateToyPairList(150);

    // CutMix-augmented training set: paste a random rectangle of a permuted
    // partner image and mix the targets by the pasted-area fraction.
    // lambda ~ Beta(0.4, 0.4).
    CutTrain := CreateCutMixVolumePairList(BaseTrain, {alpha=}0.4);

    WriteLn('CutMix data augmentation demo (synthetic 2-class image toy)');
    WriteLn('Training WITHOUT cutmix...');
    AccPlain := TrainAndEval(BaseTrain, ValPairs);
    WriteLn('Training WITH cutmix...');
    AccCutMix := TrainAndEval(CutTrain, ValPairs);

    WriteLn('');
    WriteLn('=== Validation accuracy comparison ===');
    WriteLn('  No cutmix : ', (AccPlain*100):6:2, ' %');
    WriteLn('  CutMix    : ', (AccCutMix*100):6:2, ' %');

    CutTrain.Free;
    ValPairs.Free;
    BaseTrain.Free;
  end;

var
  // Stops Lazarus errors
  Application: record Title:string; end;

begin
  Application.Title:='CutMix Example';
  RunAlgo();
end.
