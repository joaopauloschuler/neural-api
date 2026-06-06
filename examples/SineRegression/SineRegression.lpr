program SineRegression;
(*
SineRegression: smallest possible "does the library still train?" demo.

Learns the function y = sin(pi*x) + small Gaussian noise on
256 random samples drawn from x in [-1, 1] with a two-layer MLP:

  Input(1) -> FullConnectReLU(32) -> FullConnectLinear(1)

Prints MSE on a clean (no-noise) test grid every few epochs and a final
side-by-side table of predicted vs ground-truth at 11 evenly-spaced
points. Designed to finish well under a minute on a single CPU.

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

uses {$IFDEF UNIX} cthreads, {$ENDIF}
  Classes, SysUtils, Math,
  neuralnetwork,
  neuralvolume,
  neuralfit;

const
  NumTrain = 256;
  NumEpochs = 300;
  BatchSize = 16;
  LearningRate = 0.002;
  NoiseStdDev = 0.05;

  function CreateSinePairList(MaxCnt: integer; NoiseStd: TNeuralFloat):
    TNNetVolumePairList;
  var
    Cnt: integer;
    X, Y: TNeuralFloat;
  begin
    Result := TNNetVolumePairList.Create();
    for Cnt := 1 to MaxCnt do
    begin
      X := 2.0 * Random - 1.0;  // x in [-1, 1)
      Y := Sin(Pi * X) + NoiseStd * RandG(0, 1);
      Result.Add(
        TNNetVolumePair.Create(
          TNNetVolume.Create([X]),
          TNNetVolume.Create([Y])
        )
      );
    end;
  end;

  // Computes MSE on a clean sin(2*pi*x) test grid of 100 points.
  function CleanGridMSE(NN: TNNet): TNeuralFloat;
  var
    I: integer;
    X, Target, Pred, Diff, Sum: TNeuralFloat;
    Input, Output: TNNetVolume;
  begin
    Input := TNNetVolume.Create([TNeuralFloat(0)]);
    Output := TNNetVolume.Create(1, 1, 1);
    Sum := 0;
    for I := 0 to 99 do
    begin
      X := -1.0 + 2.0 * I / 99.0;
      Target := Sin(Pi * X);
      Input.FData[0] := X;
      NN.Compute(Input);
      NN.GetOutput(Output);
      Pred := Output.Raw[0];
      Diff := Pred - Target;
      Sum := Sum + Diff * Diff;
    end;
    Input.Free;
    Output.Free;
    Result := Sum / 100.0;
  end;

  // Fisher-Yates shuffle of an integer index array.
  procedure ShuffleIndices(var Idx: array of integer);
  var
    I, J, Tmp: integer;
  begin
    for I := High(Idx) downto 1 do
    begin
      J := Random(I + 1);
      Tmp := Idx[I];
      Idx[I] := Idx[J];
      Idx[J] := Tmp;
    end;
  end;

  procedure RunAlgo();
  var
    NN: TNNet;
    TrainingPairs: TNNetVolumePairList;
    Epoch, Step, I: integer;
    Pair: TNNetVolumePair;
    Output: TNNetVolume;
    X, Target, Pred: TNeuralFloat;
    Input: TNNetVolume;
    Order: array of integer;
  begin
    RandSeed := 42;
    NN := TNNet.Create();
    NN.AddLayer([
      TNNetInput.Create(1),
      TNNetFullConnectReLU.Create(64),
      TNNetFullConnectLinear.Create(1)
    ]);

    TrainingPairs := CreateSinePairList(NumTrain, NoiseStdDev);
    SetLength(Order, TrainingPairs.Count);
    for I := 0 to High(Order) do Order[I] := I;

    NN.SetLearningRate(LearningRate, {Momentum=}0.9);
    NN.SetL2Decay(0.0);

    WriteLn('Training sine regression for ', NumEpochs, ' epochs',
      ' on ', NumTrain, ' samples (batch=', BatchSize, ', lr=',
      LearningRate:0:4, ')...');

    for Epoch := 1 to NumEpochs do
    begin
      // Shuffled mini-batch SGD: one full pass over the dataset.
      ShuffleIndices(Order);
      Step := 0;
      NN.ClearDeltas();
      for I := 0 to TrainingPairs.Count - 1 do
      begin
        Pair := TrainingPairs[Order[I]];
        NN.Compute(Pair.I);
        NN.Backpropagate(Pair.O);
        Inc(Step);
        if Step = BatchSize then
        begin
          NN.UpdateWeights();
          NN.ClearDeltas();
          Step := 0;
        end;
      end;
      if Step > 0 then
      begin
        NN.UpdateWeights();
        NN.ClearDeltas();
      end;

      if (Epoch = 1) or (Epoch mod 25 = 0) or (Epoch = NumEpochs) then
        WriteLn('  Epoch ', Epoch:4, '  clean-grid MSE = ',
          CleanGridMSE(NN):0:6);
    end;

    WriteLn;
    WriteLn('Final predictions on 11 evenly-spaced points in [-1,1]:');
    WriteLn('       x         sin(pi*x)         predicted        error');
    Input := TNNetVolume.Create([TNeuralFloat(0)]);
    Output := TNNetVolume.Create(1, 1, 1);
    for I := 0 to 10 do
    begin
      X := -1.0 + I / 5.0;
      Target := Sin(Pi * X);
      Input.FData[0] := X;
      NN.Compute(Input);
      NN.GetOutput(Output);
      Pred := Output.Raw[0];
      WriteLn(X:8:4, Target:18:6, Pred:18:6, (Pred - Target):14:6);
    end;
    Input.Free;
    Output.Free;

    TrainingPairs.Free;
    NN.Free;
  end;

var
  // Stops Lazarus errors
  Application: record Title: string; end;

begin
  Application.Title := 'Sine Regression Example';
  RunAlgo();
end.
