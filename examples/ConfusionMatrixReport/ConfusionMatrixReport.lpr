program ConfusionMatrixReport;
(*
ConfusionMatrixReport: trains a small MLP classifier on a synthetic 3-class
2D-cluster dataset, then prints TNNet.ConfusionMatrixReport for the
held-out validation split. Pure-CPU, runs in a few seconds.

The confusion matrix, per-class precision/recall/F1, top-1 / balanced
accuracy, most-confused pairs and per-class hard-example indices give
an at-a-glance view of which classes the model mixes up and where the
remaining error mass sits.

Copyright (C) 2026 Joao Paulo Schwarz Schuler

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
any later version.
*)

{$mode objfpc}{$H+}

uses {$IFDEF UNIX} {$IFDEF UseCThreads}
  cthreads, {$ENDIF} {$ENDIF}
  Classes, SysUtils, Math,
  neuralnetwork,
  neuralvolume;

const
  cNumClasses     = 3;
  cTrainPerClass  = 200;
  cValPerClass    = 40;
  cEpochs         = 60;
  cLearningRate   = 0.05;

  // Three 2D Gaussian clusters with overlapping tails so the model
  // makes a few confusable predictions.
  cCenters: array[0..2, 0..1] of TNeuralFloat =
    ((-1.2, -1.2),
     ( 1.2, -1.0),
     ( 0.0,  1.3));
  cSigma: TNeuralFloat = 0.85;

procedure BuildNet(out NN: TNNet);
begin
  NN := TNNet.Create();
  NN.AddLayer(TNNetInput.Create(2, 1, 1));
  NN.AddLayer(TNNetFullConnectReLU.Create(16));
  NN.AddLayer(TNNetFullConnectReLU.Create(16));
  NN.AddLayer(TNNetFullConnectLinear.Create(cNumClasses));
  NN.AddLayer(TNNetSoftMax.Create());
end;

// Why: Box-Muller gives N(0,1) samples without pulling in a dependency.
function RandomGauss(): TNeuralFloat;
var
  U1, U2: TNeuralFloat;
begin
  repeat
    U1 := Random;
  until U1 > 1e-9;
  U2 := Random;
  Result := Sqrt(-2 * Ln(U1)) * Cos(2 * Pi * U2);
end;

procedure MakeSample(ClassId: integer; out X, Y: TNNetVolume);
begin
  X := TNNetVolume.Create(2, 1, 1);
  Y := TNNetVolume.Create(cNumClasses, 1, 1);
  X.FData[0] := cCenters[ClassId][0] + RandomGauss() * cSigma;
  X.FData[1] := cCenters[ClassId][1] + RandomGauss() * cSigma;
  Y.Fill(0);
  Y.FData[ClassId] := 1.0;
end;

procedure BuildSet(out Pairs: TNNetVolumePairList; PerClass: integer);
var
  C, I: integer;
  X, Y: TNNetVolume;
begin
  Pairs := TNNetVolumePairList.Create();
  for C := 0 to cNumClasses - 1 do
    for I := 1 to PerClass do
    begin
      MakeSample(C, X, Y);
      Pairs.Add(TNNetVolumePair.Create(X, Y));
    end;
end;

procedure RunDemo();
var
  NN: TNNet;
  TrainSet, ValSet: TNNetVolumePairList;
  Epoch, I: integer;
  Pair: TNNetVolumePair;
  Loss, Diff: TNeuralFloat;
  Report: string;
  K: integer;
begin
  RandSeed := 12345;
  BuildNet(NN);
  BuildSet(TrainSet, cTrainPerClass);
  BuildSet(ValSet, cValPerClass);
  try
    NN.SetLearningRate(cLearningRate, 0.9);

    WriteLn('Architecture:');
    NN.PrintSummary();
    WriteLn;

    WriteLn('Training for ', cEpochs, ' epochs on ',
      TrainSet.Count, ' samples...');
    for Epoch := 1 to cEpochs do
    begin
      Loss := 0;
      for I := 0 to TrainSet.Count - 1 do
      begin
        Pair := TrainSet[I];
        NN.Compute(Pair.I);
        NN.Backpropagate(Pair.O);
        Diff := -Ln(Max(1e-9,
          NN.GetLastLayer().Output.FData[Pair.O.GetClass()]));
        Loss := Loss + Diff;
      end;
      if (Epoch mod 20 = 0) or (Epoch = 1) then
        WriteLn('  epoch ', Epoch:4, '  mean_nll=',
          (Loss / TrainSet.Count):8:5);
    end;
    WriteLn;

    WriteLn('Confusion matrix report on validation set (',
      ValSet.Count, ' samples):');
    WriteLn(StringOfChar('=', 78));
    Report := TNNet.ConfusionMatrixReport(NN, ValSet, cNumClasses, 3, 2);
    Write(Report);
    WriteLn(StringOfChar('=', 78));
    // Why: silence FPC hint about K being unused if needed for debug.
    K := 0;
    if K <> 0 then K := K;
  finally
    ValSet.Free;
    TrainSet.Free;
    NN.Free;
  end;
end;

begin
  RunDemo();
end.
