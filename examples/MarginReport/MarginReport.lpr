program MarginReport;
(*
MarginReport: trains a small MLP classifier on a synthetic 4-class
2D-cluster dataset, then prints TNNet.TopLogitMarginReport for the
held-out validation split. Pure-CPU, runs in a few seconds.

The report shows the per-sample top-logit margin
(top1_logit - top2_logit) on the final-layer output: an overall 10-bin
ASCII histogram, per-class mean/median margin (so a systematically
uncertain class stands out), and the lowest-margin sample indices per
class as a ready-made "hard examples" pool.

Copyright (C) 2026 Joao Paulo Schwarz Schuler

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
any later version.

Coded by Claude (AI).
*)

{$mode objfpc}{$H+}

uses {$IFDEF UNIX} {$IFDEF UseCThreads}
  cthreads, {$ENDIF} {$ENDIF}
  Classes, SysUtils, Math,
  neuralnetwork,
  neuralvolume;

const
  cNumClasses     = 4;
  cTrainPerClass  = 200;
  cValPerClass    = 40;
  cEpochs         = 60;
  cLearningRate   = 0.05;

  // Four 2D Gaussian clusters. Classes 0..2 are well separated; class 3 sits
  // between 0 and 1 with a wider spread, so the model is systematically less
  // certain about it -> a visibly smaller per-class margin.
  cCenters: array[0..3, 0..1] of TNeuralFloat =
    ((-1.6, -1.6),
     ( 1.6, -1.6),
     ( 0.0,  1.8),
     ( 0.0, -1.5));
  cSigma: array[0..3] of TNeuralFloat = (0.55, 0.55, 0.55, 0.95);

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
  X.FData[0] := cCenters[ClassId][0] + RandomGauss() * cSigma[ClassId];
  X.FData[1] := cCenters[ClassId][1] + RandomGauss() * cSigma[ClassId];
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

    WriteLn('Top-logit margin report on validation set (',
      ValSet.Count, ' samples):');
    WriteLn(StringOfChar('=', 78));
    Report := TNNet.TopLogitMarginReport(NN, ValSet, cNumClasses, 5);
    Write(Report);
    WriteLn(StringOfChar('=', 78));
  finally
    ValSet.Free;
    TrainSet.Free;
    NN.Free;
  end;
end;

begin
  RunDemo();
end.
