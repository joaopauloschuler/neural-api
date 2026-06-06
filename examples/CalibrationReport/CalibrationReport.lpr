program CalibrationReport;
(*
CalibrationReport: trains a small MLP classifier on a synthetic 4-class
2D-cluster dataset, then prints the forward-only model-calibration /
reliability report (neuralcalibration.CalibrationReport) on a held-out
validation split BEFORE and AFTER fitting a single temperature-scaling
scalar. A deliberately aggressive learning rate makes the raw model
over-confident, so temperature scaling visibly shrinks the Expected
Calibration Error. The reliability diagram is also written out as a P2
(ASCII) PGM image. Pure-CPU, runs in a few seconds.

The temperature scaling operates on pseudo-logits z := ln(softmax_prob)
read off the final layer (see neuralcalibration's unit header): up to an
additive constant this recovers the true logits of a softmax head, so
softmax(z / T) is exactly temperature scaling. The backbone is never
re-trained -- T is found by a 1-D grid scan over validation NLL, and the
"after" report is produced by feeding the trained net's softmax output
through a tiny stateless wrapper net (Log -> *(1/T) -> SoftMax).

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
  neuralcalibration;

const
  cNumClasses     = 4;
  cTrainPerClass  = 200;
  cValPerClass    = 60;
  cEpochs         = 150;
  // A modest LR run for many epochs drives the softmax probabilities toward
  // the 0/1 extremes faster than accuracy improves -> an over-confident,
  // mis-calibrated model, so the temperature-scaling improvement is visible.
  cLearningRate   = 0.05;
  cBins           = 10;

  // Four overlapping 2D Gaussian clusters so the model makes confusable,
  // over-confident predictions.
  cCenters: array[0..3, 0..1] of TNeuralFloat =
    ((-1.0, -1.0),
     ( 1.0, -1.0),
     ( 1.0,  1.0),
     (-1.0,  1.0));
  cSigma: TNeuralFloat = 0.70;

procedure BuildNet(out NN: TNNet);
begin
  NN := TNNet.Create();
  NN.AddLayer(TNNetInput.Create(2, 1, 1));
  NN.AddLayer(TNNetFullConnectReLU.Create(16));
  NN.AddLayer(TNNetFullConnectReLU.Create(16));
  NN.AddLayer(TNNetFullConnectLinear.Create(cNumClasses));
  NN.AddLayer(TNNetSoftMax.Create());
end;

// Box-Muller N(0,1) sample, no external dependency.
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

// Builds an input list (owns its volumes), a parallel integer label array,
// and a one-hot training pair list (owns its own copies of the inputs).
procedure BuildSet(PerClass: integer;
  out Inputs: TNNetVolumeList; out Labels: TNeuralIntegerArray;
  out TrainPairs: TNNetVolumePairList);
var
  C, I, N: integer;
  X, XCopy, Y: TNNetVolume;
begin
  Inputs := TNNetVolumeList.Create();
  TrainPairs := TNNetVolumePairList.Create();
  N := cNumClasses * PerClass;
  SetLength(Labels, N);
  for C := 0 to cNumClasses - 1 do
    for I := 1 to PerClass do
    begin
      X := TNNetVolume.Create(2, 1, 1);
      X.FData[0] := cCenters[C][0] + RandomGauss() * cSigma;
      X.FData[1] := cCenters[C][1] + RandomGauss() * cSigma;
      Labels[Inputs.Count] := C;
      Inputs.Add(X);
      // independent copy for training (owned by TrainPairs).
      XCopy := TNNetVolume.Create(2, 1, 1);
      XCopy.Copy(X);
      Y := TNNetVolume.Create(cNumClasses, 1, 1);
      Y.Fill(0);
      Y.FData[C] := 1.0;
      TrainPairs.Add(TNNetVolumePair.Create(XCopy, Y));
    end;
end;

// Produces a new input list whose volumes are the trained net's softmax
// output for each validation input (so a wrapper net can post-process them).
function MakeProbInputs(NN: TNNet; Inputs: TNNetVolumeList): TNNetVolumeList;
var
  I: integer;
  V: TNNetVolume;
begin
  Result := TNNetVolumeList.Create();
  for I := 0 to Inputs.Count - 1 do
  begin
    NN.Compute(Inputs[I]);
    V := TNNetVolume.Create(NN.GetLastLayer().Output.Size, 1, 1);
    V.Copy(NN.GetLastLayer().Output);
    Result.Add(V);
  end;
end;

procedure RunDemo();
var
  NN, ScaledNN: TNNet;
  TrainInputs, ValInputs, ProbInputs: TNNetVolumeList;
  TrainLabels, ValLabels: TNeuralIntegerArray;
  TrainPairs, ValPairs: TNNetVolumePairList;
  Epoch, I: integer;
  Loss, T: TNeuralFloat;
  Rep, RepBefore, RepAfter: TNeuralCalibrationReport;
  PgmName: string;
begin
  RandSeed := 12345;
  BuildNet(NN);
  BuildSet(cTrainPerClass, TrainInputs, TrainLabels, TrainPairs);
  BuildSet(cValPerClass, ValInputs, ValLabels, ValPairs);
  ScaledNN := nil;
  ProbInputs := nil;
  try
    NN.SetLearningRate(cLearningRate, 0.9);

    WriteLn('Architecture:');
    NN.PrintSummary();
    WriteLn;

    WriteLn('Training for ', cEpochs, ' epochs on ',
      TrainPairs.Count, ' samples (long run -> over-confident)...');
    for Epoch := 1 to cEpochs do
    begin
      Loss := 0;
      for I := 0 to TrainPairs.Count - 1 do
      begin
        NN.Compute(TrainPairs[I].I);
        NN.Backpropagate(TrainPairs[I].O);
        Loss := Loss - Ln(Max(1e-9,
          NN.GetLastLayer().Output.FData[TrainPairs[I].O.GetClass()]));
      end;
      if (Epoch mod 40 = 0) or (Epoch = 1) then
        WriteLn('  epoch ', Epoch:4, '  mean_nll=',
          (Loss / TrainPairs.Count):8:5);
    end;
    WriteLn;

    // BEFORE: raw (un-scaled) calibration.
    WriteLn('=== Calibration BEFORE temperature scaling (T = 1.0) ===');
    Write(neuralcalibration.CalibrationReport(NN, ValInputs, ValLabels, cBins));
    WriteLn;
    RepBefore := ComputeCalibration(NN, ValInputs, ValLabels, cBins);

    // Fit T on the validation split (forward-only grid scan over NLL).
    T := FitTemperature(NN, ValInputs, ValLabels);
    WriteLn(Format('Fitted temperature T = %.4f', [T]));
    WriteLn;

    // AFTER: build a tiny stateless wrapper net that consumes the trained
    // net's softmax output and applies log -> *(1/T) -> softmax, i.e. exactly
    // temperature scaling on the pseudo-logits. The trained backbone is
    // untouched; we just re-evaluate calibration on the scaled probabilities.
    ScaledNN := TNNet.Create();
    ScaledNN.AddLayer(TNNetInput.Create(cNumClasses, 1, 1));
    ScaledNN.AddLayer(TNNetLog.Create());
    ScaledNN.AddLayer(TNNetMulByConstant.Create(1.0 / T));
    ScaledNN.AddLayer(TNNetSoftMax.Create());

    ProbInputs := MakeProbInputs(NN, ValInputs);
    WriteLn('=== Calibration AFTER temperature scaling ===');
    Write(neuralcalibration.CalibrationReport(ScaledNN, ProbInputs, ValLabels, cBins));
    WriteLn;
    RepAfter := ComputeCalibration(ScaledNN, ProbInputs, ValLabels, cBins);

    WriteLn(Format('ECE improvement: %.4f (before) -> %.4f (after)  delta %.4f',
      [RepBefore.ECE, RepAfter.ECE, RepBefore.ECE - RepAfter.ECE]));
    WriteLn;

    // Write the reliability-diagram PGM for the BEFORE state.
    Rep := RepBefore;
    PgmName := 'reliability.pgm';
    if WriteReliabilityPGM(Rep, PgmName) then
      WriteLn('Wrote reliability diagram to ', PgmName,
        ' (', Rep.BinCount, ' bins)')
    else
      WriteLn('Failed to write ', PgmName);
  finally
    ProbInputs.Free;
    ValPairs.Free;
    TrainPairs.Free;
    ValInputs.Free;
    TrainInputs.Free;
    ScaledNN.Free;
    NN.Free;
  end;
end;

begin
  RunDemo();
end.
