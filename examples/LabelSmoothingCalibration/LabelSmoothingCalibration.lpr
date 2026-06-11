program LabelSmoothingCalibration;
(*
LabelSmoothingCalibration: does label smoothing actually improve a
classifier's CALIBRATION?

The textbook claim (Mueller, Kornblith & Hinton, 2019, "When Does Label
Smoothing Help?") is that training with a smoothed target
    t' = (1 - eps) * onehot + eps / NumClasses
discourages the network from driving its winning logit arbitrarily high,
so the softmax confidences track the true accuracy more closely -- i.e.
lower Expected Calibration Error (ECE) and a lower Brier score -- usually
at the price of a small drop in top-1 accuracy. This program tests that
claim head-on by training the SAME tiny MLP four times, once per
eps in {0, 0.05, 0.1, 0.2}, and feeding each trained model into the
forward-only calibration report (neuralcalibration: ECE + Brier).

eps = 0 is the baseline arm: the smoothed target collapses to the plain
one-hot, so TNNetLabelSmoothingLoss(0) is exactly standard softmax
cross-entropy. Every arm shares the RNG seed, data, architecture, epochs
and learning rate, so the ONLY difference between arms is the smoothing
strength -- an apples-to-apples comparison.

TASK DESIGN (deliberately HARD so calibration differences are visible).
A model that is right with 100% confidence on every sample has ECE ~ 0
regardless of eps, which would make the comparison meaningless. To keep
the model genuinely uncertain we make the problem un-separable:
  - 6 classes laid out as 2D Gaussian clusters on a small ring, with a
    large sigma so adjacent clusters HEAVILY OVERLAP (Bayes error is far
    above zero -- no model can be confidently correct everywhere);
  - on top of that, cLabelNoise fraction of the TRAINING labels are
    randomly corrupted, which (without smoothing) pushes the network to
    memorise wrong targets at full confidence -> classic over-confidence.
The validation set uses the SAME overlapping clusters but CLEAN labels,
so we measure calibration against the true generative labels.

The classifier ends in SoftMax so its output is a probability simplex;
TNNetLabelSmoothingLoss is an identity passthrough appended as the loss
head (it only rewrites the back-prop gradient), so the last layer's
output -- which neuralcalibration reads -- is still the softmax vector.

Pure CPU, no dataset download, runs in well under a minute.

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
  cNumClasses    = 6;
  cTrainPerClass = 220;
  cValPerClass   = 80;
  cEpochs        = 120;
  cLearningRate  = 0.05;
  cInertia       = 0.9;
  cBins          = 10;
  cSeed          = 20260605;
  // Heavy cluster overlap + label noise => the model CANNOT be confidently
  // correct everywhere, so ECE is well away from 0 and discriminates eps.
  cSigma         = 1.05;
  cRingRadius    = 1.3;
  cLabelNoise    = 0.15;     // fraction of TRAIN labels randomly corrupted

  cNumEps = 4;
  cEpsList: array[0..cNumEps - 1] of TNeuralFloat = (0.0, 0.05, 0.10, 0.20);

// Classifier shared by every arm. Ends in SoftMax (probability simplex);
// the LabelSmoothingLoss head is appended separately at build time.
function BuildNet(Eps: TNeuralFloat): TNNet;
var
  NN: TNNet;
begin
  NN := TNNet.Create();
  NN.AddLayer(TNNetInput.Create(2, 1, 1));
  NN.AddLayer(TNNetFullConnectReLU.Create(24));
  NN.AddLayer(TNNetFullConnectReLU.Create(24));
  NN.AddLayer(TNNetFullConnectLinear.Create(cNumClasses));
  NN.AddLayer(TNNetSoftMax.Create());
  // Loss head: identity on forward, smoothed-CE gradient on backward.
  // eps = 0 reduces exactly to plain softmax cross-entropy (baseline arm).
  NN.AddLayer(TNNetLabelSmoothingLoss.Create(Eps));
  NN.SetLearningRate(cLearningRate, cInertia);
  Result := NN;
end;

// Box-Muller N(0,1) sample.
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

// Class center on a ring of cNumClasses overlapping Gaussian clusters.
procedure ClassCenter(C: integer; out CX, CY: TNeuralFloat);
var
  Ang: TNeuralFloat;
begin
  Ang := 2 * Pi * C / cNumClasses;
  CX := cRingRadius * Cos(Ang);
  CY := cRingRadius * Sin(Ang);
end;

// Builds an input list (owns its volumes) + parallel integer label array for
// evaluation, and a one-hot training-pair list (owns its own copies). When
// NoiseFrac > 0 the TRAINING target labels are randomly corrupted (the
// returned eval Labels stay CLEAN so calibration is measured against truth).
procedure BuildSet(PerClass: integer; NoiseFrac: TNeuralFloat;
  out Inputs: TNNetVolumeList; out Labels: TNeuralIntegerArray;
  out TrainPairs: TNNetVolumePairList);
var
  C, I, N, NoisyClass: integer;
  CX, CY: TNeuralFloat;
  X, XCopy, Y: TNNetVolume;
begin
  Inputs := TNNetVolumeList.Create();
  TrainPairs := TNNetVolumePairList.Create();
  N := cNumClasses * PerClass;
  SetLength(Labels, N);
  for C := 0 to cNumClasses - 1 do
  begin
    ClassCenter(C, CX, CY);
    for I := 1 to PerClass do
    begin
      X := TNNetVolume.Create(2, 1, 1);
      X.FData[0] := CX + RandomGauss() * cSigma;
      X.FData[1] := CY + RandomGauss() * cSigma;
      Labels[Inputs.Count] := C;   // clean label for evaluation
      Inputs.Add(X);

      // Independent copy for the training-pair list (owned by TrainPairs).
      XCopy := TNNetVolume.Create(2, 1, 1);
      XCopy.Copy(X);
      NoisyClass := C;
      if (NoiseFrac > 0) and (Random < NoiseFrac) then
        NoisyClass := Random(cNumClasses);   // corrupt the TRAINING target
      Y := TNNetVolume.Create(cNumClasses, 1, 1);
      Y.Fill(0);
      Y.FData[NoisyClass] := 1.0;
      TrainPairs.Add(TNNetVolumePair.Create(XCopy, Y));
    end;
  end;
end;

// One training arm; returns final-epoch mean (true-label) cross-entropy.
function TrainArm(NN: TNNet; TrainPairs: TNNetVolumePairList): TNeuralFloat;
var
  Epoch, I: integer;
  Loss: TNeuralFloat;
begin
  Result := 0;
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
    Result := Loss / TrainPairs.Count;
    if (Epoch = 1) or (Epoch mod 40 = 0) or (Epoch = cEpochs) then
      WriteLn(Format('    epoch %4d / %4d   mean-nll=%.5f',
        [Epoch, cEpochs, Result]));
  end;
end;

var
  Idx, BestEce, BestBrier: integer;
  NN: TNNet;
  TrainInputs, ValInputs: TNNetVolumeList;
  TrainLabels, ValLabels: TNeuralIntegerArray;
  TrainPairs, ValPairs: TNNetVolumePairList;
  Rep: array[0..cNumEps - 1] of TNeuralCalibrationReport;
  GStart, GElapsed: double;
begin
  SetExceptionMask([exInvalidOp, exDenormalized, exZeroDivide,
                    exOverflow, exUnderflow, exPrecision]);

  WriteLn('LabelSmoothingCalibration: same tiny MLP, four label-smoothing ',
    'strengths.');
  WriteLn(Format('Task: %d HEAVILY OVERLAPPING Gaussian clusters on a ring ' +
    '(sigma=%.2f, r=%.2f)', [cNumClasses, cSigma, cRingRadius]));
  WriteLn(Format('      + %.0f%% random TRAIN-label noise -> the model is ' +
    'genuinely uncertain.', [100.0 * cLabelNoise]));
  WriteLn(Format('Sweeping eps in {0, 0.05, 0.10, 0.20}  (eps=0 == plain ' +
    'cross-entropy baseline).', []));
  WriteLn('Calibration measured on a CLEAN-label validation split via ',
    'neuralcalibration (ECE + Brier).');
  WriteLn;

  // One shared dataset for every arm (built once, reused). Train labels carry
  // noise; eval labels are clean.
  RandSeed := cSeed;
  BuildSet(cTrainPerClass, cLabelNoise, TrainInputs, TrainLabels, TrainPairs);
  BuildSet(cValPerClass, 0.0, ValInputs, ValLabels, ValPairs);

  GStart := Now();
  try
    for Idx := 0 to cNumEps - 1 do
    begin
      // Re-seed before each arm so weight init + training are identical
      // except for the smoothing strength.
      RandSeed := cSeed + 1;
      NN := BuildNet(cEpsList[Idx]);
      try
        if Idx = 0 then
        begin
          WriteLn('Architecture (eps arms differ only in the ',
            'TNNetLabelSmoothingLoss eps):');
          NN.PrintSummary();
          WriteLn;
        end;
        WriteLn(Format('  --- training arm eps=%.2f ---', [cEpsList[Idx]]));
        TrainArm(NN, TrainPairs);
        // Forward-only calibration on the clean validation split.
        Rep[Idx] := ComputeCalibration(NN, ValInputs, ValLabels, cBins);
        WriteLn(Format('    val-acc=%.4f  ECE=%.4f  Brier=%.4f',
          [Rep[Idx].Accuracy, Rep[Idx].ECE, Rep[Idx].Brier]));
      finally
        NN.Free;
      end;
      WriteLn;
    end;
  finally
    ValPairs.Free;
    TrainPairs.Free;
    ValInputs.Free;
    TrainInputs.Free;
  end;
  GElapsed := (Now() - GStart) * 86400.0;

  WriteLn(StringOfChar('=', 64));
  WriteLn('RESULTS  (lower ECE / lower Brier = better calibrated)');
  WriteLn(StringOfChar('=', 64));
  WriteLn('    eps     val-acc        ECE       Brier');
  WriteLn('   -----    -------     -------     -------');
  BestEce := 0;
  BestBrier := 0;
  for Idx := 0 to cNumEps - 1 do
  begin
    WriteLn(Format('   %5.2f     %6.2f%%     %.4f      %.4f',
      [cEpsList[Idx], 100.0 * Rep[Idx].Accuracy, Rep[Idx].ECE,
       Rep[Idx].Brier]));
    if Rep[Idx].ECE   < Rep[BestEce].ECE     then BestEce := Idx;
    if Rep[Idx].Brier < Rep[BestBrier].Brier then BestBrier := Idx;
  end;
  WriteLn;
  WriteLn(Format('Best ECE   : eps=%.2f (ECE=%.4f)',
    [cEpsList[BestEce], Rep[BestEce].ECE]));
  WriteLn(Format('Best Brier : eps=%.2f (Brier=%.4f)',
    [cEpsList[BestBrier], Rep[BestBrier].Brier]));
  if BestEce > 0 then
    WriteLn('Label smoothing IMPROVED ECE over the eps=0 cross-entropy ',
      'baseline (textbook claim holds here).')
  else
    WriteLn('Label smoothing did NOT beat the eps=0 baseline on ECE here ',
      '(result is mixed/negative -- reported honestly).');
  WriteLn(Format('Baseline (eps=0) ECE=%.4f vs best ECE=%.4f  (delta %.4f).',
    [Rep[0].ECE, Rep[BestEce].ECE, Rep[0].ECE - Rep[BestEce].ECE]));
  WriteLn(Format('Total runtime for all %d arms: %.1fs.',
    [cNumEps, GElapsed]));
end.
