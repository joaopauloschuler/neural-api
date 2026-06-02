program TverskySweep;
(*
TverskySweep: a pure alpha/beta KNOB STUDY on the TNNetTverskyLoss segmentation
head. It forks DiceSegmentation's structure but, instead of contrasting Tversky
against MSE, it trains the SAME tiny net three times on the SAME deliberately
CLASS-IMBALANCED synthetic mask, once per (alpha, beta) pair:

  (0.5, 0.5)  -> reduces to Dice (balanced FP/FN penalty)
  (0.3, 0.7)  -> beta > alpha: false NEGATIVES penalised harder
  (0.7, 0.3)  -> alpha > beta: false POSITIVES penalised harder

The Tversky index is  TI = TP / (TP + alpha*FP + beta*FN),  loss = 1 - TI.
alpha weights false positives, beta weights false negatives. Raising beta above
alpha makes a missed foreground pixel (FN) more expensive than a spurious
foreground pixel (FP), so the trained model is pushed to predict MORE
foreground: RECALL rises and FN falls, at the cost of precision (more FP).
Raising alpha above beta does the opposite. With alpha = beta it is Dice.

WHAT IT SHOWS
-------------
The task is intentionally hard and imbalanced so the trade is VISIBLE rather
than saturating: a SMALL disc (a minority of pixels) on a 12x12 grid, with
heavy input noise so the net cannot trivially fit every sample. After each of
the three runs the held-out set is scored for precision, recall, F1/Dice and
the raw FP / FN pixel counts. A compact table is printed, and the program
checks (and prints) the headline trend:

  as beta grows relative to alpha  ->  recall UP, FN DOWN (precision DOWN).

WIRING (mirrors DiceSegmentation exactly, only the head differs)
----------------------------------------------------------------
  Input(W, H, 1)
  ConvolutionReLU(F, 3, 1, 1)   // pad 1 keeps W x H
  ConvolutionReLU(F, 3, 1, 1)
  ConvolutionLinear(1, 3, 1, 1) // 1-channel logit map, same W x H
  Sigmoid                       // per-pixel foreground probability in [0,1]
  TNNetTverskyLoss(alpha, beta, smooth)   // identity passthrough + analytic grad

TNNetTverskyLoss is a TNNetIdentity descendant: forward is a passthrough so
Net.Compute still returns the Sigmoid probabilities. The framework seeds the
last layer's FOutputError with (output - target); the head recovers the binary
ground-truth mask and overwrites the residual with the analytic Tversky
gradient driven by the configured alpha/beta. Because the head reads p as a
foreground PROBABILITY, the feeding layer MUST be a Sigmoid and the TARGET
supplied to Backpropagate is the binary mask. All three runs start from the
same RandSeed, so they see identical data and identical initial weights — the
only thing that changes between runs is (alpha, beta).

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

uses {$IFDEF UNIX} {$IFDEF UseCThreads}
  cthreads, {$ENDIF} {$ENDIF}
  Classes, SysUtils, Math,
  neuralnetwork,
  neuralvolume;

const
  cGrid         = 12;     // image side (cGrid x cGrid, single channel)
  cFeatures     = 6;      // conv features per hidden layer
  cTrainSamples = 192;    // synthetic training images
  cTestSamples  = 96;     // held-out images
  cEpochs       = 24;
  cLearningRate = 0.04;
  cNoise        = 0.85;    // heavy input noise -> task is genuinely hard
  cSeed         = 12345;
  cThreshold    = 0.5;     // probability -> foreground threshold

type
  TVolArray = array of TNNetVolume;

// Box-Muller N(0,1) sample (no extra dependency).
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

// Builds one synthetic (input, mask) pair: a SMALL randomly placed filled disc.
// The disc is kept small (radius 2..3) so the foreground is a clear MINORITY of
// the pixels — the imbalanced regime in which the alpha/beta knob matters. Heavy
// noise is added to the INPUT only.
procedure MakeSample(Img, Mask: TNNetVolume);
var
  Cx, Cy, R, X, Y: integer;
  Dx, Dy: integer;
  Inside: boolean;
begin
  R := 2 + Random(2);                   // radius 2..3 -> small blob
  Cx := R + Random(cGrid - 2 * R);
  Cy := R + Random(cGrid - 2 * R);
  for Y := 0 to cGrid - 1 do
    for X := 0 to cGrid - 1 do
    begin
      Dx := X - Cx;
      Dy := Y - Cy;
      Inside := (Dx * Dx + Dy * Dy) <= (R * R);
      if Inside then Mask[X, Y, 0] := 1 else Mask[X, Y, 0] := 0;
      if Inside then
        Img[X, Y, 0] := 1 + RandomGauss() * cNoise
      else
        Img[X, Y, 0] := 0 + RandomGauss() * cNoise;
    end;
end;

procedure BuildDataset(out Imgs, Masks: TVolArray; N: integer);
var
  I: integer;
begin
  SetLength(Imgs, N);
  SetLength(Masks, N);
  for I := 0 to N - 1 do
  begin
    Imgs[I] := TNNetVolume.Create(cGrid, cGrid, 1);
    Masks[I] := TNNetVolume.Create(cGrid, cGrid, 1);
    MakeSample(Imgs[I], Masks[I]);
  end;
end;

procedure FreeDataset(var Imgs, Masks: TVolArray);
var
  I: integer;
begin
  for I := 0 to Length(Imgs) - 1 do
  begin
    Imgs[I].Free;
    Masks[I].Free;
  end;
  SetLength(Imgs, 0);
  SetLength(Masks, 0);
end;

// Builds the fully-convolutional segmentation net ending with a
// TNNetTverskyLoss head configured with the given (alpha, beta).
function BuildNet(pAlpha, pBeta: TNeuralFloat): TNNet;
begin
  Result := TNNet.Create();
  Result.AddLayer(TNNetInput.Create(cGrid, cGrid, 1));
  Result.AddLayer(TNNetConvolutionReLU.Create(cFeatures, 3, 1, 1));
  Result.AddLayer(TNNetConvolutionReLU.Create(cFeatures, 3, 1, 1));
  Result.AddLayer(TNNetConvolutionLinear.Create(1, 3, 1, 1));
  Result.AddLayer(TNNetSigmoid.Create());
  Result.AddLayer(TNNetTverskyLoss.Create(pAlpha, pBeta, 1.0));
  Result.InitWeights();
end;

function ForegroundProb(Net: TNNet; X, Y: integer): TNeuralFloat;
begin
  Result := Net.GetLastLayer().Output[X, Y, 0];
end;

procedure TrainEpoch(Net: TNNet; const Imgs, Masks: TVolArray);
var
  Step, Idx, N: integer;
begin
  N := Length(Imgs);
  for Step := 0 to N - 1 do
  begin
    Idx := Random(N);
    Net.Compute(Imgs[Idx]);
    Net.Backpropagate(Masks[Idx]);
  end;
end;

// Accumulates true/false positive/negative pixel counts over a dataset
// (prediction thresholded at 0.5) and derives precision, recall and F1/Dice.
procedure Evaluate(Net: TNNet; const Imgs, Masks: TVolArray;
  out Precision, Recall, F1: TNeuralFloat; out TotFP, TotFN: integer);
var
  I, X, Y, N: integer;
  TotTP: integer;
  P, G: TNeuralFloat;
  Pred, Gt: boolean;
begin
  N := Length(Imgs);
  TotTP := 0;
  TotFP := 0;
  TotFN := 0;
  for I := 0 to N - 1 do
  begin
    Net.Compute(Imgs[I]);
    for Y := 0 to cGrid - 1 do
      for X := 0 to cGrid - 1 do
      begin
        P := ForegroundProb(Net, X, Y);
        G := Masks[I][X, Y, 0];
        Pred := P >= cThreshold;
        Gt := G >= 0.5;
        if Pred and Gt then Inc(TotTP)
        else if Pred and (not Gt) then Inc(TotFP)
        else if (not Pred) and Gt then Inc(TotFN);
      end;
  end;
  if (TotTP + TotFP) > 0 then Precision := TotTP / (TotTP + TotFP)
  else Precision := 0;
  if (TotTP + TotFN) > 0 then Recall := TotTP / (TotTP + TotFN)
  else Recall := 0;
  if (Precision + Recall) > 0 then
    F1 := 2 * Precision * Recall / (Precision + Recall)
  else F1 := 0;
end;

// Trains one net for the given (alpha, beta) and returns the held-out metrics.
procedure RunPair(pAlpha, pBeta: TNeuralFloat;
  const TrainImgs, TrainMasks, TestImgs, TestMasks: TVolArray;
  out Precision, Recall, F1: TNeuralFloat; out FP, FN: integer);
var
  Net: TNNet;
  Epoch: integer;
begin
  // Identical seed before every run -> identical initial weights and identical
  // training-sample order across the three (alpha, beta) settings.
  RandSeed := cSeed + 1;
  Net := BuildNet(pAlpha, pBeta);
  try
    Net.SetLearningRate(cLearningRate, 0.9);
    for Epoch := 1 to cEpochs do
      TrainEpoch(Net, TrainImgs, TrainMasks);
    Evaluate(Net, TestImgs, TestMasks, Precision, Recall, F1, FP, FN);
  finally
    Net.Free;
  end;
end;

procedure RunAlgo();
const
  cNPairs = 3;
  cAlpha: array[0..cNPairs - 1] of TNeuralFloat = (0.5, 0.3, 0.7);
  cBeta:  array[0..cNPairs - 1] of TNeuralFloat = (0.5, 0.7, 0.3);
var
  TrainImgs, TrainMasks, TestImgs, TestMasks: TVolArray;
  Prec, Rec, F1: array[0..cNPairs - 1] of TNeuralFloat;
  FP, FN: array[0..cNPairs - 1] of integer;
  I, FgPix, BgPix, X, Y, S: integer;
  RecallMonotone, FnMonotone: boolean;
begin
  RandSeed := cSeed;
  WriteLn('TverskySweep: alpha/beta knob study on TNNetTverskyLoss');
  WriteLn(Format('grid=%dx%d  features=%d  train=%d  test=%d  epochs=%d  lr=%.3f  noise=%.2f',
    [cGrid, cGrid, cFeatures, cTrainSamples, cTestSamples, cEpochs,
     cLearningRate, cNoise]));
  WriteLn;

  // Same RandSeed before dataset build -> all three runs see identical data.
  RandSeed := cSeed;
  BuildDataset(TrainImgs, TrainMasks, cTrainSamples);
  BuildDataset(TestImgs, TestMasks, cTestSamples);

  // Report the class imbalance of the held-out set.
  FgPix := 0;
  for I := 0 to cTestSamples - 1 do
    for Y := 0 to cGrid - 1 do
      for X := 0 to cGrid - 1 do
        if TestMasks[I][X, Y, 0] >= 0.5 then Inc(FgPix);
  S := cTestSamples * cGrid * cGrid;
  BgPix := S - FgPix;
  WriteLn(Format('Held-out class balance: foreground %d / %d pixels (%.1f%%),  background %d (%.1f%%)',
    [FgPix, S, 100.0 * FgPix / S, BgPix, 100.0 * BgPix / S]));
  WriteLn('=> foreground is a deliberate minority (class-imbalanced mask).');
  WriteLn;

  try
    for I := 0 to cNPairs - 1 do
    begin
      WriteLn(Format('Training (alpha=%.1f, beta=%.1f)%s ...',
        [cAlpha[I], cBeta[I],
         BoolToStr(SameValue(cAlpha[I], cBeta[I]), '  [= Dice]', '')]));
      RunPair(cAlpha[I], cBeta[I], TrainImgs, TrainMasks, TestImgs, TestMasks,
        Prec[I], Rec[I], F1[I], FP[I], FN[I]);
    end;

    WriteLn;
    WriteLn('Held-out results (threshold 0.5, counts summed over all test pixels)');
    WriteLn('  (alpha,beta)   note      precision   recall      F1/Dice      FP      FN');
    for I := 0 to cNPairs - 1 do
      WriteLn(Format('  (%.1f, %.1f)   %-8s  %8.4f   %8.4f   %8.4f   %5d   %5d',
        [cAlpha[I], cBeta[I],
         BoolToStr(SameValue(cAlpha[I], cBeta[I]), 'Dice', ''),
         Prec[I], Rec[I], F1[I], FP[I], FN[I]]));
    WriteLn;

    // Reorder for the headline statement by ascending beta-minus-alpha:
    //   index 2 = (0.7,0.3)  beta<alpha
    //   index 0 = (0.5,0.5)  beta=alpha (Dice)
    //   index 1 = (0.3,0.7)  beta>alpha
    WriteLn('Trend as beta rises relative to alpha (beta-alpha: -0.4 -> 0 -> +0.4):');
    WriteLn(Format('  recall:  %.4f -> %.4f -> %.4f', [Rec[2], Rec[0], Rec[1]]));
    WriteLn(Format('  FN:      %5d -> %5d -> %5d',    [FN[2], FN[0], FN[1]]));
    WriteLn(Format('  precision:%.4f -> %.4f -> %.4f', [Prec[2], Prec[0], Prec[1]]));
    WriteLn(Format('  FP:      %5d -> %5d -> %5d',    [FP[2], FP[0], FP[1]]));
    WriteLn;

    RecallMonotone := (Rec[2] <= Rec[0] + 1e-6) and (Rec[0] <= Rec[1] + 1e-6);
    FnMonotone     := (FN[2] >= FN[0]) and (FN[0] >= FN[1]);
    if RecallMonotone and FnMonotone then
      WriteLn('=> CONFIRMED: beta>alpha raises RECALL and lowers FN ',
              '(it trades precision/FP for recall).')
    else if (Rec[1] > Rec[2] + 1e-6) and (FN[1] < FN[2]) then
      WriteLn('=> Endpoints confirm the trend: (0.3,0.7) has higher recall and ',
              'fewer FN than (0.7,0.3), even if the Dice midpoint is not exactly monotone.')
    else
      WriteLn('=> On this run the recall/FN trend was not monotone; ',
              'see the table above for the actual numbers.');
  finally
    FreeDataset(TrainImgs, TrainMasks);
    FreeDataset(TestImgs, TestMasks);
  end;
end;

var
  // Stops Lazarus errors
  Application: record Title: string; end;

begin
  Application.Title := 'TverskySweep Example';
  RunAlgo();
end.
