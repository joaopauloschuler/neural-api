program DiceSegmentation;
(*
DiceSegmentation: demonstrates the TNNetDiceLoss segmentation head on a tiny
SYNTHETIC binary-mask task and contrasts the Dice/IoU it reaches against a
plain MSE-head baseline at IDENTICAL architecture. No external dataset, pure
CPU, finishes in a few seconds.

WHAT IT SHOWS
-------------
The synthetic task is foreground/background segmentation: each sample is a
small single-channel grid with a randomly placed/sized filled DISC; the
ground-truth mask is 1 inside the disc and 0 outside, and Gaussian noise is
added to the INPUT only. The foreground is a minority of the pixels, which is
the regime where a region-overlap loss (Dice) is expected to help versus a
per-pixel MSE loss.

Two identical fully-convolutional nets are trained:
  (a) head = TNNetDiceLoss, target = the binary mask.
  (b) baseline = a plain sigmoid output (MSE gradient), target = the same mask.
After training, on a held-out set both nets are scored by mean Dice coefficient
and mean IoU (prediction thresholded at 0.5). One held-out sample is rendered
as ASCII (input / ground truth / Dice prediction / MSE prediction) so the
reader can eyeball the result. The console prints whatever the numbers
ACTUALLY show.

HOW THE DICE HEAD IS WIRED
--------------------------
TNNetDiceLoss is a TNNetIdentity descendant (Tversky loss with
alpha = beta = 0.5). Forward is an identity passthrough, so its FOutput equals
the foreground-probability map produced by the preceding sigmoid. The framework
seeds the last layer's FOutputError with (output - target); the head recovers
the ground-truth mask g = p - seeded and overwrites the residual with the
analytic Dice gradient of L = 1 - 2*TP/(2*TP + FP + FN). Because the head reads
p as a foreground PROBABILITY in [0,1], the layer feeding it MUST be a sigmoid,
and the TARGET volume supplied to Backpropagate is the binary mask:

  Input(W, H, 1)
  ConvolutionReLU(F, 3, 1, 1)   // pad 1 keeps W x H
  ConvolutionReLU(F, 3, 1, 1)
  ConvolutionLinear(1, 3, 1, 1) // 1-channel logit map, same W x H
  Sigmoid                       // per-pixel foreground probability
  TNNetDiceLoss                 // identity passthrough + analytic Dice grad

The MSE baseline is the same stack WITHOUT the loss head: the last layer is the
sigmoid, so the framework-seeded (output - target) residual IS the MSE gradient.

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
*)

{$mode objfpc}{$H+}

uses {$IFDEF UNIX} {$IFDEF UseCThreads}
  cthreads, {$ENDIF} {$ENDIF}
  Classes, SysUtils, Math,
  neuralnetwork,
  neuralvolume;

const
  cGrid         = 16;     // image side (cGrid x cGrid, single channel)
  cFeatures     = 8;      // conv features per hidden layer
  cTrainSamples = 256;    // synthetic training images
  cTestSamples  = 64;     // held-out images
  cEpochs       = 30;
  cLearningRate = 0.05;
  cNoise        = 0.30;    // stddev of input noise
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

// Builds one synthetic (input, mask) pair: a randomly placed/sized filled disc.
// mask = 1 inside the disc, 0 outside. Input = clean shape signal + noise.
procedure MakeSample(Img, Mask: TNNetVolume);
var
  Cx, Cy, R, X, Y: integer;
  Dx, Dy: integer;
  Inside: boolean;
begin
  // Radius 3..5, center kept so the disc stays mostly on the grid.
  R := 3 + Random(3);
  Cx := R + Random(cGrid - 2 * R);
  Cy := R + Random(cGrid - 2 * R);
  for Y := 0 to cGrid - 1 do
    for X := 0 to cGrid - 1 do
    begin
      Dx := X - Cx;
      Dy := Y - Cy;
      Inside := (Dx * Dx + Dy * Dy) <= (R * R);
      if Inside then Mask[X, Y, 0] := 1 else Mask[X, Y, 0] := 0;
      // Input: foreground ~1, background ~0, both corrupted by noise.
      if Inside then
        Img[X, Y, 0] := 1 + RandomGauss() * cNoise
      else
        Img[X, Y, 0] := 0 + RandomGauss() * cNoise;
    end;
end;

// Allocates and fills a dataset of N (input, mask) pairs.
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

// Builds the fully-convolutional segmentation net. When AddDiceHead is true the
// stack ends with TNNetDiceLoss; otherwise the sigmoid is the output layer
// (plain MSE-gradient baseline).
function BuildNet(AddDiceHead: boolean): TNNet;
begin
  Result := TNNet.Create();
  Result.AddLayer(TNNetInput.Create(cGrid, cGrid, 1));
  // featuresize 3, inputpadding 1, stride 1 => spatial size preserved.
  Result.AddLayer(TNNetConvolutionReLU.Create(cFeatures, 3, 1, 1));
  Result.AddLayer(TNNetConvolutionReLU.Create(cFeatures, 3, 1, 1));
  Result.AddLayer(TNNetConvolutionLinear.Create(1, 3, 1, 1));
  Result.AddLayer(TNNetSigmoid.Create());
  if AddDiceHead then
    Result.AddLayer(TNNetDiceLoss.Create());
  Result.InitWeights();
end;

// Returns the per-pixel foreground-probability map of the last layer's output.
// (For both nets the last-layer output is the sigmoid probability, since the
// Dice head is an identity passthrough.)
function ForegroundProb(Net: TNNet; X, Y: integer): TNeuralFloat;
begin
  Result := Net.GetLastLayer().Output[X, Y, 0];
end;

// One training epoch over a random permutation-free shuffle of the dataset.
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

// Computes mean Dice and mean IoU on a dataset (prediction thresholded at 0.5).
procedure Evaluate(Net: TNNet; const Imgs, Masks: TVolArray;
  out MeanDice, MeanIoU: TNeuralFloat);
var
  I, X, Y, N: integer;
  Inter, PredFg, GtFg, Union: integer;
  P, G: TNeuralFloat;
  SumDice, SumIoU: TNeuralFloat;
begin
  N := Length(Imgs);
  SumDice := 0;
  SumIoU := 0;
  for I := 0 to N - 1 do
  begin
    Net.Compute(Imgs[I]);
    Inter := 0;
    PredFg := 0;
    GtFg := 0;
    for Y := 0 to cGrid - 1 do
      for X := 0 to cGrid - 1 do
      begin
        P := ForegroundProb(Net, X, Y);
        G := Masks[I][X, Y, 0];
        if P >= cThreshold then Inc(PredFg);
        if G >= 0.5 then Inc(GtFg);
        if (P >= cThreshold) and (G >= 0.5) then Inc(Inter);
      end;
    Union := PredFg + GtFg - Inter;
    // Smoothing keeps the metric defined on empty masks (none occur here).
    SumDice := SumDice + (2 * Inter + 1) / (PredFg + GtFg + 1);
    SumIoU := SumIoU + (Inter + 1) / (Union + 1);
  end;
  MeanDice := SumDice / N;
  MeanIoU := SumIoU / N;
end;

// Renders one held-out sample as ASCII for visual sanity checking.
procedure RenderSample(DiceNet, MseNet: TNNet; const Imgs, Masks: TVolArray;
  Idx: integer);
var
  X, Y: integer;

  function InputChar(V: TNeuralFloat): char;
  begin
    if V >= 0.5 then InputChar := '#'
    else if V >= 0.2 then InputChar := '+'
    else if V <= -0.2 then InputChar := ',' // negative noise
    else InputChar := '.';
  end;

  function MaskChar(V: TNeuralFloat): char;
  begin
    if V >= 0.5 then MaskChar := '#' else MaskChar := '.';
  end;

  function PredChar(V: TNeuralFloat): char;
  begin
    if V >= cThreshold then PredChar := '#' else PredChar := '.';
  end;

begin
  WriteLn;
  WriteLn('One held-out sample  ("#" = foreground, threshold 0.5)');
  WriteLn('  col 1: noisy INPUT   col 2: GROUND TRUTH   ',
          'col 3: DICE pred   col 4: MSE pred');
  WriteLn;
  DiceNet.Compute(Imgs[Idx]);
  // Cache Dice prediction before recomputing with the MSE net.
  for Y := 0 to cGrid - 1 do
  begin
    // input
    for X := 0 to cGrid - 1 do Write(InputChar(Imgs[Idx][X, Y, 0]));
    Write('   ');
    // ground truth
    for X := 0 to cGrid - 1 do Write(MaskChar(Masks[Idx][X, Y, 0]));
    Write('   ');
    // dice prediction
    for X := 0 to cGrid - 1 do
      Write(PredChar(ForegroundProb(DiceNet, X, Y)));
    WriteLn;
  end;
  // MSE column rendered on a second pass so both predictions are correct.
  MseNet.Compute(Imgs[Idx]);
  WriteLn;
  WriteLn('  MSE prediction:');
  for Y := 0 to cGrid - 1 do
  begin
    Write('  ');
    for X := 0 to cGrid - 1 do
      Write(PredChar(ForegroundProb(MseNet, X, Y)));
    WriteLn;
  end;
end;

procedure RunAlgo();
var
  DiceNet, MseNet: TNNet;
  TrainImgs, TrainMasks, TestImgs, TestMasks: TVolArray;
  Epoch: integer;
  DiceDice, DiceIoU, MseDice, MseIoU: TNeuralFloat;
begin
  RandSeed := cSeed;
  WriteLn('DiceSegmentation: TNNetDiceLoss vs MSE on a synthetic disc-mask task');
  WriteLn(Format('grid=%dx%d  features=%d  train=%d  test=%d  epochs=%d  lr=%.3f',
    [cGrid, cGrid, cFeatures, cTrainSamples, cTestSamples, cEpochs, cLearningRate]));
  WriteLn;

  // Same RandSeed before each dataset/net build keeps the comparison fair:
  // identical data and identical initial weights for both nets.
  RandSeed := cSeed;
  BuildDataset(TrainImgs, TrainMasks, cTrainSamples);
  BuildDataset(TestImgs, TestMasks, cTestSamples);

  try
    // --- Dice-loss net ---
    RandSeed := cSeed + 1;
    DiceNet := BuildNet(True);
    DiceNet.SetLearningRate(cLearningRate, 0.9);
    WriteLn('Training Dice-loss net...');
    for Epoch := 1 to cEpochs do
    begin
      TrainEpoch(DiceNet, TrainImgs, TrainMasks);
      if (Epoch = 1) or (Epoch mod 10 = 0) then
      begin
        Evaluate(DiceNet, TestImgs, TestMasks, DiceDice, DiceIoU);
        WriteLn(Format('  epoch %3d   test Dice=%6.4f  IoU=%6.4f',
          [Epoch, DiceDice, DiceIoU]));
      end;
    end;

    // --- MSE baseline net (identical init) ---
    RandSeed := cSeed + 1;
    MseNet := BuildNet(False);
    MseNet.SetLearningRate(cLearningRate, 0.9);
    WriteLn('Training MSE-baseline net...');
    for Epoch := 1 to cEpochs do
    begin
      TrainEpoch(MseNet, TrainImgs, TrainMasks);
      if (Epoch = 1) or (Epoch mod 10 = 0) then
      begin
        Evaluate(MseNet, TestImgs, TestMasks, MseDice, MseIoU);
        WriteLn(Format('  epoch %3d   test Dice=%6.4f  IoU=%6.4f',
          [Epoch, MseDice, MseIoU]));
      end;
    end;

    Evaluate(DiceNet, TestImgs, TestMasks, DiceDice, DiceIoU);
    Evaluate(MseNet, TestImgs, TestMasks, MseDice, MseIoU);

    WriteLn;
    WriteLn('Held-out results (threshold 0.5)');
    WriteLn('  head           mean Dice    mean IoU');
    WriteLn(Format('  TNNetDiceLoss   %8.4f    %8.4f', [DiceDice, DiceIoU]));
    WriteLn(Format('  MSE baseline    %8.4f    %8.4f', [MseDice, MseIoU]));
    WriteLn;
    if (DiceIoU > MseIoU + 1e-4) then
      WriteLn('=> The Dice-loss head reaches a HIGHER held-out IoU than MSE.')
    else if (MseIoU > DiceIoU + 1e-4) then
      WriteLn('=> On this toy the MSE baseline matched or beat the Dice head.')
    else
      WriteLn('=> The two heads tied on held-out IoU on this toy.');

    RenderSample(DiceNet, MseNet, TestImgs, TestMasks, 0);
  finally
    FreeDataset(TrainImgs, TrainMasks);
    FreeDataset(TestImgs, TestMasks);
    DiceNet.Free;
    MseNet.Free;
  end;
end;

var
  // Stops Lazarus errors
  Application: record Title: string; end;

begin
  Application.Title := 'DiceSegmentation Example';
  RunAlgo();
end.
