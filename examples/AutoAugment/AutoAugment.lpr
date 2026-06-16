program AutoAugment;
(*
AutoAugment: demonstrates the RandAugment / TrivialAugment automatic
augmentation policy (neuraldatasets) wired into the TNeuralImageFit
augmentation pipeline through the optional DataAugmentationFn hook.

The op bank (autocontrast, equalize, rotate, shear-x/y, translate-x/y,
posterize, solarize, color/contrast/brightness/sharpness) plus RandomErasing
operate IN PLACE on a TNNetVolume image in the library's neuronal [-2..2]
domain. RandAugment applies N ops at a fixed magnitude M; TrivialAugment
applies one op at a magnitude drawn uniformly. Both keep the default
flip+pad-crop pipeline intact and simply layer on top of it.

This toy builds a tiny synthetic image dataset (each class is a distinct
colored shape + noise) and trains the SAME small convolutional classifier
WITHOUT and WITH the TrivialAugment policy, printing a short comparison so the
opt-in path is exercised end to end on pure CPU in a few seconds.

On real CIFAR-10 (see examples/SimpleImageClassifier) enabling RandAugment or
TrivialAugment plus RandomErasing is expected to give a small but consistent
top-1 lift over plain flip+crop, the same qualitative effect reported by the
torchvision transforms-v2 originals.

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
  Classes, SysUtils, Math,
  neuralnetwork, neuralvolume, neuraldatasets, neuralfit, neuralthread;

const
  cImgSize = 16;
  cClasses = 3;

// Builds a synthetic labelled image: each class draws a distinct colored
// rectangle on a noisy background, then converts to the neuronal domain.
procedure MakeSample(V, Y: TNNetVolume; ClassId: integer);
var
  x, y2, d, x0, y0: integer;
  baseR, baseG, baseB: integer;
begin
  V.ReSize(cImgSize, cImgSize, 3);
  Y.ReSize(1, 1, cClasses);
  Y.Fill(0);
  Y[0, 0, ClassId] := 1;
  // Noisy gray background in 0..255 pixel space.
  for d := 0 to 2 do
    for y2 := 0 to cImgSize - 1 do
      for x := 0 to cImgSize - 1 do
        V[x, y2, d] := 110 + Random(40);
  // Distinct colored block per class at a class-dependent location.
  baseR := 40; baseG := 40; baseB := 40;
  case ClassId of
    0: baseR := 220;
    1: baseG := 220;
    2: baseB := 220;
  end;
  x0 := 2 + ClassId * 3;
  y0 := 2 + ClassId * 3;
  for y2 := y0 to y0 + 6 do
    for x := x0 to x0 + 6 do
    begin
      V[x, y2, 0] := baseR;
      V[x, y2, 1] := baseG;
      V[x, y2, 2] := baseB;
    end;
  // Convert pixel domain -> neuronal [-2..2].
  V.RgbImgToNeuronalInput(csEncodeRGB);
end;

procedure BuildDataset(L: TNNetVolumeList; Count: integer);
var
  i, c: integer;
  V, Y: TNNetVolume;
begin
  for i := 0 to Count - 1 do
  begin
    c := i mod cClasses;
    V := TNNetVolume.Create();
    Y := TNNetVolume.Create();
    MakeSample(V, Y, c);
    V.Tag := c;
    // TNeuralImageFit consumes a list of input volumes whose Tag holds the
    // class id (the standard CIFAR convention in this library).
    L.Add(V);
    Y.Free;
  end;
end;

function BuildNet: TNNet;
begin
  Result := TNNet.Create();
  Result.AddLayer([
    TNNetInput.Create(cImgSize, cImgSize, 3),
    TNNetConvolutionReLU.Create(16, 3, 1, 1, 0),
    TNNetMaxPool.Create(2),
    TNNetConvolutionReLU.Create(16, 3, 1, 1, 0),
    TNNetMaxPool.Create(2),
    TNNetFullConnectLinear.Create(cClasses),
    TNNetSoftMax.Create()
  ]);
end;

function RunTraining(UsePolicy: boolean): TNeuralFloat;
var
  NN: TNNet;
  Fit: TNeuralImageFit;
  Train, Valid, TestL: TNNetVolumeList;
  Pol: TNeuralAugmentationPolicy;
begin
  RandSeed := 1234;
  Train := TNNetVolumeList.Create();
  Valid := TNNetVolumeList.Create();
  TestL := TNNetVolumeList.Create();
  BuildDataset(Train, 300);
  BuildDataset(Valid, 60);
  BuildDataset(TestL, 60);

  NN := BuildNet();
  Fit := TNeuralImageFit.Create();
  Pol := nil;
  try
    Fit.FileNameBase := 'AutoAugment-' + IntToStr(GetProcessId());
    Fit.InitialLearningRate := 0.001;
    Fit.LearningRateDecay := 0;
    Fit.L2Decay := 0;
    Fit.HasFlipX := true;
    Fit.HasFlipY := false;
    Fit.MaxCropSize := 4;
    Fit.Verbose := false;
    if UsePolicy then
    begin
      // Opt-in: layer TrivialAugment + RandomErasing ON TOP of flip+crop.
      Pol := TNeuralAugmentationPolicy.Create(napTrivialAugment, 2, 9, 0.25);
      Fit.ImageAugmentationFn := @Pol.Augment;
    end;
    Fit.Fit(NN, Train, Valid, TestL, cClasses, 32, 8);
    Result := Fit.TestAccuracy;
  finally
    Fit.Free;
    NN.Free;
    if Assigned(Pol) then Pol.Free;
    Train.Free; Valid.Free; TestL.Free;
  end;
end;

var
  AccPlain, AccPolicy: TNeuralFloat;
begin
  WriteLn('AutoAugment (RandAugment / TrivialAugment) policy demo');
  WriteLn('Training WITHOUT augmentation policy (flip+crop only)...');
  AccPlain := RunTraining(false);
  WriteLn('  test accuracy: ', (AccPlain * 100):0:2, '%');
  WriteLn('Training WITH TrivialAugment + RandomErasing policy...');
  AccPolicy := RunTraining(true);
  WriteLn('  test accuracy: ', (AccPolicy * 100):0:2, '%');
  WriteLn('Done. (Policy wired via TNeuralImageFit.ImageAugmentationFn hook.)');
end.
