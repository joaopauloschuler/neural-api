program ImageNetEval;
(*
ImageNetEval: an end-to-end demo of the ImageNet top-1 / top-5 ACCURACY harness
(EvaluateImageNet / ImageNetReport in neuralimagemetrics.pas). This is to the
imported vision backbones (ResNet / ViT / Swin / DINOv2 / MobileNetV3 / VGG /
Inception-v3 / EfficientNet) what examples/MMLUEval is to the imported LLMs: the
missing import-VERIFICATION backstop. Each classifier importer's parity test
only compares raw logits on one or two tensors, which catches a transposed
weight but NOT a wrong preprocessing pipeline (resize / center-crop / normalize)
or a label permutation. Running a folder of labelled ImageNet-val images through
the REAL preprocessing transform (neuraldatasets.PreprocessImageForVisionModel:
shorter-side resize -> center-crop -> (x/255 - mean)/std) and the REAL net, then
checking top-1 / top-5 against the published numbers, is exactly that backstop.

TWO MODES
---------
(1) DEFAULT SMOKE (no arguments). To stay within a tiny CPU / memory budget (no
    multi-GB checkpoint download, no network fetch, no real ImageNet) this builds
    a small CNN, trains it for a few epochs on a TINY DETERMINISTIC synthetic
    "dataset" of per-class coloured patterns (fixed RandSeed), and evaluates it
    with the harness. Each synthetic image is rendered at a larger size and run
    through the SAME PreprocessImageForVisionModel transform the real path uses
    (shorter-side resize -> center-crop -> ImageNet mean/std), so the smoke
    exercises the real transform + the real harness end to end. The point is the
    HARNESS + TRANSFORM MECHANICS, not a real accuracy number. Runs in seconds,
    well under the 3 GB ulimit.

(2) --full <dir>  (documented real-ImageNet hook, see README.md). Loads a folder
    of labelled JPEG/PNG images, applies the chosen importer's declared
    ImageSize + csImageNetMean/csImageNetStd, runs an imported backbone, and
    reports top-1 / top-5 over real ImageNet-val. The folder layout and how to
    point it at a real checkpoint are documented in README.md. This binary ships
    the SMOKE wired; --full prints the exact recipe (it deliberately does NOT
    bundle a multi-GB importer call so CI stays self-contained).

The harness itself is checkpoint-agnostic: swap BuildSmokeClassifier for a
BuildResNetFromSafeTensors (or any classifier importer) and feed
LoadImageForVisionModel-produced volumes into the same TNNetImageNetSample
records; EvaluateImageNet / ImageNetReport are unchanged.

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
  neuraldatasets,
  neuralfit,
  neuralimagemetrics;

const
  cNumClasses = 6;          // tiny synthetic class count
  cRenderSize = 64;         // synthetic images are rendered at 64x64 ...
  cResizeSide = 32;         // ... resized (shorter side) to 32 ...
  cCropSize   = 24;         // ... and center-cropped to the net's 24x24 input
  cTrainPerClass = 24;      // training images per class
  cEvalPerClass  = 6;       // held-out eval images per class

// Renders a deterministic synthetic RGB image (cRenderSize^2, byte 0..255) for
// class ClassIdx with sample index N: a class-specific base colour plus a small
// per-sample jitter, so classes are linearly separable but not trivial. The
// returned volume is in the 0..255 byte-valued layout LoadImageFromFileIntoVolume
// produces, i.e. exactly what PreprocessImageForVisionModel expects as Src.
function RenderSynthetic(ClassIdx, N: integer): TNNetVolume;
var
  x, y, c: integer;
  baseR, baseG, baseB, jit: integer;
  v: integer;
begin
  Result := TNNetVolume.Create(cRenderSize, cRenderSize, 3);
  // Class base colour spread across the cube so argmax is learnable.
  baseR := (ClassIdx * 40 + 30) mod 256;
  baseG := (ClassIdx * 70 + 60) mod 256;
  baseB := (ClassIdx * 95 + 90) mod 256;
  // Deterministic per-sample jitter (no RNG: reproducible across runs).
  jit := ((ClassIdx * 13 + N * 7) mod 31) - 15;
  for y := 0 to cRenderSize - 1 do
    for x := 0 to cRenderSize - 1 do
    begin
      for c := 0 to 2 do
      begin
        case c of
          0: v := baseR + jit + (x mod 5);
          1: v := baseG + jit + (y mod 5);
        else v := baseB + jit + ((x + y) mod 5);
        end;
        if v < 0 then v := 0;
        if v > 255 then v := 255;
        Result[x, y, c] := v;
      end;
    end;
end;

// Build (and return) a list of preprocessed (cCropSize, cCropSize, 3) volumes +
// a parallel label list, by rendering PerClass synthetic images per class and
// pushing each through the REAL PreprocessImageForVisionModel transform with
// ImageNet mean/std. Offset shifts the per-sample index so train / eval sets are
// disjoint.
procedure BuildSet(PerClass, Offset: integer; Vols: TNNetVolumeList);
var
  ci, n: integer;
  Raw, Proc: TNNetVolume;
begin
  for ci := 0 to cNumClasses - 1 do
    for n := 0 to PerClass - 1 do
    begin
      Raw := RenderSynthetic(ci, Offset + n);
      Proc := TNNetVolume.Create;
      PreprocessImageForVisionModel(Raw, Proc, cResizeSide, cCropSize,
        csImageNetMean, csImageNetStd);
      Raw.Free;
      Proc.Tag := ci;   // class label carried on the volume (TNeuralImageFit)
      Vols.Add(Proc);
    end;
end;

// A small CNN classifier over the (cCropSize, cCropSize, 3) preprocessed input.
function BuildSmokeClassifier: TNNet;
begin
  Result := TNNet.Create;
  Result.AddLayer(TNNetInput.Create(cCropSize, cCropSize, 3));
  Result.AddLayer(TNNetConvolutionReLU.Create(16, 3, 1, 1, 1));
  Result.AddLayer(TNNetMaxPool.Create(2));
  Result.AddLayer(TNNetConvolutionReLU.Create(24, 3, 1, 1, 1));
  Result.AddLayer(TNNetMaxPool.Create(2));
  Result.AddLayer(TNNetFullConnectReLU.Create(32));
  Result.AddLayer(TNNetFullConnectLinear.Create(cNumClasses));
  Result.AddLayer(TNNetSoftMax.Create);
end;

procedure RunSmoke;
var
  NN: TNNet;
  TrainVols, EvalVols: TNNetVolumeList;
  Fit: TNeuralImageFit;
  i: integer;
  Samples: TNNetImageNetSampleArray;
  Stats: TNNetImageNetStats;
  Names: array of string;
begin
  WriteLn('ImageNetEval SMOKE: tiny synthetic ', cNumClasses, '-class set, ',
    'real resize/crop/normalize transform, real top-1/top-5 harness.');
  WriteLn;

  // --- deterministic synthetic train / eval sets (disjoint sample indices) ---
  // Each volume carries its class on .Tag, the TNeuralImageFit label convention.
  TrainVols := TNNetVolumeList.Create(true);
  EvalVols := TNNetVolumeList.Create(true);
  BuildSet(cTrainPerClass, 0, TrainVols);
  BuildSet(cEvalPerClass, 1000, EvalVols);

  // --- train the smoke classifier briefly (fixed seed, bounded epochs) ---
  RandSeed := 424242;
  NN := BuildSmokeClassifier;
  Fit := TNeuralImageFit.Create;
  try
    Fit.InitialLearningRate := 0.01;
    Fit.LearningRateDecay := 0;
    Fit.L2Decay := 0;
    Fit.HasImgCrop := false;
    Fit.HasFlipX := false;
    Fit.HasFlipY := false;
    Fit.MaxThreadNum := 1;
    Fit.Fit(NN, TrainVols, nil, nil, cNumClasses, 16, 30);
  finally
    Fit.Free;
  end;

  // --- evaluate with the ImageNet harness over the held-out set ---
  SetLength(Samples, EvalVols.Count);
  for i := 0 to EvalVols.Count - 1 do
  begin
    Samples[i].Image := EvalVols[i];
    Samples[i].GoldLabel := EvalVols[i].Tag;
    Samples[i].SourceName := Format('class%d_%d.png', [EvalVols[i].Tag, i]);
  end;

  SetLength(Names, cNumClasses);
  for i := 0 to cNumClasses - 1 do Names[i] := 'synthetic-' + IntToStr(i);

  Stats := EvaluateImageNet(NN, Samples, cNumClasses, 5, 8);
  WriteLn(ImageNetReport(Stats, Names, 'SmokeCNN synthetic-val'));

  NN.Free;
  TrainVols.Free;
  EvalVols.Free;
end;

procedure PrintFullHookHelp(const Dir: string);
begin
  WriteLn('ImageNetEval --full <dir>: real ImageNet-val hook (documented recipe)');
  WriteLn;
  WriteLn('Requested directory : ', Dir);
  WriteLn;
  WriteLn('This binary ships the SMOKE wired so CI stays self-contained (no');
  WriteLn('multi-GB importer call is bundled). To run real ImageNet-val:');
  WriteLn;
  WriteLn('  1. Lay out <dir> as documented in examples/README.md:');
  WriteLn('       <dir>/labels.txt         one "<filename> <class_index>" per line');
  WriteLn('       <dir>/<filename>.JPEG     the ImageNet-val JPEGs');
  WriteLn('  2. Import a classifier backbone, e.g.');
  WriteLn('       NN := BuildResNetFromSafeTensors(''resnet50.safetensors'', Cfg);');
  WriteLn('     and read its declared ImageSize + csImageNetMean / csImageNetStd.');
  WriteLn('  3. For each labelled file:');
  WriteLn('       LoadImageForVisionModel(File, V, ResizeSide, ImageSize,');
  WriteLn('         csImageNetMean, csImageNetStd);');
  WriteLn('     and fill a TNNetImageNetSample (Image := V; GoldLabel := idx).');
  WriteLn('  4. Stats := EvaluateImageNet(NN, Samples, 1000, 5);');
  WriteLn('     WriteLn(ImageNetReport(Stats, ClassNames, ''ResNet-50 ImageNet-val''));');
  WriteLn;
  WriteLn('See examples/README.md (ImageNetEval section) for the full recipe.');
end;

var
  Arg1: string;
begin
  WriteLn('====================================================================');
  WriteLn(' ImageNetEval - top-1 / top-5 accuracy harness (import verification)');
  WriteLn('====================================================================');
  WriteLn;
  if ParamCount >= 1 then
    Arg1 := ParamStr(1)
  else
    Arg1 := '';

  if Arg1 = '--full' then
  begin
    if ParamCount >= 2 then
      PrintFullHookHelp(ParamStr(2))
    else
      PrintFullHookHelp('(none given)');
  end
  else
    RunSmoke;
end.
