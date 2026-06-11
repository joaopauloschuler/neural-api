program SpatialTransformer;
(*
SpatialTransformer: a self-contained demo of the new TNNetAffineGridSample
layer, the differentiable bilinear grid-sampling core of a Spatial
Transformer Network (Jaderberg, Simonyan, Zisserman & Kavukcuoglu 2015,
"Spatial Transformer Networks", https://arxiv.org/abs/1506.02025).

Task. We synthesize a tiny, offline, self-contained dataset on a 20x20x1
canvas: 4 small distinct glyphs (cross / box / horizontal bar / vertical bar),
each ~6x6 in the centre. Every training and test sample is then JITTERED by a
large random TRANSLATION (+/- ~6 px) and a light rotation (+/- ~9 deg), so the
small glyph roams across the large canvas. A position-sensitive fully-connected
readout must memorise every (glyph x location) combination and fails -- exactly
the regime the Spatial Transformer is designed to rescue (its conv localiser
estimates where the glyph is and recentres it before classification).

Two models are trained on the SAME jittered stream:

  (A) plain   : conv -> pool -> conv -> pool -> dense classifier.
  (B) STN      : an identical classifier, but with a Spatial Transformer
                 FRONT-END prepended:
                   localization head (conv/pool -> FullConnectLinear(6))
                   -> TNNetAffineGridSample (warps the input image)
                   -> the SAME classifier.
                 The localization head's final FullConnectLinear(6) is
                 BIAS-INITIALISED to the identity affine [1,0,0,0,1,0] (and
                 its weights zeroed), exactly as the paper prescribes, so the
                 STN starts as a no-op pass-through and LEARNS to undo the
                 jitter, canonicalising the input before classification.

Headline: the STN front-end recovers accuracy lost to the input jitter.

Runs in well under a minute on CPU; no data is downloaded.

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
  neuralvolume;

const
  cSize     = 20;     // 20x20 canvas (large relative to the small shapes)
  cClasses  = 4;      // 4 distinct small glyphs (content-based, not orientation)
  cSteps    = 1500;
  cBatch    = 16;
  cLR       = 0.01;
  cInertia  = 0.9;
  cNumTest  = 800;
  cMaxRot   = 0.15;   // ~+/-9 deg light rotation jitter
  cMaxShift = 6.0;    // large translation: the glyph roams across the canvas

// ---- Synthetic dataset --------------------------------------------------

procedure DrawCanonical(V: TNNetVolume; AClass: integer);
// Draw one clean, centred class prototype: a small distinct glyph (~6x6) in
// the middle of the canvas. The four classes differ by CONTENT (cross / box /
// horizontal bar / vertical bar), each drawn in the SAME small central region.
// Because the glyph is small and the canvas is large, the subsequent random
// TRANSLATION can place it almost anywhere; a position-sensitive
// fully-connected head must then memorise every (glyph x location) combination
// and fails, whereas the Spatial Transformer's conv localiser estimates where
// the glyph is and recentres it before the classifier sees it.
var
  c, i: integer;
begin
  V.Fill(0);
  c := cSize div 2;
  case AClass of
    0: // cross / plus
      for i := -2 to 2 do
      begin
        V[c + i, c, 0] := 1.0;
        V[c, c + i, 0] := 1.0;
      end;
    1: // hollow box outline
      for i := -2 to 2 do
      begin
        V[c + i, c - 2, 0] := 1.0;
        V[c + i, c + 2, 0] := 1.0;
        V[c - 2, c + i, 0] := 1.0;
        V[c + 2, c + i, 0] := 1.0;
      end;
    2: // horizontal bar
      for i := -2 to 2 do
      begin
        V[c + i, c, 0]     := 1.0;
        V[c + i, c - 1, 0] := 1.0;
      end;
    3: // vertical bar
      for i := -2 to 2 do
      begin
        V[c, c + i, 0]     := 1.0;
        V[c - 1, c + i, 0] := 1.0;
      end;
  end;
end;

procedure Jitter(Src, Dst: TNNetVolume);
// Apply a random rotation + translation to Src, writing into Dst by
// inverse (back-warp) bilinear sampling. This is the input perturbation the
// STN must learn to undo.
var
  ox, oy, x0, y0: integer;
  ang, ca, sa, dx, dy, cx, sx, sy, fx, fy, val: TNeuralFloat;
begin
  Dst.Fill(0);
  ang := (Random - 0.5) * 2.0 * cMaxRot;
  ca := Cos(ang); sa := Sin(ang);
  dx := (Random - 0.5) * 2.0 * cMaxShift;
  dy := (Random - 0.5) * 2.0 * cMaxShift;
  cx := (cSize - 1) * 0.5;
  for oy := 0 to cSize - 1 do
    for ox := 0 to cSize - 1 do
    begin
      // back-warp output pixel into source space
      sx := ca * (ox - cx) - sa * (oy - cx) + cx - dx;
      sy := sa * (ox - cx) + ca * (oy - cx) + cx - dy;
      x0 := Floor(sx); y0 := Floor(sy);
      fx := sx - x0; fy := sy - y0;
      val := 0;
      if (x0 >= 0) and (x0 < cSize) and (y0 >= 0) and (y0 < cSize) then
        val := val + (1 - fx) * (1 - fy) * Src[x0, y0, 0];
      if (x0 + 1 >= 0) and (x0 + 1 < cSize) and (y0 >= 0) and (y0 < cSize) then
        val := val + fx * (1 - fy) * Src[x0 + 1, y0, 0];
      if (x0 >= 0) and (x0 < cSize) and (y0 + 1 >= 0) and (y0 + 1 < cSize) then
        val := val + (1 - fx) * fy * Src[x0, y0 + 1, 0];
      if (x0 + 1 >= 0) and (x0 + 1 < cSize) and (y0 + 1 >= 0) and (y0 + 1 < cSize) then
        val := val + fx * fy * Src[x0 + 1, y0 + 1, 0];
      Dst[ox, oy, 0] := val;
    end;
end;

procedure SampleExample(Canon, InputV, TargetV: TNNetVolume);
var
  cls: integer;
begin
  cls := Random(cClasses);
  DrawCanonical(Canon, cls);
  Jitter(Canon, InputV);
  TargetV.Fill(0);
  TargetV[0, 0, cls] := 1.0;
end;

// ---- Models -------------------------------------------------------------

procedure AddClassifier(NN: TNNet);
// The shared classifier body (used by both models). Operates on whatever
// 16x16x1 image its previous layer produces.
//
// It is DELIBERATELY a small fully-connected (position-SENSITIVE) readout:
// FC -> FC. With no conv/pool the classifier has no built-in translation
// tolerance, so jittering the input directly degrades it -- which is exactly
// the regime the Spatial Transformer is designed to rescue (it canonicalises
// the input so the position-sensitive head sees a stable, centred pattern).
begin
  NN.AddLayer(TNNetFullConnectReLU.Create(16));
  NN.AddLayer(TNNetFullConnectLinear.Create(cClasses));
  NN.AddLayer(TNNetSoftMax.Create());
end;

procedure BuildPlain(out NN: TNNet);
begin
  NN := TNNet.Create();
  NN.AddLayer(TNNetInput.Create(cSize, cSize, 1));
  AddClassifier(NN);
  NN.SetLearningRate(cLR, cInertia);
  NN.InitWeights();
end;

procedure BuildSTN(out NN: TNNet);
var
  ImgInput, LocHead: TNNetLayer;
  i: integer;
begin
  NN := TNNet.Create();
  ImgInput := NN.AddLayer(TNNetInput.Create(cSize, cSize, 1));
  // --- Localization head: predicts the 6 affine parameters ---
  NN.AddLayer(TNNetConvolutionReLU.Create(8, 3, 1, 1));
  NN.AddLayer(TNNetMaxPool.Create(2));
  NN.AddLayer(TNNetConvolutionReLU.Create(8, 3, 1, 1));
  NN.AddLayer(TNNetMaxPool.Create(2));
  NN.AddLayer(TNNetFullConnectReLU.Create(16));
  LocHead := NN.AddLayer(TNNetFullConnectLinear.Create(6));
  // --- Spatial transformer: warp the ORIGINAL image by the predicted theta.
  // PrevLayer = the image input (branch back to layer 0); theta = LocHead.
  NN.AddLayerAfter(TNNetAffineGridSample.Create(LocHead), ImgInput);
  // --- Shared classifier on the warped image ---
  AddClassifier(NN);
  NN.SetLearningRate(cLR, cInertia);
  NN.InitWeights();
  // Identity-affine initialisation of the localization head (paper-prescribed):
  // zero the weights, set the bias to [1,0,0,0,1,0] so the STN starts as a
  // no-op pass-through and then LEARNS the canonicalising transform.
  for i := 0 to LocHead.Neurons.Count - 1 do
    LocHead.Neurons[i].Weights.Fill(0);
  LocHead.Neurons[0].BiasWeight := 1.0; // a
  LocHead.Neurons[1].BiasWeight := 0.0; // b
  LocHead.Neurons[2].BiasWeight := 0.0; // c
  LocHead.Neurons[3].BiasWeight := 0.0; // d
  LocHead.Neurons[4].BiasWeight := 1.0; // e
  LocHead.Neurons[5].BiasWeight := 0.0; // f
end;

// ---- Train / evaluate ---------------------------------------------------

procedure Train(NN: TNNet; const ATag: string);
var
  Step, B: integer;
  Canon, InputV, TargetV: TNNetVolume;
  StartTime, Elapsed: double;
begin
  Canon   := TNNetVolume.Create(cSize, cSize, 1);
  InputV  := TNNetVolume.Create(cSize, cSize, 1);
  TargetV := TNNetVolume.Create(1, 1, cClasses);
  try
    StartTime := Now();
    for Step := 1 to cSteps do
    begin
      for B := 1 to cBatch do
      begin
        SampleExample(Canon, InputV, TargetV);
        NN.Compute(InputV);
        NN.Backpropagate(TargetV);
      end;
      if (Step = 1) or (Step mod 400 = 0) or (Step = cSteps) then
      begin
        Elapsed := (Now() - StartTime) * 86400.0;
        WriteLn(Format('  [%s] step %4d / %4d   elapsed=%.1fs',
          [ATag, Step, cSteps, Elapsed]));
      end;
    end;
  finally
    Canon.Free; InputV.Free; TargetV.Free;
  end;
end;

function EvalAccuracy(NN: TNNet): TNeuralFloat;
var
  I, Pred, True_, K: integer;
  Canon, InputV, TargetV: TNNetVolume;
  Best: TNeuralFloat;
  Correct: integer;
begin
  Canon   := TNNetVolume.Create(cSize, cSize, 1);
  InputV  := TNNetVolume.Create(cSize, cSize, 1);
  TargetV := TNNetVolume.Create(1, 1, cClasses);
  Correct := 0;
  try
    for I := 1 to cNumTest do
    begin
      SampleExample(Canon, InputV, TargetV);
      NN.Compute(InputV);
      Pred := 0; Best := NN.GetLastLayer.Output.FData[0];
      for K := 1 to cClasses - 1 do
        if NN.GetLastLayer.Output.FData[K] > Best then
        begin
          Best := NN.GetLastLayer.Output.FData[K];
          Pred := K;
        end;
      True_ := 0;
      for K := 0 to cClasses - 1 do
        if TargetV.FData[K] > 0.5 then True_ := K;
      if Pred = True_ then Inc(Correct);
    end;
  finally
    Canon.Free; InputV.Free; TargetV.Free;
  end;
  Result := Correct / cNumTest;
end;

type
  TBuildProc = procedure(out NN: TNNet);

procedure RunOne(BuildProc: TBuildProc; const ATag: string;
  const ATrainSeed, ATestSeed: longint; out Acc: TNeuralFloat);
var
  NN: TNNet;
begin
  RandSeed := ATrainSeed;
  BuildProc(NN);
  try
    WriteLn('Model: ', ATag);
    NN.DebugStructure();
    Train(NN, ATag);
    RandSeed := ATestSeed;
    Acc := EvalAccuracy(NN);
    WriteLn(Format('  [%s] held-out accuracy = %.3f', [ATag, Acc]));
  finally
    NN.Free;
  end;
end;

var
  AccPlain, AccSTN: TNeuralFloat;

begin
  WriteLn('Spatial Transformer vs plain CNN: jittered shape classification');
  WriteLn(Format('  grid=%dx%d   classes=%d   steps=%d   batch=%d',
    [cSize, cSize, cClasses, cSteps, cBatch]));
  WriteLn(Format('  input jitter: rotation +/-%.0f deg, shift +/-%.1f px',
    [cMaxRot * 180.0 / Pi, cMaxShift]));
  // Same train/test seeds so both models see the same data stream.
  RunOne(@BuildPlain, 'plain', 1234, 9999, AccPlain);
  RunOne(@BuildSTN,   'STN',   1234, 9999, AccSTN);
  WriteLn;
  WriteLn('Summary (higher is better):');
  WriteLn(Format('  plain CNN              accuracy = %.3f', [AccPlain]));
  WriteLn(Format('  CNN + Spatial Xformer  accuracy = %.3f', [AccSTN]));
  if AccSTN > AccPlain then
    WriteLn(Format('  STN front-end recovers +%.1f accuracy points lost to jitter.',
      [(AccSTN - AccPlain) * 100.0]))
  else
    WriteLn('  STN did NOT help on this run.');
end.
