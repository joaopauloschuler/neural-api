program GlimpseDownsampler;
(*
GlimpseDownsampler: a "hard" visual-attention GLIMPSE built on the landed
TNNetAffineGridSample bilinear grid sampler (the differentiable core of a
Spatial Transformer Network, Jaderberg et al. 2015,
https://arxiv.org/abs/1506.02025). This is the scale+translate-restricted
follow-up to examples/SpatialTransformer.

Idea. A small bright digit-like glyph (~6x6, 4 content classes: cross / box /
horizontal bar / vertical bar) is dropped at a RANDOM offset onto a larger,
CLUTTERED 28x28 canvas (sprinkled with bright noise specks). A small,
position-sensitive classifier must read a SMALL 14x14 canonical patch. The
question: how is that 14x14 patch produced?

  (A) LEARNED GLIMPSE downsampler:
        input 28x28
        -> localization head (conv/pool -> FullConnectLinear(4))
             emits (s_x, s_y, t_x, t_y)  -- WHERE and how much to zoom
        -> TNNetScatterToAffine  -- scatter to the 2x3 affine
             theta = [ s_x  0  t_x ; 0  s_y  t_y ]  (shear/rotation HARD 0)
        -> TNNetAffineGridSample -- learned crop/zoom over the 28x28 input
        -> TNNetAvgPool(2)       -- 28x28 -> 14x14 canonical patch
        -> shared classifier
      The head is identity-of-crop initialised (a fixed central zoom, no
      translation) and LEARNS to TRANSLATE the glimpse onto the roaming glyph.
      The warp is restricted to SCALE + TRANSLATE only (no rotation/shear):
      an attention-free "hard" glimpse downsampler.

  (B) FIXED center-crop (baseline): the SAME 14x14 patch size is taken by a
      FIXED central crop+resize of the 28x28 input (host-side), fed to the
      SAME classifier. Blind to where the glyph actually landed.

Headline: on jittered/cluttered inputs the LEARNED glimpse finds the glyph and
beats the fixed center-crop, which keeps staring at the (usually empty) centre.

Runs in well under a minute on CPU; no data is downloaded.

See also: examples/SpatialTransformer (the full 6-DoF affine STN this restricts).

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
  cSize     = 28;     // large cluttered canvas
  cPatch    = 14;     // small canonical glimpse / center-crop patch
  cClasses  = 4;
  cSteps    = 1200;
  cBatch    = 16;
  cLR       = 0.01;
  cInertia  = 0.9;
  cNumTest  = 800;
  cMaxShift = 7.0;    // glyph roams far from the centre
  cClutter  = 6;      // bright noise specks per canvas
  cCropExt  = 0.4;    // FIXED center-crop half-extent (misses off-centre glyphs)
  cGlimpse0 = 0.85;   // glimpse INIT half-extent: wide, so the glyph is visible
                      // at cold-start and the localizer gets a learning signal;
                      // it then learns to zoom IN and TRANSLATE onto the glyph.

// ---- Synthetic cluttered dataset ----------------------------------------

procedure DrawGlyph(V: TNNetVolume; AClass, gx, gy: integer);
// Stamp a small ~6x6 glyph centred at (gx,gy). Four CONTENT classes.
var
  i: integer;
  procedure P(x, y: integer);
  begin
    if (x >= 0) and (x < cSize) and (y >= 0) and (y < cSize) then V[x, y, 0] := 1.0;
  end;
begin
  // ~9x9 glyphs (radius 4) so they survive the downsample to the 14x14 patch.
  case AClass of
    0: for i := -4 to 4 do begin                                           // cross
         P(gx + i, gy); P(gx + i, gy - 1);
         P(gx, gy + i); P(gx - 1, gy + i);
       end;
    1: for i := -4 to 4 do begin                                           // box
         P(gx + i, gy - 4); P(gx + i, gy - 3);
         P(gx + i, gy + 4); P(gx + i, gy + 3);
         P(gx - 4, gy + i); P(gx - 3, gy + i);
         P(gx + 4, gy + i); P(gx + 3, gy + i);
       end;
    2: for i := -4 to 4 do begin P(gx + i, gy); P(gx + i, gy - 1);         // h-bar
         P(gx + i, gy - 2); end;
    3: for i := -4 to 4 do begin P(gx, gy + i); P(gx - 1, gy + i);         // v-bar
         P(gx - 2, gy + i); end;
  end;
end;

procedure SampleExample(InputV, TargetV: TNNetVolume; out gx, gy: integer);
var
  cls, k, cx, cy: integer;
begin
  InputV.Fill(0);
  // clutter: random bright specks scattered across the whole canvas
  for k := 1 to cClutter do
    InputV[Random(cSize), Random(cSize), 0] := 0.6 + 0.4 * Random;
  // the glyph at a random offset from the centre
  cx := cSize div 2; cy := cSize div 2;
  gx := cx + Round((Random - 0.5) * 2.0 * cMaxShift);
  gy := cy + Round((Random - 0.5) * 2.0 * cMaxShift);
  cls := Random(cClasses);
  DrawGlyph(InputV, cls, gx, gy);
  TargetV.Fill(0);
  TargetV.OneHotEncodingOnPixel(0, 0, cls);
end;

procedure CenterCrop(Src, Dst: TNNetVolume);
// FIXED baseline: bilinear central crop+resize of the cSize input into the
// cPatch canonical patch. Half-extent cCropExt (blind to the glyph's offset).
var
  ox, oy, x0, y0: integer;
  xn, yn, sx, sy, fx, fy, val, half: TNeuralFloat;
begin
  half := (cSize - 1) * 0.5;
  for oy := 0 to cPatch - 1 do
  begin
    yn := 2.0 * oy / (cPatch - 1) - 1.0;
    for ox := 0 to cPatch - 1 do
    begin
      xn := 2.0 * ox / (cPatch - 1) - 1.0;
      sx := (cCropExt * xn + 1.0) * half;
      sy := (cCropExt * yn + 1.0) * half;
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
end;

// ---- Models -------------------------------------------------------------

procedure AddClassifier(NN: TNNet);
// Shared position-SENSITIVE readout over a cPatch x cPatch x 1 patch.
begin
  NN.AddLayer(TNNetFullConnectReLU.Create(24));
  NN.AddLayer(TNNetFullConnectLinear.Create(cClasses));
  NN.AddLayer(TNNetSoftMax.Create());
end;

procedure BuildFixed(out NN: TNNet);
// Baseline: classifier fed a cPatch input (the fixed center-crop, done host-side).
begin
  NN := TNNet.Create();
  NN.AddLayer(TNNetInput.Create(cPatch, cPatch, 1));
  AddClassifier(NN);
  NN.SetLearningRate(cLR, cInertia);
  NN.InitWeights();
end;

procedure BuildGlimpse(out NN: TNNet);
var
  ImgInput, LocHead: TNNetLayer;
  i: integer;
begin
  NN := TNNet.Create();
  ImgInput := NN.AddLayer(TNNetInput.Create(cSize, cSize, 1));
  // --- Localization head: 4 params (s_x, s_y, t_x, t_y) ---
  NN.AddLayer(TNNetConvolutionReLU.Create(8, 3, 1, 1));
  NN.AddLayer(TNNetMaxPool.Create(2));
  NN.AddLayer(TNNetConvolutionReLU.Create(8, 3, 1, 1));
  NN.AddLayer(TNNetMaxPool.Create(2));
  NN.AddLayer(TNNetFullConnectReLU.Create(24));
  LocHead := NN.AddLayer(TNNetFullConnectLinear.Create(4));
  // --- Restrict to scale + translate: scatter into the 2x3 affine ---
  NN.AddLayer(TNNetScatterToAffine.Create());
  // --- Learned crop/zoom over the ORIGINAL 28x28 image ---
  NN.AddLayerAfter(TNNetAffineGridSample.Create(NN.GetLastLayer), ImgInput);
  // --- Downsample the warped 28x28 view to the 14x14 canonical patch ---
  NN.AddLayer(TNNetAvgPool.Create(2));
  // --- Shared classifier on the glimpse ---
  AddClassifier(NN);
  NN.SetLearningRate(cLR, cInertia);
  NN.InitWeights();
  // Wide-glimpse init: start seeing (almost) the whole canvas (s=cGlimpse0,
  // t=0) so the glyph is visible at cold-start, then LEARN to zoom IN and
  // TRANSLATE the glimpse onto the roaming glyph.
  for i := 0 to LocHead.Neurons.Count - 1 do
    LocHead.Neurons[i].Weights.Fill(0);
  LocHead.Neurons[0].BiasWeight := cGlimpse0; // s_x
  LocHead.Neurons[1].BiasWeight := cGlimpse0; // s_y
  LocHead.Neurons[2].BiasWeight := 0.0;    // t_x
  LocHead.Neurons[3].BiasWeight := 0.0;    // t_y
end;

// ---- Train / evaluate ---------------------------------------------------

procedure Train(NN: TNNet; AGlimpse: boolean; const ATag: string);
var
  Step, B, gx, gy: integer;
  InputV, PatchV, TargetV, Feed: TNNetVolume;
  StartTime, Elapsed: double;
begin
  InputV  := TNNetVolume.Create(cSize, cSize, 1);
  PatchV  := TNNetVolume.Create(cPatch, cPatch, 1);
  TargetV := TNNetVolume.Create(1, 1, cClasses);
  try
    StartTime := Now();
    for Step := 1 to cSteps do
    begin
      for B := 1 to cBatch do
      begin
        SampleExample(InputV, TargetV, gx, gy);
        if AGlimpse then Feed := InputV
        else begin CenterCrop(InputV, PatchV); Feed := PatchV; end;
        NN.Compute(Feed);
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
    InputV.Free; PatchV.Free; TargetV.Free;
  end;
end;

function EvalAccuracy(NN: TNNet; AGlimpse: boolean): TNeuralFloat;
var
  I, Pred, True_, K, gx, gy, Correct: integer;
  InputV, PatchV, TargetV, Feed: TNNetVolume;
  Best: TNeuralFloat;
begin
  InputV  := TNNetVolume.Create(cSize, cSize, 1);
  PatchV  := TNNetVolume.Create(cPatch, cPatch, 1);
  TargetV := TNNetVolume.Create(1, 1, cClasses);
  Correct := 0;
  try
    for I := 1 to cNumTest do
    begin
      SampleExample(InputV, TargetV, gx, gy);
      if AGlimpse then Feed := InputV
      else begin CenterCrop(InputV, PatchV); Feed := PatchV; end;
      NN.Compute(Feed);
      Pred := 0; Best := NN.GetLastLayer.Output.FData[0];
      for K := 1 to cClasses - 1 do
        if NN.GetLastLayer.Output.FData[K] > Best then
        begin Best := NN.GetLastLayer.Output.FData[K]; Pred := K; end;
      True_ := 0;
      for K := 0 to cClasses - 1 do
        if TargetV.FData[K] > 0.5 then True_ := K;
      if Pred = True_ then Inc(Correct);
    end;
  finally
    InputV.Free; PatchV.Free; TargetV.Free;
  end;
  Result := Correct / cNumTest;
end;

type
  TBuildProc = procedure(out NN: TNNet);

procedure RunOne(BuildProc: TBuildProc; AGlimpse: boolean; const ATag: string;
  const ATrainSeed, ATestSeed: longint; out Acc: TNeuralFloat);
var
  NN: TNNet;
begin
  RandSeed := ATrainSeed;
  BuildProc(NN);
  try
    WriteLn('Model: ', ATag);
    NN.DebugStructure();
    Train(NN, AGlimpse, ATag);
    RandSeed := ATestSeed;
    Acc := EvalAccuracy(NN, AGlimpse);
    WriteLn(Format('  [%s] held-out accuracy = %.3f', [ATag, Acc]));
  finally
    NN.Free;
  end;
end;

var
  AccFixed, AccGlimpse: TNeuralFloat;

begin
  WriteLn('Learned scale+translate GLIMPSE downsampler vs FIXED center-crop');
  WriteLn(Format('  canvas=%dx%d  patch=%dx%d  classes=%d  steps=%d  batch=%d',
    [cSize, cSize, cPatch, cPatch, cClasses, cSteps, cBatch]));
  WriteLn(Format('  clutter=%d specks   glyph shift +/-%.1f px', [cClutter, cMaxShift]));
  RunOne(@BuildFixed,   False, 'fixed-crop', 1234, 9999, AccFixed);
  RunOne(@BuildGlimpse, True,  'glimpse',    1234, 9999, AccGlimpse);
  WriteLn;
  WriteLn('Summary (higher is better):');
  WriteLn(Format('  FIXED center-crop        accuracy = %.3f', [AccFixed]));
  WriteLn(Format('  LEARNED glimpse          accuracy = %.3f', [AccGlimpse]));
  if AccGlimpse > AccFixed then
    WriteLn(Format('  Learned glimpse recovers +%.1f accuracy points the fixed crop misses.',
      [(AccGlimpse - AccFixed) * 100.0]))
  else
    WriteLn('  Learned glimpse did NOT help on this run.');
end.
