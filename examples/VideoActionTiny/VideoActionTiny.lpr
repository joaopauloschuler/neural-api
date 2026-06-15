program VideoActionTiny;
(*
VideoActionTiny: demonstrates the TNNetConvolution3D (spatiotemporal /
volumetric) convolution on a tiny SYNTHETIC Moving-MNIST-style "action
recognition" task. No external dataset, pure CPU, finishes in a few seconds.

WHAT IT SHOWS
-------------
Each sample is a short grayscale CLIP: cT frames of a cGrid x cGrid image in
which a small bright blob slides across the grid in one of FOUR directions
(right / left / down / up). The task is to classify the MOTION DIRECTION of the
clip. A single frame is ambiguous (a blob could be moving any way), so the
network MUST integrate information ACROSS TIME -- exactly what a 3-D convolution
buys you over a per-frame 2-D conv.

HOW THE CLIP IS PACKED FOR TNNetConvolution3D
---------------------------------------------
TNNetVolume has three axes (SizeX, SizeY, Depth). To feed a (T, H, W, C) clip
through one volume, the T frames are PACKED contiguously along the Depth axis as
T blocks of C channels each (here C = 1): input Depth = cT * 1, with frame t in
depth slot t. SizeX = W, SizeY = H. TNNetConvolution3D slides a
(FeatureSizeT x FeatureSizeXY x FeatureSizeXY) kernel over the spatial grid
(padded/strided like the 2-D conv) AND over the time blocks within the depth
axis, producing an output of Depth = OutputT * NumFeatures, frames packed the
same way. The C channels of any frame are contiguous, so each kernel tap is a
contiguous AVX dot product.

The net:
  Input(cGrid, cGrid, cT*1)
  Convolution3D(F=8, T=3, K=3, pad=1, stride=1, C=1)  // mixes space AND time
  ReLU
  Convolution3D(F=8, T=3, K=3, pad=1, stride=1, C=1)  // OutputT shrinks by 2/conv
  ReLU
  FullConnectLinear(4)                                // 4 motion classes
  SoftMax

To prove the temporal mixing matters, a BASELINE is trained on the SAME stack
but with FeatureSizeT = 1 (a per-frame 2-D conv shared across frames -- no
cross-frame coupling). The console prints the held-out accuracy ACTUALLY
reached by each; the 3-D conv is expected to win because direction needs time.

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
  neuralnetwork,
  neuralvolume;

const
  cGrid         = 12;     // frame side (cGrid x cGrid, single channel)
  cT            = 6;      // frames per clip (the time / Z axis, packed in Depth)
  cFeatures     = 8;      // conv features per layer
  cClasses      = 4;      // motion directions: 0=right 1=left 2=down 3=up
  cTrainSamples = 320;
  cTestSamples  = 96;
  cEpochs       = 24;
  cLearningRate = 0.01;
  cSeed         = 12345;

type
  TVolArray = array of TNNetVolume;
  TIntArray = array of integer;

// Builds one synthetic clip: a 2x2 bright blob starting at a random position
// and sliding one pixel per frame in direction ADir. The clip is written into
// Clip with the cT frames packed along the Depth axis (frame t -> depth t).
procedure MakeClip(Clip: TNNetVolume; ADir: integer);
var
  t, x, y, bx, by, dx, dy, sx, sy: integer;
begin
  Clip.Fill(0);
  // Direction -> per-frame displacement.
  dx := 0; dy := 0;
  case ADir of
    0: dx := 1;   // right
    1: dx := -1;  // left
    2: dy := 1;   // down
    3: dy := -1;  // up
  end;
  // Start so the blob stays on the grid across all cT frames.
  sx := 1 + Random(cGrid - 2 - (cT - 1) * Abs(dx));
  sy := 1 + Random(cGrid - 2 - (cT - 1) * Abs(dy));
  for t := 0 to cT - 1 do
  begin
    bx := sx + dx * t;
    by := sy + dy * t;
    // 2x2 blob with mild noise so it is not trivially separable.
    for y := 0 to 1 do
      for x := 0 to 1 do
        if (bx + x >= 0) and (bx + x < cGrid) and
           (by + y >= 0) and (by + y < cGrid) then
          Clip[bx + x, by + y, t] := 1.0;
  end;
end;

procedure BuildDataset(out Clips: TVolArray; out Labels: TIntArray; N: integer);
var
  i, dir: integer;
begin
  SetLength(Clips, N);
  SetLength(Labels, N);
  for i := 0 to N - 1 do
  begin
    dir := Random(cClasses);
    Clips[i] := TNNetVolume.Create(cGrid, cGrid, cT); // C=1 -> Depth = cT
    MakeClip(Clips[i], dir);
    Labels[i] := dir;
  end;
end;

procedure FreeDataset(var Clips: TVolArray; var Labels: TIntArray);
var
  i: integer;
begin
  for i := 0 to Length(Clips) - 1 do Clips[i].Free;
  SetLength(Clips, 0);
  SetLength(Labels, 0);
end;

// Builds the classifier. ATemporalKernel is the FeatureSizeT of both 3-D
// convs: cT-meaningful (3) for the spatiotemporal model, or 1 for the per-frame
// 2-D baseline (no cross-frame coupling).
function BuildNet(ATemporalKernel: integer): TNNet;
begin
  Result := TNNet.Create();
  Result.AddLayer(TNNetInput.Create(cGrid, cGrid, cT));
  // Convolution3D(NumFeatures, FeatureSizeT, FeatureSizeXY, pad, stride, C).
  Result.AddLayer(TNNetConvolution3D.Create(cFeatures, ATemporalKernel, 3, 1, 1, 1));
  Result.AddLayer(TNNetReLU.Create());
  Result.AddLayer(TNNetConvolution3D.Create(cFeatures, ATemporalKernel, 3, 1, 1, cFeatures));
  Result.AddLayer(TNNetReLU.Create());
  Result.AddLayer(TNNetFullConnectLinear.Create(cClasses));
  Result.AddLayer(TNNetSoftMax.Create());
  Result.InitWeights();
end;

procedure TrainEpoch(Net: TNNet; const Clips: TVolArray; const Labels: TIntArray);
var
  step, idx, n: integer;
  Target: TNNetVolume;
begin
  n := Length(Clips);
  Target := TNNetVolume.Create(cClasses, 1, 1);
  try
    for step := 0 to n - 1 do
    begin
      idx := Random(n);
      Target.Fill(0);
      Target[Labels[idx], 0, 0] := 1.0;
      Net.Compute(Clips[idx]);
      Net.Backpropagate(Target);
    end;
  finally
    Target.Free;
  end;
end;

function Accuracy(Net: TNNet; const Clips: TVolArray; const Labels: TIntArray): TNeuralFloat;
var
  i, n, correct, predicted, c: integer;
  best: TNeuralFloat;
begin
  n := Length(Clips);
  correct := 0;
  for i := 0 to n - 1 do
  begin
    Net.Compute(Clips[i]);
    predicted := 0;
    best := Net.GetLastLayer().Output[0, 0, 0];
    for c := 1 to cClasses - 1 do
      if Net.GetLastLayer().Output[c, 0, 0] > best then
      begin
        best := Net.GetLastLayer().Output[c, 0, 0];
        predicted := c;
      end;
    if predicted = Labels[i] then Inc(correct);
  end;
  Result := correct / n;
end;

procedure RunAlgo();
var
  Net3D, Net2D: TNNet;
  TrainClips, TestClips: TVolArray;
  TrainLabels, TestLabels: TIntArray;
  epoch: integer;
  acc3D, acc2D: TNeuralFloat;
begin
  RandSeed := cSeed;
  WriteLn('VideoActionTiny: TNNetConvolution3D motion-direction classifier');
  WriteLn(Format('grid=%dx%d  frames=%d  features=%d  classes=%d  train=%d  test=%d  epochs=%d',
    [cGrid, cGrid, cT, cFeatures, cClasses, cTrainSamples, cTestSamples, cEpochs]));
  WriteLn(Format('clip packing: %d frames packed along Depth (C=1) -> Depth=%d',
    [cT, cT]));
  WriteLn;

  RandSeed := cSeed;
  BuildDataset(TrainClips, TrainLabels, cTrainSamples);
  BuildDataset(TestClips, TestLabels, cTestSamples);

  try
    // --- Spatiotemporal model (FeatureSizeT = 3) ---
    RandSeed := cSeed + 1;
    Net3D := BuildNet(3);
    Net3D.SetLearningRate(cLearningRate, 0.9);
    WriteLn('Training 3-D conv (temporal kernel = 3)...');
    for epoch := 1 to cEpochs do
    begin
      TrainEpoch(Net3D, TrainClips, TrainLabels);
      if (epoch = 1) or (epoch mod 6 = 0) then
        WriteLn(Format('  epoch %3d   test acc=%6.4f',
          [epoch, Accuracy(Net3D, TestClips, TestLabels)]));
    end;

    // --- Per-frame baseline (FeatureSizeT = 1, no cross-frame coupling) ---
    RandSeed := cSeed + 1;
    Net2D := BuildNet(1);
    Net2D.SetLearningRate(cLearningRate, 0.9);
    WriteLn('Training per-frame baseline (temporal kernel = 1)...');
    for epoch := 1 to cEpochs do
    begin
      TrainEpoch(Net2D, TrainClips, TrainLabels);
      if (epoch = 1) or (epoch mod 6 = 0) then
        WriteLn(Format('  epoch %3d   test acc=%6.4f',
          [epoch, Accuracy(Net2D, TestClips, TestLabels)]));
    end;

    acc3D := Accuracy(Net3D, TestClips, TestLabels);
    acc2D := Accuracy(Net2D, TestClips, TestLabels);
    WriteLn;
    WriteLn('Held-out accuracy (chance = 0.25)');
    WriteLn(Format('  3-D conv (T=3)        %6.4f', [acc3D]));
    WriteLn(Format('  per-frame (T=1)       %6.4f', [acc2D]));
    WriteLn;
    if acc3D > acc2D + 1e-4 then
      WriteLn('=> The 3-D (spatiotemporal) conv beats the per-frame baseline: ',
              'direction needs TIME.')
    else if acc2D > acc3D + 1e-4 then
      WriteLn('=> On this run the per-frame baseline matched or beat the 3-D conv.')
    else
      WriteLn('=> The two models tied on held-out accuracy on this run.');
  finally
    FreeDataset(TrainClips, TrainLabels);
    FreeDataset(TestClips, TestLabels);
    Net3D.Free;
    Net2D.Free;
  end;
end;

var
  // Stops Lazarus errors
  Application: record Title: string; end;

begin
  Application.Title := 'VideoActionTiny Example';
  RunAlgo();
end.
