program UNetSegmentation;
(*
UNetSegmentation: trains a real symmetric U-Net (built with the TNNet.AddUNet
builder) on a SYNTHETIC shapes segmentation task and reports Dice / IoU on a
held-out split. No external dataset, pure CPU; the default SMOKE run finishes
in well under five minutes. A --full flag trains longer for a sharper result.

WHAT IT SHOWS
-------------
The task is foreground/background segmentation of geometric shapes. Each sample
is a single-channel 32x32 image containing one or two randomly placed/sized
FILLED shapes (circles and axis-aligned rectangles); the ground-truth mask is 1
on shape pixels and 0 on background, and Gaussian noise is added to the INPUT
only. Because the shapes can be small, the foreground is a minority of pixels,
the regime where a region-overlap loss (Dice) helps over per-pixel losses.

THE NETWORK (TNNet.AddUNet)
---------------------------
The whole encoder-decoder is built by ONE call:

  NN.AddLayer(TNNetInput.Create(32, 32, 1, 1));
  NN.AddUNet(Depth, BaseFeatures, 1, Taps);  // 1 output logit channel
  NN.AddLayer(TNNetSigmoid.Create());        // per-pixel foreground prob
  NN.AddLayer(TNNetDiceLoss.Create());       // analytic Dice gradient

AddUNet builds Depth encoder stages (each = 2x [Conv3x3 -> MovingStdNorm ->
ReLU] then a 2x2 stride-2 MaxPool, doubling features), a bottleneck, then Depth
decoder stages (each = nearest x2 upsample -> depth-concat with the matching
encoder tap (the SKIP connection) -> 2x conv) and a final 1x1 conv head to the
requested channel count. The output spatial size equals the input size, so the
1-channel logit map is fed to a Sigmoid + TNNetDiceLoss segmentation head, the
SAME Dice loss used by examples/DiceSegmentation. AddUNet returns the encoder
tap layer indices (Taps) for inspection. The input side (32) must be divisible
by 2^Depth, which holds for Depth up to 5.

The Dice loss head (TNNetDiceLoss, a Tversky loss with alpha=beta=0.5) is an
identity passthrough whose FOutput equals the sigmoid foreground probability; it
overwrites the framework-seeded residual with the analytic Dice gradient, so the
layer feeding it MUST be a sigmoid and the target supplied to Backpropagate is
the binary mask.

OUTPUT
------
Per-eval Dice and IoU on the held-out set (prediction thresholded at 0.5), one
held-out sample rendered as ASCII (input / ground truth / prediction) and a
small PPM visualization (input | ground-truth | prediction strip) written to
unet_sample.ppm. The console prints whatever the numbers ACTUALLY show.

USAGE
-----
  ./UNetSegmentation            smoke run (fast, default)
  ./UNetSegmentation --full     longer training for a sharper Dice/IoU

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
  cGrid       = 32;   // image side (cGrid x cGrid, single channel)
  cDepth      = 3;    // U-Net stages (32 / 2^3 = 4 bottleneck cells)
  cBaseFeat   = 8;    // base feature count (doubles per stage)
  cThreshold  = 0.5;  // probability -> foreground threshold
  cNoise      = 0.25; // stddev of input noise
  cSeed       = 20260614;

type
  TVolArray = array of TNNetVolume;

var
  // Run-mode dependent hyperparameters (set in ParseArgs).
  gTrainSamples: integer = 192;
  gTestSamples:  integer = 64;
  gEpochs:       integer = 12;
  gLearnRate:    TNeuralFloat = 0.02;
  gFull:         boolean = false;

// Box-Muller N(0,1) sample.
function RandomGauss(): TNeuralFloat;
var
  U1, U2: TNeuralFloat;
begin
  repeat U1 := Random; until U1 > 1e-9;
  U2 := Random;
  Result := Sqrt(-2 * Ln(U1)) * Cos(2 * Pi * U2);
end;

// Stamps a filled circle into Mask (value 1).
procedure StampCircle(Mask: TNNetVolume);
var
  Cx, Cy, R, X, Y, Dx, Dy: integer;
begin
  R := 3 + Random(5);                       // radius 3..7
  Cx := R + Random(cGrid - 2 * R);
  Cy := R + Random(cGrid - 2 * R);
  for Y := 0 to cGrid - 1 do
    for X := 0 to cGrid - 1 do
    begin
      Dx := X - Cx; Dy := Y - Cy;
      if (Dx * Dx + Dy * Dy) <= (R * R) then Mask[X, Y, 0] := 1;
    end;
end;

// Stamps a filled axis-aligned rectangle into Mask (value 1).
procedure StampRect(Mask: TNNetVolume);
var
  X0, Y0, W, H, X, Y: integer;
begin
  W := 5 + Random(10);                       // width 5..14
  H := 5 + Random(10);                       // height 5..14
  X0 := Random(cGrid - W);
  Y0 := Random(cGrid - H);
  for Y := Y0 to Y0 + H - 1 do
    for X := X0 to X0 + W - 1 do
      Mask[X, Y, 0] := 1;
end;

// Builds one (input, mask) pair: one or two random shapes plus input noise.
procedure MakeSample(Img, Mask: TNNetVolume);
var
  X, Y, Shapes, S: integer;
begin
  Mask.Fill(0);
  Shapes := 1 + Random(2);                    // 1 or 2 shapes
  for S := 1 to Shapes do
    if Random(2) = 0 then StampCircle(Mask) else StampRect(Mask);
  // Input: foreground ~1, background ~0, both corrupted by Gaussian noise.
  for Y := 0 to cGrid - 1 do
    for X := 0 to cGrid - 1 do
      Img[X, Y, 0] := Mask[X, Y, 0] + RandomGauss() * cNoise;
end;

procedure BuildDataset(out Imgs, Masks: TVolArray; N: integer);
var I: integer;
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
var I: integer;
begin
  for I := 0 to Length(Imgs) - 1 do begin Imgs[I].Free; Masks[I].Free; end;
  SetLength(Imgs, 0); SetLength(Masks, 0);
end;

// Builds the U-Net segmentation net via the AddUNet builder + Dice head.
function BuildNet(): TNNet;
var Taps: TNeuralIntegerArray; I: integer;
begin
  Result := TNNet.Create();
  Result.AddLayer(TNNetInput.Create(cGrid, cGrid, 1, 1));
  // UseNorm=false: a plain conv->ReLU U-Net optimizes more stably with the
  // global Dice gradient on this tiny task (moving-std norm tends to collapse
  // the shallow head to the trivial all-foreground solution here).
  Result.AddUNet(cDepth, cBaseFeat, 1, Taps, {UseNorm=}false);
  Result.AddLayer(TNNetSigmoid.Create());       // per-pixel foreground prob
  Result.AddLayer(TNNetDiceLoss.Create());      // analytic Dice gradient
  Result.InitWeights();
  Write('U-Net built: depth=', cDepth, ' base=', cBaseFeat,
        ' layers=', Result.CountLayers(), ' skip taps=[');
  for I := 0 to Length(Taps) - 1 do
  begin
    if I > 0 then Write(',');
    Write(Taps[I]);
  end;
  WriteLn(']');
end;

function ForegroundProb(Net: TNNet; X, Y: integer): TNeuralFloat;
begin
  Result := Net.GetLastLayer().Output[X, Y, 0];
end;

procedure TrainEpoch(Net: TNNet; const Imgs, Masks: TVolArray);
var Step, Idx, N: integer;
begin
  N := Length(Imgs);
  for Step := 0 to N - 1 do
  begin
    Idx := Random(N);
    Net.Compute(Imgs[Idx]);
    Net.Backpropagate(Masks[Idx]);
  end;
end;

// Mean Dice and mean IoU on a dataset (prediction thresholded at 0.5).
procedure Evaluate(Net: TNNet; const Imgs, Masks: TVolArray;
  out MeanDice, MeanIoU: TNeuralFloat);
var
  I, X, Y, N, Inter, PredFg, GtFg, Union: integer;
  P, G, SumDice, SumIoU: TNeuralFloat;
begin
  N := Length(Imgs);
  SumDice := 0; SumIoU := 0;
  for I := 0 to N - 1 do
  begin
    Net.Compute(Imgs[I]);
    Inter := 0; PredFg := 0; GtFg := 0;
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
    SumDice := SumDice + (2 * Inter + 1) / (PredFg + GtFg + 1);
    SumIoU := SumIoU + (Inter + 1) / (Union + 1);
  end;
  MeanDice := SumDice / N;
  MeanIoU := SumIoU / N;
end;

// ASCII render of one held-out sample (input / ground truth / prediction).
procedure RenderSample(Net: TNNet; const Imgs, Masks: TVolArray; Idx: integer);
var
  X, Y: integer;

  function InputChar(V: TNeuralFloat): char;
  begin
    if V >= 0.5 then Result := '#'
    else if V >= 0.2 then Result := '+'
    else if V <= -0.2 then Result := ','
    else Result := '.';
  end;
  function MaskChar(V: TNeuralFloat): char;
  begin if V >= 0.5 then Result := '#' else Result := '.'; end;
  function PredChar(V: TNeuralFloat): char;
  begin if V >= cThreshold then Result := '#' else Result := '.'; end;

begin
  Net.Compute(Imgs[Idx]);
  WriteLn;
  WriteLn('One held-out sample  ("#" = foreground, threshold ', cThreshold:3:1, ')');
  WriteLn('  col 1: noisy INPUT          col 2: GROUND TRUTH          col 3: PREDICTION');
  WriteLn;
  for Y := 0 to cGrid - 1 do
  begin
    for X := 0 to cGrid - 1 do Write(InputChar(Imgs[Idx][X, Y, 0]));
    Write('   ');
    for X := 0 to cGrid - 1 do Write(MaskChar(Masks[Idx][X, Y, 0]));
    Write('   ');
    for X := 0 to cGrid - 1 do Write(PredChar(ForegroundProb(Net, X, Y)));
    WriteLn;
  end;
end;

// Writes a PPM (P6) strip: input | ground truth | prediction, 1px black gaps.
// PPM is a trivial, dependency-free image format any viewer / ImageMagick reads.
procedure WritePPM(Net: TNNet; const Imgs, Masks: TVolArray; Idx: integer;
  const FileName: string);
const
  Gap = 1;
var
  F: TextFile;
  Bin: TFileStream;
  W, H, X, Y, Panel: integer;
  Hdr: string;
  R, GByte, B: byte;
  V, P: TNeuralFloat;

  procedure Emit(rr, gg, bb: byte);
  begin
    Bin.WriteByte(rr); Bin.WriteByte(gg); Bin.WriteByte(bb);
  end;

begin
  W := cGrid * 3 + Gap * 2;
  H := cGrid;
  Net.Compute(Imgs[Idx]);
  // Header is ASCII; pixel payload is binary, so write header then raw bytes.
  AssignFile(F, FileName); Rewrite(F);
  WriteLn(F, 'P6');
  WriteLn(F, W, ' ', H);
  WriteLn(F, 255);
  CloseFile(F);
  Bin := TFileStream.Create(FileName, fmOpenReadWrite);
  try
    Bin.Seek(0, soEnd);
    for Y := 0 to H - 1 do
      for X := 0 to W - 1 do
      begin
        if (X = cGrid) or (X = cGrid + Gap + cGrid) then
        begin
          Emit(0, 0, 0); Continue;                 // black gap column
        end;
        if X < cGrid then begin Panel := 0; end
        else if X < cGrid * 2 + Gap then begin Panel := 1; end
        else begin Panel := 2; end;
        case Panel of
          0: begin
               V := Imgs[Idx][X, Y, 0];
               if V < 0 then V := 0; if V > 1 then V := 1;
               R := Round(V * 255); GByte := R; B := R;     // grayscale input
             end;
          1: begin
               if Masks[Idx][X - cGrid - Gap, Y, 0] >= 0.5 then
                 begin R := 0; GByte := 200; B := 0; end     // GT = green
               else begin R := 30; GByte := 30; B := 30; end;
             end;
        else
          begin
            P := ForegroundProb(Net, X - (cGrid * 2 + Gap * 2), Y);
            if P >= cThreshold then begin R := 220; GByte := 60; B := 60; end // pred = red
            else begin R := 30; GByte := 30; B := 30; end;
          end;
        end;
        Emit(R, GByte, B);
      end;
  finally
    Bin.Free;
  end;
  Hdr := FileName;
  WriteLn('Wrote visualization: ', Hdr, '  (input | ground-truth | prediction)');
end;

procedure ParseArgs();
var I: integer;
begin
  for I := 1 to ParamCount do
    if (ParamStr(I) = '--full') then gFull := true;
  if gFull then
  begin
    gTrainSamples := 512;
    gTestSamples  := 128;
    gEpochs       := 60;
    gLearnRate    := 0.02;
  end;
end;

procedure RunAlgo();
var
  Net: TNNet;
  TrainImgs, TrainMasks, TestImgs, TestMasks: TVolArray;
  Epoch: integer;
  D, IoU: TNeuralFloat;
  T0: TDateTime;
begin
  ParseArgs();
  RandSeed := cSeed;
  WriteLn('UNetSegmentation: U-Net (AddUNet) + Dice loss on synthetic shapes');
  WriteLn(Format('mode=%s  grid=%dx%d  depth=%d  base=%d  train=%d  test=%d  epochs=%d  lr=%.3f',
    [BoolToStr(gFull, 'full', 'smoke'), cGrid, cGrid, cDepth, cBaseFeat,
     gTrainSamples, gTestSamples, gEpochs, gLearnRate]));
  WriteLn;

  BuildDataset(TrainImgs, TrainMasks, gTrainSamples);
  BuildDataset(TestImgs, TestMasks, gTestSamples);
  try
    RandSeed := cSeed + 1;
    Net := BuildNet();
    Net.SetLearningRate(gLearnRate, 0.9);
    WriteLn;
    WriteLn('Training...');
    T0 := Now();
    Evaluate(Net, TestImgs, TestMasks, D, IoU);
    WriteLn(Format('  epoch %3d   test Dice=%6.4f  IoU=%6.4f  (untrained)', [0, D, IoU]));
    for Epoch := 1 to gEpochs do
    begin
      TrainEpoch(Net, TrainImgs, TrainMasks);
      if (Epoch = 1) or (Epoch mod 5 = 0) or (Epoch = gEpochs) then
      begin
        Evaluate(Net, TestImgs, TestMasks, D, IoU);
        WriteLn(Format('  epoch %3d   test Dice=%6.4f  IoU=%6.4f', [Epoch, D, IoU]));
      end;
    end;
    WriteLn(Format('Training wall time: %.1f s', [(Now() - T0) * 24 * 3600]));

    Evaluate(Net, TestImgs, TestMasks, D, IoU);
    WriteLn;
    WriteLn(Format('FINAL held-out  Dice=%6.4f   IoU=%6.4f', [D, IoU]));

    RenderSample(Net, TestImgs, TestMasks, 0);
    WritePPM(Net, TestImgs, TestMasks, 0, 'unet_sample.ppm');

    Net.Free;
  finally
    FreeDataset(TrainImgs, TrainMasks);
    FreeDataset(TestImgs, TestMasks);
  end;
end;

begin
  RunAlgo();
end.
