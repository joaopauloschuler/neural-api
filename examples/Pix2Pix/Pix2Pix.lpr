program Pix2Pix;
(*
Pix2Pix: a PAIRED conditional image-to-image translation example
(Isola et al. 2017, "Image-to-Image Translation with Conditional Adversarial
Networks"). It trains a conditional GAN that maps a 1-channel GRAYSCALE shapes
image to its 3-channel COLORIZED version on a SYNTHETIC, generated-in-code task.
No dataset download, no parity fixture; pure CPU. The default SMOKE run finishes
in well under five minutes; a --full flag trains longer for a sharper result.

This is distinct from the UNCONDITIONAL examples/VisualGAN (noise -> CIFAR): the
generator here is CONDITIONED on an input image and the output is a deterministic
translation of that input, scored by a PatchGAN discriminator.

THE TASK (grayscale -> color)
-----------------------------
Each sample is one or two random FILLED shapes (circles, axis-aligned rectangles)
on a dark background. The INPUT is the single-channel grayscale rendering. The
PAIRED TARGET is the SAME geometry colorized by a fixed, learnable rule:
  - circle pixels -> red   (1, 0, 0)
  - rectangle pixels -> green (0, 1, 0)
  - background -> dark blue (0.1, 0.1, 0.4)
So the network must (a) reconstruct the shape silhouettes from grayscale and
(b) infer each shape's COLOR from its geometry (round vs straight edges) — a
genuine conditional translation, not a per-pixel lookup. RGB is encoded in
[-1,1] (so 0 -> -1, 1 -> +1) to match the generator's Tanh output range.

THE GENERATOR (TNNet.AddUNet)
-----------------------------
A U-Net built by ONE builder call, same builder as examples/UNetSegmentation:

  G.AddLayer(TNNetInput.Create(grid, grid, 1, 1));   // grayscale condition
  G.AddUNet(Depth, BaseFeatures, 3, Taps, UseNorm);  // 3 output channels (RGB)
  G.AddLayer(TNNetHyperbolicTangent.Create());                     // RGB in [-1,1]

AddUNet builds Depth encoder stages (each 2x [Conv3x3 -> (opt Norm) -> ReLU] +
2x2 MaxPool, doubling features), a bottleneck, then Depth decoder stages (nearest
x2 upsample -> depth-concat with the matching encoder SKIP tap -> 2x conv) and a
final 1x1 conv head to 3 channels. Output spatial size == input size. grid must
be divisible by 2^Depth.

THE DISCRIMINATOR (PatchGAN, composed from existing convs — NO new leaf class)
-----------------------------------------------------------------------------
A small fully-convolutional net that scores overlapping NxN patches as real/fake
instead of emitting a single image-level scalar (Isola et al.'s "PatchGAN"). Its
input is the CONDITION stacked with the IMAGE on the depth axis (4 channels:
[grayscale | R | G | B]) so it judges whether the colorization is consistent
WITH the input, not just whether it looks plausible alone. Body:

  D.AddLayer(TNNetInput.Create(grid, grid, 4, 1).EnableErrorCollection);
  Conv(64, 4, pad1, stride2) -> LeakyReLU
  Conv(128,4, pad1, stride2) -> LeakyReLU
  Conv(1,  3, pad1, stride1)            // linear patch-logit map (LSGAN)

The receptive field of each output cell covers a local patch, so the output is a
small grid of patch scores. We use the LEAST-SQUARES GAN objective (Mao et al.
2017): D regresses real->1 / fake->0 and G is pushed toward 1, which is far more
stable on CPU than the log-loss saturating sigmoid. EnableErrorCollection on D's
input layer lets us read the gradient of D's score w.r.t. the generated pixels.

ADVERSARIAL TRAINING (hand-rolled loop)
---------------------------------------
The framework seeds a layer's output error as (output - target) in
ComputeOutputErrorWith, which is exactly the LSGAN/MSE gradient. Per step:

  1. G.Compute(gray); read fake RGB.
  2. Train D: D.Compute(real pair); Backpropagate(ones)  -> push toward 1.
              D.Compute(fake pair); Backpropagate(zeros) -> push toward 0.
  3. Train G: D.Compute(fake pair); Backpropagate(ones)  -> seeds D so its
     input-layer OutputError holds d(adv)/d(pixels) (we do NOT update D here —
     LR is restored after). Take the RGB slice of that input gradient, ADD the
     L1 reconstruction gradient lambda*sign(fake - target), copy the sum into
     G's Tanh OutputError and Backpropagate() G directly.

The L1 term is what makes the output SHARP and correctly colored; the adversarial
term sharpens edges. We report both the mean L1 distance (down = learning the
mapping) and a colorization accuracy (argmax over {R,G,B,bg} per pixel vs target),
plus an ASCII triptych and a PPM strip (input | target | generated).

OUTPUT
------
Per-eval mean L1 (in [0,2] pixel units) and per-pixel color accuracy on a
held-out set, one held-out sample as ASCII (input / target / generated) and a
PPM (input | target | generated) written to pix2pix_sample.ppm. The console
prints whatever the numbers ACTUALLY show.

USAGE
-----
  ./Pix2Pix            smoke run (fast, default)
  ./Pix2Pix --full     longer training for a sharper result

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
  cGrid       = 16;   // image side (cGrid x cGrid)
  cDepth      = 2;    // U-Net stages (16 / 2^2 = 4 bottleneck cells)
  cBaseFeat   = 16;   // base feature count (doubles per stage)
  cLambdaL1   = 10.0; // weight of the L1 reconstruction term (Isola et al.: 100)
  cSeed       = 20260614;

  // Shape-class color codes (RGB in [-1,1], i.e. 0->-1, 1->+1).
  cBgR = -0.8; cBgG = -0.8; cBgB = -0.2;   // dark blue background
  // circle  -> red   (+1,-1,-1)
  // rect    -> green (-1,+1,-1)

type
  TVolArray = array of TNNetVolume;
  TLabelArray = array of TNNetVolume;  // per-pixel class id (0=bg,1=red,2=green)

var
  // Run-mode dependent hyperparameters (set in ParseArgs).
  gTrainSamples: integer = 160;
  gTestSamples:  integer = 48;
  gEpochs:       integer = 30;
  gGLearnRate:   TNeuralFloat = 0.0008;
  gDLearnRate:   TNeuralFloat = 0.0008;
  gFull:         boolean = false;

// Stamps a filled circle; marks class 1 (red) in Cls.
procedure StampCircle(Gray, Cls: TNNetVolume);
var Cx, Cy, R, X, Y, Dx, Dy: integer;
begin
  R := 2 + Random(4);                       // radius 2..5
  Cx := R + Random(cGrid - 2 * R);
  Cy := R + Random(cGrid - 2 * R);
  for Y := 0 to cGrid - 1 do
    for X := 0 to cGrid - 1 do
    begin
      Dx := X - Cx; Dy := Y - Cy;
      if (Dx * Dx + Dy * Dy) <= (R * R) then
      begin
        Gray[X, Y, 0] := 1;
        Cls[X, Y, 0]  := 1;   // red
      end;
    end;
end;

// Stamps a filled rectangle; marks class 2 (green) in Cls.
procedure StampRect(Gray, Cls: TNNetVolume);
var X0, Y0, W, H, X, Y: integer;
begin
  W := 3 + Random(7);                        // width 3..9
  H := 3 + Random(7);
  X0 := Random(cGrid - W);
  Y0 := Random(cGrid - H);
  for Y := Y0 to Y0 + H - 1 do
    for X := X0 to X0 + W - 1 do
    begin
      Gray[X, Y, 0] := 1;
      Cls[X, Y, 0]  := 2;     // green
    end;
end;

// Builds one (grayscale input, colorized target, class map) triple.
procedure MakeSample(Gray, Color, Cls: TNNetVolume);
var X, Y, Shapes, S, C: integer;
begin
  Gray.Fill(0);
  Cls.Fill(0);
  Shapes := 1 + Random(2);                    // 1 or 2 shapes
  for S := 1 to Shapes do
    if Random(2) = 0 then StampCircle(Gray, Cls) else StampRect(Gray, Cls);
  // Colorize from the class map.
  for Y := 0 to cGrid - 1 do
    for X := 0 to cGrid - 1 do
    begin
      C := Round(Cls[X, Y, 0]);
      case C of
        1: begin Color[X, Y, 0] :=  1; Color[X, Y, 1] := -1; Color[X, Y, 2] := -1; end; // red
        2: begin Color[X, Y, 0] := -1; Color[X, Y, 1] :=  1; Color[X, Y, 2] := -1; end; // green
      else
        begin Color[X, Y, 0] := cBgR; Color[X, Y, 1] := cBgG; Color[X, Y, 2] := cBgB; end;
      end;
    end;
end;

procedure BuildDataset(out Grays, Colors: TVolArray; out Cls: TLabelArray; N: integer);
var I: integer;
begin
  SetLength(Grays, N); SetLength(Colors, N); SetLength(Cls, N);
  for I := 0 to N - 1 do
  begin
    Grays[I]  := TNNetVolume.Create(cGrid, cGrid, 1);
    Colors[I] := TNNetVolume.Create(cGrid, cGrid, 3);
    Cls[I]    := TNNetVolume.Create(cGrid, cGrid, 1);
    MakeSample(Grays[I], Colors[I], Cls[I]);
  end;
end;

procedure FreeDataset(var Grays, Colors: TVolArray; var Cls: TLabelArray);
var I: integer;
begin
  for I := 0 to Length(Grays) - 1 do
  begin Grays[I].Free; Colors[I].Free; Cls[I].Free; end;
  SetLength(Grays, 0); SetLength(Colors, 0); SetLength(Cls, 0);
end;

// ---------------------------------------------------------------------------
// Networks
// ---------------------------------------------------------------------------

function BuildGenerator(): TNNet;
var Taps: TNeuralIntegerArray;
begin
  Result := TNNet.Create();
  Result.AddLayer(TNNetInput.Create(cGrid, cGrid, 1, 1));  // grayscale condition
  // Plain conv->ReLU U-Net (UseNorm=false): moving-std norm tends to destabilize
  // the tiny adversarial head here.
  Result.AddUNet(cDepth, cBaseFeat, 3, Taps, {UseNorm=}false);
  Result.AddLayer(TNNetHyperbolicTangent.Create());                     // RGB in [-1,1]
  Result.InitWeights();
end;

// PatchGAN: 4-channel input ([gray | R | G | B]) -> small patch-logit grid.
function BuildDiscriminator(): TNNet;
begin
  Result := TNNet.Create();
  Result.AddLayer(TNNetInput.Create(cGrid, cGrid, 4, 1).EnableErrorCollection);
  Result.AddLayer(TNNetConvolutionLinear.Create(64, 4, 1, 2));  // /2
  Result.AddLayer(TNNetLeakyReLU.Create());
  Result.AddLayer(TNNetConvolutionLinear.Create(128, 4, 1, 2)); // /4
  Result.AddLayer(TNNetLeakyReLU.Create());
  Result.AddLayer(TNNetConvolutionLinear.Create(1, 3, 1, 1));   // linear patch logits
  Result.InitWeights();
end;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// Packs [gray | RGB] into a 4-channel discriminator input volume.
procedure PackPair(Dst, Gray, Rgb: TNNetVolume);
var X, Y: integer;
begin
  for Y := 0 to cGrid - 1 do
    for X := 0 to cGrid - 1 do
    begin
      Dst[X, Y, 0] := Gray[X, Y, 0];
      Dst[X, Y, 1] := Rgb[X, Y, 0];
      Dst[X, Y, 2] := Rgb[X, Y, 1];
      Dst[X, Y, 3] := Rgb[X, Y, 2];
    end;
end;

// ---------------------------------------------------------------------------
// Evaluation
// ---------------------------------------------------------------------------

// Argmax over {bg, red, green} for a generated RGB pixel.
function ColorClass(R, G, B: TNeuralFloat): integer;
var dBg, dRed, dGreen: TNeuralFloat;
begin
  dBg    := Sqr(R - cBgR) + Sqr(G - cBgG) + Sqr(B - cBgB);
  dRed   := Sqr(R - 1)    + Sqr(G + 1)    + Sqr(B + 1);
  dGreen := Sqr(R + 1)    + Sqr(G - 1)    + Sqr(B + 1);
  if (dRed <= dBg) and (dRed <= dGreen) then Result := 1
  else if (dGreen <= dBg) and (dGreen <= dRed) then Result := 2
  else Result := 0;
end;

procedure Evaluate(G: TNNet; const Grays, Colors: TVolArray; const Cls: TLabelArray;
  out MeanL1, ColorAcc: TNeuralFloat);
var
  I, X, Y, N, Correct, Total: integer;
  Out_: TNNetVolume;
  SumL1: TNeuralFloat;
begin
  N := Length(Grays);
  SumL1 := 0; Correct := 0; Total := 0;
  for I := 0 to N - 1 do
  begin
    G.Compute(Grays[I]);
    Out_ := G.GetLastLayer().Output;
    for Y := 0 to cGrid - 1 do
      for X := 0 to cGrid - 1 do
      begin
        SumL1 := SumL1 +
          (Abs(Out_[X, Y, 0] - Colors[I][X, Y, 0]) +
           Abs(Out_[X, Y, 1] - Colors[I][X, Y, 1]) +
           Abs(Out_[X, Y, 2] - Colors[I][X, Y, 2])) / 3;
        if ColorClass(Out_[X, Y, 0], Out_[X, Y, 1], Out_[X, Y, 2]) = Round(Cls[I][X, Y, 0]) then
          Inc(Correct);
        Inc(Total);
      end;
  end;
  MeanL1 := SumL1 / Total;
  ColorAcc := Correct / Total;
end;

// ASCII triptych of one held-out sample.
procedure RenderSample(G: TNNet; const Grays, Colors: TVolArray; Idx: integer);
var X, Y: integer; Out_: TNNetVolume;
  function GrayCh(V: TNeuralFloat): char;
  begin if V >= 0.5 then Result := '#' else Result := '.'; end;
  function ColCh(R, GG, B: TNeuralFloat): char;
  begin
    case ColorClass(R, GG, B) of
      1: Result := 'R';
      2: Result := 'G';
    else Result := '.';
    end;
  end;
begin
  G.Compute(Grays[Idx]);
  Out_ := G.GetLastLayer().Output;
  WriteLn;
  WriteLn('One held-out sample  (R=red circle, G=green rect, .=background)');
  WriteLn('  col 1: grayscale INPUT     col 2: TARGET color     col 3: GENERATED');
  WriteLn;
  for Y := 0 to cGrid - 1 do
  begin
    for X := 0 to cGrid - 1 do Write(GrayCh(Grays[Idx][X, Y, 0]));
    Write('   ');
    for X := 0 to cGrid - 1 do Write(ColCh(Colors[Idx][X, Y, 0], Colors[Idx][X, Y, 1], Colors[Idx][X, Y, 2]));
    Write('   ');
    for X := 0 to cGrid - 1 do Write(ColCh(Out_[X, Y, 0], Out_[X, Y, 1], Out_[X, Y, 2]));
    WriteLn;
  end;
end;

// PPM (P6) strip: input | target | generated, 1px black gaps.
procedure WritePPM(G: TNNet; const Grays, Colors: TVolArray; Idx: integer;
  const FileName: string);
const Gap = 1;
var
  F: TextFile; Bin: TFileStream; W, H, X, Y, Panel: integer; Out_: TNNetVolume;
  procedure Emit(rr, gg, bb: byte);
  begin Bin.WriteByte(rr); Bin.WriteByte(gg); Bin.WriteByte(bb); end;
  function ToByte(V: TNeuralFloat): byte;   // [-1,1] -> [0,255]
  begin
    V := (V + 1) * 0.5; if V < 0 then V := 0; if V > 1 then V := 1;
    Result := Round(V * 255);
  end;
begin
  W := cGrid * 3 + Gap * 2; H := cGrid;
  G.Compute(Grays[Idx]);
  Out_ := G.GetLastLayer().Output;
  AssignFile(F, FileName); Rewrite(F);
  WriteLn(F, 'P6'); WriteLn(F, W, ' ', H); WriteLn(F, 255);
  CloseFile(F);
  Bin := TFileStream.Create(FileName, fmOpenReadWrite);
  try
    Bin.Seek(0, soEnd);
    for Y := 0 to H - 1 do
      for X := 0 to W - 1 do
      begin
        if (X = cGrid) or (X = cGrid * 2 + Gap) then begin Emit(0, 0, 0); Continue; end;
        if X < cGrid then Panel := 0
        else if X < cGrid * 2 + Gap then Panel := 1
        else Panel := 2;
        case Panel of
          0: begin
               // grayscale input rendered as gray (0..1 -> -1..1 for ToByte)
               Emit(ToByte(Grays[Idx][X, Y, 0] * 2 - 1),
                    ToByte(Grays[Idx][X, Y, 0] * 2 - 1),
                    ToByte(Grays[Idx][X, Y, 0] * 2 - 1));
             end;
          1: Emit(ToByte(Colors[Idx][X - cGrid - Gap, Y, 0]),
                  ToByte(Colors[Idx][X - cGrid - Gap, Y, 1]),
                  ToByte(Colors[Idx][X - cGrid - Gap, Y, 2]));
        else
          Emit(ToByte(Out_[X - (cGrid * 2 + Gap * 2), Y, 0]),
               ToByte(Out_[X - (cGrid * 2 + Gap * 2), Y, 1]),
               ToByte(Out_[X - (cGrid * 2 + Gap * 2), Y, 2]));
        end;
      end;
  finally
    Bin.Free;
  end;
  WriteLn('Wrote visualization: ', FileName, '  (input | target | generated)');
end;

// ---------------------------------------------------------------------------
// Training
// ---------------------------------------------------------------------------

procedure ParseArgs();
var I: integer;
begin
  for I := 1 to ParamCount do
    if ParamStr(I) = '--full' then gFull := true;
  if gFull then
  begin
    gTrainSamples := 400;
    gTestSamples  := 96;
    gEpochs       := 120;
  end;
end;

procedure RunAlgo();
var
  G, D: TNNet;
  Grays, Colors, TGrays, TColors: TVolArray;
  Cls, TCls: TLabelArray;
  Epoch, Step, Idx, X, Y: integer;
  FakePair, RealPair, Ones, Zeros: TNNetVolume;
  GOut, GErr, DInGrad: TNNetVolume;
  L1, Acc: TNeuralFloat;
  T0: TDateTime;
  PatchW, PatchH: integer;
begin
  ParseArgs();
  RandSeed := cSeed;
  WriteLn('Pix2Pix: conditional GAN (U-Net generator + PatchGAN) grayscale -> color');
  WriteLn(Format('mode=%s  grid=%dx%d  depth=%d  base=%d  train=%d  test=%d  epochs=%d  lambdaL1=%.1f',
    [BoolToStr(gFull, 'full', 'smoke'), cGrid, cGrid, cDepth, cBaseFeat,
     gTrainSamples, gTestSamples, gEpochs, cLambdaL1]));
  WriteLn;

  BuildDataset(Grays, Colors, Cls, gTrainSamples);
  BuildDataset(TGrays, TColors, TCls, gTestSamples);

  RandSeed := cSeed + 1;
  G := BuildGenerator();
  D := BuildDiscriminator();
  G.SetLearningRate(gGLearnRate, 0.5);
  D.SetLearningRate(gDLearnRate, 0.5);
  // We call G's last-layer Backpropagate() directly: prime its departing-branch
  // count once (the framework does this for the last layer automatically when
  // you call Backpropagate(target); here we seed the error ourselves).
  G.GetLastLayer().IncDepartingBranchesCnt();

  WriteLn(Format('Generator layers=%d   Discriminator layers=%d', [G.CountLayers(), D.CountLayers()]));
  PatchW := D.GetLastLayer().Output.SizeX;
  PatchH := D.GetLastLayer().Output.SizeY;
  WriteLn(Format('PatchGAN score grid: %dx%d patches', [PatchW, PatchH]));
  WriteLn;

  // Scratch volumes.
  FakePair := TNNetVolume.Create(cGrid, cGrid, 4);
  RealPair := TNNetVolume.Create(cGrid, cGrid, 4);
  GErr     := TNNetVolume.Create(cGrid, cGrid, 3);
  Ones  := TNNetVolume.Create(PatchW, PatchH, 1); Ones.Fill(1);
  Zeros := TNNetVolume.Create(PatchW, PatchH, 1); Zeros.Fill(0);

  try
    WriteLn('Training...');
    T0 := Now();
    Evaluate(G, TGrays, TColors, TCls, L1, Acc);
    WriteLn(Format('  epoch %3d   test L1=%6.4f  colorAcc=%6.4f  (untrained)', [0, L1, Acc]));

    for Epoch := 1 to gEpochs do
    begin
      for Step := 0 to gTrainSamples - 1 do
      begin
        Idx := Random(gTrainSamples);

        // ---- 1) Generator forward -------------------------------------
        G.Compute(Grays[Idx]);
        GOut := G.GetLastLayer().Output;

        // ---- 2) Train discriminator -----------------------------------
        PackPair(RealPair, Grays[Idx], Colors[Idx]);
        D.Compute(RealPair);
        D.Backpropagate(Ones);                 // real -> 1
        PackPair(FakePair, Grays[Idx], GOut);
        D.Compute(FakePair);
        D.Backpropagate(Zeros);                // fake -> 0

        // ---- 3) Train generator (adversarial + L1) --------------------
        // Adversarial: push D(fake) toward 1, freeze D (LR=0), read the input
        // gradient D collected for the RGB channels.
        D.SetLearningRate(0, 0.5);
        D.Compute(FakePair);
        D.Backpropagate(Ones);                 // want D to call fake "real"
        DInGrad := D.Layers[0].OutputError;    // 4-ch grad w.r.t. [gray|R|G|B]
        D.SetLearningRate(gDLearnRate, 0.5);

        for Y := 0 to cGrid - 1 do
          for X := 0 to cGrid - 1 do
          begin
            // adversarial gradient on the RGB channels (1..3 of D input)
            // + L1 reconstruction gradient lambda * sign(fake - target).
            GErr[X, Y, 0] := DInGrad[X, Y, 1] +
              cLambdaL1 * Sign(GOut[X, Y, 0] - Colors[Idx][X, Y, 0]) / (cGrid * cGrid);
            GErr[X, Y, 1] := DInGrad[X, Y, 2] +
              cLambdaL1 * Sign(GOut[X, Y, 1] - Colors[Idx][X, Y, 1]) / (cGrid * cGrid);
            GErr[X, Y, 2] := DInGrad[X, Y, 3] +
              cLambdaL1 * Sign(GOut[X, Y, 2] - Colors[Idx][X, Y, 2]) / (cGrid * cGrid);
          end;

        G.ResetBackpropCallCurrCnt();
        G.GetLastLayer().OutputError.Copy(GErr);
        G.GetLastLayer().Backpropagate();
      end;

      if (Epoch = 1) or (Epoch mod 5 = 0) or (Epoch = gEpochs) then
      begin
        Evaluate(G, TGrays, TColors, TCls, L1, Acc);
        WriteLn(Format('  epoch %3d   test L1=%6.4f  colorAcc=%6.4f', [Epoch, L1, Acc]));
      end;
    end;
    WriteLn(Format('Training wall time: %.1f s', [(Now() - T0) * 24 * 3600]));

    Evaluate(G, TGrays, TColors, TCls, L1, Acc);
    WriteLn;
    WriteLn(Format('FINAL held-out  L1=%6.4f   colorAcc=%6.4f', [L1, Acc]));

    RenderSample(G, TGrays, TColors, 0);
    WritePPM(G, TGrays, TColors, 0, 'pix2pix_sample.ppm');

  finally
    FakePair.Free; RealPair.Free; GErr.Free; Ones.Free; Zeros.Free;
    G.Free; D.Free;
    FreeDataset(Grays, Colors, Cls);
    FreeDataset(TGrays, TColors, TCls);
  end;
end;

begin
  RunAlgo();
end.
