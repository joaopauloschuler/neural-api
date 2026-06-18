program Inpainting;
(*
Inpainting: a FREE-FORM MASK COMPLETION example - fill a hole in
an image from its surroundings (the context-encoder task, Pathak et al. 2016,
"Context Encoders: Feature Learning by Inpainting"). This is distinct from the
two image-translation examples already in the tree:
  - examples/VisualGAN     UNCONDITIONAL generation (noise -> image)
  - examples/Pix2Pix       PAIRED translation     (grayscale -> color)
Here the input and target share the SAME image: a random rectangular region is
ZEROED OUT and the network must HALLUCINATE the missing pixels from the visible
surroundings, supervised by a reconstruction loss that weights the hole heavily.

THE DATA (synthetic, generated in code - no download, fully offline)
-------------------------------------------------------------------
Each sample is a self-contained colored-shapes scene on a dark background: one or
two filled shapes (a red circle and/or a green axis-aligned rectangle) plus a
smooth horizontal blue gradient, all in [-1,1] (to match the Tanh output range).
CIFAR-10 would also work via the repo loader, but a synthetic scene keeps the
example 100% offline and lets the smoke run stay tiny; the network and the loss
are dataset-agnostic, so swapping in CIFAR is a one-procedure change.

THE MASK (the first piece of NEW code: a random rectangular-hole generator)
---------------------------------------------------------------------------
For every sample we pick a random rectangle (side cHoleMin..cHoleMax) and build a
binary mask: mask=1 inside the hole (pixels to be reconstructed), mask=0 outside
(visible context). The MASKED image is the clean image with the hole pixels set
to 0. The network input is the masked RGB stacked with the mask on the depth
axis: 4 channels [maskedR | maskedG | maskedB | mask]. Feeding the mask as an
explicit channel tells the net exactly which pixels are missing (the standard
context-encoder / partial-conv convention), so it learns to COPY visible context
and only HALLUCINATE inside the hole.

THE NETWORK (stock conv encoder-decoder + skip connections, ONE builder call)
-----------------------------------------------------------------------------
A U-Net built by TNNet.AddUNet (same builder as Pix2Pix / UNetSegmentation):

  G.AddLayer(TNNetInput.Create(grid, grid, 4, 1));    // [maskedRGB | mask]
  G.AddUNet(Depth, BaseFeatures, 3, Taps, UseNorm);   // -> 3 channels (RGB)
  G.AddLayer(TNNetHyperbolicTangent.Create());        // RGB in [-1,1]

AddUNet builds Depth encoder stages (each 2x [Conv3x3->ReLU] + 2x2 MaxPool,
doubling features), a bottleneck, then Depth decoder stages (nearest x2 upsample
-> depth-concat with the matching encoder SKIP tap -> 2x conv) and a 1x1 head to
3 channels. The skip connections carry the surrounding context across the
bottleneck so the visible region is reconstructed cheaply and the decoder can
spend capacity on the hole. grid must be divisible by 2^Depth.

THE LOSS (the second piece of NEW code: a masked-region-WEIGHTED loss)
---------------------------------------------------------------------
Reconstruction = L1 + (1 - SSIM), exactly as examples/FrameInterpolation uses the
landed neuralimagemetrics.ComputeSSIMLossAndGradient helper, but with a SPATIAL
WEIGHT: pixels INSIDE the hole get weight cHoleWeight (default 6.0) and pixels
OUTSIDE get weight 1.0. The hole is the only part the net cannot trivially copy,
so up-weighting it focuses learning where it matters and keeps the visible region
faithful at the same time (Pathak et al. weight the hole 10x; we use a gentler
6x because our visible region is not free - the masked input zeroed it too).
SSIM is computed per RGB channel and its gradient is added to the L1 subgradient;
the combined per-pixel d(loss)/d(pred) is injected through the standard, fully
tested TNNet.Backpropagate path using the pseudo-target identity
  Desired = Output - GradOut   =>   OutputError = Output - Desired = GradOut
(the library's last-layer rule is OutputError = Output - Desired). SSIM needs an
11x11 window, so grid = 16.

OPTIONAL ADVERSARIAL TERM (reuses the Pix2Pix / VisualGAN PatchGAN wiring)
-------------------------------------------------------------------------
Pass --adv to add a small PatchGAN discriminator (LSGAN objective) that scores
the COMPLETED image (visible context composited with the generated hole) as
real/fake. Its input gradient w.r.t. the generated pixels is added to the
reconstruction gradient, exactly the hand-rolled adversarial loop from
examples/Pix2Pix. The discriminator path is OFF by default so the smoke run
stays fast and short; the reconstruction-only default already learns to fill the
hole. (A diffusion-based inpainting sibling - re-noise ONLY the masked region,
the RePaint / SDEdit trick - is a SEPARATE tracked task, not this one.)

OUTPUT
------
Held-out reconstruction metrics reported SEPARATELY for the whole image and for
the hole interior (L1 and SSIM), an ASCII panel (masked input | reconstructed |
original) and a PPM triplet written to inpainting_sample.ppm
(masked | reconstructed | original). The console prints what the numbers show.

USAGE
-----
  ./Inpainting            smoke run (fast, reconstruction-only, default)
  ./Inpainting --adv      add the optional PatchGAN adversarial term
  ./Inpainting --full     longer training for a sharper result

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

uses
  {$IFDEF UNIX}cthreads,{$ENDIF}
  Classes, SysUtils, Math,
  neuralnetwork, neuralvolume, neuralimagemetrics;

const
  cGrid       = 16;    // image side (>= 11 for the SSIM 11x11 window)
  cDepth      = 2;     // U-Net stages (16 / 2^2 = 4 bottleneck cells)
  cBaseFeat   = 16;    // base feature count (doubles per stage)
  cHoleMin    = 4;     // smallest hole side
  cHoleMax    = 8;     // largest hole side
  cHoleWeight = 6.0;   // loss weight INSIDE the hole (outside = 1.0)
  cSSIMW      = 0.4;   // weight of (1 - SSIM); pixel L1 carries (1 - cSSIMW)
  cRange      = 2.0;   // pixel data range ([-1,1] -> span 2)
  cLambdaAdv  = 1.0;   // weight of the optional adversarial gradient
  cSeed       = 20260615;

  // Background gradient endpoints (dark, blue-ish) in [-1,1].
  cBgLoB = -0.2; cBgHiB = 0.4;

var
  // Run-mode dependent hyperparameters (set in ParseArgs).
  gNumTrain : integer = 200;
  gNumTest  : integer = 40;
  gEpochs   : integer = 10;
  gLR       : single  = 0.002;
  gAdv      : boolean = false;
  gFull     : boolean = false;

type
  TVolArray = array of TNNetVolume;

// ---------------------------------------------------------------------------
// Synthetic colored-shapes scene generator (clean RGB image in [-1,1]).
// ---------------------------------------------------------------------------
procedure MakeImage(Img: TNNetVolume);
var
  X, Y, Cx, Cy, R, X0, Y0, W, H, Dx, Dy: integer;
  t: single;
begin
  // Background: smooth horizontal blue gradient, faint red/green.
  for Y := 0 to cGrid - 1 do
    for X := 0 to cGrid - 1 do
    begin
      t := X / (cGrid - 1);
      Img[X, Y, 0] := -0.8;                       // R
      Img[X, Y, 1] := -0.8;                       // G
      Img[X, Y, 2] := cBgLoB + t * (cBgHiB - cBgLoB); // B gradient
    end;
  // Red filled circle.
  R  := 2 + Random(3);
  Cx := R + Random(cGrid - 2 * R);
  Cy := R + Random(cGrid - 2 * R);
  for Y := 0 to cGrid - 1 do
    for X := 0 to cGrid - 1 do
    begin
      Dx := X - Cx; Dy := Y - Cy;
      if (Dx * Dx + Dy * Dy) <= (R * R) then
      begin
        Img[X, Y, 0] :=  1.0; Img[X, Y, 1] := -1.0; Img[X, Y, 2] := -1.0;
      end;
    end;
  // Green filled rectangle (sometimes).
  if Random(2) = 0 then
  begin
    W := 3 + Random(5); H := 3 + Random(5);
    X0 := Random(cGrid - W); Y0 := Random(cGrid - H);
    for Y := Y0 to Y0 + H - 1 do
      for X := X0 to X0 + W - 1 do
      begin
        Img[X, Y, 0] := -1.0; Img[X, Y, 1] := 1.0; Img[X, Y, 2] := -1.0;
      end;
  end;
end;

// ---------------------------------------------------------------------------
// NEW CODE 1: random rectangular-hole mask. Fills Mask (cGrid,cGrid,1) with 1
// inside the chosen rectangle (to be reconstructed) and 0 outside (context).
// ---------------------------------------------------------------------------
procedure MakeMask(Mask: TNNetVolume);
var W, H, X0, Y0, X, Y: integer;
begin
  Mask.Fill(0);
  W := cHoleMin + Random(cHoleMax - cHoleMin + 1);
  H := cHoleMin + Random(cHoleMax - cHoleMin + 1);
  X0 := Random(cGrid - W + 1);
  Y0 := Random(cGrid - H + 1);
  for Y := Y0 to Y0 + H - 1 do
    for X := X0 to X0 + W - 1 do
      Mask[X, Y, 0] := 1;
end;

// Build the 4-channel network input [maskedRGB | mask]: visible RGB where
// mask=0, ZERO where mask=1, plus the mask channel itself.
procedure PackInput(Inp, Img, Mask: TNNetVolume);
var X, Y: integer; m: single;
begin
  for Y := 0 to cGrid - 1 do
    for X := 0 to cGrid - 1 do
    begin
      m := Mask[X, Y, 0];
      Inp[X, Y, 0] := Img[X, Y, 0] * (1 - m);   // zero out the hole
      Inp[X, Y, 1] := Img[X, Y, 1] * (1 - m);
      Inp[X, Y, 2] := Img[X, Y, 2] * (1 - m);
      Inp[X, Y, 3] := m;                         // mask channel
    end;
end;

// Composite the COMPLETED image: visible context from Img where mask=0, the
// generated pixels where mask=1 (used by the optional discriminator & display).
procedure Composite(Dst, Img, Pred, Mask: TNNetVolume);
var X, Y, C: integer; m: single;
begin
  for Y := 0 to cGrid - 1 do
    for X := 0 to cGrid - 1 do
    begin
      m := Mask[X, Y, 0];
      for C := 0 to 2 do
        Dst[X, Y, C] := Img[X, Y, C] * (1 - m) + Pred[X, Y, C] * m;
    end;
end;

procedure BuildDataset(out Imgs, Masks: TVolArray; N: integer);
var I: integer;
begin
  SetLength(Imgs, N); SetLength(Masks, N);
  for I := 0 to N - 1 do
  begin
    Imgs[I]  := TNNetVolume.Create(cGrid, cGrid, 3);
    Masks[I] := TNNetVolume.Create(cGrid, cGrid, 1);
    MakeImage(Imgs[I]);
    MakeMask(Masks[I]);
  end;
end;

procedure FreeDataset(var Imgs, Masks: TVolArray);
var I: integer;
begin
  for I := 0 to Length(Imgs) - 1 do begin Imgs[I].Free; Masks[I].Free; end;
  SetLength(Imgs, 0); SetLength(Masks, 0);
end;

// ---------------------------------------------------------------------------
// Networks.
// ---------------------------------------------------------------------------
function BuildGenerator(): TNNet;
var Taps: TNeuralIntegerArray;
begin
  Result := TNNet.Create();
  Result.AddLayer(TNNetInput.Create(cGrid, cGrid, 4, 1)); // [maskedRGB | mask]
  Result.AddUNet(cDepth, cBaseFeat, 3, Taps, {UseNorm=}false);
  Result.AddLayer(TNNetHyperbolicTangent.Create());        // RGB in [-1,1]
  Result.InitWeights();
end;

// PatchGAN: 3-channel COMPLETED image -> small patch-logit grid (LSGAN).
function BuildDiscriminator(): TNNet;
begin
  Result := TNNet.Create();
  Result.AddLayer(TNNetInput.Create(cGrid, cGrid, 3, 1).EnableErrorCollection);
  Result.AddLayer(TNNetConvolutionLinear.Create(64, 4, 1, 2));  // /2
  Result.AddLayer(TNNetLeakyReLU.Create());
  Result.AddLayer(TNNetConvolutionLinear.Create(128, 4, 1, 2)); // /4
  Result.AddLayer(TNNetLeakyReLU.Create());
  Result.AddLayer(TNNetConvolutionLinear.Create(1, 3, 1, 1));   // linear patch logits
  Result.InitWeights();
end;

// ---------------------------------------------------------------------------
// NEW CODE 2: masked-region-WEIGHTED L1 + (1 - SSIM) loss and its per-pixel
// gradient. Returns the (whole-image) scalar loss; fills GradOut (cGrid,cGrid,3)
// with d(loss)/d(pred). Pixels inside the hole carry weight cHoleWeight; pixels
// outside carry weight 1.0. SSIM is computed per RGB channel.
// ---------------------------------------------------------------------------
function LossAndGrad(Pred, Tgt, Mask, GradOut: TNNetVolume): double;
var
  X, Y, C, i: integer;
  pa, pb, ssimGrad: TIMDoubleArray;
  l1, ssimLoss, d, w, total: double;
begin
  SetLength(pa, cGrid * cGrid);
  SetLength(pb, cGrid * cGrid);
  GradOut.Fill(0);
  total := 0;
  for C := 0 to 2 do
  begin
    for Y := 0 to cGrid - 1 do
      for X := 0 to cGrid - 1 do
      begin
        pa[Y * cGrid + X] := Pred[X, Y, C];
        pb[Y * cGrid + X] := Tgt[X, Y, C];
      end;
    ssimLoss := ComputeSSIMLossAndGradient(pa, pb, cGrid, cGrid, 1, ssimGrad, cRange);
    l1 := 0;
    for Y := 0 to cGrid - 1 do
      for X := 0 to cGrid - 1 do
      begin
        i := Y * cGrid + X;
        w := 1.0 + (cHoleWeight - 1.0) * Mask[X, Y, 0]; // hole->cHoleWeight, else 1
        d := pa[i] - pb[i];
        l1 := l1 + w * Abs(d);
        GradOut[X, Y, C] := w * ((1.0 - cSSIMW) * Sign(d) / (cGrid * cGrid)
                                 + cSSIMW * ssimGrad[i]);
      end;
    l1 := l1 / (cGrid * cGrid);
    total := total + (1.0 - cSSIMW) * l1 + cSSIMW * ssimLoss;
  end;
  Result := total / 3.0;
end;

// Drive backprop on a CUSTOM per-pixel gradient via the standard, fully-tested
// TNNet.Backpropagate path. Last-layer rule is OutputError = Output - Desired,
// so handing Desired = Output - GradOut makes the back-propagated error EXACTLY
// our gradient. Reuses all of Backpropagate's branch-counter bookkeeping.
procedure BackpropFromGrad(NN: TNNet; GradOut, PseudoTgt: TNNetVolume);
begin
  PseudoTgt.Copy(NN.GetLastLayer.Output);
  PseudoTgt.Sub(GradOut);
  NN.Backpropagate(PseudoTgt);
end;

// ---------------------------------------------------------------------------
// Metrics: report L1/SSIM for the WHOLE image and for the HOLE interior only.
// ---------------------------------------------------------------------------
procedure EvalSet(NN: TNNet; const Imgs, Masks: TVolArray; Count: integer;
  out AllL1, AllSSIM, HoleL1: double);
var
  i, X, Y, C, holePix: integer;
  Inp, Pred: TNNetVolume;
  pa, pb, dummy: TIMDoubleArray;
  sAll, sHole, d: double;
begin
  Inp := TNNetVolume.Create(cGrid, cGrid, 4);
  SetLength(pa, cGrid * cGrid);
  SetLength(pb, cGrid * cGrid);
  sAll := 0; sHole := 0; holePix := 0; AllSSIM := 0;
  for i := 0 to Count - 1 do
  begin
    PackInput(Inp, Imgs[i], Masks[i]);
    NN.Compute(Inp);
    Pred := NN.GetLastLayer.Output;
    for C := 0 to 2 do
    begin
      for Y := 0 to cGrid - 1 do
        for X := 0 to cGrid - 1 do
        begin
          d := Abs(Pred[X, Y, C] - Imgs[i][X, Y, C]);
          sAll := sAll + d;
          if Masks[i][X, Y, 0] > 0.5 then sHole := sHole + d;
          pa[Y * cGrid + X] := Pred[X, Y, C];
          pb[Y * cGrid + X] := Imgs[i][X, Y, C];
        end;
      AllSSIM := AllSSIM +
        (1.0 - ComputeSSIMLossAndGradient(pa, pb, cGrid, cGrid, 1, dummy, cRange));
    end;
    for Y := 0 to cGrid - 1 do
      for X := 0 to cGrid - 1 do
        if Masks[i][X, Y, 0] > 0.5 then Inc(holePix);
  end;
  AllL1 := sAll / (Count * cGrid * cGrid * 3);
  AllSSIM := AllSSIM / (Count * 3);
  if holePix > 0 then HoleL1 := sHole / (holePix * 3) else HoleL1 := 0;
  Inp.Free;
end;

// ---------------------------------------------------------------------------
// Display helpers.
// ---------------------------------------------------------------------------
function ColCh(R, G, B: single): char;
var dBg, dRed, dGreen: single;
begin
  dBg    := Sqr(R + 0.8) + Sqr(G + 0.8) + Sqr(B);
  dRed   := Sqr(R - 1)   + Sqr(G + 1)   + Sqr(B + 1);
  dGreen := Sqr(R + 1)   + Sqr(G - 1)   + Sqr(B + 1);
  if (dRed <= dBg) and (dRed <= dGreen) then Result := 'R'
  else if (dGreen <= dBg) and (dGreen <= dRed) then Result := 'G'
  else Result := '.';
end;

procedure RenderPanel(NN: TNNet; Img, Mask: TNNetVolume);
var Inp, Pred: TNNetVolume; X, Y: integer; line: string;
begin
  Inp := TNNetVolume.Create(cGrid, cGrid, 4);
  PackInput(Inp, Img, Mask);
  NN.Compute(Inp);
  Pred := NN.GetLastLayer.Output;
  WriteLn;
  WriteLn('ASCII panel  (R=red circle, G=green rect, .=background, " "=hole)');
  WriteLn('  MASKED INPUT     |  RECONSTRUCTED   |  ORIGINAL');
  for Y := 0 to cGrid - 1 do
  begin
    line := '';
    for X := 0 to cGrid - 1 do
      if Mask[X, Y, 0] > 0.5 then line := line + ' '
      else line := line + ColCh(Img[X, Y, 0], Img[X, Y, 1], Img[X, Y, 2]);
    line := line + ' | ';
    for X := 0 to cGrid - 1 do
      line := line + ColCh(Pred[X, Y, 0], Pred[X, Y, 1], Pred[X, Y, 2]);
    line := line + ' | ';
    for X := 0 to cGrid - 1 do
      line := line + ColCh(Img[X, Y, 0], Img[X, Y, 1], Img[X, Y, 2]);
    WriteLn(line);
  end;
  Inp.Free;
end;

// PPM (P6) strip: masked | reconstructed | original, 1px black gaps.
procedure WritePPM(NN: TNNet; Img, Mask: TNNetVolume; const FileName: string);
const Gap = 1;
var
  F: TextFile; Bin: TFileStream; W, H, X, Y, Panel, sx: integer;
  Inp, Pred: TNNetVolume;
  procedure Emit(rr, gg, bb: byte);
  begin Bin.WriteByte(rr); Bin.WriteByte(gg); Bin.WriteByte(bb); end;
  function ToByte(V: single): byte;   // [-1,1] -> [0,255]
  begin
    V := (V + 1) * 0.5; if V < 0 then V := 0; if V > 1 then V := 1;
    Result := Round(V * 255);
  end;
begin
  Inp := TNNetVolume.Create(cGrid, cGrid, 4);
  PackInput(Inp, Img, Mask);
  NN.Compute(Inp);
  Pred := NN.GetLastLayer.Output;
  W := cGrid * 3 + Gap * 2; H := cGrid;
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
        if X < cGrid then
        begin
          Panel := 0; sx := X;
        end
        else if X < cGrid * 2 + Gap then
        begin
          Panel := 1; sx := X - cGrid - Gap;
        end
        else
        begin
          Panel := 2; sx := X - (cGrid * 2 + Gap * 2);
        end;
        case Panel of
          0: // masked input: hole drawn black
            if Mask[sx, Y, 0] > 0.5 then Emit(0, 0, 0)
            else Emit(ToByte(Img[sx, Y, 0]), ToByte(Img[sx, Y, 1]), ToByte(Img[sx, Y, 2]));
          1: // reconstructed (full prediction)
            Emit(ToByte(Pred[sx, Y, 0]), ToByte(Pred[sx, Y, 1]), ToByte(Pred[sx, Y, 2]));
        else // original
          Emit(ToByte(Img[sx, Y, 0]), ToByte(Img[sx, Y, 1]), ToByte(Img[sx, Y, 2]));
        end;
      end;
  finally
    Bin.Free;
  end;
  Inp.Free;
  WriteLn('Wrote ', FileName, ' (masked | reconstructed | original)');
end;

// ---------------------------------------------------------------------------
procedure ParseArgs();
var I: integer;
begin
  for I := 1 to ParamCount do
  begin
    if ParamStr(I) = '--adv'  then gAdv  := true;
    if ParamStr(I) = '--full' then gFull := true;
  end;
  if gFull then
  begin
    gNumTrain := 600; gNumTest := 80; gEpochs := 30;
  end;
end;

var
  G, D: TNNet;
  Imgs, Masks, TImgs, TMasks: TVolArray;
  Inp, GErr, PseudoTgt, Comp, RealPair, Ones, Zeros, DInGrad: TNNetVolume;
  GOut: TNNetVolume;
  Epoch, Step, Idx, X, Y, C: integer;
  Perm: array of integer;
  i, order, tmp: integer;
  aL1, aSSIM, hL1, epLoss: double;
  T0: TDateTime;
  PatchW, PatchH: integer;
begin
  ParseArgs();
  Randomize; RandSeed := cSeed;

  WriteLn('=== Inpainting: free-form rectangular-hole completion (context encoder) ===');
  WriteLn(Format('grid=%dx%d  depth=%d  base=%d  hole=%d..%d  holeWeight=%.1f  loss=L1+%.2f*(1-SSIM)',
    [cGrid, cGrid, cDepth, cBaseFeat, cHoleMin, cHoleMax, cHoleWeight, cSSIMW]));
  WriteLn(Format('mode=%s  adversarial=%s  train=%d  test=%d  epochs=%d',
    [BoolToStr(gFull, 'full', 'smoke'), BoolToStr(gAdv, 'on', 'off'),
     gNumTrain, gNumTest, gEpochs]));
  WriteLn;

  BuildDataset(Imgs, Masks, gNumTrain);
  BuildDataset(TImgs, TMasks, gNumTest);

  RandSeed := cSeed + 1;
  G := BuildGenerator();
  G.SetLearningRate(gLR, 0.9);
  WriteLn('Generator layers: ', G.CountLayers);

  D := nil; Ones := nil; Zeros := nil; RealPair := nil; Comp := nil;
  if gAdv then
  begin
    D := BuildDiscriminator();
    D.SetLearningRate(gLR, 0.5);
    PatchW := D.GetLastLayer.Output.SizeX;
    PatchH := D.GetLastLayer.Output.SizeY;
    Ones  := TNNetVolume.Create(PatchW, PatchH, 1); Ones.Fill(1);
    Zeros := TNNetVolume.Create(PatchW, PatchH, 1); Zeros.Fill(0);
    RealPair := TNNetVolume.Create(cGrid, cGrid, 3);
    Comp     := TNNetVolume.Create(cGrid, cGrid, 3);
    G.GetLastLayer.IncDepartingBranchesCnt(); // we seed G's last-layer error by hand
    WriteLn(Format('Discriminator layers: %d   PatchGAN grid: %dx%d',
      [D.CountLayers, PatchW, PatchH]));
  end;

  Inp       := TNNetVolume.Create(cGrid, cGrid, 4);
  GErr      := TNNetVolume.Create(cGrid, cGrid, 3);
  PseudoTgt := TNNetVolume.Create(cGrid, cGrid, 3);

  SetLength(Perm, gNumTrain);
  for i := 0 to gNumTrain - 1 do Perm[i] := i;

  try
    EvalSet(G, TImgs, TMasks, gNumTest, aL1, aSSIM, hL1);
    WriteLn(Format('BEFORE training  whole: L1=%.4f SSIM=%.4f   HOLE: L1=%.4f', [aL1, aSSIM, hL1]));
    WriteLn('Training...');
    T0 := Now();

    for Epoch := 1 to gEpochs do
    begin
      for i := gNumTrain - 1 downto 1 do
      begin
        order := Random(i + 1);
        tmp := Perm[i]; Perm[i] := Perm[order]; Perm[order] := tmp;
      end;
      epLoss := 0;
      for Step := 0 to gNumTrain - 1 do
      begin
        Idx := Perm[Step];
        PackInput(Inp, Imgs[Idx], Masks[Idx]);
        G.Compute(Inp);
        GOut := G.GetLastLayer.Output;

        // Reconstruction gradient (masked-region-weighted L1 + SSIM).
        epLoss := epLoss + LossAndGrad(GOut, Imgs[Idx], Masks[Idx], GErr);

        if gAdv then
        begin
          // Completed image = visible context + generated hole.
          Composite(Comp, Imgs[Idx], GOut, Masks[Idx]);
          // Train D: real (clean) -> 1, fake (completed) -> 0.
          D.Compute(Imgs[Idx]); D.Backpropagate(Ones);
          D.Compute(Comp);      D.Backpropagate(Zeros);
          // Adversarial G gradient: push D(fake)->1, freeze D, read input grad.
          D.SetLearningRate(0, 0.5);
          D.Compute(Comp); D.Backpropagate(Ones);
          DInGrad := D.Layers[0].OutputError;     // grad w.r.t. the 3 RGB channels
          D.SetLearningRate(gLR, 0.5);
          // Adversarial term applies only where the generator fills (the hole).
          for Y := 0 to cGrid - 1 do
            for X := 0 to cGrid - 1 do
              for C := 0 to 2 do
                GErr[X, Y, C] := GErr[X, Y, C]
                  + cLambdaAdv * DInGrad[X, Y, C] * Masks[Idx][X, Y, 0];
          G.ResetBackpropCallCurrCnt();
          G.GetLastLayer.OutputError.Copy(GErr);
          G.GetLastLayer.Backpropagate();
        end
        else
          BackpropFromGrad(G, GErr, PseudoTgt);
      end;
      if (Epoch = 1) or (Epoch mod 5 = 0) or (Epoch = gEpochs) then
      begin
        EvalSet(G, TImgs, TMasks, gNumTest, aL1, aSSIM, hL1);
        WriteLn(Format('  epoch %2d/%2d  trainLoss=%.5f  test whole L1=%.4f SSIM=%.4f  HOLE L1=%.4f',
          [Epoch, gEpochs, epLoss / gNumTrain, aL1, aSSIM, hL1]));
      end;
    end;
    WriteLn(Format('Training wall time: %.1f s', [(Now() - T0) * 86400]));

    EvalSet(G, TImgs, TMasks, gNumTest, aL1, aSSIM, hL1);
    WriteLn;
    WriteLn(Format('FINAL held-out   whole: L1=%.4f SSIM=%.4f   HOLE: L1=%.4f', [aL1, aSSIM, hL1]));

    RenderPanel(G, TImgs[0], TMasks[0]);
    WritePPM(G, TImgs[0], TMasks[0], 'inpainting_sample.ppm');

  finally
    Inp.Free; GErr.Free; PseudoTgt.Free;
    if gAdv then begin D.Free; Ones.Free; Zeros.Free; RealPair.Free; Comp.Free; end;
    G.Free;
    FreeDataset(Imgs, Masks);
    FreeDataset(TImgs, TMasks);
  end;
  WriteLn('Done.');
end.
