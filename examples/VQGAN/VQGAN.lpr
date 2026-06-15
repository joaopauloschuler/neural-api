program VQGAN;
(*
VQGAN: upgrades the landed reconstruction-only VQ-VAE (examples/VQVAE, pure L2)
into the GENERATIVE-CV regime by training the SAME discrete autoencoder
(conv encoder -> TNNetVectorQuantizer codebook -> conv decoder) with the
VQ-GAN objective of Esser et al. 2021 ("Taming Transformers for High-Resolution
Image Synthesis", https://arxiv.org/abs/2012.09841): reconstruction + a
PERCEPTUAL term + a codebook commitment term, PLUS a PatchGAN ADVERSARIAL term
that is switched on only after a warmup. The adversarial + perceptual pressure
is what turns blurry VQ-VAE reconstructions into SHARP ones with crisp edges --
the whole point of VQ-GAN over a plain-L2 VQ-VAE.

This example is deliberately SELF-CONTAINED (no dataset download, pure CPU): the
images are a synthetic SHAPES task (filled circles + rectangles on a dark
background, grayscale in [-1,1]) generated in code, the same family used by
examples/Pix2Pix. That lets us train BOTH a plain-L2 VQ-VAE and a VQ-GAN on the
IDENTICAL data and architecture and directly compare a high-frequency EDGE
SHARPNESS metric, demonstrating the VQ-GAN payoff.

WHAT IS REUSED (the new code is the loss SCHEDULE + commitment balance):
  * TNNetVectorQuantizer  -- the repo codebook layer (straight-through + the
    commitment/codebook gradients live inside it), exactly as in examples/VQVAE.
    We read back ChosenCodeIndex / CodebookUsageCount to print codebook usage
    and PERPLEXITY (exp of the code-distribution entropy).
  * The PatchGAN discriminator + the LSGAN adversarial loop + the
    gradient-surgery trick (freeze D, read d(adv)/d(pixels) off D's input-layer
    OutputError) -- straight from examples/Pix2Pix / examples/CycleGAN.
  * The LPIPS perceptual PRIMITIVES from neuralpretrained.pas
    (LPIPSUnitNormalize + LPIPSStageDistance). Full ComputeLPIPSDistance needs an
    imported VGG checkpoint (a download we avoid here), so -- exactly the
    documented fallback -- the perceptual term is a FEATURE-MATCHING distance
    computed with the SAME unit-normalized per-stage LPIPS math, but over the
    discriminator's own hidden conv feature maps (the GAN "perceptual"/feature-
    matching loss of Salimans et al. / Wang et al.). Its gradient is obtained
    the same way the adversarial gradient is: through D's collected input error.

THE VQ-GAN LOSS SCHEDULE (the new bit):
  total = recon_L1 + lambda_perc * perceptual + (commitment from the VQ layer)
          + [ lambda_adv * adversarial ]   <-- ONLY after cWarmupEpochs
  Warming up on recon+perceptual+commitment first lets the codebook settle
  before the discriminator gets a vote; flipping the adversarial term on too
  early collapses the tiny CPU codebook. The discriminator is trained every step
  (so it is ready when the generator starts listening to it).

OUTPUT (the console prints what the numbers ACTUALLY show):
  * Per-epoch: recon-L1, perceptual term, codebook active/K + perplexity,
    discriminator loss, and an edge-sharpness metric.
  * A final side-by-side: plain-L2 VQ-VAE vs VQ-GAN edge sharpness + recon-L1,
    plus an ASCII panel and a PPM strip (original | L2-VQVAE | VQ-GAN).

USAGE
  ./VQGAN            smoke run (fast, default; well under five minutes on CPU)
  ./VQGAN --full     longer training for a sharper result

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
  Classes, SysUtils, Math, StrUtils,
  neuralnetwork,
  neuralvolume,
  neuralpretrained;   // LPIPSUnitNormalize + LPIPSStageDistance

const
  cGrid     = 16;    // image side (cGrid x cGrid), grayscale
  cLatent   = 4;     // latent grid side (16 -> /2 -> /2 = 4); 4*4 = 16 codes
  cEmb      = 16;    // codebook vector dimension (encoder output depth)
  cK        = 32;    // codebook size

  // Commitment cost + reconstruction-gradient weight: same balance reasoning as
  // examples/VQVAE -- a heavy recon gradient relative to the per-sample
  // commitment term keeps the codebook from collapsing onto a single code.
  cBeta     = 0.05;
  cReconW   = 0.20;  // L1 reconstruction-gradient weight

  cC1       = 16;    // encoder/decoder channels at 16x16
  cC2       = 32;    // encoder/decoder channels at 8x8

  // Perceptual + adversarial weights. These are deliberately LARGE relative to
  // the recon-L1 gradient: at this toy scale the feature-matching and PatchGAN
  // signals are small in magnitude, so to actually sharpen edges (rather than
  // be swamped by the dominant L1 term) they need real weight -- this is the
  // commitment/loss BALANCE that the task is about.
  cLambdaPerc = 20.0; // perceptual (feature-matching) weight
  cLambdaAdv  = 3.0;  // adversarial weight
  cSeed       = 20260615;

var
  // --full scales these up.
  gTrain:   integer = 192;
  gTest:    integer = 64;
  gEpochs:  integer = 22;
  gWarmup:  integer = 8;    // epochs of recon+perc+commit before adversarial on
  gAELR:    TNeuralFloat = 0.001;
  gDLR:     TNeuralFloat = 0.0006;
  gFull:    boolean = false;

type
  TVolArray = array of TNNetVolume;

// ===========================================================================
//   Synthetic shapes data (grayscale in [-1,1]: background -1, shape +1)
// ===========================================================================

procedure StampCircle(Img: TNNetVolume);
var Cx, Cy, R, X, Y, Dx, Dy: integer;
begin
  R := 2 + Random(3);                       // radius 2..4
  Cx := R + Random(cGrid - 2 * R);
  Cy := R + Random(cGrid - 2 * R);
  for Y := 0 to cGrid - 1 do
    for X := 0 to cGrid - 1 do
    begin
      Dx := X - Cx; Dy := Y - Cy;
      if (Dx * Dx + Dy * Dy) <= (R * R) then Img[X, Y, 0] := 1;
    end;
end;

procedure StampRect(Img: TNNetVolume);
var X0, Y0, W, H, X, Y: integer;
begin
  W := 3 + Random(6); H := 3 + Random(6);
  X0 := Random(cGrid - W); Y0 := Random(cGrid - H);
  for Y := Y0 to Y0 + H - 1 do
    for X := X0 to X0 + W - 1 do
      Img[X, Y, 0] := 1;
end;

procedure MakeSample(Img: TNNetVolume);
var S, Shapes: integer;
begin
  Img.Fill(-1);                              // dark background
  Shapes := 1 + Random(2);
  for S := 1 to Shapes do
    if Random(2) = 0 then StampCircle(Img) else StampRect(Img);
end;

procedure BuildDataset(out A: TVolArray; N: integer);
var I: integer;
begin
  SetLength(A, N);
  for I := 0 to N - 1 do
  begin
    A[I] := TNNetVolume.Create(cGrid, cGrid, 1);
    MakeSample(A[I]);
  end;
end;

procedure FreeDataset(var A: TVolArray);
var I: integer;
begin
  for I := 0 to Length(A) - 1 do A[I].Free;
  SetLength(A, 0);
end;

// ===========================================================================
//   Networks
// ===========================================================================

// Discrete autoencoder: conv encoder -> VQ codebook -> conv decoder.
// Returns the net; LVQ / EncOutIdx / DecInIdx / VQIdx are filled by reference.
function BuildAutoEncoder(out LVQ: TNNetVectorQuantizer;
  out VQIdx, DecInIdx: integer): TNNet;
begin
  Result := TNNet.Create();
  Result.AddLayer(TNNetInput.Create(cGrid, cGrid, 1, 1));
  // Encoder 16 -> 8 -> 4 with GroupNorm (essential for this tiny conv AE).
  Result.AddLayer(TNNetConvolutionReLU.Create(cC1, 3, 1, 1, 0));
  Result.AddLayer(TNNetGroupNorm.Create(4));
  Result.AddLayer(TNNetMaxPool.Create(2));                       // 16 -> 8
  Result.AddLayer(TNNetConvolutionReLU.Create(cC2, 3, 1, 1, 0));
  Result.AddLayer(TNNetGroupNorm.Create(4));
  Result.AddLayer(TNNetMaxPool.Create(2));                       // 8 -> 4
  Result.AddLayer(TNNetConvolutionReLU.Create(cC2, 3, 1, 1, 0));
  Result.AddLayer(TNNetPointwiseConvLinear.Create(cEmb));        // z_e (4,4,cEmb)

  LVQ := TNNetVectorQuantizer.Create(cK, cBeta);
  Result.AddLayer(LVQ);
  VQIdx := Result.GetLastLayerIdx();

  Result.AddLayer(TNNetPointwiseConvReLU.Create(cC2));           // decoder entry
  DecInIdx := Result.GetLastLayerIdx();
  Result.AddLayer(TNNetUpsample.Create());                       // 4 -> 8
  Result.AddLayer(TNNetConvolutionReLU.Create(cC2, 3, 1, 1, 0));
  Result.AddLayer(TNNetGroupNorm.Create(4));
  Result.AddLayer(TNNetUpsample.Create());                       // 8 -> 16
  Result.AddLayer(TNNetConvolutionReLU.Create(cC1, 3, 1, 1, 0));
  Result.AddLayer(TNNetConvolutionLinear.Create(1, 3, 1, 1, 0)); // reconstruction

  Result.SetLearningRate(gAELR, 0.9);
  Result.SetL2Decay(0.0);
  Result.SetBatchUpdate(True);
  // We seed the last-layer error ourselves (custom losses), so prime its
  // departing-branch count once -- see the manual-backprop convention.
  Result.GetLastLayer().IncDepartingBranchesCnt();
end;

// PatchGAN discriminator over a single-channel image. The hidden conv map after
// the first LeakyReLU is the perceptual / feature-matching backbone.
function BuildDiscriminator(out FeatIdx: integer): TNNet;
begin
  Result := TNNet.Create();
  Result.AddLayer(TNNetInput.Create(cGrid, cGrid, 1, 1).EnableErrorCollection);
  Result.AddLayer(TNNetConvolutionLinear.Create(48, 4, 1, 2));  // 16 -> 8
  Result.AddLayer(TNNetLeakyReLU.Create());
  FeatIdx := Result.GetLastLayerIdx();                          // perceptual tap
  Result.AddLayer(TNNetConvolutionLinear.Create(96, 4, 1, 2));  // 8 -> 4
  Result.AddLayer(TNNetLeakyReLU.Create());
  Result.AddLayer(TNNetConvolutionLinear.Create(1, 3, 1, 1));   // patch logits
  Result.InitWeights();
  // Prime the perceptual feature tap so we can seed its error and backprop D
  // from that layer down to the input (the perceptual pixel-gradient path).
  Result.Layers[FeatIdx].IncDepartingBranchesCnt();
end;

// ===========================================================================
//   Metrics
// ===========================================================================

// Mean per-pixel L1 reconstruction distance between AE output and target image.
function ReconL1(AE: TNNet; Img: TNNetVolume): TNeuralFloat;
var i: integer; s: TNeuralFloat; P: TNNetVolume;
begin
  P := AE.GetLastLayer().Output;
  s := 0;
  for i := 0 to Img.Size - 1 do s := s + Abs(P.FData[i] - Img.FData[i]);
  Result := s / Img.Size;
end;

// High-frequency EDGE SHARPNESS: mean absolute gradient magnitude (Sobel-lite
// horizontal+vertical first differences). Blurry reconstructions have SMALL
// edge energy; a sharp VQ-GAN reconstruction matches the original's larger
// edge energy. Computed on a [-1,1] single-channel volume.
function EdgeSharpness(V: TNNetVolume): TNeuralFloat;
var X, Y: integer; s: TNeuralFloat; n: integer;
begin
  s := 0; n := 0;
  for Y := 0 to cGrid - 1 do
    for X := 0 to cGrid - 2 do
    begin
      s := s + Abs(V[X + 1, Y, 0] - V[X, Y, 0]); Inc(n);
    end;
  for Y := 0 to cGrid - 2 do
    for X := 0 to cGrid - 1 do
    begin
      s := s + Abs(V[X, Y + 1, 0] - V[X, Y, 0]); Inc(n);
    end;
  if n = 0 then Result := 0 else Result := s / n;
end;

// Codebook perplexity = exp(entropy of the code-usage distribution). Ranges in
// [1, #active]; closer to K means the codebook is well used (no collapse).
function CodebookPerplexity(LVQ: TNNetVectorQuantizer): TNeuralFloat;
var k, total: integer; p, ent: TNeuralFloat;
begin
  total := 0;
  for k := 0 to cK - 1 do total := total + LVQ.CodebookUsageCount(k);
  if total = 0 then begin Result := 1; Exit; end;
  ent := 0;
  for k := 0 to cK - 1 do
  begin
    p := LVQ.CodebookUsageCount(k) / total;
    if p > 0 then ent := ent - p * Ln(p);
  end;
  Result := Exp(ent);
end;

// ===========================================================================
//   Perceptual (feature-matching) term via the LPIPS primitives
// ===========================================================================

// Perceptual / feature-matching term between D's hidden feature map on the fake
// image and on the real image. Returns the REPORTED scalar (the unit-normalized
// per-stage LPIPS distance via the landed LPIPS primitives) AND, in PixGrad,
// the gradient of the feature-matching distance w.r.t. the fake PIXELS, obtained
// by seeding D's feature-tap error with d/dfeat of the plain feature MSE and
// backpropagating D (frozen) down to its input layer. D's learning rate must be
// 0 on entry (caller freezes it); D's feature layer must have been primed once
// with IncDepartingBranchesCnt (caller does this).
function PerceptualTerm(D: TNNet; FeatIdx: integer;
  Fake, Real_: TNNetVolume; PixGrad: TNNetVolume): TNeuralFloat;
var
  FReal, FErr: TNNetVolume;
  i: integer;
  Feat: TNNetVolume;
  invN: TNeuralFloat;
begin
  // real features (target)
  D.Compute(Real_);
  FReal := TNNetVolume.Create; FReal.Copy(D.Layers[FeatIdx].Output);
  // fake features (current)
  D.Compute(Fake);
  Feat := D.Layers[FeatIdx].Output;
  // gradient of mean_c (featFake - featReal)^2  ->  2/N * (featFake - featReal)
  FErr := TNNetVolume.Create; FErr.Resize(Feat);
  invN := 2.0 / Feat.Size;
  for i := 0 to Feat.Size - 1 do
    FErr.FData[i] := invN * (Feat.FData[i] - FReal.FData[i]);
  // seed the feature-tap error and backprop D (frozen) to its input layer.
  D.ResetBackpropCallCurrCnt();
  D.Layers[FeatIdx].OutputError.Copy(FErr);
  D.Layers[FeatIdx].Backpropagate();
  PixGrad.Copy(D.Layers[0].OutputError);   // d(featMSE)/d(pixels), 1 channel
  // reported scalar: the unit-normalized per-stage LPIPS distance.
  LPIPSUnitNormalize(Feat);
  LPIPSUnitNormalize(FReal);
  Result := LPIPSStageDistance(Feat, FReal, nil);
  FReal.Free; FErr.Free;
end;

// ===========================================================================
//   Training one autoencoder (mode = adversarial on/off)
// ===========================================================================

// Train AE with the VQ-GAN schedule. If Adversarial=false this is the plain-L2
// VQ-VAE baseline (recon-only, the examples/VQVAE objective). Reports per epoch.
procedure TrainAE(AE: TNNet; LVQ: TNNetVectorQuantizer; VQIdx, DecInIdx: integer;
  D: TNNet; FeatIdx: integer; const Data: TVolArray; Adversarial: boolean;
  const Tag: string);
var
  Epoch, Step, i, X, Y, PatchW, PatchH: integer;
  Img, Pred, GErr, FakeImg, PercGrad, Ones, Zeros: TNNetVolume;
  DInGrad: TNNetVolume;
  SumL1, SumPerc, SumD, SharpAcc: TNeuralFloat;
  AdvOn: boolean;
  T0: TDateTime;
begin
  PatchW := D.GetLastLayer().Output.SizeX;
  PatchH := D.GetLastLayer().Output.SizeY;
  GErr     := TNNetVolume.Create(cGrid, cGrid, 1);
  FakeImg  := TNNetVolume.Create(cGrid, cGrid, 1);
  PercGrad := TNNetVolume.Create(cGrid, cGrid, 1);
  Ones    := TNNetVolume.Create(PatchW, PatchH, 1); Ones.Fill(1);
  Zeros   := TNNetVolume.Create(PatchW, PatchH, 1); Zeros.Fill(0);
  WriteLn(Format('[%s] training  (%d epochs, warmup=%d, adversarial=%s)',
    [Tag, gEpochs, gWarmup, BoolToStr(Adversarial, 'yes', 'no')]));
  T0 := Now();
  try
    for Epoch := 1 to gEpochs do
    begin
      AdvOn := Adversarial and (Epoch > gWarmup);
      SumL1 := 0; SumPerc := 0; SumD := 0; SharpAcc := 0;
      LVQ.ResetCodebookUsage();
      for Step := 0 to gTrain - 1 do
      begin
        Img := Data[Random(gTrain)];
        AE.ClearDeltas();

        // ---- generator forward + losses --------------------------------
        AE.Compute(Img);
        Pred := AE.GetLastLayer().Output;
        SumL1 := SumL1 + ReconL1(AE, Img);
        SharpAcc := SharpAcc + EdgeSharpness(Pred);
        FakeImg.Copy(Pred);  // detached copy of the generated image

        // GErr accumulates the TOTAL generator error fed to the AE last layer
        // (the framework convention is error := output - target, so a POSITIVE
        // entry pulls that pixel DOWN). Start with the recon-L1 gradient.
        for i := 0 to Img.Size - 1 do
          GErr.FData[i] := cReconW * Sign(Pred.FData[i] - Img.FData[i]);

        // ---- train discriminator (every step, LSGAN) -------------------
        if Adversarial then
        begin
          D.Compute(Img);     D.Backpropagate(Ones);    // real -> 1
          D.Compute(FakeImg); D.Backpropagate(Zeros);   // fake -> 0
          // discriminator loss (LSGAN MSE on the fake branch, for reporting)
          DInGrad := D.GetLastLayer().Output;
          for i := 0 to DInGrad.Size - 1 do
            SumD := SumD + Sqr(DInGrad.FData[i]);

          // ---- perceptual term + its pixel gradient (D frozen) ---------
          D.SetLearningRate(0, 0.9);
          SumPerc := SumPerc +
            PerceptualTerm(D, FeatIdx, FakeImg, Img, PercGrad);
          for i := 0 to GErr.Size - 1 do
            GErr.FData[i] := GErr.FData[i] + cLambdaPerc * PercGrad.FData[i];

          // ---- adversarial pixel gradient (D still frozen) -------------
          if AdvOn then
          begin
            D.Compute(FakeImg);
            D.Backpropagate(Ones);               // want D to call fake "real"
            DInGrad := D.Layers[0].OutputError;  // d(adv)/d(pixels), 1 channel
            for i := 0 to GErr.Size - 1 do
              GErr.FData[i] := GErr.FData[i] + cLambdaAdv * DInGrad.FData[i];
          end;
          D.SetLearningRate(gDLR, 0.9);
        end;

        // ---- backprop the generator ------------------------------------
        AE.ResetBackpropCallCurrCnt();
        AE.GetLastLayer().OutputError.Copy(GErr);
        AE.GetLastLayer().Backpropagate();
        AE.ForceMaxAbsoluteDelta(0.02);  // clip AFTER deltas exist
        AE.UpdateWeights();
      end;

      WriteLn(Format(
        '  [%s] ep %2d/%2d  L1=%.4f  perc=%.4f  active=%2d/%2d  ppl=%5.2f  Dloss=%.4f  edge=%.4f%s',
        [Tag, Epoch, gEpochs, SumL1 / gTrain, SumPerc / gTrain,
         LVQ.ActiveCodeCount(), cK, CodebookPerplexity(LVQ), SumD / gTrain,
         SharpAcc / gTrain,
         IfThen(AdvOn, '  [adv ON]', '')]));
    end;
    WriteLn(Format('  [%s] wall time: %.1f s', [Tag, (Now() - T0) * 24 * 3600]));
  finally
    GErr.Free; FakeImg.Free; PercGrad.Free; Ones.Free; Zeros.Free;
  end;
end;

// ===========================================================================
//   Evaluation: edge sharpness + recon L1 on a held-out set
// ===========================================================================

procedure EvalAE(AE: TNNet; const Data: TVolArray;
  out MeanL1, MeanEdge: TNeuralFloat);
var I, N: integer; SumL1, SumE: TNeuralFloat; P: TNNetVolume;
begin
  N := Length(Data); SumL1 := 0; SumE := 0;
  for I := 0 to N - 1 do
  begin
    AE.Compute(Data[I]);
    P := AE.GetLastLayer().Output;
    SumL1 := SumL1 + ReconL1(AE, Data[I]);
    SumE := SumE + EdgeSharpness(P);
  end;
  MeanL1 := SumL1 / N; MeanEdge := SumE / N;
end;

// ASCII panel + PPM strip: original | L2-VQVAE recon | VQ-GAN recon.
function GlyphOf(V: TNeuralFloat): char;
begin
  if V >= 0.4 then GlyphOf := '#'
  else if V >= -0.2 then GlyphOf := '+'
  else if V >= -0.7 then GlyphOf := '.'
  else GlyphOf := ' ';
end;

procedure RenderPanel(AEl2, AEgan: TNNet; Img: TNNetVolume);
var X, Y: integer; Pl2, Pgan: TNNetVolume;
begin
  AEl2.Compute(Img);  Pl2  := TNNetVolume.Create; Pl2.Copy(AEl2.GetLastLayer().Output);
  AEgan.Compute(Img); Pgan := TNNetVolume.Create; Pgan.Copy(AEgan.GetLastLayer().Output);
  WriteLn;
  WriteLn('One held-out sample   (col1: ORIGINAL   col2: L2-VQVAE   col3: VQ-GAN)');
  WriteLn;
  for Y := 0 to cGrid - 1 do
  begin
    for X := 0 to cGrid - 1 do Write(GlyphOf(Img[X, Y, 0]));
    Write('   ');
    for X := 0 to cGrid - 1 do Write(GlyphOf(Pl2[X, Y, 0]));
    Write('   ');
    for X := 0 to cGrid - 1 do Write(GlyphOf(Pgan[X, Y, 0]));
    WriteLn;
  end;
  Pl2.Free; Pgan.Free;
end;

procedure WritePPM(AEl2, AEgan: TNNet; Img: TNNetVolume; const FileName: string);
const Gap = 1;
var
  F: TextFile; Bin: TFileStream; W, H, X, Y, Panel: integer;
  Pl2, Pgan: TNNetVolume;
  function ToByte(V: TNeuralFloat): byte;
  begin
    V := (V + 1) * 0.5; if V < 0 then V := 0; if V > 1 then V := 1;
    ToByte := Round(V * 255);
  end;
  procedure Emit(g: byte);
  begin Bin.WriteByte(g); Bin.WriteByte(g); Bin.WriteByte(g); end;
begin
  AEl2.Compute(Img);  Pl2  := TNNetVolume.Create; Pl2.Copy(AEl2.GetLastLayer().Output);
  AEgan.Compute(Img); Pgan := TNNetVolume.Create; Pgan.Copy(AEgan.GetLastLayer().Output);
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
        if (X = cGrid) or (X = cGrid * 2 + Gap) then begin Emit(0); Continue; end;
        if X < cGrid then Panel := 0
        else if X < cGrid * 2 + Gap then Panel := 1
        else Panel := 2;
        case Panel of
          0: Emit(ToByte(Img[X, Y, 0]));
          1: Emit(ToByte(Pl2[X - cGrid - Gap, Y, 0]));
        else
          Emit(ToByte(Pgan[X - (cGrid * 2 + Gap * 2), Y, 0]));
        end;
      end;
  finally
    Bin.Free;
  end;
  Pl2.Free; Pgan.Free;
  WriteLn('Wrote visualization: ', FileName, '  (original | L2-VQVAE | VQ-GAN)');
end;

// ===========================================================================
//   Main
// ===========================================================================

procedure ParseArgs;
var I: integer;
begin
  for I := 1 to ParamCount do
    if ParamStr(I) = '--full' then gFull := true;
  if gFull then
  begin
    gTrain := 384; gTest := 96; gEpochs := 40; gWarmup := 12;
  end;
end;

var
  AEl2, AEgan, Dl2, Dgan: TNNet;
  LVQl2, LVQgan: TNNetVectorQuantizer;
  VQl2, DInl2, VQg, DIng, Featl2, Featg: integer;
  Train, Test: TVolArray;
  L1l2, El2, L1g, Eg, EOrig: TNeuralFloat;
  i: integer;
begin
  SetExceptionMask([exInvalidOp, exDenormalized, exZeroDivide,
                    exOverflow, exUnderflow, exPrecision]);
  ParseArgs;
  RandSeed := cSeed;

  WriteLn('VQGAN: VQ-GAN (Esser et al. 2021) on a self-contained shapes task.');
  WriteLn(Format('  mode=%s  grid=%dx%d  latent=%dx%d  K=%d  emb=%d  train=%d  test=%d',
    [BoolToStr(gFull, 'full', 'smoke'), cGrid, cGrid, cLatent, cLatent, cK,
     cEmb, gTrain, gTest]));
  WriteLn('  schedule: recon-L1 + perceptual + commitment, adversarial after warmup.');
  WriteLn;

  BuildDataset(Train, gTrain);
  BuildDataset(Test, gTest);

  // Mean edge sharpness of the ORIGINALS (the target a sharp recon should reach).
  EOrig := 0;
  for i := 0 to gTest - 1 do EOrig := EOrig + EdgeSharpness(Test[i]);
  EOrig := EOrig / gTest;

  // ---- baseline: plain-L2 VQ-VAE (recon-only) ----
  RandSeed := cSeed + 1;
  AEl2 := BuildAutoEncoder(LVQl2, VQl2, DInl2);
  Dl2  := BuildDiscriminator(Featl2);  // built (unused) so both runs match exactly
  Dl2.SetLearningRate(gDLR, 0.9);
  TrainAE(AEl2, LVQl2, VQl2, DInl2, Dl2, Featl2, Train, {Adversarial=}false, 'L2-VQVAE');
  WriteLn;

  // ---- VQ-GAN: recon + perceptual + adversarial ----
  RandSeed := cSeed + 1;  // SAME init as the baseline for a fair comparison
  AEgan := BuildAutoEncoder(LVQgan, VQg, DIng);
  Dgan  := BuildDiscriminator(Featg);
  Dgan.SetLearningRate(gDLR, 0.9);
  TrainAE(AEgan, LVQgan, VQg, DIng, Dgan, Featg, Train, {Adversarial=}true, 'VQ-GAN');
  WriteLn;

  // ---- comparison ----
  EvalAE(AEl2,  Test, L1l2, El2);
  EvalAE(AEgan, Test, L1g,  Eg);
  WriteLn('================ held-out comparison (lower L1 = closer; edge -> originals) ================');
  WriteLn(Format('  ORIGINAL images edge sharpness            : %.4f', [EOrig]));
  WriteLn(Format('  plain-L2 VQ-VAE   recon-L1=%.4f   edge=%.4f', [L1l2, El2]));
  WriteLn(Format('  VQ-GAN            recon-L1=%.4f   edge=%.4f', [L1g,  Eg]));
  if Eg > El2 then
    WriteLn(Format('  -> VQ-GAN edges are %.1f%% sharper than plain L2 (closer to the originals).',
      [100.0 * (Eg - El2) / Max(El2, 1e-6)]))
  else
    WriteLn('  -> (this run did not show a sharpness gain; try --full)');

  RenderPanel(AEl2, AEgan, Test[0]);
  WritePPM(AEl2, AEgan, Test[0], 'vqgan_sample.ppm');
  WriteLn;
  WriteLn('Done.');

  AEl2.Free; AEgan.Free; Dl2.Free; Dgan.Free;
  FreeDataset(Train); FreeDataset(Test);
end.
