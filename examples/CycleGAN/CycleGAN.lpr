program CycleGAN;
(*
CycleGAN: an UNPAIRED image-to-image translation example
(Zhu et al. 2017, "Unpaired Image-to-Image Translation using Cycle-Consistent
Adversarial Networks"). It learns to translate between two image DOMAINS that
have NO aligned pairs, using two generators plus a CYCLE-CONSISTENCY loss to
keep the translation content-preserving. Pure CPU, synthetic data generated in
code, no dataset download. The default SMOKE run finishes well under five
minutes; a --full flag trains longer for a sharper result.

This is the unpaired sibling of examples/Pix2Pix (which is PAIRED supervised:
each grayscale input has a known colorized target). Here the two domains are
sampled INDEPENDENTLY, so no per-sample target exists and an L1-to-target loss
is impossible. CycleGAN's trick is to require that translating A->B->A returns
to the original A (and B->A->B returns to the original B): that round-trip
"cycle" loss is what pins the mapping down without pairs.

THE TASK (red shapes <-> green shapes)
--------------------------------------
Domain A: one or two filled shapes (circles / rectangles) drawn in RED on a
dark-blue background. Domain B: the SAME kind of shapes drawn in GREEN. The two
domains are generated from SEPARATE independent random draws, so a sample a in A
and a sample b in B share NO geometry — they are unpaired. The learnable mapping
is therefore a RECOLORING: G must turn red into green while keeping the shape
silhouette, and F must turn green back into red. RGB is encoded in [-1,1] (so
0 -> -1, 1 -> +1) to match each generator's Tanh output range. Because the
recoloring is content-preserving, the cycle F(G(a)) should reconstruct a almost
exactly once trained — that reconstruction error is our headline metric.

THE GENERATORS (TNNet.AddUNet, x2)
----------------------------------
Two U-Nets, same builder as examples/Pix2Pix and examples/UNetSegmentation:
  G : A -> B  (red  -> green)
  F : B -> A  (green -> red)
Each: Input(grid,grid,3) -> AddUNet(Depth, BaseFeatures, 3) -> Tanh (RGB [-1,1]).

THE DISCRIMINATORS (PatchGAN, x2; composed from existing convs — NO new class)
------------------------------------------------------------------------------
Two small fully-convolutional PatchGAN nets (Isola et al.), one per domain:
  D_B scores whether an image looks like domain B (real green vs G's fake green)
  D_A scores whether an image looks like domain A (real red   vs F's fake red)
Unlike Pix2Pix the discriminator input is the IMAGE ALONE (3 channels), with NO
condition stacked in, because CycleGAN's adversary judges domain membership, not
input-output consistency. We use the LEAST-SQUARES GAN objective (Mao et al.
2017): D regresses real->1 / fake->0 and each generator is pushed toward 1.
EnableErrorCollection on each D's input layer lets us read the gradient of the
adversarial score w.r.t. the generated pixels.

THE LOSS (the crux: cycle + identity bookkeeping)
-------------------------------------------------
Total generator objective (minimised w.r.t. G and F):
  L =  adv(D_B, G(a)) + adv(D_A, F(b))                       (fool both D's)
     + lambda_cyc * ( |F(G(a)) - a|  +  |G(F(b)) - b| )      (round-trip)
     + lambda_id  * ( |G(b)     - b|  +  |F(a)     - a| )      (identity / colour anchor)
The identity term asks G to leave an already-domain-B image unchanged (and F a
domain-A image), which stabilises colours early on.

Backprop through COMPOSED generators (the bit that is new vs Pix2Pix). For the
forward cycle a -> G -> g -> F -> rec:
  1. g   := G.Compute(a);   rec := F.Compute(g)
  2. cycle gradient at F's output:  lambda_cyc * sign(rec - a)
  3. F.Backpropagate(that) — this BOTH updates F's weights on the cycle term AND,
     because F's input layer has EnableErrorCollection, leaves d(cycle)/d(g) in
     F.Layers[0].OutputError. That input gradient is the cycle contribution that
     must flow into G's output.
  4. G's full output error = adversarial grad (from D_B, frozen) + the cycle
     grad returned from F in step 3 + identity grad lambda_id*sign(G(b)-b).
     Seed it and G.GetLastLayer().Backpropagate().
The backward cycle b -> F -> G -> rec is the mirror image, training G on the
cycle term and returning its input gradient into F. We accumulate both
directions per step before the final weight-update backprop of each generator
(SetBatchUpdate keeps the framework from applying partial updates mid-accumulation).

OUTPUT
------
Per-eval mean CYCLE RECONSTRUCTION error |F(G(a))-a| and |G(F(b))-b| (down =
the unpaired mapping is becoming invertible/content-preserving), a TRANSLATION
COLOUR score (fraction of translated shape pixels whose dominant colour matches
the target domain — red->green should go green, green->red should go red), and
the two adversarial losses. Plus an ASCII panel and a PPM strip
(a | G(a) | F(G(a))  /  b | F(b) | G(F(b))) written to cyclegan_sample.ppm.
The console prints whatever the numbers ACTUALLY show.

USAGE
-----
  ./CycleGAN            smoke run (fast, default)
  ./CycleGAN --full     longer training for a sharper result

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
  cGrid      = 16;    // image side (cGrid x cGrid)
  cDepth     = 2;     // U-Net stages (16 / 2^2 = 4 bottleneck cells)
  cBaseFeat  = 16;    // base feature count (doubles per stage)
  cLambdaCyc = 10.0;  // weight of the cycle-consistency term
  cLambdaId  = 5.0;   // weight of the identity term
  cSeed      = 20260615;

  // Domain colours (RGB in [-1,1], i.e. 0->-1, 1->+1).
  cBgR = -0.8; cBgG = -0.8; cBgB = -0.2;   // dark-blue background (both domains)
  // Domain A shapes -> red   (+1,-1,-1)
  // Domain B shapes -> green (-1,+1,-1)

type
  TVolArray = array of TNNetVolume;

var
  gTrainSamples: integer = 64;
  gTestSamples:  integer = 32;
  gEpochs:       integer = 10;
  gGLearnRate:   TNeuralFloat = 0.0006;
  gDLearnRate:   TNeuralFloat = 0.0006;
  gFull:         boolean = false;

// Stamps a filled circle into the binary mask Mask.
procedure StampCircle(Mask: TNNetVolume);
var Cx, Cy, R, X, Y, Dx, Dy: integer;
begin
  R := 2 + Random(4);                       // radius 2..5
  Cx := R + Random(cGrid - 2 * R);
  Cy := R + Random(cGrid - 2 * R);
  for Y := 0 to cGrid - 1 do
    for X := 0 to cGrid - 1 do
    begin
      Dx := X - Cx; Dy := Y - Cy;
      if (Dx * Dx + Dy * Dy) <= (R * R) then Mask[X, Y, 0] := 1;
    end;
end;

// Stamps a filled rectangle into the binary mask Mask.
procedure StampRect(Mask: TNNetVolume);
var X0, Y0, W, H, X, Y: integer;
begin
  W := 3 + Random(7);                        // width 3..9
  H := 3 + Random(7);
  X0 := Random(cGrid - W);
  Y0 := Random(cGrid - H);
  for Y := Y0 to Y0 + H - 1 do
    for X := X0 to X0 + W - 1 do Mask[X, Y, 0] := 1;
end;

// Builds one image whose shape pixels carry colour (Sr,Sg,Sb).
procedure MakeImage(Img: TNNetVolume; Sr, Sg, Sb: TNeuralFloat);
var X, Y, Shapes, S: integer; Mask: TNNetVolume;
begin
  Mask := TNNetVolume.Create(cGrid, cGrid, 1);
  try
    Mask.Fill(0);
    Shapes := 1 + Random(2);                    // 1 or 2 shapes
    for S := 1 to Shapes do
      if Random(2) = 0 then StampCircle(Mask) else StampRect(Mask);
    for Y := 0 to cGrid - 1 do
      for X := 0 to cGrid - 1 do
        if Mask[X, Y, 0] >= 0.5 then
        begin Img[X, Y, 0] := Sr; Img[X, Y, 1] := Sg; Img[X, Y, 2] := Sb; end
        else
        begin Img[X, Y, 0] := cBgR; Img[X, Y, 1] := cBgG; Img[X, Y, 2] := cBgB; end;
  finally
    Mask.Free;
  end;
end;

// Domain A = red shapes; Domain B = green shapes. Drawn independently (unpaired).
procedure BuildDataset(out DomA, DomB: TVolArray; N: integer);
var I: integer;
begin
  SetLength(DomA, N); SetLength(DomB, N);
  for I := 0 to N - 1 do
  begin
    DomA[I] := TNNetVolume.Create(cGrid, cGrid, 3);
    DomB[I] := TNNetVolume.Create(cGrid, cGrid, 3);
    MakeImage(DomA[I],  1, -1, -1);   // red
    MakeImage(DomB[I], -1,  1, -1);   // green
  end;
end;

procedure FreeDataset(var DomA, DomB: TVolArray);
var I: integer;
begin
  for I := 0 to Length(DomA) - 1 do begin DomA[I].Free; DomB[I].Free; end;
  SetLength(DomA, 0); SetLength(DomB, 0);
end;

// ---------------------------------------------------------------------------
// Networks
// ---------------------------------------------------------------------------

// Generator: 3-channel image -> 3-channel image (U-Net + Tanh).
function BuildGenerator(): TNNet;
var Taps: TNeuralIntegerArray;
begin
  Result := TNNet.Create();
  Result.AddLayer(TNNetInput.Create(cGrid, cGrid, 3, 1).EnableErrorCollection);
  Result.AddUNet(cDepth, cBaseFeat, 3, Taps, {UseNorm=}false);
  Result.AddLayer(TNNetHyperbolicTangent.Create());   // RGB in [-1,1]
  Result.InitWeights();
end;

// PatchGAN discriminator: 3-channel image -> small patch-logit grid.
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
// Evaluation
// ---------------------------------------------------------------------------

// Mean |a - b| over all RGB pixels (pixel units in [0,2]).
function MeanL1(A, B: TNNetVolume): TNeuralFloat;
var X, Y, C: integer; S: TNeuralFloat;
begin
  S := 0;
  for Y := 0 to cGrid - 1 do
    for X := 0 to cGrid - 1 do
      for C := 0 to 2 do S := S + Abs(A[X, Y, C] - B[X, Y, C]);
  Result := S / (cGrid * cGrid * 3);
end;

// Cycle reconstruction error and translation-colour score on a held-out set.
// TransOK = fraction of SHAPE pixels (in the source) whose translated colour's
// dominant channel matches the target domain (A->B should be green; B->A red).
procedure Evaluate(G, F: TNNet; const DomA, DomB: TVolArray;
  out CycA, CycB, TransAB, TransBA: TNeuralFloat);
var
  I, X, Y, N, OkAB, OkBA, ShAB, ShBA: integer;
  GOut, FOut, Rec: TNNetVolume;
  SumCycA, SumCycB: TNeuralFloat;
begin
  N := Length(DomA);
  SumCycA := 0; SumCycB := 0;
  OkAB := 0; OkBA := 0; ShAB := 0; ShBA := 0;
  for I := 0 to N - 1 do
  begin
    // Forward cycle A -> G -> F.
    G.Compute(DomA[I]);  GOut := G.GetLastLayer().Output;
    F.Compute(GOut);     Rec  := F.GetLastLayer().Output;
    SumCycA := SumCycA + MeanL1(Rec, DomA[I]);
    for Y := 0 to cGrid - 1 do
      for X := 0 to cGrid - 1 do
        if DomA[I][X, Y, 0] > 0 then  // source shape pixel (red)
        begin
          Inc(ShAB);
          // translated should be green: green channel dominant.
          if (GOut[X, Y, 1] > GOut[X, Y, 0]) and (GOut[X, Y, 1] > GOut[X, Y, 2]) then Inc(OkAB);
        end;

    // Forward cycle B -> F -> G.
    F.Compute(DomB[I]);  FOut := F.GetLastLayer().Output;
    G.Compute(FOut);     Rec  := G.GetLastLayer().Output;
    SumCycB := SumCycB + MeanL1(Rec, DomB[I]);
    for Y := 0 to cGrid - 1 do
      for X := 0 to cGrid - 1 do
        if DomB[I][X, Y, 1] > 0 then  // source shape pixel (green)
        begin
          Inc(ShBA);
          // translated should be red: red channel dominant.
          if (FOut[X, Y, 0] > FOut[X, Y, 1]) and (FOut[X, Y, 0] > FOut[X, Y, 2]) then Inc(OkBA);
        end;
  end;
  CycA := SumCycA / N;
  CycB := SumCycB / N;
  if ShAB > 0 then TransAB := OkAB / ShAB else TransAB := 0;
  if ShBA > 0 then TransBA := OkBA / ShBA else TransBA := 0;
end;

// ASCII colour char: R / G / . (background) by dominant channel.
function ColCh(R, GG, B: TNeuralFloat): char;
begin
  if (R > GG) and (R > B) and (R > 0) then Result := 'R'
  else if (GG > R) and (GG > B) and (GG > 0) then Result := 'G'
  else Result := '.';
end;

procedure RenderSample(G, F: TNNet; const DomA, DomB: TVolArray; Idx: integer);
var X, Y: integer; GOut, FOut, Rec: TNNetVolume;
begin
  WriteLn;
  WriteLn('Held-out sample  (R=red, G=green, .=background)');
  WriteLn('  Row group 1 (domain A):  a            G(a)         F(G(a))=recon');
  WriteLn('  Row group 2 (domain B):  b            F(b)         G(F(b))=recon');
  WriteLn;
  // A row group.
  G.Compute(DomA[Idx]); GOut := TNNetVolume.Create; GOut.Copy(G.GetLastLayer().Output);
  F.Compute(GOut);      Rec  := F.GetLastLayer().Output;
  for Y := 0 to cGrid - 1 do
  begin
    for X := 0 to cGrid - 1 do Write(ColCh(DomA[Idx][X, Y, 0], DomA[Idx][X, Y, 1], DomA[Idx][X, Y, 2]));
    Write('   ');
    for X := 0 to cGrid - 1 do Write(ColCh(GOut[X, Y, 0], GOut[X, Y, 1], GOut[X, Y, 2]));
    Write('   ');
    for X := 0 to cGrid - 1 do Write(ColCh(Rec[X, Y, 0], Rec[X, Y, 1], Rec[X, Y, 2]));
    WriteLn;
  end;
  GOut.Free;
  WriteLn;
  // B row group.
  F.Compute(DomB[Idx]); FOut := TNNetVolume.Create; FOut.Copy(F.GetLastLayer().Output);
  G.Compute(FOut);      Rec  := G.GetLastLayer().Output;
  for Y := 0 to cGrid - 1 do
  begin
    for X := 0 to cGrid - 1 do Write(ColCh(DomB[Idx][X, Y, 0], DomB[Idx][X, Y, 1], DomB[Idx][X, Y, 2]));
    Write('   ');
    for X := 0 to cGrid - 1 do Write(ColCh(FOut[X, Y, 0], FOut[X, Y, 1], FOut[X, Y, 2]));
    Write('   ');
    for X := 0 to cGrid - 1 do Write(ColCh(Rec[X, Y, 0], Rec[X, Y, 1], Rec[X, Y, 2]));
    WriteLn;
  end;
  FOut.Free;
end;

// PPM (P6) strip: two stacked row groups, each [src | translated | recon].
procedure WritePPM(G, F: TNNet; const DomA, DomB: TVolArray; Idx: integer;
  const FileName: string);
const Gap = 1;
var
  Fh: TextFile; Bin: TFileStream; W, H, X, Y, Panel, Row: integer;
  GOutA, RecA, FOutB, RecB: TNNetVolume;
  procedure Emit(rr, gg, bb: byte);
  begin Bin.WriteByte(rr); Bin.WriteByte(gg); Bin.WriteByte(bb); end;
  function ToByte(V: TNeuralFloat): byte;   // [-1,1] -> [0,255]
  begin
    V := (V + 1) * 0.5; if V < 0 then V := 0; if V > 1 then V := 1;
    Result := Round(V * 255);
  end;
  // Returns channel c of panel p in the current row group's volumes.
  function Pix(GroupA: boolean; p, xx, yy, c: integer): TNeuralFloat;
  begin
    if GroupA then case p of
      0: Result := DomA[Idx][xx, yy, c];
      1: Result := GOutA[xx, yy, c];
    else Result := RecA[xx, yy, c]; end
    else case p of
      0: Result := DomB[Idx][xx, yy, c];
      1: Result := FOutB[xx, yy, c];
    else Result := RecB[xx, yy, c]; end;
  end;
begin
  W := cGrid * 3 + Gap * 2; H := cGrid * 2 + Gap;
  G.Compute(DomA[Idx]); GOutA := TNNetVolume.Create; GOutA.Copy(G.GetLastLayer().Output);
  F.Compute(GOutA);     RecA  := TNNetVolume.Create; RecA.Copy(F.GetLastLayer().Output);
  F.Compute(DomB[Idx]); FOutB := TNNetVolume.Create; FOutB.Copy(F.GetLastLayer().Output);
  G.Compute(FOutB);     RecB  := TNNetVolume.Create; RecB.Copy(G.GetLastLayer().Output);
  AssignFile(Fh, FileName); Rewrite(Fh);
  WriteLn(Fh, 'P6'); WriteLn(Fh, W, ' ', H); WriteLn(Fh, 255);
  CloseFile(Fh);
  Bin := TFileStream.Create(FileName, fmOpenReadWrite);
  try
    Bin.Seek(0, soEnd);
    for Y := 0 to H - 1 do
      for X := 0 to W - 1 do
      begin
        if Y = cGrid then begin Emit(0, 0, 0); Continue; end;            // row gap
        if (X = cGrid) or (X = cGrid * 2 + Gap) then begin Emit(0, 0, 0); Continue; end; // col gaps
        if Y < cGrid then Row := 0 else Row := 1;
        if X < cGrid then Panel := 0
        else if X < cGrid * 2 + Gap then Panel := 1
        else Panel := 2;
        Emit(ToByte(Pix(Row = 0, Panel, (X - Panel * (cGrid + Gap)),
                        (Y - Row * (cGrid + Gap)), 0)),
             ToByte(Pix(Row = 0, Panel, (X - Panel * (cGrid + Gap)),
                        (Y - Row * (cGrid + Gap)), 1)),
             ToByte(Pix(Row = 0, Panel, (X - Panel * (cGrid + Gap)),
                        (Y - Row * (cGrid + Gap)), 2)));
      end;
  finally
    Bin.Free;
  end;
  GOutA.Free; RecA.Free; FOutB.Free; RecB.Free;
  WriteLn('Wrote visualization: ', FileName,
    '  (rows: A|G(a)|recon  /  B|F(b)|recon)');
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
    gTrainSamples := 360;
    gTestSamples  := 80;
    gEpochs       := 100;
  end;
end;

// Reads the adversarial input-gradient that discriminator D collected for a
// generated image: freeze D (LR=0), push D(img) toward "real" (1), grab the
// 3-channel gradient w.r.t. D's input pixels. DstErr receives that gradient.
procedure AdvGradFor(D: TNNet; Img, Ones, DstErr: TNNetVolume; DLearnRate: TNeuralFloat);
var X, Y, C: integer; DIn: TNNetVolume;
begin
  D.SetLearningRate(0, 0.5);
  D.Compute(Img);
  D.Backpropagate(Ones);          // want D to call the fake "real"
  D.SetLearningRate(DLearnRate, 0.5);
  DIn := D.Layers[0].OutputError; // 3-ch grad w.r.t. image pixels
  for Y := 0 to cGrid - 1 do
    for X := 0 to cGrid - 1 do
      for C := 0 to 2 do DstErr[X, Y, C] := DIn[X, Y, C];
end;

// Translated image (a fresh copy of Gen(Src)'s output) for reporting.
function CycledFake(Gen: TNNet; Src: TNNetVolume): TNNetVolume;
begin
  Gen.Compute(Src);
  Result := TNNetVolume.Create;
  Result.Copy(Gen.GetLastLayer().Output);
end;

// LSGAN discriminator loss on one (real, fake) pair, for reporting only.
// Fake is OWNED here and freed before returning.
function DLoss(D: TNNet; Real_, Fake: TNNetVolume): TNeuralFloat;
var X, Y: integer; S: TNeuralFloat; PW, PH: integer;
begin
  PW := D.GetLastLayer().Output.SizeX; PH := D.GetLastLayer().Output.SizeY;
  S := 0;
  D.Compute(Real_);
  for Y := 0 to PH - 1 do for X := 0 to PW - 1 do
    S := S + Sqr(D.GetLastLayer().Output[X, Y, 0] - 1);
  D.Compute(Fake);
  for Y := 0 to PH - 1 do for X := 0 to PW - 1 do
    S := S + Sqr(D.GetLastLayer().Output[X, Y, 0] - 0);
  Result := S / (2 * PW * PH);
  Fake.Free;
end;

procedure RunAlgo();
var
  G, F, DB, DA: TNNet;
  DomA, DomB, TDomA, TDomB: TVolArray;
  Epoch, Step, Ia, Ib, X, Y, C: integer;
  GOut, FOut, RecA, RecB, IdG, IdF: TNNetVolume;
  AdvG, AdvF, CycToG, CycToF, GErr, FErr, Ones, Zeros: TNNetVolume;
  CycA, CycB, TransAB, TransBA, LossDA, LossDB: TNeuralFloat;
  T0: TDateTime;
  PatchW, PatchH: integer;
begin
  ParseArgs();
  RandSeed := cSeed;
  WriteLn('CycleGAN: UNPAIRED image translation (two U-Net generators + two PatchGANs)');
  WriteLn('  domain A = RED shapes   <->   domain B = GREEN shapes   (drawn independently)');
  WriteLn(Format('mode=%s  grid=%dx%d  depth=%d  base=%d  train=%d  test=%d  epochs=%d  lCyc=%.1f  lId=%.1f',
    [BoolToStr(gFull, 'full', 'smoke'), cGrid, cGrid, cDepth, cBaseFeat,
     gTrainSamples, gTestSamples, gEpochs, cLambdaCyc, cLambdaId]));
  WriteLn;

  BuildDataset(DomA, DomB, gTrainSamples);
  BuildDataset(TDomA, TDomB, gTestSamples);

  RandSeed := cSeed + 1;
  G  := BuildGenerator();   // A -> B
  F  := BuildGenerator();   // B -> A
  DB := BuildDiscriminator();   // judges domain B
  DA := BuildDiscriminator();   // judges domain A
  G.SetLearningRate(gGLearnRate, 0.5);
  F.SetLearningRate(gGLearnRate, 0.5);
  DA.SetLearningRate(gDLearnRate, 0.5);
  DB.SetLearningRate(gDLearnRate, 0.5);
  // We accumulate several gradient contributions into each generator before its
  // single weight-update backprop, so put G and F into batch-update mode (the
  // framework then applies the summed delta once per ApplyUpdate).
  G.SetBatchUpdate(true);
  F.SetBatchUpdate(true);
  G.GetLastLayer().IncDepartingBranchesCnt();
  F.GetLastLayer().IncDepartingBranchesCnt();

  WriteLn(Format('Generator layers=%d (x2)   Discriminator layers=%d (x2)',
    [G.CountLayers(), DB.CountLayers()]));
  PatchW := DB.GetLastLayer().Output.SizeX;
  PatchH := DB.GetLastLayer().Output.SizeY;
  WriteLn(Format('PatchGAN score grid: %dx%d patches', [PatchW, PatchH]));
  WriteLn;

  // Scratch volumes.
  GErr   := TNNetVolume.Create(cGrid, cGrid, 3);
  FErr   := TNNetVolume.Create(cGrid, cGrid, 3);
  AdvG   := TNNetVolume.Create(cGrid, cGrid, 3);
  AdvF   := TNNetVolume.Create(cGrid, cGrid, 3);
  CycToG := TNNetVolume.Create(cGrid, cGrid, 3);
  CycToF := TNNetVolume.Create(cGrid, cGrid, 3);
  RecA   := TNNetVolume.Create(cGrid, cGrid, 3);
  RecB   := TNNetVolume.Create(cGrid, cGrid, 3);
  Ones   := TNNetVolume.Create(PatchW, PatchH, 1); Ones.Fill(1);
  Zeros  := TNNetVolume.Create(PatchW, PatchH, 1); Zeros.Fill(0);

  try
    WriteLn('Training...');
    T0 := Now();
    Evaluate(G, F, TDomA, TDomB, CycA, CycB, TransAB, TransBA);
    WriteLn(Format('  epoch %3d   cycA=%6.4f cycB=%6.4f  transA->B=%5.3f transB->A=%5.3f  (untrained)',
      [0, CycA, CycB, TransAB, TransBA]));

    for Epoch := 1 to gEpochs do
    begin
      for Step := 0 to gTrainSamples - 1 do
      begin
        Ia := Random(gTrainSamples);   // unpaired: independent indices
        Ib := Random(gTrainSamples);

        // ============ FORWARD PASSES ============
        // Forward cycle A: a -> G -> g -> F -> recA
        G.Compute(DomA[Ia]);  GOut := G.GetLastLayer().Output;
        // keep a private copy of g; F.Compute below overwrites shared buffers.
        // (GOut points into G's buffers which survive until next G.Compute.)
        F.Compute(GOut);      RecA.Copy(F.GetLastLayer().Output);
        // Forward cycle B: b -> F -> f -> G -> recB
        // NB do this AFTER reading recA so we still have g in G's buffers? No:
        // F.Compute above changed F; we now recompute F on b which overwrites
        // F's buffers, then G on f. We will recompute g when needed for D below.
        F.Compute(DomB[Ib]);  FOut := F.GetLastLayer().Output;
        G.Compute(FOut);      RecB.Copy(G.GetLastLayer().Output);

        // ============ TRAIN DISCRIMINATORS ============
        // D_B: real green (DomB[Ib]) -> 1 ; fake green G(a) -> 0
        G.Compute(DomA[Ia]);  GOut := G.GetLastLayer().Output; // recompute g
        DB.Compute(DomB[Ib]); DB.Backpropagate(Ones);
        DB.Compute(GOut);     DB.Backpropagate(Zeros);
        // D_A: real red (DomA[Ia]) -> 1 ; fake red F(b) -> 0
        F.Compute(DomB[Ib]);  FOut := F.GetLastLayer().Output; // recompute f
        DA.Compute(DomA[Ia]); DA.Backpropagate(Ones);
        DA.Compute(FOut);     DA.Backpropagate(Zeros);

        // ============ TRAIN GENERATORS ============
        G.ClearDeltas; F.ClearDeltas;

        // ---- Forward cycle A: train F on |F(G(a))-a|, return grad into G ----
        G.Compute(DomA[Ia]);  GOut := G.GetLastLayer().Output;
        F.Compute(GOut);      // recA already captured; recompute keeps buffers fresh
        // cycle gradient at F's output = lambda_cyc * sign(F(G(a)) - a)
        for Y := 0 to cGrid - 1 do
          for X := 0 to cGrid - 1 do
            for C := 0 to 2 do
              FErr[X, Y, C] := cLambdaCyc *
                Sign(F.GetLastLayer().Output[X, Y, C] - DomA[Ia][X, Y, C]) / (cGrid * cGrid);
        // adversarial on F's OWN output? No: F's output here is recA (domain A),
        // F's adversarial target is judged below on F(b). Here we only carry the
        // cycle term through F and harvest its input gradient for G.
        F.ResetBackpropCallCurrCnt();
        F.GetLastLayer().OutputError.Copy(FErr);
        F.GetLastLayer().Backpropagate();
        // grad of cycle-A w.r.t. g (= G's output) now in F.Layers[0].OutputError
        CycToG.Copy(F.Layers[0].OutputError);

        // adversarial grad for G: push D_B(G(a)) -> real
        AdvGradFor(DB, GOut, Ones, AdvG, gDLearnRate);

        // identity grad for G: G(b) should equal b (domain B in, domain B out)
        G.Compute(DomB[Ib]); IdG := G.GetLastLayer().Output;
        for Y := 0 to cGrid - 1 do
          for X := 0 to cGrid - 1 do
            for C := 0 to 2 do
              GErr[X, Y, C] := cLambdaId *
                Sign(IdG[X, Y, C] - DomB[Ib][X, Y, C]) / (cGrid * cGrid);
        // backprop identity term through G (own gradient, separate forward)
        G.ResetBackpropCallCurrCnt();
        G.GetLastLayer().OutputError.Copy(GErr);
        G.GetLastLayer().Backpropagate();

        // main G backprop: adversarial(G(a)) + cycle-grad-from-F on G(a)
        G.Compute(DomA[Ia]); GOut := G.GetLastLayer().Output;
        for Y := 0 to cGrid - 1 do
          for X := 0 to cGrid - 1 do
            for C := 0 to 2 do
              GErr[X, Y, C] := AdvG[X, Y, C] + CycToG[X, Y, C];
        G.ResetBackpropCallCurrCnt();
        G.GetLastLayer().OutputError.Copy(GErr);
        G.GetLastLayer().Backpropagate();

        // ---- Forward cycle B: train G on |G(F(b))-b|, return grad into F ----
        F.Compute(DomB[Ib]);  FOut := F.GetLastLayer().Output;
        G.Compute(FOut);
        for Y := 0 to cGrid - 1 do
          for X := 0 to cGrid - 1 do
            for C := 0 to 2 do
              GErr[X, Y, C] := cLambdaCyc *
                Sign(G.GetLastLayer().Output[X, Y, C] - DomB[Ib][X, Y, C]) / (cGrid * cGrid);
        G.ResetBackpropCallCurrCnt();
        G.GetLastLayer().OutputError.Copy(GErr);
        G.GetLastLayer().Backpropagate();
        CycToF.Copy(G.Layers[0].OutputError);

        // adversarial grad for F: push D_A(F(b)) -> real
        F.Compute(DomB[Ib]); FOut := F.GetLastLayer().Output;
        AdvGradFor(DA, FOut, Ones, AdvF, gDLearnRate);

        // identity grad for F: F(a) should equal a (domain A in, domain A out)
        F.Compute(DomA[Ia]); IdF := F.GetLastLayer().Output;
        for Y := 0 to cGrid - 1 do
          for X := 0 to cGrid - 1 do
            for C := 0 to 2 do
              FErr[X, Y, C] := cLambdaId *
                Sign(IdF[X, Y, C] - DomA[Ia][X, Y, C]) / (cGrid * cGrid);
        F.ResetBackpropCallCurrCnt();
        F.GetLastLayer().OutputError.Copy(FErr);
        F.GetLastLayer().Backpropagate();

        // main F backprop: adversarial(F(b)) + cycle-grad-from-G on F(b)
        F.Compute(DomB[Ib]); FOut := F.GetLastLayer().Output;
        for Y := 0 to cGrid - 1 do
          for X := 0 to cGrid - 1 do
            for C := 0 to 2 do
              FErr[X, Y, C] := AdvF[X, Y, C] + CycToF[X, Y, C];
        F.ResetBackpropCallCurrCnt();
        F.GetLastLayer().OutputError.Copy(FErr);
        F.GetLastLayer().Backpropagate();

        // apply the accumulated weight updates for both generators.
        G.UpdateWeights();
        F.UpdateWeights();
      end;

      if (Epoch = 1) or (Epoch mod 4 = 0) or (Epoch = gEpochs) then
      begin
        Evaluate(G, F, TDomA, TDomB, CycA, CycB, TransAB, TransBA);
        LossDB := DLoss(DB, TDomB[0], CycledFake(G, TDomA[0]));
        LossDA := DLoss(DA, TDomA[0], CycledFake(F, TDomB[0]));
        WriteLn(Format('  epoch %3d   cycA=%6.4f cycB=%6.4f  transA->B=%5.3f transB->A=%5.3f  lossD_A=%5.3f lossD_B=%5.3f',
          [Epoch, CycA, CycB, TransAB, TransBA, LossDA, LossDB]));
      end;
    end;
    WriteLn(Format('Training wall time: %.1f s', [(Now() - T0) * 24 * 3600]));

    Evaluate(G, F, TDomA, TDomB, CycA, CycB, TransAB, TransBA);
    WriteLn;
    WriteLn(Format('FINAL held-out  cycA=%6.4f cycB=%6.4f  transA->B=%5.3f transB->A=%5.3f',
      [CycA, CycB, TransAB, TransBA]));

    RenderSample(G, F, TDomA, TDomB, 0);
    WritePPM(G, F, TDomA, TDomB, 0, 'cyclegan_sample.ppm');

  finally
    GErr.Free; FErr.Free; AdvG.Free; AdvF.Free; CycToG.Free; CycToF.Free;
    RecA.Free; RecB.Free; Ones.Free;
    G.Free; F.Free; DA.Free; DB.Free;
    FreeDataset(DomA, DomB);
    FreeDataset(TDomA, TDomB);
  end;
end;

begin
  RunAlgo();
end.
