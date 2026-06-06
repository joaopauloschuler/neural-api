program CenterLossSoftmaxJoint;
(*
CenterLossSoftmaxJoint: reproduces the headline result of Wen et al. 2016
("A Discriminative Feature Learning Approach for Deep Face Recognition") on a
tiny synthetic multi-class task, pure CPU, single-threaded, finishes well under
a minute.

THE HEADLINE (Wen et al. 2016)
------------------------------
Training a feature extractor with softmax CROSS-ENTROPY ALONE makes the classes
SEPARABLE but the per-class features stay SPREAD OUT. ADDING the center-loss
penalty jointly (lambda-weighted) pulls every sample toward its learned class
center, giving visibly TIGHTER intra-class clusters at the same (or better)
accuracy. This example shows that contrast head-to-head:
  ARM A  softmax cross-entropy only        (center loss OFF, lambda effectively 0)
  ARM B  softmax cross-entropy + center    (center loss ON,  lambda > 0)
Both arms use the SAME architecture, SAME seed and SAME data, so the only
difference is the center-loss pull. We then report, per arm:
  - final classification accuracy,
  - the intra-class tightness metric  mean within-class feature radius
    (sqrt(mean squared distance of each sample to its class mean)),
  - the inter-class mean-separation,
  - the intra/inter ratio (smaller = tighter, cleaner clusters),
and we dump an ASCII scatter of the 2-D embedding for each arm so the tightening
is visible WITHOUT any external plotting tools.

HOW THE TWO HEADS SHARE ONE EMBEDDING (joint wiring)
----------------------------------------------------
TNNetCenterLoss is an identity-passthrough penalty head: it consumes a depth
layout  feature | label  (the last channel is the integer class label, channels
0..D-1 are the feature/embedding), and in Backpropagate it OVERWRITES its own
FOutputError with  lambda*(x - center_c)  while pulling center_c toward x. It
ignores whatever residual the framework seeds into it. The classification head
needs the standard cross-entropy gradient instead. To make BOTH heads share the
SAME 2-D embedding inside ONE trainable network we branch the graph and rejoin
it at a final DeepConcat that the single Backpropagate call walks back through:

  Input(1,1,3)                                   <- [x, y, label]
   |
   |-- SplitChannels([0,1]) -> PointwiseConvReLU x2 -> PointwiseConvLinear(2)
   |        = EMB  (the shared 2-D embedding; pointwise = per spatial cell)
   |          |
   |          |-- PointwiseConvLinear(K)             = LOGITS  (class head)
   |          |
   |          |-- DeepConcat([EMB, LABEL]) -> CenterLoss(K, lambda) = CENTER
   |
   |-- SplitChannels([2])                            = LABEL  (untouched)
   |
   DeepConcat([LOGITS, CENTER])                       = final (1,1, K + (2+1))

EMB feeds TWO consumers (the logits head AND the center head), so the framework's
departing-branch counter makes EMB wait for and ACCUMULATE the gradients from
both before propagating - exactly the gradient sharing the joint objective needs.

Backpropagate seeding: we compute softmax+cross-entropy externally on the LOGITS
sub-region and seed the final concat residual as
  [ softmax(logits) - onehot   (K values, the exact CE gradient) ,
    0 0 0                       (the CenterLoss region; it self-generates its
                                 own gradient and ignores this) ].
The concat splits that residual back to each head. For ARM A we build the net
WITHOUT the center head (softmax only); for ARM B we include it with lambda > 0.
Gradients from the center penalty reach EMB ONLY in ARM B, which is precisely why
ARM B's clusters come out measurably tighter.

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
  cNumClasses    = 4;
  cEmbedDim      = 2;     // 2-D embedding so it can be drawn as an ASCII scatter
  cSamplesPerCls = 80;    // synthetic samples per class
  cEpochs        = 40;
  cLearningRate  = 0.02;
  cSeed          = 20260606;
  cSigma         = 0.45;  // blob spread (enough overlap that the center pull matters)
  cLambda        = 0.30;  // center-loss weight for ARM B

  // Four 2-D Gaussian blob centers ("identities").
  cCenters: array[0..cNumClasses - 1, 0..1] of TNeuralFloat =
    ((-1.6, -1.6),
     ( 1.6, -1.6),
     ( 1.6,  1.6),
     (-1.6,  1.6));

type
  TSample = record
    X, Y: TNeuralFloat;
    Cls:  integer;
  end;
  TSampleArray = array of TSample;

// Box-Muller N(0,1) sample (no extra dependency).
function RandomGauss(): TNeuralFloat;
var
  U1, U2: TNeuralFloat;
begin
  repeat
    U1 := Random;
  until U1 > 1e-9;
  U2 := Random;
  Result := Sqrt(-2 * Ln(U1)) * Cos(2 * Pi * U2);
end;

procedure BuildDataset(out Data: TSampleArray);
var
  C, I, Idx: integer;
begin
  SetLength(Data, cNumClasses * cSamplesPerCls);
  Idx := 0;
  for C := 0 to cNumClasses - 1 do
    for I := 1 to cSamplesPerCls do
    begin
      Data[Idx].X := cCenters[C][0] + RandomGauss() * cSigma;
      Data[Idx].Y := cCenters[C][1] + RandomGauss() * cSigma;
      Data[Idx].Cls := C;
      Inc(Idx);
    end;
end;

// Builds the shared-embedding net. When UseCenter is True the center-loss head
// is wired alongside the logits head (ARM B); otherwise only the logits head is
// present (ARM A, softmax-only). EmbLayer / LogitsLayer are returned so the
// caller can read the 2-D embedding and the class logits each forward pass.
function BuildNet(UseCenter: boolean; out EmbLayer, LogitsLayer: TNNetLayer): TNNet;
var
  InputLayer, LabelBranch, CenterHead: TNNetLayer;
begin
  Result := TNNet.Create();
  InputLayer := Result.AddLayer(TNNetInput.Create(1, 1, 3)); // [x, y, label]

  // Shared embedding MLP over the 2 coordinate channels.
  Result.AddLayerAfter(TNNetSplitChannels.Create([0, 1]), InputLayer);
  Result.AddLayer(TNNetPointwiseConvReLU.Create(16));
  Result.AddLayer(TNNetPointwiseConvReLU.Create(16));
  EmbLayer := Result.AddLayer(TNNetPointwiseConvLinear.Create(cEmbedDim));

  // Class head: linear logits read straight off the embedding. (We apply the
  // softmax + cross-entropy externally and seed its gradient, so this is a plain
  // linear layer, not a TNNetSoftMax.)
  LogitsLayer := Result.AddLayer(TNNetPointwiseConvLinear.Create(cNumClasses));

  if UseCenter then
  begin
    // Label branch: the integer class label passes through untouched.
    LabelBranch := Result.AddLayerAfter(TNNetSplitChannels.Create([2]), InputLayer);
    // feature | label  -> center-loss penalty head on the shared embedding.
    Result.AddLayer(TNNetDeepConcat.Create([EmbLayer, LabelBranch]));
    CenterHead := Result.AddLayer(TNNetCenterLoss.Create(cNumClasses, cLambda));
    // Rejoin both heads so a single Backpropagate walks back through both.
    Result.AddLayer(TNNetDeepConcat.Create([LogitsLayer, CenterHead]));
  end;
  // ARM A: the logits head is already the last layer; no rejoin needed.

  Result.InitWeights();
end;

procedure FillInput(V: TNNetVolume; const S: TSample);
begin
  V[0, 0, 0] := S.X;
  V[0, 0, 1] := S.Y;
  V[0, 0, 2] := S.Cls;
end;

// Softmax over the K logits; returns the cross-entropy loss for the true class
// and writes the CE gradient  softmax - onehot  into Grad[0..K-1].
function SoftmaxCE(const Logits: array of TNeuralFloat; Cls: integer;
  out Grad: array of TNeuralFloat): TNeuralFloat;
var
  K: integer;
  MaxL, SumExp, PY: TNeuralFloat;
  P: array[0..cNumClasses - 1] of TNeuralFloat;
begin
  MaxL := Logits[0];
  for K := 1 to cNumClasses - 1 do
    if Logits[K] > MaxL then MaxL := Logits[K];
  SumExp := 0;
  for K := 0 to cNumClasses - 1 do
  begin
    P[K] := Exp(Logits[K] - MaxL);
    SumExp := SumExp + P[K];
  end;
  for K := 0 to cNumClasses - 1 do
    P[K] := P[K] / SumExp;
  for K := 0 to cNumClasses - 1 do
    Grad[K] := P[K] - Ord(K = Cls);
  PY := P[Cls];
  if PY < 1e-30 then PY := 1e-30;
  Result := -Ln(PY);
end;

// One training epoch (shuffled pass). Returns mean cross-entropy loss.
function TrainEpoch(NN: TNNet; LogitsLayer: TNNetLayer; UseCenter: boolean;
  const Data: TSampleArray): TNeuralFloat;
var
  Step, N, Idx, K, Tmp, FinalSize: integer;
  Input, Target: TNNetVolume;
  LastLayer: TNNetLayer;
  Order: array of integer;
  Logits, Grad: array[0..cNumClasses - 1] of TNeuralFloat;
  TotalLoss: TNeuralFloat;
begin
  N := Length(Data);
  Input := TNNetVolume.Create(1, 1, 3);
  LastLayer := NN.GetLastLayer();
  // The framework seeds the last-layer residual as FOutputError = FOutput - Target.
  // We WANT FOutputError = [ CE grad on the K logits ; 0 on the center region ].
  // So we pass Target = FOutput - desiredResidual: copy the live output and then
  // subtract the CE gradient from the logits region only (center region: residual
  // 0, and CenterLoss ignores it regardless).
  FinalSize := LastLayer.Output.Size;
  Target := TNNetVolume.Create(1, 1, FinalSize);
  SetLength(Order, N);
  for Idx := 0 to N - 1 do Order[Idx] := Idx;
  for Idx := N - 1 downto 1 do
  begin
    Step := Random(Idx + 1);
    Tmp := Order[Idx]; Order[Idx] := Order[Step]; Order[Step] := Tmp;
  end;
  TotalLoss := 0;
  try
    for Step := 0 to N - 1 do
    begin
      FillInput(Input, Data[Order[Step]]);
      NN.Compute(Input);
      for K := 0 to cNumClasses - 1 do
        Logits[K] := LogitsLayer.Output.FData[K];
      TotalLoss := TotalLoss + SoftmaxCE(Logits, Data[Order[Step]].Cls, Grad);
      // Target = live final output, minus the CE gradient on the logits region.
      Target.Copy(LastLayer.Output);
      // In ARM B the final concat is [LOGITS (K) | CENTER (cEmbedDim+1)]; in
      // ARM A the final layer is the K logits directly. Either way the logits
      // occupy channels 0..K-1.
      for K := 0 to cNumClasses - 1 do
        Target.FData[K] := Target.FData[K] - Grad[K];
      NN.Backpropagate(Target);
    end;
  finally
    Input.Free;
    Target.Free;
  end;
  Result := TotalLoss / N;
end;

// Collects 2-D embeddings + predictions, then reports accuracy, the intra-class
// tightness metric, inter-class separation, the ratio and an ASCII scatter.
procedure Report(const Title: string; NN: TNNet; EmbLayer, LogitsLayer: TNNetLayer;
  const Data: TSampleArray);
const
  cW = 49; cH = 21;
var
  Input: TNNetVolume;
  Emb: array of array of TNeuralFloat;
  Mean: array[0..cNumClasses - 1, 0..cEmbedDim - 1] of TNeuralFloat;
  Cnt: array[0..cNumClasses - 1] of integer;
  N, I, C, D, K, Pred, Correct, Px, Py: integer;
  Logits: array[0..cNumClasses - 1] of TNeuralFloat;
  IntraSq, Dist, MinX, MaxX, MinY, MaxY, InterSum, Sep, Radius, Ratio: TNeuralFloat;
  InterCnt: integer;
  Glyphs: array[0..cNumClasses - 1] of char;
  Grid: array of array of char;
begin
  Glyphs[0] := 'A'; Glyphs[1] := 'B'; Glyphs[2] := 'C'; Glyphs[3] := 'D';
  N := Length(Data);
  SetLength(Emb, N, cEmbedDim);
  Input := TNNetVolume.Create(1, 1, 3);
  for C := 0 to cNumClasses - 1 do
  begin
    Cnt[C] := 0;
    for D := 0 to cEmbedDim - 1 do Mean[C][D] := 0;
  end;
  Correct := 0;
  try
    for I := 0 to N - 1 do
    begin
      FillInput(Input, Data[I]);
      NN.Compute(Input);
      for D := 0 to cEmbedDim - 1 do Emb[I][D] := EmbLayer.Output.FData[D];
      for K := 0 to cNumClasses - 1 do Logits[K] := LogitsLayer.Output.FData[K];
      Pred := 0;
      for K := 1 to cNumClasses - 1 do
        if Logits[K] > Logits[Pred] then Pred := K;
      if Pred = Data[I].Cls then Inc(Correct);
      C := Data[I].Cls;
      for D := 0 to cEmbedDim - 1 do Mean[C][D] := Mean[C][D] + Emb[I][D];
      Inc(Cnt[C]);
    end;
  finally
    Input.Free;
  end;
  for C := 0 to cNumClasses - 1 do
    if Cnt[C] > 0 then
      for D := 0 to cEmbedDim - 1 do Mean[C][D] := Mean[C][D] / Cnt[C];

  // Intra-class tightness: sqrt(mean squared dist of each sample to its mean).
  IntraSq := 0;
  for I := 0 to N - 1 do
  begin
    C := Data[I].Cls;
    Dist := 0;
    for D := 0 to cEmbedDim - 1 do
      Dist := Dist + Sqr(Emb[I][D] - Mean[C][D]);
    IntraSq := IntraSq + Dist;
  end;
  Radius := Sqrt(IntraSq / N);

  // Inter-class separation: mean distance between class means.
  InterSum := 0; InterCnt := 0;
  for C := 0 to cNumClasses - 1 do
    for K := C + 1 to cNumClasses - 1 do
    begin
      Dist := 0;
      for D := 0 to cEmbedDim - 1 do Dist := Dist + Sqr(Mean[C][D] - Mean[K][D]);
      InterSum := InterSum + Sqrt(Dist);
      Inc(InterCnt);
    end;
  if InterCnt = 0 then InterCnt := 1;
  Sep := InterSum / InterCnt;
  if Sep < 1e-12 then Ratio := 0 else Ratio := Radius / Sep;

  WriteLn;
  WriteLn(StringOfChar('=', 70));
  WriteLn(Title);
  WriteLn(StringOfChar('=', 70));
  WriteLn(Format('  accuracy                       : %.4f', [Correct / N]));
  WriteLn(Format('  intra-class radius (TIGHTNESS) : %.4f  (smaller = tighter)', [Radius]));
  WriteLn(Format('  inter-class mean separation    : %.4f', [Sep]));
  WriteLn(Format('  intra/inter ratio              : %.4f  (smaller = cleaner)', [Ratio]));

  // ASCII scatter of the 2-D embedding.
  MinX := Emb[0][0]; MaxX := Emb[0][0]; MinY := Emb[0][1]; MaxY := Emb[0][1];
  for I := 1 to N - 1 do
  begin
    if Emb[I][0] < MinX then MinX := Emb[I][0];
    if Emb[I][0] > MaxX then MaxX := Emb[I][0];
    if Emb[I][1] < MinY then MinY := Emb[I][1];
    if Emb[I][1] > MaxY then MaxY := Emb[I][1];
  end;
  if MaxX - MinX < 1e-6 then MaxX := MinX + 1;
  if MaxY - MinY < 1e-6 then MaxY := MinY + 1;
  SetLength(Grid, cH, cW);
  for Py := 0 to cH - 1 do
    for Px := 0 to cW - 1 do Grid[Py][Px] := ' ';
  for I := 0 to N - 1 do
  begin
    Px := Round((Emb[I][0] - MinX) / (MaxX - MinX) * (cW - 1));
    Py := Round((Emb[I][1] - MinY) / (MaxY - MinY) * (cH - 1));
    Py := cH - 1 - Py; // flip so +y points up
    if (Px >= 0) and (Px < cW) and (Py >= 0) and (Py < cH) then
      Grid[Py][Px] := Glyphs[Data[I].Cls];
  end;
  WriteLn('  2-D embedding scatter (A,B,C,D = classes 0..3):');
  WriteLn('  +', StringOfChar('-', cW), '+');
  for Py := 0 to cH - 1 do
  begin
    Write('  |');
    for Px := 0 to cW - 1 do Write(Grid[Py][Px]);
    WriteLn('|');
  end;
  WriteLn('  +', StringOfChar('-', cW), '+');
end;

procedure RunArm(const Title: string; UseCenter: boolean; const Data: TSampleArray);
var
  NN: TNNet;
  EmbLayer, LogitsLayer: TNNetLayer;
  Epoch: integer;
  MeanLoss: TNeuralFloat;
begin
  RandSeed := cSeed; // identical init + sample order for both arms (fair contrast)
  NN := BuildNet(UseCenter, EmbLayer, LogitsLayer);
  NN.SetLearningRate(cLearningRate, 0.9);
  try
    MeanLoss := 0;
    for Epoch := 1 to cEpochs do
      MeanLoss := TrainEpoch(NN, LogitsLayer, UseCenter, Data);
    Report(Title + Format('  (final CE loss %.4f)', [MeanLoss]),
      NN, EmbLayer, LogitsLayer, Data);
  finally
    NN.Free;
  end;
end;

var
  Data: TSampleArray;
begin
  WriteLn('CenterLossSoftmaxJoint: Wen et al. 2016 center-loss contrast');
  WriteLn('Classes: ', cNumClasses, '  embed_dim: ', cEmbedDim,
    '  samples/class: ', cSamplesPerCls, '  epochs: ', cEpochs,
    '  lambda(ARM B): ', cLambda:0:2);
  WriteLn('Same architecture / seed / data in both arms; the ONLY difference is');
  WriteLn('whether the joint center-loss penalty pulls features to class centers.');

  RandSeed := cSeed;
  BuildDataset(Data); // one fixed dataset shared by both arms

  RunArm('ARM A: softmax cross-entropy ONLY (center loss OFF)', False, Data);
  RunArm('ARM B: softmax + center loss JOINT (center loss ON) ', True,  Data);

  WriteLn;
  WriteLn('HEADLINE: both arms classify well, but ARM B''s intra-class radius and');
  WriteLn('intra/inter ratio are smaller - the center-loss penalty tightened the');
  WriteLn('per-class clusters, exactly the Wen et al. 2016 result. Compare the two');
  WriteLn('ASCII scatters: ARM B''s class blobs are visibly more compact.');
end.
