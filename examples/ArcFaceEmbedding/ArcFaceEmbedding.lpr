program ArcFaceEmbedding;
(*
ArcFaceEmbedding: a face/embedding-recognition MICRO-example showing how the
ArcFace angular-margin head (TNNetArcFace) tightens intra-class cosine clusters
of a learned embedding. No external dataset, pure CPU, single-threaded, tiny
dims, finishes well under a minute.

WHAT IT SHOWS
-------------
A synthetic multi-class "identity" dataset (cNumClasses 2D Gaussian blobs) is
mapped to a small D-dimensional embedding by a shared MLP. The embedding feeds
a TNNetArcFace head that classifies via the angular margin softmax
  logit_k = s * cos(theta_k),   with cos(theta_y) replaced by cos(theta_y + m)
                                for the TRUE class y.
We sweep the angular margin m in {0, 0.3, 0.5}. m=0 is a plain normalized
(cosine) softmax classifier; larger m forces the true-class angle to be
SMALLER (more confident) before the loss is satisfied, which pulls same-class
embeddings into a tighter cone. After training each margin we print:
  - mean SAME-class  cosine similarity (should RISE as m grows -> tighter)
  - mean DIFF-class   cosine similarity (should DROP as m grows -> separated)
  - separation = same - diff (should GROW with m)

HOW THE ARCFACE HEAD IS WIRED (label passthrough via depth concat)
------------------------------------------------------------------
TNNetArcFace consumes an input depth laid out as  embedding | label , i.e. the
last depth channel is the integer class label and channels 0..D-1 are the
embedding. forward is an identity passthrough; the loss/gradient is produced in
Backpropagate from that layout. To feed it inside ONE trainable network we split
the raw input into a feature branch and a label-passthrough branch and re-concat
along depth just before the head:

  Input(1, 1, 3)                          <- [x, y, label]
   |-- SplitChannels([0,1]) -> PointwiseConvReLU x2 -> PointwiseConvLinear(D)
   |        (the shared embedding MLP; pointwise = per spatial cell)
   |-- SplitChannels([2])               <- the label channel, untouched
   DeepConcat([embedding, label]) -> (1,1,D+1)
   TNNetArcFace(K, m, s)

Gradients from the margin flow back through the embedding branch only (the label
branch carries no gradient), so the margin genuinely shapes the embedding.

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
*)

{$mode objfpc}{$H+}

uses {$IFDEF UNIX} {$IFDEF UseCThreads}
  cthreads, {$ENDIF} {$ENDIF}
  Classes, SysUtils, Math,
  neuralnetwork,
  neuralvolume;

const
  cNumClasses    = 4;
  cEmbedDim      = 3;     // small embedding
  cSamplesPerCls = 60;    // synthetic samples per class
  cEpochs        = 50;
  cLearningRate  = 0.05;
  cScale         = 8.0;   // moderate ArcFace scale (well-conditioned softmax)
  cSeed          = 12345;
  cSigma         = 0.55;  // blob spread (enough overlap that the margin matters)

  // Margins to sweep. 0 = plain cosine softmax; larger = stronger angular margin.
  cMargins: array[0..2] of TNeuralFloat = (0.0, 0.3, 0.5);

  // Four 2D Gaussian blob centers ("identities"), well separated.
  cCenters: array[0..cNumClasses - 1, 0..1] of TNeuralFloat =
    ((-1.5, -1.5),
     ( 1.5, -1.5),
     ( 1.5,  1.5),
     (-1.5,  1.5));

type
  TSample = record
    X, Y: TNeuralFloat;
    Cls:  integer;
  end;
  TSampleArray = array of TSample;

// Why: Box-Muller gives N(0,1) samples without an extra dependency.
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

// Builds the synthetic dataset: cSamplesPerCls 2D Gaussian points per class.
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

// Builds the embedding MLP + ArcFace head. Returns the network; EmbeddingLayer
// is set to the embedding branch end so callers can read the D-dim embedding,
// and ArcLayer is the head so callers can compute the reporting loss.
function BuildNet(Margin: TNeuralFloat; out EmbeddingLayer: TNNetLayer;
  out ArcLayer: TNNetArcFace): TNNet;
var
  InputLayer, FeatBranch, LabelBranch: TNNetLayer;
begin
  Result := TNNet.Create();
  InputLayer := Result.AddLayer(TNNetInput.Create(1, 1, 3)); // [x, y, label]

  // Feature branch: shared embedding MLP over the 2 coordinate channels.
  Result.AddLayerAfter(TNNetSplitChannels.Create([0, 1]), InputLayer);
  Result.AddLayer(TNNetPointwiseConvReLU.Create(16));
  Result.AddLayer(TNNetPointwiseConvReLU.Create(16));
  FeatBranch := Result.AddLayer(TNNetPointwiseConvLinear.Create(cEmbedDim));
  EmbeddingLayer := FeatBranch;

  // Label branch: the label channel passes through untouched.
  LabelBranch := Result.AddLayerAfter(TNNetSplitChannels.Create([2]), InputLayer);

  // Concat embedding | label  -> (1,1,cEmbedDim+1) for the ArcFace head.
  Result.AddLayer(TNNetDeepConcat.Create([FeatBranch, LabelBranch]));
  ArcLayer := TNNetArcFace.Create(cNumClasses, Margin, cScale);
  Result.AddLayer(ArcLayer);
  Result.InitWeights();
end;

// ArcFace scalar loss = -log softmax(s*cos')_y, mirroring the layer forward.
// EmbVec holds the D embedding channels; the head supplies class weights/margin.
function ArcFaceLoss(ArcLayer: TNNetArcFace; const EmbVec: array of TNeuralFloat;
  Margin: TNeuralFloat; Cls: integer): TNeuralFloat;
var
  J, K: integer;
  XN, WN, Dot, Cs, St, CsM, MaxL, SumExp, PY, CosM, SinM: TNeuralFloat;
  Logits: array of TNeuralFloat;
begin
  CosM := Cos(Margin);
  SinM := Sin(Margin);
  SetLength(Logits, cNumClasses);
  XN := 0;
  for J := 0 to cEmbedDim - 1 do XN := XN + EmbVec[J] * EmbVec[J];
  XN := Sqrt(XN);
  if XN < 1e-12 then XN := 1e-12;
  for K := 0 to cNumClasses - 1 do
  begin
    WN := 0; Dot := 0;
    for J := 0 to cEmbedDim - 1 do
    begin
      WN := WN + ArcLayer.Neurons[K].Weights.Raw[J] * ArcLayer.Neurons[K].Weights.Raw[J];
      Dot := Dot + EmbVec[J] * ArcLayer.Neurons[K].Weights.Raw[J];
    end;
    WN := Sqrt(WN);
    if WN < 1e-12 then WN := 1e-12;
    Cs := Dot / (XN * WN);
    if Cs > 1.0 then Cs := 1.0;
    if Cs < -1.0 then Cs := -1.0;
    if K = Cls then
    begin
      St := Sqrt(1.0 - Cs * Cs);
      CsM := Cs * CosM - St * SinM;
      Logits[K] := cScale * CsM;
    end
    else
      Logits[K] := cScale * Cs;
  end;
  MaxL := Logits[0];
  for K := 1 to cNumClasses - 1 do
    if Logits[K] > MaxL then MaxL := Logits[K];
  SumExp := 0;
  for K := 0 to cNumClasses - 1 do
    SumExp := SumExp + Exp(Logits[K] - MaxL);
  PY := Exp(Logits[Cls] - MaxL) / SumExp;
  if PY < 1e-30 then PY := 1e-30;
  Result := -Ln(PY);
end;

// Fills the input volume [x, y, label] for one sample.
procedure FillInput(V: TNNetVolume; const S: TSample);
begin
  V[0, 0, 0] := S.X;
  V[0, 0, 1] := S.Y;
  V[0, 0, 2] := S.Cls; // label channel (read by ArcFace as the true class)
end;

// One training epoch over a shuffled pass of the dataset. Returns mean loss.
function TrainEpoch(NN: TNNet; EmbeddingLayer: TNNetLayer;
  ArcLayer: TNNetArcFace; Margin: TNeuralFloat;
  const Data: TSampleArray; Dummy: TNNetVolume): TNeuralFloat;
var
  Step, NumSamples, Idx, D: integer;
  Input: TNNetVolume;
  Order: array of integer;
  Tmp: integer;
  EmbVec: array[0..cEmbedDim - 1] of TNeuralFloat;
  TotalLoss: TNeuralFloat;
begin
  NumSamples := Length(Data);
  Input := TNNetVolume.Create(1, 1, 3);
  SetLength(Order, NumSamples);
  for Idx := 0 to NumSamples - 1 do Order[Idx] := Idx;
  // Fisher-Yates shuffle for a fresh sample order each epoch.
  for Idx := NumSamples - 1 downto 1 do
  begin
    Step := Random(Idx + 1);
    Tmp := Order[Idx]; Order[Idx] := Order[Step]; Order[Step] := Tmp;
  end;
  TotalLoss := 0;
  try
    for Step := 0 to NumSamples - 1 do
    begin
      FillInput(Input, Data[Order[Step]]);
      NN.Compute(Input);
      // Read the embedding produced THIS forward pass to report the loss.
      for D := 0 to cEmbedDim - 1 do
        EmbVec[D] := EmbeddingLayer.Output.FData[D];
      TotalLoss := TotalLoss +
        ArcFaceLoss(ArcLayer, EmbVec, Margin, Data[Order[Step]].Cls);
      NN.Backpropagate(Dummy);
    end;
  finally
    Input.Free;
  end;
  Result := TotalLoss / NumSamples;
end;

// Computes and L2-normalizes the embedding of one sample (read from the
// embedding branch end after a forward pass).
procedure EmbedSample(NN: TNNet; EmbeddingLayer: TNNetLayer; const S: TSample;
  Input: TNNetVolume; out E0, E1, E2: TNeuralFloat);
var
  Norm: TNeuralFloat;
begin
  FillInput(Input, S);
  NN.Compute(Input);
  E0 := EmbeddingLayer.Output.FData[0];
  E1 := EmbeddingLayer.Output.FData[1];
  E2 := EmbeddingLayer.Output.FData[2];
  Norm := Sqrt(E0 * E0 + E1 * E1 + E2 * E2);
  if Norm < 1e-12 then Norm := 1e-12;
  E0 := E0 / Norm; E1 := E1 / Norm; E2 := E2 / Norm;
end;

// Trains a net at the given margin and reports same/diff-class cosine stats.
procedure RunMargin(Margin: TNeuralFloat; const Data: TSampleArray);
var
  NN: TNNet;
  EmbeddingLayer: TNNetLayer;
  ArcLayer: TNNetArcFace;
  Dummy, Input: TNNetVolume;
  Embeds: array of array of TNeuralFloat; // [sample][0..2]
  Epoch, I, J, N: integer;
  MeanLoss, SameSum, DiffSum, Dot: TNeuralFloat;
  SameCnt, DiffCnt: integer;
begin
  RandSeed := cSeed; // identical init/training order per margin for a fair sweep
  N := Length(Data);
  NN := BuildNet(Margin, EmbeddingLayer, ArcLayer);
  Dummy := TNNetVolume.Create(1, 1, cEmbedDim + 1); // ArcFace ignores the target
  Dummy.Fill(0);
  Input := TNNetVolume.Create(1, 1, 3);
  SetLength(Embeds, N, cEmbedDim);
  NN.SetLearningRate(cLearningRate, 0.9);
  try
    MeanLoss := 0;
    for Epoch := 1 to cEpochs do
      MeanLoss := TrainEpoch(NN, EmbeddingLayer, ArcLayer, Margin, Data, Dummy);

    // Collect L2-normalized embeddings.
    for I := 0 to N - 1 do
      EmbedSample(NN, EmbeddingLayer, Data[I], Input,
        Embeds[I][0], Embeds[I][1], Embeds[I][2]);

    // Mean cosine similarity over all ordered sample pairs (unit vectors,
    // so cosine = dot product), split by same vs different class.
    SameSum := 0; DiffSum := 0; SameCnt := 0; DiffCnt := 0;
    for I := 0 to N - 1 do
      for J := I + 1 to N - 1 do
      begin
        Dot := Embeds[I][0] * Embeds[J][0] +
               Embeds[I][1] * Embeds[J][1] +
               Embeds[I][2] * Embeds[J][2];
        if Data[I].Cls = Data[J].Cls then
        begin
          SameSum := SameSum + Dot; Inc(SameCnt);
        end
        else
        begin
          DiffSum := DiffSum + Dot; Inc(DiffCnt);
        end;
      end;
    if SameCnt = 0 then SameCnt := 1;
    if DiffCnt = 0 then DiffCnt := 1;

    WriteLn(Format('  %5.2f   %9.4f   %9.4f   %9.4f   %9.5f',
      [Margin,
       SameSum / SameCnt,
       DiffSum / DiffCnt,
       (SameSum / SameCnt) - (DiffSum / DiffCnt),
       MeanLoss]));
  finally
    Input.Free;
    Dummy.Free;
    NN.Free;
  end;
end;

procedure RunAlgo();
var
  Data: TSampleArray;
  M: integer;
begin
  RandSeed := cSeed;
  WriteLn('ArcFaceEmbedding: angular-margin embedding on synthetic 2D blobs');
  WriteLn('Classes: ', cNumClasses, '  embed_dim: ', cEmbedDim,
    '  samples/class: ', cSamplesPerCls, '  scale: ', cScale:0:1,
    '  epochs: ', cEpochs);
  WriteLn;
  BuildDataset(Data);

  WriteLn('Sweeping the ArcFace angular margin m:');
  WriteLn('(same-class cosine should RISE and separation should GROW with m)');
  WriteLn;
  WriteLn('  margin    same     diff   separation   mean_loss');
  WriteLn('  ------   ------   ------   ----------   ---------');
  for M := Low(cMargins) to High(cMargins) do
    RunMargin(cMargins[M], Data);
  WriteLn;
  WriteLn('Done. A larger margin tightens intra-class cosine clusters.');
end;

var
  // Stops Lazarus errors
  Application: record Title: string; end;

begin
  Application.Title := 'ArcFaceEmbedding Example';
  // Direct Compute/Backpropagate is inherently single-threaded and CPU-only;
  // no thread pool is created.
  RunAlgo();
end.
