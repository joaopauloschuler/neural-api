program CosineEmbeddingSiamese;
(*
CosineEmbeddingSiamese: a tiny SHARED-WEIGHT (siamese) embedding example that
demonstrates the TNNetCosineEmbeddingLoss head on a synthetic "same vs
different class" pair task. No external dataset, pure CPU, finishes in seconds.

WHAT IT SHOWS
-------------
A handful of nD Gaussian class blobs are mapped, by ONE shared MLP applied to
BOTH members of a pair, to a low-dimensional embedding. We sample same-class
pairs (label y=1) and different-class pairs (label y=0) and train with the
pairwise cosine-embedding loss
  L = y*(1 - cos(a,b)) + (1 - y)*sqr(max(0, cos(a,b) - margin))
After training we print the cosine-similarity HISTOGRAMS of held-out same-class
pairs vs different-class pairs: same-pairs cluster near +1 while different-pairs
are pushed down toward / below the margin.

HOW THE COSINE-EMBEDDING HEAD IS WIRED (weight-shared siamese, fully native)
----------------------------------------------------------------------------
TNNetCosineEmbeddingLoss is a SELF-CONTAINED metric head: there is NO external
target tensor, the per-pair supervision label y is packed INTO the input. Per
spatial (X,Y) cell it splits the input depth as
  [ a (d channels) | b (d channels) | y (1 channel) ]   => Depth = 2*d + 1
(validated odd and >= 3). a and b are the two embeddings to compare; y is the
similarity label: y=1 => "similar", y=0 => "dissimilar". The forward pass is an
identity passthrough; Backpropagate writes the cosine-loss +gradient into the a
and b channels (0 into the y channel).

To produce that a|b|y layout natively we build a TWO-INPUT net:

  Input0(SizeX=2, SizeY=1, Depth=cInDim)   <- the pair: point a at X=0, b at X=1
  Input1(SizeX=1, SizeY=1, Depth=1)        <- the scalar label y

  branch off Input0:
    PointwiseConvReLU / PointwiseConvLinear (featuresize=1)
                                <- the SAME weights are applied at every X
                                   position => a genuine SHARED embedding MLP
                                   (siamese: a and b go through identical weights)
    L2Normalize                 <- each point's embedding on the unit sphere
      -> shape (2,1,d)
    Reshape(1,1,2*d)            <- pure reinterpretation; depth-major storage
                                   makes this exactly a|b in depth

  DeepConcat([Reshape, Input1]) <- appends the y channel -> (1,1,2*d+1) = a|b|y
  TNNetCosineEmbeddingLoss(margin)

Because TNNetVolume is depth-major (pos = ((SizeX*y)+x)*Depth + d), the two
per-X embeddings land as consecutive depth chunks after the reshape, which is
exactly the a|b layout the head expects; the depth concat then tacks on y.

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

uses {$IFDEF UNIX} {$IFDEF UseCThreads}
  cthreads, {$ENDIF} {$ENDIF}
  Classes, SysUtils,
  neuralnetwork,
  neuralvolume;

const
  cNumClasses    = 4;
  cInDim         = 4;     // raw input dimensionality of a point
  cEmbedDim      = 3;     // d: embedding dim on the unit sphere
  cSamplesPerCls = 60;    // synthetic samples per class
  cEpochs        = 60;
  cStepsEp       = 256;   // pair training steps per epoch
  cLearningRate  = 0.05;
  cMargin        = 0.2;   // dissimilar pairs are pushed below this cosine
  cSeed          = 12345;
  cSigma         = 0.45;  // blob spread
  cEvalPairs     = 600;   // held-out pairs per group for the histograms

type
  TSample = record
    X:   array[0..cInDim - 1] of TNeuralFloat;
    Cls: integer;
  end;
  TSampleArray = array of TSample;

var
  // cNumClasses random, well-separated blob centers in cInDim space.
  Centers: array[0..cNumClasses - 1, 0..cInDim - 1] of TNeuralFloat;

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

// Random class centers with a decent pairwise separation.
procedure BuildCenters();
var
  C, D: integer;
begin
  for C := 0 to cNumClasses - 1 do
    for D := 0 to cInDim - 1 do
      Centers[C][D] := RandomGauss() * 2.0;
end;

// Builds the synthetic dataset: cSamplesPerCls nD Gaussian points per class.
procedure BuildDataset(out Data: TSampleArray);
var
  C, I, D, Idx: integer;
begin
  SetLength(Data, cNumClasses * cSamplesPerCls);
  Idx := 0;
  for C := 0 to cNumClasses - 1 do
    for I := 1 to cSamplesPerCls do
    begin
      for D := 0 to cInDim - 1 do
        Data[Idx].X[D] := Centers[C][D] + RandomGauss() * cSigma;
      Data[Idx].Cls := C;
      Inc(Idx);
    end;
end;

function RandomIdxOfClass(const Data: TSampleArray; Cls: integer): integer;
begin
  repeat
    Result := Random(Length(Data));
  until Data[Result].Cls = Cls;
end;

// Samples a pair. If Same, both members share a class (y=1); otherwise the two
// members are from different classes (y=0). Returns the two sample indices.
procedure SamplePair(const Data: TSampleArray; Same: boolean;
  out IdxA, IdxB: integer);
var
  ClsA, ClsB: integer;
begin
  IdxA := Random(Length(Data));
  ClsA := Data[IdxA].Cls;
  if Same then
    IdxB := RandomIdxOfClass(Data, ClsA)
  else
  begin
    repeat
      ClsB := Random(cNumClasses);
    until ClsB <> ClsA;
    IdxB := RandomIdxOfClass(Data, ClsB);
  end;
end;

// Fills the two-input pair tensors: Pair holds point a at X=0, b at X=1; YVol
// holds the scalar similarity label.
procedure FillPair(Pair, YVol: TNNetVolume; const Data: TSampleArray;
  IdxA, IdxB: integer; Y: TNeuralFloat);
var
  D: integer;
begin
  Pair.Fill(0);
  for D := 0 to cInDim - 1 do
  begin
    Pair[0, 0, D] := Data[IdxA].X[D];
    Pair[1, 0, D] := Data[IdxB].X[D];
  end;
  YVol[0, 0, 0] := Y;
end;

// Builds the weight-shared siamese embedding + cosine-embedding-loss network.
// Input0 = the pair (2,1,cInDim); Input1 = the label y (1,1,1). They MUST be
// the first two layers so the multi-input Compute/Backpropagate overloads can
// feed them by index.
function BuildNet(): TNNet;
var
  PairIn, YIn, EmbEnd: TNNetLayer;
begin
  Result := TNNet.Create();
  PairIn := Result.AddLayer(TNNetInput.Create(2, 1, cInDim));
  YIn    := Result.AddLayer(TNNetInput.Create(1, 1, 1));
  // Shared MLP over the 2 X positions (pointwise = same weights per position).
  Result.AddLayerAfter(TNNetPointwiseConvReLU.Create(16), PairIn);
  Result.AddLayer(TNNetPointwiseConvReLU.Create(16));
  Result.AddLayer(TNNetPointwiseConvLinear.Create(cEmbedDim));
  // Each point's embedding onto the unit sphere (per-(x,y) over depth).
  Result.AddLayer(TNNetL2Normalize.Create());
  // Reinterpret (2,1,d) as (1,1,2*d) = a|b depth layout.
  EmbEnd := Result.AddLayer(TNNetReshape.Create(1, 1, 2 * cEmbedDim));
  // Append the y channel -> (1,1,2*d+1) = a|b|y, exactly what the head expects.
  Result.AddLayer(TNNetDeepConcat.Create([EmbEnd, YIn]));
  Result.AddLayer(TNNetCosineEmbeddingLoss.Create(cMargin));
  Result.InitWeights();
end;

// Computes cos(a,b) from the loss-head output (a|b|y layout). Embeddings are
// L2-normalized so the cosine is just the dot product of the two slabs.
function CosOfOutput(NN: TNNet): TNeuralFloat;
var
  D: integer;
  Dot: TNeuralFloat;
begin
  Dot := 0;
  for D := 0 to cEmbedDim - 1 do
    Dot := Dot + NN.GetLastLayer().Output.FData[D] *
                 NN.GetLastLayer().Output.FData[D + cEmbedDim];
  Result := Dot;
end;

// One training epoch: cStepsEp pairs, half same / half different.
function TrainEpoch(NN: TNNet; const Data: TSampleArray;
  Pair, YVol, Dummy: TNNetVolume): TNeuralFloat;
var
  Step, IdxA, IdxB: integer;
  Same: boolean;
  Y, Cos, HingeArg, Loss, TotalLoss: TNeuralFloat;
begin
  TotalLoss := 0;
  for Step := 0 to cStepsEp - 1 do
  begin
    Same := (Step and 1) = 0;
    if Same then Y := 1.0 else Y := 0.0;
    SamplePair(Data, Same, IdxA, IdxB);
    FillPair(Pair, YVol, Data, IdxA, IdxB, Y);

    NN.Compute([Pair, YVol]);
    // The head is self-contained; the target is an ignored placeholder.
    NN.Backpropagate(Dummy);

    Cos := CosOfOutput(NN);
    HingeArg := Cos - cMargin;
    if HingeArg < 0 then HingeArg := 0;
    Loss := Y * (1 - Cos) + (1 - Y) * HingeArg * HingeArg;
    TotalLoss := TotalLoss + Loss;
  end;
  Result := TotalLoss / cStepsEp;
end;

// Bins cosine values from -1..1 into a fixed-width histogram and prints it.
procedure PrintHistogram(const Title: string; const Cos: array of TNeuralFloat;
  N: integer);
const
  cBins = 10;
var
  Hist: array[0..cBins - 1] of integer;
  I, B, BarLen, MaxCnt: integer;
  Lo, Hi, Sum, Mean, Var_, SD: TNeuralFloat;
begin
  for B := 0 to cBins - 1 do Hist[B] := 0;
  Sum := 0;
  for I := 0 to N - 1 do
  begin
    Sum := Sum + Cos[I];
    B := Trunc((Cos[I] + 1.0) / 2.0 * cBins);   // map [-1,1] -> [0,cBins)
    if B < 0 then B := 0;
    if B > cBins - 1 then B := cBins - 1;
    Inc(Hist[B]);
  end;
  Mean := Sum / N;
  Var_ := 0;
  for I := 0 to N - 1 do Var_ := Var_ + Sqr(Cos[I] - Mean);
  SD := Sqrt(Var_ / N);

  MaxCnt := 1;
  for B := 0 to cBins - 1 do
    if Hist[B] > MaxCnt then MaxCnt := Hist[B];

  WriteLn;
  WriteLn(Format('%s  (n=%d)  mean cos = %.3f  +/- %.3f', [Title, N, Mean, SD]));
  for B := 0 to cBins - 1 do
  begin
    Lo := -1.0 + 2.0 * B / cBins;
    Hi := -1.0 + 2.0 * (B + 1) / cBins;
    BarLen := Round(40 * Hist[B] / MaxCnt);
    WriteLn(Format('  [%6.2f,%6.2f) %5d |%s',
      [Lo, Hi, Hist[B], StringOfChar('#', BarLen)]));
  end;
end;

// After training, sample held-out pairs of each group and print histograms.
procedure ReportHistograms(NN: TNNet; const Data: TSampleArray);
var
  Pair, YVol: TNNetVolume;
  SameCos, DiffCos: array of TNeuralFloat;
  I, IdxA, IdxB: integer;
begin
  Pair := TNNetVolume.Create(2, 1, cInDim);
  YVol := TNNetVolume.Create(1, 1, 1);
  SetLength(SameCos, cEvalPairs);
  SetLength(DiffCos, cEvalPairs);
  try
    for I := 0 to cEvalPairs - 1 do
    begin
      SamplePair(Data, True, IdxA, IdxB);
      FillPair(Pair, YVol, Data, IdxA, IdxB, 1.0);
      NN.Compute([Pair, YVol]);
      SameCos[I] := CosOfOutput(NN);

      SamplePair(Data, False, IdxA, IdxB);
      FillPair(Pair, YVol, Data, IdxA, IdxB, 0.0);
      NN.Compute([Pair, YVol]);
      DiffCos[I] := CosOfOutput(NN);
    end;
    WriteLn;
    WriteLn('Learned cosine-similarity histograms (margin = ', cMargin:0:2, ')');
    PrintHistogram('SAME-class pairs (target cos -> +1)', SameCos, cEvalPairs);
    PrintHistogram('DIFFERENT-class pairs (pushed below margin)',
      DiffCos, cEvalPairs);
  finally
    YVol.Free;
    Pair.Free;
  end;
end;

procedure RunAlgo();
var
  NN: TNNet;
  Data: TSampleArray;
  Pair, YVol, Dummy: TNNetVolume;
  Epoch: integer;
  MeanLoss: TNeuralFloat;
begin
  RandSeed := cSeed;
  WriteLn('CosineEmbeddingSiamese: pairwise cosine-embedding metric learning');
  WriteLn('classes: ', cNumClasses, '  in_dim: ', cInDim,
    '  embed_dim: ', cEmbedDim, '  margin: ', cMargin:0:2);

  BuildCenters();
  BuildDataset(Data);
  NN := BuildNet();
  NN.SetLearningRate(cLearningRate, 0.9);

  Pair  := TNNetVolume.Create(2, 1, cInDim);
  YVol  := TNNetVolume.Create(1, 1, 1);
  // The head is self-contained; this target placeholder matches the output.
  Dummy := TNNetVolume.Create(1, 1, 2 * cEmbedDim + 1);
  Dummy.Fill(0);
  try
    WriteLn;
    WriteLn('Training for ', cEpochs, ' epochs (', cStepsEp, ' pairs/epoch)...');
    for Epoch := 1 to cEpochs do
    begin
      MeanLoss := TrainEpoch(NN, Data, Pair, YVol, Dummy);
      if (Epoch = 1) or (Epoch mod 10 = 0) then
        WriteLn(Format('  epoch %4d   mean_loss=%8.5f', [Epoch, MeanLoss]));
    end;
    ReportHistograms(NN, Data);
  finally
    Dummy.Free;
    YVol.Free;
    Pair.Free;
    NN.Free;
  end;
end;

var
  // Stops Lazarus errors
  Application: record Title: string; end;

begin
  Application.Title := 'CosineEmbeddingSiamese Example';
  RunAlgo();
end.
