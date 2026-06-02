program TripletEmbedding;
(*
TripletEmbedding: learns a low-dimensional metric-learning embedding of a
SYNTHETIC multi-class toy dataset using the TNNetTripletLoss head and
TNNetL2Normalize. No external dataset, pure CPU, finishes in seconds.

WHAT IT SHOWS
-------------
Four classes of 2D Gaussian blobs at distinct centers are mapped to a 3-D
embedding constrained to the unit sphere. Training pulls same-class samples
together and pushes different-class samples apart using the triplet hinge
  L = max(0, ||a - p||^2 - ||a - n||^2 + margin)
After training, a per-class mean pairwise COSINE-SIMILARITY matrix is printed:
within-class entries (diagonal) come out high (~1), cross-class entries low.
The learned embeddings are also dumped to embeddings.csv for plotting.

HOW THE TRIPLET HEAD IS WIRED (weight-shared siamese, fully native)
--------------------------------------------------------------------
TNNetTripletLoss has NO external target: supervision is implicit in the input
depth layout. It splits the input depth into 3 equal anchor|positive|negative
chunks (Depth mod 3 = 0) and computes the hinge per spatial cell. To feed it we
use ONE network that processes a triplet at once:

  Input(SizeX=3, SizeY=1, Depth=2)   <- 3 points (a,p,n) at X=0,1,2; 2 coords
  PointwiseConvReLU / PointwiseConvLinear (featuresize=1)
                                     <- the SAME weights are applied at every X
                                        position => a genuine SHARED embedding
                                        MLP (siamese over the 3 points)
  L2Normalize (per-(x,y)-over-depth) <- each point's embedding on unit sphere
  -> output shape (3, 1, embed_dim)
  Reshape(1, 1, 3*embed_dim)         <- pure reinterpretation; FData order is
                                        [a_0..a_e-1, p_0.., n_0..] i.e. a|p|n
  TNNetTripletLoss                   <- consumes the a|p|n depth layout

Because the volume is stored depth-major (pos = ((SizeX*y)+x)*Depth + d), the
three per-X embeddings are already laid out as consecutive depth chunks after
the reshape, which is exactly the a|p|n layout the loss head expects.

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
  Classes, SysUtils, Math,
  neuralnetwork,
  neuralvolume;

const
  cNumClasses    = 4;
  cEmbedDim      = 3;     // 3-D embedding on the unit sphere
  cSamplesPerCls = 80;    // synthetic samples per class
  cEpochs        = 120;
  cLearningRate  = 0.05;
  cMargin        = 1.0;
  cSeed          = 12345;
  cSigma         = 0.45;  // blob spread

  // Four 2D Gaussian blob centers, well separated.
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

// Picks a random sample index belonging to the requested class.
function RandomIdxOfClass(const Data: TSampleArray; Cls: integer): integer;
begin
  repeat
    Result := Random(Length(Data));
  until Data[Result].Cls = Cls;
end;

// Fills the triplet input volume (3,1,2): X=0 anchor, X=1 positive, X=2 negative.
procedure FillTriplet(V: TNNetVolume; const Data: TSampleArray;
  AnchorIdx, PosIdx, NegIdx: integer);
begin
  V.Fill(0);
  V[0, 0, 0] := Data[AnchorIdx].X;  V[0, 0, 1] := Data[AnchorIdx].Y;
  V[1, 0, 0] := Data[PosIdx].X;     V[1, 0, 1] := Data[PosIdx].Y;
  V[2, 0, 0] := Data[NegIdx].X;     V[2, 0, 1] := Data[NegIdx].Y;
end;

// Builds the weight-shared siamese embedding + triplet-loss network.
function BuildNet(): TNNet;
begin
  Result := TNNet.Create();
  Result.AddLayer(TNNetInput.Create(3, 1, 2));
  // Shared MLP over the 3 X positions (pointwise = same weights per position).
  Result.AddLayer(TNNetPointwiseConvReLU.Create(16));
  Result.AddLayer(TNNetPointwiseConvReLU.Create(16));
  Result.AddLayer(TNNetPointwiseConvLinear.Create(cEmbedDim));
  // Project each point's embedding onto the unit sphere (per-(x,y) over depth).
  Result.AddLayer(TNNetL2Normalize.Create());
  // Reinterpret (3,1,embed) as (1,1,3*embed) = a|p|n depth layout.
  Result.AddLayer(TNNetReshape.Create(1, 1, 3 * cEmbedDim));
  Result.AddLayer(TNNetTripletLoss.Create(cMargin));
  Result.InitWeights();
end;

// Computes the embedding of a single 2D point by feeding it as the anchor of a
// dummy triplet and reading the anchor chunk of the L2-normalized output.
procedure EmbedPoint(NN: TNNet; PX, PY: TNeuralFloat;
  Emb: TNNetVolume; Input: TNNetVolume);
var
  D: integer;
begin
  Input.Fill(0);
  Input[0, 0, 0] := PX;  Input[0, 0, 1] := PY;
  Input[1, 0, 0] := PX;  Input[1, 0, 1] := PY;
  Input[2, 0, 0] := PX;  Input[2, 0, 1] := PY;
  NN.Compute(Input);
  // Output is (1,1,3*embed); the anchor chunk is the first embed_dim entries.
  for D := 0 to cEmbedDim - 1 do
    Emb.FData[D] := NN.GetLastLayer().Output.FData[D];
end;

// Mean hinge over the dataset's current triplets (for progress reporting).
function TrainEpoch(NN: TNNet; const Data: TSampleArray;
  Dummy: TNNetVolume): TNeuralFloat;
var
  Step, NumSamples, AnchorIdx, PosIdx, NegIdx, NegCls, D, ChunkD: integer;
  Input: TNNetVolume;
  av, pv, nv, DistAP, DistAN, Hinge: TNeuralFloat;
  TotalLoss: TNeuralFloat;
begin
  NumSamples := Length(Data);
  Input := TNNetVolume.Create(3, 1, 2);
  TotalLoss := 0;
  try
    for Step := 0 to NumSamples - 1 do
    begin
      AnchorIdx := Random(NumSamples);
      PosIdx := RandomIdxOfClass(Data, Data[AnchorIdx].Cls);
      repeat
        NegCls := Random(cNumClasses);
      until NegCls <> Data[AnchorIdx].Cls;
      NegIdx := RandomIdxOfClass(Data, NegCls);

      FillTriplet(Input, Data, AnchorIdx, PosIdx, NegIdx);
      NN.Compute(Input);
      // Triplet loss ignores the target; Dummy is just a placeholder.
      NN.Backpropagate(Dummy);

      // Read the hinge for reporting from the loss-head output (a|p|n layout).
      ChunkD := cEmbedDim;
      DistAP := 0;
      DistAN := 0;
      for D := 0 to ChunkD - 1 do
      begin
        av := NN.GetLastLayer().Output.FData[D];
        pv := NN.GetLastLayer().Output.FData[D + ChunkD];
        nv := NN.GetLastLayer().Output.FData[D + 2 * ChunkD];
        DistAP := DistAP + (av - pv) * (av - pv);
        DistAN := DistAN + (av - nv) * (av - nv);
      end;
      Hinge := DistAP - DistAN + cMargin;
      if Hinge < 0 then Hinge := 0;
      TotalLoss := TotalLoss + Hinge;
    end;
  finally
    Input.Free;
  end;
  Result := TotalLoss / NumSamples;
end;

// Prints the per-class mean pairwise cosine-similarity matrix and dumps the
// learned embeddings to embeddings.csv.
procedure ReportEmbeddings(NN: TNNet; const Data: TSampleArray);
var
  Input, Emb: TNNetVolume;
  Embeds: array of array of TNeuralFloat;  // [sample][dim]
  Sim: array[0..cNumClasses - 1, 0..cNumClasses - 1] of TNeuralFloat;
  Cnt: array[0..cNumClasses - 1, 0..cNumClasses - 1] of integer;
  I, J, D, Ci, Cj, N: integer;
  Dot: TNeuralFloat;
  CsvFile: TextFile;
begin
  N := Length(Data);
  Input := TNNetVolume.Create(3, 1, 2);
  Emb := TNNetVolume.Create(cEmbedDim, 1, 1);
  SetLength(Embeds, N, cEmbedDim);
  try
    for I := 0 to N - 1 do
    begin
      EmbedPoint(NN, Data[I].X, Data[I].Y, Emb, Input);
      for D := 0 to cEmbedDim - 1 do
        Embeds[I][D] := Emb.FData[D];
    end;

    // Accumulate mean cosine similarity per class pair. Embeddings are already
    // unit-norm, so cosine similarity is just the dot product.
    for Ci := 0 to cNumClasses - 1 do
      for Cj := 0 to cNumClasses - 1 do
      begin
        Sim[Ci][Cj] := 0;
        Cnt[Ci][Cj] := 0;
      end;
    for I := 0 to N - 1 do
      for J := 0 to N - 1 do
        if I <> J then
        begin
          Ci := Data[I].Cls;
          Cj := Data[J].Cls;
          Dot := 0;
          for D := 0 to cEmbedDim - 1 do
            Dot := Dot + Embeds[I][D] * Embeds[J][D];
          Sim[Ci][Cj] := Sim[Ci][Cj] + Dot;
          Inc(Cnt[Ci][Cj]);
        end;

    WriteLn;
    WriteLn('Per-class mean pairwise COSINE-SIMILARITY matrix');
    WriteLn('(diagonal = within-class should be HIGH; off-diagonal = cross-class LOW)');
    Write('          ');
    for Cj := 0 to cNumClasses - 1 do
      Write(Format('class%d  ', [Cj]));
    WriteLn;
    for Ci := 0 to cNumClasses - 1 do
    begin
      Write(Format('class%d  ', [Ci]));
      for Cj := 0 to cNumClasses - 1 do
      begin
        if Cnt[Ci][Cj] > 0 then
          Sim[Ci][Cj] := Sim[Ci][Cj] / Cnt[Ci][Cj];
        Write(Format('%8.3f', [Sim[Ci][Cj]]));
      end;
      WriteLn;
    end;

    // Dump embeddings to CSV for plotting.
    AssignFile(CsvFile, 'embeddings.csv');
    Rewrite(CsvFile);
    WriteLn(CsvFile, 'class,e0,e1,e2');
    for I := 0 to N - 1 do
      WriteLn(CsvFile, Format('%d,%.6f,%.6f,%.6f',
        [Data[I].Cls, Embeds[I][0], Embeds[I][1], Embeds[I][2]]));
    CloseFile(CsvFile);
    WriteLn;
    WriteLn('Wrote learned embeddings to embeddings.csv');
  finally
    Emb.Free;
    Input.Free;
  end;
end;

procedure RunAlgo();
var
  NN: TNNet;
  Data: TSampleArray;
  Dummy: TNNetVolume;
  Epoch: integer;
  MeanHinge: TNeuralFloat;
begin
  RandSeed := cSeed;
  WriteLn('TripletEmbedding: metric learning on synthetic 2D Gaussian blobs');
  WriteLn('Classes: ', cNumClasses, '  embed_dim: ', cEmbedDim,
    '  samples/class: ', cSamplesPerCls, '  margin: ', cMargin:0:2);
  WriteLn;

  BuildDataset(Data);
  NN := BuildNet();
  // Triplet loss ignores the target; the dummy matches the output shape.
  Dummy := TNNetVolume.Create(1, 1, 3 * cEmbedDim);
  Dummy.Fill(0);
  NN.SetLearningRate(cLearningRate, 0.9);
  try
    WriteLn('Training for ', cEpochs, ' epochs...');
    for Epoch := 1 to cEpochs do
    begin
      MeanHinge := TrainEpoch(NN, Data, Dummy);
      if (Epoch = 1) or (Epoch mod 20 = 0) then
        WriteLn(Format('  epoch %4d   mean_hinge=%8.5f', [Epoch, MeanHinge]));
    end;
    ReportEmbeddings(NN, Data);
  finally
    Dummy.Free;
    NN.Free;
  end;
end;

var
  // Stops Lazarus errors
  Application: record Title: string; end;

begin
  Application.Title := 'TripletEmbedding Example';
  RunAlgo();
end.
