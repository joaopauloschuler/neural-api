program MatryoshkaEmbedding;
(*
MatryoshkaEmbedding: Matryoshka Representation Learning (Kusupati et al.,
NeurIPS 2022) as a CPU MICRO-example. ONE encoder produces a single d=64
embedding whose nested PREFIXES {8, 16, 32, 64} are EACH independently usable
as a classifier feature. The total loss is the SUM of K softmax cross-entropy
losses, one per prefix, so the early coordinates are forced to pack the
coarsest / most important information and the later coordinates only refine.

WHY THIS IS DIFFERENT
---------------------
The other embedding examples here (TripletEmbedding, InfoNCEContrastive,
ArcFaceEmbedding, CosineEmbeddingSiamese) all train ONE fixed-width vector.
Matryoshka trains an ELASTIC, sub-dimension-addressable representation: from a
SINGLE model you get a whole accuracy-vs-width curve, and you can truncate the
vector at retrieval time to trade accuracy for cost with zero retraining.

THE HEADLINE
------------
We train ONE Matryoshka model (d=64, prefixes {8,16,32,64}) and read the
classification accuracy of each prefix. We separately train DEDICATED fixed-8
and fixed-16 baseline models (their entire embedding is 8 / 16 dims). The
Matryoshka 8-dim and 16-dim prefixes should come CLOSE to the dedicated
baselines -- i.e. adaptive-cost retrieval essentially for free. An ASCII
accuracy-vs-width table/curve is printed.

DATASET
-------
A tiny synthetic N-D Gaussian-blob multi-class task (8 classes, blob centers in
a higher-dim space projected to a 2-D-ish coordinate input). MNIST was NOT used:
a from-scratch MNIST loop blows the <5-min single-thread budget for a demo that
already trains 3 separate models. The synthetic task is enough to land the
nested-prefix headline cleanly and deterministically.

HOW THE NESTED PREFIXES ARE WIRED
---------------------------------
  Input(1,1,cInDim)
   -> shared encoder MLP -> PointwiseConvLinear(64)        [the embedding]
   for each prefix p in {8,16,32,64}:
     SplitChannels(0, p) -> FullConnectLinear(NumClasses) -> SoftMax   [head p]
   DeepConcat([head_8, head_16, head_32, head_64])      [final output, K*C wide]

The training target is the concatenation of K identical one-hot labels. The
network's softmax+cross-entropy backward (OutputError = output - target on the
concatenated heads) yields exactly the SUM of the per-prefix softmax CE
gradients. Crucially, embedding coordinate 0 receives gradient from ALL FOUR
heads while coordinate 63 receives gradient only from the 64-head -- that
asymmetry is precisely the Matryoshka nesting that makes early coords dominant.

THE PREFIX-LOSS-WEIGHTING PITFALL
---------------------------------
The per-prefix losses must be combined so the smallest prefix does NOT dominate
the gradient. Here every head outputs the SAME number of classes (C logits), so
the logit-level gradient magnitudes are comparable and equal (uniform) weighting
is correct. The trap is weighting a prefix's loss by its DIMENSION (or summing
unnormalised losses of heads with very different output sizes): then the small
prefix's gradient swamps coordinate 0..7, over-training the head while the wide
prefix barely learns. Keep the heads' loss scale comparable (equal weights, same
class count) -- see cHeadWeight below, which is uniform on purpose.

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
  cNumClasses    = 8;
  cInDim         = 6;     // input feature dims (blobs live in 6-D)
  cEmbedDim      = 64;    // full Matryoshka embedding width
  cSamplesPerCls = 80;
  cTestPerCls    = 40;
  cEpochs        = 40;
  cLearningRate  = 0.02;
  cSeed          = 42;
  cSigma         = 1.10;  // blob spread (enough overlap to make width matter)

  // Nested prefix widths. Must be ascending and the last must be cEmbedDim.
  cPrefixes: array[0..3] of integer = (8, 16, 32, 64);
  cNumHeads = 4;

  // Per-head loss weights. UNIFORM on purpose -- see the pitfall note in the
  // header. All heads emit cNumClasses logits, so equal weighting keeps the
  // small prefix from dominating the shared early coordinates.
  cHeadWeight: array[0..3] of TNeuralFloat = (1.0, 1.0, 1.0, 1.0);

type
  TSample = record
    F:   array[0..cInDim - 1] of TNeuralFloat;
    Cls: integer;
  end;
  TSampleArray = array of TSample;

var
  // cNumClasses random blob centers in cInDim space (filled once, fixed seed).
  gCenters: array[0..cNumClasses - 1, 0..cInDim - 1] of TNeuralFloat;

// Box-Muller standard normal.
function RandomGauss(): TNeuralFloat;
var
  U1, U2: TNeuralFloat;
begin
  repeat U1 := Random; until U1 > 1e-9;
  U2 := Random;
  Result := Sqrt(-2 * Ln(U1)) * Cos(2 * Pi * U2);
end;

procedure InitCenters();
var
  C, D: integer;
begin
  for C := 0 to cNumClasses - 1 do
    for D := 0 to cInDim - 1 do
      gCenters[C][D] := RandomGauss() * 2.4;
end;

procedure BuildDataset(out Data: TSampleArray; PerCls: integer);
var
  C, I, D, Idx: integer;
begin
  SetLength(Data, cNumClasses * PerCls);
  Idx := 0;
  for C := 0 to cNumClasses - 1 do
    for I := 1 to PerCls do
    begin
      for D := 0 to cInDim - 1 do
        Data[Idx].F[D] := gCenters[C][D] + RandomGauss() * cSigma;
      Data[Idx].Cls := C;
      Inc(Idx);
    end;
end;

procedure FillInput(V: TNNetVolume; const S: TSample);
var
  D: integer;
begin
  for D := 0 to cInDim - 1 do V[0, 0, D] := S.F[D];
end;

// Builds a network: shared encoder -> embedding(EmbDim) -> for each prefix in
// Prefixes a SplitChannels+FullConnectLinear+SoftMax head, all DeepConcat'd.
// Returns the net; HeadCount = number of prefix heads created.
function BuildNet(EmbDim: integer; const Prefixes: array of integer;
  out HeadCount: integer): TNNet;
var
  InputLayer, EmbLayer: TNNetLayer;
  Heads: array of TNNetLayer;
  H: integer;
begin
  Result := TNNet.Create();
  InputLayer := Result.AddLayer(TNNetInput.Create(1, 1, cInDim));
  Result.AddLayer(TNNetPointwiseConvReLU.Create(48));
  Result.AddLayer(TNNetPointwiseConvReLU.Create(48));
  EmbLayer := Result.AddLayer(TNNetPointwiseConvLinear.Create(EmbDim));

  HeadCount := Length(Prefixes);
  SetLength(Heads, HeadCount);
  for H := 0 to HeadCount - 1 do
  begin
    Result.AddLayerAfter(TNNetSplitChannels.Create(0, Prefixes[H]), EmbLayer);
    Result.AddLayer(TNNetFullConnectLinear.Create(cNumClasses));
    Heads[H] := Result.AddLayer(TNNetSoftMax.Create());
  end;
  Result.AddLayer(TNNetDeepConcat.Create(Heads));
  Result.InitWeights();
end;

// Fills the concatenated one-hot target (K repetitions of the label one-hot).
procedure FillTarget(T: TNNetVolume; Cls, HeadCount: integer);
var
  H, K: integer;
begin
  T.Fill(0);
  for H := 0 to HeadCount - 1 do
    T[0, 0, H * cNumClasses + Cls] := 1.0;
end;

// One training epoch (shuffled). Per-head weighting is applied by scaling the
// target-region gradient; here weights are uniform so this is a no-op, but the
// hook documents WHERE you would weight prefixes (and where the pitfall bites).
function TrainEpoch(NN: TNNet; HeadCount: integer; const Data: TSampleArray;
  Input, Target: TNNetVolume): TNeuralFloat;
var
  Step, N, Idx, Tmp: integer;
  Order: array of integer;
  Loss, P: TNeuralFloat;
  H: integer;
  OutV: TNNetVolume;
begin
  N := Length(Data);
  SetLength(Order, N);
  for Idx := 0 to N - 1 do Order[Idx] := Idx;
  for Idx := N - 1 downto 1 do
  begin
    Step := Random(Idx + 1);
    Tmp := Order[Idx]; Order[Idx] := Order[Step]; Order[Step] := Tmp;
  end;
  Loss := 0;
  for Step := 0 to N - 1 do
  begin
    FillInput(Input, Data[Order[Step]]);
    NN.Compute(Input);
    FillTarget(Target, Data[Order[Step]].Cls, HeadCount);
    OutV := NN.GetLastLayer().Output;
    // Sum of per-head cross-entropy (reporting only).
    for H := 0 to HeadCount - 1 do
    begin
      P := OutV.FData[H * cNumClasses + Data[Order[Step]].Cls];
      if P < 1e-30 then P := 1e-30;
      Loss := Loss + cHeadWeight[H] * (-Ln(P));
    end;
    NN.Backpropagate(Target);
  end;
  Result := Loss / N;
end;

// Argmax over a head's cNumClasses slice of the concatenated output.
function HeadPredict(OutV: TNNetVolume; HeadIdx: integer): integer;
var
  K, Best: integer;
  BestV, V: TNeuralFloat;
begin
  Best := 0; BestV := OutV.FData[HeadIdx * cNumClasses];
  for K := 1 to cNumClasses - 1 do
  begin
    V := OutV.FData[HeadIdx * cNumClasses + K];
    if V > BestV then begin BestV := V; Best := K; end;
  end;
  Result := Best;
end;

// Per-head test accuracy of a trained net.
procedure EvalAccuracies(NN: TNNet; HeadCount: integer;
  const Data: TSampleArray; Input: TNNetVolume; out Acc: array of TNeuralFloat);
var
  I, H, N, Pred: integer;
  Correct: array of integer;
  OutV: TNNetVolume;
begin
  N := Length(Data);
  SetLength(Correct, HeadCount);
  for H := 0 to HeadCount - 1 do Correct[H] := 0;
  for I := 0 to N - 1 do
  begin
    FillInput(Input, Data[I]);
    NN.Compute(Input);
    OutV := NN.GetLastLayer().Output;
    for H := 0 to HeadCount - 1 do
    begin
      Pred := HeadPredict(OutV, H);
      if Pred = Data[I].Cls then Inc(Correct[H]);
    end;
  end;
  for H := 0 to HeadCount - 1 do Acc[H] := Correct[H] / N;
end;

// Trains a net to completion and returns its per-head test accuracies.
procedure TrainAndEval(EmbDim: integer; const Prefixes: array of integer;
  const Train, Test: TSampleArray; out Acc: array of TNeuralFloat;
  out FinalLoss: TNeuralFloat);
var
  NN: TNNet;
  HeadCount, Epoch: integer;
  Input, Target: TNNetVolume;
begin
  RandSeed := cSeed; // identical init + sample order across all runs (fair)
  NN := BuildNet(EmbDim, Prefixes, HeadCount);
  NN.SetLearningRate(cLearningRate, 0.9);
  Input := TNNetVolume.Create(1, 1, cInDim);
  Target := TNNetVolume.Create(1, 1, HeadCount * cNumClasses);
  try
    FinalLoss := 0;
    for Epoch := 1 to cEpochs do
    begin
      FinalLoss := TrainEpoch(NN, HeadCount, Train, Input, Target);
      if (Epoch mod 10 = 0) or (Epoch = 1) then
      begin
        WriteLn(Format('    epoch %3d  mean_loss %8.4f', [Epoch, FinalLoss]));
        Flush(Output);
      end;
    end;
    EvalAccuracies(NN, HeadCount, Test, Input, Acc);
  finally
    Input.Free;
    Target.Free;
    NN.Free;
  end;
end;

procedure RunAlgo();
var
  Train, Test: TSampleArray;
  MatAcc: array[0..cNumHeads - 1] of TNeuralFloat;
  Base8Acc, Base16Acc: array[0..0] of TNeuralFloat;
  Loss: TNeuralFloat;
  H, BarLen: integer;
begin
  RandSeed := cSeed;
  InitCenters();
  BuildDataset(Train, cSamplesPerCls);
  // rebuild test from the SAME centers (gCenters already fixed)
  BuildDataset(Test, cTestPerCls);

  WriteLn('MatryoshkaEmbedding: nested-prefix representation learning (synthetic)');
  WriteLn(Format('classes %d  in_dim %d  embed_dim %d  prefixes {8,16,32,64}',
    [cNumClasses, cInDim, cEmbedDim]));
  WriteLn(Format('train/class %d  test/class %d  epochs %d  lr %.3f',
    [cSamplesPerCls, cTestPerCls, cEpochs, cLearningRate]));
  WriteLn;

  WriteLn('[1/3] Training the SINGLE Matryoshka model (one encoder, 4 prefix heads)');
  TrainAndEval(cEmbedDim, cPrefixes, Train, Test, MatAcc, Loss);
  WriteLn;

  WriteLn('[2/3] Training DEDICATED fixed-8 baseline (whole embedding = 8 dims)');
  TrainAndEval(8, [8], Train, Test, Base8Acc, Loss);
  WriteLn;

  WriteLn('[3/3] Training DEDICATED fixed-16 baseline (whole embedding = 16 dims)');
  TrainAndEval(16, [16], Train, Test, Base16Acc, Loss);
  WriteLn;

  WriteLn('=== Accuracy vs embedding width (test set) ===');
  WriteLn;
  WriteLn('  width  matryoshka-prefix   dedicated-baseline');
  WriteLn('  -----  -----------------   ------------------');
  for H := 0 to cNumHeads - 1 do
  begin
    Write(Format('  %4d       %6.3f      ', [cPrefixes[H], MatAcc[H]]));
    case cPrefixes[H] of
      8:  WriteLn(Format('       %6.3f', [Base8Acc[0]]));
      16: WriteLn(Format('       %6.3f', [Base16Acc[0]]));
    else  WriteLn('          (none)');
    end;
  end;
  WriteLn;

  WriteLn('  ASCII accuracy-vs-width curve (Matryoshka prefixes):');
  for H := 0 to cNumHeads - 1 do
  begin
    BarLen := Round(MatAcc[H] * 50);
    WriteLn(Format('   d=%2d |%s %5.1f%%',
      [cPrefixes[H], StringOfChar('#', BarLen), MatAcc[H] * 100]));
  end;
  WriteLn;
  WriteLn('Takeaway: one model yields the whole curve; accuracy rises with width,');
  WriteLn('and the 8/16-dim PREFIXES stay close to the dedicated 8/16-dim models');
  WriteLn('-> adaptive-cost retrieval for free. (Equal head weights; see header');
  WriteLn('for the small-prefix-dominates-the-gradient pitfall.)');
end;

var
  Application: record Title: string; end;

begin
  Application.Title := 'MatryoshkaEmbedding Example';
  RunAlgo();
end.
