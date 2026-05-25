program InfoNCEContrastive;
(*
InfoNCEContrastive: a self-contained CONTRASTIVE representation-learning
micro-example built on the TNNetInfoNCELoss head. A tiny weight-shared MLP
encoder is trained on a SYNTHETIC task where each training sample packs a
query, its positive (an augmented view of the SAME latent class), and K
negatives (views of OTHER classes) into one input. No external dataset, pure
CPU, finishes in well under a minute.

WHAT IT SHOWS
-------------
C latent classes each own a random prototype vector in input space. An
"augmented view" of a class is its prototype plus Gaussian noise. For each
training step we draw:
  - q   : a view of class c            (the query)
  - k_0 : ANOTHER view of class c      (the positive)
  - k_1 .. k_{K-1} : views of classes <> c   (the negatives)
The shared encoder maps every view to a d-dim unit-norm embedding. InfoNCE
trains the encoder so the query's embedding is closer (higher cosine) to its
positive than to any negative. We report, BEFORE and AFTER training:
  * mean positive-pair cosine and mean negative-pair cosine -> the GAP widens;
  * Wang & Isola (2020) alignment = mean ||z_q - z_k+||^2 over positive pairs
    (should DECREASE: positives collapse together);
  * Wang & Isola (2020) uniformity = log mean exp(-2 ||z_i - z_j||^2) over
    negative pairs (more negative = embeddings spread more uniformly);
  * the InfoNCE loss itself (should DECREASE).
At the end three correctness signals are asserted:
  (1) pos-vs-neg cosine gap is strictly larger after training than before,
  (2) final InfoNCE loss < initial loss,
  (3) alignment after < alignment before.

HOW THE InfoNCE HEAD IS WIRED (weight-shared encoder, fully native)
-------------------------------------------------------------------
TNNetInfoNCELoss has NO external target: supervision is implicit in the input
DEPTH layout, exactly like TNNetTripletLoss. Construct it with
TNNetInfoNCELoss.Create(EmbeddingDim, Temperature). It requires the input depth
to be divisible by EmbeddingDim d, splitting it into NumSlabs = Depth div d
slabs (>= 3 required: 1 query + at least 2 keys). Per spatial cell the head
reads its OWN FOutput as:
    slab 0          = q     (query embedding, channels 0..d-1)
    slab 1          = k_0   (THE POSITIVE,    channels d..2d-1)
    slab 2..K       = k_1..k_{K-1} (the NEGATIVES)
It forms similarities s_j = <q, k_j>/tau, and the loss is the softmax
cross-entropy that selects the positive (slab 1) among all keys:
    L = -s_0 + logsumexp_j(s_j)
The forward pass is a pure identity passthrough (so Net.Compute returns the raw
packed embeddings); Backpropagate writes the analytic InfoNCE gradient into
FOutputError and seeds it itself -- no external target, no manual gradient
surgery.

To feed that layout we use ONE network that embeds all K+1 views at once:

  Input(SizeX=K+1, SizeY=1, Depth=cInDim)   <- K+1 views at X=0..K; raw coords
  PointwiseConvReLU / PointwiseConvLinear    <- featuresize=1 => the SAME weights
                                                are applied at every X position
                                                => a genuine SHARED encoder MLP
  L2Normalize (per-(x,y)-over-depth)         <- each view's embedding on unit
                                                sphere (so dot product = cosine)
  -> output shape (K+1, 1, d)
  Reshape(1, 1, (K+1)*d)                      <- pure reinterpretation; because
                                                volumes are depth-major
                                                (pos = ((SizeX*y)+x)*Depth + d)
                                                the per-X embeddings land as
                                                consecutive depth chunks, i.e.
                                                exactly q | k_0 | k_1 | .. layout
  TNNetInfoNCELoss(d, tau)                     <- consumes the packed layout

CONTRAST WITH THE OTHER EMBEDDING EXAMPLES
------------------------------------------
- examples/TripletEmbedding uses TNNetTripletLoss: a MARGIN/hinge loss over a
  SINGLE (anchor, positive, negative) triplet,
      L = max(0, ||a-p||^2 - ||a-n||^2 + margin),
  i.e. exactly ONE negative and a hard margin. InfoNCE instead contrasts the
  positive against K negatives AT ONCE via a temperature-scaled softmax
  (a "soft", multi-negative generalization; no margin, a temperature tau).
- A TNNetCosineEmbeddingLoss-style head scores ONE pair as similar/dissimilar
  with a per-pair target label. InfoNCE needs no labels at all -- the positive
  is fixed by position (slab 1) and the contrast set is the other slabs.
More negatives generally give a tighter, more informative contrastive signal,
which is why InfoNCE (van den Oord et al. 2018; SimCLR, MoCo) underpins modern
self-supervised representation learning.

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
  Classes, SysUtils,
  neuralnetwork,
  neuralvolume;

const
  cNumClasses = 6;      // C latent classes
  cInDim      = 8;      // raw input dimensionality of a "view"
  cEmbedDim   = 4;      // d: embedding dim on the unit sphere
  cNumNeg     = 4;      // K-1 negatives; total keys K = 1 positive + cNumNeg
  cNumKeys    = cNumNeg + 1;       // K = keys (positive + negatives)
  cNumSlabs   = cNumKeys + 1;      // 1 query + K keys
  cPackedDim  = cNumSlabs * cEmbedDim;
  cTau        = 0.2;    // InfoNCE temperature
  cEpochs     = 120;
  cStepsEp    = 256;    // contrastive steps per epoch
  cLearnRate  = 0.05;
  cSeed       = 12345;
  cSigma      = 0.35;   // augmentation noise on a class prototype
  cEvalPairs  = 400;    // sample size for the before/after metrics

type
  TProto = array[0..cInDim - 1] of TNeuralFloat;
  TProtoArray = array[0..cNumClasses - 1] of TProto;

var
  Protos: TProtoArray;

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

// Random unit-spread prototypes, one per latent class.
procedure BuildProtos();
var
  C, D: integer;
begin
  for C := 0 to cNumClasses - 1 do
    for D := 0 to cInDim - 1 do
      Protos[C][D] := RandomGauss();
end;

// An augmented "view" of class C: prototype + Gaussian noise, into V at X=Slab.
procedure FillView(V: TNNetVolume; Slab, C: integer);
var
  D: integer;
begin
  for D := 0 to cInDim - 1 do
    V[Slab, 0, D] := Protos[C][D] + RandomGauss() * cSigma;
end;

// Picks a class different from Avoid.
function OtherClass(Avoid: integer): integer;
begin
  repeat
    Result := Random(cNumClasses);
  until Result <> Avoid;
end;

// Fills one contrastive input sample: slab 0 = query (class c), slab 1 = its
// positive (another view of class c), slabs 2..K = negatives (other classes).
procedure FillSample(V: TNNetVolume; QueryCls: integer);
var
  J: integer;
begin
  V.Fill(0);
  FillView(V, 0, QueryCls);          // query
  FillView(V, 1, QueryCls);          // positive: same class, fresh noise
  for J := 0 to cNumNeg - 1 do
    FillView(V, 2 + J, OtherClass(QueryCls));   // negatives
end;

// Builds the weight-shared encoder + InfoNCE-loss network.
function BuildNet(): TNNet;
begin
  Result := TNNet.Create();
  Result.AddLayer(TNNetInput.Create(cNumSlabs, 1, cInDim));
  // Shared MLP over the cNumSlabs X positions (pointwise = same weights/pos).
  Result.AddLayer(TNNetPointwiseConvReLU.Create(16));
  Result.AddLayer(TNNetPointwiseConvReLU.Create(16));
  Result.AddLayer(TNNetPointwiseConvLinear.Create(cEmbedDim));
  // Each view's embedding onto the unit sphere (per-(x,y) over depth).
  Result.AddLayer(TNNetL2Normalize.Create());
  // Reinterpret (cNumSlabs,1,d) as (1,1,cNumSlabs*d) = q|k_0|k_1|.. layout.
  Result.AddLayer(TNNetReshape.Create(1, 1, cPackedDim));
  Result.AddLayer(TNNetInfoNCELoss.Create(cEmbedDim, cTau));
  Result.InitWeights();
end;

// Reads slab Slab's d-dim (unit-norm) embedding out of the packed output.
procedure ReadSlab(NN: TNNet; Slab: integer; out E: array of TNeuralFloat);
var
  D: integer;
begin
  for D := 0 to cEmbedDim - 1 do
    E[D] := NN.GetLastLayer().Output.FData[Slab * cEmbedDim + D];
end;

function Dot(const A, B: array of TNeuralFloat): TNeuralFloat;
var
  D: integer;
begin
  Result := 0;
  for D := 0 to cEmbedDim - 1 do
    Result := Result + A[D] * B[D];
end;

function SqDist(const A, B: array of TNeuralFloat): TNeuralFloat;
var
  D: integer;
  Diff: TNeuralFloat;
begin
  Result := 0;
  for D := 0 to cEmbedDim - 1 do
  begin
    Diff := A[D] - B[D];
    Result := Result + Diff * Diff;
  end;
end;

// InfoNCE loss for one packed output: L = -s_0 + logsumexp_j(<q,k_j>/tau).
function InfoNCELossOf(NN: TNNet): TNeuralFloat;
var
  Q, Kj: array[0..cEmbedDim - 1] of TNeuralFloat;
  Sims: array[0..cNumKeys - 1] of TNeuralFloat;
  J: integer;
  MaxS, SumExp: TNeuralFloat;
begin
  ReadSlab(NN, 0, Q);
  for J := 0 to cNumKeys - 1 do
  begin
    ReadSlab(NN, 1 + J, Kj);
    Sims[J] := Dot(Q, Kj) / cTau;
  end;
  MaxS := Sims[0];
  for J := 1 to cNumKeys - 1 do
    if Sims[J] > MaxS then MaxS := Sims[J];
  SumExp := 0;
  for J := 0 to cNumKeys - 1 do
    SumExp := SumExp + Exp(Sims[J] - MaxS);
  Result := -Sims[0] + (MaxS + Ln(SumExp));
end;

// Trains one epoch; returns the mean InfoNCE loss over the steps.
function TrainEpoch(NN: TNNet; Input, Dummy: TNNetVolume): TNeuralFloat;
var
  Step, C: integer;
  Total: TNeuralFloat;
begin
  Total := 0;
  for Step := 0 to cStepsEp - 1 do
  begin
    C := Random(cNumClasses);
    FillSample(Input, C);
    NN.Compute(Input);
    Total := Total + InfoNCELossOf(NN);
    // InfoNCE seeds its own gradient; Dummy is just a shape placeholder.
    NN.Backpropagate(Dummy);
  end;
  Result := Total / cStepsEp;
end;

// Computes the report metrics over cEvalPairs freshly drawn samples:
//   PosCos  : mean cosine(query, positive)
//   NegCos  : mean cosine(query, negative) over all K-1 negatives
//   Align   : mean ||z_q - z_k+||^2          (Wang & Isola 2020 alignment)
//   Unif    : log mean exp(-2 ||z_q - z_neg||^2)  (W&I uniformity)
//   Loss    : mean InfoNCE loss
procedure Evaluate(NN: TNNet; Input: TNNetVolume;
  out PosCos, NegCos, Align, Unif, Loss: TNeuralFloat);
var
  Q, Kp, Kn: array[0..cEmbedDim - 1] of TNeuralFloat;
  S, J, C: integer;
  SumPosCos, SumNegCos, SumAlign, SumUnifExp, SumLoss: TNeuralFloat;
  NegCount: integer;
begin
  SumPosCos := 0; SumNegCos := 0; SumAlign := 0; SumUnifExp := 0; SumLoss := 0;
  NegCount := 0;
  for S := 0 to cEvalPairs - 1 do
  begin
    C := Random(cNumClasses);
    FillSample(Input, C);
    NN.Compute(Input);
    SumLoss := SumLoss + InfoNCELossOf(NN);
    ReadSlab(NN, 0, Q);          // query
    ReadSlab(NN, 1, Kp);         // positive (slab 1)
    SumPosCos := SumPosCos + Dot(Q, Kp);
    SumAlign  := SumAlign + SqDist(Q, Kp);
    for J := 0 to cNumNeg - 1 do
    begin
      ReadSlab(NN, 2 + J, Kn);   // negatives (slabs 2..K)
      SumNegCos := SumNegCos + Dot(Q, Kn);
      SumUnifExp := SumUnifExp + Exp(-2 * SqDist(Q, Kn));
      Inc(NegCount);
    end;
  end;
  PosCos := SumPosCos / cEvalPairs;
  NegCos := SumNegCos / NegCount;
  Align  := SumAlign  / cEvalPairs;
  Unif   := Ln(SumUnifExp / NegCount);
  Loss   := SumLoss   / cEvalPairs;
end;

procedure RunAlgo();
var
  NN: TNNet;
  Input, Dummy: TNNetVolume;
  Epoch: integer;
  MeanLoss: TNeuralFloat;
  PreP, PreN, PreA, PreU, PreL: TNeuralFloat;
  PostP, PostN, PostA, PostU, PostL: TNeuralFloat;
  PreGap, PostGap: TNeuralFloat;
  OkGap, OkLoss, OkAlign: boolean;
begin
  RandSeed := cSeed;
  // Determinism: fixed seed. The raw Compute/Backpropagate path runs one
  // sample at a time (no batch thread pool), so this run is single-threaded.
  WriteLn('InfoNCEContrastive: contrastive embedding learning via TNNetInfoNCELoss');
  WriteLn(Format('Classes: %d  in_dim: %d  embed_dim: %d  negatives(K-1): %d  tau: %.2f',
    [cNumClasses, cInDim, cEmbedDim, cNumNeg, cTau]));
  WriteLn(Format('Packed input layout: slabs=%d (q | k_0=positive | k_1..k_%d=negatives), packed_dim=%d',
    [cNumSlabs, cNumNeg, cPackedDim]));
  WriteLn;

  BuildProtos();
  NN := BuildNet();
  Input := TNNetVolume.Create(cNumSlabs, 1, cInDim);
  // InfoNCE ignores the target; the dummy matches the packed output shape.
  Dummy := TNNetVolume.Create(1, 1, cPackedDim);
  Dummy.Fill(0);
  NN.SetLearningRate(cLearnRate, 0.9);
  try
    Evaluate(NN, Input, PreP, PreN, PreA, PreU, PreL);
    WriteLn('BEFORE training:');
    WriteLn(Format('  mean pos cosine = %8.4f   mean neg cosine = %8.4f   gap = %8.4f',
      [PreP, PreN, PreP - PreN]));
    WriteLn(Format('  alignment       = %8.4f   uniformity      = %8.4f   InfoNCE loss = %8.4f',
      [PreA, PreU, PreL]));
    WriteLn;

    WriteLn('Training for ', cEpochs, ' epochs (', cStepsEp, ' steps each)...');
    for Epoch := 1 to cEpochs do
    begin
      MeanLoss := TrainEpoch(NN, Input, Dummy);
      if (Epoch = 1) or (Epoch mod 20 = 0) then
        WriteLn(Format('  epoch %4d   mean_InfoNCE=%8.5f', [Epoch, MeanLoss]));
    end;
    WriteLn;

    Evaluate(NN, Input, PostP, PostN, PostA, PostU, PostL);
    WriteLn('AFTER training:');
    WriteLn(Format('  mean pos cosine = %8.4f   mean neg cosine = %8.4f   gap = %8.4f',
      [PostP, PostN, PostP - PostN]));
    WriteLn(Format('  alignment       = %8.4f   uniformity      = %8.4f   InfoNCE loss = %8.4f',
      [PostA, PostU, PostL]));
    WriteLn;

    PreGap  := PreP  - PreN;
    PostGap := PostP - PostN;
    WriteLn('Summary (Wang & Isola 2020 alignment/uniformity):');
    WriteLn(Format('  pos-vs-neg cosine gap : %.4f -> %.4f   (wider is better)',
      [PreGap, PostGap]));
    WriteLn(Format('  alignment             : %.4f -> %.4f   (lower is better)',
      [PreA, PostA]));
    WriteLn(Format('  uniformity            : %.4f -> %.4f',
      [PreU, PostU]));
    WriteLn(Format('  InfoNCE loss          : %.4f -> %.4f   (lower is better)',
      [PreL, PostL]));
    WriteLn;

    // Correctness signals.
    OkGap   := PostGap > PreGap;
    OkLoss  := PostL < PreL;
    OkAlign := PostA < PreA;
    WriteLn('Correctness signals:');
    WriteLn('  [', BoolToStr(OkGap, 'PASS', 'FAIL'),
      '] pos-vs-neg cosine gap widened (positives pulled together, negatives pushed apart)');
    WriteLn('  [', BoolToStr(OkLoss, 'PASS', 'FAIL'),
      '] final InfoNCE loss < initial InfoNCE loss');
    WriteLn('  [', BoolToStr(OkAlign, 'PASS', 'FAIL'),
      '] alignment decreased (positive pairs collapsed closer)');

    if not (OkGap and OkLoss and OkAlign) then
    begin
      WriteLn;
      WriteLn('ERROR: one or more correctness signals FAILED.');
      Halt(1);
    end;
    WriteLn;
    WriteLn('All correctness signals PASSED.');
  finally
    Dummy.Free;
    Input.Free;
    NN.Free;
  end;
end;

var
  // Stops Lazarus errors
  Application: record Title: string; end;

begin
  Application.Title := 'InfoNCEContrastive Example';
  RunAlgo();
end.
