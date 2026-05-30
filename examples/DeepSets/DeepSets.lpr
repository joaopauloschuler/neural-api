program DeepSets;
(*
DeepSets: a permutation-invariant set-learning demo (Zaheer et al. 2017,
"Deep Sets").

THE IDEA. A function defined on a SET must not care about the order of its
elements: f({a,b,c}) = f({c,a,b}). Zaheer et al. show that any permutation-
invariant set function can be written as

    f(X) = rho( POOL_{x in X} phi(x) )

where phi is a SHARED per-element encoder, POOL is a symmetric reduction
(sum / mean / max), and rho is a head applied to the pooled summary. The
invariance is STRUCTURAL: it holds for every weight setting, before and after
training, because a symmetric pool is by definition order-agnostic.

This demo learns to regress the MAX of a fixed-size bag of N scalars and then
makes the invariance the centerpiece: after training, permuting the N inputs
leaves the output bit-for-bit unchanged, while editing an element's value
changes it.

ARCHITECTURE (existing layers only).
  Input(N, 1, 1)                  -- N set elements laid along the X axis
  TNNetPointwiseConvReLU(H)       -- SHARED phi: featuresize-1 conv, identical
  TNNetConvolutionLinear(H,1,0,1) --   weights applied to every element
  TNNetMaxChannel                 -- symmetric POOL: (N,1,H) -> (1,1,H)
  TNNetFullConnectReLU(H)         -- rho head
  TNNetFullConnectLinear(1)       -- scalar prediction

A pointwise (featuresize=1) conv is the natural "shared MLP over elements":
the SAME H-wide filter bank slides over the X axis, so element k is encoded by
exactly the same weights as element 0. TNNetMaxChannel then collapses the X
axis to a single value per channel. Because phi is shared and the pool is
symmetric, the whole network is permutation invariant by construction.

WHY POOL, NOT FLATTEN+DENSE. A plain flatten -> dense net assigns a distinct
weight to "the element in slot 0", "the element in slot 1", ... so reordering
the inputs lands different values on different weights and the output moves.
It also hard-codes N: it literally cannot accept a bag of a different size.
The stretch test below trains on N=5 and evaluates on N=8 -- the Deep Sets net
just works (the pool accepts any X width), while a flatten+dense baseline could
not even be fed the larger input.

WHY NOT SELF-ATTENTION. Self-attention is also permutation EQUIVARIANT (a
permutation of the inputs permutes the outputs the same way) and, with a final
symmetric pool, permutation INVARIANT. But it costs O(N^2) pairwise scores and
a stack of projections. Deep Sets gets the same invariance for O(N) with a
single shared encoder and one pooling op -- the cheapest member of the family,
and all this demo needs.

FEASIBILITY NOTE on TNNetMaxChannel / TNNetAvgChannel (settled before building
this). For an (N,1,F) input both set FPoolSize := SizeX = N and stride = N, so
they reduce (N,1,F) -> (1,1,F), one number per channel:
  * TNNetMaxChannel returns the exact per-channel MAX over the N elements.
  * TNNetAvgChannel returns sum/(FPoolSize*FPoolSize) = sum/N^2, i.e. a SCALED
    mean (1/N^2, not 1/N), because the avg-pool divides by PoolSize^2 while
    only N (= PoolSize * 1) cells are non-empty along Y=1. Still perfectly
    symmetric, so still permutation invariant; only the scale differs, which a
    linear rho head trivially absorbs.
We pick MAX-pool + a MAX target so the pool computes EXACTLY the symmetric
statistic we are regressing -- the cleanest, fastest-learning pairing, with
mathematically exact invariance.

Pure CPU, single-threaded, deterministic (fixed RandSeed). Runs in a few
seconds. Self-checking gates Halt(1) on failure, mirroring the suite idiom.

Copyright (C) 2026 Joao Paulo Schwarz Schuler

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
*)

{$mode objfpc}{$H+}

uses {$IFDEF UNIX} {$IFDEF UseCThreads}
  cthreads, {$ENDIF} {$ENDIF}
  Classes, SysUtils, Math,
  neuralnetwork,
  neuralvolume;

const
  cTrainN   = 5;      // set size at training time
  cTestN    = 8;      // larger UNSEEN set size for the generalization stretch
  cHidden   = 16;     // width of the shared encoder phi and the rho head
  cNumTrain = 512;    // synthetic training bags
  cEpochs   = 60;     // epoch budget (MAX target learns fast)
  cBatch    = 16;
  cLR       = 0.02;
  cMomentum = 0.9;
  cValLo    = -1.0;   // element values drawn from [cValLo, cValHi]
  cValHi    =  1.0;
  cSeed     = 424242; // repo idiom

// Target symmetric statistic of a bag: its MAX.
function BagTarget(V: TNNetVolume): TNeuralFloat;
var
  i: integer;
begin
  Result := V.Raw[0];
  for i := 1 to V.Size - 1 do
    if V.Raw[i] > Result then Result := V.Raw[i];
end;

// Fill a bag (N,1,1) with iid uniform values in [cValLo, cValHi].
procedure FillRandomBag(V: TNNetVolume);
var
  i: integer;
begin
  for i := 0 to V.Size - 1 do
    V.Raw[i] := cValLo + (cValHi - cValLo) * Random;
end;

// Build the Deep Sets net for a given set size N (X-width). Same weights,
// different input width -- this is what lets us train at N=5 and eval at N=8.
function MakeDeepSets(N: integer): TNNet;
begin
  Result := TNNet.Create();
  Result.AddLayer(TNNetInput.Create(N, 1, 1));
  // SHARED per-element encoder phi (featuresize=1 pointwise convs).
  Result.AddLayer(TNNetPointwiseConvReLU.Create(cHidden));
  Result.AddLayer(TNNetConvolutionLinear.Create(cHidden, 1, 0, 1));
  // Symmetric pool: (N,1,cHidden) -> (1,1,cHidden).
  Result.AddLayer(TNNetMaxChannel.Create());
  // rho head.
  Result.AddLayer(TNNetFullConnectReLU.Create(cHidden));
  Result.AddLayer(TNNetFullConnectLinear.Create(1));
end;

// Mini-batch SGD on freshly drawn bags of size N.
procedure Train(NN: TNNet; N: integer);
var
  Epoch, Step, i: integer;
  Inp, Tgt: TNNetVolume;
  Bags: array of TNNetVolume;
  Tgts: array of TNeuralFloat;
begin
  // Pre-draw a fixed training set so every epoch sees the same data.
  SetLength(Bags, cNumTrain);
  SetLength(Tgts, cNumTrain);
  for i := 0 to cNumTrain - 1 do
  begin
    Bags[i] := TNNetVolume.Create(N, 1, 1);
    FillRandomBag(Bags[i]);
    Tgts[i] := BagTarget(Bags[i]);
  end;
  Inp := TNNetVolume.Create(N, 1, 1);
  Tgt := TNNetVolume.Create(1, 1, 1);
  NN.SetLearningRate(cLR, cMomentum);
  NN.SetL2Decay(0.0);
  try
    for Epoch := 1 to cEpochs do
    begin
      Step := 0;
      NN.ClearDeltas();
      for i := 0 to cNumTrain - 1 do
      begin
        Inp.Copy(Bags[i]);
        Tgt.Raw[0] := Tgts[i];
        NN.Compute(Inp);
        NN.Backpropagate(Tgt);
        Inc(Step);
        if Step = cBatch then
        begin
          NN.UpdateWeights();
          NN.ClearDeltas();
          Step := 0;
        end;
      end;
      if Step > 0 then
      begin
        NN.UpdateWeights();
        NN.ClearDeltas();
      end;
    end;
  finally
    Inp.Free;
    Tgt.Free;
    for i := 0 to cNumTrain - 1 do Bags[i].Free;
  end;
end;

// RMSE of NN's MAX prediction over freshly drawn bags of size N.
function EvalRMSE(NN: TNNet; N, Samples: integer): TNeuralFloat;
var
  i: integer;
  Inp, Outp: TNNetVolume;
  Diff, Sum: TNeuralFloat;
begin
  Inp := TNNetVolume.Create(N, 1, 1);
  Outp := TNNetVolume.Create(1, 1, 1);
  Sum := 0;
  for i := 0 to Samples - 1 do
  begin
    FillRandomBag(Inp);
    NN.Compute(Inp);
    NN.GetOutput(Outp);
    Diff := Outp.Raw[0] - BagTarget(Inp);
    Sum := Sum + Diff * Diff;
  end;
  Inp.Free;
  Outp.Free;
  Result := Sqrt(Sum / Samples);
end;

// Compute NN's scalar output for a bag.
function Predict(NN: TNNet; V: TNNetVolume): TNeuralFloat;
var
  Outp: TNNetVolume;
begin
  NN.Compute(V);
  Outp := TNNetVolume.Create(1, 1, 1);
  NN.GetOutput(Outp);
  Result := Outp.Raw[0];
  Outp.Free;
end;

var
  NN, NNWide: TNNet;
  Bag, Shuf: TNNetVolume;
  i, j, k, Tmp: integer;
  Order: array of integer;
  Base, P, MaxDy, Dy, Edited, EditDelta: TNeuralFloat;
  TrainRMSE, TestN5, TestN8: TNeuralFloat;
  Pass: boolean;
const
  cShuffles = 200;
  cInvTol   = 1e-5;     // permutation must move the output by less than this
  cSensTol  = 1e-3;     // editing an element must move it by MORE than this
begin
  // Determinism / fast startup.
  RandSeed := cSeed;
  WriteLn('Deep Sets permutation-invariant set-learning demo (Zaheer et al. 2017)');
  WriteLn('Task: regress MAX of a bag of N=', cTrainN, ' scalars in [',
    cValLo:0:1, ', ', cValHi:0:1, '].');
  WriteLn('Net : Input -> PointwiseConvReLU -> ConvLinear -> MaxChannel -> FC-ReLU -> FC-Linear');
  WriteLn;

  NN := MakeDeepSets(cTrainN);
  try
    NN.InitWeights();
    Train(NN, cTrainN);

    TrainRMSE := EvalRMSE(NN, cTrainN, 2000);
    WriteLn(Format('Trained. RMSE on fresh N=%d bags: %.6f', [cTrainN, TrainRMSE]));
    WriteLn;

    // ----- HEADLINE 1: PERMUTATION INVARIANCE --------------------------------
    // Pick one bag, get the baseline output, then shuffle its elements many
    // times. A permutation-invariant net must return the SAME number every
    // time (up to float noise).
    Bag := TNNetVolume.Create(cTrainN, 1, 1);
    Shuf := TNNetVolume.Create(cTrainN, 1, 1);
    FillRandomBag(Bag);
    Base := Predict(NN, Bag);

    SetLength(Order, cTrainN);
    MaxDy := 0;
    for k := 1 to cShuffles do
    begin
      for i := 0 to cTrainN - 1 do Order[i] := i;
      for i := cTrainN - 1 downto 1 do      // Fisher-Yates shuffle
      begin
        j := Random(i + 1);
        Tmp := Order[i]; Order[i] := Order[j]; Order[j] := Tmp;
      end;
      for i := 0 to cTrainN - 1 do Shuf.Raw[i] := Bag.Raw[Order[i]];
      P := Predict(NN, Shuf);
      Dy := Abs(P - Base);
      if Dy > MaxDy then MaxDy := Dy;
    end;

    WriteLn('HEADLINE 1 - PERMUTATION INVARIANCE');
    WriteLn(Format('  baseline output           : %.8f', [Base]));
    WriteLn(Format('  max |dy| over %d shuffles : %.3e  (tol %.1e)',
      [cShuffles, MaxDy, cInvTol]));

    // ----- HEADLINE 2: VALUE SENSITIVITY -------------------------------------
    // Edit ONE element to a clearly larger value; the MAX, and so the output,
    // must change.
    Edited := Bag.Raw[0];
    Bag.Raw[0] := cValHi + 2.0;   // push element 0 above the whole bag
    EditDelta := Abs(Predict(NN, Bag) - Base);
    Bag.Raw[0] := Edited;         // restore

    WriteLn('HEADLINE 2 - VALUE SENSITIVITY');
    WriteLn(Format('  |dy| after editing one element: %.6f  (tol %.1e)',
      [EditDelta, cSensTol]));
    WriteLn;

    // ----- STRETCH: set-size generalization (train N=5, test N=8) ------------
    // Copy the SAME trained weights into a net built for a WIDER input. This
    // works precisely because none of the trainable layers depend on N: the
    // shared phi is a featuresize-1 conv, and the rho head sees the pooled
    // (1,1,cHidden) summary regardless of how many elements were pooled. A
    // flatten+dense baseline could not even accept this larger bag.
    TestN5 := EvalRMSE(NN, cTrainN, 2000);

    NNWide := MakeDeepSets(cTestN);
    try
      NNWide.InitWeights();
      NNWide.CopyWeights(NN);   // same layer count, N-independent weights
      TestN8 := EvalRMSE(NNWide, cTestN, 2000);
    finally
      NNWide.Free;
    end;
  finally
    Bag.Free;
    Shuf.Free;
  end;
  NN.Free;

  WriteLn('STRETCH - SET-SIZE GENERALIZATION (weights trained ONLY on N=', cTrainN, ')');
  WriteLn(Format('  RMSE on N=%d (train size)   : %.6f', [cTrainN, TestN5]));
  WriteLn(Format('  RMSE on N=%d (UNSEEN size)  : %.6f', [cTestN, TestN8]));
  WriteLn('  (a flatten+dense net cannot even be fed the N=', cTestN, ' bag.)');
  WriteLn;

  // ----- SELF-CHECKING GATES ----------------------------------------------
  Pass := (MaxDy < cInvTol) and (EditDelta > cSensTol) and (TrainRMSE < 0.1);
  if Pass then
    WriteLn('GATE: PASS - invariant to permutation, sensitive to value, and it learned MAX.')
  else
  begin
    WriteLn('GATE: FAIL');
    if not (MaxDy < cInvTol)    then WriteLn('  - permutation moved the output (not invariant).');
    if not (EditDelta > cSensTol) then WriteLn('  - editing an element did not change the output.');
    if not (TrainRMSE < 0.1)    then WriteLn('  - net did not learn the MAX target.');
  end;

  if not Pass then Halt(1);
end.
