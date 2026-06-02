program MagnitudePruning;
(*
MagnitudePruning: demonstrates TNNet.MagnitudePruningReport, the forward-only
NO-RETRAIN compressibility diagnostic. It snapshots a trained classifier, then
for each global sparsity level s in {0,10,...,90,95,99}% it zeros the smallest
s% of |weights| ACROSS the whole network, runs ONE forward pass over a probe
batch to read top-1 accuracy and loss, and restores the weights bit-for-bit
before the next level. The accuracy-vs-sparsity curve shows how much of the
model you can throw away before it breaks — without any retraining.

The story, in one run, on the SAME synthetic 3-class problem:
  (i)   an OVER-WIDE net (lots of redundant capacity) should stay flat to high
        sparsity — "highly-compressible";
  (ii)  a TIGHT-FIT net (just enough capacity) loses accuracy early — "fragile";
  (iii) the over-wide net again, but with the optional PerLayer flag, so the
        GLOBAL-vs-UNIFORM(per-layer) pruning criteria sit side by side.

Synthetic, self-contained (no dataset download), pure CPU, well under a minute.

Copyright (C) 2026 Joao Paulo Schwarz Schuler

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
any later version.

Coded by Claude (AI).
*)

{$mode objfpc}{$H+}

uses {$IFDEF UNIX} {$IFDEF UseCThreads}
  cthreads, {$ENDIF} {$ENDIF}
  Classes, SysUtils,
  neuralnetwork,
  neuralvolume;

const
  cInDim    = 12;   // input feature dimension
  cClasses  = 3;    // number of classes
  cProbeCnt = 96;   // synthetic probe samples (held-out)
  cTrainCnt = 600;  // synthetic training samples
  cEpochs   = 60;

// Build a linearly-separable-ish synthetic 3-class problem: each class has a
// random prototype; a sample is its class prototype + noise. argmax over the
// 3 class scores is the label.
procedure MakeSample(const Proto: array of TNeuralFloat;
  out X: TNNetVolume; out Cls: integer);
var
  I, C: integer;
  Score, Best: TNeuralFloat;
begin
  Cls := Random(cClasses);
  X := TNNetVolume.Create(cInDim, 1, 1);
  for I := 0 to cInDim - 1 do
    X.Raw[I] := Proto[Cls * cInDim + I] + (Random - 0.5) * 1.2;
  // Recompute the true label as argmax of (proto_c . x) so it is consistent
  // with the actual features (robust to noise flipping the class).
  Best := -1e30;
  for C := 0 to cClasses - 1 do
  begin
    Score := 0;
    for I := 0 to cInDim - 1 do
      Score := Score + Proto[C * cInDim + I] * X.Raw[I];
    if Score > Best then begin Best := Score; Cls := C; end;
  end;
end;

procedure RunCase(const Title: string; Hidden: integer; PerLayer: boolean;
  const Proto: array of TNeuralFloat);
var
  NN: TNNet;
  Probes, Labels: TNNetVolumeList;
  X, Yt, L: TNNetVolume;
  Ep, Step, Cls, Correct, K: integer;
  Acc: TNeuralFloat;
begin
  WriteLn(StringOfChar('=', 92));
  WriteLn(Title);
  WriteLn(StringOfChar('=', 92));

  NN := TNNet.Create();
  Probes := TNNetVolumeList.Create(True);
  Labels := TNNetVolumeList.Create(True);
  try
    NN.AddLayer(TNNetInput.Create(cInDim, 1, 1));
    NN.AddLayer(TNNetFullConnectReLU.Create(Hidden));
    NN.AddLayer(TNNetFullConnectReLU.Create(Hidden));
    NN.AddLayer(TNNetFullConnectLinear.Create(cClasses));
    NN.AddLayer(TNNetSoftMax.Create());
    NN.SetLearningRate(0.02, 0.9);
    NN.InitWeights();

    // train.
    for Ep := 1 to cEpochs do
      for Step := 1 to cTrainCnt do
      begin
        MakeSample(Proto, X, Cls);
        Yt := TNNetVolume.Create(cClasses, 1, 1);
        Yt.Raw[Cls] := 1.0;
        try
          NN.Compute(X);
          NN.Backpropagate(Yt);
        finally
          X.Free;
          Yt.Free;
        end;
      end;

    // held-out probe batch + one-hot labels.
    for K := 0 to cProbeCnt - 1 do
    begin
      MakeSample(Proto, X, Cls);
      Probes.Add(X);
      L := TNNetVolume.Create(cClasses, 1, 1);
      L.Raw[Cls] := 1.0;
      Labels.Add(L);
    end;

    // report trained accuracy on the probe batch for context.
    Correct := 0;
    for K := 0 to Probes.Count - 1 do
    begin
      NN.Compute(Probes[K]);
      if NN.GetLastLayer.Output.GetClass() = Labels[K].GetClass() then
        Inc(Correct);
    end;
    Acc := Correct / Probes.Count;
    WriteLn(Format('  Trained net: 2 hidden layers of width %d, ' +
      'probe top-1 = %.2f%%.', [Hidden, Acc * 100.0]));
    WriteLn;

    Write(TNNet.MagnitudePruningReport(NN, Probes, Labels, 0.01, PerLayer));
  finally
    Labels.Free;
    Probes.Free;
    NN.Free;
  end;
end;

var
  Proto: array of TNeuralFloat;
  I: integer;
begin
  RandSeed := 2026;

  WriteLn('MagnitudePruningReport demo: how much of a trained classifier can ' +
    'you zero by magnitude');
  WriteLn('before it breaks -- measured by ACTUALLY pruning and re-running, ' +
    'no retraining.');
  WriteLn('Same 3-class synthetic problem; an over-wide net should stay flat ' +
    'to high sparsity while');
  WriteLn('a tight-fit net falls early. The third run repeats the over-wide ' +
    'net with the per-layer');
  WriteLn('(uniform) criterion so global-vs-uniform pruning is visible side ' +
    'by side.');
  WriteLn;

  // Shared random class prototypes so all three cases solve the SAME problem.
  SetLength(Proto, cClasses * cInDim);
  for I := 0 to Length(Proto) - 1 do Proto[I] := (Random - 0.5) * 2.0;

  RunCase('(i) OVER-WIDE net (width 64): redundant capacity -> ' +
    'expect highly-compressible.', 64, False, Proto);
  WriteLn;
  RunCase('(ii) TIGHT-FIT net (width 6): just enough capacity -> ' +
    'expect fragile (accuracy falls early).', 6, False, Proto);
  WriteLn;
  RunCase('(iii) OVER-WIDE net (width 64) with PER-LAYER (uniform) ' +
    'criterion -- contrast vs run (i).', 64, True, Proto);

  WriteLn;
  WriteLn('Read the curves: the over-wide net holds accuracy to a much ' +
    'deeper sparsity (higher knee)');
  WriteLn('than the tight-fit net -- over-parameterised models are ' +
    'compressible. Compare runs (i) and');
  WriteLn('(iii) to see how the global pooled threshold differs from a ' +
    'uniform per-layer percentile.');
end.
