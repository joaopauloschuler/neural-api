program PermutationAlign;
(*
PermutationAlign: demonstrates TNNet.PermutationAlignReport, the "Git Re-Basin"
weight-space NEURON-PERMUTATION alignment diagnostic (Ainsworth, Hayase &
Srinivasa 2022; Entezari et al. 2021) — the DUAL of TNNet.ModeConnectivityReport.

ModeConnectivityReport MEASURES the linear-interpolation loss barrier between
two independently-trained nets of the same architecture. PermutationAlignReport
goes one step further: it shows that most of that barrier is an ILLUSION of
neuron-LABELLING. A hidden layer's units are interchangeable up to a
permutation (permute the units AND, in the next layer, the matching
input-weight columns, and the represented FUNCTION is unchanged). After aligning
net B's hidden units to net A's and re-interpolating, the barrier largely
COLLAPSES, because both nets sit in the same basin once you quotient out the
permutation symmetry.

This program trains the SAME tiny MLP twice on a synthetic 3-cluster 2D
classification task, from DIFFERENT random inits (so a real barrier exists
pre-alignment), then prints:

  RUN 1 (weight matching):     align by hidden-unit weight-row cosine.
  RUN 2 (activation matching): align by per-unit activation correlation over
                               the probe batch.
  CHECK (align-to-self):       SnapshotB := A -> identity permutations and a
                               flat zero barrier.

Each run prints the loss barrier BEFORE vs AFTER alignment, the per-layer
permutation churn, and the three built-in PASS/FAIL correctness checks
(permutation invariance, align-to-self, monotonicity). Pure forward-only; the
live net's weights are restored exactly afterwards. Self-contained synthetic
data, runs in well under a minute on CPU.

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
  Classes, SysUtils, Math,
  neuralnetwork,
  neuralvolume;

const
  cEpochs       = 60;
  cTrainPerCls  = 60;
  cProbePerCls  = 12;
  cLearningRate = 0.05;
  cClasses      = 3;
  Centers: array[0..2, 0..1] of TNeuralFloat =
    ((-2.0, -2.0), (2.0, 2.0), (2.0, -2.0));

  procedure BuildNet(out NN: TNNet);
  begin
    NN := TNNet.Create();
    NN.AddLayer(TNNetInput.Create(2, 1, 1));
    NN.AddLayer(TNNetFullConnectReLU.Create(12));
    NN.AddLayer(TNNetFullConnectReLU.Create(12));
    NN.AddLayer(TNNetFullConnectLinear.Create(cClasses));
    NN.SetLearningRate(cLearningRate, 0.9);
  end;

  // Deterministic sample for class C (so two nets can be fed the SAME data,
  // differing only in init / batch order).
  procedure MakeSample(C: integer; out X, Y: TNNetVolume);
  begin
    X := TNNetVolume.Create(2, 1, 1);
    Y := TNNetVolume.Create(cClasses, 1, 1);
    X.FData[0] := Centers[C][0] + (Random - 0.5) * 0.6;
    X.FData[1] := Centers[C][1] + (Random - 0.5) * 0.6;
    Y.Fill(0);
    Y.FData[C] := 1.0;
  end;

  procedure TrainNet(NN: TNNet);
  var
    Epoch, I, C: integer;
    X, Y: TNNetVolume;
  begin
    for Epoch := 1 to cEpochs do
      for I := 1 to cTrainPerCls do
        for C := 0 to cClasses - 1 do
        begin
          MakeSample(C, X, Y);
          try
            NN.Compute(X);
            NN.Backpropagate(Y);
          finally
            X.Free;
            Y.Free;
          end;
        end;
  end;

  procedure BuildProbes(out Probes: TNNetVolumePairList);
  var
    C, I: integer;
    X, Y: TNNetVolume;
  begin
    Probes := TNNetVolumePairList.Create();
    RandSeed := 777;
    for C := 0 to cClasses - 1 do
      for I := 1 to cProbePerCls do
      begin
        MakeSample(C, X, Y);
        Probes.Add(TNNetVolumePair.Create(X, Y));
      end;
  end;

var
  NNA, NNB: TNNet;
  Probes: TNNetVolumePairList;
  SnapB, Report: string;

begin
  WriteLn('=== PermutationAlign demo: Git Re-Basin neuron-permutation alignment ===');
  WriteLn('Two MLPs trained from DIFFERENT inits -> a real barrier pre-alignment');
  WriteLn('that should visibly shrink once the permutation symmetry is removed.');
  WriteLn;

  BuildProbes(Probes);
  try
    // ---------------------------------------------------------------
    // RUN 1: WEIGHT matching (ScoreMode = 0).
    // ---------------------------------------------------------------
    WriteLn('RUN 1 - DIFFERENT inits, WEIGHT matching (align by weight-row cosine).');
    RandSeed := 101;  BuildNet(NNA);
    RandSeed := 999;  BuildNet(NNB);
    RandSeed := 11;   TrainNet(NNA);
    RandSeed := 22;   TrainNet(NNB);

    SnapB := NNB.SaveDataToString();
    Report := TNNet.PermutationAlignReport(NNA, SnapB, Probes, 0, 10);
    WriteLn(Report);
    NNA.Free;
    NNB.Free;
    WriteLn;

    // ---------------------------------------------------------------
    // RUN 2: ACTIVATION matching (ScoreMode = 1).
    // ---------------------------------------------------------------
    WriteLn('RUN 2 - DIFFERENT inits, ACTIVATION matching (align by activation corr).');
    RandSeed := 101;  BuildNet(NNA);
    RandSeed := 999;  BuildNet(NNB);
    RandSeed := 11;   TrainNet(NNA);
    RandSeed := 22;   TrainNet(NNB);

    SnapB := NNB.SaveDataToString();
    Report := TNNet.PermutationAlignReport(NNA, SnapB, Probes, 1, 10);
    WriteLn(Report);
    NNA.Free;
    NNB.Free;
    WriteLn;

    // ---------------------------------------------------------------
    // CHECK: align-to-self (SnapshotB := A) -> identity perms, zero barrier.
    // ---------------------------------------------------------------
    WriteLn('CHECK - align-to-self (SnapshotB := A): identity perms, zero barrier.');
    RandSeed := 303;  BuildNet(NNA);
    RandSeed := 11;   TrainNet(NNA);
    SnapB := NNA.SaveDataToString();   // B == A
    Report := TNNet.PermutationAlignReport(NNA, SnapB, Probes, 0, 8);
    WriteLn(Report);
    NNA.Free;
  finally
    Probes.Free;
  end;
end.
