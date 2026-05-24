program MCDropoutUncertainty;
(*
MCDropoutUncertainty: builds a small dropout MLP classifier on a SYNTHETIC
3-cluster 2D dataset, trains it briefly (seconds), then prints
TNNet.MCDropoutUncertaintyReport. Monte-Carlo dropout keeps the TNNetDropout
layer ACTIVE at inference and runs many stochastic forward passes per probe
to separate epistemic (model) from aleatoric (data) uncertainty.

Three probe groups are evaluated:
  (1) the three CLUSTER CORES (in-distribution): the model is confident and
      every MC pass agrees -> epistemic BALD ~ 0;
  (2) an out-of-distribution (OOD) BAND placed in the empty space BETWEEN the
      clusters, far from any training point: MC passes disagree -> high BALD;
  (3) a labelled validation split feeding the correctness cross-tab.

"The model knows what it doesn't know": the OOD band lights up with high BALD
while the cluster cores read near zero.

Pure CPU, SYNTHETIC data (no download), well under a minute.

Copyright (C) 2026 Joao Paulo Schwarz Schuler

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
any later version.
*)

{$mode objfpc}{$H+}

uses {$IFDEF UNIX} {$IFDEF UseCThreads}
  cthreads, {$ENDIF} {$ENDIF}
  Classes, SysUtils, Math,
  neuralnetwork,
  neuralvolume;

const
  cClasses   = 3;
  cHidden    = 32;
  cEpochs    = 300;
  cBatch     = 48;
  cDropout   = 0.25;      // moderate dropout: peaked cores, noisy OOD
  cTrainPerC = 200;       // training points per cluster
  cCoreProbe = 12;        // probe points per cluster core
  cOODProbe  = 24;        // probe points along the OOD band
  cValPerC   = 26;        // clean validation points per cluster (cross-tab)
  cValHard   = 6;         // hard boundary points per cluster (some misclassd)

// Cluster centres: a wide triangle, leaving a large empty centre for the OOD
// band to sit in.
procedure ClusterCentre(C: integer; out cx, cy: TNeuralFloat);
begin
  case C of
    0: begin cx := -3.0; cy := -2.0; end;
    1: begin cx :=  3.0; cy := -2.0; end;
  else begin cx :=  0.0; cy :=  3.2; end;
  end;
end;

procedure MakeInput(out V: TNNetVolume; x, y: TNeuralFloat);
begin
  V := TNNetVolume.Create(2, 1, 1);
  V.Raw[0] := x;
  V.Raw[1] := y;
end;

procedure MakeClusterPoint(C: integer; out V: TNNetVolume);
var
  cx, cy: TNeuralFloat;
begin
  ClusterCentre(C, cx, cy);
  MakeInput(V, cx + (Random - 0.5) * 0.7, cy + (Random - 0.5) * 0.7);
end;

procedure BuildNet(out NN: TNNet);
begin
  NN := TNNet.Create();
  NN.AddLayer(TNNetInput.Create(2, 1, 1));
  NN.AddLayer(TNNetFullConnectReLU.Create(cHidden));
  NN.AddLayer(TNNetDropout.Create(cDropout));   // <-- the MC-dropout layer
  NN.AddLayer(TNNetFullConnectReLU.Create(cHidden));
  NN.AddLayer(TNNetFullConnectLinear.Create(cClasses));
  NN.AddLayer(TNNetSoftMax.Create());
  NN.SetLearningRate(0.02, 0.9);
end;

procedure Train(NN: TNNet; Epochs: integer);
var
  Ep, B, I, C: integer;
  X, Yt: TNNetVolume;
  Out0: TNNetVolume;
  TotalLoss, Diff: TNeuralFloat;
begin
  for Ep := 1 to Epochs do
  begin
    TotalLoss := 0;
    for B := 1 to cBatch do
    begin
      C := Random(cClasses);
      MakeClusterPoint(C, X);
      Yt := TNNetVolume.Create(cClasses, 1, 1);
      Yt.Raw[C] := 1;
      try
        NN.Compute(X);
        Out0 := NN.GetLastLayer.Output;
        for I := 0 to Out0.Size - 1 do
        begin
          Diff := Out0.Raw[I] - Yt.Raw[I];
          TotalLoss := TotalLoss + Diff * Diff;
        end;
        NN.Backpropagate(Yt);
      finally
        X.Free;
        Yt.Free;
      end;
    end;
    if (Ep = 1) or (Ep mod 40 = 0) or (Ep = Epochs) then
      WriteLn(Format('  epoch %3d  mean-MSE=%.6f', [Ep, TotalLoss / cBatch]));
  end;
end;

var
  NN: TNNet;
  Cores, OOD, Val: TNNetVolumeList;
  ValLabels: array of integer;
  V: TNNetVolume;
  C, K: integer;
  t0, t1: TDateTime;
  tx, cx, cy, nx, ny: TNeuralFloat;
begin
  RandSeed := 2026;
  t0 := Now();

  // Build probe groups.
  Cores := TNNetVolumeList.Create(True);
  OOD := TNNetVolumeList.Create(True);
  Val := TNNetVolumeList.Create(True);
  try
    // (1) cluster cores: tight around each centre (in-distribution).
    for C := 0 to cClasses - 1 do
      for K := 0 to cCoreProbe - 1 do
      begin
        MakeClusterPoint(C, V);
        Cores.Add(V);
      end;

    // (2) OOD band: a horizontal sweep through the empty centre (0,0) area,
    // far from every cluster core.
    for K := 0 to cOODProbe - 1 do
    begin
      tx := -2.0 + 4.0 * (K / (cOODProbe - 1));   // x in [-2, 2]
      MakeInput(V, tx, 0.4);                       // y=0.4, dead centre
      OOD.Add(V);
    end;

    // (3) validation split (labelled) for the correctness cross-tab: mostly
    // clean cluster points plus a handful of HARD boundary points pulled
    // halfway toward a neighbouring cluster (still labelled with their home
    // class) so the model misclassifies a few — that makes the
    // correct-vs-incorrect entropy comparison meaningful.
    SetLength(ValLabels, cClasses * (cValPerC + cValHard));
    for C := 0 to cClasses - 1 do
    begin
      for K := 0 to cValPerC - 1 do
      begin
        MakeClusterPoint(C, V);
        ValLabels[Val.Count] := C;
        Val.Add(V);
      end;
      for K := 0 to cValHard - 1 do
      begin
        ClusterCentre(C, cx, cy);
        ClusterCentre((C + 1) mod cClasses, nx, ny);
        // 60% of the way toward the neighbour: deep in contested territory.
        MakeInput(V, cx + 0.60 * (nx - cx) + (Random - 0.5) * 0.4,
                     cy + 0.60 * (ny - cy) + (Random - 0.5) * 0.4);
        ValLabels[Val.Count] := C;
        Val.Add(V);
      end;
    end;

    BuildNet(NN);
    try
      WriteLn('MCDropoutUncertainty demo: dropout MLP on a synthetic 3-cluster',
        ' 2D classifier.');
      WriteLn('Training for ', cEpochs, ' epochs of batch size ', cBatch,
        ' (dropout active during training)...');
      Train(NN, cEpochs);

      WriteLn;
      WriteLn(StringOfChar('=', 78));
      WriteLn('GROUP 1: cluster cores (in-distribution). Expect LOW BALD.');
      WriteLn(StringOfChar('=', 78));
      Write(TNNet.MCDropoutUncertaintyReport(NN, Cores, 40, 1.0, 5));

      WriteLn;
      WriteLn(StringOfChar('=', 78));
      WriteLn('GROUP 2: OOD band between clusters. Expect HIGH BALD.');
      WriteLn(StringOfChar('=', 78));
      Write(TNNet.MCDropoutUncertaintyReport(NN, OOD, 40, 1.0, 5));

      WriteLn;
      WriteLn(StringOfChar('=', 78));
      WriteLn('GROUP 3: labelled validation split (correctness cross-tab).');
      WriteLn(StringOfChar('=', 78));
      Write(TNNet.MCDropoutUncertaintyReport(NN, Val, ValLabels, 40, 1.0, 5));

      WriteLn;
      WriteLn('Expect: cluster cores read near-zero BALD (the model is sure ',
        'and every MC pass agrees), while the OOD band reads a much higher ',
        'mean BALD and high argmax flip rate — "the model knows what it ',
        'does not know".');
    finally
      NN.Free;
    end;
  finally
    Val.Free;
    OOD.Free;
    Cores.Free;
  end;

  t1 := Now();
  WriteLn(Format('Total runtime: %.2f s.', [(t1 - t0) * 24 * 3600]));
end.
