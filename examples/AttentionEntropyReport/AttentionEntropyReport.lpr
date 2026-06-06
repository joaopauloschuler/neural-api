program AttentionEntropyReport;
(*
AttentionEntropyReport: builds a tiny attention model (one TNNetInput +
linear projection to Q|K|V + two stacked TNNetScaledDotProductAttention
blocks + linear readout) and trains it briefly on a "broadcast" task —
every output position must equal a function of input position 0. A
correctly-fit model routes most queries to key 0, giving low-entropy
("spike") attention rows; an untrained model gives near-uniform
("dead") rows. The example prints TNNet.AttentionEntropyReport before
and after training so the contrast is visible.

Pure CPU, well under a minute.

Copyright (C) 2026 Joao Paulo Schwarz Schuler

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
any later version.

Coded by Claude (AI).
*)

{$mode objfpc}{$H+}

uses {$IFDEF UNIX} cthreads, {$ENDIF}
  Classes, SysUtils, Math,
  neuralnetwork,
  neuralvolume;

const
  cSeqLen     = 6;
  cEmb        = 8;     // per-token embedding dim
  cDk         = 4;     // attention head dim
  cEpochs     = 200;
  cBatch      = 32;
  cLR         = 0.05;
  cInertia    = 0.9;
  cProbeCnt   = 8;

  procedure BuildModel(out NN: TNNet);
  begin
    NN := TNNet.Create();
    NN.AddLayer(TNNetInput.Create(cSeqLen, 1, cEmb, 1));
    // Project embeddings to a Q|K|V concatenation of width 3*Dk.
    NN.AddLayer(TNNetPointwiseConvLinear.Create(3 * cDk));
    // First SDPA block (non-causal): full bidirectional attention.
    NN.AddLayer(TNNetScaledDotProductAttention.Create(cDk, False));
    // Re-pack d_k -> 3*d_k for a second SDPA block.
    NN.AddLayer(TNNetPointwiseConvLinear.Create(3 * cDk));
    // Second SDPA block (causal): every query may only see past keys.
    NN.AddLayer(TNNetScaledDotProductAttention.Create(cDk, True));
    // Read out d_k -> cEmb for the regression target.
    NN.AddLayer(TNNetPointwiseConvLinear.Create(cEmb));
    NN.SetLearningRate(cLR, cInertia);
  end;

  // Build a probe input / target pair.
  // Input: per-position random embedding. Target: at every position i,
  // output[i, 0, d] := tanh(input[0, 0, d]) — i.e. a broadcast of the
  // function of position 0 across the whole sequence. The optimal way
  // to solve this with attention is to make every query attend almost
  // entirely to key 0, producing low-entropy "spike" rows.
  procedure MakePair(out X, Y: TNNetVolume);
  var
    I, D: integer;
  begin
    X := TNNetVolume.Create(cSeqLen, 1, cEmb);
    Y := TNNetVolume.Create(cSeqLen, 1, cEmb);
    for I := 0 to cSeqLen - 1 do
      for D := 0 to cEmb - 1 do
        X[I, 0, D] := (Random - 0.5) * 2.0;
    for I := 0 to cSeqLen - 1 do
      for D := 0 to cEmb - 1 do
        Y[I, 0, D] := Tanh(X[0, 0, D]);
  end;

  procedure BuildProbes(out Probes: TNNetVolumeList);
  var
    K, I, D: integer;
    V: TNNetVolume;
  begin
    Probes := TNNetVolumeList.Create(True);
    for K := 0 to cProbeCnt - 1 do
    begin
      V := TNNetVolume.Create(cSeqLen, 1, cEmb);
      for I := 0 to cSeqLen - 1 do
        for D := 0 to cEmb - 1 do
          V[I, 0, D] := (Random - 0.5) * 2.0;
      Probes.Add(V);
    end;
  end;

  function ComputeLoss(NN: TNNet; X, Y: TNNetVolume): TNeuralFloat;
  var
    I: integer;
    Diff: TNeuralFloat;
    Out0: TNNetVolume;
  begin
    NN.Compute(X);
    Out0 := NN.GetLastLayer.Output;
    Result := 0;
    for I := 0 to Out0.Size - 1 do
    begin
      Diff := Out0.Raw[I] - Y.Raw[I];
      Result := Result + Diff * Diff;
    end;
    Result := Result / Out0.Size;
  end;

  procedure TrainOnce(NN: TNNet; Epochs: integer);
  var
    Ep, B: integer;
    X, Yt: TNNetVolume;
    TotalLoss: TNeuralFloat;
  begin
    for Ep := 1 to Epochs do
    begin
      TotalLoss := 0;
      for B := 1 to cBatch do
      begin
        MakePair(X, Yt);
        try
          TotalLoss := TotalLoss + ComputeLoss(NN, X, Yt);
          NN.Backpropagate(Yt);
        finally
          X.Free;
          Yt.Free;
        end;
      end;
      if (Ep = 1) or (Ep mod 25 = 0) or (Ep = Epochs) then
        WriteLn(Format('  epoch %3d  mean-MSE=%.6f',
          [Ep, TotalLoss / cBatch]));
    end;
  end;

var
  NN: TNNet;
  Probes: TNNetVolumeList;
begin
  RandSeed := 2026;
  WriteLn('AttentionEntropyReport demo: tiny 2-block SDPA model on a ',
    'broadcast-from-position-0 task.');

  BuildModel(NN);
  BuildProbes(Probes);
  try
    WriteLn;
    WriteLn('Architecture:');
    NN.PrintSummary();

    WriteLn;
    WriteLn(StringOfChar('=', 92));
    WriteLn('BEFORE training (random weights, attention should be near-uniform):');
    WriteLn(StringOfChar('=', 92));
    Write(TNNet.AttentionEntropyReport(NN, Probes, 0.05, 0.1));

    WriteLn;
    WriteLn('Training for ', cEpochs, ' epochs of batch size ', cBatch, '...');
    TrainOnce(NN, cEpochs);

    WriteLn;
    WriteLn(StringOfChar('=', 92));
    WriteLn('AFTER training (rows should concentrate on key 0 - more spikes):');
    WriteLn(StringOfChar('=', 92));
    Write(TNNet.AttentionEntropyReport(NN, Probes, 0.05, 0.1));

    WriteLn;
    WriteLn(
      'Expect: BEFORE has rows clustered near log(SeqLen) (uniform/dead); ' +
      'AFTER pulls mass to lower bins as queries learn to route to key 0.');
  finally
    Probes.Free;
    NN.Free;
  end;
end.
