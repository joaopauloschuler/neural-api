program LossLandscapeProbe;
(*
LossLandscapeProbe: trains a tiny 2D two-Gaussians classifier with a small
MLP, then probes the trained loss surface along a random filter-normalised
direction with TNNet.LossLandscapeProbe. The probe restores the original
weights at the end so it is purely diagnostic. Bonus: runs the same probe
on a second (less-trained) checkpoint, so the curves can be compared and
the "flat-minima" folklore inspected at a glance on this library.

Pure CPU, well under a minute.

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
  cNumSamples = 200;
  cProbeBatch = 32;
  cHiddenW    = 16;

  procedure BuildClassifier(out NN: TNNet);
  begin
    NN := TNNet.Create();
    NN.AddLayer(TNNetInput.Create(2, 1, 1));
    NN.AddLayer(TNNetFullConnectReLU.Create(cHiddenW));
    NN.AddLayer(TNNetFullConnectReLU.Create(cHiddenW));
    NN.AddLayer(TNNetFullConnectLinear.Create(2));
    NN.AddLayer(TNNetSoftMax.Create());
    NN.SetLearningRate(0.05, 0.9);
  end;

  procedure MakeTwoGaussians(out Samples: TNNetVolumePairList);
  var
    I: integer;
    X, Y: TNNetVolume;
    cls: integer;
    cx, cy: TNeuralFloat;
  begin
    Samples := TNNetVolumePairList.Create();
    for I := 0 to cNumSamples - 1 do
    begin
      cls := I mod 2;
      if cls = 0 then begin cx := -1.0; cy := -1.0; end
                 else begin cx :=  1.0; cy :=  1.0; end;
      X := TNNetVolume.Create(2, 1, 1);
      X.FData[0] := cx + 0.4 * (Random() - 0.5) * 2;
      X.FData[1] := cy + 0.4 * (Random() - 0.5) * 2;
      Y := TNNetVolume.Create(2, 1, 1);
      Y.Fill(0);
      Y.FData[cls] := 1.0;
      Samples.Add(TNNetVolumePair.Create(X, Y));
    end;
  end;

  function MeasureAcc(NN: TNNet; Samples: TNNetVolumePairList): TNeuralFloat;
  var
    I, Hit: integer;
    Pair: TNNetVolumePair;
  begin
    Hit := 0;
    for I := 0 to Samples.Count - 1 do
    begin
      Pair := Samples[I];
      NN.Compute(Pair.I);
      if NN.GetLastLayer().Output.GetClass() = Pair.O.GetClass() then
        Inc(Hit);
    end;
    if Samples.Count > 0 then
      Result := Hit / Samples.Count
    else
      Result := 0;
  end;

  procedure TrainBriefly(NN: TNNet; Samples: TNNetVolumePairList;
    Epochs: integer);
  var
    Epoch, I: integer;
    Pair: TNNetVolumePair;
  begin
    for Epoch := 1 to Epochs do
    begin
      for I := 0 to Samples.Count - 1 do
      begin
        Pair := Samples[I];
        NN.Compute(Pair.I);
        NN.Backpropagate(Pair.O);
      end;
    end;
  end;

  procedure RunOne(const Title: string; Epochs: integer);
  var
    NN: TNNet;
    All, Probe: TNNetVolumePairList;
    I: integer;
    Acc: TNeuralFloat;
    Report: string;
  begin
    RandSeed := 1234;
    BuildClassifier(NN);
    MakeTwoGaussians(All);
    // Probe batch = first cProbeBatch samples (small, fast, fixed).
    Probe := TNNetVolumePairList.Create();
    try
      for I := 0 to cProbeBatch - 1 do
      begin
        // Deep-copy so Probe owns its own pairs (avoids double-free).
        Probe.Add(TNNetVolumePair.CreateCopying(All[I].I, All[I].O));
      end;

      WriteLn;
      WriteLn(StringOfChar('=', 92));
      WriteLn(Title, '  (epochs=', Epochs, ')');
      WriteLn(StringOfChar('=', 92));

      TrainBriefly(NN, All, Epochs);
      Acc := MeasureAcc(NN, All);
      WriteLn(Format('Train accuracy after %d epochs: %.4f', [Epochs, Acc]));
      WriteLn;

      Report := TNNet.LossLandscapeProbe(NN, Probe, 21, 1.0, 1, 7);
      Write(Report);
    finally
      Probe.Free;
      All.Free;
      NN.Free;
    end;
  end;

begin
  WriteLn('LossLandscapeProbe demo: 2D two-Gaussians + small MLP.');
  WriteLn('Direction is filter-normalised (Li et al. 2018).');
  RunOne('Briefly trained (5 epochs)',   5);
  RunOne('Longer trained  (40 epochs)', 40);
  WriteLn;
  WriteLn(
    'Compare the two reports: a flatter minimum should show a smaller ' +
    'sharpness scalar and a larger (or ">R") loss-doubling radius.');
end.
