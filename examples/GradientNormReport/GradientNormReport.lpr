program GradientNormReport;
(*
GradientNormReport: builds a deep 12-layer ReLU MLP for a tiny hypotenuse-like
regression task, runs a single forward + backward pass on a probe batch, and
prints TNNet.GradientNormReport. Then rebuilds the SAME stack with a
TNNetLayerNorm inserted at the midpoint and prints the report again. The
contrast should show smoother gradient magnitudes (and fewer "vanishing"
flags / ratios closer to 1.0) with the norm layer.

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
  cDepth        = 12;
  cWidth        = 16;
  cLearningRate = 0.01;

  procedure BuildPlainMLP(out NN: TNNet);
  var
    I: integer;
  begin
    NN := TNNet.Create();
    NN.AddLayer(TNNetInput.Create(2, 1, 1));
    for I := 1 to cDepth do
      NN.AddLayer(TNNetFullConnectReLU.Create(cWidth));
    NN.AddLayer(TNNetFullConnectLinear.Create(1));
    NN.SetLearningRate(cLearningRate, 0.9);
  end;

  procedure BuildNormMLP(out NN: TNNet);
  // Same stack with a LayerNorm inserted at the midpoint.
  var
    I, MidPoint: integer;
  begin
    NN := TNNet.Create();
    NN.AddLayer(TNNetInput.Create(2, 1, 1));
    MidPoint := cDepth div 2;
    for I := 1 to cDepth do
    begin
      NN.AddLayer(TNNetFullConnectReLU.Create(cWidth));
      if I = MidPoint then
        NN.AddLayer(TNNetLayerNorm.Create());
    end;
    NN.AddLayer(TNNetFullConnectLinear.Create(1));
    NN.SetLearningRate(cLearningRate, 0.9);
  end;

  procedure MakeProbe(out X, Yt: TNNetVolume);
  var
    A, B: TNeuralFloat;
  begin
    A := 0.7;
    B := 0.4;
    X := TNNetVolume.Create(2, 1, 1);
    X.FData[0] := A;
    X.FData[1] := B;
    Yt := TNNetVolume.Create(1, 1, 1);
    Yt.FData[0] := Sqrt(A * A + B * B);
  end;

  procedure RunOne(const Title: string; UseNorm: boolean);
  var
    NN: TNNet;
    X, Yt: TNNetVolume;
    Report: string;
  begin
    RandSeed := 1234;
    if UseNorm then
      BuildNormMLP(NN)
    else
      BuildPlainMLP(NN);
    MakeProbe(X, Yt);
    try
      WriteLn;
      WriteLn(StringOfChar('=', 92));
      WriteLn(Title);
      WriteLn(StringOfChar('=', 92));
      WriteLn('Architecture:');
      NN.PrintSummary();
      WriteLn;
      Report := TNNet.GradientNormReport(NN, X, Yt);
      Write(Report);
    finally
      X.Free;
      Yt.Free;
      NN.Free;
    end;
  end;

begin
  WriteLn('GradientNormReport demo: ', cDepth,
    '-layer ReLU MLP, with and without midpoint LayerNorm.');
  RunOne('Plain ReLU MLP (no normalization)', False);
  RunOne('ReLU MLP with midpoint TNNetLayerNorm', True);
  WriteLn;
  WriteLn(
    'Expect: the norm variant tightens the log10(||dL/dx_in||) spread and ' +
    'pulls per-layer ratios closer to 1.0 versus the plain stack.');
end.
