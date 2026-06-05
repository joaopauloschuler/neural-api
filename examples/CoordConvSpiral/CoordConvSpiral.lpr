program CoordConvSpiral;
(*
CoordConvSpiral: minimal CoordConv-vs-plain-conv demo of the new
TNNetCoordConv layer (Liu et al. 2018,
"An intriguing failing of convolutional neural networks and the
CoordConv solution", https://arxiv.org/abs/1807.03247).

Toy task. The input is an 8x8x1 image whose pixels are all 0 except
for exactly one pixel set to 1.0. The target is the (x, y) coordinate
of that pixel, normalized to [-1, 1].

The two architectures are otherwise identical (same conv shapes, same
optimizer settings, same data stream); the only difference is the
leading TNNetCoordConv. The head is a global-average-pool followed by
a tiny linear map, which is a translation-INVARIANT readout: a stack
of 1x1 convolutions then global-average-pool over (x, y) cannot
encode where the "1" is. A CoordConv at the front gives the first
convolution two extra channels that ARE the normalized (x, y)
coordinates, so the network only has to learn a trivial "pick the
coord at the active pixel" selector.

We report final MSE on a held-out test set after a short training
budget. The plain conv stays near the constant-prediction baseline;
CoordConv collapses the MSE.

Runs in well under a minute on CPU.

Copyright (C) 2026 Joao Paulo Schwarz Schuler

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

Coded by Claude (AI).
*)

{$mode objfpc}{$H+}

uses {$IFDEF UNIX} {$IFDEF UseCThreads}
  cthreads, {$ENDIF} {$ENDIF}
  Classes, SysUtils, Math,
  neuralnetwork,
  neuralvolume;

const
  cSize      = 8;           // 8x8 image
  cChannels  = 16;          // conv width
  cSteps     = 4000;
  cBatch     = 16;
  cLR        = 0.01;
  cInertia   = 0.9;
  cNumTest   = 1000;

function PixelToCoord(P: integer): TNeuralFloat;
// Map pixel index in [0, cSize-1] to coord in [-1, 1].
begin
  Result := (2.0 * P / (cSize - 1)) - 1.0;
end;

procedure SampleExample(InputV, TargetV: TNNetVolume);
var
  PX, PY: integer;
begin
  PX := Random(cSize);
  PY := Random(cSize);
  InputV.Fill(0);
  InputV[PX, PY, 0] := 1.0;
  TargetV.FData[0] := PixelToCoord(PX);
  TargetV.FData[1] := PixelToCoord(PY);
end;

procedure BuildPlain(out NN: TNNet);
begin
  NN := TNNet.Create();
  NN.AddLayer(TNNetInput.Create(cSize, cSize, 1));
  // Two 1x1 ReLU convs: each output cell is a function of its OWN
  // input cell only. No mixing across (x, y). Followed by a global
  // avg-pool, which is exactly translation-INVARIANT.
  NN.AddLayer(TNNetConvolutionReLU.Create(cChannels, 1, 0, 1));
  NN.AddLayer(TNNetConvolutionReLU.Create(cChannels, 1, 0, 1));
  NN.AddLayer(TNNetAvgChannel.Create());
  NN.AddLayer(TNNetFullConnectLinear.Create(2));
  NN.SetLearningRate(cLR, cInertia);
end;

procedure BuildCoord(out NN: TNNet);
begin
  NN := TNNet.Create();
  NN.AddLayer(TNNetInput.Create(cSize, cSize, 1));
  // Single line change: stick a CoordConv at the front so the first
  // conv sees the two appended (x, y) coordinate channels.
  NN.AddLayer(TNNetCoordConv.Create());
  NN.AddLayer(TNNetConvolutionReLU.Create(cChannels, 1, 0, 1));
  NN.AddLayer(TNNetConvolutionReLU.Create(cChannels, 1, 0, 1));
  NN.AddLayer(TNNetAvgChannel.Create());
  NN.AddLayer(TNNetFullConnectLinear.Create(2));
  NN.SetLearningRate(cLR, cInertia);
end;

procedure Train(NN: TNNet; const ATag: string);
var
  Step, B: integer;
  InputV, TargetV: TNNetVolume;
  StartTime, Elapsed: double;
begin
  InputV  := TNNetVolume.Create(cSize, cSize, 1);
  TargetV := TNNetVolume.Create(1, 1, 2);
  try
    StartTime := Now();
    for Step := 1 to cSteps do
    begin
      for B := 1 to cBatch do
      begin
        SampleExample(InputV, TargetV);
        NN.Compute(InputV);
        NN.Backpropagate(TargetV);
      end;
      if (Step = 1) or (Step mod 1000 = 0) or (Step = cSteps) then
      begin
        Elapsed := (Now() - StartTime) * 86400.0;
        WriteLn(Format('  [%s] step %4d / %4d   elapsed=%.1fs',
          [ATag, Step, cSteps, Elapsed]));
      end;
    end;
  finally
    InputV.Free;
    TargetV.Free;
  end;
end;

function EvalMSE(NN: TNNet): TNeuralFloat;
var
  I: integer;
  InputV, TargetV: TNNetVolume;
  dX, dY, Acc: TNeuralFloat;
begin
  InputV  := TNNetVolume.Create(cSize, cSize, 1);
  TargetV := TNNetVolume.Create(1, 1, 2);
  Acc := 0;
  try
    for I := 1 to cNumTest do
    begin
      SampleExample(InputV, TargetV);
      NN.Compute(InputV);
      dX := NN.GetLastLayer.Output.FData[0] - TargetV.FData[0];
      dY := NN.GetLastLayer.Output.FData[1] - TargetV.FData[1];
      Acc := Acc + dX * dX + dY * dY;
    end;
  finally
    InputV.Free;
    TargetV.Free;
  end;
  Result := Acc / cNumTest;
end;

type
  TBuildProc = procedure(out NN: TNNet);

procedure RunOne(BuildProc: TBuildProc;
  const ATag: string; const ATrainSeed, ATestSeed: longint;
  out FinalMSE: TNeuralFloat);
var
  NN: TNNet;
begin
  RandSeed := ATrainSeed;
  BuildProc(NN);
  try
    WriteLn('Model: ', ATag);
    NN.DebugStructure();
    Train(NN, ATag);
    RandSeed := ATestSeed;
    FinalMSE := EvalMSE(NN);
    WriteLn(Format('  [%s] held-out MSE = %.6f', [ATag, FinalMSE]));
  finally
    NN.Free;
  end;
end;

var
  MSEPlain, MSECoord: TNeuralFloat;

begin
  WriteLn('CoordConv vs Plain Conv: locate-the-pixel regression');
  WriteLn(Format('  grid=%dx%d   conv channels=%d   steps=%d   batch=%d',
    [cSize, cSize, cChannels, cSteps, cBatch]));
  WriteLn('  head = global-avg-pool + linear (translation-INVARIANT)');
  // Same train seed so both networks see the same data stream.
  RunOne(@BuildPlain, 'plain', 1234, 9999, MSEPlain);
  RunOne(@BuildCoord, 'coord', 1234, 9999, MSECoord);
  WriteLn;
  WriteLn('Summary:');
  WriteLn(Format('  plain conv  MSE = %.6f', [MSEPlain]));
  WriteLn(Format('  CoordConv   MSE = %.6f', [MSECoord]));
  if MSECoord < MSEPlain then
    WriteLn(Format('  CoordConv reduces MSE by %.1fx (lower is better)',
      [MSEPlain / Max(MSECoord, 1e-9)]))
  else
    WriteLn('  CoordConv did NOT help on this run.');
end.
