program MixtureOfExperts;
(*
MixtureOfExperts: soft (dense) mixture-of-experts feed-forward block demo.
Copyright (C) 2026 Joao Paulo Schwarz Schuler

This program demonstrates TNNet.AddMixtureOfExperts on a synthetic
multi-task regression toy: each input carries a one-hot "task" selector in
its first channels and a payload value in the rest. Each task applies a
different non-linear function to the payload, so the gating network has an
incentive to route different tasks to different experts. The program prints
the mean squared error per epoch and confirms that the loss decreases.

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA 02111-1307, USA.
*)

{$mode objfpc}{$H+}

uses
  {$IFDEF UNIX}
  cthreads,
  {$ENDIF}
  Classes, SysUtils, neuralnetwork, neuralvolume, neuralfit;

const
  NumTasks    = 3;   // one-hot task selector width
  PayloadDim  = 2;   // payload values per sample
  d_model     = NumTasks + PayloadDim;
  NumExperts  = 4;
  ExpertHidden= 12;
  SampleCount = 96;
  Epochs      = 300;

procedure RunExample();
var
  NN: TNNet;
  I, Epoch, Task: integer;
  Inputs, Targets: array of TNNetVolume;
  Vol, Pred: TNNetVolume;
  p0, p1, y, EpochLoss, FirstLoss, LastLoss: TNeuralFloat;
begin
  RandSeed := 12345;
  NN := TNNet.Create();
  // Input (1,1,d_model) -> soft MoE FFN (shape-preserving) -> linear head.
  NN.AddLayer( TNNetInput.Create(1, 1, d_model) );
  NN.AddMixtureOfExperts(nil, NumExperts, ExpertHidden);
  NN.AddLayer( TNNetFullConnectLinear.Create(1) );

  WriteLn('Mixture-of-Experts network layers:');
  NN.DebugStructure();

  // Build the synthetic multi-task dataset.
  SetLength(Inputs, SampleCount);
  SetLength(Targets, SampleCount);
  for I := 0 to SampleCount - 1 do
  begin
    Inputs[I] := TNNetVolume.Create(1, 1, d_model);
    Targets[I] := TNNetVolume.Create(1, 1, 1);
    Vol := Inputs[I];
    Vol.Fill(0);
    Task := I mod NumTasks;
    Vol[0, 0, Task] := 1.0;                 // one-hot task selector
    p0 := Random() * 2 - 1;
    p1 := Random() * 2 - 1;
    Vol[0, 0, NumTasks + 0] := p0;
    Vol[0, 0, NumTasks + 1] := p1;
    // Each task computes a different function of the payload.
    case Task of
      0: y := p0 * p1;            // product
      1: y := p0 * p0 - p1;       // square minus
    else y := 0.5 * (p0 + p1);    // average
    end;
    Targets[I][0, 0, 0] := y;
  end;

  Pred := TNNetVolume.Create(1, 1, 1);
  NN.SetLearningRate(0.02, 0.9);

  FirstLoss := 0;
  LastLoss := 0;
  for Epoch := 1 to Epochs do
  begin
    EpochLoss := 0;
    for I := 0 to SampleCount - 1 do
    begin
      NN.Compute( Inputs[I] );
      NN.GetOutput( Pred );
      NN.Backpropagate( Targets[I] );
      EpochLoss := EpochLoss + Sqr(Pred[0,0,0] - Targets[I][0,0,0]);
    end;
    EpochLoss := EpochLoss / SampleCount;
    if Epoch = 1 then FirstLoss := EpochLoss;
    LastLoss := EpochLoss;
    if (Epoch mod 50 = 0) or (Epoch = 1) then
      WriteLn('Epoch ', Epoch:4, '  mean sq error: ', EpochLoss:8:5);
  end;

  WriteLn('Initial MSE: ', FirstLoss:8:5, '   Final MSE: ', LastLoss:8:5);
  if LastLoss < FirstLoss
    then WriteLn('OK: loss decreased - the MoE block trained.')
    else WriteLn('WARNING: loss did not decrease.');

  Pred.Free;
  for I := 0 to SampleCount - 1 do
  begin
    Inputs[I].Free;
    Targets[I].Free;
  end;
  WriteLn('Done.');
  NN.Free;
  NN := nil;
end;

begin
  RunExample();
end.
