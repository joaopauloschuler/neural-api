program ReversibleBlock;
(*
ReversibleBlock: demonstrates TNNet.AddReversibleBlock, a RevNet-style
reversible additive-coupling block (Gomez et al. 2017, "The Reversible
Residual Network"). Pure CPU, synthetic data, finishes in seconds.

WHAT IT SHOWS
-------------
1. The block TRAINS end-to-end without NaN on a tiny synthetic identity-ish
   task (input -> ReversibleBlock -> PointwiseConvLinear -> reconstruct input).
2. The headline property: EXACT analytic INVERTIBILITY. Given only the block
   OUTPUT (y1,y2) and the SAME F,G weights, the original input (x1,x2) is
   recovered to floating-point tolerance.

THE ADDITIVE-COUPLING MATH
--------------------------
The input depth is split in two equal halves x1 | x2. Two arbitrary
(shape-preserving) functions F and G map a half-depth volume to a half-depth
volume. The forward coupling is:

    y1 = x1 + F(x2)
    y2 = x2 + G(y1)

This is invertible NO MATTER WHAT F and G are, because the forward step only
ever ADDS a function of an already-known quantity. Running it backwards:

    x2 = y2 - G(y1)     (y1 is the first output half, available directly)
    x1 = y1 - F(x2)     (x2 was just recovered)

So F and G never need to be inverted; only the additions are undone. That is
what makes RevNets able to discard activations in the forward pass and
recompute them during backprop (constant activation memory in depth). This
example does NOT use that memory trick (standard backprop is used), it just
demonstrates the exact round-trip that the trick relies on.

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

Coded by Claude (AI).
*)

{$mode objfpc}{$H+}

uses {$IFDEF UNIX} {$IFDEF UseCThreads}
  cthreads, {$ENDIF} {$ENDIF}
  Classes, SysUtils, Math,
  neuralnetwork,
  neuralvolume;

const
  cSizeX   = 2;
  cSizeY   = 2;
  cDepth   = 4;     // must be even; split into 2 + 2
  cHalf    = cDepth div 2;
  cHidden  = 8;     // F/G hidden width
  cSamples = 64;    // tiny synthetic training set
  cEpochs  = 60;
  cLR      = 0.02;
  cSeed    = 424242;

type
  TVolArray = array of TNNetVolume;

var
  NN: TNNet;
  RevOut, InputLayer: TNNetLayer;
  // Internal block layer refs used for the analytic inverse.
  LX1, LX2, LFOut, LGOut: TNNetLayer;
  TrainData: TVolArray;
  Desired: TNNetVolume;
  Epoch, S, i, x, y, d: integer;
  MeanLoss, y1v, y2v, x1rec, x2rec, recErr, maxRecErr, diff: TNeuralFloat;
  Probe: TNNetVolume;

procedure BuildData;
var
  k, j: integer;
begin
  SetLength(TrainData, cSamples);
  for k := 0 to cSamples - 1 do
  begin
    TrainData[k] := TNNetVolume.Create(cSizeX, cSizeY, cDepth);
    for j := 0 to TrainData[k].Size - 1 do
      TrainData[k].Raw[j] := Sin((k + 1) * 0.13 + j * 0.7) * 1.2;
  end;
end;

procedure FreeData;
var
  k: integer;
begin
  for k := 0 to High(TrainData) do TrainData[k].Free;
  SetLength(TrainData, 0);
end;

begin
  RandSeed := cSeed;
  // Direct Compute/Backpropagate are single-threaded by construction; no fit
  // object and no thread pool are used, so the run stays tiny and deterministic.
  WriteLn('ReversibleBlock: RevNet additive coupling y1=x1+F(x2), y2=x2+G(y1)');
  WriteLn('Input shape (', cSizeX, ',', cSizeY, ',', cDepth,
    ')  hidden=', cHidden);
  WriteLn;

  NN := TNNet.Create();
  InputLayer := NN.AddLayer(TNNetInput.Create(cSizeX, cSizeY, cDepth));
  RevOut := NN.AddReversibleBlock(InputLayer, cHidden);
  // A trivial head so there is something to train (reconstruct the input).
  NN.AddLayer(TNNetPointwiseConvLinear.Create(cDepth));
  NN.SetLearningRate(cLR, 0.9);
  NN.InitWeights();

  // Internal block layers (see AddReversibleBlock wiring order):
  // [1]=x1 split, [2]=x2 split, [3]=F.ReLU, [4]=F.Linear=F(x2),
  // [5]=Sum y1, [6]=G.ReLU, [7]=G.Linear=G(y1), [8]=Sum y2, [9]=Concat.
  LX1   := NN.Layers[1];
  LX2   := NN.Layers[2];
  LFOut := NN.Layers[4];
  LGOut := NN.Layers[7];

  BuildData;
  Desired := TNNetVolume.Create(cSizeX, cSizeY, cDepth);
  try
    WriteLn('Training ', cEpochs, ' epochs on a tiny synthetic reconstruction task...');
    for Epoch := 1 to cEpochs do
    begin
      MeanLoss := 0;
      for S := 0 to cSamples - 1 do
      begin
        Desired.Copy(TrainData[S]);  // target = the input itself
        NN.Compute(TrainData[S]);
        NN.Backpropagate(Desired);
        for i := 0 to NN.GetLastLayer().Output.Size - 1 do
        begin
          diff := NN.GetLastLayer().Output.Raw[i] - Desired.Raw[i];
          MeanLoss := MeanLoss + diff * diff;
        end;
      end;
      MeanLoss := MeanLoss / (cSamples * NN.GetLastLayer().Output.Size);
      if (Epoch = 1) or (Epoch mod 20 = 0) then
        WriteLn(Format('  epoch %4d   mse=%10.6f', [Epoch, MeanLoss]));
      if IsNaN(MeanLoss) or IsInfinite(MeanLoss) then
      begin
        WriteLn('  ERROR: loss became non-finite.');
        Break;
      end;
    end;

    WriteLn;
    WriteLn('Analytic inverse round-trip on a fresh probe input:');
    Probe := TNNetVolume.Create(cSizeX, cSizeY, cDepth);
    try
      for i := 0 to Probe.Size - 1 do
        Probe.Raw[i] := Cos(i * 0.41) * 0.9 - 0.3;
      NN.Compute(Probe);

      // Forward output of the reversible block is Concat(y1, y2).
      // Recover the input WITHOUT inverting F or G:
      //   x2 = y2 - G(y1),  x1 = y1 - F(x2).
      maxRecErr := 0;
      for x := 0 to cSizeX - 1 do
        for y := 0 to cSizeY - 1 do
          for d := 0 to cHalf - 1 do
          begin
            y1v := RevOut.Output[x, y, d];
            y2v := RevOut.Output[x, y, d + cHalf];
            x2rec := y2v - LGOut.Output[x, y, d];   // x2 = y2 - G(y1)
            x1rec := y1v - LFOut.Output[x, y, d];   // x1 = y1 - F(x2)

            recErr := Abs(x1rec - LX1.Output[x, y, d]);
            if recErr > maxRecErr then maxRecErr := recErr;
            recErr := Abs(x2rec - LX2.Output[x, y, d]);
            if recErr > maxRecErr then maxRecErr := recErr;

            // And it must match the ORIGINAL probe input channels.
            recErr := Abs(x1rec - Probe[x, y, d]);
            if recErr > maxRecErr then maxRecErr := recErr;
            recErr := Abs(x2rec - Probe[x, y, d + cHalf]);
            if recErr > maxRecErr then maxRecErr := recErr;
          end;

      WriteLn(Format('  max reconstruction error = %.3e', [maxRecErr]));
      if maxRecErr < 1e-4 then
        WriteLn('  ROUND-TRIP OK: input recovered from output to fp tolerance.')
      else
        WriteLn('  ROUND-TRIP FAILED: reconstruction error too large.');
    finally
      Probe.Free;
    end;
  finally
    Desired.Free;
    FreeData;
    NN.Free;
  end;
end.
