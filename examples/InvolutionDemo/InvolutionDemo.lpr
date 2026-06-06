program InvolutionDemo;
(*
InvolutionDemo: shows that TNNetReverseChannels, TNNetReverseXY,
TNNetFlipX and TNNetFlipY are mathematical involutions, i.e.
applying any one of them twice in a row reproduces the input
within floating-point tolerance.

For each layer L the example builds a tiny two-layer net

  Input(4, 4, 3) -> L -> L

feeds a fixed random 4x4x3 input volume, and checks that the
output equals the input under L1 distance < 1e-6. A PASS/FAIL
line is printed per layer.

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

uses {$IFDEF UNIX} cthreads, {$ENDIF}
  Classes, SysUtils,
  neuralnetwork,
  neuralvolume;

const
  Tolerance = 1.0e-6;

  // Runs the per-layer involution test. Returns true on PASS.
  function TestInvolution(const Name: string;
    LayerA, LayerB: TNNetLayer;
    Input: TNNetVolume): boolean;
  var
    NN: TNNet;
    Output: TNNetVolume;
    Diff: TNeuralFloat;
    Verdict: string;
  begin
    NN := TNNet.Create();
    NN.AddLayer(TNNetInput.Create(Input.SizeX, Input.SizeY, Input.Depth));
    NN.AddLayer(LayerA);
    NN.AddLayer(LayerB);

    Output := TNNetVolume.Create(Input.SizeX, Input.SizeY, Input.Depth);
    NN.Compute(Input);
    NN.GetOutput(Output);

    Diff := Output.SumDiff(Input);
    Result := Diff < Tolerance;
    if Result then Verdict := 'PASS' else Verdict := 'FAIL';
    WriteLn(Format('  [%s] %-22s  Sum|out - in| = %.3e',
      [Verdict, Name, Diff]));

    Output.Free;
    NN.Free;
  end;

  procedure RunAlgo();
  var
    Input: TNNetVolume;
    AllPass: boolean;
  begin
    RandSeed := 1234;
    Input := TNNetVolume.Create(4, 4, 3);
    Input.RandomizeGaussian(1.0);

    WriteLn('Involution check on a 4x4x3 random input volume',
      ' (tolerance ', Tolerance:0:1, '):');

    AllPass := True;
    AllPass := TestInvolution('TNNetReverseChannels',
      TNNetReverseChannels.Create, TNNetReverseChannels.Create, Input)
      and AllPass;
    AllPass := TestInvolution('TNNetReverseXY',
      TNNetReverseXY.Create, TNNetReverseXY.Create, Input)
      and AllPass;
    AllPass := TestInvolution('TNNetFlipX',
      TNNetFlipX.Create, TNNetFlipX.Create, Input)
      and AllPass;
    AllPass := TestInvolution('TNNetFlipY',
      TNNetFlipY.Create, TNNetFlipY.Create, Input)
      and AllPass;

    WriteLn;
    if AllPass then
      WriteLn('All four layers behave as involutions. OK.')
    else
    begin
      WriteLn('At least one layer is NOT an involution. FAILED.');
      Halt(1);
    end;

    Input.Free;
  end;

var
  // Stops Lazarus errors
  Application: record Title: string; end;

begin
  Application.Title := 'Involution Demo';
  RunAlgo();
end.
