program CumSumPositionEncoding;
(*
CumSumPositionEncoding: forward-only demonstration of how TNNetCumSum
can be used to manufacture a *learned-free* linear position feature.

The trick:
  If the input is a constant 1.0 along the depth axis, then
      CumSum_depth(Input)[c] = c + 1
  This is a strictly increasing per-position ramp — exactly the kind
  of "where am I in the sequence?" signal that a permutation-invariant
  downstream layer needs in order to behave position-aware.

This example contains no training; it is purely a forward pass through
a network whose only meaningful layer is TNNetCumSum. The demo verifies
the three canonical patterns:

  1. all-ones input  → [1, 2, 3, ...] linear ramp,
  2. arbitrary input → standard prefix-sum,
  3. multi-row input → each (X,Y) location accumulates independently
     along its own depth column.

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
  cSeqLen = 8;

  // Builds a tiny 1-layer pipeline:  Input(1,1,Depth) -> CumSum
  procedure BuildCumSum1D(out NN: TNNet; ADepth: integer);
  begin
    NN := TNNet.Create();
    NN.AddLayer(TNNetInput.Create(1, 1, ADepth));
    NN.AddLayer(TNNetCumSum.Create());
  end;

  // Builds a 2D pipeline so we can show per-row independence:
  //   Input(1, Rows, Depth) -> CumSum  (each row sums on its own)
  procedure BuildCumSum2D(out NN: TNNet; ARows, ADepth: integer);
  begin
    NN := TNNet.Create();
    NN.AddLayer(TNNetInput.Create(1, ARows, ADepth));
    NN.AddLayer(TNNetCumSum.Create());
  end;

  procedure PrintDepth(const Tag: string; V: TNNetVolume);
  var
    D: integer;
  begin
    Write(Tag, ' [');
    for D := 0 to V.Depth - 1 do
    begin
      if D > 0 then Write(', ');
      Write(V.FData[D]:6:2);
    end;
    WriteLn(']');
  end;

  procedure DemoLinearRamp();
  var
    NN: TNNet;
    Input, Output: TNNetVolume;
    D: integer;
    AllOk: boolean;
  begin
    WriteLn('--- Demo 1: constant-1 input  ->  linear position ramp ---');
    BuildCumSum1D(NN, cSeqLen);
    Input  := TNNetVolume.Create(1, 1, cSeqLen, 1.0); // all ones
    Output := TNNetVolume.Create(1, 1, cSeqLen, 0.0);

    NN.Compute(Input);
    NN.GetOutput(Output);

    PrintDepth('  input  =', Input);
    PrintDepth('  cumsum =', Output);

    AllOk := True;
    for D := 0 to cSeqLen - 1 do
      if Abs(Output.FData[D] - (D + 1)) > 1e-6 then AllOk := False;
    if AllOk then
      WriteLn('  OK: CumSum of all-ones is the position index + 1.')
    else
      WriteLn('  FAIL: ramp does not match expected values.');

    Output.Free;
    Input.Free;
    NN.Free;
    WriteLn;
  end;

  procedure DemoArbitraryPrefixSum();
  var
    NN: TNNet;
    Input, Output: TNNetVolume;
    Vals: array[0..5] of TNeuralFloat = (1, 2, 3, 4, 5, 6);
    Expected: array[0..5] of TNeuralFloat = (1, 3, 6, 10, 15, 21);
    D: integer;
    AllOk: boolean;
  begin
    WriteLn('--- Demo 2: arbitrary input   ->  standard prefix-sum ---');
    BuildCumSum1D(NN, Length(Vals));
    Input  := TNNetVolume.Create(1, 1, Length(Vals), 0.0);
    Output := TNNetVolume.Create(1, 1, Length(Vals), 0.0);
    for D := 0 to Length(Vals) - 1 do Input.FData[D] := Vals[D];

    NN.Compute(Input);
    NN.GetOutput(Output);

    PrintDepth('  input  =', Input);
    PrintDepth('  cumsum =', Output);

    AllOk := True;
    for D := 0 to Length(Vals) - 1 do
      if Abs(Output.FData[D] - Expected[D]) > 1e-6 then AllOk := False;
    if AllOk then
      WriteLn('  OK: prefix-sum matches the textbook result.')
    else
      WriteLn('  FAIL: prefix-sum mismatched.');

    Output.Free;
    Input.Free;
    NN.Free;
    WriteLn;
  end;

  procedure DemoPerRowIndependence();
  const
    cRows  = 3;
    cDepth = 5;
  var
    NN: TNNet;
    Input, Output: TNNetVolume;
    Y, D: integer;
    RowVal: TNeuralFloat;
    AllOk: boolean;
  begin
    WriteLn('--- Demo 3: multi-row input   ->  each row sums independently ---');
    BuildCumSum2D(NN, cRows, cDepth);
    Input  := TNNetVolume.Create(1, cRows, cDepth, 0.0);
    Output := TNNetVolume.Create(1, cRows, cDepth, 0.0);

    // Row y has constant value (y + 1) along the depth axis.
    // After CumSum, row y should be:  [(y+1)*1, (y+1)*2, ..., (y+1)*cDepth].
    for Y := 0 to cRows - 1 do
      for D := 0 to cDepth - 1 do
        Input[0, Y, D] := (Y + 1);

    NN.Compute(Input);
    NN.GetOutput(Output);

    AllOk := True;
    for Y := 0 to cRows - 1 do
    begin
      Write('  row ', Y, '  in=');
      for D := 0 to cDepth - 1 do
      begin
        if D > 0 then Write(',');
        Write(Input[0, Y, D]:4:1);
      end;
      Write('   out=');
      for D := 0 to cDepth - 1 do
      begin
        RowVal := (Y + 1) * (D + 1);
        if D > 0 then Write(',');
        Write(Output[0, Y, D]:5:1);
        if Abs(Output[0, Y, D] - RowVal) > 1e-6 then AllOk := False;
      end;
      WriteLn;
    end;

    if AllOk then
      WriteLn('  OK: each row accumulates only its own column.')
    else
      WriteLn('  FAIL: row coupling detected.');

    Output.Free;
    Input.Free;
    NN.Free;
    WriteLn;
  end;

  procedure DemoUsageHint();
  begin
    WriteLn('--- Usage hint ---');
    WriteLn('  To inject a learned-free position feature, concatenate a');
    WriteLn('  CumSum-of-constant branch alongside your real features:');
    WriteLn;
    WriteLn('    Real    : TNNetInput(1,1,Depth)');
    WriteLn('    PosFeat : TNNetInput(1,1,Depth, FillValue=1) -> TNNetCumSum');
    WriteLn('    Merged  : TNNetConcat([Real, PosFeat])  -> downstream layers');
    WriteLn;
    WriteLn('  The downstream model now sees a strictly-increasing position');
    WriteLn('  index in the extra channels, without any trainable parameter.');
    WriteLn;
  end;

var
  // Stops Lazarus errors
  Application: record Title: string; end;

begin
  Application.Title := 'CumSum Position Encoding Demo';
  WriteLn('CumSumPositionEncoding: forward-only demonstration.');
  WriteLn;
  DemoLinearRamp();
  DemoArbitraryPrefixSum();
  DemoPerRowIndependence();
  DemoUsageHint();
end.
