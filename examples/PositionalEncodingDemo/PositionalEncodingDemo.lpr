program PositionalEncodingDemo;
(*
PositionalEncodingDemo: ASCII-heatmap visualization of the Vaswani
sin/cos table built by TNNetSinusoidalPositionalEmbedding compared
against the additive table built by TNNetAddPositionalEmbedding.

Key finding (worth knowing before you reach for these layers):
  Despite the suggestive naming, BOTH layers populate their internal
  FPositionalEmbedding via the same TVolume.PositionalEncoding(base)
  call and ADD it to the input. Neither one updates that table during
  backprop - the table is filled once in SetPrevLayer and is then a
  constant for the rest of the layer's life. Backward pass is identity.
  So in current neural-api there is no parameter-trained position
  table; the two layers are functionally equivalent (modulo the base
  argument's constructor signature). This example demonstrates that
  empirically.

Pipeline:
  1. Build a one-layer model whose only meaningful op is
     TNNetSinusoidalPositionalEmbedding over (SeqLen, 1, Depth).
     Forward a zero input - the output IS the encoding table.
     Render as an ASCII heatmap.
  2. Build the same shape using TNNetAddPositionalEmbedding.
     Forward a zero input and render the result.
  3. Try to "train" the second model on a non-sinusoidal target
     for several SGD steps; confirm the table did not change.
  4. Compare both tables cell-by-cell and report the max difference.

Forward-only on both halves. Finishes in well under a second on
a single CPU.

Copyright (C) 2026 Joao Paulo Schwarz Schuler

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
*)

{$mode objfpc}{$H+}

uses {$IFDEF UNIX} {$IFDEF UseCThreads}
  cthreads, {$ENDIF} {$ENDIF}
  Classes, SysUtils, Math,
  neuralnetwork,
  neuralvolume;

const
  cSeqLen     = 16;
  cDepth      = 16;
  cTrainSteps = 50;
  cLR         = 0.1;
  cRamp: string = ' .:-=+*#%@';

  procedure CaptureSinusoidalTable(out Table: TNNetVolume);
  var
    NN: TNNet;
    InVol: TNNetVolume;
  begin
    NN := TNNet.Create();
    NN.AddLayer(TNNetInput.Create(cSeqLen, 1, cDepth));
    NN.AddLayer(TNNetSinusoidalPositionalEmbedding.Create());

    InVol := TNNetVolume.Create(cSeqLen, 1, cDepth, 0.0);
    NN.Compute(InVol);

    Table := TNNetVolume.Create(cSeqLen, 1, cDepth, 0.0);
    NN.GetOutput(Table);

    InVol.Free;
    NN.Free;
  end;

  // Capture the AddPositionalEmbedding table both before and after a
  // brief "training" loop on a non-sinusoidal target. The before/after
  // tables should be bit-identical because the layer has no trainable
  // parameters - we use this to confirm it empirically.
  procedure CaptureAddTable(out Before, After: TNNetVolume; Target: TNNetVolume);
  var
    NN: TNNet;
    InVol: TNNetVolume;
    Step: integer;
  begin
    NN := TNNet.Create();
    NN.AddLayer(TNNetInput.Create(cSeqLen, 1, cDepth));
    TNNetInput(NN.Layers[0]).EnableErrorCollection;
    NN.AddLayer(TNNetAddPositionalEmbedding.Create());
    NN.AddLayer(TNNetIdentity.Create());

    NN.SetLearningRate(cLR, 0.9);
    NN.SetL2Decay(0.0);

    InVol := TNNetVolume.Create(cSeqLen, 1, cDepth, 0.0);

    NN.Compute(InVol);
    Before := TNNetVolume.Create(cSeqLen, 1, cDepth, 0.0);
    NN.GetOutput(Before);

    for Step := 1 to cTrainSteps do
    begin
      NN.ClearDeltas();
      NN.Compute(InVol);
      NN.Backpropagate(Target);
      NN.UpdateWeights();
    end;

    NN.Compute(InVol);
    After := TNNetVolume.Create(cSeqLen, 1, cDepth, 0.0);
    NN.GetOutput(After);

    InVol.Free;
    NN.Free;
  end;

  // Build a smooth non-sinusoidal "tent in pos times signed ramp in
  // depth" target. Used purely as a regression target for the SGD
  // probe; if the AddPositionalEmbedding table were trainable, several
  // steps at LR=0.1 would clearly drag it away from sin/cos.
  procedure BuildTarget(out V: TNNetVolume);
  var
    X, D: integer;
    pos, depFrac, val: TNeuralFloat;
  begin
    V := TNNetVolume.Create(cSeqLen, 1, cDepth, 0.0);
    for X := 0 to cSeqLen - 1 do
    begin
      pos := X / (cSeqLen - 1);
      for D := 0 to cDepth - 1 do
      begin
        depFrac := D / (cDepth - 1);
        val := (1.0 - Abs(2 * pos - 1.0)) * (2 * depFrac - 1.0);
        V[X, 0, D] := val;
      end;
    end;
  end;

  function Bucket(v, lo, hi: TNeuralFloat): char;
  var
    t: TNeuralFloat;
    idx, n: integer;
  begin
    n := Length(cRamp);
    if hi - lo < 1e-9 then
    begin
      Result := cRamp[1];
      exit;
    end;
    t := (v - lo) / (hi - lo);
    if t < 0 then t := 0;
    if t > 1 then t := 1;
    idx := 1 + Floor(t * (n - 1));
    if idx < 1 then idx := 1;
    if idx > n then idx := n;
    Result := cRamp[idx];
  end;

  procedure PrintHeatmap(const Tag: string; V: TNNetVolume);
  var
    X, D: integer;
    lo, hi, val: TNeuralFloat;
    Line: string;
  begin
    lo :=  1e30;
    hi := -1e30;
    for X := 0 to V.SizeX - 1 do
      for D := 0 to V.Depth - 1 do
      begin
        val := V[X, 0, D];
        if val < lo then lo := val;
        if val > hi then hi := val;
      end;

    WriteLn(Tag);
    WriteLn('  shape = (SeqLen=', V.SizeX, ', Depth=', V.Depth, ')',
      '   min=', lo:0:4, '   max=', hi:0:4);
    Write('  pos\dep ');
    for D := 0 to V.Depth - 1 do Write(D mod 10);
    WriteLn;
    for X := 0 to V.SizeX - 1 do
    begin
      Line := '';
      for D := 0 to V.Depth - 1 do
        Line := Line + Bucket(V[X, 0, D], lo, hi);
      WriteLn('  ', X:3, '     ', Line);
    end;
    WriteLn('  ramp:  low [', cRamp, '] high');
    WriteLn;
  end;

  // Cell-by-cell max absolute difference and mean absolute difference,
  // on the raw values (no rescaling).
  procedure ReportDiff(const Tag: string; A, B: TNNetVolume);
  var
    X, D, Count: integer;
    d_val, maxd, sumd: TNeuralFloat;
  begin
    maxd := 0; sumd := 0; Count := 0;
    for X := 0 to A.SizeX - 1 do
      for D := 0 to A.Depth - 1 do
      begin
        d_val := Abs(A[X, 0, D] - B[X, 0, D]);
        if d_val > maxd then maxd := d_val;
        sumd := sumd + d_val;
        Inc(Count);
      end;
    WriteLn(Tag);
    WriteLn('  max |A - B| = ', maxd:0:8);
    WriteLn('  mean |A - B| = ', (sumd / Count):0:8);
  end;

var
  SinTable, Target, AddBefore, AddAfter: TNNetVolume;
  // Stops Lazarus errors
  Application: record Title: string; end;

begin
  Application.Title := 'Positional Encoding Demo';
  RandSeed := 42;

  WriteLn('PositionalEncodingDemo: ASCII heatmaps of two position tables.');
  WriteLn('SeqLen=', cSeqLen, '  Depth=', cDepth,
    '  SGD-probe steps=', cTrainSteps, '  LR=', cLR:0:3);
  WriteLn;

  CaptureSinusoidalTable(SinTable);
  PrintHeatmap('=== TNNetSinusoidalPositionalEmbedding (Vaswani sin/cos) ===',
    SinTable);

  BuildTarget(Target);
  CaptureAddTable(AddBefore, AddAfter, Target);
  PrintHeatmap('=== TNNetAddPositionalEmbedding (before SGD probe) ===',
    AddBefore);
  PrintHeatmap('=== TNNetAddPositionalEmbedding (after ' +
    IntToStr(cTrainSteps) + ' SGD steps toward a non-sinusoidal target) ===',
    AddAfter);

  WriteLn('--- Sanity check 1: Add table is unchanged by SGD ---');
  ReportDiff('AddBefore vs AddAfter', AddBefore, AddAfter);
  WriteLn;

  WriteLn('--- Sanity check 2: Sinusoidal table == Add table ---');
  ReportDiff('SinTable vs AddBefore', SinTable, AddBefore);
  WriteLn;

  WriteLn('Conclusion:');
  WriteLn('  Both layers populate FPositionalEmbedding via the same');
  WriteLn('  TVolume.PositionalEncoding formula and ADD it to the input.');
  WriteLn('  Neither layer registers a TNNetNeuron for the table, so');
  WriteLn('  UpdateWeights leaves it untouched. The two layers are');
  WriteLn('  functionally equivalent additive Vaswani position encoders.');

  AddAfter.Free;
  AddBefore.Free;
  Target.Free;
  SinTable.Free;
end.
