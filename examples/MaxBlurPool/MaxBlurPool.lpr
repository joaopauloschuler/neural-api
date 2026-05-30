program MaxBlurPool;
(*
MaxBlurPool: demonstrates anti-aliased ("shift-invariant") max pooling
(Zhang 2019, "Making Convolutional Networks Shift-Invariant Again").

A naive strided max pool aliases: subsampling directly after a non-linear
max makes the output jump under tiny input translations. TNNetMaxBlurPool
instead takes a DENSE (stride-1) max and only then subsamples, applying a
fixed binomial low-pass blur with the downsampling stride. This restores
approximate shift-invariance.

This demo builds a smooth multi-channel image, then measures the MEAN
absolute output change of a plain strided TNNetMaxPool versus
TNNetMaxBlurPool under small 1..3 px input shifts. It ends with a
self-checking PASS/FAIL gate: MaxBlurPool MUST change less than MaxPool.

The demo is tiny and CPU-only (no training), runs in well under a second.

Copyright (C) 2026 Joao Paulo Schwarz Schuler

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
any later version.
*)

{$mode objfpc}{$H+}

uses {$IFDEF UNIX} {$IFDEF UseCThreads}
  cthreads, {$ENDIF} {$ENDIF}
  Classes, SysUtils,
  neuralnetwork,
  neuralvolume;

const
  cSize = 16;     // square input side (square keeps both pools on fast path)
  cDepth = 3;     // channels
  cMaxShift = 3;  // measure shifts of 1..cMaxShift px

// Mean absolute change of ANet's output between ABase and AShifted.
function MeanOutputChange(ANet: TNNet; ABase, AShifted: TNNetVolume): TNeuralFloat;
var
  k: integer;
  OutBase: TNNetVolume;
  Sum: TNeuralFloat;
begin
  ANet.Compute(ABase);
  OutBase := TNNetVolume.Create();
  OutBase.Copy(ANet.GetLastLayer.Output);
  ANet.Compute(AShifted);
  Sum := 0;
  for k := 0 to OutBase.Size - 1 do
    Sum := Sum + Abs(OutBase.Raw[k] - ANet.GetLastLayer.Output.Raw[k]);
  Result := Sum / OutBase.Size;
  OutBase.Free;
end;

var
  NNMax, NNBlur: TNNet;
  Base, Shifted: TNNetVolume;
  x, y, d, s: integer;
  MaxChange, BlurChange, mc, bc: TNeuralFloat;
  Pass: boolean;
begin
  WriteLn('MaxBlurPool shift-invariance demo (Zhang 2019)');
  WriteLn('Input: ', cSize, 'x', cSize, 'x', cDepth,
    ', pool size 2 / stride 2.');
  WriteLn;

  Base := TNNetVolume.Create(cSize, cSize, cDepth);
  Shifted := TNNetVolume.Create(cSize, cSize, cDepth);
  NNMax := TNNet.Create();
  NNBlur := TNNet.Create();
  try
    // Smooth, gently varying signal (mild high-frequency content so aliasing
    // is visible but not adversarial). Distinct values, no ties.
    for x := 0 to cSize - 1 do
      for y := 0 to cSize - 1 do
        for d := 0 to cDepth - 1 do
          Base[x, y, d] :=
            Sin(x * 0.7 + d) + 0.5 * Cos(y * 0.5 + 0.3 * d) + 0.01 * x;

    NNMax.AddLayer(TNNetInput.Create(cSize, cSize, cDepth, 1));
    NNMax.AddLayer(TNNetMaxPool.Create(2));        // plain strided maxpool

    NNBlur.AddLayer(TNNetInput.Create(cSize, cSize, cDepth, 1));
    NNBlur.AddLayer(TNNetMaxBlurPool.Create(2));   // anti-aliased maxpool

    WriteLn(Format('%-8s %14s %14s', ['shift(px)', 'MaxPool', 'MaxBlurPool']));
    MaxChange := 0;
    BlurChange := 0;
    for s := 1 to cMaxShift do
    begin
      // 1..cMaxShift px horizontal shift of Base (wrap-around).
      for x := 0 to cSize - 1 do
        for y := 0 to cSize - 1 do
          for d := 0 to cDepth - 1 do
            Shifted[x, y, d] := Base[(x + s) mod cSize, y, d];

      mc := MeanOutputChange(NNMax, Base, Shifted);
      bc := MeanOutputChange(NNBlur, Base, Shifted);
      MaxChange := MaxChange + mc;
      BlurChange := BlurChange + bc;
      WriteLn(Format('%-8d %14.6f %14.6f', [s, mc, bc]));
    end;
    MaxChange := MaxChange / cMaxShift;
    BlurChange := BlurChange / cMaxShift;

    WriteLn;
    WriteLn(Format('Mean output change under 1..%d px shift:', [cMaxShift]));
    WriteLn(Format('  strided MaxPool : %.6f', [MaxChange]));
    WriteLn(Format('  MaxBlurPool     : %.6f', [BlurChange]));
    WriteLn(Format('  reduction       : %.1f%%',
      [100.0 * (MaxChange - BlurChange) / MaxChange]));
    WriteLn;

    // Self-checking gate: anti-aliased pooling must be more shift-invariant.
    Pass := BlurChange < MaxChange;
    if Pass then
      WriteLn('GATE: PASS - MaxBlurPool is more shift-invariant than MaxPool.')
    else
      WriteLn('GATE: FAIL - MaxBlurPool did NOT reduce the shift-induced change.');
  finally
    NNMax.Free;
    NNBlur.Free;
    Base.Free;
    Shifted.Free;
  end;

  if not Pass then Halt(1);
end.
