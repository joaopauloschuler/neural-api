program GradCAM;
(*
GradCAM: demonstrates TNNet.GradCAMReport, a forward+backward CNN
class-localisation diagnostic (Grad-CAM, Selvaraju et al. 2017). It trains the
same tiny conv classifier on the same synthetic 8x8x2 two-class task as
examples/SaliencyReport (the class is signalled by a bright 3x3 blob whose
CHANNEL and CORNER depend on the label, plus background noise), then for one
probe sample prints the COARSE, class-discriminative Grad-CAM map (at the
convolution layer's feature-map resolution, nearest-upsampled to the input
plane) side by side with the FINE input-pixel saliency map.

Both reports are forward-only (the trained weights are never updated).

Built-in correctness gate: the brightest cell of the printed coarse Grad-CAM
map must fall inside the class-specific blob's corner. A clear PASS/FAIL line is
printed and the program Halt(1)s on failure, so the example doubles as a
regression test.

No dataset download, pure CPU, well under a minute.

Copyright (C) 2026 Joao Paulo Schwarz Schuler

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
any later version.

Coded by Claude (AI).
*)

{$mode objfpc}{$H+}

uses {$IFDEF UNIX} {$IFDEF UseCThreads}
  cthreads, {$ENDIF} {$ENDIF}
  Classes, SysUtils, Math,
  neuralnetwork,
  neuralvolume;

const
  cSizeX   = 8;
  cSizeY   = 8;
  cDepth   = 2;
  cClasses = 2;
  cEpochs  = 140;
  cBatch   = 24;
  cBuckets = ' .:-=+*#%@';   // same intensity ramp GradCAMReport prints with

// Synthetic labelled image. Class 0 -> bright 3x3 blob in the top-left corner
// of channel 0; class 1 -> bright 3x3 blob in the bottom-right corner of
// channel 1, plus background noise.
procedure MakeSample(out X, Y: TNNetVolume; ForcedClass: integer;
  NoiseLevel: TNeuralFloat);
var
  Cls, px, py, cx, cy, ch, i: integer;
begin
  X := TNNetVolume.Create(cSizeX, cSizeY, cDepth);
  Y := TNNetVolume.Create(cClasses, 1, 1);
  X.Fill(0);
  Y.Fill(0);
  if ForcedClass >= 0 then Cls := ForcedClass
  else Cls := Random(cClasses);
  Y.Raw[Cls] := 1.0;
  for i := 0 to X.Size - 1 do X.Raw[i] := Random * NoiseLevel;
  if Cls = 0 then
  begin
    cx := 1; cy := 1; ch := 0;
  end
  else
  begin
    cx := cSizeX - 2; cy := cSizeY - 2; ch := 1;
  end;
  for py := -1 to 1 do
    for px := -1 to 1 do
      X[cx + px, cy + py, ch] := 1.0 + Random * 0.2;
end;

procedure BuildNet(out NN: TNNet);
begin
  NN := TNNet.Create();
  NN.AddLayer(TNNetInput.Create(cSizeX, cSizeY, cDepth));
  // Keep spatial resolution at the conv layer (stride 1, pad 1) so Grad-CAM
  // has a meaningful (8x8) feature map to localise on.
  NN.AddLayer(TNNetConvolutionReLU.Create(8, 3, 1, 1));
  NN.AddLayer(TNNetMaxPool.Create(2));
  NN.AddLayer(TNNetFullConnectReLU.Create(16));
  NN.AddLayer(TNNetFullConnectLinear.Create(cClasses));
  NN.AddLayer(TNNetSoftMax.Create());
  NN.SetLearningRate(0.01, 0.9);
end;

procedure TrainOnce(NN: TNNet; Epochs: integer);
var
  Ep, B, Hit: integer;
  X, Yt: TNNetVolume;
begin
  for Ep := 1 to Epochs do
  begin
    Hit := 0;
    for B := 1 to cBatch do
    begin
      MakeSample(X, Yt, -1, 0.1);
      try
        NN.Compute(X);
        if NN.GetLastLayer.Output.GetClass() = Yt.GetClass() then Inc(Hit);
        NN.Backpropagate(Yt);
      finally
        X.Free;
        Yt.Free;
      end;
    end;
    if (Ep = 1) or (Ep mod 35 = 0) or (Ep = Epochs) then
      WriteLn(Format('  epoch %3d  train-acc=%.3f', [Ep, Hit / cBatch]));
  end;
end;

procedure PrintInput(X: TNNetVolume);
var
  ch, px, py, b: integer;
  v, mx: TNeuralFloat;
  row: string;
begin
  mx := X.GetMax();
  if mx <= 0 then mx := 1;
  for ch := 0 to cDepth - 1 do
  begin
    WriteLn(Format('  input channel %d:', [ch]));
    for py := 0 to cSizeY - 1 do
    begin
      row := '    ';
      for px := 0 to cSizeX - 1 do
      begin
        v := X[px, py, ch];
        b := Trunc((v / mx) * (Length(cBuckets) - 1) + 0.5);
        if b < 0 then b := 0;
        if b > Length(cBuckets) - 1 then b := Length(cBuckets) - 1;
        row := row + cBuckets[b + 1] + ' ';
      end;
      WriteLn(row);
    end;
  end;
end;

// Parse the printed Grad-CAM report to find the brightest heatmap cell of the
// UPSAMPLED (input-plane) coarse map. Returns its (x,y) in input-plane cells.
// We scan only the rows that are pure heatmap glyphs (made of cBuckets + space)
// of width cSizeX, indented with spaces. Returns false if none found.
function PeakOfReport(const Report: string; out PeakX, PeakY: integer): boolean;
var
  Lines: TStringList;
  i, col, glyphCol, glyphRow, bestBand, band, ci: integer;
  s, trimmed: string;
  c: char;
  isHeatRow: boolean;
begin
  Result := False;
  PeakX := 0; PeakY := 0; bestBand := -1;
  Lines := TStringList.Create();
  try
    Lines.Text := Report;
    glyphRow := 0;
    for i := 0 to Lines.Count - 1 do
    begin
      s := Lines[i];
      trimmed := Trim(s);
      // A heatmap row contains exactly cSizeX glyphs separated by spaces, each
      // glyph drawn from cBuckets. Require >= cSizeX glyph characters and that
      // every non-space character belongs to cBuckets.
      if Length(trimmed) < cSizeX then Continue;
      isHeatRow := True;
      glyphCol := 0;
      for col := 1 to Length(trimmed) do
      begin
        c := trimmed[col];
        if c = ' ' then Continue;
        if Pos(c, cBuckets) = 0 then
        begin
          isHeatRow := False;
          Break;
        end;
        Inc(glyphCol);
      end;
      // Must be a full-width heatmap row.
      if (not isHeatRow) or (glyphCol <> cSizeX) then Continue;

      // This is heat row number glyphRow. Walk its glyphs and track the
      // brightest band/position.
      ci := 0;
      for col := 1 to Length(trimmed) do
      begin
        c := trimmed[col];
        if c = ' ' then Continue;
        band := Pos(c, cBuckets) - 1; // 0..9
        if band > bestBand then
        begin
          bestBand := band;
          PeakX := ci;
          PeakY := glyphRow;
          Result := True;
        end;
        Inc(ci);
      end;
      Inc(glyphRow);
      // Only consider the first cSizeY heat rows (the upsampled map block).
      if glyphRow >= cSizeY then Break;
    end;
  finally
    Lines.Free;
  end;
end;

var
  NN: TNNet;
  X0, Y0: TNNetVolume;
  PeakX, PeakY: integer;
  c, predicted: integer;
  Report: string;
  inBlob, gotPeak: boolean;
begin
  RandSeed := 2026;

  WriteLn('GradCAM demo: 8x8x2 synthetic 2-class image classifier.');
  WriteLn('  class 0 -> bright blob top-left  of channel 0');
  WriteLn('  class 1 -> bright blob bottom-right of channel 1');
  WriteLn;

  BuildNet(NN);
  try
    WriteLn('Training for ', cEpochs, ' epochs of batch size ', cBatch, '...');
    TrainOnce(NN, cEpochs);
    WriteLn;

    // Probe: a clean class-0 sample (blob in the top-left of channel 0).
    MakeSample(X0, Y0, 0, 0.05);
    try
      NN.Compute(X0);
      predicted := NN.GetLastLayer.Output.GetClass();
      c := predicted;
      WriteLn(StringOfChar('=', 72));
      WriteLn(Format('PROBE (clean class-0 sample). true=%d predicted=%d',
        [Y0.GetClass(), predicted]));
      WriteLn(StringOfChar('=', 72));
      PrintInput(X0);
      WriteLn;

      WriteLn('--- Coarse, class-discriminative Grad-CAM (where the CNN looked) ---');
      Report := TNNet.GradCAMReport(NN, X0);
      Write(Report);
      WriteLn;
      WriteLn('--- Fine, input-pixel saliency on the SAME sample ---');
      Write(TNNet.SaliencyReport(NN, X0));
      WriteLn;

      // ---- Built-in correctness gate. ----
      // Find the brightest cell of the printed UPSAMPLED Grad-CAM map and check
      // it falls inside the class-0 blob (the 3x3 region centred at input 1,1).
      gotPeak := PeakOfReport(Report, PeakX, PeakY);
      WriteLn(StringOfChar('=', 72));
      if not gotPeak then
      begin
        WriteLn('FAIL: could not locate a Grad-CAM heatmap in the report.');
        Halt(1);
      end;
      WriteLn(Format('Grad-CAM peak (input plane) at (%d,%d).', [PeakX, PeakY]));
      inBlob := (Abs(PeakX - 1) <= 1) and (Abs(PeakY - 1) <= 1);
      if (predicted = 0) and inBlob then
        WriteLn('PASS: Grad-CAM peak falls inside the class-0 (top-left) region.')
      else
      begin
        WriteLn('FAIL: Grad-CAM peak is outside the class-0 (top-left) region.');
        Halt(1);
      end;
    finally
      X0.Free;
      Y0.Free;
    end;
  finally
    NN.Free;
  end;
end.
