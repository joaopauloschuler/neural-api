program SaliencyReport;
(*
SaliencyReport: demonstrates TNNet.SaliencyReport, a forward+backward
input-attribution (saliency) diagnostic. It trains a tiny conv classifier on a
synthetic 8x8x2 two-class dataset where the class is signalled by a bright 3x3
blob whose CHANNEL and CORNER depend on the label (plus background noise), then
runs the report on a couple of probe samples.

For a chosen probe sample and its predicted class c the report prints three
input-attribution heatmaps side by side:
  (a) vanilla input-gradient saliency |d logit_c / d x| (one forward + one
      backward pass with the final-layer error set to the one-hot e_c),
  (b) SmoothGrad   - (a) averaged over N noisy copies of the input,
  (c) Integrated Gradients - the path integral from a zero baseline to x.
It also prints, per channel, the total attribution mass, the top-K most-
attributing pixels, and the IG COMPLETENESS GAP |sum(IG) - (logit_c(x) -
logit_c(0))|, which is the built-in correctness check: a small relative gap
means the integration is faithful.

The discriminative blob sits in a class-specific corner/channel, so on a
correctly-classified sample all three heatmaps should light up around that
blob - eyeball them against the printed input. A second, deliberately noisier
probe shows a harder case.

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
  cEpochs  = 120;
  cBatch   = 24;

// Synthetic labelled image. Class 0 -> bright blob in the top-left corner of
// channel 0; class 1 -> bright blob in the bottom-right corner of channel 1.
// NoiseLevel controls the background clutter (a higher value makes the sample
// harder to attribute cleanly).
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
    if (Ep = 1) or (Ep mod 30 = 0) or (Ep = Epochs) then
      WriteLn(Format('  epoch %3d  train-acc=%.3f', [Ep, Hit / cBatch]));
  end;
end;

procedure PrintInput(X: TNNetVolume);
const
  cBuckets = ' .:-=+*#%@';
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

var
  NN: TNNet;
  X0, Y0, X1, Y1: TNNetVolume;
begin
  RandSeed := 2026;

  WriteLn('SaliencyReport demo: 8x8x2 synthetic 2-class image classifier.');
  WriteLn('  class 0 -> bright blob top-left  of channel 0');
  WriteLn('  class 1 -> bright blob bottom-right of channel 1');
  WriteLn;

  BuildNet(NN);
  try
    WriteLn('Training for ', cEpochs, ' epochs of batch size ', cBatch, '...');
    TrainOnce(NN, cEpochs);
    WriteLn;

    // ---- Probe 1: a clean, correctly-classified class-0 sample. ----
    MakeSample(X0, Y0, 0, 0.05);
    try
      NN.Compute(X0);
      WriteLn(StringOfChar('=', 72));
      WriteLn(Format('PROBE 1 (clean class-0 sample). true=%d predicted=%d',
        [Y0.GetClass(), NN.GetLastLayer.Output.GetClass()]));
      WriteLn(StringOfChar('=', 72));
      PrintInput(X0);
      WriteLn;
      Write(TNNet.SaliencyReport(NN, X0));
    finally
      X0.Free;
      Y0.Free;
    end;

    WriteLn;

    // ---- Probe 2: a noisier (harder) class-1 sample. ----
    MakeSample(X1, Y1, 1, 0.45);
    try
      NN.Compute(X1);
      WriteLn(StringOfChar('=', 72));
      WriteLn(Format('PROBE 2 (noisy class-1 sample). true=%d predicted=%d',
        [Y1.GetClass(), NN.GetLastLayer.Output.GetClass()]));
      WriteLn(StringOfChar('=', 72));
      PrintInput(X1);
      WriteLn;
      Write(TNNet.SaliencyReport(NN, X1));
    finally
      X1.Free;
      Y1.Free;
    end;

    WriteLn;
    WriteLn(
      'Expect: on the clean sample the three heatmaps concentrate around the '+
      'class-specific blob (top-left ch0 for class 0, bottom-right ch1 for '+
      'class 1) and the IG completeness gap is small. The noisier sample '+
      'spreads attribution more. The completeness gap is the IG correctness '+
      'check. Forward+backward only; the trained weights are untouched.');
  finally
    NN.Free;
  end;
end.
