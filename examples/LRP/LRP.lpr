program LRP;
(*
LRP: demonstrates TNNet.LRPReport, a Layer-wise Relevance Propagation
diagnostic (Bach et al. 2015). Unlike saliency / Grad-CAM (gradient methods),
LRP is a CONSERVATION method: it back-DISTRIBUTES the explained output logit's
relevance through the net under the epsilon-rule, so the total relevance is
preserved at every layer boundary.

The net here is a small DENSE classifier (FullConnectReLU -> FullConnectReLU ->
FullConnectLinear -> SoftMax) trained on a synthetic 6x6x1 two-class task where
the class is signalled by a bright 2x2 blob in a class-specific corner. A dense
stack is used on purpose: the epsilon-rule has an exact closed form there, so
the conservation residual the report prints should stay essentially zero.

For a probe sample and its predicted class c, the report prints:
  (a) a per-layer-boundary RELEVANCE-CONSERVATION RESIDUAL
      |sum(R_in) - sum(R_out)| (the headline LRP sanity check - should be ~0),
  (b) the TOP-K most-relevant input positions, and
  (c) a per-channel ASCII relevance heatmap over the input plane.
The SoftMax layer is SKIPPED honestly (no epsilon rule), and that is stated in
the report rather than faked.

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
  cSizeX   = 6;
  cSizeY   = 6;
  cDepth   = 1;
  cClasses = 2;
  cEpochs  = 150;
  cBatch   = 24;

// Synthetic labelled image. Class 0 -> bright 2x2 blob in the top-left corner;
// class 1 -> bright 2x2 blob in the bottom-right corner. NoiseLevel controls
// the background clutter.
procedure MakeSample(out X, Y: TNNetVolume; ForcedClass: integer;
  NoiseLevel: TNeuralFloat);
var
  Cls, px, py, cx, cy, i: integer;
begin
  X := TNNetVolume.Create(cSizeX, cSizeY, cDepth);
  Y := TNNetVolume.Create(cClasses, 1, 1);
  X.Fill(0);
  Y.Fill(0);
  if ForcedClass >= 0 then Cls := ForcedClass
  else Cls := Random(cClasses);
  Y.Raw[Cls] := 1.0;
  for i := 0 to X.Size - 1 do X.Raw[i] := Random * NoiseLevel;
  if Cls = 0 then begin cx := 0; cy := 0; end
  else begin cx := cSizeX - 2; cy := cSizeY - 2; end;
  for py := 0 to 1 do
    for px := 0 to 1 do
      X[cx + px, cy + py, 0] := 1.0 + Random * 0.2;
end;

procedure BuildNet(out NN: TNNet);
begin
  NN := TNNet.Create();
  NN.AddLayer(TNNetInput.Create(cSizeX, cSizeY, cDepth));
  NN.AddLayer(TNNetFullConnectReLU.Create(16));
  NN.AddLayer(TNNetFullConnectReLU.Create(12));
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
    if (Ep = 1) or (Ep mod 50 = 0) or (Ep = Epochs) then
      WriteLn(Format('  epoch %3d  train-acc=%.3f', [Ep, Hit / cBatch]));
  end;
end;

procedure PrintInput(X: TNNetVolume);
const
  cBuckets = ' .:-=+*#%@';
var
  px, py, b: integer;
  v, mx: TNeuralFloat;
  row: string;
begin
  mx := X.GetMax();
  if mx <= 0 then mx := 1;
  WriteLn('  input (channel 0):');
  for py := 0 to cSizeY - 1 do
  begin
    row := '    ';
    for px := 0 to cSizeX - 1 do
    begin
      v := X[px, py, 0];
      b := Trunc((v / mx) * (Length(cBuckets) - 1) + 0.5);
      if b < 0 then b := 0;
      if b > Length(cBuckets) - 1 then b := Length(cBuckets) - 1;
      row := row + cBuckets[b + 1] + ' ';
    end;
    WriteLn(row);
  end;
end;

var
  NN: TNNet;
  X0, Y0, X1, Y1: TNNetVolume;
begin
  RandSeed := 2026;

  WriteLn('LRPReport demo: 6x6x1 synthetic 2-class image classifier (dense).');
  WriteLn('  class 0 -> bright 2x2 blob top-left');
  WriteLn('  class 1 -> bright 2x2 blob bottom-right');
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
      Write(TNNet.LRPReport(NN, X0));
    finally
      X0.Free;
      Y0.Free;
    end;

    WriteLn;

    // ---- Probe 2: a noisier class-1 sample. ----
    MakeSample(X1, Y1, 1, 0.30);
    try
      NN.Compute(X1);
      WriteLn(StringOfChar('=', 72));
      WriteLn(Format('PROBE 2 (noisy class-1 sample). true=%d predicted=%d',
        [Y1.GetClass(), NN.GetLastLayer.Output.GetClass()]));
      WriteLn(StringOfChar('=', 72));
      PrintInput(X1);
      WriteLn;
      Write(TNNet.LRPReport(NN, X1));
    finally
      X1.Free;
      Y1.Free;
    end;

    WriteLn;
    WriteLn(
      'Expect: the per-layer conservation residual stays O(eps) on the dense ' +
      'epsilon-rule boundaries (-> 0 as eps -> 0, the LRP sanity check), the ' +
      'top-relevant input positions cluster on the class-specific blob, and ' +
      'the SoftMax layer is handled honestly (passthru, no epsilon rule). LRP ' +
      'is a CONSERVATION method (relevance redistributed, not differentiated). '+
      'Forward-only; weights untouched.');
  finally
    NN.Free;
  end;
end.
