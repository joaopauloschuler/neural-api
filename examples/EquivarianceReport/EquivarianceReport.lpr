program EquivarianceReport;
(*
EquivarianceReport: demonstrates TNNet.EquivarianceReport, a forward-only
input-symmetry diagnostic. It builds two image classifiers on a tiny synthetic
8x8x3 dataset and prints, per input-side symmetry transform (FlipX, FlipY,
ReverseChannels, Roll), the invariance error
  mean_x ||f(T(x)) - f(x)||_2 / ||f(x)||_2,
the top-1 argmax agreement rate, a 10-bin ASCII histogram of per-sample error,
and a one-line verdict.

NET A is a PLAIN conv classifier: it has no built-in symmetry, so it is
flip-SENSITIVE (large FlipX/FlipY invariance error).

NET B reduces the spatial plane with a global-average TNNetAvgChannel as its
first op (Input -> AvgChannel -> FC head). A per-channel spatial average is
unchanged by any spatial permutation, so this net is FlipX- AND FlipY-INVARIANT
by design: the report shows ~0 invariance error on the FlipX/FlipY rows for
NET B, in clear contrast to the flip-sensitive NET A. (ReverseChannels / Roll
permute channels, so NET B is NOT invariant to those - the contrast is visible.)

No dataset download, pure CPU, well under a minute.

Copyright (C) 2026 Joao Paulo Schwarz Schuler

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
any later version.

Coded by Claude (AI).
*)

{$mode objfpc}{$H+}

uses {$IFDEF UNIX} cthreads, {$ENDIF}
  Classes, SysUtils, Math,
  neuralnetwork,
  neuralvolume;

const
  cSizeX    = 8;
  cSizeY    = 8;
  cDepth    = 3;
  cClasses  = 3;
  cEpochs   = 60;
  cBatch    = 24;
  cProbeCnt = 40;

  // Synthetic labelled image: class index drives a bright blob in a corner
  // and a dominant channel, so the task is learnable in a handful of epochs.
  procedure MakeSample(out X, Y: TNNetVolume; ForcedClass: integer);
  var
    Cls, px, py, cx, cy, c, i: integer;
  begin
    X := TNNetVolume.Create(cSizeX, cSizeY, cDepth);
    Y := TNNetVolume.Create(cClasses, 1, 1);
    X.Fill(0);
    Y.Fill(0);
    if ForcedClass >= 0 then Cls := ForcedClass
    else Cls := Random(cClasses);
    Y.Raw[Cls] := 1.0;
    // Light background noise.
    for i := 0 to X.Size - 1 do X.Raw[i] := Random * 0.1;
    // Class-specific bright 3x3 blob in a class-specific corner / channel.
    case Cls of
      0: begin cx := 1; cy := 1; end;
      1: begin cx := cSizeX - 2; cy := 1; end;
    else  begin cx := 1; cy := cSizeY - 2; end;
    end;
    for py := -1 to 1 do
      for px := -1 to 1 do
      begin
        c := Cls mod cDepth;
        X[cx + px, cy + py, c] := 1.0 + Random * 0.2;
      end;
  end;

  procedure BuildPlainNet(out NN: TNNet);
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

  // Flip-invariant by construction: a global per-channel spatial average
  // (TNNetAvgChannel -> 1x1xDepth) is unchanged by any spatial permutation, so
  // f(FlipX(x)) == f(x) and f(FlipY(x)) == f(x) for any weights. The FC head
  // then classifies from the (flip-invariant) channel means.
  procedure BuildFlipInvariantNet(out NN: TNNet);
  begin
    NN := TNNet.Create();
    NN.AddLayer(TNNetInput.Create(cSizeX, cSizeY, cDepth));
    NN.AddLayer(TNNetAvgChannel.Create());
    NN.AddLayer(TNNetFullConnectReLU.Create(16));
    NN.AddLayer(TNNetFullConnectLinear.Create(cClasses));
    NN.AddLayer(TNNetSoftMax.Create());
    NN.SetLearningRate(0.01, 0.9);
  end;

  procedure BuildProbes(out Probes: TNNetVolumeList);
  var
    K: integer;
    X, Y: TNNetVolume;
  begin
    Probes := TNNetVolumeList.Create(True);
    for K := 0 to cProbeCnt - 1 do
    begin
      MakeSample(X, Y, -1);
      Y.Free;
      Probes.Add(X);
    end;
  end;

  procedure TrainOnce(NN: TNNet; Epochs: integer);
  var
    Ep, B: integer;
    X, Yt: TNNetVolume;
    Hit: integer;
  begin
    for Ep := 1 to Epochs do
    begin
      Hit := 0;
      for B := 1 to cBatch do
      begin
        MakeSample(X, Yt, -1);
        try
          NN.Compute(X);
          if NN.GetLastLayer.Output.GetClass() = Yt.GetClass() then Inc(Hit);
          NN.Backpropagate(Yt);
        finally
          X.Free;
          Yt.Free;
        end;
      end;
      if (Ep = 1) or (Ep mod 20 = 0) or (Ep = Epochs) then
        WriteLn(Format('  epoch %3d  train-acc=%.3f',
          [Ep, Hit / cBatch]));
    end;
  end;

var
  NNPlain, NNInv: TNNet;
  Probes: TNNetVolumeList;
begin
  RandSeed := 2026;

  BuildProbes(Probes);
  try
    WriteLn('EquivarianceReport demo: 8x8x3 synthetic 3-class image classifier.');

    // ---- NET A: plain conv classifier (flip-sensitive). ----
    BuildPlainNet(NNPlain);
    try
      WriteLn;
      WriteLn(StringOfChar('=', 92));
      WriteLn('NET A: plain conv classifier. Expect HIGH FlipX/FlipY '+
        'invariance error (sensitive).');
      WriteLn(StringOfChar('=', 92));
      WriteLn('Training for ', cEpochs, ' epochs of batch size ', cBatch, '...');
      TrainOnce(NNPlain, cEpochs);
      WriteLn;
      Write(TNNet.EquivarianceReport(NNPlain, Probes));
    finally
      NNPlain.Free;
    end;

    // ---- NET B: FlipX-invariant wired net. ----
    RandSeed := 2026;
    BuildFlipInvariantNet(NNInv);
    try
      WriteLn;
      WriteLn(StringOfChar('=', 92));
      WriteLn('NET B: Input -> AvgChannel -> FC head. Expect ~0 FlipX/FlipY '+
        'invariance error (invariant by construction).');
      WriteLn(StringOfChar('=', 92));
      WriteLn('Training for ', cEpochs, ' epochs of batch size ', cBatch, '...');
      TrainOnce(NNInv, cEpochs);
      WriteLn;
      Write(TNNet.EquivarianceReport(NNInv, Probes));
    finally
      NNInv.Free;
    end;

    WriteLn;
    WriteLn(
      'Expect: NET A is "sensitive" to FlipX/FlipY (a plain conv has no '+
      'built-in spatial symmetry); NET B reports ~0 FlipX/FlipY invariance '+
      'error ("invariant") because its global per-channel spatial average '+
      'makes f(FlipX(x))==f(x). NET B is still sensitive to the channel '+
      'permutations (ReverseChannels/Roll). Pure forward-only diagnostic.');
  finally
    Probes.Free;
  end;
end.
