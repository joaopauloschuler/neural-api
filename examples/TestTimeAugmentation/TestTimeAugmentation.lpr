program TestTimeAugmentation;
(*
TestTimeAugmentation: builds a tiny synthetic colored-pattern image classifier,
trains it briefly, then prints TNNet.TTAReport across a held-out probe batch.

The synthetic task has 3 classes of 8x8x3 images:
  class 0: a bright RED vertical stripe on the left columns,
  class 1: a bright GREEN horizontal stripe on the top rows,
  class 2: a bright BLUE checkerboard,
each plus light per-pixel noise. The model is deliberately small and the
patterns are NOT flip/channel symmetric, so the test-time-augmentation menu
(identity, FlipX, FlipY, ReverseChannels, Roll) genuinely perturbs the inputs.

The report is printed twice: once averaging raw logits (the default) and once
averaging post-softmax probabilities (soft voting), so the linear-vs-geometric
mean question is visible side by side.

Pure CPU, well under a minute.

Copyright (C) 2026 Joao Paulo Schwarz Schuler

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
any later version.
*)

{$mode objfpc}{$H+}

uses {$IFDEF UNIX} {$IFDEF UseCThreads}
  cthreads, {$ENDIF} {$ENDIF}
  Classes, SysUtils, Math,
  neuralnetwork,
  neuralvolume;

const
  cSide     = 8;
  cChannels = 3;
  cClasses  = 3;
  cEpochs   = 60;
  cBatch    = 24;
  cProbeCnt = 60;

  procedure BuildNet(out NN: TNNet);
  begin
    NN := TNNet.Create();
    NN.AddLayer(TNNetInput.Create(cSide, cSide, cChannels));
    NN.AddLayer(TNNetConvolutionReLU.Create(8, 3, 1, 1));
    NN.AddLayer(TNNetMaxPool.Create(2));
    NN.AddLayer(TNNetFullConnectReLU.Create(16));
    NN.AddLayer(TNNetFullConnectLinear.Create(cClasses));
    NN.AddLayer(TNNetSoftMax.Create());
    NN.SetLearningRate(0.01, 0.9);
  end;

  // Make one labeled synthetic image. Label is chosen by the caller.
  procedure MakeSample(Lbl: integer; out X, Y: TNNetVolume);
  var
    Px, Py, C: integer;
    Base: TNeuralFloat;
  begin
    X := TNNetVolume.Create(cSide, cSide, cChannels);
    Y := TNNetVolume.Create(1, 1, cClasses);
    X.Fill(0);
    Y.Fill(0);
    Y.Raw[Lbl] := 1.0;
    for Px := 0 to cSide - 1 do
      for Py := 0 to cSide - 1 do
        for C := 0 to cChannels - 1 do
        begin
          Base := 0.0;
          case Lbl of
            0: // red vertical stripe on the left columns -> channel 0
              if (Px < cSide div 2) and (C = 0) then Base := 1.0;
            1: // green horizontal stripe on the top rows -> channel 1
              if (Py < cSide div 2) and (C = 1) then Base := 1.0;
            2: // blue checkerboard -> channel 2
              if (((Px + Py) mod 2) = 0) and (C = 2) then Base := 1.0;
          end;
          X.Add(Px, Py, C, Base + (Random - 0.5) * 0.2);
        end;
  end;

  procedure BuildProbes(out Probes: TNNetVolumeList; out Labels: array of integer);
  var
    K: integer;
    X, Y: TNNetVolume;
  begin
    for K := 0 to cProbeCnt - 1 do
    begin
      MakeSample(K mod cClasses, X, Y);
      Y.Free;
      Probes.Add(X);
      Labels[K] := K mod cClasses;
    end;
  end;

  procedure TrainOnce(NN: TNNet; Epochs: integer);
  var
    Ep, B, I: integer;
    X, Yt, Out0: TNNetVolume;
    TotalLoss, Diff: TNeuralFloat;
  begin
    for Ep := 1 to Epochs do
    begin
      TotalLoss := 0;
      for B := 1 to cBatch do
      begin
        MakeSample(Random(cClasses), X, Yt);
        try
          NN.Compute(X);
          Out0 := NN.GetLastLayer.Output;
          for I := 0 to Out0.Size - 1 do
          begin
            Diff := Out0.Raw[I] - Yt.Raw[I];
            TotalLoss := TotalLoss + Diff * Diff;
          end;
          NN.Backpropagate(Yt);
        finally
          X.Free;
          Yt.Free;
        end;
      end;
      if (Ep = 1) or (Ep mod 20 = 0) or (Ep = Epochs) then
        WriteLn(Format('  epoch %3d  mean-MSE=%.6f', [Ep, TotalLoss / cBatch]));
    end;
  end;

var
  NN: TNNet;
  Probes: TNNetVolumeList;
  Labels: array of integer;
begin
  RandSeed := 2026;

  WriteLn('TestTimeAugmentation demo: tiny 8x8x3 colored-pattern classifier.');
  WriteLn('Classes: 0=red vertical stripe, 1=green horizontal stripe, ' +
    '2=blue checkerboard.');
  WriteLn;

  BuildNet(NN);
  Probes := TNNetVolumeList.Create(True);
  SetLength(Labels, cProbeCnt);
  try
    WriteLn('Training for ', cEpochs, ' epochs of batch size ', cBatch, '...');
    TrainOnce(NN, cEpochs);

    BuildProbes(Probes, Labels);

    WriteLn;
    WriteLn(StringOfChar('=', 92));
    WriteLn('TTA report (averaging RAW LOGITS - the default):');
    WriteLn(StringOfChar('=', 92));
    Write(TNNet.TTAReport(NN, Probes, Labels, False));

    WriteLn;
    WriteLn(StringOfChar('=', 92));
    WriteLn('TTA report (averaging POST-SOFTMAX PROBABILITIES - soft voting):');
    WriteLn(StringOfChar('=', 92));
    Write(TNNet.TTAReport(NN, Probes, Labels, True));

    WriteLn;
    WriteLn(
      'These patterns are not flip/channel symmetric, so each transform row ' +
      'shows how much that single augmentation alone degrades accuracy, while ' +
      'the ensemble row shows the net effect of averaging all five together.');
  finally
    Probes.Free;
    NN.Free;
  end;
end.
