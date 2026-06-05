program AdversarialRobustness;
(*
AdversarialRobustness: demonstrates TNNet.AdversarialRobustnessReport, a
forward+backward FGSM (fast gradient sign method, Goodfellow et al. 2015)
robustness diagnostic. It trains a tiny SoftMax classifier on a synthetic
3-class 2-D problem (three Gaussian blobs), then crafts adversarial
perturbations x_adv = x + eps * sign(d loss / d x) at an increasing menu of
epsilons and reports how top-1 accuracy degrades.

The report prints:
  (a) a top-1 accuracy-vs-eps degradation curve (eps=0 is the clean baseline),
  (b) a 10-bin histogram of the per-sample CRITICAL EPSILON (smallest eps that
      flips the prediction away from the clean argmax),
  (c) the mean clean-confidence of the earliest flippers vs the longest
      survivors,
  (d) per-class accuracy at the median eps, and
  (e) a one-line verdict (robust / moderately fragile / fragile).

For contrast a SECOND model is trained WITH input-noise augmentation (Gaussian
jitter added to every training input). Input-noise training is a cheap form of
robustness regularisation, so its degradation curve should fall off more slowly
- the eyeballable expected effect.

The network is FROZEN during the report: weights are never updated (this is an
EVALUATION of robustness, not adversarial training).

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
  Classes, SysUtils,
  neuralnetwork,
  neuralvolume;

const
  cClasses = 3;
  cEpochs  = 80;
  cBatch   = 24;
  cProbeN  = 12;   // probe samples per class

// Three 2-D Gaussian blob centres, one per class. Kept fairly close together
// (and with overlapping spread) so the decision boundaries sit near the inputs
// and FGSM produces a visible accuracy-vs-eps degradation curve.
const
  cCenters: array[0..2, 0..1] of TNeuralFloat =
    ((-0.9, -0.9), (0.9, 0.9), (0.9, -0.9));

procedure MakeSample(out X, Y: TNNetVolume; ForcedClass: integer);
var
  Cls: integer;
begin
  X := TNNetVolume.Create(2, 1, 1);
  Y := TNNetVolume.Create(cClasses, 1, 1);
  if ForcedClass >= 0 then Cls := ForcedClass
  else Cls := Random(cClasses);
  X.FData[0] := cCenters[Cls][0] + (Random - 0.5) * 1.4;
  X.FData[1] := cCenters[Cls][1] + (Random - 0.5) * 1.4;
  Y.Fill(0);
  Y.FData[Cls] := 1.0;
end;

procedure BuildNet(out NN: TNNet);
begin
  NN := TNNet.Create();
  NN.AddLayer(TNNetInput.Create(2, 1, 1));
  NN.AddLayer(TNNetFullConnectReLU.Create(16));
  NN.AddLayer(TNNetFullConnectReLU.Create(16));
  NN.AddLayer(TNNetFullConnectLinear.Create(cClasses));
  NN.AddLayer(TNNetSoftMax.Create());
  NN.SetLearningRate(0.05, 0.9);
  NN.InitWeights();
end;

// Trains NN. When NoiseAug > 0, Gaussian jitter of that scale is added to each
// training input (a cheap robustness regulariser).
procedure TrainOnce(NN: TNNet; Epochs: integer; NoiseAug: TNeuralFloat);
var
  Ep, B, Hit, I: integer;
  X, Yt: TNNetVolume;
begin
  for Ep := 1 to Epochs do
  begin
    Hit := 0;
    for B := 1 to cBatch do
    begin
      MakeSample(X, Yt, -1);
      try
        if NoiseAug > 0 then
          for I := 0 to X.Size - 1 do
            X.FData[I] := X.FData[I] + (Random - 0.5) * 2.0 * NoiseAug;
        NN.Compute(X);
        if NN.GetLastLayer.Output.GetClass() = Yt.GetClass() then Inc(Hit);
        NN.Backpropagate(Yt);
      finally
        X.Free;
        Yt.Free;
      end;
    end;
    if (Ep = 1) or (Ep mod 20 = 0) or (Ep = Epochs) then
      WriteLn(Format('  epoch %3d  train-acc=%.3f', [Ep, Hit / cBatch]));
  end;
end;

// Builds a labelled probe batch (Samples + Labels) of cProbeN samples per class.
procedure BuildProbe(Samples: TNNetVolumeList; var Labels: array of integer);
var
  C, I, Idx: integer;
  X, Yt: TNNetVolume;
begin
  Idx := 0;
  for C := 0 to cClasses - 1 do
    for I := 1 to cProbeN do
    begin
      MakeSample(X, Yt, C);
      Samples.Add(X);
      Yt.Free;
      Labels[Idx] := C;
      Inc(Idx);
    end;
end;

var
  NN, NNNoisy: TNNet;
  Samples: TNNetVolumeList;
  Labels: array of integer;
const
  cEps: array[0..5] of TNeuralFloat = (0.0, 0.05, 0.1, 0.2, 0.4, 0.8);
begin
  RandSeed := 2026;

  WriteLn('AdversarialRobustness demo: synthetic 3-class 2-D classifier.');
  WriteLn('  three Gaussian blobs; FGSM perturbations x_adv = x + eps*sign(grad).');
  WriteLn;

  Samples := TNNetVolumeList.Create();
  SetLength(Labels, cClasses * cProbeN);
  BuildNet(NN);
  BuildNet(NNNoisy);
  try
    WriteLn('Training baseline model (', cEpochs, ' epochs)...');
    TrainOnce(NN, cEpochs, 0.0);
    WriteLn;
    WriteLn('Training noise-augmented model (', cEpochs,
      ' epochs, input jitter)...');
    TrainOnce(NNNoisy, cEpochs, 0.6);
    WriteLn;

    // Probe batch is shared by both reports (same RandSeed-derived samples).
    BuildProbe(Samples, Labels);

    WriteLn(StringOfChar('=', 72));
    WriteLn('BASELINE model:');
    WriteLn(StringOfChar('=', 72));
    Write(TNNet.AdversarialRobustnessReport(NN, Samples, Labels, cEps));
    WriteLn;

    WriteLn(StringOfChar('=', 72));
    WriteLn('NOISE-AUGMENTED model (expect a flatter/slower degradation curve):');
    WriteLn(StringOfChar('=', 72));
    Write(TNNet.AdversarialRobustnessReport(NNNoisy, Samples, Labels, cEps));
    WriteLn;

    WriteLn(
      'Expect: accuracy falls as eps grows (FGSM works); the noise-augmented '+
      'model degrades more slowly (input-noise training is a cheap robustness '+
      'regulariser). The eps=0 row equals a plain evaluation pass. '+
      'Forward+backward only; the trained weights are never updated.');
  finally
    Samples.Free;
    NN.Free;
    NNNoisy.Free;
  end;
end.
