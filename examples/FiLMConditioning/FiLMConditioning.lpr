program FiLMConditioning;
(*
FiLMConditioning: end-to-end demo of TNNetFiLM (Feature-wise Linear
Modulation, Perez et al. 2018, https://arxiv.org/abs/1709.07871).

FiLM is a PARAMETER-FREE two-input layer that applies a per-channel affine
modulation to a feature map, where the gamma (scale) and beta (shift) come
from a SEPARATE conditioning branch -- not the layer's own weights:

    Out[c] = gamma[c] * feature[c] + beta[c]

Wiring:  TNNetFiLM.Create([featureLayer, condLayer])
  featureLayer -> input0, shape (SizeX, SizeY, Depth)   (the feature map)
  condLayer    -> input1, shape (1, 1, 2*Depth)         (gamma | beta packed)

THE TASK (tiny synthetic conditional transform). The feature is a fixed
3-vector. A class id (0..K-1) selects WHICH affine transform the network
should apply to that feature:
    target_k[c] = TrueGamma[k,c] * feature[c] + TrueBeta[k,c]
The class id is fed (one-hot) through a TNNetFullConnectLinear conditioning
branch that must LEARN to emit the right gamma|beta vector for each class;
TNNetFiLM then applies it. So a single shared feature path produces K
different outputs purely via conditioning -- the whole point of FiLM.

We train only the conditioning FC (FiLM itself has no parameters) and watch
the per-class MSE fall to ~0, proving error back-propagates through FiLM into
the conditioning branch. Finally we verify the identity invariant: feeding
gamma=1, beta=0 reproduces the feature unchanged.

Runs in well under a second on a single CPU thread.

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
  cDepth   = 3;   // feature channels
  cClasses = 4;   // number of conditions / transforms to learn
  cEpochs  = 4000;

var
  // Ground-truth per-class affine transforms the network must discover.
  TrueGamma: array[0..cClasses-1, 0..cDepth-1] of TNeuralFloat;
  TrueBeta:  array[0..cClasses-1, 0..cDepth-1] of TNeuralFloat;
  Feature:   array[0..cDepth-1] of TNeuralFloat;

  NN: TNNet;
  FeatureIn, CondIn: TNNetLayer;
  FilmLayer: TNNetFiLM;
  FeatVol, CondVol, Desired: TNNetVolume;
  Epoch, k, c: integer;
  TotalLoss, diff, maxIdErr: TNeuralFloat;

  procedure SetCondOneHot(ClassId: integer);
  var i: integer;
  begin
    for i := 0 to cClasses - 1 do
      if i = ClassId then CondIn.Output[0, 0, i] := 1
      else CondIn.Output[0, 0, i] := 0;
  end;

  // Forward the whole two-input net for a given class. The feature volume is
  // constant; only the one-hot condition changes.
  procedure ForwardClass(ClassId: integer);
  begin
    FeatVol.Copy(FeatureIn.Output); // keep layer 0 output as the fixed feature
    SetCondOneHot(ClassId);
    NN.Compute(FeatVol);            // recomputes cond branch + FiLM
  end;

begin
  SetExceptionMask([exInvalidOp, exDenormalized, exZeroDivide,
                    exOverflow, exUnderflow, exPrecision]);
  RandSeed := 2026;

  // Fixed feature vector.
  Feature[0] := 1.0; Feature[1] := -2.0; Feature[2] := 0.5;

  // Distinct per-class transforms (a scale and a shift per channel).
  for k := 0 to cClasses - 1 do
    for c := 0 to cDepth - 1 do
    begin
      TrueGamma[k, c] := 0.5 + 0.5 * k + 0.25 * c;       // varied scales
      TrueBeta[k, c]  := (k - 1.5) - 0.3 * c;            // varied shifts
    end;

  WriteLn('TNNetFiLM conditioning demo');
  WriteLn('  feature depth = ', cDepth, ', classes = ', cClasses);
  WriteLn('  feature vector = [', FloatToStrF(Feature[0],ffFixed,4,2), ', ',
    FloatToStrF(Feature[1],ffFixed,4,2), ', ', FloatToStrF(Feature[2],ffFixed,4,2), ']');
  WriteLn;

  NN := TNNet.Create();
  FeatVol := TNNetVolume.Create(1, 1, cDepth);
  CondVol := TNNetVolume.Create(1, 1, cClasses);
  Desired := TNNetVolume.Create(1, 1, cDepth);
  try
    // Single-threaded by construction: this demo drives Compute/Backpropagate
    // one sample at a time, so no thread pool is involved.
    // Layer 0: the feature map (input0). Error collection on (=4th arg 1).
    FeatureIn := NN.AddLayer(TNNetInput.Create(1, 1, cDepth, 1));
    // Layer 1: the condition one-hot (input1 of the conditioning sub-network).
    CondIn := NN.AddLayerAfter(TNNetInput.Create(1, 1, cClasses, 1), 0);
    // One-call FiLM conditioning: AddFiLMConditioned wires the conditioning
    // FC (one-hot -> 2*Depth) -> reshape (1,1,2*Depth) -> TNNetFiLM internally,
    // replacing the manual Depth -> 2*Depth bookkeeping.
    FilmLayer := NN.AddFiLMConditioned(FeatureIn, CondIn) as TNNetFiLM;

    NN.SetLearningRate(0.05, 0.0);
    NN.SetBatchUpdate(false);

    // Load the fixed feature into the input layer once.
    for c := 0 to cDepth - 1 do FeatureIn.Output[0, 0, c] := Feature[c];

    WriteLn('Training the conditioning FC (FiLM has no parameters of its own):');
    for Epoch := 1 to cEpochs do
    begin
      TotalLoss := 0;
      for k := 0 to cClasses - 1 do
      begin
        ForwardClass(k);
        for c := 0 to cDepth - 1 do
          Desired[0, 0, c] := TrueGamma[k, c] * Feature[c] + TrueBeta[k, c];
        // Accumulate MSE for reporting.
        for c := 0 to cDepth - 1 do
        begin
          diff := NN.GetLastLayer.Output[0, 0, c] - Desired[0, 0, c];
          TotalLoss := TotalLoss + diff * diff;
        end;
        NN.Backpropagate(Desired);
      end;
      if (Epoch = 1) or (Epoch mod 1000 = 0) then
        WriteLn(Format('  epoch %5d   mean MSE = %.6f',
          [Epoch, TotalLoss / (cClasses * cDepth)]));
    end;
    WriteLn;

    WriteLn('Per-class result after training (target vs FiLM output):');
    WriteLn(StringOfChar('-', 60));
    for k := 0 to cClasses - 1 do
    begin
      ForwardClass(k);
      Write(Format('  class %d  target [', [k]));
      for c := 0 to cDepth - 1 do
        Write(Format('%7.3f', [TrueGamma[k, c] * Feature[c] + TrueBeta[k, c]]));
      Write(' ]   got [');
      for c := 0 to cDepth - 1 do
        Write(Format('%7.3f', [NN.GetLastLayer.Output[0, 0, c]]));
      WriteLn(' ]');
    end;
    WriteLn(StringOfChar('-', 60));
    WriteLn;

    // Identity invariant: gamma = 1, beta = 0 must reproduce the feature.
    // Bypass the FC by driving the FiLM conditioning input directly.
    for c := 0 to cDepth - 1 do
    begin
      FilmLayer.PrevLayer.Output[0, 0, c]          := 1; // gamma = 1
      FilmLayer.PrevLayer.Output[0, 0, cDepth + c] := 0; // beta  = 0
    end;
    FilmLayer.Compute();
    maxIdErr := 0;
    for c := 0 to cDepth - 1 do
      maxIdErr := Max(maxIdErr, Abs(FilmLayer.Output[0, 0, c] - Feature[c]));
    WriteLn('Identity invariant (gamma=1, beta=0 -> output == feature):');
    WriteLn('  max |out - feature| = ', FloatToStrF(maxIdErr, ffExponent, 4, 2),
      ' -> ', BoolToStr(maxIdErr < 1e-6, 'HELD', 'FAILED'));
  finally
    NN.Free;
    FeatVol.Free;
    CondVol.Free;
    Desired.Free;
  end;
end.
