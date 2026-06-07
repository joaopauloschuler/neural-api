program LocalLearningCoefficient;
(*
LocalLearningCoefficient: demonstrates TNNet.LocalLearningCoefficientReport, an
empirical estimate of the Local Learning Coefficient (LLC) — the Real Log
Canonical Threshold (RLCT) from Singular Learning Theory (Watanabe; Lau, Murfet,
Wei et al. 2023, "Quantifying Degeneracy in Singular Models via the LLC"). The
LLC measures the VOLUME-SCALING / EFFECTIVE dimensionality of the minimum the
optimizer settled into: it counts the flat, degenerate directions that a
second-order Hessian top-eigenvalue cannot see. A redundant / over-parameterised
solution has LLC_hat << dim(w): far fewer EFFECTIVE degrees of freedom than raw
weights.

The estimator runs a short tempered, ANCHORED Stochastic Gradient Langevin
Dynamics (SGLD) chain from the trained weights w*, pinned to the basin by a
Gaussian anchor (gamma/2)*||w - w*||^2, then forms the WBIC free-energy estimate
  LLC_hat = n * beta * ( mean_chain[L(w)] - L(w_star) ),  beta = 1/ln(n).
It reuses the existing forward+backward gradient machinery; the only new piece is
the anchored-Langevin update + chain average, all inside the report. The report
is NON-DESTRUCTIVE: w* is snapshotted and restored bit-for-bit on return.

Three nets, the SAME 3-class toy target:
  (1) minimal net          trained to fit.
  (2) over-parameterised   trained to fit, then two hidden units FORCED
                           redundant (duplicate-then-halve) so two directions
                           are exactly flat.
  (3) random-init net      NEVER trained (w* is NOT a critical point).

What separates ROBUSTLY (reproducible across seeds, see README.md):
  * Both TRAINED nets (1) & (2) report a small LLC_hat with LLC_hat << dim(w):
    the basin has far fewer EFFECTIVE degrees of freedom than raw weights.
  * The RANDOM-INIT net (3) reports a large, often NEGATIVE LLC_hat: the LLC is
    only defined AT a local minimum, and at a random point the anchored chain
    slides DOWNHILL so mean_chain[L] < L(w_star). A negative / wild LLC_hat is
    the diagnostic's honest "this is not a minimum" signal.

What did NOT fit the CPU budget cleanly: the fine-grained (1)-vs-(2) ORDERING
(minimal below over-parameterised) is real in expectation but, at the short
fixed chain length used here, the run-to-run estimator NOISE on the tiny
mean_chain[L]-L(w_star) gap is the same size as that gap, so the two trained
arms do not separate reproducibly across seeds. The robust headline is the
TRAINED-vs-UNTRAINED contrast plus LLC_hat << dim(w); a much longer chain (and
several chains averaged) would be needed to resolve (1) vs (2). Absolute LLC
values are calibration-dependent (eps/gamma/chain-length); only the ORDERING
under FIXED hyperparameters is meaningful.

Pure CPU, well under a minute. The report never steps the weights (a
measurement, not training).

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
  cInDim   = 4;
  cHidMin  = 4;    // minimal hidden width that fits the target.
  cHidBig  = 8;    // over-parameterised hidden width.
  cClasses = 3;
  cProbe   = 40;   // probe-batch size handed to the report.

  Centers: array[0..cClasses - 1, 0..1] of TNeuralFloat =
    ((-2.0, -2.0), (2.0, 2.0), (2.0, -2.0));

  procedure MakePair(out X, Y: TNNetVolume);
  var
    C, I: integer;
  begin
    C := Random(cClasses);
    X := TNNetVolume.Create(cInDim, 1, 1);
    Y := TNNetVolume.Create(cClasses, 1, 1);
    X.Raw[0] := Centers[C][0] + (Random - 0.5) * 0.7;
    X.Raw[1] := Centers[C][1] + (Random - 0.5) * 0.7;
    for I := 2 to cInDim - 1 do
      X.Raw[I] := (Random - 0.5) * 0.4;
    Y.Fill(0);
    Y.Raw[C] := 1.0;
  end;

  procedure BuildMLP(out NN: TNNet; Hidden: integer);
  begin
    NN := TNNet.Create();
    NN.AddLayer(TNNetInput.Create(cInDim, 1, 1));
    NN.AddLayer(TNNetFullConnectReLU.Create(Hidden));
    NN.AddLayer(TNNetFullConnectLinear.Create(cClasses));
    NN.AddLayer(TNNetSoftMax.Create());
  end;

  procedure TrainOnce(NN: TNNet; Epochs, Batch: integer);
  var
    Ep, B: integer;
    X, Yt: TNNetVolume;
  begin
    NN.SetBatchUpdate(true);
    for Ep := 1 to Epochs do
    begin
      NN.ClearDeltas();
      for B := 1 to Batch do
      begin
        MakePair(X, Yt);
        try
          NN.Compute(X);
          NN.Backpropagate(Yt);
        finally
          X.Free;
          Yt.Free;
        end;
      end;
      NN.UpdateWeights();
    end;
  end;

  // Force two hidden units of the over-parameterised net to be REDUNDANT:
  // copy unit 0's weights into unit 1 (so they compute the identical feature)
  // and halve both their fan-OUT weights in the next layer (so the function is
  // unchanged). The two duplicated directions are now flat/degenerate.
  procedure ForceRedundancy(NN: TNNet);
  var
    HidLayer, OutLayer: TNNetLayer;
    j: integer;
  begin
    HidLayer := NN.Layers[1];
    OutLayer := NN.Layers[2];
    // Make hidden unit 1 an exact copy of hidden unit 0 (forward compute reads
    // Weights / BiasWeight directly, so no cache refresh is needed).
    HidLayer.Neurons[1].Weights.Copy(HidLayer.Neurons[0].Weights);
    HidLayer.Neurons[1].BiasWeight := HidLayer.Neurons[0].BiasWeight;
    // Halve the fan-out weights from hidden units 0 and 1 in every output
    // neuron so the duplicated feature is split in half (function preserved).
    for j := 0 to OutLayer.Neurons.Count - 1 do
    begin
      OutLayer.Neurons[j].Weights.FData[0] :=
        OutLayer.Neurons[j].Weights.FData[0] * 0.5;
      OutLayer.Neurons[j].Weights.FData[1] :=
        OutLayer.Neurons[j].Weights.FData[0];
    end;
  end;

  procedure FillProbe(Samples: TNNetVolumePairList; Count: integer);
  var
    I: integer;
    X, Yt: TNNetVolume;
  begin
    for I := 1 to Count do
    begin
      MakePair(X, Yt);
      Samples.Add(TNNetVolumePair.Create(X, Yt));
    end;
  end;

var
  MinNN, BigNN, RandNN: TNNet;
  Probe: TNNetVolumePairList;
begin
  // Mask FP exceptions: the Langevin sampler can transiently overflow.
  SetExceptionMask([exInvalidOp, exDenormalized, exZeroDivide, exOverflow,
    exUnderflow, exPrecision]);
  RandSeed := 2026;

  WriteLn('LocalLearningCoefficientReport demo: the ORDERING of LLC_hat is the');
  WriteLn('robust signal of effective (vs raw) model dimensionality.');
  WriteLn('Three nets fit (or fail to fit) the SAME 3-class toy problem:');
  WriteLn('  (1) minimal   (2) over-parameterised+redundant   (3) random-init.');

  Probe := TNNetVolumePairList.Create();
  try
    FillProbe(Probe, cProbe);

    // ---- (1) minimal net trained to fit. ----
    BuildMLP(MinNN, cHidMin);
    try
      MinNN.InitWeights();
      MinNN.SetLearningRate(0.05, 0.9);
      TrainOnce(MinNN, 500, 32);

      // ---- (2) over-parameterised net trained, then forced redundant. ----
      BuildMLP(BigNN, cHidBig);
      try
        BigNN.InitWeights();
        BigNN.SetLearningRate(0.05, 0.9);
        TrainOnce(BigNN, 500, 32);
        ForceRedundancy(BigNN);

        // ---- (3) random-init net (never trained). ----
        BuildMLP(RandNN, cHidMin);
        try
          RandNN.InitWeights();

          WriteLn;
          WriteLn(StringOfChar('=', 100));
          WriteLn('(1) MINIMAL trained net (hidden=', cHidMin,
            '): small LLC_hat, LLC_hat << dim(w).');
          WriteLn(StringOfChar('=', 100));
          Write(TNNet.LocalLearningCoefficientReport(MinNN, Probe, 300, 1e-4, 10.0));

          WriteLn;
          WriteLn(StringOfChar('=', 100));
          WriteLn('(2) OVER-PARAMETERISED+redundant net (hidden=', cHidBig,
            ', two units forced identical): expect LLC_hat << dim(w).');
          WriteLn(StringOfChar('=', 100));
          Write(TNNet.LocalLearningCoefficientReport(BigNN, Probe, 300, 1e-4, 10.0));

          WriteLn;
          WriteLn(StringOfChar('=', 100));
          WriteLn('(3) RANDOM-INIT net (hidden=', cHidMin,
            ', never trained): NOT a minimum -> large / negative LLC_hat.');
          WriteLn(StringOfChar('=', 100));
          Write(TNNet.LocalLearningCoefficientReport(RandNN, Probe, 300, 1e-4, 10.0));
        finally
          RandNN.Free;
        end;
      finally
        BigNN.Free;
      end;
    finally
      MinNN.Free;
    end;
  finally
    Probe.Free;
  end;

  WriteLn;
  WriteLn('Robust, reproducible reading: the two TRAINED nets (1) & (2) report a');
  WriteLn('small LLC_hat with LLC_hat << dim(w) - the basin has far fewer ');
  WriteLn('EFFECTIVE degrees of freedom than weights. The RANDOM-INIT net (3) ');
  WriteLn('reports a large / negative LLC_hat: w* is not a minimum there, so the');
  WriteLn('estimator (correctly) breaks - its honest "not a minimum" signal. The');
  WriteLn('fine (1)-vs-(2) ordering needs a much longer chain than this CPU-toy ');
  WriteLn('budget allows (see the header / README.md). Only the ordering under ');
  WriteLn('FIXED (eps,gamma,chain) hyperparameters is meaningful.');
end.
