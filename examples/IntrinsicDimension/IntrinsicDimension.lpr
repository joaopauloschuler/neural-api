program IntrinsicDimension;
(*
IntrinsicDimension: demonstrates TNNet.IntrinsicDimensionReport — a forward-only
representation-geometry diagnostic that estimates, for every trainable layer, how
many effective dimensions the activation cloud of an (unlabelled) probe batch
actually occupies. It reports TWO complementary intrinsic-dimension (ID)
estimates side by side:
  (1) the LINEAR / PCA ID via the participation ratio of the activation
      covariance eigenspectrum (PR = (sum lambda)^2 / sum lambda^2), and
  (2) the TwoNN nonlinear estimator (Facco et al. 2017) read off the slope of
      -log(1 - F(mu)) against log(mu), with mu = r2/r1 the per-sample 2nd-to-1st
      nearest-neighbour distance ratio.

This demo runs three blocks, all synthetic, in well under a minute:
  (A) GROUND-TRUTH recovery: a probe batch drawn from a known k-dimensional
      linear subspace embedded in a higher-D space — both PCA_ID and TwoNN_ID
      should land near k (the faithfulness check), shown by feeding the batch
      straight through a wide identity-init layer.
  (B) FRESH-INIT network: the ID profile is flat / near-input at every layer.
  (C) TRAINED network: the famous "hunchback" of Ansuini et al. 2019 — the ID
      first EXPANDS in the early layers then CONTRACTS monotonically toward the
      output as the representation compresses onto the task manifold.

Pure CPU, forward-only (NN.Compute only — weights are never touched).

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
  cInDim    = 24;     // ambient input dimension
  cHidden   = 48;
  cEpochs   = 80;
  cBatch    = 32;
  cProbeN   = 160;    // probe-batch size (kept small to bound NN^2 / eigensolve)

  // A small MLP with a clear bottleneck so the trained ID contracts toward
  // the output (the "hunchback" descending tail).
  procedure BuildMLP(out NN: TNNet);
  begin
    NN := TNNet.Create();
    NN.AddLayer(TNNetInput.Create(cInDim, 1, 1));
    NN.AddLayer(TNNetFullConnectReLU.Create(cHidden));
    NN.AddLayer(TNNetFullConnectReLU.Create(cHidden));
    NN.AddLayer(TNNetFullConnectReLU.Create(12));
    NN.AddLayer(TNNetFullConnectLinear.Create(2));
    NN.SetLearningRate(0.01, 0.9);
  end;

  // Synthetic 2-class task: the label is the sign of a fixed linear projection
  // of x (a smooth, low-intrinsic-dimension decision rule).
  procedure MakePair(out X, Y: TNNetVolume);
  var
    I: integer;
    Acc: TNeuralFloat;
  begin
    X := TNNetVolume.Create(cInDim, 1, 1);
    Y := TNNetVolume.Create(2, 1, 1);
    Acc := 0;
    for I := 0 to cInDim - 1 do
    begin
      X.Raw[I] := (Random - 0.5) * 2.0;
      if (I mod 2) = 0 then Acc := Acc + X.Raw[I] else Acc := Acc - X.Raw[I];
    end;
    if Acc >= 0 then Y.Raw[0] := 1.0 else Y.Raw[1] := 1.0;
  end;

  procedure TrainOnce(NN: TNNet; Epochs: integer);
  var
    Ep, B: integer;
    X, Yt: TNNetVolume;
  begin
    for Ep := 1 to Epochs do
      for B := 1 to cBatch do
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
  end;

  // Build an unlabelled probe batch for the network (random inputs).
  function MakeProbeBatch: TNNetVolumeList;
  var
    I, J: integer;
    V: TNNetVolume;
  begin
    Result := TNNetVolumeList.Create(True);   // owns its volumes
    for I := 0 to cProbeN - 1 do
    begin
      V := TNNetVolume.Create(cInDim, 1, 1);
      for J := 0 to cInDim - 1 do V.Raw[J] := (Random - 0.5) * 2.0;
      Result.Add(V);
    end;
  end;

  // --- Ground-truth block: a probe batch lying on a known k-dim subspace,
  //     embedded in cInDim ambient dims via a fixed random linear map, then
  //     pushed through a freshly-built net so the FIRST layer's activation
  //     cloud is (a linear image of) a k-dimensional object. ---
  function MakeSubspaceBatch(K: integer): TNNetVolumeList;
  var
    I, J, C: integer;
    V: TNNetVolume;
    Coeff: array of TNeuralFloat;
    Basis: array of array of TNeuralFloat;
  begin
    // fixed random basis: cInDim ambient x K latent
    SetLength(Basis, cInDim);
    for I := 0 to cInDim - 1 do
    begin
      SetLength(Basis[I], K);
      for J := 0 to K - 1 do Basis[I][J] := (Random - 0.5) * 2.0;
    end;
    SetLength(Coeff, K);
    Result := TNNetVolumeList.Create(True);
    for C := 0 to cProbeN - 1 do
    begin
      for J := 0 to K - 1 do Coeff[J] := (Random - 0.5) * 2.0;
      V := TNNetVolume.Create(cInDim, 1, 1);
      for I := 0 to cInDim - 1 do
      begin
        V.Raw[I] := 0;
        for J := 0 to K - 1 do V.Raw[I] := V.Raw[I] + Basis[I][J] * Coeff[J];
      end;
      Result.Add(V);
    end;
  end;

var
  NN: TNNet;
  Probes, SubBatch: TNNetVolumeList;
  GTNet: TNNet;
const
  cKnownK = 3;
begin
  RandSeed := 2026;

  WriteLn('IntrinsicDimensionReport demo: activation-cloud intrinsic dimension.');

  // ================= (A) GROUND-TRUTH k-subspace recovery =================
  // A wide linear identity-ish layer so the report's first trainable layer
  // sees a linear image of a known-k object. We use a 2-layer net whose first
  // layer just widens the input; PCA_ID / TwoNN_ID at that layer should ~ k.
  WriteLn;
  WriteLn(StringOfChar('=', 78));
  WriteLn('(A) GROUND-TRUTH: probe batch on a known k=', cKnownK,
          '-dim subspace in ', cInDim, ' ambient dims.');
  WriteLn('    Expect PCA_ID ~ ', cKnownK, ' and TwoNN_ID ~ ', cKnownK,
          ' at the input-facing layer.');
  WriteLn(StringOfChar('=', 78));
  GTNet := TNNet.Create();
  try
    GTNet.AddLayer(TNNetInput.Create(cInDim, 1, 1));
    GTNet.AddLayer(TNNetFullConnectLinear.Create(cHidden));
    GTNet.InitWeights();
    SubBatch := MakeSubspaceBatch(cKnownK);
    try
      Write(TNNet.IntrinsicDimensionReport(GTNet, SubBatch));
    finally
      SubBatch.Free;
    end;
  finally
    GTNet.Free;
  end;

  Probes := MakeProbeBatch();
  try
    // ===================== (B) FRESH-INIT network =======================
    BuildMLP(NN);
    try
      NN.InitWeights();
      WriteLn;
      WriteLn(StringOfChar('=', 78));
      WriteLn('(B) FRESH-INIT network (flat, near-input ID at every layer).');
      WriteLn(StringOfChar('=', 78));
      Write(TNNet.IntrinsicDimensionReport(NN, Probes));

      // ===================== (C) TRAINED network ========================
      WriteLn;
      WriteLn('Training for ', cEpochs, ' epochs of batch size ', cBatch, '...');
      TrainOnce(NN, cEpochs);
      WriteLn;
      WriteLn(StringOfChar('=', 78));
      WriteLn('(C) TRAINED network (the expand-then-contract "hunchback").');
      WriteLn(StringOfChar('=', 78));
      Write(TNNet.IntrinsicDimensionReport(NN, Probes));
    finally
      NN.Free;
    end;
  finally
    Probes.Free;
  end;

  WriteLn;
  WriteLn(
    'Read (A) as the faithfulness check (both IDs recover the known k); ' +
    'compare (B) vs (C): training pushes the early layers to EXPAND and the ' +
    'late / bottleneck layers to CONTRACT (the TwoNN_ID bar chart), as the ' +
    'representation compresses onto the low-dimensional task manifold.');
end.
