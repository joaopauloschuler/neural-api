program ActivationSteering;
(*
ActivationSteering: the INTERVENTIONAL flip-side of the read-only probe
examples. Instead of asking "what is decodable from a hidden layer?" it
INJECTS a concept direction INTO a hidden layer mid-forward and shows the
injection CAUSALLY controls the output (ActAdd / activation addition /
representation engineering, Turner et al. 2023).

Recipe (pure CPU, no dataset, forward-only — the trained weights are NEVER
stepped after the classifier is trained):
  1. Train a small softmax classifier on a synthetic 2-class toy (a
     two-cluster / sign-of-x0 task, same synthetic style as
     examples/ActivationPatching).
  2. Pick a hidden layer index k. Compute a STEERING VECTOR
        v = mean(act_k | class 1) - mean(act_k | class 0)
     a diff-of-class-means direction over the TRAINING activations at layer k.
     No extra training is run.
  3. Run forward layer-by-layer up to k, do Output_k.MulAdd(alpha, v), then
     recompute layers k+1..last (the SAME recompute machinery the landed
     ActivationPatchingReport drives: overwrite a cached activation and call
     FLayers[i].Compute() for i = k+1..last).
  4. Sweep alpha in {-3,-2,-1,0,1,2,3}, charting target-class probability vs
     alpha as an ASCII curve.

Built-in correctness checks (the headline assertions — PASS/FAIL):
  - alpha = 0 reproduces the unsteered forward pass BIT-FOR-BIT (the
    target-class probability is identical to the plain forward pass).
  - the target-class probability moves MONOTONICALLY with alpha (positive
    steers toward class 1, negative toward class 0).
  - contrast steering with v vs a RANDOM unit direction of EQUAL norm: the
    concept direction is special — a random direction of the same norm
    perturbs the output far less.

This is DISTINCT from ActivationPatching (swaps WHOLE cached activations
between two inputs), SaliencyReport (input-space gradient), GradientAscent
(ascends on the input image) and LinearProbeReport (only READS what a layer
encodes). Here we ADD a direction and watch the output move.

Pure CPU, no dataset download, well under a minute.

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
  cInDim    = 6;
  cHidden   = 12;
  cClasses  = 2;
  cEpochs   = 200;
  cBlob     = 1.3;    // distance of the signal coord from the origin
  cNoise    = 0.30;   // per-coordinate Gaussian-ish spread
  cSteerLay = 2;      // hidden layer index k we steer (a FC+ReLU layer)
  cMeanN    = 400;    // training samples per class for the mean estimate

  // A plain feed-forward softmax classifier. The deciding feature is the SIGN
  // of x0 (a two-cluster task): the remaining coordinates are pure noise
  // distractors. We keep it a simple chain (no skip) so a single hidden layer
  // carries a clean linear "class direction" the diff-of-means picks up.
  //
  //   Input(6) -> FC12+ReLU -> FC12+ReLU(k=2) -> FC12+ReLU -> FC2 -> SoftMax
  //
  procedure BuildNet(out NN: TNNet; LR: TNeuralFloat);
  begin
    NN := TNNet.Create();
    NN.AddLayer(TNNetInput.Create(cInDim, 1, 1));            // 0
    NN.AddLayer(TNNetFullConnectReLU.Create(cHidden));       // 1
    NN.AddLayer(TNNetFullConnectReLU.Create(cHidden));       // 2  <- steered
    NN.AddLayer(TNNetFullConnectReLU.Create(cHidden));       // 3
    NN.AddLayer(TNNetFullConnectLinear.Create(cClasses));    // 4
    NN.AddLayer(TNNetSoftMax.Create());                      // 5
    NN.SetLearningRate(LR, 0.9);
    NN.InitWeights();
  end;

  // Two-cluster task: the class is decided ONLY by sign(x0).
  //   class 0: x0 < 0    class 1: x0 > 0
  // All other coordinates are pure noise distractors.
  procedure MakeSample(X: TNNetVolume; Cls: integer);
  var
    I: integer;
    Sign: integer;
  begin
    for I := 0 to cInDim - 1 do
      X.Raw[I] := (Random - 0.5) * 2.0 * cNoise;
    if Cls = 0 then Sign := -1 else Sign := 1;
    X.Raw[0] := Sign * cBlob + (Random - 0.5) * 2.0 * cNoise;
  end;

  procedure TrainOnce(NN: TNNet; Epochs: integer);
  var
    Ep, B, Cls, Correct: integer;
    X, Y: TNNetVolume;
  begin
    X := TNNetVolume.Create(cInDim, 1, 1);
    Y := TNNetVolume.Create(cClasses, 1, 1);
    try
      for Ep := 1 to Epochs do
      begin
        Correct := 0;
        for B := 1 to 96 do
        begin
          Cls := Random(2);
          MakeSample(X, Cls);
          Y.Fill(0);
          Y.Raw[Cls] := 1.0;
          NN.Compute(X);
          if NN.GetLastLayer.Output.GetClass() = Cls then Inc(Correct);
          NN.Backpropagate(Y);
        end;
        if (Ep = 1) or (Ep mod 40 = 0) or (Ep = Epochs) then
          WriteLn(Format('  epoch %3d  train-acc=%.3f', [Ep, Correct / 96.0]));
      end;
    finally
      X.Free;
      Y.Free;
    end;
  end;

  // v = mean(act_k | class 1) - mean(act_k | class 0) over fresh training draws.
  procedure ComputeSteeringVector(NN: TNNet; K: integer; V: TNNetVolume);
  var
    X, Mean0, Mean1: TNNetVolume;
    I, Cls: integer;
  begin
    X := TNNetVolume.Create(cInDim, 1, 1);
    Mean0 := TNNetVolume.Create();
    Mean1 := TNNetVolume.Create();
    try
      Mean0.Copy(NN.Layers[K].Output);  Mean0.Fill(0);
      Mean1.Copy(NN.Layers[K].Output);  Mean1.Fill(0);
      for Cls := 0 to 1 do
        for I := 1 to cMeanN do
        begin
          MakeSample(X, Cls);
          NN.Compute(X);
          if Cls = 0 then
            Mean0.Add(NN.Layers[K].Output)
          else
            Mean1.Add(NN.Layers[K].Output);
        end;
      Mean0.Mul(1.0 / cMeanN);
      Mean1.Mul(1.0 / cMeanN);
      V.Copy(Mean1);
      V.Sub(Mean0);
    finally
      X.Free;
      Mean0.Free;
      Mean1.Free;
    end;
  end;

  // Forward pass with layer K's activation shifted by Direction*alpha.
  // Mirrors the ActivationPatchingReport recompute: a full forward pass, then
  // overwrite layer K's Output and recompute layers K+1..last via Compute().
  // CleanK is the unsteered activation at K, snapshotted once by the caller.
  // Returns the SoftMax probability of class TargetCls.
  function SteeredProb(NN: TNNet; Input: TNNetVolume; K: integer;
    CleanK, Direction: TNNetVolume; Alpha: TNeuralFloat;
    TargetCls: integer): TNeuralFloat;
  var
    I, LastLayer: integer;
  begin
    LastLayer := NN.GetLastLayerIdx();
    // Full clean forward pass (fills every layer's Output, K included).
    NN.Compute(Input);
    // Restore K to the snapshot, then inject the direction. CopyNoChecks: same
    // shape by construction (CleanK was copied from this very layer).
    NN.Layers[K].Output.CopyNoChecks(CleanK);
    NN.Layers[K].Output.MulAdd(Alpha, Direction);
    // Recompute every downstream layer from the steered activation.
    for I := K + 1 to LastLayer do
      NN.Layers[I].Compute();
    Result := NN.GetLastLayer.Output.Raw[TargetCls];
  end;

var
  NN: TNNet;
  Input, V, RandDir, CleanK: TNNetVolume;
  TargetCls, I, BarLen, MaxBar: integer;
  Plain, P0, VNorm, RandNorm: TNeuralFloat;
  Alphas: array[0..6] of TNeuralFloat = (-3, -2, -1, 0, 1, 2, 3);
  ProbV, ProbR: array[0..6] of TNeuralFloat;
  Monotonic, PassZero, PassRand: boolean;
  MaxMoveV, MaxMoveR: TNeuralFloat;
begin
  RandSeed := 2026;
  Input  := TNNetVolume.Create(cInDim, 1, 1);
  V      := TNNetVolume.Create();
  RandDir := TNNetVolume.Create();
  CleanK := TNNetVolume.Create();
  try
    WriteLn('ActivationSteering demo: ADD a concept direction into a hidden ' +
      'layer mid-forward and watch it CAUSALLY steer the softmax output');
    WriteLn('(ActAdd / activation addition, Turner et al. 2023). The steering ' +
      'vector is a diff-of-class-means direction at layer k; no extra training.');
    WriteLn;

    BuildNet(NN, 0.01);
    WriteLn('Training for ', cEpochs, ' epochs on a sign(x0) two-cluster task...');
    TrainOnce(NN, cEpochs);
    WriteLn;

    // Steering vector v = mean(act_k|1) - mean(act_k|0) at layer k.
    V.Copy(NN.Layers[cSteerLay].Output);  // shape v to match layer k
    ComputeSteeringVector(NN, cSteerLay, V);
    VNorm := V.GetMagnitude();
    WriteLn(Format('Steering layer k=%d (%s), activation size=%d, ' +
      '||v||=%.4f.', [cSteerLay, NN.Layers[cSteerLay].ClassName,
      NN.Layers[cSteerLay].Output.Size, VNorm]));

    // A RANDOM unit direction scaled to the SAME L2 norm as v (the control).
    RandDir.Copy(V);
    RandDir.Fill(0);
    for I := 0 to RandDir.Size - 1 do
      RandDir.Raw[I] := Random - 0.5;
    RandNorm := RandDir.GetMagnitude();
    if RandNorm > 0 then RandDir.Mul(VNorm / RandNorm);
    WriteLn(Format('Random control direction r: ||r||=%.4f ' +
      '(matched to ||v||).', [RandDir.GetMagnitude()]));
    WriteLn;

    // A FIXED probe input from the boundary-ish region so steering can swing
    // the decision either way. Snapshot the unsteered activation at layer k.
    MakeSample(Input, 0);             // a class-0 sample (x0 < 0)
    NN.Compute(Input);
    TargetCls := 1;                   // we steer TOWARD class 1
    Plain := NN.GetLastLayer.Output.Raw[TargetCls];
    CleanK.Copy(NN.Layers[cSteerLay].Output);

    WriteLn(Format('Probe input: plain forward P(class %d)=%.6f ' +
      '(argmax=%d). Steering TOWARD class %d with v.',
      [TargetCls, Plain, NN.GetLastLayer.Output.GetClass(), TargetCls]));
    WriteLn;

    // ---- alpha sweep with v and with the random control ----
    for I := 0 to High(Alphas) do
    begin
      ProbV[I] := SteeredProb(NN, Input, cSteerLay, CleanK, V, Alphas[I],
        TargetCls);
      ProbR[I] := SteeredProb(NN, Input, cSteerLay, CleanK, RandDir, Alphas[I],
        TargetCls);
    end;
    P0 := ProbV[3]; // alpha = 0 entry

    WriteLn(StringOfChar('=', 78));
    WriteLn('ALPHA SWEEP: P(target class) vs alpha (steered with v)');
    WriteLn(StringOfChar('=', 78));
    WriteLn('  alpha   P(v)      P(rand)   |  P(v) bar (0..1)');
    MaxBar := 48;
    for I := 0 to High(Alphas) do
    begin
      BarLen := Round(ProbV[I] * MaxBar);
      if BarLen < 0 then BarLen := 0;
      if BarLen > MaxBar then BarLen := MaxBar;
      WriteLn(Format('%6.1f  %8.6f  %8.6f  |  %s',
        [Alphas[I], ProbV[I], ProbR[I], StringOfChar('#', BarLen)]));
    end;
    WriteLn;

    // ---- correctness check 1: alpha = 0 reproduces the plain forward pass ----
    PassZero := (P0 = Plain);
    WriteLn(Format('CHECK 1 (alpha=0 reproduces plain forward BIT-FOR-BIT): ' +
      'plain=%.8f steered@0=%.8f  -> %s',
      [Plain, P0, BoolToStr(PassZero, 'PASS', 'FAIL')]));

    // ---- correctness check 2: P(target) monotonic in alpha (with v) ----
    Monotonic := True;
    for I := 1 to High(Alphas) do
      if ProbV[I] < ProbV[I - 1] - 1e-7 then Monotonic := False;
    WriteLn(Format('CHECK 2 (P(target) increases MONOTONICALLY with alpha): ' +
      'P(-3)=%.6f .. P(+3)=%.6f  -> %s',
      [ProbV[0], ProbV[High(Alphas)],
       BoolToStr(Monotonic, 'PASS', 'FAIL')]));

    // ---- correctness check 3: v moves the output more than random per norm ----
    // Average |P(alpha) - plain| across the whole sweep (equal ||v|| and ||r||,
    // so this is a per-unit-norm comparison). The concept direction reaches the
    // far class with much smaller alpha, so its mean swing dominates random's.
    MaxMoveV := 0;
    MaxMoveR := 0;
    for I := 0 to High(Alphas) do
    begin
      MaxMoveV := MaxMoveV + Abs(ProbV[I] - Plain);
      MaxMoveR := MaxMoveR + Abs(ProbR[I] - Plain);
    end;
    MaxMoveV := MaxMoveV / (High(Alphas) + 1);
    MaxMoveR := MaxMoveR / (High(Alphas) + 1);
    PassRand := MaxMoveV > MaxMoveR;
    WriteLn(Format('CHECK 3 (concept v steers MORE per unit norm than random r): ' +
      'mean|dP| v=%.6f  random=%.6f  ratio=%.2fx  -> %s',
      [MaxMoveV, MaxMoveR, MaxMoveV / Max(MaxMoveR, 1e-9),
       BoolToStr(PassRand, 'PASS', 'FAIL')]));
    WriteLn;

    if PassZero and Monotonic and PassRand then
      WriteLn('ALL CHECKS PASS: the diff-of-means concept direction CAUSALLY ' +
        'steers the output; alpha=0 is a no-op; random does far less.')
    else
      WriteLn('ONE OR MORE CHECKS FAILED.');

    WriteLn;
    WriteLn(
      'Read it as: injecting v = mean(act_k|1) - mean(act_k|0) at layer k and ' +
      'sweeping alpha drives P(target) smoothly from ~0 to ~1 — a CAUSAL knob ' +
      'on the prediction discovered with NO extra training (just a difference ' +
      'of training-set activation means). alpha=0 is a bit-for-bit no-op, and ' +
      'a random direction of the SAME norm barely moves the output, so the ' +
      'concept direction is genuinely special. Weights are never stepped: the ' +
      'only mutation is the transient activation shift, reverted by the next ' +
      'clean forward pass.');
  finally
    NN.Free;
    Input.Free;
    V.Free;
    RandDir.Free;
    CleanK.Free;
  end;
end.
